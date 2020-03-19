"""
Example template for defining a system
"""
import logging as log
from collections import OrderedDict
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataloader.wav_dataset import WAVDataset


import pytorch_lightning as pl


class EHNetModel(pl.LightningModule):
    """
    Sample model to show how to define a template
    Input size: (batch_size, frequency_bins, time)
    """

    def __init__(self, hparams=Namespace(**{'train_dir': None, 'val_dir': None, 'batch_size': 4, 'n_frequency_bins': 256, 'n_kernels': 256,
                                            'kernel_size_f': 32, 'kernel_size_t': 11, 'n_lstm_layers': 2, 'n_lstm_units': 1024, 'lstm_dropout': 0,
                                            'alpha': 0.35})):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(EHNetModel, self).__init__()

        self.hparams = hparams
        self.train_dir = Path(self.hparams.train_dir)
        self.val_dir = Path(self.hparams.val_dir)
        self.batch_size = self.hparams.batch_size
        self.n_frequency_bins = self.hparams.n_frequency_bins
        self.n_kernels = self.hparams.n_kernels
        self.kernel_size = (self.hparams.kernel_size_f, self.hparams.kernel_size_t)
        self.stride = (self.kernel_size[0] // 2, 1)
        self.padding = (self.kernel_size[1] // 2, self.kernel_size[1] // 2)
        self.n_lstm_layers = self.hparams.n_lstm_layers
        self.n_lstm_units = self.hparams.n_lstm_units
        self.lstm_dropout = self.hparams.lstm_dropout
        self.alpha = self.hparams.alpha

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.n_kernels,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding)
        self.batchnorm = nn.BatchNorm2d(num_features=self.n_kernels)
        n_features = int(self.n_kernels * (((self.n_frequency_bins - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]) + 1))
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=self.n_lstm_units, num_layers=self.n_lstm_layers,
                            batch_first=True, dropout=self.lstm_dropout, bidirectional=True)
        self.dense = nn.Linear(in_features=2 * self.n_lstm_units, out_features=self.n_frequency_bins)
        self.flatten = nn.Flatten(start_dim=2)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = torch.unsqueeze(x, 1)  # (batch_size, 1, frequency_bins, time)
        x = F.relu(self.conv(x))  # (batch_size, n_kernels, n_features, time)
        x = self.batchnorm(x)  # (batch_size, n_kernels, n_features, time)
        x = x.permute(0, 3, 1, 2)  # (batch_size, time, n_kernels, n_features)
        x = self.flatten(x)  # (batch_size, time, n_kernels * n_features)
        x, _ = self.lstm(x)  # (batch_size, time, 2 * n_lstm_units)
        x = torch.sigmoid(self.dense(x))  # (batch_size, time, frequency_bins)
        x = x.permute(0, 2, 1)  # (batch_size, frequency_bins, time)

        return x

    def loss(self, target, prediction):
        loss = F.mse_loss(prediction, target)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x_lps, x_ms, y_ms, VAD = batch
        noise_ms = x_ms - y_ms
        VAD_expanded = torch.unsqueeze(VAD, dim=1).expand_as(y_ms)

        y_hat = self.forward(x_lps)

        loss_speech = self.loss(y_ms[VAD_expanded], (y_hat * y_ms)[VAD_expanded])
        loss_noise = self.loss(torch.zeros_like(y_hat), y_hat * noise_ms)

        loss_val = self.alpha * loss_speech + (1 - self.alpha) * loss_noise

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # forward pass
        x_lps, x_ms, y_ms, VAD = batch
        noise_ms = x_ms - y_ms
        VAD_expanded = torch.unsqueeze(VAD, dim=1).expand_as(y_ms)

        y_hat = self.forward(x_lps)

        loss_speech = self.loss(y_ms[VAD_expanded], (y_hat * y_ms)[VAD_expanded])
        loss_noise = self.loss(torch.zeros_like(y_hat), y_hat * noise_ms)

        loss_val = self.alpha * loss_speech + (1 - self.alpha) * loss_noise

        output = OrderedDict({
            'val_loss': loss_val,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adadelta(self.parameters(), lr=1.0, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        # init data generators

        n_fft = (self.n_frequency_bins - 1) * 2
        if train:
            dataset = WAVDataset(self.train_dir, n_fft=n_fft)
        else:
            dataset = WAVDataset(self.val_dir, n_fft=n_fft)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

        return loader

    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(train=False)
