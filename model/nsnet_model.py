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


class NSNetModel(pl.LightningModule):
    """
    Sample model to show how to define a template
    Input size: (batch_size, frequency_bins, time)
    """

    def __init__(self, hparams=Namespace(**{'train_dir': None, 'val_dir': None, 'batch_size': 4, 'n_fft': 512,
                                            'n_gru_layers': 3, 'gru_dropout': 0, 'alpha': 0.35})):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(NSNetModel, self).__init__()

        self.hparams = hparams
        self.train_dir = Path(self.hparams.train_dir)
        self.val_dir = Path(self.hparams.val_dir)
        self.batch_size = self.hparams.batch_size
        self.n_fft = self.hparams.n_fft
        self.n_frequency_bins = self.n_fft // 2 + 1
        self.n_gru_layers = self.hparams.n_gru_layers
        self.gru_dropout = self.hparams.gru_dropout
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
        self.gru = nn.GRU(input_size=self.n_frequency_bins, hidden_size=self.n_frequency_bins, num_layers=self.n_gru_layers,
                          batch_first=True, dropout=self.gru_dropout)
        self.dense = nn.Linear(in_features=self.n_frequency_bins, out_features=self.n_frequency_bins)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = x.permute(0, 2, 1)  # (batch_size, time, n_frequency_bins)
        x, _ = self.gru(x)  # (batch_size, time, n_frequency_bins)
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
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True, min_lr=1e-6)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        # init data generators

        if train:
            dataset = WAVDataset(self.train_dir, n_fft=self.n_fft)
        else:
            dataset = WAVDataset(self.val_dir, n_fft=self.n_fft)

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
