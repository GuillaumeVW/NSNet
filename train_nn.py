from pytorch_lightning import Trainer
from model.nsnet_model import NSNetModel
from argparse import Namespace
import os

train_dir = './WAVs/dataset/training'
val_dir = './WAVs/dataset/validation'

hparams = {'train_dir': train_dir,
           'val_dir': val_dir,
           'batch_size': 64,
           'n_fft': 512,
           'n_gru_layers': 3,
           'gru_dropout': 0.2,
           'alpha': 0.35}

model = NSNetModel(hparams=Namespace(**hparams))

trainer = Trainer(gpus=1)
trainer.fit(model)
