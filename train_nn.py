from pytorch_lightning import Trainer
from pytorch_lightning.logging import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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

logger = NeptuneLogger(api_key=("eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91c"
                                "mwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYTc4YTJjN2"
                                "YtMzY3NC00OWVhLTk4MTItZjBhYzY2NjEyZjg5In0="),
                       project_name="guillaumevw/nsnet",
                       params=hparams,
                       upload_source_files=['train_nn.py', 'model/nsnet_model.py', 'dataloader/wav_dataset.py'])
checkpoint_path = os.path.join('lightning_logs', str(logger.version))
os.makedirs(checkpoint_path)
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_top_k=1, mode='min')
trainer = Trainer(gpus=1, min_epochs=200, logger=logger, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=1)
trainer.fit(model)
