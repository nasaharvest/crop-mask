from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


def train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=hparams.patience, verbose=True, mode="min",
    )
    trainer = pl.Trainer(
        default_save_path=hparams.data_folder,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)

    return model
