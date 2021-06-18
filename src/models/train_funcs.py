from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


def train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        default_save_path=hparams.data_folder,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=False
    )
    trainer.fit(model)

    if hparams.model_name:
        model_path = Path(f"{hparams.data_folder}/models/{hparams.model_name}.ckpt")
        if model_path.exists():
            model_path.unlink()
        trainer.save_checkpoint(model_path)

    return model
