from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


def train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:

    if not hparams.model_name:
        raise ValueError("model_name must be set.")

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

    model_path = Path(f"{hparams.data_folder}/models/{hparams.model_name}.ckpt")
    if model_path.exists():
        model_path.unlink()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_path)

    return model
