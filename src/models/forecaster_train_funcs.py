from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

def forecaster_train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:
    logger = TensorBoardLogger(
        save_dir=hparams.save_dir,
        name='lightning_logs_sandbox'
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        default_save_path=hparams.save_dir,
        max_epochs=hparams.max_epochs,
        # early_stop_callback=early_stop_callback,
        checkpoint_callback=False,
        show_progress_bar=hparams.show_progress_bar,
        logger=logger
    )
    
    trainer.fit(model)

    model_path = Path(f"{hparams.save_dir}/{hparams.model_name}.ckpt")

    if model_path.exists():
        model_path.unlink()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_path)

    return model
