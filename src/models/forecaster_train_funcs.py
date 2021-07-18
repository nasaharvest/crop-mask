from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def forecaster_train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:
    logger = TensorBoardLogger(
        save_dir="/cmlscratch/hkjoo/repo/crop-mask/data/models/sandbox",
        version=1,
        name='lightning_logs'
    )

    trainer = pl.Trainer(
        default_save_path=hparams.processed_data_folder,
        max_epochs=hparams.max_epochs,
        show_progress_bar=hparams.show_progress_bar,
        logger=logger,
        checkpoint_callback=False
    )
    
    trainer.fit(model)

    if hparams.model_name:
        model_path = Path(f"{hparams.save_dir}/{hparams.model_name}.ckpt")
        if model_path.exists():
            model_path.unlink()
        trainer.save_checkpoint(model_path)

    return model
