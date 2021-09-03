from argparse import Namespace
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:

    if not hparams.target_bbox_key:
        raise ValueError("target_bbox_key must be set.")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )

    logger = TensorBoardLogger("tb_logs", name=hparams.target_bbox_key)
    trainer = pl.Trainer(
        default_save_path=hparams.data_folder,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=False,
        logger=logger,
    )

    trainer.fit(model)

    norm_dict = model.normalizing_dict
    local_train_dataset = model.get_dataset(
        "training", is_local_only=True, normalizing_dict=norm_dict
    )
    global_train_dataset = model.get_dataset(
        "training", is_global_only=True, normalizing_dict=norm_dict
    )
    local_val_dataset = model.get_dataset(
        "validation", is_local_only=True, upsample=False, normalizing_dict=norm_dict
    )

    # local train
    hparams.local_train_original_size = local_train_dataset.original_size
    hparams.local_train_upsampled_size = len(local_train_dataset)
    hparams.local_train_crop_percentage = local_train_dataset.crop_percentage

    # global train
    hparams.global_train_original_size = global_train_dataset.original_size
    hparams.global_train_upsampled_size = len(global_train_dataset)
    hparams.global_train_crop_percentage = global_train_dataset.crop_percentage

    # local val
    hparams.local_val_size = len(local_val_dataset)
    hparams.local_val_crop_percentage = local_val_dataset.crop_percentage

    logger.experiment.add_hparams(vars(hparams), trainer.callback_metrics)

    model_path_prefix = f"{hparams.model_dir}/{hparams.target_bbox_key}"
    model_ckpt_path = Path(f"{model_path_prefix}.ckpt")
    model_pt_path = Path(f"{model_path_prefix}.pt")

    for p in [model_ckpt_path, model_pt_path]:
        if p.exists():
            p.unlink()

    model_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_ckpt_path)

    # sm = torch.jit.script(model)
    # sm.save(str(model_pt_path))

    return model
