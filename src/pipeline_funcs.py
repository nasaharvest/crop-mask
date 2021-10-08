from argparse import Namespace
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from src.datasets_labeled import labeled_datasets
from src.models import Model


all_dataset_names = [d.dataset for d in labeled_datasets]


def validate(hparams: Namespace) -> Namespace:
    # Check model name
    if not hparams.model_name:
        raise ValueError("model_name is not set")

    # Check datasets
    for datasets_to_check_str in [hparams.eval_datasets, hparams.train_datasets]:
        datasets_to_check = datasets_to_check_str.split(",")
        missing_datasets = [name for name in datasets_to_check if name not in all_dataset_names]
        if len(missing_datasets) > 0:
            raise ValueError(f"{hparams.model_name} missing datasets: {missing_datasets}")

    # Check bounding box
    if not (hparams.min_lat and hparams.max_lat and hparams.min_lon and hparams.max_lon):
        raise ValueError(f"{hparams.model_name} missing lat lon bbox")

    # All checks passed, no issues
    return hparams


def add_dataset_stats(model: Model, hparams: Namespace) -> Namespace:
    norm_dict = model.normalizing_dict
    local_train_dataset = model.get_dataset(
        "training", is_local_only=True, normalizing_dict=norm_dict
    )
    local_val_dataset = model.get_dataset(
        "validation", is_local_only=True, upsample=False, normalizing_dict=norm_dict
    )

    # local train
    hparams.local_train_original_size = local_train_dataset.original_size
    hparams.local_train_upsampled_size = len(local_train_dataset)
    hparams.local_train_crop_percentage = local_train_dataset.crop_percentage

    # local val
    hparams.local_val_size = len(local_val_dataset)
    hparams.local_val_crop_percentage = local_val_dataset.crop_percentage

    return hparams


def save_model_ckpt(trainer: pl.Trainer, model_ckpt_path: Path):
    if model_ckpt_path.exists():
        model_ckpt_path.unlink()
    model_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_ckpt_path)


def save_model_pt(model: Model, model_pt_path: Path):
    model_pt_path.parent.mkdir(parents=True, exist_ok=True)
    sm = torch.jit.script(model)
    sm.save(str(model_pt_path))


def train_model(
    hparams, model_ckpt_path: Optional[Path] = None
) -> Tuple[pl.LightningModule, Dict[str, float]]:

    model = Model(hparams)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )

    logger = TensorBoardLogger("tb_logs", name=hparams.eval_datasets)
    trainer = pl.Trainer(
        default_save_path=hparams.data_folder,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=False,
        logger=logger,
    )

    trainer.fit(model)

    hparams = add_dataset_stats(model, hparams)
    logger.experiment.add_hparams(vars(hparams), trainer.callback_metrics)

    if model_ckpt_path is None:
        model_ckpt_path = Path(f"{hparams.model_dir}/{hparams.model_name}.ckpt")
    save_model_ckpt(trainer, model_ckpt_path)

    metrics = get_metrics_from_trainer(trainer)
    metrics["local_val_size"] = hparams.local_val_size
    metrics["local_val_crop_percentage "] = hparams.local_val_crop_percentage

    return model, metrics


def get_metrics_from_trainer(trainer: pl.LightningModule) -> Dict[str, float]:
    return {
        k: round(float(v), 4)
        for k, v in trainer.callback_metrics.items()
        if ("_global_" not in k and "loss" not in k)
    }


def run_evaluation_on_one_model(model: Model, test: bool = False) -> Dict[str, float]:
    trainer = pl.Trainer()
    if test:
        trainer.test(model)
    else:
        trainer.model = model
        trainer.main_progress_bar = tqdm(disable=True)
        trainer.run_evaluation(test_mode=False)
    metrics = get_metrics_from_trainer(trainer)
    return metrics


def run_evaluation(
    model_ckpt_path: str, alternative_threshold: Optional[float] = None
) -> Tuple[Model, Dict[str, float]]:
    model = Model.load_from_checkpoint(model_ckpt_path)
    metrics = run_evaluation_on_one_model(model)
    if alternative_threshold:
        alternative_model = Model.load_from_checkpoint(model_ckpt_path)
        alternative_model.hparams.probability_threshold = alternative_threshold
        alternative_metrics = run_evaluation_on_one_model(alternative_model)
        for k, v in alternative_metrics.items():
            metrics[f"thresh{alternative_threshold}_{k}"] = v

    metrics["local_val_size"] = model.hparams.local_val_size
    metrics["local_val_crop_percentage "] = model.hparams.local_val_crop_percentage
    return model, metrics


def model_pipeline(hparams: Namespace) -> Tuple[str, Dict[str, float]]:

    hparams = validate(hparams)

    model_name = hparams.model_name
    model_ckpt_path = Path(f"{hparams.model_dir}/{model_name}.ckpt")
    model_pt_path = model_ckpt_path.with_suffix(".pt")

    if model_ckpt_path.exists():
        print(f"\n\u2714 {model_name} exists, running evaluation only")
        threshold = hparams.alternative_threshold if "alternative_threshold" in hparams else None
        model, metrics = run_evaluation(str(model_ckpt_path), threshold)
        print(f" \u2714 {model_name} completed evaluation")

    else:
        print(f"\u2714 {model_name} beginning training")
        model, metrics = train_model(hparams, model_ckpt_path)
        print(f"\n\u2714 {model_name} completed training and evaluation")
        print(metrics)

    if not model_pt_path.exists():
        for key in ["unencoded_val_local_f1_score", "encoded_val_local_f1_score"]:
            if key in metrics:
                if metrics[key] > 0.6:
                    save_model_pt(model, model_pt_path)
                break

    return model_name, metrics
