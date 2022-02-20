from argparse import Namespace
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl

from src.datasets_labeled import labeled_datasets
from src.models import Model
from src.utils import get_dvc_dir, data_dir

model_dir = get_dvc_dir("models")
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


def save_model_ckpt(trainer: pl.Trainer, model_ckpt_path: Path):
    if model_ckpt_path.exists():
        model_ckpt_path.unlink()
    model_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_ckpt_path)


def train_model(
    hparams, model_ckpt_path: Optional[Path] = None, offline: bool = False
) -> Tuple[pl.LightningModule, Dict[str, float]]:

    model = Model(hparams)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )

    wandb_logger = WandbLogger(project="crop-mask", entity="nasa-harvest", offline=offline)
    wandb_logger.experiment.config.update(
        {
            "available_timesteps": model.available_timesteps,
            "forecast_eval_data": model.forecast_eval_data,
            "forecast_training_data": model.forecast_training_data,
            "forecast_timesteps": model.forecast_timesteps,
            "train_num_timesteps": tuple(model.train_num_timesteps),
            "eval_num_timesteps": tuple(model.eval_num_timesteps),
        }
    )

    trainer = pl.Trainer(
        default_save_path=str(data_dir),
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=False,
        logger=wandb_logger,
    )

    trainer.fit(model)

    # Save as hparams so they can be output into metrics.json
    hparams.local_val_size = wandb_logger.experiment.config["local_validation_original_size"]
    hparams.local_val_crop_percentage = wandb_logger.experiment.config[
        "local_validation_crop_percentage"
    ]

    if model_ckpt_path is None:
        model_ckpt_path = model_dir / f"{hparams.model_name}.ckpt"
    save_model_ckpt(trainer, model_ckpt_path)

    metrics = get_metrics_from_trainer(trainer)
    metrics["local_val_size"] = hparams.local_val_size
    metrics["local_val_crop_percentage "] = hparams.local_val_crop_percentage

    return model, metrics


def get_metrics_from_trainer(trainer: pl.LightningModule) -> Dict[str, float]:
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        if "_global_" in k or "loss" in k or "epoch" in k:
            continue
        metrics[k] = round(float(v), 4)
    return metrics


def run_evaluation_on_one_model(model: Model, test: bool = False) -> Dict[str, float]:
    trainer = pl.Trainer(checkpoint_callback=False, logger=False)
    if test:
        trainer.test(model)
    else:
        trainer.model = model
        trainer.main_progress_bar = tqdm(disable=True)
        trainer.run_evaluation(test_mode=False)
    metrics = get_metrics_from_trainer(trainer)
    return metrics


def run_evaluation(
    model_ckpt_path: Path, alternative_threshold: Optional[float] = None
) -> Tuple[Model, Dict[str, float]]:
    if not model_ckpt_path.exists():
        raise ValueError(f"Model {str(model_ckpt_path)} does not exist")
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


def parameter_has_changed(model_ckpt_path: Path, hparams: Namespace) -> bool:
    """Checks if ckpt model parameters are different from hparams being passed"""
    model = Model.load_from_checkpoint(model_ckpt_path)
    model_hparams = model.hparams.__dict__
    params_that_can_change = [
        "alternative_threshold",
        "fail_on_error",
        "retrain_all",
        "offline",
    ]
    for k, v in hparams.__dict__.items():
        if k in model_hparams and model_hparams[k] != v and k not in params_that_can_change:
            print(f"\u2714 {hparams.model_name} exists, but new parameters for {k} were found.")
            return True
    return False


def model_pipeline(
    hparams: Namespace, retrain_all: bool = False, offline: bool = False, eval_only: bool = False
) -> Tuple[str, Dict[str, float]]:

    hparams = validate(hparams)

    model_name = hparams.model_name
    model_ckpt_path = model_dir / f"{model_name}.ckpt"

    # Determine if training is necessary
    if eval_only is False and (
        retrain_all
        or not model_ckpt_path.exists()
        or parameter_has_changed(model_ckpt_path, hparams)
    ):
        print(f"\u2714 {model_name} beginning training")
        model, metrics = train_model(hparams, model_ckpt_path, offline)
        print(f"\n\u2714 {model_name} completed training and evaluation")
        print(metrics)
        for key in ["unencoded_val_local_f1_score", "encoded_val_local_f1_score"]:
            if key in metrics and metrics[key] > 0.54:
                model.save()
                break
    else:
        print(f"\n\u2714 {model_name} exists, running evaluation only")
        threshold = hparams.alternative_threshold if "alternative_threshold" in hparams else None
        model, metrics = run_evaluation(model_ckpt_path, threshold)
        print(f" \u2714 {model_name} completed evaluation")

    return model_name, metrics
