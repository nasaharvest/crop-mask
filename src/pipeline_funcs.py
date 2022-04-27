from argparse import Namespace
from pathlib import Path

from xarray import combine_nested
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple

import json
import pytorch_lightning as pl

from src.datasets_labeled import labeled_datasets
from src.models import Model
from src.utils import get_dvc_dir, models_file

import geopandas as gpd
import numpy as np
import sklearn.metrics
import rasterio as ras
from pyproj import Proj, transform

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


def train_model(
    hparams, offline: bool = False
) -> Tuple[pl.LightningModule, Dict[str, Dict[str, Any]]]:

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )

    wandb_logger = WandbLogger(project="crop-mask", entity="nasa-harvest", offline=offline)
    hparams.wandb_url = wandb_logger.experiment.get_url()
    model = Model(hparams)

    wandb_logger.experiment.config.update(
        {
            "available_timesteps": model.available_timesteps,
            "forecast_eval_data": model.forecast_eval_data,
            "forecast_training_data": model.forecast_training_data,
            "forecast_timesteps": model.forecast_timesteps,
            "train_num_timesteps": model.train_num_timesteps,
            "eval_num_timesteps": model.eval_num_timesteps,
        }
    )

    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        checkpoint_callback=False,
        early_stop_callback=early_stop_callback,
        logger=wandb_logger,
    )

    trainer.fit(model)

    model, metrics = run_evaluation(
        model_ckpt_path=get_dvc_dir("models") / f"{hparams.model_name}.ckpt"
    )

    model.save()

    return model, metrics


def get_metrics_from_trainer(trainer: pl.LightningModule) -> Dict[str, float]:
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        if any([text in k for text in ["loss", "epoch", "f1_score_max"]]):
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
) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    if not model_ckpt_path.exists():
        raise ValueError(f"Model {str(model_ckpt_path)} does not exist")
    model = Model.load_from_checkpoint(model_ckpt_path)
    val_metrics = run_evaluation_on_one_model(model, test=False)
    test_metrics = run_evaluation_on_one_model(model, test=True)
    if alternative_threshold:
        alternative_model = Model.load_from_checkpoint(model_ckpt_path)
        alternative_model.hparams.probability_threshold = alternative_threshold
        alternative_metrics = run_evaluation_on_one_model(alternative_model, test=False)
        for k, v in alternative_metrics.items():
            val_metrics[f"thresh{alternative_threshold}_{k}"] = v

    with models_file.open() as f:
        models_dict = json.load(f)

    all_info = {
        "params": model.hparams.wandb_url,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    models_dict[model.hparams.model_name] = all_info

    with models_file.open("w") as f:
        json.dump(models_dict, f, ensure_ascii=False, indent=4, sort_keys=True)

    return model, all_info

def run_comparison(validation: str, cropmap: str):
    #assume datatypes to be csv and tiff for validation and cropmap respectively
    report = []
    pred = ras.open(cropmap, mode='r')
    truth = gpd.read_file(validation)
    truth = truth[['lat', 'lon', 'crop_probability']]

    pred_sampled = ras.sample.sample_gen(pred, ((float(x),float(y)) for x,y in zip(truth['lat'], truth['lon'])))
    #reproject pred?

    target_names = ['non_crop', 'crop']
    class_report = sklearn.metrics.classification_report(truth['crop_probability'], pred_sampled['cropmask'], target_names = target_names, output_dict=True)
    accuracy = sklearn.metrics.accuracy_score(truth['crop_probability'], pred_sampled)

    report = [accuracy, class_report['crop']['precision'], class_report['non_crop']['precision'], class_report['non_crop']['recall'], class_report['crop']['recall']]

    return report