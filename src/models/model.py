import json
import random
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from openmapflow.bands import ERA5_BANDS
from openmapflow.bbox import BBox
from openmapflow.config import DATA_DIR, PROJECT_ROOT, DataPaths
from openmapflow.constants import CLASS_PROB, EO_DATA, SUBSET
from openmapflow.engineer import BANDS, calculate_ndvi
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import datasets
from src.bboxes import bboxes

from .classifier import Classifier
from .data import CropDataset
from .forecaster import Forecaster


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class Model(pl.LightningModule):
    r"""
    An model for annual and in-season crop mapping. This model consists of a
    forecaster.Forecaster and a classifier.Classifier - it will require the arguments
    required by those models too.

    hparams
    --------
    The default values for these parameters are set in add_model_specific_args

    :param hparams.learning_rate: The learning rate. Default = 0.001
    :param hparams.batch_size: The batch size. Default = 64
    :param hparams.probability_threshold: The probability threshold to use to label GeoWiki
        instances as crop / not_crop (since the GeoWiki labels are a mean crop probability, as
        assigned by several labellers). In addition, this is the threshold used when calculating
        metrics which require binary predictions, such as accuracy score. Default = 0.5
    :param hparams.input_months: The number of input months to pass to the model. If
        hparams.forecast is True, the remaining months will be forecasted. Otherwise, only the
        partial timeseries will be passed to the classifier. Default = 5
    :param hparams.alpha: The weight to use when adding the global and local losses. This
        parameter is only used if hparams.multi_headed is True. Default = 10
    :param hparams.noise_factor: The standard deviation of the random noise to add to the
        raw inputs to the classifier. Default = 0.1
    :param hparams.forecast: Whether or not to forecast the partial time series. Default = True
    :param hparams.cache: Whether to load all the data into memory during training. Default = True
    :param hparams.upsample: Whether to oversample the under-represented class so that each class
        is equally represented in the training and validation dataset. Default = True
    :param hparams.target_bbox_key: The key to the bbox in bounding_box.py which determines which
        data is local and which is global
    :param hparams.train_datasets: A list of the datasets to use for training.
    :param hparams.eval_datasets: A list of the datasets to use for evaluation.
    """

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        if "seed" in hparams:
            set_seed(hparams.seed)
        else:
            set_seed()

        self.hparams = hparams

        self.batch_size = hparams.batch_size

        if "bbox" in hparams:
            self.target_bbox = bboxes[hparams.bbox]
        else:
            self.target_bbox = BBox(
                min_lat=hparams.min_lat,
                max_lat=hparams.max_lat,
                min_lon=hparams.min_lon,
                max_lon=hparams.max_lon,
            )

        if "skip_era5" in hparams and hparams.skip_era5:
            self.bands_to_use = [i for i, v in enumerate(BANDS) if v not in ERA5_BANDS]
        else:
            self.bands_to_use = [i for i, _ in enumerate(BANDS)]

        # --------------------------------------------------
        # Normalizing dicts
        # --------------------------------------------------
        all_dataset_params: Dict[str, Any] = {}
        all_dataset_params_path = PROJECT_ROOT / DATA_DIR / "all_dataset_params.json"
        if all_dataset_params_path.exists():
            with all_dataset_params_path.open() as f:
                all_dataset_params = json.load(f)

        self.input_months = self.hparams.input_months
        self.up_to_year = hparams.up_to_year if "up_to_year" in hparams else None
        self.start_month = hparams.start_month if "start_month" in hparams else "April"

        normalizing_dict_key = hparams.train_datasets
        if self.start_month:
            normalizing_dict_key += f"_{self.start_month}"
        if self.up_to_year:
            normalizing_dict_key += f"_{self.up_to_year}"

        if normalizing_dict_key not in all_dataset_params:
            train_dataset = self.get_dataset(subset="training", cache=False, upsample=False)
            val_dataset = self.get_dataset(
                subset="validation",
                cache=False,
                upsample=False,
                normalizing_dict=train_dataset.normalizing_dict,
            )
            # we save the normalizing dict because we calculate weighted
            # normalization values based on the datasets we combine.
            # The number of instances per dataset (and therefore the weights) can
            # vary between the train / test / val sets - this ensures the normalizing
            # dict stays constant between them
            if train_dataset.normalizing_dict is None:
                raise ValueError("Normalizing dict must be calculated using dataset.")

            all_dataset_params[normalizing_dict_key] = {
                "train_num_timesteps": train_dataset.num_timesteps,
                "val_num_timesteps": val_dataset.num_timesteps,
                "normalizing_dict": {
                    k: v.tolist() for k, v in train_dataset.normalizing_dict.items()
                },
            }

            with all_dataset_params_path.open("w") as f:
                json.dump(all_dataset_params, f, ensure_ascii=False, indent=4, sort_keys=True)

        dataset_params = all_dataset_params[normalizing_dict_key]
        self.train_num_timesteps: List[int] = dataset_params["train_num_timesteps"]
        self.eval_num_timesteps: List[int] = dataset_params["val_num_timesteps"]

        # Normalizing dict that is exposed
        self.normalizing_dict_jit: Dict[str, List[float]] = dataset_params["normalizing_dict"]
        self.normalizing_dict: Optional[Dict[str, np.ndarray]] = {
            k: np.array(v) for k, v in dataset_params["normalizing_dict"].items()
        }

        # ----------------------------------------------------------------------
        # Forecaster parameters
        # ----------------------------------------------------------------------
        # Needed so that forecast is exposed to jit
        self.forecaster = torch.nn.Identity()
        self.forecast_eval_data = self.input_months > min(self.eval_num_timesteps)
        self.forecast_training_data = self.input_months > min(self.train_num_timesteps)
        self.available_timesteps = min(self.eval_num_timesteps + self.train_num_timesteps)
        if self.input_months > self.available_timesteps:
            self.forecast_timesteps = self.input_months - self.available_timesteps
            self.forecaster = Forecaster(
                num_bands=len(self.bands_to_use),
                output_timesteps=self.forecast_timesteps,
                hparams=hparams,
            )
        else:
            self.forecast_timesteps = 0

        self.forecaster_loss = F.smooth_l1_loss

        self.classifier = Classifier(input_size=len(self.bands_to_use), hparams=hparams)
        self.global_loss_function: Callable = F.binary_cross_entropy
        self.local_loss_function: Callable = F.binary_cross_entropy

        # Used during training to track lowest val loss
        self.val_losses: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, self.bands_to_use]
        if self.forecast_eval_data:
            x_input = x[:, : self.available_timesteps, :]
            x_forecasted = self.forecaster(x_input)[:, self.available_timesteps - 1 :, :]
            x = torch.cat((x_input, x_forecasted), dim=1)
        _, local_preds = self.classifier(x)
        return local_preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def load_df(subset: str, train_datasets: str, eval_datasets: str) -> pd.DataFrame:
        """
        Loads the datasets specified in the input_dataset_names list.
        """
        dfs = []
        for d in datasets:
            # If dataset is used for evaluation, take only the right subset out of the dataframe
            if d.name in eval_datasets.split(","):
                df = d.load_df(to_np=True, disable_tqdm=True)
                dfs.append(df[(df[SUBSET] == subset) & (df[CLASS_PROB] != 0.5)])

            # If dataset is only used for training, take the whole dataframe
            elif subset == "training" and d.name in train_datasets.split(","):
                df = d.load_df(to_np=True, disable_tqdm=True)
                dfs.append(df[df[CLASS_PROB] != 0.5])

        big_df = pd.concat(dfs)

        # Recompute NDVI
        big_df[EO_DATA] = big_df[EO_DATA].apply(lambda x: calculate_ndvi(x[:, : len(BANDS) - 1]))

        return big_df

    def get_dataset(
        self,
        subset: str,
        normalizing_dict: Optional[Dict] = None,
        cache: Optional[bool] = None,
        upsample: Optional[bool] = None,
    ) -> CropDataset:
        df = self.load_df(subset, self.hparams.train_datasets, self.hparams.eval_datasets)

        return CropDataset(
            subset=subset,
            df=df,
            normalizing_dict=normalizing_dict,
            cache=self.hparams.cache if cache is None else cache,
            upsample=upsample if upsample is not None else self.hparams.upsample,
            target_bbox=self.target_bbox,
            up_to_year=self.up_to_year,
            wandb_logger=self.logger,
            start_month=self.start_month,
            input_months=self.input_months,
        )

    def train_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="training",
                normalizing_dict=self.normalizing_dict,
                upsample=self.hparams.upsample,
            ),
            shuffle=True,
            batch_size=self.hparams.batch_size,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="validation",
                normalizing_dict=self.normalizing_dict,
                upsample=False,
            ),
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="testing",
                normalizing_dict=self.normalizing_dict,
                upsample=False,
            ),
            batch_size=self.hparams.batch_size,
        )

    def _output_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if len(preds) == 0:
            # sometimes this happens in the warmup
            return {}

        output_dict: Dict[str, float] = {}
        if not (labels == labels[0]).all():
            # This can happen when lightning does its warm up on a subset of the
            # validation data
            output_dict["roc_auc_score"] = roc_auc_score(labels, preds)

        preds = (preds > self.hparams.probability_threshold).astype(int)

        output_dict["precision_score"] = precision_score(labels, preds, zero_division=0)
        output_dict["recall_score"] = recall_score(labels, preds, zero_division=0)
        output_dict["f1_score"] = f1_score(labels, preds, zero_division=0)
        output_dict["accuracy"] = accuracy_score(labels, preds)
        return output_dict

    def add_noise(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if (self.hparams.noise_factor == 0) or (not training):
            return x

        # expect input to be of shape [timesteps, bands]
        # and to be normalized with mean 0, std=1
        # if its not, it means no norm_dict was passed, so lets
        # just assume std=1
        noise = torch.normal(0, 1, size=x.shape).float() * self.hparams.noise_factor

        # the added noise is the same per band, so that the temporal relationships
        # are preserved
        # noise_per_timesteps = noise.repeat(x.shape[0], 1)
        return x + noise

    def _compute_forecaster_loss(
        self, y_true: torch.Tensor, y_forecast: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss for the forecaster
        If y_true contains nans, then all values other then nans are used for the loss
        """

        y_nans = torch.isnan(y_true)

        # If there is no nans in the batch compute forecaster loss as usual
        if y_nans.any().item() is False:
            return self.forecaster_loss(y_true, y_forecast)

        nan_batch_index = y_nans.any(dim=1).any(dim=1)
        nan_month_index = y_nans.any(dim=0).any(dim=1)

        partial_shape = (-1, sum(~nan_month_index), y_true.shape[2])
        y_true_partial = y_true[nan_batch_index][:, ~nan_month_index].reshape(partial_shape)
        y_forecast_partial = y_forecast[nan_batch_index][:, ~nan_month_index].reshape(partial_shape)
        assert bool(torch.all(torch.isnan(y_forecast_partial))) is False
        loss_partial = self.forecaster_loss(y_forecast_partial, y_true_partial)

        # If the batch contains time series with only nans, then return only the partial loss
        if nan_batch_index.all():
            assert y_forecast_partial.shape[0] == y_forecast.shape[0]
            return loss_partial

        # Otherwise the batch contains time series with at least one non nan value
        # Compute combined loss
        full_shape = (-1, y_true.shape[1], y_true.shape[2])
        y_true_full = y_true[~nan_batch_index].reshape(full_shape)
        y_forecast_full = y_forecast[~nan_batch_index].reshape(full_shape)
        assert y_forecast_full.shape[0] + y_forecast_partial.shape[0] == y_forecast.shape[0]
        assert y_forecast_full[0].shape == y_true_full[0].shape
        assert bool(torch.all(torch.isnan(y_forecast_full))) is False

        total_full_timesteps = y_forecast_full.shape[0] * y_forecast_full.shape[1]
        total_partial_timesteps = y_forecast_partial.shape[0] * y_forecast_partial.shape[1]
        w_full = total_full_timesteps / (total_full_timesteps + total_partial_timesteps)
        w_partial = total_partial_timesteps / (total_full_timesteps + total_partial_timesteps)
        loss_full = self.forecaster_loss(y_forecast_full, y_true_full)
        return (w_full * loss_full) + (w_partial * loss_partial)

    def _split_preds_and_get_loss(
        self, batch, add_preds: bool, loss_label: str, log_loss: bool, training: bool
    ) -> Dict:
        x, label, is_global = batch

        # TODO: Reconcile below with forward()
        x = x[:, :, self.bands_to_use]

        loss: Union[float, torch.Tensor] = 0
        output_dict: Dict[str, Union[float, torch.Tensor, Dict]] = {}

        if self.forecast_eval_data or (training and self.forecast_training_data):
            # -------------------------------------------------------------------------------
            # Forecast
            # -------------------------------------------------------------------------------
            input_to_encode = x[:, : self.available_timesteps, :]
            assert (
                torch.isnan(input_to_encode).any().item() is False
            ), "Forecast input contains nans"
            encoder_output = self.forecaster(input_to_encode)

            # -------------------------------------------------------------------------------
            # Compute loss
            # -------------------------------------------------------------------------------
            x_has_nans = torch.isnan(x).any().item()
            if not x_has_nans:
                loss = self._compute_forecaster_loss(y_true=x[:, 1:, :], y_forecast=encoder_output)

            # -------------------------------------------------------------------------------
            # Create a full time series by concatenating ground truth with the forecast
            # [GT for input months, Forecast for output_timesteps]
            # -------------------------------------------------------------------------------
            final_encoded_input = torch.cat(
                (
                    (
                        self.add_noise(input_to_encode, training),
                        # -1 because the encoder output has no value for the 0th
                        # timestep
                        # fmt: off
                        encoder_output[:, self.available_timesteps - 1:],
                        # fmt: on
                    )
                ),
                dim=1,
            )

            # -------------------------------------------------------------------------------
            # Set the input to the classifier (x) using the forecasted values
            # -------------------------------------------------------------------------------
            if training:
                # Use the original AND forecasted time series to train
                if x_has_nans:
                    # Use forecasted time series and the original full time series if available
                    nan_batch_index = torch.isnan(x).any(dim=1).any(dim=1)
                    x_full_time_series_w_noise = self.add_noise(
                        x[~nan_batch_index], training=training
                    )
                    assert bool(torch.any(torch.isnan(x_full_time_series_w_noise))) is False
                    x = torch.cat((x_full_time_series_w_noise, final_encoded_input), dim=0)
                    label = torch.cat((label[~nan_batch_index], label), dim=0)
                    is_global = torch.cat((is_global[~nan_batch_index], is_global), dim=0)
                else:
                    # Use forecasted time series and original full time series for training
                    x = torch.cat((x, final_encoded_input), dim=0)
                    label = torch.cat((label, label), dim=0)
                    is_global = torch.cat((is_global, is_global), dim=0)

            else:
                # Use only the forecasted time series for evaluation
                x = final_encoded_input

        else:
            x = self.add_noise(x, training=training)

        org_global_preds, local_preds = self.classifier(x)
        global_preds = org_global_preds[is_global != 0]
        global_labels = label[is_global != 0]

        local_preds = local_preds[is_global == 0]
        local_labels = label[is_global == 0]

        if local_preds.shape[0] > 0:
            local_loss = self.local_loss_function(
                local_preds.squeeze(-1),
                local_labels,
            )
            loss += local_loss

        if global_preds.shape[0] > 0:
            global_loss = self.global_loss_function(
                global_preds.squeeze(-1),
                global_labels,
            )

            num_local_labels = local_preds.shape[0]
            if num_local_labels == 0:
                alpha = 1
            else:
                ratio = global_preds.shape[0] / num_local_labels
                alpha = ratio / self.hparams.alpha
            loss += alpha * global_loss

        output_dict[loss_label] = loss
        if log_loss:
            output_dict["log"] = {loss_label: loss}
        if add_preds:
            output_dict.update(
                {
                    "global_pred": global_preds,
                    "global_label": global_labels,
                    "local_pred": local_preds,
                    "local_label": local_labels,
                }
            )
        return output_dict

    def training_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=False, loss_label="loss", log_loss=True, training=True
        )

    def validation_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="val_loss", log_loss=True, training=False
        )

    def test_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="test_loss", log_loss=True, training=False
        )

    def _interpretable_metrics(self, outputs, input_prefix: str) -> Dict:
        preds = torch.cat([x[f"{input_prefix}pred"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x[f"{input_prefix}label"] for x in outputs]).detach().cpu().numpy()
        return self._output_metrics(preds, labels)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.val_losses.append(avg_loss.item())
        logs = {
            "val_loss": avg_loss,
            "epoch": self.current_epoch,
            "val_loss_min": min(self.val_losses),
        }
        metrics = self._interpretable_metrics(outputs, "local_")
        logs.update(metrics)

        # Save model with lowest validation loss
        model_ckpt_path = PROJECT_ROOT / DataPaths.MODELS / f"{self.hparams.model_name}.ckpt"
        save_model_condition = self.current_epoch > 0 and (
            not model_ckpt_path.exists() or (self.val_losses[-1] == min(self.val_losses[1:]))
        )
        if save_model_condition:
            saved_metrics = {f"{k}_saved": v for k, v in metrics.items()}
            logs.update(saved_metrics)
            self.trainer.save_checkpoint(model_ckpt_path)
        return {"log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()
        output_dict = {"test_loss": avg_loss}
        output_dict.update(self._interpretable_metrics(outputs, "local_"))
        return {"progress_bar": output_dict}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            # assumes this is being run from "scripts"
            "--learning_rate": (float, 0.001),
            "--batch_size": (int, 128),
            "--probability_threshold": (float, 0.5),
            "--alpha": (float, 10),
            "--noise_factor": (float, 0.1),
            "--epochs": (int, 1000),
            "--patience": (int, 10),
        }

        for key, val in parser_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        parser.add_argument("--cache", dest="cache", action="store_true")
        parser.add_argument("--do_not_cache", dest="cache", action="store_false")
        parser.set_defaults(cache=True)

        parser.add_argument("--upsample", dest="upsample", action="store_true")
        parser.add_argument("--do_not_upsample", dest="upsample", action="store_false")
        parser.set_defaults(upsample=True)

        classifier_parser = Classifier.add_model_specific_args(parser)
        return Forecaster.add_model_specific_args(classifier_parser)

    def save(self):
        sm = torch.jit.script(self)
        model_path = PROJECT_ROOT / DataPaths.MODELS / f"{self.hparams.model_name}.pt"
        if model_path.exists():
            model_path.unlink()
        sm.save(str(model_path))
