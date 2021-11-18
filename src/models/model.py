from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.ETL.dataset import LabeledDataset
import numpy as np
import logging
import json
import xarray as xr
from tqdm import tqdm
from typing import cast, Callable, Tuple, Dict, Any, Type, Optional, List, Union

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
)

from src.ETL.ee_boundingbox import BoundingBox
from src.bounding_boxes import bounding_boxes
from src.utils import data_dir, get_dvc_dir, set_seed
from src.datasets_labeled import labeled_datasets
from .data import CropDataset
from .utils import tif_to_np, preds_to_xr
from .forecaster import Forecaster
from .classifier import Classifier

logger = logging.getLogger(__name__)


class Model(pl.LightningModule):
    r"""
    An model for annual and in-season crop mapping. This model consists of a
    forecaster.Forecaster and a classifier.Classifier - it will require the arguments
    required by those models too.

    hparams
    --------
    The default values for these parameters are set in add_model_specific_args

    :param hparams.data_folder: The path to the data. Default (assumes the model
        is being run from the scripts directory) = "../data"
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
    :param hparams.remove_b1_b10: Whether or not to remove the B1 and B10 bands. Default = True
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
        set_seed()

        self.hparams = hparams

        if "target_bbox_key" in hparams:
            self.target_bbox = bounding_boxes[hparams.target_bbox_key]
        else:
            # Write out parameters explicitly so they are saved in the jit model
            self.min_lon: float = hparams.min_lon
            self.max_lon: float = hparams.max_lon
            self.min_lat: float = hparams.min_lat
            self.max_lat: float = hparams.max_lat
            self.target_bbox = BoundingBox(
                hparams.min_lon, hparams.max_lon, hparams.min_lat, hparams.max_lat
            )

        self.train_datasets = self.load_datasets(hparams.train_datasets, subset="training")
        self.eval_datasets = self.load_datasets(hparams.eval_datasets, subset="evaluation")

        all_dataset_params_path = data_dir / "all_dataset_params.json"
        if all_dataset_params_path.exists():
            with (data_dir / "all_dataset_params.json").open() as f:
                all_dataset_params = json.load(f)
        else:
            all_dataset_params = {}

        normalizing_dict_key = (
            f"{hparams.train_datasets}_{hparams.up_to_year}"
            if "up_to_year" in hparams and hparams.up_to_year
            else hparams.train_datasets
        )
        if normalizing_dict_key not in all_dataset_params:
            dataset = self.get_dataset(subset="training", cache=False)
            # we save the normalizing dict because we calculate weighted
            # normalization values based on the datasets we combine.
            # The number of instances per dataset (and therefore the weights) can
            # vary between the train / test / val sets - this ensures the normalizing
            # dict stays constant between them
            if dataset.normalizing_dict is None:
                raise ValueError("Normalizing dict must be calculated using dataset.")

            all_dataset_params[normalizing_dict_key] = {
                "num_timesteps": dataset.num_timesteps,
                "input_size": dataset.num_input_features,
                "normalizing_dict": {k: v.tolist() for k, v in dataset.normalizing_dict.items()},
            }

            with all_dataset_params_path.open("w") as f:
                json.dump(all_dataset_params, f, ensure_ascii=False, indent=4, sort_keys=True)

        dataset_params = all_dataset_params[normalizing_dict_key]
        self.num_timesteps = dataset_params["num_timesteps"]
        self.input_size = dataset_params["input_size"]

        # Normalizing dict that is exposed
        self.normalizing_dict_jit: Dict[str, List[float]] = dataset_params["normalizing_dict"]
        self.normalizing_dict: Optional[Dict[str, np.ndarray]] = {
            k: np.array(v) for k, v in dataset_params["normalizing_dict"].items()
        }

        # Needed so that forecast is exposed to jit
        self.forecast = self.hparams.forecast
        self.input_months = self.hparams.input_months
        self.forecaster = torch.nn.Identity()
        self.forecaster_output_timesteps = 0
        if self.forecast:
            self.forecaster_input_timesteps = self.hparams.input_months
            self.forecaster_output_timesteps = max(self.num_timesteps) - self.hparams.input_months
        elif len(self.num_timesteps) > 1:
            self.forecaster_input_timesteps = min(self.num_timesteps)
            self.forecaster_output_timesteps = self.hparams.input_months - min(self.num_timesteps)

        self.forecaster = torch.nn.Identity()
        if self.forecaster_output_timesteps > 0:
            self.forecaster = Forecaster(
                num_bands=self.input_size,
                output_timesteps=self.forecaster_output_timesteps,
                hparams=hparams,
            )

            self.forecaster_loss = F.smooth_l1_loss

        self.classifier = Classifier(input_size=self.input_size, hparams=hparams)
        self.global_loss_function: Callable = F.binary_cross_entropy
        self.local_loss_function: Callable = F.binary_cross_entropy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_input = x[:, : self.input_months, :]
        if self.forecast:
            x_forecasted = self.forecaster(x_input)[:, self.input_months - 1 :, :]
            x_input = torch.cat((x_input, x_forecasted), dim=1)
        return self.classifier(x_input)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def load_datasets(
        input_dataset_names: Union[str, List[str]], subset: str
    ) -> List[LabeledDataset]:
        """
        Loads the datasets specified in the input_dataset_names list.
        """
        if isinstance(input_dataset_names, str):
            input_dataset_names = input_dataset_names.replace(" ", "").split(",")
            input_dataset_names = list(filter(None, input_dataset_names))

        datasets = []
        for d in labeled_datasets:
            if d.dataset in input_dataset_names:
                datasets.append(d)
                input_dataset_names.remove(d.dataset)

        for not_found_dataset in input_dataset_names:
            logger.error(f"Could not find dataset with name: {not_found_dataset}")

        logger.info(f"Using {subset} datasets: {[d.dataset for d in datasets]}")
        return datasets

    def get_dataset(
        self,
        subset: str,
        normalizing_dict: Optional[Dict] = None,
        cache: Optional[bool] = None,
        upsample: Optional[bool] = None,
        is_local_only: bool = False,
    ) -> CropDataset:
        if subset == "training":
            datasets = self.train_datasets
        else:
            datasets = self.eval_datasets

        return CropDataset(
            subset=subset,
            datasets=datasets,
            remove_b1_b10=self.hparams.remove_b1_b10,
            normalizing_dict=normalizing_dict,
            cache=self.hparams.cache if cache is None else cache,
            upsample=upsample if upsample is not None else self.hparams.upsample,
            target_bbox=self.target_bbox,
            is_local_only=is_local_only,
            up_to_year=self.hparams.up_to_year if "up_to_year" in self.hparams else None,
            wandb_logger=self.logger,
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="validation",
                normalizing_dict=self.normalizing_dict,
                is_local_only=True,
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
                is_local_only=True,
            ),
            batch_size=self.hparams.batch_size,
        )

    def predict(
        self,
        path_to_file: Path,
        with_forecaster: bool,
        batch_size: int = 64,
        add_ndvi: bool = True,
        add_ndwi: bool = False,
        nan_fill: float = 0,
        days_per_timestep: int = 30,
        local_head: bool = True,
        use_gpu: bool = True,
        disable_tqdm: bool = False,
    ) -> xr.Dataset:

        # check if a GPU is available, and if it is
        # move the model onto the GPU
        device: Optional[torch.device] = None
        if use_gpu:
            use_cuda = torch.cuda.is_available()
            if not use_cuda:
                logger.warning("No GPU - not using one")
            else:
                logger.info("Using GPU")
            device = torch.device("cuda" if use_cuda else "cpu")
            self.to(device)

        self.eval()

        input_data = tif_to_np(
            path_to_file,
            add_ndvi=add_ndvi,
            add_ndwi=add_ndwi,
            nan=nan_fill,
            normalizing_dict=self.normalizing_dict,
            days_per_timestep=days_per_timestep,
        )

        if with_forecaster:
            input_data.x = input_data.x[:, : self.input_months, :]

        predictions: List[np.ndarray] = []
        cur_i = 0

        pbar = tqdm(total=input_data.x.shape[0] - 1, disable=disable_tqdm)
        while cur_i < (input_data.x.shape[0] - 1):
            # fmt: off
            batch_x_np = input_data.x[cur_i: cur_i + batch_size]
            # fmt: on
            if self.hparams.remove_b1_b10:
                batch_x_np = CropDataset._remove_bands(batch_x_np)
            batch_x = torch.from_numpy(batch_x_np).float()

            if use_gpu and (device is not None):
                batch_x = batch_x.to(device)

            with torch.no_grad():
                if with_forecaster:
                    batch_x_next = self.forecaster(batch_x)
                    batch_x = torch.cat((batch_x, batch_x_next), dim=1)

                global_preds, local_preds = self.classifier(batch_x)
                if local_head:
                    batch_preds = local_preds
                else:
                    batch_preds = global_preds

                # back to the CPU, if necessary
                batch_preds = batch_preds.cpu()

            predictions.append(cast(torch.Tensor, batch_preds).numpy())
            cur_i += batch_size
            pbar.update(batch_size)

        all_preds = np.concatenate(predictions, axis=0)
        if len(all_preds.shape) == 1:
            all_preds = np.expand_dims(all_preds, axis=-1)

        return preds_to_xr(
            all_preds,
            lats=input_data.lat,
            lons=input_data.lon,
        )

    def _output_metrics(
        self, preds: np.ndarray, labels: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:

        if len(preds) == 0:
            # sometimes this happens in the warmup
            return {}

        output_dict: Dict[str, float] = {}
        if not (labels == labels[0]).all():
            # This can happen when lightning does its warm up on a subset of the
            # validation data
            output_dict[f"{prefix}roc_auc_score"] = roc_auc_score(labels, preds)

        preds = (preds > self.hparams.probability_threshold).astype(int)

        output_dict[f"{prefix}precision_score"] = precision_score(labels, preds)
        output_dict[f"{prefix}recall_score"] = recall_score(labels, preds)
        output_dict[f"{prefix}f1_score"] = f1_score(labels, preds)
        output_dict[f"{prefix}accuracy"] = accuracy_score(labels, preds)

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
        if y_nans.any() == False:
            return self.forecaster_loss(y_true, y_forecast)

        nan_batch_index = y_nans.any(dim=1).any(dim=1)
        nan_month_index = y_nans.any(dim=0).any(dim=1)

        full_shape = (-1, y_true.shape[1], y_true.shape[2])
        y_true_full = y_true[~nan_batch_index].reshape(full_shape)
        y_forecast_full = y_forecast[~nan_batch_index].reshape(full_shape)

        partial_shape = (-1, sum(~nan_month_index), y_true.shape[2])
        y_true_partial = y_true[nan_batch_index][:, ~nan_month_index].reshape(partial_shape)
        y_forecast_partial = y_forecast[nan_batch_index][:, ~nan_month_index].reshape(partial_shape)

        assert y_forecast_full.shape[0] + y_forecast_partial.shape[0] == y_forecast.shape[0]
        assert y_forecast_full[0].shape == y_true_full[0].shape
        assert torch.all(torch.isnan(y_forecast_full)) == False
        assert torch.all(torch.isnan(y_forecast_partial)) == False

        total_full_timesteps = y_forecast_full.shape[0] * y_forecast_full.shape[1]
        total_partial_timesteps = y_forecast_partial.shape[0] * y_forecast_partial.shape[1]
        w_full = total_full_timesteps / (total_full_timesteps + total_partial_timesteps)
        w_partial = total_partial_timesteps / (total_full_timesteps + total_partial_timesteps)
        loss_full = self.forecaster_loss(y_forecast_full, y_true_full)
        loss_partial = self.forecaster_loss(y_forecast_partial, y_true_partial)
        return (w_full * loss_full) + (w_partial * loss_partial)

    def _split_preds_and_get_loss(
        self, batch, add_preds: bool, loss_label: str, log_loss: bool, training: bool
    ) -> Dict:

        x, label, is_global = batch

        loss: Union[float, torch.Tensor] = 0
        output_dict = {}

        if self.forecaster_output_timesteps > 0:
            # -------------------------------------------------------------------------------
            # Forecast
            # -------------------------------------------------------------------------------
            input_to_encode = x[:, : self.forecaster_input_timesteps, :]
            output_to_predict = x[:, 1:, :]
            encoder_output = self.forecaster(input_to_encode)

            # -------------------------------------------------------------------------------
            # Compute loss
            # -------------------------------------------------------------------------------
            is_full_time_series = (
                x.shape[1] == self.forecaster_input_timesteps + self.forecaster.output_timesteps
            )
            if is_full_time_series:
                loss = self._compute_forecaster_loss(
                    y_true=output_to_predict, y_forecast=encoder_output
                )

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
                        encoder_output[:, self.forecaster_input_timesteps - 1:],
                        # fmt: on
                    )
                ),
                dim=1,
            )

            # -------------------------------------------------------------------------------
            # Set the input to the classifier (x) using the forecasted values
            # -------------------------------------------------------------------------------
            if is_full_time_series:
                nan_batch_index = x.any(dim=1).any(dim=1)
                x_full_time_series_w_noise = self.add_noise(x[~nan_batch_index], training=training)
                assert torch.any(torch.isnan(x_full_time_series_w_noise)) == False
                x = torch.cat((x_full_time_series_w_noise, final_encoded_input), dim=0)
                label = torch.cat((label[~nan_batch_index], label), dim=0)
                is_global = torch.cat((is_global[~nan_batch_index], is_global), dim=0)
            else:
                x = final_encoded_input

            if add_preds:
                output_dict["encoder_prediction"] = encoder_output
                output_dict["encoder_target"] = output_to_predict

        else:
            x = x[:, : self.hparams.input_months]
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

    @staticmethod
    def _split_tensor(outputs, label) -> Tuple[np.ndarray, np.ndarray]:
        encoded_all, unencoded_all = [], []
        for x in outputs:
            # the first half is unencoded, the second is encoded
            total = x[label]
            unencoded_all.append(total[: total.shape[0] // 2])
            # fmt: off
            encoded_all.append(total[total.shape[0] // 2:])
            # fmt: on
        return (
            torch.cat(unencoded_all).detach().cpu().numpy(),
            torch.cat(encoded_all).detach().cpu().numpy(),
        )

    @staticmethod
    def _get_output_as_np(outputs, label: str):
        return torch.cat([x[label] for x in outputs]).detach().cpu().numpy()

    def _interpretable_metrics(self, outputs, input_prefix: str, output_prefix: str) -> Dict:

        output_dict = {}

        if self.forecast:
            if self.hparams.evaluate_forecast:
                u_labels, e_labels = self._split_tensor(outputs, f"{input_prefix}label")
                u_preds, e_preds = self._split_tensor(outputs, f"{input_prefix}pred")
            else:
                e_preds = self._get_output_as_np(outputs, f"{input_prefix}pred")
                e_labels = self._get_output_as_np(outputs, f"{input_prefix}label")

        else:
            u_preds = self._get_output_as_np(outputs, f"{input_prefix}pred")
            u_labels = self._get_output_as_np(outputs, f"{input_prefix}label")

        if self.forecast:
            output_dict.update(
                self._output_metrics(e_preds, e_labels, f"encoded_{output_prefix}{input_prefix}")
            )

        if not self.forecast or (self.forecast and self.hparams.evaluate_forecast):
            output_dict.update(
                self._output_metrics(u_preds, u_labels, f"unencoded_{output_prefix}{input_prefix}")
            )

        return output_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        if self.forecast:
            encoder_pred = (
                torch.cat(
                    [torch.flatten(x["encoder_prediction"], start_dim=1) for x in outputs],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            encoder_target = (
                torch.cat(
                    [torch.flatten(x["encoder_target"], start_dim=1) for x in outputs],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            if self.hparams.evaluate_forecast:
                logs["val_encoder_mae"] = mean_absolute_error(encoder_target, encoder_pred)

        logs.update(self._interpretable_metrics(outputs, "local_", "val_"))
        return {"log": logs}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()
        output_dict = {"test_loss": avg_loss}

        if self.forecast:
            encoder_pred = (
                torch.cat(
                    [torch.flatten(x["encoder_prediction"], start_dim=1) for x in outputs],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            encoder_target = (
                torch.cat(
                    [torch.flatten(x["encoder_target"], start_dim=1) for x in outputs],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            if self.hparams.evaluate_forecast:
                output_dict["test_encoder_mae"] = mean_absolute_error(encoder_target, encoder_pred)

        output_dict.update(self._interpretable_metrics(outputs, "local_", "test_"))

        return {"progress_bar": output_dict}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            # assumes this is being run from "scripts"
            "--learning_rate": (float, 0.001),
            "--batch_size": (int, 256),
            "--probability_threshold": (float, 0.5),
            "--input_months": (int, 12),
            "--alpha": (float, 10),
            "--noise_factor": (float, 0.1),
            "--max_epochs": (int, 1000),
            "--patience": (int, 10),
        }

        for key, val in parser_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        parser.add_argument("--remove_b1_b10", dest="remove_b1_b10", action="store_true")
        parser.add_argument("--keep_b1_b10", dest="remove_b1_b10", action="store_false")
        parser.set_defaults(remove_b1_b10=True)

        parser.add_argument("--forecast", dest="forecast", action="store_true")
        parser.add_argument("--do_not_forecast", dest="forecast", action="store_false")
        parser.set_defaults(forecast=False)

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
        model_path = get_dvc_dir("models") / f"{self.hparams.model_name}.pt"
        if model_path.exists():
            model_path.unlink()
        sm.save(model_path)
