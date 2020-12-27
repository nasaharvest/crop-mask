from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm

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

from .data import CropDataset
from .utils import tif_to_np, preds_to_xr
from src.utils import set_seed
from .forecaster import Forecaster
from .classifier import Classifier

from typing import cast, Callable, Tuple, Dict, Any, Type, Optional, List, Union


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
    :param hparams.include_geowiki: Whether to include the global GeoWiki dataset during
        training. Default = True
    :param hparams.upsample: Whether to oversample the under-represented class so that each class
        is equally represented in the training and validation dataset. Default = True
    """

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        set_seed()
        self.hparams = hparams

        self.data_folder = Path(hparams.data_folder)

        dataset = self.get_dataset(subset="training", cache=False)
        self.num_outputs = dataset.num_output_classes
        self.num_timesteps = dataset.num_timesteps
        self.input_size = dataset.num_input_features

        # we save the normalizing dict because we calculate weighted
        # normalization values based on the datasets we combine.
        # The number of instances per dataset (and therefore the weights) can
        # vary between the train / test / val sets - this ensures the normalizing
        # dict stays constant between them
        self.normalizing_dict = dataset.normalizing_dict

        if self.hparams.forecast:
            num_output_timesteps = self.num_timesteps - self.hparams.input_months
            print(f"Predicting {num_output_timesteps} timesteps in the forecaster")
            self.forecaster = Forecaster(
                num_bands=self.input_size, output_timesteps=num_output_timesteps, hparams=hparams,
            )

            self.forecaster_loss = F.smooth_l1_loss

        self.classifier = Classifier(input_size=self.input_size, hparams=hparams)
        self.global_loss_function: Callable = F.binary_cross_entropy
        self.local_loss_function: Callable = F.binary_cross_entropy

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # To keep the ABC happy
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def get_dataset(
        self, subset: str, normalizing_dict: Optional[Dict] = None, cache: Optional[bool] = None,
    ) -> CropDataset:
        return CropDataset(
            data_folder=self.data_folder,
            subset=subset,
            probability_threshold=self.hparams.probability_threshold,
            remove_b1_b10=self.hparams.remove_b1_b10,
            normalizing_dict=normalizing_dict,
            include_geowiki=self.hparams.include_geowiki if subset != "testing" else False,
            cache=self.hparams.cache if cache is None else cache,
            upsample=self.hparams.upsample if subset != "testing" else False,
            noise_factor=self.hparams.noise_factor if subset != "testing" else 0,
        )

    def train_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="training"), shuffle=True, batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="validation", normalizing_dict=self.normalizing_dict,),
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="testing", normalizing_dict=self.normalizing_dict,),
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
    ) -> xr.Dataset:

        # check if a GPU is available, and if it is
        # move the model onto the GPU
        device: Optional[torch.device] = None
        if use_gpu:
            use_cuda = torch.cuda.is_available()
            if not use_cuda:
                print("No GPU - not using one")
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
            input_data.x = input_data.x[:, : self.hparams.input_months, :]

        predictions: List[np.ndarray] = []
        cur_i = 0

        pbar = tqdm(total=input_data.x.shape[0] - 1)
        while cur_i < (input_data.x.shape[0] - 1):

            batch_x_np = input_data.x[cur_i : cur_i + batch_size]
            if self.hparams.remove_b1_b10:
                batch_x_np = CropDataset._remove_bands(batch_x_np)
            batch_x = torch.from_numpy(batch_x_np).float()

            if use_gpu and (device is not None):
                batch_x = batch_x.to(device)

            with torch.no_grad():
                if with_forecaster:
                    batch_x_next = self.forecaster(batch_x)
                    batch_x = torch.cat((batch_x, batch_x_next), dim=1)

                batch_preds = self.classifier(batch_x)

                if self.hparams.multi_headed:
                    global_preds, local_preds = batch_preds

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

        return preds_to_xr(all_preds, lats=input_data.lat, lons=input_data.lon,)

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

    def _split_preds_and_get_loss(
        self, batch, add_preds: bool, loss_label: str, log_loss: bool, training: bool
    ) -> Dict:

        x, label, is_global = batch

        input_to_encode = x[:, : self.hparams.input_months, :]

        if self.hparams.forecast:
            # we will predict every timestep except the first one
            output_to_predict = x[:, 1:, :]
            encoder_output = self.forecaster(input_to_encode)
            encoder_loss = self.forecaster_loss(encoder_output, output_to_predict)
            loss: Union[float, torch.Tensor] = encoder_loss

            final_encoded_input = torch.cat(
                (
                    (
                        self.add_noise(input_to_encode, training),
                        # -1 because the encoder output has no value for the 0th
                        # timestep
                        encoder_output[:, self.hparams.input_months - 1 :, :],
                    )
                ),
                dim=1,
            )

            output_dict = {}
            if add_preds:
                output_dict.update(
                    {"encoder_prediction": encoder_output, "encoder_target": output_to_predict,}
                )
            if log_loss:
                output_dict["log"] = {}

            # we now repeat label and is_global
            x = torch.cat((self.add_noise(x, training), final_encoded_input), dim=0)
            label = torch.cat((label, label), dim=0)
            is_global = torch.cat((is_global, is_global), dim=0)
        else:
            loss = 0
            output_dict = {}
            if log_loss:
                output_dict["log"] = {}
            x = self.add_noise(input_to_encode, training=training)

        if self.hparams.multi_headed:
            org_global_preds, local_preds = self.classifier(x)
            global_preds = org_global_preds[is_global != 0]
            global_labels = label[is_global != 0]

            local_preds = local_preds[is_global == 0]
            local_labels = label[is_global == 0]

            if local_preds.shape[0] > 0:
                local_loss = self.local_loss_function(local_preds.squeeze(-1), local_labels,)
                loss += local_loss

            if global_preds.shape[0] > 0:
                global_loss = self.global_loss_function(global_preds.squeeze(-1), global_labels,)

                num_local_labels = local_preds.shape[0]
                if num_local_labels == 0:
                    alpha = 1
                else:
                    ratio = global_preds.shape[0] / num_local_labels
                    alpha = ratio / self.hparams.alpha
                loss += alpha * global_loss

            output_dict[loss_label] = loss
            if log_loss:
                output_dict["log"][loss_label] = loss
            if add_preds:
                output_dict.update(
                    {
                        "global_pred": global_preds,
                        "global_label": global_labels,
                        "kenya_pred": local_preds,
                        "kenya_label": local_labels,
                    }
                )
            return output_dict
        else:
            preds = cast(torch.Tensor, self.classifier(x))

            loss += self.global_loss_function(input=preds.squeeze(-1), target=label,)

            output_dict = {loss_label: loss}
            if log_loss:
                output_dict["log"][loss_label] = loss
            if add_preds:
                output_dict.update({"pred": preds, "label": label})
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
            encoded_all.append(total[total.shape[0] // 2 :])
        return (
            torch.cat(unencoded_all).detach().cpu().numpy(),
            torch.cat(encoded_all).detach().cpu().numpy(),
        )

    def _interpretable_metrics(self, outputs, input_prefix: str, output_prefix: str) -> Dict:

        output_dict = {}

        if self.hparams.forecast:
            u_labels, e_labels = self._split_tensor(outputs, f"{input_prefix}label")

            u_preds, e_preds = self._split_tensor(outputs, f"{input_prefix}pred")
        else:
            u_preds = torch.cat([x[f"{input_prefix}pred"] for x in outputs]).detach().cpu().numpy()
            u_labels = (
                torch.cat([x[f"{input_prefix}label"] for x in outputs]).detach().cpu().numpy()
            )

        output_dict.update(
            self._output_metrics(u_preds, u_labels, f"unencoded_{output_prefix}{input_prefix}")
        )

        if self.hparams.forecast:
            output_dict.update(
                self._output_metrics(e_preds, e_labels, f"encoded_{output_prefix}{input_prefix}")
            )

        return output_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        if self.hparams.forecast:
            encoder_pred = (
                torch.cat(
                    [torch.flatten(x["encoder_prediction"], start_dim=1) for x in outputs], dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            encoder_target = (
                torch.cat(
                    [torch.flatten(x["encoder_target"], start_dim=1) for x in outputs], dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )

            logs["val_encoder_mae"] = mean_absolute_error(encoder_target, encoder_pred)

        if self.hparams.multi_headed:
            logs.update(self._interpretable_metrics(outputs, "global_", "val_"))
            logs.update(self._interpretable_metrics(outputs, "kenya_", "val_"))
        else:
            logs.update(self._interpretable_metrics(outputs, "", "val_"))
        return {"val_loss": avg_loss, "log": logs}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()
        output_dict = {"val_loss": avg_loss}

        if self.hparams.forecast:
            encoder_pred = (
                torch.cat(
                    [torch.flatten(x["encoder_prediction"], start_dim=1) for x in outputs], dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            encoder_target = (
                torch.cat(
                    [torch.flatten(x["encoder_target"], start_dim=1) for x in outputs], dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )

            output_dict["test_encoder_mae"] = mean_absolute_error(encoder_target, encoder_pred)

        if self.hparams.multi_headed:
            output_dict.update(self._interpretable_metrics(outputs, "global_", "test_"))
            output_dict.update(self._interpretable_metrics(outputs, "kenya_", "test_"))
        else:
            output_dict.update(self._interpretable_metrics(outputs, "", "test_"))

        return {"progress_bar": output_dict}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            # assumes this is being run from "scripts"
            "--data_folder": (str, str(Path("../data"))),
            "--learning_rate": (float, 0.001),
            "--batch_size": (int, 64),
            "--probability_threshold": (float, 0.5),
            "--input_months": (int, 5),
            "--alpha": (float, 10),
            "--noise_factor": (float, 0.1),
        }

        for key, val in parser_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        parser.add_argument("--remove_b1_b10", dest="remove_b1_b10", action="store_true")
        parser.add_argument("--keep_b1_b10", dest="remove_b1_b10", action="store_false")
        parser.set_defaults(remove_b1_b10=True)

        parser.add_argument("--forecast", dest="forecast", action="store_true")
        parser.add_argument("--do_not_forecast", dest="forecast", action="store_false")
        parser.set_defaults(forecast=True)

        parser.add_argument("--cache", dest="cache", action="store_true")
        parser.add_argument("--do_not_cache", dest="cache", action="store_false")
        parser.set_defaults(cache=True)

        parser.add_argument("--include_geowiki", dest="include_geowiki", action="store_true")
        parser.add_argument("--exclude_geowiki", dest="include_geowiki", action="store_false")
        parser.set_defaults(include_geowiki=True)

        parser.add_argument("--upsample", dest="upsample", action="store_true")
        parser.add_argument("--do_not_upsample", dest="upsample", action="store_false")
        parser.set_defaults(upsample=True)

        classifier_parser = Classifier.add_model_specific_args(parser)
        return Forecaster.add_model_specific_args(classifier_parser)
