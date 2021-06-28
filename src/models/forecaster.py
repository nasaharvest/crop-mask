from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from typing import cast, Callable, Dict, Tuple, Type, Any, List, Optional, Union

from .lstm import UnrolledLSTM
from .forecaster_dataset import ForecasterDataset

from data.datasets_labeled import labeled_datasets

class Forecaster(pl.LightningModule):
    r"""
    An LSTM based model to predict a multispectral sequence.

    :param input_size: The number of input bands passed to the model. The
        input vector is expected to be of shape [batch_size, timesteps, bands]
    :param output_timesteps: The number of timesteps to predict

    hparams
    --------
    The default values for these parameters are set in add_model_specific_args

    :param hparams.forecasting_vector_size: The size of the hidden vector in the LSTM
        Default = 128
    :param hparams.forecasting_dropout: Variational dropout ratio to apply between timesteps in
        the LSTM base. Default = 0.2
    """

    def __init__(
        self,
        num_bands: int,
        hparams: Namespace,
    ) -> None:
        super().__init__()

        self.lstm = UnrolledLSTM(
            input_size=num_bands,
            hidden_size=hparams.forecasting_vector_size,
            dropout=hparams.forecasting_dropout,
            batch_first=True,
        )

        self.to_bands = nn.Linear(
            in_features=hparams.forecasting_vector_size, out_features=num_bands
        )

        self.hparams = hparams
        self.datasets = labeled_datasets

        dataset = self.get_dataset(subset="training", cache=False)
        
        self.num_timesteps = dataset.num_timesteps
        self.output_timesteps = self.num_timesteps - self.hparams.input_months

        # we save the normalizing dict because we calculate weighted
        # normalization values based on the datasets we combine.
        # The number of instances per dataset (and therefore the weights) can
        # vary between the train / test / val sets - this ensures the normalizing
        # dict stays constant between them
        self.normalizing_dict: Optional[Dict[str, np.ndarray]] = dataset.normalizing_dict

        self.forecaster_loss = F.smooth_l1_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        hidden_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        input_timesteps = x.shape[1]
        assert input_timesteps >= 1

        output = torch.empty((1, 1, 1, 1))
        predicted_output: List[torch.Tensor] = []
        for i in range(input_timesteps):
            # fmt: off
            input = x[:, i: i + 1, :]
            # fmt: on
            output, hidden_tuple = self.lstm(input, hidden_tuple)
            output = self.to_bands(torch.transpose(output[0, :, :, :], 0, 1))
            predicted_output.append(output)

        # we have already predicted the first output timestep (the last
        # output of the loop above)
        for i in range(self.output_timesteps - 1):
            output, hidden_tuple = self.lstm(output, hidden_tuple)
            output = self.to_bands(torch.transpose(output[0, :, :, :], 0, 1))
            predicted_output.append(output)
        return torch.cat(predicted_output, dim=1)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            "--forecasting_vector_size": (int, 256),
            "--forecasting_dropout": (float, 0.2),
        }

        for key, vals in parser_args.items():
            parser.add_argument(key, type=vals[0], default=vals[1])

        return parser

    def train_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="training"),
            shuffle=True,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="validation",
                normalizing_dict=self.normalizing_dict,
            ),
            batch_size=self.hparams.batch_size,
        )

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

        x = batch # , label, is_global = batch

        input_to_encode = x[:, : self.hparams.input_months, :]

        # we will predict every timestep except the first one
        output_to_predict = x[:, 1:, :]
        encoder_output = self.forward(input_to_encode)
        encoder_loss = self.forecaster_loss(encoder_output, output_to_predict)
        loss: Union[float, torch.Tensor] = encoder_loss

        output_dict = {
            loss_label: loss
        }
        
        if add_preds:
            output_dict.update(
                {
                    "encoder_prediction": encoder_output,
                    "encoder_target": output_to_predict,
                }
            )
        if log_loss:
            output_dict["log"] = {
                loss_label: loss
            }

        return output_dict

    def training_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=False, loss_label="loss", log_loss=True, training=True
        )

    def validation_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="val_loss", log_loss=True, training=False
        )

    def get_dataset(
        self,
        subset: str,
        normalizing_dict: Optional[Dict] = None,
        cache: Optional[bool] = None,
    ) -> ForecasterDataset:
        return ForecasterDataset(
            data_folder=Path(self.hparams.processed_data_folder),
            subset=subset,
            datasets=self.datasets,
            normalizing_dict=normalizing_dict,
            cache=self.hparams.cache if cache is None else cache
        )