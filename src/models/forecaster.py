from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Dict, Tuple, Type, Any, List, Optional
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
        output_timesteps: int,
        hparams: Namespace,
    ) -> None:
        super().__init__()
        self.output_timesteps = output_timesteps
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

    def get_dataset(
        self,
        subset: str,
        normalizing_dict: Optional[Dict] = None,
        cache: Optional[bool] = None,
    ) -> ForecasterDataset:
        return ForecasterDataset(
            data_folder=Path(self.hparams.data_folder),
            subset=subset,
            datasets=self.datasets,
            normalizing_dict=normalizing_dict,
            cache=self.hparams.cache if cache is None else cache,
            upsample=self.hparams.upsample if subset != "testing" else False,
        )