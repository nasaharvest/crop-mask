from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
from torch import nn

from typing import Any, Dict, List, Union, Tuple, Type
from .lstm import UnrolledLSTM


class Classifier(pl.LightningModule):
    r"""
    An LSTM based model to predict the presence of cropland in a pixel.

    :param input_size: The number of input bands passed to the model. The
        input vector is expected to be of shape [batch_size, timesteps, bands]

    hparams
    --------
    The default values for these parameters are set in add_model_specific_args

    :param hparams.classifier_vector_size: The size of the hidden vector in the LSTM base
        (and therefore of the first classification layer). Default = 128
    :param hparams.classifier_base_layers: The number of LSTM base layers to use. Default = 1
    :param hparams.classifier_dropout: Variational dropout ratio to apply between timesteps in
        the LSTM base. Default = 0.2
    :param hparams.num_global_layers: The number of classification layers to use on the global
        (GeoWiki) dataset. Default = 1
    :param hparams.num_local_layers: The number of classification layers to use on the local
        (Kenya) dataset. Default = 2
    :param hparams.multi_headed: Whether or not to add a local head, to classify instances within
        Togo. If False, the same classification layer will be used to classify
        all pixels. Default = True
    """

    def __init__(self, input_size: int, hparams: Namespace,) -> None:
        super().__init__()

        self.hparams = hparams

        self.base = nn.ModuleList(
            [
                UnrolledLSTM(
                    input_size=input_size if i == 0 else hparams.classifier_vector_size,
                    hidden_size=hparams.classifier_vector_size,
                    dropout=hparams.classifier_dropout,
                    batch_first=True,
                )
                for i in range(hparams.classifier_base_layers)
            ]
        )

        self.batchnorm = nn.BatchNorm1d(num_features=self.hparams.classifier_vector_size)

        global_classification_layers: List[nn.Module] = []
        num_global_layers = hparams.num_global_layers
        print(f"Using {num_global_layers} layers for the global classifier")
        for i in range(num_global_layers):
            global_classification_layers.append(
                nn.Linear(
                    in_features=hparams.classifier_vector_size,
                    out_features=1
                    if i == (num_global_layers - 1)
                    else hparams.classifier_vector_size,
                    bias=True if i == 0 else False,
                )
            )
            if i < (num_global_layers - 1):
                global_classification_layers.append(nn.ReLU())
                global_classification_layers.append(
                    nn.BatchNorm1d(num_features=hparams.classifier_vector_size)
                )

        self.global_classifier = nn.Sequential(*global_classification_layers)

        if self.hparams.multi_headed:

            num_local_layers = hparams.num_local_layers
            print(f"Using {num_local_layers} layers for the local classifier")
            local_classification_layers: List[nn.Module] = []
            for i in range(num_local_layers):
                local_classification_layers.append(
                    nn.Linear(
                        in_features=hparams.classifier_vector_size,
                        out_features=1
                        if i == (num_local_layers - 1)
                        else hparams.classifier_vector_size,
                        bias=True if i == 0 else False,
                    )
                )
                if i < (num_local_layers - 1):
                    local_classification_layers.append(nn.ReLU())
                    local_classification_layers.append(
                        nn.BatchNorm1d(num_features=hparams.classifier_vector_size,)
                    )

            self.local_classifier = nn.Sequential(*local_classification_layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        for _, lstm in enumerate(self.base):
            x, (hn, _) = lstm(x)
            x = x[:, 0, :, :]

        base = self.batchnorm(hn[-1, :, :])
        x_global = torch.sigmoid(self.global_classifier(base))

        if self.hparams.multi_headed:
            x_local = torch.sigmoid(self.local_classifier(base))
            return x_global, x_local
        else:
            return x_global

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            # assumes this is being run from "scripts"
            "--classifier_vector_size": (int, 128),
            "--classifier_base_layers": (int, 1),
            "--classifier_dropout": (float, 0.2),
            "--num_global_layers": (int, 1),
            "--num_local_layers": (int, 2),
        }

        for key, val in parser_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        parser.add_argument("--multi_headed", dest="multi_headed", action="store_true")
        parser.add_argument("--not_multi_headed", dest="multi_headed", action="store_false")
        parser.set_defaults(multi_headed=True)

        return parser
