"""
Trains the models
"""

import sys
from argparse import ArgumentParser

sys.path.append("..")

from src.models import Forecaster, forecaster_train_model


def run_training(parser):
    model_args = Forecaster.add_model_specific_args(parser).parse_args()
    model = Forecaster(num_bands=model_args.num_bands, hparams=model_args)
    forecaster_train_model(model, model_args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache", type=bool, default=False)
    parser.add_argument("--data_folder", type=str, default='./data')
    parser.add_argument("--input_months", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="temp")
    parser.add_argument("--num_bands", type=int, default=14)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--show_progress_bar", type=bool, default=False)

    run_training(parser)
