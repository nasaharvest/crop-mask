"""
Trains the models
"""

import sys
from argparse import ArgumentParser
import subprocess
from shutil import copyfile
from pathlib import Path
from clearml import Task

sys.path.append("..")

from src.models import Forecaster, forecaster_train_model


def run_training(parser):
    model_args = Forecaster.add_model_specific_args(parser).parse_args()
    model = Forecaster(num_bands=14, output_timesteps=7, hparams=model_args)
    forecaster_train_model(model, model_args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data_folder", type=str, default='./data_model')
    parser.add_argument("--show_progress_bar", type=bool, default=False)

    run_training(parser)
