"""
Trains the models
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import logging

sys.path.append("..")

module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models import Model, train_model  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Rwanda")
    parser.add_argument("--datasets", type=str, default="Rwanda,geowiki_landcover_2017")
    parser.add_argument("--local_train_dataset_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data_folder", type=str, default=str(Path("../data")))

    model_args = Model.add_model_specific_args(parser).parse_args()
    model = Model(model_args)
    train_model(model, model_args)
