"""
Script that uses argument parameters to train an individual model
"""
import os
import sys
from argparse import ArgumentParser
import logging

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.datasets_labeled import labeled_datasets  # noqa: E402
from src.bounding_boxes import bounding_boxes  # noqa: E402
from src.pipeline_funcs import train_model  # noqa: E402
from src.models import Model  # noqa: E402
from src.utils import get_dvc_dir  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_datasets_str = ",".join(
    [ld.dataset for ld in labeled_datasets if ld.dataset != "one_acre_fund"]
)

model_folder = get_dvc_dir("models")
data_folder = model_folder.parent

if __name__ == "__main__":

    bbox = bounding_boxes["Ethiopia_Tigray"]

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Ethiopia_Tigray_2020")
    parser.add_argument("--min_lat", type=float, default=bbox.min_lat)
    parser.add_argument("--max_lat", type=float, default=bbox.max_lat)
    parser.add_argument("--min_lon", type=float, default=bbox.min_lon)
    parser.add_argument("--max_lon", type=float, default=bbox.max_lon)
    parser.add_argument("--train_datasets", type=str, default=all_datasets_str)
    parser.add_argument("--eval_datasets", type=str, default="Ethiopia_Tigray_2020")
    parser.add_argument("--data_folder", type=str, default=str(data_folder))
    parser.add_argument("--model_dir", type=str, default=str(data_folder / "models"))
    hparams = Model.add_model_specific_args(parser).parse_args()
    train_model(hparams)
