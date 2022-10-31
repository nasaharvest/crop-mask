"""
Script that uses argument parameters to train an individual model
"""
from argparse import ArgumentParser
from openmapflow.bbox import BBox

from src.bboxes import bboxes
from src.pipeline_funcs import train_model
from src.models import Model

from datasets import datasets

selected_bbox = bboxes["Namibia_North"]

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="Namibia_North_2020")
parser.add_argument("--eval_datasets", type=str, default="Namibia_CEO_2020")
parser.add_argument("--train_datasets", type=str, default=",".join([d.dataset for d in datasets]))
parser.add_argument("--min_lat", type=float, default=selected_bbox.min_lat)
parser.add_argument("--max_lat", type=float, default=selected_bbox.max_lat)
parser.add_argument("--min_lon", type=float, default=selected_bbox.min_lon)
parser.add_argument("--max_lon", type=float, default=selected_bbox.max_lon)
parser.add_argument("--up_to_year", type=int, default=2021)
parser.add_argument("--start_month", type=str, default="September")
parser.add_argument("--input_months", type=int, default=12)

parser.add_argument("--skip_era5", dest="skip_era5", action="store_true")
parser.set_defaults(skip_era5=False)
parser.add_argument("--wandb", dest="wandb", action="store_true")
parser.set_defaults(wandb=False)

hparams = Model.add_model_specific_args(parser).parse_args()
print(
    BBox(
        min_lat=hparams.min_lat,
        max_lat=hparams.max_lat,
        min_lon=hparams.min_lon,
        max_lon=hparams.max_lon,
    ).url
)

_, metrics = train_model(hparams)
print(metrics)
