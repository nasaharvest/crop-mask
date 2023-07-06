"""
Script that uses argument parameters to train an individual model
"""
from argparse import ArgumentParser

from datasets import datasets
from src.bboxes import bboxes
from src.models import Model
from src.pipeline_funcs import train_model

train_datasets = [d.name for d in datasets if d.name != "NamibiaFieldBoundary2022"]

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="Sudan_Blue_Nile_2019")
parser.add_argument("--eval_datasets", type=str, default="Sudan_Blue_Nile_CEO_2019")
parser.add_argument("--train_datasets", type=str, default=",".join(train_datasets))
parser.add_argument("--bbox", type=str, default="Sudan_Blue_Nile")
parser.add_argument("--up_to_year", type=int, default=2022)
parser.add_argument("--start_month", type=str, default="February")
parser.add_argument("--input_months", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--skip_era5", dest="skip_era5", action="store_true")
parser.add_argument("--skip_era5_s1", dest="skip_era5_s1", action="store_true")
parser.set_defaults(skip_era5=False)
parser.set_defaults(skip_era5_s1=False)
parser.add_argument("--wandb", dest="wandb", action="store_true")
parser.set_defaults(wandb=False)

hparams = Model.add_model_specific_args(parser).parse_args()
if hparams.bbox not in bboxes:
    raise ValueError(f"bbox {hparams.bbox} not in {list(bboxes.keys())}")

print(bboxes[hparams.bbox].url)

_, metrics = train_model(hparams)
print(metrics)
