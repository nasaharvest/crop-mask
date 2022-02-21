"""
Script that uses argument parameters to train an individual model
"""
import os
import sys
from argparse import ArgumentParser

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.datasets_labeled import labeled_datasets  # noqa: E402
from src.ETL.boundingbox import BoundingBox  # noqa: E402
from src.pipeline_funcs import train_model  # noqa: E402
from src.models import Model  # noqa: E402

all_datasets_str = ",".join(
    [ld.dataset for ld in labeled_datasets if ld.dataset != "one_acre_fund"]
)

if __name__ == "__main__":

    # Ethiopia Tigray bounding box
    bbox = BoundingBox(min_lon=36.45, max_lon=40.00, min_lat=12.25, max_lat=14.91)

    train_datasets_str = (
        "geowiki_landcover_2017,Kenya,Mali,Mali_lower_CEO_2019,Mali_upper_CEO_2019,"
        + "Togo,Rwanda,Ethiopia,Ethiopia_Tigray_2020,"
        + "digitalearthafrica_eastern,Ethiopia_Bure_Jimma_2019,"
        + "Ethiopia_Bure_Jimma_2020"
    )

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Ethiopia_Tigray_2021")
    parser.add_argument("--min_lat", type=float, default=bbox.min_lat)
    parser.add_argument("--max_lat", type=float, default=bbox.max_lat)
    parser.add_argument("--min_lon", type=float, default=bbox.min_lon)
    parser.add_argument("--max_lon", type=float, default=bbox.max_lon)
    parser.add_argument("--train_datasets", type=str, default=train_datasets_str)
    parser.add_argument("--eval_datasets", type=str, default="Ethiopia_Tigray_2021")
    #parser.add_argument("--up_to_year", type=int, default=2020)
    parser.add_argument("--start_month", type=str, default="February")
    parser.add_argument("--input_months", type=int, default=12)
    hparams = Model.add_model_specific_args(parser).parse_args()
    _, metrics = train_model(hparams)
    print(metrics)
