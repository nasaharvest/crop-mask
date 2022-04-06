"""
Script that uses argument parameters to train an individual model
"""
import os
import sys
from argparse import ArgumentParser

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.pipeline_funcs import train_model  # noqa: E402
from src.models import Model  # noqa: E402

if __name__ == "__main__":

    train_datasets = [
        "geowiki_landcover_2017",
        "Kenya",
        "Mali",
        "Mali_lower_CEO_2019",
        "Mali_upper_CEO_2019",
        "Togo",
        "Rwanda",
        "digitalearthafrica_eastern",
        "Uganda",
        "open_buildings",
        "digitalearthafrica_eastern",
        "digitalearthafrica_sahel",
        "Ethiopia",
        "Ethiopia_Tigray_2020",
        "Ethiopia_Tigray_2021",
        "Ethiopia_Bure_Jimma_2019",
        "Ethiopia_Bure_Jimma_2020",
        "Argentina_Buenos_Aires",
        "Malawi_CEO_2020",
        "Malawi_FAO",
        "Malawi_FAO_corrected",
        "Zambia_CEO_2019",
        "Tanzania_CEO_2019",
        "Malawi_corrected",
    ]

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Malawi_2020") #separate, what model called when deployed
    parser.add_argument("--min_lat", type=float, default=-16.95)
    parser.add_argument("--max_lat", type=float, default=-9.500)
    parser.add_argument("--min_lon", type=float, default=33.109)
    parser.add_argument("--max_lon", type=float, default=35.699)
    parser.add_argument("--train_datasets", type=str, default=",".join(train_datasets))
    parser.add_argument("--eval_datasets", type=str, default="Malawi_CEO_2020")
    # parser.add_argument("--up_to_year", type=int, default=2020)
    parser.add_argument("--start_month", type=str, default="February")
    parser.add_argument("--input_months", type=int, default=12)
    hparams = Model.add_model_specific_args(parser).parse_args()
    _, metrics = train_model(hparams)
    print(metrics)
