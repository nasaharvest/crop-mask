"""
Script that uses argument parameters to train an individual model
"""
import os
import geopandas as gpd
import sys
import cartopy.io.shapereader as shpreader
from argparse import ArgumentParser

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.datasets_labeled import labeled_datasets  # noqa: E402
from src.pipeline_funcs import train_model  # noqa: E402
from src.models import Model  # noqa: E402

all_datasets_str = ",".join(
    [ld.dataset for ld in labeled_datasets if ld.dataset != "one_acre_fund"]
)

if __name__ == "__main__":

    # ne_shapefile = shpreader.natural_earth(
    #     resolution="10m", category="cultural", name="admin_0_countries"
    # )
    # ne_gdf = gpd.read_file(ne_shapefile)
    # geom = ne_gdf[ne_gdf["ADMIN"] == "Rwanda"].iloc[0].geometry
    # min_lon, min_lat, max_lon, max_lat = geom.bounds

    train_datasets_str = (
        "geowiki_landcover_2017,Kenya,Mali,Mali_lower_CEO_2019,Mali_upper_CEO_2019,"
        + "Togo,Rwanda,Ethiopia,Ethiopia_Tigray_2020,Ethiopia_Tigray_2021,"
        + "digitalearthafrica_eastern,Ethiopia_Bure_Jimma_2019,"
        + "Ethiopia_Bure_Jimma_2020,Uganda,open_buildings"
    )

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Rwanda_2019")
    parser.add_argument("--min_lat", type=float, default=-3.027)
    parser.add_argument("--max_lat", type=float, default=-1.034)
    parser.add_argument("--min_lon", type=float, default=28.431)
    parser.add_argument("--max_lon", type=float, default=31.013)
    parser.add_argument("--train_datasets", type=str, default=train_datasets_str)
    parser.add_argument("--eval_datasets", type=str, default="Rwanda")
    # parser.add_argument("--up_to_year", type=int, default=2020)
    parser.add_argument("--start_month", type=str, default="February")
    parser.add_argument("--input_months", type=int, default=12)
    hparams = Model.add_model_specific_args(parser).parse_args()
    _, metrics = train_model(hparams)
    print(metrics)
