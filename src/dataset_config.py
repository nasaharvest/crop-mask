import pandas as pd
from datetime import date, timedelta

from src.ETL.dataset import LabeledDataset, UnlabeledDataset
from src.ETL.ee_exporter import Season
from src.ETL.ee_boundingbox import BoundingBox
from src.ETL.label_downloader import RawLabels
from src.ETL.processor import Processor
from src.constants import LON, LAT


def clean_pv_kenya(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["harvest_da"] != "nan") & (df["harvest_da"] != "unknown")].copy()
    df.loc[:, "planting_d"] = pd.to_datetime(df["planting_d"])
    df.loc[:, "harvest_da"] = pd.to_datetime(df["harvest_da"])

    df["between_days"] = (df["harvest_da"] - df["planting_d"]).dt.days
    year = pd.to_timedelta(timedelta(days=365))
    df.loc[(-365 < df["between_days"]) & (df["between_days"] < 0), "harvest_da"] += year

    valid_years = [2018, 2019, 2020]
    df = df[(df["planting_d"].dt.year.isin(valid_years)) &
            (df["harvest_da"].dt.year.isin(valid_years))].copy()
    df = df[(0 < df["between_days"]) & (df["between_days"] <= 365)]
    return df


def clean_pv_kenya2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename({"Latitude": "lat", "Longitude": "lon"})
    df.loc[:, "Planting Date"] = pd.to_datetime(df["Planting Date"])
    df.loc[:, "Estimated Harvest Date"] = pd.to_datetime(df["Estimated Harvest Date"])
    return df


def clean_geowiki(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("location_id").mean()
    df = df.rename(
        {"loc_cent_X": LON, "loc_cent_Y": LAT, "sumcrop": "mean_sumcrop"},
        axis="columns",
        errors="raise",
    )
    return df.reset_index()


labeled_datasets = [
    LabeledDataset(
        dataset="geowiki_landcover_2017",
        country="global",
        sentinel_dataset="earth_engine_geowiki",
        is_global=True,
        raw_labels=(
            RawLabels("http://store.pangaea.de/Publications/See_2017/crop_all.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/crop_con.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/crop_exp.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_all.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_all_2.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_con.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_exp.zip"),
        ),
        processors=(
            Processor(
                filename="loc_all_2.txt",
                clean_df=clean_geowiki,
                crop_prob=lambda df: df.mean_sumcrop / 100,
                end_year=2018,
                end_month_day=(3, 28),
                custom_start_date=date(2017, 3, 28),
                x_y_from_centroid=False,
                train_val_test=(0.8, 0.2, 0.0),
            ),
        ),
    ),
    LabeledDataset(
        dataset="Kenya",
        country="Kenya",
        sentinel_dataset="earth_engine_kenya",
        processors=(
            Processor(
                filename="noncrop_labels_v2", crop_prob=0.0, end_year=2020, transform_crs_from=32636
            ),
            Processor(
                filename="noncrop_labels_set2",
                crop_prob=0.0,
                end_year=2020,
                transform_crs_from=32636,
            ),
            Processor(filename="2019_gepro_noncrop", crop_prob=0.0, end_year=2020),
            Processor(filename="noncrop_water_kenya_gt", crop_prob=0.0, end_year=2020),
            Processor(filename="noncrop_kenya_gt", crop_prob=0.0, end_year=2020),
            Processor(
                filename="one_acre_fund_kenya",
                crop_prob=1.0,
                end_year=2020,
                clean_df=lambda df: df.rename(columns={"Lat": LAT, "Lon": LON}),
                x_y_from_centroid=False,
            ),
            Processor(
                filename="plant_village_kenya",
                clean_df=clean_pv_kenya,
                crop_prob=1.0,
                plant_date_col="planting_d",
                harvest_date_col="harvest_da",
                transform_crs_from=32636,
            )
        ) +
        tuple([
            Processor(
                filename=f"ref_african_crops_kenya_01_labels_0{i}/labels.geojson",
                clean_df=clean_pv_kenya2,
                crop_prob=1.0,
                plant_date_col="Planting Date",
                harvest_date_col="Estimated Harvest Date",
                transform_crs_from=32636)
            for i in [0, 1, 2]
        ]),
    ),
    LabeledDataset(
        dataset="Mali",
        country="Mali",
        sentinel_dataset="earth_engine_mali",
        processors=(
            Processor(filename="mali_noncrop_2019", crop_prob=0.0, end_year=2020),
            Processor(filename="segou_bounds_07212020", crop_prob=1.0, end_year=2019),
            Processor(filename="segou_bounds_07212020", crop_prob=1.0, end_year=2020),
        ),
    ),
    LabeledDataset(
        dataset="Togo",
        country="Togo",
        sentinel_dataset="earth_engine_togo",
        processors=(
            Processor(
                filename="crop_merged_v2",
                crop_prob=1.0,
                train_val_test=(0.8, 0.2, 0.0),
                end_year=2020,
            ),
            Processor(
                filename="noncrop_merged_v2",
                crop_prob=0.0,
                train_val_test=(0.8, 0.2, 0.0),
                end_year=2020,
            ),
            Processor(
                filename="random_sample_hrk",
                crop_prob=lambda df: df["hrk-label"],
                transform_crs_from=32631,
                train_val_test=(0.0, 0.0, 1.0),
                end_year=2020,
            ),
            Processor(
                filename="random_sample_cn",
                crop_prob=lambda df: df["cn_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                end_year=2020,
            ),
            Processor(
                filename="BB_random_sample_1k",
                crop_prob=lambda df: df["bb_label"],
                train_val_test=(0.0, 0.0, 1.0),
                end_year=2020,
            ),
            Processor(
                filename="random_sample_bm",
                crop_prob=lambda df: df["bm_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                end_year=2020,
            ),
        ),
    ),
]

unlabeled_datasets = [
    UnlabeledDataset(
        sentinel_dataset="Kenya",
        region_bbox=BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Busia",
        region_bbox=BoundingBox(
            min_lon=33.88389587402344,
            min_lat=-0.04119872691853491,
            max_lon=34.44007873535156,
            max_lat=0.7779454563313616,
        ),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="MaliSegou",
        region_bbox=BoundingBox(
            min_lon=-7.266759, max_lon=-5.511693, min_lat=12.702882, max_lat=13.937639
        ),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="NorthMalawi",
        region_bbox=BoundingBox(min_lon=32.688, max_lon=35.772, min_lat=-14.636, max_lat=-9.231),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="SouthMalawi",
        region_bbox=BoundingBox(min_lon=34.211, max_lon=35.772, min_lat=-17.07, max_lat=-14.636),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Rwanda",
        region_bbox=BoundingBox(min_lon=28.841, max_lon=30.909, min_lat=-2.854, max_lat=-1.034),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="RwandaSake",
        region_bbox=BoundingBox(min_lon=30.377, max_lon=30.404, min_lat=-2.251, max_lat=-2.226),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Togo",
        region_bbox=BoundingBox(
            min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625
        ),
        season=Season.post_season,
    ),
]
