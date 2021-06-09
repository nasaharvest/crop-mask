import pandas as pd
from datetime import date, timedelta

from src.ETL.dataset import LabeledDataset
from src.ETL.label_downloader import RawLabels
from src.ETL.processor import Processor
from src.ETL.constants import LON, LAT


def clean_pv_kenya(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["harvest_da"] != "nan") & (df["harvest_da"] != "unknown")].copy()
    df.loc[:, "planting_d"] = pd.to_datetime(df["planting_d"])
    df.loc[:, "harvest_da"] = pd.to_datetime(df["harvest_da"])

    df["between_days"] = (df["harvest_da"] - df["planting_d"]).dt.days
    year = pd.to_timedelta(timedelta(days=365))
    df.loc[(-365 < df["between_days"]) & (df["between_days"] < 0), "harvest_da"] += year

    valid_years = [2018, 2019, 2020]
    df = df[
        (df["planting_d"].dt.year.isin(valid_years)) & (df["harvest_da"].dt.year.isin(valid_years))
    ].copy()
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
                filename="noncrop_labels_v2",
                crop_prob=0.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
                transform_crs_from=32636,
            ),
            Processor(
                filename="noncrop_labels_set2",
                crop_prob=0.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
                transform_crs_from=32636,
            ),
            Processor(
                filename="2019_gepro_noncrop",
                crop_prob=0.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            Processor(
                filename="noncrop_water_kenya_gt",
                crop_prob=0.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            Processor(
                filename="noncrop_kenya_gt",
                crop_prob=0.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            Processor(
                filename="one_acre_fund_kenya",
                crop_prob=1.0,
                end_year=2020,
                clean_df=lambda df: df.rename(columns={"Lat": LAT, "Lon": LON}),
                train_val_test=(0.8, 0.1, 0.1),
                x_y_from_centroid=False,
            ),
            Processor(
                filename="plant_village_kenya",
                clean_df=clean_pv_kenya,
                crop_prob=1.0,
                plant_date_col="planting_d",
                harvest_date_col="harvest_da",
                train_val_test=(0.8, 0.1, 0.1),
                transform_crs_from=32636,
            ),
        )
        + tuple(
            [
                Processor(
                    filename=f"ref_african_crops_kenya_01_labels_0{i}/labels.geojson",
                    clean_df=clean_pv_kenya2,
                    crop_prob=1.0,
                    plant_date_col="Planting Date",
                    harvest_date_col="Estimated Harvest Date",
                    train_val_test=(0.8, 0.1, 0.1),
                    transform_crs_from=32636,
                )
                for i in [0, 1, 2]
            ]
        ),
    ),
    LabeledDataset(
        dataset="Mali",
        country="Mali",
        sentinel_dataset="earth_engine_mali",
        processors=(
            Processor(
                filename="mali_noncrop_2019",
                crop_prob=0.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            Processor(
                filename="segou_bounds_07212020",
                crop_prob=1.0,
                end_year=2019,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            Processor(
                filename="segou_bounds_07212020",
                crop_prob=1.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            Processor(
                filename="sikasso_clean_fields",
                crop_prob=1.0,
                end_year=2020,
                train_val_test=(0.8, 0.1, 0.1),
            ),
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
    LabeledDataset(
        dataset="Rwanda",
        country="Rwanda",
        sentinel_dataset="earth_engine_rwanda",
        processors=tuple(
            [
                Processor(
                    filename=filename,
                    crop_prob=lambda df: df['Crop/ or not'] == 'Cropland',
                    x_y_from_centroid=False,
                    train_val_test=(0.8, 0.1, 0.1),
                    end_year=2020,
                )
                for filename in [
                    "ceo-2019-Rwanda-Cropland-(RCMRD-Set-1)-sample-data-2021-04-20.csv",
                    "ceo-2019-Rwanda-Cropland-(RCMRD-Set-2)-sample-data-2021-04-20.csv",
                    "ceo-2019-Rwanda-Cropland-sample-data-2021-04-20.csv"
                ]
            ] + [
                Processor(
                    filename="Rwanda-non-crop-corrective-v1",
                    crop_prob=0.0,
                    end_year=2020,
                    train_val_test=(1.0, 0, 0)
                ),
                Processor(
                    filename="Rwanda-crop-corrective-v1",
                    crop_prob=1.0,
                    end_year=2020,
                    train_val_test=(1.0, 0, 0)
                )
            ]
        )
    )
]
