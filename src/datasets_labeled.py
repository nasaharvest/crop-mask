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


def clean_geowiki(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("location_id").mean()
    df = df.rename(
        {"sumcrop": "mean_sumcrop"},
        axis="columns",
        errors="raise",
    )
    return df.reset_index()


def clean_one_acre_fund(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        df[LON].notnull()
        & df[LAT].notnull()
        & df["harvesting_date"].notnull()
        & df["planting_date"].notnull()
    ].copy()
    return df


def clean_ceo_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[LON].notnull() & df[LAT].notnull() & (df["flagged"] == False)].copy()  # noqa E712
    # CEO data may have duplicate samples labeled by the same person
    df = df[~df[[LAT, LON, "email"]].duplicated(keep="first")]
    return df


def add_fake_harvest_date(df: pd.DataFrame) -> pd.DataFrame:
    df["end"] = pd.to_datetime(df["start"]) + pd.to_timedelta(timedelta(days=365))
    return df


labeled_datasets = [
    LabeledDataset(
        dataset="geowiki_landcover_2017",
        country="global",
        processors=(
            Processor(
                filename="loc_all_2.txt",
                clean_df=clean_geowiki,
                longitude_col="loc_cent_X",
                latitude_col="loc_cent_Y",
                crop_prob=lambda df: df.mean_sumcrop / 100,
                end_year=2018,
                end_month_day=(3, 28),
                custom_start_date=date(2017, 3, 28),
                x_y_from_centroid=False,
                train_val_test=(1.0, 0.0, 0.0),
            ),
        ),
    ),
    LabeledDataset(
        dataset="Kenya",
        country="Kenya",
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
                longitude_col="Lon",
                latitude_col="Lat",
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
                    longitude_col="Longitude",
                    latitude_col="Latitude",
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
        processors=tuple(
            [
                Processor(
                    filename=filename,
                    crop_prob=lambda df: df["Crop/ or not"] == "Cropland",
                    x_y_from_centroid=False,
                    train_val_test=(
                        0.1,
                        0.45,
                        0.45,
                    ),  # this makes about 525 for validation and test
                    end_year=2020,
                )
                for filename in [
                    "ceo-2019-Rwanda-Cropland-(RCMRD-Set-1)-sample-data-2021-04-20.csv",
                    "ceo-2019-Rwanda-Cropland-(RCMRD-Set-2)-sample-data-2021-04-20.csv",
                    "ceo-2019-Rwanda-Cropland-sample-data-2021-04-20.csv",
                ]
            ]
            + [
                Processor(
                    filename="Rwanda-non-crop-corrective-v1",
                    crop_prob=0.0,
                    end_year=2020,
                    train_val_test=(1.0, 0, 0),
                ),
                Processor(
                    filename="Rwanda-crop-corrective-v1",
                    crop_prob=1.0,
                    end_year=2020,
                    train_val_test=(1.0, 0, 0),
                ),
            ]
        ),
    ),
    LabeledDataset(
        dataset="Uganda",
        country="Uganda",
        processors=tuple(
            [
                Processor(
                    filename=filename,
                    crop_prob=lambda df: df["Crop/non-crop"] == "Cropland",
                    x_y_from_centroid=False,
                    train_val_test=(
                        0.10,
                        0.45,
                        0.45,
                    ),  # this makes about 525 for validation and test
                    end_year=2020,
                )
                for filename in [
                    "ceo-2019-Uganda-Cropland-(RCMRD--Set-1)-sample-data-2021-06-11.csv",
                    "ceo-2019-Uganda-Cropland-(RCMRD--Set-2)-sample-data-2021-06-11.csv",
                ]
            ]
            + [
                Processor(
                    filename=filename,
                    crop_prob=0.0,
                    sample_from_polygon=True,
                    x_y_from_centroid=True,
                    train_val_test=(1.0, 0, 0),
                    end_year=2021,
                )
                for filename in [
                    "WDPA_WDOECM_Aug2021_Public_UGA_shp_0.zip",
                    "WDPA_WDOECM_Aug2021_Public_UGA_shp_1.zip",
                    "WDPA_WDOECM_Aug2021_Public_UGA_shp_2.zip",
                ]
            ]
            + [
                Processor(
                    filename="ug_in_season_monitoring_2021_08_11_17_50_46_428737.csv",
                    crop_prob=1.0,
                    longitude_col="location/_gps_longitude",
                    latitude_col="location/_gps_latitude",
                    clean_df=add_fake_harvest_date,
                    plant_date_col="start",
                    harvest_date_col="end",
                    train_val_test=(1.0, 0.0, 0.0),
                    x_y_from_centroid=False,
                ),
                Processor(
                    filename="ug_end_of_season_assessment_2021_08_11_17_47_53_813908.csv",
                    crop_prob=1.0,
                    longitude_col="district_selection/_gps_location_longitude",
                    latitude_col="district_selection/_gps_location_latitude",
                    clean_df=add_fake_harvest_date,
                    plant_date_col="start",
                    harvest_date_col="end",
                    train_val_test=(1.0, 0.0, 0.0),
                    x_y_from_centroid=False,
                ),
                Processor(
                    filename="ug_pre_season_assessment_2021_08_11_18_15_27_323695.csv",
                    crop_prob=1.0,
                    longitude_col="location/_gps_location_longitude",
                    latitude_col="location/_gps_location_latitude",
                    clean_df=add_fake_harvest_date,
                    plant_date_col="start",
                    harvest_date_col="end",
                    train_val_test=(1.0, 0.0, 0.0),
                    x_y_from_centroid=False,
                ),
            ]
        ),
    ),
    LabeledDataset(
        dataset="one_acre_fund",
        country="Kenya,Rwanda,Tanzania",
        processors=(
            Processor(
                filename="One_Acre_Fund_KE_RW_TZ_2016_17_18_19_MEL_agronomic_survey_data.csv",
                crop_prob=1.0,
                clean_df=clean_one_acre_fund,
                longitude_col="site_longitude",
                latitude_col="site_latitude",
                harvest_date_col="harvesting_date",
                plant_date_col="planting_date",
                x_y_from_centroid=False,
                train_val_test=(1.0, 0.0, 0.0),
            ),
        ),
    ),
    LabeledDataset(
        dataset="open_buildings",
        country="global",
        processors=(
            Processor(
                filename="177_buildings_confidence_0.9.csv",
                latitude_col="latitude",
                longitude_col="longitude",
                crop_prob=0.0,
                end_year=2021,
                x_y_from_centroid=False,
                train_val_test=(1.0, 0.0, 0.0),
            ),
        ),
    ),
    LabeledDataset(
        dataset="digitalearthafrica_eastern",
        country="global",
        processors=(
            Processor(
                filename="Eastern_training_data_20210427.geojson",
                crop_prob=lambda df: df["Class"].astype(float),
                end_year=2021,
                train_val_test=(1.0, 0.0, 0.0),
            ),
        ),
    ),
    LabeledDataset(
        dataset="digitalearthafrica_sahel",
        country="global",
        processors=tuple(
            [
                Processor(
                    filename="ceo_td_polys.geojson",
                    crop_prob=lambda df: df["Class"].astype(float),
                    end_year=2020,
                    train_val_test=(1.0, 0.0, 0.0),
                ),
                Processor(
                    filename="validation_samples.shp",
                    crop_prob=lambda df: (df["Class"] == "crop").astype(float),
                    end_year=2020,
                    train_val_test=(1.0, 0.0, 0.0),
                ),
                Processor(
                    filename="Sahel_region_RE_sample.geojson",
                    crop_prob=lambda df: (df["Class"] == "crop").astype(float),
                    end_year=2020,
                    train_val_test=(1.0, 0.0, 0.0),
                ),
            ]
        ),
    ),
    LabeledDataset(
        dataset="Ethiopia",
        country="Ethiopia",
        processors=tuple(
            [
                Processor(
                    filename=f"tigray/{filename}.shp",
                    crop_prob=1.0,
                    end_year=2020,
                    train_val_test=(1.0, 0.0, 0.0),
                )
                for filename in ["tigrayWW_crop", "tigrayWW_crop2"]
            ]
        )
        + tuple(
            [
                Processor(
                    filename=f"tigray/{filename}.shp",
                    crop_prob=0.0,
                    end_year=2020,
                    train_val_test=(1.0, 0.0, 0.0),
                )
                for filename in [
                    "tigrayWW_forest",
                    "tigrayWW_forest2",
                    "tigrayWW_shrub",
                    "tigrayWW_shrub2",
                    "tigrayWW_sparse",
                    "tigrayWW_sparse2",
                ]
            ]
        )
        + tuple(
            [
                Processor(
                    filename=f"tigray_non_fallow_crop/nonFallowCrop{year}.shp",
                    crop_prob=1.0,
                    end_year=year + 1,
                    train_val_test=(1.0, 0.0, 0.0),
                )
                for year in [2019, 2020]
            ]
        )
        + tuple(
            [
                # Processor(
                #     filename=f"tigray_corrective_2020/non_crop.shp",
                #     crop_prob=0.0,
                #     end_year=2021,
                #     train_val_test=(1.0, 0.0, 0.0),
                # ),
                # Processor(
                #     filename=f"tigray_corrective_2020/crop.shp",
                #     crop_prob=1.0,
                #     end_year=2021,
                #     train_val_test=(1.0, 0.0, 0.0),
                # ),
                # Processor(
                #     filename=f"tigray_corrective_2021/non_crop.shp",
                #     crop_prob=0.0,
                #     custom_start_date=date(2021, 4, 21),
                #     num_timesteps=6,
                #     train_val_test=(1.0, 0.0, 0.0),
                # ),
                # Processor(
                #     filename=f"tigray_corrective_2021/crop.shp",
                #     crop_prob=1.0,
                #     custom_start_date=date(2021, 4, 21),
                #     num_timesteps=6,
                #     train_val_test=(1.0, 0.0, 0.0),
                # ),
            ]
        ),
    ),
    LabeledDataset(
        dataset="Ethiopia_Tigray_2020",
        country="Ethiopia",
        processors=tuple(
            [
                Processor(
                    filename="ceo-2020-Ethiopia-Tigray-(Set-1)-sample-data-2021-10-04.csv",
                    crop_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                    end_year=2021,
                    x_y_from_centroid=False,
                    train_val_test=(0.0, 0.5, 0.5),
                    clean_df=clean_ceo_data,
                ),
                Processor(
                    filename="ceo-2020-Ethiopia-Tigray-(Set-2)-sample-data-2021-10-04.csv",
                    crop_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                    end_year=2021,
                    x_y_from_centroid=False,
                    train_val_test=(0.0, 0.5, 0.5),
                    clean_df=clean_ceo_data,
                ),
            ]
        ),
    ),
    LabeledDataset(
        dataset="Ethiopia_Tigray_2021",
        country="Ethiopia",
        processors=tuple(
            [
                Processor(
                    filename="ceo-2021-Ethiopia-Tigray-(Set-1-Fixed)-sample-data-2021-10-04.csv",
                    crop_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                    custom_start_date=date(2021, 4, 21),
                    x_y_from_centroid=False,
                    train_val_test=(0.0, 1.0, 0.0),
                    clean_df=clean_ceo_data,
                    num_timesteps=6,
                ),
                Processor(
                    filename="ceo-2021-Ethiopia-Tigray-(Set-2-Fixed)-sample-data-2021-10-04.csv",
                    crop_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                    custom_start_date=date(2021, 4, 21),
                    x_y_from_centroid=False,
                    train_val_test=(0.0, 1.0, 0.0),
                    clean_df=clean_ceo_data,
                    num_timesteps=6,
                ),
            ]
        ),
    ),
]
