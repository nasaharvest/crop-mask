import pandas as pd
from datetime import timedelta

from openmapflow.labeled_dataset import CustomLabeledDataset, create_datasets
from openmapflow.raw_labels import RawLabels
from openmapflow.constants import LON, LAT


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


datasets = [
    CustomLabeledDataset(
        dataset="geowiki_landcover_2017",
        country="global",
        raw_labels=(
            RawLabels(
                filename="loc_all_2.txt",
                longitude_col="loc_cent_X",
                latitude_col="loc_cent_Y",
                class_prob=lambda df: df.sumcrop / 100,
                start_year=2017,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Kenya",
        country="Kenya",
        raw_labels=(
            RawLabels(
                filename="noncrop_labels_v2",
                class_prob=0.0,
                start_year=2019,
                train_val_test=(0.8, 0.1, 0.1),
                transform_crs_from=32636,
            ),
            RawLabels(
                filename="noncrop_labels_set2",
                class_prob=0.0,
                start_year=2019,
                train_val_test=(0.8, 0.1, 0.1),
                transform_crs_from=32636,
            ),
            RawLabels(
                filename="2019_gepro_noncrop",
                class_prob=0.0,
                start_year=2019,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            RawLabels(
                filename="noncrop_water_kenya_gt",
                class_prob=0.0,
                start_year=2019,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            RawLabels(
                filename="noncrop_kenya_gt",
                class_prob=0.0,
                start_year=2019,
                train_val_test=(0.8, 0.1, 0.1),
            ),
            RawLabels(
                filename="one_acre_fund_kenya",
                class_prob=1.0,
                start_year=2019,
                longitude_col="Lon",
                latitude_col="Lat",
                train_val_test=(0.8, 0.1, 0.1),
            ),
            RawLabels(
                filename="plant_village_kenya",
                filter_df=clean_pv_kenya,
                class_prob=1.0,
                start_date_col="planting_d",
                train_val_test=(0.8, 0.1, 0.1),
                transform_crs_from=32636,
            ),
        )
        + tuple(
            [
                RawLabels(
                    filename=f"ref_african_crops_kenya_01_labels_0{i}/labels.geojson",
                    longitude_col="Longitude",
                    latitude_col="Latitude",
                    class_prob=1.0,
                    start_date_col="Planting Date",
                    train_val_test=(0.8, 0.1, 0.1),
                    transform_crs_from=32636,
                )
                for i in [0, 1, 2]
            ]
        ),
    ),
    CustomLabeledDataset(
        dataset="Mali",
        country="Mali",
        raw_labels=(
            RawLabels(filename="mali_noncrop_2019", class_prob=0.0, start_year=2019),
            RawLabels(filename="segou_bounds_07212020", class_prob=1.0, start_year=2018),
            RawLabels(filename="segou_bounds_07212020", class_prob=1.0, start_year=2019),
            RawLabels(filename="sikasso_clean_fields", class_prob=1.0, start_year=2019),
        ),
    ),
    CustomLabeledDataset(
        dataset="Mali_lower_CEO_2019",
        country="Mali",
        raw_labels=(
            RawLabels(
                filename="ceo-2019-Mali-USAID-ZOIS-lower-(Set-1)--sample-data-2021-11-29.csv",
                class_prob=lambda df: (
                    df["Does this point lie on a crop or non-crop pixel?"] == "Crop"
                ),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
            ),
            RawLabels(
                filename="ceo-2019-Mali-USAID-ZOIS-lower-(Set-2)--sample-data-2021-11-29.csv",
                class_prob=lambda df: (
                    df["Does this point lie on a crop or non-crop pixel?"] == "Crop"
                ),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Mali_upper_CEO_2019",
        country="Mali",
        raw_labels=(
            RawLabels(
                filename="ceo-2019-Mali-USAID-ZOIS-upper-(Set-1)-sample-data-2021-11-25.csv",
                class_prob=lambda df: (
                    df["Does this point lie on a crop or non-crop pixel?"] == "Crop"
                ),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
            ),
            RawLabels(
                filename="ceo-2019-Mali-USAID-ZOIS-upper-(Set-2)-sample-data-2021-11-25.csv",
                class_prob=lambda df: (
                    df["Does this point lie on a crop or non-crop pixel?"] == "Crop"
                ),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Togo",
        country="Togo",
        raw_labels=(
            RawLabels(
                filename="crop_merged_v2.zip",
                class_prob=1.0,
                train_val_test=(0.8, 0.2, 0.0),
                start_year=2019,
            ),
            RawLabels(
                filename="noncrop_merged_v2.zip",
                class_prob=0.0,
                train_val_test=(0.8, 0.2, 0.0),
                start_year=2019,
            ),
            RawLabels(
                filename="random_sample_hrk.zip",
                class_prob=lambda df: df["hrk-label"],
                transform_crs_from=32631,
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            RawLabels(
                filename="random_sample_cn.zip",
                class_prob=lambda df: df["cn_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            RawLabels(
                filename="BB_random_sample_1k.zip",
                class_prob=lambda df: df["bb_label"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            RawLabels(
                filename="random_sample_bm.zip",
                class_prob=lambda df: df["bm_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Rwanda",
        country="Rwanda",
        raw_labels=tuple(
            [
                RawLabels(
                    filename=filename,
                    class_prob=lambda df: df["Crop/ or not"] == "Cropland",
                    latitude_col="lat",
                    longitude_col="lon",
                    train_val_test=(
                        0.1,
                        0.45,
                        0.45,
                    ),  # this makes about 525 for validation and test
                    start_year=2019,
                )
                for filename in [
                    "ceo-2019-Rwanda-Cropland-(RCMRD-Set-1)-sample-data-2021-04-20.csv",
                    "ceo-2019-Rwanda-Cropland-(RCMRD-Set-2)-sample-data-2021-04-20.csv",
                    "ceo-2019-Rwanda-Cropland-sample-data-2021-04-20.csv",
                ]
            ]
            + [
                RawLabels(
                    filename="Rwanda-non-crop-corrective-v1", class_prob=0.0, start_year=2019
                ),
                RawLabels(filename="Rwanda-crop-corrective-v1", class_prob=1.0, start_year=2019),
            ]
        ),
    ),
    CustomLabeledDataset(
        dataset="Uganda",
        country="Uganda",
        raw_labels=tuple(
            [
                RawLabels(
                    filename=filename,
                    class_prob=lambda df: df["Crop/non-crop"] == "Cropland",
                    latitude_col="lat",
                    longitude_col="lon",
                    train_val_test=(
                        0.10,
                        0.45,
                        0.45,
                    ),  # this makes about 525 for validation and test
                    start_year=2019,
                )
                for filename in [
                    "ceo-2019-Uganda-Cropland-(RCMRD--Set-1)-sample-data-2021-06-11.csv",
                    "ceo-2019-Uganda-Cropland-(RCMRD--Set-2)-sample-data-2021-06-11.csv",
                ]
            ]
            + [
                RawLabels(
                    filename=filename,
                    class_prob=0.0,
                    sample_from_polygon=True,
                    x_y_from_centroid=True,
                    start_year=2020,
                )
                for filename in [
                    "WDPA_WDOECM_Aug2021_Public_UGA_shp_0.zip",
                    "WDPA_WDOECM_Aug2021_Public_UGA_shp_1.zip",
                    "WDPA_WDOECM_Aug2021_Public_UGA_shp_2.zip",
                ]
            ]
            + [
                RawLabels(
                    filename="ug_in_season_monitoring_2021_08_11_17_50_46_428737.csv",
                    class_prob=1.0,
                    longitude_col="location/_gps_longitude",
                    latitude_col="location/_gps_latitude",
                    start_date_col="start",
                ),
                RawLabels(
                    filename="ug_end_of_season_assessment_2021_08_11_17_47_53_813908.csv",
                    class_prob=1.0,
                    longitude_col="district_selection/_gps_location_longitude",
                    latitude_col="district_selection/_gps_location_latitude",
                    start_date_col="start",
                ),
                RawLabels(
                    filename="ug_pre_season_assessment_2021_08_11_18_15_27_323695.csv",
                    class_prob=1.0,
                    longitude_col="location/_gps_location_longitude",
                    latitude_col="location/_gps_location_latitude",
                    start_date_col="start",
                ),
            ]
        ),
    ),
    # CustomLabeledDataset(
    #     dataset="one_acre_fund",
    #     country="Kenya,Rwanda,Tanzania",
    #     raw_labels=(
    #         RawLabels(
    #             filename="One_Acre_Fund_KE_RW_TZ_2016_17_18_19_MEL_agronomic_survey_data.csv",
    #             class_prob=1.0,
    #             filter_df=clean_one_acre_fund,
    #             longitude_col="site_longitude",
    #             latitude_col="site_latitude",
    #             start_date_col="planting_date",
    #         ),
    #     ),
    # ),
    CustomLabeledDataset(
        dataset="open_buildings",
        country="global",
        raw_labels=(
            RawLabels(
                filename="177_buildings_confidence_0.9.csv",
                latitude_col="latitude",
                longitude_col="longitude",
                class_prob=0.0,
                start_year=2020,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="digitalearthafrica_eastern",
        country="global",
        raw_labels=(
            RawLabels(
                filename="Eastern_training_data_20210427.geojson",
                class_prob=lambda df: df["Class"].astype(float),
                start_year=2020,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="digitalearthafrica_sahel",
        country="global",
        raw_labels=(
            RawLabels(
                filename="ceo_td_polys.geojson",
                class_prob=lambda df: df["Class"].astype(float),
                start_year=2019,
            ),
            RawLabels(
                filename="validation_samples.shp",
                class_prob=lambda df: (df["Class"] == "crop").astype(float),
                start_year=2019,
            ),
            RawLabels(
                filename="Sahel_region_RE_sample.geojson",
                class_prob=lambda df: (df["Class"] == "crop").astype(float),
                start_year=2019,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Ethiopia",
        country="Ethiopia",
        raw_labels=(
            RawLabels(filename="tigray/tigrayWW_crop.shp", class_prob=1.0, start_year=2019),
            RawLabels(filename="tigray/tigrayWW_crop2.shp", class_prob=1.0, start_year=2019),
            RawLabels(filename="tigray/tigrayWW_forest.shp", class_prob=0.0, start_year=2019),
            RawLabels(filename="tigray/tigrayWW_forest2.shp", class_prob=0.0, start_year=2019),
            RawLabels(filename="tigray/tigrayWW_shrub.shp", class_prob=0.0, start_year=2019),
            RawLabels(filename="tigray/tigrayWW_shrub2.shp", class_prob=0.0, start_year=2019),
            RawLabels(filename="tigray/tigrayWW_sparse.shp", class_prob=0.0, start_year=2019),
            RawLabels(filename="tigray/tigrayWW_sparse2.shp", class_prob=0.0, start_year=2019),
            RawLabels(
                filename="tigray_non_fallow_crop/nonFallowCrop2019.shp",
                class_prob=1.0,
                start_year=2019,
            ),
            RawLabels(
                filename="tigray_non_fallow_crop/nonFallowCrop2020.shp",
                class_prob=1.0,
                start_year=2020,
            ),
            RawLabels(
                filename="tigray_corrective_2020/non_crop.shp", class_prob=0.0, start_year=2020
            ),
            RawLabels(filename="tigray_corrective_2020/crop.shp", class_prob=1.0, start_year=2020),
            RawLabels(
                filename="tigray_corrective_2021/non_crop.shp",
                class_prob=0.0,
                start_year=2021,
            ),
            RawLabels(
                filename="tigray_corrective_2021/crop.shp",
                class_prob=1.0,
                start_year=2021,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Ethiopia_Tigray_2020",
        country="Ethiopia",
        raw_labels=(
            RawLabels(
                filename="ceo-2020-Ethiopia-Tigray-(Set-1)-sample-data-2021-10-04.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
                labeler_name="email",
                label_duration="analysis_duration",
            ),
            RawLabels(
                filename="ceo-2020-Ethiopia-Tigray-(Set-2)-sample-data-2021-10-04.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
                labeler_name="email",
                label_duration="analysis_duration",
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Ethiopia_Tigray_2021",
        country="Ethiopia",
        raw_labels=(
            RawLabels(
                filename="ceo-2021-Ethiopia-Tigray-(Set-1-Fixed)-sample-data-2022-02-24.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2021,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
                labeler_name="email",
                label_duration="analysis_duration",
            ),
            RawLabels(
                filename="ceo-2021-Ethiopia-Tigray-(Set-2-Fixed)-sample-data-2022-02-24.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2021,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
                labeler_name="email",
                label_duration="analysis_duration",
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Ethiopia_Bure_Jimma_2019",
        country="Ethiopia",
        raw_labels=(
            RawLabels(
                filename="ceo-2019-Ethiopia---Bure-Jimma-(Set-1)-sample-data-2021-11-24.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-2019-Ethiopia---Bure-Jimma-(Set-2)-sample-data-2021-11-24.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Ethiopia_Bure_Jimma_2020",
        country="Ethiopia",
        raw_labels=(
            RawLabels(
                filename="ceo-2020-Ethiopia---Bure-Jimma-(Set-1)-sample-data-2021-11-24.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-2020-Ethiopia---Bure-Jimma-(Set-2)-sample-data-2021-11-24.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Argentina_Buenos_Aires",
        country="Argentina",
        raw_labels=(
            RawLabels(
                filename="bc_mapeo_del_cultivo_0.csv",
                filter_df=lambda df: df[
                    (
                        df["Seleccione el cultivo principal en el lote:"].notnull()
                        & ~df["Seleccione el cultivo principal en el lote:"].isin(
                            ["otro", "barbecho", "sin_dato"]
                        )
                    )
                ].copy(),
                longitude_col="longitud",
                latitude_col="latitud",
                class_prob=lambda df: df["Seleccione el cultivo principal en el lote:"].isin(
                    ["trigo_o_cebada", "cultive_leguminosa", "maiz", "sorgo", "soja", "girasol"]
                ),
                train_val_test=(0.8, 0.2, 0.0),
                start_year=2021,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Malawi_CEO_2020",
        country="Malawi",
        raw_labels=(
            RawLabels(
                filename="ceo-Sep-2020-Sep-2021-Malawi-(Set-1)-sample-data-2021-12-09.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-Sep-2020-Sep-2021-Malawi-(Set-2)-sample-data-2021-12-09.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                latitude_col="lat",
                longitude_col="lon",
                train_val_test=(0.0, 0.5, 0.5),
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Malawi_CEO_2019",
        country="Malawi",
        raw_labels=(
            RawLabels(
                filename="ceo-2019-Malawi-Cropland-(RCMRD-Set-1)-sample-data-2021-12-10.csv",
                class_prob=lambda df: (df["Crop/non"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-2019-Malawi-Cropland-(RCMRD-Set-2)-sample-data-2021-12-10.csv",
                class_prob=lambda df: (df["Crop/non"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Malawi_FAO",
        country="Malawi",
        raw_labels=(RawLabels(filename="malawi_fao.geojson", class_prob=1.0, start_year=2020),),
    ),
    CustomLabeledDataset(
        dataset="Malawi_FAO_corrected",
        country="Malawi",
        raw_labels=(
            RawLabels(filename="MWI_MWI_LC_FO_202106.shp", class_prob=1.0, start_year=2020),
        ),
    ),
    CustomLabeledDataset(
        dataset="Zambia_CEO_2019",
        country="Zambia",
        raw_labels=(
            RawLabels(
                filename="ceo-2019-Zambia-Cropland-(RCMRD-Set-1)-sample-data-2021-12-12.csv",
                class_prob=lambda df: (df["Crop/non-crop"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-2019-Zambia-Cropland-(RCMRD-Set-2)-sample-data-2021-12-12.csv",
                class_prob=lambda df: (df["Crop/non-crop"] == "Crop"),
                start_year=2019,
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Tanzania_CEO_2019",
        country="Tanzania",
        raw_labels=(
            RawLabels(
                filename="ceo-2019-Tanzania-Cropland-(RCMRD-Set-1)-sample-data-2021-12-13.csv",
                class_prob=lambda df: (df["Crop/non-Crop"] == "Cropland"),
                start_year=2019,
                train_val_test=(0.0, 0.5, 0.5),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-2019-Tanzania-Cropland-(RCMRD-Set-2)-sample-data-2021-12-13.csv",
                class_prob=lambda df: (df["Crop/non-Crop"] == "Cropland"),
                start_year=2019,
                train_val_test=(0.0, 0.5, 0.5),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Malawi_corrected",
        country="Malawi",
        raw_labels=(
            RawLabels(
                filename="Crops.shp",
                class_prob=1.0,
                start_year=2020,
            ),
            RawLabels(
                filename="Noncrops.shp",
                class_prob=0.0,
                start_year=2020,
            ),
            RawLabels(
                filename="Major_protected_areas.shp",
                class_prob=0.0,
                start_year=2020,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Namibia_CEO_2020",
        country="Namibia",
        raw_labels=(
            RawLabels(
                filename="ceo-Namibia-North-Jan-2020---Dec-2020-(Set-1)-sample-data-2022-04-20.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                train_val_test=(0.0, 0.5, 0.5),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-Namibia-North-Jan-2020---Dec-2020-(Set-2)-sample-data-2022-04-20.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                train_val_test=(0.0, 0.5, 0.5),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
        ),
    ),
]

if __name__ == "__main__":
    create_datasets(datasets)