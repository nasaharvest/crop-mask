from datetime import date, timedelta
from typing import List

import pandas as pd
from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import CLASS_PROB, END, LAT, LON, START, SUBSET
from openmapflow.label_utils import (
    get_lat_lon_from_centroid,
    read_zip,
    train_val_test_split,
)
from openmapflow.labeled_dataset import LabeledDataset, create_datasets

from src.labeled_dataset_custom import CustomLabeledDataset
from src.raw_labels import RawLabels

raw_dir = PROJECT_ROOT / DataPaths.RAW_LABELS


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


def join_unique(values):
    return ",".join([str(i) for i in values.unique()])


class HawaiiAgriculturalLandUse2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        df = read_zip(raw_dir / "Hawaii_Agricultural_Land_Use_-_2020_Update.zip")
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[LAT], df[LON] = get_lat_lon_from_centroid(df.geometry)
        df[SUBSET] = "training"
        df[CLASS_PROB] = 1.0
        df = df.drop_duplicates(subset=[LAT, LON])
        return df


class KenyaCEO2019(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        df1 = pd.read_csv(
            raw_dir / "ceo-Kenya-Feb-2019---Feb-2020-(Set-1)-sample-data-2022-12-05.csv"
        )
        df2 = pd.read_csv(
            raw_dir / "ceo-Kenya-Feb-2019---Feb-2020-(Set-2)-sample-data-2022-12-05.csv"
        )
        df = pd.concat([df1, df2])
        df[CLASS_PROB] = df["Does this pixel contain active cropland?"] == "Crop"
        df[CLASS_PROB] = df[CLASS_PROB].astype(int)

        df["num_labelers"] = 1
        df = df.groupby([LON, LAT], as_index=False, sort=False).agg(
            {
                CLASS_PROB: "mean",
                "num_labelers": "sum",
                "plotid": join_unique,
                "sampleid": join_unique,
                "email": join_unique,
            }
        )
        df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.5, 0.5)
        return df


class HawaiiCorrective2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        hawaii_dir = raw_dir / "Hawaii_corrective_2020"
        df1 = pd.read_csv(hawaii_dir / "corrective_Nakalembe.csv")
        df2 = pd.read_csv(hawaii_dir / "corrective-Asare-Ansah.csv")
        df3 = pd.read_csv(hawaii_dir / "corrective-devereux.csv")
        df4 = pd.read_csv(hawaii_dir / "corrective-kerner.csv")
        df5 = pd.read_csv(hawaii_dir / "corrective-zvonkov.csv")
        df = pd.concat([df1, df2, df3, df4, df5])
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df[CLASS_PROB] = (df["Wrong value"] == 0).astype(int)
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = "training"
        return df


class HawaiiCorrectiveGuided2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        hawaii_dir = raw_dir / "Hawaii_corrective_2020"
        df1 = pd.read_csv(hawaii_dir / "corrective_guided_Nakalembe.csv")
        df2 = pd.read_csv(hawaii_dir / "corrective-guided-Asare-Ansah.csv")
        df3 = pd.read_csv(hawaii_dir / "corrective-guided-kerner.csv")
        df4 = pd.read_csv(hawaii_dir / "corrective-guided-zvonkov.csv")
        df5 = pd.read_csv(hawaii_dir / "corrective-guided-Satish.csv")
        df = pd.concat([df1, df2, df3, df4, df5])
        df[CLASS_PROB] = (df["Wrong value"] == 0).astype(int)

        # All points in this dataset are non-crop
        df6 = pd.read_csv(hawaii_dir / "corrective-guided-devereux.csv")
        df6[CLASS_PROB] = 0

        # Match length of HawaiiCorrective2020 (329) by dropping points
        # from non-corrective set (351 - 329 = 22)
        df6.drop(df6.tail(22).index, inplace=True)

        df = pd.concat([df, df6])

        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = "training"
        return df


class HawaiiAgriculturalLandUse2020Subset(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        df = HawaiiAgriculturalLandUse2020().load_labels()

        # Match length of HawaiiCorrective2020 by dropping points
        df = df.sample(n=329, random_state=0)
        return df


class MalawiCorrectiveLabels2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        Malawi_dir = raw_dir / "Malawi_corrective_labels_2020"
        df1 = pd.read_csv(Malawi_dir / "Malawi_new points _Chiwalo.csv")
        df2 = pd.read_csv(Malawi_dir / "Malawi_new points _Stephen C.csv")
        df3 = pd.read_csv(Malawi_dir / "Malawi_new points _Stephen Chiwalo_1.csv")
        df4 = pd.read_csv(Malawi_dir / "Malawi_new points _Stephen Chiwalo.csv")
        df5 = pd.read_csv(Malawi_dir / "Malawi_new points _Stephen.csv")
        df6 = pd.read_csv(Malawi_dir / "Malawi_new points charles_1.csv")
        df7 = pd.read_csv(Malawi_dir / "Malawi_new points charles_2.csv")
        df8 = pd.read_csv(Malawi_dir / "Malawi_new points_Blake.csv")
        df9 = pd.read_csv(Malawi_dir / "Malawi_new points_segula_1.csv")
        df10 = pd.read_csv(Malawi_dir / "Malawi_new points_segula.csv") 
        df11 = pd.read_csv(Malawi_dir / "Malawi_new points charles_3.csv") 
        df12 = pd.read_csv(Malawi_dir / "Malawi_new points -Stephen_2.csv") 
        df13 = pd.read_csv(Malawi_dir / "Malawi_new points -Stephen_3.csv") 
        df14 = pd.read_csv(Malawi_dir / "Malawi_new points -Stephen_4.csv") 
        df15 = pd.read_csv(Malawi_dir / "Malawi_new points Stephen_5.csv")
        df16 = pd.read_csv(Malawi_dir / "Malawi_new points sungeni mnensa_1.csv")
        df17 = pd.read_csv(Malawi_dir / "Malawi_new points sungeni Mnensa_2.csv")
        df18 = pd.read_csv(Malawi_dir / "Malawi_new points_Chirwa L.csv")
        df19 = pd.read_csv(Malawi_dir / "Malawi_new points_Powel_Mbilima.csv")
        df20 = pd.read_csv(Malawi_dir / "Malawi_new points_Sungeni Mnensa_3.csv")
        df21 = pd.read_csv(Malawi_dir / "Malawi_new points_Sungeni Mnensa_4.csv")
        df22 = pd.read_csv(Malawi_dir / "Malawi_new points- Richard Kalulu- Ntcheu District.csv")
        df23 = pd.read_csv(Malawi_dir / "Malawi_new points-Benson Chirwa - Lilongwe East (1).csv")
        df24 = pd.read_csv(Malawi_dir / "Malawi_new points-Benson Chirwa - Lilongwe East (2).csv")
        df25 = pd.read_csv(Malawi_dir / "Malawi_new points-Benson Chirwa - Lilongwe East.csv")
        df26 = pd.read_csv(Malawi_dir / "Malawi_new points-Mbewe-Lilongwe West.csv")
        df27 = pd.read_csv(Malawi_dir / "Malawi_new points-Mbewe-Lilongwe West1.csv")
        df28 = pd.read_csv(Malawi_dir / "Malawi_New Points_ Peter Josamu_Dz.csv")
        df29 = pd.read_csv(Malawi_dir / "Malawi_new points - Jane_1.csv")
        df30 = pd.read_csv(Malawi_dir / "Malawi_new points  sungeni mnensa 5.csv")
        df31 = pd.read_csv(Malawi_dir / "Malawi_new points  -Sungeni Mnensa 6.csv")
        df32 = pd.read_csv(Malawi_dir / "Malawi_new points  sungeni mnensa 7.csv")
        df33 = pd.read_csv(Malawi_dir / "Malawi_new points (2) Ndelemani 2.csv")
        df34 = pd.read_csv(Malawi_dir / "Malawi_new points _chimwemwe_1.csv")
        df35 = pd.read_csv(Malawi_dir / "Malawi_new points chimwemwe (2).csv")
        df36 = pd.read_csv(Malawi_dir / "Malawi_new points chimwemwe (3).csv")
        df37 = pd.read_csv(Malawi_dir / "Malawi_new points -Jane_2.csv")
        df38 = pd.read_csv(Malawi_dir / "Malawi_new points Phillimon.csv")
        df39 = pd.read_csv(Malawi_dir / "Malawi_new points sungeni mnensa 8.csv")
        df40 = pd.read_csv(Malawi_dir / "Malawi_new points sungeni mnensa 9.csv")
        df41 = pd.read_csv(Malawi_dir / "Malawi_new points sungeni mnensa 10.csv")
        df42 = pd.read_csv(Malawi_dir / "Malawi_new points Winston.csv")
        df43 = pd.read_csv(Malawi_dir / "Malawi_new points_ chimwemwe 4.csv")
        df44 = pd.read_csv(Malawi_dir / "Malawi_new points_Benjamin.csv")
        df45 = pd.read_csv(Malawi_dir / "Malawi_new points_Lupakisyo.csv")
        df46 = pd.read_csv(Malawi_dir / "Malawi_new points_Lupakisyo.csv.csv")
        df47 = pd.read_csv(Malawi_dir / "Malawi_new points_Sekani.csv")
        df48 = pd.read_csv(Malawi_dir / "Malawi_new points_Twaibu Nathaniel Nn 2.csv")
        df49 = pd.read_csv(Malawi_dir / "Malawi_new points-Jane_3.csv")
        df50 = pd.read_csv(Malawi_dir / "Malawi_new points-Owen Mlima.csv")
        df51 = pd.read_csv(Malawi_dir / "Malawi_new points-Owen.csv")
        df52 = pd.read_csv(Malawi_dir / "Malawi_new pointswinston (1).csv")
        df53 = pd.read_csv(Malawi_dir / "Malawi_new pointsWinston.csv")
        df54 = pd.read_csv(Malawi_dir / "Malawi_new pointsWinstons.csv")
        df55 = pd.read_csv(Malawi_dir / "Malawi_new points-Elizabeth.csv")
        df56 = pd.read_csv(Malawi_dir / "Malawi_new points-Goodwell Mzembe.csv")
        df57 = pd.read_csv(Malawi_dir / "Malawi_new points_Elizabeth.csv (2).csv")
        df58 = pd.read_csv(Malawi_dir / "Malawi_new points_Elizabeth.csv(3).csv")
        df59 = pd.read_csv(Malawi_dir / "Malawi_new points_Lupakisyo_3.csv")
        df60 = pd.read_csv(Malawi_dir / "Malawi_new points (non cropland) Umali.csv")
        df61 = pd.read_csv(Malawi_dir / "Malawi_new points_Chimwemwe (5).csv")
        df62 = pd.read_csv(Malawi_dir / "Malawi_new points_Twaibu Nathaniel Nn 3.csv")
        df63 = pd.read_csv(Malawi_dir / "Malawi_new points(2)_Benjamin.csv")
        df64 = pd.read_csv(Malawi_dir / "Malawi_new points (4)-Jane.csv")
        df65 = pd.read_csv(Malawi_dir / "Malawi_Chandiwira.csv")
        df66 = pd.read_csv(Malawi_dir / "Malawi_new points(5).csv")
        df67 = pd.read_csv(Malawi_dir / "Malawi_new points_Chimwemwe (6).csv")
        df68 = pd.read_csv(Malawi_dir / "Malawi_new points_Chimwemwe_7.csv")
        df69 = pd.read_csv(Malawi_dir / "Malawi_new points - Kim.csv")
        df70 = pd.read_csv(Malawi_dir / "Malawi_new points Kim_2.csv")
        df71 = pd.read_csv(Malawi_dir / "Malawi_new points Kondwani Amin.csv")
        df72 = pd.read_csv(Malawi_dir / "Malawi_new points -Umali.csv")

        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, 
                        df17, df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28, df29, df30, df31, 
                        df32, df33, df34, df35, df36, df37, df38, df39, df40, df41, df42, df43, df44, df45, df46,
                        df47, df48, df49, df50, df51, df52, df53, df54, df55, df56, df57, df58, df59, df60, df61, 
                        df62, df63, df64, df65, df66, df67, df68, df69, df70, df71, df72])
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df[CLASS_PROB] = (df["True value"] == 1).astype(int)
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = "training"
        return df


datasets: List[LabeledDataset] = [
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
                train_val_test=(0.0, 0.5, 0.5),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-2019-Zambia-Cropland-(RCMRD-Set-2)-sample-data-2021-12-12.csv",
                class_prob=lambda df: (df["Crop/non-crop"] == "Crop"),
                start_year=2019,
                train_val_test=(0.0, 0.5, 0.5),
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
        dataset="Namibia_corrective_labels_2020",
        country="Namibia",
        raw_labels=(
            RawLabels(
                filename="Namibia_corrected_labels.csv",
                class_prob=lambda df: (df["Landcover"] == "crop"),
                start_year=2020,
                train_val_test=(1.0, 0.0, 0.0),
                latitude_col="latitude",
                longitude_col="longitude",
                # filter_df=clean_ceo_data,
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
                train_val_test=(0.2, 0.4, 0.4),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-Namibia-North-Jan-2020---Dec-2020-(Set-2)-sample-data-2022-04-20.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                train_val_test=(0.2, 0.4, 0.4),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Namibia_WFP",
        country="Namibia",
        raw_labels=(
            RawLabels(
                filename="NAM_052021.zip",
                class_prob=1.0,
                start_year=2020,
                train_val_test=(1.0, 0.0, 0.0),
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Sudan_Blue_Nile_CEO_2019",
        country="Sudan",
        raw_labels=(
            RawLabels(
                filename=(
                    "ceo-Sudan-(Blue-Nile)-Feb-2019---Feb-2020-(Set-1)-sample-data-2022-10-31.csv"
                ),
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2019,
                train_val_test=(0.2, 0.4, 0.4),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename=(
                    "ceo-Sudan-(Blue-Nile)-Feb-2019---Feb-2020-(Set-2)-sample-data-2022-10-31.csv"
                ),
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2019,
                train_val_test=(0.2, 0.4, 0.4),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="Hawaii_CEO_2020",
        country="Hawaii",
        raw_labels=(
            RawLabels(
                filename="ceo-Hawaii-Jan-Dec-2020-(Set-1)-sample-data-2022-11-14.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                train_val_test=(0.4, 0.3, 0.3),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-Hawaii-Jan-Dec-2020-(Set-2)-sample-data-2022-11-14.csv",
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2020,
                train_val_test=(0.4, 0.3, 0.3),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
        ),
    ),
    HawaiiAgriculturalLandUse2020(),
    KenyaCEO2019(),
    HawaiiCorrective2020(),
    HawaiiCorrectiveGuided2020(),
    MalawiCorrectiveLabels2020(),
]

if __name__ == "__main__":
    create_datasets(datasets)
