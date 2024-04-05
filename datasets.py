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


class EthiopiaTigrayCorrective2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        df = pd.read_csv(raw_dir / "Ethiopia_Tigray_Corrective_2020.csv")
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df[CLASS_PROB] = (df["Wrong value"] == 0).astype(int)
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = "training"
        return df


class EthiopiaTigrayGhent2021(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        df = pd.read_csv(raw_dir / "Ethiopia_Tigray_Ghent_2021.csv")
        # Rename coordinate columns
        df = df.rename(
            columns={
                "Latitude (WGS84)": LAT,
                "Longitude (WGS84)": LON,
                "Crop/Non-Crop": CLASS_PROB,
            }
        )
        df[START], df[END] = date(2021, 1, 1), date(2022, 12, 31)
        df[SUBSET] = "validation"
        return df


class HawaiiAgriculturalLandUse2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        df = read_zip(raw_dir / "Hawaii_Agricultural_Land_Use_-_2020_Update.zip")
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[LAT], df[LON] = get_lat_lon_from_centroid(df.geometry)
        df[SUBSET] = "training"
        df[CLASS_PROB] = 1.0
        for non_crop in [
            "Commercial Forestry",
            "Pasture",
            "Flowers / Foliage / Landscape",
            "Seed Production",
            "Aquaculture",
            "Dairy",
        ]:
            df.loc[df["crops_2020"] == non_crop, CLASS_PROB] = 0.0

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


class SudanBlueNileCEO2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        SudanBlueNile_dir = raw_dir / "Sudan_Blue_Nile_CEO_2020"
        df1 = pd.read_csv(
            SudanBlueNile_dir
            / "ceo-Sudan-Blue-Nile-Feb-2020---Feb-2021-(Set-1)-sample-data-2023-04-04.csv"
        )
        df2 = pd.read_csv(
            SudanBlueNile_dir
            / "ceo-Sudan-Blue-Nile-Feb-2020---Feb-2021-(Set-2)-sample-data-2023-04-04.csv"
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
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.35, 0.35)
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
        df = pd.read_csv(Malawi_dir / "Malawi_corrective_labels_cleaned.csv")
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df = df.drop_duplicates(subset=[LAT, LON]).reset_index(drop=True)
        df[CLASS_PROB] = (df["True_value"] == 1).astype(int)
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = "training"
        # Removing index=2275 because it is a duplicate of
        # another point in Malawi_FAO_corrected
        df.drop(index=2275, inplace=True)
        return df


class NamibiaFieldBoundary2022(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        Namibia_dir = raw_dir / "Namibia_field_boundaries_2022"
        df = pd.read_csv(Namibia_dir / "Namibia_field_bnd_2022.csv")
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df = df.drop_duplicates(subset=[LAT, LON]).reset_index(drop=True)
        df[CLASS_PROB] = (df["landcover"] == 1).astype(int)
        df[START], df[END] = date(2021, 1, 1), date(2022, 12, 31)
        df[SUBSET] = "training"
        return df


class SudanBlueNileCorrectiveLabels2019(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        df = pd.read_csv(raw_dir / "Sudan.Blue.Nile_new.points.csv")
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df[CLASS_PROB] = (df["Wrong value"] == 0).astype(int)
        df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)
        df[SUBSET] = "training"
        return df


class SudanAlGadarefCEO2019(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        SudanAlGadaref_dir = raw_dir / "Sudan_Al_Gadaref_CEO_2019"
        df1 = pd.read_csv(
            SudanAlGadaref_dir
            / "ceo-Sudan-Al-Gadaref-May-2019---March-2020-(Set-1)-sample-data-2023-05-25.csv"
        )
        df2 = pd.read_csv(
            SudanAlGadaref_dir
            / "ceo-Sudan-Al-Gadaref-May-2019---March-2020-(Set-2)-sample-data-2023-05-25.csv"
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
        df[SUBSET] = train_val_test_split(df.index, 0.35, 0.35)
        return df


class SudanAlGadarefCEO2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        SudanAlGadaref_20_dir = raw_dir / "Sudan_Al_Gadaref_CEO_2020"
        df1 = pd.read_csv(
            SudanAlGadaref_20_dir
            / "ceo-Sudan-Al-Gadaref-May-2020---March-2021-(Set-1)-sample-data-2023-05-30.csv"
        )
        df2 = pd.read_csv(
            SudanAlGadaref_20_dir
            / "ceo-Sudan-Al-Gadaref-May-2020---March-2021-(Set-2)-sample-data-2023-05-30.csv"
        )
        df = pd.concat([df1, df2])
        df[CLASS_PROB] = df["Does this point contain active cropland?"] == "Crop"
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
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.35, 0.35)
        return df


class MaliStratifiedCEO2019(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        MaliStratified_dir = raw_dir / "Mali_Stratified_CEO_2019"
        df1 = pd.read_csv(
            MaliStratified_dir
            / "ceo-Mali-Feb-2019---Feb-2020-Stratified-sample-(Set-1)-sample-data-2023-05-26.csv"
        )
        df2 = pd.read_csv(
            MaliStratified_dir
            / "ceo-Mali-Feb-2019---Feb-2020-Stratified-sample-(Set-2)-sample-data-2023-05-26.csv"
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
        df[SUBSET] = train_val_test_split(df.index, 0.35, 0.35)
        return df


class NamibiaNorthStratified2020(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        NamibiaNorthStratified_dir = raw_dir / "Namibia_North_stratified_2020"
        df1 = pd.read_csv(
            NamibiaNorthStratified_dir
            / (
                "ceo-Namibia_North-Sep-2020---Sep-2021-Stratified-sample-(Set-1)"
                + "-sample-data-2023-06-22.csv"
            )
        )
        df2 = pd.read_csv(
            NamibiaNorthStratified_dir
            / (
                "ceo-Namibia_North-Sep-2020---Sep-2021-Stratified-sample-(Set-2)"
                + "-sample-data-2023-06-22.csv"
            )
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
        df[START], df[END] = date(2020, 1, 1), date(2021, 12, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.5, 0.5)
        return df


class Namibia_field_samples_22_23(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        Namibia_fld_dir = raw_dir / "Namibia_field_samples_22_23"
        df = pd.read_csv(Namibia_fld_dir / "Namibia_fld_pts_Sep22_May23.csv")
        df.rename(columns={"Latitude": LAT, "Longitude": LON}, inplace=True)
        df = df.drop_duplicates(subset=[LAT, LON]).reset_index(drop=True)
        df[CLASS_PROB] = (df["Landcover"] == "crop").astype(int)
        df[START], df[END] = date(2022, 1, 1), date(2023, 3, 31)
        df[SUBSET] = "training"
        return df


class SudanGedarefDarfurAlJazirah2022(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        raw_folder = raw_dir / "Sudan_Gedaref_Darfur_Al_Jazirah_2022"
        df1 = pd.read_csv(
            raw_folder / "ceo-Sudan-Feb-2022---Feb-2023-(Set-1)-sample-data-2024-02-15.csv"
        )
        df2 = pd.read_csv(
            raw_folder / "ceo-Sudan-Feb-2022---Feb-2023-(Set-2)-sample-data-2024-02-15.csv"
        )
        df = pd.concat([df1, df2])

        # Discard rows with no label
        df = df[~df["Does this pixel contain active cropland?"].isna()].copy()
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
        df[START], df[END] = date(2022, 1, 1), date(2023, 7, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.3, 0.3)
        return df


class SudanGedarefDarfurAlJazirah2023(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        raw_folder = raw_dir / "Sudan_Gedaref_Darfur_Al_Jazirah_2023"
        df = pd.read_csv(
            raw_folder / "ceo-Sudan-Feb-2023---Feb-2024-(Set-1)-sample-data-2024-02-15.csv"
        )

        # Discard rows with no label
        df = df[~df["Does this pixel contain active cropland?"].isna()].copy()
        df[CLASS_PROB] = df["Does this pixel contain active cropland?"] == "Crop"
        df[CLASS_PROB] = df[CLASS_PROB].astype(int)
        df["num_labelers"] = 2  # Two people reviewed each point
        df = df.groupby([LON, LAT], as_index=False, sort=False).agg(
            {
                CLASS_PROB: "mean",
                "num_labelers": "sum",
                "plotid": join_unique,
                "sampleid": join_unique,
                "email": join_unique,
            }
        )
        # Only keep examples with multiple labelers
        df = df[df["num_labelers"] > 1].copy()
        df[START], df[END] = date(2023, 1, 1), date(2023, 10, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.3, 0.3)
        return df


class Uganda_NorthCEO2022(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        raw_folder = raw_dir / "Uganda_North"
        df1 = pd.read_csv(
            raw_folder
            / "ceo-UNHCR-North-Uganda-Feb-2022---Feb-2023-(Set-1)-sample-data-2023-11-13.csv"
        )
        df2 = pd.read_csv(
            raw_folder
            / "ceo-UNHCR-North-Uganda-Feb-2022---Feb-2023-(Set-2)-sample-data-2023-11-13.csv"
        )
        df = pd.concat([df1, df2])

        # Discard rows with no label
        df = df[~df["Does this pixel contain active cropland?"].isna()].copy()
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
        df[START], df[END] = date(2022, 1, 1), date(2023, 7, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.3, 0.3)
        return df


class Uganda_NorthCEO2021(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        raw_folder = raw_dir / "Uganda_North_2021"
        df1 = pd.read_csv(
            raw_folder
            / "ceo-UNHCR-North-Uganda-Feb-2021---Feb-2022-(Set-1)-sample-data-2024-02-07.csv"
        )
        df2 = pd.read_csv(
            raw_folder
            / "ceo-UNHCR-North-Uganda-Feb-2021---Feb-2022-(Set-2)-sample-data-2024-02-07.csv"
        )
        df = pd.concat([df1, df2])

        # Discard rows with no label
        df = df[~df["Does this pixel contain active cropland?"].isna()].copy()
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
        df[START], df[END] = date(2021, 1, 1), date(2022, 12, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.3, 0.3)
        return df


class UgandaNorthCEO2019(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        raw_folder = raw_dir / "Uganda_North_2019"
        df1 = pd.read_csv(
            raw_folder
            / "ceo-UNHCR-North-Uganda-Feb-2019---Feb-2020-(Set-1)-sample-data-2024-03-12.csv"
        )
        df2 = pd.read_csv(
            raw_folder
            / "ceo-UNHCR-North-Uganda-Feb-2019---Feb-2020-(Set-2)-sample-data-2024-03-12.csv"
        )
        df = pd.concat([df1, df2])

        # Discard rows with no label
        df = df[~df["Does this pixel contain active cropland?"].isna()].copy()
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
        df[SUBSET] = train_val_test_split(df.index, 0.3, 0.3)
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
    # CustomLabeledDataset(
    #     dataset="Argentina_Buenos_Aires",
    #     country="Argentina",
    #     raw_labels=(
    #         RawLabels(
    #             filename="bc_mapeo_del_cultivo_0.csv",
    #             filter_df=lambda df: df[
    #                 (
    #                     df["Seleccione el cultivo principal en el lote:"].notnull()
    #                     & ~df["Seleccione el cultivo principal en el lote:"].isin(
    #                         ["otro", "barbecho", "sin_dato"]
    #                     )
    #                 )
    #             ].copy(),
    #             longitude_col="longitud",
    #             latitude_col="latitud",
    #             class_prob=lambda df: df["Seleccione el cultivo principal en el lote:"].isin(
    #                 ["trigo_o_cebada", "cultive_leguminosa", "maiz", "sorgo", "soja", "girasol"]
    #             ),
    #             train_val_test=(0.8, 0.2, 0.0),
    #             start_year=2021,
    #         ),
    #     ),
    # ),
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
                train_val_test=(0.6, 0.2, 0.2),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-2019-Zambia-Cropland-(RCMRD-Set-2)-sample-data-2021-12-12.csv",
                class_prob=lambda df: (df["Crop/non-crop"] == "Crop"),
                start_year=2019,
                train_val_test=(0.6, 0.2, 0.2),
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
    CustomLabeledDataset(
        dataset="Senegal_CEO_2022",
        country="Senegal",
        raw_labels=(
            RawLabels(
                filename="ceo-Senegal-March-2022---March-2023-Stratified-sample-(Set-1)-sample-data-2023-08-28.csv",  # noqa: E501
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2022,
                train_val_test=(0.2, 0.4, 0.4),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-Senegal-March-2022---March-2023-Stratified-sample-(Set-2)-sample-data-2023-08-28.csv",  # noqa: E501
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2022,
                train_val_test=(0.2, 0.4, 0.4),
                latitude_col="lat",
                longitude_col="lon",
                filter_df=clean_ceo_data,
            ),
            RawLabels(
                filename="ceo-Senegal-March-2022---March-2023-Stratified-sample-(Set-3)-sample-data-2023-11-26.csv",  # noqa: E501
                class_prob=lambda df: (df["Does this pixel contain active cropland?"] == "Crop"),
                start_year=2022,
                train_val_test=(0.2, 0.4, 0.4),
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
    # NamibiaFieldBoundary2022(),
    EthiopiaTigrayGhent2021(),
    SudanBlueNileCEO2020(),
    SudanBlueNileCorrectiveLabels2019(),
    EthiopiaTigrayCorrective2020(),
    SudanAlGadarefCEO2019(),
    MaliStratifiedCEO2019(),
    SudanAlGadarefCEO2020(),
    NamibiaNorthStratified2020(),
    Namibia_field_samples_22_23(),
    SudanGedarefDarfurAlJazirah2022(),
    SudanGedarefDarfurAlJazirah2023(),
    Uganda_NorthCEO2022(),
    Uganda_NorthCEO2021(),
    UgandaNorthCEO2019(),
]

if __name__ == "__main__":
    create_datasets(datasets)
