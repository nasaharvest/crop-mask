import pandas as pd
from datetime import timedelta

from openmapflow.labeled_dataset import LabeledDataset, create_datasets
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
    LabeledDataset(
        dataset="geowiki_landcover_2017",
        country="global",
    ),
    LabeledDataset(
        dataset="open_buildings",
        country="global",
    ),
    LabeledDataset(
        dataset="digitalearthafrica_eastern",
        country="global",
    ),
    LabeledDataset(
        dataset="digitalearthafrica_sahel",
        country="global",
    ),
    LabeledDataset(
        dataset="Ethiopia",
        country="Ethiopia",
    ),
    LabeledDataset(
        dataset="Ethiopia_Tigray_2020",
        country="Ethiopia",
    ),
    LabeledDataset(
        dataset="Ethiopia_Tigray_2021",
        country="Ethiopia",
    ),
    LabeledDataset(
        dataset="Ethiopia_Bure_Jimma_2019",
        country="Ethiopia",
    ),
    LabeledDataset(
        dataset="Ethiopia_Bure_Jimma_2020",
        country="Ethiopia",
    ),
    LabeledDataset(
        dataset="Argentina_Buenos_Aires",
        country="Argentina",
    ),
    LabeledDataset(
        dataset="Malawi_CEO_2020",
        country="Malawi",
    ),
    LabeledDataset(
        dataset="Malawi_CEO_2019",
        country="Malawi",
    ),
    LabeledDataset(
        dataset="Malawi_FAO",
        country="Malawi",
    ),
    LabeledDataset(
        dataset="Malawi_FAO_corrected",
        country="Malawi",
    ),
    LabeledDataset(
        dataset="Zambia_CEO_2019",
        country="Zambia",
    ),
    LabeledDataset(
        dataset="Tanzania_CEO_2019",
        country="Tanzania",
    ),
    LabeledDataset(
        dataset="Malawi_corrected",
        country="Malawi",
    ),
    LabeledDataset(
        dataset="Namibia_CEO_2020",
        country="Namibia",
    ),
]

if __name__ == "__main__":
    create_datasets(datasets)
