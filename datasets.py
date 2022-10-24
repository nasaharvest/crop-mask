from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import LAT, LON, START, END, SUBSET, COUNTRY
from openmapflow.labeled_dataset import LabeledDataset, create_datasets
from openmapflow.label_utils import (
    get_lat_lon_from_centroid,
    read_zip,
    train_val_test_split,
)
from openmapflow.utils import to_date

import pandas as pd
import numpy as np

from datetime import date, timedelta
from typing import List


label_col = "is_crop"
raw_dir = PROJECT_ROOT / DataPaths.RAW_LABELS


def join_unique(values):
    return ",".join([str(i) for i in values.unique()])


def ceo_merge(df: pd.DataFrame):
    df["num_labelers"] = 1
    df = df.groupby([LON, LAT], as_index=False, sort=False).agg(
        {
            label_col: "mean",
            "num_labelers": "sum",
            "plotid": join_unique,
            "sampleid": join_unique,
            "email": join_unique,
        }
    )
    return df


class GeowikiLandcover2017(LabeledDataset):
    def load_labels(self):
        df = pd.read_csv(raw_dir / "geowiki_landcover_2017" / "local_all_2.txt", sep="\t")
        df.rename(columns={"loc_cent_X": LON, "loc_cent_Y": LAT}, inplace=True)
        df[START], df[END] = date(2017, 1, 1), date(2018, 12, 31)
        df[label_col] = ((df["sum_crop"] / 100) > 0.5).astype(float)
        df[SUBSET] = "train"
        df[COUNTRY] = "global"
        return df


class Kenya(LabeledDataset):
    def load_labels(self):
        kenya_dir = raw_dir / "Kenya"

        # Load data
        df1 = read_zip(kenya_dir / "noncrop_labels_v2.zip")
        df2 = read_zip(kenya_dir / "noncrop_labels_set2.zip")
        df3 = read_zip(kenya_dir / "2019_gepro_noncrop.zip")
        df4 = read_zip(kenya_dir / "noncrop_water_kenya_gt.zip")
        df5 = read_zip(kenya_dir / "noncrop_kenya_gt.zip")
        df6 = read_zip(kenya_dir / "one_acre_fund_kenya.zip")
        df7 = read_zip(kenya_dir / "plant_village_kenya.zip")

        # Filter nans
        df1 = df1[df1.geometry.notna()].copy()
        df7 = df7[(df7["harvest_da"] != "nan") & (df7["harvest_da"] != "unknown")].copy()

        # Set coordinates
        for df in [df1, df2, df7]:
            df[LAT], df[LON] = get_lat_lon_from_centroid(df.geometry, src_crs=32636)
        for df in [df3, df4, df5]:
            df[LAT], df[LON] = get_lat_lon_from_centroid(df.geometry)
        df6.rename(columns={"Lat": LAT, "Lon": LON}, inplace=True)

        # Set dates
        for df in [df1, df2, df3, df4, df5, df6]:
            df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)

        # Set custom dates for planet_villega
        df7.loc[:, "planting_d"] = pd.to_datetime(df7["planting_d"])
        df7.loc[:, "harvest_da"] = pd.to_datetime(df7["harvest_da"])
        df7["between_days"] = (df7["harvest_da"] - df7["planting_d"]).dt.days
        year = pd.to_timedelta(timedelta(days=365))
        df7.loc[(-365 < df7["between_days"]) & (df7["between_days"] < 0), "harvest_da"] += year
        valid_years = [2018, 2019, 2020]
        df7 = df7[
            (df7["planting_d"].dt.year.isin(valid_years))
            & (df7["harvest_da"].dt.year.isin(valid_years))
        ].copy()
        df7 = df7[(0 < df7["between_days"]) & (df7["between_days"] <= 365)]
        df7[START] = np.vectorize(to_date)(df7["planting_d"])
        df7[START] = np.vectorize(lambda d: d.replace(month=1, day=1))(df7[START])
        df7[END] = np.vectorize(lambda d: d.replace(year=d.year + 1, month=12, day=31))(df7[START])

        # Set label
        for df in [df1, df2, df3, df4, df5]:
            df[label_col] = 0.0
        df6[label_col] = 1.0

        # Eliminate duplicates
        for df in [df1, df6, df7]:
            df = df.drop_duplicates(subset=[LAT, LON, START, END])

        # Set splits
        for df in [df1, df2, df3, df4, df5, df6, df7]:
            df[SUBSET] = train_val_test_split(df.index, 0.1, 0.1)

        df = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)

        # Eliminate additional duplicates on combining
        df = df.drop_duplicates(subset=[LAT, LON, START, END])
        return df


class Mali(LabeledDataset):
    def load_labels(self):
        df1 = read_zip(raw_dir / "Mali" / "mali_noncrop_2019.zip")
        df2 = read_zip(raw_dir / "Mali" / "sikasso_clean_fields")
        df3 = read_zip(raw_dir / "Mali" / "segou_bounds_07212020.zip")

        # Skip nans
        df2 = df2[df2.geometry.notna()].copy()

        # Set coordinates
        for df in [df1, df2, df3]:
            print(df.geometry.notna().all())
            df[LAT], df[LON] = get_lat_lon_from_centroid(df.geometry)
            df = df.round({LON: 8, LAT: 8})

        # Set dates
        for df in [df1, df2, df3]:
            df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)

        # Bounds in Segou cover 2 years
        df4 = df3.copy()
        df4[START], df4[END] = date(2018, 1, 1), date(2019, 12, 31)

        # Set label
        df1[label_col] = 0.0
        for df in [df2, df3, df4]:
            df[label_col] = 1.0

        # Drop duplicates
        df2 = df2.drop_duplicates(subset=[LAT, LON, START, END])
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        df = df.round({LON: 8, LAT: 8})
        df = df.drop_duplicates(subset=[LAT, LON, START, END])
        df[SUBSET] = "training"
        return df


class MaliLowerCEO2019(LabeledDataset):
    def load_labels(self):
        folder = raw_dir / "Mali_lower_CEO_2019"
        ceo_files = [
            f"ceo-2019-Mali-USAID-ZOIS-lower-(Set-{i})--sample-data-2021-11-29.csv" for i in [1, 2]
        ]
        df = pd.concat([pd.read_csv(folder / file) for file in ceo_files], ignore_index=True)
        df[label_col] = df["Does this point lie on a crop or non-crop pixel?"] == "Crop"
        df[label_col] = df[label_col].astype(int)
        df = ceo_merge(df)
        df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.5, 0.5)
        return df


class MaliUpperCEO2019(LabeledDataset):
    def load_labels(self):
        folder = raw_dir / "Mali_upper_CEO_2019"
        ceo_files = [
            f"ceo-2019-Mali-USAID-ZOIS-upper-(Set-{i})-sample-data-2021-11-25.csv" for i in [1, 2]
        ]
        df = pd.concat([pd.read_csv(folder / file) for file in ceo_files], ignore_index=True)
        df[label_col] = df["Does this point lie on a crop or non-crop pixel?"] == "Crop"
        df[label_col] = df[label_col].astype(int)
        df = ceo_merge(df)
        df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)
        df[SUBSET] = train_val_test_split(df.index, 0.5, 0.5)
        return df


datasets: List[LabeledDataset] = [Kenya(), Mali(), MaliLowerCEO2019(), MaliUpperCEO2019()]

if __name__ == "__main__":
    create_datasets(datasets)
