from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import LAT, LON, START, END, SUBSET, COUNTRY
from openmapflow.labeled_dataset import LabeledDataset
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

        df1 = read_zip(kenya_dir / "noncrop_labels_v2.zip")
        df2 = read_zip(kenya_dir / "noncrop_labels_set2.zip")
        for df in [df1, df2]:
            df[LAT], df[LON] = get_lat_lon_from_centroid(df.geometry, src_crs=32636)

        df3 = read_zip(kenya_dir / "2019_gepro_noncrop.zip")
        df4 = read_zip(kenya_dir / "noncrop_water_kenya_gt.zip")
        df5 = read_zip(kenya_dir / "noncrop_labels_set3.zip")
        for df in [df3, df4, df5]:
            df[LAT], df[LON] = get_lat_lon_from_centroid(df.geometry)

        for df in [df1, df2, df3, df4, df5]:
            df[label_col] = 0.0

        df6 = read_zip(kenya_dir / "crop_labels_v2.zip")
        df6.rename(columns={"Lat": LAT, "Lon": LON}, inplace=True)
        df6[label_col] = 1.0
        for df in [df1, df2, df3, df4, df5, df6]:
            df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)
            df[SUBSET] = train_val_test_split(0.8, 0.1, 0.1)

        df7 = read_zip(kenya_dir / "plant_village_kenya")
        df7 = df7[(df7["harvest_da"] != "nan") & (df7["harvest_da"] != "unknown")].copy()
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
        df[START] = np.vectorize(to_date)(df["planting_d"])
        df[START] = np.vectorize(lambda d: d.replace(month=1, day=1))(df[START])
        df[END] = np.vectorize(lambda d: d.replace(year=d.year + 1, month=12, day=31))(df[START])
        df7[SUBSET] = train_val_test_split(0.8, 0.1, 0.1)

        return pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)


datasets: List[LabeledDataset] = [Kenya()]
