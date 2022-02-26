from dataclasses import dataclass
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Callable, Tuple, Optional, Union
from pyproj import Transformer
from src.utils import set_seed
from src.ETL.constants import (
    SOURCE,
    CROP_PROB,
    START,
    END,
    LON,
    LAT,
    SUBSET,
    CROP_TYPE,
    LABEL_DUR,
    LABELER_NAMES,
)
import geopandas as gpd
import numpy as np
import pandas as pd

# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
min_date = date(2015, 7, 1)

# Maximum date is 3 months back due to limitation of ERA5
# https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
max_date = date.today().replace(day=1) + relativedelta(months=-3)


@dataclass
class Processor:
    r"""Creates the appropriate directory in the data dir (``data_dir/processed/{dataset}``)."""
    filename: str
    crop_prob: Union[float, Callable]

    train_val_test: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    start_year: Optional[int] = None
    plant_date_col: Optional[str] = None

    latitude_col: Optional[str] = None
    longitude_col: Optional[str] = None

    label_dur: Optional[str] = None
    label_names: Optional[str] = None

    crop_type_col: Optional[str] = None

    clean_df: Optional[Callable] = None
    sample_from_polygon: bool = False
    x_y_from_centroid: bool = True
    transform_crs_from: Optional[int] = None

    def __post_init__(self):
        set_seed()
        if sum(self.train_val_test) != 1.0:
            raise ValueError("train_val_test must sum to 1.0")

    @staticmethod
    def _to_date(d):
        if type(d) == np.datetime64:
            return d.astype("M8[D]").astype("O")
        elif type(d) == str:
            return pd.to_datetime(d).date()
        else:
            return d.date()

    @staticmethod
    def train_val_test_split(df: pd.DataFrame, train_val_test: Tuple[float, float, float]):
        _, val, test = train_val_test
        random_float = np.random.rand(len(df))

        df[SUBSET] = "testing"

        train_mask = (val + test) <= random_float
        df.loc[train_mask, SUBSET] = "training"

        validation_mask = (test <= random_float) & (random_float < (val + test))
        df.loc[validation_mask, SUBSET] = "validation"

        return df

    @staticmethod
    def get_points(polygon, samples: int) -> gpd.GeoSeries:

        # find the bounds of your geodataframe
        x_min, y_min, x_max, y_max = polygon.bounds

        # generate random data within the bounds
        x = np.random.uniform(x_min, x_max, samples)
        y = np.random.uniform(y_min, y_max, samples)

        # convert them to a points GeoSeries
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        # only keep those points within polygons
        gdf_points = gdf_points[gdf_points.within(polygon)]

        return gdf_points

    def process(self, raw_folder: Path) -> pd.DataFrame:
        file_path = raw_folder / self.filename
        print(f"Reading in {file_path}")
        if file_path.suffix == ".txt":
            df = pd.read_csv(file_path, sep="\t")
        elif file_path.suffix == ".csv":
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, engine="python")
        else:
            df = gpd.read_file(file_path)

        if self.latitude_col:
            df[LAT] = df[self.latitude_col]
        if self.longitude_col:
            df[LON] = df[self.longitude_col]

        if self.clean_df:
            df = self.clean_df(df)

        if self.sample_from_polygon:
            df = df[df.geometry != None]  # noqa: E711
            df["samples"] = (df.geometry.area / 0.001).astype(int)
            list_of_points = np.vectorize(self.get_points)(df.geometry, df.samples)
            df = gpd.GeoDataFrame(geometry=pd.concat(list_of_points, ignore_index=True))

        if self.label_dur:
            df[LABEL_DUR] = df[self.label_dur].astype(str)
        else:
            df[LABEL_DUR] = ""

        if self.label_names:
            df[LABELER_NAMES] = df[self.label_names].astype(str)
        else:
            df[LABELER_NAMES] = ""

        df[SOURCE] = self.filename

        if isinstance(self.crop_prob, float):
            df[CROP_PROB] = self.crop_prob
        else:
            df[CROP_PROB] = self.crop_prob(df)
            if df[CROP_PROB].dtype == bool or df[CROP_PROB].dtype == int:
                df[CROP_PROB] = df[CROP_PROB].astype(float)
            elif df[CROP_PROB].dtype != float:
                raise ValueError("Crop probability must be a float")

        if self.crop_type_col:
            df[CROP_TYPE] = df[self.crop_type_col]
        else:
            df[CROP_TYPE] = None

        if self.start_year:
            df[START] = date(self.start_year, 1, 1)
            df[END] = date(self.start_year + 1, 12, 31)
        elif self.plant_date_col:
            df[START] = np.vectorize(self._to_date)(df[self.plant_date_col])
            df[START] = np.vectorize(lambda d: d.replace(month=1, day=1))(df[START])
            df[END] = np.vectorize(lambda d: d.replace(year=d.year + 1, month=12, day=31))(
                df[START]
            )
        else:
            raise ValueError("Must specify either start_year or plant_date_col")

        if (df[END] >= max_date).any():
            df.loc[df[END] >= max_date, END] = max_date - timedelta(days=1)
        if (df[START] <= min_date).any():
            df.loc[df[START] <= min_date, START] = min_date

        df = df[df[START] < df[END]].copy()

        df[START] = pd.to_datetime(df[START]).dt.strftime("%Y-%m-%d")
        df[END] = pd.to_datetime(df[END]).dt.strftime("%Y-%m-%d")

        if self.x_y_from_centroid:
            df = df[df.geometry != None]  # noqa: E711
            x = df.geometry.centroid.x.values
            y = df.geometry.centroid.y.values

            if self.transform_crs_from:
                transformer = Transformer.from_crs(crs_from=self.transform_crs_from, crs_to=4326)
                y, x = transformer.transform(xx=x, yy=y)

            df[LON] = x
            df[LAT] = y

        df = df.dropna(subset=[LON, LAT, CROP_PROB])
        df = df.round({LON: 8, LAT: 8})
        df = self.train_val_test_split(df, self.train_val_test)

        return df[
            [SOURCE, CROP_PROB, START, END, LON, LAT, SUBSET, CROP_TYPE, LABELER_NAMES, LABEL_DUR]
        ]
