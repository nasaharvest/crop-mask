import random
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from openmapflow.constants import (
    CLASS_PROB,
    END,
    EO_DATA,
    EO_FILE,
    EO_LAT,
    EO_LON,
    EO_STATUS,
    EO_STATUS_SKIPPED,
    EO_STATUS_WAITING,
    LABEL_DUR,
    LABELER_NAMES,
    LAT,
    LON,
    SOURCE,
    START,
    SUBSET,
)
from openmapflow.utils import to_date
from pandas.compat._optional import import_optional_dependency

# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
min_date = date(2015, 7, 1)

# Maximum date is 3 months back due to limitation of ERA5
# https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
max_date = date.today().replace(day=1) + relativedelta(months=-3)

SEED = 42


def _train_val_test_split(
    df: pd.DataFrame, train_val_test: Tuple[float, float, float]
) -> pd.DataFrame:
    _, val, test = train_val_test
    random_float = np.random.rand(len(df))
    df[SUBSET] = "testing"
    train_mask = (val + test) <= random_float
    df.loc[train_mask, SUBSET] = "training"
    validation_mask = (test <= random_float) & (random_float < (val + test))
    df.loc[validation_mask, SUBSET] = "validation"
    return df


def _read_in_file(file_path) -> pd.DataFrame:
    print(f"Reading in {file_path}")
    if file_path.suffix == ".txt":
        return pd.read_csv(file_path, sep="\t")
    elif file_path.suffix == ".csv":
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, engine="python")
    else:
        gpd = import_optional_dependency("geopandas")
        fiona = import_optional_dependency("fiona")
        try:
            return gpd.read_file(file_path)
        except fiona.errors.DriverError:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(file_path.parent)
            return gpd.read_file(file_path.parent / file_path.stem)


def _set_class_prob(df: pd.DataFrame, class_prob: Union[float, Callable]) -> pd.DataFrame:
    if isinstance(class_prob, float):
        df[CLASS_PROB] = class_prob
    elif isinstance(class_prob, int):
        df[CLASS_PROB] = float(class_prob)
    else:
        df[CLASS_PROB] = class_prob(df)
        if df[CLASS_PROB].dtype == bool or df[CLASS_PROB].dtype == int:
            df[CLASS_PROB] = df[CLASS_PROB].astype(float)
        elif df[CLASS_PROB].dtype != float:
            raise ValueError("Class probability must be a float")
    return df


def _set_start_end_dates(
    df: pd.DataFrame, start_year: Optional[int], start_date_col: Optional[str]
) -> pd.DataFrame:
    if start_year:
        df[START] = date(start_year, 1, 1)
        df[END] = date(start_year + 1, 12, 31)
    elif start_date_col:
        df[START] = np.vectorize(to_date)(df[start_date_col])
        df[START] = np.vectorize(lambda d: d.replace(month=1, day=1))(df[START])
        df[END] = np.vectorize(lambda d: d.replace(year=d.year + 1, month=12, day=31))(df[START])
    else:
        raise ValueError("Must specify either start_year or plant_date_col")

    if (df[END] >= max_date).any():
        df.loc[df[END] >= max_date, END] = max_date - timedelta(days=1)
    if (df[START] <= min_date).any():
        # Eliminate rows where the start date is way before the min date
        df = df[df[START] > min_date.replace(month=1)].copy()
        # Update the start to the min date in other rows
        df.loc[df[START] <= min_date, START] = min_date

    df = df[df[START] < df[END]].copy()

    df[START] = pd.to_datetime(df[START]).dt.strftime("%Y-%m-%d")
    df[END] = pd.to_datetime(df[END]).dt.strftime("%Y-%m-%d")
    return df


def _set_lat_lon(
    df: pd.DataFrame,
    latitude_col: Optional[str],
    longitude_col: Optional[str],
    sample_from_polygon: bool,
    x_y_from_centroid: bool,
    transform_crs_from: Optional[int],
) -> pd.DataFrame:
    if latitude_col and longitude_col:
        df[LAT] = df[latitude_col]
        df[LON] = df[longitude_col]
        return df

    if sample_from_polygon:
        gpd = import_optional_dependency("geopandas")
        df = df[df.geometry != None]  # noqa: E711
        df["samples"] = (df.geometry.area / 0.001).astype(int)

        def _get_points(polygon, samples: int):
            x_min, y_min, x_max, y_max = polygon.bounds
            x = np.random.uniform(x_min, x_max, samples)
            y = np.random.uniform(y_min, y_max, samples)
            gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
            gdf_points = gdf_points[gdf_points.within(polygon)]
            return gdf_points

        list_of_points = np.vectorize(_get_points)(df.geometry, df.samples)
        df = gpd.GeoDataFrame(geometry=pd.concat(list_of_points, ignore_index=True))

    if x_y_from_centroid:
        df = df[df.geometry != None]  # noqa: E711
        x = df.geometry.centroid.x.values
        y = df.geometry.centroid.y.values

        if transform_crs_from:
            pyproj = import_optional_dependency("pyproj")
            transformer = pyproj.Transformer.from_crs(crs_from=transform_crs_from, crs_to=4326)
            y, x = transformer.transform(xx=x, yy=y)

        df[LON] = x
        df[LAT] = y
        return df

    raise ValueError("Must specify latitude_col and longitude_col or x_y_from_centroid=True")


def _set_label_metadata(
    df, label_duration: Optional[str], labeler_name: Optional[str]
) -> pd.DataFrame:
    df[LABEL_DUR] = df[label_duration].astype(str) if label_duration else None
    df[LABELER_NAMES] = df[labeler_name].astype(str) if labeler_name else None
    return df


def _set_eo_columns(df) -> pd.DataFrame:
    df[EO_STATUS] = EO_STATUS_WAITING
    for col in [EO_DATA, EO_LAT, EO_LON, EO_FILE]:
        df[col] = None
    return df


@dataclass
class RawLabels:
    """
    Represents a raw labels file (csv, shapefile, geojson) and how it should be processed

    Args:
        filename (str): name of the raw file to be processed
            Example: "Togo_2019.csv"
        class_prob (float or Callable[[pd.DataFrame]]): A value or function to compute
            the class probability (0 negative class, 1 positive class)
            Example (float): 1.0
            Example (Callable): lambda df: df["Crop1"] == "Maize"
        train_val_test (Tuple[float, float, float]): A tuple of floats representing the ratio of
            train, validation, and test set.  The sum of the values must be 1.0
            Default: (1.0, 0.0, 0.0) [All data used for training]
        filter_df (Callable[[pd.DataFrame]]): A function to filter the dataframe before processing
            Example: lambda df: df[df["class"].notnull()]
            Default: None
        start_year (int): The year when the labels were collected, should be used when all labels
            are from the same year
            Example: 2019
        start_date_col (str): The name of the column representing the start date of the label,
            should be used when labels are from different years
            Example: "Planting Date"
        x_y_from_centroid (bool): Whether to use the centroid of the label as the latitude and
            longitude coordinates
            Default: False
        latitude_col (str): The name of the column representing the latitude of the label
            Default: None, will use the latitude of the centroid of the label
        longitude_col (str): The name of the column representing the longitude of the label
            Default: None, will use the longitude of the centroid of the label
        sample_from_polygons (bool): Whether to sample multiple points from the polygon instead
            of just using the centroid
            Default: False
        transform_crs_from (int): The EPSG code of the coordinate system of the raw data
            Default: None, assumes EPSG:4326
        label_duration (str): The name of the column representing the labeling duration of the label
            Default: None
        labeler_name (str): The name of the column representing the name of the labeler
            Default: None

    """

    filename: str

    # Label parameters
    class_prob: Union[float, Callable]
    train_val_test: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    filter_df: Optional[Callable] = None

    # Time parameters
    start_year: Optional[int] = None
    start_date_col: Optional[str] = None

    # Geolocation
    x_y_from_centroid: bool = True
    latitude_col: Optional[str] = None
    longitude_col: Optional[str] = None
    sample_from_polygon: bool = False
    transform_crs_from: Optional[int] = None

    # Label metadata
    label_duration: Optional[str] = None
    labeler_name: Optional[str] = None

    def __post_init__(self):
        np.random.seed(SEED)
        random.seed(SEED)
        if sum(self.train_val_test) != 1.0:
            raise ValueError("train_val_test must sum to 1.0")

    def process(self, raw_folder: Path) -> pd.DataFrame:
        df = _read_in_file(raw_folder / self.filename)
        if self.filter_df:
            df = self.filter_df(df)
        df = _set_lat_lon(
            df=df,
            latitude_col=self.latitude_col,
            longitude_col=self.longitude_col,
            sample_from_polygon=self.sample_from_polygon,
            x_y_from_centroid=self.x_y_from_centroid,
            transform_crs_from=self.transform_crs_from,
        )
        df[SOURCE] = self.filename
        df = _set_class_prob(df, self.class_prob)
        df = _set_start_end_dates(df, self.start_year, self.start_date_col)
        df = _set_label_metadata(df, self.label_duration, self.labeler_name)
        df = df.dropna(subset=[LON, LAT, CLASS_PROB])
        df = df.round({LON: 8, LAT: 8})
        df = _train_val_test_split(df, self.train_val_test)
        df = _set_eo_columns(df)

        return df[
            [
                SOURCE,
                CLASS_PROB,
                START,
                END,
                LON,
                LAT,
                SUBSET,
                LABELER_NAMES,
                LABEL_DUR,
                EO_DATA,
                EO_LAT,
                EO_LON,
                EO_FILE,
                EO_STATUS,
            ]
        ]
