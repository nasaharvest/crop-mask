from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Tuple, Optional, Union
from pyproj import Transformer
from src.utils import set_seed
from src.constants import SOURCE, CROP_PROB, START, END, LON, LAT, SUBSET
import logging
import xarray as xr
import geopandas
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

min_date = date(2017, 3, 28)


@dataclass
class Processor:
    r"""Creates the appropriate directory in the data dir (``data_dir/processed/{dataset}``)."""
    filename: str
    crop_prob: Union[float, Callable]

    train_val_test: Tuple[float, float, float] = (0.8, 0.2, 0.0)

    end_month_day: Tuple[int, int] = (4, 16)
    end_year: Optional[int] = None
    custom_start_date: Optional[date] = None

    plant_date_col: Optional[str] = None
    harvest_date_col: Optional[str] = None

    clean_df: Optional[Callable] = None
    x_y_from_centroid: bool = True
    transform_crs_from: Optional[int] = None

    def __post_init__(self):
        set_seed()
        if sum(self.train_val_test) != 1.0:
            raise ValueError("train_val_test must sum to 1.0")

    @staticmethod
    def end_date_using_overlap(planting_date_col, harvest_date_col, total_days, end_month_day):
        def _date_overlap(start1: date, end1: date, start2: date, end2: date) -> int:
            overlaps = start1 <= end2 and end1 >= start2
            if not overlaps:
                return 0
            overlap_days = (min(end1, end2) - max(start1, start2)).days
            return overlap_days

        def compute_end_date(planting_date, harvest_date):
            to_date = (
                lambda d: d.astype("M8[D]").astype("O") if type(d) == np.datetime64 else d.date()
            )
            planting_date = to_date(planting_date)
            harvest_date = to_date(harvest_date)
            potential_end_dates = [
                date(harvest_date.year + diff, *end_month_day) for diff in [2, 1, 0, -1]
            ]
            potential_end_dates = [d for d in potential_end_dates if d < datetime.now().date()]
            end_date = max(
                potential_end_dates,
                key=lambda d: _date_overlap(planting_date, harvest_date, d - total_days, d),
            )
            return end_date

        return np.vectorize(compute_end_date)(planting_date_col, harvest_date_col)

    def train_val_test_split(self, df: pd.DataFrame):
        train, val, test = self.train_val_test
        random_float = np.random.rand(len(df))

        df[SUBSET] = "testing"

        train_mask = (val + test) <= random_float
        df.loc[train_mask, SUBSET] = "training"

        validation_mask = (test <= random_float) & (random_float < (val + test))
        df.loc[validation_mask, SUBSET] = "validation"

        return df

    def process(self, raw_folder: Path, total_days) -> Union[pd.DataFrame, xr.DataArray]:
        file_path = raw_folder / self.filename
        logger.info(f"Reading in {file_path}")
        if file_path.suffix == ".txt":
            df = pd.read_csv(file_path, sep="\t")
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = geopandas.read_file(file_path)

        if self.clean_df:
            df = self.clean_df(df)

        df[SOURCE] = self.filename

        if isinstance(self.crop_prob, float):
            df[CROP_PROB] = self.crop_prob
        else:
            df[CROP_PROB] = self.crop_prob(df)

        if self.end_year:
            df[END] = date(self.end_year, *self.end_month_day)
        elif self.plant_date_col and self.harvest_date_col:
            df[END] = self.end_date_using_overlap(
                df[self.plant_date_col], df[self.harvest_date_col], total_days, self.end_month_day
            )
        else:
            raise ValueError(
                "end_date could not be computed please set either: end_year, "
                "or plant_date_col and harvest_date_col"
            )

        if self.custom_start_date:
            df[START] = self.custom_start_date
        else:
            df[START] = df[END] - total_days

        if (df[START] < pd.Timestamp(min_date)).any():
            raise ValueError(
                f"start_date is set earlier than {min_date} "
                f"(earlier than sentinel-2 data exists)"
            )

        df[END] = pd.to_datetime(df[END]).dt.strftime("%Y-%m-%d")
        df[START] = pd.to_datetime(df[START]).dt.strftime("%Y-%m-%d")

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
        df = df.round({LON: 12, LAT: 12})
        df = self.train_val_test_split(df)

        return df[[SOURCE, CROP_PROB, START, END, LON, LAT, SUBSET]]
