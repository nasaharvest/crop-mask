from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Tuple, Optional, Union
from pyproj import Transformer
from src.utils import set_seed
import logging
import xarray as xr
import geopandas
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Processor:
    r"""Creates the appropriate directory in the data dir (``data_dir/processed/{dataset}``)."""
    file_name: str
    crop_prob: Union[float, Callable]

    end_year: Optional[int] = None
    end_month_day: Tuple[int, int] = (4, 16)

    plant_date_col: Optional[str] = None
    harvest_date_col: Optional[str] = None

    clean_df: Optional[Callable] = None
    x_y_from_centroid: bool = True
    lat_lon_transform: bool = False

    min_date = date(2017, 3, 28)

    def __post_init__(self):
        set_seed()

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
                key=lambda d: _date_overlap(
                    planting_date, harvest_date, d - total_days, d
                ),
            )
            return end_date

        return np.vectorize(compute_end_date)(planting_date_col, harvest_date_col)

    def process(self, raw_folder: Path, total_days) -> Union[pd.DataFrame, xr.DataArray]:
        file_path = raw_folder / self.file_name
        logger.info(f"Reading in {file_path}")
        if file_path.suffix == ".txt":
            df = pd.read_csv(file_path, sep="\t")
        else:
            df = geopandas.read_file(file_path)

        if self.clean_df:
            df = self.clean_df(df)

        df["source"] = file_path.stem
        df["index"] = df.index

        if isinstance(self.crop_prob, float):
            df["crop_probability"] = self.crop_prob
        else:
            df["crop_probability"] = self.crop_prob(df)

        if self.end_year:
            df["end_date"] = date(self.end_year, *self.end_month_day)
        elif self.plant_date_col and self.harvest_date_col:
            df["end_date"] = self.end_date_using_overlap(df[self.plant_date_col], df[self.harvest_date_col], total_days, self.end_month_day)
        else:
            raise ValueError(
                "end_date could not be computed please set either: end_year, or plant_date_col and harvest_date_col"
            )

        df["start_date"] = df["end_date"] - total_days
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.strftime("%Y-%m-%d")
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d")

        if self.x_y_from_centroid:
            x = df.geometry.centroid.x.values
            y = df.geometry.centroid.y.values

            if self.lat_lon_transform:
                transformer = Transformer.from_crs(crs_from=32636, crs_to=4326)
                y, x = transformer.transform(xx=x, yy=y)

            df["lon"] = x
            df["lat"] = y

        df = df.dropna(subset=['lon', 'lat', 'crop_probability'])
        return df
