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
    default_end_month_day = (4, 16)

    custom_end_date: Optional[date] = None
    custom_start_date: Optional[date] = None

    use_harvest_date_for_date_range: bool = False

    clean_df: Optional[Callable] = None
    x_y_from_centroid: bool = True
    x_y_reversed: bool = False
    lat_lon_lowercase: bool = False
    lat_lon_transform: bool = False
    custom_geowiki_processing: bool = False
    custom_pv_end_date: Optional[Tuple[int, int]] = None

    min_date = date(2017, 3, 28)

    def __post_init__(self):
        set_seed()

    @staticmethod
    def _date_overlap(start1: date, end1: date, start2: date, end2: date) -> int:
        overlaps = start1 <= end2 and end1 >= start2
        if not overlaps:
            return 0
        overlap_days = (min(end1, end2) - max(start1, start2)).days
        return overlap_days

    @staticmethod
    def custom_geowiki_process(df) -> xr.DataArray:
        # first, we find the mean sumcrop calculated per location
        mean_per_location = df.groupby("location_id").mean()

        # then, we rename the columns
        mean_per_location = mean_per_location.rename(
            {"loc_cent_X": "lon", "loc_cent_Y": "lat", "sumcrop": "mean_sumcrop"},
            axis="columns",
            errors="raise",
        )
        mean_per_location = mean_per_location.reset_index()
        return mean_per_location

    @staticmethod
    def compute_custom_pv_end_date(df, total_days, end_month_day):
        if "harvest_da" not in df or "planting_d" not in df:
            raise ValueError(
                "Expected plant_village dataframe to include harvest_da and planting_d columns"
            )

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
                key=lambda d: Processor._date_overlap(
                    planting_date, harvest_date, d - total_days, d
                ),
            )
            return end_date

        return np.vectorize(compute_end_date)(
            pd.to_datetime(df["planting_d"]), pd.to_datetime(df["harvest_da"])
        )

    def process(self, raw_folder: Path, total_days) -> Union[pd.DataFrame, xr.DataArray]:
        file_path = raw_folder / self.file_name
        logger.info(f"Reading in {file_path}")
        if file_path.suffix == ".txt":
            df = pd.read_csv(file_path, sep="\t")
        else:
            df = geopandas.read_file(file_path)

        if self.clean_df:
            df = self.clean_df(df)

        if self.custom_geowiki_processing:
            df = self.custom_geowiki_process(df)

        df["crop_probability"] = (
            self.crop_prob if isinstance(self.crop_prob, float) else self.crop_prob(df)
        )
        df["source"] = file_path.stem
        df["index"] = df.index

        if self.custom_end_date:
            df["end_date"] = self.custom_end_date
        elif self.end_year:
            df["end_date"] = date(self.end_year, *self.default_end_month_day)
        elif self.use_harvest_date_for_date_range:
            df["end_date"] = self.compute_custom_pv_end_date(df, total_days, self.default_end_month_day)
        else:
            raise ValueError(
                "end_date could not be computed please set either: custom_end_date, end_year, or use_harvest_date_for_date_range"
            )

        if self.custom_start_date:
            df["start_date"] = self.custom_start_date
        else:
            df["start_date"] = df["end_date"] - total_days

        df["end_date"] = pd.to_datetime(df["end_date"]).dt.strftime("%Y-%m-%d")
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d")

        if self.lat_lon_lowercase:
            df = df.rename(columns={"Lat": "lat", "Lon": "lon"})

        if self.x_y_from_centroid:
            x = df.geometry.centroid.x.values
            y = df.geometry.centroid.y.values

            if self.x_y_reversed:
                x, y = y, x

            if self.lat_lon_transform:
                transformer = Transformer.from_crs(crs_from=32636, crs_to=4326)
                lat, lon = transformer.transform(xx=x, yy=y)
                df["lat"] = lat
                df["lon"] = lon
            else:
                df["lon"] = x
                df["lat"] = y

        return df
