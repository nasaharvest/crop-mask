from dataclasses import dataclass
from pathlib import Path
from typing import Union
from pyproj import Transformer
from src.utils import set_seed
import xarray as xr
import geopandas
import pandas as pd


@dataclass
class Processor:
    r"""Creates the appropriate directory in the data dir (``data_dir/processed/{dataset}``)."""
    file_name: str
    x_y_from_centroid: bool = True
    x_y_reversed: bool = False
    lat_lon_lowercase: bool = False
    lat_lon_transform: bool = False
    custom_geowiki_processing: bool = False

    def __post_init__(self):
        set_seed()

    @staticmethod
    def custom_geowiki_process(df) -> xr.DataArray:
        # first, we find the mean sumcrop calculated per location
        mean_per_location = (
            df[["location_id", "sumcrop", "loc_cent_X", "loc_cent_Y"]].groupby("location_id").mean()
        )

        # then, we rename the columns
        mean_per_location = mean_per_location.rename(
            {"loc_cent_X": "lon", "loc_cent_Y": "lat", "sumcrop": "mean_sumcrop"},
            axis="columns",
            errors="raise",
        )
        # then, we turn it into an xarray with x and y as indices
        output_xr = (
            mean_per_location.reset_index().set_index(["lon", "lat"])["mean_sumcrop"].to_xarray()
        )
        return output_xr

    def process(self, raw_folder: Path) -> Union[pd.DataFrame, xr.DataArray]:
        file_path = raw_folder / self.file_name
        if file_path.suffix == ".txt":
            df = pd.read_csv(file_path, sep="\t")
        else:
            df = geopandas.read_file(file_path)

        if self.custom_geowiki_processing:
            return self.custom_geowiki_process(df)

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
                df["lat"] = x
                df["lon"] = y

        df["index"] = df.index
        return df
