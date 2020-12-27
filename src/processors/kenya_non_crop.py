import geopandas
from pathlib import Path
import pandas as pd
from pyproj import Transformer

from .base import BaseProcessor

from typing import List


class KenyaNonCropProcessor(BaseProcessor):

    dataset = "kenya_non_crop"

    @staticmethod
    def process_set(filepath: Path, latlon: bool, reversed: bool) -> geopandas.GeoDataFrame:
        df = geopandas.read_file(filepath)

        x, y = df.geometry.centroid.x.values, df.geometry.centroid.y.values

        if reversed:
            x, y = y, x

        if not latlon:

            transformer = Transformer.from_crs(crs_from=32636, crs_to=4326)

            lat, lon = transformer.transform(xx=x, yy=y)
            df["lat"] = lat
            df["lon"] = lon
        else:
            df["lat"] = x
            df["lon"] = y

        df["index"] = df.index

        return df

    def process(self) -> None:

        filepaths = [
            (self.raw_folder / "noncrop_labels_v2", False, False),
            (self.raw_folder / "noncrop_labels_set2", False, False),
            (self.raw_folder / "2019_gepro_noncrop", True, True),
            (self.raw_folder / "noncrop_water_kenya_gt", True, True),
            (self.raw_folder / "noncrop_kenya_gt", True, True),
        ]

        dfs: List[geopandas.GeoDataFrame] = []
        for filepath, is_latlon, is_reversed in filepaths:
            dfs.append(self.process_set(filepath, is_latlon, is_reversed))

        df = pd.concat(dfs)
        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
