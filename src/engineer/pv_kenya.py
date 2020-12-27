from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import numpy as np
import geopandas
import json
from datetime import datetime

from typing import Optional


from src.processors import KenyaPVProcessor
from src.exporters import KenyaPVSentinelExporter
from .base import BaseEngineer, BaseDataInstance


@dataclass
class PVKenyaDataInstance(BaseDataInstance):

    crop_label: str
    crop_int: int


class PVKenyaEngineer(BaseEngineer):

    sentinel_dataset = KenyaPVSentinelExporter.dataset
    dataset = KenyaPVProcessor.dataset

    def __init__(self, data_folder: Path) -> None:
        super().__init__(data_folder)

        unique_classes = self.labels.crop_type.unique()

        self.classes_to_index = {
            crop: idx for idx, crop in enumerate(unique_classes[unique_classes != np.array(None)])
        }

        json.dump(self.classes_to_index, (self.savedir / "classes_to_index.json").open("w"))

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        pv_kenya = data_folder / "processed" / KenyaPVProcessor.dataset / "data.geojson"
        assert pv_kenya.exists(), "Kenya Plant Village processor must be run to load labels"
        return geopandas.read_file(pv_kenya)

    def process_single_file(
        self,
        path_to_file: Path,
        nan_fill: float,
        max_nan_ratio: float,
        add_ndvi: bool,
        add_ndwi: bool,
        calculate_normalizing_dict: bool,
        start_date: datetime,
        days_per_timestep: int,
        is_test: bool,
    ) -> Optional[PVKenyaDataInstance]:
        r"""
        Return a tuple of np.ndarrays of shape [n_timesteps, n_features] for
        1) the anchor (labelled)
        """

        da = self.load_tif(path_to_file, days_per_timestep=days_per_timestep, start_date=start_date)

        # first, we find the label encompassed within the da

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        overlap = self.labels[
            (
                (self.labels.lon <= max_lon)
                & (self.labels.lon >= min_lon)
                & (self.labels.lat <= max_lat)
                & (self.labels.lat >= min_lat)
            )
        ]
        if len(overlap) == 0:
            return None
        else:
            label_lat = overlap.iloc[0].lat
            label_lon = overlap.iloc[0].lon

            crop_type = overlap.iloc[0].crop_type

            if crop_type is None:
                return None

            else:
                crop_int = self.classes_to_index[crop_type]

                closest_lon, _ = self.find_nearest(da.x, label_lon)
                closest_lat, _ = self.find_nearest(da.y, label_lat)

                labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values

                if add_ndvi:
                    labelled_np = self.calculate_ndvi(labelled_np)
                if add_ndwi:
                    labelled_np = self.calculate_ndwi(labelled_np)

                labelled_array = self.maxed_nan_to_num(
                    labelled_np, nan=nan_fill, max_ratio=max_nan_ratio
                )

                if (not is_test) and calculate_normalizing_dict:
                    self.update_normalizing_values(self.normalizing_dict_interim, labelled_array)

                if labelled_array is not None:
                    return PVKenyaDataInstance(
                        label_lat=label_lat,
                        label_lon=label_lon,
                        instance_lat=closest_lat,
                        instance_lon=closest_lon,
                        labelled_array=labelled_array,
                        crop_label=crop_type,
                        crop_int=crop_int,
                    )
                else:
                    return None
