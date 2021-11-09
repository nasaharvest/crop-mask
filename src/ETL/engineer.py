from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import numpy as np
import logging
import pickle

from src.band_calculations import process_bands
from src.ETL.constants import FEATURE_PATH, LAT, LON, CROP_PROB, SUBSET, START, END, TIF_PATHS
from src.utils import load_tif
from .data_instance import CropDataInstance

logger = logging.getLogger(__name__)


@dataclass
class Engineer(ABC):
    r"""Combine earth engine sentinel data
    and geowiki landcover 2017 data to make
    numpy arrays which can be input into the
    machine learning model
    """
    nan_fill: float = 0.0
    max_nan_ratio: float = 0.3
    add_ndvi: bool = True
    add_ndwi: bool = False

    @staticmethod
    def _find_nearest(array, value: float) -> Tuple[float, int]:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    @staticmethod
    def _distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        haversince formula, inspired by:
        https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude/41337005
        """
        p = 0.017453292519943295
        a = (
            0.5
            - np.cos((lat2 - lat1) * p) / 2
            + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
        )
        return 12742 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _distance_point_from_center(lat_idx: int, lon_idx: int, tif) -> int:
        x_dist = np.abs((len(tif.x) - 1) / 2 - lon_idx)
        y_dist = np.abs((len(tif.y) - 1) / 2 - lat_idx)
        return x_dist + y_dist

    def _find_matching_point(
        self, start: str, tif_paths: List[Path], label_lon: float, label_lat: float
    ) -> Tuple[np.ndarray, float, float, str]:
        """
        Given a label coordinate (y) this functions finds the associated satellite data (X)
        by looking through one or multiple tif files.
        Each tif file contains satellite data for a grid of coordinates.
        So the function finds the closest grid coordinate to the label coordinate.
        Additional value is given to a grid coordinate that is close to the center of the tif.
        """
        start_date = datetime.strptime(start, "%Y-%m-%d")
        tifs = [load_tif(p, days_per_timestep=30, start_date=start_date) for p in tif_paths]
        if len(tifs) > 1:
            min_distance_from_point = np.inf
            min_distance_from_center = np.inf
            for i, tif in enumerate(tifs):
                lon, lon_idx = self._find_nearest(tif.x, label_lon)
                lat, lat_idx = self._find_nearest(tif.y, label_lat)
                distance_from_point = self._distance(label_lat, label_lon, lat, lon)
                distance_from_center = self._distance_point_from_center(lat_idx, lon_idx, tif)
                if (distance_from_point < min_distance_from_point) or (
                    distance_from_point == min_distance_from_point
                    and distance_from_center < min_distance_from_center
                ):
                    closest_lon = lon
                    closest_lat = lat
                    min_distance_from_center = distance_from_center
                    min_distance_from_point = distance_from_point
                    labelled_np = tif.sel(x=lon).sel(y=lat).values
                    source_file = tif_paths[i].name
        else:
            closest_lon = self._find_nearest(tifs[0].x, label_lon)[0]
            closest_lat = self._find_nearest(tifs[0].y, label_lat)[0]
            labelled_np = tifs[0].sel(x=closest_lon).sel(y=closest_lat).values
            source_file = tif_paths[0].name

        return labelled_np, closest_lon, closest_lat, source_file

    def create_pickled_labeled_dataset(self, labels):
        for label in tqdm(labels.to_dict(orient="records"), desc="Creating pickled instances"):
            (tif_data, tif_lon, tif_lat, tif_file) = self._find_matching_point(
                start=label[START],
                tif_paths=label[TIF_PATHS],
                label_lon=label[LON],
                label_lat=label[LAT],
            )
            labelled_array = process_bands(
                tif_data,
                nan_fill=self.nan_fill,
                max_nan_ratio=self.max_nan_ratio,
                add_ndvi=self.add_ndvi,
                add_ndwi=self.add_ndwi,
            )

            instance = CropDataInstance(
                crop_probability=label[CROP_PROB],
                label_lat=label[LAT],
                label_lon=label[LON],
                start_date_str=label[START],
                end_date_str=label[END],
                data_subset=label[SUBSET],
                labelled_array=labelled_array,
                instance_lat=tif_lat,
                instance_lon=tif_lon,
                source_file=tif_file,
            )
            label[FEATURE_PATH].parent.mkdir(exist_ok=True)
            with label[FEATURE_PATH].open("wb") as f:
                pickle.dump(instance, f)
