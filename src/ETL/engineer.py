from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import numpy as np
import logging
import pandas as pd
import pickle

from src.band_calculations import process_bands
from src.ETL.constants import LAT, LON, CROP_PROB, SUBSET, START, END, DEST_TIF
from src.utils import set_seed, load_tif
from .data_instance import CropDataInstance

logger = logging.getLogger(__name__)

mandatory_cols = {LAT, LON, CROP_PROB, SUBSET}


@dataclass
class Engineer(ABC):
    r"""Combine earth engine sentinel data
    and geowiki landcover 2017 data to make
    numpy arrays which can be input into the
    machine learning model
    """
    sentinel_files_path: Path
    labels_path: Path
    save_dir: Path
    nan_fill: float = 0.0
    max_nan_ratio: float = 0.3
    add_ndvi: bool = True
    add_ndwi: bool = False
    days_per_timestep: int = 30

    # should be True if the dataset contains data which will
    # only be used for evaluation (e.g. the TogoEvaluation dataset)
    eval_only: bool = False

    def __post_init__(self):
        set_seed()

    @staticmethod
    def _find_nearest(array, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _create_labeled_data_instance(
        self, row: Tuple[str, str, float, str, str, str, float, float, str]
    ) -> None:
        r"""
        Return a tuple of np.ndarrays of shape [n_timesteps, n_features] for
        1) the anchor (labelled)
        """
        tif_path_str, feature_path_str, crop_prob, subset, start, end, lon, lat, dest_tif = row

        tif_path = Path(tif_path_str)
        feature_path = Path(feature_path_str)
        da = load_tif(
            tif_path,
            days_per_timestep=self.days_per_timestep,
            start_date=datetime.strptime(start, "%Y-%m-%d"),
        )

        closest_lon = self._find_nearest(da.x, lon)
        closest_lat = self._find_nearest(da.y, lat)

        labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values
        labelled_array = process_bands(
            labelled_np,
            nan_fill=self.nan_fill,
            max_nan_ratio=self.max_nan_ratio,
            add_ndvi=self.add_ndvi,
            add_ndwi=self.add_ndwi,
        )

        if labelled_array is None:
            raise ValueError(f"{dest_tif} has empty labelled array")

        instance = CropDataInstance(
            crop_probability=crop_prob,
            instance_lat=closest_lat,
            instance_lon=closest_lon,
            label_lat=lat,
            label_lon=lon,
            labelled_array=labelled_array,
            data_subset=subset,
            source_file=dest_tif,
            start_date_str=start,
            end_date_str=end,
        )

        feature_path.parent.mkdir(exist_ok=True)
        with feature_path.open("wb") as f:
            pickle.dump(instance, f)

    def _create_feature_path(self, row) -> str:
        return str(self.save_dir / row[SUBSET] / f"{Path(row[DEST_TIF]).stem}.pkl")

    def _create_tif_path(self, dest_tif) -> str:
        return str(self.sentinel_files_path.joinpath(dest_tif))

    def create_pickled_labeled_dataset(self):
        labels = pd.read_csv(self.labels_path)
        labels = labels[labels[CROP_PROB] != 0.5]

        if not mandatory_cols.issubset(set(labels.columns)):
            raise ValueError(f"{self.labels_path} is missing one of {mandatory_cols}")

        logger.info(f"Creating pickled labeled dataset: {self.save_dir}")
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Check which tif files exist
        labels["tif_path"] = labels[DEST_TIF].apply(self._create_tif_path)
        existing_tifs = labels["tif_path"].apply(lambda p: Path(p).exists())
        labels_with_no_tif = labels[~existing_tifs]
        if len(labels_with_no_tif) == len(labels):
            raise ValueError(
                "No tifs exist, please export them from Google EarthEngine using export_for_labeled"
            )
        elif len(labels_with_no_tif) > 0:
            print(
                f"{len(labels_with_no_tif)} labels will be ignored because they have no tif file."
            )
            labels = labels[existing_tifs]

        # Check which features exist
        labels["feature_path"] = labels.apply(self._create_feature_path, axis=1)
        existing_features = labels["feature_path"].apply(lambda p: Path(p).exists())
        print(f"Features that already exist: {len(labels[existing_features])}/{len(labels)}")
        labels = labels[~existing_features]

        # Create the labeled dataset
        [
            self._create_labeled_data_instance(row)
            for row in tqdm(
                zip(
                    labels["tif_path"],
                    labels["feature_path"],
                    labels[CROP_PROB],
                    labels[SUBSET],
                    labels[START],
                    labels[END],
                    labels[LON],
                    labels[LAT],
                    labels[DEST_TIF],
                ),
                total=len(labels),
            )
        ]
