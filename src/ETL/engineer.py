from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Union
import numpy as np
import logging
import pandas as pd
import pickle

from src.band_calculations import process_bands
from src.ETL.constants import LAT, LON, CROP_PROB, SUBSET, START, END
from src.utils import set_seed, process_filename, load_tif
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

    # should be True if the dataset contains data which will
    # only be used for evaluation (e.g. the TogoEvaluation dataset)
    eval_only: bool = False

    def __post_init__(self):
        set_seed()
        self.geospatial_files = list(self.sentinel_files_path.glob("**/*.tif"))
        self.labels = pd.read_csv(self.labels_path)
        if not mandatory_cols.issubset(set(self.labels.columns)):
            raise ValueError(f"{self.labels_path} is missing one of {mandatory_cols}")
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.normalizing_dict_interim: Dict[str, Union[np.ndarray, int]] = {"n": 0}

    @staticmethod
    def _find_nearest(array, value: float) -> Tuple[float, int]:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    @staticmethod
    def _update_normalizing_values(
        norm_dict: Dict[str, Union[np.ndarray, int]], array: np.ndarray
    ) -> None:
        # given an input array of shape [timesteps, bands]
        # update the normalizing dict
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # https://www.johndcook.com/blog/standard_deviation/
        num_bands = array.shape[1]

        # initialize
        if "mean" not in norm_dict:
            norm_dict["mean"] = np.zeros(num_bands)
            norm_dict["M2"] = np.zeros(num_bands)

        for time_idx in range(array.shape[0]):
            norm_dict["n"] += 1

            x = array[time_idx, :]

            delta = x - norm_dict["mean"]
            norm_dict["mean"] += delta / norm_dict["n"]
            norm_dict["M2"] += delta * (x - norm_dict["mean"])

    def _update_batch_normalizing_values(
        self, norm_dict: Dict[str, Union[np.ndarray, int]], array: np.ndarray
    ) -> None:

        assert len(array.shape) == 3, "Expected array of shape [batch, timesteps, bands]"

        for idx in range(array.shape[0]):
            subarray = array[idx, :, :]
            self._update_normalizing_values(norm_dict, subarray)

    def _calculate_normalizing_dict(
        self, norm_dict: Dict[str, Union[np.ndarray, int]]
    ) -> Optional[Dict[str, np.ndarray]]:

        if "mean" not in norm_dict:
            logger.warning(
                "No normalizing dict calculated! Make sure to call _update_normalizing_values"
            )
            return None

        variance = norm_dict["M2"] / (norm_dict["n"] - 1)
        std = np.sqrt(variance)
        return {"mean": norm_dict["mean"], "std": std}

    @staticmethod
    def distance(lat1, lon1, lat2, lon2):
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

    def _create_labeled_data_instance(
        self,
        path_to_file: Path,
        calculate_normalizing_dict: bool,
        start_date: datetime,
        end_date: datetime,
        days_per_timestep: int,
    ) -> Optional[CropDataInstance]:
        r"""
        Return a tuple of np.ndarrays of shape [n_timesteps, n_features] for
        1) the anchor (labelled)
        """

        da = load_tif(path_to_file, days_per_timestep=days_per_timestep, start_date=start_date)

        # first, we find the label encompassed within the da
        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        overlap = self.labels[
            (self.labels[LON] <= max_lon)
            & (self.labels[LON] >= min_lon)
            & (self.labels[LAT] <= max_lat)
            & (self.labels[LAT] >= min_lat)
            & (self.labels[START] == str(start_date.date()))
            & (self.labels[END] == str(end_date.date()))
        ]
        if len(overlap) == 0:
            return None

        if len(overlap) == 1:
            row = overlap.iloc[0]
            idx = overlap.index[0]
        else:
            mean_lat = (min_lat + max_lat) / 2
            mean_lon = (min_lon + max_lon) / 2
            dist = self.distance(mean_lat, mean_lon, overlap[LAT], overlap[LON])
            idx = dist.idxmin()
            row = overlap.loc[dist.idxmin()]

        self.labels = self.labels.drop(idx)

        if row[CROP_PROB] == 0.5:
            logger.info("Skipping row because crop_probability is 0.5")
            return None

        closest_lon, _ = self._find_nearest(da.x, row[LON])
        closest_lat, _ = self._find_nearest(da.y, row[LAT])

        labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values
        labelled_array = process_bands(
            labelled_np,
            nan_fill=self.nan_fill,
            max_nan_ratio=self.max_nan_ratio,
            add_ndvi=self.add_ndvi,
            add_ndwi=self.add_ndwi,
        )

        if (row[SUBSET] != "testing") and calculate_normalizing_dict:
            self._update_normalizing_values(self.normalizing_dict_interim, labelled_array)

        if labelled_array is None:
            return None

        return CropDataInstance(
            crop_probability=row[CROP_PROB],
            instance_lat=closest_lat,
            instance_lon=closest_lon,
            label_lat=row[LAT],
            label_lon=row[LON],
            labelled_array=labelled_array,
            data_subset=row[SUBSET],
            source_file=path_to_file.stem,
            start_date_str=row[START],
            end_date_str=row[END],
        )

    def create_pickled_labeled_dataset(
        self,
        checkpoint: bool = True,
        include_extended_filenames: bool = True,
        calculate_normalizing_dict: bool = True,
        days_per_timestep: int = 30,
    ):
        logger.info(f"Creating pickled labeled dataset: {self.save_dir}")
        for file_path in tqdm(self.geospatial_files):
            file_info = process_filename(
                file_path.name, include_extended_filenames=include_extended_filenames
            )

            if file_info is None:
                continue

            identifier, start_date, end_date = file_info

            filename = f"{identifier}_{str(start_date.date())}_{str(end_date.date())}"

            if checkpoint:
                # we check if the file has already been written
                if (
                    (self.save_dir / "validation" / f"{filename}.pkl").exists()
                    or (self.save_dir / "training" / f"{filename}.pkl").exists()
                    or (self.save_dir / "testing" / f"{filename}.pkl").exists()
                ):
                    continue

            instance = self._create_labeled_data_instance(
                file_path,
                calculate_normalizing_dict=calculate_normalizing_dict,
                start_date=start_date,
                end_date=end_date,
                days_per_timestep=days_per_timestep,
            )
            if instance is not None:
                subset_path = self.save_dir / instance.data_subset
                subset_path.mkdir(exist_ok=True)
                save_path = subset_path / f"{filename}.pkl"
                with save_path.open("wb") as f:
                    pickle.dump(instance, f)

        if calculate_normalizing_dict:
            normalizing_dict = self._calculate_normalizing_dict(
                norm_dict=self.normalizing_dict_interim
            )

            if normalizing_dict is not None:
                save_path = self.save_dir / "normalizing_dict.pkl"
                with save_path.open("wb") as f:
                    pickle.dump(normalizing_dict, f)
            else:
                logger.warning("No normalizing dict calculated!")
