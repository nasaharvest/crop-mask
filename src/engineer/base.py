from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from pathlib import Path
import numpy as np
import logging
import pickle
from tqdm import tqdm

from typing import Dict, List, Tuple, Optional, Union
from src.utils import set_seed, process_filename
from src.data_classes import BaseDataInstance

logger = logging.getLogger(__name__)


class BaseEngineer(ABC):
    r"""Combine earth engine sentinel data
    and geowiki landcover 2017 data to make
    numpy arrays which can be input into the
    machine learning model
    """

    sentinel_dataset: str
    dataset: str

    # should be True if the dataset contains data which will
    # only be used for evaluation (e.g. the TogoEvaluation dataset)
    eval_only: bool = False

    def __init__(self, data_folder: Path) -> None:
        set_seed()
        self.data_folder = data_folder
        self.geospatial_files = self.get_geospatial_files(data_folder)
        self.labels = self.read_labels(data_folder)

        self.savedir = self.data_folder / "features" / self.dataset
        self.savedir.mkdir(exist_ok=True, parents=True)

        self.normalizing_dict_interim: Dict[str, Union[np.ndarray, int]] = {"n": 0}

    def get_geospatial_files(self, data_folder: Path) -> List[Path]:
        sentinel_files = data_folder / "raw" / self.sentinel_dataset
        return list(sentinel_files.glob("*.tif"))

    @staticmethod
    @abstractmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def find_nearest(array, value: float) -> Tuple[float, int]:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    @staticmethod
    def update_normalizing_values(
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

    def update_batch_normalizing_values(
        self, norm_dict: Dict[str, Union[np.ndarray, int]], array: np.ndarray
    ) -> None:

        assert len(array.shape) == 3, "Expected array of shape [batch, timesteps, bands]"

        for idx in range(array.shape[0]):
            subarray = array[idx, :, :]
            self.update_normalizing_values(norm_dict, subarray)

    def calculate_normalizing_dict(
        self, norm_dict: Dict[str, Union[np.ndarray, int]]
    ) -> Optional[Dict[str, np.ndarray]]:

        if "mean" not in norm_dict:
            logger.warning(
                "No normalizing dict calculated! Make sure to call update_normalizing_values"
            )
            return None

        variance = norm_dict["M2"] / (norm_dict["n"] - 1)
        std = np.sqrt(variance)
        return {"mean": norm_dict["mean"], "std": std}

    @abstractmethod
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
    ) -> Optional[BaseDataInstance]:
        raise NotImplementedError

    def engineer(
        self,
        val_set_size: float = 0.1,
        test_set_size: float = 0.1,
        nan_fill: float = 0.0,
        max_nan_ratio: float = 0.3,
        checkpoint: bool = True,
        add_ndvi: bool = True,
        add_ndwi: bool = False,
        include_extended_filenames: bool = True,
        calculate_normalizing_dict: bool = True,
        days_per_timestep: int = 30,
    ):
        for file_path in tqdm(self.geospatial_files):

            file_info = process_filename(
                file_path.name, include_extended_filenames=include_extended_filenames
            )

            if file_info is None:
                continue

            identifier, start_date, end_date = file_info

            file_name = f"{identifier}_{str(start_date.date())}_{str(end_date.date())}"

            if checkpoint:
                # we check if the file has already been written
                if (
                    (self.savedir / "validation" / f"{file_name}.pkl").exists()
                    or (self.savedir / "training" / f"{file_name}.pkl").exists()
                    or (self.savedir / "testing" / f"{file_name}.pkl").exists()
                ):
                    continue

            if self.eval_only:
                data_subset = "testing"
            else:
                random_float = np.random.uniform()
                # we split into (val, test, train)
                if random_float <= (val_set_size + test_set_size):
                    if random_float <= val_set_size:
                        data_subset = "validation"
                    else:
                        data_subset = "testing"
                else:
                    data_subset = "training"

            instance = self.process_single_file(
                file_path,
                nan_fill=nan_fill,
                max_nan_ratio=max_nan_ratio,
                add_ndvi=add_ndvi,
                add_ndwi=add_ndwi,
                calculate_normalizing_dict=calculate_normalizing_dict,
                start_date=start_date,
                days_per_timestep=days_per_timestep,
                is_test=True if data_subset == "testing" else False,
            )
            if instance is not None:
                subset_path = self.savedir / data_subset
                subset_path.mkdir(exist_ok=True)
                save_path = subset_path / f"{file_name}.pkl"
                with save_path.open("wb") as f:
                    pickle.dump(instance, f)

        if calculate_normalizing_dict:
            normalizing_dict = self.calculate_normalizing_dict(
                norm_dict=self.normalizing_dict_interim
            )

            if normalizing_dict is not None:
                save_path = self.savedir / "normalizing_dict.pkl"
                with save_path.open("wb") as f:
                    pickle.dump(normalizing_dict, f)
            else:
                logger.warning("No normalizing dict calculated!")
