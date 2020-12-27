from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
import random
import warnings
import xarray as xr

from typing import Dict, List, Tuple, Optional, Union
from src.exporters.sentinel.cloudfree import BANDS
from src.utils import set_seed
from src.utils import BoundingBox


@dataclass
class BaseDataInstance:
    label_lat: float
    label_lon: float
    instance_lat: float
    instance_lon: float
    labelled_array: np.ndarray

    def isin(self, bounding_box: BoundingBox) -> bool:
        return (
            (self.instance_lon <= bounding_box.max_lon)
            & (self.instance_lon >= bounding_box.min_lon)
            & (self.instance_lat <= bounding_box.max_lat)
            & (self.instance_lat >= bounding_box.min_lat)
        )


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
    def process_filename(
        filename: str, include_extended_filenames: bool
    ) -> Optional[Tuple[str, datetime, datetime]]:
        r"""
        Given an exported sentinel file, process it to get the start
        and end dates of the data. This assumes the filename ends with '.tif'
        """
        date_format = "%Y-%m-%d"

        identifier, start_date_str, end_date_str = filename[:-4].split("_")

        start_date = datetime.strptime(start_date_str, date_format)

        try:
            end_date = datetime.strptime(end_date_str, date_format)
            return identifier, start_date, end_date

        except ValueError:
            if include_extended_filenames:
                end_list = end_date_str.split("-")
                end_year, end_month, end_day = (
                    end_list[0],
                    end_list[1],
                    end_list[2],
                )

                # if we allow extended filenames, we want to
                # differentiate them too
                id_number = end_list[3]
                identifier = f"{identifier}-{id_number}"

                return (
                    identifier,
                    start_date,
                    datetime(int(end_year), int(end_month), int(end_day)),
                )
            else:
                print(f"Unexpected filename {filename} - skipping")
                return None

    @staticmethod
    def load_tif(filepath: Path, start_date: datetime, days_per_timestep: int) -> xr.DataArray:
        r"""
        The sentinel files exported from google earth have all the timesteps
        concatenated together. This function loads a tif files and splits the
        timesteps
        """

        # this mirrors the eo-learn approach
        # also, we divide by 10,000, to remove the scaling factor
        # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
        da = xr.open_rasterio(filepath).rename("FEATURES") / 10000

        da_split_by_time: List[xr.DataArray] = []

        bands_per_timestep = len(BANDS)
        num_bands = len(da.band)

        assert (
            num_bands % bands_per_timestep == 0
        ), f"Total number of bands not divisible by the expected bands per timestep"

        cur_band = 0
        while cur_band + bands_per_timestep <= num_bands:
            time_specific_da = da.isel(band=slice(cur_band, cur_band + bands_per_timestep))
            time_specific_da["band"] = range(bands_per_timestep)
            da_split_by_time.append(time_specific_da)
            cur_band += bands_per_timestep

        timesteps = [
            start_date + timedelta(days=days_per_timestep) * i for i in range(len(da_split_by_time))
        ]

        combined = xr.concat(da_split_by_time, pd.Index(timesteps, name="time"))
        combined.attrs["band_descriptions"] = BANDS

        return combined

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
            print("No normalizing dict calculated! Make sure to call update_normalizing_values")
            return None

        variance = norm_dict["M2"] / (norm_dict["n"] - 1)
        std = np.sqrt(variance)
        return {"mean": norm_dict["mean"], "std": std}

    @staticmethod
    def maxed_nan_to_num(
        array: np.ndarray, nan: float, max_ratio: Optional[float] = None
    ) -> Optional[np.ndarray]:

        if max_ratio is not None:
            num_nan = np.count_nonzero(np.isnan(array))
            if (num_nan / array.size) > max_ratio:
                return None
        return np.nan_to_num(array, nan=nan)

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

    @staticmethod
    def _calculate_difference_index(
        input_array: np.ndarray, num_dims: int, band_1: str, band_2: str
    ) -> np.ndarray:

        if num_dims == 2:
            band_1_np = input_array[:, BANDS.index(band_1)]
            band_2_np = input_array[:, BANDS.index(band_2)]
        elif num_dims == 3:
            band_1_np = input_array[:, :, BANDS.index(band_1)]
            band_2_np = input_array[:, :, BANDS.index(band_2)]
        else:
            raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in true_divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            ndvi = np.where(
                (band_1_np + band_2_np) > 0, (band_1_np - band_2_np) / (band_1_np + band_2_np), 0,
            )
        return np.append(input_array, np.expand_dims(ndvi, -1), axis=-1)

    @classmethod
    def calculate_ndvi(cls, input_array: np.ndarray, num_dims: int = 2) -> np.ndarray:
        r"""
        Given an input array of shape [timestep, bands] or [batches, timesteps, bands]
        where bands == len(BANDS), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b08 - b04) / (b08 + b04)
        """

        return cls._calculate_difference_index(input_array, num_dims, "B8", "B4")

    @classmethod
    def calculate_ndwi(cls, input_array: np.ndarray, num_dims: int = 2) -> np.ndarray:
        r"""
        Given an input array of shape [timestep, bands] or [batches, timesteps, bands]
        where bands == len(BANDS), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b03 - b8A) / (b3 + b8a)
        """
        return cls._calculate_difference_index(input_array, num_dims, "B3", "B8A")

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

            file_info = self.process_filename(
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
                print("No normalizing dict calculated!")
