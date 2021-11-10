from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import random
import logging

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from src.ETL.constants import BANDS, CROP_PROB, FEATURE_PATH, LAT, LON, SUBSET, START, END, IS_LOCAL

from typing import cast, Tuple, Optional, List, Dict, Union
from src.ETL.dataset import LabeledDataset, DataDir
from src.ETL.ee_boundingbox import BoundingBox

logger = logging.getLogger(__name__)


class CropDataset(Dataset):

    bands_to_remove = ["B1", "B10"]

    def __init__(
        self,
        data_folder: Path,
        subset: str,
        datasets: List[LabeledDataset],
        remove_b1_b10: bool,
        cache: bool,
        upsample: bool,
        target_bbox: BoundingBox,
        probability_threshold: float = 0.5,
        normalizing_dict: Optional[Dict] = None,
        is_local_only: bool = False,
        up_to_year: Optional[int] = None,
    ) -> None:
        logger.info(f"Initializating {subset} CropDataset")
        if not data_folder.exists():
            raise FileNotFoundError(f"{data_folder} does not exist")

        df = self._load_df_from_datasets(
            datasets,
            subset=subset,
            up_to_year=up_to_year,
            target_bbox=target_bbox,
            is_local_only=is_local_only,
        )

        self.pickle_files: List[Path] = [Path(p) for p in df[FEATURE_PATH].tolist()]
        self.normalizing_dict: Dict = (
            normalizing_dict
            if normalizing_dict
            else self._calculate_normalizing_dict(self.pickle_files)
        )

        is_crop = df[CROP_PROB] >= probability_threshold
        is_local = df[IS_LOCAL]
        if upsample:
            self.pickle_files += self._upsampled_files(
                local_crop_files=df[is_local & is_crop][FEATURE_PATH].to_list(),
                local_non_crop_files=df[is_local & ~is_crop][FEATURE_PATH].to_list(),
            )

        # Set parameters for logging
        self.original_size: int = len(df)
        self.crop_percentage: float = round(len(df[is_crop]) / len(df), 4)

        # Set parameters needed for __getitem__
        self.probability_threshold = probability_threshold
        self.target_bbox = target_bbox
        self.remove_b1_b10 = remove_b1_b10
        self.num_timesteps = self._compute_num_timesteps(start_col=df[START], end_col=df[END])

        # Cache dataset if necessary
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.cache = False
        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache

    @staticmethod
    def _load_df_from_datasets(
        datasets: List[LabeledDataset],
        subset: str,
        up_to_year: int,
        target_bbox: BoundingBox,
        is_local_only: bool,
    ) -> pd.DataFrame:
        assert subset in ["training", "validation", "testing"]
        df = pd.concat([d.load_labels(fail_if_missing_features=True) for d in datasets])
        df = df[df[SUBSET] == subset]
        if up_to_year is not None:
            df = df[pd.to_datetime(df[START]).dt.year <= up_to_year]

        df[IS_LOCAL] = (
            (df[LAT] >= target_bbox.min_lat)
            & (df[LAT] <= target_bbox.max_lat)
            & (df[LON] >= target_bbox.min_lon)
            & (df[LON] <= target_bbox.max_lon)
        )
        if is_local_only:
            df = df[df[IS_LOCAL]]

        if len(df) == 0:
            raise ValueError(f"No labels for {subset} found")

        return df

    @staticmethod
    def _compute_num_timesteps(start_col: pd.Series, end_col: pd.Series) -> Tuple[int, ...]:
        timesteps = (
            ((pd.to_datetime(end_col) - pd.to_datetime(start_col)) / np.timedelta64(1, "M"))
            .round()
            .unique()
            .astype(int)
        )
        return [int(t) for t in timesteps]

    @staticmethod
    def _upsampled_files(
        local_crop_files: List[str], local_non_crop_files: List[str]
    ) -> List[Path]:
        local_crop = len(local_crop_files)
        local_non_crop = len(local_non_crop_files)
        if local_crop == local_non_crop:
            return []

        if local_crop > local_non_crop:
            arrow = "<-"
            files_to_upsample = local_non_crop_files

        elif local_crop < local_non_crop:
            arrow = "->"
            files_to_upsample = local_crop_files

        print(f"Upsampling: local crop{arrow}non-crop: {local_crop}{arrow}{local_non_crop}")

        resample_amount = abs(local_crop - local_non_crop)
        upsampled_str_files = np.random.choice(
            files_to_upsample,
            size=abs(local_crop - local_non_crop),
            replace=resample_amount > len(files_to_upsample),
        ).tolist()
        return [Path(p) for p in upsampled_str_files]

    @staticmethod
    def _update_normalizing_values(
        norm_dict: Dict[str, Union[np.ndarray, int]], array: np.ndarray
    ) -> None:
        # given an input array of shape [timesteps, bands]
        # update the normalizing dict
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # https://www.johndcook.com/blog/standard_deviation/

        # initialize
        if "mean" not in norm_dict:
            num_bands = array.shape[1]
            norm_dict["mean"] = np.zeros(num_bands)
            norm_dict["M2"] = np.zeros(num_bands)

        for time_idx in range(array.shape[0]):
            norm_dict["n"] += 1
            x = array[time_idx, :]
            delta = x - norm_dict["mean"]
            norm_dict["mean"] += delta / norm_dict["n"]
            norm_dict["M2"] += delta * (x - norm_dict["mean"])

    @staticmethod
    def _calculate_normalizing_dict(pickle_files: List[Path]) -> Dict[str, np.ndarray]:
        norm_dict_interim = {"n": 0}
        for p in tqdm(pickle_files, desc="Calculating normalizing_dict"):
            with p.open("rb") as f:
                labelled_array = pickle.load(f).labelled_array
            CropDataset._update_normalizing_values(norm_dict_interim, labelled_array)

        variance = norm_dict_interim["M2"] / (norm_dict_interim["n"] - 1)
        std = np.sqrt(variance)
        return {"mean": norm_dict_interim["mean"], "std": std}

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]

    def __len__(self) -> int:
        return len(self.pickle_files)

    def to_array(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.x is not None:
            assert self.y is not None
            assert self.weights is not None
            return self.x, self.y, self.weights
        else:
            x_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            weight_list: List[torch.Tensor] = []
            logger.info("Loading data into memory")
            for i in tqdm(range(len(self)), desc="Caching files"):
                x, y, weight = self[i]
                x_list.append(x)
                y_list.append(y)
                weight_list.append(weight)

            return torch.stack(x_list), torch.stack(y_list), torch.stack(weight_list)

    @property
    def num_input_features(self) -> int:

        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"

        output = self[0]
        if isinstance(output, tuple):
            return output[0].shape[1]
        else:
            return output.shape[1]

    def remove_bands(self, x: np.ndarray) -> np.ndarray:
        """This nested function is so that
        _remove_bands can be called from an unitialized
        dataset, speeding things up at inference while still
        keeping the convenience of not having to check if remove
        bands is true all the time.
        """
        if self.remove_b1_b10:
            return self._remove_bands(x)
        else:
            return x

    @classmethod
    def _remove_bands(cls, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands]
        """
        indices_to_remove: List[int] = []
        for band in cls.bands_to_remove:
            indices_to_remove.append(BANDS.index(band))

        bands_index = 1 if len(x.shape) == 2 else 2
        indices_to_keep = [i for i in range(x.shape[bands_index]) if i not in indices_to_remove]
        if len(x.shape) == 2:
            # timesteps, bands
            return x[:, indices_to_keep]
        else:
            # batches, timesteps, bands
            return x[:, :, indices_to_keep]

    @property
    def num_output_classes(self) -> Tuple[int, int]:
        return 1, 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if (self.cache) & (self.x is not None):
            # if we upsample, the caching might not have happened yet
            return (
                cast(torch.Tensor, self.x)[index],
                cast(torch.Tensor, self.y)[index],
                cast(torch.Tensor, self.weights)[index],
            )

        target_file = self.pickle_files[index]

        # first, we load up the target file
        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        is_global = not target_datainstance.isin(self.target_bbox)

        if hasattr(target_datainstance, "crop_probability"):
            crop_int = int(target_datainstance.crop_probability >= self.probability_threshold)
        else:
            logger.error(
                "target_datainstance missing mandatory field crop_probability, "
                "defaulting crop_int to 0"
            )
            crop_int = 0

        x = self.remove_bands(x=self._normalize(target_datainstance.labelled_array))

        # If x is a partial time series, pad it to full length
        max_timesteps = max(self.num_timesteps)
        if x.shape[0] < max_timesteps:
            x = np.concatenate([x, np.full((max_timesteps - x.shape[0], x.shape[1]), np.nan)])

        return (
            torch.from_numpy(x).float(),
            torch.tensor(crop_int).float(),
            torch.tensor(is_global).float(),
        )
