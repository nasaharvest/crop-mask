from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import logging

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from src.ETL.constants import CROP_PROB, FEATURE_PATH, LAT, LON, START, END, MONTHS

from typing import cast, Tuple, Optional, List, Dict, Union
from src.ETL.boundingbox import BoundingBox

logger = logging.getLogger(__name__)


class CropDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        subset: str,
        cache: bool,
        upsample: bool,
        target_bbox: BoundingBox,
        wandb_logger,
        start_month: str = "April",
        probability_threshold: float = 0.5,
        normalizing_dict: Optional[Dict] = None,
        up_to_year: Optional[int] = None,
    ) -> None:

        df = df.copy()

        if subset == "training" and up_to_year is not None:
            df = df[pd.to_datetime(df[START]).dt.year <= up_to_year]

        self.start_month_index = MONTHS.index(start_month)

        df["is_crop"] = df[CROP_PROB] >= probability_threshold
        df["is_local"] = (
            (df[LAT] >= target_bbox.min_lat)
            & (df[LAT] <= target_bbox.max_lat)
            & (df[LON] >= target_bbox.min_lon)
            & (df[LON] <= target_bbox.max_lon)
        )

        local_crop = len(df[df["is_local"] & df["is_crop"]])
        local_non_crop = len(df[df["is_local"] & ~df["is_crop"]])
        local_difference = np.abs(local_crop - local_non_crop)

        if wandb_logger:
            to_log: Dict[str, Union[float, int]] = {}
            if df["is_local"].any():
                to_log[f"local_{subset}_original_size"] = len(df[df["is_local"]])
                to_log[f"local_{subset}_crop_percentage"] = round(
                    local_crop / len(df[df["is_local"]]), 4
                )

            if not df["is_local"].all():
                to_log[f"global_{subset}_original_size"] = len(df[~df["is_local"]])
                to_log[f"global_{subset}_crop_percentage"] = round(
                    len(df[~df["is_local"] & df["is_crop"]]) / len(df[~df["is_local"]]), 4
                )

            if upsample:
                to_log[f"{subset}_upsampled_size"] = len(df) + local_difference

            wandb_logger.experiment.config.update(to_log)

        if upsample:
            if local_crop > local_non_crop:
                arrow = "<-"
                df = df.append(
                    df[df["is_local"] & ~df["is_crop"]].sample(
                        n=local_difference, replace=True, random_state=42
                    ),
                    ignore_index=True,
                )
            elif local_crop < local_non_crop:
                arrow = "->"
                df = df.append(
                    df[df["is_local"] & df["is_crop"]].sample(
                        n=local_difference, replace=True, random_state=42
                    ),
                    ignore_index=True,
                )

            print(f"Upsampling: local crop{arrow}non-crop: {local_crop}{arrow}{local_non_crop}")

        self.normalizing_dict: Dict = (
            normalizing_dict
            if normalizing_dict
            else self._calculate_normalizing_dict(df[FEATURE_PATH].to_list())
        )

        self.df = df

        # Set parameters needed for __getitem__
        self.probability_threshold = probability_threshold
        self.target_bbox = target_bbox
        self.num_timesteps = self._compute_num_timesteps(start_col=df[START], end_col=df[END])

        # Cache dataset if necessary
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.cache = False
        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache

    def _compute_num_timesteps(self, start_col: pd.Series, end_col: pd.Series) -> List[int]:
        df_start_date = pd.to_datetime(start_col).apply(
            lambda dt: dt.replace(month=self.start_month_index + 1)
        )
        df_candidate_end_date = df_start_date.apply(lambda dt: dt.replace(year=dt.year + 1))
        df_data_end_date = pd.to_datetime(end_col)
        df_end_date = pd.DataFrame([df_data_end_date, df_candidate_end_date]).min(axis=0)
        # Pick min available end date
        timesteps = (
            ((df_end_date - df_start_date) / np.timedelta64(1, "M")).round().unique().astype(int)
        )
        return [int(t) for t in timesteps]

    @staticmethod
    def _update_normalizing_values(
        norm_dict: Dict[str, Union[np.ndarray, int]], array: np.ndarray
    ) -> None:
        # given an input array of shape [timesteps, bands]
        # update the normalizing dict
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # https://www.johndcook.com/blog/standard_deviation/
        if array is None:
            raise ValueError("Array is None")

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
    def _calculate_normalizing_dict(feature_files: List[str]) -> Dict[str, np.ndarray]:
        norm_dict_interim = {"n": 0}
        for p in tqdm(feature_files, desc="Calculating normalizing_dict"):
            with Path(p).open("rb") as f:
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
        return len(self.df)

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
        assert len(self.df) > 0, "No files to load!"

        output = self[0]
        if isinstance(output, tuple):
            return output[0].shape[1]
        else:
            return output.shape[1]

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

        row = self.df.iloc[index]

        target_file = Path(row[FEATURE_PATH])

        # first, we load up the target file
        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        x = target_datainstance.labelled_array
        x = x[self.start_month_index : self.start_month_index + 12]
        x = self._normalize(x)

        # If x is a partial time series, pad it to full length
        max_timesteps = max(self.num_timesteps)
        if x.shape[0] < max_timesteps:
            x = np.concatenate([x, np.full((max_timesteps - x.shape[0], x.shape[1]), np.nan)])

        crop_int = int(row["is_crop"])
        is_global = int(not row["is_local"])

        return (
            torch.from_numpy(x).float(),
            torch.tensor(crop_int).float(),
            torch.tensor(is_global).float(),
        )
