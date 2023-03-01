from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from openmapflow.bbox import BBox
from openmapflow.constants import CLASS_PROB, END, EO_DATA, LAT, LON, MONTHS, START
from torch.utils.data import Dataset
from tqdm import tqdm


class CropDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        subset: str,
        cache: bool,
        upsample: bool,
        target_bbox: BBox,
        wandb_logger,
        start_month: str = "April",
        probability_threshold: float = 0.5,
        input_months: int = 12,
        normalizing_dict: Optional[Dict] = None,
        up_to_year: Optional[int] = None,
    ) -> None:
        df = df.copy()

        if subset == "training" and up_to_year is not None:
            df = df[pd.to_datetime(df[START]).dt.year <= up_to_year]

        self.start_month_index = MONTHS.index(start_month)
        self.input_months = input_months

        df["is_crop"] = df[CLASS_PROB] >= probability_threshold
        df["is_local"] = (
            (df[LAT] >= target_bbox.min_lat)
            & (df[LAT] <= target_bbox.max_lat)
            & (df[LON] >= target_bbox.min_lon)
            & (df[LON] <= target_bbox.max_lon)
        )

        if subset != "training":
            outside_model_bbox = (~df["is_local"]).sum()
            assert outside_model_bbox == 0, (
                f"{outside_model_bbox} points outside model bbox: "
                + f"({df[LAT].min()}, {df[LON].min()}, {df[LAT].max()}, {df[LON].max()})"
            )

        local_crop = len(df[df["is_local"] & df["is_crop"]])
        local_non_crop = len(df[df["is_local"] & ~df["is_crop"]])
        local_difference = np.abs(local_crop - local_non_crop)

        self.num_timesteps = self._compute_num_timesteps(df=df)

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
            if local_crop == local_non_crop:
                print(f"No upsampling: {local_crop} == {local_non_crop}")
            elif local_crop > local_non_crop:
                df = df.append(
                    df[df["is_local"] & ~df["is_crop"]].sample(
                        n=local_difference, replace=True, random_state=42
                    ),
                    ignore_index=True,
                )
                print(f"Upsamplng local non-crop to match crop {local_non_crop} -> {local_crop}")
            elif local_crop < local_non_crop:
                df = df.append(
                    df[df["is_local"] & df["is_crop"]].sample(
                        n=local_difference, replace=True, random_state=42
                    ),
                    ignore_index=True,
                )
                print(f"Upsampling: local crop to non-crop: {local_crop} -> {local_non_crop}")

        self.normalizing_dict: Dict = (
            normalizing_dict
            if normalizing_dict
            else self._calculate_normalizing_dict(df[EO_DATA].to_list())
        )

        self.df = df

        # Set parameters needed for __getitem__
        self.probability_threshold = probability_threshold
        self.target_bbox = target_bbox

        # Cache dataset if necessary
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.cache = False
        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache

    def _compute_num_timesteps(self, df) -> List[int]:
        df_start_date = pd.to_datetime(df[START]).apply(
            lambda dt: dt.replace(month=self.start_month_index + 1)
        )
        df_candidate_end_date = df_start_date.apply(
            lambda dt: dt + relativedelta(months=+self.input_months)
        )
        df_data_end_date = pd.to_datetime(df[END])
        df_end_date = pd.DataFrame({"1": df_data_end_date, "2": df_candidate_end_date}).min(axis=1)
        df["timesteps"] = (
            ((df_end_date - df_start_date) / np.timedelta64(1, "M")).round().astype(int)
        )
        timesteps = df["timesteps"].unique().tolist()
        if len(timesteps) > 1:
            timesteps_w_dataset = (
                df[["dataset", "timesteps"]]
                .groupby("timesteps")
                .agg({"dataset": lambda ds: ",".join(ds.unique())})
            )
            print(
                "WARNING: Datasets have different amounts of timesteps available. "
                + "Forecaster will be used to fill gaps."
                + f"\n{timesteps_w_dataset}"
            )

        return timesteps

    @staticmethod
    def _update_normalizing_values(
        norm_dict: Dict[str, Union[int, Any]], array: np.ndarray
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
    def _calculate_normalizing_dict(
        eo_data_list: List[np.ndarray],
    ) -> Dict[str, Union[int, np.ndarray]]:
        norm_dict_interim = {"n": 0}
        for eo_data in tqdm(eo_data_list, desc="Calculating normalizing_dict"):
            CropDataset._update_normalizing_values(norm_dict_interim, eo_data)

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
            print("Loading data into memory")
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

        x = row[EO_DATA][self.start_month_index : self.start_month_index + self.input_months]
        x = self._normalize(x)

        # If x is a partial time series, pad it to full length
        if x.shape[0] < self.input_months:
            x = np.concatenate([x, np.full((self.input_months - x.shape[0], x.shape[1]), np.nan)])

        crop_int = int(row["is_crop"])
        is_global = int(not row["is_local"])

        return (
            torch.from_numpy(x).float(),
            torch.tensor(crop_int).float(),
            torch.tensor(is_global).float(),
        )
