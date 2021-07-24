from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pickle
import re
import os
import warnings
import xarray as xr
import json

import torch
from torch.utils.data import Dataset

from src.ETL.constants import BANDS
from src.ETL.dataset import LabeledDataset
from src.utils import load_tif

from typing import cast, Tuple, Optional, List, Dict, Sequence, Union

class ForecasterDataset(Dataset):

    """Data Handler that loads satellite data."""
    bands_to_remove = ["B1", "B10"]

    def __init__(
        self,
        data_folder: Path,
        subset: str,
        cache: bool,
        normalizing_dict: Optional[Dict]
    ) -> None:

        self.seq_len = 12

        if normalizing_dict is None:
            try:
                with (Path(data_folder) / "normalizing_dict.json").open() as f:
                    self.normalizing_dict = json.load(f)
                    bands = len(self.normalizing_dict["mean"])
                    self.mean = np.array(self.normalizing_dict["mean"]).reshape(bands, 1, 1)
                    self.std = np.array(self.normalizing_dict["std"]).reshape(bands, 1, 1)
            except:
                self.normalizing_dict = None
        else:
            self.normalizing_dict = normalizing_dict
            bands = len(normalizing_dict["mean"])
            self.mean = np.array(self.normalizing_dict["mean"]).reshape(bands, 1, 1)
            self.std = np.array(self.normalizing_dict["std"]).reshape(bands, 1, 1)

        assert subset in ["training", "validation", "testing"]

        self.subset_name = subset

        if self.subset_name in ['training', 'validation']:
            data_root_subset = Path(data_folder) / "train" 
        elif self.subset_name == 'testing':
            data_root_subset = Path(data_folder) / "test" 
        
        self.nc_files = [ str(i) for i in data_root_subset.glob("*.nc") ]

        distribution = int(len(self.nc_files) * 0.8)

        if self.subset_name == 'training':
            self.nc_files = self.nc_files[:distribution]
        elif self.subset_name == 'validation':
            self.nc_files = self.nc_files[distribution:]

        self.nc_files_size = len(self.nc_files)

        print(f"Using: {self.nc_files_size} for {self.subset_name}")
        print()

        self.seed_is_set = True

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None

        self.cache = False
        
        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.nc_files)

    def to_array(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.x is not None:
            assert self.y is not None
            assert self.weights is not None
            return self.x, self.y, self.weights
        else:
            x_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            weight_list: List[torch.Tensor] = []
            for i in tqdm(range(len(self))):
                x, y, weight = self[i]
                x_list.append(x)
                y_list.append(y)
                weight_list.append(weight)

            return torch.stack(x_list), torch.stack(y_list), torch.stack(weight_list)

    @property
    def num_input_features(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.nc_files) > 0, "No files to load!"

        output = self[0]
        if isinstance(output, tuple):
            return output[0].shape[1]
        else:
            return output.shape[1]

    @property
    def num_timesteps(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.nc_files) > 0, "No files to load!"
        output_tuple = self[0]

        return output_tuple[0].shape[0]

    def remove_bands(self, x: np.ndarray) -> np.ndarray:
        """This nested function is so that
        _remove_bands can be called from an unitialized
        dataset, speeding things up at inference while still
        keeping the convenience of not having to check if remove
        bands is true all the time.
        """

        if self.remove_bands:
            return self._remove_bands(x)
        else:
            return x

    @classmethod
    def _remove_bands(cls, x: np.ndarray) -> np.ndarray:
        indices_to_remove = [BANDS.index(band) for band in cls.bands_to_remove]
        indices_to_keep = [i for i in range(x.shape[1]) if i not in indices_to_remove]

        return x[:, indices_to_keep]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        self.set_seed(index)
        rand_i = np.random.randint(len(self.nc_files))
        tile = xr.open_dataarray(self.nc_files[rand_i]).values

        # take the last 12 months of data only
        tile = tile[tile.shape[0] - 12:]

        assert tile.shape == (self.seq_len, 14, 64, 64)

        tile = self._normalize(tile)

        tile = self.remove_bands(tile)
        assert tile.shape == (self.seq_len, 12, 64, 64)

        return torch.from_numpy(tile).float()
