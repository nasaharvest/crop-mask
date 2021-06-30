from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pickle
import re
import os
import warnings
import xarray as xr

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
        datasets: List[LabeledDataset],
        normalizing_dict: Optional[Dict] = None,
    ) -> None:
        self.normalizing_dict = None
        all_nc_files = [os.path.join(data_folder, i) for i in os.listdir(data_folder) if i.endswith(".nc")]

        print(f"Found: {len(all_nc_files )} tif files")

        train_split_index = int(len(all_nc_files)*0.7)
        val_split_index = int(len(all_nc_files)*0.85)

        assert subset in ["training", "validation", "testing"]

        self.subset_name = subset
        print(self.subset_name)

        if self.subset_name == 'training':
            self.nc_files = all_nc_files [:train_split_index]
        elif self.subset_name == 'validation':
            self.nc_files = all_nc_files [train_split_index:val_split_index]
        elif self.subset_name == 'testing':
            self.nc_files = all_nc_files [val_split_index:]

        print(f"Using: {len(self.nc_files)} for {self.subset_name}")

        self.seed_is_set = True

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None

        self.cache = False
        
        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache

        # files_and_nds: List[Tuple] = []
        # for dataset in datasets:
        #     files_and_nds.append(
        #         self.load_files_and_normalizing_dicts(
        #             dataset.features_dir,
        #             self.subset_name,
        #         )
        #     )

        # if normalizing_dict is not None:
        #     self.normalizing_dict: Optional[Dict] = normalizing_dict
        # else:
        #     self.normalizing_dict = self.adjust_normalizing_dict(
        #         [(len(x[0]), x[1]) for x in files_and_nds]
        #     )

        # pickle_files: List[Path] = []
        # for files, _ in files_and_nds:
        #     pickle_files.extend(files)
        # self.pickle_files = pickle_files

        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    # @staticmethod
    # def load_files_and_normalizing_dicts(
    #     features_dir: Path, subset_name: str, file_suffix: str = "tif"
    # ) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]]]:

    #     curr_dir = str(features_dir)[str(features_dir).rindex('/') + 1:]
        
    #     tif_files_dir = features_dir.parent.parent / "raw" / ("earth_engine_" + {
    #         'geowiki_landcover_2017': 'geowiki',
    #         'Kenya': 'kenya',
    #         'Mali': 'mali',
    #         'Rwanda': 'rwanda',
    #         'Togo': 'togo'
    #     }[curr_dir])

    #     print(tif_files_dir)
    
    #     if not tif_files_dir.exists():
    #         tif_files = []
    #     else:
    #         tif_files = list(tif_files_dir.glob(f"*.{file_suffix}"))

    #     # try loading the normalizing dict. By default, if it exists we will use it
    #     normalizing_dict_path = features_dir / "normalizing_dict.pkl"
    #     if normalizing_dict_path.exists():
    #         with normalizing_dict_path.open("rb") as f:
    #             normalizing_dict = pickle.load(f)
    #     else:
    #         normalizing_dict = None

    #     return tif_files, normalizing_dict

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]

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

    # @staticmethod
    # def adjust_normalizing_dict(
    #     dicts: Sequence[Tuple[int, Optional[Dict[str, np.ndarray]]]]
    # ) -> Optional[Dict[str, np.ndarray]]:

    #     for _, single_dict in dicts:
    #         if single_dict is None:
    #             return None

    #     dicts = cast(Sequence[Tuple[int, Dict[str, np.ndarray]]], dicts)

    #     new_total = sum([x[0] for x in dicts])

    #     new_mean = sum([single_dict["mean"] * length for length, single_dict in dicts]) / new_total

    #     new_variance = (
    #         sum(
    #             [
    #                 (single_dict["std"] ** 2 + (single_dict["mean"] - new_mean) ** 2) * length
    #                 for length, single_dict in dicts
    #             ]
    #         )
    #         / new_total
    #     )

    #     return {"mean": new_mean, "std": np.sqrt(new_variance)}

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
        # """
        # Expects the input to be of shape [timesteps, bands]
        # """
        # indices_to_remove: List[int] = []
        # for band in cls.bands_to_remove:
        #     indices_to_remove.append(BANDS.index(band))

        # bands_index = 1 if len(x.shape) == 2 else 2
        # indices_to_keep = [i for i in range(x.shape[bands_index]) if i not in indices_to_remove]
        # if len(x.shape) == 2:
        #     # timesteps, bands
        #     return x[:, indices_to_keep]
        # else:
        #     # batches, timesteps, bands
        #     return x[:, :, indices_to_keep]
        """
        Expects the input to be of shape [timesteps, bands, size, size]
        """
        indices_to_remove = [BANDS.index(band) for band in cls.bands_to_remove]
        indices_to_keep = [i for i in range(x.shape[1]) if i not in indices_to_remove]

        return x[:, indices_to_keep]

    @staticmethod
    def _calculate_ndvi(input_array: np.ndarray) -> np.ndarray:
        r"""
        Given an input array of shape [timestep, bands] or [batches, timesteps, bands]
        where bands == len(BANDS), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b08 - b04) / (b08 + b04)
        """
        b08 = input_array[:, BANDS.index("B8")]
        b04 = input_array[:, BANDS.index("B4")]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in true_divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            ndvi = np.where((b08 + b04) > 0, (b08 - b04) / (b08 + b04), 0,)
        return ndvi

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        self.set_seed(index)
        rand_i = np.random.randint(len(self.nc_files))
        tile = xr.open_dataarray(self.nc_files[rand_i]).values
        print(tile.shape)
        assert tile.shape == (12, 13, 64, 64)

        ndvi = self._calculate_ndvi(tile)
        assert ndvi.shape == (12, 64, 64)
        
        tile = np.concatenate([tile, np.expand_dims(ndvi, axis=1)], axis=1)
        assert tile.shape == (12, 14, 64, 64)

        tile = self.remove_bands(tile)

        assert tile.shape == (12, 12, 64, 64)
        
        tile_normalized = self._normalize(tile)

        # print(tile_normalized)
        
        return torch.from_numpy(tile_normalized)


        # target_file = self.pickle_files[index]

        # pattern = re.search('.*/[0-9]+_(20[0-9]{2})-([0-9]{2})-([0-9]{2})_20[0-9]{2}-[0-9]{2}-[0-9]{2}\.tif$', str(target_file))
        
        # target_datainstance = load_tif(target_file, datetime(int(pattern.group(1)), int(pattern.group(2)), int(pattern.group(3))), days_per_timestep=30)

        # x = self.remove_bands(x=self._normalize(target_datainstance))

        # return (
        #     torch.from_numpy(x).float() # 1 pixel
        # )
