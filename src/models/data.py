from pathlib import Path
import numpy as np
import pickle
import random
import logging

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from src.ETL.constants import BANDS

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
        probability_threshold: float,
        remove_b1_b10: bool,
        cache: bool,
        upsample: bool,
        noise_factor: bool,
        normalizing_dict: Optional[Dict] = None,
        target_bbox: Optional[BoundingBox] = None,
        is_local_only: bool = False,
        timesteps: Optional[List[int]] = None,
    ) -> None:

        logger.info(f"Initializating {subset} CropDataset")
        # ----------------------------------------------------------------------
        # Check dataset arguments
        # ----------------------------------------------------------------------
        if not data_folder.exists():
            raise FileNotFoundError(f"{data_folder} does not exist")

        assert subset in ["training", "validation", "testing"]

        # ----------------------------------------------------------------------
        # Set up dataset parameters
        # ----------------------------------------------------------------------
        self.probability_threshold = probability_threshold
        self.target_bbox = target_bbox

        self.remove_b1_b10 = remove_b1_b10

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None

        self.is_local_only = is_local_only

        # this is kept at False in case caching = True. It should be
        # changed to the input noise argument at the end of the
        # init function
        self.noise_factor = 0
        self.pickle_files: List[Path] = []
        self.local_crop_pickle_files: List[Path] = []
        self.local_non_crop_pickle_files: List[Path] = []
        self.global_crop_pickle_files: List[Path] = []
        self.global_non_crop_pickle_files: List[Path] = []
        self.normalizing_dict: Optional[Dict] = normalizing_dict
        self.num_timesteps: Optional[List[int]] = timesteps
        self.cache = False

        # ----------------------------------------------------------------------
        # Load in pickle files from dataset
        # ----------------------------------------------------------------------
        all_pickle_files: List[Path] = []
        for dataset in datasets:
            features_dir = dataset.get_path(DataDir.FEATURES_DIR, root_data_folder=data_folder)
            if not features_dir.exists():
                logger.warning(f"{features_dir} does not exist, skipping")
                continue
            pickle_files = self.load_pickle_files(
                features_dir=features_dir,
                subset_name=subset,
            )
            all_pickle_files += pickle_files
            logger.info(f"{dataset.dataset} - {subset}: found {len(pickle_files)} pickle files")

        # ----------------------------------------------------------------------
        # Determine if we need to go through the pickle files to either:
        # - fiter out non local files
        # - calculate the normalizing dict
        # - set the number of timesteps
        # - figure out which pickle files are local/global for upsampling
        # ----------------------------------------------------------------------
        go_through_pickle_files = (
            is_local_only or upsample or self.normalizing_dict is None or self.num_timesteps is None
        )
        if go_through_pickle_files:
            normalizing_dict_interim = {"n": 0}
            num_timesteps_set = set()
            for p in tqdm(all_pickle_files):
                with p.open("rb") as f:
                    datainstance = pickle.load(f)

                # Check if pickle file should be added to CropDataset
                is_local = datainstance.isin(self.target_bbox)
                is_crop = datainstance.crop_probability > self.probability_threshold

                if is_local and is_crop:
                    self.local_crop_pickle_files.append(p)
                elif is_local and not is_crop:
                    self.local_non_crop_pickle_files.append(p)
                elif not is_local and is_crop:
                    self.global_crop_pickle_files.append(p)
                elif not is_local and not is_crop:
                    self.global_non_crop_pickle_files.append(p)

                if is_local_only and not is_local:
                    continue

                self.pickle_files.append(p)
                labelled_array = datainstance.labelled_array
                if self.normalizing_dict is None:
                    self._update_normalizing_values(normalizing_dict_interim, labelled_array)
                if self.num_timesteps is None:
                    num_timesteps_set.add(labelled_array.shape[0])

            if self.normalizing_dict is None:
                self.normalizing_dict = self._calculate_normalizing_dict(
                    norm_dict=normalizing_dict_interim
                )
            if self.num_timesteps is None:
                self.num_timesteps = list(num_timesteps_set)

        # ----------------------------------------------------------------------
        # Check pickle files are loaded
        # ----------------------------------------------------------------------
        if len(self.pickle_files) == 0:
            pkl_file_type = "local" if is_local_only else ""
            dataset_names = [d.dataset for d in datasets]
            raise ValueError(f"No {pkl_file_type} {subset} pkl files found in {dataset_names}")

        # ----------------------------------------------------------------------
        # Upsample local crop/non-crop to have equal number of files
        # ----------------------------------------------------------------------
        if upsample:
            print(f"BEFORE UPSAMPLING: pickle_files: {len(self.pickle_files)}")

            crop = len(self.local_crop_pickle_files)
            non_crop = len(self.local_non_crop_pickle_files)
            crop_str = f"crop: {crop}"
            non_crop_str = f"non-crop: {non_crop}"

            if crop == 0 or non_crop == 0:
                print(f"WARNING: local {subset} cannot upsample: {crop_str} and {non_crop_str}")
            elif crop > non_crop:
                print(f"Upsampling: local {subset} {non_crop_str} to {crop_str}")
                while crop > non_crop:
                    self.pickle_files.append(random.choice(self.local_non_crop_pickle_files))
                    non_crop += 1
            elif crop < non_crop:
                print(f"Upsampling: local {subset} {crop_str} to {non_crop_str}")
                while crop < non_crop:
                    self.pickle_files.append(random.choice(self.local_crop_pickle_files))
                    crop += 1

            print(f"AFTER UPSAMPLING: pickle_files: {len(self.pickle_files)}")

        # ----------------------------------------------------------------------
        # Cache dataset if necessary
        # ----------------------------------------------------------------------
        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache
        # we only save the noise attribute after the arrays have been cached, to
        # ensure the saved arrays are the noiseless ones
        self.noise_factor = noise_factor

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
    def _calculate_normalizing_dict(
        norm_dict: Dict[str, Union[np.ndarray, int]]
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
    def load_pickle_files(
        features_dir: Path, subset_name: str, limit: Optional[int] = None, file_suffix: str = "pkl"
    ) -> List[Path]:

        pickle_files_dir = features_dir / subset_name
        if not pickle_files_dir.exists():
            logger.warning(
                f"Directory: {pickle_files_dir} not found. Use command: "
                f"`dvc pull` to get the latest data."
            )
            pickle_files = []
        else:
            pickle_files = list(pickle_files_dir.glob(f"*.{file_suffix}"))
            if limit and limit < len(pickle_files):
                pickle_files = pickle_files[:limit]

        return pickle_files

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
            for i in tqdm(range(len(self))):
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

        if self.remove_bands:
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

    @property
    def original_size(self):
        local_size = len(self.local_crop_pickle_files) + len(self.local_non_crop_pickle_files)
        global_size = len(self.global_crop_pickle_files) + len(self.global_non_crop_pickle_files)
        if self.is_local_only:
            return local_size
        elif self.is_global_only:
            return global_size
        else:
            return local_size + global_size

    @property
    def crop_percentage(self):
        if self.original_size == 0:
            return 0

        total_crop = 0
        if self.is_local_only:
            total_crop = len(self.local_crop_pickle_files)
        elif self.is_global_only:
            total_crop = len(self.global_crop_pickle_files)
        else:
            total_crop = len(self.local_crop_pickle_files) + len(self.global_crop_pickle_files)

        return round(float(total_crop / self.original_size), 4)
