from pathlib import Path
import numpy as np
import pickle
import random
import math
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
        is_global_only: bool = False,
    ) -> None:

        logger.info(f"Initializating {subset} CropDataset")
        self.probability_threshold = probability_threshold
        self.target_bbox = target_bbox

        if is_local_only and is_global_only:
            raise ValueError("is_local_only and is_global_only cannot both be True")

        self.is_local_only = is_local_only
        self.is_global_only = is_global_only

        assert subset in ["training", "validation", "testing"]

        self.remove_b1_b10 = remove_b1_b10

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None

        # this is kept at False in case caching = True. It should be
        # changed to the input noise argument at the end of the
        # init function
        self.noise_factor = 0

        all_pickle_files: List[Path] = []
        for dataset in datasets:
            pickle_files = self.load_pickle_files(
                features_dir=dataset.get_path(DataDir.FEATURES_DIR, root_data_folder=data_folder),
                subset_name=subset,
            )
            all_pickle_files += pickle_files
            logger.info(f"{dataset.dataset} - {subset}: found {len(pickle_files)} pickle files")

        self.pickle_files: List[Path] = []
        normalizing_dict_interim = {"n": 0}

        if normalizing_dict and (not is_local_only and not is_global_only):
            self.pickle_files = all_pickle_files
        else:
            if not normalizing_dict:
                logger.info("Calculating normalizing dict")
            else:
                logger.info("Filtering by local and global")
            for p in tqdm(all_pickle_files):
                with p.open("rb") as f:
                    datainstance = pickle.load(f)

                # Check if pickle file should be added to CropDataset
                is_local = datainstance.isin(self.target_bbox)
                if (
                    (not is_local_only and not is_global_only)
                    or (is_local_only and is_local)
                    or (is_global_only and not is_local)
                ):
                    self.pickle_files.append(p)
                    if not normalizing_dict:
                        labelled_array = datainstance.labelled_array
                        self._update_normalizing_values(normalizing_dict_interim, labelled_array)

        if len(self.pickle_files) == 0:
            local_or_global_only = ""
            if is_local_only:
                local_or_global_only = "local"
            elif is_global_only:
                local_or_global_only = "global"
            raise ValueError(f"No {local_or_global_only} {subset} pkl files found in {datasets}")

        if normalizing_dict:
            self.normalizing_dict: Optional[Dict] = normalizing_dict
        else:
            self.normalizing_dict = self._calculate_normalizing_dict(
                norm_dict=normalizing_dict_interim
            )

        self.cache = False

        self.local_class_instances: List = []
        self.global_class_instances: List = []
        if upsample:
            instances_per_class = self.local_instances_per_class
            max_instances_in_class = max(instances_per_class)

            new_pickle_files: List[Path] = []

            for idx, num_instances in enumerate(instances_per_class):
                if num_instances > 0:
                    new_pickle_files.extend(
                        self.upsample_class(idx, max_instances_in_class, is_local_only=True)
                    )
            self.pickle_files.extend(new_pickle_files)

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

    @property
    def num_timesteps(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"
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

    def upsample_class(
        self, class_idx: int, max_instances: int, is_local_only: bool = True
    ) -> List[Path]:
        """Given a class to upsample and the maximum number of classes,
        update self.pickle_files to reflect the new number of classes
        """
        class_files: List[Path] = []
        for idx, filepath in enumerate(self.pickle_files):
            _, class_int, is_global = self[idx]
            if is_local_only and (is_global == 1):
                continue

            if class_int == class_idx:
                class_files.append(filepath)

        multiplier = max_instances / len(class_files)

        # we will return files which need to be *added* to pickle files
        # multiplier will definitely be >= 1
        fraction_multiplier, int_multiplier = math.modf(multiplier - 1)

        new_files = random.sample(class_files, int(fraction_multiplier * len(class_files)))
        new_files += class_files * int(int_multiplier)
        return new_files

    @property
    def num_output_classes(self) -> Tuple[int, int]:
        return 1, 1

    @property
    def local_instances_per_class(self) -> List[int]:
        if len(self.local_class_instances) == 0:
            self.local_class_instances = self.instances_per_class(self.num_output_classes[1], True)
        return self.local_class_instances

    @property
    def global_instances_per_class(self) -> List[int]:
        if len(self.global_class_instances) == 0:
            self.global_class_instances = self.instances_per_class(
                self.num_output_classes[0], False
            )
        return self.global_class_instances

    def instances_per_class(self, num_local_output_classes, is_local) -> List[int]:
        # we set a minimum number of output classes since if its 1,
        # its really 2 (binary)
        instances_per_class = [0] * max(num_local_output_classes, 2)
        for i in range(len(self)):
            _, class_int, is_global = self[i]
            if is_local and (is_global == 0) or (not is_local and (is_global == 1)):
                instances_per_class[int(class_int)] += 1
        return instances_per_class

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

        return (
            torch.from_numpy(x).float(),
            torch.tensor(crop_int).float(),
            torch.tensor(is_global).float(),
        )

    @property
    def original_size(self):
        local_size = self.local_instances_per_class[0] + self.local_instances_per_class[1]
        global_size = self.global_instances_per_class[0] + self.global_instances_per_class[1]
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

        if self.is_local_only:
            total_crop = self.local_instances_per_class[1]
        elif self.is_global_only:
            total_crop = self.global_instances_per_class[1]
        else:
            total_crop = self.local_instances_per_class[1] + self.global_instances_per_class[1]

        return round(float(total_crop / self.original_size), 4)
