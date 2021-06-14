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

from typing import cast, Tuple, Optional, List, Dict, Sequence, Union
from src.ETL.dataset import LabeledDataset

class ForecasterDataset(Dataset):

    bands_to_remove = ["B1", "B10"]

    def __init__(
        self,
        data_folder: Path,
        subset: str,
        datasets: List[LabeledDataset],
        cache: bool,
        upsample: bool,
        normalizing_dict: Optional[Dict] = None,
    ) -> None:
        self.upsample = upsample

        self.data_folder = data_folder
        self.features_dir = data_folder / "features"

        assert subset in ["training", "validation", "testing"]
        self.subset_name = subset

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None

        files_and_nds: List[Tuple] = []
        for dataset in datasets:
            files_and_nds.append(
                self.load_files_and_normalizing_dicts(
                    dataset.features_dir,
                    self.subset_name,
                )
            )

        if normalizing_dict is not None:
            self.normalizing_dict: Optional[Dict] = normalizing_dict
        else:
            # if no normalizing dict was passed to the consturctor,
            # then we want to make our own
            self.normalizing_dict = self.adjust_normalizing_dict(
                [(len(x[0]), x[1]) for x in files_and_nds]
            )

        pickle_files: List[Path] = []
        for files, _ in files_and_nds:
            pickle_files.extend(files)
        self.pickle_files = pickle_files

        self.cache = False

        self.class_instances: List = []
        if upsample:
            instances_per_class = self.instances_per_class
            max_instances_in_class = max(instances_per_class)

            new_pickle_files: List[Path] = []

            for idx, num_instances in enumerate(instances_per_class):
                if num_instances > 0:
                    new_pickle_files.extend(self.upsample_class(idx, max_instances_in_class))
            self.pickle_files.extend(new_pickle_files)

        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache

    @staticmethod
    def load_files_and_normalizing_dicts(
        features_dir: Path, subset_name: str, file_suffix: str = "pkl"
    ) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]]]:

        pickle_files_dir = features_dir / subset_name
        if not pickle_files_dir.exists():
            logger.warning(
                f"Directory: {pickle_files_dir} not found. Use command: "
                f"`dvc pull` to get the latest data."
            )
            pickle_files = []
        else:
            pickle_files = list(pickle_files_dir.glob(f"*.{file_suffix}"))

        # try loading the normalizing dict. By default, if it exists we will use it
        normalizing_dict_path = features_dir / "normalizing_dict.pkl"
        if normalizing_dict_path.exists():
            with normalizing_dict_path.open("rb") as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None

        return pickle_files, normalizing_dict

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

    @staticmethod
    def adjust_normalizing_dict(
        dicts: Sequence[Tuple[int, Optional[Dict[str, np.ndarray]]]]
    ) -> Optional[Dict[str, np.ndarray]]:

        for _, single_dict in dicts:
            if single_dict is None:
                return None

        dicts = cast(Sequence[Tuple[int, Dict[str, np.ndarray]]], dicts)

        new_total = sum([x[0] for x in dicts])

        new_mean = sum([single_dict["mean"] * length for length, single_dict in dicts]) / new_total

        new_variance = (
            sum(
                [
                    (single_dict["std"] ** 2 + (single_dict["mean"] - new_mean) ** 2) * length
                    for length, single_dict in dicts
                ]
            )
            / new_total
        )

        return {"mean": new_mean, "std": np.sqrt(new_variance)}

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

    def upsample_class(self, class_idx: int, max_instances: int) -> List[Path]:
        """Given a class to upsample and the maximum number of classes,
        update self.pickle_files to reflect the new number of classes
        """
        class_files: List[Path] = []
        for idx, filepath in enumerate(self.pickle_files):
            _, class_int, is_global = self[idx]
            if is_global == 0:
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
    def num_output_classes(self) -> Union[int, Tuple[int, int]]:
        return 1

    @property
    def instances_per_class(self) -> List[int]:

        num_output_classes = self.num_output_classes
        num_local_output_classes = (
            num_output_classes[1] if isinstance(num_output_classes, tuple) else num_output_classes
        )
        if len(self.class_instances) == 0:
            # we set a minimum number of output classes since if its 1,
            # its really 2 (binary)
            instances_per_class = [0] * max(num_local_output_classes, 2)
            for i in range(len(self)):
                _, class_int, is_global = self[i]
                if is_global == 0:
                    instances_per_class[int(class_int)] += 1
            self.class_instances = instances_per_class
        return self.class_instances

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

        x = self.remove_bands(x=self._normalize(target_datainstance.labelled_array))

        return (
            torch.from_numpy(x).float() # 1 pixel
        )
