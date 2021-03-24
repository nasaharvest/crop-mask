from dataclasses import dataclass
from typing import Callable, Optional, Union
from .engineer import Engineer


@dataclass
class Dataset:
    dataset: str
    sentinel_dataset: str

    # Engineer parameters
    crop_probability: Union[float, Callable]
    labels_file: str = "data.geojson"
    is_global: bool = False
    is_maize: bool = False
    crop_type_func: Optional[Callable] = None
    val_set_size: float = 0.1
    test_set_size: float = 0.1
    nan_fill: float = 0.0
    max_nan_ratio: float = 0.3
    checkpoint: bool = True
    add_ndvi: bool = True
    add_ndwi: bool = False
    include_extended_filenames: bool = True
    calculate_normalizing_dict: bool = True
    days_per_timestep: int = 30

    def create_pickled_labeled_dataset(self):
        return Engineer(
            dataset=self.dataset,
            sentinel_dataset=self.sentinel_dataset,
            labels_file=self.labels_file,
        ).create_pickled_labeled_dataset(
            crop_probability=self.crop_probability,
            is_global=self.is_global,
            is_maize=self.is_maize,
            crop_type_func=self.crop_type_func,
            val_set_size=self.val_set_size,
            test_set_size=self.test_set_size,
            nan_fill=self.nan_fill,
            max_nan_ratio=self.max_nan_ratio,
            checkpoint=self.checkpoint,
            add_ndvi=self.add_ndvi,
            add_ndwi=self.add_ndwi,
            include_extended_filenames=self.include_extended_filenames,
            calculate_normalizing_dict=self.calculate_normalizing_dict,
            days_per_timestep=self.days_per_timestep,
        )
