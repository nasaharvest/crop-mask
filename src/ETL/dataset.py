from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from .engineer import Engineer
from .processor import Processor
import pandas as pd


@dataclass
class Dataset:
    dataset: str
    sentinel_dataset: str

    data_folder: Path = Path(__file__).parent.parent.parent / "data"

    # Process parameters
    processors: Tuple[Processor, ...] = ()
    labels_file: str = "data.geojson"

    # Engineer parameters
    crop_probability: Union[float, Callable] = 0.0
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

    def __post_init__(self):
        self.raw_files_path = self.data_folder / "raw" / self.dataset
        self.sentinel_files_path = self.data_folder / "raw" / self.sentinel_dataset
        self.labels_path = self.data_folder / "processed" / self.dataset / self.labels_file
        self.features_path = self.data_folder / "features" / self.dataset

    def create_pickled_labeled_dataset(self):
        return Engineer(
            sentinel_files_path=self.sentinel_files_path,
            labels_path=self.labels_path,
            features_path=self.features_path,
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

    def process_labels(self):
        processed_label_list = [p.process(self.raw_files_path) for p in self.processors]
        self.labels_path.parent.mkdir(exist_ok=True, parents=True)
        if self.labels_path.suffix == ".geojson":
            labels = pd.concat(processed_label_list)
            labels.to_file(self.labels_path, driver="GeoJSON")
        elif self.labels_path.suffix == ".nc":
            labels = processed_label_list[0]
            labels.to_netcdf(self.labels_path)
