from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from .engineer import Engineer
from .label_downloader import RawLabels
from .processor import Processor
from .ee_exporter import EarthEngineExporter
import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    sentinel_dataset: str

    # Exporter parameters
    exporter: EarthEngineExporter

    data_folder: Path = Path(__file__).parent.parent.parent / "data"

    def __post_init__(self):
        self.sentinel_files_path = self.data_folder / "raw" / self.sentinel_dataset

    def export_earth_engine_data(self):
        raise NotImplementedError


@dataclass
class LabeledDataset(Dataset):
    dataset: str = ""

    # Raw label parameters
    raw_labels: Tuple[RawLabels, ...] = ()

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

    def download_raw_labels(self):
        if len(self.raw_labels) == 0:
            logger.warning(f"No raw labels set for {self.dataset}")
        else:
            self.raw_files_path.mkdir(parents=True, exist_ok=True)
            for label in self.raw_labels:
                label.download_file(output_folder=self.raw_files_path)

    def export_earth_engine_data(self):
        self.exporter.export_for_labels(
            labels_path=self.labels_path,
            sentinel_dataset=self.sentinel_dataset,
            output_folder=self.sentinel_files_path,
            num_labelled_points=None,
            monitor=False,
            checkpoint=True,
        )


@dataclass
class UnlabeledDataset(Dataset):
    def export_earth_engine_data(self):
        self.exporter.export_for_region(
            sentinel_dataset=self.sentinel_dataset,
            output_folder=self.sentinel_files_path,
            monitor=False,
            checkpoint=True,
            metres_per_polygon=None,
            fast=False,
        )
