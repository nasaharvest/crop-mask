from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional, Tuple
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
    exporter: EarthEngineExporter = EarthEngineExporter()

    data_folder: Path = Path(__file__).parent.parent.parent / "data"

    def export_earth_engine_data(self):
        raise NotImplementedError

    @staticmethod
    def is_output_folder_ready(output_folder: Path, check_if_folder_empty=False):
        if output_folder.exists():
            if check_if_folder_empty:
                is_empty = not any(output_folder.iterdir())
                return is_empty
            else:
                logger.warning(f"{output_folder} already exists skipping for {output_folder.stem}")
                return False
        elif output_folder.parent.exists():
            logger.info(f"Creating directory: {output_folder}")
            output_folder.mkdir()
            return True
        else:
            logger.warning(
                f"{output_folder.parent} does not exist locally, use `dvc pull data/{output_folder.parent.stem}` to download latest"
            )
            return False


@dataclass
class LabeledDataset(Dataset):
    dataset: str = ""
    country: str = ""

    # Raw label parameters
    raw_labels: Tuple[RawLabels, ...] = ()

    # Process parameters
    processors: Tuple[Processor, ...] = ()
    labels_extension: str = ".geojson"

    # Engineer parameters
    is_global: bool = False
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
    num_timesteps: int = 12

    def __post_init__(self):
        raw_dir = self.data_folder / "raw"
        self.raw_labels_dir = raw_dir / self.dataset
        self.raw_images_dir = raw_dir / self.sentinel_dataset
        self.labels_path = self.data_folder / "processed" / (self.dataset + self.labels_extension)
        self.features_dir = self.data_folder / "features" / self.dataset

    def create_pickled_labeled_dataset(self):
        if self.is_output_folder_ready(self.features_dir):
            Engineer(
                sentinel_files_path=self.raw_images_dir,
                labels_path=self.labels_path,
                features_path=self.features_dir,
            ).create_pickled_labeled_dataset(
                is_global=self.is_global,
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
        total_days = timedelta(days=self.num_timesteps * self.days_per_timestep)
        processed_label_list = [p.process(self.raw_labels_dir, total_days) for p in self.processors]
        if self.labels_path.suffix == ".geojson" or self.labels_path.suffix == ".csv":
            labels = pd.concat(processed_label_list)
            labels["Country"] = self.country
            if self.labels_path.suffix == ".geojson":
                labels.to_file(self.labels_path, driver="GeoJSON")
            elif self.labels_path.suffix == ".csv":
                labels.to_csv(self.labels_path, index=False)
        elif self.labels_path.suffix == ".nc":
            labels = processed_label_list[0]
            labels.to_netcdf(self.labels_path)

    def download_raw_labels(self):
        if len(self.raw_labels) == 0:
            logger.warning(f"No raw labels set for {self.dataset}")
        elif self.is_output_folder_ready(self.raw_labels_dir):
            for label in self.raw_labels:
                label.download_file(output_folder=self.raw_labels_dir)

    def export_earth_engine_data(self):
        if self.is_output_folder_ready(self.raw_images_dir):
            self.exporter.export_for_labels(
                labels_path=self.labels_path,
                sentinel_dataset=self.sentinel_dataset,
                output_folder=self.raw_images_dir,
                num_labelled_points=None,
                monitor=False,
                checkpoint=True,
            )


@dataclass
class UnlabeledDataset(Dataset):
    def __post_init__(self):
        self.raw_images_dir = self.data_folder / "raw" / self.sentinel_dataset

    def export_earth_engine_data(self):
        if self.is_output_folder_ready(self.raw_images_dir):
            self.exporter.export_for_region(
                sentinel_dataset=self.sentinel_dataset,
                output_folder=self.raw_images_dir,
                monitor=False,
                checkpoint=True,
                metres_per_polygon=None,
                fast=False,
            )
