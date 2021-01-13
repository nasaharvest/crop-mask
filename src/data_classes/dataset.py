from .boundingbox import BoundingBox
import numpy as np
from dataclasses import dataclass
from typing import Type


@dataclass
class BaseDataInstance:
    label_lat: float
    label_lon: float
    instance_lat: float
    instance_lon: float
    labelled_array: np.ndarray

    def isin(self, bounding_box: BoundingBox) -> bool:
        return (
                (self.instance_lon <= bounding_box.max_lon)
                & (self.instance_lon >= bounding_box.min_lon)
                & (self.instance_lat <= bounding_box.max_lat)
                & (self.instance_lat >= bounding_box.min_lat)
        )

@dataclass
class _GeoWikiDataInstance(BaseDataInstance):
    crop_probability: float


@dataclass
class _KenyaNonCropDataInstance(BaseDataInstance):
    is_crop: bool = False


@dataclass
class _PVKenyaDataInstance(BaseDataInstance):
    crop_label: str
    crop_int: int


@dataclass
class _KenyaOneAcreFundDataInstance(BaseDataInstance):
    is_maize: bool = True


@dataclass
class DatasetMetadata:
    name: str
    instance: Type[BaseDataInstance]


# Available datasets
GeoWiki = DatasetMetadata("geowiki_landcover_2017", _GeoWikiDataInstance)
KenyaNonCrop = DatasetMetadata("kenya_non_crop", _KenyaNonCropDataInstance)
KenyaOAF = DatasetMetadata("one_acre_fund_kenya", _KenyaOneAcreFundDataInstance)
KenyaPV = DatasetMetadata("plant_village_kenya", _PVKenyaDataInstance)

all_datasets = [GeoWiki, KenyaNonCrop, KenyaOAF, KenyaPV]