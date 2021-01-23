from .boundingbox import BoundingBox
import numpy as np
from dataclasses import dataclass
from typing import Type, Union


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
class GeoWikiDataInstance(BaseDataInstance):
    crop_probability: float


@dataclass
class KenyaNonCropDataInstance(BaseDataInstance):
    is_crop: bool = False


@dataclass
class PVKenyaDataInstance(BaseDataInstance):
    crop_label: str
    crop_int: int


@dataclass
class KenyaOneAcreFundDataInstance(BaseDataInstance):
    is_maize: bool = True


@dataclass
class DatasetMetadata:
    name: str
    instance: Type[
        Union[
            GeoWikiDataInstance,
            KenyaOneAcreFundDataInstance,
            KenyaNonCropDataInstance,
            PVKenyaDataInstance,
        ]
    ]


# Available datasets
GeoWiki = DatasetMetadata(name="geowiki_landcover_2017", instance=GeoWikiDataInstance)
KenyaNonCrop = DatasetMetadata(name="kenya_non_crop", instance=KenyaNonCropDataInstance)
KenyaOAF = DatasetMetadata(name="one_acre_fund_kenya", instance=KenyaOneAcreFundDataInstance)
KenyaPV = DatasetMetadata(name="plant_village_kenya", instance=PVKenyaDataInstance)

all_datasets = [GeoWiki, KenyaNonCrop, KenyaOAF, KenyaPV]
