import numpy as np
from enum import Enum
from .dataset import Dataset


class DatasetName(Enum):
    GeoWiki = "geowiki_landcover_2017"
    KenyaNonCrop = "kenya_non_crop"
    KenyaOAF = "one_acre_fund_kenya"
    KenyaPV = "plant_village_kenya"


def kenya_pv_crop_probability_func(overlap, labels):
    unique_classes = labels.crop_type.unique()
    classes_to_index = {
        crop: idx for idx, crop in enumerate(unique_classes[unique_classes != np.array(None)])
    }
    if overlap.crop_type:
        return float(classes_to_index[overlap.crop_type])
    return None


datasets = [
    Dataset(
        dataset=DatasetName.GeoWiki.value,
        sentinel_dataset="earth_engine_geowiki",
        labels_file="data.nc",
        crop_probability=lambda overlap, _: overlap.mean_sumcrop / 100,
        val_set_size=0.2,
        test_set_size=0,
        is_global=True,
    ),
    Dataset(
        dataset=DatasetName.KenyaNonCrop.value,
        sentinel_dataset="earth_engine_kenya_non_crop",
        crop_probability=0.0,
    ),
    Dataset(
        dataset=DatasetName.KenyaOAF.value,
        sentinel_dataset="earth_engine_one_acre_fund_kenya",
        crop_probability=1.0,
        is_maize=True,
    ),
    Dataset(
        dataset=DatasetName.KenyaPV.value,
        sentinel_dataset="earth_engine_plant_village_kenya",
        crop_probability=kenya_pv_crop_probability_func,
        crop_type_func=lambda overlap: overlap.crop_type,
    ),
]
