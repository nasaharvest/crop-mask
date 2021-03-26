from enum import Enum
from src.ETL.dataset import Dataset, Processor


class DatasetName(Enum):
    GeoWiki = "geowiki_landcover_2017"
    KenyaNonCrop = "kenya_non_crop"
    KenyaOAF = "one_acre_fund_kenya"
    KenyaPV = "plant_village_kenya"


def geowiki_file_name(participants: str = "all") -> str:
    participants_to_file_labels = {"all": "all", "students": "con", "experts": "exp"}
    file_label = participants_to_file_labels.get(participants, participants)
    assert file_label in participants_to_file_labels.values(), f"Unknown participant {file_label}"
    return f"loc_{file_label}{'_2' if file_label == 'all' else ''}.txt"


datasets = [
    Dataset(
        dataset=DatasetName.GeoWiki.value,
        sentinel_dataset="earth_engine_geowiki",
        labels_file="data.nc",
        crop_probability=lambda overlap: overlap.mean_sumcrop / 100,
        val_set_size=0.2,
        test_set_size=0,
        is_global=True,
        processors=(Processor(file_name=geowiki_file_name(), custom_geowiki_processing=True),),
    ),
    Dataset(
        dataset=DatasetName.KenyaNonCrop.value,
        sentinel_dataset="earth_engine_kenya_non_crop",
        crop_probability=0.0,
        processors=(
            Processor(file_name="noncrop_labels_v2", lat_lon_transform=False),
            Processor(file_name="noncrop_labels_set2", lat_lon_transform=False),
            Processor(file_name="2019_gepro_noncrop", x_y_reversed=True, lat_lon_transform=True),
            Processor(
                file_name="noncrop_water_kenya_gt", x_y_reversed=True, lat_lon_transform=True
            ),
            Processor(file_name="noncrop_kenya_gt", x_y_reversed=True, lat_lon_transform=True),
        ),
    ),
    Dataset(
        dataset=DatasetName.KenyaOAF.value,
        sentinel_dataset="earth_engine_one_acre_fund_kenya",
        crop_probability=1.0,
        is_maize=True,
        processors=(Processor(file_name="", x_y_from_centroid=False, lat_lon_lowercase=True),),
    ),
    Dataset(
        dataset=DatasetName.KenyaPV.value,
        sentinel_dataset="earth_engine_plant_village_kenya",
        crop_probability=1.0,
        crop_type_func=lambda overlap: overlap.crop_type,
        processors=(
            Processor(file_name="field_boundaries_pv_04282020.shp", lat_lon_transform=True),
        ),
    ),
]
