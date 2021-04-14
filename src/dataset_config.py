from enum import Enum
from src.ETL.dataset import LabeledDataset, UnlabeledDataset
from src.ETL.ee_exporter import EarthEngineExporter, Season
from src.ETL.ee_boundingbox import BoundingBox
from src.ETL.label_downloader import RawLabels
from src.ETL.processor import Processor
from datetime import date


class DatasetName(Enum):
    GeoWiki = "geowiki_landcover_2017"
    KenyaNonCrop = "kenya_non_crop"
    KenyaOAF = "one_acre_fund_kenya"
    KenyaPV = "plant_village_kenya"
    MaliNonCrop = "mali_non_crop"
    MaliSegou2018 = "mali_segou_2018"
    MaliSegou2019 = "mali_segou_2019"


def geowiki_file_name(participants: str = "all") -> str:
    participants_to_file_labels = {"all": "all", "students": "con", "experts": "exp"}
    file_label = participants_to_file_labels.get(participants, participants)
    assert file_label in participants_to_file_labels.values(), f"Unknown participant {file_label}"
    return f"loc_{file_label}{'_2' if file_label == 'all' else ''}.txt"


labeled_datasets = [
    LabeledDataset(
        dataset=DatasetName.GeoWiki.value,
        sentinel_dataset="earth_engine_geowiki",
        labels_file="data.nc",
        crop_probability=lambda overlap: overlap.mean_sumcrop / 100,
        val_set_size=0.2,
        test_set_size=0,
        is_global=True,
        raw_labels=(
            RawLabels("http://store.pangaea.de/Publications/See_2017/crop_all.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/crop_con.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/crop_exp.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_all.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_all_2.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_con.zip"),
            RawLabels("http://store.pangaea.de/Publications/See_2017/loc_exp.zip"),
        ),
        processors=(Processor(file_name=geowiki_file_name(), custom_geowiki_processing=True),),
        exporter=EarthEngineExporter(start_date=date(2017, 3, 28), end_date=date(2018, 3, 28)),
    ),
    LabeledDataset(
        dataset=DatasetName.KenyaNonCrop.value,
        country="Kenya",
        sentinel_dataset="earth_engine_kenya_non_crop",
        crop_probability=0.0,
        processors=(
            Processor(file_name="noncrop_labels_v2", lat_lon_transform=True),
            Processor(file_name="noncrop_labels_set2", lat_lon_transform=True),
            Processor(file_name="2019_gepro_noncrop"),
            Processor(file_name="noncrop_water_kenya_gt"),
            Processor(file_name="noncrop_kenya_gt"),
        ),
        exporter=EarthEngineExporter(end_date=date(2020, 4, 16)),
    ),
    LabeledDataset(
        dataset=DatasetName.KenyaOAF.value,
        country="Kenya",
        sentinel_dataset="earth_engine_one_acre_fund_kenya",
        crop_probability=1.0,
        is_maize=True,
        processors=(Processor(file_name="", x_y_from_centroid=False, lat_lon_lowercase=True),),
        exporter=EarthEngineExporter(
            end_date=date(2020, 4, 16),
        ),
    ),
    LabeledDataset(
        dataset=DatasetName.KenyaPV.value,
        country="Kenya",
        sentinel_dataset="earth_engine_plant_village_kenya",
        crop_probability=1.0,
        crop_type_func=lambda overlap: overlap.crop_type,
        processors=(
            Processor(file_name="field_boundaries_pv_04282020.shp", lat_lon_transform=True),
        ),
        exporter=EarthEngineExporter(
            additional_cols=["index", "planting_d", "harvest_da"],
            end_month_day=(4, 16),
        ),
    ),
    LabeledDataset(
        dataset=DatasetName.MaliNonCrop.value,
        country="Mali",
        sentinel_dataset="earth_engine_mali_noncrop",
        crop_probability=0.0,
        processors=(Processor(file_name="mali_noncrop_2019"),),
        exporter=EarthEngineExporter(end_date=date(2020, 4, 16))
    ),
    LabeledDataset(
        dataset=DatasetName.MaliSegou2018.value,
        country="Mali",
        sentinel_dataset="earth_engine_mali_segou_2018",
        crop_probability=1.0,
        processors=(Processor(file_name="segou_bounds_07212020"), ),
        exporter=EarthEngineExporter(end_date=date(2019, 4, 16))
    ),
    LabeledDataset(
        dataset=DatasetName.MaliSegou2019.value,
        country="Mali",
        sentinel_dataset="earth_engine_mali_segou_2019",
        crop_probability=1.0,
        processors=(Processor(file_name="segou_bounds_07212020"), ),
        exporter=EarthEngineExporter(end_date=date(2020, 4, 16))
    )
]

unlabeled_datasets = [
    UnlabeledDataset(
        sentinel_dataset="Kenya",
        exporter=EarthEngineExporter(
            region_bbox=BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
            season=Season.post_season,
        ),
    ),
    UnlabeledDataset(
        sentinel_dataset="Busia",
        exporter=EarthEngineExporter(
            region_bbox=BoundingBox(
                min_lon=33.88389587402344,
                min_lat=-0.04119872691853491,
                max_lon=34.44007873535156,
                max_lat=0.7779454563313616,
            ),
            season=Season.post_season,
        ),
    ),
    UnlabeledDataset(
        sentinel_dataset="NorthMalawi",
        exporter=EarthEngineExporter(
            region_bbox=BoundingBox(
                min_lon=32.688, max_lon=35.772, min_lat=-14.636, max_lat=-9.231
            ),
            season=Season.post_season,
        ),
    ),
    UnlabeledDataset(
        sentinel_dataset="SouthMalawi",
        exporter=EarthEngineExporter(
            region_bbox=BoundingBox(
                min_lon=34.211, max_lon=35.772, min_lat=-17.07, max_lat=-14.636
            ),
            season=Season.post_season,
        ),
    ),
    UnlabeledDataset(
        sentinel_dataset="Rwanda",
        exporter=EarthEngineExporter(
            region_bbox=BoundingBox(min_lon=28.841, max_lon=30.909, min_lat=-2.854, max_lat=-1.034),
            season=Season.post_season,
        ),
    ),
    UnlabeledDataset(
        sentinel_dataset="RwandaSake",
        exporter=EarthEngineExporter(
            region_bbox=BoundingBox(min_lon=30.377, max_lon=30.404, min_lat=-2.251, max_lat=-2.226),
            season=Season.post_season,
        ),
    ),
    UnlabeledDataset(
        sentinel_dataset="Togo",
        exporter=EarthEngineExporter(
            region_bbox=BoundingBox(
                min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625
            ),
            season=Season.post_season,
        ),
    ),
]
