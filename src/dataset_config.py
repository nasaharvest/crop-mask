from datetime import date
from src.ETL.dataset import LabeledDataset, UnlabeledDataset
from src.ETL.ee_exporter import EarthEngineExporter, Season
from src.ETL.ee_boundingbox import BoundingBox
from src.ETL.label_downloader import RawLabels
from src.ETL.processor import Processor


def geowiki_file_name(participants: str = "all") -> str:
    participants_to_file_labels = {"all": "all", "students": "con", "experts": "exp"}
    file_label = participants_to_file_labels.get(participants, participants)
    assert file_label in participants_to_file_labels.values(), f"Unknown participant {file_label}"
    return f"loc_{file_label}{'_2' if file_label == 'all' else ''}.txt"


labeled_datasets = [
    LabeledDataset(
        dataset="geowiki_landcover_2017",
        country="global",
        sentinel_dataset="earth_engine_geowiki",
        labels_extension=".csv",
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
        processors=(
            Processor(
                file_name=geowiki_file_name(),
                custom_geowiki_processing=True,
                crop_prob=lambda df: df.mean_sumcrop / 100,
                custom_start_date=date(2017, 3, 28),
                custom_end_date=date(2018, 3, 28),
                x_y_from_centroid=False,
            ),
        ),
    ),
    LabeledDataset(
        dataset="Kenya",
        country="Kenya",
        sentinel_dataset="earth_engine_kenya",
        processors=(
            Processor(
                file_name="noncrop_labels_v2",
                crop_prob=0.0,
                end_year=2020,
                lat_lon_transform=True,
            ),
            Processor(
                file_name="noncrop_labels_set2",
                crop_prob=0.0,
                end_year=2020,
                lat_lon_transform=True,
            ),
            Processor(file_name="2019_gepro_noncrop", crop_prob=0.0, end_year=2020),
            Processor(file_name="noncrop_water_kenya_gt", crop_prob=0.0, end_year=2020),
            Processor(file_name="noncrop_kenya_gt", crop_prob=0.0, end_year=2020),
            Processor(
                file_name="one_acre_fund_kenya",
                crop_prob=1.0,
                end_year=2020,
                x_y_from_centroid=False,
                lat_lon_lowercase=True,
            ),
            Processor(
                file_name="plant_village_kenya",
                clean_df=lambda df: df[
                    (df["harvest_da"] != "nan") & (df["harvest_da"] != "unknown")
                ],
                crop_prob=1.0,
                use_harvest_date_for_date_range=True,
                lat_lon_transform=True,
            ),
        ),
    ),
    LabeledDataset(
        dataset="Mali",
        country="Mali",
        sentinel_dataset="earth_engine_mali",
        processors=(
            Processor(file_name="mali_noncrop_2019", crop_prob=0.0, end_year=2020),
            Processor(file_name="segou_bounds_07212020", crop_prob=1.0, end_year=2019),
            Processor(file_name="segou_bounds_07212020", crop_prob=1.0, end_year=2020),
        ),
    ),
    LabeledDataset(
        dataset="Togo",
        country="Togo",
        sentinel_dataset="earth_engine_togo",
        processors=(
            Processor(file_name="crop_merged_v2", crop_prob=1.0, end_year=2020),
            Processor(file_name="noncrop_merged_v2", crop_prob=0.0, end_year=2020),
        ),
    ),
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
