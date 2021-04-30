from src.ETL.dataset import UnlabeledDataset
from src.ETL.ee_exporter import Season
from src.ETL.ee_boundingbox import BoundingBox

unlabeled_datasets = [
    UnlabeledDataset(
        sentinel_dataset="Kenya",
        region_bbox=BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Busia",
        region_bbox=BoundingBox(
            min_lon=33.88389587402344,
            min_lat=-0.04119872691853491,
            max_lon=34.44007873535156,
            max_lat=0.7779454563313616,
        ),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="MaliSegou",
        region_bbox=BoundingBox(
            min_lon=-7.266759, max_lon=-5.511693, min_lat=12.702882, max_lat=13.937639
        ),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="NorthMalawi",
        region_bbox=BoundingBox(min_lon=32.688, max_lon=35.772, min_lat=-14.636, max_lat=-9.231),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="SouthMalawi",
        region_bbox=BoundingBox(min_lon=34.211, max_lon=35.772, min_lat=-17.07, max_lat=-14.636),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Rwanda",
        region_bbox=BoundingBox(min_lon=28.841, max_lon=30.909, min_lat=-2.854, max_lat=-1.034),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="RwandaSake",
        region_bbox=BoundingBox(min_lon=30.377, max_lon=30.404, min_lat=-2.251, max_lat=-2.226),
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Togo",
        region_bbox=BoundingBox(
            min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625
        ),
        season=Season.post_season,
    ),
]
