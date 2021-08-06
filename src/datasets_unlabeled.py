from src.ETL.dataset import UnlabeledDataset
from src.ETL.ee_exporter import Season

unlabeled_datasets = [
    UnlabeledDataset(
        sentinel_dataset="Kenya",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Busia",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Mali_USAID_ZOIS_upper",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Mali_USAID_ZOIS_lower",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="NorthMalawi",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="SouthMalawi",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Rwanda",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="RwandaSake",
        season=Season.post_season,
    ),
    UnlabeledDataset(
        sentinel_dataset="Togo",
        season=Season.post_season,
    ),
]
