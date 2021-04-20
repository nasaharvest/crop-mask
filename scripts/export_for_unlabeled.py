import logging

from src.dataset_config import unlabeled_datasets

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset_name = "<dataset name>"
    for d in unlabeled_datasets:
        if d.sentinel_dataset == dataset_name:
            d.export_earth_engine_data()
