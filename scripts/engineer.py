"""
Combines the earth observation data with the labels to create (x, y) training data
"""
import logging
import sys

sys.path.append("..")
from src.datasets_labeled import labeled_datasets  # noqa: E402

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset_name = "earth_engine_uganda"
    for d in labeled_datasets:
        if d.sentinel_dataset == dataset_name:
            d.create_pickled_labeled_dataset()
