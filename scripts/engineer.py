import logging
import sys

sys.path.append("..")
from src.dataset_config import labeled_datasets  # noqa: E402

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    for d in labeled_datasets:
        d.create_pickled_labeled_dataset()
