import logging
import sys

sys.path.append("..")

from src.dataset_config import datasets

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    for d in datasets:
        d.process_labels()
