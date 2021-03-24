import logging
import sys

sys.path.append("..")
from src.ETL import datasets

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    for d in datasets:
        d.create_pickled_labeled_dataset()
