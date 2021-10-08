"""
Combines the earth observation data with the labels to create (x, y) training data
"""
import logging
import os
import sys

# Change the working directory to the directory of the script
os.chdir(os.path.dirname(sys.argv[0]))

sys.path.append("..")
from src.datasets_labeled import labeled_datasets  # noqa: E402

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset_name = "digitalearthafrica"
    for d in labeled_datasets:
        if d.dataset == dataset_name:
            d.create_pickled_labeled_dataset()
