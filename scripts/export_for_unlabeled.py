"""
Exports specified unlabeled dataset using Google Earth Engine
(locally, or to Google Drive)
"""

import logging
from argparse import ArgumentParser
import sys

sys.path.append("..")

from data.datasets_unlabeled import unlabeled_datasets  # noqa: E402

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    params = parser.parse_args()
    try:
        dataset = next(d for d in unlabeled_datasets if d.sentinel_dataset == params.dataset_name)
        dataset.export_earth_engine_data()
    except Exception:
        print(f"ERROR: no dataset was found with name: {params.dataset_name}")
