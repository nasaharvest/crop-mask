"""
Downloads labels if available, processed labels to have common format, 
and exports datasets using Google Earth Engine
(locally, or to Google Drive)
"""
import logging
import sys

sys.path.append("..")
from src.dataset_config import labeled_datasets

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    for d in labeled_datasets:
        d.download_raw_labels()
        d.process_labels()
        # d.export_earth_engine_data()
