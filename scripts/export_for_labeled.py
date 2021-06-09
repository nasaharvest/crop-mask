"""
Downloads labels if available, processed labels to have common format,
and exports datasets using Google Earth Engine
(locally, or to Google Drive)
"""
import sys

sys.path.append("..")
from data.datasets_labeled import labeled_datasets  # noqa: E402

if __name__ == "__main__":
    for d in labeled_datasets:
        #d.download_raw_labels()
        d.process_labels()
        #d.export_earth_engine_data()
