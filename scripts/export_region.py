"""
Exports region using Google Earth Engine
(locally, or to Google Drive)
"""

import os
from src.ETL.ee_exporter import RegionExporter
from src.ETL.ee_boundingbox import BoundingBox

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))


if __name__ == "__main__":
    dest_folder = "<your destination folder>"
    bounding_box = BoundingBox(0, 1, 0, 1)
    RegionExporter().export(
        metres_per_polygon=None, dest_path=dest_folder, region_bbox=bounding_box
    )
