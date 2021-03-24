import logging
import sys
from pathlib import Path

sys.path.append("..")

from src.exporters import (
    GeoWikiExporter,
    GeoWikiSentinelExporter,
    KenyaPVSentinelExporter,
    KenyaNonCropSentinelExporter,
    KenyaOAFSentinelExporter,
    RegionalExporter,
    Season,
)
from src.boundingbox import BoundingBox

logging.basicConfig(level=logging.INFO)


def export_from_labeled(
    export_geowiki=True,
    export_geowiki_sentinel_ee=True,
    export_plant_village_sentinel_ee=True,
    export_kenya_non_crop=True,
    export_oaf=True,
    data_dir: Path = Path("../data"),
):
    if export_geowiki:
        GeoWikiExporter(data_dir).export()
    if export_geowiki_sentinel_ee:
        GeoWikiSentinelExporter(data_dir).export_for_labels(
            num_labelled_points=None, monitor=False, checkpoint=True
        )
    if export_plant_village_sentinel_ee:
        KenyaPVSentinelExporter(data_dir).export_for_labels(
            num_labelled_points=None, monitor=False, checkpoint=True
        )
    if export_kenya_non_crop:
        KenyaNonCropSentinelExporter(data_dir).export_for_labels(
            num_labelled_points=None, monitor=False, checkpoint=True
        )
    if export_oaf:
        KenyaOAFSentinelExporter(data_dir).export_for_labels(
            num_labelled_points=None, monitor=False, checkpoint=True
        )


def export_from_bbox(region_name_in_STR2BB: str, data_dir: Path = Path("../data")):
    if region_name_in_STR2BB not in STR2BB:
        raise ValueError(f"{region_name_in_STR2BB} not found in STR2BB")
    if "_" in region_name_in_STR2BB:
        raise ValueError(
            f"{region_name_in_STR2BB} shoud not include underscores (_), please use CamelCase"
        )
    RegionalExporter(data_dir).export_for_region(
        region_name=region_name_in_STR2BB,
        region_bbox=STR2BB[region_name_in_STR2BB],
        season=Season.post_season,
        monitor=False,
        checkpoint=True,
        metres_per_polygon=None,
        fast=False,
    )


STR2BB = {
    "Kenya": BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
    "Busia": BoundingBox(
        min_lon=33.88389587402344,
        min_lat=-0.04119872691853491,
        max_lon=34.44007873535156,
        max_lat=0.7779454563313616,
    ),
    "NorthMalawi": BoundingBox(min_lon=32.688, max_lon=35.772, min_lat=-14.636, max_lat=-9.231),
    "SouthMalawi": BoundingBox(min_lon=34.211, max_lon=35.772, min_lat=-17.07, max_lat=-14.636),
    "Rwanda": BoundingBox(min_lon=28.841, max_lon=30.909, min_lat=-2.854, max_lat=-1.034),
    "RwandaSake": BoundingBox(min_lon=30.377, max_lon=30.404, min_lat=-2.251, max_lat=-2.226),
    "Togo": BoundingBox(
        min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625
    ),
}

if __name__ == "__main__":
    export_from_bbox(region_name_in_STR2BB="RwandaSake")
    export_from_labeled(
        export_geowiki=True,  
        export_geowiki_sentinel_ee=False,
        export_plant_village_sentinel_ee=False,
        export_kenya_non_crop=True,
        export_oaf=False,
    )
