from datetime import date, timedelta
import pandas as pd
import sys

from .base import BaseSentinelExporter
from .utils import bounding_box_to_earth_engine_bounding_box
from src.boundingbox import BoundingBox

from typing import Optional, Tuple
from enum import Enum


class Season(Enum):
    in_season = "in_season"
    post_season = "post_season"


def get_user_input(text_prompt: str) -> str:
    return input(text_prompt)


class RegionalExporter(BaseSentinelExporter):
    r"""
    This is useful if you are trying to export
    full regions for predictions
    """

    dataset = "earth_engine_region_partial_slow_cloudfree"

    def load_labels(self) -> pd.DataFrame:
        # We don't need any labels for this exporter,
        # so we can return an empty dataframe
        return pd.DataFrame()

    @staticmethod
    def determine_start_end_dates(season: Season) -> Tuple[date, date]:
        today = date.today()
        after_april = today.month > 4
        prev_year = today.year - 1
        prev_prev_year = today.year - 2

        if season == Season.in_season:
            start_date = date(today.year if after_april else prev_year, 4, 1)
            months_between = (today.year - start_date.year) * 12 + today.month - start_date.month
            if months_between < 7:
                user_input = get_user_input(
                    f"WARNING: There are only {months_between} month(s) between today and the "
                    f"start of the season (April 1st). \nAre you sure you'd like proceed "
                    f"exporting only {months_between} months? (Y/N):\n"
                )
                if any(user_input == no for no in ["n", "N", "no", "NO"]):
                    sys.exit("Exiting script.")

            return start_date, today

        if season == Season.post_season:
            start_date = date(prev_year if after_april else prev_prev_year, 4, 1)
            end_date = date(today.year if after_april else prev_year, 4, 1)
            return start_date, end_date

        raise ValueError("Season must be in_season or post_season")

    def export_for_region(
        self,
        region_name: str,
        region_bbox: BoundingBox,
        season: Optional[Season] = None,
        end_date: Optional[date] = None,
        num_timesteps: Optional[int] = None,
        days_per_timestep: int = 30,
        checkpoint: bool = True,
        monitor: bool = True,
        metres_per_polygon: Optional[int] = 10000,
        fast: bool = True,
    ):
        r"""
        Run the regional exporter. For each label, the exporter will export
        data from (end_date - timedelta(days=days_per_timestep * num_timesteps)) to end_date
        where each timestep consists of a mosaic of all available images within the
        days_per_timestep of that timestep.
        :param region_name: The name of the region to export.
        :param region_bbox: BoundingBox for region
        :param season: Season enum to determine what type of date range to use for export
        :param end_date: The end date of the data export if season is not specified
        :param num_timesteps: The number of timesteps to export if season is not specified
        :param days_per_timestep: The number of days of data to use for each mosaiced image.
        :param checkpoint: Whether or not to check in self.data_folder to see if the file has
            already been exported. If it has, skip it
        :param monitor: Whether to monitor each task until it has been run
        :param metres_per_polygon: Whether to split the export of a large region into smaller
            boxes of (max) area metres_per_polygon * metres_per_polygon. It is better to instead
            split the area once it has been exported
        :param fast: Whether to use the faster cloudfree exporter. This function is considerably
            faster, but cloud artefacts can be more pronounced. Default = True
        """
        self.dataset = f"earth_engine_region_{region_name}_partial_slow_cloudfree"

        if season:
            start_date, end_date = self.determine_start_end_dates(season)
        elif end_date and num_timesteps:
            start_date = end_date - num_timesteps * timedelta(days=days_per_timestep)
        else:
            raise ValueError(
                "Unable to determine start_date, either 'season' or 'end_date' and 'num_timesteps' must "
                "be set."
            )

        region = bounding_box_to_earth_engine_bounding_box(region_bbox)

        if metres_per_polygon is not None:

            regions = region.to_polygons(metres_per_patch=metres_per_polygon)

            for idx, region in enumerate(regions):
                self._export_for_polygon(
                    polygon=region,
                    polygon_identifier=f"{idx}-{region_name}",
                    start_date=start_date,
                    end_date=end_date,
                    days_per_timestep=days_per_timestep,
                    checkpoint=checkpoint,
                    monitor=monitor,
                    fast=fast,
                )
        else:
            self._export_for_polygon(
                polygon=region.to_ee_polygon(),
                polygon_identifier=region_name,
                start_date=start_date,
                end_date=end_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
