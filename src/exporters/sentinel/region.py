from datetime import date, timedelta
import pandas as pd

from .base import BaseSentinelExporter
from .utils import bounding_box_to_earth_engine_bounding_box
from src.utils import STR2BB

from typing import Optional


class RegionalExporter(BaseSentinelExporter):
    r"""
    This is useful if you are trying to export
    full regions for predictions
    """

    dataset = "earth_engine_region_busia_partial_slow_cloudfree"

    def load_labels(self) -> pd.DataFrame:
        # We don't need any labels for this exporter,
        # so we can return an empty dataframe
        return pd.DataFrame()

    def export_for_region(
        self,
        region_name: str,
        end_date: date,
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
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
        :param region_name: The name of the region to export. This must be defined in
            src.utils.STR2BB
        :param end_date: The end date of the data export
        :param days_per_timestep: The number of days of data to use for each mosaiced image.
        :param num_timesteps: The number of timesteps to export
        :param checkpoint: Whether or not to check in self.data_folder to see if the file has
            already been exported. If it has, skip it
        :param monitor: Whether to monitor each task until it has been run
        :param metres_per_polygon: Whether to split the export of a large region into smaller
            boxes of (max) area metres_per_polygon * metres_per_polygon. It is better to instead
            split the area once it has been exported
        :param fast: Whether to use the faster cloudfree exporter. This function is considerably
            faster, but cloud artefacts can be more pronounced. Default = True
        """
        start_date = end_date - num_timesteps * timedelta(days=days_per_timestep)

        region = bounding_box_to_earth_engine_bounding_box(STR2BB[region_name])

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
