import pandas as pd
import geopandas
from tqdm import tqdm
from datetime import timedelta, date

from .base import BaseSentinelExporter
from src.processors.oaf_kenya import KenyaOAFProcessor
from .utils import EEBoundingBox, bounding_box_from_centre

from typing import Optional, List


class KenyaOAFSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_one_acre_fund_kenya"

    # data collection date, to be consistent
    # with the non crop data
    data_date = date(2020, 4, 16)

    def load_labels(self) -> pd.DataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        oaf = self.data_folder / "processed" / KenyaOAFProcessor.dataset / "data.geojson"
        assert oaf.exists(), "Kenya one acre fund processor must be run to load labels"
        return geopandas.read_file(oaf)[["lat", "lon"]]

    def labels_to_bounding_boxes(
        self, num_labelled_points: Optional[int], surrounding_metres: int
    ) -> List[EEBoundingBox]:

        output: List[EEBoundingBox] = []

        for idx, row in tqdm(self.labels.iterrows()):

            output.append(
                bounding_box_from_centre(
                    mid_lat=row["lat"], mid_lon=row["lon"], surrounding_metres=surrounding_metres,
                ),
            )

            if num_labelled_points is not None:
                if len(output) >= num_labelled_points:
                    return output
        return output

    def export_for_labels(
        self,
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
        fast: bool = True,
    ) -> None:
        r"""
        :param days_per_timestep: The number of days of data to use for each mosaiced image.
            Default = 30
        :param num_timesteps: The number of timesteps to export. Default = 12
        :param num_labelled_points: If not None, then only this many points will be exported.
            Default = None.
        :param surrouning_metres: The patch will be [2 * surrounding_metres,
            2 * surrounding_metres], centered around the labelled point. Default = 80
        :param checkpoint: Whether or not to check in self.data_folder to see if the file has
            already been exported. If it has, skip it. Default = True
        :param monitor: Whether to monitor each task until it has been run. Default = True
        :param fast: Whether to use the faster cloudfree exporter. This function is considerably
            faster, but cloud artefacts can be more pronounced. Default = True
        """
        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            num_labelled_points=num_labelled_points, surrounding_metres=surrounding_metres,
        )

        start_date = self.data_date - num_timesteps * timedelta(days=days_per_timestep)

        for idx, bounding_info in enumerate(bounding_boxes_to_download):

            self._export_for_polygon(
                polygon=bounding_info.to_ee_polygon(),
                polygon_identifier=idx,
                start_date=start_date,
                end_date=self.data_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
