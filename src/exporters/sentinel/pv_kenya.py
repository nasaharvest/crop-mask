import pandas as pd
import geopandas
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta, date

from .base import BaseSentinelExporter
from src.processors.pv_kenya import KenyaPVProcessor
from .utils import EEBoundingBox, bounding_box_from_centre, date_overlap

from typing import Dict, Optional, List, Tuple


class KenyaPVSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_plant_village_kenya"

    def load_labels(self) -> pd.DataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        plantvillage = self.data_folder / "processed" / KenyaPVProcessor.dataset / "data.geojson"
        assert plantvillage.exists(), "Plant Village processor must be run to load labels"
        return geopandas.read_file(plantvillage)[
            ["lat", "lon", "index", "planting_d", "harvest_da"]
        ]

    @staticmethod
    def overlapping_year(
        end_month: int, num_days: int, harvest_date: date, planting_date: date
    ) -> Tuple[Optional[int], Optional[int]]:
        r"""
        Return the end_year of the most overlapping years
        """
        harvest_year = harvest_date.year

        overlap_dict: Dict[int, int] = {}

        for diff in range(-1, 2):
            end_date = date(harvest_year + diff, end_month, 1)

            if end_date > datetime.now().date():
                continue
            else:
                overlap_dict[harvest_year + diff] = date_overlap(
                    planting_date, harvest_date, end_date - timedelta(days=num_days), end_date,
                )
        if len(overlap_dict) > 0:
            return max(overlap_dict.items(), key=lambda x: x[1])
        else:
            # sometimes the harvest date is in the future? in which case
            # we will just skip the datapoint for now
            return None, None

    def labels_to_bounding_boxes(
        self,
        num_labelled_points: Optional[int],
        surrounding_metres: int,
        end_month_day: Optional[Tuple[int, int]],
        num_days: int,
    ) -> List[Tuple[int, EEBoundingBox, date, Optional[int]]]:

        output: List[Tuple[int, EEBoundingBox, date, Optional[int]]] = []

        if end_month_day is not None:
            end_month: Optional[int]
            end_day: Optional[int]
            end_month, end_day = end_month_day
        else:
            end_month = end_day = None

        for idx, row in tqdm(self.labels.iterrows()):

            try:
                harvest_date = datetime.strptime(row["harvest_da"], "%Y-%m-%d %H:%M:%S").date()
            except ValueError:
                continue

            # this is only used if end_month is not None
            overlapping_days: Optional[int] = 0
            if end_month is not None:
                planting_date = datetime.strptime(row["planting_d"], "%Y-%m-%d %H:%M:%S").date()

                end_year, overlapping_days = self.overlapping_year(
                    end_month, num_days, harvest_date, planting_date
                )

                if end_year is None:
                    continue

                if end_day is None:
                    # if no end_day is passed, we will take the first month
                    end_day = 1
                harvest_date = date(end_year, end_month, end_day)

            output.append(
                (
                    row["index"],
                    bounding_box_from_centre(
                        mid_lat=row["lat"],
                        mid_lon=row["lon"],
                        surrounding_metres=surrounding_metres,
                    ),
                    harvest_date,
                    overlapping_days,
                )
            )

            if num_labelled_points is not None:
                if len(output) >= num_labelled_points:
                    return output
        return output

    def get_start_and_end_dates(
        self, harvest_date: date, days_per_timestep: int, num_timesteps: int
    ) -> Optional[Tuple[date, date]]:

        if harvest_date < self.min_date:
            print("Harvest date < min date - skipping")
            return None
        else:
            start_date = max(
                harvest_date - timedelta(days_per_timestep * num_timesteps), self.min_date,
            )
            end_date = start_date + timedelta(days_per_timestep * num_timesteps)

            return start_date, end_date

    def export_for_labels(
        self,
        end_month_day: Optional[Tuple[int, int]] = (4, 16),
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
        fast: bool = True,
    ) -> None:
        r"""
        :param end_month_day: The final month-day to use. If None is passed, the harvest date
            will be used. Default = (4, 16)
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
            num_labelled_points=num_labelled_points,
            surrounding_metres=surrounding_metres,
            end_month_day=end_month_day,
            num_days=days_per_timestep * num_timesteps,
        )

        if end_month_day is not None:
            print(
                f"Average overlapping days between planting to harvest and "
                f"export dates: {np.mean([x[3] for x in bounding_boxes_to_download])}"
            )
        for idx, bounding_info in enumerate(bounding_boxes_to_download):

            harvest_date = bounding_info[-2]

            dates = self.get_start_and_end_dates(harvest_date, days_per_timestep, num_timesteps)

            if dates is not None:

                self._export_for_polygon(
                    polygon=bounding_info[1].to_ee_polygon(),
                    polygon_identifier=bounding_info[0],
                    start_date=dates[0],
                    end_date=dates[1],
                    days_per_timestep=days_per_timestep,
                    checkpoint=checkpoint,
                    monitor=monitor,
                    fast=fast,
                )
