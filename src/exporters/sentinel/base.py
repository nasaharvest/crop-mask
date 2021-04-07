from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path
import logging
import pandas as pd
import ee

from . import cloudfree

from typing import List, Union

logger = logging.getLogger(__name__)


class BaseSentinelExporter(ABC):

    r"""
    Download cloud free sentinel data for countries,
    where countries are defined by the simplified large scale
    international boundaries.
    """

    dataset: str
    min_date = date(2017, 3, 28)

    def __init__(self, data_folder: Path = Path("data")) -> None:
        self.data_folder = data_folder
        self.raw_folder = self.data_folder / "raw"
        self.output_folder = self.raw_folder / self.dataset
        self.output_folder.mkdir(parents=True, exist_ok=True)

        try:
            ee.Initialize()
        except Exception:
            logger.error(
                "This code doesn't work unless you have authenticated your earthengine account"
            )

        self.labels = self.load_labels()

    @abstractmethod
    def load_labels(self) -> pd.DataFrame:
        raise NotImplementedError

    def _export_for_polygon(
        self,
        polygon: ee.Geometry.Polygon,
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        days_per_timestep: int,
        checkpoint: bool,
        monitor: bool,
        fast: bool,
    ) -> None:

        if fast:
            export_func = cloudfree.get_single_image_fast
        else:
            export_func = cloudfree.get_single_image

        cur_date = start_date
        cur_end_date = cur_date + timedelta(days=days_per_timestep)

        image_collection_list: List[ee.Image] = []

        logger.info(
            f"Exporting image for polygon {polygon_identifier} from "
            f"aggregated images between {str(cur_date)} and {str(end_date)}"
        )
        filename = f"{polygon_identifier}_{str(cur_date)}_{str(end_date)}"

        if checkpoint and (self.output_folder / f"{filename}.tif").exists():
            logger.warning("File already exists! Skipping")
            return None

        while cur_end_date <= end_date:

            image_collection_list.append(
                export_func(region=polygon, start_date=cur_date, end_date=cur_end_date)
            )
            cur_date += timedelta(days=days_per_timestep)
            cur_end_date += timedelta(days=days_per_timestep)

        # now, we want to take our image collection and append the bands into a single image
        imcoll = ee.ImageCollection(image_collection_list)
        img = ee.Image(imcoll.iterate(cloudfree.combine_bands))

        # and finally, export the image
        cloudfree.export(
            image=img,
            region=polygon,
            filename=filename,
            drive_folder=self.dataset,
            monitor=monitor,
        )
