r"""
Functions shared by both the fast and slow
cloudfree algorithm
"""
import ee
import logging
from datetime import date
from src.constants import BANDS
from typing import Union

logger = logging.getLogger(__name__)


def combine_bands(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Author: Jamie Vleeshouwer

    # Rename the band
    previous = ee.Image(previous)
    current = current.select(BANDS)
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )


def export(
    image: ee.Image,
    region: ee.Geometry,
    filename: str,
    drive_folder: str,
    monitor: bool = False,
) -> ee.batch.Export:

    # task = ee.batch.Export.image(
    #     image.clip(region),
    #     filename,
    #     {"scale": 10, "region": region, "maxPixels": 1e13, "driveFolder": drive_folder},
    # )

    task = ee.batch.Export.image.toCloudStorage(
        image=image.clip(region),
        bucket="ee-data-to-be-split",
        description=filename,
        fileNamePrefix=drive_folder + "/" + filename,
        scale=10,
        region=region,
        maxPixels=1e13,
    )

    try:
        task.start()
    except ee.ee_exception.EEException as e:
        logger.error(f"Task not started! Got exception {e}")
        return task

    if monitor:
        monitor_task(task)

    return task


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")


def monitor_task(task: ee.batch.Export) -> None:
    while task.status()["state"] in ["READY", "RUNNING"]:
        logger.info(task.status())


def rescale(img, exp, thresholds):
    return (
        img.expression(exp, {"img": img})
        .subtract(thresholds[0])
        .divide(thresholds[1] - thresholds[0])
    )
