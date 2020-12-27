from dataclasses import dataclass
from datetime import date
from math import cos, radians
import ee

from typing import List, Tuple, Union

from src.utils import BoundingBox


def date_overlap(start1: date, end1: date, start2: date, end2: date) -> int:
    overlaps = start1 <= end2 and end1 >= start2
    if not overlaps:
        return 0
    return (min(end1, end2) - max(start1, start2)).days


def metre_per_degree(mid_lat: float) -> Tuple[float, float]:
    # https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula
    # see the link above to explain the magic numbers
    m_per_deg_lat = 111132.954 - 559.822 * cos(2.0 * mid_lat) + 1.175 * cos(radians(4.0 * mid_lat))
    m_per_deg_lon = (3.14159265359 / 180) * 6367449 * cos(radians(mid_lat))

    return m_per_deg_lat, m_per_deg_lon


@dataclass
class EEBoundingBox(BoundingBox):
    r"""
    A bounding box with additional earth-engine specific
    functionality
    """

    def to_ee_polygon(self) -> ee.Geometry.Polygon:
        return ee.Geometry.Polygon(
            [
                [
                    [self.min_lon, self.min_lat],
                    [self.min_lon, self.max_lat],
                    [self.max_lon, self.max_lat],
                    [self.max_lon, self.min_lat],
                ]
            ]
        )

    def to_metres(self) -> Tuple[float, float]:
        r"""
        :return: [lat metres, lon metres]
        """
        # https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula
        mid_lat = (self.min_lat + self.max_lat) / 2.0
        m_per_deg_lat, m_per_deg_lon = metre_per_degree(mid_lat)

        delta_lat = self.max_lat - self.min_lat
        delta_lon = self.max_lon - self.min_lon

        return delta_lat * m_per_deg_lat, delta_lon * m_per_deg_lon

    def to_polygons(self, metres_per_patch: int = 3300) -> List[ee.Geometry.Polygon]:

        lat_metres, lon_metres = self.to_metres()

        num_cols = int(lon_metres / metres_per_patch)
        num_rows = int(lat_metres / metres_per_patch)

        print(f"Splitting into {num_cols} columns and {num_rows} rows")

        lon_size = (self.max_lon - self.min_lon) / num_cols
        lat_size = (self.max_lat - self.min_lat) / num_rows

        output_polygons: List[ee.Geometry.Polygon] = []

        cur_lon = self.min_lon
        while cur_lon < self.max_lon:
            cur_lat = self.min_lat
            while cur_lat < self.max_lat:
                output_polygons.append(
                    ee.Geometry.Polygon(
                        [
                            [
                                [cur_lon, cur_lat],
                                [cur_lon, cur_lat + lat_size],
                                [cur_lon + lon_size, cur_lat + lat_size],
                                [cur_lon + lon_size, cur_lat],
                            ]
                        ]
                    )
                )
                cur_lat += lat_size
            cur_lon += lon_size

        return output_polygons


def bounding_box_from_centre(
    mid_lat: float, mid_lon: float, surrounding_metres: Union[int, Tuple[int, int]]
) -> EEBoundingBox:

    m_per_deg_lat, m_per_deg_lon = metre_per_degree(mid_lat)

    if isinstance(surrounding_metres, int):
        surrounding_metres = (surrounding_metres, surrounding_metres)

    surrounding_lat, surrounding_lon = surrounding_metres

    deg_lat = surrounding_lat / m_per_deg_lat
    deg_lon = surrounding_lon / m_per_deg_lon

    max_lat, min_lat = mid_lat + deg_lat, mid_lat - deg_lat
    max_lon, min_lon = mid_lon + deg_lon, mid_lon - deg_lon

    return EEBoundingBox(max_lon=max_lon, min_lon=min_lon, max_lat=max_lat, min_lat=min_lat)


def bounding_box_to_earth_engine_bounding_box(bounding_box: BoundingBox,) -> EEBoundingBox:
    return EEBoundingBox(
        max_lat=bounding_box.max_lat,
        min_lat=bounding_box.min_lat,
        max_lon=bounding_box.max_lon,
        min_lon=bounding_box.min_lon,
    )


def cancel_all_tasks() -> None:

    ee.Initialize()

    tasks = ee.batch.Task.list()
    print(f"Cancelling up to {len(tasks)} tasks")
    # Cancel running and ready tasks
    for task in tasks:
        task_id = task.status()["id"]
        task_state = task.status()["state"]
        if task_state == "RUNNING" or task_state == "READY":
            task.cancel()
            print(f"Task {task_id} cancelled")
        else:
            print(f"Task {task_id} state is {task_state}")
