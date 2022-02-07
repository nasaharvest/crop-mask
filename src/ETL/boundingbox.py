from dataclasses import dataclass
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def __post_init__(self):
        if self.max_lon < self.min_lon:
            raise ValueError("max_lon should be larger than min_lon")
        if self.max_lat < self.min_lat:
            raise ValueError("max_lat should be larger than min_lat")

        self.url = (
            f"http://bboxfinder.com/#{self.min_lat},{self.min_lon},{self.max_lat},{self.max_lon}"
        )

    @classmethod
    def from_path(cls, p: Path):
        decimals_in_p = re.findall(r"=-?\d*\.?\d*", p.stem)
        coords = [float(d[1:]) for d in decimals_in_p[0:4]]
        bbox = cls(min_lat=coords[0], min_lon=coords[1], max_lat=coords[2], max_lon=coords[3])
        return bbox

    def contains(self, lat: float, lon: float):
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon

    def overlaps(self, other: "BoundingBox"):
        return (
            self.min_lat < other.max_lat
            and self.max_lat > other.min_lat
            and self.min_lon < other.max_lon
            and self.max_lon > other.min_lon
        )
