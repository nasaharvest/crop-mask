from dataclasses import dataclass


@dataclass
class BoundingBox:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
