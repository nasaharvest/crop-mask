import numpy as np
from dataclasses import dataclass
from typing import Union
from .boundingbox import BoundingBox


@dataclass
class CropDataInstance:
    instance_lat: float
    instance_lon: float
    labelled_array: Union[float, np.ndarray]
    source_file: str

    def isin(self, bounding_box: BoundingBox) -> bool:
        return (
            (self.instance_lon <= bounding_box.max_lon)
            & (self.instance_lon >= bounding_box.min_lon)
            & (self.instance_lat <= bounding_box.max_lat)
            & (self.instance_lat >= bounding_box.min_lat)
        )
