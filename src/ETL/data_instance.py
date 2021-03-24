import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from src.boundingbox import BoundingBox


@dataclass
class CropDataInstance:
    crop_probability: float
    instance_lat: float
    instance_lon: float
    is_global: bool
    label_lat: float
    label_lon: float
    labelled_array: Union[float, np.ndarray]
    crop_label: Optional[str] = ""
    is_maize: Optional[bool] = False

    def isin(self, bounding_box: BoundingBox) -> bool:
        return (
            (self.instance_lon <= bounding_box.max_lon)
            & (self.instance_lon >= bounding_box.min_lon)
            & (self.instance_lat <= bounding_box.max_lat)
            & (self.instance_lat >= bounding_box.min_lat)
        )
