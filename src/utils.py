import torch
import numpy as np
import random

from dataclasses import dataclass


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@dataclass
class BoundingBox:

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float


STR2BB = {
    "Kenya": BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
    "Busia": BoundingBox(
        min_lon=33.88389587402344,
        min_lat=-0.04119872691853491,
        max_lon=34.44007873535156,
        max_lat=0.7779454563313616,
    ),
    "NorthMalawi": BoundingBox(min_lon=32.688, max_lon=35.772, min_lat=-14.636, max_lat=-9.231),
    "SouthMalawi": BoundingBox(min_lon=34.211, max_lon=35.772, min_lat=-17.07, max_lat=-14.636),
    "Rwanda": BoundingBox(min_lon=28.841, max_lon=30.909, min_lat=-2.854, max_lat=-1.034),
    "Togo": BoundingBox(min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625),
}
