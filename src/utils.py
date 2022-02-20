from typing import Tuple
import torch
import numpy as np
import random
from pathlib import Path
import subprocess

data_dir = Path(__file__).parent.parent / "data"
tifs_dir = data_dir / "tifs"
features_dir = data_dir / "features"
models_dir = data_dir / "models"
raw_dir = data_dir / "raw"
metrics_file = data_dir / "model_metrics_validation.json"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_dvc_dir(dvc_dir_name: str) -> Path:
    dvc_dir = Path(__file__).parent.parent / f"data/{dvc_dir_name}"
    if not dvc_dir.exists():
        subprocess.run(["dvc", "pull", f"data/{dvc_dir_name}"], check=True)
        if not dvc_dir.exists():
            raise FileExistsError(f"{str(dvc_dir)} was not found.")
        if not any(dvc_dir.iterdir()):
            raise FileExistsError(f"{str(dvc_dir)} should not be empty.")
    return dvc_dir


def memoize(f):
    memo = {}

    def helper(x="default"):
        if x not in memo:
            memo[x] = f() if x == "default" else f(x)
        return memo[x]

    return helper


def find_nearest(array, value: float) -> Tuple[float, int]:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    haversince formula, inspired by:
    https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude/41337005
    """
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))


def distance_point_from_center(lat_idx: int, lon_idx: int, tif) -> int:
    x_dist = np.abs((len(tif.x) - 1) / 2 - lon_idx)
    y_dist = np.abs((len(tif.y) - 1) / 2 - lat_idx)
    return x_dist + y_dist
