import pandas as pd
from pathlib import Path
import geopandas
from datetime import datetime

from typing import Optional

from src.exporters import KenyaOAFSentinelExporter
from src.data_classes import KenyaOAF
from src.utils import load_tif
from src.band_calculations import process_bands
from .base import BaseEngineer


# Pickled feature data relies on this class so it is temporarily necessary
class KenyaOneAcreFundDataInstance(KenyaOAF.instance):
    pass


class KenyaOAFEngineer(BaseEngineer):

    sentinel_dataset = KenyaOAFSentinelExporter.dataset
    dataset = KenyaOAF.name

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        pv_kenya = data_folder / "processed" / KenyaOAF.name / "data.geojson"
        assert pv_kenya.exists(), "Kenya One Acre Fund processor must be run to load labels"
        return geopandas.read_file(pv_kenya)

    def process_single_file(
        self,
        path_to_file: Path,
        nan_fill: float,
        max_nan_ratio: float,
        add_ndvi: bool,
        add_ndwi: bool,
        calculate_normalizing_dict: bool,
        start_date: datetime,
        days_per_timestep: int,
        is_test: bool,
    ) -> Optional[KenyaOAF.instance]:
        r"""
        Return a tuple of np.ndarrays of shape [n_timesteps, n_features] for
        1) the anchor (labelled)
        """

        da = load_tif(path_to_file, days_per_timestep=days_per_timestep, start_date=start_date)

        # first, we find the label encompassed within the da

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        overlap = self.labels[
            (
                (self.labels.lon <= max_lon)
                & (self.labels.lon >= min_lon)
                & (self.labels.lat <= max_lat)
                & (self.labels.lat >= min_lat)
            )
        ]
        if len(overlap) == 0:
            return None

        label_lat = overlap.iloc[0].lat
        label_lon = overlap.iloc[0].lon

        closest_lon, _ = self.find_nearest(da.x, label_lon)
        closest_lat, _ = self.find_nearest(da.y, label_lat)

        labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values
        labelled_array = process_bands(labelled_np,
                                       nan_fill=nan_fill,
                                       max_nan_ratio=max_nan_ratio,
                                       add_ndvi=add_ndvi,
                                       add_ndwi=add_ndwi)

        if (not is_test) and calculate_normalizing_dict:
            self.update_normalizing_values(self.normalizing_dict_interim, labelled_array)

        if labelled_array is not None:
            return KenyaOneAcreFundDataInstance(
                label_lat=label_lat,
                label_lon=label_lon,
                instance_lat=closest_lat,
                instance_lon=closest_lon,
                labelled_array=labelled_array,
            )
        else:
            return None
