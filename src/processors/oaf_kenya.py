import geopandas

from .base import BaseProcessor
from src.dataset_config import DatasetName


class KenyaOAFProcessor(BaseProcessor):
    dataset = DatasetName.KenyaOAF.value

    def process(self) -> None:

        df = geopandas.read_file(self.raw_folder)

        df = df.rename(columns={"Lat": "lat", "Lon": "lon"})

        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
