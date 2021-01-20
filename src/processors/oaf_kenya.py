import geopandas

from .base import BaseProcessor
from src.data_classes import KenyaOAF


class KenyaOAFProcessor(BaseProcessor):
    dataset = KenyaOAF.name

    def process(self) -> None:

        df = geopandas.read_file(self.raw_folder)

        df = df.rename(columns={"Lat": "lat", "Lon": "lon"})

        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
