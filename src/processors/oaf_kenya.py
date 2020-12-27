import geopandas

from .base import BaseProcessor


class KenyaOAFProcessor(BaseProcessor):
    dataset = "one_acre_fund_kenya"

    def process(self) -> None:

        df = geopandas.read_file(self.raw_folder)

        df = df.rename(columns={"Lat": "lat", "Lon": "lon"})

        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
