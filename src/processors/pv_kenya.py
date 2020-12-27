import geopandas
from pyproj import Transformer

from .base import BaseProcessor


class KenyaPVProcessor(BaseProcessor):
    dataset = "plant_village_kenya"

    def process(self) -> None:

        df = geopandas.read_file(self.raw_folder / "field_boundaries_pv_04282020.shp")

        x = df.geometry.centroid.x.values
        y = df.geometry.centroid.y.values

        transformer = Transformer.from_crs(crs_from=32636, crs_to=4326)

        lat, lon = transformer.transform(xx=x, yy=y)
        df["lat"] = lat
        df["lon"] = lon

        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
