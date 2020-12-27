import pandas as pd

from .base import BaseProcessor


class GeoWikiProcessor(BaseProcessor):

    dataset = "geowiki_landcover_2017"

    def load_raw_data(self, participants: str) -> pd.DataFrame:

        participants_to_file_labels = {
            "all": "all",
            "students": "con",
            "experts": "exp",
        }

        file_label = participants_to_file_labels.get(participants, participants)
        assert (
            file_label in participants_to_file_labels.values()
        ), f"Unknown participant {file_label}"

        return pd.read_csv(
            self.raw_folder / f"loc_{file_label}{'_2' if file_label == 'all' else ''}.txt",
            sep="\t",
        )

    def process(self, participants: str = "all") -> None:

        location_data = self.load_raw_data(participants)

        # first, we find the mean sumcrop calculated per location
        mean_per_location = (
            location_data[["location_id", "sumcrop", "loc_cent_X", "loc_cent_Y"]]
            .groupby("location_id")
            .mean()
        )

        # then, we rename the columns
        mean_per_location = mean_per_location.rename(
            {"loc_cent_X": "lon", "loc_cent_Y": "lat", "sumcrop": "mean_sumcrop"},
            axis="columns",
            errors="raise",
        )
        # then, we turn it into an xarray with x and y as indices
        output_xr = (
            mean_per_location.reset_index().set_index(["lon", "lat"])["mean_sumcrop"].to_xarray()
        )

        # and save
        output_xr.to_netcdf(self.output_folder / "data.nc")
