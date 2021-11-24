from unittest import TestCase

import json

from src.utils import data_dir
from src.ETL.constants import IS_LOCAL, LAT, LON
from src.datasets_labeled import labeled_datasets
from src.ETL.ee_boundingbox import BoundingBox
from src.models.data import CropDataset


class ModelBooxTest(TestCase):
    def test_model_bbox(self):
        # Read in models.json
        with (data_dir / "models.json").open("rb") as f:
            model_configurations = json.load(f)

        non_local_examples_in_eval = False
        for c in model_configurations:
            target_bbox = BoundingBox(
                min_lon=c["min_lon"],
                max_lon=c["max_lon"],
                min_lat=c["min_lat"],
                max_lat=c["max_lat"],
            )
            ds = [d for d in labeled_datasets if d.dataset in c["eval_datasets"]]
            print("--------------------------------------------------")
            print(c["model_name"])
            for subset in ["validation", "testing"]:
                try:
                    df = CropDataset._load_df_from_datasets(
                        datasets=ds,
                        subset=subset,
                        target_bbox=target_bbox,
                        is_local_only=False,
                        up_to_year=2050,
                    )
                    if df[IS_LOCAL].all():
                        print(f"\u2714 {subset}: all {len(df)} examples are local")
                    else:
                        print(f"\u2716 {subset}: {len(df[~df[IS_LOCAL]])} examples are not local")
                        print(
                            f"bbox should contain: "
                            f"min_lat={df[LAT].min()}, max_lat={df[LAT].max()}, "
                            f"min_lon={df[LON].min()}, max_lon={df[LON].max()}"
                        )
                        non_local_examples_in_eval = True
                except ValueError as e:
                    print(f"{subset}: {e}")
        self.assertFalse(
            non_local_examples_in_eval,
            "Some evaluation sets contain non-local examples, check logs.",
        )
