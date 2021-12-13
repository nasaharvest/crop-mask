from unittest import TestCase

import json
from src.models.model import Model

from src.utils import data_dir
from src.ETL.constants import LAT, LON


class ModelBboxTest(TestCase):
    def test_model_bbox(self):
        # Read in models.json
        with (data_dir / "models.json").open("rb") as f:
            model_configurations = json.load(f)

        non_local_examples_in_eval = False
        for c in model_configurations:
            print("--------------------------------------------------")
            print(c["model_name"])
            for subset in ["validation", "testing"]:
                try:
                    df = Model.load_df(subset, c["train_datasets"], c["eval_datasets"])
                    is_local = (
                        (df[LAT] >= c["min_lat"])
                        & (df[LAT] <= c["max_lat"])
                        & (df[LON] >= c["min_lon"])
                        & (df[LON] <= c["max_lon"])
                    )
                    if is_local.all():
                        print(f"\u2714 {subset}: all {len(df)} examples are local")
                    else:
                        print(f"\u2716 {subset}: {len(df[~is_local])} examples are not local")
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
