import json
from unittest import TestCase

from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import LAT, LON

from src.models.model import Model


class ModelBboxTest(TestCase):
    def test_model_bbox(self):
        # Read in models.json
        with (PROJECT_ROOT / DataPaths.METRICS).open("rb") as f:
            models_dict = json.load(f)

        non_local_examples_in_eval = False
        for model_name, _ in models_dict.items():
            print("--------------------------------------------------")
            print(model_name)
            model = Model.load_from_checkpoint(
                PROJECT_ROOT / DataPaths.MODELS / f"{model_name}.ckpt"
            )
            for subset in ["validation", "testing"]:
                try:
                    df = Model.load_df(
                        subset, model.hparams.train_datasets, model.hparams.eval_datasets
                    )
                    if "bbox" in model.hparams:
                        bbox = model.hparams.bbox
                    else:
                        bbox = model.hparams

                    is_local = (
                        (df[LAT] >= bbox.min_lat)
                        & (df[LAT] <= bbox.max_lat)
                        & (df[LON] >= bbox.min_lon)
                        & (df[LON] <= bbox.max_lon)
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
