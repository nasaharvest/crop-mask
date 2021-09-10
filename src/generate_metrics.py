import pytorch_lightning as pl
import json
from collections import OrderedDict
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import os
import sys

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("..")
from src.models import Model
from src.bounding_boxes import bounding_boxes

data_dir = Path(__file__).parent.parent / "data"


def get_metrics(
    model: pl.LightningModule,
    test_mode: bool,
    alternate_test_sets: Tuple[str, ...] = (),
    alternate_bbox_key=None,
):
    if alternate_test_sets:
        model.target_bbox = bounding_boxes[alternate_bbox_key]
        model.eval_datasets = model.load_datasets(list(alternate_test_sets), subset="evaluation")

    trainer = pl.Trainer()
    if test_mode:
        trainer.test(model)
    else:
        trainer.model = model
        trainer.main_progress_bar = tqdm(disable=True)
        trainer.run_evaluation(test_mode=False)

    metrics = {
        k: round(float(v), 4)
        for k, v in trainer.callback_metrics.items()
        if (not k.startswith("encoded")) and ("_global_" not in k)
    }
    return_obj = {"metrics": metrics, "data": model.hparams.train_datasets}
    print(f"\n{return_obj}")
    return return_obj


def get_metrics_for_all_models(test_mode: bool = False):
    metric_type = "testing" if test_mode else "validation"
    model_metrics = {}
    model_paths = [p for p in (data_dir / "models").iterdir() if p.suffix == ".ckpt"]
    for i, model_path in enumerate(model_paths):
        print(f"\n{i+1}/{len(model_paths)}: {model_path.name}")

        model = Model.load_from_checkpoint(str(model_path))

        local_dataset = model.get_dataset(
            metric_type, is_local_only=True, upsample=False, normalizing_dict=model.normalizing_dict
        )

        key = model.hparams.target_bbox_key

        if key not in model_metrics:
            model_metrics[key] = {
                f"local_{metric_type}_dataset_size": len(local_dataset),
                f"local_{metric_type}_crop_percentage": local_dataset.crop_percentage,
                "models": {},
            }

        if model_path.name not in model_metrics[key]["models"]:
            metrics = get_metrics(model, test_mode)
            model_metrics[key]["models"][model_path.name] = metrics

    for country_key, value in model_metrics.items():
        model_metrics[country_key]["models"] = OrderedDict(sorted(value["models"].items()))

    with (data_dir / f"model_metrics_{metric_type}.json").open("w") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    model = Model.load_from_checkpoint(str(data_dir / "models/Kenya.ckpt"))
    get_metrics(
        model, test_mode=False, alternate_test_sets=("Uganda",), alternate_bbox_key="Uganda"
    )
    # get_metrics_for_all_models(test_mode=False)
