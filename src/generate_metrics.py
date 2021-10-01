import pytorch_lightning as pl
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

import os
import sys

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("..")
from src.models import Model  # noqa: E402
from src.bounding_boxes import bounding_boxes  # noqa: E402

data_dir = Path(__file__).parent.parent / "data"


def get_metrics(
    model: pl.LightningModule,
    test_mode: bool,
    alternate_test_sets: Tuple[str, ...] = (),
    alternate_bbox_key=None,
):

    if alternate_test_sets:
        if alternate_bbox_key:
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

    metric_type = "testing" if test_mode else "validation"
    local_dataset = model.get_dataset(
        metric_type, is_local_only=True, upsample=False, normalizing_dict=model.normalizing_dict
    )

    return_obj = {
        "metrics": metrics,
        "training_data": model.hparams.train_datasets,
        f"local_{metric_type}_dataset_size": len(local_dataset),
        f"local_{metric_type}_crop_percentage": local_dataset.crop_percentage,
    }

    print(f"\n{return_obj}")
    return return_obj


def get_metrics_for_all_models(test_mode: bool = False):
    model_metrics: Dict[str, Dict] = {}
    model_paths = [p for p in (data_dir / "models").iterdir() if p.suffix == ".ckpt"]
    for i, model_path in enumerate(model_paths):
        print(f"\n{i+1}/{len(model_paths)}: {model_path.name}")

        # datetime.fromtimestamp(p.stat().st_ctime).strftime("%Y-%m-%d-%H:%M")

        model = Model.load_from_checkpoint(str(model_path))

        key = model.hparams.eval_datasets

        if key not in model_metrics:
            model_metrics[key] = {"models": {}}

        if model_path.name not in model_metrics[key]["models"]:
            metrics = get_metrics(model, test_mode)
            model_metrics[key]["models"][model_path.name] = metrics

    for key, value in model_metrics.items():
        model_metrics[key]["models"] = OrderedDict(sorted(value["models"].items()))

    metric_type = "testing" if test_mode else "validation"
    with (data_dir / f"model_metrics_{metric_type}.json").open("w") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    # Experiment running models on different validation sets
    # eval_sets = ["Rwanda", "Kenya", "Togo", "Uganda"]
    eval_sets = ["Ethiopia"]
    metrics_for_datasets: Dict[str, Dict] = {}
    for model_name in ["Global", "Rwanda", "Kenya", "Togo", "Uganda", "Uganda_surrounding_5"]:
        metrics_for_datasets[model_name] = {}
        model = Model.load_from_checkpoint(str(data_dir / f"models/{model_name}.ckpt"))

        for eval_set in eval_sets:

            metrics = get_metrics(
                model,
                test_mode=False,
                alternate_test_sets=(eval_set,),
                alternate_bbox_key="Ethiopia_Tigray",
            )["metrics"]

            metrics_for_datasets[model_name][eval_set] = metrics

    print("\n")
    print(metrics_for_datasets)
    get_metrics_for_all_models(test_mode=False)
