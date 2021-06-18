import pytorch_lightning as pl
import json
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

from src.models import Model

data_dir = Path(__file__).parent.parent / "data"


def test_one_model(model, alternate_test_sets: Tuple[str, ...] = ()):
    if alternate_test_sets:
        model.datasets = model.load_datasets(list(alternate_test_sets))
    trainer = pl.Trainer()
    trainer.test(model)
    formatted_metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()
                         if not k.startswith("encoded")}
    return formatted_metrics


def test_all_models():
    model_metrics = {}

    for model_path in (data_dir / "models").iterdir():
        if model_path.suffix != ".ckpt":
            continue
        
        print(model_path.name)
        model = Model.load_from_checkpoint(str(model_path))
        test_dataset = model.get_dataset("testing")
        country_key = "_".join(test_dataset.countries)
        if country_key not in model_metrics:
            model_metrics[country_key] = {
                "test_set_size": len(test_dataset),
                "test_set_crop_percentage": (test_dataset.instances_per_class[1]/len(test_dataset)),
                "models": {}
            }

        if model_path.name not in model_metrics[country_key]["models"]:
            model_metrics[country_key]["models"][model_path.name] = test_one_model(model)

    model_metrics_sorted = {
        country_key: {
            "test_set_size": value["test_set_size"],
            "test_set_crop_percentage": value["test_set_crop_percentage"],
            "models": OrderedDict(sorted(value["models"].items()))
        }
        for country_key, value in model_metrics.items()
    }

    with (data_dir / "model_metrics.json").open("w") as f:
        json.dump(model_metrics_sorted, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    test_all_models()
    # model = Model.load_from_checkpoint(str(data_dir / "models/Kenya1000.ckpt"))
    # test_one_model(model)
