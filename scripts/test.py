import pytorch_lightning as pl
import json
from pathlib import Path
from typing import Tuple

from src.models import Model

data_dir = Path(__file__).parent.parent / "data"


def test_one_model(model_path: Path, alternate_test_sets: Tuple[str, ...] = ()):
    model = Model.load_from_checkpoint(str(model_path))
    if alternate_test_sets:
        model.datasets = model.load_datasets(list(alternate_test_sets))
    trainer = pl.Trainer()
    trainer.test(model)
    formatted_metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()}
    return formatted_metrics


def test_all_models():
    model_metrics_path = data_dir / "model_metrics.json"
    with model_metrics_path.open() as f:
        prev_model_metrics = json.load(f)

    new_model_metrics = {}
    for model_path in (data_dir / "models").iterdir():
        if model_path.suffix != ".ckpt":
            continue
        elif model_path.stem in prev_model_metrics:
            new_model_metrics[model_path.stem] = prev_model_metrics[model_path.stem]
        else:
            new_model_metrics[model_path.stem] = test_one_model(model_path)

    with model_metrics_path.open("w") as f:
        json.dump(new_model_metrics, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # test_all_models()
    test_one_model(data_dir / "models/Kenya.ckpt", alternate_test_sets=("Uganda",))
