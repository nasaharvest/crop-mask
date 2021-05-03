import pytorch_lightning as pl
import json
from pathlib import Path

from src.models import Model

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
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
            model = Model.load_from_checkpoint(str(model_path))
            trainer = pl.Trainer()
            trainer.test(model)
            formatted_metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()}
            new_model_metrics[model_path.stem] = formatted_metrics

    with model_metrics_path.open("w") as f:
        json.dump(new_model_metrics, f, ensure_ascii=False, indent=4)
