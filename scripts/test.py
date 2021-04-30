import pytorch_lightning as pl
import json
from pathlib import Path

from src.models import Model

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    model_metrics_path = data_dir / "model_metrics.json"
    with model_metrics_path.open() as f:
        model_metrics = json.load(f)

    for model_path in (data_dir / "models").iterdir():
        if model_path.suffix != ".ckpt" or model_path.stem in model_metrics:
            continue

        model = Model.load_from_checkpoint(str(model_path))
        trainer = pl.Trainer()
        trainer.test(model)
        model_metrics[model_path.stem] = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()}

    with model_metrics_path.open("w") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=4)
