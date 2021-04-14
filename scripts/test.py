import pytorch_lightning as pl
from pathlib import Path

from src.models import Model


def get_model_path(name: str) -> str:
    model_path = (
        Path(__file__).parent.parent
        / "data"
        / "models"
        / f"{name}.ckpt"
    )
    return str(model_path)


if __name__ == "__main__":
    name = "mali"
    print(f"Using model {name}")
    model_path = get_model_path(name)
    model = Model.load_from_checkpoint(model_path)
    trainer = pl.Trainer()
    trainer.test(model)
