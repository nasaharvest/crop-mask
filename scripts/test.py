import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser

from src.models import Model


def get_checkpoint(version: int) -> str:

    log_folder = (
        Path(__file__).parent.parent
        / "data"
        / "lightning_logs"
        / f"version_{version}"
        / "checkpoints"
    )
    checkpoint = list(log_folder.glob("*.ckpt"))
    return str(checkpoint[0])


def test_model():
    parser = ArgumentParser()

    parser.add_argument("--version", type=int, default=0)

    args = parser.parse_args()

    model_path = get_checkpoint(args.version)

    print(f"Using model {model_path}")

    model = Model.load_from_checkpoint(model_path)

    trainer = pl.Trainer()
    trainer.test(model)


if __name__ == "__main__":
    test_model()
