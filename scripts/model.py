import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append("..")

from src.models import Model
from src.models import train_model


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)

    model_args = Model.add_model_specific_args(parser).parse_args()
    model = Model(model_args)

    train_model(model, model_args)
