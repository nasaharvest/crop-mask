"""
Trains the models
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import logging

module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.datasets_labeled import labeled_datasets  # noqa: E402
from src.models import Model, train_model  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_datasets_str = ",".join(
    [ld.dataset for ld in labeled_datasets if ld.dataset != "one_acre_fund"]
)

data_folder = str(Path(module_path) / "data")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--target_bbox_key", type=str, default="Ethiopia_Tigray")
    parser.add_argument("--train_datasets", type=str, default=all_datasets_str)
    parser.add_argument("--eval_datasets", type=str, default="Ethiopia,")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data_folder", type=str, default=data_folder)
    parser.add_argument("--model_dir", type=str, default=data_folder + "/models")

    model_args = Model.add_model_specific_args(parser).parse_args()
    model = Model(model_args)
    train_model(model, model_args)

    print("Done")
