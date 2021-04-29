"""
Trains the models
"""

import sys
from argparse import ArgumentParser
import subprocess
from shutil import copyfile
from pathlib import Path
from clearml import Task
import logging

sys.path.append("..")

from src.models import Model, train_model  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training(parser):
    model_args = Model.add_model_specific_args(parser).parse_args()

    Task.init(project_name="NASA Harvest", task_name=f"{model_args.model_name}")

    model = Model(model_args)
    train_model(model, model_args)

    output_vol = Path("/vol")
    if output_vol.exists():
        # Push model to dvc, this will update the models.dvc file
        subprocess.run(["dvc", "add", str(model.data_folder / "models")], check=True)
        logger.info("Successfully ran dvc add")
        subprocess.run(["dvc", "push"], check=True)
        logger.info("Successfully ran dvc push")
        # Copy over updates file to output volume to make it accessible
        copyfile(str(model.data_folder / "models.dvc"), str(output_vol / "models.dvc"))
        logger.info("Successfully copied models.dvc file to host.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--show_progress_bar", type=bool, default=False)

    run_training(parser)
