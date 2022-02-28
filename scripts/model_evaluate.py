"""
Script to evaluate an individual model
"""
import os
import sys

from pathlib import Path

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.utils import models_dir
from src.pipeline_funcs import run_evaluation  # noqa: E402

if __name__ == "__main__":
    model_name = "Rwanda_2019"
    model_ckpt_path = models_dir / f"{model_name}.ckpt"
    _, metrics = run_evaluation(model_ckpt_path)
    print(metrics)
