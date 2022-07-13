"""
Script to evaluate an individual model
"""
import os
import sys

from openmapflow.config import PROJECT_ROOT, DataPaths

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.pipeline_funcs import run_evaluation  # noqa: E402

if __name__ == "__main__":
    model_name = "Ethiopia_Tigray_2021"
    model_ckpt_path = PROJECT_ROOT / DataPaths / f"{model_name}.ckpt"
    _, metrics = run_evaluation(model_ckpt_path)
    print(metrics)
