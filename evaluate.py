"""
Script to evaluate an individual model
"""
from openmapflow.config import PROJECT_ROOT, DataPaths

from src.pipeline_funcs import run_evaluation  # noqa: E402

model_name = "Rwanda_2019_skip_era5"
model_ckpt_path = PROJECT_ROOT / DataPaths.MODELS / f"{model_name}.ckpt"
_, metrics = run_evaluation(model_ckpt_path)
print(metrics)
