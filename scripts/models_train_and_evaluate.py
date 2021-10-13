"""
Script to train and evaluate models (if necessary)
Input: ../data/models.json
Output: ../data/model_validation_metrics.json

If a model has already been trained but you'd like to retrain it,
delete the ckpt file in ../data/models

If a model has already been evauated but you'd like to reevaluate it,
delete the metric entry for the model in ../data/model_valudation_metrics.json
"""

from argparse import ArgumentParser

import json
import os
import sys

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")


from src.pipeline_funcs import model_pipeline  # noqa: E402
from src.models import Model  # noqa: E402
from src.utils import get_dvc_dir  # noqa: E402

models_folder = get_dvc_dir("models")
data_folder = models_folder.parent


def hparams_from_json(params):
    hparams = Model.add_model_specific_args(ArgumentParser()).parse_args()
    for key, val in params.items():
        if type(val) == list:
            val = ",".join(val)
        setattr(hparams, key, val)

    hparams.data_folder = data_folder
    hparams.model_dir = models_folder
    return hparams


if __name__ == "__main__":
    models_json = data_folder / "models.json"
    model_validation_metrics = data_folder / "model_metrics_validation.json"

    with models_json.open() as f:
        models_params_list = json.load(f)

    new_model_metrics = {}
    for params in models_params_list:
        hparams = hparams_from_json(params)
        try:
            model_name, metrics = model_pipeline(hparams)
            new_model_metrics[model_name] = metrics
        except Exception as e:
            print(f"\u2716 {str(e)}")

    with model_validation_metrics.open("w") as f:
        json.dump(new_model_metrics, f, ensure_ascii=False, indent=4, sort_keys=True)
