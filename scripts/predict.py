from pathlib import Path
import sys
import os
from typing import Optional
import subprocess

sys.path.append("..")

from clearml import Task
from argparse import ArgumentParser
from src.models import Model
from src.analysis import plot_results
from src.split_tiff import run_split_tiff


def get_file_prefix(with_forecaster: bool):
    if with_forecaster:
        return "forecasted"
    else:
        return "normal"


def make_prediction(
    model: Model,
    test_path: Path,
    save_dir: Path,
    with_forecaster: bool = False,
    plot_results_enabled: bool = False,
) -> Optional[Path]:
    prefix = get_file_prefix(with_forecaster)

    file_path = save_dir / f"preds_{prefix}_{test_path.name}.nc"
    if file_path.exists():
        print("File already generated. Skipping")
        return None

    out = model.predict(test_path, with_forecaster=with_forecaster)
    if plot_results_enabled:
        plot_results(out, test_path, savepath=save_dir, prefix=prefix)
    out.to_netcdf(file_path)
    return file_path


def gdal_merge(save_dir: Path, with_forecaster: bool = False) -> Path:
    prefix = get_file_prefix(with_forecaster)
    file_list = save_dir.glob(f"*{prefix}*.nc")
    files_string = " ".join([str(file) for file in file_list])
    merged_file = save_dir / f"merged_{prefix}.tif"
    command = f"gdal_merge.py -o {merged_file} -of gtiff {files_string}"
    os.system(command)
    return merged_file


def run_inference(
    model_name: str,
    model_dir: str,
    local_path_to_tif_files: str,
    gdrive_path_to_tif_files: Optional[str] = None,
    split_tif_files: bool = False,
    merge_predictions: bool = False,
    plot_results_enabled: bool = False,
    predict_with_forecaster: bool = True,
    predict_without_forecaster: bool = True,
    predict_dir: str = "../data/predictions"
):
    if not predict_with_forecaster and not predict_without_forecaster:
        raise ValueError(
            "One of 'predict_with_forecaster' and 'predict_without_forecaster' must be True"
        )

    if gdrive_path_to_tif_files:
        subprocess.run(["rclone", "copy", gdrive_path_to_tif_files, local_path_to_tif_files])

    if split_tif_files:
        local_path_to_tif_files = run_split_tiff(local_path_to_tif_files)

    test_folder = Path(local_path_to_tif_files)
    test_files = test_folder.glob("*.tif")

    print(f"Using model {model_name}")
    model = Model.load_from_checkpoint(f"{model_dir}/models/{model_name}.ckpt")

    save_dir = Path(predict_dir)
    save_dir.mkdir(exist_ok=True)

    for test_path in test_files:
        print(f"Running for {test_path}")
        if predict_with_forecaster:
            make_prediction(
                model,
                test_path,
                save_dir,
                with_forecaster=True,
                plot_results_enabled=plot_results_enabled,
            )
        if predict_without_forecaster:
            make_prediction(
                model,
                test_path,
                save_dir,
                with_forecaster=False,
                plot_results_enabled=plot_results_enabled,
            )

    if merge_predictions:
        if predict_with_forecaster:
            gdal_merge(save_dir, with_forecaster=True)
        if predict_without_forecaster:
            gdal_merge(save_dir, with_forecaster=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_dir", type=str, default="../data")
    parser.add_argument("--gdrive_path_to_tif_files", type=str, default=None)
    parser.add_argument("--local_path_to_tif_files", type=str)
    parser.add_argument("--merge_predictions", type=bool, default=True)
    parser.add_argument("--predict_with_forecaster", type=bool, default=True)
    parser.add_argument("--predict_without_forecaster", type=bool, default=False)
    parser.add_argument("--predict_dir", type=str, default="../data/predictions")

    params = parser.parse_args()
    Task.init(
        project_name="NASA Harvest",
        task_name=f"Inference with model {params.model_name}",
        task_type=Task.TaskTypes.inference,
    )
    run_inference(
        model_name=params.model_name,
        model_dir=params.model_dir,
        gdrive_path_to_tif_files=params.gdrive_path_to_tif_files,
        local_path_to_tif_files=params.local_path_to_tif_files,
        merge_predictions=params.merge_predictions,
        predict_with_forecaster=params.predict_with_forecaster,
        predict_without_forecaster=params.predict_without_forecaster,
        predict_dir=params.predict_dir
    )
