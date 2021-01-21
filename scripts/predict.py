from pathlib import Path
import sys
import os
from typing import Optional

sys.path.append("..")

from clearml import Task
from argparse import ArgumentParser
from src.models import Model
from src.analysis import plot_results


def make_prediction(
        model: Model,
        test_path: Path,
        save_dir: Path,
        with_forecaster: bool = False,
        plot_results_enabled: bool = False) -> Optional[Path]:
    if with_forecaster:
        prefix = "forecasted_"
    else:
        prefix = "normal_"

    file_path = save_dir / f"preds_{prefix}{test_path.name}.nc"
    if file_path.exists():
        print("File already generated. Skipping")
        return

    out = model.predict(test_path, with_forecaster=with_forecaster)
    if plot_results_enabled:
        plot_results(out, test_path, savepath=save_dir, prefix=prefix)
    out.to_netcdf(file_path)
    return file_path


def gdal_merge(save_dir: Path) -> Path:
    file_list = save_dir.glob('*.nc')
    files_string = " ".join([str(file) for file in file_list])
    merged_file = save_dir / "merged.tif"
    command = f"gdal_merge.py -o {merged_file} -of gtiff {files_string}"
    os.system(command)
    return merged_file


def main(path_to_tif_files: str, model_name: str, merge_predictions: bool = False, data_dir: str = "../data"):
    test_folder = Path(path_to_tif_files)
    test_files = test_folder.glob("*.tif")

    print(f"Using model {model_name}")
    model = Model.load_from_checkpoint(f"{data_dir}/models/{model_name}")

    save_dir = Path(data_dir) / "predictions"
    save_dir.mkdir(exist_ok=True)

    for test_path in test_files:
        print(f"Running for {test_path}")
        make_prediction(model, test_path, save_dir, with_forecaster=True, plot_results_enabled=False)
        make_prediction(model, test_path, save_dir, with_forecaster=False, plot_results_enabled=False)

    if merge_predictions:
        gdal_merge(save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument("--path_to_tif_files", type=str)
    parser.add_argument("--merge_predictions", type=bool, default=False)
    parser.add_argument("--data_dir", type=bool, default=False)
    params = parser.parse_args()
    Task.init(project_name="NASA Harvest", task_name=f"Inference with model {params.model_name}")
    main(params.path_to_tif_files, params.model_name, params.merge_predictions, params.data_dir)
