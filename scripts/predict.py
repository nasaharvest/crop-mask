from pathlib import Path
import sys
from typing import Optional
import subprocess

sys.path.append("..")

from clearml import Task
import logging
from argparse import ArgumentParser
from src.models import Model
from src.analysis import plot_results
from src.split_tiff import run_split_tiff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    disable_tqdm: bool = True,
) -> Optional[Path]:
    prefix = get_file_prefix(with_forecaster)

    file_path = save_dir / f"preds_{prefix}_{test_path.name}.nc"
    if file_path.exists():
        logger.warning("File already generated. Skipping")
        return None

    out = model.predict(test_path, with_forecaster=with_forecaster, disable_tqdm=disable_tqdm)
    if plot_results_enabled:
        plot_results(out, test_path, savepath=save_dir, prefix=prefix)
    out.to_netcdf(file_path)
    return file_path


def gdal_merge(save_dir: Path, with_forecaster: bool = False) -> Path:
    prefix = get_file_prefix(with_forecaster)
    file_list = save_dir.glob(f"*{prefix}*.nc")
    files_string = " ".join([str(file) for file in file_list])
    merged_file = save_dir / f"merged_{prefix}.tif"
    logger.info(f"Merging files *{prefix}*.nc to {merged_file}")
    command = f"gdal_merge.py -o {merged_file} -of gtiff {files_string}"
    subprocess.Popen(command, shell=True).wait()
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
    predict_dir: str = "../data/predictions",
    disable_tqdm=True,
):
    if not predict_with_forecaster and not predict_without_forecaster:
        raise ValueError(
            "One of 'predict_with_forecaster' and 'predict_without_forecaster' must be True"
        )

    if gdrive_path_to_tif_files:
        logger.info(
            f"Using rclone to copy files Google Drive: {gdrive_path_to_tif_files} to {local_path_to_tif_files}"
        )
        subprocess.run(
            ["rclone", "copy", gdrive_path_to_tif_files, local_path_to_tif_files], check=True
        )
        logger.info(f"All files successfully copied over to {local_path_to_tif_files}")

    if split_tif_files:
        logger.info(f"Split tif is enabled, splitting all tifs in {local_path_to_tif_files}")
        local_path_to_tif_files = run_split_tiff(local_path_to_tif_files)
        logger.info(f"All files successfully split in {local_path_to_tif_files}")

    test_folder = Path(local_path_to_tif_files)
    test_files = list(test_folder.glob("*.tif"))

    logger.info(f"Using model {model_name}")
    model = Model.load_from_checkpoint(f"{model_dir}/models/{model_name}.ckpt")

    save_dir = Path(predict_dir)
    save_dir.mkdir(exist_ok=True)

    file_amount = len(test_files)
    logger.info(f"Beginning inference on tif files in {local_path_to_tif_files}")
    for i, test_path in enumerate(test_files):
        logger.info(f"Inference on file: {i+1}/{file_amount}")
        if predict_with_forecaster:
            make_prediction(
                model,
                test_path,
                save_dir,
                with_forecaster=True,
                plot_results_enabled=plot_results_enabled,
                disable_tqdm=disable_tqdm,
            )
        if predict_without_forecaster:
            make_prediction(
                model,
                test_path,
                save_dir,
                with_forecaster=False,
                plot_results_enabled=plot_results_enabled,
                disable_tqdm=disable_tqdm,
            )

    if merge_predictions:
        logger.info("Merge predictions enabled")
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
    parser.add_argument("--split_tif_files", type=bool, default=False)
    parser.add_argument("--merge_predictions", type=bool, default=True)
    parser.add_argument("--predict_with_forecaster", type=bool, default=True)
    parser.add_argument("--predict_without_forecaster", type=bool, default=False)
    parser.add_argument("--predict_dir", type=str, default="../data/predictions")
    parser.add_argument("--disable_tqdm", type=bool, default=True)

    params = parser.parse_args()
    Task.init(
        project_name="NASA Harvest",
        task_name=params.model_name,
        task_type=Task.TaskTypes.inference,
    )
    run_inference(
        model_name=params.model_name,
        model_dir=params.model_dir,
        gdrive_path_to_tif_files=params.gdrive_path_to_tif_files,
        local_path_to_tif_files=params.local_path_to_tif_files,
        split_tif_files=params.split_tif_files,
        merge_predictions=params.merge_predictions,
        predict_with_forecaster=params.predict_with_forecaster,
        predict_without_forecaster=params.predict_without_forecaster,
        predict_dir=params.predict_dir,
        disable_tqdm=params.disable_tqdm,
    )
