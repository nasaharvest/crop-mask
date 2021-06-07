"""
Takes a trained model and runs it on an area
"""

from argparse import ArgumentParser
from clearml import Task
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import boto3
import logging
import subprocess
import sys

sys.path.append("..")

from src.models import Model  # noqa: E402
from src.analysis import plot_results  # noqa: E402
from src.ETL.split_tiff import run_split_tiff  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_prediction(
    model: Model,
    test_path: Path,
    save_dir: Path,
    with_forecaster: bool = False,
    plot_results_enabled: bool = False,
    disable_tqdm: bool = True,
) -> Optional[Path]:
    prefix = "forecasted" if with_forecaster else "normal"

    file_path = save_dir / f"preds_{prefix}_{test_path.name}.nc"
    if file_path.exists():
        logger.warning("File already generated. Skipping")
        return None

    out = model.predict(test_path, with_forecaster=with_forecaster, disable_tqdm=disable_tqdm)
    if plot_results_enabled:
        plot_results(out, test_path, savepath=save_dir, prefix=prefix)
    out.to_netcdf(file_path)
    return file_path


def gdal_merge(unmerged_tifs_folder: Path, output_file: Path) -> Path:
    command = f"gdal_merge.py -o {output_file} -of gtiff {unmerged_tifs_folder}/*"
    logger.info(f"Running: {command}")
    subprocess.Popen(command, shell=True).wait()
    if not output_file.exists():
        raise FileExistsError(
            f"Output file: {output_file} was not created from files in {unmerged_tifs_folder}"
        )
    return output_file


def upload_to_s3(merged_files: List[Path], upload_prefix: str):
    if not merged_files:
        logger.error(
            "upload_predictions was set to True but there are no merged predictions to upload. "
            "Ensure merge_predictions is set to True"
        )
        return

    s3 = boto3.resource("s3")
    for merged_file in merged_files:
        upload_path = f"output/{upload_prefix}_{merged_file.name}"
        logger.info(f"Uploading {str(merged_file)} to S3 {upload_path}")
        s3.Bucket("crop-mask-data").upload_file(str(merged_file), upload_path)


def run_inference(
    model_name: str,
    data_dir: str,
    local_path_to_tif_files: str,
    gdrive_path_to_tif_files: Optional[str] = None,
    split_tif_files: bool = False,
    plot_results_enabled: bool = False,
    predict_with_forecaster: bool = True,
    predict_without_forecaster: bool = True,
    predict_dir: str = "../data/predictions",
    merge_predictions: bool = False,
    upload_prefix: str = "",
    disable_tqdm=True,
):
    if not predict_with_forecaster and not predict_without_forecaster:
        raise ValueError(
            "One of 'predict_with_forecaster' and 'predict_without_forecaster' must be True"
        )

    if gdrive_path_to_tif_files:
        logger.info(
            f"Using rclone to copy files Google Drive: {gdrive_path_to_tif_files} "
            f"to {local_path_to_tif_files}"
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
    model = Model.load_from_checkpoint(f"{data_dir}/models/{model_name}.ckpt")
    save_dir = Path(predict_dir)
    save_dir_forecasted = save_dir / "forecasted"
    save_dir_normal = save_dir / "normal"
    for dir in [save_dir, save_dir_forecasted, save_dir_normal]:
        dir.mkdir(exist_ok=True)

    file_amount = len(test_files)
    logger.info(f"Beginning inference on tif files in {local_path_to_tif_files}")
    for i, test_path in enumerate(test_files):
        logger.info(f"Inference on file: {i+1}/{file_amount}")
        if predict_with_forecaster:
            make_prediction(
                model,
                test_path,
                save_dir_forecasted,
                with_forecaster=True,
                plot_results_enabled=plot_results_enabled,
                disable_tqdm=disable_tqdm,
            )
        if predict_without_forecaster:
            make_prediction(
                model,
                test_path,
                save_dir_normal,
                with_forecaster=False,
                plot_results_enabled=plot_results_enabled,
                disable_tqdm=disable_tqdm,
            )

    merged_files = []
    if merge_predictions:
        logger.info("Merge predictions enabled")
        if predict_with_forecaster:
            merged_files.append(
                gdal_merge(
                    unmerged_tifs_folder=save_dir_forecasted,
                    output_file=save_dir / "merged_forecasted.tif",
                )
            )
        if predict_without_forecaster:
            merged_files.append(
                gdal_merge(
                    unmerged_tifs_folder=save_dir_normal, output_file=save_dir / "merged_normal.tif"
                )
            )

    if upload_prefix:
        upload_to_s3(merged_files, upload_prefix)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--gdrive_path_to_tif_files", type=str, default=None)
    parser.add_argument("--local_path_to_tif_files", type=str)
    parser.add_argument("--split_tif_files", type=bool, default=False)
    parser.add_argument("--predict_with_forecaster", type=bool, default=False)
    parser.add_argument("--predict_without_forecaster", type=bool, default=True)
    parser.add_argument("--predict_dir", type=str, default="../data/predictions")
    parser.add_argument("--merge_predictions", type=bool, default=True)
    parser.add_argument("--upload_prefix", type=str, default="")
    parser.add_argument("--disable_tqdm", type=bool, default=True)

    params = parser.parse_args()
    Task.init(
        project_name="NASA Harvest",
        task_name=f"{params.model_name} {datetime.now()}",
        task_type=Task.TaskTypes.inference,
    )
    run_inference(
        model_name=params.model_name,
        data_dir=params.data_dir,
        gdrive_path_to_tif_files=params.gdrive_path_to_tif_files,
        local_path_to_tif_files=params.local_path_to_tif_files,
        split_tif_files=params.split_tif_files,
        predict_with_forecaster=params.predict_with_forecaster,
        predict_without_forecaster=params.predict_without_forecaster,
        predict_dir=params.predict_dir,
        merge_predictions=params.merge_predictions,
        upload_prefix=params.upload_prefix,
        disable_tqdm=params.disable_tqdm,
    )
