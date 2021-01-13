from pathlib import Path
import sys

sys.path.append("..")

from clearml import Task
from argparse import ArgumentParser
from src.models import Model
from src.analysis import plot_results

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str)
    parser.add_argument("--path_to_tif_files", type=str)

    params = parser.parse_args()

    data_dir = "../data"

    test_folder = Path(params.path_to_tif_files)
    test_files = test_folder.glob("*.tif")
    print(test_files)

    task = Task.init(project_name="NASA Harvest", task_name=f"Inference with model {params.model_name}")
    print(f"Using model {params.ckpt_path}")
    model = Model.load_from_checkpoint(f"{data_dir}")

    for test_path in test_files:

        save_dir = Path(data_dir) / "Autoencoder"
        save_dir.mkdir(exist_ok=True)

        print(f"Running for {test_path}")

        savepath = save_dir / f"preds_{test_path.name}"
        if savepath.exists():
            print("File already generated. Skipping")
            continue

        out_forecasted = model.predict(test_path, with_forecaster=True)
        plot_results(out_forecasted, test_path, savepath=save_dir, prefix="forecasted_")

        out_normal = model.predict(test_path, with_forecaster=False)
        plot_results(out_normal, test_path, savepath=save_dir, prefix="full_input_")

        out_forecasted.to_netcdf(save_dir / f"preds_forecasted_{test_path.name}.nc")
        out_normal.to_netcdf(save_dir / f"preds_normal_{test_path.name}.nc")
