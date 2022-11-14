from argparse import ArgumentParser

from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.inference import Inference

from src.models.model import Model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tif_path", type=str)
    parser.add_argument("--dest_path", type=str)

    args = parser.parse_args()
    model = Model.load_from_checkpoint(PROJECT_ROOT / DataPaths.MODELS / f"{args.model_name}.ckpt")
    Inference(model).run(local_path=args.tif_path, dest_path=args.dest_path)
