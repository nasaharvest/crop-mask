from argparse import ArgumentParser

from src.inference import Inference
from src.models.model import Model
from src.utils import models_dir

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tif_path", type=str)
    parser.add_argument("--dest_path", type=str)

    args = parser.parse_args()
    model = Model.load_from_checkpoint(models_dir / f"{args.model_name}.ckpt")
    Inference(model).run(local_path=args.tif_path, dest_path=args.dest_path)
