import os
import sys
import torch

from pathlib import Path
from tqdm import tqdm

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("..")

from src.models.model import Model

if __name__ == "__main__":
    model_dir = Path(__file__).parent.parent / f"data/models"
    for model_ckpt in tqdm(list(model_dir.glob("*.ckpt"))):

        model_pt_path = str(model_ckpt).replace(".ckpt", ".pt")
        if Path(model_pt_path).exists():
            m = torch.jit.load(model_pt_path)
            continue

        model = Model.load_from_checkpoint(str(model_ckpt))
        sm = torch.jit.script(model)
        sm.save(model_pt_path)
