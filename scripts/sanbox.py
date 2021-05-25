from src.models import Model
import torch
import json
import numpy as np

from argparse import ArgumentParser, Namespace

if __name__ == "__main__":
    data_dir = "../data"
    model_name = "Togo"

    model = Model.load_from_checkpoint(f"{data_dir}/models/{model_name}.ckpt")
    model.save()

    # model2 = torch.jit.load(f"{data_dir}/models/{model_name}.pt")
    # test = model2.get_normalizing_dict()
    # test2 = model2.normalizing_dict_jit
    # print("ok")

