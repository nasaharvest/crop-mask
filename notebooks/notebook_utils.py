from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Tuple

import numpy as np
import pandas as pd
import pickle
import sys
import torch
from pathlib import Path

sys.path.append("..")
from src.ETL.constants import FEATURE_PATH  # noqa: E402s
from src.models import Model  # noqa: E402s


def get_validation_df(model_name: str, default_threshold: float = 0.5) -> pd.DataFrame:
    """Gets validation set for model as a pandas dataframe"""
    model = Model.load_from_checkpoint(f"../data/models/{model_name}.ckpt")
    model = model.eval()

    # Load model validation set
    val = model.get_dataset(
        subset="validation", normalizing_dict=model.normalizing_dict, upsample=False
    )

    df = val.df

    def get_feature(feature_file):
        with Path(feature_file).open("rb") as f:
            f = pickle.load(f)
        return f.instance_lat, f.instance_lon, f.source_file

    df["instance_lat"], df["instance_lon"], df["source_file"] = zip(
        *df[FEATURE_PATH].apply(get_feature)
    )
    df["y_true"] = df["crop_probability"].apply(lambda prob: 1 if prob > 0.5 else 0)

    # Make predictions on validation set
    x = torch.stack([v[0] for v in val])
    with torch.no_grad():
        # model(x) is indexed to get the local predictions (not global at index 0)
        df["y_pred_decimal"] = model(x).numpy().flatten()

    df["y_pred"] = df["y_pred_decimal"].apply(lambda pred: 1 if pred > default_threshold else 0)
    df["errors"] = df["y_true"] != df["y_pred"]
    return df


def best_f1_threshold(model_name: str, plot: bool = False) -> Tuple[float, float]:
    """Plots precision recall graphs for model"""
    df = get_validation_df(model_name)
    thresholds = np.arange(0, 1.0, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    y_true = df["y_true"]
    for threshold in thresholds:
        y_pred = df["y_pred_decimal"].apply(lambda pred: 1 if pred > threshold else 0)
        f1_scores.append(f1_score(y_true, y_pred))
        precision_scores.append(precision_score(y_true, y_pred))
        recall_scores.append(recall_score(y_true, y_pred))

    best_i = np.argmax(f1_scores)
    threshold = round(thresholds[best_i], 3)
    best_f1 = round(f1_scores[best_i], 3)

    if plot:
        _, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(recall_scores, precision_scores)
        axs[0].set_xlabel("Recall")
        axs[0].set_ylabel("Precision")
        axs[0].set_title("Precision Recall Curve")
        axs[0].plot(
            recall_scores[best_i], precision_scores[best_i], "r*", label=f"Best F1: {best_f1}"
        )
        axs[0].legend()

        axs[1].plot(thresholds, f1_scores)
        axs[1].set_xlabel("Threshold")
        axs[1].set_ylabel("F1-Score")
        axs[1].set_title("F1 Scores")
        axs[1].plot(thresholds[best_i], f1_scores[best_i], "r*")
        axs[1].annotate(
            f"F1: {best_f1}\nThreshold: {threshold}",
            (thresholds[best_i], f1_scores[best_i] - 0.1),
        )

        print(f"Threshold: {thresholds[best_i]}, F1: {best_f1}")
    return thresholds[best_i], best_f1
