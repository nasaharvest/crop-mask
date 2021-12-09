from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import pickle
import sys
import torch

sys.path.append("..")

from src.models import Model  # noqa: E402s


def get_validation_df(model_name: str, default_threshold: float = 0.5) -> pd.DataFrame:
    """Gets validation set for model as a pandas dataframe"""
    model = Model.load_from_checkpoint(f"../data/models/{model_name}.ckpt")
    model = model.eval()

    # Load model validation set
    val = model.get_dataset(
        subset="validation", normalizing_dict=model.normalizing_dict, upsample=False
    )

    # Load validation features
    validation_features = []
    for i, target_file in tqdm(enumerate(val.pickle_files)):
        with target_file.open("rb") as f:
            feature = pickle.load(f)
            # Check that model valdilation set and features validation set is the same
            assert val[i][1].numpy() == round(
                feature.crop_probability
            ), f"{val[i][1].numpy()} != {feature.crop_probability}"
            validation_features.append(feature.__dict__)

    # Initialize a dataframe with all features
    df = pd.DataFrame(validation_features)
    df["y_true"] = df["crop_probability"].apply(lambda prob: 1 if prob > 0.5 else 0)

    # Make predictions on validation set
    x = torch.stack([v[0] for v in val])
    with torch.no_grad():
        # model(x) is indexed to get the local predictions (not global at index 0)
        df["y_pred_decimal"] = model(x)[1].numpy().flatten()

    df["y_pred"] = df["y_pred_decimal"].apply(lambda pred: 1 if pred > default_threshold else 0)
    return df


def plot_precision_recall_graphs(model_name: str):
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

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    best_i = np.argmax(f1_scores)
    threshold = round(thresholds[best_i], 3)
    best_f1 = round(f1_scores[best_i], 3)

    axes[0].plot(recall_scores, precision_scores)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision Recall Curve")
    axes[0].plot(recall_scores[best_i], precision_scores[best_i], "r*", label=f"Best F1: {best_f1}")
    axes[0].legend()

    axes[1].plot(thresholds, f1_scores)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title("F1 Scores")
    axes[1].plot(thresholds[best_i], f1_scores[best_i], "r*")
    axes[1].annotate(
        f"F1: {best_f1}\nThreshold: {threshold}",
        (thresholds[best_i], f1_scores[best_i] - 0.1),
    )
    print(f"Threshold: {thresholds[best_i]}, F1: {best_f1}")
    return best_i, best_f1
