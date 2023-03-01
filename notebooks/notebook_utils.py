from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score


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
