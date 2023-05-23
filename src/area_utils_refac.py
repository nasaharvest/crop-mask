from typing import List, Union

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def gdal_reproject(
        target_crs : str,
        source_crs : str,
        source_fn : str,
        dest_fn : str
        ) -> None:
    cmd = f"gdalwarp -t_srs {target_crs} -s_srs {source_crs} -tr 10 10 {source_fn} {dest_fn} -dstnodata 255"
    os.system(cmd)

def gdal_cutline(
        shape_fn : str,
        source_fn : str,
        dest_fn : str,
    ) -> None:
    cmd = f"gdalwarp -cutline {shape_fn} -crop_to_cutline {source_fn} {dest_fn} -dstnodata 255"
    os.system(cmd)

def compute_confusion_matrix(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> np.ndarray:
    """Computes confusion matrix of reference and map samples.

    Returns confusion matrix in row 'Truth' and column 'Prediction' order.

    """

    y_true = np.array(df["Reference label"]).astype(np.uint8)
    y_pred = np.array(df["Mapped class"]).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred)
    return cm


def compute_area_error_matrix(cm: np.ndarray, w_j: np.ndarray) -> np.ndarray:
    """Computes error matrix in terms of area proportion, p[i,j].

    Args:
        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.
        w_j:
            Array containing the marginal pixel area proportion of each mapped class.

    Returns:
        Error matrix of reference and map samples expressed in terms of area proportion.

    """

    n_dotj = cm.sum(axis=0)
    area_matrix = (w_j * cm) / n_dotj
    return area_matrix


def compute_u_j(am: np.ndarray) -> np.ndarray:
    """Computes the user's accuracy of mapped classes.

    Args:
        am:
            Error matrix of reference and map samples expressed in terms of
            area proportions, p[i,j]. Row-column ordered reference-row, map-column.

    Returns:
        An array containing the user accuracy of each mapped class 'j'.

    """

    p_jjs = np.diag(am)
    p_dotjs = am.sum(axis=0)
    return p_jjs / p_dotjs


def compute_var_u_j(u_j: np.ndarray, cm: np.ndarray) -> np.ndarray:
    """Estimates the variance of user's accuracy of mapped classes.

    Args:
        u_j:
            Array containing the user's accuracy of each mapped class.
        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.

    Returns:
        An array containing the variance of user accuracy for the mapped classes.
        Each entry is the variance of producer accuracy for the corresponding class index.

    """

    n_dotj = cm.sum(axis=0)
    return u_j * (1 - u_j) / (n_dotj - 1)


def compute_p_i(am: np.ndarray) -> np.ndarray:
    """Computes the producer's accuracy of reference classes.

    Args:
        am:
            Error matrix of reference and map samples expressed in terms of
            area proportions, p[i,j]. Row-column ordered reference-row, map-column.

    Returns:
        An array containing the producer's accuracy of each reference class 'i'.

    """

    p_iis = np.diag(am)
    p_idots = am.sum(axis=1)
    return p_iis / p_idots


def compute_var_p_i(
    p_i: np.ndarray, u_j: np.ndarray, a_j: np.ndarray, cm: np.ndarray
) -> np.ndarray:
    """Estimates the variance of producer's accuracy of reference classes.

    For more details, reference Eq. 7 from Olofsson et al., 2014 "Good Practices...".

    Args:
        p_i:
            Array containing the producer's accuracy of each reference class.
        u_j:
            Array containing the user's accuracy of each mapped class.
        a_j:
            Array containing the pixel total of each mapped class.
        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.

    Returns:
        An array containing the variance of producer accuracy for reference classes.
        Each entry is the variance of producer accuracy for the corresponding class index.

    """

    # Estimated marginal total of pixels of reference class
    n_i_px = ((a_j / cm.sum(axis=0) * cm).sum(axis=1)).astype(np.uint64)

    # Marginal total number of pixels of mapped class
    n_j_px = a_j.astype(np.uint64)

    # Total number of sample units of mapped class
    n_j_su = cm.sum(axis=0)

    # Confusion matrix divided by total number of sample units per mapped class
    cm_div = cm / n_j_su
    cm_div_comp = 1 - cm_div

    # We fill the diagonals to '0' because of summation condition that i =/= j
    # in the second expression of equation
    np.fill_diagonal(cm_div, 0.0)
    np.fill_diagonal(cm_div_comp, 0.0)

    sigma = ((n_j_px**2) * (cm_div) * (cm_div_comp) / (n_j_su - 1)).sum(axis=1)
    expr_2 = (p_i**2) * sigma
    expr_1 = (n_j_px**2) * ((1 - p_i) ** 2) * u_j * (1 - u_j) / (n_j_su - 1)

    return (1 / n_i_px**2) * (expr_1 + expr_2)


def compute_acc(am: np.ndarray) -> float:
    """Computes the overall accuracy.

    Args:
        am:
            Error matrix of reference and map samples expressed in terms of
            area proportions, p[i,j]. Row-column ordered reference-row, map-column.

    """

    acc = np.diag(am).sum()
    return acc


def compute_var_acc(w_j: np.ndarray, u_j: np.ndarray, cm: np.ndarray) -> float:
    """Estimates the variance of overall accuracy.

    Args:
        w_j:
            Array containing the marginal area proportion of each mapped class.
        u_j:
            Array containing the user's accuracy of each mapped class.
        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.

    """

    sigma = (w_j**2) * (u_j) * (1 - u_j) / (cm.sum(axis=0) - 1)
    return sigma.sum()


def compute_std_p_i(w_j: np.ndarray, am: np.ndarray, cm: np.ndarray) -> np.ndarray:
    """Estimates the standard error of area estimator, p_{i.}.

    Args:
        w_j:
            Array containing the area proportion of each mapped class.
        am:
            Confusion matrix of reference and map samples expressed in terms of
            area proportion, p[i,j]. Row-column ordered reference-row, map-column.

        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.

    Returns:
        An array containing the estimatated standard deviation of area estimator.
        Each entry is the stdDev of estimated area for the corresponding class index.

    """

    sigma = (w_j * am - am**2) / (cm.sum(axis=0) - 1)
    return np.sqrt(sigma.sum(axis=1))


def compute_area_estimate(cm: np.ndarray, a_j: np.ndarray, px_size: float) -> dict:
    """Computes area estimate from confusion matrix, pixel total, and area totals.

    Args:
        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.
        a_j:
            Array containing the pixel total of each mapped class.
        px_size:
            Spatial resolution of pixels in map (unsquared).

    Returns:
        summary:
            Dictionary with estimates of user's accuracy, producer's accuracy, area
            estimates, and the 95% confidence interval of each for every class of
            the confusion matrix.

            Value of area estimate key in 'summary' is nested dictionary containing
            the estimates and interval for area in proportion [pr], pixels [px], and
            hectacres [ha].

    """

    total_px = a_j.sum()

    w_j = a_j / total_px

    am = compute_area_error_matrix(cm, w_j)

    # User's accuracy
    u_j = compute_u_j(am)
    var_u_j = compute_var_u_j(u_j, cm)
    err_u_j = 1.96 * np.sqrt(var_u_j)

    # Producer's accuracy
    p_i = compute_p_i(am)
    var_p_i = compute_var_p_i(p_i, u_j, a_j, cm)
    err_p_i = 1.96 * np.sqrt(var_p_i)

    # Overall accuracy
    acc = compute_acc(am)
    var_acc = compute_var_acc(w_j, u_j, cm)
    err_acc = 1.96 * np.sqrt(var_acc)

    # Area estimate
    a_i = am.sum(axis=1)
    std_a_i = compute_std_p_i(w_j, am, cm)
    err_a_i = 1.96 * std_a_i

    # Adjusted marginal area estimate in [px] and [ha]
    a_px = total_px * a_i
    a_ha = a_px * (px_size**2) / (100**2)
    err_px = err_a_i * total_px
    err_ha = err_px * (px_size**2) / (100**2)

    summary = {
        "user": (u_j, err_u_j),
        "producer": (p_i, err_p_i),
        "accuracy": (acc, err_acc),
        "area": {"pr": (a_i, err_a_i), "px": (a_px, err_px), "ha": (a_ha, err_ha)},
    }

    return summary


def create_area_estimate_summary(
    a_ha: np.ndarray,
    err_ha: np.ndarray,
    u_j: np.ndarray,
    err_u_j: np.ndarray,
    p_i: np.ndarray,
    err_p_i: np.ndarray,
    columns: Union[List[str], np.ndarray],
) -> pd.DataFrame:
    """Generates summary table of area estimation and statistics.

    Args:
        a_ha:
            Area estimate of each class, in units of hectacres.
        err_ha:
            95% confidence interval of each class area estimate, in hectacres.
        u_j:
            User's accuray of each mapped class, "j".
        err_u_j:
            95% confidence interval of user's accuracy for each mapped class, "j".
        p_i:
            Producer's accuracy of each reference class, "i".
        err_p_i:
            95% confidence interval of producer's accuracy fo each reference class, "i".
        columns:
            List-like containing labels in same order as confusion matrix. For
            example:

            ["Stable NP", "PGain", "PLoss", "Stable P"]

            ["Non-Crop", "Crop"]

    Returns:
        summary:
            Table with estimates of area [ha], user's accuracy, producer's accuracy, and 95%
            confidence interval of each for every class.

    """

    summary = pd.DataFrame(
        data=[
            a_ha,
            err_ha,
            u_j,
            err_u_j,
            p_i,
            err_p_i,
        ],
        index=pd.Index(
            [
                "Estimated area [ha]",
                "95% CI of area [ha]",
                "User's accuracy",
                "95% CI of user acc.",
                "Producer's accuracy",
                "95% CI of prod acc.",
            ]
        ),
        columns=columns,
    )

    print(summary.round(2))
    return summary


def create_confusion_matrix_summary(
    cm: np.ndarray, columns: Union[List[str], np.ndarray]
) -> pd.DataFrame:
    """Generates summary table of confusion matrix.

    Computes and displays false positive rate (FPR), true positive rate (TPR), and accuracies.

    Args:
        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.
        columns:
            List-like containing labels in same order as confusion matrix. For
            example:

            ["Stable NP", "PGain", "PLoss", "Stable P"]

            ["Non-Crop", "Crop"]

    Returns:
        summary:
            Table with FPR, TPR, and accuracies for each class of confusion matrix.

    """

    fp = cm.sum(axis=0) - np.diag(cm)  # Column-wise (Prediction)
    fn = cm.sum(axis=1) - np.diag(cm)  # Row-wise (Truth)
    tp = np.diag(cm)  # Diagonals (Prediction and Truth)
    tn = cm.sum() - (fp + fn + tp)

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    acc = (tp + fn) / (tp + tn + fp + fn)
    summary = pd.DataFrame(
        data=[fpr, tpr, acc],
        index=["False Positive Rate", "True Positive Rate", "Accuracy"],
        columns=columns,
    )

    print(summary.round(2))
    return summary


def plot_confusion_matrix(cm: np.ndarray, labels: Union[List[str], np.ndarray]) -> None:
    """Pretty prints confusion matrix.

    Expects row 'Reference' and column 'Prediction/Map' ordered confusion matrix.

    Args:
        cm:
            Confusion matrix of reference and map samples expressed in terms of
            sample counts, n[i,j]. Row-column ordered reference-row, map-column.
        labels:
            List-like containing labels in same order as confusion matrix. For
            example:

            ["Stable NP", "PGain", "PLoss", "Stable P"]

            ["Non-Crop", "Crop"]

    """

    _, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(cm, cmap="crest", annot=True, fmt="d", cbar=False, square=True, ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_coords(0.50, 1.125)
    ax.yaxis.set_label_coords(-0.125, 0.50)
    ax.set_xticklabels(labels=labels)
    ax.set_yticklabels(labels=labels)
    ax.set_xlabel("Map", fontsize=12)
    ax.set_ylabel("Reference", fontsize=12)
    plt.tight_layout()
