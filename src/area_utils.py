import json
import os
from typing import List, Optional, Tuple, Union

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import gdal
from rasterio import transform
from rasterio.mask import mask
from shapely.geometry import box
from sklearn.metrics import confusion_matrix


def gdal_reproject(target_crs: str, source_crs: str, source_fn: str, dest_fn: str) -> None:
    cmd = (
        f"gdalwarp -t_srs {target_crs} -s_srs {source_crs} -tr 10 10 "
        f"{source_fn} {dest_fn} -dstnodata 255"
    )
    os.system(cmd)


def gdal_cutline(
    shape_fn: str,
    source_fn: str,
    dest_fn: str,
) -> None:
    cmd = f"gdalwarp -cutline {shape_fn} -crop_to_cutline " f"{source_fn} {dest_fn} -dstnodata 255"
    os.system(cmd)


def load_ne(country_code: str, regions_of_interest: List[str]) -> gpd.GeoDataFrame:
    """
    Load the Natural Earth country and region shapefiles.
    country_code: ISO 3166-1 alpha-3 country code
    regions_of_interest: list of regions of interest
    """

    ne_shapefile = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_1_states_provinces"
    )
    ne_gdf = gpd.read_file(ne_shapefile)

    if len(regions_of_interest) == 0:
        condition = ne_gdf["adm1_code"].str.startswith(country_code)
        boundary = ne_gdf[condition].copy()
        print("Entire country found!")
        boundary = boundary.dissolve(by="admin")
        return boundary

    else:
        available_regions = ne_gdf[ne_gdf["adm1_code"].str.startswith(country_code)][
            "name"
        ].tolist()
        regions_not_found = [
            region for region in regions_of_interest if region not in available_regions
        ]

        if len(regions_not_found) > 0:
            condition = ne_gdf["adm1_code"].str.startswith(country_code)
            boundary = None
            print(
                f"WARNING: {regions_not_found} was not found. Please select \
                regions only seen in below plot."
            )
            ne_gdf[condition].plot(
                column="name",
                legend=True,
                legend_kwds={"loc": "lower right"},
                figsize=(10, 10),
            )
        else:
            available_ = ne_gdf[ne_gdf["adm1_code"].str.startswith(country_code)]
            condition = available_["name"].isin(regions_of_interest)
            boundary = available_[condition].copy()
            boundary = boundary.dissolve(by="admin")
        return boundary


def clip_raster(
    in_raster: str, boundary: Optional[gpd.GeoDataFrame] = None
) -> Tuple[Optional[np.ma.core.MaskedArray], dict]:
    """Clip the raster to the boundary
    in_raster: path to the input raster
    boundary: GeoDataFrame of the boundary
    """
    with rio.open(in_raster) as src:
        if boundary is None:
            print("No boundary provided. Clipping to map bounds.")
            boundary = gpd.GeoDataFrame(geometry=[box(*src.bounds)], crs=src.crs)

        else:
            print("Clipping to boundary.")
        boundary = boundary.to_crs(src.crs)
        boundary = [json.loads(boundary.to_json())["features"][0]["geometry"]]
        raster, out_transform = mask(
            src, shapes=boundary, crop=True, all_touched=True, nodata=src.nodata
        )
        raster = raster[0]
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": raster.shape[0],
                "width": raster.shape[1],
                "transform": out_transform,
                "crs": src.meta["crs"],
            }
        )
        print("The pixel size is {:.3f} meters".format(out_meta["transform"][0]))
        return np.ma.masked_equal(raster, src.meta["nodata"]), out_meta


def load_raster(
    in_raster: str, boundary: Optional[gpd.GeoDataFrame] = None
) -> Tuple[Optional[np.ma.core.MaskedArray], dict]:
    """
    Check if the raster is projected in the correct CRS.
    If not, reproject it.
    Clip the raster to the boundary. If no boundary is provided, clip to map bounds.
    in_raster: path to the input raster
    boundary: GeoDataFrame of the boundary
    """
    in_raster_basename = os.path.basename(in_raster)

    with rio.open(in_raster) as src:
        if src.meta["crs"] == "EPSG:4326" or not src.meta["crs"].startswith("EPSG:326"):
            print(
                """WARNING: The map CRS is EPSG:4326. This means the map unit is degrees \
                and the pixel-wise areas will not be in meters.
                \n You need to project the map to the local UTM Zone \
                (EPSG:XXXXX)."""
            )
            t_srs = input("Input EPSG Code; EPSG:XXXX:")
            options = {"dstSRS": f"EPSG:{t_srs}", "dstNodata": 255}
            gdal.Warp(f"prj_{in_raster_basename}", in_raster_basename, **options)
            in_raster = f"prj_{in_raster_basename}"
            return clip_raster(in_raster, boundary)
        else:
            print("Map CRS is %s. Loading map into memory." % src.crs)
            return clip_raster(in_raster, boundary)


def binarize(
    raster: np.ma.core.MaskedArray, meta: dict, threshold: Optional[float] = 0.5
) -> np.ma.core.MaskedArray:
    raster.data[raster.data < threshold] = 0
    raster.data[((raster.data >= threshold) & (raster.data != meta["nodata"]))] = 1
    return raster.astype(np.uint8)


def cal_map_area_class(
    binary_map: np.ndarray, unit: str = "pixels", px_size: float = 10
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate the area of each class in the map.
    Print the area when the unit is specified to be in pixel and ha.
    In case of fraction, the area printed and returned to be assigned to a variable.
    binary_map: numpy array of the map
    """
    crop_px = np.where(binary_map.flatten() == 1)
    noncrop_px = np.where(binary_map.flatten() == 0)
    total = crop_px[0].shape[0] + noncrop_px[0].shape[0]
    if unit == "ha":
        crop_area = crop_px[0].shape[0] * (px_size * px_size) / 10000
        noncrop_area = noncrop_px[0].shape[0] * (px_size * px_size) / 10000
        print(
            f"Crop area: {crop_area:.2f} ha, Non-crop area: {noncrop_area:.2f} ha \n \
             Total area: {crop_area + noncrop_area:.2f} ha"
        )

    elif unit == "pixels":
        crop_area = int(crop_px[0].shape[0])
        noncrop_area = int(noncrop_px[0].shape[0])
        print(
            f"Crop pixels count: {crop_area}, Non-crop pixels count: {noncrop_area} pixels \n \
            Total counts: {crop_area + noncrop_area} pixels"
        )

    elif unit == "fraction":
        crop_area = int(crop_px[0].shape[0]) / total
        noncrop_area = int(noncrop_px[0].shape[0]) / total
        print(f"Crop area: {crop_area:.2f} fraction, Non-crop area: {noncrop_area:.2f} fraction")
        assert crop_area + noncrop_area == 1

    else:
        print("Please specify the unit as either 'pixels', 'ha', or 'fraction'")
    return crop_area, noncrop_area


def estimate_num_sample_per_class(
    crop_area_fraction: float,
    non_crop_area_fraction: float,
    u_crop: float,
    u_noncrop: float,
    stderr: float = 0.02,
) -> Tuple[float, float]:
    s_crop = np.sqrt(u_crop * (1 - u_crop))
    s_noncrop = np.sqrt(u_noncrop * (1 - u_crop))

    n = np.round(((crop_area_fraction * s_crop + non_crop_area_fraction * s_noncrop) / stderr) ** 2)
    print(f"Num of sample size: {n}")

    n_crop = int(n / 2)
    n_noncrop = int(n - n_crop)

    print(f"Num sample size for crop: {n_crop}")
    print(f"Num sample size for non-crop: {n_noncrop}")
    return n_crop, n_noncrop


def random_inds(
    binary_map: np.ndarray, strata: int, sample_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random indices for sampling from a binary map."""
    inds = np.where(binary_map == strata)
    rand_inds = np.random.permutation(np.arange(inds[0].shape[0]))[:sample_size]
    rand_px = inds[0][rand_inds]
    rand_py = inds[1][rand_inds]

    return rand_px, rand_py


def generate_ref_samples(binary_map: np.ndarray, meta: dict, n_crop: int, n_noncrop: int) -> None:
    df_noncrop = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_noncrop["px"], df_noncrop["py"] = random_inds(binary_map, 0, int(n_noncrop))
    df_noncrop["pred_class"] = 0

    df_crop = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_crop["px"], df_crop["py"] = random_inds(binary_map, 1, int(n_crop))
    df_crop["pred_class"] = 1

    df_combined = pd.concat([df_crop, df_noncrop]).reset_index(drop=True)

    for r, row in df_combined.iterrows():
        lx, ly = transform.xy(meta["transform"], row["px"], row["py"])
        df_combined.loc[r, "lx"] = lx
        df_combined.loc[r, "ly"] = ly

    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    gdf = gpd.GeoDataFrame(df_combined, geometry=gpd.points_from_xy(df_combined.lx, df_combined.ly))

    gdf.crs = meta["crs"]
    gdf_ceo = gdf.to_crs("EPSG:4326")
    gdf_ceo["PLOTID"] = gdf_ceo.index
    gdf_ceo["SAMPLEID"] = gdf_ceo.index

    gdf_ceo[["geometry", "PLOTID", "SAMPLEID"]].to_file("ceo_reference_sample.shp", index=False)


def reference_sample_agree(
    binary_map: np.ndarray, meta: dict, ceo_ref1: str, ceo_ref2: str
) -> gpd.GeoDataFrame:
    ceo_set1 = pd.read_csv(ceo_ref1)
    ceo_set2 = pd.read_csv(ceo_ref2)

    assert ceo_set1.columns[-1] == ceo_set2.columns[-1]

    label_question = ceo_set1.columns[-1]

    print(f"Number of NANs/ missing answers in set 1: {ceo_set1[label_question].isna().sum()}")
    print(f"Number of NANs/ missing answers in set 2: {ceo_set2[label_question].isna().sum()}")

    if ceo_set1.shape[0] != ceo_set2.shape[0]:
        print("The number of rows in the reference sets are not equal.")
        print("Checking for duplictes on 'plotid'..")
        print(
            " Number of duplicated in set 1: %s" % ceo_set1[ceo_set1.plotid.duplicated()].shape[0]
        )
        print(
            " Number of duplicated in set 2: %s" % ceo_set2[ceo_set2.plotid.duplicated()].shape[0]
        )
        print("Removing duplicates and keeping the first...")
        ceo_set1 = ceo_set1.drop_duplicates(subset="plotid", keep="first")
        ceo_set2 = ceo_set2.drop_duplicates(subset="plotid", keep="first")

        ceo_set1.set_index("plotid", inplace=True)
        ceo_set2.set_index("plotid", inplace=True)
    else:
        print("The number of rows in the reference sets are equal.")

    ceo_agree = ceo_set1[ceo_set1[label_question] == ceo_set2[label_question]]

    print(
        "Number of samples that are in agreement: %d out of %d (%.2f%%)"
        % (
            ceo_agree.shape[0],
            ceo_set1.shape[0],
            ceo_agree.shape[0] / ceo_set1.shape[0] * 100,
        )
    )
    ceo_agree_geom = gpd.GeoDataFrame(
        ceo_agree,
        geometry=gpd.points_from_xy(ceo_agree.lon, ceo_agree.lat),
        crs="EPSG:4326",
    )

    ceo_agree_geom = ceo_agree_geom.to_crs(meta["crs"])

    label_responses = ceo_agree_geom[label_question].unique()
    assert len(label_responses) == 2

    for r, row in ceo_agree_geom.iterrows():
        lon, lat = row["geometry"].y, row["geometry"].x
        px, py = transform.rowcol(meta["transform"], lat, lon)

        try:
            ceo_agree_geom.loc[r, "Mapped class"] = int(binary_map[px, py])
            if (
                row[label_question].lower() == "cropland"
                or row[label_question].lower() == "crop"
                or row[label_question].lower() == "planted"
            ):
                ceo_agree_geom.loc[r, "Reference label"] = 1
            elif (
                row[label_question].lower() == "non-cropland"
                or row[label_question].lower() == "non-crop"
                or row[label_question].lower() == "not planted"
            ):
                ceo_agree_geom.loc[r, "Reference label"] = 0
        except IndexError:
            ceo_agree_geom.loc[r, "Mapped class"] = 255
            ceo_agree_geom.loc[r, "Reference label"] = 0

    return ceo_agree_geom


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
    # if no predictions for class j, n_dotj will be 0
    area_matrix[np.where(np.isnan(area_matrix))] = 0
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
    u_j = p_jjs / p_dotjs
    # if no predictions for class j, p_dotj will be 0
    u_j[np.where(np.isnan(u_j))] = 0
    return u_j


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
    # if no predictions for class j, n_j_su will be 0
    cm_div[np.where(np.isnan(cm_div))] = 0
    cm_div_comp = 1 - cm_div

    # We fill the diagonals to '0' because of summation condition that i =/= j
    # in the second expression of equation
    np.fill_diagonal(cm_div, 0.0)
    np.fill_diagonal(cm_div_comp, 0.0)

    sigma = ((n_j_px**2) * (cm_div) * (cm_div_comp) / (n_j_su - 1)).sum(axis=1)
    expr_2 = (p_i**2) * sigma
    expr_1 = (n_j_px**2) * ((1 - p_i) ** 2) * u_j * (1 - u_j) / (n_j_su - 1)
    expr_3 = 1 / n_i_px**2
    # convert inf to 0 (can result from divide by 0)
    expr_3[np.where(np.isinf(expr_3))] = 0

    return expr_3 * (expr_1 + expr_2)


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


def plot_area(summary: pd.DataFrame) -> None:
    area_class = summary.columns
    x_pos = np.arange(len(area_class))
    est_area = summary.loc["Estimated area [ha]"]
    ci_area = summary.loc["95% CI of area [ha]"]

    fig, ax = plt.subplots()
    ax.bar(
        x_pos,
        est_area,
        yerr=ci_area,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.set_ylabel("Estimated Area [ha]")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(area_class)
    ax.set_title("Area estimation with standard error at 95% confidence interval")
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.show()
