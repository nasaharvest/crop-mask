import json
import os
from typing import List, Optional, Tuple

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import transform
from rasterio.mask import mask
from shapely.geometry import box
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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
        raster, out_transform = mask(src, shapes=boundary, crop=True, all_touched=True, nodata=src.nodata)
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
    Chcked if the raster is projected in the correct CRS.
    If not, reproject it.
    Clip the raster to the boundary. If no boundary is provided, clip to map bounds.
    in_raster: path to the input raster
    boundary: GeoDataFrame of the boundary
    """
    in_raster_basename = os.path.basename(in_raster)

    with rio.open(in_raster) as src:
        if src.meta["crs"] == "EPSG:4326":
            print(
                """WARNING: The map CRS is EPSG:4326. This means the map unit is degrees \
                and the pixel-wise areas will not be in meters.
                \n You need to project the  project the map to using the local UTM Zone \
                (EPSG:XXXXX)."""
            )
            t_srs = input("Input EPSG Code; EPSG:XXXX:")
            cmd = f"gdalwarp -t_srs EPSG:{t_srs} {in_raster} -overwrite \
                prj_{in_raster_basename} -dstnodata 255"
            print(cmd)
            print("Reprojecting the raster...")
            os.system(cmd)
            in_raster = f"prj_{in_raster_basename}"
            return clip_raster(in_raster, boundary)
        else:
            print("Map CRS is %s. Loading map into memory." % src.crs)
            return clip_raster(in_raster, boundary)


def binarize(raster: np.ma.core.MaskedArray, meta: dict, threshold: float = 0.5) -> np.ndarray:
    raster[raster < threshold] = 0
    raster[((raster >= threshold) & (raster != meta["nodata"]))] = 1
    return raster.data.astype(np.uint8)


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
        crop_area = crop_px[0].shape[0] * (px_size * px_size) / 100000
        noncrop_area = noncrop_px[0].shape[0] * (px_size * px_size) / 100000
        print(
            f"Crop area: {crop_area:.2f} ha, Non-crop area: {noncrop_area:.2f} ha \n \
             Total area: {crop_area + noncrop_area:.2f} ha"
        )

    elif unit == "pixels":
        crop_area = int(crop_px[0].shape[0])
        noncrop_area = int(noncrop_px[0].shape[0])
        print(
            f"Crop area: {crop_area} pixels, Non-crop area: {noncrop_area} pixels \n \
            Total area: {crop_area + noncrop_area} pixels"
        )

    elif unit == "fraction":
        crop_area = int(crop_px[0].shape[0]) / total
        noncrop_area = int(noncrop_px[0].shape[0]) / total
        print(f"Crop area: {crop_area:.2f} fraction, Non-crop area: {noncrop_area:.2f} fraction")
        assert crop_area + noncrop_area == 1

    else:
        print("Please specify the unit as either 'pixels', 'ha', or 'fraction'")
    return crop_area, noncrop_area


def cal_map_area_change_class(
    change_map: np.ndarray, unit: str = "pixels", px_size: float = 10
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate the area of each class in the map.
    Print the area of each class and total area.
    change_map: numpy array of the map
    """
    stable_np = np.where(change_map.flatten() == 0)[0]
    p_gain = np.where(change_map.flatten() == 1)[0]
    p_loss = np.where(change_map.flatten() == 2)[0]
    stable_p = np.where(change_map.flatten() == 3)[0]
    total = change_map.flatten().shape[0]
    # Make sure nodata values are not being counted
    assert total == np.sum([stable_np.shape[0], p_gain.shape[0], p_loss.shape[0], stable_p.shape[0]])

    if unit == "ha":
        stable_np_area = stable_np.shape[0] * (px_size*px_size) / 100000
        p_gain_area = p_gain.shape[0] * (px_size*px_size) / 100000
        p_loss_area = p_loss.shape[0] * (px_size*px_size) / 100000
        stable_p_area = stable_p.shape[0] * (px_size*px_size) / 100000

    elif unit == "pixels":
        stable_np_area = int(stable_np.shape[0])
        p_gain_area = int(p_gain.shape[0])
        p_loss_area = int(p_loss.shape[0])
        stable_p_area = int(stable_p.shape[0])

    elif unit == "fraction":
        stable_np_area = int(stable_np.shape[0]) / total
        p_gain_area = int(p_gain.shape[0]) / total
        p_loss_area = int(p_loss.shape[0]) / total
        stable_p_area = int(stable_p.shape[0]) / total
        assert sum([stable_np_area, p_gain_area, p_loss_area, stable_p_area]) == 1

    else:
        print("Please specify the unit as either 'pixels', 'ha', or 'fraction'")

    print(
            f'Stable NP area: {stable_np_area} {unit} \n'
            f'P gain area: {p_gain_area} {unit} \n'
            f'P loss area: {p_loss_area} {unit} \n'
            f'Stable P area: {stable_p_area} {unit} \n'
            f'Total area: '
            f'{stable_np_area + p_gain_area + p_loss_area + stable_p_area:.2f}'
            f' {unit} \n'
    )

    return stable_np_area, p_gain_area, p_loss_area, stable_p_area


def estimate_num_sample_per_change_class(
    stable_np_fraction: float,
    p_gain_fraction: float,
    p_loss_fraction: float,
    stable_p_fraction: float,
    u_stable_np: float,
    u_p_gain: float,
    u_p_loss: float,
    u_stable_p: float,
    stderr: float = 0.02,
) -> Tuple[int, int, int, int]:

    s_stable_np = np.sqrt(u_stable_np * (1 - u_stable_np))
    s_p_gain = np.sqrt(u_p_gain * (1 - u_p_gain))
    s_p_loss = np.sqrt(u_p_loss * (1 - u_p_loss))
    s_stable_p = np.sqrt(u_stable_p * (1 - u_stable_p))

    n = np.round(((
                    stable_np_fraction * s_stable_np +
                    p_gain_fraction * s_p_gain +
                    p_loss_fraction * s_p_loss +
                    stable_p_fraction * s_stable_p
                  ) / stderr) ** 2)
    print(f"Num total sample size: {int(n)}")

    n_stable_np = int(n / 4)
    n_p_gain = int(n / 4)
    n_p_loss = int((n / 4) + (n % 4))
    n_stable_p = int(n / 4)

    print(f"Num sample size for stable NP: {n_stable_np}")
    print(f"Num sample size for P gain: {n_p_gain}")
    print(f"Num sample size for P loss: {n_p_loss}")
    print(f"Num sample size for stable P: {n_stable_p}")
    return n_stable_np, n_p_gain, n_p_loss, n_stable_p


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
    # """ Generate random indices for sampling from a binary map."""
    inds = np.where(binary_map == strata)
    rand_inds = np.random.permutation(np.arange(inds[0].shape[0]))[:sample_size]
    rand_px = inds[0][rand_inds]
    rand_py = inds[1][rand_inds]
    # del inds
    return rand_px, rand_py


def generate_change_ref_samples(
    change_map: np.ndarray,
    meta: dict,
    n_stablenp: int,
    n_pgain: int,
    n_ploss: int,
    n_stablep: int
) -> None:

    df_stablenp = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_stablenp["px"], df_stablenp["py"] = random_inds(change_map, 0, int(n_stablenp))
    df_stablenp["pred_class"] = 0

    df_pgain = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_pgain["px"], df_pgain["py"] = random_inds(change_map, 1, int(n_pgain))
    df_pgain["pred_class"] = 0

    df_ploss = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_ploss["px"], df_ploss["py"] = random_inds(change_map, 2, int(n_ploss))
    df_ploss["pred_class"] = 0

    df_stablep = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_stablep["px"], df_stablep["py"] = random_inds(change_map, 0, int(n_stablep))
    df_stablep["pred_class"] = 3

    df_combined = pd.concat([df_stablenp,
                             df_pgain,
                             df_ploss,
                             df_stablep]).reset_index(drop=True)

    for r, row in df_combined.iterrows():
        lx, ly = transform.xy(meta["transform"], row["px"], row["py"])
        df_combined.loc[r, "lx"] = lx
        df_combined.loc[r, "ly"] = ly

    # Shuffle the samples so they don't appear in order of class
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    gdf = gpd.GeoDataFrame(df_combined, geometry=gpd.points_from_xy(df_combined.lx, df_combined.ly))

    gdf.crs = meta["crs"]
    gdf_ceo = gdf.to_crs("EPSG:4326")
    gdf_ceo["PLOTID"] = gdf_ceo.index
    gdf_ceo["SAMPLEID"] = gdf_ceo.index

    gdf_ceo[["geometry", "PLOTID", "SAMPLEID"]].to_file("ceo_change_reference_sample.shp", index=False)


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

        ceo_agree_geom.loc[r, "Mapped class"] = int(binary_map[px, py])

        if label_responses[1].startswith("C") or label_responses[1].startswith("c"):
            ceo_agree_geom.loc[
                ceo_agree_geom[label_question] == label_responses[1], "Reference label"
            ] = 1
            ceo_agree_geom.loc[
                ceo_agree_geom[label_question] == label_responses[0], "Reference label"
            ] = 0

        ceo_agree_geom["Reference label"] = ceo_agree_geom["Reference label"].astype(np.uint8)

    return ceo_agree_geom


def change_reference_sample_agree(
    change_map: np.ndarray, meta: dict, ceo_ref1: str, ceo_ref2: str
) -> gpd.GeoDataFrame:

    ceo_set1 = pd.read_csv(ceo_ref1)
    ceo_set2 = pd.read_csv(ceo_ref2)

    assert ceo_set1.columns[-1] == ceo_set2.columns[-1]

    label_question_y1 = ceo_set1.columns[-2]
    label_question_y2 = ceo_set1.columns[-1]

    print(f"Number of NaNs/missing answers in set 1 q1: {ceo_set1[label_question_y1].isna().sum()}")
    print(f"Number of NaNs/missing answers in set 2 q1: {ceo_set2[label_question_y1].isna().sum()}")

    print(f"Number of NaNs/missing answers in set 1 q2: {ceo_set1[label_question_y2].isna().sum()}")
    print(f"Number of NaNs/missing answers in set 2 q2: {ceo_set2[label_question_y2].isna().sum()}")

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

    # Convert the individual year classifications to the change classes
    ceo_set1['Reference label'] = None
    ceo_set2['Reference label'] = None

    for r, row in ceo_set1.iterrows():
        label_y1 = row[label_question_y1]
        label_y2 = row[label_question_y2]
        if label_y1 == 'Not planted' and label_y2 == 'Not planted':
            ceo_set1.loc[r,'Reference label'] = 0
        elif label_y1 == 'Not planted' and label_y2 == 'Planted':
            ceo_set1.loc[r,'Reference label'] = 1
        elif label_y1 == 'Planted' and label_y2 == 'Not planted':
            ceo_set1.loc[r,'Reference label'] = 2
        elif label_y1 == 'Planted' and label_y2 == 'Planted':
            ceo_set1.loc[r,'Reference label'] = 3

    for r, row in ceo_set2.iterrows():
        label_y1 = row[label_question_y1]
        label_y2 = row[label_question_y2]
        if label_y1 == 'Not planted' and label_y2 == 'Not planted':
            ceo_set2.loc[r,'Reference label'] = 0
        elif label_y1 == 'Not planted' and label_y2 == 'Planted':
            ceo_set2.loc[r,'Reference label'] = 1
        elif label_y1 == 'Planted' and label_y2 == 'Not planted':
            ceo_set2.loc[r,'Reference label'] = 2
        elif label_y1 == 'Planted' and label_y2 == 'Planted':
            ceo_set2.loc[r,'Reference label'] = 3

    ceo_agree = ceo_set1[ceo_set1['Reference label'] == ceo_set2['Reference label']]
    ceo_disagree = ceo_set1[ceo_set1['Reference label'] != ceo_set2['Reference label']]

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

    ceo_disagree_geom = gpd.GeoDataFrame(
        ceo_disagree,
        geometry=gpd.points_from_xy(ceo_disagree.lon, ceo_disagree.lat),
        crs="EPSG:4326",
    )

    ceo_disagree_geom = ceo_disagree_geom.to_crs(meta["crs"])

    for r, row in ceo_agree_geom.iterrows():
        lon, lat = row["geometry"].y, row["geometry"].x
        px, py = transform.rowcol(meta["transform"], lat, lon)

        ceo_agree_geom.loc[r, "Mapped class"] = int(change_map[px, py])

        ceo_agree_geom["Reference label"] = ceo_agree_geom["Reference label"].astype(np.uint8)

    for r, row in ceo_disagree_geom.iterrows():
        lon, lat = row["geometry"].y, row["geometry"].x
        px, py = transform.rowcol(meta["transform"], lat, lon)

        ceo_disagree_geom.loc[r, "Mapped class"] = int(change_map[px, py])

        ceo_disagree_geom["Reference label"] = ceo_disagree_geom["Reference label"].astype(np.uint8)

    return ceo_agree_geom, ceo_disagree_geom


def compute_confusion_matrix(ceo_agree_geom: gpd.GeoDataFrame) -> np.ndarray:
    """ """
    y_true = np.array(ceo_agree_geom["Reference label"]).astype(np.uint8)
    y_pred = np.array(ceo_agree_geom["Mapped class"]).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred).ravel()
    if np.max(y_true) == 1:
        print("Error matrix \n")
        print("True negatives: %d" % cm[0])
        print("False positives: %d" % cm[1])
        print("False negatives: %d" % cm[2])
        print("True positives: %d" % cm[3])
    else:
        print(confusion_matrix(y_true, y_pred))
    print('Classification report:')
    print(classification_report(y_true, y_pred))
    return cm


def compute_area_estimate(
    crop_area_px: int, noncrop_area_px: int, cm: np.ndarray, meta: dict
) -> pd.DataFrame:
    tn, fp, fn, tp = cm

    total_area_px = crop_area_px + noncrop_area_px

    wh_crop = crop_area_px / total_area_px
    wh_noncrop = noncrop_area_px / total_area_px
    print("Proportion of mapped area for each class")
    print("Crop: %.2f" % wh_crop)
    print("Non-crop: %.2f \n" % wh_noncrop)

    tp_area = tp / (tp + fp) * wh_crop
    fp_area = fp / (tp + fp) * wh_crop
    fn_area = fn / (fn + tn) * wh_noncrop
    tn_area = tn / (fn + tn) * wh_noncrop
    print("Fraction of the proportional area of each class")
    print(
        "TP crop: %f \t FP crop: %f \n FN noncrop: %f \t TN noncrop: %f \n"
        % (tp_area, fp_area, fn_area, tn_area)
    )

    u_crop = tp_area / (tp_area + fp_area)
    print("User's accuracy")
    print("U_crop = %f" % u_crop)

    u_noncrop = tn_area / (tn_area + fn_area)
    print("U_noncrop = %f \n" % u_noncrop)

    v_u_crop = u_crop * (1 - u_crop) / (tp + fp)
    print("Estimated variance of user accuracy for each mapped class")
    print("V(U)_crop = %f" % v_u_crop)

    v_u_noncrop = u_noncrop * (1 - u_noncrop) / (fn + tn)
    print("V(U)_noncrop = %f \n" % v_u_noncrop)

    s_u_crop = np.sqrt(v_u_crop)
    print("Estimated standard error of user accuracy for each mapped class")
    print("S(U)_crop = %f" % s_u_crop)

    s_u_noncrop = np.sqrt(v_u_noncrop)
    print("S(U)_noncrop = %f \n" % s_u_noncrop)

    u_crop_err = s_u_crop * 1.96
    print("95% confidence interval for User's accuracy")
    print("95%% CI of User accuracy for crop = %f" % u_crop_err)

    u_noncrop_err = s_u_noncrop * 1.96
    print("95%% CI of User accuracy for noncrop = %f \n" % u_noncrop_err)

    p_crop = tp_area / (tp_area + fn_area)
    print("Producer's accuracy")
    print("P_crop = %f" % p_crop)

    p_noncrop = tn_area / (tn_area + fp_area)
    print("P_noncrop = %f \n" % p_noncrop)

    n_j_crop = (crop_area_px * tp) / (tp + fp) + (noncrop_area_px * fn) / (fn + tn)
    print("Estimated marginal total number of pixels of each reference class")
    print("N_j_crop = %f" % n_j_crop)

    n_j_noncrop = (crop_area_px * fp) / (tp + fp) + (noncrop_area_px * tn) / (fn + tn)
    print("N_j_crop = %f \n" % n_j_noncrop)

    expr1_crop = crop_area_px**2 * (1 - p_crop) ** 2 * u_crop * (1 - u_crop) / (tp + fp - 1)
    print("expr1 crop = %f" % expr1_crop)

    expr1_noncrop = (
        noncrop_area_px**2 * (1 - p_noncrop) ** 2 * u_noncrop * (1 - u_noncrop) / (fp + tn - 1)
    )
    print("expr1 noncrop = %f \n" % expr1_noncrop)

    expr2_crop = p_crop**2 * (
        noncrop_area_px**2 * fn / (fn + tn) * (1 - fn / (fn + tn)) / (fn + tn - 1)
    )
    print("expr2 crop = %f" % expr2_crop)

    expr2_noncrop = p_crop**2 * (
        crop_area_px**2 * fp / (fp + tp) * (1 - fp / (fp + tp)) / (fp + tp - 1)
    )
    print("expr2 noncrop = %f \n" % expr2_noncrop)

    v_p_crop = (1 / n_j_crop**2) * (expr1_crop + expr2_crop)
    print("Variance of producer's accuracy for each mapped class")
    print("V(P) crop = %f" % v_p_crop)

    v_p_noncrop = (1 / n_j_noncrop**2) * (expr1_noncrop + expr2_noncrop)
    print("V(P) noncrop = %f \n" % v_p_noncrop)

    s_p_crop = np.sqrt(v_p_crop)
    print("Estimated standard error of producer accuracy for each mapped class")
    print("S(P) crop = %f" % s_p_crop)

    s_p_noncrop = np.sqrt(v_p_noncrop)
    print("S(P) noncrop = %f \n" % s_p_noncrop)

    p_crop_err = s_p_crop * 1.96
    print("95% confidence interval for Producer's accuracy")
    print("95%% CI of Producer accuracy for crop = %f" % p_crop_err)

    p_noncrop_err = s_p_noncrop * 1.96
    print("95%% CI of Producer accuracy for noncrop = %f \n" % p_noncrop_err)

    acc = tp_area + tn_area
    print("Overall accuracy")
    print("Overall accuracy = %f \n" % acc)

    v_acc = wh_crop**2 * u_crop * (1 - u_crop) / (tp + fp - 1) + wh_noncrop**2 * u_noncrop * (
        1 - u_noncrop
    ) / (fn + tn - 1)
    print("Estimated variance of the overall accuracy")
    print("V(O) = %f \n" % v_acc)

    s_acc = np.sqrt(v_acc)
    print("Estimated standard error of the overall accuracy")
    print("S(O) = %f \n" % s_acc)

    acc_err = s_acc * 1.96
    print("95% confidence interval for overall accuracy")
    print("95%% CI of overall accuracy = %f \n" % acc_err)

    a_pixels_crop = total_area_px * (tp_area + fn_area)
    print(
        "Adjusted map area in units of pixels \
            A^[pixels] crop = %f"
        % a_pixels_crop
    )

    a_pixels_noncrop = total_area_px * (tn_area + fp_area)
    print("A^[pixels] noncrop = %f \n" % a_pixels_noncrop)

    pixel_size = meta["transform"][0]

    a_ha_crop = a_pixels_crop * (pixel_size * pixel_size) / (100 * 100)
    print(
        "Adjusted map area in units of hectares \
        A^[ha] crop = %f"
        % a_ha_crop
    )

    a_ha_noncrop = a_pixels_noncrop * (pixel_size * pixel_size) / (100 * 100)
    print("A^[ha] noncrop = %f \n" % a_ha_noncrop)

    S_pk_crop = (
        np.sqrt(
            (wh_crop * tp_area - tp_area**2) / (tp + fp - 1)
            + (wh_noncrop * fn_area - fn_area**2) / (fn + tn - 1)
        )
        * total_area_px
    )
    print("Standard error for the area")
    print("S_pk_crop = %f" % S_pk_crop)

    S_pk_noncrop = (
        np.sqrt(
            (wh_crop * fp_area - fp_area**2) / (tp + fp - 1)
            + (wh_noncrop * tn_area - tn_area**2) / (fn + tn - 1)
        )
        * total_area_px
    )
    print("S_pk_noncrop = %f \n" % S_pk_noncrop)

    a_pixels_crop_err = S_pk_crop * 1.96
    print("Margin of error for the 95% confidence interval")
    print("Crop area standard error 95%% confidence interval [pixels] = %f" % a_pixels_crop_err)

    a_pixels_noncrop_err = S_pk_noncrop * 1.96
    print(
        "Non-crop area standard error 95%% confidence interval [pixels] = %f \n"
        % a_pixels_noncrop_err
    )

    a_ha_crop_err = a_pixels_crop_err * (pixel_size**2) / (100**2)
    print("Margin of error for the 95% confidence interval in hectares")
    print("Crop area standard error 95%% confidence interval [ha] = %f" % a_ha_crop_err)

    a_ha_noncrop_err = a_pixels_noncrop_err * (pixel_size**2) / (100**2)
    print("Non-crop area standard error 95%% confidence interval [ha] = %f" % a_ha_noncrop_err)

    summary = pd.DataFrame(
        [
            [a_ha_crop, a_ha_noncrop],
            [a_ha_crop_err, a_ha_noncrop_err],
            [u_crop, u_noncrop],
            [u_crop_err, u_noncrop_err],
            [p_crop, p_noncrop],
            [p_crop_err, p_noncrop_err],
            [acc, acc],
            [acc_err, acc_err],
        ],
        index=pd.Index(
            [
                "Estimated area [ha]",
                "95% CI of area [ha]",
                "User accuracy",
                "95% CI of user acc",
                "Producer accuracy",
                "95% CI of prod acc",
                "Overall accuracy",
                "95% CI of overall acc",
            ]
        ),
        columns=["Crop", "Non-crop"],
    )

    summary.round(2)
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
