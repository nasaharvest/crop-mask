# Import libraries
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from sklearn.metrics import confusion_matrix


def return_map_bounds_as_df():
    "Read the bounding coordinates of the crop mask"
    root_dir = os.getcwd()
    prj_raster = [f for f in os.listdir(root_dir) if f.startswith("prj")][0]

    with rio.open(prj_raster) as src:
        bbx = src.bounds
    geom = box(*bbx)
    bbx_gdf = gpd.GeoDataFrame(geometry=[geom], crs=src.crs)
    return bbx_gdf


def load_ne(country_code, regions_of_interest):
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
        # region_of_interest & country_code will be collected using dropdown
        condition = ne_gdf["adm1_code"].str.startswith(country_code)
        boundary = ne_gdf[condition].copy()
        print("Entire country found!")

    else:
        # Check regions
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
                column="name", legend=True, legend_kwds={"loc": "lower right"}, figsize=(10, 10)
            )
        else:
            condition = ne_gdf["name"].isin(regions_of_interest)
            boundary = ne_gdf[condition].copy()
            print("All regions found!")
    return boundary


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json

    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def clip_raster(in_raster, boundary=None):
    """Clip the raster to the boundary
    in_raster: path to the input raster
    boundary: GeoDataFrame of the boundary
    """

    with rio.open(in_raster) as src:
        if boundary is None:
            print("No boundary provided. Clipping to map bounds.")
            bbx = src.bounds
            geom = box(*bbx)
            boundary = gpd.GeoDataFrame(geometry=[geom], crs=src.crs)

        else:
            print("Clipping to boundary.")
        boundary = boundary.to_crs(src.crs)
        boundary = getFeatures(boundary)
        raster, out_transform = mask(src, shapes=boundary, crop=True)
        raster = raster[0]
        # Update the metadata
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


def load_raster(in_raster, boundary=None):
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
            # Project the raster to the local UTM Zone
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


def binarize(raster, threshold=0.5):
    nodata = get_prj_src().nodata
    raster[raster < threshold] = 0
    raster[((raster >= threshold) & (raster != nodata))] = 1
    return raster.data.astype(np.uint8)


def cal_map_area_class(map_array, unit="pixels", px_size=10):
    """
    Calculate the area of each class in the map.
    Print the area when the unit is specified to be in pixel and ha.
    In case of fraction, the area printed and returned to be assigned to a variable.
    map_array: numpy array of the map
    """
    crop_px = np.where(map_array.flatten() == 1)
    noncrop_px = np.where(map_array.flatten() == 0)
    total = crop_px[0].shape[0] + noncrop_px[0].shape[0]
    if unit == "ha":
        # Multiply pixels by area per pixel and convert m to hectares
        crop_area = crop_px[0].shape[0] * (px_size * px_size) / 100000
        noncrop_area = noncrop_px[0].shape[0] * (px_size * px_size) / 100000
        print(
            f"Crop area: {crop_area:.2f} ha, Non-crop area: {noncrop_area:.2f} ha \n \
             Total area: {crop_area + noncrop_area:.2f} ha"
        )
        return crop_area, noncrop_area
    elif unit == "pixels":
        crop_area = int(crop_px[0].shape[0])
        noncrop_area = int(noncrop_px[0].shape[0])
        print(
            f"Crop area: {crop_area} pixels, Non-crop area: {noncrop_area} pixels \n \
            Total area: {crop_area + noncrop_area} pixels"
        )
        return crop_area, noncrop_area
    elif unit == "fraction":
        crop_area = int(crop_px[0].shape[0]) / total
        noncrop_area = int(noncrop_px[0].shape[0]) / total
        print(f"Crop area: {crop_area:.2f} fraction, Non-crop area: {noncrop_area:.2f} fraction")
        assert crop_area + noncrop_area == 1
        return crop_area, noncrop_area
    else:
        print("Please specify the unit as either 'pixels', 'ha', or 'fraction'")


def estimate_num_sample_per_class(
    f_croparea, f_noncroparea, u_crop, u_noncrop, equal_alloc=True, stderr=0.02
):

    s_crop = np.sqrt(u_crop * (1 - u_crop))
    s_noncrop = np.sqrt(u_noncrop * (1 - u_crop))

    n = np.round(((f_croparea * s_crop + f_noncroparea * s_noncrop) / stderr) ** 2)
    print(f"Num of sample size: {n}")

    if equal_alloc:
        # equal allocation
        n_crop = int(n / 2)
        n_noncrop = int(n - n_crop)
    else:
        # strata proportion allocation
        n_crop = np.round(n * f_croparea)
        n_noncrop = np.round(n * f_noncroparea)
    print(f"Num sample size for crop: {n_crop}")
    print(f"Num sample size for non-crop: {n_noncrop}")
    return n_crop, n_noncrop


def random_inds(binary_map, strata, sample_size):
    inds = np.where(binary_map == strata)
    rand_inds = np.random.choice(np.arange(inds[0].shape[0]), size=sample_size, replace=False)
    rand_px = inds[0][rand_inds]
    rand_py = inds[1][rand_inds]
    del inds
    return rand_px, rand_py


def get_prj_src():
    root_dir = os.getcwd()
    prj_raster = [f for f in os.listdir(root_dir) if f.startswith("prj")][0]
    return rio.open(prj_raster)


def sample_df(binary_map, n_crop, n_noncrop):
    """ """
    df_noncrop = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_noncrop["px"], df_noncrop["py"] = random_inds(binary_map, 0, int(n_noncrop))
    df_noncrop["pred_class"] = 0

    df_crop = pd.DataFrame([], columns=["px", "py", "pred_class"])
    df_crop["px"], df_crop["py"] = random_inds(
        binary_map, 1, int(n_crop)
    )  # binary_map, n_crop, n_noncrop TODO
    df_crop["pred_class"] = 1

    df_combined = pd.concat([df_crop, df_noncrop]).reset_index(drop=True)

    # get src
    src = get_prj_src()
    for r, row in df_combined.iterrows():
        lx, ly = src.xy(row["px"], row["py"])
        df_combined.loc[r, "lx"] = lx
        df_combined.loc[r, "ly"] = ly

    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    gdf = gpd.GeoDataFrame(df_combined, geometry=gpd.points_from_xy(df_combined.lx, df_combined.ly))

    gdf.crs = src.crs
    gdf_ceo = gdf.to_crs("EPSG:4326")
    gdf_ceo["PLOTID"] = gdf_ceo.index
    gdf_ceo["SAMPLEID"] = gdf_ceo.index

    # save to file
    gdf_ceo[["geometry", "PLOTID", "SAMPLEID"]].to_file("ceo_reference_sample.shp", index=False)


# reference sample
def reference_sample_agree(binary_map, ceo_ref1, ceo_ref2):
    """ """
    src = get_prj_src()
    ceo_set1 = pd.read_csv(ceo_ref1)
    ceo_set2 = pd.read_csv(ceo_ref2)

    assert ceo_set1.columns[-1] == ceo_set2.columns[-1]

    label_question = ceo_set1.columns[-1]
    # check for any NANs/ missing answers
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
        # print("Number of duplicates removed:")

        ceo_set1.set_index("plotid", inplace=True)
        ceo_set2.set_index("plotid", inplace=True)
    else:
        print("The number of rows in the reference sets are equal.")

    ceo_agree = ceo_set1[ceo_set1[label_question] == ceo_set2[label_question]]

    print(
        "Number of samples that are in agreement: %d out of %d (%.2f%%)"
        % (ceo_agree.shape[0], ceo_set1.shape[0], ceo_agree.shape[0] / ceo_set1.shape[0] * 100)
    )
    ceo_agree_geom = gpd.GeoDataFrame(
        ceo_agree, geometry=gpd.points_from_xy(ceo_agree.lon, ceo_agree.lat), crs="EPSG:4326"
    )
    ceo_agree_geom = ceo_agree_geom.to_crs(src.crs)
    # ceo_agree_geom.plot()

    # clip the agree samples to the map bbox TODO: remove this step later
    bbx = return_map_bounds_as_df()
    ceo_agree_geom = ceo_agree_geom[ceo_agree_geom.within(bbx.geometry[0])]

    label_responses = ceo_agree_geom[label_question].unique()
    assert len(label_responses) == 2

    for r, row in ceo_agree_geom.iterrows():
        lon, lat = row["geometry"].y, row["geometry"].x
        px, py = src.index(lat, lon)  # TODO self.src read from the projected raster
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


def compute_confusion_matrix(ceo_agree_geom, crop_area_px, noncrop_area_px):
    y_true = np.array(ceo_agree_geom["Reference label"]).astype(np.uint8)
    y_pred = np.array(ceo_agree_geom["Mapped class"]).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred).ravel()
    print("Error matrix \n")
    print("True negatives: %d" % cm[0])
    print("False positives: %d" % cm[1])
    print("False negatives: %d" % cm[2])
    print("True positives: %d" % cm[3])
    return cm


def compute_area_estimate(crop_area_px, noncrop_area_px, cm):

    tn, fp, fn, tp = cm

    tot_area_px = crop_area_px + noncrop_area_px

    # Wh  is the proportion of mapped area for each class
    wh_crop = crop_area_px / tot_area_px
    wh_noncrop = noncrop_area_px / tot_area_px
    print("Proportion of mapped area for each class")
    print("Crop: %.2f" % wh_crop)
    print("Non-crop: %.2f \n" % wh_noncrop)

    # fraction of the proportional area of each class
    # that was mapped as each category in the confusion matrix
    tp_area = tp / (tp + fp) * wh_crop
    fp_area = fp / (tp + fp) * wh_crop
    fn_area = fn / (fn + tn) * wh_noncrop
    tn_area = tn / (fn + tn) * wh_noncrop
    print("Fraction of the proportional area of each class")
    print(
        "TP crop: %f \t FP crop: %f \n FN noncrop: %f \t TN noncrop: %f \n"
        % (tp_area, fp_area, fn_area, tn_area)
    )

    # User's accuracy
    u_crop = tp_area / (tp_area + fp_area)
    print("User's accuracy")
    print("U_crop = %f" % u_crop)

    u_noncrop = tn_area / (tn_area + fn_area)
    print("U_noncrop = %f \n" % u_noncrop)

    #  estimated variance of user accuracy for each mapped class
    v_u_crop = u_crop * (1 - u_crop) / (tp + fp)
    print("Estimated variance of user accuracy for each mapped class")
    print("V(U)_crop = %f" % v_u_crop)

    v_u_noncrop = u_noncrop * (1 - u_noncrop) / (fn + tn)
    print("V(U)_noncrop = %f \n" % v_u_noncrop)

    # estimated standard error of user accuracy for each mapped class.
    s_u_crop = np.sqrt(v_u_crop)
    print("Estimated standard error of user accuracy for each mapped class")
    print("S(U)_crop = %f" % s_u_crop)

    s_u_noncrop = np.sqrt(v_u_noncrop)
    print("S(U)_noncrop = %f \n" % s_u_noncrop)

    # Get the 95% confidence interval for User's accuracy
    u_crop_err = s_u_crop * 1.96
    print("95% confidence interval for User's accuracy")
    print("95%% CI of User accuracy for crop = %f" % u_crop_err)

    u_noncrop_err = s_u_noncrop * 1.96
    print("95%% CI of User accuracy for noncrop = %f \n" % u_noncrop_err)

    # From 4.2 Producer's accuracy
    #  producer's accuracy (i.e., recall). We calculate it here in terms of proportion of area.
    p_crop = tp_area / (tp_area + fn_area)
    print("Producer's accuracy")
    print("P_crop = %f" % p_crop)

    p_noncrop = tn_area / (tn_area + fp_area)
    print("P_noncrop = %f \n" % p_noncrop)

    # Nj  is the estimated marginal total number of pixels of each reference class j
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

    # TODO: long_scalars error
    expr2_crop = p_crop**2 * (
        noncrop_area_px**2 * fn / (fn + tn) * (1 - fn / (fn + tn)) / (fn + tn - 1)
    )
    print("expr2 crop = %f" % expr2_crop)

    expr2_noncrop = p_crop**2 * (
        crop_area_px**2 * fp / (fp + tp) * (1 - fp / (fp + tp)) / (fp + tp - 1)
    )
    print("expr2 noncrop = %f \n" % expr2_noncrop)

    # variance of producer's accuracy for each mapped class.
    v_p_crop = (1 / n_j_crop**2) * (expr1_crop + expr2_crop)
    print("Variance of producer's accuracy for each mapped class")
    print("V(P) crop = %f" % v_p_crop)

    v_p_noncrop = (1 / n_j_noncrop**2) * (expr1_noncrop + expr2_noncrop)
    print("V(P) noncrop = %f \n" % v_p_noncrop)

    # stimated standard error of producer accuracy for each mapped class.
    s_p_crop = np.sqrt(v_p_crop)
    print("Estimated standard error of producer accuracy for each mapped class")
    print("S(P) crop = %f" % s_p_crop)

    s_p_noncrop = np.sqrt(v_p_noncrop)
    print("S(P) noncrop = %f \n" % s_p_noncrop)

    # Get the 95% confidence interval for Producer's accuracy
    p_crop_err = s_p_crop * 1.96
    print("95% confidence interval for Producer's accuracy")
    print("95%% CI of Producer accuracy for crop = %f" % p_crop_err)

    p_noncrop_err = s_p_noncrop * 1.96
    print("95%% CI of Producer accuracy for noncrop = %f \n" % p_noncrop_err)

    # O is the overall accuracy. We calculate it here in terms of proportion of area.
    acc = tp_area + tn_area
    print("Overall accuracy")
    print("Overall accuracy = %f \n" % acc)

    # V(O)  is the estimated variance of the overall accuracy
    v_acc = wh_crop**2 * u_crop * (1 - u_crop) / (tp + fp - 1) + wh_noncrop**2 * u_noncrop * (
        1 - u_noncrop
    ) / (fn + tn - 1)
    print("Estimated variance of the overall accuracy")
    print("V(O) = %f \n" % v_acc)

    # S(O)  is the estimated standard error of the overall accuracy
    s_acc = np.sqrt(v_acc)
    print("Estimated standard error of the overall accuracy")
    print("S(O) = %f \n" % s_acc)

    # Get the 95% confidence interval for overall accuracy
    acc_err = s_acc * 1.96
    print("95% confidence interval for overall accuracy")
    print("95%% CI of overall accuracy = %f \n" % acc_err)

    # Apixels  is the adjusted map area in units of pixels
    a_pixels_crop = tot_area_px * (tp_area + fn_area)
    print("Adjusted map area in units of pixels")
    print("A^[pixels] crop = %f" % a_pixels_crop)

    a_pixels_noncrop = tot_area_px * (tn_area + fp_area)
    print("A^[pixels] noncrop = %f \n" % a_pixels_noncrop)

    # Get pixel size
    src = get_prj_src()
    pixel_size = src.transform[0]

    # Aha  is the adjusted map area in units of hectares
    a_ha_crop = a_pixels_crop * (pixel_size * pixel_size) / (100 * 100)
    print("Adjusted map area in units of hectares")
    print("A^[ha] crop = %f" % a_ha_crop)

    a_ha_noncrop = a_pixels_noncrop * (pixel_size * pixel_size) / (100 * 100)
    print("A^[ha] noncrop = %f \n" % a_ha_noncrop)

    # The following equations are used to estimate the standard error for the area.
    # They are based on the calculations in Olofsson et al., 2014

    S_pk_crop = (
        np.sqrt(
            (wh_crop * tp_area - tp_area**2) / (tp + fp - 1)
            + (wh_noncrop * fn_area - fn_area**2) / (fn + tn - 1)
        )
        * tot_area_px
    )
    print("Standard error for the area")
    print("S_pk_crop = %f" % S_pk_crop)

    S_pk_noncrop = (
        np.sqrt(
            (wh_crop * fp_area - fp_area**2) / (tp + fp - 1)
            + (wh_noncrop * tn_area - tn_area**2) / (fn + tn - 1)
        )
        * tot_area_px
    )
    print("S_pk_noncrop = %f \n" % S_pk_noncrop)

    # Multiply  S(pk)  by 1.96 to get the margin of error for the 95% confidence interval
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

    # Summary of the final estimates of accuracy and area with
    # standard error at 95% confidence intervals:
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
