import numpy as np
import pandas as pd

# (1) Crop land **mapping** <- MOST GENERAL
#     -> NOTE: With crop land map there is no 'final' agreement between two labeler 
#              sets b/c there is typically no *forced* agreement or resolvement.

# (2) Crop land **area estimation**
#     -> NOTE: With area estimation there *is* final agreement between the two labeler sets. <- MOST COMMON
#     -> NOTE: Additionally; area estimation may also be for either single year (map) or 
#              multi-year (area change).

# (3) Area estimation there are additionally two types:
#     -> Single-year crop map area estimation
#     -> Multi-year crop map change area estimation 

# (3) Difference in **mapping** and **area estimation**
#     -> mapping : two csv files (set 1, set 2)
#     -> area est. : three csv files (set 1, set 2, 'final')

# (4) Goal:
#     -> Generalize functions st behavior adjusted depending on if labeling project is **mapping** 
#        or **area estimation** 
#     -> Don't require additional script file; instead maybe have two separate notebooks for mapping
#        and area estimation but all util functions in one .py

def load_dataframes(
        path_fn, 
        completed_date = "",
        final_date = ""
    ) -> tuple :
    """ Loads labeling CSVs to dataframe.
    
    Args:
    
    Returns:

    """

    if (completed_date and final_date):
        completed_dataframe_set_1 = pd.read_csv(path_fn("set-1", completed_date))
        completed_dataframe_set_2 = pd.read_csv(path_fn("set-2", completed_date))
        final_dataframe = pd.read_csv(path_fn("set-1", final_date))

        return completed_dataframe_set_1, completed_dataframe_set_2, final_dataframe    
    else:
        completed_dataframe_set_1 = pd.read_csv(path_fn("set-1"))
        completed_dataframe_set_2 = pd.read_csv(path_fn("set-2"))

        return completed_dataframe_set_1, completed_dataframe_set_2

def compute_area_change(year_1_label : str, year_2_label : str) -> str :
    """ Computes planting change. """

    match = {
        ("Planted", "Planted") : "Stable P",
        ("Not planted", "Not planted") : "Stable NP",
        ("Planted", "Not planted") : "P loss",
        ("Not planted", "Planted") : "P gain",
    }

    return match[year_1_label, year_2_label]


def compute_disagreements(df1 : pd.DataFrame, df2 : pd.DataFrame, area_change = False) -> pd.Series :
    """ Computes disagreements between labeler sets. """
    
    if area_change:
        disagreements = (df1["area_change"] != df2["area_change"])
    else:
        disagreements = (df1["crop_noncrop"] != df2["crop_noncrop"])
    
    return disagreements


def create_meta_features(meta_dataframe):
    """ Creates and adds meta features to meta dataframe. """

    # Create "meta-feature" columns
    #   -> (1) Label overridden
    #   -> (2) LabelER overridden
    #   -> (3) 'Correct' and 'incorrect' analysis duration

    # Convert analysis duration to float
    tofloat = lambda string : float(string.split(" ")[0])
    meta_dataframe[["set_1_analysis_duration", "set_2_analysis_duration"]] = meta_dataframe[["set_1_analysis_duration", "set_2_analysis_duration"]].applymap(tofloat)
    
    # (1) 
    compute_incorrect_label = lambda l1, l2, f : l2 if l1 == f else l1 if l2 == f else "Both"
    meta_dataframe["overridden_label"] = meta_dataframe.apply(
        lambda df : compute_incorrect_label(df["set_1_label"], df["set_2_label"], df["final_label"]),
        axis = 1
        )
    
    # (2)
    compute_incorrect_email = lambda e1, e2, l1, l2, f : e2 if l1 == f else e1 if l2 == f else "Both" 
    meta_dataframe["overridden_email"] = meta_dataframe.apply(
        lambda df : compute_incorrect_email(df["set_1_email"], df["set_2_email"], df["set_1_label"], df["set_2_label"], df["final_label"]),
        axis = 1
        )
    
    # (3)
    compute_incorrect_analysis = lambda t1, t2, l1, l2, f: t2 if l1 == f else t1 if l2 == f else 'Both'
    compute_correct_analysis = lambda t1, t2, l1, l2, f: t1 if l1 == f else t2 if l2 == f else 'None'
    meta_dataframe["overridden_analysis"] = meta_dataframe.apply(
        lambda df : compute_incorrect_analysis(df["set_1_analysis_duration"], df["set_2_analysis_duration"], df["set_1_label"], df["set_2_label"], df["final_label"]),
        axis = 1
    )
    meta_dataframe["nonoverridden_analysis"] = meta_dataframe.apply(
        lambda df : compute_correct_analysis(df["set_1_analysis_duration"], df["set_2_analysis_duration"], df["set_1_label"], df["set_2_label"], df["final_label"]),
        axis = 1
    )

    return meta_dataframe

def create_meta_dataframe_aux(
        cdf1 : pd.DataFrame,
        cdf2 : pd.DataFrame,
        disagreements : pd.Series,
        fdf : pd.DataFrame = None,
        area_change = False
    ):
    """ Auxiliary function to create meta dataframe.

    Args:

    Returns:    
    
    """

    # Pull lat and lon from one of the dataframes
    #   -> There could be conflict if merging includes `lon` and `lat` due to slight 
    #      variation between saved CSV files - but otherwise plotid/sampleid/lon/lat
    #      refer to the same locations 
    lon, lat = cdf1.loc[disagreements, "lon"], cdf1.loc[disagreements, "lat"]

    # Extract columns to subset and eventually merge dataframes on 
    columns = ["plotid", "sampleid", "email", "analysis_duration"]

    # (1) If `fdf`` is not None, then area estimation!
    if fdf is not None:
        # If area estimation, either area or area change estimation
        if area_change:
            columns.append("area_change")
            renamed = lambda s : {
                "area_change" : f"{s}_label",
                "email" : f"{s}_email",
                "analysis_duration" : f"{s}_analysis_duration"
            }
        else:
            columns.append("crop_noncrop")
            renamed = lambda s : {
                "crop_noncrop" : f"{s}_label",
                "email" : f"{s}_email",
                "analysis_duration" : f"{s}_analysis_duration"
            }

        # Subset and rename by set
        cdf1 = cdf1.loc[disagreements, columns].rename(columns = renamed("set_1"))
        cdf2 = cdf2.loc[disagreements, columns].rename(columns = renamed("set_2"))
        fdf = fdf.loc[disagreements, columns].rename(columns = renamed("final")).drop(columns = ['final_email', 'final_analysis_duration'])
        
        # Assemble dataframe
        meta_dataframe = cdf1.merge(
            cdf2, left_on = ["plotid","sampleid"], right_on = ["plotid","sampleid"]
            ).merge(
            fdf, left_on = ["plotid","sampleid"], right_on = ["plotid","sampleid"]
            )
        
        # Insert lon and lat columns
        meta_dataframe["lon"], meta_dataframe["lat"] = lon, lat

        # Create and add meta features
        meta_dataframe = create_meta_features(meta_dataframe)

        # Rearrange columns
        rcolumns = [
            "plotid", "sampleid", "lon", "lat", "set_1_email", "set_2_email", "overridden_email", 
            "set_1_analysis_duration", "set_2_analysis_duration", "overridden_analysis", "nonoverridden_analysis", 
            "set_1_label", "set_2_label", "final_label", "overridden_label"
        ]
        meta_dataframe = meta_dataframe[rcolumns]

        return meta_dataframe

    # (2) Else `fdf` is None, then crop mapping
    else:
        columns.append("crop_noncrop")
        renamed = lambda s : {
            "crop_noncrop" : f"{s}_label",
            "email" : f"{s}_email",
            "analysis_duration" : f"{s}_analysis_duration"
        }
        
        # Subset dataframes by disagreeing points and columns
        cdf1 = cdf1.loc[disagreements, columns].rename(columns = renamed("set_1"))
        cdf2 = cdf2.loc[disagreements, columns].rename(columns = renamed("set_2"))

        # Assemble dataframe
        meta_dataframe = cdf1.merge(
            cdf2, left_on = ["plotid", "sampleid"], right_on = ["plotid", "sampleid"]
        )

        # Insert lon and lat columns
        meta_dataframe["lon"], meta_dataframe["lat"] = lon, lat

        # Rearrange columns
        rcolumns = [
            "plotid", "sampleid", "lon", "lat", "set_1_email", "set_2_email", 
            "set_1_analysis_duration", "set_2_analysis_duration", "set_1_label", "set_2_label", 
        ]
        meta_dataframe = meta_dataframe[rcolumns]

        return meta_dataframe


def create_meta_dataframe(
        path_fn, 
        area_estimate = False, 
        area_change = False,
        year_1 = "",
        year_2 = "",
        completed_date = "",
        final_date = ""
    ) -> pd.DataFrame :
    """ Creates meta dataframe.

    Args:

    Returns:
    
    """
    
    # (1) Crop **area estimation**
    #     -> Crop area
    #     -> Crop area change
    if area_estimate:
        # (1.1) Load labeling CSVs to dataframes
        cdf1, cdf2, fdf = load_dataframes(path_fn, completed_date, final_date)
        
        # (1.2) If area change estimate
        if area_change:
            assert year_1 and year_2, "Area change `True` but `year_1` and `year_2` unspecified."

            for df in [cdf1, cdf2, fdf]:
                df["area_change"] = df.apply(
                    lambda df : compute_area_change(df[f"Was this a planted crop in {year_1}?"], df[f"Was this a planted crop in {year_2}?"]),
                    axis = 1
                    )
        # (1.2) Else is area estimate
        else:
            for df in [cdf1, cdf2, fdf]:
                df = df.rename(
                    columns = {"Does this pixel contain active cropland?" : "crop_noncrop"}
                )

        # (1.3) Compute disagreements
        disagreements = compute_disagreements(cdf1, cdf2, area_change)
        print(f"Disagreements Between Labeler Sets 1 and 2 : {disagreements.sum()}")

        # (1.4) Create dataframe from disagreements
        meta_dataframe = create_meta_dataframe(cdf1, cdf2, fdf, area_change)
        
        return meta_dataframe
    
    # (2) Crop **mapping**
    else:
        # (2.1) Load labeling CSVs to dataframes
        cdf1, cdf2 = load_dataframes(path_fn)

        # (2.2) Rename label column
        for df in [cdf1, cdf2]:
            df = df.rename(
                columns = {"Does this pixel contain active cropland?" : "crop_noncrop"}
            )

        # (2.3) Compute disagreements
        disagreements = compute_disagreements(cdf1, cdf2)
        print(f"Disagreements Between Labeler Sets 1 and 2 : {disagreements.sum()}")

        # (2.4) Create dataframe from disagreements
        meta_dataframe = create_meta_dataframe_aux(cdf1, cdf2, disagreements)

        return meta_dataframe