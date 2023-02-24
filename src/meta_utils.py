import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable

# (1) Crop land **mapping** <- MOST GENERAL
#     -> NOTE: With crop land map there is no 'final' agreement between two labeler 
#              sets b/c there is typically no *forced* agreement or resolvement.

# (2) Crop land **area estimation**
#     -> NOTE: With area estimation there *is* final agreement between the two labeler sets. <- MOST COMMON
#     -> NOTE: Additionally; area estimation may also be for either single year (map) or 
#              multi-year (area change).

# (3) With area estimation there are additionally two types:
#     -> Single-year crop map area estimation
#     -> Multi-year crop map change area estimation 

# (3) Difference in **mapping** and **area estimation**
#     -> mapping : two csv files (set 1, set 2)
#     -> area est. : three csv files (set 1, set 2, 'final')

# (4) Goal:
#     -> Generalize functions st behavior adjusted depending on if labeling project is **mapping** 
#        or **area estimation** 
#     -> Don't require additional script file; instead have two separate notebooks for mapping
#        and area estimation but all util functions in one .py

def check_dataframes(
        df1 : pd.DataFrame, 
        df2 : pd.DataFrame, 
        df3 : Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, ...]:
    """ Performs checks on labeling CSVs loaded to dataframes. """

    if df3 is not None:
        labels = df1.columns[-2:].to_list()

        # (1) Check for equal shapes
        print(f"Native dataframe shapes : {df1.shape} , {df2.shape} , {df3.shape}")
        if not (df1.shape == df2.shape == df3.shape):
            print("Asymmetry found, attempting to make symmetry...")
            for df in [df1, df2, df3]:
                df.drop_duplicates(subset = ["plotid", "sampleid"], inplace = True, ignore_index = True)
            print(f"Adjusted dataframe shapes : {df1.shape}, {df2.shape}, {df3.shape}")

            if not (df1.shape == df2.shape == df3.shape):
                raise AssertionError("Unable to create symmetry between dataframes")

        # (2) Check for NaNs
        isna = lambda df : df[labels].isna().any().any()
        if isna(df1) or isna(df2) or isna(df3):
            print("NaN values found, dropping rows containing NaNs...")
            for df in [df1, df2, df3]:
                df.dropna(axis = 0, subset = [labels], inplace = True)

            print(f"Adjusted dataframe shapes : {df1.shape} , {df2.shape}")
            # Take the intersection of indices b/twn two dataframes after dropping NaNs and subset
            print(f"Taking index intersection of adjusted indices...")
            indices = df1.index.intersection(df2.index).intersection(df3.index)
            df1 = df1.loc[indices, :]
            df2 = df2.loc[indices, :]

        # (3) Check that ids are corresponding
        if not (df1.plotid == df2.plotid).all() and (df1.plotid == df3.plotid).all():
            raise AssertionError("IDs are not corresponding")

        print("Loading and checking dataframes complete!")
        return df1, df2, df3
    
    else:
        label = "Does this pixel contain active cropland?"

        # (1) Check for equal shape
        print(f"Native dataframe shapes   : {df1.shape} , {df2.shape}")
        if df1.shape != df2.shape:
            # Attempt to force symmetry by dropping potential duplicate values
            #   -> NOTE: Both dataframes can contain duplicate values -> TODO: Add handling...
            print("Asymmetry found, attempting to make symmetry...")
            for df in [df1, df2]: 
                df.drop_duplicates(subset = ["plotid", "sampleid"], inplace = True, ignore_index = True)
            # max(df1, df2, key = len).drop_duplicates(subset = ["plotid", "sampleid"], inplace = True, ignore_index = True)
            print(f"Adjusted dataframe shapes : {df1.shape} , {df2.shape}")
            
            # If shapes are still not equal; raise a ValueError
            if df1.shape != df2.shape:
                raise AssertionError("Unable to create symmetry between dataframes")

        # (2) Check for NaNs
        if df1[label].isna().any() or df2[label].isna().any():
            print("NaN values found, dropping rows containing NaNs...")
            for df in [df1, df2]:
                df.dropna(axis = 0, subset = [label], inplace = True)

            print(f"Adjusted dataframe shapes : {df1.shape} , {df2.shape}")
            # Take the intersection of indices b/twn two dataframes after dropping NaNs and subset
            print(f"Taking index intersection of adjusted indices...")
            indices = df1.index.intersection(df2.index)
            df1 = df1.loc[indices, :]
            df2 = df2.loc[indices, :]

        # (3) Check that ids are corresponding
        if (df1.plotid != df2.plotid).all():
            raise AssertionError("IDs are not corresponding.")
        
        print("Loading and checking dataframes complete!")
        return df1, df2

def load_dataframes(
        path_fn : Callable[[str], str], 
        completed_date : Optional[str] = None,
        final_date : Optional[str] = None
    ) -> Tuple[pd.DataFrame, ...]:
    """ Loads labeling CSVs to dataframe.
    
    Args:
    
    Returns:

    """

    if (completed_date is not None) and (final_date is not None):
        print("{:^61}\n{}".format("Loading dataframes from file...", "-" * 59))
        # Dataframes @ completed date for set 1 and 2
        cdf1 = pd.read_csv(path_fn("set-1", completed_date))
        cdf2 = pd.read_csv(path_fn("set-2", completed_date))
        # Dataframe @ final date 
        #   -> Arbitrarily choose "set-1", both sets are in agreement by this point. 
        fdf = pd.read_csv(path_fn("set-1", final_date))

        return check_dataframes(cdf1, cdf2, fdf)

    else:
        print("{:^53}\n{}".format("Loading dataframes from file...", "-" * 51))
        # Dataframes @ completed date for set 1 and 2
        cdf1 = pd.read_csv(path_fn("set-1"))
        cdf2 = pd.read_csv(path_fn("set-2"))

        return check_dataframes(cdf1, cdf2)

def compute_area_change(year_1_label : str, year_2_label : str) -> str:
    """ Computes planting change. """

    match = {
        ("Planted", "Planted") : "Stable P",
        ("Not planted", "Not planted") : "Stable NP",
        ("Planted", "Not planted") : "P loss",
        ("Not planted", "Planted") : "P gain",
    }

    return match[year_1_label, year_2_label]


def compute_disagreements(df1 : pd.DataFrame, df2 : pd.DataFrame, area_change : bool = False) -> pd.Series:
    """ Computes disagreements between labeler sets. """
    
    if area_change:
        print("\n{:^61}\n{}".format("Computing disagreements...", "-"*59))
        disagreements = (df1["area_change"] != df2["area_change"])
    else:
        print("\n{:^53}\n{}".format("Computing disagreements...", "-"*51))
        disagreements = (df1["crop_noncrop"] != df2["crop_noncrop"])
    
    print(f"Disagreements between labeler sets 1 and 2 : {disagreements.sum()}")
    return disagreements


def create_meta_features(meta_dataframe : pd.DataFrame) -> pd.DataFrame:
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
        fdf : Optional[pd.DataFrame] = None,
        area_change : bool = False
    ) -> pd.DataFrame:
    """ Auxiliary function to create meta dataframe.

    Args:

    Returns:    
    
    """

    # Pull lat and lon from one of the dataframes
    #   -> There could be conflict if merging includes `lon` and `lat` due to slight 
    #      variation between saved CSV files - but otherwise plotid/sampleid/lon/lat
    #      refer to the same locations 
    lon, lat = cdf1.loc[disagreements, "lon"].values, cdf1.loc[disagreements, "lat"].values

    # Extract columns to subset and eventually merge dataframes on 
    columns = ["plotid", "sampleid", "email", "analysis_duration"]

    # (1) If `fdf`` is not None, then area estimation!
    if fdf is not None:
        print("\n{:^61}".format("Creating meta dataframe..."))
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
        print("\n{:^53}".format("Creating meta dataframe..."))

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

        # Convert analysis duration to float
        tofloat = lambda string : float(string.split(" ")[0])
        meta_dataframe[["set_1_analysis_duration", "set_2_analysis_duration"]] = meta_dataframe[["set_1_analysis_duration", "set_2_analysis_duration"]].applymap(tofloat)

        # Rearrange columns
        rcolumns = [
            "plotid", "sampleid", "lon", "lat", "set_1_email", "set_2_email", 
            "set_1_analysis_duration", "set_2_analysis_duration", "set_1_label", "set_2_label", 
        ]
        meta_dataframe = meta_dataframe[rcolumns]

        return meta_dataframe


def create_meta_dataframe(
        path_fn : Callable[[str], str],
        cdate : Optional[str] = None,
        fdate : Optional[str] = None,
        area_change : bool = False,
        y1 : Optional[str] = None,
        y2 : Optional[str] = None
    ) -> pd.DataFrame :
    """ Creates meta dataframe.

    Args:

    Returns:
    
    """
    
    # (1) Crop **area estimation**
    #     -> Crop **area**
    #     -> Crop **area change**
    if (cdate is not None) and (fdate is not None):
        # (1.1) Load labeling CSVs to dataframes
        cdf1, cdf2, fdf = load_dataframes(path_fn, cdate, fdate)
        
        # (1.2) If **area change** estimate
        if area_change:
            if y1 is None or y2 is None:
                raise ValueError("Area change `True` but both/either `y1` and/or `y2` unspecified.")

            for df in [cdf1, cdf2, fdf]:
                df["area_change"] = df.apply(
                    lambda df : compute_area_change(
                        df[f"Was this a planted crop in {y1}?"], 
                        df[f"Was this a planted crop in {y2}?"]
                        ),
                    axis = 1
                    )
                
        # (1.2) Else, is just **area** estimate
        else:
            for df in [cdf1, cdf2, fdf]:
                df.rename(
                    columns = {"Does this pixel contain active cropland?" : "crop_noncrop"},
                    inplace = True
                )

        # (1.3) Compute disagreements
        disagreements = compute_disagreements(cdf1, cdf2, area_change)

        # (1.4) Create dataframe from disagreements
        meta_dataframe = create_meta_dataframe_aux(cdf1, cdf2, disagreements, fdf, area_change)

        return meta_dataframe
    
    # (2) Crop **mapping**
    else:
        # (2.1) Load labeling CSVs to dataframes
        cdf1, cdf2 = load_dataframes(path_fn)

        # (2.2) Rename label column
        for df in [cdf1, cdf2]:
            df.rename(
                columns = {"Does this pixel contain active cropland?" : "crop_noncrop"},
                inplace = True
            )

        # (2.3) Compute disagreements
        disagreements = compute_disagreements(cdf1, cdf2)

        # (2.4) Create dataframe from disagreements
        meta_dataframe = create_meta_dataframe_aux(cdf1, cdf2, disagreements)

        return meta_dataframe
    
# (1a) Distribution of overridden labels

def label_overrides(df : pd.DataFrame) -> None:
    # Subset 
    sdf = df[df["overridden_label"] != "Both"]

    # Counts of each label overridden
    counts = sdf["overridden_label"].value_counts().sort_index()

    # Increment with instances of both
    bdf = df[df["overridden_label"] == "Both"]
    if bdf.shape[0] != 0:
        for label_1, label_2 in zip(bdf["set_1_label"], bdf["set_2_label"]):
            counts[label_1] += 1
            counts[label_2] += 1

    # Print 
    print("{:^25}\n{}".format("Incorrect Labels", "-"*25))
    for label, count in zip(counts.index, counts.values):
        print("{:^17}: {:>2}".format(label, count))

# (1b) Distribution of mistaken labels

def label_mistakes(df : pd.DataFrame) -> None:
    # Counts of mistaken label
    counts = df["final_label"].value_counts().sort_index()
    
    # Print
    print("{:^25}\n{}".format("Mistaken Labels", "-"*25))
    for label, count in zip(counts.index, counts.values):
        print("{:^17}: {:>2}".format(label, count))

# (1c) Distribution of disagreements

def label_disagreements(df):
    permutations = list(zip(df["set_1_label"], df["set_2_label"]))
    permutations_sorted = [tuple(sorted(pair)) for pair in permutations]
    counts = pd.Series(permutations_sorted).value_counts().sort_index()
    
    print("{:^43}\n{}".format("Distribution of Disagreements", "-"*42))
    for (label_1, label_2), count in zip(counts.index, counts.values):
        print("{:^15} x {:^15} : {:^3}".format(label_1, label_2, count))


# (1d) Distribution of exact label-label changes

def label_transitions(df : pd.DataFrame) -> None:
    # Subset
    sdf = df[df["overridden_label"] != "Both"]

    # Counts of each label-label transition
    transitions = pd.Series(list(zip(sdf["overridden_label"], sdf["final_label"]))).value_counts().sort_index()

    # Increment transitions with instances from both incidents
    #   -> TODO: Add robustness if none; 
    bdf = df[df["overridden_label"] == "Both"]
    if bdf.shape[0] != 0:
        for set_label in ["set_1_label", "set_2_label"]:
            temp_transitions = pd.Series(list(zip(bdf[set_label], bdf["final_label"]))).value_counts().sort_index()
            transitions = transitions.add(temp_transitions, fill_value = 0)
        transitions = transitions.astype(int)

    # Print 
    print("{:^43}\n{}".format("Label-Label Transitions", "-"*42))
    for (initial, final), count in zip(transitions.index, transitions.values):
        print("{:^15} -> {:^15} : {:^3}".format(initial, final, count))

# (2a) Number of times labeler overridden

def labeler_overrides(df : pd.DataFrame) -> None:
    # Counts of each labeler overridden
    counts = df["overridden_email"].value_counts().sort_values(ascending = False)

    # Print
    print("{:^43}\n{}".format("Frequency of Labeler Overridden", "-"*42))
    for labeler, count in zip(counts.index, counts.values):
        print(" {:<34} : {:>3}".format(labeler, count))

# (3a) What is the difference in analysis duration for labels overridden?

def median_duration(df : pd.DataFrame) -> None:
    # Subset 
    sdf = df[df["overridden_label"] != "Both"]

    # Subset overridden and nonoverridden analysis times
    overridden = sdf["overridden_analysis"].astype(np.float64)
    nonoverridden = sdf["nonoverridden_analysis"].astype(np.float64)

    # Append overridden analysis time with durations from both incidents
    #   -> TODO: Add robustness if none; 
    bdf = df[df["overridden_label"] == "Both"]
    if bdf.shape[0] != 0:
        overridden = pd.concat([
            overridden,
            pd.Series(bdf[["set_1_analysis_duration", "set_2_analysis_duration"]].astype(np.float64).values.flatten())
        ])

    # Print median duration times
    print("{:^37}\n{}".format("Median Analysis Duration", "-"*35))
    print(
        "Overridden Points     : {:.2f} secs \nNon-Overridden Points : {:.2f} secs"
        .format(overridden.median(), nonoverridden.median())
    )

# (3b) Which overridden labels have the highest analysis duration?

def highest_duration(df : pd.DataFrame, q : float) -> None:
    # (2) Combine durations across both sets
    durations = df[["set_1_analysis_duration", "set_2_analysis_duration"]].values.flatten()
    
    # (3) Find qth quantile of analysis durations
    quantile = np.quantile(durations, q) 

    # (4) Subset df where analysis durations higher than q 
    #       -> In either set 1 or set 2
    sdf = df[(df["set_1_analysis_duration"] >= quantile) | (df["set_2_analysis_duration"] >= quantile)]
    
    # (5) Print number of points with analysis duration higher than quantile
    print("{:^53}\n{}".format("Highest Analysis Durations", "-"*52))
    print(
        "{:.2f} Quantile of Analysis Durations : {:.2f} secs \nAnalysis Time Greater than {:.2f} Quantile : {} points"
        .format(q, quantile, q, sdf.shape[0])
    )
    
    # (6) Label-label transitions from points with analysis duration higher than quantile
    tdf = sdf[sdf["overridden_label"] != "Both"]
    transitions = pd.Series(list(zip(tdf["overridden_label"], tdf["final_label"]))).value_counts().sort_index()

    # (6) Increment transitions count with instances from both incidents
    #   -> TODO: Add robustness if none; 
    bdf = sdf[sdf["overridden_label"] == "Both"]
    if bdf.shape[0] != 0:
        for set_label in ["set_1_label", "set_2_label"]:
            temp_transitions = pd.Series(list(zip(bdf[set_label], bdf["final_label"]))).value_counts().sort_index()
            transitions = transitions.add(temp_transitions, fill_value = 0)
        transitions = transitions.astype(int)

    # Print label-label transitions
    print("\n{:^53}\n{}".format("Label-Label Transitions", "-"*52))
    for (initial, final), count in zip(transitions.index, transitions.values):
        print("{:^25} -> {:^15} : {:^3}".format(initial, final, count))