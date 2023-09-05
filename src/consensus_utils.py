from typing import Callable, List, Optional

import numpy as np
import pandas as pd


def path_fn(set_id: str, date: str) -> str:
    """Returns string path to CEO *.csv file.

    Gives the path + file name to the csv label file by labeler set `set_id` at the timestamp
    `date`. For labeled CEO files, the files are named identically except for labeler set and
    timestamp date.

    Example : how to generalize the file name

    -> File for set 1 :
        ceo-Tigray-2020-2021-Change-(set-1)-sample-data-2022-01-10.csv

    -> File for set 2 :
        ceo-Tigray-2020-2021-Change-(set-2)-sample-data-2022-01-17.csv

    -> Generalized file name:
        ceo-Tigray-2020-2021-Change-({set_id})-sample-data-2020-{date}.csv

    Args
        set_id:
          String indicating the label set as it appears on the labeling csv file.
          Example: 'set-1', or 'set-2'.
        date:
          String indicating the date as it appears on the labeling csv file.
    Returns
        path:
          String indicating path to csv label file for `set_id` at `date`.

    """

    path = f"../data/raw/ceo-Tigray-2020-2021-Change-({set_id})-sample-data-2022-{date}.csv"
    return path


def isna(df: pd.DataFrame, column: str) -> bool:
    """Checks for presence of any NaN values in specified column."""
    return df[column].isna().any().any()


def tofloat(string: str) -> float:
    return float(string.split(" ")[0])


def check_dataframes(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Peforms check on set of CEO files loaded as dataframe.

    Checks that the set of dataframes all -
        (1) Have the same shape
        (2) Do not contain duplicate rows/points
        (3) Do not contain any NaNs/missing values

    Args:
        dfs:
            List-like containing up to three dataframes - minimum of two. Each dataframe is a
            labeled CEO file of the same ROI by a different set (two).

            In the case of three dataframes - the third is considered the "final" agreement from
            either of the two labeler sets.

    Returns:
        dfs:
            List-like containing the same dataframes after passing checks for shape, duplicates, and
            NaNs/missing values.

    """

    label = dfs[0].columns[-1]
    if len(dfs) > 2:
        label = dfs[0].columns[-2:].to_list()

    # Shape
    if not all([df.shape for df in dfs]):
        for i, df in enumerate(dfs):
            dfs[i] = df.drop_duplicates(subset=["plotid", "sampleid"], ignore_index=True)
    # NaNs
    if any([isna(df, label) for df in dfs]):
        for i, df in enumerate(dfs):
            dfs[i] = df.dropna(axis=0, subset=[])
        indices = dfs[0].index.intersection(dfs[1].index).intersection(dfs[2].index)
        for i, df in enumerate(dfs):
            dfs[i] = df.loc[indices, :]
    return dfs


def load_dataframes(
    path_fn: Callable[[str, str], str],
    completed_date: Optional[str] = None,
    final_date: Optional[str] = None,
) -> List[pd.DataFrame]:
    """Loads multiple CEO files of the same project from *.csv to a dataframe.

    There are two types of CEO projects:
        (1) Mapping, consisting of at least two CEO files.

        (2) Estimation, consisting of potentially several CEO files.
            -> There will be two CEO files stamped at a date when labeling
               is completed for all points for both sets, the "completed date".

            -> There will be two CEO files from a much later date, after
               labeling is completed, and where any disagreements between
               the two sets have been forced into "agreement". There are no
               disagreements between the two sets for any points at this stage.
               This is the "final date".

            -> At the "final" agreement date, the CEO files of the two sets will
               be identical.

    Args:
        path_fn:
            A helper function to read in multiple CEO files of the same project.
        completed_date:
            String indicating the "completed" date as it appears on the CEO .csv file.
        final_date:
            String indicating the "final" date as it appears on the CEO .csv file.

    Returns:
        dfs:
            List-like containing the set of CEO *.csv files loaded to dataframe.

    """

    if (completed_date is not None) and (final_date is not None):
        print("{:^61}\n{}".format("Loading dataframes from file...", "-" * 59))
        # Dataframes @ completed date for set 1 and 2
        df1 = pd.read_csv(path_fn("set-1", completed_date))
        df2 = pd.read_csv(path_fn("set-2", completed_date))
        # Dataframe @ final date
        #   -> Arbitrarily choose "set-1", both sets are in agreement by this point.
        df3 = pd.read_csv(path_fn("set-1", final_date))

        dfs = check_dataframes([df1, df2, df3])
        return dfs

    else:
        print("{:^53}\n{}".format("Loading dataframes from file...", "-" * 51))
        # Dataframes @ completed date for set 1 and 2
        df1 = pd.read_csv(path_fn("set-1", ""))
        df2 = pd.read_csv(path_fn("set-2", ""))

        dfs = check_dataframes([df1, df2])
        return dfs


def compute_disagreements(df1: pd.DataFrame, df2: pd.DataFrame, column_name: str) -> pd.Series:
    """Computes the disagreements between labeler sets.

    Args:
        df1:
            Dataframe of CEO file from a labeler set.
        df2:
            Dataframe of CEO file from a labeler set, different from df1.
        column_name:
            Name of column to make comparison from df1 and df2 for differences.

    Returns
        disagreements:
            Indices of where values of column_name in df1 and df2 are not equal to eachother.

    """

    print("\n{:^61}\n{}".format("Computing disagreements...", "-" * 59))
    disagreements = df1[column_name] != df2[column_name]
    print(f"Disagreements between labeler sets 1 and 2 : {disagreements.sum()}")
    return disagreements


def create_consensus_features(consensus_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Creates and adds features to consensus dataframe."""

    # Convert analysis duration to float
    consensus_dataframe[
        ["set_1_analysis_duration", "set_2_analysis_duration"]
    ] = consensus_dataframe[["set_1_analysis_duration", "set_2_analysis_duration"]].applymap(
        tofloat
    )

    # (1)
    def compute_incorrect_label_aux(l1, l2, f):
        return l2 if l1 == f else l1 if l2 == f else "Both"

    def compute_incorrect_label(df):
        return compute_incorrect_label_aux(df["set_1_label"], df["set_2_label"], df["final_label"])

    consensus_dataframe["overridden_label"] = consensus_dataframe.apply(
        compute_incorrect_label,
        axis=1,
    )

    def compute_incorrect_email_aux(e1, e2, l1, l2, f):
        return e2 if l1 == f else e1 if l2 == f else "Both"

    def compute_incorrect_email(df):
        return compute_incorrect_email_aux(
            df["set_1_email"],
            df["set_2_email"],
            df["set_1_label"],
            df["set_2_label"],
            df["final_label"],
        )

    consensus_dataframe["overridden_email"] = consensus_dataframe.apply(
        compute_incorrect_email,
        axis=1,
    )

    def compute_incorrect_analysis_aux(t1, t2, l1, l2, f):
        return t2 if l1 == f else t1 if l2 == f else "Both"

    def compute_correct_analysis_aux(t1, t2, l1, l2, f):
        return t1 if l1 == f else t2 if l2 == f else "None"

    def compute_incorrect_analysis(df):
        return compute_incorrect_analysis_aux(
            df["set_1_analysis_duration"],
            df["set_2_analysis_duration"],
            df["set_1_label"],
            df["set_2_label"],
            df["final_label"],
        )

    def compute_correct_analysis(df):
        return compute_correct_analysis_aux(
            df["set_1_analysis_duration"],
            df["set_2_analysis_duration"],
            df["set_1_label"],
            df["set_2_label"],
            df["final_label"],
        )

    consensus_dataframe["overridden_analysis"] = consensus_dataframe.apply(
        compute_incorrect_analysis,
        axis=1,
    )

    consensus_dataframe["nonoverridden_analysis"] = consensus_dataframe.apply(
        compute_correct_analysis,
        axis=1,
    )
    return consensus_dataframe


def create_consensus_dataframe_aux(
    dfs: List[pd.DataFrame], disagreements: pd.Series, area_change: bool = False
) -> pd.DataFrame:
    """Auxiliary function to create consensus dataframe."""

    label = "area_change" if area_change else "crop_noncrop"
    columns = ["plotid", "sampleid", "email", "analysis_duration", label]

    def renaming_func(s):
        return {
            label: f"{s}_label",
            "email": f"{s}_email",
            "analysis_duration": f"{s}_analysis_duration",
        }

    df1, df2, *df_list = dfs
    lon, lat = df1.loc[disagreements, "lon"].values, df1.loc[disagreements, "lat"].values
    df1 = df1.loc[disagreements, columns].rename(columns=renaming_func("set_1"))
    df2 = df2.loc[disagreements, columns].rename(columns=renaming_func("set_2"))

    if df_list:
        print("\n{:^61}".format("Creating consensus dataframe..."))
        df3 = df_list[0]
        df3 = (
            df3.loc[disagreements, columns]
            .rename(columns=renaming_func("final"))
            .drop(columns=["final_email", "final_analysis_duration"])
        )

        consensus_dataframe = df1.merge(
            df2, left_on=["plotid", "sampleid"], right_on=["plotid", "sampleid"]
        ).merge(df3, left_on=["plotid", "sampleid"], right_on=["plotid", "sampleid"])
        consensus_dataframe = create_consensus_features(consensus_dataframe)

        rcolumns = [
            "plotid",
            "sampleid",
            "lon",
            "lat",
            "set_1_email",
            "set_2_email",
            "overridden_email",
            "set_1_analysis_duration",
            "set_2_analysis_duration",
            "overridden_analysis",
            "nonoverridden_analysis",
            "set_1_label",
            "set_2_label",
            "final_label",
            "overridden_label",
        ]

    else:
        print("\n{:^53}".format("Creating consensus dataframe..."))
        consensus_dataframe = df1.merge(
            df2, left_on=["plotid", "sampleid"], right_on=["plotid", "sampleid"]
        )
        consensus_dataframe[
            ["set_1_analysis_duration", "set_2_analysis_duration"]
        ] = consensus_dataframe[["set_1_analysis_duration", "set_2_analysis_duration"]].applymap(
            tofloat
        )

        rcolumns = [
            "plotid",
            "sampleid",
            "lon",
            "lat",
            "set_1_email",
            "set_2_email",
            "set_1_analysis_duration",
            "set_2_analysis_duration",
            "set_1_label",
            "set_2_label",
        ]

    consensus_dataframe["lon"], consensus_dataframe["lat"] = lon, lat
    consensus_dataframe = consensus_dataframe[rcolumns]
    return consensus_dataframe


def create_consensus_dataframe(
    path_fn: Callable[[str, str], str],
    cdate: Optional[str] = None,
    fdate: Optional[str] = None,
    area_change: bool = False,
    y1: Optional[str] = None,
    y2: Optional[str] = None,
) -> pd.DataFrame:
    """Creates consensus dataframe.

    There are two types of CEO projects:
    (1) Mapping, consisting of at least two CEO files.

    (2) Estimation, consisting of potentially several CEO files.
        -> There will be two CEO files stamped at a date when labeling
            is completed for all points for both sets, the "completed date".

        -> There will be two CEO files from a much later date, after
            labeling is completed, and where any disagreements between
            the two sets have been forced into "agreement". There are no
            disagreements between the two sets for any points at this stage.
            This is the "final date".

        -> At the "final" agreement date, the CEO files of the two sets will
            be identical.

    Args:
        path_fn:
            A helper function to read in multiple CEO files of the same project.
        completed_date:
            String indicating the "completed" date as it appears on the CEO .csv file.
        final_date:
            String indicating the "final" date as it appears on the CEO .csv file.
        area_change:
            Bool indicating if CEO project is single year or multi-year.
        y1, y2:
            For multi-year change estimation - strings indicating the first and second
            year.

            With multi-year change estimation - CEO file will have two columns denoting
            active cropland in the first year, and second year.

    Returns:
        consensus_dataframe:
            A dataframe containing the disagreements between the two labeled CEO files.

    """

    def compute_area_change_aux(year_1_label: str, year_2_label: str) -> str:
        """Computes planting change."""

        match = {
            ("Planted", "Planted"): "Stable P",
            ("Not planted", "Not planted"): "Stable NP",
            ("Planted", "Not planted"): "P loss",
            ("Not planted", "Planted"): "P gain",
        }
        return match[year_1_label, year_2_label]

    def compute_area_change(df):
        return compute_area_change_aux(
            df[f"Was this a planted crop in {y1}?"], df[f"Was this a planted crop in {y2}?"]
        )

    label = "area_change" if area_change else "crop_noncrop"
    dfs = load_dataframes(path_fn, cdate, fdate)
    for df in dfs:
        if area_change:
            df[label] = df.apply(
                compute_area_change,
                axis=1,
            )
        else:
            df.rename(columns={"Does this pixel contain active cropland?": label}, inplace=True)

    disagreements = compute_disagreements(dfs[0], dfs[1], label)
    consensus_dataframe = create_consensus_dataframe_aux(dfs, disagreements, area_change)
    return consensus_dataframe


# (1a) Distribution of overridden labels
def label_overrides(df: pd.DataFrame) -> None:
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
    print("{:^25}\n{}".format("Incorrect Labels", "-" * 25))
    for label, count in zip(counts.index, counts.values):
        print("{:^17}: {:>2}".format(label, count))


# (1b) Distribution of mistaken labels
def label_mistakes(df: pd.DataFrame) -> None:
    # Counts of mistaken label
    counts = df["final_label"].value_counts().sort_index()

    # Print
    print("{:^25}\n{}".format("Mistaken Labels", "-" * 25))
    for label, count in zip(counts.index, counts.values):
        print("{:^17}: {:>2}".format(label, count))


# (1c) Distribution of disagreements
def label_disagreements(df):
    permutations = list(zip(df["set_1_label"], df["set_2_label"]))
    permutations_sorted = [tuple(sorted(pair)) for pair in permutations]
    counts = pd.Series(permutations_sorted).value_counts().sort_index()

    print("{:^43}\n{}".format("Distribution of Disagreements", "-" * 42))
    for (label_1, label_2), count in zip(counts.index, counts.values):
        print("{:^15} x {:^15} : {:^3}".format(label_1, label_2, count))


# (1d) Distribution of exact label-label changes
def label_transitions(df: pd.DataFrame) -> None:
    # Subset
    sdf = df[df["overridden_label"] != "Both"]

    # Counts of each label-label transition
    transitions = (
        pd.Series(list(zip(sdf["overridden_label"], sdf["final_label"])))
        .value_counts()
        .sort_index()
    )

    # Increment transitions with instances from both incidents
    #   -> TODO: Add robustness if none;
    bdf = df[df["overridden_label"] == "Both"]
    if bdf.shape[0] != 0:
        for set_label in ["set_1_label", "set_2_label"]:
            temp_transitions = (
                pd.Series(list(zip(bdf[set_label], bdf["final_label"]))).value_counts().sort_index()
            )
            transitions = transitions.add(temp_transitions, fill_value=0)
        transitions = transitions.astype(int)

    # Print
    print("{:^43}\n{}".format("Label-Label Transitions", "-" * 42))
    for (initial, final), count in zip(transitions.index, transitions.values):
        print("{:^15} -> {:^15} : {:^3}".format(initial, final, count))


# (2a) Number of times labeler overridden
def labeler_overrides(df: pd.DataFrame) -> None:
    # Counts of each labeler overridden
    counts = df["overridden_email"].value_counts().sort_values(ascending=False)

    # Print
    print("{:^43}\n{}".format("Frequency of Labeler Overridden", "-" * 42))
    for labeler, count in zip(counts.index, counts.values):
        print(" {:<34} : {:>3}".format(labeler, count))


# (3a) What is the difference in analysis duration for labels overridden?
def median_duration(df: pd.DataFrame) -> None:
    # Subset
    sdf = df[df["overridden_label"] != "Both"]

    # Subset overridden and nonoverridden analysis times
    overridden = sdf["overridden_analysis"].astype(np.float64)
    nonoverridden = sdf["nonoverridden_analysis"].astype(np.float64)

    # Append overridden analysis time with durations from both incidents
    bdf = df[df["overridden_label"] == "Both"]
    if bdf.shape[0] != 0:
        overridden = pd.concat(
            [
                overridden,
                pd.Series(
                    bdf[["set_1_analysis_duration", "set_2_analysis_duration"]]
                    .astype(np.float64)
                    .values.flatten()
                ),
            ]
        )

    # Print median duration times
    print("{:^37}\n{}".format("Median Analysis Duration", "-" * 35))
    print(
        "Overridden Points     : {:.2f} secs \nNon-Overridden Points : {:.2f} secs".format(
            overridden.median(), nonoverridden.median()
        )
    )


# (3b) Which overridden labels have the highest analysis duration?
def highest_duration(df: pd.DataFrame, q: float) -> None:
    # (2) Combine durations across both sets
    durations = df[["set_1_analysis_duration", "set_2_analysis_duration"]].values.flatten()

    # (3) Find qth quantile of analysis durations
    quantile = np.quantile(durations, q)

    # (4) Subset df where analysis durations higher than q
    #       -> In either set 1 or set 2
    sdf = df[
        (df["set_1_analysis_duration"] >= quantile) | (df["set_2_analysis_duration"] >= quantile)
    ]

    # (5) Print number of points with analysis duration higher than quantile
    print("{:^53}\n{}".format("Highest Analysis Durations", "-" * 52))
    print(
        """{:.2f} Quantile of Analysis Durations : {:.2f} secs
        \nAnalysis Time Greater than {:.2f} Quantile : {} points""".format(
            q, quantile, q, sdf.shape[0]
        )
    )

    # (6) Label-label transitions from points with analysis duration higher than quantile
    tdf = sdf[sdf["overridden_label"] != "Both"]
    transitions = (
        pd.Series(list(zip(tdf["overridden_label"], tdf["final_label"])))
        .value_counts()
        .sort_index()
    )

    # (6) Increment transitions count with instances from both incidents
    bdf = sdf[sdf["overridden_label"] == "Both"]
    if bdf.shape[0] != 0:
        for set_label in ["set_1_label", "set_2_label"]:
            temp_transitions = (
                pd.Series(list(zip(bdf[set_label], bdf["final_label"]))).value_counts().sort_index()
            )
            transitions = transitions.add(temp_transitions, fill_value=0)
        transitions = transitions.astype(int)

    # Print label-label transitions
    print("\n{:^53}\n{}".format("Label-Label Transitions", "-" * 52))
    for (initial, final), count in zip(transitions.index, transitions.values):
        print("{:^25} -> {:^15} : {:^3}".format(initial, final, count))
