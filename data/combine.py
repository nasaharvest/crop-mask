# Combines all datasets into one file
from pathlib import Path

import pandas as pd

dfs = []
for p in Path("data/datasets").glob("*.csv"):
    df = pd.read_csv(p)
    df["name"] = p.stem
    dfs.append(df[df["class_probability"] != 0.5])
df = pd.concat(dfs)
df["is_crop"] = df["class_probability"] > 0.5
columns = [
    "is_crop",
    "lon",
    "lat",
    "start_date",
    "end_date",
    "class_probability",
    "num_labelers",
    "subset",
    "eo_data",
    "eo_lon",
    "eo_lat",
    "eo_file",
    "eo_status",
    "name",
]
df[~df["eo_data"].isna()][columns].to_csv("data/all.csv", index=False)
