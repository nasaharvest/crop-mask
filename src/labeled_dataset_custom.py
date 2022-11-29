import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from openmapflow.config import PROJECT_ROOT
from openmapflow.config import DataPaths as dp
from openmapflow.constants import (
    CLASS_PROB,
    COUNTRY,
    DATASET,
    END,
    EO_DATA,
    EO_FILE,
    EO_LAT,
    EO_LON,
    EO_STATUS,
    EO_STATUS_SKIPPED,
    LABEL_DUR,
    LABELER_NAMES,
    LAT,
    LON,
    NUM_LABELERS,
    SOURCE,
    START,
    SUBSET,
)
from openmapflow.labeled_dataset import (
    LabeledDataset,
    _label_eo_counts,
    get_label_timesteps,
)

from src.raw_labels import RawLabels

temp_dir = tempfile.gettempdir()


@dataclass
class CustomLabeledDataset(LabeledDataset):

    dataset: str = ""
    country: str = ""
    raw_labels: Tuple[RawLabels, ...] = ()

    def __post_init__(self):
        self.name = self.dataset
        self.df_path = PROJECT_ROOT / dp.DATASETS / (self.name + ".csv")
        self.raw_dir = PROJECT_ROOT / dp.RAW_LABELS / self.name

    def load_labels(self):
        """
        Creates a single processed labels file from a list of raw labels.
        """
        df = pd.DataFrame({})
        already_processed = []
        if self.df_path.exists():
            df = pd.read_csv(self.df_path)
            already_processed = df[SOURCE].unique()

        new_labels: List[pd.DataFrame] = []
        raw_year_files = [(p.filename, p.start_year) for p in self.raw_labels]
        if len(raw_year_files) != len(set(raw_year_files)):
            raise ValueError(f"Duplicate raw files found in: {raw_year_files}")
        for p in self.raw_labels:
            if p.filename not in str(already_processed):
                new_labels.append(p.process(self.raw_dir))

        if len(new_labels) == 0:
            return df

        df = pd.concat([df] + new_labels)

        # Combine duplicate labels
        df[NUM_LABELERS] = 1

        def join_if_exists(values):
            if all((isinstance(v, str) for v in values)):
                return ",".join(values)
            return ""

        df = df.groupby([LON, LAT, START, END], as_index=False, sort=False).agg(
            {
                SOURCE: lambda sources: ",".join(sources.unique()),
                CLASS_PROB: "mean",
                NUM_LABELERS: "sum",
                SUBSET: "first",
                LABEL_DUR: join_if_exists,
                LABELER_NAMES: join_if_exists,
                EO_DATA: "first",
                EO_LAT: "first",
                EO_LON: "first",
                EO_FILE: "first",
                EO_STATUS: "first",
            }
        )
        df[COUNTRY] = self.country
        df[DATASET] = self.dataset
        df.loc[df[CLASS_PROB] == 0.5, EO_STATUS] = EO_STATUS_SKIPPED

        df = df.reset_index(drop=True)
        df.to_csv(self.df_path, index=False)
        return df

    def summary(self, df: pd.DataFrame) -> str:
        timesteps = get_label_timesteps(df).unique()
        eo_status_str = str(df[EO_STATUS].value_counts()).rsplit("\n", 1)[0]
        return (
            f"{self.name} (Timesteps: {','.join([str(int(t)) for t in timesteps])})\n"
            + "----------------------------------------------------------------------------\n"
            + f"disagreement: {len(df[df[CLASS_PROB] == 0.5]) / len(df):.1%}\n"
            + eo_status_str
            + "\n"
            + _label_eo_counts(df)
            + "\n"
        )
