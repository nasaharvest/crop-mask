from pathlib import Path

from typing import Any, Dict


class BaseExporter:
    r"""Base for all exporter classes. It creates the appropriate
    directory in the data dir (``data_dir/raw/{dataset}``).

    All classes which extend this should implement an export function.

    :param data_folder (pathlib.Path, optional)``: The location of the data folder.
            Default: ``pathlib.Path("data")``
    """

    dataset: str
    default_args_dict: Dict[str, Any] = {}

    def __init__(self, data_folder: Path = Path("data")) -> None:

        self.data_folder = data_folder

        self.raw_folder = self.data_folder / "raw"
        self.output_folder = self.raw_folder / self.dataset
        self.output_folder.mkdir(parents=True, exist_ok=True)
