from pathlib import Path
from src.utils import set_seed


class BaseProcessor:
    r"""Base for all processor classes. It creates the appropriate
    directory in the data dir (``data_dir/processed/{dataset}``).

    :param data_folder (pathlib.Path, optional)``: The location of the data folder.
            Default: ``pathlib.Path("data")``
    """

    dataset: str

    def __init__(self, data_folder: Path) -> None:

        set_seed()
        self.data_folder = data_folder
        self.raw_folder = self.data_folder / "raw" / self.dataset
        assert self.raw_folder.exists(), f"{self.raw_folder} does not exist!"

        self.output_folder = self.data_folder / "processed" / self.dataset
        self.output_folder.mkdir(exist_ok=True, parents=True)
