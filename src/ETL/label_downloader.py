from dataclasses import dataclass
from pathlib import Path
import logging
import urllib.request
import zipfile

logger = logging.getLogger(__name__)


@dataclass
class RawLabels:
    url: str

    def download_file(self, output_folder: Path, remove_zip: bool = True):

        filename = self.url.split("/")[-1]
        output_path = output_folder / filename

        if output_path.exists():
            logger.warning(f"{filename} already exists! Skipping")
            return None

        logger.info(f"Downloading {self.url}")
        urllib.request.urlretrieve(self.url, output_path)

        if filename.endswith("zip"):

            logger.info(f"Downloaded! Unzipping to {output_folder}")
            with zipfile.ZipFile(output_path, "r") as zip_file:
                zip_file.extractall(output_folder)

            if remove_zip:
                logger.info("Deleting zip file")
                (output_path).unlink()
