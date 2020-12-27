import sys
from pathlib import Path

sys.path.append("..")

from src.processors import (
    GeoWikiProcessor,
    KenyaPVProcessor,
    KenyaNonCropProcessor,
    KenyaOAFProcessor,
)


def process_geowiki():
    processor = GeoWikiProcessor(Path("../data"))
    processor.process()


def process_plantvillage():
    processor = KenyaPVProcessor(Path("../data"))
    processor.process()


def process_kenya_noncrop():
    processor = KenyaNonCropProcessor(Path("../data"))
    processor.process()


def process_one_acre():
    processor = KenyaOAFProcessor(Path("../data"))
    processor.process()


if __name__ == "__main__":
    process_geowiki()
    process_plantvillage()
    process_kenya_noncrop()
    process_one_acre()
