import sys
from pathlib import Path

sys.path.append("..")

from src.engineer import (
    GeoWikiEngineer,
    PVKenyaEngineer,
    KenyaNonCropEngineer,
    KenyaOAFEngineer,
)


def engineer_geowiki():
    engineer = GeoWikiEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.2)


def engineer_kenya():
    engineer = PVKenyaEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.1, test_set_size=0.1)


def engineer_kenya_noncrop():
    engineer = KenyaNonCropEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.1, test_set_size=0.1)


def engineer_oaf_kenya():
    engineer = KenyaOAFEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.1, test_set_size=0.1)


if __name__ == "__main__":
    engineer_geowiki()
    engineer_kenya()
    engineer_kenya_noncrop()
    engineer_oaf_kenya()
