import sys
from argparse import ArgumentParser

sys.path.append("..")

from src.models import Model
from src.models import train_model
from src.processors.pv_kenya import KenyaPVProcessor
from src.processors.kenya_non_crop import KenyaNonCropProcessor
from src.processors.oaf_kenya import KenyaOAFProcessor
from src.exporters.geowiki import GeoWikiExporter

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)

    model_args = Model.add_model_specific_args(parser).parse_args()
    model = Model(model_args, datasets=[
        KenyaPVProcessor.dataset,
        KenyaNonCropProcessor.dataset,
        GeoWikiExporter.dataset,
        KenyaOAFProcessor.dataset,
    ])

    train_model(model, model_args)
