import logging

from src.dataset_config import labeled_datasets, unlabeled_datasets

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    for d in labeled_datasets:
        d.download_raw_labels()
        # d.export_earth_engine_data()

    # for d in unlabeled_datasets:
    #     d.export_earth_engine_data()
