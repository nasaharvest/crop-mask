"""
Trains the models
"""

import sys
from argparse import ArgumentParser

sys.path.append("..")

from src.models import Forecaster, forecaster_train_model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cache", type=bool, default=False)
    parser.add_argument("--input_months", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--logger_name", type=str, default="lightning_logs")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="Arizona")
    parser.add_argument("--num_bands", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--processed_data_folder", type=str, default='/cmlscratch/izvonkov/Arizona-processed2')
    parser.add_argument("--save_dir", type=str, default='/cmlscratch/hkjoo/repo/crop-mask/data/models')    
    parser.add_argument("--show_progress_bar", type=bool, default=False)

    model_args = Forecaster.add_model_specific_args(parser).parse_args()
    model = Forecaster(num_bands=model_args.num_bands, hparams=model_args)
    forecaster_train_model(model, model_args)
