import sys
from argparse import ArgumentParser

sys.path.append("..")

from src.models import Model, train_model

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)

    model_args = Model.add_model_specific_args(parser).parse_args()
    model = Model(model_args)

    train_model(model, model_args)
