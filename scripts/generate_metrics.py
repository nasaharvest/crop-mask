import os
import sys

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("..")

from src.generate_metrics import get_metrics_for_all_models

if __name__ == "__main__":
    get_metrics_for_all_models(test_mode=False)
