from unittest import TestCase

import pandas as pd

from src.ETL.constants import SUBSET
from src.ETL.processor import Processor


class TestProcessor(TestCase):
    def test_train_val_test_split(self):
        df = pd.DataFrame({"col": list(range(100))})

        df = Processor.train_val_test_split(df, (1.0, 0.0, 0.0))
        expected_subsets = {"training": 100}
        self.assertEqual(df[SUBSET].value_counts().to_dict(), expected_subsets)

        df = Processor.train_val_test_split(df, (0.8, 0.1, 0.1))
        actual_subsets = df[SUBSET].value_counts().to_dict()
        threshold = 10
        self.assertTrue(
            abs(actual_subsets["training"] - 80) < threshold, actual_subsets["training"]
        )
        self.assertTrue(
            abs(actual_subsets["validation"] - 10) < threshold, actual_subsets["validation"]
        )
        self.assertTrue(abs(actual_subsets["testing"] - 10) < threshold, actual_subsets["testing"])
