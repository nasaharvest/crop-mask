from datetime import date, timedelta
from pathlib import Path
from unittest import TestCase

import pandas as pd
import tempfile
import shutil

from src.ETL.processor import Processor


class TestProcessor(TestCase):

    temp_data_dir: Path

    @classmethod
    def setUpClass(cls):
        cls.temp_data_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_data_dir)

    def test_end_date_using_max_overlap(self):
        kwargs = {
            "planting_date_col": pd.to_datetime([date(2019, 11, 1)]),
            "harvest_date_col": pd.to_datetime([date(2020, 3, 1)]),
            "end_month_day": (4, 16),
            "total_days": timedelta(30 * 12),
        }
        end_date = Processor.end_date_using_overlap(**kwargs)
        self.assertEqual(2020, end_date[0].year)

        # Test when range misses end date by a bit
        kwargs["planting_date_col"] = pd.to_datetime([date(2019, 11, 1)])
        kwargs["harvest_date_col"] = pd.to_datetime([date(2020, 5, 1)])
        end_date = Processor.end_date_using_overlap(**kwargs)
        self.assertEqual(2020, end_date[0].year)

        # Test when range misses end date by a lot
        kwargs["planting_date_col"] = pd.to_datetime([date(2019, 11, 1)])
        kwargs["harvest_date_col"] = pd.to_datetime([date(2020, 11, 1)])
        end_date = Processor.end_date_using_overlap(**kwargs)
        self.assertEqual(2021, end_date[0].year)
