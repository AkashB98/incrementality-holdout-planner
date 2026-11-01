"""Validation: clean data passes, planted defects are reported."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset
from helpers import grid, row
from validation import validate_dataset, validate_row


class TestValidation(unittest.TestCase):
    def test_generated_dataset_is_clean(self):
        self.assertEqual(validate_dataset(generate_dataset()), [])

    def test_empty_dataset(self):
        self.assertEqual(validate_dataset([]), ["dataset is empty"])

    def test_missing_field(self):
        bad = row()
        del bad["sales"]
        issues = validate_row(bad)
        self.assertTrue(any("missing field 'sales'" in i for i in issues))

    def test_negative_sales(self):
        self.assertTrue(any("sales must be >= 0" in i
                            for i in validate_row(row(sales=-1.0))))

    def test_sales_must_be_number(self):
        self.assertTrue(any("sales must be a number" in i
                            for i in validate_row(row(sales="lots"))))

    def test_bool_week_rejected(self):
        self.assertTrue(any("week must be an int" in i
                            for i in validate_row(row(week=True))))

    def test_bad_period(self):
        self.assertTrue(any("period must be" in i
                            for i in validate_row(row(period="during"))))

    def test_duplicate_rows_flagged(self):
        rows = grid({"region_01": 100.0}, pre_weeks=2, post_weeks=0)
        issues = validate_dataset(rows + [rows[0]])
        self.assertTrue(any("duplicate" in i for i in issues))

    def test_week_gap_flagged(self):
        rows = [row("region_01", 0), row("region_01", 3)]
        issues = validate_dataset(rows)
        self.assertTrue(any("week gap" in i for i in issues))

    def test_never_raises_on_garbage(self):
        issues = validate_dataset([{}, {"region_id": 5}, row(sales=None)])
        self.assertTrue(issues)


if __name__ == "__main__":
    unittest.main()
