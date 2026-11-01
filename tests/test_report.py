"""Report assembly: matching + power + readout in one document."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset
from report import build_report


class TestReport(unittest.TestCase):
    def setUp(self):
        self.report = build_report(generate_dataset())

    def test_sections_present(self):
        for key in ("dataset", "matching", "power", "readout"):
            self.assertIn(key, self.report)

    def test_dataset_summary(self):
        self.assertEqual(self.report["dataset"]["rows"], 2400)
        self.assertEqual(self.report["dataset"]["regions"], 40)

    def test_matching_section(self):
        self.assertEqual(self.report["matching"]["pair_count"], 20)
        self.assertLess(self.report["matching"]["quality"]
                        ["abs_mean_difference_pct"], 0.05)

    def test_power_section(self):
        self.assertAlmostEqual(self.report["power"]["weekly_diff_std"],
                               176.7402, places=2)
        self.assertAlmostEqual(self.report["power"]["pair_mde_absolute"],
                               301.4982, places=3)
        self.assertEqual(len(self.report["power"]["table"]), 3)

    def test_readout_section_covers_truth(self):
        self.assertTrue(self.report["readout"]["covers_planted_lift"])
        self.assertEqual(self.report["readout"]["planted_lift_pct"], 0.05)

    def test_report_deterministic(self):
        self.assertEqual(build_report(generate_dataset()), self.report)


if __name__ == "__main__":
    unittest.main()
