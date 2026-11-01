"""Matching: features, OLS, pairing, assignment, quality."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset
from helpers import grid
from matching import (correlation, matching_quality, ols_slope,
                      pair_distance, pair_regions, region_features,
                      weekly_group_means)


class TestHelpersMath(unittest.TestCase):
    def test_ols_slope_flat(self):
        self.assertEqual(ols_slope([0, 1, 2], [5.0, 5.0, 5.0]), 0.0)

    def test_ols_slope_known(self):
        self.assertAlmostEqual(ols_slope([0, 1, 2, 3], [1.0, 3.0, 5.0, 7.0]), 2.0)

    def test_ols_slope_single_point(self):
        self.assertEqual(ols_slope([0], [3.0]), 0.0)

    def test_correlation_perfect(self):
        self.assertAlmostEqual(correlation([1, 2, 3], [2, 4, 6]), 1.0)

    def test_correlation_negative(self):
        self.assertAlmostEqual(correlation([1, 2, 3], [6, 4, 2]), -1.0)

    def test_correlation_constant_series(self):
        self.assertEqual(correlation([1, 1, 1], [1, 2, 3]), 0.0)


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.rows = generate_dataset()
        self.matched = pair_regions(self.rows)

    def test_twenty_pairs_cover_all_regions(self):
        self.assertEqual(self.matched["pair_count"], 20)
        flat = [r for pair in self.matched["pairs"] for r in pair]
        self.assertEqual(sorted(flat),
                         sorted({r["region_id"] for r in self.rows}))
        self.assertEqual(self.matched["unmatched"], [])

    def test_assignment_splits_pairs(self):
        test = set(self.matched["assignment"]["test"])
        control = set(self.matched["assignment"]["control"])
        self.assertEqual((len(test), len(control)), (20, 20))
        self.assertFalse(test & control)
        for a, b in self.matched["pairs"]:
            self.assertEqual((a in test) + (b in test), 1)

    def test_assignment_alternates_by_pair_index(self):
        pairs = self.matched["pairs"]
        test = set(self.matched["assignment"]["test"])
        self.assertIn(pairs[0][0], test)
        self.assertIn(pairs[1][1], test)

    def test_matching_is_deterministic(self):
        again = pair_regions(self.rows)
        self.assertEqual(again["pairs"], self.matched["pairs"])
        self.assertEqual(again["assignment"], self.matched["assignment"])

    def test_features_shape(self):
        feats = region_features(self.rows)
        self.assertEqual(len(feats), 40)
        for f in feats.values():
            self.assertEqual(f["weeks"], 52)
            self.assertGreater(f["mean"], 0)

    def test_features_hand_computed(self):
        rows = grid({"region_01": 100.0, "region_02": 200.0},
                    pre_weeks=4, post_weeks=0)
        feats = region_features(rows)
        self.assertEqual(feats["region_01"]["mean"], 100.0)
        self.assertEqual(feats["region_01"]["std"], 0.0)
        self.assertEqual(feats["region_01"]["slope"], 0.0)

    def test_identical_regions_pair_together(self):
        rows = grid({"region_01": 100.0, "region_02": 100.0,
                     "region_03": 900.0, "region_04": 900.0},
                    pre_weeks=4, post_weeks=0)
        matched = pair_regions(rows)
        as_sets = {frozenset(p) for p in matched["pairs"]}
        self.assertIn(frozenset({"region_01", "region_02"}), as_sets)
        self.assertIn(frozenset({"region_03", "region_04"}), as_sets)

    def test_pair_distance_ingredients(self):
        # Same level + perfectly correlated = distance 0; level gap and
        # correlation gap each add exactly their term.
        self.assertEqual(pair_distance(1.0, 1.0, 1.0), 0.0)
        self.assertEqual(pair_distance(2.0, 1.0, 1.0), 1.0)
        self.assertEqual(pair_distance(1.0, 1.0, 0.5), 0.5)
        self.assertAlmostEqual(pair_distance(0.0, 1.0, 0.0), 2.0)

    def test_quality_seeded_values(self):
        q = self.matched["quality"]
        self.assertEqual(q["pre_weeks_compared"], 52)
        self.assertLess(q["abs_mean_difference_pct"], 0.05)
        self.assertGreater(q["weekly_correlation"], 0.8)

    def test_quality_perfect_balance_on_identical_arms(self):
        rows = grid({"region_01": 100.0, "region_02": 100.0},
                    pre_weeks=4, post_weeks=0)
        q = matching_quality(rows, ["region_01"], ["region_02"])
        self.assertEqual(q["mean_difference"], 0.0)
        self.assertEqual(q["weekly_correlation"], 0.0)  # constant series

    def test_weekly_group_means(self):
        rows = grid({"region_01": 100.0, "region_02": 300.0},
                    pre_weeks=2, post_weeks=0)
        means = weekly_group_means(rows, ["region_01", "region_02"], "pre")
        self.assertEqual(means, {0: 200.0, 1: 200.0})


if __name__ == "__main__":
    unittest.main()
