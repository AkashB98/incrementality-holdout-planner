"""Experiment: lift planting + pair-level DiD + permutation inference."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import (apply_lift, estimate_did, pair_differences,
                        permutation_p_value, simulate_experiment)
from generator import generate_dataset
from helpers import grid
from matching import pair_regions


def matched_for(pairs, test):
    all_r = [r for pair in pairs for r in pair]
    return {"pairs": pairs,
            "assignment": {"test": sorted(test),
                           "control": sorted(set(all_r) - set(test))}}


class TestApplyLift(unittest.TestCase):
    def setUp(self):
        self.rows = generate_dataset()

    def test_lift_only_touches_test_post(self):
        lifted = apply_lift(self.rows, ["region_01"], 0.10)
        for orig, new in zip(self.rows, lifted):
            if orig["region_id"] == "region_01" and orig["period"] == "post":
                self.assertAlmostEqual(new["sales"], orig["sales"] * 1.10,
                                       places=2)
            else:
                self.assertEqual(new["sales"], orig["sales"])

    def test_zero_lift_is_identity(self):
        self.assertEqual(apply_lift(self.rows, ["region_01"], 0.0), self.rows)

    def test_original_rows_not_mutated(self):
        before = [dict(r) for r in self.rows]
        apply_lift(self.rows, ["region_01"], 0.5)
        self.assertEqual(self.rows, before)


class TestPairDifferences(unittest.TestCase):
    def test_hand_computed(self):
        # test region rises 100 -> 130, control flat 100: pair diff = 30.
        rows = grid({"region_01": 100.0, "region_02": 100.0},
                    pre_weeks=2, post_weeks=2)
        rows = [dict(r, sales=130.0) if r["region_id"] == "region_01"
                and r["period"] == "post" else r for r in rows]
        matched = matched_for([["region_01", "region_02"]], ["region_01"])
        self.assertEqual(pair_differences(rows, matched), [30.0])

    def test_level_difference_cancels(self):
        # Test runs 20 above control in BOTH periods: pair diff = 0.
        rows = grid({"region_01": 120.0, "region_02": 100.0},
                    pre_weeks=4, post_weeks=2)
        matched = matched_for([["region_01", "region_02"]], ["region_01"])
        self.assertEqual(pair_differences(rows, matched), [0.0])

    def test_one_difference_per_pair_seeded(self):
        rows = generate_dataset()
        matched = pair_regions(rows)
        self.assertEqual(len(pair_differences(rows, matched)), 20)


class TestPermutation(unittest.TestCase):
    def test_zero_observed_is_p_one(self):
        self.assertEqual(permutation_p_value([0.0, 0.0, 0.0]), 1.0)

    def test_deterministic(self):
        diffs = [10.0, 12.0, 9.0, 11.0, 10.5]
        self.assertEqual(permutation_p_value(diffs, n_permutations=200),
                         permutation_p_value(diffs, n_permutations=200))

    def test_strong_effect_small_p(self):
        diffs = [10.0] * 20
        # Every flip pattern except all-same-sign is less extreme; with
        # 2000 seeded draws the share at least as extreme is tiny.
        self.assertLess(permutation_p_value(diffs), 0.01)

    def test_symmetric_null_large_p(self):
        diffs = [10.0, -10.0] * 10
        self.assertEqual(permutation_p_value(diffs), 1.0)  # observed mean 0


class TestDidHandComputed(unittest.TestCase):
    def test_exact_recovery_on_constant_data(self):
        rows = grid({"region_01": 100.0, "region_02": 100.0},
                    pre_weeks=4, post_weeks=2)
        matched = matched_for([["region_01", "region_02"]], ["region_01"])
        readout = simulate_experiment(rows, matched, lift=0.10)
        self.assertAlmostEqual(readout["did_lift_pct"], 0.10)
        self.assertTrue(readout["covers_planted_lift"])
        self.assertEqual(readout["standard_error"], 0.0)

    def test_no_lift_gives_zero_did_on_constant_data(self):
        rows = grid({"region_01": 100.0, "region_02": 100.0},
                    pre_weeks=4, post_weeks=2)
        readout = estimate_did(rows, matched_for(
            [["region_01", "region_02"]], ["region_01"]))
        self.assertEqual(readout["did_absolute"], 0.0)
        self.assertEqual(readout["p_value"], 1.0)

    def test_missing_post_period_raises(self):
        rows = grid({"region_01": 100.0, "region_02": 100.0},
                    pre_weeks=4, post_weeks=0)
        with self.assertRaises(ValueError):
            estimate_did(rows, matched_for([["region_01", "region_02"]],
                                           ["region_01"]))


class TestDidSeeded(unittest.TestCase):
    def setUp(self):
        self.rows = generate_dataset()
        self.matched = pair_regions(self.rows)
        self.readout = simulate_experiment(self.rows, self.matched, lift=0.05)

    def test_recovers_planted_lift_within_ci(self):
        self.assertTrue(self.readout["covers_planted_lift"])
        lo, hi = self.readout["ci_95_pct"]
        self.assertLessEqual(lo, 0.05)
        self.assertGreaterEqual(hi, 0.05)

    def test_seeded_estimate_pinned(self):
        self.assertAlmostEqual(self.readout["did_lift_pct"], 0.043661, places=5)
        self.assertAlmostEqual(self.readout["did_absolute"], 485.2815, places=3)

    def test_primary_inference_is_permutation(self):
        self.assertEqual(self.readout["p_value_method"],
                         "randomisation (sign-flip) inference within pairs")
        self.assertEqual(self.readout["n_permutations"], 2000)
        self.assertLessEqual(self.readout["p_value"], 0.005)
        self.assertTrue(self.readout["significant_at_0_05"])

    def test_ci_brackets_estimate(self):
        lo, hi = self.readout["ci_95"]
        self.assertLess(lo, self.readout["did_absolute"])
        self.assertGreater(hi, self.readout["did_absolute"])

    def test_counts(self):
        self.assertEqual(self.readout["n_pre_weeks"], 52)
        self.assertEqual(self.readout["n_post_weeks"], 8)
        self.assertEqual(self.readout["n_test_regions"], 20)
        self.assertEqual(self.readout["pair_count"], 20)

    def test_zero_lift_simulation_covers_zero(self):
        readout = simulate_experiment(self.rows, self.matched, lift=0.0)
        self.assertTrue(readout["covers_planted_lift"])


if __name__ == "__main__":
    unittest.main()
