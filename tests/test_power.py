"""Power math: hand-computed cases against NormalDist known values."""

import math
import os
import sys
import unittest
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset
from matching import pair_regions
from power import (did_standard_error, mde, mde_from_se, normal_cdf,
                   normal_cdf_erf, pair_mde, pair_standard_error,
                   placebo_pair_differences, pooled_placebo_sd, power_for_lift,
                   power_table, predicted_pair_sd,
                   required_post_weeks_for_lift, required_weeks,
                   weekly_diff_std, z_value)

ND = NormalDist()


class TestNormalFunctions(unittest.TestCase):
    def test_z_975_known(self):
        self.assertAlmostEqual(z_value(0.975), 1.9599639845400536, places=12)

    def test_z_80_known(self):
        self.assertAlmostEqual(z_value(0.80), 0.8416212335729144, places=12)

    def test_z_median(self):
        self.assertAlmostEqual(z_value(0.5), 0.0, places=12)

    def test_cdf_known(self):
        self.assertAlmostEqual(normal_cdf(1.9599639845400536), 0.975, places=9)
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=12)

    def test_cdf_matches_erf(self):
        for x in (-2.5, -1.0, 0.0, 0.7, 1.96, 3.1):
            self.assertAlmostEqual(normal_cdf(x), normal_cdf_erf(x), places=12)

    def test_cdf_matches_erf_formula(self):
        self.assertAlmostEqual(normal_cdf_erf(1.0),
                               0.5 * (1 + math.erf(1 / math.sqrt(2))), places=15)

    def test_z_rejects_bounds(self):
        for p in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                z_value(p)


class TestMdeFormulas(unittest.TestCase):
    def test_se_equal_periods(self):
        # sigma * sqrt(2/n): 100 * sqrt(2/8) = 50
        self.assertAlmostEqual(did_standard_error(100.0, 8), 50.0)

    def test_se_explicit_pre(self):
        # 100 * sqrt(1/8 + 1/52)
        self.assertAlmostEqual(did_standard_error(100.0, 8, 52),
                               100 * math.sqrt(1 / 8 + 1 / 52))

    def test_se_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            did_standard_error(-1.0, 8)
        with self.assertRaises(ValueError):
            did_standard_error(100.0, 0)

    def test_mde_hand_computed(self):
        # (z.975 + z.80) * 100 * sqrt(2/8) = 2.801585218112968 * 50
        expected = (ND.inv_cdf(0.975) + ND.inv_cdf(0.80)) * 50.0
        self.assertAlmostEqual(mde(8, 100.0), expected, places=9)
        self.assertAlmostEqual(mde(8, 100.0), 140.0792609056484, places=6)

    def test_mde_scales_with_sigma(self):
        self.assertAlmostEqual(mde(8, 200.0), 2 * mde(8, 100.0))

    def test_mde_shrinks_with_more_weeks(self):
        self.assertLess(mde(16, 100.0), mde(8, 100.0))

    def test_mde_zero_sigma(self):
        self.assertEqual(mde(8, 0.0), 0.0)

    def test_required_weeks_hand_computed(self):
        # n = ceil(2 * ((z_a+z_b)*sigma / lift)^2); lift == mde(8) -> n == 8
        lift = mde(8, 100.0)
        self.assertEqual(required_weeks(lift, 100.0), 8)

    def test_required_weeks_bigger_lift_needs_fewer(self):
        self.assertLess(required_weeks(500.0, 100.0),
                        required_weeks(100.0, 100.0))

    def test_required_weeks_zero_sigma(self):
        self.assertEqual(required_weeks(10.0, 0.0), 1)

    def test_required_weeks_rejects_nonpositive_lift(self):
        with self.assertRaises(ValueError):
            required_weeks(0.0, 100.0)

    def test_power_at_mde_is_target_power(self):
        # By construction, power at the MDE equals the target power (0.80)
        # when pre and post lengths match the MDE formula's assumption.
        lift = mde(8, 100.0)
        self.assertAlmostEqual(power_for_lift(lift, 8, 100.0), 0.80, places=6)

    def test_power_zero_lift_is_alpha(self):
        self.assertAlmostEqual(power_for_lift(0.0, 8, 100.0), 0.025, places=6)

    def test_power_grows_with_lift(self):
        self.assertLess(power_for_lift(50.0, 8, 100.0),
                        power_for_lift(200.0, 8, 100.0))

    def test_power_zero_sigma(self):
        self.assertEqual(power_for_lift(5.0, 8, 0.0), 1.0)


class TestPowerOnSeededData(unittest.TestCase):
    def setUp(self):
        self.rows = generate_dataset()
        self.assignment = pair_regions(self.rows)["assignment"]

    def test_weekly_diff_std_positive_and_pinned(self):
        sigma = weekly_diff_std(self.rows, self.assignment["test"],
                                self.assignment["control"])
        self.assertAlmostEqual(sigma, 176.7402, places=2)
        self.assertGreater(sigma, 0)

    def test_power_table_shape(self):
        from matching import pair_regions as _pr
        table = power_table(self.rows, _pr(self.rows))
        self.assertEqual([t["lift_pct"] for t in table], [0.02, 0.05, 0.10])
        # Pair-level sizing is more conservative than the old weekly
        # sizing: a 2% lift is below the pair-level MDE at 8 weeks.
        self.assertFalse(table[0]["detectable"])
        self.assertTrue(table[1]["detectable"])
        self.assertEqual(table[0]["required_post_weeks"], 41)  # >40: a 2% lift is a long test at pair level
        self.assertAlmostEqual(table[0]["mde_absolute"], 301.4982, places=3)


if __name__ == "__main__":
    unittest.main()


class TestPairLevelPlanner(unittest.TestCase):
    def setUp(self):
        self.rows = generate_dataset()
        self.matched = pair_regions(self.rows)

    def test_placebo_differences_count(self):
        diffs = placebo_pair_differences(self.rows, self.matched, 8)
        self.assertEqual(len(diffs), 20)

    def test_placebo_rejects_bad_weeks(self):
        with self.assertRaises(ValueError):
            placebo_pair_differences(self.rows, self.matched, 52)

    def test_pair_se_formula(self):
        self.assertAlmostEqual(pair_standard_error(100.0, 25), 20.0)
        with self.assertRaises(ValueError):
            pair_standard_error(100.0, 0)

    def test_mde_from_se_hand(self):
        expected = (ND.inv_cdf(0.975) + ND.inv_cdf(0.80)) * 10.0
        self.assertAlmostEqual(mde_from_se(10.0), expected)

    def test_mde_falls_with_more_weeks(self):
        self.assertLess(pair_mde(self.rows, self.matched, post_weeks=16),
                        pair_mde(self.rows, self.matched, post_weeks=4))

    def test_mde_falls_with_more_pairs(self):
        self.assertLess(pair_mde(self.rows, self.matched, post_weeks=8,
                                 n_pairs=40),
                        pair_mde(self.rows, self.matched, post_weeks=8,
                                 n_pairs=10))

    def test_required_weeks_falls_with_bigger_lift(self):
        small = required_post_weeks_for_lift(self.rows, self.matched, 150.0)
        big = required_post_weeks_for_lift(self.rows, self.matched, 600.0)
        self.assertLess(big, small)


    def test_predicted_pair_sd_positive(self):
        sd = predicted_pair_sd(self.rows, self.matched)
        self.assertGreater(sd, 0)
        # Model-based prediction should land near the realised readout
        # pair sd on the seeded dataset (456.31) - same ballpark, not
        # the placebo split's systematic under-estimate.
        self.assertGreater(sd, 400.0)
        self.assertLess(sd, 600.0)

    def test_pooled_placebo_sd_positive(self):
        self.assertGreater(pooled_placebo_sd(self.rows, self.matched), 0)
