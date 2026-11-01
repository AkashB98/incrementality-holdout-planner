"""Generator: determinism, shape, process properties, serialisers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import (REGION_NAMES, generate_dataset, generate_pre_period,
                       load_rows, region_ids, rows_to_csv, rows_to_jsonl,
                       write_rows)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.rows = generate_dataset()

    def test_default_shape(self):
        self.assertEqual(len(self.rows), 40 * 60)
        self.assertEqual(len({r["region_id"] for r in self.rows}), 40)
        self.assertEqual(len({r["week"] for r in self.rows}), 60)

    def test_deterministic_same_seed(self):
        self.assertEqual(generate_dataset(), generate_dataset())

    def test_different_seed_differs(self):
        self.assertNotEqual(generate_dataset(seed=7), self.rows)

    def test_region_ids_and_names(self):
        self.assertEqual(region_ids(3), ["region_01", "region_02", "region_03"])
        self.assertEqual(len(REGION_NAMES), 40)
        names = {r["region_id"]: r["region_name"] for r in self.rows}
        self.assertEqual(names["region_01"], REGION_NAMES[0])

    def test_period_split(self):
        pre = [r for r in self.rows if r["period"] == "pre"]
        post = [r for r in self.rows if r["period"] == "post"]
        self.assertEqual((len(pre), len(post)), (40 * 52, 40 * 8))
        self.assertTrue(all(r["week"] < 52 for r in pre))
        self.assertTrue(all(r["week"] >= 52 for r in post))

    def test_sales_positive(self):
        self.assertTrue(all(r["sales"] > 0 for r in self.rows))

    def test_custom_sizes(self):
        rows = generate_dataset(pre_weeks=10, post_weeks=3, n_regions=6)
        self.assertEqual(len(rows), 6 * 13)

    def test_negative_weeks_rejected(self):
        with self.assertRaises(ValueError):
            generate_dataset(pre_weeks=-1)

    def test_too_many_regions_rejected(self):
        with self.assertRaises(ValueError):
            generate_dataset(n_regions=41)

    def test_pre_period_helper_matches(self):
        pre = generate_pre_period()
        expected = [r for r in generate_dataset(post_weeks=0)
                    if r["period"] == "pre"]
        self.assertEqual(pre, expected)
        self.assertEqual(len(pre), 40 * 52)

    def test_regions_correlate_via_shared_shocks(self):
        # Two regions' pre series should correlate positively because of
        # the shared seasonality/shock components.
        from matching import correlation
        def series(rid):
            return [r["sales"] for r in sorted(
                (x for x in self.rows
                 if x["region_id"] == rid and x["period"] == "pre"),
                key=lambda x: x["week"])]
        self.assertGreater(correlation(series("region_01"),
                                       series("region_02")), 0.0)

    def test_jsonl_round_trip(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "d.jsonl")
        write_rows(self.rows, path)
        self.assertEqual(load_rows(path), self.rows)

    def test_csv_round_trip(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "d.csv")
        write_rows(self.rows, path)
        self.assertEqual(load_rows(path), self.rows)

    def test_jsonl_text_deterministic(self):
        self.assertEqual(rows_to_jsonl(self.rows),
                         rows_to_jsonl(generate_dataset()))

    def test_csv_header(self):
        self.assertTrue(rows_to_csv(self.rows).startswith(
            "region_id,region_name,week,period,sales"))


if __name__ == "__main__":
    unittest.main()
