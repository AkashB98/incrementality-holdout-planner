"""Shared row builders for the test suite (not a test module)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def row(region_id="region_01", week=0, period="pre", sales=10000.0,
        region_name="Aurora Vale"):
    return {"region_id": region_id, "region_name": region_name,
            "week": week, "period": period, "sales": sales}


def grid(region_sales, pre_weeks=4, post_weeks=2):
    """Rows for {region_id: constant weekly sales} over pre+post weeks."""
    rows = []
    for rid, sales in region_sales.items():
        for wk in range(pre_weeks + post_weeks):
            rows.append(row(rid, wk, "pre" if wk < pre_weeks else "post", sales))
    return rows
