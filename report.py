"""Full experiment plan report: matching quality + power table + readout.

Assembles the three questions a geo-holdout plan must answer before
anyone spends media money behind it:

  1. Are the arms comparable?     -> ``matching`` (pairs + quality)
  2. What could this design see?  -> ``power`` (pair-level MDE / power
     per lift, sized from placebo pair differences so the planner and
     the readout speak the same units)
  3. Does the estimator work?     -> ``readout`` (a seeded simulation
     plants a known lift and the pair-level DiD readout must recover
     it inside its own 95% confidence interval)
"""

from __future__ import annotations

from typing import Dict, List

from experiment import DEFAULT_LIFT, simulate_experiment
from matching import pair_regions
from power import (pair_mde, pooled_placebo_sd, power_table,
                   predicted_pair_sd, weekly_diff_std)

DEFAULT_LIFTS = (0.02, 0.05, 0.10)


def build_report(rows: List[Dict], lift: float = DEFAULT_LIFT,
                 post_weeks: int = 8, pre_weeks: int = 52) -> Dict:
    matched = pair_regions(rows)
    assignment = matched["assignment"]
    sigma = weekly_diff_std(rows, assignment["test"], assignment["control"])
    placebo_sd = pooled_placebo_sd(rows, matched)
    table = power_table(rows, matched, lifts=DEFAULT_LIFTS,
                        post_weeks=post_weeks, pre_weeks=pre_weeks)
    readout = simulate_experiment(rows, matched, lift=lift)
    regions = sorted({r["region_id"] for r in rows})
    weeks = sorted({r["week"] for r in rows})
    return {
        "dataset": {
            "rows": len(rows),
            "regions": len(regions),
            "weeks": len(weeks),
            "pre_weeks": len({r["week"] for r in rows if r["period"] == "pre"}),
        },
        "matching": {
            "pair_count": matched["pair_count"],
            "pairs": matched["pairs"],
            "assignment": assignment,
            "quality": matched["quality"],
        },
        "power": {
            "weekly_diff_std": round(sigma, 4),
            "placebo_pair_difference_sd": round(placebo_sd, 4),
            "predicted_pair_difference_sd": round(predicted_pair_sd(rows, matched, post_weeks=post_weeks), 4),
            "pair_mde_absolute": round(pair_mde(rows, matched,
                                                post_weeks=post_weeks), 4),
            "post_weeks": post_weeks,
            "pre_weeks": pre_weeks,
            "table": table,
        },
        "readout": readout,
    }
