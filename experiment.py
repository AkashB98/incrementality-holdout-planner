"""Experiment simulation + pair-level difference-in-differences readout.

Inference design (the credibility-critical choice): a naive DiD that
treats every region-week as an independent observation understates
standard errors badly, because a region's weekly sales are serially
correlated - the Bertrand / Duflo / Mullainathan critique of DiD
inference. This module therefore collapses the data to ONE number per
matched pair before inferring anything:

  pair_diff[i] = (post_test - pre_test) - (post_control - pre_control)
                 (each term a region's mean over that period)
  estimate     = mean(pair_diff)          over the 20 pairs

Primary inference is *randomisation (permutation) inference* that
respects the design: under the sharp null of no effect, the test /
control labels inside a pair could just as well have been flipped, so
re-flipping them (sign-flipping each pair difference) and recomputing
the estimate builds the null distribution of the estimator itself.
The p-value is the share of permuted estimates at least as extreme as
the observed one, over a deterministic seeded set of permutations.
A normal-approximation confidence interval across the pair differences
(standard error = std(pair_diff) / sqrt(n_pairs), quantiles from
``statistics.NormalDist`` - never a hard-coded 1.96) is reported as
the secondary inference.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

from power import normal_cdf, z_value

DEFAULT_LIFT = 0.05
DEFAULT_POST_WEEKS = 8
DEFAULT_PERMUTATIONS = 2000
PERMUTATION_SEED = 20240521


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def region_period_mean(rows: List[Dict], region_id: str, period: str) -> float:
    vals = [float(r["sales"]) for r in rows
            if r["region_id"] == region_id and r.get("period") == period]
    if not vals:
        raise ValueError(f"no {period} rows for {region_id}")
    return _mean(vals)


def pair_differences(rows: List[Dict], matched: Dict) -> List[float]:
    """One DiD difference per matched pair, in pair order."""
    test_set = set(matched["assignment"]["test"])
    diffs = []
    for a, b in matched["pairs"]:
        test_rid, ctrl_rid = (a, b) if a in test_set else (b, a)
        d = ((region_period_mean(rows, test_rid, "post")
              - region_period_mean(rows, test_rid, "pre"))
             - (region_period_mean(rows, ctrl_rid, "post")
                - region_period_mean(rows, ctrl_rid, "pre")))
        diffs.append(d)
    return diffs


def permutation_p_value(pair_diffs: List[float],
                        n_permutations: int = DEFAULT_PERMUTATIONS,
                        seed: int = PERMUTATION_SEED) -> float:
    """Sign-flip randomisation p-value for mean(pair_diffs) == 0."""
    observed = abs(_mean(pair_diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    n = len(pair_diffs)
    for _ in range(n_permutations):
        total = 0.0
        for d in pair_diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed:
            extreme += 1
    return extreme / n_permutations


def apply_lift(rows: List[Dict], test_ids, lift: float) -> List[Dict]:
    """Copy of ``rows`` with post-period test sales scaled by 1+lift."""
    wanted = set(test_ids)
    out = []
    for row in rows:
        new = dict(row)
        if row["region_id"] in wanted and row.get("period") == "post":
            new["sales"] = round(float(row["sales"]) * (1.0 + lift), 2)
        out.append(new)
    return out


def estimate_did(rows: List[Dict], matched: Dict,
                 n_permutations: int = DEFAULT_PERMUTATIONS) -> Dict:
    """Pair-level DiD readout for a (possibly lift-planted) dataset."""
    assignment = matched["assignment"]
    diffs = pair_differences(rows, matched)
    if not diffs:
        raise ValueError("need at least one matched pair for a readout")
    estimate = _mean(diffs)
    pair_sd = _std(diffs)
    se = pair_sd / math.sqrt(len(diffs)) if diffs else 0.0
    ctrl_means = [region_period_mean(rows, rid, "pre")
                  for rid in assignment["control"]]
    baseline = _mean(ctrl_means) or 1.0
    z_crit = z_value(0.975)
    half = z_crit * se
    ci_low, ci_high = estimate - half, estimate + half
    z = estimate / se if se else 0.0
    p_perm = permutation_p_value(diffs, n_permutations=n_permutations)
    n_post = len({r["week"] for r in rows if r.get("period") == "post"})
    n_pre = len({r["week"] for r in rows if r.get("period") == "pre"})
    return {
        "did_absolute": round(estimate, 4),
        "did_lift_pct": round(estimate / baseline, 6),
        "pair_count": len(diffs),
        "pair_difference_sd": round(pair_sd, 4),
        "standard_error": round(se, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "ci_95_pct": [round(ci_low / baseline, 6),
                      round(ci_high / baseline, 6)],
        "ci_method": "normal approximation across pair differences",
        "p_value": round(p_perm, 6),
        "p_value_method": "randomisation (sign-flip) inference within pairs",
        "n_permutations": n_permutations,
        "normal_p_value": round(2.0 * (1.0 - normal_cdf(abs(z))), 6) if se else 1.0,
        "z_statistic": round(z, 4),
        "control_pre_mean": round(baseline, 4),
        "n_pre_weeks": n_pre,
        "n_post_weeks": n_post,
        "n_test_regions": len(assignment["test"]),
        "n_control_regions": len(assignment["control"]),
        "significant_at_0_05": bool(p_perm < 0.05),
    }


def simulate_experiment(rows: List[Dict], matched: Dict,
                        lift: float = DEFAULT_LIFT,
                        n_permutations: int = DEFAULT_PERMUTATIONS) -> Dict:
    """Plant ``lift`` in test post weeks, estimate it back, score it."""
    lifted = apply_lift(rows, matched["assignment"]["test"], lift)
    readout = estimate_did(lifted, matched, n_permutations=n_permutations)
    lo, hi = readout["ci_95_pct"]
    readout["planted_lift_pct"] = lift
    readout["covers_planted_lift"] = bool(lo <= lift <= hi)
    readout["estimate_error_pct"] = round(readout["did_lift_pct"] - lift, 6)
    return readout
