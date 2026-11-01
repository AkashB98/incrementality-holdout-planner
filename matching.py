"""Matched-market selection: pair similar regions, split pairs across arms.

For each region we summarise its pre-period weekly sales with a small
feature vector (mean level, volatility, trend slope from an ordinary
least squares - OLS - fit of sales on week, and the region's average
level relative to the field). Pairs are formed
greedily by an explainable two-ingredient distance - scaled pre-period
level gap plus correlation gap (see ``pair_distance``) - and one member
of each pair is assigned to test and the other to control, alternating
by pair index so the assignment is deterministic and unbiased by
construction.

Quality is reported, not assumed: the pre-period mean difference
between the two arms (absolute and as a share of the overall mean) and
the correlation between the two arms' weekly mean series.
"""

from __future__ import annotations

import math
from typing import Dict, List

FEATURE_NAMES = ["mean", "std", "slope"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ols_slope(weeks: List[float], values: List[float]) -> float:
    """Slope of the least-squares line values ~ weeks."""
    n = len(weeks)
    if n < 2:
        return 0.0
    mw, mv = _mean(weeks), _mean(values)
    denom = sum((w - mw) ** 2 for w in weeks)
    if denom == 0:
        return 0.0
    return sum((w - mw) * (v - mv) for w, v in zip(weeks, values)) / denom


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)
                    * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def region_features(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Per-region pre-period features: mean, std, OLS trend slope."""
    by_region: Dict[str, List[Dict]] = {}
    for row in rows:
        if row.get("period") == "pre":
            by_region.setdefault(row["region_id"], []).append(row)
    features: Dict[str, Dict[str, float]] = {}
    for rid, rrows in sorted(by_region.items()):
        rrows = sorted(rrows, key=lambda r: r["week"])
        weeks = [float(r["week"]) for r in rrows]
        sales = [float(r["sales"]) for r in rrows]
        features[rid] = {
            "mean": round(_mean(sales), 4),
            "std": round(_std(sales), 4),
            "slope": round(ols_slope(weeks, sales), 4),
            "weeks": len(sales),
        }
    return features


def _standardized_means(features):
    vals = [features[r]["mean"] for r in sorted(features)]
    m = _mean(vals)
    sd = _std(vals) or 1.0
    return {r: (features[r]["mean"] - m) / sd for r in features}


def pre_series(rows):
    """Per-region pre-period sales series, ordered by week."""
    by_region = {}
    for row in rows:
        if row.get("period") == "pre":
            by_region.setdefault(row["region_id"], []).append(row)
    return {rid: [float(r["sales"]) for r in sorted(rs, key=lambda r: r["week"])]
            for rid, rs in sorted(by_region.items())}


def pair_distance(zmean_a, zmean_b, corr_ab):
    """Explainable pair distance: scaled level gap + correlation gap.

    Two ingredients only - how far apart the regions' pre-period sales
    *levels* are (in standard-deviation units of the level distribution)
    and how poorly their weekly pre-period series *move together*
    (1 - correlation). A good matched market is close on both.
    """
    return abs(zmean_a - zmean_b) + (1.0 - corr_ab)


def pair_regions(rows):
    """Greedy closest-pair matching on level + correlation.

    Features (mean / std / OLS slope) are still computed and reported
    per region, but the pairing decision uses only the two ingredients
    in ``pair_distance`` so the method stays explainable in one
    sentence. No combinatorial optimisation: repeatedly take the
    closest remaining pair. One member of each pair goes to test, the
    other to control, alternating by pair index (deterministic).
    """
    features = region_features(rows)
    zmean = _standardized_means(features)
    series = pre_series(rows)
    unmatched = sorted(features)
    pairs = []
    while len(unmatched) >= 2:
        best = None
        for i in range(len(unmatched)):
            for j in range(i + 1, len(unmatched)):
                a, b = unmatched[i], unmatched[j]
                d = pair_distance(zmean[a], zmean[b],
                                  correlation(series[a], series[b]))
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        pairs.append([a, b])
        unmatched.remove(a)
        unmatched.remove(b)
    test, control = [], []
    for idx, (a, b) in enumerate(pairs):
        if idx % 2 == 0:
            test.append(a)
            control.append(b)
        else:
            test.append(b)
            control.append(a)
    quality = matching_quality(rows, test, control)
    return {
        "pairs": pairs,
        "pair_count": len(pairs),
        "unmatched": unmatched,
        "assignment": {"test": sorted(test), "control": sorted(control)},
        "features": features,
        "quality": quality,
    }


def weekly_group_means(rows: List[Dict], region_ids: List[str],
                       period: str) -> Dict[int, float]:
    wanted = set(region_ids)
    by_week: Dict[int, List[float]] = {}
    for row in rows:
        if row["region_id"] in wanted and row.get("period") == period:
            by_week.setdefault(int(row["week"]), []).append(float(row["sales"]))
    return {w: _mean(v) for w, v in sorted(by_week.items())}


def matching_quality(rows: List[Dict], test_ids: List[str],
                     control_ids: List[Dict]) -> Dict:
    """Pre-period balance: mean difference + weekly-series correlation."""
    test_means = weekly_group_means(rows, test_ids, "pre")
    ctrl_means = weekly_group_means(rows, control_ids, "pre")
    weeks = sorted(set(test_means) & set(ctrl_means))
    t_series = [test_means[w] for w in weeks]
    c_series = [ctrl_means[w] for w in weeks]
    t_mean, c_mean = _mean(t_series), _mean(c_series)
    overall = _mean(t_series + c_series) or 1.0
    diff = t_mean - c_mean
    return {
        "test_pre_mean": round(t_mean, 4),
        "control_pre_mean": round(c_mean, 4),
        "mean_difference": round(diff, 4),
        "mean_difference_pct": round(diff / overall, 6),
        "abs_mean_difference_pct": round(abs(diff) / overall, 6),
        "weekly_correlation": round(correlation(t_series, c_series), 6),
        "pre_weeks_compared": len(weeks),
    }
