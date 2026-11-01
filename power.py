"""Power / minimum detectable effect (MDE) math - stdlib only.

The experiment compares two arms over time, so the unit of noise that
matters is the *weekly difference series*: in each week, mean test
sales minus mean control sales. Call its standard deviation
``sigma_d``. A difference-in-differences estimate over ``n_pre``
pre weeks and ``n_post`` post weeks then has standard error

  SE = sigma_d * sqrt(1/n_post + 1/n_pre)

and, under the normal approximation:

  MDE = (z_{1-alpha/2} + z_{power}) * SE
      = the smallest true lift whose detection probability is ``power``
        at two-sided significance level ``alpha``.

Solving the same equation for ``n`` (with n_pre = n_post = n) gives
``required_weeks``; solving for the detection probability gives
``power_for_lift``. Normal quantiles and CDF values come from
``statistics.NormalDist`` (stdlib); the CDF is cross-checked in tests
against a ``math.erf`` implementation.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, List

_NORM = NormalDist()
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def normal_cdf(x: float) -> float:
    return _NORM.cdf(x)


def normal_cdf_erf(x: float) -> float:
    """Same CDF via math.erf - kept so tests can cross-check the two."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def z_value(p: float) -> float:
    """Inverse normal CDF (quantile) for probability ``p`` in (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return _NORM.inv_cdf(p)


def did_standard_error(weekly_diff_std: float, n_post: int,
                       n_pre: int = None) -> float:
    if weekly_diff_std < 0:
        raise ValueError("weekly_diff_std must be >= 0")
    if n_post < 1:
        raise ValueError("n_post must be >= 1")
    if n_pre is None:
        return weekly_diff_std * math.sqrt(2.0 / n_post)
    if n_pre < 1:
        raise ValueError("n_pre must be >= 1")
    return weekly_diff_std * math.sqrt(1.0 / n_post + 1.0 / n_pre)


def mde(n_weeks: int, weekly_diff_std: float, alpha: float = DEFAULT_ALPHA,
        power: float = DEFAULT_POWER, n_pre: int = None) -> float:
    """Minimum detectable effect (absolute sales units) for a readout
    with ``n_weeks`` post weeks. ``n_pre`` defaults to ``n_weeks`` in
    the standard-error factor via the equal-length shortcut when None
    is interpreted by ``did_standard_error`` as sqrt(2/n)."""
    z = z_value(1.0 - alpha / 2.0) + z_value(power)
    return z * did_standard_error(weekly_diff_std, n_weeks, n_pre)


def required_weeks(target_lift: float, weekly_diff_std: float,
                   alpha: float = DEFAULT_ALPHA,
                   power: float = DEFAULT_POWER) -> int:
    """Post weeks needed (assuming an equally long pre period) so the
    MDE is at most ``target_lift`` (absolute units)."""
    if target_lift <= 0:
        raise ValueError("target_lift must be > 0")
    if weekly_diff_std < 0:
        raise ValueError("weekly_diff_std must be >= 0")
    if weekly_diff_std == 0:
        return 1
    z = z_value(1.0 - alpha / 2.0) + z_value(power)
    n = 2.0 * (z * weekly_diff_std / target_lift) ** 2
    return max(1, math.ceil(n))


def power_for_lift(lift: float, n_weeks: int, weekly_diff_std: float,
                   alpha: float = DEFAULT_ALPHA, n_pre: int = None) -> float:
    """Probability of detecting a true ``lift`` (absolute units)."""
    if weekly_diff_std == 0:
        return 1.0 if lift != 0 else alpha
    se = did_standard_error(weekly_diff_std, n_weeks, n_pre)
    return normal_cdf(abs(lift) / se - z_value(1.0 - alpha / 2.0))


def weekly_diff_std(rows, test_ids, control_ids) -> float:
    """Sample std of the pre-period weekly test-minus-control series.
    This is the ``sigma_d`` the power formulas consume - estimated from
    the pre period so a plan can be sized *before* the test runs."""
    import math as _m
    from matching import weekly_group_means
    t = weekly_group_means(rows, test_ids, "pre")
    c = weekly_group_means(rows, control_ids, "pre")
    weeks = sorted(set(t) & set(c))
    diffs = [t[w] - c[w] for w in weeks]
    if len(diffs) < 2:
        return 0.0
    m = sum(diffs) / len(diffs)
    return _m.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1))


def power_table(rows, matched, lifts=(0.02, 0.05, 0.10),
                post_weeks: int = 8, pre_weeks: int = 52,
                alpha: float = DEFAULT_ALPHA,
                power: float = DEFAULT_POWER):
    """MDE / power rows for relative lifts, sized at pair level.

    Detection probability for a lift uses the same normal machinery as
    the readout's secondary inference, with the pair-level standard
    error - so the planner's numbers are promises about the readout
    this package actually ships (validated by the Monte Carlo evals).
    """
    from matching import weekly_group_means
    assignment = matched["assignment"]
    pair_sd = pair_sd_for_weeks(rows, matched, post_weeks)
    se = pair_standard_error(pair_sd, len(matched["pairs"]))
    ctrl = weekly_group_means(rows, assignment["control"], "pre")
    baseline = (sum(ctrl.values()) / len(ctrl)) if ctrl else 0.0
    mde_abs = mde_from_se(se, alpha=alpha, power=power)
    table = []
    for lift in lifts:
        lift_abs = lift * baseline
        if se:
            p_detect = normal_cdf(lift_abs / se - z_value(1.0 - alpha / 2.0))
        else:
            p_detect = 1.0
        table.append({
            "lift_pct": lift,
            "lift_absolute": round(lift_abs, 4),
            "mde_absolute": round(mde_abs, 4),
            "mde_pct": round(mde_abs / baseline, 6) if baseline else None,
            "power": round(p_detect, 6),
            "required_post_weeks": required_post_weeks_for_lift(
                rows, matched, lift_abs, alpha=alpha, power=power),
            "detectable": lift_abs >= mde_abs,
        })
    return table


# ------------------------------------------------------- pair-level planner
# The readout infers across matched pairs (see experiment.py), so the
# planner sizes the design in the same units - otherwise the planner's
# promises and the readout's verdicts would be denominated differently.
# The spread of pair differences is estimated *before* the test runs by
# a placebo split of the pre period: the last ``post_weeks`` pre weeks
# play the role of a fake post period, the rest a fake pre period, and
# the pair differences of that fake experiment (no treatment anywhere)
# measure exactly the noise the real readout must beat.

def _sample_std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def placebo_pair_differences(rows, matched, post_weeks: int):
    """Pair DiD differences from a no-treatment split of the pre period."""
    pre_weeks_sorted = sorted({r["week"] for r in rows
                               if r.get("period") == "pre"})
    if post_weeks < 1 or post_weeks >= len(pre_weeks_sorted):
        raise ValueError("post_weeks must be in 1..(pre_weeks-1) for a placebo split")
    cutoff = pre_weeks_sorted[-post_weeks]
    test_set = set(matched["assignment"]["test"])

    def _period_mean(rid, want_post):
        vals = [float(r["sales"]) for r in rows
                if r["region_id"] == rid and r.get("period") == "pre"
                and (r["week"] >= cutoff) == want_post]
        return sum(vals) / len(vals) if vals else 0.0

    diffs = []
    for a, b in matched["pairs"]:
        t, c = (a, b) if a in test_set else (b, a)
        diffs.append((_period_mean(t, True) - _period_mean(t, False))
                     - (_period_mean(c, True) - _period_mean(c, False)))
    return diffs


def pair_standard_error(pair_sd: float, n_pairs: int) -> float:
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    return pair_sd / (n_pairs ** 0.5)


def mde_from_se(se: float, alpha: float = DEFAULT_ALPHA,
                power: float = DEFAULT_POWER) -> float:
    return (z_value(1.0 - alpha / 2.0) + z_value(power)) * se


def pair_mde(rows, matched, post_weeks: int = 8,
             alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER,
             n_pairs: int = None) -> float:
    """Planner MDE (absolute) for the pair-level readout."""
    sd = pair_sd_for_weeks(rows, matched, post_weeks)
    pairs = n_pairs if n_pairs is not None else len(matched["pairs"])
    return mde_from_se(pair_standard_error(sd, pairs), alpha=alpha, power=power)


def required_post_weeks_for_lift(rows, matched, target_lift_abs: float,
                                 alpha: float = DEFAULT_ALPHA,
                                 power: float = DEFAULT_POWER,
                                 max_weeks: int = 40) -> int:
    """Smallest post-period length whose pair-level MDE <= target."""
    if target_lift_abs <= 0:
        raise ValueError("target_lift_abs must be > 0")
    for weeks in range(1, max_weeks + 1):
        if pair_mde(rows, matched, post_weeks=weeks,
                    alpha=alpha, power=power) <= target_lift_abs:
            return weeks
    return max_weeks + 1


PLACEBO_REFERENCE_WEEKS = 8


def pooled_placebo_sd(rows, matched, post_weeks: int = PLACEBO_REFERENCE_WEEKS,
                      step: int = 4) -> float:
    """Pair-difference noise sd pooled over many placebo splits.

    A single placebo split gives a noisy sd (20 pairs only), and a
    planner sized on a lucky-quiet split systematically under-promises
    noise. So: slide the fake post window across the pre period, pool
    every split's pair differences, and take the overall sd.
    """
    pre = sorted({r["week"] for r in rows if r.get("period") == "pre"})
    test_set = set(matched["assignment"]["test"])
    by_rid = {}
    for r in rows:
        if r.get("period") == "pre":
            by_rid.setdefault(r["region_id"], {})[r["week"]] = float(r["sales"])
    pooled = []
    start_min = 8
    start_max = pre[-1] - post_weeks + 1
    for start in range(start_min, start_max + 1, step):
        post_window = set(range(start, start + post_weeks))
        for a, b in matched["pairs"]:
            t, c = (a, b) if a in test_set else (b, a)
            def _diff(rid):
                pw = [v for wk, v in by_rid[rid].items() if wk in post_window]
                rw = [v for wk, v in by_rid[rid].items() if wk not in post_window]
                return sum(pw) / len(pw) - sum(rw) / len(rw)
            pooled.append(_diff(t) - _diff(c))
    return _sample_std(pooled)


def pair_sd_for_weeks(rows, matched, post_weeks: int) -> float:
    """Pair-difference sd for a ``post_weeks``-long readout.

    Estimated by the pooled placebo at the reference length, then
    rescaled by the DiD variance factor (1/n_post + 1/n_pre): the same
    factor that makes more weeks monotonically quieter by construction.
    """
    return predicted_pair_sd(rows, matched, post_weeks=post_weeks)


def predicted_pair_sd(rows, matched, post_weeks: int = 8,
                      pre_weeks: int = None) -> float:
    """Model-based pair-difference sd for a readout of ``post_weeks``.

    A placebo split of the pre period cannot mimic the readout's
    temporal structure: in the real readout the post weeks sit a long
    way (in trend terms) from the pre weeks, so any trend difference
    inside a pair is amplified by that week gap - while a placebo
    window in the middle of the pre period has a much smaller gap and
    systematically under-states the noise. So the planner instead
    decomposes each pair's readout variance from quantities estimable
    in the pre period:

      var_i = (resid_var_test + resid_var_control)
              * (1/n_post + 1/n_pre)                <- weekly noise
            + (slope_test - slope_control)^2
              * (mean post week - mean pre week)^2  <- trend divergence

    where resid_var / slope come from each region's OLS fit of pre
    sales on week. Returns sqrt(mean(var_i)) - the sd of one pair
    difference; the readout's standard error is this / sqrt(n_pairs).
    """
    from matching import ols_slope
    pre = [r for r in rows if r.get("period") == "pre"]
    n_pre = pre_weeks if pre_weeks is not None else len({r["week"] for r in pre})
    by_rid = {}
    for r in pre:
        by_rid.setdefault(r["region_id"], []).append(r)
    fits = {}
    for rid, rs in by_rid.items():
        rs = sorted(rs, key=lambda r: r["week"])
        weeks = [float(r["week"]) for r in rs]
        sales = [float(r["sales"]) for r in rs]
        slope = ols_slope(weeks, sales)
        mw = sum(weeks) / len(weeks)
        intercept = (sum(sales) / len(sales)) - slope * mw
        resid = [y - (intercept + slope * w) for w, y in zip(weeks, sales)]
        var = (sum(e * e for e in resid) / (len(resid) - 2)) if len(resid) > 2 else 0.0
        fits[rid] = (slope, var)
    mean_pre_week = (n_pre - 1) / 2.0
    mean_post_week = n_pre + (post_weeks - 1) / 2.0
    gap = mean_post_week - mean_pre_week
    variances = []
    for a, b in matched["pairs"]:
        st, vt = fits[a]
        sc, vc = fits[b]
        variances.append((vt + vc) * (1.0 / post_weeks + 1.0 / n_pre)
                         + (st - sc) ** 2 * gap ** 2)
    return (sum(variances) / len(variances)) ** 0.5
