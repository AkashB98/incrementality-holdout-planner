"""Golden evals for incrementality-holdout-planner.

SYNTHETIC DATA ONLY (fictional Helios Home dataset). Every eval is
deterministic: run_evals.py writes eval_report.json and the file must
be byte-identical across runs. No timestamps, ports, or temp paths
enter the report - only names, booleans, pinned counts and hashes.

E1-E13 pin the seeded pipeline. E14-E17 are Monte Carlo calibration
evals: across 100 fresh seeds each, they check that the planner's
promises hold up in simulation - the A/A false-positive rate, the
95% CI's coverage, detection at the planner's own MDE, and MDE
monotonicity. Counts are pinned exactly; bands are the binomial
tolerances the counts must sit inside.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from experiment import simulate_experiment  # noqa: E402
from generator import generate_dataset, load_rows, rows_to_jsonl  # noqa: E402
from matching import correlation, ols_slope, pair_regions, weekly_group_means  # noqa: E402
from power import (mde, normal_cdf, pair_mde, power_for_lift,  # noqa: E402
                   required_post_weeks_for_lift, required_weeks, z_value)
from report import build_report  # noqa: E402
from validation import validate_dataset, validate_row  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(ROOT, "fixtures", "sample_regions.jsonl")
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: incrementality-holdout-planner ==")
    rows = generate_dataset()
    matched = pair_regions(rows)
    assignment = matched["assignment"]

    # E1: generator determinism (bytes pinned)
    text1 = rows_to_jsonl(generate_dataset())
    text2 = rows_to_jsonl(generate_dataset())
    md5 = hashlib.md5(text1.encode()).hexdigest()
    check("E1 generator determinism (seed=42, 40 regions x 60 weeks)",
          text1 == text2 and len(rows) == 2400
          and md5 == "4f486f46310642dd142afac60a181a66",
          f"jsonl_md5={md5}")

    # E2: matched pairs pinned (count + first pair + assignment sizes)
    check("E2 matched pairs pinned (20 pairs, first pair region_22/region_27)",
          matched["pair_count"] == 20
          and matched["pairs"][0] == ["region_22", "region_27"]
          and len(assignment["test"]) == 20
          and len(assignment["control"]) == 20)

    # E3: matching quality pinned (imbalance 0.36%, correlation 0.853)
    q = matched["quality"]
    check("E3 matching quality pinned (imbalance 0.003595, corr 0.852906)",
          q["abs_mean_difference_pct"] == 0.003595
          and q["weekly_correlation"] == 0.852906
          and q["test_pre_mean"] == 11074.9677,
          f"imbalance={q['abs_mean_difference_pct']} corr={q['weekly_correlation']}")

    # E4: OLS slope + correlation hand-computed
    check("E4 OLS slope / correlation hand-computed (2.0, 1.0, -1.0)",
          ols_slope([0, 1, 2, 3], [1.0, 3.0, 5.0, 7.0]) == 2.0
          and correlation([1, 2, 3], [2, 4, 6]) == 1.0
          and correlation([1, 2, 3], [6, 4, 2]) == -1.0)

    # E5: normal quantiles / CDF pinned to known values
    check("E5 normal math pinned (z.975, z.80, Phi(0)=0.5)",
          abs(z_value(0.975) - 1.9599639845400536) < 1e-12
          and abs(z_value(0.80) - 0.8416212335729144) < 1e-12
          and normal_cdf(0.0) == 0.5)

    # E6: MDE hand-computed pinned (weekly form, exact algebra)
    check("E6 MDE hand-computed (sigma=100, n=8 -> 140.0793)",
          abs(mde(8, 100.0) - 140.0792609056484) < 1e-6
          and required_weeks(mde(8, 100.0), 100.0) == 8
          and abs(power_for_lift(mde(8, 100.0), 8, 100.0) - 0.80) < 1e-6,
          f"mde={mde(8, 100.0):.4f}")

    # E7: seeded pair-level power table pinned
    report = build_report(rows)
    table = report["power"]["table"]
    check("E7 power table pinned (pair MDE $301.50 = 2.71%, sd 481.28)",
          report["power"]["predicted_pair_difference_sd"] == 481.2779
          and report["power"]["pair_mde_absolute"] == 301.4982
          and table[0]["power"] == 0.542077
          and table[0]["required_post_weeks"] == 41
          and table[1]["power"] == 0.999323
          and table[1]["required_post_weeks"] == 2,
          f"pair_mde={report['power']['pair_mde_absolute']}")

    # E8: pair-level DiD recovers the planted lift, estimate pinned
    readout = simulate_experiment(rows, matched, lift=0.05)
    check("E8 DiD pinned (estimate 4.3661%, perm p=0.001, CI covers 5%)",
          readout["did_lift_pct"] == 0.043661
          and readout["did_absolute"] == 485.2815
          and readout["p_value"] == 0.001
          and readout["covers_planted_lift"]
          and readout["ci_95_pct"] == [0.025668, 0.061653],
          f"estimate={readout['did_lift_pct']}")

    # E9: no-lift simulation is null-ish (pre difference differenced out)
    null_readout = simulate_experiment(rows, matched, lift=0.0)
    check("E9 zero-lift simulation covers zero",
          null_readout["covers_planted_lift"]
          and null_readout["planted_lift_pct"] == 0.0)

    # E10: validation flags planted defects, generated book clean
    bad = dict(rows[0], sales=-5.0, period="during")
    planted = validate_row(bad)
    check("E10 validation flags planted defects, generated book clean",
          len(planted) == 2 and validate_dataset(rows) == [],
          f"planted={planted}")

    # E11: CLI check exit codes 0 / 1 / 2
    import tempfile
    tmp = tempfile.mkdtemp(prefix="holdout-evals-")
    seeded_path = os.path.join(tmp, "seeded.jsonl")
    with open(seeded_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(rows))
    codes = (_run_cli(["check", "--input", seeded_path]),
             _run_cli(["check", "--input", seeded_path,
                       "--max-imbalance", "0.001"]),
             _run_cli(["check", "--input", "/tmp/no-such-holdout-xyz.jsonl"]))
    check("E11 CLI check exit codes 0/1/2",
          codes == (EXIT_OK, EXIT_FAIL, EXIT_ERROR), f"codes={codes}")

    # E12: serve round-trip matches local computation
    reset_state(rows)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/readout", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            served_readout = json.loads(resp.read())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/report",
                                    timeout=10) as resp:
            served_report = json.loads(resp.read())
        serve_ok = served_readout == readout and served_report == report
    finally:
        server.shutdown()
        server.server_close()
        reset_state()
    check("E12 serve round-trip matches local readout + report", serve_ok)

    # E13: bundled fixture pinned (2400 rows, clean, totals pinned)
    fixture = load_rows(FIXTURE)
    total = round(sum(r["sales"] for r in fixture), 2)
    check("E13 fixture pinned (2400 rows, total sales 26847442.5, clean)",
          len(fixture) == 2400 and total == 26847442.5
          and validate_dataset(fixture) == []
          and build_report(fixture) == report,
          f"total_sales={total}")

    # ---- Monte Carlo calibration evals (100 fresh seeds each) ----
    print("  .. Monte Carlo calibration evals (100 seeds each) ..")

    # E14: A/A null test - no lift planted; the permutation test at
    # alpha=0.05 must fire rarely. Nominal rate 5/100; the permutation
    # p-value is discrete so the test is conservative. Pinned: 2.
    fp = 0
    for seed in range(1000, 1100):
        r = generate_dataset(seed=seed)
        out = simulate_experiment(r, pair_regions(r), lift=0.0,
                                  n_permutations=500)
        fp += out["p_value"] < 0.05
    check("E14 A/A null test (false positives pinned 2/100, band 0-10)",
          fp == 2 and 0 <= fp <= 10, f"false_positives={fp}/100")

    # E15: CI coverage - planted +5%, the 95% pair-level CI must cover
    # it about 95% of the time. Pinned: 91 (the normal approximation
    # across 20 pairs undercovers slightly - documented in README).
    covered = 0
    for seed in range(2000, 2100):
        r = generate_dataset(seed=seed)
        out = simulate_experiment(r, pair_regions(r), lift=0.05,
                                  n_permutations=100)
        covered += out["covers_planted_lift"]
    check("E15 CI coverage (pinned 91/100, band 84-98)",
          covered == 91 and 84 <= covered <= 98, f"covered={covered}/100")

    # E16: planner-matches-readout (headline) - plant exactly each
    # seed's planner MDE; the readout must detect it about 80% of the
    # time (the planner's stated power). Pinned: 87 - the variance
    # model is slightly conservative, so detection runs a few points
    # above the promise rather than below it.
    detected = 0
    for seed in range(3000, 3100):
        r = generate_dataset(seed=seed)
        m = pair_regions(r)
        ctrl = weekly_group_means(r, m["assignment"]["control"], "pre")
        baseline = sum(ctrl.values()) / len(ctrl)
        lift = pair_mde(r, m, post_weeks=8) / baseline
        out = simulate_experiment(r, m, lift=lift, n_permutations=500)
        detected += out["p_value"] < 0.05
    check("E16 planner MDE detection (pinned 87/100, band 75-95)",
          detected == 87 and 75 <= detected <= 95,
          f"detected={detected}/100")

    # E17: monotonicity - more weeks or more pairs shrink the MDE;
    # a bigger target lift needs fewer weeks.
    mde_weeks = [pair_mde(rows, matched, post_weeks=w) for w in (2, 4, 8, 16)]
    mde_pairs = [pair_mde(rows, matched, post_weeks=8, n_pairs=n)
                 for n in (10, 20, 40)]
    req = [required_post_weeks_for_lift(rows, matched, lift)
           for lift in (150.0, 300.0, 600.0)]
    mono = (all(a > b for a, b in zip(mde_weeks, mde_weeks[1:]))
            and all(a > b for a, b in zip(mde_pairs, mde_pairs[1:]))
            and all(a > b for a, b in zip(req, req[1:])))
    check("E17 MDE monotonicity (weeks down, pairs down, lift up)",
          mono, f"mde_weeks={[round(x, 1) for x in mde_weeks]} req={req}")

    passed = sum(1 for r in RESULTS if r["passed"])
    total_n = len(RESULTS)
    summary_doc = {"passed": passed, "total": total_n,
                   "all_passed": passed == total_n, "evals": RESULTS}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_doc, sort_keys=True) + "\n")
    print(f"\n{passed}/{total_n} evals passed -> evals/eval_report.json")
    return 0 if passed == total_n else 1


if __name__ == "__main__":
    sys.exit(main())
