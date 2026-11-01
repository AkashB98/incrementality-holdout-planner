# incrementality-holdout-planner

**Geo-holdout experiment planning whose promises hold up in simulation: matched-market pairing, pair-level power sizing, and a permutation-tested difference-in-differences readout — calibrated by Monte Carlo evals that run an A/A null test, a coverage test, and a planner-vs-readout detection test across 100 fresh seeds each.**

Every marketing team eventually asks the only question attribution cannot answer: *what would have happened if we had not run this?* The rigorous answer is an **incrementality test** — an experiment that withholds the treatment from a comparable holdout group and measures the difference the treatment actually caused. For regional media (TV, out-of-home, geo-targeted spend) the unit of randomisation is the region, so the design questions are concrete: which regions pair with which, how long must the test run to see a plausible lift, and how will the result be read out? This project answers all three — and then does the part most planners skip: it **simulates its own design hundreds of times and checks that the promised false-positive rate, confidence-interval coverage, and detection power actually happen**. A plan whose stated 80% power detects 66% of the time in simulation is not an 80%-power plan, and this repo's evals exist to catch exactly that lie (they caught it here once — see the development loop).

> **Synthetic data only.** Every figure is fabricated by a seeded generator for a fictional smart-home store (**Helios Home**). No real company, customer, or revenue data is used anywhere in this project.

Stdlib-only Python — deliberately. An experiment plan is an audit artifact: anyone should be able to read every line of the statistics, re-run it offline, and get byte-identical outputs, with no dependency tree between them and the math. Zero network, zero API keys, zero external dependencies.

This project completes a three-part measurement story alongside its sibling builds: **attribution-pipeline** (credit assignment inside observed data) → **marketing-mix-model** (aggregate response curves from historical spend) → **incrementality-holdout-planner** (the experimental calibration both of the others ultimately answer to).

## Quickstart

```bash
# 1. Generate the synthetic Helios Home regional dataset
#    (40 regions, 52 pre weeks + 8 post weeks, seed 42)
python3 cli.py generate --out dataset.jsonl

# 2. Matched markets: 20 pairs, balance quality reported
python3 cli.py match --input dataset.jsonl

# 3. Power / MDE table: what lifts of 2% / 5% / 10% this design can see
python3 cli.py power --input dataset.jsonl

# 4. Simulate: plant a +5% lift in test regions, recover it with DiD
python3 cli.py simulate --input dataset.jsonl

# 5. Full report: matching + power + readout (JSON)
python3 cli.py report --input dataset.jsonl

# 6. CI gate: exit 0 = pass, 1 = breach (matching imbalance above
#    --max-imbalance, or the readout CI misses the planted lift),
#    2 = data-quality / IO error
python3 cli.py check --input dataset.jsonl

# 7. One-command end-to-end demo (generate -> match -> power -> serve)
python3 demo.py

# 8. Tiny localhost API: POST /readout, GET /report, GET /health on 127.0.0.1
python3 cli.py serve --port 8765
```

## The calibration story (the headline)

Four Monte Carlo evals, 100 fresh seeds each, pinned exactly in `evals/run_evals.py`:

| Promise | Simulation says | Verdict |
|---|---|---|
| **A/A null test** — no lift planted, 5% significance level | **2/100** false positives | ✅ Conservative: the permutation p-value is discrete, so the test fires *less* often than nominal rather than more |
| **95% CI coverage** — planted +5%, interval should cover it ~95% of the time | **91/100** covered | ⚠️ Honest miss: the normal approximation across only 20 pairs undercovers by ~4 points (see Limits) |
| **Planner-matches-readout** — plant exactly the planner's MDE, promised 80% detection | **87/100** detected | ✅ The planner is slightly conservative — detection runs *above* the promise, the safe direction |
| **Monotonicity** — more weeks / more pairs shrink the MDE; bigger lifts need fewer weeks | MDE $447.48 → $352.65 → $301.50 → $290.15 for 2/4/8/16 post weeks | ✅ By construction and by measurement |

## The seeded plan at a glance (seed 42, 40 regions, 52 pre + 8 post weeks)

The dataset holds **2,400 rows** and **$26,847,442.50** of synthetic weekly sales. Matching forms **20 pairs** (first pair: `region_22` ↔ `region_27`) and splits each pair across the arms. Pre-period balance: mean weekly sales of **$11,074.97** (test) vs **$11,114.86** (control) — an imbalance of **0.36%** — and the arms' weekly mean series correlate at **0.853**, because the generator gives every region the same seasonality and shared weekly shocks.

Power, sized at **pair level** (see below) from a variance model fitted on the pre period — predicted pair-difference standard deviation **$481.28**:

| True lift | Lift ($) | Power at 8 post weeks | Post weeks required for 80% power |
|---|---|---|---|
| 2% | $222.30 | 0.542 | 41 (a long test — the planner says so rather than rounding it down) |
| 5% | $555.74 | 0.999 | 2 |
| 10% | $1,111.49 | 1.000 | 1 |

The **minimum detectable effect (MDE)** — the smallest true lift this design detects with 80% probability at the 5% significance level — is **$301.50, about 2.71%** of the control baseline. The seeded simulation plants a **+5% lift** in the test regions' post weeks; the readout recovers **+4.37%** with a 95% **confidence interval (CI)** — the range that would cover the true value in 95% of repeated experiments — of **[+2.57%, +6.17%]**, a **randomisation (permutation) p-value** — the share of label-shuffled re-runs at least as extreme as the observed result — of **0.001** over 2,000 seeded permutations. The planted truth sits inside the interval.

## How it works

**The generator.** Each region gets a baseline level, a small weekly trend, and region noise; every region also shares one seasonal sine wave and one shared weekly shock draw. Those shared components are deliberate: they are why regions correlate (0.853 between the arms here) and why pairing has signal to work with. The post-period rows the generator emits are the *untreated* continuation; the lift exists only when `experiment.py` plants it.

**Matched-market selection.** Pairing uses exactly two ingredients, so the method explains itself in one sentence: regions are paired greedily by **scaled level gap + correlation gap** — how far apart their pre-period sales levels are (in standard-deviation units) plus how poorly their weekly pre-period series move together (1 − correlation). A good **matched market** — a region chosen as another's near-twin so the pair can be split across arms — is close on both. Each region's mean, volatility, and **trend slope** (fitted by **ordinary least squares (OLS)** — the line through the points that minimises squared vertical error) are still computed and reported, but they do not drive the pairing. One member of each pair goes to test, the other to control, alternating by pair index: deterministic and unbiased by construction rather than by a lucky random draw. No combinatorial optimisation — greedy closest-pair, auditable by hand.

**Inference: pair-level, permutation-first.** A naive **difference-in-differences (DiD)** — comparing the test-minus-control gap after the test to the same gap before it — that treats every region-week as an independent observation understates its standard errors badly, because a region's weekly sales are serially correlated. That is the Bertrand–Duflo–Mullainathan critique of DiD inference, and it is the biggest credibility risk in a project like this, so the readout is built to be immune to it: the data collapse to **one difference per matched pair** — (test post − test pre) − (control post − control pre) — and all inference happens across the 20 pair differences. The primary p-value is **randomisation inference**: under the null of no effect, the test/control labels inside a pair could just as well have been flipped, so sign-flipping each pair difference and recomputing the estimate (2,000 deterministic seeded permutations) builds the null distribution of the estimator itself. A normal-approximation CI across the pair differences (quantiles from `statistics.NormalDist`, never a hard-coded 1.96) is reported as the secondary inference.

**Power sizing in the readout's own units.** The planner sizes the design at pair level too — otherwise its promises and the readout's verdicts would be denominated differently. The pair-difference variance is decomposed from pre-period quantities: each region's residual variance around its OLS trend (the weekly-noise part, scaled by 1/n_post + 1/n_pre) plus the pair's squared trend-slope difference times the squared week-gap between the post and pre midpoints (the trend-divergence part — small trend differences amplify over a long gap, and an honest planner charges for that). `MDE = (z_{1−α/2} + z_{power}) · σ_pair / √n_pairs`. A placebo split of the pre period is also computed and reported as a diagnostic — it reads $289.24 against the model's $481.28, and the development loop below explains why the placebo number is *not* trusted for sizing.

**Data quality first.** `validation.py` runs before any plan: required fields, duplicate region-weeks, week continuity per region, non-negative numeric sales, the int/bool trap on `week`. It reports issues and never raises on bad data, so the `check` gate can map them to exit code 2, distinct from a design breach (exit 1).

## Architecture

| Module | Job |
|---|---|
| `generator.py` | Deterministic synthetic Helios Home regional dataset (40 regions × 60 weeks): per-region baseline/trend/noise plus shared seasonality and shared shocks; CSV / JSONL serialisers and loaders |
| `matching.py` | Per-region pre-period features (mean, std, OLS slope) for reporting; greedy pairing on scaled level gap + correlation gap; alternating arm assignment; balance quality (mean difference + weekly correlation) |
| `power.py` | Normal CDF/quantiles (stdlib `NormalDist`, erf cross-checked); pair-level planner: variance-decomposition pair sd, MDE, required post weeks, power table; placebo-split diagnostic; weekly-form MDE algebra kept for hand-verified cases |
| `experiment.py` | One DiD difference per matched pair; randomisation (sign-flip) permutation p-value, seeded; normal-approximation CI across pairs as secondary inference; lift planting + simulation scoring |
| `report.py` | Full plan assembly: dataset summary + matching quality + power/MDE table for 2%/5%/10% lifts + simulation readout |
| `validation.py` | Data-quality gate: required fields, duplicates, week gaps, ranges — reports issues, never raises on bad data |
| `cli.py` | `generate` / `match` / `power` / `simulate` / `readout` / `report` / `check` (CI exit 0/1/2) / `demo` / `serve` (stdlib `http.server` on 127.0.0.1: `POST /readout`, `GET /report`, `GET /health`); CSV, JSONL, and JSON-array inputs |
| `demo.py` | End-to-end narrative with asserts: determinism, 20 balanced pairs, planted lift recovered inside its CI, check exit codes, live localhost serve round-trip |
| `fixtures/` | The full seeded dataset (`sample_regions.jsonl` + `.csv`) with pinned totals |

## Evals

`python3 evals/run_evals.py` → **17/17 golden evals** (~40 s, dominated by the Monte Carlo evals), report committed at `evals/eval_report.json` (md5 `6adddb63414b6f5c656c9175a15b5c36`, byte-identical across 3 verified runs):

- **E1** generator determinism — seed 42 JSONL bytes pinned by md5
- **E2** matched pairs pinned (20 pairs, first pair `region_22`/`region_27`, 20/20 arm split)
- **E3** matching quality pinned (imbalance 0.36%, weekly correlation 0.853)
- **E4** OLS slope and correlation hand-computed (slope 2.0; correlations ±1.0)
- **E5** normal math pinned to known values (z₀.₉₇₅, z₀.₈₀, Φ(0) = 0.5)
- **E6** MDE algebra hand-computed (σ=100, n=8 → 140.0793; required-weeks and power invert it exactly)
- **E7** seeded pair-level power table pinned (σ_pair = $481.28, MDE $301.50 = 2.71%, 2% lift needs 41 post weeks)
- **E8** DiD pinned (estimate +4.3661%, permutation p = 0.001, CI [+2.57%, +6.17%] covers the planted +5%)
- **E9** zero-lift simulation covers zero (the estimator does not invent effects)
- **E10** validation flags planted defects; the generated dataset validates clean
- **E11** CLI `check` exit codes 0 / 1 / 2
- **E12** serve round-trip matches local readout + report exactly
- **E13** bundled fixture pinned (2,400 rows, total sales $26,847,442.50, validates clean)
- **E14** A/A null test — 2/100 false positives across 100 seeds (band 0–10)
- **E15** CI coverage — 91/100 across 100 seeds (band 84–98)
- **E16** planner-matches-readout — planting each seed's own MDE is detected 87/100 (band 75–95)
- **E17** MDE monotonicity — weeks ↓, pairs ↓, bigger lift → fewer required weeks

Unit tests: `python3 -m unittest discover -s tests` → **118 hermetic tests**, including a no-network enforcement test (sockets stubbed), hand-computed power/MDE cases against known `NormalDist` values, pair-difference and permutation-inference tests (including the level-difference-cancels case that is the entire point of DiD), matching determinism, planner monotonicity, validation defects, CLI exit codes, JSON/CSV round-trips, and serve round-trips.

## Development loop — what the checks actually caught

1. **The first planner lied about its power, and the Monte Carlo eval caught it.** The original sizing estimated pair noise from a placebo split of the pre period (fake post = last 8 pre weeks). Detection at that MDE across seeds: **66/100 against a promised 80**. The diagnosis is structural, not bad luck: a placebo window sits *inside* the pre period, so its week-gap to its own baseline is short — but the real readout's post weeks sit ~30 weeks of trend away from the pre midpoint, and any trend difference inside a pair is amplified by that gap. The placebo systematically understates the noise. Fix: the variance-decomposition planner above (residual noise + trend divergence charged at the real week-gap). Predicted pair sd on the seeded data: $481.28, against $456.31 actually realised in the readout — close, and slightly conservative. Detection at the new MDE: **87/100**, pinned by E16.
2. **Serial correlation was designed out, not argued out.** The first readout draft pooled region-weeks. Rather than defend it, the estimator was rebuilt around one difference per pair with sign-flip randomisation inference as the primary test — the design-based answer to the Bertrand–Duflo–Mullainathan critique — and the A/A eval (E14) now verifies the false-positive behaviour empirically instead of trusting the asymptotics.
3. **Coverage is 91%, and the README says so.** The normal-approximation CI across 20 pair differences undercovers by ~4 points (a t-approximation would be wider). It is reported as the secondary inference, its measured coverage is pinned by E15, and the gap is listed under Limits rather than tuned away by quietly widening the interval and keeping the "95%" label.

## Scope and honest limits

- **Coverage is 91%, not 95%.** The pair-level CI uses the normal approximation; with 20 pairs a t₁₉ critical value (2.093 vs 1.960) would cover closer to nominal. The permutation p-value — the primary inference — does not rely on the approximation, and its A/A behaviour is verified (E14).
- **No synthetic-control weighting.** Arms are simple matched pairs with equal weights. A production geo design might weight control regions to fit the test arm's pre period more tightly — deliberately out of scope; the pairing quality numbers show what plain matching achieves.
- **Geo spillover is ignored.** The simulation assumes a region's sales respond only to its own assignment. Real geo tests leak — commuters, shared media markets, national pricing — and no estimator in this repo models that.
- **The lift is planted as a clean, constant percentage.** Real treatment effects ramp, decay, and vary by region. The readout machinery would still run, but the simulation's tidy recovery is a property of the tidy plant.
- **Very long tests stop helping.** Beyond ~16 post weeks the trend-divergence term grows faster than the noise term shrinks, so the MDE flattens ($290.15 at 16 weeks) — the model says a longer test is not free accuracy, and the required-weeks figure of 41 for a 2% lift should be read as "this design cannot see 2% in any practical window", which is the honest answer.
- **The API is a demo surface.** It binds 127.0.0.1 only, holds its dataset in memory, and resets on restart.

## Why this project

It takes the author's day-job craft — marketing analytics: media measurement, KPI reporting, arguing about what a number actually proves — and rebuilds the hardest version of it as an analytics-engineering artifact: an experiment you design *before* spending, sized from a variance model, read out with design-based inference, and calibrated against simulation before anyone believes its promises. The same discipline a forward-deployed engineer brings to a customer problem: make the design inspectable, make the outputs reproducible, and make the failure modes exit codes.
