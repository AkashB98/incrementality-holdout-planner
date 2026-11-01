"""Deterministic synthetic regional sales generator - Helios Home (fictional).

SYNTHETIC DATA ONLY. No real company, customer, or revenue data is ever
used. Every row is fabricated from a seeded PRNG so the same seed always
yields the same dataset, byte-for-byte.

One row per region per week:

  region_id, region_name, week, period, sales

40 regions, 52 pre-period weeks and (by default) 8 post-period weeks.
The post-period rows are the *untreated* continuation of the same data
process - ``experiment.py`` plants a lift on top of them; the generator
itself never applies a treatment effect.

Data process (per week ``w``, per region ``r``):

  sales[r][w] = baseline[r] + trend[r] * w
              + seasonality[w]            (shared sine wave)
              + shared_shock[w]           (one draw per week, all regions)
              + noise[r][w]               (region noise)

The shared seasonality and shared shocks are what make regions
correlate with each other - the property matched-market selection and
difference-in-differences both rely on. Shared shocks cancel out of a
test-minus-control difference, which is why the experiment readout can
be precise even though raw weekly sales are noisy.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

DEFAULT_SEED = 42
DEFAULT_REGIONS = 40
DEFAULT_PRE_WEEKS = 52
DEFAULT_POST_WEEKS = 8

REGION_NAMES = [
    "Aurora Vale", "Birch Hollow", "Cedar Junction", "Dune Harbor",
    "Elm Crossing", "Foxglove Ridge", "Granite Falls", "Harbor Lights",
    "Iron Meadow", "Juniper Bay", "Kestrel Point", "Larkspur Flats",
    "Maple Terrace", "Northwind Bluff", "Osprey Landing", "Pinecrest",
    "Quarry Hill", "Riverbend", "Sable Coast", "Tamarack Glen",
    "Umber Plains", "Verdant Row", "Willow Shoals", "Xenon District",
    "Yucca Mesa", "Zephyr Heights", "Alder Grove", "Briar Patch",
    "Cobalt Springs", "Driftwood Cove", "Ember Ridge", "Fern Hollow",
    "Golden Acre", "Hollow Creek", "Indigo Basin", "Jasper Field",
    "Karma Flats", "Linden Park", "Moss Landing", "Nimbus Valley",
]

ROW_FIELDS = ["region_id", "region_name", "week", "period", "sales"]


def region_ids(n: int = DEFAULT_REGIONS) -> List[str]:
    return [f"region_{i + 1:02d}" for i in range(n)]


def generate_dataset(pre_weeks: int = DEFAULT_PRE_WEEKS,
                     post_weeks: int = DEFAULT_POST_WEEKS,
                     seed: int = DEFAULT_SEED,
                     n_regions: int = DEFAULT_REGIONS) -> List[Dict]:
    """Return synthetic weekly sales rows, deterministic for ``seed``."""
    if pre_weeks < 0 or post_weeks < 0:
        raise ValueError("week counts must be >= 0")
    if n_regions < 1 or n_regions > len(REGION_NAMES):
        raise ValueError("n_regions out of supported range")
    rng = random.Random(seed)
    total_weeks = pre_weeks + post_weeks

    # Shared (common) components: one draw per week, seen by all regions.
    shared_shock = [rng.gauss(0.0, 250.0) for _ in range(total_weeks)]
    seasonality = [400.0 * math.sin(2.0 * math.pi * w / 52.0)
                   for w in range(total_weeks)]

    # Per-region parameters, drawn in region order.
    params = []
    for i in range(n_regions):
        baseline = max(3000.0, rng.gauss(10000.0, 2200.0))
        trend = rng.gauss(12.0, 7.0)
        noise_sd = baseline * 0.05
        params.append((baseline, trend, noise_sd))

    rows: List[Dict] = []
    ids = region_ids(n_regions)
    for i, rid in enumerate(ids):
        baseline, trend, noise_sd = params[i]
        for wk in range(total_weeks):
            sales = (baseline + trend * wk + seasonality[wk]
                     + shared_shock[wk] + rng.gauss(0.0, noise_sd))
            rows.append({
                "region_id": rid,
                "region_name": REGION_NAMES[i],
                "week": wk,
                "period": "pre" if wk < pre_weeks else "post",
                "sales": round(max(1.0, sales), 2),
            })
    return rows


def generate_pre_period(pre_weeks: int = DEFAULT_PRE_WEEKS,
                        seed: int = DEFAULT_SEED,
                        n_regions: int = DEFAULT_REGIONS) -> List[Dict]:
    """Just the pre-period rows of the seeded dataset."""
    full = generate_dataset(pre_weeks=pre_weeks, post_weeks=0,
                            seed=seed, n_regions=n_regions)
    return full


# ------------------------------------------------------------ serializers

def rows_to_jsonl(rows: List[Dict]) -> str:
    import json
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def rows_to_csv(rows: List[Dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in ROW_FIELDS})
    return buf.getvalue()


# --------------------------------------------------------------- loaders

def _coerce(row: Dict) -> Dict:
    out = dict(row)
    if isinstance(out.get("week"), str):
        try:
            out["week"] = int(out["week"])
        except ValueError:
            pass
    if isinstance(out.get("sales"), str):
        try:
            out["sales"] = float(out["sales"])
        except ValueError:
            pass
    return out


def load_rows(path: str) -> List[Dict]:
    import csv
    import io
    import json
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.strip()
    if not stripped:
        return []
    if path.endswith(".csv"):
        return [_coerce(dict(r)) for r in csv.DictReader(io.StringIO(text))]
    if stripped.startswith("["):
        return [_coerce(dict(x)) for x in json.loads(stripped)]
    return [_coerce(json.loads(line))
            for line in stripped.splitlines() if line.strip()]


def write_rows(rows: List[Dict], path: str, fmt: str = None) -> None:
    if fmt is None:
        fmt = "csv" if path.endswith(".csv") else "jsonl"
    text = rows_to_csv(rows) if fmt == "csv" else rows_to_jsonl(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# Aliases matching the sibling builds' naming.
load_dataset = load_rows
write_dataset = write_rows
