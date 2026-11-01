"""Data-quality validation for regional weekly sales rows.

Returns a list of human-readable issue strings; an empty list means the
dataset is clean. Validation never raises on bad *data* - it reports -
so the CLI ``check`` gate can map issues to exit code 2.
"""

from __future__ import annotations

from typing import Dict, List

from generator import ROW_FIELDS

REQUIRED_FIELDS = ROW_FIELDS


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_row(row: Dict, index: int = 0) -> List[str]:
    issues: List[str] = []
    label = row.get("region_id", f"<row {index}>")
    for field in REQUIRED_FIELDS:
        if field not in row or row[field] is None or row[field] == "":
            issues.append(f"{label}: missing field '{field}'")
    if issues:
        return issues
    if not isinstance(row["region_id"], str) or not row["region_id"]:
        issues.append(f"{label}: region_id must be a non-empty string")
    if row.get("period") not in ("pre", "post"):
        issues.append(f"{label}: period must be 'pre' or 'post'")
    week = row.get("week")
    if not isinstance(week, int) or isinstance(week, bool):
        issues.append(f"{label}: week must be an int")
    elif week < 0:
        issues.append(f"{label}: week must be >= 0")
    sales = row.get("sales")
    if not _is_number(sales):
        issues.append(f"{label}: sales must be a number")
    elif sales < 0:
        issues.append(f"{label}: sales must be >= 0")
    return issues


def validate_dataset(rows: List[Dict]) -> List[str]:
    issues: List[str] = []
    if not rows:
        return ["dataset is empty"]
    for i, row in enumerate(rows):
        issues.extend(validate_row(row, index=i))
    seen = set()
    for row in rows:
        key = (row.get("region_id"), row.get("week"))
        if key in seen:
            issues.append(f"{key[0]} week {key[1]}: duplicate row")
        seen.add(key)
    # Week continuity per region (only over int weeks).
    by_region: Dict[str, List[int]] = {}
    for row in rows:
        if isinstance(row.get("week"), int) and not isinstance(row.get("week"), bool):
            by_region.setdefault(row.get("region_id"), []).append(row["week"])
    for rid, weeks in sorted(by_region.items()):
        ordered = sorted(set(weeks))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur - prev > 1:
                issues.append(f"{rid}: week gap {prev} -> {cur}")
    periods = {r.get("period") for r in rows}
    if rows and "pre" not in periods:
        issues.append("dataset has no pre-period rows")
    return issues
