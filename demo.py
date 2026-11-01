"""End-to-end demo: generate -> validate -> match -> power -> simulate
-> readout -> check -> serve.

SYNTHETIC DATA ONLY (fictional Helios Home dataset). Runs fully
offline; the serve round-trip is loopback-only on an ephemeral port.
Every stage asserts.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from experiment import simulate_experiment  # noqa: E402
from generator import generate_dataset, rows_to_jsonl  # noqa: E402
from matching import pair_regions  # noqa: E402
from report import build_report  # noqa: E402
from validation import validate_dataset  # noqa: E402


def _request(port: int, path: str, payload=None, method: str = "GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def main() -> None:
    print("== 1. generate synthetic Helios Home regional dataset (seed=42) ==")
    rows = generate_dataset()
    again = generate_dataset()
    assert rows == again, "generator must be deterministic"
    assert len(rows) == 40 * 60, len(rows)
    assert validate_dataset(rows) == [], validate_dataset(rows)[:3]
    print(f"   {len(rows)} rows (40 regions x 60 weeks), deterministic, clean")

    print("\n== 2. matched-market selection ==")
    matched = pair_regions(rows)
    assert matched["pair_count"] == 20, matched["pair_count"]
    quality = matched["quality"]
    assert quality["abs_mean_difference_pct"] < 0.05, quality
    print(f"   20 pairs, pre-period imbalance "
          f"{quality['abs_mean_difference_pct']:.2%}, weekly correlation "
          f"{quality['weekly_correlation']:.3f}")

    print("\n== 3. power / MDE table ==")
    report = build_report(rows)
    for entry in report["power"]["table"]:
        print(f"   lift {entry['lift_pct']:.0%}: power={entry['power']:.3f} "
              f"detectable={entry['detectable']} "
              f"required_post_weeks={entry['required_post_weeks']}")

    print("\n== 4. simulate + DiD readout recovers the planted +5% lift ==")
    readout = simulate_experiment(rows, matched, lift=0.05)
    assert readout["covers_planted_lift"], readout
    print(f"   planted=+5.00% estimated={readout['did_lift_pct']:.2%} "
          f"95% CI [{readout['ci_95_pct'][0]:.2%}, {readout['ci_95_pct'][1]:.2%}] "
          f"p={readout['p_value']}")

    print("\n== 5. CLI check gate ==")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="holdout-demo-")
    path = os.path.join(tmp, "data.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(rows))
    code = cli_main(["check", "--input", path])
    code_err = cli_main(["check", "--input", "/tmp/no-such-holdout-xyz.jsonl"])
    print(f"   seeded={code} missing-file={code_err}")
    assert code == EXIT_OK and code_err == 2

    print("\n== 6. live localhost serve round-trip ==")
    reset_state(rows)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = _request(port, "/health")
        assert status == 200 and health == {"status": "ok"}
        status, served = _request(port, "/readout", payload={}, method="POST")
        assert status == 200 and served == readout, (served, readout)
        status, served_report = _request(port, "/report")
        assert status == 200 and served_report == report
        print("   POST /readout and GET /report match local computation")
    finally:
        server.shutdown()
        server.server_close()
        reset_state()

    print("\nDemo OK: generate, validate, match, power, simulate, "
          "readout, check and serve verified.")


if __name__ == "__main__":
    main()
