"""CLI: generate | match | power | simulate | readout | report | check | demo | serve.

SYNTHETIC DATA ONLY - Helios Home is a fictional company and every
dataset handled here is fabricated sample data.

Exit codes:
  check : 0 = pass, 1 = breach (matching imbalance above threshold, or
          the seeded readout's 95% CI fails to cover the planted lift),
          2 = usage / IO / data-quality error
  others: 0 = success, 2 = usage / IO error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import DEFAULT_LIFT, simulate_experiment  # noqa: E402
from generator import DEFAULT_POST_WEEKS, DEFAULT_PRE_WEEKS, generate_dataset, load_rows, write_rows  # noqa: E402
from matching import pair_regions  # noqa: E402
from power import power_table, weekly_diff_std  # noqa: E402
from report import build_report  # noqa: E402
from validation import validate_dataset  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
DEFAULT_MAX_IMBALANCE_PCT = 0.05


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)


def _load_or_error(path: str) -> Optional[List[Dict]]:
    try:
        return load_rows(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _validated_or_error(path: str):
    rows = _load_or_error(path)
    if rows is None:
        return None, EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}", file=sys.stderr)
        return None, EXIT_ERROR
    return rows, EXIT_OK


def _write_out(args, text: str, summary: str) -> int:
    if getattr(args, "out", None):
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(summary)
    else:
        print(text, end="")
    return EXIT_OK


def cmd_generate(args) -> int:
    rows = generate_dataset(pre_weeks=args.pre_weeks,
                            post_weeks=args.post_weeks, seed=args.seed)
    fmt = "csv" if (args.out and args.out.endswith(".csv")) else "jsonl"
    if args.out:
        try:
            write_rows(rows, args.out, fmt=fmt)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"generated {len(rows)} synthetic rows (seed={args.seed}) -> {args.out}")
        return EXIT_OK
    from generator import rows_to_csv, rows_to_jsonl
    print(rows_to_csv(rows) if fmt == "csv" else rows_to_jsonl(rows), end="")
    return EXIT_OK


def cmd_match(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    matched = pair_regions(rows)
    payload = {"pair_count": matched["pair_count"], "pairs": matched["pairs"],
               "assignment": matched["assignment"], "quality": matched["quality"]}
    return _write_out(args, canonical(payload) + "\n", f"match -> {args.out}")


def cmd_power(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    matched = pair_regions(rows)
    assignment = matched["assignment"]
    sigma = weekly_diff_std(rows, assignment["test"], assignment["control"])
    table = power_table(rows, matched, post_weeks=args.post_weeks,
                        pre_weeks=args.pre_weeks)
    payload = {"weekly_diff_std": round(sigma, 4), "post_weeks": args.post_weeks,
               "pre_weeks": args.pre_weeks, "table": table}
    return _write_out(args, canonical(payload) + "\n", f"power -> {args.out}")


def cmd_simulate(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    matched = pair_regions(rows)
    readout = simulate_experiment(rows, matched, lift=args.lift)
    return _write_out(args, canonical(readout) + "\n", f"simulate -> {args.out}")


def cmd_readout(args) -> int:
    # readout = simulate with the default planted lift, named for the
    # analyst workflow (plan -> simulate -> read out).
    return cmd_simulate(args)


def cmd_report(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    report = build_report(rows, lift=args.lift, post_weeks=args.post_weeks,
                          pre_weeks=args.pre_weeks)
    return _write_out(args, canonical(report) + "\n", f"report -> {args.out}")


def cmd_check(args) -> int:
    rows = _load_or_error(args.input)
    if rows is None:
        return EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}")
        print(f"check FAIL (data quality): {len(issues)} issue(s)")
        return EXIT_ERROR
    matched = pair_regions(rows)
    quality = matched["quality"]
    readout = simulate_experiment(rows, matched, lift=args.lift)
    print(f"pairs={matched['pair_count']} "
          f"imbalance_pct={quality['abs_mean_difference_pct']:.4f} "
          f"max_imbalance_pct={args.max_imbalance} "
          f"correlation={quality['weekly_correlation']:.4f} "
          f"did_lift_pct={readout['did_lift_pct']:.4f} "
          f"planted={args.lift} covers={readout['covers_planted_lift']}")
    if quality["abs_mean_difference_pct"] > args.max_imbalance:
        print("check FAIL: matching imbalance above threshold")
        return EXIT_FAIL
    if not readout["covers_planted_lift"]:
        print("check FAIL: readout CI does not cover the planted lift")
        return EXIT_FAIL
    print("check PASS")
    return EXIT_OK


def cmd_demo(args) -> int:
    import demo as demo_mod
    try:
        demo_mod.main()
    except AssertionError as exc:
        print(f"demo FAILED: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


# ------------------------------------------------------------------- serve

class _State:
    def __init__(self):
        self.rows: List[Dict] = generate_dataset()


_STATE = _State()


def get_state() -> _State:
    return _STATE


def reset_state(rows: Optional[List[Dict]] = None) -> None:
    _STATE.rows = list(rows) if rows is not None else generate_dataset()


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Any) -> None:
        body = (canonical(payload) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/report":
            try:
                return self._send_json(200, build_report(_STATE.rows))
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})
        return self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):  # noqa: N802
        if self.path != "/readout":
            return self._send_json(404, {"error": "unknown endpoint"})
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw.strip() else {}
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            rows = body.get("rows", _STATE.rows)
            lift = float(body.get("lift", DEFAULT_LIFT))
            matched = pair_regions(rows)
            readout = simulate_experiment(rows, matched, lift=lift)
            return self._send_json(200, readout)
        except (ValueError, TypeError, KeyError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

    def log_message(self, *a: Any) -> None:
        pass


def make_server(port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), _Handler)


def cmd_serve(args) -> int:
    if args.input:
        rows = _load_or_error(args.input)
        if rows is None:
            return EXIT_ERROR
        reset_state(rows)
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{args.port}  "
          f"(POST /readout, GET /report, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="incrementality-holdout-planner",
        description="Geo-holdout experiment planner (Helios Home sample data).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a synthetic regional dataset")
    g.add_argument("--pre-weeks", type=int, default=DEFAULT_PRE_WEEKS)
    g.add_argument("--post-weeks", type=int, default=DEFAULT_POST_WEEKS)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_generate)

    for name, fn, helptext in (
            ("match", cmd_match, "matched-market pairing + balance quality"),
            ("power", cmd_power, "power / MDE table for the matched design"),
            ("simulate", cmd_simulate, "plant a lift, recover it with DiD"),
            ("readout", cmd_readout, "DiD readout of the seeded simulation"),
            ("report", cmd_report, "matching + power + readout, one document")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--input", required=True)
        s.add_argument("--out", default=None)
        if name in ("power", "report"):
            s.add_argument("--post-weeks", type=int, default=DEFAULT_POST_WEEKS)
            s.add_argument("--pre-weeks", type=int, default=DEFAULT_PRE_WEEKS)
        if name in ("simulate", "readout", "report"):
            s.add_argument("--lift", type=float, default=DEFAULT_LIFT)
        s.set_defaults(func=fn)

    c = sub.add_parser("check", help="CI gate: exit 0 pass / 1 breach / 2 error")
    c.add_argument("--input", required=True)
    c.add_argument("--lift", type=float, default=DEFAULT_LIFT)
    c.add_argument("--max-imbalance", type=float, default=DEFAULT_MAX_IMBALANCE_PCT)
    c.set_defaults(func=cmd_check)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="localhost JSON API")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--input", default=None)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
