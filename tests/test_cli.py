"""CLI: commands, exit codes, JSON/CSV round-trips, serve round-trips."""

import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from generator import generate_dataset, rows_to_jsonl  # noqa: E402
from report import build_report  # noqa: E402


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


class CliBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="holdout-test-")
        self.good = os.path.join(self.tmp, "good.jsonl")
        with open(self.good, "w", encoding="utf-8") as fh:
            fh.write(rows_to_jsonl(generate_dataset()))


class TestGenerate(CliBase):
    def test_generate_to_file(self):
        out_path = os.path.join(self.tmp, "gen.jsonl")
        code, _, _ = run_cli(["generate", "--out", out_path])
        self.assertEqual(code, EXIT_OK)
        with open(out_path, encoding="utf-8") as fh:
            self.assertEqual(len(fh.read().strip().splitlines()), 2400)

    def test_generate_csv(self):
        out_path = os.path.join(self.tmp, "gen.csv")
        code, _, _ = run_cli(["generate", "--pre-weeks", "4",
                              "--post-weeks", "1", "--out", out_path])
        self.assertEqual(code, EXIT_OK)
        with open(out_path, encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
        self.assertEqual(len(lines), 40 * 5 + 1)
        self.assertTrue(lines[0].startswith("region_id,region_name"))


class TestCommands(CliBase):
    def test_match_command(self):
        code, out, _ = run_cli(["match", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertEqual(payload["pair_count"], 20)

    def test_power_command(self):
        code, out, _ = run_cli(["power", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertEqual(len(payload["table"]), 3)

    def test_simulate_command(self):
        code, out, _ = run_cli(["simulate", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(json.loads(out)["covers_planted_lift"])

    def test_readout_command(self):
        code, out, _ = run_cli(["readout", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        self.assertAlmostEqual(json.loads(out)["did_lift_pct"],
                               0.043661, places=5)

    def test_report_command_matches_local(self):
        code, out, _ = run_cli(["report", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out), build_report(generate_dataset()))

    def test_missing_file_is_error(self):
        code, _, _ = run_cli(["match", "--input", "/tmp/nope-holdout.jsonl"])
        self.assertEqual(code, EXIT_ERROR)


class TestCheck(CliBase):
    def test_check_passes_on_seeded(self):
        code, out, _ = run_cli(["check", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("check PASS", out)

    def test_check_breach_on_tight_imbalance_threshold(self):
        code, out, _ = run_cli(["check", "--input", self.good,
                                "--max-imbalance", "0.001"])
        self.assertEqual(code, EXIT_FAIL)
        self.assertIn("imbalance", out)

    def test_check_error_on_missing_file(self):
        code, _, _ = run_cli(["check", "--input", "/tmp/nope-holdout.jsonl"])
        self.assertEqual(code, EXIT_ERROR)

    def test_check_error_on_bad_data(self):
        bad = os.path.join(self.tmp, "bad.jsonl")
        rows = generate_dataset()
        rows[0] = dict(rows[0], sales=-5.0)
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(rows_to_jsonl(rows))
        code, _, _ = run_cli(["check", "--input", bad])
        self.assertEqual(code, EXIT_ERROR)


class TestServe(CliBase):
    def _serve(self):
        reset_state(generate_dataset())
        server = make_server(0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, port

    def _get(self, port, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                    timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    def test_health(self):
        server, port = self._serve()
        try:
            status, payload = self._get(port, "/health")
            self.assertEqual((status, payload), (200, {"status": "ok"}))
        finally:
            server.shutdown(); server.server_close(); reset_state()

    def test_report_round_trip(self):
        server, port = self._serve()
        try:
            status, payload = self._get(port, "/report")
            self.assertEqual(status, 200)
            self.assertEqual(payload, build_report(generate_dataset()))
        finally:
            server.shutdown(); server.server_close(); reset_state()

    def test_readout_post_round_trip(self):
        server, port = self._serve()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/readout", data=b"{}",
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read())
            self.assertTrue(payload["covers_planted_lift"])
        finally:
            server.shutdown(); server.server_close(); reset_state()

    def test_unknown_endpoint_404(self):
        server, port = self._serve()
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._get(port, "/nope")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown(); server.server_close(); reset_state()


if __name__ == "__main__":
    unittest.main()
