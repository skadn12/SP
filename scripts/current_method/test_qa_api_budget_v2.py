"""Network-free durability, concurrency and refusal tests for the P0 budget."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

try:
    from . import qa_api_budget_v2 as b
except ImportError:
    import qa_api_budget_v2 as b

HASH = "a" * 64


def binding(stage="DISCOVERY", root=None):
    value = {"stage": stage, "max_tokens": 4096, "packet_sha256": "b" * 64}
    if root is not None:
        value["root_id"] = root
    return value


def _race_claim(path, output):
    try:
        with b.Ledger(path, "fixture", HASH) as ledger:
            ledger.claim_root(b.ROOTS[0])
        output.put("claimed")
    except b.BudgetError:
        output.put("refused")


def _race_reserve(path, output):
    try:
        with b.Ledger(path, "fixture", HASH) as ledger:
            ledger.reserve(b.ROOTS[0], "DISCOVERY", binding())
        output.put("reserved")
    except b.BudgetError:
        output.put("refused")


def _crash_after_reservation(path):
    ledger = b.Ledger(path, "fixture", HASH)
    ledger.claim_root(b.ROOTS[0])
    ledger.reserve(b.ROOTS[0], "DISCOVERY", binding())
    os._exit(0)  # No connection close or finally: emulate process termination.


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite3"
        self.ledger = b.Ledger(self.path, "fixture", HASH)
        self.root = b.ROOTS[0]

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def success(self, root, stage):
        attempt = self.ledger.reserve(root, stage, binding(stage, root))
        self.ledger.settle(attempt, "SUCCESS", {"response_sha256": "c" * 64,
                                              "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}})
        return attempt

    def test_full_12_attempt_budget_and_stage_order(self):
        for root in b.ROOTS:
            self.ledger.claim_root(root)
            for stage in b.STAGES:
                self.success(root, stage)
        state = self.ledger.snapshot()
        self.assertEqual(state["attempt_count"], 12)
        self.assertEqual(state["remaining_attempts"], 0)
        with self.assertRaisesRegex(b.BudgetError, "12-attempt"):
            self.ledger.reserve(self.root, "COMPARE", binding("COMPARE"))
        self.assertEqual(self.ledger.snapshot()["attempt_count"], 12)

    def test_stage_must_follow_success_and_cannot_repeat(self):
        self.ledger.claim_root(self.root)
        with self.assertRaises(b.BudgetError):
            self.ledger.reserve(self.root, "AUTHOR", binding("AUTHOR"))
        n = self.ledger.reserve(self.root, "DISCOVERY", binding())
        with self.assertRaises(b.BudgetError):
            self.ledger.reserve(self.root, "AUTHOR", binding("AUTHOR"))
        with self.assertRaises(b.BudgetError):
            self.ledger.reserve(self.root, "DISCOVERY", binding())
        self.ledger.settle(n, "SUCCESS", {})
        self.success(self.root, "AUTHOR")
        self.assertEqual(self.ledger.snapshot()["attempt_count"], 2)

    def test_timeout_and_disconnect_both_consume_and_never_retry(self):
        for root, status in zip(b.ROOTS, ("UNCERTAIN", "FAILED")):
            self.ledger.claim_root(root)
            n = self.ledger.reserve(root, "DISCOVERY", binding())
            self.ledger.settle(n, status, {"error_type": "fixture_timeout_or_disconnect"})
            for stage in ("DISCOVERY", "AUTHOR"):
                with self.assertRaises(b.BudgetError):
                    self.ledger.reserve(root, stage, binding(stage))
            self.ledger.stop_root(root, "Fixture request failed; consumed once.")
        self.assertEqual(self.ledger.snapshot()["attempt_count"], 2)

    def test_restart_preserves_reservation_and_refuses_reclaim(self):
        self.ledger.claim_root(self.root)
        self.ledger.reserve(self.root, "DISCOVERY", binding())
        self.ledger.close()
        with b.Ledger(self.path, "fixture", HASH) as restored:
            self.assertEqual(restored.snapshot()["attempt_count"], 1)
            with self.assertRaises(b.BudgetError):
                restored.claim_root(self.root)
            with self.assertRaises(b.BudgetError):
                restored.reserve(self.root, "DISCOVERY", binding())
            with self.assertRaises(b.BudgetError):
                restored.reserve(self.root, "AUTHOR", binding("AUTHOR"))

    def test_mode_manifest_experiment_and_unknown_root_refused(self):
        for kwargs in ({"mode": "live"}, {"manifest_sha256": "f" * 64}, {"experiment_id": "different"}):
            args = {"experiment_id": "fixture", "manifest_sha256": HASH, "mode": "mock", **kwargs}
            with self.assertRaises(b.BudgetError):
                b.Ledger(self.path, **args)
        with self.assertRaises(b.BudgetError):
            self.ledger.claim_root("root-not-in-three-root-pilot")
        self.assertEqual(self.ledger.snapshot()["attempt_count"], 0)

    def test_output_limit_and_root_stage_binding_fail_before_count(self):
        self.ledger.claim_root(self.root)
        for bad in ({"max_tokens": 8192}, {"max_tokens": True}, {"stage": "AUTHOR"}, {"root_id": b.ROOTS[1]}, {"invalid": float("nan")}):
            with self.assertRaises(b.BudgetError):
                self.ledger.reserve(self.root, "DISCOVERY", {**binding(), **bad})
        self.assertEqual(self.ledger.snapshot()["attempt_count"], 0)
        with self.assertRaises(b.BudgetError):
            self.ledger.reserve(self.root, "DISCOVERY", {"packet_sha256": "d"*64})

    def test_settlement_is_once_and_no_refund(self):
        self.ledger.claim_root(self.root)
        n = self.ledger.reserve(self.root, "DISCOVERY", binding())
        self.ledger.settle(n, "UNCERTAIN", {})
        with self.assertRaises(b.BudgetError):
            self.ledger.settle(n, "SUCCESS", {})
        self.assertEqual(self.ledger.snapshot()["attempt_count"], 1)

    def test_finished_root_cannot_be_claimed_or_reopened(self):
        self.ledger.claim_root(self.root)
        self.success(self.root, "DISCOVERY")
        self.ledger.finish_root(self.root, "NO_SAFE_QA", "No natural source relation.")
        for op in (lambda: self.ledger.claim_root(self.root),
                   lambda: self.ledger.reserve(self.root, "AUTHOR", binding("AUTHOR")),
                   lambda: self.ledger.finish_root(self.root, "HOLD", "second terminal")):
            with self.assertRaises(b.BudgetError):
                op()

    def test_pass_requires_all_four_successes(self):
        self.ledger.claim_root(self.root)
        self.success(self.root, "DISCOVERY")
        with self.assertRaises(b.BudgetError):
            self.ledger.finish_root(self.root, "PASS_STATIC_REVIEW")
        for stage in b.STAGES[1:]:
            self.success(self.root, stage)
        self.ledger.finish_root(self.root, "PASS_STATIC_REVIEW")
        self.assertEqual(next(r["status"] for r in self.ledger.snapshot()["roots"] if r["root_id"] == self.root), "PASS_STATIC_REVIEW")

    def test_secret_content_not_persisted_in_audit_json(self):
        self.ledger.claim_root(self.root)
        n = self.ledger.reserve(self.root, "DISCOVERY", {**binding(), "api_key": "secret-key-fixture", "messages": [{"content": "secret-prompt-fixture"}]})
        self.ledger.settle(n, "SUCCESS", {"raw_response": "secret-output-fixture", "api_key": "secret-receipt-fixture"})
        serialized = json.dumps(self.ledger.snapshot())
        self.assertNotIn("secret-", serialized)
        self.assertIn("request_sha256", serialized)

    def test_export_exclusive_and_durable(self):
        target = Path(self.tmp.name) / "export.json"
        self.ledger.export(target)
        self.assertEqual(json.loads(target.read_text())["attempt_count"], 0)
        with self.assertRaises(b.BudgetError):
            self.ledger.export(target)

    def test_corrupt_and_empty_existing_file_fail_closed(self):
        for name, content in (("empty.sqlite3", b""), ("bad.sqlite3", b"not a SQLite DB")):
            path = Path(self.tmp.name) / name
            path.write_bytes(content)
            with self.assertRaises(b.BudgetError):
                b.Ledger(path, "fixture", HASH)

    def test_missing_or_deleted_request_history_fails_closed(self):
        self.ledger.claim_root(self.root)
        self.success(self.root, "DISCOVERY")
        self.ledger.close()
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM attempts")
        with self.assertRaises(b.BudgetError):
            b.Ledger(self.path, "fixture", HASH)

    def test_multiprocess_claim_is_exclusive(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        jobs = [context.Process(target=_race_claim, args=(str(self.path), output)) for _ in range(4)]
        for p in jobs:
            p.start()
        results = [output.get(timeout=30) for _ in jobs]
        for p in jobs:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(results.count("claimed"), 1)
        self.assertEqual(results.count("refused"), 3)

    def test_multiprocess_reserve_is_exclusive(self):
        self.ledger.claim_root(self.root)
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        jobs = [context.Process(target=_race_reserve, args=(str(self.path), output)) for _ in range(4)]
        for p in jobs:
            p.start()
        results = [output.get(timeout=30) for _ in jobs]
        for p in jobs:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(results.count("reserved"), 1)
        self.assertEqual(self.ledger.snapshot()["attempt_count"], 1)

    def test_abrupt_process_exit_keeps_consumed_attempt(self):
        self.ledger.close()
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_crash_after_reservation, args=(str(self.path),))
        child.start()
        child.join(timeout=30)
        self.assertEqual(child.exitcode, 0)
        with b.Ledger(self.path, "fixture", HASH) as restored:
            state = restored.snapshot()
            self.assertEqual(state["attempt_count"], 1)
            self.assertEqual(state["attempts"][0]["status"], "RESERVED")
            with self.assertRaises(b.BudgetError):
                restored.claim_root(self.root)

    def test_unknown_table_schema_fails_closed(self):
        self.ledger.close()
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE foreign_table(payload TEXT)")
        with self.assertRaisesRegex(b.BudgetError, "unknown ledger schema"):
            b.Ledger(self.path, "fixture", HASH)

    def test_tampered_usage_and_response_hash_fail_closed(self):
        self.ledger.claim_root(self.root)
        self.success(self.root, "DISCOVERY")
        self.ledger.close()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE attempts SET response_sha256=?", ("f"*64,))
        with self.assertRaises(b.BudgetError):
            b.Ledger(self.path, "fixture", HASH)


if __name__ == "__main__":
    unittest.main()
