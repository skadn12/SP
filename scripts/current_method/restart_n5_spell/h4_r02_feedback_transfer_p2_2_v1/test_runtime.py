"""No-network tests of the authorized first-acceptance boundary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..backends import BackendResponse
from ..core import atomic_json
from . import runtime as r


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.root_id = r.FIRST_ROOTS[0]
        self.root = r.root_path(self.run, self.root_id)
        self.ledger = r.BoundedLedger(self.root, "freeze", self.run, self.root_id, backend_mode="mock")
        self.calls = 0

    def call(self):
        self.calls += 1
        return BackendResponse('{"intents":[]}', "fixture", {"transport_attempt_count": 1})

    def kwargs(self):
        return dict(request_id=f"{self.root_id}-intent-miner", intent="miner", attempt=1,
                    prompt_sha256="prompt", call=self.call, budget_class="qa",
                    budget_limit=10000, total_budget_limit=10000,
                    request_metadata={"max_tokens": 4096}, backend_identity={"model": "fixture"})

    def test_completed_cache_exact_and_drift_guard(self):
        a = self.ledger.invoke(**self.kwargs())
        b = self.ledger.invoke(**self.kwargs())
        self.assertEqual(a.raw_text, b.raw_text)
        self.assertEqual(self.calls, 1)
        for field, value in (("prompt_sha256", "different"), ("backend_identity", {"model": "other"}),
                             ("request_metadata", {"max_tokens": 8192})):
            args = self.kwargs(); args[field] = value
            with self.assertRaises(r.GuardrailStop):
                self.ledger.invoke(**args)
        state = self.ledger.delegate.read(self.kwargs()["request_id"])
        Path(state["raw_response_path"]).write_text("tampered")
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**self.kwargs())

    def test_failed_logical_call_counted_and_not_resent(self):
        def fail():
            self.calls += 1
            raise TimeoutError("fixture timeout")
        args = {**self.kwargs(), "call": fail}
        for _ in range(2):
            with self.assertRaisesRegex(Exception, "fixture timeout"):
                self.ledger.invoke(**args)
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(r.state_paths(self.run)), 1)
        failure = self.root / "ledger/observed_failures" / (args["request_id"] + ".json")
        self.assertFalse(r.read(failure)["automatic_resend"])
        failure.unlink()
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**args)

    def test_orphan_any_root_blocks_new_call(self):
        other = r.root_path(self.run, r.FIRST_ROOTS[1])
        atomic_json(other / "ledger/requests/orphan.json", {"status": "reserved", "request_id": "orphan"})
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 0)

    def test_token_guard_before_reservation(self):
        def preflight(*args):
            raise r.GuardrailStop("fixture token overflow")
        self.ledger.before_request = preflight
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**self.kwargs())
        self.assertEqual(len(r.state_paths(self.run)), 0)
        self.assertEqual(self.calls, 0)

    def test_root_and_global_caps_including_failures(self):
        with patch.object(r, "LIMIT_PER_ROOT", 0):
            with self.assertRaises(r.GuardrailStop):
                self.ledger.invoke(**self.kwargs())
        with patch.object(r, "LIMIT_TOTAL", 0):
            with self.assertRaises(r.GuardrailStop):
                self.ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 0)

    def test_unique_lock_and_unknown_request(self):
        with r.exclusive_run(self.run):
            with self.assertRaisesRegex(RuntimeError, "already_running"):
                with r.exclusive_run(self.run):
                    pass
        atomic_json(self.root / "reasoning_contracts.json", [{"intent_id": "I002"}])
        args = self.kwargs(); args["request_id"] = f"{self.root_id}-contract_guided-generator-I001"
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**args)
        args["request_id"] = f"{self.root_id}-contract_guided-repair-I002"
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**args)
        self.assertEqual(self.calls, 0)

    def test_usage_drift_guard_preserves_completed_response(self):
        def mismatch(request_id, response):
            raise r.GuardrailStop("fixture service count drift")
        self.ledger.after_response = mismatch
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**self.kwargs())
        state = self.ledger.delegate.read(self.kwargs()["request_id"])
        self.assertEqual(state["status"], "completed")
        self.assertTrue(Path(state["raw_response_path"]).is_file())
        self.assertEqual(self.calls, 1)

    def test_only_fixed_roots(self):
        with self.assertRaises(r.GuardrailStop):
            r.BoundedLedger(self.root, "freeze", self.run, "cc-train-episode-94289")


if __name__ == "__main__":
    unittest.main()
