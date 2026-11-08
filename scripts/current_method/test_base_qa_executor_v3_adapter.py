"""Offline tests for the v3 global-ledger/proxy integration adapter."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

NETWORK_EVENTS = []


def _deny_network(event, args):
    if event.startswith("socket."):
        NETWORK_EVENTS.append(event)
        raise AssertionError("offline v3 adapter test blocked a network attempt")


sys.addaudithook(_deny_network)

from scripts.current_method import base_qa_executor_v2 as v2
from scripts.current_method import base_qa_executor_v3_adapter as executor_v3
from scripts.current_method import base_qa_global_budget_adapter_v3 as budget_v3
from scripts.current_method import base_qa_proxy_transport_v3 as proxy_v3
from scripts.current_method import qa_api_transport_v2 as transport_api


def make_body(stage="DISCOVERY"):
    packet = {"stage": stage, "fixture": "synthetic-only"}
    metadata = {"packet_sha256": transport_api._digest(packet),
                "author_id": "offline-test"}
    return transport_api.build_body(packet, metadata)


class Response:
    def __init__(self, status=200, content=None):
        self.status = status
        body = {"id": "mock-request-1", "model": transport_api.EXPECTED_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(content or {"status": "NO_SAFE_QA",
                    "reason": "synthetic fixture"}, separators=(",", ":"))}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
        self.raw = json.dumps(body, separators=(",", ":")).encode()

    def read(self, _limit):
        return self.raw

    def getheader(self, name):
        return "mock-request-1" if name.lower() == "x-request-id" else None


class FakeConnection:
    def __init__(self, events, *, fail_request=False):
        self.events = events
        self.fail_request = fail_request

    def connect(self):
        self.events.append("connect")

    def request(self, method, path, body, headers):
        self.events.append("request")
        assert method == "POST"
        assert path == "/api/v3/chat/completions"
        assert headers["Authorization"] == "Bearer fixture-token"
        assert json.loads(body)["max_tokens"] == 4096
        if self.fail_request:
            raise TimeoutError("synthetic timeout")

    def getresponse(self):
        self.events.append("response")
        return Response()

    def close(self):
        self.events.append("close")


class FakeStageTransport:
    def __init__(self):
        self.calls = 0

    def send(self, body, reservation):
        self.calls += 1
        assert body["max_tokens"] == 4096
        assert reservation.stage in budget_v3.STAGES
        return {"status": "NO_SAFE_QA", "reason": "offline fixture"}, {
            "returned_model": transport_api.EXPECTED_MODEL,
            "raw_response_sha256": "a" * 64, "request_id": "mock",
            "http_status": 200, "finish_reason": "stop",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "reasoning_tokens": None}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qa-v3-adapter-")
        self.root = Path(self.temp.name)
        self.ledger_path = self.root / "GLOBAL_12_ATTEMPTS.sqlite3"
        self.state = self.root / "state"
        experiment = self.root / "EXPERIMENT_PROBE.json"
        experiment.write_text('{"fixture":true}\n', encoding="utf-8")
        raw = experiment.read_bytes()
        binding = {"path": str(experiment.resolve()), "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}
        self.guard = {"purpose": "GLOBAL_BUDGET_TAKEOVER_PROBE_ONLY_QA_HOLD",
            "global_maximum_attempts": 12, "experiment_binding": binding,
            "canonical_directory": str(self.root.resolve()),
            "authoritative_ledger": str(self.ledger_path.resolve()), "qa_authorized": False}
        self.policy = {"schema": "base_qa_capability_probe_20261029_v1",
            "original_global_limit": 12, "experiment_binding": binding,
            "canonical_directory": str(self.root.resolve())}
        self._create_ledger(seed=4)

    def tearDown(self):
        self.temp.cleanup()

    def _create_ledger(self, seed):
        with sqlite3.connect(self.ledger_path) as db:
            db.execute("CREATE TABLE policy (singleton INTEGER PRIMARY KEY CHECK(singleton=1), value TEXT NOT NULL)")
            db.execute("CREATE TABLE attempts (attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, purpose TEXT NOT NULL UNIQUE, status TEXT NOT NULL CHECK(status IN ('RESERVED','SUCCEEDED','REJECTED','UNCERTAIN')), reserved_utc TEXT NOT NULL, settled_utc TEXT, request_sha256 TEXT NOT NULL, max_tokens INTEGER NOT NULL, receipt TEXT)")
            db.execute("CREATE TRIGGER global12 BEFORE INSERT ON attempts WHEN (SELECT COUNT(*) FROM attempts)>=12 BEGIN SELECT RAISE(ABORT,'GLOBAL_12_ATTEMPT_LIMIT'); END")
            db.execute("CREATE TRIGGER no_attempt_delete BEFORE DELETE ON attempts BEGIN SELECT RAISE(ABORT,'NO_REFUND_OR_RESET'); END")
            db.execute("CREATE TRIGGER immutable_attempt_identity BEFORE UPDATE OF purpose,reserved_utc,request_sha256,max_tokens ON attempts BEGIN SELECT RAISE(ABORT,'IMMUTABLE_ATTEMPT'); END")
            canonical = json.dumps(self.policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            db.execute("INSERT INTO policy(singleton,value) VALUES(1,?)", (canonical,))
            old = [("SYNTHETIC_CAPABILITY_PROBE_64", "UNCERTAIN"),
                   ("MANUAL_RETRY_20261030_V1", "UNCERTAIN"),
                   ("PROXY_ROUTE_PROBE_20261030_V1", "REJECTED"),
                   ("UPDATED_KEY_PROXY_PROBE_20261030_V1", "SUCCEEDED")]
            for index, (purpose, status) in enumerate(old[:seed]):
                db.execute("INSERT INTO attempts(purpose,status,reserved_utc,settled_utc,request_sha256,max_tokens,receipt) VALUES(?,?,?,?,?,?,?)",
                    (purpose, status, "2026-10-30T00:00:00+00:00", "2026-10-30T00:00:01+00:00",
                     str(index + 1) * 64, 64, "{}"))

    def _adapter(self):
        return budget_v3.GlobalBudgetAdapter(mode="mock", ledger_path=self.ledger_path,
            state_dir=self.state, test_policy=self.policy, test_guard=self.guard)

    def test_actual_authoritative_ledger_is_read_only_and_has_eight_left(self):
        live_path = budget_v3.CANONICAL_LEDGER
        before = hashlib.sha256(live_path.read_bytes()).hexdigest()
        with budget_v3.GlobalBudgetAdapter(mode="inspect") as ledger:
            snapshot = ledger.snapshot()
        after = hashlib.sha256(live_path.read_bytes()).hexdigest()
        self.assertEqual((snapshot["used"], snapshot["remaining"]), (4, 8))
        self.assertEqual(before, after)
        with self.assertRaises(budget_v3.BudgetHold):
            budget_v3.GlobalBudgetAdapter(mode="live")
        self.assertEqual(before, hashlib.sha256(live_path.read_bytes()).hexdigest())

    def test_fake_proxy_send_is_reserved_first_and_single_attempt(self):
        events = []
        body = make_body()
        with self._adapter() as ledger:
            ledger.claim_root(budget_v3.ROOTS[0])
            fake = proxy_v3.ProxyTransportV3(ledger, mode="mock",
                proxy_environment={"HTTPS_PROXY": "http://127.0.0.1:7897"},
                connection_factory=lambda trace: FakeConnection(events),
                credential_reader=lambda: "fixture-token")
            value, receipt, reservation = ledger.execute_stage(
                budget_v3.ROOTS[0], "DISCOVERY", body, fake)
            self.assertEqual(value["status"], "NO_SAFE_QA")
            self.assertEqual(receipt["usage"]["total_tokens"], 18)
            self.assertEqual(events, ["connect", "request", "response", "close"])
            self.assertEqual(ledger.snapshot()["used"], 5)
            self.assertTrue((self.state / "ATTEMPT_SEND_CLAIMS" /
                             ("attempt-%d.json" % reservation.attempt_id)).is_file())

    def test_stage_order_root_claim_and_restart_are_persistent(self):
        body = make_body()
        fake = FakeStageTransport()
        with self._adapter() as ledger:
            ledger.claim_root(budget_v3.ROOTS[0])
            with self.assertRaises(budget_v3.BudgetHold):
                ledger.execute_stage(budget_v3.ROOTS[0], "BLIND", body, fake)
            ledger.execute_stage(budget_v3.ROOTS[0], "DISCOVERY", body, fake)
        with self._adapter() as restarted:
            with self.assertRaises(budget_v3.BudgetHold):
                restarted.claim_root(budget_v3.ROOTS[0])
            self.assertEqual(restarted.snapshot()["used"], 5)
        self.assertEqual(fake.calls, 1)

    def test_only_one_executor_can_own_the_global_budget(self):
        first = self._adapter()
        try:
            with self.assertRaises(budget_v3.BudgetHold):
                self._adapter()
        finally:
            first.close()

    def test_timeout_consumes_budget_and_cannot_be_retried(self):
        events = []
        body = make_body()
        with self._adapter() as ledger:
            ledger.claim_root(budget_v3.ROOTS[0])
            fake = proxy_v3.ProxyTransportV3(ledger, mode="mock",
                proxy_environment={"HTTPS_PROXY": "http://127.0.0.1:7897"},
                connection_factory=lambda trace: FakeConnection(events, fail_request=True),
                credential_reader=lambda: "fixture-token")
            with self.assertRaises(proxy_v3.ProxyTransportError) as caught:
                ledger.execute_stage(budget_v3.ROOTS[0], "DISCOVERY", body, fake)
            self.assertTrue(caught.exception.uncertain)
            self.assertEqual(ledger.snapshot()["used"], 5)
            calls = events.count("request")
            with self.assertRaises(budget_v3.BudgetHold):
                ledger.execute_stage(budget_v3.ROOTS[0], "DISCOVERY", body, fake)
            self.assertEqual(events.count("request"), calls)

    def test_unsettled_reservation_after_restart_is_a_hard_hold(self):
        with self._adapter() as ledger:
            ledger.claim_root(budget_v3.ROOTS[0])
            ledger.reserve(budget_v3.ROOTS[0], "DISCOVERY", make_body())
        with self.assertRaises(budget_v3.BudgetHold):
            self._adapter()

    def test_64_token_body_cannot_be_reserved_as_qa(self):
        body = make_body()
        body["max_tokens"] = 64
        with self._adapter() as ledger:
            ledger.claim_root(budget_v3.ROOTS[0])
            with self.assertRaises(transport_api.TransportError):
                ledger.reserve(budget_v3.ROOTS[0], "DISCOVERY", body)
            self.assertEqual(ledger.snapshot()["used"], 4)

    def test_eight_mocked_stage_attempts_fill_global_twelve_cap(self):
        fake = FakeStageTransport()
        with self._adapter() as ledger:
            for root_id in budget_v3.ROOTS[:2]:
                ledger.claim_root(root_id)
                for stage in budget_v3.STAGES:
                    ledger.execute_stage(root_id, stage, make_body(stage), fake)
                ledger.finish_root(root_id, "HOLD", "Mock budget accounting only; not a QA decision")
            self.assertEqual((ledger.snapshot()["used"], ledger.snapshot()["remaining"]), (12, 0))
            with self.assertRaises(budget_v3.BudgetHold):
                ledger.claim_root(budget_v3.ROOTS[2])
        self.assertEqual(fake.calls, 8)

    def test_executor_is_dry_run_by_default_and_live_holds_without_evidence(self):
        selection = (budget_v3.ARTIFACTS.parent / "natural_question_forms_fg_15root_extension_20261026_v1"
                     / "SOURCE_SELECTION_FINAL.json")
        with tempfile.TemporaryDirectory(prefix="qa-v3-dry-run-") as directory:
            state = Path(directory) / "experiment"
            manifest = executor_v3.initialize(state, selection)
            engine = executor_v3.ExecutorV3(state)
            results = [engine.run_root(root_id) for root_id in budget_v3.ROOTS]
            self.assertEqual(manifest["version"], budget_v3.VERSION)
            self.assertTrue(all(result["model_calls"] == 0 for result in results))
            with self.assertRaises(v2.ExecutionHold):
                executor_v3.ExecutorV3(state, mode="live")
            self.assertFalse((state / "attempts.sqlite3").exists())

    def test_v3_transport_live_path_is_release_disabled(self):
        with self._adapter() as ledger:
            with self.assertRaises(proxy_v3.ProxyTransportError):
                proxy_v3.ProxyTransportV3(ledger, mode="live",
                    credential_file=Path(self.root / "not-a-real-key"), capability={})


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful() or NETWORK_EVENTS:
        raise SystemExit(1)
    print(json.dumps({"tests": result.result.testsRun, "failures": len(result.result.failures),
        "errors": len(result.result.errors), "network_events": NETWORK_EVENTS,
        "real_api_calls": 0, "real_credentials_read": 0,
        "actual_authoritative_ledger_mutations": 0}, ensure_ascii=False))
