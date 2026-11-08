"""Offline end-to-end Mock tests. Fixtures are not dataset QA or model results."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import base_qa_executor_v2 as ex
from . import base_qa_pipeline_v1 as qa
from . import qa_api_transport_v2 as tr
from .test_base_qa_pipeline_v1 import fixture, fixture_source, review_fixture


def make_selection(directory):
    directory = Path(directory)
    rows = []
    for root in ex.ROOTS:
        source = fixture_source()
        source["root_id"] = root
        path = directory / (root + ".json")
        qa.write_new(path, source)
        rows.append({"root_id": root, **qa.legacy.file_binding(path)})
    selection = directory / "selection.json"
    qa.write_new(selection, {"roots": rows})
    return selection


def envelope(value, *, content=None, model=tr.EXPECTED_MODEL, finish="stop"):
    body = {"id": "mock-response-id", "model": model,
        "choices": [{"index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": json.dumps(value) if content is None else content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}}
    return {"http_status": 200, "body": body, "response_headers": {}, "request_id": "mock-request-id",
            "raw_response_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest()}


class ScriptedMock:
    is_mock = True

    def __init__(self, failure=None, on_send=None):
        self.failure, self.on_send = failure, on_send
        self.calls = []

    def send(self, body):
        tr.validate_body(body)
        packet = json.loads(body["messages"][1]["content"])
        metadata = json.loads(body["messages"][0]["content"].split("CALL_METADATA=", 1)[1])
        self.calls.append(copy.deepcopy(body))
        if self.on_send:
            self.on_send(body)
        stage = packet["stage"]
        if self.failure == "timeout": raise TimeoutError("synthetic fixture timeout")
        if self.failure == "disconnect": raise ConnectionResetError("synthetic fixture disconnect")
        if stage == "DISCOVER_RELATION":
            if self.failure == "no-safe": return envelope({"status": "NO_SAFE_QA", "reason": "Synthetic no-safe fixture"})
            _, relation, _ = fixture()
            relation["author_id"] = metadata["author_id"]
            return envelope(relation)
        if stage == "WRITE_NATURAL_QUESTION":
            _, _, candidate = fixture()
            candidate["root_id"] = packet["source"]["root_id"]
            candidate["author_id"] = metadata["author_id"]
            candidate["candidate_id"] = "mock-candidate-" + candidate["root_id"]
            candidate["relation_sha256"] = packet["relation_sha256"]
            if self.failure == "bad-author": candidate["gold"] = ["missing"]
            return envelope(candidate)
        if stage == "INDEPENDENT_SOLVE":
            if self.failure == "invalid-review": return envelope({}, content="not JSON")
            if self.failure == "incomplete-review": return envelope({"decision": "SOLVED"})
            source = packet["source"]
            blind = {"packet_sha256": metadata["packet_sha256"],
                "reviewer_id": metadata["reviewer_id"], "model_id": tr.EXPECTED_MODEL,
                "decision": "SOLVED", "independent_gold": ["c1"],
                "evidence": [{"turn_ref": t["turn_ref"], "speaker": t["speaker"], "quote": t["text"]}
                             for t in source["turns"][:2]],
                "reasoning_steps": ["Synthetic independent response fixture."], "ambiguities": []}
            if self.failure == "gold-conflict": blind["independent_gold"] = ["c2"]
            if self.failure == "ambiguous": blind["ambiguities"] = ["Synthetic ambiguity"]
            if self.failure == "insufficient": blind["decision"] = "HOLD"
            return envelope(blind)
        if stage == "COMPARE_AFTER_BLIND_FREEZE":
            blind = packet["blind_solution"]
            return envelope({"reviewer_id": blind["reviewer_id"], "decision": "PASS",
                "blind_sha256": metadata["blind_sha256"], "candidate_sha256": metadata["candidate_sha256"],
                "checks": {k: {"decision": "PASS", "rationale": "Synthetic comparison fixture."} for k in qa.CHECKS},
                "necessity_review": [{"group_id": g["group_id"], "removed_turn_refs": g["turn_refs"],
                    "target_underdetermined": True, "single_group_insufficient": True,
                    "no_remaining_bypass": True, "rationale": "Synthetic necessity fixture."}
                    for g in packet["relation"]["necessity_groups"]]})
        raise AssertionError("Execution did not enter new relation-first flow")


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.selection = make_selection(self.base)
        self.state = self.base / "state"
        ex.initialize(self.state, self.selection)

    def executor(self, failure=None, on_send=None):
        transport = ScriptedMock(failure, on_send)
        executor = ex.Executor(self.state, mode="mock", transport=transport, token_counter=lambda _: 100)
        self.addCleanup(executor.ledger.close)
        return executor, transport

    def test_default_dry_run_never_invokes_transport_or_counter_or_credentials(self):
        bomb = unittest.mock.Mock(side_effect=AssertionError("must not be called"))
        e = ex.Executor(self.state, transport=bomb, token_counter=bomb)
        result = e.run_root(ex.ROOTS[0])
        self.assertEqual(result["status"], "DRY_RUN_ONLY")
        self.assertFalse((self.state / "attempts.sqlite3").exists())
        bomb.assert_not_called()

    def test_normal_new_flow_and_durable_pre_send_budget(self):
        executor, transport = self.executor()
        def on_send(body):
            snapshot = executor.ledger.snapshot()
            self.assertEqual(snapshot["attempts"][-1]["status"], "RESERVED")
            self.assertEqual(snapshot["attempt_count"], len(transport.calls))
            self.assertEqual(body["max_tokens"], 4096)
        transport.on_send = on_send
        result = executor.run_root(ex.ROOTS[0])
        self.assertEqual(result["status"], "PASS_STATIC_REVIEW", result)
        self.assertEqual([json.loads(c["messages"][1]["content"])["stage"] for c in transport.calls],
            ["DISCOVER_RELATION", "WRITE_NATURAL_QUESTION", "INDEPENDENT_SOLVE", "COMPARE_AFTER_BLIND_FREEZE"])
        self.assertEqual(executor.ledger.snapshot()["attempt_count"], 4)
        run = self.state / "roots" / ex.ROOTS[0]
        for attempt in executor.ledger.snapshot()["attempts"]:
            stage = attempt["stage"]
            binding = qa.read(run / (stage + "_REQUEST_BINDING.json"))
            self.assertEqual(qa.legacy.fingerprint(binding), attempt["request_sha256"])
            self.assertEqual(binding["body_sha256"], qa.legacy.fingerprint(qa.read(run / (stage + "_REQUEST.json"))))
            self.assertEqual(binding["max_tokens"], 4096)
        self.assertEqual(ex.frozen(run, "FROZEN_QA")["decision"]["decision"], "PASS_STATIC_REVIEW")
        self.assertFalse(result["empirical_quality_established"])

    def test_blind_gold_conflict_frozen_but_no_comparison_constructed(self):
        executor, transport = self.executor("gold-conflict")
        result = executor.run_root(ex.ROOTS[0])
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(len(transport.calls), 3)
        run = self.state / "roots" / ex.ROOTS[0]
        self.assertEqual(ex.frozen(run, "BLIND_SOLUTION")["independent_gold"], ["c2"])
        self.assertFalse((run / "COMPARISON_PACKET.json").exists())
        self.assertFalse((run / "COMPARE_REQUEST.json").exists())

    def test_invalid_reviewer_and_ambiguity_stop_without_retry(self):
        for failure in ("invalid-review", "incomplete-review", "ambiguous", "insufficient"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "state"
                ex.initialize(state, self.selection)
                transport = ScriptedMock(failure)
                e = ex.Executor(state, mode="mock", transport=transport, token_counter=lambda _: 100)
                try:
                    self.assertEqual(e.run_root(ex.ROOTS[0])["status"], "HOLD")
                    self.assertEqual(len(transport.calls), 3)
                    self.assertFalse((state / "roots" / ex.ROOTS[0] / "COMPARISON_PACKET.json").exists())
                    self.assertEqual(e.ledger.snapshot()["attempt_count"], 3)
                    if failure == "invalid-review":
                        error = qa.read(state / "roots" / ex.ROOTS[0] / "BLIND_ERROR.json")
                        self.assertEqual(error["reason_code"], "invalid_or_ambiguous_json")
                        self.assertTrue(error["attempt_consumed"])
                finally: e.ledger.close()

    def test_timeouts_and_disconnections_consumed_once_and_restart_cannot_retry(self):
        for failure in ("timeout", "disconnect"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "state"
                ex.initialize(state, self.selection)
                transport = ScriptedMock(failure)
                e = ex.Executor(state, mode="mock", transport=transport, token_counter=lambda _: 100)
                self.assertEqual(e.run_root(ex.ROOTS[0])["status"], "HOLD")
                self.assertEqual(e.ledger.snapshot()["attempt_count"], 1)
                self.assertEqual(e.ledger.snapshot()["attempts"][0]["status"], "UNCERTAIN")
                e.ledger.close()
                restarted = ex.Executor(state, mode="mock", transport=transport, token_counter=lambda _: 100)
                try:
                    with self.assertRaises(ex.BudgetError): restarted.run_root(ex.ROOTS[0])
                    self.assertEqual(restarted.ledger.snapshot()["attempt_count"], 1)
                    self.assertEqual(len(transport.calls), 1)
                finally: restarted.ledger.close()

    def test_three_roots_twelve_max_and_no_second_opportunity(self):
        e, transport = self.executor()
        for root in ex.ROOTS:
            self.assertEqual(e.run_root(root)["status"], "PASS_STATIC_REVIEW")
        self.assertEqual(len(transport.calls), 12)
        self.assertEqual(e.ledger.snapshot()["attempt_count"], 12)
        with self.assertRaises(ex.BudgetError): e.run_root(ex.ROOTS[0])
        with self.assertRaises(ex.ExecutionHold): e.run_root("unapproved-root")
        self.assertEqual(len(transport.calls), 12)

    def test_blind_payload_excludes_author_and_compare_only_after_freeze(self):
        e, transport = self.executor()
        run = self.state / "roots" / ex.ROOTS[0]
        def on_send(body):
            packet = json.loads(body["messages"][1]["content"])
            if packet["stage"] == "INDEPENDENT_SOLVE":
                self.assertFalse((run / "COMPARISON_PACKET.json").exists())
                self.assertFalse((run / "BLIND_SOLUTION.json").exists())
                for field in ("gold", "relation", "expected_answer", "item_analysis", "certificate"):
                    self.assertNotIn(field, packet)
                self.assertEqual(len(body["messages"]), 2)
                metadata = json.loads(body["messages"][0]["content"].split("CALL_METADATA=", 1)[1])
                self.assertEqual(set(metadata), {"packet_sha256", "reviewer_id", "model_id"})
            elif packet["stage"] == "COMPARE_AFTER_BLIND_FREEZE":
                self.assertEqual(ex.frozen(run, "BLIND_SOLUTION"), packet["blind_solution"])
                self.assertTrue((run / "BLIND_SOLUTION_FREEZE.json").exists())
        transport.on_send = on_send
        self.assertEqual(e.run_root(ex.ROOTS[0])["status"], "PASS_STATIC_REVIEW")

    def test_no_safe_qa_is_final_and_no_quota_filler(self):
        e, transport = self.executor("no-safe")
        self.assertEqual(e.run_root(ex.ROOTS[0])["status"], "NO_SAFE_QA")
        self.assertEqual(len(transport.calls), 1)
        with self.assertRaises(ex.BudgetError): e.run_root(ex.ROOTS[0])

    def test_invalid_author_stops_without_review_or_repair(self):
        e, transport = self.executor("bad-author")
        self.assertEqual(e.run_root(ex.ROOTS[0])["status"], "HOLD")
        self.assertEqual(len(transport.calls), 2)

    def test_input_overflow_holds_before_attempt_or_transport(self):
        e, transport = self.executor()
        e.counter = lambda _: 1000000
        self.assertEqual(e.run_root(ex.ROOTS[0])["status"], "HOLD")
        self.assertEqual(e.ledger.snapshot()["attempt_count"], 0)
        self.assertEqual(transport.calls, [])

    def test_explicit_live_without_approval_or_service_proof_is_hold_before_network(self):
        with patch.object(tr.http.client, "HTTPSConnection", side_effect=AssertionError("NETWORK FORBIDDEN")) as network:
            with self.assertRaises(ex.ExecutionHold): ex.Executor(self.state, mode="live")
            network.assert_not_called()

    def test_f_and_g_discovery_inputs_identical(self):
        states = []
        for arm in ("F", "G"):
            state = self.base / ("state-" + arm)
            ex.initialize(state, self.selection, arm)
            e = ex.Executor(state, mode="mock", transport=ScriptedMock("no-safe"), token_counter=lambda _: 100)
            try:
                e.run_root(ex.ROOTS[0])
                states.append(ex.frozen(state / "roots" / ex.ROOTS[0], "DISCOVERY_PACKET"))
            finally: e.ledger.close()
        self.assertEqual(*states)


if __name__ == "__main__":
    unittest.main()
