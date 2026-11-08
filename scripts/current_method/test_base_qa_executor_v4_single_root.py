from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from . import base_qa_executor_v2 as v2
from . import base_qa_executor_v3_adapter as v3
from . import base_qa_executor_v4_single_root as v4
from . import base_qa_global_budget_adapter_v4 as budget
from . import base_qa_pipeline_v1 as qa
from . import base_qa_context_preflight_v4 as context_preflight
from . import base_qa_proxy_transport_v3 as proxy
from . import qa_api_transport_v2 as transport_api


FIXTURE_POLICY = {"schema": "v4_mock_fixture", "original_global_limit": 12,
                  "experiment_binding": {"path": "fixture", "bytes": 0, "sha256": "0" * 64},
                  "canonical_directory": "/fixture"}


def _source():
    return {"root_id": v4.ROOT_ID, "turns": [
        {"turn_ref": "turn-001", "scene_ref": "scene-1", "session_ref": "session-1",
         "speaker": "Alice", "text": "I booked train 10 for the 8:00 departure."},
        {"turn_ref": "turn-002", "scene_ref": "scene-1", "session_ref": "session-1",
         "speaker": "Bob", "text": "I booked train 10 for the 8:00 departure."},
    ]}


def _evidence(turn_ref, speaker, quote):
    return [{"turn_ref": turn_ref, "speaker": speaker, "quote": quote}]


def _relation(source):
    t1, t2 = source["turns"]
    f1, f2 = "fact-alice", "fact-bob"
    return {"relation_id": "mock-relation", "author_id": "p0-v4-author-" + v4.ROOT_ID,
        "target": "Identify the repeated departure detail.", "relation_type": "cross_turn_confirmation",
        "why_ask": "Both statements must be checked together.",
        "expected_answer": ["The train booking was for train 10 at 8:00."],
        "requires_deterministic_check": False,
        "facts": [
            {"fact_id": f1, "claim": "Alice booked train 10 at 8:00.",
             "evidence": _evidence(t1["turn_ref"], t1["speaker"], t1["text"]),
             "evidence_turn_ids": [t1["turn_ref"]]},
            {"fact_id": f2, "claim": "Bob confirms the same booking detail.",
             "evidence": _evidence(t2["turn_ref"], t2["speaker"], t2["text"]),
             "evidence_turn_ids": [t2["turn_ref"]]},
        ],
        "inference_steps": [{"step_id": "join", "inputs": [f1, f2],
            "conclusion": "The repeated statements jointly identify one departure detail."}],
        "necessity_groups": [
            {"group_id": "g1", "fact_ids": [f1], "turn_refs": [t1["turn_ref"]],
             "why_necessary": "The first statement supplies one side of the match.",
             "removal_effect": "Removing it leaves only Bob's detail."},
            {"group_id": "g2", "fact_ids": [f2], "turn_refs": [t2["turn_ref"]],
             "why_necessary": "The second statement supplies the confirming side.",
             "removal_effect": "Removing it leaves only Alice's detail."},
        ]}


def _candidate(source, relation):
    t1, t2 = source["turns"]
    return {"candidate_id": "mock-candidate", "root_id": v4.ROOT_ID,
        "author_id": relation["author_id"], "relation_sha256": qa.legacy.fingerprint(relation),
        "question_type": "single_choice",
        "question": "Which departure detail is stated by both speakers?",
        "options": {"A": "The train booking was for train 10 at 8:00.",
                    "B": "The train booking was for train 10 at 9:00."},
        "gold": ["A"], "ordering_edges": [], "deterministic_checks": [],
        "item_analysis": {
            "A": {"status": "supported", "answer_value": relation["expected_answer"][0],
                "rationale": "Both turns state train 10 and 8:00.",
                "evidence": _evidence(t1["turn_ref"], t1["speaker"], t1["text"]) +
                            _evidence(t2["turn_ref"], t2["speaker"], t2["text"])},
            "B": {"status": "contradicted", "answer_value": "The train booking was for train 10 at 9:00.",
                "rationale": "Neither quoted statement gives 9:00.",
                "error_reason": "The time conflicts with the two source statements.",
                "evidence": _evidence(t2["turn_ref"], t2["speaker"], t2["text"])},
        }}


class FixtureLedger:
    """Workflow-only fake that never touches SQLite or a network."""
    mode = "mock"

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir).resolve()
        self.stages = []
        self.terminal = None

    def claim_root(self, root_id):
        if root_id != v4.ROOT_ID or self.stages:
            raise budget.BudgetHold("fixture root already claimed")

    def execute_stage(self, root_id, stage, body, model):
        if root_id != v4.ROOT_ID or stage != budget.STAGES[len(self.stages)]:
            raise budget.BudgetHold("fixture stage order")
        self.stages.append(stage)
        return model.send(body, None), {
            "returned_model": transport_api.EXPECTED_MODEL, "raw_response_sha256": "a" * 64,
            "request_id": "mock-request", "http_status": 200, "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    def finish_root(self, root_id, status, reason=""):
        self.terminal = (root_id, status, reason)


class WorkflowTransport:
    def __init__(self, blind_answer_override=None):
        self.stages = []
        self.blind_packet_keys = None
        self.blind_answer_override = blind_answer_override

    @staticmethod
    def _reply(value):
        return value, {
            "returned_model": transport_api.EXPECTED_MODEL,
            "raw_response_sha256": "a" * 64,
            "request_id": "mock-request",
            "http_status": 200,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def send(self, body, reservation):
        transport_api.validate_body(body)
        self.asserted_body = body
        packet = json.loads(body["messages"][1]["content"])
        stage = packet["stage"]
        self.stages.append(stage)
        source = _source()
        relation = _relation(source)
        candidate = _candidate(source, relation)
        if stage == "DISCOVER_RELATION":
            return self._reply(relation)
        if stage == "WRITE_NATURAL_QUESTION":
            return self._reply(candidate)
        if stage == "INDEPENDENT_SOLVE":
            self.blind_packet_keys = set(packet)
            answer_label = next(key for key, value in packet["options"].items()
                                if value == candidate["options"]["A"])
            return self._reply({"packet_sha256": qa.legacy.fingerprint(packet),
                "reviewer_id": "p0-v4-reviewer-" + v4.ROOT_ID,
                "model_id": transport_api.EXPECTED_MODEL, "decision": "SOLVED",
                "independent_gold": [self.blind_answer_override or answer_label], "ambiguities": [],
                "evidence": _evidence("turn-001", "Alice", source["turns"][0]["text"]) +
                            _evidence("turn-002", "Bob", source["turns"][1]["text"]),
                "reasoning_steps": ["Alice states train 10 at 8:00.",
                                     "Bob confirms train 10 at 8:00."]})
        if stage == "COMPARE_AFTER_BLIND_FREEZE":
            rel = packet["relation"]
            return self._reply({"blind_sha256": qa.legacy.fingerprint(packet["blind_solution"]),
                "candidate_sha256": qa.legacy.fingerprint(packet["candidate"]),
                "reviewer_id": "p0-v4-reviewer-" + v4.ROOT_ID, "decision": "PASS",
                "checks": {name: {"decision": "PASS", "rationale": "Mock static check."}
                           for name in qa.CHECKS},
                "necessity_review": [{"group_id": group["group_id"],
                    "removed_turn_refs": group["turn_refs"], "target_underdetermined": True,
                    "single_group_insufficient": True, "no_remaining_bypass": True,
                    "rationale": "Mock counterfactual review."}
                    for group in rel["necessity_groups"]]})
        raise AssertionError("unexpected workflow stage")


def _create_fixture_ledger(path, baseline):
    policy = dict(FIXTURE_POLICY)
    policy["canonical_directory"] = str(path.parent.resolve())
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE policy (singleton INTEGER PRIMARY KEY CHECK(singleton=1), value TEXT NOT NULL)")
        db.execute("CREATE TABLE attempts (attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, purpose TEXT NOT NULL UNIQUE, status TEXT NOT NULL CHECK(status IN ('RESERVED','SUCCEEDED','REJECTED','UNCERTAIN')), reserved_utc TEXT NOT NULL, settled_utc TEXT, request_sha256 TEXT NOT NULL, max_tokens INTEGER NOT NULL, receipt TEXT)")
        db.execute("CREATE TRIGGER global12 BEFORE INSERT ON attempts WHEN (SELECT COUNT(*) FROM attempts)>=12 BEGIN SELECT RAISE(ABORT,'GLOBAL_12_ATTEMPT_LIMIT'); END")
        db.execute("CREATE TRIGGER no_attempt_delete BEFORE DELETE ON attempts BEGIN SELECT RAISE(ABORT,'NO_REFUND_OR_RESET'); END")
        db.execute("CREATE TRIGGER immutable_attempt_identity BEFORE UPDATE OF purpose,reserved_utc,request_sha256,max_tokens ON attempts BEGIN SELECT RAISE(ABORT,'IMMUTABLE_ATTEMPT'); END")
        db.execute("INSERT INTO policy(singleton,value) VALUES(1,?)", (budget._canonical(policy),))
        for row in baseline:
            db.execute("INSERT INTO attempts(attempt_id,purpose,status,reserved_utc,settled_utc,request_sha256,max_tokens,receipt) VALUES(?,?,?,?,?,?,?,?)",
                (row["attempt_id"], row["purpose"], row["status"], row["reserved_utc"],
                 row["settled_utc"], row["request_sha256"], row["max_tokens"], "{}"))
    return policy


def _make_request(stage="DISCOVERY"):
    packet = {"stage": stage, "fixture": True}
    return transport_api.build_body(packet, {"packet_sha256": qa.legacy.fingerprint(packet),
        "author_id": "p0-v4-author-" + v4.ROOT_ID})


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qa-v4-single-root-")
        self.root = Path(self.temp.name)
        self.source_path = self.root / "source.json"
        self.warning_path = self.root / "warnings.json"
        self.selection_path = self.root / "selection.json"
        self.state = self.root / "state"
        self.source_path.write_text(json.dumps(_source(), ensure_ascii=False), encoding="utf-8")
        self.warning_path.write_text("[]\n", encoding="utf-8")
        source_binding = qa.legacy.file_binding(self.source_path)
        self.selection_path.write_text(json.dumps({"roots": [{"root_id": v4.ROOT_ID,
            "path": str(self.source_path), "sha256": source_binding["sha256"],
            "bytes": source_binding["bytes"], "warnings_path": str(self.warning_path)}]},
            ensure_ascii=False), encoding="utf-8")
        self.manifest = v4.initialize(self.state, self.selection_path)
        self.ledger_path = self.root / "fixture-ledger.sqlite3"
        self.policy = _create_fixture_ledger(self.ledger_path, self.manifest["baseline_attempts"])
        self.ledger = self._adapter()

    def tearDown(self):
        if getattr(self, "ledger", None) is not None:
            self.ledger.close()
        self.temp.cleanup()

    def _adapter(self):
        return budget.GlobalBudgetAdapterV4(mode="mock", ledger_path=self.ledger_path,
            state_dir=self.state, expected_policy=self.policy,
            baseline_attempts=self.manifest["baseline_attempts"])

    def test_manifest_and_executor_are_single_root_four_stage_dry_run(self):
        self.assertEqual(self.manifest["roots"], [v4.ROOT_ID])
        self.assertEqual(self.manifest["maximum_attempts"], 4)
        self.assertEqual(self.manifest["stages"], ["DISCOVERY", "AUTHOR", "BLIND", "COMPARE"])
        engine = v4.ExecutorV4(self.state)
        self.assertEqual(engine.run_root(v4.ROOT_ID)["planned_stages"], list(budget.STAGES))
        with self.assertRaises(v2.ExecutionHold):
            engine.run_root("cc-train-episode-24719")
        with self.assertRaises(v2.ExecutionHold):
            v4.ExecutorV4(self.state, mode="live")
        self.assertEqual(self.ledger.snapshot()["used"], 4)

    def test_request_body_is_explicit_and_reserved_before_mock_send(self):
        class Capture:
            def send(inner, body, reservation):
                self.assertEqual(body["model"], "tju-llm")
                self.assertEqual(body["max_tokens"], 4096)
                self.assertEqual(body["response_format"], {"type": "json_object"})
                self.assertIs(body["stream"], False)
                self.assertEqual(body["n"], 1)
                self.assertEqual(len(body["messages"]), 2)
                db = sqlite3.connect(self.ledger_path)
                row = db.execute("SELECT status,max_tokens FROM attempts WHERE attempt_id=?",
                                 (reservation.attempt_id,)).fetchone()
                db.close()
                self.assertEqual(row, ("RESERVED", 4096))
                return {"status": "NO_SAFE_QA", "reason": "fixture"}, {
                    "returned_model": transport_api.EXPECTED_MODEL, "raw_response_sha256": "a" * 64,
                    "request_id": "fixture", "http_status": 200, "finish_reason": "stop",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        self.ledger.claim_root(v4.ROOT_ID)
        self.ledger.execute_stage(v4.ROOT_ID, "DISCOVERY", _make_request(), Capture())
        self.assertEqual(self.ledger.snapshot()["v4_used"], 1)

    def test_mock_proxy_passes_explicit_request_fields_once(self):
        self.ledger.claim_root(v4.ROOT_ID)
        body = _make_request()
        raw = json.dumps({"id": "mock-completion", "model": transport_api.EXPECTED_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"status":"NO_SAFE_QA","reason":"mock"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
            separators=(",", ":")).encode("utf-8")
        captured = {"connections": 0, "requests": 0}

        class Response:
            status = 200
            def read(self, _limit):
                return raw
            def getheader(self, name):
                return "mock-request-id" if name == "x-request-id" else None

        class Connection:
            def connect(self):
                captured["connects"] = captured.get("connects", 0) + 1
            def request(self, method, path, *, body, headers):
                captured["requests"] += 1
                captured["method"] = method
                captured["path"] = path
                captured["body"] = body
                captured["headers"] = headers
                with sqlite3.connect(self_outer.ledger_path) as db:
                    row = db.execute("SELECT status,max_tokens FROM attempts WHERE purpose LIKE ?",
                                     (budget.NAMESPACE + "%",)).fetchone()
                self_outer.assertEqual(row, ("RESERVED", 4096))
            def getresponse(self):
                return Response()
            def close(self):
                captured["closed"] = True

        self_outer = self
        transport = proxy.ProxyTransportV3(self.ledger, mode="mock",
            proxy_environment={"HTTPS_PROXY": "http://127.0.0.1:7897"},
            connection_factory=lambda _trace: (captured.__setitem__("connections",
                captured["connections"] + 1) or Connection()),
            credential_reader=lambda: "mock-credential")
        self.ledger.execute_stage(v4.ROOT_ID, "DISCOVERY", body, transport)
        self.assertEqual(captured["connections"], 1)
        self.assertEqual(captured["requests"], 1)
        self.assertEqual(captured["connects"], 1)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/api/v3/chat/completions")
        self.assertEqual(captured["body"], transport_api._compact(body).encode("utf-8"))
        self.assertEqual(json.loads(captured["body"]), body)
        self.assertEqual(captured["headers"]["Content-Type"], "application/json")
        self.assertTrue(captured["closed"])
        self.assertEqual(transport.trace["retry_count"], 0)

    def test_single_root_four_attempt_limit_and_global_budget(self):
        with self.assertRaises(budget.BudgetHold):
            self.ledger.claim_root("cc-train-episode-24719")
        self.ledger.claim_root(v4.ROOT_ID)
        with self.assertRaises(budget.BudgetHold):
            self.ledger.reserve(v4.ROOT_ID, "AUTHOR", _make_request("AUTHOR"))
        self.assertEqual(self.ledger.snapshot()["v4_used"], 0)
        class Capture:
            def send(_, body, reservation):
                return {"stage": reservation.stage}, {
                    "returned_model": transport_api.EXPECTED_MODEL, "raw_response_sha256": "b" * 64,
                    "request_id": "fixture", "http_status": 200, "finish_reason": "stop",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        for stage in budget.STAGES:
            self.ledger.execute_stage(v4.ROOT_ID, stage, _make_request(stage), Capture())
        with self.assertRaises(budget.BudgetHold):
            self.ledger.reserve(v4.ROOT_ID, "COMPARE", _make_request("COMPARE"))
        snapshot = self.ledger.snapshot()
        self.assertEqual((snapshot["used"], snapshot["remaining"], snapshot["v4_used"],
                          snapshot["v4_remaining"]), (8, 4, 4, 0))

    def test_failed_request_consumes_once_and_cannot_retry(self):
        self.ledger.claim_root(v4.ROOT_ID)
        class Fails:
            calls = 0
            def send(inner, body, reservation):
                inner.calls += 1
                raise TimeoutError("mock timeout")
        transport = Fails()
        with self.assertRaises(TimeoutError):
            self.ledger.execute_stage(v4.ROOT_ID, "DISCOVERY", _make_request(), transport)
        with self.assertRaises(budget.BudgetHold):
            self.ledger.execute_stage(v4.ROOT_ID, "DISCOVERY", _make_request(), transport)
        self.assertEqual(transport.calls, 1)
        self.assertEqual(self.ledger.snapshot()["used"], 5)

    def test_restart_does_not_reset_root_opportunity_or_budget(self):
        self.ledger.claim_root(v4.ROOT_ID)
        class Capture:
            def send(_, body, reservation):
                return {"ok": True}, {"returned_model": transport_api.EXPECTED_MODEL,
                    "raw_response_sha256": "c" * 64, "http_status": 200, "finish_reason": "stop",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        self.ledger.execute_stage(v4.ROOT_ID, "DISCOVERY", _make_request(), Capture())
        self.ledger.close()
        self.ledger = self._adapter()
        with self.assertRaises(budget.BudgetHold):
            self.ledger.claim_root(v4.ROOT_ID)
        self.assertEqual(self.ledger.snapshot()["used"], 5)

    def test_full_workflow_order_and_blind_gold_isolation(self):
        model = WorkflowTransport()
        engine = v4.ExecutorV4(self.state, mode="mock", ledger=self.ledger,
                               transport=model, token_counter=lambda _: 100,
                               context_window_tokens=10000)
        result = engine.run_root(v4.ROOT_ID)
        self.assertEqual(result["status"], "PASS_STATIC_REVIEW")
        recorded_stages = [row["purpose"].rsplit(":", 1)[-1]
                           for row in self.ledger.snapshot()["attempts"]
                           if row["purpose"].startswith(budget.NAMESPACE)]
        self.assertEqual(recorded_stages, ["DISCOVERY", "AUTHOR", "BLIND", "COMPARE"])
        self.assertEqual(model.stages, ["DISCOVER_RELATION", "WRITE_NATURAL_QUESTION",
                                        "INDEPENDENT_SOLVE", "COMPARE_AFTER_BLIND_FREEZE"])
        self.assertEqual(model.blind_packet_keys,
            {"stage", "language", "instruction", "candidate_id", "source",
             "question_type", "question", "options", "source_warnings"})
        blind = json.loads((self.state / "roots" / v4.ROOT_ID / "BLIND_REQUEST.json").read_text())
        blind_packet = json.loads(blind["messages"][1]["content"])
        for forbidden in ("gold", "expected_answer", "relation", "certificate", "item_analysis"):
            self.assertNotIn(forbidden, blind_packet)
        self.assertEqual(self.ledger.snapshot()["v4_used"], 4)

    def test_context_preflight_builds_each_stage_with_runtime_caps(self):
        result = context_preflight.inspect_context(
            self.selection_path,
            token_counter=lambda body: len(body["messages"][1]["content"]))
        self.assertEqual([row["stage"] for row in result["stages"]], list(budget.STAGES))
        self.assertTrue(all(row["sample_input_tokens_local"] > 0 for row in result["stages"]))
        self.assertTrue(all(row["sample_within_v4_input_limit"] for row in result["stages"]))
        self.assertTrue(all(row["requested_max_output_tokens"] == 4096 for row in result["stages"]))
        self.assertEqual(result["schema_limit_finding"]["status"],
                         "NO_FINITE_SCHEMA_MAX_FOR_DEPENDENT_STAGES")
        self.assertFalse(result["service_capacity"]["send_size_proven_safe"])

    def test_blind_gold_conflict_stops_before_comparison(self):
        model = WorkflowTransport(blind_answer_override="B")
        engine = v4.ExecutorV4(self.state, mode="mock", ledger=self.ledger,
                               transport=model, token_counter=lambda _: 100,
                               context_window_tokens=10000)
        result = engine.run_root(v4.ROOT_ID)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(model.stages, ["DISCOVER_RELATION", "WRITE_NATURAL_QUESTION",
                                        "INDEPENDENT_SOLVE"])
        self.assertEqual(self.ledger.snapshot()["v4_used"], 3)

    def test_input_context_overflow_stops_before_reservation(self):
        model = WorkflowTransport()
        def count_input(body):
            packet = json.loads(body["messages"][1]["content"])
            return (v4.INPUT_LIMITS["AUTHOR"] + 1
                    if packet["stage"] == "WRITE_NATURAL_QUESTION" else 100)
        engine = v4.ExecutorV4(self.state, mode="mock", ledger=self.ledger,
                               transport=model, token_counter=count_input,
                               context_window_tokens=100000)
        result = engine.run_root(v4.ROOT_ID)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(self.ledger.snapshot()["used"], 5)
        self.assertEqual(self.ledger.snapshot()["v4_used"], 1)
        self.assertEqual(model.stages, ["DISCOVER_RELATION"])

    def test_live_budget_adapter_stays_disabled_and_canonical_ledger_read_only(self):
        path = budget.CANONICAL_LEDGER
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with budget.GlobalBudgetAdapterV4(mode="inspect") as inspector:
            snapshot = inspector.snapshot()
        with self.assertRaises(budget.BudgetHold):
            budget.GlobalBudgetAdapterV4(mode="live", baseline_attempts=self.manifest["baseline_attempts"])
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual((snapshot["used"], snapshot["remaining"]), (4, 8))
        self.assertEqual(before, after)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"tests": result.result.testsRun, "failures": len(result.result.failures),
        "errors": len(result.result.errors), "network_calls": 0, "real_api_calls": 0,
        "solver_runs": 0, "training_runs": 0, "authoritative_ledger_mutations": 0}, ensure_ascii=False))
