"""Independent fail-closed integration checks; all QA and service data are fixtures.

Network sockets are forbidden for every test. These tests neither invoke a model
nor assert empirical QA quality or actual service capability.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from . import base_qa_executor_v2 as executor
from . import base_qa_pipeline_v1 as qa
from . import qa_api_transport_v2 as transport
from .qa_api_budget_v2 import BudgetError, ROOTS
from .test_base_qa_pipeline_v1 import fixture, evidence, review_fixture


def selection_fixture(directory):
    rows = []
    for root_id in ROOTS:
        source, _, _ = fixture()
        source["root_id"] = root_id
        path = directory / (root_id + ".json")
        qa.write_new(path, source)
        rows.append({"root_id": root_id, **qa.legacy.file_binding(path)})
    path = directory / "SELECTION.json"
    qa.write_new(path, {"roots": rows})
    return path


def response_envelope(value):
    body = {"id": "offline-fixture", "model": transport.EXPECTED_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}}
    return {"http_status": 200, "body": body, "response_headers": {},
            "request_id": "offline-fixture",
            "raw_response_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest()}


class ScriptedMock:
    is_mock = True

    def __init__(self, state, *, fail_stage=None, failure=None, wrong_gold=False,
                 malformed_blind=False, uncertain_response=False, callback=None):
        self.state = Path(state)
        self.fail_stage, self.failure = fail_stage, failure
        self.wrong_gold, self.malformed_blind = wrong_gold, malformed_blind
        self.uncertain_response, self.callback = uncertain_response, callback
        self.sent = []

    def send(self, body):
        transport.validate_body(body)
        self.sent.append(copy.deepcopy(body))
        packet = json.loads(body["messages"][1]["content"])
        metadata = json.loads(body["messages"][0]["content"][len(transport.META_PREFIX):])
        stage = packet["stage"]
        if self.callback:
            self.callback(stage, packet, self)
        if stage == self.fail_stage:
            raise self.failure or TimeoutError("synthetic timeout")
        if self.uncertain_response:
            return {"http_status": 503, "body": {}}
        source, relation, candidate = fixture()
        source["root_id"] = packet["source"]["root_id"]
        if stage == "DISCOVER_RELATION":
            relation["author_id"] = metadata["author_id"]
            value = relation
        elif stage == "WRITE_NATURAL_QUESTION":
            candidate["root_id"] = source["root_id"]
            candidate["author_id"] = metadata["author_id"]
            candidate["relation_sha256"] = packet["relation_sha256"]
            value = candidate
        elif stage == "INDEPENDENT_SOLVE":
            if self.malformed_blind:
                value = {"reviewer_id": metadata["reviewer_id"], "decision": "SOLVED"}
            else:
                value = {"packet_sha256": qa.legacy.fingerprint(packet),
                    "reviewer_id": metadata["reviewer_id"], "decision": "SOLVED", "ambiguities": [],
                    "independent_gold": ["c2" if self.wrong_gold else "c1"],
                    "evidence": evidence(source, "T1", "T2"),
                    "reasoning_steps": ["Synthetic independently supplied reasoning"]}
        elif stage == "COMPARE_AFTER_BLIND_FREEZE":
            _, _, value = review_fixture(source, packet["relation"], packet["candidate"])
            value["blind_sha256"] = qa.legacy.fingerprint(packet["blind_solution"])
            value["reviewer_id"] = metadata["reviewer_id"]
        else:
            raise AssertionError("unexpected execution path: " + stage)
        return response_envelope(value)


class ExecutorAdversarialTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket, "socket", side_effect=AssertionError("real network is forbidden"))
        guard.start()
        self.addCleanup(guard.stop)
        tmp = tempfile.TemporaryDirectory(prefix="qa-executor-independent-")
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.selection = selection_fixture(self.directory)
        self.state = self.directory / "state"
        executor.initialize(self.state, self.selection)

    def make_executor(self, **kwargs):
        mock = ScriptedMock(self.state, **kwargs)
        run = executor.Executor(self.state, mode="mock", transport=mock, token_counter=lambda _: 100)
        self.addCleanup(run.ledger.close)
        return run, mock

    def test_normal_four_stage_path_and_blind_privacy(self):
        seen = []
        def boundary(stage, packet, _):
            if stage == "INDEPENDENT_SOLVE":
                forbidden = {"gold", "expected_answer", "relation", "item_analysis", "inference_steps", "necessity_groups"}
                self.assertFalse(set(packet) & forbidden)
                self.assertEqual(set(packet), {"stage", "language", "instruction", "candidate_id", "source", "question_type", "question", "options", "source_warnings"})
                self.assertFalse((self.state / "roots" / ROOTS[0] / "COMPARISON_PACKET.json").exists())
            if stage == "COMPARE_AFTER_BLIND_FREEZE":
                run = self.state / "roots" / ROOTS[0]
                frozen = qa._frozen(run, "BLIND_SOLUTION.json", "BLIND_SOLUTION_FREEZE.json")
                self.assertEqual(packet["blind_solution"], frozen)
            seen.append(stage)
        run, mock = self.make_executor(callback=boundary)
        result = run.run_root(ROOTS[0])
        self.assertEqual(result["status"], "PASS_STATIC_REVIEW", result)
        self.assertEqual(seen, ["DISCOVER_RELATION", "WRITE_NATURAL_QUESTION", "INDEPENDENT_SOLVE", "COMPARE_AFTER_BLIND_FREEZE"])
        self.assertEqual(run.ledger.snapshot()["attempt_count"], 4)
        for body in mock.sent:
            self.assertEqual(body["max_tokens"], 4096)
            self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])

    def test_gold_disagreement_stops_before_any_comparison(self):
        run, mock = self.make_executor(wrong_gold=True)
        result = run.run_root(ROOTS[0])
        self.assertEqual(result["status"], "HOLD", result)
        self.assertIn("GOLD_DISAGREEMENT", result["reason"])
        self.assertEqual(len(mock.sent), 3)
        root = self.state / "roots" / ROOTS[0]
        self.assertTrue((root / "BLIND_SOLUTION_FREEZE.json").exists())
        self.assertFalse((root / "COMPARISON_PACKET.json").exists())

    def test_invalid_blind_stops_without_repair_or_comparison(self):
        run, mock = self.make_executor(malformed_blind=True)
        result = run.run_root(ROOTS[0])
        self.assertEqual(result["status"], "HOLD", result)
        self.assertEqual(len(mock.sent), 3)
        self.assertFalse((self.state / "roots" / ROOTS[0] / "COMPARISON_PACKET.json").exists())

    def test_timeout_and_connection_failure_never_retry(self):
        for exc in (TimeoutError("synthetic"), ConnectionResetError("synthetic")):
            with self.subTest(error=type(exc).__name__):
                # Each subcase uses a different root, still one opportunity each.
                root = ROOTS[0] if isinstance(exc, TimeoutError) else ROOTS[1]
                run, mock = self.make_executor(fail_stage="DISCOVER_RELATION", failure=exc)
                result = run.run_root(root)
                self.assertEqual(result["status"], "HOLD", result)
                self.assertEqual(len(mock.sent), 1)
                self.assertEqual(run.ledger.snapshot()["attempts"][-1]["status"], "UNCERTAIN")
                with self.assertRaises(BudgetError):
                    run.run_root(root)
                self.assertEqual(len(mock.sent), 1)

    def test_uncertain_http_response_consumes_attempt_and_stops(self):
        run, mock = self.make_executor(uncertain_response=True)
        self.assertEqual(run.run_root(ROOTS[0])["status"], "HOLD")
        self.assertEqual(len(mock.sent), 1)
        self.assertEqual(run.ledger.snapshot()["attempt_count"], 1)

    def test_restart_reads_consumed_root_claim_and_attempt(self):
        first, _ = self.make_executor(fail_stage="DISCOVER_RELATION", failure=TimeoutError())
        first.run_root(ROOTS[0])
        first.ledger.close()
        reopened, second_mock = self.make_executor()
        self.assertEqual(reopened.ledger.snapshot()["attempt_count"], 1)
        with self.assertRaises(BudgetError):
            reopened.run_root(ROOTS[0])
        self.assertEqual(second_mock.sent, [])

    def test_three_successful_roots_exhaust_twelve_attempts(self):
        run, mock = self.make_executor()
        for root in ROOTS:
            self.assertEqual(run.run_root(root)["status"], "PASS_STATIC_REVIEW")
        snapshot = run.ledger.snapshot()
        self.assertEqual(snapshot["attempt_count"], 12)
        self.assertEqual(snapshot["remaining_attempts"], 0)
        self.assertEqual(len(mock.sent), 12)
        with self.assertRaises(BudgetError):
            run.run_root(ROOTS[0])
        self.assertEqual(len(mock.sent), 12)

    def test_default_dry_run_never_constructs_transport_or_ledger(self):
        mock = ScriptedMock(self.state)
        with patch.object(transport, "SingleAttemptHTTPTransport", side_effect=AssertionError("no HTTP adapter allowed")):
            run = executor.Executor(self.state, transport=mock, token_counter=lambda _: (_ for _ in ()).throw(AssertionError("no tokenizer")))
            result = run.run_root(ROOTS[0])
        self.assertEqual(result["status"], "DRY_RUN_ONLY")
        self.assertIsNone(run.ledger)
        self.assertFalse((self.state / "attempts.sqlite3").exists())
        self.assertEqual(mock.sent, [])

    def test_live_without_approval_capability_stops_before_transport(self):
        with patch.object(transport, "SingleAttemptHTTPTransport", side_effect=AssertionError("not authorized")):
            with self.assertRaises(executor.ExecutionHold):
                executor.Executor(self.state, mode="live")
        self.assertFalse((self.state / "attempts.sqlite3").exists())

    def test_copied_state_directory_cannot_reset_or_launch_live(self):
        copied = self.directory / "copied"
        shutil.copytree(self.state, copied)
        for mode in ("dry-run", "live"):
            with self.subTest(mode=mode):
                with self.assertRaises(executor.ExecutionHold):
                    executor.Executor(copied, mode=mode)

    def test_input_budget_violation_consumes_opportunity_but_never_sends(self):
        mock = ScriptedMock(self.state)
        run = executor.Executor(self.state, mode="mock", transport=mock, token_counter=lambda _: 999999)
        self.addCleanup(run.ledger.close)
        result = run.run_root(ROOTS[0])
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(mock.sent, [])
        self.assertEqual(run.ledger.snapshot()["attempt_count"], 0)
        with self.assertRaises(BudgetError):
            run.run_root(ROOTS[0])

    def test_unverified_4096_server_semantics_block_live_before_adapter(self):
        # Capability and approval files are synthetic assertions, not evidence
        # about the actual school service. Unknown assertions must fail closed.
        state = self.directory / "capability-state"
        profile = qa.read(executor.PROFILE)
        profile["live_state_directory"] = str(state)
        profile_path = self.directory / "synthetic-profile.json"
        qa.write_new(profile_path, profile)
        evidence_path = self.directory / "synthetic-evidence.txt"
        evidence_path.write_text("Synthetic test evidence only; no service claim.\n")
        with patch.object(executor, "PROFILE", profile_path):
            manifest = executor.initialize(state, self.selection)
            capability = {"endpoint": transport.ENDPOINT, "request_model": transport.MODEL,
                "response_model": transport.EXPECTED_MODEL, "output_parameter": "max_tokens",
                "max_output_tokens": 4096, "max_tokens_hard_limit_confirmed": False,
                "reasoning_in_output_budget_confirmed": True, "context_window_tokens": 65536,
                "tokenizer_compatible": True, "evidence_ref": "synthetic-evidence",
                "evidence_binding": qa.legacy.file_binding(evidence_path)}
            cap_path = self.directory / "synthetic-capability.json"
            qa.write_new(cap_path, capability)
            approval_path = self.directory / "synthetic-approval.json"
            qa.write_new(approval_path, {"approved": True,
                "experiment_sha256": qa.legacy.fingerprint(manifest), "max_attempts": 12,
                "max_output_tokens": 4096, "roots": list(ROOTS),
                "approval_reference": "synthetic fixture only; not user approval",
                "capability_binding": qa.legacy.file_binding(cap_path)})
            with patch.object(transport, "SingleAttemptHTTPTransport", side_effect=AssertionError("unverified service must not instantiate transport")):
                with self.assertRaises(executor.ExecutionHold):
                    executor.Executor(state, mode="live", authorization=approval_path, capability=cap_path)
        self.assertFalse((state / "attempts.sqlite3").exists())

    def test_disagreement_cannot_be_retried_by_new_executor(self):
        first, mock = self.make_executor(wrong_gold=True)
        self.assertEqual(first.run_root(ROOTS[0])["status"], "HOLD")
        self.assertEqual(len(mock.sent), 3)
        first.ledger.close()
        reopened, other_mock = self.make_executor()
        with self.assertRaises(BudgetError):
            reopened.run_root(ROOTS[0])
        self.assertEqual(reopened.ledger.snapshot()["attempt_count"], 3)
        self.assertEqual(other_mock.sent, [])

    def manual_prefix(self, run, root_id=ROOTS[0]):
        """Freeze local fixtures only; reserve/send no request."""
        run.ledger.claim_root(root_id)
        directory = self.state / "roots" / root_id
        directory.mkdir(parents=True)
        source, warnings = run._source(root_id)
        _, relation, candidate = fixture()
        relation["author_id"] = "p0-v2-author-" + root_id
        candidate["root_id"] = root_id
        candidate["author_id"] = relation["author_id"]
        candidate["relation_sha256"] = qa.legacy.fingerprint(relation)
        executor.freeze(directory, "RELATION", relation)
        executor.freeze(directory, "CANDIDATE", candidate)
        packet = run._blind_packet(source, warnings, candidate)
        executor.freeze(directory, "BLIND_PACKET", packet)
        return directory, source, relation, candidate, packet

    def test_direct_compare_call_cannot_bypass_missing_blind_freeze(self):
        run, mock = self.make_executor()
        directory, source, _, candidate, _ = self.manual_prefix(run)
        packet = {"stage": "COMPARE_AFTER_BLIND_FREEZE", "source": qa.source_view(source)}
        executor.freeze(directory, "COMPARISON_PACKET", packet)
        with self.assertRaises((executor.ExecutionHold, ValueError, FileNotFoundError)):
            run._call(ROOTS[0], "COMPARE", packet, {
                "reviewer_id": "p0-v2-reviewer-" + ROOTS[0],
                "candidate_sha256": qa.legacy.fingerprint(candidate),
                "blind_sha256": "0" * 64}, directory)
        self.assertEqual(mock.sent, [])
        self.assertEqual(run.ledger.snapshot()["attempt_count"], 0)

    def test_direct_blind_call_rejects_author_certificate_metadata(self):
        run, mock = self.make_executor()
        directory, _, relation, _, packet = self.manual_prefix(run)
        with self.assertRaises(executor.ExecutionHold):
            run._call(ROOTS[0], "BLIND", packet, {
                "reviewer_id": "p0-v2-reviewer-" + ROOTS[0],
                "model_id": transport.EXPECTED_MODEL,
                "relation_sha256": qa.legacy.fingerprint(relation)}, directory)
        self.assertEqual(mock.sent, [])
        self.assertEqual(run.ledger.snapshot()["attempt_count"], 0)

    def test_frozen_packet_drift_rejected_before_reservation(self):
        run, mock = self.make_executor()
        root_id = ROOTS[0]
        run.ledger.claim_root(root_id)
        directory = self.state / "roots" / root_id
        directory.mkdir(parents=True)
        source, warnings = run._source(root_id)
        packet = run._discovery_packet(source, warnings)
        executor.freeze(directory, "DISCOVERY_PACKET", packet)
        changed = copy.deepcopy(packet)
        changed["question"] = "Unfrozen instruction injection"
        with self.assertRaises(executor.ExecutionHold):
            run._call(root_id, "DISCOVERY", changed,
                      {"author_id": "p0-v2-author-" + root_id}, directory)
        # Tampering with both caller input and raw on-disk JSON still fails
        # because the independently stored receipt no longer matches.
        (directory / "DISCOVERY_PACKET.json").write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            run._call(root_id, "DISCOVERY", changed,
                      {"author_id": "p0-v2-author-" + root_id}, directory)
        self.assertEqual(mock.sent, [])
        self.assertEqual(run.ledger.snapshot()["attempt_count"], 0)

    def test_blind_identifier_is_neutral_and_not_author_controlled(self):
        run, _ = self.make_executor()
        _, source, _, candidate, packet = self.manual_prefix(run)
        candidate["candidate_id"] = "author-gold-c1-certificate-hint"
        actual = run._blind_packet(source, [], candidate)
        self.assertEqual(actual["candidate_id"], "p0-v2-blind-" + ROOTS[0])
        self.assertNotIn("author-gold", actual["candidate_id"])
        self.assertEqual(actual["candidate_id"], packet["candidate_id"])


if __name__ == "__main__":
    unittest.main()
