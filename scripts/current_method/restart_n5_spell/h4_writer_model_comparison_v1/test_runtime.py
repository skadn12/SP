"""Offline tests of the real shared Writer route with synthetic transport outputs."""
import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..backends import BackendResponse
from ..core import atomic_json, atomic_text
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import expanded_fixture
from ..h4_r02_ordering_temporal_contract_v1.test_runtime import synthetic_candidates
from . import runtime
from .transport import ComparisonTransportError, build_payload


def seed_run(run):
    from .constants import SCHEDULE
    profiles = {}
    for arm, model in zip(runtime.ARMS, ("mock-qwen", "mock-strong")):
        profiles[arm] = {"model": model, "base_url": "https://invalid.test/v1",
                         "endpoint": "https://invalid.test/v1/chat/completions",
                         "request_parameters": {"temperature": 0.1, "max_tokens": 8192},
                         "output_limit": 8192}
    atomic_json(run / "profiles.json", profiles)
    atomic_json(run / "schedule.json", SCHEDULE)
    atomic_json(run / "execution_contract.json", {"planned_primary_calls": 12,
        "maximum_repair_calls": 12, "maximum_logical_calls": 24, "maximum_http_attempts": 24})
    atomic_json(run / "audit/frozen_manifest.json", {"fixture": "SYNTHETIC_NO_NETWORK"})
    for row in SCHEDULE:
        shared = run / "inputs/shared" / row["root_id"]
        expanded = expanded_fixture()
        expanded["root_id"] = row["root_id"]
        contracts = [{"intent_id": f"intent_{i}", "intent_type": "state_transition",
                      "discriminative_variable": f"Synthetic fixture {i}",
                      "required_evidence_groups": {"before_state": ["D001"], "trigger": ["D002"], "after_state": ["D003"]}}
                     for i in range(1, 4)]
        atomic_json(shared / "expanded_dialogue_clean.json", expanded)
        atomic_json(shared / "contracts.json", contracts)
        prompt = build_prompt(expanded, contracts[row["raw_index"]], ordering_policy=runtime.POLICY)["prompt"]
        target = run / "policy_snapshot/primary_requests" / row["arm_id"] / row["root_id"] / row["slot_id"]
        atomic_text(target / "prompt.txt", prompt)
        atomic_json(target / "request.json", {"payload": build_payload(profiles[row["arm_id"]], prompt)})


class Client:
    def __init__(self, row, profile, calls, reply):
        self.row, self.calls, self.reply = row, calls, reply
        self.config = SimpleNamespace(model=profile["model"], base_url=profile["base_url"])

    def complete(self, **kwargs):
        self.calls.append({**self.row, **kwargs, "model": self.config.model})
        raw = self.reply(kwargs) if self.reply else synthetic_candidates(kwargs["prompt"])
        if isinstance(raw, BaseException):
            raise raw
        return BackendResponse(raw if isinstance(raw, str) else json.dumps(raw), self.config.model,
                               {"usage": None, "actual_model": self.config.model})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        seed_run(self.run)
        self.calls = []
        self.capacities = []
        blocker = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def execute(self, reply=None):
        return runtime.execute_schedule(self.run,
            client_factory=lambda row, profile, callback: Client(row, profile, self.calls, reply),
            verifier=lambda: None,
            capacity=lambda *args: self.capacities.append(args))

    def test_paired_same_prompts_models_and_resume(self):
        outcome = self.execute()
        self.assertEqual(outcome["primary_calls"], 12)
        self.assertEqual(outcome["repair_calls"], 0)
        self.assertEqual(len(runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")), 48)
        for a, b in zip(self.calls[::2], self.calls[1::2]):
            self.assertEqual(a["prompt"], b["prompt"])
            self.assertNotEqual(a["model"], b["model"])
            self.assertIn("ORDERING EVENT CONTRACT", a["prompt"])
        self.assertEqual(self.execute(), outcome)
        self.assertEqual(len(self.calls), 12)

    def test_invalid_json_keeps_original_one_repair_per_slot(self):
        def reply(request):
            return synthetic_candidates(request["prompt"]) if "REPAIR THE FOLLOWING BATCH" in request["prompt"] else "{invalid"
        result = self.execute(reply)
        self.assertEqual((result["primary_calls"], result["repair_calls"]), (12, 12))
        self.assertEqual(len(self.calls), 24)
        self.assertEqual(len(self.capacities), 24)
        self.assertTrue(all(x["stage"] == "repair" for x in runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")))

    def test_graph_failure_keeps_batch_discard_rule_for_both_models(self):
        def reply(request):
            raw = synthetic_candidates(request["prompt"])
            raw["candidates"][-1]["temporal_relations"] = []
            return raw
        result = self.execute(reply)
        self.assertEqual((result["primary_calls"], result["repair_calls"]), (12, 12))
        self.assertEqual(runtime.rows(self.run / "reports/runtime_final_candidates.jsonl"), [])

    def test_unknown_completion_stops_before_repair_and_resume(self):
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(lambda _: TimeoutError("synthetic unknown"))
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(runtime.rows(self.run / "reports/runtime_stage_records.jsonl")), 1)

    def test_known_http_rejection_stops_without_content_repair_or_resume(self):
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(lambda _: ComparisonTransportError("http_error", {"http_status": 429}))
        self.assertEqual(len(self.calls), 1)
        calls = runtime.rows(self.run / "audit/logical_calls.jsonl")
        self.assertEqual([row["stage"] for row in calls], ["primary"])
        self.assertTrue((self.run / "audit/execution_stop.json").is_file())
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.calls), 1)

    def test_technical_failure_in_repair_stops_without_another_slot(self):
        def reply(request):
            if "REPAIR THE FOLLOWING BATCH" in request["prompt"]:
                return ComparisonTransportError("http_error", {"http_status": 400})
            return "{invalid"
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(reply)
        self.assertEqual(len(self.calls), 2)
        calls = runtime.rows(self.run / "audit/logical_calls.jsonl")
        self.assertEqual([row["stage"] for row in calls], ["primary", "repair"])
        self.assertEqual(len({row["order"] for row in calls}), 1)
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.calls), 2)

    def test_context_failure_stops_before_reservation_and_send(self):
        def capacity(*args):
            raise runtime.ExecutionStopped("MOCK_CONTEXT_EXCEEDED")
        with self.assertRaises(runtime.ExecutionStopped):
            runtime.execute_schedule(self.run,
                client_factory=lambda row, profile, callback: Client(row, profile, self.calls, None),
                verifier=lambda: None, capacity=capacity)
        self.assertFalse(self.calls)
        self.assertFalse(list(self.run.glob("**/ledger/requests/*.json")))

    def test_incomplete_checkpoint_reuses_completed_response_without_new_request(self):
        original = runtime.generation.execute_writer_batch
        def interrupt(**kwargs):
            original(**kwargs)
            raise KeyboardInterrupt()
        with patch.object(runtime.generation, "execute_writer_batch", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.execute()
        self.assertEqual(len(self.calls), 1)
        self.execute()
        self.assertEqual(len(self.calls), 12)

    def test_candidate_checkpoint_tamper_blocks(self):
        self.execute()
        checkpoint = next(self.run.glob("**/cells/*.json"))
        checkpoint.write_text("{}")
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.calls), 12)

    def test_cached_raw_drift_cannot_trigger_repair(self):
        original = runtime.generation.execute_writer_batch
        def interrupt(**kwargs):
            original(**kwargs)
            raise KeyboardInterrupt()
        with patch.object(runtime.generation, "execute_writer_batch", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.execute()
        response = next(self.run.glob("**/raw/backend_responses/*.txt"))
        response.write_text("tampered cached response")
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.calls), 1)

    def test_deleted_reservation_cannot_restore_budget(self):
        self.execute()
        next(self.run.glob("**/ledger/requests/*.json")).unlink()
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.calls), 12)

    def test_shared_parameter_extension_preserves_defaults_and_can_record_omitted_temperature(self):
        profiles = runtime.read(self.run / "profiles.json")
        profile = profiles[runtime.ARMS[1]]
        profile["request_parameters"].pop("temperature")
        profile["request_parameters"]["max_tokens"] = 4096
        profile["output_limit"] = 4096
        atomic_json(self.run / "profiles.json", profiles)
        for path in (self.run / "policy_snapshot/primary_requests" / runtime.ARMS[1]).glob("*/*/request.json"):
            prompt = path.with_name("prompt.txt").read_text()
            atomic_json(path, {"payload": build_payload(profile, prompt)})
        self.execute()
        for path in (self.run / "arms" / runtime.ARMS[1]).glob("**/ledger/requests/*.json"):
            metadata = runtime.read(path)["request_metadata"]
            self.assertIsNone(metadata["temperature"])
            self.assertEqual(metadata["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
