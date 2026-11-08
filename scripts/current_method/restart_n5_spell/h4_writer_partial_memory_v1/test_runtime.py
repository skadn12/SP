"""No-network stop/accounting tests around the shared Writer runtime."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..backends import BackendResponse
from ..core import atomic_json, atomic_text
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import expanded_fixture
from ..h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
from ..h4_r02_ordering_temporal_contract_v1.test_runtime import synthetic_candidates
from ..h4_writer_model_comparison_v1.transport import build_payload
from . import runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name)
        self.profile = {"arm_id": "STRONG_TEMPORAL", "model": "mock", "base_url": "https://invalid.example/v1",
            "endpoint": "https://invalid.example/v1/chat/completions", "timeout_seconds": 1,
            "output_limit": 8192, "request_parameters": {"temperature": 0.1, "max_tokens": 8192,
            "response_format": {"type": "json_object"}}, "unresolved": ["capacity"]}
        atomic_json(self.run / "profile.json", self.profile)
        expanded = expanded_fixture()
        contracts = [{"intent_id": f"intent_{i}", "required_evidence_groups": {}} for i in range(12)]
        folder = self.run / "inputs/TRANSCRIPT_ONLY/test-root"
        atomic_json(folder / "expanded.json", expanded)
        atomic_json(folder / "contracts.json", contracts)
        schedule = []
        for index, contract in enumerate(contracts):
            row = {"scope": "memory_assist", "arm_id": "TRANSCRIPT_ONLY", "source_mode": "TRANSCRIPT_ONLY",
                   "root_id": "test-root", "slot_id": contract["intent_id"], "raw_index": index, "order": index+1}
            schedule.append(row)
            prompt = build_prompt(expanded, contract, ordering_policy=runtime.POLICY)["prompt"]
            target = self.run / "policy_snapshot/primary_requests/TRANSCRIPT_ONLY/test-root" / row["slot_id"]
            atomic_text(target / "prompt.txt", prompt)
            atomic_json(target / "request.json", {"payload": build_payload(self.profile, prompt)})
        atomic_json(self.run / "schedule.json", schedule)
        guard = patch("scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.prepare.verify", return_value={"status": "PASS"})
        guard.start()
        self.addCleanup(guard.stop)

    def execute(self, response_for):
        calls = []
        class Client:
            def __init__(client, row, profile, callback):
                client.config = SimpleNamespace(model=profile["model"], base_url=profile["base_url"])
                client.callback = callback
            def complete(client, *, prompt, **kwargs):
                calls.append(prompt)
                client.callback(build_payload(self.profile, prompt))
                return response_for(prompt, len(calls))
        self.calls = calls
        with forbid_network():
            return runtime.execute(self.run, "memory_assist", authorized=True,
                allow_unknown_service=True, client_factory=Client)

    def summary(self):
        return runtime.read(self.run / "executions/memory_assist/reports/summary.json")

    def test_repair_timeout_keeps_primary_and_stops_remaining_slots(self):
        def response(prompt, count):
            if count == 2:
                raise TimeoutError("synthetic timeout")
            raw = synthetic_candidates(prompt)
            raw["candidates"][-1]["temporal_relations"] = []
            return BackendResponse(json.dumps(raw), "mock", {"finish_reason": "stop", "usage": None})
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(response)
        summary = self.summary()
        self.assertEqual((summary["primary_calls"], summary["repair_calls"], summary["http_attempts"]), (1, 1, 2))
        self.assertEqual(summary["program_accepted"], 3)
        self.assertEqual(summary["unattempted_primary_slots"], 11)
        with self.assertRaisesRegex(runtime.ExecutionStopped, "CANNOT_AUTOMATICALLY_RESUME"):
            self.execute(response)
        self.assertEqual(self.calls, [])

    def test_unknown_finish_retains_raw_and_never_repairs(self):
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(lambda prompt, count: BackendResponse(json.dumps(synthetic_candidates(prompt)), "mock", {"usage": None}))
        self.assertEqual(self.summary()["http_attempts"], 1)
        self.assertEqual(self.summary()["repair_calls"], 0)
        self.assertEqual(len(list((self.run / "executions").rglob("*.json.txt"))), 1)

    def test_repair_failure_preserves_ordering_derivation_identity(self):
        def response(prompt, count):
            if count == 2:
                raise TimeoutError("synthetic timeout")
            raw = synthetic_candidates(prompt)
            raw["candidates"] = [raw["candidates"][-1]]
            return BackendResponse(json.dumps(raw), "mock", {"finish_reason": "stop", "usage": None})
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(response)
        self.assertEqual(self.summary()["program_accepted"], 1)
        final = runtime.rows(self.run / "executions/memory_assist/reports/final_candidates.jsonl")
        self.assertEqual(final[0]["question_type"], "ordering")
        self.assertEqual(final[0]["ordering_derivation"]["identity"]["intent_id"], "intent_0")
        self.assertEqual(final[0]["ordering_derivation"]["identity"]["source"], "contract_guided")

    def test_length_finish_stops_without_content_repair(self):
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(lambda prompt, count: BackendResponse("{", "mock", {"finish_reason": "length", "usage": None}))
        self.assertEqual(self.summary()["http_attempts"], 1)
        self.assertEqual(self.summary()["repair_calls"], 0)

    def test_completed_invalid_json_gets_exactly_one_content_repair(self):
        def response(prompt, count):
            raw = "{invalid" if "REPAIR THE FOLLOWING BATCH" not in prompt else json.dumps(synthetic_candidates(prompt))
            return BackendResponse(raw, "mock", {"finish_reason": "stop", "usage": None})
        summary = self.execute(response)
        self.assertEqual((summary["primary_calls"], summary["repair_calls"], summary["http_attempts"]), (12, 12, 24))
        self.assertEqual(summary["program_accepted"], 48)


if __name__ == "__main__":
    unittest.main()
