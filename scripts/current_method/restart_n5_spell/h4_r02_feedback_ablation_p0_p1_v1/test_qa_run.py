"""Direct tests for new budget, isolation, observation and safe-resume seams."""
from __future__ import annotations

import io
import json
from pathlib import Path
import ssl
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from ..backends import BackendResponse
from ..core import atomic_json, sha256_bytes, sha256_file
from ..h4_batch30_memory_event_k3_probe_v1 import qa_generation_only as original
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa
from ..ledger import RequestLedger
from ..live_backends import ChallengerAPIConfig, OpenAICompatibleChallengerClient
from .qa_run import (ARMS, ROOTS, BoundedLedger, GuardrailStop, ObservedClient,
                     checkpoint_root, read, request_scope, schedule_rows, verify_checkpoint)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.rid = ROOTS[0]
        self.root = self.run / ARMS[0] / "qa/shards/shard-00/roots" / self.rid
        self.root.mkdir(parents=True)
        self.ledger = BoundedLedger(self.root, "frozen", self.run, ARMS[0], self.rid, "mock")
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def kwargs(self, request_id=None, call=None):
        def complete():
            self.calls += 1
            return BackendResponse("{}", "fixture", {})
        return dict(request_id=request_id or f"{self.rid}-intent-miner", intent="fixture", attempt=1,
                    prompt_sha256=sha256_bytes(b"prompt"), call=call or complete,
                    budget_class="fresh_qa_generation", budget_limit=10_000, total_budget_limit=10_000,
                    request_metadata={"max_tokens": 4096}, backend_identity={"model": "fixture"})

    def test_schedule_alternates_first_arm_without_reordering_roots(self):
        rows = schedule_rows()
        self.assertEqual([r["order"] for r in rows], list(range(1, 13)))
        self.assertEqual([r["root_id"] for r in rows[::2]], ROOTS)
        self.assertEqual([r["arm"] for r in rows[::2]], [ARMS[i % 2] for i in range(6)])

    def test_completed_call_is_verified_and_never_resent(self):
        self.ledger.invoke(**self.kwargs())
        response = self.ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 1)
        self.assertTrue(response.metadata["reused"])
        with self.assertRaises(GuardrailStop):
            self.ledger.invoke(**{**self.kwargs(), "prompt_sha256": "changed"})
        state = self.ledger.delegate.read(f"{self.rid}-intent-miner")
        Path(state["raw_response_path"]).write_text("corrupt")
        with self.assertRaises(GuardrailStop):
            self.ledger.invoke(**self.kwargs())

    def test_same_request_id_in_opposite_arm_has_separate_cache(self):
        self.ledger.invoke(**self.kwargs())
        root = self.run / ARMS[1] / "qa/shards/shard-00/roots" / self.rid
        other = BoundedLedger(root, "other_frozen", self.run, ARMS[1], self.rid, "mock")
        other.invoke(**self.kwargs())
        self.assertEqual(self.calls, 2)

    def test_observed_failure_replays_without_resend_but_unknown_reservation_stops(self):
        def fail():
            self.calls += 1
            raise TimeoutError("test timeout")
        for _ in range(2):
            with self.assertRaises(Exception) as caught:
                self.ledger.invoke(**self.kwargs(call=fail))
            self.assertEqual(type(caught.exception).__name__, "TimeoutError")
            self.assertEqual(str(caught.exception), "test timeout")
        self.assertEqual(self.calls, 1)
        failure = self.root / "ledger/observed_failures" / f"{self.rid}-intent-miner.json"
        failure.unlink()
        with self.assertRaises(GuardrailStop):
            self.ledger.invoke(**self.kwargs(call=fail))
        self.assertEqual(self.calls, 1)

    def test_root_and_global_budgets_stop_before_call(self):
        for index in range(13):
            atomic_json(self.ledger.delegate.state_dir / f"existing-{index}.json", {})
        with self.assertRaises(GuardrailStop):
            self.ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 0)
        for path in self.ledger.delegate.state_dir.glob("*.json"):
            path.unlink()
        with patch("scripts.current_method.restart_n5_spell.h4_r02_feedback_ablation_p0_p1_v1.qa_run.state_paths", return_value=[None] * 156):
            with self.assertRaises(GuardrailStop):
                self.ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 0)

    def test_out_of_scope_requests_and_repairs_stop(self):
        atomic_json(self.root / "reasoning_contracts.json", [{"intent_id": "I001"}])
        with self.assertRaises(GuardrailStop):
            request_scope(self.root, self.rid, "unapproved-miner-retry")
        with self.assertRaises(GuardrailStop):
            request_scope(self.root, self.rid, f"{self.rid}-contract_guided-repair-I001")

    def test_checkpoint_detects_artifact_mutation(self):
        target = self.root / "candidates_valid.json"
        atomic_json(target, [])
        checkpoint_root(self.root)
        self.assertTrue(verify_checkpoint(self.root))
        atomic_json(target, [{"candidate": "changed"}])
        with self.assertRaises(GuardrailStop):
            verify_checkpoint(self.root)

    def test_transport_observer_preserves_retry_payload_and_original_counts(self):
        config = ChallengerAPIConfig("https://fixture.invalid", "TEST_ONLY_SECRET", "fixture", temperature=0.1, enable_thinking=False)
        client = ObservedClient(OpenAICompatibleChallengerClient(config), self.run / "transport.jsonl", ARMS[0], self.rid)
        body = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"total_tokens": 9}}
        calls = []
        def urlopen(request, **kwargs):
            calls.append(request.data)
            if len(calls) == 1:
                raise urllib.error.HTTPError("https://fixture.invalid", 429, "rate limit", {"Retry-After": "0"}, None)
            return io.BytesIO(json.dumps(body).encode())
        with patch("urllib.request.urlopen", urlopen), patch("time.sleep"):
            response = client.complete(prompt="fixture", max_tokens=4096, json_object=True)
        self.assertEqual(response.metadata["transport_attempt_count"], 2)
        self.assertEqual(calls[0], calls[1])
        payload = json.loads(calls[0])
        self.assertNotIn("seed", payload)
        self.assertFalse(payload["enable_thinking"])
        events = (self.run / "transport.jsonl").read_text()
        self.assertNotIn("TEST_ONLY_SECRET", events)
        self.assertEqual(sum(json.loads(line)["event"] == "http_attempt_started" for line in events.splitlines()), 2)

    def test_wrapper_matches_original_generator_thirteen_call_repair_path_and_resume(self):
        expanded = {"root_id": self.rid, "turns": [
            {"turn_ref": f"T{i}", "speaker": "person", "text": f"fixture {i}"} for i in range(1, 4)]}
        intents = {"intents": [{"intent_id": f"I{i:03}", "intent_type": "state_transition",
                   "discriminative_variable": f"fixture {i}",
                   "reasoning_chain": {"before_state": [1], "trigger": [2], "after_state": [3]}} for i in range(1, 4)]}
        class Client:
            config = SimpleNamespace(model="fixture", base_url="fixture")
            def __init__(self):
                self.calls = []
            def complete(self, **kwargs):
                self.calls.append(kwargs)
                raw = intents if len(self.calls) == 1 else {"candidates": []}
                return BackendResponse(json.dumps(raw), "fixture", {})
        control = self.run / "control"
        client_original, client_bounded = Client(), Client()
        control_ledger = RequestLedger(control, "control", "mock")
        original_report = generate_root_qa(expanded=expanded, root_dir=control, root_id=self.rid,
                           api_config=Path("unused"), ledger=control_ledger, client=client_original)
        bounded_report = generate_root_qa(expanded=expanded, root_dir=self.root, root_id=self.rid,
                          api_config=Path("unused"), ledger=self.ledger, client=client_bounded,
                          feedback_context={"intent_suffix": "", "slot_suffixes": ["", "", ""]})
        self.assertEqual(client_original.calls, client_bounded.calls)
        self.assertEqual(len(client_bounded.calls), 13)
        self.assertEqual(original_report, bounded_report)
        for filename in ("intents.json", "reasoning_contracts.json", "candidate_generation.json", "candidate_pool.json", "candidates_valid.json", "validity_audit.json", "permutations.json"):
            self.assertEqual(read(control / filename), read(self.root / filename))
        generate_root_qa(expanded=expanded, root_dir=self.root, root_id=self.rid, api_config=Path("unused"),
                         ledger=self.ledger, client=client_bounded,
                         feedback_context={"intent_suffix": "", "slot_suffixes": ["", "", ""]})
        self.assertEqual(len(client_bounded.calls), 13)

    def test_staged_audit_accepts_own_different_intents_and_detects_wrong_suffix(self):
        from . import prompt_audit
        source = self.run / "source.json"
        atomic_json(source, {"root_id": self.rid, "turns": [
            {"turn_ref": f"T{i}", "speaker": "Maya", "text": f"fixture {i}"} for i in range(1, 4)]})
        for arm in ARMS:
            qa = self.run / arm / "qa"
            context_path = self.run / "inputs" / arm / self.rid / "feedback_context.json"
            suffix = "fixture on guidance" if arm == "feedback_on" else ""
            atomic_json(context_path, {"intent_suffix": suffix, "slot_suffixes": [suffix] * 3})
            entry = {"root_id": self.rid, "shard": "shard-00", "source_path": str(source), "source_sha256": sha256_file(source)}
            manifest = {"roots": [entry], "manifest_sha256": arm,
                        "feedback_context": {"roots": {self.rid: {"path": str(context_path), "sha256": sha256_file(context_path)}}}}
            atomic_json(qa / "manifest.json", manifest)
            root, expanded = original._prepare_root(entry, qa)
            ledger = BoundedLedger(root, arm, self.run, arm, self.rid, "mock")
            class Client:
                config = SimpleNamespace(model="Qwen/Qwen3-14B", base_url="fixture")
                count = 0
                def complete(self, **kwargs):
                    self.count += 1
                    raw = {"intents": [{"intent_id": f"I{i:03}", "intent_type": "state_transition",
                           "discriminative_variable": "arm's own content " + arm,
                           "reasoning_chain": {"before_state": [1], "trigger": [2], "after_state": [3]}}
                           for i in range(1, 4)]} if self.count == 1 else {"candidates": []}
                    return BackendResponse(json.dumps(raw), self.config.model, {})
            generate_root_qa(expanded=expanded, root_dir=root, root_id=self.rid, api_config=Path("unused"),
                             ledger=ledger, client=Client(), feedback_context=read(context_path))
        prefix = "scripts.current_method.restart_n5_spell.h4_r02_feedback_ablation_p0_p1_v1.prompt_audit."
        with patch(prefix + "verify_schedule"), patch(prefix + "ROOTS", [self.rid]), patch.object(original, "_verify_feedback_context", side_effect=lambda manifest, _: manifest["feedback_context"]):
            result = prompt_audit.audit(self.run)
            self.assertEqual(result["errors"], [])
            guided_rows = [r for r in result["live_prompt_rows"] if r["mode"] == "contract_guided" and r["attempt"] == "primary"]
            self.assertNotEqual(guided_rows[0]["base_sha256"], guided_rows[3]["base_sha256"])
            prompt_path = self.root / "raw/prompts" / f"{self.rid}-contract_guided-generator-I001.txt"
            prompt_path.write_text(prompt_path.read_text() + "\nUNFROZEN_SUFFIX")
            result = prompt_audit.audit(self.run)
            self.assertTrue(any("own_builder_and_suffix_bytes" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
