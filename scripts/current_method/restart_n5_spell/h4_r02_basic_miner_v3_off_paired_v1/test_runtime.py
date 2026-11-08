"""Offline shared-entry tests of four-cell scope and sticky execution stops."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from ..backends import BackendResponse
from ..core import atomic_json
from ..live_backends import ChallengerAPIConfig, OpenAICompatibleChallengerClient
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_intents
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import expanded_fixture, intents_fixture
from . import runtime as r


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.pairing = [{**row, "prompt_sha256": "same-prompt", "max_output_tokens": 4096,
                         "maximum_logical_calls": 1} for row in r.SCHEDULE]
        self.freeze_pairing()
        self.calls = 0
        for name in ("urllib.request.urlopen", "socket.create_connection"):
            blocker = patch(name, side_effect=AssertionError("network forbidden"))
            blocker.start(); self.addCleanup(blocker.stop)

    def freeze_pairing(self):
        (self.run / "pairing_manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.pairing))
        atomic_json(self.run / "schedule.json", {"version": r.VERSION, "schedule": list(r.SCHEDULE)})

    def ledger(self, index=0, *, live=False, **callbacks):
        row = self.pairing[index]
        if live:
            callbacks = {"before_request": lambda *args: None, "after_response": lambda *args: None,
                         **callbacks}
        return r.BoundedLedger(r.root_path(self.run, row["arm_id"], row["root_id"]), "freeze",
            self.run, row["arm_id"], row["root_id"], backend_mode="live" if live else "mock", **callbacks)

    def success(self):
        self.calls += 1
        return BackendResponse('{"intents":[]}', "fixture", {"transport_attempt_count": 1})

    def kwargs(self, index=0):
        row = self.pairing[index]
        return dict(request_id=row["request_id"], intent="h4_k3_intent_miner", attempt=1,
                    prompt_sha256=row["prompt_sha256"], call=self.success,
                    budget_class="fresh_qa_generation", budget_limit=10000, total_budget_limit=10000,
                    request_metadata={"max_tokens": 4096, "temperature": 0.1,
                                      "request_id": row["request_id"]},
                    backend_identity={"model": "fixture"})

    def post(self, ledger):
        return r.post_miner_check(self.run, ledger.root_dir, ledger.arm_id, ledger.root_id, ledger)

    def shared_prepare(self):
        self.expanded = expanded_fixture()
        self.prompt = build_intent_prompt(self.expanded)
        for row in self.pairing:
            row["prompt_sha256"] = self.prompt["prompt_sha256"]
        self.freeze_pairing()

    def shared(self, index=0, *, before_request=None, after_response=None):
        config = ChallengerAPIConfig("https://fixture.invalid/v1", "not-a-real-key", "Qwen/Qwen3-14B",
                                     temperature=0.1, enable_thinking=False, timeout_seconds=180)
        ledger = self.ledger(index, live=True, **{
            key: value for key, value in (("before_request", before_request), ("after_response", after_response))
            if value is not None})
        row = self.pairing[index]
        client = r.ObservedClient(OpenAICompatibleChallengerClient(config),
            self.run / "audit/transport_events.jsonl", row["arm_id"], row["root_id"])
        result = generate_root_intents(expanded=self.expanded, root_dir=ledger.root_dir,
            root_id=row["root_id"], api_config=Path("/unused"), ledger=ledger, client=client)
        return ledger, result

    @staticmethod
    def response(raw=None):
        body = {"choices": [{"message": {"content": json.dumps(intents_fixture()) if raw is None else raw},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
        response = io.BytesIO(json.dumps(body).encode()); response.status = 200
        return response

    @staticmethod
    def http_error(code):
        return urllib.error.HTTPError("https://fixture.invalid/v1/chat/completions", code, "fixture", {}, None)

    def test_four_new_calls_and_exact_cache_are_arm_isolated(self):
        self.assertEqual((r.LIMIT_TOTAL, r.LIMIT_PER_ROOT, r.LIMIT_PER_ARM, r.LIMIT_PER_PAIR), (4, 2, 2, 1))
        for index, row in enumerate(self.pairing):
            ledger = self.ledger(index)
            result = ledger.invoke(**self.kwargs(index)); self.assertNotIn("reused", result.metadata)
            self.assertEqual(self.post(ledger)["status"], "COMPLETED")
            self.assertTrue(ledger.invoke(**self.kwargs(index)).metadata["reused"])
            state = ledger.delegate.read(row["request_id"])
            self.assertEqual(state["request_metadata"]["global_request_id"], row["global_request_id"])
        self.assertEqual(self.calls, 4)
        self.assertEqual(len(r.audit_reservations(self.run)), 4)
        sends = [row for row in r._read_rows(self.run / "audit/call_events.jsonl") if row["event"] == "new_logical_call"]
        self.assertEqual([r._identity(row) for row in sends], list(r.SCHEDULE))

    def test_fixed_order_violation_is_sticky_without_reservation(self):
        ledger = self.ledger(1)
        with self.assertRaisesRegex(r.GuardrailStop, "out_of_frozen_order"):
            ledger.invoke(**self.kwargs(1))
        with self.assertRaisesRegex(r.GuardrailStop, "sticky_guardrail"):
            self.ledger(0)
        self.assertEqual(r.state_paths(self.run), [])

    def test_forbidden_role_scope_is_sticky(self):
        ledger = self.ledger()
        args = self.kwargs(); args["request_id"] = self.pairing[0]["root_id"] + "-contract_guided-generator-I001"
        with self.assertRaisesRegex(r.GuardrailStop, "outside_four_miner_scope"):
            ledger.invoke(**args)
        self.assertEqual(self.calls, 0)
        self.assertTrue((self.run / "audit/guardrail_events.jsonl").is_file())

    def test_other_root_and_arm_cannot_create_ledger(self):
        with self.assertRaises(r.GuardrailStop):
            r.BoundedLedger(self.run / "invalid", "freeze", self.run, "A_OFF",
                            "cc-train-episode-197635", backend_mode="mock")
        self.assertEqual(r.state_paths(self.run), [])

    def test_pairing_identity_and_prompt_hash_guard_stick(self):
        ledger = self.ledger()
        with self.assertRaisesRegex(r.GuardrailStop, "prompt_outside_frozen_pairing"):
            ledger.invoke(**{**self.kwargs(), "prompt_sha256": "changed"})
        with self.assertRaisesRegex(r.GuardrailStop, "sticky_guardrail"):
            self.post(ledger)
        self.assertEqual(self.calls, 0)

    def test_pairing_configuration_change_is_blocked_before_send(self):
        ledger = self.ledger()
        self.pairing[0]["configuration_sha256"] = "changed"; self.freeze_pairing()
        with self.assertRaisesRegex(r.GuardrailStop, "changed_after_ledger_creation"):
            ledger.invoke(**self.kwargs())
        self.assertEqual(r.state_paths(self.run), [])

    def test_observed_known_http_failure_counts_and_other_cells_continue(self):
        def rejected():
            self.calls += 1
            raise self.http_error(400)
        ledger = self.ledger()
        for _ in range(2):
            with self.assertRaises(Exception):
                ledger.invoke(**{**self.kwargs(), "call": rejected})
        self.assertEqual(self.post(ledger)["status"], "KNOWN_HTTP_FAILURE")
        for index in range(1, 4):
            item = self.ledger(index); item.invoke(**self.kwargs(index)); self.post(item)
        self.assertEqual(self.calls, 4)
        self.assertEqual(len(r.audit_reservations(self.run)), 4)
        self.assertFalse((self.run / "audit/guardrail_events.jsonl").exists())

    def test_timeout_swallowed_by_actual_shared_miner_still_stops_all_paid_calls(self):
        self.shared_prepare()
        with patch("urllib.request.urlopen", side_effect=TimeoutError("service outcome unknown")) as network:
            ledger, result = self.shared()
        self.assertEqual(network.call_count, 1)
        self.assertEqual(len(result["contracts"]), 0)
        self.assertTrue(any("TimeoutError" in error for error in result["intent_result"]["errors"]))
        self.assertEqual(ledger.delegate.read(self.pairing[0]["request_id"])["status"], "reserved")
        with self.assertRaisesRegex(r.GuardrailStop, "sticky_guardrail"):
            self.post(ledger)
        with self.assertRaisesRegex(r.GuardrailStop, "sticky_guardrail"):
            self.ledger(1)
        failure = r.read(ledger.root_dir / "ledger/observed_failures" / (self.pairing[0]["request_id"] + ".json"))
        self.assertEqual(failure["service_completion_status"], "UNKNOWN_COMPLETION")
        self.assertEqual(len(r.state_paths(self.run)), 1)

    def test_http_500_swallowed_by_shared_entry_is_unknown_and_stops(self):
        self.shared_prepare()
        with patch("urllib.request.urlopen", side_effect=self.http_error(500)) as network:
            ledger, result = self.shared()
        self.assertEqual(network.call_count, 1)
        self.assertTrue(result["intent_result"]["errors"])
        with self.assertRaises(r.GuardrailStop):
            self.post(ledger)
        self.assertEqual(r.read(r._transport_path(self.run, ledger.arm_id, ledger.root_id))["http_status"], 500)

    def test_shared_entry_known_429_failure_preserves_retries_then_continues(self):
        self.shared_prepare()
        with (patch("urllib.request.urlopen", side_effect=[self.http_error(429) for _ in range(3)]) as network,
              patch("scripts.current_method.restart_n5_spell.live_backends.time.sleep")):
            ledger, result = self.shared()
        self.assertEqual(network.call_count, 3)
        self.assertTrue(result["intent_result"]["errors"])
        checked = self.post(ledger)
        self.assertEqual(checked["status"], "KNOWN_HTTP_FAILURE")
        self.assertEqual(checked["transport"]["transport_attempt_count"], 3)
        with patch("urllib.request.urlopen", return_value=self.response()) as next_network:
            next_ledger, next_result = self.shared(1)
        self.assertEqual(next_network.call_count, 1)
        self.assertEqual(len(next_result["contracts"]), 3)
        self.assertEqual(self.post(next_ledger)["status"], "COMPLETED")

    def test_received_invalid_model_json_is_completed_and_does_not_stop_schedule(self):
        self.shared_prepare()
        with patch("urllib.request.urlopen", return_value=self.response("{invalid model JSON")):
            ledger, result = self.shared()
        self.assertTrue(result["intent_result"]["errors"])
        self.assertEqual(self.post(ledger)["status"], "COMPLETED")
        with patch("urllib.request.urlopen", return_value=self.response()):
            next_ledger, _ = self.shared(1)
        self.assertEqual(self.post(next_ledger)["status"], "COMPLETED")

    def test_before_callback_exception_swallowed_by_shared_entry_is_sticky(self):
        self.shared_prepare()
        def failed_preflight(*args):
            raise RuntimeError("frozen_dependency_or_token_guard")
        ledger, result = self.shared(before_request=failed_preflight)
        self.assertTrue(any("frozen_dependency_or_token_guard" in error for error in result["intent_result"]["errors"]))
        self.assertEqual(r.state_paths(self.run), [])
        with self.assertRaisesRegex(r.GuardrailStop, "sticky_guardrail"):
            self.post(ledger)

    def test_after_callback_exception_swallowed_by_shared_entry_preserves_complete_and_stops(self):
        self.shared_prepare()
        def failed_usage(*args):
            raise RuntimeError("actual_service_token_mismatch")
        with patch("urllib.request.urlopen", return_value=self.response()):
            ledger, result = self.shared(after_response=failed_usage)
        self.assertTrue(any("actual_service_token_mismatch" in error for error in result["intent_result"]["errors"]))
        state = ledger.delegate.read(self.pairing[0]["request_id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(json.loads(Path(state["raw_response_path"]).read_text()), intents_fixture())
        with self.assertRaisesRegex(r.GuardrailStop, "sticky_guardrail"):
            self.post(ledger)
        with self.assertRaises(r.GuardrailStop):
            self.ledger(1)

    def test_missing_actual_transport_record_rejected_despite_completed_live_ledger(self):
        ledger = self.ledger(live=True)
        ledger.invoke(**self.kwargs())
        with self.assertRaisesRegex(r.GuardrailStop, "missing_actual_transport_outcome"):
            self.post(ledger)
        self.assertEqual(self.calls, 1)

    def test_moved_reservation_cannot_restore_budget(self):
        ledger = self.ledger(); ledger.invoke(**self.kwargs())
        state = ledger.delegate._path(self.pairing[0]["request_id"])
        target = self.run / "backup/ledger/requests" / state.name
        target.parent.mkdir(parents=True); state.replace(target)
        with self.assertRaisesRegex(r.GuardrailStop, "reservation_directory_scope_drift"):
            self.ledger(1).invoke(**self.kwargs(1))
        self.assertEqual(self.calls, 1)

    def test_deleted_reservation_is_blocked_by_global_call_event(self):
        ledger = self.ledger(); ledger.invoke(**self.kwargs())
        ledger.delegate._path(self.pairing[0]["request_id"]).unlink()
        with self.assertRaisesRegex(r.GuardrailStop, "event_reservation_inventory_mismatch"):
            self.ledger(1).invoke(**self.kwargs(1))

    def test_orphan_interruption_is_sticky(self):
        def interrupted():
            raise KeyboardInterrupt("process interrupted")
        ledger = self.ledger()
        with self.assertRaises(KeyboardInterrupt):
            ledger.invoke(**{**self.kwargs(), "call": interrupted})
        with self.assertRaises(r.GuardrailStop):
            self.ledger(1)
        self.assertEqual(len(r.state_paths(self.run)), 1)

    def test_exact_cache_validates_raw_and_manifest(self):
        ledger = self.ledger(); ledger.invoke(**self.kwargs())
        ledger.delegate.manifest_sha256 = "other"
        with self.assertRaisesRegex(r.GuardrailStop, "cached_request_contract_drift"):
            ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 1)

    def test_lock_and_checkpoint_include_arm_artifacts(self):
        with r.exclusive_run(self.run):
            with self.assertRaisesRegex(RuntimeError, "already_running"):
                with r.exclusive_run(self.run):
                    pass
        ledger = self.ledger(); ledger.invoke(**self.kwargs()); self.post(ledger)
        r.checkpoint_root(ledger.root_dir)
        self.assertTrue(r.verify_checkpoint(ledger.root_dir))
        (ledger.root_dir / "post_miner_check.json").write_text("changed")
        with self.assertRaisesRegex(r.GuardrailStop, "completed_artifact_drift"):
            r.verify_checkpoint(ledger.root_dir)

    def test_original_client_retry_payload_is_unchanged(self):
        self.shared_prepare()
        with (patch("urllib.request.urlopen", side_effect=[self.http_error(429), self.http_error(429), self.response()]) as network,
              patch("scripts.current_method.restart_n5_spell.live_backends.time.sleep")):
            ledger, result = self.shared()
        self.assertEqual(len(result["contracts"]), 3)
        self.assertEqual(self.post(ledger)["transport"]["transport_attempt_count"], 3)
        for call in network.call_args_list:
            payload = json.loads(call.args[0].data)
            self.assertNotIn("seed", payload)
            self.assertEqual(payload["temperature"], .1)
            self.assertIs(payload["enable_thinking"], False)
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertEqual(call.kwargs["timeout"], 180)


if __name__ == "__main__":
    unittest.main()
