"""Real transport/runtime wiring exercised with an in-memory HTTP opener only."""
import io
import json
import socket
import ssl
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from ..core import atomic_json
from ..h4_r02_ordering_temporal_contract_v1.test_runtime import synthetic_candidates
from . import prepare, runtime
from .test_runtime import seed_run
from .transport import ProfileClient, build_payload


class Response(io.BytesIO):
    def getcode(self):
        return 200


class LiveGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        seed_run(self.run)
        self.secret = "offline-only-private-credential-never-persist"
        self.sent, self.clients = [], []
        profiles = runtime.read(self.run / "profiles.json")
        for index, arm in enumerate(runtime.ARMS):
            profile = profiles[arm]
            profile.update(endpoint=f"https://arm-{index}.invalid/custom/chat/completions",
                           timeout_seconds=1, api_key_env=f"H4_TEST_ARM_{index}")
            if index:
                profile["output_limit"] = 4096
                profile["request_parameters"] = {"max_completion_tokens": 4096}
            else:
                profile["request_parameters"].update(enable_thinking=False,
                                                      response_format={"type": "json_object"})
            for path in (self.run / "policy_snapshot/primary_requests" / arm).glob("*/*/request.json"):
                atomic_json(path, {"payload": build_payload(profile, path.with_name("prompt.txt").read_text())})
        atomic_json(self.run / "profiles.json", profiles)
        blocker = patch.object(socket.socket, "connect", side_effect=AssertionError("NETWORK_FORBIDDEN"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def execute(self, behavior=None):
        def open_response(request, timeout):
            payload = json.loads(request.data)
            # Keep only non-secret request evidence, as the production ledger does.
            self.sent.append({"endpoint": request.full_url, "payload": payload})
            self.assertEqual(request.get_header("Authorization"), "Bearer " + self.secret)
            if behavior:
                outcome = behavior(payload, len(self.sent))
                if isinstance(outcome, BaseException):
                    raise outcome
                if isinstance(outcome, bytes):
                    return Response(outcome)
                text = outcome
            else:
                text = json.dumps(synthetic_candidates(payload["messages"][0]["content"]))
            return Response(json.dumps({"model": "returned-" + payload["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": text}}],
                "usage": {"prompt_tokens": 101, "completion_tokens": 67, "total_tokens": 168,
                          "completion_tokens_details": {"reasoning_tokens": 9}}}).encode())

        opener = Mock()
        opener.open.side_effect = open_response

        def factory(row, profile, callback):
            self.clients.append(row)
            return ProfileClient(profile, credential=self.secret, before_send=callback)

        with patch("urllib.request.build_opener", return_value=opener):
            return runtime.execute_schedule(self.run, client_factory=factory,
                                            verifier=lambda: None, capacity=lambda *args: None)

    def test_profile_transport_callback_payload_accounting_and_resume(self):
        result = self.execute()
        self.assertEqual((result["primary_calls"], result["repair_calls"], result["http_attempts"]), (12, 0, 12))
        self.assertEqual(len(self.sent), 12)
        self.assertEqual(len(self.clients), 12)
        events = runtime.rows(self.run / "audit/http_events.jsonl")
        profiles = runtime.read(self.run / "profiles.json")
        for sent, event in zip(self.sent, events):
            profile = profiles[event["arm_id"]]
            frozen = self.run / "policy_snapshot/primary_requests" / event["arm_id"] / event["root_id"] / event["slot_id"] / "request.json"
            self.assertEqual(sent["payload"], runtime.read(frozen)["payload"])
            self.assertEqual(sent["endpoint"], profile["endpoint"])
            outcome = runtime.read(runtime.directory(self.run, event) / "ledger/transport_outcomes" / (event["request_id"] + ".json"))
            self.assertEqual(outcome["actual_model"], "returned-" + profile["model"])
            self.assertEqual(outcome["transport_attempt_count"], 1)
            self.assertEqual(outcome["transport_retry_count"], 0)
            self.assertEqual(outcome["reasoning_tokens"], 9)
        for first, second in zip(self.sent[::2], self.sent[1::2]):
            self.assertEqual(first["payload"]["messages"], second["payload"]["messages"])
            self.assertNotEqual(first["payload"]["model"], second["payload"]["model"])
            self.assertNotEqual(first["endpoint"], second["endpoint"])
            strong = next(value for value in (first, second)
                          if value["payload"]["model"] == profiles[runtime.ARMS[1]]["model"])
            self.assertNotIn("response_format", strong["payload"])
            self.assertNotIn("temperature", strong["payload"])
        self.assertEqual(self.execute(), result)
        self.assertEqual(len(self.sent), 12)
        self.assertEqual(len(self.clients), 12)
        for path in self.run.rglob("*"):
            if path.is_file():
                self.assertNotIn(self.secret.encode(), path.read_bytes(), str(path))

    def test_invalid_generated_json_uses_one_own_model_repair(self):
        def behavior(payload, count):
            prompt = payload["messages"][0]["content"]
            return json.dumps(synthetic_candidates(prompt)) if "REPAIR THE FOLLOWING BATCH" in prompt else "{invalid"
        result = self.execute(behavior)
        self.assertEqual((result["primary_calls"], result["repair_calls"], result["http_attempts"]), (12, 12, 24))
        self.assertEqual(len(self.sent), 24)
        events = runtime.rows(self.run / "audit/http_events.jsonl")
        for primary, repair in zip(events[::2], events[1::2]):
            self.assertEqual((primary["stage"], repair["stage"]), ("primary", "repair"))
            for key in ("arm_id", "root_id", "slot_id"):
                self.assertEqual(primary[key], repair[key])
        self.execute()
        self.assertEqual(len(self.sent), 24)

    def assert_technical_failure_stops(self, error):
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(lambda payload, count: error)
        self.assertEqual(len(self.sent), 1)
        events = runtime.rows(self.run / "audit/http_events.jsonl")
        logical = runtime.rows(self.run / "audit/logical_calls.jsonl")
        self.assertEqual([row["stage"] for row in events], ["primary"])
        self.assertEqual([row["stage"] for row in logical], ["primary"])
        self.assertTrue((self.run / "audit/execution_stop.json").is_file())
        event = events[0]
        outcome = runtime.read(runtime.directory(self.run, event) / "ledger/transport_outcomes" / (event["request_id"] + ".json"))
        self.assertEqual(outcome["transport_attempt_count"], 1)
        self.assertEqual(outcome["transport_retry_count"], 0)
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(len(runtime.rows(self.run / "audit/http_events.jsonl")), 1)

    def test_429_stops_without_repair_or_resend(self):
        self.assert_technical_failure_stops(urllib.error.HTTPError(
            "https://invalid", 429, "rejected", {}, io.BytesIO(b"rate limit")))

    def test_parameter_rejection_stops_without_parameter_fallback(self):
        self.assert_technical_failure_stops(urllib.error.HTTPError(
            "https://invalid", 400, "rejected", {}, io.BytesIO(b"unsupported response_format")))
        self.assertEqual(self.sent[0]["payload"]["response_format"], {"type": "json_object"})

    def test_context_rejection_stops_without_truncation_or_repair(self):
        self.assert_technical_failure_stops(urllib.error.HTTPError(
            "https://invalid", 413, "rejected", {}, io.BytesIO(b"context length exceeded")))

    def test_ssl_failure_stops_without_ssl_fallback_or_repair(self):
        self.assert_technical_failure_stops(urllib.error.URLError(ssl.SSLError("EOF occurred")))

    def test_server_failure_stops_without_repair(self):
        self.assert_technical_failure_stops(urllib.error.HTTPError(
            "https://invalid", 503, "unavailable", {}, io.BytesIO(b"unavailable")))

    def test_invalid_response_envelope_json_stops_without_content_repair(self):
        self.assert_technical_failure_stops(b"{invalid service envelope")

    def test_invalid_response_envelope_schema_stops_without_content_repair(self):
        self.assert_technical_failure_stops(json.dumps({
            "model": "actual-model", "error": "upstream failure",
            "usage": {"prompt_tokens": 9}}).encode())
        outcome_path = next(self.run.glob("**/ledger/transport_outcomes/*.json"))
        self.assertEqual(runtime.read(outcome_path)["usage"], {"prompt_tokens": 9})

    def test_failure_after_completed_slot_keeps_accepted_candidates_and_costs(self):
        def behavior(payload, count):
            if count == 2:
                return urllib.error.HTTPError(
                    "https://invalid", 400, "rejected", {}, io.BytesIO(b"unsupported max_tokens"))
            return json.dumps(synthetic_candidates(payload["messages"][0]["content"]))
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(behavior)
        self.assertEqual(len(self.sent), 2)
        summary = runtime.read(self.run / "reports/runtime_summary.json")
        self.assertEqual(summary["status"], "STOPPED")
        self.assertEqual((summary["primary_calls"], summary["repair_calls"],
                          summary["logical_calls"], summary["http_attempts"]), (2, 0, 2, 2))
        self.assertEqual(len(runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")), 4)
        stages = runtime.rows(self.run / "reports/runtime_stage_records.jsonl")
        self.assertEqual(len(stages), 2)
        self.assertEqual(len(list(self.run.glob("**/ledger/transport_outcomes/*.json"))), 2)
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.sent), 2)

    def test_failure_during_content_repair_stops_and_consumes_both_attempts(self):
        def behavior(payload, count):
            return "{invalid" if count == 1 else urllib.error.HTTPError(
                "https://invalid", 429, "rejected", {}, io.BytesIO(b"rate limit"))
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute(behavior)
        self.assertEqual(len(self.sent), 2)
        events = runtime.rows(self.run / "audit/http_events.jsonl")
        self.assertEqual([row["stage"] for row in events], ["primary", "repair"])
        self.assertEqual(events[0]["order"], events[1]["order"])
        self.assertEqual(self.sent[0]["payload"]["model"], self.sent[1]["payload"]["model"])
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(len(self.sent), 2)

    def test_transport_timeout_stops_all_following_requests_and_resume(self):
        self.assert_technical_failure_stops(TimeoutError())

    def test_live_missing_authorization_cannot_construct_client_or_check_credentials(self):
        with patch.object(runtime, "ProfileClient") as client, \
             patch.object(runtime, "resolve_credential") as credential, \
             patch.object(prepare, "preflight") as preflight:
            with self.assertRaisesRegex(runtime.ExecutionStopped, "EXPLICIT_LIVE_BUDGET_AUTHORIZATION_REQUIRED"):
                runtime.execute_live(self.run)
            preflight.assert_not_called()
            credential.assert_not_called()
            client.assert_not_called()

    def test_live_unready_preflight_cannot_construct_client_or_read_credentials(self):
        with patch.object(runtime, "ProfileClient") as client, \
             patch.object(runtime, "resolve_credential") as credential, \
             patch.object(prepare, "preflight", return_value={"status": "WAITING_FOR_STRONG_MODEL_CONFIG"}) as preflight:
            with self.assertRaisesRegex(runtime.ExecutionStopped, "LIVE_PREFLIGHT_NOT_READY"):
                runtime.execute_live(self.run, authorized=True)
            preflight.assert_called_once_with(self.run, require_live=True)
            credential.assert_not_called()
            client.assert_not_called()

    def test_live_raised_preflight_cannot_construct_client(self):
        with patch.object(runtime, "ProfileClient") as client, \
             patch.object(runtime, "resolve_credential") as credential, \
             patch.object(prepare, "preflight", side_effect=RuntimeError("LIVE_PREFLIGHT_BLOCKED")):
            with self.assertRaisesRegex(RuntimeError, "LIVE_PREFLIGHT_BLOCKED"):
                runtime.execute_live(self.run, authorized=True)
            credential.assert_not_called()
            client.assert_not_called()

    def test_budget_drift_stops_before_constructing_any_client(self):
        contract = runtime.read(self.run / "execution_contract.json")
        contract["maximum_http_attempts"] = 25
        atomic_json(self.run / "execution_contract.json", contract)
        with self.assertRaises(runtime.ExecutionStopped):
            self.execute()
        self.assertEqual(self.clients, [])
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
