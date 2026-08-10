import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.live_backends import ChallengerAPIConfig
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    prompt_schema_versions_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_role_runtime_v11 import (
    LiveJsonRoleCallerV11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_transport_v11 import (
    PLANNER_MAX_OUTPUT_TOKENS, PLANNER_TIMEOUT_RETRY_CLASSIFICATIONS,
    PlannerTransportHTTPErrorV11, PlannerTransportSettingsV11,
    PlannerTransportRecoveryClientV11,
    PlannerTransportTimeoutV11, build_frozen_planner_transport_request_v11,
    execute_frozen_planner_transport_v11, planner_transport_configuration_v11,
    validate_frozen_planner_request_v11,
)


CANARY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_1")
INPUT_SHA = "c8976ae6bbb2c91403466a3987d4054bad099475c8eac16c9ab86a7be7244332"
SCHEMA_SHA = "c5c8845cb1587e85b97fc67b70594a3a93e59a66e5140ca79f967f1535e1566c"


def config():
    return ChallengerAPIConfig(
        base_url="https://api.invalid/v1", api_key="not-used",
        model="Qwen/Qwen3-14B", timeout_seconds=180,
        temperature=0.25)


def request(prompt="frozen planner prompt"):
    return build_frozen_planner_transport_request_v11(
        config=config(), prompt=prompt, authoritative_input_sha256=INPUT_SHA,
        orchestration_seed=61555, prompt_schema_contract_sha256=SCHEMA_SHA)


class PlannerTransportRecoveryTests(unittest.TestCase):
    def test_transport_configuration_is_planner_specific_recovery(self):
        value = planner_transport_configuration_v11()
        self.assertEqual(value["connect_timeout_seconds"], 30)
        self.assertEqual(value["read_timeout_seconds"], 600)
        self.assertTrue(value["streaming"])
        self.assertEqual(value["max_output_tokens"], 10_240)
        self.assertEqual(value["max_output_tokens_logical_field"],
                         "max_output_tokens")
        self.assertEqual(value["max_output_tokens_wire_field"], "max_tokens")
        self.assertEqual(value["timeout_retry_limit"], 1)
        self.assertEqual(set(value["retryable_timeout_classifications"]),
                         set(PLANNER_TIMEOUT_RETRY_CLASSIFICATIONS))

    def test_frozen_envelope_contains_unchanged_contract_fields(self):
        value = request()
        body = value.body
        self.assertTrue(validate_frozen_planner_request_v11(value)["valid"])
        self.assertEqual(value.authoritative_input_sha256, INPUT_SHA)
        self.assertEqual(value.orchestration_seed, 61555)
        self.assertEqual(value.model, "Qwen/Qwen3-14B")
        self.assertEqual(value.prompt_schema_contract_sha256, SCHEMA_SHA)
        self.assertEqual(body["max_tokens"], PLANNER_MAX_OUTPUT_TOKENS)
        self.assertNotIn("max_output_tokens", body)
        self.assertNotIn("seed", body)
        self.assertEqual(body["model"], "Qwen/Qwen3-14B")
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_read_timeout_retries_once_with_identical_fingerprint(self):
        frozen = request()
        seen = []

        def send_once(value, attempt):
            seen.append({
                "object_id": id(value), "fingerprint": value.request_fingerprint_sha256,
                "body_json": value.body_json, "prompt_sha": value.prompt_sha256,
                "input_sha": value.authoritative_input_sha256,
                "seed": value.orchestration_seed, "model": value.model,
                "schema_sha": value.prompt_schema_contract_sha256,
            })
            if attempt == 1:
                raise PlannerTransportTimeoutV11("read_timeout_before_first_event")
            return BackendResponse('{"ok":true}', value.model, {"usage": {}})

        response = execute_frozen_planner_transport_v11(
            request=frozen, send_once=send_once)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(response.metadata["transport_retry_count"], 1)
        self.assertEqual(response.metadata["transport_attempt_count"], 2)
        self.assertEqual(response.metadata["request_fingerprint_sha256"],
                         frozen.request_fingerprint_sha256)

    def test_http_rejection_does_not_retry(self):
        calls = []

        def send_once(value, attempt):
            calls.append((value.request_fingerprint_sha256, attempt))
            raise PlannerTransportHTTPErrorV11(400, "bad request")

        with self.assertRaises(PlannerTransportHTTPErrorV11):
            execute_frozen_planner_transport_v11(
                request=request(), send_once=send_once)
        self.assertEqual(len(calls), 1)

    def test_second_timeout_stops_after_one_retry(self):
        calls = []

        def send_once(value, attempt):
            calls.append((value.request_fingerprint_sha256, attempt))
            raise PlannerTransportTimeoutV11(
                "read_timeout_after_partial_stream", partial_event_count=3)

        with self.assertRaises(PlannerTransportTimeoutV11):
            execute_frozen_planner_transport_v11(
                request=request(), send_once=send_once)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], calls[1][0])

    def test_streaming_transport_joins_sse_and_uses_separate_timeouts(self):
        events = [
            b'data: {"model":"Qwen/Qwen3-14B","choices":[{"delta":{"content":"{\\"ok\\":"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"content":"true}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n',
            b'data: [DONE]\n', b'',
        ]

        class Sock:
            def __init__(self):
                self.timeouts = []

            def settimeout(self, value):
                self.timeouts.append(value)

        class Response:
            status = 200
            reason = "OK"

            def readline(self):
                return events.pop(0)

        class Connection:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.sock = Sock()
                self.request_body = None
                self.__class__.instances.append(self)

            def connect(self):
                return None

            def request(self, method, path, body, headers):
                self.request_body = json.loads(body)

            def getresponse(self):
                return Response()

            def close(self):
                return None

        client = PlannerTransportRecoveryClientV11(
            config=config(), authoritative_input_sha256=INPUT_SHA,
            orchestration_seed=61555, prompt_schema_contract_sha256=SCHEMA_SHA)
        frozen = request()
        with patch(
            "scripts.current_method.restart_n5_spell.world_memory_slim_v1."
            "planner_transport_v11.http.client.HTTPSConnection", Connection):
            response = client._send_once(frozen, 1)
        connection = Connection.instances[0]
        self.assertEqual(connection.kwargs["timeout"], 30)
        self.assertEqual(connection.sock.timeouts, [600])
        self.assertTrue(connection.request_body["stream"])
        self.assertEqual(response.raw_text, '{"ok":true}')
        self.assertEqual(response.metadata["stream_event_count"], 2)
        self.assertEqual(response.metadata["usage"]["total_tokens"], 12)

    def test_v11_failed_request_evidence_is_read_timeout_before_parse(self):
        attempt = json.loads((CANARY / "attempts/planner_v11_attempt_01.json").read_text())
        ledger = json.loads((CANARY / "ledger/requests/planner_v11_attempt_01.json").read_text())
        self.assertEqual(attempt["runtime_error"],
                         "TimeoutError:The read operation timed out")
        self.assertFalse(attempt["request_completed"])
        self.assertNotIn("parsed_response", attempt)
        self.assertEqual(ledger["status"], "reserved")
        self.assertFalse((CANARY / "raw/backend_responses/planner_v11_attempt_01.txt").exists())

    def test_historical_prompt_is_unchanged_after_new_contract_version(self):
        current_schema_sha = hashlib.sha256(json.dumps(
            prompt_schema_versions_v11(), sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        freeze = json.loads((CANARY / "run_freeze_manifest.json").read_text())
        ledger = json.loads((CANARY / "ledger/requests/planner_v11_attempt_01.json").read_text())
        prompt = (CANARY / "raw/prompts/planner_v11_attempt_01.txt").read_text()
        self.assertNotEqual(current_schema_sha, SCHEMA_SHA)
        self.assertEqual(freeze["prompt_schema_contract_sha256"], SCHEMA_SHA)
        self.assertEqual(freeze["input_sha256"], INPUT_SHA)
        self.assertEqual(freeze["orchestration_seed"], 61555)
        self.assertEqual(freeze["api_configuration"]["model"], "Qwen/Qwen3-14B")
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(),
                         ledger["prompt_sha256"])

    def test_role_runtime_routes_only_planner_to_recovery_method(self):
        class FakeClient:
            config = config()

            def __init__(self):
                self.calls = []

            def complete_planner(self, **kwargs):
                self.calls.append(("planner", kwargs))
                return BackendResponse('{"ok":true}', self.config.model, {"usage": {}})

            def complete(self, **kwargs):
                self.calls.append(("fallback", kwargs))
                return BackendResponse('{"ok":true}', self.config.model, {"usage": {}})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient()
            caller = LiveJsonRoleCallerV11(
                run_root=root, ledger=RequestLedger(root, "m" * 64, "live"),
                client=client, token_counter=lambda text: len(text), root_id="root",
                role_budget_limits={"planner": 1, "realizer": 1,
                                    "round_validator": 1},
                total_request_limit=3, context_limit=20000,
                before_request=lambda logical_role, role_version: None)
            parsed, _, errors = caller.call_json(
                request_id="planner-one", role="planner_v11", attempt=1,
                prompt="prompt", prompt_version="v",
                max_tokens=PLANNER_MAX_OUTPUT_TOKENS)
            self.assertEqual(parsed, {"ok": True})
            self.assertFalse(errors)
            self.assertEqual(client.calls[0][0], "planner")


if __name__ == "__main__":
    unittest.main()
