import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.live_backends import ChallengerAPIConfig
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_role_runtime_v11 import (
    LiveJsonRoleCallerV11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_transport_v11 import (
    ROUND_VALIDATOR_MAX_OUTPUT_TOKENS,
    RoundValidatorStreamingClientV11,
    build_frozen_round_validator_transport_request_v11,
    round_validator_transport_configuration_v11,
    validate_frozen_round_validator_request_v11,
)


INPUT_SHA = "c8976ae6bbb2c91403466a3987d4054bad099475c8eac16c9ab86a7be7244332"
SCHEMA_SHA = "a" * 64


def config():
    return ChallengerAPIConfig(
        base_url="https://api.invalid/v1", api_key="not-used",
        model="Qwen/Qwen3-14B", timeout_seconds=180, temperature=0.25)


class RoundValidatorTransportTests(unittest.TestCase):
    def test_role_transport_is_long_read_streaming_without_automatic_retry(self):
        value = round_validator_transport_configuration_v11()
        self.assertEqual(value["connect_timeout_seconds"], 30)
        self.assertEqual(value["read_timeout_seconds"], 600)
        self.assertTrue(value["streaming"])
        self.assertEqual(value["max_output_tokens"], 12_288)
        self.assertEqual(value["transport_retry_limit"], 0)

    def test_frozen_request_preserves_model_input_schema_and_seed(self):
        request = build_frozen_round_validator_transport_request_v11(
            config=config(), prompt="frozen validator prompt",
            authoritative_input_sha256=INPUT_SHA, orchestration_seed=61555,
            prompt_schema_contract_sha256=SCHEMA_SHA)
        self.assertTrue(validate_frozen_round_validator_request_v11(request)["valid"])
        self.assertEqual(request.authoritative_input_sha256, INPUT_SHA)
        self.assertEqual(request.orchestration_seed, 61555)
        self.assertEqual(request.model, "Qwen/Qwen3-14B")
        self.assertEqual(request.prompt_schema_contract_sha256, SCHEMA_SHA)
        self.assertTrue(request.body["stream"])
        self.assertEqual(request.body["max_tokens"], 12_288)

    def test_streaming_uses_separate_connect_and_read_timeout(self):
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

        client = RoundValidatorStreamingClientV11(
            config=config(), authoritative_input_sha256=INPUT_SHA,
            orchestration_seed=61555, prompt_schema_contract_sha256=SCHEMA_SHA)
        with patch(
            "scripts.current_method.restart_n5_spell.world_memory_slim_v1."
            "round_validator_transport_v11.http.client.HTTPSConnection", Connection):
            response = client.complete_round_validator(
                prompt="frozen validator prompt",
                max_tokens=ROUND_VALIDATOR_MAX_OUTPUT_TOKENS, json_object=True)
        connection = Connection.instances[0]
        self.assertEqual(connection.kwargs["timeout"], 30)
        self.assertEqual(connection.sock.timeouts, [600])
        self.assertTrue(connection.request_body["stream"])
        self.assertEqual(response.raw_text, '{"ok":true}')
        self.assertEqual(response.metadata["transport_attempt_count"], 1)
        self.assertEqual(response.metadata["transport_retry_count"], 0)

    def test_role_runtime_routes_validator_only_to_validator_transport(self):
        class FakeClient:
            config = config()

            def __init__(self):
                self.calls = []

            def complete_round_validator(self, **kwargs):
                self.calls.append(("validator", kwargs))
                return BackendResponse('{"ok":true}', self.config.model,
                                       {"usage": {}})

            def complete(self, **kwargs):
                self.calls.append(("fallback", kwargs))
                return BackendResponse('{"ok":true}', self.config.model,
                                       {"usage": {}})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient()
            caller = LiveJsonRoleCallerV11(
                run_root=root, ledger=RequestLedger(root, "m" * 64, "live"),
                client=client, token_counter=lambda text: len(text), root_id="root",
                role_budget_limits={"planner": 1, "realizer": 1,
                                    "round_validator": 1},
                total_request_limit=3, context_limit=20_000,
                before_request=lambda logical_role, role_version: None)
            parsed, _, errors = caller.call_json(
                request_id="validator-one",
                role="round_expansion_validator_v1", attempt=1,
                prompt="prompt", prompt_version="v",
                max_tokens=ROUND_VALIDATOR_MAX_OUTPUT_TOKENS)
        self.assertEqual(parsed, {"ok": True})
        self.assertFalse(errors)
        self.assertEqual(client.calls[0][0], "validator")


if __name__ == "__main__":
    unittest.main()
