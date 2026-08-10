import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_v1.live_canary import (
    DEFAULT_API_CONFIG,
    DEFAULT_INPUT,
    DEFAULT_TOKENIZER,
    run_preflight,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.live_runtime import (
    ALLOWED_LIVE_ROLES,
    SimpleAppendLiveCaller,
    aggregate_usage,
)


class FakeConfig:
    model = "mock-model"
    base_url = "https://example.invalid/v1"
    temperature = 0.0


class FakeClient:
    config = FakeConfig()

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, *, prompt, max_tokens, json_object):
        self.calls += 1
        return BackendResponse(
            json.dumps(self.responses.pop(0)), self.config.model,
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                       "total_tokens": 15}, "finish_reason": "stop"})


class SimpleAppendLiveCanaryTests(unittest.TestCase):
    def test_preflight_does_not_create_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "fresh"
            result = run_preflight(
                input_path=DEFAULT_INPUT, run_root=run_root,
                api_config_path=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER)
            self.assertTrue(result["valid"], result)
            self.assertFalse(run_root.exists())
            self.assertEqual({"planner": 1, "round_realizer": 1, "total": 2},
                             result["request_budget"])

    def test_runtime_allows_exactly_two_roles_and_two_requests(self):
        self.assertEqual({"simple_append_planner_v1",
                          "simple_append_round_realizer_v1"},
                         set(ALLOWED_LIVE_ROLES))
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            ledger = RequestLedger(run_root, "a" * 64, "live")
            client = FakeClient([{"ok": 1}, {"ok": 2}])
            caller = SimpleAppendLiveCaller(
                run_root=run_root, ledger=ledger, client=client,
                token_counter=lambda text: len(text), root_id="root",
                context_limit=10000, before_request=lambda role, version: None)
            for request_id, role in (
                    ("p", "simple_append_planner_v1"),
                    ("r", "simple_append_round_realizer_v1")):
                parsed, _, errors = caller.call_json(
                    request_id=request_id, role=role, prompt="{}",
                    prompt_version="v", max_tokens=10)
                self.assertFalse(errors)
                self.assertIsInstance(parsed, dict)
            self.assertEqual(2, client.calls)
            self.assertEqual(2, aggregate_usage(caller.records)["total_requests"])
            with self.assertRaises(RuntimeError):
                caller.call_json(request_id="x", role="questioner", prompt="{}",
                                 prompt_version="v", max_tokens=10)


if __name__ == "__main__":
    unittest.main()
