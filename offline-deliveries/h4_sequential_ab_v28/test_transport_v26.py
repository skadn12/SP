"""Real ProfileClient request-contract checks, with a fake HTTP opener only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import runner


class TransportContractTests(unittest.TestCase):
    def setUp(self):
        profile_fn, factory, *_ = runner.base.transport()
        self.profile = profile_fn()
        self.client_type = factory.__globals__["ProfileClient"]
        self.card = {"card_id": "mock-card", "guidance_level": "seed", "actors": [],
                     "steps": [{"step_id": "s1", "actor_id": "a", "action": "MOCK request", "state": "proposed"}],
                     "dependencies": [], "confusion_axes": [], "open_ends": [], "downgrade_reason": "MOCK"}
        self.payload = runner.purpose_contract.build_request("r", "S01", self.card, ["A", "B"], runner.MODEL)

    def test_real_client_accepts_frozen_purpose_payload_without_network(self):
        observed = []
        opened = []
        body = json.dumps({"model": runner.MODEL,
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}).encode()
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return body
            def getcode(self): return 200
        class Opener:
            def open(self, request, timeout):
                opened.append(json.loads(request.data))
                return Reply()
        with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
             patch("urllib.request.build_opener", return_value=Opener()):
            client = self.client_type(self.profile, credential="offline-fixture-only", on_http_attempt=observed.append)
            response = client.complete(prompt=self.payload["messages"][0]["content"],
                                       max_tokens=self.payload["max_tokens"], json_object=True)
        self.assertEqual(observed, [self.payload])
        self.assertEqual(opened, [self.payload])
        self.assertEqual(response.raw_text, "{}")

    def test_real_client_reproduces_old_local_1024_failure_before_http(self):
        observed = []
        opener = SimpleNamespace(open=lambda *a, **k: self.fail("must not reach HTTP"))
        with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
             patch("urllib.request.build_opener", return_value=opener):
            client = self.client_type(self.profile, credential="offline-fixture-only", on_http_attempt=observed.append)
            with self.assertRaisesRegex(ValueError, "caller output budget must match"):
                client.complete(prompt=self.payload["messages"][0]["content"], max_tokens=1024, json_object=True)
        self.assertEqual(observed, [])

    def caller(self, folder):
        value = runner.Caller.__new__(runner.Caller)
        value.run = folder
        value.profile = self.profile
        value.limit = 5
        value.request_counter = value.http_counter = 0
        value.last_http_completed = None
        value.tokenizer = None
        value.capacity = lambda *a: {"status": "PASS"}
        value.annotate = lambda x: x
        value.factory = lambda *a, **k: self.fail("factory must not be reached")
        return value

    def test_profile_differences_fail_before_factory_with_local_record(self):
        for key, value, code in [("max_tokens", 1024, "request_profile_output_budget_mismatch"),
                                 ("temperature", .8, "request_profile_payload_mismatch")]:
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                caller = self.caller(folder)
                payload = deepcopy(self.payload)
                payload[key] = value
                with self.assertRaisesRegex(runner.LocalRequestError, code):
                    caller.call("fixture", payload, {"role": "purpose_proposer"})
                errors = list(folder.glob("requests/*/LOCAL_ERROR.json"))
                self.assertEqual(len(errors), 1)
                self.assertEqual(json.loads(errors[0].read_text())["error_code"], code)
                self.assertFalse(list(folder.glob("requests/*/HTTP_ATTEMPT*.json")))
                self.assertFalse(list(folder.glob("requests/*/ERROR.json")))

    def test_factory_local_error_is_not_provider_shortfall_or_raw_text(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            caller = self.caller(folder)
            def broken(*a, **k): raise ValueError("sensitive-fixture-not-for-log")
            caller.factory = broken
            with self.assertRaisesRegex(runner.LocalRequestError, "pre_http_transport_contract_failure"):
                caller.call("fixture", self.payload, {"role": "purpose_proposer"})
            record = next(folder.glob("requests/*/LOCAL_ERROR.json")).read_text()
            self.assertNotIn("sensitive-fixture", record)
            self.assertFalse(list(folder.glob("requests/*/HTTP_ATTEMPT*.json")))
            self.assertFalse(list(folder.glob("requests/*/ERROR.json")))


if __name__ == "__main__":
    unittest.main()
