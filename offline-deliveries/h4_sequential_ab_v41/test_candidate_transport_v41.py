"""Exercise actual Caller receipts/retries with fake time and no network."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import candidate_transport as transport

r = transport.r


class FakeClock:
    def __init__(self):
        self.monotonic = 1000.0
        self.wall = 100000.0
        self.sleeps = []
        self.wake_early_once = False

    def advance(self, seconds):
        self.monotonic += seconds
        self.wall += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if self.wake_early_once:
            self.wake_early_once = False
            seconds /= 2
        self.advance(seconds)


class ProviderFailure(RuntimeError):
    def __init__(self, status):
        super().__init__("MOCK_ONLY_PROVIDER_FAILURE")
        self.metadata = {"http_status": status}


class FakeProvider:
    def __init__(self, clock, outcomes=None):
        self.clock = clock
        self.outcomes = list(outcomes or [200] * 10)
        self.calls = []

    def factory(self, profile, *, on_http_attempt):
        owner = self

        class Client:
            def complete(self, *, prompt, max_tokens, json_object):
                payload = {"model": r.MODEL, "messages": [{"role": "user", "content": prompt}],
                           **profile["request_parameters"]}
                on_http_attempt(payload)
                row = {"started": owner.clock.monotonic, "payload": deepcopy(payload)}
                owner.calls.append(row)
                owner.clock.advance(4.0)
                row["completed"] = owner.clock.monotonic
                status = owner.outcomes.pop(0)
                if status != 200:
                    raise ProviderFailure(status)
                return SimpleNamespace(raw_text='{"MOCK_ONLY":true}', metadata={
                    "returned_model": r.MODEL, "finish_reason": "stop", "http_status": 200})

        return Client()


class CandidateTransportV41Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v41_transport_MOCK_ONLY_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock = FakeClock()
        self.provider = FakeProvider(self.clock)
        self.profile = {"model": r.MODEL, "output_limit": 4096, "request_parameters": {
            "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
            "response_format": {"type": "json_object"}}}
        self.payload = {"model": r.MODEL,
            "messages": [{"role": "user", "content": "MOCK ONLY: transport fixture"}],
            **self.profile["request_parameters"]}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("socket.socket", side_effect=AssertionError("offline_transport_only")))
        stack.enter_context(patch.object(r, "LOCK_PATH", self.root / "shared.lock"))
        stack.enter_context(patch.object(r.base, "transport", return_value=(
            lambda: deepcopy(self.profile), self.provider.factory, lambda value: value,
            lambda *_args: {"status": "PASS", "mock_only": True},
            SimpleNamespace(tokenizer_from=lambda _path: object()))))
        stack.enter_context(patch.object(transport.Caller, "_now", side_effect=lambda: self.clock.wall))
        stack.enter_context(patch.object(transport.Caller, "_monotonic",
                                        side_effect=lambda: self.clock.monotonic))
        stack.enter_context(patch.object(transport.Caller, "_boot_id", return_value="MOCK_BOOT"))
        stack.enter_context(patch.object(transport.Caller, "_sleep", side_effect=self.clock.sleep))

    def caller(self, name, *, logical=5, http=10):
        run = self.root / name
        if not run.exists():
            run.mkdir()
            (run / "MANIFEST.json").write_text(json.dumps({"logical_max": logical, "http_max": http}))
            (run / "provider_profile.json").write_text(json.dumps(self.profile))
        return transport.Caller(run)

    def call(self, caller, label="fixture"):
        return caller.call(label, self.payload, {"role": "MOCK_ONLY"})

    def assert_spacing(self):
        for previous, current in zip(self.provider.calls, self.provider.calls[1:]):
            self.assertGreaterEqual(current["started"] - previous["completed"], 35.0)

    def test_first_call_and_two_instances_share_durable_completion_clock(self):
        first, second = self.caller("one"), self.caller("two")
        self.call(first)
        self.assertEqual(self.provider.calls[0]["started"], 1035.0)
        # An arbitrarily large wall-clock jump cannot bypass the 35-second gap.
        self.clock.wall += 100000000.0
        self.call(second)
        self.assertEqual(self.provider.calls[1]["started"], 1074.0)
        self.assert_spacing()
        marker = json.loads(first.completion_path.read_text())
        self.assertEqual(marker["completed_at_monotonic"], self.provider.calls[-1]["completed"])
        self.assertEqual(marker["run"], str(second.run.resolve()))
        receipt = r.base.read(next((second.run / "requests").glob("*/THROTTLE_01.json")))
        self.assertEqual(receipt["waited_seconds"], 35.0)

    def test_failure_updates_completion_before_another_instance_calls(self):
        self.provider.outcomes = [500, 200]
        first = self.caller("failed")
        with self.assertRaisesRegex(RuntimeError, "provider_request_failed_after_retry_policy"):
            self.call(first)
        self.assertEqual(len(self.provider.calls), 1)
        marker = json.loads(first.completion_path.read_text())
        self.assertEqual(marker["completed_at_monotonic"], self.provider.calls[0]["completed"])
        # New object reads the marker from disk, with no instance clock state.
        second = self.caller("following")
        self.assertIsNone(second.last_http_completed)
        self.call(second)
        self.assert_spacing()

    def test_503_then_504_stops_after_one_identical_payload_retry(self):
        self.provider.outcomes = [503, 504, 200]
        caller = self.caller("retry_failure")
        with self.assertRaisesRegex(RuntimeError, "provider_request_failed_after_retry_policy"):
            self.call(caller)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(caller.request_counter, 1)
        self.assertEqual(caller.http_counter, 2)
        self.assertEqual(self.provider.calls[0]["payload"], self.provider.calls[1]["payload"])
        self.assertEqual(self.provider.calls[0]["payload"], self.payload)
        self.assert_spacing()
        receipt = r.base.read(next((caller.run / "requests").glob("*/ERROR.json")))
        self.assertEqual(receipt["automatic_retries"], 1)
        self.assertEqual(receipt["metadata"]["http_status"], 504)

    def test_504_retry_success_keeps_logical_and_http_budgets_distinct(self):
        self.provider.outcomes = [504, 200]
        caller = self.caller("retry_success", logical=1, http=2)
        self.call(caller)
        self.assertEqual((caller.request_counter, caller.http_counter), (1, 2))
        self.assert_spacing()
        replacement = self.caller("retry_success")
        self.assertEqual((replacement.request_counter, replacement.http_counter), (1, 2))
        with self.assertRaisesRegex(r.LocalRequestError, "logical_budget_exhausted"):
            self.call(replacement, "over_budget")
        self.assertEqual(len(self.provider.calls), 2)

    def test_precreated_same_run_instance_cannot_reset_logical_budget(self):
        first = self.caller("one_logical", logical=1, http=8)
        second = self.caller("one_logical")
        self.call(first)
        with self.assertRaisesRegex(r.LocalRequestError, "logical_budget_exhausted"):
            self.call(second, "second")
        self.assertEqual(len(self.provider.calls), 1)

    def test_reconstructed_http_budget_blocks_before_another_request(self):
        first = self.caller("one_http", logical=8, http=1)
        second = self.caller("one_http")
        self.call(first)
        with self.assertRaisesRegex(r.LocalRequestError, "global_http_budget_exhausted"):
            self.call(second, "second")
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(len(list((first.run / "requests").glob("*"))), 1)

    def test_http_budget_also_applies_to_retry(self):
        self.provider.outcomes = [503, 200]
        caller = self.caller("retry_budget", logical=2, http=1)
        with self.assertRaisesRegex(r.LocalRequestError, "pre_http_transport_contract_failure"):
            self.call(caller)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(caller.http_counter, 1)
        self.assertEqual(len(list((caller.run / "requests").glob("*/HTTP_ATTEMPT*.json"))), 1)

    def test_non_retryable_status_never_gets_a_second_attempt(self):
        for status in (400, 429, 500):
            self.provider.outcomes = [status, 200]
            before = len(self.provider.calls)
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, "provider_request_failed_after_retry_policy"):
                    self.call(self.caller(f"failure_{status}"))
                self.assertEqual(len(self.provider.calls), before + 1)
                self.assertEqual(self.provider.outcomes, [200])
        self.assert_spacing()

    def test_sleep_early_wake_and_different_boot_keep_full_gap(self):
        caller = self.caller("first")
        self.clock.wake_early_once = True
        self.call(caller)
        self.assertEqual(self.provider.calls[0]["started"], 1035.0)
        with patch.object(transport.Caller, "_boot_id", return_value="OTHER_MOCK_BOOT"):
            self.call(self.caller("second"))
        self.assert_spacing()

    def test_invalid_marker_fails_closed_before_http(self):
        caller = self.caller("bad_marker")
        caller.completion_path.write_text('{"schema":"corrupted"}')
        with self.assertRaisesRegex(ValueError, "candidate_completion_marker_invalid"):
            self.call(caller)
        self.assertEqual(self.provider.calls, [])


if __name__ == "__main__":
    unittest.main()
