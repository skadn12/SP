"""Offline transport contracts: durable gaps and bounded unchanged requests."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "h4_feedback_training_pool_v15_qa"))
import qa_transport as t


class Clock:
    value = 100.0
    def monotonic(self): return self.value
    def sleep(self, seconds): self.value += seconds


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / "live.lock"
        self.run = self.root / "run"
        self.clock = Clock()
        for patcher in (patch.object(t, "LOCK_PATH", self.lock),
                        patch.object(t.time, "monotonic", self.clock.monotonic),
                        patch.object(t.time, "sleep", self.clock.sleep),
                        patch.object(t, "boot_id", return_value="test-boot")):
            patcher.start(); self.addCleanup(patcher.stop)
        self.profile = {"model": t.MODEL, "output_limit": 4096,
                        "request_parameters": {"temperature": .25, "max_tokens": 4096,
                            "enable_thinking": False, "response_format": {"type": "json_object"}}}
        t.base.write(self.run / "MANIFEST.json", {"budget": {"http_attempts_max": 2}})
        t.base.write(self.run / "provider_profile.json", self.profile)
        self.payload = {"model": t.MODEL, "messages": [{"role": "user", "content": "synthetic fixture"}],
                        **self.profile["request_parameters"]}

    def caller(self, fail=False):
        def factory(profile, on_http_attempt):
            def complete(**kwargs):
                on_http_attempt(self.payload)
                self.clock.value += 8
                if fail:
                    error = RuntimeError("synthetic service error")
                    error.metadata = {"http_status": 503}
                    raise error
                return SimpleNamespace(raw_text='{}', metadata={"requested_model":t.MODEL,
                    "returned_model":t.MODEL,"finish_reason":"stop","http_status":200})
            return SimpleNamespace(complete=complete)
        deps = (lambda: self.profile, factory, lambda x:x, lambda *a:{"status":"PASS"},
                SimpleNamespace(tokenizer_from=lambda *a:None))
        with patch.object(t.base, "transport", return_value=deps):
            return t.Caller(self.run)

    def test_gap_survives_new_caller_and_budget_cannot_reset(self):
        with t.shared_lock():
            first = self.caller()
            raw, folder, _ = first.call("first", self.payload, {})
            self.assertEqual(raw, '{}')
            self.assertEqual(t.base.read(folder/'THROTTLE.json')['waited_seconds'],35)
            second = self.caller()
            _, next_folder, _ = second.call("second", self.payload, {})
            self.assertEqual(t.base.read(next_folder/'THROTTLE.json')['waited_seconds'],35)
            self.assertEqual(second.counter,2)
            with self.assertRaisesRegex(ValueError,"budget"):
                self.caller().call("third",self.payload,{})
            self.assertEqual(len(list((self.run/'requests').glob('*/HTTP_ATTEMPT.json'))),2)

    def test_error_records_completion_and_never_retries(self):
        with t.shared_lock():
            raw, folder, _ = self.caller(fail=True).call("failed", self.payload, {})
            self.assertIsNone(raw)
            self.assertEqual(t.base.read(folder/'HTTP_RESULT.json')['http_status'],503)
            self.assertIsNotNone(t.read_marker())
            self.assertEqual(len(list(folder.glob('HTTP_ATTEMPT*.json'))),1)
            _, nxt, _ = self.caller().call("next",self.payload,{})
            self.assertEqual(t.base.read(nxt/'THROTTLE.json')['waited_seconds'],35)

    def test_new_boot_gets_full_gap_and_bad_clock_is_rejected(self):
        with t.shared_lock():
            t.record_completion(self.run)
            self.clock.value += 100
            with patch.object(t,"boot_id",return_value="different-boot"):
                self.assertEqual(t.wait_gap()[0],35)
            t.marker_path().write_text('{"schema":"broken"}')
            with self.assertRaisesRegex(ValueError,"marker_invalid"):
                t.wait_gap()

    def test_profile_mismatch_fails_before_http(self):
        with t.shared_lock():
            with self.assertRaisesRegex(ValueError,"profile_payload"):
                self.caller().call("bad",{**self.payload,"temperature":.9},{})
            self.assertFalse((self.run/'requests').exists())


if __name__ == '__main__':
    unittest.main()
