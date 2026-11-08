"""Caller-only tests. Reuse mocked existing-budget and transport interfaces."""
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("proxy_live_runner_test_subject", HERE / "RUN_ONCE.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CallerTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket, "socket", side_effect=AssertionError("NETWORK FORBIDDEN"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_actual_preflight_dry_run_no_ledger_transport_or_key(self):
        with patch.object(runner.p, "ProbeBudget") as budget, \
             patch.object(runner.proxy, "ProxyProbeTransport") as transport, \
             patch.object(runner.p.old_transport, "_read_credential") as key:
            self.assertEqual(runner.run()["status"], "DRY_RUN_ONLY")
            budget.assert_not_called()
            transport.assert_not_called()
            key.assert_not_called()

    def exercise(self, failure=None, bad_usage=False):
        events = []
        ledger = Mock()
        ledger.snapshot.return_value = {"attempts_consumed": 2}
        reservation = Mock(attempt_id=3)
        transport = Mock(is_mock=True, trace={"phase": "MOCK"})
        def reserve(*args):
            events.append("COMMITTED_RESERVATION")
            return reservation
        def send(*args, **kwargs):
            self.assertIn("COMMITTED_RESERVATION", events)
            self.assertIn("PRE_SEND_RESERVATION.json", events)
            self.assertIs(kwargs["reservation"], reservation)
            events.append("SEND")
            if failure:
                raise failure
            raw = json.dumps({"model": "qwen3.6-35b-a3b", "id": "mock", "choices": [{
                "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
                "usage": {} if bad_usage else {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}}).encode()
            return {"http_status": 200, "headers": {}, "raw_sha256": "0" * 64, "raw": raw}
        transport.send.side_effect = send
        with patch.object(runner, "preflight", return_value={"credential_file": "/not-read", "prior_snapshot_sha256": "mock"}), \
             patch.object(runner.p, "ProbeBudget", return_value=ledger), \
             patch.object(runner.proxy, "reserve_once", side_effect=reserve) as reserve_mock, \
             patch.object(runner.p, "durable_write", side_effect=lambda path,*a,**k:events.append(Path(path).name)):
            result = runner.run(live=True, transport=transport)
        reserve_mock.assert_called_once()
        transport.send.assert_called_once()
        ledger.settle.assert_called_once()
        ledger.close.assert_called_once()
        return result, ledger, events

    def test_normal_order_and_settlement(self):
        result, ledger, events = self.exercise()
        self.assertEqual(result["status"], "PROBE_RESPONSE_VALID_QA_HOLD")
        self.assertEqual(ledger.settle.call_args.args[1], "SUCCEEDED")
        self.assertFalse(result["server_4096_hard_limit_proved"])

    def test_timeout_and_disconnect_settle_once_no_retry(self):
        for error in (TimeoutError(), ConnectionResetError()):
            with self.subTest(error=type(error).__name__):
                result, ledger, events = self.exercise(failure=error)
                self.assertEqual(result["status"], "HOLD_NO_AUTOMATIC_RETRY")
                self.assertEqual(ledger.settle.call_args.args[1], "UNCERTAIN")
                self.assertEqual(events.count("SEND"), 1)

    def test_invalid_usage_rejects_no_followup(self):
        result, ledger, events = self.exercise(bad_usage=True)
        self.assertEqual(result["status"], "HOLD_NO_AUTOMATIC_RETRY")
        self.assertEqual(ledger.settle.call_args.args[1], "REJECTED")
        self.assertEqual(events.count("SEND"), 1)

    def test_pre_send_record_failure_keeps_budget_consumed_and_no_send(self):
        ledger = Mock()
        ledger.snapshot.return_value = {"attempts_consumed": 2}
        transport = Mock(is_mock=True, trace={"phase": "NOT_STARTED"})
        with patch.object(runner, "preflight", return_value={"credential_file": "/not-read", "prior_snapshot_sha256": "mock"}), \
             patch.object(runner.p, "ProbeBudget", return_value=ledger), \
             patch.object(runner.proxy, "reserve_once", return_value=Mock(attempt_id=3)), \
             patch.object(runner.p, "durable_write", side_effect=[OSError("disk failure"), None, None]):
            result = runner.run(live=True, transport=transport)
        transport.send.assert_not_called()
        self.assertEqual(result["new_attempts_consumed"], 1)
        self.assertEqual(ledger.settle.call_args.args[1], "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
