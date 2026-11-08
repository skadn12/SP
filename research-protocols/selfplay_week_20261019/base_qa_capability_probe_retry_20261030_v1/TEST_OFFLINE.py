import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch
import RUN_ONCE as retry

p = retry.p


class Tests(unittest.TestCase):
    def setUp(self):
        blocker = patch.object(socket, "socket", side_effect=AssertionError("NETWORK_FORBIDDEN"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_original_attempt_preserved_and_second_reserved_once(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = p.ProbeBudget(Path(d) / "state", {"path": "fixture", "bytes": 1, "sha256": "0" * 64}, install=True)
            try:
                first = ledger.reserve_probe(p.body())
                ledger.settle(first, "UNCERTAIN", {"fixture": True})
                before = ledger.snapshot()
                second = retry.reserve(ledger, p.digest(before))
                after = ledger.snapshot()
                self.assertEqual(after["attempts_consumed"], 2)
                self.assertEqual(after["remaining_global_attempts"], 10)
                self.assertEqual(before["attempts"][0], after["attempts"][0])
                self.assertEqual(after["attempts"][1]["status"], "RESERVED")
                self.assertEqual(after["attempts"][1]["max_tokens"], 64)
                with self.assertRaises(p.ProbeHold):
                    retry.reserve(ledger, p.digest(before))
                self.assertEqual(ledger.snapshot()["attempts_consumed"], 2)
            finally:
                ledger.close()

    def test_default_transport_never_reads_key_or_connects(self):
        with patch.object(p.old_transport, "_read_credential") as secret, \
             patch.object(retry.http.client, "HTTPSConnection") as connection:
            with self.assertRaises(p.ProbeHold):
                retry.ObservedTransport("/not-read").send()
            secret.assert_not_called()
            connection.assert_not_called()

    def test_timeout_cause_and_phase_preserved_no_retry(self):
        for phase in ("request", "getresponse"):
            conn = Mock()
            getattr(conn, phase).side_effect = TimeoutError("must not copy this private text")
            with patch.object(p.old_transport, "_read_credential", return_value="fixture-secret"), \
                 patch.object(retry.http.client, "HTTPSConnection", return_value=conn) as connection:
                transport = retry.ObservedTransport("/not-read", live=True)
                with self.assertRaises(TimeoutError):
                    transport.send()
                self.assertEqual(transport.trace["underlying_error_type"], "TimeoutError")
                self.assertEqual(transport.trace["phase"], "CONNECT_OR_REQUEST_SEND" if phase == "request" else "AWAIT_RESPONSE_HEADERS")
                self.assertNotIn("private text", json.dumps(transport.trace))
                connection.assert_called_once()
                conn.request.assert_called_once()
                with self.assertRaises(p.ProbeHold):
                    transport.send()
                conn.request.assert_called_once()

    def test_mock_success_uses_same64_payload(self):
        conn, response = Mock(), Mock(status=200)
        value = {"model": "qwen3.6-35b-a3b", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}}
        response.read.return_value = json.dumps(value).encode()
        response.getheader.return_value = None
        conn.getresponse.return_value = response
        with patch.object(p.old_transport, "_read_credential", return_value="fixture-secret"), \
             patch.object(retry.http.client, "HTTPSConnection", return_value=conn):
            envelope = retry.ObservedTransport("/not-read", live=True).send()
            self.assertTrue(p.validate_observation(p.response_observation(envelope))["response_valid"])
            self.assertEqual(p.old_transport._strict_json(conn.request.call_args.kwargs["body"])["max_tokens"], 64)
            conn.request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
