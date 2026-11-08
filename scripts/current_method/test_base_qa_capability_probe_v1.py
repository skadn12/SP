"""Synthetic, network-forbidden tests for the one authorized capability probe."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import base_qa_capability_probe_v1 as probe


def envelope(*, model="qwen3.6-35b-a3b", usage=None, finish="stop", text="OK", status=200):
    value = {"model": model, "id": "mock-probe", "choices": [{"index": 0, "finish_reason": finish,
        "message": {"role": "assistant", "content": text}}],
        "usage": usage if usage is not None else {"prompt_tokens": 19, "completion_tokens": 2, "total_tokens": 21}}
    raw = json.dumps(value).encode()
    return {"http_status": status, "raw": raw, "raw_sha256": hashlib.sha256(raw).hexdigest(), "headers": {}}


class MockTransport:
    is_mock = True

    def __init__(self, callback=None, failure=None, response=None):
        self.calls = []
        self.callback, self.failure = callback, failure
        self.response = response or envelope()

    def send(self, body):
        self.calls.append(copy.deepcopy(body))
        if self.callback:
            self.callback()
        if self.failure:
            raise self.failure
        return self.response


class ProbeTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket, "socket", side_effect=AssertionError("NETWORK FORBIDDEN"))
        guard.start()
        self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.artifacts = self.base / "probe"
        self.artifacts.mkdir()
        self.canonical = self.base / "canonical"
        probe.durable_write(self.artifacts / "REQUEST.json", probe.body())
        experiment = {"version": probe.VERSION, "probe_maximum_attempts": 1,
            "original_global_maximum_attempts": 12, "output_limit": 64,
            "qa_authorized": False, "bindings": [probe.file_binding(self.artifacts / "REQUEST.json")],
            "canonical_directory": str(self.canonical), "credential_file": "/not-read"}
        probe.durable_write(self.artifacts / "EXPERIMENT_PROBE.json", experiment)
        probe.durable_write(self.artifacts / "EXPERIMENT_PROBE_FREEZE.json", {"sha256": probe.digest(experiment)})
        self.binding = probe.file_binding(self.artifacts / "EXPERIMENT_PROBE.json")

    def budget(self, install=False):
        return probe.ProbeBudget(self.canonical, self.binding, install=install)

    def test_dry_run_no_ledger_no_transport_no_secret(self):
        with patch.object(probe, "SingleProbeTransport") as factory:
            self.assertEqual(probe.execute_once(self.artifacts, self.canonical)["status"], "DRY_RUN_ONLY")
            factory.assert_not_called()
        self.assertFalse(self.canonical.exists())

    def test_one_request_precommitted_and_restart_forbidden(self):
        b = self.budget(True)
        b.close()
        def check():
            ledger = self.budget()
            try:
                snapshot = ledger.snapshot()
                self.assertEqual(snapshot["attempts_consumed"], 1)
                self.assertEqual(snapshot["remaining_global_attempts"], 11)
                self.assertEqual(snapshot["attempts"][0]["status"], "RESERVED")
            finally:
                ledger.close()
        transport = MockTransport(callback=check)
        result = probe.execute_once(self.artifacts, self.canonical, live=True, transport=transport)
        self.assertEqual(result["status"], "PROBE_RESPONSE_VALID_QA_HOLD")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0]["max_tokens"], 64)
        with self.assertRaises(probe.ProbeHold):
            probe.execute_once(self.artifacts, self.canonical, live=True, transport=transport)
        self.assertEqual(len(transport.calls), 1)

    def test_timeout_and_disconnect_consume_no_retry(self):
        for cls in (TimeoutError, ConnectionResetError):
            with self.subTest(error=cls.__name__), tempfile.TemporaryDirectory() as d:
                # Each subcase is a new synthetic experiment, never the live budget.
                state = Path(d) / "state"
                ledger = probe.ProbeBudget(state, self.binding, install=True)
                try:
                    attempt = ledger.reserve_probe(probe.body())
                    transport = MockTransport(failure=cls())
                    with self.assertRaises(cls):
                        transport.send(probe.body())
                    ledger.settle(attempt, "UNCERTAIN", {"error_type": cls.__name__})
                finally:
                    ledger.close()
                reopened = probe.ProbeBudget(state, self.binding)
                try:
                    self.assertEqual(reopened.snapshot()["attempts_consumed"], 1)
                    with self.assertRaises(probe.ProbeHold):
                        reopened.reserve_probe(probe.body())
                finally:
                    reopened.close()
                self.assertEqual(len(transport.calls), 1)

    def test_global12_cap_no_13th(self):
        b = self.budget(True)
        try:
            for i in range(12):
                b.connection.execute("INSERT INTO attempts(purpose,status,reserved_utc,request_sha256,max_tokens) VALUES (?,?,?,?,?)",
                    ("SYNTHETIC_EXISTING_SLOT_" + str(i), "UNCERTAIN", "fixture", "0" * 64, 4096))
            with self.assertRaises(probe.ProbeHold):
                b.reserve_probe(probe.body())
            self.assertEqual(b.snapshot()["attempts_consumed"], 12)
        finally:
            b.close()

    def test_missing_existing_ledger_is_not_recreated(self):
        b = self.budget(True)
        b.reserve_probe(probe.body())
        b.close()
        (self.canonical / probe.LEDGER_NAME).unlink()
        with self.assertRaisesRegex(probe.ProbeHold, "never recreate/reset"):
            self.budget()
        self.assertFalse((self.canonical / probe.LEDGER_NAME).exists())

    def test_payload_is_exact_and_synthetic(self):
        expected = probe.body()
        self.assertEqual(set(expected), {"model", "messages", "temperature", "max_tokens", "n", "stream"})
        self.assertEqual(len(expected["messages"]), 1)
        for key, value in (("max_tokens", 4096), ("max_tokens", 65), ("n", 2), ("model", "other"),
                           ("response_format", {"type": "json_object"}), ("retries", 0)):
            modified = copy.deepcopy(expected)
            modified[key] = value
            with self.subTest(key=key), self.assertRaises(probe.ProbeHold):
                probe.validate_body(modified)
        self.assertFalse(any(word in json.dumps(expected).lower() for word in ("root", "gold", "certificate")))

    def test_response_checks_model_usage_finish_and_confirmation(self):
        good = probe.validate_observation(probe.response_observation(envelope()))
        self.assertTrue(good["response_valid"])
        self.assertIsNone(good["reasoning_tokens"])
        self.assertFalse(good["server_4096_hard_limit_proved"])
        bad = [envelope(model="other"), envelope(status=302), envelope(finish="length"), envelope(text="analysis OK"),
               envelope(usage={}), envelope(usage={"prompt_tokens": 19, "completion_tokens": 65, "total_tokens": 84}),
               envelope(usage={"prompt_tokens": 19, "completion_tokens": 2, "total_tokens": 22}),
               envelope(usage={"prompt_tokens": 19, "completion_tokens": 2, "total_tokens": 21, "reasoning_tokens": 3})]
        for response in bad:
            with self.subTest(response=response), self.assertRaises(probe.ProbeHold):
                probe.validate_observation(probe.response_observation(response))

    def test_transport_one_post_no_retry_on_failure(self):
        for error in (TimeoutError(), ConnectionResetError()):
            conn = Mock()
            conn.request.side_effect = error
            with patch.object(probe.old_transport, "_read_credential", return_value="synthetic-private-key"), \
                 patch.object(probe.http.client, "HTTPSConnection", return_value=conn) as factory:
                client = probe.SingleProbeTransport(Path("/not-read"), explicitly_live=True)
                with self.assertRaises(probe.ProbeHold):
                    client.send(probe.body())
                conn.request.assert_called_once()
                factory.assert_called_once()
                with self.assertRaises(probe.ProbeHold):
                    client.send(probe.body())
                conn.request.assert_called_once()

    def test_transport_default_no_secret_or_socket(self):
        with patch.object(probe.old_transport, "_read_credential") as secret, \
             patch.object(probe.http.client, "HTTPSConnection") as connection:
            with self.assertRaises(probe.ProbeHold):
                probe.SingleProbeTransport(Path("/not-read")).send(probe.body())
            secret.assert_not_called()
            connection.assert_not_called()

    def test_credential_echo_in_headers_is_withheld(self):
        conn = Mock()
        response = Mock(status=200)
        response.read.return_value = envelope()["raw"]
        response.getheader.return_value = "synthetic-private-key"
        conn.getresponse.return_value = response
        with patch.object(probe.old_transport, "_read_credential", return_value="synthetic-private-key"), \
             patch.object(probe.http.client, "HTTPSConnection", return_value=conn):
            client = probe.SingleProbeTransport(Path("/not-read"), explicitly_live=True)
            with self.assertRaisesRegex(probe.ProbeHold, "Credential header echo withheld") as error:
                client.send(probe.body())
            self.assertNotIn("synthetic-private-key", str(error.exception))
            conn.request.assert_called_once()

    def test_json_escaped_credential_echo_is_withheld(self):
        conn = Mock()
        response = Mock(status=200)
        secret = "synthetic-private-key"
        escaped = "".join("\\u%04x" % ord(c) for c in secret)
        response.read.return_value = ('{"echo":"' + escaped + '"}').encode()
        response.getheader.return_value = None
        conn.getresponse.return_value = response
        with patch.object(probe.old_transport, "_read_credential", return_value=secret), \
             patch.object(probe.http.client, "HTTPSConnection", return_value=conn):
            with self.assertRaisesRegex(probe.ProbeHold, "Escaped credential echo withheld") as error:
                probe.SingleProbeTransport(Path("/not-read"), explicitly_live=True).send(probe.body())
            self.assertNotIn(secret, str(error.exception))
            conn.request.assert_called_once()

    def test_runtime_takeover_blocks_frozen_v2_init_and_run(self):
        b = self.budget(True)
        b.close()
        from . import base_qa_executor_v2 as old
        with self.assertRaises((FileNotFoundError, ValueError, old.ExecutionHold)):
            old.Executor(self.canonical)
        self.assertTrue(self.canonical.is_dir())
        self.assertFalse((self.canonical / "EXPERIMENT.json").exists())

    def test_invalid_response_stops_without_replay(self):
        b = self.budget(True)
        b.close()
        transport = MockTransport(response=envelope(model="other"))
        result = probe.execute_once(self.artifacts, self.canonical, live=True, transport=transport)
        self.assertEqual(result["status"], "HOLD_NO_RETRY")
        self.assertEqual(result["reason"], "MODEL_MISMATCH_STOP")
        self.assertEqual(result["budget"]["remaining_global_attempts"], 11)
        self.assertEqual(len(transport.calls), 1)


if __name__ == "__main__":
    unittest.main()
