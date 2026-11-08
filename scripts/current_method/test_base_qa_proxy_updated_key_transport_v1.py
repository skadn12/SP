"""Network-forbidden preparation tests; all budget writes use temp ledgers."""
from __future__ import annotations

import copy
import contextlib
import http.client
import io
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import base_qa_proxy_updated_key_transport_v1 as proxy
from . import base_qa_capability_probe_v1 as probe

ENV = {"HTTPS_PROXY": "http://127.0.0.1:7897"}
KEY = "synthetic-key-never-read-from-disk"


class ProxyTransportTests(unittest.TestCase):
    def setUp(self):
        for name in ("socket", "create_connection", "getaddrinfo"):
            guard = patch.object(socket, name, side_effect=AssertionError("NETWORK FORBIDDEN"))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.case_number = 0
        self.experiment = self.base / "synthetic_experiment.json"
        probe.durable_write(self.experiment, {"synthetic": True})
        self.binding = probe.file_binding(self.experiment)
        self.ledger = probe.ProbeBudget(self.base / "ledger", self.binding, install=True)
        self.addCleanup(self.ledger.close)
        for attr, value in (("LIVE_LEDGER", self.ledger.path), ("LIVE_EXPERIMENT", self.experiment)):
            guard = patch.object(proxy, attr, value)
            guard.start()
            self.addCleanup(guard.stop)
        env = patch.dict(proxy.os.environ, ENV, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def reserve(self):
        return proxy.reserve_once(self.ledger, probe.digest(self.ledger.snapshot()), probe.body())

    def fresh_reservation(self):
        self.case_number += 1
        ledger = probe.ProbeBudget(self.base / ("case_%d" % self.case_number), self.binding, install=True)
        try:
            proxy.LIVE_LEDGER = ledger.path
            return proxy.reserve_once(ledger, probe.digest(ledger.snapshot()), probe.body())
        finally:
            ledger.close()

    def response(self, **changes):
        data = {"id": "mock-proxy", "model": "qwen3.6-35b-a3b",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}}
        data.update(changes)
        response = Mock(status=200)
        response.read.return_value = json.dumps(data).encode()
        response.getheader.return_value = None
        return response

    def test_dry_run_neither_reads_key_nor_creates_connection_or_reservation(self):
        with patch.object(probe.old_transport, "_read_credential") as key, \
             patch.object(proxy, "TunnelConnection") as connection:
            self.assertEqual(proxy.offline_plan()["api_requests"], 0)
            with self.assertRaisesRegex(probe.ProbeHold, "DRY_RUN"):
                proxy.ProxyProbeTransport("/not-read").send(probe.body())
            key.assert_not_called()
            connection.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["attempts_consumed"], 0)

    def test_unreserved_live_send_stops_before_key_or_network(self):
        with patch.object(probe.old_transport, "_read_credential") as key, \
             patch.object(proxy, "TunnelConnection") as connection:
            with self.assertRaisesRegex(probe.ProbeHold, "COMMITTED_RESERVATION_REQUIRED"):
                proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(probe.body())
            key.assert_not_called()
            connection.assert_not_called()

    def test_proxy_drift_credentials_bypass_and_schemes_rejected(self):
        bad = [{}, {"HTTPS_PROXY": "https://127.0.0.1:7897"},
               {"HTTPS_PROXY": "http://127.0.0.1:7898"},
               {"HTTPS_PROXY": "http://user:secret@127.0.0.1:7897"},
               {"HTTPS_PROXY": "http://example.invalid:7897"},
               {"HTTPS_PROXY": "http://127.0.0.1:bad"},
               {**ENV, "https_proxy": "http://127.0.0.1:7898"},
               {**ENV, "NO_PROXY": "ai.tju.edu.cn"},
               {"HTTPS_PROXY": "http://127.0.0.1:7897/unapproved"}]
        for env in bad:
            with self.subTest(env=env), self.assertRaises(probe.ProbeHold):
                proxy.validate_proxy_environment(env)

    def test_payload_remains_exact_synthetic_64(self):
        for field, value in [("max_tokens", 4096), ("model", "other"), ("n", 2)]:
            request = copy.deepcopy(probe.body())
            request[field] = value
            with self.assertRaises(probe.ProbeHold):
                proxy.reserve_once(self.ledger, probe.digest(self.ledger.snapshot()), request)
        self.assertEqual(self.ledger.snapshot()["attempts_consumed"], 0)

    def test_reservation_visible_from_another_connection_before_send(self):
        reservation = self.reserve()
        reservation.verify(probe.body())
        self.assertEqual(self.ledger.snapshot()["attempts"][0]["status"], "RESERVED")

    def test_normal_single_post_and_response_checks(self):
        reservation = self.reserve()
        conn = Mock()
        conn.getresponse.return_value = self.response()
        client = proxy.ProxyProbeTransport("/not-read", explicitly_live=True)
        def connected():
            reservation.verify(probe.body())
        conn.connect.side_effect = connected
        with patch.object(probe.old_transport, "_read_credential", return_value=KEY), \
             patch.object(proxy, "TunnelConnection", return_value=conn) as factory:
            envelope = client.send(probe.body(), reservation=reservation)
            validated = probe.validate_observation(probe.response_observation(envelope))
            self.assertTrue(validated["response_valid"])
            self.assertFalse(validated["server_4096_hard_limit_proved"])
            factory.assert_called_once()
            conn.connect.assert_called_once()
            conn.request.assert_called_once()
            self.assertEqual(conn.request.call_args.args[:2], ("POST", proxy.TARGET_PATH))
            kwargs = conn.request.call_args.kwargs
            self.assertEqual(json.loads(kwargs["body"])["max_tokens"], 64)
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer " + KEY)
            with self.assertRaises(probe.ProbeHold):
                client.send(probe.body(), reservation=reservation)
            conn.request.assert_called_once()
            conn.close.assert_called_once()

    def test_connect_and_request_failures_never_retry_or_fallback(self):
        for phase in ("connect", "request", "getresponse"):
            for error in (TimeoutError(KEY), ConnectionResetError(KEY),
                          OSError("407 proxy auth " + KEY), ssl.SSLError(KEY)):
                reservation = self.fresh_reservation()
                conn = Mock()
                getattr(conn, phase).side_effect = error
                client = proxy.ProxyProbeTransport("/not-read", explicitly_live=True)
                with self.subTest(phase=phase, error=type(error).__name__), \
                     patch.object(probe.old_transport, "_read_credential", return_value=KEY), \
                     patch.object(proxy, "TunnelConnection", return_value=conn) as factory:
                    with self.assertRaises(probe.ProbeHold) as caught:
                        client.send(probe.body(), reservation=reservation)
                    self.assertNotIn(KEY, str(caught.exception))
                    self.assertNotIn(KEY, json.dumps(client.trace))
                    factory.assert_called_once()
                    conn.connect.assert_called_once()
                    self.assertEqual(conn.request.call_count, 0 if phase == "connect" else 1)
                    with self.assertRaises(probe.ProbeHold):
                        client.send(probe.body(), reservation=reservation)
                    factory.assert_called_once()
                    conn.close.assert_called_once()

    def test_read_disconnect_never_retries(self):
        reservation = self.reserve()
        conn = Mock()
        conn.getresponse.return_value.read.side_effect = ConnectionResetError(KEY)
        client = proxy.ProxyProbeTransport("/not-read", explicitly_live=True)
        with patch.object(probe.old_transport, "_read_credential", return_value=KEY), \
             patch.object(proxy, "TunnelConnection", return_value=conn):
            with self.assertRaises(probe.ProbeHold):
                client.send(probe.body(), reservation=reservation)
        self.assertEqual(client.trace["phase"], "READ_RESPONSE_BODY")
        conn.request.assert_called_once()

    def test_connect_target_and_no_auth_on_tunnel(self):
        context = Mock(check_hostname=True, verify_mode=ssl.CERT_REQUIRED)
        trace = {}
        with patch.object(proxy.ssl, "create_default_context", return_value=context):
            conn = proxy.TunnelConnection(trace)
        self.assertEqual((conn.host, conn.port), ("127.0.0.1", 7897))
        self.assertEqual((conn._tunnel_host, conn._tunnel_port), ("ai.tju.edu.cn", 443))
        self.assertEqual(conn._tunnel_headers, {})
        self.assertEqual(conn.auto_open, 0)
        sock = Mock()
        def tcp_connect(instance):
            self.assertEqual(trace["phase"], "PROXY_TCP_CONNECT")
            instance.sock = sock
        with patch.object(http.client.HTTPConnection, "connect", autospec=True, side_effect=tcp_connect):
            conn.connect()
        context.wrap_socket.assert_called_once_with(sock, server_hostname="ai.tju.edu.cn")
        self.assertEqual(trace["phase"], "TARGET_TLS_HANDSHAKE")

    def test_tls_uses_verification_and_proxy_tunnel_phase(self):
        conn = proxy.TunnelConnection({})
        self.assertTrue(conn._context.check_hostname)
        self.assertEqual(conn._context.verify_mode, ssl.CERT_REQUIRED)
        with patch.object(http.client.HTTPConnection, "_tunnel") as tunnel:
            conn._tunnel()
        tunnel.assert_called_once()
        self.assertEqual(conn.trace["phase"], "PROXY_CONNECT_TUNNEL")

    def test_no_hidden_http_reconnect_when_socket_missing(self):
        conn = proxy.TunnelConnection({})
        with patch.object(conn, "connect") as connect:
            with self.assertRaises(http.client.NotConnected):
                conn.send(b"synthetic")
        connect.assert_not_called()

    def test_restart_cannot_reserve_same_purpose(self):
        self.reserve()
        reopened = probe.ProbeBudget(self.base / "ledger", self.binding)
        try:
            with self.assertRaises(probe.ProbeHold):
                proxy.reserve_once(reopened, probe.digest(reopened.snapshot()), probe.body())
            self.assertEqual(reopened.snapshot()["attempts_consumed"], 1)
        finally:
            reopened.close()

    def test_settled_attempt_cannot_reuse_reservation_or_purpose(self):
        reservation = self.reserve()
        self.ledger.settle(reservation.attempt_id, "UNCERTAIN", {"mock": True})
        with self.assertRaises(probe.ProbeHold):
            reservation.verify(probe.body())
        with self.assertRaises(probe.ProbeHold):
            self.reserve()
        self.assertEqual(self.ledger.snapshot()["attempts_consumed"], 1)

    def test_wrong_ledger_path_cannot_enable_network(self):
        reservation = self.reserve()
        alternate = self.base / "copy.sqlite3"
        alternate.write_bytes(self.ledger.path.read_bytes())
        forged = proxy.CommittedReservation(alternate, reservation.attempt_id, reservation.request_sha256)
        with patch.object(probe.old_transport, "_read_credential") as key, \
             patch.object(proxy, "TunnelConnection") as connection:
            with self.assertRaisesRegex(probe.ProbeHold, "CANONICAL_GLOBAL_LEDGER_REQUIRED"):
                proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(
                    probe.body(), reservation=forged)
            key.assert_not_called()
            connection.assert_not_called()

    def test_original_guard_and_policy_are_required(self):
        reservation = self.reserve()
        self.ledger.connection.execute("UPDATE policy SET value='{}' WHERE singleton=1")
        with self.assertRaisesRegex(probe.ProbeHold, "POLICY_DRIFT"):
            reservation.verify(probe.body())
        guard = self.ledger.path.parent / "BUDGET_TAKEOVER.json"
        guard.write_text("{}")
        with self.assertRaisesRegex(probe.ProbeHold, "GUARD_DRIFT"):
            reservation.verify(probe.body())

    def test_global_twelve_and_snapshot_drift_fail_before_reserving(self):
        with self.assertRaisesRegex(probe.ProbeHold, "PRIOR_BUDGET_DRIFT"):
            proxy.reserve_once(self.ledger, "0" * 64, probe.body())
        for i in range(12):
            self.ledger.connection.execute(
                "INSERT INTO attempts(purpose,status,reserved_utc,request_sha256,max_tokens) VALUES (?,?,?,?,?)",
                ("synthetic_" + str(i), "UNCERTAIN", "fixture", "0" * 64, 64))
        with self.assertRaisesRegex(probe.ProbeHold, "GLOBAL_12"):
            self.reserve()
        self.assertEqual(self.ledger.snapshot()["attempts_consumed"], 12)

    def test_credential_echo_and_invalid_response_remain_hold(self):
        for response in (self.response(model="wrong"), self.response(usage={})):
            reservation = self.fresh_reservation()
            conn = Mock()
            conn.getresponse.return_value = response
            with patch.object(probe.old_transport, "_read_credential", return_value=KEY), \
                 patch.object(proxy, "TunnelConnection", return_value=conn):
                result = proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(
                    probe.body(), reservation=reservation)
            with self.assertRaises(probe.ProbeHold):
                probe.validate_observation(probe.response_observation(result))
        conn = Mock()
        conn.getresponse.return_value = self.response(echo=KEY)
        reservation = self.fresh_reservation()
        with patch.object(probe.old_transport, "_read_credential", return_value=KEY), \
             patch.object(proxy, "TunnelConnection", return_value=conn):
            with self.assertRaises(probe.ProbeHold):
                proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(
                    probe.body(), reservation=reservation)

    def test_new_transport_instance_cannot_reuse_reserved_slot(self):
        reservation = self.reserve()
        conn = Mock()
        conn.getresponse.return_value = self.response()
        with patch.object(probe.old_transport, "_read_credential", return_value=KEY) as key, \
             patch.object(proxy, "TunnelConnection", return_value=conn) as factory:
            proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(
                probe.body(), reservation=reservation)
            with self.assertRaises(probe.ProbeHold):
                proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(
                    probe.body(), reservation=reservation)
            factory.assert_called_once()
            key.assert_called_once()
            conn.request.assert_called_once()

    def test_failed_claim_write_stops_before_key_or_connection(self):
        reservation = self.reserve()
        with patch.object(probe, "durable_write", side_effect=OSError("synthetic disk failure")), \
             patch.object(probe.old_transport, "_read_credential") as key, \
             patch.object(proxy, "TunnelConnection") as factory:
            with self.assertRaises(probe.ProbeHold):
                proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(
                    probe.body(), reservation=reservation)
            key.assert_not_called()
            factory.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["attempts_consumed"], 1)
        self.assertEqual(self.ledger.snapshot()["attempts"][0]["status"], "RESERVED")

    def test_fresh_process_observes_committed_budget_and_exclusive_claim(self):
        reservation = self.reserve()
        conn = Mock()
        conn.getresponse.return_value = self.response()
        with patch.object(probe.old_transport, "_read_credential", return_value=KEY), \
             patch.object(proxy, "TunnelConnection", return_value=conn):
            proxy.ProxyProbeTransport("/not-read", explicitly_live=True).send(
                probe.body(), reservation=reservation)
        script = '''import pathlib,sqlite3,sys
def deny(event,args):
    if event.startswith("socket."): raise AssertionError("NETWORK FORBIDDEN")
sys.addaudithook(deny)
p=pathlib.Path(sys.argv[1])
with sqlite3.connect(p.as_uri()+"?mode=ro",uri=True) as db:
    assert db.execute("select count(*) from attempts").fetchone()[0]==1
    assert db.execute("select status from attempts").fetchone()[0]=="RESERVED"
claim=p.parent/"PROXY_SEND_CLAIM_1.json"
try:
    claim.open("x")
except FileExistsError:
    print("COMMITTED_RESERVATION_AND_SEND_CLAIM_PERSIST")
else:
    raise AssertionError("Duplicate send claim accepted")
'''
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.ledger.path)],
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertIn("SEND_CLAIM_PERSIST", result.stdout)

    def test_cli_rejects_live_flag_before_network_or_key(self):
        with patch.object(sys, "argv", ["proxy-prep", "--live"]), \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as caught:
            proxy.main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
