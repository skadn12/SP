"""Offline-prepared, proxy-only transport. There is no live command-line runner.

Reuse the original synthetic payload, response validation and global ledger.
A later, separately authorized caller must reserve exactly one attempt before
calling this transport. Importing or running this module makes no requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import os
from pathlib import Path
import sqlite3
import ssl
from urllib.parse import urlsplit

from . import base_qa_capability_probe_v1 as probe

VERSION = "base_qa_proxy_updated_key_probe_20261030_v1"
PURPOSE = "UPDATED_KEY_PROXY_PROBE_20261030_V1"
PROXY_HOST, PROXY_PORT = "127.0.0.1", 7897
TARGET_HOST, TARGET_PORT = "ai.tju.edu.cn", 443
TARGET_PATH = "/api/v3/chat/completions"
TIMEOUT = 60
LIVE_LEDGER = probe.CANONICAL / probe.LEDGER_NAME
LIVE_EXPERIMENT = probe.ARTIFACTS / "EXPERIMENT_PROBE.json"


def validate_canonical_state(path):
    path = Path(path)
    probe.require(path.resolve() == LIVE_LEDGER.resolve() and not path.is_symlink() and
                  not path.parent.is_symlink() and path.is_file(), "CANONICAL_GLOBAL_LEDGER_REQUIRED")
    binding = probe.file_binding(LIVE_EXPERIMENT)
    probe.require(probe.read(path.parent / "BUDGET_TAKEOVER.json") ==
                  probe.takeover_value(binding, path.parent), "ORIGINAL_BUDGET_GUARD_DRIFT")
    return {"schema": probe.VERSION, "original_global_limit": 12,
            "experiment_binding": binding, "canonical_directory": str(path.parent.resolve())}


def validate_proxy_environment(environment=None):
    """Inspect strings only. No discovery, DNS, sockets or credential reads."""
    env = os.environ if environment is None else environment
    values = [env[k] for k in ("HTTPS_PROXY", "https_proxy") if env.get(k)]
    probe.require(bool(values), "EXPLICIT_APPROVED_HTTPS_PROXY_REQUIRED")
    probe.require(not any(env.get(k) for k in ("NO_PROXY", "no_proxy")),
                  "PROXY_BYPASS_CONFIGURATION_HOLD")
    for value in values:
        try:
            parsed = urlsplit(value)
            valid = (parsed.scheme == "http" and parsed.hostname == PROXY_HOST and
                     parsed.port == PROXY_PORT and parsed.username is None and
                     parsed.password is None and parsed.path in ("", "/") and
                     not parsed.query and not parsed.fragment)
        except (TypeError, ValueError):
            valid = False
        probe.require(valid, "APPROVED_PROXY_CONFIGURATION_DRIFT")
    return {"scheme": "http", "host": PROXY_HOST, "port": PROXY_PORT,
            "direct_fallback": False}


@dataclass(frozen=True)
class CommittedReservation:
    ledger_path: Path
    attempt_id: int
    request_sha256: str

    def verify(self, request):
        probe.validate_body(request)
        probe.require(self.request_sha256 == probe.digest(request), "RESERVATION_REQUEST_DRIFT")
        policy = validate_canonical_state(self.ledger_path)
        with sqlite3.connect(self.ledger_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            stored = db.execute("SELECT value FROM policy WHERE singleton=1").fetchone()
            probe.require(stored is not None and probe.old_transport._strict_json(stored[0]) == policy,
                          "ORIGINAL_BUDGET_POLICY_DRIFT")
            row = db.execute("SELECT purpose,status,request_sha256,max_tokens FROM attempts WHERE attempt_id=?",
                             (self.attempt_id,)).fetchone()
            probe.require(row == (PURPOSE, "RESERVED", self.request_sha256, 64),
                          "DURABLY_COMMITTED_RESERVATION_REQUIRED")
            probe.require(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] <= 12,
                          "GLOBAL_BUDGET_INVALID")


def reserve_once(ledger, expected_snapshot_sha256, request):
    """Future caller only: same existing ledger, one unique purpose, no reset.

    This helper is tested exclusively on temporary ledgers in this release.
    The transaction commits before the returned reservation can enable send.
    """
    probe.validate_body(request)
    validate_canonical_state(ledger.path)
    ledger.connection.execute("BEGIN IMMEDIATE")
    try:
        snapshot = ledger.snapshot()
        probe.require(probe.digest(snapshot) == expected_snapshot_sha256, "PRIOR_BUDGET_DRIFT")
        probe.require(snapshot["attempts_consumed"] < 12, "GLOBAL_12_ATTEMPT_LIMIT")
        probe.require(all(r["status"] != "RESERVED" for r in snapshot["attempts"]),
                      "UNSETTLED_ATTEMPT_HOLD")
        probe.require(not any(r["purpose"] == PURPOSE for r in snapshot["attempts"]),
                      "PROXY_PROBE_ALREADY_ATTEMPTED_NO_RETRY")
        cursor = ledger.connection.execute(
            "INSERT INTO attempts(purpose,status,reserved_utc,request_sha256,max_tokens) VALUES (?,?,?,?,?)",
            (PURPOSE, "RESERVED", datetime.now(timezone.utc).isoformat(), probe.digest(request), 64))
        ledger.connection.execute("COMMIT")
        return CommittedReservation(ledger.path, cursor.lastrowid, probe.digest(request))
    except BaseException:
        ledger.connection.execute("ROLLBACK")
        raise


class TunnelConnection(http.client.HTTPSConnection):
    """One HTTP CONNECT tunnel, TLS verified against the school domain."""
    def __init__(self, trace):
        self.trace = trace
        super().__init__(PROXY_HOST, PROXY_PORT, timeout=TIMEOUT,
                         context=ssl.create_default_context())
        self.set_tunnel(TARGET_HOST, TARGET_PORT, headers={})
        # No implicit reopening by HTTPConnection.send after explicit connect.
        self.auto_open = 0

    def _tunnel(self):
        self.trace["phase"] = "PROXY_CONNECT_TUNNEL"
        return super()._tunnel()

    def connect(self):
        self.trace["phase"] = "PROXY_TCP_CONNECT"
        http.client.HTTPConnection.connect(self)
        self.trace["phase"] = "TARGET_TLS_HANDSHAKE"
        self.sock = self._context.wrap_socket(self.sock, server_hostname=TARGET_HOST)


class ProxyProbeTransport:
    """One reserved synthetic POST. Default disabled; no retry or direct path."""
    def __init__(self, credential_file, *, explicitly_live=False):
        self.credential_file = Path(credential_file)
        self.live = explicitly_live
        self.used = False
        self.trace = {"phase": "NOT_STARTED", "underlying_error_type": None,
                      "errno": None, "retry_count": 0, "direct_fallback": False}

    def send(self, request, *, reservation=None):
        probe.validate_body(request)
        probe.require(self.live is True, "DRY_RUN_NO_REQUEST")
        probe.require(not self.used, "TRANSPORT_ALREADY_USED_NO_RETRY")
        self.used = True
        validate_proxy_environment()
        probe.require(isinstance(reservation, CommittedReservation), "COMMITTED_RESERVATION_REQUIRED")
        reservation.verify(request)
        connection = None
        try:
            self.trace["phase"] = "DURABLE_SEND_CLAIM"
            # O_EXCL + fsync: another instance/process cannot reuse this slot,
            # even if the caller crashes before recording an HTTP outcome.
            probe.durable_write(
                reservation.ledger_path.parent / ("PROXY_SEND_CLAIM_%d.json" % reservation.attempt_id),
                {"version": VERSION, "purpose": PURPOSE, "attempt_id": reservation.attempt_id,
                 "request_sha256": reservation.request_sha256,
                 "proxy": "http://127.0.0.1:7897", "max_tokens": 64})
            self.trace["phase"] = "LOCAL_CREDENTIAL_READ"
            key = probe.old_transport._read_credential(self.credential_file)
            connection = TunnelConnection(self.trace)
            # CONNECT carries no school credential. Bearer is sent only after TLS.
            connection.connect()
            self.trace["phase"] = "API_REQUEST_SEND"
            connection.request("POST", TARGET_PATH, body=probe.canonical(request).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                         "Accept": "application/json"})
            self.trace["phase"] = "AWAIT_RESPONSE_HEADERS"
            response = connection.getresponse()
            self.trace["http_status"] = response.status
            self.trace["phase"] = "READ_RESPONSE_BODY"
            raw = response.read(1024 * 1024 + 1)
            probe.require(len(raw) <= 1024 * 1024, "RESPONSE_TOO_LARGE")
            probe.require(key.encode() not in raw, "CREDENTIAL_ECHO_WITHHELD")
            decoded = probe.old_transport._strict_json(raw)
            probe.require(key not in probe.canonical(decoded), "ESCAPED_CREDENTIAL_ECHO_WITHHELD")
            headers = {}
            for name in ("x-request-id", "request-id", "x-correlation-id"):
                value = response.getheader(name)
                probe.require(not isinstance(value, str) or key not in value,
                              "CREDENTIAL_HEADER_ECHO_WITHHELD")
                headers[name] = probe.old_transport._safe_id(value)
            self.trace["phase"] = "RESPONSE_RECEIVED"
            return {"http_status": response.status, "headers": headers,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw": raw}
        except BaseException as error:
            self.trace["underlying_error_type"] = type(error).__name__
            number = getattr(error, "errno", None)
            self.trace["errno"] = number if type(number) is int else None
            # No raw exception text: proxy/server exceptions can contain secrets.
            raise probe.ProbeHold("PROXY_PROBE_FAILED_NO_RETRY") from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass


def offline_plan():
    return {"version": VERSION, "status": "OFFLINE_PREPARATION_ONLY",
            "live_authorized": False, "live_runner_present": False,
            "api_requests": 0, "proxy": validate_proxy_environment(),
            "request": probe.body(), "qa_authorized": False,
            "requires_separate_authorization_and_budget_reservation": True}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # Deliberately offers no live flag.
    print(json.dumps(offline_plan(), ensure_ascii=False))


if __name__ == "__main__":
    main()
