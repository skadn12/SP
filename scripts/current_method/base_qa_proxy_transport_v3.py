"""Single-attempt proxy transport for the versioned QA executor adapter.

Live mode requires explicit capabilities and an authoritative reservation.
Tests may inject in-memory connections only in mock mode; there is no direct
fallback, SDK, redirect handling, or retry implementation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from . import base_qa_global_budget_adapter_v3 as budget
from . import base_qa_proxy_updated_key_transport_v1 as proxy_probe
from . import qa_api_transport_v2 as transport_api

MAX_RESPONSE_BYTES = transport_api.MAX_RESPONSE_BYTES
LIVE_SEND_ENABLED = False


class ProxyTransportError(RuntimeError):
    def __init__(self, reason, *, uncertain, receipt=None):
        self.reason = reason
        self.uncertain = bool(uncertain)
        self.receipt = {"error_type": "ProxyTransportError", "reason_code": reason,
                        "uncertain": self.uncertain, "retry_count": 0,
                        "transport_attempt_count": 1, **(receipt or {})}
        super().__init__(reason)


class ProxyTransportV3:
    def __init__(self, budget_adapter, *, mode="dry-run", credential_file=None,
                 capability=None, proxy_environment=None, connection_factory=None,
                 credential_reader=None):
        if mode not in ("dry-run", "mock", "live"):
            raise ProxyTransportError("unknown_transport_mode", uncertain=False)
        self.budget = budget_adapter
        self.mode = mode
        self.credential_file = Path(credential_file) if credential_file is not None else None
        self.capability = capability
        self.proxy_environment = proxy_environment
        self.connection_factory = connection_factory
        self.credential_reader = credential_reader
        self._used_attempts = set()
        self.trace = {"phase": "NOT_STARTED", "retry_count": 0, "direct_fallback": False}

        if mode == "mock":
            if not callable(connection_factory) or not callable(credential_reader):
                raise ProxyTransportError("mock transport requires isolated connector and credential stub",
                                          uncertain=False)
        elif connection_factory is not None or credential_reader is not None:
            raise ProxyTransportError("custom connector injection is mock-only", uncertain=False)
        elif mode == "live":
            if not LIVE_SEND_ENABLED:
                raise ProxyTransportError("v3 live send is disabled pending capability review", uncertain=False)
            self._validate_capability(capability)
            if self.credential_file is None:
                raise ProxyTransportError("private_credential_file_required", uncertain=False)

    @staticmethod
    def _validate_capability(capability):
        try:
            transport_api.validate_capability(capability)
            binding = capability.get("evidence_binding")
            if not isinstance(binding, dict):
                raise ValueError("evidence_binding_missing")
            path = Path(binding["path"])
            raw = path.read_bytes()
            observed = {"path": str(path.resolve()), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()}
            if observed != binding:
                raise ValueError("capability_evidence_drift")
        except Exception:
            raise ProxyTransportError("capability_evidence_unverified", uncertain=False) from None

    def validate_configuration(self):
        if self.mode == "dry-run":
            raise ProxyTransportError("dry_run_network_disabled", uncertain=False)
        try:
            proxy_probe.validate_proxy_environment(self.proxy_environment)
        except Exception:
            raise ProxyTransportError("approved_proxy_configuration_required", uncertain=False) from None
        if self.mode == "live":
            if self.budget.mode != "live":
                raise ProxyTransportError("authoritative_live_budget_required", uncertain=False)
            try:
                info = os.stat(self.credential_file, follow_symlinks=False)
                if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or
                        info.st_uid != os.getuid()):
                    raise OSError("permissions")
            except OSError:
                raise ProxyTransportError("private_credential_file_unavailable", uncertain=False) from None
        return {"proxy": "http://127.0.0.1:7897", "direct_fallback": False,
                "automatic_retries": 0, "redirects": False}

    def _send_claim(self, reservation):
        if not isinstance(reservation, budget.Reservation):
            raise ProxyTransportError("committed_reservation_required", uncertain=False)
        if self.mode == "live" and not reservation.authoritative:
            raise ProxyTransportError("non_authoritative_reservation_rejected", uncertain=False)
        claims = self.budget.state_dir / "ATTEMPT_SEND_CLAIMS"
        claims.mkdir(parents=True, exist_ok=True, mode=0o700)
        if claims.is_symlink():
            raise ProxyTransportError("send_claim_directory_redirected", uncertain=False)
        path = claims / ("attempt-%d.json" % reservation.attempt_id)
        try:
            budget._durable_exclusive_json(path, {
                "version": budget.VERSION, "attempt_id": reservation.attempt_id,
                "purpose": reservation.purpose, "request_sha256": reservation.request_sha256,
                "max_tokens": budget.MAX_OUTPUT, "proxy": "http://127.0.0.1:7897"})
        except FileExistsError:
            raise ProxyTransportError("send_claim_exists_no_retry", uncertain=False) from None
        return path

    def send(self, body, reservation):
        transport_api.validate_body(body)
        self.budget.verify_reservation(reservation, body)
        self.validate_configuration()
        if reservation.attempt_id in self._used_attempts:
            raise ProxyTransportError("transport_attempt_reused_no_retry", uncertain=False)
        self._used_attempts.add(reservation.attempt_id)
        self._send_claim(reservation)

        connection = None
        request_attempted = False
        response_complete = False
        try:
            self.trace = {"phase": "DURABLE_SEND_CLAIM", "retry_count": 0, "direct_fallback": False}
            if self.mode == "mock":
                credential = self.credential_reader()
                connection = self.connection_factory(self.trace)
            else:
                self.trace["phase"] = "LOCAL_CREDENTIAL_READ"
                credential = transport_api._read_credential(self.credential_file)
                connection = proxy_probe.TunnelConnection(self.trace)

            connection.connect()
            self.trace["phase"] = "API_REQUEST_SEND"
            request_attempted = True
            connection.request("POST", proxy_probe.TARGET_PATH,
                body=transport_api._compact(body).encode("utf-8"),
                headers={"Authorization": "Bearer " + credential,
                         "Content-Type": "application/json", "Accept": "application/json"})
            self.trace["phase"] = "AWAIT_RESPONSE_HEADERS"
            response = connection.getresponse()
            self.trace["http_status"] = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            response_complete = True
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ProxyTransportError("response_size_limit_exceeded", uncertain=True)
            if credential.encode("utf-8") in raw:
                raise ProxyTransportError("credential_echo_rejected", uncertain=True)
            data = transport_api._strict_json(raw)
            if not isinstance(data, dict) or credential in transport_api._compact(data):
                raise ProxyTransportError("invalid_or_credential_echo_response", uncertain=True)
            if response.status != 200:
                raise ProxyTransportError("non_200_response_no_retry", uncertain=False,
                                          receipt={"http_status": response.status})
            headers = {}
            for name in ("x-request-id", "request-id", "x-correlation-id", "cf-ray"):
                value = transport_api._safe_id(response.getheader(name))
                if value:
                    headers[name] = value
            envelope = {"http_status": response.status, "body": data,
                        "response_headers": headers,
                        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
                        "request_id": headers.get("x-request-id") or headers.get("request-id") or
                                      transport_api._safe_id(data.get("id"))}
            value, receipt = transport_api.parse_completion(envelope)
            self.trace["phase"] = "RESPONSE_VALIDATED"
            return value, receipt
        except ProxyTransportError:
            raise
        except transport_api.TransportError as error:
            raise ProxyTransportError(error.reason, uncertain=False,
                receipt={"http_status": self.trace.get("http_status"),
                         "response_sha256": None}) from None
        except BaseException as error:
            self.trace["underlying_error_type"] = type(error).__name__
            number = getattr(error, "errno", None)
            self.trace["errno"] = number if type(number) is int else None
            raise ProxyTransportError("proxy_request_failed_no_retry",
                uncertain=request_attempted and not response_complete,
                receipt={"http_status": self.trace.get("http_status")}) from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
