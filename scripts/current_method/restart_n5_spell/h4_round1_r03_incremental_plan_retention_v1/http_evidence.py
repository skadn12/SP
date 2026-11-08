"""Single-attempt HTTP with response evidence captured before parsing.

The optional sink receives distinct response_received/completed/failure snapshots.
It must persist each snapshot locally; persistence failure never permits resending.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request


EVIDENCE_VERSION = "incremental_retention_http_evidence_v2"
_SAFE_HEADERS = frozenset({
    "content-type", "content-length", "date", "retry-after", "request-id",
    "x-request-id", "x-trace-id", "traceparent", "cf-ray", "server-timing",
    "x-siliconcloud-trace-id",
})
_SENSITIVE_HEADER = re.compile(r"authorization|cookie|api[-_]?key|token|secret", re.I)
_JSON_SECRET = re.compile(
    r'(?P<prefix>"(?:authorization|proxy-authorization|api[-_]?key|access[-_]?token|'
    r'refresh[-_]?token|secret|password|credential)"\s*:\s*)'
    r'(?P<value>"(?:[^"\\]|\\.)*")', re.I)
_BEARER = re.compile(r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.I)
_ASSIGNED_SECRET = re.compile(
    r"\b((?:api[-_]?key|access[-_]?token|refresh[-_]?token|password|secret)\s*[=:]\s*)"
    r"([^\s&;,}\]<>]+)", re.I)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _response_format(request):
    if "response_format" not in request:
        return None
    value = request["response_format"]
    if request["role"] != "adherence":
        raise ValueError("response_format_requires_adherence_request")
    if (not isinstance(value, dict) or set(value) != {"type", "json_schema"}
            or value["type"] != "json_schema"):
        raise ValueError("invalid_response_format_json_schema")
    schema = value["json_schema"]
    if (not isinstance(schema, dict) or not {"name", "schema"} <= set(schema)
            or set(schema) - {"name", "schema", "strict", "description"}
            or not isinstance(schema["name"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", schema["name"])
            or not isinstance(schema["schema"], dict)
            or ("strict" in schema and not isinstance(schema["strict"], bool))
            or ("description" in schema and not isinstance(schema["description"], str))):
        raise ValueError("invalid_response_format_json_schema")
    try:
        snapshot = deepcopy(value)
        serialized = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False)
        if json.loads(serialized) != snapshot:
            raise ValueError("non_json_schema_value")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid_response_format_json_schema") from exc
    return snapshot


class _Redactor:
    def __init__(self, secret):
        self.secret_forms = set()
        if secret:
            self.secret_forms.update((secret, secret.encode().decode("latin-1"),
                json.dumps(secret)[1:-1], urllib.parse.quote(secret, safe="")))
        self.reasons = set()

    def text(self, value):
        text = str(value)
        for secret in sorted(self.secret_forms, key=len, reverse=True):
            if secret in text:
                text = text.replace(secret, "[REDACTED]")
                self.reasons.add("configured_credential")
        text, count = _JSON_SECRET.subn(lambda m: m["prefix"] + '"[REDACTED]"', text)
        if count:
            self.reasons.add("sensitive_json_field")
        text, count = _BEARER.subn(lambda m: m[1] + "[REDACTED]", text)
        if count:
            self.reasons.add("authorization_value")
        text, count = _ASSIGNED_SECRET.subn(lambda m: m[1] + "[REDACTED]", text)
        if count:
            self.reasons.add("sensitive_assignment")
        return text

    def bytes(self, body):
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            # Latin-1 is reversible: sanitization cannot discard non-UTF8 bytes.
            return self.text(body.decode("latin-1")).encode("latin-1")
        safe = self.text(text)
        # Credential scrubbing may inspect JSON, but response acceptance/schema
        # parsing only happens after evidence persistence. Clean bytes stay exact.
        try:
            parsed = json.loads(safe)
        except ValueError:
            return safe.encode()
        cleaned = self.value(parsed)
        if cleaned != parsed:
            return json.dumps(cleaned, ensure_ascii=False).encode()
        return safe.encode()

    def value(self, value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if re.fullmatch(r"authorization|proxy-authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|secret|password|credential", key, re.I):
                    result[key] = "[REDACTED]"
                    self.reasons.add("sensitive_json_field")
                else:
                    result[key] = self.value(item)
            return result
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            safe = self.text(value)
            try:
                nested = json.loads(safe)
            except ValueError:
                return safe
            if isinstance(nested, (dict, list)):
                cleaned = self.value(nested)
                if cleaned != nested:
                    return json.dumps(cleaned, ensure_ascii=False)
            return safe
        return value


def request_once(transport, request):
    """Use transport.config/tokenizer; return content, or raise without retry.

    last_response_evidence is always sanitized. last_response_envelope becomes
    available for any JSON response, including HTTP or business-level failures.
    """
    transport.last_response_envelope = None
    transport.last_response_evidence = None
    response_format = _response_format(request)
    config = transport.config
    credential = os.environ[config["api_key_env"]]
    redactor = _Redactor(credential)
    started = time.monotonic()
    evidence = {
        "evidence_version": EVIDENCE_VERSION,
        "request_identity": {
            "purpose": request["role"], "call_id": request["call_id"],
            "prompt_version": request["prompt_version"], "model": config["model"],
            "messages_sha256": hashlib.sha256(json.dumps(request["messages"], ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
        "request_started_at": _now(), "response_received_at": None,
        "http_status": None, "response_unavailable": True,
        "response_body_unavailable": True, "response_headers": {},
        "exception_class": None, "exception_message": None,
        "request_attempt_count": 1, "retry_count": 0, "fallback_count": 0,
    }
    if "identity" in request:
        evidence["request_identity"]["execution_identity"] = redactor.value(request["identity"])
    if response_format is not None:
        evidence["request_identity"]["response_format_sha256"] = hashlib.sha256(
            json.dumps(response_format, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def publish(event):
        evidence.update(evidence_event=event, recorded_at=_now(),
            elapsed_seconds=round(time.monotonic() - started, 6),
            redaction_applied=bool(redactor.reasons), redaction_reasons=sorted(redactor.reasons))
        snapshot = redactor.value(deepcopy(evidence))
        # Update flags after recursive sanitization as identity/header values can
        # themselves introduce an additional redaction reason.
        snapshot.update(redaction_applied=bool(redactor.reasons), redaction_reasons=sorted(redactor.reasons))
        transport.last_response_evidence = snapshot
        sink = getattr(transport, "response_evidence_sink", None)
        if sink is not None:
            sink(deepcopy(snapshot))

    def capture_body(body, *, complete):
        safe_body = redactor.bytes(body)
        evidence.update(response_body_unavailable=False, response_body_byte_count=len(body),
            saved_body_sha256=hashlib.sha256(safe_body).hexdigest(),
            saved_body_byte_count=len(safe_body), body_redacted=safe_body != body,
            response_body_complete=complete)
        try:
            evidence["response_body"] = safe_body.decode("utf-8")
            evidence["response_body_encoding"] = "utf-8"
        except UnicodeDecodeError:
            evidence["response_body_base64"] = base64.b64encode(safe_body).decode("ascii")
            evidence["response_body_encoding"] = "base64_non_utf8"
        return safe_body

    def capture_response(response):
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        evidence.update(http_status=status, response_unavailable=False, response_received_at=_now())
        headers = dict(response.headers.items()) if response.headers else {}
        evidence["response_headers"] = {key.lower(): redactor.text(value) for key, value in headers.items()
            if key.lower() in _SAFE_HEADERS}
        sensitive = sorted(key.lower() for key in headers if _SENSITIVE_HEADER.search(key))
        evidence["omitted_response_header_names"] = sorted(key.lower() for key in headers
            if key.lower() not in _SAFE_HEADERS)
        if sensitive:
            redactor.reasons.add("sensitive_response_header_omitted")
        try:
            body = response.read()
        except Exception as exc:
            partial = getattr(exc, "partial", None)
            if isinstance(partial, bytes):
                capture_body(partial, complete=False)
            raise
        return capture_body(body, complete=True)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            # Returning None leaves a single HTTPError with the original body.
            return None

    payload = {"model": config["model"], "messages": request["messages"],
        "temperature": config["temperature"], "max_tokens": request["max_tokens"],
        "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    if response_format is not None:
        payload["response_format"] = response_format
    http_request = urllib.request.Request(config["endpoint"],
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + credential}, method="POST")
    http_error = None
    try:
        try:
            response = urllib.request.build_opener(NoRedirect).open(http_request, timeout=config["timeout_seconds"])
        except urllib.error.HTTPError as exc:
            http_error = exc
            response = exc
        with response:
            body = capture_response(response)
    except Exception as exc:
        evidence.update(exception_class=type(exc).__name__, exception_message=redactor.text(exc),
            failure_classification="HTTP_RESPONSE_READ_FAILURE" if not evidence["response_unavailable"] else "RESPONSE_UNAVAILABLE")
        publish("failure")
        raise

    publish("response_received")
    try:
        try:
            envelope = json.loads(body)
            transport.last_response_envelope = redactor.value(envelope)
        except (ValueError, UnicodeError):
            if http_error is not None:
                raise http_error
            raise
        if http_error is not None:
            raise http_error
        if not 200 <= evidence["http_status"] < 300:
            raise ValueError("non_success_status_no_retry")
        if evidence["body_redacted"]:
            # Safe evidence is not the original model output. Never let a
            # credential scrub silently produce dialogue eligible for promotion.
            raise ValueError("credential_redaction_changed_success_response")
        if not isinstance(envelope, dict) or len(envelope.get("choices", [])) != 1:
            raise ValueError("incomplete_or_ambiguous_response_no_retry")
        choice = envelope["choices"][0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError("incomplete_or_ambiguous_response_no_retry")
        content = choice["message"]["content"]
        if not isinstance(content, str) or len(transport.tokenizer.encode(content, add_special_tokens=False)) > request["max_tokens"]:
            raise ValueError("response_output_budget_exceeded")
    except Exception as exc:
        classification = "HTTP_NON_SUCCESS" if http_error is not None else "RESPONSE_PARSE_OR_BUSINESS_FAILURE"
        if str(exc) == "credential_redaction_changed_success_response":
            classification = "CREDENTIAL_REDACTION_CHANGED_SUCCESS_RESPONSE"
        evidence.update(exception_class=type(exc).__name__, exception_message=redactor.text(exc),
            failure_classification=classification)
        publish("failure")
        raise
    publish("completed")
    return content
