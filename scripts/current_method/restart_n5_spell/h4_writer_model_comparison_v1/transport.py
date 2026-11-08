"""Explicit, single-attempt transport for the paired Writer experiment.

The existing generator's ``complete`` interface is retained. The historical
client is left untouched because it retries 429 and some TLS failures and
constructs a fixed payload; neither behavior is suitable for this experiment.
No credential is resolved and no connection is made while preparing payloads.
"""
from __future__ import annotations

import copy
import json
import os
import socket
import stat
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ..backends import BackendResponse
from ..live_backends import OpenAICompatibleChallengerClient


def _reject_inline_credentials(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"api_key", "authorization", "access_token", "bearer_token"}:
                raise ValueError("inline credentials are forbidden; use the configured environment variable or private file")
            _reject_inline_credentials(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_inline_credentials(item)


def build_payload(profile: dict[str, Any], messages: str | list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve exact messages and send only explicitly configured parameters."""
    _reject_inline_credentials(profile)
    model = profile.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model ID is required")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("a non-empty frozen messages list is required")
    if any(not isinstance(message, dict) or not isinstance(message.get("role"), str)
           or "content" not in message for message in messages):
        raise ValueError("each message requires its original role and content")
    parameters = profile.get("request_parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("request_parameters must be an object")
    if {"model", "messages"}.intersection(parameters):
        raise ValueError("request_parameters cannot override model or frozen messages")
    payload = {"model": model, "messages": copy.deepcopy(messages)}
    payload.update(copy.deepcopy({key: value for key, value in parameters.items() if value is not None}))
    return payload


def resolve_credential(profile: dict[str, Any], credential_file: str | Path | None = None) -> str:
    """Read only the explicitly named variable/file; never fall back to old keys."""
    _reject_inline_credentials(profile)
    file_name = credential_file if credential_file is not None else profile.get("credential_file")
    if file_name:
        descriptor = None
        try:
            descriptor = os.open(os.fspath(file_name), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
                raise ValueError("credential file must be an owned regular file with mode 0600")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = None
                credential = handle.read().strip()
        except OSError:
            raise ValueError("configured private credential file cannot be read") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    else:
        name = profile.get("api_key_env")
        credential = os.environ.get(name, "") if isinstance(name, str) else ""
    return _validate_credential(credential)


def _validate_credential(credential: str) -> str:
    if not isinstance(credential, str) or not credential or any(char.isspace() for char in credential):
        raise ValueError("configured credential is missing or invalid")
    return credential


def _redact(value: Any, credential: str) -> Any:
    if isinstance(value, str):
        return value.replace(credential, "[REDACTED]")
    if isinstance(value, dict):
        return {str(key).replace(credential, "[REDACTED]"):
                ("[REDACTED]" if str(key).lower() in {"authorization", "api_key", "access_token"}
                 else _redact(item, credential)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, credential) for item in value]
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class ComparisonTransportError(RuntimeError):
    """Sanitized failure with accounting retained; never retried by this client."""
    def __init__(self, status: str, metadata: dict[str, Any]):
        self.status = status
        self.metadata = metadata
        super().__init__("Writer request failed: " + status)


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class ProfileClient(OpenAICompatibleChallengerClient):
    """Common Writer adapter with exactly one HTTP attempt per ``complete``."""
    def __init__(self, profile: dict[str, Any], credential: str | None = None, *,
                 credential_file: str | Path | None = None,
                 on_http_attempt: Callable[[dict[str, Any]], None] | None = None,
                 before_send: Callable[[dict[str, Any]], None] | None = None):
        _reject_inline_credentials(profile)
        self.profile = copy.deepcopy(profile)
        # Compatibility with shared generator logging, without a key in repr.
        self.config = SimpleNamespace(model=profile.get("model"), base_url=profile.get("base_url"),
                                      temperature=profile.get("request_parameters", {}).get("temperature"))
        endpoint = profile.get("endpoint")
        parsed = urllib.parse.urlsplit(endpoint or "")
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("an explicit HTTPS endpoint without embedded credentials is required")
        timeout = profile.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("an explicit positive timeout_seconds is required")
        if before_send is not None and on_http_attempt is not None:
            raise ValueError("configure only one HTTP-attempt callback")
        self._on_http_attempt = on_http_attempt or before_send
        self._credential = (_validate_credential(credential) if credential is not None
                            else resolve_credential(profile, credential_file))
        self._opener = urllib.request.build_opener(_NoRedirect())

    def complete(self, *, prompt: str, max_tokens: int, json_object: bool) -> BackendResponse:
        payload = build_payload(self.profile, prompt)
        declared_limit = self.profile.get("output_limit")
        if declared_limit is None:
            declared_limit = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if declared_limit is None or declared_limit != max_tokens:
            raise ValueError("caller output budget must match the explicit profile output_limit")
        for field in ("max_tokens", "max_completion_tokens"):
            if field in payload and payload[field] != declared_limit:
                raise ValueError("output request parameter differs from the profile output_limit")
        # json_object indicates the shared parser's expectation. It does not
        # authorize an unsupported response_format parameter for either service.
        request = urllib.request.Request(self.profile["endpoint"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._credential, "Content-Type": "application/json"},
            method="POST")
        metadata: dict[str, Any] = {
            "requested_model": self.profile["model"], "actual_model": None,
            "returned_model": None, "finish_reason": None, "usage": None,
            "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
            "total_tokens": None, "transport_attempt_count": 1,
            "transport_retry_count": 0, "rate_limit_retry_count": 0,
            "http_attempt_count": 1, "timeout": False, "completion_status": "UNKNOWN",
        }
        if self._on_http_attempt is not None:
            self._on_http_attempt(copy.deepcopy(payload))
        try:
            with self._opener.open(request, timeout=self.profile["timeout_seconds"]) as response:
                raw_body = response.read().decode("utf-8")
                metadata["http_status"] = response.getcode()
        except urllib.error.HTTPError as error:
            metadata["http_status"] = error.code
            try:
                metadata["raw_response_text"] = _redact(error.read().decode("utf-8", errors="replace"), self._credential)
            except (OSError, ValueError):
                metadata["raw_response_text"] = None
            metadata["completion_status"] = "REJECTED" if error.code == 429 or 300 <= error.code < 400 else "UNKNOWN"
            status = "redirect_blocked" if 300 <= error.code < 400 else "http_error"
            raise ComparisonTransportError(status, metadata) from None
        except (TimeoutError, socket.timeout):
            metadata["timeout"] = True
            raise ComparisonTransportError("timeout_unknown", metadata) from None
        except urllib.error.URLError as error:
            metadata["timeout"] = isinstance(error.reason, (TimeoutError, socket.timeout))
            raise ComparisonTransportError("timeout_unknown" if metadata["timeout"] else "transport_unknown", metadata) from None
        except (OSError, UnicodeError):
            raise ComparisonTransportError("transport_or_decode_unknown", metadata) from None
        safe_text = _redact(raw_body, self._credential)
        metadata["raw_response_text"] = safe_text
        try:
            body = json.loads(safe_text)
        except (ValueError, TypeError):
            raise ComparisonTransportError("invalid_response_json", metadata) from None
        metadata["raw_response"] = _redact(body, self._credential)
        if not isinstance(body, dict):
            raise ComparisonTransportError("invalid_response_schema", metadata) from None
        actual_model = body.get("model")
        actual_model = actual_model if isinstance(actual_model, str) and actual_model else None
        metadata.update(actual_model=actual_model, returned_model=actual_model)
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else None
        metadata["usage"] = usage
        if usage is not None:
            details = usage.get("completion_tokens_details", usage.get("output_tokens_details"))
            details = details if isinstance(details, dict) else {}
            metadata.update(input_tokens=_token_count(usage.get("prompt_tokens", usage.get("input_tokens"))),
                            output_tokens=_token_count(usage.get("completion_tokens", usage.get("output_tokens"))),
                            reasoning_tokens=_token_count(details.get("reasoning_tokens", usage.get("reasoning_tokens"))),
                            total_tokens=_token_count(usage.get("total_tokens")))
        try:
            choice = body["choices"][0]
            metadata["finish_reason"] = choice.get("finish_reason")
            raw_text = choice["message"]["content"]
            if not isinstance(raw_text, str):
                raise TypeError("content must be a string")
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ComparisonTransportError("invalid_response_schema", metadata) from None
        metadata["completion_status"] = "RESPONSE_RECEIVED"
        return BackendResponse(raw_text, actual_model or "UNKNOWN", metadata)


ComparisonClient = ProfileClient
