"""Planner-only streaming transport recovery for v11.

This module changes no Planner semantics.  It freezes the complete transport
envelope and permits one byte-identical retry only for classified transport
timeouts.  HTTP responses and model/schema decisions are never retried here.
"""

from __future__ import annotations

import http.client
import json
import socket
import ssl
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from ..backends import BackendResponse
from ..core import canonical_json, sha256_bytes
from ..live_backends import (
    ChallengerAPIConfig, OpenAICompatibleChallengerClient,
)


PLANNER_TRANSPORT_RECOVERY_VERSION = "planner_transport_recovery_v11_v1"
PLANNER_CONNECT_TIMEOUT_SECONDS = 30
PLANNER_READ_TIMEOUT_SECONDS = 600
PLANNER_STREAMING_ENABLED = True
PLANNER_MAX_OUTPUT_TOKENS = 10_240
PLANNER_OUTPUT_LIMIT_LOGICAL_FIELD = "max_output_tokens"
# This project uses the OpenAI-compatible Chat Completions endpoint.  Its wire
# compatibility field is ``max_tokens``; the role-level frozen setting remains
# named ``max_output_tokens`` and is recorded alongside the wire field.
PLANNER_OUTPUT_LIMIT_WIRE_FIELD = "max_tokens"
PLANNER_TIMEOUT_RETRY_LIMIT = 1
PLANNER_TIMEOUT_RETRY_CLASSIFICATIONS = frozenset({
    "connect_timeout", "write_timeout", "read_timeout_before_first_event",
    "read_timeout_after_partial_stream",
})


@dataclass(frozen=True)
class PlannerTransportSettingsV11:
    connect_timeout_seconds: int = PLANNER_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: int = PLANNER_READ_TIMEOUT_SECONDS
    streaming: bool = PLANNER_STREAMING_ENABLED
    max_output_tokens: int = PLANNER_MAX_OUTPUT_TOKENS
    timeout_retry_limit: int = PLANNER_TIMEOUT_RETRY_LIMIT


@dataclass(frozen=True)
class FrozenPlannerTransportRequestV11:
    endpoint: str
    body_json: str
    prompt_sha256: str
    authoritative_input_sha256: str
    orchestration_seed: int
    model: str
    prompt_schema_contract_sha256: str
    request_fingerprint_sha256: str

    @property
    def body(self) -> dict[str, Any]:
        value = json.loads(self.body_json)
        if not isinstance(value, dict):
            raise AssertionError("planner_transport_body_not_object")
        return value


class PlannerTransportTimeoutV11(TimeoutError):
    def __init__(self, classification: str, *, partial_event_count: int = 0,
                 transport_attempt_count: int = 1):
        if classification not in PLANNER_TIMEOUT_RETRY_CLASSIFICATIONS:
            raise ValueError(f"planner_timeout_classification_invalid:{classification}")
        self.classification = classification
        self.partial_event_count = int(partial_event_count)
        self.transport_attempt_count = int(transport_attempt_count)
        super().__init__(
            f"{classification}:partial_event_count={self.partial_event_count}:"
            f"transport_attempt_count={self.transport_attempt_count}")


class PlannerTransportHTTPErrorV11(RuntimeError):
    def __init__(self, status: int, reason: str):
        self.status = int(status)
        self.reason = str(reason)
        super().__init__(f"planner_http_error:{self.status}:{self.reason}")


def build_frozen_planner_transport_request_v11(
    *, config: ChallengerAPIConfig, prompt: str,
    authoritative_input_sha256: str, orchestration_seed: int,
    prompt_schema_contract_sha256: str,
    settings: PlannerTransportSettingsV11 = PlannerTransportSettingsV11(),
) -> FrozenPlannerTransportRequestV11:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        PLANNER_OUTPUT_LIMIT_WIRE_FIELD: settings.max_output_tokens,
        "response_format": {"type": "json_object"},
        "stream": settings.streaming,
    }
    if settings.streaming:
        body["stream_options"] = {"include_usage": True}
    if config.enable_thinking is not None:
        body["enable_thinking"] = config.enable_thinking
    body_json = json.dumps(body, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    material = {
        "endpoint": config.base_url.rstrip("/") + "/chat/completions",
        "body_sha256": sha256_bytes(body_json.encode()),
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "authoritative_input_sha256": authoritative_input_sha256,
        "orchestration_seed": int(orchestration_seed),
        "model": config.model,
        "prompt_schema_contract_sha256": prompt_schema_contract_sha256,
    }
    return FrozenPlannerTransportRequestV11(
        endpoint=material["endpoint"], body_json=body_json,
        prompt_sha256=material["prompt_sha256"],
        authoritative_input_sha256=authoritative_input_sha256,
        orchestration_seed=int(orchestration_seed), model=config.model,
        prompt_schema_contract_sha256=prompt_schema_contract_sha256,
        request_fingerprint_sha256=sha256_bytes(canonical_json(material).encode()),
    )


def validate_frozen_planner_request_v11(
    request: FrozenPlannerTransportRequestV11,
) -> dict[str, Any]:
    body = request.body
    material = {
        "endpoint": request.endpoint,
        "body_sha256": sha256_bytes(request.body_json.encode()),
        "prompt_sha256": request.prompt_sha256,
        "authoritative_input_sha256": request.authoritative_input_sha256,
        "orchestration_seed": request.orchestration_seed,
        "model": request.model,
        "prompt_schema_contract_sha256": request.prompt_schema_contract_sha256,
    }
    checks = {
        "prompt_sha_matches_body": sha256_bytes(
            str((body.get("messages") or [{}])[0].get("content") or "").encode()
        ) == request.prompt_sha256,
        "model_matches_body": body.get("model") == request.model,
        # The seed is root-specific in production batches.  Immutability is
        # guaranteed by the request fingerprint below, not by requiring the
        # historical episode-61555 value for every root.
        "seed_frozen_in_envelope": request.orchestration_seed > 0,
        "streaming_enabled": body.get("stream") is True,
        "max_output_tokens_frozen": body.get(
            PLANNER_OUTPUT_LIMIT_WIRE_FIELD) == PLANNER_MAX_OUTPUT_TOKENS,
        "output_limit_wire_field_exact": (
            PLANNER_OUTPUT_LIMIT_WIRE_FIELD == "max_tokens" and
            "max_output_tokens" not in body),
        "input_sha_frozen": len(request.authoritative_input_sha256) == 64,
        "schema_sha_frozen": len(request.prompt_schema_contract_sha256) == 64,
        "fingerprint_valid": sha256_bytes(canonical_json(material).encode()) ==
            request.request_fingerprint_sha256,
    }
    return {"valid": all(checks.values()), "checks": checks,
            "request_fingerprint_sha256": request.request_fingerprint_sha256}


def execute_frozen_planner_transport_v11(
    *, request: FrozenPlannerTransportRequestV11,
    send_once: Callable[[FrozenPlannerTransportRequestV11, int], BackendResponse],
    timeout_retry_limit: int = PLANNER_TIMEOUT_RETRY_LIMIT,
) -> BackendResponse:
    if timeout_retry_limit != PLANNER_TIMEOUT_RETRY_LIMIT:
        raise ValueError("planner_transport_retry_limit_not_frozen")
    validation = validate_frozen_planner_request_v11(request)
    if not validation["valid"]:
        raise RuntimeError("planner_transport_request_freeze_invalid")
    initial_fingerprint = request.request_fingerprint_sha256
    attempts: list[dict[str, Any]] = []
    for transport_attempt in range(1, timeout_retry_limit + 2):
        current = validate_frozen_planner_request_v11(request)
        if (not current["valid"] or
                request.request_fingerprint_sha256 != initial_fingerprint):
            raise RuntimeError("planner_transport_request_drift_before_retry")
        try:
            response = send_once(request, transport_attempt)
        except PlannerTransportTimeoutV11 as error:
            attempts.append({
                "transport_attempt": transport_attempt,
                "classification": error.classification,
                "partial_event_count": error.partial_event_count,
                "request_fingerprint_sha256": initial_fingerprint,
                "retry_allowed": transport_attempt <= timeout_retry_limit,
            })
            if transport_attempt <= timeout_retry_limit:
                continue
            raise PlannerTransportTimeoutV11(
                error.classification,
                partial_event_count=error.partial_event_count,
                transport_attempt_count=transport_attempt) from error
        metadata = dict(response.metadata or {})
        attempts.append({
            "transport_attempt": transport_attempt, "classification": "completed",
            "request_fingerprint_sha256": initial_fingerprint,
            "retry_allowed": False,
        })
        metadata.update({
            "planner_transport_recovery_version": PLANNER_TRANSPORT_RECOVERY_VERSION,
            "transport_attempt_count": transport_attempt,
            "transport_retry_count": transport_attempt - 1,
            "transport_attempts": attempts,
            "request_fingerprint_sha256": initial_fingerprint,
            "connect_timeout_seconds": PLANNER_CONNECT_TIMEOUT_SECONDS,
            "read_timeout_seconds": PLANNER_READ_TIMEOUT_SECONDS,
            "streaming": True,
            "planner_output_limit_logical_field": (
                PLANNER_OUTPUT_LIMIT_LOGICAL_FIELD),
            "planner_output_limit_wire_field": PLANNER_OUTPUT_LIMIT_WIRE_FIELD,
            "planner_max_output_tokens": PLANNER_MAX_OUTPUT_TOKENS,
        })
        return BackendResponse(response.raw_text, response.model_identifier, metadata)
    raise AssertionError("planner_transport_retry_loop_unreachable")


class PlannerTransportRecoveryClientV11:
    """Use the recovered path only for Planner; other roles retain the old client."""

    def __init__(
        self, *, config: ChallengerAPIConfig, authoritative_input_sha256: str,
        orchestration_seed: int, prompt_schema_contract_sha256: str,
        settings: PlannerTransportSettingsV11 = PlannerTransportSettingsV11(),
    ):
        self.config = config
        self.authoritative_input_sha256 = authoritative_input_sha256
        self.orchestration_seed = int(orchestration_seed)
        self.prompt_schema_contract_sha256 = prompt_schema_contract_sha256
        self.settings = settings
        self._fallback = OpenAICompatibleChallengerClient(config)

    def complete(self, *, prompt: str, max_tokens: int,
                 json_object: bool) -> BackendResponse:
        return self._fallback.complete(
            prompt=prompt, max_tokens=max_tokens, json_object=json_object)

    def complete_planner(self, *, prompt: str, max_tokens: int,
                         json_object: bool) -> BackendResponse:
        if not json_object:
            raise ValueError("planner_transport_json_object_required")
        if int(max_tokens) != self.settings.max_output_tokens:
            raise ValueError(
                f"planner_max_output_tokens_not_frozen:{max_tokens}!={self.settings.max_output_tokens}")
        request = build_frozen_planner_transport_request_v11(
            config=self.config, prompt=prompt,
            authoritative_input_sha256=self.authoritative_input_sha256,
            orchestration_seed=self.orchestration_seed,
            prompt_schema_contract_sha256=self.prompt_schema_contract_sha256,
            settings=self.settings)
        return execute_frozen_planner_transport_v11(
            request=request, send_once=self._send_once,
            timeout_retry_limit=self.settings.timeout_retry_limit)

    def _send_once(
        self, request: FrozenPlannerTransportRequestV11,
        transport_attempt: int,
    ) -> BackendResponse:
        del transport_attempt
        parsed = urlsplit(request.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("planner_transport_endpoint_invalid")
        connection_class = (http.client.HTTPSConnection if parsed.scheme == "https"
                            else http.client.HTTPConnection)
        kwargs: dict[str, Any] = {
            "host": parsed.hostname, "port": parsed.port,
            "timeout": self.settings.connect_timeout_seconds,
        }
        if parsed.scheme == "https":
            kwargs["context"] = ssl.create_default_context()
        connection = connection_class(**kwargs)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json", "Accept": "text/event-stream",
        }
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        event_count = 0
        try:
            try:
                connection.connect()
            except (TimeoutError, socket.timeout) as error:
                raise PlannerTransportTimeoutV11("connect_timeout") from error
            try:
                connection.request("POST", path, body=request.body_json.encode(),
                                   headers=headers)
            except (TimeoutError, socket.timeout) as error:
                raise PlannerTransportTimeoutV11("write_timeout") from error
            if connection.sock is not None:
                connection.sock.settimeout(self.settings.read_timeout_seconds)
            try:
                response = connection.getresponse()
            except (TimeoutError, socket.timeout) as error:
                raise PlannerTransportTimeoutV11(
                    "read_timeout_before_first_event") from error
            if not 200 <= response.status < 300:
                raise PlannerTransportHTTPErrorV11(response.status, response.reason)
            content_parts: list[str] = []
            usage: Mapping[str, Any] | None = None
            finish_reason: str | None = None
            model = self.config.model
            while True:
                try:
                    raw_line = response.readline()
                except (TimeoutError, socket.timeout) as error:
                    classification = ("read_timeout_after_partial_stream" if event_count
                                      else "read_timeout_before_first_event")
                    raise PlannerTransportTimeoutV11(
                        classification, partial_event_count=event_count) from error
                if not raw_line:
                    break
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise RuntimeError("planner_stream_event_json_invalid") from error
                event_count += 1
                if event.get("model"):
                    model = str(event["model"])
                if isinstance(event.get("usage"), Mapping):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content") is not None:
                        content_parts.append(str(delta["content"]))
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
            if not event_count or not content_parts:
                raise RuntimeError("planner_stream_completed_without_content")
            return BackendResponse(
                "".join(content_parts), model,
                {"finish_reason": finish_reason, "usage": dict(usage or {}),
                 "stream_event_count": event_count,
                 "streaming": True})
        finally:
            connection.close()


def planner_transport_configuration_v11() -> dict[str, Any]:
    settings = PlannerTransportSettingsV11()
    return {
        "transport_recovery_version": PLANNER_TRANSPORT_RECOVERY_VERSION,
        "connect_timeout_seconds": settings.connect_timeout_seconds,
        "read_timeout_seconds": settings.read_timeout_seconds,
        "streaming": settings.streaming,
        "max_output_tokens": settings.max_output_tokens,
        "max_output_tokens_logical_field": PLANNER_OUTPUT_LIMIT_LOGICAL_FIELD,
        "max_output_tokens_wire_field": PLANNER_OUTPUT_LIMIT_WIRE_FIELD,
        "timeout_retry_limit": settings.timeout_retry_limit,
        "retryable_timeout_classifications": sorted(
            PLANNER_TIMEOUT_RETRY_CLASSIFICATIONS),
        "non_retryable": ["http_error", "parse_invalid", "schema_invalid",
                          "reference_invalid", "legal_model_decision"],
    }


__all__ = [
    "FrozenPlannerTransportRequestV11", "PlannerTransportHTTPErrorV11",
    "PlannerTransportRecoveryClientV11", "PlannerTransportSettingsV11",
    "PlannerTransportTimeoutV11", "build_frozen_planner_transport_request_v11",
    "execute_frozen_planner_transport_v11", "planner_transport_configuration_v11",
    "validate_frozen_planner_request_v11",
]
