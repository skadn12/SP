"""Role-specific streaming transport for the v11 Round Validator.

Unlike Planner transport recovery, this transport performs exactly one HTTP
attempt.  A timeout is ambiguous after a paid generation may have started, so
the round orchestrator—not the transport—owns the frozen two-request contract
budget.
"""

from __future__ import annotations

import http.client
import json
import socket
import ssl
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..backends import BackendResponse
from ..core import canonical_json, sha256_bytes
from ..live_backends import ChallengerAPIConfig, OpenAICompatibleChallengerClient
from .planner_transport_v11 import (
    PlannerTransportRecoveryClientV11, PlannerTransportSettingsV11,
)


ROUND_VALIDATOR_TRANSPORT_VERSION_V11 = (
    "round_validator_streaming_transport_v11_v2_output_12288"
)
ROUND_VALIDATOR_CONNECT_TIMEOUT_SECONDS = 30
ROUND_VALIDATOR_READ_TIMEOUT_SECONDS = 600
ROUND_VALIDATOR_STREAMING_ENABLED = True
ROUND_VALIDATOR_MAX_OUTPUT_TOKENS = 12_288
ROUND_VALIDATOR_OUTPUT_LIMIT_WIRE_FIELD = "max_tokens"
ROUND_VALIDATOR_TRANSPORT_RETRY_LIMIT = 0
ROUND_VALIDATOR_TIMEOUT_CLASSIFICATIONS = frozenset({
    "connect_timeout", "write_timeout", "read_timeout_before_first_event",
    "read_timeout_after_partial_stream",
})


@dataclass(frozen=True)
class RoundValidatorTransportSettingsV11:
    connect_timeout_seconds: int = ROUND_VALIDATOR_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: int = ROUND_VALIDATOR_READ_TIMEOUT_SECONDS
    streaming: bool = ROUND_VALIDATOR_STREAMING_ENABLED
    max_output_tokens: int = ROUND_VALIDATOR_MAX_OUTPUT_TOKENS
    timeout_retry_limit: int = ROUND_VALIDATOR_TRANSPORT_RETRY_LIMIT


@dataclass(frozen=True)
class FrozenRoundValidatorTransportRequestV11:
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
            raise AssertionError("round_validator_transport_body_not_object")
        return value


class RoundValidatorTransportTimeoutV11(TimeoutError):
    def __init__(self, classification: str, *, partial_event_count: int = 0):
        if classification not in ROUND_VALIDATOR_TIMEOUT_CLASSIFICATIONS:
            raise ValueError(
                f"round_validator_timeout_classification_invalid:{classification}")
        self.classification = classification
        self.partial_event_count = int(partial_event_count)
        self.transport_attempt_count = 1
        super().__init__(
            f"{classification}:partial_event_count={self.partial_event_count}:"
            "transport_attempt_count=1")


class RoundValidatorTransportHTTPErrorV11(RuntimeError):
    def __init__(self, status: int, reason: str):
        self.status = int(status)
        self.reason = str(reason)
        super().__init__(f"round_validator_http_error:{self.status}:{self.reason}")


def build_frozen_round_validator_transport_request_v11(
    *, config: ChallengerAPIConfig, prompt: str,
    authoritative_input_sha256: str, orchestration_seed: int,
    prompt_schema_contract_sha256: str,
    settings: RoundValidatorTransportSettingsV11 = (
        RoundValidatorTransportSettingsV11()),
) -> FrozenRoundValidatorTransportRequestV11:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        ROUND_VALIDATOR_OUTPUT_LIMIT_WIRE_FIELD: settings.max_output_tokens,
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
    return FrozenRoundValidatorTransportRequestV11(
        endpoint=material["endpoint"], body_json=body_json,
        prompt_sha256=material["prompt_sha256"],
        authoritative_input_sha256=authoritative_input_sha256,
        orchestration_seed=int(orchestration_seed), model=config.model,
        prompt_schema_contract_sha256=prompt_schema_contract_sha256,
        request_fingerprint_sha256=sha256_bytes(canonical_json(material).encode()),
    )


def validate_frozen_round_validator_request_v11(
    request: FrozenRoundValidatorTransportRequestV11,
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
        # Batch roots have deterministic per-root seeds.  The fingerprint
        # freezes the exact value for this request.
        "seed_frozen_in_envelope": request.orchestration_seed > 0,
        "streaming_enabled": body.get("stream") is True,
        "max_output_tokens_frozen": body.get(
            ROUND_VALIDATOR_OUTPUT_LIMIT_WIRE_FIELD
        ) == ROUND_VALIDATOR_MAX_OUTPUT_TOKENS,
        "input_sha_frozen": len(request.authoritative_input_sha256) == 64,
        "schema_sha_frozen": len(request.prompt_schema_contract_sha256) == 64,
        "fingerprint_valid": sha256_bytes(canonical_json(material).encode()) ==
            request.request_fingerprint_sha256,
        "no_transport_retry": ROUND_VALIDATOR_TRANSPORT_RETRY_LIMIT == 0,
    }
    return {"valid": all(checks.values()), "checks": checks,
            "request_fingerprint_sha256": request.request_fingerprint_sha256}


class RoundValidatorStreamingClientV11:
    def __init__(
        self, *, config: ChallengerAPIConfig, authoritative_input_sha256: str,
        orchestration_seed: int, prompt_schema_contract_sha256: str,
        settings: RoundValidatorTransportSettingsV11 = (
            RoundValidatorTransportSettingsV11()),
    ):
        self.config = config
        self.authoritative_input_sha256 = authoritative_input_sha256
        self.orchestration_seed = int(orchestration_seed)
        self.prompt_schema_contract_sha256 = prompt_schema_contract_sha256
        self.settings = settings

    def complete_round_validator(
        self, *, prompt: str, max_tokens: int, json_object: bool,
    ) -> BackendResponse:
        if not json_object:
            raise ValueError("round_validator_transport_json_object_required")
        if int(max_tokens) != self.settings.max_output_tokens:
            raise ValueError(
                "round_validator_max_output_tokens_not_frozen:"
                f"{max_tokens}!={self.settings.max_output_tokens}")
        request = build_frozen_round_validator_transport_request_v11(
            config=self.config, prompt=prompt,
            authoritative_input_sha256=self.authoritative_input_sha256,
            orchestration_seed=self.orchestration_seed,
            prompt_schema_contract_sha256=self.prompt_schema_contract_sha256,
            settings=self.settings)
        validation = validate_frozen_round_validator_request_v11(request)
        if not validation["valid"]:
            raise RuntimeError("round_validator_transport_request_freeze_invalid")
        return self._send_once(request)

    def _send_once(
        self, request: FrozenRoundValidatorTransportRequestV11,
    ) -> BackendResponse:
        parsed = urlsplit(request.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("round_validator_transport_endpoint_invalid")
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
                raise RoundValidatorTransportTimeoutV11("connect_timeout") from error
            try:
                connection.request("POST", path, body=request.body_json.encode(),
                                   headers=headers)
            except (TimeoutError, socket.timeout) as error:
                raise RoundValidatorTransportTimeoutV11("write_timeout") from error
            if connection.sock is not None:
                connection.sock.settimeout(self.settings.read_timeout_seconds)
            try:
                response = connection.getresponse()
            except (TimeoutError, socket.timeout) as error:
                raise RoundValidatorTransportTimeoutV11(
                    "read_timeout_before_first_event") from error
            if not 200 <= response.status < 300:
                raise RoundValidatorTransportHTTPErrorV11(
                    response.status, response.reason)
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
                    raise RoundValidatorTransportTimeoutV11(
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
                    raise RuntimeError(
                        "round_validator_stream_event_json_invalid") from error
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
            metadata = {
                "finish_reason": finish_reason, "usage": usage,
                "transport_attempt_count": 1, "transport_retry_count": 0,
                "round_validator_transport_version": (
                    ROUND_VALIDATOR_TRANSPORT_VERSION_V11),
                "request_fingerprint_sha256": request.request_fingerprint_sha256,
                "connect_timeout_seconds": self.settings.connect_timeout_seconds,
                "read_timeout_seconds": self.settings.read_timeout_seconds,
                "streaming": self.settings.streaming,
                "round_validator_max_output_tokens": self.settings.max_output_tokens,
                "stream_event_count": event_count,
            }
            return BackendResponse("".join(content_parts), model, metadata)
        finally:
            connection.close()


class V11LiveRoleTransportClient:
    """Planner recovery + unchanged Realizer + long-read Round Validator."""

    def __init__(
        self, *, config: ChallengerAPIConfig, authoritative_input_sha256: str,
        orchestration_seed: int, prompt_schema_contract_sha256: str,
        planner_settings: PlannerTransportSettingsV11 = PlannerTransportSettingsV11(),
        round_validator_settings: RoundValidatorTransportSettingsV11 = (
            RoundValidatorTransportSettingsV11()),
    ):
        self.config = config
        self._fallback = OpenAICompatibleChallengerClient(config)
        self._planner = PlannerTransportRecoveryClientV11(
            config=config, authoritative_input_sha256=authoritative_input_sha256,
            orchestration_seed=orchestration_seed,
            prompt_schema_contract_sha256=prompt_schema_contract_sha256,
            settings=planner_settings)
        self._round_validator = RoundValidatorStreamingClientV11(
            config=config, authoritative_input_sha256=authoritative_input_sha256,
            orchestration_seed=orchestration_seed,
            prompt_schema_contract_sha256=prompt_schema_contract_sha256,
            settings=round_validator_settings)

    def complete(self, *, prompt: str, max_tokens: int,
                 json_object: bool) -> BackendResponse:
        return self._fallback.complete(
            prompt=prompt, max_tokens=max_tokens, json_object=json_object)

    def complete_planner(self, *, prompt: str, max_tokens: int,
                         json_object: bool) -> BackendResponse:
        return self._planner.complete_planner(
            prompt=prompt, max_tokens=max_tokens, json_object=json_object)

    def complete_round_validator(
        self, *, prompt: str, max_tokens: int, json_object: bool,
    ) -> BackendResponse:
        return self._round_validator.complete_round_validator(
            prompt=prompt, max_tokens=max_tokens, json_object=json_object)


def round_validator_transport_configuration_v11() -> dict[str, Any]:
    return {
        "transport_version": ROUND_VALIDATOR_TRANSPORT_VERSION_V11,
        "connect_timeout_seconds": ROUND_VALIDATOR_CONNECT_TIMEOUT_SECONDS,
        "read_timeout_seconds": ROUND_VALIDATOR_READ_TIMEOUT_SECONDS,
        "streaming": ROUND_VALIDATOR_STREAMING_ENABLED,
        "max_output_tokens": ROUND_VALIDATOR_MAX_OUTPUT_TOKENS,
        "transport_retry_limit": ROUND_VALIDATOR_TRANSPORT_RETRY_LIMIT,
    }


__all__ = [
    "ROUND_VALIDATOR_CONNECT_TIMEOUT_SECONDS",
    "ROUND_VALIDATOR_MAX_OUTPUT_TOKENS",
    "ROUND_VALIDATOR_READ_TIMEOUT_SECONDS",
    "ROUND_VALIDATOR_STREAMING_ENABLED",
    "ROUND_VALIDATOR_TRANSPORT_RETRY_LIMIT",
    "RoundValidatorStreamingClientV11",
    "RoundValidatorTransportSettingsV11",
    "RoundValidatorTransportTimeoutV11",
    "V11LiveRoleTransportClient",
    "build_frozen_round_validator_transport_request_v11",
    "round_validator_transport_configuration_v11",
    "validate_frozen_round_validator_request_v11",
]
