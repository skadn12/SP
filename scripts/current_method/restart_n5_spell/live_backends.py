from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import BackendResponse, QuestionerRequest, RealizerRequest


RATE_LIMIT_RETRY_LIMIT = 2
RATE_LIMIT_DEFAULT_BACKOFF_SECONDS = 30.0
RATE_LIMIT_MAX_BACKOFF_SECONDS = 120.0


@dataclass(frozen=True)
class ChallengerAPIConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 180
    realizer_max_tokens: int = 4096
    questioner_max_tokens: int = 4096
    temperature: float = 0.25
    enable_thinking: bool | None = None


def load_challenger_config(path: Path) -> ChallengerAPIConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    api = payload.get("api") if isinstance(payload.get("api"), dict) else {}
    missing = [key for key in ("base_url", "api_key", "model") if not api.get(key)]
    if missing:
        raise RuntimeError(f"missing Challenger API config fields: {missing}")
    return ChallengerAPIConfig(base_url=str(api["base_url"]).rstrip("/"),
        api_key=str(api["api_key"]), model=str(api["model"]),
        timeout_seconds=int(api.get("timeout") or 180),
        realizer_max_tokens=int(api.get("realizer_max_tokens") or api.get("max_tokens") or 4096),
        questioner_max_tokens=int(api.get("questioner_max_tokens") or api.get("max_tokens") or 4096),
        temperature=float(api.get("temperature") if api.get("temperature") is not None else 0.25),
        enable_thinking=(bool(api["enable_thinking"])
                         if "enable_thinking" in api else None))


class OpenAICompatibleChallengerClient:
    """One HTTP request per call. Retry ownership remains with the orchestrator/ledger."""

    def __init__(self, config: ChallengerAPIConfig):
        self.config = config

    def complete(self, *, prompt: str, max_tokens: int, json_object: bool) -> BackendResponse:
        payload: dict[str, Any] = {"model": self.config.model,
            "messages": [{"role":"user", "content":prompt}],
            "temperature":self.config.temperature, "max_tokens":max_tokens}
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
        if json_object:
            payload["response_format"] = {"type":"json_object"}
        request = urllib.request.Request(self.config.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization":f"Bearer {self.config.api_key}", "Content-Type":"application/json"},
            method="POST")
        transport_attempts = 0
        rate_limit_retries = 0
        while True:
            transport_attempts += 1
            try:
                with urllib.request.urlopen(
                        request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                # HTTP 429 is an explicit pre-generation rejection, so retrying
                # the byte-identical frozen request does not duplicate a paid
                # completion. Other HTTP failures remain non-retryable.
                if (error.code != 429 or
                        rate_limit_retries >= RATE_LIMIT_RETRY_LIMIT):
                    raise
                retry_after = (error.headers.get("Retry-After")
                               if error.headers is not None else None)
                try:
                    delay = float(retry_after) if retry_after is not None else (
                        RATE_LIMIT_DEFAULT_BACKOFF_SECONDS *
                        (2 ** rate_limit_retries))
                except (TypeError, ValueError):
                    delay = (RATE_LIMIT_DEFAULT_BACKOFF_SECONDS *
                             (2 ** rate_limit_retries))
                rate_limit_retries += 1
                time.sleep(max(0.0, min(delay, RATE_LIMIT_MAX_BACKOFF_SECONDS)))
                continue
            except urllib.error.URLError as error:
                # A TLS EOF raised while establishing the socket occurs before an
                # HTTP request/paid generation exists.  One reconnect is safe; all
                # other transport failures remain ambiguous and propagate.
                if (transport_attempts == 1 and isinstance(error.reason, ssl.SSLError)
                        and "EOF occurred" in str(error.reason)):
                    continue
                raise
        choice = body["choices"][0]
        raw = str(choice["message"]["content"])
        metadata = {"finish_reason":choice.get("finish_reason"),
                    "usage":body.get("usage") if isinstance(body.get("usage"), dict) else None,
                    "transport_attempt_count":transport_attempts,
                    "transport_retry_count":transport_attempts - 1,
                    "rate_limit_retry_count":rate_limit_retries}
        return BackendResponse(raw, self.config.model, metadata)


class LiveRealizerBackend:
    def __init__(self, client: OpenAICompatibleChallengerClient):
        self.client = client

    def generate_session(self, request: RealizerRequest) -> BackendResponse:
        return self.client.complete(prompt=request.prompt,
            max_tokens=self.client.config.realizer_max_tokens, json_object=True)


class LiveQuestionerBackend:
    def __init__(self, client: OpenAICompatibleChallengerClient):
        self.client = client

    def generate_batch(self, request: QuestionerRequest) -> BackendResponse:
        return self.client.complete(prompt=request.prompt,
            max_tokens=self.client.config.questioner_max_tokens, json_object=False)


class LiveSemanticRepairBackend:
    """Single-object Challenger task; all parsing and admission remain in orchestration."""
    def __init__(self, client: OpenAICompatibleChallengerClient,
                 max_tokens: int | None = None):
        self.client = client
        self.max_tokens = (client.config.questioner_max_tokens if max_tokens is None
                           else int(max_tokens))

    def generate_batch(self, request: QuestionerRequest) -> BackendResponse:
        return self.client.complete(prompt=request.prompt,
            max_tokens=self.max_tokens, json_object=True)
