"""One-physical-request JSON runtime for the three isolated live roles."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from ..backends import BackendResponse
from ..core import atomic_json, atomic_text, sha256_bytes
from ..ledger import RequestLedger
from ..live_backends import ChallengerAPIConfig


ROLE_METADATA = {
    "state_resolver_v1": ("state_resolver", "state_resolver_v1"),
    "state_resolved_planner_v1": ("planner", "state_resolved_planner_v1"),
    "state_resolved_round_realizer_v1": (
        "round_realizer", "state_resolved_round_realizer_v1"),
}


class NoRetryOpenAIClient:
    """Issue exactly one HTTP request; transport and HTTP failures propagate."""

    def __init__(self, config: ChallengerAPIConfig):
        self.config = config

    def complete(self, *, prompt: str, max_tokens: int) -> BackendResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        return BackendResponse(
            str(choice["message"]["content"]), self.config.model,
            {
                "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage") if isinstance(body.get("usage"), dict) else None,
                "transport_attempt_count": 1,
                "transport_retry_count": 0,
                "rate_limit_retry_count": 0,
            },
        )


def load_json_object_strict(raw: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_object_key:{key}")
            result[key] = value
        return result
    value = json.loads(raw.strip(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("response_json_not_object")
    return value


class StateResolvedLiveCaller:
    def __init__(
        self, *, run_root: Path, ledger: RequestLedger,
        client: NoRetryOpenAIClient, token_counter: Callable[[str], int],
        root_id: str, context_limit: int,
        before_request: Callable[[str, str], None],
    ) -> None:
        self.run_root = Path(run_root)
        self.ledger = ledger
        self.client = client
        self.token_counter = token_counter
        self.root_id = root_id
        self.context_limit = int(context_limit)
        self.before_request = before_request
        self.records: list[dict[str, Any]] = []

    def call_json(
        self, *, request_id: str, role: str, prompt: str,
        prompt_version: str, max_tokens: int,
    ) -> tuple[dict[str, Any] | None, str, list[str]]:
        if role not in ROLE_METADATA:
            raise RuntimeError(f"state_resolved_live_role_not_allowed:{role}")
        logical_role, role_version = ROLE_METADATA[role]
        if self.ledger.read(request_id) is not None:
            raise RuntimeError(f"state_resolved_request_reuse_forbidden:{request_id}")
        prompt_tokens = self.token_counter(prompt)
        if prompt_tokens + int(max_tokens) > self.context_limit:
            raise RuntimeError(
                f"state_resolved_context_overflow:{prompt_tokens}+{max_tokens}>"
                f"{self.context_limit}")
        self.before_request(logical_role, role_version)
        prompt_path = self.run_root / "raw/prompts" / f"{request_id}.txt"
        atomic_text(prompt_path, prompt)
        atomic_json(prompt_path.with_suffix(".metadata.json"), {
            "request_id": request_id,
            "logical_role": logical_role,
            "role_version": role_version,
            "prompt_version": prompt_version,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "prompt_tokens_local": prompt_tokens,
            "max_completion_tokens": int(max_tokens),
            "context_limit": self.context_limit,
            "orchestrator_retry_allowed": False,
            "transport_retry_allowed": False,
        })
        record = {
            "request_id": request_id,
            "logical_role": logical_role,
            "role_version": role_version,
            "prompt_version": prompt_version,
            "prompt_tokens_local": prompt_tokens,
            "completion_tokens_local": 0,
            "total_tokens_local": prompt_tokens,
            "request_started": True,
            "request_completed": False,
        }
        self.records.append(record)

        try:
            response = self.ledger.invoke(
                request_id=request_id, intent=logical_role, attempt=1,
                prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
                call=lambda: self.client.complete(
                    prompt=prompt, max_tokens=int(max_tokens)),
                budget_class=logical_role, budget_limit=1, total_budget_limit=3,
                request_metadata={
                    "logical_role": logical_role,
                    "role_version": role_version,
                    "root_id": self.root_id,
                    "prompt_version": prompt_version,
                    "prompt_tokens_local": prompt_tokens,
                    "cache_allowed": False,
                    "imported_response_allowed": False,
                    "orchestrator_retry_allowed": False,
                    "transport_retry_allowed": False,
                },
                backend_identity={
                    "transport": "openai_compatible_no_retry",
                    "model": self.client.config.model,
                    "base_url": self.client.config.base_url,
                    "temperature": self.client.config.temperature,
                },
            )
        except Exception as error:
            record["runtime_error"] = f"{type(error).__name__}:{error}"
            atomic_json(self.run_root / "attempts" / f"{request_id}.json", record)
            raise

        raw = response.raw_text
        atomic_text(
            self.run_root / "raw/backend_responses" / f"{request_id}.txt", raw)
        completion_tokens = self.token_counter(raw)
        usage = response.metadata.get("usage") if isinstance(
            response.metadata, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else {}
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
        finish_reason = response.metadata.get("finish_reason")
        if finish_reason != "stop":
            errors.append(f"response_finish_reason:{finish_reason}")
        try:
            parsed = load_json_object_strict(raw)
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(f"response_json_invalid:{error}")
        record.update({
            "completion_tokens_local": completion_tokens,
            "total_tokens_local": prompt_tokens + completion_tokens,
            "request_completed": True,
            "provider_prompt_tokens": usage.get("prompt_tokens"),
            "provider_completion_tokens": usage.get("completion_tokens"),
            "provider_total_tokens": usage.get("total_tokens"),
            "model_identifier": response.model_identifier,
            "response_metadata": response.metadata,
            "raw_response_sha256": sha256_bytes(raw.encode("utf-8")),
            "parse_errors": errors,
        })
        atomic_json(self.run_root / "attempts" / f"{request_id}.json", {
            **record, "parsed_response": parsed,
        })
        return parsed, raw, errors


def aggregate_usage(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = {}
    for role in ("state_resolver", "planner", "round_realizer"):
        selected = [row for row in records if row.get("logical_role") == role]

        def total(provider: str, local: str) -> int:
            return sum(int(
                row.get(provider) if row.get(provider) is not None
                else row.get(local) or 0) for row in selected)

        roles[role] = {
            "requests": len(selected),
            "completed_requests": sum(bool(row.get("request_completed")) for row in selected),
            "prompt_tokens": total("provider_prompt_tokens", "prompt_tokens_local"),
            "completion_tokens": total("provider_completion_tokens", "completion_tokens_local"),
            "total_tokens": total("provider_total_tokens", "total_tokens_local"),
        }
    return {
        "roles": roles,
        "total_requests": sum(row["requests"] for row in roles.values()),
        "prompt_tokens": sum(row["prompt_tokens"] for row in roles.values()),
        "completion_tokens": sum(row["completion_tokens"] for row in roles.values()),
        "total_tokens": sum(row["total_tokens"] for row in roles.values()),
        "cache_response_count": 0,
        "imported_response_count": 0,
        "transport_retry_count": 0,
        "memorax_calls": 0,
        "qa_calls": 0,
        "solver_calls": 0,
    }


__all__ = [
    "NoRetryOpenAIClient", "ROLE_METADATA", "StateResolvedLiveCaller",
    "aggregate_usage", "load_json_object_strict",
]
