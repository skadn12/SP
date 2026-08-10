"""Live JSON transport restricted to the three v11 Hybrid roles."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ..core import atomic_json, atomic_text, sha256_bytes
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient


LIVE_ROLE_RUNTIME_V11_VERSION = "slim_v11_live_role_runtime_v1"
ALLOWED_LIVE_ROLES_V11 = frozenset({
    "planner_v11", "realizer_v11", "round_expansion_validator_v1",
})
ROLE_METADATA_V11 = {
    "planner_v11": ("planner", "anchor_first_continuous_planner_v11"),
    "realizer_v11": ("realizer", "continuous_natural_realizer_v11"),
    "round_expansion_validator_v1": (
        "round_validator", "round_expansion_validator_v1"),
}


def load_json_object_strict(raw: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                taxonomy = ("duplicate_action_key" if re.fullmatch(r"A\d{4,}", key)
                            else "duplicate_json_object_key")
                raise ValueError(f"{taxonomy}:{key}")
            result[key] = value
        return result
    value = json.loads(raw.strip(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("response_json_not_object")
    return value


class V11RoleCaller(Protocol):
    records: list[dict[str, Any]]

    def call_json(self, *, request_id: str, role: str, attempt: int,
                  prompt: str, prompt_version: str, max_tokens: int) -> tuple[
                      dict[str, Any] | None, str, list[str]]:
        ...


class LiveJsonRoleCallerV11:
    def __init__(
        self, *, run_root: Path, ledger: RequestLedger,
        client: OpenAICompatibleChallengerClient,
        token_counter: Callable[[str], int], root_id: str,
        role_budget_limits: Mapping[str, int], total_request_limit: int,
        context_limit: int, before_request: Callable[[str, str], None],
        paid_request_observer: Callable[[str, bool, int], None] | None = None,
    ):
        self.run_root = Path(run_root)
        self.ledger = ledger
        self.client = client
        self.token_counter = token_counter
        self.root_id = root_id
        self.role_budget_limits = dict(role_budget_limits)
        self.total_request_limit = int(total_request_limit)
        self.context_limit = int(context_limit)
        self.before_request = before_request
        self.paid_request_observer = paid_request_observer
        self.records: list[dict[str, Any]] = []

    def call_json(self, *, request_id: str, role: str, attempt: int,
                  prompt: str, prompt_version: str, max_tokens: int) -> tuple[
                      dict[str, Any] | None, str, list[str]]:
        if role not in ALLOWED_LIVE_ROLES_V11:
            raise RuntimeError(f"v11_live_role_not_allowed:{role}")
        logical_role, role_version = ROLE_METADATA_V11[role]
        prompt_tokens = self.token_counter(prompt)
        if prompt_tokens + max_tokens > self.context_limit:
            raise RuntimeError(
                f"v11_context_overflow:{prompt_tokens}+{max_tokens}>{self.context_limit}")
        if self.ledger.read(request_id) is not None:
            raise RuntimeError(f"v11_request_reuse_forbidden:{request_id}")
        if self.paid_request_observer is not None:
            self.paid_request_observer(logical_role, attempt > 1, attempt - 1)
        self.before_request(logical_role, role_version)
        prompt_path = self.run_root / "raw" / "prompts" / f"{request_id}.txt"
        atomic_text(prompt_path, prompt)
        atomic_json(prompt_path.with_suffix(".metadata.json"), {
            "request_id": request_id, "logical_role": logical_role,
            "role_version": role_version, "attempt": attempt,
            "prompt_version": prompt_version,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "prompt_tokens_local": prompt_tokens,
            "max_completion_tokens": max_tokens, "context_limit": self.context_limit,
        })
        record = {
            "request_id": request_id, "logical_role": logical_role,
            "role_version": role_version, "attempt": attempt,
            "prompt_version": prompt_version,
            "prompt_tokens_local": prompt_tokens,
            "completion_tokens_local": 0, "total_tokens_local": prompt_tokens,
            "request_started": True, "request_completed": False,
        }
        self.records.append(record)
        def transport_call():
            if role == "planner_v11" and hasattr(self.client, "complete_planner"):
                return self.client.complete_planner(
                    prompt=prompt, max_tokens=max_tokens, json_object=True)
            if (role == "round_expansion_validator_v1" and
                    hasattr(self.client, "complete_round_validator")):
                return self.client.complete_round_validator(
                    prompt=prompt, max_tokens=max_tokens, json_object=True)
            return self.client.complete(
                prompt=prompt, max_tokens=max_tokens, json_object=True)

        try:
            response = self.ledger.invoke(
                request_id=request_id, intent=logical_role, attempt=attempt,
                prompt_sha256=sha256_bytes(prompt.encode()),
                call=transport_call,
                budget_class=logical_role,
                budget_limit=self.role_budget_limits[logical_role],
                total_budget_limit=self.total_request_limit,
                request_metadata={
                    "logical_role": logical_role, "role_version": role_version,
                    "root_id": self.root_id, "prompt_version": prompt_version,
                    "prompt_tokens_local": prompt_tokens, "cache_allowed": False,
                    "imported_response_allowed": False,
                },
                backend_identity={
                    "transport": "openai_compatible_challenger",
                    "model": self.client.config.model,
                    "base_url": self.client.config.base_url,
                    "temperature": self.client.config.temperature,
                })
        except Exception as error:
            record["runtime_error"] = f"{type(error).__name__}:{error}"
            record["transport_attempt_count"] = int(
                getattr(error, "transport_attempt_count", 1))
            if getattr(error, "classification", None):
                record["transport_timeout_classification"] = error.classification
            atomic_json(self.run_root / "attempts" / f"{request_id}.json", record)
            raise
        raw = response.raw_text
        completion_tokens = self.token_counter(raw)
        usage = response.metadata.get("usage") if isinstance(
            response.metadata, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else {}
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
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
            "raw_response_sha256": sha256_bytes(raw.encode()),
            "parse_errors": errors,
        })
        atomic_json(self.run_root / "attempts" / f"{request_id}.json", {
            **record, "parsed_response": parsed,
        })
        return parsed, raw, errors


def aggregate_v11_usage(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = {}
    for role in ("planner", "realizer", "round_validator"):
        selected = [row for row in records if row.get("logical_role") == role]
        def total(provider: str, local: str) -> int:
            return sum(int(row.get(provider) if row.get(provider) is not None
                           else row.get(local) or 0) for row in selected)
        roles[role] = {
            "logical_requests": len(selected),
            "physical_requests": sum(int(
                (row.get("response_metadata") or {}).get("transport_attempt_count") or
                row.get("transport_attempt_count") or 1) for row in selected),
            "completed_requests": sum(bool(row.get("request_completed")) for row in selected),
            "retries": sum(int(row.get("attempt") or 1) > 1 for row in selected),
            "prompt_tokens": total("provider_prompt_tokens", "prompt_tokens_local"),
            "completion_tokens": total(
                "provider_completion_tokens", "completion_tokens_local"),
            "total_tokens": total("provider_total_tokens", "total_tokens_local"),
        }
    return {
        "roles": roles,
        "total_physical_requests": sum(row["physical_requests"] for row in roles.values()),
        "prompt_tokens": sum(row["prompt_tokens"] for row in roles.values()),
        "completion_tokens": sum(row["completion_tokens"] for row in roles.values()),
        "total_tokens": sum(row["total_tokens"] for row in roles.values()),
        "evidence_committer_calls": 0, "initial_structure_extractor_calls": 0,
        "cache_response_count": 0, "imported_response_count": 0,
        "legacy_response_reuse_count": 0,
    }
