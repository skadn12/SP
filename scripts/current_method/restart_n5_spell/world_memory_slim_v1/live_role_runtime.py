"""Business-neutral live JSON role runtime for the frozen v7 runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ..core import atomic_json, atomic_text, sha256_bytes
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient


LIVE_ROLE_RUNTIME_VERSION = "slim_v7_live_role_runtime_v1"
ALLOWED_LIVE_ROLES = frozenset({
    "planner_v3", "realizer_v3", "evidence_committer_scope_v1",
})
ROLE_METADATA = {
    "planner_v3": ("planner", "slim_world_planner_v3"),
    "realizer_v3": ("realizer", "natural_realizer_v3"),
    "evidence_committer_scope_v1": (
        "evidence_committer", "evidence_committer_scope_audit_v1"),
}


class RoleCaller(Protocol):
    records: list[dict[str, Any]]

    def call_json(self, *, request_id: str, role: str, attempt: int,
                  prompt: str, prompt_version: str, max_tokens: int) -> tuple[
                      dict[str, Any] | None, str, list[str]]:
        ...


class LiveJsonRoleCaller:
    def __init__(
        self, *, run_root: Path, ledger: RequestLedger,
        client: OpenAICompatibleChallengerClient,
        token_counter: Callable[[str], int], root_id: str,
        role_budget_limits: Mapping[str, int], total_request_limit: int,
        context_limit: int, before_request: Callable[[str, str], None],
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
        self.records: list[dict[str, Any]] = []

    def call_json(self, *, request_id: str, role: str, attempt: int,
                  prompt: str, prompt_version: str, max_tokens: int) -> tuple[
                      dict[str, Any] | None, str, list[str]]:
        if role not in ALLOWED_LIVE_ROLES:
            raise RuntimeError(f"live_role_not_allowed:{role}")
        logical_role, role_version = ROLE_METADATA[role]
        prompt_tokens = self.token_counter(prompt)
        if prompt_tokens + max_tokens > self.context_limit:
            raise RuntimeError(
                f"context_overflow:{prompt_tokens}+{max_tokens}>{self.context_limit}")
        if self.ledger.read(request_id) is not None:
            raise RuntimeError(f"v7_request_reuse_forbidden:{request_id}")
        self.before_request(logical_role, role_version)
        prompt_path = self.run_root / "raw" / "prompts" / f"{request_id}.txt"
        atomic_text(prompt_path, prompt)
        atomic_json(prompt_path.with_suffix(".metadata.json"), {
            "request_id": request_id, "logical_role": logical_role,
            "role_version": role_version, "attempt": attempt,
            "prompt_version": prompt_version,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "prompt_tokens_local": prompt_tokens,
            "max_completion_tokens": max_tokens,
            "context_limit": self.context_limit,
        })
        record = {
            "request_id": request_id, "logical_role": logical_role,
            "role_version": role_version, "attempt": attempt,
            "prompt_version": prompt_version,
            "prompt_tokens_local": prompt_tokens,
            "completion_tokens_local": 0,
            "total_tokens_local": prompt_tokens,
            "request_started": True, "request_completed": False,
        }
        self.records.append(record)
        try:
            response = self.ledger.invoke(
                request_id=request_id, intent=logical_role, attempt=attempt,
                prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
                call=lambda: self.client.complete(
                    prompt=prompt, max_tokens=max_tokens, json_object=True),
                budget_class=logical_role,
                budget_limit=self.role_budget_limits[logical_role],
                total_budget_limit=self.total_request_limit,
                request_metadata={
                    "logical_role": logical_role, "role_version": role_version,
                    "root_id": self.root_id, "prompt_version": prompt_version,
                    "prompt_tokens_local": prompt_tokens,
                    "cache_allowed": False, "imported_response_allowed": False,
                },
                backend_identity={
                    "transport": "openai_compatible_challenger",
                    "model": self.client.config.model,
                    "base_url": self.client.config.base_url,
                    "temperature": self.client.config.temperature,
                })
        except Exception as error:
            record["runtime_error"] = f"{type(error).__name__}:{error}"
            atomic_json(self.run_root / "attempts" / f"{request_id}.json", record)
            raise
        raw = response.raw_text
        completion_tokens = self.token_counter(raw)
        provider_usage = (response.metadata.get("usage")
                          if isinstance(response.metadata, Mapping) and
                          isinstance(response.metadata.get("usage"), Mapping)
                          else {})
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
        try:
            value = (load_json_object_strict(raw)
                     if role == "planner_v3" else json.loads(raw.strip()))
            if not isinstance(value, dict):
                raise ValueError("response_json_not_object")
            parsed = value
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(f"response_json_invalid:{error}")
        record.update({
            "completion_tokens_local": completion_tokens,
            "total_tokens_local": prompt_tokens + completion_tokens,
            "request_completed": True,
            "provider_prompt_tokens": provider_usage.get("prompt_tokens"),
            "provider_completion_tokens": provider_usage.get("completion_tokens"),
            "provider_total_tokens": provider_usage.get("total_tokens"),
            "model_identifier": response.model_identifier,
            "response_metadata": response.metadata,
            "raw_response_sha256": sha256_bytes(raw.encode("utf-8")),
            "parse_errors": errors,
        })
        atomic_json(self.run_root / "attempts" / f"{request_id}.json", {
            **record, "parsed_response": parsed,
        })
        return parsed, raw, errors


def load_json_object_strict(raw: str) -> dict[str, Any]:
    """Parse one JSON object without silently overwriting duplicate keys."""
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                taxonomy = ("duplicate_bundle_action_key"
                            if re.fullmatch(r"A\d{4,}", key)
                            else "duplicate_json_object_key")
                raise ValueError(f"{taxonomy}:{key}")
            result[key] = value
        return result

    value = json.loads(raw.strip(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("response_json_not_object")
    return value


def aggregate_role_usage(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = {}
    for role in ("planner", "realizer", "evidence_committer"):
        selected = [row for row in records if row.get("logical_role") == role]
        prompt_tokens = sum(int(row.get("provider_prompt_tokens")
                                if row.get("provider_prompt_tokens") is not None
                                else row.get("prompt_tokens_local") or 0)
                            for row in selected)
        completion_tokens = sum(int(row.get("provider_completion_tokens")
                                    if row.get("provider_completion_tokens") is not None
                                    else row.get("completion_tokens_local") or 0)
                                for row in selected)
        total_tokens = sum(int(row.get("provider_total_tokens")
                               if row.get("provider_total_tokens") is not None
                               else row.get("total_tokens_local") or 0)
                           for row in selected)
        roles[role] = {
            "physical_requests": len(selected),
            "completed_requests": sum(bool(row.get("request_completed", True))
                                      for row in selected),
            "failed_or_ambiguous_requests": sum(not bool(row.get("request_completed", True))
                                                for row in selected),
            "retries": sum(int(row.get("attempt") or 1) > 1 for row in selected),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_local": sum(int(row.get("prompt_tokens_local") or 0)
                                       for row in selected),
            "completion_tokens_local": sum(int(row.get("completion_tokens_local") or 0)
                                           for row in selected),
            "total_tokens_local": sum(int(row.get("total_tokens_local") or 0)
                                      for row in selected),
        }
    return {
        "roles": roles,
        "total_physical_requests": sum(value["physical_requests"] for value in roles.values()),
        "prompt_tokens": sum(value["prompt_tokens"] for value in roles.values()),
        "completion_tokens": sum(value["completion_tokens"] for value in roles.values()),
        "total_tokens": sum(value["total_tokens"] for value in roles.values()),
        "cache_response_count": 0, "imported_response_count": 0,
        "legacy_response_reuse_count": 0,
    }
