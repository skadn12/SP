"""Live JSON runtime for Verifier, Planner, outline audit, Gate, and Realizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from ..core import atomic_json, atomic_text, sha256_bytes
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient


LIVE_RUNTIME_VERSION = "memory_event_continuation_live_runtime_v1"
PER_ROLE_REQUEST_BUDGET = 1
TOTAL_REQUEST_BUDGET = 12
ROLE_REQUEST_BUDGETS = {
    "semantic_continuation_gate": 2,
    # Initial audit, one failed-scene continuation re-audit, and (if needed)
    # one replacement-scene audit are all bounded attempts in one round.
    "realizer_adherence_auditor": 3,
}
ALLOWED_LIVE_ROLES = frozenset({
    "transcript_event_verifier_v1",
    "simple_append_planner_v1",
    "simple_append_planner_candidate_repair_v1",
    "planner_outline_semantic_auditor_v1",
    "memory_event_semantic_continuation_gate_v1",
    "memory_event_realization_aware_semantic_gate_v1",
    "simple_append_round_realizer_v1",
    "memory_event_realizer_adherence_auditor_v1",
    "simple_append_scene_continuation_realizer_v1",
    "simple_append_scene_replacement_realizer_v1",
    "memory_event_realizer_adherence_reaudit_v1",
})
ROLE_METADATA = {
    "transcript_event_verifier_v1": (
        "event_verifier", "transcript_event_verifier_v1"),
    "simple_append_planner_v1": ("planner", "simple_append_planner_v1"),
    "simple_append_planner_candidate_repair_v1": (
        "planner_repair", "simple_append_planner_candidate_repair_v1"),
    "planner_outline_semantic_auditor_v1": (
        "planner_outline_semantic_auditor",
        "planner_outline_semantic_auditor_v1"),
    "memory_event_semantic_continuation_gate_v1": (
        "semantic_continuation_gate",
        "memory_event_semantic_continuation_gate_v1"),
    "memory_event_realization_aware_semantic_gate_v1": (
        "semantic_continuation_gate",
        "memory_event_realization_aware_semantic_gate_v1"),
    "simple_append_round_realizer_v1": (
        "round_realizer", "simple_append_round_realizer_v1"),
    "memory_event_realizer_adherence_auditor_v1": (
        "realizer_adherence_auditor",
        "memory_event_realizer_adherence_auditor_v1"),
    "simple_append_scene_continuation_realizer_v1": (
        "scene_continuation_realizer",
        "simple_append_scene_continuation_realizer_v1"),
    "simple_append_scene_replacement_realizer_v1": (
        "scene_replacement_realizer",
        "simple_append_scene_replacement_realizer_v1"),
    "memory_event_realizer_adherence_reaudit_v1": (
        "realizer_adherence_reaudit",
        "memory_event_realizer_adherence_reaudit_v1"),
}


def validate_outbound_role_coverage(
    required_roles: tuple[str, ...] | list[str], *,
    allowed_roles: frozenset[str] | set[str] = ALLOWED_LIVE_ROLES,
) -> dict[str, Any]:
    """Check every configured outbound role before any live request."""
    required = [str(role) for role in required_roles]
    allowed = {str(role) for role in allowed_roles}
    missing = [role for role in required if role not in allowed]
    metadata_missing = [role for role in required if role not in ROLE_METADATA]
    return {
        "required_outbound_roles": required,
        "allowed_outbound_roles": sorted(allowed),
        "missing_outbound_roles": missing,
        "missing_role_metadata": metadata_missing,
        "valid": not missing and not metadata_missing,
    }


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


class SimpleAppendLiveCaller:
    def __init__(
        self, *, run_root: Path, ledger: RequestLedger,
        client: OpenAICompatibleChallengerClient,
        token_counter: Callable[[str], int], root_id: str,
        context_limit: int, before_request: Callable[[str, str], None],
        total_request_budget: int = TOTAL_REQUEST_BUDGET,
        role_request_budgets: Mapping[str, int] | None = None,
    ):
        self.run_root = Path(run_root)
        self.ledger = ledger
        self.client = client
        self.token_counter = token_counter
        self.root_id = root_id
        self.context_limit = int(context_limit)
        self.before_request = before_request
        self.total_request_budget = int(total_request_budget)
        self.role_request_budgets = dict(role_request_budgets or ROLE_REQUEST_BUDGETS)
        self.records: list[dict[str, Any]] = []

    def call_json(
        self, *, request_id: str, role: str, prompt: str,
        prompt_version: str, max_tokens: int,
    ) -> tuple[dict[str, Any] | None, str, list[str]]:
        if role not in ALLOWED_LIVE_ROLES:
            raise RuntimeError(f"simple_append_live_role_not_allowed:{role}")
        logical_role, role_version = ROLE_METADATA[role]
        if self.ledger.read(request_id) is not None:
            raise RuntimeError(f"simple_append_request_reuse_forbidden:{request_id}")
        prompt_tokens = self.token_counter(prompt)
        if prompt_tokens + int(max_tokens) > self.context_limit:
            raise RuntimeError(
                f"simple_append_context_overflow:{prompt_tokens}+{max_tokens}>"
                f"{self.context_limit}")
        self.before_request(logical_role, role_version)
        prompt_path = self.run_root / "raw" / "prompts" / f"{request_id}.txt"
        atomic_text(prompt_path, prompt)
        atomic_json(prompt_path.with_suffix(".metadata.json"), {
            "request_id": request_id,
            "logical_role": logical_role,
            "role_version": role_version,
            "prompt_version": prompt_version,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "prompt_tokens_local": prompt_tokens,
            "max_completion_tokens": int(max_tokens),
            "context_limit": self.context_limit,
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

        def transport_call():
            return self.client.complete(
                prompt=prompt, max_tokens=int(max_tokens), json_object=True)

        try:
            response = self.ledger.invoke(
                request_id=request_id, intent=logical_role, attempt=1,
                prompt_sha256=sha256_bytes(prompt.encode()),
                call=transport_call, budget_class=logical_role,
                budget_limit=self.role_request_budgets.get(
                    logical_role, PER_ROLE_REQUEST_BUDGET),
                total_budget_limit=self.total_request_budget,
                request_metadata={
                    "logical_role": logical_role,
                    "role_version": role_version,
                    "root_id": self.root_id,
                    "prompt_version": prompt_version,
                    "prompt_tokens_local": prompt_tokens,
                    "cache_allowed": False,
                    "imported_response_allowed": False,
                    "orchestrator_retry_allowed": False,
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


def aggregate_usage(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = {}
    for role in ("event_verifier", "planner", "planner_repair",
                 "planner_outline_semantic_auditor",
                 "semantic_continuation_gate", "round_realizer",
                 "scene_continuation_realizer", "realizer_adherence_auditor",
                 "scene_replacement_realizer",
                 "realizer_adherence_reaudit"):
        selected = [row for row in records if row.get("logical_role") == role]

        def total(provider: str, local: str) -> int:
            return sum(int(row.get(provider) if row.get(provider) is not None
                           else row.get(local) or 0) for row in selected)

        roles[role] = {
            "requests": len(selected),
            "completed_requests": sum(bool(row.get("request_completed"))
                                      for row in selected),
            "prompt_tokens": total("provider_prompt_tokens", "prompt_tokens_local"),
            "completion_tokens": total(
                "provider_completion_tokens", "completion_tokens_local"),
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
        "memory_calls": 0,
        "qa_calls": 0,
    }
