"""Isolated live Fresh E2E v6 runner for World Memory Slim Planner v2."""

from __future__ import annotations

import argparse
import collections
import copy
import difflib
import hashlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..unified_dialogue_root import read_unified_parent, validate_unified_dialogue_root
from .affordance_compiler import (
    AFFORDANCE_SCHEMA_VERSION,
    action_menu_parent_turn_ids,
    compile_delta_affordances,
    compile_selected_actions,
)
from .binder import LOCAL_STRUCTURAL_BINDER_VERSION, bind_scene
from .memory import (
    commit_confirmed_delta,
    commit_emergent_thread,
    initialize_world_memory,
    stage_emergent_claims,
    stage_model_claims,
)
from .prompts import (
    EVIDENCE_COMMITTER_PROMPT_VERSION,
    EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION,
    NATURAL_REALIZER_CLAIM_ONLY_RETRY_PROMPT_VERSION,
    NATURAL_REALIZER_PROMPT_VERSION,
    SLIM_PLANNER_PROMPT_VERSION,
    SLIM_PLANNER_RETRY_PROMPT_VERSION,
    build_evidence_committer_prompt,
    build_evidence_committer_retry_prompt,
    build_natural_realizer_prompt,
    build_realizer_claim_only_retry_prompt,
    build_slim_planner_prompt,
    build_slim_planner_retry_prompt,
)
from .provenance import (
    audit_legacy_import_isolation,
    git_value,
    hash_tree,
    project_diff_sha256,
)
from .realizer_claims import (
    build_emergent_committer_routes,
    compile_emergent_pending_proposals,
    reconstruct_scene_from_claim_only,
    validate_claim_only_retry_contract,
    validate_committer_candidate_subset,
    validate_realizer_scene_contract,
)
from .schemas import (
    DIFFICULTY_STRATEGY_SCHEMA_VERSION,
    EVIDENCE_COMMITTER_SCHEMA_VERSION,
    PLANNER_MUTATION_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
    QA_REPLAY_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
    WORLD_MEMORY_SCHEMA_VERSION,
    schema_sha256s,
    validate_evidence_committer_output,
    validate_slim_planner_output,
    validate_world_memory,
)
from .structural import (
    STRUCTURAL_DELTA_VALIDATOR_VERSION,
    validate_planner_memory_references,
    validate_structural_deltas,
)


LIVE_FRESH_E2E_RUNNER_VERSION = "world_memory_slim_live_fresh_e2e_v6"
CHILD_DIALOGUE_SCHEMA_VERSION = "world_memory_slim_child_dialogue_v6"
SUMMARY_SCHEMA_VERSION = "world_memory_slim_fresh_e2e_v6_summary_v1"
POST_RUN_PROVENANCE_VERSION = "world_memory_slim_post_run_provenance_v6"
DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v6")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
PROJECT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parent
HISTORY_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
HISTORICAL_V1_RUN = HISTORY_ROOT / (
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v1")
HISTORICAL_V2_RUN = HISTORY_ROOT / (
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v2")
HISTORICAL_V3_RUN = HISTORY_ROOT / (
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v3")


@dataclass(frozen=True)
class RunLimits:
    planner_attempts: int = 3
    realizer_attempts_per_scene: int = 3
    committer_attempts_per_claim: int = 3
    scene_minimum: int = 2
    scene_target: int = 3
    scene_maximum: int = 4
    maximum_total_deltas: int = 6
    planner_max_tokens: int = 4096
    realizer_max_tokens: int = 4096
    committer_max_tokens: int = 2048
    context_limit: int = 40960
    total_physical_request_limit: int = 32


class StageFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_metadata_fingerprint(root: Path, excluded: Path) -> dict[str, Any]:
    """Audit old runs without opening or reusing any historical artifact body."""
    root = root.resolve()
    excluded = excluded.resolve()
    digest = hashlib.sha256()
    file_count = total_bytes = 0
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file() and excluded not in candidate.parents):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        record = f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\0"
        digest.update(record.encode())
        file_count += 1
        total_bytes += stat.st_size
    return {
        "root": str(root), "excluded_current_run": str(excluded),
        "file_count": file_count, "total_bytes": total_bytes,
        "metadata_fingerprint_sha256": digest.hexdigest(),
        "scope": "sorted(relative_path NUL size NUL mtime_ns NUL); file bodies not read",
    }


def _source_manifest() -> dict[str, Any]:
    paths = sorted(PACKAGE_ROOT.glob("*.py")) + [
        PACKAGE_ROOT.parent / "core.py",
        PACKAGE_ROOT.parent / "ledger.py",
        PACKAGE_ROOT.parent / "live_backends.py",
        PACKAGE_ROOT.parent / "backends.py",
        PACKAGE_ROOT.parent / "unified_dialogue_root.py",
        PACKAGE_ROOT.parent / "input_adapter.py",
    ]
    return {
        "files": [{
            "path": str(path.relative_to(PROJECT)),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        } for path in sorted(set(paths))],
    }


def _memory_prompt_view(memory: Mapping[str, Any], *, include_history: bool) -> dict[str, Any]:
    fields = [
        "world_memory_schema_version", "root_id", "memory_version", "entities",
        "relations", "current_states", "open_threads", "completed_threads",
        "pending_memory_updates",
    ]
    view = {field: memory[field] for field in fields}
    if include_history:
        view["event_history"] = memory["event_history"]
    return view


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"response_not_exact_json:{error.msg}:line={error.lineno}") from error
    if not isinstance(value, dict):
        raise ValueError("response_json_not_object")
    return value


def _retry_prompt(base: str, *, role: str, errors: Sequence[str], previous: Any) -> str:
    previous_text = (canonical_json(previous) if isinstance(previous, (dict, list))
                     else str(previous))
    return "\n".join([
        base,
        "",
        f"RETRY FOR {role}: the previous full response was rejected.",
        "Return a new complete JSON object. Do not omit, rename, or add fields.",
        "Validation errors:",
        canonical_json(list(errors)),
        "Previous rejected response:",
        previous_text,
    ])


class LiveRoleCaller:
    def __init__(self, *, run_root: Path, ledger: RequestLedger,
                 client: OpenAICompatibleChallengerClient, tokenizer: Any,
                 root_id: str, limits: RunLimits):
        self.run_root = run_root
        self.ledger = ledger
        self.client = client
        self.tokenizer = tokenizer
        self.root_id = root_id
        self.limits = limits
        self.records: list[dict[str, Any]] = []

    def call_json(self, *, request_id: str, role: str, attempt: int,
                  prompt: str, prompt_version: str, max_tokens: int) -> tuple[
                      dict[str, Any] | None, str, list[str]]:
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens + max_tokens > self.limits.context_limit:
            raise StageFailure(role, f"context_overflow:{prompt_tokens}+{max_tokens}")
        prompt_path = self.run_root / "raw" / "prompts" / f"{request_id}.txt"
        atomic_text(prompt_path, prompt)
        atomic_json(prompt_path.with_suffix(".metadata.json"), {
            "request_id": request_id, "role": role, "attempt": attempt,
            "prompt_version": prompt_version,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "prompt_tokens_local": prompt_tokens,
            "max_completion_tokens": max_tokens,
            "context_limit": self.limits.context_limit,
        })
        response = self.ledger.invoke(
            request_id=request_id, intent=role, attempt=attempt,
            prompt_sha256=sha256_bytes(prompt.encode()),
            call=lambda: self.client.complete(
                prompt=prompt, max_tokens=max_tokens, json_object=True),
            budget_class=role,
            budget_limit={
                "planner": self.limits.planner_attempts,
                "realizer": (self.limits.scene_maximum *
                             self.limits.realizer_attempts_per_scene),
                "evidence_committer": (self.limits.maximum_total_deltas *
                                       self.limits.committer_attempts_per_claim),
            }[role],
            total_budget_limit=self.limits.total_physical_request_limit,
            request_metadata={
                "role": role, "root_id": self.root_id,
                "prompt_version": prompt_version,
                "prompt_tokens_local": prompt_tokens,
                "max_completion_tokens": max_tokens,
                "cache_allowed": False, "imported_response_allowed": False,
            },
            backend_identity={
                "transport": "openai_compatible_challenger",
                "model": self.client.config.model,
                "base_url": self.client.config.base_url,
                "temperature": self.client.config.temperature,
            },
        )
        raw = response.raw_text
        completion_tokens = len(self.tokenizer.encode(raw, add_special_tokens=False))
        errors: list[str] = []
        parsed: dict[str, Any] | None = None
        try:
            parsed = _parse_json_object(raw)
        except ValueError as error:
            errors.append(str(error))
        record = {
            "request_id": request_id, "role": role, "attempt": attempt,
            "prompt_version": prompt_version,
            "prompt_tokens_local": prompt_tokens,
            "completion_tokens_local": completion_tokens,
            "total_tokens_local": prompt_tokens + completion_tokens,
            "model_identifier": response.model_identifier,
            "response_metadata": response.metadata,
            "raw_response_sha256": sha256_bytes(raw.encode()),
            "parse_errors": errors,
        }
        self.records.append(record)
        atomic_json(self.run_root / "attempts" / f"{request_id}.json", {
            **record, "parsed_response": parsed,
        })
        return parsed, raw, errors


def _planner_result(
    plan: Mapping[str, Any], memory: Mapping[str, Any], *, root_id: str,
    limits: RunLimits, action_menu: Mapping[str, Any],
) -> dict[str, Any]:
    schema = validate_slim_planner_output(plan, action_menu=action_menu)
    if not schema["valid"]:
        return {"accepted": False, "errors": list(schema["errors"]),
                "schema_validation": schema}
    normalized = compile_selected_actions(plan, action_menu)
    references = validate_planner_memory_references(
        memory, normalized, action_menu=action_menu)
    all_deltas = [delta for scene in normalized["scene_goals"]
                  for delta in scene["proposed_graph_deltas"]]
    structural = validate_structural_deltas(
        memory, all_deltas, expected_root_id=root_id, action_menu=action_menu)
    accepted_ids = set(structural["accepted_delta_ids"])
    executable = []
    rejected_scenes = []
    for scene in normalized["scene_goals"]:
        accepted = [delta for delta in scene["proposed_graph_deltas"]
                    if delta["proposed_delta_id"] in accepted_ids]
        rejected = [delta["proposed_delta_id"] for delta in
                    scene["proposed_graph_deltas"]
                    if delta["proposed_delta_id"] not in accepted_ids]
        row = {**scene, "accepted_deltas": accepted,
               "rejected_delta_ids": rejected}
        if accepted:
            executable.append(row)
        else:
            rejected_scenes.append(scene["scene_key"])
    errors: list[str] = []
    if not references["valid"]:
        errors.extend(references["errors"])
    if not limits.scene_minimum <= len(normalized["scene_goals"]) <= limits.scene_maximum:
        errors.append("scene_goal_count_outside_2_to_4_budget")
    if len(all_deltas) > limits.maximum_total_deltas:
        errors.append("total_delta_canary_budget_exceeded")
    if len(executable) < limits.scene_minimum:
        errors.append("fewer_than_two_executable_scene_goals")
    return {
        "accepted": not errors,
        "errors": errors,
        "schema_validation": schema,
        "memory_reference_validation": references,
        "normalized_plan": normalized,
        "structural_validation": structural,
        "executable_scenes": executable,
        "rejected_scene_keys": rejected_scenes,
        "proposed_delta_count": len(all_deltas),
        "accepted_delta_count": len(accepted_ids),
        "rejected_delta_count": len(structural["rejected_delta_ids"]),
    }


def _realizer_result(
    output: Mapping[str, Any], *, scene: Mapping[str, Any],
    accepted_deltas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_root_id = (str(accepted_deltas[0].get("root_id"))
                        if accepted_deltas else None)
    return validate_realizer_scene_contract(
        output, scene=scene, accepted_deltas=accepted_deltas,
        expected_root_id=expected_root_id)


def _committer_result(
    output: Mapping[str, Any], *, delta_id: str,
    candidate_evidence_turn_keys: set[str],
    realizer_claimed_status: str | None = None,
) -> dict[str, Any]:
    validation = validate_evidence_committer_output(output)
    errors = list(validation["errors"])
    if output.get("proposed_delta_id") != delta_id:
        errors.append("committer_delta_identity_mismatch")
    subset = validate_committer_candidate_subset(
        output, candidate_evidence_turn_keys=sorted(candidate_evidence_turn_keys),
        realizer_claimed_status=realizer_claimed_status)
    errors.extend(subset["errors"])
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "schema_validation": validation, "candidate_subset_validation": subset}


def _related_parent_turns(
    *, parent_turns: Sequence[Mapping[str, Any]], memory: Mapping[str, Any],
    scene: Mapping[str, Any], selected_memory_ids: Sequence[str],
) -> list[dict[str, Any]]:
    memory_ids = set(scene.get("related_memory_ids") or []) | set(selected_memory_ids)
    source_ids = {
        turn_id for record in memory.get("event_history") or []
        if record.get("memory_id") in memory_ids
        for turn_id in record.get("source_turn_ids") or []
    }
    participants = set(scene.get("participants") or [])
    participant_indices = [index for index, turn in enumerate(parent_turns)
                           if turn.get("speaker") in participants]
    source_ids.update(parent_turns[index]["turn_id"]
                      for index in participant_indices[-6:])
    return [dict(turn) for turn in parent_turns if turn.get("turn_id") in source_ids]


def _scene_memory_subset(memory: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, Any]:
    ids = set(scene.get("related_memory_ids") or [])
    records = [record for record in memory.get("event_history") or []
               if record.get("memory_id") in ids]
    participants = set(scene.get("participants") or [])
    entities = [entity for entity in memory.get("entities") or []
                if entity.get("display_name") in participants]
    return {
        "world_memory_schema_version": memory["world_memory_schema_version"],
        "root_id": memory["root_id"], "memory_version": memory["memory_version"],
        "entities": entities, "related_records": records,
        "current_states": memory["current_states"],
        "open_threads": memory["open_threads"],
        "completed_threads": memory["completed_threads"],
    }


def _quality_stats(output: Mapping[str, Any], *, tokenizer: Any,
                   committer_outputs: Sequence[Mapping[str, Any]],
                   known_entities: set[str]) -> dict[str, Any]:
    turns = output["dialogue_turns"]
    normalized = [" ".join(str(turn["text"]).casefold().split()) for turn in turns]
    exact_duplicates = len(normalized) - len(set(normalized))
    near_pairs = []
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            ratio = difflib.SequenceMatcher(None, normalized[left], normalized[right]).ratio()
            if normalized[left] != normalized[right] and ratio >= 0.9:
                near_pairs.append({"left": left, "right": right,
                                   "similarity": round(ratio, 4)})
    counts = collections.Counter(str(row["decision"]) for row in committer_outputs)
    speakers = list(dict.fromkeys(str(turn["speaker"]) for turn in turns))
    return {
        "turn_count": len(turns),
        "token_count": len(tokenizer.encode(
            "\n".join(str(turn["text"]) for turn in turns),
            add_special_tokens=False)),
        "participants": speakers,
        "duplicate_utterance_count": exact_duplicates,
        "near_duplicate_utterance_count": len(near_pairs),
        "near_duplicate_pairs": near_pairs,
        "new_character_names": sorted(set(speakers) - known_entities),
        "new_character_connection_valid": not (set(speakers) - known_entities),
        "claimed_delta_count": len(output["claimed_graph_updates"]),
        "confirmed_count": counts["confirmed"],
        "rejected_count": counts["rejected"],
        "uncertain_count": counts["uncertain"],
    }


def _aggregate_request_ledger(run_root: Path, ledger: RequestLedger,
                              records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events_path = run_root / "ledger" / "request_events.jsonl"
    atomic_text(run_root / "request_ledger.jsonl",
                events_path.read_text(encoding="utf-8") if events_path.exists() else "")
    by_role: dict[str, dict[str, int]] = {}
    provider_tokens = {"prompt_tokens": 0, "completion_tokens": 0,
                       "total_tokens": 0, "responses_with_usage": 0}
    for record in records:
        role = str(record["role"])
        row = by_role.setdefault(role, {
            "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "retries": 0,
        })
        row["requests"] += 1
        row["prompt_tokens"] += int(record["prompt_tokens_local"])
        row["completion_tokens"] += int(record["completion_tokens_local"])
        row["total_tokens"] += int(record["total_tokens_local"])
        row["retries"] += int(record["attempt"] > 1)
        usage = (record.get("response_metadata") or {}).get("usage")
        if isinstance(usage, Mapping):
            provider_tokens["responses_with_usage"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                provider_tokens[key] += int(usage.get(key) or 0)
    summary = ledger.summary()
    return {
        "by_role": by_role, "provider_tokens": provider_tokens, **summary,
        "physical_requests": summary["paid_api_generation_count"],
        "cache_reuse_count": 0, "imported_response_count": 0,
        "legacy_artifact_reuse_count": 0,
        "estimated_cost_usd": None,
        "estimated_cost_note": "No authoritative price table was supplied; token usage is recorded.",
    }


def _memory_log_jsonl(memory: Mapping[str, Any]) -> str:
    rows = [json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in memory.get("memory_update_log") or []]
    return "\n".join(rows) + ("\n" if rows else "")


def _child_payload(root_id: str, sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    turns = [turn for session in sessions for turn in session["dialogue_turns"]]
    return {
        "schema_version": CHILD_DIALOGUE_SCHEMA_VERSION,
        "root_id": root_id, "child_id": f"{root_id}:child:fresh_n1",
        "scene_count": len(sessions), "turn_count": len(turns),
        "sessions": list(sessions),
        "dialogue_sha256": sha256_bytes(canonical_json(turns).encode()),
    }


def _import_graph() -> dict[str, Any]:
    rows = []
    prefix = "scripts.current_method.restart_n5_spell"
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith(prefix) and path:
            rows.append({"module": name, "path": str(Path(path).resolve())})
    banned = {
        "stateful_graph_expansion", "stateful_outline_planner",
        "stateful_graph_orchestrator", "dynamic_generation_canary",
    }
    findings = [row for row in rows if row["module"].rsplit(".", 1)[-1] in banned]
    return {
        "entry_module": __name__, "modules": rows,
        "banned_legacy_semantic_modules": sorted(banned),
        "banned_findings": findings,
        "legacy_semantic_validator_called": False,
        "isolated": not findings,
    }


def _render_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# World Memory Slim Fresh E2E {summary['run_version']} n=1", "",
        f"- root: `{summary['root_id']}`",
        f"- generation_ready: `{str(summary['generation_ready']).lower()}`",
        f"- fresh_e2e_ready: `{str(summary['fresh_e2e_ready']).lower()}`",
        f"- failure_stage: `{summary.get('failure_stage')}`",
        f"- action menu create/update/transition: "
        f"`{summary['affordance_menu']['operation_class_distribution'].get('create', 0)} / "
        f"{summary['affordance_menu']['operation_class_distribution'].get('update', 0)} / "
        f"{summary['affordance_menu']['operation_class_distribution'].get('transition', 0)}`",
        f"- Planner attempts / scenes: `{summary['planner']['attempt_count']} / "
        f"{summary['planner']['final_scene_count']}`",
        f"- selected actions: `{', '.join(summary['planner']['selected_action_ids'])}`",
        f"- scenes: `{summary['scene_count']}`",
        f"- child sessions / turns: `{summary['child_session_count']} / "
        f"{summary['child_turn_count']}`",
        f"- proposed / accepted / structurally rejected: "
        f"`{summary['delta_counts']['proposed']} / {summary['delta_counts']['accepted']} / "
        f"{summary['delta_counts']['structurally_rejected']}`",
        f"- confirmed / rejected / uncertain: "
        f"`{summary['delta_counts']['confirmed']} / {summary['delta_counts']['rejected']} / "
        f"{summary['delta_counts']['uncertain']}`",
        f"- authoritative memory updates: `{summary['authoritative_memory_update_count']}`",
        f"- pending memory updates: `{summary['pending_memory_update_count']}`",
        f"- physical API requests: `{summary['api_usage']['physical_requests']}`", "",
        "## Scenes", "",
        "| Scene | Turns | Tokens | Claims | Confirmed | Rejected | Uncertain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scenes"]:
        lines.append(
            f"| `{row['scene_key']}` | {row['turn_count']} | {row['token_count']} | "
            f"{row['claimed_delta_count']} | {row['confirmed_count']} | "
            f"{row['rejected_count']} | {row['uncertain_count']} |")
    lines.extend([
        "", "## Provenance", "",
        f"- parent SHA unchanged: `{str(summary['parent_sha_unchanged']).lower()}`",
        f"- Git HEAD before/after: `{summary['git_head_before']}` / "
        f"`{summary['git_head_after']}`",
        f"- project diff SHA before/after: `{summary['project_diff_sha256_before']}` / "
        f"`{summary['project_diff_sha256_after']}`",
        f"- old historical runs unchanged: "
        f"`{str(summary['historical_runs_unchanged']).lower()}`",
        f"- legacy semantic path called: "
        f"`{str(summary['legacy_semantic_path_called']).lower()}`",
        f"- cache / imported / legacy reuse: "
        f"`{summary['api_usage']['cache_reuse_count']} / "
        f"{summary['api_usage']['imported_response_count']} / "
        f"{summary['api_usage']['legacy_artifact_reuse_count']}`", "",
        "Questioner, Distractor, Solver, QA execution, Difficulty Strategy execution, "
        "legacy Verifier, and Training were not invoked.", "",
    ])
    return "\n".join(lines)


def _run_version_and_summary_stem(run_root: Path) -> tuple[str, str]:
    """Derive report naming from the actual isolated run root."""
    match = re.search(r"_fresh_e2e_(v[0-9]+)$", run_root.name)
    run_version = match.group(1) if match else "v6"
    return run_version, f"slim_fresh_e2e_{run_version}_summary"


def run(*, input_path: Path, run_root: Path, api_config_path: Path,
        tokenizer_path: Path = DEFAULT_TOKENIZER) -> dict[str, Any]:
    input_path = input_path.resolve()
    run_root = run_root.resolve()
    api_config_path = api_config_path.resolve()
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_exists:{run_root}")
    if run_root.parent != HISTORY_ROOT.resolve():
        raise ValueError("run_root_must_be_direct_child_of_restart_n5_runs")
    run_version, summary_stem = _run_version_and_summary_stem(run_root)
    input_sha_before = sha256_file(input_path)
    root = json.loads(input_path.read_text(encoding="utf-8"))
    root_validation = validate_unified_dialogue_root(root)
    if not root_validation["valid"]:
        raise ValueError("authoritative_root_invalid:" + ",".join(root_validation["errors"]))
    if root.get("root_id") != "cc-train-episode-61555":
        raise ValueError("unexpected_authoritative_root_identity")
    parent = read_unified_parent(input_path)
    parent_dialogue_sha = str(parent["dialogue"]["dialogue_hash"])
    git_head = git_value(PROJECT, "rev-parse", "HEAD")
    diff_sha = project_diff_sha256(PROJECT)
    history_before = _history_metadata_fingerprint(HISTORY_ROOT, run_root)
    if not HISTORICAL_V1_RUN.is_dir():
        raise FileNotFoundError(f"historical_v1_run_missing:{HISTORICAL_V1_RUN}")
    if not HISTORICAL_V2_RUN.is_dir():
        raise FileNotFoundError(f"historical_v2_run_missing:{HISTORICAL_V2_RUN}")
    if not HISTORICAL_V3_RUN.is_dir():
        raise FileNotFoundError(f"historical_v3_run_missing:{HISTORICAL_V3_RUN}")
    historical_v1_before = hash_tree(HISTORICAL_V1_RUN)
    historical_v2_before = hash_tree(HISTORICAL_V2_RUN)
    historical_v3_before = hash_tree(HISTORICAL_V3_RUN)
    limits = RunLimits()
    api = load_challenger_config(api_config_path)
    source_manifest = _source_manifest()
    package_isolation = audit_legacy_import_isolation(PACKAGE_ROOT)
    if not package_isolation["isolated"]:
        raise RuntimeError("new_package_imports_banned_legacy_module")

    run_root.mkdir(parents=True)
    manifest = {
        "schema_version": "world_memory_slim_fresh_run_manifest_v6",
        "runner_version": LIVE_FRESH_E2E_RUNNER_VERSION,
        "run_id": run_root.name, "root_id": root["root_id"], "n": 1,
        "created_at_utc": _utc_now(), "fresh": True, "no_resume": True,
        "cache_reuse_allowed": False, "imported_response_allowed": False,
        "legacy_artifact_reuse_allowed": False,
        "input_path": str(input_path), "input_sha256": input_sha_before,
        "parent_dialogue_sha256": parent_dialogue_sha,
        "parent_session_count": root["dialogue"]["session_count"],
        "parent_turn_count": root["dialogue"]["turn_count"],
        "git_head": git_head, "project_diff_sha256": diff_sha,
        "schema_sha256": schema_sha256s(),
        "component_source_sha256": {
            "world_memory": sha256_file(PACKAGE_ROOT / "memory.py"),
            "delta_affordance_compiler": sha256_file(
                PACKAGE_ROOT / "affordance_compiler.py"),
            "planner_and_role_schemas": sha256_file(PACKAGE_ROOT / "schemas.py"),
            "structural_validator": sha256_file(PACKAGE_ROOT / "structural.py"),
            "local_binder": sha256_file(PACKAGE_ROOT / "binder.py"),
            "prompt_builders": sha256_file(PACKAGE_ROOT / "prompts.py"),
            "live_runner": sha256_file(Path(__file__).resolve()),
        },
        "versions": {
            "world_memory": WORLD_MEMORY_SCHEMA_VERSION,
            "delta_affordance": AFFORDANCE_SCHEMA_VERSION,
            "planner": PLANNER_SCHEMA_VERSION,
            "realizer": REALIZER_SCHEMA_VERSION,
            "evidence_committer": EVIDENCE_COMMITTER_SCHEMA_VERSION,
            "structural_validator": STRUCTURAL_DELTA_VALIDATOR_VERSION,
            "local_binder": LOCAL_STRUCTURAL_BINDER_VERSION,
            "qa_replay_schema_only": QA_REPLAY_SCHEMA_VERSION,
            "difficulty_strategy_schema_only": DIFFICULTY_STRATEGY_SCHEMA_VERSION,
            "planner_mutation_schema_only": PLANNER_MUTATION_SCHEMA_VERSION,
        },
        "prompt_versions": {
            "planner": SLIM_PLANNER_PROMPT_VERSION,
            "planner_retry": SLIM_PLANNER_RETRY_PROMPT_VERSION,
            "realizer": NATURAL_REALIZER_PROMPT_VERSION,
            "realizer_claim_only_retry":
                NATURAL_REALIZER_CLAIM_ONLY_RETRY_PROMPT_VERSION,
            "evidence_committer": EVIDENCE_COMMITTER_PROMPT_VERSION,
            "evidence_committer_retry": EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION,
            "prompt_source_sha256": sha256_file(PACKAGE_ROOT / "prompts.py"),
        },
        "run_limits": limits.__dict__,
        "model_config": {
            "model": api.model, "base_url": api.base_url,
            "temperature": api.temperature, "timeout_seconds": api.timeout_seconds,
            "enable_thinking": api.enable_thinking,
            "api_key_recorded": False,
            "api_config_sha256": sha256_file(api_config_path),
        },
        "source_manifest": source_manifest,
        "historical_runs_before": history_before,
        "historical_v1_content_sha_before": historical_v1_before,
        "historical_v2_content_sha_before": historical_v2_before,
        "historical_v3_content_sha_before": historical_v3_before,
        "forbidden_execution_counts": {
            "questioner": 0, "distractor": 0, "solver": 0,
            "qa_replay": 0, "difficulty_strategy": 0,
            "training": 0, "legacy_verifier": 0,
        },
    }
    manifest_path = run_root / "config" / "run_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(run_root / "provenance" / "source_file_manifest.json", source_manifest)
    atomic_json(run_root / "provenance" / "historical_runs_before.json", history_before)
    atomic_json(run_root / "provenance" / "historical_v1_content_before.json",
                historical_v1_before)
    atomic_json(run_root / "provenance" / "historical_v2_content_before.json",
                historical_v2_before)
    atomic_json(run_root / "provenance" / "historical_v3_content_before.json",
                historical_v3_before)
    atomic_json(run_root / "frozen" / "authoritative_parent.json", parent)
    atomic_json(run_root / "frozen" / "qa_replay_pool_empty.json", {
        "schema_version": QA_REPLAY_SCHEMA_VERSION, "items": [], "execution_count": 0})
    atomic_json(run_root / "frozen" / "difficulty_strategy_memory_empty.json", {
        "schema_version": DIFFICULTY_STRATEGY_SCHEMA_VERSION,
        "items": [], "execution_count": 0})
    atomic_json(run_root / "frozen" / "planner_mutation_queue_empty.json", {
        "schema_version": PLANNER_MUTATION_SCHEMA_VERSION,
        "items": [], "execution_count": 0})

    memory = initialize_world_memory(root, parent)
    initial_memory = json.loads(canonical_json(memory))
    atomic_json(run_root / "world_memory_initial.json", initial_memory)
    action_menu = compile_delta_affordances(
        memory, root_identity=str(root["root_id"]), mutation_requests=[])
    atomic_json(run_root / "planner" / "delta_affordance_menu.json", action_menu)
    atomic_json(run_root / "affordance_menu.json", action_menu)
    client = OpenAICompatibleChallengerClient(api)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    ledger = RequestLedger(run_root, sha256_file(manifest_path), "live")
    caller = LiveRoleCaller(
        run_root=run_root, ledger=ledger, client=client, tokenizer=tokenizer,
        root_id=root["root_id"], limits=limits)

    planner_report: dict[str, Any] = {}
    planned_result: dict[str, Any] | None = None
    planner_model_output: dict[str, Any] | None = None
    sessions: list[dict[str, Any]] = []
    scene_stats: list[dict[str, Any]] = []
    all_committer_outputs: list[dict[str, Any]] = []
    global_handoff_index = 1
    global_thread_index = 1
    failure_stage: str | None = None
    failure_message: str | None = None
    try:
        if not action_menu["actions"]:
            raise StageFailure("Affordance compilation", "action_menu_empty")
        relevant_turn_ids = set(action_menu_parent_turn_ids(action_menu))
        planner_parent_turns = [
            turn for turn in parent["dialogue"]["turns"]
            if turn.get("turn_id") in relevant_turn_ids
        ]
        planner_base = build_slim_planner_prompt(
            parent_turns=planner_parent_turns,
            action_menu=action_menu, role_table=memory["entities"])
        prompt = planner_base
        for attempt in range(1, limits.planner_attempts + 1):
            parsed, raw, parse_errors = caller.call_json(
                request_id=f"planner-a{attempt:02d}", role="planner", attempt=attempt,
                prompt=prompt,
                prompt_version=(SLIM_PLANNER_PROMPT_VERSION if attempt == 1 else
                                SLIM_PLANNER_RETRY_PROMPT_VERSION),
                max_tokens=limits.planner_max_tokens)
            result = (_planner_result(parsed, memory, root_id=root["root_id"],
                                      limits=limits, action_menu=action_menu)
                      if parsed is not None else
                      {"accepted": False, "errors": parse_errors})
            atomic_json(run_root / "planner" / f"validation_attempt_{attempt:02d}.json",
                        result)
            planner_report = result
            if result["accepted"]:
                planned_result = result
                planner_model_output = dict(parsed)
                break
            prompt = build_slim_planner_retry_prompt(
                action_menu=action_menu, role_table=memory["entities"],
                errors=result["errors"])
        if planned_result is None:
            raise StageFailure("Planner schema/selection",
                               "planner_retry_limit_exhausted")
        atomic_json(run_root / "planner_output.json", planner_model_output)
        atomic_json(run_root / "planner" / "slim_plan.json",
                    planned_result["normalized_plan"])
        compiled_delta_payload = {
            "root_id": root["root_id"],
            "planner_schema_version": PLANNER_SCHEMA_VERSION,
            "affordance_schema_version": AFFORDANCE_SCHEMA_VERSION,
            "action_menu_sha256": planned_result["normalized_plan"][
                "action_menu_sha256"],
            "scenes": [{
                "scene_key": scene["scene_key"],
                "selected_action_ids": scene["selected_action_ids"],
                "proposed_graph_deltas": scene["proposed_graph_deltas"],
            } for scene in planned_result["normalized_plan"]["scene_goals"]],
            "structural_validation": planned_result["structural_validation"],
        }
        atomic_json(run_root / "compiled_deltas.json", compiled_delta_payload)
        atomic_json(run_root / "planner" / "structural_delta_validation.json", {
            "validation": planned_result["structural_validation"],
            "executable_scene_keys": [row["scene_key"] for row in
                                      planned_result["executable_scenes"]],
            "rejected_scene_keys": planned_result["rejected_scene_keys"],
        })

        prior_scenes: list[dict[str, Any]] = []
        known_entities = {str(entity["display_name"]) for entity in memory["entities"]}
        for scene_position, planned_scene in enumerate(
                planned_result["executable_scenes"], 1):
            scene_key = str(planned_scene["scene_key"])
            accepted_deltas = list(planned_scene["accepted_deltas"])
            scene_view = {key: planned_scene[key] for key in (
                "scene_key", "goal", "participants", "related_memory_ids")}
            related_turns = _related_parent_turns(
                parent_turns=parent["dialogue"]["turns"], memory=memory,
                scene=scene_view, selected_memory_ids=[])
            realizer_base = build_natural_realizer_prompt(
                relevant_parent_turns=related_turns,
                memory_subset=_scene_memory_subset(memory, scene_view),
                scene_goal=scene_view, accepted_deltas=accepted_deltas,
                prior_scenes=prior_scenes)
            prompt = realizer_base
            retry_mode = "full_scene_retry"
            frozen_scene_key: str | None = None
            frozen_dialogue_turns: list[dict[str, Any]] | None = None
            realizer_output: dict[str, Any] | None = None
            realizer_validation: dict[str, Any] | None = None
            last_realizer_result: dict[str, Any] | None = None
            for attempt in range(1, limits.realizer_attempts_per_scene + 1):
                parsed, raw, parse_errors = caller.call_json(
                    request_id=f"realizer-{scene_key}-a{attempt:02d}",
                    role="realizer", attempt=attempt, prompt=prompt,
                    prompt_version=(
                        NATURAL_REALIZER_CLAIM_ONLY_RETRY_PROMPT_VERSION
                        if retry_mode == "claim_only_retry" else
                        NATURAL_REALIZER_PROMPT_VERSION),
                    max_tokens=limits.realizer_max_tokens)
                if retry_mode == "claim_only_retry":
                    if frozen_scene_key is None or frozen_dialogue_turns is None:
                        raise RuntimeError("claim_only_retry_missing_frozen_dialogue")
                    if parsed is None:
                        result = {
                            "valid": False, "errors": parse_errors,
                            "dialogue_valid": True, "claims_valid": False,
                            "dialogue_errors": [], "claim_errors": parse_errors,
                            "retry_mode": "claim_only_retry",
                            "frozen_dialogue_unchanged": True,
                        }
                    else:
                        claim_validation = validate_claim_only_retry_contract(
                            parsed, frozen_scene_key=frozen_scene_key,
                            frozen_dialogue_turns=frozen_dialogue_turns,
                            accepted_deltas=accepted_deltas)
                        if claim_validation["valid"]:
                            reconstructed = reconstruct_scene_from_claim_only(
                                frozen_scene_key=frozen_scene_key,
                                frozen_dialogue_turns=frozen_dialogue_turns,
                                claim_only_output=parsed)
                            frozen_unchanged = (
                                canonical_json(reconstructed["dialogue_turns"]) ==
                                canonical_json(frozen_dialogue_turns))
                            if not frozen_unchanged:
                                raise RuntimeError(
                                    "claim_only_retry_mutated_frozen_dialogue")
                            result = _realizer_result(
                                reconstructed, scene=scene_view,
                                accepted_deltas=accepted_deltas)
                            result["claim_only_validation"] = claim_validation
                            result["frozen_dialogue_unchanged"] = frozen_unchanged
                        else:
                            result = {
                                "valid": False,
                                "errors": claim_validation["errors"],
                                "dialogue_valid": True,
                                "claims_valid": False,
                                "dialogue_errors": [],
                                "claim_errors": claim_validation["errors"],
                                "retry_mode": "claim_only_retry",
                                "claim_only_validation": claim_validation,
                                "frozen_dialogue_unchanged": True,
                            }
                else:
                    result = (_realizer_result(
                        parsed, scene=scene_view, accepted_deltas=accepted_deltas)
                        if parsed is not None else {
                            "valid": False, "errors": parse_errors,
                            "dialogue_valid": False, "claims_valid": False,
                            "dialogue_errors": parse_errors,
                            "claim_errors": [],
                            "retry_mode": "full_scene_retry",
                        })
                last_realizer_result = result
                atomic_json(run_root / "scenes" / scene_key /
                            f"realizer_validation_attempt_{attempt:02d}.json", result)
                if result["valid"]:
                    realizer_output = result["normalized_output"]
                    realizer_validation = result
                    break
                if (result.get("retry_mode") == "claim_only_retry" and
                        result.get("dialogue_valid")):
                    if frozen_dialogue_turns is None:
                        normalized = result.get("normalized_output") or {}
                        frozen_scene_key = str(normalized.get("scene_key") or "")
                        frozen_dialogue_turns = copy.deepcopy(
                            list(normalized.get("dialogue_turns") or []))
                    if frozen_scene_key != scene_key or not frozen_dialogue_turns:
                        raise RuntimeError("claim_only_retry_invalid_frozen_dialogue")
                    retry_mode = "claim_only_retry"
                    prompt = build_realizer_claim_only_retry_prompt(
                        scene_key=frozen_scene_key,
                        frozen_dialogue_turns=frozen_dialogue_turns,
                        proposed_delta_ids=[str(delta["proposed_delta_id"])
                                            for delta in accepted_deltas],
                        errors=result["errors"])
                else:
                    retry_mode = "full_scene_retry"
                    prompt = _retry_prompt(
                        realizer_base, role=f"Natural Realizer {scene_key}",
                        errors=result["errors"],
                        previous=parsed if parsed is not None else raw)
            if realizer_output is None:
                failure_stage = (
                    "Natural Realizer claim/reference"
                    if (last_realizer_result or {}).get("dialogue_valid") and
                    not (last_realizer_result or {}).get("claims_valid")
                    else "Natural Realizer structure")
                raise StageFailure(failure_stage,
                                   f"{scene_key}:retry_limit_exhausted")
            atomic_json(run_root / "scenes" / scene_key / "realizer_output.json",
                        realizer_output)
            atomic_json(run_root / "scenes" / scene_key / "realizer_validation.json",
                        realizer_validation)
            staged = stage_model_claims(memory, accepted_deltas, realizer_output)
            emergent_proposals = compile_emergent_pending_proposals(
                root_id=root["root_id"], scene=scene_view,
                realizer_output=realizer_output, memory=memory)
            staged = stage_emergent_claims(staged, emergent_proposals)
            if staged["current_states"] != memory["current_states"]:
                raise StageFailure("Memory commit", "model_claim_polluted_authoritative_state")

            decision_outputs: list[dict[str, Any]] = []
            claims = {str(claim["proposed_delta_id"]): claim for claim in
                      realizer_output["claimed_graph_updates"]}
            for delta in accepted_deltas:
                delta_id = str(delta["proposed_delta_id"])
                claim = claims[delta_id]
                candidate_keys = set(str(key) for key in
                                     claim["candidate_evidence_turn_keys"])
                committer_base = build_evidence_committer_prompt(
                    proposed_delta=delta,
                    dialogue_turns=realizer_output["dialogue_turns"],
                    realizer_claims=[claim],
                    memory_view=_memory_prompt_view(memory, include_history=False))
                prompt = committer_base
                accepted_decision: dict[str, Any] | None = None
                for attempt in range(1, limits.committer_attempts_per_claim + 1):
                    safe_delta = delta_id.rsplit(":", 1)[-1]
                    parsed, raw, parse_errors = caller.call_json(
                        request_id=(f"committer-{scene_key}-{safe_delta}-a{attempt:02d}"),
                        role="evidence_committer", attempt=attempt, prompt=prompt,
                        prompt_version=(EVIDENCE_COMMITTER_PROMPT_VERSION
                                        if attempt == 1 else
                                        EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION),
                        max_tokens=limits.committer_max_tokens)
                    result = (_committer_result(
                        parsed, delta_id=delta_id,
                        candidate_evidence_turn_keys=candidate_keys,
                        realizer_claimed_status=str(claim["claimed_status"]))
                        if parsed is not None else {"valid": False,
                                                    "errors": parse_errors})
                    atomic_json(run_root / "scenes" / scene_key / "committer" /
                                f"{safe_delta}_validation_attempt_{attempt:02d}.json",
                                result)
                    if result["valid"]:
                        accepted_decision = parsed
                        break
                    prompt = build_evidence_committer_retry_prompt(
                        frozen_base_prompt=committer_base,
                        errors=result["errors"])
                if accepted_decision is None:
                    raise StageFailure("Evidence Committer schema/reference",
                                       f"{delta_id}:retry_limit_exhausted")
                decision_outputs.append(accepted_decision)
                all_committer_outputs.append(accepted_decision)
            emergent_decision_outputs: list[dict[str, Any]] = []
            emergent_routes = build_emergent_committer_routes(
                proposals=emergent_proposals, realizer_output=realizer_output,
                memory_view=_memory_prompt_view(memory, include_history=False))
            for route in emergent_routes:
                proposal_id = str(route["proposal_id"])
                safe_proposal = proposal_id.rsplit(":", 1)[-1]
                committer_base = str(route["prompt"])
                prompt = committer_base
                accepted_decision = None
                for attempt in range(1, limits.committer_attempts_per_claim + 1):
                    parsed, raw, parse_errors = caller.call_json(
                        request_id=(f"committer-{scene_key}-emergent-"
                                    f"{safe_proposal}-a{attempt:02d}"),
                        role="evidence_committer", attempt=attempt, prompt=prompt,
                        prompt_version=(EVIDENCE_COMMITTER_PROMPT_VERSION
                                        if attempt == 1 else
                                        EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION),
                        max_tokens=limits.committer_max_tokens)
                    result = (_committer_result(
                        parsed, delta_id=proposal_id,
                        candidate_evidence_turn_keys=set(
                            str(key) for key in
                            route["candidate_evidence_turn_keys"]))
                        if parsed is not None else {"valid": False,
                                                    "errors": parse_errors})
                    atomic_json(run_root / "scenes" / scene_key / "committer" /
                                f"emergent_{safe_proposal}_validation_attempt_"
                                f"{attempt:02d}.json", result)
                    if result["valid"]:
                        accepted_decision = parsed
                        break
                    prompt = build_evidence_committer_retry_prompt(
                        frozen_base_prompt=committer_base,
                        errors=result["errors"])
                if accepted_decision is None:
                    raise StageFailure(
                        "Evidence Committer schema/reference",
                        f"{proposal_id}:retry_limit_exhausted")
                emergent_decision_outputs.append(accepted_decision)
                all_committer_outputs.append(accepted_decision)
            atomic_json(run_root / "scenes" / scene_key /
                        "evidence_committer_results.json", {
                            "planned": decision_outputs,
                            "emergent": emergent_decision_outputs,
                        })
            try:
                binding = bind_scene(
                    root_id=root["root_id"], round_index=1, scene_goal=scene_view,
                    realizer_output=realizer_output, deltas=accepted_deltas,
                    committer_outputs=decision_outputs,
                    handoff_start_index=global_handoff_index,
                    emergent_proposals=emergent_proposals,
                    emergent_committer_outputs=emergent_decision_outputs,
                    thread_start_index=global_thread_index)
            except Exception as error:
                raise StageFailure("Binder/reference", f"{scene_key}:{error}") from error
            global_handoff_index += len(binding["allocated_handoffs"])
            global_thread_index += len(binding["allocated_threads"])
            atomic_json(run_root / "scenes" / scene_key / "binding.json", binding)
            for delta, decision in zip(accepted_deltas, decision_outputs):
                try:
                    memory = commit_confirmed_delta(
                        memory, delta, decision, binding, scene_key=scene_key,
                        round_index=1, dry_run=False)
                except Exception as error:
                    raise StageFailure("Memory commit", f"{scene_key}:{error}") from error
            for proposal, decision in zip(
                    emergent_proposals, emergent_decision_outputs):
                try:
                    memory = commit_emergent_thread(
                        memory, proposal, decision, binding,
                        round_index=1, dry_run=False)
                except Exception as error:
                    raise StageFailure("Memory commit", f"{scene_key}:{error}") from error
            atomic_json(run_root / "scenes" / scene_key / "world_memory_after.json",
                        memory)
            session = {
                "scene_key": scene_key, "scene_position": scene_position,
                "session_id": binding["session_id"],
                "turn_ids": [turn["turn_id"] for turn in binding["dialogue_turns"]],
                "dialogue_turns": binding["dialogue_turns"],
                "evidence_bindings": binding["evidence_bindings"],
                "allocated_handoffs": binding["allocated_handoffs"],
                "allocated_threads": binding["allocated_threads"],
            }
            sessions.append(session)
            prior_scenes.append({
                "scene_key": scene_key,
                "dialogue_turns": realizer_output["dialogue_turns"],
            })
            stats = {"scene_key": scene_key, **_quality_stats(
                realizer_output, tokenizer=tokenizer,
                committer_outputs=(decision_outputs + emergent_decision_outputs),
                known_entities=known_entities)}
            stats["action_ids"] = list(planned_scene["selected_action_ids"])
            scene_stats.append(stats)
            atomic_json(run_root / "scenes" / scene_key / "quality_stats.json", stats)
    except StageFailure as error:
        failure_stage = error.stage
        failure_message = str(error)
    except Exception as error:
        failure_stage = "unhandled_system_error"
        failure_message = f"{type(error).__name__}:{error}"

    child = _child_payload(str(root["root_id"]), sessions)
    atomic_json(run_root / "child_dialogue.json", child)
    atomic_json(run_root / "world_memory_final.json", memory)
    atomic_json(run_root / "pending_memory_updates.json", {
        "root_id": root["root_id"],
        "pending_memory_updates": memory["pending_memory_updates"],
    })
    atomic_text(run_root / "memory_update_log.jsonl", _memory_log_jsonl(memory))
    api_usage = _aggregate_request_ledger(run_root, ledger, caller.records)
    input_sha_after = sha256_file(input_path)
    history_after = _history_metadata_fingerprint(HISTORY_ROOT, run_root)
    historical_v1_after = hash_tree(HISTORICAL_V1_RUN)
    historical_v2_after = hash_tree(HISTORICAL_V2_RUN)
    historical_v3_after = hash_tree(HISTORICAL_V3_RUN)
    git_head_after = git_value(PROJECT, "rev-parse", "HEAD")
    diff_sha_after = project_diff_sha256(PROJECT)
    import_graph = _import_graph()
    atomic_json(run_root / "reports" / "import_graph.json", import_graph)
    atomic_json(run_root / "provenance" / "historical_runs_after.json", history_after)
    atomic_json(run_root / "provenance" / "historical_v1_content_after.json",
                historical_v1_after)
    atomic_json(run_root / "provenance" / "historical_v2_content_after.json",
                historical_v2_after)
    atomic_json(run_root / "provenance" / "historical_v3_content_after.json",
                historical_v3_after)
    source_after = _source_manifest()
    source_unchanged = source_manifest == source_after
    current_turns = [turn for session in sessions for turn in session["dialogue_turns"]]
    turn_ids = [turn["turn_id"] for turn in current_turns]
    session_ids = [session["session_id"] for session in sessions]
    evidence_ids = [turn_id for session in sessions
                    for row in session["evidence_bindings"]
                    for turn_id in row["evidence_turn_ids"]]
    confirmed = sum(row["decision"] == "confirmed" for row in all_committer_outputs)
    rejected = sum(row["decision"] == "rejected" for row in all_committer_outputs)
    uncertain = sum(row["decision"] == "uncertain" for row in all_committer_outputs)
    proposed = int((planned_result or {}).get("proposed_delta_count", 0))
    accepted = int((planned_result or {}).get("accepted_delta_count", 0))
    structurally_rejected = int((planned_result or {}).get("rejected_delta_count", 0))
    authoritative_updates = sum(
        row.get("authoritative_applied") is True
        for row in memory.get("memory_update_log") or [])
    generation_checks = {
        "at_least_two_scenes": len(sessions) >= 2,
        "all_planned_executable_scenes_generated": (
            planned_result is not None and len(sessions) ==
            len(planned_result["executable_scenes"])),
        "turn_ids_unique": len(turn_ids) == len(set(turn_ids)),
        "session_ids_unique": len(session_ids) == len(set(session_ids)),
        "turns_in_program_order": all(
            turn["turn_index"] == index
            for session in sessions for index, turn in
            enumerate(session["dialogue_turns"], 1)),
        "all_evidence_final_turn_ids_exist": set(evidence_ids) <= set(turn_ids),
        "no_new_unconnected_character": all(
            row["new_character_connection_valid"] for row in scene_stats),
        "no_system_failure": failure_stage is None,
    }
    generation_ready = all(generation_checks.values())
    memory_validation = validate_world_memory(memory)
    authority_checks = {
        "at_least_one_confirmed_delta": confirmed >= 1,
        "at_least_one_authoritative_memory_update": authoritative_updates >= 1,
        "authoritative_states_verified_only": all(
            row["semantic_status"] == "semantically_verified"
            for row in memory["current_states"]),
        "world_memory_valid": memory_validation["valid"],
        "model_claims_not_committed_directly": all(
            row["semantic_status"] != "model_claimed"
            for row in memory["current_states"]),
    }
    provenance_checks = {
        "input_sha_unchanged": input_sha_before == input_sha_after,
        "parent_dialogue_sha_unchanged": parent_dialogue_sha ==
            root["dialogue"]["dialogue_hash"],
        "git_head_unchanged": git_head_after == git_head,
        "project_diff_sha_unchanged": diff_sha_after == diff_sha,
        "source_files_unchanged": source_unchanged,
        "historical_runs_excluding_current_unchanged": history_before == history_after,
        "historical_v1_content_unchanged": historical_v1_before == historical_v1_after,
        "historical_v2_content_unchanged": historical_v2_before == historical_v2_after,
        "historical_v3_content_unchanged": historical_v3_before == historical_v3_after,
        "legacy_import_isolation": import_graph["isolated"],
        "fresh_no_resume": True,
        "cache_reuse_zero": api_usage["cache_reuse_count"] == 0,
        "imported_response_zero": api_usage["imported_response_count"] == 0,
        "legacy_artifact_reuse_zero": api_usage["legacy_artifact_reuse_count"] == 0,
    }
    fresh_ready = (generation_ready and all(authority_checks.values()) and
                   all(provenance_checks.values()))
    post_run = {
        "schema_version": POST_RUN_PROVENANCE_VERSION,
        "run_manifest_sha256": sha256_file(manifest_path),
        "checks": provenance_checks,
        "authoritative_release_provenance": all(provenance_checks.values()),
        "provenance_mixed": not all(provenance_checks.values()),
        "input_sha256_before": input_sha_before,
        "input_sha256_after": input_sha_after,
        "parent_dialogue_sha256_before": parent_dialogue_sha,
        "parent_dialogue_sha256_after": root["dialogue"]["dialogue_hash"],
        "git_head_before": git_head, "git_head_after": git_head_after,
        "project_diff_sha256_before": diff_sha,
        "project_diff_sha256_after": diff_sha_after,
        "historical_runs_before": history_before,
        "historical_runs_after": history_after,
        "historical_v1_content_before": historical_v1_before,
        "historical_v1_content_after": historical_v1_after,
        "historical_v2_content_before": historical_v2_before,
        "historical_v2_content_after": historical_v2_after,
        "historical_v3_content_before": historical_v3_before,
        "historical_v3_content_after": historical_v3_after,
        "source_manifest_before": source_manifest,
        "source_manifest_after": source_after,
        "import_graph": import_graph,
        "api_usage": api_usage,
    }
    atomic_json(run_root / "reports" / "post_run_provenance.json", post_run)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "runner_version": LIVE_FRESH_E2E_RUNNER_VERSION,
        "run_id": run_root.name, "run_version": run_version,
        "root_id": root["root_id"],
        "completed_at_utc": _utc_now(),
        "failure_stage": failure_stage, "failure_message": failure_message,
        "scene_count": len(sessions), "scenes": scene_stats,
        "child_session_count": len(sessions),
        "child_turn_count": len(current_turns),
        "affordance_menu": {
            "action_count": len(action_menu["actions"]),
            "operation_class_distribution": dict(collections.Counter(
                row["operation_class"] for row in action_menu["actions"])),
            "sha256": sha256_bytes(canonical_json(action_menu).encode()),
        },
        "planner": {
            "attempt_count": sum(row["role"] == "planner" for row in caller.records),
            "retry_count": sum(row["role"] == "planner" and row["attempt"] > 1
                               for row in caller.records),
            "final_scene_count": len((planned_result or {}).get(
                "normalized_plan", {}).get("scene_goals", [])),
            "selected_action_ids": [
                action_id
                for scene in (planned_result or {}).get(
                    "normalized_plan", {}).get("scene_goals", [])
                for action_id in scene.get("selected_action_ids") or []],
        },
        "realizer": {
            "scene_count": len(sessions),
            "request_count": sum(row["role"] == "realizer" for row in caller.records),
            "retry_count": sum(row["role"] == "realizer" and row["attempt"] > 1
                               for row in caller.records),
        },
        "evidence_committer": {
            "request_count": sum(row["role"] == "evidence_committer"
                                 for row in caller.records),
            "retry_count": sum(row["role"] == "evidence_committer"
                               and row["attempt"] > 1 for row in caller.records),
            "confirmed": confirmed, "rejected": rejected, "uncertain": uncertain,
        },
        "delta_counts": {
            "proposed": proposed, "accepted": accepted,
            "structurally_rejected": structurally_rejected,
            "confirmed": confirmed, "rejected": rejected, "uncertain": uncertain,
        },
        "authoritative_memory_update_count": authoritative_updates,
        "pending_memory_update_count": len(memory["pending_memory_updates"]),
        "memory_update_log_count": len(memory["memory_update_log"]),
        "initial_memory_statistics": {key: len(initial_memory[key]) for key in (
            "entities", "relations", "event_history", "current_states",
            "knowledge_visibility", "open_threads", "completed_threads")},
        "final_memory_statistics": {key: len(memory[key]) for key in (
            "entities", "relations", "event_history", "current_states",
            "knowledge_visibility", "open_threads", "completed_threads")},
        "generation_checks": generation_checks,
        "authority_checks": authority_checks,
        "provenance_checks": provenance_checks,
        "api_usage": api_usage,
        "input_sha256_before": input_sha_before,
        "input_sha256_after": input_sha_after,
        "parent_dialogue_sha256_before": parent_dialogue_sha,
        "parent_dialogue_sha256_after": root["dialogue"]["dialogue_hash"],
        "parent_sha_unchanged": parent_dialogue_sha == root["dialogue"]["dialogue_hash"],
        "git_head_before": git_head, "git_head_after": git_head_after,
        "project_diff_sha256_before": diff_sha,
        "project_diff_sha256_after": diff_sha_after,
        "historical_runs_unchanged": history_before == history_after,
        "historical_v1_content_sha_before": historical_v1_before,
        "historical_v1_content_sha_after": historical_v1_after,
        "historical_v2_content_sha_before": historical_v2_before,
        "historical_v2_content_sha_after": historical_v2_after,
        "historical_v3_content_sha_before": historical_v3_before,
        "historical_v3_content_sha_after": historical_v3_after,
        "legacy_semantic_path_called": not import_graph["isolated"],
        "forbidden_execution_counts": manifest["forbidden_execution_counts"],
        "delta_affordance_v1_ready": bool(action_menu["actions"]),
        "slim_planner_v2_ready": planned_result is not None,
        "compiled_delta_ready": planned_result is not None and accepted >= 1,
        "generation_ready": generation_ready,
        "fresh_e2e_ready": fresh_ready,
        "qa_pipeline_ready": False,
        "training_ready": False,
    }
    atomic_json(run_root / "reports" / f"{summary_stem}.json", summary)
    atomic_text(run_root / "reports" / f"{summary_stem}.md",
                _render_summary(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args(argv)
    summary = run(input_path=args.input, run_root=args.run_root,
                  api_config_path=args.api_config, tokenizer_path=args.tokenizer)
    print(json.dumps({key: summary[key] for key in (
        "run_id", "root_id", "failure_stage", "scene_count",
        "delta_counts", "generation_ready", "fresh_e2e_ready")},
        ensure_ascii=False, indent=2))
    return 1 if summary["failure_stage"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
