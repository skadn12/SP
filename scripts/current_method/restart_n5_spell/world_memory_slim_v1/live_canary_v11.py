"""Single-use live runner for Slim Continuous Expansion v11 Phase B2.

The entry point is intentionally independent of the v7/v10 orchestration.  It
allows only Planner v11, Realizer v11, and one round-wide semantic Validator.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..unified_dialogue_root import read_unified_parent
from .action_availability_v11 import (
    compile_action_availability_view, validate_action_availability_view,
)
from .append_only_memory import apply_scene_commit
from .complete_dialogue_view import compose_complete_dialogue_view, validate_complete_dialogue_view
from .context_compiler_v2 import ContextInvariantFailure, compile_scene_context_v2
from .continuous_round_plan_v1 import (
    build_scene_dependency_dag,
)
from .controlled_expansion import compile_action_menu_v3, validate_action_menu_v3
from .durable_update_compiler import compile_durable_updates
from .expanded_dialogue_assembly import (
    assemble_expanded_dialogue, complete_dialogue_manifest,
    expanded_dialogue_markdown, validate_expanded_dialogue,
)
from .growth_admission import audit_authoritative_growth
from .growth_profile_v11 import (
    evaluate_growth_profile_admission, validate_growth_profile_definition,
    validate_planner_growth_profile_v11,
)
from .interaction_links import compile_source_interaction_links
from .live_contracts_v11 import (
    PLANNER_PROMPT_VERSION_V11, PLANNER_RETRY_PROMPT_VERSION_V11,
    REALIZER_PROMPT_VERSION_V11, REALIZER_RETRY_PROMPT_VERSION_V11,
    build_planner_prompt_v11, build_planner_retry_prompt_v11,
    build_realizer_prompt_v11, build_realizer_retry_prompt_v11,
    normalize_realizer_dialogue_v11,
)
from .live_preflight_b2_v11 import (
    DEFAULT_API_CONFIG, DEFAULT_INPUT, DEFAULT_RUN_ROOT, DEFAULT_TOKENIZER,
    PROJECT_ROOT, historical_manifests_v11, implementation_manifest_v11,
    run_live_preflight_b2_v11,
)
from .live_role_runtime_v11 import LiveJsonRoleCallerV11, aggregate_v11_usage
from .local_scene_gate_v11 import dependency_unavailable_result, validate_local_scene_v11
from .narrative_order import build_narrative_order_manifest
from .narrative_action_contracts import action_contract_registry
from .parent_anchor_compiler import compile_anchor_registry, validate_anchor_registry
from .placement_choice_registry import (
    compile_placement_choice_registry, compile_source_anchor_registry,
    validate_placement_choice_registry, validate_source_anchor_registry,
)
from .planning_state_view import (
    audit_interaction_relation_separation, compile_planning_state_view,
    validate_planning_state_view,
)
from .planner_outline_v11 import (
    compile_planner_outline_v11, validate_planner_outline_v11,
)
from .planner_action_repair_v11 import (
    PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11,
    apply_planner_action_repairs_v11,
    build_planner_action_only_retry_prompt_v11,
    compile_planner_action_repair_spec_v11,
    planner_action_only_retry_eligible,
)
from .planner_namespace_repair_v11 import (
    PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11,
    apply_planner_namespace_repairs_v11,
    build_planner_namespace_only_retry_prompt_v11,
    compile_planner_namespace_repair_spec_v11,
    planner_namespace_only_retry_eligible,
)
from .planner_speaker_repair_v11 import (
    PLANNER_SPEAKER_REPAIR_PROMPT_VERSION_V11,
    apply_planner_speaker_repairs_v11,
    build_planner_speaker_only_retry_prompt_v11,
    compile_planner_speaker_repair_spec_v11,
    planner_speaker_only_retry_eligible,
)
from .planner_transport_v11 import PLANNER_MAX_OUTPUT_TOKENS
from .round_validator_contract_repair_v11 import (
    ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11,
)
from .round_validator_transport_v11 import V11LiveRoleTransportClient
from .promotion_v2 import promote_scenes_v2
from .provenance import git_value, hash_tree, project_diff_sha256
from .review_world_view import build_review_world_view, validate_review_world_view
from .round_expansion_validator_v1 import (
    ROUND_VALIDATOR_MAX_OUTPUT_TOKENS, audit_round_validator_prompt_budget,
    build_round_validator_input, run_round_validator_with_retry,
)
from .round_validator_contract import (
    ROUND_VALIDATOR_PROMPT_VERSION, ROUND_VALIDATOR_RETRY_PROMPT_VERSION,
)
from .scene_decision_closure import compute_scene_decision_closure
from .source_memory_initializer_v1 import (
    authoritative_affordance_view, initialize_source_world_memory,
)
from .source_memory_initializer_v2 import initialize_source_world_memory_v2
from .structural_expansion_gate import validate_round_blueprint_structure
from .transactional_binder import (
    build_allocation_plan, commit_scene_transaction,
    validate_scene_local_evidence_refs,
)
from .turn_budget import compile_turn_budget


LIVE_CANARY_V11_VERSION = "slim_continuous_expansion_v11_live_canary_b2_v1"
RUN_ID = "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_1"
ROOT_ID = "cc-train-episode-61555"
ORCHESTRATION_SEED = 61555
PLANNER_BASE_MAX_ATTEMPTS = 3
PLANNER_TERMINAL_NAMESPACE_REPAIR_ATTEMPTS = 1
PLANNER_ABSOLUTE_MAX_ATTEMPTS = (
    PLANNER_BASE_MAX_ATTEMPTS + PLANNER_TERMINAL_NAMESPACE_REPAIR_ATTEMPTS
)
PARENT_TAIL_REFS = [f"{ROOT_ID}:P05:T{value:03d}" for value in range(10, 17)]
REALIZER_READ_TIMEOUT_SECONDS_V11 = 600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def live_api_configuration_v11(api_config_path: Path):
    """Freeze the non-streaming Realizer fallback to the v11 long-read budget."""
    return replace(
        load_challenger_config(api_config_path),
        timeout_seconds=REALIZER_READ_TIMEOUT_SECONDS_V11,
    )


def _sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else deepcopy(default)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                              for row in rows))


def _sessions(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_ref: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for turn in (parent.get("dialogue") or {}).get("turns") or []:
        ref = str(turn.get("session_id") or "")
        if ref not in by_ref:
            order.append(ref)
            by_ref[ref] = []
        by_ref[ref].append(dict(turn))
    return [{"session_id": ref, "turns": by_ref[ref]} for ref in order]


def _role_table(planning_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"entity_ref": row["entity_ref"], "display_name": row["display_name"],
             "entity_type": row.get("entity_type") or "person"}
            for row in planning_state.get("entities") or []]


def _round_turn_budget(scene: Mapping[str, Any], scene_count: int) -> dict[str, Any]:
    """Freeze a soft round target plus flexible per-scene safety bounds.

    The old allocation made the round-level 30--45 turn target a narrow hard
    limit for every scene. Keep the round target as planning guidance while
    allowing bounded slack for scenes with more speakers, beats, or actions.
    """
    base = compile_turn_budget(scene)
    preferred = {5: (6, 7, 9), 6: (5, 6, 7), 7: (5, 6, 6)}[scene_count]
    complexity_slack = min(2, len(base.get("adjustments") or []))
    hard_minimum = max(4, preferred[0] - 1)
    hard_maximum = min(12, max(
        preferred[2] + 2 + complexity_slack,
        min(int(base["maximum"]), 12),
    ))
    return {**base,
            "minimum": hard_minimum,
            "target": preferred[1],
            "maximum": hard_maximum,
            "preferred_minimum": preferred[0],
            "preferred_maximum": preferred[2],
            "round_turn_target_range": {"minimum": 30, "maximum": 45},
            "single_scene_limit_policy": "flexible_safety_bounds",
            "round_allocation_rule_version": "continuous_round_turn_allocation_v2_flexible",
            "base_turn_budget": base, "round_scene_count": scene_count}


def _freeze_guard(freeze: Mapping[str, Any]) -> dict[str, Any]:
    current_history = historical_manifests_v11()
    checks = {
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD") == freeze["git_head"],
        "implementation_manifest": implementation_manifest_v11()[
            "implementation_manifest_sha256"] == freeze["implementation_manifest"][
                "implementation_manifest_sha256"],
        "working_tree_manifest": project_diff_sha256(PROJECT_ROOT) == freeze[
            "project_working_tree_manifest_sha256"],
        "input_sha": sha256_file(Path(freeze["input_path"])) == freeze["input_sha256"],
        "api_config_sha": sha256_file(Path(freeze["api_config_path"])) == freeze[
            "api_config_sha256"],
        "historical_manifests": current_history == freeze["historical_manifests"],
    }
    return {"valid": all(checks.values()), "checks": checks}


def _planner_retryable(errors: Sequence[str]) -> bool:
    return bool(errors) and all(not str(error).startswith((
        "registry_", "authoritative_", "local_invariant_", "module_version_"))
        for error in errors)


def _terminal_planner_namespace_retry_eligible(
    *, attempt: int, request_mode: str, outline: Any,
    errors: Sequence[str], parse_errors: Sequence[str],
) -> bool:
    """Allow one choice-ID-only repair after the three normal attempts."""
    return bool(
        attempt == PLANNER_BASE_MAX_ATTEMPTS and
        request_mode == "full_outline" and
        planner_namespace_only_retry_eligible(
            outline=outline, errors=errors, parse_errors=parse_errors)
    )


def _scene_metadata(scene: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(output))
    for key in (
        "scene_alias", "source_anchor_id", "source_anchor",
        "placement_choice_id", "placement_choice", "placement",
        "link_type", "linked_existing_refs", "bridge_reason", "new_development",
        "goal", "speaker_participants", "relationship_dynamic", "interaction_beats",
        "continuity_requirements", "predecessor_scene_refs", "dependency_scene_refs",
        "dependency_edges", "transition_to_next_scene", "branch_role",
        "merge_target_scene_ref", "impact_target_refs", "primary_action",
        "secondary_action", "narrative_action_contracts",
        "durable_update_compilation_stage", "compiled_proposals", "desired_outcome",
    ):
        row[key] = deepcopy(scene.get(key))
    return row


def _child(root_id: str, scenes: Sequence[Mapping[str, Any]], *, role: str,
           parent_path: Path, parent_sha: str) -> dict[str, Any]:
    rows = deepcopy(list(scenes))
    return {
        "child_schema_version": "continuous_expansion_child_v1",
        "root_id": root_id, "round_index": 1, "artifact_role": role,
        "parent_path": str(parent_path), "parent_sha256": parent_sha,
        "storage_mode": "append_only", "parent_rewritten": False,
        "scenes": rows, "scene_count": len(rows),
        "turn_count": sum(len(row.get("dialogue_turns") or []) for row in rows),
    }


def _world_artifacts(memory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "world_memory_final.json": deepcopy(dict(memory)),
        "memory_transitions.json": deepcopy(memory.get("memory_transitions") or []),
        "active_head_index.json": deepcopy(memory.get("active_head_index") or {}),
        "provenance_ledger.json": deepcopy(memory.get("provenance_ledger") or []),
        "pending_updates.json": deepcopy(memory.get("pending_updates") or []),
        "rejected_audit_log.json": deepcopy(memory.get("rejected_audit_log") or []),
    }


def _apply_durable_updates(
    *, run_root: Path, memory: Mapping[str, Any], compiled_plan: Mapping[str, Any],
    generated: Mapping[str, Mapping[str, Any]], accepted_updates: Sequence[Mapping[str, Any]],
    run_id: str = RUN_ID,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = deepcopy(dict(memory))
    transactions: list[dict[str, Any]] = []
    committed_proposal_refs: set[str] = set()
    # Exact temporary-reference bindings survive scene transaction boundaries.
    # This lets a later accepted event refer to a person allocated by an earlier
    # accepted scene without name matching or provisional authoritative IDs.
    committed_temporary_ref_allocations: dict[str, str] = {}
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in accepted_updates:
        by_scene[str(row["source_scene_refs"][0])].append(row)
    scene_by_ref = {str(row["scene_ref"]): row for row in compiled_plan.get("scenes") or []}
    for scene_ref in [str(row["scene_ref"]) for row in compiled_plan.get("scenes") or []
                      if str(row["scene_ref"]) in by_scene]:
        updates = by_scene[scene_ref]
        proposals = [deepcopy(row["compiled_proposal"]) for row in updates]
        evidence_by_proposal = {
            str(row["compiled_proposal"]["proposal_ref"]): list(row["evidence_turn_refs"])
            for row in updates}
        evidence = validate_scene_local_evidence_refs(
            scene_ref=scene_ref, dialogue_turns=generated[scene_ref]["dialogue_turns"],
            accepted_evidence_by_proposal=evidence_by_proposal)
        scene_index = int(scene_ref.split(":S", 1)[1])
        allocation = build_allocation_plan(
            root_id=str(current["root_id"]), round_index=1, scene_index=scene_index,
            compiled_proposals=proposals,
            high_water_marks=current.get("high_water_marks") or {})
        current_allocations = dict(
            allocation.get("temporary_ref_allocations") or {})
        conflicts = {
            ref for ref, value in current_allocations.items()
            if ref in committed_temporary_ref_allocations and
            committed_temporary_ref_allocations[ref] != value
        }
        if conflicts:
            raise RuntimeError(
                "transaction_temporary_ref_reallocation_conflict:" +
                ",".join(sorted(conflicts)))
        allocation["temporary_ref_allocations"] = {
            **committed_temporary_ref_allocations,
            **current_allocations,
        }
        claims = [{"proposal_ref": ref, "realization_status": "realized",
                   "candidate_evidence_turn_refs": refs}
                  for ref, refs in evidence_by_proposal.items()]
        reviewer = {"proposal_reviews": [{
            "proposal_ref": ref, "review_decision": "confirmed",
            "verification_status": "semantically_verified",
            "accepted_evidence_turn_refs": refs,
            "reason_code": "round_expansion_validator_confirmed",
        } for ref, refs in evidence_by_proposal.items()],
            "scope_status": "clean", "world_consistency_status": "clean"}
        candidate = apply_scene_commit(
            current, scene_ref=scene_ref, round_index=1, scene_index=scene_index,
            compiled_proposals=proposals, proposal_claims=claims,
            committer_output=reviewer, allocation_plan=allocation,
            evidence_final_turn_refs=evidence["evidence_final_turn_refs"],
            evidence_text_sha256=evidence["evidence_text_sha256"],
            previously_committed_proposal_refs=sorted(committed_proposal_refs))
        expected_allocations = {
            str(ref): str(row["memory_ref"])
            for ref, row in allocation["proposal_allocations"].items()
        }
        candidate_records = {
            str(row.get("proposal_ref")): str(row.get("memory_ref"))
            for row in candidate.get("authoritative_records") or []
            if row.get("proposal_ref")
        }
        missing_allocations = {
            ref: memory_ref for ref, memory_ref in expected_allocations.items()
            if candidate_records.get(ref) != memory_ref
        }
        if missing_allocations:
            raise RuntimeError(
                "transaction_candidate_allocation_not_materialized:" +
                ",".join(f"{ref}={memory_ref}" for ref, memory_ref
                         in sorted(missing_allocations.items())))
        artifacts = _world_artifacts(candidate)
        tx = commit_scene_transaction(
            run_root=run_root, run_id=run_id, scene_plan=scene_by_ref[scene_ref],
            proposal_fingerprints=[str(row["proposal_fingerprint"]) for row in proposals],
            allocation_plan=allocation, artifacts=artifacts)
        transactions.append(tx)
        current = candidate
        committed_proposal_refs.update(expected_allocations)
        committed_temporary_ref_allocations.update(current_allocations)
    return current, transactions


def _audit_transaction_memory_consistency(
    *, memory: Mapping[str, Any], transactions: Sequence[Mapping[str, Any]],
    journal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that every committed allocation was materialized in authoritative Memory."""
    expected: dict[str, str] = {}
    transaction_ids: list[str] = []
    for transaction in transactions:
        if transaction.get("status") != "COMMITTED":
            continue
        transaction_ids.append(str(transaction.get("transaction_id")))
        allocations = (transaction.get("allocation_plan") or {}).get(
            "proposal_allocations") or {}
        for proposal_ref, allocation in allocations.items():
            expected[str(proposal_ref)] = str(allocation.get("memory_ref"))
    materialized = {
        str(row.get("proposal_ref")): str(row.get("memory_ref"))
        for row in memory.get("authoritative_records") or []
        if row.get("origin") == "generated_extension" and row.get("proposal_ref")
    }
    provenance_refs = {
        str(row.get("record_ref")) for row in memory.get("provenance_ledger") or []
        if row.get("record_ref")
    }
    missing_or_mismatched = {
        proposal_ref: {"expected_memory_ref": memory_ref,
                       "materialized_memory_ref": materialized.get(proposal_ref)}
        for proposal_ref, memory_ref in expected.items()
        if materialized.get(proposal_ref) != memory_ref
    }
    missing_provenance = sorted(
        memory_ref for memory_ref in expected.values() if memory_ref not in provenance_refs)
    prepared_ids = [str(row.get("transaction_id")) for row in journal_rows
                    if row.get("status") == "PREPARED"]
    committed_ids = [str(row.get("transaction_id")) for row in journal_rows
                     if row.get("status") == "COMMITTED"]
    checks = {
        "all_committed_allocations_materialized": not missing_or_mismatched,
        "all_committed_allocations_have_provenance": not missing_provenance,
        "transactions_have_prepared_records": sorted(transaction_ids) == sorted(prepared_ids),
        "transactions_have_committed_records": sorted(transaction_ids) == sorted(committed_ids),
        "prepared_committed_counts_match": len(prepared_ids) == len(committed_ids),
    }
    return {
        "audit_version": "transaction_memory_consistency_v1",
        "valid": all(checks.values()), "checks": checks,
        "expected_proposal_memory_refs": expected,
        "materialized_proposal_memory_refs": materialized,
        "missing_or_mismatched_allocations": missing_or_mismatched,
        "missing_provenance_memory_refs": missing_provenance,
        "prepared_count": len(prepared_ids), "committed_count": len(committed_ids),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    usage = summary.get("api_usage") or {}
    return f"""# Slim Continuous Expansion v11 — Live Canary Report

## Result

- Conclusion: **{summary.get('conclusion')}**
- Status: `{summary.get('status')}`
- Failure stage: `{summary.get('failure_stage')}`
- Run root: `{summary.get('run_root')}`
- generation_ready: `{str(summary.get('generation_ready')).lower()}`
- fresh_e2e_ready: `{str(summary.get('fresh_e2e_ready')).lower()}`
- qa_pipeline_ready: `false`

## Expansion

- v10 baseline: 3 scenes / 19 turns
- v11 planned scenes: {summary.get('planned_scene_count')}
- v11 realized local-pass scenes: {summary.get('realized_scene_count')}
- v11 full child: {summary.get('full_scene_count')} scenes / {summary.get('full_turn_count')} turns
- v11 promotable child: {summary.get('promotable_scene_count')} scenes / {summary.get('promotable_turn_count')} turns
- Canonical Parent + Child materialized: {summary.get('complete_dialogue_materialized')}
- Canonical cumulative turns: {summary.get('complete_dialogue_turn_count')}
- Parent tail P05:T010–P05:T016 retained: {summary.get('parent_tail_retained')}
- Same-round predecessor text included: {summary.get('same_round_text_included')}

## Semantic review and durable state

- Round decision: `{summary.get('round_decision')}`
- Round Validator prompt budget: `{summary.get('round_validator_prompt_budget')}`
- Accepted/rejected closure scenes: {summary.get('accepted_scene_refs')} / {summary.get('rejected_scene_refs')}
- Durable accepted/rejected/audit-only: {summary.get('durable_update_counts')}
- Binder allocations / transactions: {summary.get('binder_allocation_count')} / {summary.get('transaction_count')}
- PREPARED / COMMITTED: {summary.get('prepared_count')} / {summary.get('committed_count')}

## API discipline

- Planner / Realizer / Round Validator calls: {((usage.get('roles') or {}).get('planner') or {}).get('physical_requests')} / {((usage.get('roles') or {}).get('realizer') or {}).get('physical_requests')} / {((usage.get('roles') or {}).get('round_validator') or {}).get('physical_requests')}
- Total calls / tokens: {usage.get('total_physical_requests')} / {usage.get('total_tokens')}
- Evidence Committer calls: {usage.get('evidence_committer_calls')}
- Initial Structure Extractor calls: {usage.get('initial_structure_extractor_calls')}
- Cache/imported/legacy reuse: 0 / 0 / 0

## Integrity

Post-run frozen-integrity valid: `{str((summary.get('post_run_integrity') or {}).get('valid')).lower()}`. Historical v6-v10 and Phase A/B1 manifests were checked against the pre-request freeze. No Round 2, QA, Solver, Training, or Git commit was performed.
"""


def _semantic_markdown(summary: Mapping[str, Any]) -> str:
    return f"""# Live Canary Semantic Audit

This audit separates mechanical readiness from dialogue quality. The v10 baseline is 3 scenes / 19 turns; v11 produced {summary.get('full_scene_count')} full-child scenes / {summary.get('full_turn_count')} turns and {summary.get('promotable_scene_count')} promotable scenes / {summary.get('promotable_turn_count')} turns.

The round-wide validator decision was `{summary.get('round_decision')}`. Accepted closure scenes: `{summary.get('accepted_scene_refs')}`. Rejected or dependency-blocked scenes: `{summary.get('rejected_scene_refs')}`. Durable update counts: `{summary.get('durable_update_counts')}`.

Parent tail preservation is `{summary.get('parent_tail_retained')}` and same-round predecessor text propagation is `{summary.get('same_round_text_included')}`. The structured per-scene validator fields, exact evidence refs, jump/anchor metadata, future-knowledge flags, duplicates, continuity, and new-development decisions are preserved in `round_validator_output.json`, `scene_decision_closure.json`, and `reports/scene_semantic_matrix.json` for direct human inspection.

Final selection: **{summary.get('conclusion')}**. This run does not authorize Round 2 automatically.
"""


def execution_progress_v11(
    result: Mapping[str, Any], *,
    parent_tail_refs: Sequence[str] = PARENT_TAIL_REFS,
) -> dict[str, Any]:
    """Summarize durable milestones even when a later live stage fails."""
    compiled = result.get("compiled_round_plan") or {}
    realized = result.get("realized") or []
    contexts = result.get("contexts") or {}
    return {
        "planned_scene_count": len(compiled.get("scenes") or []),
        "realized_scene_count": len(realized),
        "realized_turn_count": sum(
            len(scene.get("dialogue_turns") or []) for scene in realized),
        "parent_tail_retained": bool(contexts) and all(
            set(parent_tail_refs).issubset(
                set(row.get("parent_true_tail_refs") or []))
            for row in contexts.values()),
        "same_round_text_included": all(
            not row.get("same_round_predecessor_refs") or
            row.get("same_round_scene_text_included")
            for row in contexts.values()),
    }


class CanaryState:
    def __init__(self, run_root: Path, freeze: Mapping[str, Any]):
        self.run_root = run_root
        self.freeze = deepcopy(dict(freeze))
        self.first_api_at: str | None = None

    def before_request(self, logical_role: str, role_version: str) -> None:
        audit = _freeze_guard(self.freeze)
        if not audit["valid"]:
            atomic_json(self.run_root / "reports/freeze_guard_failure.json", audit)
            raise RuntimeError("post_freeze_integrity_drift")
        if logical_role not in {"planner", "realizer", "round_validator"}:
            raise RuntimeError(f"forbidden_live_role:{logical_role}:{role_version}")
        if self.first_api_at is None:
            self.first_api_at = _now()
            atomic_json(self.run_root / "run_status.json", {
                "status": "live_requests_started", "first_api_request_at": self.first_api_at,
                "code_prompt_schema_configuration_frozen": True})


def execute_live_canary(
    *, preflight: Mapping[str, Any], input_path: Path, run_root: Path,
    api_config_path: Path, tokenizer_path: Path,
    growth_profile: Mapping[str, Any] | None = None,
    dialogue_first_participants: bool = False,
    durable_update_policy: str = "strict",
    root_id: str = ROOT_ID,
    orchestration_seed: int = ORCHESTRATION_SEED,
    run_id: str = RUN_ID,
    parent_tail_refs: Sequence[str] = PARENT_TAIL_REFS,
    before_paid_request: Callable[[str, bool, int], None] | None = None,
) -> dict[str, Any]:
    if not preflight.get("valid"):
        raise RuntimeError("live_canary_preflight_not_valid")
    if growth_profile is not None and preflight.get("growth_profile") != dict(
            growth_profile):
        raise RuntimeError("live_canary_growth_profile_not_frozen")
    if dialogue_first_participants and preflight.get(
            "dialogue_first_participants") is not True:
        raise RuntimeError("dialogue_first_participant_mode_not_frozen")
    if preflight.get("durable_update_policy", "strict") != durable_update_policy:
        raise RuntimeError("live_canary_durable_update_policy_not_frozen")
    if run_root.exists():
        raise FileExistsError(f"live_canary_run_root_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    freeze = deepcopy(dict(preflight))
    freeze.update({"freeze_manifest_version": "v11_live_canary_freeze_v1",
                   "frozen_at": _now(), "fresh": True, "no_resume": True,
                   "cache_reuse": 0, "imported_response_reuse": 0,
                   "legacy_artifact_reuse": 0})
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0,
        "run_id": run_id, "run_root": str(run_root)})
    state = CanaryState(run_root, freeze)
    client = V11LiveRoleTransportClient(
        config=live_api_configuration_v11(api_config_path),
        authoritative_input_sha256=freeze["input_sha256"],
        orchestration_seed=orchestration_seed,
        prompt_schema_contract_sha256=freeze["prompt_schema_contract_sha256"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_counter = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
    ledger = RequestLedger(
        run_root, manifest_sha256=freeze["implementation_manifest"][
            "implementation_manifest_sha256"], backend_mode="live")
    frozen_request_budget = freeze.get("request_budget") or {}
    batch_cost_control = frozen_request_budget.get("batch_cost_control") is True
    planner_attempt_limit = int(
        frozen_request_budget.get("planner_max") or PLANNER_BASE_MAX_ATTEMPTS)
    realizer_attempt_limit = int(
        frozen_request_budget.get("realizer_per_scene_max") or 3)
    validator_attempt_limit = int(
        frozen_request_budget.get("round_validator_max") or 2)
    if batch_cost_control and (
            planner_attempt_limit not in {1, 2, 3} or
            realizer_attempt_limit not in {1, 2, 3} or
            validator_attempt_limit not in {1, 2}):
        raise RuntimeError("batch_cost_control_attempt_limits_invalid")
    planner_loop_limit = (planner_attempt_limit if batch_cost_control else
                          PLANNER_ABSOLUTE_MAX_ATTEMPTS)
    realizer_role_limit = 7 * realizer_attempt_limit
    total_request_limit = (
        planner_loop_limit + realizer_role_limit + validator_attempt_limit)
    caller = LiveJsonRoleCallerV11(
        run_root=run_root, ledger=ledger, client=client, token_counter=token_counter,
        root_id=root_id,
        role_budget_limits={"planner": planner_loop_limit,
                            "realizer": realizer_role_limit,
                            "round_validator": validator_attempt_limit},
        total_request_limit=total_request_limit, context_limit=40_960,
        before_request=state.before_request,
        paid_request_observer=before_paid_request)
    result: dict[str, Any] = {}
    memory_initial: dict[str, Any] = {}
    failure_stage: str | None = None
    try:
        root = json.loads(input_path.read_text(encoding="utf-8"))
        if str(root.get("root_id") or "") != root_id:
            raise RuntimeError(
                f"local_invariant_failure:root_identity:{root.get('root_id')}:{root_id}")
        parent = read_unified_parent(input_path)
        source = (initialize_source_world_memory_v2(root, parent)
                  if dialogue_first_participants
                  else initialize_source_world_memory(root, parent))
        memory_initial = source["memory"]
        interaction_links = compile_source_interaction_links(
            root_id=root_id, sessions=_sessions(parent),
            speaker_to_entity=source["speaker_to_entity_ref"])
        complete = compose_complete_dialogue_view(
            original_parent=parent, promoted_rounds=(),
            speaker_to_entity_ref=source["speaker_to_entity_ref"])
        complete_validation = validate_complete_dialogue_view(complete)
        planning = compile_planning_state_view(
            memory=memory_initial, interaction_links=interaction_links,
            scene_lineage=(), recent_round_end={})
        planning_validation = validate_planning_state_view(planning)
        separation = audit_interaction_relation_separation(planning)
        anchors = compile_anchor_registry(
            complete_dialogue=complete, planning_state=planning, tail_turn_count=7)
        anchor_validation = validate_anchor_registry(anchors, complete_dialogue=complete)
        source_anchors = compile_source_anchor_registry(anchors)
        source_anchor_validation = validate_source_anchor_registry(source_anchors)
        placement_choices = compile_placement_choice_registry(
            complete_dialogue=complete, anchor_registry=anchors)
        placement_choice_validation = validate_placement_choice_registry(
            placement_choices, complete_dialogue=complete)
        action_menu = compile_action_menu_v3(
            authoritative_affordance_view(memory_initial), root_id=root_id,
            interaction_links=interaction_links, maximum_actions=16)
        action_validation = validate_action_menu_v3(action_menu)
        local_initial = [complete_validation, planning_validation, separation,
                         anchor_validation, source_anchor_validation,
                         placement_choice_validation, action_validation]
        if not all(row.get("valid") for row in local_initial):
            raise RuntimeError("local_invariant_failure:initial_views")
        roles = _role_table(planning)
        shared_action_contracts = action_contract_registry()
        action_availability = compile_action_availability_view(
            planning_state=planning)
        action_availability_validation = validate_action_availability_view(
            action_availability)
        if not action_availability_validation["valid"]:
            raise RuntimeError("local_invariant_failure:action_availability")
        if growth_profile is not None:
            profile_validation = validate_growth_profile_definition(growth_profile)
            if not profile_validation["valid"]:
                raise RuntimeError("local_invariant_failure:growth_profile_definition")
        for name, value in {
            "world_memory_initial.json": memory_initial,
            "source_initialization.json": source,
            "source_turn_registry.json": source.get("source_turn_registry"),
            "interaction_links.json": interaction_links,
            "complete_dialogue_view.json": complete,
            "planning_state_view.json": planning,
            "anchor_registry.json": anchors,
            "internal_durable_affordance_menu.json": action_menu,
            "source_anchor_registry.json": source_anchors,
            "placement_choice_registry.json": placement_choices,
            "shared_narrative_action_contracts.json": shared_action_contracts,
            "action_availability_view.json": action_availability,
            "growth_profile.json": deepcopy(dict(growth_profile))
                if growth_profile is not None else None,
            "initial_local_validations.json": local_initial,
        }.items():
            atomic_json(run_root / name, value)

        base_planner_prompt, frozen_planner_sha = build_planner_prompt_v11(
            root_id=root_id, planning_state=planning,
            source_anchor_registry=source_anchors,
            placement_choice_registry=placement_choices,
            action_contracts=shared_action_contracts,
            action_availability_view=action_availability,
            growth_profile=growth_profile,
            dialogue_first_participants=dialogue_first_participants)
        prompt = base_planner_prompt
        planner_request_mode = "full_outline"
        action_repair_spec = None
        action_repair_base = None
        namespace_repair_spec = None
        namespace_repair_base = None
        speaker_repair_spec = None
        speaker_repair_base = None
        planner_attempts, model_plan, compiled = [], None, None
        for attempt in range(1, planner_loop_limit + 1):
            parsed, _, parse_errors = caller.call_json(
                request_id=f"planner_v11_attempt_{attempt:02d}", role="planner_v11",
                attempt=attempt, prompt=prompt,
                prompt_version=(
                    PLANNER_PROMPT_VERSION_V11 if attempt == 1 else
                    PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11
                    if planner_request_mode == "action_only" else
                    PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11
                    if planner_request_mode == "namespace_only" else
                    PLANNER_SPEAKER_REPAIR_PROMPT_VERSION_V11
                    if planner_request_mode == "speaker_only" else
                    PLANNER_RETRY_PROMPT_VERSION_V11),
                max_tokens=PLANNER_MAX_OUTPUT_TOKENS)
            errors = list(parse_errors)
            validation = None
            growth_profile_validation = None
            structural = None
            candidate_compiled = None
            candidate_outline = parsed
            action_repair_result = None
            namespace_repair_result = None
            speaker_repair_result = None
            if planner_request_mode == "action_only" and parsed is not None:
                action_repair_result = apply_planner_action_repairs_v11(
                    frozen_outline=action_repair_base,
                    response=parsed, spec=action_repair_spec)
                errors.extend(action_repair_result["errors"])
                candidate_outline = action_repair_result.get("reconstructed_outline")
            if planner_request_mode == "namespace_only" and parsed is not None:
                namespace_repair_result = apply_planner_namespace_repairs_v11(
                    frozen_outline=namespace_repair_base,
                    response=parsed, spec=namespace_repair_spec)
                errors.extend(namespace_repair_result["errors"])
                candidate_outline = namespace_repair_result.get(
                    "reconstructed_outline")
            if planner_request_mode == "speaker_only" and parsed is not None:
                speaker_repair_result = apply_planner_speaker_repairs_v11(
                    frozen_outline=speaker_repair_base,
                    response=parsed, spec=speaker_repair_spec)
                errors.extend(speaker_repair_result["errors"])
                candidate_outline = speaker_repair_result.get(
                    "reconstructed_outline")
            if candidate_outline is not None:
                validation = validate_planner_outline_v11(
                    candidate_outline, source_anchor_registry=source_anchors,
                    placement_choice_registry=placement_choices,
                    action_availability_view=action_availability,
                    authoritative_entity_refs={row["entity_ref"] for row in roles},
                    authoritative_entities=planning.get("entities") or [])
                errors.extend(validation["errors"])
                # Coverage is independent of action/reference validity and is
                # safe to inspect on any parsed outline. Aggregate it now so a
                # retry sees latent growth errors instead of discovering them
                # only after an action-only patch consumes the final attempt.
                if growth_profile is not None:
                    growth_profile_validation = validate_planner_growth_profile_v11(
                        candidate_outline, profile=growth_profile)
                    errors.extend(growth_profile_validation["errors"])
                if validation["valid"]:
                    try:
                        candidate_compiled = compile_planner_outline_v11(
                            candidate_outline,
                            source_anchor_registry=source_anchors,
                            placement_choice_registry=placement_choices,
                            planning_state=planning,
                            action_availability_view=action_availability,
                            round_index=1)
                        structural = validate_round_blueprint_structure(
                            compiled_plan=candidate_compiled,
                            source_anchor_registry=source_anchors,
                            placement_choice_registry=placement_choices,
                            planning_state=planning,
                            complete_dialogue=complete)
                        errors.extend(structural["errors"])
                    except (ValueError, KeyError) as error:
                        errors.append(f"planner_compile_contract:{error}")
            repair_audit = None
            if action_repair_result is not None:
                repair_audit = {key: deepcopy(value)
                                for key, value in action_repair_result.items()
                                if key != "reconstructed_outline"}
            namespace_repair_audit = None
            if namespace_repair_result is not None:
                namespace_repair_audit = {
                    key: deepcopy(value)
                    for key, value in namespace_repair_result.items()
                    if key != "reconstructed_outline"
                }
            speaker_repair_audit = None
            if speaker_repair_result is not None:
                speaker_repair_audit = {
                    key: deepcopy(value)
                    for key, value in speaker_repair_result.items()
                    if key != "reconstructed_outline"
                }
            row = {"attempt": attempt, "request_mode": planner_request_mode,
                   "frozen_input_sha256": frozen_planner_sha,
                   "parse_errors": parse_errors, "validation": validation,
                   "growth_profile_validation": growth_profile_validation,
                   "structural_validation": structural, "errors": list(dict.fromkeys(errors)),
                   "action_repair_spec_sha256": (
                       (action_repair_spec or {}).get("repair_spec_sha256")),
                   "action_repair_audit": repair_audit,
                   "namespace_repair_spec_sha256": (
                       (namespace_repair_spec or {}).get("repair_spec_sha256")),
                   "namespace_repair_audit": namespace_repair_audit,
                   "speaker_repair_spec_sha256": (
                       (speaker_repair_spec or {}).get("repair_spec_sha256")),
                   "speaker_repair_audit": speaker_repair_audit,
                   "retryable": _planner_retryable(errors)}
            planner_attempts.append(row)
            atomic_json(run_root / f"planner/attempt_{attempt:02d}_validation.json", row)
            if not errors and candidate_compiled is not None:
                model_plan, compiled = candidate_outline, candidate_compiled
                break
            if (attempt == planner_loop_limit or
                    not _planner_retryable(errors)):
                break
            terminal_namespace_retry = _terminal_planner_namespace_retry_eligible(
                attempt=attempt, request_mode=planner_request_mode,
                outline=candidate_outline, errors=errors,
                parse_errors=parse_errors)
            if attempt >= PLANNER_BASE_MAX_ATTEMPTS and not terminal_namespace_retry:
                break
            if terminal_namespace_retry:
                namespace_repair_base = deepcopy(candidate_outline)
                namespace_repair_spec = compile_planner_namespace_repair_spec_v11(
                    outline=namespace_repair_base, errors=errors,
                    validation=validation or {},
                    action_contracts=shared_action_contracts,
                    action_availability_view=action_availability,
                    authoritative_role_table=roles)
                prompt = build_planner_namespace_only_retry_prompt_v11(
                    namespace_repair_spec)
                planner_request_mode = "namespace_only"
                action_repair_spec = None
                action_repair_base = None
                speaker_repair_spec = None
                speaker_repair_base = None
                continue
            if (planner_request_mode == "action_only" and
                    (candidate_outline is None or validation is None)):
                prompt = build_planner_action_only_retry_prompt_v11(
                    action_repair_spec, previous_contract_errors=errors)
                continue
            if (planner_request_mode == "namespace_only" and
                    (candidate_outline is None or validation is None)):
                prompt = build_planner_namespace_only_retry_prompt_v11(
                    namespace_repair_spec, previous_contract_errors=errors)
                continue
            if (planner_request_mode == "speaker_only" and
                    (candidate_outline is None or validation is None)):
                prompt = build_planner_speaker_only_retry_prompt_v11(
                    speaker_repair_spec, previous_contract_errors=errors)
                continue
            if planner_namespace_only_retry_eligible(
                    outline=candidate_outline, errors=errors,
                    parse_errors=parse_errors):
                namespace_repair_base = deepcopy(candidate_outline)
                namespace_repair_spec = compile_planner_namespace_repair_spec_v11(
                    outline=namespace_repair_base, errors=errors,
                    validation=validation or {},
                    action_contracts=shared_action_contracts,
                    action_availability_view=action_availability,
                    authoritative_role_table=roles)
                prompt = build_planner_namespace_only_retry_prompt_v11(
                    namespace_repair_spec)
                planner_request_mode = "namespace_only"
                action_repair_spec = None
                action_repair_base = None
                speaker_repair_spec = None
                speaker_repair_base = None
            elif planner_action_only_retry_eligible(
                    outline=candidate_outline, errors=errors,
                    parse_errors=parse_errors):
                action_repair_base = deepcopy(candidate_outline)
                action_repair_spec = compile_planner_action_repair_spec_v11(
                    outline=action_repair_base, errors=errors,
                    validation=validation or {},
                    action_contracts=shared_action_contracts,
                    action_availability_view=action_availability,
                    authoritative_role_table=roles)
                prompt = build_planner_action_only_retry_prompt_v11(
                    action_repair_spec)
                planner_request_mode = "action_only"
                speaker_repair_spec = None
                speaker_repair_base = None
                namespace_repair_spec = None
                namespace_repair_base = None
            elif planner_speaker_only_retry_eligible(
                    outline=candidate_outline, errors=errors,
                    parse_errors=parse_errors):
                speaker_repair_base = deepcopy(candidate_outline)
                speaker_repair_spec = compile_planner_speaker_repair_spec_v11(
                    outline=speaker_repair_base, errors=errors,
                    authoritative_role_table=roles)
                prompt = build_planner_speaker_only_retry_prompt_v11(
                    speaker_repair_spec)
                planner_request_mode = "speaker_only"
                action_repair_spec = None
                action_repair_base = None
                namespace_repair_spec = None
                namespace_repair_base = None
            else:
                prompt = build_planner_retry_prompt_v11(
                    base_prompt=base_planner_prompt,
                    frozen_input_sha256=frozen_planner_sha,
                    errors=errors, validation=validation,
                    action_availability_view=action_availability)
                planner_request_mode = "full_outline"
                action_repair_spec = None
                action_repair_base = None
                namespace_repair_spec = None
                namespace_repair_base = None
                speaker_repair_spec = None
                speaker_repair_base = None
        atomic_json(run_root / "planner/planner_attempts.json", planner_attempts)
        result["planner_attempts"] = planner_attempts
        if model_plan is None or compiled is None:
            failure_stage = "Planner schema/compile/structural validation"
            raise RuntimeError("planner_attempts_exhausted")
        atomic_json(run_root / "planner/planner_output.json", model_plan)
        atomic_json(run_root / "planner/compiled_round_plan.json", compiled)
        dag = build_scene_dependency_dag(compiled)
        narrative = build_narrative_order_manifest(
            compiled_plan=compiled, complete_dialogue=complete)
        atomic_json(run_root / "scene_dependency_dag.json", dag)
        atomic_json(run_root / "narrative_order_manifest.json", narrative)
        result.update({"compiled_round_plan": compiled, "model_plan": model_plan,
                       "scene_dependency_dag": dag,
                       "narrative_order_manifest": narrative})

        generated: dict[str, dict[str, Any]] = {}
        contexts: dict[str, dict[str, Any]] = {}
        local_results: dict[str, dict[str, Any]] = {}
        realizer_attempts: dict[str, list[dict[str, Any]]] = {}
        parent_text = [str(row.get("text") or "") for row in complete.get("turns") or []]
        previous_text: list[str] = []
        scene_count = len(compiled["scenes"])
        for scene in compiled["scenes"]:
            scene_ref = str(scene["scene_ref"])
            hard = [str(edge["predecessor_scene_ref"])
                    for edge in scene.get("dependency_edges") or []
                    if edge.get("hard_dependency")]
            blockers = [ref for ref in hard if (local_results.get(ref) or {}).get(
                "status") != "local_gate_passed"]
            round_prefix = scene_ref.split(":S", 1)[0]
            continuation_sources: list[str] = []
            for action in (scene.get("primary_action"),
                           scene.get("secondary_action")):
                if (not isinstance(action, Mapping) or
                        action.get("action_mode") != "continuation"):
                    continue
                continuation_source = str(action.get("continuation_of") or "")
                if (len(continuation_source) >= 3 and
                        continuation_source[1:3].isdigit()):
                    source_scene_ref = (
                        f"{round_prefix}:S{continuation_source[1:3]}")
                    if ((local_results.get(source_scene_ref) or {}).get(
                            "status") != "local_gate_passed"):
                        continuation_sources.append(source_scene_ref)
            blockers = list(dict.fromkeys([*blockers, *continuation_sources]))
            if blockers:
                local_results[scene_ref] = dependency_unavailable_result(scene, blockers)
                realizer_attempts[scene_ref] = []
                continue
            try:
                context = compile_scene_context_v2(
                    scene=scene, complete_dialogue=complete,
                    source_anchor_registry=source_anchors,
                    placement_choice_registry=placement_choices,
                    planning_state=planning, same_round_scenes=generated)
            except ContextInvariantFailure as error:
                failure_stage = "Local invariant failure"
                raise RuntimeError(f"local_invariant_failure:{scene_ref}:{error}") from error
            contexts[scene_ref] = context
            atomic_json(run_root / f"scenes/{scene_ref.replace(':', '_')}/context.json", context)
            budget = _round_turn_budget(scene, scene_count)
            atomic_json(run_root / f"scenes/{scene_ref.replace(':', '_')}/turn_budget.json", budget)
            base_prompt, frozen_scene_sha = build_realizer_prompt_v11(
                scene=scene, context_manifest=context, turn_budget=budget,
                round_blueprint=compiled["round_blueprint"])
            prompt = base_prompt
            attempts: list[dict[str, Any]] = []
            allowed_speaker_refs = {
                str(row["participant_ref"]) for row in scene["speaker_participants"]
            }
            for attempt in range(1, realizer_attempt_limit + 1):
                parsed, _, parse_errors = caller.call_json(
                    request_id=f"realizer_{scene_ref.replace(':', '_')}_attempt_{attempt:02d}",
                    role="realizer_v11", attempt=attempt, prompt=prompt,
                    prompt_version=(REALIZER_PROMPT_VERSION_V11 if attempt == 1
                                    else REALIZER_RETRY_PROMPT_VERSION_V11),
                    max_tokens=4096)
                normalized = normalize_realizer_dialogue_v11(
                    parsed, scene_ref=scene_ref,
                    allowed_speaker_refs=allowed_speaker_refs)
                errors = [*parse_errors, *normalized["errors"]]
                gate = None
                if normalized["valid"]:
                    gate = validate_local_scene_v11(
                        scene=scene, scene_output=normalized["scene_output"],
                        context_manifest=context, turn_budget=budget,
                        complete_dialogue=complete,
                        source_anchor_registry=source_anchors,
                        placement_choice_registry=placement_choices,
                        generated_scenes=generated, parent_utterances=parent_text,
                        previous_child_utterances=previous_text)
                    errors.extend(gate["errors"])
                attempt_row = {"attempt": attempt,
                    "frozen_realizer_input_sha256": frozen_scene_sha,
                    "parse_errors": parse_errors, "normalization": normalized,
                    "local_gate": gate, "errors": list(dict.fromkeys(errors)),
                    "retry_mode": "full_scene_retry" if errors else "accepted"}
                attempts.append(attempt_row)
                atomic_json(run_root / f"scenes/{scene_ref.replace(':', '_')}/attempt_{attempt:02d}_validation.json", attempt_row)
                if not errors and gate is not None and gate["valid"]:
                    output = _scene_metadata(scene, normalized["scene_output"])
                    generated[scene_ref] = output
                    local_results[scene_ref] = gate
                    previous_text.extend(str(row["text"]) for row in output["dialogue_turns"])
                    atomic_json(run_root / f"scenes/{scene_ref.replace(':', '_')}/realized_scene.json", output)
                    break
                if attempt < realizer_attempt_limit:
                    quality = ((gate or {}).get("quality") or {})
                    forbidden_utterances = [
                        *(quality.get("parent_exact_copies") or []),
                        *(quality.get("previous_child_exact_copies") or []),
                        *(quality.get("internal_exact_duplicates") or []),
                    ]
                    invalid_speaker_refs = []
                    if isinstance(parsed, Mapping):
                        for row in parsed.get("dialogue_turns") or []:
                            if not isinstance(row, Mapping):
                                continue
                            speaker_ref = str(row.get("speaker_ref") or "")
                            if speaker_ref and speaker_ref not in allowed_speaker_refs:
                                invalid_speaker_refs.append(speaker_ref)
                    prompt = build_realizer_retry_prompt_v11(
                        base_prompt=base_prompt, frozen_input_sha256=frozen_scene_sha,
                        errors=errors, turn_budget=budget,
                        retry_attempt=attempt + 1,
                        retry_limit=realizer_attempt_limit,
                        forbidden_utterances=forbidden_utterances,
                        allowed_speaker_refs=sorted(allowed_speaker_refs),
                        invalid_speaker_refs=invalid_speaker_refs,
                        compact_retry_context={
                            "scene_plan": deepcopy(scene),
                            "turn_budget": deepcopy(budget),
                            "required_dialogue_read_only": deepcopy(
                                context.get("p0_required_dialogue") or []),
                            "dependency_dialogue_read_only": deepcopy(
                                context.get("p1_dependency_dialogue") or []),
                            "relevant_planning_state": deepcopy(
                                context.get("p1_planning_state") or {}),
                            "following_context_constraint_only": (
                                "must prevent contradiction but is not known to speakers"),
                            "parent_true_tail_refs": list(
                                context.get("parent_true_tail_refs") or []),
                            "same_round_predecessor_refs": list(
                                context.get("same_round_predecessor_refs") or []),
                        })
            realizer_attempts[scene_ref] = attempts
            if scene_ref not in local_results:
                local_results[scene_ref] = (attempts[-1].get("local_gate") or {
                    "scene_ref": scene_ref, "scene_alias": scene.get("scene_alias"),
                    "status": "local_gate_failed", "valid": False,
                    "errors": attempts[-1]["errors"]})
        atomic_json(run_root / "scene_contexts.json", contexts)
        atomic_json(run_root / "local_scene_gate_results.json", local_results)
        atomic_json(run_root / "realizer_attempt_routes.json", realizer_attempts)
        realized = [generated[str(scene["scene_ref"])] for scene in compiled["scenes"]
                    if str(scene["scene_ref"]) in generated]
        result.update({"generated": generated, "contexts": contexts,
                       "local_results": local_results,
                       "realizer_attempts": realizer_attempts,
                       "realized": realized})
        if not realized:
            failure_stage = "Natural Realizer local gates"
            raise RuntimeError("no_realized_scene_available_for_round_review")

        review = build_review_world_view(
            complete_dialogue=complete, planning_state=planning,
            compiled_round_plan=compiled, realized_scenes=realized,
            scene_contexts=contexts, scene_dependency_dag=dag,
            narrative_order_manifest=narrative)
        review_validation = validate_review_world_view(review)
        atomic_json(run_root / "review_world_view.json", review)
        atomic_json(run_root / "review_world_view_validation.json", review_validation)
        result.update({"review_world_view": review,
                       "review_world_view_validation": review_validation})
        if not review_validation["valid"]:
            failure_stage = "Review World View local invariant"
            raise RuntimeError("local_invariant_failure:review_world_view")
        validator_input = build_round_validator_input(
            round_blueprint=compiled, review_world_view=review,
            realized_scenes=realized, scene_dependency_dag=dag,
            structural_gate_results=local_results)
        atomic_json(run_root / "round_validator_input.json", validator_input)
        validator_budget = audit_round_validator_prompt_budget(
            validator_input, token_counter=token_counter,
            durable_update_policy=durable_update_policy)
        atomic_json(run_root / "round_validator_prompt_budget.json", validator_budget)
        result.update({"round_validator_input": validator_input,
                       "round_validator_prompt_budget": validator_budget})
        if not validator_budget["valid"]:
            failure_stage = "Round Expansion Validator request/context"
            raise RuntimeError(
                "round_validator_prompt_budget_exceeded:"
                f"{validator_budget['total_reserved_tokens']}>"
                f"{validator_budget['context_limit']}")

        def validator_backend(prompt_value: str, attempt: int) -> Any:
            prompt_version = (
                ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11
                if prompt_value.startswith(
                    "ROUND VALIDATOR CONTINUATION CONTRACT REPAIR")
                else (ROUND_VALIDATOR_PROMPT_VERSION if attempt == 1
                      else ROUND_VALIDATOR_RETRY_PROMPT_VERSION)
            )
            parsed, raw, parse_errors = caller.call_json(
                request_id=f"round_validator_attempt_{attempt:02d}",
                role="round_expansion_validator_v1", attempt=attempt,
                prompt=prompt_value, prompt_version=prompt_version,
                max_tokens=ROUND_VALIDATOR_MAX_OUTPUT_TOKENS)
            return parsed if not parse_errors and parsed is not None else raw

        failure_stage = "Round Expansion Validator request/context"
        validator_run = run_round_validator_with_retry(
            backend_call=validator_backend, validator_input=validator_input,
            realized_scenes=realized, review_world_view=review,
            local_gate_results=local_results,
            durable_update_policy=durable_update_policy,
            allow_terminal_contract_repair=not batch_cost_control)
        atomic_json(run_root / "round_validator_run.json", validator_run)
        if validator_run.get("durable_update_projection_audit") is not None:
            atomic_json(
                run_root / "durable_update_projection_audit.json",
                validator_run["durable_update_projection_audit"],
            )
        if validator_run.get("evidence_alignment_audit") is not None:
            atomic_json(
                run_root / "dialogue_evidence_alignment_audit.json",
                validator_run["evidence_alignment_audit"],
            )
        if not validator_run["completed"]:
            failure_stage = "Round Expansion Validator contract"
            raise RuntimeError("round_validator_contract_exhausted")
        validator_output = validator_run["valid_output"]
        atomic_json(run_root / "round_validator_output.json", validator_output)
        result.update({"validator_output": validator_output,
                       "validator_run": validator_run})
        failure_stage = None
        closure = compute_scene_decision_closure(
            realized_scenes=realized, local_gate_results=local_results,
            validator_output=validator_output, scene_dependency_dag=dag)
        durable = compile_durable_updates(
            candidates=validator_output["durable_updates"],
            compiled_round_plan=compiled, scene_decision_closure=closure,
            realized_scenes=realized, authoritative_memory=memory_initial,
            planning_state=planning, action_menu=action_menu)
        durable["audit_only_observations"].extend({
            "candidate_update_key": row.get("observation_key"),
            "disposition": "audit_only_unhinted_update",
            "reason": "round_validator_unhinted_semantic_observation",
            "candidate": deepcopy(dict(row)),
        } for row in validator_output.get("semantic_review_observations") or [])
        atomic_json(run_root / "scene_decision_closure.json", closure)
        atomic_json(run_root / "durable_update_compile_result.json", durable)
        result.update({"closure": closure, "durable": durable})
        memory_final, transactions = _apply_durable_updates(
            run_root=run_root, memory=memory_initial, compiled_plan=compiled,
            generated=generated, accepted_updates=durable["accepted_updates"],
            run_id=run_id)
        growth_audit = audit_authoritative_growth(
            memory_initial=memory_initial, memory_final=memory_final,
            accepted_updates=durable["accepted_updates"])
        growth_profile_admission = (
            evaluate_growth_profile_admission(
                growth_audit, profile=growth_profile)
            if growth_profile is not None else None)
        atomic_json(run_root / "reports/authoritative_growth_audit.json",
                    growth_audit)
        if growth_profile_admission is not None:
            atomic_json(run_root / "reports/growth_profile_admission.json",
                        growth_profile_admission)
        for relative, value in _world_artifacts(memory_final).items():
            atomic_json(run_root / relative, value)
        promotion = promote_scenes_v2(
            realized_scenes=realized, local_gate_results=local_results,
            validator_output=validator_output, scene_decision_closure=closure,
            durable_update_compile_result=durable,
            narrative_order_manifest=narrative)
        atomic_json(run_root / "promotion_v2_result.json", promotion)
        full_child = _child(root_id, promotion["full_child_scenes"],
                            role="full_child", parent_path=input_path,
                            parent_sha=freeze["input_sha256"])
        promotable_child = _child(root_id, promotion["promotable_child_scenes"],
                                  role="promotable_child", parent_path=input_path,
                                  parent_sha=freeze["input_sha256"])
        atomic_json(run_root / "child_dialogue_full.json", full_child)
        atomic_json(run_root / "child_dialogue_promotable.json", promotable_child)
        atomic_json(run_root / "child_dialogue.json", promotable_child)
        promotion_manifest = {
            "promotion_manifest_version": "continuous_expansion_promotion_manifest_v1",
            "artifact_role": "promotable_child",
            "promoted_scene_refs": [row["scene_ref"] for row in promotion["promotable_child_scenes"]],
            "excluded_scene_refs": [*promotion["rejected_scenes"],
                                    *promotion["dependency_blocked_scenes"]],
            "excluded_scene_reasons": {row["scene_ref"]: row["closure_reason"]
                                       for row in closure["scenes"]
                                       if row["closure_adjusted_decision"] != "accept"},
            "committed_proposal_refs": [row["compiled_proposal"]["proposal_ref"]
                                        for row in durable["accepted_updates"]],
            "committed_memory_refs": [row["memory_ref"]
                                      for row in memory_final["authoritative_records"]
                                      if row.get("origin") == "generated_extension"],
            "source_full_child_path": str(run_root / "child_dialogue_full.json"),
            "narrative_order_manifest": str(run_root / "narrative_order_manifest.json"),
        }
        atomic_json(run_root / "promotion_manifest.json", promotion_manifest)
        expanded_path = run_root / "expanded_dialogue_round_1.json"
        expanded_markdown_path = run_root / "expanded_dialogue_round_1.md"
        expanded = assemble_expanded_dialogue(
            original_parent=root, promotable_child=promotable_child,
            narrative_order_manifest=narrative,
            speaker_to_entity_ref=source["speaker_to_entity_ref"],
            parent_path=input_path, parent_sha256=freeze["input_sha256"],
            child_path=run_root / "child_dialogue_promotable.json",
            child_sha256=sha256_file(run_root / "child_dialogue_promotable.json"))
        expanded_validation = validate_expanded_dialogue(
            expanded, original_parent=root, promotable_child=promotable_child)
        if not expanded_validation["valid"]:
            raise RuntimeError("expanded_dialogue_materialization_failed:" +
                               ",".join(expanded_validation["errors"]))
        atomic_json(expanded_path, expanded)
        atomic_text(expanded_markdown_path, expanded_dialogue_markdown(expanded))
        complete_manifest = complete_dialogue_manifest(
            expanded_dialogue=expanded, validation=expanded_validation,
            expanded_json_path=expanded_path,
            expanded_json_sha256=sha256_file(expanded_path),
            expanded_markdown_path=expanded_markdown_path,
            expanded_markdown_sha256=sha256_file(expanded_markdown_path))
        atomic_json(run_root / "complete_dialogue_manifest.json", complete_manifest)
        result = {"completed": True, "planner_attempts": planner_attempts,
                  "compiled_round_plan": compiled, "generated": generated,
                  "contexts": contexts, "local_results": local_results,
                  "realizer_attempts": realizer_attempts, "realized": realized,
                  "validator_output": validator_output, "validator_run": validator_run,
                  "closure": closure, "durable": durable, "memory_final": memory_final,
                  "transactions": transactions, "promotion": promotion,
                  "growth_audit": growth_audit,
                  "growth_profile_admission": growth_profile_admission,
                  "full_child": full_child, "promotable_child": promotable_child,
                  "promotion_manifest": promotion_manifest,
                  "expanded_dialogue": expanded,
                  "expanded_dialogue_validation": expanded_validation,
                  "complete_dialogue_manifest": complete_manifest}
    except Exception as error:
        if failure_stage is None:
            failure_stage = "Live chain runtime"
        result = {**result, "completed": False, "failure_stage": failure_stage,
                  "error": f"{type(error).__name__}:{error}"}
        atomic_json(run_root / "failure.json", result)
    finally:
        usage = aggregate_v11_usage(caller.records)
        atomic_json(run_root / "api_usage.json", usage)
        if ledger.events.is_file():
            shutil.copyfile(ledger.events, run_root / "request_ledger.jsonl")
        else:
            atomic_text(run_root / "request_ledger.jsonl", "")

    post_integrity = _freeze_guard(freeze)
    compiled = result.get("compiled_round_plan") or {}
    realized = result.get("realized") or []
    contexts = result.get("contexts") or {}
    closure = result.get("closure") or {}
    durable = result.get("durable") or {}
    full_child = result.get("full_child") or {"scene_count": 0, "turn_count": 0}
    promotable = result.get("promotable_child") or {"scene_count": 0, "turn_count": 0}
    expanded_validation = result.get("expanded_dialogue_validation") or {"valid": False}
    journal_rows = []
    journal_path = run_root / "run_commit_journal.jsonl"
    if journal_path.is_file():
        journal_rows = [json.loads(line) for line in journal_path.read_text().splitlines()
                        if line.strip()]
    transaction_audit = _audit_transaction_memory_consistency(
        memory=result.get("memory_final") or memory_initial,
        transactions=result.get("transactions") or [], journal_rows=journal_rows)
    atomic_json(run_root / "reports/transaction_memory_consistency_audit.json",
                transaction_audit)
    usage = aggregate_v11_usage(caller.records)
    progress = execution_progress_v11(
        result, parent_tail_refs=parent_tail_refs)
    promotable_minimum = int((compiled.get("scene_budget") or {}).get("minimum") or 5)
    generation_ready = bool(
        result.get("completed") and promotable.get("scene_count", 0) >= promotable_minimum
        and usage["roles"]["round_validator"]["physical_requests"] >= 1
        and usage["evidence_committer_calls"] == 0
        and post_integrity["valid"] and expanded_validation["valid"])
    growth_profile_admission = result.get("growth_profile_admission")
    growth_profile_ready = bool(
        growth_profile_admission.get("profile_ready")) if isinstance(
            growth_profile_admission, Mapping) else growth_profile is None
    fresh_ready = bool(
        generation_ready and len(durable.get("accepted_updates") or []) >= 1
        and transaction_audit["valid"] and transaction_audit["committed_count"] >= 1
        and growth_profile_ready)
    dialogue_ready = generation_ready
    memory_growth_ready = bool(
        len(durable.get("accepted_updates") or []) >= 1
        and transaction_audit["valid"] and transaction_audit["committed_count"] >= 1)
    if not result.get("completed"):
        conclusion = "C. 运行链路失败，保留失败产物且不得自动重跑"
    elif (not post_integrity["valid"] or usage["evidence_committer_calls"] != 0
          or not transaction_audit["valid"]):
        conclusion = "E. 历史完整性或请求纪律失败，本次结果无效"
    elif (generation_ready and growth_profile_ready and
          (promotable.get("turn_count") or 0) > 19):
        conclusion = "A. 完整通过，可以开始设计 Round 2 前置验证"
    elif generation_ready:
        conclusion = "D. 机械完成但语义质量未优于 v10，禁止进入 Round 2"
    else:
        conclusion = "B. 机械通过但有局部语义问题，修复后才能考虑第二次 canary"
    summary = {
        "report_version": "slim_continuous_expansion_v11_live_report_v1",
        "run_root": str(run_root), "status": "completed" if result.get("completed") else "failed",
        "failure_stage": result.get("failure_stage"), "failure_error": result.get("error"),
        "conclusion": conclusion,
        "planned_scene_count": progress["planned_scene_count"],
        "realized_scene_count": progress["realized_scene_count"],
        "realized_turn_count": progress["realized_turn_count"],
        "full_scene_count": full_child.get("scene_count", 0),
        "full_turn_count": full_child.get("turn_count", 0),
        "promotable_scene_count": promotable.get("scene_count", 0),
        "promotable_turn_count": promotable.get("turn_count", 0),
        "complete_dialogue_materialized": bool(expanded_validation.get("valid")),
        "complete_dialogue_turn_count": expanded_validation.get("turn_count", 0),
        "qa_source_ready": bool(expanded_validation.get("qa_source_ready")),
        "parent_tail_retained": progress["parent_tail_retained"],
        "same_round_text_included": progress["same_round_text_included"],
        "scene_dag": result.get("compiled_round_plan", {}).get("scene_alias_to_ref"),
        "round_decision": (result.get("validator_output") or {}).get("round_decision"),
        "validator_scene_reviews": (result.get("validator_output") or {}).get("scene_reviews") or [],
        "accepted_scene_refs": closure.get("accepted_scene_refs") or [],
        "rejected_scene_refs": closure.get("rejected_scene_refs") or [],
        "durable_update_counts": {"accepted": len(durable.get("accepted_updates") or []),
                                  "rejected": len(durable.get("rejected_updates") or []),
                                  "audit_only": len(durable.get("audit_only_observations") or [])},
        "binder_allocation_count": len(durable.get("accepted_updates") or []),
        "transaction_count": len(result.get("transactions") or []),
        "prepared_count": sum(row.get("status") == "PREPARED" for row in journal_rows),
        "committed_count": sum(row.get("status") == "COMMITTED" for row in journal_rows),
        "transaction_memory_consistency": transaction_audit,
        "growth_profile": deepcopy(dict(growth_profile)) if growth_profile else None,
        "dialogue_first_participants": dialogue_first_participants,
        "authoritative_growth_audit": result.get("growth_audit"),
        "growth_profile_admission": growth_profile_admission,
        "growth_profile_ready": growth_profile_ready,
        "durable_update_policy": durable_update_policy,
        "dialogue_ready": dialogue_ready,
        "memory_growth_ready": memory_growth_ready,
        "round_validator_prompt_budget": result.get("round_validator_prompt_budget"),
        "api_usage": usage, "post_run_integrity": post_integrity,
        "generation_ready": generation_ready, "fresh_e2e_ready": fresh_ready,
        "qa_pipeline_ready": False, "round_2_executed": False,
        "questioner_calls": 0, "distractor_calls": 0, "solver_calls": 0,
        "qa_replay_calls": 0, "strategy_calls": 0, "training_calls": 0,
        "initial_structure_extractor_calls": 0, "evidence_committer_calls": 0,
        "history_modified": not post_integrity["checks"].get("historical_manifests", False),
        "v10_baseline": {"scene_count": 3, "turn_count": 19},
        "artifacts": {"full_child": str(run_root / "child_dialogue_full.json"),
                      "promotable_child": str(run_root / "child_dialogue_promotable.json"),
                      "world_memory_final": str(run_root / "world_memory_final.json"),
                      "round_validator_output": str(run_root / "round_validator_output.json"),
                      "round_validator_prompt_budget": str(
                          run_root / "round_validator_prompt_budget.json"),
                      "expanded_dialogue": str(
                          run_root / "expanded_dialogue_round_1.json"),
                      "complete_dialogue_manifest": str(
                          run_root / "complete_dialogue_manifest.json"),
                      "durable_update_projection_audit": str(
                          run_root / "durable_update_projection_audit.json")},
    }
    atomic_json(run_root / "reports/SLIM_CONTINUOUS_EXPANSION_V11_LIVE_CANARY_REPORT.json", summary)
    atomic_text(run_root / "reports/SLIM_CONTINUOUS_EXPANSION_V11_LIVE_CANARY_REPORT.md", _markdown(summary))
    atomic_text(run_root / "reports/LIVE_CANARY_SEMANTIC_AUDIT.md", _semantic_markdown(summary))
    atomic_json(run_root / "reports/scene_semantic_matrix.json", summary["validator_scene_reviews"])
    atomic_json(run_root / "reports/post_run_provenance.json", {
        "freeze": freeze, "post_run_integrity": post_integrity,
        "api_usage": usage, "run_tree": hash_tree(run_root),
        "self_referential_run_tree_note": "computed before this report was published",
    })
    atomic_json(run_root / "run_status.json", {
        "status": summary["status"], "failure_stage": summary["failure_stage"],
        "conclusion": conclusion, "generation_ready": generation_ready,
        "fresh_e2e_ready": fresh_ready, "qa_pipeline_ready": False,
        "completed_at": _now(), "api_usage": usage})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-use v11 live continuous-expansion canary")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    runner = Path(__file__).resolve()
    preflight = run_live_preflight_b2_v11(
        runner_path=runner, input_path=args.input, run_root=args.run_root,
        api_config_path=args.api_config, tokenizer_path=args.tokenizer)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["valid"] else 2
    if not preflight["valid"]:
        print(json.dumps({"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    random.seed(ORCHESTRATION_SEED)
    summary = execute_live_canary(
        preflight=preflight, input_path=args.input.resolve(),
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
