"""Eleven-root, zero-API preflight for affordance-based Slim Planner v2."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from ..core import canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from .affordance_compiler import (
    AFFORDANCE_SCHEMA_VERSION,
    OPERATION_CATEGORIES,
    action_menu_parent_turn_ids,
    compile_delta_affordances,
    compile_selected_actions,
    validate_affordance_menu,
)
from .binder import bind_scene
from .inputs import load_locomo_parent_read_only, load_unified_root_read_only
from .interfaces import EvidenceCommitterUnavailable, run_evidence_committer
from .memory import commit_confirmed_delta, initialize_world_memory, stage_model_claims
from .prompts import (
    build_evidence_committer_prompt,
    build_natural_realizer_prompt,
    build_slim_planner_prompt,
    build_slim_planner_retry_prompt,
    estimate_prompt_tokens,
)
from .provenance import audit_legacy_import_isolation
from .schemas import (
    DIFFICULTY_STRATEGY_SCHEMA_VERSION,
    EVIDENCE_COMMITTER_SCHEMA_VERSION,
    PLANNER_MUTATION_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
    QA_REPLAY_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
    schema_sha256s,
    validate_difficulty_strategy,
    validate_evidence_committer_output,
    validate_natural_realizer_output,
    validate_planner_mutation,
    validate_qa_replay,
    validate_slim_planner_output,
    validate_world_memory,
)
from .structural import validate_planner_memory_references, validate_structural_deltas


SLIM_PIPELINE_PREFLIGHT_VERSION = "world_memory_slim_pipeline_preflight_v2"


def _memory_subset(
    memory: Mapping[str, Any], related_ids: Sequence[str],
) -> dict[str, Any]:
    ids = set(related_ids)
    records = [
        record for field in ("event_history", "current_states", "open_threads",
                             "completed_threads")
        for record in memory.get(field) or [] if record.get("memory_id") in ids
    ]
    subject_ids = {value for row in records for value in row.get("subject_ids") or []}
    return {
        "world_memory_schema_version": memory["world_memory_schema_version"],
        "root_id": memory["root_id"],
        "memory_version": memory["memory_version"],
        "entities": [row for row in memory["entities"]
                     if row["entity_id"] in subject_ids],
        "related_records": records,
    }


def _relevant_parent_turns(
    parent: Mapping[str, Any], turn_ids: Sequence[str],
) -> list[dict[str, Any]]:
    selected = set(turn_ids)
    return [dict(turn) for turn in parent["dialogue"]["turns"]
            if turn.get("turn_id") in selected]


def _schema_fixtures(
    root_id: str, parent_turn_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    qa = {
        "qa_replay_schema_version": QA_REPLAY_SCHEMA_VERSION,
        "qa_id": f"{root_id}:QA:preflight", "parent_qa_id": None,
        "generation_mode": "fresh", "question": "Offline schema fixture",
        "answer": "Offline schema fixture", "options": [],
        "evidence_turn_ids": [parent_turn_id], "difficulty_vector": {},
        "solver_results": [], "difficulty_bucket": "grounded_easy",
        "mutation_operation": None,
    }
    strategy = {
        "difficulty_strategy_schema_version": DIFFICULTY_STRATEGY_SCHEMA_VERSION,
        "operator": "open_thread", "attempts": 0, "grounded_rate": 0.0,
        "frontier_rate": 0.0, "too_hard_rate": 0.0, "bad_ground_rate": 0.0,
        "recommended_weight": 1.0,
    }
    mutation = {
        "planner_mutation_schema_version": PLANNER_MUTATION_SCHEMA_VERSION,
        "source_qa_id": qa["qa_id"], "missing_dependency": "offline_only",
        "requested_graph_mutation": "offline_only", "status": "pending",
    }
    return qa, strategy, mutation


def _fixture_plan(action_menu: Mapping[str, Any]) -> dict[str, Any]:
    chosen: list[Mapping[str, Any]] = []
    seen_groups: set[str] = set()
    for action in action_menu["actions"]:
        group = str(action["mutually_exclusive_group"])
        if group in seen_groups:
            continue
        chosen.append(action)
        seen_groups.add(group)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        raise ValueError("preflight_requires_two_non_conflicting_actions")
    threads = list(dict.fromkeys(str(row["thread_id"]) for row in chosen))
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "selected_memory_threads": threads,
        "scene_budget": dict(action_menu["scene_budget"]),
        "scene_goals": [{
            "goal": f"Offline structural scene {index}.",
            "participants": list(action["eligible_participants"]),
            "selected_action_ids": [str(action["action_id"])],
        } for index, action in enumerate(chosen, 1)],
    }


def _operation_class_dry_runs(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise create/update/transition validators on in-memory schema fixtures."""
    root_id = str(memory["root_id"])
    first_record = copy.deepcopy(memory["event_history"][0])
    entities = [row["entity_id"] for row in memory["entities"]]
    overlay = copy.deepcopy(dict(memory))
    update_record = copy.deepcopy(first_record)
    update_record.update({
        "memory_id": f"{root_id}:PREFLIGHT_STATE",
        "event_identity_key": f"{root_id}:preflight:authoritative_state",
        "record_type": "state", "value": "baseline_state", "status": "active",
        "semantic_status": "source_grounded",
    })
    overlay["current_states"].append(update_record)
    transition_applicable = len(entities) >= 2
    if transition_applicable:
        task_record = copy.deepcopy(first_record)
        task_record.update({
            "memory_id": f"{root_id}:PREFLIGHT_TASK",
            "event_identity_key": f"{root_id}:preflight:authoritative_task",
            "record_type": "responsibility", "value": "assigned",
            "subject_ids": [entities[0]], "status": "active",
            "semantic_status": "source_grounded",
        })
        overlay["current_states"].append(task_record)
    menu = compile_delta_affordances(
        overlay, root_identity=root_id, maximum_actions=20)

    results: dict[str, Any] = {}
    for category in ("create", "update", "transition"):
        action = next((row for row in menu["actions"]
                       if row["operation_class"] == category), None)
        if action is None:
            results[category] = {
                "applicable": category != "transition" or transition_applicable,
                "pass": category == "transition" and not transition_applicable,
                "reason": "no_structurally_grounded_action",
            }
            continue
        partner = next((row for row in menu["actions"]
                        if row["action_id"] != action["action_id"]
                        and row["mutually_exclusive_group"] !=
                        action["mutually_exclusive_group"]), None)
        if partner is None:
            results[category] = {"applicable": True, "pass": False,
                                 "reason": "selection_partner_missing"}
            continue
        selected = [action, partner]
        plan = {
            "planner_schema_version": PLANNER_SCHEMA_VERSION,
            "selected_memory_threads": list(dict.fromkeys(
                str(row["thread_id"]) for row in selected)),
            "scene_budget": dict(menu["scene_budget"]),
            "scene_goals": [{
                "goal": f"{category} validator fixture",
                "participants": list(row["eligible_participants"]),
                "selected_action_ids": [str(row["action_id"])],
            } for row in selected],
        }
        compiled = compile_selected_actions(plan, menu)
        delta = compiled["scene_goals"][0]["proposed_graph_deltas"][0]
        validation = validate_structural_deltas(
            overlay, [delta], expected_root_id=root_id, action_menu=menu)
        results[category] = {
            "applicable": True,
            "pass": validation["structurally_valid"],
            "operation": delta["operation"],
            "errors": validation["errors"],
        }
    return results


def _root_preflight(
    loaded: Mapping[str, Any], *, token_counter: Callable[[str], int] | None,
) -> dict[str, Any]:
    root = loaded["root"]
    parent = loaded["parent"]
    root_id = str(root["root_id"])
    memory = initialize_world_memory(root, parent)
    memory_validation = validate_world_memory(memory)
    action_menu = compile_delta_affordances(
        memory, root_identity=root_id, mutation_requests=[])
    affordance_validation = validate_affordance_menu(
        action_menu, memory=memory, expected_root_id=root_id)
    relevant_turn_ids = action_menu_parent_turn_ids(action_menu)
    relevant_turns = _relevant_parent_turns(parent, relevant_turn_ids)
    planner_prompt = build_slim_planner_prompt(
        parent_turns=relevant_turns, action_menu=action_menu,
        role_table=memory["entities"])
    retry_prompt = build_slim_planner_retry_prompt(
        action_menu=action_menu, role_table=memory["entities"],
        errors=["selected_action_unknown:A9999"])
    prompt_tokens = (token_counter(planner_prompt) if token_counter else
                     estimate_prompt_tokens(planner_prompt))
    retry_tokens = (token_counter(retry_prompt) if token_counter else
                    estimate_prompt_tokens(retry_prompt))

    empty_reason = None
    if not action_menu["actions"]:
        empty_reason = "no authoritative source-grounded or verified records"
    plan = _fixture_plan(action_menu)
    planner_validation = validate_slim_planner_output(
        plan, action_menu=action_menu)
    compiled = compile_selected_actions(plan, action_menu)
    references = validate_planner_memory_references(
        memory, compiled, action_menu=action_menu)
    deltas = [delta for scene in compiled["scene_goals"]
              for delta in scene["proposed_graph_deltas"]]
    structural = validate_structural_deltas(
        memory, deltas, expected_root_id=root_id, action_menu=action_menu)

    first_scene = compiled["scene_goals"][0]
    first_delta = first_scene["proposed_graph_deltas"][0]
    first_action = next(row for row in action_menu["actions"]
                        if row["action_id"] == first_delta["action_id"])
    scene_turns = _relevant_parent_turns(parent, first_action["source_turn_ids"])
    subset = _memory_subset(memory, first_scene["related_memory_ids"])
    realizer_prompt = build_natural_realizer_prompt(
        relevant_parent_turns=scene_turns, memory_subset=subset,
        scene_goal=first_scene, accepted_deltas=[first_delta], prior_scenes=[])
    speaker = first_scene["participants"][0]
    realizer_output = {
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "scene_key": first_scene["scene_key"],
        "dialogue_turns": [{
            "turn_key": "t01", "speaker": speaker,
            "text": "Offline structural preflight fixture.",
        }],
        "claimed_graph_updates": [{
            "proposed_delta_id": first_delta["proposed_delta_id"],
            "claimed_status": "realized", "candidate_evidence_turn_keys": ["t01"],
        }],
        "new_open_thread_claims": [],
    }
    realizer_validation = validate_natural_realizer_output(realizer_output)
    pending = stage_model_claims(memory, [first_delta], realizer_output)
    pending_safe = (
        pending["current_states"] == memory["current_states"]
        and pending["pending_memory_updates"][0]["semantic_status"] == "model_claimed"
        and pending["pending_memory_updates"][0]["training_eligible"] is False)
    committer_prompt = build_evidence_committer_prompt(
        proposed_delta=first_delta, dialogue_turns=realizer_output["dialogue_turns"],
        realizer_claims=realizer_output["claimed_graph_updates"],
        memory_view=subset)
    no_runner_blocked = False
    try:
        run_evidence_committer(prompt=committer_prompt, request={}, runner=None)
    except EvidenceCommitterUnavailable:
        no_runner_blocked = True
    confirmed = {
        "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
        "proposed_delta_id": first_delta["proposed_delta_id"],
        "decision": "confirmed",
        "confirmed_post_state": first_delta["intended_post_state"],
        "evidence_turn_keys": ["t01"], "reason": "Schema-only dry-run fixture.",
        "semantic_status": "semantically_verified",
    }
    committer_validation = validate_evidence_committer_output(confirmed)
    binding = bind_scene(
        root_id=root_id, round_index=1, scene_goal=first_scene,
        realizer_output=realizer_output, deltas=[first_delta],
        committer_outputs=[confirmed])
    committed = commit_confirmed_delta(
        memory, first_delta, confirmed, binding,
        scene_key=first_scene["scene_key"], round_index=1, dry_run=True)
    commit_valid = (
        committed["memory_version"] == memory["memory_version"] + 1
        and any(row["event_identity_key"] == first_delta["event_identity_key"]
                and row["semantic_status"] == "semantically_verified"
                for row in committed["current_states"])
        and memory["memory_version"] == 1)
    operation_class_checks = _operation_class_dry_runs(memory)
    qa, strategy, mutation = _schema_fixtures(
        root_id, parent["dialogue"]["turns"][0]["turn_id"])
    auxiliary_valid = all(row["valid"] for row in (
        validate_qa_replay(qa), validate_difficulty_strategy(strategy),
        validate_planner_mutation(mutation)))
    checks = {
        "world_memory_initialized": memory_validation["valid"],
        "affordance_menu_valid": affordance_validation["valid"],
        "action_menu_nonempty": bool(action_menu["actions"]),
        "relevant_memory_retrieval_valid": bool(relevant_turns),
        "planner_v2_prompt_built": bool(planner_prompt),
        "planner_v2_schema_valid": planner_validation["valid"],
        "compiled_delta_valid": bool(deltas),
        "planner_memory_references_valid": references["valid"],
        "structural_deltas_valid": structural["structurally_valid"],
        "create_update_transition_validators": all(
            row["pass"] for row in operation_class_checks.values()),
        "realizer_prompt_built": bool(realizer_prompt),
        "realizer_schema_valid": realizer_validation["valid"],
        "model_claim_not_authoritative": pending_safe,
        "evidence_committer_prompt_built": bool(committer_prompt),
        "unconfigured_committer_fails_closed": no_runner_blocked,
        "evidence_committer_schema_valid": committer_validation["valid"],
        "local_binder_dry_run_valid": bool(binding["turn_key_to_turn_id"]),
        "memory_commit_dry_run_valid": commit_valid,
        "qa_memory_schemas_initialized_only": auxiliary_valid,
        "api_ledger_zero": binding["api_ledger"]["paid_api_generation_count"] == 0,
    }
    authoritative_count = sum(
        1 for field in ("event_history", "current_states", "open_threads",
                        "completed_threads")
        for row in memory[field]
        if row["semantic_status"] in {"source_grounded", "semantically_verified"}
    )
    distribution = Counter(row["operation_class"] for row in action_menu["actions"])
    return {
        "root_id": root_id, "dataset": root["source"]["dataset"],
        "source_kind": loaded["source_kind"], "input_path": loaded["input_path"],
        "input_sha256": loaded["input_sha256"],
        "parent_session_count": root["dialogue"]["session_count"],
        "parent_turn_count": root["dialogue"]["turn_count"],
        "parent_dialogue_sha256": root["dialogue"]["dialogue_hash"],
        "authoritative_memory_record_count": authoritative_count,
        "world_memory_initial_entity_count": len(memory["entities"]),
        "executable_action_count": len(action_menu["actions"]),
        "action_distribution": {
            category: distribution.get(category, 0)
            for category in ("create", "update", "transition")},
        "planner_prompt_token_estimate": prompt_tokens,
        "planner_retry_prompt_token_estimate": retry_tokens,
        "referenced_parent_turn_count": len(relevant_turns),
        "empty_action_menu": not action_menu["actions"],
        "empty_action_menu_reason": empty_reason,
        "operation_class_validator_dry_runs": operation_class_checks,
        "planner_prompt_sha256": sha256_bytes(planner_prompt.encode()),
        "planner_retry_prompt_sha256": sha256_bytes(retry_prompt.encode()),
        "realizer_prompt_sha256": sha256_bytes(realizer_prompt.encode()),
        "evidence_committer_prompt_sha256": sha256_bytes(committer_prompt.encode()),
        "model_runner_used": False, "mock_runner_used": False, "api_calls": 0,
        "checks": checks, "preflight_pass": all(checks.values()),
        **({"locomo_compatibility_audit": loaded["compatibility_audit"]}
           if "compatibility_audit" in loaded else {}),
    }


def run_preflight_batch(
    *, cc_root_paths: Sequence[Path], locomo_source_path: Path,
    locomo_attestation_path: Path,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    paths = [path.resolve() for path in cc_root_paths]
    if len(paths) != 10 or len(set(paths)) != 10:
        raise ValueError("preflight_requires_ten_unique_cc_roots")
    source_paths = [*paths, locomo_source_path.resolve(),
                    locomo_attestation_path.resolve()]
    before = {str(path): sha256_file(path) for path in source_paths}
    loaded = [load_unified_root_read_only(path) for path in paths]
    loaded.append(load_locomo_parent_read_only(
        locomo_source_path, locomo_attestation_path,
        line_number=77, expected_identity="conv-43"))
    roots = [_root_preflight(row, token_counter=token_counter) for row in loaded]
    after = {str(path): sha256_file(path) for path in source_paths}
    with TemporaryDirectory(prefix="world-memory-slim-v2-ledger-") as temp:
        manifest_sha = sha256_bytes(canonical_json({
            "preflight_version": SLIM_PIPELINE_PREFLIGHT_VERSION,
            "root_ids": [row["root_id"] for row in roots],
        }).encode())
        request_summary = RequestLedger(Path(temp), manifest_sha, "audit").summary()
    isolation = audit_legacy_import_isolation(Path(__file__).resolve().parent)
    api_calls = sum(row["api_calls"] for row in roots)
    ready = (
        len(roots) == 11 and all(row["preflight_pass"] for row in roots)
        and before == after and api_calls == 0
        and request_summary["request_count"] == 0 and isolation["isolated"])
    return {
        "preflight_version": SLIM_PIPELINE_PREFLIGHT_VERSION,
        "affordance_schema_version": AFFORDANCE_SCHEMA_VERSION,
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "root_count": len(roots),
        "cc_root_count": sum(row["dataset"] == "conversation_chronicles"
                             for row in roots),
        "locomo_root_count": sum(row["dataset"] == "locomo" for row in roots),
        "roots": roots,
        "source_sha256_before": before, "source_sha256_after": after,
        "source_inputs_unchanged": before == after,
        "request_ledger_summary": request_summary,
        "legacy_import_isolation": isolation,
        "schema_sha256": schema_sha256s(),
        "prompt_token_estimator": (
            "provided_local_tokenizer" if token_counter else "utf8_bytes_div_4"),
        "api_calls": api_calls, "planner_calls": 0, "realizer_calls": 0,
        "evidence_committer_calls": 0, "questioner_calls": 0,
        "distractor_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
        "delta_affordance_v1_ready": ready,
        "slim_planner_v2_ready": ready,
        "compiled_delta_ready": ready,
        "planner_prompt_compaction_ready": ready,
        "slim_pipeline_v2_preflight_ready": ready,
        "generation_ready": False, "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }
