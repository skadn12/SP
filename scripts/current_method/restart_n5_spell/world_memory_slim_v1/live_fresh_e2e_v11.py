"""Mock-only v11 Hybrid runtime; no live transport is imported or callable."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .action_availability_v11 import compile_action_availability_view
from .append_only_memory import apply_scene_commit
from .context_compiler_v2 import compile_scene_context_v2
from .continuous_round_plan_v1 import (
    build_scene_dependency_dag, compile_continuous_round_plan,
    validate_continuous_round_plan,
)
from .durable_update_compiler import compile_durable_updates
from .local_scene_gate_v11 import (
    dependency_unavailable_result, validate_local_scene_v11,
)
from .mock_hybrid_backends import DeterministicHybridMock
from .narrative_order import build_narrative_order_manifest
from .narrative_action_contracts import migrate_legacy_action_hints_to_narrative_actions
from .placement_choice_registry import (
    compile_placement_choice_registry, compile_source_anchor_registry,
    migrate_legacy_round_plan_source_placement,
)
from .promotion_v2 import promote_scenes_v2
from .review_world_view import build_review_world_view, validate_review_world_view
from .round_expansion_validator_v1 import (
    build_round_validator_input, run_round_validator_with_retry,
)
from .scene_decision_closure import compute_scene_decision_closure
from .transactional_binder import (
    build_allocation_plan, commit_scene_transaction,
    validate_scene_local_evidence_refs,
)
from .turn_budget import compile_turn_budget


V11_RUNNER_VERSION = "slim_continuous_expansion_hybrid_runtime_v11_b1"
V11_RUNTIME_MODE = "mock_only_zero_api"


def hybrid_request_budget() -> dict[str, Any]:
    return {
        "request_budget_version": "hybrid_request_budget_v1",
        "normal_min": {"scene_count": 5, "physical_calls": 7},
        "normal_max": {"scene_count": 7, "physical_calls": 9},
        "normal_formula": "1 planner + N realizer + 1 round_validator",
        "worst_case": {"scene_count": 7, "physical_calls": 26,
                       "formula": "3 + (7 * 3) + 2"},
        "role_attempt_limits": {"planner": 3, "realizer_per_scene": 3,
                                "round_validator": 2, "evidence_committer": 0},
        "token_accounting_categories": ["planner", "realizer", "round_validator"],
        "committer_physical_calls": 0, "budget_ready": True,
    }


def run_mock_hybrid_v11(
    *, complete_dialogue: Mapping[str, Any], planning_state: Mapping[str, Any],
    anchor_registry: Mapping[str, Any], action_menu: Mapping[str, Any],
    planner_output: Mapping[str, Any], authoritative_memory: Mapping[str, Any],
    run_root: Path, case: str = "all_accept",
) -> dict[str, Any]:
    """Execute a deterministic full local chain and, if needed, real disk transactions."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    backend = DeterministicHybridMock(case=case, planner_output=planner_output)
    source_anchors = compile_source_anchor_registry(anchor_registry)
    placement_choices = compile_placement_choice_registry(
        complete_dialogue=complete_dialogue, anchor_registry=anchor_registry)
    frozen_planner_input = {
        "planning_state_sha256": planning_state.get("planning_state_sha256"),
        "complete_dialogue_sha256": complete_dialogue.get("complete_dialogue_sha256"),
        "anchor_registry_version": anchor_registry.get("anchor_registry_version"),
        "scene_budget": {"minimum": 5, "target": 6, "maximum": 7},
    }
    model_plan = backend.planner(frozen_planner_input)
    # Phase B1's frozen mock fixture predates the live v11 source/placement
    # split.  Migrate it explicitly inside this mock-only compatibility runner;
    # the live Planner validator never accepts or auto-migrates the old shape.
    if any("historical_anchor_id" in scene or "placement" in scene
           for scene in model_plan.get("scenes") or []):
        model_plan = migrate_legacy_round_plan_source_placement(
            model_plan, placement_choice_id="PC_TAIL_AFTER")
    if any("candidate_update_hints" in scene
           for scene in model_plan.get("scenes") or []):
        model_plan = migrate_legacy_action_hints_to_narrative_actions(
            model_plan, action_menu=action_menu)
    entities = {str(row.get("entity_ref")) for row in planning_state.get("entities") or []}
    action_availability = compile_action_availability_view(
        planning_state=planning_state)
    plan_validation = validate_continuous_round_plan(
        model_plan, source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        action_availability_view=action_availability,
        authoritative_entity_refs=entities)
    if not plan_validation["valid"]:
        return _failed("planner_validation", backend, plan_validation)
    compiled = compile_continuous_round_plan(
        model_plan,
        source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        round_index=int(complete_dialogue.get("next_round_index") or 2))
    dag = build_scene_dependency_dag(compiled)
    narrative = build_narrative_order_manifest(
        compiled_plan=compiled, complete_dialogue=complete_dialogue)
    generated: dict[str, dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    local_results: dict[str, dict[str, Any]] = {}
    parent_text = [str(row.get("text") or "") for row in complete_dialogue.get("turns") or []]
    prior_text: list[str] = []
    for scene in compiled.get("scenes") or []:
        ref = str(scene["scene_ref"])
        hard_predecessors = [str(edge["predecessor_scene_ref"])
                             for edge in scene.get("dependency_edges") or []
                             if edge.get("hard_dependency")]
        blockers = [value for value in hard_predecessors
                    if (local_results.get(value) or {}).get("status") != "local_gate_passed"]
        if blockers:
            local_results[ref] = dependency_unavailable_result(scene, blockers)
            continue
        context = compile_scene_context_v2(
            scene=scene, complete_dialogue=complete_dialogue,
            source_anchor_registry=source_anchors,
            placement_choice_registry=placement_choices,
            planning_state=planning_state,
            same_round_scenes=generated)
        budget = compile_turn_budget(scene)
        output = backend.realizer(scene=scene, context=context, turn_budget=budget)
        gate = validate_local_scene_v11(
            scene=scene, scene_output=output, context_manifest=context,
            turn_budget=budget, complete_dialogue=complete_dialogue,
            source_anchor_registry=source_anchors,
            placement_choice_registry=placement_choices,
            generated_scenes=generated,
            parent_utterances=parent_text, previous_child_utterances=prior_text)
        contexts[ref], local_results[ref] = context, gate
        if gate["valid"]:
            output.update({key: deepcopy(scene.get(key)) for key in (
                "predecessor_scene_refs", "dependency_scene_refs", "branch_role")})
            generated[ref] = output
            prior_text.extend(str(row["text"]) for row in output["dialogue_turns"])
    realized = [dict(generated[str(scene["scene_ref"])] ,
                     scene_alias=scene.get("scene_alias"))
                for scene in compiled.get("scenes") or []
                if str(scene["scene_ref"]) in generated]
    # Restore metadata required by review/closure without permitting model-owned refs.
    scene_meta = {str(row["scene_ref"]): row for row in compiled.get("scenes") or []}
    for row in realized:
        meta = scene_meta[str(row["scene_ref"])]
        row.update({key: deepcopy(meta.get(key)) for key in (
            "source_anchor_id", "placement_choice_id", "placement",
            "linked_existing_refs",
            "primary_action", "secondary_action", "narrative_action_contracts")})
    review_view = build_review_world_view(
        complete_dialogue=complete_dialogue, planning_state=planning_state,
        compiled_round_plan=compiled, realized_scenes=realized,
        scene_contexts=contexts, scene_dependency_dag=dag,
        narrative_order_manifest=narrative)
    review_validation = validate_review_world_view(review_view)
    if not review_validation["valid"]:
        return _failed("review_world_view", backend, review_validation)
    validator_input = build_round_validator_input(
        round_blueprint=compiled, review_world_view=review_view,
        realized_scenes=realized, scene_dependency_dag=dag,
        structural_gate_results=local_results)
    validator_run = run_round_validator_with_retry(
        backend_call=lambda prompt, attempt: backend.validator(
            prompt=prompt, attempt=attempt, realized_scenes=realized,
            compiled_plan=compiled), validator_input=validator_input,
        realized_scenes=realized, review_world_view=review_view,
        local_gate_results=local_results)
    if not validator_run["completed"]:
        return _failed("round_validator_output", backend, validator_run)
    validator_output = validator_run["valid_output"]
    closure = compute_scene_decision_closure(
        realized_scenes=realized, local_gate_results=local_results,
        validator_output=validator_output, scene_dependency_dag=dag)
    durable = compile_durable_updates(
        candidates=validator_output["durable_updates"], compiled_round_plan=compiled,
        scene_decision_closure=closure, realized_scenes=realized,
        authoritative_memory=authoritative_memory, planning_state=planning_state,
        action_menu=action_menu)
    durable["audit_only_observations"].extend({
        "candidate_update_key": row.get("observation_key"),
        "disposition": "audit_only_unhinted_update",
        "reason": "round_validator_unhinted_semantic_observation",
        "candidate": deepcopy(dict(row)),
    } for row in validator_output.get("semantic_review_observations") or [])
    memory = deepcopy(dict(authoritative_memory))
    transactions = []
    committed_proposal_refs: set[str] = set()
    updates_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for update in durable["accepted_updates"]:
        updates_by_scene[str(update["source_scene_refs"][0])].append(update)
    ordered_scene_refs = [str(row["scene_ref"]) for row in compiled.get("scenes") or []
                          if str(row["scene_ref"]) in updates_by_scene]
    for scene_ref in ordered_scene_refs:
        updates = updates_by_scene[scene_ref]
        proposals = [deepcopy(update["compiled_proposal"]) for update in updates]
        scene = scene_meta[scene_ref]
        dialogue = generated[scene_ref]["dialogue_turns"]
        evidence_by_proposal = {
            str(update["compiled_proposal"]["proposal_ref"]):
                list(update["evidence_turn_refs"])
            for update in updates}
        evidence = validate_scene_local_evidence_refs(
            scene_ref=scene_ref, dialogue_turns=dialogue,
            accepted_evidence_by_proposal=evidence_by_proposal)
        allocation = build_allocation_plan(
            root_id=str(memory["root_id"]), round_index=int(compiled["round_index"]),
            scene_index=int(scene_ref.split(":S")[1]), compiled_proposals=proposals,
            high_water_marks=memory.get("high_water_marks") or {})
        claims = [{"proposal_ref": proposal_ref, "realization_status": "realized",
                   "candidate_evidence_turn_refs": refs}
                  for proposal_ref, refs in evidence_by_proposal.items()]
        committer_adapter = {
            "proposal_reviews": [{"proposal_ref": proposal_ref,
                "review_decision": "confirmed", "verification_status": "semantically_verified",
                "accepted_evidence_turn_refs": refs,
                "reason_code": "round_validator_confirmed"}
                for proposal_ref, refs in evidence_by_proposal.items()],
            "scope_status": "clean", "world_consistency_status": "clean",
        }
        memory = apply_scene_commit(
            memory, scene_ref=scene_ref, round_index=int(compiled["round_index"]),
            scene_index=int(scene_ref.split(":S")[1]), compiled_proposals=proposals,
            proposal_claims=claims, committer_output=committer_adapter,
            allocation_plan=allocation,
            evidence_final_turn_refs=evidence["evidence_final_turn_refs"],
            evidence_text_sha256=evidence["evidence_text_sha256"],
            previously_committed_proposal_refs=sorted(committed_proposal_refs))
        transactions.append(commit_scene_transaction(
            run_root=run_root, run_id="v11_phase_b1_mock", scene_plan=scene,
            proposal_fingerprints=[str(proposal["proposal_fingerprint"])
                                   for proposal in proposals],
            allocation_plan=allocation,
            artifacts={f"transaction_snapshots/{scene_ref.replace(':', '_')}.json": memory}))
        committed_proposal_refs.update(evidence_by_proposal)
    promotion = promote_scenes_v2(
        realized_scenes=realized, local_gate_results=local_results,
        validator_output=validator_output, scene_decision_closure=closure,
        durable_update_compile_result=durable,
        narrative_order_manifest=narrative)
    return {
        "v11_runner_version": V11_RUNNER_VERSION, "runtime_mode": V11_RUNTIME_MODE,
        "completed": True, "failure_stage": None,
        "planner_validation": plan_validation, "compiled_round_plan": compiled,
        "scene_dependency_dag": dag, "narrative_order_manifest": narrative,
        "scene_contexts": contexts, "local_scene_gate_results": local_results,
        "realized_scenes": realized, "review_world_view": review_view,
        "round_validator_input": validator_input, "round_validator_run": validator_run,
        "round_validator_output": validator_output, "scene_decision_closure": closure,
        "durable_update_compile_result": durable, "transactions": transactions,
        "memory_final": memory, "promotion_v2_result": promotion,
        "request_budget": hybrid_request_budget(), "mock_calls": dict(backend.calls),
        "mock_trace": backend.trace, "api_calls": 0, "network_calls": 0,
        "evidence_committer_calls": 0,
    }


def _failed(stage: str, backend: DeterministicHybridMock,
            detail: Mapping[str, Any]) -> dict[str, Any]:
    return {"v11_runner_version": V11_RUNNER_VERSION, "runtime_mode": V11_RUNTIME_MODE,
            "completed": False, "failure_stage": stage, "detail": deepcopy(dict(detail)),
            "mock_calls": dict(backend.calls), "mock_trace": backend.trace,
            "api_calls": 0, "network_calls": 0, "evidence_committer_calls": 0}


def main() -> int:
    raise SystemExit("v11_phase_b1_runner_is_mock_library_only; use live_preflight_v11")


if __name__ == "__main__":
    main()
