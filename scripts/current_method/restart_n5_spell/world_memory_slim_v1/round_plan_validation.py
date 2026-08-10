"""Planner v3 schema → compile → semantic validation with retry classification."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .interaction_links import validate_scene_connectivity
from .planner_contract_v3_1 import (
    compile_allowed_proposal_bundles, compile_planner_action_view,
    validate_compiled_proposal_novelty, validate_planner_action_view,
    validate_bundle_selection_registry,
)
from .planner_v3 import compile_scene_plan_v3, validate_planner_v3


ROUND_PLAN_VALIDATOR_VERSION = "round_plan_semantic_validator_v1"


def compile_validate_round_plan(
    plan: Any, *, action_menu: Mapping[str, Any],
    authoritative_memory: Mapping[str, Any],
    interaction_links: Sequence[Mapping[str, Any]], round_index: int,
    planner_action_view: Mapping[str, Any] | None = None,
    allowed_bundles: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entity_refs = {str(row.get("entity_ref") or row.get("entity_id"))
                   for row in authoritative_memory.get("entities") or []}
    role_table = [{"entity_ref": str(row.get("entity_ref") or row.get("entity_id")),
                   "display_name": str(row.get("display_name") or
                                       row.get("entity_ref") or row.get("entity_id"))}
                  for row in authoritative_memory.get("entities") or []]
    try:
        if planner_action_view is None:
            planner_action_view = compile_planner_action_view(
                action_menu, role_table=role_table)
        if allowed_bundles is None:
            allowed_bundles = compile_allowed_proposal_bundles(action_menu)
    except (KeyError, ValueError) as error:
        return _result("local_invariant_failure",
                       [f"planner_action_projection:{error}"], schema={})
    projection_validation = validate_planner_action_view(
        action_menu, planner_action_view)
    if not projection_validation["valid"]:
        return _result("local_invariant_failure",
                       projection_validation["errors"], schema={})
    bundle_validation = validate_bundle_selection_registry(
        action_menu, allowed_bundles, planner_action_view)
    if not bundle_validation["valid"]:
        return _result("local_invariant_failure",
                       bundle_validation["errors"], schema={})
    schema = validate_planner_v3(
        plan, action_menu=action_menu, authoritative_entity_refs=entity_refs,
        planner_action_view=planner_action_view, allowed_bundles=allowed_bundles)
    if not schema["valid"]:
        infeasible_only = all(
            "action_speaker_incompatible" in error or
            "temporary_or_unknown_speaker" in error
            for error in schema["errors"])
        return _result("scene_infeasible" if infeasible_only else "planner_repairable",
                       schema["errors"], schema=schema)
    try:
        compiled = compile_scene_plan_v3(
            plan, action_menu=action_menu, round_index=round_index,
            allowed_bundles=allowed_bundles)
    except (KeyError, ValueError) as error:
        message = str(error).strip("'")
        repairable_prefixes = (
            "planner_attempted_to_override_frozen_payload", "proposal_payload_missing",
            "proposal_payload_extra", "proposal_relation_type_unregistered",
            "temporary_entity_scene_presence_invalid", "proposal_entity_ref_not_allowed",
            "planner_bundle_unknown", "duplicate_round_proposal",
        )
        classification = ("planner_repairable" if message.startswith(repairable_prefixes)
                          else "local_invariant_failure")
        return _result(classification, [f"round_plan_compile:{message}"], schema=schema)

    novelty = validate_compiled_proposal_novelty(
        compiled, authoritative_memory=authoritative_memory)
    if not novelty["valid"]:
        return _result("planner_repairable", novelty["errors"],
                       schema=schema, compiled=compiled, novelty=novelty)

    action_by_id = {str(row["action_id"]): row for row in action_menu.get("actions") or []}
    active_refs = set(str(value) for value in
                      (authoritative_memory.get("active_head_index") or {}).values())
    relation_edges = []
    shared_groups = []
    for record in authoritative_memory.get("authoritative_records") or []:
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        if record.get("record_type") == "relation":
            source, target = payload.get("source_ref"), payload.get("target_ref")
            if source and target:
                relation_edges.append([str(source), str(target)])
        if record.get("record_type") in {"thread", "event", "task", "conflict"}:
            refs = list(record.get("subject_refs") or [])
            if len(refs) >= 2:
                shared_groups.append(refs)
    infeasible: list[str] = []
    local_failures: list[str] = []
    for scene in compiled["scenes"]:
        scene_ref = str(scene["scene_ref"])
        primary = [str(row["participant_ref"]) for row in scene["speaker_participants"]
                   if row["scene_role"] == "primary"]
        supporting = [str(row["participant_ref"]) for row in scene["speaker_participants"]
                      if row["scene_role"] == "supporting"]
        selected_actions = [action_by_id[str(proposal["action_id"])]
                            for proposal in scene["compiled_proposals"]]
        connectivity = validate_scene_connectivity(
            primary_refs=primary, supporting_refs=supporting,
            relation_edges=relation_edges, interaction_links=interaction_links,
            shared_memory_groups=shared_groups,
            allows_connectivity_bridge=any(bool(row.get("allows_connectivity_bridge"))
                                           for row in selected_actions))
        if not connectivity["valid"]:
            infeasible.extend(f"{scene_ref}:{error}" for error in connectivity["errors"])
        for proposal in scene["compiled_proposals"]:
            target = proposal.get("target_memory_ref")
            if target and str(target) not in active_refs:
                local_failures.append(f"{scene_ref}:frozen_action_target_not_active:{target}")
            for temporary_ref in _temporary_refs(proposal):
                if not temporary_ref.startswith(scene_ref + ":TMP:"):
                    infeasible.append(f"{scene_ref}:cross_scene_provisional_ref:{temporary_ref}")
    if local_failures:
        return _result("local_invariant_failure", local_failures,
                       schema=schema, compiled=compiled)
    if infeasible:
        return _result("scene_infeasible", infeasible, schema=schema, compiled=compiled)
    return {"valid": True, "classification": "accepted", "errors": [],
            "schema_validation": schema, "compiled_round_plan": compiled,
            "novelty_validation": novelty,
            "validator_version": ROUND_PLAN_VALIDATOR_VERSION}


def _temporary_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str) and ":TMP:" in value:
        refs.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            refs.update(_temporary_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_temporary_refs(child))
    return refs


def _result(classification: str, errors: Sequence[str], *, schema: Mapping[str, Any],
            compiled: Mapping[str, Any] | None = None,
            novelty: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"valid": False, "classification": classification,
            "errors": list(errors), "schema_validation": dict(schema),
            "compiled_round_plan": compiled,
            "novelty_validation": dict(novelty or {}),
            "validator_version": ROUND_PLAN_VALIDATOR_VERSION}
