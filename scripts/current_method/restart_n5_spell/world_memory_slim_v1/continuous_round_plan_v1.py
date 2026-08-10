"""Anchor-first long-round blueprint using shared narrative actions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from .action_availability_v11 import (
    compile_action_availability_view, validate_action_availability_view,
)
from .narrative_action_contracts import (
    NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION,
)
from .narrative_action_continuation import (
    NARRATIVE_ACTION_CONTINUATION_VERSION, action_slot_ref,
    resolved_contract_subset, validate_and_resolve_round_actions,
)
from .placement_choice_registry import (
    resolve_placement_choice, resolve_source_anchor,
)


CONTINUOUS_ROUND_PLAN_VERSION = "continuous_round_plan_v4"
CONTINUOUS_ROUND_COMPILER_VERSION = "continuous_round_plan_compiler_v4"
SCENE_DEPENDENCY_DAG_VERSION = "scene_dependency_dag_v1"
SCENE_BUDGET_V11 = {"minimum": 5, "target": 6, "maximum": 7}
ADDED_TURN_BUDGET_V11 = {"minimum": 30, "target": 36, "maximum": 45}
PLANNER_OUTPUT_TOKEN_BUDGET_V11 = 10_240
PLANNER_OUTPUT_SAFETY_MARGIN_TOKENS_V11 = 1_024
PLACEMENT_LINK_TYPES = frozenset({
    "direct_continuation", "causal_followup", "consequence_followup",
    "backstory_expansion",
    "missing_event_fill", "time_skip", "location_shift",
    "related_character_shift", "parallel_thread", "connected_new_entity",
    "connected_new_event", "external_constraint", "information_transfer",
    "task_followup", "relationship_followup", "state_followup",
})
DEPENDENCY_TYPES = frozenset({
    "continues", "causes", "reveals", "motivates", "contradicts",
    "resolves", "opens", "parallel_to", "informs", "impacts",
})
HARD_DEPENDENCY_TYPES = DEPENDENCY_TYPES - {"parallel_to"}
_PROGRAM_FIELDS = frozenset({
    "scene_ref", "session_ref", "turn_ref", "turn_refs", "proposal_ref",
    "claim_ref", "operation", "operation_class",
    "pre_state", "post_state", "current_state", "evidence_turn_ref",
    "evidence_turn_refs", "final_id", "memory_ref", "event_ref",
    "thread_ref", "handoff_ref", "authoritative_version",
    "historical_anchor_id", "anchor_id", "placement", "after_turn_ref",
    "before_turn_ref", "preceding_known_context",
    "following_context_constraint_only", "narrative_order",
    "action_id", "bundle_id", "candidate_update_hints", "hint_mode",
})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _find_program_fields(value: Any, path: str = "$model") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            narrative_existing_ref = (
                key == "thread_ref" and
                (".primary_action.arguments" in path or
                 ".secondary_action.arguments" in path))
            if key in _PROGRAM_FIELDS and not narrative_existing_ref:
                errors.append(f"continuous_plan_program_owned:{path}.{key}")
            errors.extend(_find_program_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_program_fields(child, f"{path}[{index}]"))
    return errors


def validate_continuous_round_plan(
    plan: Any, *, source_anchor_registry: Mapping[str, Any],
    placement_choice_registry: Mapping[str, Any],
    authoritative_entity_refs: set[str], action_menu: Mapping[str, Any] | None = None,
    known_world_refs: set[str] | None = None,
    action_availability_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_top = {
        "round_plan_version", "scene_budget", "target_added_turns", "round_goal",
        "starting_situation", "central_development", "secondary_development",
        "ending_direction", "main_thread", "side_threads", "continued_old_threads",
        "new_connected_threads", "new_connected_events",
        "relationship_developments", "event_consequences",
        "state_developments", "information_flow", "difficulty_contributions",
        "scene_order", "branch_entry_points", "branch_merge_points",
        "cross_scene_impacts", "ending_open_threads", "scenes",
    }
    if not isinstance(plan, Mapping) or set(plan) != expected_top:
        return {"valid": False, "errors": ["continuous_round_top_schema"],
                "round_plan_version": CONTINUOUS_ROUND_PLAN_VERSION}
    if plan.get("round_plan_version") != CONTINUOUS_ROUND_PLAN_VERSION:
        errors.append("continuous_round_plan_version")
    errors.extend(_find_program_fields(plan))
    if dict(plan.get("scene_budget") or {}) != SCENE_BUDGET_V11:
        errors.append("continuous_round_scene_budget")
    if dict(plan.get("target_added_turns") or {}) != ADDED_TURN_BUDGET_V11:
        errors.append("continuous_round_added_turn_budget")
    for key in ("round_goal", "starting_situation", "central_development",
                "secondary_development", "ending_direction",
                "main_thread"):
        if not _text(plan.get(key)):
            errors.append(f"continuous_round_text:{key}")
    for key in ("side_threads", "continued_old_threads", "new_connected_threads",
                "new_connected_events", "relationship_developments",
                "event_consequences", "state_developments", "information_flow",
                "difficulty_contributions", "branch_entry_points",
                "branch_merge_points", "cross_scene_impacts", "ending_open_threads"):
        if not isinstance(plan.get(key), list) or not all(
                _text(value) for value in plan.get(key) or []):
            errors.append(f"continuous_round_list:{key}")
    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or not 5 <= len(scenes) <= 7:
        errors.append("continuous_round_scene_count")
        scenes = scenes if isinstance(scenes, list) else []
    if list(plan.get("scene_order") or []) != [
            str(scene.get("scene_alias") or "") for scene in scenes
            if isinstance(scene, Mapping)]:
        errors.append("continuous_round_scene_order")
    source_anchor_by_id = {
        str(row.get("source_anchor_id")): row
        for row in source_anchor_registry.get("source_anchors") or []}
    placement_choice_ids = {
        str(row.get("placement_choice_id"))
        for row in placement_choice_registry.get("choices") or []}
    del action_menu  # accepted only so historical offline callers fail on schema, not signature
    availability_is_explicit = action_availability_view is not None
    if action_availability_view is None:
        # Compatibility for local fixtures. Live callers pass a view compiled
        # directly from the authoritative Planning State.
        fallback_state = {
            "root_id": source_anchor_registry.get("root_id"),
            "entities": [{"entity_ref": ref, "entity_type": "person"}
                         for ref in sorted(authoritative_entity_refs)],
            "explicit_relations": [], "interaction_links": [],
            "active_events": [], "open_threads": [], "tasks": [],
            "key_states": [], "knowledge_asymmetries": [],
        }
        field_to_state = {
            "linked_relation_refs": ("explicit_relations", "relation"),
            "linked_event_refs": ("active_events", "event"),
            "linked_task_refs": ("tasks", "task"),
            "linked_thread_refs": ("open_threads", "thread"),
            "linked_state_refs": ("key_states", "state"),
            "linked_information_refs": ("knowledge_asymmetries", "knowledge"),
        }
        seen: set[tuple[str, str]] = set()
        for anchor in source_anchor_by_id.values():
            for field, (state_field, record_type) in field_to_state.items():
                for ref in anchor.get(field) or []:
                    key = (state_field, str(ref))
                    if key not in seen:
                        fallback_state[state_field].append(
                            {"memory_ref": str(ref), "record_type": record_type})
                        seen.add(key)
            for ref in anchor.get("linked_interaction_refs") or []:
                key = ("interaction_links", str(ref))
                if key not in seen:
                    fallback_state["interaction_links"].append(
                        {"interaction_link_ref": str(ref)})
                    seen.add(key)
        action_availability_view = compile_action_availability_view(
            planning_state=fallback_state)
    availability_validation = validate_action_availability_view(
        action_availability_view)
    errors.extend(availability_validation["errors"])
    if known_world_refs is None:
        known_world_refs = set(authoritative_entity_refs)
        for anchor in source_anchor_by_id.values():
            for field in ("linked_entity_refs", "linked_interaction_refs",
                          "linked_relation_refs", "linked_event_refs", "linked_task_refs",
                          "linked_thread_refs", "linked_state_refs"):
                known_world_refs.update(str(value) for value in anchor.get(field) or [])
    aliases: list[str] = []
    scene_fields = {
        "scene_alias", "source_anchor_id", "placement_choice_id", "link_type",
        "linked_existing_refs", "bridge_reason", "new_development", "goal",
        "speaker_participants", "relationship_dynamic", "interaction_beats",
        "continuity_requirements",
        "predecessor_scene_aliases", "dependency_edges",
        "transition_to_next_scene", "branch_role", "merge_target_scene_alias",
        "impact_target_refs", "primary_action", "secondary_action", "desired_outcome",
    }
    for index, scene in enumerate(scenes, 1):
        prefix = f"scene:{index}"
        if not isinstance(scene, Mapping):
            errors.append(f"{prefix}:schema")
            continue
        actual_scene_fields = set(scene)
        if actual_scene_fields != scene_fields:
            errors.append(f"{prefix}:schema")
            errors.extend(f"{prefix}:missing_required_field:{field}"
                          for field in sorted(scene_fields - actual_scene_fields))
            errors.extend(f"{prefix}:unexpected_field:{field}"
                          for field in sorted(actual_scene_fields - scene_fields))
            # Continue validating every present field so one exact-schema
            # omission cannot mask independent selection/graph errors.
        alias = str(scene.get("scene_alias") or "")
        aliases.append(alias)
        if not alias or alias in aliases[:-1]:
            errors.append(f"{prefix}:scene_alias")
        source_anchor_id = str(scene.get("source_anchor_id") or "")
        source_anchor = source_anchor_by_id.get(source_anchor_id)
        if source_anchor is None:
            errors.append(f"{prefix}:source_anchor_unknown:{source_anchor_id}")
        placement_choice_id = str(scene.get("placement_choice_id") or "")
        if placement_choice_id not in placement_choice_ids:
            errors.append(f"{prefix}:placement_choice_unknown:{placement_choice_id}")
        link_type = str(scene.get("link_type") or "")
        if link_type not in PLACEMENT_LINK_TYPES:
            errors.append(f"{prefix}:link_type")
        elif source_anchor is not None and link_type not in set(
                source_anchor.get("allowed_link_types") or []):
            errors.append(f"{prefix}:source_anchor_link_type:{link_type}")
        for key in ("bridge_reason", "new_development", "goal",
                    "relationship_dynamic",
                    "transition_to_next_scene", "desired_outcome"):
            if not _text(scene.get(key)):
                errors.append(f"{prefix}:text:{key}")
        linked = scene.get("linked_existing_refs")
        link_fields = {"entity_refs", "relation_refs", "event_refs",
                       "task_refs", "thread_refs", "state_refs"}
        if (not isinstance(linked, Mapping) or set(linked) != link_fields or
                not all(isinstance(linked.get(key), list) for key in link_fields) or
                not any(linked.get(key) for key in link_fields)):
            errors.append(f"{prefix}:linked_existing_refs")
        speakers = scene.get("speaker_participants")
        if not isinstance(speakers, list) or not 2 <= len(speakers) <= 4:
            errors.append(f"{prefix}:speaker_count")
            speakers = speakers if isinstance(speakers, list) else []
        speaker_refs: list[str] = []
        for speaker in speakers:
            if (not isinstance(speaker, Mapping) or
                    set(speaker) != {"participant_ref", "scene_role", "intents"}):
                errors.append(f"{prefix}:speaker_schema")
                continue
            ref = str(speaker.get("participant_ref") or "")
            speaker_refs.append(ref)
            if ref not in authoritative_entity_refs:
                errors.append(f"{prefix}:non_authoritative_speaker:{ref}")
            if speaker.get("scene_role") not in {"primary", "supporting"}:
                errors.append(f"{prefix}:speaker_role:{ref}")
            intents = speaker.get("intents")
            if not isinstance(intents, list) or not 1 <= len(intents) <= 3 or not all(
                    _text(value) for value in intents):
                errors.append(f"{prefix}:speaker_intents:{ref}")
        if len(speaker_refs) != len(set(speaker_refs)):
            errors.append(f"{prefix}:speaker_duplicate")
        if not isinstance(scene.get("interaction_beats"), list) or not 3 <= len(
                scene.get("interaction_beats") or []) <= 7 or not all(
                    _text(value) for value in scene.get("interaction_beats") or []):
            errors.append(f"{prefix}:interaction_beats")
        if not isinstance(scene.get("continuity_requirements"), list) or not (
                scene.get("continuity_requirements")) or not all(
                    _text(value) for value in scene.get("continuity_requirements") or []):
            errors.append(f"{prefix}:continuity_requirements")
        predecessors = scene.get("predecessor_scene_aliases")
        if not isinstance(predecessors, list) or len(predecessors) != len(set(
                str(value) for value in predecessors)):
            errors.append(f"{prefix}:predecessor_schema")
            predecessors = predecessors if isinstance(predecessors, list) else []
        for predecessor in predecessors:
            if str(predecessor) not in aliases[:-1]:
                errors.append(f"{prefix}:predecessor_not_prior:{predecessor}")
        edges = scene.get("dependency_edges")
        if not isinstance(edges, list):
            errors.append(f"{prefix}:dependency_edges")
            edges = []
        edge_predecessors: set[str] = set()
        for edge in edges:
            if (not isinstance(edge, Mapping) or
                    set(edge) != {"predecessor_scene_alias", "dependency_type"}):
                errors.append(f"{prefix}:dependency_edge_schema")
                continue
            predecessor = str(edge.get("predecessor_scene_alias") or "")
            edge_predecessors.add(predecessor)
            if predecessor not in aliases[:-1]:
                errors.append(f"{prefix}:dependency_not_prior:{predecessor}")
            if edge.get("dependency_type") not in DEPENDENCY_TYPES:
                errors.append(f"{prefix}:dependency_type")
        if set(str(value) for value in predecessors) != edge_predecessors:
            errors.append(f"{prefix}:dependency_predecessor_mismatch")
        if scene.get("branch_role") not in {
                "mainline", "parallel_branch", "character_branch",
                "event_branch", "backstory_branch"}:
            errors.append(f"{prefix}:branch_role")
        merge_alias = scene.get("merge_target_scene_alias")
        if merge_alias is not None and (not _text(merge_alias) or
                                        str(merge_alias) == alias):
            errors.append(f"{prefix}:merge_target")
        impacts = scene.get("impact_target_refs")
        if not isinstance(impacts, list) or not all(_text(value) for value in impacts):
            errors.append(f"{prefix}:impact_target_refs")
        if scene.get("branch_role") != "mainline" and not merge_alias and not impacts:
            errors.append(f"{prefix}:branch_without_merge_or_impact")
    action_resolution = validate_and_resolve_round_actions(
        scenes, known_world_refs=known_world_refs,
        authoritative_entity_refs=authoritative_entity_refs,
        action_availability_view=(action_availability_view
                                  if availability_is_explicit else None))
    errors.extend(action_resolution["errors"])
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "warnings": [], "action_continuation_validation": action_resolution,
            "action_availability_validation": availability_validation,
            "round_plan_version": CONTINUOUS_ROUND_PLAN_VERSION,
            "action_contract_registry_version": NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION,
            "action_continuation_contract_version": NARRATIVE_ACTION_CONTINUATION_VERSION,
            "scene_count": len(scenes)}


def compile_continuous_round_plan(
    plan: Mapping[str, Any], *, action_menu: Mapping[str, Any] | None = None,
    source_anchor_registry: Mapping[str, Any],
    placement_choice_registry: Mapping[str, Any], round_index: int = 1,
) -> dict[str, Any]:
    """Assign stable scene refs; Memory proposals are intentionally deferred."""
    del action_menu
    action_resolution = validate_and_resolve_round_actions(plan["scenes"])
    if not action_resolution["valid"]:
        raise ValueError("continuous_round_action_resolution_invalid:" +
                         ",".join(action_resolution["errors"]))
    resolved_slots = action_resolution["resolved_action_slots"]
    aliases = [str(scene["scene_alias"]) for scene in plan["scenes"]]
    alias_to_ref = {alias: f"R{round_index:02d}:S{index:02d}"
                    for index, alias in enumerate(aliases, 1)}
    compiled_scenes = []
    for index, model_scene in enumerate(plan["scenes"], 1):
        scene_ref = alias_to_ref[model_scene["scene_alias"]]
        source_anchor = resolve_source_anchor(
            source_anchor_registry, model_scene["source_anchor_id"])
        placement_choice = resolve_placement_choice(
            placement_choice_registry, model_scene["placement_choice_id"])
        scene = {
            "scene_ref": scene_ref,
            "goal": model_scene["goal"],
            "speaker_participants": deepcopy(model_scene["speaker_participants"]),
            "interaction_beats": list(model_scene["interaction_beats"]),
            "continuity_requirements": list(model_scene["continuity_requirements"]),
            "desired_outcome": model_scene["desired_outcome"],
            "scene_alias": model_scene["scene_alias"],
            "source_anchor_id": model_scene["source_anchor_id"],
            "source_anchor": source_anchor,
            "placement_choice_id": model_scene["placement_choice_id"],
            "placement_choice": placement_choice,
            "placement": placement_choice["placement"],
            "placement_boundary": {
                "after_turn_ref": placement_choice["after_turn_ref"],
                "before_turn_ref": placement_choice["before_turn_ref"],
            },
            "preceding_known_turn_refs": list(
                placement_choice["preceding_known_turn_refs"]),
            "following_constraint_turn_refs": list(
                placement_choice["following_constraint_turn_refs"]),
            "narrative_order_key": placement_choice["narrative_order_key"],
            "link_type": model_scene["link_type"],
            "linked_existing_refs": deepcopy(model_scene["linked_existing_refs"]),
            "bridge_reason": model_scene["bridge_reason"],
            "new_development": model_scene["new_development"],
            "predecessor_scene_refs": [alias_to_ref[value] for value in
                                       model_scene["predecessor_scene_aliases"]],
            "dependency_scene_refs": [alias_to_ref[edge["predecessor_scene_alias"]]
                                      for edge in model_scene["dependency_edges"]],
            "dependency_edges": [{
                "predecessor_scene_ref": alias_to_ref[
                    edge["predecessor_scene_alias"]],
                "dependency_type": edge["dependency_type"],
                "hard_dependency": edge["dependency_type"] in HARD_DEPENDENCY_TYPES,
            } for edge in model_scene["dependency_edges"]],
            "transition_to_next_scene": model_scene["transition_to_next_scene"],
            "branch_role": model_scene["branch_role"],
            "merge_target_scene_ref": (
                alias_to_ref.get(str(model_scene["merge_target_scene_alias"]))
                if model_scene["merge_target_scene_alias"] else None),
            "impact_target_refs": list(model_scene["impact_target_refs"]),
            "primary_action": deepcopy(resolved_slots[
                action_slot_ref(index, "primary_action")]),
            "secondary_action": deepcopy(resolved_slots.get(
                action_slot_ref(index, "secondary_action"))),
            "planned_primary_action": deepcopy(model_scene["primary_action"]),
            "planned_secondary_action": deepcopy(model_scene["secondary_action"]),
            "narrative_action_contracts": resolved_contract_subset([
                resolved_slots[action_slot_ref(index, "primary_action")],
                resolved_slots.get(action_slot_ref(index, "secondary_action"))]),
            "compiled_proposals": [],
            "durable_update_compilation_stage": "post_round_validation",
        }
        compiled_scenes.append(scene)
    compiled = {
        "root_id": source_anchor_registry.get("root_id"),
        "round_index": round_index,
        "round_arc": {"starting_situation": plan["starting_situation"],
                      "central_tension": plan["central_development"],
                      "ending_direction": plan["ending_direction"]},
        "scenes": compiled_scenes,
        "continuous_round_plan_compiler_version": CONTINUOUS_ROUND_COMPILER_VERSION,
        "round_plan_version": CONTINUOUS_ROUND_PLAN_VERSION,
        "scene_budget": deepcopy(plan["scene_budget"]),
        "target_added_turns": deepcopy(plan["target_added_turns"]),
        "round_blueprint": {key: deepcopy(plan[key]) for key in (
            "round_goal", "starting_situation", "central_development",
            "secondary_development", "ending_direction", "main_thread",
            "side_threads", "continued_old_threads", "new_connected_threads",
            "new_connected_events", "relationship_developments",
            "event_consequences", "state_developments", "information_flow",
            "difficulty_contributions", "scene_order", "branch_entry_points",
            "branch_merge_points", "cross_scene_impacts", "ending_open_threads")},
        "scene_alias_to_ref": alias_to_ref,
        "resolved_action_slots": deepcopy(resolved_slots),
        "action_continuation_contract_version": NARRATIVE_ACTION_CONTINUATION_VERSION,
    }
    compiled["compiled_round_plan_sha256"] = _sha(compiled)
    return compiled


def audit_planner_output_size_v11(
    plan: Mapping[str, Any], *, token_budget: int = PLANNER_OUTPUT_TOKEN_BUDGET_V11,
    safety_margin_tokens: int = PLANNER_OUTPUT_SAFETY_MARGIN_TOKENS_V11,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Audit compact JSON against the frozen output ceiling.

    Offline unit tests can use the deterministic character estimate.  Freeze
    reports may additionally provide the actual configured tokenizer so the
    ledger-facing completion budget has a model-specific safety measurement.
    """
    serialized = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    estimated_tokens = max(1, (len(serialized) + 3) // 4)
    measured_tokens = (int(token_counter(serialized))
                       if token_counter is not None else None)
    budgeted_tokens = (measured_tokens if measured_tokens is not None
                       else estimated_tokens)
    remaining = int(token_budget) - budgeted_tokens
    return {
        "planner_output_budget_version": "planner_output_budget_v11_v1",
        "serialized_utf8_bytes": len(serialized.encode("utf-8")),
        "serialized_characters": len(serialized),
        "estimated_tokens_conservative": estimated_tokens,
        "measured_tokens": measured_tokens,
        "token_count_method": ("configured_tokenizer" if measured_tokens is not None
                               else "deterministic_char_div_4_estimate"),
        "budgeted_token_count": budgeted_tokens,
        "max_output_tokens": int(token_budget),
        "safety_margin_tokens_required": int(safety_margin_tokens),
        "estimated_safety_margin_tokens": remaining,
        "valid": remaining >= int(safety_margin_tokens),
    }


def build_scene_dependency_dag(compiled_plan: Mapping[str, Any]) -> dict[str, Any]:
    nodes = [str(scene.get("scene_ref")) for scene in compiled_plan.get("scenes") or []]
    edges = []
    for scene in compiled_plan.get("scenes") or []:
        for edge in scene.get("dependency_edges") or []:
            edges.append({
                "from_scene_ref": str(edge["predecessor_scene_ref"]),
                "to_scene_ref": str(scene["scene_ref"]),
                "dependency_type": str(edge["dependency_type"]),
                "hard_dependency": bool(edge["hard_dependency"]),
            })
    result = {
        "scene_dependency_dag_version": SCENE_DEPENDENCY_DAG_VERSION,
        "root_id": compiled_plan.get("root_id"),
        "round_index": compiled_plan.get("round_index"),
        "nodes": nodes, "edges": edges,
    }
    result["scene_dependency_dag_sha256"] = _sha(result)
    return result
