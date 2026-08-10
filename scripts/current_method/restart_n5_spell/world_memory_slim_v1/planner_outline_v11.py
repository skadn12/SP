"""Slim Planner outline contract with deterministic v4 compatibility compilation."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .action_availability_v11 import validate_action_availability_view
from .continuous_round_plan_v1 import (
    ADDED_TURN_BUDGET_V11,
    CONTINUOUS_ROUND_PLAN_VERSION,
    DEPENDENCY_TYPES,
    PLACEMENT_LINK_TYPES,
    SCENE_BUDGET_V11,
    compile_continuous_round_plan,
    validate_continuous_round_plan,
)
from .narrative_action_continuation import validate_and_resolve_round_actions
from .placement_choice_registry import resolve_source_anchor
from .round_participant_registry import (
    ALIAS_SPEAKER_REFERENCE_FIELDS, EXISTING_SPEAKER_FIELDS,
    NEW_SPEAKER_DECLARATION_FIELDS,
    compile_round_participant_registry, legacy_scene_speaker_bindings,
)


PLANNER_OUTLINE_SCHEMA_VERSION_V11 = (
    "continuous_round_outline_v10_authoritative_connection_refs"
)
PLANNER_OUTLINE_COMPILER_VERSION_V11 = (
    "continuous_round_outline_compiler_v7_canonical_participant_names"
)

_TOP_FIELDS = frozenset({
    "round_goal", "starting_situation", "ending_direction", "scenes",
})
_SCENE_FIELDS = frozenset({
    "scene_alias", "source_anchor_id", "placement_choice_id", "link_type",
    "bridge_reason", "new_development", "goal", "speaker_participants",
    "interaction_beats", "dependencies", "primary_action", "secondary_action",
    "desired_outcome",
})
_DEPENDENCY_FIELDS = frozenset({"predecessor_scene_alias", "dependency_type"})
_SPEAKER_FIELDS = EXISTING_SPEAKER_FIELDS
_PROGRAM_OWNED_FIELDS = frozenset({
    "round_plan_version", "scene_budget", "target_added_turns", "scene_order",
    "central_development", "secondary_development", "main_thread", "side_threads",
    "continued_old_threads", "new_connected_threads", "new_connected_events",
    "relationship_developments", "event_consequences", "state_developments",
    "information_flow", "difficulty_contributions", "branch_entry_points",
    "branch_merge_points", "cross_scene_impacts", "ending_open_threads",
    "linked_existing_refs", "relationship_dynamic", "continuity_requirements",
    "predecessor_scene_aliases", "dependency_edges", "transition_to_next_scene",
    "branch_role", "merge_target_scene_alias", "impact_target_refs",
    "scene_ref", "proposal_ref", "operation", "pre_state", "post_state",
    "final_id", "memory_ref", "event_ref", "handoff_ref",
})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized_name(value: Any) -> str:
    return " ".join(unicodedata.normalize(
        "NFKC", str(value)).casefold().split())


_ACTION_PARTICIPANT_LIST_FIELDS = frozenset({
    "affected_participants", "responsible_participants", "participants",
    "participant_refs",
})
_ACTION_PARTICIPANT_SINGLE_FIELDS = frozenset({
    "informed_participant", "uninformed_participant", "sender", "receiver",
    "mistaken_participant", "from_participant", "to_participant", "target",
})


def _resolve_action_participant_aliases(
    action: Any, alias_to_round_ref: Mapping[str, str],
) -> Any:
    """Replace only typed participant arguments; narrative alias fields remain prose IDs."""
    if not isinstance(action, Mapping) or action.get("action_mode") != "new":
        return deepcopy(action)
    result = deepcopy(dict(action))
    arguments = result.get("arguments")
    if not isinstance(arguments, dict):
        return result
    for field in _ACTION_PARTICIPANT_LIST_FIELDS:
        values = arguments.get(field)
        if isinstance(values, list):
            arguments[field] = [alias_to_round_ref.get(str(value), value)
                                for value in values]
    for field in _ACTION_PARTICIPANT_SINGLE_FIELDS:
        value = arguments.get(field)
        if value is not None:
            arguments[field] = alias_to_round_ref.get(str(value), value)
    return result


def _unique_name_map(rows: Sequence[tuple[str, str]]) -> dict[str, str]:
    candidates: dict[str, list[str]] = {}
    for display_name, ref in rows:
        key = _normalized_name(display_name)
        if key and ref:
            candidates.setdefault(key, []).append(ref)
    return {key: values[0] for key, values in candidates.items()
            if len(set(values)) == 1}


def _canonicalize_action_participant_names(
    action: Any, *, name_to_ref: Mapping[str, str],
) -> Any:
    """Resolve only unique participant labels; ambiguous prose remains invalid."""
    if not isinstance(action, Mapping) or action.get("action_mode") != "new":
        return deepcopy(action)
    result = deepcopy(dict(action))
    arguments = result.get("arguments")
    if not isinstance(arguments, dict):
        return result

    def resolve(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return name_to_ref.get(_normalized_name(value), value)

    for field in _ACTION_PARTICIPANT_LIST_FIELDS:
        values = arguments.get(field)
        if isinstance(values, list):
            arguments[field] = [resolve(value) for value in values]
    for field in _ACTION_PARTICIPANT_SINGLE_FIELDS | {"connection_target_ref"}:
        if field in arguments and arguments.get(field) is not None:
            arguments[field] = resolve(arguments[field])
    connected = arguments.get("connected_existing")
    if isinstance(connected, dict) and connected.get("ref_type") == "person":
        connected["ref"] = resolve(connected.get("ref"))
    return result


def _canonical_action_scenes(
    scenes: Sequence[Any], *, authoritative_entities: Sequence[Mapping[str, Any]],
    participant_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    authoritative_names = _unique_name_map([
        (str(row.get("display_name") or ""),
         str(row.get("entity_ref") or row.get("entity_id") or ""))
        for row in authoritative_entities
    ])
    participant_names = {
        str(row.get("participant_alias") or ""): str(row.get("display_name") or "")
        for row in participant_registry.get("participants") or []
        if isinstance(row, Mapping)
    }
    available_aliases = participant_registry.get("available_aliases_by_scene") or {}
    result: list[dict[str, Any]] = []
    for raw_scene in scenes:
        scene = deepcopy(dict(raw_scene)) if isinstance(raw_scene, Mapping) else {}
        alias = str(scene.get("scene_alias") or "")
        local_names = _unique_name_map([
            (participant_names.get(value, ""), value)
            for value in available_aliases.get(alias) or []
        ])
        name_to_ref = {**authoritative_names, **local_names}
        for slot in ("primary_action", "secondary_action"):
            scene[slot] = _canonicalize_action_participant_names(
                scene.get(slot), name_to_ref=name_to_ref)
        result.append(scene)
    return result


def _availability_with_round_participants(
    availability: Mapping[str, Any], participant_refs: Sequence[str],
) -> dict[str, Any]:
    """Create an internal validation view without changing the frozen Planner menu."""
    result = deepcopy(dict(availability))
    refs = result.get("typed_reference_registry")
    if not isinstance(refs, dict):
        return result
    additions = {str(value) for value in participant_refs}
    for field in ("participant_refs", "state_target_refs", "all_typed_refs"):
        refs[field] = sorted(set(str(value) for value in refs.get(field) or []) | additions)
    for row in (result.get("actions") or {}).values():
        if isinstance(row, dict) and "allowed_participant_refs" in row:
            row["allowed_participant_refs"] = sorted(
                set(str(value) for value in row.get("allowed_participant_refs") or []) |
                additions)
    result["compiler_round_participant_extension"] = sorted(additions)
    return result


def _find_program_fields(value: Any, path: str = "$planner") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            narrative_thread_ref = (
                key == "thread_ref" and
                (".primary_action.arguments" in path or
                 ".secondary_action.arguments" in path)
            )
            if key in _PROGRAM_OWNED_FIELDS and not narrative_thread_ref:
                errors.append(f"planner_outline_program_owned:{path}.{key}")
            errors.extend(_find_program_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_program_fields(child, f"{path}[{index}]"))
    return errors


def _action_scene_view(scenes: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        value = scene if isinstance(scene, Mapping) else {}
        dependencies = value.get("dependencies")
        dependencies = dependencies if isinstance(dependencies, list) else []
        rows.append({
            "scene_alias": value.get("scene_alias"),
            "predecessor_scene_aliases": [
                str(row.get("predecessor_scene_alias") or "")
                for row in dependencies if isinstance(row, Mapping)
            ],
            "primary_action": value.get("primary_action"),
            "secondary_action": value.get("secondary_action"),
            "desired_outcome": value.get("desired_outcome"),
        })
    return rows


def validate_planner_outline_v11(
    outline: Any, *, source_anchor_registry: Mapping[str, Any],
    placement_choice_registry: Mapping[str, Any],
    authoritative_entity_refs: set[str],
    action_availability_view: Mapping[str, Any],
    authoritative_entities: Sequence[Mapping[str, Any]] | None = None,
    round_index: int = 1,
) -> dict[str, Any]:
    """Validate the model-owned outline and aggregate all observable errors."""
    errors: list[str] = []
    if not isinstance(outline, Mapping):
        return {
            "valid": False, "errors": ["planner_outline_not_object"],
            "scene_count": 0,
            "planner_outline_schema_version": PLANNER_OUTLINE_SCHEMA_VERSION_V11,
        }

    actual_top = set(outline)
    errors.extend(f"planner_outline:missing_required_field:{field}"
                  for field in sorted(_TOP_FIELDS - actual_top))
    errors.extend(f"planner_outline:unexpected_field:{field}"
                  for field in sorted(actual_top - _TOP_FIELDS))
    errors.extend(_find_program_fields(outline))
    for field in ("round_goal", "starting_situation", "ending_direction"):
        if not _text(outline.get(field)):
            errors.append(f"planner_outline:text:{field}")

    source_by_id = {
        str(row.get("source_anchor_id")): row
        for row in source_anchor_registry.get("source_anchors") or []
    }
    placement_ids = {
        str(row.get("placement_choice_id"))
        for row in placement_choice_registry.get("choices") or []
    }
    availability_validation = validate_action_availability_view(
        action_availability_view)
    errors.extend(availability_validation["errors"])

    raw_scenes = outline.get("scenes")
    scenes = raw_scenes if isinstance(raw_scenes, list) else []
    if not isinstance(raw_scenes, list):
        errors.append("planner_outline:scenes_not_array")
    if not 5 <= len(scenes) <= 7:
        errors.append("planner_outline:scene_count")

    prior_aliases: list[str] = []
    for index, raw_scene in enumerate(scenes, 1):
        prefix = f"scene:{index}"
        if not isinstance(raw_scene, Mapping):
            errors.append(f"{prefix}:not_object")
            prior_aliases.append("")
            continue
        actual_fields = set(raw_scene)
        errors.extend(f"{prefix}:missing_required_field:{field}"
                      for field in sorted(_SCENE_FIELDS - actual_fields))
        errors.extend(f"{prefix}:unexpected_field:{field}"
                      for field in sorted(actual_fields - _SCENE_FIELDS))

        alias = str(raw_scene.get("scene_alias") or "")
        expected_alias = f"s{index}"
        if alias != expected_alias:
            errors.append(f"{prefix}:scene_alias_expected:{expected_alias}:{alias}")
        if alias in prior_aliases:
            errors.append(f"{prefix}:scene_alias_duplicate:{alias}")

        source_id = str(raw_scene.get("source_anchor_id") or "")
        source = source_by_id.get(source_id)
        if source is None:
            errors.append(f"{prefix}:source_anchor_unknown:{source_id}")
        placement_id = str(raw_scene.get("placement_choice_id") or "")
        if placement_id not in placement_ids:
            errors.append(f"{prefix}:placement_choice_unknown:{placement_id}")
        link_type = str(raw_scene.get("link_type") or "")
        if link_type not in PLACEMENT_LINK_TYPES:
            errors.append(f"{prefix}:link_type:{link_type}")
        elif source is not None and link_type not in set(
                source.get("allowed_link_types") or []):
            errors.append(f"{prefix}:source_anchor_link_type:{link_type}")

        for field in ("bridge_reason", "new_development", "goal", "desired_outcome"):
            if not _text(raw_scene.get(field)):
                errors.append(f"{prefix}:text:{field}")

        speakers = raw_scene.get("speaker_participants")
        speakers = speakers if isinstance(speakers, list) else []
        if not 2 <= len(speakers) <= 4:
            errors.append(f"{prefix}:speaker_count")
        speaker_refs: list[str] = []
        for speaker_index, speaker in enumerate(speakers, 1):
            speaker_path = f"{prefix}:speaker:{speaker_index}"
            if (not isinstance(speaker, Mapping) or
                    set(speaker) not in {
                        _SPEAKER_FIELDS, NEW_SPEAKER_DECLARATION_FIELDS,
                        ALIAS_SPEAKER_REFERENCE_FIELDS,
                    }):
                errors.append(f"{speaker_path}:schema")
                continue
            ref = str(speaker.get("participant_ref") or
                      speaker.get("participant_alias") or "")
            speaker_refs.append(ref)
            if set(speaker) == _SPEAKER_FIELDS and ref not in authoritative_entity_refs:
                errors.append(f"{speaker_path}:non_authoritative:{ref}")
            if speaker.get("scene_role") not in {"primary", "supporting"}:
                errors.append(f"{speaker_path}:role")
            intents = speaker.get("intents")
            if not isinstance(intents, list) or not 1 <= len(intents) <= 3 or not all(
                    _text(value) for value in intents):
                errors.append(f"{speaker_path}:intents")
        if len(speaker_refs) != len(set(speaker_refs)):
            errors.append(f"{prefix}:speaker_duplicate")

        beats = raw_scene.get("interaction_beats")
        if not isinstance(beats, list) or not 3 <= len(beats) <= 7 or not all(
                _text(value) for value in beats):
            errors.append(f"{prefix}:interaction_beats")

        dependencies = raw_scene.get("dependencies")
        dependencies = dependencies if isinstance(dependencies, list) else []
        if not isinstance(raw_scene.get("dependencies"), list):
            errors.append(f"{prefix}:dependencies_not_array")
        dependency_aliases: list[str] = []
        for dependency_index, dependency in enumerate(dependencies, 1):
            path = f"{prefix}:dependency:{dependency_index}"
            if not isinstance(dependency, Mapping) or set(dependency) != _DEPENDENCY_FIELDS:
                errors.append(f"{path}:schema")
                continue
            predecessor = str(dependency.get("predecessor_scene_alias") or "")
            dependency_aliases.append(predecessor)
            if predecessor not in prior_aliases:
                errors.append(f"{path}:predecessor_not_prior:{predecessor}")
            if dependency.get("dependency_type") not in DEPENDENCY_TYPES:
                errors.append(f"{path}:dependency_type:{dependency.get('dependency_type')}")
        if len(dependency_aliases) != len(set(dependency_aliases)):
            errors.append(f"{prefix}:dependency_duplicate")
        if index == 1 and dependencies:
            errors.append("scene:1:first_scene_dependencies_must_be_empty")
        for slot in ("primary_action", "secondary_action"):
            action = raw_scene.get(slot)
            if index == 1 and isinstance(action, Mapping) and action.get(
                    "action_mode") == "continuation":
                errors.append(f"action:S01.{slot}:first_scene_action_must_be_new")
                errors.append(
                    f"action:S01.{slot}:continuation_source_not_prior:"
                    f"{action.get('continuation_of')}")
        prior_aliases.append(alias)

    entity_rows = list(authoritative_entities or ({
        "entity_ref": ref, "display_name": ref, "entity_type": "person"}
        for ref in sorted(authoritative_entity_refs)))
    participant_registry = compile_round_participant_registry(
        scenes=[scene for scene in scenes if isinstance(scene, Mapping)],
        authoritative_entities=entity_rows, round_index=round_index)
    errors.extend(participant_registry["errors"])

    canonical_scenes = _canonical_action_scenes(
        scenes, authoritative_entities=entity_rows,
        participant_registry=participant_registry)
    action_scenes = _action_scene_view(canonical_scenes)
    typed_refs = action_availability_view.get("typed_reference_registry") or {}
    action_resolution = validate_and_resolve_round_actions(
        action_scenes,
        known_world_refs=set(str(value) for value in
                             typed_refs.get("all_typed_refs") or []),
        authoritative_entity_refs=authoritative_entity_refs,
        action_availability_view=action_availability_view,
        round_participant_aliases_by_scene=participant_registry[
            "available_aliases_by_scene"],
    )
    errors.extend(action_resolution["errors"])
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": [],
        "scene_count": len(scenes),
        "action_continuation_validation": action_resolution,
        "action_availability_validation": availability_validation,
        "planner_outline_schema_version": PLANNER_OUTLINE_SCHEMA_VERSION_V11,
        "model_owned_top_fields": sorted(_TOP_FIELDS),
        "model_owned_scene_fields": sorted(_SCENE_FIELDS),
        "round_participant_registry_validation": {
            "valid": participant_registry["valid"],
            "errors": participant_registry["errors"],
            "warnings": participant_registry["warnings"],
        },
    }


def _linked_refs(anchor: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, list[str]]:
    result = {
        "entity_refs": list(anchor.get("linked_entity_refs") or []),
        "relation_refs": list(anchor.get("linked_relation_refs") or []),
        "event_refs": list(anchor.get("linked_event_refs") or []),
        "task_refs": list(anchor.get("linked_task_refs") or []),
        "thread_refs": list(anchor.get("linked_thread_refs") or []),
        "state_refs": list(anchor.get("linked_state_refs") or []),
    }
    if not any(result.values()):
        result["entity_refs"] = [
            str(row.get("participant_ref") or "")
            for row in scene.get("speaker_participants") or []
            if row.get("participant_ref")
        ]
    return result


def compile_planner_outline_v11(
    outline: Mapping[str, Any], *, source_anchor_registry: Mapping[str, Any],
    placement_choice_registry: Mapping[str, Any], planning_state: Mapping[str, Any],
    action_availability_view: Mapping[str, Any], round_index: int = 1,
) -> dict[str, Any]:
    """Compile model semantics into the existing downstream v4 plan shape."""
    authoritative_entity_refs = {
        str(row.get("entity_ref") or "")
        for row in planning_state.get("entities") or [] if row.get("entity_ref")
    }
    validation = validate_planner_outline_v11(
        outline,
        source_anchor_registry=source_anchor_registry,
        placement_choice_registry=placement_choice_registry,
        authoritative_entity_refs=authoritative_entity_refs,
        action_availability_view=action_availability_view,
        authoritative_entities=list(planning_state.get("entities") or []),
        round_index=round_index,
    )
    if not validation["valid"]:
        raise ValueError("planner_outline_invalid:" + ",".join(validation["errors"]))

    participant_registry = compile_round_participant_registry(
        scenes=outline["scenes"],
        authoritative_entities=list(planning_state.get("entities") or []),
        round_index=round_index)
    if not participant_registry["valid"]:
        raise RuntimeError("round_participant_registry_invalid:" +
                           ",".join(participant_registry["errors"]))
    alias_to_round_ref = participant_registry["alias_to_round_ref"]
    round_participant_refs = [str(row["participant_ref"])
                              for row in participant_registry["participants"]]
    internal_action_availability = _availability_with_round_participants(
        action_availability_view, round_participant_refs)
    canonical_outline_scenes = _canonical_action_scenes(
        outline["scenes"],
        authoritative_entities=list(planning_state.get("entities") or []),
        participant_registry=participant_registry)
    scenes: list[dict[str, Any]] = []
    for raw_scene, scene in zip(outline["scenes"], canonical_outline_scenes):
        anchor = resolve_source_anchor(
            source_anchor_registry, str(scene["source_anchor_id"]))
        dependencies = deepcopy(list(scene["dependencies"]))
        predecessor_aliases = [
            str(row["predecessor_scene_alias"]) for row in dependencies
        ]
        continuity = [
            f"Preserve the source facts and knowledge boundary of {scene['source_anchor_id']}.",
            f"Use only context authorized by placement choice {scene['placement_choice_id']}.",
        ]
        if predecessor_aliases:
            continuity.append(
                "Continue from declared predecessor scenes: " +
                ", ".join(predecessor_aliases) + ".")
        scenes.append({
            "scene_alias": scene["scene_alias"],
            "source_anchor_id": scene["source_anchor_id"],
            "placement_choice_id": scene["placement_choice_id"],
            "link_type": scene["link_type"],
            "linked_existing_refs": _linked_refs(anchor, scene),
            "bridge_reason": scene["bridge_reason"],
            "new_development": scene["new_development"],
            "goal": scene["goal"],
            "speaker_participants": legacy_scene_speaker_bindings(
                participant_registry, str(raw_scene["scene_alias"])),
            "relationship_dynamic": scene["bridge_reason"],
            "interaction_beats": list(scene["interaction_beats"]),
            "continuity_requirements": continuity,
            "predecessor_scene_aliases": predecessor_aliases,
            "dependency_edges": dependencies,
            "transition_to_next_scene": (
                "Reach this scene's desired outcome and preserve declared dependencies: " +
                str(scene["desired_outcome"])),
            "branch_role": "mainline",
            "merge_target_scene_alias": None,
            "impact_target_refs": [],
            "primary_action": _resolve_action_participant_aliases(
                scene["primary_action"], alias_to_round_ref),
            "secondary_action": _resolve_action_participant_aliases(
                scene["secondary_action"], alias_to_round_ref),
            "desired_outcome": scene["desired_outcome"],
        })

    internal_plan = {
        "round_plan_version": CONTINUOUS_ROUND_PLAN_VERSION,
        "scene_budget": deepcopy(SCENE_BUDGET_V11),
        "target_added_turns": deepcopy(ADDED_TURN_BUDGET_V11),
        "round_goal": outline["round_goal"],
        "starting_situation": outline["starting_situation"],
        "central_development": outline["round_goal"],
        "secondary_development": outline["ending_direction"],
        "ending_direction": outline["ending_direction"],
        "main_thread": outline["round_goal"],
        "side_threads": [],
        "continued_old_threads": [],
        "new_connected_threads": [],
        "new_connected_events": [],
        "relationship_developments": [],
        "event_consequences": [],
        "state_developments": [],
        "information_flow": [],
        "difficulty_contributions": [],
        "scene_order": [str(scene["scene_alias"]) for scene in outline["scenes"]],
        "branch_entry_points": [],
        "branch_merge_points": [],
        "cross_scene_impacts": [],
        # Persistent ending threads are derived after dialogue review. This is
        # an internal non-authoritative placeholder, not a model or Memory claim.
        "ending_open_threads": [],
        "scenes": scenes,
    }
    internal_validation = validate_continuous_round_plan(
        internal_plan,
        source_anchor_registry=source_anchor_registry,
        placement_choice_registry=placement_choice_registry,
        authoritative_entity_refs=(authoritative_entity_refs | {
            str(row["participant_ref"])
            for row in participant_registry["participants"]}),
        action_availability_view=internal_action_availability,
    )
    if not internal_validation["valid"]:
        raise RuntimeError(
            "planner_outline_internal_invariant:" +
            ",".join(internal_validation["errors"]))
    compiled = compile_continuous_round_plan(
        internal_plan,
        source_anchor_registry=source_anchor_registry,
        placement_choice_registry=placement_choice_registry,
        round_index=round_index,
    )
    for compiled_scene in compiled["scenes"]:
        alias = str(compiled_scene["scene_alias"])
        compiled_scene["speaker_participants"] = deepcopy(
            participant_registry["scene_bindings"][alias])
    compiled["round_participant_registry"] = deepcopy(participant_registry)
    compiled.update({
        "planner_outline_schema_version": PLANNER_OUTLINE_SCHEMA_VERSION_V11,
        "planner_outline_compiler_version": PLANNER_OUTLINE_COMPILER_VERSION_V11,
        "planner_outline_sha256": _sha(outline),
        "model_owned_round_fields": [
            "round_goal", "starting_situation", "ending_direction"],
        "program_owned_round_fields": sorted(
            set(internal_plan) - {
                "scenes", "round_goal", "starting_situation", "ending_direction"}),
        "ending_open_threads_policy": "derive_from_accepted_dialogue_after_round_review",
        "participant_persistence_policy": (
            "round_local_identity_first; authoritative entity only after round review"),
    })
    compiled["compiled_round_plan_sha256"] = _sha({
        key: value for key, value in compiled.items()
        if key != "compiled_round_plan_sha256"
    })
    return compiled


def project_internal_v4_fixture_to_outline_v11(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Project trusted offline v4 fixtures; never repair live model output."""
    return {
        "round_goal": plan.get("round_goal"),
        "starting_situation": plan.get("starting_situation"),
        "ending_direction": plan.get("ending_direction"),
        "scenes": [{
            "scene_alias": scene.get("scene_alias"),
            "source_anchor_id": scene.get("source_anchor_id"),
            "placement_choice_id": scene.get("placement_choice_id"),
            "link_type": scene.get("link_type"),
            "bridge_reason": scene.get("bridge_reason"),
            "new_development": scene.get("new_development"),
            "goal": scene.get("goal"),
            "speaker_participants": deepcopy(scene.get("speaker_participants")),
            "interaction_beats": deepcopy(scene.get("interaction_beats")),
            "dependencies": deepcopy(scene.get("dependency_edges")),
            "primary_action": deepcopy(scene.get("primary_action")),
            "secondary_action": deepcopy(scene.get("secondary_action")),
            "desired_outcome": scene.get("desired_outcome"),
        } for scene in plan.get("scenes") or []],
    }


__all__ = [
    "PLANNER_OUTLINE_COMPILER_VERSION_V11",
    "PLANNER_OUTLINE_SCHEMA_VERSION_V11",
    "compile_planner_outline_v11",
    "project_internal_v4_fixture_to_outline_v11",
    "validate_planner_outline_v11",
]
