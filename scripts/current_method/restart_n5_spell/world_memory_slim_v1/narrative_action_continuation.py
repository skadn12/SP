"""Deterministic continuity contract for cross-scene narrative actions."""

from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .narrative_action_contracts import (
    NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION,
    CANONICAL_ACTION_SLOT_PATTERN, action_contract, validate_action_instance,
)


NARRATIVE_ACTION_CONTINUATION_VERSION = (
    "narrative_action_continuation_v4_round_participant_alias_refs"
)
ACTION_MODES = frozenset({"new", "continuation"})
PROGRESSION_STAGES = frozenset({
    "development", "escalation", "consolidation", "resolution",
})
_MECHANICAL_PROGRESSIONS = frozenset({
    "continue", "continue it", "continue the action", "continue the development",
    "continue development", "develop it further", "further develop it",
})
_SLOT_PATTERN = CANONICAL_ACTION_SLOT_PATTERN


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().lower().split())


def _normalized_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalized_text(value)
    if isinstance(value, Mapping):
        return {str(key): _normalized_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalized_value(row) for row in value]
    return value


def action_slot_ref(scene_index: int, role: str) -> str:
    if scene_index < 1 or role not in {"primary_action", "secondary_action"}:
        raise ValueError("narrative_action_slot_ref_invalid")
    return f"S{scene_index:02d}.{role}"


def compile_action_slot_namespace(maximum_scenes: int = 7) -> dict[str, list[str]]:
    """Return canonical prior-slot spellings; this is not an action allocator."""
    if not 1 <= maximum_scenes <= 7:
        raise ValueError("action_slot_namespace_scene_limit")
    return {
        f"S{current:02d}": [
            action_slot_ref(prior, role)
            for prior in range(1, current)
            for role in ("primary_action", "secondary_action")
        ]
        for current in range(1, maximum_scenes + 1)
    }


def compile_allowed_continuation_slot_view(
    scenes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve exact existing action slots in each scene's dependency ancestry."""
    aliases = [str(scene.get("scene_alias") or "") for scene in scenes]
    predecessors = {
        str(scene.get("scene_alias") or ""): {
            str(value) for value in scene.get("predecessor_scene_aliases") or []
        }
        for scene in scenes
    }
    slots_by_alias: dict[str, list[str]] = {}
    for index, scene in enumerate(scenes, 1):
        values = []
        for role in ("primary_action", "secondary_action"):
            if isinstance(scene.get(role), Mapping):
                values.append(action_slot_ref(index, role))
        slots_by_alias[aliases[index - 1]] = values

    def ancestors(alias: str) -> set[str]:
        found: set[str] = set()
        queue = list(predecessors.get(alias, set()))
        while queue:
            value = queue.pop(0)
            if value in found:
                continue
            found.add(value)
            queue.extend(sorted(predecessors.get(value, set())))
        return found

    scene_rows = []
    by_scene: dict[str, list[str]] = {}
    for index, alias in enumerate(aliases, 1):
        ancestry = ancestors(alias)
        allowed = [
            slot
            for prior_alias in aliases[:index - 1]
            if prior_alias in ancestry
            for slot in slots_by_alias.get(prior_alias, [])
        ]
        scene_ref = f"S{index:02d}"
        by_scene[scene_ref] = allowed
        scene_rows.append({
            "scene_ref": scene_ref, "scene_alias": alias,
            "dependency_ancestor_aliases": sorted(ancestry),
            "allowed_continuation_slots": allowed,
        })
    return {
        "allowed_continuation_slot_view_version": "allowed_continuation_slots_v11_v1",
        "canonical_slot_pattern": r"^S(0[1-7])\.(primary_action|secondary_action)$",
        "scenes": scene_rows,
        "by_scene_ref": by_scene,
    }


def _dependency_path(
    current_alias: str, source_alias: str,
    predecessors: Mapping[str, set[str]],
) -> list[str]:
    queue: list[tuple[str, list[str]]] = [(current_alias, [current_alias])]
    visited: set[str] = set()
    while queue:
        node, path = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for parent in sorted(predecessors.get(node, set())):
            candidate = [*path, parent]
            if parent == source_alias:
                return candidate
            queue.append((parent, candidate))
    return []


def _cycle_nodes(slots: Mapping[str, Mapping[str, Any]]) -> set[str]:
    edges = {ref: str(action.get("continuation_of") or "")
             for ref, action in slots.items()
             if action.get("action_mode") == "continuation"}
    active: list[str] = []
    visited: set[str] = set()
    cycles: set[str] = set()

    def walk(ref: str) -> None:
        if ref in active:
            cycles.update(active[active.index(ref):])
            return
        if ref in visited:
            return
        active.append(ref)
        target = edges.get(ref)
        if target in edges:
            walk(target)
        active.pop()
        visited.add(ref)

    for ref in edges:
        walk(ref)
    return cycles


def _argument_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_normalized_text(value)]
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _argument_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _argument_strings(child)]
    return []


def validate_and_resolve_round_actions(
    scenes: Sequence[Mapping[str, Any]], *,
    known_world_refs: set[str] | None = None,
    authoritative_entity_refs: set[str] | None = None,
    action_availability_view: Mapping[str, Any] | None = None,
    round_participant_aliases_by_scene: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Validate action slots and resolve continuations without adding DAG edges."""
    errors: list[str] = []
    details: list[dict[str, Any]] = []
    aliases = [str(scene.get("scene_alias") or "") for scene in scenes]
    alias_by_index = {index: alias for index, alias in enumerate(aliases, 1)}
    predecessors = {
        str(scene.get("scene_alias") or ""): {
            str(value) for value in scene.get("predecessor_scene_aliases") or []}
        for scene in scenes
    }
    slots: dict[str, Mapping[str, Any]] = {}
    slot_scene: dict[str, str] = {}
    slot_role: dict[str, str] = {}
    for index, scene in enumerate(scenes, 1):
        for role in ("primary_action", "secondary_action"):
            action = scene.get(role)
            if action is None and role == "secondary_action":
                continue
            ref = action_slot_ref(index, role)
            if isinstance(action, Mapping):
                slots[ref] = action
            slot_scene[ref] = alias_by_index[index]
            slot_role[ref] = role
    allowed_slot_view = compile_allowed_continuation_slot_view(scenes)

    cycles = _cycle_nodes(slots)
    for ref in sorted(cycles):
        errors.append(f"action:{ref}:continuation_cycle")
        details.append({"error": "continuation_cycle", "current_action_ref": ref})

    resolved: dict[str, dict[str, Any]] = {}
    new_signature_owner: dict[str, str] = {}
    narrative_alias_owner: dict[tuple[str, str], str] = {}
    for index, scene in enumerate(scenes, 1):
        current_alias = alias_by_index[index]
        available_round_aliases = set(str(value) for value in (
            (round_participant_aliases_by_scene or {}).get(current_alias) or []))
        for role in ("primary_action", "secondary_action"):
            action = scene.get(role)
            if action is None and role == "secondary_action":
                continue
            ref = action_slot_ref(index, role)
            validation = validate_action_instance(
                action, path=f"action:{ref}", known_world_refs=known_world_refs,
                authoritative_entity_refs=authoritative_entity_refs,
                action_availability_view=action_availability_view,
                round_participant_aliases=available_round_aliases)
            errors.extend(validation["errors"])
            if not isinstance(action, Mapping) or not validation["valid"]:
                continue
            mode = str(action.get("action_mode") or "")
            if mode == "new":
                action_type = str(action["action_type"])
                alias_field = {
                    "introduce_connected_person": "participant_alias",
                    "introduce_connected_event": "event_alias",
                }.get(action_type)
                if alias_field:
                    narrative_alias = str(action["arguments"].get(
                        alias_field) or "")
                    alias_key = (action_type, narrative_alias)
                    first_slot_owner = narrative_alias_owner.get(alias_key)
                    if narrative_alias and first_slot_owner is not None:
                        errors.append(
                            f"action:{ref}:narrative_alias_reused:"
                            f"{first_slot_owner}:{narrative_alias}")
                        details.append({
                            "error": "narrative_alias_reused",
                            "current_action_ref": ref,
                            "first_action_ref": first_slot_owner,
                            "narrative_alias": narrative_alias,
                        })
                    elif narrative_alias:
                        narrative_alias_owner[alias_key] = ref
                signature_value = {
                    "action_type": action["action_type"],
                    "arguments": _normalized_value(action["arguments"]),
                }
                signature = _canonical(signature_value)
                first = new_signature_owner.get(signature)
                if first is not None:
                    errors.append(f"action:{ref}:redundant_independent_action:{first}")
                    details.append({
                        "error": "redundant_independent_action",
                        "current_action_ref": ref, "first_action_ref": first,
                        "normalized_signature": signature_value,
                    })
                else:
                    new_signature_owner[signature] = ref
                resolved[ref] = {
                    "action_ref": ref, "action_mode": "new", "continuation_of": None,
                    "root_action_ref": ref,
                    "resolved_action_type": action["action_type"],
                    "resolved_arguments": deepcopy(action["arguments"]),
                    "progression": None,
                "original_scene_alias": current_alias,
                "original_action_slot": role,
                "root_desired_outcome": scene.get("desired_outcome"),
                "action_contract": action_contract(str(action["action_type"])),
                }
                continue
            source_ref = str(action.get("continuation_of") or "")
            match = _SLOT_PATTERN.fullmatch(source_ref)
            source_action = slots.get(source_ref)
            source_alias = slot_scene.get(source_ref)
            source_index = int(match.group(1)) if match else None
            if not match or source_action is None or source_alias is None:
                errors.append(f"action:{ref}:continuation_source_slot_missing:{source_ref}")
                details.append({"error": "continuation_source_slot_missing",
                    "current_action_ref": ref, "continuation_of": source_ref})
                continue
            if source_ref in cycles or ref in cycles:
                continue
            if source_index is None or source_index >= index:
                errors.append(f"action:{ref}:continuation_source_not_prior:{source_ref}")
                details.append({"error": "continuation_source_not_prior",
                    "current_action_ref": ref, "referenced_action_ref": source_ref})
                continue
            path = _dependency_path(current_alias, source_alias, predecessors)
            if not path:
                errors.append(
                    f"action:{ref}:continuation_source_not_in_dependency_ancestry:{source_ref}")
                details.append({
                    "error": "continuation_source_not_in_dependency_ancestry",
                    "current_scene": current_alias, "referenced_scene": source_alias,
                    "current_action_ref": ref, "referenced_action_ref": source_ref,
                    "dependency_path": [],
                })
                continue
            source_resolved = resolved.get(source_ref)
            if source_resolved is None:
                errors.append(f"action:{ref}:continuation_source_unresolved:{source_ref}")
                continue
            progression = action["progression"]
            new_step = _normalized_text(progression["new_step"])
            source_argument_text = set(_argument_strings(
                source_resolved["resolved_arguments"]))
            if new_step in _MECHANICAL_PROGRESSIONS or new_step in source_argument_text:
                errors.append(f"action:{ref}:continuation_progression_not_new")
                details.append({"error": "continuation_progression_not_new",
                    "current_action_ref": ref, "continuation_of": source_ref,
                    "normalized_new_step": new_step})
            source_index_value = int(_SLOT_PATTERN.fullmatch(source_ref).group(1))
            source_outcome = _normalized_text(
                scenes[source_index_value - 1].get("desired_outcome") or "")
            current_outcome = _normalized_text(scene.get("desired_outcome") or "")
            if source_outcome and current_outcome == source_outcome:
                errors.append(f"action:{ref}:continuation_desired_outcome_duplicate")
                details.append({"error": "continuation_desired_outcome_duplicate",
                    "current_action_ref": ref, "continuation_of": source_ref})
            resolved[ref] = {
                "action_ref": ref, "action_mode": "continuation",
                "continuation_of": source_ref,
                "root_action_ref": source_resolved["root_action_ref"],
                "resolved_action_type": source_resolved["resolved_action_type"],
                "resolved_arguments": deepcopy(source_resolved["resolved_arguments"]),
                "progression": deepcopy(progression),
                "original_scene_alias": source_resolved["original_scene_alias"],
                "original_action_slot": source_resolved["original_action_slot"],
                "source_scene_alias": source_alias,
                "source_desired_outcome": scenes[source_index_value - 1].get(
                    "desired_outcome"),
                "root_desired_outcome": source_resolved.get("root_desired_outcome"),
                "dependency_path": path,
                "action_contract": deepcopy(source_resolved["action_contract"]),
            }
    return {
        "continuation_contract_version": NARRATIVE_ACTION_CONTINUATION_VERSION,
        "valid": not errors, "errors": list(dict.fromkeys(errors)),
        "error_details": details, "resolved_action_slots": resolved,
        "action_slot_count": len(slots),
        "allowed_continuation_slot_view": allowed_slot_view,
    }


def resolved_contract_subset(actions: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    action_types = sorted({str(action.get("resolved_action_type"))
                           for action in actions if isinstance(action, Mapping)})
    return {
        "registry_version": NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION,
        "actions": [action_contract(action_type) for action_type in action_types],
    }
