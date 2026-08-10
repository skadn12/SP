"""Read-only shadow helpers for truncated v11 Planner responses.

These helpers deliberately do not create a Planner result.  They extract only
complete scene objects for contract diagnosis and emit explicit migration
examples.  A truncated response remains parse-invalid and cannot be compiled,
promoted, or written back to a historical run.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping


PLANNER_SOURCE_PLACEMENT_SHADOW_VERSION = (
    "planner_source_anchor_placement_shadow_v1")


def extract_complete_scene_objects_from_truncated_response(
    raw: str,
) -> list[dict[str, Any]]:
    marker = '"scenes"'
    marker_at = raw.find(marker)
    if marker_at < 0:
        return []
    array_at = raw.find("[", marker_at + len(marker))
    if array_at < 0:
        return []
    scenes: list[dict[str, Any]] = []
    in_string = False
    escaped = False
    brace_depth = 0
    object_start: int | None = None
    for index in range(array_at + 1, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if brace_depth == 0:
                object_start = index
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
            if brace_depth == 0 and object_start is not None:
                value = json.loads(raw[object_start:index + 1])
                if not isinstance(value, dict):
                    raise ValueError("planner_shadow_scene_not_object")
                scenes.append(value)
                object_start = None
        elif char == "]" and brace_depth == 0:
            break
    return scenes


def analyze_probe4_source_placement_shadow(
    raw: str, *, placement_choice_registry: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        json.loads(raw)
        json_complete = True
    except json.JSONDecodeError:
        json_complete = False
    choice_ids = {str(row.get("placement_choice_id") or "")
                  for row in placement_choice_registry.get("choices") or []}
    if "PC_TAIL_AFTER" not in choice_ids:
        raise ValueError("probe4_shadow_tail_after_choice_missing")
    scenes = extract_complete_scene_objects_from_truncated_response(raw)
    migrations = []
    for index, scene in enumerate(scenes, 1):
        source_anchor_id = str(scene.get("historical_anchor_id") or "")
        old_placement = str(scene.get("placement") or "")
        migration = {
            "scene_index": index,
            "scene_alias": scene.get("scene_alias"),
            "old_contract": {
                "historical_anchor_id": source_anchor_id,
                "placement": old_placement,
            },
            "new_contract_example": {
                "source_anchor_id": source_anchor_id,
                "placement_choice_id": (
                    "PC_TAIL_AFTER" if old_placement == "after" else None),
            },
            "migration_interpretation": (
                "content_from_historical_source_but_narrated_after_current_tail"
                if old_placement == "after" else
                "requires_explicit_gap_or_parallel_choice_review"),
            "shadow_only": True,
        }
        migrations.append(migration)
    return {
        "shadow_version": PLANNER_SOURCE_PLACEMENT_SHADOW_VERSION,
        "historical_response_json_complete": json_complete,
        "historical_response_remains_parse_invalid": not json_complete,
        "complete_scene_objects_extracted": len(scenes),
        "truncated_scene_object_present": raw.rstrip().endswith(",") or bool(
            scenes and '"scenes"' in raw and not json_complete),
        "migrations": migrations,
        "shadow_does_not_claim_planner_success": True,
        "shadow_does_not_compile_or_write_history": True,
    }


def migrate_shadow_scene_to_new_contract(
    scene: Mapping[str, Any], *, placement_choice_id: str,
) -> dict[str, Any]:
    """Explicit fixture migration; never invoked by the live schema path."""
    if "historical_anchor_id" not in scene or "placement" not in scene:
        raise ValueError("planner_shadow_legacy_fields_missing")
    value = deepcopy(dict(scene))
    value["source_anchor_id"] = str(value.pop("historical_anchor_id"))
    value.pop("placement")
    value["placement_choice_id"] = str(placement_choice_id)
    return value


__all__ = [
    "PLANNER_SOURCE_PLACEMENT_SHADOW_VERSION",
    "analyze_probe4_source_placement_shadow",
    "extract_complete_scene_objects_from_truncated_response",
    "migrate_shadow_scene_to_new_contract",
]
