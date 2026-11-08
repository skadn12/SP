from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TRANSITION_TYPES = {
    "proposal", "challenge", "tradeoff", "reconsideration",
    "revision", "resolution", "event_progress", "other",
}
STATE_STAGES = {
    "unset", "tentative", "contested", "reconsidering",
    "revised", "final", "in_progress", "completed", "other",
}
RESOLUTION_STATUSES = {"open", "provisional", "final"}
TRIGGER_TYPES = {
    "constraint", "new_requirement", "new_information",
    "external_event", "stakeholder_objection", "none",
}


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    scene_count: int
    min_beats: int
    max_beats: int


SETTINGS = {
    "A": Setting("A", "current_scene_capacity", 3, 3, 7),
    "B": Setting("B", "increased_per_scene_capacity", 3, 7, 12),
    "C": Setting("C", "more_scenes", 5, 3, 7),
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def validate_plan(value: Any, *, setting: Setting, case_id: str) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["plan_not_object"]}
    if str(value.get("case_id") or "") != case_id:
        errors.append("case_id_mismatch")
    if str(value.get("setting") or "") != setting.key:
        errors.append("setting_mismatch")
    scenes = value.get("scenes")
    if not isinstance(scenes, list):
        return {"valid": False, "errors": errors + ["scenes_not_list"]}
    if len(scenes) != setting.scene_count:
        errors.append("scene_count_mismatch")
    seen_indexes: set[int] = set()
    state_slots: set[str] = set()
    for position, scene in enumerate(scenes, 1):
        prefix = f"scene_{position}"
        if not isinstance(scene, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        try:
            scene_index = int(scene.get("scene_index"))
        except (TypeError, ValueError):
            scene_index = -1
        if scene_index != position or scene_index in seen_indexes:
            errors.append(f"{prefix}_index")
        seen_indexes.add(scene_index)
        transition_type = str(scene.get("transition_type") or "")
        if transition_type not in TRANSITION_TYPES:
            errors.append(f"{prefix}_transition_type")
        if str(scene.get("primary_action") or "") not in TRANSITION_TYPES:
            errors.append(f"{prefix}_primary_action")
        slot = _text(scene.get("state_slot"))
        if not slot:
            errors.append(f"{prefix}_state_slot")
        else:
            state_slots.add(slot.casefold())
        for field in ("before_value", "after_value", "required_transition"):
            if not _text(scene.get(field)):
                errors.append(f"{prefix}_{field}")
        for field in ("before_stage", "after_stage"):
            if str(scene.get(field) or "") not in STATE_STAGES:
                errors.append(f"{prefix}_{field}")
        if str(scene.get("resolution_status") or "") not in RESOLUTION_STATUSES:
            errors.append(f"{prefix}_resolution_status")
        trigger = scene.get("trigger")
        if not isinstance(trigger, Mapping):
            errors.append(f"{prefix}_trigger")
        else:
            if str(trigger.get("type") or "") not in TRIGGER_TYPES:
                errors.append(f"{prefix}_trigger_type")
            if not _text(trigger.get("content")):
                errors.append(f"{prefix}_trigger_content")
        beats = scene.get("interaction_beats")
        if not isinstance(beats, list) or not (setting.min_beats <= len(beats) <= setting.max_beats):
            errors.append(f"{prefix}_interaction_beats")
        elif any(not _text(beat) for beat in beats):
            errors.append(f"{prefix}_empty_beat")
    if len(state_slots) > 1:
        errors.append("multiple_state_slots")
    for position, (left, right) in enumerate(zip(scenes, scenes[1:]), 1):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        if _text(left.get("after_value")).casefold() != _text(right.get("before_value")).casefold():
            errors.append(f"scene_{position}_to_{position + 1}_value_discontinuity")
        if str(left.get("after_stage") or "") != str(right.get("before_stage") or ""):
            errors.append(f"scene_{position}_to_{position + 1}_stage_discontinuity")
    return {
        "valid": not errors, "errors": errors,
        "scene_count": len(scenes), "state_slots": sorted(state_slots),
    }
