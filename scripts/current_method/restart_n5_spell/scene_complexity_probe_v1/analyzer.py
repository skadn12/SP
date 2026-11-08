from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


FINAL_STAGES = {"final"}
PROGRESS_STAGES = {"in_progress", "completed"}
MEANINGFUL_TYPES = {
    "proposal", "challenge", "tradeoff", "reconsideration",
    "revision", "resolution",
}
VALID_TRIGGER_TYPES = {
    "constraint", "new_requirement", "new_information",
    "external_event", "stakeholder_objection",
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split()).strip()


def _trigger_backed(scene: Mapping[str, Any]) -> bool:
    trigger = scene.get("trigger")
    if not isinstance(trigger, Mapping):
        return False
    content = _norm(trigger.get("content"))
    return str(trigger.get("type") or "") in VALID_TRIGGER_TYPES and len(content.split()) >= 3


def _is_normal_progress(scene: Mapping[str, Any]) -> bool:
    before = str(scene.get("before_stage") or "")
    after = str(scene.get("after_stage") or "")
    return (str(scene.get("transition_type") or "") == "event_progress" or
            ({before, after} <= PROGRESS_STAGES and before != after))


def analyze_chain(plan: Mapping[str, Any]) -> dict[str, Any]:
    scenes = [scene for scene in (plan.get("scenes") or []) if isinstance(scene, Mapping)]
    transition_counts = Counter(str(scene.get("transition_type") or "unknown") for scene in scenes)
    role_counts = Counter(str(scene.get("primary_action") or "unknown") for scene in scenes)
    value_changes = 0
    stage_changes = 0
    meaningful_changes = 0
    normal_progress = 0
    trigger_checks: list[bool] = []
    change_rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        value_changed = _norm(scene.get("before_value")) != _norm(scene.get("after_value"))
        stage_changed = str(scene.get("before_stage") or "") != str(scene.get("after_stage") or "")
        progress = _is_normal_progress(scene)
        if progress:
            normal_progress += int(value_changed or stage_changed)
        else:
            value_changes += int(value_changed)
            stage_changes += int(stage_changed)
            if value_changed or stage_changed:
                meaningful_changes += 1
                # Scene 1 establishes the initial proposal. Every later
                # meaningful change must be backed by a concrete trigger.
                if index > 1:
                    trigger_checks.append(_trigger_backed(scene))
        change_rows.append({
            "scene_index": index, "value_changed": value_changed,
            "stage_changed": stage_changed, "normal_event_progress": progress,
            "trigger_backed": _trigger_backed(scene),
        })
    premature_indexes = [
        index for index, scene in enumerate(scenes[:-1], 1)
        if str(scene.get("resolution_status") or "") == "final"
        or str(scene.get("after_stage") or "") in FINAL_STAGES
    ]
    final_at_last = bool(scenes) and (
        str(scenes[-1].get("resolution_status") or "") == "final"
        and str(scenes[-1].get("after_stage") or "") in FINAL_STAGES)
    semantic_actions = set(transition_counts) | set(role_counts)
    has_reconsider_or_revision = bool(
        semantic_actions & {"reconsideration", "revision"})
    role_requirement = "proposal" in semantic_actions and has_reconsider_or_revision
    reconsideration_scenes = sum(
        str(scene.get("transition_type") or "") == "reconsideration"
        or str(scene.get("primary_action") or "") == "reconsideration"
        for scene in scenes)
    revision_scenes = sum(
        str(scene.get("transition_type") or "") == "revision"
        or str(scene.get("primary_action") or "") == "revision"
        for scene in scenes)
    trigger_rate = sum(trigger_checks) / len(trigger_checks) if trigger_checks else 0.0
    complex_chain = bool(
        meaningful_changes >= 2
        and role_requirement
        and not premature_indexes
        and final_at_last
        and trigger_checks
        and all(trigger_checks)
    )
    simple_completion_chain = bool(
        scenes and not complex_chain
        and set(transition_counts) <= {"proposal", "event_progress", "resolution", "other"}
    )
    return {
        "scene_count": len(scenes),
        "state_slot": _norm(scenes[0].get("state_slot")) if scenes else "",
        "state_value_change_count": value_changes,
        "state_stage_change_count": stage_changes,
        "meaningful_state_change_count": meaningful_changes,
        "normal_event_progress_count": normal_progress,
        "transition_type_distribution": dict(sorted(transition_counts.items())),
        "scene_role_distribution": dict(sorted(role_counts.items())),
        "reconsideration_rate": reconsideration_scenes / len(scenes) if scenes else 0.0,
        "revision_rate": revision_scenes / len(scenes) if scenes else 0.0,
        "premature_closure": bool(premature_indexes),
        "premature_closure_scene_indexes": premature_indexes,
        "final_resolution_at_last_scene": final_at_last,
        "trigger_backed_transition_rate": trigger_rate,
        "has_explicit_trigger": any(trigger_checks),
        "all_post_initial_changes_trigger_backed": bool(trigger_checks) and all(trigger_checks),
        "role_requirement_met": role_requirement,
        "simple_completion_chain": simple_completion_chain,
        "complex_chain": complex_chain,
        "scene_changes": change_rows,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("contract", {}).get("valid")]
    analyses = [row.get("analysis") or {} for row in valid]
    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in analyses if row.get(field) is not None]
        return sum(values) / len(values) if values else None
    transition_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for row in analyses:
        transition_counts.update(row.get("transition_type_distribution") or {})
        role_counts.update(row.get("scene_role_distribution") or {})
    return {
        "planner_cases": len(rows),
        "valid_planner_cases": len(valid),
        "valid_rate": len(valid) / len(rows) if rows else 0.0,
        "scenes_per_case": mean("scene_count"),
        "complex_chain_rate": mean("complex_chain"),
        "mean_state_value_changes": mean("state_value_change_count"),
        "mean_state_stage_changes": mean("state_stage_change_count"),
        "premature_closure_rate": mean("premature_closure"),
        "trigger_backed_transition_rate": mean("trigger_backed_transition_rate"),
        "reconsideration_rate": mean("reconsideration_rate"),
        "revision_rate": mean("revision_rate"),
        "transition_type_distribution": dict(sorted(transition_counts.items())),
        "scene_role_distribution": dict(sorted(role_counts.items())),
    }
