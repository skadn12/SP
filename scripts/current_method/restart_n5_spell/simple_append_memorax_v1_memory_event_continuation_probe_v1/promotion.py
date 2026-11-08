"""Deterministic scene-level promotion after the whole-round audit.

The raw Child is kept as an audit artifact.  Only scenes that pass the
adherence audit and the mechanical append/closure checks are promoted to the
Child that is written to Memory or used as a QA source.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping


def _normalise_text(value: Any) -> str:
    """Normalise only enough for exact/near-exact duplicate detection."""
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def _is_meaningful_duplicate(value: str) -> bool:
    """Ignore short conversational fillers when checking duplicate turns."""
    return len(value) >= 20 and len(value.split()) >= 4


def _scene_ref(scene: Mapping[str, Any], index: int, *, round_index: int = 1) -> str:
    return str(scene.get("scene_ref") or
               f"R{int(round_index):02d}:S{index:02d}")


def _dependency_refs(scene: Mapping[str, Any]) -> list[str]:
    """Read optional explicit scene dependencies without inventing edges."""
    for key in ("depends_on_scene_refs", "dependency_scene_refs",
                "predecessor_scene_refs"):
        value = scene.get(key)
        if value is not None:
            return [str(ref) for ref in value] if isinstance(value, list) else []
    return []


def _child_turns(child_scene: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    turns = child_scene.get("dialogue_turns")
    return [turn for turn in turns if isinstance(turn, Mapping)] \
        if isinstance(turns, list) else []


def _parent_texts(parent: Mapping[str, Any]) -> set[str]:
    turns = list((parent.get("dialogue") or {}).get("turns") or [])
    return {_normalise_text(turn.get("text")) for turn in turns
            if isinstance(turn, Mapping) and _normalise_text(turn.get("text"))}


def build_promotable_child(
    *, parent: Mapping[str, Any], plan: Mapping[str, Any],
    child: Mapping[str, Any], adherence_audit: Mapping[str, Any],
    satisfied_dependency_refs: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a partial/full Child and its deterministic promotion audit.

    Semantic acceptance comes from the existing one-call Auditor.  This
    function adds only mechanical checks: plan/Child alignment, explicit
    dependency closure, and exact/normalised duplicate turns against Parent
    and already promoted scenes.  It never infers hidden dependencies.
    """
    errors: list[str] = []
    plan_scenes = plan.get("scenes")
    child_scenes = child.get("scenes")
    audit_scenes = adherence_audit.get("scenes")
    if not isinstance(plan_scenes, list) or not isinstance(child_scenes, list):
        return {"valid": False, "errors": ["promotion_scene_lists_required"],
                "promotable_child": None, "audit": None}
    if not isinstance(audit_scenes, list):
        return {"valid": False, "errors": ["promotion_audit_scenes_required"],
                "promotable_child": None, "audit": None}
    if len(plan_scenes) != len(child_scenes):
        errors.append("promotion_plan_child_scene_count")
    if len(audit_scenes) != len(child_scenes):
        errors.append("promotion_audit_child_scene_count")
    if errors:
        return {"valid": False, "errors": errors,
                "promotable_child": None, "audit": None}

    try:
        round_index = int(child.get("round_index") or 1)
    except (TypeError, ValueError):
        return {"valid": False, "errors": ["promotion_round_index_invalid"],
                "promotable_child": None, "audit": None}
    if round_index < 1:
        return {"valid": False, "errors": ["promotion_round_index_invalid"],
                "promotable_child": None, "audit": None}

    plan_by_ref = {_scene_ref(scene, index, round_index=round_index): scene
                   for index, scene in enumerate(plan_scenes, 1)
                   if isinstance(scene, Mapping)}
    child_by_ref = {_scene_ref(scene, index, round_index=round_index): scene
                    for index, scene in enumerate(child_scenes, 1)
                    if isinstance(scene, Mapping)}
    audit_by_ref = {}
    for index, row in enumerate(audit_scenes, 1):
        if not isinstance(row, Mapping):
            errors.append(f"promotion_audit_scene_{index}_object")
            continue
        ref = str(row.get("scene_ref") or _scene_ref(
            child_scenes[index - 1], index, round_index=round_index))
        audit_by_ref[ref] = row
    if set(plan_by_ref) != set(child_by_ref):
        errors.append("promotion_scene_refs_mismatch")
    if set(audit_by_ref) != set(child_by_ref):
        errors.append("promotion_audit_scene_refs_mismatch")
    if errors:
        return {"valid": False, "errors": errors,
                "promotable_child": None, "audit": None}

    parent_texts = _parent_texts(parent)
    promoted_texts: set[str] = set()
    decisions: list[dict[str, Any]] = []
    initial_pass: set[str] = set()
    for index, child_scene in enumerate(child_scenes, 1):
        ref = _scene_ref(child_scene, index, round_index=round_index)
        row = audit_by_ref[ref]
        scene_errors: list[str] = []
        if str(row.get("scene_verdict") or "") == "PASS":
            initial_pass.add(ref)
        else:
            scene_errors.append("adherence_failed")
        if str(plan_by_ref[ref].get("memory_event_id") or "") != str(
                child_scene.get("memory_event_id") or ""):
            scene_errors.append("event_ref_mismatch")
        turns = _child_turns(child_scene)
        if not turns:
            scene_errors.append("child_turns_missing")
        duplicate_parent_refs: list[str] = []
        duplicate_promoted_positions: list[int] = []
        for turn_index, turn in enumerate(turns, 1):
            normal = _normalise_text(turn.get("text"))
            if not normal:
                continue
            if normal in parent_texts and _is_meaningful_duplicate(normal):
                duplicate_parent_refs.append(str(turn.get("turn_ref") or ""))
            if normal in promoted_texts and _is_meaningful_duplicate(normal):
                duplicate_promoted_positions.append(turn_index)
        if duplicate_parent_refs:
            scene_errors.append("exact_or_normalized_parent_duplicate")
        if duplicate_promoted_positions:
            scene_errors.append("exact_or_normalized_promoted_duplicate")
        decisions.append({
            "scene_ref": ref,
            "adherence_verdict": str(row.get("scene_verdict") or ""),
            "dependency_refs": _dependency_refs(plan_by_ref[ref]),
            "duplicate_parent_turn_refs": duplicate_parent_refs,
            "duplicate_promoted_positions": duplicate_promoted_positions,
            "mechanical_errors": scene_errors,
            "initial_pass": not scene_errors,
            "promotable": not scene_errors,
        })
        # Duplicate checks are against the original Parent and already
        # accepted scenes only; rejected scene text must not poison promotion.
        if not scene_errors:
            promoted_texts.update(
                _normalise_text(turn.get("text")) for turn in turns
                if _is_meaningful_duplicate(_normalise_text(turn.get("text"))))

    by_ref = {row["scene_ref"]: row for row in decisions}
    externally_satisfied = {str(ref) for ref in satisfied_dependency_refs}
    # Closure is explicit-only.  A passing scene depending on a failed or
    # mechanically rejected scene is not independently promotable.  Resolve
    # dependencies to a fixed point so a dependent scene listed before its
    # passing predecessor is not rejected merely because of list order.
    for row in decisions:
        row["promotable"] = False
    changed = True
    while changed:
        changed = False
        for row in decisions:
            if row["mechanical_errors"]:
                continue
            missing = [dep for dep in row["dependency_refs"]
                       if dep not in by_ref and dep not in externally_satisfied]
            if missing:
                row["mechanical_errors"].append(
                    "dependency_ref_unknown:" + ",".join(missing))
                changed = True
                continue
            blocked = [dep for dep in row["dependency_refs"]
                       if dep in by_ref and not by_ref[dep]["initial_pass"]]
            unresolved = [dep for dep in row["dependency_refs"]
                          if dep in by_ref and not by_ref[dep]["promotable"]]
            if blocked:
                row["mechanical_errors"].append(
                    "dependency_not_promotable:" + ",".join(blocked))
                changed = True
            elif unresolved:
                continue
            elif row["initial_pass"] and not row["promotable"]:
                row["promotable"] = True
                changed = True
    for row in decisions:
        if (row["initial_pass"] and not row["promotable"]
                and not row["mechanical_errors"]):
            row["mechanical_errors"].append("dependency_cycle_or_unresolved")

    promoted_refs = [row["scene_ref"] for row in decisions if row["promotable"]]
    rejected_refs = [row["scene_ref"] for row in decisions if not row["promotable"]]
    promoted_scenes = [deepcopy(child_by_ref[ref]) for ref in child_by_ref
                       if ref in promoted_refs]
    # Preserve the original generated order; do not renumber or rewrite refs.
    promotable_child = deepcopy(dict(child))
    promotable_child["scenes"] = promoted_scenes
    promotable_child["promotion"] = {
        "promotion_version": "scene_partial_promotion_v1",
        "mode": "full" if len(promoted_refs) == len(child_scenes)
        else ("partial" if promoted_refs else "none"),
        "source_scene_count": len(child_scenes),
        "promoted_scene_refs": promoted_refs,
        "rejected_scene_refs": rejected_refs,
    }
    audit = {
        "promotion_version": "scene_partial_promotion_v1",
        "source_scene_count": len(child_scenes),
        "promoted_scene_count": len(promoted_refs),
        "rejected_scene_count": len(rejected_refs),
        "promoted_scene_refs": promoted_refs,
        "rejected_scene_refs": rejected_refs,
        "full_round_ready": len(promoted_refs) == len(child_scenes),
        "promotable_child_ready": bool(promoted_refs),
        "partial_promotion": bool(promoted_refs) and len(promoted_refs) < len(child_scenes),
        "scenes": decisions,
    }
    return {"valid": True, "errors": [],
            "promotable_child": promotable_child, "audit": audit}


def validate_promotable_child(
    value: Mapping[str, Any], *, source_child: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a promoted Child is a strict ordered subset of source."""
    errors: list[str] = []
    source = source_child.get("scenes")
    promoted = value.get("scenes")
    if not isinstance(source, list) or not isinstance(promoted, list):
        return {"valid": False, "errors": ["promotable_scene_lists_required"]}
    source_refs = [str(row.get("scene_ref") or "") for row in source]
    promoted_refs = [str(row.get("scene_ref") or "") for row in promoted]
    if not promoted_refs:
        errors.append("promotable_child_empty")
    if len(promoted_refs) != len(set(promoted_refs)):
        errors.append("promotable_scene_refs_duplicate")
    if any(ref not in source_refs for ref in promoted_refs):
        errors.append("promotable_scene_ref_not_in_source")
    if [ref for ref in source_refs if ref in promoted_refs] != promoted_refs:
        errors.append("promotable_scene_order_changed")
    source_by_ref = {str(row.get("scene_ref")): row for row in source}
    for row in promoted:
        ref = str(row.get("scene_ref") or "")
        if row != source_by_ref.get(ref):
            errors.append(f"promotable_scene_changed:{ref}")
    return {"valid": not errors, "errors": errors,
            "promoted_scene_count": len(promoted_refs)}
