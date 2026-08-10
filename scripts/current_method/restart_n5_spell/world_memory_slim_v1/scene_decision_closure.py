"""Local closure over reviewer decisions and hard same-round dependencies."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SCENE_DECISION_CLOSURE_VERSION = (
    "scene_decision_closure_v4_partial_narrative_progress_is_scene_valid")


def compute_scene_decision_closure(
    *, realized_scenes: Sequence[Mapping[str, Any]],
    local_gate_results: Mapping[str, Mapping[str, Any]],
    validator_output: Mapping[str, Any], scene_dependency_dag: Mapping[str, Any],
) -> dict[str, Any]:
    alias_to_ref = {str(row.get("scene_alias")): str(row.get("scene_ref"))
                    for row in realized_scenes}
    review_by_ref = {
        alias_to_ref[str(row.get("scene_alias"))]: row
        for row in validator_output.get("scene_reviews") or []
        if str(row.get("scene_alias")) in alias_to_ref
    }
    action_locations: dict[str, tuple[str, str]] = {}
    action_statuses: dict[tuple[str, str], str] = {}
    review_decisions = {
        ref: str(review.get("decision") or "unreviewed")
        for ref, review in review_by_ref.items()
    }
    for scene in realized_scenes:
        scene_ref = str(scene.get("scene_ref") or "")
        review = review_by_ref.get(scene_ref) or {}
        for role in ("primary", "secondary"):
            action = scene.get(f"{role}_action")
            realization = review.get(f"{role}_action_realization")
            if isinstance(action, Mapping) and action.get("action_ref"):
                action_locations[str(action["action_ref"])] = (scene_ref, role)
            if isinstance(realization, Mapping):
                action_statuses[(scene_ref, role)] = str(
                    realization.get("status") or "")
    continuation_blockers: dict[str, list[dict[str, str]]] = {}
    primary_action_blockers: dict[str, list[dict[str, str]]] = {}
    for scene in realized_scenes:
        scene_ref = str(scene.get("scene_ref") or "")
        review = review_by_ref.get(scene_ref) or {}
        primary = scene.get("primary_action")
        primary_realization = review.get("primary_action_realization")
        if isinstance(primary, Mapping) and isinstance(primary_realization, Mapping):
            primary_status = str(primary_realization.get("status") or "")
            if primary_status not in {"realized", "partially_realized"}:
                primary_action_blockers.setdefault(scene_ref, []).append({
                    "action_ref": str(primary.get("action_ref") or ""),
                    "status": primary_status,
                    "reason": "primary_action_not_realized",
                })
        for role in ("primary", "secondary"):
            action = scene.get(f"{role}_action")
            realization = review.get(f"{role}_action_realization")
            if (not isinstance(action, Mapping) or
                    action.get("action_mode") != "continuation" or
                    not isinstance(realization, Mapping) or
                    realization.get("status") not in {
                        "realized", "partially_realized"}):
                continue
            source_action_ref = str(action.get("continuation_of") or "")
            source = action_locations.get(source_action_ref)
            reason = None
            source_scene_ref = ""
            if source is None:
                reason = "source_action_missing"
            else:
                source_scene_ref, source_role = source
                if action_statuses.get((source_scene_ref, source_role)) != "realized":
                    reason = "source_action_not_realized"
                elif review_decisions.get(source_scene_ref) != "accept":
                    reason = "source_scene_not_accepted"
            if reason:
                continuation_blockers.setdefault(scene_ref, []).append({
                    "action_ref": str(action.get("action_ref") or ""),
                    "source_action_ref": source_action_ref,
                    "source_scene_ref": source_scene_ref,
                    "reason": reason,
                })
    hard_parents: dict[str, set[str]] = {ref: set() for ref in alias_to_ref.values()}
    for edge in scene_dependency_dag.get("edges") or []:
        if edge.get("hard_dependency"):
            hard_parents.setdefault(str(edge.get("to_scene_ref")), set()).add(
                str(edge.get("from_scene_ref")))
    # A narrative-action continuation is information-dependent on its source
    # even when the declared DAG edge is otherwise soft.  This does not add or
    # repair a Planner edge; validation already requires an existing ancestry.
    for scene in realized_scenes:
        current_ref = str(scene.get("scene_ref") or "")
        round_prefix = current_ref.split(":S", 1)[0]
        for action in (scene.get("primary_action"), scene.get("secondary_action")):
            if not isinstance(action, Mapping) or action.get("action_mode") != "continuation":
                continue
            source = str(action.get("continuation_of") or "")
            if len(source) >= 3 and source[1:3].isdigit():
                hard_parents.setdefault(current_ref, set()).add(
                    f"{round_prefix}:S{source[1:3]}")
    rows: dict[str, dict[str, Any]] = {}
    for ref in alias_to_ref.values():
        local = local_gate_results.get(ref) or {}
        review = review_by_ref.get(ref) or {}
        initial = str(review.get("decision") or "unreviewed")
        if local.get("status") != "local_gate_passed":
            adjusted, reason = "reject", "local_gate_not_passed"
        elif primary_action_blockers.get(ref):
            adjusted, reason = "reject", "primary_action_not_realized"
        elif continuation_blockers.get(ref):
            adjusted, reason = "reject", "continuation_source_unavailable"
        elif initial == "accept":
            adjusted, reason = "accept", "validator_accept"
        elif initial == "uncertain":
            adjusted, reason = "reject", "validator_uncertain_default_reject"
        else:
            adjusted, reason = "reject", "validator_reject"
        rows[ref] = {
            "scene_ref": ref, "scene_alias": next(
                alias for alias, value in alias_to_ref.items() if value == ref),
            "initial_validator_decision": initial,
            "closure_adjusted_decision": adjusted,
            "closure_reason": reason, "blocking_ancestor_refs": [],
            "uses_scene_refs": list(review.get("uses_scene_refs") or []),
            "primary_action_blockers": primary_action_blockers.get(ref, []),
            "continuation_action_blockers": continuation_blockers.get(ref, []),
        }
    changed = True
    while changed:
        changed = False
        for ref, row in rows.items():
            if row["closure_adjusted_decision"] != "accept":
                continue
            information_parents = set(str(value) for value in row["uses_scene_refs"])
            blockers = sorted(parent for parent in hard_parents.get(ref, set()) |
                              information_parents
                              if parent not in rows or rows[parent][
                                  "closure_adjusted_decision"] != "accept")
            if blockers:
                row["closure_adjusted_decision"] = "reject"
                row["closure_reason"] = "dependency_or_information_unavailable"
                row["blocking_ancestor_refs"] = blockers
                changed = True
    ordered = [rows[str(scene.get("scene_ref"))] for scene in realized_scenes]
    return {
        "scene_decision_closure_version": SCENE_DECISION_CLOSURE_VERSION,
        "scenes": ordered,
        "accepted_scene_refs": [row["scene_ref"] for row in ordered
                                if row["closure_adjusted_decision"] == "accept"],
        "rejected_scene_refs": [row["scene_ref"] for row in ordered
                                if row["closure_adjusted_decision"] != "accept"],
    }
