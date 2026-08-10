"""Promotion v2 keeps dialogue validity independent from durable update validity."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


PROMOTION_V2_VERSION = "promotion_v2_scene_update_decoupled_v2"


def promote_scenes_v2(
    *, realized_scenes: Sequence[Mapping[str, Any]],
    local_gate_results: Mapping[str, Mapping[str, Any]],
    validator_output: Mapping[str, Any], scene_decision_closure: Mapping[str, Any],
    durable_update_compile_result: Mapping[str, Any],
    narrative_order_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    closure = {str(row.get("scene_ref")): row
               for row in scene_decision_closure.get("scenes") or []}
    review_alias = {str(row.get("scene_alias")): row
                    for row in validator_output.get("scene_reviews") or []}
    accepted_updates = durable_update_compile_result.get("accepted_updates") or []
    rejected_updates = durable_update_compile_result.get("rejected_updates") or []
    valid_update_scenes = {str(ref) for row in accepted_updates
                           for ref in row.get("source_scene_refs") or []}
    invalid_update_scenes = {
        str((row.get("candidate") or {}).get("source_scene_ref"))
        for row in rejected_updates
        if (row.get("candidate") or {}).get("source_scene_ref")}
    full, promotable, rejected, blocked = [], [], [], []
    accepted_without, accepted_with = [], []
    decisions = []
    for scene in realized_scenes:
        ref, alias = str(scene.get("scene_ref")), str(scene.get("scene_alias"))
        local_valid = (local_gate_results.get(ref) or {}).get("status") == "local_gate_passed"
        review = review_alias.get(alias) or {}
        reviewed = review.get("decision") in {"accept", "reject", "uncertain"}
        if local_valid and reviewed:
            full.append(deepcopy(dict(scene)))
        closed = (closure.get(ref) or {}).get("closure_adjusted_decision") == "accept"
        update_valid = ref in valid_update_scenes
        update_invalid = ref in invalid_update_scenes
        requires = bool(review.get("scene_requires_update_commit"))
        # Dialogue acceptance and durable-state persistence are independent.
        # ``scene_requires_update_commit`` is retained as a semantic-review audit
        # signal, but it cannot delete an otherwise accepted scene merely because
        # the post-review durable compiler rejects or emits no update.
        promoted = bool(local_valid and closed)
        if promoted:
            promotable.append(deepcopy(dict(scene)))
            (accepted_with if update_valid else accepted_without).append(ref)
        elif (closure.get(ref) or {}).get("blocking_ancestor_refs"):
            blocked.append(ref)
        else:
            rejected.append(ref)
        decisions.append({
            "scene_ref": ref, "scene_valid": bool(local_valid and closed),
            "durable_update_valid": update_valid,
            "durable_update_invalid": update_invalid,
            "scene_requires_update_commit": requires,
            "scene_requires_update_commit_advisory_only": True,
            "promoted": promoted,
        })
    return {
        "promotion_v2_version": PROMOTION_V2_VERSION,
        "narrative_order_manifest_version": narrative_order_manifest.get(
            "narrative_order_manifest_version"),
        "full_child_scenes": full, "promotable_child_scenes": promotable,
        "rejected_scenes": rejected, "dependency_blocked_scenes": blocked,
        "accepted_without_update_scenes": accepted_without,
        "accepted_with_update_scenes": accepted_with,
        "scene_decisions": decisions,
    }
