"""Full/promotable child separation and Promotion Gate v1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


PROMOTION_GATE_VERSION = "promotion_gate_v1"
PROMOTABLE_CHILD_FILENAME = "child_dialogue_promotable.json"


def promotion_decision(
    *, compiled_proposals: Sequence[Mapping[str, Any]], quality_valid: bool,
    claims_valid: bool, committer_output: Mapping[str, Any],
    scene_commit_result: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not quality_valid:
        reasons.append("dialogue_quality_failed")
    if not claims_valid:
        reasons.append("claim_validation_failed")
    if committer_output.get("scope_status") != "clean":
        reasons.append("scope_not_clean")
    consistency_allowed = {"clean", "authorized_change"} if compiled_proposals else {"clean"}
    if committer_output.get("world_consistency_status") not in consistency_allowed:
        reasons.append("world_consistency_not_clean")
    if compiled_proposals and not scene_commit_result.get("commit_ready"):
        reasons.append("scene_not_committed")
    return {"promotion_gate_version": PROMOTION_GATE_VERSION,
            "promotable": not reasons, "reasons": reasons}


def assemble_child_surfaces(
    *, child_id: str, root_id: str, scene_results: Sequence[Mapping[str, Any]],
    source_full_child_path: str = "child_dialogue_full.json",
) -> dict[str, Any]:
    full_scenes = []
    promotable_scenes = []
    manifest_scenes = []
    committed_refs: set[str] = set()
    committed_proposal_refs: set[str] = set()
    for result in scene_results:
        full_eligible = bool(result.get("quality_valid") and result.get("claims_valid") and
                             result.get("committer_contract_valid"))
        decision = promotion_decision(
            compiled_proposals=result.get("compiled_proposals") or [],
            quality_valid=bool(result.get("quality_valid")),
            claims_valid=bool(result.get("claims_valid")),
            committer_output=result.get("committer_output") or {},
            scene_commit_result=result.get("scene_commit_result") or {})
        scene = deepcopy(dict(result.get("bound_scene") or {}))
        if full_eligible:
            full_scenes.append(scene)
        if full_eligible and decision["promotable"]:
            promotable_scenes.append(scene)
            committed_refs.update(result.get("committed_memory_refs") or [])
            committed_proposal_refs.update(
                str(row.get("proposal_ref")) for row in
                result.get("compiled_proposals") or [] if row.get("proposal_ref"))
        manifest_scenes.append({"scene_ref": result.get("scene_ref"),
                                "full_included": full_eligible, **decision})
    full = {"child_id": child_id, "root_id": root_id, "surface": "full", "scenes": full_scenes}
    promotable = {"child_id": child_id, "root_id": root_id, "surface": "promotable",
                  "scenes": promotable_scenes}
    return {
        "child_dialogue_full.json": full,
        "child_dialogue_promotable.json": promotable,
        "child_dialogue.json": deepcopy(promotable),
        "promotion_manifest.json": {
            "promotion_gate_version": PROMOTION_GATE_VERSION,
            "child_id": child_id, "root_id": root_id,
            "artifact_role": "promotable_child",
            "source_full_child_path": source_full_child_path,
            "promoted_scene_refs": [str(row.get("scene_ref")) for row in manifest_scenes
                                     if row.get("promotable")],
            "excluded_scene_refs": [str(row.get("scene_ref")) for row in manifest_scenes
                                     if not row.get("promotable")],
            "excluded_scene_reasons": {
                str(row.get("scene_ref")): list(row.get("reasons") or [])
                for row in manifest_scenes if not row.get("promotable")},
            "committed_proposal_refs": sorted(committed_proposal_refs),
            "committed_memory_refs": sorted(committed_refs), "scenes": manifest_scenes,
        },
    }


def load_promotable_child(parent_directory: Path) -> dict[str, Any]:
    """The only supported cross-round/lineage/resume child surface loader."""
    import json
    path = Path(parent_directory) / PROMOTABLE_CHILD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"promotable_child_missing:{path}")
    return json.loads(path.read_text(encoding="utf-8"))
