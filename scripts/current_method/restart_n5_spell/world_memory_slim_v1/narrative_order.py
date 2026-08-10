"""Narrative placement metadata over append-only dialogue storage."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


NARRATIVE_ORDER_MANIFEST_VERSION = "narrative_order_manifest_v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def build_narrative_order_manifest(
    *, compiled_plan: Mapping[str, Any], complete_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    turns = list(complete_dialogue.get("turns") or [])
    positions = {str(row.get("turn_ref")): index for index, row in enumerate(turns)}
    entries = []
    for generation_index, scene in enumerate(compiled_plan.get("scenes") or [], 1):
        choice = scene.get("placement_choice") or {}
        after_ref = str(choice.get("after_turn_ref") or "")
        before_ref = str(choice.get("before_turn_ref") or "")
        placement = str(choice.get("placement") or "")
        if after_ref and after_ref not in positions:
            raise ValueError(f"narrative_order_after_ref_invalid:{after_ref}")
        if before_ref and before_ref not in positions:
            raise ValueError(f"narrative_order_before_ref_invalid:{before_ref}")
        try:
            base = float(choice["narrative_order_key"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("narrative_order_choice_key_invalid") from error
        entries.append({
            "scene_ref": str(scene.get("scene_ref") or ""),
            "generation_order": generation_index,
            "generation_order_index": generation_index,
            "intended_narrative_position": base,
            "narrative_order_key": f"{base:012.3f}:{generation_index:03d}",
            "placement": placement,
            "source_anchor_id": str(scene.get("source_anchor_id") or ""),
            "placement_choice_id": str(scene.get("placement_choice_id") or ""),
            "after_turn_ref": after_ref or None,
            "before_turn_ref": before_ref or None,
            "dependency_scene_refs": list(scene.get("dependency_scene_refs") or []),
            "branch_role": scene.get("branch_role"),
            "merge_target_scene_ref": scene.get("merge_target_scene_ref"),
            "impact_target_refs": list(scene.get("impact_target_refs") or []),
            "storage_action": "append_scene_without_rewriting_history",
        })
    narrative = sorted(entries, key=lambda row: row["narrative_order_key"])
    result = {
        "narrative_order_manifest_version": NARRATIVE_ORDER_MANIFEST_VERSION,
        "root_id": compiled_plan.get("root_id"),
        "round_index": compiled_plan.get("round_index"),
        "storage_mode": "append_only",
        "historical_transcript_rewrite_count": 0,
        "generation_order_scene_refs": [row["scene_ref"] for row in entries],
        "ordered_scene_refs": [row["scene_ref"] for row in narrative],
        "scenes": entries,
        "stable_dialogue_refs_preserved": True,
        "qa_alignment_preserved": True,
    }
    result["narrative_order_manifest_sha256"] = _sha(result)
    return result


def validate_narrative_order_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("narrative_order_manifest_version") != NARRATIVE_ORDER_MANIFEST_VERSION:
        errors.append("narrative_order_manifest_version")
    generation = list(value.get("generation_order_scene_refs") or [])
    narrative = list(value.get("ordered_scene_refs") or [])
    if (not generation or len(generation) != len(set(generation)) or
            set(generation) != set(narrative) or len(narrative) != len(generation)):
        errors.append("narrative_order_scene_set")
    if value.get("storage_mode") != "append_only":
        errors.append("narrative_order_storage_mode")
    if value.get("historical_transcript_rewrite_count") != 0:
        errors.append("narrative_order_history_rewritten")
    return {"valid": not errors, "errors": errors,
            "narrative_order_manifest_version": NARRATIVE_ORDER_MANIFEST_VERSION}
