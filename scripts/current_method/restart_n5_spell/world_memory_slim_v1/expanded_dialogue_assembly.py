"""Materialize an immutable Parent + promoted-child cumulative dialogue.

World Memory remains a structural state store.  This derived dialogue is the
canonical text/evidence namespace for later rounds and (only after separate
admission) QA generation.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_dialogue_view import (
    compose_complete_dialogue_view,
    validate_complete_dialogue_view,
)


EXPANDED_DIALOGUE_SCHEMA_VERSION = "expanded_dialogue_round_v1"
COMPLETE_DIALOGUE_MANIFEST_VERSION = "complete_dialogue_manifest_v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _source_parent_turns(original_parent: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    dialogue = original_parent.get("dialogue") or {}
    turns = dialogue.get("turns") or []
    if not isinstance(turns, list) or not turns:
        raise ValueError("expanded_dialogue_parent_turns_missing")
    return turns


def _source_child_turns(child: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    scenes = child.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("expanded_dialogue_child_scenes_missing")
    return [turn for scene in scenes for turn in (scene.get("dialogue_turns") or [])]


def assemble_expanded_dialogue(
    *, original_parent: Mapping[str, Any], promotable_child: Mapping[str, Any],
    narrative_order_manifest: Mapping[str, Any],
    speaker_to_entity_ref: Mapping[str, str], parent_path: Path,
    parent_sha256: str, child_path: Path, child_sha256: str,
) -> dict[str, Any]:
    """Build the canonical narrative-order transcript without rewriting sources."""
    root_id = str(original_parent.get("root_id") or original_parent.get("sample_id") or "")
    if not root_id or promotable_child.get("root_id") != root_id:
        raise ValueError("expanded_dialogue_root_mismatch")
    if promotable_child.get("artifact_role") != "promotable_child":
        raise ValueError("expanded_dialogue_child_not_promotable")
    if promotable_child.get("parent_rewritten") is not False:
        raise ValueError("expanded_dialogue_parent_rewrite_contract")
    if str(promotable_child.get("parent_sha256") or "") != str(parent_sha256):
        raise ValueError("expanded_dialogue_parent_sha_binding")
    round_index = int(promotable_child.get("round_index") or 0)
    if round_index < 1:
        raise ValueError("expanded_dialogue_round_index_invalid")

    cumulative = compose_complete_dialogue_view(
        original_parent=original_parent,
        promoted_rounds=[{
            "round_index": round_index, "child": promotable_child,
            "narrative_order_manifest": narrative_order_manifest,
        }],
        speaker_to_entity_ref=speaker_to_entity_ref,
    )
    cumulative_validation = validate_complete_dialogue_view(cumulative)
    if not cumulative_validation["valid"]:
        raise ValueError("expanded_dialogue_cumulative_view_invalid:" +
                         ",".join(cumulative_validation["errors"]))

    label_by_ref = {str(ref): str(label)
                    for label, ref in speaker_to_entity_ref.items()}
    generation_turns = [deepcopy(row) for row in cumulative["turns"]]
    by_ref = {str(row["turn_ref"]): row for row in generation_turns}
    canonical_turns: list[dict[str, Any]] = []
    for narrative_index, ref in enumerate(cumulative["narrative_order_turn_refs"], 1):
        row = deepcopy(by_ref[str(ref)])
        generated = row["source_partition"] == "promoted_child"
        row["origin"] = "generated_extension" if generated else "source_grounded"
        row["verification_status"] = (
            "round_validated_promoted" if generated else "source_grounded")
        row["narrative_index"] = narrative_index
        if generated:
            row["speaker_label"] = label_by_ref.get(
                str(row.get("speaker_ref") or ""), str(row.get("speaker_label") or ""))
        canonical_turns.append(row)

    parent_turn_count = int(cumulative["original_parent"]["turn_count"])
    child_turn_count = len(_source_child_turns(promotable_child))
    child_indices = [index for index, row in enumerate(canonical_turns, 1)
                     if row["source_partition"] == "promoted_child"]
    if not child_indices:
        raise AssertionError("expanded_dialogue_child_partition_empty")
    segments: list[dict[str, Any]] = []
    segment_start = child_indices[0]
    previous = child_indices[0]
    for index in child_indices[1:] + [None]:
        if index is None or index != previous + 1:
            segment_rows = canonical_turns[segment_start - 1:previous]
            segments.append({
                "segment_index": len(segments) + 1,
                "start_narrative_index": segment_start,
                "end_narrative_index": previous,
                "first_turn_ref": segment_rows[0]["turn_ref"],
                "last_turn_ref": segment_rows[-1]["turn_ref"],
                "scene_refs": list(dict.fromkeys(
                    str(row["scene_ref"]) for row in segment_rows)),
            })
            if index is not None:
                segment_start = index
        if index is not None:
            previous = index

    result = {
        "expanded_dialogue_schema_version": EXPANDED_DIALOGUE_SCHEMA_VERSION,
        "root_id": root_id, "round_index": round_index,
        "artifact_role": "canonical_expanded_dialogue",
        "fact_source_role": "complete_cumulative_dialogue",
        "qa_source_candidate": True, "qa_generated": False,
        "canonical_order": "narrative_order",
        "storage_mode": "derived_append_only_projection",
        "source_parent": {
            "path": str(Path(parent_path).resolve()), "sha256": parent_sha256,
            "turn_count": parent_turn_count,
            "first_turn_ref": cumulative["original_parent"]["turn_refs"][0],
            "last_turn_ref": cumulative["original_parent"]["turn_refs"][-1],
            "rewritten": False,
        },
        "source_promotable_child": {
            "path": str(Path(child_path).resolve()), "sha256": child_sha256,
            "scene_count": int(promotable_child.get("scene_count") or 0),
            "turn_count": child_turn_count,
            "scene_refs": [str(row["scene_ref"])
                           for row in promotable_child.get("scenes") or []],
            "rewritten": False,
        },
        "boundary": {
            "marker": "CHILD_START_ROUND_1",
            "first_child_narrative_index": child_indices[0],
            "first_child_turn_ref": canonical_turns[child_indices[0] - 1]["turn_ref"],
            "child_segments": segments,
        },
        "parent_turn_count": parent_turn_count,
        "child_turn_count": child_turn_count,
        "turn_count": len(canonical_turns),
        "scene_count": int(promotable_child.get("scene_count") or 0),
        "turns": canonical_turns,
        "canonical_turn_refs": [str(row["turn_ref"]) for row in canonical_turns],
        "generation_order_turn_refs": list(cumulative["generation_order_turn_refs"]),
        "narrative_order_turn_refs": list(cumulative["narrative_order_turn_refs"]),
        "narrative_order_scene_refs": list(cumulative["narrative_order_scene_refs"]),
        "stable_evidence_turn_refs": True,
        "parent_child_lineage_preserved": True,
    }
    result["expanded_dialogue_sha256"] = _sha(result)
    return result


def validate_expanded_dialogue(
    value: Mapping[str, Any], *, original_parent: Mapping[str, Any] | None = None,
    promotable_child: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("expanded_dialogue_schema_version") != EXPANDED_DIALOGUE_SCHEMA_VERSION:
        errors.append("expanded_dialogue_schema_version")
    if value.get("artifact_role") != "canonical_expanded_dialogue":
        errors.append("expanded_dialogue_artifact_role")
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        errors.append("expanded_dialogue_turns_missing")
        turns = []
    refs = [str(row.get("turn_ref") or "") for row in turns
            if isinstance(row, Mapping)]
    if len(refs) != len(turns) or any(not ref for ref in refs) or len(refs) != len(set(refs)):
        errors.append("expanded_dialogue_turn_refs_invalid")
    if refs != list(value.get("canonical_turn_refs") or []):
        errors.append("expanded_dialogue_canonical_refs_mismatch")
    if refs != list(value.get("narrative_order_turn_refs") or []):
        errors.append("expanded_dialogue_not_narrative_order")
    parent_count = int(value.get("parent_turn_count") or 0)
    child_count = int(value.get("child_turn_count") or 0)
    if parent_count + child_count != len(turns) or value.get("turn_count") != len(turns):
        errors.append("expanded_dialogue_count_mismatch")
    parent_rows = [row for row in turns if row.get("source_partition") == "original_parent"]
    child_rows = [row for row in turns if row.get("source_partition") == "promoted_child"]
    if len(parent_rows) != parent_count or any(
            row.get("origin") != "source_grounded" for row in parent_rows):
        errors.append("expanded_dialogue_parent_partition")
    if len(child_rows) != child_count or any(
            row.get("origin") != "generated_extension" for row in child_rows):
        errors.append("expanded_dialogue_child_partition")
    boundary = value.get("boundary") or {}
    first_child = next((index for index, row in enumerate(turns, 1)
                        if row.get("source_partition") == "promoted_child"), None)
    if (boundary.get("marker") != "CHILD_START_ROUND_1" or
            boundary.get("first_child_narrative_index") != first_child or
            (first_child and boundary.get("first_child_turn_ref") !=
             turns[first_child - 1].get("turn_ref"))):
        errors.append("expanded_dialogue_boundary_invalid")
    if original_parent is not None:
        source = {str(row.get("turn_id") or row.get("turn_ref")): row
                  for row in _source_parent_turns(original_parent)}
        if set(source) != {str(row.get("turn_ref")) for row in parent_rows}:
            errors.append("expanded_dialogue_parent_ref_set")
        elif any(str(source[str(row["turn_ref"])].get("text")) != str(row.get("text"))
                 for row in parent_rows):
            errors.append("expanded_dialogue_parent_text_changed")
    if promotable_child is not None:
        source = {str(row.get("turn_ref") or row.get("turn_id")): row
                  for row in _source_child_turns(promotable_child)}
        if set(source) != {str(row.get("turn_ref")) for row in child_rows}:
            errors.append("expanded_dialogue_child_ref_set")
        elif any(str(source[str(row["turn_ref"])].get("text")) != str(row.get("text"))
                 for row in child_rows):
            errors.append("expanded_dialogue_child_text_changed")
    expected_sha = _sha({key: deepcopy(item) for key, item in value.items()
                         if key != "expanded_dialogue_sha256"})
    if value.get("expanded_dialogue_sha256") != expected_sha:
        errors.append("expanded_dialogue_sha_invalid")
    return {
        "validation_version": "expanded_dialogue_validation_v1",
        "valid": not errors, "errors": errors,
        "parent_turn_count": parent_count, "child_turn_count": child_count,
        "turn_count": len(turns), "unique_turn_ref_count": len(set(refs)),
        "complete_dialogue_materialized": not errors,
        "qa_source_ready": not errors, "qa_pipeline_ready": False,
    }


def expanded_dialogue_markdown(value: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical Expanded Dialogue — Round 1", "",
        f"Root: `{value['root_id']}`", "",
        f"Parent / Child / Total turns: {value['parent_turn_count']} / "
        f"{value['child_turn_count']} / {value['turn_count']}", "",
    ]
    current_unit: tuple[str, str] | None = None
    child_started = False
    for row in value.get("turns") or []:
        if row["source_partition"] == "promoted_child" and not child_started:
            lines.extend(["", "---", "", "# CHILD START — ROUND 1", ""])
            child_started = True
        unit = (str(row["source_partition"]),
                str(row.get("scene_ref") or row.get("session_ref") or ""))
        if unit != current_unit:
            title = (f"Parent session `{row['session_ref']}`" if unit[0] == "original_parent"
                     else f"Child scene `{row['scene_ref']}`")
            lines.extend([f"## {title}", ""])
            current_unit = unit
        lines.append(
            f"- `{row['turn_ref']}` | **{row.get('speaker_label') or row.get('speaker_ref')}**: "
            f"{row['text']}")
    lines.append("")
    return "\n".join(lines)


def complete_dialogue_manifest(
    *, expanded_dialogue: Mapping[str, Any], validation: Mapping[str, Any],
    expanded_json_path: Path, expanded_json_sha256: str,
    expanded_markdown_path: Path, expanded_markdown_sha256: str,
) -> dict[str, Any]:
    return {
        "manifest_version": COMPLETE_DIALOGUE_MANIFEST_VERSION,
        "root_id": expanded_dialogue["root_id"],
        "round_index": expanded_dialogue["round_index"],
        "artifact_role": "complete_dialogue_manifest",
        "expanded_dialogue_json": {"path": str(expanded_json_path.resolve()),
                                     "sha256": expanded_json_sha256},
        "expanded_dialogue_markdown": {"path": str(expanded_markdown_path.resolve()),
                                         "sha256": expanded_markdown_sha256},
        "source_parent": deepcopy(expanded_dialogue["source_parent"]),
        "source_promotable_child": deepcopy(
            expanded_dialogue["source_promotable_child"]),
        "counts": {key: expanded_dialogue[key] for key in
                   ("parent_turn_count", "child_turn_count", "turn_count", "scene_count")},
        "boundary": deepcopy(expanded_dialogue["boundary"]),
        "validation": deepcopy(dict(validation)),
        "readiness": {
            "complete_dialogue_materialized": bool(validation.get("valid")),
            "qa_source_ready": bool(validation.get("valid")),
            "qa_pipeline_ready": False,
            "round_2_input_ready": bool(validation.get("valid")),
        },
        "memory_is_qa_text_source": False,
        "qa_gold_source_role": "canonical_expanded_dialogue",
        "evidence_ref_namespace": "stable_parent_and_scene_local_turn_refs",
    }
