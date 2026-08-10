"""Versioned cumulative-dialogue projection for continuous expansion.

The composer is deliberately read-only.  It never rewrites source turn or scene
references and it keeps storage/generation order separate from narrative order.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


COMPLETE_DIALOGUE_VIEW_VERSION = "complete_dialogue_view_v1"


def _sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parent_identity(parent: Mapping[str, Any]) -> str:
    value = str(parent.get("sample_id") or parent.get("root_id") or "")
    if not value:
        raise ValueError("complete_dialogue_parent_identity_missing")
    return value


def compose_complete_dialogue_view(
    *, original_parent: Mapping[str, Any],
    promoted_rounds: Sequence[Mapping[str, Any]] = (),
    speaker_to_entity_ref: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compose Original Parent + every promoted round without renumbering refs.

    Each promoted-round item has ``round_index`` and ``child``.  It may also
    carry a ``narrative_order_manifest`` whose ``ordered_scene_refs`` changes
    only the narrative projection, never append-only storage order.
    """
    root_id = _parent_identity(original_parent)
    dialogue = original_parent.get("dialogue")
    turns = dialogue.get("turns") if isinstance(dialogue, Mapping) else None
    if not isinstance(turns, list) or not turns:
        raise ValueError("complete_dialogue_parent_turns_missing")
    speaker_map = dict(speaker_to_entity_ref or {})
    normalized: list[dict[str, Any]] = []
    generation_scene_refs: list[str] = []
    round_rows: list[dict[str, Any]] = []
    seen_turn_refs: set[str] = set()
    seen_scene_refs: set[str] = set()

    for index, raw in enumerate(turns, 1):
        if not isinstance(raw, Mapping):
            raise ValueError("complete_dialogue_parent_turn_invalid")
        ref = str(raw.get("turn_id") or raw.get("turn_ref") or "")
        session_ref = str(raw.get("session_id") or raw.get("session_ref") or "")
        speaker = str(raw.get("speaker") or "")
        text = raw.get("text")
        if not ref or ref in seen_turn_refs or not session_ref or not speaker or not _text(text):
            raise ValueError("complete_dialogue_parent_turn_contract")
        seen_turn_refs.add(ref)
        normalized.append({
            "turn_ref": ref, "session_ref": session_ref, "scene_ref": None,
            "speaker_ref": speaker_map.get(speaker), "speaker_label": speaker,
            "text": str(text), "source_partition": "original_parent",
            "round_index": 0, "generation_index": index,
        })

    parent_turn_count = len(normalized)
    ordered_rounds = sorted(promoted_rounds, key=lambda row: int(row.get("round_index") or 0))
    expected_round = 1
    narrative_by_round: dict[int, list[str]] = {}
    narrative_position_by_scene: dict[str, float] = {}
    for wrapper in ordered_rounds:
        round_index = int(wrapper.get("round_index") or 0)
        if round_index != expected_round:
            raise ValueError("complete_dialogue_round_sequence_invalid")
        expected_round += 1
        child = wrapper.get("child")
        if not isinstance(child, Mapping) or child.get("root_id") != root_id:
            raise ValueError("complete_dialogue_child_root_mismatch")
        scenes = child.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError("complete_dialogue_child_scenes_invalid")
        round_scene_refs: list[str] = []
        round_turn_refs: list[str] = []
        for scene in scenes:
            if not isinstance(scene, Mapping):
                raise ValueError("complete_dialogue_child_scene_invalid")
            scene_ref = str(scene.get("scene_ref") or "")
            if not scene_ref or scene_ref in seen_scene_refs:
                raise ValueError("complete_dialogue_scene_ref_duplicate_or_missing")
            seen_scene_refs.add(scene_ref)
            generation_scene_refs.append(scene_ref)
            round_scene_refs.append(scene_ref)
            session_ref = str(scene.get("session_ref") or f"{scene_ref}:SESSION")
            scene_turns = scene.get("dialogue_turns")
            if not isinstance(scene_turns, list) or not scene_turns:
                raise ValueError("complete_dialogue_child_turns_missing")
            for raw in scene_turns:
                if not isinstance(raw, Mapping):
                    raise ValueError("complete_dialogue_child_turn_invalid")
                ref = str(raw.get("turn_ref") or raw.get("turn_id") or "")
                speaker_ref = str(raw.get("speaker_ref") or "")
                text = raw.get("text")
                if not ref or ref in seen_turn_refs or not speaker_ref or not _text(text):
                    raise ValueError("complete_dialogue_child_turn_contract")
                seen_turn_refs.add(ref)
                round_turn_refs.append(ref)
                normalized.append({
                    "turn_ref": ref, "session_ref": session_ref,
                    "scene_ref": scene_ref, "speaker_ref": speaker_ref,
                    "speaker_label": str(raw.get("speaker_label") or speaker_ref),
                    "text": str(text), "source_partition": "promoted_child",
                    "round_index": round_index,
                    "generation_index": len(normalized) + 1,
                })
        manifest = wrapper.get("narrative_order_manifest")
        ordered_scene_refs = list(round_scene_refs)
        if isinstance(manifest, Mapping):
            candidate = manifest.get("ordered_scene_refs")
            if candidate is None and isinstance(manifest.get("scenes"), list):
                candidate = [row.get("scene_ref") for row in sorted(
                    manifest["scenes"], key=lambda row: str(row.get("narrative_order_key") or ""))]
            if candidate is not None:
                manifest_order = [str(value) for value in candidate]
                if (any(not value for value in manifest_order) or
                        len(manifest_order) != len(set(manifest_order))):
                    raise ValueError("complete_dialogue_narrative_scene_set_mismatch")
                declared = manifest.get("generation_order_scene_refs")
                if declared is None and isinstance(manifest.get("scenes"), list):
                    declared = [row.get("scene_ref") for row in manifest["scenes"]
                                if isinstance(row, Mapping)]
                declared_refs = ([str(value) for value in declared]
                                 if isinstance(declared, list) else None)
                if (declared_refs is not None and
                        (len(declared_refs) != len(set(declared_refs)) or
                         set(manifest_order) != set(declared_refs))):
                    raise ValueError("complete_dialogue_narrative_scene_set_mismatch")
                # A round-level narrative manifest describes every realized
                # scene, while a promotable child can be a strict subset after
                # semantic/dependency closure. Project that authoritative order
                # onto the child; never invent or silently lose a child scene.
                if not set(round_scene_refs).issubset(set(manifest_order)):
                    raise ValueError("complete_dialogue_narrative_scene_set_mismatch")
                ordered_scene_refs = [ref for ref in manifest_order
                                      if ref in set(round_scene_refs)]
                if (set(ordered_scene_refs) != set(round_scene_refs) or
                        len(ordered_scene_refs) != len(round_scene_refs)):
                    raise ValueError("complete_dialogue_narrative_scene_set_mismatch")
            manifest_scenes = manifest.get("scenes")
            if isinstance(manifest_scenes, list):
                for manifest_scene in manifest_scenes:
                    if not isinstance(manifest_scene, Mapping):
                        raise ValueError("complete_dialogue_narrative_scene_invalid")
                    ref = str(manifest_scene.get("scene_ref") or "")
                    position = manifest_scene.get("intended_narrative_position")
                    if not ref or not isinstance(position, (int, float)):
                        raise ValueError("complete_dialogue_narrative_position_invalid")
                    if ref in round_scene_refs:
                        narrative_position_by_scene[ref] = float(position)
        for local_index, scene_ref in enumerate(ordered_scene_refs, 1):
            narrative_position_by_scene.setdefault(
                scene_ref, float(parent_turn_count + (round_index * 1000) + local_index))
        narrative_by_round[round_index] = ordered_scene_refs
        round_rows.append({
            "round_index": round_index,
            "child_id": str(child.get("child_id") or ""),
            "scene_refs": round_scene_refs,
            "turn_refs": round_turn_refs,
            "turn_count": len(round_turn_refs),
            "narrative_scene_refs": ordered_scene_refs,
            "source_child_sha256": _sha(child),
        })

    turn_by_scene: dict[str, list[str]] = {}
    for row in normalized[parent_turn_count:]:
        turn_by_scene.setdefault(str(row["scene_ref"]), []).append(str(row["turn_ref"]))
    narrative_units: list[tuple[float, int, list[str]]] = [
        (float(index), 0, [str(row["turn_ref"])])
        for index, row in enumerate(normalized[:parent_turn_count])]
    narrative_scene_refs = sorted(
        generation_scene_refs,
        key=lambda ref: (narrative_position_by_scene[ref],
                         generation_scene_refs.index(ref)))
    for scene_ref in narrative_scene_refs:
        narrative_units.append((narrative_position_by_scene[scene_ref], 1,
                                turn_by_scene[scene_ref]))
    narrative_turn_refs = [ref for _, _, refs in sorted(
        narrative_units, key=lambda value: (value[0], value[1])) for ref in refs]
    generation_turn_refs = [str(row["turn_ref"]) for row in normalized]
    if set(narrative_turn_refs) != set(generation_turn_refs) or len(
            narrative_turn_refs) != len(generation_turn_refs):
        raise AssertionError("complete_dialogue_narrative_turn_set_mismatch")
    result = {
        "complete_dialogue_schema_version": COMPLETE_DIALOGUE_VIEW_VERSION,
        "root_id": root_id,
        "original_parent": {
            "turn_count": parent_turn_count,
            "dialogue_sha256": str((dialogue or {}).get("dialogue_hash") or _sha(turns)),
            "turn_refs": generation_turn_refs[:parent_turn_count],
        },
        "rounds": round_rows,
        "turns": normalized,
        "turn_count": len(normalized),
        "scene_count": len(generation_scene_refs),
        "generation_order_scene_refs": generation_scene_refs,
        "generation_order_turn_refs": generation_turn_refs,
        "narrative_order_scene_refs": narrative_scene_refs,
        "narrative_order_turn_refs": narrative_turn_refs,
        "stable_refs_preserved": True,
        "storage_mode": "append_only",
        "fact_source_role": "complete_cumulative_dialogue",
    }
    result["complete_dialogue_sha256"] = _sha(result)
    return result


def validate_complete_dialogue_view(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("complete_dialogue_schema_version") != COMPLETE_DIALOGUE_VIEW_VERSION:
        errors.append("complete_dialogue_schema_version")
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        errors.append("complete_dialogue_turns_missing")
        turns = []
    refs = [str(row.get("turn_ref") or "") for row in turns if isinstance(row, Mapping)]
    if len(refs) != len(turns) or any(not ref for ref in refs) or len(refs) != len(set(refs)):
        errors.append("complete_dialogue_turn_refs_invalid")
    generation = list(value.get("generation_order_turn_refs") or [])
    narrative = list(value.get("narrative_order_turn_refs") or [])
    if generation != refs:
        errors.append("complete_dialogue_generation_order_mismatch")
    if set(narrative) != set(refs) or len(narrative) != len(refs):
        errors.append("complete_dialogue_narrative_order_mismatch")
    parent_count = int((value.get("original_parent") or {}).get("turn_count") or 0)
    if parent_count < 1 or any(row.get("source_partition") != "original_parent"
                              for row in turns[:parent_count]):
        errors.append("complete_dialogue_parent_partition_invalid")
    return {"valid": not errors, "errors": errors,
            "complete_dialogue_schema_version": COMPLETE_DIALOGUE_VIEW_VERSION}
