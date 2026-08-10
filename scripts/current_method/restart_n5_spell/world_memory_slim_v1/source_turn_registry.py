"""Text-free turn index backed by the authoritative Complete Dialogue view."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


SOURCE_TURN_REGISTRY_VERSION = "source_turn_registry_v1"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compile_source_turn_registry(
    complete_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Index every authoritative dialogue turn without copying its text."""
    root_id = str(complete_dialogue.get("root_id") or "")
    turns = complete_dialogue.get("turns")
    if not root_id or not isinstance(turns, list) or not turns:
        raise ValueError("source_turn_registry_complete_dialogue_invalid")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in turns:
        if not isinstance(row, Mapping):
            raise ValueError("source_turn_registry_turn_invalid")
        turn_ref = str(row.get("turn_ref") or "")
        text = row.get("text")
        speaker_ref = str(row.get("speaker_ref") or row.get("speaker_label") or "")
        if (not turn_ref or turn_ref in seen or not isinstance(text, str) or
                not text.strip() or not speaker_ref):
            raise ValueError("source_turn_registry_turn_contract")
        seen.add(turn_ref)
        partition = str(row.get("source_partition") or "")
        records.append({
            "turn_ref": turn_ref,
            "session_ref": str(row.get("session_ref") or ""),
            "scene_ref": row.get("scene_ref"),
            "speaker_ref": speaker_ref,
            "speaker_display_name": str(row.get("speaker_label") or speaker_ref),
            "text_sha256": _sha_text(text),
            "provenance": ("source_grounded" if partition == "original_parent"
                           else "generated_extension"),
            "origin_round": int(row.get("round_index") or 0),
        })
    result = {
        "source_turn_registry_version": SOURCE_TURN_REGISTRY_VERSION,
        "root_id": root_id,
        "complete_dialogue_sha256": complete_dialogue.get(
            "complete_dialogue_sha256"),
        "turns": records,
        "turn_count": len(records),
        "contains_dialogue_text": False,
    }
    result["source_turn_registry_sha256"] = _sha_json(result)
    return result


def validate_source_turn_registry(
    registry: Mapping[str, Any], complete_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve every registry ref and verify the exact Complete Dialogue text SHA."""
    errors: list[str] = []
    if registry.get("source_turn_registry_version") != SOURCE_TURN_REGISTRY_VERSION:
        errors.append("source_turn_registry_version")
    if registry.get("root_id") != complete_dialogue.get("root_id"):
        errors.append("source_turn_registry_root_mismatch")
    dialogue_rows = complete_dialogue.get("turns")
    dialogue_by_ref = {
        str(row.get("turn_ref")): row for row in dialogue_rows or []
        if isinstance(row, Mapping) and row.get("turn_ref")
    }
    records = registry.get("turns")
    if not isinstance(records, list):
        errors.append("source_turn_registry_turns_not_list")
        records = []
    refs: list[str] = []
    forbidden_text_count = 0
    for index, row in enumerate(records):
        prefix = f"source_turn:{index}"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}:not_object")
            continue
        if "text" in row:
            forbidden_text_count += 1
            errors.append(f"{prefix}:dialogue_text_forbidden")
        ref = str(row.get("turn_ref") or "")
        refs.append(ref)
        source = dialogue_by_ref.get(ref)
        if source is None:
            errors.append(f"{prefix}:unresolved_turn_ref:{ref}")
            continue
        if row.get("text_sha256") != _sha_text(str(source.get("text") or "")):
            errors.append(f"{prefix}:text_sha_mismatch:{ref}")
        expected_speaker = str(source.get("speaker_ref") or
                               source.get("speaker_label") or "")
        if row.get("speaker_ref") != expected_speaker:
            errors.append(f"{prefix}:speaker_ref_mismatch:{ref}")
    if len(refs) != len(set(refs)) or any(not value for value in refs):
        errors.append("source_turn_registry_refs_invalid")
    if set(refs) != set(dialogue_by_ref) or len(refs) != len(dialogue_by_ref):
        errors.append("source_turn_registry_turn_set_mismatch")
    unsigned = deepcopy(dict(registry))
    signature = unsigned.pop("source_turn_registry_sha256", None)
    if signature != _sha_json(unsigned):
        errors.append("source_turn_registry_sha_invalid")
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "resolved_turn_count": len(refs) - sum(
            ref not in dialogue_by_ref for ref in refs),
        "forbidden_text_count": forbidden_text_count,
        "source_turn_registry_version": SOURCE_TURN_REGISTRY_VERSION,
    }

