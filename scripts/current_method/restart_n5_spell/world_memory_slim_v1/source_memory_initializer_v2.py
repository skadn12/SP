"""Dialogue-first World Memory initialization without utterance records."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .append_only_memory import initialize_append_only_memory
from .complete_dialogue_view import compose_complete_dialogue_view
from .provenance_ledger import initialize_provenance_ledger
from .source_turn_registry import (
    compile_source_turn_registry, validate_source_turn_registry,
)


SOURCE_MEMORY_INITIALIZER_VERSION_V2 = "dialogue_first_source_memory_initializer_v2"


def _entity_ref(root_id: str, display_name: str) -> str:
    digest = hashlib.sha256(display_name.strip().casefold().encode("utf-8")).hexdigest()[:12]
    return f"{root_id}:E:{digest}"


def initialize_source_world_memory_v2(
    root: Mapping[str, Any], parent: Mapping[str, Any], *,
    verified_source_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Initialize reusable world state while keeping transcript text separate."""
    root_id = str(root.get("root_id") or "")
    if not root_id or parent.get("sample_id") != root_id:
        raise ValueError("source_memory_v2_root_identity_mismatch")
    root_dialogue = root.get("dialogue") if isinstance(root.get("dialogue"), Mapping) else {}
    parent_dialogue = (parent.get("dialogue")
                       if isinstance(parent.get("dialogue"), Mapping) else {})
    if parent_dialogue.get("dialogue_hash") != root_dialogue.get("dialogue_hash"):
        raise ValueError("source_memory_v2_dialogue_hash_mismatch")
    turns = parent_dialogue.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("source_memory_v2_turns_missing")

    speaker_turn_refs: dict[str, list[str]] = {}
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("source_memory_v2_turn_invalid")
        speaker = str(turn.get("speaker") or "")
        turn_ref = str(turn.get("turn_id") or "")
        text = turn.get("text")
        if not speaker or not turn_ref or not isinstance(text, str) or not text.strip():
            raise ValueError("source_memory_v2_turn_contract")
        if not turn_ref.startswith(root_id + ":") and not turn_ref.startswith("D"):
            raise ValueError("source_memory_v2_cross_root_turn")
        speaker_turn_refs.setdefault(speaker, []).append(turn_ref)

    entities: list[dict[str, Any]] = []
    speaker_to_entity_ref: dict[str, str] = {}
    for speaker in sorted(speaker_turn_refs):
        ref = _entity_ref(root_id, speaker)
        speaker_to_entity_ref[speaker] = ref
        entities.append({
            "entity_id": ref, "entity_ref": ref, "display_name": speaker,
            "entity_type": "person", "source_turn_refs": speaker_turn_refs[speaker],
            "origin": "source_grounded", "verification_status": "source_grounded",
            "semantic_status": "source_grounded", "introduced_round": 0,
        })

    records: list[dict[str, Any]] = []
    original_inputs: list[dict[str, Any]] = []
    for raw in verified_source_records:
        row = deepcopy(dict(raw))
        ref = str(row.get("memory_ref") or row.get("memory_id") or "")
        if not ref or row.get("record_type") == "parent_utterance":
            raise ValueError("source_memory_v2_verified_record_invalid")
        if not ref.startswith(root_id + ":"):
            raise ValueError("source_memory_v2_verified_record_cross_root")
        if (row.get("origin") not in {None, "source_grounded"} or
                row.get("verification_status") not in {None, "source_grounded"}):
            raise ValueError("source_memory_v2_verified_record_not_source_grounded")
        row["memory_ref"] = ref
        row["origin"] = "source_grounded"
        row["verification_status"] = "source_grounded"
        records.append(row)
        original_inputs.append({
            "record_ref": ref,
            "support_turn_refs": list(row.get("support_turn_refs") or []),
            "authoritative_memory_ref": ref,
        })
    for entity in entities:
        original_inputs.append({
            "record_ref": entity["entity_ref"],
            "support_turn_refs": list(entity["source_turn_refs"]),
            "authoritative_memory_ref": entity["entity_ref"],
        })
    provenance = initialize_provenance_ledger(
        root_source_id=root_id, original_source_records=original_inputs)
    memory = initialize_append_only_memory(
        root_id=root_id, entities=entities, authoritative_records=records,
        provenance_ledger=provenance["provenance_ledger"])
    memory["source_initializer_version"] = SOURCE_MEMORY_INITIALIZER_VERSION_V2
    memory["original_dialogue_sha256"] = str(root_dialogue.get("dialogue_hash") or "")
    complete_dialogue = compose_complete_dialogue_view(
        original_parent=parent, speaker_to_entity_ref=speaker_to_entity_ref)
    turn_registry = compile_source_turn_registry(complete_dialogue)
    turn_validation = validate_source_turn_registry(turn_registry, complete_dialogue)
    if not turn_validation["valid"]:
        raise AssertionError("source_memory_v2_turn_registry_invalid")
    return {
        "initializer_version": SOURCE_MEMORY_INITIALIZER_VERSION_V2,
        "memory": memory,
        "provenance_partitions": provenance,
        "speaker_to_entity_ref": speaker_to_entity_ref,
        "complete_dialogue": complete_dialogue,
        "source_turn_registry": turn_registry,
        "source_turn_registry_validation": turn_validation,
        "source_statistics": {
            "entity_count": len(entities),
            "authoritative_record_count": len(records),
            "parent_utterance_record_count": 0,
            "source_grounded_record_count": len(records) + len(entities),
            "source_turn_registry_count": turn_registry["turn_count"],
        },
    }
