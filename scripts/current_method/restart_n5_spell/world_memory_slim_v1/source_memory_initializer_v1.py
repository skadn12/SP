"""Original-root-only initialization for append-only World Memory v1."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .append_only_memory import initialize_append_only_memory
from .provenance_ledger import initialize_provenance_ledger


SOURCE_MEMORY_INITIALIZER_VERSION = "provenance_aware_source_memory_initializer_v1"


def _entity_ref(root_id: str, display_name: str) -> str:
    digest = hashlib.sha256(display_name.strip().casefold().encode("utf-8")).hexdigest()[:12]
    return f"{root_id}:E:{digest}"


def initialize_source_world_memory(
    root: Mapping[str, Any], parent: Mapping[str, Any],
) -> dict[str, Any]:
    """Create source truth only from original root spans, never promoted text."""
    root_id = str(root.get("root_id") or "")
    if not root_id or parent.get("sample_id") != root_id:
        raise ValueError("source_memory_root_identity_mismatch")
    root_dialogue = root.get("dialogue") if isinstance(root.get("dialogue"), Mapping) else {}
    parent_dialogue = parent.get("dialogue") if isinstance(parent.get("dialogue"), Mapping) else {}
    if parent_dialogue.get("dialogue_hash") != root_dialogue.get("dialogue_hash"):
        raise ValueError("source_memory_dialogue_hash_mismatch")
    turns = parent_dialogue.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("source_memory_turns_missing")

    speaker_turn_refs: dict[str, list[str]] = {}
    for turn in turns:
        speaker = str(turn.get("speaker") or "")
        turn_ref = str(turn.get("turn_id") or "")
        text = turn.get("text")
        if not speaker or not turn_ref or not isinstance(text, str) or not text.strip():
            raise ValueError("source_memory_turn_invalid")
        if not turn_ref.startswith(root_id + ":") and not turn_ref.startswith("D"):
            raise ValueError("source_memory_cross_root_turn")
        speaker_turn_refs.setdefault(speaker, []).append(turn_ref)

    entities: list[dict[str, Any]] = []
    entity_by_speaker: dict[str, str] = {}
    for speaker in sorted(speaker_turn_refs):
        ref = _entity_ref(root_id, speaker)
        entity_by_speaker[speaker] = ref
        entities.append({
            "entity_id": ref, "entity_ref": ref, "display_name": speaker,
            "entity_type": "person", "source_turn_refs": list(speaker_turn_refs[speaker]),
            "origin": "source_grounded", "verification_status": "source_grounded",
            "semantic_status": "source_grounded", "introduced_round": 0,
        })

    records: list[dict[str, Any]] = []
    original_ledger_inputs: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, 1):
        turn_ref = str(turn["turn_id"])
        memory_ref = f"{root_id}:M{index:06d}"
        session_ref = str(turn.get("session_id") or "")
        record = {
            "memory_ref": memory_ref,
            "logical_key": f"source_utterance:{turn_ref}",
            "record_type": "parent_utterance",
            "value": str(turn["text"]),
            "payload": {"speaker_ref": entity_by_speaker[str(turn["speaker"])],
                        "text": str(turn["text"])},
            "subject_refs": [entity_by_speaker[str(turn["speaker"])]],
            "authoritative_version": 1,
            "origin": "source_grounded", "verification_status": "source_grounded",
            "proposal_ref": None, "scene_ref": None, "round_index": 0,
            "scene_index": 0, "event_sequence_index": index,
            "global_time_index": index, "narrative_time_expression": None,
            "support_turn_refs": [turn_ref], "evidence_text_sha256": [
                hashlib.sha256(str(turn["text"]).encode("utf-8")).hexdigest()],
            "formal_thread_id": None, "handoff_id": None,
            "allocated_entity_ref": None, "allocated_event_ref": None,
            "allocated_relation_ref": None,
            "provenance": {"root_source_id": root_id,
                           "source_session_refs": [session_ref] if session_ref else [],
                           "source_turn_refs": [turn_ref]},
        }
        records.append(record)
        original_ledger_inputs.append({
            "record_ref": memory_ref, "support_turn_refs": [turn_ref],
            "authoritative_memory_ref": memory_ref,
        })
    for entity in entities:
        original_ledger_inputs.append({
            "record_ref": entity["entity_ref"],
            "support_turn_refs": entity["source_turn_refs"],
            "authoritative_memory_ref": entity["entity_ref"],
        })
    provenance = initialize_provenance_ledger(
        root_source_id=root_id, original_source_records=original_ledger_inputs)
    memory = initialize_append_only_memory(
        root_id=root_id, entities=entities, authoritative_records=records,
        provenance_ledger=provenance["provenance_ledger"])
    memory["source_initializer_version"] = SOURCE_MEMORY_INITIALIZER_VERSION
    memory["original_dialogue_sha256"] = str(root_dialogue.get("dialogue_hash") or "")
    return {
        "initializer_version": SOURCE_MEMORY_INITIALIZER_VERSION,
        "memory": memory,
        "provenance_partitions": provenance,
        "speaker_to_entity_ref": entity_by_speaker,
        "source_statistics": {"entity_count": len(entities),
                              "authoritative_record_count": len(records),
                              "source_grounded_record_count": len(records) + len(entities)},
    }


def authoritative_affordance_view(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Compact compatibility view for the v3 affordance compiler only."""
    records = list(memory.get("authoritative_records") or [])
    return {
        "root_id": memory["root_id"],
        "entities": [{
            "entity_id": row.get("entity_ref") or row.get("entity_id"),
            "display_name": row["display_name"],
            "semantic_status": row.get("verification_status"),
            "verification_status": row.get("verification_status"),
        } for row in memory.get("entities") or []],
        "authoritative_records": records,
        "event_history": [row for row in records if row.get("record_type") in {
            "parent_utterance", "event"}],
        "current_states": [row for row in records if row.get("record_type") in {
            "state", "relationship_state", "responsibility"}],
        "open_threads": [row for row in records if row.get("record_type") == "thread"],
        "relations": [row for row in records if row.get("record_type") == "relation"],
        "active_head_index": dict(memory.get("active_head_index") or {}),
    }
