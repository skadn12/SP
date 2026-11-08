"""Explicit Writer evidence sources; Memory entries are never transcript turns.

This module contains no retrieval or model calls.  A caller provides an already
frozen local projection.  Only its visible fields reach the Writer prompt.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import re
from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes


TRANSCRIPT_ONLY = "TRANSCRIPT_ONLY"
TRANSCRIPT_PLUS_MEMORY = "TRANSCRIPT_PLUS_MEMORY"
MEMORY_ONLY = "MEMORY_ONLY"
SOURCE_MODES = (TRANSCRIPT_ONLY, TRANSCRIPT_PLUS_MEMORY, MEMORY_ONLY)


def render_dialogue(expanded: Mapping[str, Any]) -> str:
    """Render frozen turns; dates are explicit conversation annotations only.

    The flag is opt-in so old primary/repair prompts stay byte-identical.
    No turn is rewritten, sorted, or assigned an inferred event time.
    """
    turns = expanded.get("turns") or []

    def plain(rows):
        return "\n".join(f"{x.get('turn_ref')} | {x.get('speaker')} | {x.get('text')}" for x in rows)

    mode = expanded.get("dialogue_rendering")
    if mode is None:
        return plain(turns)
    if mode != "conversation_dates_v1":
        raise ValueError("UNKNOWN_DIALOGUE_RENDERING")
    segments = expanded.get("conversation_date_segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("DATE_SEGMENTS_REQUIRED")
    chunks, cursor, seen = [], 0, set()
    for segment in segments:
        first, last = segment.get("first_turn_index_1_based"), segment.get("last_turn_index_1_based")
        if (type(first) is not int or type(last) is not int or first != cursor + 1
                or last < first or last > len(turns)):
            raise ValueError("DATE_SEGMENT_COVERAGE_INVALID")
        name, value = segment.get("segment_id"), segment.get("conversation_date")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9:_-]+", name) or name in seen:
            raise ValueError("DATE_SEGMENT_ID_INVALID")
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError("DATE_SEGMENT_DATE_INVALID")
        members = turns[first - 1:last]
        if (members[0].get("turn_ref"), members[-1].get("turn_ref")) != (
                segment.get("first_turn_ref"), segment.get("last_turn_ref")):
            raise ValueError("DATE_SEGMENT_REFERENCE_MISMATCH")
        chunks.extend((f"[Conversation date: {value} | Session: {name}]", plain(members)))
        cursor = last
        seen.add(name)
    if cursor != len(turns):
        raise ValueError("DATE_SEGMENT_TAIL_MISSING")
    return "\n".join(chunks)
MEMORY_FIELDS = ("memory_ref", "bucket_type", "key", "content")
ENVELOPE_FIELDS = ("evidence_source_mode", "memory_snapshot_sha256",
                   "memory_projection_sha256", "memory_items")
GENERIC_MEMORY_CONTRACT = {"intent_id": "memory_only_mixed",
                           "required_evidence_groups": {}, "memory_only_generic": True}


def projection_sha256(memory_items: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json(list(memory_items)).encode("utf-8"))


def source_mode(expanded: Mapping[str, Any]) -> str:
    mode = expanded.get("evidence_source_mode", TRANSCRIPT_ONLY)
    if not isinstance(mode, str) or mode not in SOURCE_MODES:
        raise ValueError("WRITER_SOURCE_MODE_INVALID")
    return str(mode)


def _memory_items(expanded: Mapping[str, Any]) -> list[dict[str, str]]:
    items = expanded.get("memory_items")
    if not isinstance(items, list):
        raise ValueError("WRITER_MEMORY_ITEMS_INVALID")
    visible = []
    for index, item in enumerate(items, 1):
        if (not isinstance(item, Mapping) or set(item) != set(MEMORY_FIELDS)
                or any(not isinstance(item[field], str) for field in MEMORY_FIELDS)
                or not item["content"].strip()
                or item["memory_ref"] != f"MEM_ITEM_{index:03d}"):
            raise ValueError("WRITER_MEMORY_ITEM_INVALID")
        visible.append({field: item[field] for field in MEMORY_FIELDS})
    for field in ("memory_snapshot_sha256", "memory_projection_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(expanded.get(field) or "")):
            raise ValueError(f"WRITER_MEMORY_IDENTITY_INVALID:{field}")
    if projection_sha256(visible) != expanded["memory_projection_sha256"]:
        raise ValueError("WRITER_MEMORY_PROJECTION_DRIFT")
    return visible


def validate_source_envelope(expanded: Mapping[str, Any]) -> str:
    mode = source_mode(expanded)
    if mode == MEMORY_ONLY:
        allowed = {"root_id", "turns", *ENVELOPE_FIELDS}
        if set(expanded) != allowed or expanded.get("turns") != []:
            raise ValueError("WRITER_MEMORY_ONLY_INPUT_LEAK")
        if not isinstance(expanded.get("root_id"), str) or not expanded["root_id"]:
            raise ValueError("WRITER_MEMORY_ROOT_INVALID")
        _memory_items(expanded)
    elif mode == TRANSCRIPT_PLUS_MEMORY:
        _memory_items(expanded)
    elif any(field in expanded for field in ENVELOPE_FIELDS[1:]):
        raise ValueError("WRITER_TRANSCRIPT_ONLY_MEMORY_PRESENT")
    return mode


def source_envelope(expanded: Mapping[str, Any], mode: str, *,
                    memory_items: Sequence[Mapping[str, Any]] | None = None,
                    snapshot_sha256: str | None = None,
                    projection_sha256_value: str | None = None) -> dict[str, Any]:
    """Construct one source view, discarding all transcript fields for Memory-only."""
    if mode not in SOURCE_MODES:
        raise ValueError("WRITER_SOURCE_MODE_INVALID")
    if source_mode(expanded) != TRANSCRIPT_ONLY:
        raise ValueError("WRITER_SOURCE_ENVELOPE_ALREADY_ADAPTED")
    if mode == TRANSCRIPT_ONLY:
        if memory_items is not None or snapshot_sha256 is not None or projection_sha256_value is not None:
            raise ValueError("WRITER_TRANSCRIPT_ONLY_MEMORY_PRESENT")
        result = deepcopy(dict(expanded))
        result["evidence_source_mode"] = mode
    else:
        result = ({"root_id": str(expanded.get("root_id") or ""), "turns": []}
                  if mode == MEMORY_ONLY else deepcopy(dict(expanded)))
        result.update(evidence_source_mode=mode, memory_items=deepcopy(list(memory_items or [])),
                      memory_snapshot_sha256=snapshot_sha256,
                      memory_projection_sha256=projection_sha256_value)
    validate_source_envelope(result)
    return result


def memory_only_envelope(root_id: str, memory_items: Sequence[Mapping[str, Any]], *,
                         snapshot_sha256: str, projection_sha256: str) -> dict[str, Any]:
    return source_envelope({"root_id": root_id}, MEMORY_ONLY, memory_items=memory_items,
                           snapshot_sha256=snapshot_sha256,
                           projection_sha256_value=projection_sha256)


def generic_memory_contract() -> dict[str, Any]:
    return deepcopy(GENERIC_MEMORY_CONTRACT)


def validate_memory_contract(contract: Mapping[str, Any]) -> None:
    if dict(contract) != GENERIC_MEMORY_CONTRACT:
        raise ValueError("WRITER_MEMORY_ONLY_INTENT_LEAK")


def evidence_catalog(expanded: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mode = validate_source_envelope(expanded)
    if mode == MEMORY_ONLY:
        return {item["memory_ref"]: item for item in _memory_items(expanded)}
    return {str(turn["turn_ref"]): dict(turn) for turn in expanded.get("turns") or []
            if isinstance(turn, Mapping) and isinstance(turn.get("turn_ref"), str)}


def transcript_envelope(expanded: Mapping[str, Any]) -> dict[str, Any]:
    if validate_source_envelope(expanded) == MEMORY_ONLY:
        raise ValueError("WRITER_MEMORY_ONLY_HAS_NO_TRANSCRIPT")
    return {key: value for key, value in expanded.items() if key not in ENVELOPE_FIELDS}


def source_identity(expanded: Mapping[str, Any]) -> dict[str, Any]:
    mode = validate_source_envelope(expanded)
    identity = {"evidence_source_mode": mode,
                "reference_namespace": "memory_ref" if mode == MEMORY_ONLY else "turn_ref"}
    if mode != TRANSCRIPT_ONLY:
        identity.update(evidence_root_id=expanded.get("root_id"),
                        memory_snapshot_sha256=expanded["memory_snapshot_sha256"],
                        memory_projection_sha256=expanded["memory_projection_sha256"])
    if mode != MEMORY_ONLY:
        identity["transcript_sha256"] = sha256_bytes(canonical_json(transcript_envelope(expanded)).encode())
    return identity


def memory_navigation_block(expanded: Mapping[str, Any]) -> str:
    mode = validate_source_envelope(expanded)
    if mode == TRANSCRIPT_ONLY:
        return ""
    header = ("FROZEN MEMORY NAVIGATION (selection aid only; not QA evidence):" if mode == TRANSCRIPT_PLUS_MEMORY
              else "FROZEN MEMORY ITEMS (the only evidence source for this batch):")
    boundary = (
        "Cite only exact current DIALOGUE turn_ref values for every candidate, event, and temporal relation. "
        "Memory may guide selection but cannot supply QA evidence or Gold. Current dialogue facts outside Memory remain usable. "
        "Absence from Memory does not mean an event did not happen."
        if mode == TRANSCRIPT_PLUS_MEMORY else
        "Cite only the exact memory_ref values below in evidence_refs and evidence_groups. "
        "These are Memory-source checks, not transcript evidence or a certification that Memory is true. "
        "No transcript or transcript-derived intent is supplied."
    )
    return "\n".join((header, boundary,
        "Do not infer event time from entry order, titles, keys, or Memory write time. Do not upgrade a plan into a completed event. "
        "Use only stated evidence; do not add common-sense chronology or fabricate an order to meet a quota.",
        json.dumps(_memory_items(expanded), ensure_ascii=False, separators=(",", ":"))))


def memory_ordering_guidance(policy: Mapping[str, Any]) -> list[str]:
    """Reuse all contract guidance and the same schema, omitting its transcript example."""
    from ..h4_r02_ordering_temporal_contract_v1.guidance import ordering_guidance
    lines = ordering_guidance(policy)
    result = []
    for line in lines:
        if line.startswith("SYNTHETIC ORDERING FORMAT EXAMPLE ONLY:"):
            break
        if line.startswith("Every node and relation uses exact turn_ref"):
            line = ("Every node and relation uses exact memory_ref strings from FROZEN MEMORY ITEMS. "
                    "Do not use DIALOGUE turn_ref values, positions, near-matching text, or guessed refs. "
                    "The program retains exact Memory excerpts and source identity; do not copy excerpts into new fields.")
        elif line.startswith("Keep discriminative_variable"):
            line = line.replace("exact current turn refs", "exact current memory_ref values")
        elif line.startswith("If the dialogue cannot support"):
            line = ("If Memory cannot support an ordering candidate, omit it and declare an INSUFFICIENT_EVIDENCE "
                    "abstention for ordering. Never fabricate a complete chain to meet the requested count.")
        result.append(line)
    return result
