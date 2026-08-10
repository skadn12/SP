"""Deterministic Conversation Chronicles to unified-root adapter."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .input_adapter import dialogue_content_hash
from .unified_dialogue_root import (
    UNIFIED_DIALOGUE_ROOT_VERSION, unified_root_schema_sha256,
    validate_unified_root_collection,
)


CONVERSATION_CHRONICLES_ADAPTER_VERSION = (
    "conversation_chronicles_unified_root_adapter_v1")
DEFAULT_SELECTION_SEED = 20260721
DEFAULT_ROOT_COUNT = 10
SESSION_NAMES = ("first", "second", "third", "fourth", "fifth")


def stream_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cc_source_record(record: Any) -> dict[str, Any]:
    """Validate only CC fields used by this adapter; QA fields are untouched."""
    checks: dict[str, bool] = {
        "record_object": isinstance(record, Mapping),
    }
    if not checks["record_object"]:
        return {"valid": False, "checks": checks,
                "errors": ["record_object"]}
    source_id = record.get("dataID")
    checks["source_id_nonempty"] = (
        isinstance(source_id, str) and bool(source_id.strip()))
    checks["relationship_nonempty"] = (
        isinstance(record.get("relationship"), str) and
        bool(record["relationship"].strip()))
    intervals = record.get("time_interval")
    summaries = record.get("summary")
    checks["time_interval_count"] = (
        isinstance(intervals, list) and len(intervals) == 5 and
        all(isinstance(value, str) and value.strip() for value in intervals))
    checks["summary_count"] = (
        isinstance(summaries, list) and len(summaries) == 5 and
        all(isinstance(value, str) and value.strip() for value in summaries))
    complete = True
    aligned = True
    nonempty = True
    for name in SESSION_NAMES:
        dialogue = record.get(f"{name}_session_dialogue")
        speakers = record.get(f"{name}_session_speakers")
        if not isinstance(dialogue, list) or not isinstance(speakers, list):
            complete = aligned = nonempty = False
            continue
        complete = complete and bool(dialogue) and bool(speakers)
        aligned = aligned and len(dialogue) == len(speakers)
        nonempty = nonempty and all(
            isinstance(value, str) and value.strip()
            for value in [*dialogue, *speakers])
    checks["exactly_five_complete_sessions"] = complete
    checks["dialogue_speaker_arrays_aligned"] = aligned
    checks["speaker_text_nonempty"] = nonempty
    errors = [name for name, passed in checks.items() if not passed]
    return {"valid": not errors, "checks": checks, "errors": errors}


def _safe_source_id(source_id: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", source_id.strip().casefold()).strip("-")
    if not value:
        raise ValueError("cc_source_id_cannot_form_root_id")
    return value


def _root_id(source_id: str) -> str:
    return f"cc-train-{_safe_source_id(source_id)}"


def _session_id(root_id: str, session_index: int) -> str:
    return f"{root_id}:parent_session_{session_index:02d}"


def _turn_id(root_id: str, session_index: int, turn_index: int) -> str:
    return f"{root_id}:P{session_index:02d}:T{turn_index:03d}"


def build_unified_root(
    record: Mapping[str, Any], *, source_path: Path,
    source_file_sha256: str, source_row_index: int,
    source_record_sha256: str,
) -> dict[str, Any]:
    check = validate_cc_source_record(record)
    if not check["valid"]:
        raise ValueError("cc_source_record_invalid:" +
                         ",".join(check["errors"]))
    source_id = str(record["dataID"]).strip()
    root_id = _root_id(source_id)
    turns: list[dict[str, Any]] = []
    session_metadata: list[dict[str, Any]] = []
    for session_index, name in enumerate(SESSION_NAMES, 1):
        session_id = _session_id(root_id, session_index)
        session_turn_ids: list[str] = []
        dialogue = record[f"{name}_session_dialogue"]
        speakers = record[f"{name}_session_speakers"]
        for turn_index, (speaker, text) in enumerate(
                zip(speakers, dialogue), 1):
            turn_id = _turn_id(root_id, session_index, turn_index)
            session_turn_ids.append(turn_id)
            turns.append({
                "turn_id": turn_id, "session_id": session_id,
                "speaker": str(speaker), "text": str(text),
                "partition": "parent", "source_id": source_id,
                "source_session_index": session_index,
                "source_turn_index": turn_index,
            })
        session_metadata.append({
            "session_id": session_id,
            "source_session_index": session_index,
            "time_interval": str(record["time_interval"][session_index - 1]),
            "summary": str(record["summary"][session_index - 1]),
            "turn_ids": session_turn_ids,
        })
    return {
        "schema_version": UNIFIED_DIALOGUE_ROOT_VERSION,
        "root_id": root_id, "sample_id": root_id,
        "lineage_group": root_id,
        "source": {
            "dataset": "conversation_chronicles", "split": "train",
            "source_id": source_id, "source_row_index": source_row_index,
            "source_path": str(source_path.resolve()),
            "source_file_sha256": source_file_sha256,
            "source_record_sha256": source_record_sha256,
            "source_record_sha256_scope":
                "raw_jsonl_line_bytes_including_line_terminator",
        },
        "dialogue": {
            "dialogue_hash": dialogue_content_hash(turns),
            "session_count": 5, "turn_count": len(turns), "turns": turns,
        },
        "session_metadata": session_metadata,
        "source_metadata": {
            "relationship": str(record["relationship"]),
        },
    }


def validate_source_mapping(
    root: Mapping[str, Any], record: Mapping[str, Any],
) -> dict[str, bool]:
    """Prove that adapter output preserves the selected source record."""
    expected_pairs: list[tuple[str, str, int, int]] = []
    for session_index, name in enumerate(SESSION_NAMES, 1):
        expected_pairs.extend(
            (str(speaker), str(text), session_index, turn_index)
            for turn_index, (speaker, text) in enumerate(zip(
                record[f"{name}_session_speakers"],
                record[f"{name}_session_dialogue"]), 1))
    actual_pairs = [(turn["speaker"], turn["text"],
                     turn["source_session_index"],
                     turn["source_turn_index"])
                    for turn in root["dialogue"]["turns"]]
    return {
        "source_id_exact": root["source"]["source_id"] == record["dataID"],
        "session_count_exact": root["dialogue"]["session_count"] == 5,
        "turn_order_and_content_exact": actual_pairs == expected_pairs,
        "time_interval_metadata_exact": [
            row["time_interval"] for row in root["session_metadata"]
        ] == record["time_interval"],
        "summary_metadata_exact": [
            row["summary"] for row in root["session_metadata"]
        ] == record["summary"],
        "relationship_metadata_exact": (
            root["source_metadata"]["relationship"] ==
            record["relationship"]),
        "summary_not_in_dialogue_turns": all(
            turn["text"] not in set(record["summary"])
            for turn in root["dialogue"]["turns"]),
        "time_interval_not_in_dialogue_turns": all(
            turn["text"] not in set(record["time_interval"])
            for turn in root["dialogue"]["turns"]),
    }


def select_cc_records(
    source_path: Path, *, count: int = DEFAULT_ROOT_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    """Select the fixed hash-priority sample while hashing the full source."""
    if count <= 0:
        raise ValueError("cc_selection_count_must_be_positive")
    source_path = source_path.resolve()
    file_digest = hashlib.sha256()
    # A max-heap implemented with negative integer priorities retains the
    # lexicographically smallest seeded SHA256 priorities.
    heap: list[tuple[int, str, int, bytes, dict[str, Any]]] = []
    seen_source_ids: set[str] = set()
    total_records = valid_records = invalid_json = invalid_schema = 0
    with source_path.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            total_records += 1
            file_digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_json += 1
                continue
            check = validate_cc_source_record(record)
            if not check["valid"]:
                invalid_schema += 1
                continue
            source_id = str(record["dataID"]).strip()
            if source_id in seen_source_ids:
                raise ValueError(f"cc_source_id_duplicate:{source_id}")
            seen_source_ids.add(source_id)
            valid_records += 1
            priority = int(hashlib.sha256(
                f"{seed}\0{source_id}".encode()).hexdigest(), 16)
            entry = (-priority, source_id, row_index, raw_line, dict(record))
            if len(heap) < count:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    if len(heap) != count:
        raise RuntimeError(
            f"cc_not_enough_valid_unique_records:{len(heap)}/{count}")
    source_file_sha256 = file_digest.hexdigest()
    selected = sorted(heap, key=lambda row: row[2])
    records = [{
        "source_id": source_id, "source_row_index": row_index,
        "source_record_sha256": hashlib.sha256(raw_line).hexdigest(),
        "raw_line": raw_line, "record": record,
    } for _, source_id, row_index, raw_line, record in selected]
    return {
        "source_path": source_path,
        "source_file_sha256": source_file_sha256,
        "selection_seed": seed, "selection_count": count,
        "selection_algorithm": "seeded_source_id_sha256_lowest_v1",
        "source_record_count": total_records,
        "valid_source_record_count": valid_records,
        "invalid_json_record_count": invalid_json,
        "invalid_schema_record_count": invalid_schema,
        "records": records,
    }


def prepare_cc_roots(
    source_path: Path, *, count: int = DEFAULT_ROOT_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    selection = select_cc_records(source_path, count=count, seed=seed)
    roots = [build_unified_root(
        item["record"], source_path=selection["source_path"],
        source_file_sha256=selection["source_file_sha256"],
        source_row_index=item["source_row_index"],
        source_record_sha256=item["source_record_sha256"])
        for item in selection["records"]]
    collection = validate_unified_root_collection(
        roots, expected_count=count,
        expected_source_path=selection["source_path"],
        expected_source_file_sha256=selection["source_file_sha256"],
        expected_session_count=5)
    mappings = [validate_source_mapping(root, item["record"])
                for root, item in zip(roots, selection["records"])]
    record_sha_checks = [
        root["source"]["source_record_sha256"] ==
        hashlib.sha256(item["raw_line"]).hexdigest()
        for root, item in zip(roots, selection["records"])]
    return {
        **{key: value for key, value in selection.items()
           if key != "records"},
        "adapter_version": CONVERSATION_CHRONICLES_ADAPTER_VERSION,
        "unified_root_schema_version": UNIFIED_DIALOGUE_ROOT_VERSION,
        "unified_root_schema_sha256": unified_root_schema_sha256(),
        "roots": roots, "collection_validation": collection,
        "source_mapping_checks": mappings,
        "source_record_sha_checks": record_sha_checks,
        "selected_records": selection["records"],
    }
