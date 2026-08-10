"""Read-only authoritative parent loading for slim-pipeline preflight."""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scripts.current_method.build_stage3_parent_v2 import (
    dialogue_and_turns,
    strip_historical_deterministic_padding,
    strip_legacy_qa_dialogue_scaffolding,
)

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from ..input_adapter import dialogue_content_hash
from ..unified_dialogue_root import (
    UNIFIED_DIALOGUE_ROOT_VERSION,
    read_unified_parent,
    validate_unified_dialogue_root,
)


LOCOMO_COMPAT_VERSION = "locomo_attested_jsonl_parent_compat_v1"
_ORIGINAL_TURN_ID = re.compile(r"^D([1-9][0-9]*):([1-9][0-9]*)$")


def load_unified_root_read_only(path: Path) -> dict[str, Any]:
    """Load an existing authoritative root through the common reader."""
    path = path.resolve()
    before = sha256_file(path)
    root = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_unified_dialogue_root(root)
    if not validation["valid"]:
        raise ValueError("unified_root_invalid:" + ",".join(validation["errors"]))
    parent = read_unified_parent(path)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("authoritative_unified_root_changed_during_read")
    return {
        "root": root,
        "parent": parent,
        "input_path": str(path),
        "input_sha256": before,
        "source_kind": "authoritative_unified_root",
        "temporary_unified_root_created": False,
    }


def _read_jsonl_record(path: Path, zero_based_index: int) -> tuple[bytes, dict[str, Any]]:
    if zero_based_index < 0:
        raise ValueError("jsonl_index_must_be_nonnegative")
    with path.open("rb") as handle:
        for index, raw_line in enumerate(handle):
            if index == zero_based_index:
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("locomo_authoritative_record_invalid_json") from error
                if not isinstance(value, dict):
                    raise ValueError("locomo_authoritative_record_not_object")
                return raw_line, value
    raise IndexError(f"locomo_authoritative_record_missing:{zero_based_index}")


def _identity_values(record: dict[str, Any]) -> dict[str, Any]:
    seed = record.get("seed") if isinstance(record.get("seed"), dict) else {}
    seed_source = (seed.get("source")
                   if isinstance(seed.get("source"), dict) else {})
    top_source = (record.get("seed_source")
                  if isinstance(record.get("seed_source"), dict) else {})
    return {
        "seed.parent_sample_id": seed.get("parent_sample_id"),
        "seed.source.parent_sample_id": seed_source.get("parent_sample_id"),
        "seed_source.parent_sample_id": top_source.get("parent_sample_id"),
    }


def _attested_parent_turns(
    record: dict[str, Any], attestation: dict[str, Any], *, source_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dialogue, turns = dialogue_and_turns(record)
    if not turns:
        raise ValueError("locomo_authoritative_dialogue_missing")
    dialogue, turns, qa_cleanup = strip_legacy_qa_dialogue_scaffolding(
        dialogue, turns, [], source_label=source_label)
    _, turns, padding_cleanup = strip_historical_deterministic_padding(
        dialogue, turns, [], source_label=source_label)
    content_sha = dialogue_content_hash(turns)
    if (len(turns) != attestation.get("turn_count") or
            content_sha != attestation.get("dialogue_content_sha256")):
        raise ValueError("locomo_attested_parent_dialogue_mismatch")
    return turns, {
        "raw_turn_count": len((record.get("dialogue") or {}).get("turns") or []),
        "qa_cleanup_version": qa_cleanup["cleanup_version"],
        "qa_cleanup_removed_turn_count": qa_cleanup["removed_turn_count"],
        "qa_cleanup_removed_turn_ids": qa_cleanup["removed_turn_ids"],
        "padding_cleanup_version": padding_cleanup["cleanup_version"],
        "padding_cleanup_signature_sha256": padding_cleanup[
            "known_padding_signature_sha256"],
        "padding_cleanup_removed_turn_count": padding_cleanup["removed_turn_count"],
        "padding_cleanup_removed_turn_ids": padding_cleanup["removed_turn_ids"],
        "attested_parent_turn_count": len(turns),
        "attested_parent_dialogue_sha256": content_sha,
    }


def build_locomo_attested_compat_root(
    source_path: Path, attestation_path: Path, *, line_number: int = 77,
    expected_identity: str = "conv-43", root_id: str = "locomo-conv-43",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a validated in-memory root from record 77 plus its attestation.

    The caller may temporarily serialize the result for ``read_unified_parent``;
    this function never writes beside or back into the source JSONL.
    """
    source_path = source_path.resolve()
    attestation_path = attestation_path.resolve()
    if line_number < 1:
        raise ValueError("locomo_line_number_must_be_positive")
    raw_line, record = _read_jsonl_record(source_path, line_number - 1)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if not isinstance(attestation, dict):
        raise ValueError("locomo_attestation_not_object")
    canonical_row_sha = sha256_bytes(canonical_json(record).encode())
    identity_values = _identity_values(record)
    checks = {
        "attestation_source_path": (
            Path(str(attestation.get("source_path") or "")).resolve() == source_path),
        "attestation_line_number": attestation.get("source_line_number") == line_number,
        "attestation_lineage": attestation.get("lineage") == expected_identity,
        "record_identity_fields": (
            bool(identity_values) and
            all(value == expected_identity for value in identity_values.values())),
        "source_row_canonical_sha256": (
            canonical_row_sha == attestation.get("source_row_sha256")),
        "attested_parent_source_id": (
            isinstance(attestation.get("parent_source_id"), str) and
            bool(attestation["parent_source_id"].strip())),
        "attested_parent_sample_id": (
            attestation.get("parent_sample_id") == attestation.get("parent_source_id")),
        "attested_parent_split": attestation.get("parent_split") == "train",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("locomo_attestation_identity_failed:" + ",".join(failed))
    source_turns, cleanup = _attested_parent_turns(
        record, attestation, source_label=f"{source_path}:{line_number}")

    session_order: list[str] = []
    session_turns: dict[str, list[dict[str, Any]]] = {}
    for turn in source_turns:
        original_turn_id = str(turn.get("turn_id") or "")
        match = _ORIGINAL_TURN_ID.fullmatch(original_turn_id)
        if match is None:
            raise ValueError(f"locomo_original_turn_id_invalid:{original_turn_id}")
        original_session = f"D{int(match.group(1))}"
        if original_session not in session_turns:
            session_order.append(original_session)
            session_turns[original_session] = []
        session_turns[original_session].append(turn)

    turns: list[dict[str, Any]] = []
    session_metadata: list[dict[str, Any]] = []
    seen_turn_ids: set[str] = set()
    for session_index, original_session in enumerate(session_order, 1):
        session_id = original_session
        ids: list[str] = []
        for turn_index, source_turn in enumerate(session_turns[original_session], 1):
            turn_id = str(source_turn["turn_id"])
            if turn_id in seen_turn_ids:
                raise ValueError(f"locomo_duplicate_original_turn_id:{turn_id}")
            seen_turn_ids.add(turn_id)
            ids.append(turn_id)
            turns.append({
                "turn_id": turn_id,
                "session_id": session_id,
                "speaker": str(source_turn.get("speaker") or ""),
                "text": str(source_turn.get("text") or ""),
                "partition": "parent",
                "source_id": str(attestation["parent_source_id"]),
                "source_session_index": session_index,
                "source_turn_index": turn_index,
            })
        session_metadata.append({
            "session_id": session_id,
            "source_session_index": session_index,
            "time_interval": "not_provided_by_authoritative_jsonl_record",
            "summary": "not_provided_by_authoritative_jsonl_record",
            "turn_ids": ids,
        })

    source_file_sha = sha256_file(source_path)
    root = {
        "schema_version": UNIFIED_DIALOGUE_ROOT_VERSION,
        "root_id": root_id,
        "sample_id": root_id,
        "lineage_group": root_id,
        "source": {
            "dataset": "locomo",
            "split": "train",
            "source_id": str(attestation["parent_source_id"]),
            "source_row_index": line_number - 1,
            "source_path": str(source_path),
            "source_file_sha256": source_file_sha,
            "source_record_sha256": canonical_row_sha,
            "source_record_sha256_scope": "canonical_json_record_utf8",
        },
        "dialogue": {
            "dialogue_hash": dialogue_content_hash(turns),
            "session_count": len(session_metadata),
            "turn_count": len(turns),
            "turns": turns,
        },
        "session_metadata": session_metadata,
        "source_metadata": {
            "compatibility_version": LOCOMO_COMPAT_VERSION,
            "authoritative_line_number": str(line_number),
            "attestation_path": str(attestation_path),
            "attestation_sha256": sha256_file(attestation_path),
            "attested_lineage": expected_identity,
            "attested_dialogue_content_sha256": str(
                attestation["dialogue_content_sha256"]),
            "raw_line_sha256": sha256_bytes(raw_line),
            "record_sample_id": str(record.get("sample_id") or ""),
            "record_source_id": str(record.get("source_id") or ""),
        },
    }
    validation = validate_unified_dialogue_root(
        root, expected_source_path=source_path,
        expected_source_file_sha256=source_file_sha)
    if not validation["valid"]:
        raise ValueError("locomo_compat_unified_root_invalid:" +
                         ",".join(validation["errors"]))
    audit = {
        "compatibility_version": LOCOMO_COMPAT_VERSION,
        "source_path": str(source_path),
        "source_line_number": line_number,
        "source_row_index": line_number - 1,
        "source_file_sha256": source_file_sha,
        "source_row_canonical_sha256": canonical_row_sha,
        "attestation_path": str(attestation_path),
        "attestation_sha256": sha256_file(attestation_path),
        "identity_values": identity_values,
        "identity_checks": checks,
        "all_identity_checks_pass": all(checks.values()),
        **cleanup,
        "compat_root_dialogue_sha256": root["dialogue"]["dialogue_hash"],
        "temporary_unified_root_only": True,
        "source_jsonl_write_count": 0,
    }
    return root, audit


def load_locomo_parent_read_only(
    source_path: Path, attestation_path: Path, *, line_number: int = 77,
    expected_identity: str = "conv-43",
) -> dict[str, Any]:
    """Use a temporary serialization to exercise the existing unified reader."""
    source_path = source_path.resolve()
    source_before = sha256_file(source_path)
    root, audit = build_locomo_attested_compat_root(
        source_path, attestation_path, line_number=line_number,
        expected_identity=expected_identity)
    with TemporaryDirectory(prefix="world-memory-slim-locomo-") as temp:
        temporary_path = Path(temp) / "unified_root.json"
        atomic_json(temporary_path, root)
        parent = read_unified_parent(temporary_path)
    source_after = sha256_file(source_path)
    if source_before != source_after:
        raise RuntimeError("locomo_authoritative_jsonl_changed_during_read")
    return {
        "root": root,
        "parent": parent,
        "input_path": str(source_path),
        "input_sha256": source_before,
        "source_kind": "attested_jsonl_record_77",
        "temporary_unified_root_created": True,
        "temporary_unified_root_persisted": False,
        "compatibility_audit": audit,
    }

