"""Dataset-neutral frozen parent roots for restart-n5 generation inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes
from .input_adapter import dialogue_content_hash


UNIFIED_DIALOGUE_ROOT_VERSION = "unified_dialogue_root_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


UNIFIED_DIALOGUE_ROOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "root_id", "sample_id", "lineage_group",
        "source", "dialogue", "session_metadata", "source_metadata",
    ],
    "properties": {
        "schema_version": {"const": UNIFIED_DIALOGUE_ROOT_VERSION},
        "root_id": {"type": "string", "minLength": 1},
        "sample_id": {"type": "string", "minLength": 1},
        "lineage_group": {"type": "string", "minLength": 1},
        "source": {"type": "object"},
        "dialogue": {"type": "object"},
        "session_metadata": {
            "type": "array", "minItems": 1,
        },
        "source_metadata": {"type": "object"},
    },
}


def unified_root_schema_sha256() -> str:
    return sha256_bytes(canonical_json(UNIFIED_DIALOGUE_ROOT_SCHEMA).encode())


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_fields(value: Any, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def _check_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_unified_dialogue_root(
    root: Any, *, expected_source_path: Path | None = None,
    expected_source_file_sha256: str | None = None,
    expected_session_count: int | None = None,
) -> dict[str, Any]:
    """Validate one root, including order and provenance bindings."""
    errors: list[str] = []
    checks: dict[str, bool] = {}
    top_fields = {
        "schema_version", "root_id", "sample_id", "lineage_group",
        "source", "dialogue", "session_metadata", "source_metadata",
    }
    checks["top_level_schema"] = _exact_fields(root, top_fields)
    if not checks["top_level_schema"]:
        return {"valid": False, "errors": ["top_level_schema"],
                "checks": checks}

    root_id = root.get("root_id")
    source = root.get("source")
    dialogue = root.get("dialogue")
    sessions = root.get("session_metadata")
    source_metadata = root.get("source_metadata")
    checks["schema_version"] = (
        root.get("schema_version") == UNIFIED_DIALOGUE_ROOT_VERSION)
    checks["root_identity"] = (
        _text(root_id) and root.get("sample_id") == root_id and
        root.get("lineage_group") == root_id)

    source_fields = {
        "dataset", "split", "source_id", "source_row_index", "source_path",
        "source_file_sha256", "source_record_sha256",
        "source_record_sha256_scope",
    }
    checks["source_schema"] = _exact_fields(source, source_fields)
    if checks["source_schema"]:
        checks["source_identity"] = (
            _text(source.get("dataset")) and
            _text(source.get("split")) and
            _text(source.get("source_id")) and
            isinstance(source.get("source_row_index"), int) and
            source["source_row_index"] >= 0 and
            _text(source.get("source_path")) and
            Path(source["source_path"]).is_absolute() and
            _text(source.get("source_record_sha256_scope")))
        checks["source_sha256_fields"] = (
            _check_sha256(source.get("source_file_sha256")) and
            _check_sha256(source.get("source_record_sha256")))
        checks["source_path_binding"] = (
            expected_source_path is None or
            Path(source["source_path"]).resolve() ==
            expected_source_path.resolve())
        checks["source_file_sha_binding"] = (
            expected_source_file_sha256 is None or
            source.get("source_file_sha256") ==
            expected_source_file_sha256)
    else:
        checks.update({"source_identity": False,
                       "source_sha256_fields": False,
                       "source_path_binding": False,
                       "source_file_sha_binding": False})

    dialogue_fields = {"dialogue_hash", "session_count", "turn_count", "turns"}
    checks["dialogue_schema"] = _exact_fields(dialogue, dialogue_fields)
    turns = dialogue.get("turns") if checks["dialogue_schema"] else None
    checks["session_count_consistent"] = (
        isinstance(sessions, list) and bool(sessions) and
        dialogue.get("session_count") == len(sessions)
        if checks["dialogue_schema"] else False)
    if expected_session_count is not None:
        checks["expected_session_count"] = (
            isinstance(sessions, list) and
            len(sessions) == expected_session_count)
    checks["turn_container"] = (
        isinstance(turns, list) and bool(turns) and
        dialogue.get("turn_count") == len(turns)
        if checks["dialogue_schema"] else False)
    checks["source_metadata_schema"] = (
        isinstance(source_metadata, Mapping) and bool(source_metadata) and
        all(_text(key) for key in source_metadata))

    turn_fields = {
        "turn_id", "session_id", "speaker", "text", "partition",
        "source_id", "source_session_index", "source_turn_index",
    }
    turn_schema_ok = checks["turn_container"] and all(
        _exact_fields(turn, turn_fields) for turn in turns)
    checks["turn_schema"] = turn_schema_ok
    if turn_schema_ok and checks["source_schema"]:
        checks["speaker_text_nonempty"] = all(
            _text(turn["speaker"]) and _text(turn["text"]) for turn in turns)
        checks["all_parent"] = all(
            turn["partition"] == "parent" for turn in turns)
        checks["single_source_only"] = all(
            turn["source_id"] == source["source_id"] for turn in turns)
        ids = [turn["turn_id"] for turn in turns]
        checks["turn_ids_unique"] = (
            all(_text(value) for value in ids) and len(ids) == len(set(ids)))
        expected_order = sorted(
            turns, key=lambda turn: (
                turn["source_session_index"], turn["source_turn_index"]))
        checks["turn_order_preserved"] = list(turns) == expected_order
        checks["turn_indices_valid"] = all(
            isinstance(turn["source_session_index"], int) and
            1 <= turn["source_session_index"] <= len(sessions) and
            isinstance(turn["source_turn_index"], int) and
            turn["source_turn_index"] >= 1 for turn in turns)
    else:
        checks.update({"speaker_text_nonempty": False, "all_parent": False,
                       "single_source_only": False, "turn_ids_unique": False,
                       "turn_order_preserved": False,
                       "turn_indices_valid": False})

    session_fields = {
        "session_id", "source_session_index", "time_interval", "summary",
        "turn_ids",
    }
    session_schema_ok = isinstance(sessions, list) and all(
        _exact_fields(session, session_fields) for session in sessions)
    checks["session_metadata_schema"] = session_schema_ok
    if session_schema_ok and turn_schema_ok:
        checks["session_order_preserved"] = [
            session["source_session_index"] for session in sessions
        ] == list(range(1, len(sessions) + 1))
        checks["session_metadata_nonempty"] = all(
            _text(session["session_id"]) and
            _text(session["time_interval"]) and
            _text(session["summary"]) and
            isinstance(session["turn_ids"], list) and
            bool(session["turn_ids"]) for session in sessions)
        checks["session_turn_ids_exact"] = all(
            session["turn_ids"] == [
                turn["turn_id"] for turn in turns
                if turn["session_id"] == session["session_id"]
            ] for session in sessions)
        checks["turn_session_membership_exact"] = (
            {turn["session_id"] for turn in turns} ==
            {session["session_id"] for session in sessions})
    else:
        checks.update({"session_order_preserved": False,
                       "session_metadata_nonempty": False,
                       "session_turn_ids_exact": False,
                       "turn_session_membership_exact": False})

    checks["dialogue_hash_format"] = (
        checks["dialogue_schema"] and
        _check_sha256(dialogue.get("dialogue_hash")))
    checks["dialogue_hash_recomputed"] = (
        turn_schema_ok and dialogue.get("dialogue_hash") ==
        dialogue_content_hash([dict(turn) for turn in turns]))
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {"valid": not errors, "errors": errors, "checks": checks}


def validate_unified_root_collection(
    roots: Sequence[Mapping[str, Any]], *, expected_count: int = 10,
    expected_source_path: Path | None = None,
    expected_source_file_sha256: str | None = None,
    expected_session_count: int | None = None,
) -> dict[str, Any]:
    """Validate cross-root uniqueness in addition to each root contract."""
    per_root = [validate_unified_dialogue_root(
        root, expected_source_path=expected_source_path,
        expected_source_file_sha256=expected_source_file_sha256,
        expected_session_count=expected_session_count)
        for root in roots]
    root_ids = [str(root.get("root_id") or "") for root in roots]
    source_ids = [str((root.get("source") or {}).get("source_id") or "")
                  for root in roots]
    turn_ids = [str(turn.get("turn_id") or "") for root in roots
                for turn in ((root.get("dialogue") or {}).get("turns") or [])]
    global_checks = {
        "root_count_exact": len(roots) == expected_count,
        "root_ids_globally_unique": (
            len(root_ids) == len(set(root_ids)) and all(root_ids)),
        "source_ids_unique": (
            len(source_ids) == len(set(source_ids)) and all(source_ids)),
        "turn_ids_globally_unique": (
            len(turn_ids) == len(set(turn_ids)) and all(turn_ids)),
        "all_roots_valid": all(row["valid"] for row in per_root),
    }
    return {"valid": all(global_checks.values()),
            "global_checks": global_checks, "roots": per_root}


def read_unified_parent(path: Path) -> dict[str, Any]:
    """Read one root and expose only the existing generator parent contract."""
    with path.open(encoding="utf-8") as handle:
        root = json.load(handle)
    result = validate_unified_dialogue_root(root)
    if not result["valid"]:
        raise ValueError("unified_dialogue_root_invalid:" +
                         ",".join(result["errors"]))
    turns = [{key: turn[key] for key in (
        "turn_id", "session_id", "speaker", "text", "partition")}
        for turn in root["dialogue"]["turns"]]
    if dialogue_content_hash(turns) != root["dialogue"]["dialogue_hash"]:
        raise ValueError("unified_dialogue_root_reader_hash_mismatch")
    return {
        "sample_id": root["sample_id"],
        "lineage_group": root["lineage_group"],
        "dialogue": {
            "dialogue_hash": root["dialogue"]["dialogue_hash"],
            "turns": turns,
        },
    }
