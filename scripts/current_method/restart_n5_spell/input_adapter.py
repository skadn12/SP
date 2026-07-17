from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any


def dialogue_content_hash(turns: list[dict[str, Any]]) -> str:
    normalize = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
    content = [{"speaker":normalize(turn.get("speaker") or turn.get("role")).casefold(),
        "session":normalize(turn.get("session_id") or turn.get("session") or turn.get("date")).casefold(),
        "text":normalize(turn.get("text") or turn.get("content") or turn.get("utterance")).casefold()}
        for turn in turns]
    material = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_run_sessions(source_root: Path, session_filenames: list[str]) -> dict[str, Any]:
    """Verify an immutable source run before importing any accepted session."""
    source_root = source_root.resolve()
    index_path = source_root / "accepted/index.json"
    manifest_path = source_root / "config/run_manifest.json"
    provenance_path = source_root / "reports/post_run_provenance.json"
    if not index_path.exists() or not manifest_path.exists() or not provenance_path.exists():
        raise RuntimeError("source_run_integrity_failed: required frozen provenance is missing")
    try:
        index = read_json(index_path)
        manifest = read_json(manifest_path)
        provenance = read_json(provenance_path)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("source_run_integrity_failed: unreadable frozen provenance") from error
    manifest_sha = _sha256_file(manifest_path)
    if (index.get("run_id") != manifest.get("run_id") or
            index.get("run_manifest_sha256") != manifest_sha or
            provenance.get("run_manifest_sha256") != manifest_sha):
        raise RuntimeError("source_run_integrity_failed: source identity binding mismatch")
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("source_run_not_authoritative")
    frozen_hashes = index.get("session_sha256")
    if not isinstance(frozen_hashes, dict):
        raise RuntimeError("source_run_integrity_failed: invalid accepted index")
    sessions, session_hashes = [], {}
    for filename in session_filenames:
        path = source_root / "accepted" / filename
        if not path.exists() or filename not in frozen_hashes:
            raise RuntimeError("source_run_integrity_failed: indexed session missing")
        actual = _sha256_file(path)
        if actual != frozen_hashes[filename]:
            raise RuntimeError(f"source_session_sha_mismatch: {filename}")
        sessions.append(read_json(path)); session_hashes[filename] = actual
    return {"source_root":str(source_root), "source_run_id":manifest["run_id"],
        "source_manifest_sha256":manifest_sha, "source_index_sha256":_sha256_file(index_path),
        "source_provenance_sha256":_sha256_file(provenance_path),
        "source_session_sha256":session_hashes, "sessions":sessions}


def find_parent(readonly_root: Path, sample_id: str, expected_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only adapter for legacy parent JSONL; never imports an entrypoint."""
    candidates = sorted(readonly_root.glob("*/reports/round_dataset_train_seed_rows.jsonl"), reverse=True)
    for path in candidates:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if sample_id not in line:
                    continue
                row = json.loads(line)
                if row.get("sample_id") != sample_id:
                    continue
                dialogue = row.get("dialogue") or {}
                if dialogue.get("dialogue_hash") != expected_hash:
                    continue
                turns = dialogue.get("turns") or []
                if not isinstance(turns, list) or dialogue_content_hash(turns) != expected_hash:
                    continue
                return row, {"source_path": str(path), "source_line_number": line_number,
                             "read_only": True, "dialogue_hash_verified": True,
                             "dialogue_hash_recomputed":True}
    raise FileNotFoundError(f"parent {sample_id} with expected hash not found")


def load_lineage(input_root: Path, lineage: str, legacy_root: Path) -> dict[str, Any]:
    lineage_root = input_root / lineage
    outline = read_json(lineage_root / "expansion_outline.json")
    anchors = read_json(lineage_root / "parent_anchor_snapshot.json")
    parent, provenance = find_parent(legacy_root, anchors["sample_id"], anchors["dialogue_hash"])
    return {"outline": outline, "anchors": anchors, "parent": parent, "provenance": provenance}
