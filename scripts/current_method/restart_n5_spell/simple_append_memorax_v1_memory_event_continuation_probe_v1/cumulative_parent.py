"""Build a fresh unified Parent from one promoted complete dialogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, sha256_file
from ..input_adapter import dialogue_content_hash
from ..unified_dialogue_root import validate_unified_dialogue_root


CUMULATIVE_PARENT_VERSION = "memory_event_cumulative_parent_v1"


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def build_cumulative_parent(
    expanded: Mapping[str, Any], *, source_path: Path, root_id: str,
) -> dict[str, Any]:
    source_turns = list(expanded.get("turns") or [])
    if not source_turns:
        raise ValueError("cumulative_parent_source_turns_required")
    if len(source_turns) != int(expanded.get("turn_count") or 0):
        raise ValueError("cumulative_parent_source_turn_count_mismatch")

    source_root_id = _text(expanded.get("root_id"))
    target_root_id = _text(root_id)
    if not source_root_id or not target_root_id:
        raise ValueError("cumulative_parent_root_id_required")
    source_sha = sha256_file(source_path)
    source_id = source_root_id.removeprefix("cc-train-") + "-cumulative-r1"

    session_order: list[str] = []
    session_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in source_turns:
        session_ref = _text(row.get("session_ref") or row.get("scene_ref"))
        if not session_ref:
            raise ValueError("cumulative_parent_source_session_ref_required")
        if session_ref not in session_rows:
            session_order.append(session_ref)
            session_rows[session_ref] = []
        session_rows[session_ref].append(row)

    turns: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for session_index, source_session_ref in enumerate(session_order, 1):
        session_id = f"{target_root_id}:parent_session_{session_index:02d}"
        session_turn_ids: list[str] = []
        for turn_index, row in enumerate(session_rows[source_session_ref], 1):
            speaker = _text(row.get("speaker"))
            text = _text(row.get("text"))
            if not speaker or not text:
                raise ValueError("cumulative_parent_source_turn_invalid")
            turn_id = (
                f"{target_root_id}:P{session_index:02d}:T{turn_index:03d}")
            session_turn_ids.append(turn_id)
            turns.append({
                "turn_id": turn_id,
                "session_id": session_id,
                "speaker": speaker,
                "text": text,
                "partition": "parent",
                "source_id": source_id,
                "source_session_index": session_index,
                "source_turn_index": turn_index,
            })
        sessions.append({
            "session_id": session_id,
            "source_session_index": session_index,
            "time_interval": f"cumulative_session_{session_index:02d}",
            "summary": (
                "Cumulative Parent session derived from " +
                source_session_ref + "."),
            "turn_ids": session_turn_ids,
        })

    result = {
        "schema_version": "unified_dialogue_root_v1",
        "root_id": target_root_id,
        "sample_id": target_root_id,
        "lineage_group": target_root_id,
        "source": {
            "dataset": "conversation_chronicles_cumulative_expansion",
            "split": "derived",
            "source_id": source_id,
            "source_row_index": 0,
            "source_path": str(source_path.resolve()),
            "source_file_sha256": source_sha,
            "source_record_sha256": source_sha,
            "source_record_sha256_scope": (
                "promotable_expanded_dialogue_file_bytes"),
        },
        "dialogue": {
            "dialogue_hash": dialogue_content_hash(turns),
            "session_count": len(sessions),
            "turn_count": len(turns),
            "turns": turns,
        },
        "session_metadata": sessions,
        "source_metadata": {
            "cumulative_parent_version": CUMULATIVE_PARENT_VERSION,
            "source_root_id": source_root_id,
            "source_complete_dialogue_sha256": _text(
                expanded.get("complete_dialogue_sha256")),
            "source_turn_count": len(source_turns),
            "turn_id_policy": "regenerated_parent_ids_for_fresh_root",
        },
    }
    validation = validate_unified_dialogue_root(result)
    if not validation["valid"]:
        raise ValueError(
            "cumulative_parent_invalid:" + ",".join(validation["errors"]))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-id", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"cumulative_parent_output_exists:{args.output}")
    expanded = json.loads(args.expanded_dialogue.read_text(encoding="utf-8"))
    result = build_cumulative_parent(
        expanded, source_path=args.expanded_dialogue, root_id=args.root_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "root_id": result["root_id"],
        "session_count": result["dialogue"]["session_count"],
        "turn_count": result["dialogue"]["turn_count"],
        "dialogue_hash": result["dialogue"]["dialogue_hash"],
        "output_sha256": sha256_file(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
