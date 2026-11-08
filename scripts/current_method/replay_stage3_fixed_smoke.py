#!/usr/bin/env python3
"""Materialize a saved fixed-G3 smoke reject without another API call.

This is deliberately narrow: it replays an already captured dialogue and final
QA proposal against the immutable graph artifact and parent row, runs the same
validators as generation, and writes a normal clean JSONL row only on success.
It never contacts the challenger or solver.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def required_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def get_reject_row(path: Path, line_number: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    if line_number < 1 or line_number > len(rows):
        raise ValueError(
            f"--reject-line={line_number} is outside {path} (rows={len(rows)})"
        )
    return rows[line_number - 1]


def matching_parent_row(parent_rows: list[dict[str, Any]], seed_sample_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in parent_rows
        if native.normalize_space(row.get("sample_id") or row.get("source_id"))
        == seed_sample_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one parent for seed_sample_id={seed_sample_id!r}, found {len(matches)}"
        )
    return matches[0]


def replay_one(
    artifact: dict[str, Any],
    reject: dict[str, Any],
    parent_rows: list[dict[str, Any]],
    *,
    tokenizer_path: str,
    min_added_tokens: int,
    min_added_turns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_record = required_object(reject.get("raw_record"), "reject.raw_record")
    seed_sample_id = native.normalize_space(reject.get("seed_sample_id"))
    if not seed_sample_id:
        raise ValueError("reject row has no seed_sample_id")
    if native.normalize_space(artifact.get("seed_sample_id")) != seed_sample_id:
        raise ValueError("fixed graph artifact and reject row have different parents")
    seed_row = matching_parent_row(parent_rows, seed_sample_id)
    target = required_object(reject.get("stage64_target"), "reject.stage64_target")
    current_event_plan = copy.deepcopy(
        required_object(artifact.get("current_event_plan"), "artifact.current_event_plan")
    )
    if current_event_plan.get("operation") != target.get("qa_operation"):
        raise ValueError("fixed graph operation does not match the reject target")

    anchor_to_event = required_object(
        current_event_plan.get("current_anchor_to_event_instance_id"),
        "current_event_plan.current_anchor_to_event_instance_id",
    )
    known_anchors = list(anchor_to_event) + native.collect_legacy_anchor_strings(seed_row)
    raw_parent_turns, parent_dialogue_source = native.parent_dialogue_turns_from_row(
        seed_row
    )
    parent_turns = native.solver_visible_parent_dialogue_turns(
        raw_parent_turns, known_anchors
    )

    dialogue_attempts = raw_record.get("api_dialogue_attempts")
    if not isinstance(dialogue_attempts, list) or not dialogue_attempts:
        raise ValueError("reject row has no saved dialogue attempt")
    dialogue_proposal = copy.deepcopy(
        required_object(dialogue_attempts[0].get("payload"), "dialogue payload")
    )
    appended_turns = list(
        native._as_dialogue_list(
            native.appended_dialogue_from_proposal(dialogue_proposal)
        )
    )
    sample_index = int(reject.get("generation_slot_id") or 0)
    candidate = native.validate_appended_dialogue_candidate(
        appended_dialogue_marked=appended_turns,
        event_alignment_proposal=dialogue_proposal.get("event_alignment_proposal"),
        current_event_plan=current_event_plan,
        anchor_to_event=anchor_to_event,
        parent_dialogue_turns=parent_turns,
        known_anchors=known_anchors,
        sample_index=sample_index,
        min_added_turns=min_added_turns,
        min_added_tokens=min_added_tokens,
        tokenizer_path=tokenizer_path,
        parent_baseline_tokens=0,
        fallback_min_added_tokens=0,
        emergency_min_added_tokens=0,
        require_evidence_dispersion=True,
        deterministic_scaffold_used=False,
        formal_mining=True,
        local_evidence_compiler_used=True,
        allow_length_shortfall=True,
    )
    if candidate.get("pass") is not True:
        raise native.NativeGraphDialogueReject(
            f"saved dialogue did not pass replay validation: {candidate.get('reason')}"
        )
    alignment_result = required_object(candidate.get("alignment_result"), "alignment result")
    current_alignment = required_object(
        alignment_result.get("current_alignment"), "current alignment"
    )
    final_dialogue = required_object(candidate.get("final_dialogue"), "final dialogue")
    hashes = {
        "graph_snapshot_hash": str(current_event_plan.get("graph_snapshot_hash")),
        "dialogue_hash": str(final_dialogue.get("dialogue_hash")),
        "alignment_hash": str(alignment_result.get("alignment_hash")),
    }
    graph_update = native.compile_graph_update_from_current_event_plan(
        current_event_plan, current_alignment, hashes
    )

    final_qa_attempts = raw_record.get("api_stage64_final_qa_attempts")
    if not isinstance(final_qa_attempts, list) or not final_qa_attempts:
        raise ValueError("reject row has no saved final-QA attempt")
    final_qa_payload = required_object(
        final_qa_attempts[-1].get("payload"), "final QA payload"
    )
    qa_proposal = required_object(final_qa_payload.get("qa_proposal"), "qa proposal")
    qa = native._build_stage64_api_qa_from_proposal(
        current_event_plan=current_event_plan,
        graph_update=graph_update,
        current_alignment=current_alignment,
        qa_proposal=copy.deepcopy(qa_proposal),
        hashes=hashes,
        stage64_target=target,
    )

    length_metadata = copy.deepcopy(candidate.get("length_metadata") or {})
    length_metadata.update(
        {
            "parent_dialogue_source": parent_dialogue_source,
            "formal_mining_local_evidence_compiler": True,
            "formal_mining_local_evidence_compiler_turn_count": len(
                (raw_record.get("long_form_local_evidence_compiler") or {}).get(
                    "provenance"
                )
                or []
            ),
        }
    )
    clean = native.build_clean_sample(
        sample_index=sample_index,
        seed_row=seed_row,
        current_event_plan=current_event_plan,
        final_dialogue=final_dialogue,
        graph_update=graph_update,
        qa=qa,
        hashes=hashes,
        length_metadata=length_metadata,
    )
    target_identity = native.normalize_space(reject.get("stage64_target_identity"))
    target_plan_sha256 = native.normalize_space(reject.get("stage64_target_plan_sha256"))
    clean.update(
        {
            "generation_slot_id": sample_index,
            "target_index": reject.get("target_index"),
            "stage64_target": copy.deepcopy(target),
            "stage64_target_identity": target_identity,
            "stage64_target_id": native.normalize_space(
                reject.get("stage64_target_id")
            )
            or target_identity,
            "stage64_target_plan_sha256": target_plan_sha256,
        }
    )
    clean["metadata"].update(
        {
            "generation_slot_id": sample_index,
            "stage64_target_index": reject.get("target_index"),
            "stage64_target_identity": target_identity,
            "stage64_target_id": native.normalize_space(
                reject.get("stage64_target_id")
            )
            or target_identity,
            "stage64_target_plan_sha256": target_plan_sha256,
            "offline_replayed_from_reject": True,
            "offline_replay_source_reject_error": reject.get("error"),
        }
    )
    native.validate_final_clean_sample(clean, known_anchors, hashes)
    provenance = {
        "schema_version": "stage3_fixed_smoke_offline_replay_v1",
        "offline_only": True,
        "api_calls_made": 0,
        "source_reject_error": reject.get("error"),
        "fixed_stage3_graph_artifact_id": artifact.get("artifact_id"),
        "seed_sample_id": seed_sample_id,
        "generation_slot_id": sample_index,
        "length_guard_pass": bool(length_metadata.get("length_guard_pass")),
        "length_guard_exemption_used": bool(
            length_metadata.get("length_guard_exemption_used")
        ),
        "length_delta_tokens": length_metadata.get("length_delta_tokens"),
        "length_guard_effective_delta_tokens": length_metadata.get(
            "length_guard_effective_delta_tokens"
        ),
        "qa_operation": qa.get("qa_operation"),
        "gold": qa.get("gold"),
        "query_plan_dependency_closure_refs": qa.get("query_plan", {}).get(
            "dependency_closure_refs"
        ),
    }
    return clean, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-graph-artifact", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reject-line", type=int, default=1)
    parser.add_argument("--solver-tokenizer", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--min-added-tokens", type=int, default=4000)
    parser.add_argument("--min-added-turns", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = required_object(
        json.loads(args.fixed_graph_artifact.read_text(encoding="utf-8")),
        "fixed graph artifact",
    )
    reject = get_reject_row(args.rejects, args.reject_line)
    clean, provenance = replay_one(
        artifact,
        reject,
        read_jsonl(args.parent),
        tokenizer_path=args.solver_tokenizer,
        min_added_tokens=args.min_added_tokens,
        min_added_turns=args.min_added_turns,
    )
    generation_dir = args.output_dir / "generation"
    clean_path = generation_dir / "clean_samples.jsonl"
    if clean_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {clean_path}; pass --overwrite")
    native.write_jsonl(clean_path, [clean])
    native.write_jsonl(generation_dir / "samples.jsonl", [clean])
    native.write_jsonl(generation_dir / "valid_samples.jsonl", [clean])
    native.write_json(
        args.output_dir / "offline_replay_summary.json",
        {
            **provenance,
            "clean_path": str(clean_path),
            "clean_count": 1,
        },
    )
    print(json.dumps({**provenance, "clean_path": str(clean_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
