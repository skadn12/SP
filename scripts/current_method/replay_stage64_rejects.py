#!/usr/bin/env python3
"""Offline replay of Stage64 QA candidates captured in reject audit JSONL.

The script never calls the challenger or solver. It reconstructs the normalized
semantic graph from each reject row, supplies synthetic evidence turn IDs, and
runs the current local Stage64 QA builder against the captured proposal. This is
intended for measuring whether a local parser/normalizer patch converts prior
candidate failures into passes.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native  # noqa: E402


SCHEMA_VERSION = "stage64_reject_offline_replay_v1"
EVENT_ID_RE = re.compile(r"^(?P<round>.+):S\d+:E\d+$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(payload)
    return rows


def proposal_event_ids(proposal: dict[str, Any]) -> list[str]:
    event_ids: list[str] = []
    for option in proposal.get("options") or []:
        if not isinstance(option, dict):
            continue
        for field in ("source_event_instance_id", "event_instance_id", "source_event_id"):
            value = str(option.get(field) or "").strip()
            if value and value not in event_ids:
                event_ids.append(value)
    return event_ids


def infer_round_id(proposal: dict[str, Any], fallback: str) -> str:
    for event_id in proposal_event_ids(proposal):
        match = EVENT_ID_RE.fullmatch(event_id)
        if match:
            return match.group("round")
    return fallback


def attach_proven_aliases(
    plan: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, str]:
    """Attach only marker aliases proven by an option's durable source ID."""

    known_event_ids = {
        str(event.get("event_instance_id") or "")
        for event in plan.get("events") or []
        if isinstance(event, dict)
    }
    alias_to_sources: dict[str, set[str]] = {}
    for option in proposal.get("options") or []:
        if not isinstance(option, dict):
            continue
        source = str(
            option.get("source_event_instance_id")
            or option.get("event_instance_id")
            or option.get("source_event_id")
            or ""
        ).strip()
        alias = str(option.get("anchor_ref") or "").strip()
        if source in known_event_ids and alias and alias != source:
            alias_to_sources.setdefault(alias, set()).add(source)

    proven = {
        alias: next(iter(sources))
        for alias, sources in alias_to_sources.items()
        if len(sources) == 1
    }
    event_to_anchor: dict[str, str] = {}
    for alias, event_id in sorted(proven.items()):
        event_to_anchor.setdefault(event_id, alias)
    plan["current_event_instance_to_anchor"] = event_to_anchor
    plan["current_anchor_to_event_instance_id"] = {
        alias: event_id for event_id, alias in event_to_anchor.items()
    }
    for event in plan.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_instance_id") or "")
        if event_id in event_to_anchor:
            event["alignment_anchor"] = event_to_anchor[event_id]
    return proven


def failure_family(reason: str) -> str:
    if "missing=['required_anchor_refs', 'supporting_anchor_refs']" in reason:
        return "partial_query_plan_required_supporting"
    if "refuses partial query_plan" in reason:
        return "other_partial_query_plan"
    if "query_plan completion normalizer does not support operation" in reason:
        return "partial_query_plan_unsupported_operation"
    if "refuses conflicting explicit query_plan." in reason:
        return "explicit_query_plan_gold_conflict"
    if "refuses to guess missing API fields" in reason:
        return "missing_api_fields"
    if "reasoning target drift" in reason:
        return "reasoning_target_drift"
    if "exact duplicate option text" in reason:
        return "exact_duplicate_options"
    if "semantic quality failed" in reason:
        return "semantic_quality"
    if "unresolved graph reference" in reason or "unresolved query_plan refs" in reason:
        return "unresolved_graph_reference"
    if "generic question" in reason:
        return "generic_question"
    return "other"


def _frozen_hashes(raw_record: dict[str, Any]) -> dict[str, str]:
    attempts = raw_record.get("api_stage64_final_qa_attempts")
    attempts = attempts if isinstance(attempts, list) else []
    last = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    frozen = last.get("frozen_dependencies")
    frozen = frozen if isinstance(frozen, dict) else {}
    return {
        "graph_snapshot_hash": str(frozen.get("graph_snapshot_hash") or "offline-graph"),
        "dialogue_hash": str(frozen.get("dialogue_hash") or "offline-dialogue"),
        "alignment_hash": str(frozen.get("alignment_hash") or "offline-alignment"),
    }


def replay_candidate(
    row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    line_number: int,
    fallback_round_id: str,
) -> dict[str, Any]:
    raw_record = row.get("raw_record") if isinstance(row.get("raw_record"), dict) else {}
    raw_graph = raw_record.get("api_graph_payload")
    proposal = candidate.get("proposal")
    target = row.get("stage64_target") or raw_record.get("stage64_target")
    if not isinstance(raw_graph, dict):
        raise ValueError("reject row has no api_graph_payload object")
    if not isinstance(proposal, dict):
        raise ValueError("candidate has no proposal object")
    if not isinstance(target, dict):
        raise ValueError("reject row has no stage64_target object")

    slot = int(row.get("generation_slot_id") or row.get("target_sample_index") or 0)
    round_id = infer_round_id(proposal, fallback_round_id)
    policy = {
        "stage64_target": copy.deepcopy(target),
        "semantic_graph_v2": True,
        "native_curriculum_mode": native.EASY_V2_MODE,
    }
    plan = native.normalize_current_event_plan(
        copy.deepcopy(raw_graph),
        {},
        question_type="single_choice",
        attack_skill=str(raw_graph.get("attack_skill") or "conditional_plan"),
        policy=policy,
        sample_index=slot,
        round_id=round_id,
    )
    proven_aliases = attach_proven_aliases(plan, proposal)
    alignment = {
        str(event["event_instance_id"]): {
            "turn_ids": [f"OFFLINE_T{slot:04d}_{index:02d}"]
        }
        for index, event in enumerate(plan.get("events") or [], start=1)
        if isinstance(event, dict) and event.get("event_instance_id")
    }
    hashes = _frozen_hashes(raw_record)
    graph_update = native.compile_graph_update_from_current_event_plan(
        plan, alignment, hashes
    )

    record: dict[str, Any] = {
        "line_number": line_number,
        "attempt": row.get("attempt"),
        "generation_slot_id": slot,
        "target_index": row.get("target_index"),
        "qa_operation": target.get("qa_operation"),
        "candidate_index": candidate.get("candidate_index"),
        "candidate_source": candidate.get("source"),
        "proposal_sha256": candidate.get("proposal_sha256"),
        "round_id": round_id,
        "proven_alias_count": len(proven_aliases),
        "original_build_failure_reasons": [
            str(attempt.get("failure_reason") or "")
            for attempt in candidate.get("build_attempts") or []
            if isinstance(attempt, dict) and attempt.get("failure_reason")
        ],
    }
    try:
        qa = native._build_stage64_api_qa_from_proposal(
            plan,
            graph_update,
            alignment,
            copy.deepcopy(proposal),
            hashes,
            target,
        )
    except Exception as exc:
        reason = str(exc)
        record.update(
            {
                "outcome": "FAIL",
                "failure_type": type(exc).__name__,
                "failure_reason": reason,
                "failure_family": failure_family(reason),
            }
        )
    else:
        record.update(
            {
                "outcome": "PASS",
                "failure_type": None,
                "failure_reason": None,
                "failure_family": None,
                "result_qa_operation": qa.get("qa_operation"),
                "result_answer_type": qa.get("answer_type"),
                "reasoning_identity_completion_used": bool(
                    qa.get("stage64_reasoning_identity_completion_normalizer")
                ),
            }
        )
    return record


def replay_rejects(
    rows: list[dict[str, Any]],
    *,
    candidate_scope: str,
    fallback_round_id: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=1):
        raw_record = row.get("raw_record") if isinstance(row.get("raw_record"), dict) else {}
        history = raw_record.get("stage64_qa_candidate_history")
        history = history if isinstance(history, list) else []
        candidates = [item for item in history if isinstance(item, dict)]
        if candidate_scope == "initial":
            candidates = [
                item
                for item in candidates
                if item.get("candidate_index") == 0
                or item.get("source") == "initial_final_qa"
            ][:1]
        if not candidates:
            skipped.append(
                {
                    "line_number": line_number,
                    "attempt": row.get("attempt"),
                    "reason": "no_selected_qa_candidate",
                    "top_level_reason": row.get("reason"),
                }
            )
            continue
        for candidate in candidates:
            try:
                records.append(
                    replay_candidate(
                        row,
                        candidate,
                        line_number=line_number,
                        fallback_round_id=fallback_round_id,
                    )
                )
            except Exception as exc:
                skipped.append(
                    {
                        "line_number": line_number,
                        "attempt": row.get("attempt"),
                        "candidate_index": candidate.get("candidate_index"),
                        "reason": "replay_setup_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    outcome_counts = Counter(record["outcome"] for record in records)
    exact_failure_counts = Counter(
        str(record["failure_reason"])
        for record in records
        if record["outcome"] == "FAIL"
    )
    family_counts = Counter(
        str(record["failure_family"])
        for record in records
        if record["outcome"] == "FAIL"
    )
    replay_count = len(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_scope": candidate_scope,
        "input_row_count": len(rows),
        "replay_candidate_count": replay_count,
        "pass_count": int(outcome_counts.get("PASS") or 0),
        "failure_count": int(outcome_counts.get("FAIL") or 0),
        "pass_rate": (
            round(int(outcome_counts.get("PASS") or 0) / replay_count, 6)
            if replay_count
            else None
        ),
        "skipped_count": len(skipped),
        "failure_family_counts": dict(family_counts.most_common()),
        "failure_reason_counts": dict(exact_failure_counts.most_common()),
        "records": records,
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay captured Stage64 QA reject candidates without any API calls."
    )
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument(
        "--candidate-scope", choices=("initial", "all"), default="initial"
    )
    parser.add_argument(
        "--fallback-round-id",
        default="current_method_stage3",
        help="Used only when no durable event ID reveals the original round prefix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always emitted.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-candidate records from stdout (the optional output remains complete).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = replay_rejects(
        read_jsonl(args.rejects),
        candidate_scope=args.candidate_scope,
        fallback_round_id=args.fallback_round_id,
    )
    summary["rejects"] = str(args.rejects.resolve())
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    stdout_payload = summary
    if args.summary_only:
        stdout_payload = {
            key: value
            for key, value in summary.items()
            if key not in {"records", "skipped"}
        }
    print(
        json.dumps(stdout_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
