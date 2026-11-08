#!/usr/bin/env python3
"""Persist a verified Stage3 child graph independently of dialogue attempts.

The graph challenger proposes a child delta once.  If later dialogue rendering
fails, that failure must not discard the valid graph or cause a new graph API
call.  This utility binds one raw graph proposal to its canonical Stage2 parent
and freezes the child alignment anchors into a reusable artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for directory in (ROOT, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.current_method import native_api_graph_dialogue_expansion as native  # noqa: E402
from scripts.current_method import semantic_graph_v2  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def child_anchor_map(raw_record: dict[str, Any]) -> dict[str, str]:
    """Recover the exact child markers validated during the failed dialogue run."""

    candidates: list[dict[str, Any]] = []
    for attempt in raw_record.get("api_dialogue_attempts") or []:
        if not isinstance(attempt, dict):
            continue
        meta = attempt.get("meta") if isinstance(attempt.get("meta"), dict) else {}
        preflight = (
            meta.get("exact_claim_preflight")
            if isinstance(meta.get("exact_claim_preflight"), dict)
            else {}
        )
        if preflight.get("pass") is True:
            candidates.append(preflight)
    if not candidates:
        raise ValueError("raw graph record has no passing exact_claim_preflight anchor ledger")
    provenance = candidates[-1].get("claim_provenance") or []
    mapping: dict[str, str] = {}
    for item in provenance:
        if not isinstance(item, dict):
            continue
        event_id = native.normalize_space(item.get("event_instance_id"))
        marker = native.normalize_space(item.get("marker"))
        if not event_id or not marker:
            continue
        prior = mapping.get(event_id)
        if prior is not None and prior != marker:
            raise ValueError(f"child event {event_id} has conflicting markers")
        mapping[event_id] = marker
    if not mapping:
        raise ValueError("passing preflight did not contain any child event markers")
    return mapping


def materialize_artifact(
    *,
    raw_record: dict[str, Any],
    parent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    child_graph = raw_record.get("api_graph_payload")
    if not isinstance(child_graph, dict) or not isinstance(child_graph.get("events"), list):
        raise ValueError("raw record has no graph payload with events")
    graph_meta = raw_record.get("api_graph_meta") if isinstance(raw_record.get("api_graph_meta"), dict) else {}
    graph_quality = (
        graph_meta.get("semantic_graph_v2_quality")
        if isinstance(graph_meta.get("semantic_graph_v2_quality"), dict)
        else {}
    )
    if graph_quality.get("pass") is not True:
        raise ValueError("raw child graph did not pass semantic_graph_v2_quality")

    seed_id = native.normalize_space(raw_record.get("seed_sample_id"))
    parent_id = native.normalize_space(child_graph.get("parent_graph_id"))
    parent_row = next(
        (
            row
            for row in parent_rows
            if native.normalize_space(row.get("sample_id") or row.get("source_id")) == seed_id
            and isinstance(row.get("semantic_parent_graph"), dict)
            and native.normalize_space(row["semantic_parent_graph"].get("graph_id"))
            == parent_id
        ),
        None,
    )
    if parent_row is None:
        raise ValueError(
            "could not find the exact Stage2 parent row for raw seed_sample_id and parent_graph_id"
        )
    parent_graph = copy.deepcopy(parent_row["semantic_parent_graph"])
    stage64_target = (
        copy.deepcopy(raw_record.get("stage64_target"))
        if isinstance(raw_record.get("stage64_target"), dict)
        else {}
    )
    target_identity = native.normalize_space(raw_record.get("stage64_target_identity"))
    operation = native.normalize_space(stage64_target.get("qa_operation"))
    if not target_identity or not operation:
        raise ValueError("raw record lacks its immutable Stage3 target identity or operation")

    raw_quality_issues = semantic_graph_v2.validate_graph_delta(
        child_graph,
        operation=operation,
        reasoning_target=stage64_target,
    )
    if raw_quality_issues:
        raise ValueError(f"raw child graph failed current semantic validation: {raw_quality_issues[:4]}")

    policy = {
        "qa_operation_answer_type_diversification": True,
        "stage64_target": stage64_target,
        "item_difficulty_target": "boundary",
        "native_curriculum_mode": "r3_long_easy_v2",
        "planned_cell": copy.deepcopy(raw_record.get("planned_cell") or {}),
    }
    plan = native.normalize_current_event_plan(
        child_graph,
        parent_row,
        question_type=native.normalize_space(child_graph.get("question_type")),
        attack_skill=native.normalize_space(child_graph.get("attack_skill")),
        policy=policy,
        sample_index=int(raw_record.get("target_sample_index") or 0),
        round_id="current_method_stage3",
    )
    if native._is_v4_memory_parent(parent_row):
        plan = native.stage3_parent_continuity.materialize_child_global_time(
            plan,
            native._graph_parent_source_events(parent_row),
        )
        plan.setdefault("metadata", {})
        if isinstance(plan.get("metadata"), dict):
            if isinstance(graph_meta.get("parent_event_retrieval"), dict):
                plan["metadata"]["parent_event_retrieval"] = copy.deepcopy(
                    graph_meta["parent_event_retrieval"]
                )
            continuity_audit = (
                graph_meta.get("parent_child_continuity_audit")
                if isinstance(graph_meta.get("parent_child_continuity_audit"), dict)
                else {}
            )
            if continuity_audit:
                plan["metadata"]["parent_child_continuity_audit"] = copy.deepcopy(
                    continuity_audit
                )
                child_metadata = (
                    child_graph.get("metadata")
                    if isinstance(child_graph.get("metadata"), dict)
                    else {}
                )
                raw_continuity = (
                    child_metadata.get("parent_continuity")
                    if isinstance(child_metadata.get("parent_continuity"), dict)
                    else {}
                )
                plan["metadata"]["parent_continuity"] = {
                    "depends_on_parent_event_ids": copy.deepcopy(
                        continuity_audit.get("depends_on_parent_event_ids") or []
                    ),
                    "parent_topic_anchor_event_ids": copy.deepcopy(
                        continuity_audit.get("parent_topic_anchor_event_ids") or []
                    ),
                    "continuity_relations": copy.deepcopy(
                        raw_continuity.get("continuity_relations") or []
                    ),
                    "continuity_score": continuity_audit.get("continuity_score"),
                    "continuity_reasons": copy.deepcopy(
                        continuity_audit.get("continuity_reasons") or []
                    ),
                }
        plan["graph_snapshot_hash"] = native.compute_graph_snapshot_hash(plan)
    markers = child_anchor_map(raw_record)
    plan_event_ids = {
        native.normalize_space(event.get("event_instance_id"))
        for event in plan.get("events") or []
        if isinstance(event, dict) and native.normalize_space(event.get("event_instance_id"))
    }
    if set(markers) != plan_event_ids:
        raise ValueError(
            "child marker ledger does not cover exactly the normalized child graph events"
        )
    plan = copy.deepcopy(plan)
    for event in plan.get("events") or []:
        if isinstance(event, dict):
            event_id = native.normalize_space(event.get("event_instance_id"))
            event["alignment_anchor"] = markers[event_id]
    plan["current_event_instance_to_anchor"] = dict(sorted(markers.items()))
    plan["current_anchor_to_event_instance_id"] = {
        marker: event_id for event_id, marker in sorted(markers.items())
    }
    plan_issues = native.semantic_graph_plan_quality_issues(
        plan, operation, stage64_target
    )
    if plan_issues:
        raise ValueError(f"normalized fixed plan failed semantic validation: {plan_issues[:4]}")

    parent_anchor_ledger = {
        native.normalize_space(event.get("event_id")): list(event.get("evidence_turn_ids") or [])
        for event in parent_graph.get("events") or []
        if isinstance(event, dict) and native.normalize_space(event.get("event_id"))
    }
    durable_child_graph = copy.deepcopy(child_graph)
    plan_by_origin_event_id = {
        native.normalize_space(event.get("origin_event_id")): event
        for event in plan.get("events") or []
        if isinstance(event, dict)
        and native.normalize_space(event.get("origin_event_id"))
    }
    for child_event in durable_child_graph.get("events") or []:
        if not isinstance(child_event, dict):
            continue
        plan_event = plan_by_origin_event_id.get(
            native.normalize_space(child_event.get("event_id"))
        )
        if not isinstance(plan_event, dict):
            continue
        for time_field in ("local_time_index", "global_time_index"):
            if time_field in plan_event:
                child_event[time_field] = copy.deepcopy(plan_event[time_field])
    parent_events = copy.deepcopy(parent_graph.get("events") or [])
    durable_child_events = copy.deepcopy(durable_child_graph.get("events") or [])
    parent_graph_sha256 = canonical_json_sha256(parent_graph)
    child_graph_sha256 = canonical_json_sha256(child_graph)
    durable_child_graph_sha256 = canonical_json_sha256(durable_child_graph)
    current_event_plan_sha256 = canonical_json_sha256(plan)
    combined_lineage_basis = {
        "parent_graph_id": parent_id,
        "child_graph_id": native.normalize_space(child_graph.get("graph_id")),
        "parent_graph_sha256": parent_graph_sha256,
        "durable_child_graph_sha256": durable_child_graph_sha256,
        "current_event_plan_sha256": current_event_plan_sha256,
        "parent_event_count": len(parent_events),
        "child_event_count": len(durable_child_events),
    }
    combined_lineage_sha256 = canonical_json_sha256(combined_lineage_basis)
    plan_metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
    combined_graph = {
        "schema_version": "stage3_lineage_graph_v2",
        "graph_id": f"stage3_lineage:{combined_lineage_sha256[:24]}",
        "parent_graph_id": parent_id,
        "child_graph_id": native.normalize_space(child_graph.get("graph_id")),
        "parent_graph_schema_version": native.normalize_space(
            parent_graph.get("schema_version")
        ),
        "parent_event_count": len(parent_events),
        "child_event_count": len(durable_child_events),
        "event_count": len(parent_events) + len(durable_child_events),
        "parent_graph_sha256": parent_graph_sha256,
        "child_graph_sha256": child_graph_sha256,
        "durable_child_graph_sha256": durable_child_graph_sha256,
        "current_event_plan_sha256": current_event_plan_sha256,
        "combined_lineage_sha256": combined_lineage_sha256,
        "parent_max_global_time_index": (
            native.stage3_parent_continuity.parent_max_global_time_index(parent_events)
        ),
        "parent_events": parent_events,
        "child_events": durable_child_events,
        "events": copy.deepcopy(parent_events) + copy.deepcopy(durable_child_events),
        "edges": copy.deepcopy(parent_graph.get("edges") or []),
        "parent_continuity": copy.deepcopy(
            plan_metadata.get("parent_continuity") or {}
        ),
        "parent_event_retrieval": copy.deepcopy(
            plan_metadata.get("parent_event_retrieval") or {}
        ),
    }
    artifact_basis = {
        "seed_sample_id": seed_id,
        "stage64_target_identity": target_identity,
        "parent_graph_sha256": parent_graph_sha256,
        "child_graph_sha256": child_graph_sha256,
        "durable_child_graph_sha256": durable_child_graph_sha256,
        "anchor_ledger": markers,
        # The same semantic delta can be rendered with a corrected canonical
        # dialogue evidence contract.  Bind that plan hash too, so a repaired
        # graph artifact never silently reuses an earlier artifact_id.
        "current_event_plan_sha256": current_event_plan_sha256,
    }
    artifact_id = "stage3_canonical_" + canonical_json_sha256(artifact_basis)[:24]
    return {
        "schema_version": native.FIXED_STAGE3_GRAPH_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "seed_sample_id": seed_id,
        "stage64_target_identity": target_identity,
        "stage64_target": stage64_target,
        "parent_graph": parent_graph,
        "child_graph_delta": copy.deepcopy(child_graph),
        "durable_child_graph_delta": durable_child_graph,
        "current_event_plan": plan,
        "anchor_ledger": {
            "parent_event_evidence_turn_ids": parent_anchor_ledger,
            "child_event_instance_to_marker": dict(sorted(markers.items())),
            "child_marker_to_event_instance": {
                marker: event_id for event_id, marker in sorted(markers.items())
            },
        },
        "combined_lineage_graph": combined_graph,
        "lineage_hashes": {
            "parent_graph_sha256": parent_graph_sha256,
            "child_graph_sha256": child_graph_sha256,
            "durable_child_graph_sha256": durable_child_graph_sha256,
            "current_event_plan_sha256": current_event_plan_sha256,
            "combined_lineage_sha256": combined_lineage_sha256,
        },
        "source_graph_meta": graph_meta,
        "materialization_audit": {
            "pass": True,
            "raw_graph_semantic_issue_count": 0,
            "normalized_plan_semantic_issue_count": 0,
            "parent_graph_id_matches_child": parent_id
            == native.normalize_space(child_graph.get("parent_graph_id")),
            "child_anchor_count": len(markers),
            "parent_anchor_count": len(parent_anchor_ledger),
            "parent_graph_sha256": parent_graph_sha256,
            "child_graph_sha256": child_graph_sha256,
            "durable_child_graph_sha256": durable_child_graph_sha256,
            "current_event_plan_sha256": current_event_plan_sha256,
            "combined_lineage_sha256": combined_lineage_sha256,
            "combined_lineage_graph_sha256": canonical_json_sha256(combined_graph),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-records", required=True, help="Raw graph_conditioned_api_dialogue_raw.jsonl")
    parser.add_argument("--parent-records", required=True, help="Stage2 canonical stage3_parent_v3.jsonl")
    parser.add_argument("--output", required=True, help="Output canonical Stage3 graph artifact JSON")
    parser.add_argument("--record-index", type=int, default=0, help="Zero-based raw JSONL record index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = read_jsonl(args.raw_records)
    if args.record_index < 0 or args.record_index >= len(raw_rows):
        raise ValueError(f"--record-index is outside [0, {len(raw_rows) - 1}]")
    artifact = materialize_artifact(
        raw_record=raw_rows[args.record_index],
        parent_rows=read_jsonl(args.parent_records),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "artifact_id": artifact["artifact_id"],
                "seed_sample_id": artifact["seed_sample_id"],
                "parent_event_count": len(artifact["parent_graph"].get("events") or []),
                "child_event_count": len(artifact["child_graph_delta"].get("events") or []),
                "child_anchor_count": len(
                    artifact["anchor_ledger"]["child_event_instance_to_marker"]
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
