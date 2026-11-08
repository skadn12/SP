"""Offline diagnostic policy, pending inventories, and historical fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .constants import ARMS, DEFAULT_RUN, ROOTS, SCHEDULE, SOURCE_WRITER, VERSION


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def _rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _review_schema(dimensions):
    text = {"type": "string", "minLength": 1}
    quote = {"type": "object", "required": ["turn_ref", "quote"], "additionalProperties": False,
             "properties": {"turn_ref": text, "quote": text}}
    judgment = {"type": "object", "required": ["judgment", "reason", "quotes"], "additionalProperties": False,
                "properties": {"judgment": {"enum": ["PASS", "FAIL", "UNDETERMINED", "NOT_APPLICABLE", "NOT_EVALUATED_STRUCTURE"]}, "reason": text, "quotes": {"type": "array", "items": quote}}}
    dimprops = {key: judgment for key in dimensions}
    event = {"type": "object", "required": ["item_id", "event_support", "actor_accuracy", "state_accuracy"], "additionalProperties": False,
             "properties": {"item_id": text, "event_support": judgment, "actor_accuracy": judgment, "state_accuracy": judgment}}
    edge = {"type": "object", "required": ["before", "after", "claim_origin", "support", "temporal_basis_assessment"], "additionalProperties": False,
            "properties": {"before": text, "after": text, "claim_origin": {"enum": ["WRITER_TEMPORAL_RELATION", "BASE_GOLD_IMPLIED_RELATION"]}, "support": judgment, "temporal_basis_assessment": judgment}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "required": ["view_id", "question_type", "reviewer", "judgment", "reason", "dimensions", "option_checks"],
            "properties": {"view_id": text, "question_type": {"enum": ["single_choice", "multi_select", "ordering", "unreadable"]}, "reviewer": {"const": "Codex offline diagnostic"}, "judgment": {"enum": ["PASS", "FAIL", "UNDETERMINED", "NOT_EVALUATED_STRUCTURE"]}, "reason": text, "dimensions": {"type": "object", "required": dimensions, "additionalProperties": False, "properties": dimprops}, "option_checks": {"type": "array", "items": {"type": "object"}}, "event_checks": {"type": "array", "items": event}, "edge_checks": {"type": "array", "items": edge}, "time_scope_uniqueness": judgment},
            "allOf": [{"if": {"properties": {"question_type": {"const": "ordering"}}}, "then": {"required": ["event_checks", "edge_checks", "time_scope_uniqueness"]}}]}


def prepare_review_assets(run=DEFAULT_RUN):
    """Create policy/templates only; never author a quality result."""
    run = Path(run)
    policy = copy.deepcopy(_read(SOURCE_WRITER / "review_policy.json", {}))
    policy["version"] = VERSION + "_review_v1"
    policy["arms"] = list(ARMS)
    policy["method_comparison"] = "Combined ordering generation contract plus deterministic constraint derivation; no isolated graph/prompt causal claim."
    policy["ordering_review"] = {
        "events": "Every item: exact quotes, event support, actor and event_state accuracy; distinguish speaker from actor, recollection, intention, decision, start, ongoing and completion.",
        "edges": "Every new temporal_relations edge and its temporal_basis gets exact quote support. BASE receives equivalent review of every Gold-implied adjacent relation and any needed relation; metadata absence is not a BASE content failure.",
        "time_scope_uniqueness": "Assess answer uniqueness under the stated event/state/time scope; separate unsupported edges, missing Writer constraints and true textual ambiguity. Overlapping continuous events are not automatically totally ordered.",
        "semantic_separation": "Program PASS means unique order under submitted constraints only. semantic_support_status stays NOT_ASSESSED; review never changes generation admission or repair.",
        "failure_policy": "Review all readable rows, including graph rejects, replaced primary, repair rejects and final discarded batches. Keep unreadable structure explicit.",
    }
    policy["quality_sequence"] = [
        "Expose candidate-owned fields and complete frozen transcript; retain event/edge fields for their own support review and hide arm, intent diagnosis and program status from quality judgment.",
        "Review every readable primary/repair row using inherited SC/MS dimensions plus event, actor, state, edge support, temporal basis and scoped uniqueness for ordering.",
        "Freeze quality rows, exact quotations, view/transcript hashes and policy before post-quality alignment, error propagation, coverage or dedup.",
        "After freeze, compare both arms by root, stage and type, including failed slots, repair collateral loss, shared facts, selection and all token costs.",
    ]
    policy["quality_sequence_status"] = "NOT_RUN"
    policy["review_api_calls"] = 0
    policy["content_status"] = "NOT_EVALUATED"
    policy["future_denominators"] = {"root": "two roots equal weight", "scheduled_slots_per_root_arm": 3, "primary": 12, "repair_max": 12, "total_max": 24, "difference": "TEMPORAL_CONTRACT_ORDERING - BASE_ORDERING", "type_scopes": ["single_choice", "multi_select", "SCMS", "ordering"]}
    policy["cost_policy"]["effective_cost"] = "All actual primary/repair tokens divided by final quality PASS and explicit deduplicated clusters; missing usage is incomplete and no zero-yield efficiency claim is made."
    atomic_json(run / "review_policy.json", policy)
    atomic_json(run / "policy_snapshot/review_schemas/quality_review.schema.json", _review_schema(policy["dimensions"]))
    atomic_json(run / "policy_snapshot/review_templates.json", {"status": "EMPTY_NOT_RUN", "authored_quality_rows": 0, "quality": "reports/quality_review.jsonl", "post_quality": "reports/post_quality_review.jsonl", "dedup": "reports/dedup_review.jsonl", "history": "reports/history_overlap_review.jsonl", "shared_facts": "reports/shared_fact_coverage.jsonl", "freeze": "audit/quality_review_freeze.json", "required_scopes": ["primary", "repair", "final", "single_choice", "multi_select", "SCMS", "ordering"]})
    for name in ("quality_review.jsonl", "post_quality_review.jsonl", "dedup_review.jsonl", "history_overlap_review.jsonl", "shared_fact_coverage.jsonl"):
        path = run / "reports" / name
        if not path.exists():
            atomic_text(path, "")
    return {"status": "PREPARED_NOT_RUN", "external_calls": 0, "authored_quality_rows": 0, "policy": str(run / "review_policy.json"), "source_policy_sha256": sha256_file(SOURCE_WRITER / "review_policy.json")}


def _quality_inventory(run, stages):
    views, mapping, unreadable = [], [], []
    for stage in stages:
        raw = stage.get("raw_parsed")
        rows = raw.get("candidates") if isinstance(raw, dict) else None
        identity = {key: stage.get(key) for key in ("arm_id", "root_id", "slot_id", "stage", "request_id", "global_request_id")}
        if not isinstance(rows, list):
            unreadable.append({**identity, "status": "NOT_EVALUATED_STRUCTURE", "reason": stage.get("parse_error") or "candidate_list_missing"})
            continue
        for index, row in enumerate(rows, 1):
            row = row if isinstance(row, dict) else {}
            identity_row = {**identity, "raw_row_index": index}
            view_id = "Q-" + sha256_bytes(canonical_json(identity_row).encode())[:20]
            fields = ("question_type", "question", "options", "correct_option", "gold", "items", "evidence_refs", "evidence_groups", "discriminative_variable", "event_nodes", "temporal_relations")
            content = {key: row[key] for key in fields if key in row}
            views.append({"view_id": view_id, "candidate": content, "root_id": stage.get("root_id"), "complete_transcript_path": f"inputs/shared/{stage.get('root_id')}/expanded_dialogue_clean.json", "content_reviewable": bool(row.get("question")), "reviewer": "Codex offline diagnostic"})
            mapping.append({**identity_row, "view_id": view_id, "program_schema_valid": None, "raw_row": row})
    _jsonl(run / "reports/quality_review_views.jsonl", views)
    _jsonl(run / "audit/quality_view_mapping.jsonl", mapping)
    _jsonl(run / "reports/nonreviewable_requests.jsonl", unreadable)
    return views, mapping, unreadable


def report(run=DEFAULT_RUN):
    run = Path(run)
    preflight = _read(run / "audit/preflight_report.json", {"status": "NOT_PREPARED"})
    runtime = _read(run / "reports/runtime_summary.json")
    stages = _rows(run / "reports/runtime_stage_records.jsonl")
    events = _rows(run / "audit/call_events.jsonl")
    reservations = list((run / "arms").glob("*/qa/shards/shard-00/roots/*/ledger/requests/*.json"))
    has_runtime = bool(runtime or stages or events or reservations)
    views, mapping, unreadable = _quality_inventory(run, stages)
    quality_rows = _rows(run / "reports/quality_review.jsonl")
    reviewed = {row.get("view_id") for row in quality_rows}
    pending = [view["view_id"] for view in views if view["view_id"] not in reviewed]
    guardrails = _rows(run / "audit/guardrail_events.jsonl")
    historical = _read(run / "audit/historical_ordering_fixtures.json")
    status = "BLOCKED" if guardrails else "PENDING_OFFLINE_DIAGNOSTICS" if has_runtime else "PREPARED_NOT_RUN" if preflight.get("status") == "PASS" else "NOT_PREPARED"
    data = {"version": VERSION, "status": status, "engineering": "PASS" if preflight.get("status") == "PASS" else "NOT_VERIFIED", "content_status": "PENDING_REVIEW" if has_runtime else "NOT_EVALUATED", "actual_external_calls": (runtime or {}).get("external_calls", 0) if has_runtime else 0, "request_reservations": len(reservations), "quality_views": len(views), "quality_review_rows_present": len(quality_rows), "pending_quality_views": pending, "nonreviewable_requests": unreadable, "historical_fixture_status": historical.get("status", "NOT_RUN") if historical else "NOT_RUN", "historical_fixture_cases": len(historical.get("cases", [])) if historical else 0, "future_budget": {"primary": 12, "repair_max": 12, "total_max": 24}, "ordering_guarantee": "Unique topological sequence under proposed constraints; semantic support not assessed.", "remaining": ["Validate every review against schema and exact quotes; freeze quality before post-quality work.", "Complete both-arm root-paired intent, error, dedup, history, coverage and complete token-cost tables."] if has_runtime else [], "limits": ["Fixture success is not model quality.", "No Solver difficulty or training benefit is measured.", "Future repair capacity is checked dynamically before send."]}
    atomic_json(run / "reports/diagnostic_pending_inventory.json", {"status": status, "slots": SCHEDULE, "views": mapping, "unreadable": unreadable, "pending_quality_view_ids": pending})
    if has_runtime:
        atomic_json(run / "reports/diagnostic_summary.json", data)
        atomic_text(run / "reports/diagnostic_status.md", f"# Diagnostic Status\n\n{status}. No service call was made by this report command. Readable views: {len(views)}; unreadable requests: {len(unreadable)}; pending quality rows: {len(pending)}. This is not a completed comparison.\n")
    else:
        atomic_json(run / "reports/preparation_summary.json", data)
        atomic_text(run / "reports/preparation_report.md", f"# Ordering Temporal Contract v1\n\nENGINEERING={data['engineering']} / {status}\n\nGuided and Free primary/repair use shared Writer/normalizer/finalizer paths. Four, five and six event nodes are validated offline; both future arms use n=4. Program Gold guarantees only a unique topological sequence under submitted constraints; semantic support is NOT_ASSESSED. Graph failures set row schema invalid and enter original minima/one whole-batch repair. Free keeps request target 4/2/2 while repair minima remain 2/1/1.\n\nHistorical ordering fixtures: {data['historical_fixture_status']} ({data['historical_fixture_cases']} anchored cases). They preserve frozen transcript references and are diagnostic examples only; they do not represent model quality or alter inputs.\n\nExperiment service calls: 0. Future budget: 12 primary plus at most 12 replacement repairs, 24 logical calls maximum. Content: NOT_EVALUATED.\n")
    return data
