"""Offline review preparation and observed-cost reporting; never call a model.

The report intentionally cannot certify semantic quality. Authored review rows
are inventory, not automatically validated judgments or deduplicated yield.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_r02_ordering_temporal_contract_v1.reporting import _quality_inventory, _review_schema
from .constants import ARMS, DEFAULT_RUN, ROOTS, SCHEDULE, SOURCE_RUN, VERSION

REVIEWER = "Codex offline diagnostic"
UNKNOWN = "UNKNOWN"
ERROR_CATEGORIES = (
    "PLAN_AS_COMPLETED", "ACTOR_OR_EVENT_ATTRIBUTION", "EVENT_NOT_SUPPORTED_BY_QUOTE",
    "TEMPORAL_EDGE_REVERSED", "BASIS_TEXT_RIGHT_EDGE_REVERSED", "BASIS_AND_EDGE_WRONG",
    "BASIS_EDGE_CONTRADICTION_OTHER", "MENTION_ORDER_AS_EVENT_TIME", "PROCESS_ASSUMPTION",
    "UNSUPPORTED_UNIQUE_ORDER",
)


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def _rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _stable_json(path, value):
    """Do not quietly change a policy under already authored reviews."""
    if path.is_file() and _read(path) != value:
        raise ValueError("existing_review_asset_differs: " + str(path))
    if not path.is_file():
        atomic_json(path, value)


def prepare_review_assets(run=DEFAULT_RUN, source_run=SOURCE_RUN):
    """Reuse the existing rubric; create empty review destinations, not audits."""
    run, source_run = Path(run), Path(source_run)
    source_path = source_run / "review_policy.json"
    policy = copy.deepcopy(_read(source_path))
    if not isinstance(policy, dict) or not policy.get("dimensions"):
        raise ValueError("frozen_source_review_policy_required")
    policy.update(version=VERSION + "_review_v1", arms=list(ARMS), reviewer=REVIEWER,
                  source_policy_sha256=sha256_file(source_path),
                  independent_blind_review=False, human_ground_truth_certification=False,
                  content_status="NOT_EVALUATED", quality_sequence_status="NOT_RUN",
                  review_api_calls=0, production_admission_changed=False,
                  method_comparison="Same mixed TEMPORAL generator and frozen primary messages; compare Writer model/service configurations. Neither model supplies reference truth.")
    policy["ordering_review"]["edges"] = (
        "Both arms use TEMPORAL. Assess every event and temporal_relations edge against exact frozen transcript quotes, "
        "including whether temporal_basis prose agrees with before/after. Separate correct prose with reversed fields from prose and fields both wrong.")
    policy["ordering_review"]["error_categories"] = list(ERROR_CATEGORIES)
    policy["ordering_review"]["diagnostic_only"] = "No added semantic generation gate; unique DAG order alone never establishes content PASS."
    policy["denominators"]["paired_difference"] = "STRONG_TEMPORAL minus QWEN14B_TEMPORAL, by root; two roots receive equal weight. One sample per intent is screening evidence only."
    policy["future_denominators"]["difference"] = "STRONG_TEMPORAL - QWEN14B_TEMPORAL"
    policy["dedup_rubric"]["cross_arm"] = "Describe cross-arm overlap; do not remove one arm's questions as duplicates of the other arm."
    policy["interpretation"] = {
        "better": "Local screening benefit for these two known roots and model/service settings only; account for errors, yield, coverage and complete cost together.",
        "similar_or_mixed": "Insufficient evidence; do not resample until a preferred result appears.",
        "worse": "Does not establish that a model size is intrinsically incapable of ordering.",
        "excluded_claims": ["all ordering reliable", "universal model capability", "training benefit", "independent human certification", "model output as reference truth"],
    }
    policy["cost_policy"]["reasoning"] = "Record returned reasoning tokens separately. Do not add them again to total/completion tokens; service accounting may already include them. Missing usage or verified money is UNKNOWN."
    policy["quality_sequence"][0] = "Sort opaque model-free views; hide model, arm, request order, program status and source intent diagnosis; expose complete frozen transcript and all candidate event/edge fields. This is identity hiding, not independent blind certification."
    policy["post_quality"]["new_error"] = "After quality freeze, separately record inherited source error, corrected error, bypassed intent, independent new error or UNDETERMINED; no required manually chosen target."
    policy["completion"]["requirements"] = [
        "All 12 planned primary slots represented; every attempted HTTP call and failure retained.",
        "Every raw primary and repair candidate reviewed, including replaced, rejected and malformed rows; schema/quote validation and quality freeze completed.",
        "Post-quality intent alignment, every known source FAIL dimension and independent new errors reviewed.",
        "Explicit within-root/arm partitions for primary, repair and final quality-PASS SCMS and ordering; missing clusters are not singletons.",
        "Every view compared only with its finite frozen history_only.json pool; cross-arm and discarded-to-retained coverage explicit.",
        "Shared fact/selection coverage for both roots and all three layers, including empty populations.",
        "Cost counts all attempted primary/repair, failures and replaced batches; unavailable usage or money stays UNKNOWN.",
    ]
    schema = _review_schema(policy["dimensions"])
    schema["properties"]["error_categories"] = {"type": "array", "uniqueItems": True, "items": {"enum": list(ERROR_CATEGORIES)}}
    schema["allOf"][0]["then"]["required"].append("error_categories")
    edge = schema["properties"]["edge_checks"]["items"]
    edge["properties"]["claim_origin"] = {"const": "WRITER_TEMPORAL_RELATION"}
    edge["required"].append("basis_direction_case")
    edge["properties"]["basis_direction_case"] = {"enum": ["CONSISTENT_SUPPORTED", "BASIS_TEXT_RIGHT_EDGE_REVERSED", "BASIS_AND_EDGE_WRONG", "OTHER_CONTRADICTION", "UNDETERMINED"]}
    _stable_json(run / "review_policy.json", policy)
    _stable_json(run / "policy_snapshot/review_schemas/quality_review.schema.json", schema)
    _stable_json(run / "policy_snapshot/review_templates.json", {
        "status": "EMPTY_NOT_RUN", "reviewer": REVIEWER, "review_api_calls": 0,
        "quality": "reports/quality_review.jsonl", "post_quality": "reports/post_quality_review.jsonl",
        "dedup": "reports/dedup_review.jsonl", "history": "reports/history_overlap_review.jsonl",
        "shared_facts": "reports/shared_fact_coverage.jsonl", "pairs": "reports/cross_arm_and_discarded_pairs.jsonl",
        "freeze": "audit/quality_review_freeze.json", "validation": "audit/offline_review_validation.json",
        "requirements": policy["completion"]["requirements"],
        "status_rule": "Authored rows are not automatically validated by inventory reporting. Semantic review, exact-quote validation, freeze and post-quality checks remain explicit offline work.",
    })
    for name in ("quality_review", "post_quality_review", "dedup_review", "history_overlap_review", "shared_fact_coverage", "cross_arm_and_discarded_pairs"):
        path = run / "reports" / (name + ".jsonl")
        if not path.exists():
            atomic_text(path, "")
    return {"status": "PREPARED_NOT_RUN", "external_calls": 0, "authored_quality_rows": 0,
            "source_policy_sha256": sha256_file(source_path), "policy": str(run / "review_policy.json")}


def _identity(row):
    return tuple(row.get(key) for key in ("arm_id", "root_id", "slot_id", "stage"))


def _number(value):
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and value >= 0 else None


def _sum_complete(rows, field):
    values = [_number(row.get(field)) for row in rows]
    known = sum(value for value in values if value is not None)
    missing = sum(value is None for value in values)
    return {"value": UNKNOWN if missing else known, "known_subtotal": known, "unknown_calls": missing}


def call_accounting(run):
    """Join request and outcome ledgers once, retaining outcome-only failures."""
    run = Path(run)
    calls = []
    for ledger in sorted((run / "arms").glob("*/qa/shards/shard-00/roots/*/ledger")):
        rel = ledger.relative_to(run).parts
        arm, root = rel[1], rel[6]
        requests = {p.stem: _read(p) for p in (ledger / "requests").glob("*.json")}
        outcomes = {p.stem: _read(p) for p in (ledger / "transport_outcomes").glob("*.json")}
        for key in sorted(requests.keys() | outcomes.keys()):
            request, outcome = requests.get(key, {}), outcomes.get(key, {})
            response = request.get("response_metadata") or {}
            meta = {**response, **outcome}
            reqmeta = request.get("request_metadata") or {}
            usage = meta.get("usage") or response.get("usage") or {}
            details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
            def token(field, *aliases):
                if _number(meta.get(field)) is not None:
                    return meta[field]
                for alias in aliases:
                    if _number(usage.get(alias)) is not None:
                        return usage[alias]
                return UNKNOWN
            attempted = meta.get("http_attempt_count", meta.get("transport_attempt_count", UNKNOWN))
            cost = meta.get("verified_cost") or {}
            money_verified = (_number(cost.get("amount")) is not None and bool(cost.get("currency"))
                              and bool(cost.get("evidence_sha256")))
            calls.append({
                "arm_id": arm, "root_id": root, "request_id": request.get("request_id", key),
                "global_request_id": reqmeta.get("global_request_id", outcome.get("global_request_id")),
                "slot_id": reqmeta.get("slot_id", outcome.get("slot_id", UNKNOWN)),
                "stage": reqmeta.get("stage", outcome.get("stage", "repair" if "-repair-" in key else "primary")),
                "ledger_status": request.get("status", "OUTCOME_WITHOUT_RESERVATION"),
                "service_completion_status": meta.get("service_completion_status", meta.get("completion_status", UNKNOWN)),
                "http_attempts": attempted,
                "transport_retries": meta.get("transport_retry_count", UNKNOWN),
                "rate_limit_retries": meta.get("rate_limit_retry_count", UNKNOWN),
                "timeout": meta.get("timeout", UNKNOWN), "http_status": meta.get("http_status", UNKNOWN),
                "actual_model": meta.get("actual_model") or meta.get("returned_model") or UNKNOWN,
                "finish_reason": meta.get("finish_reason") or UNKNOWN,
                "usage": usage if isinstance(usage, dict) and usage else UNKNOWN,
                "input_tokens": token("input_tokens", "prompt_tokens", "input_tokens"),
                "output_tokens": token("output_tokens", "completion_tokens", "output_tokens"),
                "total_tokens": token("total_tokens", "total_tokens"),
                "reasoning_tokens": token("reasoning_tokens", "reasoning_tokens") if _number(details.get("reasoning_tokens")) is None else details["reasoning_tokens"],
                "verified_money": cost["amount"] if money_verified else UNKNOWN,
                "currency": cost.get("currency") if money_verified else UNKNOWN,
                "pricing_or_bill_evidence_sha256": cost.get("evidence_sha256") if money_verified else None,
            })
    def aggregate(rows):
        currencies = {row["currency"] for row in rows}
        money = _sum_complete(rows, "verified_money")
        if len(currencies) > 1:
            money = {"value": UNKNOWN, "known_subtotal": None, "unknown_calls": sum(row["verified_money"] == UNKNOWN for row in rows), "reason": "MIXED_CURRENCIES"}
        return {"logical_calls": len(rows), "status": "OBSERVED" if rows else "NOT_RUN",
                **{field: _sum_complete(rows, field) for field in ("http_attempts", "transport_retries", "rate_limit_retries", "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")},
                "verified_money": money, "currency": next(iter(currencies)) if len(currencies) == 1 else UNKNOWN,
                "money_per_effective_ordering": None, "tokens_per_effective_ordering": None,
                "effective_cost_status": "PENDING_VALIDATED_QUALITY_AND_DEDUP" if rows else "NOT_RUN"}
    groups = [{"arm_id": arm, "root_id": root, "stage": stage,
               **aggregate([row for row in calls if row["arm_id"] == arm and row["root_id"] == root and (stage == "all" or row["stage"] == stage)])}
              for root in ROOTS for arm in ARMS for stage in ("primary", "repair", "all")]
    return {"calls": calls, "groups": groups, "total": aggregate(calls),
            "note": "Includes every ledger request, replaced primary and repair without yield-based filtering. Unknown usage stays UNKNOWN; known subtotals are not full cost. Reasoning tokens are not added again to total tokens."}


def _inventory(run, stages, finals):
    views, mapping, unreadable = _quality_inventory(run, stages)
    stage_index = {_identity(row): row for row in stages}
    for view in views:
        transcript = run / view["complete_transcript_path"]
        view["complete_transcript_sha256"] = sha256_file(transcript) if transcript.is_file() else UNKNOWN
        view["candidate_sha256"] = sha256_bytes(canonical_json(view["candidate"]).encode())
    for item in mapping:
        stage = stage_index[_identity(item)]
        validation = stage.get("validation") or {}
        normalized = next((row for row in validation.get("candidates", [])
                           if row.get("source_row_index") == item["raw_row_index"]), None)
        if normalized is None:
            # Older shared normalizers use positional identity, never content matching.
            rows = validation.get("candidates", [])
            if len(rows) >= item["raw_row_index"] and all(not row.get("source_row_index") for row in rows):
                normalized = rows[item["raw_row_index"] - 1]
        final = next((row for row in finals if _identity(row) == _identity(item)
                      and (row.get("source_row_index") == item["raw_row_index"]
                           or (normalized and row.get("source_candidate_id") == normalized.get("candidate_id"))
                           or (normalized and row.get("candidate_id") == normalized.get("candidate_id")))), None)
        item.update(program_schema_valid=normalized.get("schema_valid") if normalized else None,
                    program_schema_errors=normalized.get("schema_errors", []) if normalized else [],
                    program_derived_gold=normalized.get("gold") if normalized else None,
                    minimum_batch_accepted=validation.get("minimum_acceptance", {}).get("accepted"),
                    replaced_by_repair=stage.get("replaced_by_repair", False),
                    final_validator_accepted=final is not None and final.get("schema_valid") is True,
                    question_type=item["raw_row"].get("question_type", "unreadable"))
    # Opaque ids alone do not hide sequential arm blocks; sort before handing off.
    views.sort(key=lambda row: row["view_id"])
    _jsonl(run / "reports/quality_review_views.jsonl", views)
    _jsonl(run / "audit/quality_view_mapping.jsonl", mapping)
    (run / "audit/quality_view_mapping.jsonl").chmod(0o600)
    return views, mapping, unreadable


def _technical_failure(call):
    return call.get("service_completion_status") not in (None, UNKNOWN, "COMPLETED")


def _slot_outcomes(stages, mapping, calls=()):
    outcomes = []
    for slot in SCHEDULE:
        identity = tuple(slot[key] for key in ("arm_id", "root_id", "slot_id"))
        slot_stages = [row for row in stages if _identity(row)[:3] == identity]
        primary = next((row for row in slot_stages if row.get("stage") == "primary"), None)
        repair = next((row for row in slot_stages if row.get("stage") == "repair"), None)
        selected = repair or primary
        acceptance = ((selected or {}).get("validation") or {}).get("minimum_acceptance", {}).get("accepted")
        slot_items = [row for row in mapping if _identity(row)[:3] == identity]
        slot_calls = [row for row in calls if _identity(row)[:3] == identity]
        failures = [row for row in slot_calls if _technical_failure(row)]
        attempts = _sum_complete(slot_calls, "http_attempts")
        status = ("TECHNICAL_FAILURE" if failures else "RECORDED" if selected else
                  "ATTEMPTED_WITHOUT_STAGE_RECORD" if slot_calls else "NOT_RUN")
        outcomes.append({**{key: slot[key] for key in ("arm_id", "root_id", "slot_id", "order")},
                         "logical_calls": len(slot_calls), "http_attempts": attempts,
                         "technical_failure_calls": len(failures),
                         "service_completion_statuses": [row["service_completion_status"] for row in slot_calls],
                         "attempted": bool(slot_calls),
                         "primary_recorded": primary is not None, "repair_recorded": repair is not None,
                         "primary_replaced": bool((primary or {}).get("replaced_by_repair")),
                         "primary_parse_error": (primary or {}).get("parse_error"),
                         "repair_parse_error": (repair or {}).get("parse_error"),
                         "selected_stage": selected.get("stage") if selected else None,
                         "selected_batch_minimum_accepted": acceptance,
                         "whole_batch_rejected": acceptance is False if acceptance is not None else None,
                         "raw_primary_candidates": sum(row["stage"] == "primary" for row in slot_items),
                         "raw_repair_candidates": sum(row["stage"] == "repair" for row in slot_items),
                         "final_program_accepted": sum(row["final_validator_accepted"] for row in slot_items),
                         "lost_primary_scms_schema_accepted": sum(row["stage"] == "primary" and row["replaced_by_repair"]
                             and row["program_schema_valid"] is True and row["question_type"] in ("single_choice", "multi_select") for row in slot_items),
                         "lost_primary_scms_content_pass": UNKNOWN,
                         "status": status})
    return outcomes


def report(run=DEFAULT_RUN):
    """Produce inventories and cost tables, with semantic results explicitly pending."""
    run = Path(run)
    stages = _rows(run / "reports/runtime_stage_records.jsonl")
    finals = _rows(run / "reports/runtime_final_candidates.jsonl")
    runtime = _read(run / "reports/runtime_summary.json", {})
    views, mapping, unreadable = _inventory(run, stages, finals)
    costs = call_accounting(run)
    slot_outcomes = _slot_outcomes(stages, mapping, costs["calls"])
    stop_record = _read(run / "audit/execution_stop.json", {})
    authored = _rows(run / "reports/quality_review.jsonl")
    authored_ids = Counter(row.get("view_id") for row in authored)
    view_ids = {row["view_id"] for row in views}
    invalid_inventory = {"duplicate_authored_ids": sorted(str(key) for key, count in authored_ids.items() if count > 1),
                         "unknown_authored_ids": sorted(str(key) for key in authored_ids.keys() - view_ids)}
    has_runtime = bool(stages or runtime or costs["calls"])
    preflight = _read(run / "audit/preflight_report.json", {})
    preparation = _read(run / "preparation_manifest.json", {})
    status = "PENDING_OFFLINE_DIAGNOSTICS" if has_runtime else preparation.get("status", "PREPARED_NOT_RUN")
    if any(invalid_inventory.values()):
        status = "BLOCKED_REVIEW_IDENTITY"
    groups = []
    for root in ROOTS:
        for arm in ARMS:
            for layer in ("primary", "repair", "final"):
                population = [row for row in mapping if row["root_id"] == root and row["arm_id"] == arm
                              and (row["final_validator_accepted"] if layer == "final" else row["stage"] == layer)]
                relevant_stages = [row for row in stages if row.get("root_id") == root and row.get("arm_id") == arm and (layer == "final" or row.get("stage") == layer)]
                relevant_calls = [row for row in costs["calls"] if row["root_id"] == root and row["arm_id"] == arm and (layer == "final" or row["stage"] == layer)]
                failed_identities = {_identity(row) for row in relevant_calls if _technical_failure(row)}
                for qtype in ("ordering", "single_choice", "multi_select", "SCMS"):
                    items = [row for row in population if (row["question_type"] in ("single_choice", "multi_select") if qtype == "SCMS" else row["question_type"] == qtype)]
                    groups.append({"root_id": root, "arm_id": arm, "layer": layer, "question_type": qtype,
                                   "scheduled_primary_slots": 3, "raw_candidates": len(items) if layer != "final" else None,
                                   "program_schema_accepted": sum(row["program_schema_valid"] is True for row in items),
                                   "program_final_accepted": sum(row["final_validator_accepted"] for row in items),
                                   "replaced_primary_candidates": sum(row["stage"] == "primary" and row["replaced_by_repair"] for row in items),
                                   "parse_failed_requests": sum(bool(row.get("parse_error")) and _identity(row) not in failed_identities for row in relevant_stages),
                                   "technical_failed_requests": sum(_technical_failure(row) for row in relevant_calls),
                                   "batch_minimum_rejected_requests": sum((row.get("validation") or {}).get("minimum_acceptance", {}).get("accepted") is False for row in relevant_stages),
                                   "authored_review_rows": sum(row["view_id"] in authored_ids for row in items),
                                   "content_PASS": UNKNOWN, "content_FAIL": UNKNOWN, "content_UNDETERMINED": UNKNOWN,
                                   "deduplicated_supported_count": UNKNOWN,
                                   "errors": {category: UNKNOWN for category in ERROR_CATEGORIES},
                                   "quality_status": "PENDING_VALIDATED_REVIEW" if has_runtime else "NOT_EVALUATED"})
    history = []
    for root in ROOTS:
        path = run / "inputs/shared" / root / "history_only.json"
        data = _read(path)
        records = data if isinstance(data, list) else data.get("records", []) if isinstance(data, dict) else None
        history.append({"root_id": root, "path": str(path.relative_to(run)),
                        "sha256": sha256_file(path) if path.is_file() else UNKNOWN,
                        "records": len(records) if isinstance(records, list) else UNKNOWN,
                        "scope": "Only this frozen finite pool; never all historical QA."})
    summary = {
        "version": VERSION, "status": status,
        "engineering": "PASS" if preflight.get("status") == "PASS" else "NOT_VERIFIED",
        "content_status": "PENDING_VALIDATED_REVIEW" if has_runtime else "NOT_EVALUATED",
        "external_calls_by_report": 0, "review_api_calls": 0,
        "runtime_external_calls": runtime.get("external_calls", UNKNOWN if has_runtime else 0),
        "runtime_status": runtime.get("status", "NOT_RUN"),
        "execution_stop_reason": stop_record.get("reason"),
        "planned_primary_calls": 12, "max_repair_calls": 12, "max_logical_calls": 24,
        "raw_quality_views": len(views), "nonreviewable_requests": unreadable,
        "recorded_primary_slots": sum(row["primary_recorded"] for row in slot_outcomes),
        "recorded_repairs": sum(row["repair_recorded"] for row in slot_outcomes),
        "attempted_primary_slots": sum(any(call["stage"] == "primary" and _identity(call)[:3] == _identity(row)[:3] for call in costs["calls"]) for row in slot_outcomes),
        "unattempted_primary_slots": sum(not any(call["stage"] == "primary" and _identity(call)[:3] == _identity(row)[:3] for call in costs["calls"]) for row in slot_outcomes),
        "technical_failure_calls": sum(_technical_failure(row) for row in costs["calls"]),
        "whole_batch_rejected_slots": sum(row["whole_batch_rejected"] is True for row in slot_outcomes),
        "authored_quality_rows": len(authored), "pending_quality_view_ids": sorted(view_ids - authored_ids.keys()),
        "authored_rows_pending_validation": len(authored), "invalid_review_inventory": invalid_inventory,
        "groups": groups, "costs": costs["total"], "frozen_history_pools": history,
        "effective_ordering_cost": None,
        "cost_status": "PENDING_VALIDATED_QUALITY_AND_DEDUP" if has_runtime else "NOT_RUN",
        "limitations": ["Identity hiding is not independent blind or human certification; reviewer is Codex offline diagnostic.",
                       "A unique program Gold does not establish supported events, temporal edges or semantic quality.",
                       "Two roots and single sampling are known-case screening; mixed/close outcomes are insufficient evidence.",
                       "No strong-model reference truth, Solver, training, Memory API or review API is used.",
                       "This command inventories authored reviews; exact-quote validation, quality freeze and post-quality audits are separate required offline work.",
                       "Zero supported ordering yield has undefined effective cost; no finite zero-yield cost is reported."],
    }
    requirements = _read(run / "policy_snapshot/review_templates.json", {})
    atomic_json(run / "reports/diagnostic_requirements.json", {"requirements": requirements.get("requirements", []),
                "quality_before_post_quality": True, "reviewer": REVIEWER,
                "history_pools": history, "quality_freeze_required": True,
                "unknown_semantic_counts_must_not_be_zero": True})
    # Identity-bearing inventories belong in audit, outside the blinded handoff.
    atomic_json(run / "audit/diagnostic_pending_inventory.json", {"slots": SCHEDULE, "views": mapping, "nonreviewable_requests": unreadable})
    atomic_json(run / "reports/complete_call_costs.json", costs)
    atomic_json(run / "reports/slot_outcomes_and_collateral.json", slot_outcomes)
    atomic_json(run / "reports/quality_comparison_by_root_arm_layer_type.json", groups)
    atomic_json(run / "reports/diagnostic_summary.json", summary)
    lines = ["# Writer model comparison", "", f"{status}; content: {summary['content_status']}.", "",
             "Both arms use the same frozen primary TEMPORAL messages and formal mixed generator. No API call is made by this report command.", "",
             f"Raw candidate views: {len(views)}; unreadable requests: {len(unreadable)}; authored but not certified quality rows: {len(authored)}.", "",
             f"Runtime: {summary['runtime_status']}; attempted primary slots: {summary['attempted_primary_slots']}; unattempted primary slots: {summary['unattempted_primary_slots']}; technical failures: {summary['technical_failure_calls']}; stop reason: {summary['execution_stop_reason'] or 'NONE'}.", "",
             "| Root | Arm | Layer | Type | Raw | Program schema accepted | Program final accepted | Content / dedup |",
             "|---|---|---|---|---:|---:|---:|---|"]
    lines.extend(f"| {g['root_id']} | {g['arm_id']} | {g['layer']} | {g['question_type']} | {g['raw_candidates'] if g['raw_candidates'] is not None else 'N/A'} | {g['program_schema_accepted']} | {g['program_final_accepted']} | UNKNOWN / UNKNOWN |" for g in groups)
    lines += ["", "All reserved/attempted primary and repair requests, including replacements and failures, are included in complete_call_costs.json. Known token subtotals are not complete cost when usage is missing. Monetary cost is UNKNOWN without verified billing/pricing evidence. Effective cost is unset pending validated quality and explicit deduplication.", "", "Review completion requires:", ""]
    lines.extend("- " + value for value in requirements.get("requirements", []))
    lines += ["", "The review is Codex offline diagnostic with model identities hidden where feasible. It is not independent human certification. No model's answer is treated as truth.", ""]
    atomic_text(run / "reports/diagnostic_status.md", "\n".join(lines))
    return summary
