"""Versioned offline role adapter with an explicit absence contract."""

from __future__ import annotations

from . import HISTORY_ONLY, NO_FEEDBACK, VERSION
from .sources import canonical, value_sha
from ..h4_round1_r03_feedback_view_v2.compiler import (
    _performance, expansion_view_v3, qa_composition_view, qa_targeted_view,
)
from ..h4_round1_r03_feedback_view_v2.templates import (
    EXPANSION_PLANNER_TEMPLATE_V4, QA_TEMPLATE_V3, render,
)
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.prompts import (
    dialogue_text, multi_pass_prompt, targeted_prompt,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import build_planner_prompt


def backend_record(entry, cluster, source_manifest_sha):
    qa = entry["candidate"]
    if (qa["candidate_id"] != entry["candidate_id"] or cluster["root_id"] != qa["root_id"]
            or cluster["canonical_representative_candidate_id"] != entry["candidate_id"]):
        raise ValueError("backend_observation_candidate_identity_mismatch")
    metrics = cluster.get("metrics") or {}
    confusion = {"basis": "canonical_option_or_item_ids_only"}
    for model in ("M_clean_policy", "M_inclusive_policy"):
        metric = metrics.get(model) or {}
        confusion[model] = {key: metric.get(key) for key in ("full_correct", "full_total", "no_correct", "no_total")}
    card = {"canonical_confusion_summary": confusion,
            "source_sha256": {"candidate": value_sha(qa)}}
    options = qa.get("options") or qa.get("items") or {}
    return {
        "record_version": VERSION, "root_id": qa["root_id"],
        "source_candidate_id": entry["candidate_id"],
        "source_cluster_id": cluster["solver_prompt_fingerprint"],
        "source_arm": entry["normalized_experiment_arm"],
        "historical_qa": {"question_type": qa["question_type"], "question": qa["question"],
                          "options_or_items": options, "canonical_gold": qa.get("gold"),
                          "evidence_refs": list(qa.get("evidence_refs") or []),
                          "source_data_status": {"options_or_items": "present" if options else "missing",
                                                 "canonical_gold": "present" if qa.get("gold") is not None else "missing",
                                                 "evidence_refs": "present" if qa.get("evidence_refs") else "empty_or_missing"}},
        "performance": _performance(card), "canonical_observation": confusion,
        "backend_source": {"source_manifest_sha256": source_manifest_sha,
                           "candidate_record_sha256": value_sha(qa), "bank_record_sha256": value_sha(entry),
                           "cluster_projection_sha256": value_sha(cluster)},
    }


def feedback_envelope(root, records, reason=None):
    return {"filter_version": VERSION, "root_id": root,
            "status": "ELIGIBLE_FEEDBACK" if records else NO_FEEDBACK,
            "reason": None if records else reason,
            "records": list(records)}


def history_envelope(root):
    return {"filter_version": VERSION, "root_id": root, "status": HISTORY_ONLY,
            "reason": "FRESH_USES_HISTORY_ONLY_NO_ACTIVE_SOURCE_BINDING", "records": []}


def validate_envelope(envelope, allowed_ids):
    required = {"filter_version", "root_id", "status", "reason", "records"}
    if not isinstance(envelope, dict) or not required <= set(envelope):
        raise ValueError("feedback_envelope_required_fields_missing")
    if envelope["filter_version"] != VERSION or not envelope["root_id"]:
        raise ValueError("feedback_envelope_identity_mismatch")
    records = envelope["records"]
    if not isinstance(records, list):
        raise ValueError("feedback_records_not_list")
    if envelope["status"] in {NO_FEEDBACK, HISTORY_ONLY}:
        if records or not isinstance(envelope["reason"], str) or not envelope["reason"]:
            raise ValueError("invalid_explicit_no_feedback")
        return
    if envelope["status"] != "ELIGIBLE_FEEDBACK" or not records or envelope["reason"] is not None:
        raise ValueError("invalid_eligible_feedback_state")
    seen = set()
    for record in records:
        cid = record.get("source_candidate_id")
        if cid not in allowed_ids or cid in seen or record.get("root_id") != envelope["root_id"]:
            raise ValueError("feedback_source_outside_scope_or_duplicate")
        seen.add(cid)
        for key in ("source_cluster_id", "performance", "canonical_observation", "backend_source"):
            if key not in record or not record[key]:
                raise ValueError(f"feedback_record_required_field_missing:{key}")
        qa = record.get("historical_qa")
        if not isinstance(qa, dict) or {"question_type", "question", "options_or_items", "canonical_gold", "evidence_refs", "source_data_status"} != set(qa):
            raise ValueError("historical_qa_incomplete")
        if not qa["question"] or not qa["options_or_items"] or qa["canonical_gold"] is None:
            raise ValueError("historical_qa_required_content_missing")
        expected_status = {"options_or_items": "present", "canonical_gold": "present",
                           "evidence_refs": "present" if qa["evidence_refs"] else "empty_or_missing"}
        if qa["source_data_status"] != expected_status:
            raise ValueError("historical_qa_status_content_mismatch")
        if set(record["performance"]) != {"M_clean_policy", "M_inclusive_policy"}:
            raise ValueError("unexpected_performance_source")
        confusion = record["canonical_observation"]
        if set(confusion) != {"basis", "M_clean_policy", "M_inclusive_policy"}:
            raise ValueError("unexpected_canonical_observation_fields")
        if confusion["basis"] != "canonical_option_or_item_ids_only":
            raise ValueError("unexpected_canonical_observation_basis")
        for model in ("M_clean_policy", "M_inclusive_policy"):
            if set(confusion[model]) != {"full_correct", "full_total", "no_correct", "no_total"}:
                raise ValueError("unexpected_observation_metric_fields")
            for context in ("full", "no"):
                correct, total = confusion[model][context + "_correct"], confusion[model][context + "_total"]
                if correct is None and total is None:
                    continue
                if type(correct) is not int or type(total) is not int or not 0 <= correct <= total:
                    raise ValueError("invalid_canonical_observation_counts")
        expected_performance = _performance({"canonical_confusion_summary": confusion,
                                            "source_sha256": {"candidate": record["backend_source"]["candidate_record_sha256"]}})
        if record["performance"] != expected_performance:
            raise ValueError("performance_observation_projection_mismatch")


def fresh_view(root, history):
    rows = [row for row in history if row["root_id"] == root]
    if len(rows) > 10:
        raise ValueError("fresh_history_limit_exceeded")
    return {"root_id": root, "historical_coverage": [
        {key: row[key] for key in ("normalized_stem_signature", "prompt_cluster_count", "candidate_count")}
        for row in rows], "constraint_note": "signature_only_duplicate_avoidance; no source QA binding"}


def render_role(role, envelope, dialogue, allowed_ids, *, history=(), schema=None,
                strategy_schema=None):
    validate_envelope(envelope, allowed_ids)
    root, records = envelope["root_id"], envelope["records"]
    if dialogue.get("root_id") != root or not dialogue.get("turns"):
        raise ValueError("role_dialogue_identity_or_content_missing")
    if role not in {"expansion", "targeted", "composition", "fresh"}:
        raise ValueError("unsupported_role")
    if envelope["status"] == HISTORY_ONLY and role != "fresh":
        raise ValueError("history_only_status_is_fresh_only")
    base = {"filter_version": VERSION, "root_id": root, "role": role,
            "offline_only": True, "api_call_authorized": False,
            "feedback_status": envelope["status"]}
    if role in {"targeted", "composition"} and len(records) < (2 if role == "composition" else 1):
        return {**base, "dispatch_status": "UNSCHEDULABLE_INSUFFICIENT_SOURCES",
                "reason": envelope["reason"] or "INSUFFICIENT_COMPOSITION_PARENTS",
                "feedback_view": {"status": envelope["status"], "source_count": len(records)}, "messages": []}
    if role == "targeted" and len(records) != 1:
        raise ValueError("targeted_requires_exactly_one_source")
    if role == "composition" and (len(records) != 2 or len({r["source_cluster_id"] for r in records}) != 2):
        raise ValueError("composition_requires_two_distinct_clusters")
    if role == "fresh":
        view = fresh_view(root, history)
    elif role == "composition":
        view = qa_composition_view(records)
    elif role == "targeted":
        view = qa_targeted_view(records[0])
    else:
        view = {"filter_version": VERSION, "root_id": root, "status": envelope["status"],
                "reason": envelope["reason"], "feedback": [expansion_view_v3(r) for r in records]}
    template = render(EXPANSION_PLANNER_TEMPLATE_V4 if role == "expansion" else QA_TEMPLATE_V3, {
        "ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW": canonical(view),
        "AUTHORITATIVE_CUMULATIVE_DIALOGUE": "The authoritative transcript is supplied once by the role builder.",
        "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT": "Use the existing Planner JSON output contract. Missing historical feedback requires no replacement question.",
        "EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT": "Use the frozen revision-1 output contract.",
    })
    if role == "expansion":
        parent = {"root_id": root, "dialogue": {"turns": [
            {"turn_id": t.get("turn_ref") or t.get("turn_id"), "speaker": t["speaker"], "text": t["text"]}
            for t in dialogue["turns"]]}}
        prompt = build_planner_prompt(parent=parent, memory_overview={}, candidates=[], verified_tracks=[],
                                      include_authoritative_transcript=True, event_bucket_only=True,
                                      memory_only_fast=True, strict_speaker_refs=True,
                                      expose_complete_memory=True,
                                      memory_source_mode="event_bucket_only")["prompt"] + "\n\n" + template
    else:
        if not schema:
            raise ValueError("qa_schema_required")
        if role == "targeted":
            if not strategy_schema:
                raise ValueError("targeted_strategy_schema_required")
            prompt = targeted_prompt(view, dialogue, template=template, schema=schema, strategy_schema=strategy_schema)
        else:
            prompt = multi_pass_prompt(role, {"mode": role, "shared_feedback_view": view},
                                       dialogue, template=template, schema=schema)
    return {**base, "dispatch_status": "OFFLINE_RENDERED_NOT_SENT", "feedback_view": view,
            "messages": [{"role": "user", "content": prompt}]}
