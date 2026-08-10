"""Independent semantic audit contract for generated core QA objects."""

from __future__ import annotations

import inspect
import json
import re
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes


QA_SEMANTIC_AUDIT_VERSION = (
    "slim_core_qa_semantic_audit_v5_explicit_evidence_gaps")
QA_SEMANTIC_AUDIT_PROMPT_VERSION = (
    "slim_core_qa_entailment_reviewer_v5_explicit_evidence_gaps")
QA_SEMANTIC_AUDIT_MAX_TOKENS = 1536
DECISIONS = frozenset({"accept", "reject", "uncertain"})
ENTAILMENT = frozenset({"fully_supported", "partially_supported",
                        "unsupported", "uncertain"})
TERNARY = frozenset({"correct", "incorrect", "uncertain"})
RELATION_ENTAILMENT = frozenset({
    "explicitly_supported", "requires_inference", "unrelated", "uncertain"})

QA_SEMANTIC_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "audit_decision", "question_answerable", "answer_entailment",
        "speaker_attribution", "temporality_and_modality",
        "cross_boundary_same_development", "cross_boundary_relation_entailment",
        "support_set_sufficient",
        "question_answer_leakage", "grounded_answer", "unsupported_claims",
        "evidence_gaps",
        "evidence_span_ids_checked", "reason",
    ],
    "properties": {
        "audit_decision": {"enum": sorted(DECISIONS)},
        "question_answerable": {"type": "boolean"},
        "answer_entailment": {"enum": sorted(ENTAILMENT)},
        "speaker_attribution": {"enum": sorted(TERNARY)},
        "temporality_and_modality": {"enum": sorted(TERNARY)},
        "cross_boundary_same_development": {"type": "boolean"},
        "cross_boundary_relation_entailment": {
            "enum": sorted(RELATION_ENTAILMENT)},
        "support_set_sufficient": {"type": "boolean"},
        "question_answer_leakage": {
            "enum": ["clean", "leaking", "uncertain"]},
        "grounded_answer": {"type": "string"},
        "unsupported_claims": {
            "type": "array", "items": {"type": "string"}},
        "evidence_gaps": {
            "type": "array", "items": {"type": "string"}},
        "evidence_span_ids_checked": {
            "type": "array", "minItems": 2, "maxItems": 6,
            "uniqueItems": True, "items": {"type": "string"}},
        "reason": {"type": "string", "minLength": 1},
    },
}


def build_qa_semantic_audit_prompt(
    *, core_qa: Mapping[str, Any],
    selected_evidence: Sequence[Mapping[str, Any]],
    complete_dialogue: Sequence[Mapping[str, Any]] = (),
) -> str:
    evidence = [{
        "span_id": row["span_id"],
        "source_partition": row["source_partition"],
        "session_id": row["session_id"],
        "speaker": row["speaker"],
        "exact_text": row["exact_text"],
    } for row in selected_evidence]
    submitted_partitions = {
        str(row.get("source_partition") or "") for row in selected_evidence}
    cross_boundary = submitted_partitions == {
        "original_parent", "promoted_child"}
    dialogue = [{
        "turn_id": row["turn_id"],
        "source_partition": row.get("source_partition"),
        "session_id": row["session_id"],
        "speaker": row["speaker"],
        "text": row.get("text", row.get("exact_text")),
    } for row in complete_dialogue]
    return "\n\n".join([
        f"PROMPT_VERSION: {QA_SEMANTIC_AUDIT_PROMPT_VERSION}",
        "ROLE: Independent QA evidence auditor.",
        "Audit the submitted question and answer against COMPLETE_DIALOGUE. Separately audit whether SELECTED_EVIDENCE is itself a sufficient support set. Do not use World Memory or outside knowledge. Do not rewrite the question, replace evidence, or search for a better evidence set.",
        "answer_entailment, speaker_attribution, and temporality_and_modality describe support in COMPLETE_DIALOGUE. For a parent_child evidence set, cross_boundary_same_development and cross_boundary_relation_entailment also describe the claimed cross-boundary relation in COMPLETE_DIALOGUE. For parent_only, child_only, or multi_child scope, set cross_boundary_same_development=false and cross_boundary_relation_entailment=unrelated; those two fields are not rejection gates.",
        "audit_decision and unsupported_claims describe only whether the submitted question and answer are correct in COMPLETE_DIALOGUE. unsupported_claims must never contain a claim merely because SELECTED_EVIDENCE omitted its supporting turn.",
        "support_set_sufficient and evidence_gaps describe only the Questioner's SELECTED_EVIDENCE. Set support_set_sufficient=false and list concrete omissions in evidence_gaps when the answer is correct in COMPLETE_DIALOGUE but the submitted IDs omit an antecedent, cite unrelated turns, or otherwise fail to prove it. Evidence insufficiency alone must not turn a correct full-dialogue answer into audit_decision=reject.",
        f"SUBMITTED_EVIDENCE_SCOPE: {'parent_child' if cross_boundary else 'single_partition_or_multi_child'}",
        "Preserve modality exactly when auditing both sources: planned/intended/possible language may not be upgraded to a completed fact.",
        "Temporal order, the same speaker, or a shared broad topic does not by itself prove influence, causation, motivation, or change. If the complete dialogue does not explicitly support the relation asked by the question, set cross_boundary_relation_entailment to requires_inference (or uncertain/unrelated) and do not accept. Never manufacture connectors such as 'led to', 'therefore', or 'influenced'.",
        "Independently check whether the question itself reveals the decisive initial or later answer. A topic anchor is allowed, but a question that states the concrete outcome it asks for has question_answer_leakage=leaking and must not be accepted.",
        "When and only when the submitted answer is supported by COMPLETE_DIALOGUE, write grounded_answer as a concise 1-2 sentence semantic synthesis that directly answers the question. It must be 8-40 words, preserve attribution and modality, and must not copy the submitted answer. For reject/uncertain, grounded_answer must be an empty string.",
        "If the question or answer is unsupported, overstated, misattributed, temporally changed, or requires an unstated relation even with COMPLETE_DIALOGUE, audit_decision must be reject or uncertain and unsupported_claims must identify the full-dialogue defect. A valid reject is final and must not be retried for a better outcome.",
        "Consistency is mandatory: answer_entailment=fully_supported requires unsupported_claims=[]; support_set_sufficient=true requires evidence_gaps=[]; support_set_sufficient=false requires at least one evidence_gaps item.",
        "evidence_span_ids_checked must contain exactly the submitted answer_support_span_ids in the same order. Return no additional fields.",
        "CORE_QA\n" + json.dumps(dict(core_qa), ensure_ascii=False,
                                  sort_keys=True, separators=(",", ":")),
        "COMPLETE_DIALOGUE\n" + json.dumps(dialogue, ensure_ascii=False,
                                             sort_keys=True,
                                             separators=(",", ":")),
        "SELECTED_EVIDENCE\n" + json.dumps(evidence, ensure_ascii=False,
                                             sort_keys=True,
                                             separators=(",", ":")),
        "OUTPUT_SCHEMA\n" + json.dumps(QA_SEMANTIC_AUDIT_SCHEMA,
                                         ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")),
    ])


def validate_qa_semantic_audit_output(
    raw: Any, *, expected_span_ids: Sequence[str],
    submitted_extractive_answer: str | None = None,
    requires_cross_boundary: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    required = set(QA_SEMANTIC_AUDIT_SCHEMA["required"])
    if not isinstance(raw, dict):
        return {"decision": "qa_semantic_audit_schema_invalid",
                "errors": ["audit_not_object"], "audit": None}
    # Read-only compatibility for v4 accepted/sufficient fixtures.  An old
    # insufficient-evidence object cannot be migrated because its missing gap
    # description is exactly the ambiguity v5 removes.
    legacy_required = required - {"evidence_gaps"}
    if (set(raw) == legacy_required and
            raw.get("support_set_sufficient") is True):
        raw = {**raw, "evidence_gaps": []}
    if set(raw) != required:
        errors.append("audit_field_set_invalid")
    if raw.get("audit_decision") not in DECISIONS:
        errors.append("audit_decision_invalid")
    if not isinstance(raw.get("question_answerable"), bool):
        errors.append("audit_question_answerable_invalid")
    if raw.get("answer_entailment") not in ENTAILMENT:
        errors.append("audit_answer_entailment_invalid")
    for field in ("speaker_attribution", "temporality_and_modality"):
        if raw.get(field) not in TERNARY:
            errors.append(f"audit_{field}_invalid")
    for field in ("cross_boundary_same_development", "support_set_sufficient"):
        if not isinstance(raw.get(field), bool):
            errors.append(f"audit_{field}_invalid")
    if raw.get("cross_boundary_relation_entailment") not in RELATION_ENTAILMENT:
        errors.append("audit_cross_boundary_relation_entailment_invalid")
    if raw.get("question_answer_leakage") not in {
            "clean", "leaking", "uncertain"}:
        errors.append("audit_question_answer_leakage_invalid")
    grounded_answer = raw.get("grounded_answer")
    if not isinstance(grounded_answer, str):
        errors.append("audit_grounded_answer_invalid")
    claims = raw.get("unsupported_claims")
    if (not isinstance(claims, list) or
            any(not isinstance(value, str) or not value.strip()
                for value in claims)):
        errors.append("audit_unsupported_claims_invalid")
    evidence_gaps = raw.get("evidence_gaps")
    if (not isinstance(evidence_gaps, list) or
            any(not isinstance(value, str) or not value.strip()
                for value in evidence_gaps)):
        errors.append("audit_evidence_gaps_invalid")
    checked = raw.get("evidence_span_ids_checked")
    if checked != list(expected_span_ids):
        errors.append("audit_evidence_set_changed")
    if not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
        errors.append("audit_reason_invalid")
    if not errors:
        if raw["answer_entailment"] == "fully_supported" and claims:
            errors.append(
                "audit_full_dialogue_entailment_claims_inconsistent")
        if raw["support_set_sufficient"] is True and evidence_gaps:
            errors.append("audit_evidence_gaps_present_when_sufficient")
        if raw["support_set_sufficient"] is False and not evidence_gaps:
            errors.append("audit_evidence_gaps_missing_when_insufficient")
    if not errors:
        answer_words = re.findall(r"[\w']+", grounded_answer)
        if raw["audit_decision"] == "accept":
            if not 8 <= len(answer_words) <= 40:
                errors.append("audit_grounded_answer_length_invalid")
            if submitted_extractive_answer:
                normalized_grounded = " ".join(answer_words).casefold()
                normalized_extractive = " ".join(re.findall(
                    r"[\w']+", submitted_extractive_answer)).casefold()
                if (normalized_grounded == normalized_extractive or
                        SequenceMatcher(
                            a=normalized_grounded, b=normalized_extractive,
                            autojunk=False).ratio() >= 0.92):
                    errors.append("audit_grounded_answer_is_extractive_copy")
        elif grounded_answer.strip():
            errors.append("audit_nonaccept_grounded_answer_must_be_empty")
    if not errors:
        answer_accept_conditions = all((
            raw["question_answerable"] is True,
            raw["answer_entailment"] == "fully_supported",
            raw["speaker_attribution"] == "correct",
            raw["temporality_and_modality"] == "correct",
            (not requires_cross_boundary or
             raw["cross_boundary_same_development"] is True),
            (not requires_cross_boundary or
             raw["cross_boundary_relation_entailment"] ==
             "explicitly_supported"),
            raw["question_answer_leakage"] == "clean",
            raw["unsupported_claims"] == [],
        ))
        if raw["audit_decision"] == "accept" and not answer_accept_conditions:
            errors.append("audit_accept_conditions_not_met")
        if raw["audit_decision"] != "accept" and answer_accept_conditions:
            errors.append("audit_nonaccept_without_defect")
    if errors:
        return {"decision": "qa_semantic_audit_schema_invalid",
                "errors": errors, "audit": None}
    answer_verified = raw["audit_decision"] == "accept"
    evidence_sufficient = bool(raw["support_set_sufficient"])
    return {"decision": "qa_semantic_audit_contract_accept",
            "errors": [], "audit": dict(raw),
            "answer_decision": raw["audit_decision"],
            "evidence_decision": (
                "sufficient" if evidence_sufficient else "insufficient"),
            "answer_semantically_verified": answer_verified,
            "submitted_evidence_sufficient": evidence_sufficient,
            "semantic_entailment_verified": (
                answer_verified and evidence_sufficient)}


def qa_semantic_audit_protocol_sha256() -> str:
    material = canonical_json({
        "version": QA_SEMANTIC_AUDIT_VERSION,
        "prompt_version": QA_SEMANTIC_AUDIT_PROMPT_VERSION,
        "schema": QA_SEMANTIC_AUDIT_SCHEMA,
        "prompt_source": inspect.getsource(build_qa_semantic_audit_prompt),
        "validator_source": inspect.getsource(validate_qa_semantic_audit_output),
    })
    return sha256_bytes(material.encode("utf-8"))
