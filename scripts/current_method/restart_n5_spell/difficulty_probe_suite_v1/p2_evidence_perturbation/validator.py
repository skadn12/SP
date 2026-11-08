from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


PERTURBATION_TYPES = {
    "background": {"background", "related_event"},
    "competing_state": {"alternative_state", "intermediate_state"},
    "entity_event_interference": {"entity_interference", "related_event"},
}
RELATIONS_TO_GOLD = {
    "background": "non_answer_background",
    "competing_state": "historical_non_final_alternative",
    "entity_event_interference": "other_entity_only",
}
TEMPORAL_RELATIONS = {
    "before_final_state", "concurrent_background", "after_final_state_nonchanging",
}
STATE_STATUSES = {
    "background_only", "historically_true_not_final", "other_entity_only",
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE) is not None


def validate_generation(
    value: Any, *, perturbation_type: str, qa: Mapping[str, Any],
    valid_turn_refs: Sequence[str], valid_speakers: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["generation_not_object"]}
    if str(value.get("perturbation_type") or "") != perturbation_type:
        errors.append("perturbation_type_mismatch")
    additions = value.get("added_information")
    if not isinstance(additions, list) or not additions:
        errors.append("added_information_required")
        additions = []
    allowed_types = PERTURBATION_TYPES.get(perturbation_type, set())
    seen_content: set[str] = set()
    gold = _text(qa.get("gold_text")).casefold()
    for index, row in enumerate(additions, 1):
        prefix = f"addition_{index}"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        if str(row.get("type") or "") not in allowed_types:
            errors.append(f"{prefix}_classification")
        content = _text(row.get("content"))
        if not content:
            errors.append(f"{prefix}_content_empty")
        elif content.casefold() in seen_content:
            errors.append(f"{prefix}_duplicate")
        seen_content.add(content.casefold())
        if row.get("affects_answer") is not False:
            errors.append(f"{prefix}_affects_answer")
        if str(row.get("relation_to_gold") or "") != RELATIONS_TO_GOLD.get(perturbation_type):
            errors.append(f"{prefix}_relation_to_gold")
        if str(row.get("insert_after_turn_ref") or "") not in set(valid_turn_refs):
            errors.append(f"{prefix}_insertion_ref")
        speaker = _text(row.get("speaker"))
        if not speaker:
            errors.append(f"{prefix}_speaker_binding")
        elif speaker not in set(valid_speakers):
            if perturbation_type != "entity_event_interference" or speaker.casefold() not in content.casefold():
                errors.append(f"{prefix}_speaker_binding")
        if str(row.get("temporal_relation") or "") not in TEMPORAL_RELATIONS:
            errors.append(f"{prefix}_temporal_relation")
        if str(row.get("state_status") or "") not in STATE_STATUSES:
            errors.append(f"{prefix}_state_status")
        if _contains_phrase(content, gold):
            errors.append(f"{prefix}_gold_text_leakage")
    claim = value.get("preservation_claim")
    if not isinstance(claim, Mapping):
        errors.append("preservation_claim_required")
    else:
        for field in ("answer_unchanged", "world_state_unchanged", "timeline_unchanged",
                      "entity_bindings_unchanged", "original_evidence_sufficient"):
            if claim.get(field) is not True:
                errors.append(f"preservation_claim_{field}")
    return {"valid": not errors, "errors": errors, "addition_count": len(additions)}


def validate_semantic_audit(
    value: Any, *, perturbation_type: str | None = None,
    expected_addition_count: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["semantic_audit_not_object"]}
    required_true = ("answer_preserved", "initial_state_preserved", "current_state_preserved",
                     "final_outcome_preserved", "entity_bindings_preserved",
                     "temporal_order_preserved", "original_evidence_still_sufficient")
    for field in required_true:
        if value.get(field) is not True:
            errors.append(field)
    if value.get("contradiction_added") is not False:
        errors.append("contradiction_added")
    if value.get("answer_leakage_added") is not False:
        errors.append("answer_leakage_added")
    if value.get("decision_stage_changed") is not False:
        errors.append("decision_stage_changed")
    judgments = value.get("added_fact_judgments")
    if not isinstance(judgments, list) or not judgments:
        errors.append("added_fact_judgments_required")
    else:
        if expected_addition_count is not None and len(judgments) != expected_addition_count:
            errors.append("added_fact_judgment_count")
        allowed = PERTURBATION_TYPES.get(str(perturbation_type), set())
        for index, row in enumerate(judgments, 1):
            if not isinstance(row, Mapping) or row.get("affects_answer") is not False:
                errors.append(f"added_fact_{index}_affects_answer")
            elif allowed and str(row.get("classification") or "") not in allowed:
                errors.append(f"added_fact_{index}_classification")
    return {"valid": not errors, "errors": errors, "raw_audit": dict(value)}


def materialize_view(
    transcript: Sequence[Mapping[str, Any]], *, qa_id: str,
    perturbation_type: str, additions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_ref: dict[str, list[Mapping[str, Any]]] = {}
    for row in additions:
        by_ref.setdefault(str(row.get("insert_after_turn_ref") or ""), []).append(row)
    output: list[dict[str, Any]] = []
    added_refs: list[str] = []
    counter = 0
    for original in transcript:
        output.append(dict(original))
        for addition in by_ref.get(str(original.get("turn_ref") or ""), []):
            counter += 1
            ref = f"P2E:{qa_id}:{perturbation_type}:A{counter:02d}"
            output.append({"turn_ref": ref, "speaker": str(addition["speaker"]),
                           "text": _text(addition["content"]), "origin": "probe_added_context",
                           "scene_ref": original.get("scene_ref")})
            added_refs.append(ref)
    return output, added_refs


def attest_original_preservation(
    original: Sequence[Mapping[str, Any]], perturbed: Sequence[Mapping[str, Any]],
    evidence_turn_refs: Sequence[str],
) -> dict[str, Any]:
    original_by_ref = {str(row.get("turn_ref")): row for row in original}
    perturbed_by_ref = {str(row.get("turn_ref")): row for row in perturbed}
    changed = []
    missing = []
    for ref, row in original_by_ref.items():
        if ref not in perturbed_by_ref:
            missing.append(ref)
        elif any(str(perturbed_by_ref[ref].get(key) or "") != str(row.get(key) or "")
                 for key in ("speaker", "text", "origin")):
            changed.append(ref)
    evidence_missing = [ref for ref in evidence_turn_refs if ref not in perturbed_by_ref]
    return {"valid": not missing and not changed and not evidence_missing,
            "missing_original_turn_refs": missing, "changed_original_turn_refs": changed,
            "missing_evidence_turn_refs": evidence_missing,
            "original_turn_count": len(original), "perturbed_turn_count": len(perturbed)}
