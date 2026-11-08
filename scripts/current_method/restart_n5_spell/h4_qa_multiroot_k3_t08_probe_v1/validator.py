"""Minimum deterministic candidate validation for the formal fresh-QA flow."""

from __future__ import annotations

from typing import Any, Mapping


QUESTION_TYPES = {"single_choice", "multi_select", "ordering"}
SINGLE_KEYS = ("A", "B", "C", "D")


def _expected_keys(question_type: str, count: int) -> tuple[str, ...]:
    if question_type == "single_choice":
        return SINGLE_KEYS if count == 4 else ()
    if question_type == "multi_select" and 4 <= count <= 6:
        return tuple(chr(ord("A") + index) for index in range(count))
    if question_type == "ordering" and 4 <= count <= 6:
        return tuple(f"I{index + 1}" for index in range(count))
    return ()


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def validate_candidate(candidate: Mapping[str, Any], *, valid_turn_refs: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    qtype = str(candidate.get("question_type") or "")
    if qtype not in QUESTION_TYPES:
        errors.append("question_type_invalid")
    question = candidate.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append("question_missing")

    container = candidate.get("items") if qtype == "ordering" else candidate.get("options")
    actual_count = len(container) if isinstance(container, Mapping) else 0
    expected_keys = _expected_keys(qtype, actual_count)
    if not expected_keys or not isinstance(container, Mapping) or tuple(container.keys()) != expected_keys:
        errors.append("options_or_items_invalid")
        container = container if isinstance(container, Mapping) else {}
    texts = [_norm(container.get(key)) for key in expected_keys]
    if any(not text for text in texts):
        errors.append("option_or_item_text_missing")
    if len(texts) != len(set(texts)):
        errors.append("duplicate_option_or_item_text")

    if qtype == "single_choice":
        gold = str(candidate.get("correct_option") or "")
        if gold not in expected_keys:
            errors.append("single_gold_invalid")
    elif qtype == "multi_select":
        gold = [str(x) for x in candidate.get("gold") or []]
        if not 2 <= len(gold) <= len(expected_keys) or len(gold) != len(set(gold)) or any(x not in expected_keys for x in gold):
            errors.append("multi_gold_invalid")
    elif qtype == "ordering":
        gold = [str(x) for x in candidate.get("gold") or []]
        if len(gold) != len(expected_keys) or set(gold) != set(expected_keys):
            errors.append("ordering_gold_invalid")

    refs = [str(x) for x in candidate.get("evidence_refs") or []]
    if not refs:
        errors.append("evidence_refs_missing")
    if len(refs) != len(set(refs)):
        errors.append("duplicate_evidence_refs")
    if any(ref not in valid_turn_refs for ref in refs):
        errors.append("evidence_ref_unknown")

    if len(refs) < 2:
        warnings.append("narrow_evidence_scope")
    if not candidate.get("reasoning_groups_used"):
        warnings.append("reasoning_groups_missing")
    if str(candidate.get("hypothesized_reasoning_type") or "unknown") == "unknown":
        warnings.append("unknown_reasoning_type")

    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "decision": "pass" if not errors else "reject",
        "schema_valid": not errors,
        "blocking_errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "evidence_refs_present": bool(refs),
        "evidence_refs_all_exist": bool(refs) and not any(ref not in valid_turn_refs for ref in refs),
    }


def validate_batch(candidates: list[Mapping[str, Any]], *, valid_turn_refs: set[str]) -> dict[str, Any]:
    rows = [validate_candidate(row, valid_turn_refs=valid_turn_refs) for row in candidates]
    seen: set[str] = set()
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in seen:
            row["decision"] = "reject"
            row["schema_valid"] = False
            row["blocking_errors"] = sorted(set(row["blocking_errors"] + ["duplicate_candidate_id"]))
        seen.add(candidate_id)
    return {
        "valid": all(row["decision"] == "pass" for row in rows),
        "candidate_count": len(rows),
        "valid_count": sum(row["decision"] == "pass" for row in rows),
        "rows": rows,
    }
