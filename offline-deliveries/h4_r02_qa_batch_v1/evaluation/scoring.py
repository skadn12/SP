"""Parsing and canonical scoring for full-only Solver responses."""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Mapping

BOX = re.compile(r"\s*<box>\s*(.*?)\s*</box>\s*\Z", re.S)
ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def parse_box(raw: Any) -> tuple[list[str] | None, str | None]:
    if not isinstance(raw, str):
        return None, "raw_output_missing_or_not_text"
    match = BOX.fullmatch(raw)
    if match is None:
        return None, "exactly_one_box_required"
    text = match.group(1).strip()
    if not text:
        return [], None
    values = [part.strip().upper() for part in text.split(",")]
    if any(not ID.fullmatch(value) for value in values):
        return None, "malformed_id_list"
    if len(values) != len(set(values)):
        return None, "duplicate_id"
    return values, None


def score_response(question: Mapping[str, Any], request: Mapping[str, Any], raw: Any) -> dict[str, Any]:
    """Parse displayed IDs, map them to stable canonical content IDs, and score."""
    qtype = question.get("question_type")
    field = "options" if qtype == "multi_select" else "items"
    displayed = request.get("variant", {}).get(field)
    mapping = request.get("variant", {}).get("display_to_content")
    gold = question.get("gold_content_ids")
    if qtype not in {"multi_select", "ordering"} or not isinstance(displayed, dict):
        raise ValueError("question_variant_contract_invalid")
    if not isinstance(mapping, dict) or set(mapping) != set(displayed):
        raise ValueError("display_to_content_not_bijective")
    if not isinstance(gold, list) or len(gold) < 2:
        raise ValueError("gold_content_ids_required")
    values, reason = parse_box(raw)
    result: dict[str, Any] = {
        "parse_status": "success" if reason is None else "failure",
        "parse_reason": reason,
        "scored": False,
        "display_prediction": values,
        "content_prediction": None,
        "content_gold": list(gold),
        "tp": None, "fp": None, "fn": None,
        "exact_match": None, "partial_score": None,
    }
    if values is None:
        result["scoring_status"] = "PARSE_FAILURE"
        return result
    invalid = [value for value in values if value not in mapping]
    if invalid:
        result.update(scoring_status="INVALID_ID", invalid_ids=invalid)
        return result
    prediction = [mapping[value] for value in values]
    if len(set(prediction)) != len(prediction):
        result.update(scoring_status="DUPLICATE_CONTENT_ID", invalid_ids=[])
        return result
    if qtype == "multi_select":
        pred_set, gold_set = set(prediction), set(gold)
        tp, fp, fn = len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
        n = len(displayed); g = len(gold_set)
        partial = Fraction(tp, g) if g == n else Fraction(tp, g) - Fraction(fp, n - g)
        exact = pred_set == gold_set
        content_prediction = sorted(pred_set)
    else:
        if len(prediction) != len(gold) or set(prediction) != set(gold):
            result.update(scoring_status="INCOMPLETE_ORDER", content_prediction=prediction)
            return result
        total = len(gold) * (len(gold) - 1) // 2
        rank = {value: index for index, value in enumerate(prediction)}
        wrong = sum(rank[a] > rank[b] for index, a in enumerate(gold) for b in gold[index + 1:])
        correct = total - wrong
        tp, fp, fn = correct, wrong, 0
        partial = Fraction(correct - wrong, total)
        exact = prediction == gold
        content_prediction = prediction
    result.update(scoring_status="SCORED", scored=True, content_prediction=content_prediction,
                  tp=tp, fp=fp, fn=fn, exact_match=exact, partial_score=str(partial),
                  partial_score_fraction=partial)
    # Fraction is useful internally, but JSON records must remain serializable.
    result["partial_score_fraction"] = str(partial)
    return result

