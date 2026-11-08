"""Independent, pure scoring for the probe_v3 multi-select response format."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_BOX = re.compile(r"\s*<box>\s*(.*?)\s*</box>\s*", re.DOTALL)
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _inputs(question: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[dict, dict, set]:
    """Reject inconsistent question/display metadata before scoring a response."""
    if question.get("question_type") != "multi_select" or request.get("question_type") != "multi_select":
        raise ValueError("multi_select_question_and_request_required")
    if question.get("candidate_id") != request.get("candidate_id"):
        raise ValueError("candidate_id_mismatch")
    options = question.get("options")
    if not isinstance(options, Mapping) or not options:
        raise ValueError("canonical_options_required")
    if "gold" not in question or not isinstance(question["gold"], list):
        raise ValueError("question_gold_list_required")
    canonical = set(options)
    gold = question["gold"]
    if any(not isinstance(item, str) or item not in canonical for item in gold) or len(gold) != len(set(gold)):
        raise ValueError("invalid_canonical_gold")
    variant = request.get("variant")
    if not isinstance(variant, Mapping):
        raise ValueError("request_variant_required")
    mapping = variant.get("display_to_canonical")
    displayed = variant.get("options")
    if not isinstance(mapping, Mapping) or not isinstance(displayed, Mapping):
        raise ValueError("display_mapping_and_options_required")
    if (set(mapping) != set(displayed) or set(mapping.values()) != canonical
            or len(mapping) != len(canonical)):
        raise ValueError("display_mapping_not_bijective")
    if any(displayed[display_id] != options[canonical_id]
           for display_id, canonical_id in mapping.items()):
        raise ValueError("display_option_text_mismatch")
    if "displayed_gold" in variant:
        expected = {display_id for display_id, canonical_id in mapping.items() if canonical_id in gold}
        if (not isinstance(variant["displayed_gold"], list)
                or len(variant["displayed_gold"]) != len(expected)
                or set(variant["displayed_gold"]) != expected):
            raise ValueError("displayed_gold_mismatch")
    return dict(options), dict(mapping), set(gold)


def score_response(
    question: Mapping[str, Any],
    request: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one probe_v3 multi-select response against embedded canonical gold.

    ``question`` is a candidates.json entry, or a run/questions.json entry
    merged with run/gold.json by candidate_id. ``request`` is one manifest
    request and ``response`` is the matching responses/<request_id>.json row.
    Stored response.score is ignored. Bad metadata raises ValueError; malformed
    model output is returned as an unscored result.
    """
    options, mapping, gold = _inputs(question, request)
    if response.get("request_id") != request.get("request_id"):
        raise ValueError("response_request_id_mismatch")
    raw = response.get("raw_output")
    result: dict[str, Any] = {
        "request_id": request["request_id"],
        "candidate_id": question["candidate_id"],
        "raw_output": raw,
        "parse_status": "parse_failure",
        "scoring_status": "PARSE_FAILURE",
        "parse_reason": None,
        "scored": False,
        "mapping": mapping.copy(),
        "display_prediction": None,
        "canonical_prediction": None,
        "canonical_gold": sorted(gold),
        "invalid_ids": [],
        "n": len(options),
        "g": len(gold),
        "tp": None,
        "fp": None,
        "fn": None,
        "exact_match": None,
        "partial_score": None,
    }
    if not isinstance(raw, str):
        result["parse_status"] = "call_failure"
        result["scoring_status"] = "CALL_FAILURE"
        result["parse_reason"] = "raw_output_missing_or_not_text"
        return result
    match = _BOX.fullmatch(raw)
    if match is None:
        result["parse_reason"] = "exactly_one_box_required"
        return result
    inner = match.group(1).strip()
    if not inner:
        tokens: list[str] = []
    else:
        tokens = [part.strip().upper() for part in inner.split(",")]
        if any(not _ID.fullmatch(token) for token in tokens):
            result["parse_reason"] = "malformed_id_list"
            return result
        if len(tokens) != len(set(tokens)):
            result["parse_reason"] = "duplicate_id"
            return result
    result["display_prediction"] = tokens
    invalid = [token for token in tokens if token not in mapping]
    if invalid:
        result["parse_status"] = "invalid_id"
        result["scoring_status"] = "INVALID_ID"
        result["parse_reason"] = "unknown_display_id"
        result["invalid_ids"] = invalid
        return result

    prediction = {mapping[token] for token in tokens}
    tp = len(prediction & gold)
    fp = len(prediction - gold)
    fn = len(gold - prediction)
    n = len(options)
    g = len(gold)
    result.update({
        "parse_status": "success",
        "scoring_status": "SCORED" if n == 6 and 2 <= g <= n else "PARTIAL_NOT_APPLICABLE",
        "scored": True,
        "canonical_prediction": sorted(prediction),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_match": prediction == gold,
        "partial_score": (tp / g - fp / (n - g) if g < n else tp / g)
        if n == 6 and 2 <= g <= n else None,
    })
    return result


__all__ = ["score_response"]
