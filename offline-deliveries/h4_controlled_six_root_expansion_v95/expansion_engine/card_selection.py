"""Single-source event-card extraction with strict legacy validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parents[1]
LEGACY_PATH = DELIVERIES / "h4_r02_dualroute_feedback_v1" / "dataset_cards.py"
_spec = importlib.util.spec_from_file_location("r02_card_selection_legacy", LEGACY_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load frozen dataset card validator: {LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

MODEL = _legacy.MODEL
SCHEMA_VERSION = _legacy.SCHEMA_VERSION
SELECTION_POLICY = "first_fully_valid_candidate_in_model_order_one_card_per_source"


class DuplicateJSONKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _parse(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
    elif isinstance(raw, dict):
        parsed = raw
    else:
        raise ValueError("response: expected JSON text or object")
    if not isinstance(parsed, dict):
        raise ValueError("response: expected object")
    return parsed


def _one_source(subset: dict[str, Any]) -> dict[str, Any]:
    items = subset.get("items") if isinstance(subset, dict) else None
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise ValueError("single_source_required")
    source_id = items[0].get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("single_source_id_required")
    return items[0]


def build_extraction_request(
    subset: dict[str, Any], repair: dict[str, Any] | str | None = None
) -> dict[str, Any]:
    """Build a Qwen request bound to one source; repair feedback never changes source data."""
    item = _one_source(subset)
    payload = _legacy.build_extraction_request(subset)
    content = payload["messages"][0]["content"]
    content = (
        "For this one source item, select at most ONE event, even if it contains several events. "
        "Return cards with zero or one card only. If several events are usable, choose the first "
        "complete event encountered in source order; do not return one card per event. "
        f"Every returned card must use source_id {item['source_id']!r}.\n"
        + content
    )
    if repair is not None:
        content += (
            "\nREPAIR_FEEDBACK_JSON:\n"
            + json.dumps(repair, ensure_ascii=True, sort_keys=True)
            + "\nCorrect only the reported extraction/schema errors using the same source item. "
            "Still return zero or one card, and preserve exact evidence and source state."
        )
    content += (
        "\nCURRENT EXTRACTION CONTRACT: downgrade_reason is always a JSON string, "
        "never null. A card has 1-6 steps and at most 3 explicit dependencies. "
        "On a repair, keep these same limits; do not expand to one step per source turn. "
        "Group related source actions semantically while preserving distinct actors, "
        "requests, failures, retries and outcome states in their exact source evidence. "
        "If no complete faithful card fits, return an empty cards array.\n"
    )
    payload["messages"][0]["content"] = content
    return payload


def _error_code(reason: str) -> str:
    reason_lower = reason.lower()
    if "duplicate_json_key" in reason_lower:
        return "DUPLICATE_JSON_KEY"
    if "json" in reason_lower or "response:" in reason_lower:
        return "MALFORMED_RESPONSE"
    if "unknown source_id" in reason_lower:
        return "UNKNOWN_SOURCE"
    if "quote" in reason_lower or "evidence" in reason_lower or "source unit" in reason_lower:
        return "EVIDENCE_INVALID"
    if "state" in reason_lower or "speaker" in reason_lower or "actor" in reason_lower:
        return "SOURCE_STATE_OR_ACTOR_INVALID"
    return "CARD_SCHEMA_INVALID"


def select_cards(raw: str | dict[str, Any], subset: dict[str, Any]) -> dict[str, Any]:
    """Validate candidates independently and select at most the first fully valid card."""
    source_item = _one_source(subset)
    source_id = source_item["source_id"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": subset.get("manifest_sha256"),
        "accepted_cards": [],
        "rejected_records": [],
        "candidate_results": [],
        "selection_policy": SELECTION_POLICY,
        "missing_source_ids": [source_id],
        "retryable": False,
        "repair_feedback": {"source_id": source_id, "candidate_errors": []},
    }

    try:
        parsed = _parse(raw)
        _legacy._object_keys(parsed, {"schema_version", "cards"}, "response")
        if parsed["schema_version"] != SCHEMA_VERSION:
            raise ValueError("response: schema version mismatch")
        candidates = parsed["cards"]
        if not isinstance(candidates, list):
            raise ValueError("response: cards must be a list")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        reason = str(exc)
        record = {"scope": "response", "reason": reason, "error_code": _error_code(reason)}
        result["rejected_records"].append(record)
        result["repair_feedback"]["response_error"] = record
        result["retryable"] = True
        return result

    if not candidates:
        return result

    for index, candidate in enumerate(candidates):
        candidate_result = {"candidate_index": index}
        one_card = {"schema_version": SCHEMA_VERSION, "cards": [candidate]}
        checked = _legacy.validate_extracted_cards(one_card, subset)
        if checked["accepted_cards"]:
            normalized = checked["accepted_cards"][0]
            if not result["accepted_cards"]:
                result["accepted_cards"].append(normalized)
                result["missing_source_ids"] = []
                candidate_result.update({"status": "selected", "card_id": normalized["card_id"]})
            else:
                record = {
                    "scope": "candidate", "index": index,
                    "source_id": normalized["source_id"], "card_id": normalized["card_id"],
                    "reason": "valid_unselected_never_reuse_source",
                }
                result["rejected_records"].append(record)
                candidate_result.update({"status": "valid_unselected_never_reuse_source",
                                         "card_id": normalized["card_id"]})
        else:
            rejection = checked["rejected_records"][0] if checked["rejected_records"] else {
                "scope": "candidate", "index": index, "reason": "card_validation_failed"
            }
            reason = rejection.get("reason", "card_validation_failed")
            record = {
                "scope": "candidate", "index": index,
                "source_id": candidate.get("source_id") if isinstance(candidate, dict) else None,
                "reason": reason, "error_code": _error_code(reason),
            }
            result["rejected_records"].append(record)
            result["repair_feedback"]["candidate_errors"].append({
                "candidate_index": index,
                "source_id": record["source_id"],
                "error_code": record["error_code"],
                "reason": reason,
            })
            candidate_result.update({"status": "invalid", "reason": reason,
                                     "error_code": record["error_code"]})
        result["candidate_results"].append(candidate_result)

    result["retryable"] = bool(
        not result["accepted_cards"] and result["repair_feedback"]["candidate_errors"]
    )
    return result
