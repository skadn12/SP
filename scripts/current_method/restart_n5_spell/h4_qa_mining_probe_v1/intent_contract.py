from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


INTENT_TYPES = {
    "state_transition",
    "decision_boundary",
    "causal_chain",
    "temporal_evolution",
    "necessary_evidence",
}
ROLE_KEYS = ("before_state", "trigger", "after_state")
QUESTION_TYPES = ("single_choice", "multi_select", "ordering")


def _positions(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in value)
        and len(set(value)) == len(value)
    )


def allowed_question_types(intent_type: str) -> list[str]:
    # Phase 0 requires all three candidate families per accepted intent. The
    # mapping is retained for audit and to reject an intent that cannot support
    # one family without inventing a relation.
    return list(QUESTION_TYPES)


def canonicalize_intents(
    raw: Any, *, expanded_dialogue: Mapping[str, Any], expected_count: int = 3,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(raw, Mapping) or set(raw) != {"intents"}:
        return {"valid": False, "errors": ["intent_top_level_schema"], "intents": []}
    rows = raw.get("intents")
    if not isinstance(rows, list):
        return {"valid": False, "errors": ["intent_list_schema"], "intents": []}
    if len(rows) != expected_count:
        errors.append(f"intent_count:{len(rows)}")
    turns = list(expanded_dialogue.get("turns") or [])
    position_map = {index: row for index, row in enumerate(turns, 1)}
    seen_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        row_errors: list[str] = []
        if not isinstance(row, Mapping):
            result.append({"intent_id": f"I{index:03d}", "valid": False,
                           "errors": ["intent_row_schema"]})
            errors.append(f"intent_{index}:row_schema")
            continue
        intent_id = str(row.get("intent_id") or f"I{index:03d}")
        if intent_id in seen_ids:
            row_errors.append("duplicate_intent_id")
        seen_ids.add(intent_id)
        intent_type = str(row.get("intent_type") or "")
        if intent_type not in INTENT_TYPES:
            row_errors.append("intent_type_invalid")
        variable = row.get("discriminative_variable")
        if not isinstance(variable, str) or not variable.strip():
            row_errors.append("discriminative_variable_missing")
        chain = row.get("reasoning_chain")
        if not isinstance(chain, Mapping):
            row_errors.append("reasoning_chain_schema")
            chain = {}
        canonical_groups: dict[str, list[str]] = {}
        positions_seen: list[int] = []
        for role in ROLE_KEYS:
            values = chain.get(role)
            if not _positions(values):
                row_errors.append(f"{role}_positions_schema")
                values = []
            valid_positions = [x for x in values if x in position_map]
            if len(valid_positions) != len(values):
                row_errors.append(f"{role}_position_out_of_range")
            canonical_groups[role] = [str(position_map[x].get("turn_ref") or "")
                                      for x in valid_positions]
            positions_seen.extend(valid_positions)
        if len(positions_seen) != len(set(positions_seen)):
            row_errors.append("evidence_group_overlap")
        if len(positions_seen) < 3:
            row_errors.append("reasoning_chain_too_small")
        requested = row.get("candidate_question_types")
        if requested is None:
            requested = allowed_question_types(intent_type)
        if (not isinstance(requested, list) or
                set(requested) != set(QUESTION_TYPES)):
            row_errors.append("candidate_question_types_not_phase0_complete")
            requested = allowed_question_types(intent_type)
        item = {
            "intent_id": intent_id,
            "intent_type": intent_type,
            "reasoning_chain": canonical_groups,
            "discriminative_variable": str(variable).strip() if isinstance(variable, str) else "",
            "candidate_question_types": list(QUESTION_TYPES),
            "source_positions": sorted(set(positions_seen)),
            "valid": not row_errors,
            "errors": sorted(set(row_errors)),
        }
        result.append(item)
        errors.extend(f"{intent_id}:{error}" for error in row_errors)
    distribution = Counter(row.get("intent_type") for row in result if row.get("valid"))
    return {
        "valid": not errors and len(result) == expected_count,
        "errors": sorted(set(errors)),
        "intents": result,
        "valid_intent_count": sum(bool(row.get("valid")) for row in result),
        "intent_type_distribution": dict(distribution),
    }

