from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


QUESTION_TYPES = {"single_choice", "multi_select", "ordering"}


def _string_list(value: Any, minimum: int = 1) -> bool:
    return isinstance(value, list) and len(value) >= minimum and all(
        isinstance(item, str) and item.strip() for item in value
    ) and len(set(value)) == len(value)


def _options(value: Any, keys: list[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == set(keys) and all(
        isinstance(value[key], str) and value[key].strip() for key in keys
    )


def _gold_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item) for item in value or []]


def normalize_candidates(
    raw: Any, *, intent_id: str, contract: Mapping[str, Any], expanded_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    rows = raw.get("candidates") if isinstance(raw, Mapping) else None
    if not isinstance(rows, list):
        return {"valid": False, "errors": ["candidate_top_level_schema"], "candidates": []}
    turn_refs = {str(row.get("turn_ref")) for row in expanded_dialogue.get("turns") or []}
    contract_groups = {
        str(key): {str(ref) for ref in value}
        for key, value in (contract.get("required_evidence_groups") or {}).items()
        if isinstance(value, list)
    }
    normalized: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        row_errors: list[str] = []
        row = row if isinstance(row, Mapping) else {}
        qtype = str(row.get("question_type") or "")
        counts[qtype] += 1
        if qtype not in QUESTION_TYPES:
            row_errors.append("question_type_invalid")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            row_errors.append("question_missing")
        if qtype == "single_choice":
            if not _options(row.get("options"), list("ABCD")):
                row_errors.append("single_options_invalid")
            correct = str(row.get("correct_option") or "")
            if correct not in "ABCD":
                row_errors.append("correct_option_invalid")
            gold = [correct] if correct in "ABCD" else []
        elif qtype == "multi_select":
            if not _options(row.get("options"), list("ABCDE")):
                row_errors.append("multi_options_invalid")
            gold = _gold_list(row.get("gold"))
            if not gold or any(x not in "ABCDE" for x in gold) or len(set(gold)) != len(gold):
                row_errors.append("multi_gold_invalid")
        else:
            # The first isolated model replay used ``options`` for ordering
            # items.  Accept that harmless alias, while materializing the
            # canonical ``items`` field below.
            items = row.get("items") or row.get("options")
            if not _options(items, ["I1", "I2", "I3", "I4"]):
                row_errors.append("ordering_items_invalid")
            gold = _gold_list(row.get("gold"))
            if len(gold) != 4 or set(gold) != {"I1", "I2", "I3", "I4"}:
                row_errors.append("ordering_gold_invalid")
        evidence_refs = [str(x) for x in row.get("evidence_refs") or []]
        if not _string_list(evidence_refs):
            row_errors.append("evidence_refs_invalid")
        if any(ref not in turn_refs for ref in evidence_refs):
            row_errors.append("evidence_ref_unknown")
        groups_raw = row.get("evidence_groups")
        groups = {
            str(key): [str(x) for x in value]
            for key, value in (groups_raw.items() if isinstance(groups_raw, Mapping) else [])
            if isinstance(value, list)
        }
        if not groups or any(ref not in turn_refs for values in groups.values() for ref in values):
            row_errors.append("evidence_groups_outside_contract")
        alignment = {
            key: bool(set(values) & contract_groups.get(key, set()))
            for key, values in groups.items()
        }
        required = [str(x) for x in row.get("required_reasoning") or []]
        if not _string_list(required):
            row_errors.append("required_reasoning_invalid")
        variable = str(row.get("discriminative_variable") or "").strip()
        if not variable:
            row_errors.append("candidate_variable_missing")
        normalized.append({
            "candidate_id": f"{intent_id}-Q{index:02d}",
            "intent_id": intent_id,
            "question_type": qtype,
            "question": str(question).strip() if isinstance(question, str) else "",
            "options": dict(row.get("options") or {}) if qtype != "ordering" else {},
            "items": dict((row.get("items") or row.get("options") or {})) if qtype == "ordering" else {},
            "correct_option": str(row.get("correct_option") or "") if qtype == "single_choice" else None,
            "gold": gold,
            "evidence_refs": evidence_refs,
            "evidence_groups": groups,
            "contract_group_alignment": alignment,
            "required_reasoning": required,
            "discriminative_variable": variable,
            "distractor_strategy": str(row.get("distractor_strategy") or ""),
            "schema_errors": sorted(set(row_errors)),
            "schema_valid": not row_errors,
        })
    expected = {"single_choice": 2, "multi_select": 1, "ordering": 1}
    for qtype, minimum in expected.items():
        if counts[qtype] < minimum:
            errors.append(f"candidate_count_below_minimum:{qtype}:{counts[qtype]}")
    return {
        "valid": not errors and all(row["schema_valid"] for row in normalized),
        "errors": sorted(set(errors)),
        "candidates": normalized,
        "type_counts": dict(counts),
    }
