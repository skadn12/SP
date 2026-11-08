"""Auditable formatting repair for exact source turn references."""

from __future__ import annotations


SCALAR_REF_FIELDS = ("proposal_ref", "acceptance_ref")
LIST_REF_FIELDS = ("plan_refs", "outcome_refs")


def normalize_ref(value: object, valid_refs: set[str]) -> tuple[object, bool]:
    if not isinstance(value, str) or not (value.startswith("[") and value.endswith("]")):
        return value, False
    inner = value[1:-1]
    if not inner or "[" in inner or "]" in inner or inner not in valid_refs:
        return value, False
    return inner, True


def normalize_event(
    event: dict, valid_refs: set[str], event_index: int
) -> tuple[dict, list[dict]]:
    normalized = dict(event)
    changes = []

    def replace(value: object, field: str) -> object:
        result, changed = normalize_ref(value, valid_refs)
        if changed:
            changes.append({
                "event_index": event_index,
                "field": field,
                "original": value,
                "normalized": result,
            })
        return result

    for field in SCALAR_REF_FIELDS:
        if field in event:
            normalized[field] = replace(event[field], field)
    for field in LIST_REF_FIELDS:
        if isinstance(event.get(field), list):
            normalized[field] = [
                replace(value, f"{field}[{index}]")
                for index, value in enumerate(event[field])
            ]
    return normalized, changes
