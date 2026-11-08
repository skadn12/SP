from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def rate(values: Iterable[bool | int | None]) -> float | None:
    rows = [bool(value) for value in values if value is not None]
    return (sum(rows) / len(rows)) if rows else None


def solver_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("parse_status", "").startswith("valid_")]
    correct = [row for row in valid if row.get("local_reward") == 1]
    return {"rollouts": len(rows), "valid_parse": len(valid), "correct": len(correct),
            "accuracy": (len(correct) / len(valid)) if valid else None,
            "invalid_parse": len(rows) - len(valid)}


def contingency(original: list[bool], variant: list[bool]) -> dict[str, int]:
    return {"both_correct": sum(a and b for a, b in zip(original, variant)),
            "original_only": sum(a and not b for a, b in zip(original, variant)),
            "variant_only": sum(not a and b for a, b in zip(original, variant)),
            "both_wrong": sum(not a and not b for a, b in zip(original, variant))}


def counts(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "unknown") for row in rows))

