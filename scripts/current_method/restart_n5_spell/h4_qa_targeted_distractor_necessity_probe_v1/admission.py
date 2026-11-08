"""Strict QA difficulty admission for the isolated targeted probe."""

from __future__ import annotations

from typing import Any, Mapping


def strict_necessity_admission(solver_eval: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Require context dependence and necessity of every declared role."""
    grounding = solver_eval.get("grounding") or {}
    full = solver_eval.get("full_context") or {}
    ablations = solver_eval.get("ablations") or {}
    no_context_accuracy = grounding.get("semantic_correct_rate")
    ablation_accuracy = {
        name: (ablations.get(name) or {}).get("accuracy")
        for name in ("remove_fact_A", "remove_trigger", "remove_fact_B")
    }
    full_pass = full.get("correct") == 5 and full.get("total") == 5
    no_context_fail = no_context_accuracy is None or float(no_context_accuracy) < 1.0
    role_failures = {
        name: value is None or float(value) < 1.0
        for name, value in ablation_accuracy.items()
    }
    passed = bool(full_pass and no_context_fail and all(role_failures.values()))
    return passed, {
        "full_context_pass": full_pass,
        "no_context_accuracy": no_context_accuracy,
        "no_context_fail": no_context_fail,
        "ablation_accuracy": ablation_accuracy,
        "role_failures": role_failures,
        "rule": "full K5 pass AND no-context not all-correct AND every role ablation not all-correct",
    }
