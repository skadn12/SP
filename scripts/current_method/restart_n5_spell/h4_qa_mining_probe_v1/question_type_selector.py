from __future__ import annotations

from typing import Any, Mapping


PHASE0_TYPES = ("single_choice", "multi_select", "ordering")


def select_question_types(intent: Mapping[str, Any]) -> dict[str, Any]:
    requested = list(intent.get("candidate_question_types") or [])
    selected = [name for name in PHASE0_TYPES if name in requested]
    return {
        "intent_id": str(intent["intent_id"]),
        "selected_question_types": selected,
        "phase0_complete": set(selected) == set(PHASE0_TYPES),
        "selection_mode": "deterministic_contract_gate",
    }

