from __future__ import annotations

from typing import Any, Mapping


MINIMUMS = {"single_choice": 2, "multi_select": 1, "ordering": 1}


def build_reasoning_contract(intent: Mapping[str, Any]) -> dict[str, Any]:
    chain = dict(intent.get("reasoning_chain") or {})
    return {
        "contract_version": "h4_reasoning_contract_v1",
        "intent_id": str(intent["intent_id"]),
        "intent_type": str(intent["intent_type"]),
        "discriminative_variable": str(intent["discriminative_variable"]),
        "required_evidence_groups": {
            "before_state": list(chain.get("before_state") or []),
            "trigger": list(chain.get("trigger") or []),
            "after_state": list(chain.get("after_state") or []),
        },
        "minimum_candidate_counts": dict(MINIMUMS),
        "question_types": ["single_choice", "multi_select", "ordering"],
    }


def contract_has_three_disjoint_groups(contract: Mapping[str, Any]) -> bool:
    groups = list((contract.get("required_evidence_groups") or {}).values())
    flat = [ref for group in groups for ref in group]
    return bool(groups and all(group for group in groups) and len(flat) == len(set(flat)))

