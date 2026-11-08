"""Deterministic paired-atomic-QA target plans, without natural-language QA."""

from __future__ import annotations

from typing import Any, Mapping


def build_paired_qa_target(relation: Mapping[str, Any]) -> dict[str, Any]:
    if relation.get("status") != "eligible":
        raise ValueError("paired_targets_require_eligible_relation")
    proposal = relation["proposal"]
    outcome = relation["outcome"]
    value_relation = str(relation["value_relation"])
    if value_relation == "same":
        intent = "identify whether the proposal became the final decision"
    elif value_relation == "changed":
        intent = "compare the original proposal with the final decision"
    else:
        raise ValueError("unknown_value_relation")
    return {
        "relation_id": relation["relation_id"],
        "relation_type": "proposal_outcome",
        "atomic_targets": [
            {
                "target_id": f"{relation['relation_id']}_A",
                "target_role": "proposal",
                "required_turn_refs": list(proposal["turn_refs"]),
                "target_answer": proposal["value"],
                "question_intent": "identify the value proposed by the actor",
            },
            {
                "target_id": f"{relation['relation_id']}_B",
                "target_role": "outcome",
                "required_turn_refs": list(outcome["turn_refs"]),
                "target_answer": outcome["value"],
                "question_intent": "identify the later agreed or adopted value",
            },
        ],
        "composite_intent": intent,
        "value_relation": value_relation,
        "required_evidence_groups": [
            {"source_role": "proposal", "turn_refs": list(proposal["turn_refs"])},
            {"source_role": "outcome", "turn_refs": list(outcome["turn_refs"])},
        ],
    }

