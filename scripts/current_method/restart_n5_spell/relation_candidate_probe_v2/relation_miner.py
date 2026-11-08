"""Mine exactly one relation type from rule-extracted transcript facts."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

from .contracts import Fact, Rejection, RelationCandidate
from .validator import validate_relation


def _fact_payload(fact: Fact) -> dict[str, Any]:
    return {
        "actor": fact.actor,
        "actors": list(fact.actors),
        "value": fact.value,
        "turn_refs": [fact.turn_ref],
        "turn_indices": [fact.turn_index],
        "evidence_texts": [fact.evidence_text],
        "speaker": fact.speaker,
        "origin": fact.origin,
        "strength": fact.strength,
        "fact_id": fact.fact_id,
    }


def _scope(proposal: Fact, outcome: Fact) -> tuple[str, bool]:
    origins = {proposal.origin, outcome.origin}
    if origins == {"source_grounded"}:
        return "parent_only", False
    if origins == {"generated_extension"}:
        return "child_only", False
    return "parent_child", True


def _confidence(validation: dict[str, Any], proposal: Fact, outcome: Fact) -> str:
    if not validation["valid"]:
        return "uncertain"
    if outcome.strength == "medium" or validation["warnings"]:
        return "medium"
    return "high"


def mine_relations(
    proposals: Iterable[Fact], outcomes: Iterable[Fact],
) -> tuple[list[RelationCandidate], list[Rejection]]:
    proposals = list(proposals)
    outcomes = list(outcomes)
    candidates: list[RelationCandidate] = []
    rejections: list[Rejection] = []
    relation_counter = 0
    pair_counter = 0
    for proposal, outcome in product(proposals, outcomes):
        if proposal.root_id != outcome.root_id:
            pair_counter += 1
            rejections.append(Rejection(f"PAIR{pair_counter:05d}", proposal.root_id, proposal.fact_id, outcome.fact_id, "different_root"))
            continue
        if proposal.event_key != outcome.event_key:
            pair_counter += 1
            rejections.append(Rejection(f"PAIR{pair_counter:05d}", proposal.root_id, proposal.fact_id, outcome.fact_id, "event_slot_mismatch", f"{proposal.event_key}!={outcome.event_key}"))
            continue
        pair_counter += 1
        validation = validate_relation(proposal, outcome)
        if not validation["valid"]:
            status = "uncertain" if validation["warnings"] and not validation["errors"] else "rejected"
            reason = ";".join(validation["errors"] or validation["warnings"])
            rejections.append(Rejection(f"PAIR{pair_counter:05d}", proposal.root_id, proposal.fact_id, outcome.fact_id, reason))
            continue
        relation_counter += 1
        scope, cross = _scope(proposal, outcome)
        confidence = _confidence(validation, proposal, outcome)
        status = "eligible" if confidence == "high" else "uncertain"
        source_refs = list(dict.fromkeys(proposal.source_structured_refs + outcome.source_structured_refs))
        candidates.append(RelationCandidate(
            relation_id=f"REL{relation_counter:04d}", root_id=proposal.root_id,
            relation_type="proposal_outcome", event_key=proposal.event_key,
            event_summary=proposal.event_summary, proposal=_fact_payload(proposal),
            outcome=_fact_payload(outcome), value_relation=validation["value_relation"],
            turn_distance=validation["turn_distance"], source_structured_refs=source_refs,
            confidence=confidence,
            reason=("explicit proposal followed by explicit adoption/decision"
                    if status == "eligible" else "proposal followed by weak or locally adjacent outcome"),
            status=status, cross_parent_child=cross, scope=scope,
            validation_errors=list(validation["errors"]),
        ))
    return candidates, rejections

