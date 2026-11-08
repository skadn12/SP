"""Deterministic hard-condition validator for one proposal/outcome pair."""

from __future__ import annotations

import re
from typing import Any

from .contracts import Fact, VALID_ROLES


def _is_broad_event(event_key: str) -> bool:
    return event_key in {"topic:environment", "topic:charity", "topic:event", "person:shared"}


def validate_relation(proposal: Fact, outcome: Fact) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if proposal.root_id != outcome.root_id:
        errors.append("different_root")
    if proposal.relation_role != "proposal" or proposal.relation_role not in VALID_ROLES:
        errors.append("proposal_role_invalid")
    if outcome.relation_role != "outcome" or outcome.relation_role not in VALID_ROLES:
        errors.append("outcome_role_invalid")
    if proposal.turn_index >= outcome.turn_index:
        errors.append("proposal_not_before_outcome")
    if proposal.turn_ref == outcome.turn_ref:
        errors.append("same_turn_evidence")
    if not proposal.event_key or not outcome.event_key:
        errors.append("event_key_empty")
    if proposal.event_key != outcome.event_key:
        errors.append("event_slot_mismatch")
    if _is_broad_event(proposal.event_key):
        errors.append("broad_topic_only")
    if not proposal.evidence_text.strip() or not outcome.evidence_text.strip():
        errors.append("direct_evidence_missing")
    if not proposal.value.strip() or not outcome.value.strip():
        errors.append("value_missing")
    if proposal.turn_ref == outcome.turn_ref:
        errors.append("evidence_turns_not_distinct")
    distance = outcome.turn_index - proposal.turn_index
    if distance <= 2:
        warnings.append("local_adjacent_relation")
    if proposal.value == outcome.value:
        value_relation = "same"
    elif proposal.value and outcome.value:
        value_relation = "changed"
    else:
        value_relation = "unknown"
    if proposal.evidence_text.casefold() == outcome.evidence_text.casefold():
        errors.append("same_sentence_paraphrase")
    if not re.search(r"\b(?:agree|decid|confirm|final|settle|chosen|going to|include|start)\w*\b", outcome.evidence_text, re.I):
        if outcome.strength == "medium":
            warnings.append("outcome_is_weak_interest_only")
        else:
            errors.append("outcome_not_explicit")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "value_relation": value_relation,
        "turn_distance": distance,
    }

