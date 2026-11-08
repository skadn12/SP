"""Static audit for complete relation slots and near-neighbor options."""

from __future__ import annotations

import re
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.contracts import OPTION_KEYS
from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.static_audit import audit_composite


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _hit(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def audit_targeted(
    composite: Mapping[str, Any], *, atomic_by_id: Mapping[str, Any],
    relation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    base = audit_composite(composite, atomic_by_id=atomic_by_id)
    # The generic audit treats every "and what" as a mechanical conjunction.
    # A causal slot question deliberately asks for two linked attributes, so
    # accept this one pattern only when all three relation slots are explicit.
    errors = [error for error in base["errors"] if error != "mechanical_conjunction"]
    question = _text(composite.get("question"))
    options = composite.get("options") or {}
    gold = _text(composite.get("gold_text"))
    fact_a = " ".join(str(value) for value in (
        (relation_contract.get("answer_components") or {}).get("fact_A") or []))
    trigger = " ".join(str(value) for value in (
        (relation_contract.get("answer_components") or {}).get("trigger") or []))
    fact_b = " ".join(str(value) for value in (
        (relation_contract.get("answer_components") or {}).get("fact_B") or []))
    slot_patterns = {
        "fact_A": (r"two parts honey", r"2:1", r"one part mustard",
                   r"honey-to-mustard ratio"),
        "trigger": (r"too thick", r"spread evenly", r"coat(?:ing)? the ham"),
        "fact_B": (r"add(?:ed|ition)? .*water", r"water .*thin", r"thin(?:ned)?"),
    }
    slot_hits = {
        key: _hit(gold, patterns) for key, patterns in slot_patterns.items()
    }
    if not all(slot_hits.values()):
        errors.append("gold_missing_relation_slot")
    question_slot_hits = {
        "fact_A": _hit(question, (r"original", r"initial", r"ratio", r"formulation")),
        "trigger": _hit(question, (r"problem", r"prompted", r"why", r"reason", r"issue")),
        "fact_B": _hit(question, (r"added", r"revision", r"revised", r"change")),
    }
    if not all(question_slot_hits.values()):
        errors.append("question_does_not_request_all_relation_slots")
    # Distractors must preserve the tuple shape, but each is expected to
    # mutate one factual slot. Check structural labels here, not the gold
    # value of every slot.
    option_structure_patterns = {
        "fact_A": (r"original", r"initial", r"ratio", r"formulation"),
        "trigger": (r"problem", r"prompted", r"reason", r"issue"),
        "fact_B": (r"revision", r"add(?:ing|ed)?", r"change", r"action"),
    }
    option_slot_hits = {
        key: {
            slot: _hit(_text(options.get(key)), patterns)
            for slot, patterns in option_structure_patterns.items()
        }
        for key in OPTION_KEYS
    }
    if any(not all(slots.values()) for slots in option_slot_hits.values()):
        errors.append("distractor_missing_relation_slot")
    dimension = str(composite.get("answer_dimension") or "")
    if dimension != "causal_revision_tuple":
        errors.append("targeted_answer_dimension_mismatch")
    # These values are recorded to make the audit explainable; the contract
    # text itself is not used to repair a malformed Composer response.
    return {
        "valid": not errors,
        "errors": errors,
        "base_static_audit": base,
        "relation_slot_values": {
            "fact_A": fact_a, "trigger": trigger, "fact_B": fact_b,
        },
        "gold_slot_hits": slot_hits,
        "question_slot_hits": question_slot_hits,
        "option_slot_hits": option_slot_hits,
    }
