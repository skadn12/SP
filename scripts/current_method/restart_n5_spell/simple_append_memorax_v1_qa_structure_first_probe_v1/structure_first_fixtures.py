"""Deterministic fixtures for the structure/evidence-first Writer contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .structure_first_contract import STRUCTURE_FIRST_SCHEMA_VERSION


def make_candidate(index: int, *, evidence_count: int = 2) -> dict[str, Any]:
    if not 2 <= evidence_count <= 8:
        raise ValueError("fixture_evidence_count_out_of_range")
    candidate_id = f"Q{index:02d}"
    refs = [f"D{position:03d}" for position in range(1, evidence_count + 1)]
    gold = f"The locally supported cause for case {index}"
    wrong = [
        f"A nearby but different event for case {index}",
        f"An earlier state that did not cause case {index}",
        f"A different actor's action in case {index}",
    ]
    return {
        "candidate_id": candidate_id,
        "question_type": "causal",
        "source_scope": "parent_only",
        "structure_plan": {
            "structure_type": "causal",
            "local_evidence_turn_ids": refs,
            "discriminative_variable": "dialogue_specific_cause",
            "gold_uniqueness": {
                "unique": True,
                "supported_option_texts": [gold],
                "support_turn_ids": [refs[0]],
            },
            "distractor_sources": [
                {
                    "option_text": wrong[0],
                    "source_turn_ids": [refs[-1]],
                    "near_miss_kind": "nearby_event",
                },
                {
                    "option_text": wrong[1],
                    "source_turn_ids": [refs[0]],
                    "near_miss_kind": "previous_state",
                },
                {
                    "option_text": wrong[2],
                    "source_turn_ids": [refs[-1]],
                    "near_miss_kind": "wrong_actor",
                },
            ],
        },
        "reasoning_structure": {
            "discriminative_variable": "dialogue_specific_cause",
            "supporting_turn_ids": refs,
            "short_rationale": (
                "The locked local turns distinguish the stated cause from nearby facts."),
            "type_contract": {
                "dialogue_specific_cause": gold,
                "competing_causes": wrong[:2],
                "turn_ids": refs,
            },
        },
        "question": f"What caused the relevant outcome in local case {index}?",
        "options": {"A": wrong[0], "B": gold, "C": wrong[1], "D": wrong[2]},
        "gold_text": gold,
        "evidence_turn_ids": refs,
    }


def make_envelope(count: int, *, evidence_count: int = 2) -> dict[str, Any]:
    if not 0 <= count <= 12:
        raise ValueError("fixture_question_count_out_of_range")
    return {
        "structure_first_schema_version": STRUCTURE_FIRST_SCHEMA_VERSION,
        "abstention_summary": {
            "estimated_skipped_count": 1 if count == 0 else 0,
            "reason_codes": ["no_discriminative_structure"] if count == 0 else [],
        },
        "questions": [make_candidate(index, evidence_count=evidence_count)
                      for index in range(1, count + 1)],
    }


def ambiguous_gold_envelope() -> dict[str, Any]:
    value = make_envelope(1)
    candidate = value["questions"][0]
    candidate["structure_plan"]["gold_uniqueness"] = {
        "unique": False,
        "supported_option_texts": [
            candidate["gold_text"], candidate["options"]["A"]],
        "support_turn_ids": ["D001", "D002"],
    }
    return value


def weak_distractor_envelope() -> dict[str, Any]:
    value = make_envelope(1)
    candidate = value["questions"][0]
    weak = "They ignored the matter forever."
    candidate["options"]["A"] = weak
    candidate["structure_plan"]["distractor_sources"][0]["option_text"] = weak
    return value


def with_evidence_count(value: dict[str, Any], count: int) -> dict[str, Any]:
    """Rebuild rather than mutate references so fixture intent stays explicit."""
    if len(value.get("questions") or []) != 1:
        raise ValueError("single_question_fixture_required")
    rebuilt = make_envelope(1, evidence_count=count)
    return deepcopy(rebuilt)
