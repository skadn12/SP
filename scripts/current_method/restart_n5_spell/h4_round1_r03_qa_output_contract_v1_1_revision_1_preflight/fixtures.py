"""Deterministic fixture set required by the v1.1 revision-1 contract."""

from __future__ import annotations

from typing import Any


KNOWN_REFS = {"P01:T001", "R03:S01:T001"}


def options(qtype: str, count: int) -> dict[str, str]:
    if qtype == "ordering":
        return {f"I{index + 1}": f"item-{index + 1}" for index in range(count)}
    return {chr(ord("A") + index): f"option-{chr(ord('A') + index)}" for index in range(count)}


def multipass(qtype: str, count: int, gold: list[str], *, pass_name: str = "fresh", no_safe: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outcome": "NO_SAFE_EVOLUTION" if no_safe else "QA",
        "question_type": None if no_safe else qtype,
        "new_question": None if no_safe else "Which result is supported?",
        "options_or_items": None if no_safe else options(qtype, count),
        "canonical_gold": None if no_safe else gold,
        "evidence_refs": [] if no_safe else ["R03:S01:T001"],
        "selected_evidence_relations": [] if no_safe else (["relation-a", "relation-b"] if pass_name == "composition" else ["relation-a"]),
        "expected_context_dependency": None if no_safe else "The answer uses the supplied dialogue.",
        "r03_semantic_contribution_claimed": None if no_safe else True,
        "answerable_without_r03_claim": None if no_safe else False,
    }
    if no_safe:
        row["no_safe_evolution_reason"] = "No safe successor is supported."
    if pass_name == "composition":
        row.update({"source_cluster_ids": ["cluster-a", "cluster-b"], "composition_relation": None if no_safe else "joint relation"})
    else:
        row.update({"independent_new_qa_reason": None if no_safe else "It targets a new relation.", "novelty_basis": None if no_safe else "R03 state change."})
    return row


def targeted(qtype: str | None, count: int = 4, gold: list[str] | None = None, *, no_safe: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_question_id": "question-1", "source_cluster_id": "cluster-1", "root_id": "root-1",
        "selected_strategy": "NO_SAFE_EVOLUTION" if no_safe else "ENTITY_RELATION_BINDING",
        "strategy_rationale": "The selected strategy is supported by the observed dialogue distinction.",
        "feedback_observations_used": ["M0"], "tentative_hypothesis": "Hypothesis only.",
        "question_type": None if no_safe else qtype,
        "new_question": None if no_safe else "Which result is supported?",
        "options_or_items": None if no_safe else options(str(qtype), count),
        "canonical_gold": None if no_safe else list(gold or ["A"]),
        "evidence_refs": [] if no_safe else ["R03:S01:T001"],
        "source_question_relation": "" if no_safe else "The answer target changes.",
        "expected_context_dependency": "" if no_safe else "The answer uses the supplied dialogue.",
    }
    if no_safe:
        row["no_safe_evolution_reason"] = "No unambiguous successor is supported."
    return row

