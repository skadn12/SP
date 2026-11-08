"""Frozen atomic pair selection for the relation-guided gate condition."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.contracts import (
    AtomicQA, is_duplicate_fact,
)


CORE_GROUPS = ("fact_A", "trigger", "fact_B")


def load_atomic_ir(path: Path) -> list[AtomicQA]:
    rows: list[AtomicQA] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("record_type") != "eligible":
            continue
        fields = {
            key: value.get(key)
            for key in (
                "qa_id", "root_id", "question", "gold_text", "correct_option",
                "options", "evidence_turn_refs", "evidence_texts", "source_scope",
                "entities", "event_terms", "turn_indices", "question_type",
                "grounding_decision", "source_path",
            )
        }
        rows.append(AtomicQA(**fields))
    return rows


def _group_hits(qa: AtomicQA, relation_source_turns: Mapping[str, Any]) -> list[str]:
    refs = set(qa.evidence_turn_refs)
    return sorted(
        key for key, values in relation_source_turns.items()
        if refs & {str(value) for value in values}
    )


def relation_hits(atomic: list[AtomicQA], relation_source_turns: Mapping[str, Any]) -> dict[str, list[str]]:
    return {qa.qa_id: _group_hits(qa, relation_source_turns) for qa in atomic}


def _pair_score(left: AtomicQA, right: AtomicQA, hits: Mapping[str, list[str]]) -> tuple[int, ...]:
    left_hits, right_hits = set(hits[left.qa_id]), set(hits[right.qa_id])
    union = left_hits | right_hits
    distinct_gold = int(left.gold_text.casefold() != right.gold_text.casefold())
    causal_revision_shape = int({left.question_type, right.question_type} == {
        "causal", "information_revision"})
    complementary_types = int(bool({left.question_type, right.question_type} & {
        "causal", "task_dependency_or_constraint"}) and bool(
        {left.question_type, right.question_type} & {"information_revision", "state_change"}))
    unique_group_count = len((left_hits ^ right_hits) & set(CORE_GROUPS))
    evidence_distance = max(left.turn_indices + right.turn_indices) - min(
        left.turn_indices + right.turn_indices)
    return (
        int(set(CORE_GROUPS).issubset(union)),
        causal_revision_shape,
        complementary_types,
        distinct_gold,
        unique_group_count,
        -len(set(left.evidence_turn_refs) & set(right.evidence_turn_refs)),
        -evidence_distance,
        -min(int(left.qa_id.rsplit("-", 1)[-1]), int(right.qa_id.rsplit("-", 1)[-1])),
    )


def build_current_summary(current_composition: Path) -> dict[str, Any]:
    def lines(name: str) -> list[dict[str, Any]]:
        path = current_composition / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []

    candidates = lines("pair_candidates.jsonl")
    requests = lines("composer_requests.jsonl")
    return {
        "current_pair_candidates": len(candidates),
        "current_composer_ready": len(requests),
        "current_relation_classification": {
            key: sum(row.get("candidate_type") == key for row in candidates)
            for key in ("state_change", "proposal_vs_outcome", "actor_outcome_binding", "uncertain")
        },
        "current_rejected_reason_counts": _counts(lines("pair_rejections.jsonl"), "reason"),
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def build_relation_guided_candidates(
    atomic: list[AtomicQA], *, relation_audit: Mapping[str, Any], relation_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_turns = relation_audit.get("relation_source_turns") or {}
    hits = relation_hits(atomic, source_turns)
    ranked: list[tuple[tuple[int, ...], AtomicQA, AtomicQA]] = []
    for left, right in combinations(sorted(atomic, key=lambda qa: qa.qa_id), 2):
        if left.root_id != right.root_id or is_duplicate_fact(left, right):
            continue
        left_hits, right_hits = set(hits[left.qa_id]), set(hits[right.qa_id])
        if not set(CORE_GROUPS).issubset(left_hits | right_hits):
            continue
        if not (left_hits & set(CORE_GROUPS)) or not (right_hits & set(CORE_GROUPS)):
            continue
        # A pair is necessity-eligible only when both atomics contribute a
        # unique core relation role. If one atomic already covers A+trigger+B,
        # the second atomic is redundant for any later ablation.
        left_unique = (left_hits - right_hits) & set(CORE_GROUPS)
        right_unique = (right_hits - left_hits) & set(CORE_GROUPS)
        if not left_unique or not right_unique:
            continue
        ranked.append((_pair_score(left, right, hits), left, right))
    ranked.sort(key=lambda row: row[0], reverse=True)
    result: list[dict[str, Any]] = []
    for index, (score, left, right) in enumerate(ranked[:3], 1):
        union = sorted(set(hits[left.qa_id]) | set(hits[right.qa_id]))
        result.append({
            "pair_id": f"RGPAIR{index:04d}",
            "qa_ids": [left.qa_id, right.qa_id],
            "candidate_type": "state_change",
            "relation_type": relation_contract.get("relation_type"),
            "relation_id": relation_contract.get("relation_id"),
            "relation_groups_covered": union,
            "qa_relation_groups": {
                left.qa_id: hits[left.qa_id], right.qa_id: hits[right.qa_id],
            },
            "relation_complete": set(CORE_GROUPS).issubset(union),
            "unique_relation_groups": {
                left.qa_id: sorted((set(hits[left.qa_id]) - set(hits[right.qa_id])) & set(CORE_GROUPS)),
                right.qa_id: sorted((set(hits[right.qa_id]) - set(hits[left.qa_id])) & set(CORE_GROUPS)),
            },
            "gate_score": list(score),
            "selection_reason": "relation union is complete and each atomic contributes a unique core relation group",
            "composer_ready": True,
        })
    return result
