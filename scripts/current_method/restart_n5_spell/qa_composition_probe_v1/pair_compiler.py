"""Deterministic compiler for exactly-two-atomic-QA composition candidates."""

from __future__ import annotations

from itertools import combinations
import re
from typing import Any, Iterable

from .contracts import (
    AtomicQA,
    COMPOSITION_TYPES,
    is_duplicate_fact,
    lexical_tokens,
    shared_nontrivial_topics,
)


_PROPOSAL_CUES = re.compile(
    r"\b(suggest(?:ed)?|propos(?:e|ed|al)|idea|plan(?:ned)?|thinking|recommend)\b",
    re.IGNORECASE,
)
_OUTCOME_CUES = re.compile(
    r"\b(decid(?:e|ed|es|ing)|agreed|chosen|final|outcome|integrat|settled)\b",
    re.IGNORECASE,
)
_ACTOR_CUES = re.compile(
    r"\b(who|whose)\b.*\b(prefer|favou?r|want|suggest)",
    re.IGNORECASE,
)
_PREFERENCE_CUES = re.compile(r"\b(prefer|favou?r|want)\w*\b", re.I)
_GENERIC_TOPICS = {"charity", "support"}
_GENERIC_DIMENSIONS = {"state", "goal", "decision"}


def _all_text(qa: AtomicQA) -> str:
    return " ".join((qa.question, qa.gold_text, *qa.evidence_texts))


def _answer_text(qa: AtomicQA) -> str:
    return " ".join((qa.question, qa.gold_text))


def _stage_cues(qa: AtomicQA) -> set[str]:
    text = _all_text(qa)
    cues: set[str] = set()
    if re.search(r"\b(original|earlier|initial|previous|before|from)\b", text, re.I):
        cues.add("earlier")
    if re.search(r"\b(final|finally|later|current|after|ultimately|to)\b", text, re.I):
        cues.add("later")
    if qa.question_type in {"state_change", "information_revision"}:
        cues.add("change")
    return cues


def _has_distinct_information(left: AtomicQA, right: AtomicQA) -> bool:
    if set(left.evidence_turn_refs) == set(right.evidence_turn_refs):
        return False
    if not left.evidence_texts or not right.evidence_texts:
        return False
    return bool(set(left.evidence_turn_refs) ^ set(right.evidence_turn_refs))


def _evidence_distance(left: AtomicQA, right: AtomicQA) -> int | None:
    positions = list(left.turn_indices) + list(right.turn_indices)
    return max(positions) - min(positions) if positions else None


def _same_root(left: AtomicQA, right: AtomicQA) -> bool:
    return bool(left.root_id and left.root_id == right.root_id)


def _shared_entities(left: AtomicQA, right: AtomicQA) -> list[str]:
    return sorted(set(left.entities) & set(right.entities))


def _state_change_type(left: AtomicQA, right: AtomicQA, topics: list[str]) -> bool:
    left_dims = set(left.state_dimensions)
    right_dims = set(right.state_dimensions)
    shared_dims = left_dims & right_dims
    specific_dims = shared_dims - _GENERIC_DIMENSIONS
    if not specific_dims or not (set(topics) - _GENERIC_TOPICS):
        return False
    if not (set(left.entities) & set(right.entities)):
        return False
    state_types = {"state_change", "information_revision"}
    if not ({left.question_type, right.question_type} & state_types):
        return False
    if left.question_type not in state_types or right.question_type not in state_types:
        state_qa = left if left.question_type in state_types else right
        other_qa = right if state_qa is left else left
        has_transition = bool(re.search(
            r"\bfrom\b.+\bto\b", state_qa.gold_text, re.IGNORECASE))
        ordered_evidence = bool(
            state_qa.turn_indices and other_qa.turn_indices and
            (max(state_qa.turn_indices) < min(other_qa.turn_indices) or
             max(other_qa.turn_indices) < min(state_qa.turn_indices)))
        if "perspective" not in specific_dims or not has_transition or not ordered_evidence:
            return False
    if left.gold_tokens == right.gold_tokens:
        return False
    stages = _stage_cues(left) | _stage_cues(right)
    change_like = ({"change", "earlier", "later"} & stages)
    return bool(change_like)


def _proposal_outcome_type(left: AtomicQA, right: AtomicQA, topics: list[str]) -> bool:
    if not (set(topics) - _GENERIC_TOPICS):
        return False
    left_text, right_text = _answer_text(left), _answer_text(right)

    def is_proposal(qa: AtomicQA, text: str) -> bool:
        return bool(_PROPOSAL_CUES.search(text))

    def is_outcome(qa: AtomicQA, text: str) -> bool:
        return bool(_OUTCOME_CUES.search(text))

    return bool(
        (is_proposal(left, left_text) and not is_outcome(left, left_text) and
         is_outcome(right, right_text)) or
        (is_proposal(right, right_text) and not is_outcome(right, right_text) and
         is_outcome(left, left_text))
    )


def _actor_outcome_type(left: AtomicQA, right: AtomicQA, topics: list[str]) -> bool:
    if not (set(topics) - _GENERIC_TOPICS):
        return False
    left_text, right_text = _all_text(left), _all_text(right)
    actor_left = bool(_ACTOR_CUES.search(left.question)) and bool(_PREFERENCE_CUES.search(left_text))
    actor_right = bool(_ACTOR_CUES.search(right.question)) and bool(_PREFERENCE_CUES.search(right_text))
    outcome_left = bool(_OUTCOME_CUES.search(left_text))
    outcome_right = bool(_OUTCOME_CUES.search(right_text))
    if not ((actor_left and outcome_right) or (actor_right and outcome_left)):
        return False
    # Require an actor token to occur in the outcome evidence too.  This keeps
    # a role question from being paired with an unrelated event decision.
    actor_names = set(left.entities) | set(right.entities)
    outcome_qa = right if actor_left and outcome_right else left
    actor_qa = left if actor_left and outcome_right else right
    value_overlap = lexical_tokens(actor_qa.question) & outcome_qa.gold_tokens
    return bool(value_overlap) and (bool(actor_names) or bool(topics))


def normalize_for_match(value: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(value).casefold())


def _classify(left: AtomicQA, right: AtomicQA, topics: list[str]) -> tuple[str, str]:
    if _state_change_type(left, right, topics):
        return "state_change", "shared state dimension and topic with distinct stages or values"
    if _actor_outcome_type(left, right, topics):
        return "actor_outcome_binding", "an actor preference and a matching outcome share the same value"
    if _proposal_outcome_type(left, right, topics):
        return "proposal_vs_outcome", "one QA supplies a proposal and the other a later outcome"
    return "uncertain", "shared topic exists, but no closed composition type is reliable"


def _rejection(left: AtomicQA, right: AtomicQA, reason: str) -> dict[str, Any]:
    return {
        "pair_id": None,
        "qa_ids": [left.qa_id, right.qa_id],
        "reason": reason,
    }


def compile_pairs(atomic_qas: Iterable[AtomicQA]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return stable candidate and rejection records for exactly-two pairs."""
    ordered = sorted(list(atomic_qas), key=lambda qa: (qa.root_id, qa.qa_id))
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    pair_index = 0
    for left, right in combinations(ordered, 2):
        if not _same_root(left, right):
            rejections.append(_rejection(left, right, "different_root"))
            continue
        if left.qa_id == right.qa_id:
            rejections.append(_rejection(left, right, "same_qa_id"))
            continue
        if set(left.evidence_turn_refs) == set(right.evidence_turn_refs):
            rejections.append(_rejection(left, right, "same_evidence"))
            continue
        if is_duplicate_fact(left, right):
            rejections.append(_rejection(left, right, "duplicate_fact"))
            continue
        if not _has_distinct_information(left, right):
            rejections.append(_rejection(left, right, "no_information_gain"))
            continue
        shared_entities = _shared_entities(left, right)
        topics = shared_nontrivial_topics(left, right)
        shared_dimensions = sorted(set(left.state_dimensions) &
                                   set(right.state_dimensions))
        if not (shared_entities or topics or shared_dimensions):
            rejections.append(_rejection(left, right, "unrelated_pair"))
            continue
        # A/B appears in almost every dialogue question.  Require at least a
        # topical bridge when entities are the only shared feature.
        if shared_entities and not topics and not shared_dimensions:
            rejections.append(_rejection(left, right, "unrelated_generic_entity_only"))
            continue
        strong_topics = set(topics) - _GENERIC_TOPICS
        strong_dimensions = set(shared_dimensions) - _GENERIC_DIMENSIONS
        if not strong_topics and not strong_dimensions:
            rejections.append(_rejection(left, right, "unrelated_broad_theme_only"))
            continue
        pair_index += 1
        composition_type, reason = _classify(left, right, topics)
        evidence_distance = _evidence_distance(left, right)
        candidates.append({
            "pair_id": f"PAIR{pair_index:04d}",
            "qa_ids": [left.qa_id, right.qa_id],
            "candidate_type": composition_type,
            "shared_entities": shared_entities,
            "shared_event_terms": topics,
            "shared_state_dimensions": shared_dimensions,
            "evidence_distance": evidence_distance,
            "pair_reason": reason,
            "evidence_turn_refs": sorted(set(left.evidence_turn_refs + right.evidence_turn_refs)),
            "composer_ready": composition_type in {"state_change", "proposal_vs_outcome", "actor_outcome_binding"},
        })
    return candidates, rejections
