"""Static checks for future Composer responses and mechanical conjunctions."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .contracts import OPTION_KEYS, AtomicQA, lexical_tokens


_MECHANICAL_PATTERNS = (
    re.compile(r"\band what\b", re.I),
    re.compile(r"\band where\b", re.I),
    re.compile(r"\band when\b", re.I),
    re.compile(r"\bwhat did .+ and what did\b", re.I),
    re.compile(r"\bwhat was .+ and what was\b", re.I),
    re.compile(r"\bwhat was the answer to\b", re.I),
)


def is_mechanical_conjunction(question: Any) -> bool:
    text = str(question or "").strip()
    return any(pattern.search(text) for pattern in _MECHANICAL_PATTERNS)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _similarity(left: Any, right: Any) -> float:
    left_tokens, right_tokens = lexical_tokens(left), lexical_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _option_dimension(value: Any) -> str:
    text = _norm(value)
    if re.search(r"(?:->|→)|\bfrom\b.+\bto\b|\bchanged?\s+from\b", text):
        return "transition"
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b", text):
        return "time"
    if re.search(r"\b(?:preference|preferred|final choice|final decision|rejected)\b", text):
        return "actor_outcome"
    if re.fullmatch(r"[a-z][a-z .'-]{0,30}", text):
        return "short_label"
    return "relation_statement"


def audit_composite(
    composite: Mapping[str, Any], *, atomic_by_id: Mapping[str, AtomicQA],
) -> dict[str, Any]:
    """Audit a Composer response without using transcript or Solver calls."""
    errors: list[str] = []
    required = composite.get("required_qa_ids")
    if not isinstance(required, list) or len(required) != 2 or len(set(required)) != 2:
        errors.append("exactly_two_required_qa_ids")
        required = required if isinstance(required, list) else []
    source_qas = [atomic_by_id.get(str(value)) for value in required]
    if any(qa is None for qa in source_qas):
        errors.append("required_qa_id_unknown")
    qa1, qa2 = (source_qas + [None, None])[:2]
    evidence_groups = composite.get("required_evidence_groups")
    if not isinstance(evidence_groups, list) or len(evidence_groups) != 2:
        errors.append("exactly_two_required_evidence_groups")
        evidence_groups = evidence_groups if isinstance(evidence_groups, list) else []
    group_refs = []
    group_sources = []
    for group in evidence_groups:
        if not isinstance(group, Mapping):
            errors.append("evidence_group_schema")
            continue
        refs = group.get("turn_refs")
        source_id = str(group.get("source_qa_id") or "")
        if not source_id or not isinstance(refs, list) or not refs:
            errors.append("evidence_group_missing_source_or_refs")
        group_refs.append(set(str(ref) for ref in refs) if isinstance(refs, list) else set())
        group_sources.append(source_id)
    if len(group_sources) == 2 and isinstance(required, list):
        if group_sources != [str(value) for value in required]:
            errors.append("evidence_group_source_binding")
        for source_id, refs in zip(group_sources, group_refs):
            source_qa = atomic_by_id.get(source_id)
            if source_qa is None or not refs.issubset(set(source_qa.evidence_turn_refs)):
                errors.append(f"evidence_group_outside_source:{source_id}")
    if len(group_refs) == 2 and group_refs[0] == group_refs[1]:
        errors.append("evidence_sets_not_distinct")
    question = str(composite.get("question") or "").strip()
    if not question:
        errors.append("missing_composite_question")
    if is_mechanical_conjunction(question):
        errors.append("mechanical_conjunction")
    if composite.get("is_mechanical_conjunction") is True:
        errors.append("composer_flagged_mechanical_conjunction")
    options = composite.get("options")
    correct = str(composite.get("correct_option") or "").upper()
    if not isinstance(options, Mapping) or set(options) != set(OPTION_KEYS):
        errors.append("options_schema")
    else:
        normalized_options = [_norm(options[key]) for key in OPTION_KEYS]
        if any(not value for value in normalized_options):
            errors.append("empty_option")
        if len(set(normalized_options)) != 4:
            errors.append("options_not_unique")
        if correct not in OPTION_KEYS:
            errors.append("correct_option_schema")
        dimensions = {_option_dimension(options[key]) for key in OPTION_KEYS}
        if len(dimensions) != 1:
            errors.append("options_answer_dimension_mismatch")
    if not str(composite.get("answer_dimension") or "").strip():
        errors.append("answer_dimension_missing")
    gold = str(composite.get("gold_text") or "").strip()
    if not gold:
        errors.append("missing_composite_gold")
    elif isinstance(options, Mapping) and correct in OPTION_KEYS:
        matching = [key for key in OPTION_KEYS if _norm(options.get(key)) == _norm(gold)]
        if matching != [correct]:
            errors.append("composite_gold_not_uniquely_bound")
    required_evidence = composite.get("required_evidence_turn_refs")
    if not isinstance(required_evidence, list) or len(set(required_evidence)) < 2:
        errors.append("required_evidence_turn_refs_incomplete")
    elif len(group_refs) == 2 and set(str(ref) for ref in required_evidence) != set.union(*group_refs):
        errors.append("required_evidence_turn_refs_not_group_union")
    if qa1 is not None and qa2 is not None:
        if _norm(gold) in {_norm(qa1.gold_text), _norm(qa2.gold_text)}:
            errors.append("composite_gold_is_atomic_gold")
        relation_text = " ".join((gold, str(composite.get("relation_summary") or "")))
        for qa in (qa1, qa2):
            atomic_source = " ".join((qa.question, qa.gold_text))
            if not (lexical_tokens(relation_text) & lexical_tokens(atomic_source)):
                errors.append(f"composite_relation_missing_atomic:{qa.qa_id}")
        if (_norm(question) in {_norm(qa1.question), _norm(qa2.question)} or
                max(_similarity(question, qa1.question),
                    _similarity(question, qa2.question)) >= 0.9):
            errors.append("composite_question_is_atomic_question")
    return {
        "valid": not errors,
        "errors": errors,
        "checks": {
            "exactly_two_required_qa_ids": "exactly_two_required_qa_ids" not in errors,
            "evidence_sets_differ": "evidence_sets_not_distinct" not in errors,
            "evidence_groups_bound_to_source_qas": not any(
                error == "evidence_group_source_binding" or
                error.startswith("evidence_group_outside_source:")
                for error in errors),
            "gold_not_atomic": "composite_gold_is_atomic_gold" not in errors,
            "question_not_atomic": "composite_question_is_atomic_question" not in errors,
            "gold_relates_to_both_atomic_qas": not any(
                error.startswith("composite_relation_missing_atomic:") for error in errors),
            "not_mechanical_conjunction": "mechanical_conjunction" not in errors,
            "options_unique": "options_not_unique" not in errors,
            "correct_option_unique": "correct_option_schema" not in errors,
            "gold_bound": "composite_gold_not_uniquely_bound" not in errors,
            "same_answer_dimension": "options_answer_dimension_mismatch" not in errors,
            "required_evidence_is_group_union": "required_evidence_turn_refs_not_group_union" not in errors,
        },
    }
