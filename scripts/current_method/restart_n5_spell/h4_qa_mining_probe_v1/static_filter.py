from __future__ import annotations

import re
from typing import Any, Mapping


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z'-]+", text)}


def audit_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    qtype = str(candidate.get("question_type") or "")
    question = str(candidate.get("question") or "")
    if not candidate.get("schema_valid"):
        blocking.extend(str(x) for x in candidate.get("schema_errors") or [])
    groups = candidate.get("evidence_groups")
    required = [str(x) for x in candidate.get("required_reasoning") or []]
    if not isinstance(groups, Mapping):
        blocking.append("evidence_groups_missing")
        groups = {}
    if len([key for key in required if groups.get(key)]) < 2:
        blocking.append("fewer_than_two_required_evidence_groups")
    alignment = candidate.get("contract_group_alignment")
    if isinstance(alignment, Mapping):
        missing_alignment = [key for key in required if not alignment.get(key)]
        if missing_alignment:
            warnings.append("evidence_group_not_aligned_to_intent_contract")
    evidence_refs = list(candidate.get("evidence_refs") or [])
    if not evidence_refs:
        blocking.append("no_evidence_refs")
    if len(evidence_refs) != len(set(evidence_refs)):
        blocking.append("duplicate_evidence_refs")

    options = candidate.get("options") if qtype != "ordering" else candidate.get("items")
    options = options if isinstance(options, Mapping) else {}
    lengths = [len(_tokens(str(value))) for value in options.values()]
    if lengths and max(lengths) - min(lengths) > max(8, int(max(1, min(lengths)) * 0.8)):
        warnings.append("option_length_imbalance")
    if qtype == "single_choice":
        correct = str(candidate.get("correct_option") or "")
        correct_len = len(_tokens(str(options.get(correct) or "")))
        if lengths and correct_len == max(lengths) and lengths.count(correct_len) == 1:
            warnings.append("gold_uniquely_longest")
    question_tokens = _tokens(question)
    overlap = {
        key: len(question_tokens & _tokens(str(value))) / max(1, len(question_tokens))
        for key, value in options.items()
    }
    if overlap and max(overlap.values()) > 0.75:
        warnings.append("question_option_keyword_overlap")
    if any(question.strip().lower() == str(value).strip().lower() for value in options.values()):
        blocking.append("question_copies_option")
    if qtype == "ordering":
        all_text = " ".join(str(value).lower() for value in options.values())
        if re.search(r"\b(first|second|third|then|finally|initially|later|afterward)\b", all_text):
            warnings.append("possible_lexical_temporal_leakage")
    strategy = str(candidate.get("distractor_strategy") or "")
    if strategy != "same_topic_wrong_relation":
        warnings.append("distractor_strategy_unverified")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "decision": "reject" if blocking else "pass",
        "blocking_errors": sorted(set(blocking)),
        "warnings": sorted(set(warnings)),
        "option_token_counts": dict(zip(options.keys(), lengths)),
        "question_option_overlap": overlap,
    }
