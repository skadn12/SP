"""Static audit for candidate-bound composite QA."""

from __future__ import annotations

import re
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.static_audit import audit_composite


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _tokens(value: Any) -> set[str]:
    from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.contracts import lexical_tokens
    return lexical_tokens(value)


def audit_candidate_composite(
    composite: Mapping[str, Any], *, candidate: Mapping[str, Any],
    atomic_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    base = audit_composite(composite, atomic_by_id=atomic_by_id)
    errors = [error for error in base["errors"] if error != "mechanical_conjunction"]
    source_qids = [str(qid) for qid in candidate["source_qids"]]
    required_qids = [str(qid) for qid in (composite.get("required_qa_ids") or [])]
    if required_qids != source_qids:
        errors.append("required_qa_ids_not_candidate_bound")
    if str(composite.get("candidate_id") or "") != str(candidate["candidate_id"]):
        errors.append("candidate_id_mismatch")
    if composite.get("source_contributions") != candidate.get("source_contributions"):
        errors.append("source_contributions_not_copied_unchanged")
    question = _tokens(composite.get("question"))
    gold = _tokens(composite.get("gold_text"))
    for item in candidate["source_contributions"]:
        contribution = _tokens(item["unique_information"])
        # This is only a static anchor check. Paraphrase equivalence and actual
        # necessity are evaluated later by the full/no-context/ablation Solver
        # gate, so one lexical anchor is sufficient here.
        if not contribution & gold:
            errors.append(f"gold_missing_source_contribution:{item['source_qid']}")
        if contribution and len(contribution & question) / max(1, len(contribution)) >= 0.7:
            errors.append(f"question_leaks_source_contribution:{item['source_qid']}")
    required_groups = composite.get("required_evidence_groups") or []
    expected_groups = {
        str(qid): set(refs)
        for qid, refs in (candidate.get("unique_contribution_closure") or {}).items()
    }
    seen_groups: dict[str, set[str]] = {}
    for group in required_groups:
        if isinstance(group, Mapping):
            seen_groups[str(group.get("source_qa_id") or "")] = {str(ref) for ref in (group.get("turn_refs") or [])}
    if set(seen_groups) != set(expected_groups):
        errors.append("required_evidence_groups_not_candidate_bound")
    for qid, refs in expected_groups.items():
        if seen_groups.get(qid) != refs:
            errors.append(f"required_evidence_group_mismatch:{qid}")
    options = composite.get("options") or {}
    correct = str(composite.get("correct_option") or "").upper()
    if isinstance(options, Mapping) and correct in {"A", "B", "C", "D"}:
        unique_support = set().union(*(
            {str(token) for token in (item.get("unique_support_tokens") or [])}
            for item in candidate["source_contributions"]
        ))
        gold_specificity = len(_tokens(options.get(correct)) & unique_support) / max(1, len(unique_support))
        for letter in "ABCD":
            if letter == correct:
                continue
            distractor_specificity = len(_tokens(options.get(letter)) & unique_support) / max(1, len(unique_support))
            if distractor_specificity > gold_specificity:
                errors.append(f"distractor_more_evidence_specific_than_gold:{letter}")
    mutations = composite.get("distractor_mutations") or {}
    for letter in "ABCD":
        if letter == correct:
            continue
        if str(mutations.get(letter) or "") not in source_qids:
            errors.append(f"distractor_mutation_missing_or_invalid:{letter}")
    return {
        "valid": not errors,
        "errors": errors,
        "base_static_audit": base,
        "candidate_id": candidate["candidate_id"],
        "source_qids": source_qids,
        "gold_contribution_checks": {
            item["source_qid"]: sorted(_tokens(item["unique_information"]) & gold)
            for item in candidate["source_contributions"]
        },
        "question_contribution_checks": {
            item["source_qid"]: sorted(_tokens(item["unique_information"]) & question)
            for item in candidate["source_contributions"]
        },
    }
