"""Deterministic reasoning-contract alignment audit.

No model, HTTP client, vLLM, or Solver imports are allowed in this module.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping


ROLES = ("before_state", "trigger", "after_state")
QUESTION_TYPES = ("single_choice", "multi_select", "ordering")


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", str(value or ""))
        if len(token) > 2
    }


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(v) for v in value)
    return str(value or "")


def _as_refs(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _safe_accuracy(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _question_cues(question: str, qtype: str) -> dict[str, bool]:
    lowered = question.casefold()
    return {
        "causal_cue": bool(re.search(r"\bwhy|what led|what caused|because|factor", lowered)),
        "transition_cue": bool(re.search(r"change|changed|shift|evolv|became|updated|revised|from .* to", lowered)),
        "temporal_or_order_cue": bool(re.search(r"order|sequence|chronolog|before|after|first|later|then", lowered)),
        "multi_factor_cue": bool(re.search(r"factor|combination|which .* apply|statements|select all", lowered)),
        "type_expected_cue": (
            bool(re.search(r"order|sequence|chronolog|arrange", lowered)) if qtype == "ordering" else
            bool(re.search(r"factor|which|what|why|how", lowered))
        ),
    }


def _gold_texts(candidate: Mapping[str, Any]) -> list[str]:
    qtype = str(candidate.get("question_type") or "")
    if qtype == "single_choice":
        key = str(candidate.get("correct_option") or "")
        return [str((candidate.get("options") or {}).get(key) or "")]
    if qtype == "multi_select":
        options = candidate.get("options") or {}
        return [str(options.get(key) or "") for key in candidate.get("gold") or []]
    items = candidate.get("items") or {}
    return [str(items.get(key) or "") for key in candidate.get("gold") or []]


def _turn_index(dialogue: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row.get("turn_ref")): str(row.get("text") or "")
        for row in dialogue.get("turns") or []
        if row.get("turn_ref")
    }


def _gold_support(candidate: Mapping[str, Any], evidence_text: str) -> dict[str, Any]:
    evidence_tokens = _tokens(evidence_text)
    rows = []
    for text in _gold_texts(candidate):
        gold_tokens = _tokens(text)
        overlap = len(gold_tokens & evidence_tokens) / max(1, len(gold_tokens))
        rows.append({"text": text, "token_overlap": overlap, "supported": overlap > 0.0})
    return {
        "items": rows,
        "supported_count": sum(bool(row["supported"]) for row in rows),
        "item_count": len(rows),
        "support_rate": (
            sum(bool(row["supported"]) for row in rows) / len(rows) if rows else None
        ),
        "heuristic_only": True,
    }


def audit_candidate(
    candidate: Mapping[str, Any], *, contract: Mapping[str, Any] | None,
    turn_text: Mapping[str, str], difficulty: Mapping[str, Any] | None,
    static: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract = contract or {}
    groups = candidate.get("evidence_groups") or {}
    groups = {str(key): _as_refs(value) for key, value in groups.items()}
    contract_groups = {
        str(key): set(_as_refs(value))
        for key, value in (contract.get("required_evidence_groups") or {}).items()
    }
    required = [str(key) for key in candidate.get("required_reasoning") or []]
    evidence_refs = set(_as_refs(candidate.get("evidence_refs")))
    valid_turn_refs = set(turn_text)
    group_union = [ref for values in groups.values() for ref in values]
    group_overlap = {
        role: sorted(set(groups.get(role, [])) & contract_groups.get(role, set()))
        for role in ROLES
    }
    exact_match = {
        role: bool(groups.get(role, []) and
                   set(groups.get(role, [])) == contract_groups.get(role, set()))
        for role in ROLES
    }
    aligned_roles = [role for role in required if group_overlap.get(role)]
    exact_roles = [role for role in required if exact_match.get(role)]
    if len(exact_roles) == len(required) and required:
        alignment_bucket = "aligned"
    elif aligned_roles:
        alignment_bucket = "partial"
    else:
        alignment_bucket = "misaligned"
    cross_group_duplicates = sorted(ref for ref, count in Counter(group_union).items() if count > 1)
    required_presence = {role: bool(groups.get(role)) for role in ROLES}
    required_valid = {role: bool(groups.get(role)) and
                      all(ref in valid_turn_refs for ref in groups.get(role, []))
                      for role in ROLES}
    contract_anchor_present = {
        role: bool(contract_groups.get(role)) for role in ROLES
    }
    declared_group_refs = set(group_union)
    missing_from_evidence = sorted(declared_group_refs - evidence_refs)
    unsupported_evidence = sorted(evidence_refs - declared_group_refs)
    valid_contract = bool(contract.get("contract_valid", True))
    question = str(candidate.get("question") or "")
    qtype = str(candidate.get("question_type") or "")
    variable = str(candidate.get("discriminative_variable") or contract.get("discriminative_variable") or "")
    variable_tokens = _tokens(variable)
    question_tokens = _tokens(question)
    variable_overlap = sorted(variable_tokens & question_tokens)
    dialogue_evidence = " ".join(turn_text.get(ref, "") for ref in sorted(declared_group_refs))
    gold_support = _gold_support(candidate, dialogue_evidence)
    cues = _question_cues(question, qtype)
    required_all_roles = set(required) == set(ROLES)
    contract_alignment_pass = bool(
        valid_contract and required and all(required_presence.get(role) for role in required)
        and all(required_valid.get(role) for role in required)
        and not cross_group_duplicates and not missing_from_evidence
        and alignment_bucket == "aligned"
    )
    structural_pass = bool(
        qtype in QUESTION_TYPES and cues["type_expected_cue"] and
        (required_all_roles or len(required) >= 2)
    )
    issues: list[str] = []
    if not valid_contract:
        issues.append("intent_contract_invalid")
    if not required:
        issues.append("required_reasoning_missing")
    if any(not required_presence.get(role) for role in required):
        issues.append("required_group_missing")
    if any(not required_valid.get(role) for role in required):
        issues.append("evidence_turn_ref_unknown")
    if cross_group_duplicates:
        issues.append("evidence_group_overlap")
    if missing_from_evidence:
        issues.append("declared_group_not_in_evidence_refs")
    if unsupported_evidence:
        issues.append("evidence_ref_not_declared_in_group")
    if alignment_bucket == "misaligned":
        issues.append("no_contract_anchor_overlap")
    elif alignment_bucket == "partial":
        issues.append("partial_contract_anchor_overlap")
    if not variable_overlap:
        issues.append("discriminative_variable_not_lexically_in_question")
    if not structural_pass:
        issues.append("question_structure_not_relation_cued")
    if gold_support.get("support_rate") == 0:
        issues.append("gold_has_no_lexical_evidence_support")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "intent_id": candidate.get("intent_id"),
        "question_type": qtype,
        "alignment_bucket": alignment_bucket,
        "contract_alignment_pass": contract_alignment_pass,
        "structural_relation_cue_pass": structural_pass,
        "contract_valid": valid_contract,
        "required_reasoning": required,
        "required_group_presence": required_presence,
        "required_group_turn_refs_valid": required_valid,
        "contract_anchor_present": contract_anchor_present,
        "contract_anchor_overlap": group_overlap,
        "contract_group_exact_match": exact_match,
        "contract_anchor_overlap_rate": len(aligned_roles) / len(required) if required else 0.0,
        "contract_anchor_exact_match_rate": len(exact_roles) / len(required) if required else 0.0,
        "evidence_refs": sorted(evidence_refs),
        "declared_group_refs": sorted(declared_group_refs),
        "missing_group_refs_from_evidence": missing_from_evidence,
        "evidence_refs_outside_declared_groups": unsupported_evidence,
        "cross_group_duplicate_refs": cross_group_duplicates,
        "discriminative_variable": variable,
        "discriminative_variable_question_overlap": variable_overlap,
        "question_cues": cues,
        "gold_evidence_support": gold_support,
        "issues": sorted(set(issues)),
        "static_decision": (static or {}).get("decision"),
        "static_warnings": list((static or {}).get("warnings") or []),
        "difficulty_bucket": (difficulty or {}).get("difficulty_bucket"),
        "full_context_accuracy": _safe_accuracy(((difficulty or {}).get("full_context") or {}).get("accuracy")),
        "no_context_accuracy": _safe_accuracy(((difficulty or {}).get("no_context") or {}).get("accuracy")),
        "position_accuracy_ranges": dict((difficulty or {}).get("position_accuracy_ranges") or {}),
        "audit_method": "deterministic_contract_and_lexical_v1",
    }


def audit_bundle(
    *, candidates: list[Mapping[str, Any]], contracts: list[Mapping[str, Any]],
    turn_text: Mapping[str, str], static_rows: Mapping[str, Any],
    difficulty_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    contract_by_intent = {str(row.get("intent_id")): row for row in contracts}
    difficulty_by_id = {str(row.get("candidate_id")): row for row in difficulty_rows}
    items = [audit_candidate(
        candidate, contract=contract_by_intent.get(str(candidate.get("intent_id"))),
        turn_text=turn_text, difficulty=difficulty_by_id.get(str(candidate.get("candidate_id"))),
        static=static_rows.get(str(candidate.get("candidate_id"))),
    ) for candidate in candidates]
    alignment_counts = Counter(str(row["alignment_bucket"]) for row in items)
    contract_pass_count = sum(bool(row["contract_alignment_pass"]) for row in items)
    structural_pass_count = sum(bool(row["structural_relation_cue_pass"]) for row in items)
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for row in items:
        by_bucket[str(row.get("difficulty_bucket") or "unknown")][str(row["alignment_bucket"])] += 1
    issue_counts = Counter(issue for row in items for issue in row.get("issues") or [])
    return {
        "audit_version": "h4_qa_alignment_audit_v1",
        "offline_only": True,
        "challenger_api_calls": 0,
        "solver_calls": 0,
        "candidate_count": len(items),
        "contract_alignment_pass_count": contract_pass_count,
        "structural_relation_cue_pass_count": structural_pass_count,
        "alignment_bucket_counts": dict(sorted(alignment_counts.items())),
        "difficulty_by_alignment": {
            bucket: dict(sorted(counter.items()))
            for bucket, counter in sorted(by_bucket.items())
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "items": items,
    }
