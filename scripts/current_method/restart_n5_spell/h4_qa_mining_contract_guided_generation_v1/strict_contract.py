from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..h4_qa_mining_probe_v1.candidate_contract import normalize_candidates


ROLES = ("before_state", "trigger", "after_state")


def validate_guided_candidates(
    raw: Any, *, contract: Mapping[str, Any], expanded_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the shared schema validator, then enforce exact contract binding."""
    normalized = normalize_candidates(
        raw, intent_id=str(contract.get("intent_id")), contract=contract,
        expanded_dialogue=expanded_dialogue)
    errors = list(normalized.get("errors") or [])
    candidates: list[dict[str, Any]] = []
    expected_groups = {
        role: [str(ref) for ref in (contract.get("required_evidence_groups") or {}).get(role) or []]
        for role in ROLES
    }
    expected_refs = {ref for refs in expected_groups.values() for ref in refs}
    expected_variable = str(contract.get("discriminative_variable") or "").strip()
    for candidate in normalized.get("candidates") or []:
        row = dict(candidate)
        row_errors = list(row.get("schema_errors") or [])
        groups = {str(key): [str(ref) for ref in value]
                  for key, value in (row.get("evidence_groups") or {}).items()}
        if set(groups) != set(ROLES) or any(groups.get(role) != expected_groups[role] for role in ROLES):
            row_errors.append("evidence_groups_not_exact_contract")
        if set(str(ref) for ref in row.get("evidence_refs") or []) != expected_refs:
            row_errors.append("evidence_refs_not_exact_contract_union")
        if set(str(x) for x in row.get("required_reasoning") or []) != set(ROLES):
            row_errors.append("required_reasoning_not_complete_chain")
        if str(row.get("discriminative_variable") or "").strip() != expected_variable:
            row_errors.append("discriminative_variable_not_exact_contract")
        if len(set(ref for refs in groups.values() for ref in refs)) != sum(len(refs) for refs in groups.values()):
            row_errors.append("contract_evidence_groups_overlap")
        row["guided_contract_errors"] = sorted(set(row_errors))
        row["guided_contract_valid"] = not row_errors
        candidates.append(row)
        errors.extend(f"{row.get('candidate_id')}:{error}" for error in row_errors)
    counts = Counter(str(row.get("question_type") or "") for row in candidates if row.get("guided_contract_valid"))
    minimums = {"single_choice": 2, "multi_select": 1, "ordering": 1}
    for qtype, minimum in minimums.items():
        if counts[qtype] < minimum:
            errors.append(f"guided_candidate_count_below_minimum:{qtype}:{counts[qtype]}")
    return {
        "valid": not errors and all(row.get("guided_contract_valid") for row in candidates),
        "errors": sorted(set(errors)), "candidates": candidates,
        "type_counts": dict(counts),
        "strict_contract": {
            "required_reasoning": list(ROLES),
            "required_evidence_groups": expected_groups,
            "evidence_ref_union": sorted(expected_refs),
            "discriminative_variable": expected_variable,
        },
    }
