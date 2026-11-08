from __future__ import annotations

from typing import Any, Mapping

from .hybrid_contract import ROLES


def validate_candidate(candidate: Mapping[str, Any], *, valid_turn_refs: set[str]) -> dict[str, Any]:
    """Structural filter only. Semantic evidence support is intentionally not judged."""
    blocking = list(candidate.get("schema_errors") or [])
    warnings: list[str] = []
    refs = [str(x) for x in candidate.get("evidence_refs") or []]
    if not refs:
        blocking.append("no_evidence_refs")
    if any(x not in valid_turn_refs for x in refs):
        blocking.append("evidence_ref_unknown")
    if len(refs) != len(set(refs)):
        blocking.append("duplicate_evidence_refs")
    groups = candidate.get("evidence_groups") or {}
    if not isinstance(groups, Mapping) or not groups:
        blocking.append("evidence_groups_missing")
    else:
        group_refs = [str(ref) for values in groups.values() for ref in (values if isinstance(values, list) else [])]
        if any(ref not in valid_turn_refs for ref in group_refs): blocking.append("evidence_group_ref_unknown")
        if not set(group_refs).issubset(set(refs)):
            warnings.append("group_ref_not_in_flat_evidence_refs")
    used = [str(x) for x in candidate.get("reasoning_groups_used") or []]
    if len(used) <= 1: warnings.append("single_hop")
    if not set(used).intersection(ROLES): warnings.append("reasoning_groups_unlabeled")
    if not str(candidate.get("hypothesized_reasoning_type") or "").strip() or candidate.get("hypothesized_reasoning_type") == "unknown":
        warnings.append("unknown_reasoning_type")
    alignment = candidate.get("contract_group_alignment") or {}
    if alignment and not all(bool(x) for x in alignment.values()): warnings.append("weak_contract_alignment")
    if len(set(str(x) for x in candidate.get("evidence_refs") or [])) < 2: warnings.append("narrow_evidence_scope")
    return {"candidate_id": candidate.get("candidate_id"),
            "decision": "reject" if blocking else "pass",
            "blocking_errors": sorted(set(blocking)), "warnings": sorted(set(warnings)),
            "structural_grounding_pass": not blocking}
