"""Claim Validator v2; validates references without modifying dialogue."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


CLAIM_VALIDATOR_VERSION = "claim_validator_v2"
REALIZATION_STATUSES = frozenset({"realized", "not_realized", "realization_uncertain"})


def validate_proposal_claims(
    *, scene_ref: str, proposal_claims: Any,
    compiled_proposals: Sequence[Mapping[str, Any]], dialogue_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(proposal_claims, list):
        return {"valid": False, "errors": ["proposal_claims_not_list"],
                "claim_validator_version": CLAIM_VALIDATOR_VERSION}
    proposal_refs = {str(row.get("proposal_ref")) for row in compiled_proposals}
    turn_refs = {str(row.get("turn_ref")) for row in dialogue_turns}
    seen: list[str] = []
    for index, claim in enumerate(proposal_claims):
        if (not isinstance(claim, Mapping) or
                set(claim) != {"proposal_ref", "realization_status", "candidate_evidence_turn_refs"}):
            errors.append(f"claim_schema:{index}")
            continue
        ref = str(claim.get("proposal_ref") or "")
        seen.append(ref)
        status = claim.get("realization_status")
        evidence = claim.get("candidate_evidence_turn_refs")
        if ref not in proposal_refs:
            errors.append(f"claim_unknown_proposal:{ref}")
        if status not in REALIZATION_STATUSES:
            errors.append(f"claim_realization_status:{ref}")
        if not isinstance(evidence, list) or not all(isinstance(value, str) for value in evidence):
            errors.append(f"claim_evidence_schema:{ref}")
            continue
        if status == "realized" and not evidence:
            errors.append(f"claim_realized_evidence_empty:{ref}")
        if status == "not_realized" and evidence:
            errors.append(f"claim_not_realized_evidence_nonempty:{ref}")
        if len(evidence) != len(set(evidence)):
            errors.append(f"claim_evidence_duplicate:{ref}")
        for turn_ref in evidence:
            if turn_ref not in turn_refs or not turn_ref.startswith(scene_ref + ":T"):
                errors.append(f"claim_evidence_not_current_scene:{ref}:{turn_ref}")
    counts = Counter(seen)
    if proposal_refs:
        for ref in sorted(proposal_refs):
            if counts[ref] != 1:
                errors.append(f"claim_cardinality:{ref}:{counts[ref]}")
    elif proposal_claims:
        errors.append("proposal_free_scene_claims_nonempty")
    return {"valid": not errors, "errors": errors,
            "claim_validator_version": CLAIM_VALIDATOR_VERSION}


def realizer_retry_route(
    *, json_parsed: bool, dialogue_quality: Mapping[str, Any] | None,
    claim_validation: Mapping[str, Any] | None, claims: Sequence[Mapping[str, Any]] = (),
    requests_used: int, maximum_requests: int = 3,
) -> str:
    dialogue_valid = bool(json_parsed and dialogue_quality and
                          dialogue_quality.get("valid"))
    claims_valid = bool(claim_validation and claim_validation.get("valid"))
    claims_realized = not any(
        row.get("realization_status") in {"not_realized", "realization_uncertain"}
        for row in claims)
    # A valid final attempt must still be accepted.  The request ceiling controls
    # whether another retry may be sent; it is not a reason to discard success.
    if dialogue_valid and claims_valid and claims_realized:
        return "submit_to_committer"
    if requests_used >= maximum_requests:
        return "scene_failed"
    if not dialogue_valid:
        return "full_scene_retry"
    if not claims_valid:
        return "claim_only_retry"
    return "full_scene_retry"
