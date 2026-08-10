"""Evidence Committer contract plus candidate-subset and scope audit."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


EVIDENCE_SCOPE_AUDIT_VERSION = "evidence_committer_scope_audit_v1"
EVIDENCE_SCOPE_PROMPT_VERSION = "evidence_committer_scope_v1_prompt_v1"
EVIDENCE_SCOPE_RETRY_PROMPT_VERSION = "evidence_committer_scope_v1_retry_prompt_v1"
REVIEW_DECISIONS = frozenset({"confirmed", "rejected", "review_uncertain"})
SCOPE_STATUSES = frozenset({"clean", "extraneous_memory_claim_detected", "uncertain_scope"})
CONSISTENCY_STATUSES = frozenset({"clean", "authorized_change", "contradiction", "uncertain"})


def validate_committer_scope_output(
    value: Any, *, compiled_proposals: Sequence[Mapping[str, Any]],
    proposal_claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if (not isinstance(value, Mapping) or
            set(value) != {"proposal_reviews", "scope_status", "world_consistency_status"}):
        return {"valid": False, "errors": ["committer_top_level_schema"]}
    if value.get("scope_status") not in SCOPE_STATUSES:
        errors.append("committer_scope_status")
    if value.get("world_consistency_status") not in CONSISTENCY_STATUSES:
        errors.append("committer_world_consistency_status")
    proposals = {str(row.get("proposal_ref")) for row in compiled_proposals}
    candidates = {str(row.get("proposal_ref")): set(row.get("candidate_evidence_turn_refs") or [])
                  for row in proposal_claims}
    reviews = value.get("proposal_reviews")
    if not isinstance(reviews, list):
        return {"valid": False, "errors": [*errors, "committer_reviews_not_list"]}
    seen: list[str] = []
    for index, review in enumerate(reviews):
        expected = {"proposal_ref", "review_decision", "verification_status",
                    "accepted_evidence_turn_refs", "reason_code"}
        if not isinstance(review, Mapping) or set(review) != expected:
            errors.append(f"committer_review_schema:{index}")
            continue
        ref = str(review.get("proposal_ref") or "")
        seen.append(ref)
        decision = review.get("review_decision")
        accepted = review.get("accepted_evidence_turn_refs")
        if ref not in proposals:
            errors.append(f"committer_unknown_proposal:{ref}")
        if decision not in REVIEW_DECISIONS:
            errors.append(f"committer_review_decision:{ref}")
        if not isinstance(accepted, list) or not all(isinstance(item, str) for item in accepted):
            errors.append(f"committer_evidence_schema:{ref}")
            continue
        if len(accepted) != len(set(accepted)):
            errors.append(f"committer_evidence_duplicate:{ref}")
        if not set(accepted).issubset(candidates.get(ref, set())):
            errors.append(f"committer_evidence_not_candidate_subset:{ref}")
        if not isinstance(review.get("reason_code"), str) or not review["reason_code"].strip():
            errors.append(f"committer_reason_code:{ref}")
        if decision == "confirmed":
            if not accepted:
                errors.append(f"committer_confirmed_evidence_empty:{ref}")
            if review.get("verification_status") != "semantically_verified":
                errors.append(f"committer_confirmed_status:{ref}")
        elif decision == "rejected" and review.get("verification_status") != "rejected":
            errors.append(f"committer_rejected_status:{ref}")
        elif decision == "review_uncertain" and review.get("verification_status") != "uncertain":
            errors.append(f"committer_uncertain_status:{ref}")
    if len(seen) != len(set(seen)):
        errors.append("committer_proposal_duplicate")
    if set(seen) != proposals:
        errors.append("committer_proposal_cardinality")
    return {"valid": not errors, "errors": errors,
            "scope_audit_version": EVIDENCE_SCOPE_AUDIT_VERSION}


def build_committer_scope_prompt(
    *, compiled_proposals: Sequence[Mapping[str, Any]],
    dialogue_turns: Sequence[Mapping[str, Any]], proposal_claims: Sequence[Mapping[str, Any]],
    authoritative_world: Mapping[str, Any],
) -> str:
    mode = "scope-only" if not compiled_proposals else "proposal-review-and-scope"
    review_template = [{
        "proposal_ref": str(row["proposal_ref"]),
        "review_decision": "confirmed|rejected|review_uncertain",
        "verification_status": "semantically_verified|rejected|uncertain",
        "accepted_evidence_turn_refs": [],
        "reason_code": "concise_code",
    } for row in compiled_proposals]
    return "\n\n".join([
        "INDEPENDENT EVIDENCE COMMITTER + SCOPE AUDIT. Return JSON only. Never select evidence outside Realizer candidates and never create proposals.",
        "Review every proposal exactly once in this one scene-level response. confirmed requires semantic support, verification_status=semantically_verified, and at least one accepted candidate evidence ref. rejected/review_uncertain use rejected/uncertain status. Independently flag persistent facts outside proposals via scope_status, and contradictions via world_consistency_status. A no-proposal scene still requires scope-only review with proposal_reviews=[].",
        "MODE\n" + mode,
        "FROZEN PROPOSALS\n" + json.dumps(list(compiled_proposals), ensure_ascii=False, separators=(",", ":")),
        "ACTUAL DIALOGUE\n" + json.dumps(list(dialogue_turns), ensure_ascii=False, separators=(",", ":")),
        "REALIZER CLAIMS AND EXCLUSIVE CANDIDATES\n" + json.dumps(list(proposal_claims), ensure_ascii=False, separators=(",", ":")),
        "AUTHORITATIVE WORLD ONLY\n" + json.dumps(dict(authoritative_world), ensure_ascii=False, separators=(",", ":")),
        "OUTPUT EXACTLY\n" + json.dumps({"proposal_reviews": review_template, "scope_status": "clean",
                                       "world_consistency_status": "clean"}, separators=(",", ":")),
    ])


def build_committer_scope_retry_prompt(
    *, frozen_base_prompt: str, contract_errors: Sequence[str],
) -> str:
    """Repair serialization only; semantic inputs and review standard stay frozen."""
    return "\n\n".join([
        "EVIDENCE COMMITTER CONTRACT RETRY. Do not change the frozen evidence standard or seek new evidence.",
        "CONTRACT ERROR CODES\n" + json.dumps(list(contract_errors), ensure_ascii=False, separators=(",", ":")),
        frozen_base_prompt,
    ])


def committer_retry_disposition(validation: Mapping[str, Any], output: Mapping[str, Any] | None) -> str:
    if not validation.get("valid"):
        return "contract_retry"
    decisions = {row.get("review_decision") for row in (output or {}).get("proposal_reviews") or []}
    if decisions & {"rejected", "review_uncertain"}:
        return "final_nonconfirmed"
    return "final"
