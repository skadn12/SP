"""Deterministic Realizer claim validation and Evidence Committer routing.

Planned graph changes and emergent open-thread observations are deliberately
separate.  This module performs reference checks only; it never judges dialogue
semantics, assigns authoritative thread IDs, or upgrades a model claim.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .prompts import (
    build_emergent_evidence_committer_prompt,
    build_evidence_committer_prompt,
)
from .schemas import (
    REALIZER_SCHEMA_VERSION,
    validate_natural_realizer_output,
    validate_realizer_claim_only_output,
)


REALIZER_CLAIM_CONTRACT_VERSION = "realizer_claim_contract_v2"
REALIZER_CLAIM_ONLY_RETRY_VERSION = "realizer_claim_only_retry_v1"
LEGACY_REALIZER_SCHEMA_VERSION = "natural_realizer_v1"

_CLAIM_ONLY_ERROR_PREFIXES = (
    "realizer_claim_evidence_not_in_current_scene:",
    "realizer_claim_realized_evidence_empty:",
    "realizer_claim_not_realized_evidence_nonempty:",
    "realizer_claim_duplicate_delta:",
)


def normalize_realizer_output(
    output: Mapping[str, Any], *, allow_legacy_shadow_migration: bool = False,
) -> dict[str, Any]:
    """Return a copy with optional fields filled and audited v1 status migration.

    The legacy migration is intentionally opt-in for read-only shadow replay.
    Production v2 validation never accepts ``completed``/``not_completed`` as
    Realizer claim statuses.
    """
    normalized = copy.deepcopy(dict(output))
    migrations: list[str] = []
    if (allow_legacy_shadow_migration and
            normalized.get("realizer_schema_version") ==
            LEGACY_REALIZER_SCHEMA_VERSION):
        normalized["realizer_schema_version"] = REALIZER_SCHEMA_VERSION
        for claim in normalized.get("claimed_graph_updates") or []:
            if claim.get("claimed_status") == "completed":
                claim["claimed_status"] = "realized"
                migrations.append("claimed_status:completed->realized")
            elif claim.get("claimed_status") == "not_completed":
                claim["claimed_status"] = "not_realized"
                migrations.append("claimed_status:not_completed->not_realized")
        for claim in normalized.get("new_open_thread_claims") or []:
            if "related_proposed_delta_id" not in claim:
                claim["related_proposed_delta_id"] = None
                migrations.append("emergent_claim:related_proposed_delta_id->null")
        migrations.append(
            f"schema:{LEGACY_REALIZER_SCHEMA_VERSION}->{REALIZER_SCHEMA_VERSION}")
    normalized.setdefault("new_open_thread_claims", [])
    return {"output": normalized, "migrations": migrations}


def validate_realizer_scene_contract(
    output: Mapping[str, Any], *, scene: Mapping[str, Any],
    accepted_deltas: Sequence[Mapping[str, Any]], expected_root_id: str | None = None,
    allow_legacy_shadow_migration: bool = False,
) -> dict[str, Any]:
    """Validate local structure without comparing or interpreting utterance text."""
    normalized_result = normalize_realizer_output(
        output, allow_legacy_shadow_migration=allow_legacy_shadow_migration)
    normalized = normalized_result["output"]
    validation = validate_natural_realizer_output(normalized)
    dialogue_errors = list(validation.get("dialogue_errors") or [])
    claim_errors = list(validation.get("claim_errors") or [])
    component_errors = set(dialogue_errors) | set(claim_errors)
    general_errors = [error for error in validation["errors"]
                      if error not in component_errors]

    if normalized.get("scene_key") != scene.get("scene_key"):
        dialogue_errors.append("realizer_scene_key_mismatch")
    participants = set(str(value) for value in scene.get("participants") or [])
    for turn in normalized.get("dialogue_turns") or []:
        if turn.get("speaker") not in participants:
            dialogue_errors.append(
                f"realizer_speaker_not_participant:{turn.get('speaker')}")

    delta_by_id: dict[str, Mapping[str, Any]] = {}
    for delta in accepted_deltas:
        delta_id = str(delta.get("proposed_delta_id") or "")
        if not delta_id:
            general_errors.append("accepted_delta_missing_identity")
            continue
        if delta_id in delta_by_id:
            general_errors.append(f"accepted_delta_identity_duplicate:{delta_id}")
        delta_by_id[delta_id] = delta
        if expected_root_id is not None and (
                delta.get("root_id") != expected_root_id or
                not delta_id.startswith(f"{expected_root_id}:")):
            general_errors.append(f"accepted_delta_root_mismatch:{delta_id}")

    claimed_ids: list[str] = []
    for claim in normalized.get("claimed_graph_updates") or []:
        delta_id = str(claim.get("proposed_delta_id") or "")
        claimed_ids.append(delta_id)
        if delta_id not in delta_by_id:
            claim_errors.append(f"realizer_claim_not_current_scene_delta:{delta_id}")
    seen: set[str] = set()
    for delta_id in claimed_ids:
        if delta_id in seen:
            claim_errors.append(f"realizer_claim_duplicate_delta:{delta_id}")
        seen.add(delta_id)
    for delta_id in delta_by_id:
        if delta_id not in seen:
            claim_errors.append(f"realizer_claim_missing_delta:{delta_id}")

    dialogue_errors = list(dict.fromkeys(dialogue_errors))
    claim_errors = list(dict.fromkeys(claim_errors))
    general_errors = list(dict.fromkeys(general_errors))
    errors = list(dict.fromkeys([*general_errors, *dialogue_errors, *claim_errors]))
    dialogue_valid = not dialogue_errors
    claims_valid = not claim_errors
    claim_only_eligible = bool(claim_errors) and all(
        error.startswith(_CLAIM_ONLY_ERROR_PREFIXES) or
        error.startswith("claimed_update_value:")
        for error in claim_errors)
    retry_mode = None
    if errors:
        retry_mode = (
            "claim_only_retry"
            if dialogue_valid and claims_valid is False and not general_errors and
            claim_only_eligible
            else "full_scene_retry")

    # There is intentionally no planned-open-thread/emergent-claim count gate.
    # Emergent claims are optional and schema validation requires a null related
    # planned-delta reference.  No keyword, distance, or similarity test occurs.
    return {
        "valid": not errors,
        "errors": errors,
        "dialogue_valid": dialogue_valid,
        "claims_valid": claims_valid,
        "dialogue_errors": dialogue_errors,
        "claim_errors": claim_errors,
        "retry_mode": retry_mode,
        "schema_validation": validation,
        "normalized_output": normalized,
        "legacy_shadow_migrations": normalized_result["migrations"],
        "planned_claim_count": len(claimed_ids),
        "emergent_claim_count": len(normalized.get("new_open_thread_claims") or []),
        "semantic_decision_performed": False,
    }


def validate_claim_only_retry_contract(
    output: Any, *, frozen_scene_key: str,
    frozen_dialogue_turns: Sequence[Mapping[str, Any]],
    accepted_deltas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate claims-only JSON against frozen dialogue and current deltas."""
    del frozen_scene_key  # Identity is program-owned and never accepted from the model.
    valid_turn_keys = {
        str(turn.get("turn_key")) for turn in frozen_dialogue_turns
        if isinstance(turn, Mapping) and turn.get("turn_key") is not None
    }
    expected_delta_ids = {
        str(delta.get("proposed_delta_id")) for delta in accepted_deltas
        if delta.get("proposed_delta_id") is not None
    }
    validation = validate_realizer_claim_only_output(
        output, valid_turn_keys=valid_turn_keys,
        expected_delta_ids=expected_delta_ids)
    return {
        **validation,
        "dialogue_valid": True,
        "claims_valid": bool(validation["valid"]),
        "retry_mode": None if validation["valid"] else "claim_only_retry",
        "allowed_evidence_turn_keys": sorted(valid_turn_keys),
        "expected_delta_ids": sorted(expected_delta_ids),
    }


def reconstruct_scene_from_claim_only(
    *, frozen_scene_key: str,
    frozen_dialogue_turns: Sequence[Mapping[str, Any]],
    claim_only_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine model-repaired claims with an immutable program-owned dialogue."""
    return {
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "scene_key": str(frozen_scene_key),
        "dialogue_turns": copy.deepcopy(list(frozen_dialogue_turns)),
        "claimed_graph_updates": copy.deepcopy(
            list(claim_only_output["claimed_graph_updates"])),
        "new_open_thread_claims": copy.deepcopy(
            list(claim_only_output["new_open_thread_claims"])),
    }


def validate_committer_candidate_subset(
    output: Mapping[str, Any], *, candidate_evidence_turn_keys: Sequence[str],
    realizer_claimed_status: str | None = None,
) -> dict[str, Any]:
    """Prevent the reviewer from discovering evidence outside Realizer candidates."""
    errors: list[str] = []
    allowed = set(str(key) for key in candidate_evidence_turn_keys)
    evidence = output.get("evidence_turn_keys")
    if not isinstance(evidence, list):
        errors.append("committer_evidence_turn_keys_not_list")
        evidence = []
    outside = [str(key) for key in evidence if str(key) not in allowed]
    if outside:
        errors.append(
            "committer_evidence_not_realizer_candidate:" + ",".join(outside))
    if (realizer_claimed_status == "not_realized" and
            output.get("decision") == "confirmed"):
        errors.append("committer_cannot_confirm_not_realized_claim")
    return {"valid": not errors, "errors": errors,
            "candidate_evidence_turn_keys": sorted(allowed)}


def _turns_by_key(realizer_output: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(turn["turn_key"]): turn
            for turn in realizer_output.get("dialogue_turns") or []}


def build_planned_committer_routes(
    *, realizer_output: Mapping[str, Any],
    deltas: Sequence[Mapping[str, Any]], memory_view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build one independent semantic request per planned Realizer claim."""
    delta_by_id = {str(delta["proposed_delta_id"]): delta for delta in deltas}
    turns = _turns_by_key(realizer_output)
    routes: list[dict[str, Any]] = []
    for claim in realizer_output.get("claimed_graph_updates") or []:
        delta_id = str(claim.get("proposed_delta_id") or "")
        delta = delta_by_id.get(delta_id)
        if delta is None:
            raise ValueError(f"planned_claim_unknown_delta:{delta_id}")
        evidence_keys = list(claim.get("candidate_evidence_turn_keys") or [])
        status = str(claim.get("claimed_status") or "")
        if (status == "realized" and not evidence_keys) or any(
                key not in turns for key in evidence_keys):
            raise ValueError(f"planned_claim_evidence_reference_invalid:{delta_id}")
        prompt = build_evidence_committer_prompt(
            proposed_delta=delta,
            dialogue_turns=list(realizer_output.get("dialogue_turns") or []),
            realizer_claims=[claim], memory_view=memory_view)
        routes.append({
            "route_type": "planned_graph_delta",
            "proposed_delta_id": delta_id,
            "compiled_proposed_delta": copy.deepcopy(dict(delta)),
            "realizer_claimed_status": status,
            "candidate_evidence_turn_keys": evidence_keys,
            "candidate_evidence_turns": [copy.deepcopy(dict(turns[key]))
                                          for key in evidence_keys],
            "prompt": prompt,
            "semantic_status": "model_claimed",
            "evidence_decision": None,
        })
    return routes


def compile_emergent_pending_proposals(
    *, root_id: str, scene: Mapping[str, Any],
    realizer_output: Mapping[str, Any], memory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile emergent claims into non-authoritative, program-owned proposals."""
    if memory.get("root_id") != root_id:
        raise ValueError("emergent_claim_memory_root_mismatch")
    entity_by_name = {str(row.get("display_name")): str(row.get("entity_id"))
                      for row in memory.get("entities") or []}
    subject_ids = []
    for participant in scene.get("participants") or []:
        entity_id = entity_by_name.get(str(participant))
        if not entity_id:
            raise ValueError(f"emergent_claim_unknown_participant:{participant}")
        subject_ids.append(entity_id)
    turn_keys = set(_turns_by_key(realizer_output))
    proposals: list[dict[str, Any]] = []
    for index, claim in enumerate(
            realizer_output.get("new_open_thread_claims") or [], 1):
        if claim.get("related_proposed_delta_id") is not None:
            raise ValueError("emergent_claim_must_not_reference_planned_delta")
        evidence_keys = list(claim.get("candidate_evidence_turn_keys") or [])
        if not evidence_keys or not set(evidence_keys) <= turn_keys:
            raise ValueError("emergent_claim_evidence_reference_invalid")
        local_id = f"E{index:04d}"
        proposal_id = f"{root_id}:emergent:{scene['scene_key']}:{local_id}"
        proposals.append({
            "proposal_id": proposal_id,
            "root_id": root_id,
            "source_scene": str(scene["scene_key"]),
            "operation": "open_thread",
            "event_identity_key": (
                f"{root_id}:emergent_open_thread:{scene['scene_key']}:{local_id}"),
            "subject_ids": subject_ids,
            "description": str(claim["description"]),
            "candidate_evidence_turn_keys": evidence_keys,
            "semantic_status": "model_claimed",
            "formal_thread_id": None,
            "authoritative": False,
            "provenance": {
                "generated_by": REALIZER_CLAIM_CONTRACT_VERSION,
                "role": "natural_realizer",
            },
        })
    return proposals


def build_emergent_committer_routes(
    *, proposals: Sequence[Mapping[str, Any]],
    realizer_output: Mapping[str, Any], memory_view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build semantic review requests; proposals remain non-authoritative."""
    turns = _turns_by_key(realizer_output)
    routes: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        evidence_keys = list(proposal.get("candidate_evidence_turn_keys") or [])
        if not evidence_keys or any(key not in turns for key in evidence_keys):
            raise ValueError(f"emergent_claim_evidence_reference_invalid:{proposal_id}")
        prompt = build_emergent_evidence_committer_prompt(
            pending_proposal=proposal,
            dialogue_turns=list(realizer_output.get("dialogue_turns") or []),
            candidate_evidence_turns=[turns[key] for key in evidence_keys],
            memory_view=memory_view)
        routes.append({
            "route_type": "emergent_open_thread",
            "proposal_id": proposal_id,
            "candidate_evidence_turn_keys": evidence_keys,
            "candidate_evidence_turns": [copy.deepcopy(dict(turns[key]))
                                          for key in evidence_keys],
            "prompt": prompt,
            "semantic_status": "model_claimed",
            "evidence_decision": None,
            "formal_thread_id": None,
        })
    return routes
