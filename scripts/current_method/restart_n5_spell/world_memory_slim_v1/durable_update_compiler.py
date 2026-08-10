"""Post-review compiler from verified narrative actions to Binder proposals.

Planner-facing narrative actions intentionally have no concrete operation or
one-shot action ID.  This module is the only compatibility boundary where an
accepted, evidenced action may be matched to a frozen Memory affordance and
become an internal proposal.  Existing target/pre-state/version, reference,
deduplication, dependency, and atomic closure checks remain mandatory.
"""

from __future__ import annotations

import json
import hashlib
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .operation_registry import OPERATION_REGISTRY, operation_contract
from .participant_persistence import validate_participant_persistence_reviews
from .relation_registry import RELATION_TYPE_REGISTRY


DURABLE_UPDATE_COMPILER_VERSION = (
    "durable_update_compiler_post_review_v8_round_participant_binding"
)
ALLOW_UNHINTED_DURABLE_UPDATES_DEFAULT = False


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value).strip()
    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(row) for row in value]
    return value


def _proposal_fingerprint(operation: str, targets: Sequence[str], payload: Mapping[str, Any]) -> str:
    value = {"operation": operation, "target_refs": sorted(set(targets)),
             "proposed_content": _normalize(payload)}
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _internal_temporary_ref(
    *, root_id: str, kind: str, candidate_key: str, narrative_alias: str,
) -> str:
    if not root_id or kind not in {"E", "EV"} or not narrative_alias:
        raise ValueError("internal_temporary_ref_input_invalid")
    digest = hashlib.sha256(_canonical({
        "root_id": root_id, "kind": kind, "candidate_key": candidate_key,
        "narrative_alias": narrative_alias,
    }).encode()).hexdigest()[:16]
    return f"{root_id}:TMP:{kind}:DU{digest}"


def _operation_intent(action: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Translate only high-confidence narrative contracts to operation intent.

    A return value of ``(None, {})`` means that the scene may still be retained,
    but the high-level action does not safely imply a durable transaction.
    """
    kind = str(action.get("resolved_action_type") or action.get("action_type") or "")
    args = dict(action.get("resolved_arguments") or action.get("arguments") or {})
    if kind == "introduce_external_constraint":
        return "introduce_external_constraint", {
            "affected_refs": list(args.get("affected_participants") or []),
            "constraint": args.get("constraint"),
        }
    if kind == "add_information_asymmetry":
        return "add_information_asymmetry", {
            "knower_refs": [args.get("informed_participant")],
            "non_knower_refs": [args.get("uninformed_participant")],
            "information": args.get("hidden_information"),
        }
    if kind == "create_misunderstanding":
        return "create_misunderstanding", {
            "participant_refs": [args.get("mistaken_participant")],
            "misunderstanding": (
                f"{args.get('mistaken_belief')} (actual situation: "
                f"{args.get('actual_situation')})"),
        }
    if kind == "create_or_escalate_conflict" and args.get("escalation_or_creation") == "create":
        return "create_conflict", {
            "participant_refs": list(args.get("participants") or []),
            "conflict_description": (
                f"{args.get('conflict_subject')}: {args.get('conflict_source')}"),
        }
    if kind == "develop_character_or_task_state":
        return "create_state", {
            "subject_ref": args.get("target"), "state_category": "custom",
            "state_dimension": args.get("dimension"),
            "initial_value": args.get("direction"),
            "custom_description": args.get("trigger"),
        }
    if kind == "resolve_or_deescalate_conflict":
        return "resolve_conflict", {
            "target_memory_ref": args.get("conflict_ref_or_subject"),
            "resolution": args.get("new_working_arrangement"),
        }
    if kind == "close_or_defer_thread" and args.get("closure_mode") in {"resolved", "cancelled"}:
        return "close_thread", {
            "target_memory_ref": args.get("thread_ref"),
            "closure": f"{args.get('closure_mode')}: {args.get('reason')}",
        }
    if kind == "assign_or_handoff_role" and args.get("from_participant") is not None:
        return "add_role_handoff", {
            "target_memory_ref": args.get("task_or_responsibility"),
            "source_ref": args.get("from_participant"),
            "target_ref": args.get("to_participant"),
            "handoff_description": args.get("handoff_reason"),
        }
    # continue/advance/reveal/connected-person/correction/defer are semantically
    # useful but do not deterministically imply one safe operation here.
    return None, {}


def durable_candidate_action_eligible(action: Any) -> bool:
    """Return whether a reviewed narrative action has a safe compiler route.

    This is deliberately a capability check, not a promise that an update will
    commit.  Evidence, frozen-affordance matching, target/pre-state checks,
    deduplication, and atomic closure still run after the round review.
    """
    if not isinstance(action, Mapping):
        return False
    action_mode = str(action.get("action_mode") or "")
    action_type = str(action.get("resolved_action_type") or
                      action.get("action_type") or "")
    arguments = action.get("resolved_arguments") or action.get("arguments") or {}
    if action_type == "introduce_connected_person":
        return action_mode == "new"
    if action_type == "introduce_connected_event":
        return (action_mode == "new" and isinstance(arguments, Mapping) and
                arguments.get("occurrence_status") == "occurred")
    operation, _ = _operation_intent(action)
    return operation is not None


def _matches_frozen(action: Mapping[str, Any], intended: Mapping[str, Any]) -> bool:
    frozen = action.get("frozen_payload") or {}
    for key, frozen_value in frozen.items():
        intended_value = intended.get(key)
        if isinstance(frozen_value, list) and isinstance(intended_value, list):
            if not set(str(value) for value in frozen_value).issubset(
                    str(value) for value in intended_value):
                return False
        elif intended_value is not None and intended_value != frozen_value:
            return False
    target = action.get("target_memory_ref")
    if target and intended.get("target_memory_ref") != target:
        return False
    return True


def _proposal_from_payload(
    *, operation: str, payload: Mapping[str, Any], scene_ref: str,
    candidate_key: str, action_role: str, narrative_action: Mapping[str, Any],
    evidence_turn_refs: Sequence[str], proposal_index: int,
    connected_memory_refs: Sequence[str] = (), temporary_refs: Sequence[str] = (),
    atomic_group_ref: str | None = None,
    proposal_dependencies: Sequence[str] = (), action_id: str | None = None,
    target_memory_ref: str | None = None, pre_state: Any = None,
    authoritative_version: int | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    contract = operation_contract(operation)
    schema = contract["payload_schema"]
    required = {key for key, kind in schema.items()
                if not str(kind).startswith("optional_")}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - set(schema))
    errors = [*(f"compiled_payload_missing:{key}" for key in missing),
              *(f"compiled_payload_unexpected:{key}" for key in extra)]
    if any(value is None for key, value in payload.items() if key in required):
        errors.append("compiled_payload_required_value_null")
    if operation == "add_relation" and payload.get("relation_type") not in RELATION_TYPE_REGISTRY:
        errors.append("candidate_relation_type_unregistered")
    if errors:
        return None, list(dict.fromkeys(errors))
    targets = sorted(set(str(value) for value in connected_memory_refs if value))
    fingerprint = _proposal_fingerprint(operation, targets, payload)
    proposal_ref = f"{scene_ref}:PDU{proposal_index:02d}"
    return {
        "proposal_ref": proposal_ref,
        "action_id": action_id or f"POST:{candidate_key}:{proposal_index:02d}",
        "operation": operation,
        "operation_class": contract["operation_class"],
        "compiled_payload": deepcopy(dict(payload)),
        "connected_memory_refs": targets,
        "target_memory_ref": target_memory_ref,
        "pre_state": deepcopy(pre_state) if contract["requires_pre_state"] else None,
        "authoritative_version": (authoritative_version
                                  if contract["requires_pre_state"] else None),
        "proposal_fingerprint": fingerprint,
        "temporary_refs": list(temporary_refs),
        "proposal_dependencies": list(proposal_dependencies),
        "atomic_group_ref": atomic_group_ref,
        "proposal_bundle_id": None,
        "lifecycle": {"realization_status": "verified_by_round_validator",
                      "review_decision": "confirmed", "commit_status": "not_submitted"},
        "provenance": {
            "generated_by": DURABLE_UPDATE_COMPILER_VERSION,
            "source_candidate_update_key": candidate_key,
            "source_action_role": action_role,
            "source_action_type": (narrative_action.get("resolved_action_type") or
                                   narrative_action.get("action_type")),
            "source_action_mode": narrative_action.get("action_mode"),
            "source_narrative_action_ref": narrative_action.get("action_ref"),
            "evidence_turn_refs": list(evidence_turn_refs),
        },
    }, []


def _compile_connected_person_bundle(
    *, candidate: Mapping[str, Any], scene_ref: str, action_role: str,
    narrative_action: Mapping[str, Any], evidence_turn_refs: Sequence[str],
    proposal_index: int, derived_scene_presence: str,
    round_participant_ref: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    args = dict(narrative_action.get("resolved_arguments") or
                narrative_action.get("arguments") or {})
    claim = candidate.get("verified_connection")
    if not isinstance(claim, Mapping):
        return [], ["connected_person_verified_connection_missing"]
    if claim.get("verification_status") != "confirmed":
        return [], ["connected_person_connection_not_confirmed"]
    connection_evidence = [str(value) for value in
                           claim.get("evidence_turn_refs") or []]
    if not connection_evidence or any(
            value not in set(evidence_turn_refs) for value in connection_evidence):
        return [], ["connected_person_connection_evidence_invalid"]
    participant_alias = str(args.get("participant_alias") or "")
    display_name = str(args.get("display_name") or "").strip()
    target_ref = str(args.get("connection_target_ref") or "")
    description = str(args.get("connection_description") or "").strip()
    if not participant_alias or not display_name or not target_ref or not description:
        return [], ["connected_person_identity_or_connection_incomplete"]
    target_root = target_ref.split(":", 1)[0]
    if not target_root:
        return [], ["connected_person_target_root_invalid"]
    temporary_ref = _internal_temporary_ref(
        root_id=target_root, kind="E",
        candidate_key=str(candidate.get("candidate_update_key") or ""),
        narrative_alias=participant_alias)
    group = str(candidate.get("atomic_group_key") or
                f"AG:{candidate.get('candidate_update_key')}:person_connection")
    person, person_errors = _proposal_from_payload(
        operation="add_person",
        payload={"display_role": display_name,
                 "scene_presence": derived_scene_presence},
        scene_ref=scene_ref, candidate_key=str(candidate.get("candidate_update_key")),
        action_role=action_role, narrative_action=narrative_action,
        evidence_turn_refs=evidence_turn_refs, proposal_index=proposal_index,
        connected_memory_refs=[target_ref], temporary_refs=list(dict.fromkeys([
            temporary_ref, *([round_participant_ref] if round_participant_ref else []),
        ])),
        atomic_group_ref=group)
    if person_errors or person is None:
        return [], person_errors
    kind = str(claim.get("connection_kind") or "")
    relation_type = claim.get("relation_type")
    operation_payload = {
        "interaction_link": ("add_interaction_link", {
            "source_ref": target_ref, "target_ref": temporary_ref,
            "link_type": "connected_via_existing_person", "description": description}),
        "explicit_relation": ("add_relation", {
            "source_ref": target_ref, "target_ref": temporary_ref,
            "relation_type": relation_type}),
        "event_participant": ("add_event_participant", {
            "event_ref": target_ref, "participant_ref": temporary_ref,
            "participation_description": description}),
        "task_participant": ("add_task_participant", {
            "task_ref": target_ref, "participant_ref": temporary_ref,
            "participation_description": description}),
        "information_source": ("add_information_source", {
            "information_ref": target_ref, "source_ref": temporary_ref,
            "source_description": description}),
        "thread_participant": ("add_thread_participant", {
            "thread_ref": target_ref, "participant_ref": temporary_ref,
            "participation_description": description}),
    }.get(kind)
    if operation_payload is None:
        return [], ["connected_person_connection_kind_unmapped"]
    connection_operation, connection_payload = operation_payload
    connection, connection_errors = _proposal_from_payload(
        operation=connection_operation, payload=connection_payload,
        scene_ref=scene_ref, candidate_key=str(candidate.get("candidate_update_key")),
        action_role=action_role, narrative_action=narrative_action,
        evidence_turn_refs=connection_evidence, proposal_index=proposal_index + 1,
        connected_memory_refs=[target_ref], temporary_refs=[temporary_ref],
        atomic_group_ref=group, proposal_dependencies=[person["proposal_ref"]])
    if connection_errors or connection is None:
        return [], connection_errors
    connection["provenance"]["verified_connection_kind"] = kind
    connection["provenance"]["verified_connection_evidence_turn_refs"] = connection_evidence
    person["provenance"]["source_participant_alias"] = participant_alias
    person["provenance"]["round_participant_ref"] = round_participant_ref
    person["provenance"]["internal_temporary_ref"] = temporary_ref
    connection["provenance"]["source_participant_alias"] = participant_alias
    return [person, connection], []


def _compile_connected_event(
    *, candidate: Mapping[str, Any], scene_ref: str, action_role: str,
    narrative_action: Mapping[str, Any], evidence_turn_refs: Sequence[str],
    proposal_index: int,
    round_participant_bindings: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    args = dict(narrative_action.get("resolved_arguments") or
                narrative_action.get("arguments") or {})
    if args.get("occurrence_status") != "occurred":
        return [], ["connected_event_not_occurred_audit_only"]
    if args.get("narrative_time_relation") not in {"past", "present_completed"}:
        return [], ["connected_event_temporal_status_mismatch_audit_only"]
    connected = args.get("connected_existing") or {}
    connected_ref = str(connected.get("ref") or "") if isinstance(connected, Mapping) else ""
    event_alias = str(args.get("event_alias") or "")
    if not event_alias or not connected_ref.split(":", 1)[0]:
        return [], ["connected_event_alias_or_root_invalid"]
    submitted_participants = [str(value) for value in
                              args.get("participant_refs") or []]
    participant_refs: list[str] = []
    temporary_refs: list[str] = []
    required_candidate_keys: set[str] = set()
    provenance_bindings: list[dict[str, str]] = []
    for participant_ref in submitted_participants:
        binding = round_participant_bindings.get(participant_ref)
        if binding is None:
            participant_refs.append(participant_ref)
            continue
        temporary_ref = str(binding.get("temporary_ref") or "")
        candidate_key = str(binding.get("candidate_update_key") or "")
        if not temporary_ref or not candidate_key:
            return [], ["connected_event_round_participant_binding_invalid"]
        participant_refs.append(temporary_ref)
        temporary_refs.append(temporary_ref)
        required_candidate_keys.add(candidate_key)
        provenance_bindings.append({
            "round_participant_ref": participant_ref,
            "temporary_ref": temporary_ref,
            "source_candidate_update_key": candidate_key,
        })
    submitted_dependencies = {str(value) for value in
                              candidate.get("depends_on_candidate_keys") or []}
    if not required_candidate_keys.issubset(submitted_dependencies):
        return [], ["connected_event_round_participant_dependency_missing"]
    payload = {
        "participant_refs": participant_refs,
        "description": args.get("description"),
        "narrative_time_expression": args.get("narrative_time_expression"),
    }
    proposal, errors = _proposal_from_payload(
        operation="add_event", payload=payload, scene_ref=scene_ref,
        candidate_key=str(candidate.get("candidate_update_key")), action_role=action_role,
        narrative_action=narrative_action, evidence_turn_refs=evidence_turn_refs,
        proposal_index=proposal_index, connected_memory_refs=[connected_ref],
        temporary_refs=temporary_refs,
        atomic_group_ref=(str(candidate["atomic_group_key"])
                          if candidate.get("atomic_group_key") else None))
    if proposal is not None:
        proposal["provenance"]["source_event_alias"] = event_alias
        proposal["provenance"]["round_participant_bindings"] = provenance_bindings
    return ([proposal] if proposal is not None else []), errors


def _compile_internal_proposal(
    *, candidate_key: str, scene_ref: str, action_role: str,
    narrative_action: Mapping[str, Any], action_menu: Mapping[str, Any],
    evidence_turn_refs: Sequence[str], atomic_group_key: str | None,
    proposal_index: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    operation, intended = _operation_intent(narrative_action)
    if operation is None:
        return None, ["narrative_action_has_no_safe_durable_mapping"]
    matches = [row for row in action_menu.get("actions") or []
               if row.get("operation") == operation and _matches_frozen(row, intended)]
    if not matches:
        return None, ["no_frozen_affordance_matches_verified_action"]
    affordance = sorted(matches, key=lambda row: str(row.get("action_id")))[0]
    schema = affordance.get("payload_schema") or operation_contract(operation)["payload_schema"]
    payload = deepcopy(dict(affordance.get("frozen_payload") or {}))
    for key in schema:
        if key not in payload and key in intended:
            payload[key] = deepcopy(intended[key])
    required = {key for key, kind in schema.items()
                if not str(kind).startswith("optional_")}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - set(schema))
    errors = [*(f"compiled_payload_missing:{key}" for key in missing),
              *(f"compiled_payload_unexpected:{key}" for key in extra)]
    if operation == "add_relation" and payload.get("relation_type") not in RELATION_TYPE_REGISTRY:
        errors.append("candidate_relation_type_unregistered")
    if errors:
        return None, errors
    targets = list(affordance.get("connected_memory_refs") or [])
    if affordance.get("target_memory_ref"):
        targets.append(str(affordance["target_memory_ref"]))
    fingerprint = _proposal_fingerprint(operation, targets, payload)
    proposal_ref = f"{scene_ref}:PDU{proposal_index:02d}"
    contract = operation_contract(operation)
    proposal = {
        "proposal_ref": proposal_ref,
        "action_id": affordance.get("action_id"),
        "operation": operation,
        "operation_class": contract["operation_class"],
        "compiled_payload": payload,
        "connected_memory_refs": list(affordance.get("connected_memory_refs") or []),
        "target_memory_ref": affordance.get("target_memory_ref"),
        "pre_state": (deepcopy(affordance.get("current_value"))
                      if contract["requires_pre_state"] else None),
        "authoritative_version": (affordance.get("authoritative_version")
                                  if contract["requires_pre_state"] else None),
        "proposal_fingerprint": fingerprint,
        "temporary_refs": [], "proposal_dependencies": [],
        "atomic_group_ref": atomic_group_key,
        "proposal_bundle_id": None,
        "lifecycle": {"realization_status": "verified_by_round_validator",
                      "review_decision": "confirmed", "commit_status": "not_submitted"},
        "provenance": {
            "generated_by": DURABLE_UPDATE_COMPILER_VERSION,
            "source_candidate_update_key": candidate_key,
            "source_action_role": action_role,
            "source_action_type": (narrative_action.get("resolved_action_type") or
                                   narrative_action.get("action_type")),
            "source_action_mode": narrative_action.get("action_mode"),
            "source_narrative_action_ref": narrative_action.get("action_ref"),
            "evidence_turn_refs": list(evidence_turn_refs),
        },
    }
    return proposal, []


def _compile_internal_proposals(
    *, candidate: Mapping[str, Any], scene_ref: str, action_role: str,
    narrative_action: Mapping[str, Any], action_menu: Mapping[str, Any],
    evidence_turn_refs: Sequence[str], proposal_index: int,
    derived_scene_presence: str, round_participant_ref: str | None,
    round_participant_bindings: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    kind = str(narrative_action.get("resolved_action_type") or
               narrative_action.get("action_type") or "")
    if kind == "introduce_connected_person":
        return _compile_connected_person_bundle(
            candidate=candidate, scene_ref=scene_ref, action_role=action_role,
            narrative_action=narrative_action,
            evidence_turn_refs=evidence_turn_refs, proposal_index=proposal_index,
            derived_scene_presence=derived_scene_presence,
            round_participant_ref=round_participant_ref)
    if kind == "introduce_connected_event":
        return _compile_connected_event(
            candidate=candidate, scene_ref=scene_ref, action_role=action_role,
            narrative_action=narrative_action,
            evidence_turn_refs=evidence_turn_refs, proposal_index=proposal_index,
            round_participant_bindings=round_participant_bindings)
    proposal, errors = _compile_internal_proposal(
        candidate_key=str(candidate.get("candidate_update_key") or ""),
        scene_ref=scene_ref, action_role=action_role,
        narrative_action=narrative_action, action_menu=action_menu,
        evidence_turn_refs=evidence_turn_refs,
        atomic_group_key=(str(candidate["atomic_group_key"])
                          if candidate.get("atomic_group_key") else None),
        proposal_index=proposal_index)
    return ([proposal] if proposal is not None else []), errors


def compile_durable_updates(
    *, candidates: Sequence[Mapping[str, Any]], compiled_round_plan: Mapping[str, Any],
    scene_decision_closure: Mapping[str, Any],
    realized_scenes: Sequence[Mapping[str, Any]], authoritative_memory: Mapping[str, Any],
    planning_state: Mapping[str, Any], action_menu: Mapping[str, Any] | None = None,
    allow_unhinted_durable_updates: bool = False,
) -> dict[str, Any]:
    if allow_unhinted_durable_updates != ALLOW_UNHINTED_DURABLE_UPDATES_DEFAULT:
        raise ValueError("unhinted_durable_updates_feature_not_enabled")
    scene_plan = {str(row.get("scene_ref")): row
                  for row in compiled_round_plan.get("scenes") or []}
    realized = {str(row.get("scene_ref")): row for row in realized_scenes}
    accepted_scenes = set(scene_decision_closure.get("accepted_scene_refs") or [])
    turn_scene = {str(turn.get("turn_ref")): scene_ref
                  for scene_ref, scene in realized.items()
                  for turn in scene.get("dialogue_turns") or []}
    entity_refs = {str(row.get("entity_ref") or row.get("entity_id") or "")
                   for row in planning_state.get("entities") or []}
    records = {str(row.get("memory_ref") or row.get("memory_id") or ""): row
               for row in authoritative_memory.get("authoritative_records") or []}
    active = authoritative_memory.get("active_head_index") or {}
    menu = action_menu or {"actions": []}
    prelim: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_fingerprints: dict[str, str] = {}
    existing_names = {unicodedata.normalize("NFKC", str(row.get("display_name") or ""))
                      .casefold().strip()
                      for row in authoritative_memory.get("entities") or []
                      if str(row.get("display_name") or "").strip()}
    existing_operation_payloads = {
        (str(row.get("operation") or ""), _canonical(_normalize(row.get("payload") or {})))
        for row in authoritative_memory.get("authoritative_records") or []
        if row.get("operation") and isinstance(row.get("payload"), Mapping)
    }
    root_id = str(authoritative_memory.get("root_id") or planning_state.get("root_id") or "")
    alias_to_round_ref = {
        str(alias): str(ref) for alias, ref in
        ((compiled_round_plan.get("round_participant_registry") or {}).get(
            "alias_to_round_ref") or {}).items()
    }
    round_participant_bindings: dict[str, dict[str, str]] = {}
    proposal_serial = 1
    for raw in candidates:
        candidate = deepcopy(dict(raw))
        key = str(candidate.get("candidate_update_key") or "")
        scene_ref = str(candidate.get("source_scene_ref") or "")
        role = str(candidate.get("source_action_role") or "")
        evidence = [str(value) for value in candidate.get("evidence_turn_refs") or []]
        errors: list[str] = []
        if not key or key in seen_keys:
            errors.append("candidate_update_key_invalid")
        seen_keys.add(key)
        if candidate.get("verification_status") != "confirmed":
            errors.append("candidate_not_confirmed")
        if scene_ref not in accepted_scenes or scene_ref not in realized:
            errors.append("candidate_source_scene_not_accepted")
        if not evidence or any(turn_scene.get(ref) != scene_ref for ref in evidence):
            errors.append("candidate_evidence_ref_invalid")
        scene = scene_plan.get(scene_ref) or {}
        narrative_action = scene.get(f"{role}_action") if role in {"primary", "secondary"} else None
        round_ref: str | None = None
        participant_alias = ""
        if not isinstance(narrative_action, Mapping):
            errors.append("candidate_source_action_invalid")
            proposals: list[dict[str, Any]] = []
        else:
            arguments = (narrative_action.get("resolved_arguments") or
                         narrative_action.get("arguments") or {})
            participant_alias = (str(arguments.get("participant_alias") or "")
                                 if isinstance(arguments, Mapping) else "")
            round_ref = alias_to_round_ref.get(participant_alias)
            actual_speaker_refs = {
                str(turn.get("speaker_ref") or "")
                for turn in (realized.get(scene_ref) or {}).get("dialogue_turns") or []
            }
            derived_scene_presence = (
                "speaking" if round_ref and round_ref in actual_speaker_refs
                else "mentioned_only")
            action_type = str(narrative_action.get("resolved_action_type") or
                              narrative_action.get("action_type") or "")
            if (narrative_action.get("action_mode") == "continuation" and
                    action_type in {"introduce_connected_person",
                                    "introduce_connected_event"}):
                proposals = []
                errors.append(
                    "identity_introduction_continuation_not_new_durable_identity")
            else:
                proposals, compile_errors = _compile_internal_proposals(
                    candidate=candidate, scene_ref=scene_ref, action_role=role,
                    narrative_action=narrative_action, action_menu=menu,
                    evidence_turn_refs=evidence, proposal_index=proposal_serial,
                    derived_scene_presence=derived_scene_presence,
                    round_participant_ref=round_ref,
                    round_participant_bindings=round_participant_bindings)
                errors.extend(compile_errors)
        proposal_serial += max(1, len(proposals))
        staged_names: set[str] = set()
        staged_fingerprints: set[str] = set()
        for proposal in proposals:
            operation = str(proposal["operation"])
            if operation not in OPERATION_REGISTRY:
                errors.append("candidate_operation_unregistered")
            payload = proposal.get("compiled_payload") or {}
            temporary_refs = {str(value) for value in proposal.get("temporary_refs") or []}
            subjects = {str(value) for key_name, value in payload.items()
                        for value in ([value] if key_name.endswith("_ref") and isinstance(value, str)
                                      else value if key_name.endswith("_refs") and isinstance(value, list)
                                      else [])}
            if any(ref not in entity_refs and ref not in records and ref not in temporary_refs
                   for ref in subjects):
                errors.append("candidate_subject_or_target_ref_invalid")
            connected = {str(value) for value in
                         proposal.get("connected_memory_refs") or []}
            if any(ref not in entity_refs and ref not in records for ref in connected):
                errors.append("candidate_connected_ref_invalid")
            authoritative_refs = (subjects | connected) - temporary_refs
            if root_id and any(not ref.startswith(root_id + ":")
                               for ref in authoritative_refs):
                errors.append("candidate_cross_root_ref")
            required_target_types = {
                "add_event_participant": ("event_ref", {"event"}),
                "add_task_participant": ("task_ref", {"task", "responsibility"}),
                "add_information_source": ("information_ref", {
                    "knowledge", "information_asymmetry", "third_party_report",
                    "information_source"}),
                "add_thread_participant": ("thread_ref", {"thread"}),
            }
            if operation in required_target_types:
                target_field, record_types = required_target_types[operation]
                target_record = records.get(str(payload.get(target_field) or ""))
                if target_record is None or str(target_record.get("record_type") or "") not in record_types:
                    errors.append("candidate_connection_target_type_invalid")
            if operation in {"add_interaction_link", "add_relation"}:
                source_ref = str(payload.get("source_ref") or "")
                target_ref = str(payload.get("target_ref") or "")
                if source_ref == target_ref or source_ref not in entity_refs or (
                        target_ref not in entity_refs and target_ref not in temporary_refs):
                    errors.append("candidate_person_connection_endpoint_invalid")
            if operation == "add_person":
                name = unicodedata.normalize(
                    "NFKC", str(payload.get("display_role") or "")).casefold().strip()
                if not name or name in existing_names or name in staged_names:
                    errors.append("candidate_person_identity_duplicate")
                else:
                    staged_names.add(name)
            exact_operation_payload = (operation, _canonical(_normalize(payload)))
            if exact_operation_payload in existing_operation_payloads:
                errors.append("candidate_authoritative_operation_duplicate")
            if proposal.get("operation_class") in {"update", "transition"}:
                target = str(proposal.get("target_memory_ref") or "")
                record = records.get(target)
                if record is None:
                    errors.append("candidate_target_memory_missing")
                else:
                    logical = str(record.get("logical_key") or target)
                    if active.get(logical) != target:
                        errors.append("candidate_target_not_active_head")
                    if record.get("value") != proposal.get("pre_state"):
                        errors.append("candidate_pre_state_mismatch")
                    if int(record.get("authoritative_version") or 1) != int(
                            proposal.get("authoritative_version") or 0):
                        errors.append("candidate_authoritative_version_stale")
            fingerprint = str(proposal.get("proposal_fingerprint") or "")
            if fingerprint in seen_fingerprints or fingerprint in staged_fingerprints:
                errors.append("candidate_semantic_or_fingerprint_duplicate")
            else:
                staged_fingerprints.add(fingerprint)
        row = {"candidate": candidate, "compiled_proposals": deepcopy(proposals),
               "errors": list(dict.fromkeys(errors))}
        if errors:
            if ("narrative_action_has_no_safe_durable_mapping" in errors or
                    "connected_event_not_occurred_audit_only" in errors):
                observations.append({"candidate_update_key": key,
                    "disposition": "scene_retained_without_durable_update",
                    "reason": ("connected_event_not_occurred" if
                               "connected_event_not_occurred_audit_only" in errors else
                               "verified_narrative_action_does_not_imply_safe_operation"),
                    "candidate": candidate})
            rejected.append(row)
        else:
            prelim[key] = row
            existing_names.update(staged_names)
            seen_fingerprints.update({fingerprint: key
                                      for fingerprint in staged_fingerprints})
            if round_ref:
                person_proposals = [proposal for proposal in proposals
                                    if proposal.get("operation") == "add_person"]
                if person_proposals:
                    provenance = person_proposals[0].get("provenance") or {}
                    temporary_ref = str(provenance.get(
                        "internal_temporary_ref") or "")
                    if temporary_ref:
                        round_participant_bindings[round_ref] = {
                            "temporary_ref": temporary_ref,
                            "candidate_update_key": key,
                        }

    changed = True
    while changed:
        changed = False
        for key, row in list(prelim.items()):
            dependencies = set(str(value) for value in
                               row["candidate"].get("depends_on_candidate_keys") or [])
            if any(value not in prelim or value == key for value in dependencies):
                row["errors"] = ["candidate_dependency_rejected"]
                rejected.append(row); del prelim[key]; changed = True
    all_groups: dict[str, set[str]] = {}
    accepted_groups: dict[str, set[str]] = {}
    for raw in candidates:
        if raw.get("atomic_group_key"):
            all_groups.setdefault(str(raw["atomic_group_key"]), set()).add(
                str(raw.get("candidate_update_key")))
    for key, row in prelim.items():
        group = row["candidate"].get("atomic_group_key")
        if group:
            accepted_groups.setdefault(str(group), set()).add(key)
    incomplete = {group for group, keys in all_groups.items()
                  if accepted_groups.get(group, set()) != keys}
    for key, row in list(prelim.items()):
        if str(row["candidate"].get("atomic_group_key") or "") in incomplete:
            row["errors"] = ["candidate_atomic_group_incomplete"]
            rejected.append(row); del prelim[key]

    candidate_to_proposals = {
        key: [str(value["proposal_ref"]) for value in row["compiled_proposals"]]
        for key, row in prelim.items()}
    accepted = []
    for key, row in prelim.items():
        candidate = row["candidate"]
        external_dependencies = [proposal_ref
            for value in candidate.get("depends_on_candidate_keys") or []
            for proposal_ref in candidate_to_proposals.get(value, [])]
        for operation_index, proposal in enumerate(row["compiled_proposals"], 1):
            proposal["proposal_dependencies"] = list(dict.fromkeys([
                *proposal.get("proposal_dependencies", []), *external_dependencies]))
            accepted.append({
                "candidate_update_key": key,
                "compiled_operation_index": operation_index,
                "compiled_proposal": proposal,
                "source_scene_refs": [candidate["source_scene_ref"]],
                "source_action_role": candidate["source_action_role"],
                "evidence_turn_refs": list(proposal.get("provenance", {}).get(
                    "evidence_turn_refs") or candidate["evidence_turn_refs"]),
                "verification_status": "confirmed", "hinted_proposal_refs": [],
            })
    return {
        "durable_update_compiler_version": DURABLE_UPDATE_COMPILER_VERSION,
        "planner_action_allocation_used": False,
        "post_review_internal_operation_compilation": True,
        "allow_unhinted_durable_updates": False,
        "accepted_updates": accepted, "rejected_updates": rejected,
        "audit_only_observations": observations,
        "binder_allocation_candidate_count": len(accepted),
        "transaction_candidate_count": len(accepted),
        "accepted_source_candidate_count": len(prelim),
        "accepted_internal_proposal_count": len(accepted),
    }


def compile_participant_persistence_updates(
    *, reviews: Sequence[Mapping[str, Any]],
    round_participant_registry: Mapping[str, Any],
    realized_scenes: Sequence[Mapping[str, Any]],
    source_turn_registry: Mapping[str, Any],
    authoritative_memory: Mapping[str, Any],
    accepted_scene_refs: Sequence[str],
) -> dict[str, Any]:
    """Compile reviewed speakers into atomic person-plus-connection proposals."""
    validation = validate_participant_persistence_reviews(
        reviews,
        round_participant_registry=round_participant_registry,
        realized_scenes=realized_scenes,
        source_turn_registry=source_turn_registry,
        authoritative_memory=authoritative_memory,
        accepted_scene_refs=accepted_scene_refs,
    )
    empty = {
        "participant_review_validation": validation,
        "compiled_proposals": [], "accepted_updates": [],
        "audit_only_reviews": list(reviews),
        "orphan_person_count": 0,
        "partial_atomic_commit_count": 0,
    }
    if not validation["valid"]:
        return {"valid": False, "errors": validation["errors"], **empty}
    participant_by_ref = {
        str(row.get("participant_ref") or ""): row
        for row in round_participant_registry.get("participants") or []}
    scene_by_alias = {
        str(row.get("scene_alias") or ""): str(row.get("scene_ref") or "")
        for row in realized_scenes}
    normalized_by_ref = {
        str(row.get("participant_ref") or ""): row
        for row in validation["normalized_reviews"]}
    records = {
        str(row.get("memory_ref") or row.get("memory_id") or ""): row
        for row in authoritative_memory.get("authoritative_records") or []}
    entities = {
        str(row.get("entity_ref") or row.get("entity_id") or "")
        for row in authoritative_memory.get("entities") or []}
    counters: dict[str, int] = {}
    compiled: list[dict[str, Any]] = []
    accepted_updates: list[dict[str, Any]] = []
    errors: list[str] = []
    for participant_ref in validation["persistable_participant_refs"]:
        participant = participant_by_ref[participant_ref]
        review = normalized_by_ref[participant_ref]
        connection = review["verified_connection"]
        first_scene = scene_by_alias[str(participant["first_scene_alias"])]
        proposal_index = counters.get(first_scene, 1)
        counters[first_scene] = proposal_index + 2
        group = f"AG:{participant_ref}:identity_connection"
        candidate_key = f"participant_persistence:{participant_ref}"
        root_id = str(authoritative_memory.get("root_id") or "")
        binder_temporary_ref = _internal_temporary_ref(
            root_id=root_id, kind="E", candidate_key=candidate_key,
            narrative_alias=str(participant["participant_alias"]))
        narrative_identity = {
            "action_ref": f"{participant_ref}.persistence_review",
            "action_mode": "post_review",
            "action_type": "participant_persistence",
        }
        person, person_errors = _proposal_from_payload(
            operation="add_person",
            payload={"display_role": participant["display_name"],
                     "scene_presence": "speaking"},
            scene_ref=first_scene, candidate_key=candidate_key,
            action_role="participant_persistence",
            narrative_action=narrative_identity,
            evidence_turn_refs=[str(value) for value in
                                review["current_scene_evidence_turn_refs"]],
            proposal_index=proposal_index,
            connected_memory_refs=[str(connection["target_ref"])],
            temporary_refs=[participant_ref, binder_temporary_ref], atomic_group_ref=group,
        )
        if person is None:
            errors.extend(f"{participant_ref}:{value}" for value in person_errors)
            continue
        target_ref = str(connection["target_ref"])
        kind = str(connection["connection_kind"])
        description = str(connection["description"])
        mapped = {
            "interaction_link": ("add_interaction_link", {
                "source_ref": target_ref, "target_ref": binder_temporary_ref,
                "link_type": "dialogue_verified_connection",
                "description": description}),
            "explicit_relation": ("add_relation", {
                "source_ref": target_ref, "target_ref": binder_temporary_ref,
                "relation_type": connection.get("relation_type")}),
            "event_participant": ("add_event_participant", {
                "event_ref": target_ref, "participant_ref": binder_temporary_ref,
                "participation_description": description}),
            "task_participant": ("add_task_participant", {
                "task_ref": target_ref, "participant_ref": binder_temporary_ref,
                "participation_description": description}),
            "information_source": ("add_information_source", {
                "information_ref": target_ref, "source_ref": binder_temporary_ref,
                "source_description": description}),
            "thread_participant": ("add_thread_participant", {
                "thread_ref": target_ref, "participant_ref": binder_temporary_ref,
                "participation_description": description}),
        }[kind]
        operation, payload = mapped
        if kind in {"interaction_link", "explicit_relation"} and target_ref not in entities:
            errors.append(f"{participant_ref}:connection_target_not_entity")
            continue
        allowed_types = {
            "event_participant": {"event"},
            "task_participant": {"task", "responsibility"},
            "information_source": {"knowledge", "information_asymmetry",
                                   "third_party_report", "information_source"},
            "thread_participant": {"thread"},
        }
        if kind in allowed_types and str(
                (records.get(target_ref) or {}).get("record_type") or "") not in allowed_types[kind]:
            errors.append(f"{participant_ref}:connection_target_type_invalid")
            continue
        connection_proposal, connection_errors = _proposal_from_payload(
            operation=operation, payload=payload, scene_ref=first_scene,
            candidate_key=candidate_key, action_role="participant_persistence",
            narrative_action=narrative_identity,
            evidence_turn_refs=[str(value) for value in
                                connection["evidence_turn_refs"]],
            proposal_index=proposal_index + 1,
            connected_memory_refs=[target_ref],
            temporary_refs=[participant_ref, binder_temporary_ref],
            atomic_group_ref=group, proposal_dependencies=[person["proposal_ref"]],
        )
        if connection_proposal is None:
            errors.extend(f"{participant_ref}:{value}" for value in connection_errors)
            continue
        for proposal in (person, connection_proposal):
            proposal["provenance"]["source_participant_ref"] = participant_ref
            proposal["provenance"]["dialogue_presence"] = "speaking"
            proposal["provenance"]["legacy_operation_scene_presence_projection"] = (
                "post-review dialogue-first extension; v1 registry file remains unchanged")
            proposal["provenance"]["participant_persistence_review_version"] = (
                validation["participant_persistence_review_version"])
            compiled.append(proposal)
            accepted_updates.append({
                "candidate_update_key": candidate_key,
                "source_scene_ref": first_scene,
                "compiled_proposal": deepcopy(proposal),
                "atomic_group_key": group,
            })
    if errors:
        return {"valid": False, "errors": list(dict.fromkeys(errors)), **empty}
    persistable = set(validation["persistable_participant_refs"])
    return {
        "valid": True, "errors": [],
        "participant_review_validation": validation,
        "compiled_proposals": compiled,
        "accepted_updates": accepted_updates,
        "audit_only_reviews": [row for row in reviews if str(
            row.get("participant_ref") or "") not in persistable],
        "orphan_person_count": 0,
        "partial_atomic_commit_count": 0,
        "atomic_group_count": len(persistable),
    }
