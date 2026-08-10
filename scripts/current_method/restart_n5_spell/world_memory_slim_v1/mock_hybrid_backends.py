"""Deterministic, network-free Planner/Realizer/Round Validator fixtures."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

MOCK_HYBRID_BACKEND_VERSION = "mock_hybrid_backends_v1"
MOCK_CASES = frozenset({
    "all_accept", "accept_no_updates", "reject_middle_scene",
    "uncertain_middle_scene", "independent_parallel_branch_survives",
    "validator_parse_invalid_then_valid", "validator_invalid_ref_then_valid",
    "validator_legal_reject_no_retry", "one_valid_one_invalid_update",
    "atomic_group_partial_failure", "unhinted_update_observation",
    "primary_action_not_realized", "secondary_action_partial",
    "continuation_repeats_source", "continuation_source_rejected",
})


def frozen_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class DeterministicHybridMock:
    def __init__(self, *, case: str, planner_output: Mapping[str, Any]):
        if case not in MOCK_CASES:
            raise ValueError(f"unknown_mock_hybrid_case:{case}")
        self.case = case
        self.planner_output = deepcopy(dict(planner_output))
        self.calls = {"planner": 0, "realizer": 0, "round_validator": 0,
                      "evidence_committer": 0}
        self.trace: list[dict[str, Any]] = []

    def planner(self, frozen_input: Mapping[str, Any]) -> dict[str, Any]:
        self.calls["planner"] += 1
        value = deepcopy(self.planner_output)
        self._record("planner", frozen_input, value)
        return value

    def realizer(self, *, scene: Mapping[str, Any], context: Mapping[str, Any],
                 turn_budget: Mapping[str, Any]) -> dict[str, Any]:
        self.calls["realizer"] += 1
        ref, alias = str(scene["scene_ref"]), str(scene["scene_alias"])
        speakers = [str(row["participant_ref"])
                    for row in scene.get("speaker_participants") or []]
        count = int(turn_budget["target"])
        turns = []
        for index in range(1, count + 1):
            turns.append({
                "turn_ref": f"{ref}:T{index:03d}",
                "speaker_ref": speakers[(index - 1) % len(speakers)],
                "text": (f"In {alias}, speaker {(index - 1) % len(speakers) + 1} "
                         f"adds concrete development {index} while preserving the "
                         "established situation and the previous scene's consequence."),
            })
        value = {"scene_ref": ref, "scene_alias": alias,
                 "dialogue_turns": turns}
        self._record("realizer", {"scene": scene, "context": context,
                                  "turn_budget": turn_budget}, value)
        return value

    def validator(self, *, prompt: str, attempt: int,
                  realized_scenes: Sequence[Mapping[str, Any]],
                  compiled_plan: Mapping[str, Any]) -> Any:
        self.calls["round_validator"] += 1
        valid = make_validator_output(self.case, realized_scenes, compiled_plan)
        if self.case == "validator_parse_invalid_then_valid" and attempt == 1:
            value: Any = "{not-json"
        elif self.case == "validator_invalid_ref_then_valid" and attempt == 1:
            value = deepcopy(valid)
            value["scene_reviews"][0]["scene_alias"] = "S999"
        else:
            value = valid
        self._record("round_validator", {"prompt": prompt, "attempt": attempt}, value)
        return value

    def _record(self, role: str, request: Any, response: Any) -> None:
        self.trace.append({
            "sequence": len(self.trace) + 1, "role": role,
            "request_sha256": frozen_sha(request),
            "response_sha256": frozen_sha(response),
            "network_attempts": 0, "api_calls": 0,
        })


def make_validator_output(
    case: str, realized_scenes: Sequence[Mapping[str, Any]],
    compiled_plan: Mapping[str, Any],
) -> dict[str, Any]:
    decision_by_alias = {str(row["scene_alias"]): "accept" for row in realized_scenes}
    middle = str(realized_scenes[1]["scene_alias"]) if len(realized_scenes) > 1 else ""
    if case in {"reject_middle_scene", "independent_parallel_branch_survives",
                "validator_legal_reject_no_retry"}:
        decision_by_alias[middle] = "reject"
    if case == "uncertain_middle_scene":
        decision_by_alias[middle] = "uncertain"
    if case == "primary_action_not_realized":
        decision_by_alias[middle] = "reject"
    continuation_aliases = []
    for scene in realized_scenes:
        for action in (scene.get("primary_action"), scene.get("secondary_action")):
            if isinstance(action, Mapping) and action.get("action_mode") == "continuation":
                continuation_aliases.append(str(scene["scene_alias"]))
                if case == "continuation_source_rejected":
                    source_index = int(str(action["continuation_of"])[1:3]) - 1
                    source_alias = str((compiled_plan.get("scenes") or [])[source_index][
                        "scene_alias"])
                    decision_by_alias[source_alias] = "reject"
                    decision_by_alias[str(scene["scene_alias"])] = "reject"
    if case == "continuation_repeats_source":
        for alias in continuation_aliases:
            decision_by_alias[alias] = "reject"

    def action_review(action: Mapping[str, Any], *, status: str,
                      evidence_turn_ref: str | None) -> dict[str, Any]:
        evidence = [evidence_turn_ref] if evidence_turn_ref else []
        value = {"status": status, "evidence_turn_refs": evidence,
                 "reason": "deterministic mock action review"}
        if action.get("action_mode") == "continuation":
            value["progression_evidence_turn_refs"] = list(evidence)
        return value

    reviews = []
    for scene in realized_scenes:
        alias, ref = str(scene["scene_alias"]), str(scene["scene_ref"])
        decision = decision_by_alias[alias]
        primary_is_blocked_continuation = (
            case == "continuation_source_rejected" and
            (scene.get("primary_action") or {}).get("action_mode") == "continuation")
        secondary_is_blocked_continuation = (
            case == "continuation_source_rejected" and
            (scene.get("secondary_action") or {}).get("action_mode") == "continuation")
        uses = list(scene.get("predecessor_scene_refs") or [])
        if case == "independent_parallel_branch_survives" and scene.get(
                "branch_role") == "event_branch":
            uses = []
        primary_status = ("not_realized" if (
            case == "primary_action_not_realized" and alias == middle) or (
            case == "continuation_repeats_source" and
            (scene.get("primary_action") or {}).get("action_mode") == "continuation") or
            primary_is_blocked_continuation else "realized")
        primary_evidence = (None if primary_status == "not_realized" else
                            str(scene["dialogue_turns"][0]["turn_ref"]))
        secondary_status = ("not_realized" if secondary_is_blocked_continuation else
                            "partially_realized" if case == "secondary_action_partial" and
                            alias == middle else "realized")
        secondary_evidence = (None if secondary_is_blocked_continuation else
                              str(scene["dialogue_turns"][1]["turn_ref"]))
        reviews.append({
            "scene_alias": alias, "decision": decision,
            "historical_connection_status": "valid" if decision == "accept" else "uncertain",
            "continuity_status": "valid" if decision == "accept" else "uncertain",
            "future_knowledge_leak": False,
            "historical_consistency_status": "valid",
            "semantic_duplicate_status": "not_duplicate",
            "new_development_status": "valid" if decision == "accept" else "uncertain",
            "reason_codes": (["local_gate_passed", "anchor_supported"]
                             if decision == "accept" else ["uncertain_review"]),
            "historical_supporting_turn_refs": [],
            "current_scene_supporting_turn_refs": [
                str(scene["dialogue_turns"][0]["turn_ref"])],
            "dependency_context_turn_refs": [],
            "uses_scene_refs": uses,
            "scene_requires_update_commit": False,
            "primary_action_realization": action_review(
                scene.get("primary_action") or {}, status=primary_status,
                evidence_turn_ref=primary_evidence),
            "secondary_action_realization": (action_review(
                scene.get("secondary_action") or {}, status=secondary_status,
                evidence_turn_ref=secondary_evidence)
                if scene.get("secondary_action") else None),
        })
        if case == "primary_action_not_realized" and alias == middle:
            reviews[-1]["decision"] = "reject"
            reviews[-1]["reason_codes"] = ["primary_action_not_realized"]
        if (case == "continuation_repeats_source" and
                (scene.get("primary_action") or {}).get("action_mode") == "continuation"):
            reviews[-1]["decision"] = "reject"
            reviews[-1]["reason_codes"] = ["primary_action_not_realized"]
    scene_by_ref = {str(row["scene_ref"]): row for row in realized_scenes}
    updates = []
    if case not in {"accept_no_updates", "reject_middle_scene",
                    "uncertain_middle_scene", "independent_parallel_branch_survives",
                    "validator_legal_reject_no_retry", "unhinted_update_observation"}:
        for scene in compiled_plan.get("scenes") or []:
            ref = str(scene.get("scene_ref"))
            if ref not in scene_by_ref or decision_by_alias.get(
                    str(scene.get("scene_alias"))) != "accept":
                continue
            action_type = str((scene.get("primary_action") or {}).get(
                "resolved_action_type") or "")
            if action_type not in {"introduce_external_constraint", "add_information_asymmetry",
                                   "create_misunderstanding", "create_or_escalate_conflict",
                                   "develop_character_or_task_state",
                                   "resolve_or_deescalate_conflict", "close_or_defer_thread",
                                   "assign_or_handoff_role", "introduce_connected_person",
                                   "introduce_connected_event"}:
                continue
            action = scene.get("primary_action") or {}
            arguments = action.get("resolved_arguments") or action.get("arguments") or {}
            verified_connection = None
            if action_type == "introduce_connected_person":
                kind_by_target = {
                    "existing_person": "interaction_link",
                    "existing_event": "event_participant",
                    "existing_task": "task_participant",
                    "information_source": "information_source",
                    "open_thread": "thread_participant",
                }
                verified_connection = {
                    "verification_status": "confirmed",
                    "connection_kind": kind_by_target.get(
                        str(arguments.get("connection_type") or ""),
                        "interaction_link"),
                    "relation_type": None,
                    "evidence_turn_refs": [
                        scene_by_ref[ref]["dialogue_turns"][0]["turn_ref"]],
                    "reason": "deterministic mock verifies the planned typed connection",
                }
            update = {
                "candidate_update_key": f"DU{len(updates) + 1:03d}",
                "source_scene_ref": ref, "source_action_role": "primary",
                "evidence_turn_refs": [scene_by_ref[ref]["dialogue_turns"][0]["turn_ref"]],
                "verification_status": "confirmed", "atomic_group_key": None,
                "depends_on_candidate_keys": [],
            }
            if verified_connection is not None:
                update["verified_connection"] = verified_connection
            updates.append(update)
            if len(updates) >= 2:
                break
    if case == "one_valid_one_invalid_update" and len(updates) > 1:
        updates[1]["verification_status"] = "rejected"
    if case == "atomic_group_partial_failure" and len(updates) > 1:
        updates[0]["atomic_group_key"] = "AG001"
        updates[1]["atomic_group_key"] = "AG001"
        updates[1]["verification_status"] = "rejected"
    observations = []
    if case == "unhinted_update_observation":
        scene = realized_scenes[0]
        observations.append({
            "observation_key": "OBS001", "update_type": "add_relation",
            "source_scene_refs": [scene["scene_ref"]],
            "evidence_turn_refs": [scene["dialogue_turns"][0]["turn_ref"]],
            "description": "A possible unhinted relation requires a future explicit proposal.",
            "disposition": "audit_only_unhinted_update",
        })
    rejected = any(value != "accept" for value in decision_by_alias.values())
    return {
        "round_decision": "partial_accept" if rejected else "accept",
        "round_reason_codes": ["partial_scene_rejection" if rejected else
                               ("no_durable_update" if not updates else "supported")],
        "scene_reviews": reviews, "durable_updates": updates,
        "semantic_review_observations": observations,
    }
