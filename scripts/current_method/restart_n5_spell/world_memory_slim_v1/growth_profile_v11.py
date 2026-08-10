"""Frozen narrative-growth profiles for dedicated v11 canaries.

A growth profile constrains Planner coverage and post-run admission.  It never
confirms a model claim, manufactures a durable update, or weakens the round
Validator and Binder gates.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


GROWTH_PROFILE_SCHEMA_VERSION = (
    "narrative_growth_profile_v11_v2_same_scene_growth_allowed"
)
PERSON_EVENT_GROWTH_CANARY_PROFILE = {
    "growth_profile_schema_version": GROWTH_PROFILE_SCHEMA_VERSION,
    "profile_id": "person_event_growth_canary_v1",
    "planner_contract": {
        "required_new_action_types": [
            "introduce_connected_person", "introduce_connected_event"],
        "required_event_occurrence_status": "occurred",
        "minimum_distinct_growth_scenes": 1,
        "preferred_distinct_growth_scenes": 2,
        "distinct_scene_distribution_is_advisory": True,
        "preserve_anchor_first_story_design": True,
    },
    "admission_contract": {
        "require_person_growth_ready": True,
        "require_event_growth_ready": True,
        "require_world_growth_ready": True,
        "require_zero_orphan_entities": True,
        "require_zero_partial_atomic_commits": True,
        "require_zero_cross_root_refs": True,
        "require_zero_memory_id_collisions": True,
    },
    "non_authority_statement": (
        "This profile requests narrative coverage only. Round Validator evidence, "
        "Durable Update Compiler checks, Binder allocation, and transaction commit "
        "remain independently authoritative."),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def frozen_person_event_growth_profile() -> dict[str, Any]:
    value = deepcopy(PERSON_EVENT_GROWTH_CANARY_PROFILE)
    value["profile_sha256"] = hashlib.sha256(
        _canonical(value).encode("utf-8")).hexdigest()
    return value


def validate_growth_profile_definition(profile: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if profile.get("growth_profile_schema_version") != GROWTH_PROFILE_SCHEMA_VERSION:
        errors.append("growth_profile_schema_version")
    planner = profile.get("planner_contract")
    admission = profile.get("admission_contract")
    if not isinstance(planner, Mapping):
        errors.append("growth_profile_planner_contract")
    else:
        if planner.get("required_new_action_types") != [
                "introduce_connected_person", "introduce_connected_event"]:
            errors.append("growth_profile_required_action_types")
        if planner.get("required_event_occurrence_status") != "occurred":
            errors.append("growth_profile_event_occurrence_status")
        if int(planner.get("minimum_distinct_growth_scenes") or 0) != 1:
            errors.append("growth_profile_distinct_scene_minimum")
        if int(planner.get("preferred_distinct_growth_scenes") or 0) != 2:
            errors.append("growth_profile_distinct_scene_preference")
        if planner.get("distinct_scene_distribution_is_advisory") is not True:
            errors.append("growth_profile_distinct_scene_advisory")
    if not isinstance(admission, Mapping) or not all(
            value is True for value in admission.values()):
        errors.append("growth_profile_admission_contract")
    value = deepcopy(dict(profile))
    supplied_sha = value.pop("profile_sha256", None)
    expected_sha = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    if supplied_sha is not None and supplied_sha != expected_sha:
        errors.append("growth_profile_sha256")
    return {"valid": not errors, "errors": errors,
            "expected_profile_sha256": expected_sha}


def validate_planner_growth_profile_v11(
    outline: Mapping[str, Any], *, profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Require canary coverage without claiming that growth was realized."""
    errors: list[str] = []
    definition = validate_growth_profile_definition(profile)
    errors.extend(definition["errors"])
    matches: dict[str, list[str]] = {
        "introduce_connected_person": [], "introduce_connected_event": []}
    occurred_event_refs: list[str] = []
    for index, scene in enumerate(outline.get("scenes") or [], 1):
        alias = str(scene.get("scene_alias") or f"s{index}")
        for role in ("primary", "secondary"):
            action = scene.get(f"{role}_action")
            if not isinstance(action, Mapping) or action.get("action_mode") != "new":
                continue
            action_type = str(action.get("action_type") or "")
            if action_type in matches:
                action_ref = f"S{index:02d}.{role}_action"
                matches[action_type].append(action_ref)
                if (action_type == "introduce_connected_event" and
                        (action.get("arguments") or {}).get(
                            "occurrence_status") == "occurred"):
                    occurred_event_refs.append(action_ref)
    for action_type, refs in matches.items():
        if not refs:
            errors.append(f"growth_profile_missing_new_action:{action_type}")
    if matches["introduce_connected_event"] and not occurred_event_refs:
        errors.append("growth_profile_missing_occurred_event")
    scene_numbers = {ref.split(".", 1)[0] for refs in matches.values() for ref in refs}
    planner_contract = profile.get("planner_contract") or {}
    required_scenes = int(planner_contract.get(
        "minimum_distinct_growth_scenes") or 1)
    preferred_scenes = int(planner_contract.get(
        "preferred_distinct_growth_scenes") or 2)
    if len(scene_numbers) < required_scenes:
        errors.append("growth_profile_distinct_growth_scenes")
    return {
        "valid": not errors, "errors": list(dict.fromkeys(errors)),
        "matched_action_refs": matches,
        "occurred_event_action_refs": occurred_event_refs,
        "distinct_growth_scene_count": len(scene_numbers),
        "distinct_scene_distribution_preferred": (
            len(scene_numbers) >= preferred_scenes),
        "distinct_scene_distribution_is_advisory": True,
        "memory_confirmation_performed": False,
    }


def evaluate_growth_profile_admission(
    growth_audit: Mapping[str, Any], *, profile: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "person_growth_ready": growth_audit.get("person_growth_ready") is True,
        "event_growth_ready": growth_audit.get("event_growth_ready") is True,
        "world_growth_ready": growth_audit.get("world_growth_ready") is True,
        "orphan_entity_count_zero": not growth_audit.get("orphan_entity_refs"),
        "partial_atomic_commit_count_zero": not growth_audit.get(
            "partial_atomic_groups"),
        "cross_root_ref_count_zero": not growth_audit.get("cross_root_refs"),
        "memory_id_collision_count_zero": not growth_audit.get(
            "memory_id_collisions"),
    }
    return {
        "growth_profile_schema_version": GROWTH_PROFILE_SCHEMA_VERSION,
        "profile_id": profile.get("profile_id"),
        "checks": checks, "profile_ready": all(checks.values()),
        "confirmation_shopping_allowed": False,
    }


__all__ = [
    "GROWTH_PROFILE_SCHEMA_VERSION", "PERSON_EVENT_GROWTH_CANARY_PROFILE",
    "evaluate_growth_profile_admission", "frozen_person_event_growth_profile",
    "validate_growth_profile_definition", "validate_planner_growth_profile_v11",
]
