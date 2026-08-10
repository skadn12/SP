from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Sequence


GRAPH_EXPANSION_VERSION = "realizer_graph_expansion_v1"
GRAPH_SCENE_CONTRACT_VERSION = "graph_scene_contract_v2"
GRAPH_REALIZER_PROMPT_VERSION = "rolling_dialogue_realizer_graph_v2_error_feedback"
GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY = "coverage_only_v1"
GRAPH_REALIZER_TURN_NORMALIZATION_VERSION = "graph_realizer_turn_id_normalization_v1"
GRAPH_REALIZER_RETRY_FEEDBACK_VERSION = "graph_realizer_error_feedback_v1"
GRAPH_SCENE_HARD_TOKEN_RANGE = (220, 650)
GRAPH_TOTAL_TARGET_TOKEN_RANGE = (4000, 4800)
GRAPH_TOTAL_HARD_MINIMUM_TOKENS = 3800


# The audited restart-n5 outlines all use four ordered four-stage arcs over ten
# scenes. Multiple stages in one scene are local bookkeeping; the model still
# reports each affected event-arc ID once.
_STANDARD_STAGE_SCHEDULE: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 1),),
    ((0, 2), (1, 0)),
    ((0, 3), (1, 1)),
    ((1, 2), (2, 0)),
    ((2, 1),),
    ((1, 3), (2, 2), (2, 3)),
    ((3, 0),),
    ((3, 1), (3, 2)),
    ((3, 3),),
)

_STANDARD_LOCATION_INDEX = (0, 0, 0, 1, 2, 1, 1, 1, 1, 1)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def graph_contract_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_graph_realizer_managed_ids(
        parsed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove only caller-managed turn/session IDs while preserving raw output."""
    normalized = copy.deepcopy(parsed)
    removed_top_level_fields = sorted(
        set(normalized).intersection({"turn_id", "session_id"}))
    for field in removed_top_level_fields:
        normalized.pop(field, None)
    removed_turn_fields: list[dict[str, Any]] = []
    turns = normalized.get("dialogue_turns")
    if isinstance(turns, list):
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            removed = sorted(set(turn).intersection({"turn_id", "session_id"}))
            for field in removed:
                turn.pop(field, None)
            if removed:
                removed_turn_fields.append({"turn_index": index, "fields": removed})
    changed = bool(removed_top_level_fields or removed_turn_fields)
    return normalized, {
        "normalization_version": GRAPH_REALIZER_TURN_NORMALIZATION_VERSION,
        "decision": "normalized_managed_ids" if changed else "no_managed_ids_present",
        "removed_top_level_fields": removed_top_level_fields,
        "removed_turn_fields": removed_turn_fields,
        "semantic_text_modified": False,
    }


def _validate_standard_outline(outline: dict[str, Any]) -> None:
    scenes = outline.get("scene_plan") or []
    arcs = outline.get("planned_event_arcs") or []
    if len(scenes) != len(_STANDARD_STAGE_SCHEDULE):
        raise ValueError("graph_outline_requires_ten_scenes")
    if len(arcs) != 4 or any(len(arc.get("stages") or []) != 4 for arc in arcs):
        raise ValueError("graph_outline_requires_four_four_stage_arcs")
    declared = int((outline.get("target") or {}).get("scene_count") or 0)
    if declared != len(scenes):
        raise ValueError("graph_outline_scene_count_mismatch")
    scene_ids = [scene.get("scene_id") for scene in scenes]
    if scene_ids != [f"S{index}" for index in range(1, len(scenes) + 1)]:
        raise ValueError("graph_outline_scene_ids_not_monotonic")


def compile_graph_scene_contracts(
        outline: dict[str, Any], *, coverage_only: bool = False,
) -> list[dict[str, Any]]:
    _validate_standard_outline(outline)
    scenes = list(outline["scene_plan"])
    arcs = list(outline["planned_event_arcs"])
    locations = list(outline.get("allowed_locations") or [])
    if not locations:
        raise ValueError("graph_outline_has_no_locations")
    target = dict(outline.get("target") or {})
    introductions: dict[str, list[str]] = {}
    for character in outline.get("allowed_new_characters") or []:
        introductions.setdefault(str(character.get("introduction_scene")), []).append(
            str(character.get("name")))
    contracts: list[dict[str, Any]] = []
    for position, (scene, schedule) in enumerate(zip(scenes, _STANDARD_STAGE_SCHEDULE), 1):
        stage_records = []
        for arc_position, stage_position in schedule:
            arc = arcs[arc_position]
            event_id = str(arc["event_id"])
            stage_records.append({
                "stage_id":f"{event_id}:stage_{stage_position + 1:02d}",
                "event_id":event_id,
                "event_summary":str(arc.get("summary") or ""),
                "stage_index":stage_position,
                "stage":str(arc["stages"][stage_position]),
            })
        required_people = list(scene.get("participants") or [])
        for name in introductions.get(str(scene["scene_id"]), []):
            if name not in required_people:
                required_people.append(name)
        declared_range = list(scene.get("acceptable_token_range") or [])
        if len(declared_range) != 2:
            declared_range = [int(scene.get("target_child_tokens") or 420) - 60,
                              int(scene.get("target_child_tokens") or 420) + 60]
        target_token_range = [int(declared_range[0]), int(declared_range[1])]
        hard_token_range = list(GRAPH_SCENE_HARD_TOKEN_RANGE)
        if target_token_range[0] > target_token_range[1]:
            raise ValueError("graph_scene_token_range_cannot_meet_minimum")
        turn_range = list(scene.get("target_turns") or [8, 14])
        if len(turn_range) != 2 or int(turn_range[0]) < 1 or int(turn_range[1]) < int(turn_range[0]):
            raise ValueError("graph_scene_turn_range_invalid")
        location_index = _STANDARD_LOCATION_INDEX[position - 1] % len(locations)
        contract = {
            "contract_version":GRAPH_SCENE_CONTRACT_VERSION,
            "scene_index":position,
            "scene_id":str(scene["scene_id"]),
            "purpose":str(scene.get("purpose") or ""),
            "required_participants":required_people,
            "required_parent_anchor_ids":list(scene.get("required_parent_anchor_ids") or []),
            "session_location":str(locations[location_index]),
            "relative_time":(
                f"chronological scene {position} of {len(scenes)} within "
                f"{outline.get('time_window') or 'the declared expansion window'}"
            ),
            "target_dialogue_tokens":int(scene.get("target_child_tokens") or
                                          target.get("tokens_per_scene") or 420),
            "target_approximate_english_word_range":[270, 310],
            "target_dialogue_token_range":target_token_range,
            "hard_dialogue_token_range":hard_token_range,
            "target_turn_range":[int(turn_range[0]), int(turn_range[1])],
            "hard_turn_range":[max(1, int(turn_range[0]) - 2), int(turn_range[1]) + 4],
            "handoff_requirement":str(scene.get("handoff_requirement") or ""),
            "new_unresolved_hook_policy": (
                "forbidden" if coverage_only else
                "exactly_one" if position == 1 else "forbidden"),
            "stage_accounting_mode": (
                GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY if coverage_only else
                "legacy_completion"),
            "assigned_event_stages":stage_records,
            "assigned_event_arc_ids":list(dict.fromkeys(
                record["event_id"] for record in stage_records)),
            "assigned_event_stage_ids":[record["stage_id"] for record in stage_records],
            "is_final_scene":position == len(scenes),
        }
        contract["contract_sha256"] = graph_contract_sha256(contract)
        contracts.append(contract)
    all_stage_ids = [stage["stage_id"] for contract in contracts
                     for stage in contract["assigned_event_stages"]]
    expected_stage_ids = [f"{arc['event_id']}:stage_{index:02d}"
                          for arc in arcs for index in range(1, 5)]
    if sorted(all_stage_ids) != sorted(expected_stage_ids):
        raise ValueError("graph_scene_contract_stage_coverage_invalid")
    return contracts


def validate_graph_scene_output(parsed: dict[str, Any], contract: dict[str, Any],
                                dialogue_token_count: int) -> list[str]:
    errors: list[str] = []
    turns = parsed.get("dialogue_turns") or []
    speakers = {str(turn.get("speaker")) for turn in turns if isinstance(turn, dict)}
    required_people = set(contract["required_participants"])
    if not required_people <= speakers:
        errors.append("graph_required_participant_missing")
    required_anchors = set(contract["required_parent_anchor_ids"])
    if not required_anchors <= set(parsed.get("used_parent_anchor_ids") or []):
        errors.append("graph_required_parent_anchor_missing")
    if set(parsed.get("advanced_event_arc_ids") or []) != set(
            contract["assigned_event_arc_ids"]):
        errors.append("graph_assigned_event_arcs_incomplete")
    new_hooks = parsed.get("new_unresolved_hooks") or []
    hook_policy = contract["new_unresolved_hook_policy"]
    if hook_policy == "exactly_one" and len(new_hooks) != 1:
        errors.append("graph_scene_hook_policy_violation")
    if hook_policy == "forbidden" and new_hooks:
        errors.append("graph_scene_hook_policy_violation")
    minimum_turns, maximum_turns = contract["hard_turn_range"]
    if not minimum_turns <= len(turns) <= maximum_turns:
        errors.append("graph_scene_turn_count_out_of_range")
    minimum_tokens, maximum_tokens = contract["hard_dialogue_token_range"]
    if not minimum_tokens <= dialogue_token_count <= maximum_tokens:
        errors.append("graph_scene_token_count_out_of_range")
    return sorted(set(errors))


def graph_completion_report(outline: dict[str, Any], contracts: Sequence[dict[str, Any]],
                            sessions: Sequence[dict[str, Any]], child_tokens: int) -> dict[str, Any]:
    expected_scenes = [contract["scene_id"] for contract in contracts]
    accepted_scenes = [session.get("graph_scene_id") for session in sessions]
    expected_stages = {stage_id for contract in contracts
                       for stage_id in contract["assigned_event_stage_ids"]}
    completed_stages = {str(stage_id) for session in sessions
                        for stage_id in session.get("completed_event_stage_ids") or []}
    minimum = int((outline.get("target") or {}).get("minimum_accepted_child_tokens") or 0)
    checks = {
        "all_scenes_accepted":accepted_scenes == expected_scenes,
        "all_event_stages_completed":completed_stages == expected_stages,
        "minimum_child_tokens_reached":child_tokens >= minimum,
        "scene_count_matches_outline":len(sessions) == len(contracts),
    }
    return {"graph_expansion_version":GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version":GRAPH_SCENE_CONTRACT_VERSION,
        "expected_scene_ids":expected_scenes, "accepted_scene_ids":accepted_scenes,
        "expected_event_stage_ids":sorted(expected_stages),
        "completed_event_stage_ids":sorted(completed_stages),
        "minimum_accepted_child_tokens":minimum, "actual_child_tokens":child_tokens,
        "checks":checks, "complete":all(checks.values())}


def graph_scene_coverage_report(
        outline: dict[str, Any], contracts: Sequence[dict[str, Any]],
        sessions: Sequence[dict[str, Any]], child_tokens: int, *,
        hard_minimum_child_tokens: int | None = None,
        target_child_token_range: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Report operational scene coverage without claiming semantic completion."""
    expected_scenes = [contract["scene_id"] for contract in contracts]
    accepted_scenes = [session.get("graph_scene_id") for session in sessions]
    expected_stages = {stage_id for contract in contracts
                       for stage_id in contract["assigned_event_stage_ids"]}
    covered_stages = {str(stage_id) for session in sessions
                      for stage_id in session.get("covered_event_stage_ids") or []}
    declared_minimum = int((outline.get("target") or {}).get(
        "minimum_accepted_child_tokens") or 0)
    minimum = int(hard_minimum_child_tokens if hard_minimum_child_tokens is not None
                  else declared_minimum)
    target_range = list(target_child_token_range or
                        (declared_minimum, int((outline.get("target") or {}).get(
                            "child_token_growth") or declared_minimum)))
    if len(target_range) != 2 or int(target_range[0]) > int(target_range[1]):
        raise ValueError("graph_total_target_token_range_invalid")
    target_range = [int(target_range[0]), int(target_range[1])]
    operational_checks = {
        "all_scenes_accepted": accepted_scenes == expected_scenes,
        "all_event_stages_covered": covered_stages == expected_stages,
        "minimum_child_tokens_reached": child_tokens >= minimum,
        "scene_count_matches_outline": len(sessions) == len(contracts),
    }
    stage_states = [{
        "stage_id": stage_id,
        "assigned": True,
        "covered": stage_id in covered_stages,
        "semantic_status": "unknown_not_audited",
        "authoritative_completed": False,
    } for stage_id in sorted(expected_stages)]
    return {
        "graph_expansion_version": GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version": GRAPH_SCENE_CONTRACT_VERSION,
        "stage_accounting_mode": GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "expected_scene_ids": expected_scenes,
        "accepted_scene_ids": accepted_scenes,
        "expected_event_stage_ids": sorted(expected_stages),
        "covered_event_stage_ids": sorted(covered_stages),
        "authoritative_completed_event_stage_ids": [],
        "stage_states": stage_states,
        "minimum_accepted_child_tokens": minimum,
        "source_declared_minimum_accepted_child_tokens": declared_minimum,
        "target_child_token_range": target_range,
        "target_child_token_range_met": target_range[0] <= child_tokens <= target_range[1],
        "actual_child_tokens": child_tokens,
        "layers": {
            "scene_realization_complete": accepted_scenes == expected_scenes,
            "stage_coverage_complete": covered_stages == expected_stages,
            "semantic_transition_complete": False,
            "handoff_accounting_complete": False,
        },
        "operational_checks": operational_checks,
        "semantic_transition_status": "unknown_not_audited",
        "operational_round_complete": all(operational_checks.values()),
        "complete": all(operational_checks.values()),
    }
