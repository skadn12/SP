"""Continuous-expansion context compiler with non-droppable continuity text.

P0 and P1 are structural requirements.  Only P2 may be trimmed.  Dialogue
that lies after a before/between insertion boundary is labelled as a future
constraint and must not be presented as knowledge already available in-scene.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .placement_choice_registry import (
    resolve_placement_choice, resolve_source_anchor, validate_placement_choice,
)


CONTEXT_MANIFEST_VERSION_V2 = "context_manifest_v2"
DEFAULT_CONTEXT_TOKEN_BUDGET_V2 = 12_000
DEFAULT_PARENT_TAIL_TURN_COUNT = 7


class ContextInvariantFailure(RuntimeError):
    """Raised when required context alone cannot fit the frozen budget."""


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _tokens(value: Any) -> int:
    return max(1, (len(json.dumps(value, ensure_ascii=False)) + 3) // 4)


def _compact_turn(row: Mapping[str, Any], *, role: str, priority: str) -> dict[str, Any]:
    return {
        "turn_ref": str(row.get("turn_ref") or row.get("turn_id") or ""),
        "session_ref": str(row.get("session_ref") or row.get("session_id") or ""),
        "scene_ref": row.get("scene_ref"),
        "speaker_ref": row.get("speaker_ref"),
        "speaker_label": row.get("speaker_label") or row.get("speaker"),
        "text": str(row.get("text") or ""),
        "knowledge_role": role,
        "context_priority": priority,
    }


def _scene_turns(scene: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    turns = scene.get("dialogue_turns")
    if not isinstance(turns, list) or not turns:
        raise ContextInvariantFailure(
            f"same_round_predecessor_dialogue_missing:{scene.get('scene_ref')}")
    scene_ref = str(scene.get("scene_ref") or "")
    return [{**dict(row), "scene_ref": row.get("scene_ref") or scene_ref}
            for row in turns if isinstance(row, Mapping)]


def compile_scene_context_v2(
    *, scene: Mapping[str, Any], complete_dialogue: Mapping[str, Any],
    source_anchor_registry: Mapping[str, Any],
    placement_choice_registry: Mapping[str, Any],
    planning_state: Mapping[str, Any],
    same_round_scenes: Mapping[str, Mapping[str, Any]] | None = None,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET_V2,
    parent_tail_turn_count: int = DEFAULT_PARENT_TAIL_TURN_COUNT,
) -> dict[str, Any]:
    """Compile frozen context for one scene in a continuous round.

    Direct predecessors are included verbatim in P0.  Additional dependency
    scenes are included verbatim in P1.  Required text is never silently
    truncated: an undersized budget is a local invariant failure.
    """
    root_id = str(complete_dialogue.get("root_id") or "")
    if not root_id or planning_state.get("root_id") != root_id:
        raise ContextInvariantFailure("context_v2_root_mismatch")
    scene_ref = str(scene.get("scene_ref") or "")
    source_anchor_id = str(scene.get("source_anchor_id") or "")
    placement_choice_id = str(scene.get("placement_choice_id") or "")
    source_anchor = resolve_source_anchor(source_anchor_registry, source_anchor_id)
    placement_choice = resolve_placement_choice(
        placement_choice_registry, placement_choice_id)
    choice_validation = validate_placement_choice(
        placement_choice, complete_dialogue=complete_dialogue)
    if not choice_validation["valid"]:
        raise ContextInvariantFailure(
            "context_v2_placement_choice:" +
            ",".join(choice_validation["errors"]))
    if scene.get("source_anchor") != source_anchor:
        raise ContextInvariantFailure("context_v2_source_anchor_drift")
    if scene.get("placement_choice") != placement_choice:
        raise ContextInvariantFailure("context_v2_placement_choice_drift")
    placement = str(placement_choice["placement"])

    all_turns = [row for row in complete_dialogue.get("turns") or []
                 if isinstance(row, Mapping)]
    positions = {str(row.get("turn_ref")): index for index, row in enumerate(all_turns)}
    parent_turns = [row for row in all_turns
                    if row.get("source_partition") == "original_parent"]
    p0: list[dict[str, Any]] = []
    p1: list[dict[str, Any]] = []
    p2_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(rows: Sequence[Mapping[str, Any]], target: list[dict[str, Any]],
            *, role: str, priority: str) -> None:
        for row in rows:
            ref = str(row.get("turn_ref") or row.get("turn_id") or "")
            if not ref or ref in seen:
                continue
            if not str(row.get("text") or "").strip():
                raise ContextInvariantFailure(f"context_v2_empty_turn:{ref}")
            seen.add(ref)
            target.append(_compact_turn(row, role=role, priority=priority))

    anchor_rows = [all_turns[positions[ref]] for ref in
                   source_anchor.get("source_turn_refs") or []
                   if ref in positions]
    if not anchor_rows:
        raise ContextInvariantFailure("context_v2_anchor_text_missing")
    future_refs = set(placement_choice["following_constraint_turn_refs"])
    add([row for row in anchor_rows if str(row["turn_ref"]) not in future_refs],
        p0, role="preceding_known_context", priority="P0_source_anchor_exact")
    add([row for row in anchor_rows if str(row["turn_ref"]) in future_refs],
        p0, role="following_context_constraint_only",
        priority="P0_source_anchor_exact")

    tail = parent_turns[-min(parent_tail_turn_count, len(parent_turns)):]
    add([row for row in tail if str(row.get("turn_ref")) not in future_refs],
        p0, role="preceding_known_context", priority="P0_parent_true_tail")
    add([row for row in tail if str(row.get("turn_ref")) in future_refs],
        p0, role="following_context_constraint_only", priority="P0_parent_true_tail")

    same_round_scenes = same_round_scenes or {}
    declared_predecessors = [str(value) for value in
                             scene.get("predecessor_scene_refs") or []]
    hard_predecessors = {
        str(edge.get("predecessor_scene_ref") or "")
        for edge in scene.get("dependency_edges") or []
        if isinstance(edge, Mapping) and edge.get("hard_dependency")
    }
    unavailable_soft_predecessors: list[str] = []
    predecessors: list[str] = []
    for predecessor in declared_predecessors:
        if predecessor in same_round_scenes:
            predecessors.append(predecessor)
        elif predecessor in hard_predecessors:
            raise ContextInvariantFailure(
                f"context_v2_predecessor_not_generated:{predecessor}")
        else:
            unavailable_soft_predecessors.append(predecessor)
    declared_dependency_refs = [str(value) for value in
                                scene.get("dependency_scene_refs") or []]
    unavailable_soft_dependencies: list[str] = []
    dependency_refs: list[str] = []
    for dependency in declared_dependency_refs:
        if dependency in same_round_scenes:
            dependency_refs.append(dependency)
        elif dependency in hard_predecessors:
            raise ContextInvariantFailure(
                f"context_v2_dependency_not_generated:{dependency}")
        else:
            unavailable_soft_dependencies.append(dependency)
    continuation_source_refs: list[str] = []
    round_prefix = scene_ref.split(":S", 1)[0]
    for action in (scene.get("primary_action"), scene.get("secondary_action")):
        if not isinstance(action, Mapping) or action.get("action_mode") != "continuation":
            continue
        source_action_ref = str(action.get("continuation_of") or "")
        if len(source_action_ref) >= 3 and source_action_ref[1:3].isdigit():
            source_scene_ref = f"{round_prefix}:S{source_action_ref[1:3]}"
            if source_scene_ref not in continuation_source_refs:
                continuation_source_refs.append(source_scene_ref)
    for predecessor in predecessors:
        add(_scene_turns(same_round_scenes[predecessor]), p0,
            role="same_round_predecessor_dialogue", priority="P0_direct_predecessor")
    for dependency in dependency_refs:
        if dependency in predecessors:
            continue
        add(_scene_turns(same_round_scenes[dependency]), p1,
            role="same_round_dependency_dialogue", priority="P1_dependency")
    for source_ref in continuation_source_refs:
        if source_ref in predecessors or source_ref in dependency_refs:
            continue
        if source_ref not in same_round_scenes:
            raise ContextInvariantFailure(
                f"context_v2_continuation_source_not_generated:{source_ref}")
        add(_scene_turns(same_round_scenes[source_ref]), p1,
            role="same_round_dependency_dialogue",
            priority="P1_continuation_source_actual_dialogue")

    predecessor_dialogue_sha256 = {
        predecessor: _sha([{
            "turn_ref": str(row.get("turn_ref") or ""),
            "speaker_ref": row.get("speaker_ref"),
            "text": str(row.get("text") or ""),
        } for row in _scene_turns(same_round_scenes[predecessor])])
        for predecessor in predecessors
    }

    # Other historical dialogue is optional P2.  Narrative order is preserved.
    add(all_turns, p2_candidates, role="supplemental_history", priority="P2_history")
    planning_projection = {
        key: deepcopy(value) for key, value in planning_state.items()
        if key not in {"planning_state_sha256"}
    }
    required_payload = {"p0": p0, "p1": p1,
                        "planning_state": planning_projection}
    required_tokens = _tokens(required_payload)
    if required_tokens > token_budget:
        raise ContextInvariantFailure(
            f"context_v2_required_context_overflow:{required_tokens}>{token_budget}")
    selected_p2: list[dict[str, Any]] = []
    dropped_p2: list[str] = []
    used_tokens = required_tokens
    for row in p2_candidates:
        cost = _tokens(row)
        if used_tokens + cost <= token_budget:
            selected_p2.append(row)
            used_tokens += cost
        else:
            dropped_p2.append(str(row["turn_ref"]))

    selected = [*p0, *p1, *selected_p2]
    result = {
        "context_manifest_version": CONTEXT_MANIFEST_VERSION_V2,
        "root_id": root_id,
        "scene_ref": scene_ref,
        "source_anchor_id": source_anchor_id,
        "placement_choice_id": placement_choice_id,
        "placement": placement,
        "placement_boundary": {
            "after_turn_ref": placement_choice["after_turn_ref"],
            "before_turn_ref": placement_choice["before_turn_ref"],
        },
        "p0_required_dialogue": p0,
        "p1_dependency_dialogue": p1,
        "p1_planning_state": planning_projection,
        "p2_optional_history": selected_p2,
        "selected_context_turn_refs": [row["turn_ref"] for row in selected],
        "p0_required_turn_refs": [row["turn_ref"] for row in p0],
        "same_round_predecessor_refs": predecessors,
        "declared_same_round_predecessor_refs": declared_predecessors,
        "unavailable_soft_predecessor_refs": unavailable_soft_predecessors,
        "same_round_dependency_refs": dependency_refs,
        "declared_same_round_dependency_refs": declared_dependency_refs,
        "unavailable_soft_dependency_refs": unavailable_soft_dependencies,
        "continuation_source_scene_refs": continuation_source_refs,
        "predecessor_dialogue_sha256": predecessor_dialogue_sha256,
        "same_round_scene_text_included": bool(predecessors or dependency_refs),
        "parent_true_tail_refs": [str(row.get("turn_ref")) for row in tail],
        "parent_true_tail_fully_retained": all(
            str(row.get("turn_ref")) in {item["turn_ref"] for item in p0}
            for row in tail),
        "dropped_p2_turn_refs": dropped_p2,
        "p0_p1_token_estimate": required_tokens,
        "token_estimate": used_tokens,
        "token_budget": token_budget,
        "trimming_policy": "P2_only",
        "future_knowledge_is_constraint_only": placement in {
            "before", "between", "parallel"},
        "preceding_known_context_refs": [row["turn_ref"] for row in p0
                                          if row["knowledge_role"] ==
                                          "preceding_known_context"],
        "following_context_constraint_refs": [row["turn_ref"] for row in p0
                                                if row["knowledge_role"] ==
                                                "following_context_constraint_only"],
        "pending_or_rejected_included": False,
    }
    result["frozen_context_sha256"] = _sha(result)
    return result


def validate_context_manifest_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("context_manifest_version") != CONTEXT_MANIFEST_VERSION_V2:
        errors.append("context_manifest_version")
    selected = list(value.get("selected_context_turn_refs") or [])
    p0 = list(value.get("p0_required_turn_refs") or [])
    if any(ref not in selected for ref in p0):
        errors.append("context_v2_p0_dropped")
    tail = list(value.get("parent_true_tail_refs") or [])
    if any(ref not in selected for ref in tail):
        errors.append("context_v2_parent_tail_dropped")
    if value.get("trimming_policy") != "P2_only":
        errors.append("context_v2_trimming_policy")
    predecessors = list(value.get("same_round_predecessor_refs") or [])
    continuation_sources = list(value.get("continuation_source_scene_refs") or [])
    rows = list(value.get("p0_required_dialogue") or [])
    included_scenes = {str(row.get("scene_ref") or "") for row in rows}
    if any(ref not in included_scenes for ref in predecessors):
        errors.append("context_v2_predecessor_text_missing")
    all_required_rows = [*(value.get("p0_required_dialogue") or []),
                         *(value.get("p1_dependency_dialogue") or [])]
    all_included_scenes = {str(row.get("scene_ref") or "")
                           for row in all_required_rows}
    if any(ref not in all_included_scenes for ref in continuation_sources):
        errors.append("context_v2_continuation_source_text_missing")
    return {"valid": not errors, "errors": errors,
            "context_manifest_version": CONTEXT_MANIFEST_VERSION_V2}
