"""Deterministic per-scene gates used before the semantic round review."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .context_compiler_v2 import validate_context_manifest_v2
from .dialogue_quality import validate_dialogue_quality
from .placement_choice_registry import (
    resolve_placement_choice, resolve_source_anchor, validate_placement_choice,
)
from .narrative_action_contracts import (
    contract_subset_for_actions,
)


LOCAL_SCENE_GATE_VERSION = "local_scene_gate_v11"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_local_scene_v11(
    *, scene: Mapping[str, Any], scene_output: Mapping[str, Any],
    context_manifest: Mapping[str, Any], turn_budget: Mapping[str, Any],
    complete_dialogue: Mapping[str, Any],
    source_anchor_registry: Mapping[str, Any],
    placement_choice_registry: Mapping[str, Any],
    generated_scenes: Mapping[str, Mapping[str, Any]],
    parent_utterances: Sequence[str], previous_child_utterances: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    scene_ref = str(scene.get("scene_ref") or "")
    if not re.fullmatch(r"R\d{2}:S\d{2}", scene_ref) or not re.fullmatch(
            r"s\d+", str(scene.get("scene_alias") or "")):
        errors.append("local_scene_identity_invalid")
    if scene_output.get("scene_ref") not in {None, scene_ref}:
        errors.append("local_scene_ref_mismatch")
    context_validation = validate_context_manifest_v2(context_manifest)
    errors.extend(context_validation["errors"])
    quality = validate_dialogue_quality(
        scene_ref=scene_ref, dialogue_turns=scene_output.get("dialogue_turns"),
        speaker_participants=scene.get("speaker_participants") or [],
        turn_budget=turn_budget, parent_utterances=parent_utterances,
        previous_child_utterance_blacklist=previous_child_utterances)
    errors.extend(quality["errors"])
    try:
        source_anchor = resolve_source_anchor(
            source_anchor_registry, str(scene.get("source_anchor_id") or ""))
        placement_choice = resolve_placement_choice(
            placement_choice_registry, str(scene.get("placement_choice_id") or ""))
        errors.extend(validate_placement_choice(
            placement_choice, complete_dialogue=complete_dialogue)["errors"])
        if scene.get("source_anchor") != source_anchor:
            errors.append("local_source_anchor_drift")
        if scene.get("placement_choice") != placement_choice:
            errors.append("local_placement_choice_drift")
    except ValueError as error:
        errors.append(str(error))
    if context_manifest.get("source_anchor_id") != scene.get("source_anchor_id"):
        errors.append("local_context_anchor_mismatch")
    if context_manifest.get("placement_choice_id") != scene.get("placement_choice_id"):
        errors.append("local_context_placement_choice_mismatch")
    allowed_roles = {"preceding_known_context", "following_context_constraint_only",
                     "same_round_predecessor_dialogue", "same_round_dependency_dialogue",
                     "supplemental_history"}
    if any(row.get("knowledge_role") not in allowed_roles
           for field in ("p0_required_dialogue", "p1_dependency_dialogue",
                         "p2_optional_history")
           for row in context_manifest.get(field) or []):
        errors.append("local_context_partition_role_invalid")
    linked = scene.get("linked_existing_refs") or {}
    if not any(values for values in linked.values() if isinstance(values, list)):
        errors.append("local_historical_connection_missing")
    declared_predecessors = list(scene.get("predecessor_scene_refs") or [])
    expected_predecessors = [str(ref) for ref in declared_predecessors
                             if str(ref) in generated_scenes]
    unavailable_predecessors = [str(ref) for ref in declared_predecessors
                                if str(ref) not in generated_scenes]
    hard_predecessors = {
        str(edge.get("predecessor_scene_ref") or "")
        for edge in scene.get("dependency_edges") or []
        if isinstance(edge, Mapping) and edge.get("hard_dependency")
    }
    if list(context_manifest.get("declared_same_round_predecessor_refs") or
            context_manifest.get("same_round_predecessor_refs") or []) != [
                str(ref) for ref in declared_predecessors]:
        errors.append("local_declared_predecessor_context_mismatch")
    if list(context_manifest.get("same_round_predecessor_refs") or []) != expected_predecessors:
        errors.append("local_predecessor_context_mismatch")
    if list(context_manifest.get("unavailable_soft_predecessor_refs") or []) != (
            unavailable_predecessors):
        errors.append("local_unavailable_soft_predecessor_mismatch")
    for predecessor in unavailable_predecessors:
        if predecessor in hard_predecessors:
            errors.append(f"local_hard_predecessor_not_generated:{predecessor}")
    for predecessor in expected_predecessors:
        previous = generated_scenes.get(str(predecessor))
        expected_sha = _sha([{key: row.get(key) for key in ("turn_ref", "speaker_ref", "text")}
                             for row in previous.get("dialogue_turns") or []])
        if (context_manifest.get("predecessor_dialogue_sha256") or {}).get(
                predecessor) != expected_sha:
            errors.append(f"local_predecessor_dialogue_sha:{predecessor}")
    primary = scene.get("primary_action")
    secondary = scene.get("secondary_action")
    if not isinstance(primary, Mapping) or not primary.get("resolved_action_type"):
        errors.append("local_primary_resolved_action_invalid")
    if secondary is not None and (not isinstance(secondary, Mapping) or
                                  not secondary.get("resolved_action_type")):
        errors.append("local_secondary_resolved_action_invalid")
    context_scene_refs = {str(row.get("scene_ref") or "")
                          for field in ("p0_required_dialogue", "p1_dependency_dialogue")
                          for row in context_manifest.get(field) or []}
    for role, action in (("primary", primary), ("secondary", secondary)):
        if not isinstance(action, Mapping) or action.get("action_mode") != "continuation":
            continue
        source_ref = str(action.get("continuation_of") or "")
        source_number = source_ref[1:3]
        source_scene_ref = f"{scene_ref.split(':S', 1)[0]}:S{source_number}"
        if source_scene_ref not in context_scene_refs:
            errors.append(f"local_{role}_continuation_source_dialogue_missing:{source_scene_ref}")
    expected_contracts = contract_subset_for_actions([primary, secondary])
    if scene.get("narrative_action_contracts") != expected_contracts:
        errors.append("local_narrative_action_contract_drift")
    if scene.get("compiled_proposals") != []:
        errors.append("local_pre_review_compiled_proposal_forbidden")
    if scene.get("durable_update_compilation_stage") != "post_round_validation":
        errors.append("local_durable_update_stage_invalid")
    status = "local_gate_passed" if not errors else "local_gate_failed"
    return {
        "local_scene_gate_version": LOCAL_SCENE_GATE_VERSION,
        "scene_ref": scene_ref, "scene_alias": scene.get("scene_alias"),
        "status": status, "valid": not errors, "errors": list(dict.fromkeys(errors)),
        "quality": quality, "context_validation": context_validation,
        "dialogue_sha256": _sha(scene_output.get("dialogue_turns") or []),
    }


def dependency_unavailable_result(scene: Mapping[str, Any], blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "local_scene_gate_version": LOCAL_SCENE_GATE_VERSION,
        "scene_ref": scene.get("scene_ref"), "scene_alias": scene.get("scene_alias"),
        "status": "dependency_unavailable", "valid": False,
        "errors": ["dependency_unavailable:" + ",".join(sorted(blockers))],
        "blocking_predecessor_refs": sorted(blockers),
    }
