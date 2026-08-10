"""Deterministic historical anchor registry and placement-boundary validation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


ANCHOR_REGISTRY_VERSION = "anchor_registry_v1"
PLACEMENT_BOUNDARY_VERSION = "placement_boundary_v1"
PLACEMENTS = frozenset({"after", "before", "between", "parallel"})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _turn_index(complete_dialogue: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, Mapping[str, Any]]]:
    rows = complete_dialogue.get("turns") or []
    return ({str(row["turn_ref"]): index for index, row in enumerate(rows)},
            {str(row["turn_ref"]): row for row in rows})


def _entity_refs(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row.get("speaker_ref") or "") for row in rows
                   if row.get("speaker_ref")})


def _anchor(
    *, anchor_type: str, source_kind: str, session_ref: str | None,
    scene_ref: str | None, rows: Sequence[Mapping[str, Any]],
    linked_entity_refs: Sequence[str] = (), linked_interaction_refs: Sequence[str] = (),
    linked_relation_refs: Sequence[str] = (), linked_event_refs: Sequence[str] = (),
    linked_task_refs: Sequence[str] = (), linked_thread_refs: Sequence[str] = (),
    linked_state_refs: Sequence[str] = (),
    after_turn_ref: str | None, before_turn_ref: str | None,
    available_placements: Sequence[str],
) -> dict[str, Any]:
    return {
        "anchor_type": anchor_type, "source_kind": source_kind,
        "session_ref": session_ref, "scene_ref": scene_ref,
        "turn_refs": [str(row["turn_ref"]) for row in rows],
        "exact_excerpt": [{"turn_ref": str(row["turn_ref"]),
                           "speaker_ref": row.get("speaker_ref"),
                           "speaker_label": row.get("speaker_label"),
                           "text": str(row["text"])} for row in rows],
        "linked_entity_refs": sorted(set(linked_entity_refs) | set(_entity_refs(rows))),
        "linked_interaction_refs": sorted(set(linked_interaction_refs)),
        "linked_relation_refs": sorted(set(linked_relation_refs)),
        "linked_event_refs": sorted(set(linked_event_refs)),
        "linked_task_refs": sorted(set(linked_task_refs)),
        "linked_thread_refs": sorted(set(linked_thread_refs)),
        "linked_state_refs": sorted(set(linked_state_refs)),
        "boundary": {"after_turn_ref": after_turn_ref,
                     "before_turn_ref": before_turn_ref},
        "available_placements": list(dict.fromkeys(available_placements)),
    }


def compile_anchor_registry(
    *, complete_dialogue: Mapping[str, Any], planning_state: Mapping[str, Any],
    tail_turn_count: int = 7,
) -> dict[str, Any]:
    if complete_dialogue.get("root_id") != planning_state.get("root_id"):
        raise ValueError("anchor_registry_root_mismatch")
    turns = list(complete_dialogue.get("turns") or [])
    if not turns:
        raise ValueError("anchor_registry_dialogue_empty")
    by_session: dict[str, list[Mapping[str, Any]]] = {}
    session_order: list[str] = []
    for row in turns:
        session = str(row.get("session_ref") or "")
        if session not in by_session:
            session_order.append(session)
            by_session[session] = []
        by_session[session].append(row)
    candidates: list[dict[str, Any]] = []
    tail = turns[-min(tail_turn_count, len(turns)):]
    candidates.append(_anchor(
        anchor_type="tail_anchor", source_kind="historical_dialogue",
        session_ref=str(tail[-1].get("session_ref") or ""),
        scene_ref=tail[-1].get("scene_ref"), rows=tail,
        after_turn_ref=str(tail[-1]["turn_ref"]),
        before_turn_ref=str(tail[0]["turn_ref"]),
        available_placements=("after", "before", "parallel")))
    for left_session, right_session in zip(session_order, session_order[1:]):
        left = by_session[left_session][-min(3, len(by_session[left_session])):]
        right = by_session[right_session][:min(3, len(by_session[right_session]))]
        candidates.append(_anchor(
            anchor_type="historical_gap_anchor", source_kind="historical_dialogue",
            session_ref=left_session, scene_ref=left[-1].get("scene_ref"),
            rows=[*left, *right], after_turn_ref=str(left[-1]["turn_ref"]),
            before_turn_ref=str(right[0]["turn_ref"]),
            available_placements=("between", "parallel")))
    turn_by_ref = {str(row["turn_ref"]): row for row in turns}
    for link in planning_state.get("interaction_links") or []:
        refs = [str(value) for value in link.get("source_turn_refs") or []
                if str(value) in turn_by_ref]
        rows = sorted((turn_by_ref[ref] for ref in refs),
                      key=lambda row: turns.index(row))
        if not rows:
            continue
        candidates.append(_anchor(
            anchor_type="interaction_anchor", source_kind="planning_state",
            session_ref=str(rows[0].get("session_ref") or ""),
            scene_ref=rows[0].get("scene_ref"), rows=rows,
            linked_entity_refs=list(link.get("participant_refs") or []),
            linked_interaction_refs=[str(link.get("interaction_link_ref") or "")],
            after_turn_ref=str(rows[-1]["turn_ref"]),
            before_turn_ref=str(rows[0]["turn_ref"]),
            available_placements=("after", "before", "parallel")))
    structured = (
        ("relationship_anchor", "explicit_relations", "linked_relation_refs"),
        ("event_consequence_anchor", "active_events", "linked_event_refs"),
        ("task_anchor", "tasks", "linked_task_refs"),
        ("open_thread_anchor", "open_threads", "linked_thread_refs"),
        ("state_transition_anchor", "key_states", "linked_state_refs"),
    )
    for anchor_type, field, link_field in structured:
        for record in planning_state.get(field) or []:
            refs = [str(value) for value in record.get("support_turn_refs") or []
                    if str(value) in turn_by_ref]
            rows = sorted((turn_by_ref[ref] for ref in refs),
                          key=lambda row: turns.index(row))
            if not rows:
                continue
            kwargs = {link_field: [str(record.get("memory_ref") or "")]}
            candidates.append(_anchor(
                anchor_type=anchor_type, source_kind="planning_state",
                session_ref=str(rows[0].get("session_ref") or ""),
                scene_ref=rows[0].get("scene_ref"), rows=rows,
                linked_entity_refs=list(record.get("subject_refs") or []),
                after_turn_ref=str(rows[-1]["turn_ref"]),
                before_turn_ref=str(rows[0]["turn_ref"]),
                available_placements=("after", "before", "parallel"), **kwargs))
    # Canonical content ordering makes PA#### stable for identical inputs.
    candidates.sort(key=lambda row: (
        row["anchor_type"], row["turn_refs"], row["linked_entity_refs"]))
    for index, row in enumerate(candidates, 1):
        row["anchor_id"] = f"PA{index:04d}"
        row["root_id"] = str(complete_dialogue["root_id"])
    result = {
        "anchor_registry_version": ANCHOR_REGISTRY_VERSION,
        "root_id": str(complete_dialogue["root_id"]),
        "complete_dialogue_sha256": complete_dialogue.get("complete_dialogue_sha256"),
        "planning_state_sha256": planning_state.get("planning_state_sha256"),
        "anchors": candidates,
    }
    result["anchor_registry_sha256"] = _sha(result)
    return result


def validate_anchor_registry(
    registry: Mapping[str, Any], *, complete_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if registry.get("anchor_registry_version") != ANCHOR_REGISTRY_VERSION:
        errors.append("anchor_registry_version")
    if registry.get("root_id") != complete_dialogue.get("root_id"):
        errors.append("anchor_registry_root")
    turn_refs = {str(row.get("turn_ref")) for row in complete_dialogue.get("turns") or []}
    ids: list[str] = []
    for row in registry.get("anchors") or []:
        anchor_id = str(row.get("anchor_id") or "")
        ids.append(anchor_id)
        if row.get("root_id") != registry.get("root_id"):
            errors.append(f"anchor_cross_root:{anchor_id}")
        refs = list(row.get("turn_refs") or [])
        if not refs or any(str(ref) not in turn_refs for ref in refs):
            errors.append(f"anchor_turn_ref_missing:{anchor_id}")
        if not set(row.get("available_placements") or []).issubset(PLACEMENTS):
            errors.append(f"anchor_placement_enum:{anchor_id}")
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append("anchor_id_invalid")
    return {"valid": not errors, "errors": errors,
            "anchor_registry_version": ANCHOR_REGISTRY_VERSION}


def validate_placement_boundary(
    *, anchor: Mapping[str, Any], placement: str,
    complete_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    placement = str(placement)
    if placement not in PLACEMENTS:
        errors.append("placement_enum_invalid")
    if placement not in set(anchor.get("available_placements") or []):
        errors.append("placement_not_available_for_anchor")
    boundary = anchor.get("boundary") if isinstance(anchor.get("boundary"), Mapping) else {}
    after_ref = boundary.get("after_turn_ref")
    before_ref = boundary.get("before_turn_ref")
    positions, _ = _turn_index(complete_dialogue)
    if placement == "after" and (not after_ref or after_ref not in positions):
        errors.append("after_turn_ref_required")
    if placement == "before" and (not before_ref or before_ref not in positions):
        errors.append("before_turn_ref_required")
    if placement == "between":
        if not after_ref or after_ref not in positions:
            errors.append("between_after_turn_ref_required")
        if not before_ref or before_ref not in positions:
            errors.append("between_before_turn_ref_required")
        if after_ref in positions and before_ref in positions and positions[after_ref] >= positions[before_ref]:
            errors.append("between_boundary_order_invalid")
    if placement == "parallel":
        if not anchor.get("session_ref") or not (after_ref or before_ref):
            errors.append("parallel_interval_boundary_required")
    return {
        "valid": not errors, "errors": errors,
        "placement_boundary_version": PLACEMENT_BOUNDARY_VERSION,
        "placement": placement, "after_turn_ref": after_ref,
        "before_turn_ref": before_ref,
    }


def resolve_anchor(
    registry: Mapping[str, Any], anchor_id: str,
) -> dict[str, Any]:
    matches = [row for row in registry.get("anchors") or []
               if row.get("anchor_id") == anchor_id]
    if len(matches) != 1:
        raise ValueError(f"unknown_or_duplicate_anchor_id:{anchor_id}")
    return deepcopy(dict(matches[0]))
