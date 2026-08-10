"""Deterministic, provenance-aware parent context compiler."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


CONTEXT_COMPILER_VERSION = "planner_context_compiler_v1"
DEFAULT_CONTEXT_TOKEN_BUDGET = 2400


def _estimate_tokens(value: Any) -> int:
    return max(1, (len(json.dumps(value, ensure_ascii=False)) + 3) // 4)


def compile_scene_context(
    *, scene: Mapping[str, Any], actions: Sequence[Mapping[str, Any]],
    authoritative_memory: Mapping[str, Any], parent_turns: Sequence[Mapping[str, Any]],
    interaction_links: Sequence[Mapping[str, Any]] = (),
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Select by explicit refs and provenance; never include pending/rejected text."""
    selected_action_refs = {
        str(ref) for action in actions for ref in action.get("connected_memory_refs") or []}
    speaker_refs = {str(row.get("participant_ref"))
                    for row in scene.get("speaker_participants") or []}
    authoritative_records: list[Mapping[str, Any]] = []
    for field in ("authoritative_records", "current_states", "open_threads",
                  "event_history", "relations"):
        for row in authoritative_memory.get(field) or []:
            status = str(row.get("verification_status") or row.get("semantic_status") or "")
            if status not in {"source_grounded", "semantically_verified"}:
                continue
            authoritative_records.append(row)
    ranked: list[tuple[int, str, Mapping[str, Any]]] = []
    selected_turn_refs: set[str] = set()
    selected_memory_refs: list[str] = []
    context_thread_refs: list[str] = []
    context_parent_session_refs: set[str] = set()
    for row in authoritative_records:
        ref = str(row.get("memory_ref") or row.get("memory_id") or "")
        subjects = {str(value) for value in row.get("subject_refs") or row.get("subject_ids") or []}
        priority = 0 if ref in selected_action_refs else 1 if subjects & speaker_refs else 2
        ranked.append((priority, ref, row))
    for _, ref, row in sorted(ranked, key=lambda value: (value[0], value[1])):
        if ref:
            selected_memory_refs.append(ref)
        if str(row.get("record_type")) in {"thread", "open_thread"}:
            context_thread_refs.append(ref)
        selected_turn_refs.update(str(value) for value in
                                  row.get("support_turn_refs") or
                                  row.get("source_turn_refs") or
                                  row.get("source_turn_ids") or [])
        context_parent_session_refs.update(str(value) for value in
                                           (row.get("provenance") or {}).get("source_session_refs") or [])
    for link in interaction_links:
        if set(link.get("entity_refs") or []).issubset(speaker_refs):
            selected_turn_refs.update(str(value) for value in link.get("source_turn_refs") or [])
            context_parent_session_refs.update(str(value) for value in link.get("source_session_refs") or [])

    ranked_turns = sorted(parent_turns, key=lambda row: (
        0 if str(row.get("turn_id") or row.get("turn_ref")) in selected_turn_refs else 1,
        str(row.get("turn_id") or row.get("turn_ref") or "")))
    selected_turns: list[dict[str, Any]] = []
    dropped: list[str] = []
    tokens = 0
    for turn in ranked_turns:
        turn_ref = str(turn.get("turn_id") or turn.get("turn_ref") or "")
        compact = {"turn_ref": turn_ref, "speaker": turn.get("speaker"), "text": turn.get("text")}
        cost = _estimate_tokens(compact)
        if tokens + cost <= token_budget and (turn_ref in selected_turn_refs or not selected_turns):
            selected_turns.append(compact)
            tokens += cost
        else:
            dropped.append(turn_ref)
    return {
        "context_compiler_version": CONTEXT_COMPILER_VERSION,
        "context_memory_refs": sorted(set(selected_memory_refs)),
        "context_thread_refs": sorted(set(context_thread_refs)),
        "context_parent_session_refs": sorted(ref for ref in context_parent_session_refs if ref),
        "selected_context_refs": [row["turn_ref"] for row in selected_turns],
        "dropped_context_refs": dropped,
        "parent_turns_untrusted": selected_turns,
        "token_estimate": tokens,
        "token_budget": token_budget,
        "truncation_reason": "token_budget" if dropped else None,
        "same_round_scene_text_included": False,
        "pending_or_rejected_included": False,
    }


def compile_round_planner_context(
    *, action_menu: Mapping[str, Any], authoritative_memory: Mapping[str, Any],
    parent_turns: Sequence[Mapping[str, Any]],
    interaction_links: Sequence[Mapping[str, Any]] = (),
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Compile the frozen pre-Plan context from the complete action menu."""
    actions = list(action_menu.get("actions") or [])
    entity_refs = sorted({str(ref) for action in actions
                          for ref in action.get("allowed_existing_entity_refs") or []})
    pseudo_scene = {
        "speaker_participants": [{"participant_ref": ref} for ref in entity_refs]
    }
    result = compile_scene_context(
        scene=pseudo_scene, actions=actions,
        authoritative_memory=authoritative_memory, parent_turns=parent_turns,
        interaction_links=interaction_links, token_budget=token_budget)
    result["context_compiler_version"] = "round_planner_context_compiler_v1"
    result["action_ids"] = [str(row.get("action_id")) for row in actions]
    return result
