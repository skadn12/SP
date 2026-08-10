"""Slim durable-development Planner and one-request Round Realizer prompts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ..simple_append_v1.contracts import REALIZER_SCHEMA_VERSION
from ..simple_append_v1.prompts import build_round_realizer_prompt as build_base_realizer_prompt
from .durable_contracts import (
    PARENT_ANCHOR_REGISTRY_VERSION,
    PLANNER_SCHEMA_VERSION,
)


PLANNER_PROMPT_VERSION = "simple_append_durable_devdiv_local_span_planner_prompt_v1"
REALIZER_PROMPT_VERSION = (
    "simple_append_durable_devdiv_local_span_round_realizer_prompt_v1")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _result(prompt: str, *, version: str) -> dict[str, Any]:
    return {
        "prompt_version": version,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
    }


def _tail_view(
    context: Mapping[str, Any], parent_anchor_registry: Mapping[str, Any],
) -> dict[str, Any]:
    if parent_anchor_registry.get("parent_anchor_registry_version") != \
            PARENT_ANCHOR_REGISTRY_VERSION:
        raise ValueError("durable_planner_anchor_registry_invalid")
    positions = {
        str(row["turn_ref"]): int(row["position"])
        for row in parent_anchor_registry.get("entries") or []
    }
    return {
        "root_id": context["root_id"],
        "parent_dialogue_sha256": context["parent_dialogue_sha256"],
        "selected_session_ids": list(context["selected_session_ids"]),
        "parent_speakers": list(context["parent_speakers"]),
        "turns": [{
            "global_parent_position": positions[str(row["turn_id"])],
            "turn_id": row["turn_id"],
            "session_id": row["session_id"],
            "speaker": row["speaker"],
            "text": row["text"],
        } for row in context["selected_turns"]],
    }


def build_planner_prompt(
    tail_context: Mapping[str, Any],
    parent_anchor_registry: Mapping[str, Any],
) -> dict[str, Any]:
    templates = []
    session_starts: list[int] = []
    seen_sessions: set[str] = set()
    for row in parent_anchor_registry.get("entries") or []:
        session_ref = str(row["session_ref"])
        if session_ref not in seen_sessions:
            seen_sessions.add(session_ref)
            session_starts.append(int(row["position"]))
    for index in range(3):
        example_start = session_starts[index] if index < len(session_starts) else 1
        templates.append({
            "goal": f"distinct goal for scene {index + 1}",
            "participants": ["exact speaker display name", "exact speaker display name"],
            "primary_anchor_span": {
                "start_position": example_start,
                "end_position": example_start,
            },
            "new_development": "a concrete event, decision, action, assignment, information change, arrangement, or consequence",
            "interaction_beats": ["beat 1", "beat 2", "beat 3"],
            "transition_from_previous": (
                "continues naturally after the Parent tail" if index == 0
                else "natural time, place, or participant transition without taking over the new primary track"),
            "desired_outcome": "the changed situation at scene end",
            "durable_outcome": "a concrete fact later rounds can revisit, update, complete, cancel, or experience consequences from",
        })
    template = {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "...",
        "scenes": templates,
    }
    prompt = "\n\n".join([
        "SIMPLE APPEND DURABLE-DEVELOPMENT PLANNER v1. Return JSON only.",
        "Plan two or three sequential substantial dialogue scenes; prefer three and preserve the later 42-54-turn round target. "
        "For each scene, choose one short contiguous passage from the Parent dialogue that most directly establishes the prior situation this scene will continue or change. Return only its inclusive primary_anchor_span start_position and end_position. The span must contain one to six consecutive global_parent_position turns within one Parent session; choose the shortest passage that is sufficient. It is not a collection of all related historical evidence: do not collect every earlier mention or combine callbacks from multiple sessions. Never output source_anchor_positions, session_ref, session_id, source_scope, turn IDs, or a source_anchor object. The three scenes must use spans from three different Parent sessions. new_development must directly continue, change, complete, fail, update, or cause a consequence of the selected local passage, not merely share a person or broad topic. "
        "Copy every existing Parent speaker display name exactly. Each scene needs two to five participants and three to seven interaction beats. Newly connected speakers are allowed when named in participants. "
        "new_development must produce a concrete new event, decision, performed action, assignment, information change, arrangement, state change, or consequence. Merely talking, remembering, supporting, considering, or hoping is not a development. "
        "durable_outcome is mandatory and must state a concrete fact that later rounds can revisit, update, complete, cancel, or experience consequences from. A vague feeling, lesson, intention to improve, or useful conversation is not enough. "
        "transition_from_previous only makes the sequence natural; it does not require the next scene to share the previous primary storyline. A callback may support continuity, but it must not take over the selected Parent passage's concrete progress. Scene 3 must own its third anchor, development, and durable outcome rather than merely resolving scenes 1 and 2. "
        "Do not output scene refs, turn IDs, Memory operations, QA targets, graphs, reasoning types, fixed dialogue, or extra fields.",
        "AUTHORITATIVE SELECTED PARENT CONTEXT\n" + _json(
            _tail_view(tail_context, parent_anchor_registry)),
        "OUTPUT EXACTLY THIS SHAPE\n" + _json(template),
    ])
    return _result(prompt, version=PLANNER_PROMPT_VERSION)


def build_round_realizer_prompt(
    *, compiled_plan: Mapping[str, Any], tail_context: Mapping[str, Any],
) -> dict[str, Any]:
    base = build_base_realizer_prompt(
        compiled_plan=compiled_plan, tail_context=tail_context)
    marker = "FROZEN ROUND INPUT\n"
    contract = (
        "DURABLE DEVELOPMENT REALIZATION CONTRACT\n"
        "For every scene, primary_anchor_materialization is the program-derived contiguous Parent source passage. It includes exact positions, speakers, source text, and derived_anchor_session_ref; treat it as the primary prior situation the scene must advance and do not choose or reinterpret a different anchor. Most concrete progress must remain centered on that passage and new_development; callbacks may aid continuity but must not take over. durable_outcome is not a topic suggestion: before the scene ends, the dialogue must actually establish or perform that concrete result. Do not weaken an actual action, assignment, arrangement, information change, or consequence into a possibility that characters merely discuss for later. Preserve natural depth rather than filling turns with repeated encouragement, agreement, moralizing, paraphrase, or unchanged small talk. Scene 3 has its own third track and may only lightly callback earlier scenes.\n\n"
    )
    if marker not in base["prompt"]:
        raise ValueError("durable_realizer_prompt_marker_missing")
    prompt = base["prompt"].replace(marker, contract + marker, 1)
    if f'"realizer_schema_version":"{REALIZER_SCHEMA_VERSION}"' not in prompt:
        raise ValueError("durable_realizer_schema_missing")
    return _result(prompt, version=REALIZER_PROMPT_VERSION)


__all__ = [
    "PLANNER_PROMPT_VERSION",
    "REALIZER_PROMPT_VERSION",
    "build_planner_prompt",
    "build_round_realizer_prompt",
]
