"""Prompts for one thin Resolver, one slim Planner, and one Round Realizer."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ..simple_append_v1.contracts import REALIZER_SCHEMA_VERSION
from ..simple_append_v1.prompts import build_round_realizer_prompt as build_base_realizer_prompt
from .contracts import PLANNER_SCHEMA_VERSION, RESOLVER_SCHEMA_VERSION


RESOLVER_PROMPT_VERSION = "simple_append_state_resolver_prompt_v1"
PLANNER_PROMPT_VERSION = "simple_append_state_resolved_planner_prompt_v1"
REALIZER_PROMPT_VERSION = "simple_append_state_resolved_round_realizer_prompt_v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _result(prompt: str, version: str) -> dict[str, Any]:
    return {
        "prompt_version": version,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
    }


def _parent_view(tail_context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_id": tail_context["root_id"],
        "parent_dialogue_sha256": tail_context["parent_dialogue_sha256"],
        "parent_speakers": list(tail_context["parent_speakers"]),
        "turns": [{
            "turn_ref": row["turn_id"],
            "session_ref": row["session_id"],
            "speaker": row["speaker"],
            "text": row["text"],
        } for row in tail_context["selected_turns"]],
    }


def _planner_resolved_view(resolved_set: Mapping[str, Any]) -> dict[str, Any]:
    """Expose semantic source text without reintroducing anchor identifiers."""
    return {
        "resolved_set_version": resolved_set["resolved_set_version"],
        "root_id": resolved_set["root_id"],
        "parent_dialogue_sha256": resolved_set["parent_dialogue_sha256"],
        "resolved_set_sha256": resolved_set["resolved_set_sha256"],
        "developments": [{
            "development_id": row["development_id"],
            "event_summary": row["event_summary"],
            "participants": list(row["participants"]),
            "latest_state": row["latest_state"],
            "evidence_source_text": [{
                "speaker": evidence["speaker"],
                "text": evidence["text"],
            } for evidence in row["evidence_source_turns"]],
        } for row in resolved_set["developments"]],
    }


def build_state_resolver_prompt(
    *, tail_context: Mapping[str, Any], memory_overview: Mapping[str, Any],
) -> dict[str, Any]:
    template = {
        "resolver_schema_version": RESOLVER_SCHEMA_VERSION,
        "developments": [{
            "event_summary": f"concise development line {index}",
            "participants": ["exact Parent speaker", "exact Parent speaker"],
            "latest_state": "the latest state directly supported by Parent turns",
            "evidence_turn_refs": ["exact Parent turn_ref"],
        } for index in range(1, 4)],
    }
    prompt = "\n\n".join([
        "THIN AUTHORITATIVE LATEST-STATE RESOLVER v1. Return JSON only.",
        "Select three to six high-confidence development lines that have real room for a "
        "later scene. For each, state only where the development has already reached by the "
        "end of the authoritative Parent. Use exact Parent speaker display names and one to "
        "eight exact Parent turn_ref values that directly support the latest state. The "
        "Memorax overview is a non-authoritative long-term recall hint: it may be incomplete, "
        "duplicated, paraphrased, or wrong. Whenever Memory conflicts with the transcript, "
        "the transcript wins. Never upgrade a proposal, plan, intention, attempted contact, "
        "or unresolved action into completed execution. Do not output development IDs, old "
        "states, versions, transitions, graphs, questions, plans, or extra fields. Do not "
        "force coverage of every Parent topic; omit any line whose latest state or evidence "
        "is unclear.",
        "NON-AUTHORITATIVE MEMORAX OVERVIEW\n" + _json(memory_overview),
        "AUTHORITATIVE PARENT TRANSCRIPT\n" + _json(_parent_view(tail_context)),
        "OUTPUT EXACTLY THIS SHAPE WITH 3-6 DEVELOPMENTS\n" + _json(template),
    ])
    return _result(prompt, RESOLVER_PROMPT_VERSION)


def build_planner_prompt(
    *, resolved_set: Mapping[str, Any], parent_speakers: list[str],
) -> dict[str, Any]:
    template = {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "a coherent goal joining three distinct continuing developments",
        "scenes": [{
            "development_id": f"D{index:02d}",
            "goal": f"goal after the frozen latest state for scene {index}",
            "participants": ["exact participant from the selected development",
                             "exact participant from the selected development"],
            "new_development": "a concrete next event after the frozen latest state",
            "interaction_beats": ["beat 1", "beat 2", "beat 3"],
            "transition_from_previous": (
                "continues naturally after the Parent tail" if index == 1
                else "a natural shift from the previous generated scene"
            ),
            "desired_outcome": "the changed situation at scene end",
            "durable_outcome": "a concrete new fact later rounds can revisit",
        } for index in range(1, 4)],
    }
    prompt = "\n\n".join([
        "STATE-RESOLVED SIMPLE APPEND PLANNER v1. Return JSON only.",
        "Plan exactly three sequential substantial scenes. Select three different frozen "
        "development_id values. Each scene must continue after that development's latest_state. "
        "Treat latest_state and its evidence as already realized history: do not repeat, "
        "re-decide, re-plan, re-perform, or re-resolve it. Plan a concrete next event, actual "
        "execution, consequence, changed arrangement, new obstacle, new information, or later "
        "result. Copy participant display names exactly and use only participants listed on "
        "that development. Include three to seven interaction beats. durable_outcome must be "
        "a new concrete fact established after latest_state. Do not rewrite latest_state, "
        "reinterpret evidence, output anchors/session refs/turn refs, generate IDs, write fixed "
        "dialogue, emit Memory operations, or add extra fields.",
        "AVAILABLE EXACT PARENT SPEAKERS\n" + _json(parent_speakers),
        "FROZEN RESOLVED DEVELOPMENT TRACKS\n" + _json(
            _planner_resolved_view(resolved_set)),
        "OUTPUT EXACTLY THIS SHAPE\n" + _json(template),
    ])
    return _result(prompt, PLANNER_PROMPT_VERSION)


def build_round_realizer_prompt(
    *, compiled_plan: Mapping[str, Any], tail_context: Mapping[str, Any],
) -> dict[str, Any]:
    base = build_base_realizer_prompt(
        compiled_plan=compiled_plan, tail_context=tail_context)
    marker = "FROZEN ROUND INPUT\n"
    contract = (
        "AUTHORITATIVE LATEST-STATE CONTINUATION CONTRACT\n"
        "Each compiled scene contains development_id and resolved_development with an "
        "authoritative latest_state plus its exact Parent evidence source text. Treat that "
        "latest state as history that has already happened. Begin after it. Do not replay, "
        "re-decide, re-establish, weaken, or contradict it. Actually realize new_development "
        "and establish durable_outcome before that scene ends. Keep all three development "
        "lines distinct while preserving a natural sequential round.\n\n"
    )
    if marker not in base["prompt"]:
        raise ValueError("state_resolved_realizer_marker_missing")
    prompt = base["prompt"].replace(marker, contract + marker, 1)
    if f'"realizer_schema_version":"{REALIZER_SCHEMA_VERSION}"' not in prompt:
        raise ValueError("state_resolved_realizer_schema_missing")
    return _result(prompt, REALIZER_PROMPT_VERSION)


__all__ = [
    "PLANNER_PROMPT_VERSION",
    "REALIZER_PROMPT_VERSION",
    "RESOLVER_PROMPT_VERSION",
    "build_planner_prompt",
    "build_round_realizer_prompt",
    "build_state_resolver_prompt",
]
