"""Prompts for Chain Initializer, Planner, Realizer, and Delta Updater."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ..simple_append_v1.contracts import REALIZER_SCHEMA_VERSION
from ..simple_append_v1.prompts import build_round_realizer_prompt as build_base_realizer_prompt
from .contracts import (
    DELTA_UPDATER_SCHEMA_VERSION, INITIALIZER_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
)


INITIALIZER_PROMPT_VERSION = "simple_append_local_event_chain_initializer_prompt_v2"
PLANNER_PROMPT_VERSION = "simple_append_event_chain_planner_prompt_v1"
REALIZER_PROMPT_VERSION = "simple_append_event_chain_round_realizer_prompt_v1"
DELTA_UPDATER_PROMPT_VERSION = "simple_append_event_chain_delta_updater_prompt_v1"


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


def _parent_session_windows(tail_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    by_session: dict[str, dict[str, Any]] = {}
    for row in tail_context["selected_turns"]:
        session_ref = str(row["session_id"])
        if session_ref not in by_session:
            window = {"session_ref": session_ref, "turns": []}
            by_session[session_ref] = window
            windows.append(window)
        by_session[session_ref]["turns"].append({
            "turn_ref": row["turn_id"], "speaker": row["speaker"],
            "text": row["text"],
        })
    return windows


def _planner_chain_view(chain_set: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chain_set_version": chain_set["chain_set_version"],
        "root_id": chain_set["root_id"],
        "chain_set_sha256": chain_set["chain_set_sha256"],
        "chains": [{
            "chain_id": row["chain_id"],
            "event_summary": row["event_summary"],
            "participants": list(row["participants"]),
            "current_state": row["current_state"],
            "evidence_source_text": [{
                "speaker": evidence["speaker"], "text": evidence["text"],
            } for evidence in row["current_evidence_source_turns"]],
        } for row in chain_set["chains"]],
    }


def build_chain_initializer_prompt(
    *, tail_context: Mapping[str, Any], memory_overview: Mapping[str, Any],
) -> dict[str, Any]:
    windows = _parent_session_windows(tail_context)
    template = {
        "chain_initializer_schema_version": INITIALIZER_SCHEMA_VERSION,
        "session_event_candidates": [{
            "session_ref": window["session_ref"],
            "events": [{
                "event_summary": "one concrete short event from only this session",
                "event_state": "the furthest state this event reaches in this session",
                "participants": ["exact Parent speaker"],
                "evidence_turn_refs": ["exact turn_ref from this session"],
            }],
        } for window in windows],
        "continuation_links": [{
            "earlier_session_ref": "exact session_ref containing the earlier candidate",
            "earlier_candidate_index": 1,
            "later_session_ref": "same or later exact session_ref",
            "later_candidate_index": 2,
        }],
    }
    prompt = "\n\n".join([
        "SESSION-LOCAL AUTHORITATIVE EVENT CHAIN INITIALIZER v2. Return JSON only.",
        "FIRST process each Parent session as an independent window. Extract zero to four "
        "concrete short-event candidates from that window only. Each candidate must name a "
        "specific event, action, plan, decision, accepted arrangement, result, or explicit "
        "information state; give the furthest event_state reached in that same window and "
        "one to eight exact evidence refs from that window. Do not output broad themes such "
        "as responsibility, personal growth, support, work ethic, or friendship as events. "
        "Do not emit a past event merely because a later session mentions or analogizes it "
        "without changing it. Preserve concrete independent incidents even when they share a "
        "theme. The Memorax overview is a non-authoritative hint; transcript always wins.",
        "SECOND, after local extraction is fixed, emit only high-confidence continuation "
        "links. A link means the later candidate is a real subsequent development of the "
        "same specific event identity, not merely the same topic, participant, value, or "
        "emotion. It is valid to link a soccer miss to a later accepted plan to resume soccer "
        "practice. It is invalid to link a friend apology to a report mistake just because "
        "both involve responsibility. When uncertain, do not link; independent candidates "
        "are preferred over a false merge. Candidate indexes are one-based within each "
        "session's events array, and a link must point forward. Distinguish proposal, "
        "acceptance, decision, attempted action, execution, and result. Never upgrade a "
        "pending call into a completed disclosure. Do not plan future scenes, generate chain "
        "IDs, produce graphs, questions, or extra fields.",
        "NON-AUTHORITATIVE FROZEN PRE-GENERATION MEMORAX OVERVIEW\n" + _json(memory_overview),
        "AUTHORITATIVE PARENT NATURAL SESSION WINDOWS\n" + _json(windows),
        "OUTPUT EXACTLY THIS SHAPE. Preserve every listed session_ref in the same order. "
        "The example event object shows the field contract, not a minimum quota. Remove the "
        "placeholder event when a session has no reliable candidate, and use an empty links "
        "array when no concrete continuation is certain.\n" + _json(template),
    ])
    return _result(prompt, INITIALIZER_PROMPT_VERSION)


def build_planner_prompt(
    *, chain_set: Mapping[str, Any], parent_speakers: list[str],
) -> dict[str, Any]:
    template = {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "a coherent goal joining three distinct continuing event chains",
        "scenes": [{
            "chain_id": f"EC{index:04d}",
            "goal": f"goal after the frozen current state for scene {index}",
            "participants": ["exact participant", "exact participant"],
            "new_development": "a concrete next event after current_state",
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
        "INCREMENTAL EVENT CHAIN SIMPLE APPEND PLANNER v1. Return JSON only.",
        "Plan exactly three sequential substantial scenes using three different existing "
        "chain_id values. Treat each current_state and its evidence as history already "
        "realized. Continue strictly after it: do not repeat a proposal, acceptance, "
        "decision, plan, action, outcome, or resolution already recorded. Plan an actual "
        "next event, consequence, changed arrangement, new obstacle, new information, or "
        "later result. Use only the selected chain's exact participants and include three to "
        "seven interaction beats. durable_outcome must be a new concrete fact established "
        "after current_state. Do not rewrite current_state, rediscover identities, output "
        "anchors/session refs/turn refs, invent chain IDs, write fixed dialogue, emit Memory "
        "operations, or add extra fields.",
        "AVAILABLE EXACT PARENT SPEAKERS\n" + _json(parent_speakers),
        "FROZEN CURRENT EVENT CHAINS\n" + _json(_planner_chain_view(chain_set)),
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
        "INCREMENTAL EVENT CHAIN CONTINUATION CONTRACT\n"
        "Each compiled scene contains chain_id and event_chain with a frozen current_state "
        "and exact Parent evidence source text. That current state is already realized "
        "history. Begin after it. Do not replay, re-decide, re-propose, re-accept, "
        "re-establish, weaken, or contradict it. Actually realize new_development and "
        "establish durable_outcome before the scene ends. Keep all three chains distinct "
        "while preserving a natural sequential round.\n\n"
    )
    if marker not in base["prompt"]:
        raise ValueError("event_chain_realizer_marker_missing")
    prompt = base["prompt"].replace(marker, contract + marker, 1)
    if f'"realizer_schema_version":"{REALIZER_SCHEMA_VERSION}"' not in prompt:
        raise ValueError("event_chain_realizer_schema_missing")
    return _result(prompt, REALIZER_PROMPT_VERSION)


def build_delta_updater_prompt(
    *, chain_set: Mapping[str, Any], compiled_plan: Mapping[str, Any],
    child: Mapping[str, Any],
) -> dict[str, Any]:
    scene_mapping = [{
        "scene_ref": scene["scene_ref"], "chain_id": scene["chain_id"],
    } for scene in compiled_plan["scenes"]]
    chain_view = [{
        "chain_id": row["chain_id"], "event_summary": row["event_summary"],
        "participants": list(row["participants"]), "current_state": row["current_state"],
    } for row in chain_set["chains"]]
    child_view = [{
        "scene_ref": scene["scene_ref"],
        "dialogue_turns": [{
            "turn_ref": turn["turn_ref"], "speaker": turn["speaker"],
            "text": turn["text"],
        } for turn in scene["dialogue_turns"]],
    } for scene in child["scenes"]]
    template = {
        "delta_updater_schema_version": DELTA_UPDATER_SCHEMA_VERSION,
        "chain_updates": [{
            "chain_id": row["chain_id"],
            "update_status": "updated",
            "new_current_state": "furthest new state directly supported by this Child scene",
            "evidence_turn_refs": [f"R01:S{index:02d}:T001"],
        } for index, row in enumerate(scene_mapping, 1)],
        "untracked_new_event_candidates": [],
    }
    prompt = "\n\n".join([
        "CHILD-ONLY INCREMENTAL EVENT CHAIN DELTA UPDATER v1. Return JSON only.",
        "For each mapped scene, decide whether its actual Child dialogue advances the mapped "
        "existing chain beyond old current_state. Planner intentions are deliberately absent "
        "and are not facts. If advanced, report the furthest new state directly supported by "
        "one to eight exact turn_ref values from that same Child scene. If not advanced, use "
        "update_status=\"unchanged\", new_current_state=null, and an empty evidence list. "
        "Return exactly one update for every mapped chain. Never use Parent evidence, never "
        "update a different chain, never invent an EC ID, and never create a new chain. A "
        "clearly separate event that arose naturally may be recorded only as an untracked "
        "candidate with Child evidence. Do not infer completion beyond what the dialogue says.",
        "OLD FROZEN EVENT CHAINS\n" + _json(chain_view),
        "SCENE TO CHAIN MAPPING\n" + _json(scene_mapping),
        "AUTHORITATIVE NEW R1 CHILD ONLY\n" + _json(child_view),
        "OUTPUT EXACTLY THIS SHAPE\n" + _json(template),
    ])
    return _result(prompt, DELTA_UPDATER_PROMPT_VERSION)


__all__ = [
    "DELTA_UPDATER_PROMPT_VERSION", "INITIALIZER_PROMPT_VERSION",
    "PLANNER_PROMPT_VERSION", "REALIZER_PROMPT_VERSION",
    "build_chain_initializer_prompt", "build_delta_updater_prompt",
    "build_planner_prompt", "build_round_realizer_prompt",
]
