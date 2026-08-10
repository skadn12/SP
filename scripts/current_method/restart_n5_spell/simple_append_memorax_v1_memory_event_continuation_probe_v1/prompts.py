"""Prompts for the single Planner and single round Realizer calls."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _transcript(parent: Mapping[str, Any]) -> str:
    return "\n".join(
        f"{row['turn_id']} | {row['speaker']} | {row['text']}"
        for row in parent["dialogue"]["turns"])


def build_planner_prompt(*, parent: Mapping[str, Any], memory_overview: Mapping[str, Any],
                         candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_view = [{
        "memory_event_id": row["memory_event_id"],
        "memory_event_key": row["memory_event_key"],
        "memory_event_display": row["memory_event_display"],
        "memory_content": row["memory_event_content"],
    } for row in candidates]
    text = f"""You are the Planner for a fresh append-only expansion.

Memorax is only a non-authoritative index of long-term people and event hints.
The authoritative transcript below decides what has actually happened.
Choose exactly three different concrete memory events and continue each one
after its verified current state. For every scene, draft exactly 2 or 3
distinct next-step candidates in recommendation order. Each candidate must
contain its candidate-owned new_development, durable_outcome, and interaction
beats, and must represent a state change that has not already happened in
Parent. A candidate's interaction_beats must enact only that candidate and must
not contain content from any other candidate. Do not select a final
candidate; the local program selects the first candidate approved by the Gate.
Do not repeat an existing intention, decision, action,
agreement, resolution, or result. Do not use a synonym, another discussion,
or another confirmation as a next step. Do not invent evidence or upgrade a
plan into an execution.
Keep all participants as exact Parent speaker names. Each candidate interaction
beat must be a concrete beat that a later Realizer can enact.

CANDIDATE MEMORY EVENTS
{_json(candidate_view)}

AUTHORITATIVE PARENT TRANSCRIPT
{_transcript(parent)}

Output only this JSON object. No markdown, no extra keys:
{{
  "round_goal": "...",
  "scenes": [
    {{
      "memory_event_id": "ME01",
      "verified_current_state": "exactly what the transcript supports as latest",
      "evidence_turn_refs": ["cc-train-episode-...:P01:T001"],
      "goal": "...",
      "participants": ["Exact Parent speaker", "Exact Parent speaker"],
      "next_step_candidates": [
        {{
          "new_development": "A concrete next state change after the current state",
          "durable_outcome": "The durable consequence of that candidate",
          "interaction_beats": ["...", "...", "..."]
        }},
        {{
          "new_development": "A different concrete next state change",
          "durable_outcome": "The durable consequence of the second candidate",
          "interaction_beats": ["...", "...", "..."]
        }}
      ],
      "transition_from_previous": "..."
    }}
  ]
}}"""
    return {"prompt_version":"memory_event_continuation_planner_prompt_v5",
            "prompt":text,"candidate_count":len(candidates)}


def build_realizer_prompt(*, parent: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    compact = [{k: scene.get(k) for k in (
        "memory_event_id", "verified_current_state", "goal", "participants",
        "new_development", "interaction_beats", "transition_from_previous",
        "durable_outcome")} for scene in plan["scenes"]]
    text = f"""You are the Natural Realizer. Write one coherent append-only Child
round for the three frozen scene plans below. Treat every verified_current_state
as already realized Parent history. Never replay or re-decide it. Continue from
that state, enact new_development, and let later scenes follow the actual prior
Child scene. Use natural dialogue, concrete actions, information, decisions,
constraints, or consequences. Use only exact Parent speakers listed per scene.
Each scene must contain 9 to 48 turns and at least two speakers. Do not copy any
Parent sentence verbatim. Return only JSON with the exact top-level key scenes;
each scene contains dialogue_turns with speaker and text only.

FROZEN PARENT TRANSCRIPT
{_transcript(parent)}

FROZEN SCENE PLANS
{_json(compact)}

OUTPUT
{{"scenes":[{{"dialogue_turns":[{{"speaker":"...","text":"..."}}]}}]}}"""
    return {"prompt_version":"memory_event_continuation_realizer_prompt_v1",
            "prompt":text}


def build_semantic_continuation_gate_prompt(
    *, gate_context: Mapping[str, Any],
) -> dict[str, Any]:
    text = f"""You are a strict Semantic Continuation Gate between Planner and
Realizer. Audit only whether each candidate's new_development, durable_outcome,
or interaction_beats contain a development that has already happened
in the authoritative Parent evidence. Memorax is a non-authoritative hint;
Parent evidence is authoritative.

Check only these six duplicate categories:
- decision
- intention
- action
- agreement
- resolution
- result

Return reject if the proposed new_development or durable_outcome merely repeats
one of those developments already present in Parent, even when paraphrased.
Return pass only when it genuinely occurs after the Parent's latest state.
Do not repair, rewrite, improve, or suggest an alternative development. For a
reject, cite exact Parent turn refs proving the duplicate. For a pass, use an
empty evidence_turn_refs list. Audit every candidate_ref exactly once. Judge
each candidate independently; do not select, rank, repair, or rewrite it.

GATE INPUT
{_json(gate_context)}

OUTPUT ONLY THIS JSON OBJECT, WITH NO EXTRA KEYS
{{
  "candidate_reviews": [
    {{
      "candidate_ref": "ME01:C01",
      "decision": "pass",
      "duplicate_with_parent": false,
      "evidence_turn_refs": [],
      "reason": "The development occurs after the latest Parent state."
    }}
  ]
}}"""
    return {"prompt_version": "semantic_continuation_gate_prompt_v3",
            "prompt": text,
            "scene_count": len(gate_context.get("scenes") or [])}
