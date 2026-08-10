"""Per-natural-window prompt for local reasoning-structure recognition."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import sha256_bytes
from .local_structure_recognizer_contract import PER_WINDOW_HARD_MAX
from .structure_miner_contract import STRUCTURE_TYPES


LOCAL_RECOGNIZER_PROMPT_VERSION = "local_structure_recognizer_prompt_v1"


def build_local_recognizer_prompt(window: Mapping[str, Any]) -> dict[str, Any]:
    dialogue = [
        f"{turn.get('global_position')} | {turn.get('speaker')} | {turn.get('text')}"
        for turn in window.get("turns") or []
    ]
    prompt = "\n".join([
        "ROLE: local reasoning-structure Recognizer.",
        "You see exactly one complete natural dialogue window. Do not infer from",
        "other sessions or scenes. The displayed numbers are global positions in",
        "the complete dialogue and must be copied exactly when used as evidence.",
        "Return one JSON object only.",
        "",
        "TASK",
        "Decide whether a small set of turns in this window forms a real structure",
        "that distinguishes two or more transcript-grounded states, events, people,",
        "causes, results, plans, or values. If 2-6 local turns cannot express it",
        "clearly, do not output the candidate.",
        "",
        "OUTPUT CONTRACT",
        "The top-level object has exactly one key: structures.",
        f"structures contains 0-{PER_WINDOW_HARD_MAX} candidates. There is no minimum",
        "and no type quota. Empty is valid and preferable to a weak candidate.",
        "Each candidate has exactly: structure_type, evidence_positions,",
        "discriminative_variable. evidence_positions has 2-6 unique global",
        "positions, all from this window. discriminative_variable is a short phrase",
        "describing the actual distinction; it is not a question or answer.",
        "Do not output questions, answers, gold, options, distractors, IDs, scope,",
        "canonical references, evidence IDs, or nested structure objects.",
        "",
        "ALLOWED TYPES",
        *[f"- {name}" for name in sorted(STRUCTURE_TYPES)],
        "",
        "TYPE CONDITIONS",
        "temporal_competition: multiple real events whose order must be distinguished.",
        "revision: old state/belief/information, update, then revised state/belief/information.",
        "considered_vs_adopted: a real considered/proposed alternative and adopted choice.",
        "intended_vs_executed: an intention plus later execution or non-execution.",
        "information_visibility: a specific fact and time with candidate knowers/holders.",
        "dependency: a true prerequisite, not simple chronology.",
        "competing_causes_results: multiple real causes/results for one local issue.",
        "same_dimension_comparison: two explicit values on the same dimension.",
        "Ordinary fact recall is not a reasoning structure. Repetition of one fact",
        "is not competing causes/results. Making a plan more concrete is not",
        "considered_vs_adopted. Earlier occurrence alone is not dependency.",
        "",
        f"WINDOW {window.get('window_id')} ({window.get('window_kind')})",
        *dialogue,
        "",
        "Return only the JSON object with structures.",
    ])
    return {
        "prompt_version": LOCAL_RECOGNIZER_PROMPT_VERSION,
        "window_id": window.get("window_id"),
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "window_turn_count": len(window.get("turns") or []),
        "allowed_global_positions": list(window.get("global_positions") or []),
    }
