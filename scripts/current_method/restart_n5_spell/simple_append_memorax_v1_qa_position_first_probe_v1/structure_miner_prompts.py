"""Prompt for the minimal dialogue reasoning-structure Miner."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import sha256_bytes
from .structure_miner_contract import STRUCTURE_HARD_MAX, STRUCTURE_TYPES


STRUCTURE_MINER_PROMPT_VERSION = "structure_miner_minimal_prompt_v1"


def build_structure_miner_prompt(expanded: Mapping[str, Any]) -> dict[str, Any]:
    turns = [turn for turn in expanded.get("turns") or [] if isinstance(turn, Mapping)]
    dialogue = [
        f"{index} | {turn.get('speaker')} | {turn.get('text')}"
        for index, turn in enumerate(turns, 1)
    ]
    prompt = "\n".join([
        "ROLE: local reasoning-structure Miner.",
        "The numbered dialogue is the only source. Find only genuine local",
        "structures that could later support a discriminative question.",
        "Do not write any question, answer, option, or distractor.",
        "Return one JSON object only.",
        "",
        "OUTPUT CONTRACT",
        "The top-level object has exactly one key: structures.",
        f"structures is an array of 0-{STRUCTURE_HARD_MAX} candidates; there is no minimum",
        "and no type quota. Prefer an empty or short array to an uncertain claim.",
        "Each candidate has exactly three fields: structure_type,",
        "evidence_positions, discriminative_variable.",
        "evidence_positions contains 2-6 unique 1-based dialogue positions.",
        "discriminative_variable is a short phrase naming what must be distinguished.",
        "Never output IDs, D###, source_scope, canonical turn references, supporting",
        "turns, evidence IDs, questions, gold, options, distractors, or structure objects.",
        "",
        "ALLOWED TYPES",
        *[f"- {name}" for name in sorted(STRUCTURE_TYPES)],
        "",
        "MINIMUM SEMANTIC CONDITIONS",
        "temporal_competition: at least two real events whose order must be distinguished.",
        "revision: old state/belief/information, an update, and a revised state/belief/information.",
        "considered_vs_adopted: both a considered/proposed alternative and the adopted choice.",
        "intended_vs_executed: both an intention/plan and later execution or non-execution.",
        "information_visibility: a specific fact, timepoint, and candidate knowers/holders.",
        "dependency: a true prerequisite without which the dependent outcome would not occur as described.",
        "competing_causes_results: multiple real causes or results in the same local thread.",
        "same_dimension_comparison: two transcript-grounded values on the same explicit dimension.",
        "Do not relabel ordinary recall, chronology, decision, association, or topic",
        "similarity as one of these structures. Do not combine unrelated story arcs.",
        "Select minimal sufficient evidence, not a whole session or story arc.",
        "",
        "DISCRIMINATIVE VARIABLE",
        "State the actual distinction, such as event order, old versus revised state,",
        "proposed versus adopted action, intended versus executed action, who knew a",
        "fact at a timepoint, prerequisite versus non-prerequisite, competing real",
        "causes/results, or two values on the same comparison dimension.",
        "It is not a question and it is not an answer.",
        "",
        "NUMBERED AUTHORITATIVE DIALOGUE",
        *dialogue,
        "",
        "Return only the JSON object with structures. Do not explain your choices.",
    ])
    return {
        "prompt_version": STRUCTURE_MINER_PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "dialogue_turn_count": len(turns),
        "output_fields": [
            "structure_type", "evidence_positions", "discriminative_variable",
        ],
    }
