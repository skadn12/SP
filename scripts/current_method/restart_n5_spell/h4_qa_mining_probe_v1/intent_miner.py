from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from .intent_contract import INTENT_TYPES, ROLE_KEYS


INTENT_MINER_PROMPT_VERSION = "h4_qa_mining_intent_miner_v1"


def build_intent_prompt(expanded_dialogue: Mapping[str, Any]) -> dict[str, Any]:
    turns = list(expanded_dialogue.get("turns") or [])
    numbered = [
        f"{index} | {row.get('turn_ref')} | {row.get('speaker')} | {row.get('text')}"
        for index, row in enumerate(turns, 1)
    ]
    schema = {
        "intents": [{
            "intent_id": "I001",
            "intent_type": "state_transition",
            "reasoning_chain": {
                "before_state": [12],
                "trigger": [31],
                "after_state": [42],
            },
            "discriminative_variable": "workload sustainability",
            "candidate_question_types": ["single_choice", "multi_select", "ordering"],
        }]
    }
    prompt = "\n".join([
        "ROLE: H4 QA intent miner.",
        "The numbered expanded dialogue is the only factual source.",
        "Find exactly three distinct, transcript-grounded reasoning intents.",
        "Do not write questions, options, answers, distractors, or explanations.",
        "Each accepted intent must support all three Phase-0 families: two single-choice, one multi-select, and one ordering candidate.",
        "Use 1-based dialogue positions in reasoning_chain; the program will map positions to canonical turn_refs.",
        "The three chain groups must be non-empty, disjoint, and together express a real relation.",
        "discriminative_variable must name the distinction a question must test, not a topic.",
        "Allowed intent_type values: " + ", ".join(sorted(INTENT_TYPES)),
        "Allowed reasoning_chain keys: " + ", ".join(ROLE_KEYS),
        "Return exactly one JSON object with this shape:",
        json.dumps(schema, ensure_ascii=False, indent=2),
        "NUMBERED AUTHORITATIVE DIALOGUE:",
        *numbered,
        "Return JSON only.",
    ])
    return {
        "prompt_version": INTENT_MINER_PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "dialogue_turn_count": len(turns),
        "expected_intent_count": 3,
    }

