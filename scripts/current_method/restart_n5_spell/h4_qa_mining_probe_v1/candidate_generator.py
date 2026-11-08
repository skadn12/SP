from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes


CANDIDATE_PROMPT_VERSION = "h4_qa_mining_candidate_generator_v1"


def build_candidate_prompt(
    expanded_dialogue: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    turns = list(expanded_dialogue.get("turns") or [])
    dialogue = [
        f"{row.get('turn_ref')} | {row.get('speaker')} | {row.get('text')}"
        for row in turns
    ]
    output_schema = {
        "candidates": [
            {
                "candidate_id": "I001-S01",
                "question_type": "single_choice",
                "question": "...",
                "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
                "correct_option": "A",
                "gold": [],
                "evidence_refs": ["R02:S01:T003"],
                "evidence_groups": {
                    "before_state": ["R02:S01:T003"],
                    "trigger": ["R02:S02:T002"],
                    "after_state": ["R02:S03:T003"],
                },
                "required_reasoning": ["before_state", "trigger"],
                "discriminative_variable": "...",
                "distractor_strategy": "same_topic_wrong_relation",
            }
        ]
    }
    prompt = "\n".join([
        "ROLE: H4 QA mining candidate generator.",
        "Use only the authoritative dialogue and the supplied reasoning contract.",
        "Generate exactly four candidates for this one intent: two single_choice, one multi_select, and one ordering.",
        "The four candidates must test the same discriminative relation in different valid forms.",
        "For each candidate, required_reasoning must include before_state, trigger, and after_state; the answer must require the transition chain, not only one endpoint.",
        "Do not generate a surface final-state recall question.",
        "Every distractor must be transcript-grounded, same-topic, and wrong because of relation/state/order, not because it is unrelated.",
        "Single choice: options A-D and one correct_option.",
        "Multi-select: options A-E and gold as a non-empty list of option letters.",
        "Ordering: items I1-I4 and gold as a complete ID permutation. Use the `items` key, not `options`, for ordering. The ordering candidate MUST include a non-empty `question` string; never omit that field.",
        "For every candidate, evidence_groups must contain valid canonical turn_refs and should align each group with the supplied contract anchors; extra supporting refs are allowed only when they are necessary.",
        "The candidate question must not copy a future resolution sentence as its answer.",
        "Return JSON only with exactly one top-level key candidates.",
        "REASONING CONTRACT:",
        json.dumps(contract, ensure_ascii=False, indent=2),
        "OUTPUT SHAPE:",
        json.dumps(output_schema, ensure_ascii=False, indent=2),
        "AUTHORITATIVE DIALOGUE:",
        *dialogue,
    ])
    return {
        "prompt_version": CANDIDATE_PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "candidate_minimums": {"single_choice": 2, "multi_select": 1, "ordering": 1},
    }


def build_candidate_repair_prompt(
    contract: Mapping[str, Any], raw_candidates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """One isolated schema repair for a failed intent batch.

    This is not the production QA Writer: it receives the already mined batch
    and may only repair missing/invalid structural fields while preserving the
    relation and answer content.
    """
    prompt = "\n".join([
        "ROLE: H4 QA mining candidate schema repair.",
        "Repair the supplied batch for exactly one intent.",
        "Return exactly four candidates: 2 single_choice, 1 multi_select, 1 ordering.",
        "Preserve the questions, options/items, gold answers, and evidence whenever valid.",
        "Every row MUST contain a non-empty question field.",
        "Ordering MUST use items with I1,I2,I3,I4 and a complete gold permutation.",
        "Do not add surface final-state recall questions or unrelated distractors.",
        "Return JSON only with top-level key candidates.",
        "REASONING CONTRACT:", json.dumps(contract, ensure_ascii=False, indent=2),
        "SUPPLIED BATCH:", json.dumps(raw_candidates or {}, ensure_ascii=False, indent=2),
    ])
    return {
        "prompt_version": CANDIDATE_PROMPT_VERSION + "_repair",
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "candidate_minimums": {"single_choice": 2, "multi_select": 1, "ordering": 1},
    }
