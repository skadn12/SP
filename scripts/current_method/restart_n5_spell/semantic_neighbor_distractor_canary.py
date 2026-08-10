from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Sequence

from .contrastive_distractor_canary import (
    CONTRASTIVE_DISTRACTOR_SCHEMA,
    materialize_contrastive_candidate,
    option_length_diagnostics,
)
from .core import canonical_json, sha256_bytes


SEMANTIC_NEIGHBOR_CANARY_VERSION = "semantic_neighbor_distractor_canary_v1"
SEMANTIC_NEIGHBOR_PROMPT_VERSION = "semantic_neighbor_distractor_retry_v1"
SEMANTIC_NEIGHBOR_SCHEMA_VERSION = "semantic_neighbor_distractor_schema_v1"
SEMANTIC_NEIGHBOR_PLACEMENT_VERSION = "semantic_neighbor_answer_placement_v1"

_TARGET_LETTERS = {
    "johns_motivation_for_politics": "A",
    "marias_volunteering_inspiration": "D",
}


def semantic_neighbor_schema_sha256() -> str:
    return sha256_bytes(canonical_json(CONTRASTIVE_DISTRACTOR_SCHEMA).encode())


def semantic_neighbor_prompt_sha256() -> str:
    material = (
        SEMANTIC_NEIGHBOR_PROMPT_VERSION
        + SEMANTIC_NEIGHBOR_SCHEMA_VERSION
        + inspect.getsource(build_semantic_neighbor_prompt)
        + canonical_json(CONTRASTIVE_DISTRACTOR_SCHEMA)
    )
    return sha256_bytes(material.encode())


def deterministic_semantic_neighbor_placement(qa_id: str) -> dict[str, Any]:
    if qa_id not in _TARGET_LETTERS:
        raise ValueError("semantic_neighbor_placement_input_invalid")
    seed_material = canonical_json({
        "version": SEMANTIC_NEIGHBOR_PLACEMENT_VERSION,
        "qa_id": qa_id,
    })
    return {
        "answer_placement_version": SEMANTIC_NEIGHBOR_PLACEMENT_VERSION,
        "placement_seed": int(sha256_bytes(seed_material.encode())[:8], 16),
        "target_letter": _TARGET_LETTERS[qa_id],
        "content_dependent": False,
    }


def build_semantic_neighbor_prompt(
    *,
    frozen_dialogue: Sequence[Mapping[str, Any]],
    base_qa: Mapping[str, Any],
    previous_distractors: Sequence[str],
) -> str:
    if len(previous_distractors) != 3:
        raise ValueError("previous_distractors_must_have_exactly_three_items")
    payload = {
        "frozen_question_answer": {
            "qa_id": str(base_qa["qa_id"]),
            "question": str(base_qa["question"]),
            "correct_answer_text": str(base_qa["correct_answer_text"]),
            "answer_support_span_ids": list(base_qa["answer_support_span_ids"]),
        },
        "previous_failed_distractors": list(previous_distractors),
        "previous_final_decision": "bad_ground",
        "previous_grounding_diagnosis": "bad_ground_semantic_leakage",
        "full_frozen_dialogue": [dict(turn) for turn in frozen_dialogue],
    }
    rules = [
        "Return one JSON object containing exactly three distinct distractors.",
        "Keep the frozen question, correct answer, and evidence IDs unchanged; output distractors only.",
        "Regenerate all three distractors. Do not reuse any previous failed distractor.",
        "All distractors must stay in the exact semantic domain asked by the question.",
        "Each distractor must answer the same question with the same answer type, causal role, grammatical structure, level of detail, specificity, topical vocabulary, and approximately the same length as the correct answer.",
        "Construct each distractor as a plausible counterfactual semantic neighbor of the correct answer by changing exactly one decisive slot.",
        "Allowed single-slot changes are actor or source, trigger or motivation, time, location, initial state, revised state, consequence, affected group, or ordering.",
        "Keep every non-decisive slot parallel to the correct answer. Do not combine several unrelated changes in one distractor.",
        "Plausible counterfactual distractors do not need to be facts stated elsewhere in the dialogue, but they must be false answers to this exact question.",
        "Do not mine unrelated real events, occupations, hobbies, education, military service, or hardships merely because they appear in the dialogue.",
        "Without the dialogue, no option should be preferable because it is more natural, more topical, more detailed, more causally coherent, or longer.",
        "Do not copy the correct answer, create a synonymous second correct answer, contradict the question's presupposition, or repeat a distractor.",
        "Use the full dialogue only to preserve falsity and avoid accidentally creating another supported answer.",
        "Return JSON only, without Markdown, explanation, evidence, labels, or extra fields.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {SEMANTIC_NEIGHBOR_PROMPT_VERSION}",
        "SEMANTIC-NEIGHBOR DISTRACTOR RULES\n- " + "\n- ".join(rules),
        "FROZEN INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                        sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            CONTRASTIVE_DISTRACTOR_SCHEMA, ensure_ascii=False,
            sort_keys=True, indent=2),
    ])


__all__ = [
    "SEMANTIC_NEIGHBOR_CANARY_VERSION",
    "SEMANTIC_NEIGHBOR_PLACEMENT_VERSION",
    "SEMANTIC_NEIGHBOR_PROMPT_VERSION",
    "SEMANTIC_NEIGHBOR_SCHEMA_VERSION",
    "build_semantic_neighbor_prompt",
    "deterministic_semantic_neighbor_placement",
    "materialize_contrastive_candidate",
    "option_length_diagnostics",
    "semantic_neighbor_prompt_sha256",
    "semantic_neighbor_schema_sha256",
]
