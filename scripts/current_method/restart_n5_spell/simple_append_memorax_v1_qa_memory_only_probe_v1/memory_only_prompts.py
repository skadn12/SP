"""Writer prompt for the isolated Memory-only QA experiment."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from .memory_only_contract import QUESTION_TYPES, source_ref_rows

PROMPT_VERSION = "qa_memory_only_writer_prompt_v2_exact_question_types"


def build_memory_only_writer_prompt(
    overview: Mapping[str, Any], *, maximum_questions: int = 12,
) -> dict[str, Any]:
    rows = source_ref_rows(overview)
    memory_text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    prompt = "\n".join([
        "ROLE: Memory-only exploratory MCQ writer.",
        "The following is the complete input for this probe: frozen Memorax memory records.",
        "No transcript, turn IDs, or hidden answer key is available to you.",
        "Use only the supplied memory records and their memory_ref labels.",
        "This is an exploratory diagnostic, not final training-data admission.",
        "",
        "TASK",
        f"Produce 0 to {maximum_questions} questions; the maximum is capacity, not a quota.",
        "A question must target a concrete event, decision, state, relationship, cause,",
        "result, time/order, or dependency present in one or more supplied records.",
        "Use only these exact question_type values; do not invent synonyms:",
        ", ".join(sorted(QUESTION_TYPES)) + ".",
        "Use one or more memory_ref values that actually support the topic.",
        "The correct option must be uniquely supported by the supplied Memory.",
        "Wrong options should be controlled near-misses: change one actor, time, state,",
        "plan, cause, result, or detail while staying in the same Memory event/thread.",
        "Do not use generic opposites such as ignore, hide, blame, quit, fired, or promoted",
        "unless the supplied Memory explicitly contains that competing fact.",
        "Do not pretend that Memory is transcript evidence and do not output turn IDs.",
        "If Memory does not support three plausible same-dimension near-misses, skip the item.",
        "",
        "OUTPUT CONTRACT",
        "Return exactly one JSON object with exactly the key questions.",
        "Each question object has exactly question_type, question, options, gold_text,",
        "memory_source_refs.",
        "question_type is a contract label, not a free-form description. For example,",
        "use causal rather than cause, relationship_or_decision rather than relationship,",
        "temporal_sequence rather than time/order, and task_dependency_or_constraint",
        "rather than dependency. If no exact type fits, skip the question.",
        "options has exactly A, B, C, D. gold_text must exactly equal one option value.",
        "memory_source_refs is a non-empty array of supplied memory_ref labels.",
        "Do not output correct_option; the program derives it from gold_text.",
        "Do not output evidence_turn_refs, source_scope, canonical IDs, or extra fields.",
        "",
        "FROZEN MEMORY RECORDS",
        memory_text,
        "",
        "Return only the JSON object.",
    ])
    return {"prompt_version": PROMPT_VERSION, "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "character_count": len(prompt), "memory_record_count": len(rows),
            "memory_overview_sha256": overview.get("memory_overview_sha256"),
            "writer_input_mode": "memory_only_no_transcript"}
