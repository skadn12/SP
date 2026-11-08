"""Compact prompt for selecting facts from frozen candidate bundles."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..core import sha256_bytes
from .fact_bundle_writer_contract import SCHEMA_VERSION


PROMPT_VERSION = "fact_bundle_writer_prompt_v1"


def build_fact_bundle_writer_prompt(
    facts: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fact_index = {str(row.get("fact_id")): row for row in facts}
    bundle_lines: list[str] = []
    for bundle in bundles:
        ids = [str(value) for value in bundle.get("fact_ids") or []]
        bundle_lines.append(json.dumps({
            "bundle_id": bundle.get("bundle_id"),
            "bundle_class": bundle.get("bundle_class"),
            "window_ids": bundle.get("window_ids"),
            "facts": [
                {"fact_id": fact_id, "fact_text": fact_index.get(fact_id, {}).get("fact_text")}
                for fact_id in ids
            ],
        }, ensure_ascii=False, separators=(",", ":")))
    prompt = "\n".join([
        "ROLE: grounded reverse-MCQ selector.",
        "The facts and bundles below are frozen transcript-grounded records.",
        "Select only from these records. Do not invent facts, option text,",
        "evidence, IDs, answer letters, or source positions.",
        "Return exactly one JSON object and nothing else.",
        "",
        "TASK",
        "For each question, choose one bundle and write a question whose answer",
        "is exactly one fact in that bundle. Set gold_fact_id to that fact and",
        "set distractor_fact_ids to the other three facts in the same bundle.",
        "The three distractors must be real, local, plausible competitors on the",
        "same answer dimension; if no bundle supports a unique grounded question,",
        "skip it. Do not reuse a fact as gold and distractor in one item.",
        "Use 0-12 questions; 12 is capacity, not a quota.",
        "Allowed question_type values: causal, temporal_sequence, state_change,",
        "cross_turn_comparison, relationship_or_decision, cross_context_synthesis,",
        "information_visibility, information_revision, task_dependency_or_constraint.",
        "answer_dimension must be a short concrete dimension such as order, cause,",
        "result, state, decision, actor, time, relationship, information, or comparison.",
        "Do not use weak invented opposites such as ignore, hide, blame, quit,",
        "fired, or promoted unless the frozen fact itself contains that event.",
        "",
        "OUTPUT CONTRACT",
        f"fact_bundle_writer_schema_version must be {SCHEMA_VERSION!r}.",
        "Top-level keys are exactly fact_bundle_writer_schema_version and questions.",
        "Each question object has exactly bundle_id, question_type, question,",
        "answer_dimension, gold_fact_id, distractor_fact_ids.",
        "distractor_fact_ids is exactly three distinct IDs from the same bundle.",
        "Never output options, gold letters, fact text, evidence refs, positions,",
        "canonical IDs, source scopes, or extra metadata.",
        "",
        "FROZEN CANDIDATE BUNDLES",
        *bundle_lines,
        "",
        "Return only the JSON object.",
    ])
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "fact_count": len(facts),
        "bundle_count": len(bundles),
    }
