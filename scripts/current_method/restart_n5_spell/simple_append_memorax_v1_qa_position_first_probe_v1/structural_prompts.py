"""Prompt builder for discriminative-variable-first Structural QA Writer v19."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from ..simple_append_v1.qa import build_evidence_alias_registry
from .structural_contract import (
    DISCRIMINATIVE_VARIABLES,
    MAX_EVIDENCE_TURNS,
    STRUCTURAL_QA_SCHEMA_VERSION,
    QUESTION_CAPACITY_MAX,
)


STRUCTURAL_QA_PROMPT_VERSION = "structural_qa_writer_prompt_v19_2"


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dialogue_lines(expanded: Mapping[str, Any]) -> list[str]:
    registry = build_evidence_alias_registry(expanded)
    turns = {str(row.get("turn_ref")): row
             for row in expanded.get("turns") or []}
    lines: list[str] = []
    for entry in registry["entries"]:
        turn = turns[str(entry["turn_ref"])]
        lines.append(
            f"{entry['evidence_ref']} | {entry['origin']} | "
            f"{turn.get('scene_ref') or turn.get('session_ref') or ''} | "
            f"{turn.get('speaker')} | {turn.get('text')}")
    return lines


def build_structural_qa_prompt(
    expanded: Mapping[str, Any], memory_overview: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one complete-dialogue Writer request with concise diagnostics."""
    if str(expanded.get("root_id") or "") != str(
            memory_overview.get("root_id") or ""):
        raise ValueError("structural_qa_memory_root_mismatch")
    registry = build_evidence_alias_registry(expanded)
    allowed_variables = ", ".join(sorted(DISCRIMINATIVE_VARIABLES))
    prompt = "\n".join([
        "ROLE: Structural QA Writer v19.2.",
        "Return one JSON object only. Do not reveal private chain-of-thought. The",
        "short_rationale is a concise, externally verifiable structural claim.",
        "",
        "GOAL",
        "Return only independently valid four-option MCQs. First count the real",
        "structures that meet a type contract; it is correct and preferred to return",
        f"fewer items or an empty list. {QUESTION_CAPACITY_MAX} is a capacity ceiling, never a target or",
        "quota. Abstain instead of forcing a weak question or filling unused capacity.",
        "",
        "SELECTION ORDER",
        "1. Locate a real reasoning structure in the transcript.",
        "2. Name its discriminative variable.",
        "3. Check that three near-miss alternatives can compete on that same variable.",
        "4. Select the minimal sufficient evidence, then materialize the MCQ.",
        "",
        "CANONICAL DISCRIMINATIVE VARIABLES",
        "Use exactly the canonical variable for each type: temporal_sequence=event_order;",
        "state_change=previous_vs_current_state; information_revision=old_vs_revised_plan;",
        "information_visibility=knowledge_holder_at_timepoint;",
        "task_dependency_or_constraint=prerequisite; cross_turn_comparison=",
        "same_dimension_comparison; relationship_or_decision=adopted_decision;",
        "causal=dialogue_specific_cause; cross_context_synthesis=cross_context_synthesis.",
        "Do not copy a type_contract field name unless it is the",
        "canonical variable listed above.",
        "Registry reference (only these names are available): " + allowed_variables,
        "Do not use responsibility, maturity, positivity, reasonableness, or generic",
        "commonsense as the main discriminator.",
        "",
        "QUESTION-TYPE CONTRACTS",
        "temporal_sequence: type_contract={events:[{event,turn_ids},...]}; at least",
        "three transcript events. Every option must copy each event description exactly",
        "once and differ only in order; the gold uses the transcript's chronological order.",
        "state_change or information_revision: type_contract={subject,old_state,update,new_state,",
        "before_turn_ids,update_turn_ids,after_turn_ids}; all three stages are required.",
        "Use the exact question form: After <update>, what was <subject>'s new state?",
        "gold_text must equal new_state, and old_state must be one incorrect option.",
        "information_visibility: type_contract={fact,timepoint,candidate_knowers,turn_ids};",
        "candidate_knowers must be exactly the four option answer candidates. Use the exact",
        "question form: Who knew that <fact> at the timepoint <timepoint>? Do not add an action, actor",
        "or outcome clause such as 'and encouraged ...'. This type tests knowledge only.",
        "task_dependency_or_constraint: type_contract={prerequisite,dependent_outcome,",
        "turn_ids}; use the exact question form: What had to happen before <dependent_outcome>?",
        "The transcript must state that the outcome depends on the prerequisite.",
        "cross_turn_comparison: type_contract={comparison_dimension,left,right,",
        "left_turn_ids,right_turn_ids}; both sides use the same concrete dimension.",
        "relationship_or_decision: type_contract={adopted_decision,",
        "considered_alternative,turn_ids}; use null only if no competing plan exists.",
        "causal: type_contract={dialogue_specific_cause,competing_causes,turn_ids};",
        "competing_causes must contain at least two plausible local alternatives.",
        "cross_context_synthesis: type_contract={synthesis,evidence_groups:[",
        "{group,turn_ids},...]}; at least two independent evidence groups are required.",
        "",
        "DISTRACTORS",
        "All four options must fill the same answer slot and compete on the same local",
        "event/state dimension. Prefer wrong actor, timepoint, order, previous state,",
        "considered-but-not-adopted plan, intended-but-not-executed action, or nearby",
        "incorrect result. Never use obvious moral opposites such as ignore, hide, blame,",
        "quit forever, or do nothing against a responsible gold. If three near-misses are",
        "not available, skip the candidate.",
        "",
        "EVIDENCE",
        f"Use 1-{MAX_EVIDENCE_TURNS} exact D### references. Select the smallest subset",
        "that proves both the gold and the stated reasoning structure. Do not cite an",
        "entire session or scene. evidence_turn_ids and reasoning_structure.",
        "supporting_turn_ids must contain exactly the same unique references. Every",
        "type_contract turn_ids list must be a nonempty subset of that evidence.",
        "",
        "SOURCE COVERAGE",
        "Derive source_scope from evidence: parent_only, child_only, or parent_child.",
        "When any good candidate exists, strongly prefer at least one candidate supported",
        "by generated_extension turns. Never lower quality merely to satisfy coverage.",
        "",
        "OUTPUT FIELD CONTRACT (this is a schema description, not an answer example)",
        f"structural_qa_schema_version must be {STRUCTURAL_QA_SCHEMA_VERSION}.",
        f"questions: array with length 0-{QUESTION_CAPACITY_MAX}. candidate_id values are",
        "Q01, Q02, ... in list order. gold_text must exactly equal exactly one option.",
        "Each included question has exactly: candidate_id, question_type, source_scope,",
        "reasoning_structure, question, options, gold_text, evidence_turn_ids.",
        "reasoning_structure has exactly: discriminative_variable, supporting_turn_ids,",
        "short_rationale, type_contract. options has exactly A, B, C, D.",
        "abstention_summary.estimated_skipped_count: non-negative integer.",
        "abstention_summary.reason_codes: array using only:",
        "no_discriminative_structure,",
        "generic_commonsense, weak_distractors, evidence_over_limit,",
        "duplicate_candidate, other.",
        "If questions is empty, estimated_skipped_count must be positive and",
        "reason_codes must be nonempty. An unexplained empty batch is invalid.",
        "",
        "MEMORAX OVERVIEW (NON-AUTHORITATIVE)",
        "This is retained only to keep the baseline input controlled. It may suggest a",
        "topic, but it is not evidence and cannot support any question or answer.",
        _compact(memory_overview),
        "",
        "FULL DIALOGUE (AUTHORITATIVE COMPLETE PARENT + CHILD)",
        *_dialogue_lines(expanded),
        "",
        "FINAL TASK",
        "Inspect the supplied dialogue for genuine dialogue-specific discriminative",
        "structures. Materialize only candidates satisfying every contract. Returning",
        "fewer strong questions is preferred to weak questions. Return zero only when",
        "no candidate satisfies any valid contract, and explain that abstention using",
        "the required summary fields. Return one JSON object and no other text.",
    ])
    return {
        "prompt_version": STRUCTURAL_QA_PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "evidence_alias_registry_sha256": registry["registry_sha256"],
        "question_capacity_max": QUESTION_CAPACITY_MAX,
    }
