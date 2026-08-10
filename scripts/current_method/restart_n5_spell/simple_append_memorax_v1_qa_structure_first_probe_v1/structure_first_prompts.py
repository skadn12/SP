"""Single-call structure/evidence-first QA Writer prompt."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from ..simple_append_v1.qa import build_evidence_alias_registry
from .structure_first_contract import (
    HARD_MAX_EVIDENCE_TURNS,
    PREFERRED_MAX_EVIDENCE_TURNS,
    PREFERRED_MIN_EVIDENCE_TURNS,
    QUESTION_CAPACITY_MAX,
    STRUCTURE_FIRST_SCHEMA_VERSION,
)


STRUCTURE_FIRST_PROMPT_VERSION = "structure_evidence_first_writer_prompt_v1"


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dialogue_lines(expanded: Mapping[str, Any]) -> list[str]:
    registry = build_evidence_alias_registry(expanded)
    turns = {str(row.get("turn_ref")): row
             for row in expanded.get("turns") or []}
    return [
        f"{entry['evidence_ref']} | {entry['origin']} | "
        f"{turns[str(entry['turn_ref'])].get('scene_ref') or turns[str(entry['turn_ref'])].get('session_ref') or ''} | "
        f"{turns[str(entry['turn_ref'])].get('speaker')} | "
        f"{turns[str(entry['turn_ref'])].get('text')}"
        for entry in registry["entries"]
    ]


def build_structure_first_prompt(
    expanded: Mapping[str, Any], memory_overview: Mapping[str, Any],
) -> dict[str, Any]:
    if str(expanded.get("root_id") or "") != str(
            memory_overview.get("root_id") or ""):
        raise ValueError("structure_first_memory_root_mismatch")
    registry = build_evidence_alias_registry(expanded)
    prompt = "\n".join([
        "ROLE: Structure/evidence-first QA Writer.",
        "Return one JSON object only. Do not reveal private chain-of-thought. Record",
        "only the compact, externally checkable structure_plan conclusions requested",
        "below. The complete dialogue is the only authoritative fact source.",
        "",
        "MANDATORY GENERATION ORDER",
        "For each candidate, complete these checks in this exact order before writing",
        "question or options:",
        "1. Identify one real local reasoning structure in the transcript.",
        f"2. Lock {PREFERRED_MIN_EVIDENCE_TURNS}-{PREFERRED_MAX_EVIDENCE_TURNS} minimal sufficient D### turns; never exceed {HARD_MAX_EVIDENCE_TURNS}.",
        "3. Select the one discriminative variable required by that structure.",
        "4. Verify that exactly one proposed option is supported as the answer.",
        "5. Locate three plausible near misses in the same locked local evidence.",
        "6. Only then materialize question, four options, and gold_text.",
        "If any step fails, skip the candidate. Never write an MCQ first and search",
        "backwards for evidence.",
        "",
        "ALLOWED STRUCTURES AND MATERIALIZATION TYPES",
        "temporal_sequence -> question_type temporal_sequence, variable event_order.",
        "state_change -> question_type state_change, variable previous_vs_current_state.",
        "information_revision -> question_type information_revision, variable old_vs_revised_plan.",
        "intended_vs_executed -> question_type cross_turn_comparison, variable same_dimension_comparison.",
        "information_visibility -> question_type information_visibility, variable knowledge_holder_at_timepoint.",
        "task_dependency_or_constraint -> same question_type, variable prerequisite.",
        "causal or competing_result -> question_type causal, variable dialogue_specific_cause.",
        "cross_turn_comparison -> same question_type, variable same_dimension_comparison.",
        "Do not generate considered-vs-adopted questions in this probe. Do not turn",
        "actor recall into visibility, an ordinary decision into revision, or a causal",
        "association into a prerequisite.",
        "",
        "TYPE CONTRACTS (unchanged deterministic downstream contract)",
        "temporal_sequence: type_contract={events:[{event,turn_ids},...]}; at least",
        "three events, and every option is a permutation of the same event texts.",
        "state_change/information_revision: type_contract={subject,old_state,update,",
        "new_state,before_turn_ids,update_turn_ids,after_turn_ids}. Ask exactly:",
        "After <update>, what was <subject>'s new state? gold_text equals new_state",
        "and old_state is exactly one incorrect option.",
        "information_visibility: type_contract={fact,timepoint,candidate_knowers,turn_ids};",
        "ask only who knew the fact at that timepoint. Skip if multiple candidates",
        "satisfy the answer slot.",
        "task_dependency_or_constraint: type_contract={prerequisite,dependent_outcome,",
        "turn_ids}; the dialogue must explicitly make the outcome conditional on it.",
        "cross_turn_comparison: type_contract={comparison_dimension,left,right,",
        "left_turn_ids,right_turn_ids}; both sides use exactly the same dimension.",
        "causal: type_contract={dialogue_specific_cause,competing_causes,turn_ids};",
        "skip if more than one option is a true cause under the question wording.",
        "",
        "LOCAL NEAR-MISS CONTRACT",
        "Each wrong option must have one distractor_sources entry. option_text must",
        "exactly equal that wrong option. source_turn_ids must be a nonempty subset of",
        "the locked evidence. near_miss_kind is exactly one of: nearby_event,",
        "previous_state, wrong_timepoint, wrong_actor, proposed_not_adopted,",
        "intended_not_executed, nearby_alternative_cause,",
        "nearby_alternative_result, event_order_permutation, cross_pairing.",
        "Use real local material. Do not manufacture weak opposites based on ignoring,",
        "hiding, blaming, or quitting forever. Skip when three local competitors do not exist.",
        "",
        "GOLD UNIQUENESS CONTRACT",
        "gold_uniqueness.unique must be true. supported_option_texts must contain",
        "exactly [gold_text]. support_turn_ids must be a nonempty subset of locked",
        "evidence. If two options are supported, skip the candidate.",
        "",
        "STRUCTURE_PLAN FIELD CONTRACT",
        "Each candidate has structure_plan with exactly: structure_type,",
        "local_evidence_turn_ids, discriminative_variable, gold_uniqueness,",
        "distractor_sources. local_evidence_turn_ids must exactly equal both",
        "evidence_turn_ids and reasoning_structure.supporting_turn_ids in the same order.",
        "gold_uniqueness has exactly: unique, supported_option_texts, support_turn_ids.",
        "Each distractor source has exactly: option_text, source_turn_ids, near_miss_kind.",
        "",
        "OUTPUT CONTRACT (field description, not an answer example)",
        f"structure_first_schema_version={STRUCTURE_FIRST_SCHEMA_VERSION}.",
        f"questions is an array of 0-{QUESTION_CAPACITY_MAX}; {QUESTION_CAPACITY_MAX} is capacity, not a target.",
        "candidate_id is Q01, Q02, ... in returned order. Each question has exactly:",
        "candidate_id, question_type, source_scope, structure_plan,",
        "reasoning_structure, question, options, gold_text, evidence_turn_ids.",
        "reasoning_structure has exactly: discriminative_variable,",
        "supporting_turn_ids, short_rationale, type_contract. options has exactly",
        "A,B,C,D. gold_text exactly equals one option. source_scope is derived from",
        "evidence: parent_only, child_only, or parent_child.",
        "abstention_summary has estimated_skipped_count and reason_codes. Allowed",
        "reason codes: no_discriminative_structure, generic_commonsense,",
        "weak_distractors, evidence_over_limit, duplicate_candidate, other.",
        "An empty question list requires a positive skipped count and nonempty reasons.",
        "There is no minimum question count and no coverage quota.",
        "",
        "MEMORAX OVERVIEW (NON-AUTHORITATIVE; NEVER EVIDENCE)",
        _compact(memory_overview),
        "",
        "FULL DIALOGUE (AUTHORITATIVE COMPLETE PARENT + CHILD)",
        *_dialogue_lines(expanded),
        "",
        "FINAL TASK",
        "Discover local structures first, freeze minimal evidence, check unique gold and",
        "three transcript-grounded near misses, and only then materialize MCQs. Return",
        "fewer questions or an explained empty list whenever a contract cannot be met.",
        "Return one JSON object and no other text.",
    ])
    return {
        "prompt_version": STRUCTURE_FIRST_PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "evidence_alias_registry_sha256": registry["registry_sha256"],
        "question_capacity_max": QUESTION_CAPACITY_MAX,
    }
