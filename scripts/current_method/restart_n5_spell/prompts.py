"""Versioned prompt builders for the isolated restart-n5 SPELL-style route.

Method inspiration: Tongyi-Zhiwen/Qwen-Doc SPELL, commit
4e5aee32280febb7b0844237ba087f837e3c8905, Apache-2.0 as declared by
SPELL/README.md. No upstream prompt is copied verbatim: the official single-item
document protocol is redesigned here for rolling dialogue and strict 3-item MCQ JSON.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Iterable, Mapping, Sequence

from .questioner_contract import (
    SPAN_CATALOG_VERSION, REASONING_CONTRACT_VERSION, QUESTIONER_SCHEMA_VERSION,
    build_span_catalog, default_reasoning_contracts, model_reasoning_contract_view,
    model_span_catalog_view, span_catalog_index, validate_reasoning_contracts,
)

REALIZER_PROMPT_VERSION = "rolling_dialogue_realizer_v4_local_entity_ledger"
QUESTIONER_PROMPT_VERSION = "spell_dialogue_mcq_3x_v7_span_contract_local_derivation"
RESPONDER_PROMPT_VERSION = "mcq_boxed_letter_v1"

REALIZER_SYSTEM_INSTRUCTION = """You are the Realizer in a rolling dialogue-expansion pipeline.
Realize exactly one continuous session from the supplied plan and continuity state. Treat the
parent dialogue and accepted sessions as immutable facts. Return strict JSON only: no Markdown,
code fences, commentary, questions for the operator, or text outside the JSON object."""

QUESTIONER_SYSTEM_INSTRUCTION = """You are the MCQ Questioner in a SPELL-inspired frontier-data
pipeline. The frozen dialogue is the only factual source. Generate exactly three independent,
unambiguous four-option questions that require dialogue comprehension and compositional reasoning.
Return one strict JSON array only: no Markdown, code fences, analysis, or surrounding text."""

RESPONDER_PROMPT = """Answer the multiple-choice question using the supplied context.

Return exactly one final answer in this format:
<box>A</box>

Replace A with exactly one of A, B, C, or D.

Do not output reasoning, punctuation, Markdown, or any text outside the box tags."""

SOLVER_NOTHINK_RESPONDER_PROMPT = RESPONDER_PROMPT

REALIZER_OUTPUT_SCHEMA: dict[str, Any] = {
    "dialogue_turns": [{"speaker": "string", "text": "string"}],
    "used_parent_anchor_ids": ["string"],
    "advanced_event_arc_ids": ["string"],
    "continued_unresolved_hook_ids": ["string"],
    "durable_assertion_candidates": [{
        "subject": "string", "predicate": "string", "object_or_state": "string",
        "supporting_turn_indices": [0],
    }],
    "local_scene_facts": ["string"],
    "resolved_hook_ids": ["existing hook ID copied from allowed_unresolved_hooks"],
    "new_unresolved_hooks": [{"description": "string"}],
}

QUESTIONER_GENERATED_ITEM_SCHEMA: dict[str, Any] = {
    "qa_id": "string",
    "question": "string",
    "correct_answer_text": "string",
    "distractors": ["string", "string", "string"],
    "answer_support_span_ids": ["span_id", "span_id"],
    "answer_evidence_values": [{"span_id": "span_id", "value": "minimal meaningful phrase"}],
    "distractor_support": [{
        "distractor_index": 0,
        "span_ids": ["span_id"],
        "evidence_values": ["minimal meaningful phrase"],
    }],
    "correct_answer_explanation": "string",
    "distractor_explanations": ["string", "string", "string"],
    "reasoning_steps": ["string"],
}

# Backward-compatible exported name: this is now the model-facing, unlabeled schema.
QUESTIONER_ITEM_SCHEMA = QUESTIONER_GENERATED_ITEM_SCHEMA


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _list(values: Iterable[Any]) -> list[Any]:
    return list(values)


def build_realizer_prompt(
    *, session_index: int, narrative_beats: Sequence[Mapping[str, Any]],
    recent_sessions: Sequence[Mapping[str, Any]], continuity_summary: Any,
    active_plans: Sequence[Any], pending_decisions: Sequence[Any],
    superseded_information: Sequence[Any], character_knowledge: Mapping[str, Any],
    visibility_boundaries: Sequence[Any], unresolved_hooks: Sequence[Any],
    allowed_people: Sequence[Any], session_location: str,
    allowed_event_arc_ids: Sequence[str], forbidden_conflicts: Sequence[Any],
    parent_anchors: Sequence[Mapping[str, Any]] = (),
    allowed_durable_subjects: Sequence[str] = (),
    durable_predicate_whitelist: Sequence[str] = (),
) -> str:
    """Build one session prompt. API turn IDs are forbidden and assigned after acceptance."""
    if session_index < 1:
        raise ValueError("session_index must be >= 1")
    if not narrative_beats:
        raise ValueError("at least one narrative beat is required")
    if session_index > 1 and not unresolved_hooks:
        raise ValueError("session 2+ requires at least one unresolved hook")
    start_rule = (
        "Session 1: explicitly continue at least one supplied parent anchor or opening hook."
        if session_index == 1 else
        "Session 2+: explicitly continue at least one supplied unresolved hook; identify its ID in continued_unresolved_hook_ids and optionally resolved_hook_ids."
    )
    payload = {
        "session_index": session_index,
        "narrative_beats_to_advance": _list(narrative_beats),
        "recent_accepted_sessions_full_text": _list(recent_sessions)[-2:],
        "earlier_continuity_summary": continuity_summary,
        "active_plans": _list(active_plans),
        "pending_decisions": _list(pending_decisions),
        "superseded_information_do_not_treat_as_active": _list(superseded_information),
        "character_knowledge": dict(character_knowledge),
        "visibility_boundaries": _list(visibility_boundaries),
        "allowed_unresolved_hooks": _list(unresolved_hooks),
        "parent_anchors": _list(parent_anchors),
        "allowed_parent_anchor_ids": [x.get("anchor_id") for x in parent_anchors],
        "allowed_speaker_names": _list(allowed_people),
        "fixed_session_location_do_not_echo_as_metadata": session_location,
        "allowed_event_arc_ids": _list(allowed_event_arc_ids),
        "forbidden_conflicts": _list(forbidden_conflicts),
        "allowed_durable_subjects": _list(allowed_durable_subjects),
        "durable_predicate_whitelist": _list(durable_predicate_whitelist),
    }
    rules = [
        start_rule,
        "Advance at least one supplied event arc. One session may cover adjacent beats as one continuous scene.",
        "Every advanced_event_arc_ids value must be copied exactly from allowed_event_arc_ids; never return an arc summary in place of its ID.",
        "Every used_parent_anchor_ids value must be copied exactly from allowed_parent_anchor_ids.",
        "Every continued_unresolved_hook_ids and resolved_hook_ids value must be copied exactly from an ID in allowed_unresolved_hooks.",
        "All *_ids fields contain IDs only: never summaries, names, descriptions, explanations, or newly invented IDs.",
        "ID namespaces are disjoint: parent anchors use P*, event arcs use E*, and unresolved hooks use H*. Never place a P* or E* value in a hook-ID field.",
        "new_unresolved_hooks contains objects with description only. Do not output hook_id; the caller assigns it after acceptance.",
        "Do not classify or list named entities, characters, institutions, or locations. The caller derives character and location ledgers locally.",
        "Introduce at most one necessary minor problem; do not branch into an independent story.",
        "Do not repeat or paraphrase existing dialogue merely to increase length; do not add filler.",
        "Do not contradict immutable facts, revive superseded state, cross visibility boundaries, or move away from the fixed session location.",
        "Do not invent durable biographical, relational, ownership, commitment, or outcome facts unless a supplied beat requires them.",
        "Never output question, options, gold, answer, QA operation, evidence plan, quiz, or verifier content.",
        "Do not output turn_id. dialogue_turns preserve order; the caller assigns globally unique monotonic IDs after acceptance.",
        "supporting_turn_indices are zero-based indices into this response's dialogue_turns only; never invent global IDs.",
        "Every durable assertion candidate must be atomic and directly supported by exact words in its indexed turns.",
        "A durable assertion subject must be copied from allowed_durable_subjects or appear explicitly in every supporting scene quote. Its predicate must be copied exactly from durable_predicate_whitelist.",
        "Top-level keys must match the required schema exactly. Do not add session_location, new_named_entities, event_arcs_closed, turn/session IDs, token counts, or bookkeeping metadata.",
        "Every output key in the schema is required. Use [] rather than omitting an empty collection.",
    ]
    if not unresolved_hooks:
        rules.append("There are no allowed existing unresolved hooks in this request: continued_unresolved_hook_ids and resolved_hook_ids MUST both be [].")
    example = {
        **REALIZER_OUTPUT_SCHEMA,
        "dialogue_turns": [{"speaker": "Allowed Speaker", "text": "A concrete continuation."}],
        "durable_assertion_candidates": [],
    }
    return "\n\n".join([
        REALIZER_SYSTEM_INSTRUCTION,
        f"PROMPT_VERSION: {REALIZER_PROMPT_VERSION}",
        "SESSION RULES\n- " + "\n- ".join(rules),
        "DYNAMIC INPUT\n" + _json(payload),
        "REQUIRED OUTPUT SCHEMA\n" + _json(REALIZER_OUTPUT_SCHEMA),
        "MINIMAL LEGAL SHAPE (illustrative values only; do not copy facts)\n" + _json(example),
    ])


def build_questioner_prompt(
    *, frozen_dialogue: Sequence[Mapping[str, Any]], valid_turn_ids: Sequence[str],
    valid_session_ids: Sequence[str], assigned_gold_letters: Sequence[str],
    recent_frontier_history: Sequence[Mapping[str, Any]], batch_index: int = 0,
    span_catalog: Mapping[str, Any] | None = None,
    reasoning_contracts: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Build a strict three-item batch protocol; each item remains independently gateable."""
    letters = [str(x).upper() for x in assigned_gold_letters]
    if len(letters) != 3 or any(x not in "ABCD" for x in letters):
        raise ValueError("assigned_gold_letters must contain exactly three A/B/C/D letters")
    if len(valid_turn_ids) != len(set(valid_turn_ids)):
        raise ValueError("valid_turn_ids must be unique")
    if not frozen_dialogue:
        raise ValueError("frozen_dialogue must not be empty")
    catalog = dict(span_catalog or build_span_catalog(list(frozen_dialogue)))
    catalog_index = span_catalog_index(catalog)
    if set(valid_turn_ids) != {span["turn_id"] for span in catalog_index.values()}:
        raise ValueError("span catalog does not cover the frozen turn whitelist")
    if not {span["session_id"] for span in catalog_index.values()} <= set(valid_session_ids):
        raise ValueError("span catalog contains a non-whitelisted session")
    contracts = list(reasoning_contracts or default_reasoning_contracts(letters))
    validate_reasoning_contracts(contracts)
    history = [{k: item.get(k) for k in (
        "question", "canonical_answer", "reasoning_type", "evidence_session_pattern",
        "verified_reward_mean", "round")}
        for item in recent_frontier_history[-10:]]
    payload = {
        "batch_index": batch_index,
        "span_catalog_version": SPAN_CATALOG_VERSION,
        "questioner_schema_version": QUESTIONER_SCHEMA_VERSION,
        "reasoning_contract_version": REASONING_CONTRACT_VERSION,
        "span_catalog": model_span_catalog_view(catalog),
        "reasoning_contracts_by_array_position": model_reasoning_contract_view(contracts),
        "recent_accepted_frontier_history_for_novelty_only": history,
    }
    rules = [
        "Return an array of exactly three objects. Fewer or more invalidates the entire batch; never add a fourth item.",
        "The span catalog is the sole factual source. Do not copy or reproduce dialogue sentences; select only supplied span_id values.",
        "Never invent a span ID. Never output exact_quote, turn_id, session_id, answer_supporting_quotes, evidence_turn_ids, or evidence_session_ids; the caller restores those fields from the catalog.",
        "Generate correct_answer_text and exactly three unlabeled distractors. Never output options, A/B/C/D labels, target letter, reference_answer_letter, reference_answer_text, or canonical_answer_text; the caller binds those fields locally.",
        "Never output reasoning_type or difficulty_level. Each array position has a different immutable reasoning contract; the caller derives reasoning type and difficulty locally from validated spans, sessions, values, and steps.",
        "correct_answer_text and all three distractors must be non-empty, distinct, and the same answer type.",
        "Each distractor must embody a different error path and distractor_explanations must contain exactly three explanations in matching array order.",
        "answer_support_span_ids must meet the assigned position's distinct span/turn/session minimums. Reusing one span as multiple information points is forbidden.",
        "Every answer_support_span_id must have a matching answer_evidence_values entry. Each answer evidence value must be the shortest meaningful whole phrase present in both the selected span text and correct_answer_text.",
        "For each distractor_index 0, 1, and 2, provide at least one real supporting span ID and at least one minimal evidence value present in both that span and the matching distractor text.",
        "Evidence values must be distinct meaningful content phrases such as education, aunt, or infrastructure, never generic words such as the, is, during session, the person, or this event.",
        "Distractors must be plausible candidate values grounded in the dialogue, not irrelevant, absurd, or removable using common knowledge alone.",
        "History is only a novelty signal. Never use a historical answer or fact unless independently present in the span catalog.",
        "The three questions must differ in subject, evidence combination, and reasoning path; paraphrases, option swaps, and answer-position variants are forbidden.",
        "Follow each position's concrete reasoning contract exactly. Do not substitute a simpler one-turn fact and do not reuse a reasoning path assigned to another position.",
        "Use only directly supported facts. No outside knowledge, assumptions, subjective judgment, placeholders, answer-length hints, or formatting hints.",
        "Question text must not reveal span IDs, evidence, explanations, or mention the dialogue as a document.",
        "Every key in the schema is required. Output strict JSON only with no Markdown or additional text.",
    ]
    minimal = [{**QUESTIONER_ITEM_SCHEMA, "qa_id": f"batch{batch_index}_item{i+1}",
        "correct_answer_text":"evidence-supported answer",
        "distractors":["same-type distractor 1","same-type distractor 2","same-type distractor 3"],
        "distractor_explanations":["error path 1","error path 2","error path 3"],
        "answer_support_span_ids":["supplied span ID 1", "supplied span ID 2"],
        "answer_evidence_values":[
            {"span_id":"supplied span ID 1", "value":"meaningful answer value 1"},
            {"span_id":"supplied span ID 2", "value":"meaningful answer value 2"}],
        "distractor_support":[
            {"distractor_index":0,"span_ids":["supplied span ID"],"evidence_values":["candidate 1"]},
            {"distractor_index":1,"span_ids":["supplied span ID"],"evidence_values":["candidate 2"]},
            {"distractor_index":2,"span_ids":["supplied span ID"],"evidence_values":["candidate 3"]}],
        "reasoning_steps":["Use the first selected information point.",
                           "Combine it with the second selected information point."]} for i in range(3)]
    return "\n\n".join([
        QUESTIONER_SYSTEM_INSTRUCTION,
        f"PROMPT_VERSION: {QUESTIONER_PROMPT_VERSION}",
        "BATCH AND QUALITY RULES\n- " + "\n- ".join(rules),
        "DYNAMIC INPUT\n" + _json(payload),
        "REQUIRED ITEM SCHEMA\n" + _json(QUESTIONER_ITEM_SCHEMA),
        "MINIMAL THREE-ITEM SHAPE (illustrative values only; do not copy facts)\n" + _json(minimal),
    ])


def prompt_template_hashes() -> dict[str, str]:
    """Stable hashes of instructions/schemas; dynamic input is hashed in each request ledger."""
    materials = {
        "realizer": (REALIZER_PROMPT_VERSION + REALIZER_SYSTEM_INSTRUCTION +
                     _json(REALIZER_OUTPUT_SCHEMA) + inspect.getsource(build_realizer_prompt)),
        "questioner": (QUESTIONER_PROMPT_VERSION + QUESTIONER_SYSTEM_INSTRUCTION +
                       _json(QUESTIONER_ITEM_SCHEMA) + inspect.getsource(build_questioner_prompt)),
        "responder": RESPONDER_PROMPT_VERSION + RESPONDER_PROMPT,
    }
    return {key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in materials.items()}


# Backward-compatible names for manifest code; these now identify full versioned templates,
# not the old placeholder prompts. Callers must use the builders for actual requests.
REALIZER_PROMPT = REALIZER_SYSTEM_INSTRUCTION
QUESTIONER_PROMPT = QUESTIONER_SYSTEM_INSTRUCTION
