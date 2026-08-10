"""QA adapter for canonical parent + promoted-child dialogue artifacts.

The transcript remains the sole evidence source.  World Memory is deliberately
absent from every interface in this module.
"""

from __future__ import annotations

import inspect
import json
import re
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes
from ..questioner_contract import build_span_catalog, span_catalog_index
from ..single_item_qa_canary import validate_single_item_core_object


CONTINUOUS_EXPANSION_QA_VERSION = (
    "slim_continuous_expansion_core_qa_v12_soft_evidence_scope"
)
CONTINUOUS_EXPANSION_QA_PROMPT_VERSION = (
    "slim_continuous_expansion_questioner_v15_soft_child_coverage"
)
LINEAGE_PAIR_REGISTRY_VERSION = (
    "cross_boundary_lineage_pair_registry_v2_exact_same_speaker_pairs"
)
MAX_COMPLETION_TOKENS = 1024
MODEL_CONTEXT_LIMIT = 40960
SOURCE_PARTITIONS = frozenset({"original_parent", "promoted_child"})
MAX_ELIGIBLE_PAIRS_PER_LINEAGE_ROW = 4
QUESTION_SIDE_DETAIL_LIMIT = 2
_QUESTION_DETAIL_STOPWORDS = frozenset({
    "about", "after", "again", "before", "between", "could", "earlier",
    "experience", "first", "from", "their", "later", "moment", "other",
    "speaker", "they", "this", "through", "what", "when", "where", "which",
    "while", "with", "would",
})


def _phrase_tokens(value: str) -> list[str]:
    return re.findall(r"[\w]+", str(value or "").casefold())


def _phrase_occurs(phrase: str, text: str) -> bool:
    needle = _phrase_tokens(phrase)
    haystack = _phrase_tokens(text)
    width = len(needle)
    return bool(needle) and any(
        haystack[index:index + width] == needle
        for index in range(len(haystack) - width + 1))


def _lineage_pair_relevance(parent: Mapping[str, Any],
                            child: Mapping[str, Any]) -> tuple[int, float]:
    parent_tokens = {token for token in _phrase_tokens(
        str(parent.get("exact_text") or "")) if len(token) >= 4}
    child_tokens = {token for token in _phrase_tokens(
        str(child.get("exact_text") or "")) if len(token) >= 4}
    overlap = parent_tokens & child_tokens
    union = parent_tokens | child_tokens
    return len(overlap), (len(overlap) / len(union) if union else 0.0)


def diagnose_question_support_detail_leakage(
    question: str, *, parent_span: Mapping[str, Any],
    child_span: Mapping[str, Any],
) -> dict[str, Any]:
    """Detect a question that copies enough one-sided detail to reveal its answer."""
    question_tokens = set(_phrase_tokens(question))
    parent_tokens = {
        token for token in _phrase_tokens(str(parent_span.get("exact_text") or ""))
        if len(token) >= 4 and token not in _QUESTION_DETAIL_STOPWORDS
    }
    child_tokens = {
        token for token in _phrase_tokens(str(child_span.get("exact_text") or ""))
        if len(token) >= 4 and token not in _QUESTION_DETAIL_STOPWORDS
    }
    parent_matches = sorted(question_tokens & (parent_tokens - child_tokens))
    child_matches = sorted(question_tokens & (child_tokens - parent_tokens))
    leaking_partitions = [
        partition for partition, matches in (
            ("original_parent", parent_matches),
            ("promoted_child", child_matches),
        ) if len(matches) >= QUESTION_SIDE_DETAIL_LIMIT
    ]
    endpoint_pattern = bool(re.search(
        r"\b(?:evolve\w*|chang\w*|shift\w*|mov\w*)\b.{0,120}"
        r"\bfrom\b.{1,120}\bto\b", question, flags=re.IGNORECASE))
    return {
        "diagnostic_version": "qa_question_support_detail_leakage_v1",
        "detail_limit_per_partition": QUESTION_SIDE_DETAIL_LIMIT,
        "parent_only_question_tokens": parent_matches,
        "child_only_question_tokens": child_matches,
        "leaking_partitions": leaking_partitions,
        "states_both_transition_endpoints": endpoint_pattern,
        "valid": not leaking_partitions and not endpoint_pattern,
    }


def diagnose_question_speaker_grounding(
    question: str, *, catalog: Mapping[str, Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Find dialogue speaker labels named by the question but absent from evidence."""
    speakers = sorted({str(span.get("speaker") or "") for span in catalog.values()
                       if str(span.get("speaker") or "").strip()})
    named = [speaker for speaker in speakers if _phrase_occurs(speaker, question)]
    ungrounded = [
        speaker for speaker in named
        if not any(
            _phrase_tokens(str(span.get("speaker") or "")) ==
            _phrase_tokens(speaker) or
            _phrase_occurs(speaker, str(span.get("exact_text") or ""))
            for span in selected)
    ]
    return {
        "named_dialogue_speakers": named,
        "ungrounded_named_speakers": ungrounded,
        "valid": not ungrounded,
    }


def diagnose_question_relation_form(question: str) -> dict[str, Any]:
    """Reject causal answer types that two state observations cannot establish."""
    text = " ".join(str(question or "").casefold().split())
    causal_patterns = (
        r"\bwhy\b",
        r"\bwhat\s+(?:prompted|caused|motivated|influenced|led)\b",
        r"\bhow\s+did\b.{0,100}\b(?:cause|prompt|motivate|influence|lead)\b",
        r"\bin\s+response\s+to\b",
        r"\bas\s+a\s+result\s+of\b",
    )
    matched = [pattern for pattern in causal_patterns
               if re.search(pattern, text, flags=re.IGNORECASE)]
    return {
        "diagnostic_version": "cross_boundary_question_relation_form_v1",
        "relation_mode": "unsupported_causal" if matched else
            "descriptive_comparison",
        "matched_patterns": matched,
        "valid": not matched,
    }


def classify_evidence_scope(
    support_span_ids: Sequence[str], *, span_catalog: Mapping[str, Any],
    partition_by_turn: Mapping[str, str],
) -> dict[str, Any]:
    """Describe evidence coverage without making scope an admission gate."""
    catalog = span_catalog_index(dict(span_catalog))
    partitions = [partition_by_turn.get(catalog[span_id]["turn_id"])
                  for span_id in support_span_ids]
    parent_count = partitions.count("original_parent")
    child_count = partitions.count("promoted_child")
    child_sessions = list(dict.fromkeys(
        catalog[span_id]["session_id"] for span_id, partition in
        zip(support_span_ids, partitions) if partition == "promoted_child"))
    if parent_count and child_count:
        scope = "parent_child"
    elif child_count and len(child_sessions) >= 2:
        scope = "multi_child"
    elif child_count:
        scope = "child_only"
    else:
        scope = "parent_only"
    return {
        "evidence_scope": scope,
        "parent_support_count": parent_count,
        "child_support_count": child_count,
        "child_involved": child_count > 0,
        "child_session_count": len(child_sessions),
        "scope_is_quality_gate": False,
    }


def adapt_expanded_dialogue(
    expanded: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Adapt stable expanded-dialogue refs to the canonical QA turn shape."""
    if expanded.get("artifact_role") != "canonical_expanded_dialogue":
        raise ValueError("qa_source_artifact_role_invalid")
    if not expanded.get("qa_source_candidate"):
        raise ValueError("qa_source_not_ready")
    turns = expanded.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("qa_source_turns_invalid")
    adapted: list[dict[str, str]] = []
    partition_by_turn: dict[str, str] = {}
    for row in turns:
        if not isinstance(row, dict):
            raise ValueError("qa_source_turn_invalid")
        turn_id = str(row.get("turn_ref") or "").strip()
        session_id = str(row.get("session_ref") or "").strip()
        speaker = str(row.get("speaker_label") or "").strip()
        text = str(row.get("text") or "")
        partition = str(row.get("source_partition") or "").strip()
        if (not turn_id or turn_id in partition_by_turn or not session_id or
                not speaker or not text or partition not in SOURCE_PARTITIONS):
            raise ValueError("qa_source_turn_invalid")
        adapted.append({"turn_id": turn_id, "session_id": session_id,
                        "speaker": speaker, "text": text})
        partition_by_turn[turn_id] = partition
    declared_refs = expanded.get("canonical_turn_refs")
    if (not isinstance(declared_refs, list) or
            declared_refs != [turn["turn_id"] for turn in adapted]):
        raise ValueError("qa_source_canonical_order_mismatch")
    return adapted, partition_by_turn


def build_partitioned_span_rows(
    span_catalog: Mapping[str, Any],
    *,
    partition_by_turn: Mapping[str, str],
) -> str:
    catalog = span_catalog_index(dict(span_catalog))
    rows: list[str] = []
    for span in catalog.values():
        partition = partition_by_turn.get(span["turn_id"])
        if partition not in SOURCE_PARTITIONS:
            raise ValueError("qa_span_partition_missing")
        rows.append(json.dumps({
            "span_id": span["span_id"],
            "turn_id": span["turn_id"],
            "session_id": span["session_id"],
            "source_partition": partition,
            "speaker": span["speaker"],
            "text": span["exact_text"],
        }, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(rows)


def build_cross_boundary_questioner_prompt(
    *,
    root_id: str,
    span_catalog: Mapping[str, Any],
    partition_by_turn: Mapping[str, str],
    lineage_pair_registry: Mapping[str, Any] | None = None,
    frozen_support_pair: Mapping[str, Any] | None = None,
    excluded_questions: Sequence[str] = (),
    excluded_support_pairs: Sequence[Sequence[str]] = (),
) -> str:
    root = str(root_id or "").strip()
    if not root:
        raise ValueError("qa_root_id_invalid")
    dialogue = build_partitioned_span_rows(
        span_catalog, partition_by_turn=partition_by_turn)
    catalog = span_catalog_index(dict(span_catalog))
    parent_sessions = list(dict.fromkeys(
        span["session_id"] for span in catalog.values()
        if partition_by_turn.get(span["turn_id"]) == "original_parent"))
    if not parent_sessions:
        raise ValueError("qa_parent_partition_empty")
    legacy_pair_mode = (lineage_pair_registry is not None or
                        frozen_support_pair is not None)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": (["question"] if frozen_support_pair is not None else
                     (["question", "answer_support_span_ids"]
                      if legacy_pair_mode else
                      ["question", "correct_answer_text",
                       "answer_support_span_ids"])),
        "properties": {
            "question": {"type": "string", "minLength": 1},
            **({} if legacy_pair_mode else {
              "correct_answer_text": {"type": "string", "minLength": 1},
            }),
            **({} if frozen_support_pair is not None else {
              "answer_support_span_ids": {
                "type": "array", "minItems": 2,
                "maxItems": 2 if legacy_pair_mode else 6,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
              }}),
        },
    }
    rules = [
        "Generate exactly one natural, answerable question, its concise correct answer, and its minimal sufficient evidence.",
        "Use the complete dialogue. Prefer meaningful, under-covered promoted_child developments, but a strong parent_only, child_only, parent_child, or multi_child question is valid. Scope is a batch coverage preference, not a per-question correctness rule.",
        "Select 2 to 6 exact evidence spans from whichever partition or partitions actually support the question. Select antecedent or supporting turns when needed; the program will not choose evidence for you.",
        "Use only the frozen dialogue. Do not use World Memory, outside knowledge, or unstated inference.",
        "Ask one focused question supported by the selected evidence. A comparison may ask how a position, decision, commitment, or handling developed; a single-partition question may ask about one concrete development. Do not force an unrelated cross-stage comparison.",
        "Evidence speakers may differ. Attribute every statement, action, and comparison endpoint to the correct participant shown in the selected evidence.",
        "Do not put the decisive answer values from both selected turns into the question. Refer to the shared situation at an abstract but unambiguous level so the question alone cannot select the gold from plausible alternatives.",
        "Use an evaluatively neutral comparison stem. Do not name a qualitative dimension such as empathy, emotional support, responsibility, maturity, professional impact, withdrawal, or helpfulness when that label describes the correct trajectory more strongly than plausible alternatives. Ask how the speaker's approach differed across the two moments and leave the evaluative characterization to the answer choices.",
        "Do not echo a parent/child category pair from the correct answer in the stem, such as personal-versus-professional focus. A no-context reader must not be able to choose an option merely by matching the question's abstract endpoint labels.",
        "Keep the question abstract: from either selected turn, do not copy two or more distinctive one-sided content words. Ask about the change or consequence; leave the concrete initial and later details for the answer.",
        "Do not use an 'evolve/change from X to Y' stem that states both endpoints. Ask how the approach differed or developed without supplying either concrete endpoint.",
        "Do not ask why, what prompted/caused/motivated/influenced the change, or what happened in response to something unless the submitted evidence explicitly establishes that causal relation.",
        "Avoid templates such as 'what did X initially say about VALUE_A, and what did X later decide about VALUE_B' when VALUE_A and VALUE_B are themselves the answer-bearing details.",
        "When evidence crosses partitions or child sessions, all combined clauses must concern the same concrete event, problem, decision, relationship, or state development; do not concatenate unrelated facts merely to broaden scope.",
        "If a selected child turn uses pronouns such as this, that, it, or everything, also select an antecedent turn that makes the referenced event explicit.",
        "Choose self-contained turns that do not need an unselected antecedent to understand the question or answer.",
        "Write correct_answer_text yourself. It must directly answer the question and be fully supported by the complete dialogue and the submitted evidence; the program will not synthesize or overwrite it.",
        "Preserve modality exactly: planned, intended, offered, or was going to must not be rewritten as already completed.",
        "Attribute every statement or action to the exact speaker shown on its selected evidence row; never attribute another speaker's turn to the question subject or their co-worker.",
        "Every factual clause implied by the question must be explicitly supported by the submitted evidence spans.",
        "Every dialogue speaker label explicitly named in the question must be grounded by the submitted evidence: that speaker must speak in a submitted span, or a submitted span must explicitly name that speaker.",
        "Do not output choices, distractors, explanations, reasoning steps, QA IDs, or any additional fields.",
        "Return one strict JSON object only, with no Markdown or surrounding text.",
    ]
    if legacy_pair_mode:
        # Historical replay helpers retain their frozen-pair contract.  The live
        # v11 path calls this function without either legacy argument.
        rules[0] = "Generate exactly one natural, answerable question and select its evidence."
        rules[2] = "Select exactly two evidence spans: one original_parent span and one promoted_child span from two sessions."
        rules[4] = "Ask a focused cross-stage comparison question about one existing speaker. Ask how the speaker's position, decision, commitment, or handling differed or developed across the two moments. Do not ask a bare two-quote lookup question."
        rules[5] = "The two selected evidence spans must have the same exact speaker label. The question must attribute both moments to that speaker."
        rules[11] = "Do not ask why, what prompted/caused/motivated/influenced the change, or what happened in response to something. This two-span contract supports a descriptive comparison, not a causal explanation."
        rules[13] = "The parent and child clauses must concern the same concrete event, problem, decision, relationship, or state development; do not concatenate unrelated facts merely because the speaker is the same."
        rules[16] = "The program will materialize canonical gold directly from the two selected frozen spans. Do not output correct_answer_text."
        rules[19] = "Every factual clause implied by the question must be explicitly supported by the two selected turns."
        rules[20] = "Every dialogue speaker label explicitly named in the question must be grounded by the selected evidence: that speaker must speak one selected turn, or a selected turn explicitly names another dialogue speaker. With exactly two evidence spans, do not name an unsupported third speaker."
    pair_section = ""
    if lineage_pair_registry is not None:
        rows = lineage_pair_registry.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("qa_lineage_pair_registry_empty")
        rules.insert(3,
            "Select one exact parent_span_id/child_span_id pair from an "
            "ALLOWED LINEAGE PAIRS row. Never form a new pair by mixing entries. "
            "This finite same-speaker menu is a hard boundary, not a suggestion.")
        pair_rows = [{
            "scene_ref": row.get("scene_ref"),
            "source_anchor_id": row.get("source_anchor_id"),
            "same_speaker_pairs": row.get("eligible_same_speaker_pairs") or [],
        } for row in rows]
        pair_section = "ALLOWED LINEAGE PAIRS\n" + json.dumps(
            pair_rows, ensure_ascii=False, separators=(",", ":"))
    frozen_pair_section = ""
    if frozen_support_pair is not None:
        parent_span_id = str(frozen_support_pair.get("parent_span_id") or "")
        child_span_id = str(frozen_support_pair.get("child_span_id") or "")
        speaker = str(frozen_support_pair.get("speaker") or "")
        if not parent_span_id or not child_span_id or not speaker:
            raise ValueError("qa_frozen_support_pair_invalid")
        rules.insert(1,
            "Use exactly the FROZEN SUPPORT PAIR in the declared parent-then-child "
            "order. The program selected and will bind this anchor-linked pair. "
            "Your only task is to write the question; do not return either ID.")
        rules.insert(2,
            f"The only dialogue speaker label you may name is {speaker!r}, the "
            "speaker of both frozen spans. Do not name an interlocutor, helper, "
            "co-worker, friend, or other dialogue speaker, even in a phrase such "
            "as 'in discussions with X'.")
        rules.insert(3,
            "Describe the shared topic at one abstract level. Do not name the "
            "specific parent-side object/event and the specific child-side "
            "project/action in the question; those endpoints belong in the answer.")
        frozen_pair_section = "FROZEN SUPPORT PAIR\n" + json.dumps({
            "parent_span_id": parent_span_id,
            "child_span_id": child_span_id,
            "speaker": speaker,
            "scene_ref": frozen_support_pair.get("scene_ref"),
            "source_anchor_id": frozen_support_pair.get("source_anchor_id"),
        }, ensure_ascii=False, separators=(",", ":"))
    exclusions = [str(value).strip() for value in excluded_questions
                  if str(value).strip()]
    exclusion_section = ""
    if exclusions:
        rules.insert(1,
            "Generate an independent candidate. Do not reproduce or lightly "
            "paraphrase any EXCLUDED PRIOR QUESTION.")
        exclusion_section = "EXCLUDED PRIOR QUESTIONS\n" + json.dumps(
            exclusions, ensure_ascii=False, separators=(",", ":"))
    support_set_exclusions = [
        sorted(str(span_id).strip() for span_id in support_set
               if str(span_id).strip())
        for support_set in excluded_support_pairs
    ]
    support_set_exclusions = [
        support_set for support_set in support_set_exclusions
        if 2 <= len(support_set) <= 6 and
        len(set(support_set)) == len(support_set)
    ]
    support_pair_exclusion_section = ""
    if support_set_exclusions:
        if legacy_pair_mode:
            rules.insert(2,
                "Do not select either ordering of any EXCLUDED PRIOR SUPPORT "
                "PAIR. Choose a genuinely different parent/child evidence pair.")
            exclusion_heading = "EXCLUDED PRIOR SUPPORT PAIRS"
        else:
            rules.insert(2,
                "Do not reuse any EXCLUDED PRIOR SUPPORT SET in another "
                "ordering. Choose a genuinely different minimal evidence set.")
            exclusion_heading = "EXCLUDED PRIOR SUPPORT SETS"
        support_pair_exclusion_section = exclusion_heading + "\n" + json.dumps(
            support_set_exclusions, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join([
        f"PROMPT_VERSION: {CONTINUOUS_EXPANSION_QA_PROMPT_VERSION}",
        f"ROOT_ID: {root}",
        "CROSS-BOUNDARY CORE QA RULES\n- " + "\n- ".join(rules),
        frozen_pair_section,
        pair_section,
        exclusion_section,
        support_pair_exclusion_section,
        "FULL CANONICAL EXPANDED DIALOGUE\n" + dialogue,
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ])


def validate_cross_boundary_core_object(
    raw: Any,
    *,
    span_catalog: Mapping[str, Any],
    partition_by_turn: Mapping[str, str],
    lineage_pair_registry: Mapping[str, Any] | None = None,
    frozen_support_pair: Mapping[str, Any] | None = None,
    evidence_mismatch_policy: str = "strict",
    excluded_questions: Sequence[str] = (),
    excluded_support_pairs: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    if evidence_mismatch_policy not in {"strict", "mark_only"}:
        raise ValueError("evidence_mismatch_policy_invalid")
    # The current live contract gives the Questioner semantic ownership of both
    # its answer and its 2-6 evidence spans.  Legacy registry/frozen-pair calls
    # below remain only for read-only historical fixtures.
    if lineage_pair_registry is None and frozen_support_pair is None:
        if not isinstance(raw, Mapping) or set(raw) != {
                "question", "correct_answer_text", "answer_support_span_ids"}:
            return {
                "decision": "cross_boundary_qa_schema_invalid",
                "reasons": ["cross_boundary_qa_model_top_level_schema"],
                "core_qa": None,
            }
        base = validate_single_item_core_object(raw, span_catalog=span_catalog)
        if base.get("decision") != "single_item_questioner_accept":
            return {**base, "decision": "cross_boundary_qa_schema_invalid"}
        core = base["core_qa"]
        question_tokens = re.findall(
            r"[\w']+", str(core.get("question") or "").casefold())
        for excluded in excluded_questions:
            excluded_tokens = re.findall(
                r"[\w']+", str(excluded or "").casefold())
            if (excluded_tokens and SequenceMatcher(
                    a=question_tokens, b=excluded_tokens,
                    autojunk=False).ratio() >= 0.90):
                return {
                    **base,
                    "decision": "cross_boundary_qa_contract_reject",
                    "reasons": ["cross_boundary_qa_repeats_excluded_question"],
                    "core_qa": None,
                    "rejected_core_qa": core,
                }
        catalog = span_catalog_index(dict(span_catalog))
        support = list(core["answer_support_span_ids"])
        partitions = [partition_by_turn.get(catalog[span_id]["turn_id"])
                      for span_id in support]
        sessions = list(dict.fromkeys(
            catalog[span_id]["session_id"] for span_id in support))
        reasons: list[str] = []
        if not 2 <= len(support) <= 6:
            reasons.append("cross_boundary_qa_requires_2_to_6_support_spans")
        support_key = tuple(sorted(support))
        excluded_keys = {
            tuple(sorted(str(span_id).strip() for span_id in prior))
            for prior in excluded_support_pairs
            if 2 <= len(prior) <= 6 and
            all(str(span_id).strip() for span_id in prior)
        }
        if support_key in excluded_keys:
            reasons.append("cross_boundary_qa_repeats_excluded_support_set")
        selected_spans = [catalog[span_id] for span_id in support]
        scope = classify_evidence_scope(
            support, span_catalog=span_catalog,
            partition_by_turn=partition_by_turn)
        # These are audit-only observations.  They are deliberately not local
        # semantic rejection gates.
        speaker_grounding = diagnose_question_speaker_grounding(
            str(core.get("question") or ""), catalog=catalog,
            selected=selected_spans)
        relation_form = diagnose_question_relation_form(
            str(core.get("question") or ""))
        if reasons:
            return {
                **base,
                "decision": "cross_boundary_qa_contract_reject",
                "reasons": reasons,
                "core_qa": None,
                "rejected_core_qa": core,
                "support_partitions": partitions,
                "support_sessions": sessions,
                "question_speaker_grounding_diagnostic": speaker_grounding,
                "question_relation_form_diagnostic": relation_form,
                **scope,
            }
        return {
            **base,
            "decision": "cross_boundary_qa_accept",
            "support_partitions": partitions,
            "support_sessions": sessions,
            "question_speaker_grounding_diagnostic": speaker_grounding,
            "question_relation_form_diagnostic": relation_form,
            **scope,
            "reference_integrity_verified": True,
            "semantic_entailment_verified": False,
            "questioner_selected_evidence": True,
            "gold_materialized_locally": False,
            "model_correct_answer_text": core["correct_answer_text"],
            "evidence_mismatch_policy": evidence_mismatch_policy,
            "evidence_mismatch": False,
            "evidence_alignment_status": "awaiting_semantic_audit",
            "evidence_mismatch_reasons": [],
            "parent_support_count": scope["parent_support_count"],
            "child_support_count": scope["child_support_count"],
        }
    if (isinstance(raw, Mapping) and set(raw) == {"question"} and
            frozen_support_pair is not None):
        raw = {
            "question": raw.get("question"),
            "answer_support_span_ids": [
                str(frozen_support_pair.get("parent_span_id") or ""),
                str(frozen_support_pair.get("child_span_id") or ""),
            ],
        }
    if not isinstance(raw, Mapping) or set(raw) not in ({
            "question", "answer_support_span_ids"}, {
            "question", "correct_answer_text", "answer_support_span_ids"}):
        return {"decision": "cross_boundary_qa_schema_invalid",
                "reasons": ["cross_boundary_qa_model_top_level_schema"],
                "core_qa": None}
    model_answer = raw.get("correct_answer_text")
    catalog = span_catalog_index(dict(span_catalog))
    support_value = raw.get("answer_support_span_ids")
    expected_answer = None
    if (isinstance(support_value, list) and len(support_value) == 2 and
            all(isinstance(span_id, str) and span_id in catalog
                for span_id in support_value)):
        selected = [catalog[span_id] for span_id in support_value]
        selected_by_partition = {
            partition_by_turn.get(span["turn_id"]): span for span in selected}
        parent_span = selected_by_partition.get("original_parent")
        child_span = selected_by_partition.get("promoted_child")
        if parent_span is not None and child_span is not None:
            expected_answer = (
                f"Initially: {parent_span['exact_text'].strip()} "
                f"Later: {child_span['exact_text'].strip()}")
    normalized_raw = {
        "question": raw.get("question"),
        "correct_answer_text": expected_answer or "<INVALID_EVIDENCE_SELECTION>",
        "answer_support_span_ids": support_value,
    }
    base = validate_single_item_core_object(
        normalized_raw, span_catalog=span_catalog)
    if base.get("decision") != "single_item_questioner_accept":
        return {**base, "decision": "cross_boundary_qa_schema_invalid"}
    core = base["core_qa"]
    question_tokens = re.findall(
        r"[\w']+", str(core.get("question") or "").casefold())
    for excluded in excluded_questions:
        excluded_tokens = re.findall(
            r"[\w']+", str(excluded or "").casefold())
        if (excluded_tokens and SequenceMatcher(
                a=question_tokens, b=excluded_tokens,
                autojunk=False).ratio() >= 0.90):
            return {
                **base,
                "decision": "cross_boundary_qa_contract_reject",
                "reasons": ["cross_boundary_qa_repeats_excluded_question"],
                "core_qa": None,
                "rejected_core_qa": core,
            }
    support = list(core["answer_support_span_ids"])
    support_key = tuple(sorted(support))
    excluded_pair_keys = {
        tuple(sorted(str(span_id).strip() for span_id in pair))
        for pair in excluded_support_pairs
        if (len(pair) == 2 and all(str(span_id).strip() for span_id in pair))
    }
    partitions = [partition_by_turn.get(catalog[span_id]["turn_id"])
                  for span_id in support]
    sessions = list(dict.fromkeys(catalog[span_id]["session_id"]
                                  for span_id in support))
    reasons: list[str] = []
    evidence_mismatch_reasons: list[str] = []
    if len(support) != 2:
        reasons.append("cross_boundary_qa_requires_exactly_two_support_spans")
    if frozen_support_pair is not None:
        frozen_ids = {
            str(frozen_support_pair.get("parent_span_id") or ""),
            str(frozen_support_pair.get("child_span_id") or ""),
        }
        # Core QA canonicalizes evidence by narrative order. Inserted or parallel
        # child scenes may therefore sort before their source Parent turn even
        # though the frozen contract names parent/child by source partition.
        if set(support) != frozen_ids:
            reasons.append("cross_boundary_qa_frozen_support_pair_mismatch")
    for required in sorted(SOURCE_PARTITIONS):
        if required not in partitions:
            reasons.append(f"cross_boundary_qa_missing_partition:{required}")
    if len(sessions) < 2:
        reasons.append("cross_boundary_qa_requires_two_sessions")
    selected_spans = [catalog[span_id] for span_id in support]
    selected_speakers = {
        str(span.get("speaker") or "").strip() for span in selected_spans}
    if len(selected_speakers) != 1:
        reasons.append("cross_boundary_qa_requires_same_speaker")
    if support_key in excluded_pair_keys:
        reasons.append("cross_boundary_qa_repeats_excluded_support_pair")
    speaker_grounding = diagnose_question_speaker_grounding(
        str(core.get("question") or ""), catalog=catalog,
        selected=selected_spans)
    reasons.extend(
        "cross_boundary_qa_question_speaker_not_grounded:" + speaker
        for speaker in speaker_grounding["ungrounded_named_speakers"])
    relation_form = diagnose_question_relation_form(
        str(core.get("question") or ""))
    if not relation_form["valid"]:
        reasons.append("cross_boundary_qa_unsupported_causal_question_form")
    if lineage_pair_registry is not None and len(support) == 2:
        selected_by_partition = {
            partition_by_turn.get(catalog[span_id]["turn_id"]): span_id
            for span_id in support
        }
        parent_id = selected_by_partition.get("original_parent")
        child_id = selected_by_partition.get("promoted_child")
        registry_rows = [row for row in lineage_pair_registry.get("rows") or []
                         if isinstance(row, Mapping)]
        exact_pairs_declared = any(
            isinstance(row.get("eligible_same_speaker_pairs"), list)
            for row in registry_rows)
        if exact_pairs_declared:
            allowed = any(
                parent_id == pair.get("parent_span_id") and
                child_id == pair.get("child_span_id")
                for row in registry_rows
                for pair in row.get("eligible_same_speaker_pairs") or []
                if isinstance(pair, Mapping))
        else:
            allowed = any(
                parent_id in set(row.get("parent_span_ids") or []) and
                child_id in set(row.get("child_span_ids") or [])
                for row in registry_rows)
        if not allowed:
            mismatch = "cross_boundary_qa_pair_not_anchor_linked"
            if evidence_mismatch_policy == "mark_only":
                evidence_mismatch_reasons.append(mismatch)
            else:
                reasons.append(mismatch)
    parent_spans = [catalog[span_id] for span_id, partition in
                    zip(support, partitions) if partition == "original_parent"]
    child_spans = [catalog[span_id] for span_id, partition in
                   zip(support, partitions) if partition == "promoted_child"]
    if len(parent_spans) == 1 and len(child_spans) == 1:
        expected_answer = (
            f"Initially: {parent_spans[0]['exact_text'].strip()} "
            f"Later: {child_spans[0]['exact_text'].strip()}")
        question_detail_leakage = diagnose_question_support_detail_leakage(
            str(core.get("question") or ""), parent_span=parent_spans[0],
            child_span=child_spans[0])
        if not question_detail_leakage["valid"]:
            reasons.append(
                "cross_boundary_qa_question_repeats_support_details")
    else:
        question_detail_leakage = None
    if reasons:
        return {**base, "decision": "cross_boundary_qa_contract_reject",
                "reasons": reasons, "core_qa": None,
                "rejected_core_qa": core,
                "support_partitions": partitions,
                "support_sessions": sessions,
                "question_speaker_grounding": speaker_grounding,
                "question_relation_form": relation_form,
                "question_support_detail_leakage": question_detail_leakage,
                "expected_extractive_answer": expected_answer}
    return {**base, "decision": "cross_boundary_qa_accept",
            "support_partitions": partitions,
            "support_sessions": sessions,
            "question_speaker_grounding": speaker_grounding,
            "question_relation_form": relation_form,
            "question_support_detail_leakage": question_detail_leakage,
            "extractive_answer_verified": True,
            "gold_materialized_locally": True,
            "model_correct_answer_text": model_answer,
            "model_answer_exact_match": model_answer == expected_answer,
            "evidence_mismatch_policy": evidence_mismatch_policy,
            "evidence_mismatch": bool(evidence_mismatch_reasons),
            "evidence_alignment_status": (
                "mismatch_marked" if evidence_mismatch_reasons else "aligned"),
            "evidence_mismatch_reasons": evidence_mismatch_reasons,
            "parent_support_count": partitions.count("original_parent"),
            "child_support_count": partitions.count("promoted_child")}


def selected_evidence_rows(
    core_qa: Mapping[str, Any],
    *,
    span_catalog: Mapping[str, Any],
    partition_by_turn: Mapping[str, str],
) -> list[dict[str, str]]:
    catalog = span_catalog_index(dict(span_catalog))
    rows: list[dict[str, str]] = []
    for span_id in core_qa.get("answer_support_span_ids") or []:
        span = catalog[str(span_id)]
        rows.append({**span,
                     "source_partition": partition_by_turn[span["turn_id"]]})
    return rows


def protocol_sha256() -> str:
    material = canonical_json({
        "version": CONTINUOUS_EXPANSION_QA_VERSION,
        "prompt_version": CONTINUOUS_EXPANSION_QA_PROMPT_VERSION,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "adapter": inspect.getsource(adapt_expanded_dialogue),
        "prompt": inspect.getsource(build_cross_boundary_questioner_prompt),
        "validator": inspect.getsource(validate_cross_boundary_core_object),
        "pair_registry": inspect.getsource(build_cross_boundary_lineage_pair_registry),
        "pair_selection": inspect.getsource(select_frozen_lineage_pair),
    })
    return sha256_bytes(material.encode("utf-8"))


def build_qa_source_view(expanded: Mapping[str, Any]) -> dict[str, Any]:
    dialogue, partition_by_turn = adapt_expanded_dialogue(expanded)
    span_catalog = build_span_catalog(dialogue)
    return {
        "combined_dialogue": dialogue,
        "span_catalog": span_catalog,
        "partition_by_turn": partition_by_turn,
        "view_sha256": sha256_bytes(canonical_json({
            "combined_dialogue": dialogue,
            "span_catalog": span_catalog,
            "partition_by_turn": partition_by_turn,
        }).encode("utf-8")),
    }


def build_cross_boundary_lineage_pair_registry(
    *,
    round_validator_input: Mapping[str, Any],
    span_catalog: Mapping[str, Any],
    partition_by_turn: Mapping[str, str],
) -> dict[str, Any]:
    """Compile finite Parent-anchor/child-scene evidence pair choices."""
    catalog = span_catalog_index(dict(span_catalog))
    span_ids_by_turn: dict[str, list[str]] = {}
    for span_id, span in catalog.items():
        span_ids_by_turn.setdefault(str(span["turn_id"]), []).append(span_id)
    anchor_rows = [row for row in
                   round_validator_input.get("historical_anchor_index") or []
                   if isinstance(row, Mapping)]
    specific_anchor_rows = [row for row in anchor_rows
                            if row.get("source_kind") == "historical_dialogue"]
    # Current rounds must use their finite historical anchors.  The fallback is
    # only for old artifacts that predate those anchors entirely.
    selected_anchor_rows = specific_anchor_rows or [
        row for row in anchor_rows if row.get("source_kind") == "planning_state"]
    anchors = {str(row.get("source_anchor_id")): row
               for row in selected_anchor_rows}
    rows: list[dict[str, Any]] = []
    for scene in round_validator_input.get("scenes") or []:
        if not isinstance(scene, Mapping):
            continue
        anchor_id = str(scene.get("source_anchor_id") or "")
        anchor = anchors.get(anchor_id)
        if anchor is None:
            continue
        parent_span_ids = [
            span_id
            for turn_ref in anchor.get("source_refs") or []
            for span_id in span_ids_by_turn.get(str(turn_ref), [])
            if partition_by_turn.get(catalog[span_id]["turn_id"]) == "original_parent"
        ]
        child_span_ids = [
            span_id
            for turn in scene.get("dialogue_turns") or []
            if isinstance(turn, Mapping)
            for span_id in span_ids_by_turn.get(str(turn.get("turn_ref") or ""), [])
            if partition_by_turn.get(catalog[span_id]["turn_id"]) == "promoted_child"
        ]
        if parent_span_ids and child_span_ids:
            pair_candidates = []
            for parent_index, parent_span_id in enumerate(parent_span_ids):
                for child_index, child_span_id in enumerate(child_span_ids):
                    if (str(catalog[parent_span_id]["speaker"]) !=
                            str(catalog[child_span_id]["speaker"])):
                        continue
                    overlap, ratio = _lineage_pair_relevance(
                        catalog[parent_span_id], catalog[child_span_id])
                    pair_candidates.append((
                        -overlap, -ratio, parent_index, child_index,
                        {
                            "parent_span_id": parent_span_id,
                            "child_span_id": child_span_id,
                            "speaker": str(catalog[parent_span_id]["speaker"]),
                        }))
            pair_candidates.sort(key=lambda row: row[:4])
            eligible_same_speaker_pairs = [
                row[4] for row in pair_candidates[
                    :MAX_ELIGIBLE_PAIRS_PER_LINEAGE_ROW]]
            if not eligible_same_speaker_pairs:
                continue
            rows.append({
                "scene_ref": str(scene.get("scene_ref") or ""),
                "source_anchor_id": anchor_id,
                "parent_span_ids": parent_span_ids,
                "child_span_ids": child_span_ids,
                "eligible_same_speaker_pairs": eligible_same_speaker_pairs,
            })
    material = {"lineage_pair_registry_version": LINEAGE_PAIR_REGISTRY_VERSION,
                "rows": rows}
    return {**material, "registry_sha256": sha256_bytes(
        canonical_json(material).encode("utf-8"))}


def select_frozen_lineage_pair(
    lineage_pair_registry: Mapping[str, Any], *, candidate_index: int,
    excluded_support_pairs: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Choose one stable anchor-linked pair while distributing candidates by scene."""
    if candidate_index < 1:
        raise ValueError("qa_candidate_index_invalid")
    rows = [row for row in lineage_pair_registry.get("rows") or []
            if isinstance(row, Mapping) and
            isinstance(row.get("eligible_same_speaker_pairs"), list) and
            row.get("eligible_same_speaker_pairs")]
    if not rows:
        raise ValueError("qa_lineage_pair_registry_empty")
    excluded = {
        tuple(sorted(str(value).strip() for value in pair))
        for pair in excluded_support_pairs
        if len(pair) == 2 and all(str(value).strip() for value in pair)
    }
    start = (candidate_index - 1) % len(rows)
    for row in rows[start:] + rows[:start]:
        for pair in row["eligible_same_speaker_pairs"]:
            if not isinstance(pair, Mapping):
                continue
            parent = str(pair.get("parent_span_id") or "")
            child = str(pair.get("child_span_id") or "")
            speaker = str(pair.get("speaker") or "")
            if (not parent or not child or not speaker or
                    tuple(sorted((parent, child))) in excluded):
                continue
            return {
                "parent_span_id": parent,
                "child_span_id": child,
                "speaker": speaker,
                "scene_ref": str(row.get("scene_ref") or ""),
                "source_anchor_id": str(row.get("source_anchor_id") or ""),
                "selection_version": "candidate_scene_rotation_v1",
                "candidate_index": candidate_index,
            }
    raise ValueError("qa_no_unconsumed_lineage_pair")
