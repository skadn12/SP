from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes
from .lineage_distractor_writer import DISTRACTOR_WRITER_JSON_SCHEMA
from .lineage_qa_writer import QA_WRITER_JSON_SCHEMA
from .lineage_relation_miner import RELATION_MINER_JSON_SCHEMA
from .questioner_contract import span_catalog_index


FULL_DIALOGUE_INPUT_VERSION = "lineage_full_dialogue_input_v1"
RELATION_MINER_FULL_DIALOGUE_PROMPT_VERSION = (
    "spell_lineage_relation_miner_single_full_dialogue_v1")
QA_WRITER_FULL_DIALOGUE_PROMPT_VERSION = (
    "spell_lineage_question_answer_writer_single_full_dialogue_v1")
DISTRACTOR_WRITER_FULL_DIALOGUE_PROMPT_VERSION = (
    "spell_lineage_distractor_writer_single_full_dialogue_v1")


def _dialogue_rows(full_dialogue: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{"turn_id":str(turn["turn_id"]),
             "session_id":str(turn["session_id"]),
             "speaker":str(turn["speaker"]),
             "text":str(turn["text"])} for turn in full_dialogue]


def _lineage_rows(lineage_view: Mapping[str, Any],
                  span_stage_membership: Mapping[str, Sequence[str]] | None = None
                  ) -> list[dict[str, Any]]:
    catalog = span_catalog_index(dict(lineage_view["span_catalog"]))
    parent_ids = set(lineage_view["selected_parent_span_ids"])
    membership = span_stage_membership or {}
    return [{"span_id":span_id,
             "source_partition":"parent" if span_id in parent_ids else "child",
             "event_stage_ids":list(membership.get(span_id, ())),
             "turn_id":str(span["turn_id"]),
             "session_id":str(span["session_id"]),
             "speaker":str(span["speaker"]),
             "exact_text":str(span["exact_text"])}
            for span_id, span in catalog.items()]


def build_full_dialogue_relation_miner_prompt(*, arc_id: str,
        candidate_index: int, lineage_view: Mapping[str, Any],
        full_dialogue: Sequence[Mapping[str, Any]],
        span_stage_membership: Mapping[str, Sequence[str]],
        prior_relation: Mapping[str, Any] | None) -> str:
    prior = None if prior_relation is None else {key:prior_relation[key] for key in (
        "relation_id", "parent_condition_span_ids", "child_consequence_span_ids",
        "relation_type", "relation_description")}
    payload = {"input_version":FULL_DIALOGUE_INPUT_VERSION,
        "arc_id":arc_id, "candidate_index":candidate_index,
        "required_relation_id":f"{arc_id}_R{candidate_index + 1}",
        "lineage_view_sha256":lineage_view["view_sha256"],
        "full_dialogue_turn_count":len(full_dialogue),
        "full_frozen_dialogue_in_original_order":_dialogue_rows(full_dialogue),
        "selectable_lineage_spans":_lineage_rows(
            lineage_view, span_stage_membership),
        "accepted_prior_relation_for_distinctness":prior}
    rules = [
        "Return exactly one JSON relation object matching the schema.",
        "Read the complete frozen dialogue in its original session and turn order.",
        "Select support IDs only from selectable_lineage_spans for the assigned arc.",
        "Select a parent initial condition, plan, assumption, state, commitment, or information and a child action, discovery, revision, state change, or outcome.",
        "The parent condition and child consequence must have a real semantic dependency, not merely share a topic.",
        "The complete relation must be underdetermined if either side is removed.",
        "Choose the smallest sufficient evidence span set and copy the required relation_id exactly.",
        "If a prior accepted relation is supplied, choose a different relation with a different canonical support set.",
        "Do not generate a question, answer, distractor, option, answer letter, quote, or extra field.",
        "Only the frozen dialogue text is factual evidence; stage and partition labels are navigation metadata, not additional facts.",
        "Return JSON only without Markdown or surrounding text."]
    return "\n\n".join([
        f"PROMPT_VERSION: {RELATION_MINER_FULL_DIALOGUE_PROMPT_VERSION}",
        "RELATION MINER RULES\n- " + "\n- ".join(rules),
        "COMPLETE FROZEN DIALOGUE INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            RELATION_MINER_JSON_SCHEMA, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"))])


def build_full_dialogue_qa_writer_prompt(*, lineage_view: Mapping[str, Any],
        full_dialogue: Sequence[Mapping[str, Any]],
        relation_bundle: Mapping[str, Any],
        relation_bundle_sha256: str) -> str:
    catalog = span_catalog_index(dict(lineage_view["span_catalog"]))
    evidence = [{"span_id":span_id,
                 "source_partition":"parent" if span_id in set(
                     relation_bundle["parent_condition_span_ids"]) else "child",
                 "turn_id":str(catalog[span_id]["turn_id"]),
                 "session_id":str(catalog[span_id]["session_id"]),
                 "speaker":str(catalog[span_id]["speaker"]),
                 "exact_text":str(catalog[span_id]["exact_text"])}
                for span_id in relation_bundle["canonical_support_span_ids"]]
    payload = {"input_version":FULL_DIALOGUE_INPUT_VERSION,
        "required_qa_id":f"{relation_bundle['relation_id']}_QA",
        "full_dialogue_turn_count":len(full_dialogue),
        "full_frozen_dialogue_in_original_order":_dialogue_rows(full_dialogue),
        "relation_bundle_sha256":relation_bundle_sha256,
        "frozen_relation_bundle":dict(relation_bundle),
        "frozen_relation_evidence":evidence}
    rules = [
        "Return exactly one question-answer JSON object matching the schema.",
        "Read the complete frozen dialogue in original order, but use exactly the frozen relation support for the answer.",
        "Generate one question and correct answer only; do not generate distractors or options.",
        "The question must require the frozen parent condition and child consequence together.",
        "Do not switch facts, add evidence, remove evidence, or ask a parent-only or child-only lookup question.",
        "Copy answer_support_span_ids as exactly the frozen canonical support set and copy the required qa_id.",
        "Do not add unsupported answer details. reasoning_steps must be an array and may be empty.",
        "Do not output answer letters, quotes, difficulty, reasoning type, explanations, or extra fields.",
        "Return JSON only without Markdown or surrounding text."]
    return "\n\n".join([
        f"PROMPT_VERSION: {QA_WRITER_FULL_DIALOGUE_PROMPT_VERSION}",
        "QUESTION-ANSWER WRITER RULES\n- " + "\n- ".join(rules),
        "COMPLETE FROZEN DIALOGUE AND RELATION INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            QA_WRITER_JSON_SCHEMA, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"))])


def build_full_dialogue_distractor_writer_prompt(*,
        lineage_view: Mapping[str, Any],
        full_dialogue: Sequence[Mapping[str, Any]],
        relation_bundle: Mapping[str, Any], relation_bundle_sha256: str,
        question_answer_object: Mapping[str, Any], question_answer_sha256: str,
        attempt_index: int,
        structured_feedback: Mapping[str, Any] | None) -> str:
    from .lineage_distractor_writer import validate_attempt_feedback
    feedback = validate_attempt_feedback(attempt_index, structured_feedback)
    catalog = span_catalog_index(dict(lineage_view["span_catalog"]))
    evidence = [{"span_id":span_id,
                 "turn_id":str(catalog[span_id]["turn_id"]),
                 "session_id":str(catalog[span_id]["session_id"]),
                 "speaker":str(catalog[span_id]["speaker"]),
                 "exact_text":str(catalog[span_id]["exact_text"])}
                for span_id in relation_bundle["canonical_support_span_ids"]]
    payload = {"input_version":FULL_DIALOGUE_INPUT_VERSION,
        "attempt_index":attempt_index,
        "full_dialogue_turn_count":len(full_dialogue),
        "full_frozen_dialogue_in_original_order":_dialogue_rows(full_dialogue),
        "relation_bundle_sha256":relation_bundle_sha256,
        "question_answer_sha256":question_answer_sha256,
        "frozen_relation_bundle":dict(relation_bundle),
        "frozen_relation_evidence":evidence,
        "frozen_question_answer":{key:question_answer_object[key] for key in (
            "qa_id", "question", "correct_answer_text",
            "answer_support_span_ids")},
        "previous_attempt_structured_feedback":feedback}
    rules = [
        "Return exactly one JSON object containing exactly three distinct distractors.",
        "Read the complete frozen dialogue in original order; use its real alternatives to make near-neighbor errors.",
        "Do not modify or restate the relation, question, correct answer, or support IDs.",
        "Each distractor must answer the same question and have the same answer type as the correct answer.",
        "Use parallel grammar and comparable length, specificity, detail, sentiment, and plausibility.",
        "All four choices should look plausible without the dialogue; prefer errors in actor, time, trigger, state, consequence, or order.",
        "Do not use absurd, unrelated, generic, cross-type, or uniquely short distractors.",
        "Do not copy the correct answer or repeat a distractor.",
        "If attempt 1 feedback exists, use only its fixed labels and prior strings; do not infer hidden Solver state.",
        "Return JSON only without Markdown or surrounding text."]
    return "\n\n".join([
        f"PROMPT_VERSION: {DISTRACTOR_WRITER_FULL_DIALOGUE_PROMPT_VERSION}",
        "CONTRASTIVE DISTRACTOR WRITER RULES\n- " + "\n- ".join(rules),
        "COMPLETE FROZEN DIALOGUE AND QA INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            DISTRACTOR_WRITER_JSON_SCHEMA, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"))])


def _template_sha(version: str, builder: Any, schema: Mapping[str, Any]) -> str:
    material = (FULL_DIALOGUE_INPUT_VERSION + version + inspect.getsource(builder) +
                canonical_json(schema))
    return sha256_bytes(material.encode())


def full_dialogue_relation_prompt_sha256() -> str:
    return _template_sha(RELATION_MINER_FULL_DIALOGUE_PROMPT_VERSION,
                         build_full_dialogue_relation_miner_prompt,
                         RELATION_MINER_JSON_SCHEMA)


def full_dialogue_qa_prompt_sha256() -> str:
    return _template_sha(QA_WRITER_FULL_DIALOGUE_PROMPT_VERSION,
                         build_full_dialogue_qa_writer_prompt,
                         QA_WRITER_JSON_SCHEMA)


def full_dialogue_distractor_prompt_sha256() -> str:
    return _template_sha(DISTRACTOR_WRITER_FULL_DIALOGUE_PROMPT_VERSION,
                         build_full_dialogue_distractor_writer_prompt,
                         DISTRACTOR_WRITER_JSON_SCHEMA)
