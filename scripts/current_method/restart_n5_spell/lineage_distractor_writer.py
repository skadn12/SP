from __future__ import annotations

import inspect
import json
import re
import unicodedata
from typing import Any, Mapping

from .core import canonical_json, sha256_bytes


DISTRACTOR_WRITER_PROMPT_VERSION = "spell_lineage_distractor_writer_single_v1"
DISTRACTOR_WRITER_SCHEMA_VERSION = "spell_lineage_distractor_schema_v1"

DISTRACTOR_WRITER_JSON_SCHEMA: dict[str, Any] = {
    "type":"object", "additionalProperties":False,
    "required":["distractors"],
    "properties":{"distractors":{"type":"array", "minItems":3,
        "maxItems":3, "uniqueItems":True,
        "items":{"type":"string", "minLength":1}}},
}

STRUCTURED_FEEDBACK_LABELS = frozenset({
    "bad_ground", "minimal_admission_reject",
    "bad_ground_semantic_leakage", "bad_ground_position_bias",
    "bad_ground_mixed_bias", "bad_ground_permutation_inconclusive",
    "reject_easy", "reject_all_wrong", "reject_responder_format",
    "bad_no_context_eval", "bad_evidence_relevance", "bad_option_bias",
    "frontier_audit_inconclusive", "evidence_audit_inconclusive",
    "option_audit_inconclusive",
})


def distractor_schema_sha256() -> str:
    return sha256_bytes(canonical_json(DISTRACTOR_WRITER_JSON_SCHEMA).encode())


def distractor_prompt_template_sha256() -> str:
    material = (DISTRACTOR_WRITER_PROMPT_VERSION + DISTRACTOR_WRITER_SCHEMA_VERSION +
                inspect.getsource(build_distractor_writer_prompt) +
                canonical_json(DISTRACTOR_WRITER_JSON_SCHEMA))
    return sha256_bytes(material.encode())


def normalize_distractor_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"\s+", " ", text)


def validate_attempt_feedback(attempt_index: int,
                              feedback: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if attempt_index == 0:
        if feedback is not None:
            raise ValueError("attempt0_feedback_forbidden")
        return None
    if attempt_index != 1 or not isinstance(feedback, Mapping):
        raise ValueError("attempt1_structured_feedback_required")
    if set(feedback) != {"previous_distractors", "previous_final_decision",
                         "previous_grounding_diagnosis"}:
        raise ValueError("attempt1_feedback_schema_invalid")
    distractors = feedback["previous_distractors"]
    if (not isinstance(distractors, list) or len(distractors) != 3 or
            any(not isinstance(value, str) for value in distractors)):
        raise ValueError("attempt1_feedback_schema_invalid")
    for key in ("previous_final_decision", "previous_grounding_diagnosis"):
        if feedback[key] not in STRUCTURED_FEEDBACK_LABELS:
            raise ValueError("attempt1_feedback_label_invalid")
    return {"previous_distractors":list(distractors),
        "previous_final_decision":str(feedback["previous_final_decision"]),
        "previous_grounding_diagnosis":str(feedback["previous_grounding_diagnosis"])}


def build_distractor_writer_prompt(*, lineage_view: Mapping[str, Any],
                                   relation_bundle: Mapping[str, Any],
                                   relation_bundle_sha256: str,
                                   question_answer_object: Mapping[str, Any],
                                   question_answer_sha256: str,
                                   attempt_index: int,
                                   structured_feedback: Mapping[str, Any] | None) -> str:
    feedback = validate_attempt_feedback(attempt_index, structured_feedback)
    catalog = {span["span_id"]:span for span in
               lineage_view["span_catalog"]["spans"]}
    evidence = [{"span_id":span_id, "speaker":catalog[span_id]["speaker"],
                 "exact_text":catalog[span_id]["exact_text"]}
                for span_id in relation_bundle["canonical_support_span_ids"]]
    parent_ids = set(lineage_view["selected_parent_span_ids"])
    lineage_catalog = [{"span_id":span["span_id"],
        "source_partition":"parent" if span["span_id"] in parent_ids else "child",
        "speaker":span["speaker"], "exact_text":span["exact_text"]}
        for span in lineage_view["span_catalog"]["spans"]]
    payload = {"attempt_index":attempt_index,
        "frozen_lineage_view_sha256":lineage_view["view_sha256"],
        "frozen_lineage_span_catalog":lineage_catalog,
        "relation_bundle_sha256":relation_bundle_sha256,
        "question_answer_sha256":question_answer_sha256,
        "frozen_relation_bundle":dict(relation_bundle),
        "frozen_relation_evidence":evidence,
        "frozen_question_answer":{
            "qa_id":question_answer_object["qa_id"],
            "question":question_answer_object["question"],
            "correct_answer_text":question_answer_object["correct_answer_text"],
            "answer_support_span_ids":question_answer_object["answer_support_span_ids"]},
        "previous_attempt_structured_feedback":feedback}
    rules = [
        "Return exactly one JSON object containing exactly three distinct distractors.",
        "Do not modify or restate the relation, question, correct answer, or support span IDs.",
        "Each distractor must answer the same question and have the same answer type as the correct answer.",
        "Use parallel grammar and comparable length, specificity, detail, sentiment, and plausibility.",
        "All four answer choices should look plausible without seeing the dialogue.",
        "Prefer a near-neighbor error changing one key relation: actor, time, trigger, initial state, revised state, consequence, or ordering.",
        "Do not use absurd, unrelated, generic, cross-type, or uniquely short distractors.",
        "Do not copy the correct answer and do not repeat a distractor.",
        "The dialogue evidence is the only factual source. Do not output support, quotes, explanations, notes, or fields outside the schema.",
        "If attempt 1 feedback exists, use only those fixed labels and prior distractor strings; do not infer hidden Solver state.",
        "Return JSON only without Markdown or surrounding text.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {DISTRACTOR_WRITER_PROMPT_VERSION}",
        "CONTRASTIVE DISTRACTOR WRITER RULES\n- " + "\n- ".join(rules),
        "FROZEN INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                       sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(DISTRACTOR_WRITER_JSON_SCHEMA,
                                             ensure_ascii=False, sort_keys=True,
                                             indent=2),
    ])


def validate_distractor_object(raw: Any, *, correct_answer_text: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"distractors"}:
        return {"decision":"distractor_schema_invalid",
                "reasons":["distractor_schema_invalid"], "distractors":None}
    distractors = raw.get("distractors")
    if (not isinstance(distractors, list) or len(distractors) != 3 or
            any(not isinstance(value, str) or not value.strip()
                for value in distractors)):
        return {"decision":"distractor_schema_invalid",
                "reasons":["distractor_schema_invalid"], "distractors":None}
    normalized = [normalize_distractor_text(value) for value in distractors]
    if len(set(normalized)) != 3:
        return {"decision":"distractor_schema_invalid",
                "reasons":["distractor_duplicate"], "distractors":None}
    if normalize_distractor_text(correct_answer_text) in set(normalized):
        return {"decision":"distractor_schema_invalid",
                "reasons":["distractor_matches_correct_answer"],
                "distractors":None}
    return {"decision":"distractor_accept", "reasons":[],
            "distractors":[value.strip() for value in distractors]}
