from __future__ import annotations

import inspect
import json
from typing import Any, Mapping

from .core import canonical_json, sha256_bytes


QA_WRITER_PROMPT_VERSION = "spell_lineage_question_answer_writer_single_v1"
QA_WRITER_SCHEMA_VERSION = "spell_lineage_question_answer_schema_v1"
QUESTION_ANSWER_OBJECT_VERSION = "spell_lineage_question_answer_object_v1"

QA_WRITER_JSON_SCHEMA: dict[str, Any] = {
    "type":"object", "additionalProperties":False,
    "required":["qa_id", "question", "correct_answer_text",
                "answer_support_span_ids", "reasoning_steps"],
    "properties":{
        "qa_id":{"type":"string", "minLength":1},
        "question":{"type":"string", "minLength":1},
        "correct_answer_text":{"type":"string", "minLength":1},
        "answer_support_span_ids":{"type":"array", "minItems":2,
            "items":{"type":"string", "minLength":1}, "uniqueItems":True},
        "reasoning_steps":{"type":"array", "items":{"type":"string"}},
    },
}


def qa_writer_schema_sha256() -> str:
    return sha256_bytes(canonical_json(QA_WRITER_JSON_SCHEMA).encode())


def qa_writer_prompt_template_sha256() -> str:
    material = (QA_WRITER_PROMPT_VERSION + QA_WRITER_SCHEMA_VERSION +
                inspect.getsource(build_qa_writer_prompt) +
                canonical_json(QA_WRITER_JSON_SCHEMA))
    return sha256_bytes(material.encode())


def build_qa_writer_prompt(*, lineage_view: Mapping[str, Any],
                           relation_bundle: Mapping[str, Any],
                           relation_bundle_sha256: str) -> str:
    catalog = {span["span_id"]:span for span in
               lineage_view["span_catalog"]["spans"]}
    evidence = [{"span_id":span_id,
                 "source_partition":"parent" if span_id in set(
                     relation_bundle["parent_condition_span_ids"]) else "child",
                 "speaker":catalog[span_id]["speaker"],
                 "exact_text":catalog[span_id]["exact_text"]}
                for span_id in relation_bundle["canonical_support_span_ids"]]
    parent_ids = set(lineage_view["selected_parent_span_ids"])
    lineage_catalog = [{"span_id":span["span_id"],
        "source_partition":"parent" if span["span_id"] in parent_ids else "child",
        "speaker":span["speaker"], "exact_text":span["exact_text"]}
        for span in lineage_view["span_catalog"]["spans"]]
    payload = {"required_qa_id":f"{relation_bundle['relation_id']}_QA",
        "frozen_lineage_view_sha256":lineage_view["view_sha256"],
        "frozen_lineage_span_catalog":lineage_catalog,
        "relation_bundle_sha256":relation_bundle_sha256,
        "frozen_relation_bundle":dict(relation_bundle),
        "frozen_relation_evidence":evidence}
    rules = [
        "Return exactly one question-answer JSON object matching the schema.",
        "Generate one question and its correct answer only; do not generate distractors or options.",
        "The question must require the frozen parent condition and child consequence together.",
        "Do not switch to another fact, add evidence, remove evidence, or ask a parent-only or child-only lookup question.",
        "Copy answer_support_span_ids as exactly the frozen relation canonical support set.",
        "Do not add answer details unsupported by the frozen relation evidence.",
        "Copy the required qa_id exactly. reasoning_steps must be an array and may be empty.",
        "Do not output answer letters, quotes, difficulty, reasoning type, explanations, or fields outside the schema.",
        "Return JSON only without Markdown or surrounding text.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {QA_WRITER_PROMPT_VERSION}",
        "QUESTION-ANSWER WRITER RULES\n- " + "\n- ".join(rules),
        "FROZEN RELATION INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                                sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(QA_WRITER_JSON_SCHEMA,
                                             ensure_ascii=False, sort_keys=True,
                                             indent=2),
    ])


def validate_question_answer_object(raw: Any, *, relation_bundle: Mapping[str, Any],
                                    relation_bundle_sha256: str,
                                    lineage_view: Mapping[str, Any]) -> dict[str, Any]:
    required = set(QA_WRITER_JSON_SCHEMA["required"])
    if not isinstance(raw, dict) or set(raw) != required:
        return {"decision":"qa_schema_invalid", "reasons":["qa_schema_invalid"],
                "question_answer_object":None}
    errors: list[str] = []
    expected_id = f"{relation_bundle['relation_id']}_QA"
    if not isinstance(raw.get("qa_id"), str) or raw["qa_id"] != expected_id:
        errors.append("qa_schema_invalid")
    if not isinstance(raw.get("question"), str) or not raw["question"].strip():
        errors.append("qa_empty_question")
    if (not isinstance(raw.get("correct_answer_text"), str) or
            not raw["correct_answer_text"].strip()):
        errors.append("qa_empty_answer")
    support = raw.get("answer_support_span_ids")
    if (not isinstance(support, list) or
            any(not isinstance(value, str) or not value.strip() for value in support) or
            len(set(support or [])) != len(support or [])):
        errors.append("qa_schema_invalid")
        support = support if isinstance(support, list) else []
    steps = raw.get("reasoning_steps")
    if (not isinstance(steps, list) or
            any(not isinstance(value, str) for value in steps)):
        errors.append("qa_schema_invalid")
    catalog_order = {span["span_id"]:index for index, span in enumerate(
        lineage_view["span_catalog"]["spans"])}
    expected = list(relation_bundle["canonical_support_span_ids"])
    if set(support) != set(expected):
        errors.append("qa_support_mismatch")
        if any(span_id not in set(expected) for span_id in support):
            errors.append("qa_relation_drift")
    if any(span_id not in catalog_order for span_id in support):
        errors.append("qa_relation_drift")
    parent = set(relation_bundle["parent_condition_span_ids"])
    child = set(relation_bundle["child_consequence_span_ids"])
    if not parent.intersection(support) or not child.intersection(support):
        errors.append("qa_support_mismatch")
    if errors:
        ordered = [label for label in ("qa_schema_invalid", "qa_relation_drift",
            "qa_support_mismatch", "qa_empty_question", "qa_empty_answer")
            if label in errors]
        return {"decision":ordered[0], "reasons":ordered,
                "question_answer_object":None}
    ordered_support = sorted(support, key=catalog_order.__getitem__)
    obj = {"question_answer_object_version":QUESTION_ANSWER_OBJECT_VERSION,
        "qa_id":raw["qa_id"], "question":raw["question"].strip(),
        "correct_answer_text":raw["correct_answer_text"].strip(),
        "answer_support_span_ids":ordered_support,
        "reasoning_steps":list(steps),
        "relation_id":relation_bundle["relation_id"],
        "relation_bundle_sha256":relation_bundle_sha256,
        "source_lineage_view_sha256":lineage_view["view_sha256"]}
    obj_sha = sha256_bytes(canonical_json(obj).encode())
    return {"decision":"qa_accept", "reasons":[],
            "question_answer_object":obj,
            "question_answer_object_sha256":obj_sha}
