"""Authoritative v1.1 revision-1 schemas and deterministic QA validation."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from jsonschema import validate


QUESTION_TYPES = ("single_choice", "multi_select", "ordering")


def option_keys(question_type: str, count: int) -> tuple[str, ...]:
    if question_type == "single_choice":
        if count != 4:
            raise ValueError("single_choice_requires_exactly_four_options")
        return ("A", "B", "C", "D")
    if question_type == "multi_select":
        if count not in (4, 5, 6):
            raise ValueError("multi_select_requires_four_to_six_options")
        return tuple(chr(ord("A") + index) for index in range(count))
    if question_type == "ordering":
        if count not in (4, 5, 6):
            raise ValueError("ordering_requires_four_to_six_items")
        return tuple(f"I{index + 1}" for index in range(count))
    raise ValueError(f"unsupported_question_type:{question_type}")


def _text_properties(prefix: str, maximum: int) -> dict[str, Any]:
    if prefix == "A" and maximum == 4:
        properties = {key: {"type": "string", "minLength": 1} for key in "ABCD"}
        return {"type": "object", "additionalProperties": False, "properties": properties,
                "required": list("ABCD"), "minProperties": 4, "maxProperties": 4}
    variants = []
    for count in range(4, maximum + 1):
        keys = (
            tuple(chr(ord("A") + index) for index in range(count))
            if prefix == "A" else tuple(f"I{index + 1}" for index in range(count))
        )
        properties = {key: {"type": "string", "minLength": 1} for key in keys}
        variants.append({"type": "object", "additionalProperties": False,
                         "properties": properties, "required": list(keys),
                         "minProperties": count, "maxProperties": count})
    return {"oneOf": variants}


def _base_properties() -> dict[str, Any]:
    return {
        "outcome": {"enum": ["QA", "NO_SAFE_EVOLUTION"]},
        "question_type": {"type": ["string", "null"], "enum": [*QUESTION_TYPES, None]},
        "new_question": {"type": ["string", "null"]},
        "options_or_items": {"type": ["object", "null"]},
        "canonical_gold": {"type": ["array", "null"], "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "selected_evidence_relations": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "expected_context_dependency": {"type": ["string", "null"]},
        "r03_semantic_contribution_claimed": {"type": ["boolean", "null"]},
        "answerable_without_r03_claim": {"type": ["boolean", "null"]},
        "no_safe_evolution_reason": {"type": ["string", "null"]},
    }


STRATEGIES = (
    "STATE_UPDATE", "OBSOLETE_VS_CURRENT", "CROSS_STAGE_EVIDENCE",
    "MULTI_EVENT_DEPENDENCY", "LONG_DISTANCE_DEPENDENCY", "ENTITY_RELATION_BINDING",
    "TEMPORAL_RELATION", "CONSTRAINT_COMPOSITION", "CONTRASTIVE_CONTEXT",
    "OTHER_SUPPORTED_EVOLUTION", "NO_SAFE_EVOLUTION",
)


def build_multi_pass_schema(source_schema: Mapping[str, Any], pass_name: str) -> dict[str, Any]:
    """Build a fresh schema; source_schema is read only and never mutated."""
    schema: dict[str, Any] = copy.deepcopy(dict(source_schema))
    schema["$id"] = f"h4.r03.multi_pass.{pass_name}.output.v1.1.revision_1"
    schema["title"] = f"R03 {pass_name} QA output v1.1 revision 1"
    schema["properties"] = _base_properties()
    schema["required"] = [
        "outcome", "question_type", "new_question", "options_or_items", "canonical_gold",
        "evidence_refs", "selected_evidence_relations", "expected_context_dependency",
        "r03_semantic_contribution_claimed", "answerable_without_r03_claim",
    ]
    if pass_name == "composition":
        schema["properties"].update({
            "source_cluster_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "composition_relation": {"type": ["string", "null"]},
        })
        schema["required"] += ["source_cluster_ids", "composition_relation"]
    else:
        schema["properties"].update({
            "independent_new_qa_reason": {"type": ["string", "null"]},
            "novelty_basis": {"type": ["string", "null"]},
        })
        schema["required"] += ["independent_new_qa_reason", "novelty_basis"]

    qa_common = {
        "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
        "new_question": {"type": "string", "minLength": 1},
        "options_or_items": {"type": "object"},
        "canonical_gold": {"type": "array", "minItems": 1, "maxItems": 6, "uniqueItems": True},
        "evidence_refs": {"type": "array", "minItems": 1},
        "expected_context_dependency": {"type": "string", "minLength": 1},
        "r03_semantic_contribution_claimed": {"type": "boolean"},
        "answerable_without_r03_claim": {"type": "boolean"},
        "no_safe_evolution_reason": {"type": ["string", "null"], "enum": [None, ""]},
        "selected_evidence_relations": {"type": "array", "minItems": 2 if pass_name == "composition" else 1},
    }
    if pass_name == "composition":
        qa_common.update({
            "source_cluster_ids": {"type": "array", "minItems": 2},
            "composition_relation": {"type": "string", "minLength": 1},
        })
    else:
        qa_common.update({
            "independent_new_qa_reason": {"type": "string", "minLength": 1},
            "novelty_basis": {"type": "string", "minLength": 1},
        })
    no_safe = {
        "question_type": {"type": "null"}, "new_question": {"type": "null"},
        "options_or_items": {"type": "null"}, "canonical_gold": {"type": "null"},
        "evidence_refs": {"const": []}, "selected_evidence_relations": {"const": []},
        "expected_context_dependency": {"type": "null"},
        "r03_semantic_contribution_claimed": {"type": "null"},
        "answerable_without_r03_claim": {"type": "null"},
        "no_safe_evolution_reason": {"type": "string", "minLength": 1},
    }
    if pass_name == "composition":
        no_safe.update({"source_cluster_ids": {"type": "array", "minItems": 2}, "composition_relation": {"type": "null"}})
    else:
        no_safe.update({"independent_new_qa_reason": {"type": "null"}, "novelty_basis": {"type": "null"}})

    schema["allOf"] = [
        {"if": {"properties": {"outcome": {"const": "QA"}}}, "then": {"properties": qa_common}},
        {"if": {"properties": {"outcome": {"const": "NO_SAFE_EVOLUTION"}}}, "then": {"required": ["no_safe_evolution_reason"], "properties": no_safe}},
        {"if": {"properties": {"outcome": {"const": "QA"}, "question_type": {"const": "single_choice"}}}, "then": {"properties": {"options_or_items": _text_properties("A", 4), "canonical_gold": {"minItems": 1, "maxItems": 1}}}},
        {"if": {"properties": {"outcome": {"const": "QA"}, "question_type": {"const": "multi_select"}}}, "then": {"properties": {"options_or_items": _text_properties("A", 6), "canonical_gold": {"minItems": 2, "maxItems": 6}}}},
        {"if": {"properties": {"outcome": {"const": "QA"}, "question_type": {"const": "ordering"}}}, "then": {"properties": {"options_or_items": _text_properties("I", 6), "canonical_gold": {"minItems": 4, "maxItems": 6}}}},
    ]
    return schema


def build_targeted_schema(source_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Build targeted schema with explicit question_type discriminator."""
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "h4.round1.r03.qa_feedback_output.v1.1.revision_1",
        "title": "R03 targeted feedback QA output v1.1 revision 1",
        "type": "object", "additionalProperties": False,
        "properties": {
            "source_question_id": {"type": "string", "minLength": 1},
            "source_cluster_id": {"type": "string", "minLength": 1},
            "root_id": {"type": "string", "minLength": 1},
            "selected_strategy": {"type": "string", "enum": list(STRATEGIES)},
            "strategy_rationale": {"type": "string", "minLength": 1},
            "feedback_observations_used": {"type": "array", "items": {"type": "string"}},
            "tentative_hypothesis": {"type": "string"},
            "question_type": {"type": ["string", "null"], "enum": [*QUESTION_TYPES, None]},
            "new_question": {"type": ["string", "null"]},
            "options_or_items": {"type": ["object", "null"]},
            "canonical_gold": {"type": ["array", "null"], "items": {"type": "string"}},
            "evidence_refs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "source_question_relation": {"type": "string"},
            "expected_context_dependency": {"type": "string"},
            "no_safe_evolution_reason": {"type": ["string", "null"]},
        },
        "required": [
            "source_question_id", "source_cluster_id", "root_id", "selected_strategy",
            "strategy_rationale", "feedback_observations_used", "tentative_hypothesis",
            "question_type", "new_question", "options_or_items", "canonical_gold",
            "evidence_refs", "source_question_relation", "expected_context_dependency",
        ],
    }
    qa = {
        "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
        "new_question": {"type": "string", "minLength": 1},
        "options_or_items": {"type": "object"},
        "canonical_gold": {"type": "array", "minItems": 1, "maxItems": 6, "uniqueItems": True},
        "evidence_refs": {"type": "array", "minItems": 1},
        "source_question_relation": {"type": "string", "minLength": 1},
        "expected_context_dependency": {"type": "string", "minLength": 1},
        "no_safe_evolution_reason": {"type": ["string", "null"], "enum": [None, ""]},
    }
    no_safe = {
        "question_type": {"type": "null"}, "new_question": {"type": "null"},
        "options_or_items": {"type": "null"}, "canonical_gold": {"type": "null"},
        "evidence_refs": {"const": []}, "source_question_relation": {"const": ""},
        "expected_context_dependency": {"const": ""},
        "no_safe_evolution_reason": {"type": "string", "minLength": 1},
    }
    schema["allOf"] = [
        {"if": {"properties": {"selected_strategy": {"const": "NO_SAFE_EVOLUTION"}}}, "then": {"required": ["no_safe_evolution_reason"], "properties": no_safe}},
        {"if": {"properties": {"selected_strategy": {"const": "NO_SAFE_EVOLUTION"}}}, "else": {"properties": qa}},
        {"if": {"properties": {"selected_strategy": {"const": "NO_SAFE_EVOLUTION"}}}, "else": {"allOf": [
            {"if": {"properties": {"question_type": {"const": "single_choice"}}}, "then": {"properties": {"options_or_items": _text_properties("A", 4), "canonical_gold": {"minItems": 1, "maxItems": 1}}}},
            {"if": {"properties": {"question_type": {"const": "multi_select"}}}, "then": {"properties": {"options_or_items": _text_properties("A", 6), "canonical_gold": {"minItems": 2, "maxItems": 6}}}},
            {"if": {"properties": {"question_type": {"const": "ordering"}}}, "then": {"properties": {"options_or_items": _text_properties("I", 6), "canonical_gold": {"minItems": 4, "maxItems": 6}}}},
        ]}},
    ]
    return schema


def derive_r03_evidence_refs(evidence_refs: list[str]) -> list[str]:
    return [str(ref) for ref in evidence_refs if str(ref).startswith("R03:")]


def shape_errors(question_type: Any, options: Any, gold: Any) -> list[str]:
    errors: list[str] = []
    if question_type not in QUESTION_TYPES:
        return ["question_type_invalid"]
    if not isinstance(options, dict):
        return ["options_or_items_not_object"]
    keys = set(str(key) for key in options)
    try:
        expected = set(option_keys(str(question_type), len(keys)))
    except ValueError:
        return ["option_count_invalid"]
    if keys != expected:
        errors.append("option_key_pattern_mismatch")
    if not isinstance(gold, list) or len(gold) != len(set(gold)):
        errors.append("gold_not_unique_list")
        return errors
    gold_keys = set(str(key) for key in gold)
    if question_type == "single_choice":
        if len(gold) != 1 or not gold_keys.issubset(keys):
            errors.append("single_gold_invalid")
    elif question_type == "multi_select":
        if not (2 <= len(gold) <= len(keys)) or not gold_keys.issubset(keys):
            errors.append("multi_gold_invalid")
    elif question_type == "ordering":
        if len(gold) != len(keys) or gold_keys != keys:
            errors.append("ordering_gold_not_complete_permutation")
    return errors


def inferred_type(options: Any, gold: Any) -> str:
    """Compatibility-only inference; explicit question_type is authoritative."""
    candidates = [qtype for qtype in QUESTION_TYPES if not shape_errors(qtype, options, gold)]
    return candidates[0] if len(candidates) == 1 else "UNKNOWN"


def local_closure(output: Mapping[str, Any], *, mode: str, known_refs: set[str]) -> dict[str, Any]:
    no_safe = output.get("outcome") == "NO_SAFE_EVOLUTION" or output.get("selected_strategy") == "NO_SAFE_EVOLUTION"
    if no_safe:
        return {"status": "NO_SAFE_EVOLUTION", "errors": [], "evidence_refs_real": True,
                "evidence_completeness": "NOT_REQUIRED_FIRST_PASS", "derived_new_r03_evidence_refs": []}
    qtype = output.get("question_type")
    errors = shape_errors(qtype, output.get("options_or_items"), output.get("canonical_gold"))
    refs = [str(ref) for ref in output.get("evidence_refs") or []]
    if not refs:
        errors.append("evidence_refs_missing")
    if len(refs) != len(set(refs)):
        errors.append("duplicate_evidence_refs")
    unknown = sorted(set(refs) - known_refs)
    if unknown:
        errors.append("evidence_ref_unknown")
    derived = derive_r03_evidence_refs(refs)
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "evidence_refs_real": not unknown, "evidence_completeness": "NOT_REQUIRED_FIRST_PASS",
        "evidence_ref_count": len(refs), "derived_new_r03_evidence_refs": derived,
        "r03_evidence_touch": bool(derived), "question_type_explicit": qtype in QUESTION_TYPES,
        "question_type_inferred_consistency": inferred_type(output.get("options_or_items"), output.get("canonical_gold")) == qtype,
        "semantic_support": "UNKNOWN_NO_SEMANTIC_AUDITOR",
    }


def schema_valid(output: Mapping[str, Any], schema: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        validate(output, schema)
    except Exception as error:
        return False, f"{type(error).__name__}:{error}".split("\n", 1)[0]
    return True, None
