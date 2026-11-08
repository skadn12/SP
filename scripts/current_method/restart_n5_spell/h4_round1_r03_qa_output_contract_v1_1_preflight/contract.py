"""Schema and local closure rules for the proposed QA output contract."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from jsonschema import validate


QUESTION_TYPES = ("single_choice", "multi_select", "ordering")


def _option_schema(question_type: str) -> dict[str, Any]:
    if question_type == "single_choice":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {key: {"type": "string", "minLength": 1} for key in ("A", "B", "C", "D")},
            "required": ["A", "B", "C", "D"],
            "minProperties": 4,
            "maxProperties": 4,
        }
    if question_type == "multi_select":
        return {
            "type": "object",
            "additionalProperties": {"type": "string", "minLength": 1},
            "propertyNames": {"pattern": "^[A-F]$"},
            "minProperties": 4,
            "maxProperties": 6,
        }
    return {
        "type": "object",
        "additionalProperties": {"type": "string", "minLength": 1},
        "propertyNames": {"pattern": "^I[1-6]$"},
        "minProperties": 4,
        "maxProperties": 6,
    }


def build_schema(source_schema: Mapping[str, Any], pass_name: str) -> dict[str, Any]:
    """Derive v1.1 without changing the immutable v1 schema artifact."""
    schema = copy.deepcopy(dict(source_schema))
    schema["$id"] = f"h4.r03.multi_pass.{pass_name}.output.v1.1"
    schema["title"] = f"R03 {pass_name} QA output v1.1"
    schema["required"] = [
        field for field in schema.get("required", [])
        if field not in {"new_r03_evidence_refs", "no_safe_evolution_reason"}
    ]
    schema.get("properties", {}).pop("new_r03_evidence_refs", None)
    schema.setdefault("properties", {})["no_safe_evolution_reason"] = {
        "type": ["string", "null"]
    }

    qa_properties: dict[str, Any] = {
        "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
        "new_question": {"type": "string", "minLength": 1},
        "options_or_items": {"type": "object"},
        "canonical_gold": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": True},
        "expected_context_dependency": {"type": "string", "minLength": 1},
        "r03_semantic_contribution_claimed": {"type": "boolean"},
        "answerable_without_r03_claim": {"type": "boolean"},
        "no_safe_evolution_reason": {"enum": [None, ""], "type": ["string", "null"]},
    }
    if pass_name == "composition":
        qa_properties.update({
            "source_cluster_ids": {"type": "array", "minItems": 2, "items": {"type": "string"}, "uniqueItems": True},
            "composition_relation": {"type": "string", "minLength": 1},
            "selected_evidence_relations": {"type": "array", "minItems": 2, "items": {"type": "string"}, "uniqueItems": True},
        })
    else:
        qa_properties.update({
            "independent_new_qa_reason": {"type": "string", "minLength": 1},
            "novelty_basis": {"type": "string", "minLength": 1},
            "selected_evidence_relations": {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": True},
        })

    no_safe_properties = {
        "question_type": {"type": "null"},
        "new_question": {"type": "null"},
        "options_or_items": {"type": "null"},
        "canonical_gold": {"type": "null"},
        "evidence_refs": {"const": []},
        "selected_evidence_relations": {"const": []},
        "expected_context_dependency": {"type": "null"},
        "r03_semantic_contribution_claimed": {"type": "null"},
        "answerable_without_r03_claim": {"type": "null"},
        "no_safe_evolution_reason": {"type": "string", "minLength": 1},
    }
    if pass_name == "composition":
        no_safe_properties.update({
            "source_cluster_ids": {"type": "array", "minItems": 2, "items": {"type": "string"}},
            "composition_relation": {"type": "null"},
        })
    else:
        no_safe_properties.update({
            "independent_new_qa_reason": {"type": "null"},
            "novelty_basis": {"type": "null"},
        })

    schema["allOf"] = [
        {
            "if": {"properties": {"outcome": {"const": "QA"}}},
            "then": {"properties": qa_properties},
        },
        {
            "if": {"properties": {"outcome": {"const": "NO_SAFE_EVOLUTION"}}},
            "then": {"required": ["no_safe_evolution_reason"], "properties": no_safe_properties},
            "else": {"properties": {"no_safe_evolution_reason": {"enum": [None, ""]}}},
        },
        {
            "if": {"properties": {"outcome": {"const": "QA"}, "question_type": {"const": "single_choice"}}},
            "then": {"properties": {"options_or_items": _option_schema("single_choice"), "canonical_gold": {"minItems": 1, "maxItems": 1, "uniqueItems": True}}},
        },
        {
            "if": {"properties": {"outcome": {"const": "QA"}, "question_type": {"const": "multi_select"}}},
            "then": {"properties": {"options_or_items": _option_schema("multi_select"), "canonical_gold": {"minItems": 2, "uniqueItems": True}}},
        },
        {
            "if": {"properties": {"outcome": {"const": "QA"}, "question_type": {"const": "ordering"}}},
            "then": {"properties": {"options_or_items": _option_schema("ordering"), "canonical_gold": {"minItems": 4, "uniqueItems": True}}},
        },
    ]
    return schema


def build_targeted_schema(source_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Build the same v1.1 question/evidence contract for Targeted Evolution."""
    schema = copy.deepcopy(dict(source_schema))
    schema["$id"] = "h4.round1.r03.qa_feedback_output.v1.1"
    schema["title"] = "R03 targeted feedback QA output v1.1"
    schema["required"] = [
        field for field in schema.get("required", [])
        if field not in {"new_r03_evidence_refs", "no_safe_evolution_reason"}
    ]
    schema.get("properties", {}).pop("new_r03_evidence_refs", None)
    schema.setdefault("properties", {})["no_safe_evolution_reason"] = {"type": ["string", "null"]}
    qa_shape = {
        "properties": {
            "new_question": {"type": "string", "minLength": 1},
            "options_or_items": {"type": "object", "minProperties": 4, "maxProperties": 6},
            "canonical_gold": {"type": "array", "minItems": 1, "uniqueItems": True},
            "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": True},
            "source_question_relation": {"type": "string", "minLength": 1},
            "expected_context_dependency": {"type": "string", "minLength": 1},
            "no_safe_evolution_reason": {"enum": [None, ""]},
        },
        "oneOf": [
            {"properties": {"options_or_items": _option_schema("single_choice"), "canonical_gold": {"minItems": 1, "maxItems": 1, "uniqueItems": True}}},
            {"properties": {"options_or_items": _option_schema("multi_select"), "canonical_gold": {"minItems": 2, "maxItems": 5, "uniqueItems": True}}},
            {"properties": {"options_or_items": _option_schema("ordering"), "canonical_gold": {"minItems": 4, "maxItems": 6, "uniqueItems": True}}},
        ],
    }
    schema["allOf"] = [
        {
            "if": {"properties": {"selected_strategy": {"const": "NO_SAFE_EVOLUTION"}}},
            "then": {
                "required": ["no_safe_evolution_reason"],
                "properties": {
                    "new_question": {"type": "null"},
                    "options_or_items": {"type": "null"},
                    "canonical_gold": {"type": "null"},
                    "evidence_refs": {"const": []},
                    "source_question_relation": {"const": ""},
                    "expected_context_dependency": {"const": ""},
                    "no_safe_evolution_reason": {"type": "string", "minLength": 1},
                },
            },
            "else": qa_shape,
        },
        {
            "if": {"properties": {"selected_strategy": {"const": "OTHER_SUPPORTED_EVOLUTION"}}},
            "then": {"properties": {"strategy_rationale": {"type": "string", "minLength": 1}}},
        },
    ]
    return schema


def derive_r03_evidence_refs(evidence_refs: list[str]) -> list[str]:
    """Derive the R03 subset locally; the model no longer emits it twice."""
    return [ref for ref in evidence_refs if str(ref).startswith("R03:")]


def _contiguous_keys(keys: set[str], prefix: str, maximum: int) -> bool:
    expected = {f"{prefix}{index + 1}" if prefix == "I" else chr(ord("A") + index) for index in range(len(keys))}
    return len(keys) >= 4 and len(keys) <= maximum and keys == expected


def infer_question_type(options: Any, gold: Any) -> str:
    keys = set(options) if isinstance(options, dict) else set()
    if keys == {"A", "B", "C", "D"} and isinstance(gold, list) and len(gold) == 1:
        return "single_choice"
    if _contiguous_keys(keys, "A", 6) and isinstance(gold, list) and 2 <= len(gold) < len(keys):
        return "multi_select"
    if _contiguous_keys(keys, "I", 6) and isinstance(gold, list) and len(gold) == len(keys):
        return "ordering"
    return "UNKNOWN"


def local_closure(output: Mapping[str, Any], pass_name: str, known_refs: set[str]) -> dict[str, Any]:
    """Apply deterministic checks that JSON Schema cannot express completely."""
    errors: list[str] = []
    outcome = output.get("outcome")
    selected_strategy = output.get("selected_strategy")
    if outcome == "NO_SAFE_EVOLUTION" or selected_strategy == "NO_SAFE_EVOLUTION":
        return {
            "status": "NO_SAFE_EVOLUTION",
            "errors": [],
            "evidence_refs_real": True,
            "evidence_completeness": "NOT_REQUIRED_FIRST_PASS",
            "derived_new_r03_evidence_refs": [],
        }
    options = output.get("options_or_items")
    gold = output.get("canonical_gold")
    inferred = infer_question_type(options, gold)
    qtype = output.get("question_type") or inferred
    keys = set(options) if isinstance(options, dict) else set()
    if qtype == "single_choice":
        valid_keys = keys == {"A", "B", "C", "D"}
        valid_gold = isinstance(gold, list) and len(gold) == 1 and gold[0] in keys
    elif qtype == "multi_select":
        valid_keys = _contiguous_keys(keys, "A", 6)
        valid_gold = isinstance(gold, list) and 2 <= len(gold) < len(keys) and len(set(gold)) == len(gold) and set(gold) <= keys
    elif qtype == "ordering":
        valid_keys = _contiguous_keys(keys, "I", 6)
        valid_gold = isinstance(gold, list) and len(gold) == len(keys) and len(set(gold)) == len(gold) and set(gold) == keys
    else:
        valid_keys = valid_gold = False
    if inferred != qtype:
        valid_keys = valid_gold = False
    if not valid_keys:
        errors.append("option_key_shape_invalid")
    if not valid_gold:
        errors.append("gold_shape_invalid")
    refs = [str(ref) for ref in output.get("evidence_refs") or []]
    if not refs:
        errors.append("evidence_refs_missing")
    unknown = sorted(set(refs) - known_refs)
    if unknown:
        errors.append("evidence_ref_unknown")
    derived = derive_r03_evidence_refs(refs)
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "evidence_refs_real": not unknown,
        "evidence_completeness": "NOT_REQUIRED_FIRST_PASS",
        "evidence_ref_count": len(refs),
        "derived_new_r03_evidence_refs": derived,
        "r03_evidence_touch": bool(derived),
        "semantic_support": "UNKNOWN_NO_SEMANTIC_AUDITOR",
    }


def schema_valid(output: Mapping[str, Any], schema: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        validate(output, schema)
    except Exception as error:  # jsonschema raises several validation types
        return False, f"{type(error).__name__}:{error}".split("\n", 1)[0]
    return True, None
