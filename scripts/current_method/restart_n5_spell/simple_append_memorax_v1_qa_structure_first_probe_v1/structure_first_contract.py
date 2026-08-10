"""Structure/evidence-first Writer gate layered over the frozen v19.2 gate."""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, median
from typing import Any, Mapping

from .structural_contract import (
    QUESTION_CAPACITY_MAX,
    solver_partition as _legacy_solver_partition,
    validate_structural_writer_output as _validate_legacy,
)


STRUCTURE_FIRST_SCHEMA_VERSION = "structure_evidence_first_writer_v1"
PREFERRED_MIN_EVIDENCE_TURNS = 2
PREFERRED_MAX_EVIDENCE_TURNS = 6
HARD_MAX_EVIDENCE_TURNS = 8

TOP_LEVEL_KEYS = {
    "structure_first_schema_version", "abstention_summary", "questions",
}
QUESTION_KEYS = {
    "candidate_id", "question_type", "source_scope", "structure_plan",
    "reasoning_structure", "question", "options", "gold_text",
    "evidence_turn_ids",
}
STRUCTURE_PLAN_KEYS = {
    "structure_type", "local_evidence_turn_ids", "discriminative_variable",
    "gold_uniqueness", "distractor_sources",
}
GOLD_UNIQUENESS_KEYS = {
    "unique", "supported_option_texts", "support_turn_ids",
}
DISTRACTOR_SOURCE_KEYS = {
    "option_text", "source_turn_ids", "near_miss_kind",
}

STRUCTURE_TYPE_TO_QUESTION_TYPES = {
    "temporal_sequence": {"temporal_sequence"},
    "state_change": {"state_change"},
    "information_revision": {"information_revision"},
    "intended_vs_executed": {"cross_turn_comparison"},
    "information_visibility": {"information_visibility"},
    "task_dependency_or_constraint": {"task_dependency_or_constraint"},
    "causal": {"causal"},
    "competing_result": {"causal"},
    "cross_turn_comparison": {"cross_turn_comparison"},
}
NEAR_MISS_KINDS = {
    "nearby_event", "previous_state", "wrong_timepoint", "wrong_actor",
    "proposed_not_adopted", "intended_not_executed",
    "nearby_alternative_cause", "nearby_alternative_result",
    "event_order_permutation", "cross_pairing",
}
WEAK_DISTRACTOR_PATTERN = re.compile(
    r"\b(ignore(?:d|s|ing)?|hid(?:e|es|ing|den)|blam(?:e|es|ed|ing)|"
    r"quit(?:s|ting)?\s+forever)\b",
    re.IGNORECASE,
)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text_list(value: Any, *, minimum: int = 1) -> bool:
    return (isinstance(value, list) and len(value) >= minimum and
            all(_text(row) for row in value))


def _legacy_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    questions = []
    for row in value.get("questions") or []:
        if isinstance(row, Mapping):
            questions.append({key: item for key, item in row.items()
                              if key != "structure_plan"})
        else:
            questions.append(row)
    return {
        "structural_qa_schema_version": "structural_qa_writer_v19_2",
        "abstention_summary": value.get("abstention_summary"),
        "questions": questions,
    }


def _validate_structure_plan(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    plan = row.get("structure_plan")
    if not isinstance(plan, Mapping) or set(plan) != STRUCTURE_PLAN_KEYS:
        return ["structure_plan_schema"], warnings

    structure_type = str(plan.get("structure_type") or "")
    if structure_type not in STRUCTURE_TYPE_TO_QUESTION_TYPES:
        errors.append("structure_type")
    elif str(row.get("question_type") or "") not in \
            STRUCTURE_TYPE_TO_QUESTION_TYPES[structure_type]:
        errors.append("structure_type_question_type_mismatch")

    evidence = row.get("evidence_turn_ids")
    local_evidence = plan.get("local_evidence_turn_ids")
    if (not _text_list(local_evidence) or
            len(local_evidence) != len(set(local_evidence))):
        errors.append("local_evidence_turn_ids")
        local_evidence = []
    if isinstance(local_evidence, list) and len(local_evidence) < \
            PREFERRED_MIN_EVIDENCE_TURNS:
        errors.append("local_evidence_below_minimum")
    if isinstance(local_evidence, list) and len(local_evidence) > \
            HARD_MAX_EVIDENCE_TURNS:
        errors.append("local_evidence_over_hard_maximum")
    elif isinstance(local_evidence, list) and len(local_evidence) > \
            PREFERRED_MAX_EVIDENCE_TURNS:
        warnings.append("local_evidence_above_preferred_range")
    if not isinstance(evidence, list) or list(local_evidence) != list(evidence):
        errors.append("structure_plan_evidence_mismatch")

    variable = str(plan.get("discriminative_variable") or "")
    reasoning = row.get("reasoning_structure")
    reasoning_variable = (str(reasoning.get("discriminative_variable") or "")
                          if isinstance(reasoning, Mapping) else "")
    if not variable or variable != reasoning_variable:
        errors.append("structure_plan_variable_mismatch")

    options = row.get("options") if isinstance(row.get("options"), Mapping) else {}
    gold = row.get("gold_text")
    incorrect = [str(value) for value in options.values() if value != gold]
    uniqueness = plan.get("gold_uniqueness")
    if (not isinstance(uniqueness, Mapping) or
            set(uniqueness) != GOLD_UNIQUENESS_KEYS):
        errors.append("gold_uniqueness_schema")
    else:
        supported = uniqueness.get("supported_option_texts")
        support_refs = uniqueness.get("support_turn_ids")
        if uniqueness.get("unique") is not True or supported != [gold]:
            errors.append("gold_not_uniquely_supported")
        if (not _text_list(support_refs) or
                not set(support_refs).issubset(set(local_evidence))):
            errors.append("gold_support_not_local")

    sources = plan.get("distractor_sources")
    if not isinstance(sources, list) or len(sources) != 3:
        errors.append("distractor_source_count")
        sources = []
    source_texts: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != DISTRACTOR_SOURCE_KEYS:
            errors.append("distractor_source_schema")
            continue
        option_text = source.get("option_text")
        refs = source.get("source_turn_ids")
        kind = source.get("near_miss_kind")
        if not _text(option_text):
            errors.append("distractor_source_option_text")
        else:
            source_texts.append(str(option_text))
            if WEAK_DISTRACTOR_PATTERN.search(str(option_text)):
                errors.append("weak_invented_distractor")
        if (not _text_list(refs) or
                not set(refs).issubset(set(local_evidence))):
            errors.append("distractor_source_not_local")
        if kind not in NEAR_MISS_KINDS:
            errors.append("near_miss_kind")
    if Counter(source_texts) != Counter(incorrect):
        errors.append("distractor_option_source_mismatch")
    return sorted(set(errors)), sorted(set(warnings))


def validate_structure_first_writer_output(
    value: Any, *, expanded_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate discovery artifacts, then run the frozen v19.2 validator."""
    top_errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != TOP_LEVEL_KEYS:
        top_errors.append("structure_first_top_level_schema")
        value = value if isinstance(value, Mapping) else {}
    if value.get("structure_first_schema_version") != \
            STRUCTURE_FIRST_SCHEMA_VERSION:
        top_errors.append("structure_first_schema_version")

    questions = value.get("questions")
    if not isinstance(questions, list):
        top_errors.append("questions_schema")
        questions = []
    legacy = _validate_legacy(
        _legacy_envelope(value), expanded_dialogue=expanded_dialogue)

    base_items = list(legacy.get("items") or [])
    combined_items: list[dict[str, Any]] = []
    structure_plans: list[dict[str, Any]] = []
    valid_ids: set[str] = set()
    for index, raw_row in enumerate(questions, 1):
        row = raw_row if isinstance(raw_row, Mapping) else {}
        plan_errors: list[str] = []
        plan_warnings: list[str] = []
        if not isinstance(raw_row, Mapping) or set(row) != QUESTION_KEYS:
            plan_errors.append("structure_first_candidate_schema")
        else:
            plan_errors, plan_warnings = _validate_structure_plan(row)
        base = base_items[index - 1] if index <= len(base_items) else {
            "candidate_id": str(row.get("candidate_id") or ""),
            "structural_errors": ["legacy_candidate_missing"],
            "warnings": [],
        }
        errors = sorted(set(
            list(base.get("structural_errors") or []) + plan_errors + top_errors))
        warnings = sorted(set(
            list(base.get("warnings") or []) + plan_warnings))
        item = {**base, "valid": not errors, "structural_errors": errors,
                "warnings": warnings}
        combined_items.append(item)
        plan = row.get("structure_plan") if isinstance(row, Mapping) else {}
        structure_plans.append({
            "candidate_id": str(row.get("candidate_id") or ""),
            "structure_type": (plan.get("structure_type")
                               if isinstance(plan, Mapping) else None),
            "gold_unique": ((plan.get("gold_uniqueness") or {}).get("unique")
                            if isinstance(plan, Mapping) else None),
            "distractor_sources": (list(plan.get("distractor_sources") or [])
                                   if isinstance(plan, Mapping) else []),
        })
        if not errors:
            valid_ids.add(str(row.get("candidate_id") or ""))

    base_maps = list(legacy.get("materialization_map") or [])
    base_normalized = list(
        (legacy.get("normalized_batch") or {}).get("questions") or [])
    base_statuses = list(legacy.get("normalized_statuses") or [])
    normalized_by_raw: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for position, mapping in enumerate(base_maps):
        if position < len(base_normalized) and position < len(base_statuses):
            normalized_by_raw[str(mapping.get("raw_candidate_id") or "")] = (
                base_normalized[position], base_statuses[position])
    normalized_rows: list[dict[str, Any]] = []
    normalized_statuses: list[dict[str, Any]] = []
    materialization_map: list[dict[str, str]] = []
    for item in combined_items:
        candidate_id = str(item.get("candidate_id") or "")
        pair = normalized_by_raw.get(candidate_id)
        if candidate_id not in valid_ids or pair is None:
            continue
        normalized_rows.append(pair[0])
        status = {**pair[1], "question_index": len(normalized_rows)}
        normalized_statuses.append(status)
        materialization_map.append({
            "raw_candidate_id": candidate_id,
            "materialized_candidate_id": f"Q{len(normalized_rows):02d}",
        })

    counts = [int(row.get("evidence_count") or 0)
              for row in combined_items if row.get("valid")]
    valid_scopes = [str(row.get("actual_source_scope") or "")
                    for row in combined_items if row.get("valid")]
    valid_types = [str(row.get("question_type") or "")
                   for row in combined_items if row.get("valid")]
    valid_variables = [str(row.get("discriminative_variable") or "")
                       for row in combined_items if row.get("valid")]
    kind_counts = Counter(
        str(source.get("near_miss_kind") or "")
        for index, plan in enumerate(structure_plans)
        if index < len(combined_items) and combined_items[index].get("valid")
        for source in plan.get("distractor_sources") or [])
    return {
        "envelope_valid": not top_errors and bool(legacy.get("envelope_valid")),
        "envelope_errors": sorted(set(
            top_errors + list(legacy.get("envelope_errors") or []))),
        "question_count": len(questions),
        "structural_valid_count": len(normalized_rows),
        "structural_invalid_count": len(questions) - len(normalized_rows),
        "validity_rate": (len(normalized_rows) / len(questions)
                          if questions else None),
        "items": combined_items,
        "diagnostics": [
            {**item, "structure_plan": structure_plans[index]}
            for index, item in enumerate(combined_items)
        ],
        "normalized_batch": {
            "mcq_batch_schema_version": "simple_append_mcq_batch_v5",
            "questions": normalized_rows,
        },
        "normalized_statuses": normalized_statuses,
        "materialization_map": materialization_map,
        "zero_question_batch_valid": (
            not questions and not top_errors and
            bool(legacy.get("zero_question_batch_valid"))),
        "child_supported_count": sum(
            scope in {"child_only", "parent_child"} for scope in valid_scopes),
        "nonempty_batch_has_child_support": (
            not normalized_rows or any(
                scope in {"child_only", "parent_child"}
                for scope in valid_scopes)),
        "discriminative_variable_distribution": dict(
            sorted(Counter(valid_variables).items())),
        "question_type_distribution": dict(sorted(Counter(valid_types).items())),
        "distractor_source_distribution": dict(sorted(kind_counts.items())),
        "evidence_statistics": {
            "min": min(counts) if counts else None,
            "median": median(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": mean(counts) if counts else None,
            "over_8_count": sum(
                int(row.get("evidence_count") or 0) > 8
                for row in combined_items),
            "over_6_valid_count": sum(count > 6 for count in counts),
        },
    }


def solver_partition(validation: Mapping[str, Any]) -> dict[str, Any]:
    return _legacy_solver_partition(validation)
