"""Deterministic Structural QA Writer v19 validation and materialization."""

from __future__ import annotations

import re
from statistics import mean, median
from typing import Any, Mapping

from ..core import normalize_option
from ..simple_append_v1.qa import (
    QUESTION_TYPES,
    assess_option_surface_symmetry,
    resolve_evidence_turn_refs,
)


STRUCTURAL_QA_SCHEMA_VERSION = "structural_qa_writer_v19_2"
QUESTION_CAPACITY_MAX = 12
MAX_EVIDENCE_TURNS = 8
OPTION_KEYS = {"A", "B", "C", "D"}
SOURCE_SCOPES = {"parent_only", "child_only", "parent_child"}
DISCRIMINATIVE_VARIABLES = {
    "event_order",
    "knowledge_holder_at_timepoint",
    "previous_vs_current_state",
    "old_vs_revised_plan",
    "adopted_decision",
    "prerequisite",
    "same_dimension_comparison",
    "dialogue_specific_cause",
    "cross_context_synthesis",
}
ABSTENTION_REASONS = {
    "no_discriminative_structure",
    "generic_commonsense",
    "weak_distractors",
    "evidence_over_limit",
    "duplicate_candidate",
    "other",
}
COMMON_REASONING_KEYS = {
    "discriminative_variable", "supporting_turn_ids", "short_rationale",
    "type_contract",
}
QUESTION_KEYS = {
    "candidate_id", "question_type", "source_scope", "reasoning_structure",
    "question", "options", "gold_text", "evidence_turn_ids",
}

TYPE_CONTRACT_KEYS = {
    "temporal_sequence": {"events"},
    "state_change": {
        "subject", "old_state", "update", "new_state", "before_turn_ids",
        "update_turn_ids", "after_turn_ids",
    },
    "information_revision": {
        "subject", "old_state", "update", "new_state", "before_turn_ids",
        "update_turn_ids", "after_turn_ids",
    },
    "information_visibility": {
        "fact", "timepoint", "candidate_knowers", "turn_ids",
    },
    "task_dependency_or_constraint": {
        "prerequisite", "dependent_outcome", "turn_ids",
    },
    "cross_turn_comparison": {
        "comparison_dimension", "left", "right", "left_turn_ids",
        "right_turn_ids",
    },
    "relationship_or_decision": {
        "adopted_decision", "considered_alternative", "turn_ids",
    },
    "causal": {
        "dialogue_specific_cause", "competing_causes", "turn_ids",
    },
    "cross_context_synthesis": {"synthesis", "evidence_groups"},
}

VARIABLES_BY_TYPE = {
    "temporal_sequence": {"event_order"},
    "state_change": {"previous_vs_current_state"},
    "information_revision": {"old_vs_revised_plan"},
    "information_visibility": {"knowledge_holder_at_timepoint"},
    "task_dependency_or_constraint": {"prerequisite"},
    "cross_turn_comparison": {"same_dimension_comparison"},
    "relationship_or_decision": {"adopted_decision"},
    "causal": {"dialogue_specific_cause"},
    "cross_context_synthesis": {"cross_context_synthesis"},
}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text_list(value: Any, *, minimum: int = 1) -> bool:
    return (isinstance(value, list) and len(value) >= minimum and
            all(_text(row) for row in value))


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _exact_duplicate_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    options = row.get("options") if isinstance(row.get("options"), Mapping) else {}
    return (
        _normalize_text(row.get("question")),
        _normalize_text(row.get("gold_text")),
        tuple(sorted(_normalize_text(value) for value in options.values())),
    )


def _contract_ref_lists(question_type: str, value: Any) -> list[list[str]]:
    if not isinstance(value, Mapping):
        return []
    if question_type == "temporal_sequence":
        return [list(row.get("turn_ids") or [])
                for row in value.get("events") or []
                if isinstance(row, Mapping)]
    if question_type == "cross_context_synthesis":
        return [list(row.get("turn_ids") or [])
                for row in value.get("evidence_groups") or []
                if isinstance(row, Mapping)]
    return [
        list(row) for key, row in value.items()
        if key.endswith("turn_ids") and isinstance(row, list)
    ]


def _validate_type_contract(
    question_type: str, contract: Any, supporting_refs: set[str],
) -> list[str]:
    errors: list[str] = []
    expected = TYPE_CONTRACT_KEYS.get(question_type)
    if not isinstance(contract, Mapping) or expected is None:
        return ["type_contract_schema"]
    if set(contract) != expected:
        errors.append("type_contract_keys")
        return errors

    if question_type == "temporal_sequence":
        events = contract.get("events")
        if (not isinstance(events, list) or len(events) < 3 or
                any(not isinstance(row, Mapping) or
                    set(row) != {"event", "turn_ids"} or
                    not _text(row.get("event")) or
                    not _text_list(row.get("turn_ids")) for row in events)):
            errors.append("temporal_events")
    elif question_type in {"state_change", "information_revision"}:
        if any(not _text(contract.get(key))
               for key in ("subject", "old_state", "update", "new_state")):
            errors.append("revision_states")
        if any(not _text_list(contract.get(key)) for key in (
                "before_turn_ids", "update_turn_ids", "after_turn_ids")):
            errors.append("revision_evidence_groups")
    elif question_type == "information_visibility":
        if (not _text(contract.get("fact")) or
                not _text(contract.get("timepoint")) or
                not _text_list(contract.get("candidate_knowers"), minimum=2) or
                len(set(contract.get("candidate_knowers") or [])) < 2 or
                not _text_list(contract.get("turn_ids"))):
            errors.append("visibility_contract")
    elif question_type == "task_dependency_or_constraint":
        if (not _text(contract.get("prerequisite")) or
                not _text(contract.get("dependent_outcome")) or
                not _text_list(contract.get("turn_ids"))):
            errors.append("dependency_contract")
    elif question_type == "cross_turn_comparison":
        if any(not _text(contract.get(key))
               for key in ("comparison_dimension", "left", "right")):
            errors.append("comparison_contract")
        if (not _text_list(contract.get("left_turn_ids")) or
                not _text_list(contract.get("right_turn_ids"))):
            errors.append("comparison_evidence_groups")
    elif question_type == "relationship_or_decision":
        alternative = contract.get("considered_alternative")
        if (not _text(contract.get("adopted_decision")) or
                alternative is not None and not _text(alternative) or
                not _text_list(contract.get("turn_ids"))):
            errors.append("decision_contract")
    elif question_type == "causal":
        if (not _text(contract.get("dialogue_specific_cause")) or
                not _text_list(contract.get("competing_causes"), minimum=2) or
                not _text_list(contract.get("turn_ids"))):
            errors.append("causal_contract")
    elif question_type == "cross_context_synthesis":
        groups = contract.get("evidence_groups")
        if (not _text(contract.get("synthesis")) or
                not isinstance(groups, list) or len(groups) < 2 or
                any(not isinstance(row, Mapping) or
                    set(row) != {"group", "turn_ids"} or
                    not _text(row.get("group")) or
                    not _text_list(row.get("turn_ids")) for row in groups)):
            errors.append("synthesis_contract")

    for refs in _contract_ref_lists(question_type, contract):
        if not refs or len(refs) != len(set(refs)):
            errors.append("type_contract_evidence_duplicate_or_empty")
        if any(ref not in supporting_refs for ref in refs):
            errors.append("type_contract_evidence_outside_support")
    return sorted(set(errors))


def _validate_type_specific_answer_shape(
    question_type: str,
    contract: Mapping[str, Any] | None,
    question: Any,
    options: Mapping[str, Any],
    gold_text: Any,
) -> list[str]:
    """Check that the item asks the declared type's answer question.

    These checks are deliberately structural. They prevent an actor-recall item
    from being labelled as visibility, and prevent a state/revision or dependency
    item from being a loose temporal/cause paraphrase. They do not decide whether
    the dialogue's claim is true; that remains an evaluation concern.
    """
    if not isinstance(contract, Mapping) or not _text(question):
        return []
    normalized_question = _normalize_text(question)
    normalized_gold = _normalize_text(gold_text)
    errors: list[str] = []
    if question_type == "information_visibility":
        fact = _normalize_text(contract.get("fact"))
        timepoint = _normalize_text(contract.get("timepoint"))
        candidates = [_normalize_text(value)
                     for value in contract.get("candidate_knowers") or []]
        expected = _normalize_text(
            f"Who knew that {contract.get('fact')} at the timepoint "
            f"{contract.get('timepoint')}?")
        if normalized_question != expected:
            errors.append("visibility_question_not_knowledge_only")
        if len(candidates) != 4 or len(set(candidates)) != 4:
            errors.append("visibility_candidate_knower_count")
        option_values = {_normalize_text(options.get(key)) for key in OPTION_KEYS}
        if option_values != set(candidates):
            errors.append("visibility_options_not_candidate_knowers")
        if normalized_gold not in set(candidates):
            errors.append("visibility_gold_not_candidate_knower")
        if not fact or not timepoint:
            errors.append("visibility_contract")
    elif question_type in {"state_change", "information_revision"}:
        subject = contract.get("subject")
        update = contract.get("update")
        old_state = _normalize_text(contract.get("old_state"))
        new_state = _normalize_text(contract.get("new_state"))
        expected = _normalize_text(
            f"After {update}, what was {subject}'s new state?")
        if normalized_question != expected:
            errors.append("revision_question_not_update_focused")
        if normalized_gold != new_state:
            errors.append("revision_gold_not_new_state")
        option_values = {_normalize_text(options.get(key)) for key in OPTION_KEYS}
        if old_state not in option_values:
            errors.append("revision_old_state_not_distractor")
    elif question_type == "task_dependency_or_constraint":
        expected = _normalize_text(
            f"What had to happen before {contract.get('dependent_outcome')}?")
        if normalized_question != expected:
            errors.append("dependency_question_not_prerequisite_focused")
        if normalized_gold != _normalize_text(contract.get("prerequisite")):
            errors.append("dependency_gold_not_prerequisite")
    elif question_type == "temporal_sequence":
        events = contract.get("events") or []
        event_texts = [_normalize_text(row.get("event"))
                       for row in events if isinstance(row, Mapping)]
        permutations: dict[tuple[int, ...], str] = {}
        for key in sorted(OPTION_KEYS):
            option = _normalize_text(options.get(key))
            positions: list[int] = []
            for index, event in enumerate(event_texts):
                position = option.find(event)
                if not event or position < 0:
                    positions = []
                    break
                positions.append(position)
            if len(positions) != len(event_texts) or len(set(positions)) != len(positions):
                errors.append("temporal_option_not_event_permutation")
                continue
            order = tuple(index for _, index in sorted(zip(positions, range(len(positions)))))
            if order in permutations:
                errors.append("temporal_duplicate_option_order")
            permutations[order] = key
        chronological = tuple(range(len(event_texts)))
        if len(event_texts) < 3 or permutations.get(chronological) != next(
                (key for key in OPTION_KEYS if options.get(key) == gold_text), None):
            errors.append("temporal_gold_not_chronological")
    return sorted(set(errors))


def _actual_scope(
    resolved_refs: list[str], expanded_dialogue: Mapping[str, Any],
) -> str | None:
    turns = {str(row.get("turn_ref")): str(row.get("origin"))
             for row in expanded_dialogue.get("turns") or []}
    origins = {turns.get(ref) for ref in resolved_refs}
    return {
        frozenset({"source_grounded"}): "parent_only",
        frozenset({"generated_extension"}): "child_only",
        frozenset({"source_grounded", "generated_extension"}): "parent_child",
    }.get(frozenset(origins))


def validate_structural_writer_output(
    value: Any, *, expanded_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate raw v19 output without making semantic truth judgments."""
    envelope_errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {
            "structural_qa_schema_version", "abstention_summary", "questions"}:
        return {
            "envelope_valid": False,
            "envelope_errors": ["structural_qa_top_level_schema"],
            "items": [], "normalized_batch": None,
        }
    if value.get("structural_qa_schema_version") != STRUCTURAL_QA_SCHEMA_VERSION:
        envelope_errors.append("structural_qa_schema_version")
    abstention = value.get("abstention_summary")
    if (not isinstance(abstention, Mapping) or set(abstention) != {
            "estimated_skipped_count", "reason_codes"} or
            not isinstance(abstention.get("estimated_skipped_count"), int) or
            int(abstention.get("estimated_skipped_count", -1)) < 0 or
            not isinstance(abstention.get("reason_codes"), list) or
            any(reason not in ABSTENTION_REASONS
                for reason in abstention.get("reason_codes") or [])):
        envelope_errors.append("abstention_summary")
    questions = value.get("questions")
    if not isinstance(questions, list):
        envelope_errors.append("questions_not_list")
        questions = []
    elif not questions and isinstance(abstention, Mapping):
        if (int(abstention.get("estimated_skipped_count") or 0) < 1 or
                not abstention.get("reason_codes")):
            envelope_errors.append("empty_batch_without_abstention_evidence")

    items: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    normalized_statuses: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    seen_signatures: dict[tuple[Any, ...], str] = {}
    evidence_counts: list[int] = []
    for index, raw_row in enumerate(questions, 1):
        errors: list[str] = []
        warnings: list[str] = []
        row = raw_row if isinstance(raw_row, Mapping) else {}
        if not isinstance(raw_row, Mapping) or set(row) != QUESTION_KEYS:
            errors.append("candidate_schema")
        candidate_id = str(row.get("candidate_id") or "")
        expected_candidate_id = f"Q{index:02d}"
        if candidate_id != expected_candidate_id or candidate_id in seen_candidates:
            errors.append("candidate_id")
        seen_candidates.add(candidate_id)
        if index > QUESTION_CAPACITY_MAX:
            errors.append("candidate_above_capacity_max")

        question_type = str(row.get("question_type") or "")
        if question_type not in QUESTION_TYPES:
            errors.append("question_type")
        if row.get("source_scope") not in SOURCE_SCOPES:
            errors.append("source_scope")
        if not _text(row.get("question")):
            errors.append("question")
        options = row.get("options")
        if (not isinstance(options, Mapping) or set(options) != OPTION_KEYS or
                any(not _text(options.get(key)) for key in OPTION_KEYS)):
            errors.append("options")
            options = {}
        elif len({normalize_option(options[key]) for key in OPTION_KEYS}) != 4:
            errors.append("duplicate_options")
        gold_text = row.get("gold_text")
        matching_gold = [key for key in OPTION_KEYS
                         if isinstance(options, Mapping) and
                         options.get(key) == gold_text]
        if not _text(gold_text) or len(matching_gold) != 1:
            errors.append("gold_text_binding")

        evidence = row.get("evidence_turn_ids")
        resolution = resolve_evidence_turn_refs(
            evidence, expanded_dialogue=expanded_dialogue)
        raw_refs = list(resolution["raw_refs"])
        evidence_counts.append(len(raw_refs))
        if not raw_refs:
            errors.append("evidence_empty")
        if len(raw_refs) > MAX_EVIDENCE_TURNS:
            errors.append("evidence_over_limit")
        if (resolution["duplicate_raw_refs"] or
                resolution["duplicate_resolved_refs"]):
            errors.append("evidence_duplicate_refs")
        if resolution["invalid_evidence_refs"]:
            errors.append("evidence_reference_invalid")
        actual_scope = _actual_scope(
            list(resolution["resolved_turn_refs"]), expanded_dialogue)
        if actual_scope != row.get("source_scope"):
            errors.append("source_scope_mismatch")

        reasoning = row.get("reasoning_structure")
        if not isinstance(reasoning, Mapping) or set(reasoning) != COMMON_REASONING_KEYS:
            errors.append("reasoning_structure")
            reasoning = {}
        variable = str(reasoning.get("discriminative_variable") or "")
        if (variable not in DISCRIMINATIVE_VARIABLES or
                variable not in VARIABLES_BY_TYPE.get(question_type, set())):
            errors.append("discriminative_variable")
        support = reasoning.get("supporting_turn_ids")
        if (not isinstance(support, list) or not support or
                any(not _text(ref) for ref in support) or
                len(support) != len(set(support))):
            errors.append("supporting_turn_ids")
            support = []
        support_resolution = resolve_evidence_turn_refs(
            support, expanded_dialogue=expanded_dialogue)
        if support_resolution["invalid_evidence_refs"]:
            errors.append("supporting_turn_reference_invalid")
        if set(support) != set(raw_refs):
            errors.append("supporting_evidence_mismatch")
        if not _text(reasoning.get("short_rationale")):
            errors.append("short_rationale")
        errors.extend(_validate_type_contract(
            question_type, reasoning.get("type_contract"), set(support)))
        errors.extend(_validate_type_specific_answer_shape(
            question_type, reasoning.get("type_contract"), row.get("question"),
            options, gold_text))

        signature = _exact_duplicate_signature(row)
        if signature in seen_signatures:
            errors.append(
                "exact_duplicate_overflow:" + seen_signatures[signature])
        else:
            seen_signatures[signature] = candidate_id
        option_surface = assess_option_surface_symmetry(
            options, matching_gold[0] if len(matching_gold) == 1 else None)
        warnings.extend(option_surface.get("warnings") or [])
        errors = sorted(set(errors))
        valid = not errors and not envelope_errors
        status = {
            "question_index": len(normalized_rows) + 1,
            "candidate_id": candidate_id,
            "solver_ready": valid,
            "evidence_ready": valid,
            "structural_errors": errors,
            "evidence_issues": [],
            "invalid_evidence_turn_refs": list(
                resolution["invalid_evidence_refs"]),
            "valid_evidence_turn_refs": list(
                resolution["resolved_turn_refs"]),
            "actual_source_scope": actual_scope,
            "option_surface_diagnostics": option_surface,
        }
        item = {
            "candidate_index": index,
            "candidate_id": candidate_id,
            "valid": valid,
            "structural_errors": errors,
            "warnings": warnings,
            "question_type": question_type,
            "discriminative_variable": variable,
            "declared_source_scope": row.get("source_scope"),
            "actual_source_scope": actual_scope,
            "evidence_count": len(raw_refs),
        }
        items.append(item)
        diagnostics.append({
            **item,
            "reasoning_structure": reasoning,
            "evidence_turn_ids": raw_refs,
            "resolved_evidence_turn_ids": list(
                resolution["resolved_turn_refs"]),
        })
        if valid:
            normalized_rows.append({
                "question_type": question_type,
                "question": str(row["question"]).strip(),
                "options": dict(options),
                "correct_option": matching_gold[0],
                "evidence_turn_refs": raw_refs,
            })
            normalized_statuses.append(status)

    valid_scopes = [row["actual_source_scope"] for row in items if row["valid"]]
    valid_variables = [row["discriminative_variable"]
                       for row in items if row["valid"]]
    counts = [row["evidence_count"] for row in items if row["valid"]]
    child_supported = sum(scope in {"child_only", "parent_child"}
                          for scope in valid_scopes)
    materialization_map = [{
        "raw_candidate_id": row["candidate_id"],
        "materialized_candidate_id": f"Q{index:02d}",
    } for index, row in enumerate(
        (row for row in items if row["valid"]), 1)]
    return {
        "envelope_valid": not envelope_errors,
        "envelope_errors": envelope_errors,
        "question_count": len(questions),
        "structural_valid_count": len(normalized_rows),
        "structural_invalid_count": len(questions) - len(normalized_rows),
        "items": items,
        "diagnostics": diagnostics,
        "normalized_batch": {
            "mcq_batch_schema_version": "simple_append_mcq_batch_v5",
            "questions": normalized_rows,
        },
        "normalized_statuses": normalized_statuses,
        "materialization_map": materialization_map,
        "zero_question_batch_valid": not questions and not envelope_errors,
        "child_supported_count": child_supported,
        "nonempty_batch_has_child_support": (
            not normalized_rows or child_supported > 0),
        "discriminative_variable_distribution": {
            key: valid_variables.count(key) for key in sorted(set(valid_variables))},
        "question_type_distribution": {
            key: sum(row["valid"] and row["question_type"] == key
                     for row in items)
            for key in sorted({row["question_type"] for row in items
                               if row["valid"]})},
        "evidence_statistics": {
            "min": min(counts) if counts else None,
            "median": median(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": mean(counts) if counts else None,
            "over_8_count": sum(count > 8 for count in evidence_counts),
            "over_6_valid_count": sum(count > 6 for count in counts),
        },
    }


def solver_partition(validation: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt only structurally valid v19 items to the frozen Solver runner."""
    return {
        "envelope_valid": bool(validation.get("envelope_valid")),
        "items": list(validation.get("normalized_statuses") or []),
        "solver_ready_count": int(validation.get("structural_valid_count") or 0),
        "focus_round": {"items": []},
    }
