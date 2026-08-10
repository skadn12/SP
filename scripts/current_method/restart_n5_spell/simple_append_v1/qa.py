"""Local batch-MCQ contract over the complete Parent+Child fact source."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from ..core import classify_frontier_pre_audit, normalize_option
from ..minimal_admission import SPELL_STYLE_MINIMAL_MODE
from ..questioner_contract import span_catalog_sha256


MCQ_BATCH_SCHEMA_VERSION = "simple_append_mcq_batch_v5"
MCQ_FOCUS_WRITER_SCHEMA_VERSION = "simple_append_mcq_focus_writer_v4"
EVIDENCE_ALIAS_SCHEMA_VERSION = "simple_append_evidence_alias_v1"
MIN_FOCUS_ROUND_SUPPORTED = 4
MIN_FOCUS_ROUND_JOINT_WITH_HISTORY = 2
FULL_CONTEXT_FRONTIER_K = 5
_OPTION_KEYS = {"A", "B", "C", "D"}
QUESTION_TYPES = {
    "causal",
    "temporal_sequence",
    "state_change",
    "cross_turn_comparison",
    "relationship_or_decision",
    "cross_context_synthesis",
    "information_visibility",
    "information_revision",
    "task_dependency_or_constraint",
}
_QUESTION_KEYS = {
    "question_type", "question", "options", "correct_option",
    "evidence_turn_refs",
}
_FOCUS_WRITER_KEYS = {
    "mcq_focus_writer_schema_version",
    "questions",
}
_COVERAGE_ROLES = {
    "current_round_only",
    "current_round_joint_history",
    "parent_child",
    "general",
}
_OPTION_GIVEAWAY_PATTERN = re.compile(
    r"\b(always|never|obviously|merely|only|unrelated|completely|entirely)\b",
    re.IGNORECASE,
)


def _normalized_question(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def assess_option_surface_symmetry(
    options: Mapping[str, Any], correct_option: Any,
) -> dict[str, Any]:
    """Record cheap option-design warnings without replacing Solver filtering."""
    if (set(options) != _OPTION_KEYS or correct_option not in _OPTION_KEYS or
            any(not _text(options.get(key)) for key in _OPTION_KEYS)):
        return {"warnings": ["option_surface_schema_invalid"],
                "blocking": False}
    word_counts = {
        key: len(re.findall(r"\b\w+\b", str(options[key])))
        for key in sorted(_OPTION_KEYS)
    }
    qualifier_hits = {
        key: _OPTION_GIVEAWAY_PATTERN.findall(str(options[key]))
        for key in sorted(_OPTION_KEYS)
    }
    warnings: list[str] = []
    lengths = list(word_counts.values())
    if max(lengths) - min(lengths) > max(6, int(min(lengths) * 0.6)):
        warnings.append("option_length_imbalance")
    correct_length = word_counts[str(correct_option)]
    if (correct_length == max(lengths) and lengths.count(correct_length) == 1 and
            correct_length - min(lengths) >= 4):
        warnings.append("gold_uniquely_longest")
    options_with_qualifier = sum(bool(value) for value in qualifier_hits.values())
    if 0 < options_with_qualifier < 4:
        warnings.append("asymmetric_giveaway_qualifier")
    return {
        "warnings": warnings,
        "word_counts": word_counts,
        "giveaway_qualifier_hits": qualifier_hits,
        "blocking": False,
    }


def build_evidence_alias_registry(
    expanded_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Build compact Writer-facing refs without changing authoritative IDs."""
    turns = expanded_dialogue.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("simple_append_evidence_alias_turns_required")
    entries: list[dict[str, Any]] = []
    alias_to_turn_ref: dict[str, str] = {}
    turn_ref_to_alias: dict[str, str] = {}
    for index, row in enumerate(turns, 1):
        if not isinstance(row, Mapping):
            raise ValueError("simple_append_evidence_alias_turn_invalid")
        turn_ref = str(row.get("turn_ref") or "")
        origin = str(row.get("origin") or "")
        if (not turn_ref or turn_ref in turn_ref_to_alias or
                origin not in {"source_grounded", "generated_extension"}):
            raise ValueError("simple_append_evidence_alias_turn_invalid")
        alias = f"D{index:03d}"
        alias_to_turn_ref[alias] = turn_ref
        turn_ref_to_alias[turn_ref] = alias
        entries.append({
            "evidence_ref": alias,
            "turn_ref": turn_ref,
            "origin": origin,
            "scene_ref": row.get("scene_ref"),
        })
    fingerprint = hashlib.sha256(json.dumps(
        entries, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "evidence_alias_schema_version": EVIDENCE_ALIAS_SCHEMA_VERSION,
        "turn_count": len(entries),
        "parent_count": sum(row["origin"] == "source_grounded"
                            for row in entries),
        "child_count": sum(row["origin"] == "generated_extension"
                           for row in entries),
        "entries": entries,
        "alias_to_turn_ref": alias_to_turn_ref,
        "turn_ref_to_alias": turn_ref_to_alias,
        "registry_sha256": fingerprint,
    }


def resolve_evidence_turn_refs(
    refs: Any, *, expanded_dialogue: Mapping[str, Any],
    allow_legacy_canonical_refs: bool = True,
) -> dict[str, Any]:
    """Resolve exact aliases; never guess, edit, or fuzzily match a reference."""
    registry = build_evidence_alias_registry(expanded_dialogue)
    aliases = registry["alias_to_turn_ref"]
    canonical = registry["turn_ref_to_alias"]
    raw_refs = ([str(ref) for ref in refs]
                if isinstance(refs, list) else [])
    resolved: list[str] = []
    valid_aliases: list[str] = []
    legacy_refs: list[str] = []
    invalid: list[str] = []
    for ref in raw_refs:
        if ref in aliases:
            resolved.append(str(aliases[ref]))
            valid_aliases.append(ref)
        elif allow_legacy_canonical_refs and ref in canonical:
            resolved.append(ref)
            valid_aliases.append(str(canonical[ref]))
            legacy_refs.append(ref)
        else:
            invalid.append(ref)
    return {
        "raw_refs": raw_refs,
        "resolved_turn_refs": list(dict.fromkeys(resolved)),
        "valid_evidence_aliases": list(dict.fromkeys(valid_aliases)),
        "invalid_evidence_refs": sorted(set(invalid)),
        "legacy_canonical_refs": list(dict.fromkeys(legacy_refs)),
        "duplicate_raw_refs": len(raw_refs) != len(set(raw_refs)),
        "duplicate_resolved_refs": len(resolved) != len(set(resolved)),
        "registry_sha256": registry["registry_sha256"],
    }


def normalize_focus_writer_batch(
    value: Any, *, minimum_current_round_only: int = 2,
    minimum_current_round_joint_history: int = 2,
    minimum_parent_child: int = 2,
    minimum_general: int = 2,
    maximum_current_round_only: int | None = None,
    maximum_current_round_joint_history: int | None = None,
    maximum_parent_child: int | None = None,
    maximum_general: int | None = None,
) -> dict[str, Any]:
    """Validate the Writer's four coverage roles and flatten them locally.

    ``coverage_role`` is only a generation aid.  The later focus evidence gate
    independently verifies actual resolved references and never trusts the
    model's declared role.
    """
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _FOCUS_WRITER_KEYS:
        return {
            "valid": False,
            "errors": ["mcq_focus_writer_top_level_schema"],
            "current_round_only_question_count": 0,
            "current_round_joint_history_question_count": 0,
            "parent_child_question_count": 0,
            "general_question_count": 0,
            "pool_assignments": [],
            "batch": None,
        }
    if (value.get("mcq_focus_writer_schema_version") !=
            MCQ_FOCUS_WRITER_SCHEMA_VERSION):
        errors.append("mcq_focus_writer_schema_version")
    questions = value.get("questions")
    if not isinstance(questions, list):
        errors.append("mcq_focus_writer_questions")
        questions = []
    pools: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(questions, 1):
        expected_keys = _QUESTION_KEYS | {"coverage_role"}
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            errors.append(f"question_{index:02d}:mcq_focus_writer_schema")
            continue
        role = str(row.get("coverage_role") or "")
        if role not in _COVERAGE_ROLES:
            errors.append(f"question_{index:02d}:coverage_role")
            continue
        pools.append((role, {key: row[key] for key in _QUESTION_KEYS}))
    current_only = [row for role, row in pools
                    if role == "current_round_only"]
    current_joint = [row for role, row in pools
                     if role == "current_round_joint_history"]
    parent_child = [row for role, row in pools if role == "parent_child"]
    general = [row for role, row in pools if role == "general"]
    if len(current_only) < minimum_current_round_only:
        errors.append("mcq_focus_writer_current_round_only_below_minimum")
    if (maximum_current_round_only is not None and
            len(current_only) > maximum_current_round_only):
        errors.append("mcq_focus_writer_current_round_only_above_maximum")
    if len(current_joint) < minimum_current_round_joint_history:
        errors.append(
            "mcq_focus_writer_current_round_joint_history_below_minimum")
    if (maximum_current_round_joint_history is not None and
            len(current_joint) > maximum_current_round_joint_history):
        errors.append(
            "mcq_focus_writer_current_round_joint_history_above_maximum")
    if len(parent_child) < minimum_parent_child:
        errors.append("mcq_focus_writer_parent_child_below_minimum")
    if (maximum_parent_child is not None and
            len(parent_child) > maximum_parent_child):
        errors.append("mcq_focus_writer_parent_child_above_maximum")
    if len(general) < minimum_general:
        errors.append("mcq_focus_writer_general_below_minimum")
    if maximum_general is not None and len(general) > maximum_general:
        errors.append("mcq_focus_writer_general_above_maximum")
    flattened = [row for _, row in pools]
    pool_assignments = [{
        "candidate_id": f"Q{index:02d}",
        "writer_pool": pool,
    } for index, (pool, _) in enumerate(pools, 1)]
    return {
        "valid": not errors,
        "errors": errors,
        "current_round_only_question_count": len(current_only),
        "current_round_joint_history_question_count": len(current_joint),
        "parent_child_question_count": len(parent_child),
        "general_question_count": len(general),
        "question_count": len(flattened),
        "pool_assignments": pool_assignments,
        "batch": {
            "mcq_batch_schema_version": MCQ_BATCH_SCHEMA_VERSION,
            "questions": flattened,
        } if not errors else None,
    }


def retain_focus_writer_overflow(
    value: Any, *, target_count: int = 8, maximum_count: int = 12,
) -> dict[str, Any]:
    """Audit surplus Writer candidates without discarding paid output."""
    if not isinstance(value, Mapping) or not isinstance(value.get("questions"), list):
        return {"applied": False, "reason": "writer_output_schema_invalid",
                "value": value, "overflow_questions": []}
    questions = list(value["questions"])
    if len(questions) <= maximum_count:
        return {"applied": False, "reason": "within_maximum",
                "value": value, "overflow_questions": []}
    return {
        "applied": False,
        "reason": "surplus_retained_for_solver",
        "value": value,
        "original_question_count": len(questions),
        "retained_question_count": len(questions),
        "preferred_maximum_count": maximum_count,
        "target_count": target_count,
        "overflow_question_count": len(questions) - maximum_count,
        "overflow_questions": questions[maximum_count:],
        "selection_policy": "retain_all_writer_candidates_for_solver",
    }


def cap_focus_writer_overflow(
    value: Any, *, target_count: int = 8, maximum_count: int = 12,
) -> dict[str, Any]:
    """Compatibility alias for the former cap; surplus is no longer dropped."""
    return retain_focus_writer_overflow(
        value, target_count=target_count, maximum_count=maximum_count)


def validate_mcq_batch(
    value: Any, *, expanded_dialogue: Mapping[str, Any],
    expected_count: int | None = None,
    minimum_count: int | None = None,
    maximum_count: int | None = None,
    min_child_supported: int = 1,
    min_joint_parent_child: int = 0,
    min_distinct_question_types: int = 0,
) -> dict[str, Any]:
    errors: list[str] = []
    if (not isinstance(value, Mapping) or
            set(value) != {"mcq_batch_schema_version", "questions"}):
        return {"valid": False, "errors": ["mcq_batch_top_level_schema"]}
    if value.get("mcq_batch_schema_version") != MCQ_BATCH_SCHEMA_VERSION:
        errors.append("mcq_batch_schema_version")
    questions = value.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("mcq_batch_questions")
        questions = []
    if expected_count is not None and len(questions) != expected_count:
        errors.append("mcq_batch_question_count")
    if minimum_count is not None and len(questions) < minimum_count:
        errors.append("mcq_batch_question_count_below_minimum")
    if maximum_count is not None and len(questions) > maximum_count:
        errors.append("mcq_batch_question_count_above_maximum")

    turns = {str(row.get("turn_ref") or ""): row
             for row in expanded_dialogue.get("turns") or []}
    child_question_count = 0
    joint_question_count = 0
    parent_only_question_count = 0
    question_types: list[str] = []
    source_scopes: list[str] = []
    seen_questions: dict[str, int] = {}
    for index, row in enumerate(questions, 1):
        prefix = f"question_{index:02d}"
        if not isinstance(row, Mapping) or set(row) != _QUESTION_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        normalized_question = _normalized_question(row.get("question"))
        if normalized_question and normalized_question in seen_questions:
            errors.append(
                f"{prefix}:duplicate_question:question_"
                f"{seen_questions[normalized_question]:02d}")
        elif normalized_question:
            seen_questions[normalized_question] = index
        question_type = str(row.get("question_type") or "")
        if question_type not in QUESTION_TYPES:
            errors.append(f"{prefix}:question_type")
        else:
            question_types.append(question_type)
        if not _text(row.get("question")):
            errors.append(f"{prefix}:question")
        options = row.get("options")
        if (not isinstance(options, Mapping) or set(options) != _OPTION_KEYS or
                any(not _text(options.get(key)) for key in _OPTION_KEYS)):
            errors.append(f"{prefix}:options")
        else:
            normalized = [" ".join(str(options[key]).split()).casefold()
                          for key in sorted(_OPTION_KEYS)]
            if len(normalized) != len(set(normalized)):
                errors.append(f"{prefix}:duplicate_options")
        if row.get("correct_option") not in _OPTION_KEYS:
            errors.append(f"{prefix}:correct_option")
        evidence = row.get("evidence_turn_refs")
        resolution = resolve_evidence_turn_refs(
            evidence, expanded_dialogue=expanded_dialogue)
        if (not isinstance(evidence, list) or not evidence or
                any(not _text(ref) for ref in evidence) or
                resolution["invalid_evidence_refs"] or
                resolution["duplicate_raw_refs"] or
                resolution["duplicate_resolved_refs"]):
            errors.append(f"{prefix}:evidence_turn_refs")
            continue
        origins = {str(turns[ref].get("origin"))
                   for ref in resolution["resolved_turn_refs"]}
        source_scope = {
            frozenset({"source_grounded"}): "parent_only",
            frozenset({"generated_extension"}): "child_only",
            frozenset({"source_grounded", "generated_extension"}):
                "parent_child",
        }.get(frozenset(origins))
        if source_scope is None:
            errors.append(f"{prefix}:evidence_source_scope")
            continue
        source_scopes.append(source_scope)
        if "generated_extension" in origins:
            child_question_count += 1
        if origins == {"source_grounded", "generated_extension"}:
            joint_question_count += 1
        if origins == {"source_grounded"}:
            parent_only_question_count += 1
    if questions and child_question_count < min_child_supported:
        errors.append("mcq_batch_missing_child_supported_question")
    if questions and joint_question_count < min_joint_parent_child:
        errors.append("mcq_batch_insufficient_joint_parent_child_questions")
    return {
        "valid": not errors,
        "errors": errors,
        "question_count": len(questions),
        "child_supported_question_count": child_question_count,
        "joint_parent_child_question_count": joint_question_count,
        "parent_only_question_count": parent_only_question_count,
        "distinct_question_types": sorted(set(question_types)),
        "question_type_count": len(set(question_types)),
        "source_scopes": source_scopes,
        "source_scope_counts": {
            scope: source_scopes.count(scope)
            for scope in ("parent_only", "child_only", "parent_child")},
        "question_type_diversity_target_met": (
            len(set(question_types)) >= min_distinct_question_types),
        "required_min_child_supported": min_child_supported,
        "required_min_joint_parent_child": min_joint_parent_child,
        "required_min_distinct_question_types": min_distinct_question_types,
        "every_question_must_use_child": False,
    }


def partition_mcq_candidates(
    value: Any, *, expanded_dialogue: Mapping[str, Any],
    minimum_count: int = 8, maximum_count: int | None = None,
    min_child_supported: int = 4, min_joint_parent_child: int = 2,
) -> dict[str, Any]:
    """Separate batch-fatal envelope errors from candidate-local defects.

    Evidence defects are annotations, not Solver blockers. A malformed question,
    option set, answer binding, type, or duplicate question remains ineligible.
    """
    envelope_errors: list[str] = []
    if (not isinstance(value, Mapping) or
            set(value) != {"mcq_batch_schema_version", "questions"}):
        return {"envelope_valid": False,
                "envelope_errors": ["mcq_batch_top_level_schema"],
                "items": [], "solver_ready_count": 0,
                "evidence_ready_count": 0}
    if value.get("mcq_batch_schema_version") != MCQ_BATCH_SCHEMA_VERSION:
        envelope_errors.append("mcq_batch_schema_version")
    questions = value.get("questions")
    if not isinstance(questions, list) or not questions:
        envelope_errors.append("mcq_batch_questions")
        questions = []
    if len(questions) < minimum_count:
        envelope_errors.append("mcq_batch_question_count_below_minimum")
    if maximum_count is not None and len(questions) > maximum_count:
        envelope_errors.append("mcq_batch_question_count_above_maximum")

    turns = {str(row.get("turn_ref") or ""): row
             for row in expanded_dialogue.get("turns") or []}
    seen_questions: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    for index, row in enumerate(questions, 1):
        structural_errors: list[str] = []
        evidence_issues: list[str] = []
        candidate_id = f"Q{index:02d}"
        if not isinstance(row, Mapping) or set(row) != _QUESTION_KEYS:
            structural_errors.append("candidate_schema")
            row = row if isinstance(row, Mapping) else {}
        else:
            question_type = str(row.get("question_type") or "")
            if question_type not in QUESTION_TYPES:
                structural_errors.append("question_type")
            question = row.get("question")
            if not _text(question):
                structural_errors.append("question")
            normalized_question = _normalized_question(question)
            if normalized_question and normalized_question in seen_questions:
                structural_errors.append(
                    f"duplicate_question:question_"
                    f"{seen_questions[normalized_question]:02d}")
            elif normalized_question:
                seen_questions[normalized_question] = index
            options = row.get("options")
            if (not isinstance(options, Mapping) or set(options) != _OPTION_KEYS or
                    any(not _text(options.get(key)) for key in _OPTION_KEYS)):
                structural_errors.append("options")
            else:
                normalized_options = [normalize_option(options[key])
                                      for key in sorted(_OPTION_KEYS)]
                if len(set(normalized_options)) != 4:
                    structural_errors.append("duplicate_options")
            if row.get("correct_option") not in _OPTION_KEYS:
                structural_errors.append("correct_option")

        evidence = row.get("evidence_turn_refs")
        resolution = resolve_evidence_turn_refs(
            evidence, expanded_dialogue=expanded_dialogue)
        evidence_refs = resolution["raw_refs"]
        if not evidence_refs:
            evidence_issues.append("evidence_empty_or_not_list")
        if (resolution["duplicate_raw_refs"] or
                resolution["duplicate_resolved_refs"]):
            evidence_issues.append("evidence_duplicate_refs")
        invalid_refs = resolution["invalid_evidence_refs"]
        if invalid_refs:
            evidence_issues.append("evidence_reference_invalid")
        if len(evidence_refs) > 4:
            evidence_issues.append("evidence_too_many")
        valid_refs = resolution["resolved_turn_refs"]
        origins = {str(turns[ref].get("origin")) for ref in valid_refs}
        actual_scope = {
            frozenset({"source_grounded"}): "parent_only",
            frozenset({"generated_extension"}): "child_only",
            frozenset({"source_grounded", "generated_extension"}):
                "parent_child",
        }.get(frozenset(origins))
        solver_ready = not structural_errors
        evidence_ready = solver_ready and not evidence_issues
        option_surface = (
            assess_option_surface_symmetry(
                row.get("options") or {}, row.get("correct_option"))
            if isinstance(row, Mapping) else
            {"warnings": ["option_surface_schema_invalid"],
             "blocking": False})
        items.append({
            "question_index": index,
            "candidate_id": candidate_id,
            "structural_errors": structural_errors,
            "evidence_issues": evidence_issues,
            "invalid_evidence_turn_refs": invalid_refs,
            "valid_evidence_aliases": resolution["valid_evidence_aliases"],
            "valid_evidence_turn_refs": valid_refs,
            "legacy_canonical_evidence_refs":
                resolution["legacy_canonical_refs"],
            "evidence_alias_registry_sha256": resolution["registry_sha256"],
            "actual_source_scope": actual_scope,
            "solver_ready": solver_ready,
            "evidence_ready": evidence_ready,
            "option_surface_diagnostics": option_surface,
        })

    child_supported = sum(
        row.get("actual_source_scope") in {"child_only", "parent_child"}
        for row in items)
    joint = sum(row.get("actual_source_scope") == "parent_child"
                for row in items)
    batch_profile_errors: list[str] = []
    if child_supported < min_child_supported:
        batch_profile_errors.append("child_supported_below_minimum")
    if joint < min_joint_parent_child:
        batch_profile_errors.append("parent_child_below_minimum")
    return {
        "envelope_valid": not envelope_errors,
        "envelope_errors": envelope_errors,
        "items": items,
        "batch_profile_ready": not batch_profile_errors,
        "batch_profile_errors": batch_profile_errors,
        "child_supported_count": child_supported,
        "parent_child_count": joint,
        "distinct_question_types": sorted({
            str(row.get("question_type")) for row in questions
            if isinstance(row, Mapping) and
            str(row.get("question_type")) in QUESTION_TYPES}),
        "solver_ready_count": sum(bool(row["solver_ready"]) for row in items),
        "evidence_ready_count": sum(bool(row["evidence_ready"]) for row in items),
        "evidence_issue_count": sum(bool(row["evidence_issues"]) for row in items),
    }


def evaluate_focus_round_profile(
    partition: Mapping[str, Any], *, expanded_dialogue: Mapping[str, Any],
    focus_round_index: int,
    minimum_supported: int = MIN_FOCUS_ROUND_SUPPORTED,
    minimum_joint_with_history: int =
        MIN_FOCUS_ROUND_JOINT_WITH_HISTORY,
    pool_assignments: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Enforce batch-level use of the newly generated round.

    Only structurally Solver-ready questions and exact, successfully resolved
    evidence references count.  Evidence from an older generated round remains
    Child evidence, but cannot satisfy the current-round quota.
    """
    if focus_round_index < 1:
        raise ValueError("simple_append_focus_round_index_invalid")
    if minimum_supported < 1 or minimum_joint_with_history < 0:
        raise ValueError("simple_append_focus_round_quota_invalid")
    if minimum_joint_with_history > minimum_supported:
        raise ValueError("simple_append_focus_round_quota_invalid")

    turns = {str(row.get("turn_ref") or ""): row
             for row in expanded_dialogue.get("turns") or []}
    prefix = f"R{focus_round_index:02d}:"
    if not any(str(row.get("scene_ref") or "").startswith(prefix)
               for row in turns.values()):
        raise ValueError("simple_append_focus_round_missing")

    pool_by_candidate = {
        str(row.get("candidate_id")): str(row.get("writer_pool") or "")
        for row in (pool_assignments or [])}
    rows: list[dict[str, Any]] = []
    coverage_role_warnings: list[str] = []
    for item in partition.get("items") or []:
        refs = list(item.get("valid_evidence_turn_refs") or [])
        solver_ready = bool(item.get("solver_ready"))
        focus = [ref for ref in refs
                 if str((turns.get(str(ref)) or {}).get(
                     "scene_ref") or "").startswith(prefix)]
        earlier = [ref for ref in refs if ref not in focus]
        parent = [ref for ref in refs
                  if str((turns.get(str(ref)) or {}).get("origin") or "") ==
                  "source_grounded"]
        child = [ref for ref in refs
                 if str((turns.get(str(ref)) or {}).get("origin") or "") ==
                 "generated_extension"]
        supported = bool(solver_ready and focus)
        joint = bool(supported and earlier)
        candidate_id = str(item.get("candidate_id") or "")
        writer_pool = pool_by_candidate.get(candidate_id)
        if writer_pool == "current_round_only" and not supported:
            coverage_role_warnings.append(
                f"{candidate_id}:current_round_only_missing_focus_evidence")
        if writer_pool == "current_round_only" and supported and earlier:
            coverage_role_warnings.append(
                f"{candidate_id}:current_round_only_has_earlier_evidence")
        if writer_pool == "current_round_joint_history" and not supported:
            coverage_role_warnings.append(
                f"{candidate_id}:current_round_joint_history_missing_focus_side")
        elif writer_pool == "current_round_joint_history" and not earlier:
            coverage_role_warnings.append(
                f"{candidate_id}:current_round_joint_history_missing_history_side")
        if writer_pool == "parent_child" and not (
                solver_ready and parent and child):
            coverage_role_warnings.append(
                f"{candidate_id}:parent_child_missing_two_sided_evidence")
        candidate_warnings = [
            warning for warning in coverage_role_warnings
            if warning.startswith(candidate_id + ":")]
        rows.append({
            "candidate_id": item.get("candidate_id"),
            "writer_pool": writer_pool,
            "solver_ready": solver_ready,
            "focus_round_supported": supported,
            "focus_round_joint_with_history": joint,
            "focus_round_evidence_turn_refs": focus if solver_ready else [],
            "earlier_evidence_turn_refs": earlier if solver_ready else [],
            "parent_evidence_turn_refs": parent if solver_ready else [],
            "child_evidence_turn_refs": child if solver_ready else [],
            "coverage_role_evidence_match": not candidate_warnings,
            "coverage_role_warnings": candidate_warnings,
        })

    supported_count = sum(bool(row["focus_round_supported"]) for row in rows)
    joint_count = sum(bool(row["focus_round_joint_with_history"]) for row in rows)
    errors: list[str] = []
    if supported_count < minimum_supported:
        errors.append("current_round_supported_below_minimum")
    if joint_count < minimum_joint_with_history:
        errors.append("current_round_joint_with_history_below_minimum")
    return {
        "focus_round_index": focus_round_index,
        "items": rows,
        "focus_round_supported_count": supported_count,
        "focus_round_joint_with_history_count": joint_count,
        "required_minimum_supported": minimum_supported,
        "required_minimum_joint_with_history": minimum_joint_with_history,
        "focus_round_profile_ready": not errors,
        "focus_round_profile_errors": errors,
        "coverage_role_warning_count": len(coverage_role_warnings),
        "coverage_role_warnings": coverage_role_warnings,
        # Compatibility name retained for historical reports. These are
        # annotations and never participate in Solver admission.
        "writer_pool_contract_errors": coverage_role_warnings,
    }


def evaluate_qa_quota_status(
    partition: Mapping[str, Any], focus_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Treat source-coverage quotas as top-up targets, not Solver blockers."""
    deficits = list(dict.fromkeys(
        list(partition.get("batch_profile_errors") or []) +
        list(focus_profile.get("focus_round_profile_errors") or [])))
    return {
        "quota_profile_ready": not deficits,
        "top_up_required": bool(deficits),
        "quota_deficits": deficits,
        "existing_solver_candidates_preserved": True,
        "quota_deficits_block_solver": False,
    }


def materialize_solver_candidate(
    row: Mapping[str, Any], *, index: int,
    expanded_dialogue: Mapping[str, Any], span_catalog: Mapping[str, Any],
    candidate_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize one structurally valid candidate without repairing evidence."""
    if not candidate_status.get("solver_ready"):
        raise ValueError("simple_append_candidate_not_solver_ready")
    target = assigned_answer_letters(
        str(expanded_dialogue["root_id"]), index)[-1]
    original_options = dict(row["options"])
    model_gold = str(row["correct_option"])
    correct_text = str(original_options[model_gold])
    distractors = [str(original_options[letter]) for letter in "ABCD"
                   if letter != model_gold]
    options: dict[str, str] = {}
    cursor = 0
    for letter in "ABCD":
        if letter == target:
            options[letter] = correct_text
        else:
            options[letter] = distractors[cursor]
            cursor += 1
    valid_evidence = list(candidate_status.get("valid_evidence_turn_refs") or [])
    return {
        "qa_id": f"{expanded_dialogue['root_id']}-simple-append-mcq-{index:02d}",
        "candidate_id": f"Q{index:02d}",
        "question_type": str(row["question_type"]),
        "source_scope": candidate_status.get("actual_source_scope"),
        "question": str(row["question"]).strip(),
        "options": options,
        "reference_answer_letter": target,
        "reference_answer_text": correct_text,
        "canonical_answer_text": correct_text,
        "evidence_span_ids": [f"{ref}#S0001" for ref in valid_evidence],
        "evidence_turn_ids": valid_evidence,
        "invalid_evidence_turn_ids": list(
            candidate_status.get("invalid_evidence_turn_refs") or []),
        "evidence_issues": list(candidate_status.get("evidence_issues") or []),
        "option_surface_diagnostics": dict(
            candidate_status.get("option_surface_diagnostics") or {}),
        "evidence_ready": bool(candidate_status.get("evidence_ready")),
        "difficulty_level": "unclassified",
        "admission_mode": SPELL_STYLE_MINIMAL_MODE,
        "span_catalog_sha256": span_catalog_sha256(dict(span_catalog)),
        "local_option_binding": True,
        "writer_correct_option_ignored_after_text_binding": True,
    }


def assigned_answer_letters(root_id: str, count: int) -> list[str]:
    """Rotate answer positions deterministically; the writer does not own placement."""
    offset = int(hashlib.sha256(str(root_id).encode()).hexdigest()[:8], 16) % 4
    return ["ABCD"[(offset + index) % 4] for index in range(count)]


def materialize_solver_batch(
    value: Mapping[str, Any], *, expanded_dialogue: Mapping[str, Any],
    span_catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind validated MCQs to program-owned answer positions and evidence spans."""
    validation = validate_mcq_batch(value, expanded_dialogue=expanded_dialogue)
    if not validation["valid"]:
        raise ValueError("simple_append_mcq_batch_invalid:" +
                         ",".join(validation["errors"]))
    questions = list(value["questions"])
    letters = assigned_answer_letters(
        str(expanded_dialogue["root_id"]), len(questions))
    materialized: list[dict[str, Any]] = []
    for index, (row, target) in enumerate(zip(questions, letters), 1):
        original_options = dict(row["options"])
        model_gold = str(row["correct_option"])
        correct_text = str(original_options[model_gold])
        distractors = [str(original_options[letter]) for letter in "ABCD"
                       if letter != model_gold]
        options: dict[str, str] = {}
        cursor = 0
        for letter in "ABCD":
            if letter == target:
                options[letter] = correct_text
            else:
                options[letter] = distractors[cursor]
                cursor += 1
        resolution = resolve_evidence_turn_refs(
            row["evidence_turn_refs"], expanded_dialogue=expanded_dialogue)
        evidence_turn_ids = resolution["resolved_turn_refs"]
        turn_origins = {
            str(turn["turn_ref"]): str(turn["origin"])
            for turn in expanded_dialogue.get("turns") or []}
        origins = {turn_origins[ref] for ref in evidence_turn_ids}
        source_scope = {
            frozenset({"source_grounded"}): "parent_only",
            frozenset({"generated_extension"}): "child_only",
            frozenset({"source_grounded", "generated_extension"}):
                "parent_child",
        }[frozenset(origins)]
        materialized.append({
            "qa_id": f"{expanded_dialogue['root_id']}-simple-append-mcq-{index:02d}",
            "question_type": str(row["question_type"]),
            "source_scope": source_scope,
            "question": str(row["question"]).strip(),
            "options": options,
            "reference_answer_letter": target,
            "reference_answer_text": correct_text,
            "canonical_answer_text": correct_text,
            "evidence_span_ids": [f"{ref}#S0001" for ref in evidence_turn_ids],
            "evidence_turn_ids": evidence_turn_ids,
            "difficulty_level": "unclassified",
            "admission_mode": SPELL_STYLE_MINIMAL_MODE,
            "span_catalog_sha256": span_catalog_sha256(dict(span_catalog)),
            "local_option_binding": True,
            "writer_correct_option_ignored_after_text_binding": True,
        })
    return materialized


def classify_balanced_no_context(
    orders: list[Mapping[str, Any]], *, samples_per_position: int,
) -> dict[str, Any]:
    """Reject only all-correct leakage or a dominant fixed answer position."""
    if (samples_per_position not in (1, 2) or len(orders) != 4 or
            {str(row.get("gold_letter")) for row in orders} != set("ABCD")):
        return {"decision": "bad_no_context_eval", "reason": "four_positions_required"}
    answers: list[str] = []
    rewards: list[int] = []
    summaries: list[dict[str, Any]] = []
    for row in orders:
        local_answers = list(row.get("valid_answer_letters") or [])
        local_rewards = list(row.get("valid_rewards") or [])
        if (len(local_answers) != samples_per_position or
                len(local_rewards) != samples_per_position or
                any(answer not in "ABCD" for answer in local_answers) or
                any(reward not in (0, 1) for reward in local_rewards)):
            return {"decision": "bad_no_context_eval", "reason": "valid_samples_required"}
        answers.extend(local_answers)
        rewards.extend(local_rewards)
        summaries.append({
            "order_id": row.get("order_id"),
            "gold_letter": row.get("gold_letter"),
            "valid_answer_letters": local_answers,
            "valid_rewards": local_rewards,
        })
    counts = {letter: answers.count(letter) for letter in "ABCD"}
    total = len(rewards)
    correct = sum(rewards)
    dominant_count = max(counts.values())
    if correct == total:
        decision = "bad_ground_semantic_leakage"
    elif len(set(answers)) == 1 or dominant_count >= (3 if total == 4 else 6):
        decision = "no_context_position_bias"
    else:
        decision = "pass_grounding"
    return {
        "decision": decision,
        "grounding_protocol": (
            f"balanced_four_position_k{samples_per_position}_any_error_pass_v1"),
        "orders": summaries,
        "semantic_correct_count": correct,
        "total_valid_rollouts": total,
        "semantic_correct_rate": correct / total,
        "answer_letter_counts": counts,
        "dominant_answer_letter_count": dominant_count,
    }


def classify_full_context_frontier(rewards: list[Any]) -> dict[str, Any]:
    """Apply the frozen SPELL K=5 frontier gate to full-context rewards."""
    valid = (
        isinstance(rewards, list) and
        len(rewards) == FULL_CONTEXT_FRONTIER_K and
        all(type(reward) is int and reward in (0, 1) for reward in rewards)
    )
    if not valid:
        return {
            "decision": "reject_responder_format",
            "admitted": False,
            "k_eval": FULL_CONTEXT_FRONTIER_K,
            "exact_correct_count": None,
            "bucket": None,
            "verified_rewards": list(rewards) if isinstance(rewards, list) else [],
        }
    result = classify_frontier_pre_audit(rewards)
    correct = sum(rewards)
    return {
        **result,
        "admitted": result["decision"] == "retain_frontier_pre_audit",
        "k_eval": FULL_CONTEXT_FRONTIER_K,
        "exact_correct_count": correct,
        "bucket": f"{correct}/{FULL_CONTEXT_FRONTIER_K}",
        "verified_rewards": list(rewards),
    }


def permuted_options(options: Mapping[str, str], shift: int) -> dict[str, str]:
    letters = "ABCD"
    values = [str(options[letter]) for letter in letters]
    return {letter: values[(index - shift) % 4]
            for index, letter in enumerate(letters)}


def gold_letter_for_options(options: Mapping[str, str], gold_text: str) -> str:
    matches = [letter for letter, value in options.items()
               if normalize_option(value) == normalize_option(gold_text)]
    if len(matches) != 1:
        raise ValueError("simple_append_gold_option_binding_invalid")
    return matches[0]
