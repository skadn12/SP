"""Compact, semantic-preserving repair for Round Validator model contracts.

The repair is deliberately narrow: it can add a missing continuation evidence
member, re-submit invalid durable-update evidence as a frozen candidate subset,
or explicitly re-home an otherwise exact top-level payload that the model put
inside ``scene_reviews``.  It never changes a review decision, status, reason,
update identity, or any other semantic field.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes
from .relation_registry import RELATION_TYPE_REGISTRY
from .round_validator_contract import (
    ACTION_REALIZATION_STATUSES, MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS,
    CONNECTION_STATUSES, CONSISTENCY_STATUSES, CONTINUITY_STATUSES,
    CONNECTED_PERSON_DURABLE_UPDATE_FIELDS, DURABLE_UPDATE_FIELDS,
    DEVELOPMENT_STATUSES, DUPLICATE_STATUSES,
    MODEL_NEW_ACTION_REALIZATION_FIELDS, MODEL_SCENE_REVIEW_FIELDS,
    SCENE_DECISIONS, SCENE_REASON_CODES, VERIFICATION_STATUSES,
    VERIFIED_CONNECTION_FIELDS,
)


ROUND_VALIDATOR_CONTRACT_REPAIR_VERSION_V11 = (
    "round_validator_contract_only_repair_v11_v24_indexed_historical_selection"
)
ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11 = (
    "round_validator_contract_only_repair_v11_prompt_v24_indexed_historical_selection"
)
REPAIR_TOP_LEVEL_FIELDS = frozenset({"repairs"})
_GROUPED_FIXTURE_TOP_LEVEL_FIELDS = frozenset({
    "top_level_rehome", "scene_reference_repairs", "action_realization_repairs",
    "durable_update_evidence_repairs", "observation_reference_repairs",
})
TOP_LEVEL_REHOME_FIELDS = frozenset({
    "misplaced_scene_review_index", "durable_updates",
    "semantic_review_observations",
})
SPLIT_TOP_LEVEL_REHOME_FIELDS = frozenset({
    "misplaced_scene_review_indices", "durable_updates",
    "semantic_review_observations",
})
NEW_ACTION_REPAIR_ITEM_FIELDS = frozenset({
    "action_ref", "evidence_turn_refs",
})
CONTINUATION_ACTION_REPAIR_ITEM_FIELDS = frozenset({
    "action_ref", "evidence_turn_refs", "progression_evidence_turn_refs",
})
_SCENE_REFERENCE_BASE_FIELDS = frozenset({
    "scene_alias", "current_scene_supporting_turn_refs",
    "dependency_context_turn_refs",
})
LEGACY_SCENE_REFERENCE_REPAIR_ITEM_FIELDS = _SCENE_REFERENCE_BASE_FIELDS
SCENE_REFERENCE_REPAIR_ITEM_FIELDS = (
    _SCENE_REFERENCE_BASE_FIELDS |
    {"historical_supporting_turn_refs", "uses_scene_refs"}
)
DURABLE_REPAIR_ITEM_FIELDS = frozenset({
    "candidate_update_key", "evidence_turn_refs",
})
OBSERVATION_REPAIR_ITEM_FIELDS = frozenset({
    "observation_key", "source_scene_refs", "evidence_turn_refs",
})
UNPLANNED_ACTION_REPAIR_ITEM_FIELDS = frozenset({
    "scene_alias", "action_role", "replacement",
})
FUTURE_KNOWLEDGE_REPAIR_ITEM_FIELDS = frozenset({"scene_alias", "value"})
DURABLE_REMOVAL_REPAIR_ITEM_FIELDS = frozenset({
    "candidate_update_key", "replacement",
})
DURABLE_DEPENDENCY_REPAIR_ITEM_FIELDS = frozenset({
    "candidate_update_key", "depends_on_candidate_keys",
})
VERIFIED_CONNECTION_REPAIR_ITEM_FIELDS = frozenset({
    "candidate_update_key", "verified_connection",
})
VERIFIED_CONNECTION_REMOVAL_ITEM_FIELDS = frozenset({
    "candidate_update_key", "replacement",
})
MISSING_SCENE_REVIEW_REPAIR_ITEM_FIELDS = frozenset({
    "scene_alias", "scene_review",
})
_REALIZED_STATUSES = frozenset({"realized", "partially_realized"})
REPAIR_KINDS = frozenset({
    "top_level_rehome", "scene_reference", "action_realization",
    "durable_update_evidence", "observation_reference",
    "unplanned_action_nullification",
    "future_knowledge_boolean", "durable_update_nullification",
    "durable_update_dependencies", "verified_connection",
    "verified_connection_nullification", "missing_scene_review",
})


def _sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def pack_round_validator_contract_repair_output(
    grouped: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize grouped repair data as one unambiguous tagged array."""
    rows: list[dict[str, Any]] = []
    top = grouped.get("top_level_rehome")
    if isinstance(top, Mapping):
        rows.append({"repair_kind": "top_level_rehome", **deepcopy(dict(top))})
    for field, kind in (
        ("scene_reference_repairs", "scene_reference"),
        ("action_realization_repairs", "action_realization"),
        ("durable_update_evidence_repairs", "durable_update_evidence"),
        ("observation_reference_repairs", "observation_reference"),
        ("unplanned_action_repairs", "unplanned_action_nullification"),
        ("future_knowledge_repairs", "future_knowledge_boolean"),
        ("durable_update_removals", "durable_update_nullification"),
        ("durable_dependency_repairs", "durable_update_dependencies"),
        ("verified_connection_repairs", "verified_connection"),
        ("verified_connection_removals", "verified_connection_nullification"),
        ("missing_scene_reviews", "missing_scene_review"),
    ):
        for item in grouped.get(field) or []:
            rows.append({"repair_kind": kind, **deepcopy(dict(item))})
    return {"repairs": rows}


def _unpack_round_validator_contract_repair_output(
    output: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    # Retain the explicit grouped form only for deterministic historical fixtures;
    # live prompts emit the single tagged ``repairs`` array below.  Partial grouped
    # objects remain invalid, so the malformed canary-9 response is never accepted.
    if isinstance(output, Mapping) and set(output) == _GROUPED_FIXTURE_TOP_LEVEL_FIELDS:
        grouped_fixture = deepcopy(dict(output))
        grouped_fixture["unplanned_action_repairs"] = []
        grouped_fixture["future_knowledge_repairs"] = []
        grouped_fixture["durable_update_removals"] = []
        grouped_fixture["durable_dependency_repairs"] = []
        grouped_fixture["verified_connection_repairs"] = []
        grouped_fixture["verified_connection_removals"] = []
        grouped_fixture["missing_scene_reviews"] = []
        return grouped_fixture, []
    if not isinstance(output, Mapping) or set(output) != REPAIR_TOP_LEVEL_FIELDS:
        return None, ["round_validator_contract_repair_top_level_schema"]
    rows = output.get("repairs")
    if not isinstance(rows, list):
        return None, ["round_validator_contract_repair_items_not_array"]
    grouped: dict[str, Any] = {
        "top_level_rehome": None,
        "scene_reference_repairs": [],
        "action_realization_repairs": [],
        "durable_update_evidence_repairs": [],
        "observation_reference_repairs": [],
        "unplanned_action_repairs": [],
        "future_knowledge_repairs": [],
        "durable_update_removals": [],
        "durable_dependency_repairs": [],
        "verified_connection_repairs": [],
        "verified_connection_removals": [],
        "missing_scene_reviews": [],
    }
    field_by_kind = {
        "scene_reference": "scene_reference_repairs",
        "action_realization": "action_realization_repairs",
        "durable_update_evidence": "durable_update_evidence_repairs",
        "observation_reference": "observation_reference_repairs",
        "unplanned_action_nullification": "unplanned_action_repairs",
        "future_knowledge_boolean": "future_knowledge_repairs",
        "durable_update_nullification": "durable_update_removals",
        "durable_update_dependencies": "durable_dependency_repairs",
        "verified_connection": "verified_connection_repairs",
        "verified_connection_nullification": "verified_connection_removals",
        "missing_scene_review": "missing_scene_reviews",
    }
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"contract_repair_item:{index}:schema")
            continue
        kind = row.get("repair_kind")
        if kind not in REPAIR_KINDS:
            errors.append(f"contract_repair_item:{index}:unknown_kind:{kind}")
            continue
        item = {key: deepcopy(value) for key, value in row.items()
                if key != "repair_kind"}
        if kind == "top_level_rehome":
            if grouped["top_level_rehome"] is not None:
                errors.append("contract_repair:duplicate_top_level_rehome")
            grouped["top_level_rehome"] = item
        else:
            grouped[field_by_kind[str(kind)]].append(item)
    return grouped, errors


def _resolve_indexed_historical_selections(
    grouped: Mapping[str, Any], *, repair_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve explicit model-selected menu indices to frozen historical refs."""
    resolved = deepcopy(dict(grouped))
    targets = {
        str(row.get("scene_alias") or ""): row
        for row in repair_plan.get("scene_reference_targets") or []
    }
    errors: list[str] = []
    for index, item in enumerate(resolved.get("scene_reference_repairs") or []):
        if not isinstance(item, dict) or \
                "historical_supporting_turn_indices" not in item:
            continue
        prefix = f"scene_reference_repair:{index}"
        alias = str(item.get("scene_alias") or "")
        target = targets.get(alias)
        if target is None:
            errors.append(f"{prefix}:historical_selection_unknown_scene")
            continue
        if "historical_supporting_turn_refs" in item:
            errors.append(f"{prefix}:historical_selection_mixed_representation")
            continue
        if not target.get("repair_historical_supporting_turn_refs"):
            errors.append(f"{prefix}:historical_selection_not_requested")
            continue
        selections = item.pop("historical_supporting_turn_indices")
        if (not isinstance(selections, list) or
                any(isinstance(value, bool) or not isinstance(value, int)
                    for value in selections) or
                len(set(selections)) != len(selections)):
            errors.append(f"{prefix}:historical_selection_indices")
            continue
        allowed = list(target.get("allowed_historical_turn_refs") or [])
        if any(value < 1 or value > len(allowed) for value in selections):
            errors.append(f"{prefix}:historical_selection_out_of_range")
            continue
        item["historical_supporting_turn_refs"] = [
            allowed[value - 1] for value in selections
        ]
    return resolved, errors


def _scene_maps(
    realized_scenes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    by_alias: dict[str, Mapping[str, Any]] = {}
    actions: dict[str, Mapping[str, Any]] = {}
    for scene in realized_scenes:
        alias = str(scene.get("scene_alias") or "")
        if alias:
            by_alias[alias] = scene
        for role in ("primary", "secondary"):
            action = scene.get(f"{role}_action")
            if isinstance(action, Mapping) and action.get("action_ref"):
                actions[str(action["action_ref"])] = action
    return by_alias, actions


def _verified_connection_constraints(
    action: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    arguments = action.get("resolved_arguments") or action.get("arguments") or {}
    legacy = arguments.get("connection_to_existing_world") or {}
    target_type = str(
        arguments.get("connection_type") or legacy.get("connection_type") or ""
    )
    kinds = {
        "existing_person": {"interaction_link", "explicit_relation"},
        "person": {"interaction_link", "explicit_relation"},
        "relation": {"interaction_link", "explicit_relation"},
        "interaction_link": {"interaction_link", "explicit_relation"},
        "existing_event": {"event_participant"},
        "event": {"event_participant"},
        "existing_task": {"task_participant"},
        "task": {"task_participant"},
        "information_source": {"information_source"},
        "open_thread": {"thread_participant"},
    }.get(target_type, set())
    relation_types = sorted(RELATION_TYPE_REGISTRY) if "explicit_relation" in kinds else []
    return sorted(kinds), relation_types


def build_round_validator_contract_repair_plan(
    *, model_output: Any, realized_scenes: Sequence[Mapping[str, Any]],
    compilation_errors: Sequence[str],
    historical_anchor_refs: Mapping[str, Sequence[str]] | None = None,
    historical_turns: Sequence[Mapping[str, Any]] | None = None,
    eligible_durable_action_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a frozen repair plan only for the one supported omission."""
    if not isinstance(model_output, Mapping):
        return {"valid": False, "errors": ["repair_base_output_not_object"],
                "targets": []}
    reviews = model_output.get("scene_reviews")
    if not isinstance(reviews, list):
        return {"valid": False, "errors": ["repair_scene_reviews_not_array"],
                "targets": []}

    scenes_by_alias, actions_by_ref = _scene_maps(realized_scenes)
    historical_turn_by_ref = {
        str(row.get("turn_ref") or ""): row for row in historical_turns or []
        if isinstance(row, Mapping) and row.get("turn_ref")
    }
    expected_alias_order = [
        str(scene.get("scene_alias") or "") for scene in realized_scenes
        if scene.get("scene_alias")
    ]
    submitted_aliases = [
        str(row.get("scene_alias") or "") for row in reviews
        if isinstance(row, Mapping) and row.get("scene_alias")
    ]
    missing_aliases = [
        alias for alias in expected_alias_order if alias not in submitted_aliases
    ]
    unexpected_aliases = [
        alias for alias in submitted_aliases if alias not in scenes_by_alias
    ]
    duplicate_aliases = sorted({
        alias for alias in submitted_aliases if submitted_aliases.count(alias) > 1
    })

    def action_review_template(action: Any) -> dict[str, Any] | None:
        if not isinstance(action, Mapping):
            return None
        result = {
            "status": "<STATUS>",
            "evidence_turn_refs": ["!"],
            "progression_evidence_turn_refs": (
                ["<COPY_EXACT_REF_FROM_EVIDENCE_TURN_REFS_ABOVE>"]
                if action.get("action_mode") == "continuation" else []),
            "reason": "<NONEMPTY_REASON>",
        }
        return result

    missing_scene_review_targets: list[dict[str, Any]] = []
    if missing_aliases and not unexpected_aliases and not duplicate_aliases:
        by_ref = {str(scene.get("scene_ref") or ""): scene
                  for scene in realized_scenes}

        def ancestry(scene: Mapping[str, Any]) -> set[str]:
            seen: set[str] = set()
            stack = list(scene.get("dependency_scene_refs") or
                         scene.get("predecessor_scene_refs") or [])
            while stack:
                ref = str(stack.pop())
                if ref in seen:
                    continue
                seen.add(ref)
                parent = by_ref.get(ref) or {}
                stack.extend(parent.get("dependency_scene_refs") or
                             parent.get("predecessor_scene_refs") or [])
            return seen

        for alias in missing_aliases:
            scene = scenes_by_alias[alias]
            scene_turn_refs = [
                str(row.get("turn_ref")) for row in scene.get("dialogue_turns") or []
                if isinstance(row, Mapping) and row.get("turn_ref")
            ]
            dependency_scene_refs = sorted(ancestry(scene))
            dependency_turn_refs = [
                str(turn.get("turn_ref"))
                for ref in dependency_scene_refs for turn in
                (by_ref.get(ref) or {}).get("dialogue_turns") or []
                if isinstance(turn, Mapping) and turn.get("turn_ref")
            ]
            source_turn_refs = [
                str(row.get("turn_ref"))
                for row in (scene.get("source_anchor") or {}).get(
                    "exact_excerpt") or []
                if isinstance(row, Mapping) and row.get("turn_ref")
            ]
            template = {
                "scene_alias": alias, "decision": "<SCENE_DECISION>",
                "historical_connection_status": "<CONNECTION_STATUS>",
                "continuity_status": "<CONTINUITY_STATUS>",
                "future_knowledge_leak": "<BOOLEAN>",
                "historical_consistency_status": "<CONSISTENCY_STATUS>",
                "semantic_duplicate_status": "<DUPLICATE_STATUS>",
                "new_development_status": "<DEVELOPMENT_STATUS>",
                "reason_codes": ["<REASON_CODE>"],
                "historical_supporting_turn_refs": (["!"] if source_turn_refs else []),
                "current_scene_supporting_turn_refs": ["!"],
                "dependency_context_turn_refs": [], "uses_scene_refs": [],
                "scene_requires_update_commit": False,
                "primary_action_realization": action_review_template(
                    scene.get("primary_action")),
                "secondary_action_realization": action_review_template(
                    scene.get("secondary_action")),
            }
            missing_scene_review_targets.append({
                "scene_alias": alias,
                "scene_ref": str(scene.get("scene_ref") or ""),
                "scene": deepcopy(dict(scene)),
                "allowed_current_scene_turn_refs": scene_turn_refs,
                "allowed_historical_turn_refs": source_turn_refs,
                "allowed_dependency_scene_refs": dependency_scene_refs,
                "allowed_dependency_turn_refs": dependency_turn_refs,
                "output_template": template,
            })
    top_level_rehome: dict[str, Any] | None = None
    missing_top = {"durable_updates", "semantic_review_observations"} - set(
        model_output)
    extra_top = set(model_output) - {
        "round_decision", "round_reason_codes", "scene_reviews",
        "durable_updates", "semantic_review_observations",
    }
    if (missing_top == {"durable_updates", "semantic_review_observations"} and
            not extra_top):
        misplaced = [(index, row) for index, row in enumerate(reviews)
                     if isinstance(row, Mapping) and set(row) in ({
                         "durable_updates", "semantic_review_observations"}, {
                         "durable_updates"}, {"semantic_review_observations"})]
        combined = [row for row in misplaced if set(row[1]) == {
            "durable_updates", "semantic_review_observations"}]
        split = [row for row in misplaced if len(row[1]) == 1]
        selected: list[tuple[int, Mapping[str, Any]]] = []
        if len(combined) == 1 and len(reviews) == len(realized_scenes) + 1:
            selected = combined
        elif (len(split) == 2 and len(reviews) == len(realized_scenes) + 2 and
              set().union(*(set(row) for _, row in split)) == {
                  "durable_updates", "semantic_review_observations"}):
            selected = split
        if selected:
            payload = {
                key: deepcopy(next(row[key] for _, row in selected if key in row))
                for key in ("durable_updates", "semantic_review_observations")
            }
            source_rows = [{"index": index, "value": deepcopy(dict(row))}
                           for index, row in selected]
            top_level_rehome = {
                **({"misplaced_scene_review_index": selected[0][0]}
                   if len(selected) == 1 else {
                       "misplaced_scene_review_indices": [
                           index for index, _ in selected]}),
                "expected_scene_review_count": len(realized_scenes),
                **payload,
                "misplaced_payload_sha256": _sha(payload),
                "misplaced_source_rows": source_rows,
                "misplaced_source_rows_sha256": _sha(source_rows),
            }
    aliases_by_ref = {str(scene.get("scene_ref") or ""): alias
                      for alias, scene in scenes_by_alias.items()}
    turn_ref_to_scene = {
        str(turn.get("turn_ref")): str(scene.get("scene_ref") or "")
        for scene in realized_scenes
        for turn in scene.get("dialogue_turns") or []
        if isinstance(turn, Mapping) and turn.get("turn_ref")
    }

    def dependency_ancestry(scene_ref: str) -> set[str]:
        by_ref = {str(scene.get("scene_ref") or ""): scene
                  for scene in realized_scenes}
        seen: set[str] = set()
        stack = list((by_ref.get(scene_ref) or {}).get(
            "dependency_scene_refs") or (by_ref.get(scene_ref) or {}).get(
                "predecessor_scene_refs") or [])
        while stack:
            ref = str(stack.pop())
            if ref in seen:
                continue
            seen.add(ref)
            parent = by_ref.get(ref) or {}
            stack.extend(parent.get("dependency_scene_refs") or
                         parent.get("predecessor_scene_refs") or [])
        return seen

    scene_reference_targets: list[dict[str, Any]] = []
    scene_reference_errors: list[str] = []
    future_knowledge_targets: list[dict[str, Any]] = []
    future_knowledge_errors: list[str] = []
    for review_index, review in enumerate(reviews):
        if not isinstance(review, Mapping):
            continue
        alias = str(review.get("scene_alias") or "")
        scene = scenes_by_alias.get(alias)
        if scene is None:
            continue
        scene_ref = str(scene.get("scene_ref") or "")
        if not isinstance(review.get("future_knowledge_leak"), bool):
            future_knowledge_targets.append({
                "scene_alias": alias,
                "scene_ref": scene_ref,
                "original_value": deepcopy(review.get("future_knowledge_leak")),
                "dialogue_turns": deepcopy(scene.get("dialogue_turns") or []),
                "following_constraint_turn_refs": deepcopy(
                    (scene.get("placement_choice") or {}).get(
                        "following_constraint_turn_refs") or []),
            })
            future_knowledge_errors.append(
                f"scene_review:{review_index}:future_knowledge")
        allowed_current = [str(row.get("turn_ref"))
                           for row in scene.get("dialogue_turns") or []
                           if isinstance(row, Mapping) and row.get("turn_ref")]
        current = review.get("current_scene_supporting_turn_refs")
        current_valid = (isinstance(current, list) and
                         len(set(current)) == len(current) and
                         all(isinstance(ref, str) and ref in allowed_current
                             for ref in current))
        source_anchor_id = str(scene.get("source_anchor_id") or "")
        allowed_historical = [
            str(ref) for ref in (historical_anchor_refs or {}).get(
                source_anchor_id, [])]
        historical = review.get("historical_supporting_turn_refs")
        historical_valid = (
            True if historical_anchor_refs is None else
            isinstance(historical, list) and
            len(set(historical)) == len(historical) and
            all(isinstance(ref, str) and ref in allowed_historical
                for ref in historical)
        )
        ancestry = dependency_ancestry(scene_ref)
        uses = review.get("uses_scene_refs")
        declared_sources = ({str(ref) for ref in uses}
                            if isinstance(uses, list) else set())
        uses_shape_valid = (
            isinstance(uses, list) and len(declared_sources) == len(uses) and
            all(isinstance(ref, str) and ref in ancestry for ref in uses)
        )
        allowed_dependency = [
            ref for ref, source in turn_ref_to_scene.items()
            if source in ancestry
        ]
        dependency = review.get("dependency_context_turn_refs")
        dependency_sources = {
            turn_ref_to_scene[str(ref)] for ref in dependency or []
            if isinstance(ref, str) and str(ref) in turn_ref_to_scene
        } if isinstance(dependency, list) else set()
        dependency_valid = (
            isinstance(dependency, list) and
            len(set(dependency)) == len(dependency) and
            all(isinstance(ref, str) and ref in allowed_dependency
                for ref in dependency)
        )
        uses_valid = uses_shape_valid and dependency_sources.issubset(
            declared_sources)
        if current_valid and dependency_valid and historical_valid and uses_valid:
            continue
        target = {
            "scene_alias": alias,
            "scene_ref": scene_ref,
            "original_current_scene_supporting_turn_refs": deepcopy(current),
            "original_dependency_context_turn_refs": deepcopy(dependency),
            "original_historical_supporting_turn_refs": deepcopy(historical),
            "original_uses_scene_refs": deepcopy(uses),
            "repair_current_scene_supporting_turn_refs": not current_valid,
            "repair_dependency_context_turn_refs": not dependency_valid,
            "repair_historical_supporting_turn_refs": not historical_valid,
            "repair_uses_scene_refs": not uses_valid,
            "historical_scope_enabled": historical_anchor_refs is not None,
            "allowed_current_scene_turn_refs": allowed_current,
            "allowed_dependency_turn_refs": (
                [ref for ref in allowed_dependency
                 if turn_ref_to_scene.get(str(ref)) in declared_sources]
                if uses_shape_valid and uses_valid else allowed_dependency),
            "allowed_dependency_scene_refs": sorted(ancestry),
            "allowed_historical_turn_refs": allowed_historical,
            "allowed_historical_turn_options": [
                {
                    "selection_index": option_index,
                    "turn_ref": ref,
                    **({"speaker_label": str(source.get("speaker_label") or "")}
                       if (source := historical_turn_by_ref.get(ref)) is not None
                       else {}),
                    **({"text": str(source.get("text") or "")}
                       if source is not None else {}),
                }
                for option_index, ref in enumerate(allowed_historical, start=1)
            ],
            "declared_dependency_scene_refs": sorted(declared_sources),
        }
        scene_reference_targets.append(target)
        if not current_valid:
            scene_reference_errors.append(
                f"scene_review:{review_index}:current_scene_supporting_turn_refs")
        if not dependency_valid:
            scene_reference_errors.extend([
                f"scene_review:{review_index}:dependency_context_turn_refs",
                f"scene_review:{review_index}:dependency_context_source_not_declared",
            ])
        if not uses_valid:
            scene_reference_errors.append(
                f"scene_review:{review_index}:uses_scene_refs")
            if not dependency_sources.issubset(declared_sources):
                scene_reference_errors.append(
                    f"scene_review:{review_index}:dependency_context_source_not_declared")
        if not historical_valid:
            scene_reference_errors.append(
                f"scene_review:{review_index}:historical_supporting_turn_refs")
    targets: list[dict[str, Any]] = []
    unplanned_action_targets: list[dict[str, Any]] = []
    unplanned_action_errors: list[str] = []
    expected_errors: list[str] = []
    errors: list[str] = []
    unique_compilation_errors = list(dict.fromkeys(
        str(row) for row in compilation_errors))
    repairable_action_error_suffixes = {
        "new_model_schema", "continuation_model_schema",
        "evidence_turn_refs", "evidence_required", "evidence_for_nonrealized",
        "progression_evidence_turn_refs",
        "continuation_progression_evidence_required",
        "progression_evidence_not_action_evidence",
        "new_action_progression_evidence_forbidden",
    }
    for index, review in enumerate(reviews):
        if not isinstance(review, Mapping):
            continue
        alias = str(review.get("scene_alias") or "")
        scene = scenes_by_alias.get(alias)
        if scene is None:
            continue
        scene_turns = {
            str(row.get("turn_ref")): str(row.get("text") or "")
            for row in scene.get("dialogue_turns") or []
            if isinstance(row, Mapping) and row.get("turn_ref")
        }
        continuation_roles = [role for role in ("primary", "secondary")
                              if isinstance(scene.get(f"{role}_action"), Mapping)
                              and scene[f"{role}_action"].get(
                                  "action_mode") == "continuation"]
        misplaced_progression = review.get("progression_evidence_turn_refs")
        progression_rehome_role = (
            continuation_roles[0]
            if "progression_evidence_turn_refs" in review and
            len(continuation_roles) == 1 else None)
        for role in ("primary", "secondary"):
            action = scene.get(f"{role}_action")
            value = review.get(f"{role}_action_realization")
            if not isinstance(action, Mapping) and value is not None:
                unplanned_action_targets.append({
                    "scene_alias": alias,
                    "scene_ref": str(scene.get("scene_ref") or ""),
                    "action_role": role,
                    "original_value": deepcopy(value),
                })
                unplanned_action_errors.append(
                    f"scene_review:{index}:{role}_action_realization:"
                    "unexpected_without_planned_action")
                continue
            if not (isinstance(action, Mapping) and isinstance(value, Mapping)):
                continue
            action_mode = str(action.get("action_mode") or "")
            if action_mode not in {"new", "continuation"}:
                continue
            error_prefix = (
                f"scene_review:{index}:{role}_action_realization:")
            action_errors = [
                error for error in unique_compilation_errors
                if error.startswith(error_prefix)
                and error[len(error_prefix):] in repairable_action_error_suffixes
            ]
            progression_rehome_available = role == progression_rehome_role
            progression_rehome = (
                progression_rehome_available and
                f"scene_review:{index}:model_schema" in
                unique_compilation_errors)
            if progression_rehome:
                action_errors.append(f"scene_review:{index}:model_schema")
            allowed_input_shapes = {
                MODEL_NEW_ACTION_REALIZATION_FIELDS,
                MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS,
                MODEL_NEW_ACTION_REALIZATION_FIELDS - {
                    "progression_evidence_turn_refs"},
            }
            if progression_rehome_available:
                allowed_input_shapes.add(
                    MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS - {
                        "progression_evidence_turn_refs"})
            if frozenset(value) not in allowed_input_shapes:
                errors.append(f"repair_target_shape_unsupported:{alias}:{role}")
                continue
            status = str(value.get("status") or "")
            evidence = value.get("evidence_turn_refs")
            reason = value.get("reason")
            if (status not in ACTION_REALIZATION_STATUSES or
                    not isinstance(reason, str) or not reason.strip()):
                errors.append(f"repair_target_status_invalid:{alias}:{role}")
                continue
            evidence_valid = (
                isinstance(evidence, list)
                and all(isinstance(ref, str) and ref in scene_turns
                        for ref in evidence)
                and len(set(evidence)) == len(evidence)
                and ((status in _REALIZED_STATUSES and bool(evidence)) or
                     (status not in _REALIZED_STATUSES and not evidence))
            )
            if not evidence_valid:
                action_errors.append(error_prefix + "evidence_turn_refs")
            submitted_progression = (
                misplaced_progression if progression_rehome_available else
                value.get("progression_evidence_turn_refs"))
            progression_valid = True
            if action_mode == "continuation":
                progression_valid = (
                    isinstance(submitted_progression, list)
                    and len(set(submitted_progression)) == len(
                        submitted_progression)
                    and all(isinstance(ref, str) and ref in scene_turns
                            and ref in set(evidence or [])
                            for ref in submitted_progression)
                    and ((status in _REALIZED_STATUSES and
                          bool(submitted_progression)) or
                         (status not in _REALIZED_STATUSES and
                          not submitted_progression)))
                if not progression_valid:
                    action_errors.append(
                        error_prefix + "progression_evidence_not_action_evidence")
            elif submitted_progression not in (None, []):
                action_errors.append(
                    error_prefix + "new_action_progression_evidence_forbidden")
            action_errors = list(dict.fromkeys(action_errors))
            if not action_errors:
                continue
            action_ref = str(action.get("action_ref") or "")
            resolved = actions_by_ref.get(action_ref)
            if resolved is None:
                errors.append(f"repair_target_action_ref_invalid:{alias}:{role}")
                continue
            target = {
                "scene_alias": alias,
                "scene_ref": str(scene.get("scene_ref") or ""),
                "action_role": role,
                "action_ref": action_ref,
                "action_mode": action_mode,
                "status": status,
                "reason": reason,
                "original_evidence_turn_refs": deepcopy(evidence),
                "evidence_turn_refs_requires_repair": not evidence_valid,
                "allowed_evidence_turn_refs": list(scene_turns),
                "candidate_evidence_turn_refs": (
                    list(evidence) if evidence_valid else list(scene_turns)),
                "candidate_evidence_turns": [
                    {"turn_ref": ref, "text": scene_turns[ref]}
                    for ref in (evidence if evidence_valid else scene_turns)
                ],
                "continuation_of": action.get("continuation_of"),
                "progression": deepcopy(action.get("progression")),
                "progression_rehome_from_scene": progression_rehome,
                "original_progression_present": (
                    "progression_evidence_turn_refs" in value),
                "original_progression_evidence_turn_refs": deepcopy(
                    misplaced_progression if progression_rehome else
                    value.get("progression_evidence_turn_refs")),
            }
            if action_mode == "continuation":
                target["progression_evidence_turn_refs_requires_repair"] = (
                    not progression_valid)
            else:
                target["progression_evidence_turn_refs_requires_repair"] = False
            targets.append(target)
            expected_errors.extend(action_errors)

    if len({row["action_ref"] for row in targets}) != len(targets):
        errors.append("repair_duplicate_target_action_ref")
    durable_targets: list[dict[str, Any]] = []
    durable_removal_targets: list[dict[str, Any]] = []
    durable_dependency_targets: list[dict[str, Any]] = []
    verified_connection_targets: list[dict[str, Any]] = []
    verified_connection_removal_targets: list[dict[str, Any]] = []
    reviews_by_alias = {str(row.get("scene_alias") or ""): row for row in reviews
                        if isinstance(row, Mapping)}
    durable_updates = model_output.get("durable_updates")
    if not isinstance(durable_updates, list) and top_level_rehome is not None:
        durable_updates = top_level_rehome.get("durable_updates")
    removed_update_keys: set[str] = set()
    if isinstance(durable_updates, list):
        for update_index, update in enumerate(durable_updates):
            if not isinstance(update, Mapping):
                continue
            update_key = str(update.get("candidate_update_key") or "")
            source_scene_ref = str(update.get("source_scene_ref") or "")
            source_role = str(update.get("source_action_role") or "")
            alias = aliases_by_ref.get(source_scene_ref)
            scene = scenes_by_alias.get(alias or "")
            planned_action = (scene.get(f"{source_role}_action")
                              if isinstance(scene, Mapping) and source_role in {
                                  "primary", "secondary"} else None)
            if update_key and not isinstance(planned_action, Mapping):
                removed_update_keys.add(update_key)
                durable_removal_targets.append({
                    "candidate_update_key": update_key,
                    "original_index": update_index,
                    "original_value": deepcopy(update),
                    "reason": "source_action_not_planned",
                    "validation_errors": [
                        error for error in unique_compilation_errors
                        if error.startswith(f"durable_update:{update_index}:")],
                })
            elif update_key and alias in missing_aliases:
                # The first response omitted the source scene's semantic review.  Its
                # durable candidate cannot survive that response.  The repair model must
                # explicitly null it; the newly supplied scene review is still eligible
                # for promotion, but no new durable candidate is manufactured locally.
                removed_update_keys.add(update_key)
                durable_removal_targets.append({
                    "candidate_update_key": update_key,
                    "original_index": update_index,
                    "original_value": deepcopy(update),
                    "reason": "source_scene_review_missing",
                    "validation_errors": [
                        error for error in unique_compilation_errors
                        if error.startswith(f"durable_update:{update_index}:")],
                })
            elif update_key and isinstance(planned_action, Mapping):
                action_type = str(planned_action.get("resolved_action_type") or
                                  planned_action.get("action_type") or "")
                action_mode = str(planned_action.get("action_mode") or "")
                action_ref = str(planned_action.get("action_ref") or "")
                if (eligible_durable_action_refs is not None and
                        action_ref not in set(eligible_durable_action_refs)):
                    removed_update_keys.add(update_key)
                    durable_removal_targets.append({
                        "candidate_update_key": update_key,
                        "original_index": update_index,
                        "original_value": deepcopy(update),
                        "reason": "outside_frozen_durable_candidate_scope",
                        "validation_errors": [*[
                            error for error in unique_compilation_errors
                            if error.startswith(f"durable_update:{update_index}:")
                        ],
                            f"durable_update:{update_index}:"
                            "source_action_not_eligible"],
                    })
                    continue
                connection = update.get("verified_connection")
                if update.get("verification_status") not in VERIFICATION_STATUSES:
                    # A contract retry may repair references but must never choose a
                    # semantic decision that the Validator failed to make. Preserve
                    # the scene and require explicit removal of this candidate.
                    removed_update_keys.add(update_key)
                    durable_removal_targets.append({
                        "candidate_update_key": update_key,
                        "original_index": update_index,
                        "original_value": deepcopy(update),
                        "reason": "verification_status_invalid_or_missing",
                        "validation_errors": [
                            error for error in unique_compilation_errors
                            if error.startswith(f"durable_update:{update_index}:")
                        ],
                    })
                    continue
                # A continuation of an identity-introduction action advances the
                # already introduced person/event.  It cannot safely create that
                # identity a second time.  The model must explicitly null the
                # duplicate candidate; no semantic decision is upgraded locally.
                if (action_mode == "continuation" and action_type in {
                        "introduce_connected_person", "introduce_connected_event"}):
                    removed_update_keys.add(update_key)
                    durable_removal_targets.append({
                        "candidate_update_key": update_key,
                        "original_index": update_index,
                        "original_value": deepcopy(update),
                        "reason": "identity_introduction_continuation_not_new_identity",
                        "validation_errors": [*[
                            error for error in unique_compilation_errors
                            if error.startswith(f"durable_update:{update_index}:")
                        ],
                            f"durable_update:{update_index}:"
                            "identity_introduction_continuation_forbidden"],
                    })
                    continue
                if (action_type == "introduce_connected_person" and
                        (not isinstance(connection, Mapping) or
                         connection.get("verification_status") not in
                         VERIFICATION_STATUSES)):
                    # Connection verification is a second semantic decision required
                    # by a newly introduced person. A contract repair may not invent it.
                    removed_update_keys.add(update_key)
                    durable_removal_targets.append({
                        "candidate_update_key": update_key,
                        "original_index": update_index,
                        "original_value": deepcopy(update),
                        "reason": "verification_status_invalid_or_missing",
                        "validation_errors": [
                            error for error in unique_compilation_errors
                            if error.startswith(f"durable_update:{update_index}:")
                        ],
                    })
                    continue
                expected_shape = (
                    CONNECTED_PERSON_DURABLE_UPDATE_FIELDS
                    if action_type == "introduce_connected_person"
                    else DURABLE_UPDATE_FIELDS)
                # The common failure is a semantically complete connected-person
                # candidate emitted with the base durable shape.  Its missing
                # verified_connection is repairable by the dedicated typed
                # connection branch below, so do not discard the whole candidate.
                repairable_person_missing_connection = (
                    action_type == "introduce_connected_person" and
                    set(update) == DURABLE_UPDATE_FIELDS)
                removable_non_person_connection_only = (
                    action_type != "introduce_connected_person" and
                    set(update) == DURABLE_UPDATE_FIELDS | {
                        "verified_connection"})
                if (set(update) != expected_shape and
                        not removable_non_person_connection_only and
                        not repairable_person_missing_connection):
                    removed_update_keys.add(update_key)
                    durable_removal_targets.append({
                        "candidate_update_key": update_key,
                        "original_index": update_index,
                        "original_value": deepcopy(update),
                        "reason": "durable_update_schema_invalid",
                        "validation_errors": [
                            f"durable_update:{update_index}:schema"],
                    })
    if isinstance(durable_updates, list):
        for update_index, update in enumerate(durable_updates):
            if not isinstance(update, Mapping):
                continue
            update_key = str(update.get("candidate_update_key") or "")
            if update_key in removed_update_keys:
                continue
            source_scene_ref = str(update.get("source_scene_ref") or "")
            source_role = str(update.get("source_action_role") or "")
            alias = aliases_by_ref.get(source_scene_ref)
            review = reviews_by_alias.get(alias or "")
            realization = (review.get(f"{source_role}_action_realization")
                           if isinstance(review, Mapping) else None)
            allowed_evidence = (realization.get("evidence_turn_refs")
                                if isinstance(realization, Mapping) else None)
            submitted_evidence = update.get("evidence_turn_refs")
            if (not update_key or alias is None or source_role not in {
                    "primary", "secondary"} or
                    not isinstance(allowed_evidence, list) or
                    any(not isinstance(ref, str) for ref in allowed_evidence)):
                continue
            allowed_set = set(allowed_evidence)
            submitted_evidence_valid = (
                isinstance(submitted_evidence, list) and bool(submitted_evidence) and
                all(isinstance(ref, str) and ref in allowed_set
                    for ref in submitted_evidence))
            scene = scenes_by_alias[alias]
            planned_action = scene.get(f"{source_role}_action") or {}
            action_type = str(planned_action.get("resolved_action_type") or
                              planned_action.get("action_type") or "")
            turn_text = {str(row.get("turn_ref")): str(row.get("text") or "")
                         for row in scene.get("dialogue_turns") or []
                         if isinstance(row, Mapping) and row.get("turn_ref")}
            connection = update.get("verified_connection")
            if action_type != "introduce_connected_person":
                if "verified_connection" in update:
                    verified_connection_removal_targets.append({
                        "candidate_update_key": update_key,
                        "source_scene_ref": source_scene_ref,
                        "source_action_role": source_role,
                        "source_action_type": action_type,
                        "original_verified_connection": deepcopy(connection),
                        "validation_error": (
                            f"durable_update:{update_index}:"
                            "verified_connection_for_non_person_action"),
                    })
            else:
                allowed_kinds, allowed_relation_types = (
                    _verified_connection_constraints(planned_action))
                connection_errors: list[str] = []
                preserve_fields: list[str] = []
                if (not isinstance(connection, Mapping) or
                        set(connection) != VERIFIED_CONNECTION_FIELDS):
                    connection_errors.append(
                        f"durable_update:{update_index}:verified_connection_schema")
                if isinstance(connection, Mapping):
                    status_valid = connection.get(
                        "verification_status") in VERIFICATION_STATUSES
                    kind = str(connection.get("connection_kind") or "")
                    kind_valid = kind in allowed_kinds
                    connection_evidence = connection.get("evidence_turn_refs")
                    evidence_valid = (
                        isinstance(connection_evidence, list) and
                        bool(connection_evidence) and
                        all(isinstance(ref, str) and ref in allowed_set
                            for ref in connection_evidence) and
                        len(set(connection_evidence)) == len(connection_evidence))
                    reason_valid = (isinstance(connection.get("reason"), str) and
                                    bool(str(connection.get("reason") or "").strip()))
                    relation_type = connection.get("relation_type")
                    relation_valid = (
                        relation_type in allowed_relation_types
                        if kind == "explicit_relation" else relation_type is None)
                    for field, valid in (
                        ("verification_status", status_valid),
                        ("connection_kind", kind_valid),
                        ("relation_type", relation_valid),
                        ("evidence_turn_refs", evidence_valid),
                        ("reason", reason_valid),
                    ):
                        if valid:
                            preserve_fields.append(field)
                    if not status_valid:
                        connection_errors.append(
                            f"durable_update:{update_index}:verified_connection_status")
                    if str(connection.get("connection_kind") or "") not in {
                            "interaction_link", "explicit_relation", "event_participant",
                            "task_participant", "information_source", "thread_participant"}:
                        connection_errors.append(
                            f"durable_update:{update_index}:verified_connection_kind")
                    if not evidence_valid:
                        connection_errors.append(
                            f"durable_update:{update_index}:"
                            "verified_connection_evidence_turn_refs")
                    if not reason_valid:
                        connection_errors.append(
                            f"durable_update:{update_index}:verified_connection_reason")
                    if not kind_valid:
                        connection_errors.append(
                            f"durable_update:{update_index}:"
                            "verified_connection_target_kind_mismatch")
                    if kind == "explicit_relation" and relation_type not in \
                            RELATION_TYPE_REGISTRY:
                        connection_errors.append(
                            f"durable_update:{update_index}:"
                            "verified_connection_relation_type")
                    elif kind != "explicit_relation" and relation_type is not None:
                        connection_errors.append(
                            f"durable_update:{update_index}:"
                            "verified_connection_relation_type_forbidden")
                # A durable-evidence repair already re-submits the same evidence-bearing
                # connected-person claim (see apply below). Do not demand a redundant
                # verified_connection item when that is the only invalid member.
                target_kind_mismatch = (
                    f"durable_update:{update_index}:"
                    "verified_connection_target_kind_mismatch"
                )
                if target_kind_mismatch in connection_errors:
                    # A mismatched connection kind is not a formatting omission: choosing
                    # a different kind would create a new semantic assertion.  Preserve the
                    # accepted scene but require explicit nullification of this durable
                    # candidate, matching the scene-valid/update-invalid contract.
                    removed_update_keys.add(update_key)
                    durable_removal_targets.append({
                        "candidate_update_key": update_key,
                        "original_index": update_index,
                        "original_value": deepcopy(update),
                        "reason": "verified_connection_target_kind_mismatch",
                        "validation_errors": list(dict.fromkeys(connection_errors)),
                    })
                    continue
                evidence_error = (
                    f"durable_update:{update_index}:"
                    "verified_connection_evidence_turn_refs")
                connection_covered_by_durable_evidence = (
                    not submitted_evidence_valid and
                    set(connection_errors) == {evidence_error})
                if connection_errors and not connection_covered_by_durable_evidence:
                    verified_connection_targets.append({
                        "candidate_update_key": update_key,
                        "source_scene_ref": source_scene_ref,
                        "source_action_role": source_role,
                        "source_action_type": action_type,
                        "original_verified_connection": deepcopy(connection),
                        "preserve_fields": preserve_fields,
                        "allowed_verification_statuses": sorted(VERIFICATION_STATUSES),
                        "allowed_connection_kinds": allowed_kinds,
                        "allowed_relation_types": allowed_relation_types,
                        "allowed_evidence_turn_refs": list(allowed_evidence),
                        "allowed_evidence_turns": [
                            {"turn_ref": ref, "text": turn_text.get(ref, "")}
                            for ref in allowed_evidence],
                        "validation_errors": list(dict.fromkeys(connection_errors)),
                    })
            if not submitted_evidence_valid:
                durable_targets.append({
                    "candidate_update_key": update_key,
                    "source_scene_ref": source_scene_ref,
                    "source_action_role": source_role,
                    "source_action_ref": (
                        str((scene.get(f"{source_role}_action") or {}).get(
                            "action_ref") or "")),
                    "verification_status": update.get("verification_status"),
                    "allowed_evidence_turn_refs": list(allowed_evidence),
                    "allowed_evidence_turns": [
                        {"turn_ref": ref, "text": turn_text.get(ref, "")}
                        for ref in allowed_evidence
                    ],
                    "original_evidence_turn_refs": deepcopy(submitted_evidence),
                    "original_verified_connection_evidence_turn_refs": (
                        deepcopy(connection.get("evidence_turn_refs"))
                        if isinstance(connection, Mapping) else None),
                    "validation_error": (
                        f"durable_update:{update_index}:evidence_turn_refs"),
                })
        # A candidate cannot survive after any dependency it explicitly names is
        # removed. Do not ask the repair model to erase or redirect that dependency;
        # doing so would change the proposed transaction semantics.
        removal_changed = True
        while removal_changed:
            removal_changed = False
            for update_index, update in enumerate(durable_updates):
                if not isinstance(update, Mapping):
                    continue
                key = str(update.get("candidate_update_key") or "")
                dependencies = update.get("depends_on_candidate_keys")
                if (not key or key in removed_update_keys or
                        not isinstance(dependencies, list)):
                    continue
                removed_dependencies = sorted(
                    str(dep) for dep in dependencies if str(dep) in removed_update_keys)
                if not removed_dependencies:
                    continue
                removed_update_keys.add(key)
                durable_removal_targets.append({
                    "candidate_update_key": key,
                    "original_index": update_index,
                    "original_value": deepcopy(update),
                    "reason": "dependency_candidate_removed",
                    "removed_dependency_keys": removed_dependencies,
                    "validation_errors": [
                        f"durable_update:{update_index}:dependency_candidate_removed"
                    ],
                })
                removal_changed = True
        durable_targets = [
            row for row in durable_targets
            if row["candidate_update_key"] not in removed_update_keys
        ]
        verified_connection_targets = [
            row for row in verified_connection_targets
            if row["candidate_update_key"] not in removed_update_keys
        ]
        verified_connection_removal_targets = [
            row for row in verified_connection_removal_targets
            if row["candidate_update_key"] not in removed_update_keys
        ]
        all_update_keys = {str(row.get("candidate_update_key") or "")
                           for row in durable_updates if isinstance(row, Mapping)}
        allowed_remaining_keys = sorted(all_update_keys - removed_update_keys)
        for update_index, update in enumerate(durable_updates):
            if not isinstance(update, Mapping):
                continue
            key = str(update.get("candidate_update_key") or "")
            if not key or key in removed_update_keys:
                continue
            dependencies = update.get("depends_on_candidate_keys")
            dependency_valid = (isinstance(dependencies, list) and
                                len(set(dependencies)) == len(dependencies) and
                                all(isinstance(dep, str) and
                                    dep in allowed_remaining_keys and dep != key
                                    for dep in dependencies))
            if not dependency_valid:
                durable_dependency_targets.append({
                    "candidate_update_key": key,
                    "original_depends_on_candidate_keys": deepcopy(dependencies),
                    "allowed_dependency_keys": [row for row in allowed_remaining_keys
                                                if row != key],
                    "validation_error": (
                        f"durable_update:{key}:depends_on_candidate_keys"),
                })
    if len({row["candidate_update_key"] for row in durable_targets}) != len(
            durable_targets):
        errors.append("repair_duplicate_durable_update_key")
    latent_durable_errors = [row["validation_error"] for row in durable_targets]
    latent_dependency_errors = [row["validation_error"]
                                for row in durable_dependency_targets]
    latent_connection_errors = [
        error for row in verified_connection_targets
        for error in row["validation_errors"]]
    latent_connection_errors.extend(
        row["validation_error"] for row in verified_connection_removal_targets)
    latent_removal_errors = [
        error for row in durable_removal_targets
        for error in row.get("validation_errors") or []
    ]
    observation_targets: list[dict[str, Any]] = []
    observation_errors: list[str] = []
    observations = model_output.get("semantic_review_observations")
    if not isinstance(observations, list) and top_level_rehome is not None:
        observations = top_level_rehome.get("semantic_review_observations")
    allowed_scene_refs = [str(scene.get("scene_ref") or "")
                          for scene in realized_scenes]
    scene_turn_refs = {
        str(scene.get("scene_ref") or ""): [
            str(turn.get("turn_ref"))
            for turn in scene.get("dialogue_turns") or []
            if isinstance(turn, Mapping) and turn.get("turn_ref")]
        for scene in realized_scenes
    }
    if isinstance(observations, list):
        for observation_index, observation in enumerate(observations):
            if not isinstance(observation, Mapping):
                continue
            key = str(observation.get("observation_key") or "")
            sources = observation.get("source_scene_refs")
            evidence = observation.get("evidence_turn_refs")
            selected_sources = ({str(ref) for ref in sources}
                                if isinstance(sources, list) else set())
            allowed_evidence = {
                ref for scene_ref in selected_sources
                for ref in scene_turn_refs.get(scene_ref, [])
            }
            sources_valid = (
                isinstance(sources, list) and bool(sources) and
                len(set(sources)) == len(sources) and
                all(isinstance(ref, str) and ref in allowed_scene_refs
                    for ref in sources)
            )
            evidence_valid = (
                isinstance(evidence, list) and bool(evidence) and
                len(set(evidence)) == len(evidence) and
                all(isinstance(ref, str) and ref in allowed_evidence
                    for ref in evidence)
            )
            if sources_valid and evidence_valid:
                continue
            observation_targets.append({
                "observation_key": key,
                "original_source_scene_refs": deepcopy(sources),
                "original_evidence_turn_refs": deepcopy(evidence),
                "allowed_scene_refs": allowed_scene_refs,
                "allowed_evidence_turn_refs_by_scene": scene_turn_refs,
                "validation_errors": [
                    *( [f"observation:{observation_index}:source_scene_refs"]
                       if not sources_valid else []),
                    *( [f"observation:{observation_index}:evidence_turn_refs"]
                       if not evidence_valid else []),
                ],
            })
            observation_errors.extend(
                observation_targets[-1]["validation_errors"])
    supported_errors = (set(expected_errors) | set(latent_durable_errors) |
                        set(scene_reference_errors) | set(observation_errors) |
                        set(unplanned_action_errors) |
                        set(future_knowledge_errors) | set(latent_dependency_errors) |
                        set(latent_connection_errors) | set(latent_removal_errors))
    if missing_scene_review_targets:
        supported_errors.add("validator_scene_review_coverage")
    if top_level_rehome is not None:
        supported_errors.add("validator_model_top_level_schema")
    if (not targets and not durable_targets and not scene_reference_targets and
            not observation_targets and not unplanned_action_targets and
            not future_knowledge_targets and not durable_removal_targets and
            not durable_dependency_targets and not verified_connection_targets and
            not verified_connection_removal_targets and
            not missing_scene_review_targets and top_level_rehome is None):
        errors.append("repair_no_supported_targets")
    if (not unique_compilation_errors or
            not set(unique_compilation_errors).issubset(supported_errors)):
        errors.append("repair_errors_not_exclusively_supported_contract_fields")
    return {
        "repair_version": ROUND_VALIDATOR_CONTRACT_REPAIR_VERSION_V11,
        "valid": not errors,
        "errors": errors,
        "base_model_output_sha256": _sha(model_output),
        "top_level_rehome": top_level_rehome,
        "scene_reference_targets": scene_reference_targets,
        "target_scene_aliases": [row["scene_alias"]
                                 for row in scene_reference_targets],
        "targets": targets,
        "target_action_refs": [row["action_ref"] for row in targets],
        "durable_update_targets": durable_targets,
        "target_candidate_update_keys": [
            row["candidate_update_key"] for row in durable_targets],
        "observation_targets": observation_targets,
        "target_observation_keys": [row["observation_key"]
                                    for row in observation_targets],
        "unplanned_action_targets": unplanned_action_targets,
        "target_unplanned_action_slots": [
            f"{row['scene_alias']}.{row['action_role']}"
            for row in unplanned_action_targets],
        "future_knowledge_targets": future_knowledge_targets,
        "target_future_knowledge_scenes": [row["scene_alias"]
                                           for row in future_knowledge_targets],
        "durable_removal_targets": durable_removal_targets,
        "target_removed_update_keys": sorted(removed_update_keys),
        "durable_dependency_targets": durable_dependency_targets,
        "verified_connection_targets": verified_connection_targets,
        "verified_connection_removal_targets": verified_connection_removal_targets,
        "missing_scene_review_targets": missing_scene_review_targets,
        "expected_scene_alias_order": expected_alias_order,
    }


def build_round_validator_contract_repair_prompt(
    *, repair_plan: Mapping[str, Any], frozen_input_sha256: str,
) -> str:
    if not repair_plan.get("valid"):
        raise ValueError("round_validator_contract_repair_plan_invalid")
    selection_marker = "!"
    compact_targets = []
    for row in repair_plan.get("targets") or []:
        compact_row = {
            key: deepcopy(row.get(key)) for key in (
            "scene_alias", "scene_ref", "action_role", "action_ref", "action_mode", "status",
            "reason", "original_evidence_turn_refs",
            "evidence_turn_refs_requires_repair", "allowed_evidence_turn_refs",
            "candidate_evidence_turn_refs", "candidate_evidence_turns",
            "continuation_of", "progression", "progression_rehome_from_scene",
            "original_progression_evidence_turn_refs",
            "progression_evidence_turn_refs_requires_repair",
        )
        }
        # Invalid refs are useful in the local audit plan but harmful in the
        # model-facing repair prompt: they become a tempting copy target.  The
        # model receives only the exact valid choices and their dialogue text.
        if row.get("evidence_turn_refs_requires_repair"):
            compact_row.pop("original_evidence_turn_refs", None)
        else:
            # A progression-only repair may choose only from the action evidence
            # that the first semantic review already froze. Exposing every scene
            # turn here invites the model to repeat the exact subset violation the
            # repair is intended to fix.
            compact_row["allowed_evidence_turn_refs"] = deepcopy(
                row.get("original_evidence_turn_refs") or [])
        if row.get("progression_evidence_turn_refs_requires_repair"):
            compact_row.pop("original_progression_evidence_turn_refs", None)
        compact_targets.append(compact_row)
    compact_durable_targets = [{
        key: deepcopy(row.get(key)) for key in (
            "candidate_update_key", "source_scene_ref", "source_action_role",
            "source_action_ref", "verification_status",
            "allowed_evidence_turn_refs", "allowed_evidence_turns",
        )
    } for row in repair_plan.get("durable_update_targets") or []]
    # The output template below already carries every frozen value.  Expose an
    # allowed set only when the corresponding field actually needs repair;
    # repeating full historical/dependency scopes for preserved fields made a
    # small contract retry nearly as large as the original review request.
    compact_scene_targets = []
    for row in repair_plan.get("scene_reference_targets") or []:
        compact_row = {
            key: deepcopy(row.get(key)) for key in (
                "scene_alias", "scene_ref",
                "repair_current_scene_supporting_turn_refs",
                "repair_dependency_context_turn_refs",
                "repair_uses_scene_refs",
            )
        }
        if row.get("repair_current_scene_supporting_turn_refs"):
            compact_row["allowed_current_scene_turn_refs"] = deepcopy(
                row.get("allowed_current_scene_turn_refs"))
        if row.get("repair_dependency_context_turn_refs"):
            compact_row["allowed_dependency_turn_refs"] = deepcopy(
                row.get("allowed_dependency_turn_refs"))
        if row.get("repair_uses_scene_refs"):
            compact_row["allowed_dependency_scene_refs"] = deepcopy(
                row.get("allowed_dependency_scene_refs"))
        if row.get("historical_scope_enabled"):
            compact_row["repair_historical_supporting_turn_refs"] = bool(
                row.get("repair_historical_supporting_turn_refs"))
            if row.get("repair_historical_supporting_turn_refs"):
                compact_row["allowed_historical_turn_options"] = deepcopy(
                    row.get("allowed_historical_turn_options") or [
                        {"selection_index": option_index, "turn_ref": ref}
                        for option_index, ref in enumerate(
                            row.get("allowed_historical_turn_refs") or [], start=1)
                    ])
        compact_scene_targets.append(compact_row)
    compact_observation_targets = [{
        key: deepcopy(row.get(key)) for key in (
            "observation_key", "original_source_scene_refs",
            "original_evidence_turn_refs", "allowed_scene_refs",
            "allowed_evidence_turn_refs_by_scene",
        )
    } for row in repair_plan.get("observation_targets") or []]
    compact_unplanned_targets = [{
        key: deepcopy(row.get(key)) for key in (
            "scene_alias", "scene_ref", "action_role",
        )
    } for row in repair_plan.get("unplanned_action_targets") or []]
    compact_future_targets = [{
        key: deepcopy(row.get(key)) for key in (
            "scene_alias", "scene_ref", "original_value", "dialogue_turns",
            "following_constraint_turn_refs",
        )
    } for row in repair_plan.get("future_knowledge_targets") or []]
    compact_removal_targets = [{
        key: deepcopy(row.get(key)) for key in (
            "candidate_update_key", "reason",
        )
    } for row in repair_plan.get("durable_removal_targets") or []]
    compact_dependency_targets = [{
        key: deepcopy(row.get(key)) for key in (
            "candidate_update_key", "original_depends_on_candidate_keys",
            "allowed_dependency_keys",
        )
    } for row in repair_plan.get("durable_dependency_targets") or []]
    compact_connection_targets = [{
        key: deepcopy(row.get(key)) for key in (
            "candidate_update_key", "source_scene_ref", "source_action_role",
            "source_action_type", "original_verified_connection", "preserve_fields",
            "allowed_verification_statuses", "allowed_connection_kinds",
            "allowed_relation_types", "allowed_evidence_turn_refs",
            "allowed_evidence_turns", "validation_errors",
        )
    } for row in repair_plan.get("verified_connection_targets") or []]
    compact_connection_removals = [{
        key: deepcopy(row.get(key)) for key in (
            "candidate_update_key", "source_scene_ref", "source_action_role",
            "source_action_type", "original_verified_connection", "validation_error",
        )
    } for row in repair_plan.get("verified_connection_removal_targets") or []]
    compact_missing_scene_targets = []
    missing_scene_output_templates = {}
    for row in repair_plan.get("missing_scene_review_targets") or []:
        scene = row.get("scene") or {}
        compact_scene = {
            key: deepcopy(scene.get(key)) for key in (
                "source_anchor_id",
                "link_type", "bridge_reason", "new_development", "goal",
                "primary_action", "secondary_action",
                "desired_outcome", "dialogue_turns",
            ) if key in scene
        }
        compact_missing_scene_targets.append({
            "scene_alias": deepcopy(row.get("scene_alias")),
            "scene_ref": deepcopy(row.get("scene_ref")),
            "scene": compact_scene,
            "allowed_current_scene_turn_refs": deepcopy(
                row.get("allowed_current_scene_turn_refs")),
            "allowed_historical_turn_refs": deepcopy(
                row.get("allowed_historical_turn_refs")),
            "allowed_dependency_scene_refs": deepcopy(
                row.get("allowed_dependency_scene_refs")),
            "allowed_dependency_turn_refs": deepcopy(
                row.get("allowed_dependency_turn_refs")),
        })
        missing_scene_output_templates[str(row.get("scene_alias") or "")] = (
            deepcopy(row.get("output_template")))
    top_level_target = deepcopy(repair_plan.get("top_level_rehome"))
    compact_top_level_target = None
    if isinstance(top_level_target, Mapping):
        compact_top_level_target = {
            key: deepcopy(top_level_target.get(key)) for key in (
                "misplaced_scene_review_index", "misplaced_scene_review_indices",
                "expected_scene_review_count",
                "durable_updates", "semantic_review_observations",
                "misplaced_payload_sha256",
            ) if key in top_level_target
        }
    grouped_template = {
        "top_level_rehome": ({
            **({"misplaced_scene_review_index": top_level_target[
                "misplaced_scene_review_index"]}
               if "misplaced_scene_review_index" in top_level_target else {
                   "misplaced_scene_review_indices": deepcopy(
                       top_level_target["misplaced_scene_review_indices"])}),
            "durable_updates": deepcopy(top_level_target["durable_updates"]),
            "semantic_review_observations": deepcopy(
                top_level_target["semantic_review_observations"]),
        } if isinstance(top_level_target, Mapping) else None),
        "scene_reference_repairs": [({
            "scene_alias": row["scene_alias"],
        } | ({
            "current_scene_supporting_turn_refs": [selection_marker],
        } if row["repair_current_scene_supporting_turn_refs"] else {}) | ({
            "dependency_context_turn_refs": (
                [selection_marker] if row.get("allowed_dependency_turn_refs") else []),
        } if row["repair_dependency_context_turn_refs"] else {}) | ({
            "uses_scene_refs": (
                [selection_marker] if row.get("allowed_dependency_scene_refs") else []),
        } if row["repair_uses_scene_refs"] else {}) | ({
            "historical_supporting_turn_indices": (
                ["<CHOOSE_ZERO_OR_MORE_1_BASED_INDICES>"]
                if row["repair_historical_supporting_turn_refs"]
                else deepcopy(row[
                    "original_historical_supporting_turn_refs"])),
        } if row.get("repair_historical_supporting_turn_refs") else {}))
            for row in repair_plan.get("scene_reference_targets") or []],
        "action_realization_repairs": [({
            "action_ref": row["action_ref"],
        } | ({
            "evidence_turn_refs": [selection_marker],
        } if row["evidence_turn_refs_requires_repair"] else {}) | ({
            "progression_evidence_turn_refs": (
                [selection_marker] if row.get("status") in _REALIZED_STATUSES else []),
        } if row.get("action_mode") == "continuation" and
             row["progression_evidence_turn_refs_requires_repair"] else {}))
            for row in compact_targets],
        "durable_update_evidence_repairs": [{
            "candidate_update_key": row["candidate_update_key"],
            "evidence_turn_refs": [selection_marker],
        } for row in compact_durable_targets],
        "observation_reference_repairs": [{
            "observation_key": row["observation_key"],
            "source_scene_refs": ["<SCENE_REF>"],
            "evidence_turn_refs": [selection_marker],
        } for row in compact_observation_targets],
        "unplanned_action_repairs": [{
            "scene_alias": row["scene_alias"],
            "action_role": row["action_role"],
            "replacement": None,
        } for row in compact_unplanned_targets],
        "future_knowledge_repairs": [{
            "scene_alias": row["scene_alias"], "value": False,
        } for row in compact_future_targets],
        "durable_update_removals": [{
            "candidate_update_key": row["candidate_update_key"],
            "replacement": None,
        } for row in compact_removal_targets],
        "durable_dependency_repairs": [{
            "candidate_update_key": row["candidate_update_key"],
            "depends_on_candidate_keys": [],
        } for row in compact_dependency_targets],
        "verified_connection_repairs": [{
            "candidate_update_key": row["candidate_update_key"],
            "verified_connection": {
                field: (deepcopy((row.get("original_verified_connection") or {}).get(field))
                        if field in row["preserve_fields"] else
                        ([selection_marker] if field == "evidence_turn_refs" else
                         None if field == "relation_type" else
                         "<VALUE>"))
                for field in sorted(VERIFIED_CONNECTION_FIELDS)
            },
        } for row in compact_connection_targets],
        "verified_connection_removals": [{
            "candidate_update_key": row["candidate_update_key"],
            "replacement": None,
        } for row in compact_connection_removals],
        "missing_scene_reviews": [{
            "scene_alias": row["scene_alias"],
            "scene_review": deepcopy(missing_scene_output_templates[
                str(row["scene_alias"])]),
        } for row in compact_missing_scene_targets],
    }
    sections = [
        "ROUND VALIDATOR CONTINUATION CONTRACT REPAIR — "
        f"{ROUND_VALIDATOR_CONTRACT_REPAIR_VERSION_V11}. Return JSON only.",
        f"FROZEN ROUND INPUT SHA256\n{frozen_input_sha256}",
        "The first review is semantically frozen. Do not re-review the round and do not "
        "change a decision, status, reason, any non-target evidence, durable update, "
        "observation, or scene order. Repair only the explicitly listed contract fields.",
    ]
    if compact_top_level_target is not None:
        sections.extend([
            "Copy the durable updates and semantic observations from the frozen "
            "top-level rehome target exactly into top_level_rehome and preserve the "
            "exact misplaced scene-review index or indices. This explicitly re-submits "
            "misplaced data; do not edit, summarize, add, or remove any item.",
            "FROZEN TOP-LEVEL REHOME TARGET\n" + json.dumps(
                compact_top_level_target, ensure_ascii=False,
                separators=(",", ":")),
        ])
    if compact_missing_scene_targets:
        sections.extend([
            "Review only each listed missing scene once. Do not revisit existing scene "
            "reviews. Replace every example semantic value in scene_review using the "
            "frozen scene dialogue and action plan. Use exact current, historical, and "
            "dependency refs only from the corresponding allowed lists. A realized or "
            "partially_realized action needs non-empty current-scene evidence; a realized "
            "continuation also needs non-empty progression_evidence_turn_refs, and every "
            "progression ref MUST be copied from that same action realization's "
            "evidence_turn_refs. It is a subset of action evidence, not a subset of all "
            "current-scene turns. Preserve the "
            "exact scene-review keys and return no durable update for this newly reviewed "
            "scene in this repair response.",
            "ALLOWED MISSING-SCENE SEMANTIC VALUES\n" + json.dumps({
                "decision": sorted(SCENE_DECISIONS),
                "historical_connection_status": sorted(CONNECTION_STATUSES),
                "continuity_status": sorted(CONTINUITY_STATUSES),
                "historical_consistency_status": sorted(CONSISTENCY_STATUSES),
                "semantic_duplicate_status": sorted(DUPLICATE_STATUSES),
                "new_development_status": sorted(DEVELOPMENT_STATUSES),
                "reason_codes": sorted(SCENE_REASON_CODES),
                "future_knowledge_leak": "JSON boolean",
                "scene_requires_update_commit": "JSON boolean",
            }, ensure_ascii=False, separators=(",", ":")),
            "FROZEN MISSING SCENE REVIEW TARGETS\n" + json.dumps(
                compact_missing_scene_targets, ensure_ascii=False,
                separators=(",", ":")),
        ])
    if compact_scene_targets:
        sections.extend([
            "For each scene reference repair, preserve a field exactly when its repair "
            "flag is false. When repair_current_scene_supporting_turn_refs is true, "
            "explicitly select a non-empty exact subset of allowed_current_scene_turn_refs. "
            "When repair_dependency_context_turn_refs is true, select exact refs only from "
            "allowed_dependency_turn_refs; use an empty list only when that allowed list is "
            "empty. When repair_uses_scene_refs is true, select the exact dependency scene "
            "refs that own the submitted dependency turns from allowed_dependency_scene_refs; "
            "use [] only when no dependency turns are cited. Never put a dependency scene "
            "turn in historical support." + (
                " When repair_historical_supporting_turn_refs is true, submit "
                "historical_supporting_turn_indices using only the 1-based selection_index "
                "values from that scene's allowed_historical_turn_options. Select only "
                "turns that actually support the frozen review. Use [] when none support "
                "it; the main Validator contract permits empty historical support. Never "
                "copy a turn_ref into the index field, and never use R01 scene-local refs."
                if any("repair_historical_supporting_turn_refs" in row
                       for row in compact_scene_targets) else ""),
            "FROZEN SCENE REFERENCE TARGETS\n" + json.dumps(
                compact_scene_targets, ensure_ascii=False, separators=(",", ":")),
        ])
    if compact_targets:
        sections.extend([
        "Action repairs are sparse. Return action_ref plus only the fields shown in the "
        "final output template; omitted valid fields remain frozen locally. For a realized "
        "or partially_realized continuation, select a non-empty exact subset of the repaired "
        "evidence_turn_refs that demonstrates the concrete new progression. For status "
        "not_realized or invalid, return an empty progression list. Never invent or copy an "
        "unlisted turn ref.",
        "For evidence_turn_refs: when evidence_turn_refs_requires_repair is false, omit the "
        "field. When it is true, explicitly select a valid "
        "status-compatible exact subset of allowed_evidence_turn_refs. Do not guess by "
        "numeric proximity, and do not silently delete an invalid ref locally.",
        "FROZEN REPAIR TARGETS\n" + json.dumps(
            compact_targets, ensure_ascii=False, separators=(",", ":")),
        ])
    if compact_durable_targets:
        sections.extend([
            "Each durable update repair must select a non-empty exact subset of its listed "
            "allowed_evidence_turn_refs. It may not cite another scene or action.",
            "FROZEN DURABLE UPDATE EVIDENCE TARGETS\n" + json.dumps(
                compact_durable_targets, ensure_ascii=False, separators=(",", ":")),
        ])
    if compact_observation_targets:
        sections.extend([
            "For every observation repair, select at least one exact source_scene_ref "
            "from allowed_scene_refs, then select at least one exact evidence turn from "
            "allowed_evidence_turn_refs_by_scene for those selected scenes only. Do not "
            "change the observation description, type, disposition, or key.",
            "FROZEN OBSERVATION REFERENCE TARGETS\n" + json.dumps(
                compact_observation_targets, ensure_ascii=False,
                separators=(",", ":")),
        ])
    if compact_unplanned_targets:
        sections.extend([
            "For every unplanned action target, explicitly return replacement=null. "
            "There is no planned action in that slot, so no realization may be retained, "
            "invented, or moved to another slot.",
            "FROZEN UNPLANNED ACTION TARGETS\n" + json.dumps(
                compact_unplanned_targets, ensure_ascii=False,
                separators=(",", ":")),
        ])
    if compact_future_targets:
        sections.extend([
            "For every future-knowledge target, explicitly return a JSON boolean. true "
            "means the scene uses knowledge unavailable at its narrative placement; "
            "false means it does not. Re-evaluate only this field from the frozen scene.",
            "FROZEN FUTURE-KNOWLEDGE TARGETS\n" + json.dumps(
                compact_future_targets, ensure_ascii=False, separators=(",", ":")),
        ])
    if compact_removal_targets:
        sections.extend([
            "For every durable update nullification target, explicitly return "
            "replacement=null. Its source action was not planned or its source scene review "
            "was missing, so that candidate cannot survive the frozen first response.",
            "FROZEN DURABLE UPDATE NULLIFICATION TARGETS\n" + json.dumps(
                compact_removal_targets, ensure_ascii=False, separators=(",", ":")),
        ])
    if compact_dependency_targets:
        sections.extend([
            "For every durable dependency target, select zero or more exact keys from "
            "allowed_dependency_keys. Do not cite a nullified, unknown, or self key.",
            "FROZEN DURABLE DEPENDENCY TARGETS\n" + json.dumps(
                compact_dependency_targets, ensure_ascii=False,
                separators=(",", ":")),
        ])
    if compact_connection_targets:
        sections.extend([
            "For each connected-person repair, re-submit the complete verified_connection "
            "object. Copy every preserve_fields value exactly. Select a non-empty exact "
            "evidence subset from allowed_evidence_turn_refs. Select connection_kind only "
            "from allowed_connection_kinds. relation_type must be null unless kind is "
            "explicit_relation, in which case it must come from allowed_relation_types. "
            "Do not change the durable update or invent evidence.",
            "FROZEN VERIFIED CONNECTION TARGETS\n" + json.dumps(
                compact_connection_targets, ensure_ascii=False,
                separators=(",", ":")),
        ])
    if compact_connection_removals:
        sections.extend([
            "For every verified-connection nullification target, explicitly return "
            "replacement=null. The source action is not introduce_connected_person, so the "
            "field is forbidden; preserve the durable update itself unchanged.",
            "FROZEN VERIFIED CONNECTION NULLIFICATION TARGETS\n" + json.dumps(
                compact_connection_removals, ensure_ascii=False,
                separators=(",", ":")),
        ])
    template = pack_round_validator_contract_repair_output(grouped_template)
    sections.extend([
        "FINAL: emit each target once. Use non-empty allowed action refs only for "
        "realized/partially_realized actions; use [] for not_realized/invalid action "
        "evidence and when no allowed historical turn supports the frozen review.",
        "OUTPUT ONLY THIS OBJECT\n" + json.dumps(
            template, ensure_ascii=False, separators=(",", ":")),
    ])
    return "\n\n".join(sections)


def validate_round_validator_contract_repair_output(
    output: Any, *, repair_plan: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    grouped_output, unpack_errors = _unpack_round_validator_contract_repair_output(
        output)
    if grouped_output is None or unpack_errors:
        return {"valid": False, "classification": "schema_invalid",
                "errors": unpack_errors}
    grouped_output, selection_errors = _resolve_indexed_historical_selections(
        grouped_output, repair_plan=repair_plan)
    if selection_errors:
        return {"valid": False, "classification": "schema_invalid",
                "errors": selection_errors}
    repairs = grouped_output.get("action_realization_repairs")
    durable_repairs = grouped_output.get("durable_update_evidence_repairs")
    scene_repairs = grouped_output.get("scene_reference_repairs")
    observation_repairs = grouped_output.get("observation_reference_repairs")
    unplanned_action_repairs = grouped_output.get("unplanned_action_repairs")
    future_knowledge_repairs = grouped_output.get("future_knowledge_repairs")
    durable_update_removals = grouped_output.get("durable_update_removals")
    durable_dependency_repairs = grouped_output.get("durable_dependency_repairs")
    verified_connection_repairs = grouped_output.get("verified_connection_repairs")
    verified_connection_removals = grouped_output.get("verified_connection_removals")
    missing_scene_reviews = grouped_output.get("missing_scene_reviews")
    top_level_rehome = grouped_output.get("top_level_rehome")
    planned_rehome = repair_plan.get("top_level_rehome")
    if planned_rehome is None:
        if top_level_rehome is not None:
            errors.append("top_level_rehome:unexpected")
    elif not isinstance(top_level_rehome, Mapping) or frozenset(
            top_level_rehome) not in {
                TOP_LEVEL_REHOME_FIELDS, SPLIT_TOP_LEVEL_REHOME_FIELDS}:
        errors.append("top_level_rehome:schema")
    else:
        submitted_indices = (
            top_level_rehome.get("misplaced_scene_review_indices")
            if "misplaced_scene_review_indices" in top_level_rehome else
            [top_level_rehome.get("misplaced_scene_review_index")])
        planned_indices = (
            planned_rehome.get("misplaced_scene_review_indices")
            if "misplaced_scene_review_indices" in planned_rehome else
            [planned_rehome.get("misplaced_scene_review_index")])
        if submitted_indices != planned_indices:
            errors.append("top_level_rehome:index_mismatch")
        submitted_payload = {
            "durable_updates": top_level_rehome.get("durable_updates"),
            "semantic_review_observations": top_level_rehome.get(
                "semantic_review_observations"),
        }
        if _sha(submitted_payload) != planned_rehome.get(
                "misplaced_payload_sha256"):
            errors.append("top_level_rehome:payload_drift")
    if not isinstance(repairs, list):
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["round_validator_contract_repair_items_not_array"]}
    if not isinstance(durable_repairs, list):
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["round_validator_durable_repair_items_not_array"]}
    if not isinstance(scene_repairs, list):
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["round_validator_scene_repair_items_not_array"]}
    if not isinstance(observation_repairs, list):
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["round_validator_observation_repair_items_not_array"]}
    if not isinstance(unplanned_action_repairs, list):
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["round_validator_unplanned_action_repairs_not_array"]}
    for name, value in (
        ("future_knowledge_repairs", future_knowledge_repairs),
        ("durable_update_removals", durable_update_removals),
        ("durable_dependency_repairs", durable_dependency_repairs),
        ("verified_connection_repairs", verified_connection_repairs),
        ("verified_connection_removals", verified_connection_removals),
        ("missing_scene_reviews", missing_scene_reviews),
    ):
        if not isinstance(value, list):
            return {"valid": False, "classification": "schema_invalid",
                    "errors": [f"round_validator_{name}_not_array"]}
    missing_review_targets = {
        str(row.get("scene_alias")): row
        for row in repair_plan.get("missing_scene_review_targets") or []
    }
    seen_missing_reviews: set[str] = set()
    for index, item in enumerate(missing_scene_reviews):
        prefix = f"missing_scene_review:{index}"
        if (not isinstance(item, Mapping) or
                set(item) != MISSING_SCENE_REVIEW_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        alias = str(item.get("scene_alias") or "")
        target = missing_review_targets.get(alias)
        if target is None:
            errors.append(f"{prefix}:unknown_scene_alias:{alias}")
            continue
        if alias in seen_missing_reviews:
            errors.append(f"{prefix}:duplicate_scene_alias:{alias}")
        seen_missing_reviews.add(alias)
        review = item.get("scene_review")
        if not isinstance(review, Mapping) or set(review) != MODEL_SCENE_REVIEW_FIELDS:
            errors.append(f"{prefix}:scene_review_schema")
            continue
        if str(review.get("scene_alias") or "") != alias:
            errors.append(f"{prefix}:scene_alias_mismatch")
        for field, allowed in (
            ("decision", SCENE_DECISIONS),
            ("historical_connection_status", CONNECTION_STATUSES),
            ("continuity_status", CONTINUITY_STATUSES),
            ("historical_consistency_status", CONSISTENCY_STATUSES),
            ("semantic_duplicate_status", DUPLICATE_STATUSES),
            ("new_development_status", DEVELOPMENT_STATUSES),
        ):
            if review.get(field) not in allowed:
                errors.append(f"{prefix}:{field}")
        if not isinstance(review.get("future_knowledge_leak"), bool):
            errors.append(f"{prefix}:future_knowledge_leak")
        if not isinstance(review.get("scene_requires_update_commit"), bool):
            errors.append(f"{prefix}:scene_requires_update_commit")
        reason_codes = review.get("reason_codes")
        if (not isinstance(reason_codes, list) or
                any(code not in SCENE_REASON_CODES for code in reason_codes)):
            errors.append(f"{prefix}:reason_codes")
        allowed_current = set(target["allowed_current_scene_turn_refs"])
        allowed_historical = set(target["allowed_historical_turn_refs"])
        allowed_dependency_scenes = set(target["allowed_dependency_scene_refs"])
        allowed_dependency_turns = set(target["allowed_dependency_turn_refs"])
        for field, allowed, nonempty in (
            ("current_scene_supporting_turn_refs", allowed_current, True),
            ("historical_supporting_turn_refs", allowed_historical, False),
            ("dependency_context_turn_refs", allowed_dependency_turns, False),
            ("uses_scene_refs", allowed_dependency_scenes, False),
        ):
            refs = review.get(field)
            if (not isinstance(refs, list) or
                    any(not isinstance(ref, str) for ref in refs) or
                    len(set(refs)) != len(refs) or
                    any(ref not in allowed for ref in refs) or
                    (nonempty and not refs)):
                errors.append(f"{prefix}:{field}")
        scene = target["scene"]
        for role in ("primary", "secondary"):
            planned = scene.get(f"{role}_action")
            realization = review.get(f"{role}_action_realization")
            if not isinstance(planned, Mapping):
                if realization is not None:
                    errors.append(f"{prefix}:{role}_action_unplanned")
                continue
            expected = (MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS
                        if planned.get("action_mode") == "continuation"
                        else MODEL_NEW_ACTION_REALIZATION_FIELDS)
            legacy_new_fields = expected - {"progression_evidence_turn_refs"}
            if (not isinstance(realization, Mapping) or
                    (set(realization) != expected and not (
                        planned.get("action_mode") == "new" and
                        set(realization) == legacy_new_fields))):
                errors.append(f"{prefix}:{role}_action_schema")
                continue
            status = str(realization.get("status") or "")
            evidence = realization.get("evidence_turn_refs")
            if (status not in ACTION_REALIZATION_STATUSES or
                    not isinstance(evidence, list) or
                    any(not isinstance(ref, str) or ref not in allowed_current
                        for ref in evidence) or len(set(evidence)) != len(evidence) or
                    (status in _REALIZED_STATUSES and not evidence) or
                    (status not in _REALIZED_STATUSES and bool(evidence))):
                errors.append(f"{prefix}:{role}_action_evidence")
                evidence = [] if not isinstance(evidence, list) else evidence
            if planned.get("action_mode") == "continuation":
                progression = realization.get("progression_evidence_turn_refs")
                if (not isinstance(progression, list) or
                        any(not isinstance(ref, str) or ref not in set(evidence)
                            for ref in progression) or
                        len(set(progression)) != len(progression) or
                        (status in _REALIZED_STATUSES and not progression) or
                        (status not in _REALIZED_STATUSES and bool(progression))):
                    errors.append(f"{prefix}:{role}_progression_evidence")
            elif realization.get("progression_evidence_turn_refs") not in (None, []):
                errors.append(f"{prefix}:{role}_new_action_progression_evidence")
    missing_review_aliases = set(missing_review_targets) - seen_missing_reviews
    if missing_review_aliases:
        errors.append("missing_scene_review:missing_aliases:" +
                      ",".join(sorted(missing_review_aliases)))
    if len(missing_scene_reviews) != len(missing_review_targets) and not \
            missing_review_aliases:
        errors.append("missing_scene_review:item_count")
    scene_targets = {str(row.get("scene_alias")): row
                     for row in repair_plan.get("scene_reference_targets") or []}
    seen_scenes: set[str] = set()
    for index, repair in enumerate(scene_repairs):
        prefix = f"scene_reference_repair:{index}"
        preliminary_alias = str(repair.get("scene_alias") or "") \
            if isinstance(repair, Mapping) else ""
        preliminary_target = scene_targets.get(preliminary_alias)
        historical_scope = bool(preliminary_target and preliminary_target.get(
            "historical_scope_enabled"))
        repair_uses = bool(preliminary_target and preliminary_target.get(
            "repair_uses_scene_refs"))
        repair_historical = bool(
            historical_scope and preliminary_target and preliminary_target.get(
                "repair_historical_supporting_turn_refs"))
        required_shape = {"scene_alias"} | (
            {"current_scene_supporting_turn_refs"}
            if preliminary_target and preliminary_target.get(
                "repair_current_scene_supporting_turn_refs") else set()) | (
            {"dependency_context_turn_refs"}
            if preliminary_target and preliminary_target.get(
                "repair_dependency_context_turn_refs") else set()) | (
            {"uses_scene_refs"} if repair_uses else set()) | (
            {"historical_supporting_turn_refs"} if repair_historical else set())
        allowed_shapes = {frozenset(required_shape)}
        legacy_shape = _SCENE_REFERENCE_BASE_FIELDS | (
            {"uses_scene_refs"} if repair_uses else set()) | (
            {"historical_supporting_turn_refs"} if repair_historical else set())
        allowed_shapes.add(frozenset(legacy_shape))
        # Historical deterministic fixtures may resubmit frozen fields.  Live
        # repair prompts omit them, reducing drift surface and prompt size.
        if not repair_uses:
            allowed_shapes |= {shape | {"uses_scene_refs"}
                               for shape in list(allowed_shapes)}
        if historical_scope and not repair_historical:
            allowed_shapes |= {
                shape | {"historical_supporting_turn_refs"}
                for shape in list(allowed_shapes)
            }
        if (not isinstance(repair, Mapping) or
                frozenset(repair) not in allowed_shapes):
            errors.append(f"{prefix}:schema")
            continue
        alias = str(repair.get("scene_alias") or "")
        target = scene_targets.get(alias)
        if target is None:
            errors.append(f"{prefix}:unknown_scene_alias:{alias}")
            continue
        if alias in seen_scenes:
            errors.append(f"{prefix}:duplicate_scene_alias:{alias}")
        seen_scenes.add(alias)
        current_refs = repair.get(
            "current_scene_supporting_turn_refs",
            target.get("original_current_scene_supporting_turn_refs"))
        dependency_refs = repair.get(
            "dependency_context_turn_refs",
            target.get("original_dependency_context_turn_refs"))
        historical_refs = repair.get(
            "historical_supporting_turn_refs",
            target.get("original_historical_supporting_turn_refs"))
        uses_scene_refs = repair.get(
            "uses_scene_refs", target.get("original_uses_scene_refs"))
        if (not isinstance(current_refs, list) or
                any(not isinstance(ref, str) for ref in current_refs)):
            errors.append(f"{prefix}:current_scene_supporting_turn_refs")
            current_refs = []
        if (not isinstance(dependency_refs, list) or
                any(not isinstance(ref, str) for ref in dependency_refs)):
            errors.append(f"{prefix}:dependency_context_turn_refs")
            dependency_refs = []
        historical_scope_enabled = bool(target.get("historical_scope_enabled"))
        if (historical_scope_enabled and
                (not isinstance(historical_refs, list) or
                 any(not isinstance(ref, str) for ref in historical_refs))):
            errors.append(f"{prefix}:historical_supporting_turn_refs")
            historical_refs = []
        if (not isinstance(uses_scene_refs, list) or
                any(not isinstance(ref, str) for ref in uses_scene_refs)):
            errors.append(f"{prefix}:uses_scene_refs")
            uses_scene_refs = []
        if len(set(current_refs)) != len(current_refs):
            errors.append(f"{prefix}:duplicate_current_scene_turn_ref")
        if len(set(dependency_refs)) != len(dependency_refs):
            errors.append(f"{prefix}:duplicate_dependency_turn_ref")
        if (historical_scope_enabled and
                len(set(historical_refs)) != len(historical_refs)):
            errors.append(f"{prefix}:duplicate_historical_turn_ref")
        if len(set(uses_scene_refs)) != len(uses_scene_refs):
            errors.append(f"{prefix}:duplicate_uses_scene_ref")
        if any(ref not in set(target["allowed_current_scene_turn_refs"])
               for ref in current_refs):
            errors.append(f"{prefix}:current_scene_not_allowed_subset")
        if any(ref not in set(target["allowed_dependency_turn_refs"])
               for ref in dependency_refs):
            errors.append(f"{prefix}:dependency_not_allowed_subset")
        if (historical_scope_enabled and any(
                ref not in set(target.get("allowed_historical_turn_refs") or [])
                for ref in historical_refs)):
            errors.append(f"{prefix}:historical_not_allowed_subset")
        if any(ref not in set(target.get("allowed_dependency_scene_refs") or [])
               for ref in uses_scene_refs):
            errors.append(f"{prefix}:uses_scene_not_allowed_subset")
        dependency_owner = {
            str(ref).rsplit(":T", 1)[0] for ref in dependency_refs
            if isinstance(ref, str) and ":T" in ref
        }
        if not dependency_owner.issubset(set(uses_scene_refs)):
            errors.append(f"{prefix}:dependency_owner_not_declared")
        if target["repair_current_scene_supporting_turn_refs"]:
            if not current_refs:
                errors.append(f"{prefix}:current_scene_support_required")
        elif current_refs != target[
                "original_current_scene_supporting_turn_refs"]:
            errors.append(f"{prefix}:valid_current_scene_support_changed")
        if target["repair_dependency_context_turn_refs"]:
            if target["allowed_dependency_turn_refs"] and not dependency_refs:
                errors.append(f"{prefix}:dependency_support_required")
        elif dependency_refs != target["original_dependency_context_turn_refs"]:
            errors.append(f"{prefix}:valid_dependency_support_changed")
        if target.get("repair_uses_scene_refs"):
            if dependency_refs and not uses_scene_refs:
                errors.append(f"{prefix}:uses_scene_refs_required")
        elif uses_scene_refs != target.get("original_uses_scene_refs"):
            errors.append(f"{prefix}:valid_uses_scene_refs_changed")
        if target.get("repair_historical_supporting_turn_refs"):
            # The primary Validator contract accepts an empty historical support
            # list. A compact repair must not introduce a stricter evidence gate.
            pass
        elif historical_refs != target.get(
                "original_historical_supporting_turn_refs"):
            errors.append(f"{prefix}:valid_historical_support_changed")
    missing_scenes = set(scene_targets) - seen_scenes
    if missing_scenes:
        errors.append("scene_reference_repair:missing_scene_aliases:" +
                      ",".join(sorted(missing_scenes)))
    if len(scene_repairs) != len(scene_targets) and not missing_scenes:
        errors.append("scene_reference_repair:item_count")
    observation_targets = {
        str(row.get("observation_key")): row
        for row in repair_plan.get("observation_targets") or []}
    seen_observations: set[str] = set()
    for index, repair in enumerate(observation_repairs):
        prefix = f"observation_reference_repair:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != OBSERVATION_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        key = str(repair.get("observation_key") or "")
        target = observation_targets.get(key)
        if target is None:
            errors.append(f"{prefix}:unknown_observation_key:{key}")
            continue
        if key in seen_observations:
            errors.append(f"{prefix}:duplicate_observation_key:{key}")
        seen_observations.add(key)
        sources = repair.get("source_scene_refs")
        evidence = repair.get("evidence_turn_refs")
        if (not isinstance(sources, list) or not sources or
                any(not isinstance(ref, str) for ref in sources)):
            errors.append(f"{prefix}:source_scene_refs")
            sources = []
        if (not isinstance(evidence, list) or not evidence or
                any(not isinstance(ref, str) for ref in evidence)):
            errors.append(f"{prefix}:evidence_turn_refs")
            evidence = []
        if len(set(sources)) != len(sources):
            errors.append(f"{prefix}:duplicate_source_scene_ref")
        if len(set(evidence)) != len(evidence):
            errors.append(f"{prefix}:duplicate_evidence_turn_ref")
        if any(ref not in set(target["allowed_scene_refs"]) for ref in sources):
            errors.append(f"{prefix}:source_scene_not_allowed")
        allowed_evidence = {
            ref for scene_ref in sources
            for ref in target["allowed_evidence_turn_refs_by_scene"].get(
                scene_ref, [])
        }
        if any(ref not in allowed_evidence for ref in evidence):
            errors.append(f"{prefix}:evidence_not_selected_scene_subset")
    missing_observations = set(observation_targets) - seen_observations
    if missing_observations:
        errors.append("observation_reference_repair:missing_observation_keys:" +
                      ",".join(sorted(missing_observations)))
    if (len(observation_repairs) != len(observation_targets) and
            not missing_observations):
        errors.append("observation_reference_repair:item_count")
    unplanned_targets = {
        (str(row.get("scene_alias")), str(row.get("action_role"))): row
        for row in repair_plan.get("unplanned_action_targets") or []}
    seen_unplanned: set[tuple[str, str]] = set()
    for index, repair in enumerate(unplanned_action_repairs):
        prefix = f"unplanned_action_repair:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != UNPLANNED_ACTION_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        key = (str(repair.get("scene_alias") or ""),
               str(repair.get("action_role") or ""))
        if key not in unplanned_targets:
            errors.append(f"{prefix}:unknown_slot:{key[0]}:{key[1]}")
            continue
        if key in seen_unplanned:
            errors.append(f"{prefix}:duplicate_slot:{key[0]}:{key[1]}")
        seen_unplanned.add(key)
        if repair.get("replacement") is not None:
            errors.append(f"{prefix}:replacement_must_be_null")
    missing_unplanned = set(unplanned_targets) - seen_unplanned
    if missing_unplanned:
        errors.append("unplanned_action_repair:missing_slots:" + ",".join(
            f"{alias}.{role}" for alias, role in sorted(missing_unplanned)))
    if (len(unplanned_action_repairs) != len(unplanned_targets) and
            not missing_unplanned):
        errors.append("unplanned_action_repair:item_count")
    future_targets = {str(row.get("scene_alias")): row
                      for row in repair_plan.get("future_knowledge_targets") or []}
    seen_future: set[str] = set()
    for index, repair in enumerate(future_knowledge_repairs):
        prefix = f"future_knowledge_repair:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != FUTURE_KNOWLEDGE_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        alias = str(repair.get("scene_alias") or "")
        if alias not in future_targets:
            errors.append(f"{prefix}:unknown_scene_alias:{alias}")
            continue
        if alias in seen_future:
            errors.append(f"{prefix}:duplicate_scene_alias:{alias}")
        seen_future.add(alias)
        if not isinstance(repair.get("value"), bool):
            errors.append(f"{prefix}:value_not_boolean")
    missing_future = set(future_targets) - seen_future
    if missing_future:
        errors.append("future_knowledge_repair:missing_scenes:" +
                      ",".join(sorted(missing_future)))
    removal_targets = {str(row.get("candidate_update_key")): row
                       for row in repair_plan.get("durable_removal_targets") or []}
    seen_removals: set[str] = set()
    for index, repair in enumerate(durable_update_removals):
        prefix = f"durable_update_removal:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != DURABLE_REMOVAL_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        key = str(repair.get("candidate_update_key") or "")
        if key not in removal_targets:
            errors.append(f"{prefix}:unknown_candidate_update_key:{key}")
            continue
        if key in seen_removals:
            errors.append(f"{prefix}:duplicate_candidate_update_key:{key}")
        seen_removals.add(key)
        if repair.get("replacement") is not None:
            errors.append(f"{prefix}:replacement_must_be_null")
    missing_removals = set(removal_targets) - seen_removals
    if missing_removals:
        errors.append("durable_update_removal:missing_keys:" +
                      ",".join(sorted(missing_removals)))
    dependency_targets = {
        str(row.get("candidate_update_key")): row
        for row in repair_plan.get("durable_dependency_targets") or []}
    seen_dependencies: set[str] = set()
    for index, repair in enumerate(durable_dependency_repairs):
        prefix = f"durable_dependency_repair:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != DURABLE_DEPENDENCY_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        key = str(repair.get("candidate_update_key") or "")
        target = dependency_targets.get(key)
        if target is None:
            errors.append(f"{prefix}:unknown_candidate_update_key:{key}")
            continue
        if key in seen_dependencies:
            errors.append(f"{prefix}:duplicate_candidate_update_key:{key}")
        seen_dependencies.add(key)
        dependencies = repair.get("depends_on_candidate_keys")
        if (not isinstance(dependencies, list) or
                any(not isinstance(dep, str) for dep in dependencies)):
            errors.append(f"{prefix}:depends_on_candidate_keys")
            continue
        if len(set(dependencies)) != len(dependencies):
            errors.append(f"{prefix}:duplicate_dependency_key")
        if any(dep not in set(target["allowed_dependency_keys"])
               for dep in dependencies):
            errors.append(f"{prefix}:dependency_not_allowed")
    missing_dependencies = set(dependency_targets) - seen_dependencies
    if missing_dependencies:
        errors.append("durable_dependency_repair:missing_keys:" +
                      ",".join(sorted(missing_dependencies)))
    connection_targets = {
        str(row.get("candidate_update_key")): row
        for row in repair_plan.get("verified_connection_targets") or []}
    seen_connections: set[str] = set()
    for index, repair in enumerate(verified_connection_repairs):
        prefix = f"verified_connection_repair:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != VERIFIED_CONNECTION_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        key = str(repair.get("candidate_update_key") or "")
        target = connection_targets.get(key)
        if target is None:
            errors.append(f"{prefix}:unknown_candidate_update_key:{key}")
            continue
        if key in seen_connections:
            errors.append(f"{prefix}:duplicate_candidate_update_key:{key}")
        seen_connections.add(key)
        connection = repair.get("verified_connection")
        if (not isinstance(connection, Mapping) or
                set(connection) != VERIFIED_CONNECTION_FIELDS):
            errors.append(f"{prefix}:verified_connection_schema")
            continue
        if connection.get("verification_status") not in set(
                target["allowed_verification_statuses"]):
            errors.append(f"{prefix}:verification_status")
        kind = str(connection.get("connection_kind") or "")
        if kind not in set(target["allowed_connection_kinds"]):
            errors.append(f"{prefix}:connection_kind")
        refs = connection.get("evidence_turn_refs")
        if (not isinstance(refs, list) or not refs or
                any(not isinstance(ref, str) for ref in refs)):
            errors.append(f"{prefix}:evidence_turn_refs")
            refs = []
        if len(set(refs)) != len(refs):
            errors.append(f"{prefix}:duplicate_evidence_turn_ref")
        if any(ref not in set(target["allowed_evidence_turn_refs"]) for ref in refs):
            errors.append(f"{prefix}:evidence_not_source_action_subset")
        if not isinstance(connection.get("reason"), str) or not str(
                connection.get("reason") or "").strip():
            errors.append(f"{prefix}:reason")
        relation_type = connection.get("relation_type")
        if kind == "explicit_relation":
            if relation_type not in set(target["allowed_relation_types"]):
                errors.append(f"{prefix}:relation_type")
        elif relation_type is not None:
            errors.append(f"{prefix}:relation_type_forbidden")
        original = target.get("original_verified_connection")
        if isinstance(original, Mapping):
            for field in target.get("preserve_fields") or []:
                if connection.get(field) != original.get(field):
                    errors.append(f"{prefix}:preserved_field_changed:{field}")
    missing_connections = set(connection_targets) - seen_connections
    if missing_connections:
        errors.append("verified_connection_repair:missing_keys:" +
                      ",".join(sorted(missing_connections)))
    if (len(verified_connection_repairs) != len(connection_targets) and
            not missing_connections):
        errors.append("verified_connection_repair:item_count")
    connection_removal_targets = {
        str(row.get("candidate_update_key")): row
        for row in repair_plan.get("verified_connection_removal_targets") or []}
    seen_connection_removals: set[str] = set()
    for index, repair in enumerate(verified_connection_removals):
        prefix = f"verified_connection_removal:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != VERIFIED_CONNECTION_REMOVAL_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        key = str(repair.get("candidate_update_key") or "")
        if key not in connection_removal_targets:
            errors.append(f"{prefix}:unknown_candidate_update_key:{key}")
            continue
        if key in seen_connection_removals:
            errors.append(f"{prefix}:duplicate_candidate_update_key:{key}")
        seen_connection_removals.add(key)
        if repair.get("replacement") is not None:
            errors.append(f"{prefix}:replacement_must_be_null")
    missing_connection_removals = (
        set(connection_removal_targets) - seen_connection_removals)
    if missing_connection_removals:
        errors.append("verified_connection_removal:missing_keys:" +
                      ",".join(sorted(missing_connection_removals)))
    targets = {str(row.get("action_ref")): row
               for row in repair_plan.get("targets") or []}
    seen: set[str] = set()
    for index, repair in enumerate(repairs):
        prefix = f"contract_repair:{index}"
        preliminary_ref = str(repair.get("action_ref") or "") \
            if isinstance(repair, Mapping) else ""
        preliminary_target = targets.get(preliminary_ref)
        legacy_fields = (
            CONTINUATION_ACTION_REPAIR_ITEM_FIELDS
            if preliminary_target is not None and
            preliminary_target.get("action_mode") == "continuation"
            else NEW_ACTION_REPAIR_ITEM_FIELDS
        )
        sparse_fields = {"action_ref"} | (
            {"evidence_turn_refs"}
            if preliminary_target is not None and preliminary_target.get(
                "evidence_turn_refs_requires_repair") else set()) | (
            {"progression_evidence_turn_refs"}
            if preliminary_target is not None and
            preliminary_target.get("action_mode") == "continuation" and
            preliminary_target.get(
                "progression_evidence_turn_refs_requires_repair") else set())
        if (not isinstance(repair, Mapping) or
                frozenset(repair) not in {legacy_fields, frozenset(sparse_fields)}):
            errors.append(f"{prefix}:schema")
            continue
        action_ref = str(repair.get("action_ref") or "")
        if action_ref not in targets:
            errors.append(f"{prefix}:unknown_action_ref:{action_ref}")
            continue
        if action_ref in seen:
            errors.append(f"{prefix}:duplicate_action_ref:{action_ref}")
        seen.add(action_ref)
        action_refs = repair.get(
            "evidence_turn_refs", targets[action_ref]["original_evidence_turn_refs"])
        if (not isinstance(action_refs, list) or
                any(not isinstance(ref, str) for ref in action_refs)):
            errors.append(f"{prefix}:evidence_turn_refs")
            action_refs = []
        elif len(set(action_refs)) != len(action_refs):
            errors.append(f"{prefix}:duplicate_action_evidence_turn_ref")
        allowed_action_refs = set(targets[action_ref]["allowed_evidence_turn_refs"])
        if any(ref not in allowed_action_refs for ref in action_refs):
            errors.append(f"{prefix}:action_evidence_not_scene_subset")
        status = targets[action_ref]["status"]
        if status in _REALIZED_STATUSES and not action_refs:
            errors.append(f"{prefix}:action_evidence_required")
        if status not in _REALIZED_STATUSES and action_refs:
            errors.append(f"{prefix}:action_evidence_for_nonrealized")
        if (not targets[action_ref]["evidence_turn_refs_requires_repair"] and
                action_refs != targets[action_ref]["original_evidence_turn_refs"]):
            errors.append(f"{prefix}:valid_action_evidence_changed")
        if targets[action_ref].get("action_mode") == "continuation":
            refs = repair.get(
                "progression_evidence_turn_refs",
                targets[action_ref]["original_progression_evidence_turn_refs"])
            if (not isinstance(refs, list) or
                    any(not isinstance(ref, str) for ref in refs)):
                errors.append(f"{prefix}:progression_evidence_turn_refs")
                continue
            if len(set(refs)) != len(refs):
                errors.append(f"{prefix}:duplicate_evidence_turn_ref")
            allowed = set(action_refs)
            if any(ref not in allowed for ref in refs):
                errors.append(f"{prefix}:evidence_not_candidate_subset")
            if status in _REALIZED_STATUSES and not refs:
                errors.append(f"{prefix}:progression_evidence_required")
            if status not in _REALIZED_STATUSES and refs:
                errors.append(f"{prefix}:evidence_for_nonrealized")
            if (not targets[action_ref][
                    "progression_evidence_turn_refs_requires_repair"] and
                    refs != targets[action_ref][
                        "original_progression_evidence_turn_refs"]):
                errors.append(f"{prefix}:valid_progression_evidence_changed")
    missing = set(targets) - seen
    extra_count = len(repairs) != len(targets)
    if missing:
        errors.append("contract_repair:missing_action_refs:" + ",".join(sorted(missing)))
    if extra_count and not missing:
        errors.append("contract_repair:item_count")

    durable_targets = {str(row.get("candidate_update_key")): row
                       for row in repair_plan.get("durable_update_targets") or []}
    seen_updates: set[str] = set()
    for index, repair in enumerate(durable_repairs):
        prefix = f"durable_contract_repair:{index}"
        if (not isinstance(repair, Mapping) or
                set(repair) != DURABLE_REPAIR_ITEM_FIELDS):
            errors.append(f"{prefix}:schema")
            continue
        update_key = str(repair.get("candidate_update_key") or "")
        if update_key not in durable_targets:
            errors.append(f"{prefix}:unknown_candidate_update_key:{update_key}")
            continue
        if update_key in seen_updates:
            errors.append(f"{prefix}:duplicate_candidate_update_key:{update_key}")
        seen_updates.add(update_key)
        refs = repair.get("evidence_turn_refs")
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            errors.append(f"{prefix}:evidence_turn_refs")
            continue
        if not refs:
            errors.append(f"{prefix}:evidence_required")
        if len(set(refs)) != len(refs):
            errors.append(f"{prefix}:duplicate_evidence_turn_ref")
        allowed = set(durable_targets[update_key]["allowed_evidence_turn_refs"])
        if any(ref not in allowed for ref in refs):
            errors.append(f"{prefix}:evidence_not_source_action_subset")
    missing_updates = set(durable_targets) - seen_updates
    if missing_updates:
        errors.append("durable_contract_repair:missing_candidate_update_keys:" +
                      ",".join(sorted(missing_updates)))
    if len(durable_repairs) != len(durable_targets) and not missing_updates:
        errors.append("durable_contract_repair:item_count")
    classification = ("valid" if not errors else
                      "reference_invalid" if any(
                          "action_ref" in row or "candidate_subset" in row or
                          "candidate_update_key" in row or "source_action_subset" in row
                          for row in errors) else "schema_invalid")
    return {"valid": not errors, "classification": classification,
            "errors": errors}


def apply_round_validator_contract_repair(
    *, base_model_output: Mapping[str, Any], repair_output: Any,
    repair_plan: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_round_validator_contract_repair_output(
        repair_output, repair_plan=repair_plan)
    if not validation["valid"]:
        return {"valid": False, "errors": validation["errors"],
                "classification": validation["classification"],
                "reconstructed_output": None}
    if _sha(base_model_output) != repair_plan.get("base_model_output_sha256"):
        return {"valid": False, "errors": ["contract_repair_base_output_drift"],
                "classification": "schema_invalid", "reconstructed_output": None}

    grouped_output, unpack_errors = _unpack_round_validator_contract_repair_output(
        repair_output)
    if grouped_output is None or unpack_errors:
        return {"valid": False, "errors": unpack_errors,
                "classification": "schema_invalid", "reconstructed_output": None}
    grouped_output, selection_errors = _resolve_indexed_historical_selections(
        grouped_output, repair_plan=repair_plan)
    if selection_errors:
        return {"valid": False, "errors": selection_errors,
                "classification": "schema_invalid", "reconstructed_output": None}

    reconstructed = deepcopy(dict(base_model_output))
    planned_rehome = repair_plan.get("top_level_rehome")
    if isinstance(planned_rehome, Mapping):
        indices = (list(planned_rehome["misplaced_scene_review_indices"])
                   if "misplaced_scene_review_indices" in planned_rehome else
                   [int(planned_rehome["misplaced_scene_review_index"])])
        reviews = reconstructed.get("scene_reviews")
        if (not isinstance(reviews, list) or len(set(indices)) != len(indices) or
                any(not isinstance(index, int) or not 0 <= index < len(reviews)
                    for index in indices)):
            return {"valid": False, "errors": ["top_level_rehome:index_invalid"],
                    "classification": "schema_invalid",
                    "reconstructed_output": None}
        source_rows = [{"index": index, "value": deepcopy(reviews[index])}
                       for index in indices]
        if _sha(source_rows) != planned_rehome.get(
                "misplaced_source_rows_sha256"):
            return {"valid": False, "errors": ["top_level_rehome:source_drift"],
                    "classification": "schema_invalid",
                    "reconstructed_output": None}
        for index in sorted(indices, reverse=True):
            reviews.pop(index)
        submitted = grouped_output["top_level_rehome"]
        reconstructed["durable_updates"] = deepcopy(submitted["durable_updates"])
        reconstructed["semantic_review_observations"] = deepcopy(
            submitted["semantic_review_observations"])
    missing_review_items = grouped_output["missing_scene_reviews"]
    if missing_review_items:
        reconstructed_reviews = reconstructed.get("scene_reviews")
        if not isinstance(reconstructed_reviews, list):
            return {"valid": False, "errors": ["missing_scene_review:base_not_list"],
                    "classification": "schema_invalid",
                    "reconstructed_output": None}
        reconstructed_reviews.extend(
            deepcopy(item["scene_review"]) for item in missing_review_items)
        order = {alias: index for index, alias in enumerate(
            repair_plan.get("expected_scene_alias_order") or [])}
        reconstructed_reviews.sort(
            key=lambda row: order.get(str(row.get("scene_alias") or ""), len(order))
            if isinstance(row, Mapping) else len(order))
    review_locations = {
        str(row.get("scene_alias") or ""): row
        for row in reconstructed.get("scene_reviews") or []
        if isinstance(row, dict)
    }
    for item in grouped_output["scene_reference_repairs"]:
        location = review_locations[str(item["scene_alias"])]
        if "historical_supporting_turn_refs" in item:
            location["historical_supporting_turn_refs"] = deepcopy(
                item["historical_supporting_turn_refs"])
        if "current_scene_supporting_turn_refs" in item:
            location["current_scene_supporting_turn_refs"] = deepcopy(
                item["current_scene_supporting_turn_refs"])
        if "dependency_context_turn_refs" in item:
            location["dependency_context_turn_refs"] = deepcopy(
                item["dependency_context_turn_refs"])
        if "uses_scene_refs" in item:
            location["uses_scene_refs"] = deepcopy(item["uses_scene_refs"])
    observation_locations = {
        str(row.get("observation_key") or ""): row
        for row in reconstructed.get("semantic_review_observations") or []
        if isinstance(row, dict)
    }
    for item in grouped_output["observation_reference_repairs"]:
        location = observation_locations[str(item["observation_key"])]
        location["source_scene_refs"] = deepcopy(item["source_scene_refs"])
        location["evidence_turn_refs"] = deepcopy(item["evidence_turn_refs"])
    for item in grouped_output["unplanned_action_repairs"]:
        review_locations[str(item["scene_alias"])][
            f"{item['action_role']}_action_realization"] = None
    for item in grouped_output["future_knowledge_repairs"]:
        review_locations[str(item["scene_alias"])]["future_knowledge_leak"] = (
            item["value"])
    durable_updates_by_key = {
        str(row.get("candidate_update_key") or ""): row
        for row in reconstructed.get("durable_updates") or []
        if isinstance(row, dict)
    }
    for item in grouped_output["durable_dependency_repairs"]:
        durable_updates_by_key[str(item["candidate_update_key"])][
            "depends_on_candidate_keys"] = deepcopy(
                item["depends_on_candidate_keys"])
    removal_keys = {str(item["candidate_update_key"])
                    for item in grouped_output["durable_update_removals"]}
    if removal_keys:
        reconstructed["durable_updates"] = [
            row for row in reconstructed.get("durable_updates") or []
            if str(row.get("candidate_update_key") or "") not in removal_keys]
    locations: dict[str, dict[str, Any]] = {}
    targets = {str(row.get("action_ref")): row
               for row in repair_plan.get("targets") or []}
    reviews = reconstructed.get("scene_reviews") or []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        alias = str(review.get("scene_alias") or "")
        for role in ("primary", "secondary"):
            for action_ref, target in targets.items():
                if target["scene_alias"] == alias and target["action_role"] == role:
                    value = review.get(f"{role}_action_realization")
                    if isinstance(value, dict):
                        locations[action_ref] = value
    repair_items = grouped_output["action_realization_repairs"]
    for item in repair_items:
        location = locations[str(item["action_ref"])]
        if "evidence_turn_refs" in item:
            location["evidence_turn_refs"] = deepcopy(item["evidence_turn_refs"])
        target = targets[str(item["action_ref"])]
        if target.get("action_mode") == "continuation" and (
                "progression_evidence_turn_refs" in item or
                target["progression_rehome_from_scene"]):
            submitted_progression = (
                item.get("progression_evidence_turn_refs")
                if "progression_evidence_turn_refs" in item else
                review_locations[target["scene_alias"]].get(
                    "progression_evidence_turn_refs"))
            location["progression_evidence_turn_refs"] = deepcopy(
                submitted_progression)
        elif target.get("action_mode") != "continuation":
            location.pop("progression_evidence_turn_refs", None)
        if target["progression_rehome_from_scene"]:
            review_locations[target["scene_alias"]].pop(
                "progression_evidence_turn_refs", None)
    durable_locations = {
        str(row.get("candidate_update_key") or ""): row
        for row in reconstructed.get("durable_updates") or []
        if isinstance(row, dict)
    }
    for item in grouped_output["durable_update_evidence_repairs"]:
        location = durable_locations[str(item["candidate_update_key"])]
        location["evidence_turn_refs"] = deepcopy(item["evidence_turn_refs"])
        # The v5 connected-person envelope reviews the same evidence-bearing
        # narrative claim.  An explicit contract-only evidence resubmission
        # therefore updates both exact reference fields; no local turn is
        # selected or substituted here.
        if isinstance(location.get("verified_connection"), dict):
            location["verified_connection"]["evidence_turn_refs"] = deepcopy(
                item["evidence_turn_refs"])
    for item in grouped_output["verified_connection_repairs"]:
        durable_locations[str(item["candidate_update_key"])][
            "verified_connection"] = deepcopy(item["verified_connection"])
    for item in grouped_output["verified_connection_removals"]:
        durable_locations[str(item["candidate_update_key"])].pop(
            "verified_connection", None)

    # Prove that deleting the newly inserted members restores the exact base object.
    frozen_projection = deepcopy(reconstructed)
    frozen_reviews_by_alias = {
        str(row.get("scene_alias") or ""): row
        for row in frozen_projection.get("scene_reviews") or []
        if isinstance(row, dict)
    }
    for target in repair_plan.get("scene_reference_targets") or []:
        review = frozen_reviews_by_alias[target["scene_alias"]]
        review["historical_supporting_turn_refs"] = deepcopy(
            target["original_historical_supporting_turn_refs"])
        review["current_scene_supporting_turn_refs"] = deepcopy(
            target["original_current_scene_supporting_turn_refs"])
        review["dependency_context_turn_refs"] = deepcopy(
            target["original_dependency_context_turn_refs"])
        review["uses_scene_refs"] = deepcopy(
            target.get("original_uses_scene_refs"))
    for review in frozen_projection.get("scene_reviews") or []:
        if not isinstance(review, dict):
            continue
        alias = str(review.get("scene_alias") or "")
        for role in ("primary", "secondary"):
            if any(target["scene_alias"] == alias and
                   target["action_role"] == role for target in targets.values()):
                value = review.get(f"{role}_action_realization")
                if isinstance(value, dict):
                    target = next(
                        target for target in targets.values()
                        if target["scene_alias"] == alias and
                        target["action_role"] == role)
                    value["evidence_turn_refs"] = deepcopy(
                        target["original_evidence_turn_refs"])
                    if target.get("original_progression_present"):
                        value["progression_evidence_turn_refs"] = deepcopy(
                            target["original_progression_evidence_turn_refs"])
                    else:
                        value.pop("progression_evidence_turn_refs", None)
                    if target["progression_rehome_from_scene"]:
                        review["progression_evidence_turn_refs"] = deepcopy(
                            target["original_progression_evidence_turn_refs"])
    original_durable_evidence = {
        str(row.get("candidate_update_key")): deepcopy(
            row.get("original_evidence_turn_refs"))
        for row in repair_plan.get("durable_update_targets") or []
    }
    original_connection_evidence = {
        str(row.get("candidate_update_key")): deepcopy(
            row.get("original_verified_connection_evidence_turn_refs"))
        for row in repair_plan.get("durable_update_targets") or []
    }
    for update in frozen_projection.get("durable_updates") or []:
        if (isinstance(update, dict) and
                str(update.get("candidate_update_key") or "") in
                original_durable_evidence):
            update["evidence_turn_refs"] = deepcopy(original_durable_evidence[
                str(update["candidate_update_key"])])
            if isinstance(update.get("verified_connection"), dict):
                update["verified_connection"]["evidence_turn_refs"] = deepcopy(
                    original_connection_evidence[
                        str(update["candidate_update_key"])]
                )
    frozen_updates_by_key = {
        str(row.get("candidate_update_key") or ""): row
        for row in frozen_projection.get("durable_updates") or []
        if isinstance(row, dict)
    }
    for target in repair_plan.get("verified_connection_targets") or []:
        update = frozen_updates_by_key[target["candidate_update_key"]]
        original = target.get("original_verified_connection")
        if original is None:
            update.pop("verified_connection", None)
        else:
            update["verified_connection"] = deepcopy(original)
    for target in repair_plan.get("verified_connection_removal_targets") or []:
        frozen_updates_by_key[target["candidate_update_key"]][
            "verified_connection"] = deepcopy(
                target["original_verified_connection"])
    frozen_observations = {
        str(row.get("observation_key") or ""): row
        for row in frozen_projection.get("semantic_review_observations") or []
        if isinstance(row, dict)
    }
    for target in repair_plan.get("observation_targets") or []:
        observation = frozen_observations[target["observation_key"]]
        observation["source_scene_refs"] = deepcopy(
            target["original_source_scene_refs"])
        observation["evidence_turn_refs"] = deepcopy(
            target["original_evidence_turn_refs"])
    for target in repair_plan.get("unplanned_action_targets") or []:
        frozen_reviews_by_alias[target["scene_alias"]][
            f"{target['action_role']}_action_realization"] = deepcopy(
                target["original_value"])
    for target in repair_plan.get("future_knowledge_targets") or []:
        frozen_reviews_by_alias[target["scene_alias"]]["future_knowledge_leak"] = (
            deepcopy(target["original_value"]))
    frozen_updates = frozen_projection.get("durable_updates")
    if not isinstance(frozen_updates, list):
        frozen_updates = []
        frozen_projection["durable_updates"] = frozen_updates
    frozen_updates_by_key = {
        str(row.get("candidate_update_key") or ""): row
        for row in frozen_updates if isinstance(row, dict)
    }
    for target in repair_plan.get("durable_dependency_targets") or []:
        if target["candidate_update_key"] in frozen_updates_by_key:
            frozen_updates_by_key[target["candidate_update_key"]][
                "depends_on_candidate_keys"] = deepcopy(
                    target["original_depends_on_candidate_keys"])
    for target in sorted(repair_plan.get("durable_removal_targets") or [],
                         key=lambda row: int(row["original_index"])):
        frozen_updates.insert(int(target["original_index"]),
                              deepcopy(target["original_value"]))
    missing_review_aliases = {
        str(row.get("scene_alias"))
        for row in repair_plan.get("missing_scene_review_targets") or []
    }
    if missing_review_aliases:
        frozen_projection["scene_reviews"] = [
            row for row in frozen_projection.get("scene_reviews") or []
            if not (isinstance(row, Mapping) and
                    str(row.get("scene_alias") or "") in missing_review_aliases)
        ]
    if isinstance(planned_rehome, Mapping):
        frozen_projection.pop("durable_updates")
        frozen_projection.pop("semantic_review_observations")
        for source_row in planned_rehome["misplaced_source_rows"]:
            frozen_projection["scene_reviews"].insert(
                int(source_row["index"]), deepcopy(source_row["value"]))
    if _sha(frozen_projection) != repair_plan.get("base_model_output_sha256"):
        return {"valid": False, "errors": ["contract_repair_non_target_drift"],
                "classification": "schema_invalid", "reconstructed_output": None}
    return {
        "valid": True, "errors": [], "classification": "valid",
        "reconstructed_output": reconstructed,
        "base_model_output_sha256": repair_plan["base_model_output_sha256"],
        "reconstructed_output_sha256": _sha(reconstructed),
        "non_target_fields_unchanged": True,
    }


__all__ = [
    "ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11",
    "ROUND_VALIDATOR_CONTRACT_REPAIR_VERSION_V11",
    "apply_round_validator_contract_repair",
    "build_round_validator_contract_repair_plan",
    "build_round_validator_contract_repair_prompt",
    "pack_round_validator_contract_repair_output",
    "validate_round_validator_contract_repair_output",
]
