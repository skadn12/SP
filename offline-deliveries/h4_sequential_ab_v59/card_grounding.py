"""Mechanical source-card to planner mapping checks for v36.

This module validates actor bindings and the current development references used
by selected source steps/dependencies. It deliberately does not assign or
change compatible/revise judgments. Semantic review remains a model task.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _err(code: str, path: str, actual: Any, **details: Any) -> dict[str, Any]:
    return {"code": code, "path": path, "actual": deepcopy(actual), **deepcopy(details)}


def _guidance(card: dict[str, Any]) -> tuple[dict[str, dict], dict[int, dict], dict[str, str]]:
    steps = {row["step_id"]: row for row in card.get("steps", []) if isinstance(row, dict) and isinstance(row.get("step_id"), str)}
    dependencies = {index: row for index, row in enumerate(card.get("dependencies", []), 1) if isinstance(row, dict)}
    actors = {row["actor_id"]: row.get("role", "") for row in card.get("actors", []) if isinstance(row, dict) and isinstance(row.get("actor_id"), str)}
    return steps, dependencies, actors


def _links(scene: dict[str, Any]) -> list[dict[str, Any]]:
    links = scene.get("card_development_links")
    return links if isinstance(links, list) else []


def _selected_maps(scene: dict[str, Any]) -> tuple[dict[str, list[int]], dict[int, list[int]]]:
    steps: dict[str, list[int]] = {}
    deps: dict[int, list[int]] = {}
    for link in _links(scene):
        if not isinstance(link, dict):
            continue
        devs = link.get("development_indices")
        if not isinstance(devs, list):
            continue
        for step in link.get("card_step_ids", []):
            if isinstance(step, str):
                steps.setdefault(step, []).extend(x for x in devs if type(x) is int)
        for dep in link.get("dependency_indices", []):
            if type(dep) is int:
                deps.setdefault(dep, []).extend(x for x in devs if type(x) is int)
    return steps, deps


def expected_card_grounding(scene: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Dereference source actors, selected work and dependency endpoints.

    The returned object is input binding only. It contains no semantic verdict.
    """
    if not isinstance(cards, list) or len(cards) != 1 or not isinstance(cards[0], dict):
        raise ValueError("card_grounding_one_card_required")
    card = cards[0]
    from planner_contract import validate_mapping, MappingValidationError
    validate_mapping(scene, card)
    errors = collect_mapping_grounding_errors(scene, card)
    if errors:
        raise MappingValidationError(errors)
    steps, dependencies, actors = _guidance(card)
    step_maps, dep_maps = _selected_maps(scene)
    binding_values = scene.get("card_actor_bindings", [])
    bindings = {row.get("actor_id"): row.get("participant") for row in binding_values
                if isinstance(row, dict) and isinstance(row.get("actor_id"), str)} if isinstance(binding_values, list) else {}
    selected_actor_ids: set[str] = set()
    for step_id in step_maps:
        if step_id in steps:
            selected_actor_ids.add(steps[step_id].get("actor_id"))
    for dep_id in dep_maps:
        dep = dependencies.get(dep_id)
        if dep:
            for step_id in (dep.get("before"), dep.get("after")):
                if step_id in steps:
                    selected_actor_ids.add(steps[step_id].get("actor_id"))
    actor_rows = [{"actor_id": actor_id, "role": actors.get(actor_id, ""), "participant": bindings.get(actor_id)}
                  for actor_id in sorted(x for x in selected_actor_ids if isinstance(x, str))]
    step_rows = []
    for step_id, devs in step_maps.items():
        step = steps.get(step_id)
        if not step:
            continue
        step_rows.append({"kind": "step", "id": step_id, "actor_id": step.get("actor_id"),
                          "actor_role": actors.get(step.get("actor_id"), ""),
                          "participant": bindings.get(step.get("actor_id")),
                          "action": step.get("action"), "state": step.get("state"),
                          "development_indices": sorted(set(devs))})
    dep_rows = []
    for dep_id, link_devs in dep_maps.items():
        dep = dependencies.get(dep_id)
        if not dep:
            continue
        before = dep.get("before"); after = dep.get("after")
        dep_rows.append({"kind": "dependency", "id": dep_id, "relation": dep.get("relation"),
                         "before_step_id": before, "after_step_id": after,
                         "before_development_indices": sorted(set(step_maps.get(before, []))),
                         "after_development_indices": sorted(set(step_maps.get(after, []))),
                         "development_indices": sorted(set(link_devs))})
    return {"actors": actor_rows, "steps": step_rows, "dependencies": dep_rows}


def collect_mapping_grounding_errors(scene: Any, card: Any) -> list[dict[str, Any]]:
    """Return structural grounding errors; no judgment is inferred."""
    errors: list[dict[str, Any]] = []
    if not isinstance(scene, dict) or not isinstance(card, dict):
        return [_err("card_grounding_input_invalid", "scene/card", {"scene": scene, "card": card})]
    from planner_contract import collect_mapping_errors
    reference_errors = collect_mapping_errors(scene, card)
    if reference_errors:
        return reference_errors
    source_actors = card.get("actors")
    if (not isinstance(source_actors, list)
            or any(not isinstance(row, dict) or not isinstance(row.get("actor_id"), str)
                   or not row["actor_id"].strip() for row in source_actors)
            or len({row["actor_id"] for row in source_actors}) != len(source_actors)):
        return [_err("card_source_actor_ids_invalid", "card.actors", source_actors)]
    actor_ids = {row["actor_id"] for row in source_actors}
    for index, step in enumerate(card["steps"]):
        actor_id = step.get("actor_id")
        if not isinstance(actor_id, str) or actor_id not in actor_ids:
            errors.append(_err("card_source_step_actor_invalid", f"card.steps[{index}].actor_id", actor_id,
                               allowed=sorted(actor_ids)))
    if errors:
        return errors
    steps, deps, actors = _guidance(card)
    expected_step_maps, expected_dep_maps = _selected_maps(scene)
    if "card_actor_bindings" not in scene:
        errors.append(_err("card_actor_bindings_required", "scene.card_actor_bindings", None))
        return errors
    bindings = scene.get("card_actor_bindings")
    if not isinstance(bindings, list):
        errors.append(_err("card_actor_bindings_not_array", "scene.card_actor_bindings", bindings))
        return errors
    selected_actor_ids: set[str] = set()
    for step_id in expected_step_maps:
        if step_id in steps:
            selected_actor_ids.add(steps[step_id].get("actor_id"))
    for dep_id in expected_dep_maps:
        dep = deps.get(dep_id)
        if dep:
            for step_id in (dep.get("before"), dep.get("after")):
                if step_id in steps:
                    selected_actor_ids.add(steps[step_id].get("actor_id"))
    observed: dict[str, str] = {}
    for index, row in enumerate(bindings):
        path = f"scene.card_actor_bindings[{index}]"
        if not isinstance(row, dict) or set(row) != {"actor_id", "participant"}:
            errors.append(_err("card_actor_binding_fields_invalid", path, row,
                               allowed=["actor_id", "participant"])); continue
        actor_id, participant = row.get("actor_id"), row.get("participant")
        if not isinstance(actor_id, str) or actor_id not in selected_actor_ids:
            errors.append(_err("card_actor_binding_actor_invalid", path + ".actor_id", actor_id,
                               allowed=sorted(x for x in selected_actor_ids if isinstance(x, str))))
            continue
        if not isinstance(participant, str) or not participant.strip():
            errors.append(_err("card_actor_binding_participant_invalid", path + ".participant", participant))
        if actor_id in observed:
            errors.append(_err("card_actor_binding_duplicate", path + ".actor_id", actor_id))
        observed[actor_id] = participant
    selected_actor_ids = {x for x in selected_actor_ids if isinstance(x, str)}
    if set(observed) != selected_actor_ids:
        errors.append(_err("card_actor_binding_coverage_invalid", "scene.card_actor_bindings", sorted(observed),
                           missing=sorted(selected_actor_ids - set(observed))))
    for step_id, devs in expected_step_maps.items():
        if step_id not in steps:
            continue
        if not devs:
            errors.append(_err("card_step_development_evidence_missing", f"scene.card_development_links.{step_id}", devs))
        if any(type(x) is not int or x < 1 or x > len(scene.get("developments", [])) for x in devs):
            errors.append(_err("card_step_development_evidence_invalid", f"scene.card_development_links.{step_id}", devs,
                               allowed=list(range(1, len(scene.get("developments", [])) + 1))))
    for dep_id, devs in expected_dep_maps.items():
        dep = deps.get(dep_id)
        if not dep:
            continue
        missing = [step_id for step_id in (dep.get("before"), dep.get("after")) if not expected_step_maps.get(step_id)]
        if missing:
            errors.append(_err("card_dependency_endpoint_mapping_missing", f"scene.card_development_links.dependency[{dep_id}]", missing,
                               dependency=dep))
        if not devs:
            errors.append(_err("card_dependency_development_evidence_missing", f"scene.card_development_links.dependency[{dep_id}]", devs))
    return errors


def validate_card_grounding_checks(value: Any, scene: dict[str, Any], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate scanner evidence indices against the current plan mapping.

    Rows retain the existing verdict/reason/correction fields; this function
    checks only identity and evidence references.
    """
    errors = []
    try:
        expected = expected_card_grounding(scene, cards)
    except (ValueError, TypeError, KeyError) as exc:
        return [_err("card_grounding_input_invalid", "card_plan_checks", str(exc))]
    expected_rows = {(row["kind"], row["id"]): row for row in [*expected["steps"], *expected["dependencies"]]}
    if not isinstance(value, list):
        return [_err("card_plan_checks_grounding_not_array", "card_plan_checks", value)]
    observed = set()
    judgments = {}
    for row in value:
        if isinstance(row, dict) and row.get("kind") == "step" and isinstance(row.get("id"), str):
            judgments[row["id"]] = row.get("verdict")
    for index, row in enumerate(value):
        path = f"card_plan_checks[{index}]"
        if not isinstance(row, dict):
            errors.append(_err("card_plan_check_grounding_row_invalid", path, row)); continue
        if not isinstance(row.get("kind"), str) or type(row.get("id")) not in (str, int):
            errors.append(_err("card_plan_check_grounding_identity_invalid", path, row)); continue
        identity = (row.get("kind"), row.get("id"))
        if identity not in expected_rows:
            errors.append(_err("card_plan_check_grounding_identity_invalid", path, identity,
                               allowed=sorted(expected_rows))); continue
        if identity in observed:
            errors.append(_err("card_plan_check_grounding_identity_duplicate", path, identity))
        observed.add(identity)
        evidence = row.get("evidence_development_indices")
        if not isinstance(evidence, list) or any(type(x) is not int for x in evidence):
            errors.append(_err("card_plan_check_grounding_evidence_invalid", path + ".evidence_development_indices", evidence)); continue
        expected_row = expected_rows[identity]
        expected_indices = (expected_row["development_indices"] if identity[0] == "step" else
                            sorted(set(expected_row["before_development_indices"]) |
                                   set(expected_row["after_development_indices"]) |
                                   set(expected_row["development_indices"])))
        if len(evidence) != len(set(evidence)) or not set(evidence) <= set(expected_indices):
            errors.append(_err("card_plan_check_grounding_evidence_outside_mapping", path + ".evidence_development_indices", evidence,
                               allowed=expected_indices))
        if row.get("verdict") == "compatible":
            if not evidence:
                errors.append(_err("card_plan_check_grounding_positive_evidence_required", path + ".evidence_development_indices", evidence))
            if identity[0] == "dependency":
                before = expected_row["before_step_id"]; after = expected_row["after_step_id"]
                for side in ("before", "after"):
                    if not set(evidence) & set(expected_row[side + "_development_indices"]):
                        errors.append(_err("card_dependency_current_endpoint_evidence_required", path + ".evidence_development_indices", evidence,
                                           endpoint=side, allowed=expected_row[side + "_development_indices"]))
                negative_endpoints = [step_id for step_id in (before, after) if judgments.get(step_id) != "compatible"]
                if negative_endpoints:
                    errors.append(_err("card_dependency_positive_with_noncompatible_endpoint", path, row,
                                       endpoint_step_ids=negative_endpoints,
                                       policy="Only current explicit endpoint verdicts count; imagined corrections cannot satisfy a dependency."))
    if observed != set(expected_rows):
        errors.append(_err("card_plan_check_grounding_coverage_invalid", "card_plan_checks", sorted(observed),
                           missing=sorted(set(expected_rows) - observed)))
    return errors

