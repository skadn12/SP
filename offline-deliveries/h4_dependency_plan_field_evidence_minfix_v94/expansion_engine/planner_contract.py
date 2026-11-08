"""Explicit, source-free card reference contract and complete mapping diagnostics.

Validation checks reference integrity, not whether the proposed event realizes
the card's meaning. Qwen's later review remains responsible for that judgment.
The original card and model response are never edited here.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


MIN_SCENE_DEVELOPMENTS = 2
MAX_SCENE_DEVELOPMENTS = 5


PLANNER_CONTRACT_PROMPT = """
DATASET CARD MAPPING CONTRACT (dataset_new_event only):
There are THREE INDEPENDENT reference systems. Do not copy one system's numbers
into another. card_step_ids are literal IDs such as "s1" through "s6" from
steps (at most 6). dependency_indices are 1-based POSITIONS in dependencies
(at most 3), NOT step numbers. development_indices are 1-based POSITIONS in
YOUR scene's developments (2-5 entries), NOT step or dependency numbers.
Read card_mapping_context.numbered_dependencies and allowed_reference_values.
When there are 6 steps, 3 dependencies and 5 developments, the valid values are
respectively ["s1","s2","s3","s4","s5","s6"], [1,2,3], and [1,2,3,4,5].
Dependency 4 and development 6 do not exist in that example.

Return card_development_links as a list of objects with EXACTLY the keys
{card_step_ids, dependency_indices, development_indices}. Each entry needs
nonempty card_step_ids and development_indices; dependency_indices may be [].
Map every explicit card step exactly once and every explicit dependency exactly
once. Several steps may share a development, and one link may reference several
developments. A dependency's mapped development must express the relation
between its before/after steps, not just mention their two actions. It need not
put both endpoint steps in the same link. Do not add a sixth development merely
because there are six steps; combine related steps while preserving their
distinct states. Never discard a step, dependency, or invalid reference just
to pass validation. Correct the plan and its mappings together when necessary.

For example, ONLY as a mapping-shape illustration for a hypothetical 6-step,
3-dependency, 5-development card (NOT dialogue, NOT a plan to copy):
[{"card_step_ids":["s1","s2"],"dependency_indices":[1],"development_indices":[1]},
 {"card_step_ids":["s3"],"dependency_indices":[2],"development_indices":[2]},
 {"card_step_ids":["s4"],"dependency_indices":[],"development_indices":[3]},
 {"card_step_ids":["s5"],"dependency_indices":[3],"development_indices":[4]},
 {"card_step_ids":["s6"],"dependency_indices":[],"development_indices":[5]}]
Use the actual card's dependencies and actual relation-bearing developments;
this illustration does not assert that arbitrary relations fit this grouping.

For a scaffold, instantiate its abstract actions, conditions and relations in
a distinct NEW event. For a seed, develop its premise freely while preserving
every explicit step and dependency it actually contains. The label "seed"
does NOT mean there are no dependencies. Only an actually empty dependencies
array permits all dependency_indices to be empty. Keep requested, proposed,
agreed, attempted and reported outcomes distinct.
History and source_anchors establish speaker/state continuity only. Do not
relabel an old donation, camping, cooking or other ongoing event as the new
dataset event by attaching unrelated card IDs. Instantiate the supplied card's
abstract event in new concrete developments; do not import source identities,
quotes or outcomes as existing root facts. All new facts must later be realized
in the generated dialogue. Correct indices alone do not prove card fidelity.
"""


def _guidance(card: dict[str, Any]) -> dict[str, Any]:
    """Allowlist abstraction fields so source text/evidence cannot enter context."""
    output = {
        key: deepcopy(card[key])
        for key in ("card_id", "card_sha256", "guidance_level", "confusion_axes",
                    "open_ends", "downgrade_reason") if key in card
    }
    for name, keys in (
        ("actors", ("actor_id", "role")),
        ("steps", ("step_id", "actor_id", "action", "state")),
        ("dependencies", ("before", "after", "relation")),
    ):
        entries = card.get(name, [])
        output[name] = [
            {key: deepcopy(entry[key]) for key in keys if key in entry}
            for entry in entries if isinstance(entry, dict)
        ] if isinstance(entries, list) else []
    output["source_is_root_evidence"] = False
    output["new_details_must_be_realized_in_root_dialogue"] = True
    return output


def mapping_context(card: dict[str, Any], scene: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return legal output references, never authorize an invalid old scene.

    A rejected scene remains unchanged in previous_response. Its length must
    not expand the corrected response's permitted development index space.
    """
    guidance = _guidance(card)
    developments = scene.get("developments") if isinstance(scene, dict) else None
    actual_count = len(developments) if isinstance(developments, list) else None
    count_valid = (actual_count is not None
                   and MIN_SCENE_DEVELOPMENTS <= actual_count <= MAX_SCENE_DEVELOPMENTS)
    reference_count = actual_count if count_valid else MAX_SCENE_DEVELOPMENTS
    reference_scope = (
        "actual_scene_developments" if count_valid else
        "potential_positions_only; after planning use 1..len(scene.developments)"
    )
    if actual_count is not None and not count_valid:
        reference_scope = (
            "corrected_response_potential_positions_only; the previous scene count is invalid; "
            "rebuild developments and use 1..len(corrected_scene.developments), with count 2..5"
        )
    return {
        "card_guidance": guidance,
        "numbered_steps": deepcopy(guidance["steps"]),
        "numbered_dependencies": [
            {"dependency_index": index, **deepcopy(dependency)}
            for index, dependency in enumerate(guidance["dependencies"], 1)
        ],
        "allowed_reference_values": {
            "card_step_ids": [step["step_id"] for step in guidance["steps"] if "step_id" in step],
            "dependency_indices": list(range(1, len(guidance["dependencies"]) + 1)),
            "development_indices": list(range(1, reference_count + 1)),
        },
        "development_reference_scope": reference_scope,
        "observed_scene_development_count": actual_count,
        "observed_scene_development_count_valid": count_valid if actual_count is not None else None,
        "scene_development_count_limits": {
            "minimum": MIN_SCENE_DEVELOPMENTS, "maximum": MAX_SCENE_DEVELOPMENTS,
        },
        "coverage_requirement": "every explicit step and dependency exactly once; seed and scaffold alike",
        "semantic_requirement": "instantiate the card in a distinct new event; valid references alone are insufficient",
    }


def final_planner_instruction(repair: dict[str, Any] | None = None) -> str:
    """Pure reminder to append after historical feedback, without copying it."""
    instruction = (
        "\nCURRENT SCENE OUTPUT CHECK: If returning one scene, developments must contain "
        "2-5 entries. Six card steps do not require six developments. Several related steps "
        "may share one development, but its text must preserve their distinct actions, states "
        "and dependency relations. Before returning JSON, count the development entries and "
        "check every explicit card step and dependency is mapped exactly once, using only "
        "indices within the returned development array. Do not delete card steps or dependencies "
        "to satisfy the count; the assigned card and complete history remain unchanged. "
        "Build the MAIN new event from the card's actual actions, not a metaphor or a continuation "
        "of historical feedback examples. Keep the requesting/responding actor, action object "
        "and each failed outcome/retry explicit in the mapped development text. Mere step IDs "
        "or an unrelated action with the same broad shape are insufficient. Verify speaker "
        "and relationship identities against the full history. For a new trip state the origin, "
        "destination and outward/return leg; for later completion locate the new action interval "
        "and completion report after the old pending state. Do not claim this new progress was "
        "already established by historical evidence. A separate pre-realization semantic "
        "review will check each mapped action/state/dependency and the plan's consistency.\n"
    )
    errors = repair.get("errors", []) if isinstance(repair, dict) else []
    if isinstance(errors, list) and any(
        isinstance(error, dict) and error.get("code") == "scene_developments_count_invalid"
        for error in errors
    ):
        instruction += (
            "CARDINALITY REPAIR REQUIRED: The previous developments array is rejected. "
            "First revise the array itself to 2-5 entries by combining related actions in the "
            "same interaction without losing their actors, order or requested/reported/outcome "
            "states. Then rebuild the complete card_development_links against that new array. "
            "Changing only indices while returning the same oversized array does not repair "
            "this failure. Return the complete corrected response for the same scene slot.\n"
        )
    return instruction


class MappingValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = deepcopy(errors)
        # Preserve the legacy first-code exception interface for stage dispatch.
        super().__init__(errors[0]["code"] if errors else "card_mapping_invalid")


def collect_mapping_errors(scene: Any, card: Any) -> list[dict[str, Any]]:
    """Report every independently detectable fault without repairing any value."""
    errors: list[dict[str, Any]] = []

    def add(code: str, path: str, actual: Any, **details: Any) -> None:
        errors.append({"code": code, "path": path, "actual": deepcopy(actual), **deepcopy(details)})

    if not isinstance(scene, dict):
        add("scene_not_object", "scene", scene, allowed="object")
        scene = {}
    if not isinstance(card, dict):
        add("card_not_object", "card", card, allowed="object")
        card = {}
    if scene.get("dataset_card_id") != card.get("card_id"):
        add("dataset_card_id_mismatch", "scene.dataset_card_id", scene.get("dataset_card_id"),
            allowed=[card.get("card_id")])
    links = scene.get("card_development_links")
    if not isinstance(links, list) or not links:
        add("card_development_links_missing", "scene.card_development_links", links, allowed="nonempty array")
    developments = scene.get("developments")
    if not isinstance(developments, list) or not developments:
        add("scene_developments_missing", "scene.developments", developments, allowed="nonempty array")
    elif not MIN_SCENE_DEVELOPMENTS <= len(developments) <= MAX_SCENE_DEVELOPMENTS:
        add("scene_developments_count_invalid", "scene.developments", len(developments),
            allowed=list(range(MIN_SCENE_DEVELOPMENTS, MAX_SCENE_DEVELOPMENTS + 1)))
    steps = card.get("steps")
    if not isinstance(steps, list) or not steps:
        add("card_steps_missing", "card.steps", steps, allowed="nonempty array")
    dependencies = card.get("dependencies")
    if not isinstance(dependencies, list):
        add("card_dependencies_invalid", "card.dependencies", dependencies, allowed="array")

    step_values = [step.get("step_id") if isinstance(step, dict) else None
                   for step in steps] if isinstance(steps, list) else []
    step_ids = {value for value in step_values if isinstance(value, str) and value}
    if isinstance(steps, list) and (len(step_ids) != len(steps) or not all(
            isinstance(value, str) and value for value in step_values)):
        add("card_step_ids_invalid", "card.steps[*].step_id", step_values,
            allowed="distinct nonempty string IDs")
    dependency_ids = set(range(1, len(dependencies) + 1)) if isinstance(dependencies, list) else set()
    development_ids = set(range(1, len(developments) + 1)) if isinstance(developments, list) else set()
    mapped_steps: set[str] = set()
    mapped_dependencies: set[int] = set()
    mapped_developments: set[int] = set()
    fields = {"card_step_ids", "dependency_indices", "development_indices"}

    for index, link in enumerate(links if isinstance(links, list) else []):
        path = f"scene.card_development_links[{index}]"
        if not isinstance(link, dict):
            add("card_development_link_fields_invalid", path, link, allowed=sorted(fields))
            continue
        if set(link) != fields:
            add("card_development_link_fields_invalid", path, sorted(link), allowed=sorted(fields),
                missing=sorted(fields - set(link)), unexpected=sorted(set(link) - fields))
        linked_steps = link.get("card_step_ids")
        linked_dependencies = link.get("dependency_indices")
        linked_developments = link.get("development_indices")
        valid_steps = [value for value in linked_steps if isinstance(value, str) and value in step_ids
                       ] if isinstance(linked_steps, list) else []
        valid_dependencies = [value for value in linked_dependencies if type(value) is int and value in dependency_ids
                              ] if isinstance(linked_dependencies, list) else []
        valid_developments = [value for value in linked_developments if type(value) is int and value in development_ids
                              ] if isinstance(linked_developments, list) else []

        for name, values, valid, allowed, nonempty, code in (
            ("card_step_ids", linked_steps, valid_steps, sorted(step_ids), True, "card_development_link_steps_invalid"),
            ("dependency_indices", linked_dependencies, valid_dependencies, sorted(dependency_ids), False,
             "card_development_link_dependencies_invalid"),
            ("development_indices", linked_developments, valid_developments, sorted(development_ids), True,
             "card_development_link_indices_invalid"),
        ):
            if (not isinstance(values, list) or (nonempty and not values)
                    or len(valid) != len(values)):
                add(code, f"{path}.{name}", values, allowed=allowed)
        repeated_steps = mapped_steps & set(valid_steps)
        repeated_steps.update(value for value in valid_steps if valid_steps.count(value) > 1)
        repeated_dependencies = mapped_dependencies & set(valid_dependencies)
        repeated_dependencies.update(value for value in valid_dependencies if valid_dependencies.count(value) > 1)
        if repeated_steps:
            add("card_step_mapped_more_than_once", f"{path}.card_step_ids", linked_steps,
                repeated=sorted(repeated_steps), allowed="each step exactly once across all links")
        if repeated_dependencies:
            add("card_dependency_mapped_more_than_once", f"{path}.dependency_indices", linked_dependencies,
                repeated=sorted(repeated_dependencies), allowed="each dependency exactly once across all links")
        mapped_steps.update(valid_steps)
        mapped_dependencies.update(valid_dependencies)
        mapped_developments.update(valid_developments)

    if step_ids - mapped_steps:
        add("card_steps_not_fully_mapped", "scene.card_development_links[*].card_step_ids",
            sorted(mapped_steps), allowed=sorted(step_ids), missing=sorted(step_ids - mapped_steps))
    if dependency_ids - mapped_dependencies:
        add("card_dependencies_not_fully_mapped", "scene.card_development_links[*].dependency_indices",
            sorted(mapped_dependencies), allowed=sorted(dependency_ids), missing=sorted(dependency_ids - mapped_dependencies))
    if not mapped_developments:
        add("card_premise_not_mapped_to_development", "scene.card_development_links[*].development_indices",
            [], allowed=sorted(development_ids))
    return errors


def validate_mapping(scene: Any, card: Any) -> bool:
    errors = collect_mapping_errors(scene, card)
    if errors:
        raise MappingValidationError(errors)
    return True


def build_repair_feedback(error: Exception, raw: Any, card: dict[str, Any] | None = None) -> dict[str, Any]:
    """Supply all faults and legal references to Qwen; never rewrite its answer."""
    feedback = {
        "failure": str(error),
        "errors": deepcopy(getattr(error, "errors", [
            {"code": str(error), "path": "response", "actual": None}
        ])),
        "previous_response": deepcopy(raw),
        "instruction": (
            "Return the complete corrected response for the same scene slot and card. "
            "Fix every listed error, including missing coverage. The three reference systems are distinct. "
            "Do not delete card steps/dependencies, drop invalid references silently, change the supplied card, "
            "or exceed five developments. Revise developments and mappings together so the new event "
            "actually instantiates the card's actions and relations, preserving requested/proposed/outcome states. "
            "A seed still requires all explicit steps and dependencies; history only supplies continuity. "
            "References must be valid for your corrected response's actual development count."
        ),
    }
    if card is not None:
        scene = None
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                scenes = parsed.get("scenes")
                if isinstance(scenes, list) and len(scenes) == 1 and isinstance(scenes[0], dict):
                    scene = scenes[0]
        except (ValueError, TypeError):
            pass
        feedback["card_mapping_context"] = mapping_context(card, scene)
        feedback["allowed_reference_values"] = deepcopy(feedback["card_mapping_context"]["allowed_reference_values"])
    if any(isinstance(item, dict) and item.get("code") == "scene_developments_count_invalid"
           for item in feedback["errors"]):
        feedback["development_count_repair"] = {
            "required_count_limits": {
                "minimum": MIN_SCENE_DEVELOPMENTS, "maximum": MAX_SCENE_DEVELOPMENTS,
            },
            "must_revise_developments_array": True,
            "must_rebuild_all_card_development_links": True,
            "preserve_all_card_steps_and_dependencies": True,
            "instruction": final_planner_instruction(feedback),
        }
    return feedback
