"""Explicit, source-free card reference contract and complete mapping diagnostics.

Validation checks reference integrity, not whether the proposed event realizes
the card's meaning. Qwen's later review remains responsible for that judgment.
The original card and model response are never edited here.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any


ROUTES = ("old_event_continuation", "dataset_new_event")


def _text_schema(example: str, description: str = "") -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "minLength": 1, "pattern": r"\S",
                              "x-shape-example": example}
    if description:
        schema["description"] = description
    return schema


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _array_schema(items: dict[str, Any], minimum: int = 0,
                  maximum: int | None = None, *, unique: bool = False,
                  shape_count: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        schema["maxItems"] = maximum
    if unique:
        schema["uniqueItems"] = True
    if shape_count is not None:
        schema["x-shape-count"] = shape_count
    return schema


def planner_output_schema(route: str) -> dict[str, Any]:
    """Single authoritative v24 shape contract; examples are never repairs.

    Dynamic membership, exact evidence, and cross-field relations remain checked
    by the existing validators. JSON shape checks never certify event semantics.
    """
    if route not in ROUTES:
        raise ValueError("invalid_plan_route")
    new = route == ROUTES[1]
    definition = _object_schema({
        "trigger": _text_schema("NEW_INITIATING_CIRCUMSTANCE_FROM_MODEL",
            "For B, define an independent initiating occasion from the card first; "
            "history then constrains participants, state and timing."),
        "main_goal": _text_schema("CONCRETE_GOAL_FROM_MODEL", "Copy scene.goal verbatim."),
        "history_relation": {"type": "string", "const": "distinct_event" if new else "continuation"},
    })
    transition_properties = {
        "basis_fact_ids": _array_schema(_text_schema("ACTUAL_FACT_ID_{index}"),
                                         0 if new else 1, unique=True),
    }
    if new:
        transition_properties["basis_reason"] = _text_schema("HISTORY_CONSTRAINT_EXPLANATION_FROM_MODEL",
            "The sole explanation for both source_anchors and basis_fact_ids: explain each "
            "identity/state/time constraint, or why empty references need no specific historical "
            "premise. The full history still binds. History must not replace the new main goal.")
    transition_properties.update({
        "new_event": _text_schema("NEW_PROGRESS_OR_EVENT_FROM_MODEL"),
        "temporal_placement": _text_schema("RELATIVE_TIMING_FROM_MODEL"),
    })
    link = _object_schema({
        "card_step_ids": _array_schema(_text_schema("ACTUAL_CARD_STEP_ID_{index}"), 1, unique=True),
        "dependency_indices": _array_schema({"type": "integer", "minimum": 1}, unique=True),
        "development_indices": _array_schema({"type": "integer", "minimum": 1, "maximum": 5}, 1, unique=True),
    })
    # Key order guides generation; it is not a rejection criterion.
    scene = _object_schema({
        "event_definition": definition,
        "scene_id": _text_schema("TARGET_SCENE_ID"),
        "actors": _array_schema(_text_schema("ALLOWED_SPEAKER_{index}"), 2, unique=True),
        "source_anchors": _array_schema(_object_schema({
            "turn_ref": _text_schema("ACTUAL_HISTORY_TURN_REF_{index}"),
            "quote": _text_schema("EXACT_HISTORY_SUBSTRING_{index}"),
        }), 0 if new else 1, 3),
        "goal": _text_schema("CONCRETE_GOAL_FROM_MODEL"),
        "developments": _array_schema(_text_schema("CONCRETE_NEW_DEVELOPMENT_{index}"), 2, 5),
        "dataset_card_id": _text_schema("ASSIGNED_CARD_ID") if new else {"type": "null"},
        "card_development_links": _array_schema(link, 1) if new else _array_schema(link, 0, 0),
        "feedback_topic_ids": _array_schema(_text_schema("SUPPLIED_FEEDBACK_TOPIC_ID_{index}"), 0, 4, unique=True),
        "state_transition": _object_schema(transition_properties),
    })
    schema = _object_schema({
        "root_id": _text_schema("ROOT_ID_FROM_INPUT"),
        "route": {"type": "string", "const": route},
        "scenes": _array_schema(scene, 0, 1, shape_count=1),
        "shortfall_reason": {"type": "string", "x-shape-example": "",
            "description": "Empty string for one scene; nonblank explanation for zero scenes. Never null."},
    })
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = f"v24 planner response: {route}"
    return schema


def _shape_from_schema(schema: dict[str, Any], index: int = 1) -> Any:
    if "const" in schema:
        return deepcopy(schema["const"])
    if "x-shape-example" in schema:
        return schema["x-shape-example"].replace("{index}", str(index))
    kind = schema["type"]
    if kind == "object":
        return {name: _shape_from_schema(child, index) for name, child in schema["properties"].items()}
    if kind == "array":
        return [_shape_from_schema(schema["items"], number + 1)
                for number in range(schema.get("x-shape-count", schema.get("minItems", 0)))]
    if kind == "integer":
        return max(index, schema.get("minimum", 0))
    if kind == "null":
        return None
    raise ValueError(f"planner_shape_example_missing:{kind}")


def planner_shape_example(route: str) -> dict[str, Any]:
    """Generate an illustrative shape, never a candidate or fallback scene."""
    return _shape_from_schema(planner_output_schema(route))


def planner_contract_prompt(route: str) -> str:
    schema = planner_output_schema(route)
    order = (
        "For B, first form event_definition.trigger and main_goal from the assigned card's "
        "independent purpose/occasion. Then fit that event to ALL cumulative_history, "
        "including accepted A. Do not start by picking an unfinished old task. Return "
        "event_definition first in each scene. state_transition.basis_reason is the sole "
        "explanation of history anchors and fact references; do not emit history_context_reason."
        if route == ROUTES[1] else
        "For A, define the old event and substantive new progress; cite actual history and facts."
    )
    return ("\nAUTHORITATIVE PLANNER OUTPUT CONTRACT (v24):\n" + order
        + "\nAll objects have exactly their schema keys. This is ONE scene slot, zero or one "
          "scene; the runner handles three slots per route sequentially. Shape validity does "
          "not prove a distinct B event, card premise fidelity, or coherent history. "
          "The example uses placeholders only; author all scene values from input.\n"
        + "PLANNER_OUTPUT_SCHEMA_JSON\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        + "\nPLANNER_SHAPE_EXAMPLE_JSON\n" + json.dumps(_shape_from_schema(schema), ensure_ascii=False, separators=(",", ":")))


class PlannerSchemaValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = deepcopy(errors)
        super().__init__(errors[0]["code"] if errors else "planner_schema_invalid")


def collect_planner_schema_errors(plan: Any, route: str) -> list[dict[str, Any]]:
    """Collect every detectable structural fault with exact paths; never mutate.

    Implements the JSON Schema keywords used by planner_output_schema.
    Returning [] approves structure only, not semantics.
    """
    schema = planner_output_schema(route)
    errors: list[dict[str, Any]] = []

    def add(code: str, path: str, actual: Any, **details: Any) -> None:
        errors.append({"code": code, "path": path, "actual": deepcopy(actual), **deepcopy(details)})

    def visit(value: Any, contract: dict[str, Any], path: str) -> None:
        kind = contract["type"]
        valid_type = {
            "object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str), "integer": type(value) is int,
            "null": value is None,
        }[kind]
        if not valid_type:
            add("planner_schema_type_invalid", path, value, expected_type=kind,
                actual_type=type(value).__name__)
            return
        if "const" in contract and value != contract["const"]:
            add("planner_schema_value_invalid", path, value, allowed=[contract["const"]])
        if kind == "object":
            properties = contract["properties"]
            for name in contract["required"]:
                if name not in value:
                    add("planner_schema_missing_field", f"{path}.{name}", None,
                        missing=[name], expected_type=properties[name]["type"],
                        object_path=path, allowed=list(properties))
            for name in value:
                if name not in properties:
                    add("planner_schema_unexpected_field", f"{path}.{name}", value[name],
                        unexpected=[name], object_path=path, allowed=list(properties))
            for name, child in properties.items():
                if name in value:
                    visit(value[name], child, f"{path}.{name}")
        elif kind == "array":
            if len(value) < contract.get("minItems", 0) or len(value) > contract.get("maxItems", float("inf")):
                add("planner_schema_array_count_invalid", path, len(value),
                    minimum=contract.get("minItems", 0), maximum=contract.get("maxItems"))
            if contract.get("uniqueItems"):
                seen = set()
                for number, item in enumerate(value):
                    identity = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if identity in seen:
                        add("planner_schema_duplicate_item", f"{path}[{number}]", item, allowed="distinct array values")
                    seen.add(identity)
            for number, item in enumerate(value):
                visit(item, contract["items"], f"{path}[{number}]")
        elif kind == "string":
            if len(value) < contract.get("minLength", 0) or ("pattern" in contract and not re.search(contract["pattern"], value)):
                add("planner_schema_string_invalid", path, value, allowed="nonblank string")
        elif kind == "integer":
            if value < contract.get("minimum", -float("inf")) or value > contract.get("maximum", float("inf")):
                add("planner_schema_number_invalid", path, value,
                    minimum=contract.get("minimum"), maximum=contract.get("maximum"))

    visit(plan, schema, "response")
    return errors


def validate_planner_schema(plan: Any, route: str) -> bool:
    errors = collect_planner_schema_errors(plan, route)
    if errors:
        raise PlannerSchemaValidationError(errors)
    return True


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
For a scaffold, map every explicit card step and dependency exactly once.
For a seed, instantiate the CORE PREMISE, choosing relevant source steps freely:
map at least one source step to concrete developments. Other source steps and
dependencies are optional. A seed is not a mandatory source workflow.
Each selected step or dependency may occur at most once across all links.
Several steps may share a development, and one link may reference several
developments. A dependency's mapped development must express the relation
between its before/after steps, not just mention their two actions. It need not
put both endpoint steps in the same link. Do not add a sixth development merely
because there are six steps; combine related steps while preserving their
distinct states. Never silently discard an invalid reference just to pass
validation. For a scaffold, never drop a required step or dependency.
Correct the plan and its mappings together when necessary.

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
a distinct NEW event. For a seed, develop its premise freely; unselected source
steps and dependencies do not require realization. Selected relations still
need valid references and coherent realization. A seed may use [] for all
dependency_indices even when the source has dependencies. Mapping an incidental
step alone does not prove premise fidelity: Qwen must judge the CORE PREMISE.
Keep requested, proposed, agreed, attempted and reported outcomes distinct.
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
    """Return explicit reference tables, never provenance or raw source quotes."""
    guidance = _guidance(card)
    developments = scene.get("developments") if isinstance(scene, dict) else None
    actual_count = len(developments) if isinstance(developments, list) else None
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
            "development_indices": list(range(1, (actual_count if actual_count is not None else 5) + 1)),
        },
        "development_reference_scope": (
            "actual_scene_developments" if actual_count is not None
            else "potential_positions_only; after planning use 1..len(scene.developments)"
        ),
        "scene_development_count_limits": {"minimum": 2, "maximum": 5},
        "coverage_requirement": (
            "instantiate the core premise; map at least one relevant source step; remaining steps/dependencies optional"
            if guidance.get("guidance_level") == "seed" else
            "every explicit step and dependency exactly once"
        ),
        "semantic_requirement": "instantiate the card in a distinct new event; valid references alone are insufficient",
    }


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
    elif not 2 <= len(developments) <= 5:
        add("scene_developments_count_invalid", "scene.developments", len(developments), allowed=[2, 3, 4, 5])
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
                repeated=sorted(repeated_steps), allowed="each selected step at most once across all links")
        if repeated_dependencies:
            add("card_dependency_mapped_more_than_once", f"{path}.dependency_indices", linked_dependencies,
                repeated=sorted(repeated_dependencies), allowed="each selected dependency at most once across all links")
        mapped_steps.update(valid_steps)
        mapped_dependencies.update(valid_dependencies)
        mapped_developments.update(valid_developments)

    seed = card.get("guidance_level") == "seed"
    if not seed and step_ids - mapped_steps:
        add("card_steps_not_fully_mapped", "scene.card_development_links[*].card_step_ids",
            sorted(mapped_steps), allowed=sorted(step_ids), missing=sorted(step_ids - mapped_steps))
    if not seed and dependency_ids - mapped_dependencies:
        add("card_dependencies_not_fully_mapped", "scene.card_development_links[*].dependency_indices",
            sorted(mapped_dependencies), allowed=sorted(dependency_ids), missing=sorted(dependency_ids - mapped_dependencies))
    if seed and not mapped_steps:
        add("card_seed_premise_source_step_missing", "scene.card_development_links[*].card_step_ids",
            [], allowed=sorted(step_ids), requirement="at least one selected actual source step")
    if not mapped_developments:
        add("card_premise_not_mapped_to_development", "scene.card_development_links[*].development_indices",
            [], allowed=sorted(development_ids))
    return errors


def validate_mapping(scene: Any, card: Any) -> bool:
    errors = collect_mapping_errors(scene, card)
    if errors:
        raise MappingValidationError(errors)
    return True


def build_repair_feedback(error: Exception, raw: Any, card: dict[str, Any] | None = None,
                          *, route: str | None = None) -> dict[str, Any]:
    """Supply all faults and legal references to Qwen; never rewrite its answer."""
    feedback = {
        "failure": str(error),
        "errors": deepcopy(getattr(error, "errors", [
            {"code": str(error), "path": "response", "actual": None}
        ])),
        "previous_response": deepcopy(raw),
        "instruction": (
            "Return the complete corrected response for the same scene slot and card. "
            "Fix every listed field/path error. Match the authoritative output schema exactly. "
            "The three reference systems are distinct. "
            "Do not change the supplied card, silently drop invalid references, "
            "or exceed five developments. Revise developments and mappings together so the new event "
            "actually instantiates the card's actions and relations, preserving requested/proposed/outcome states. "
            "A scaffold requires all source steps/dependencies; a seed requires its core premise "
            "and at least one relevant mapped source step, not its full source workflow. "
            "History supplies continuity; B defines its independent purpose before adapting to history. "
            "References must be valid for your corrected response's actual development count."
        ),
    }
    parsed = None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        pass
    if route is None and isinstance(parsed, dict) and parsed.get("route") in ROUTES:
        route = parsed["route"]
    if route in ROUTES:
        feedback["planner_output_schema"] = planner_output_schema(route)
        feedback["planner_shape_example"] = planner_shape_example(route)
        feedback["shape_example_is_not_a_candidate"] = True
    feedback["semantic_certified"] = False
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
    return feedback
