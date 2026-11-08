"""One model-visible output contract for initial and additive planning."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator


VERSION = "shared_planner_output_v2"
RENDER_VERSION = "shared_planner_schema_only_v3"
ROOT_SHARED_PROMPT_VERSION = "incremental_retention_v1.supplement.shared_planner_v2"
COLLECTIONS = ("selected_continuations", "new_development_candidates")
# Legacy cap remains the default; cap-9 is an explicit opt-in contract revision.
MAX_NEW_DEVELOPMENT_CANDIDATES = 8
CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES = 9
CAP9_VERSION = "shared_planner_output_v3_cap9"
RELATION_FIELDS = ("source_candidate_ref", "dependencies")


def _text(example, **extra):
    return {"type": "string", "minLength": 1, "pattern": r"\S",
            "examples": [example], **extra}


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _schema(*, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES):
    source_ref = {"anyOf": [{"$ref": "#/$defs/source_candidate_ref"}, {"type": "null"}],
                  "examples": [None]}
    output_ref = {"anyOf": [{"$ref": "#/$defs/output_ref"}, {"type": "null"}],
                  "examples": [None]}
    relation_properties = {
        "source_candidate_ref": source_ref,
        "dependencies": {"type": "array", "items": {"$ref": "#/$defs/dependency"},
                         "examples": [[]]},
    }
    people = {"type": "array", "maxItems": 2,
              "items": {"$ref": "#/$defs/new_person"}, "examples": [[]]}
    common = {
        "new_people": people,
        "goal": _text("Prepare a labelled community workshop card."),
        "speaker_refs": {"type": "array", "minItems": 1, "uniqueItems": True,
                         "items": {"$ref": "#/$defs/speaker_ref"},
                         "examples": [["SP01", "SP02"]]},
        "transition_from_previous": _text("The participants discuss their next workshop arrangement."),
        "required_transition": _text("The participants choose and record a workshop card label."),
        "after_state": _text("The desired label is agreed; this is a planned target, not an observed fact."),
        "actor_bindings": {"type": "array", "minItems": 1,
                           "items": {"$ref": "#/$defs/actor_binding"}},
        "interaction_beats": {"type": "array", "minItems": 3, "maxItems": 7,
                              "items": {"$ref": "#/$defs/nonempty_text"},
                              "examples": [["One participant proposes a label.",
                                            "Another participant checks the proposed label.",
                                            "They record the agreed label on the card."]]},
        **relation_properties,
    }
    continuation = {
        "verified_track_ref": {
            "anyOf": [_text("VT01"), {"type": "null"}], "examples": ["VT01"],
            "description": "Verified Memory track reference; null for a continuation of an original new-development candidate.",
        },
        **common,
    }
    development = {
        "source_basis": {"type": "array", "minItems": 1, "maxItems": 4,
                         "items": {"$ref": "#/$defs/nonempty_text"},
                         "examples": [["The existing participants share an interest in community activities."]]},
        **common,
        "before_state": _text("The participants begin discussing the workshop arrangement."),
    }
    dependency = _object({
        "source_candidate_ref": source_ref,
        "output_ref": output_ref,
        "condition": _text("The earlier scene established the agreed workshop arrangement."),
    })
    dependency["oneOf"] = [
        {"properties": {"source_candidate_ref": {"$ref": "#/$defs/source_candidate_ref"},
                        "output_ref": {"type": "null"}}},
        {"properties": {"source_candidate_ref": {"type": "null"},
                        "output_ref": {"$ref": "#/$defs/output_ref"}}},
    ]
    schema = _object({
        "round_goal": _text("Develop concrete workshop arrangements from the existing conversation."),
        "selected_continuations": {"type": "array", "items": {"$ref": "#/$defs/continuation"}},
        "new_development_candidates": {"type": "array", "maxItems": max_new_candidates,
                                       "items": {"$ref": "#/$defs/new_development"}},
    })
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema",
                   "title": VERSION,
                   "$defs": {
                       "nonempty_text": _text("A concrete action or fact."),
                       "source_candidate_ref": _text("ND08:C01", description="An exact reference from the frozen original candidate registry."),
                       "output_ref": _text("/selected_continuations/0",
                                           pattern=r"^/(selected_continuations|new_development_candidates)/(0|[1-9][0-9]*)$",
                                           **{"not": {"pattern": r"\s"}}),
                       "speaker_ref": _text("SP01", description="An existing reference from speaker_registry."),
                       "new_person": _object({
                           "name": _text("Elena"),
                           "relationship_basis": _text("A neighbor introduced by an existing participant."),
                       }),
                       "actor_binding": _object({
                           "actor_ref": {"$ref": "#/$defs/speaker_ref"},
                           "binding": _text("This participant records the agreed workshop label."),
                       }),
                       "dependency": dependency,
                       "continuation": _object(continuation),
                       "new_development": _object(development),
                   }})
    return schema


def _without_examples(value):
    if isinstance(value, dict):
        return {key: _without_examples(item) for key, item in value.items() if key != "examples"}
    if isinstance(value, list):
        return [_without_examples(item) for item in value]
    return value


# Example values remain offline fixtures and never enter either model request.
_FIXTURE_SCHEMA = _schema()
_OUTPUT_SCHEMA = _without_examples(_FIXTURE_SCHEMA)
_FIELD_DESCRIPTIONS = {
    "round_goal": "Overall direction grounded in the actual history and feedback; copy the frozen round_goal exactly in append_only mode.",
    "goal": "The desired outcome for this candidate, grounded in the actual input. In append_only continuations, copy the source goal exactly.",
    "transition_from_previous": "How this development connects to the actual prior context; copy the source field for append_only continuations.",
    "required_transition": "The concrete new action, decision or state change this scene should enact.",
    "after_state": "The desired resulting state, not an observed fact. In append_only continuations, copy the source target exactly.",
    "before_state": "The supported starting situation from the actual context; do not invent a missing action or absence.",
    "source_basis": "One to four concrete supporting facts from the actual input, tying the proposed development to existing people and circumstances.",
    "interaction_beats": "Three to seven third-person action summaries for this development, not dialogue lines or repeated general discussion.",
    "new_people": "Zero to two declared people with their name and relationship_basis; none newly introduced in a continuation.",
    "condition": "The specific prerequisite that actual retained dialogue must establish, derived from the referenced development.",
    "binding": "Which participating actor owns a key action; copy existing source ownership exactly for append_only continuations.",
    "name": "The exact name of the person declared in this candidate.",
    "relationship_basis": "The grounded connection between the declared person and existing participants.",
}


def _describe_fields(value):
    if isinstance(value, dict):
        for name, field in value.get("properties", {}).items():
            if name in _FIELD_DESCRIPTIONS:
                field["description"] = _FIELD_DESCRIPTIONS[name]
        for item in value.values():
            _describe_fields(item)
    elif isinstance(value, list):
        for item in value:
            _describe_fields(item)


_describe_fields(_OUTPUT_SCHEMA)
Draft202012Validator.check_schema(_OUTPUT_SCHEMA)
_VALIDATOR = Draft202012Validator(_OUTPUT_SCHEMA)


def output_schema(*, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES):
    """Return the requested schema; default preserves the legacy cap-8 contract."""
    if max_new_candidates == MAX_NEW_DEVELOPMENT_CANDIDATES:
        return deepcopy(_OUTPUT_SCHEMA)
    if type(max_new_candidates) is not int or max_new_candidates != CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES:
        raise ValueError("invalid_max_new_candidates")
    schema = deepcopy(_OUTPUT_SCHEMA)
    schema["properties"]["new_development_candidates"]["maxItems"] = max_new_candidates
    schema["title"] = CAP9_VERSION
    return schema


def validate_output(value, *, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES):
    errors = []
    validator = _VALIDATOR if max_new_candidates == MAX_NEW_DEVELOPMENT_CANDIDATES else Draft202012Validator(output_schema(max_new_candidates=max_new_candidates))
    for error in sorted(validator.iter_errors(value),
                        key=lambda item: (str(list(item.absolute_path)), item.message)):
        pointer = "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1")
                                 for part in error.absolute_path)
        errors.append(f"{pointer}: {error.message}")
    return {"valid": not errors, "errors": errors}


def parse_output(text):
    """Keep duplicate JSON keys from silently overwriting model fields."""
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("shared_planner_duplicate_json_key:" + key)
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError("shared_planner_invalid_json_constant:" + value)

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def _example(schema):
    if "examples" in schema:
        return deepcopy(schema["examples"][0])
    if "$ref" in schema:
        return _example(_FIXTURE_SCHEMA["$defs"][schema["$ref"].rsplit("/", 1)[1]])
    if schema.get("type") == "object":
        return {key: _example(value) for key, value in schema["properties"].items()}
    if schema.get("type") == "array":
        return [_example(schema["items"]) for _ in range(schema.get("minItems", 0))]
    raise ValueError("shared_planner_schema_example_missing")


def output_examples():
    """Schema-validated offline fixtures, excluded from the model-visible text."""
    empty = _example(_FIXTURE_SCHEMA)
    nonempty = deepcopy(empty)
    for collection in COLLECTIONS:
        nonempty[collection].append(_example(_FIXTURE_SCHEMA["properties"][collection]["items"]))
    append = deepcopy(nonempty)
    source = append["selected_continuations"][0]
    source.update(verified_track_ref=None, source_candidate_ref="ND08:C01",
                  dependencies=[{"source_candidate_ref": "ND08:C01", "output_ref": None,
                                 "condition": "The retained original scene established the workshop arrangement."}])
    append["new_development_candidates"][0]["dependencies"] = [
        {"source_candidate_ref": None, "output_ref": "/selected_continuations/0",
         "condition": "The earlier added scene established the workshop label."}]
    examples = [nonempty, empty, append]
    for example in examples:
        result = validate_output(example)
        if not result["valid"]:
            raise ValueError("shared_planner_invalid_contract_example: " + str(result["errors"]))
    return examples


def render_output_contract(*, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES, contract_version=None):
    version = contract_version or (CAP9_VERSION if max_new_candidates == CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES else VERSION)
    return ("SHARED PLANNER OUTPUT CONTRACT " + version + "\n"
            "RENDER VERSION " + RENDER_VERSION + "\n"
            "Return one JSON object matching this schema. All candidate fields stay at one level.\n"
            "Do not emit phase IDs, hashes, status, candidate wrappers, or additional keys.\n"
            "The schema defines field types and constraints only. Derive all story content "
            "from the supplied history, Memory navigation and feedback. Do not return the "
            "schema itself, its descriptions, or placeholder text as the plan.\n"
            "Use references from the actual supplied menus. A dependency condition must "
            "describe a real prerequisite for that proposed development. Use empty "
            "dependencies when no genuine prerequisite exists; otherwise declare the "
            "applicable dependency. Source continuations must declare their prerequisites.\n"
            "In action-bearing text use the exact canonical person names from the speaker "
            "registry or the exact new_people declarations. Transport roles user and assistant "
            "do not identify story participants; never infer an actor from them. If the "
            "input does not identify the actor, skip that continuation rather than guessing.\n"
            "JSON SCHEMA\n" + json.dumps(output_schema(max_new_candidates=max_new_candidates), ensure_ascii=False, indent=2)
            + "\nEND JSON SCHEMA\n"
            "In append_only mode copy round_goal and immutable source fields from the frozen input.\n")


def iter_candidates(value):
    for collection in COLLECTIONS:
        for index, row in enumerate(value[collection]):
            yield f"/{collection}/{index}", row, collection


def strip_relations(value):
    """Project a validated shared output into the existing candidate contract."""
    legacy = deepcopy(value)
    for _, row, _ in iter_candidates(legacy):
        for key in RELATION_FIELDS:
            row.pop(key, None)
    return legacy
