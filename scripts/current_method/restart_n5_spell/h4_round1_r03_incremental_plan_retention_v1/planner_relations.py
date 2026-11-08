"""Explicit planning relation version; historical output contracts stay intact."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator

from . import shared_planner_contract as legacy_contract
from . import shared_planning as legacy_planning
from .planning import sha


VERSION = "shared_planner_relations_v1"
RENDER_VERSION = "planner_relations_schema_v1"
NATURAL = "natural_continuity"
HARD = "hard_prerequisite"
MAX_NEW_DEVELOPMENT_CANDIDATES = legacy_contract.MAX_NEW_DEVELOPMENT_CANDIDATES
CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES = legacy_contract.CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES
CAP9_VERSION = "shared_planner_relations_v2_cap9"


def output_schema(*, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES, contract_version=VERSION):
    schema = legacy_contract.output_schema(max_new_candidates=max_new_candidates)
    schema["title"] = contract_version
    dependency = schema["$defs"]["dependency"]
    dependency["properties"]["relation_kind"] = {"enum": [NATURAL, HARD]}
    dependency["required"].append("relation_kind")
    dependency["properties"]["condition"] = {
        "anyOf": [{"type": "string", "minLength": 1, "pattern": r"\S"},
                  {"type": "null"}],
        "description": "The actual result needed by a hard prerequisite; null for natural continuity.",
    }
    dependency["allOf"] = [{
        "if": {"properties": {"relation_kind": {"const": HARD}}},
        "then": {"properties": {"condition": {"type": "string", "minLength": 1, "pattern": r"\S"}}},
        "else": {"properties": {"condition": {"type": "null"}}},
    }]
    for name in ("continuation", "new_development"):
        properties = schema["$defs"][name]["properties"]
        properties["required_transition"]["description"] = (
            "The concrete change this scene aims to advance, not an observed fact or guaranteed completion.")
        properties["dependencies"]["description"] = (
            "Ordered predecessor relations declared during planning. Empty means undeclared, not proven independent.")
    Draft202012Validator.check_schema(schema)
    return schema


def initial_output_schema(*, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES, contract_version=VERSION):
    """Initial-only schema; default preserves cap-8, cap-9 is explicit."""
    schema = deepcopy(output_schema(max_new_candidates=max_new_candidates, contract_version=contract_version))
    null_ref = {"type": "null"}
    dependency = schema["$defs"]["dependency"]
    dependency["properties"]["source_candidate_ref"] = null_ref
    for name in ("continuation", "new_development"):
        schema["$defs"][name]["properties"]["source_candidate_ref"] = null_ref
    schema["title"] = contract_version + "_initial"
    def scrub(value):
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            return (value.replace("append_only mode", "initial mode")
                         .replace("append_only continuations", "initial candidates")
                         .replace("frozen original candidate registry", "initial candidate registry (none exists)"))
        return value
    schema = scrub(schema)
    schema["$defs"]["source_candidate_ref"]["description"] = (
        "Must be null in initial mode; no prior Planner candidate registry exists.")
    schema["$defs"]["continuation"]["properties"]["dependencies"]["description"] = (
        "Usually []; a continuation is grounded by verified_track_ref. Declare an earlier "
        "output_ref only when another candidate in this same response truly must precede it; "
        "never reference the current candidate.")
    schema["$defs"]["new_development"]["properties"]["dependencies"]["description"] = (
        "Usually []; declare an earlier output_ref only for a real same-response predecessor, "
        "and never reference the current candidate.")
    return schema


def validate_output(value):
    errors = []
    for error in sorted(Draft202012Validator(output_schema()).iter_errors(value),
                        key=lambda item: (str(list(item.absolute_path)), item.message)):
        errors.append("/" + "/".join(map(str, error.absolute_path)) + ": " + error.message)
    return {"valid": not errors, "errors": errors}


def validate_initial_output(value, *, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES, contract_version=VERSION):
    errors = []
    validator = Draft202012Validator(initial_output_schema(max_new_candidates=max_new_candidates, contract_version=contract_version))
    for error in sorted(validator.iter_errors(value),
                        key=lambda item: (str(list(item.absolute_path)), item.message)):
        errors.append("/" + "/".join(map(str, error.absolute_path)) + ": " + error.message)
    if not errors:
        errors.extend(_relation_errors(value))
    return {"valid": not errors, "errors": errors}


def output_examples():
    examples = legacy_contract.output_examples()
    for example in examples:
        for _, row, _ in legacy_contract.iter_candidates(example):
            for relation in row["dependencies"]:
                relation["relation_kind"] = HARD
    natural = deepcopy(examples[0])
    natural["new_development_candidates"][0]["dependencies"] = [{
        "source_candidate_ref": None, "output_ref": "/selected_continuations/0",
        "relation_kind": NATURAL, "condition": None,
    }]
    examples.append(natural)
    for example in examples:
        validation = validate_output(example)
        if not validation["valid"]:
            raise ValueError("planner_relation_example_invalid:" + str(validation["errors"]))
    return examples


def render_output_contract(mode="append_only", *, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES):
    if mode not in {"initial", "append_only"}:
        raise ValueError("unknown_planner_relation_render_mode:" + str(mode))
    if mode != "initial" and max_new_candidates != MAX_NEW_DEVELOPMENT_CANDIDATES:
        raise ValueError("cap9_initial_only")
    contract_version = CAP9_VERSION if max_new_candidates == CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES else VERSION
    schema = (initial_output_schema(max_new_candidates=max_new_candidates, contract_version=contract_version)
              if mode == "initial" else output_schema())
    mode_rules = (
        "This is initial mode. No prior Planner candidate registry exists. Every candidate "
        "source_candidate_ref MUST be null. Every dependency source_candidate_ref MUST be null; "
        "dependencies may only use an earlier output_ref from this same response. Memory event IDs "
        "such as ME01 and track IDs such as VT01 are not Planner candidate refs. "
        "A selected_continuation is already grounded by its verified_track_ref; its dependencies "
        "array may be empty and normally should be empty unless it truly requires an earlier "
        "candidate from this response. Never make a candidate depend on itself.\n"
        if mode == "initial" else
        "This is append_only mode. A source_candidate_ref may use only an exact frozen Planner "
        "candidate identity from the supplied source registry; a same-response dependency uses an "
        "earlier output_ref. Set the other reference field to null.\n")
    return ("SHARED PLANNER OUTPUT CONTRACT " + contract_version + "\n"
            + mode_rules
            + "Return exactly the existing flat Planner object. Do not emit phase IDs, scene IDs, hashes, "
            "group tables, candidate wrappers, or extra fields. Formal IDs are assigned by the program.\n"
            "dependencies is the single relation representation. A natural_continuity relation means "
            "the earlier candidate is suitable to write immediately before this one; condition must be null. "
            "A hard_prerequisite relation names the concrete actual result required before this candidate "
            "can proceed; condition must be nonempty. Do not turn natural transitions into hard requirements.\n"
            "Process selected_continuations then new_development_candidates in array order. "
            "A continuation means continuation of its historical verified track, not automatic "
            "dependency on another current candidate. No self, forward or unknown references. "
            "Empty dependencies means no relation declared, not proof of independence.\n"
            "Topic similarity, the same participants, or a continuation label alone does not establish "
            "a dependency. A prerequisite already true in authoritative history need not depend on a "
            "new scene. Explain the needed old facts in the existing state or transition fields.\n"
            "In action-bearing text use the exact canonical person names from speaker_registry or "
            "the exact new_people declarations. Transport roles user and assistant do not identify "
            "story participants; never infer an actor from them. Skip an unidentified actor rather "
            "than guessing. The schema constrains fields only; derive story content from the supplied "
            "history, Memory navigation and feedback, not its descriptions or example story.\n"
            "JSON SCHEMA\n" + json.dumps(schema, ensure_ascii=False, indent=2)
            + "\nEND JSON SCHEMA\n"
            "JSON FIELD EXAMPLE (syntax only; derive actual content from the supplied input)\n"
            + json.dumps(output_examples()[-1], ensure_ascii=False, indent=2)
            + "\nEND JSON FIELD EXAMPLE\n")


def cap9_output_schema(*, initial=False):
    """Explicit cap-9 schema helpers; legacy callers remain cap-8."""
    if initial:
        return initial_output_schema(max_new_candidates=CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES,
                                     contract_version=CAP9_VERSION)
    return output_schema(max_new_candidates=CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES,
                         contract_version=CAP9_VERSION)

def validate_initial_output_cap9(value):
    return validate_initial_output(value, max_new_candidates=CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES,
                                   contract_version=CAP9_VERSION)

def _parsed(raw):
    value = legacy_contract.parse_output(raw) if isinstance(raw, str) else deepcopy(raw)
    json.dumps(value, allow_nan=False)
    return value


def _failure(raw, errors, *, contract_version=VERSION):
    return {"valid": False, "errors": list(errors), "planner_contract_version": contract_version,
            "shared_output": deepcopy(raw)}


def _relation_errors(raw, source_refs=()):
    earlier, errors = set(), []
    for pointer, row, _ in legacy_contract.iter_candidates(raw):
        seen = set()
        for relation in row["dependencies"]:
            source, output = relation["source_candidate_ref"], relation["output_ref"]
            if source is not None and source not in source_refs:
                errors.append(pointer + ":unknown_or_forbidden_dependency_source:" + source)
            if output is not None and output == pointer:
                errors.append(pointer + ":self_output_dependency_forbidden:" + output)
            elif output is not None and output not in earlier:
                errors.append(pointer + ":unknown_self_or_forward_output_dependency:" + output)
            identity = (source, output)
            if identity in seen:
                errors.append(pointer + ":duplicate_predecessor_relation")
            seen.add(identity)
        earlier.add(pointer)
    # Backward-only references also rule out every directed cycle.
    return errors


def _execution_projection(raw):
    projected = deepcopy(raw)
    for _, row, _ in legacy_contract.iter_candidates(projected):
        row["dependencies"] = [
            {key: value for key, value in relation.items() if key != "relation_kind"}
            for relation in row["dependencies"] if relation["relation_kind"] == HARD]
    return projected


def _sidecar(raw, phase_map, candidate_map, source_refs=()):
    nodes, relations, errors = [], [], []
    for pointer, row, _ in legacy_contract.iter_candidates(raw):
        phase_id = phase_map.get(pointer)
        nodes.append({"output_ref": pointer, "phase_id": phase_id,
                      "candidate_ref": candidate_map.get(pointer), "scene_ref": None,
                      "relations_status": "declared" if row["dependencies"] else "undeclared"})
        for relation in row["dependencies"]:
            source, output = relation["source_candidate_ref"], relation["output_ref"]
            predecessor = "initial:" + source if source is not None else phase_map.get(output)
            if phase_id is None or predecessor is None:
                errors.append(pointer + ":relation_owner_or_target_unavailable")
            relations.append({"owner_output_ref": pointer, "owner_phase_id": phase_id,
                              "predecessor_phase_id": predecessor, **deepcopy(relation)})
    return {"version": VERSION, "raw_output_sha256": sha(raw), "nodes": nodes,
            "relations": relations, "output_phase_map": deepcopy(phase_map),
            "output_candidate_map": deepcopy(candidate_map),
            "source_candidate_phase_map": {ref: "initial:" + ref for ref in source_refs},
            "scene_mapping_status": "not_materialized",
            "empty_relation_meaning": "undeclared_not_proven_independent"}, errors


def compile_initial_output(raw, *, parent, candidates, verified_tracks, max_new_candidates=MAX_NEW_DEVELOPMENT_CANDIDATES):
    contract_version = CAP9_VERSION if max_new_candidates == CAP9_MAX_NEW_DEVELOPMENT_CANDIDATES else VERSION
    try:
        raw = _parsed(raw)
    except (ValueError, TypeError) as exc:
        return _failure(raw, ["planner_relation_json_invalid:" + str(exc)], contract_version=contract_version)
    structural = validate_initial_output(raw, max_new_candidates=max_new_candidates, contract_version=contract_version)
    if not structural["valid"]:
        return _failure(raw, structural["errors"], contract_version=contract_version)
    errors = _relation_errors(raw)
    if errors:
        return _failure(raw, errors, contract_version=contract_version)
    result = legacy_planning.compile_initial_output(_execution_projection(raw), parent=parent,
        candidates=candidates, verified_tracks=verified_tracks, max_new_candidates=max_new_candidates)
    if not result["valid"]:
        return _failure(raw, result["errors"], contract_version=contract_version)
    prepared = result["prepared"]
    phase_map = prepared["initial_output_phase_map"]
    candidate_map = {pointer: phase.removeprefix("initial:") for pointer, phase in phase_map.items()}
    sidecar, errors = _sidecar(raw, phase_map, candidate_map)
    if errors:
        return _failure(raw, errors, contract_version=contract_version)
    sidecar["version"] = contract_version
    prepared.update(planner_contract_version=contract_version, planner_relation_sidecar=deepcopy(sidecar),
                    _initial_relation_output=deepcopy(raw))
    result.update(planner_contract_version=contract_version, shared_output=deepcopy(raw), relation_sidecar=sidecar)
    return result


def merge_output(prepared, raw):
    try:
        raw = _parsed(raw)
    except (ValueError, TypeError) as exc:
        return _failure(raw, ["planner_relation_json_invalid:" + str(exc)])
    structural = validate_output(raw)
    if not structural["valid"]:
        return _failure(raw, structural["errors"])
    source_refs = list(prepared["candidate_registry"])
    errors = _relation_errors(raw, source_refs)
    if errors:
        return _failure(raw, errors)
    before = sha(prepared)
    result = legacy_planning.merge_shared_output(
        prepared, _execution_projection(raw), allow_unconditional_continuations=True)
    if sha(prepared) != before:
        raise ValueError("planner_relation_merge_mutated_original")
    if not result["valid"]:
        return _failure(raw, result["errors"])
    phase_map = result["output_phase_map"]
    phase_candidates = {entry["scene"]["phase_id"]: entry["scene"]["source_candidate_id"]
                        for entry in result["merged"]["added_phases"]}
    candidate_map = {pointer: phase_candidates[phase] for pointer, phase in phase_map.items()}
    sidecar, errors = _sidecar(raw, phase_map, candidate_map, source_refs)
    if errors:
        return _failure(raw, errors)
    result.update(planner_contract_version=VERSION, shared_output=deepcopy(raw),
                  raw_patch=deepcopy(raw), relation_sidecar=sidecar)
    result["merged"].update(planner_contract_version=VERSION, planner_relation_sidecar=deepcopy(sidecar))
    result["merged"]["version_chain"][-1].update(patch_sha256=sha(raw))
    result["merged_sha256"] = sha(result["merged"])
    return result


def map_formal_scene_ids(sidecar, phase_scene_map):
    """Attach actual materializer-assigned IDs; never predict them from row order."""
    result = deepcopy(sidecar)
    allowed = set(result["output_phase_map"].values()) | set(result["source_candidate_phase_map"].values())
    if (not isinstance(phase_scene_map, dict) or set(phase_scene_map) - allowed
            or any(not isinstance(value, str) or not value.strip() for value in phase_scene_map.values())
            or len(set(phase_scene_map.values())) != len(phase_scene_map)):
        raise ValueError("planner_relation_scene_mapping_invalid")
    for node in result["nodes"]:
        node["scene_ref"] = phase_scene_map.get(node["phase_id"])
    for relation in result["relations"]:
        relation["owner_scene_ref"] = phase_scene_map.get(relation["owner_phase_id"])
        relation["predecessor_scene_ref"] = phase_scene_map.get(relation["predecessor_phase_id"])
    result["phase_scene_map"] = deepcopy(phase_scene_map)
    endpoints_mapped = all(
        relation["owner_scene_ref"] is not None and relation["predecessor_scene_ref"] is not None
        for relation in result["relations"])
    result["scene_mapping_status"] = (
        "complete" if endpoints_mapped and all(node["scene_ref"] is not None for node in result["nodes"])
        else "partial")
    return result


def compile_versioned_output(raw, *, version, prepared=None, **initial_inputs):
    """Historical v2 is explicit and is never silently upgraded."""
    if version == VERSION:
        return merge_output(prepared, raw) if prepared is not None else compile_initial_output(raw, **initial_inputs)
    if version == legacy_contract.VERSION:
        return (legacy_planning.merge_shared_output(prepared, raw) if prepared is not None
                else legacy_planning.compile_initial_output(raw, **initial_inputs))
    raise ValueError("unknown_planner_relation_contract:" + str(version))
