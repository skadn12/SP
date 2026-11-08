"""Lossless, deterministic model projection; original runtime objects stay intact."""
from copy import deepcopy
import hashlib
import json

INPUT_VIEW_VERSION = "incremental_lossless_input_view_v2"
REFERENCE_KEY = "$input_ref"
REFERENCE_RULE = (
    "Objects containing only $input_ref reuse the exact value at that JSON "
    "Pointer in this INPUT DATA. They do not omit or replace the referenced "
    "content. Read original_plan once; registry differences remain explicit. "
    "History groups are consecutive: common fields apply to every row, and "
    "turn_columns name each row value in order. All wording is unchanged."
)
_BACKEND_ONLY = {"frozen_sources", "fixture_notice", "source_paths", "artifact_paths"}
_HISTORY_VARIABLE = {"turn_id", "turn_ref", "speaker", "text", "source_turn_index"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _path(parent, key):
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _parts(pointer):
    if not pointer.startswith("/"):
        raise ValueError("input_reference_absolute_pointer_required")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _leaves(value, pointer=""):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _leaves(item, _path(pointer, key))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, _path(pointer, index))
    else:
        yield pointer, value


def _assert_unambiguous_references(value):
    if isinstance(value, dict):
        if REFERENCE_KEY in value:
            raise ValueError("input_reserved_reference_key")
        for item in value.values():
            _assert_unambiguous_references(item)
    elif isinstance(value, list):
        for item in value:
            _assert_unambiguous_references(item)


def _resolve_at(document, pointer, visited=()):
    if pointer in visited:
        raise ValueError("input_reference_cycle")
    value = document
    traversed = ""
    parts = _parts(pointer)
    for index in range(len(parts) + 1):
        if isinstance(value, dict) and set(value) == {REFERENCE_KEY}:
            tail = "".join(_path("", part) for part in parts[index:])
            return _resolve_at(document, value[REFERENCE_KEY] + tail, visited + (pointer,))
        if index == len(parts):
            return value, traversed
        part = parts[index]
        value = value[int(part)] if isinstance(value, list) else value[part]
        traversed = _path(traversed, part)


def _register(value, pointer, index):
    index.setdefault(_json(value), pointer)
    if isinstance(value, dict):
        for key, item in value.items():
            _register(item, _path(pointer, key), index)
    elif isinstance(value, list):
        for position, item in enumerate(value):
            _register(item, _path(pointer, position), index)


def _deduplicate(value, pointer, index, references):
    target = index.get(_json(value))
    name = _parts(pointer)[-1]
    identity = name.endswith(("_id", "_ref", "_sha256", "_key"))
    eligible = (isinstance(value, str) and len(value) >= 40 and not identity) or (
        isinstance(value, (dict, list)) and bool(value))
    if target is not None and target != pointer and eligible:
        replacement = {REFERENCE_KEY: target}
        if len(_json(replacement)) < len(_json(value)):
            references.append({"original_field": pointer, "unique_value_location": target,
                               "comparison": "exact_typed_json_equality"})
            return replacement
    if isinstance(value, dict):
        projected = {key: _deduplicate(item, _path(pointer, key), index, references)
                     for key, item in value.items()}
    elif isinstance(value, list):
        projected = [_deduplicate(item, _path(pointer, position), index, references)
                     for position, item in enumerate(value)]
    else:
        projected = deepcopy(value)
    index.setdefault(_json(value), pointer)
    return projected


def _project_history(turns):
    if not isinstance(turns, list):
        raise ValueError("input_history_list_required")
    groups = []
    field_locations = {}
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict) or not isinstance(turn.get("text"), str) or not isinstance(turn.get("speaker"), str):
            raise ValueError("input_history_turn_shape")
        if not (turn.get("turn_id") or turn.get("turn_ref")):
            raise ValueError("input_history_turn_identity")
        common = {key: deepcopy(value) for key, value in turn.items() if key not in _HISTORY_VARIABLE}
        columns = [key for key in turn if key in _HISTORY_VARIABLE]
        if not groups or groups[-1]["common"] != common or groups[-1]["turn_columns"] != columns:
            groups.append({"common": common, "turn_columns": columns, "turns": []})
        group = groups[-1]
        row_index = len(group["turns"])
        group["turns"].append([deepcopy(turn[key]) for key in columns])
        group_pointer = "/authoritative_history/groups/" + str(len(groups) - 1)
        for key in turn:
            source = _path("/authoritative_history/" + str(turn_index), key)
            target = (_path(group_pointer + "/common", key) if key in common else
                      group_pointer + "/turns/" + str(row_index) + "/" + str(columns.index(key)))
            for suffix, _ in _leaves(turn[key]):
                field_locations[source + suffix] = target + suffix
    return {"format": "complete_ordered_turn_table_v1", "turn_count": len(turns),
            "groups": groups}, field_locations


def _original_plan_association(candidate_id, row, plan):
    gate = row.get("gate_scene", {})
    track = gate.get("verified_track_ref")
    if track:
        matches = [index for index, value in enumerate(plan.get("selected_continuations", []))
                   if value.get("verified_track_ref") == track]
        if len(matches) == 1:
            return "/original_plan/selected_continuations/" + str(matches[0])
        raise ValueError("input_registry_track_association")
    event = str(gate.get("memory_event_id", ""))
    if event.startswith("ND") and event[2:].isdigit() and candidate_id == event + ":C01":
        position = int(event[2:]) - 1
        if 0 <= position < len(plan.get("new_development_candidates", [])):
            return "/original_plan/new_development_candidates/" + str(position)
    raise ValueError("input_registry_plan_association")


def reconstruct_supplement_inputs(projected, mapping):
    """Rebuild the original input without using a hidden full copy of its body."""
    def expand(value):
        if isinstance(value, dict) and set(value) == {REFERENCE_KEY}:
            resolved, _ = _resolve_at(projected, value[REFERENCE_KEY])
            return expand(resolved)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return deepcopy(value)

    restored = {key: expand(value) for key, value in projected.items() if key != "input_view_contract"}
    restored.update(deepcopy(mapping["backend_only_fields"]))
    history = restored["authoritative_history"]
    turns = []
    for group in history["groups"]:
        for row in group["turns"]:
            if len(row) != len(group["turn_columns"]):
                raise ValueError("input_history_row_length")
            turns.append({**deepcopy(group["common"]),
                          **dict(zip(group["turn_columns"], deepcopy(row)))})
    if len(turns) != history["turn_count"]:
        raise ValueError("input_history_count_mismatch")
    restored["authoritative_history"] = turns
    for row in restored["stable_candidates"].values():
        row.pop("original_plan_input_pointer", None)
    return restored


def project_supplement_inputs(inputs, *, extra_reference_fields=()):
    """Return model data and a backend-only field map with exact reconstruction proof."""
    _assert_unambiguous_references(inputs)
    if not isinstance(inputs.get("original_plan"), dict) or not isinstance(inputs.get("stable_candidates"), dict):
        raise ValueError("input_projection_plan_and_registry_required")
    projected = {key: deepcopy(value) for key, value in inputs.items() if key not in _BACKEND_ONLY}
    mapping = {"version": INPUT_VIEW_VERSION, "source_sha256": _sha(inputs),
               "backend_only_fields": {key: deepcopy(value) for key, value in inputs.items() if key in _BACKEND_ONLY},
               "reference_replacements": [], "fields": []}
    index = {}
    _register(inputs["original_plan"], "/original_plan", index)
    registry = {}
    for candidate_id, row in inputs["stable_candidates"].items():
        if "original_plan_input_pointer" in row:
            raise ValueError("input_registry_reserved_association_key")
        pointer = _path("/stable_candidates", candidate_id)
        registry[candidate_id] = _deduplicate(row, pointer, index, mapping["reference_replacements"])
        registry[candidate_id]["original_plan_input_pointer"] = _original_plan_association(candidate_id, row, inputs["original_plan"])
    projected["stable_candidates"] = registry
    for field in extra_reference_fields:
        if field not in projected or field in {"original_plan", "stable_candidates", "authoritative_history"}:
            raise ValueError("extra_reference_field_not_available")
        projected[field] = _deduplicate(inputs[field], _path("", field), index, mapping["reference_replacements"])
    projected["authoritative_history"], locations = _project_history(inputs["authoritative_history"])
    projected["input_view_contract"] = {"version": INPUT_VIEW_VERSION, "reference_and_history_rules": REFERENCE_RULE}
    for source, value in _leaves(inputs):
        top = _parts(source)[0]
        if top in _BACKEND_ONLY:
            mapping["fields"].append({"original_field": source, "duplicate": False,
                "disposition": "backend_only", "unique_value_location": "/backend_only_fields" + source})
            continue
        target = locations.get(source, source)
        actual, target = _resolve_at(projected, target)
        if _json(actual) != _json(value):
            raise ValueError("input_projection_field_changed:" + source)
        mapping["fields"].append({"original_field": source, "duplicate": target != source and top != "authoritative_history",
            "disposition": "model_visible_exact_value", "unique_value_location": target})
    restored = reconstruct_supplement_inputs(projected, mapping)
    if _json(restored) != _json(inputs):
        raise ValueError("input_projection_not_lossless")
    mapping.update(projected_sha256=_sha(projected), reconstruction_sha256=_sha(restored),
                   reconstruction_exact=True, original_candidate_count=len(registry),
                   original_turn_count=len(inputs["authoritative_history"]))
    return projected, mapping
