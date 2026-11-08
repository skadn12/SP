"""Lossless structural formatting before the legacy append-only merger."""
from copy import deepcopy
import json
import re

from .root_planning import NEW_DEVELOPMENT_KEYS, PHASE_ID_PATTERN


VERSION = "root_supplement_formatting_v1"
TOP_KEYS = {"status", "original_plan_sha256", "phases", "new_developments", "reason"}
PHASE_KEYS = {"phase_id", "source_candidate_id", "original_goal", "required_transition",
              "interaction_beats", "dependencies"}
NEW_ENTRY_KEYS = {"phase_id", "dependencies", "candidate"}


def _parse(raw):
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_json_key:" + key)
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError("invalid_json_constant:" + value)

    if isinstance(raw, str):
        value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        json.dumps(value, allow_nan=False)
        return value
    if not isinstance(raw, dict):
        raise ValueError("root_patch_shape")
    # Reject non-JSON Python values rather than silently coercing them.
    encoded = json.dumps(raw, allow_nan=False)
    if json.loads(encoded) != raw:
        raise ValueError("non_json_value")
    return deepcopy(raw)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _candidate_entry(row, path, transformations, errors):
    if not isinstance(row, dict):
        errors.append(path + ":new_development_shape")
        return
    unknown = set(row) - NEW_ENTRY_KEYS - NEW_DEVELOPMENT_KEYS
    if unknown:
        errors.append(path + ":unknown_fields:" + ",".join(sorted(unknown)))
    if "candidate" not in row:
        if not NEW_DEVELOPMENT_KEYS.intersection(row):
            errors.append(path + ":candidate_required")
            return
        row["candidate"] = {}
        transformations.append({"operation": "create_candidate_wrapper", "path": path + "/candidate"})
    candidate = row["candidate"]
    if not isinstance(candidate, dict):
        errors.append(path + ":candidate_shape")
        return
    unknown = set(candidate) - NEW_DEVELOPMENT_KEYS
    if unknown:
        errors.append(path + "/candidate:unknown_fields:" + ",".join(sorted(unknown)))
    for field in sorted(NEW_DEVELOPMENT_KEYS.intersection(row)):
        source, target = path + "/" + field, path + "/candidate/" + field
        if field in candidate:
            if (type(candidate[field]) is not type(row[field])
                    or json.dumps(candidate[field], sort_keys=True) != json.dumps(row[field], sort_keys=True)):
                errors.append(source + ":conflicting_candidate_field")
                continue
            operation = "collapse_identical_candidate_field"
        else:
            candidate[field] = row[field]
            operation = "move_candidate_field"
        del row[field]
        transformations.append({"operation": operation, "from": source, "to": target})
    if set(candidate) != NEW_DEVELOPMENT_KEYS:
        errors.append(path + ":candidate_fields")
    if set(row) != NEW_ENTRY_KEYS:
        errors.append(path + ":new_development_fields")


def format_root_supplement(raw, *, prepared):
    """Normalize unambiguous wrappers and phase identifiers without inventing content.

    Semantic candidate and actor checks remain the responsibility of
    ``merge_root_patch``. A failed result never exposes a partially formatted
    output. ``raw_output`` preserves the caller's input, including JSON text.
    """
    result = {"valid": False, "errors": [], "raw_output": deepcopy(raw),
              "transformations": [], "phase_id_map": {}, "formatter_version": VERSION}
    errors, transformations = result["errors"], result["transformations"]
    try:
        output = _parse(raw)
    except (ValueError, TypeError, OverflowError) as error:
        errors.append("supplement_format_parse:" + str(error))
        return result
    if not isinstance(output, dict) or set(output) != TOP_KEYS:
        errors.append("root_patch_shape")
        return result
    registry = prepared.get("candidate_registry") if isinstance(prepared, dict) else None
    if not isinstance(registry, dict) or not _text(prepared.get("original_sha256")):
        errors.append("supplement_format_prepared_registry_required")
        return result
    if output["original_plan_sha256"] != prepared["original_sha256"]:
        errors.append("patch_original_sha_mismatch")
    if (not isinstance(output["status"], str) or output["status"] not in {"PATCH", "NO_SUPPLEMENT"}
            or not isinstance(output["phases"], list) or not isinstance(output["new_developments"], list)):
        errors.append("root_patch_status_or_items")
        return result
    if not _text(output["reason"]):
        errors.append("patch_reason_required")
    if (output["status"] == "NO_SUPPLEMENT") != (not output["phases"] and not output["new_developments"]):
        errors.append("root_patch_empty_status_mismatch")

    rows = [(f"/phases/{index}", row, True) for index, row in enumerate(output["phases"])]
    rows += [(f"/new_developments/{index}", row, False)
             for index, row in enumerate(output["new_developments"])]
    raw_ids = set()
    for path, row, is_continuation in rows:
        if not isinstance(row, dict):
            errors.append(path + ":item_shape")
            continue
        if is_continuation:
            if set(row) != PHASE_KEYS:
                errors.append(path + ":phase_fields")
            source_ref = row.get("source_candidate_id")
            source = registry.get(source_ref) if isinstance(source_ref, str) else None
            if not source:
                errors.append(path + ":unknown_source")
            elif row.get("original_goal") != source["scene"]["goal"]:
                errors.append(path + ":original_goal_changed")
        else:
            _candidate_entry(row, path, transformations, errors)
        phase_id = row.get("phase_id")
        if not _text(phase_id):
            errors.append(path + ":phase_id_required")
        elif phase_id.strip().startswith("initial:"):
            errors.append(path + ":reserved_initial_phase_id")
        elif phase_id in raw_ids:
            errors.append(path + ":duplicate_phase_id")
        else:
            raw_ids.add(phase_id)
        dependencies = row.get("dependencies")
        if not isinstance(dependencies, list):
            errors.append(path + ":dependencies_required")
        elif is_continuation and not dependencies:
            errors.append(path + ":explicit_dependency_required")
        else:
            for index, dependency in enumerate(dependencies):
                if (not isinstance(dependency, dict) or set(dependency) != {"phase_id", "condition"}
                        or not _text(dependency["phase_id"]) or not _text(dependency["condition"])):
                    errors.append(path + f"/dependencies/{index}:dependency_shape")
    if errors:
        return result

    # Reserve every valid supplied identifier before assigning any replacements.
    # This prevents an early malformed ID from claiming a later valid ID.
    reserved = {ref for ref in raw_ids if re.fullmatch(PHASE_ID_PATTERN, ref)}
    id_map = result["phase_id_map"]
    for ordinal, (path, row, _) in enumerate(rows, 1):
        old_id = row["phase_id"]
        new_id = old_id
        if not re.fullmatch(PHASE_ID_PATTERN, old_id):
            base = f"phase:formatted_{ordinal}"
            new_id, suffix = base, 2
            while new_id in reserved:
                new_id = f"{base}_{suffix}"
                suffix += 1
            reserved.add(new_id)
            transformations.append({"operation": "rename_phase_id", "path": path + "/phase_id",
                                    "from": old_id, "to": new_id})
        id_map[old_id] = new_id

    initial_refs = {"initial:" + ref for ref in registry}
    known = set(initial_refs)
    for path, row, _ in rows:
        current_id = id_map[row["phase_id"]]
        row["phase_id"] = current_id
        for index, dependency in enumerate(row["dependencies"]):
            original_ref = dependency["phase_id"]
            target = id_map.get(original_ref, original_ref)
            pointer = path + f"/dependencies/{index}/phase_id"
            if original_ref not in id_map and original_ref not in initial_refs:
                errors.append(pointer + ":unknown_dependency")
            elif target == current_id:
                errors.append(pointer + ":self_dependency")
            elif target not in known:
                kind = "forward_dependency" if original_ref in id_map else "unknown_dependency"
                errors.append(pointer + ":" + kind)
            if target != original_ref:
                dependency["phase_id"] = target
                transformations.append({"operation": "rewrite_dependency_phase_id", "path": pointer,
                                        "from": original_ref, "to": target})
        known.add(current_id)
    if errors:
        return result
    result.update(valid=True, formatted_output=output)
    return result
