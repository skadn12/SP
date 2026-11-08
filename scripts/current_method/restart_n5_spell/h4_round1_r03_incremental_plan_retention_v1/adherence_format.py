"""Offline-only structural recovery of saved adherence responses.

This module does not change judgments or promote dialogue. A semantic review
is required when formatting excludes conditions invented by the old auditor.
"""
from copy import deepcopy
import json


VERSION = "adherence_saved_formatting_v2"
PROVENANCE_KEY = "_formatting_provenance"
PROVENANCE_KEYS = {"formatter_version", "semantic_reaudit_required_scene_refs", "transformations"}
AUDIT_KEYS = {"scene_ref", "goal_status", "content_decision", "participant_adherence",
              "history_consistency", "novel_progress", "actual_states",
              "goal_evidence_refs", "condition_results", "reason"}


def _parse(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key:" + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("invalid_json_constant:" + value)

    if isinstance(raw, str):
        value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        json.dumps(value, allow_nan=False)
        return value
    if not isinstance(raw, dict):
        raise ValueError("audit_object_required")
    encoded = json.dumps(raw, allow_nan=False)
    if json.loads(encoded) != raw:
        raise ValueError("non_json_value")
    return deepcopy(raw)


def _refs(value):
    return isinstance(value, list) and all(isinstance(ref, str) and ref for ref in value)


def validate_formatting_provenance(output, scene_refs):
    """Validate durable recovery metadata shared with the retention assessor.

    Raw model output has no provenance. Once this field exists its full shape
    and the link between excluded conditions and blocked scenes are mandatory.
    """
    if PROVENANCE_KEY not in output:
        return []
    provenance = output[PROVENANCE_KEY]
    if (not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS
            or provenance["formatter_version"] != VERSION):
        raise ValueError("formatting_provenance_shape")
    required = provenance["semantic_reaudit_required_scene_refs"]
    changes = provenance["transformations"]
    if (not _refs(required) or len(required) != len(set(required))
            or not set(required).issubset(scene_refs) or not isinstance(changes, list)):
        raise ValueError("formatting_provenance_scene_refs")
    affected = set()
    known_operations = {"rewrite_reference", "flatten_goal_evidence", "deduplicate_goal_references",
                        "exclude_unrequested_condition"}
    for change in changes:
        if (not isinstance(change, dict) or change.get("operation") not in known_operations
                or not isinstance(change.get("path"), str)):
            raise ValueError("formatting_provenance_transformation")
        if change["operation"] == "exclude_unrequested_condition":
            original = change.get("original")
            ref = change.get("scene_ref")
            if (not isinstance(ref, str) or ref not in scene_refs
                    or not isinstance(original, dict) or not isinstance(original.get("condition"), str)):
                raise ValueError("formatting_provenance_excluded_condition")
            parts = change["path"].split("/")
            rows = output.get("scene_audits")
            if (len(parts) != 5 or parts[:2] != ["", "scene_audits"]
                    or parts[3] != "condition_results" or not parts[2].isascii()
                    or not parts[2].isdigit() or not parts[4].isascii() or not parts[4].isdigit()
                    or not isinstance(rows, list) or int(parts[2]) >= len(rows)
                    or not isinstance(rows[int(parts[2])], dict)
                    or rows[int(parts[2])].get("scene_ref") != ref):
                raise ValueError("formatting_provenance_exclusion_identity")
            affected.add(ref)
    if set(required) != affected:
        raise ValueError("formatting_provenance_semantic_coverage")
    return list(required)


def _format_scene(row, scene, expected, other_scene_refs, path, changes, errors):
    if not isinstance(row, dict) or set(row) != AUDIT_KEYS or row["scene_ref"] != scene["scene_ref"]:
        errors.append(path + ":audit_dimensions_missing_or_misaligned")
        return
    if (not isinstance(expected, list) or not all(isinstance(c, str) and c for c in expected)
            or len(expected) != len(set(expected))):
        errors.append(path + ":requested_conditions_shape")
        return
    states, goals, conditions = (row[field] for field in
                                 ("actual_states", "goal_evidence_refs", "condition_results"))
    if not all(isinstance(value, list) for value in (states, goals, conditions)):
        errors.append(path + ":evidence_lists_required")
        return
    original_goals = deepcopy(goals)
    valid_refs = {turn["turn_ref"] for turn in scene["dialogue_turns"]}
    by_text = {}
    for turn in scene["dialogue_turns"]:
        by_text.setdefault(turn["text"], []).append(turn["turn_ref"])

    # Build aliases only from a complete exact quote with one supplied reference
    # and one matching current-scene turn. Never infer aliases by array position.
    aliases, alias_evidence = {}, {}
    evidence = []
    for index, state in enumerate(states):
        pointer = path + f"/actual_states/{index}"
        if (not isinstance(state, dict) or set(state) != {"statement", "stage", "turn_refs"}
                or not isinstance(state["statement"], str) or not _refs(state["turn_refs"])):
            errors.append(pointer + ":actual_state_shape")
        else:
            evidence.append((pointer, state))
    for index, goal in enumerate(goals):
        pointer = path + f"/goal_evidence_refs/{index}"
        if isinstance(goal, str):
            continue
        if (not isinstance(goal, dict) or set(goal) != {"statement", "turn_refs"}
                or not isinstance(goal["statement"], str) or not _refs(goal["turn_refs"])):
            errors.append(pointer + ":goal_evidence_shape")
        else:
            evidence.append((pointer, goal))
    for pointer, item in evidence:
        refs, matches = item["turn_refs"], by_text.get(item["statement"], [])
        bad = [ref for ref in refs if ref not in valid_refs]
        if not bad:
            if len(refs) == 1 and len(matches) == 1 and refs[0] != matches[0]:
                errors.append(pointer + ":contradictory_ref_text")
            continue
        if any(ref in other_scene_refs for ref in bad):
            errors.append(pointer + ":cross_scene_reference")
        elif len(refs) != 1 or len(matches) != 1:
            errors.append(pointer + ":reference_requires_exact_unique_quote")
        elif refs[0] in aliases and aliases[refs[0]] != matches[0]:
            errors.append(pointer + ":contradictory_reference_mapping")
        else:
            aliases[refs[0]] = matches[0]
            alias_evidence[refs[0]] = {"path": pointer, "statement": item["statement"],
                                        "turn_ref": matches[0]}

    def rewrite(refs, pointer):
        if not _refs(refs):
            errors.append(pointer + ":reference_list_required")
            return refs
        mapped = []
        for index, ref in enumerate(refs):
            target = ref if ref in valid_refs else aliases.get(ref)
            if target is None:
                errors.append(pointer + f"/{index}:unresolved_reference")
                mapped.append(ref)
            else:
                mapped.append(target)
                if target != ref:
                    changes.append({"operation": "rewrite_reference", "path": pointer + f"/{index}",
                                    "from": ref, "to": target,
                                    "basis": "exact_unique_current_scene_quote",
                                    "evidence": deepcopy(alias_evidence[ref])})
        return mapped

    for pointer, item in evidence:
        item["turn_refs"] = rewrite(item["turn_refs"], pointer + "/turn_refs")
    flattened = []
    for index, goal in enumerate(goals):
        pointer = path + f"/goal_evidence_refs/{index}"
        if isinstance(goal, str):
            flattened.extend(rewrite([goal], pointer))
        elif isinstance(goal, dict) and _refs(goal.get("turn_refs")):
            flattened.extend(goal["turn_refs"])
            changes.append({"operation": "flatten_goal_evidence", "path": pointer,
                            "original": original_goals[index], "to": deepcopy(goal["turn_refs"])})
    unique = list(dict.fromkeys(flattened))
    if unique != flattened:
        changes.append({"operation": "deduplicate_goal_references", "path": path + "/goal_evidence_refs",
                        "from": flattened, "to": unique})
    row["goal_evidence_refs"] = unique

    selected = []
    for index, condition in enumerate(conditions):
        pointer = path + f"/condition_results/{index}"
        if not isinstance(condition, dict) or not isinstance(condition.get("condition"), str):
            errors.append(pointer + ":condition_shape")
            continue
        if condition["condition"] not in expected:
            changes.append({"operation": "exclude_unrequested_condition", "path": pointer,
                            "scene_ref": scene["scene_ref"], "original": deepcopy(condition)})
            continue
        if set(condition) != {"condition", "status", "turn_refs"}:
            errors.append(pointer + ":condition_shape")
            continue
        condition["turn_refs"] = rewrite(condition["turn_refs"], pointer + "/turn_refs")
        selected.append(condition)
    if sorted(condition["condition"] for condition in selected) != sorted(expected):
        errors.append(path + ":requested_condition_coverage")
    row["condition_results"] = selected


def format_adherence(raw, *, child, requested_conditions):
    """Return a separate formatted audit view, never a retention decision.

    ``raw_output`` preserves the original input, including its exact JSON text.
    A failed conversion exposes no partly repaired ``formatted_output``.
    """
    result = {"valid": False, "errors": [], "raw_output": deepcopy(raw),
              "transformations": [], "formatter_version": VERSION,
              "semantic_reaudit_required": False, "semantic_reaudit_required_scene_refs": []}
    errors = result["errors"]
    try:
        output = _parse(raw)
    except (ValueError, TypeError, OverflowError) as error:
        errors.append("adherence_format_parse:" + str(error))
        return result
    rows = output.get("scene_audits") if isinstance(output, dict) else None
    scenes = child.get("scenes") if isinstance(child, dict) else None
    if (not isinstance(output, dict)
            or set(output) not in ({"scene_audits"}, {"scene_audits", PROVENANCE_KEY})
            or not isinstance(rows, list)):
        errors.append("audit_shape")
        return result
    if not isinstance(scenes, list) or not scenes or len(rows) != len(scenes):
        errors.append("child_or_scene_count")
        return result
    if not isinstance(requested_conditions, dict):
        errors.append("requested_conditions_shape")
        return result
    all_refs, scene_refs = set(), set()
    for scene in scenes:
        if (not isinstance(scene, dict) or not isinstance(scene.get("scene_ref"), str)
                or scene["scene_ref"] in scene_refs or not isinstance(scene.get("dialogue_turns"), list)):
            errors.append("child_scene_identity")
            return result
        scene_refs.add(scene["scene_ref"])
        for turn in scene["dialogue_turns"]:
            if (not isinstance(turn, dict) or not isinstance(turn.get("text"), str)
                    or not isinstance(turn.get("turn_ref"), str) or not turn["turn_ref"]
                    or turn["turn_ref"] in all_refs):
                errors.append("child_turn_identity")
                return result
            all_refs.add(turn["turn_ref"])
    if set(requested_conditions) != scene_refs:
        errors.append("requested_conditions_scene_coverage")
        return result
    try:
        inherited_required = validate_formatting_provenance(output, scene_refs)
    except ValueError as error:
        errors.append(str(error))
        return result
    inherited_changes = deepcopy(output.get(PROVENANCE_KEY, {}).get("transformations", []))
    for index, (row, scene) in enumerate(zip(rows, scenes)):
        others = all_refs - {turn["turn_ref"] for turn in scene["dialogue_turns"]}
        _format_scene(row, scene, requested_conditions[scene["scene_ref"]], others,
                      f"/scene_audits/{index}", result["transformations"], errors)
    affected = set(inherited_required) | {change["scene_ref"] for change in result["transformations"]
                                         if change["operation"] == "exclude_unrequested_condition"}
    required = [scene["scene_ref"] for scene in scenes if scene["scene_ref"] in affected]
    result["semantic_reaudit_required"] = bool(required)
    result["semantic_reaudit_required_scene_refs"] = required
    if not errors:
        changes = inherited_changes + result["transformations"]
        output[PROVENANCE_KEY] = {"formatter_version": VERSION,
                                 "semantic_reaudit_required_scene_refs": required,
                                 "transformations": changes}
        result.update(valid=True, formatted_output=output)
    return result
