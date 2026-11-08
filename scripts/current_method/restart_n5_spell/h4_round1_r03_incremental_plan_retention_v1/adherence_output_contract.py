"""Input-specific audit schema and lossless reference-list deduplication."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator

from .retention import ADHERENCE_VERSION, _history_refs, render_adherence
from .shared_planner_contract import parse_output


VERSION = "adherence.input_schema_v1"


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _enum(values):
    values = sorted(set(values))
    return {"type": "string", "enum": values} if values else {"not": {}}


def output_schema(*, parent, child, requested_conditions):
    history = _history_refs(parent)
    scenes = child["scenes"]
    if not scenes or len({s["scene_ref"] for s in scenes}) != len(scenes):
        raise ValueError("audit_schema_scene_identity")
    if set(requested_conditions) != {s["scene_ref"] for s in scenes}:
        raise ValueError("audit_schema_conditions_scene_coverage")
    definitions = {"history_ref": _enum(history)} if history else {}
    rows = []
    for index, scene in enumerate(scenes):
        current = {t["turn_ref"] for t in scene["dialogue_turns"]}
        if not current or current & history:
            raise ValueError("audit_schema_current_reference_identity")
        expected = requested_conditions[scene["scene_ref"]]
        if (not isinstance(expected, list) or any(not isinstance(c, str) or not c.strip() for c in expected)
                or len(expected) != len(set(expected))):
            raise ValueError("audit_schema_conditions_shape")
        definitions[f"scene_{index}_ref"] = _enum(current)
        current_item = {"$ref": f"#/$defs/scene_{index}_ref"}
        supported_item = {"anyOf": [current_item, {"$ref": "#/$defs/history_ref"}]} if history else current_item
        def refs(item):
            return {"type": "array", "items": deepcopy(item)}
        conditions = {"type": "array", "minItems": len(expected), "maxItems": len(expected),
            "items": _object({"condition": _enum(expected),
                "status": _enum(("established", "not_established", "UNKNOWN")),
                "turn_refs": refs(supported_item)})}
        if not expected:
            # Keep an ordinary item type for providers that reject an empty enum.
            conditions["items"] = _object({})
        rows.append(_object({
            "scene_ref": _enum((scene["scene_ref"],)),
            "goal_status": _enum(("realized", "partial", "not_realized", "UNKNOWN")),
            "content_decision": _enum(("retain", "reject", "insufficient")),
            "participant_adherence": _enum(("pass", "fail", "UNKNOWN")),
            "history_consistency": _enum(("pass", "fail", "UNKNOWN")),
            "novel_progress": _enum(("pass", "fail", "UNKNOWN")),
            "actual_states": {"type": "array", "items": _object({
                "statement": {"type": "string", "minLength": 1},
                "stage": _enum(("intention", "arrangement", "action", "completed")),
                "turn_refs": {**refs(current_item), "minItems": 1}})},
            "goal_evidence_refs": refs(supported_item),
            "history_evidence_refs": refs({"$ref": "#/$defs/history_ref"}) if history else {
                "type": "array", "maxItems": 0, "items": {"type": "string"}},
            "condition_results": conditions,
            "reason": {"type": "string", "minLength": 1},
        }))
    schema = _object({"scene_audits": {"type": "array", "minItems": len(rows),
        "maxItems": len(rows), "items": {"anyOf": rows}}})
    schema["$defs"] = definitions
    Draft202012Validator.check_schema(schema)
    return schema


def render_contract(*, parent, plan, child, requested_conditions, generated_response):
    built = render_adherence(parent=parent, plan=plan, child=child,
        requested_conditions=requested_conditions, generated_response=generated_response,
        prompt_version=ADHERENCE_VERSION)
    schema = output_schema(parent=parent, child=child, requested_conditions=requested_conditions)
    brief = [{"scene_ref": scene["scene_ref"], "original_goal": target["goal"],
        "required_condition_texts": requested_conditions[scene["scene_ref"]],
        "condition_results_must_equal_empty_list": not requested_conditions[scene["scene_ref"]]}
        for scene, target in zip(child["scenes"], plan["scenes"])]
    if [s["scene_ref"] for s in child["scenes"]] != [s["scene_ref"] for s in plan["scenes"]]:
        raise ValueError("audit_schema_plan_scene_identity")
    built["messages"].append({"role": "user", "content":
        "Return one complete JSON audit in the input scene order. All eleven fields are required for every scene. "
        "The following summary repeats the original goals and the only requested conditions; "
        "do not turn a goal into a condition. [] means there are NO downstream conditions to evaluate. "
        "Do not add success criteria to goal_status or reason. A goal to plan or discuss does not require "
        "the planned activity to have happened; distinguish that original goal from durable planned targets. "
        "A plan is not evidence of action. In reason, compare the exact original goal with the actual stage "
        "observed, then explain any new progress versus history. All evidence ID lists must be unique.\n" +
        json.dumps(brief, ensure_ascii=False, separators=(",", ":"))})
    built["prompt_version"] = VERSION
    built["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "adherence_input_contract", "strict": True, "schema": schema}}
    built["audit_version"] = ADHERENCE_VERSION
    return built


def normalize_references(raw):
    """Remove identical repeated IDs only; never remove conditions or judgments."""
    value = parse_output(raw) if isinstance(raw, str) else deepcopy(raw)
    json.dumps(value, allow_nan=False)
    original = deepcopy(value)
    changes = []
    if not isinstance(value, dict) or not isinstance(value.get("scene_audits"), list):
        return {"raw_output": original, "formatted_output": value, "transformations": changes}

    def deduplicate(container, field, path):
        refs = container.get(field)
        if not isinstance(refs, list) or not all(type(ref) is str for ref in refs):
            return
        unique = list(dict.fromkeys(refs))
        if unique != refs:
            changes.append({"operation": "deduplicate_references", "path": path, "from": refs, "to": unique})
            container[field] = unique

    for index, row in enumerate(value["scene_audits"]):
        if not isinstance(row, dict):
            continue
        for field in ("goal_evidence_refs", "history_evidence_refs"):
            deduplicate(row, field, f"/scene_audits/{index}/{field}")
        for field in ("actual_states", "condition_results"):
            items = row.get(field)
            if isinstance(items, list):
                for number, item in enumerate(items):
                    if isinstance(item, dict):
                        deduplicate(item, "turn_refs", f"/scene_audits/{index}/{field}/{number}/turn_refs")
    return {"raw_output": original, "formatted_output": value, "transformations": changes}


def validate_response(value, *, schema, scene_order):
    errors = [{"path": list(error.absolute_path), "message": error.message}
              for error in Draft202012Validator(schema).iter_errors(value)]
    rows = value.get("scene_audits", []) if isinstance(value, dict) else []
    if not isinstance(rows, list) or [r.get("scene_ref") if isinstance(r, dict) else None for r in rows] != scene_order:
        errors.append({"path": ["scene_audits"], "message": "scene order or coverage mismatch"})
    return {"valid": not errors, "errors": errors}
