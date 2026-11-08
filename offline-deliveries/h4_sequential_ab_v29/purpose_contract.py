"""Small card-first proposal contract. Proposals are never historical facts."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path

FIELDS = ("root_id", "route", "scene_id", "dataset_card_id", "trigger", "goal")
FEEDBACK_FIELDS = ("evaluation_run_id", "target_solver_model", "target_solver_identity_sha256",
                   "root_id", "route", "guidance")


def project_feedback(feedback_dir, root, feedback_module):
    """Copy only fixed guidance and identity fields, with no semantic rewrite."""
    feedback_dir = Path(feedback_dir)
    compiled = json.loads((feedback_dir / "COMPILED_FEEDBACK.json").read_text())
    verified = feedback_module.load_feedback_context(feedback_dir, root, compiled["target_solver_model"],
                                                     compiled["target_solver_identity_sha256"])
    original = verified["projections"][("dataset_new_event", "planner")]
    projection = {key: deepcopy(original[key]) for key in FEEDBACK_FIELDS}
    source = feedback_dir / root / "dataset_new_event__planner.txt"
    digest = lambda value: hashlib.sha256(value).hexdigest()
    serialized = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"projection_policy": "verbatim_guidance_and_identity_only_no_history_examples_or_statistics",
            "source_path": str(source), "source_sha256": digest(source.read_bytes()),
            "projected_feedback": projection, "projected_feedback_sha256": digest(serialized.encode()),
            "guidance_verbatim": True, "current_generation_feedback_used": False}


def feedback_suffix(record):
    return ("\n[PURPOSE_PREVIOUS_ROUND_GUIDANCE_PROJECTION]\n"
            + json.dumps(record["projected_feedback"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n[/PURPOSE_PREVIOUS_ROUND_GUIDANCE_PROJECTION]\n")


class PurposeValidationError(ValueError):
    def __init__(self, errors):
        self.errors = deepcopy(errors)
        super().__init__(errors[0]["code"])


def output_schema():
    return {"type": "object", "required": list(FIELDS), "additionalProperties": False,
            "properties": {key: {"type": "string", "minLength": 1} for key in FIELDS}}


def validate_purpose(raw, root, route, scene_id, card):
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as error:
        raise PurposeValidationError([{"code": "purpose_json_invalid", "path": "response",
                                       "actual": str(error)}]) from None
    errors = []
    if not isinstance(value, dict):
        raise PurposeValidationError([{"code": "purpose_object_required", "path": "response", "actual": value}])
    for key in FIELDS:
        if key not in value:
            errors.append({"code": "purpose_missing_field", "path": f"response.{key}", "missing": [key]})
        elif not isinstance(value[key], str) or not value[key].strip():
            errors.append({"code": "purpose_nonblank_string_required", "path": f"response.{key}", "actual": value[key]})
    for key in value:
        if key not in FIELDS:
            errors.append({"code": "purpose_unexpected_field", "path": f"response.{key}", "actual": value[key]})
    for key, expected in (("root_id", root), ("route", route), ("scene_id", scene_id),
                          ("dataset_card_id", card["card_id"])):
        if key in value and value[key] != expected:
            errors.append({"code": "purpose_identity_mismatch", "path": f"response.{key}",
                           "actual": value[key], "allowed": [expected]})
    if errors:
        raise PurposeValidationError(errors)
    return deepcopy(value)


def build_request(root, scene_id, card, allowed_speakers, model):
    from planner_contract import mapping_context
    context = {"root_id": root, "route": "dataset_new_event", "scene_id": scene_id,
               "allowed_speakers": list(allowed_speakers),
               "dataset_card": mapping_context(card)["card_guidance"]}
    instruction = (
        "Propose ONE concrete new event from the assigned dataset card before adapting it to root history. "
        "Return only the exact purpose schema below. trigger is a specific initiating circumstance; "
        "goal is the concrete problem/activity this event will accomplish in its own right. "
        "Instantiate a seed's core premise or a scaffold's actual actions and relations. "
        "Do not replace the assigned event with an example event in previous-round feedback. "
        "Feedback guides error avoidance only. Allowed speakers identify the people, not their past. "
        "No root dialogue or state is supplied at this proposal stage, so make no assertions about "
        "their past commitments or completed events. Start with a NEW initiating report, incident or "
        "occasion; do not invent earlier conversations, prior reassurance or previous unsuccessful "
        "attempts by these speakers as its premise. Card history belongs to its source characters, "
        "not automatically to these root speakers. This proposal is not history and has not been "
        "accepted; the final planner must later read the complete root history and adapt or abstain.\n"
        "PURPOSE_OUTPUT_SCHEMA_JSON\n" + json.dumps(output_schema(), ensure_ascii=False)
        + "\nPURPOSE_INPUT_JSON\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":")))
    return {"model": model, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": instruction}]}
