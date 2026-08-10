"""Operator-specific single-arc dependency planner contracts."""

from __future__ import annotations

from typing import Any, Mapping

from .core import canonical_json
from .dependency_conditioned_child_canary import normalize_text
from .outline_dependency_mainline import CHILD_SCHEMA, REALIZER_SCHEMA, validate_child_graph


VERSION = "single_arc_operator_specific_typed_dependency_planner_v1"
PROMPT_VERSION = "spell_single_arc_typed_dependency_planner_v1"
SCHEMA_VERSION = "single_arc_typed_dependency_object_v1"


def _event_schema(operator: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    b = candidate["required_child_state_B"]
    if operator == "qualification_requirement":
        props = {"requirement_key": {"const": b["requirement_key"]},
            "required_value": {"const": b["required_value"]},
            "time_relation": {"const": "later"}}
    elif operator == "commitment_outcome":
        props = {"action_key_ref": {"const": b["action_key_ref"]},
            "outcome_state": {"enum": ["completed", "not_completed"]},
            "time_relation": {"const": "later"}}
    else:
        raise ValueError(f"unsupported_typed_operator:{operator}")
    return {"type": "object", "additionalProperties": False,
        "required": list(props), "properties": props}


def schema_for(candidate: Mapping[str, Any]) -> dict[str, Any]:
    operator = candidate["preferred_operator"]
    valid = {"type": "object", "additionalProperties": False,
        "required": ["status", "parent_anchor_ids", "operator", "planned_child_event"],
        "properties": {"status": {"const": "valid"},
            "parent_anchor_ids": {"type": "array", "minItems": 1, "maxItems": 1,
                "uniqueItems": True, "items": {"enum": candidate["candidate_parent_anchor_ids"]}},
            "operator": {"const": operator},
            "planned_child_event": _event_schema(operator, candidate)}}
    empty = {"type": "object", "additionalProperties": False,
        "required": ["status"], "properties": {"status": {"const": "no_valid_dependency"}}}
    return {"oneOf": [valid, empty]}


def validate(raw: Any, candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"decision": "typed_dependency_schema_invalid", "reasons": ["root"]}
    if raw == {"status": "no_valid_dependency"}:
        return {"decision": "typed_dependency_no_valid_accept", "reasons": []}
    errors = []
    if set(raw) != {"status", "parent_anchor_ids", "operator", "planned_child_event"}:
        errors.append("fields")
    if raw.get("status") != "valid": errors.append("status")
    if raw.get("operator") != candidate["preferred_operator"]: errors.append("operator")
    if raw.get("parent_anchor_ids") not in ([anchor] for anchor in candidate[
            "candidate_parent_anchor_ids"]): errors.append("parent_anchor_ids")
    event = raw.get("planned_child_event")
    expected = candidate["required_child_state_B"]
    if not isinstance(event, dict):
        errors.append("planned_child_event")
    elif raw.get("operator") == "qualification_requirement":
        if set(event) != {"requirement_key", "required_value", "time_relation"}:
            errors.append("planned_child_event_fields")
        if event.get("requirement_key") != expected["requirement_key"]: errors.append("state_key_mismatch")
        if event.get("required_value") != expected["required_value"]: errors.append("required_value_mismatch")
        if event.get("time_relation") != "later": errors.append("time_relation")
    elif raw.get("operator") == "commitment_outcome":
        if set(event) != {"action_key_ref", "outcome_state", "time_relation"}:
            errors.append("planned_child_event_fields")
        if event.get("action_key_ref") != expected["action_key_ref"]: errors.append("action_key_mismatch")
        if event.get("outcome_state") not in {"completed", "not_completed"}: errors.append("outcome_state")
        if event.get("time_relation") != "later": errors.append("time_relation")
    return {"decision": "typed_dependency_accept" if not errors else
        "typed_dependency_schema_invalid", "reasons": sorted(set(errors)),
        **({"typed_dependency": raw} if not errors else {})}


def materialize(raw: Mapping[str, Any], candidate: Mapping[str, Any],
                anchor: Mapping[str, Any], turn: Mapping[str, Any]) -> dict[str, Any]:
    if validate(raw, candidate)["decision"] != "typed_dependency_accept":
        raise ValueError("cannot_materialize_invalid_typed_dependency")
    operator, event = raw["operator"], raw["planned_child_event"]
    if operator == "qualification_requirement":
        parent = {"subject": "John", "predicate": "has_qualification",
            "object": candidate["parent_state_A"]["qualification_key"],
            "value": candidate["parent_state_A"]["value"], "time": "parent"}
        planned = {"subject": "city advisory opening", "predicate": "requires_qualification",
            "object": event["requirement_key"], "value": event["required_value"], "time": "later"}
        result_value = "meets_requirement" if normalize_text(parent["value"]) == normalize_text(
            planned["value"]) else "does_not_meet_requirement"
        answer_type = "boolean"
    else:
        parent = {"subject": candidate["parent_state_A"]["subject"],
            "predicate": "plans_action", "object": candidate["parent_state_A"]["action_key"],
            "value": "planned", "time": "parent"}
        planned = {"subject": candidate["required_child_state_B"]["subject"],
            "predicate": "action_outcome", "object": event["action_key_ref"],
            "value": event["outcome_state"], "time": "later"}
        result_value = "fulfilled" if event["outcome_state"] == "completed" else "not_fulfilled"
        answer_type = "state"
    did = "D" + candidate["arc_id"][1:]
    return {"dependency_id": did, "arc_id": candidate["arc_id"], "status": "valid",
        "operator": operator, "parent_anchor_ids": list(raw["parent_anchor_ids"]),
        "parent_condition": parent, "planned_child_stage": candidate[
            "required_child_state_B"]["allowed_stage"], "planned_child_event": planned,
        "joint_result": {"answer_type": answer_type, "value": result_value,
            "meaning": result_value},
        "unknown_from_parent_only": "later child state is absent",
        "unknown_from_planned_child_only": "earlier parent operand is absent",
        "forbidden_child_restatements": [anchor["text"]],
        "natural_relation_explanation": "program-computed typed dependency",
        "resolved_parent_anchors": [dict(anchor)],
        "resolved_parent_turns": [{**dict(turn), "outline_anchor_id": anchor["anchor_id"]}],
        "canonical_support": [anchor["anchor_id"], candidate[
            "required_child_state_B"]["allowed_stage"]],
        "local_validation": "accept", "local_validation_errors": [],
        "typed_gate": {"T1_arc_program_fixed": True, "T2_anchor_allowed": True,
            "T3_operator_allowed": True, "T4_operator_anchor_compatible": True,
            "T5_future_state": True, "T6_future_not_parent_fact": True,
            "T7_parent_unchanged": True, "T8_child_not_parent_restatement": True,
            "T9_child_does_not_contain_C": True, "T10_parent_only_unknown": True,
            "T11_child_only_unknown": True, "T12_combined_unique": True,
            "T13_no_third_evidence": True, "T14_no_free_causality": True}}


def build_prompt(arc: Mapping[str, Any], anchors: list[Mapping[str, Any]],
                 candidate: Mapping[str, Any], contract: str) -> str:
    material = {"arc": {"event_id": arc["event_id"], "summary": arc["summary"],
        "stages": arc["stages"]}, "candidate_parent_anchors": [{"anchor_id": row[
            "anchor_id"], "speaker": row["speaker"], "text": row["text"]} for row in anchors],
        "preferred_operator": candidate["preferred_operator"],
        "typed_child_event_contract": candidate["required_child_state_B"]}
    return "\n\n".join([f"PROMPT_VERSION: {PROMPT_VERSION}", contract,
        "ASSIGNED ARC AND CANDIDATES\n" + canonical_json(material),
        "OUTPUT JSON SCHEMA\n" + canonical_json(schema_for(candidate)),
        "Return one object. Output no IDs assigned by the program, no quotes, no joint result, and no explanation."])


def build_child_expander_prompt(arc: Mapping[str, Any], dependency: Mapping[str, Any],
                                contract: str) -> str:
    # Deliberately exclude parent_condition, parent text, and joint_result.
    frozen = {"dependency_id": dependency["dependency_id"],
        "arc_id": dependency["arc_id"], "planned_child_stage": dependency[
            "planned_child_stage"], "planned_child_event": dependency["planned_child_event"]}
    arc_view = {"event_id": arc["event_id"], "summary": arc["summary"],
                "stages": arc["stages"]}
    return "\n\n".join([contract, "ASSIGNED ARC\n" + canonical_json(arc_view),
        "FROZEN CHILD-SIDE EVENT B\n" + canonical_json(frozen),
        "OUTPUT JSON SCHEMA\n" + canonical_json(CHILD_SCHEMA),
        "Return one child graph object only."])


def build_realizer_prompt(child_graph: Mapping[str, Any], contract: str) -> str:
    # The Realizer never receives A, C, or the final QA.
    return "\n\n".join([contract,
        "ANTI-LEAK RULE\nVerbalize only the child graph. Do not mention an earlier plan, promise, qualification, claim, prediction, or whether anything was fulfilled, confirmed, correct, changed, or met.",
        "FROZEN CHILD GRAPH\n" + canonical_json(child_graph),
        "OUTPUT JSON SCHEMA\n" + canonical_json(REALIZER_SCHEMA),
        "Return one JSON object only."])


def validate_child_graph_with_semantic_split(raw: Any,
                                             dependency: Mapping[str, Any]) -> dict[str, Any]:
    """Do not let frozen-value drift masquerade as format-retryable schema failure."""
    result = validate_child_graph(raw, dependency)
    mismatch = [reason for reason in result.get("reasons", [])
                if reason.startswith("planned_child_event_mismatch:")]
    if mismatch and len(mismatch) == len(result.get("reasons", [])):
        return {**result, "decision": "outline_child_graph_semantic_mismatch"}
    return result
