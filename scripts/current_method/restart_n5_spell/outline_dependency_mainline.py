"""Outline-conditioned dependency generation and constrained QA contracts."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes
from .dependency_conditioned_child_canary import (
    ANSWER_TYPES, OPERATORS, VAGUE_RESULT_PHRASES, normalize_text,
)


MAINLINE_VERSION = "outline_dependency_constrained_questioner_canary_v1"
PLANNER_PROMPT_VERSION = "spell_outline_dependency_planner_4x_positional_v1"
PLANNER_SCHEMA_VERSION = "outline_dependency_batch_4x_positional_v1"
CHILD_PROMPT_VERSION = "spell_outline_dependency_child_expander_single_v1"
CHILD_SCHEMA_VERSION = "outline_dependency_child_graph_single_v1"
REALIZER_PROMPT_VERSION = "spell_outline_dependency_minimal_realizer_single_v1"
REALIZER_SCHEMA_VERSION = "outline_dependency_child_dialogue_single_v1"
QA_PROMPT_VERSION = "spell_frozen_outline_dependency_qa_writer_single_v1"
QA_SCHEMA_VERSION = "frozen_outline_dependency_qa_single_v1"

PLANNED_FIELDS = ("subject", "predicate", "object", "value", "time")
VALID_FIELDS = (
    "status", "operator", "parent_anchor_ids", "parent_condition",
    "planned_child_stage", "planned_child_event", "joint_result",
    "unknown_from_parent_only", "unknown_from_planned_child_only",
    "forbidden_child_restatements", "natural_relation_explanation",
)
EMPTY_FIELDS = ("status", "reason")


def _sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _operand_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
        "required": list(PLANNED_FIELDS), "properties": {
            field: {"type": "string"} for field in PLANNED_FIELDS}}


PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["dependencies"],
    "properties": {"dependencies": {"type": "array", "minItems": 4,
        "maxItems": 4, "items": {"oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": list(VALID_FIELDS), "properties": {
                 "status": {"const": "valid"}, "operator": {"enum": list(OPERATORS)},
                 "parent_anchor_ids": {"type": "array", "minItems": 1,
                    "maxItems": 4, "uniqueItems": True,
                    "items": {"type": "string", "pattern": "^P[1-8]$"}},
                 "parent_condition": _operand_schema(),
                 "planned_child_stage": {"type": "string", "minLength": 1},
                 "planned_child_event": _operand_schema(),
                 "joint_result": {"type": "object", "additionalProperties": False,
                    "required": ["answer_type", "value", "meaning"], "properties": {
                        "answer_type": {"enum": list(ANSWER_TYPES)},
                        "value": {"type": "string", "minLength": 1},
                        "meaning": {"type": "string", "minLength": 1}}},
                 "unknown_from_parent_only": {"type": "string", "minLength": 1},
                 "unknown_from_planned_child_only": {"type": "string", "minLength": 1},
                 "forbidden_child_restatements": {"type": "array", "minItems": 1,
                    "maxItems": 4, "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1}},
                 "natural_relation_explanation": {"type": "string", "minLength": 1}}},
            {"type": "object", "additionalProperties": False,
             "required": list(EMPTY_FIELDS), "properties": {
                 "status": {"const": "no_valid_dependency"},
                 "reason": {"type": "string", "minLength": 1}}},
        ]}}},
}


def assign_outline_dependency_ids(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"dependencies"}:
        raise ValueError("outline_dependency_root_invalid")
    rows = raw.get("dependencies")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("outline_dependency_slot_count_invalid")
    assigned = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or any(key in row for key in
                                           ("dependency_id", "arc_id")):
            raise ValueError(f"outline_dependency_item_invalid:D{index}")
        assigned.append({"dependency_id": f"D{index}", "arc_id": f"E{index}", **row})
    return {"dependencies": assigned}


def validate_planner_schema(raw: Any) -> dict[str, Any]:
    try:
        assigned = assign_outline_dependency_ids(raw)
    except ValueError as exc:
        return {"decision": "outline_dependency_schema_invalid", "reasons": [str(exc)]}
    errors = []
    for index, row in enumerate(raw["dependencies"], 1):
        label = f"D{index}"
        if row.get("status") == "no_valid_dependency":
            if set(row) != set(EMPTY_FIELDS) or not _nonempty(row.get("reason")):
                errors.append(f"{label}:no_valid_shape")
            continue
        if row.get("status") != "valid" or set(row) != set(VALID_FIELDS):
            errors.append(f"{label}:fields"); continue
        if row.get("operator") not in OPERATORS:
            errors.append(f"{label}:operator")
        anchors = row.get("parent_anchor_ids")
        if (not isinstance(anchors, list) or not 1 <= len(anchors) <= 4 or
                len(set(anchors)) != len(anchors) or any(not isinstance(anchor, str)
                or not re.fullmatch(r"P[1-8]", anchor) for anchor in anchors)):
            errors.append(f"{label}:parent_anchor_ids")
        for field in ("parent_condition", "planned_child_event"):
            value = row.get(field)
            if (not isinstance(value, dict) or set(value) != set(PLANNED_FIELDS) or
                    not _nonempty(value.get("subject")) or
                    not _nonempty(value.get("predicate"))):
                errors.append(f"{label}:{field}")
        joint = row.get("joint_result")
        if (not isinstance(joint, dict) or set(joint) != {
                "answer_type", "value", "meaning"} or
                joint.get("answer_type") not in ANSWER_TYPES or
                not _nonempty(joint.get("value")) or not _nonempty(joint.get("meaning"))):
            errors.append(f"{label}:joint_result")
        for field in ("planned_child_stage", "unknown_from_parent_only",
                      "unknown_from_planned_child_only", "natural_relation_explanation"):
            if not _nonempty(row.get(field)):
                errors.append(f"{label}:{field}")
        forbidden = row.get("forbidden_child_restatements")
        if (not isinstance(forbidden, list) or not 1 <= len(forbidden) <= 4 or
                len(set(forbidden)) != len(forbidden) or
                any(not _nonempty(value) for value in forbidden)):
            errors.append(f"{label}:forbidden_child_restatements")
    return {"decision": "outline_dependency_batch_accept" if not errors else
            "outline_dependency_schema_invalid", "reasons": errors,
            **({"assigned_batch": assigned} if not errors else {})}


def normalize_outline_dependencies(raw: Any, outline: Mapping[str, Any],
                                   parent_turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validation = validate_planner_schema(raw)
    if validation["decision"] != "outline_dependency_batch_accept":
        raise ValueError("cannot_normalize_invalid_outline_dependency_batch")
    assigned = validation["assigned_batch"]
    arcs = {arc["event_id"]: arc for arc in outline["planned_event_arcs"]}
    outline_anchors = {anchor["anchor_id"]: anchor for anchor in outline["parent_anchors"]}
    turns = {turn["turn_id"]: turn for turn in parent_turns}
    seen: set[tuple[str, ...]] = set()
    output = []
    for row in assigned["dependencies"]:
        item = dict(row)
        if row["status"] == "no_valid_dependency":
            item["local_validation"] = "accept_empty"; output.append(item); continue
        errors = []
        arc = arcs.get(row["arc_id"])
        allowed = set((arc or {}).get("parent_anchor_ids") or [])
        if not set(row["parent_anchor_ids"]) <= allowed:
            errors.append("parent_anchor_outside_arc")
        selected = [outline_anchors[anchor] for anchor in row["parent_anchor_ids"]
                    if anchor in outline_anchors]
        if len(selected) != len(row["parent_anchor_ids"]):
            errors.append("parent_anchor_missing")
        if not arc or row["planned_child_stage"] not in arc.get("stages", []):
            errors.append("planned_stage_outside_arc")
        subject = normalize_text(row["parent_condition"]["subject"])
        anchor_text = " ".join(str(anchor.get("speaker") or "") + " " +
                               str(anchor.get("text") or "") for anchor in selected)
        if subject and subject not in normalize_text(anchor_text):
            errors.append("parent_condition_subject_unsupported")
        planned = row["planned_child_event"]
        if normalize_text(planned["subject"]) != subject:
            errors.append("planned_child_subject_mismatch")
        if all(normalize_text(planned[field]) == normalize_text(
                row["parent_condition"][field]) for field in PLANNED_FIELDS):
            errors.append("parent_restatement_plan")
        joint_text = normalize_text(row["joint_result"]["value"] + " " +
                                    row["joint_result"]["meaning"])
        if any(phrase in joint_text for phrase in VAGUE_RESULT_PHRASES):
            errors.append("vague_joint_result")
        support = tuple(sorted(row["parent_anchor_ids"] + [row["planned_child_stage"]]))
        if support in seen:
            errors.append("duplicate_dependency_support")
        resolved = []
        for anchor in selected:
            turn = turns.get(anchor["turn_id"])
            if not turn or turn.get("speaker") != anchor.get("speaker") or \
                    turn.get("text") != anchor.get("text"):
                errors.append("parent_anchor_text_binding_invalid")
            else:
                resolved.append({**turn, "outline_anchor_id": anchor["anchor_id"]})
        if not errors:
            seen.add(support)
        item.update({"resolved_parent_anchors": selected,
            "resolved_parent_turns": resolved, "canonical_support": list(support),
            "local_validation": "accept" if not errors else "reject",
            "local_validation_errors": sorted(set(errors))})
        output.append(item)
    return {"schema_version": "normalized_outline_dependencies_v1",
        "dependencies": output,
        "local_valid_count": sum(row.get("local_validation") == "accept" for row in output),
        "empty_count": sum(row.get("status") == "no_valid_dependency" for row in output)}


def build_planner_prompt(outline: Mapping[str, Any], contract_text: str) -> str:
    clean = json.loads(json.dumps({key: outline[key] for key in (
        "outline_id", "outline_role", "title", "expansion_objective", "time_window",
        "allowed_new_characters", "allowed_locations", "parent_anchors",
        "planned_event_arcs", "scene_plan", "forbidden_conflicts")}))
    for anchor in clean["parent_anchors"]:
        anchor.pop("turn_id", None)
    return "\n\n".join([f"PROMPT_VERSION: {PLANNER_PROMPT_VERSION}", contract_text,
        "FROZEN EXPANSION OUTLINE\n" + canonical_json(clean),
        "POSITIONAL OUTPUT CONTRACT\nArray positions 1-4 correspond to E1-E4. "
        "Do not output dependency_id or arc_id; the local program assigns both.",
        "OUTPUT JSON SCHEMA\n" + canonical_json(PLANNER_SCHEMA),
        "Return exactly four array items in one JSON object."])


CHILD_EVENT_FIELDS = ("event_id", *PLANNED_FIELDS, "status")
CHILD_FIELDS = ("dependency_id", "arc_id", "stage", "child_graph_id",
                "child_events", "realizes_planned_child_event_id")
CHILD_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": list(CHILD_FIELDS), "properties": {
        "dependency_id": {"type": "string"}, "arc_id": {"type": "string"},
        "stage": {"type": "string"}, "child_graph_id": {"type": "string"},
        "child_events": {"type": "array", "minItems": 1, "maxItems": 3,
            "items": {"type": "object", "additionalProperties": False,
                "required": list(CHILD_EVENT_FIELDS), "properties": {
                    field: {"type": "string"} for field in CHILD_EVENT_FIELDS}}},
        "realizes_planned_child_event_id": {"type": "string"}}}


def validate_child_graph(raw: Any, dependency: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if not isinstance(raw, dict) or set(raw) != set(CHILD_FIELDS):
        return {"decision": "outline_child_graph_schema_invalid", "reasons": ["fields"]}
    for field in ("dependency_id", "arc_id", "stage"):
        expected = dependency["planned_child_stage"] if field == "stage" else dependency[field]
        if raw.get(field) != expected:
            errors.append(f"{field}_drift")
    events = raw.get("child_events")
    if not isinstance(events, list) or not 1 <= len(events) <= 3:
        errors.append("child_events"); events = []
    ids = []
    for event in events:
        if (not isinstance(event, dict) or set(event) != set(CHILD_EVENT_FIELDS) or
                any(not isinstance(event.get(field), str) for field in CHILD_EVENT_FIELDS) or
                not _nonempty(event.get("event_id"))):
            errors.append("child_event_schema"); continue
        ids.append(event["event_id"])
    if len(ids) != len(set(ids)):
        errors.append("duplicate_child_event_id")
    realized = raw.get("realizes_planned_child_event_id")
    if realized not in ids:
        errors.append("planned_child_event_missing")
    else:
        actual = next(event for event in events if event["event_id"] == realized)
        for field in PLANNED_FIELDS:
            if normalize_text(actual[field]) != normalize_text(
                    dependency["planned_child_event"][field]):
                errors.append(f"planned_child_event_mismatch:{field}")
    return {"decision": "outline_child_graph_accept" if not errors else
            "outline_child_graph_schema_invalid", "reasons": sorted(set(errors)),
            **({"child_graph": raw} if not errors else {})}


def build_child_prompt(outline: Mapping[str, Any], dependency: Mapping[str, Any],
                       contract_text: str) -> str:
    arc = next(arc for arc in outline["planned_event_arcs"]
               if arc["event_id"] == dependency["arc_id"])
    frozen = {key: dependency[key] for key in (
        "dependency_id", "arc_id", "operator", "parent_condition",
        "planned_child_stage", "planned_child_event", "joint_result",
        "forbidden_child_restatements")}
    return "\n\n".join([f"PROMPT_VERSION: {CHILD_PROMPT_VERSION}", contract_text,
        "ASSIGNED OUTLINE ARC\n" + canonical_json(arc),
        "FROZEN DEPENDENCY\n" + canonical_json(frozen),
        "OUTPUT JSON SCHEMA\n" + canonical_json(CHILD_SCHEMA),
        "Return one child graph object only."])


REALIZER_FIELDS = ("dependency_id", "session_id", "dialogue_turns",
                   "verbalized_child_event_ids", "event_realization_turn_indices")
REALIZER_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": list(REALIZER_FIELDS), "properties": {
        "dependency_id": {"type": "string"}, "session_id": {"type": "string"},
        "dialogue_turns": {"type": "array", "minItems": 4, "maxItems": 8,
            "items": {"type": "object", "additionalProperties": False,
                "required": ["speaker", "text"], "properties": {
                    "speaker": {"type": "string"}, "text": {"type": "string"}}}},
        "verbalized_child_event_ids": {"type": "array", "uniqueItems": True,
            "items": {"type": "string"}},
        "event_realization_turn_indices": {"type": "object"}}}


def build_realizer_prompt(dependency: Mapping[str, Any], child_graph: Mapping[str, Any],
                          contract_text: str) -> str:
    return "\n\n".join([f"PROMPT_VERSION: {REALIZER_PROMPT_VERSION}", contract_text,
        "FROZEN DEPENDENCY ANTI-LEAK CONTRACT\n" + canonical_json({key:
            dependency[key] for key in ("dependency_id", "parent_condition",
            "planned_child_event", "joint_result", "forbidden_child_restatements")}),
        "FROZEN CHILD GRAPH\n" + canonical_json(child_graph),
        "OUTPUT JSON SCHEMA\n" + canonical_json(REALIZER_SCHEMA),
        "Return one JSON object only."])


def validate_realizer(raw: Any, dependency: Mapping[str, Any],
                      child_graph: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if not isinstance(raw, dict) or set(raw) != set(REALIZER_FIELDS):
        return {"decision": "outline_realizer_schema_invalid", "reasons": ["fields"]}
    if raw.get("dependency_id") != dependency["dependency_id"]:
        errors.append("dependency_id_drift")
    turns = raw.get("dialogue_turns")
    if not isinstance(turns, list) or not 4 <= len(turns) <= 8 or any(
            not isinstance(turn, dict) or set(turn) != {"speaker", "text"} or
            not _nonempty(turn.get("speaker")) or not _nonempty(turn.get("text"))
            for turn in (turns or [])):
        errors.append("dialogue_turns"); turns = []
    realized = child_graph["realizes_planned_child_event_id"]
    if realized not in (raw.get("verbalized_child_event_ids") or []):
        errors.append("unverbalized_child_event")
    indices = (raw.get("event_realization_turn_indices") or {}).get(realized)
    if (not isinstance(indices, list) or not indices or any(type(i) is not int or
            i < 0 or i >= len(turns) for i in indices)):
        errors.append("event_realization_indices_invalid")
    return {"decision": "outline_realizer_accept" if not errors else
            "outline_realizer_schema_invalid", "reasons": sorted(set(errors)),
            **({"realizer": raw} if not errors else {})}


def apply_anti_leak_gate(realizer: Mapping[str, Any], dependency: Mapping[str, Any],
                         child_graph: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    turns = realizer["dialogue_turns"]
    texts = [normalize_text(turn["text"]) for turn in turns]
    combined = " ".join(texts)
    for phrase in dependency["forbidden_child_restatements"]:
        normalized = normalize_text(phrase)
        if len(normalized) >= 12 and normalized in combined:
            errors.append("parent_fact_restatement")
    for value in dependency["joint_result"].values():
        normalized = normalize_text(value)
        if len(normalized) >= 12 and normalized in combined:
            errors.append("joint_answer_leakage")
    realized = child_graph["realizes_planned_child_event_id"]
    indices = realizer["event_realization_turn_indices"][realized]
    aliased = [{"alias": f"C{index + 1:03d}",
        "turn_id": f"{dependency['dependency_id']}:C{index + 1:03d}",
        "session_id": realizer["session_id"], "speaker": turn["speaker"],
        "text": turn["text"], "source_partition": "child"}
        for index, turn in enumerate(turns)]
    evidence = [aliased[index] for index in indices]
    return {"decision": "outline_realization_local_gate_pass" if not errors else
            "outline_realization_local_gate_reject", "reasons": sorted(set(errors)),
            "child_dialogue_with_aliases": aliased,
            "planned_event_evidence_turns": evidence}


QA_FIELDS = ("dependency_id", "question", "canonical_answer", "answer_type",
             "relation_preserved")
QA_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": list(QA_FIELDS), "properties": {
        "dependency_id": {"type": "string"},
        "question": {"type": "string", "minLength": 1},
        "canonical_answer": {"type": "string", "minLength": 1},
        "answer_type": {"enum": list(ANSWER_TYPES)},
        "relation_preserved": {"const": True}}}


def build_qa_prompt(dependency: Mapping[str, Any],
                    parent_turns: Sequence[Mapping[str, Any]],
                    child_turns: Sequence[Mapping[str, Any]], contract_text: str) -> str:
    frozen = {"dependency_id": dependency["dependency_id"],
        "operator": dependency["operator"], "joint_result": dependency["joint_result"],
        "fixed_parent_evidence": [{key: turn[key] for key in ("alias", "speaker", "text")}
                                  for turn in parent_turns],
        "fixed_child_evidence": [{key: turn[key] for key in ("alias", "speaker", "text")}
                                 for turn in child_turns]}
    return "\n\n".join([f"PROMPT_VERSION: {QA_PROMPT_VERSION}", contract_text,
        "FROZEN MINIMAL EVIDENCE AND ANSWER\n" + canonical_json(frozen),
        "OUTPUT JSON SCHEMA\n" + canonical_json(QA_SCHEMA),
        "Return one JSON object only."])


def validate_qa(raw: Any, dependency: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if not isinstance(raw, dict) or set(raw) != set(QA_FIELDS):
        return {"decision": "outline_qa_schema_invalid", "reasons": ["fields"]}
    if raw.get("dependency_id") != dependency["dependency_id"]:
        errors.append("qa_dependency_drift")
    if normalize_text(raw.get("canonical_answer")) != normalize_text(
            dependency["joint_result"]["value"]):
        errors.append("qa_joint_result_drift")
    if raw.get("answer_type") != dependency["joint_result"]["answer_type"]:
        errors.append("qa_joint_result_drift")
    if raw.get("relation_preserved") is not True or not _nonempty(raw.get("question")):
        errors.append("qa_schema_invalid")
    return {"decision": "outline_qa_accept" if not errors else
            "outline_qa_schema_invalid", "reasons": sorted(set(errors)),
            **({"qa": raw} if not errors else {})}


def contract_shas() -> dict[str, str]:
    return {"planner_schema_sha256": _sha(PLANNER_SCHEMA),
        "child_schema_sha256": _sha(CHILD_SCHEMA),
        "realizer_schema_sha256": _sha(REALIZER_SCHEMA),
        "qa_schema_sha256": _sha(QA_SCHEMA),
        "contract_source_sha256": sha256_bytes((inspect.getsource(
            normalize_outline_dependencies) + inspect.getsource(
            apply_anti_leak_gate) + inspect.getsource(validate_qa)).encode())}
