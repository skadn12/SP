"""Contracts and deterministic gates for dependency-conditioned child expansion."""

from __future__ import annotations

import inspect
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes
from .input_adapter import dialogue_content_hash


CANARY_VERSION = "dependency_conditioned_child_expansion_canary_v1"
PARENT_GRAPH_PROJECTION_VERSION = "authoritative_parent_graph_projection_v1"
DEPENDENCY_PROMPT_VERSION = "spell_dependency_planner_4x_alias_event_only_v1"
DEPENDENCY_SCHEMA_VERSION = "dependency_planner_batch_4x_v1"
POSITIONAL_DEPENDENCY_PROMPT_VERSION = (
    "spell_dependency_planner_4x_alias_event_positional_id_v1"
)
POSITIONAL_DEPENDENCY_SCHEMA_VERSION = "dependency_planner_batch_4x_positional_id_v1"
CHILD_EXPANDER_PROMPT_VERSION = "spell_dependency_conditioned_child_expander_single_v1"
CHILD_GRAPH_SCHEMA_VERSION = "dependency_conditioned_child_graph_single_v1"
REALIZER_PROMPT_VERSION = "spell_dependency_conditioned_minimal_realizer_single_v1"
REALIZER_SCHEMA_VERSION = "dependency_conditioned_child_dialogue_single_v1"
QA_PROMPT_VERSION = "spell_frozen_dependency_qa_writer_single_v1"
QA_SCHEMA_VERSION = "frozen_dependency_qa_single_v1"
DEPENDENCY_AUDIT_VERSION = "offline_dependency_contract_audit_v1"
REALIZATION_AUDIT_VERSION = "offline_dependency_realization_audit_v1"
FINAL_QA_AUDIT_VERSION = "offline_dependency_joint_qa_audit_v1"

OPERATORS = (
    "commitment_outcome", "prediction_outcome", "qualification_requirement",
    "claim_verification", "prior_state_later_decision",
)
ANSWER_TYPES = ("boolean", "entity", "state", "number", "choice")
VAGUE_RESULT_PHRASES = (
    "aligns with", "align with", "is related to", "provides a foundation",
    "provide a foundation", "supports the general goal", "support the general goal",
    "both emphasize", "is relevant to", "are relevant to",
)
LEAKAGE_PHRASES = (
    "as you promised earlier", "this fulfills your earlier commitment",
    "this fulfils your earlier commitment", "because you previously",
    "because you said earlier", "you said you would", "as you predicted earlier",
    "this confirms your earlier prediction", "this proves your earlier claim",
)
EVENT_FIELDS = ("event_id", "subject", "predicate", "object", "value", "time", "status")
OPERAND_FIELDS = ("subject", "predicate", "object", "value", "time")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", " ", text).casefold()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def project_authoritative_parent(parent_row: Mapping[str, Any]) -> dict[str, Any]:
    dialogue = parent_row.get("dialogue") or {}
    turns = dialogue.get("turns") or []
    graph = parent_row.get("semantic_parent_graph") or {}
    events = graph.get("events") or []
    if len(turns) != 237 or dialogue_content_hash(turns) != dialogue.get("dialogue_hash"):
        raise ValueError("authoritative_parent_dialogue_invalid")
    if not events:
        raise ValueError("authoritative_parent_graph_empty")
    projected_events = [{key: event.get(key) for key in (
        "event_id", "entity", "slot", "value", "status", "time_index",
        "evidence_turn_ids")}
        for event in events]
    if any(not _nonempty(event.get("event_id")) or
           not isinstance(event.get("evidence_turn_ids"), list)
           for event in projected_events):
        raise ValueError("authoritative_parent_graph_invalid")
    aliased = []
    for index, turn in enumerate(turns, 1):
        turn_id = str(turn.get("turn_id") or "")
        if not turn_id:
            raise ValueError("parent_turn_id_missing")
        aliased.append({
            "alias": f"P{index:03d}", "turn_id": turn_id,
            "session_id": turn_id.split(":", 1)[0],
            "speaker": str(turn.get("speaker") or ""),
            "text": str(turn.get("text") or ""), "source_partition": "parent",
        })
    projection = {
        "schema_version": PARENT_GRAPH_PROJECTION_VERSION,
        "graph_id": graph.get("graph_id"),
        "parent_graph_id": graph.get("parent_graph_id"),
        "graph_role": graph.get("graph_role"), "events": projected_events,
    }
    return {
        "parent_graph": projection, "parent_dialogue": aliased,
        "parent_count": len(aliased), "child_count_in_generation_input": 0,
        "parent_graph_source_sha256": _sha(graph),
        "parent_graph_projection_sha256": _sha(projection),
        "parent_dialogue_sha256": _sha(aliased),
        "source_dialogue_content_sha256": dialogue["dialogue_hash"],
    }


def _operand_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
        "required": list(OPERAND_FIELDS), "properties": {
            field: {"type": "string"} for field in OPERAND_FIELDS}}


DEPENDENCY_VALID_FIELDS = (
    "dependency_id", "status", "operator", "parent_event_ids",
    "parent_turn_aliases", "parent_operand", "planned_child_event",
    "joint_result", "unknown_from_parent_only", "unknown_from_planned_child_only",
    "forbidden_child_restatements", "natural_relation_explanation",
)
DEPENDENCY_EMPTY_FIELDS = ("dependency_id", "status", "reason")

DEPENDENCY_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["dependencies"],
    "properties": {"dependencies": {"type": "array", "minItems": 4, "maxItems": 4,
        "items": {"oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": list(DEPENDENCY_VALID_FIELDS), "properties": {
                 "dependency_id": {"type": "string"}, "status": {"const": "valid"},
                 "operator": {"enum": list(OPERATORS)},
                 "parent_event_ids": {"type": "array", "minItems": 1,
                    "maxItems": 2, "uniqueItems": True, "items": {"type": "string"}},
                 "parent_turn_aliases": {"type": "array", "minItems": 1,
                    "maxItems": 4, "uniqueItems": True, "items": {"type": "string"}},
                 "parent_operand": _operand_schema(),
                 "planned_child_event": _operand_schema(),
                 "joint_result": {"type": "object", "additionalProperties": False,
                    "required": ["answer_type", "value", "meaning"], "properties": {
                        "answer_type": {"enum": list(ANSWER_TYPES)},
                        "value": {"type": "string", "minLength": 1},
                        "meaning": {"type": "string", "minLength": 1}}},
                 "unknown_from_parent_only": {"type": "string", "minLength": 1},
                 "unknown_from_planned_child_only": {"type": "string", "minLength": 1},
                 "forbidden_child_restatements": {"type": "array", "minItems": 1,
                    "maxItems": 4, "uniqueItems": True, "items": {
                        "type": "string", "minLength": 1}},
                 "natural_relation_explanation": {"type": "string", "minLength": 1}}},
            {"type": "object", "additionalProperties": False,
             "required": list(DEPENDENCY_EMPTY_FIELDS), "properties": {
                 "dependency_id": {"type": "string"},
                 "status": {"const": "no_valid_dependency"},
                 "reason": {"type": "string", "minLength": 1}}},
        ]}}},
}


def _without_dependency_id(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the positional Stage-0 schema without changing semantic fields."""
    output = json.loads(json.dumps(schema))
    variants = output["properties"]["dependencies"]["items"]["oneOf"]
    for variant in variants:
        variant["required"].remove("dependency_id")
        del variant["properties"]["dependency_id"]
    return output


POSITIONAL_DEPENDENCY_BATCH_SCHEMA = _without_dependency_id(DEPENDENCY_BATCH_SCHEMA)


def assign_dependency_ids_by_position(raw: Any) -> dict[str, Any]:
    """Assign D1-D4 locally; never repair or otherwise change model fields."""
    if not isinstance(raw, dict) or set(raw) != {"dependencies"}:
        raise ValueError("positional_dependency_root_invalid")
    rows = raw.get("dependencies")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("positional_dependency_slot_count_invalid")
    assigned = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or "dependency_id" in row:
            raise ValueError(f"positional_dependency_item_invalid:D{index}")
        assigned.append({"dependency_id": f"D{index}", **row})
    return {"dependencies": assigned}


def validate_positional_dependency_batch_schema(raw: Any) -> dict[str, Any]:
    """Validate model output without IDs, then validate the locally assigned form."""
    try:
        assigned = assign_dependency_ids_by_position(raw)
    except ValueError as exc:
        return {"decision": "dependency_schema_invalid", "reasons": [str(exc)]}
    validation = validate_dependency_batch_schema(assigned)
    if validation["decision"] != "dependency_batch_accept":
        return validation
    return {"decision": "dependency_batch_accept", "reasons": [],
            "raw_batch_without_ids": raw, "assigned_batch": assigned,
            "assignment_rule": "array_position_1_to_4_maps_to_D1_to_D4"}


def validate_dependency_batch_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"dependencies"}:
        return {"decision": "dependency_schema_invalid", "reasons": ["root"]}
    rows = raw.get("dependencies")
    if not isinstance(rows, list) or len(rows) != 4:
        return {"decision": "dependency_schema_invalid", "reasons": ["slot_count"]}
    errors = []
    for index, row in enumerate(rows, 1):
        did = f"D{index}"
        if not isinstance(row, dict) or row.get("dependency_id") != did:
            errors.append(f"{did}:dependency_id"); continue
        if row.get("status") == "no_valid_dependency":
            if set(row) != set(DEPENDENCY_EMPTY_FIELDS) or not _nonempty(row.get("reason")):
                errors.append(f"{did}:no_valid_shape")
            continue
        if row.get("status") != "valid" or set(row) != set(DEPENDENCY_VALID_FIELDS):
            errors.append(f"{did}:fields"); continue
        if row.get("operator") not in OPERATORS:
            errors.append(f"{did}:operator")
        for field, maximum in (("parent_event_ids", 2), ("parent_turn_aliases", 4),
                               ("forbidden_child_restatements", 4)):
            values = row.get(field)
            if (not isinstance(values, list) or not 1 <= len(values) <= maximum or
                    any(not _nonempty(value) for value in values) or
                    len(set(values)) != len(values)):
                errors.append(f"{did}:{field}")
        for field in ("parent_operand", "planned_child_event"):
            value = row.get(field)
            if (not isinstance(value, dict) or set(value) != set(OPERAND_FIELDS) or
                    not _nonempty(value.get("subject")) or
                    not _nonempty(value.get("predicate"))):
                errors.append(f"{did}:{field}")
        joint = row.get("joint_result")
        if (not isinstance(joint, dict) or set(joint) != {"answer_type", "value", "meaning"}
                or joint.get("answer_type") not in ANSWER_TYPES
                or not _nonempty(joint.get("value")) or not _nonempty(joint.get("meaning"))):
            errors.append(f"{did}:joint_result")
        for field in ("unknown_from_parent_only", "unknown_from_planned_child_only",
                      "natural_relation_explanation"):
            if not _nonempty(row.get(field)):
                errors.append(f"{did}:{field}")
    return {"decision": "dependency_batch_accept" if not errors else
            "dependency_schema_invalid", "reasons": errors,
            **({"raw_batch": raw} if not errors else {})}


def _supported_object(value: str, events: Sequence[Mapping[str, Any]],
                      turns: Sequence[Mapping[str, Any]]) -> bool:
    if not normalize_text(value):
        return True
    needle = normalize_text(value)
    haystacks = [normalize_text(event.get("value")) for event in events]
    haystacks.extend(normalize_text(turn.get("text")) for turn in turns)
    return any(needle in haystack for haystack in haystacks)


def normalize_dependency_batch(raw: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    if validate_dependency_batch_schema(raw)["decision"] != "dependency_batch_accept":
        raise ValueError("cannot_normalize_invalid_dependency_batch")
    aliases = {turn["alias"]: turn for turn in parent["parent_dialogue"]}
    events = {event["event_id"]: event for event in parent["parent_graph"]["events"]}
    seen: set[tuple[str, ...]] = set()
    output = []
    for row in raw["dependencies"]:
        item = dict(row)
        if row["status"] == "no_valid_dependency":
            item["local_validation"] = "accept_empty"; output.append(item); continue
        errors: list[str] = []
        selected_events = [events[eid] for eid in row["parent_event_ids"] if eid in events]
        missing_events = [eid for eid in row["parent_event_ids"] if eid not in events]
        selected_turns = [aliases[alias] for alias in row["parent_turn_aliases"]
                          if alias in aliases]
        missing_aliases = [alias for alias in row["parent_turn_aliases"] if alias not in aliases]
        errors.extend(f"parent_event_missing:{eid}" for eid in missing_events)
        errors.extend(f"parent_alias_missing:{alias}" for alias in missing_aliases)
        evidence_ids = {tid for event in selected_events
                        for tid in event.get("evidence_turn_ids") or []}
        if selected_turns and any(turn["turn_id"] not in evidence_ids for turn in selected_turns):
            errors.append("event_alias_fact_mismatch")
        operand = row["parent_operand"]
        subjects = {str(event.get("entity") or "") for event in selected_events}
        if selected_events and operand["subject"] not in subjects:
            errors.append("subject_mismatch")
        if selected_events and operand["predicate"] not in {
                str(event.get("slot") or "") for event in selected_events}:
            errors.append("unsupported_parent_operand")
        if selected_events and operand["value"] and operand["value"] not in {
                str(event.get("value") or "") for event in selected_events}:
            errors.append("fabricated_value")
        if not _supported_object(operand["object"], selected_events, selected_turns):
            errors.append("unsupported_parent_operand")
        valid_times = {"", *[str(event.get("time_index") if event.get("time_index")
                                  is not None else "") for event in selected_events]}
        if operand["time"] not in valid_times:
            errors.append("fabricated_value")
        planned = row["planned_child_event"]
        if all(normalize_text(planned[key]) == normalize_text(operand[key])
               for key in OPERAND_FIELDS):
            errors.append("parent_restatement_plan")
        joint_text = normalize_text(row["joint_result"]["meaning"] + " " +
                                    row["joint_result"]["value"])
        if any(phrase in joint_text for phrase in VAGUE_RESULT_PHRASES):
            errors.append("vague_joint_result")
        support = tuple(sorted(row["parent_event_ids"] + row["parent_turn_aliases"]))
        if support in seen:
            errors.append("duplicate_dependency_support")
        if not errors:
            seen.add(support)
        item.update({
            "resolved_parent_events": selected_events,
            "resolved_parent_turns": selected_turns,
            "canonical_parent_support": list(support),
            "local_validation": "accept" if not errors else "reject",
            "local_validation_errors": sorted(set(errors)),
        })
        output.append(item)
    return {"schema_version": "normalized_dependency_contracts_v1",
        "dependencies": output,
        "local_valid_count": sum(row.get("local_validation") == "accept" for row in output),
        "empty_count": sum(row.get("status") == "no_valid_dependency" for row in output)}


def build_dependency_planner_prompt(parent: Mapping[str, Any], contract_text: str) -> str:
    if parent["parent_count"] != 237 or parent["child_count_in_generation_input"] != 0:
        raise ValueError("parent_only_generation_input_required")
    dialogue = "\n".join(
        f"[{turn['alias']}] original_turn_id={turn['turn_id']} speaker={turn['speaker']} | {turn['text']}"
        for turn in parent["parent_dialogue"])
    return "\n\n".join([
        f"PROMPT_VERSION: {DEPENDENCY_PROMPT_VERSION}", contract_text,
        "AUTHORITATIVE PARENT GRAPH/STATE (canonical projection)\n" +
            canonical_json(parent["parent_graph"]),
        "AUTHORITATIVE 237-TURN PARENT DIALOGUE\n" + dialogue,
        "OUTPUT JSON SCHEMA\n" + canonical_json(DEPENDENCY_BATCH_SCHEMA),
        "Return exactly D1-D4 as one JSON object. No child dialogue has been supplied.",
    ])


def build_positional_dependency_planner_prompt(
        parent: Mapping[str, Any], contract_text: str) -> str:
    """Render the same semantic task with program-owned dependency identifiers."""
    if parent["parent_count"] != 237 or parent["child_count_in_generation_input"] != 0:
        raise ValueError("parent_only_generation_input_required")
    dialogue = "\n".join(
        f"[{turn['alias']}] original_turn_id={turn['turn_id']} speaker={turn['speaker']} | {turn['text']}"
        for turn in parent["parent_dialogue"])
    return "\n\n".join([
        f"PROMPT_VERSION: {POSITIONAL_DEPENDENCY_PROMPT_VERSION}", contract_text,
        "AUTHORITATIVE PARENT GRAPH/STATE (canonical projection)\n" +
            canonical_json(parent["parent_graph"]),
        "AUTHORITATIVE 237-TURN PARENT DIALOGUE\n" + dialogue,
        "OUTPUT JSON SCHEMA\n" + canonical_json(POSITIONAL_DEPENDENCY_BATCH_SCHEMA),
        "Return exactly four dependency array items as one JSON object. Do not output "
        "dependency_id. The local program assigns D1-D4 from array positions 1-4. "
        "No child dialogue has been supplied.",
    ])


CHILD_GRAPH_FIELDS = ("dependency_id", "child_graph_id", "entity_bindings",
                      "child_events", "time_order_constraints",
                      "realizes_planned_child_event_id")
CHILD_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": list(CHILD_GRAPH_FIELDS),
    "properties": {
        "dependency_id": {"type": "string"}, "child_graph_id": {"type": "string"},
        "entity_bindings": {"type": "array", "items": {"type": "object"}},
        "child_events": {"type": "array", "minItems": 1, "maxItems": 4,
            "items": {"type": "object", "additionalProperties": False,
                "required": list(EVENT_FIELDS), "properties": {
                    field: {"type": "string"} for field in EVENT_FIELDS}}},
        "time_order_constraints": {"type": "array", "items": {"type": "string"}},
        "realizes_planned_child_event_id": {"type": "string"},
    }}


def validate_child_graph(raw: Any, dependency: Mapping[str, Any],
                         parent_event_ids: set[str]) -> dict[str, Any]:
    errors = []
    if not isinstance(raw, dict) or set(raw) != set(CHILD_GRAPH_FIELDS):
        return {"decision": "child_graph_schema_invalid", "reasons": ["fields"]}
    if raw.get("dependency_id") != dependency["dependency_id"] or not _nonempty(
            raw.get("child_graph_id")):
        errors.append("dependency_or_graph_id")
    events = raw.get("child_events")
    if not isinstance(events, list) or not 1 <= len(events) <= 4:
        errors.append("child_events")
        events = []
    ids = []
    for event in events:
        if (not isinstance(event, dict) or set(event) != set(EVENT_FIELDS) or
                any(not isinstance(event.get(field), str) for field in EVENT_FIELDS) or
                not _nonempty(event.get("event_id")) or not _nonempty(event.get("subject")) or
                not _nonempty(event.get("predicate"))):
            errors.append("child_event_schema"); continue
        ids.append(event["event_id"])
    if len(ids) != len(set(ids)) or set(ids) & parent_event_ids:
        errors.append("child_event_id_invalid")
    realized = raw.get("realizes_planned_child_event_id")
    if realized not in ids:
        errors.append("planned_child_event_missing")
    else:
        actual = next(event for event in events if event["event_id"] == realized)
        planned = dependency["planned_child_event"]
        for field in OPERAND_FIELDS:
            if normalize_text(actual[field]) != normalize_text(planned[field]):
                errors.append(f"planned_child_event_mismatch:{field}")
    if not isinstance(raw.get("entity_bindings"), list) or not isinstance(
            raw.get("time_order_constraints"), list):
        errors.append("child_graph_metadata")
    return {"decision": "child_graph_accept" if not errors else
            "child_graph_schema_invalid", "reasons": sorted(set(errors)),
            **({"child_graph": raw} if not errors else {})}


def build_child_expander_prompt(parent_graph: Mapping[str, Any], dependency: Mapping[str, Any],
                                contract_text: str) -> str:
    frozen = {key: dependency[key] for key in (
        "dependency_id", "operator", "parent_event_ids", "parent_operand",
        "planned_child_event", "joint_result", "forbidden_child_restatements")}
    return "\n\n".join([
        f"PROMPT_VERSION: {CHILD_EXPANDER_PROMPT_VERSION}", contract_text,
        "AUTHORITATIVE PARENT GRAPH/STATE\n" + canonical_json(parent_graph),
        "FROZEN DEPENDENCY CONTRACT (immutable)\n" + canonical_json(frozen),
        "OUTPUT JSON SCHEMA\n" + canonical_json(CHILD_GRAPH_SCHEMA),
        "Return one child graph object only. Do not output dialogue, QA, or the joint result as narration.",
    ])


REALIZER_FIELDS = ("dependency_id", "session_id", "dialogue_turns",
                   "verbalized_child_event_ids", "event_realization_turn_indices")
REALIZER_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": list(REALIZER_FIELDS),
    "properties": {
        "dependency_id": {"type": "string"}, "session_id": {"type": "string"},
        "dialogue_turns": {"type": "array", "minItems": 4, "maxItems": 12,
            "items": {"type": "object", "additionalProperties": False,
                "required": ["speaker", "text"], "properties": {
                    "speaker": {"type": "string"}, "text": {"type": "string"}}}},
        "verbalized_child_event_ids": {"type": "array", "uniqueItems": True,
            "items": {"type": "string"}},
        "event_realization_turn_indices": {"type": "object"},
    }}


def validate_realizer_output(raw: Any, dependency: Mapping[str, Any],
                             child_graph: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if not isinstance(raw, dict) or set(raw) != set(REALIZER_FIELDS):
        return {"decision": "realizer_schema_invalid", "reasons": ["fields"]}
    if raw.get("dependency_id") != dependency["dependency_id"] or not _nonempty(
            raw.get("session_id")):
        errors.append("identity")
    turns = raw.get("dialogue_turns")
    if not isinstance(turns, list) or not 4 <= len(turns) <= 12:
        errors.append("dialogue_turn_count"); turns = []
    elif any(not isinstance(turn, dict) or set(turn) != {"speaker", "text"} or
             not _nonempty(turn.get("speaker")) or not _nonempty(turn.get("text"))
             for turn in turns):
        errors.append("dialogue_turn_schema")
    event_ids = {event["event_id"] for event in child_graph["child_events"]}
    verbalized = raw.get("verbalized_child_event_ids")
    indices = raw.get("event_realization_turn_indices")
    if (not isinstance(verbalized, list) or not verbalized or
            not set(verbalized) <= event_ids or
            child_graph["realizes_planned_child_event_id"] not in verbalized):
        errors.append("unverbalized_child_event")
    if not isinstance(indices, dict):
        errors.append("event_realization_indices")
    else:
        for event_id in verbalized if isinstance(verbalized, list) else []:
            values = indices.get(event_id)
            if (not isinstance(values, list) or not values or
                    any(not isinstance(i, int) or not 0 <= i < len(turns) for i in values)):
                errors.append("event_realization_indices")
    return {"decision": "realizer_accept" if not errors else
            "realizer_schema_invalid", "reasons": sorted(set(errors)),
            **({"realizer_output": raw} if not errors else {})}


def build_realizer_prompt(dependency: Mapping[str, Any], child_graph: Mapping[str, Any]) -> str:
    return "\n\n".join([
        f"PROMPT_VERSION: {REALIZER_PROMPT_VERSION}",
        "You are the formal Realizer. Verbalize the frozen child graph as a natural 4-12 turn continuation. Do not change or add key facts.",
        "Do not restate the parent operand, mention an earlier promise/prediction/claim, explain the dependency, state the joint result, or generate QA.",
        "FROZEN CHILD GRAPH\n" + canonical_json(child_graph),
        "ANTI-LEAK CONSTRAINTS\n" + canonical_json({
            "forbidden_child_restatements": dependency["forbidden_child_restatements"],
            "fixed_joint_result_must_not_be_said": dependency["joint_result"],
            "fixed_parent_operand_must_not_be_repeated": dependency["parent_operand"],
        }),
        "OUTPUT JSON SCHEMA\n" + canonical_json(REALIZER_SCHEMA),
        "Return one JSON object only.",
    ])


def apply_anti_leak_gate(realizer: Mapping[str, Any], dependency: Mapping[str, Any],
                         child_graph: Mapping[str, Any]) -> dict[str, Any]:
    turns = realizer["dialogue_turns"]
    combined = normalize_text(" ".join(turn["text"] for turn in turns))
    errors = []
    for phrase in (*LEAKAGE_PHRASES, *dependency["forbidden_child_restatements"]):
        needle = normalize_text(phrase)
        if len(needle) >= 8 and needle in combined:
            errors.append("parent_fact_restatement")
    joint = dependency["joint_result"]
    for value in (joint["value"], joint["meaning"]):
        needle = normalize_text(value)
        if len(needle) >= 5 and needle in combined:
            errors.append("joint_answer_leakage")
    planned = dependency["planned_child_event"]
    realization_id = child_graph["realizes_planned_child_event_id"]
    indices = realizer["event_realization_turn_indices"].get(realization_id, [])
    realization_text = normalize_text(" ".join(turns[index]["text"] for index in indices))
    critical = [planned["subject"], planned["object"], planned["value"]]
    critical = [normalize_text(value) for value in critical if len(normalize_text(value)) >= 3]
    if critical and not all(value in realization_text or value in combined for value in critical):
        errors.append("unverbalized_child_event")
    parent_needles = [normalize_text(dependency["parent_operand"][key])
                      for key in ("object", "value")
                      if len(normalize_text(dependency["parent_operand"][key])) >= 4]
    child_needles = [normalize_text(planned[key]) for key in ("object", "value")
                     if len(normalize_text(planned[key])) >= 4]
    result_needles = [normalize_text(joint[key]) for key in ("value", "meaning")
                      if len(normalize_text(joint[key])) >= 5]
    for turn in turns:
        text = normalize_text(turn["text"])
        if (parent_needles and child_needles and result_needles and
                any(x in text for x in parent_needles) and
                any(x in text for x in child_needles) and
                any(x in text for x in result_needles)):
            errors.append("single_turn_shortcut")
    aliased = [{"alias": f"C{index:03d}", "turn_id": f"{dependency['dependency_id']}:C{index:03d}",
                "session_id": realizer["session_id"], "speaker": turn["speaker"],
                "text": turn["text"], "source_partition": "child"}
               for index, turn in enumerate(turns, 1)]
    return {"decision": "realization_local_gate_pass" if not errors else
            "realization_local_gate_reject", "reasons": sorted(set(errors)),
            "child_dialogue_with_aliases": aliased}


def build_qa_writer_prompt(dependency: Mapping[str, Any], parent_turns: Sequence[Mapping[str, Any]],
                           child_turns: Sequence[Mapping[str, Any]], contract_text: str) -> str:
    frozen = {"dependency_id": dependency["dependency_id"],
        "operator": dependency["operator"], "joint_result": dependency["joint_result"],
        "fixed_parent_evidence": [{key: turn[key] for key in ("alias", "speaker", "text")}
                                  for turn in parent_turns],
        "fixed_child_evidence": [{key: turn[key] for key in ("alias", "speaker", "text")}
                                 for turn in child_turns]}
    return "\n\n".join([f"PROMPT_VERSION: {QA_PROMPT_VERSION}", contract_text,
        "FROZEN MINIMAL DEPENDENCY EVIDENCE\n" + canonical_json(frozen),
        "OUTPUT JSON SCHEMA\n" + canonical_json(QA_SCHEMA), "Return one JSON object only."])


QA_FIELDS = ("dependency_id", "fixed_parent_turn_aliases", "fixed_child_turn_aliases",
             "question", "canonical_answer", "answer_type", "relation_preserved")
QA_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": False,
    "required": list(QA_FIELDS), "properties": {
        "dependency_id": {"type": "string"},
        "fixed_parent_turn_aliases": {"type": "array", "items": {"type": "string"}},
        "fixed_child_turn_aliases": {"type": "array", "items": {"type": "string"}},
        "question": {"type": "string", "minLength": 1},
        "canonical_answer": {"type": "string", "minLength": 1},
        "answer_type": {"enum": list(ANSWER_TYPES)},
        "relation_preserved": {"const": True}}}


def validate_qa_object(raw: Any, dependency: Mapping[str, Any],
                       parent_aliases: Sequence[str], child_aliases: Sequence[str]) -> dict[str, Any]:
    errors = []
    if not isinstance(raw, dict) or set(raw) != set(QA_FIELDS):
        return {"decision": "qa_schema_invalid", "reasons": ["fields"]}
    if raw.get("dependency_id") != dependency["dependency_id"]:
        errors.append("qa_dependency_drift")
    if raw.get("fixed_parent_turn_aliases") != list(parent_aliases) or raw.get(
            "fixed_child_turn_aliases") != list(child_aliases):
        errors.append("qa_alias_drift")
    if normalize_text(raw.get("canonical_answer")) != normalize_text(
            dependency["joint_result"]["value"]):
        errors.append("qa_joint_result_drift")
    if raw.get("answer_type") != dependency["joint_result"]["answer_type"]:
        errors.append("qa_joint_result_drift")
    if raw.get("relation_preserved") is not True or not _nonempty(raw.get("question")):
        errors.append("qa_schema_invalid")
    return {"decision": "qa_accept" if not errors else "qa_schema_invalid",
            "reasons": sorted(set(errors)), **({"qa": raw} if not errors else {})}


def contract_shas() -> dict[str, str]:
    return {
        "dependency_schema_sha256": _sha(DEPENDENCY_BATCH_SCHEMA),
        "positional_dependency_schema_sha256": _sha(
            POSITIONAL_DEPENDENCY_BATCH_SCHEMA),
        "child_graph_schema_sha256": _sha(CHILD_GRAPH_SCHEMA),
        "realizer_schema_sha256": _sha(REALIZER_SCHEMA),
        "qa_schema_sha256": _sha(QA_SCHEMA),
        "contract_source_sha256": sha256_bytes(inspect.getsource(
            project_authoritative_parent).encode() + inspect.getsource(
            normalize_dependency_batch).encode() + inspect.getsource(
            apply_anti_leak_gate).encode()),
    }
