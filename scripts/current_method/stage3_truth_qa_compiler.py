"""Pure-local QA compiler for Truth-first Stage3 graphs.

The truth constructor owns the graph and its proof contract.  This module only
projects that frozen truth into several QA skeletons.  It never asks a model to
choose an operation, predicate, answer role, gold value, proof edge, or option.

The compiler deliberately consumes both the constructor materialization and a
pre-dialogue Claim IR bundle.  The former is authoritative for truth; the
latter supplies content-addressed claim IDs.  Every query-plan claim therefore
retains its subject event and all relation endpoints without collapsing two
claims that happen to share an event.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from scripts.current_method import stage3_claim_ir
from scripts.current_method import stage3_truth_constructors


SCHEMA_VERSION = "stage3_truth_qa_bundle_v1"
QA_SCHEMA_VERSION = "stage3_truth_qa_skeleton_v1"
VALIDATION_SCHEMA_VERSION = "stage3_truth_qa_validation_v1"

MIN_QAS = 3
MAX_QAS = 6
PRIMARY_TARGET_FIELDS = (
    "qa_operation",
    "preferred_answer_type",
    "reasoning_subtype",
    "reasoning_template",
    "hop_count",
    "hard_negative_strategy",
)


class TruthQACompileError(ValueError):
    """Raised when frozen constructor truth cannot yield an exact QA family."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _fold(value: Any) -> str:
    return re.sub(r"[\W_]+", " ", _text(value).casefold(), flags=re.UNICODE).strip()


def _slot_surface(value: Any) -> str:
    return _text(value).replace("_", " ").replace("-", " ")


def _possessive(value: Any) -> str:
    text = _text(value).replace("_", " ")
    return f"{text}'" if text.casefold().endswith("s") else f"{text}'s"


def _event_id(event: Mapping[str, Any]) -> str:
    return _text(
        event.get("event_instance_id")
        or event.get("event_id")
        or event.get("id")
        or event.get("node_id")
    )


def _events_by_id(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = graph.get("events")
    if not isinstance(rows, list):
        raise TruthQACompileError("materialized graph has no events array")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TruthQACompileError(f"graph.events[{index}] is not an object")
        event_id = _event_id(row)
        if not event_id or event_id in result:
            raise TruthQACompileError(f"missing or duplicate event id: {event_id!r}")
        result[event_id] = row
    return result


def _claims_by_id(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    claims = bundle.get("claims")
    if not isinstance(claims, list):
        raise TruthQACompileError("Claim IR bundle has no claims array")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(claims):
        if not isinstance(row, dict):
            raise TruthQACompileError(f"claims[{index}] is not an object")
        claim_id = _text(row.get("claim_id"))
        if not claim_id or claim_id in result:
            raise TruthQACompileError(f"missing or duplicate claim_id: {claim_id!r}")
        result[claim_id] = row
    return result


def _node_claim(bundle: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    matches = [
        claim
        for claim in _claims_by_id(bundle).values()
        if claim.get("claim_type") == "node"
        and _text(claim.get("subject_event_id")) == event_id
    ]
    if len(matches) != 1:
        raise TruthQACompileError(
            f"expected exactly one node claim for {event_id}, found {len(matches)}"
        )
    return matches[0]


def _relation_claim(
    bundle: Mapping[str, Any],
    *,
    claim_type: str,
    subject_event_id: str | None = None,
    object_event_id: str | None = None,
    predicate: str | None = None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for claim in _claims_by_id(bundle).values():
        if _text(claim.get("claim_type")) != claim_type:
            continue
        if subject_event_id is not None and _text(claim.get("subject_event_id")) != subject_event_id:
            continue
        if object_event_id is not None and object_event_id not in {
            _text(item) for item in claim.get("object_event_ids") or []
        }:
            continue
        if predicate is not None and _text(claim.get("predicate")) != predicate:
            continue
        matches.append(claim)
    if len(matches) != 1:
        raise TruthQACompileError(
            "expected exactly one relation claim for "
            f"type={claim_type}, subject={subject_event_id}, object={object_event_id}, "
            f"predicate={predicate}; found {len(matches)}"
        )
    return matches[0]


def _membership_claim(bundle: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    return _relation_claim(
        bundle,
        claim_type="membership",
        subject_event_id=event_id,
    )


def _claim_mapping(
    bundle: Mapping[str, Any], claim_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    """Return lossless claim-to-event mappings in query-plan order.

    Several semantic claims may legitimately share a subject event.  The
    reverse mapping therefore stores arrays instead of imposing a false
    one-to-one constraint.
    """

    claims = _claims_by_id(bundle)
    rows: list[dict[str, Any]] = []
    claim_to_event: dict[str, str] = {}
    event_to_claims: dict[str, list[str]] = defaultdict(list)
    for claim_id in dict.fromkeys(_text(item) for item in claim_ids):
        claim = claims.get(claim_id)
        if claim is None:
            raise TruthQACompileError(f"query plan references unknown claim_id: {claim_id}")
        subject = _text(claim.get("subject_event_id"))
        if not subject:
            raise TruthQACompileError(f"claim {claim_id} has no subject event")
        objects = [_text(item) for item in claim.get("object_event_ids") or [] if _text(item)]
        sources = [_text(item) for item in claim.get("source_event_ids") or [] if _text(item)]
        claim_to_event[claim_id] = subject
        event_to_claims[subject].append(claim_id)
        rows.append(
            {
                "claim_id": claim_id,
                "claim_type": _text(claim.get("claim_type")),
                "predicate": _text(claim.get("predicate")),
                "subject_event_id": subject,
                "object_event_ids": objects,
                "source_event_ids": sources,
            }
        )
    return rows, claim_to_event, dict(event_to_claims)


def _resolve_target(
    materialized: Mapping[str, Any], supplied: Mapping[str, Any] | None
) -> dict[str, Any]:
    manifest = materialized.get("truth_manifest")
    graph = materialized.get("graph")
    candidates: list[Any] = [supplied]
    if isinstance(manifest, Mapping):
        candidates.append(manifest.get("immutable_stage64_target"))
    if isinstance(graph, Mapping):
        metadata = graph.get("metadata")
        if isinstance(metadata, Mapping):
            constructor = metadata.get("truth_constructor")
            if isinstance(constructor, Mapping):
                candidates.append(constructor.get("immutable_stage64_target"))
    target = next((copy.deepcopy(item) for item in candidates if isinstance(item, Mapping)), None)
    if target is None:
        raise TruthQACompileError(
            "immutable Stage64 target is required; pass it explicitly or retain it in the truth manifest"
        )
    for field in PRIMARY_TARGET_FIELDS:
        if target.get(field) in (None, ""):
            raise TruthQACompileError(f"immutable Stage64 target is missing {field}")
    return target


def _validate_inputs(
    materialized: Mapping[str, Any], bundle: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(materialized, Mapping):
        raise TruthQACompileError("materialized truth result must be an object")
    graph = materialized.get("graph")
    manifest = materialized.get("truth_manifest")
    if not isinstance(graph, dict) or not isinstance(manifest, dict):
        raise TruthQACompileError("materialized truth result requires graph and truth_manifest")
    if manifest.get("schema_version") != stage3_truth_constructors.MATERIALIZATION_SCHEMA_VERSION:
        raise TruthQACompileError("unsupported truth manifest schema")
    claimed_graph_hash = _text(manifest.get("graph_sha256"))
    if not claimed_graph_hash or canonical_sha256(graph) != claimed_graph_hash:
        raise TruthQACompileError("truth manifest graph hash mismatch")
    proof = manifest.get("proof_contract")
    if not isinstance(proof, dict) or "compiled_gold" not in proof:
        raise TruthQACompileError("truth manifest has no compiled proof contract")
    if not isinstance(bundle, Mapping):
        raise TruthQACompileError("pre-dialogue Claim IR must be an object")
    if bundle.get("schema_version") != stage3_claim_ir.SCHEMA_VERSION:
        raise TruthQACompileError("unsupported Claim IR schema")
    current_plan = bundle.get("current_event_plan")
    if not isinstance(current_plan, dict) or current_plan != graph:
        raise TruthQACompileError("Claim IR graph snapshot differs from truth graph")
    if _text(bundle.get("source_plan_hash")) != stage3_claim_ir.exact_json_sha256(graph):
        raise TruthQACompileError("Claim IR source graph hash mismatch")
    replay = stage3_claim_ir.compile_claim_ir(graph)
    if replay.get("claim_bundle_hash") != bundle.get("claim_bundle_hash"):
        raise TruthQACompileError("Claim IR frozen bundle hash mismatch")
    if replay.get("claims") != bundle.get("claims"):
        raise TruthQACompileError("Claim IR claim rows differ from the frozen graph replay")
    _events_by_id(graph)
    _claims_by_id(bundle)
    return graph, manifest, proof


def _project_gold(projection: Mapping[str, Any], events: Mapping[str, Mapping[str, Any]]) -> Any:
    kind = _text(projection.get("kind"))
    if kind in {"event_value", "event_field", "event_entity"}:
        event_id = _text(projection.get("event_id"))
        event = events.get(event_id)
        if event is None:
            raise TruthQACompileError(f"gold projection references unknown event: {event_id}")
        if kind == "event_value":
            field = "value"
        elif kind == "event_entity":
            field = "entity"
        else:
            field = _text(projection.get("field"))
        if not field or field not in event:
            raise TruthQACompileError(f"gold projection field is unavailable: {field!r}")
        return copy.deepcopy(event[field])
    if kind == "absolute_difference":
        event_ids = projection.get("event_ids")
        if not isinstance(event_ids, list) or len(event_ids) != 2:
            raise TruthQACompileError("absolute_difference needs two event_ids")
        left, right = (events.get(_text(item)) for item in event_ids)
        if left is None or right is None:
            raise TruthQACompileError("absolute_difference references an unknown event")
        for value in (left.get("value"), right.get("value")):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TruthQACompileError("absolute_difference operands must be numeric")
        return abs(left["value"] - right["value"])
    if kind == "count_membership":
        included = projection.get("included")
        if not isinstance(included, bool):
            raise TruthQACompileError("count_membership requires boolean included")
        set_id = projection.get("set_id")
        return sum(
            isinstance(event.get("set_membership"), dict)
            and event["set_membership"].get("included") is included
            and (set_id is None or event["set_membership"].get("set_id") == set_id)
            for event in events.values()
        )
    if kind == "membership_boolean":
        event = events.get(_text(projection.get("event_id")))
        membership = event.get("set_membership") if event else None
        if not isinstance(membership, dict) or not isinstance(membership.get("included"), bool):
            raise TruthQACompileError("membership_boolean needs an explicit membership claim")
        return membership["included"]
    raise TruthQACompileError(f"unsupported local gold projection: {kind!r}")


def _value_type(value: Any, fallback: str = "AttributeValue") -> str:
    if isinstance(value, bool):
        return "YesNo"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "Number"
    return fallback


def _unique_values(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _canonical_json(value)
        if key not in seen:
            result.append(copy.deepcopy(value))
            seen.add(key)
    return result


def _typed_option_defaults(answer_type: str) -> list[Any]:
    return {
        "Person": ["Alex Rivera", "Jordan Lee", "Taylor Morgan", "Casey Park"],
        "GroupOrOrg": ["Harbor Group", "Cedar Association", "North Committee"],
        "LocationOrScope": ["North Hall", "East Wing", "Riverside Studio"],
        "DateOrTime": ["Monday", "June 12", "09:30"],
        "Event": ["Planning Workshop", "Review Meeting", "Community Session"],
        "ObjectOrCommonNoun": ["blue folder", "studio key", "silver badge"],
        "AttributeValue": ["approved", "pending", "restricted"],
        "ProperNounOrNamedEntity": ["Harbor Record", "Cedar Plan", "North File"],
    }.get(answer_type, [])


def _generic_option_values(gold: Any, answer_type: str) -> list[Any]:
    if isinstance(gold, bool):
        return [True, False]
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        values = [gold, max(0, gold - 1), gold + 1, gold + 2]
        return _unique_values(values)[:4]
    return _unique_values([gold, *_typed_option_defaults(answer_type)])[:4]


def _options(gold: Any, preferred: Iterable[Any], answer_type: str) -> list[dict[str, Any]]:
    values = _unique_values([gold, *preferred])
    if len(values) < 2:
        values = _generic_option_values(gold, answer_type)
    else:
        values = values[:4]
    if not any(value == gold and type(value) is type(gold) for value in values):
        values.insert(0, copy.deepcopy(gold))
    # Keep exactly one type-strict gold even when Python would equate 1/True.
    deduped: list[Any] = []
    keys: set[tuple[type, str]] = set()
    for value in values:
        key = (type(value), _canonical_json(value))
        if key not in keys:
            keys.add(key)
            deduped.append(value)
    gold_key = (type(gold), _canonical_json(gold))
    return [
        {
            "option_id": chr(ord("A") + index),
            "id": chr(ord("A") + index),
            "value": copy.deepcopy(value),
            "text": _text(value),
            "is_gold": (type(value), _canonical_json(value)) == gold_key,
            "truth_value": (type(value), _canonical_json(value)) == gold_key,
        }
        for index, value in enumerate(deduped[:4])
    ]


def _primary_option_values(operation: str, events: Mapping[str, Mapping[str, Any]], gold: Any) -> list[Any]:
    if operation == "temporal_state_update":
        return [events[event_id]["value"] for event_id in ("E1", "E2", "E3", "E4")]
    if operation in {"bridge", "cause_consequence", "relation_lookup"}:
        return [events[event_id]["value"] for event_id in ("E2", "E3", "E4", "E5")]
    if operation == "visibility":
        return [events[event_id].get("observer") for event_id in ("E2", "E3", "E6", "E7") if event_id in events]
    if operation == "comparison":
        return [max(0, gold - 2), max(0, gold - 1), gold + 1, gold + 2]
    if operation == "count":
        return [max(0, gold - 1), gold + 1, gold + 2]
    return []


def _primary_question(operation: str, events: Mapping[str, Mapping[str, Any]]) -> str:
    if operation == "temporal_state_update":
        event = events["E1"]
        return (
            f"After resolving every reported, outdated, canceled, and active update, what is the current "
            f"{_slot_surface(event['slot'])} for {event['entity']}?"
        )
    if operation == "bridge":
        first, second = events["E1"], events["E2"]
        return (
            f"Use {_possessive(first['entity'])} {_slot_surface(first['slot'])} record to identify the person; "
            f"what is that person's {_slot_surface(second['slot'])}?"
        )
    if operation == "comparison":
        left, right = events["E1"], events["E2"]
        return (
            f"Currently, what is the absolute difference between {left['entity']} and "
            f"{right['entity']} in {_slot_surface(left['slot'])}?"
        )
    if operation == "visibility":
        event = events["E2"]
        return (
            f"Who may {event['action']} {event['resource']} when {event['condition']}, under the active "
            f"{event['access_class']} permission?"
        )
    if operation == "count":
        event = events["E1"]
        membership = event["set_membership"]
        return (
            f"How many {_slot_surface(event['slot'])} entries for {event['entity']} explicitly qualify "
            f"for {membership['set_id']} under the condition that {membership['condition']}?"
        )
    if operation == "cause_consequence":
        cause, effect = events["E1"], events["E2"]
        return (
            f"What was the consequence for {_possessive(effect['entity'])} {_slot_surface(effect['slot'])} "
            f"after {_possessive(cause['entity'])} {_slot_surface(cause['slot'])} being {cause['value']} "
            "caused the change?"
        )
    if operation == "relation_lookup":
        first, second = events["E1"], events["E2"]
        return (
            f"Use {_possessive(first['entity'])} {_slot_surface(first['slot'])} to identify the subject; "
            f"what is that subject's {_slot_surface(second['slot'])}?"
        )
    raise TruthQACompileError(f"unsupported operation: {operation}")


def _primary_claims(
    operation: str,
    proof: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasoning_ids = [_text(item) for item in proof.get("reasoning_event_ids") or []]
    nodes = [_node_claim(bundle, event_id) for event_id in reasoning_ids]
    if operation == "comparison":
        nodes.append(_relation_claim(bundle, claim_type="comparison", subject_event_id="E1", object_event_id="E2"))
    elif operation == "visibility":
        nodes.append(_relation_claim(bundle, claim_type="visibility", subject_event_id="E2"))
    elif operation == "count":
        nodes = [_membership_claim(bundle, f"E{index}") for index in range(1, 9)]
    elif operation == "cause_consequence":
        nodes.append(_relation_claim(bundle, claim_type="cause", subject_event_id="E1", object_event_id="E2", predicate="causes"))
    elif operation == "relation_lookup":
        nodes.append(_relation_claim(bundle, claim_type="dependency", subject_event_id="E2", object_event_id="E1"))
    return nodes


def _query_steps(claims: Sequence[Mapping[str, Any]], final_operator: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, start=1):
        steps.append(
            {
                "step_index": index,
                "operator": "read_relation_claim" if claim.get("claim_type") != "node" else "read_node_claim",
                "claim_id": claim["claim_id"],
                "subject_event_id": claim.get("subject_event_id"),
                "object_event_ids": copy.deepcopy(claim.get("object_event_ids") or []),
            }
        )
    steps.append(
        {
            "step_index": len(steps) + 1,
            "operator": final_operator,
            "input_claim_ids": [claim["claim_id"] for claim in claims],
        }
    )
    return steps


def _projection_event_ids(projection: Mapping[str, Any]) -> list[str]:
    if isinstance(projection.get("event_ids"), list):
        return [_text(item) for item in projection["event_ids"] if _text(item)]
    event_id = _text(projection.get("event_id"))
    return [event_id] if event_id else []


def _sibling_specs(
    operation: str,
    events: Mapping[str, Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    def node(event_id: str) -> dict[str, Any]:
        return _node_claim(bundle, event_id)

    if operation == "temporal_state_update":
        slot_values = [
            item.get("value")
            for item in events.values()
            if item.get("slot") == events["E1"].get("slot")
        ]
        rows = []
        for status in ("active", "outdated", "canceled"):
            event = next((item for item in events.values() if item.get("status") == status and item.get("slot") == events["E1"].get("slot")), None)
            if event is None:
                continue
            rows.append(
                {
                    "qa_operation": "relation_lookup",
                    "subtype": f"status_qualified_{status}_attribute_lookup",
                    "question": f"Which {_slot_surface(event['slot'])} value for {event['entity']} is marked {status}?",
                    "projection": {"kind": "event_value", "event_id": event["event_id"]},
                    "claims": [node(event["event_id"])],
                    "answer_type": "LocationOrScope",
                    "final_operator": "select_value_by_status",
                    "preferred_options": slot_values,
                }
            )
        return rows
    if operation == "bridge":
        first, second, contrast = events["E1"], events["E2"], events["E4"]
        return [
            {
                "qa_operation": "relation_lookup",
                "subtype": "entity_binder_value_lookup",
                "question": f"Who is listed in {_possessive(first['entity'])} {_slot_surface(first['slot'])} record?",
                "projection": {"kind": "event_value", "event_id": "E1"},
                "claims": [node("E1")],
                "answer_type": "Person",
                "final_operator": "read_binder_value",
                "preferred_options": [events[key]["value"] for key in ("E1", "E4", "E5", "E6")],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "entity_constrained_attribute_lookup",
                "question": f"What is {_possessive(second['entity'])} {_slot_surface(second['slot'])}?",
                "projection": {"kind": "event_value", "event_id": "E2"},
                "claims": [node("E2")],
                "answer_type": "Person",
                "final_operator": "read_terminal_value",
                "preferred_options": [events[key]["value"] for key in ("E2", "E4", "E5", "E6")],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "status_qualified_outdated_attribute_lookup",
                "question": f"Which {_slot_surface(contrast['slot'])} value for {contrast['entity']} is explicitly outdated?",
                "projection": {"kind": "event_value", "event_id": "E4"},
                "claims": [node("E4")],
                "answer_type": "Person",
                "final_operator": "read_outdated_contrast",
                "preferred_options": [events[key]["value"] for key in ("E2", "E4", "E5", "E6")],
            },
        ]
    if operation == "comparison":
        return [
            {
                "qa_operation": "relation_lookup",
                "subtype": "numeric_attribute_lookup",
                "question": f"What is {_possessive(events['E1']['entity'])} current {_slot_surface(events['E1']['slot'])} at time {events['E1']['time_index']}?",
                "projection": {"kind": "event_value", "event_id": "E1"},
                "claims": [node("E1")],
                "answer_type": "Number",
                "final_operator": "read_left_operand",
                "preferred_options": [events[key]["value"] for key in ("E1", "E2", "E3", "E4")],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "numeric_attribute_lookup",
                "question": f"What is {_possessive(events['E2']['entity'])} current {_slot_surface(events['E2']['slot'])} at time {events['E2']['time_index']}?",
                "projection": {"kind": "event_value", "event_id": "E2"},
                "claims": [node("E2")],
                "answer_type": "Number",
                "final_operator": "read_right_operand",
                "preferred_options": [events[key]["value"] for key in ("E1", "E2", "E3", "E4")],
            },
            {
                "qa_operation": "comparison",
                "subtype": "comparison_historical_difference",
                "question": f"At historical time {events['E3']['time_index']}, what was the absolute {_slot_surface(events['E3']['slot'])} difference between {events['E3']['entity']} and {events['E4']['entity']}?",
                "projection": {"kind": "absolute_difference", "event_ids": ["E3", "E4"]},
                "claims": [node("E3"), node("E4")],
                "answer_type": "Number",
                "final_operator": "compute_absolute_difference",
                "preferred_options": [],
            },
        ]
    if operation == "visibility":
        permission = events["E2"]
        wrong_condition = events["E5"]
        return [
            {
                "qa_operation": "relation_lookup",
                "subtype": "resource_owner_lookup",
                "question": f"Who owns {events['E1']['entity']}?",
                "projection": {"kind": "event_value", "event_id": "E1"},
                "claims": [node("E1")],
                "answer_type": "Person",
                "final_operator": "read_resource_owner",
                "preferred_options": [events["E1"]["value"], *[events[key]["observer"] for key in ("E2", "E3", "E6", "E7")]],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "conditioned_permission_action_lookup",
                "question": f"What may {permission['observer']} do with {permission['resource']} when {permission['condition']}?",
                "projection": {"kind": "event_field", "event_id": "E2", "field": "action"},
                "claims": [node("E2"), _relation_claim(bundle, claim_type="visibility", subject_event_id="E2")],
                "answer_type": "AttributeValue",
                "final_operator": "read_typed_permission_action",
                "preferred_options": [events[key]["action"] for key in ("E2", "E3", "E4", "E5")],
            },
            {
                "qa_operation": "visibility",
                "subtype": "observer_permission_scope",
                "question": f"Who is listed for {wrong_condition['action']} access to {wrong_condition['resource']} when {wrong_condition['condition']}?",
                "projection": {"kind": "event_field", "event_id": "E5", "field": "observer"},
                "claims": [node("E5"), _relation_claim(bundle, claim_type="visibility", subject_event_id="E5")],
                "answer_type": "Person",
                "final_operator": "read_observer_under_exact_scope",
                "preferred_options": [events[key]["observer"] for key in ("E2", "E3", "E6", "E7")],
            },
        ]
    if operation == "count":
        first, second = events["E1"], events["E2"]
        set_id = first["set_membership"]["set_id"]
        all_memberships = [_membership_claim(bundle, f"E{index}") for index in range(1, 9)]
        return [
            {
                "qa_operation": "count",
                "subtype": "count_explicit_exclusions",
                "question": (
                    f"How many candidate member entries for {first['entity']} explicitly "
                    f"do not qualify for {set_id} under the condition that "
                    f"{first['set_membership']['condition']}?"
                ),
                "projection": {"kind": "count_membership", "included": False, "set_id": set_id},
                "claims": all_memberships,
                "answer_type": "Number",
                "final_operator": "filter_explicit_memberships_then_count",
                "preferred_options": [],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "explicit_membership_lookup",
                "question": f"Does {first['value']} explicitly qualify for {set_id}?",
                "projection": {"kind": "membership_boolean", "event_id": "E1"},
                "claims": [_membership_claim(bundle, "E1")],
                "answer_type": "YesNo",
                "final_operator": "read_explicit_membership",
                "preferred_options": [True, False],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "explicit_membership_lookup",
                "question": f"Does {second['value']} explicitly qualify for {set_id}?",
                "projection": {"kind": "membership_boolean", "event_id": "E2"},
                "claims": [_membership_claim(bundle, "E2")],
                "answer_type": "YesNo",
                "final_operator": "read_explicit_membership",
                "preferred_options": [True, False],
            },
        ]
    if operation == "cause_consequence":
        cause, effect, contrast = events["E1"], events["E2"], events["E3"]
        cause_claim = _relation_claim(bundle, claim_type="cause", subject_event_id="E1", object_event_id="E2", predicate="causes")
        return [
            {
                "qa_operation": "relation_lookup",
                "subtype": "trigger_reason_attribute_lookup",
                "question": f"What trigger reason is recorded for {cause['entity']}?",
                "projection": {"kind": "event_value", "event_id": "E1"},
                "claims": [node("E1")],
                "answer_type": "AttributeValue",
                "final_operator": "read_cause_trigger",
                "preferred_options": [cause["value"], "schedule conflict", "equipment failure", "staff illness"],
            },
            {
                "qa_operation": "cause_consequence",
                "subtype": "cause_affected_entity_lookup",
                "question": f"Which plan's {_slot_surface(effect['slot'])} was changed by the directed cause from {cause['entity']}?",
                "projection": {"kind": "event_entity", "event_id": "E2"},
                "claims": [node("E1"), node("E2"), cause_claim],
                "answer_type": "ProperNounOrNamedEntity",
                "final_operator": "follow_cause_to_effect_entity",
                "preferred_options": [events[key]["entity"] for key in ("E2", "E3", "E4", "E5")],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "status_qualified_outdated_attribute_lookup",
                "question": f"Which {_slot_surface(contrast['slot'])} value for {contrast['entity']} is marked outdated rather than caused?",
                "projection": {"kind": "event_value", "event_id": "E3"},
                "claims": [node("E3")],
                "answer_type": "Person",
                "final_operator": "read_noncausal_contrast",
                "preferred_options": [events[key]["value"] for key in ("E2", "E3", "E4", "E5")],
            },
        ]
    if operation == "relation_lookup":
        first, second, contrast = events["E1"], events["E2"], events["E3"]
        return [
            {
                "qa_operation": "relation_lookup",
                "subtype": "entity_binder_value_lookup",
                "question": f"Who is selected by {_possessive(first['entity'])} {_slot_surface(first['slot'])} record?",
                "projection": {"kind": "event_value", "event_id": "E1"},
                "claims": [node("E1")],
                "answer_type": "Person",
                "final_operator": "read_entity_binder",
                "preferred_options": [events[key]["value"] for key in ("E1", "E2", "E3", "E4")],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "entity_constrained_attribute_lookup",
                "question": f"What is {_possessive(second['entity'])} {_slot_surface(second['slot'])}?",
                "projection": {"kind": "event_value", "event_id": "E2"},
                "claims": [node("E2")],
                "answer_type": "Person",
                "final_operator": "read_bound_relation",
                "preferred_options": [events[key]["value"] for key in ("E2", "E3", "E4", "E5")],
            },
            {
                "qa_operation": "relation_lookup",
                "subtype": "status_qualified_outdated_attribute_lookup",
                "question": f"Which {_slot_surface(contrast['slot'])} value for {contrast['entity']} is explicitly outdated?",
                "projection": {"kind": "event_value", "event_id": "E3"},
                "claims": [node("E3")],
                "answer_type": "Person",
                "final_operator": "read_status_qualified_relation",
                "preferred_options": [events[key]["value"] for key in ("E2", "E3", "E4", "E5")],
            },
        ]
    raise TruthQACompileError(f"unsupported operation: {operation}")


def _qa_semantics_payload(qa: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in qa.items()
        if key
        not in {
            "qa_semantics_sha256",
            "semantic_validation",
            "evidence_ledger",
            "support_binding_status",
            "independent_validation_report",
            "independent_semantic_validation_report",
            "semantic_validation_report",
        }
    }


def qa_semantics_sha256(qa: Mapping[str, Any]) -> str:
    return canonical_sha256(_qa_semantics_payload(qa))


def _qa_id_for_fields(
    *,
    family_id: str,
    index: int,
    role: str,
    question: str,
    projection: Mapping[str, Any],
    claim_ids: Sequence[str],
) -> str:
    return "tqa_" + canonical_sha256(
        {
            "family_id": family_id,
            "index": index,
            "role": role,
            "question": question,
            "projection": projection,
            "claim_ids": list(claim_ids),
        }
    )[:24]


def _build_qa(
    *,
    family_id: str,
    index: int,
    role: str,
    operation: str,
    subtype: str,
    question: str,
    projection: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    answer_type: str,
    final_operator: str,
    preferred_options: Iterable[Any],
    events: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    target: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gold = _project_gold(projection, events)
    claim_ids = [_text(claim.get("claim_id")) for claim in claims]
    mapping_rows, claim_to_event, event_to_claims = _claim_mapping(bundle, claim_ids)
    reasoning_event_ids = list(
        dict.fromkeys(
            event_id
            for row in mapping_rows
            for event_id in [row["subject_event_id"], *row["object_event_ids"]]
            if event_id in events
        )
    )
    proof = {
        "schema_version": "stage3_truth_qa_proof_v1",
        "truth_skeleton_sha256": manifest["skeleton_sha256"],
        "truth_graph_sha256": manifest["graph_sha256"],
        "claim_bundle_hash": bundle["claim_bundle_hash"],
        "gold_projection": copy.deepcopy(projection),
        "compiled_gold": copy.deepcopy(gold),
        "required_claim_ids": claim_ids,
        "reasoning_path_claim_ids": claim_ids,
        "reasoning_event_ids": reasoning_event_ids,
        "claim_event_mapping": mapping_rows,
        "claim_id_to_event_id": claim_to_event,
        "event_id_to_claim_ids": event_to_claims,
        "derivation_rule": final_operator,
    }
    answer_event_ids = _projection_event_ids(projection)
    if not answer_event_ids and reasoning_event_ids:
        answer_event_ids = list(reasoning_event_ids)
    proof.update(
        {
            "proof_id": "tproof_"
            + canonical_sha256(
                {
                    "claims": claim_ids,
                    "projection": projection,
                    "operator": final_operator,
                }
            )[:24],
            "required_event_instance_ids": list(reasoning_event_ids),
            "reasoning_path_event_instance_ids": list(reasoning_event_ids),
            "answer_event_instance_ids": list(answer_event_ids),
            "dependency_closure_event_instance_ids": list(reasoning_event_ids),
        }
    )
    query_plan = {
        "schema_version": "stage3_truth_query_plan_v1",
        "steps": _query_steps(claims, final_operator),
        "required_claim_ids": claim_ids,
        "reasoning_event_ids": reasoning_event_ids,
        "claim_id_to_event_id": copy.deepcopy(claim_to_event),
        "final_operator": final_operator,
        "gold_projection": copy.deepcopy(projection),
        "resolved_reasoning_event_instance_ids": list(reasoning_event_ids),
        "resolved_required_event_instance_ids": list(reasoning_event_ids),
        "resolved_supporting_event_instance_ids": list(reasoning_event_ids),
        "resolved_gold_event_instance_ids": list(answer_event_ids),
        "resolved_distractor_event_instance_ids": [],
        "reasoning_anchor_path": list(reasoning_event_ids),
        "required_anchor_refs": list(reasoning_event_ids),
        "supporting_anchor_refs": list(reasoning_event_ids),
        "gold_anchor_refs": list(answer_event_ids),
        "distractor_anchor_refs": [],
        "claim_id_to_event_instance_id": copy.deepcopy(claim_to_event),
        "event_instance_id_to_claim_id": {
            event_id: claim_ids[0]
            for event_id, claim_ids in event_to_claims.items()
            if claim_ids
        },
        "event_instance_id_to_claim_ids": copy.deepcopy(event_to_claims),
    }
    options = _options(gold, preferred_options, answer_type)
    gold_labels = [item["id"] for item in options if item["is_gold"] is True]
    for option in options:
        option_value = option["value"]
        matching_event_ids = [
            event_id
            for event_id, event in events.items()
            if any(
                type(candidate) is type(option_value) and candidate == option_value
                for candidate in (
                    event.get("value"),
                    event.get("entity"),
                    event.get("observer"),
                    event.get("owner"),
                    event.get("resource"),
                    event.get("action"),
                )
            )
        ]
        source_event_id = (
            answer_event_ids[0]
            if option["is_gold"] is True and answer_event_ids
            else matching_event_ids[0]
            if matching_event_ids
            else reasoning_event_ids[0]
            if reasoning_event_ids
            else None
        )
        option["source_event_instance_id"] = source_event_id
        option["anchor_ref"] = source_event_id
        option["anchor_kind"] = (
            "local_derived_truth" if not matching_event_ids else "constructor_event"
        )
    query_plan["resolved_distractor_event_instance_ids"] = list(
        dict.fromkeys(
            _text(option.get("source_event_instance_id"))
            for option in options
            if option["is_gold"] is not True
            and _text(option.get("source_event_instance_id"))
        )
    )
    query_plan["distractor_anchor_refs"] = copy.deepcopy(
        query_plan["resolved_distractor_event_instance_ids"]
    )
    qa: dict[str, Any] = {
        "schema_version": QA_SCHEMA_VERSION,
        "qa_id": "",
        "dialogue_family_id": family_id,
        "qa_index": index,
        "qa_role": role,
        "qa_operation": operation,
        "qa_subtype": subtype,
        "reasoning_subtype": subtype,
        "reasoning_template": final_operator,
        "hop_count": max(1, len(claims)),
        "hard_negative_strategy": "truth_first_locally_grounded_options",
        "forbid_max_time_shortcut": False,
        "question_type": "single_choice",
        "question": question,
        "question_text": question,
        "question_text_draft": question,
        "gold_answer": copy.deepcopy(gold),
        "gold_answer_text": _text(gold),
        "normalized_gold_answer": _fold(gold),
        "gold": gold_labels,
        "answer_type": answer_type,
        "answer_format": "single_choice_mcq",
        "options": options,
        "proof": proof,
        "query_plan": query_plan,
        "provenance": {
            "source": "local_truth_constructor",
            "graph_id": manifest.get("graph_id"),
            "truth_skeleton_sha256": manifest["skeleton_sha256"],
            "truth_graph_sha256": manifest["graph_sha256"],
            "claim_bundle_hash": bundle["claim_bundle_hash"],
            "api_calls_made": 0,
        },
        "semantic_contract_frozen": True,
        "truth_first_generate_by_construction": True,
    }
    if role == "primary":
        assert target is not None
        qa.update(
            {
                "qa_operation": target["qa_operation"],
                "answer_type": target["preferred_answer_type"],
                "reasoning_subtype": target["reasoning_subtype"],
                "reasoning_template": target["reasoning_template"],
                "hop_count": target["hop_count"],
                "hard_negative_strategy": target["hard_negative_strategy"],
                "forbid_max_time_shortcut": bool(target.get("forbid_max_time_shortcut", False)),
                "immutable_stage64_target": copy.deepcopy(target),
                "fulfills_immutable_stage64_target": True,
                "truth_manifest_proof_contract": copy.deepcopy(manifest["proof_contract"]),
            }
        )
        query_plan.update(
            {
                "reasoning_subtype": target["reasoning_subtype"],
                "reasoning_template": target["reasoning_template"],
                "hop_count": target["hop_count"],
                "hard_negative_strategy": target["hard_negative_strategy"],
                "forbid_max_time_shortcut": bool(target.get("forbid_max_time_shortcut", False)),
            }
        )
    else:
        query_plan.update(
            {
                "qa_operation": operation,
                "qa_subtype": subtype,
                "reasoning_subtype": subtype,
                "reasoning_template": final_operator,
                "hop_count": max(1, len(claims)),
                "hard_negative_strategy": "truth_first_locally_grounded_options",
                "forbid_max_time_shortcut": False,
                "answer_type": answer_type,
                "gold_answer": _text(gold),
                "normalized_gold_answer": _fold(gold),
            }
        )
    query_plan.setdefault("qa_operation", qa["qa_operation"])
    query_plan.setdefault("qa_subtype", qa["qa_subtype"])
    query_plan.setdefault("answer_type", qa["answer_type"])
    query_plan.setdefault("gold_answer", qa["gold_answer_text"])
    query_plan.setdefault("normalized_gold_answer", qa["normalized_gold_answer"])
    if operation == "count" and role == "primary":
        membership = events["E1"].get("set_membership") or {}
        query_plan["filter_predicate"] = copy.deepcopy(
            membership.get("condition")
        )
    qa["qa_id"] = _qa_id_for_fields(
        family_id=family_id,
        index=index,
        role=role,
        question=question,
        projection=projection,
        claim_ids=claim_ids,
    )
    qa["qa_semantics_sha256"] = qa_semantics_sha256(qa)
    return qa


def _bridge_question_issues(qa: Mapping[str, Any], events: Mapping[str, Mapping[str, Any]]) -> list[str]:
    question = _fold(qa.get("question_text_draft"))
    first, second = events["E1"], events["E2"]
    required = (_fold(first["entity"]), _fold(_slot_surface(first["slot"])), _fold(_slot_surface(second["slot"])))
    issues = [f"bridge_question_missing_{name}" for name, value in zip(("binder_entity", "binder_slot", "answer_slot"), required) if value not in question]
    forbidden = (_fold(first["value"]), _fold(second["value"]), _fold(second["entity"]))
    if any(value and value in question for value in forbidden):
        issues.append("bridge_question_names_intermediate_or_gold")
    if qa.get("proof", {}).get("reasoning_event_ids") != ["E1", "E2"]:
        issues.append("bridge_reasoning_path_is_not_true_two_hop")
    if "E3" in qa.get("proof", {}).get("reasoning_event_ids", []):
        issues.append("bridge_control_event_entered_reasoning_path")
    return issues


def validate_compiled_qa(
    qa: Mapping[str, Any],
    materialized: Mapping[str, Any],
    claim_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate one sibling without invalidating its family."""

    issues: list[dict[str, Any]] = []
    try:
        graph, manifest, manifest_proof = _validate_inputs(materialized, claim_bundle)
        events = _events_by_id(graph)
    except TruthQACompileError as error:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "pass": False,
            "issue_count": 1,
            "issues": [{"code": "frozen_input_invalid", "detail": str(error)}],
        }
    if not isinstance(qa, Mapping) or qa.get("schema_version") != QA_SCHEMA_VERSION:
        issues.append({"code": "qa_schema_invalid"})
    elif _text(qa.get("qa_semantics_sha256")) != qa_semantics_sha256(qa):
        issues.append({"code": "qa_semantics_hash_mismatch"})
    proof = qa.get("proof") if isinstance(qa, Mapping) else None
    if not isinstance(proof, Mapping):
        issues.append({"code": "qa_proof_missing"})
        proof = {}
    else:
        if proof.get("truth_graph_sha256") != manifest.get("graph_sha256"):
            issues.append({"code": "qa_truth_graph_hash_mismatch"})
        if proof.get("claim_bundle_hash") != claim_bundle.get("claim_bundle_hash"):
            issues.append({"code": "qa_claim_bundle_hash_mismatch"})
    projection = proof.get("gold_projection")
    if isinstance(projection, Mapping):
        try:
            expected_gold = _project_gold(projection, events)
            if type(expected_gold) is not type(qa.get("gold_answer")) or expected_gold != qa.get("gold_answer"):
                issues.append({"code": "qa_gold_does_not_match_projection"})
            if type(expected_gold) is not type(proof.get("compiled_gold")) or expected_gold != proof.get("compiled_gold"):
                issues.append({"code": "proof_gold_does_not_match_projection"})
        except TruthQACompileError as error:
            issues.append({"code": "gold_projection_invalid", "detail": str(error)})
    else:
        issues.append({"code": "gold_projection_missing"})
    if isinstance(projection, Mapping) and isinstance(proof.get("required_claim_ids"), list):
        expected_qa_id = _qa_id_for_fields(
            family_id=_text(qa.get("dialogue_family_id")),
            index=qa.get("qa_index") if isinstance(qa.get("qa_index"), int) else -1,
            role=_text(qa.get("qa_role")),
            question=_text(qa.get("question_text_draft")),
            projection=projection,
            claim_ids=proof["required_claim_ids"],
        )
        if qa.get("qa_id") != expected_qa_id:
            issues.append({"code": "qa_stable_id_mismatch"})
    claim_ids = proof.get("required_claim_ids")
    if not isinstance(claim_ids, list) or not claim_ids:
        issues.append({"code": "required_claim_ids_missing"})
    else:
        try:
            expected_rows, expected_forward, expected_reverse = _claim_mapping(claim_bundle, claim_ids)
            if proof.get("claim_event_mapping") != expected_rows:
                issues.append({"code": "claim_event_mapping_lossy_or_incorrect"})
            if proof.get("claim_id_to_event_id") != expected_forward:
                issues.append({"code": "claim_to_event_mapping_incorrect"})
            if proof.get("event_id_to_claim_ids") != expected_reverse:
                issues.append({"code": "event_to_claim_mapping_incorrect"})
        except TruthQACompileError as error:
            issues.append({"code": "query_claim_mapping_invalid", "detail": str(error)})
    query_plan = qa.get("query_plan") if isinstance(qa, Mapping) else None
    if not isinstance(query_plan, Mapping):
        issues.append({"code": "query_plan_missing"})
    else:
        if query_plan.get("required_claim_ids") != proof.get("required_claim_ids"):
            issues.append({"code": "query_plan_claim_path_drift"})
        if query_plan.get("reasoning_event_ids") != proof.get("reasoning_event_ids"):
            issues.append({"code": "query_plan_event_path_drift"})
        if query_plan.get("claim_id_to_event_id") != proof.get("claim_id_to_event_id"):
            issues.append({"code": "query_plan_claim_mapping_drift"})
        if query_plan.get("gold_projection") != proof.get("gold_projection"):
            issues.append({"code": "query_plan_gold_projection_drift"})
        if query_plan.get("final_operator") != proof.get("derivation_rule"):
            issues.append({"code": "query_plan_derivation_drift"})
        steps = query_plan.get("steps")
        if not isinstance(steps, list):
            issues.append({"code": "query_plan_steps_missing"})
        else:
            step_claim_ids = [
                _text(step.get("claim_id"))
                for step in steps
                if isinstance(step, Mapping) and step.get("claim_id") is not None
            ]
            if step_claim_ids != proof.get("required_claim_ids"):
                issues.append({"code": "query_plan_step_claim_order_drift"})
    options = qa.get("options")
    if not isinstance(options, list) or len(options) < 2:
        issues.append({"code": "qa_options_missing"})
    elif sum(item.get("is_gold") is True for item in options if isinstance(item, Mapping)) != 1:
        issues.append({"code": "qa_options_do_not_have_exactly_one_gold"})
    elif not any(
        isinstance(item, Mapping)
        and item.get("is_gold") is True
        and type(item.get("value")) is type(qa.get("gold_answer"))
        and item.get("value") == qa.get("gold_answer")
        for item in options
    ):
        issues.append({"code": "qa_gold_option_value_mismatch"})
    if qa.get("qa_role") == "primary":
        target = qa.get("immutable_stage64_target")
        if not isinstance(target, Mapping):
            issues.append({"code": "primary_target_missing"})
        else:
            for field in PRIMARY_TARGET_FIELDS:
                qa_field = "answer_type" if field == "preferred_answer_type" else field
                if qa.get(qa_field) != target.get(field):
                    issues.append({"code": "primary_target_field_mismatch", "field": field})
            if target.get("qa_operation") != graph.get("operation"):
                issues.append({"code": "primary_operation_not_graph_operation"})
        if qa.get("truth_manifest_proof_contract") != manifest_proof:
            issues.append({"code": "primary_truth_manifest_proof_drift"})
        manifest_projection = manifest_proof.get("gold_projection")
        if projection != manifest_projection or qa.get("gold_answer") != manifest_proof.get("compiled_gold"):
            issues.append({"code": "primary_gold_or_projection_drift"})
        expected_reasoning = [_text(item) for item in manifest_proof.get("reasoning_event_ids") or []]
        if proof.get("reasoning_event_ids") != expected_reasoning:
            issues.append({"code": "primary_reasoning_events_drift"})
        operation = _text(graph.get("operation"))
        if operation == "bridge":
            issues.extend({"code": code} for code in _bridge_question_issues(qa, events))
        elif operation == "count":
            if projection.get("kind") != "count_membership":
                issues.append({"code": "count_not_derived_from_membership"})
            claim_types = {
                row.get("claim_type") for row in proof.get("claim_event_mapping") or [] if isinstance(row, Mapping)
            }
            if claim_types != {"membership"}:
                issues.append({"code": "count_proof_uses_nonmembership_claims"})
        elif operation == "cause_consequence":
            cause_rows = [
                row for row in proof.get("claim_event_mapping") or []
                if isinstance(row, Mapping) and row.get("claim_type") == "cause"
            ]
            if len(cause_rows) != 1 or cause_rows[0].get("subject_event_id") != "E1" or cause_rows[0].get("object_event_ids") != ["E2"]:
                issues.append({"code": "cause_direction_missing_or_wrong"})
        elif operation == "visibility":
            event = events.get("E2", {})
            for field in ("observer", "action", "resource", "condition", "scope", "status"):
                if field not in event:
                    issues.append({"code": "visibility_typed_field_missing", "field": field})
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "qa_id": qa.get("qa_id") if isinstance(qa, Mapping) else None,
        "pass": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def validate_and_filter_qas(
    compiled: Mapping[str, Any],
    materialized: Mapping[str, Any],
    claim_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate siblings independently and retain every semantic-clean row."""

    rows = compiled.get("qas") if isinstance(compiled, Mapping) else None
    if not isinstance(rows, list):
        raise TruthQACompileError("compiled QA bundle has no qas array")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        report = validate_compiled_qa(row, materialized, claim_bundle)
        if report["pass"]:
            accepted.append(copy.deepcopy(row))
        else:
            rejected.append(
                {
                    "qa_id": row.get("qa_id") if isinstance(row, Mapping) else None,
                    "validation": report,
                }
            )
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "dialogue_family_id": compiled.get("dialogue_family_id"),
        "dialogue_retained": bool(accepted),
        "semantic_clean_qa_count": len(accepted),
        "semantic_reject_qa_count": len(rejected),
        "semantic_clean_qas": accepted,
        "semantic_rejects": rejected,
        "api_calls_made": 0,
    }


def compile_truth_first_qas(
    materialized: Mapping[str, Any],
    pre_dialogue_claim_ir: Mapping[str, Any],
    *,
    immutable_stage64_target: Mapping[str, Any] | None = None,
    qa_count: int = 4,
    seed_sample_id: str = "",
    lineage_group_id: str = "",
) -> dict[str, Any]:
    """Compile one immutable primary QA and 2--5 independent sibling QAs."""

    if isinstance(qa_count, bool) or not isinstance(qa_count, int) or not MIN_QAS <= qa_count <= MAX_QAS:
        raise ValueError(f"qa_count must be between {MIN_QAS} and {MAX_QAS}")
    graph, manifest, proof = _validate_inputs(materialized, pre_dialogue_claim_ir)
    events = _events_by_id(graph)
    operation = _text(graph.get("operation"))
    if operation not in stage3_truth_constructors.OPERATIONS:
        raise TruthQACompileError(f"unsupported truth operation: {operation!r}")
    target = _resolve_target(materialized, immutable_stage64_target)
    if _text(target.get("qa_operation")) != operation:
        raise TruthQACompileError("immutable target operation differs from truth graph")
    projected = _project_gold(proof["gold_projection"], events)
    if type(projected) is not type(proof["compiled_gold"]) or projected != proof["compiled_gold"]:
        raise TruthQACompileError("truth manifest compiled gold is inconsistent")
    family_id = "tfam_" + canonical_sha256(
        {
            "graph_sha256": manifest["graph_sha256"],
            "claim_bundle_hash": pre_dialogue_claim_ir["claim_bundle_hash"],
            "seed_sample_id": seed_sample_id,
            "lineage_group_id": lineage_group_id,
        }
    )[:24]
    primary_claims = _primary_claims(operation, proof, pre_dialogue_claim_ir)
    primary = _build_qa(
        family_id=family_id,
        index=0,
        role="primary",
        operation=operation,
        subtype=f"{operation}_immutable_target",
        question=_primary_question(operation, events),
        projection=proof["gold_projection"],
        claims=primary_claims,
        answer_type=_text(target["preferred_answer_type"]),
        final_operator=_text(proof["derivation_rule"]),
        preferred_options=_primary_option_values(operation, events, proof["compiled_gold"]),
        events=events,
        manifest=manifest,
        bundle=pre_dialogue_claim_ir,
        target=target,
    )
    sibling_specs = _sibling_specs(operation, events, pre_dialogue_claim_ir)
    if len(sibling_specs) + 1 < qa_count:
        raise TruthQACompileError(
            f"operation {operation} provides only {len(sibling_specs) + 1} local QA skeletons"
        )
    qas = [primary]
    for index, spec in enumerate(sibling_specs[: qa_count - 1], start=1):
        gold = _project_gold(spec["projection"], events)
        qas.append(
            _build_qa(
                family_id=family_id,
                index=index,
                role="sibling",
                operation=_text(spec.get("qa_operation") or operation),
                subtype=spec["subtype"],
                question=spec["question"],
                projection=spec["projection"],
                claims=spec["claims"],
                answer_type=spec.get("answer_type") or _value_type(gold),
                final_operator=spec["final_operator"],
                preferred_options=spec.get("preferred_options") or [],
                events=events,
                manifest=manifest,
                bundle=pre_dialogue_claim_ir,
                target=None,
            )
        )
    reports = [validate_compiled_qa(qa, materialized, pre_dialogue_claim_ir) for qa in qas]
    failures = [report for report in reports if not report["pass"]]
    if failures:
        raise TruthQACompileError(f"local QA self-validation failed: {failures}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "dialogue_family_id": family_id,
        "lineage_group_id": _text(lineage_group_id),
        "source_graph_id": graph.get("graph_id"),
        "source_graph_operation": operation,
        "truth_graph_sha256": manifest["graph_sha256"],
        "claim_bundle_hash": pre_dialogue_claim_ir["claim_bundle_hash"],
        "immutable_stage64_target": target,
        "target_fulfillment_qa_id": primary["qa_id"],
        "qa_count": len(qas),
        "qas": qas,
        "semantic_validation": reports,
        "api_calls_made": 0,
        "semantic_contract_frozen": True,
    }
    result["bundle_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "MAX_QAS",
    "MIN_QAS",
    "QA_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TruthQACompileError",
    "canonical_sha256",
    "compile_truth_first_qas",
    "qa_semantics_sha256",
    "validate_and_filter_qas",
    "validate_compiled_qa",
]
