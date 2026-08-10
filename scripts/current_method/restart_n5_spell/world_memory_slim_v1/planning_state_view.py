"""Compact authoritative planning projection; never a transcript or QA store."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


PLANNING_STATE_SCHEMA_VERSION = "planning_state_view_v2_entity_connections"
INTERACTION_RELATION_AUDIT_VERSION = "interaction_relation_separation_audit_v1"
_AUTHORITATIVE = frozenset({"source_grounded", "semantically_verified"})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("verification_status") or row.get("semantic_status") or "")


def _active_records(memory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    active = set(str(value) for value in (memory.get("active_head_index") or {}).values())
    values = []
    for row in memory.get("authoritative_records") or []:
        ref = str(row.get("memory_ref") or row.get("memory_id") or "")
        if (_status(row) in _AUTHORITATIVE and
                row.get("record_type") != "parent_utterance" and
                (not active or ref in active)):
            values.append(row)
    return values


def _compact_record(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(row.get("payload") or {}))
    payload.pop("text", None)
    record_type = {
        "add_interaction_link": "interaction_link",
        "add_event_participant": "event_participation",
        "add_task_participant": "task_participation",
        "add_information_source": "information_source",
        "add_thread_participant": "thread_participation",
    }.get(str(row.get("record_type") or ""), str(row.get("record_type") or ""))
    return {
        "memory_ref": str(row.get("memory_ref") or row.get("memory_id") or ""),
        "logical_key": str(row.get("logical_key") or ""),
        "record_type": record_type,
        "value": deepcopy(row.get("value")),
        "payload": payload,
        "subject_refs": list(row.get("subject_refs") or row.get("subject_ids") or []),
        "authoritative_version": int(row.get("authoritative_version") or 1),
        "origin": str(row.get("origin") or ""),
        "verification_status": _status(row),
        "support_turn_refs": list(row.get("support_turn_refs") or
                                  row.get("source_turn_refs") or
                                  row.get("source_turn_ids") or []),
        "scene_ref": row.get("scene_ref"),
    }


def compile_planning_state_view(
    *, memory: Mapping[str, Any],
    interaction_links: Sequence[Mapping[str, Any]] | Mapping[str, Any] = (),
    scene_lineage: Sequence[Mapping[str, Any]] = (),
    recent_round_end: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only explicit, active, authoritative long-lived state."""
    entities = []
    for row in memory.get("entities") or []:
        if _status(row) not in _AUTHORITATIVE:
            continue
        entities.append({
            "entity_ref": str(row.get("entity_ref") or row.get("entity_id") or ""),
            "display_name": str(row.get("display_name") or ""),
            "entity_type": str(row.get("entity_type") or "person"),
            "origin": str(row.get("origin") or ""),
            "verification_status": _status(row),
            "source_turn_refs": list(row.get("source_turn_refs") or
                                     row.get("support_turn_refs") or []),
        })
    records = [_compact_record(row) for row in _active_records(memory)]
    explicit_relations = [row for row in records if row["record_type"] == "relation"]
    connection_record_types = {
        "interaction_link", "event_participation", "task_participation",
        "information_source", "thread_participation",
    }
    entity_connections = [row for row in records
                          if row["record_type"] in connection_record_types]
    interactions = []
    if isinstance(interaction_links, Mapping):
        interaction_rows = [row for partition in interaction_links.values()
                            if isinstance(partition, list) for row in partition]
    else:
        interaction_rows = list(interaction_links)
    for row in interaction_rows:
        if _status(row) not in _AUTHORITATIVE:
            continue
        participants = list(row.get("participant_refs") or row.get("entity_refs") or [])
        interactions.append({
            "interaction_link_ref": str(row.get("interaction_link_ref") or
                                        row.get("interaction_link_id") or ""),
            "participant_refs": participants,
            "link_type": str(row.get("link_type") or "verified_dialogue_interaction"),
            "source_turn_refs": list(row.get("source_turn_refs") or []),
            "source_session_refs": list(row.get("source_session_refs") or []),
            "origin": str(row.get("origin") or ""),
            "verification_status": _status(row),
        })
    for row in entity_connections:
        if row["record_type"] not in {"interaction_link", "add_interaction_link"}:
            continue
        payload = row.get("payload") or {}
        participants = [str(payload.get("source_ref") or ""),
                        str(payload.get("target_ref") or "")]
        interactions.append({
            "interaction_link_ref": str(row.get("memory_ref") or ""),
            "participant_refs": [value for value in participants if value],
            "link_type": str(payload.get("link_type") or
                             "verified_generated_interaction"),
            "source_turn_refs": list(row.get("support_turn_refs") or []),
            "source_session_refs": [str(row.get("scene_ref") or "")],
            "origin": str(row.get("origin") or "generated_extension"),
            "verification_status": _status(row),
        })
    result = {
        "planning_state_schema_version": PLANNING_STATE_SCHEMA_VERSION,
        "root_id": str(memory.get("root_id") or ""),
        "entities": entities,
        "explicit_relations": explicit_relations,
        "entity_connections": entity_connections,
        "interaction_links": interactions,
        "active_events": [row for row in records if row["record_type"] == "event"],
        "open_threads": [row for row in records if row["record_type"] == "thread"],
        "tasks": [row for row in records if row["record_type"] in {"task", "responsibility"}],
        "key_states": [row for row in records if row["record_type"] in {
            "state", "relationship_state", "conflict", "constraint"}],
        "knowledge_asymmetries": [row for row in records if row["record_type"] in {
            "knowledge", "information_asymmetry", "third_party_report"}],
        "causal_links": [row for row in records if row["record_type"] == "causal_link"],
        "scene_lineage": [deepcopy(dict(row)) for row in scene_lineage],
        "recent_round_end": deepcopy(dict(recent_round_end or {})),
        "excluded_record_types": ["parent_utterance"],
        "fact_source_role": "planning_projection_only",
        "absence_is_not_novelty_evidence": True,
    }
    result["planning_state_sha256"] = _sha(result)
    return result


def validate_planning_state_view(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("planning_state_schema_version") != PLANNING_STATE_SCHEMA_VERSION:
        errors.append("planning_state_schema_version")
    fields = ("entities", "explicit_relations", "entity_connections",
              "interaction_links", "active_events",
              "open_threads", "tasks", "key_states", "knowledge_asymmetries",
              "causal_links", "scene_lineage")
    for field in fields:
        if not isinstance(value.get(field), list):
            errors.append(f"planning_state_field:{field}")
    for field in ("explicit_relations", "entity_connections", "active_events", "open_threads", "tasks",
                  "key_states", "knowledge_asymmetries", "causal_links"):
        for row in value.get(field) or []:
            if row.get("record_type") == "parent_utterance" or "text" in (row.get("payload") or {}):
                errors.append(f"planning_state_utterance_leak:{field}")
            if row.get("verification_status") not in _AUTHORITATIVE:
                errors.append(f"planning_state_non_authoritative:{field}")
    return {"valid": not errors, "errors": errors,
            "planning_state_schema_version": PLANNING_STATE_SCHEMA_VERSION}


def audit_interaction_relation_separation(
    planning_state: Mapping[str, Any],
) -> dict[str, Any]:
    relation_refs = {str(row.get("memory_ref") or "")
                     for row in planning_state.get("explicit_relations") or []}
    interaction_refs = {str(row.get("interaction_link_ref") or "")
                        for row in planning_state.get("interaction_links") or []}
    social_labels = {"friend", "coworker", "family", "trust", "authority",
                     "manager", "partner"}
    inferred = []
    for row in planning_state.get("interaction_links") or []:
        if str(row.get("link_type") or "").casefold() in social_labels:
            inferred.append(str(row.get("interaction_link_ref") or ""))
    checks = {
        "reference_sets_disjoint": not (relation_refs & interaction_refs),
        "interaction_not_social_relation": not inferred,
        "empty_explicit_relation_graph_allowed": isinstance(
            planning_state.get("explicit_relations"), list),
        "explicit_relations_authoritative": all(
            row.get("verification_status") in _AUTHORITATIVE and
            bool(row.get("support_turn_refs"))
            for row in planning_state.get("explicit_relations") or []),
    }
    return {
        "audit_version": INTERACTION_RELATION_AUDIT_VERSION,
        "checks": checks, "valid": all(checks.values()),
        "interaction_count": len(interaction_refs),
        "explicit_relation_count": len(relation_refs),
        "interaction_social_label_violations": inferred,
        "cooccurrence_relation_upgrade_count": 0,
    }
