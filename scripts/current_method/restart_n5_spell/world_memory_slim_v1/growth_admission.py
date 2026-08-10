"""Deterministic admission audit for authoritative person/event growth.

This module never promotes model claims.  It only compares append-only Memory
snapshots after the normal semantic review, Durable Compiler, Binder, and
transaction path have completed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


GROWTH_ADMISSION_VERSION = "authoritative_growth_admission_v1"
CONNECTION_RECORD_TYPES = frozenset({
    "relation", "interaction_link", "event_participation",
    "task_participation", "information_source", "thread_participation",
    "add_interaction_link", "add_event_participant",
    "add_task_participant", "add_information_source",
    "add_thread_participant",
})


def _ref(row: Mapping[str, Any]) -> str:
    return str(row.get("memory_ref") or row.get("memory_id") or "")


def _entity_ref(row: Mapping[str, Any]) -> str:
    return str(row.get("entity_ref") or row.get("entity_id") or "")


def _world_ref_values(value: Any, key: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            result.extend(_world_ref_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            result.extend(_world_ref_values(child, key))
    elif isinstance(value, str) and (key.endswith("_ref") or key.endswith("_refs") or
                                     key.endswith("_id") or key.endswith("_ids")):
        if any(marker in value for marker in (
                ":M", ":E", ":EV", ":REL", ":thread_", ":H", ":IL:")):
            result.append(value)
    return result


def audit_authoritative_growth(
    *, memory_initial: Mapping[str, Any], memory_final: Mapping[str, Any],
    accepted_updates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    root_id = str(memory_final.get("root_id") or "")
    initial_records = {_ref(row) for row in memory_initial.get("authoritative_records") or []}
    initial_entities = {_entity_ref(row) for row in memory_initial.get("entities") or []}
    new_records = [row for row in memory_final.get("authoritative_records") or []
                   if _ref(row) not in initial_records]
    new_entities = [row for row in memory_final.get("entities") or []
                    if _entity_ref(row) not in initial_entities]
    new_entity_refs = {_entity_ref(row) for row in new_entities}
    connection_records = [row for row in new_records
                          if str(row.get("record_type") or "") in CONNECTION_RECORD_TYPES]
    connected_entities: set[str] = set()
    for row in connection_records:
        connected_entities.update(str(ref) for ref in row.get("subject_refs") or [])
        payload = row.get("payload") or {}
        connected_entities.update(str(ref) for key, value in payload.items()
                                  for ref in ([value] if key.endswith("_ref") and
                                              isinstance(value, str)
                                              else value if key.endswith("_refs") and
                                              isinstance(value, list) else []))
    orphan_entities = sorted(new_entity_refs - connected_entities)
    generated_records = [row for row in new_records
                         if row.get("origin") == "generated_extension" and
                         row.get("verification_status") == "semantically_verified"]
    cross_root_refs = sorted({ref for row in generated_records
                              for ref in _world_ref_values(row)
                              if root_id and not ref.startswith(root_id + ":")})

    ids: list[str] = []
    for row in memory_final.get("authoritative_records") or []:
        ids.extend(str(row.get(field)) for field in (
            "memory_ref", "allocated_entity_ref", "allocated_event_ref",
            "allocated_relation_ref", "formal_thread_id", "handoff_id")
                   if row.get(field))
    ids.extend(_entity_ref(row) for row in memory_final.get("entities") or []
               if _entity_ref(row))
    duplicate_ids = sorted(value for value, count in Counter(ids).items()
                           if count > 1 and not value.endswith(":M000000"))
    # Entity refs legitimately appear once in the entity registry and once as
    # an allocation field.  Count collisions within each identity namespace,
    # not those intentional cross-artifact references.
    namespace_values: dict[str, list[str]] = defaultdict(list)
    for row in memory_final.get("authoritative_records") or []:
        if row.get("memory_ref"):
            namespace_values["memory"].append(str(row["memory_ref"]))
        for field, namespace in (("allocated_event_ref", "event"),
                                 ("allocated_relation_ref", "relation"),
                                 ("formal_thread_id", "thread"),
                                 ("handoff_id", "handoff")):
            if row.get(field):
                namespace_values[namespace].append(str(row[field]))
    namespace_values["entity"] = [_entity_ref(row)
                                  for row in memory_final.get("entities") or []
                                  if _entity_ref(row)]
    collisions = sorted({value for values in namespace_values.values()
                         for value, count in Counter(values).items() if count > 1})

    group_expected: dict[str, set[str]] = defaultdict(set)
    for update in accepted_updates:
        proposal = update.get("compiled_proposal") or {}
        group = proposal.get("atomic_group_ref")
        if group:
            group_expected[str(group)].add(str(proposal.get("proposal_ref") or ""))
    committed = {str(row.get("proposal_ref") or "") for row in new_records}
    partial_groups = sorted(group for group, expected in group_expected.items()
                            if expected & committed and not expected <= committed)
    counts = Counter(str(row.get("record_type") or "") for row in new_records)
    checks = {
        "root_identity_unchanged": memory_initial.get("root_id") == memory_final.get("root_id"),
        "cross_root_ref_count_zero": not cross_root_refs,
        "orphan_entity_count_zero": not orphan_entities,
        "partial_atomic_commit_count_zero": not partial_groups,
        "memory_id_collision_count_zero": not collisions,
        "generated_records_semantically_verified": len(generated_records) == len(new_records),
    }
    return {
        "growth_admission_version": GROWTH_ADMISSION_VERSION,
        "root_id": root_id,
        "new_authoritative_record_count": len(new_records),
        "new_authoritative_entity_count": len(new_entities),
        "new_authoritative_event_count": counts["event"],
        "new_authoritative_connection_count": sum(
            counts[value] for value in CONNECTION_RECORD_TYPES),
        "new_record_type_distribution": dict(sorted(counts.items())),
        "new_entity_refs": sorted(new_entity_refs),
        "orphan_entity_refs": orphan_entities,
        "cross_root_refs": cross_root_refs,
        "partial_atomic_groups": partial_groups,
        "memory_id_collisions": collisions,
        "diagnostic_duplicate_ids_including_cross_artifact_references": duplicate_ids,
        "checks": checks,
        "world_growth_ready": bool(new_records) and all(checks.values()),
        "person_growth_ready": bool(new_entities) and not orphan_entities and all(checks.values()),
        "event_growth_ready": counts["event"] > 0 and all(checks.values()),
    }
