"""Controlled World Expansion v1 and Delta Affordance Compiler v3."""

from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from .operation_registry import OPERATION_REGISTRY, operation_contract


CONTROLLED_WORLD_EXPANSION_VERSION = "controlled_world_expansion_v1"
AFFORDANCE_SCHEMA_VERSION_V3 = "delta_affordance_compiler_v3"
ACTION_MENU_MAXIMUM = 16


def _memory_ref(record: Mapping[str, Any]) -> str:
    return str(record.get("memory_ref") or record.get("memory_id") or
               record.get("entity_ref") or record.get("entity_id") or "")


def _authoritative(record: Mapping[str, Any]) -> bool:
    status = str(record.get("verification_status") or record.get("semantic_status") or "")
    return status in {"source_grounded", "semantically_verified"}


def _entities(memory: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = []
    for row in memory.get("entities") or []:
        if _authoritative(row):
            ref = _memory_ref(row)
            if ref:
                values.append({"entity_ref": ref,
                               "display_name": str(row.get("display_name") or ref)})
    return sorted(values, key=lambda row: row["entity_ref"])


def _active_records(memory: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = list(memory.get("authoritative_records") or [])
    if not records:
        for field in ("current_states", "open_threads", "event_history", "relations"):
            records.extend(memory.get(field) or [])
    active_index = memory.get("active_head_index") or {}
    if active_index:
        active_refs = set(str(ref) for ref in active_index.values())
        records = [row for row in records if _memory_ref(row) in active_refs]
    return sorted((deepcopy(dict(row)) for row in records if _authoritative(row)),
                  key=_memory_ref)


def _action(
    operation: str, connected: Sequence[str], entities: Sequence[str], *,
    frozen_payload: Mapping[str, Any] | None = None,
    target_memory_ref: str | None = None,
    current_value: Any = None,
    authoritative_version: int | None = None,
    narrative: Sequence[str] = (), constraints: Sequence[str] = (),
) -> dict[str, Any]:
    contract = operation_contract(operation)
    canonical = "|".join([operation, *(sorted(connected)), target_memory_ref or ""])
    return {
        "_sort_key": (-len(set(connected)), operation, canonical),
        "operation": operation,
        "operation_class": contract["operation_class"],
        "connected_memory_refs": sorted(set(connected)),
        "allowed_existing_entity_refs": sorted(set(entities)),
        "allows_temporary_entities": bool(contract["allows_temporary_refs"]),
        "payload_schema": deepcopy(contract["payload_schema"]),
        "frozen_payload": deepcopy(dict(frozen_payload or {})),
        "target_memory_ref": target_memory_ref,
        "current_value": deepcopy(current_value),
        "authoritative_version": authoritative_version,
        "narrative_opportunities": list(narrative),
        "consistency_constraints": list(constraints),
        "allows_connectivity_bridge": contract["allows_connectivity_bridge"],
        "complexity_weight": contract["complexity_weight"],
    }


def compile_action_menu_v3(
    memory: Mapping[str, Any], *, root_id: str,
    interaction_links: Sequence[Mapping[str, Any]] = (), maximum_actions: int = 16,
) -> dict[str, Any]:
    if str(memory.get("root_id")) != root_id:
        raise ValueError("affordance_root_identity_mismatch")
    maximum_actions = min(maximum_actions, ACTION_MENU_MAXIMUM)
    entities = _entities(memory)
    refs = [row["entity_ref"] for row in entities]
    records = _active_records(memory)
    candidates: list[dict[str, Any]] = []
    if refs:
        first = refs[0]
        candidates.extend([
            _action("add_person", [first], refs,
                    frozen_payload={"scene_presence": "mentioned_only"},
                    narrative=["introduce a reported or mentioned participant"],
                    constraints=["temporary person cannot be a speaker", "display_role cannot assert a relation"]),
            _action("create_state", [first], refs,
                    frozen_payload={"subject_ref": first},
                    narrative=["create a previously absent explicit state dimension"]),
            _action("open_thread", [first], refs,
                    frozen_payload={"subject_refs": [first]},
                    narrative=["open one concrete follow-up thread"]),
            _action("introduce_external_constraint", [first], refs,
                    frozen_payload={"affected_refs": [first]}),
        ])
    if len(refs) >= 2:
        pair = refs[:2]
        candidates.extend([
            _action("add_relation", pair, refs,
                    frozen_payload={"source_ref": pair[0], "target_ref": pair[1]},
                    narrative=["make an explicit registered relationship"]),
            _action("add_event", pair, refs,
                    frozen_payload={"participant_refs": pair},
                    narrative=["introduce a concrete shared event"]),
            _action("add_shared_task", pair, refs,
                    frozen_payload={"participant_refs": pair},
                    narrative=["give existing speakers a shared task"]),
            _action("create_conflict", pair, refs,
                    frozen_payload={"participant_refs": pair}),
            _action("create_misunderstanding", pair, refs,
                    frozen_payload={"participant_refs": pair}),
            _action("add_information_asymmetry", pair, refs,
                    frozen_payload={"knower_refs": [pair[0]], "non_knower_refs": [pair[1]]}),
        ])
    if len(refs) >= 3:
        candidates.append(_action(
            "add_third_party_report", refs[:3], refs,
            frozen_payload={"reporter_ref": refs[0], "about_refs": refs[1:3]}))

    for record in records:
        ref = _memory_ref(record)
        record_type = str(record.get("record_type") or record.get("target_type") or "")
        value = deepcopy(record.get("value"))
        version = int(record.get("authoritative_version") or record.get("version") or 1)
        subject_refs = [str(value) for value in record.get("subject_ids") or record.get("subject_refs") or []]
        connected = [ref, *subject_refs]
        if record_type in {"state", "current_state"}:
            candidates.append(_action("update_state", connected, refs,
                                      frozen_payload={"target_memory_ref": ref},
                                      target_memory_ref=ref, current_value=value,
                                      authoritative_version=version))
        if record_type in {"trust", "trust_state"}:
            candidates.append(_action("change_trust", connected, refs,
                                      frozen_payload={"target_memory_ref": ref},
                                      target_memory_ref=ref, current_value=value,
                                      authoritative_version=version))
        if record_type in {"authority", "authority_state"}:
            candidates.append(_action("change_authority", connected, refs,
                                      frozen_payload={"target_memory_ref": ref},
                                      target_memory_ref=ref, current_value=value,
                                      authoritative_version=version))
        if record_type in {"thread", "open_thread"}:
            candidates.append(_action("close_thread", connected, refs,
                                      frozen_payload={"target_memory_ref": ref},
                                      target_memory_ref=ref, current_value=value,
                                      authoritative_version=version))
        if record_type in {"conflict"}:
            candidates.append(_action("resolve_conflict", connected, refs,
                                      frozen_payload={"target_memory_ref": ref},
                                      target_memory_ref=ref, current_value=value,
                                      authoritative_version=version))
        if record_type in {"task", "responsibility", "role"} and len(refs) >= 2:
            candidates.append(_action(
                "add_role_handoff", connected, refs, target_memory_ref=ref,
                current_value=value, authoritative_version=version,
                frozen_payload={"target_memory_ref": ref,
                                "source_ref": refs[0], "target_ref": refs[1]}))

    # Deduplicate deterministically, then enforce class/operation quotas.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in candidates:
        key = (row["operation"], tuple(row["connected_memory_refs"]), row["target_memory_ref"])
        unique.setdefault(key, row)
    ordered = sorted(unique.values(), key=lambda row: row["_sort_key"])
    selected: list[dict[str, Any]] = []
    operation_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    class_max = {"create": maximum_actions, "update": 4, "transition": 4}
    for row in ordered:
        if len(selected) >= maximum_actions:
            break
        if operation_counts[row["operation"]] >= 2:
            continue
        if class_counts[row["operation_class"]] >= class_max[row["operation_class"]]:
            continue
        selected.append(row)
        operation_counts[row["operation"]] += 1
        class_counts[row["operation_class"]] += 1
    for index, row in enumerate(selected, 1):
        row.pop("_sort_key", None)
        row["action_id"] = f"A{index:04d}"
        row["root_id"] = root_id
    return {
        "controlled_world_expansion_version": CONTROLLED_WORLD_EXPANSION_VERSION,
        "affordance_schema_version": AFFORDANCE_SCHEMA_VERSION_V3,
        "root_id": root_id,
        "actions": selected,
        "budget": {
            "maximum": ACTION_MENU_MAXIMUM, "create_target_minimum": 6,
            "update_maximum": 4, "transition_maximum": 4,
            "per_operation_maximum": 2,
        },
        "distribution": dict(sorted(class_counts.items())),
        "create_action_deficit": max(0, 6 - class_counts["create"]),
        "interaction_link_refs": sorted(str(row.get("interaction_link_id"))
                                          for row in interaction_links),
    }


def validate_action_menu_v3(menu: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    actions = menu.get("actions")
    if menu.get("affordance_schema_version") != AFFORDANCE_SCHEMA_VERSION_V3:
        errors.append("affordance_schema_version")
    if not isinstance(actions, list):
        return {"valid": False, "errors": [*errors, "actions_not_list"]}
    if len(actions) > ACTION_MENU_MAXIMUM:
        errors.append("action_menu_over_budget")
    ids = [row.get("action_id") for row in actions if isinstance(row, Mapping)]
    if len(ids) != len(set(ids)):
        errors.append("action_id_duplicate")
    op_counts = Counter(str(row.get("operation")) for row in actions)
    class_counts = Counter(str(row.get("operation_class")) for row in actions)
    if any(count > 2 for count in op_counts.values()):
        errors.append("operation_quota_exceeded")
    if class_counts["update"] > 4 or class_counts["transition"] > 4:
        errors.append("class_quota_exceeded")
    for row in actions:
        operation = str(row.get("operation") or "")
        if operation not in OPERATION_REGISTRY:
            errors.append(f"unknown_operation:{operation}")
        elif row.get("operation_class") != OPERATION_REGISTRY[operation]["operation_class"]:
            errors.append(f"operation_class_mismatch:{operation}")
        if row.get("root_id") != menu.get("root_id"):
            errors.append(f"action_root_mismatch:{row.get('action_id')}")
    return {"valid": not errors, "errors": errors}
