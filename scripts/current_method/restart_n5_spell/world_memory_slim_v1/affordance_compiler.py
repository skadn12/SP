"""Compile authoritative World Memory into a compact, executable action menu.

The compiler owns identities, references, current state, and transition shape.
The Planner may select an ``action_id`` but cannot author any of those fields.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes


AFFORDANCE_SCHEMA_VERSION = "delta_affordance_v1"
AFFORDANCE_COMPILER_VERSION = "delta_affordance_compiler_v1"
COMPILED_DELTA_SCHEMA_VERSION = "compiled_graph_delta_v1"

CREATE_OPERATIONS = frozenset({
    "add_person", "add_shared_task", "add_information_asymmetry",
    "add_third_party_report", "open_thread", "create_conflict",
})
UPDATE_OPERATIONS = frozenset({
    "update_state", "change_trust", "change_authority", "resolve_conflict",
    "close_thread",
})
TRANSITION_OPERATIONS = frozenset({"add_role_handoff"})
OPERATION_CATEGORIES = {
    **{operation: "create" for operation in CREATE_OPERATIONS},
    **{operation: "update" for operation in UPDATE_OPERATIONS},
    **{operation: "transition" for operation in TRANSITION_OPERATIONS},
}
AUTHORITATIVE_SEMANTIC_STATUSES = frozenset({
    "source_grounded", "semantically_verified",
})

_ACTION_FIELDS = frozenset({
    "action_id", "root_id", "operation", "operation_class",
    "target_memory_id", "event_identity_key", "subject_ids", "current_state",
    "allowed_post_state_directions", "default_post_state",
    "related_memory_ids", "eligible_participants", "source_turn_ids", "reason",
    "thread_id", "mutually_exclusive_group", "derived_from_completed",
    "transition_participants",
})

DELTA_AFFORDANCE_SCHEMA: dict[str, Any] = {
    "schema_version": AFFORDANCE_SCHEMA_VERSION,
    "top_level_fields": [
        "affordance_schema_version", "root_id", "memory_version", "scene_budget",
        "memory_threads", "actions", "compiler_provenance",
    ],
    "action_fields": sorted(_ACTION_FIELDS),
    "operation_categories": {
        "create": sorted(CREATE_OPERATIONS),
        "update": sorted(UPDATE_OPERATIONS),
        "transition": sorted(TRANSITION_OPERATIONS),
    },
    "authoritative_semantic_statuses": sorted(AUTHORITATIVE_SEMANTIC_STATUSES),
}

COMPILED_DELTA_SCHEMA: dict[str, Any] = {
    "schema_version": COMPILED_DELTA_SCHEMA_VERSION,
    "program_owned_fields": [
        "delta_id", "proposed_delta_id", "scene_key", "action_id", "root_id",
        "operation", "operation_class", "target_memory_id",
        "event_identity_key", "subject_ids", "pre_state",
        "intended_post_state", "predecessor_memory_id", "related_memory_ids",
        "transition_participants", "rationale", "provenance",
    ],
    "generated_by": AFFORDANCE_COMPILER_VERSION,
}


def _authoritative(record: Mapping[str, Any]) -> bool:
    return (
        record.get("semantic_status") in AUTHORITATIVE_SEMANTIC_STATUSES
        and record.get("status") not in {"superseded", "disputed"}
    )


def _compact_text(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"


def _stable_suffix(*values: str) -> str:
    return sha256_bytes("\x1f".join(values).encode())[:16]


def _record_collections(memory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        record
        for field in ("event_history", "current_states", "open_threads",
                      "completed_threads")
        for record in (memory.get(field) or [])
        if isinstance(record, Mapping)
    ]


def _structured_update_operation(record: Mapping[str, Any]) -> str:
    """Classify a structured record without inspecting natural-language values."""
    record_type = str(record.get("record_type") or "").casefold()
    if record_type in {"trust", "trust_state", "change_trust"}:
        return "change_trust"
    if record_type in {"authority", "authority_state", "change_authority"}:
        return "change_authority"
    if record_type in {"conflict", "conflict_state", "create_conflict"}:
        return "resolve_conflict"
    return "update_state"


def _directions_for(operation: str) -> list[str]:
    return {
        "change_trust": ["trust_reassessed"],
        "change_authority": ["authority_rebalanced"],
        "resolve_conflict": ["conflict_resolved"],
        "close_thread": ["thread_closed"],
        "add_role_handoff": ["responsibility_transferred"],
        "update_state": ["state_reflected", "future_planning"],
    }.get(operation, ["state_reflected"])


def _action(
    *, root_id: str, operation: str, target_memory_id: str | None,
    event_identity_key: str, subject_ids: Sequence[str],
    current_state: str | None, allowed_directions: Sequence[str],
    related_memory_ids: Sequence[str], eligible_participants: Sequence[str],
    source_turn_ids: Sequence[str], reason: str, thread_id: str,
    mutually_exclusive_group: str, derived_from_completed: bool = False,
    transition_participants: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "action_id": "",  # assigned after deterministic ordering
        "root_id": root_id,
        "operation": operation,
        "operation_class": OPERATION_CATEGORIES[operation],
        "target_memory_id": target_memory_id,
        "event_identity_key": event_identity_key,
        "subject_ids": list(dict.fromkeys(str(value) for value in subject_ids)),
        "current_state": current_state,
        "allowed_post_state_directions": list(allowed_directions),
        "default_post_state": str(allowed_directions[0]),
        "related_memory_ids": list(dict.fromkeys(
            str(value) for value in related_memory_ids)),
        "eligible_participants": list(dict.fromkeys(
            str(value) for value in eligible_participants)),
        "source_turn_ids": list(dict.fromkeys(
            str(value) for value in source_turn_ids)),
        "reason": reason,
        "thread_id": thread_id,
        "mutually_exclusive_group": mutually_exclusive_group,
        "derived_from_completed": bool(derived_from_completed),
        "transition_participants": (
            dict(transition_participants) if transition_participants else None),
    }


def compile_delta_affordances(
    memory: Mapping[str, Any], *, root_identity: str,
    mutation_requests: Sequence[Mapping[str, Any]] = (),
    maximum_source_threads: int = 6, maximum_actions: int = 10,
) -> dict[str, Any]:
    """Return a deterministic menu; never upgrade pending/model claims to actions."""
    root_id = str(memory.get("root_id") or "")
    if not root_id or root_id != root_identity:
        raise ValueError("affordance_root_identity_mismatch")
    if maximum_source_threads < 1 or maximum_actions < 1:
        raise ValueError("affordance_budget_invalid")

    entities = [row for row in memory.get("entities") or []
                if isinstance(row, Mapping) and _authoritative(row)]
    entity_name = {str(row["entity_id"]): str(row["display_name"])
                   for row in entities}
    authoritative_records = [row for row in _record_collections(memory)
                             if _authoritative(row)]
    record_by_id = {str(row.get("memory_id")): row
                    for row in authoritative_records if row.get("memory_id")}

    # Recent authoritative source records form a bounded, source-linked retrieval view.
    source_records = [
        row for row in (memory.get("event_history") or [])
        if isinstance(row, Mapping) and _authoritative(row)
        and row.get("source_turn_ids") and row.get("memory_id")
    ][-maximum_source_threads:]
    threads: list[dict[str, Any]] = []
    thread_by_memory: dict[str, str] = {}
    for index, record in enumerate(source_records, 1):
        memory_id = str(record["memory_id"])
        thread_id = f"thread_{index:03d}"
        thread_by_memory[memory_id] = thread_id
        threads.append({
            "thread_id": thread_id,
            "anchor_memory_id": memory_id,
            "related_memory_ids": [memory_id],
            "source_turn_ids": list(record["source_turn_ids"]),
            "participant_ids": [value for value in record.get("subject_ids") or []
                                if value in entity_name],
            "summary": _compact_text(record.get("value")),
        })

    candidates: list[dict[str, Any]] = []
    for record in source_records:
        memory_id = str(record["memory_id"])
        subjects = [str(value) for value in record.get("subject_ids") or []
                    if str(value) in entity_name]
        if not subjects:
            continue
        participants = [entity_name[value] for value in subjects]
        thread_id = thread_by_memory[memory_id]
        suffix = _stable_suffix(root_id, memory_id, "open_thread")
        candidates.append(_action(
            root_id=root_id,
            operation="open_thread",
            target_memory_id=None,
            event_identity_key=f"{root_id}:affordance:open_thread:{suffix}",
            subject_ids=subjects,
            current_state=None,
            allowed_directions=["reflection_open", "follow_up_open"],
            related_memory_ids=[memory_id],
            eligible_participants=participants,
            source_turn_ids=record["source_turn_ids"],
            reason=(f"Follow up source memory {memory_id}: "
                    f"{_compact_text(record.get('value'), 88)}"),
            thread_id=thread_id,
            mutually_exclusive_group=f"source:{memory_id}:follow_up",
        ))

    # A shared-task affordance is structural only: it requires two grounded entities
    # and two linked source records; no dialogue semantics are inferred.
    for left, right in zip(source_records, source_records[1:]):
        left_subjects = [str(value) for value in left.get("subject_ids") or []
                         if str(value) in entity_name]
        right_subjects = [str(value) for value in right.get("subject_ids") or []
                          if str(value) in entity_name]
        subjects = list(dict.fromkeys(left_subjects + right_subjects))
        if len(subjects) < 2:
            continue
        related = [str(left["memory_id"]), str(right["memory_id"])]
        suffix = _stable_suffix(root_id, *related, "shared_task")
        candidates.append(_action(
            root_id=root_id,
            operation="add_shared_task",
            target_memory_id=None,
            event_identity_key=f"{root_id}:affordance:shared_task:{suffix}",
            subject_ids=subjects,
            current_state=None,
            allowed_directions=["shared_task_open", "coordination_planned"],
            related_memory_ids=related,
            eligible_participants=[entity_name[value] for value in subjects],
            source_turn_ids=[*(left.get("source_turn_ids") or []),
                             *(right.get("source_turn_ids") or [])],
            reason="Create a new shared task connected to two grounded participants.",
            thread_id=thread_by_memory[related[0]],
            mutually_exclusive_group=f"source_pair:{related[0]}:{related[1]}",
        ))

    # Update actions are emitted only for authoritative current records. Trust and
    # authority actions therefore cannot appear merely because dialogue mentions them.
    for record in memory.get("current_states") or []:
        if not isinstance(record, Mapping) or not _authoritative(record):
            continue
        memory_id = str(record.get("memory_id") or "")
        event_key = str(record.get("event_identity_key") or "")
        subjects = [str(value) for value in record.get("subject_ids") or []
                    if str(value) in entity_name]
        if not memory_id or not event_key or not subjects:
            continue
        operation = _structured_update_operation(record)
        source_ids = list(record.get("source_turn_ids") or [])
        thread_id = thread_by_memory.get(memory_id, f"state_{_stable_suffix(memory_id)[:8]}")
        candidates.append(_action(
            root_id=root_id,
            operation=operation,
            target_memory_id=memory_id,
            event_identity_key=event_key,
            subject_ids=subjects,
            current_state=str(record.get("value") or ""),
            allowed_directions=_directions_for(operation),
            related_memory_ids=[memory_id],
            eligible_participants=[entity_name[value] for value in subjects],
            source_turn_ids=source_ids,
            reason=f"Update authoritative {record.get('record_type')} state {memory_id}.",
            thread_id=thread_id,
            mutually_exclusive_group=f"target:{memory_id}",
        ))

    for record in memory.get("open_threads") or []:
        if not isinstance(record, Mapping) or not _authoritative(record):
            continue
        memory_id = str(record.get("memory_id") or "")
        subjects = [str(value) for value in record.get("subject_ids") or []
                    if str(value) in entity_name]
        if not memory_id or not subjects:
            continue
        candidates.append(_action(
            root_id=root_id, operation="close_thread", target_memory_id=memory_id,
            event_identity_key=str(record["event_identity_key"]),
            subject_ids=subjects, current_state=str(record.get("value") or ""),
            allowed_directions=_directions_for("close_thread"),
            related_memory_ids=[memory_id],
            eligible_participants=[entity_name[value] for value in subjects],
            source_turn_ids=record.get("source_turn_ids") or [],
            reason=f"Close existing authoritative open thread {memory_id}.",
            thread_id=thread_by_memory.get(
                memory_id, f"thread_{_stable_suffix(memory_id)[:8]}"),
            mutually_exclusive_group=f"target:{memory_id}",
        ))

    # Completed facts may seed a distinct derived reflection; their base identity is
    # never reused or marked complete again.
    seen_completed: set[str] = set()
    completed = [*(memory.get("completed_threads") or []),
                 *(row for row in (memory.get("event_history") or [])
                   if isinstance(row, Mapping) and row.get("status") == "completed")]
    for record in completed:
        if not isinstance(record, Mapping) or not _authoritative(record):
            continue
        memory_id = str(record.get("memory_id") or "")
        event_key = str(record.get("event_identity_key") or "")
        if not memory_id or event_key in seen_completed:
            continue
        seen_completed.add(event_key)
        subjects = [str(value) for value in record.get("subject_ids") or []
                    if str(value) in entity_name]
        if not subjects:
            continue
        suffix = _stable_suffix(root_id, memory_id, "derived_reflection")
        candidates.append(_action(
            root_id=root_id, operation="open_thread", target_memory_id=None,
            event_identity_key=f"{root_id}:derived_reflection:{suffix}",
            subject_ids=subjects, current_state=None,
            allowed_directions=["responsibility_reflected", "future_planning"],
            related_memory_ids=[memory_id],
            eligible_participants=[entity_name[value] for value in subjects],
            source_turn_ids=record.get("source_turn_ids") or [],
            reason=f"Derive a new reflection from completed fact {memory_id}.",
            thread_id=thread_by_memory.get(
                memory_id, f"completed_{_stable_suffix(memory_id)[:8]}"),
            mutually_exclusive_group=f"completed:{event_key}:derived",
            derived_from_completed=True,
        ))

    task_types = {"add_shared_task", "shared_task", "task", "role", "responsibility"}
    for record in memory.get("current_states") or []:
        if (not isinstance(record, Mapping) or not _authoritative(record)
                or str(record.get("record_type") or "").casefold() not in task_types):
            continue
        memory_id = str(record.get("memory_id") or "")
        source_ids = [str(value) for value in record.get("subject_ids") or []
                      if str(value) in entity_name]
        if not source_ids:
            continue
        target = next((value for value in entity_name if value not in source_ids), None)
        if target is None and len(source_ids) > 1:
            target = source_ids[1]
        if target is None:
            continue
        source = source_ids[0]
        subjects = list(dict.fromkeys([source, target]))
        candidates.append(_action(
            root_id=root_id, operation="add_role_handoff",
            target_memory_id=memory_id,
            event_identity_key=str(record.get("event_identity_key") or ""),
            subject_ids=subjects, current_state=str(record.get("value") or ""),
            allowed_directions=_directions_for("add_role_handoff"),
            related_memory_ids=[memory_id],
            eligible_participants=[entity_name[value] for value in subjects],
            source_turn_ids=record.get("source_turn_ids") or [],
            reason=f"Transfer an existing task or responsibility from {memory_id}.",
            thread_id=thread_by_memory.get(
                memory_id, f"task_{_stable_suffix(memory_id)[:8]}"),
            mutually_exclusive_group=f"target:{memory_id}:handoff",
            transition_participants={"source_entity_id": source,
                                     "target_entity_id": target},
        ))

    # Mutation requests may prioritize an already-safe action, never manufacture one.
    requested_ops = {
        str(row.get("requested_graph_mutation") or "")
        for row in mutation_requests if isinstance(row, Mapping)
        and row.get("status") in {None, "pending", "accepted"}
    }
    candidates.sort(key=lambda row: (
        0 if row["operation"] in requested_ops else 1,
        {"update": 0, "transition": 1, "create": 2}[row["operation_class"]],
        row["operation"], row["event_identity_key"],
    ))
    actions = candidates[:maximum_actions]
    for index, action in enumerate(actions, 1):
        action["action_id"] = f"A{index:04d}"

    referenced_thread_ids = {str(row["thread_id"]) for row in actions}
    # Update/transition states can introduce a thread not present in the bounded source view.
    known_threads = {str(row["thread_id"]) for row in threads}
    for action in actions:
        thread_id = str(action["thread_id"])
        if thread_id not in known_threads:
            threads.append({
                "thread_id": thread_id,
                "anchor_memory_id": action["target_memory_id"],
                "related_memory_ids": list(action["related_memory_ids"]),
                "source_turn_ids": list(action["source_turn_ids"]),
                "participant_ids": list(action["subject_ids"]),
                "summary": action["reason"],
            })
            known_threads.add(thread_id)
    threads = [row for row in threads if row["thread_id"] in referenced_thread_ids]

    menu = {
        "affordance_schema_version": AFFORDANCE_SCHEMA_VERSION,
        "root_id": root_id,
        "memory_version": int(memory.get("memory_version") or 0),
        "scene_budget": {"target": 3, "minimum": 2, "maximum": 4},
        "memory_threads": threads,
        "actions": actions,
        "compiler_provenance": {
            "generated_by": AFFORDANCE_COMPILER_VERSION,
            "authoritative_semantic_statuses": sorted(
                AUTHORITATIVE_SEMANTIC_STATUSES),
            "pending_records_used": 0,
            "mutation_request_count": len(mutation_requests),
        },
    }
    validation = validate_affordance_menu(menu, memory=memory,
                                          expected_root_id=root_identity)
    if not validation["valid"]:
        raise RuntimeError("compiled_affordance_menu_invalid:" +
                           ",".join(validation["errors"]))
    return menu


def validate_affordance_menu(
    menu: Any, *, memory: Mapping[str, Any] | None = None,
    expected_root_id: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "affordance_schema_version", "root_id", "memory_version", "scene_budget",
        "memory_threads", "actions", "compiler_provenance",
    }
    if not isinstance(menu, Mapping) or set(menu) != required:
        return {"valid": False, "errors": ["affordance_top_level_schema"]}
    if menu.get("affordance_schema_version") != AFFORDANCE_SCHEMA_VERSION:
        errors.append("affordance_schema_version")
    root_id = str(menu.get("root_id") or "")
    if not root_id or (expected_root_id is not None and root_id != expected_root_id):
        errors.append("affordance_root_identity")
    actions = menu.get("actions")
    if not isinstance(actions, list):
        return {"valid": False, "errors": errors + ["affordance_actions"]}
    action_ids: list[str] = []
    authoritative_ids = {
        str(row.get("memory_id")) for row in _record_collections(memory or {})
        if _authoritative(row) and row.get("memory_id")
    }
    for index, action in enumerate(actions):
        prefix = f"action:{index}"
        if not isinstance(action, Mapping) or set(action) != _ACTION_FIELDS:
            errors.append(f"{prefix}:schema")
            continue
        action_id = str(action.get("action_id") or "")
        action_ids.append(action_id)
        operation = str(action.get("operation") or "")
        category = OPERATION_CATEGORIES.get(operation)
        if not action_id or category != action.get("operation_class"):
            errors.append(f"{prefix}:identity_or_category")
        if action.get("root_id") != root_id:
            errors.append(f"{prefix}:mixed_root")
        event_key = str(action.get("event_identity_key") or "")
        if not event_key.startswith(root_id + ":"):
            errors.append(f"{prefix}:event_root")
        if (not isinstance(action.get("subject_ids"), list)
                or not action["subject_ids"]
                or not isinstance(action.get("eligible_participants"), list)
                or not action["eligible_participants"]):
            errors.append(f"{prefix}:participants")
        directions = action.get("allowed_post_state_directions")
        if (not isinstance(directions, list) or not directions
                or action.get("default_post_state") not in directions):
            errors.append(f"{prefix}:directions")
        target = action.get("target_memory_id")
        if category == "create":
            if target is not None or action.get("current_state") is not None:
                errors.append(f"{prefix}:create_has_existing_state")
        elif target not in authoritative_ids and memory is not None:
            errors.append(f"{prefix}:target_not_authoritative")
    if len(action_ids) != len(set(action_ids)):
        errors.append("duplicate_action_id")
    return {"valid": not errors, "errors": errors}


def _compiled_delta(
    action: Mapping[str, Any], *, root_id: str, scene_key: str,
    delta_index: int, menu_sha256: str,
) -> dict[str, Any]:
    category = str(action["operation_class"])
    pre_state = (str(action["current_state"])
                 if category in {"update", "transition"} else None)
    delta_id = f"D{delta_index:04d}"
    return {
        "delta_id": delta_id,
        "proposed_delta_id": f"{root_id}:{delta_id}",
        "scene_key": scene_key,
        "action_id": str(action["action_id"]),
        "root_id": root_id,
        "operation": str(action["operation"]),
        "operation_class": category,
        "target_memory_id": action["target_memory_id"],
        "event_identity_key": str(action["event_identity_key"]),
        "subject_ids": list(action["subject_ids"]),
        "pre_state": pre_state,
        "intended_post_state": str(action["default_post_state"]),
        "predecessor_memory_id": (
            str(action["related_memory_ids"][0])
            if action["related_memory_ids"] else None),
        "related_memory_ids": list(action["related_memory_ids"]),
        "transition_participants": copy.deepcopy(action["transition_participants"]),
        "rationale": str(action["reason"]),
        "provenance": {
            "generated_by": AFFORDANCE_COMPILER_VERSION,
            "affordance_schema_version": AFFORDANCE_SCHEMA_VERSION,
            "action_menu_sha256": menu_sha256,
        },
    }


def compile_selected_actions(
    plan: Mapping[str, Any], action_menu: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign scene/delta IDs and expand model selections deterministically."""
    from .schemas import validate_slim_planner_output  # avoid module cycle

    validation = validate_slim_planner_output(plan, action_menu=action_menu)
    if not validation["valid"]:
        raise ValueError("slim_planner_v2_invalid:" + ",".join(validation["errors"]))
    root_id = str(action_menu["root_id"])
    action_by_id = {str(row["action_id"]): row for row in action_menu["actions"]}
    menu_sha = sha256_bytes(canonical_json(action_menu).encode())
    result = copy.deepcopy(dict(plan))
    delta_index = 0
    compiled_scenes: list[dict[str, Any]] = []
    for scene_index, selected_scene in enumerate(result["scene_goals"], 1):
        scene_key = f"scene_{scene_index:02d}"
        selected_ids = list(selected_scene["selected_action_ids"])
        actions = [action_by_id[value] for value in selected_ids]
        deltas = []
        for action in actions:
            delta_index += 1
            deltas.append(_compiled_delta(
                action, root_id=root_id, scene_key=scene_key,
                delta_index=delta_index, menu_sha256=menu_sha))
        compiled_scenes.append({
            "scene_key": scene_key,
            "goal": str(selected_scene["goal"]),
            "participants": list(selected_scene["participants"]),
            "selected_action_ids": selected_ids,
            "related_memory_ids": list(dict.fromkeys(
                memory_id for action in actions
                for memory_id in action["related_memory_ids"])),
            "proposed_graph_deltas": deltas,
        })
    result["scene_goals"] = compiled_scenes
    result["compiled_by"] = AFFORDANCE_COMPILER_VERSION
    result["action_menu_sha256"] = menu_sha
    return result


def action_menu_parent_turn_ids(action_menu: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(turn_id) for action in action_menu.get("actions") or []
        for turn_id in action.get("source_turn_ids") or []))


def compact_action_menu(action_menu: Mapping[str, Any]) -> dict[str, Any]:
    """Planner-facing projection: enough to choose, no complete World Memory dump."""
    return {
        "affordance_schema_version": action_menu["affordance_schema_version"],
        "root_id": action_menu["root_id"],
        "scene_budget": dict(action_menu["scene_budget"]),
        "memory_threads": [{
            "thread_id": row["thread_id"],
            "summary": row["summary"],
        } for row in action_menu["memory_threads"]],
        "actions": [{
            "action_id": row["action_id"],
            "operation": row["operation"],
            "operation_class": row["operation_class"],
            "thread_id": row["thread_id"],
            "eligible_participants": list(row["eligible_participants"]),
            "allowed_direction": row["default_post_state"],
            "reason": row["reason"],
        } for row in action_menu["actions"]],
    }
