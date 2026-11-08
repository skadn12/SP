"""Canonical, non-mutating validation for Stage3 semantic graph deltas.

The validator deliberately does not coerce, fill, normalize, or repair input.  A
challenger response either satisfies the ``semantic_graph_v2`` contract or
returns structured issues that can be sent back to the challenger for retry.
"""

from __future__ import annotations

import json
import re
from typing import Any


SCHEMA_VERSION = "semantic_graph_v2"

REQUIRED_GRAPH_FIELDS = (
    "schema_version",
    "graph_id",
    "parent_graph_id",
    "events",
)
REQUIRED_EVENT_FIELDS = (
    "event_id",
    "entity",
    "slot",
    "value",
    "status",
    "time_index",
)
OPTIONAL_EVENT_FIELDS = (
    "event_type",
    "node_type",
    "constructor_role",
    "claim_edge_role",
    "qa_eligible",
    "answer_bearing",
    "answer_role",
    "dependency_edge_role",
    "previous_value",
    "depends_on",
    "scope",
    "set_membership",
    "owner",
    "resource",
    "observer",
    "action",
    "condition",
    "access_class",
    "global_time",
    "local_time",
    "local_time_index",
    "global_time_index",
    "session_index",
    "evidence_turn_ids",
    "provenance",
    "required_for_dialogue",
    "depends_on_parent_event_ids",
    "parent_topic_anchor_event_ids",
    "parent_relation",
)

SUPPORTED_STATUSES = frozenset(
    {
        "active",
        "belief",
        "canceled",
        "delayed_not_active",
        "expired",
        "false",
        "outdated",
        "report",
        "restored",
        "scope_limited",
    }
)
ACTIVE_STATUSES = frozenset({"active", "restored"})
NON_ACTIVE_STATUSES = SUPPORTED_STATUSES - ACTIVE_STATUSES

OPERATION_ALIASES = {
    "temporal": "temporal_state_update",
    "temporal_state_update": "temporal_state_update",
    "bridge": "bridge",
    "comparison": "comparison",
    "visibility": "visibility",
    "count": "count",
    "cause": "cause_consequence",
    "cause_consequence": "cause_consequence",
    "relation": "relation_lookup",
    "relation_lookup": "relation_lookup",
}

# Formal Stage3 graphs must remain small enough to be realized naturally in a
# dialogue, while carrying enough competing evidence to avoid direct lookup.
# Operation-specific validators below impose the tighter internal shape (for
# example, a 3-4 event temporal chain inside this 6-8 event envelope).
OPERATION_EVENT_COUNT_RANGES = {
    operation: (6, 8) for operation in set(OPERATION_ALIASES.values())
}

_GENERIC_SLOT_RE = re.compile(
    r"^(?:raw qa(?: .*)?|qa(?: .*)?|generic(?: .*)?|fact|state|memory state|answer|value)$",
    re.IGNORECASE,
)
_QA_MARKER_RE = re.compile(r"(?:^|\s)[QA]\s*:", re.IGNORECASE)
_QUESTION_VALUE_RE = re.compile(
    r"^[\s\"']*(?:who|what|when|where|which|how|why)\b",
    re.IGNORECASE,
)
_JSON_SCALAR_TYPES = (str, int, float, bool)


def _issue(code: str, path: str, message: str, **details: Any) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "path": path,
        "message": message,
    }
    if details:
        issue["details"] = details
    return issue


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_json_scalar(value: Any) -> bool:
    return isinstance(value, _JSON_SCALAR_TYPES)


def _normalized_text(value: Any) -> str:
    return re.sub(r"[\W_]+", " ", str(value).casefold(), flags=re.UNICODE).strip()


_PERSON_ATOM_TOKEN = re.compile(
    r"[^\W\d_]+(?:[-'][^\W\d_]+)*\.?",
    re.UNICODE,
)
_PERSON_ATOM_PARTICLES = frozenset(
    {"al", "bin", "da", "de", "del", "der", "di", "du", "la", "le", "van", "von"}
)
_PERSON_ROLE_TOKENS = frozenset(
    {
        "advisor", "adviser", "artist", "assignee", "author", "captain",
        "child", "coach", "colleague", "companion", "contact", "coordinator",
        "curator", "delegate", "director", "discoverer", "employee", "founder",
        "friend", "guest", "host", "leader", "manager", "member", "mentor",
        "name", "observer", "organizer", "owner", "parent", "participant",
        "partner", "person", "recipient", "relative", "resident", "sibling",
        "speaker", "spouse", "teacher", "winner",
    }
)
_NON_PERSON_ROLE_TOKENS = frozenset(
    {
        "address", "age", "amount", "appointment", "approval", "condition",
        "confirmation", "count", "date", "day", "deadline", "destination",
        "duration", "hour", "level", "location", "minute", "month", "number",
        "permission", "quantity", "result", "room", "route", "schedule", "scope",
        "score", "site", "state", "status", "time", "venue", "verification",
        "week", "year",
    }
)


def _is_atomic_person_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    tokens = value.strip().split()
    if not 1 <= len(tokens) <= 5:
        return False
    for token in tokens:
        if _PERSON_ATOM_TOKEN.fullmatch(token) is None:
            return False
        components = re.split(r"[-']", token.rstrip("."))
        if any(
            not component
            or (
                not component[0].isupper()
                and component.casefold() not in _PERSON_ATOM_PARTICLES
            )
            for component in components
        ):
            return False
    return True


def _is_person_role_slot(slot: Any) -> bool:
    normalized = re.sub(r"[\W]+", "_", str(slot or "").casefold()).strip("_")
    if not normalized:
        return False
    tokens = [token for token in normalized.split("_") if token]
    variants = set(tokens)
    variants.update(
        token[:-1] for token in tokens if len(token) > 3 and token.endswith("s")
    )
    if variants.intersection(_NON_PERSON_ROLE_TOKENS):
        return False
    return bool(
        variants.intersection(_PERSON_ROLE_TOKENS)
        or normalized.endswith("_by")
    )


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return _normalized_text(left) == _normalized_text(right)
    return type(left) is type(right) and left == right


def _value_key(value: Any) -> str:
    """Return a comparison key even when an already-invalid value is not JSON serializable."""

    if isinstance(value, str):
        return json.dumps(["string", _normalized_text(value)], ensure_ascii=False)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _validate_string_list(value: Any, path: str, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return [
            _issue(
                f"invalid_{field}",
                path,
                f"{field} must be a JSON array of non-empty strings.",
            )
        ]
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not _is_nonempty_string(item):
            issues.append(
                _issue(
                    f"invalid_{field}_item",
                    item_path,
                    f"Every {field} item must be a non-empty string.",
                )
            )
            continue
        if item in seen:
            issues.append(
                _issue(
                    f"duplicate_{field}_item",
                    item_path,
                    f"{field} must not contain duplicate identifiers.",
                    value=item,
                )
            )
        seen.add(item)
    return issues


def validate_event(event: Any, *, path: str = "$.events[0]") -> list[dict[str, Any]]:
    """Validate one event without changing it."""

    if not isinstance(event, dict):
        return [_issue("invalid_event_type", path, "An event must be a JSON object.")]

    issues: list[dict[str, Any]] = []
    for field in REQUIRED_EVENT_FIELDS:
        if field not in event:
            issues.append(
                _issue(
                    "missing_event_field",
                    f"{path}.{field}",
                    f"Required event field '{field}' is missing.",
                    field=field,
                )
            )

    for field in ("event_id", "entity", "slot", "status"):
        if field in event and not _is_nonempty_string(event[field]):
            issues.append(
                _issue(
                    f"invalid_{field}",
                    f"{path}.{field}",
                    f"{field} must be a non-empty string.",
                )
            )

    slot = event.get("slot")
    if _is_nonempty_string(slot) and _GENERIC_SLOT_RE.fullmatch(_normalized_text(slot)):
        issues.append(
            _issue(
                "generic_or_raw_qa_slot",
                f"{path}.slot",
                "slot must name a concrete semantic relation; raw_qa and generic slots are forbidden.",
                slot=slot,
            )
        )

    if "value" in event:
        value = event["value"]
        if not _is_json_scalar(value) or (isinstance(value, str) and not value.strip()):
            issues.append(
                _issue(
                    "invalid_value",
                    f"{path}.value",
                    "value must be a non-empty string or another JSON scalar.",
                )
            )
        elif isinstance(value, str) and (
            "?" in value or _QA_MARKER_RE.search(value) or _QUESTION_VALUE_RE.match(value)
        ):
            issues.append(
                _issue(
                    "question_or_qa_wrapper_value",
                    f"{path}.value",
                    "value must be an answer-like fact, not a question or Q:/A: wrapper.",
                    value_preview=value[:160],
                )
            )

    status = event.get("status")
    if _is_nonempty_string(status) and status not in SUPPORTED_STATUSES:
        issues.append(
            _issue(
                "unsupported_status",
                f"{path}.status",
                "status is not part of the semantic_graph_v2 status vocabulary.",
                status=status,
                supported=sorted(SUPPORTED_STATUSES),
            )
        )

    if "time_index" in event:
        time_index = event["time_index"]
        if isinstance(time_index, bool) or not isinstance(time_index, int) or time_index < 0:
            issues.append(
                _issue(
                    "invalid_time_index",
                    f"{path}.time_index",
                    "time_index must be a non-negative integer.",
                )
            )

    for time_field in ("local_time_index", "global_time_index", "session_index"):
        if time_field not in event:
            continue
        time_value = event[time_field]
        if (
            isinstance(time_value, bool)
            or not isinstance(time_value, int)
            or time_value < 0
        ):
            issues.append(
                _issue(
                    f"invalid_{time_field}",
                    f"{path}.{time_field}",
                    f"{time_field} must be a non-negative integer when present.",
                )
            )

    if "previous_value" in event:
        previous_value = event["previous_value"]
        if previous_value is not None and not _is_json_scalar(previous_value):
            issues.append(
                _issue(
                    "invalid_previous_value",
                    f"{path}.previous_value",
                    "previous_value must be null or a JSON scalar.",
                )
            )

    if "depends_on" in event:
        issues.extend(_validate_string_list(event["depends_on"], f"{path}.depends_on", "depends_on"))

    for parent_ref_field in (
        "depends_on_parent_event_ids",
        "parent_topic_anchor_event_ids",
    ):
        if parent_ref_field in event:
            issues.extend(
                _validate_string_list(
                    event[parent_ref_field],
                    f"{path}.{parent_ref_field}",
                    parent_ref_field,
                )
            )

    for role_field in (
        "constructor_role",
        "claim_edge_role",
        "answer_role",
        "dependency_edge_role",
        "parent_relation",
    ):
        if role_field in event and not _is_nonempty_string(event[role_field]):
            issues.append(
                _issue(
                    f"invalid_{role_field}",
                    f"{path}.{role_field}",
                    f"{role_field} must be a non-empty string when present.",
                )
            )

    for boolean_field in ("qa_eligible", "answer_bearing"):
        if boolean_field in event and not isinstance(event[boolean_field], bool):
            issues.append(
                _issue(
                    f"invalid_{boolean_field}",
                    f"{path}.{boolean_field}",
                    f"{boolean_field} must be a JSON boolean when present.",
                )
            )

    if "scope" in event:
        scope = event["scope"]
        if not isinstance(scope, dict) or not scope:
            issues.append(
                _issue(
                    "invalid_scope",
                    f"{path}.scope",
                    "scope must be a non-empty JSON object.",
                )
            )

    if "set_membership" in event:
        membership = event["set_membership"]
        if not isinstance(membership, dict):
            issues.append(
                _issue(
                    "invalid_set_membership",
                    f"{path}.set_membership",
                    "set_membership must be a JSON object with set_id and included.",
                )
            )
        else:
            if not _is_nonempty_string(membership.get("set_id")):
                issues.append(
                    _issue(
                        "invalid_set_membership_set_id",
                        f"{path}.set_membership.set_id",
                        "set_membership.set_id must be a non-empty string.",
                    )
                )
            if not isinstance(membership.get("included"), bool):
                issues.append(
                    _issue(
                        "invalid_set_membership_included",
                        f"{path}.set_membership.included",
                        "set_membership.included must be a boolean.",
                    )
                )

    if "evidence_turn_ids" in event:
        issues.extend(
            _validate_string_list(
                event["evidence_turn_ids"],
                f"{path}.evidence_turn_ids",
                "evidence_turn_ids",
            )
        )

    if "provenance" in event and not isinstance(event["provenance"], dict):
        issues.append(
            _issue(
                "invalid_provenance",
                f"{path}.provenance",
                "provenance must be a JSON object.",
            )
        )

    if "required_for_dialogue" in event and not isinstance(event["required_for_dialogue"], bool):
        issues.append(
            _issue(
                "invalid_required_for_dialogue",
                f"{path}.required_for_dialogue",
                "required_for_dialogue must be a JSON boolean when present.",
            )
        )

    return issues


def _operation_name(graph_delta: dict[str, Any], operation: str | None) -> tuple[str | None, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    embedded = graph_delta.get("operation")
    if embedded is not None and not _is_nonempty_string(embedded):
        issues.append(
            _issue("invalid_operation", "$.operation", "operation must be a non-empty string when present.")
        )
        embedded = None

    requested = operation if operation is not None else embedded
    if operation is not None and embedded is not None:
        requested_name = OPERATION_ALIASES.get(str(operation).strip())
        embedded_name = OPERATION_ALIASES.get(str(embedded).strip())
        if requested_name != embedded_name:
            issues.append(
                _issue(
                    "operation_mismatch",
                    "$.operation",
                    "The embedded operation does not match the requested operation.",
                    embedded=embedded,
                    requested=operation,
                )
            )

    if requested is None:
        return None, issues
    canonical = OPERATION_ALIASES.get(str(requested).strip())
    if canonical is None:
        issues.append(
            _issue(
                "unsupported_operation",
                "$.operation",
                "operation is not supported by semantic_graph_v2.",
                operation=requested,
                supported=sorted(set(OPERATION_ALIASES.values())),
            )
        )
    return canonical, issues


def _event_key(event: dict[str, Any]) -> tuple[str, str]:
    return (_normalized_text(event.get("entity", "")), _normalized_text(event.get("slot", "")))


def _valid_events(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _has_reasoning_target(reasoning_target: dict[str, Any] | None) -> bool:
    """Return whether target-aware graph checks were explicitly requested."""

    return isinstance(reasoning_target, dict)


def _events_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(event["event_id"]): event
        for event in events
        if _is_nonempty_string(event.get("event_id"))
    }


def _dependency_ancestors(
    event_id: str,
    by_id: dict[str, dict[str, Any]],
) -> set[str]:
    """Return the transitive dependency closure without looping on invalid cycles."""

    ancestors: set[str] = set()
    pending = [event_id]
    while pending:
        current_id = pending.pop()
        current = by_id.get(current_id)
        if current is None or not isinstance(current.get("depends_on"), list):
            continue
        for dependency in current["depends_on"]:
            if (
                _is_nonempty_string(dependency)
                and dependency in by_id
                and dependency != event_id
                and dependency not in ancestors
            ):
                ancestors.add(dependency)
                pending.append(dependency)
    return ancestors


_TEMPORAL_HARD_NEGATIVE_STATUSES = {
    "same_slot_canceled_value": frozenset({"canceled"}),
    "same_slot_outdated_value": frozenset({"outdated"}),
    "same_slot_report_or_belief_value": frozenset({"report", "belief"}),
}


def _temporal_forbid_max_time_target(
    reasoning_target: dict[str, Any] | None,
) -> tuple[bool, str, frozenset[str]]:
    """Return the explicit target-aware temporal validity layout.

    Legacy graphs do not carry a reasoning target and therefore retain the
    historical latest-active contract.  The anti-shortcut layout is enabled
    only by the explicit boolean plus a recognized hard-negative strategy;
    callers cannot accidentally opt in through natural-language metadata.
    """

    if not isinstance(reasoning_target, dict) or reasoning_target.get(
        "forbid_max_time_shortcut"
    ) is not True:
        return False, "", frozenset()
    strategy = str(reasoning_target.get("hard_negative_strategy") or "").strip()
    return True, strategy, _TEMPORAL_HARD_NEGATIVE_STATUSES.get(
        strategy, frozenset()
    )


def _temporal_issues(
    events: list[dict[str, Any]],
    reasoning_target: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        if _is_nonempty_string(event.get("entity")) and _is_nonempty_string(event.get("slot")):
            grouped.setdefault(_event_key(event), []).append(event)

    candidate_groups: list[tuple[tuple[str, str], list[dict[str, Any]]]] = []
    for key, group in grouped.items():
        usable = [
            event
            for event in group
            if isinstance(event.get("time_index"), int)
            and not isinstance(event.get("time_index"), bool)
            and event.get("status") in SUPPORTED_STATUSES
            and _is_json_scalar(event.get("value"))
        ]
        if len(usable) >= 3:
            candidate_groups.append((key, usable))

    if not candidate_groups:
        return [
            _issue(
                "temporal_missing_update_chain",
                "$.events",
                "temporal requires exactly one 3-4 event update chain for one entity/slot.",
            )
        ]
    if len(candidate_groups) != 1:
        return [
            _issue(
                "temporal_multiple_update_chains",
                "$.events",
                "temporal requires exactly one answer-bearing entity/slot update chain.",
                candidate_chains=[
                    {
                        "entity": entity,
                        "slot": slot,
                        "event_ids": [str(event.get("event_id") or "") for event in group],
                    }
                    for (entity, slot), group in candidate_groups
                ],
            )
        ]

    (entity, slot), usable = candidate_groups[0]
    usable.sort(key=lambda event: (event["time_index"], str(event.get("event_id") or "")))
    chain_issues: list[dict[str, Any]] = []
    if not 3 <= len(usable) <= 4:
        chain_issues.append(
            _issue(
                "temporal_chain_length_out_of_range",
                "$.events",
                "The single temporal update chain must contain exactly 3 or 4 events.",
                entity=entity,
                slot=slot,
                event_count=len(usable),
                minimum=3,
                maximum=4,
            )
        )

    value_to_event_ids: dict[str, list[str]] = {}
    for event in usable:
        value_to_event_ids.setdefault(_value_key(event.get("value")), []).append(
            str(event.get("event_id") or "")
        )
    duplicate_value_event_ids = [
        event_ids for event_ids in value_to_event_ids.values() if len(event_ids) > 1
    ]
    if duplicate_value_event_ids:
        chain_issues.append(
            _issue(
                "temporal_duplicate_update_value",
                "$.events",
                "Every value in the temporal update chain must be semantically distinct.",
                entity=entity,
                slot=slot,
                duplicate_value_event_ids=duplicate_value_event_ids,
            )
        )

    broken_links = []
    for previous, current in zip(usable, usable[1:]):
        if "previous_value" not in current or not _same_value(
            current.get("previous_value"), previous.get("value")
        ):
            broken_links.append(
                {
                    "event_id": str(current.get("event_id") or ""),
                    "predecessor_event_id": str(previous.get("event_id") or ""),
                    "expected_previous_value": previous.get("value"),
                    "actual_previous_value": current.get("previous_value"),
                }
            )
    if broken_links:
        chain_issues.append(
            _issue(
                "temporal_broken_previous_value_chain",
                "$.events",
                "Each temporal update after the first must reference the immediately preceding value exactly.",
                entity=entity,
                slot=slot,
                broken_links=broken_links,
            )
        )

    if _has_reasoning_target(reasoning_target):
        broken_dependencies = []
        for previous, current in zip(usable, usable[1:]):
            expected_dependency = str(previous.get("event_id") or "")
            actual_dependencies = current.get("depends_on")
            if actual_dependencies != [expected_dependency]:
                broken_dependencies.append(
                    {
                        "event_id": str(current.get("event_id") or ""),
                        "predecessor_event_id": expected_dependency,
                        "expected_depends_on": [expected_dependency],
                        "actual_depends_on": actual_dependencies,
                    }
                )
        if broken_dependencies:
            chain_issues.append(
                _issue(
                    "temporal_target_broken_dependency_chain",
                    "$.events",
                    "Each target-aware temporal successor must depend only on its immediate predecessor.",
                    entity=entity,
                    slot=slot,
                    broken_links=broken_dependencies,
                )
            )

    time_indexes = [event["time_index"] for event in usable]
    if len(time_indexes) != len(set(time_indexes)):
        chain_issues.append(
            _issue(
                "temporal_latest_time_not_unique",
                "$.events",
                "Events in the temporal update chain must have distinct time_index values.",
                entity=entity,
                slot=slot,
            )
        )

    forbid_max_time, hard_negative_strategy, required_later_statuses = (
        _temporal_forbid_max_time_target(reasoning_target)
    )
    if forbid_max_time and not required_later_statuses:
        chain_issues.append(
            _issue(
                "temporal_unknown_hard_negative_strategy",
                "$.events",
                "A max-time-forbidden temporal target must name a supported hard-negative strategy.",
                hard_negative_strategy=hard_negative_strategy,
                supported_strategies=sorted(_TEMPORAL_HARD_NEGATIVE_STATUSES),
            )
        )

    active_events = [event for event in usable if event.get("status") in ACTIVE_STATUSES]
    if len(active_events) != 1:
        chain_issues.append(
            _issue(
                "temporal_current_state_not_unique",
                "$.events",
                "The temporal chain must contain exactly one active/restored event.",
                entity=entity,
                slot=slot,
                current_event_ids=[str(event.get("event_id") or "") for event in active_events],
            )
        )
    elif not forbid_max_time and active_events[0] is not usable[-1]:
        chain_issues.append(
            _issue(
                "temporal_current_state_not_latest",
                "$.events",
                "The only active/restored event must have the chain's highest time_index.",
                entity=entity,
                slot=slot,
                current_event_id=active_events[0].get("event_id"),
                latest_event_id=usable[-1].get("event_id"),
            )
        )

    if not forbid_max_time and any(
        event.get("status") not in NON_ACTIVE_STATUSES for event in usable[:-1]
    ):
        chain_issues.append(
            _issue(
                "temporal_earlier_update_still_active",
                "$.events",
                "Every temporal update before the highest-time event must be non-active.",
                entity=entity,
                slot=slot,
                offending_event_ids=[
                    str(event.get("event_id") or "")
                    for event in usable[:-1]
                    if event.get("status") not in NON_ACTIVE_STATUSES
                ],
            )
        )
    if forbid_max_time and len(active_events) == 1:
        active_event = active_events[0]
        active_index = usable.index(active_event)
        later_events = usable[active_index + 1 :]
        if not later_events:
            chain_issues.append(
                _issue(
                    "temporal_anti_shortcut_current_is_latest",
                    "$.events",
                    "The active/current gold must precede at least one later invalid same-slot update.",
                    entity=entity,
                    slot=slot,
                    current_event_id=active_event.get("event_id"),
                    current_time_index=active_event.get("time_index"),
                )
            )
        if any(event.get("status") not in NON_ACTIVE_STATUSES for event in later_events):
            chain_issues.append(
                _issue(
                    "temporal_anti_shortcut_later_update_active",
                    "$.events",
                    "Every same-slot update after the current gold must be explicitly non-active.",
                    entity=entity,
                    slot=slot,
                    offending_event_ids=[
                        str(event.get("event_id") or "")
                        for event in later_events
                        if event.get("status") not in NON_ACTIVE_STATUSES
                    ],
                )
            )
        matching_later = [
            event
            for event in later_events
            if event.get("status") in required_later_statuses
        ]
        if required_later_statuses and not matching_later:
            chain_issues.append(
                _issue(
                    "temporal_anti_shortcut_missing_later_strategy_update",
                    "$.events",
                    "A later same-slot invalid update must realize the target hard-negative strategy.",
                    entity=entity,
                    slot=slot,
                    hard_negative_strategy=hard_negative_strategy,
                    required_later_statuses=sorted(required_later_statuses),
                    later_statuses=[str(event.get("status") or "") for event in later_events],
                )
            )
    return chain_issues


def _bridge_issues(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dependencies = {
        event.get("event_id"): [
            dependency
            for dependency in (event.get("depends_on") or [])
            if _is_nonempty_string(dependency)
        ]
        for event in events
        if _is_nonempty_string(event.get("event_id")) and isinstance(event.get("depends_on", []), list)
    }
    if any(len(set(deps)) >= 2 for deps in dependencies.values()):
        return []
    for event_id, direct in dependencies.items():
        if any(dependencies.get(dependency) for dependency in direct if dependency != event_id):
            return []
    return [
        _issue(
            "bridge_missing_two_hop_dependency",
            "$.events",
            "bridge requires two independent dependencies or a dependency path spanning two edges.",
        )
    ]


def _comparison_issues(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in events:
        slot = event.get("slot")
        time_index = event.get("time_index")
        if _is_nonempty_string(slot) and isinstance(time_index, int) and not isinstance(time_index, bool):
            grouped.setdefault((_normalized_text(slot), time_index), []).append(event)
    comparable_pairs: list[list[dict[str, Any]]] = []
    for group in grouped.values():
        active = [event for event in group if event.get("status") in ACTIVE_STATUSES]
        entities = {_normalized_text(event.get("entity", "")) for event in active}
        values = [event.get("value") for event in active]
        # comparison is deliberately restricted to one machine-verifiable
        # integer-difference pair. Rich multi-slot bundles are ambiguous unless
        # their derived answer has an explicit schema, which semantic_graph_v2
        # does not currently provide.
        if (
            len(active) == 2
            and len(entities - {""}) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            and values[0] != values[1]
        ):
            comparable_pairs.append(active)
    if len(comparable_pairs) == 1:
        return []
    return [
        _issue(
            "comparison_missing_unique_numeric_pair",
            "$.events",
            "comparison requires exactly one active pair: two different entities, the same slot and time_index, "
            "and two distinct JSON integer values whose difference is machine derivable.",
            comparable_pair_count=len(comparable_pairs),
        )
    ]


def _visibility_issues(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(isinstance(event.get("scope"), dict) and bool(event["scope"]) for event in events):
        return []
    return [
        _issue(
            "visibility_missing_scope",
            "$.events",
            "visibility requires at least one event with an explicit non-empty scope object.",
        )
    ]


def _count_issues(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        membership = event.get("set_membership")
        if (
            isinstance(membership, dict)
            and _is_nonempty_string(membership.get("set_id"))
            and isinstance(membership.get("included"), bool)
            and _is_nonempty_string(event.get("entity"))
            and _is_nonempty_string(event.get("slot"))
        ):
            grouped.setdefault(_normalized_text(membership["set_id"]), []).append(event)
    for group in grouped.values():
        dimensions = {
            (_normalized_text(event["entity"]), _normalized_text(event["slot"]))
            for event in group
        }
        distinct_values = {_value_key(event.get("value")) for event in group}
        included_count = sum(bool(event["set_membership"]["included"]) for event in group)
        excluded_count = len(group) - included_count
        if (
            6 <= len(group) <= 8
            and len(group) == len(events)
            and len(dimensions) == 1
            and len(distinct_values) == len(group)
            and included_count >= 2
            and excluded_count >= 2
        ):
            return []
    return [
        _issue(
            "count_missing_candidate_set",
            "$.events",
            "count requires all 6-8 events to be distinct members of the same named set, concrete entity, and semantic "
            "slot, with at least two included and two excluded members.",
        )
    ]


def _cause_issues(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        event.get("event_id"): event
        for event in events
        if _is_nonempty_string(event.get("event_id"))
    }
    for consequence in events:
        consequence_time = consequence.get("time_index")
        if isinstance(consequence_time, bool) or not isinstance(consequence_time, int):
            continue
        dependencies = consequence.get("depends_on")
        if not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if not _is_nonempty_string(dependency):
                continue
            cause = by_id.get(dependency)
            if cause is None:
                continue
            cause_time = cause.get("time_index")
            if isinstance(cause_time, int) and not isinstance(cause_time, bool) and cause_time < consequence_time:
                return []
    return [
        _issue(
            "cause_missing_ordered_dependency",
            "$.events",
            "cause requires a later consequence event that explicitly depends on an earlier cause event.",
        )
    ]


def _relation_issues(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_entity_slot: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        entity, slot = _event_key(event)
        if not entity or not slot:
            continue
        by_entity_slot.setdefault((entity, slot), []).append(event)
        by_slot.setdefault(slot, []).append(event)
    same_entity_hard_negative = any(
        len({_value_key(event.get("value")) for event in group}) >= 2
        for group in by_entity_slot.values()
    )
    different_entity_hard_negative = any(
        len({_normalized_text(event.get("entity", "")) for event in group}) >= 2
        for group in by_slot.values()
    )
    if same_entity_hard_negative and different_entity_hard_negative:
        return []
    return [
        _issue(
            "relation_missing_hard_negative_structure",
            "$.events",
            "relation requires distinct values for one entity/slot and the same slot on another entity.",
            same_entity_hard_negative=same_entity_hard_negative,
            different_entity_hard_negative=different_entity_hard_negative,
        )
    ]


def _bridge_target_issues(
    events: list[dict[str, Any]],
    reasoning_target: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if len(events) < 3:
        return [
            _issue(
                "bridge_target_insufficient_events",
                "1events",
                "A target-aware bridge requires E1, E2, and E3 roles.",
            )
        ]
    event_ids = [str(event.get("event_id") or "") for event in events]
    expected_dependencies = [
        [],
        [],
        [event_ids[0], event_ids[1]],
        *([] for _event in events[3:]),
    ]
    actual_dependencies = [
        list(event.get("depends_on") or [])
        if isinstance(event.get("depends_on"), list)
        else []
        for event in events
    ]
    if actual_dependencies != expected_dependencies:
        issues.append(
            _issue(
                "bridge_target_dependency_roles_mismatch",
                "1events",
                "Only E3 may depend on events, and it must depend exactly on E1 then E2.",
                expected=expected_dependencies,
                actual=actual_dependencies,
            )
        )

    binder_value = events[0].get("value")
    target_entity = events[1].get("entity")
    if (
        not _is_nonempty_string(binder_value)
        or not _is_nonempty_string(target_entity)
        or binder_value != target_entity
    ):
        issues.append(
            _issue(
                "bridge_target_missing_binder_join",
                "1events[0].value",
                "E1.value must equal E2.entity byte-for-byte.",
                binder_value=binder_value,
                target_entity=target_entity,
            )
        )

    answer_value = events[1].get("value")
    preferred_answer_type = str(
        reasoning_target.get("preferred_answer_type") or ""
    ).strip()
    answer_slot = events[1].get("slot")
    if preferred_answer_type == "Person" and not _is_person_role_slot(
        answer_slot
    ):
        issues.append(
            _issue(
                "bridge_target_answer_slot_type_mismatch",
                "$.events[1].slot",
                "A Person bridge target requires a person-valued semantic role as E2.slot.",
                preferred_answer_type=preferred_answer_type,
                answer_slot=answer_slot,
            )
        )
    if preferred_answer_type == "Person" and not _is_atomic_person_name(
        answer_value
    ):
        issues.append(
            _issue(
                "bridge_target_answer_type_mismatch",
                "1events[1].value",
                "A Person bridge target requires one atomic person name as E2.value.",
                preferred_answer_type=preferred_answer_type,
                answer_value=answer_value,
            )
        )
    if preferred_answer_type == "Person":
        answer_key = _normalized_text(answer_value)
        answer_slot_key = _normalized_text(answer_slot)
        typed_negative_values: list[str] = []
        seen_values = {answer_key}
        for event in events[3:6]:
            candidate = event.get("value")
            candidate_key = _normalized_text(candidate)
            if (
                _is_atomic_person_name(candidate)
                and _is_person_role_slot(event.get("slot"))
                and _normalized_text(event.get("slot")) == answer_slot_key
                and candidate_key
                and candidate_key not in seen_values
            ):
                seen_values.add(candidate_key)
                typed_negative_values.append(str(candidate).strip())
        if len(typed_negative_values) < 3:
            issues.append(
                _issue(
                    "bridge_target_insufficient_typed_distractors",
                    "$.events[3:]",
                    "A Person bridge target requires three distinct Person-valued hard negatives.",
                    typed_distractor_count=len(typed_negative_values),
                    typed_distractor_values=typed_negative_values,
                )
            )
    if (
        str(reasoning_target.get("hard_negative_strategy") or "").strip()
        == "one_hop_preserving_alternative"
        and len(events) >= 4
        and _normalized_text(events[3].get("entity"))
        != _normalized_text(events[2].get("entity"))
    ):
        issues.append(
            _issue(
                "bridge_target_one_hop_negative_mismatch",
                "$.events[3].entity",
                "E4 must preserve the terminal entity for this bridge hard-negative strategy.",
                terminal_entity=events[2].get("entity"),
                hard_negative_entity=events[3].get("entity"),
            )
        )

    answer_surface = _normalized_text(answer_value)
    terminal_surfaces = [
        _normalized_text(events[2].get(field))
        for field in ("entity", "slot", "value")
    ]
    if not answer_surface or any(
        f" {answer_surface} " in f" {surface} "
        for surface in terminal_surfaces
        if surface
    ):
        issues.append(
            _issue(
                "bridge_target_terminal_copies_answer",
                "1events[2]",
                "E3 is a proof connector and must not expose E2.value.",
                answer_value=events[1].get("value"),
            )
        )
    if not any(
        _normalized_text(event.get("entity"))
        == _normalized_text(events[2].get("entity"))
        or _normalized_text(event.get("slot"))
        == _normalized_text(events[2].get("slot"))
        for event in events[3:]
    ):
        issues.append(
            _issue(
                "bridge_target_unrelated_hard_negatives",
                "$.events[3:]",
                "A bridge hard negative must share the terminal entity or slot.",
            )
        )

    by_id = _events_by_id(events)
    referenced_ids = {
        dependency
        for event in events
        for dependency in (
            event.get("depends_on")
            if isinstance(event.get("depends_on"), list)
            else []
        )
        if _is_nonempty_string(dependency) and dependency in by_id
    }
    endpoint_candidates = []
    for event_id in sorted(set(by_id) - referenced_ids):
        ancestors = _dependency_ancestors(event_id, by_id)
        if len(ancestors) >= 2:
            endpoint_candidates.append(
                {
                    "event_id": event_id,
                    "dependency_ancestors": sorted(ancestors),
                }
            )
    if len(endpoint_candidates) != 1:
        issues.append(
            _issue(
                "bridge_target_missing_unique_dependency_endpoint",
                "1events",
                "A target-aware bridge must have exactly one sink endpoint whose dependency closure contains at least two events.",
                endpoint_candidates=endpoint_candidates,
                candidate_count=len(endpoint_candidates),
            )
        )
    return issues


def _comparison_target_issues(
    events: list[dict[str, Any]],
    reasoning_target: dict[str, Any],
) -> list[dict[str, Any]]:
    active_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in events:
        slot = event.get("slot")
        time_index = event.get("time_index")
        if (
            event.get("status") in ACTIVE_STATUSES
            and _is_nonempty_string(slot)
            and isinstance(time_index, int)
            and not isinstance(time_index, bool)
        ):
            active_groups.setdefault((_normalized_text(slot), time_index), []).append(event)

    primary_candidates = []
    for (slot, time_index), group in active_groups.items():
        values = [event.get("value") for event in group]
        entities = {_normalized_text(event.get("entity", "")) for event in group}
        if (
            len(group) == 2
            and len(entities - {""}) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            and values[0] != values[1]
        ):
            primary_candidates.append((slot, time_index, group))
    if len(primary_candidates) != 1:
        return []

    primary_slot, primary_time, primary = primary_candidates[0]
    primary_entities = {
        _normalized_text(event.get("entity", "")) for event in primary
    }
    primary_difference = abs(int(primary[0]["value"]) - int(primary[1]["value"]))
    contrast_groups: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        time_index = event.get("time_index")
        if (
            event.get("status") in NON_ACTIVE_STATUSES
            and _normalized_text(event.get("slot", "")) == primary_slot
            and isinstance(time_index, int)
            and not isinstance(time_index, bool)
            and time_index != primary_time
        ):
            contrast_groups.setdefault(time_index, []).append(event)

    contrast_candidates = []
    for time_index, group in contrast_groups.items():
        values = [event.get("value") for event in group]
        entities = {_normalized_text(event.get("entity", "")) for event in group}
        if (
            len(group) == 2
            and entities == primary_entities
            and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            and values[0] != values[1]
        ):
            contrast_candidates.append((time_index, group))

    if len(contrast_candidates) != 1:
        return [
            _issue(
                "comparison_target_missing_unique_wrong_time_pair",
                "$.events",
                "A target-aware comparison must contain exactly one non-active wrong-time pair for the same entities and slot.",
                primary_event_ids=[str(event.get("event_id") or "") for event in primary],
                contrast_pair_count=len(contrast_candidates),
            )
        ]

    contrast_time, contrast = contrast_candidates[0]
    contrast_difference = abs(int(contrast[0]["value"]) - int(contrast[1]["value"]))
    issues: list[dict[str, Any]] = []
    if contrast_difference == primary_difference:
        issues.append(
            _issue(
                "comparison_target_contrast_difference_not_distinct",
                "$.events",
                "The wrong-time contrast pair must imply a different absolute difference from the primary pair.",
                primary_difference=primary_difference,
                contrast_difference=contrast_difference,
                contrast_time_index=contrast_time,
            )
        )
    strategy = str(reasoning_target.get("hard_negative_strategy") or "").strip()
    if (
        strategy == "near_numeric_off_by_one"
        and abs(primary_difference - contrast_difference) != 1
    ):
        issues.append(
            _issue(
                "comparison_target_near_numeric_strategy_mismatch",
                "$.events",
                "near_numeric_off_by_one requires the contrast difference to be exactly one away from the primary difference.",
                primary_difference=primary_difference,
                contrast_difference=contrast_difference,
            )
        )
    return issues


def _visibility_target_issues(
    events: list[dict[str, Any]],
    reasoning_target: dict[str, Any],
) -> list[dict[str, Any]]:
    structured: list[tuple[dict[str, Any], dict[str, Any]]] = []
    malformed_ids: list[str] = []
    for event in events:
        if "scope" not in event:
            continue
        scope = event.get("scope")
        if not (
            isinstance(scope, dict)
            and _is_nonempty_string(scope.get("observer"))
            and _is_nonempty_string(scope.get("permission"))
        ):
            malformed_ids.append(str(event.get("event_id") or ""))
            continue
        structured.append((event, scope))

    issues: list[dict[str, Any]] = []
    if malformed_ids or not structured:
        issues.append(
            _issue(
                "visibility_target_missing_structured_scope",
                "$.events",
                "Target-aware visibility scopes must contain non-empty observer and permission fields.",
                malformed_event_ids=malformed_ids,
                structured_scope_count=len(structured),
            )
        )

    active_subjects_by_scope: dict[tuple[str, str, str, str], set[str]] = {}
    active_event_ids_by_scope: dict[tuple[str, str, str, str], list[str]] = {}
    for event, scope in structured:
        if event.get("status") not in ACTIVE_STATUSES:
            continue
        key = (
            _normalized_text(event.get("slot", "")),
            _value_key(event.get("value")),
            _normalized_text(scope["observer"]),
            _normalized_text(scope["permission"]),
        )
        active_subjects_by_scope.setdefault(key, set()).add(
            _normalized_text(event.get("entity", ""))
        )
        active_event_ids_by_scope.setdefault(key, []).append(
            str(event.get("event_id") or "")
        )
    ambiguous = [
        {
            "slot": key[0],
            "value_key": key[1],
            "observer": key[2],
            "permission": key[3],
            "event_ids": active_event_ids_by_scope[key],
            "subjects": sorted(subjects),
        }
        for key, subjects in active_subjects_by_scope.items()
        if len(subjects - {""}) > 1
    ]
    if ambiguous:
        issues.append(
            _issue(
                "visibility_target_ambiguous_active_scope_subject",
                "$.events",
                "One structured scope tuple must not resolve to multiple active subjects.",
                ambiguous_scope_tuples=ambiguous,
            )
        )
    return issues


def _count_target_issues(
    events: list[dict[str, Any]],
    reasoning_target: dict[str, Any],
) -> list[dict[str, Any]]:
    conditions = []
    missing_condition_ids = []
    for event in events:
        membership = event.get("set_membership")
        condition = membership.get("condition") if isinstance(membership, dict) else None
        if not _is_nonempty_string(condition):
            missing_condition_ids.append(str(event.get("event_id") or ""))
        else:
            conditions.append(_normalized_text(condition))
    if not missing_condition_ids and len(set(conditions)) == 1:
        return []
    return [
        _issue(
            "count_target_missing_shared_condition",
            "$.events",
            "Every target-aware count membership must state the same non-empty inclusion condition.",
            missing_condition_event_ids=missing_condition_ids,
            normalized_conditions=sorted(set(conditions)),
        )
    ]


def _ordered_dependency_edges(
    events: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = _events_by_id(events)
    edges = []
    for consequence in events:
        consequence_time = consequence.get("time_index")
        if isinstance(consequence_time, bool) or not isinstance(consequence_time, int):
            continue
        dependencies = consequence.get("depends_on")
        if not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            cause = by_id.get(dependency) if _is_nonempty_string(dependency) else None
            cause_time = cause.get("time_index") if cause is not None else None
            if (
                cause is not None
                and isinstance(cause_time, int)
                and not isinstance(cause_time, bool)
                and cause_time < consequence_time
            ):
                edges.append((cause, consequence))
    return edges


def _cause_target_issues(
    events: list[dict[str, Any]],
    reasoning_target: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        str(reasoning_target.get("reasoning_subtype") or "").strip()
        != "direct_causal_dependency"
        and str(reasoning_target.get("reasoning_template") or "").strip()
        != "trace_cause_to_effect"
    ):
        return []

    edges = _ordered_dependency_edges(events)
    if len(edges) != 1:
        return [
            _issue(
                "cause_target_not_unique_direct_pair",
                "$.events",
                "A direct-causal target must contain exactly one ordered dependency edge.",
                ordered_edges=[
                    {
                        "cause_event_id": str(cause.get("event_id") or ""),
                        "effect_event_id": str(effect.get("event_id") or ""),
                    }
                    for cause, effect in edges
                ],
            )
        ]

    cause, effect = edges[0]
    cause_id = str(cause.get("event_id") or "")
    effect_id = str(effect.get("event_id") or "")
    cause_time = int(cause["time_index"])
    effect_time = int(effect["time_index"])
    adjacent_noncausal_ids = []
    for event in events:
        event_id = str(event.get("event_id") or "")
        time_index = event.get("time_index")
        if (
            event_id in {cause_id, effect_id}
            or isinstance(time_index, bool)
            or not isinstance(time_index, int)
            or min(abs(time_index - cause_time), abs(time_index - effect_time)) > 1
        ):
            continue
        event_dependencies = {
            dependency
            for dependency in (
                event.get("depends_on")
                if isinstance(event.get("depends_on"), list)
                else []
            )
            if _is_nonempty_string(dependency)
        }
        cause_dependencies = set(cause.get("depends_on") or [])
        effect_dependencies = set(effect.get("depends_on") or [])
        if (
            not ({cause_id, effect_id} & event_dependencies)
            and event_id not in cause_dependencies
            and event_id not in effect_dependencies
        ):
            adjacent_noncausal_ids.append(event_id)
    effect_related_ids = [
        str(event.get("event_id") or "")
        for event in events
        if str(event.get("event_id") or "") not in {cause_id, effect_id}
        and (
            _normalized_text(event.get("entity"))
            == _normalized_text(effect.get("entity"))
            or _normalized_text(event.get("slot"))
            == _normalized_text(effect.get("slot"))
        )
    ]
    if adjacent_noncausal_ids and effect_related_ids:
        return []
    return [
        _issue(
            "cause_target_missing_structured_noncausal_negative",
            "$.events",
            "A direct-causal target needs both an adjacent noncausal event and an effect-related wrong answer.",
            cause_event_id=cause_id,
            effect_event_id=effect_id,
            adjacent_noncausal_event_ids=adjacent_noncausal_ids,
            effect_related_event_ids=effect_related_ids,
        )
    ]


def _relation_target_issues(
    events: list[dict[str, Any]],
    reasoning_target: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = _events_by_id(events)
    binder_edges = []
    for relation in events:
        dependencies = relation.get("depends_on")
        if not isinstance(dependencies, list) or len(dependencies) != 1:
            continue
        binder = by_id.get(dependencies[0])
        if binder is None:
            continue
        same_entity_new_slot = (
            _normalized_text(binder.get("entity", ""))
            == _normalized_text(relation.get("entity", ""))
            and _normalized_text(binder.get("slot", ""))
            != _normalized_text(relation.get("slot", ""))
        )
        value_binds_entity = _same_value(
            binder.get("value"), relation.get("entity")
        )
        if same_entity_new_slot or value_binds_entity:
            binder_edges.append((binder, relation))
    if len(binder_edges) == 1:
        return []
    return [
        _issue(
            "relation_target_missing_unique_binder_dependency",
            "$.events",
            "A target-aware relation lookup must have exactly one dependency edge that binds an entity before resolving a different relation constraint.",
            binder_edges=[
                {
                    "binder_event_id": str(binder.get("event_id") or ""),
                    "relation_event_id": str(relation.get("event_id") or ""),
                }
                for binder, relation in binder_edges
            ],
        )
    ]


_OPERATION_VALIDATORS = {
    "temporal_state_update": _temporal_issues,
    "bridge": _bridge_issues,
    "comparison": _comparison_issues,
    "visibility": _visibility_issues,
    "count": _count_issues,
    "cause_consequence": _cause_issues,
    "relation_lookup": _relation_issues,
}

_TARGET_OPERATION_VALIDATORS = {
    "bridge": _bridge_target_issues,
    "comparison": _comparison_target_issues,
    "visibility": _visibility_target_issues,
    "count": _count_target_issues,
    "cause_consequence": _cause_target_issues,
    "relation_lookup": _relation_target_issues,
}


def validate_graph_delta(
    graph_delta: Any,
    *,
    operation: str | None = None,
    reasoning_target: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic validation issues for a semantic graph delta.

    ``operation`` may use the short aliases ``temporal``, ``cause``, and
    ``relation``.  If no operation is supplied (and none is embedded in the
    delta), only the canonical graph/event contract is checked.
    """

    if not isinstance(graph_delta, dict):
        return [_issue("invalid_graph_type", "$", "A graph delta must be a JSON object.")]

    issues: list[dict[str, Any]] = []
    for field in REQUIRED_GRAPH_FIELDS:
        if field not in graph_delta:
            issues.append(
                _issue(
                    "missing_graph_field",
                    f"$.{field}",
                    f"Required graph field '{field}' is missing.",
                    field=field,
                )
            )

    if graph_delta.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            _issue(
                "unsupported_schema_version",
                "$.schema_version",
                f"schema_version must be exactly '{SCHEMA_VERSION}'.",
                actual=graph_delta.get("schema_version"),
                expected=SCHEMA_VERSION,
            )
        )

    for field in ("graph_id", "parent_graph_id"):
        if field in graph_delta and not _is_nonempty_string(graph_delta[field]):
            issues.append(
                _issue(
                    f"invalid_{field}",
                    f"$.{field}",
                    f"{field} must be a non-empty string.",
                )
            )
    if (
        _is_nonempty_string(graph_delta.get("graph_id"))
        and graph_delta.get("graph_id") == graph_delta.get("parent_graph_id")
    ):
        issues.append(
            _issue(
                "graph_id_matches_parent",
                "$.parent_graph_id",
                "graph_id and parent_graph_id must identify different graph snapshots.",
            )
        )

    events = graph_delta.get("events")
    if not isinstance(events, list):
        issues.append(_issue("invalid_events", "$.events", "events must be a non-empty JSON array."))
    elif not events:
        issues.append(_issue("empty_events", "$.events", "events must contain at least one event."))
    else:
        for index, event in enumerate(events):
            issues.extend(validate_event(event, path=f"$.events[{index}]"))

        by_id: dict[str, list[int]] = {}
        for index, event in enumerate(events):
            if isinstance(event, dict) and _is_nonempty_string(event.get("event_id")):
                by_id.setdefault(event["event_id"], []).append(index)
        for event_id, indexes in by_id.items():
            if len(indexes) > 1:
                for index in indexes[1:]:
                    issues.append(
                        _issue(
                            "duplicate_event_id",
                            f"$.events[{index}].event_id",
                            "event_id must be unique within the graph delta.",
                            event_id=event_id,
                            first_index=indexes[0],
                        )
                    )

        known_ids = set(by_id)
        for index, event in enumerate(events):
            if not isinstance(event, dict) or not isinstance(event.get("depends_on"), list):
                continue
            event_id = event.get("event_id")
            for dep_index, dependency in enumerate(event["depends_on"]):
                if not _is_nonempty_string(dependency):
                    continue
                path = f"$.events[{index}].depends_on[{dep_index}]"
                if dependency == event_id:
                    issues.append(
                        _issue(
                            "self_dependency",
                            path,
                            "An event cannot depend on itself.",
                            event_id=event_id,
                        )
                    )
                elif dependency not in known_ids:
                    issues.append(
                        _issue(
                            "dependency_not_found",
                            path,
                            "depends_on must reference an event_id present in this graph delta.",
                            dependency=dependency,
                        )
                    )

    canonical_operation, operation_issues = _operation_name(graph_delta, operation)
    issues.extend(operation_issues)
    if canonical_operation is not None and isinstance(events, list):
        operation_events = _valid_events(events)
        minimum_events, maximum_events = OPERATION_EVENT_COUNT_RANGES[canonical_operation]
        if len(operation_events) < minimum_events:
            issues.append(
                _issue(
                    "operation_too_few_events",
                    "$.events",
                    "A formal Stage3 operation graph requires 6-8 events before normalization.",
                    operation=canonical_operation,
                    event_count=len(operation_events),
                    minimum=minimum_events,
                )
            )
        elif len(operation_events) > maximum_events:
            issues.append(
                _issue(
                    "operation_too_many_events",
                    "$.events",
                    "A formal Stage3 operation graph requires 6-8 events before normalization.",
                    operation=canonical_operation,
                    event_count=len(operation_events),
                    maximum=maximum_events,
                )
            )
        required_operation_events = [
            event
            for event in operation_events
            if event.get("required_for_dialogue") is not False
        ]
        if len(required_operation_events) < minimum_events:
            issues.append(
                _issue(
                    "operation_too_few_required_events",
                    "$.events",
                    "All 6-8 formal Stage3 operation events must be required for dialogue realization.",
                    operation=canonical_operation,
                    required_event_count=len(required_operation_events),
                    minimum=minimum_events,
                )
            )
        if canonical_operation == "temporal_state_update":
            issues.extend(_temporal_issues(operation_events, reasoning_target))
        else:
            issues.extend(_OPERATION_VALIDATORS[canonical_operation](operation_events))
            if _has_reasoning_target(reasoning_target):
                issues.extend(
                    _TARGET_OPERATION_VALIDATORS[canonical_operation](
                        operation_events,
                        reasoning_target,
                    )
                )
    return issues


def is_valid_graph_delta(
    graph_delta: Any,
    *,
    operation: str | None = None,
    reasoning_target: dict[str, Any] | None = None,
) -> bool:
    """Return whether ``graph_delta`` satisfies the validator without mutating it."""

    return not validate_graph_delta(
        graph_delta,
        operation=operation,
        reasoning_target=reasoning_target,
    )
