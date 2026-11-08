"""Fail-closed Stage3 parent/child continuity and timeline helpers.

These helpers validate API-authored child facts against a retrieved parent-event
view.  They never invent graph facts or repair an API response.  The only
materialization performed here is assigning child global time coordinates after
the child has passed semantic validation; the operation-local ``time_index`` is
preserved verbatim.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_GENERIC_TOPIC_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "event",
    "fact",
    "group",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "memory",
    "new",
    "old",
    "of",
    "on",
    "or",
    "state",
    "status",
    "thing",
    "update",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "with",
}
_EXPLICIT_RELATION_MARKERS = {
    "caused_by",
    "causes",
    "continues",
    "depends_on",
    "follows",
    "led_to",
    "modifies",
    "resolves",
    "supersedes",
    "updates",
    "same_identity_exact_slot_extension",
    "same_identity_slot_family_extension",
    "updates_parent_state_timeline",
}

_SLOT_FAMILY_TERMS = {
    "cause": {
        "cause",
        "caused",
        "consequence",
        "effect",
        "outcome",
        "reason",
        "result",
        "trigger",
    },
    "community_service": {"charity", "community", "service", "volunteer"},
    "education": {"class", "course", "degree", "education", "school", "study"},
    "employment": {"career", "employ", "job", "work"},
    "event_schedule": {
        "appointment",
        "event",
        "meeting",
        "party",
        "schedule",
        "scheduled",
        "trip",
    },
    "location": {
        "address",
        "destination",
        "location",
        "place",
        "room",
        "site",
        "travel",
        "venue",
        "went",
    },
    "membership": {
        "candidate",
        "count",
        "group",
        "include",
        "joined",
        "member",
        "membership",
        "roster",
    },
    "person_relation": {
        "assigned",
        "friend",
        "host",
        "owner",
        "partner",
        "person",
        "relationship",
    },
    "plan_status": {
        "active",
        "cancel",
        "complete",
        "goal",
        "plan",
        "planned",
        "status",
    },
    "resource": {
        "donated",
        "floor",
        "flooring",
        "item",
        "object",
        "possession",
        "resource",
        "selected",
    },
    "visibility": {
        "access",
        "audience",
        "observer",
        "permission",
        "scope",
        "visibility",
    },
}

_VALUE_IDENTITY_SLOT_TERMS = {
    "assigned",
    "candidate",
    "companion",
    "friend",
    "host",
    "member",
    "observer",
    "owner",
    "partner",
    "person",
    "replacement",
    "selected",
}

_CAUSAL_REFERENCE_FIELDS = {
    "cause_event_id",
    "caused_by_event_id",
    "cause_parent_event_id",
    "source_cause_event_id",
}
_TEMPORAL_REFERENCE_FIELDS = {
    "follows_event_id",
    "previous_event_id",
    "supersedes_event_id",
    "updates_event_id",
}
_DEPENDENCY_REFERENCE_FIELDS = {
    "depends_on_event_ids",
    "depends_on_parent_event_ids",
    "parent_topic_anchor_event_ids",
}
_RESOURCE_FIELDS = {
    "item",
    "object",
    "resource",
    "target",
    "value",
}
_GENERIC_RESOURCE_VALUES = {
    "active",
    "allowed",
    "canceled",
    "completed",
    "denied",
    "expired",
    "false",
    "historical",
    "outdated",
    "planned",
    "private",
    "proposed",
    "public",
    "reported",
    "true",
    "unconfirmed",
}


def _norm(value: Any) -> str:
    return re.sub(r"[\W_]+", " ", str(value or "").casefold()).strip()


def _topic_token(value: str) -> str:
    """Return a conservative lexical key without external NLP dependencies."""

    token = value.casefold().strip("_")
    if len(token) > 6 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 6 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _event_id(event: dict[str, Any]) -> str:
    return str(
        event.get("event_id")
        or event.get("event_instance_id")
        or event.get("node_id")
        or ""
    ).strip()


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)


def _reference_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, list):
        return {
            str(item).strip()
            for item in value
            if not isinstance(item, (dict, list)) and str(item).strip()
        }
    return set()


def _identity_values(event: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in ("entity", "owner", "observer", "subject"):
        raw = event.get(field)
        if isinstance(raw, str) and (normalized := _norm(raw)):
            values.add(normalized)
    scope = event.get("scope")
    if isinstance(scope, dict):
        for field in ("entity", "owner", "observer", "subject"):
            raw = scope.get(field)
            if isinstance(raw, str) and (normalized := _norm(raw)):
                values.add(normalized)
    slot = _norm(event.get("slot") or event.get("event_type") or event.get("predicate"))
    slot_tokens = set(_TOKEN_RE.findall(slot))
    raw_value = event.get("value")
    if (
        slot_tokens.intersection(_VALUE_IDENTITY_SLOT_TERMS)
        and isinstance(raw_value, str)
        and 1 <= len(_TOKEN_RE.findall(_norm(raw_value))) <= 8
        and (normalized_value := _norm(raw_value))
    ):
        values.add(normalized_value)
    return values


def _slot_families(event: dict[str, Any]) -> set[str]:
    slot = _norm(event.get("slot") or event.get("event_type") or event.get("predicate"))
    tokens = {_topic_token(token) for token in _TOKEN_RE.findall(slot)}
    families = {
        family
        for family, family_terms in _SLOT_FAMILY_TERMS.items()
        if tokens.intersection({_topic_token(term) for term in family_terms})
    }
    return families


def _resource_values(event: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    candidates = [event.get(field) for field in _RESOURCE_FIELDS]
    scope = event.get("scope")
    if isinstance(scope, dict):
        candidates.extend(
            scope.get(field)
            for field in ("item", "location", "object", "resource", "target")
        )
    for candidate in candidates:
        for raw in _string_values(candidate):
            normalized = _norm(raw)
            tokens = {_topic_token(token) for token in _TOKEN_RE.findall(normalized)}
            meaningful = {
                token
                for token in tokens
                if len(token) > 1 and token not in _GENERIC_TOPIC_TOKENS
            }
            if (
                normalized
                and normalized not in _GENERIC_RESOURCE_VALUES
                and meaningful
            ):
                values.add(normalized)
    return values


def _event_reference_kinds(event: dict[str, Any], parent_id: str) -> set[str]:
    kinds: set[str] = set()
    for field in _CAUSAL_REFERENCE_FIELDS:
        if parent_id in _reference_values(event.get(field)):
            kinds.add("causal")
    for field in _TEMPORAL_REFERENCE_FIELDS:
        if parent_id in _reference_values(event.get(field)):
            kinds.add("temporal")
    for field in _DEPENDENCY_REFERENCE_FIELDS:
        if parent_id in _reference_values(event.get(field)):
            kinds.add("dependency")
    return kinds


def _time_value(event: dict[str, Any]) -> int | None:
    for field in ("global_time_index", "time_index"):
        raw = event.get(field)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _topic_tokens(event: dict[str, Any]) -> set[str]:
    topic_fields = {
        "slot": event.get("slot") or event.get("event_type"),
        "value": event.get("value"),
        "memory_text": event.get("memory_text"),
        "supporting_span": event.get("supporting_span"),
        "supporting_spans": event.get("supporting_spans"),
        "resource": event.get("resource"),
    }
    identity_tokens = {
        _topic_token(token)
        for value in _identity_values(event)
        for token in _TOKEN_RE.findall(value)
    }
    return {
        _topic_token(token)
        for text in _string_values(topic_fields)
        for token in _TOKEN_RE.findall(_norm(text))
        if len(_topic_token(token)) > 1
        and _topic_token(token) not in _GENERIC_TOPIC_TOKENS
        and _topic_token(token) not in identity_tokens
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _continuity_payload(child_graph: dict[str, Any]) -> dict[str, Any]:
    metadata = child_graph.get("metadata") if isinstance(child_graph.get("metadata"), dict) else {}
    nested = (
        metadata.get("parent_continuity")
        if isinstance(metadata.get("parent_continuity"), dict)
        else {}
    )
    events = [
        event
        for event in (child_graph.get("events") or [])
        if isinstance(event, dict)
    ]

    def combined_ids(field: str) -> list[str]:
        values: list[str] = []
        for owner in (child_graph, metadata, nested, *events):
            values.extend(_string_list(owner.get(field)))
        return list(dict.fromkeys(values))

    relations: list[Any] = []
    for owner in (child_graph, metadata, nested):
        raw = owner.get("continuity_relations")
        if isinstance(raw, list):
            relations.extend(copy.deepcopy(raw))
    return {
        "depends_on_parent_event_ids": combined_ids(
            "depends_on_parent_event_ids"
        ),
        "parent_topic_anchor_event_ids": combined_ids(
            "parent_topic_anchor_event_ids"
        ),
        "continuity_relations": relations,
    }


def _shared_resource_values(
    parent: dict[str, Any], child: dict[str, Any]
) -> set[str]:
    parents = _resource_values(parent)
    children = _resource_values(child)
    shared = parents.intersection(children)
    for parent_value in parents:
        parent_tokens = _topic_tokens({"value": parent_value})
        if len(parent_tokens) < 2:
            continue
        for child_value in children:
            child_tokens = _topic_tokens({"value": child_value})
            if len(child_tokens) < 2:
                continue
            if parent_tokens.issubset(child_tokens) or child_tokens.issubset(parent_tokens):
                shared.add(
                    parent_value if len(parent_value) <= len(child_value) else child_value
                )
    return shared


def _semantic_pair_link(
    parent: dict[str, Any], child: dict[str, Any]
) -> dict[str, Any]:
    """Independently derive semantic continuity for one parent/child pair."""

    parent_id = _event_id(parent)
    child_id = _event_id(child)
    parent_entities = _identity_values(parent)
    child_entities = _identity_values(child)
    shared_entities = parent_entities.intersection(child_entities)
    parent_slot = _norm(
        parent.get("slot") or parent.get("event_type") or parent.get("predicate")
    )
    child_slot = _norm(
        child.get("slot") or child.get("event_type") or child.get("predicate")
    )
    slot_exact = bool(parent_slot and parent_slot == child_slot)
    shared_slot_families = _slot_families(parent).intersection(_slot_families(child))
    parent_value = _norm(parent.get("value"))
    child_value = _norm(child.get("value"))
    value_exact = bool(parent_value and parent_value == child_value)
    shared_resources = _shared_resource_values(parent, child)
    parent_topics = _topic_tokens(parent)
    child_topics = _topic_tokens(child)
    topic_overlap = parent_topics.intersection(child_topics)
    min_topic_size = min(len(parent_topics), len(child_topics))
    topic_coverage = len(topic_overlap) / min_topic_size if min_topic_size else 0.0
    strong_topic_overlap = len(topic_overlap) >= 2 and topic_coverage >= 0.40
    reference_kinds = _event_reference_kinds(child, parent_id)
    parent_time = _time_value(parent)
    child_time = _time_value(child)
    child_after_parent = (
        parent_time is not None
        and child_time is not None
        and child_time > parent_time
    )
    parent_status = _norm(parent.get("status"))
    child_status = _norm(child.get("status"))

    reasons: list[str] = []
    if shared_entities and slot_exact and value_exact:
        reasons.append("exact_parent_fact")
    if shared_entities and slot_exact:
        reasons.append("same_entity_exact_slot")
    elif shared_entities and shared_slot_families:
        reasons.append("same_entity_slot_family")
    if shared_resources:
        reasons.append("shared_parent_resource")
    if strong_topic_overlap:
        reasons.append("shared_parent_topic")
    if (
        shared_entities
        and (slot_exact or shared_slot_families)
        and child_after_parent
        and parent_status
        and child_status
        and parent_status != child_status
    ):
        reasons.append("verified_status_or_temporal_extension")
    if "temporal" in reference_kinds and child_after_parent:
        reasons.append("temporal_parent_event_reference")
    if "causal" in reference_kinds and child_after_parent:
        reasons.append("causal_parent_event_reference")

    # A bare event-level dependency is still only a declaration.  It can bind a
    # relation to the right pair, but cannot manufacture semantic continuity.
    supported = bool(reasons)
    return {
        "parent_event_id": parent_id,
        "child_event_id": child_id,
        "supported": supported,
        "reasons": reasons,
        "shared_entities": sorted(shared_entities),
        "slot_exact": slot_exact,
        "shared_slot_families": sorted(shared_slot_families),
        "value_exact": value_exact,
        "shared_resources": sorted(shared_resources),
        "shared_topic_tokens": sorted(topic_overlap) if strong_topic_overlap else [],
        "topic_overlap_count": len(topic_overlap),
        "topic_coverage": round(topic_coverage, 6),
        "event_reference_kinds": sorted(reference_kinds),
        "child_after_parent": child_after_parent,
    }


def _relation_marker(value: Any) -> str:
    normalized = _norm(value).replace(" ", "_")
    if normalized in _EXPLICIT_RELATION_MARKERS:
        return normalized
    for marker in sorted(_EXPLICIT_RELATION_MARKERS, key=len, reverse=True):
        if marker in normalized:
            return marker
    return ""


def _relation_supported(marker: str, link: dict[str, Any]) -> bool:
    reasons = set(link.get("reasons") or [])
    if marker == "same_identity_exact_slot_extension":
        return "same_entity_exact_slot" in reasons
    if marker == "same_identity_slot_family_extension":
        return "same_entity_slot_family" in reasons
    if marker == "updates_parent_state_timeline":
        return bool(
            reasons.intersection(
                {
                    "same_entity_exact_slot",
                    "same_entity_slot_family",
                    "verified_status_or_temporal_extension",
                }
            )
        ) and bool(link.get("child_after_parent"))
    if marker == "continues":
        return bool(reasons)
    if marker in {"modifies", "resolves", "supersedes", "updates"}:
        return bool(
            reasons.intersection(
                {
                    "exact_parent_fact",
                    "same_entity_exact_slot",
                    "same_entity_slot_family",
                    "verified_status_or_temporal_extension",
                    "temporal_parent_event_reference",
                }
            )
        ) and bool(link.get("child_after_parent"))
    if marker in {"caused_by", "causes", "led_to"}:
        return "causal_parent_event_reference" in reasons
    if marker == "follows":
        return bool(link.get("child_after_parent")) and bool(reasons)
    if marker == "depends_on":
        return bool(reasons) and "dependency" in set(
            link.get("event_reference_kinds") or []
        )
    return False


def parent_child_continuity_audit(
    child_graph: dict[str, Any],
    parent_events: list[dict[str, Any]],
    *,
    selected_parent_event_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic continuity audit; no input object is mutated.

    Formal v4 generation requires an explicit reference to a retrieved parent
    event.  This prevents a shared person name from being treated as a lineage
    extension when every topic/resource/slot changed.
    """

    parent_by_id = {
        event_id: event
        for event in parent_events
        if isinstance(event, dict) and (event_id := _event_id(event))
    }
    selected_source = (
        parent_by_id.keys()
        if selected_parent_event_ids is None
        else selected_parent_event_ids
    )
    selected_ids = {
        str(value).strip() for value in selected_source if str(value).strip()
    }
    payload = _continuity_payload(child_graph if isinstance(child_graph, dict) else {})
    declared_dependencies = payload["depends_on_parent_event_ids"]
    declared_anchors = payload["parent_topic_anchor_event_ids"]
    relations = payload["continuity_relations"]
    relation_parent_refs = [
        str(relation.get("parent_event_id") or "").strip()
        for relation in relations
        if isinstance(relation, dict)
        and str(relation.get("parent_event_id") or "").strip()
    ]
    declared_refs = list(
        dict.fromkeys(
            declared_dependencies + declared_anchors + relation_parent_refs
        )
    )
    known_refs = [ref for ref in declared_refs if ref in parent_by_id and ref in selected_ids]
    unknown_refs = [ref for ref in declared_refs if ref not in parent_by_id]
    unselected_refs = [
        ref for ref in declared_refs if ref in parent_by_id and ref not in selected_ids
    ]

    children = [
        event
        for event in (child_graph.get("events") if isinstance(child_graph, dict) else []) or []
        if isinstance(event, dict)
    ]
    candidate_parent_ids = sorted(selected_ids.intersection(parent_by_id))
    links = [
        _semantic_pair_link(parent_by_id[parent_id], child)
        for parent_id in candidate_parent_ids
        for child in children
    ]
    link_by_pair = {
        (link["parent_event_id"], link["child_event_id"]): link
        for link in links
        if link["parent_event_id"] and link["child_event_id"]
    }
    supported_links = [link for link in links if link["supported"]]
    supported_parent_ids = {
        link["parent_event_id"] for link in supported_links
    }
    verified_refs = [ref for ref in known_refs if ref in supported_parent_ids]
    unsupported_refs = [ref for ref in known_refs if ref not in supported_parent_ids]
    verified_dependency_refs = [
        ref for ref in declared_dependencies if ref in verified_refs
    ]
    verified_anchor_refs = [ref for ref in declared_anchors if ref in verified_refs]
    entity_overlap = any(
        _identity_values(parent_by_id[parent_id]).intersection(
            _identity_values(child)
        )
        for parent_id in candidate_parent_ids
        for child in children
    )

    verified_relation_markers: list[str] = []
    relation_issues: list[dict[str, Any]] = []
    for relation_index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            relation_issues.append(
                {
                    "code": "parent_continuity_relation_malformed",
                    "relation_index": relation_index,
                }
            )
            continue
        parent_id = str(relation.get("parent_event_id") or "").strip()
        child_id = str(relation.get("child_event_id") or "").strip()
        marker = _relation_marker(relation.get("relation") or relation.get("type"))
        if not parent_id or not child_id or not marker:
            relation_issues.append(
                {
                    "code": "parent_continuity_relation_incomplete",
                    "relation_index": relation_index,
                    "parent_event_id": parent_id,
                    "child_event_id": child_id,
                    "relation_marker": marker,
                }
            )
            continue
        link = link_by_pair.get((parent_id, child_id))
        if link is None or not _relation_supported(marker, link):
            relation_issues.append(
                {
                    "code": "parent_continuity_relation_semantically_unsupported",
                    "relation_index": relation_index,
                    "parent_event_id": parent_id,
                    "child_event_id": child_id,
                    "relation_marker": marker,
                    "independent_link_reasons": (
                        copy.deepcopy(link.get("reasons") or []) if link else []
                    ),
                }
            )
            continue
        verified_relation_markers.append(marker)

    reason_set = {
        reason for link in supported_links for reason in link.get("reasons") or []
    }
    reasons: list[str] = []
    if verified_refs:
        reasons.append("verified_parent_event_reference")
    if reason_set.intersection(
        {"exact_parent_fact", "same_entity_exact_slot", "same_entity_slot_family"}
    ):
        reasons.append("same_entity_slot_extension")
    if reason_set.intersection({"shared_parent_resource", "shared_parent_topic"}):
        reasons.append("shared_parent_resource_or_topic")
    if reason_set.intersection(
        {
            "verified_status_or_temporal_extension",
            "temporal_parent_event_reference",
        }
    ):
        reasons.append("verified_temporal_or_status_relation")
    if "causal_parent_event_reference" in reason_set:
        reasons.append("verified_causal_relation")
    if verified_relation_markers:
        reasons.append("verified_declared_continuity_relation")

    evidence_weights = {
        "exact_parent_fact": 1.0,
        "same_entity_exact_slot": 0.80,
        "same_entity_slot_family": 0.65,
        "shared_parent_resource": 0.60,
        "shared_parent_topic": 0.50,
        "verified_status_or_temporal_extension": 0.80,
        "temporal_parent_event_reference": 0.75,
        "causal_parent_event_reference": 0.85,
    }
    score = max(
        (
            max(
                (evidence_weights.get(reason, 0.0) for reason in link["reasons"]),
                default=0.0,
            )
            for link in supported_links
            if link["parent_event_id"] in set(verified_refs)
        ),
        default=0.0,
    )
    issues: list[dict[str, Any]] = []
    if not declared_refs:
        issues.append(
            {
                "code": "parent_continuity_reference_missing",
                "message": "Child graph must explicitly reference at least one retrieved parent event.",
            }
        )
    if unknown_refs:
        issues.append(
            {
                "code": "parent_continuity_reference_unknown",
                "parent_event_ids": unknown_refs,
            }
        )
    if unselected_refs:
        issues.append(
            {
                "code": "parent_continuity_reference_not_retrieved",
                "parent_event_ids": unselected_refs,
            }
        )
    if declared_refs and not known_refs:
        issues.append(
            {
                "code": "parent_continuity_has_no_valid_reference",
                "message": "Declared parent references do not bind to the retrieved parent view.",
            }
        )
    if unsupported_refs:
        issues.append(
            {
                "code": "parent_continuity_reference_semantically_unsupported",
                "parent_event_ids": unsupported_refs,
                "message": (
                    "Declared parent references have no independently verified "
                    "slot-family, resource/topic, temporal, or causal link to a child event."
                ),
            }
        )
    issues.extend(relation_issues)
    if entity_overlap and not supported_links:
        issues.append(
            {
                "code": "parent_continuity_entity_only",
                "message": "Entity-name overlap alone is not a valid lineage extension.",
            }
        )
    if known_refs and score < 0.45:
        issues.append(
            {
                "code": "parent_continuity_score_too_low",
                "continuity_score": round(score, 6),
            }
        )

    return {
        "pass": not issues,
        "depends_on_parent_event_ids": verified_dependency_refs,
        "parent_topic_anchor_event_ids": verified_anchor_refs,
        "declared_parent_event_ids": declared_refs,
        "verified_parent_event_ids": verified_refs,
        "unsupported_declared_parent_event_ids": unsupported_refs,
        "unknown_parent_event_ids": unknown_refs,
        "unselected_parent_event_ids": unselected_refs,
        "continuity_score": round(score, 6),
        "continuity_reasons": reasons,
        "same_entity_slot_pairs": [
            [link["parent_event_id"], link["child_event_id"]]
            for link in supported_links
            if set(link["reasons"]).intersection(
                {"exact_parent_fact", "same_entity_exact_slot", "same_entity_slot_family"}
            )
        ],
        "shared_topic_tokens": sorted(
            {
                token
                for link in supported_links
                for token in link.get("shared_topic_tokens") or []
            }
        ),
        "explicit_relation_markers": sorted(set(verified_relation_markers)),
        "independently_verified_links": copy.deepcopy(supported_links),
        "entity_overlap_detected": entity_overlap,
        "issues": issues,
    }


def parent_max_global_time_index(parent_events: Iterable[dict[str, Any]]) -> int:
    values: list[int] = []
    for event in parent_events:
        if not isinstance(event, dict):
            continue
        raw = event.get("global_time_index")
        if raw is None:
            raw = event.get("time_index")
        if isinstance(raw, bool):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            values.append(value)
    return max(values, default=0)


def materialize_child_global_time(
    current_event_plan: dict[str, Any],
    parent_events: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return a copy with child global time strictly after its parent timeline."""

    plan = copy.deepcopy(current_event_plan)
    events = [event for event in (plan.get("events") or []) if isinstance(event, dict)]
    parent_max = parent_max_global_time_index(parent_events)
    local_values: list[int] = []
    for index, event in enumerate(events, start=1):
        raw_local = event.get("local_time_index", event.get("time_index", index))
        if isinstance(raw_local, bool):
            raw_local = index
        try:
            local = int(raw_local)
        except (TypeError, ValueError):
            local = index
        event["local_time_index"] = local
        local_values.append(local)
    local_to_global = {
        local: parent_max + offset
        for offset, local in enumerate(sorted(set(local_values)), start=1)
    }
    for event in events:
        event["global_time_index"] = local_to_global[event["local_time_index"]]
    metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
    plan["metadata"] = {
        **metadata,
        "parent_max_global_time_index": parent_max,
        "child_global_time_start": min(local_to_global.values(), default=parent_max + 1),
        "child_global_time_end": max(local_to_global.values(), default=parent_max),
        "child_global_time_strictly_after_parent": all(
            int(event["global_time_index"]) > parent_max for event in events
        ),
    }
    return plan
