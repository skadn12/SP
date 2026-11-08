"""Truth-first Stage3 operation constructors.

The constructors in this module own every field that determines truth:
semantic slots, value types, event roles, statuses, topology, gold projection,
and the reasoning path.  A model may supply only the typed content bindings
declared by a skeleton.  It cannot return events, slots, edges, membership
flags, gold labels, or proof fields.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from typing import Any, Callable
from zoneinfo import available_timezones


SKELETON_SCHEMA_VERSION = "stage3_truth_skeleton_v2"
BINDING_SCHEMA_VERSION = "stage3_truth_content_bindings_v2"
BINDING_ENVELOPE_NORMALIZER_VERSION = (
    "stage3_truth_content_binding_envelope_normalizer_v2"
)
MATERIALIZATION_SCHEMA_VERSION = "stage3_truth_materialization_v1"
DISCOURSE_SCHEMA_VERSION = "stage3_truth_discourse_plan_v1"

OPERATIONS = (
    "temporal_state_update",
    "bridge",
    "comparison",
    "visibility",
    "count",
    "cause_consequence",
    "relation_lookup",
)

ANSWER_TYPES = frozenset(
    {
        "Person",
        "GroupOrOrg",
        "LocationOrScope",
        "DateOrTime",
        "Event",
        "ObjectOrCommonNoun",
        "AttributeValue",
        "ProperNounOrNamedEntity",
        "Number",
        "YesNo",
    }
)

_ATOM_TYPES = ANSWER_TYPES - {"Number", "YesNo"}
_QA_LEAK_RE = re.compile(r"(?:^|\s)[QA]\s*:", re.IGNORECASE)
_DATE_TIME_RE = re.compile(
    r"(?:\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|today|tomorrow|yesterday|noon|midnight)\b)",
    re.IGNORECASE,
)
_LOCATION_HEADS = frozenset(
    {
        "airport", "annex", "area", "avenue", "beach", "building", "cafe",
        "campus", "castle", "center", "centre", "city", "clinic", "coast", "coastline",
        "complex", "countries", "country", "county", "court", "courtyard", "department", "district",
        "farm", "field", "floor", "forest", "gallery", "garden", "gym", "hall",
        "harbor", "harbour", "home", "hospital", "hostel", "hotel", "house",
        "island", "lab", "laboratory", "lake", "library", "mall", "market",
        "mountain", "museum", "neighborhood", "office", "park", "plaza", "pool",
        "pavilion", "port", "ranch", "region", "reserve", "restaurant", "retreat", "river", "road",
        "room", "school", "shop", "square", "stadium", "state", "station",
        "store", "street", "studio", "theater", "theatre", "town", "trail",
        "university", "venue", "village", "wilderness", "woods", "workshop",
    }
)
_SCOPE_AUDIENCE_HEADS = frozenset(
    {
        "department", "employees", "faculty", "family", "guests", "household",
        "members", "public", "staff", "students", "team", "users",
    }
)
_ACCESS_CLASS_ATOMS = frozenset(
    {
        "classified", "confidential", "internal", "private", "protected", "public",
        "restricted", "secret",
    }
)
_LOCATIVE_ATOMS = frozenset(
    {
        "abroad", "downtown", "home", "indoors", "midwest", "nearby",
        "northeast", "northwest", "online", "outdoors", "overseas",
        "southeast", "southwest",
    }
)
_SPATIAL_MODIFIERS = frozenset(
    {
        "central", "eastern", "east", "global", "international", "local", "lower",
        "national", "northern", "north", "regional", "southern", "south", "upper",
        "western", "west",
    }
)
_COMMON_LOCATION_MODIFIERS = _SPATIAL_MODIFIERS | frozenset(
    {
        "community", "county", "dedicated", "indoor", "main", "municipal", "new",
        "old", "outdoor", "private", "project", "public", "remote", "shared", "sports",
    }
)
_LOCATION_CONNECTORS = frozenset({"at", "by", "in", "near", "on", "within"})
_GEOGRAPHIC_FEATURE_HEADS = frozenset(
    {"bay", "cape", "fort", "lake", "mount", "mountain", "port", "river", "saint"}
)
_TOPONYM_SUFFIXES = (
    "borough", "burg", "burgh", "chester", "field", "ford", "haven",
    "land", "mouth", "shire", "stead", "ton", "vale", "ville", "wood",
)
_US_STATE_NAMES = frozenset(
    {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "rhode island", "south carolina", "south dakota",
        "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming",
    }
)
_CONTINENT_REGION_NAMES = frozenset(
    {
        "africa", "antarctica", "asia", "australia", "caribbean",
        "central america", "europe", "middle east", "north america",
        "oceania", "south america",
    }
)
_LOCATION_ACRONYMS = frozenset({"eu", "uae", "uk", "us", "usa"})


def _geographic_key(value: Any) -> str:
    folded = unicodedata.normalize("NFKD", str(value or ""))
    ascii_folded = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_folded.casefold()).strip()


_TIMEZONE_PLACE_NAMES = frozenset(
    _geographic_key(zone.rsplit("/", 1)[-1].replace("_", " "))
    for zone in available_timezones()
    if "/" in zone
)
_NP_DETERMINERS = frozenset({"a", "an", "the"})
_NP_CLAUSE_WORDS = frozenset(
    {
        "am", "are", "at", "be", "been", "being", "by", "can", "could", "did",
        "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers",
        "him", "his", "i", "in", "into", "is", "it", "its", "may", "might", "must",
        "our", "ours", "she", "should", "their", "theirs", "them", "they", "to", "was",
        "we", "were", "will", "would", "you", "your", "yours",
    }
)
_CONTROL_STATUS_ATOMS = frozenset(
    {
        "active", "belief", "canceled", "cancelled", "completed", "confirmed",
        "expired", "historical", "outdated", "planned", "proposed", "report", "reported",
    }
)
_SCOPE_AUDIENCE_MODIFIERS = frozenset(
    {
        "approved", "authorized", "designated", "eligible", "external",
        "internal", "invited", "local", "project", "registered", "restricted",
    }
)
_EVENT_NAME_HEADS = frozenset(
    {
        "appointment", "assembly", "audition", "briefing", "celebration", "ceremony", "class", "clinic",
        "concert", "conference", "convention", "demonstration", "exhibition",
        "fair", "festival", "forum", "fundraiser", "gala", "game", "gathering", "hearing",
        "interview", "launch", "lecture", "meeting", "orientation", "panel", "party", "performance",
        "practice", "premiere", "presentation", "race", "reception", "rehearsal", "release", "retreat",
        "review", "seminar", "session", "show", "showcase", "signing", "summit", "symposium",
        "tournament", "tour", "training", "trip", "vacation", "wedding", "workshop",
    }
)
_EVENT_OUTCOME_HEADS = frozenset(
    {
        "cancellation", "conflict", "delay", "failure", "postponement", "problem",
        "result", "setback", "shortage", "success",
    }
)
_EVENT_OUTCOME_VERBS = frozenset(
    {
        "became", "caused", "completed", "encountered", "ended", "failed", "faced",
        "hit", "lost", "missed", "postponed", "recovered", "resolved", "secured",
        "suffered", "triggered", "won",
    }
)
_STATUS_PHRASE_MARKERS = frozenset(
    {
        "active", "canceled", "cancelled", "closed", "completed", "confirmed",
        "delayed", "expired", "inactive", "open", "outdated", "pending", "planned",
        "postponed", "proposed", "reported", "resolved", "suspended", "verified",
    }
)
_CLAUSE_ACTION_VERBS = _EVENT_OUTCOME_VERBS | frozenset(
    {
        "approved", "entered", "enter", "get", "gets", "jog", "jogs", "moved",
        "move", "opened", "secure", "visited", "visit", "worked", "works",
    }
)
_PERSON_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*\.?", re.UNICODE)
_PERSON_PARTICLES = frozenset(
    {"al", "bin", "da", "de", "del", "der", "di", "du", "la", "le", "van", "von"}
)
_TOPIC_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "for", "from", "in", "is", "of", "on", "the", "to", "with"}
)

VALUE_TYPE_RULES = {
    "Person": "one person's name, not an activity, object, title, or full relation sentence",
    "GroupOrOrg": "a short group or organization name",
    "LocationOrScope": (
        "a short place/location noun phrase or an audience/access-scope noun phrase; "
        "the binding's semantic_value_type selects Location versus AudienceScope, so the two "
        "representations are never interchangeable"
    ),
    "DateOrTime": "a concrete date, day, or clock time",
    "Event": (
        "a short event expression whose semantic_value_type distinguishes an EventName from an "
        "EventOutcome or StatusPhrase"
    ),
    "ObjectOrCommonNoun": "a short concrete object or common-noun phrase, not a relation sentence",
    "AttributeValue": "a short attribute value or state noun phrase",
    "ProperNounOrNamedEntity": "a short named record, plan, place, project, or other proper entity",
    "Number": "a JSON integer, not a numeric string",
    "YesNo": "a JSON boolean",
    "Condition": "a short condition phrase",
}

SEMANTIC_VALUE_RULES = {
    "Location": (
        "a place, venue, geographic region, room, or locative noun phrase; never an audience "
        "scope, action, outcome, or status phrase"
    ),
    "AudienceScope": (
        "an audience or access class such as an authorized group; never a physical or virtual "
        "destination"
    ),
    "EventName": (
        "the name of an activity or occurrence, expressed as a noun phrase; never what happened "
        "to an event or its current status"
    ),
    "EventOutcome": "an outcome or consequence phrase, not an event name",
    "StatusPhrase": "a state or lifecycle-status phrase, not an event name",
}

SEMANTIC_VALUE_EXAMPLES = {
    "Location": {
        "valid": ["Cedar Hall", "London", "the Midwest", "community gym"],
        "invalid": ["members only", "local writers group", "pre-season", "cooking show", "Private"],
    },
    "AudienceScope": {
        "valid": ["members only", "authorized users", "Private", "registered guests"],
        "invalid": ["Cedar Hall", "London", "community gym"],
    },
    "EventName": {
        "valid": ["book launch", "gaming party", "charity concert", "Autumn Symposium"],
        "invalid": ["faced a setback", "signed a deal", "was postponed"],
    },
    "EventOutcome": {
        "valid": ["faced a setback", "schedule conflict", "project success"],
        "invalid": ["book launch", "gaming party"],
    },
    "StatusPhrase": {
        "valid": ["was postponed", "currently canceled", "on hold"],
        "invalid": ["book launch", "gaming party"],
    },
}

_SEMANTIC_VALUE_BASE_TYPES = {
    "Location": frozenset({"LocationOrScope"}),
    "AudienceScope": frozenset({"LocationOrScope"}),
    "EventName": frozenset({"Event"}),
    "EventOutcome": frozenset({"Event", "AttributeValue"}),
    "StatusPhrase": frozenset({"Event", "AttributeValue"}),
}

_SLOT_SEMANTIC_VALUE_TYPES = {
    "destination": "Location",
    "access_scope": "AudienceScope",
    "audience_scope": "AudienceScope",
    "event_outcome": "EventOutcome",
    "event_status": "StatusPhrase",
    "scheduled_event": "EventName",
    "visibility_scope": "AudienceScope",
}


class TruthConstructionError(ValueError):
    """Raised when a target or content binding cannot realize a fixed truth skeleton."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _fold(value: Any) -> str:
    return re.sub(r"[\W_]+", " ", _text(value).casefold(), flags=re.UNICODE).strip()


def _topic_terms(*values: Any) -> list[str]:
    terms: list[str] = []
    for value in values:
        for token in _fold(value).split():
            if len(token) >= 3 and token not in _TOPIC_STOPWORDS and token not in terms:
                terms.append(token)
    return terms


def _ref(binding_id: str) -> dict[str, str]:
    return {"$binding": binding_id}


def _binding(
    binding_id: str,
    value_type: str,
    semantic_role: str,
    *,
    distinct_group: str | None = None,
    enum: list[Any] | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
    semantic_value_type: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "binding_id": binding_id,
        "value_type": value_type,
        "semantic_role": semantic_role,
        "api_writable": True,
    }
    if semantic_value_type is None and value_type == "Event":
        semantic_value_type = "EventName"
    if semantic_value_type is not None:
        allowed_base_types = _SEMANTIC_VALUE_BASE_TYPES.get(semantic_value_type)
        if allowed_base_types is None or value_type not in allowed_base_types:
            raise TruthConstructionError(
                f"semantic value type {semantic_value_type!r} is incompatible with {value_type!r}"
            )
        row["semantic_value_type"] = semantic_value_type
    if distinct_group:
        row["distinct_group"] = distinct_group
    if enum is not None:
        row["enum"] = copy.deepcopy(enum)
    if minimum is not None:
        row["minimum"] = minimum
    if maximum is not None:
        row["maximum"] = maximum
    return row


def _type_key(value_type: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value_type).casefold()


def _event(
    event_id: str,
    role: str,
    entity: Any,
    slot: str,
    value: Any,
    *,
    status: str = "active",
    time_index: int,
    depends_on: list[str] | None = None,
    previous_value: Any = None,
    claim_edge_role: str = "semantic",
    qa_eligible: bool = True,
    answer_bearing: bool = False,
    answer_role: str = "supporting_fact",
    dependency_edge_role: str | None = None,
    required_for_dialogue: bool = True,
    optional: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": event_id,
        "constructor_role": role,
        "entity": copy.deepcopy(entity),
        "slot": slot,
        "value": copy.deepcopy(value),
        "status": status,
        "time_index": time_index,
        "depends_on": copy.deepcopy(depends_on or []),
        "required_for_dialogue": required_for_dialogue,
        "claim_edge_role": claim_edge_role,
        "qa_eligible": qa_eligible,
        "answer_bearing": answer_bearing,
        "answer_role": answer_role,
    }
    if dependency_edge_role is not None:
        row["dependency_edge_role"] = dependency_edge_role
    if previous_value is not None:
        row["previous_value"] = copy.deepcopy(previous_value)
    if optional:
        row.update(copy.deepcopy(optional))
    return row


def _apply_slot_semantic_value_types(
    bindings: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Refine broad API value types from locally frozen slot usage.

    ``LocationOrScope`` and ``Event`` are Stage64 answer families, not proof
    that every member of those unions is valid in every slot.  The local
    constructor owns the slot, so it also owns this compositional refinement.
    """

    specs = {spec["binding_id"]: spec for spec in bindings}
    for event in events:
        value_ref = event.get("value")
        if not isinstance(value_ref, dict) or set(value_ref) != {"$binding"}:
            continue
        semantic_value_type = _SLOT_SEMANTIC_VALUE_TYPES.get(_text(event.get("slot")))
        if semantic_value_type is None:
            continue
        binding_id = _text(value_ref.get("$binding"))
        spec = specs.get(binding_id)
        if spec is None:
            raise TruthConstructionError(
                f"event references unknown content binding: {binding_id!r}"
            )
        allowed_base_types = _SEMANTIC_VALUE_BASE_TYPES[semantic_value_type]
        if spec["value_type"] not in allowed_base_types:
            raise TruthConstructionError(
                f"slot {event['slot']!r} requires {semantic_value_type}, not "
                f"{spec['value_type']}"
            )
        previous = spec.get("semantic_value_type")
        if previous not in (None, semantic_value_type):
            raise TruthConstructionError(
                f"binding {binding_id!r} has incompatible semantic slot uses: "
                f"{previous!r} and {semantic_value_type!r}"
            )
        spec["semantic_value_type"] = semantic_value_type


def _answer_slot(answer_type: str, operation: str) -> str:
    if operation == "cause_consequence" and answer_type == "Person":
        return "replacement_host"
    mapping = {
        "Person": "assigned_partner",
        "GroupOrOrg": "affiliated_group",
        "LocationOrScope": "destination",
        "DateOrTime": "appointment_date",
        "Event": "scheduled_event",
        "ObjectOrCommonNoun": "selected_item",
        "AttributeValue": "assigned_attribute",
        "ProperNounOrNamedEntity": "linked_entity",
        "Number": "metric_count",
        "YesNo": "approval_flag",
    }
    return mapping[answer_type]


def _target_contract(target: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(target, dict):
        raise TruthConstructionError("stage64 target must be an object")
    operation = _text(target.get("qa_operation"))
    answer_type = _text(target.get("preferred_answer_type"))
    if operation not in OPERATIONS:
        raise TruthConstructionError(f"unsupported truth constructor operation: {operation!r}")
    if answer_type not in ANSWER_TYPES:
        raise TruthConstructionError(f"unsupported preferred answer type: {answer_type!r}")
    if operation in {"comparison", "count"} and answer_type != "Number":
        raise TruthConstructionError(f"{operation} requires preferred_answer_type=Number")
    if operation == "visibility" and answer_type != "Person":
        raise TruthConstructionError("visibility requires preferred_answer_type=Person")
    if operation == "bridge" and answer_type == "YesNo":
        raise TruthConstructionError("bridge cannot use preferred_answer_type=YesNo")
    for field in (
        "reasoning_subtype",
        "reasoning_template",
        "hop_count",
        "hard_negative_strategy",
    ):
        if target.get(field) in (None, ""):
            raise TruthConstructionError(f"stage64 target is missing {field}")
    return operation, answer_type


def _context_bindings() -> list[dict[str, Any]]:
    return [
        _binding("context_named_entity_1", "ProperNounOrNamedEntity", "independent named context entity 1"),
        _binding("context_object_1", "ObjectOrCommonNoun", "independent context object 1"),
        _binding("context_named_entity_2", "ProperNounOrNamedEntity", "independent named context entity 2"),
        _binding("context_event_name_2", "Event", "independent context EventName 2"),
    ]


def _context_events(start_index: int = 7) -> list[dict[str, Any]]:
    return [
        _event(
            f"E{start_index}",
            "independent_context_object",
            _ref("context_named_entity_1"),
            "selected_item",
            _ref("context_object_1"),
            time_index=start_index,
        ),
        _event(
            f"E{start_index + 1}",
            "independent_context_event",
            _ref("context_named_entity_2"),
            "scheduled_event",
            _ref("context_event_name_2"),
            time_index=start_index + 1,
        ),
    ]


def _temporal_constructor(target: dict[str, Any], answer_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    type_key = {
        "LocationOrScope": "location",
        "Event": "event_name",
    }.get(answer_type, _type_key(answer_type))
    temporal_value_ids = [
        f"temporal_candidate_{index}_{type_key}" for index in range(1, 5)
    ]
    bindings = [
        _binding("temporal_subject_named_entity", "ProperNounOrNamedEntity", "named entity whose locally controlled status history is recorded"),
        *[
            _binding(
                binding_id,
                answer_type,
                f"temporal candidate {index}: one {answer_type} content value; never a status label",
                distinct_group="state_values",
            )
            for index, binding_id in enumerate(temporal_value_ids, start=1)
        ],
        _binding("context_named_entity_1", "ProperNounOrNamedEntity", "named context entity 1"),
        _binding("context_object_1", "ObjectOrCommonNoun", "context object 1"),
        _binding("context_named_entity_2", "ProperNounOrNamedEntity", "named context entity 2"),
        _binding("context_event_name_2", "Event", "context EventName 2"),
        _binding("context_named_entity_3", "ProperNounOrNamedEntity", "named context entity 3"),
        _binding("context_location_3", "LocationOrScope", "short context Location 3"),
        _binding("context_named_entity_4", "ProperNounOrNamedEntity", "named context entity 4"),
        _binding("context_person_4", "Person", "one person's name for context relation 4"),
    ]
    strategy = _text(target.get("hard_negative_strategy"))
    anti_shortcut = bool(target.get("forbid_max_time_shortcut"))
    if anti_shortcut:
        strategy_status = {
            "same_slot_canceled_value": "canceled",
            "same_slot_outdated_value": "outdated",
            "same_slot_report_or_belief_value": "report",
        }.get(strategy)
        if strategy_status is None:
            raise TruthConstructionError(
                f"unsupported temporal hard-negative strategy: {strategy!r}"
            )
        statuses = ["belief", "active", "outdated", strategy_status]
        gold_event_id = "E2"
    else:
        statuses = ["belief", "outdated", "canceled", "active"]
        gold_event_id = "E4"
    slot = _answer_slot(answer_type, "temporal_state_update")
    events = []
    for index in range(1, 5):
        events.append(
            _event(
                f"E{index}",
                "temporal_state" if f"E{index}" != gold_event_id else "temporal_current_gold",
                _ref("temporal_subject_named_entity"),
                slot,
                _ref(temporal_value_ids[index - 1]),
                status=statuses[index - 1],
                time_index=index,
                depends_on=[f"E{index - 1}"] if index > 1 else [],
                previous_value=_ref(temporal_value_ids[index - 2]) if index > 1 else None,
                answer_bearing=f"E{index}" == gold_event_id,
                answer_role=(
                    "gold_source" if f"E{index}" == gold_event_id else "status_contrast"
                ),
                dependency_edge_role="control" if index > 1 else None,
            )
        )
    events.extend(
        [
            _event("E5", "context_object", _ref("context_named_entity_1"), "selected_item", _ref("context_object_1"), time_index=5),
            _event("E6", "context_event", _ref("context_named_entity_2"), "scheduled_event", _ref("context_event_name_2"), time_index=6),
            _event("E7", "context_location", _ref("context_named_entity_3"), "destination", _ref("context_location_3"), time_index=7),
            _event("E8", "context_person", _ref("context_named_entity_4"), "assigned_partner", _ref("context_person_4"), time_index=8),
        ]
    )
    proof = {
        "gold_projection": {"kind": "event_value", "event_id": gold_event_id},
        "reasoning_event_ids": ["E1", "E2", "E3", "E4"],
        "required_event_ids": ["E1", "E2", "E3", "E4"],
        "answer_type": answer_type,
        "derivation_rule": "resolve_status_gated_update_chain",
        "forbid_max_time_shortcut": anti_shortcut,
    }
    return bindings, events, proof


def _bridge_constructor(target: dict[str, Any], answer_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    slot = _answer_slot(answer_type, "bridge")
    bindings = [
        _binding("binder_record", "ProperNounOrNamedEntity", "record whose person-valued field starts the join"),
        _binding("bound_person", "Person", "person selected by the binder"),
        _binding("bridge_answer", answer_type, "two-hop answer", distinct_group="bridge_options"),
        _binding("bridge_negative_a", answer_type, "one-hop-preserving alternative A", distinct_group="bridge_options"),
        _binding("bridge_negative_b", answer_type, "typed alternative B", distinct_group="bridge_options"),
        _binding("bridge_negative_c", answer_type, "typed alternative C", distinct_group="bridge_options"),
        _binding("terminal_record", "ProperNounOrNamedEntity", "join verification record"),
        _binding("negative_entity_b", "ProperNounOrNamedEntity", "alternative entity B"),
        _binding("negative_entity_c", "ProperNounOrNamedEntity", "alternative entity C"),
        *_context_bindings(),
    ]
    events = [
        _event("E1", "binder_support", _ref("binder_record"), "selected_person", _ref("bound_person"), time_index=1),
        _event(
            "E2",
            "answer_support",
            _ref("bound_person"),
            slot,
            _ref("bridge_answer"),
            time_index=2,
            answer_bearing=True,
            answer_role="gold_source",
        ),
        _event(
            "E3",
            "two_support_join_control",
            _ref("terminal_record"),
            "proof_status",
            "verified",
            time_index=3,
            depends_on=["E1", "E2"],
            claim_edge_role="control",
            qa_eligible=False,
            dependency_edge_role="control",
            required_for_dialogue=False,
        ),
        _event("E4", "one_hop_preserving_negative", _ref("terminal_record"), slot, _ref("bridge_negative_a"), status="outdated", time_index=4),
        _event("E5", "typed_negative", _ref("negative_entity_b"), slot, _ref("bridge_negative_b"), status="report", time_index=5),
        _event("E6", "typed_negative", _ref("negative_entity_c"), slot, _ref("bridge_negative_c"), status="canceled", time_index=6),
        *_context_events(7),
    ]
    proof = {
        "gold_projection": {"kind": "event_value", "event_id": "E2"},
        "reasoning_event_ids": ["E1", "E2"],
        "required_event_ids": ["E1", "E2", "E3"],
        "answer_type": answer_type,
        "derivation_rule": "bind_e1_value_to_e2_entity_then_read_e2_value",
        "question_contract": {
            "name_event_ids": ["E1"],
            "forbidden_surface_event_ids": ["E2"],
            "must_not_name_binding_ids": ["bound_person", "bridge_answer"],
        },
        "gold_source_event_id": "E2",
        "proof_connector_event_id": "E3",
    }
    return bindings, events, proof


def _comparison_constructor(_target: dict[str, Any], _answer_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bindings = [
        _binding("left_entity", "Person", "left comparison entity"),
        _binding("right_entity", "Person", "right comparison entity", distinct_group="comparison_entities"),
        _binding("left_value", "Number", "left current metric", minimum=1, maximum=100),
        _binding("right_value", "Number", "right current metric", minimum=1, maximum=100),
        _binding("left_history", "Number", "left historical metric", minimum=1, maximum=100),
        _binding("right_history", "Number", "right historical metric", minimum=1, maximum=100),
        *_context_bindings(),
    ]
    # left_entity participates in the explicit entity distinctness check below.
    bindings[0]["distinct_group"] = "comparison_entities"
    events = [
        _event("E1", "left_operand", _ref("left_entity"), "service_hours", _ref("left_value"), time_index=9, answer_bearing=True, answer_role="derived_gold_operand"),
        _event("E2", "right_operand", _ref("right_entity"), "service_hours", _ref("right_value"), time_index=9, answer_bearing=True, answer_role="derived_gold_operand"),
        _event("E3", "left_historical_contrast", _ref("left_entity"), "service_hours", _ref("left_history"), status="outdated", time_index=4),
        _event("E4", "right_historical_contrast", _ref("right_entity"), "service_hours", _ref("right_history"), status="outdated", time_index=4),
        _event("E5", "context_object", _ref("context_named_entity_1"), "selected_item", _ref("context_object_1"), time_index=5),
        _event("E6", "context_event", _ref("context_named_entity_2"), "scheduled_event", _ref("context_event_name_2"), time_index=6),
        _event("E7", "context_number", _ref("left_entity"), "badge_count", 2, time_index=7),
        _event("E8", "context_location", _ref("right_entity"), "destination", "North Hall", time_index=8),
    ]
    proof = {
        "gold_projection": {"kind": "absolute_difference", "event_ids": ["E1", "E2"]},
        "reasoning_event_ids": ["E1", "E2"],
        "required_event_ids": ["E1", "E2", "E3", "E4"],
        "answer_type": "Number",
        "derivation_rule": "absolute_difference",
        "binding_constraints": ["abs(primary_difference - historical_difference) == 1"],
    }
    return bindings, events, proof


def _visibility_event(
    event_id: str,
    role: str,
    *,
    owner: Any,
    resource: Any,
    observer: Any,
    action: str,
    condition: Any,
    status: str,
    time_index: int,
    depends_on: list[str] | None = None,
    answer_bearing: bool = False,
    answer_role: str = "typed_scope_contrast",
) -> dict[str, Any]:
    return _event(
        event_id,
        role,
        resource,
        "access_level",
        "Private",
        status=status,
        time_index=time_index,
        depends_on=depends_on,
        answer_bearing=answer_bearing,
        answer_role=answer_role,
        dependency_edge_role="control" if depends_on else None,
        optional={
            "event_type": "visibility_permission",
            "owner": owner,
            "resource": resource,
            "observer": observer,
            "action": action,
            "condition": condition,
            "access_class": "Private",
            "scope": {
                "observer": observer,
                "permission": action,
                "condition": condition,
            },
        },
    )


def _visibility_constructor(_target: dict[str, Any], _answer_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bindings = [
        _binding("owner", "Person", "resource owner"),
        _binding("resource", "ObjectOrCommonNoun", "protected resource"),
        *[
            _binding(f"observer_{index}", "Person", f"observer {index}", distinct_group="visibility_observers")
            for index in range(1, 6)
        ],
        _binding("condition_primary", "Condition", "gold permission condition"),
        _binding("condition_wrong", "Condition", "wrong condition", distinct_group="visibility_conditions"),
        _binding("condition_extra", "Condition", "additional wrong condition", distinct_group="visibility_conditions"),
    ]
    # Include the primary condition in the same distinctness group.
    bindings[-3]["distinct_group"] = "visibility_conditions"
    events = [
        _event("E1", "protected_resource", _ref("resource"), "resource_owner", _ref("owner"), time_index=1),
        _visibility_event("E2", "gold_permission", owner=_ref("owner"), resource=_ref("resource"), observer=_ref("observer_1"), action="view", condition=_ref("condition_primary"), status="active", time_index=2, depends_on=["E1"], answer_bearing=True, answer_role="gold_source"),
        _visibility_event("E3", "wrong_observer", owner=_ref("owner"), resource=_ref("resource"), observer=_ref("observer_2"), action="view", condition=_ref("condition_primary"), status="scope_limited", time_index=3),
        _visibility_event("E4", "wrong_permission", owner=_ref("owner"), resource=_ref("resource"), observer=_ref("observer_1"), action="edit", condition=_ref("condition_primary"), status="scope_limited", time_index=4),
        _visibility_event("E5", "wrong_condition", owner=_ref("owner"), resource=_ref("resource"), observer=_ref("observer_1"), action="view", condition=_ref("condition_wrong"), status="scope_limited", time_index=5),
        _visibility_event("E6", "extra_scope_negative", owner=_ref("owner"), resource=_ref("resource"), observer=_ref("observer_3"), action="edit", condition=_ref("condition_wrong"), status="scope_limited", time_index=6),
        _visibility_event("E7", "extra_scope_negative", owner=_ref("owner"), resource=_ref("resource"), observer=_ref("observer_4"), action="view", condition=_ref("condition_extra"), status="scope_limited", time_index=7),
        _visibility_event("E8", "extra_scope_negative", owner=_ref("owner"), resource=_ref("resource"), observer=_ref("observer_5"), action="edit", condition=_ref("condition_extra"), status="scope_limited", time_index=8),
    ]
    proof = {
        "gold_projection": {"kind": "event_field", "event_id": "E2", "field": "observer"},
        "reasoning_event_ids": ["E1", "E2"],
        "required_event_ids": ["E1", "E2"],
        "answer_type": "Person",
        "derivation_rule": "resolve_observer_under_permission_scope",
    }
    return bindings, events, proof


def _count_constructor(_target: dict[str, Any], _answer_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bindings = [
        _binding("set_entity", "GroupOrOrg", "candidate set owner"),
        _binding("set_id", "ProperNounOrNamedEntity", "candidate set name"),
        _binding("membership_condition", "Condition", "explicit membership predicate"),
        *[
            _binding(f"member_{index}", "Person", f"candidate member {index}", distinct_group="count_members")
            for index in range(1, 9)
        ],
    ]
    included = [True, False, True, True, False, True, False, False]
    cohorts = ["morning", "morning", "morning", "morning", "afternoon", "afternoon", "afternoon", "afternoon"]
    events = [
        _event(
            f"E{index}",
            "included_member" if included[index - 1] else "excluded_member",
            _ref("set_entity"),
            "candidate_member",
            _ref(f"member_{index}"),
            time_index=index,
            optional={
                "set_membership": {
                    "set_id": _ref("set_id"),
                    "condition": _ref("membership_condition"),
                    "included": included[index - 1],
                    "cohort": cohorts[index - 1],
                }
            },
            answer_bearing=True,
            answer_role="derived_count_operand",
        )
        for index in range(1, 9)
    ]
    proof = {
        "gold_projection": {"kind": "count_membership", "included": True},
        "reasoning_event_ids": [f"E{index}" for index in range(1, 9)],
        "required_event_ids": [f"E{index}" for index in range(1, 9)],
        "answer_type": "Number",
        "derivation_rule": "filter_explicit_membership_claims_then_count",
        "membership_source": "events[*].set_membership.included",
        "cohort_source": "events[*].set_membership.cohort",
        "array_position_is_truth_forbidden": True,
    }
    return bindings, events, proof


def _cause_constructor(_target: dict[str, Any], answer_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    slot = _answer_slot(answer_type, "cause_consequence")
    bindings = [
        _binding("cause_event", "Event", "earlier causal event"),
        _binding("cause_reason", "AttributeValue", "causal trigger"),
        _binding("effect_entity", "ProperNounOrNamedEntity", "affected plan or activity"),
        _binding("effect_value", answer_type, "caused result", distinct_group="cause_options"),
        _binding("wrong_effect_a", answer_type, "adjacent non-causal alternative", distinct_group="cause_options"),
        _binding("wrong_effect_b", answer_type, "wrong effect alternative", distinct_group="cause_options"),
        _binding("wrong_effect_c", answer_type, "second wrong effect alternative", distinct_group="cause_options"),
        _binding("other_entity", "ProperNounOrNamedEntity", "unrelated adjacent entity"),
        _binding("other_entity_b", "ProperNounOrNamedEntity", "second unrelated entity"),
        *_context_bindings(),
    ]
    events = [
        _event("E1", "directed_cause", _ref("cause_event"), "trigger_reason", _ref("cause_reason"), time_index=1),
        _event(
            "E2",
            "directed_effect",
            _ref("effect_entity"),
            slot,
            _ref("effect_value"),
            time_index=2,
            depends_on=["E1"],
            answer_bearing=True,
            answer_role="gold_source",
            dependency_edge_role="control",
        ),
        _event("E3", "adjacent_non_cause", _ref("effect_entity"), slot, _ref("wrong_effect_a"), status="outdated", time_index=3),
        _event("E4", "wrong_effect", _ref("other_entity"), slot, _ref("wrong_effect_b"), status="report", time_index=4),
        _event("E5", "wrong_effect", _ref("other_entity_b"), slot, _ref("wrong_effect_c"), status="canceled", time_index=5),
        _event("E6", "context_object", _ref("context_named_entity_1"), "selected_item", _ref("context_object_1"), time_index=6),
        _event("E7", "context_event", _ref("context_named_entity_2"), "scheduled_event", _ref("context_event_name_2"), time_index=7),
        _event("E8", "context_location", _ref("cause_event"), "destination", "South Hall", time_index=8),
    ]
    proof = {
        "gold_projection": {"kind": "event_value", "event_id": "E2"},
        "reasoning_event_ids": ["E1", "E2"],
        "required_event_ids": ["E1", "E2"],
        "answer_type": answer_type,
        "derivation_rule": "follow_directed_cause_to_effect",
        "required_relation": {"predicate": "caused", "source_event_id": "E1", "target_event_id": "E2"},
    }
    return bindings, events, proof


def _relation_constructor(_target: dict[str, Any], answer_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    slot = _answer_slot(answer_type, "relation_lookup")
    bindings = [
        _binding("binder_record", "ProperNounOrNamedEntity", "entity-selection record"),
        _binding("bound_person", "Person", "selected relation subject"),
        _binding("relation_answer", answer_type, "bound relation value", distinct_group="relation_options"),
        _binding("same_entity_negative", answer_type, "same-entity relation alternative", distinct_group="relation_options"),
        _binding("wrong_entity_answer", answer_type, "wrong-entity relation value", distinct_group="relation_options"),
        _binding("wrong_entity_answer_b", answer_type, "second wrong-entity relation value", distinct_group="relation_options"),
        _binding("other_person", "Person", "wrong relation subject"),
        _binding("other_person_b", "Person", "second wrong relation subject"),
        *_context_bindings(),
    ]
    events = [
        _event("E1", "entity_binder", _ref("binder_record"), "selected_person", _ref("bound_person"), time_index=1),
        _event(
            "E2",
            "bound_relation_gold",
            _ref("bound_person"),
            slot,
            _ref("relation_answer"),
            time_index=2,
            depends_on=["E1"],
            answer_bearing=True,
            answer_role="gold_source",
            dependency_edge_role="semantic",
        ),
        _event("E3", "same_entity_relation_negative", _ref("bound_person"), slot, _ref("same_entity_negative"), status="outdated", time_index=3),
        _event("E4", "same_relation_wrong_entity", _ref("other_person"), slot, _ref("wrong_entity_answer"), time_index=4),
        _event("E5", "same_relation_wrong_entity", _ref("other_person_b"), slot, _ref("wrong_entity_answer_b"), time_index=5),
        _event("E6", "context_object", _ref("context_named_entity_1"), "selected_item", _ref("context_object_1"), time_index=6),
        _event("E7", "context_event", _ref("context_named_entity_2"), "scheduled_event", _ref("context_event_name_2"), time_index=7),
        _event("E8", "context_count", _ref("bound_person"), "badge_count", 2, time_index=8),
    ]
    proof = {
        "gold_projection": {"kind": "event_value", "event_id": "E2"},
        "reasoning_event_ids": ["E1", "E2"],
        "required_event_ids": ["E1", "E2"],
        "answer_type": answer_type,
        "derivation_rule": "bind_entity_then_read_fixed_relation",
    }
    return bindings, events, proof


_CONSTRUCTORS: dict[
    str,
    Callable[
        [dict[str, Any], str],
        tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    ],
] = {
    "temporal_state_update": _temporal_constructor,
    "bridge": _bridge_constructor,
    "comparison": _comparison_constructor,
    "visibility": _visibility_constructor,
    "count": _count_constructor,
    "cause_consequence": _cause_constructor,
    "relation_lookup": _relation_constructor,
}


_CONTINUITY_SLOT_FAMILY_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "visibility",
        frozenset({"access", "observer", "permission", "scope", "visibility"}),
    ),
    (
        "membership",
        frozenset(
            {
                "candidate", "church", "club", "forum", "group", "joined",
                "member", "membership", "organization", "roster", "team",
            }
        ),
    ),
    (
        "location",
        frozenset(
            {
                "address", "beach", "coast", "coastline", "destination",
                "hiking", "location", "office", "photoshoot", "residence",
                "room", "travel", "trip", "venue", "visited", "went",
            }
        ),
    ),
    (
        "event_plan",
        frozenset(
            {
                "activity", "appointment", "attendance", "business", "class",
                "community", "creative", "education", "employment", "event",
                "goal", "hosted", "life", "meeting", "party", "plan",
                "project", "scheduled", "service", "show", "workshop",
            }
        ),
    ),
    (
        "person_relation",
        frozenset(
            {
                "assigned", "chats", "companion", "family", "friend", "host",
                "partner", "person", "relationship", "relies", "selected", "wife",
            }
        ),
    ),
    (
        "resource",
        frozenset(
            {
                "certificate", "equipment", "gear", "item", "object", "owner",
                "pet", "plate", "possession", "resource", "tattoo",
            }
        ),
    ),
    (
        "metric",
        frozenset({"count", "frequency", "hours", "metric", "number", "score", "total"}),
    ),
    (
        "cause",
        frozenset(
            {
                "cause", "caused", "challenge", "consequence", "effect", "failure",
                "inspired", "outcome", "reason", "setback", "trigger",
            }
        ),
    ),
    (
        "status",
        frozenset({"confirmation", "health", "state", "status", "wellbeing"}),
    ),
)


def _continuity_slot_family(slot: Any) -> str | None:
    tokens = set(_fold(slot).split())
    if not tokens:
        return None
    if (
        {"service", "hours"}.issubset(tokens)
        or {"community", "service"}.issubset(tokens)
        or ("volunteer" in tokens and tokens.intersection({"frequency", "hours"}))
    ):
        return "community_service_metric"
    for family, keywords in _CONTINUITY_SLOT_FAMILY_KEYWORDS:
        if tokens.intersection(keywords):
            return family
    return None


def _binding_reference(value: Any) -> str | None:
    if isinstance(value, dict) and set(value) == {"$binding"}:
        binding_id = _text(value.get("$binding"))
        return binding_id or None
    return None


def _continuity_child_binding_fields(
    event: dict[str, Any],
) -> list[tuple[str, str]]:
    slot = _fold(event.get("slot"))
    if slot == "resource owner":
        preferred_fields = ["value"]
    elif slot == "selected person":
        preferred_fields = ["value"]
    else:
        preferred_fields = (
        ["entity", "owner", "observer", "resource", "value"]
        if slot
        in {
            "assigned partner",
            "candidate member",
            "replacement host",
        }
        else ["entity", "owner", "observer", "resource"]
        )
    result: list[tuple[str, str]] = []
    for field in preferred_fields:
        binding_id = _binding_reference(event.get(field))
        if binding_id is not None and (field, binding_id) not in result:
            result.append((field, binding_id))
    return result


def _continuity_parent_identities(event: dict[str, Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    family = _continuity_slot_family(event.get("slot") or event.get("event_type"))
    fields = ["entity", "owner", "observer", "resource"]
    if family in {"membership", "person_relation"}:
        fields.append("value")
    for field in fields:
        value = event.get(field)
        if not isinstance(value, str) or not _atom_value(value, max_words=12):
            continue
        normalized = _fold(value)
        if normalized and all(normalized != _fold(existing) for _field, existing in result):
            result.append((field, _text(value)))
    return result


def _select_parent_continuity_anchor(
    *,
    operation: str,
    bindings: list[dict[str, Any]],
    events: list[dict[str, Any]],
    proof: dict[str, Any],
    parent_anchor_facts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select and freeze one independently checkable parent/child semantic link."""

    if not parent_anchor_facts:
        return None
    specs = {spec["binding_id"]: spec for spec in bindings}
    required_ids = set(proof.get("required_event_ids") or [])
    candidates: list[
        tuple[tuple[int, int, int, int, int, int], dict[str, Any]]
    ] = []
    status_rank = {
        "active": 5,
        "planned": 4,
        "completed": 3,
        "historical": 2,
        "reported": 1,
        "outdated": 0,
        "expired": 0,
        "canceled": 0,
        "cancelled": 0,
    }
    for parent_index, parent in enumerate(parent_anchor_facts):
        parent_slot = _fold(parent.get("slot") or parent.get("event_type"))
        parent_family = _continuity_slot_family(parent_slot)
        if parent_family is None:
            continue
        identities = _continuity_parent_identities(parent)
        if not identities:
            continue
        raw_time = parent.get("global_time_index")
        if raw_time is None:
            raw_time = parent.get("time_index")
        try:
            parent_time = int(raw_time)
        except (TypeError, ValueError):
            parent_time = -1
        if operation == "temporal_state_update" and parent_time < 0:
            continue
        for child_index, child in enumerate(events):
            if child.get("event_id") not in required_ids:
                continue
            if child.get("qa_eligible") is not True:
                continue
            child_slot = _fold(child.get("slot") or child.get("event_type"))
            child_family = _continuity_slot_family(child_slot)
            exact_slot = bool(parent_slot and parent_slot == child_slot)
            if not exact_slot and (
                parent_family is None or parent_family != child_family
            ):
                continue
            for child_field, binding_id in _continuity_child_binding_fields(child):
                spec = specs.get(binding_id)
                if spec is None:
                    continue
                for parent_field, identity in identities:
                    if not _valid_typed_value(identity, spec["value_type"]):
                        continue
                    semantic_type = spec.get("semantic_value_type")
                    if semantic_type and not _valid_semantic_value(
                        identity, semantic_type
                    ):
                        continue
                    score = (
                        1 if exact_slot else 0,
                        status_rank.get(_fold(parent.get("status")), 1),
                        parent_time,
                        1 if child_field != "value" else 0,
                        -parent_index,
                        -child_index,
                    )
                    candidates.append(
                        (
                            score,
                            {
                                "parent_event_id": parent["event_id"],
                                "parent_identity_field": parent_field,
                                "parent_identity": identity,
                                "parent_slot": parent_slot,
                                "parent_slot_family": parent_family,
                                "child_event_id": child["event_id"],
                                "child_identity_field": child_field,
                                "child_binding_id": binding_id,
                                "child_slot": child_slot,
                                "child_slot_family": child_family,
                                "relation": (
                                    "same_identity_exact_slot_extension"
                                    if exact_slot
                                    else "same_identity_slot_family_extension"
                                ),
                            },
                        )
                    )
    if not candidates:
        raise TruthConstructionError(
            "selected parent events contain no type-compatible semantic continuity "
            "anchor for the required child reasoning events"
        )
    _score, selected = max(candidates, key=lambda item: item[0])
    spec = specs[selected["child_binding_id"]]
    spec["enum"] = [selected["parent_identity"]]
    spec["continuity_anchor"] = {
        "parent_event_id": selected["parent_event_id"],
        "child_event_id": selected["child_event_id"],
        "relation": selected["relation"],
    }
    return selected


def construct_truth_skeleton(
    target: dict[str, Any],
    *,
    graph_id: str,
    parent_graph_id: str,
    selected_parent_event_ids: list[str] | None = None,
    selected_parent_events: list[dict[str, Any]] | None = None,
    parent_max_global_time_index: int = 0,
    parent_max_session_index: int = 0,
) -> dict[str, Any]:
    """Construct one immutable operation truth skeleton without model output."""

    operation, answer_type = _target_contract(target)
    for name, value in (
        ("parent_max_global_time_index", parent_max_global_time_index),
        ("parent_max_session_index", parent_max_session_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TruthConstructionError(f"{name} must be a non-negative integer")
    bindings, events, proof = _CONSTRUCTORS[operation](target, answer_type)
    _apply_slot_semantic_value_types(bindings, events)
    selected_ids = list(
        dict.fromkeys(_text(item) for item in selected_parent_event_ids or [] if _text(item))
    )
    parent_anchor_facts: list[dict[str, Any]] = []
    for event in selected_parent_events or []:
        if not isinstance(event, dict):
            raise TruthConstructionError("selected_parent_events must contain objects")
        event_id = _text(event.get("event_id") or event.get("event_instance_id"))
        if not event_id:
            raise TruthConstructionError("selected parent anchor is missing event_id")
        if selected_ids and event_id not in selected_ids:
            continue
        anchor_fact = copy.deepcopy(event)
        anchor_fact.update(
            {
                "event_id": event_id,
                "entity": copy.deepcopy(event.get("entity")),
                "slot": copy.deepcopy(event.get("slot")),
                "value": copy.deepcopy(event.get("value")),
                "status": copy.deepcopy(event.get("status")),
                "time_index": copy.deepcopy(event.get("time_index")),
                "global_time_index": copy.deepcopy(event.get("global_time_index")),
                "session_index": copy.deepcopy(event.get("session_index")),
                "evidence_turn_ids": copy.deepcopy(event.get("evidence_turn_ids") or []),
                "node_type": copy.deepcopy(event.get("node_type")),
                "provenance": copy.deepcopy(event.get("provenance")),
            }
        )
        parent_anchor_facts.append(anchor_fact)
    if parent_anchor_facts and not selected_ids:
        selected_ids = [item["event_id"] for item in parent_anchor_facts]
    continuity_match = _select_parent_continuity_anchor(
        operation=operation,
        bindings=bindings,
        events=events,
        proof=proof,
        parent_anchor_facts=parent_anchor_facts,
    )
    if continuity_match is not None:
        anchor_id = continuity_match["parent_event_id"]
        parent_anchor_facts.sort(
            key=lambda item: 0 if item["event_id"] == anchor_id else 1
        )
    primary_anchor = parent_anchor_facts[0] if parent_anchor_facts else None
    anchor_terms = (
        _topic_terms(
            primary_anchor.get("entity"),
            primary_anchor.get("slot"),
            primary_anchor.get("value"),
        )
        if primary_anchor
        else []
    )
    skeleton = {
        "schema_version": SKELETON_SCHEMA_VERSION,
        "graph_id": _text(graph_id),
        "parent_graph_id": _text(parent_graph_id),
        "operation": operation,
        "immutable_stage64_target": copy.deepcopy(target),
        "preferred_answer_type": answer_type,
        "binding_specs": bindings,
        "event_templates": events,
        "proof_contract": proof,
        "selected_parent_event_ids": selected_ids,
        "parent_anchor_contract": {
            "primary_parent_event_id": primary_anchor.get("event_id") if primary_anchor else None,
            "anchor_facts": parent_anchor_facts,
            "topic_terms": anchor_terms,
            "required_relation": (
                continuity_match["relation"]
                if continuity_match is not None
                else "continues_topic"
            ),
            "continuity_match": copy.deepcopy(continuity_match),
            "entity_overlap_alone_is_insufficient": True,
        },
        "timeline_contract": {
            "parent_max_global_time_index": parent_max_global_time_index,
            "parent_max_session_index": parent_max_session_index,
            "child_session_index": parent_max_session_index + 1,
            "rule": "child_global_time_starts_after_parent_max",
        },
        "api_writable_paths": [
            f"$.bindings.{item['binding_id']}" for item in bindings
        ],
        "api_forbidden_truth_fields": [
            "events",
            "slots",
            "status",
            "depends_on",
            "set_membership",
            "scope",
            "gold",
            "proof",
            "reasoning_path",
            "answer_role",
        ],
    }
    if not skeleton["graph_id"] or not skeleton["parent_graph_id"]:
        raise TruthConstructionError("truth skeleton requires graph_id and parent_graph_id")
    skeleton["skeleton_sha256"] = canonical_sha256(skeleton)
    return skeleton


def api_binding_contract(skeleton: dict[str, Any]) -> dict[str, Any]:
    """Return the only schema a content-generation API is allowed to fill."""

    _require_skeleton(skeleton)
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "skeleton_sha256": skeleton["skeleton_sha256"],
        "operation": skeleton["operation"],
        "bindings": {
            spec["binding_id"]: {
                # Expose the narrow slot-owned type as the primary API type.
                # The broader Stage64 answer family remains audit metadata,
                # but must not invite a LocationOrScope/Event union value
                # where the frozen slot requires Location/EventName.
                "value_type": spec.get("semantic_value_type", spec["value_type"]),
                "base_value_type": spec["value_type"],
                "semantic_role": spec["semantic_role"],
                "value_rule": (
                    SEMANTIC_VALUE_RULES[spec["semantic_value_type"]]
                    if "semantic_value_type" in spec
                    else VALUE_TYPE_RULES[spec["value_type"]]
                ),
                **(
                    {
                        "semantic_value_type": spec["semantic_value_type"],
                        "semantic_value_rule": SEMANTIC_VALUE_RULES[
                            spec["semantic_value_type"]
                        ],
                        "semantic_value_examples": copy.deepcopy(
                            SEMANTIC_VALUE_EXAMPLES[spec["semantic_value_type"]]
                        ),
                    }
                    if "semantic_value_type" in spec
                    else {}
                ),
                **(
                    {"enum": copy.deepcopy(spec["enum"])}
                    if "enum" in spec
                    else {}
                ),
                **(
                    {"minimum": spec["minimum"]}
                    if "minimum" in spec
                    else {}
                ),
                **(
                    {"maximum": spec["maximum"]}
                    if "maximum" in spec
                    else {}
                ),
            }
            for spec in skeleton["binding_specs"]
        },
        "forbidden_output_keys": copy.deepcopy(
            skeleton["api_forbidden_truth_fields"]
        ),
        "value_type_rules": copy.deepcopy(VALUE_TYPE_RULES),
        "semantic_value_type_rules": copy.deepcopy(SEMANTIC_VALUE_RULES),
        "semantic_value_type_examples": copy.deepcopy(SEMANTIC_VALUE_EXAMPLES),
        "output_schema": {
            "bindings": {
                spec["binding_id"]: (
                    f"{spec['semantic_role']}. MUST be "
                    f"{spec.get('semantic_value_type', spec['value_type'])}: "
                    + (
                        SEMANTIC_VALUE_RULES[spec["semantic_value_type"]]
                        if "semantic_value_type" in spec
                        else VALUE_TYPE_RULES[spec["value_type"]]
                    )
                    + (
                        f" BASE FAMILY {spec['value_type']}; SLOT-SPECIFIC TYPE "
                        f"{spec['semantic_value_type']}. Valid examples: "
                        f"{SEMANTIC_VALUE_EXAMPLES[spec['semantic_value_type']]['valid']}. "
                        f"Invalid examples: "
                        f"{SEMANTIC_VALUE_EXAMPLES[spec['semantic_value_type']]['invalid']}"
                        if "semantic_value_type" in spec
                        else ""
                    )
                    + (
                        f" ALLOWED VALUE IS EXACTLY {spec['enum']!r}; do not paraphrase it."
                        if "enum" in spec
                        else ""
                    )
                )
                for spec in skeleton["binding_specs"]
            },
            "discourse_frame": {
                "topic": "short natural topic label",
                "setting": "short natural setting label",
            },
        },
    }


def normalize_content_binding_envelope(
    skeleton: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach locally owned envelope fields when the API omits them.

    The model owns only ``bindings`` and ``discourse_frame``.  Schema and
    skeleton identity are frozen before the call, so an omitted envelope is
    unambiguous representation loss.  Explicit values are never overwritten:
    a supplied wrong schema or hash remains visible to the strict validator.
    """

    _require_skeleton(skeleton)
    if not isinstance(payload, dict):
        return copy.deepcopy(payload), {
            "schema_version": BINDING_ENVELOPE_NORMALIZER_VERSION,
            "used": False,
            "injected_fields": [],
            "reason": "binding_payload_not_object",
        }
    normalized = copy.deepcopy(payload)
    expected = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "skeleton_sha256": skeleton["skeleton_sha256"],
    }
    injected_fields: list[str] = []
    for field, value in expected.items():
        if field not in normalized:
            normalized[field] = value
            injected_fields.append(field)
    return normalized, {
        "schema_version": BINDING_ENVELOPE_NORMALIZER_VERSION,
        "used": bool(injected_fields),
        "injected_fields": injected_fields,
        "preserved_explicit_fields": sorted(set(expected).intersection(payload)),
        "api_writable_fields": ["bindings", "discourse_frame"],
        "truth_fields_modified": [],
    }


def _require_skeleton(skeleton: dict[str, Any]) -> None:
    if not isinstance(skeleton, dict) or skeleton.get("schema_version") != SKELETON_SCHEMA_VERSION:
        raise TruthConstructionError("invalid truth skeleton schema")
    expected = copy.deepcopy(skeleton)
    claimed = _text(expected.pop("skeleton_sha256", ""))
    if not claimed or canonical_sha256(expected) != claimed:
        raise TruthConstructionError("truth skeleton hash mismatch")


def _person_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    tokens = value.strip().split()
    if not 1 <= len(tokens) <= 5:
        return False
    for token in tokens:
        if _PERSON_TOKEN_RE.fullmatch(token) is None:
            return False
        parts = re.split(r"[-']", token.rstrip("."))
        if any(
            not part
            or (
                not part[0].isupper()
                and part.casefold() not in _PERSON_PARTICLES
            )
            for part in parts
        ):
            return False
    return True


def _atom_value(value: Any, *, max_words: int = 12) -> bool:
    if not isinstance(value, str):
        return False
    text = _text(value)
    return bool(
        text
        and len(text.split()) <= max_words
        and not _QA_LEAK_RE.search(text)
        and "\n" not in value
        and not text.endswith("?")
    )


def _semantic_words(value: Any) -> tuple[list[str], list[str]]:
    words = re.findall(
        r"[^\W\d_]+(?:[-'][^\W\d_]+)*",
        _text(value),
        flags=re.UNICODE,
    )
    return words, [word.casefold() for word in words]


def _audience_scope_value(value: Any) -> bool:
    if not _atom_value(value, max_words=8):
        return False
    _words, folded = _semantic_words(value)
    if not folded or any(token in _CONTROL_STATUS_ATOMS for token in folded):
        return False
    if len(folded) == 1 and folded[0] in _ACCESS_CLASS_ATOMS:
        return True
    if folded[-1:] == ["only"] and len(folded) >= 2:
        audience = folded[-2]
        modifiers = folded[:-2]
        return bool(
            audience in _SCOPE_AUDIENCE_HEADS
            and all(
                modifier not in _NP_CLAUSE_WORDS
                and modifier not in _CLAUSE_ACTION_VERBS
                for modifier in modifiers
            )
        )
    return bool(
        len(folded) >= 2
        and folded[-1] in _SCOPE_AUDIENCE_HEADS
        and all(token in _SCOPE_AUDIENCE_MODIFIERS for token in folded[:-1])
    )


def _status_phrase_value(value: Any) -> bool:
    if not _atom_value(value, max_words=8):
        return False
    _words, folded = _semantic_words(value)
    if not folded:
        return False
    joined = " ".join(folded)
    if len(folded) == 1 and folded[0] in (
        _CONTROL_STATUS_ATOMS | _STATUS_PHRASE_MARKERS
    ):
        return True
    if any(
        marker in joined
        for marker in ("called off", "in progress", "no longer", "on hold", "under review")
    ):
        return True
    return bool(
        set(folded).intersection(_STATUS_PHRASE_MARKERS)
        and (
            folded[0] in {"currently", "still"}
            or set(folded).intersection({"am", "are", "be", "been", "being", "is", "was", "were"})
        )
    )


def _event_outcome_value(value: Any) -> bool:
    if not _atom_value(value, max_words=8):
        return False
    _words, folded = _semantic_words(value)
    if not folded:
        return False
    if folded[0] in _EVENT_OUTCOME_VERBS:
        return True
    if folded[-1] in _EVENT_OUTCOME_HEADS:
        return True
    return bool(
        set(folded).intersection(_EVENT_OUTCOME_VERBS)
        and set(folded).intersection(_EVENT_OUTCOME_HEADS)
    )


def _location_core(words: list[str], folded: list[str]) -> bool:
    if not words:
        return False
    if len(folded) == 1 and folded[0] in _LOCATIVE_ATOMS:
        return True
    if (
        folded[-1] in _EVENT_NAME_HEADS
        and folded[-1] not in _LOCATION_HEADS
    ):
        return False
    geographic_words = list(words)
    geographic_folded = list(folded)
    if geographic_folded and geographic_folded[0] in _NP_DETERMINERS:
        geographic_words = geographic_words[1:]
        geographic_folded = geographic_folded[1:]
    if not geographic_words:
        return False
    geographic_key = _geographic_key(" ".join(geographic_words))
    geographic_core_key = geographic_key
    if len(geographic_folded) > 1 and geographic_folded[0] in _SPATIAL_MODIFIERS:
        geographic_core_key = _geographic_key(" ".join(geographic_words[1:]))
    if geographic_core_key in (
        _TIMEZONE_PLACE_NAMES
        | _US_STATE_NAMES
        | _CONTINENT_REGION_NAMES
        | _LOCATIVE_ATOMS
    ):
        return True
    if (
        len(geographic_words) == 1
        and geographic_words[0].casefold() in _LOCATION_ACRONYMS
        and geographic_words[0].isupper()
    ):
        return True
    if (
        len(geographic_words) == 1
        and any(
            geographic_folded[0].endswith(suffix)
            for suffix in _TOPONYM_SUFFIXES
        )
    ):
        return bool(geographic_words[0][0].isupper())
    if (
        len(geographic_words) >= 2
        and geographic_folded[0] in _GEOGRAPHIC_FEATURE_HEADS
        and all(
            word[0].isupper()
            or word.isupper()
            or token in {"and", "of", "the"}
            for word, token in zip(geographic_words[1:], geographic_folded[1:])
        )
    ):
        return True

    tokens = list(folded)
    if tokens and tokens[0] in _NP_DETERMINERS:
        tokens = tokens[1:]
    if not tokens or tokens[-1] not in _LOCATION_HEADS:
        return False
    modifiers = tokens[:-1]
    return not any(
        token in _NP_CLAUSE_WORDS
        or token in _NP_DETERMINERS
        or token in _CLAUSE_ACTION_VERBS
        for token in modifiers
    )


def _location_value(value: Any) -> bool:
    if not _atom_value(value, max_words=8):
        return False
    if (
        _audience_scope_value(value)
        or _event_outcome_value(value)
        or _status_phrase_value(value)
    ):
        return False
    words, folded = _semantic_words(value)
    if not words or any(token in _CONTROL_STATUS_ATOMS for token in folded):
        return False
    connector_indices = [
        index for index, token in enumerate(folded) if token in _LOCATION_CONNECTORS
    ]
    if connector_indices:
        if len(connector_indices) != 1:
            return False
        index = connector_indices[0]
        if index == 0 or index == len(folded) - 1:
            return False
        return _location_core(words[:index], folded[:index]) and _location_core(
            words[index + 1 :], folded[index + 1 :]
        )
    return _location_core(words, folded)


def _location_or_scope_value(value: Any) -> bool:
    return _location_value(value) or _audience_scope_value(value)


def _event_name_value(value: Any) -> bool:
    if not _atom_value(value, max_words=8):
        return False
    if (
        _audience_scope_value(value)
        or _event_outcome_value(value)
        or _status_phrase_value(value)
    ):
        return False
    words, folded = _semantic_words(value)
    if not words:
        return False
    tokens = list(folded)
    token_words = list(words)
    if tokens and tokens[0] in _NP_DETERMINERS:
        tokens = tokens[1:]
        token_words = token_words[1:]
    if not tokens or any(token in _CLAUSE_ACTION_VERBS for token in tokens):
        return False
    disallowed_clause_words = _NP_CLAUSE_WORDS - {"and", "for", "of", "the"}
    if any(token in disallowed_clause_words for token in tokens):
        return False
    if tokens[-1] in _LOCATION_HEADS and tokens[-1] not in _EVENT_NAME_HEADS:
        return False
    if tokens[-1] in _EVENT_NAME_HEADS:
        return True
    return False


def _valid_semantic_value(value: Any, semantic_value_type: str) -> bool:
    validators = {
        "AudienceScope": _audience_scope_value,
        "EventName": _event_name_value,
        "EventOutcome": _event_outcome_value,
        "Location": _location_value,
        "StatusPhrase": _status_phrase_value,
    }
    validator = validators.get(semantic_value_type)
    return bool(validator and validator(value))


def _detected_semantic_value_types(value: Any) -> list[str]:
    return [
        semantic_value_type
        for semantic_value_type in SEMANTIC_VALUE_RULES
        if _valid_semantic_value(value, semantic_value_type)
    ]


def _valid_typed_value(value: Any, value_type: str) -> bool:
    if value_type == "Number":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "YesNo":
        return isinstance(value, bool)
    if value_type == "Person":
        return _person_value(value)
    if value_type == "DateOrTime":
        return _atom_value(value) and bool(_DATE_TIME_RE.search(str(value)))
    if value_type == "LocationOrScope":
        return _location_or_scope_value(value)
    if value_type == "Condition":
        return _atom_value(value, max_words=16)
    if value_type in _ATOM_TYPES:
        return _atom_value(value)
    return False


def validate_content_bindings(
    skeleton: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate an API content payload without repairing truth or graph fields."""

    _require_skeleton(skeleton)
    issues: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return {"pass": False, "issues": [{"code": "binding_payload_not_object"}]}
    allowed_top = {"schema_version", "skeleton_sha256", "bindings", "discourse_frame"}
    extra_top = sorted(set(payload) - allowed_top)
    if extra_top:
        issues.append({"code": "forbidden_binding_payload_keys", "keys": extra_top})
    if payload.get("schema_version") != BINDING_SCHEMA_VERSION:
        issues.append({"code": "binding_schema_mismatch"})
    if _text(payload.get("skeleton_sha256")) != skeleton["skeleton_sha256"]:
        issues.append({"code": "binding_skeleton_hash_mismatch"})
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        issues.append({"code": "bindings_not_object"})
        bindings = {}
    specs = {spec["binding_id"]: spec for spec in skeleton["binding_specs"]}
    missing = sorted(set(specs) - set(bindings))
    extra = sorted(set(bindings) - set(specs))
    if missing:
        issues.append({"code": "missing_bindings", "binding_ids": missing})
    if extra:
        issues.append({"code": "unknown_bindings", "binding_ids": extra})
    for binding_id, spec in specs.items():
        if binding_id not in bindings:
            continue
        value = bindings[binding_id]
        if not _valid_typed_value(value, spec["value_type"]):
            issue = {
                "code": "binding_type_mismatch",
                "binding_id": binding_id,
                "value_type": spec["value_type"],
                "value": value,
            }
            if spec.get("semantic_value_type"):
                issue.update(
                    {
                        "required_semantic_value_type": spec[
                            "semantic_value_type"
                        ],
                        "detected_semantic_value_types": (
                            _detected_semantic_value_types(value)
                        ),
                    }
                )
            issues.append(issue)
            continue
        semantic_value_type = spec.get("semantic_value_type")
        if semantic_value_type is not None and not _valid_semantic_value(
            value, semantic_value_type
        ):
            issues.append(
                {
                    "code": "binding_semantic_type_mismatch",
                    "binding_id": binding_id,
                    "value_type": spec["value_type"],
                    "required_semantic_value_type": semantic_value_type,
                    "detected_semantic_value_types": _detected_semantic_value_types(
                        value
                    ),
                    "value": value,
                }
            )
            continue
        if "enum" in spec and value not in spec["enum"]:
            issues.append(
                {
                    "code": "binding_enum_mismatch",
                    "binding_id": binding_id,
                    "value_type": spec["value_type"],
                    "value": copy.deepcopy(value),
                    "allowed_values": copy.deepcopy(spec["enum"]),
                }
            )
        if isinstance(value, int) and not isinstance(value, bool):
            if "minimum" in spec and value < int(spec["minimum"]):
                issues.append({"code": "binding_below_minimum", "binding_id": binding_id})
            if "maximum" in spec and value > int(spec["maximum"]):
                issues.append({"code": "binding_above_maximum", "binding_id": binding_id})
    distinct_groups: dict[str, list[tuple[str, Any]]] = {}
    for binding_id, spec in specs.items():
        group = _text(spec.get("distinct_group"))
        if group and binding_id in bindings:
            distinct_groups.setdefault(group, []).append((binding_id, bindings[binding_id]))
    for group, values in distinct_groups.items():
        normalized = [_fold(value) for _binding_id, value in values]
        if "" in normalized or len(set(normalized)) != len(normalized):
            issues.append(
                {
                    "code": "binding_distinctness_violation",
                    "distinct_group": group,
                    "binding_ids": [binding_id for binding_id, _value in values],
                }
            )
    if skeleton["operation"] == "comparison" and not issues:
        primary = abs(bindings["left_value"] - bindings["right_value"])
        contrast = abs(bindings["left_history"] - bindings["right_history"])
        if primary == 0 or contrast == 0 or abs(primary - contrast) != 1:
            issues.append(
                {
                    "code": "comparison_difference_contract_violation",
                    "primary_difference": primary,
                    "historical_difference": contrast,
                }
            )
    frame = payload.get("discourse_frame")
    if not isinstance(frame, dict) or not _atom_value(frame.get("topic")) or not _atom_value(frame.get("setting")):
        issues.append({"code": "invalid_discourse_frame"})
    else:
        anchor_terms = list(
            (skeleton.get("parent_anchor_contract") or {}).get("topic_terms") or []
        )
        frame_terms = set(_topic_terms(frame.get("topic"), frame.get("setting")))
        if anchor_terms and not frame_terms.intersection(anchor_terms):
            issues.append(
                {
                    "code": "parent_topic_anchor_not_realized",
                    "required_any_topic_terms": anchor_terms,
                }
            )
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "pass": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def _resolve(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, dict) and set(value) == {"$binding"}:
        binding_id = value["$binding"]
        if binding_id not in bindings:
            raise TruthConstructionError(f"unresolved binding: {binding_id}")
        return copy.deepcopy(bindings[binding_id])
    if isinstance(value, dict):
        return {key: _resolve(child, bindings) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve(child, bindings) for child in value]
    return copy.deepcopy(value)


def _possessive(entity: Any) -> str:
    text = _text(entity).replace("_", " ")
    return f"{text}'" if text.casefold().endswith("s") else f"{text}'s"


def _slot_surface(slot: Any) -> str:
    return _text(slot).replace("_", " ").replace("-", " ")


def _node_claim_text(event: dict[str, Any]) -> str:
    entity = _text(event["entity"]).replace("_", " ")
    slot = _slot_surface(event["slot"])
    value = _text(event["value"])
    status = event["status"]
    if isinstance(event.get("set_membership"), dict):
        membership = event["set_membership"]
        decision = "qualifies" if membership["included"] else "does not qualify"
        return (
            f"For {entity}, {value} {decision} for {membership['set_id']} under the condition "
            f"that {membership['condition']}."
        )
    if event.get("event_type") == "visibility_permission":
        scope = event["scope"]
        return (
            f"{event['observer']} may {event['action']} {event['resource']} when {scope['condition']}; "
            f"the access class is {event['access_class']} and the permission is {status}."
        )
    base = f"{_possessive(entity)} {slot} is {value}."
    suffix = {
        "belief": " This remains uncertain.",
        "outdated": " This is outdated and no longer applies.",
        "canceled": " This was canceled before taking effect.",
        "report": " This is an unconfirmed report.",
        "scope_limited": " This applies only within the stated scope.",
    }.get(status, "")
    return f"{base}{suffix}"


def _node_semantic_contract(event: dict[str, Any]) -> dict[str, Any]:
    scope = copy.deepcopy(event.get("scope"))
    condition = event.get("condition")
    if condition is None and isinstance(scope, dict):
        condition = scope.get("condition")
    return {
        "subject": copy.deepcopy(event["entity"]),
        "predicate": event["slot"],
        "objects": [copy.deepcopy(event["value"])],
        "polarity": "positive",
        "status": event["status"],
        "time": {
            "time_index": event["time_index"],
            "global_time_index": event.get("global_time_index"),
            "session_index": event.get("session_index"),
        },
        "condition": copy.deepcopy(condition),
        "scope": scope,
    }


def _relation_claims(operation: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {event["event_id"]: event for event in events}
    relations: list[dict[str, Any]] = []
    for event in events:
        dependencies = event.get("depends_on") or []
        if not dependencies:
            continue
        if operation == "cause_consequence" and event["event_id"] == "E2":
            cause = by_id[dependencies[0]]
            text = (
                f"Because {_possessive(cause['entity'])} {_slot_surface(cause['slot'])} was {cause['value']}, "
                f"{_possessive(event['entity'])} {_slot_surface(event['slot'])} became {event['value']}."
            )
            predicate = "caused"
        elif operation == "bridge" and event["event_id"] == "E3":
            left, right = (by_id[item] for item in dependencies)
            text = (
                f"The verified join combines {_possessive(left['entity'])} {_slot_surface(left['slot'])} record "
                f"with the {_slot_surface(right['slot'])} record for {right['entity']}."
            )
            predicate = "joins_supports"
        elif operation == "temporal_state_update":
            prior = by_id[dependencies[0]]
            text = (
                f"The {_slot_surface(event['slot'])} update follows the earlier {prior['status']} entry for "
                f"{prior['entity']}."
            )
            predicate = "updates"
        else:
            prior = by_id[dependencies[0]]
            text = (
                f"The {_slot_surface(event['slot'])} record applies to the entity selected by "
                f"{_possessive(prior['entity'])} {_slot_surface(prior['slot'])} entry."
            )
            predicate = "depends_on"
        relations.append(
            {
                "claim_id": f"rel:{operation}:{event['event_id']}:{predicate}",
                "claim_type": "cause" if predicate == "caused" else "dependency",
                "predicate": predicate,
                "subject_event_id": event["event_id"],
                "object_event_ids": copy.deepcopy(dependencies),
                "required_for_dialogue": True,
                "canonical_support_claim": {
                    "subject": copy.deepcopy(event["entity"]),
                    "predicate": predicate,
                    "objects": [
                        copy.deepcopy(by_id[item]["entity"])
                        for item in dependencies
                    ],
                    "polarity": "positive",
                    "status": event["status"],
                    "time": {
                        "time_index": event["time_index"],
                        "global_time_index": event.get("global_time_index"),
                        "session_index": event.get("session_index"),
                    },
                    "condition": copy.deepcopy(event.get("condition")),
                    "scope": copy.deepcopy(event.get("scope")),
                    "direction": {
                        "source_event_ids": copy.deepcopy(dependencies),
                        "target_event_id": event["event_id"],
                    },
                },
                "support_realization_hint": text,
                "surface_realization_api_writable": True,
            }
        )
    return relations


def _discourse_plan(
    operation: str,
    events: list[dict[str, Any]],
    relation_claims: list[dict[str, Any]],
    frame: dict[str, Any],
) -> dict[str, Any]:
    node_claims = [
        {
            "claim_id": f"node:{operation}:{event['event_id']}",
            "claim_type": "membership" if isinstance(event.get("set_membership"), dict) else "visibility" if event.get("event_type") == "visibility_permission" else "node",
            "subject_event_id": event["event_id"],
            "required_for_dialogue": True,
            "canonical_support_claim": _node_semantic_contract(event),
            "support_realization_hint": _node_claim_text(event),
            "surface_realization_api_writable": True,
        }
        for event in events
    ]
    all_claims = node_claims + relation_claims
    phase_ids = {
        0: [claim["claim_id"] for claim in all_claims if claim.get("subject_event_id") in {"E1", "E7"}],
        1: [claim["claim_id"] for claim in all_claims if claim.get("subject_event_id") in {"E2", "E3"}],
        2: [claim["claim_id"] for claim in all_claims if claim.get("subject_event_id") in {"E4", "E5", "E6"}],
        3: [claim["claim_id"] for claim in all_claims if claim.get("subject_event_id") == "E8"],
    }
    return {
        "schema_version": DISCOURSE_SCHEMA_VERSION,
        "operation": operation,
        "topic": frame["topic"],
        "setting": frame["setting"],
        "claims": all_claims,
        "phases": [
            {"phase_index": 0, "purpose": "continue the parent topic and establish the scenario", "required_claim_ids": phase_ids[0]},
            {"phase_index": 1, "purpose": "introduce the answer-bearing facts and their semantic relation", "required_claim_ids": phase_ids[1]},
            {"phase_index": 2, "purpose": "clarify competing alternatives without changing frozen truth", "required_claim_ids": phase_ids[2]},
            {"phase_index": 3, "purpose": "close the discussion with the remaining independent context", "required_claim_ids": phase_ids[3]},
        ],
        "api_may_change_claim_text": True,
        "api_may_change_claim_semantics": False,
        "api_may_change_phase_order": False,
        "isolated_fact_turns_forbidden": True,
        "unrelated_filler_forbidden": True,
    }


def _compile_parent_continuity(
    skeleton: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compile one real parent/child link from frozen facts and resolved events."""

    anchor_contract = skeleton.get("parent_anchor_contract") or {}
    parent_facts = [
        item
        for item in anchor_contract.get("anchor_facts") or []
        if isinstance(item, dict)
    ]
    match = anchor_contract.get("continuity_match")
    if not parent_facts:
        if match is not None:
            raise TruthConstructionError(
                "continuity match exists without frozen parent anchor facts"
            )
        return None
    if not isinstance(match, dict):
        raise TruthConstructionError(
            "frozen parent facts require a semantic continuity match"
        )
    parent_by_id = {item.get("event_id"): item for item in parent_facts}
    child_by_id = {item.get("event_id"): item for item in events}
    parent = parent_by_id.get(match.get("parent_event_id"))
    child = child_by_id.get(match.get("child_event_id"))
    if not isinstance(parent, dict) or not isinstance(child, dict):
        raise TruthConstructionError(
            "semantic continuity match references a missing parent or child event"
        )
    child_identity_field = _text(match.get("child_identity_field"))
    parent_identity = _text(match.get("parent_identity"))
    if not child_identity_field or not parent_identity:
        raise TruthConstructionError("semantic continuity identity contract is incomplete")
    if _fold(child.get(child_identity_field)) != _fold(parent_identity):
        raise TruthConstructionError(
            "resolved child identity does not equal its frozen parent continuity identity"
        )
    parent_slot = _fold(parent.get("slot") or parent.get("event_type"))
    child_slot = _fold(child.get("slot") or child.get("event_type"))
    parent_family = _continuity_slot_family(parent_slot)
    child_family = _continuity_slot_family(child_slot)
    exact_slot = bool(parent_slot and parent_slot == child_slot)
    if not exact_slot and (
        parent_family is None or parent_family != child_family
    ):
        raise TruthConstructionError(
            "resolved child slot no longer matches the frozen parent slot family"
        )
    parent_max = int(skeleton["timeline_contract"]["parent_max_global_time_index"])
    child_global = child.get("global_time_index")
    if (
        isinstance(child_global, bool)
        or not isinstance(child_global, int)
        or child_global <= parent_max
    ):
        raise TruthConstructionError(
            "continuity child global time must be after the complete parent timeline"
        )
    parent_id = _text(parent.get("event_id"))
    child_id = _text(child.get("event_id"))
    relation = (
        "updates"
        if skeleton["operation"] == "temporal_state_update"
        else (
            "modifies"
            if exact_slot
            else "continues"
        )
    )
    parent_topics = set(
        _topic_terms(parent.get("slot"), parent.get("value"))
    )
    child_topics = set(_topic_terms(child.get("slot"), child.get("value")))
    shared_topics = sorted(parent_topics.intersection(child_topics))
    reasons = [
        "explicit_parent_event_reference",
        (
            "same_identity_exact_slot"
            if exact_slot
            else "same_identity_slot_family"
        ),
        "child_global_time_after_parent_timeline",
    ]
    if skeleton["operation"] == "temporal_state_update":
        reasons.append("temporal_state_extension")
    if _fold(parent.get("status")) != _fold(child.get("status")):
        reasons.append("status_transition")
    trace = {
        "parent_event_id": parent_id,
        "child_event_id": child_id,
        "relation": relation,
        "matched_features": [
            "same_identity",
            "exact_slot" if exact_slot else "same_slot_family",
            "child_after_parent_timeline",
        ],
        "shared_topic_terms": shared_topics,
        "parent_identity_field": match.get("parent_identity_field"),
        "child_identity_field": child_identity_field,
        "identity": parent_identity,
        "parent_slot": parent_slot,
        "child_slot": child_slot,
        "slot_family": parent_family,
        "parent_status": copy.deepcopy(parent.get("status")),
        "child_status": copy.deepcopy(child.get("status")),
        "parent_global_time_index": copy.deepcopy(
            parent.get("global_time_index")
        ),
        "child_global_time_index": child_global,
        "parent_timeline_max_global_time_index": parent_max,
    }
    child["depends_on_parent_event_ids"] = [parent_id]
    child["parent_topic_anchor_event_ids"] = [parent_id]
    child["parent_relation"] = relation
    return {
        "depends_on_parent_event_ids": [parent_id],
        "parent_topic_anchor_event_ids": [parent_id],
        "continuity_relations": [trace],
        "continuity_score": 1.0 if exact_slot else 0.9,
        "continuity_reasons": reasons,
        "entity_overlap_only": False,
        "semantic_link_compiled_locally": True,
    }


def materialize_truth_skeleton(
    skeleton: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Resolve typed content into a canonical graph, proof contract, and discourse plan."""

    validation = validate_content_bindings(skeleton, payload)
    if validation["pass"] is not True:
        raise TruthConstructionError(
            f"content bindings failed truth construction: {validation['issues']}"
        )
    bindings = payload["bindings"]
    events = _resolve(skeleton["event_templates"], bindings)
    timeline = skeleton["timeline_contract"]
    for event in events:
        event["local_time_index"] = event["time_index"]
        event["global_time_index"] = (
            timeline["parent_max_global_time_index"] + event["time_index"]
        )
        event["session_index"] = timeline["child_session_index"]
        event["source"] = "local_truth_constructor"
        event["alignment_hints"] = {
            "support_claim": _node_claim_text(event),
            "must_mention_any": [_text(event["entity"]), _slot_surface(event["slot"])],
            "must_mention_value_any": [_text(event["value"])],
        }
    continuity = _compile_parent_continuity(skeleton, events)
    graph = {
        "schema_version": "semantic_graph_v2",
        "graph_id": skeleton["graph_id"],
        "parent_graph_id": skeleton["parent_graph_id"],
        "operation": skeleton["operation"],
        "events": events,
        "false_candidates": [],
        "metadata": {
            "truth_constructor": {
                "schema_version": MATERIALIZATION_SCHEMA_VERSION,
                "skeleton_sha256": skeleton["skeleton_sha256"],
                "api_writable_paths": copy.deepcopy(skeleton["api_writable_paths"]),
                "truth_fields_local_only": True,
                "timeline_contract": copy.deepcopy(timeline),
            },
            **({"parent_continuity": continuity} if continuity is not None else {}),
        },
    }
    relation_claims = _relation_claims(skeleton["operation"], events)
    discourse = _discourse_plan(
        skeleton["operation"],
        events,
        relation_claims,
        payload["discourse_frame"],
    )
    proof = _resolve(skeleton["proof_contract"], bindings)
    if proof["gold_projection"]["kind"] == "absolute_difference":
        by_id = {event["event_id"]: event for event in events}
        left, right = proof["gold_projection"]["event_ids"]
        proof["compiled_gold"] = abs(by_id[left]["value"] - by_id[right]["value"])
    elif proof["gold_projection"]["kind"] == "count_membership":
        proof["compiled_gold"] = sum(
            event["set_membership"]["included"] is True
            for event in events
            if isinstance(event.get("set_membership"), dict)
        )
    elif proof["gold_projection"]["kind"] == "event_field":
        by_id = {event["event_id"]: event for event in events}
        projection = proof["gold_projection"]
        proof["compiled_gold"] = by_id[projection["event_id"]][projection["field"]]
    else:
        by_id = {event["event_id"]: event for event in events}
        proof["compiled_gold"] = by_id[proof["gold_projection"]["event_id"]]["value"]
    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "graph_id": graph["graph_id"],
        "operation": skeleton["operation"],
        "skeleton_sha256": skeleton["skeleton_sha256"],
        "immutable_stage64_target": copy.deepcopy(
            skeleton["immutable_stage64_target"]
        ),
        "binding_payload_sha256": canonical_sha256(payload),
        "graph_sha256": canonical_sha256(graph),
        "proof_contract": proof,
        "relation_claims": relation_claims,
        "discourse_plan": discourse,
        "api_defined_truth_fields": [],
        "local_truth_fields": [
            "events[*].slot",
            "events[*].status",
            "events[*].depends_on",
            "events[*].set_membership",
            "events[*].scope",
            "proof_contract",
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return {
        "graph": graph,
        "truth_manifest": manifest,
        "discourse_plan": discourse,
        "binding_validation": validation,
    }


def validate_materialized_truth(
    skeleton: dict[str, Any],
    materialized: Any,
) -> dict[str, Any]:
    """Admit only an in-memory result signed by the local constructor.

    This gate intentionally has no legacy graph fallback.  Historical graph
    artifacts may be replayed as negative fixtures, but they cannot enter the
    Truth-first dialogue path without a matching local skeleton and manifest.
    """

    issues: list[dict[str, Any]] = []
    try:
        _require_skeleton(skeleton)
    except TruthConstructionError as error:
        return {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "pass": False,
            "issue_count": 1,
            "issues": [{"code": "invalid_truth_skeleton", "message": str(error)}],
        }
    if not isinstance(materialized, dict):
        return {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "pass": False,
            "issue_count": 1,
            "issues": [{"code": "materialized_truth_not_object"}],
        }
    graph = materialized.get("graph")
    manifest = materialized.get("truth_manifest")
    discourse = materialized.get("discourse_plan")
    if not isinstance(graph, dict):
        issues.append({"code": "materialized_graph_missing"})
    if not isinstance(manifest, dict):
        issues.append({"code": "truth_manifest_missing"})
    if not isinstance(discourse, dict):
        issues.append({"code": "discourse_plan_missing"})
    if isinstance(manifest, dict):
        manifest_copy = copy.deepcopy(manifest)
        claimed_manifest_hash = _text(manifest_copy.pop("manifest_sha256", ""))
        if not claimed_manifest_hash or canonical_sha256(manifest_copy) != claimed_manifest_hash:
            issues.append({"code": "truth_manifest_hash_mismatch"})
        if manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
            issues.append({"code": "truth_manifest_schema_mismatch"})
        if manifest.get("skeleton_sha256") != skeleton["skeleton_sha256"]:
            issues.append({"code": "truth_manifest_skeleton_mismatch"})
        if manifest.get("api_defined_truth_fields") != []:
            issues.append({"code": "api_defined_truth_fields_present"})
        if isinstance(graph, dict) and manifest.get("graph_sha256") != canonical_sha256(graph):
            issues.append({"code": "materialized_graph_hash_mismatch"})
    if isinstance(graph, dict):
        constructor_meta = (graph.get("metadata") or {}).get("truth_constructor")
        if not isinstance(constructor_meta, dict):
            issues.append({"code": "graph_truth_constructor_metadata_missing"})
        else:
            if constructor_meta.get("skeleton_sha256") != skeleton["skeleton_sha256"]:
                issues.append({"code": "graph_truth_constructor_skeleton_mismatch"})
            if constructor_meta.get("truth_fields_local_only") is not True:
                issues.append({"code": "graph_truth_fields_not_local_only"})
        if graph.get("operation") != skeleton["operation"]:
            issues.append({"code": "materialized_operation_mismatch"})
        if graph.get("graph_id") != skeleton["graph_id"]:
            issues.append({"code": "materialized_graph_id_mismatch"})
        if graph.get("parent_graph_id") != skeleton["parent_graph_id"]:
            issues.append({"code": "materialized_parent_graph_id_mismatch"})
    if isinstance(discourse, dict) and discourse.get("operation") != skeleton["operation"]:
        issues.append({"code": "discourse_operation_mismatch"})
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "pass": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def example_content_bindings(skeleton: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic valid bindings for tests and offline property checks."""

    _require_skeleton(skeleton)
    counters: dict[str, int] = {}
    values_by_type = {
        "Person": ["Maya", "Noah", "Ava", "Liam", "Zoe", "Omar", "Iris", "Eli", "Pia", "Quinn", "Rhea", "Nina"],
        "GroupOrOrg": ["Cedar Guild", "Harbor Club", "North Team"],
        "LocationOrScope": ["Cedar Hall", "Riverside Studio", "North Campus", "West Annex"],
        "DateOrTime": ["Monday", "June 12", "Friday", "09:30"],
        "Event": ["Spring Workshop", "Harbor Festival", "Design Review", "Community Concert"],
        "ObjectOrCommonNoun": ["red notebook", "silver badge", "project folder", "studio key"],
        "AttributeValue": ["severe weather", "schedule conflict", "priority change", "staff shortage"],
        "ProperNounOrNamedEntity": ["Workshop Roster", "Trail Record", "Archive Desk", "Studio Plan", "Panel File", "Harbor Schedule"],
        "Number": [12, 7, 10, 6, 3, 9],
        "YesNo": [True, False],
        "Condition": ["after final approval", "before noon", "on weekends", "after six PM"],
    }
    bindings: dict[str, Any] = {}
    for spec in skeleton["binding_specs"]:
        value_type = spec["value_type"]
        index = counters.get(value_type, 0)
        candidates = values_by_type[value_type]
        bindings[spec["binding_id"]] = copy.deepcopy(
            spec["enum"][0]
            if spec.get("enum")
            else candidates[index % len(candidates)]
        )
        counters[value_type] = index + 1
    if skeleton["operation"] == "comparison":
        bindings.update(
            {
                "left_value": 12,
                "right_value": 7,
                "left_history": 10,
                "right_history": 6,
            }
        )
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "skeleton_sha256": skeleton["skeleton_sha256"],
        "bindings": bindings,
        "discourse_frame": {
            "topic": "a community planning discussion",
            "setting": "a follow-up conversation between established participants",
        },
    }


__all__ = [
    "ANSWER_TYPES",
    "BINDING_ENVELOPE_NORMALIZER_VERSION",
    "BINDING_SCHEMA_VERSION",
    "DISCOURSE_SCHEMA_VERSION",
    "MATERIALIZATION_SCHEMA_VERSION",
    "OPERATIONS",
    "SEMANTIC_VALUE_RULES",
    "SKELETON_SCHEMA_VERSION",
    "TruthConstructionError",
    "api_binding_contract",
    "canonical_sha256",
    "construct_truth_skeleton",
    "example_content_bindings",
    "materialize_truth_skeleton",
    "normalize_content_binding_envelope",
    "validate_materialized_truth",
    "validate_content_bindings",
]
