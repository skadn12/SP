"""Shared structural and semantic quality gates for Stage64 API MCQs."""

from __future__ import annotations

import re
from typing import Any

try:
    from scripts.current_method import semantic_graph_v2
except ModuleNotFoundError:
    from current_method import semantic_graph_v2


INTERROGATIVE_RE = re.compile(
    r"^(?:who|what|when|where|which|how|why)\b",
    re.IGNORECASE,
)
YES_NO_QUESTION_RE = re.compile(
    r"^(?:did|does|do|is|are|was|were|can|could|has|have|had|will|would)\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
}
NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
NUMBER_SURFACE_RE = re.compile(
    r"^(?:(?:about|approximately|around|exactly|roughly|nearly|over|under|more than|fewer than)\s+)?"
    r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(NUMBER_WORDS)) + r")"
    r"(?:\s+(?:people|persons?|items?|objects?|events?|groups?|places?|times?|days?|weeks?|months?|years?))?$",
    re.IGNORECASE,
)
DATE_TIME_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"today|tomorrow|yesterday|morning|afternoon|evening|night|noon|midnight|"
    r"day|week|month|year|hour|minute|am|pm|\d{1,4})\b",
    re.IGNORECASE,
)
OPERATION_QUESTION_SIGNALS = {
    "temporal_state_update": re.compile(
        r"\b(?:when|latest|most recent|currently|current|after|before|eventually|now|final|updated|instead|settled|ended up)\b",
        re.IGNORECASE,
    ),
    "comparison": re.compile(
        r"\b(?:both|compare|comparison|difference|different|same|more|less|earlier|later|older|newer|between)\b",
        re.IGNORECASE,
    ),
    "visibility": re.compile(
        r"\b(?:who|visible|visibility|see|access|know|allowed|permission|scope|only person|only people)\b",
        re.IGNORECASE,
    ),
    "count": re.compile(r"\b(?:how many|number of|count)\b", re.IGNORECASE),
    "cause_consequence": re.compile(
        r"\b(?:why|cause|caused|because|due to|result|resulted|lead|led|consequence|effect|happened after)\b",
        re.IGNORECASE,
    ),
}
OPERATION_SUBTYPE_SIGNALS = {
    "temporal_state_update": re.compile(r"temporal|time|date|latest|final_state|before_after|update", re.IGNORECASE),
    "bridge": re.compile(r"bridge|multi.?hop|entity_to_event|cross", re.IGNORECASE),
    "comparison": re.compile(r"compar|contrast|difference|shared|before_after", re.IGNORECASE),
    "visibility": re.compile(r"visib|permission|scope|access|who_knows|scope_yesno", re.IGNORECASE),
    "count": re.compile(r"count|number|quant", re.IGNORECASE),
    "cause_consequence": re.compile(r"cause|consequence|result|effect|why", re.IGNORECASE),
    "relation_lookup": re.compile(r"relation|lookup|entity_to_event|fact|attribute|object|location", re.IGNORECASE),
}
RESERVED_SUBTYPE_SIGNALS = {
    "temporal_state_update": re.compile(r"temporal|time|date|latest|before_after|update", re.IGNORECASE),
    "bridge": re.compile(r"bridge|multi.?hop|cross", re.IGNORECASE),
    "comparison": re.compile(r"compar|contrast|difference|shared", re.IGNORECASE),
    "visibility": re.compile(r"visib|permission|scope|access|who_knows", re.IGNORECASE),
    "count": re.compile(r"count|number|quant", re.IGNORECASE),
    "cause_consequence": re.compile(r"cause|consequence|result|effect|why", re.IGNORECASE),
    "relation_lookup": re.compile(r"relation|entity_to_event|fact_lookup|attribute_lookup|object_location", re.IGNORECASE),
}
SHORT_ANSWER_MAX_WORDS = {
    "Person": 12,
    "GroupOrOrg": 12,
    "LocationOrScope": 14,
    "DateOrTime": 12,
    "Number": 8,
    "YesNo": 3,
    "ObjectOrCommonNoun": 16,
    "AttributeValue": 18,
    "ProperNounOrNamedEntity": 12,
    "Event": 45,
}
NON_ACTIVE_STATUSES = {
    "outdated",
    "canceled",
    "expired",
    "delayed_not_active",
    "false",
    "belief",
    "report",
    "scope_limited",
}
ACTIVE_STATUSES = {"active", "restored"}
GENERIC_GRAPH_SLOT_RE = re.compile(
    r"^(?:raw qa(?: .*)?|qa(?: .*)?|generic(?: fact| state)?|fact|state|memory state|answer|value)$",
    re.IGNORECASE,
)
TEMPORAL_STATUS_PARENTHETICAL_RE = re.compile(
    r"\s*[\(\[][^\)\]]*\b(?:active|current|restore\w*|cancel\w*|outdated|expire\w*|"
    r"delay\w*|supersed\w*|earlier|previous|assumption|update\w*)\b[^\)\]]*[\)\]]\s*$",
    re.IGNORECASE,
)
COMPARISON_NATURAL_TIME_QUALIFIER_RE = re.compile(
    r"\b(?:currently|current|now|latest|most\s+recent|earlier|later|previous|prior|"
    r"initial|original|first|last|final|then|today|yesterday|tomorrow|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\b",
    re.IGNORECASE,
)
COMPARISON_INTERNAL_TIME_INDEX_RE = re.compile(
    r"\btime[\s_-]*index\b",
    re.IGNORECASE,
)
COMPARISON_SLOT_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "with",
}
COMPARISON_SLOT_TOKEN_ALIASES = {
    "annually": "year",
    "daily": "day",
    "hourly": "hour",
    "monthly": "month",
    "weekly": "week",
    "yearly": "year",
}
TEMPORAL_DIRECT_GOLD_CUE_RE = re.compile(
    r"\b(?:current(?:ly)?|now|confirm(?:ed|s|ation)?|active)\b",
    re.IGNORECASE,
)
CAUSE_QUESTION_RE = re.compile(
    r"\b(?:what|which|who)\s+caused\b|\bwhy\b|\b(?:cause|reason|trigger)\s+(?:of|for)\b|\bdue\s+to\b",
    re.IGNORECASE,
)
EFFECT_QUESTION_RE = re.compile(
    r"\b(?:effect|consequence|result)\b|\bwhat\s+happened\b|\b(?:led|lead|resulted)\s+(?:to|from|in)\b|\bfollows?\b",
    re.IGNORECASE,
)


def normalized_option_text(value: Any) -> str:
    """Normalize case, punctuation, underscores, and whitespace."""

    return re.sub(r"[\W_]+", " ", str(value or "").strip().casefold(), flags=re.UNICODE).strip()


def _graph_answer_value(event: dict[str, Any]) -> Any:
    return event.get("value") if "value" in event else event.get("raw_value")


def _graph_answer_value_key(event: dict[str, Any]) -> str:
    return normalized_option_text(_graph_answer_value(event))


def _temporal_option_surface_key(value: Any) -> str:
    text = str(value or "").strip()
    while TEMPORAL_STATUS_PARENTHETICAL_RE.search(text):
        text = TEMPORAL_STATUS_PARENTHETICAL_RE.sub("", text).strip()
    return normalized_option_text(text)


def _word_count(value: Any) -> int:
    return len(re.findall(r"\b\w+\b", str(value or ""), flags=re.UNICODE))


def _event_primary_id(event: dict[str, Any]) -> str:
    return str(
        event.get("event_instance_id")
        or event.get("event_id")
        or event.get("node_id")
        or ""
    ).strip()


def _compiled_event_index(
    sample: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    graph = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    events = [event for event in (graph.get("events") or []) if isinstance(event, dict)]
    if not events:
        plan = sample.get("current_event_plan") if isinstance(sample.get("current_event_plan"), dict) else {}
        events = [event for event in (plan.get("events") or []) if isinstance(event, dict)]
    by_id: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for event in events:
        primary = _event_primary_id(event)
        if not primary:
            continue
        by_id[primary] = event
        for key in ("event_instance_id", "event_id", "node_id", "origin_event_id"):
            alias = str(event.get(key) or "").strip()
            if alias:
                aliases[alias] = primary
    return events, by_id, aliases


def _canonical_event_ids(values: Any, aliases: dict[str, str]) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    for value in values:
        raw = str(value or "").strip()
        canonical = aliases.get(raw, raw if raw in aliases.values() else "")
        if canonical and canonical not in output:
            output.append(canonical)
    return output


def _reasoning_event_ids(qa: dict[str, Any], aliases: dict[str, str]) -> list[str]:
    query_plan = qa.get("query_plan") if isinstance(qa.get("query_plan"), dict) else {}
    resolved = _canonical_event_ids(
        query_plan.get("resolved_reasoning_event_instance_ids"), aliases
    )
    if resolved:
        return resolved
    return _canonical_event_ids(query_plan.get("reasoning_anchor_path"), aliases)


def _option_events(
    qa: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    aliases: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    options = [option for option in (qa.get("options") or []) if isinstance(option, dict)]
    by_label: dict[str, dict[str, Any]] = {}
    events_by_label: dict[str, dict[str, Any]] = {}
    for option in options:
        label = str(option.get("id") or option.get("option_id") or "").strip()
        if not label:
            continue
        by_label[label] = option
        raw_ref = str(
            option.get("source_event_instance_id")
            or option.get("event_instance_id")
            or option.get("source_event_id")
            or option.get("anchor_ref")
            or ""
        ).strip()
        event = by_id.get(aliases.get(raw_ref, raw_ref))
        if event is not None:
            events_by_label[label] = event
    return by_label, events_by_label


def _single_gold_label(qa: dict[str, Any]) -> str:
    raw = qa.get("gold")
    labels = [str(value) for value in raw] if isinstance(raw, list) else [str(raw or "")]
    return labels[0] if len(labels) == 1 else ""


def _dependency_map(
    events: list[dict[str, Any]], aliases: dict[str, str]
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for event in events:
        event_id = _event_primary_id(event)
        if not event_id:
            continue
        output[event_id] = set(
            _canonical_event_ids(event.get("depends_on"), aliases)
        )
    return output


def _dependency_closure(endpoint: str, dependencies: dict[str, set[str]]) -> set[str]:
    closure: set[str] = set()
    pending = [endpoint]
    while pending:
        event_id = pending.pop()
        if not event_id or event_id in closure:
            continue
        closure.add(event_id)
        pending.extend(dependencies.get(event_id, set()) - closure)
    return closure


def _dialogue_turn_text_by_id(sample: dict[str, Any]) -> dict[str, str]:
    dialogue = sample.get("dialogue") if isinstance(sample.get("dialogue"), dict) else {}
    output: dict[str, str] = {}
    for turn in dialogue.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        turn_id = str(turn.get("turn_id") or turn.get("id") or "").strip()
        text = str(turn.get("text") or turn.get("content") or "").strip()
        if turn_id and text:
            output[turn_id] = text
    return output


def _dialogue_graph_fact_repetition_issues(
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reject graph facts restated by filler outside their evidence turns.

    This intentionally uses a narrow lexical rule: a turn must contain both the
    exact graph entity and the exact graph value (or the slot and status).  The
    union of all evidence turns for the same entity is exempt so that one
    canonical event may explain its dependency on another event without being
    mistaken for filler.  Events without evidence turn ids are ignored for
    backward compatibility.
    """

    turn_text_by_id = _dialogue_turn_text_by_id(sample)
    if not turn_text_by_id:
        return []

    events, _, _ = _compiled_event_index(sample)
    eligible_by_entity: dict[str, list[dict[str, Any]]] = {}
    evidence_by_entity: dict[str, set[str]] = {}
    entity_surface_by_key: dict[str, str] = {}
    for event in events:
        entity_surface = str(event.get("entity") or "").strip()
        entity_key = normalized_option_text(entity_surface)
        evidence_turn_ids = {
            str(turn_id).strip()
            for field in ("turn_ids", "evidence_turn_ids")
            for turn_id in (
                event.get(field) if isinstance(event.get(field), list) else []
            )
            if str(turn_id).strip()
        }
        if not entity_key or not evidence_turn_ids:
            continue
        eligible_by_entity.setdefault(entity_key, []).append(event)
        evidence_by_entity.setdefault(entity_key, set()).update(evidence_turn_ids)
        entity_surface_by_key.setdefault(entity_key, entity_surface)

    issues: list[dict[str, Any]] = []
    for turn_id, text in turn_text_by_id.items():
        normalized_turn = f" {normalized_option_text(text)} "
        for entity_key, entity_events in eligible_by_entity.items():
            if turn_id in evidence_by_entity[entity_key]:
                continue
            if f" {entity_key} " not in normalized_turn:
                continue

            matched_event_ids: list[str] = []
            matched_facts: list[dict[str, str]] = []
            for event in entity_events:
                value_surface = str(_graph_answer_value(event) or "").strip()
                value_key = normalized_option_text(value_surface)
                slot_surface = str(event.get("slot") or "").strip()
                slot_key = normalized_option_text(slot_surface)
                status_surface = str(event.get("status") or "").strip()
                status_key = normalized_option_text(status_surface)
                value_match = bool(
                    value_key
                    and value_key != entity_key
                    and f" {value_key} " in normalized_turn
                )
                slot_status_match = bool(
                    slot_key
                    and status_key
                    and f" {slot_key} " in normalized_turn
                    and f" {status_key} " in normalized_turn
                )
                if not value_match and not slot_status_match:
                    continue
                event_id = _event_primary_id(event)
                if event_id:
                    matched_event_ids.append(event_id)
                matched_facts.append(
                    {
                        "slot": slot_surface,
                        "value": value_surface,
                        "status": status_surface,
                    }
                )

            if matched_facts:
                issues.append(
                    _issue(
                        "graph_fact_repeated_outside_evidence_turn",
                        turn_id=turn_id,
                        entity=entity_surface_by_key[entity_key],
                        matched_event_ids=matched_event_ids,
                        matched_facts=matched_facts,
                    )
                )
    return issues


def _filter_condition_text(value: Any) -> str:
    if isinstance(value, str):
        return normalized_option_text(value)
    if not isinstance(value, dict):
        return ""
    direct = value.get("condition") or value.get("predicate") or value.get("name")
    if isinstance(direct, str) and direct.strip():
        return normalized_option_text(direct)
    scalar_parts: list[str] = []
    for key in sorted(value):
        child = value[key]
        if isinstance(child, (str, int, float, bool)):
            scalar_parts.extend((str(key), str(child)))
        elif isinstance(child, list) and all(
            isinstance(item, (str, int, float, bool)) for item in child
        ):
            scalar_parts.append(str(key))
            scalar_parts.extend(str(item) for item in child)
    return normalized_option_text(" ".join(scalar_parts))


def _issue(reason: str, **details: Any) -> dict[str, Any]:
    return {"reason": reason, **details}


def _parse_count_integer(value: Any) -> int | None:
    normalized = normalized_option_text(value)
    if re.fullmatch(r"\d+", normalized):
        return int(normalized)
    return NUMBER_WORD_VALUES.get(normalized)


def _question_dimension_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in normalized_option_text(value).split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token and token not in {
            "candidate",
            "collection",
            "item",
            "list",
            "member",
            "membership",
            "record",
            "set",
            "status",
        }:
            tokens.add(token)
    return tokens


def _question_names_dimension(question_tokens: set[str], raw_dimension: Any) -> bool:
    dimension_tokens = _question_dimension_tokens(raw_dimension)
    if not dimension_tokens:
        return True
    aliases = {
        "destination": {"city", "location", "place"},
        "participant": {"attendee", "people", "person"},
        "quality": {"attribute", "trait"},
    }
    expanded_question = set(question_tokens)
    for token, alternatives in aliases.items():
        if alternatives.intersection(question_tokens):
            expanded_question.add(token)
    joined = "".join(dimension_tokens)
    overlap = dimension_tokens.intersection(expanded_question)
    return len(overlap) >= max(1, (len(dimension_tokens) + 1) // 2) or any(
        len(token) >= 5 and token in joined for token in expanded_question
    )


def _comparison_slot_tokens(value: Any) -> set[str]:
    return {
        COMPARISON_SLOT_TOKEN_ALIASES.get(token, token)
        for token in _question_dimension_tokens(value)
        if token not in COMPARISON_SLOT_STOP_TOKENS
    }


def graph_plan_quality_issues(
    current_event_plan: dict[str, Any],
    qa_operation: str | None,
    reasoning_target: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Reject intrinsically bad Stage64 graph plans before dialogue/QA generation or caching."""

    operation = str(qa_operation or "").strip()
    if not operation:
        return []
    events = [
        event
        for event in (current_event_plan.get("events") or [])
        if isinstance(event, dict)
    ]
    issues: list[dict[str, Any]] = []
    if len(events) < 4:
        issues.append(_issue("stage64_graph_too_few_events", event_count=len(events)))

    generic_slots: list[str] = []
    question_values: list[str] = []
    for index, event in enumerate(events):
        event_id = str(event.get("event_instance_id") or event.get("event_id") or index)
        slot = normalized_option_text(event.get("slot"))
        value = str(event.get("value") or event.get("raw_value") or "").strip()
        if not slot or GENERIC_GRAPH_SLOT_RE.fullmatch(slot):
            generic_slots.append(event_id)
        if "?" in value or INTERROGATIVE_RE.match(value):
            question_values.append(event_id)
    if generic_slots:
        issues.append(_issue("generic_graph_slot", event_ids=generic_slots))
    if question_values:
        issues.append(_issue("question_used_as_graph_value", event_ids=question_values))

    if operation == "temporal_state_update":
        forbid_max_time = bool(
            isinstance(reasoning_target, dict)
            and reasoning_target.get("forbid_max_time_shortcut") is True
        )
        strategy = (
            str(reasoning_target.get("hard_negative_strategy") or "").strip()
            if forbid_max_time
            else ""
        )
        required_later_statuses = {
            "same_slot_canceled_value": {"canceled"},
            "same_slot_outdated_value": {"outdated"},
            "same_slot_report_or_belief_value": {"report", "belief"},
        }.get(strategy, set())
        if forbid_max_time and not required_later_statuses:
            issues.append(
                _issue(
                    "temporal_graph_unknown_hard_negative_strategy",
                    hard_negative_strategy=strategy,
                )
            )
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for event in events:
            key = (
                normalized_option_text(event.get("entity")),
                normalized_option_text(event.get("slot")),
            )
            groups.setdefault(key, []).append(event)
        valid_update_chain = False
        candidate_group_found = False
        for (entity, slot), group in groups.items():
            if len(group) < 3:
                continue
            candidate_group_found = True
            ordered = sorted(
                group,
                key=lambda event: (
                    event.get("time_index")
                    if isinstance(event.get("time_index"), int) and not isinstance(event.get("time_index"), bool)
                    else -1
                ),
            )
            value_to_event_ids: dict[str, list[str]] = {}
            for index, event in enumerate(ordered):
                value_to_event_ids.setdefault(_graph_answer_value_key(event), []).append(
                    str(event.get("event_instance_id") or event.get("event_id") or index)
                )
            duplicate_value_event_ids = [
                event_ids
                for value, event_ids in value_to_event_ids.items()
                if value and len(event_ids) > 1
            ]
            if duplicate_value_event_ids:
                issues.append(
                    _issue(
                        "temporal_graph_duplicate_update_value",
                        entity=entity,
                        slot=slot,
                        duplicate_value_event_ids=duplicate_value_event_ids,
                    )
                )

            current_events = [
                event
                for event in ordered
                if str(event.get("status") or "").casefold() in ACTIVE_STATUSES
            ]
            if len(current_events) != 1:
                issues.append(
                    _issue(
                        "temporal_graph_current_state_not_unique",
                        entity=entity,
                        slot=slot,
                        current_event_ids=[
                            str(event.get("event_instance_id") or event.get("event_id") or "")
                            for event in current_events
                        ],
                    )
                )
            elif not forbid_max_time and current_events[0] is not ordered[-1]:
                issues.append(
                    _issue(
                        "temporal_graph_current_state_not_latest",
                        entity=entity,
                        slot=slot,
                        current_event_id=current_events[0].get("event_instance_id") or current_events[0].get("event_id"),
                        latest_event_id=ordered[-1].get("event_instance_id") or ordered[-1].get("event_id"),
                    )
                )

            time_indexes = [event.get("time_index") for event in ordered]
            distinct_time_indexes = (
                all(isinstance(value, int) and not isinstance(value, bool) for value in time_indexes)
                and len(time_indexes) == len(set(time_indexes))
            )
            if not distinct_time_indexes:
                issues.append(
                    _issue(
                        "temporal_graph_time_index_not_unique",
                        entity=entity,
                        slot=slot,
                    )
                )
            if forbid_max_time and len(current_events) == 1:
                current_index = ordered.index(current_events[0])
                later_events = ordered[current_index + 1 :]
                matching_later_ids = [
                    str(
                        event.get("event_instance_id")
                        or event.get("event_id")
                        or ""
                    )
                    for event in later_events
                    if str(event.get("status") or "").casefold()
                    in required_later_statuses
                ]
                if not later_events:
                    issues.append(
                        _issue(
                            "temporal_graph_anti_shortcut_current_is_latest",
                            entity=entity,
                            slot=slot,
                        )
                    )
                elif not matching_later_ids:
                    issues.append(
                        _issue(
                            "temporal_graph_missing_later_strategy_update",
                            entity=entity,
                            slot=slot,
                            hard_negative_strategy=strategy,
                            required_later_statuses=sorted(required_later_statuses),
                        )
                    )
                valid_update_chain = valid_update_chain or (
                    not duplicate_value_event_ids
                    and current_events[0] is not ordered[-1]
                    and distinct_time_indexes
                    and bool(matching_later_ids)
                    and all(
                        str(event.get("status") or "").casefold()
                        in NON_ACTIVE_STATUSES
                        for event in ordered
                        if event is not current_events[0]
                    )
                )
            else:
                valid_update_chain = valid_update_chain or (
                    not duplicate_value_event_ids
                    and len(current_events) == 1
                    and current_events[0] is ordered[-1]
                    and distinct_time_indexes
                    and any(
                        str(event.get("status") or "").casefold()
                        in NON_ACTIVE_STATUSES
                        for event in ordered[:-1]
                    )
                )
        if not candidate_group_found or (not valid_update_chain and not issues):
            issues.append(_issue("temporal_graph_missing_same_slot_update_chain"))
    elif operation == "visibility":
        if not any(
            str(event.get("status") or "").casefold() == "scope_limited"
            or str(event.get("event_type") or "").casefold() == "permission_scope"
            or (isinstance(event.get("scope"), dict) and bool(event["scope"]))
            for event in events
        ):
            issues.append(_issue("visibility_graph_missing_scope_event"))

    deduplicated: dict[str, dict[str, Any]] = {}
    for issue in issues:
        deduplicated.setdefault(str(issue.get("reason") or "unknown"), issue)
    return list(deduplicated.values())


def _answer_surface_issues(qa: dict[str, Any]) -> list[dict[str, Any]]:
    answer_type = str(qa.get("answer_type") or "")
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    issues: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("id") or option.get("option_id") or "")
        text = str(option.get("text") or "").strip()
        normalized = normalized_option_text(text)
        if "?" in text or INTERROGATIVE_RE.match(text):
            issues.append(_issue("option_restates_question", label=label, text=text[:160]))
        max_words = SHORT_ANSWER_MAX_WORDS.get(answer_type)
        if max_words is not None and _word_count(text) > max_words:
            issues.append(
                _issue(
                    "answer_type_surface_too_long",
                    label=label,
                    answer_type=answer_type,
                    word_count=_word_count(text),
                    max_words=max_words,
                )
            )
        if answer_type == "Number" and not NUMBER_SURFACE_RE.fullmatch(normalized):
            issues.append(_issue("number_answer_surface_mismatch", label=label, text=text[:160]))
        elif answer_type == "YesNo" and normalized not in {"yes", "no", "true", "false"}:
            issues.append(_issue("yesno_answer_surface_mismatch", label=label, text=text[:160]))
        elif answer_type == "DateOrTime" and not DATE_TIME_RE.search(text):
            issues.append(_issue("date_time_answer_surface_mismatch", label=label, text=text[:160]))
    return issues


def _question_answer_type_issues(qa: dict[str, Any]) -> list[dict[str, Any]]:
    question = str(qa.get("question") or qa.get("question_text") or "").strip()
    answer_type = str(qa.get("answer_type") or "")
    expected: set[str] | None = None
    if re.match(r"^\s*when\b", question, flags=re.IGNORECASE) or re.match(
        r"^\s*(?:(?:what|which)|in\s+which)\s+(?:date|time|day|week|month|year)\b",
        question,
        flags=re.IGNORECASE,
    ):
        expected = {"DateOrTime"}
    elif re.match(r"^\s*who\b", question, flags=re.IGNORECASE):
        expected = {"Person", "GroupOrOrg", "ProperNounOrNamedEntity"}
    elif re.match(r"^\s*where\b", question, flags=re.IGNORECASE):
        expected = {"LocationOrScope", "ProperNounOrNamedEntity"}
    elif re.match(r"^\s*how\s+many\b", question, flags=re.IGNORECASE):
        expected = {"Number"}
    elif YES_NO_QUESTION_RE.match(question):
        expected = {"YesNo"}
    if expected is not None and answer_type not in expected:
        return [
            _issue(
                "question_answer_type_mismatch",
                question=question[:200],
                answer_type=answer_type,
                expected_answer_types=sorted(expected),
            )
        ]
    return []


def _operation_semantic_issues(qa: dict[str, Any]) -> list[dict[str, Any]]:
    operation = str(qa.get("qa_operation") or "")
    subtype = str(qa.get("qa_subtype") or "")
    answer_type = str(qa.get("answer_type") or "")
    question = str(qa.get("question") or qa.get("question_text") or "")
    issues: list[dict[str, Any]] = []
    if operation == "bridge" and answer_type == "YesNo":
        issues.append(_issue("bridge_yesno_shortcut", answer_type=answer_type))
    signal = OPERATION_QUESTION_SIGNALS.get(operation)
    if signal is not None and not signal.search(question):
        issues.append(_issue("qa_operation_question_mismatch", qa_operation=operation, question=question[:200]))
    subtype_signal = OPERATION_SUBTYPE_SIGNALS.get(operation)
    if subtype_signal is not None and not subtype_signal.search(subtype):
        issues.append(_issue("qa_operation_subtype_mismatch", qa_operation=operation, qa_subtype=subtype))
    conflicting_operations = sorted(
        candidate
        for candidate, candidate_signal in RESERVED_SUBTYPE_SIGNALS.items()
        if candidate != operation and candidate_signal.search(subtype)
    )
    if conflicting_operations:
        issues.append(
            _issue(
                "qa_subtype_conflicts_with_operation",
                qa_operation=operation,
                qa_subtype=subtype,
                conflicting_operations=conflicting_operations,
            )
        )
    return issues


def _compiled_operation_contract_issues(sample: dict[str, Any], qa: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-run structure-sensitive operation checks on the durable compiled graph."""

    operation = str(qa.get("qa_operation") or "")
    blocking_codes = {
        "temporal_state_update": {
            "temporal_current_state_not_unique",
            "temporal_current_state_not_latest",
            "temporal_earlier_update_still_active",
            "temporal_anti_shortcut_current_is_latest",
            "temporal_anti_shortcut_later_update_active",
            "temporal_anti_shortcut_missing_later_strategy_update",
            "temporal_unknown_hard_negative_strategy",
        },
        "bridge": {"bridge_missing_two_hop_dependency"},
        "comparison": {"comparison_missing_unique_numeric_pair"},
        "visibility": {"visibility_missing_scope"},
        "count": {"count_missing_candidate_set"},
        "cause_consequence": {"cause_missing_ordered_dependency"},
    }.get(operation)
    if blocking_codes is None:
        return []
    graph = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    events = graph.get("events") if isinstance(graph.get("events"), list) else []
    delta = {
        "schema_version": semantic_graph_v2.SCHEMA_VERSION,
        "graph_id": str(graph.get("graph_snapshot_id") or "compiled:child"),
        "parent_graph_id": "compiled:parent",
        "operation": operation,
        "events": events,
    }
    semantic_issues = semantic_graph_v2.validate_graph_delta(
        delta,
        operation=operation,
        reasoning_target=qa if operation == "temporal_state_update" else None,
    )
    return [
        _issue(
            str(issue.get("code") or "compiled_graph_operation_contract_failed"),
            path=issue.get("path"),
            message=issue.get("message"),
            compiled_graph=True,
        )
        for issue in semantic_issues
        if issue.get("code") in blocking_codes
    ]


def _count_derivation_issues(sample: dict[str, Any], qa: dict[str, Any]) -> list[dict[str, Any]]:
    if str(qa.get("qa_operation") or "") != "count":
        return []
    graph = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    events = [event for event in (graph.get("events") or []) if isinstance(event, dict)]
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        for key in ("event_instance_id", "event_id", "node_id"):
            event_id = str(event.get(key) or "")
            if event_id:
                by_id[event_id] = event

    query_plan = qa.get("query_plan") if isinstance(qa.get("query_plan"), dict) else {}
    reasoning_ids = [
        str(item)
        for item in (query_plan.get("resolved_reasoning_event_instance_ids") or [])
        if str(item)
    ]
    issues: list[dict[str, Any]] = []
    resolved_group_fields = (
        "resolved_required_event_instance_ids",
        "resolved_supporting_event_instance_ids",
        "resolved_gold_event_instance_ids",
        "resolved_distractor_event_instance_ids",
    )
    missing_resolved_groups = [
        field
        for field in resolved_group_fields
        if not isinstance(query_plan.get(field), list) or not query_plan.get(field)
    ]
    if missing_resolved_groups:
        issues.append(
            _issue(
                "count_resolved_reference_groups_missing",
                fields=missing_resolved_groups,
            )
        )
    if len(reasoning_ids) != len(set(reasoning_ids)):
        issues.append(_issue("count_duplicate_resolved_reasoning_events"))
    unresolved_ids = sorted({event_id for event_id in reasoning_ids if event_id not in by_id})
    if not reasoning_ids or unresolved_ids:
        issues.append(
            _issue(
                "count_reasoning_event_ids_unresolved",
                reasoning_event_count=len(reasoning_ids),
                unresolved_event_ids=unresolved_ids,
            )
        )
        return issues

    reasoning_events = [by_id[event_id] for event_id in dict.fromkeys(reasoning_ids)]
    dimensions: list[tuple[str, str, str]] = []
    invalid_membership_ids: list[str] = []
    for event_id, event in zip(dict.fromkeys(reasoning_ids), reasoning_events):
        membership = event.get("set_membership")
        if (
            not isinstance(membership, dict)
            or not str(membership.get("set_id") or "").strip()
            or not isinstance(membership.get("included"), bool)
        ):
            invalid_membership_ids.append(event_id)
            continue
        dimensions.append(
            (
                normalized_option_text(membership["set_id"]),
                normalized_option_text(event.get("entity")),
                normalized_option_text(event.get("slot")),
            )
        )
    if invalid_membership_ids:
        issues.append(_issue("count_reasoning_event_missing_membership", event_ids=invalid_membership_ids))
        return issues
    membership_conditions = [
        _filter_condition_text(event["set_membership"].get("condition"))
        for event in reasoning_events
    ]
    missing_condition_ids = [
        event_id
        for event_id, condition in zip(dict.fromkeys(reasoning_ids), membership_conditions)
        if not condition
    ]
    distinct_conditions = {condition for condition in membership_conditions if condition}
    if missing_condition_ids:
        issues.append(
            _issue(
                "count_membership_condition_missing",
                event_ids=missing_condition_ids,
            )
        )
    if len(distinct_conditions) > 1:
        issues.append(
            _issue(
                "count_membership_condition_inconsistent",
                conditions=sorted(distinct_conditions),
            )
        )
    graph_condition = next(iter(distinct_conditions), "")
    filter_predicate = _filter_condition_text(query_plan.get("filter_predicate"))
    if not filter_predicate:
        issues.append(_issue("count_filter_predicate_missing"))
    elif not graph_condition or filter_predicate != graph_condition:
        issues.append(
            _issue(
                "count_filter_predicate_mismatch",
                graph_condition=graph_condition,
                filter_predicate=filter_predicate,
            )
        )
    if graph_condition:
        condition_question_tokens = _question_dimension_tokens(
            qa.get("question") or qa.get("question_text")
        )
        if not _question_names_dimension(
            condition_question_tokens, graph_condition
        ):
            issues.append(
                _issue(
                    "count_question_missing_filter_condition",
                    condition=graph_condition,
                )
            )
    distinct_dimensions = set(dimensions)
    included_count = sum(
        bool(event["set_membership"]["included"])
        for event in reasoning_events
        if isinstance(event.get("set_membership"), dict)
    )
    excluded_count = len(reasoning_events) - included_count
    if (
        len(reasoning_events) < 6
        or included_count < 2
        or excluded_count < 2
        or len(distinct_dimensions) != 1
        or any(not value for value in dimensions[0])
    ):
        issues.append(
            _issue(
                "count_reasoning_events_not_one_candidate_set",
                reasoning_event_count=len(reasoning_events),
                included_count=included_count,
                excluded_count=excluded_count,
                candidate_dimensions=sorted(distinct_dimensions),
            )
        )
        return issues

    member_values = [normalized_option_text(_graph_answer_value(event)) for event in reasoning_events]
    if any(not value for value in member_values) or len(member_values) != len(set(member_values)):
        issues.append(_issue("count_candidate_member_values_not_distinct", member_values=member_values))

    target_dimension = dimensions[0]
    target_set_id = target_dimension[0]
    same_set_events = [
        event
        for event in events
        if isinstance(event.get("set_membership"), dict)
        and normalized_option_text(event["set_membership"].get("set_id")) == target_set_id
    ]
    same_set_dimensions = {
        (
            normalized_option_text(event.get("entity")),
            normalized_option_text(event.get("slot")),
        )
        for event in same_set_events
    }
    if len(same_set_dimensions) != 1:
        issues.append(
            _issue(
                "count_set_id_has_mixed_semantic_dimensions",
                set_id=target_set_id,
                entity_slot_dimensions=sorted(same_set_dimensions),
            )
        )
    all_candidate_ids = {
        str(event.get("event_instance_id") or event.get("event_id") or "")
        for event in events
        if isinstance(event.get("set_membership"), dict)
        and (
            normalized_option_text(event["set_membership"].get("set_id")),
            normalized_option_text(event.get("entity")),
            normalized_option_text(event.get("slot")),
        )
        == target_dimension
    }
    cited_ids = set(reasoning_ids)
    if all_candidate_ids != cited_ids:
        issues.append(
            _issue(
                "count_reasoning_does_not_cover_candidate_set",
                missing_event_ids=sorted(all_candidate_ids - cited_ids),
                extraneous_event_ids=sorted(cited_ids - all_candidate_ids),
            )
        )

    for field in ("resolved_required_event_instance_ids", "resolved_supporting_event_instance_ids"):
        field_ids = {str(item) for item in (query_plan.get(field) or []) if str(item)}
        if field_ids and field_ids != all_candidate_ids:
            issues.append(
                _issue(
                    "count_reference_group_does_not_cover_candidate_set",
                    field=field,
                    missing_event_ids=sorted(all_candidate_ids - field_ids),
                    extraneous_event_ids=sorted(field_ids - all_candidate_ids),
                )
            )

    for field in ("resolved_gold_event_instance_ids", "resolved_distractor_event_instance_ids"):
        field_ids = {str(item) for item in (query_plan.get(field) or []) if str(item)}
        if field_ids and not field_ids.issubset(all_candidate_ids):
            issues.append(
                _issue(
                    "count_reference_group_outside_candidate_set",
                    field=field,
                    extraneous_event_ids=sorted(field_ids - all_candidate_ids),
                )
            )

    option_source_ids = {
        str(option.get("source_event_instance_id") or option.get("event_instance_id") or "")
        for option in (qa.get("options") or [])
        if isinstance(option, dict)
    }
    if "" in option_source_ids or not option_source_ids.issubset(all_candidate_ids):
        issues.append(
            _issue(
                "count_option_source_outside_candidate_set",
                extraneous_event_ids=sorted(option_source_ids - all_candidate_ids),
            )
        )

    expected_count = included_count
    options = [option for option in (qa.get("options") or []) if isinstance(option, dict)]
    option_values = [_parse_count_integer(option.get("text")) for option in options]
    if len(options) != 4 or any(value is None for value in option_values) or len(set(option_values)) != 4:
        issues.append(_issue("count_options_not_four_distinct_integers", option_values=option_values))
    elif expected_count not in option_values:
        issues.append(_issue("count_options_missing_derived_count", expected_count=expected_count))
    elif not (min(option_values) < expected_count < max(option_values)) or max(option_values) - min(option_values) > 4:
        issues.append(
            _issue(
                "count_options_not_nearby_bracketing_values",
                expected_count=expected_count,
                option_values=option_values,
            )
        )

    gold_raw = qa.get("gold")
    gold_labels = [str(item) for item in gold_raw] if isinstance(gold_raw, list) else [str(gold_raw or "")]
    option_by_label = {
        str(option.get("id") or option.get("option_id") or ""): option for option in options
    }
    gold_value = (
        _parse_count_integer(option_by_label[gold_labels[0]].get("text"))
        if len(gold_labels) == 1 and gold_labels[0] in option_by_label
        else None
    )
    if gold_value != expected_count:
        issues.append(
            _issue(
                "count_gold_not_exact_included_count",
                expected_count=expected_count,
                gold_value=gold_value,
            )
        )
    declared_gold_value = _parse_count_integer(qa.get("gold_answer_text"))
    if qa.get("gold_answer_text") is not None and declared_gold_value != expected_count:
        issues.append(
            _issue(
                "count_gold_answer_text_not_exact_included_count",
                expected_count=expected_count,
                gold_answer_text=qa.get("gold_answer_text"),
            )
        )

    question_tokens = _question_dimension_tokens(qa.get("question") or qa.get("question_text"))
    missing_dimensions = [
        name
        for name, raw_dimension in zip(("set_id", "entity", "slot"), target_dimension)
        if not _question_names_dimension(question_tokens, raw_dimension)
    ]
    if missing_dimensions:
        issues.append(
            _issue(
                "count_question_missing_candidate_set_dimensions",
                missing_dimensions=missing_dimensions,
                set_id=target_dimension[0],
                entity=target_dimension[1],
                slot=target_dimension[2],
            )
        )
    return issues


def _comparison_derivation_issues(sample: dict[str, Any], qa: dict[str, Any]) -> list[dict[str, Any]]:
    if str(qa.get("qa_operation") or "") != "comparison":
        return []
    issues: list[dict[str, Any]] = []
    if str(qa.get("answer_type") or "") != "Number":
        return [_issue("comparison_requires_number_answer")]

    graph = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    events = [event for event in (graph.get("events") or []) if isinstance(event, dict)]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in events:
        slot = normalized_option_text(event.get("slot"))
        time_index = event.get("time_index")
        if slot and isinstance(time_index, int) and not isinstance(time_index, bool):
            grouped.setdefault((slot, time_index), []).append(event)
    valid_pairs: list[list[dict[str, Any]]] = []
    for group in grouped.values():
        active = [
            event
            for event in group
            if str(event.get("status") or "").casefold() in ACTIVE_STATUSES
        ]
        values = [event.get("value") for event in active]
        entities = {normalized_option_text(event.get("entity")) for event in active}
        if (
            len(active) == 2
            and len(entities - {""}) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            and values[0] != values[1]
        ):
            valid_pairs.append(active)
    if len(valid_pairs) != 1:
        return [_issue("comparison_unique_numeric_pair_missing", pair_count=len(valid_pairs))]

    pair = valid_pairs[0]
    pair_ids = {
        str(event.get("event_instance_id") or event.get("event_id") or "") for event in pair
    }
    if "" in pair_ids or len(pair_ids) != 2:
        return [_issue("comparison_pair_event_ids_missing")]
    query_plan = qa.get("query_plan") if isinstance(qa.get("query_plan"), dict) else {}
    for field in (
        "resolved_reasoning_event_instance_ids",
        "resolved_required_event_instance_ids",
        "resolved_supporting_event_instance_ids",
        "resolved_gold_event_instance_ids",
    ):
        ids = {str(item) for item in (query_plan.get(field) or []) if str(item)}
        if ids != pair_ids:
            issues.append(
                _issue(
                    "comparison_reference_group_not_exact_pair",
                    field=field,
                    missing_event_ids=sorted(pair_ids - ids),
                    extraneous_event_ids=sorted(ids - pair_ids),
                )
            )

    question = str(qa.get("question") or qa.get("question_text") or "")
    question_tokens = _question_dimension_tokens(question)
    missing_entities = [
        str(event.get("entity") or "")
        for event in pair
        if not _question_names_dimension(question_tokens, event.get("entity"))
    ]
    if missing_entities:
        issues.append(_issue("comparison_question_missing_pair_entities", entities=missing_entities))

    pair_slot = normalized_option_text(pair[0].get("slot"))
    required_slot_tokens = sorted(_comparison_slot_tokens(pair_slot))
    normalized_question_slot_tokens = {
        COMPARISON_SLOT_TOKEN_ALIASES.get(token, token) for token in question_tokens
    }
    missing_slot_tokens = sorted(
        set(required_slot_tokens) - normalized_question_slot_tokens
    )
    if missing_slot_tokens:
        issues.append(
            _issue(
                "comparison_question_missing_pair_slot",
                slot=pair_slot,
                required_slot_tokens=required_slot_tokens,
                missing_slot_tokens=missing_slot_tokens,
            )
        )

    pair_time_index = pair[0].get("time_index")
    ambiguous_entity_slots: list[dict[str, Any]] = []
    for pair_event in pair:
        entity = normalized_option_text(pair_event.get("entity"))
        time_indices = sorted(
            {
                event.get("time_index")
                for event in events
                if normalized_option_text(event.get("entity")) == entity
                and normalized_option_text(event.get("slot")) == pair_slot
                and isinstance(event.get("time_index"), int)
                and not isinstance(event.get("time_index"), bool)
            }
        )
        if len(time_indices) > 1:
            ambiguous_entity_slots.append(
                {
                    "entity": str(pair_event.get("entity") or ""),
                    "slot": pair_slot,
                    "time_indices": time_indices,
                }
            )
    if COMPARISON_INTERNAL_TIME_INDEX_RE.search(question):
        issues.append(_issue("comparison_question_exposes_internal_time_index"))
    if ambiguous_entity_slots and not COMPARISON_NATURAL_TIME_QUALIFIER_RE.search(question):
        issues.append(
            _issue(
                "comparison_question_missing_pair_time_qualifier",
                pair_time_index=pair_time_index,
                ambiguous_entity_slots=ambiguous_entity_slots,
            )
        )

    expected_difference = abs(int(pair[0]["value"]) - int(pair[1]["value"]))
    options = [option for option in (qa.get("options") or []) if isinstance(option, dict)]
    option_values = [_parse_count_integer(option.get("text")) for option in options]
    if len(options) != 4 or any(value is None for value in option_values) or len(set(option_values)) != 4:
        issues.append(_issue("comparison_options_not_four_distinct_integers", option_values=option_values))
    elif expected_difference not in option_values:
        issues.append(
            _issue(
                "comparison_options_missing_derived_difference",
                expected_difference=expected_difference,
            )
        )
    elif (
        not (min(option_values) < expected_difference < max(option_values))
        or max(option_values) - min(option_values) > 4
    ):
        issues.append(
            _issue(
                "comparison_options_not_nearby_bracketing_values",
                expected_difference=expected_difference,
                option_values=option_values,
            )
        )

    gold_raw = qa.get("gold")
    gold_labels = [str(item) for item in gold_raw] if isinstance(gold_raw, list) else [str(gold_raw or "")]
    option_by_label = {
        str(option.get("id") or option.get("option_id") or ""): option for option in options
    }
    gold_value = (
        _parse_count_integer(option_by_label[gold_labels[0]].get("text"))
        if len(gold_labels) == 1 and gold_labels[0] in option_by_label
        else None
    )
    if gold_value != expected_difference:
        issues.append(
            _issue(
                "comparison_gold_not_derived_difference",
                expected_difference=expected_difference,
                gold_value=gold_value,
            )
        )
    declared_gold = _parse_count_integer(qa.get("gold_answer_text"))
    if qa.get("gold_answer_text") is not None and declared_gold != expected_difference:
        issues.append(
            _issue(
                "comparison_gold_answer_text_not_derived_difference",
                expected_difference=expected_difference,
                gold_answer_text=qa.get("gold_answer_text"),
            )
        )
    return issues


def _target_answer_type_issues(
    sample: dict[str, Any], qa: dict[str, Any]
) -> list[dict[str, Any]]:
    target = sample.get("stage64_target")
    if not isinstance(target, dict):
        return []
    preferred = str(target.get("preferred_answer_type") or "").strip()
    actual = str(qa.get("answer_type") or "").strip()
    if preferred and actual != preferred:
        return [
            _issue(
                "stage64_preferred_answer_type_mismatch",
                qa_operation=qa.get("qa_operation"),
                preferred_answer_type=preferred,
                answer_type=actual,
            )
        ]
    return []


def _bridge_derivation_issues(
    sample: dict[str, Any], qa: dict[str, Any]
) -> list[dict[str, Any]]:
    if str(qa.get("qa_operation") or "") != "bridge":
        return []
    events, by_id, aliases = _compiled_event_index(sample)
    option_by_label, option_events = _option_events(qa, by_id, aliases)
    gold_label = _single_gold_label(qa)
    gold_event = option_events.get(gold_label)
    if gold_event is None:
        return [_issue("bridge_gold_endpoint_unresolved", gold=gold_label)]
    endpoint = _event_primary_id(gold_event)
    dependencies = _dependency_map(events, aliases)
    closure = _dependency_closure(endpoint, dependencies)
    reasoning_ids = set(_reasoning_event_ids(qa, aliases))
    issues: list[dict[str, Any]] = []
    if len(closure) <= 1:
        issues.append(
            _issue(
                "bridge_gold_endpoint_has_no_dependency_closure",
                gold_endpoint=endpoint,
            )
        )
    missing = sorted(closure - reasoning_ids)
    if missing:
        issues.append(
            _issue(
                "bridge_reasoning_missing_dependency_closure",
                gold_endpoint=endpoint,
                dependency_closure=sorted(closure),
                missing_event_ids=missing,
                reasoning_event_ids=sorted(reasoning_ids),
            )
        )
    if reasoning_ids == {endpoint}:
        issues.append(
            _issue("bridge_single_endpoint_direct_lookup", gold_endpoint=endpoint)
        )

    gold_option = option_by_label.get(gold_label) or {}
    gold_surface = normalized_option_text(
        gold_option.get("text") or qa.get("gold_answer_text")
    )
    if gold_surface:
        turn_text_by_id = _dialogue_turn_text_by_id(sample)
        endpoint_turn_ids = {
            str(value)
            for value in (
                gold_event.get("turn_ids")
                or gold_event.get("evidence_turn_ids")
                or []
            )
            if str(value)
        }
        copied_turn_ids = sorted(
            turn_id
            for turn_id in endpoint_turn_ids
            if gold_surface
            and gold_surface in normalized_option_text(turn_text_by_id.get(turn_id))
        )
        if copied_turn_ids:
            issues.append(
                _issue(
                    "bridge_gold_answer_copied_from_single_endpoint",
                    gold_endpoint=endpoint,
                    evidence_turn_ids=copied_turn_ids,
                )
            )
    return issues


def _cause_question_direction(question: Any) -> str | None:
    text = str(question or "").strip()
    effect_match = EFFECT_QUESTION_RE.search(text) is not None
    cause_match = CAUSE_QUESTION_RE.search(text) is not None
    if re.search(r"\bwhat\s+caused\b|^\s*why\b", text, flags=re.IGNORECASE):
        return "cause"
    if re.search(
        r"\bwhat\s+(?:was\s+)?(?:the\s+)?(?:effect|consequence|result)\b|\bwhat\s+happened\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "effect"
    if cause_match and not effect_match:
        return "cause"
    if effect_match and not cause_match:
        return "effect"
    return None


def _cause_derivation_issues(
    sample: dict[str, Any], qa: dict[str, Any]
) -> list[dict[str, Any]]:
    if str(qa.get("qa_operation") or "") != "cause_consequence":
        return []
    events, by_id, aliases = _compiled_event_index(sample)
    _, option_events = _option_events(qa, by_id, aliases)
    gold_label = _single_gold_label(qa)
    gold_event = option_events.get(gold_label)
    reasoning_ids = set(_reasoning_event_ids(qa, aliases))
    dependencies = _dependency_map(events, aliases)
    directed_edges = sorted(
        (dependency, consequence)
        for consequence in reasoning_ids
        for dependency in dependencies.get(consequence, set())
        if dependency in reasoning_ids
    )
    issues: list[dict[str, Any]] = []
    if not directed_edges:
        issues.append(
            _issue(
                "cause_reasoning_path_missing_directed_edge",
                reasoning_event_ids=sorted(reasoning_ids),
            )
        )
    direction = _cause_question_direction(
        qa.get("question") or qa.get("question_text")
    )
    if direction is None:
        issues.append(_issue("cause_question_direction_ambiguous"))
        return issues
    if gold_event is None:
        issues.append(_issue("cause_gold_event_unresolved", gold=gold_label))
        return issues
    gold_id = _event_primary_id(gold_event)
    cause_ids = {cause for cause, _ in directed_edges}
    effect_ids = {effect for _, effect in directed_edges}
    expected_ids = cause_ids if direction == "cause" else effect_ids
    if gold_id not in expected_ids:
        issues.append(
            _issue(
                "cause_gold_endpoint_direction_mismatch",
                question_direction=direction,
                gold_event_id=gold_id,
                expected_event_ids=sorted(expected_ids),
                directed_edges=[list(edge) for edge in directed_edges],
            )
        )
    return issues


def _visibility_projection_candidates(
    option: dict[str, Any], event: dict[str, Any]
) -> set[str]:
    option_text = normalized_option_text(option.get("text"))
    candidates: set[str] = set()
    if option_text and option_text == normalized_option_text(event.get("entity")):
        candidates.add("entity")
    scope = event.get("scope") if isinstance(event.get("scope"), dict) else {}
    if option_text and option_text == normalized_option_text(scope.get("observer")):
        candidates.add("scope.observer")
    return candidates


def _visibility_derivation_issues(
    sample: dict[str, Any], qa: dict[str, Any]
) -> list[dict[str, Any]]:
    if str(qa.get("qa_operation") or "") != "visibility":
        return []
    events, by_id, aliases = _compiled_event_index(sample)
    option_by_label, option_events = _option_events(qa, by_id, aliases)
    issues: list[dict[str, Any]] = []
    candidate_sets: list[set[str]] = []
    unresolved_labels: list[str] = []
    for label, option in option_by_label.items():
        event = option_events.get(label)
        if event is None:
            unresolved_labels.append(label)
            continue
        candidates = _visibility_projection_candidates(option, event)
        if not candidates:
            unresolved_labels.append(label)
        candidate_sets.append(candidates)
    common_projection = (
        set.intersection(*candidate_sets) if candidate_sets and all(candidate_sets) else set()
    )
    if unresolved_labels or len(common_projection) != 1:
        issues.append(
            _issue(
                "visibility_option_projection_inconsistent",
                unresolved_option_labels=sorted(unresolved_labels),
                common_answer_projections=sorted(common_projection),
            )
        )
        return issues
    projection = next(iter(common_projection))
    gold_label = _single_gold_label(qa)
    gold_event = option_events.get(gold_label)
    if gold_event is None:
        return issues + [_issue("visibility_gold_event_unresolved", gold=gold_label)]
    gold_id = _event_primary_id(gold_event)
    gold_slot = normalized_option_text(gold_event.get("slot"))
    gold_value = _graph_answer_value_key(gold_event)

    valid_projection_values: set[str] = set()
    invalid_scope_ids: list[str] = []
    for event in events:
        if normalized_option_text(event.get("slot")) != gold_slot:
            continue
        if _graph_answer_value_key(event) != gold_value:
            continue
        scope = event.get("scope") if isinstance(event.get("scope"), dict) else {}
        observer = normalized_option_text(scope.get("observer"))
        permission = normalized_option_text(scope.get("permission"))
        if not observer or not permission:
            invalid_scope_ids.append(_event_primary_id(event))
            continue
        if str(event.get("status") or "").casefold() not in ACTIVE_STATUSES:
            continue
        projected = (
            normalized_option_text(event.get("entity"))
            if projection == "entity"
            else observer
        )
        if projected:
            valid_projection_values.add(projected)
    gold_option_value = normalized_option_text(
        (option_by_label.get(gold_label) or {}).get("text")
    )
    if len(valid_projection_values) != 1 or gold_option_value not in valid_projection_values:
        issues.append(
            _issue(
                "visibility_gold_not_unique_in_valid_scope",
                answer_projection=projection,
                gold_value=gold_option_value,
                valid_projection_values=sorted(valid_projection_values),
                invalid_scope_event_ids=sorted(value for value in invalid_scope_ids if value),
            )
        )

    reasoning_ids = set(_reasoning_event_ids(qa, aliases))
    scoped_reasoning_ids = {
        event_id
        for event_id in reasoning_ids
        if isinstance((by_id.get(event_id) or {}).get("scope"), dict)
        and bool((by_id.get(event_id) or {}).get("scope"))
    }
    same_fact_ids = {
        event_id
        for event_id in reasoning_ids
        if normalized_option_text((by_id.get(event_id) or {}).get("slot")) == gold_slot
        and _graph_answer_value_key(by_id.get(event_id) or {}) == gold_value
    }
    if (
        gold_id not in reasoning_ids
        or len(reasoning_ids) < 2
        or not scoped_reasoning_ids
        or not same_fact_ids
    ):
        issues.append(
            _issue(
                "visibility_reasoning_missing_fact_scope_refs",
                gold_event_id=gold_id,
                reasoning_event_ids=sorted(reasoning_ids),
                scoped_reasoning_event_ids=sorted(scoped_reasoning_ids),
                fact_reasoning_event_ids=sorted(same_fact_ids),
            )
        )
    return issues


def _relation_derivation_issues(
    sample: dict[str, Any], qa: dict[str, Any]
) -> list[dict[str, Any]]:
    if str(qa.get("qa_operation") or "") != "relation_lookup":
        return []
    _, by_id, aliases = _compiled_event_index(sample)
    _, option_events = _option_events(qa, by_id, aliases)
    gold_label = _single_gold_label(qa)
    gold_event = option_events.get(gold_label)
    if gold_event is None:
        return [_issue("relation_gold_event_unresolved", gold=gold_label)]
    gold_id = _event_primary_id(gold_event)
    gold_entity = normalized_option_text(gold_event.get("entity"))
    gold_slot = normalized_option_text(gold_event.get("slot"))
    reasoning_ids = set(_reasoning_event_ids(qa, aliases))
    same_entity_competitors = {
        event_id
        for event_id in reasoning_ids
        if event_id != gold_id
        and normalized_option_text((by_id.get(event_id) or {}).get("entity")) == gold_entity
        and normalized_option_text((by_id.get(event_id) or {}).get("slot")) == gold_slot
    }
    wrong_entity_options = {
        _event_primary_id(event)
        for label, event in option_events.items()
        if label != gold_label
        and normalized_option_text(event.get("slot")) == gold_slot
        and normalized_option_text(event.get("entity")) != gold_entity
    }
    wrong_entity_options.discard("")
    issues: list[dict[str, Any]] = []
    if gold_id not in reasoning_ids or not same_entity_competitors:
        issues.append(
            _issue(
                "relation_reasoning_missing_dual_constraint_path",
                gold_event_id=gold_id,
                reasoning_event_ids=sorted(reasoning_ids),
                same_entity_same_relation_competitors=sorted(same_entity_competitors),
            )
        )
    if not wrong_entity_options or not wrong_entity_options.intersection(reasoning_ids):
        issues.append(
            _issue(
                "relation_missing_wrong_entity_hard_negative",
                gold_entity=gold_entity,
                gold_slot=gold_slot,
                wrong_entity_option_event_ids=sorted(wrong_entity_options),
                reasoning_event_ids=sorted(reasoning_ids),
            )
        )
    question_tokens = _question_dimension_tokens(
        qa.get("question") or qa.get("question_text")
    )
    missing_dimensions = []
    if not _question_names_dimension(question_tokens, gold_event.get("entity")):
        missing_dimensions.append("entity")
    if not _question_names_dimension(question_tokens, gold_event.get("slot")):
        missing_dimensions.append("relation")
    if missing_dimensions:
        issues.append(
            _issue(
                "relation_question_missing_dual_constraints",
                missing_dimensions=missing_dimensions,
                entity=gold_event.get("entity"),
                slot=gold_event.get("slot"),
            )
        )
    return issues


def _longest_dependency_chain_length(
    nodes: set[str], dependencies: dict[str, set[str]]
) -> int:
    memo: dict[str, int] = {}

    def visit(node: str, active: set[str]) -> int:
        if node in memo:
            return memo[node]
        if node in active:
            return 0
        parents = dependencies.get(node, set()).intersection(nodes)
        result = 1 + max((visit(parent, active | {node}) for parent in parents), default=0)
        memo[node] = result
        return result

    return max((visit(node, set()) for node in nodes), default=0)


def _temporal_reasoning_issues(
    sample: dict[str, Any], qa: dict[str, Any]
) -> list[dict[str, Any]]:
    if str(qa.get("qa_operation") or "") != "temporal_state_update":
        return []
    target = sample.get("stage64_target")
    strict_target = isinstance(target, dict) and str(target.get("qa_operation") or "") == "temporal_state_update"
    hop_count = (target or {}).get("hop_count") if strict_target else qa.get("hop_count")
    if isinstance(hop_count, bool) or not isinstance(hop_count, int) or hop_count < 2:
        return []
    events, by_id, aliases = _compiled_event_index(sample)
    reasoning_ids = set(_reasoning_event_ids(qa, aliases))
    dependencies = _dependency_map(events, aliases)
    longest_chain = _longest_dependency_chain_length(reasoning_ids, dependencies)
    issues: list[dict[str, Any]] = []
    if longest_chain < hop_count:
        issues.append(
            _issue(
                "temporal_reasoning_path_not_dependency_chain",
                required_hop_count=hop_count,
                longest_dependency_chain=longest_chain,
                reasoning_event_ids=sorted(reasoning_ids),
            )
        )

    _, option_events = _option_events(qa, by_id, aliases)
    gold_event = option_events.get(_single_gold_label(qa))
    if gold_event is not None:
        turn_text_by_id = _dialogue_turn_text_by_id(sample)
        turn_ids = {
            str(value)
            for value in (
                gold_event.get("turn_ids")
                or gold_event.get("evidence_turn_ids")
                or []
            )
            if str(value)
        }
        shortcut_turn_ids = sorted(
            turn_id
            for turn_id in turn_ids
            if TEMPORAL_DIRECT_GOLD_CUE_RE.search(turn_text_by_id.get(turn_id, ""))
        )
        if shortcut_turn_ids:
            issues.append(
                _issue(
                    "temporal_gold_evidence_direct_status_hint",
                    gold_event_id=_event_primary_id(gold_event),
                    evidence_turn_ids=shortcut_turn_ids,
                )
            )
    return issues


def _graph_hardness_issues(sample: dict[str, Any], qa: dict[str, Any]) -> list[dict[str, Any]]:
    graph = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    events = graph.get("events") if isinstance(graph.get("events"), list) else []
    if not events:
        plan = sample.get("current_event_plan") if isinstance(sample.get("current_event_plan"), dict) else {}
        events = plan.get("events") if isinstance(plan.get("events"), list) else []
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        for key in ("event_instance_id", "event_id", "node_id"):
            value = str(event.get(key) or "")
            if value:
                by_id[value] = event

    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    gold_raw = qa.get("gold")
    gold = [str(item) for item in gold_raw] if isinstance(gold_raw, list) else [str(gold_raw or "")]
    option_events: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("id") or option.get("option_id") or "")
        ref = str(
            option.get("source_event_instance_id")
            or option.get("event_instance_id")
            or option.get("source_event_id")
            or ""
        )
        event = by_id.get(ref)
        if event is None:
            unresolved.append(label)
        else:
            option_events[label] = event
    issues: list[dict[str, Any]] = []
    if unresolved:
        issues.append(_issue("option_source_event_missing_from_graph", labels=sorted(unresolved)))
        return issues

    operation = str(qa.get("qa_operation") or "")
    if operation == "temporal_state_update":
        value_to_labels: dict[str, list[str]] = {}
        surface_to_labels: dict[str, list[str]] = {}
        annotated_labels: list[str] = []
        option_by_label = {
            str(option.get("id") or option.get("option_id") or ""): option
            for option in options
            if isinstance(option, dict)
        }
        for label, event in option_events.items():
            value_key = _graph_answer_value_key(event)
            if value_key:
                value_to_labels.setdefault(value_key, []).append(label)
            option_text = str((option_by_label.get(label) or {}).get("text") or "")
            surface_key = _temporal_option_surface_key(option_text)
            if surface_key:
                surface_to_labels.setdefault(surface_key, []).append(label)
            if TEMPORAL_STATUS_PARENTHETICAL_RE.search(option_text):
                annotated_labels.append(label)
        duplicate_value_labels = [
            labels for labels in value_to_labels.values() if len(labels) > 1
        ]
        if duplicate_value_labels:
            issues.append(
                _issue(
                    "temporal_duplicate_graph_answer_values",
                    duplicate_option_labels=duplicate_value_labels,
                )
            )
        duplicate_surface_labels = [
            labels for labels in surface_to_labels.values() if len(labels) > 1
        ]
        if duplicate_surface_labels:
            issues.append(
                _issue(
                    "temporal_duplicate_option_value_surface",
                    duplicate_option_labels=duplicate_surface_labels,
                )
            )
        if annotated_labels:
            issues.append(
                _issue(
                    "temporal_option_status_annotation",
                    labels=sorted(annotated_labels),
                )
            )

    raw_slot_labels = sorted(
        label
        for label, event in option_events.items()
        if str(event.get("slot") or "").strip().casefold().startswith("raw_qa")
    )
    if raw_slot_labels:
        issues.append(_issue("generic_raw_qa_slot", labels=raw_slot_labels))

    gold_label = gold[0] if len(gold) == 1 else ""
    gold_event = option_events.get(gold_label)
    answer_type = str(qa.get("answer_type") or "")
    if gold_event is not None and answer_type not in {"YesNo", "Number"}:
        gold_entity = normalized_option_text(gold_event.get("entity"))
        gold_slot = normalized_option_text(gold_event.get("slot"))
        similar_distractors = 0
        for label, event in option_events.items():
            if label == gold_label:
                continue
            entity = normalized_option_text(event.get("entity"))
            slot = normalized_option_text(event.get("slot"))
            if (gold_entity and entity == gold_entity) or (gold_slot and slot == gold_slot):
                similar_distractors += 1
        if similar_distractors < 1:
            issues.append(_issue("unrelated_distractor_events", gold=gold_label))

    query_plan = qa.get("query_plan") if isinstance(qa.get("query_plan"), dict) else {}
    reasoning_path = [str(item) for item in (query_plan.get("reasoning_anchor_path") or []) if str(item)]
    minimum_reasoning_anchors = {
        "bridge": 2,
        "comparison": 2,
        "count": 4,
        "cause_consequence": 2,
        "relation_lookup": 2,
    }.get(operation)
    if minimum_reasoning_anchors is not None and len(set(reasoning_path)) < minimum_reasoning_anchors:
        issues.append(
            _issue(
                "insufficient_reasoning_anchors",
                qa_operation=operation,
                reasoning_anchor_count=len(set(reasoning_path)),
                minimum_reasoning_anchor_count=minimum_reasoning_anchors,
            )
        )

    if operation == "temporal_state_update" and gold_event is not None:
        gold_entity = normalized_option_text(gold_event.get("entity"))
        gold_slot = normalized_option_text(gold_event.get("slot"))
        competing = [
            event
            for event in option_events.values()
            if normalized_option_text(event.get("entity")) == gold_entity
            and normalized_option_text(event.get("slot")) == gold_slot
        ]
        statuses = {str(event.get("status") or "").casefold() for event in competing}
        if len(competing) < 3 or not statuses.intersection(NON_ACTIVE_STATUSES):
            issues.append(
                _issue(
                    "temporal_distractors_not_same_slot_updates",
                    competing_option_count=len(competing),
                    statuses=sorted(statuses),
                )
            )
        if str(gold_event.get("status") or "").casefold() not in {"active", "restored"}:
            issues.append(_issue("temporal_gold_not_current_active", status=gold_event.get("status")))
        graph_competing = [
            event
            for event in events
            if isinstance(event, dict)
            and normalized_option_text(event.get("entity")) == gold_entity
            and normalized_option_text(event.get("slot")) == gold_slot
        ]
        current_events = [
            event
            for event in graph_competing
            if str(event.get("status") or "").casefold() in ACTIVE_STATUSES
        ]
        valid_current = current_events[0] if len(current_events) == 1 else None
        valid_times = [
            event.get("time_index")
            for event in graph_competing
            if isinstance(event.get("time_index"), int) and not isinstance(event.get("time_index"), bool)
        ]
        highest_time = max(valid_times) if valid_times else None
        gold_time = gold_event.get("time_index")
        current_time = valid_current.get("time_index") if valid_current is not None else None
        forbid_max_time = qa.get("forbid_max_time_shortcut") is True
        if forbid_max_time:
            strategy = str(qa.get("hard_negative_strategy") or "").strip()
            required_statuses = {
                "same_slot_canceled_value": {"canceled"},
                "same_slot_outdated_value": {"outdated"},
                "same_slot_report_or_belief_value": {"report", "belief"},
            }.get(strategy, set())
            later_invalid = [
                event
                for event in graph_competing
                if isinstance(event.get("time_index"), int)
                and not isinstance(event.get("time_index"), bool)
                and isinstance(gold_time, int)
                and not isinstance(gold_time, bool)
                and event["time_index"] > gold_time
                and str(event.get("status") or "").casefold()
                in required_statuses
            ]
            later_invalid_ids = {
                str(event.get("event_instance_id") or event.get("event_id") or "")
                for event in later_invalid
            } - {""}
            distractor_ids = {
                str(item)
                for item in (
                    query_plan.get("resolved_distractor_event_instance_ids")
                    or query_plan.get("distractor_anchor_refs")
                    or []
                )
                if str(item)
            }
            reasoning_ids = set(reasoning_path)
            if (
                valid_current is None
                or highest_time is None
                or gold_event is not valid_current
                or not isinstance(gold_time, int)
                or gold_time >= highest_time
            ):
                issues.append(
                    _issue(
                        "temporal_anti_shortcut_gold_not_earlier_current",
                        gold=gold_label,
                        gold_time_index=gold_time,
                        highest_time_index=highest_time,
                        current_event_count=len(current_events),
                    )
                )
            if not required_statuses or not later_invalid_ids:
                issues.append(
                    _issue(
                        "temporal_anti_shortcut_missing_later_strategy_update",
                        hard_negative_strategy=strategy,
                        required_statuses=sorted(required_statuses),
                    )
                )
            elif not later_invalid_ids.intersection(reasoning_ids) or not later_invalid_ids.intersection(
                distractor_ids
            ):
                issues.append(
                    _issue(
                        "temporal_anti_shortcut_later_update_not_referenced",
                        later_invalid_event_ids=sorted(later_invalid_ids),
                        reasoning_event_ids=sorted(reasoning_ids),
                        distractor_event_ids=sorted(distractor_ids),
                    )
                )
        elif (
            valid_current is None
            or highest_time is None
            or current_time != highest_time
            or gold_event is not valid_current
            or gold_time != highest_time
        ):
            issues.append(
                _issue(
                    "temporal_gold_not_final_update",
                    gold=gold_label,
                    gold_time_index=gold_time,
                    highest_time_index=highest_time,
                    current_event_count=len(current_events),
                )
            )
    return issues


def semantic_quality_issues(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Return blocking semantic issues for an otherwise valid Stage64 sample."""

    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    if not qa.get("qa_operation") and not qa.get("answer_type"):
        return []
    issues = [
        *_answer_surface_issues(qa),
        *_question_answer_type_issues(qa),
        *_operation_semantic_issues(qa),
        *_compiled_operation_contract_issues(sample, qa),
        *_target_answer_type_issues(sample, qa),
        *_bridge_derivation_issues(sample, qa),
        *_cause_derivation_issues(sample, qa),
        *_visibility_derivation_issues(sample, qa),
        *_relation_derivation_issues(sample, qa),
        *_comparison_derivation_issues(sample, qa),
        *_count_derivation_issues(sample, qa),
        *_temporal_reasoning_issues(sample, qa),
        *_graph_hardness_issues(sample, qa),
        *_dialogue_graph_fact_repetition_issues(sample),
    ]
    # One detailed example per reason is enough for gating and summary counts.
    deduplicated: dict[str, dict[str, Any]] = {}
    for issue in issues:
        deduplicated.setdefault(str(issue.get("reason") or "unknown"), issue)
    return list(deduplicated.values())
