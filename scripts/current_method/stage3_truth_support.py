"""Independent typed frozen-support rendering for Truth-first Stage3.

This module deliberately does not import the general Claim IR validator.  It
renders the locally frozen claim into one deterministic support clause and
then validates semantic-path coverage with a separate, claim-type-aware
implementation.  The API may write surrounding discourse, but it never owns
the quote, its claim id, or any of the typed semantic roles checked here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

try:
    from scripts.current_method import stage3_truth_support_validator as _validator
except ModuleNotFoundError:  # pragma: no cover - direct script import fallback
    from current_method import stage3_truth_support_validator as _validator


SCHEMA_VERSION = "stage3_typed_frozen_support_v1"
VALIDATION_SCHEMA_VERSION = _validator.VALIDATION_SCHEMA_VERSION

_CORE_FIELDS = (
    "subject",
    "predicate",
    "objects",
    "polarity",
    "status",
    "time",
    "condition",
    "scope",
)
_DESCRIPTOR_FIELDS = (
    "entity",
    "slot",
    "value",
    "status",
    "time_index",
    "global_time_index",
)
_TIME_LABELS = {
    "from_time_index": "from local step",
    "cause_time_index": "cause local step",
    "time_index": "local step",
    "global_time_index": "global step",
}
_SPACE_RE = re.compile(r"\s+")
_NEGATION_RE = re.compile(r"\b(?:not|never|no)\b", re.IGNORECASE)
_ACTIVE_STATUS = "active"
_INVALIDATION_RE = re.compile(
    r"\b(?:cancel(?:ed|led)?|expired|inactive|revoked|superseded|unconfirmed|tentative)\b",
    re.IGNORECASE,
)


class TruthSupportError(ValueError):
    """Raised when a frozen claim cannot be rendered without semantic loss."""


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TruthSupportError(f"value is not finite JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def claim_event_binding(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Expose graph endpoints separately from the visible natural-language quote."""

    subject_event_id = _text(claim.get("subject_event_id"))
    object_event_ids = [
        _text(value)
        for value in claim.get("object_event_ids", [])
        if _text(value)
    ] if isinstance(claim.get("object_event_ids"), list) else []
    source_event_ids = [
        _text(value)
        for value in claim.get("source_event_ids", [])
        if _text(value)
    ] if isinstance(claim.get("source_event_ids"), list) else []
    endpoint_event_ids = list(
        dict.fromkeys([subject_event_id, *object_event_ids, *source_event_ids])
    )
    return {
        "subject_event_id": subject_event_id or None,
        "object_event_ids": object_event_ids,
        "source_event_ids": source_event_ids,
        "endpoint_event_ids": [value for value in endpoint_event_ids if value],
    }


def _text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _token(value: Any) -> str:
    return re.sub(r"[\W_]+", "_", _text(value).casefold()).strip("_")


def _display(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _possessive(value: Any) -> str:
    text = _display(value)
    return f"{text}'" if text.casefold().endswith("s") else f"{text}'s"


def _predicate(value: Any) -> str:
    return _text(value).replace("_", " ").replace("-", " ")


def _claim_type(claim: Mapping[str, Any]) -> str:
    kind = _token(claim.get("claim_kind"))
    claim_type = _token(claim.get("claim_type"))
    if kind == "node" or claim_type == "node":
        return "node"
    return claim_type


def _canonical(claim: Mapping[str, Any]) -> Mapping[str, Any]:
    canonical = claim.get("canonical_support_claim")
    if not isinstance(canonical, Mapping):
        raise TruthSupportError("claim requires canonical_support_claim")
    missing = [field for field in _CORE_FIELDS if field not in canonical]
    if missing:
        raise TruthSupportError(f"canonical support is missing fields: {missing}")
    objects = canonical.get("objects")
    if not isinstance(objects, list) or not objects:
        raise TruthSupportError("canonical support requires a non-empty objects array")
    return canonical


def _require_nonempty(
    value: Mapping[str, Any], fields: Sequence[str], *, role: str
) -> None:
    missing = [
        field for field in fields
        if field not in value or value.get(field) in (None, "")
    ]
    if missing:
        raise TruthSupportError(f"{role} is missing required fields: {missing}")


def _require_descriptor(value: Any, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TruthSupportError(f"{role} must be an object")
    _require_nonempty(
        value,
        ("entity", "slot", "value", "status", "time_index", "global_time_index"),
        role=role,
    )
    return value


def _require_renderable_claim(
    claim_type: str, canonical: Mapping[str, Any]
) -> None:
    _require_nonempty(
        canonical,
        ("subject", "predicate", "polarity", "status"),
        role=f"{claim_type} claim",
    )
    if not isinstance(canonical.get("time"), Mapping):
        raise TruthSupportError(f"{claim_type} claim requires a time object")
    if claim_type == "membership":
        _require_nonempty(canonical, ("entity", "slot"), role="membership claim")
        if not isinstance(canonical.get("included"), bool):
            raise TruthSupportError("membership claim requires boolean included")
    elif claim_type == "visibility":
        _require_nonempty(
            canonical,
            ("owner", "access_class", "condition"),
            role="visibility claim",
        )
        scope = canonical.get("scope")
        if not isinstance(scope, Mapping):
            raise TruthSupportError("visibility claim requires typed scope")
        _require_nonempty(
            scope,
            ("observer", "permission", "condition"),
            role="visibility scope",
        )
    elif claim_type == "status_transition":
        if len(_objects(canonical)) != 2:
            raise TruthSupportError(
                "status_transition requires exactly two ordered values"
            )
        _require_nonempty(
            canonical,
            ("slot", "from_status", "to_status"),
            role="status_transition claim",
        )
        _require_nonempty(
            canonical["time"],
            ("from_time_index", "time_index", "global_time_index"),
            role="status_transition time",
        )
    elif claim_type == "comparison":
        _require_nonempty(
            canonical,
            ("slot", "left_value", "right_value", "absolute_difference"),
            role="comparison claim",
        )
        if len(_objects(canonical)) != 2:
            raise TruthSupportError(
                "comparison requires right entity and derived difference objects"
            )
        for field in ("left_value", "right_value", "absolute_difference"):
            value = canonical[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TruthSupportError(
                    f"comparison {field} must be numeric"
                )
    elif claim_type == "cause":
        cause = _require_descriptor(
            canonical.get("cause_event"), role="cause event"
        )
        effect = _require_descriptor(
            canonical.get("effect_event"), role="effect event"
        )
        if int(cause["time_index"]) >= int(effect["time_index"]):
            raise TruthSupportError("cause must precede effect")
        _require_nonempty(
            canonical["time"],
            ("cause_time_index", "time_index", "global_time_index"),
            role="cause time",
        )
    elif claim_type == "dependency":
        _require_descriptor(
            canonical.get("subject_event"), role="dependency subject event"
        )
        targets = canonical.get("object_events")
        if not isinstance(targets, list) or not targets:
            raise TruthSupportError(
                "dependency requires non-empty object_events"
            )
        for index, target in enumerate(targets):
            _require_descriptor(
                target, role=f"dependency object event {index + 1}"
            )
    elif claim_type != "node":
        raise TruthSupportError(
            f"unsupported truth support claim_type: {claim_type!r}"
        )


def _objects(canonical: Mapping[str, Any]) -> list[Any]:
    return list(canonical.get("objects") or [])


def _time_sentence(canonical: Mapping[str, Any]) -> str:
    time = canonical.get("time")
    if not isinstance(time, Mapping):
        return ""
    natural_values = [
        _display(time[field])
        for field in ("label", "text", "date", "timestamp", "local_time", "global_time")
        if time.get(field) not in (None, "")
    ]
    return (
        f"This was discussed at {' and '.join(natural_values)}."
        if natural_values
        else ""
    )


def _scope_leaves(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        rows: list[tuple[tuple[str, ...], Any]] = []
        for key in sorted(value, key=str):
            rows.extend(_scope_leaves(value[key], (*path, _token(key))))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_scope_leaves(item, (*path, str(index))))
        return rows
    return [(path, value)]


def _scope_sentence(canonical: Mapping[str, Any]) -> str:
    leaves = _scope_leaves(canonical.get("scope"))
    if not leaves:
        return ""
    condition = _display(canonical.get("condition"))
    values = []
    for _path, value in leaves:
        surface = _display(value)
        if surface and surface != condition and surface not in values:
            values.append(surface)
    return f"This applies only to {' and '.join(values)}." if values else ""


def _status_sentence(status: Any) -> str:
    return {
        "belief": "The participants remain unsure whether this is true.",
        "report": "The participants treat this as an unconfirmed report.",
        "reported": "The participants treat this as an unconfirmed report.",
        "outdated": "This is outdated and no longer applies.",
        "canceled": "This was canceled before taking effect.",
        "cancelled": "This was canceled before taking effect.",
        "expired": "This expired and no longer applies.",
        "scope_limited": "This applies only within the stated limits.",
        "proposed": "This is only a proposal and has not taken effect.",
    }.get(_token(status), f"This remains {_predicate(status)}.")


def _qualifier_sentences(
    canonical: Mapping[str, Any], claim_type: str
) -> list[str]:
    rows: list[str] = []
    status = canonical.get("status")
    # An affirmative fact clause is the textual realization of ``active``.
    # Spelling the state out leaks a direct temporal-gold cue to Stage64.
    if status is not None and status != "" and _token(status) != _ACTIVE_STATUS:
        rows.append(_status_sentence(status))
    timing = _time_sentence(canonical)
    if timing:
        rows.append(timing)
    condition = canonical.get("condition")
    if (
        claim_type != "visibility"
        and condition is not None
        and condition != ""
    ):
        rows.append(f"This applies only if {_display(condition)}.")
    scope = "" if claim_type == "visibility" else _scope_sentence(canonical)
    if scope:
        rows.append(scope)
    return rows


def _event_fact(descriptor: Mapping[str, Any]) -> str:
    entity = _display(descriptor.get("entity"))
    slot = _predicate(descriptor.get("slot"))
    value = _display(descriptor.get("value"))
    fact = f"{_possessive(entity)} {slot} is {value}"
    status = descriptor.get("status")
    if status not in (None, "") and _token(status) != _ACTIVE_STATUS:
        fact += f", and {_status_sentence(status).rstrip('.').casefold()}"
    return fact


def _render_node(canonical: Mapping[str, Any], variant: int) -> str:
    subject = _display(canonical["subject"])
    predicate = _predicate(canonical["predicate"])
    object_text = " and ".join(_display(value) for value in _objects(canonical))
    negative = _token(canonical.get("polarity")) == "negative"
    verbs = (
        "lists",
        "describes",
        "identifies",
    )
    base_verbs = ("list", "describe", "identify")
    if negative:
        return (
            f"The discussion does not {base_verbs[variant]} "
            f"{_possessive(subject)} {predicate} as {object_text}."
        )
    return (
        f"The discussion {verbs[variant]} {_possessive(subject)} {predicate} "
        f"as {object_text}."
    )


def _render_membership(canonical: Mapping[str, Any], variant: int) -> str:
    member = _display(canonical["subject"])
    set_name = _display(_objects(canonical)[0])
    included = canonical.get("included")
    negative = _token(canonical.get("polarity")) == "negative"
    if isinstance(included, bool):
        negative = not included
    relation = "is not a member of" if negative else "is a member of"
    entity = _display(canonical["entity"])
    slot = _predicate(canonical["slot"])
    if variant == 0:
        return (
            f"{member}, listed under {_possessive(entity)} {slot}, "
            f"{relation} {set_name}."
        )
    if variant == 1:
        return f"For {_possessive(entity)} {slot}, {member} {relation} {set_name}."
    return (
        f"Within {_possessive(entity)} {slot} records, {member} "
        f"{relation} {set_name}."
    )


def _render_visibility(canonical: Mapping[str, Any], variant: int) -> str:
    observer = _display(canonical["subject"])
    action = _predicate(canonical["predicate"])
    resource = _display(_objects(canonical)[0])
    negative = _token(canonical.get("polarity")) == "negative"
    modal = "may not" if negative else "may"
    owner = _display(canonical["owner"])
    access_class = _display(canonical["access_class"])
    condition = _display(canonical["condition"])
    if variant == 0:
        return (
            f"{observer} {modal} {action} {resource} only under this "
            f"requirement: {condition}; {owner} keeps that resource under "
            f"{access_class} access."
        )
    if variant == 1:
        return (
            f"For {observer} to {action} {resource}, the requirement is "
            f"{condition}; {owner} keeps the resource under {access_class} access."
        )
    return (
        f"{observer} {modal} {action} {resource} once the following "
        f"requirement is satisfied: {condition}; {owner} keeps it under "
        f"{access_class} access."
    )


def _transition_value(value: Any, status: Any) -> str:
    status_key = _token(status)
    if status_key == _ACTIVE_STATUS:
        return _display(value)
    modifier = {
        "belief": "uncertain",
        "report": "reported",
        "reported": "reported",
        "proposed": "proposed",
        "tentative": "tentative",
        "outdated": "outdated",
        "canceled": "canceled",
        "cancelled": "canceled",
        "expired": "expired",
        "inactive": "inactive",
        "revoked": "revoked",
        "superseded": "superseded",
    }.get(status_key, _predicate(status))
    return f"the {modifier} {_display(value)} entry"


def _render_transition(canonical: Mapping[str, Any], variant: int) -> str:
    old_value, new_value = _objects(canonical)[:2]
    subject = _display(canonical["subject"])
    slot = _predicate(canonical.get("slot") or canonical.get("predicate"))
    from_status = canonical.get("from_status")
    to_status = canonical.get("to_status") or canonical.get("status")
    verb = ("transitions", "changes", "moves")[variant]
    old_clause = _transition_value(old_value, from_status)
    new_clause = _transition_value(new_value, to_status)
    return (
        f"{_possessive(subject)} {slot} {verb} from {old_clause} "
        f"to {new_clause}."
    )


def _render_comparison(canonical: Mapping[str, Any], variant: int) -> str:
    right_entity = _display(_objects(canonical)[0])
    difference = canonical.get("absolute_difference", _objects(canonical)[-1])
    return (
        f"For {_predicate(canonical.get('slot'))}, "
        f"{_display(canonical['subject'])} has {_display(canonical.get('left_value'))}, "
        f"while {right_entity} has {_display(canonical.get('right_value'))}; "
        f"the absolute difference is {_display(difference)}."
    )


def _render_cause(canonical: Mapping[str, Any], variant: int) -> str:
    cause = canonical.get("cause_event")
    effect = canonical.get("effect_event")
    if not isinstance(cause, Mapping) or not isinstance(effect, Mapping):
        raise TruthSupportError("cause claim requires cause_event and effect_event descriptors")
    cause_fact = _event_fact(cause)
    effect_entity = _possessive(effect.get("entity"))
    effect_slot = _predicate(effect.get("slot"))
    effect_value = _display(effect.get("value"))
    if variant == 0:
        outcome = f"caused {effect_entity} {effect_slot} to become {effect_value}"
    elif variant == 1:
        outcome = f"led to {effect_entity} {effect_slot} becoming {effect_value}"
    else:
        outcome = f"resulted in {effect_entity} {effect_slot} becoming {effect_value}"
    effect_status = effect.get("status")
    if effect_status not in (None, "") and _token(effect_status) != _ACTIVE_STATUS:
        outcome += f", while {_status_sentence(effect_status).rstrip('.').casefold()}"
    return f"The fact that {cause_fact} {outcome}."


def _render_dependency(canonical: Mapping[str, Any], variant: int) -> str:
    source = canonical.get("subject_event")
    targets = canonical.get("object_events")
    predicate = _predicate(canonical.get("predicate") or "depends_on")
    if isinstance(source, Mapping) and isinstance(targets, list) and targets:
        target_text = " and ".join(
            f"the fact that {_event_fact(target)}"
            for target in targets
            if isinstance(target, Mapping)
        )
        if target_text:
            return (
                f"The fact that {_event_fact(source)} {predicate} {target_text}."
            )
    object_text = " and ".join(_display(value) for value in _objects(canonical))
    return f"{_display(canonical['subject'])} {predicate} {object_text}."


def _render_relation(
    claim_type: str, canonical: Mapping[str, Any], variant: int
) -> str:
    if claim_type == "node":
        return _render_node(canonical, variant)
    if claim_type == "membership":
        return _render_membership(canonical, variant)
    if claim_type == "visibility":
        return _render_visibility(canonical, variant)
    if claim_type == "status_transition":
        if len(_objects(canonical)) != 2:
            raise TruthSupportError("status_transition requires exactly two ordered values")
        return _render_transition(canonical, variant)
    if claim_type == "comparison":
        return _render_comparison(canonical, variant)
    if claim_type == "cause":
        return _render_cause(canonical, variant)
    if claim_type == "dependency":
        return _render_dependency(canonical, variant)
    raise TruthSupportError(f"unsupported truth support claim_type: {claim_type!r}")


def _path_rows(value: Any, path: str) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key in sorted(value, key=str):
            if str(key) == "event_id":
                continue
            rows.extend(_path_rows(value[key], f"{path}.{key}"))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_path_rows(item, f"{path}[{index}]"))
        return rows
    return [{"path": path, "value": copy.deepcopy(value)}]


def required_semantic_paths(claim: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return semantic leaves that a typed quote must preserve.

    Internal event identifiers are excluded from visible support.  Endpoint
    descriptors still require their entity, slot, value, status, and time
    roles, so cause/dependency direction cannot collapse to entity co-occurrence.
    """

    canonical = _canonical(claim)
    claim_type = _claim_type(claim)
    fields: list[str] = list(_CORE_FIELDS)
    extras = {
        "membership": ("included", "entity", "slot"),
        "visibility": ("owner", "access_class"),
        "status_transition": ("slot", "from_status", "to_status"),
        "comparison": ("slot", "left_value", "right_value", "absolute_difference"),
        "cause": ("cause_event", "effect_event"),
        "dependency": ("subject_event", "object_events"),
    }.get(claim_type, ())
    fields.extend(extras)
    rows: list[dict[str, Any]] = []
    for field in fields:
        if field in canonical:
            rows.extend(_path_rows(canonical[field], f"$.canonical_support_claim.{field}"))
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduplicated[row["path"]] = row
    return [deduplicated[path] for path in sorted(deduplicated)]


def _fold(value: Any) -> str:
    return _SPACE_RE.sub(" ", _display(value).casefold()).strip()


def _position(text: str, value: Any, start: int = 0) -> int:
    needle = _fold(value)
    return text.find(needle, start) if needle else -1


def _after_label(text: str, label: str, value: Any) -> bool:
    folded_label = _fold(label)
    folded_value = _fold(value)
    if not folded_label or not folded_value:
        return False
    cursor = 0
    while True:
        label_pos = text.find(folded_label, cursor)
        if label_pos < 0:
            return False
        value_pos = label_pos + len(folded_label)
        while value_pos < len(text) and text[value_pos] in " :":
            value_pos += 1
        if text.startswith(folded_value, value_pos):
            end = value_pos + len(folded_value)
            if end == len(text) or text[end] in ",.;" or text.startswith(" to ", end):
                return True
        cursor = label_pos + len(folded_label)


def _after_sentence_label(text: str, label: str, value: Any) -> bool:
    folded_label = _fold(label)
    folded_value = _fold(value)
    pattern = re.compile(
        rf"(?:^|[.!?]\s+){re.escape(folded_label)}\s*:\s*"
        rf"{re.escape(folded_value)}(?=$|[,.;])"
    )
    return pattern.search(text) is not None


def _issue(code: str, path: str, message: str, **details: Any) -> dict[str, Any]:
    row = {"code": code, "path": path, "message": message}
    if details:
        row["details"] = details
    return row


def _descriptor_checks(
    text: str,
    descriptor: Mapping[str, Any],
    path: str,
    prefix: str,
    issues: list[dict[str, Any]],
    covered: set[str],
) -> None:
    anchors = [
        text.find(_fold(f"{prefix} {field.replace('_', ' ')}"))
        for field in _DESCRIPTOR_FIELDS
    ]
    anchors = [position for position in anchors if position >= 0]
    segment_start = min(anchors) if anchors else -1
    segment_end = text.find(".", segment_start) if segment_start >= 0 else -1
    segment = (
        text[segment_start : segment_end + 1]
        if segment_start >= 0 and segment_end >= 0
        else text[segment_start:] if segment_start >= 0 else ""
    )
    for field in _DESCRIPTOR_FIELDS:
        value = descriptor.get(field)
        if value is None or value == "":
            continue
        semantic_path = f"{path}.{field}"
        label = f"{prefix} {field.replace('_', ' ')}"
        if field == "status" and _token(value) == _ACTIVE_STATUS:
            conflicting_status = re.search(
                rf"\b{re.escape(_fold(prefix))}\s+status\s*:\s*([^,.;]+)",
                segment,
            )
            if conflicting_status is None:
                covered.add(semantic_path)
            else:
                issues.append(
                    _issue(
                        "typed_status_contradiction",
                        semantic_path,
                        "Affirmative endpoint has an explicit conflicting status.",
                        actual=conflicting_status.group(1).strip(),
                    )
                )
        elif _after_label(segment, label, value):
            covered.add(semantic_path)
        else:
            issues.append(
                _issue(
                    "typed_role_missing",
                    semantic_path,
                    "Endpoint field is absent from its typed role.",
                    role=prefix,
                    expected=value,
                )
            )


def _generic_checks(
    claim: Mapping[str, Any],
    canonical: Mapping[str, Any],
    text: str,
    issues: list[dict[str, Any]],
    covered: set[str],
) -> None:
    subject_path = "$.canonical_support_claim.subject"
    predicate_path = "$.canonical_support_claim.predicate"
    subject_pos = _position(text, canonical.get("subject"))
    if subject_pos >= 0:
        covered.add(subject_path)
    else:
        issues.append(_issue("subject_missing", subject_path, "Claim subject is absent."))

    claim_type = _claim_type(claim)
    predicate_markers = {
        "status_transition": "transitions from",
        "comparison": "absolute difference",
        "membership": "member of",
    }
    marker = predicate_markers.get(claim_type, _predicate(canonical.get("predicate")))
    predicate_pos = text.find(_fold(marker))
    if predicate_pos >= 0:
        covered.add(predicate_path)
    else:
        issues.append(
            _issue("predicate_missing", predicate_path, "Typed predicate marker is absent.", expected=marker)
        )

    # A comparison states its right entity before the derived predicate.  Its
    # object roles are therefore admitted by the dedicated left/right check.
    if claim_type != "comparison":
        cursor = max(0, predicate_pos)
        for index, value in enumerate(_objects(canonical)):
            path = f"$.canonical_support_claim.objects[{index}]"
            position = _position(text, value, cursor)
            if position >= 0:
                covered.add(path)
                cursor = position + len(_fold(value))
            else:
                issues.append(
                    _issue(
                        "object_role_or_order_mismatch",
                        path,
                        "Object is missing or appears outside its ordered role.",
                        expected=value,
                    )
                )
    if subject_pos >= 0 and predicate_pos >= 0 and subject_pos > predicate_pos:
        issues.append(
            _issue(
                "subject_predicate_direction_mismatch",
                subject_path,
                "Subject must precede the directed predicate.",
            )
        )

    polarity_path = "$.canonical_support_claim.polarity"
    negative = _token(canonical.get("polarity")) == "negative"
    relation_end = text.find(".")
    relation_text = text if relation_end < 0 else text[: relation_end + 1]
    has_negation = bool(_NEGATION_RE.search(relation_text))
    if negative == has_negation:
        covered.add(polarity_path)
    else:
        issues.append(
            _issue(
                "polarity_mismatch",
                polarity_path,
                "Relation-clause negation does not match frozen polarity.",
                expected=canonical.get("polarity"),
            )
        )

    status = canonical.get("status")
    if status is not None and status != "":
        path = "$.canonical_support_claim.status"
        if _token(status) == _ACTIVE_STATUS:
            conflicting_status = re.search(r"\bclaim status\s*:\s*([^.;]+)", text)
            relation_end = text.find(".")
            relation_text = text if relation_end < 0 else text[: relation_end + 1]
            if conflicting_status is None and not _INVALIDATION_RE.search(relation_text):
                covered.add(path)
            else:
                issues.append(
                    _issue(
                        "status_contradiction",
                        path,
                        "An active claim must be an affirmative assertion without invalidation.",
                    )
                )
        elif _after_sentence_label(text, "claim status", status):
            covered.add(path)
        else:
            issues.append(_issue("status_missing", path, "Frozen claim status is absent."))

    time = canonical.get("time")
    if isinstance(time, Mapping):
        for field, label in _TIME_LABELS.items():
            if time.get(field) is None:
                continue
            path = f"$.canonical_support_claim.time.{field}"
            if _after_label(text, label, time[field]):
                covered.add(path)
            else:
                issues.append(
                    _issue("time_role_missing", path, "Frozen time coordinate is absent.", role=label)
                )

    condition = canonical.get("condition")
    if condition is not None and condition != "":
        path = "$.canonical_support_claim.condition"
        if _after_sentence_label(text, "condition", condition):
            covered.add(path)
        else:
            issues.append(_issue("condition_missing", path, "Frozen condition is absent."))

    for scope_path, value in _scope_leaves(canonical.get("scope")):
        path = "$.canonical_support_claim.scope." + ".".join(scope_path)
        label = "scope " + " ".join(scope_path)
        if _after_label(text, label, value):
            covered.add(path)
        else:
            issues.append(_issue("scope_role_missing", path, "Frozen scope field is absent."))


def _typed_checks(
    claim: Mapping[str, Any],
    canonical: Mapping[str, Any],
    text: str,
    issues: list[dict[str, Any]],
    covered: set[str],
) -> None:
    claim_type = _claim_type(claim)
    if claim_type == "membership":
        included = canonical.get("included")
        negative = _token(canonical.get("polarity")) == "negative"
        expected_negative = (not included) if isinstance(included, bool) else negative
        marker = "is not a member of" if expected_negative else "is a member of"
        if _fold(marker) not in text:
            issues.append(
                _issue(
                    "membership_polarity_mismatch",
                    "$.canonical_support_claim.polarity",
                    "Membership inclusion and relation phrase disagree.",
                )
            )
        for field in ("entity", "slot", "included"):
            if field not in canonical:
                continue
            path = f"$.canonical_support_claim.{field}"
            if _after_label(text, f"membership {field}", canonical[field]):
                covered.add(path)
            else:
                issues.append(_issue("membership_role_missing", path, "Membership role is absent."))

    elif claim_type == "visibility":
        observer = canonical.get("subject")
        action = canonical.get("predicate")
        resource = _objects(canonical)[0]
        observer_pos = _position(text, observer)
        action_pos = _position(text, _predicate(action), max(0, observer_pos))
        resource_pos = _position(text, resource, max(0, action_pos))
        if not (0 <= observer_pos < action_pos < resource_pos):
            issues.append(
                _issue(
                    "visibility_direction_mismatch",
                    "$.canonical_support_claim",
                    "Observer, action, and resource roles are not directed in order.",
                )
            )
        for field, label in (("owner", "resource owner"), ("access_class", "access class")):
            if field not in canonical:
                continue
            path = f"$.canonical_support_claim.{field}"
            if _after_label(text, label, canonical[field]):
                covered.add(path)
            else:
                issues.append(_issue("visibility_role_missing", path, "Visibility role is absent."))

    elif claim_type == "status_transition":
        objects = _objects(canonical)
        from_pos = text.find(" from ")
        old_pos = _position(text, objects[0], max(0, from_pos))
        to_pos = text.find(" to ", max(0, old_pos))
        new_pos = _position(text, objects[1], max(0, to_pos))
        if not (0 <= from_pos < old_pos < to_pos < new_pos):
            issues.append(
                _issue(
                    "transition_direction_mismatch",
                    "$.canonical_support_claim.objects",
                    "Transition must preserve ordered from-value and to-value roles.",
                )
            )
        for field, label in (
            ("slot", "transitions from"),
            ("from_status", "from status"),
            ("to_status", "to status"),
        ):
            if field not in canonical:
                continue
            path = f"$.canonical_support_claim.{field}"
            if field == "slot":
                ok = _position(text, canonical[field]) >= 0 and _position(text, canonical[field]) < from_pos
            elif _token(canonical[field]) == _ACTIVE_STATUS:
                conflicting_status = re.search(rf"\b{re.escape(label)}\s+([^.;]+)", text)
                ok = conflicting_status is None
            else:
                ok = _after_label(text, label, canonical[field])
            if ok:
                covered.add(path)
            else:
                issues.append(_issue("transition_role_missing", path, "Transition role is absent."))

    elif claim_type == "comparison":
        right_entity = _objects(canonical)[0]
        left_entity_pos = _position(text, canonical.get("subject"))
        left_value_pos = _position(text, canonical.get("left_value"), max(0, left_entity_pos))
        while_pos = text.find(" while ", max(0, left_value_pos))
        right_entity_pos = _position(text, right_entity, max(0, while_pos))
        right_value_pos = _position(text, canonical.get("right_value"), max(0, right_entity_pos))
        if not (
            0 <= left_entity_pos < left_value_pos < while_pos < right_entity_pos < right_value_pos
        ):
            issues.append(
                _issue(
                    "comparison_role_mismatch",
                    "$.canonical_support_claim",
                    "Comparison left/right entity and value roles are not ordered.",
                )
            )
        else:
            covered.add("$.canonical_support_claim.objects[0]")
        difference_path = "$.canonical_support_claim.objects[1]"
        difference = _objects(canonical)[1] if len(_objects(canonical)) > 1 else None
        if difference is not None and _after_label(text, "absolute difference is", difference):
            covered.add(difference_path)
        elif difference is not None:
            issues.append(
                _issue(
                    "comparison_role_missing",
                    difference_path,
                    "Derived comparison object is absent from its role.",
                )
            )
        for field, label in (
            ("slot", "comparison slot"),
            ("left_value", "left value"),
            ("right_value", "right value"),
            ("absolute_difference", "absolute difference is"),
        ):
            if field not in canonical:
                continue
            path = f"$.canonical_support_claim.{field}"
            if _after_label(text, label, canonical[field]):
                covered.add(path)
            else:
                issues.append(_issue("comparison_role_missing", path, "Comparison role is absent."))

    elif claim_type == "cause":
        cause = canonical.get("cause_event")
        effect = canonical.get("effect_event")
        if not isinstance(cause, Mapping) or not isinstance(effect, Mapping):
            issues.append(
                _issue("cause_descriptors_missing", "$.canonical_support_claim", "Cause descriptors are required.")
            )
            return
        _descriptor_checks(
            text,
            cause,
            "$.canonical_support_claim.cause_event",
            "cause",
            issues,
            covered,
        )
        _descriptor_checks(
            text,
            effect,
            "$.canonical_support_claim.effect_event",
            "effect",
            issues,
            covered,
        )
        cause_pos = _position(text, cause.get("entity"))
        predicate_pos = text.find(
            f" {_fold(_predicate(canonical.get('predicate') or 'causes'))} ",
            max(0, cause_pos),
        )
        effect_pos = _position(text, effect.get("entity"), max(0, predicate_pos))
        if not (0 <= cause_pos < predicate_pos < effect_pos):
            issues.append(
                _issue(
                    "cause_direction_mismatch",
                    "$.canonical_support_claim",
                    "Cause endpoint must precede the directed predicate and effect endpoint.",
                )
            )

    elif claim_type == "dependency":
        source = canonical.get("subject_event")
        targets = canonical.get("object_events")
        if not isinstance(source, Mapping) or not isinstance(targets, list) or not targets:
            return
        _descriptor_checks(
            text,
            source,
            "$.canonical_support_claim.subject_event",
            "dependency source",
            issues,
            covered,
        )
        source_pos = _position(text, source.get("entity"))
        predicate_pos = _position(text, _predicate(canonical.get("predicate")), max(0, source_pos))
        cursor = max(0, predicate_pos)
        for index, target in enumerate(targets):
            if not isinstance(target, Mapping):
                issues.append(
                    _issue(
                        "dependency_target_invalid",
                        f"$.canonical_support_claim.object_events[{index}]",
                        "Dependency target descriptor is invalid.",
                    )
                )
                continue
            prefix = f"dependency target {index + 1}"
            _descriptor_checks(
                text,
                target,
                f"$.canonical_support_claim.object_events[{index}]",
                prefix,
                issues,
                covered,
            )
            target_pos = _position(text, target.get("entity"), cursor)
            if target_pos < 0:
                issues.append(
                    _issue(
                        "dependency_direction_mismatch",
                        f"$.canonical_support_claim.object_events[{index}]",
                        "Dependency target is absent or precedes its source/predicate.",
                    )
                )
            else:
                cursor = target_pos + len(_fold(target.get("entity")))


def validate_typed_frozen_support(
    claim: Mapping[str, Any], support: Mapping[str, Any] | str
) -> dict[str, Any]:
    """Validate typed semantic coverage without using the general quote validator."""

    issues: list[dict[str, Any]] = []
    if not isinstance(claim, Mapping):
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "pass": False,
            "issue_count": 1,
            "issues": [_issue("claim_not_object", "$.claim", "Claim must be an object.")],
        }
    claim_id = _text(claim.get("claim_id"))
    if isinstance(support, Mapping):
        quote = support.get("quote")
        supplied_claim_id = _text(support.get("claim_id"))
        if supplied_claim_id != claim_id:
            issues.append(
                _issue(
                    "claim_id_mismatch",
                    "$.support.claim_id",
                    "Support is bound to a different claim id.",
                    expected=claim_id,
                    actual=supplied_claim_id,
                )
            )
        if "event_binding" in support:
            expected_binding = claim_event_binding(claim)
            if support.get("event_binding") != expected_binding:
                issues.append(
                    _issue(
                        "event_binding_mismatch",
                        "$.support.event_binding",
                        "Support graph endpoints differ from the frozen claim.",
                        expected=expected_binding,
                        actual=support.get("event_binding"),
                    )
                )
    else:
        quote = support
    if not isinstance(quote, str) or not _text(quote):
        issues.append(_issue("quote_missing", "$.support.quote", "Support quote is required."))
        quote = ""

    try:
        canonical = _canonical(claim)
        required = required_semantic_paths(claim)
    except TruthSupportError as error:
        issues.append(_issue("invalid_canonical_claim", "$.claim", str(error)))
        canonical = {}
        required = []

    covered: set[str] = set()
    folded_quote = _fold(quote)
    if canonical and folded_quote:
        _generic_checks(claim, canonical, folded_quote, issues, covered)
        _typed_checks(claim, canonical, folded_quote, issues, covered)

    required_paths = [row["path"] for row in required]
    missing_paths = sorted(set(required_paths) - covered)
    for path in missing_paths:
        if not any(issue.get("path") == path for issue in issues):
            issues.append(
                _issue(
                    "semantic_path_uncovered",
                    path,
                    "Frozen semantic path is not independently covered by the quote.",
                )
            )
    counts: dict[str, int] = {}
    for issue in issues:
        code = str(issue.get("code") or "unknown")
        counts[code] = counts.get(code, 0) + 1
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "pass": not issues,
        "claim_id": claim_id,
        "claim_type": _claim_type(claim),
        "required_semantic_paths": required_paths,
        "covered_semantic_paths": sorted(covered),
        "missing_semantic_paths": missing_paths,
        "issue_count": len(issues),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": issues,
    }


def required_semantic_paths(claim: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Delegate manifest compilation to the independent fail-closed validator."""

    try:
        return _validator.required_semantic_paths(claim)
    except _validator.TruthSupportValidationError as error:
        raise TruthSupportError(str(error)) from error


def validate_typed_frozen_support(
    claim: Mapping[str, Any], support: Mapping[str, Any] | str
) -> dict[str, Any]:
    """Delegate admission to the independent validator implementation."""

    return _validator.validate_typed_frozen_support(claim, support)


def compile_typed_frozen_support(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Render and independently admit one typed frozen-support record."""

    if not isinstance(claim, Mapping):
        raise TruthSupportError("claim must be an object")
    claim_id = _text(claim.get("claim_id"))
    if not claim_id:
        raise TruthSupportError("claim_id must be non-empty")
    canonical = _canonical(claim)
    claim_type = _claim_type(claim)
    _require_renderable_claim(claim_type, canonical)
    variant = int(hashlib.sha256(claim_id.encode("utf-8")).hexdigest()[:8], 16) % 3
    quote_parts = [
        _render_relation(claim_type, canonical, variant),
        *_qualifier_sentences(canonical, claim_type),
    ]
    quote = " ".join(part for part in quote_parts if part)
    semantic_paths = required_semantic_paths(claim)
    record = {
        "schema_version": SCHEMA_VERSION,
        "claim_id": claim_id,
        "claim_type": claim_type,
        "operation": _text(claim.get("operation")) or None,
        "edge_role": _text(claim.get("edge_role")) or None,
        "qa_eligible": claim.get("qa_eligible") is True,
        "event_binding": claim_event_binding(claim),
        "quote": quote,
        "semantic_path_manifest": semantic_paths,
    }
    record["support_sha256"] = canonical_sha256(record)
    validation = validate_typed_frozen_support(claim, record)
    if validation["pass"] is not True:
        raise TruthSupportError(
            f"typed frozen-support validation failed for {claim_id}: {validation['issues'][:6]}"
        )
    return record


def compile_frozen_support_quote(claim: Mapping[str, Any]) -> str:
    """Compatibility interface for callers that store only the visible quote."""

    return compile_typed_frozen_support(claim)["quote"]


__all__ = [
    "SCHEMA_VERSION",
    "VALIDATION_SCHEMA_VERSION",
    "TruthSupportError",
    "canonical_sha256",
    "claim_event_binding",
    "compile_frozen_support_quote",
    "compile_typed_frozen_support",
    "required_semantic_paths",
    "validate_typed_frozen_support",
]
