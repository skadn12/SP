"""Independent validator for Truth-first typed frozen support.

The renderer is intentionally *not* imported here.  This module owns a
separate, fail-closed claim schema and validates both the hidden support
record and its user-visible natural-language quote.  Internal graph time is
bound by the signed semantic manifest; it never has to be printed in the
dialogue.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


VALIDATION_SCHEMA_VERSION = "stage3_typed_frozen_support_validation_v2"
EXPECTED_SUPPORT_SCHEMA_VERSION = "stage3_typed_frozen_support_v1"

_OPERATIONS = {
    "temporal_state_update",
    "bridge",
    "comparison",
    "visibility",
    "count",
    "cause_consequence",
    "relation_lookup",
}
_CLAIM_TYPES = {
    "node",
    "membership",
    "visibility",
    "status_transition",
    "comparison",
    "cause",
    "dependency",
}
_EDGE_ROLES = {"control", "semantic"}
_POLARITIES = {"positive", "negative"}
_ACTIVE_STATUSES = {"active"}
_STATUS_CUES: dict[str, tuple[str, ...]] = {
    "restored": ("restored", "reinstated", "put back"),
    "confirmed": ("confirmed", "verified", "established"),
    "completed": ("completed", "finished", "done"),
    "belief": ("believed", "belief", "thought", "suspected", "unsure", "uncertain"),
    "report": ("report", "reported", "reports said", "according to", "was said"),
    "reported": ("report", "reported", "reports said", "according to", "was said"),
    "proposed": ("proposed", "proposal", "suggested"),
    "tentative": ("tentative", "provisional", "not final"),
    "unconfirmed": ("unconfirmed", "not confirmed", "unverified"),
    "outdated": ("outdated", "superseded", "used to", "formerly", "previously"),
    "canceled": ("canceled", "cancelled", "called off", "withdrawn", "scrapped"),
    "cancelled": ("canceled", "cancelled", "called off", "withdrawn", "scrapped"),
    "expired": ("expired", "lapsed", "no longer valid"),
    "inactive": ("inactive", "deactivated", "no longer in effect"),
    "revoked": ("revoked", "withdrawn", "rescinded"),
    "superseded": ("superseded", "replaced", "outdated"),
    "planned": ("planned", "intended", "scheduled"),
    "pending": ("pending", "awaiting", "not yet decided"),
    "delayed_not_active": ("delayed", "postponed", "not yet active"),
    # A condition plus typed observer/action is the semantic realization of
    # this state; a literal ``scope_limited`` token is forbidden in dialogue.
    "scope_limited": (),
}
_ALLOWED_STATUSES = _ACTIVE_STATUSES | set(_STATUS_CUES)

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
_TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    "node": (),
    "membership": ("included", "entity", "slot"),
    "visibility": ("owner", "access_class"),
    "status_transition": ("slot", "from_status", "to_status"),
    "comparison": ("slot", "left_value", "right_value", "absolute_difference"),
    "cause": ("cause_event", "effect_event"),
    "dependency": ("subject_event", "object_events"),
}
_DESCRIPTOR_FIELDS = (
    "entity",
    "slot",
    "value",
    "status",
    "time_index",
    "global_time_index",
)
_RECORD_FIELDS = {
    "schema_version",
    "claim_id",
    "claim_type",
    "operation",
    "edge_role",
    "qa_eligible",
    "event_binding",
    "quote",
    "semantic_path_manifest",
    "support_sha256",
}

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_RELATION_NEGATION_RE = re.compile(
    r"\b(?:is|are|was|were|may|can|could|does|did|has|have|had|will|would)\s+not\b|"
    r"\bnever\s+(?:is|was|became|joined|viewed|edited|depended|caused|changed|moved)\b|"
    r"\bnot\s+(?:a\s+)?(?:member|part|included|dependent)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(?:reportedly|allegedly|apparently|perhaps|probably|possibly|maybe|"
    r"might|rumou?red|seems?|appears?|believed|suspected|unsure|uncertain|"
    r"reported|unconfirmed|according\s+to)\b",
    re.IGNORECASE,
)
_TRAILING_CONTRADICTION_RE = re.compile(
    r"\b(?:that|this|the preceding|the previous)\s+(?:statement|claim|fact)?\s*"
    r"(?:is|was)\s+(?:not true|false|wrong)\b|"
    r"\b(?:ignore|disregard|retract)\b.{0,48}\b(?:preceding|previous|statement|claim|fact)\b|"
    r"\bthe opposite is true\b|\bcontrary to (?:that|this|the preceding)\b",
    re.IGNORECASE,
)
_INTERNAL_LABEL_RES = (
    re.compile(r"(?:^|[.!?]\s+)\s*timing\s*:", re.IGNORECASE),
    re.compile(r"(?:^|[.!?]\s+)\s*claim\s+status\s*:", re.IGNORECASE),
    re.compile(r"(?:^|[.!?]\s+)\s*condition\s*:", re.IGNORECASE),
    re.compile(r"\bscope\s+(?:details|observer|permission|condition)\s*:", re.IGNORECASE),
    re.compile(r"\bmembership\s+(?:entity|slot|included)\s*:", re.IGNORECASE),
    re.compile(r"\bresource\s+owner\s*:|\baccess\s+class\s*:", re.IGNORECASE),
    re.compile(r"\bcomparison\s+slot\s*:|\bleft\s+value\s*:|\bright\s+value\s*:", re.IGNORECASE),
    re.compile(r"\b(?:cause|effect)\s+endpoint\b", re.IGNORECASE),
    re.compile(r"\bdependency\s+(?:source|target)\b", re.IGNORECASE),
    re.compile(r"\b(?:local|global)\s+(?:step|time|index)\b", re.IGNORECASE),
    re.compile(r"\b(?:time_index|global_time_index|event_id|claim_id|qa_eligible|edge_role)\b", re.IGNORECASE),
    re.compile(r"\bcanonical_support_claim\b|\bsemantic_path\b", re.IGNORECASE),
)
_EXPLICIT_INTERNAL_TIME_RE = re.compile(
    r"\b(?:time|step|sequence\s+position)\s*(?:number\s*)?"
    r"(?:is|was|=|:|#)?\s*(-?\d+)\b",
    re.IGNORECASE,
)


class TruthSupportValidationError(ValueError):
    """Raised when semantic paths are requested for an invalid typed claim."""


def _text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _text(value).casefold()).strip("_")


def _natural(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return _text(value).replace("_", " ").replace("-", " ")


def _fold(value: Any) -> str:
    return _SPACE_RE.sub(" ", _WORD_RE.sub(" ", _natural(value).casefold())).strip()


def _phrase_position(text: str, value: Any, start: int = 0) -> int:
    phrase = _fold(value)
    if not phrase:
        return -1
    match = re.compile(
        rf"(?<!\w){re.escape(phrase)}(?:\s+s)?(?=\s|$)",
        re.IGNORECASE,
    ).search(text, max(0, start))
    return match.start() if match is not None else -1


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return _fold(left) == _fold(right)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _issue(code: str, path: str, message: str, **details: Any) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "path": path, "message": message}
    if details:
        issue["details"] = details
    return issue


def _claim_type(claim: Mapping[str, Any]) -> str:
    if _token(claim.get("claim_kind")) == "node":
        return "node"
    return _token(claim.get("claim_type"))


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TruthSupportValidationError(f"record is not finite JSON: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _event_binding(claim: Mapping[str, Any]) -> dict[str, Any]:
    subject = _text(claim.get("subject_event_id"))
    objects = claim.get("object_event_ids")
    sources = claim.get("source_event_ids")
    object_ids = [_text(value) for value in objects] if isinstance(objects, list) else []
    source_ids = [_text(value) for value in sources] if isinstance(sources, list) else []
    endpoints = list(dict.fromkeys([subject, *object_ids, *source_ids]))
    return {
        "subject_event_id": subject or None,
        "object_event_ids": [value for value in object_ids if value],
        "source_event_ids": [value for value in source_ids if value],
        "endpoint_event_ids": [value for value in endpoints if value],
    }


def _require_key(
    mapping: Mapping[str, Any], key: str, path: str, issues: list[dict[str, Any]]
) -> bool:
    if key in mapping:
        return True
    issues.append(_issue("typed_field_missing", f"{path}.{key}", "Required typed field is absent."))
    return False


def _validate_status(value: Any, path: str, issues: list[dict[str, Any]]) -> None:
    status = _token(value)
    if not status or status not in _ALLOWED_STATUSES:
        issues.append(
            _issue(
                "typed_status_invalid",
                path,
                "Status is missing or outside the closed semantic status vocabulary.",
                actual=value,
            )
        )


def _validate_time(value: Any, path: str, issues: list[dict[str, Any]]) -> None:
    if not isinstance(value, Mapping):
        issues.append(_issue("typed_time_invalid", path, "Time must be an object."))
        return
    for field in ("time_index", "global_time_index"):
        if not _require_key(value, field, path, issues):
            continue
        if not _is_index(value.get(field)):
            issues.append(
                _issue("typed_time_invalid", f"{path}.{field}", "Time coordinate must be a non-negative integer.")
            )


def _validate_descriptor(
    descriptor: Any, path: str, issues: list[dict[str, Any]]
) -> Mapping[str, Any] | None:
    if not isinstance(descriptor, Mapping):
        issues.append(_issue("typed_descriptor_invalid", path, "Endpoint descriptor must be an object."))
        return None
    for field in ("event_id", *_DESCRIPTOR_FIELDS):
        _require_key(descriptor, field, path, issues)
    for field in ("event_id", "entity", "slot", "value"):
        value = descriptor.get(field)
        if not _is_scalar(value) or not _text(value):
            issues.append(
                _issue("typed_descriptor_field_invalid", f"{path}.{field}", "Endpoint field must be a non-empty scalar.")
            )
    _validate_status(descriptor.get("status"), f"{path}.status", issues)
    for field in ("time_index", "global_time_index"):
        if not _is_index(descriptor.get(field)):
            issues.append(
                _issue("typed_time_invalid", f"{path}.{field}", "Endpoint time must be a non-negative integer.")
            )
    return descriptor


def _schema_issues(claim: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None, str]:
    issues: list[dict[str, Any]] = []
    claim_type = _claim_type(claim)
    if claim_type not in _CLAIM_TYPES:
        issues.append(_issue("claim_type_invalid", "$.claim.claim_type", "Unsupported typed claim type.", actual=claim_type))

    for field in (
        "claim_id",
        "claim_kind",
        "claim_type",
        "operation",
        "edge_role",
        "qa_eligible",
        "subject_event_id",
        "object_event_ids",
        "source_event_ids",
        "canonical_support_claim",
    ):
        _require_key(claim, field, "$.claim", issues)
    if not _text(claim.get("claim_id")):
        issues.append(_issue("claim_id_invalid", "$.claim.claim_id", "Claim id must be non-empty."))
    if _token(claim.get("operation")) not in _OPERATIONS:
        issues.append(_issue("operation_invalid", "$.claim.operation", "Unsupported truth operation."))
    if _token(claim.get("edge_role")) not in _EDGE_ROLES:
        issues.append(_issue("edge_role_invalid", "$.claim.edge_role", "Edge role must be control or semantic."))
    if not isinstance(claim.get("qa_eligible"), bool):
        issues.append(_issue("qa_eligible_invalid", "$.claim.qa_eligible", "qa_eligible must be boolean."))
    for field in ("object_event_ids", "source_event_ids"):
        value = claim.get(field)
        if not isinstance(value, list) or any(not _text(item) for item in value):
            issues.append(_issue("event_id_list_invalid", f"$.claim.{field}", "Event ids must be a list of non-empty ids."))
    if not _text(claim.get("subject_event_id")):
        issues.append(_issue("subject_event_id_invalid", "$.claim.subject_event_id", "Subject event id is required."))

    canonical = claim.get("canonical_support_claim")
    if not isinstance(canonical, Mapping):
        issues.append(_issue("canonical_claim_invalid", "$.claim.canonical_support_claim", "Canonical support claim must be an object."))
        return issues, None, claim_type
    for field in _CORE_FIELDS:
        _require_key(canonical, field, "$.claim.canonical_support_claim", issues)
    for field in ("subject", "predicate"):
        value = canonical.get(field)
        if not _is_scalar(value) or not _text(value):
            issues.append(_issue("typed_field_invalid", f"$.claim.canonical_support_claim.{field}", "Field must be a non-empty scalar."))
    objects = canonical.get("objects")
    if not isinstance(objects, list) or not objects or any(not _is_scalar(value) or not _text(value) for value in objects):
        issues.append(_issue("typed_objects_invalid", "$.claim.canonical_support_claim.objects", "Objects must be a non-empty scalar list."))
    if _token(canonical.get("polarity")) not in _POLARITIES:
        issues.append(_issue("typed_polarity_invalid", "$.claim.canonical_support_claim.polarity", "Polarity must be positive or negative."))
    _validate_status(canonical.get("status"), "$.claim.canonical_support_claim.status", issues)
    _validate_time(canonical.get("time"), "$.claim.canonical_support_claim.time", issues)
    if canonical.get("condition") is not None and (
        not _is_scalar(canonical.get("condition")) or not _text(canonical.get("condition"))
    ):
        issues.append(_issue("typed_condition_invalid", "$.claim.canonical_support_claim.condition", "Condition must be null or a non-empty scalar."))
    scope = canonical.get("scope")
    if scope is not None and not isinstance(scope, Mapping):
        issues.append(_issue("typed_scope_invalid", "$.claim.canonical_support_claim.scope", "Scope must be null or an object."))

    for field in _TYPE_FIELDS.get(claim_type, ()):
        _require_key(canonical, field, "$.claim.canonical_support_claim", issues)

    if claim_type == "membership":
        if not isinstance(canonical.get("included"), bool):
            issues.append(_issue("membership_included_invalid", "$.claim.canonical_support_claim.included", "Membership included must be boolean."))
        for field in ("entity", "slot"):
            if not _is_scalar(canonical.get(field)) or not _text(canonical.get(field)):
                issues.append(_issue("membership_role_invalid", f"$.claim.canonical_support_claim.{field}", "Membership role must be non-empty."))
        included = canonical.get("included")
        polarity = _token(canonical.get("polarity"))
        predicate = _token(canonical.get("predicate"))
        if isinstance(included, bool):
            expected_polarity = "positive" if included else "negative"
            expected_predicate = "member_of" if included else "not_member_of"
            if polarity != expected_polarity or predicate != expected_predicate:
                issues.append(_issue("membership_truth_inconsistent", "$.claim.canonical_support_claim", "Membership predicate, inclusion, and polarity disagree."))
        if isinstance(objects, list) and len(objects) != 1:
            issues.append(_issue("membership_object_arity", "$.claim.canonical_support_claim.objects", "Membership requires one set object."))

    elif claim_type == "visibility":
        for field in ("owner", "access_class"):
            if not _is_scalar(canonical.get(field)) or not _text(canonical.get(field)):
                issues.append(_issue("visibility_role_invalid", f"$.claim.canonical_support_claim.{field}", "Visibility role must be non-empty."))
        if not isinstance(objects, list) or len(objects) != 1:
            issues.append(_issue("visibility_object_arity", "$.claim.canonical_support_claim.objects", "Visibility requires one resource object."))
        if not isinstance(scope, Mapping):
            issues.append(_issue("visibility_scope_missing", "$.claim.canonical_support_claim.scope", "Visibility requires typed scope."))
        else:
            for field in ("observer", "permission", "condition"):
                if not _require_key(scope, field, "$.claim.canonical_support_claim.scope", issues):
                    continue
                if not _is_scalar(scope.get(field)) or not _text(scope.get(field)):
                    issues.append(_issue("visibility_scope_role_invalid", f"$.claim.canonical_support_claim.scope.{field}", "Visibility scope role must be non-empty."))
            if not _same(scope.get("observer"), canonical.get("subject")):
                issues.append(_issue("visibility_observer_mismatch", "$.claim.canonical_support_claim.scope.observer", "Scope observer must equal the visibility subject."))
            if not _same(scope.get("permission"), canonical.get("predicate")):
                issues.append(_issue("visibility_permission_mismatch", "$.claim.canonical_support_claim.scope.permission", "Scope permission must equal the visibility action."))
            if not _same(scope.get("condition"), canonical.get("condition")):
                issues.append(_issue("visibility_condition_mismatch", "$.claim.canonical_support_claim.scope.condition", "Scope condition must equal the claim condition."))
        if canonical.get("condition") in (None, ""):
            issues.append(_issue("visibility_condition_missing", "$.claim.canonical_support_claim.condition", "Visibility requires an explicit condition."))

    elif claim_type == "status_transition":
        if _token(canonical.get("predicate")) != "transitions_to":
            issues.append(_issue("transition_predicate_invalid", "$.claim.canonical_support_claim.predicate", "Transition predicate must be transitions_to."))
        for field in ("slot", "from_status", "to_status"):
            if not _is_scalar(canonical.get(field)) or not _text(canonical.get(field)):
                issues.append(_issue("transition_role_invalid", f"$.claim.canonical_support_claim.{field}", "Transition role must be non-empty."))
        _validate_status(canonical.get("from_status"), "$.claim.canonical_support_claim.from_status", issues)
        _validate_status(canonical.get("to_status"), "$.claim.canonical_support_claim.to_status", issues)
        if not isinstance(objects, list) or len(objects) != 2 or (len(objects) == 2 and _same(objects[0], objects[1])):
            issues.append(_issue("transition_objects_invalid", "$.claim.canonical_support_claim.objects", "Transition requires two distinct ordered values."))
        if not _same(canonical.get("status"), canonical.get("to_status")):
            issues.append(_issue("transition_status_mismatch", "$.claim.canonical_support_claim.to_status", "Top-level status must equal to_status."))
        time = canonical.get("time")
        if isinstance(time, Mapping):
            if not _require_key(time, "from_time_index", "$.claim.canonical_support_claim.time", issues):
                pass
            elif not _is_index(time.get("from_time_index")):
                issues.append(_issue("typed_time_invalid", "$.claim.canonical_support_claim.time.from_time_index", "from_time_index must be a non-negative integer."))
            elif _is_index(time.get("time_index")) and time["from_time_index"] >= time["time_index"]:
                issues.append(_issue("transition_time_order_invalid", "$.claim.canonical_support_claim.time", "from_time_index must precede time_index."))

    elif claim_type == "comparison":
        if _token(canonical.get("predicate")) != "absolute_difference":
            issues.append(_issue("comparison_predicate_invalid", "$.claim.canonical_support_claim.predicate", "Comparison predicate must be absolute_difference."))
        for field in ("slot", "left_value", "right_value", "absolute_difference"):
            if field == "slot":
                valid = _is_scalar(canonical.get(field)) and bool(_text(canonical.get(field)))
            else:
                valid = isinstance(canonical.get(field), (int, float)) and not isinstance(canonical.get(field), bool) and math.isfinite(float(canonical.get(field)))
            if not valid:
                issues.append(_issue("comparison_role_invalid", f"$.claim.canonical_support_claim.{field}", "Comparison role has the wrong type."))
        if not isinstance(objects, list) or len(objects) != 2:
            issues.append(_issue("comparison_object_arity", "$.claim.canonical_support_claim.objects", "Comparison requires right entity and derived difference objects."))
        elif not _same(objects[1], canonical.get("absolute_difference")):
            issues.append(_issue("comparison_object_mismatch", "$.claim.canonical_support_claim.objects[1]", "Derived object must equal absolute_difference."))
        left, right, difference = canonical.get("left_value"), canonical.get("right_value"), canonical.get("absolute_difference")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (left, right, difference)):
            if not math.isclose(abs(float(left) - float(right)), float(difference), rel_tol=0.0, abs_tol=1e-12):
                issues.append(_issue("comparison_arithmetic_invalid", "$.claim.canonical_support_claim.absolute_difference", "Absolute difference is arithmetically false."))

    elif claim_type == "cause":
        cause = _validate_descriptor(canonical.get("cause_event"), "$.claim.canonical_support_claim.cause_event", issues)
        effect = _validate_descriptor(canonical.get("effect_event"), "$.claim.canonical_support_claim.effect_event", issues)
        if _token(canonical.get("predicate")) not in {"causes", "caused_by", "led_to"}:
            issues.append(_issue("cause_predicate_invalid", "$.claim.canonical_support_claim.predicate", "Cause claim requires a directed causal predicate."))
        if cause is not None and effect is not None:
            if not _same(canonical.get("subject"), cause.get("entity")):
                issues.append(_issue("cause_subject_mismatch", "$.claim.canonical_support_claim.subject", "Cause subject must equal cause endpoint entity."))
            if not isinstance(objects, list) or len(objects) != 1 or not _same(objects[0], effect.get("entity")):
                issues.append(_issue("cause_object_mismatch", "$.claim.canonical_support_claim.objects", "Cause object must equal effect endpoint entity."))
            if _is_index(cause.get("global_time_index")) and _is_index(effect.get("global_time_index")) and cause["global_time_index"] >= effect["global_time_index"]:
                issues.append(_issue("cause_time_order_invalid", "$.claim.canonical_support_claim", "Cause must precede effect globally."))
            time = canonical.get("time")
            if isinstance(time, Mapping):
                for field in ("cause_time_index",):
                    if not _require_key(time, field, "$.claim.canonical_support_claim.time", issues):
                        continue
                    if not _is_index(time.get(field)):
                        issues.append(_issue("typed_time_invalid", f"$.claim.canonical_support_claim.time.{field}", "Cause time must be a non-negative integer."))
                if time.get("cause_time_index") != cause.get("time_index") or time.get("time_index") != effect.get("time_index") or time.get("global_time_index") != effect.get("global_time_index"):
                    issues.append(_issue("cause_time_binding_mismatch", "$.claim.canonical_support_claim.time", "Causal time manifest does not match endpoint descriptors."))

    elif claim_type == "dependency":
        source = _validate_descriptor(canonical.get("subject_event"), "$.claim.canonical_support_claim.subject_event", issues)
        targets = canonical.get("object_events")
        if not isinstance(targets, list) or not targets:
            issues.append(_issue("dependency_targets_missing", "$.claim.canonical_support_claim.object_events", "Dependency requires at least one typed target."))
            target_rows: list[Mapping[str, Any]] = []
        else:
            target_rows = []
            for index, target in enumerate(targets):
                row = _validate_descriptor(target, f"$.claim.canonical_support_claim.object_events[{index}]", issues)
                if row is not None:
                    target_rows.append(row)
        if _token(canonical.get("predicate")) != "depends_on":
            issues.append(_issue("dependency_predicate_invalid", "$.claim.canonical_support_claim.predicate", "Dependency predicate must be depends_on."))
        if source is not None:
            if not _same(canonical.get("subject"), source.get("entity")):
                issues.append(_issue("dependency_subject_mismatch", "$.claim.canonical_support_claim.subject", "Dependency subject must equal source endpoint entity."))
            if _text(source.get("event_id")) != _text(claim.get("subject_event_id")):
                issues.append(_issue("dependency_subject_event_mismatch", "$.claim.canonical_support_claim.subject_event.event_id", "Source descriptor must bind the subject event."))
            time = canonical.get("time")
            if isinstance(time, Mapping) and (time.get("time_index") != source.get("time_index") or time.get("global_time_index") != source.get("global_time_index")):
                issues.append(_issue("dependency_time_binding_mismatch", "$.claim.canonical_support_claim.time", "Dependency time must match source endpoint."))
        if isinstance(objects, list) and target_rows:
            if len(objects) != len(target_rows) or any(not _same(obj, target.get("entity")) for obj, target in zip(objects, target_rows)):
                issues.append(_issue("dependency_objects_mismatch", "$.claim.canonical_support_claim.objects", "Dependency objects must match ordered target entities."))
        expected_target_ids = [_text(row.get("event_id")) for row in target_rows]
        if isinstance(claim.get("object_event_ids"), list) and expected_target_ids != [_text(value) for value in claim["object_event_ids"]]:
            issues.append(_issue("dependency_target_event_mismatch", "$.claim.object_event_ids", "Dependency target descriptors must match object event ids."))
        if source is not None:
            for index, target in enumerate(target_rows):
                if _is_index(source.get("global_time_index")) and _is_index(target.get("global_time_index")) and target["global_time_index"] > source["global_time_index"]:
                    issues.append(_issue("dependency_time_direction_invalid", f"$.claim.canonical_support_claim.object_events[{index}]", "Dependency target cannot occur after its source."))

    return issues, canonical, claim_type


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
    """Return the complete typed manifest, rejecting malformed claims first."""

    if not isinstance(claim, Mapping):
        raise TruthSupportValidationError("claim must be an object")
    issues, canonical, claim_type = _schema_issues(claim)
    if issues or canonical is None:
        summary = "; ".join(f"{row['code']}@{row['path']}" for row in issues[:12])
        raise TruthSupportValidationError(f"typed claim schema failed: {summary}")
    fields = [*_CORE_FIELDS, *_TYPE_FIELDS[claim_type]]
    rows: list[dict[str, Any]] = []
    for field in fields:
        rows.extend(_path_rows(canonical[field], f"$.canonical_support_claim.{field}"))
    by_path = {row["path"]: row for row in rows}
    return [by_path[path] for path in sorted(by_path)]


def _add_present(
    text: str,
    value: Any,
    path: str,
    covered: set[str],
    issues: list[dict[str, Any]],
    *,
    code: str = "semantic_value_missing",
    start: int = 0,
) -> int:
    position = _phrase_position(text, value, start)
    if position >= 0:
        covered.add(path)
    else:
        issues.append(_issue(code, path, "Required natural-language value is absent from its semantic role.", expected=value))
    return position


def _relation_negation_check(
    text: str,
    start: int,
    end: int,
    polarity: Any,
    path: str,
    covered: set[str],
    issues: list[dict[str, Any]],
) -> None:
    segment = text[max(0, start) : max(start, end) + 1]
    has_negation = bool(_RELATION_NEGATION_RE.search(segment))
    expected = _token(polarity) == "negative"
    if has_negation == expected:
        covered.add(path)
    else:
        issues.append(_issue("polarity_mismatch", path, "Natural relation negation disagrees with frozen polarity."))


def _status_check(
    text: str,
    status: Any,
    path: str,
    covered: set[str],
    issues: list[dict[str, Any]],
    *,
    segment: str | None = None,
    has_scope: bool = False,
) -> None:
    status_key = _token(status)
    search_text = segment if segment is not None else text
    if status_key in _ACTIVE_STATUSES:
        if _UNCERTAINTY_RE.search(search_text):
            issues.append(_issue("active_uncertainty_contradiction", path, "An active fact is expressed as uncertain or merely reported."))
        else:
            covered.add(path)
        return
    if status_key == "scope_limited":
        if has_scope:
            covered.add(path)
        else:
            issues.append(_issue("scope_limited_without_scope", path, "scope_limited status requires typed scope in the quote."))
        return
    cues = _STATUS_CUES.get(status_key)
    if cues and any(_fold(cue) in search_text for cue in cues):
        covered.add(path)
    else:
        issues.append(_issue("status_semantics_missing", path, "Natural quote does not express the frozen status.", expected=status))


def _scope_checks(
    canonical: Mapping[str, Any],
    text: str,
    covered: set[str],
    issues: list[dict[str, Any]],
) -> None:
    condition = canonical.get("condition")
    if condition not in (None, ""):
        _add_present(text, condition, "$.canonical_support_claim.condition", covered, issues, code="condition_missing")
    scope = canonical.get("scope")
    if not isinstance(scope, Mapping):
        return
    positions: dict[str, int] = {}
    for field in ("observer", "permission", "condition"):
        if field not in scope:
            continue
        path = f"$.canonical_support_claim.scope.{field}"
        positions[field] = _add_present(text, scope[field], path, covered, issues, code="scope_role_missing")
    if positions.get("observer", -1) >= 0 and positions.get("permission", -1) >= 0 and positions["observer"] > positions["permission"]:
        issues.append(_issue("scope_direction_mismatch", "$.canonical_support_claim.scope", "Observer must precede the permission action in the natural support."))


def _descriptor_text_checks(
    descriptor: Mapping[str, Any],
    base_path: str,
    text: str,
    segment_start: int,
    segment_end: int,
    covered: set[str],
    issues: list[dict[str, Any]],
) -> None:
    segment = text[max(0, segment_start) : segment_end if segment_end > segment_start else len(text)]
    for field in ("entity", "slot", "value"):
        _add_present(segment, descriptor[field], f"{base_path}.{field}", covered, issues, code="typed_role_missing")
    _status_check(
        text,
        descriptor["status"],
        f"{base_path}.status",
        covered,
        issues,
        segment=segment,
    )
    for field in ("time_index", "global_time_index"):
        covered.add(f"{base_path}.{field}")


def _quote_checks(
    claim: Mapping[str, Any],
    canonical: Mapping[str, Any],
    claim_type: str,
    quote: str,
    covered: set[str],
    issues: list[dict[str, Any]],
) -> None:
    folded = _fold(quote)
    if not folded:
        issues.append(_issue("quote_missing", "$.support.quote", "Natural support quote is required."))
        return
    if "_" in quote:
        issues.append(_issue("internal_ir_surface", "$.support.quote", "Visible support contains an underscore-delimited IR token."))
    for pattern in _INTERNAL_LABEL_RES:
        match = pattern.search(quote)
        if match:
            issues.append(_issue("internal_ir_surface", "$.support.quote", "Visible support contains a machine-facing IR label.", surface=match.group(0)))
    binding = _event_binding(claim)
    for internal_id in [_text(claim.get("claim_id")), *binding["endpoint_event_ids"]]:
        if internal_id and re.search(rf"(?<![\w]){re.escape(internal_id)}(?![\w])", quote, re.IGNORECASE):
            issues.append(_issue("internal_id_leak", "$.support.quote", "Visible support leaks an internal claim/event id.", internal_id=internal_id))
    if _TRAILING_CONTRADICTION_RE.search(quote):
        issues.append(_issue("trailing_contradiction", "$.support.quote", "Visible support contains a retraction or trailing contradiction."))

    allowed_times: set[int] = set()
    for row in _path_rows(canonical.get("time"), "$.canonical_support_claim.time"):
        if _is_index(row.get("value")):
            allowed_times.add(row["value"])
    for descriptor_field in ("cause_event", "effect_event", "subject_event"):
        descriptor = canonical.get(descriptor_field)
        if isinstance(descriptor, Mapping):
            allowed_times.update(value for value in (descriptor.get("time_index"), descriptor.get("global_time_index")) if _is_index(value))
    for descriptor in canonical.get("object_events", []) if isinstance(canonical.get("object_events"), list) else []:
        if isinstance(descriptor, Mapping):
            allowed_times.update(value for value in (descriptor.get("time_index"), descriptor.get("global_time_index")) if _is_index(value))
    for match in _EXPLICIT_INTERNAL_TIME_RE.finditer(quote):
        value = int(match.group(1))
        if value not in allowed_times:
            issues.append(_issue("explicit_time_contradiction", "$.support.quote", "Explicit time coordinate contradicts the frozen time manifest.", actual=value, allowed=sorted(allowed_times)))

    subject_path = "$.canonical_support_claim.subject"
    predicate_path = "$.canonical_support_claim.predicate"
    object_paths = [f"$.canonical_support_claim.objects[{index}]" for index, _ in enumerate(canonical["objects"])]
    subject_pos = _add_present(folded, canonical["subject"], subject_path, covered, issues, code="subject_missing")
    if (
        claim_type != "node"
        and
        _token(canonical.get("status")) in _ACTIVE_STATUSES
        and subject_pos > 0
        and _UNCERTAINTY_RE.search(folded[:subject_pos])
    ):
        issues.append(
            _issue(
                "active_uncertainty_contradiction",
                "$.canonical_support_claim.status",
                "An active fact is prefaced as uncertain or merely reported.",
            )
        )
    _scope_checks(canonical, folded, covered, issues)

    if claim_type == "node":
        predicate_pos = _add_present(folded, canonical["predicate"], predicate_path, covered, issues, code="predicate_missing", start=max(0, subject_pos))
        cursor = max(0, predicate_pos)
        object_positions: list[int] = []
        for path, value in zip(object_paths, canonical["objects"]):
            position = _add_present(folded, value, path, covered, issues, code="object_role_missing", start=cursor)
            object_positions.append(position)
            if position >= 0:
                cursor = position + len(_fold(value))
        if not (0 <= subject_pos < predicate_pos and all(position > predicate_pos for position in object_positions)):
            issues.append(_issue("node_direction_mismatch", "$.canonical_support_claim", "Node subject, slot, and value are not directed in order."))
        relation_end = max(object_positions) if object_positions else len(folded)
        _relation_negation_check(
            folded,
            0,
            relation_end,
            canonical["polarity"],
            "$.canonical_support_claim.polarity",
            covered,
            issues,
        )
        relation_segment = folded[max(0, subject_pos) : relation_end + max((len(_fold(value)) for value in canonical["objects"]), default=0)]
        _status_check(
            folded,
            canonical["status"],
            "$.canonical_support_claim.status",
            covered,
            issues,
            segment=folded if _token(canonical["status"]) == "active" else None,
            has_scope=isinstance(canonical.get("scope"), Mapping),
        )

    elif claim_type == "membership":
        cues = ("member of", "belongs to", "included in", "part of")
        cue_positions = [folded.find(cue, max(0, subject_pos)) for cue in cues]
        cue_positions = [value for value in cue_positions if value >= 0]
        predicate_pos = min(cue_positions) if cue_positions else -1
        if predicate_pos >= 0:
            covered.add(predicate_path)
        else:
            issues.append(_issue("predicate_missing", predicate_path, "Natural membership predicate is absent."))
        set_pos = _add_present(folded, canonical["objects"][0], object_paths[0], covered, issues, code="membership_set_missing", start=max(0, predicate_pos))
        if not (0 <= subject_pos < predicate_pos < set_pos):
            issues.append(_issue("membership_direction_mismatch", "$.canonical_support_claim", "Member, membership relation, and set are not directed in order."))
        _relation_negation_check(folded, subject_pos, set_pos + len(_fold(canonical["objects"][0])), canonical["polarity"], "$.canonical_support_claim.polarity", covered, issues)
        _add_present(folded, canonical["entity"], "$.canonical_support_claim.entity", covered, issues, code="membership_entity_missing")
        _add_present(folded, canonical["slot"], "$.canonical_support_claim.slot", covered, issues, code="membership_slot_missing")
        covered.add("$.canonical_support_claim.included")
        _status_check(folded, canonical["status"], "$.canonical_support_claim.status", covered, issues)

    elif claim_type == "visibility":
        action_pos = _add_present(folded, canonical["predicate"], predicate_path, covered, issues, code="visibility_action_missing", start=max(0, subject_pos))
        resource_pos = _add_present(folded, canonical["objects"][0], object_paths[0], covered, issues, code="visibility_resource_missing", start=max(0, action_pos))
        if not (0 <= subject_pos < action_pos < resource_pos):
            issues.append(_issue("visibility_direction_mismatch", "$.canonical_support_claim", "Observer, action, and resource are not directed in order."))
        _relation_negation_check(folded, subject_pos, resource_pos + len(_fold(canonical["objects"][0])), canonical["polarity"], "$.canonical_support_claim.polarity", covered, issues)
        owner_pos = _add_present(folded, canonical["owner"], "$.canonical_support_claim.owner", covered, issues, code="visibility_owner_missing")
        access_pos = _add_present(folded, canonical["access_class"], "$.canonical_support_claim.access_class", covered, issues, code="visibility_access_class_missing")
        ownership_cue = any(
            cue in folded
            for cue in ("owns", "owned by", "belongs to", "keeps", "maintains", "set by")
        )
        if owner_pos < 0 or not ownership_cue:
            issues.append(_issue("visibility_owner_role_missing", "$.canonical_support_claim.owner", "Owner is not expressed in an ownership role."))
        if access_pos < 0:
            issues.append(_issue("visibility_access_role_missing", "$.canonical_support_claim.access_class", "Access class is absent."))
        _status_check(folded, canonical["status"], "$.canonical_support_claim.status", covered, issues, has_scope=True)

    elif claim_type == "status_transition":
        slot_pos = _add_present(folded, canonical["slot"], "$.canonical_support_claim.slot", covered, issues, code="transition_slot_missing", start=max(0, subject_pos))
        old_pos = _add_present(folded, canonical["objects"][0], object_paths[0], covered, issues, code="transition_from_value_missing", start=max(0, slot_pos))
        transition_markers = (
            "changed",
            "changes",
            "moved",
            "moves",
            "switched",
            "transitions",
            "updated",
            "became",
            "from",
        )
        marker_positions = [
            folded.find(marker, max(0, slot_pos)) for marker in transition_markers
        ]
        marker_positions = [value for value in marker_positions if value >= 0]
        transition_pos = min(marker_positions) if marker_positions else -1
        new_pos = _add_present(folded, canonical["objects"][1], object_paths[1], covered, issues, code="transition_to_value_missing", start=max(old_pos + 1, transition_pos))
        if transition_pos >= 0:
            covered.add(predicate_path)
        else:
            issues.append(_issue("predicate_missing", predicate_path, "Natural transition marker is absent."))
        if not (0 <= subject_pos < slot_pos <= transition_pos < old_pos < new_pos):
            issues.append(_issue("transition_direction_mismatch", "$.canonical_support_claim", "Transition does not preserve subject/slot/from/to order."))
        _relation_negation_check(folded, subject_pos, new_pos + len(_fold(canonical["objects"][1])), canonical["polarity"], "$.canonical_support_claim.polarity", covered, issues)
        old_segment = folded[max(0, subject_pos) : max(new_pos, old_pos + 1)]
        new_segment = folded[max(0, new_pos) :]
        _status_check(folded, canonical["from_status"], "$.canonical_support_claim.from_status", covered, issues, segment=old_segment)
        _status_check(folded, canonical["to_status"], "$.canonical_support_claim.to_status", covered, issues, segment=new_segment)
        _status_check(folded, canonical["status"], "$.canonical_support_claim.status", covered, issues, segment=new_segment)

    elif claim_type == "comparison":
        slot_pos = _add_present(folded, canonical["slot"], "$.canonical_support_claim.slot", covered, issues, code="comparison_slot_missing")
        left_pos = _add_present(folded, canonical["left_value"], "$.canonical_support_claim.left_value", covered, issues, code="comparison_left_missing", start=max(0, subject_pos))
        right_entity_pos = _add_present(folded, canonical["objects"][0], object_paths[0], covered, issues, code="comparison_right_entity_missing", start=max(0, left_pos + 1))
        right_pos = _add_present(folded, canonical["right_value"], "$.canonical_support_claim.right_value", covered, issues, code="comparison_right_missing", start=max(0, right_entity_pos))
        difference_pos = _add_present(folded, canonical["absolute_difference"], "$.canonical_support_claim.absolute_difference", covered, issues, code="comparison_difference_missing", start=max(0, right_pos + 1))
        if difference_pos >= 0:
            covered.add(object_paths[1])
        cue_positions = [folded.find(cue) for cue in ("absolute difference", "difference between", "differ by")]
        cue_positions = [value for value in cue_positions if value >= 0]
        if cue_positions:
            covered.add(predicate_path)
        else:
            issues.append(_issue("predicate_missing", predicate_path, "Natural comparison predicate is absent."))
        if not (0 <= subject_pos < left_pos < right_entity_pos < right_pos < difference_pos):
            issues.append(_issue("comparison_direction_mismatch", "$.canonical_support_claim", "Comparison left/right roles are not ordered."))
        covered.add("$.canonical_support_claim.polarity")
        _status_check(folded, canonical["status"], "$.canonical_support_claim.status", covered, issues)

    elif claim_type == "cause":
        cause = canonical["cause_event"]
        effect = canonical["effect_event"]
        cause_entity_pos = _phrase_position(folded, cause["entity"])
        cause_markers = (" caused ", " causes ", " led to ", " leads to ", " resulted in ", " triggered ")
        marker_positions = [folded.find(marker.strip(), max(0, cause_entity_pos)) for marker in cause_markers]
        marker_positions = [value for value in marker_positions if value >= 0]
        predicate_pos = min(marker_positions) if marker_positions else -1
        effect_entity_pos = _phrase_position(folded, effect["entity"], max(0, predicate_pos))
        _descriptor_text_checks(cause, "$.canonical_support_claim.cause_event", folded, 0, predicate_pos, covered, issues)
        _descriptor_text_checks(effect, "$.canonical_support_claim.effect_event", folded, predicate_pos, len(folded), covered, issues)
        if predicate_pos >= 0:
            covered.add(predicate_path)
        else:
            issues.append(_issue("predicate_missing", predicate_path, "Directed causal predicate is absent."))
        if not (0 <= cause_entity_pos < predicate_pos < effect_entity_pos):
            issues.append(_issue("cause_direction_mismatch", "$.canonical_support_claim", "Cause endpoint must precede causal predicate and effect endpoint."))
        if cause_entity_pos >= 0:
            covered.add(subject_path)
        if effect_entity_pos >= 0:
            covered.add(object_paths[0])
        _relation_negation_check(folded, cause_entity_pos, effect_entity_pos, canonical["polarity"], "$.canonical_support_claim.polarity", covered, issues)
        _status_check(folded, canonical["status"], "$.canonical_support_claim.status", covered, issues)

    elif claim_type == "dependency":
        source = canonical["subject_event"]
        targets = canonical["object_events"]
        source_entity_pos = _phrase_position(folded, source["entity"])
        markers = (" depends on ", " relies on ", " requires ", " is based on ")
        marker_positions = [folded.find(marker.strip(), max(0, source_entity_pos)) for marker in markers]
        marker_positions = [value for value in marker_positions if value >= 0]
        predicate_pos = min(marker_positions) if marker_positions else -1
        _descriptor_text_checks(source, "$.canonical_support_claim.subject_event", folded, 0, predicate_pos, covered, issues)
        if source_entity_pos >= 0:
            covered.add(subject_path)
        if predicate_pos >= 0:
            covered.add(predicate_path)
        else:
            issues.append(_issue("predicate_missing", predicate_path, "Natural dependency predicate is absent."))
        cursor = max(0, predicate_pos)
        for index, target in enumerate(targets):
            next_entity = _phrase_position(folded, target["entity"], cursor)
            next_boundary = len(folded)
            if index + 1 < len(targets):
                candidate = _phrase_position(folded, targets[index + 1]["entity"], max(cursor, next_entity + 1))
                if candidate >= 0:
                    next_boundary = candidate
            _descriptor_text_checks(target, f"$.canonical_support_claim.object_events[{index}]", folded, cursor, next_boundary, covered, issues)
            if next_entity >= 0:
                covered.add(object_paths[index])
                cursor = next_entity + len(_fold(target["entity"]))
            else:
                issues.append(_issue("dependency_target_missing", f"$.canonical_support_claim.object_events[{index}]", "Dependency target entity is absent after the directed predicate."))
        if not (0 <= source_entity_pos < predicate_pos):
            issues.append(_issue("dependency_direction_mismatch", "$.canonical_support_claim", "Dependency source must precede its directed predicate and targets."))
        _relation_negation_check(folded, source_entity_pos, len(folded), canonical["polarity"], "$.canonical_support_claim.polarity", covered, issues)
        _status_check(folded, canonical["status"], "$.canonical_support_claim.status", covered, issues, segment=folded[:predicate_pos] if predicate_pos >= 0 else folded)

    for row in _path_rows(canonical.get("time"), "$.canonical_support_claim.time"):
        covered.add(row["path"])


def validate_typed_frozen_support(claim: Mapping[str, Any], support: Mapping[str, Any] | str) -> dict[str, Any]:
    """Validate a complete frozen-support record and its natural quote.

    A bare string is inspected for semantic diagnostics but is never admitted:
    without the signed manifest and event binding there is no lossless proof
    that the quote belongs to this claim.
    """

    issues: list[dict[str, Any]] = []
    if not isinstance(claim, Mapping):
        issues.append(_issue("claim_not_object", "$.claim", "Claim must be an object."))
        canonical: Mapping[str, Any] | None = None
        claim_type = ""
        manifest: list[dict[str, Any]] = []
    else:
        schema_issues, canonical, claim_type = _schema_issues(claim)
        issues.extend(schema_issues)
        if not schema_issues and canonical is not None:
            try:
                manifest = required_semantic_paths(claim)
            except TruthSupportValidationError as error:
                issues.append(_issue("manifest_compile_failed", "$.claim", str(error)))
                manifest = []
        else:
            manifest = []

    record: Mapping[str, Any] | None
    if isinstance(support, Mapping):
        record = support
        quote = support.get("quote")
        missing_fields = sorted(_RECORD_FIELDS - set(support))
        unexpected_fields = sorted(set(support) - _RECORD_FIELDS)
        for field in missing_fields:
            issues.append(_issue("support_field_missing", f"$.support.{field}", "Required support envelope field is absent."))
        for field in unexpected_fields:
            issues.append(_issue("support_field_unexpected", f"$.support.{field}", "Unexpected unvalidated support envelope field is present."))
        if support.get("schema_version") != EXPECTED_SUPPORT_SCHEMA_VERSION:
            issues.append(_issue("support_schema_mismatch", "$.support.schema_version", "Support schema version is not the independently admitted version.", expected=EXPECTED_SUPPORT_SCHEMA_VERSION, actual=support.get("schema_version")))
        if isinstance(claim, Mapping):
            expectations = {
                "claim_id": _text(claim.get("claim_id")),
                "claim_type": claim_type,
                "operation": _text(claim.get("operation")) or None,
                "edge_role": _text(claim.get("edge_role")) or None,
                "qa_eligible": claim.get("qa_eligible") is True,
                "event_binding": _event_binding(claim),
                "semantic_path_manifest": manifest,
            }
            for field, expected in expectations.items():
                if support.get(field) != expected:
                    issues.append(_issue(f"{field}_mismatch", f"$.support.{field}", "Support envelope differs from the frozen claim.", expected=expected, actual=support.get(field)))
        supplied_hash = support.get("support_sha256")
        if not isinstance(supplied_hash, str) or re.fullmatch(r"[0-9a-f]{64}", supplied_hash) is None:
            issues.append(_issue("support_hash_invalid", "$.support.support_sha256", "Support hash must be lowercase SHA-256."))
        else:
            unhashed = {key: copy.deepcopy(value) for key, value in support.items() if key != "support_sha256"}
            try:
                expected_hash = _canonical_json_sha256(unhashed)
            except TruthSupportValidationError as error:
                issues.append(_issue("support_hash_input_invalid", "$.support", str(error)))
            else:
                if supplied_hash != expected_hash:
                    issues.append(_issue("support_hash_mismatch", "$.support.support_sha256", "Support record hash does not match its envelope and quote.", expected=expected_hash, actual=supplied_hash))
    else:
        record = None
        quote = support
        issues.append(_issue("support_record_required", "$.support", "A complete hashed support record is required; a bare quote cannot be admitted."))

    if not isinstance(quote, str) or not _text(quote):
        issues.append(_issue("quote_missing", "$.support.quote", "Natural support quote must be a non-empty string."))
        quote = ""

    covered: set[str] = set()
    if canonical is not None and not any(issue["code"].startswith("typed_") or issue["code"] in {"canonical_claim_invalid", "claim_type_invalid"} for issue in issues):
        _quote_checks(claim, canonical, claim_type, quote, covered, issues)

    required_paths = [row["path"] for row in manifest]
    missing_paths = sorted(set(required_paths) - covered)
    for path in missing_paths:
        if not any(issue.get("path") == path for issue in issues):
            issues.append(_issue("semantic_path_uncovered", path, "Typed semantic path is not covered by natural text or the bound hidden time manifest."))
    counts: dict[str, int] = {}
    for issue in issues:
        code = str(issue.get("code") or "unknown")
        counts[code] = counts.get(code, 0) + 1
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "pass": not issues,
        "claim_id": _text(claim.get("claim_id")) if isinstance(claim, Mapping) else "",
        "claim_type": claim_type,
        "required_semantic_paths": required_paths,
        "covered_semantic_paths": sorted(covered),
        "missing_semantic_paths": missing_paths,
        "manifest_only_semantic_paths": sorted(path for path in covered if ".time" in path),
        "issue_count": len(issues),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": issues,
    }


__all__ = [
    "EXPECTED_SUPPORT_SCHEMA_VERSION",
    "VALIDATION_SCHEMA_VERSION",
    "TruthSupportValidationError",
    "required_semantic_paths",
    "validate_typed_frozen_support",
]
