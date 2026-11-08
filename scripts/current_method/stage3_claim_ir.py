"""Deterministic, API-free Claim IR for Stage3 event plans.

The dialogue model must not be allowed to decide graph truth.  This module is
the small local boundary between a normalized ``current_event_plan`` and all
later dialogue/QA work:

* every event becomes a content-addressed node claim;
* graph topology is made explicit as typed relation claims;
* generation-control dependencies are separated from QA-eligible semantics;
* a frozen claim-bundle hash detects any later truth drift; and
* an initially unbound support ledger reserves ``claim_id / turn_id / exact
  support quote`` slots for dialogue realization.

There are deliberately no imports from the native API generation pipeline.
All public functions are pure with respect to their inputs and use only the
Python standard library.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage3_claim_ir_v1"
CLAIM_SCHEMA_VERSION = "stage3_claim_v1"
SUPPORT_LEDGER_SCHEMA_VERSION = "stage3_claim_support_ledger_v1"

NODE_CLAIM_TYPE = "node"
RELATION_CLAIM_TYPES = frozenset(
    {
        "dependency",
        "membership",
        "visibility",
        "status_transition",
        "cause",
        "comparison",
    }
)
CLAIM_TYPES = frozenset({NODE_CLAIM_TYPE, *RELATION_CLAIM_TYPES})
EDGE_ROLES = frozenset({"control", "semantic"})

ACTIVE_STATUSES = frozenset({"active", "restored"})
FALSE_STATUSES = frozenset({"false"})
SEMANTIC_DEPENDENCY_OPERATIONS = frozenset({"bridge", "relation_lookup"})

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

RELATION_TYPE_ALIASES = {
    "depends": "dependency",
    "depends_on": "dependency",
    "dependency": "dependency",
    "member": "membership",
    "member_of": "membership",
    "membership": "membership",
    "permission": "visibility",
    "access": "visibility",
    "visibility": "visibility",
    "transition": "status_transition",
    "state_transition": "status_transition",
    "status_transition": "status_transition",
    "causal": "cause",
    "caused_by": "cause",
    "causes": "cause",
    "cause": "cause",
    "compare": "comparison",
    "difference": "comparison",
    "comparison": "comparison",
}

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_PREDICATE_SEPARATOR_RE = re.compile(r"[\s\-]+", flags=re.UNICODE)


class ClaimIRError(ValueError):
    """Raised when a plan cannot be represented without inventing truth."""


def normalize_text(value: Any) -> str:
    """Return stable display text without changing case or punctuation."""

    if value is None:
        return ""
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", str(value))
    ).strip()


def normalize_predicate(value: Any) -> str:
    """Normalize a schema token while preserving its semantic direction."""

    text = normalize_text(value).casefold()
    return _PREDICATE_SEPARATOR_RE.sub("_", text).strip("_")


def _normalize_status(value: Any) -> str | None:
    text = normalize_predicate(value)
    return text or None


def _json_safe_copy(value: Any, *, path: str) -> Any:
    copied = copy.deepcopy(value)
    try:
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ClaimIRError(f"{path} must be finite JSON data: {error}") from error
    return copied


def _normalized_json(value: Any) -> Any:
    """Normalize JSON strings recursively for content-addressed semantics."""

    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [_normalized_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalized_json(item) for item in value]
    if isinstance(value, dict):
        return {
            normalize_text(key): _normalized_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically after Unicode/whitespace normalization."""

    normalized = _normalized_json(value)
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ClaimIRError(f"value is not canonical JSON: {error}") from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def exact_json_sha256(value: Any) -> str:
    """Hash exact JSON content without semantic string normalization."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ClaimIRError(f"value is not exact canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _virtual_event_id(kind: str, semantic_value: Any) -> str:
    """Return a stable endpoint ID for a semantic object without an event row."""

    kind = normalize_predicate(kind) or "object"
    return f"virtual:{kind}:{canonical_json_sha256(semantic_value)[:20]}"


def _event_id(event: Mapping[str, Any]) -> str:
    return normalize_text(
        event.get("event_instance_id")
        or event.get("event_id")
        or event.get("id")
        or event.get("node_id")
    )


def _event_descriptor(event: Mapping[str, Any] | None, event_id: str) -> dict[str, Any]:
    if event is None:
        return {"event_id": event_id, "entity": None, "slot": None, "value": None}
    return {
        "event_id": event_id,
        "entity": _normalized_json(event.get("entity")),
        "slot": normalize_predicate(event.get("slot")) or None,
        "value": _normalized_json(event.get("value")),
        "status": _normalize_status(event.get("status")),
        "time_index": _time_index(event.get("time_index")),
        "global_time_index": _time_index(event.get("global_time_index")),
    }


def _time_index(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _condition_from_event(event: Mapping[str, Any]) -> Any:
    if event.get("condition") is not None:
        return _normalized_json(event.get("condition"))
    membership = event.get("set_membership")
    if isinstance(membership, dict) and membership.get("condition") is not None:
        return _normalized_json(membership.get("condition"))
    scope = event.get("scope")
    if isinstance(scope, dict) and scope.get("condition") is not None:
        return _normalized_json(scope.get("condition"))
    return None


def _node_condition_from_event(event: Mapping[str, Any]) -> Any:
    """Return only condition semantics asserted by the node itself.

    Nested ``set_membership.condition`` and ``scope.condition`` qualify their
    respective relation claims.  Promoting either into the ordinary node fact
    would make a node support quote prove relation semantics it never states.
    The full membership/scope objects remain available to their dedicated
    relation compilers below.
    """

    if event.get("condition") is not None:
        return _normalized_json(event.get("condition"))
    return None


def _scope_from_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    scope = event.get("scope")
    if not isinstance(scope, dict):
        return None
    return _normalized_json(scope)


def _polarity_from_event(event: Mapping[str, Any]) -> str:
    explicit = normalize_predicate(event.get("polarity"))
    if explicit in {"positive", "negative"}:
        return explicit
    if event.get("negated") is True or _normalize_status(event.get("status")) in FALSE_STATUSES:
        return "negative"
    return "positive"


def _normalize_edge_role(value: Any, *, default: str) -> str:
    role = normalize_predicate(value) or default
    if role not in EDGE_ROLES:
        raise ClaimIRError(
            f"edge_role must be one of {sorted(EDGE_ROLES)}, got {value!r}"
        )
    return role


def _normalize_qa_eligible(value: Any, *, edge_role: str, default: bool) -> bool:
    if edge_role == "control":
        return False
    if value is None:
        return bool(default)
    if not isinstance(value, bool):
        raise ClaimIRError("qa_eligible must be a JSON boolean when present")
    return value


def _identity_payload(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable graph truth used by both IDs and frozen hashes.

    ``qa_eligible`` is deliberately absent.  It is an evidence-realization
    state: the same frozen node becomes eligible only after an exact dialogue
    quote is bound.  Pre-dialogue and post-dialogue bundles must therefore
    share claim IDs and ``claim_bundle_hash``; ``validated_bundle_hash`` below
    records the later eligibility/ledger state.
    """

    fields = (
        "claim_type",
        "node_type",
        "subject_event_id",
        "predicate",
        "object_event_ids",
        "edge_role",
        "polarity",
        "status",
        "time_index",
        "global_time_index",
        "condition",
        "scope",
        "source_event_ids",
        "entity",
        "slot",
        "value",
        "canonical_support_claim",
        "truth_signature",
    )
    return {
        "identity_schema_version": CLAIM_SCHEMA_VERSION,
        **{field: _normalized_json(claim.get(field)) for field in fields},
    }


def _finalize_claim(claim: dict[str, Any]) -> dict[str, Any]:
    claim_type = normalize_predicate(claim.get("claim_type"))
    if claim_type not in CLAIM_TYPES:
        raise ClaimIRError(
            f"claim_type must be one of {sorted(CLAIM_TYPES)}, got {claim_type!r}"
        )
    claim["claim_type"] = claim_type
    claim["predicate"] = normalize_predicate(claim.get("predicate"))
    if not claim["predicate"]:
        raise ClaimIRError("claim predicate must be non-empty")
    claim["edge_role"] = _normalize_edge_role(
        claim.get("edge_role"), default="semantic"
    )
    claim["qa_eligible"] = _normalize_qa_eligible(
        claim.get("qa_eligible"),
        edge_role=claim["edge_role"],
        default=claim["edge_role"] == "semantic",
    )
    claim["object_event_ids"] = list(
        dict.fromkeys(
            normalize_text(event_id)
            for event_id in claim.get("object_event_ids", [])
            if normalize_text(event_id)
        )
    )
    claim["source_event_ids"] = list(
        dict.fromkeys(
            normalize_text(event_id)
            for event_id in claim.get("source_event_ids", [])
            if normalize_text(event_id)
        )
    )
    # Optional time axes are omitted when unavailable.  Emitting JSON null for
    # an index falsely claims that an index field exists but has an invalid
    # value under the independent validator contract.
    for time_key in ("time_index", "global_time_index"):
        if claim.get(time_key) is None:
            claim.pop(time_key, None)
    claim["claim_id"] = "clm_" + canonical_json_sha256(_identity_payload(claim))[:32]
    claim["truth_signature_hash"] = canonical_json_sha256(claim["truth_signature"])
    return claim


def normalize_node_claim(
    event: Mapping[str, Any],
    *,
    operation: str | None = None,
    exact_support_bound: bool = False,
) -> dict[str, Any]:
    """Compile one event into a canonical node claim without mutating it."""

    if not isinstance(event, Mapping):
        raise ClaimIRError("node claim source must be an event object")
    source_event = _json_safe_copy(event, path="event")
    event_id = _event_id(event)
    if not event_id:
        raise ClaimIRError("event requires event_instance_id, event_id, id, or node_id")
    entity = _normalized_json(event.get("entity"))
    slot = normalize_predicate(event.get("slot"))
    if entity in (None, "") or not slot:
        raise ClaimIRError(f"event {event_id!r} requires non-empty entity and slot")
    if "value" not in event:
        raise ClaimIRError(f"event {event_id!r} requires value")
    value = _normalized_json(event.get("value"))
    status = _normalize_status(event.get("status"))
    polarity = _polarity_from_event(event)
    condition = _node_condition_from_event(event)
    scope = _scope_from_event(event)
    time_index = _time_index(event.get("time_index"))
    global_time_index = _time_index(event.get("global_time_index"))
    required = event.get("required_for_dialogue", True)
    if not isinstance(required, bool):
        raise ClaimIRError(f"event {event_id!r} required_for_dialogue must be boolean")
    qa_eligible = event.get("qa_eligible")
    if qa_eligible is None:
        # A node is a semantic candidate as soon as it exists in the frozen
        # graph, but it cannot enter a QA path until dialogue realization has
        # produced an exact support-ledger binding.  Callers may explicitly
        # set qa_eligible for fully local fixtures that already carry their own
        # evidence contract.
        qa_eligible = required and exact_support_bound
    edge_role = _normalize_edge_role(
        event.get("claim_edge_role"), default="semantic"
    )
    qa_eligible = _normalize_qa_eligible(
        qa_eligible, edge_role=edge_role, default=required
    )

    canonical_support_claim = {
        "subject": entity,
        "predicate": slot,
        "objects": [value],
        "polarity": polarity,
        "status": status,
        "time": {
            "time_index": time_index,
            "global_time_index": global_time_index,
        },
        "condition": condition,
        "scope": scope,
    }
    truth_signature = {
        "subject_event_id": event_id,
        **copy.deepcopy(canonical_support_claim),
    }
    support_text = normalize_text(
        event.get("dialogue_support_claim")
        or event.get("raw_support_claim")
        or (event.get("alignment_hints") or {}).get("support_claim")
        if isinstance(event.get("alignment_hints"), dict)
        else event.get("dialogue_support_claim") or event.get("raw_support_claim")
    )
    claim = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "claim_id": "",
        "claim_kind": "node",
        "claim_type": NODE_CLAIM_TYPE,
        "node_type": normalize_predicate(
            event.get("node_type") or event.get("event_type") or "event_fact"
        ),
        "subject_event_id": event_id,
        "predicate": slot,
        "object_event_ids": [],
        "edge_role": edge_role,
        "qa_eligible": qa_eligible,
        "polarity": polarity,
        "status": status,
        "time_index": time_index,
        "global_time_index": global_time_index,
        "condition": condition,
        "scope": scope,
        "source_event_ids": [event_id],
        "entity": entity,
        "slot": slot,
        "value": value,
        "canonical_support_claim": canonical_support_claim,
        "truth_signature": truth_signature,
        "canonical_support_text": _node_support_text(
            entity, slot, value, polarity, status, time_index, condition
        ),
        "support_text": support_text or None,
        "answer_bearing": bool(event.get("answer_bearing", qa_eligible)),
        "qa_candidate": required,
        "exact_support_bound": bool(exact_support_bound),
        "operation": OPERATION_ALIASES.get(normalize_predicate(operation), normalize_predicate(operation))
        or None,
        "provenance": {
            "source": "current_event_plan.events",
            "source_event_id": event_id,
        },
        # A deep copy is intentionally retained for replay/debugging. It is not
        # part of the claim identity, so non-semantic presentation metadata
        # cannot silently change truth IDs.
        "source_event": source_event,
    }
    return _finalize_claim(claim)


def _node_support_text(
    entity: Any,
    slot: str,
    value: Any,
    polarity: str,
    status: str | None,
    time_index: int | None,
    condition: Any,
) -> str:
    payload = {
        "subject": entity,
        "predicate": slot,
        "object": value,
        "polarity": polarity,
        "status": status,
        "time_index": time_index,
        "condition": condition,
    }
    return canonical_json(payload)


def _relation_claim(
    *,
    claim_type: str,
    subject_event_id: str,
    predicate: str,
    object_event_ids: Sequence[str],
    edge_role: str,
    qa_eligible: bool,
    polarity: str,
    status: str | None,
    time_index: int | None,
    global_time_index: int | None,
    condition: Any,
    scope: Any,
    source_event_ids: Sequence[str],
    canonical_support_claim: Mapping[str, Any],
    operation: str | None,
    provenance: Mapping[str, Any],
    support_text: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_support = _normalized_json(canonical_support_claim)
    # The independent validator resolves omitted support fields from the
    # claim-level semantics. Freeze those values into the canonical support
    # itself so the truth signature cannot silently drop a condition or scope.
    normalized_support.setdefault("condition", _normalized_json(condition))
    normalized_support.setdefault("scope", _normalized_json(scope))
    truth_signature = {
        "subject_event_id": normalize_text(subject_event_id),
        "object_event_ids": [normalize_text(item) for item in object_event_ids],
        **copy.deepcopy(normalized_support),
    }
    qa_candidate = bool(qa_eligible) and edge_role == "semantic"
    claim: dict[str, Any] = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "claim_id": "",
        "claim_kind": "relation",
        "claim_type": claim_type,
        "subject_event_id": normalize_text(subject_event_id),
        "predicate": predicate,
        "object_event_ids": list(object_event_ids),
        "edge_role": edge_role,
        # Relation truth is frozen before dialogue, while effective QA use is
        # evidence gated.  extract_claims promotes these two fields only when
        # a claim-addressed exact quote exists.
        "qa_eligible": False,
        "polarity": polarity,
        "status": status,
        "time_index": time_index,
        "global_time_index": global_time_index,
        "condition": _normalized_json(condition),
        "scope": _normalized_json(scope),
        "source_event_ids": list(source_event_ids),
        "canonical_support_claim": copy.deepcopy(normalized_support),
        "truth_signature": truth_signature,
        "canonical_support_text": canonical_json(normalized_support),
        "support_text": normalize_text(support_text) or None,
        "answer_bearing": False,
        "qa_candidate": qa_candidate,
        "semantic_candidate": edge_role == "semantic",
        "exact_support_bound": False,
        "operation": operation,
        "provenance": _normalized_json(provenance),
    }
    if extra:
        claim.update(_json_safe_copy(extra, path="relation extra"))
    return _finalize_claim(claim)


def _dependency_specs(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_dependencies = event.get("depends_on")
    if raw_dependencies is None:
        return []
    if not isinstance(raw_dependencies, list):
        raw_dependencies = [raw_dependencies]
    specs: list[dict[str, Any]] = []
    for raw in raw_dependencies:
        if isinstance(raw, Mapping):
            dependency_id = normalize_text(
                raw.get("event_id")
                or raw.get("target_event_id")
                or raw.get("object_event_id")
                or raw.get("id")
            )
            if dependency_id:
                specs.append(
                    {
                        "event_id": dependency_id,
                        "edge_role": raw.get("edge_role"),
                        "qa_eligible": raw.get("qa_eligible"),
                        "predicate": raw.get("predicate"),
                    }
                )
        else:
            dependency_id = normalize_text(raw)
            if dependency_id:
                specs.append({"event_id": dependency_id})
    return specs


def _dependency_claims(
    events: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    operation: str | None,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    semantic_default = operation in SEMANTIC_DEPENDENCY_OPERATIONS
    for event in events:
        current_id = _event_id(event)
        for spec in _dependency_specs(event):
            dependency_id = spec["event_id"]
            dependency = by_id.get(dependency_id)
            explicit_role = spec.get("edge_role")
            if explicit_role is None:
                explicit_role = event.get("dependency_edge_role")
            role = _normalize_edge_role(
                explicit_role,
                default="semantic" if semantic_default else "control",
            )
            explicit_qa = spec.get("qa_eligible")
            if explicit_qa is None:
                explicit_qa = event.get("dependency_qa_eligible")
            qa_eligible = _normalize_qa_eligible(
                explicit_qa,
                edge_role=role,
                default=semantic_default,
            )
            current_descriptor = _event_descriptor(event, current_id)
            dependency_descriptor = _event_descriptor(dependency, dependency_id)
            canonical = {
                "subject": current_descriptor.get("entity"),
                "predicate": normalize_predicate(spec.get("predicate")) or "depends_on",
                "objects": [dependency_descriptor.get("entity")],
                "polarity": "positive",
                "status": _normalize_status(event.get("status")),
                "time": {
                    "time_index": _time_index(event.get("time_index")),
                    "global_time_index": _time_index(event.get("global_time_index")),
                },
                "condition": _condition_from_event(event),
                "subject_event": current_descriptor,
                "object_events": [dependency_descriptor],
            }
            claims.append(
                _relation_claim(
                    claim_type="dependency",
                    subject_event_id=current_id,
                    predicate=canonical["predicate"],
                    object_event_ids=[dependency_id],
                    edge_role=role,
                    qa_eligible=qa_eligible,
                    polarity="positive",
                    status=_normalize_status(event.get("status")),
                    time_index=_time_index(event.get("time_index")),
                    global_time_index=_time_index(event.get("global_time_index")),
                    condition=_condition_from_event(event),
                    scope=_scope_from_event(event),
                    source_event_ids=[dependency_id, current_id],
                    canonical_support_claim=canonical,
                    operation=operation,
                    provenance={"source": "event.depends_on"},
                )
            )
    return claims


def _membership_claims(
    events: Sequence[Mapping[str, Any]], operation: str | None
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for event in events:
        membership = event.get("set_membership")
        if not isinstance(membership, dict):
            continue
        set_id = normalize_text(membership.get("set_id"))
        included = membership.get("included")
        if not set_id or not isinstance(included, bool):
            raise ClaimIRError(
                f"event {_event_id(event)!r} set_membership requires set_id and boolean included"
            )
        event_id = _event_id(event)
        polarity = "positive" if included else "negative"
        predicate = "member_of" if included else "not_member_of"
        member = _normalized_json(event.get("value"))
        set_event_id = _virtual_event_id("set", set_id)
        condition = _condition_from_event(event)
        canonical = {
            "subject": member,
            "predicate": predicate,
            "objects": [set_id],
            "polarity": polarity,
            "status": _normalize_status(event.get("status")),
            "time": {
                "time_index": _time_index(event.get("time_index")),
                "global_time_index": _time_index(event.get("global_time_index")),
            },
            "condition": condition,
            "included": included,
            "entity": _normalized_json(event.get("entity")),
            "slot": normalize_predicate(event.get("slot")),
        }
        claims.append(
            _relation_claim(
                claim_type="membership",
                subject_event_id=event_id,
                predicate=predicate,
                # The event/value is the directed member subject; the named
                # set is a stable virtual object endpoint.  Reusing the source
                # event as its own object made membership direction ambiguous
                # and could produce duplicate endpoint IDs downstream.
                object_event_ids=[set_event_id],
                edge_role="semantic",
                qa_eligible=True,
                polarity=polarity,
                status=_normalize_status(event.get("status")),
                time_index=_time_index(event.get("time_index")),
                global_time_index=_time_index(event.get("global_time_index")),
                condition=condition,
                scope=_scope_from_event(event),
                source_event_ids=[event_id, set_event_id],
                canonical_support_claim=canonical,
                operation=operation,
                provenance={"source": "event.set_membership"},
                extra={
                    "set_id": set_id,
                    "set_event_id": set_event_id,
                    "included": included,
                },
            )
        )
    return claims


def _visibility_claims(
    events: Sequence[Mapping[str, Any]], operation: str | None
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for event in events:
        scope = event.get("scope")
        if not isinstance(scope, dict):
            continue
        has_visibility_shape = operation == "visibility" or any(
            key in event or key in scope
            for key in ("observer", "permission", "action", "owner", "resource", "access_class")
        )
        if not has_visibility_shape:
            continue
        event_id = _event_id(event)
        observer = _normalized_json(event.get("observer") or scope.get("observer"))
        action = normalize_predicate(
            event.get("action")
            or scope.get("permission")
            or scope.get("action")
            or "view"
        )
        resource = _normalized_json(
            event.get("resource") or scope.get("resource") or event.get("entity")
        )
        resource_event_id = _virtual_event_id("resource", resource)
        owner = _normalized_json(event.get("owner") or scope.get("owner"))
        condition = (
            _normalized_json(event.get("condition"))
            if event.get("condition") is not None
            else _normalized_json(scope.get("condition"))
            if scope.get("condition") is not None
            else None
        )
        canonical = {
            "subject": observer,
            "predicate": action,
            "objects": [resource],
            "polarity": _polarity_from_event(event),
            "status": _normalize_status(event.get("status")),
            "time": {
                "time_index": _time_index(event.get("time_index")),
                "global_time_index": _time_index(event.get("global_time_index")),
            },
            "condition": condition,
            "owner": owner,
            "access_class": _normalized_json(
                event.get("access_class") or scope.get("access_class") or event.get("value")
            ),
        }
        claims.append(
            _relation_claim(
                claim_type="visibility",
                subject_event_id=event_id,
                predicate=action,
                object_event_ids=[resource_event_id],
                edge_role="semantic",
                qa_eligible=True,
                polarity=_polarity_from_event(event),
                status=_normalize_status(event.get("status")),
                time_index=_time_index(event.get("time_index")),
                global_time_index=_time_index(event.get("global_time_index")),
                condition=condition,
                scope=_scope_from_event(event),
                source_event_ids=[event_id, resource_event_id],
                canonical_support_claim=canonical,
                operation=operation,
                provenance={"source": "event.scope"},
                extra={
                    "observer": observer,
                    "action": action,
                    "resource": resource,
                    "resource_event_id": resource_event_id,
                    "owner": owner,
                },
            )
        )
    return claims


def _same_semantic_value(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _status_transition_claims(
    events: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    operation: str | None,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for current in events:
        if current.get("previous_value") is None:
            continue
        current_id = _event_id(current)
        candidates: list[Mapping[str, Any]] = []
        for spec in _dependency_specs(current):
            dependency = by_id.get(spec["event_id"])
            if dependency is not None:
                candidates.append(dependency)
        if not candidates:
            candidates = [
                event
                for event in events
                if _event_id(event) != current_id
                and normalize_text(event.get("entity")) == normalize_text(current.get("entity"))
                and normalize_predicate(event.get("slot")) == normalize_predicate(current.get("slot"))
                and _same_semantic_value(event.get("value"), current.get("previous_value"))
                and (_time_index(event.get("time_index")) or -1)
                < (_time_index(current.get("time_index")) or 10**18)
            ]
            candidates.sort(
                key=lambda event: (
                    _time_index(event.get("time_index")) or -1,
                    _event_id(event),
                ),
                reverse=True,
            )
        previous = next(
            (
                event
                for event in candidates
                if normalize_text(event.get("entity")) == normalize_text(current.get("entity"))
                and normalize_predicate(event.get("slot")) == normalize_predicate(current.get("slot"))
                and _same_semantic_value(event.get("value"), current.get("previous_value"))
            ),
            None,
        )
        if previous is None:
            continue
        previous_id = _event_id(previous)
        canonical = {
            "subject": _normalized_json(current.get("entity")),
            "predicate": "transitions_to",
            "objects": [
                _normalized_json(previous.get("value")),
                _normalized_json(current.get("value")),
            ],
            "polarity": "positive",
            "status": _normalize_status(current.get("status")),
            "time": {
                "from_time_index": _time_index(previous.get("time_index")),
                "time_index": _time_index(current.get("time_index")),
                "global_time_index": _time_index(current.get("global_time_index")),
            },
            "condition": _condition_from_event(current),
            "slot": normalize_predicate(current.get("slot")),
            "from_status": _normalize_status(previous.get("status")),
            "to_status": _normalize_status(current.get("status")),
        }
        claims.append(
            _relation_claim(
                claim_type="status_transition",
                subject_event_id=previous_id,
                predicate="transitions_to",
                object_event_ids=[current_id],
                edge_role="semantic",
                qa_eligible=True,
                polarity="positive",
                status=_normalize_status(current.get("status")),
                time_index=_time_index(current.get("time_index")),
                global_time_index=_time_index(current.get("global_time_index")),
                condition=_condition_from_event(current),
                scope=_scope_from_event(current),
                source_event_ids=[previous_id, current_id],
                canonical_support_claim=canonical,
                operation=operation,
                provenance={
                    "source": "event.previous_value",
                    "matched_predecessor": previous_id,
                },
                extra={
                    "from_status": _normalize_status(previous.get("status")),
                    "to_status": _normalize_status(current.get("status")),
                    "previous_value": _normalized_json(previous.get("value")),
                    "value": _normalized_json(current.get("value")),
                },
            )
        )
    return claims


def _cause_claims(
    events: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    operation: str | None,
) -> list[dict[str, Any]]:
    if operation != "cause_consequence":
        return []
    claims: list[dict[str, Any]] = []
    for effect in events:
        effect_id = _event_id(effect)
        for spec in _dependency_specs(effect):
            cause = by_id.get(spec["event_id"])
            if cause is None:
                continue
            cause_id = _event_id(cause)
            cause_time = _time_index(cause.get("time_index"))
            effect_time = _time_index(effect.get("time_index"))
            if cause_time is not None and effect_time is not None and cause_time >= effect_time:
                continue
            canonical = {
                "subject": _normalized_json(cause.get("entity")),
                "predicate": "causes",
                "objects": [_normalized_json(effect.get("entity"))],
                "polarity": "positive",
                "status": _normalize_status(effect.get("status")),
                "time": {
                    "cause_time_index": cause_time,
                    "time_index": effect_time,
                    "global_time_index": _time_index(effect.get("global_time_index")),
                },
                "condition": _condition_from_event(effect),
                "cause_event": _event_descriptor(cause, cause_id),
                "effect_event": _event_descriptor(effect, effect_id),
            }
            claims.append(
                _relation_claim(
                    claim_type="cause",
                    subject_event_id=cause_id,
                    predicate="causes",
                    object_event_ids=[effect_id],
                    edge_role="semantic",
                    qa_eligible=True,
                    polarity="positive",
                    status=_normalize_status(effect.get("status")),
                    time_index=effect_time,
                    global_time_index=_time_index(effect.get("global_time_index")),
                    condition=_condition_from_event(effect),
                    scope=_scope_from_event(effect),
                    source_event_ids=[cause_id, effect_id],
                    canonical_support_claim=canonical,
                    operation=operation,
                    provenance={"source": "ordered event.depends_on"},
                )
            )
    return claims


def _comparison_claims(
    events: Sequence[Mapping[str, Any]], operation: str | None
) -> list[dict[str, Any]]:
    if operation != "comparison":
        return []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for event in events:
        time_index = _time_index(event.get("time_index"))
        slot = normalize_predicate(event.get("slot"))
        value = event.get("value")
        if (
            slot
            and time_index is not None
            and _normalize_status(event.get("status")) in ACTIVE_STATUSES
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            grouped.setdefault((slot, time_index), []).append(event)
    claims: list[dict[str, Any]] = []
    for (slot, time_index), group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        first, second = sorted(group, key=_event_id)
        if normalize_text(first.get("entity")) == normalize_text(second.get("entity")):
            continue
        if first.get("value") == second.get("value"):
            continue
        first_id, second_id = _event_id(first), _event_id(second)
        difference = abs(first["value"] - second["value"])
        canonical = {
            "subject": _normalized_json(first.get("entity")),
            "predicate": "absolute_difference",
            "objects": [_normalized_json(second.get("entity")), difference],
            "polarity": "positive",
            "status": "active",
            "time": {
                "time_index": time_index,
                "global_time_index": _time_index(first.get("global_time_index")),
            },
            "condition": None,
            "slot": slot,
            "left_value": _normalized_json(first.get("value")),
            "right_value": _normalized_json(second.get("value")),
            "absolute_difference": difference,
        }
        claims.append(
            _relation_claim(
                claim_type="comparison",
                subject_event_id=first_id,
                predicate="absolute_difference",
                object_event_ids=[second_id],
                edge_role="semantic",
                qa_eligible=True,
                polarity="positive",
                status="active",
                time_index=time_index,
                global_time_index=_time_index(first.get("global_time_index")),
                condition=None,
                scope=None,
                source_event_ids=[first_id, second_id],
                canonical_support_claim=canonical,
                operation=operation,
                provenance={"source": "local numeric comparison"},
                extra={
                    "comparison_slot": slot,
                    "left_value": _normalized_json(first.get("value")),
                    "right_value": _normalized_json(second.get("value")),
                    "derived_value": difference,
                },
            )
        )
    return claims


def _explicit_relation_claims(
    current_event_plan: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    operation: str | None,
) -> list[dict[str, Any]]:
    """Compile optional explicit relations used by offline replay fixtures."""

    specs: list[tuple[Mapping[str, Any], str | None]] = []
    raw_relations = current_event_plan.get("relations")
    if isinstance(raw_relations, list):
        specs.extend((item, None) for item in raw_relations if isinstance(item, Mapping))
    for event in events:
        relations = event.get("relations")
        if isinstance(relations, Mapping):
            relations = [relations]
        if isinstance(relations, list):
            specs.extend(
                (item, _event_id(event))
                for item in relations
                if isinstance(item, Mapping)
            )

    claims: list[dict[str, Any]] = []
    for spec, default_subject in specs:
        raw_type = normalize_predicate(
            spec.get("claim_type") or spec.get("relation_type") or spec.get("type")
        )
        claim_type = RELATION_TYPE_ALIASES.get(raw_type, raw_type)
        if claim_type not in RELATION_CLAIM_TYPES:
            raise ClaimIRError(f"unsupported explicit relation type: {raw_type!r}")
        subject_id = normalize_text(
            spec.get("subject_event_id")
            or spec.get("source_event_id")
            or spec.get("from_event_id")
            or default_subject
        )
        raw_objects = spec.get("object_event_ids")
        if raw_objects is None:
            raw_objects = [
                spec.get("object_event_id")
                or spec.get("target_event_id")
                or spec.get("to_event_id")
            ]
        elif not isinstance(raw_objects, list):
            raw_objects = [raw_objects]
        object_ids = [normalize_text(item) for item in raw_objects if normalize_text(item)]
        if not subject_id:
            raise ClaimIRError("explicit relation requires subject_event_id")
        default_role = "control" if claim_type == "dependency" else "semantic"
        role = _normalize_edge_role(spec.get("edge_role"), default=default_role)
        qa_eligible = _normalize_qa_eligible(
            spec.get("qa_eligible"),
            edge_role=role,
            default=role == "semantic",
        )
        subject = by_id.get(subject_id)
        objects = [by_id.get(event_id) for event_id in object_ids]
        predicate = normalize_predicate(spec.get("predicate")) or {
            "dependency": "depends_on",
            "membership": "member_of",
            "visibility": "view",
            "status_transition": "transitions_to",
            "cause": "causes",
            "comparison": "compares_with",
        }[claim_type]
        canonical = spec.get("canonical_support_claim")
        if not isinstance(canonical, Mapping):
            canonical = {
                "subject": _event_descriptor(subject, subject_id).get("entity"),
                "predicate": predicate,
                "objects": [
                    _event_descriptor(item, event_id).get("entity")
                    for item, event_id in zip(objects, object_ids)
                ],
                "polarity": normalize_predicate(spec.get("polarity")) or "positive",
                "status": _normalize_status(spec.get("status")),
                "time": {
                    "time_index": _time_index(spec.get("time_index")),
                    "global_time_index": _time_index(spec.get("global_time_index")),
                },
                "condition": _normalized_json(spec.get("condition")),
                "subject_event": _event_descriptor(subject, subject_id),
                "object_events": [
                    _event_descriptor(item, event_id)
                    for item, event_id in zip(objects, object_ids)
                ],
            }
        claims.append(
            _relation_claim(
                claim_type=claim_type,
                subject_event_id=subject_id,
                predicate=predicate,
                object_event_ids=object_ids,
                edge_role=role,
                qa_eligible=qa_eligible,
                polarity=normalize_predicate(spec.get("polarity")) or "positive",
                status=_normalize_status(spec.get("status")),
                time_index=_time_index(spec.get("time_index")),
                global_time_index=_time_index(spec.get("global_time_index")),
                condition=_normalized_json(spec.get("condition")),
                scope=_normalized_json(spec.get("scope")),
                source_event_ids=[subject_id, *object_ids],
                canonical_support_claim=canonical,
                operation=operation,
                provenance={"source": "current_event_plan explicit relation"},
                support_text=normalize_text(spec.get("support_text")) or None,
            )
        )
    return claims


def _claim_sort_key(claim: Mapping[str, Any]) -> tuple[Any, ...]:
    type_order = {
        "node": 0,
        "dependency": 1,
        "membership": 2,
        "visibility": 3,
        "status_transition": 4,
        "cause": 5,
        "comparison": 6,
    }
    return (
        type_order.get(str(claim.get("claim_type")), 99),
        str(claim.get("subject_event_id") or ""),
        tuple(claim.get("object_event_ids") or []),
        str(claim.get("predicate") or ""),
        str(claim.get("claim_id") or ""),
    )


def _dialogue_turn_map(dialogue_turns: Any) -> dict[str, str]:
    if isinstance(dialogue_turns, Mapping):
        dialogue_turns = dialogue_turns.get("turns")
    if dialogue_turns is None:
        return {}
    if not isinstance(dialogue_turns, list):
        raise ClaimIRError("dialogue_turns must be an array or an object with turns")
    turns: dict[str, str] = {}
    for index, turn in enumerate(dialogue_turns):
        if not isinstance(turn, Mapping):
            raise ClaimIRError(f"dialogue_turns[{index}] must be an object")
        turn_id = normalize_text(turn.get("turn_id") or turn.get("id"))
        text = turn.get("text")
        if turn_id and isinstance(text, str):
            turns[turn_id] = text
    return turns


def _alignment_map(current_alignment: Any) -> Mapping[str, Any]:
    if current_alignment is None:
        return {}
    if not isinstance(current_alignment, Mapping):
        raise ClaimIRError("current_alignment must be an object when present")
    nested = current_alignment.get("current_alignment")
    if isinstance(nested, Mapping):
        return nested
    return current_alignment


def _candidate_alignment_pairs(row: Any) -> list[tuple[str, str]]:
    if not isinstance(row, Mapping):
        return []
    direct_turn = normalize_text(row.get("turn_id"))
    direct_quote = row.get("exact_support_quote")
    if direct_quote is None:
        direct_quote = row.get("support_quote")
    pairs: list[tuple[str, str]] = []
    if direct_turn and isinstance(direct_quote, str) and direct_quote:
        pairs.append((direct_turn, direct_quote))
    turn_ids = row.get("turn_ids")
    support_quotes = row.get("support_quotes")
    if isinstance(turn_ids, list) and isinstance(support_quotes, list):
        pairs.extend(
            (normalize_text(turn_id), quote)
            for turn_id, quote in zip(turn_ids, support_quotes)
            if normalize_text(turn_id) and isinstance(quote, str) and quote
        )
    return pairs


def alignment_support_binding(
    current_alignment: Mapping[str, Any] | None,
    *,
    alignment_key: str,
    dialogue_turns: Any = None,
) -> dict[str, Any] | None:
    """Resolve one deterministic exact-quote binding from current alignment.

    When dialogue turns are supplied the quote must occur byte-for-byte in the
    referenced turn.  Without turns, the alignment's explicitly named support
    quote is retained for the independent validator to check later.
    """

    alignment = _alignment_map(current_alignment)
    key = normalize_text(alignment_key)
    row = alignment.get(key)
    turn_map = _dialogue_turn_map(dialogue_turns)
    for turn_id, quote in _candidate_alignment_pairs(row):
        if dialogue_turns is not None:
            turn_text = turn_map.get(turn_id)
            if turn_text is None or quote not in turn_text:
                continue
            status = "exact_quote_bound"
        else:
            status = "alignment_bound"
        return {
            "turn_id": turn_id,
            "exact_support_quote": quote,
            "verification_status": status,
        }
    return None


def _relation_event_alignment_keys(claim: Mapping[str, Any]) -> list[str]:
    claim_type = normalize_predicate(claim.get("claim_type"))
    if claim_type not in {"visibility", "membership", "cause"}:
        return []
    subject = normalize_text(claim.get("subject_event_id"))
    sources = [
        normalize_text(item)
        for item in claim.get("source_event_ids") or []
        if normalize_text(item)
    ]
    # Cause evidence is normally verbalized at the effect event. Membership
    # and visibility are event-local typed facts.
    ordered = list(reversed(sources)) if claim_type == "cause" else [subject, *sources]
    return list(dict.fromkeys(item for item in ordered if item))


def extract_claims(
    current_event_plan: Mapping[str, Any],
    *,
    current_alignment: Mapping[str, Any] | None = None,
    dialogue_turns: Any = None,
) -> list[dict[str, Any]]:
    """Extract canonical node/relation claims from a frozen event plan."""

    if not isinstance(current_event_plan, Mapping):
        raise ClaimIRError("current_event_plan must be a JSON object")
    raw_events = current_event_plan.get("events")
    if not isinstance(raw_events, list):
        raise ClaimIRError("current_event_plan.events must be a JSON array")
    events: list[Mapping[str, Any]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise ClaimIRError(f"current_event_plan.events[{index}] must be an object")
        event_id = _event_id(event)
        if not event_id:
            raise ClaimIRError(f"current_event_plan.events[{index}] has no event id")
        if event_id in by_id:
            raise ClaimIRError(f"duplicate event id: {event_id!r}")
        events.append(event)
        by_id[event_id] = event

    operation_key = normalize_predicate(
        current_event_plan.get("operation") or current_event_plan.get("qa_operation")
    )
    operation = OPERATION_ALIASES.get(operation_key, operation_key) or None
    claims: list[dict[str, Any]] = []
    for event in events:
        exact_support_bound = alignment_support_binding(
            current_alignment,
            alignment_key=_event_id(event),
            dialogue_turns=dialogue_turns,
        ) is not None
        claims.append(
            normalize_node_claim(
                event,
                operation=operation,
                exact_support_bound=exact_support_bound,
            )
        )
    claims.extend(_dependency_claims(events, by_id, operation))
    claims.extend(_membership_claims(events, operation))
    claims.extend(_visibility_claims(events, operation))
    claims.extend(_status_transition_claims(events, by_id, operation))
    claims.extend(_cause_claims(events, by_id, operation))
    claims.extend(_comparison_claims(events, operation))
    claims.extend(
        _explicit_relation_claims(current_event_plan, events, by_id, operation)
    )

    deduplicated: dict[str, dict[str, Any]] = {}
    for claim in claims:
        prior = deduplicated.get(claim["claim_id"])
        if prior is not None and _identity_payload(prior) != _identity_payload(claim):
            raise ClaimIRError(f"claim id collision: {claim['claim_id']}")
        deduplicated.setdefault(claim["claim_id"], claim)
    for claim in deduplicated.values():
        if claim.get("claim_kind") != "relation":
            continue
        binding = alignment_support_binding(
            current_alignment,
            alignment_key=claim["claim_id"],
            dialogue_turns=dialogue_turns,
        )
        binding_source = "claim_id" if binding is not None else None
        if binding is None and claim.get("qa_candidate"):
            for fallback_key in _relation_event_alignment_keys(claim):
                binding = alignment_support_binding(
                    current_alignment,
                    alignment_key=fallback_key,
                    dialogue_turns=dialogue_turns,
                )
                if binding is not None:
                    binding_source = f"event_alignment:{fallback_key}"
                    break
        effective = bool(claim.get("qa_candidate")) and binding is not None
        claim["support_binding_source"] = binding_source
        claim["qa_eligible"] = effective
        claim["answer_bearing"] = effective
        claim["exact_support_bound"] = binding is not None
    return sorted(deduplicated.values(), key=_claim_sort_key)


def build_support_ledger(
    claims: Iterable[Mapping[str, Any]],
    *,
    claim_bundle_hash: str | None = None,
    current_alignment: Mapping[str, Any] | None = None,
    dialogue_turns: Any = None,
) -> dict[str, Any]:
    """Create an immutable-by-convention, initially unbound evidence ledger."""

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in sorted(claims, key=_claim_sort_key):
        claim_id = normalize_text(claim.get("claim_id"))
        if not claim_id:
            raise ClaimIRError("support ledger claim is missing claim_id")
        if claim_id in seen:
            raise ClaimIRError(f"duplicate support-ledger claim id: {claim_id}")
        seen.add(claim_id)
        binding = alignment_support_binding(
            current_alignment,
            alignment_key=claim_id,
            dialogue_turns=dialogue_turns,
        )
        # Current alignment is event-addressed. A relation may reuse only one
        # exact event quote; the independent semantic validator still checks
        # subject, direction, status, condition and scope before acceptance.
        if binding is None:
            if claim.get("claim_type") == NODE_CLAIM_TYPE:
                fallback_keys = [claim.get("subject_event_id")]
            elif claim.get("qa_candidate"):
                fallback_keys = _relation_event_alignment_keys(claim)
            else:
                fallback_keys = []
            for fallback_key in dict.fromkeys(
                normalize_text(item) for item in fallback_keys if normalize_text(item)
            ):
                binding = alignment_support_binding(
                    current_alignment,
                    alignment_key=fallback_key,
                    dialogue_turns=dialogue_turns,
                )
                if binding is not None:
                    break
        entry = {
            "claim_id": claim_id,
            "turn_id": None,
            "exact_support_quote": None,
            "verification_status": "unbound",
            # Candidates are what the naturalizer must realize; effective
            # answer-bearing status remains false until this entry is bound.
            "required": bool(
                claim.get(
                    "qa_candidate",
                    claim.get("answer_bearing", claim.get("qa_eligible")),
                )
            ),
            "source_event_ids": copy.deepcopy(claim.get("source_event_ids") or []),
        }
        if binding is not None:
            entry.update(binding)
        entries.append(entry)
    by_claim_id = {entry["claim_id"]: copy.deepcopy(entry) for entry in entries}
    ledger = {
        "schema_version": SUPPORT_LEDGER_SCHEMA_VERSION,
        "claim_bundle_hash": claim_bundle_hash,
        "entries": entries,
        "by_claim_id": by_claim_id,
        "required_claim_ids": [
            entry["claim_id"] for entry in entries if entry["required"]
        ],
    }
    ledger["support_ledger_hash"] = exact_json_sha256(ledger)
    return ledger


def bind_support_evidence(
    support_ledger: Mapping[str, Any],
    *,
    claim_id: str,
    turn_id: str,
    exact_support_quote: str,
) -> dict[str, Any]:
    """Return a copied ledger with one exact quote bound for later validation."""

    claim_id = normalize_text(claim_id)
    turn_id = normalize_text(turn_id)
    if not claim_id or not turn_id:
        raise ClaimIRError("claim_id and turn_id must be non-empty")
    if not isinstance(exact_support_quote, str) or not exact_support_quote:
        raise ClaimIRError("exact_support_quote must be a non-empty exact string")
    ledger = _json_safe_copy(support_ledger, path="support_ledger")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ClaimIRError("support_ledger.entries must be an array")
    matched = False
    for entry in entries:
        if isinstance(entry, dict) and entry.get("claim_id") == claim_id:
            entry["turn_id"] = turn_id
            entry["exact_support_quote"] = exact_support_quote
            entry["verification_status"] = "pending_semantic_validation"
            matched = True
    if not matched:
        raise ClaimIRError(f"unknown support-ledger claim_id: {claim_id!r}")
    ledger["by_claim_id"] = {
        entry["claim_id"]: copy.deepcopy(entry)
        for entry in entries
        if isinstance(entry, dict) and normalize_text(entry.get("claim_id"))
    }
    ledger.pop("support_ledger_hash", None)
    ledger["support_ledger_hash"] = exact_json_sha256(ledger)
    return ledger


def compile_claim_ir(
    current_event_plan: Mapping[str, Any],
    *,
    current_alignment: Mapping[str, Any] | None = None,
    dialogue_turns: Any = None,
) -> dict[str, Any]:
    """Compile a complete, frozen Claim IR bundle from ``current_event_plan``."""

    plan_copy = _json_safe_copy(current_event_plan, path="current_event_plan")
    claims = extract_claims(
        current_event_plan,
        current_alignment=current_alignment,
        dialogue_turns=dialogue_turns,
    )
    frozen_claim_rows = [
        {"claim_id": claim["claim_id"], **_identity_payload(claim)}
        for claim in claims
    ]
    claim_bundle_hash = canonical_json_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "claims": frozen_claim_rows,
        }
    )
    source_plan_hash = exact_json_sha256(plan_copy)
    graph_snapshot_hash = normalize_text(current_event_plan.get("graph_snapshot_hash"))
    if not graph_snapshot_hash:
        graph_snapshot_hash = source_plan_hash
    support_ledger = build_support_ledger(
        claims,
        claim_bundle_hash=claim_bundle_hash,
        current_alignment=current_alignment,
        dialogue_turns=dialogue_turns,
    )
    validated_bundle_hash = canonical_json_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "claim_bundle_hash": claim_bundle_hash,
            "support_ledger_hash": support_ledger["support_ledger_hash"],
            "effective_qa_eligibility": {
                claim["claim_id"]: bool(claim["qa_eligible"])
                for claim in claims
            },
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_snapshot_hash": graph_snapshot_hash,
        "source_plan_hash": source_plan_hash,
        "source_graph_id": normalize_text(
            current_event_plan.get("graph_snapshot_id")
            or current_event_plan.get("graph_id")
        )
        or None,
        "source_event_count": len(current_event_plan.get("events") or []),
        "claims": claims,
        "claims_by_id": {claim["claim_id"]: copy.deepcopy(claim) for claim in claims},
        "claim_bundle_hash": claim_bundle_hash,
        "frozen_claims_hash": claim_bundle_hash,
        "frozen_graph_hash": claim_bundle_hash,
        "frozen_hash": claim_bundle_hash,
        "support_ledger_hash": support_ledger["support_ledger_hash"],
        "validated_bundle_hash": validated_bundle_hash,
        "support_ledger": support_ledger,
        "current_event_plan": plan_copy,
    }


def attach_claim_ir(
    current_event_plan: Mapping[str, Any],
    *,
    current_alignment: Mapping[str, Any] | None = None,
    dialogue_turns: Any = None,
) -> dict[str, Any]:
    """Return a plan copy with its Claim IR attached; never alter the source."""

    plan_copy = _json_safe_copy(current_event_plan, path="current_event_plan")
    plan_copy["claim_ir"] = compile_claim_ir(
        current_event_plan,
        current_alignment=current_alignment,
        dialogue_turns=dialogue_turns,
    )
    return plan_copy


# Explicit alias for callers that name the source structure in their API.
compile_current_event_plan_claims = compile_claim_ir


__all__ = [
    "ACTIVE_STATUSES",
    "CLAIM_SCHEMA_VERSION",
    "CLAIM_TYPES",
    "ClaimIRError",
    "EDGE_ROLES",
    "NODE_CLAIM_TYPE",
    "RELATION_CLAIM_TYPES",
    "SCHEMA_VERSION",
    "SUPPORT_LEDGER_SCHEMA_VERSION",
    "attach_claim_ir",
    "alignment_support_binding",
    "bind_support_evidence",
    "build_support_ledger",
    "canonical_json",
    "canonical_json_sha256",
    "compile_claim_ir",
    "compile_current_event_plan_claims",
    "extract_claims",
    "exact_json_sha256",
    "normalize_node_claim",
    "normalize_predicate",
    "normalize_text",
]
