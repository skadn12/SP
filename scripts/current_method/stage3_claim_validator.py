"""Independent validation for Stage3 claims, proofs, and support ledgers.

This module intentionally imports no claim compiler code.  It validates plain
JSON artifacts a second time, including the evidence text produced by a
naturalisation API.  The compiler is the producer of the frozen truth; this
module is an independently implemented consumer of that truth.

Canonical bundle shape::

    {
      "claims": [{...}],
      "proof": {...},                 # or "proofs": [{...}]
      "support_ledger": [{...}],      # {"entries": [...]} is also accepted
      "dialogue": {"turns": [{"turn_id": "D1:1", "text": "..."}]}
    }

``canonical_support_claim`` and a mapping-valued ``truth_signature`` use the
surface semantic fields ``subject``, ``predicate``, ``objects``, ``polarity``,
``status``, ``time``, and ``condition``.  Keeping these roles structured is
what makes direction, negation, time, and condition validation possible; a
bag-of-words or exact-string-only check is deliberately insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage3_claim_validator_v1"

CANONICAL_CLAIM_FIELDS = (
    "claim_id",
    "claim_type",
    "subject_event_id",
    "predicate",
    "object_event_ids",
    "edge_role",
    "qa_eligible",
    "polarity",
    "status",
    "condition",
    "scope",
    "source_event_ids",
    "canonical_support_claim",
    "truth_signature",
)
CANONICAL_PROOF_FIELDS = (
    "proof_id",
    "operation",
    "required_claim_ids",
    "reasoning_path_claim_ids",
    "gold",
    "answer_type",
    "derivation",
)
CANONICAL_LEDGER_FIELDS = ("claim_id", "turn_id", "exact_support_quote")

_RELATION_CLAIM_TYPES = frozenset(
    {
        "dependency",
        "membership",
        "visibility",
        "status_transition",
        "cause",
        "comparison",
        "relation",
    }
)
_ACTIVE_STATUSES = frozenset({"", "active", "current", "restored", "true"})

_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CONDITION_RE = re.compile(
    r"\b(?:if|provided\s+that|assuming\s+that|unless|as\s+long\s+as|only\s+if)\b",
    flags=re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|neither|nor|without|denied|refused|cannot|can't|"
    r"didn't|doesn't|isn't|wasn't|won't|wouldn't|hasn't|haven't|hadn't)\b",
    flags=re.IGNORECASE,
)
_OVERRIDE_RE = re.compile(
    r"\b(?:now|instead|replaced|changed|updated|superseded|overrode|overridden|"
    r"no\s+longer|cancel(?:ed|led)|revoked|restored)\b",
    flags=re.IGNORECASE,
)

_STATUS_MARKERS: dict[str, tuple[str, ...]] = {
    "active": ("active", "current", "now", "in effect"),
    "current": ("active", "current", "now", "in effect"),
    "restored": ("restored", "reinstated", "back in effect", "active again"),
    "canceled": (
        "canceled",
        "cancelled",
        "called off",
        "withdrawn",
        "did not take effect",
        "before taking effect",
    ),
    "outdated": (
        "outdated",
        "obsolete",
        "superseded",
        "no longer applies",
        "no longer applicable",
    ),
    "expired": ("expired", "lapsed", "no longer valid"),
    "false": ("false", "incorrect", "not true", "denied"),
    "belief": (
        "believes",
        "believed",
        "belief",
        "thinks",
        "might",
        "may be",
        "uncertain",
        "unsure",
    ),
    "report": ("reported", "report says", "alleged", "unconfirmed"),
    "scope_limited": ("limited", "restricted", "only"),
    "delayed_not_active": ("delayed", "not active", "not yet", "pending"),
}
_NON_ACTIVE_STATUS_NAMES = frozenset(
    {
        "canceled",
        "outdated",
        "expired",
        "false",
        "belief",
        "report",
        "scope_limited",
        "delayed_not_active",
        "superseded",
        "inactive",
    }
)

# Each family contains words that can express the same relation in a natural
# quote.  These are intentionally small and auditable, not an embedding-based
# similarity escape hatch.
_PREDICATE_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"depend", "require", "rely", "prerequisite", "contingent"}),
    frozenset({"member", "belong", "include", "part", "membership"}),
    frozenset({"visible", "visibility", "access", "see", "share", "view"}),
    frozenset({"cause", "because", "lead", "result", "due", "trigger"}),
    frozenset({"compare", "more", "less", "greater", "fewer", "higher", "lower", "than"}),
    frozenset({"grant", "allow", "permit", "permission", "access"}),
    frozenset({"own", "have", "possess", "belong"}),
    frozenset({"cancel", "revoke", "withdraw", "expire"}),
    frozenset({"change", "update", "replace", "become", "transition", "restore"}),
)

_NUMBER_WORDS = {
    0: ("zero", "zeroth"),
    1: ("one", "first"),
    2: ("two", "second"),
    3: ("three", "third"),
    4: ("four", "fourth"),
    5: ("five", "fifth"),
    6: ("six", "sixth"),
    7: ("seven", "seventh"),
    8: ("eight", "eighth"),
    9: ("nine", "ninth"),
    10: ("ten", "tenth"),
    11: ("eleven", "eleventh"),
    12: ("twelve", "twelfth"),
    13: ("thirteen", "thirteenth"),
    14: ("fourteen", "fourteenth"),
    15: ("fifteen", "fifteenth"),
    16: ("sixteen", "sixteenth"),
    17: ("seventeen", "seventeenth"),
    18: ("eighteen", "eighteenth"),
    19: ("nineteen", "nineteenth"),
    20: ("twenty", "twentieth"),
}


def _issue(code: str, path: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "path": path, "message": message}
    if details:
        result["details"] = details
    return result


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalized_text(value: Any) -> str:
    return " ".join(_WORD_RE.findall(str(value).casefold()))


def _tokens(value: Any) -> list[str]:
    return [_stem(token) for token in _WORD_RE.findall(str(value).casefold())]


def _stem(token: str) -> str:
    """A deliberately conservative stemmer for quote/claim surface checks."""

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(token) > 4 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return _normalized_text(left) == _normalized_text(right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_json_equal(left[key], right[key]) for key in left)
    return type(left) is type(right) and left == right


def _string_list(value: Any, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_nonempty_string(item) for item in value)
        and len(value) == len(set(value))
    )


def _claim_kind(claim: Mapping[str, Any]) -> str:
    claim_type = str(claim.get("claim_type") or "").strip().casefold()
    if claim_type in {"node", "relation"}:
        return claim_type
    alias = str(claim.get("claim_kind") or "").strip().casefold()
    if alias in {"node", "relation"}:
        return alias
    if claim_type in _RELATION_CLAIM_TYPES:
        return "relation"
    return ""


def _claim_time(claim: Mapping[str, Any]) -> Any:
    for key in ("global_time_index", "time_index", "time"):
        if key in claim:
            return claim[key]
    return None


def _canonical_support(claim: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = claim.get("canonical_support_claim")
    return value if isinstance(value, Mapping) else None


def _surface_semantics(claim: Mapping[str, Any]) -> dict[str, Any]:
    support = _canonical_support(claim) or {}
    objects = support.get("objects")
    if objects is None and "object" in support:
        objects = support["object"]
    if objects is None and "value" in support:
        objects = support["value"]
    if objects is None:
        objects = claim.get("objects", claim.get("object", []))
    if not isinstance(objects, list):
        objects = [objects]
    objects = [value for value in objects if value is not None and value != ""]

    subject = support.get("subject", claim.get("subject"))
    predicate = support.get("predicate", claim.get("predicate"))
    polarity = support.get("polarity", claim.get("polarity"))
    status = support.get("status", claim.get("status"))
    time_value = support.get("time")
    if time_value is None:
        for key in ("global_time_index", "time_index"):
            if key in support:
                time_value = support[key]
                break
    if time_value is None:
        time_value = _claim_time(claim)
    condition = support.get("condition", claim.get("condition"))
    scope = support.get("scope", claim.get("scope"))
    return {
        "subject": subject,
        "predicate": predicate,
        "objects": objects,
        "polarity": polarity,
        "status": status,
        "time": time_value,
        "condition": condition,
        "scope": scope,
    }


def _normalized_polarity(value: Any) -> str:
    if value is True:
        return "positive"
    if value is False:
        return "negative"
    normalized = _normalized_text(value)
    if normalized in {"positive", "affirmed", "true", "1"}:
        return "positive"
    if normalized in {"negative", "negated", "false", "0", "minus 1"}:
        return "negative"
    return normalized


def _normalized_status(value: Any) -> str:
    """Normalize a status as a vocabulary symbol, preserving word boundaries."""

    return re.sub(r"[\W_]+", "_", str(value or "").casefold()).strip("_")


def _semantic_signature(semantics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subject": semantics.get("subject"),
        "predicate": semantics.get("predicate"),
        "objects": list(semantics.get("objects") or []),
        "polarity": _normalized_polarity(semantics.get("polarity")),
        "status": _normalized_text(semantics.get("status")),
        "time": semantics.get("time"),
        "condition": semantics.get("condition"),
        "scope": semantics.get("scope"),
    }


def _signature_digest(signature: Mapping[str, Any]) -> str:
    payload = json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    objects = value.get("objects")
    if objects is None and "object" in value:
        objects = value["object"]
    if objects is None and "value" in value:
        objects = value["value"]
    if not isinstance(objects, list):
        objects = [] if objects is None else [objects]
    time_value = value.get("time")
    if time_value is None:
        time_value = value.get("global_time_index", value.get("time_index"))
    return _semantic_signature(
        {
            "subject": value.get("subject"),
            "predicate": value.get("predicate"),
            "objects": objects,
            "polarity": value.get("polarity"),
            "status": value.get("status"),
            "time": time_value,
            "condition": value.get("condition"),
            "scope": value.get("scope"),
        }
    )


def _signature_matches(value: Any, expected: Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        actual = _mapping_signature(value)
        # ``scope`` was added after the original seven-field signature.  A
        # missing scope is compatible only when the claim itself is unscoped.
        return _json_equal(actual, expected)
    if _nonempty_string(value):
        digest = _signature_digest(expected)
        return value.strip().casefold() in {digest, f"sha256:{digest}"}
    return False


def validate_claim(claim: Any, *, path: str = "$.claims[0]") -> list[dict[str, Any]]:
    """Validate one canonical claim without mutating it."""

    if not isinstance(claim, Mapping):
        return [_issue("invalid_claim_type", path, "A claim must be a JSON object.")]

    issues: list[dict[str, Any]] = []
    for field in CANONICAL_CLAIM_FIELDS:
        if field not in claim:
            # Transitional Claim IR producers may use claim_kind while their
            # claim_type carries the semantic subtype.  Keep reading those
            # artifacts but still require every other canonical field.
            if field == "claim_type" and "claim_kind" in claim:
                continue
            issues.append(
                _issue(
                    "missing_claim_field",
                    f"{path}.{field}",
                    f"Required claim field '{field}' is missing.",
                    field=field,
                )
            )
    if not any(key in claim for key in ("time_index", "global_time_index", "time")):
        issues.append(
            _issue(
                "missing_claim_time",
                path,
                "A claim must carry time_index or global_time_index.",
            )
        )

    if not _nonempty_string(claim.get("claim_id")):
        issues.append(_issue("invalid_claim_id", f"{path}.claim_id", "claim_id must be a non-empty string."))
    if not _claim_kind(claim):
        issues.append(
            _issue(
                "unsupported_claim_type",
                f"{path}.claim_type",
                "claim_type must identify a node or relation claim.",
                claim_type=claim.get("claim_type"),
            )
        )
    if not _nonempty_string(claim.get("subject_event_id")):
        issues.append(
            _issue(
                "invalid_subject_event_id",
                f"{path}.subject_event_id",
                "subject_event_id must be a non-empty string.",
            )
        )
    if not _nonempty_string(claim.get("predicate")):
        issues.append(_issue("invalid_predicate", f"{path}.predicate", "predicate must be a non-empty string."))

    object_ids = claim.get("object_event_ids")
    allow_empty_objects = _claim_kind(claim) == "node"
    if not _string_list(object_ids, allow_empty=allow_empty_objects):
        issues.append(
            _issue(
                "invalid_object_event_ids",
                f"{path}.object_event_ids",
                "object_event_ids must be a duplicate-free array of event identifiers.",
            )
        )
    edge_role = claim.get("edge_role")
    if edge_role not in {"control", "semantic"}:
        issues.append(
            _issue(
                "invalid_edge_role",
                f"{path}.edge_role",
                "edge_role must be 'control' or 'semantic'.",
            )
        )
    if not isinstance(claim.get("qa_eligible"), bool):
        issues.append(_issue("invalid_qa_eligible", f"{path}.qa_eligible", "qa_eligible must be a JSON boolean."))
    elif edge_role == "control" and claim.get("qa_eligible") is True:
        issues.append(
            _issue(
                "control_claim_qa_eligible",
                f"{path}.qa_eligible",
                "A generation-control claim cannot be QA eligible.",
            )
        )
    if _normalized_polarity(claim.get("polarity")) not in {"positive", "negative"}:
        issues.append(
            _issue(
                "invalid_polarity",
                f"{path}.polarity",
                "polarity must be positive or negative.",
            )
        )
    status = claim.get("status")
    if status is not None and not _nonempty_string(status):
        issues.append(_issue("invalid_status", f"{path}.status", "status must be null or a non-empty string."))

    for time_key in ("time_index", "global_time_index"):
        if time_key in claim and (
            not _is_int(claim[time_key]) or claim[time_key] < 0
        ):
            issues.append(
                _issue(
                    f"invalid_{time_key}",
                    f"{path}.{time_key}",
                    f"{time_key} must be a non-negative integer.",
                )
            )

    source_ids = claim.get("source_event_ids")
    if not _string_list(source_ids):
        issues.append(
            _issue(
                "invalid_source_event_ids",
                f"{path}.source_event_ids",
                "source_event_ids must be a non-empty duplicate-free array.",
            )
        )
    elif _nonempty_string(claim.get("subject_event_id")) and claim["subject_event_id"] not in source_ids:
        issues.append(
            _issue(
                "subject_event_not_in_sources",
                f"{path}.source_event_ids",
                "subject_event_id must be represented in source_event_ids.",
            )
        )
    if isinstance(source_ids, list) and isinstance(object_ids, list):
        missing_object_sources = [item for item in object_ids if item not in source_ids]
        if missing_object_sources:
            issues.append(
                _issue(
                    "object_event_not_in_sources",
                    f"{path}.source_event_ids",
                    "Every object_event_id must be represented in source_event_ids.",
                    object_event_ids=missing_object_sources,
                )
            )

    support = claim.get("canonical_support_claim")
    if not isinstance(support, Mapping):
        issues.append(
            _issue(
                "invalid_canonical_support_claim",
                f"{path}.canonical_support_claim",
                "canonical_support_claim must be a structured semantic JSON object.",
            )
        )
        return issues

    for field in ("subject", "predicate", "objects", "polarity", "status", "time", "condition"):
        if field not in support:
            issues.append(
                _issue(
                    "missing_support_semantic_field",
                    f"{path}.canonical_support_claim.{field}",
                    f"Structured support semantics are missing '{field}'.",
                    field=field,
                )
            )
    semantics = _surface_semantics(claim)
    if not _nonempty_string(semantics["subject"]):
        issues.append(
            _issue(
                "invalid_support_subject",
                f"{path}.canonical_support_claim.subject",
                "The surface subject must be a non-empty string.",
            )
        )
    if not _nonempty_string(semantics["predicate"]):
        issues.append(
            _issue(
                "invalid_support_predicate",
                f"{path}.canonical_support_claim.predicate",
                "The surface predicate must be a non-empty string.",
            )
        )
    if not isinstance(support.get("objects"), list) or not semantics["objects"]:
        issues.append(
            _issue(
                "invalid_support_objects",
                f"{path}.canonical_support_claim.objects",
                "Surface objects must be a non-empty JSON array.",
            )
        )

    expected_signature = _semantic_signature(semantics)
    if not _signature_matches(claim.get("truth_signature"), expected_signature):
        issues.append(
            _issue(
                "truth_signature_mismatch",
                f"{path}.truth_signature",
                "truth_signature does not independently match canonical_support_claim.",
                expected_sha256=_signature_digest(expected_signature),
            )
        )

    top_checks = {
        "predicate": claim.get("predicate"),
        "polarity": _normalized_polarity(claim.get("polarity")),
        "status": _normalized_text(claim.get("status")),
        "condition": claim.get("condition"),
        "scope": claim.get("scope"),
    }
    support_checks = {
        "predicate": semantics["predicate"],
        "polarity": _normalized_polarity(semantics["polarity"]),
        "status": _normalized_text(semantics["status"]),
        "condition": semantics["condition"],
        "scope": semantics["scope"],
    }
    for field in top_checks:
        if field == "scope" and "scope" not in support:
            continue
        if not _json_equal(top_checks[field], support_checks[field]):
            issues.append(
                _issue(
                    "claim_support_semantic_mismatch",
                    f"{path}.canonical_support_claim.{field}",
                    f"Top-level claim {field} disagrees with support semantics.",
                    field=field,
                )
            )
    support_time = semantics["time"]
    if isinstance(support_time, Mapping):
        for time_field in ("time_index", "global_time_index"):
            if (
                time_field in claim
                and time_field in support_time
                and not _json_equal(claim[time_field], support_time[time_field])
            ):
                issues.append(
                    _issue(
                        "claim_support_semantic_mismatch",
                        f"{path}.canonical_support_claim.time.{time_field}",
                        f"Top-level {time_field} disagrees with support semantics.",
                        field=time_field,
                    )
                )
    else:
        top_time = _claim_time(claim)
        if top_time is not None and not _json_equal(top_time, support_time):
            issues.append(
                _issue(
                    "claim_support_semantic_mismatch",
                    f"{path}.canonical_support_claim.time",
                    "Top-level claim time disagrees with support semantics.",
                    field="time",
                )
            )
    return issues


def _extract_answer_claim_ids(derivation: Any) -> list[str]:
    ids: list[str] = []
    singular_keys = {
        "answer_claim_id",
        "gold_claim_id",
        "selected_claim_id",
        "current_claim_id",
    }
    plural_keys = {
        "answer_claim_ids",
        "gold_claim_ids",
        "selected_claim_ids",
        "current_claim_ids",
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in singular_keys and _nonempty_string(child):
                    ids.append(child)
                elif key in plural_keys and isinstance(child, list):
                    ids.extend(item for item in child if _nonempty_string(item))
                elif isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(derivation)
    return list(dict.fromkeys(ids))


def _has_explicit_answer_claim_ids(derivation: Any) -> bool:
    """Distinguish an explicit empty derived answer from a missing witness."""

    if isinstance(derivation, Mapping):
        if "answer_claim_ids" in derivation:
            return isinstance(derivation["answer_claim_ids"], list)
        return any(
            _has_explicit_answer_claim_ids(child)
            for child in derivation.values()
            if isinstance(child, (Mapping, list))
        )
    if isinstance(derivation, list):
        return any(_has_explicit_answer_claim_ids(child) for child in derivation)
    return False


def _validate_claim_id_list_field(
    proof: Mapping[str, Any],
    field: str,
    by_id: Mapping[str, Mapping[str, Any]],
    required_set: set[str],
    *,
    path: str,
    allow_empty: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    if field not in proof:
        return [], []
    value = proof.get(field)
    if not _string_list(value, allow_empty=allow_empty):
        return [], [
            _issue(
                f"invalid_{field}",
                f"{path}.{field}",
                f"{field} must be a duplicate-free array of claim identifiers.",
            )
        ]
    issues: list[dict[str, Any]] = []
    for index, claim_id in enumerate(value):
        item_path = f"{path}.{field}[{index}]"
        if claim_id not in by_id:
            issues.append(
                _issue(
                    "proof_unknown_claim",
                    item_path,
                    "The proof references a claim that is not frozen in this bundle.",
                    claim_id=claim_id,
                )
            )
        if claim_id not in required_set:
            issues.append(
                _issue(
                    f"{field}_claim_not_required",
                    item_path,
                    f"Every {field} claim must also occur in required_claim_ids.",
                    claim_id=claim_id,
                )
            )
    return list(value), issues


def _numeric_claim_value(claim: Mapping[str, Any]) -> int | float | None:
    support = _canonical_support(claim) or {}
    candidates: list[Any] = []
    for key in ("numeric_value", "answer_value", "value"):
        if key in support:
            candidates.append(support[key])
    candidates.extend(_answer_values(claim))
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)):
            return candidate
        if isinstance(candidate, str):
            normalized = candidate.strip().replace(",", "")
            try:
                parsed = float(normalized)
            except ValueError:
                continue
            return int(parsed) if parsed.is_integer() else parsed
    return None


def _numeric_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _validate_derived_proof(
    proof: Mapping[str, Any],
    derivation: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    required_set: set[str],
    *,
    path: str,
) -> tuple[bool, list[dict[str, Any]]]:
    """Independently recompute supported claim-to-value derivation rules."""

    if not _has_explicit_answer_claim_ids(derivation):
        return False, []
    explicit_answer_ids = derivation.get("answer_claim_ids")
    if explicit_answer_ids != []:
        return False, []

    rule = _normalized_text(derivation.get("rule"))
    if not rule:
        return True, [
            _issue(
                "derived_proof_missing_rule",
                f"{path}.derivation.rule",
                "An empty answer_claim_ids array denotes a derived answer and requires a rule.",
            )
        ]

    operand_ids = proof.get("gold_operand_claim_ids")
    if operand_ids is None:
        operand_ids = derivation.get("operand_claim_ids")
    if not _string_list(operand_ids):
        return True, [
            _issue(
                "invalid_gold_operand_claim_ids",
                f"{path}.gold_operand_claim_ids",
                "A derived proof requires duplicate-free gold_operand_claim_ids.",
            )
        ]

    issues: list[dict[str, Any]] = []
    for index, claim_id in enumerate(operand_ids):
        item_path = f"{path}.gold_operand_claim_ids[{index}]"
        if claim_id not in by_id:
            issues.append(
                _issue(
                    "proof_unknown_claim",
                    item_path,
                    "A derived operand is not a frozen claim.",
                    claim_id=claim_id,
                )
            )
        if claim_id not in required_set:
            issues.append(
                _issue(
                    "gold_operand_claim_not_required",
                    item_path,
                    "Every derived operand must occur in required_claim_ids.",
                    claim_id=claim_id,
                )
            )
    if issues:
        return True, issues

    if rule == "count distinct":
        membership_rows: list[tuple[str, Any, str, Any]] = []
        for claim_id in operand_ids:
            claim = by_id[claim_id]
            support = _canonical_support(claim) or {}
            included = claim.get("included", support.get("included"))
            set_id = claim.get("set_id", support.get("set_id"))
            condition = support.get("condition", claim.get("condition"))
            member = support.get("subject")
            if (
                _normalized_text(claim.get("claim_type")) != "membership"
                or not isinstance(included, bool)
                or not _nonempty_string(set_id)
                or member is None
            ):
                issues.append(
                    _issue(
                        "count_invalid_membership_operand",
                        f"{path}.gold_operand_claim_ids",
                        "count_distinct operands must be typed membership claims with boolean inclusion.",
                        claim_id=claim_id,
                    )
                )
                continue
            membership_rows.append((claim_id, included, str(set_id), member))
        if issues:
            return True, issues
        set_ids = {_normalized_text(row[2]) for row in membership_rows}
        conditions = {
            json.dumps(
                _normalized_answer(
                    (_canonical_support(by_id[row[0]]) or {}).get(
                        "condition", by_id[row[0]].get("condition")
                    )
                ),
                sort_keys=True,
                ensure_ascii=False,
            )
            for row in membership_rows
        }
        members = {
            json.dumps(
                _normalized_answer(row[3]),
                sort_keys=True,
                ensure_ascii=False,
            )
            for row in membership_rows
        }
        if len(set_ids) != 1 or len(conditions) != 1 or len(members) != len(membership_rows):
            return True, [
                _issue(
                    "count_membership_domain_mismatch",
                    f"{path}.gold_operand_claim_ids",
                    "count_distinct operands must be unique members of one set under one condition.",
                )
            ]
        expected_included_ids = [
            claim_id
            for claim_id, included, _set_id, _member in membership_rows
            if included
        ]
        declared_included_ids = derivation.get("included_claim_ids")
        if (
            not _string_list(declared_included_ids, allow_empty=True)
            or set(declared_included_ids) != set(expected_included_ids)
            or len(declared_included_ids) != len(expected_included_ids)
        ):
            issues.append(
                _issue(
                    "count_included_claim_ids_mismatch",
                    f"{path}.derivation.included_claim_ids",
                    "included_claim_ids disagree with independently read membership polarity.",
                    expected_claim_ids=expected_included_ids,
                )
            )
        recomputed = len(expected_included_ids)
        if not _numeric_equal(derivation.get("computed_value"), recomputed):
            issues.append(
                _issue(
                    "derivation_computed_value_mismatch",
                    f"{path}.derivation.computed_value",
                    "computed_value disagrees with the independently counted memberships.",
                    expected=recomputed,
                )
            )
        if not _numeric_equal(_proof_semantic_gold(proof), recomputed):
            issues.append(
                _issue(
                    "proof_gold_mismatch",
                    f"{path}.gold",
                    "gold disagrees with the independently counted memberships.",
                    expected=recomputed,
                    operand_claim_ids=operand_ids,
                )
            )
        return True, issues

    if rule != "absolute difference":
        return True, [
            _issue(
                "unsupported_derivation_rule",
                f"{path}.derivation.rule",
                "The independent validator does not implement this derivation rule.",
                rule=derivation.get("rule"),
                supported=["absolute_difference", "count_distinct"],
            )
        ]
    if len(operand_ids) != 2:
        return True, [
            _issue(
                "absolute_difference_arity",
                f"{path}.gold_operand_claim_ids",
                "absolute_difference requires exactly two operand claims.",
                operand_count=len(operand_ids),
            )
        ]

    values = [_numeric_claim_value(by_id[claim_id]) for claim_id in operand_ids]
    if any(value is None for value in values):
        return True, [
            _issue(
                "absolute_difference_non_numeric_operand",
                f"{path}.gold_operand_claim_ids",
                "Both absolute_difference operand claims must expose one numeric value.",
                operand_claim_ids=operand_ids,
            )
        ]
    recomputed = abs(float(values[0]) - float(values[1]))  # type: ignore[arg-type]
    if recomputed.is_integer():
        recomputed = int(recomputed)
    if not _numeric_equal(derivation.get("computed_value"), recomputed):
        issues.append(
            _issue(
                "derivation_computed_value_mismatch",
                f"{path}.derivation.computed_value",
                "computed_value disagrees with the independently recomputed operands.",
                expected=recomputed,
            )
        )
    if not _numeric_equal(_proof_semantic_gold(proof), recomputed):
        issues.append(
            _issue(
                "proof_gold_mismatch",
                f"{path}.gold",
                "gold disagrees with the independently recomputed derived value.",
                expected=recomputed,
                operand_claim_ids=operand_ids,
            )
        )
    return True, issues


def _validate_cause_proof(
    proof: Mapping[str, Any],
    derivation: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    *,
    path: str,
) -> list[dict[str, Any]]:
    if _normalized_text(proof.get("operation")) != "cause consequence":
        return []
    cause_ids = derivation.get("cause_claim_ids")
    target_ids = derivation.get("target_claim_ids")
    if not _string_list(cause_ids) or len(cause_ids) != 1:
        return [
            _issue(
                "cause_proof_missing_relation",
                f"{path}.derivation.cause_claim_ids",
                "A cause proof requires exactly one semantic cause relation claim.",
            )
        ]
    if not _string_list(target_ids) or len(target_ids) != 1:
        return [
            _issue(
                "cause_proof_missing_effect",
                f"{path}.derivation.target_claim_ids",
                "A cause proof requires exactly one effect node claim.",
            )
        ]
    cause_claim = by_id.get(cause_ids[0])
    effect_claim = by_id.get(target_ids[0])
    if (
        not isinstance(cause_claim, Mapping)
        or _normalized_text(cause_claim.get("claim_type")) != "cause"
        or cause_claim.get("edge_role") != "semantic"
    ):
        return [
            _issue(
                "cause_proof_invalid_relation",
                f"{path}.derivation.cause_claim_ids",
                "The cause witness must be one semantic cause relation.",
            )
        ]
    if not isinstance(effect_claim, Mapping) or _claim_kind(effect_claim) != "node":
        return [
            _issue(
                "cause_proof_invalid_effect",
                f"{path}.derivation.target_claim_ids",
                "The effect answer witness must be one node claim.",
            )
        ]
    cause_event_id = str(cause_claim.get("subject_event_id") or "")
    object_ids = cause_claim.get("object_event_ids")
    effect_event_id = str(effect_claim.get("subject_event_id") or "")
    issues: list[dict[str, Any]] = []
    if object_ids != [effect_event_id]:
        issues.append(
            _issue(
                "cause_proof_direction_mismatch",
                f"{path}.derivation",
                "The semantic cause relation must point to the answer node event.",
                relation_object_event_ids=object_ids,
                effect_event_id=effect_event_id,
            )
        )
    if str(derivation.get("cause_event_instance_id") or "") != cause_event_id:
        issues.append(
            _issue(
                "cause_proof_source_ref_mismatch",
                f"{path}.derivation.cause_event_instance_id",
                "The frozen cause event ref disagrees with the cause relation subject.",
                expected=cause_event_id,
            )
        )
    if str(derivation.get("effect_event_instance_id") or "") != effect_event_id:
        issues.append(
            _issue(
                "cause_proof_effect_ref_mismatch",
                f"{path}.derivation.effect_event_instance_id",
                "The frozen effect event ref disagrees with the answer node.",
                expected=effect_event_id,
            )
        )
    return issues


def validate_proof(
    proof: Any,
    claims: Sequence[Mapping[str, Any]],
    *,
    path: str = "$.proof",
) -> list[dict[str, Any]]:
    """Validate proof references and its QA-eligible reasoning path."""

    if not isinstance(proof, Mapping):
        return [_issue("invalid_proof_type", path, "A proof must be a JSON object.")]
    issues: list[dict[str, Any]] = []
    for field in CANONICAL_PROOF_FIELDS:
        if field not in proof:
            issues.append(
                _issue(
                    "missing_proof_field",
                    f"{path}.{field}",
                    f"Required proof field '{field}' is missing.",
                    field=field,
                )
            )
    for field in ("proof_id", "operation", "answer_type"):
        if not _nonempty_string(proof.get(field)):
            issues.append(
                _issue(
                    f"invalid_{field}",
                    f"{path}.{field}",
                    f"{field} must be a non-empty string.",
                )
            )

    required = proof.get("required_claim_ids")
    reasoning = proof.get("reasoning_path_claim_ids")
    if not _string_list(required):
        issues.append(
            _issue(
                "invalid_required_claim_ids",
                f"{path}.required_claim_ids",
                "required_claim_ids must be a non-empty duplicate-free string array.",
            )
        )
        required = []
    if not _string_list(reasoning):
        issues.append(
            _issue(
                "invalid_reasoning_path_claim_ids",
                f"{path}.reasoning_path_claim_ids",
                "reasoning_path_claim_ids must be a non-empty duplicate-free string array.",
            )
        )
        reasoning = []

    by_id = {
        claim.get("claim_id"): claim
        for claim in claims
        if isinstance(claim, Mapping) and _nonempty_string(claim.get("claim_id"))
    }
    for field, values in (
        ("required_claim_ids", required),
        ("reasoning_path_claim_ids", reasoning),
    ):
        for index, claim_id in enumerate(values):
            if claim_id not in by_id:
                issues.append(
                    _issue(
                        "proof_unknown_claim",
                        f"{path}.{field}[{index}]",
                        "The proof references a claim that is not frozen in this bundle.",
                        claim_id=claim_id,
                    )
                )

    required_set = set(required)
    for optional_field, allow_empty in (
        # Direct-answer proofs canonically carry an empty operand array; the
        # derived-proof branch below requires it to be non-empty as needed.
        ("gold_operand_claim_ids", True),
        ("dependency_closure_claim_ids", True),
    ):
        _, optional_issues = _validate_claim_id_list_field(
            proof,
            optional_field,
            by_id,
            required_set,
            path=path,
            allow_empty=allow_empty,
        )
        issues.extend(optional_issues)
    for index, claim_id in enumerate(reasoning):
        claim = by_id.get(claim_id)
        item_path = f"{path}.reasoning_path_claim_ids[{index}]"
        if claim_id not in required_set:
            issues.append(
                _issue(
                    "reasoning_claim_not_required",
                    item_path,
                    "Every reasoning-path claim must also be required by the proof.",
                    claim_id=claim_id,
                )
            )
        if claim is None:
            continue
        if claim.get("edge_role") != "semantic":
            issues.append(
                _issue(
                    "reasoning_path_control_claim",
                    item_path,
                    "Generation-control claims cannot enter a QA reasoning path.",
                    claim_id=claim_id,
                )
            )
        if claim.get("qa_eligible") is not True:
            issues.append(
                _issue(
                    "reasoning_path_not_qa_eligible",
                    item_path,
                    "Only explicitly QA-eligible claims may enter a reasoning path.",
                    claim_id=claim_id,
                )
            )

    if proof.get("gold") is None:
        issues.append(_issue("missing_proof_gold", f"{path}.gold", "gold must not be null."))
    derivation = proof.get("derivation")
    if not isinstance(derivation, (Mapping, list)) or not derivation:
        issues.append(
            _issue(
                "invalid_derivation",
                f"{path}.derivation",
                "derivation must be a non-empty JSON object or array.",
            )
        )
        return issues

    answer_ids = _extract_answer_claim_ids(derivation)
    for index, claim_id in enumerate(answer_ids):
        if claim_id not in by_id:
            issues.append(
                _issue(
                    "derivation_unknown_answer_claim",
                    f"{path}.derivation",
                    "The derivation's answer witness is not a frozen claim.",
                    claim_id=claim_id,
                    answer_index=index,
                )
            )
        elif claim_id not in required_set:
            issues.append(
                _issue(
                    "answer_claim_not_required",
                    f"{path}.derivation",
                    "Every answer witness must be present in required_claim_ids.",
                    claim_id=claim_id,
                )
            )

    if isinstance(derivation, Mapping):
        derived, derived_issues = _validate_derived_proof(
            proof,
            derivation,
            by_id,
            required_set,
            path=path,
        )
        issues.extend(derived_issues)
        issues.extend(
            _validate_cause_proof(
                proof,
                derivation,
                by_id,
                path=path,
            )
        )
    else:
        derived = False

    if not answer_ids and not derived:
        inferred = _infer_answer_claim_ids(_proof_semantic_gold(proof), required, by_id)
        if len(inferred) == 1:
            answer_ids = inferred
        else:
            issues.append(
                _issue(
                    "proof_answer_witness_not_unique",
                    f"{path}.derivation",
                    "derivation must identify answer_claim_ids unless gold uniquely identifies one required claim.",
                    inferred_claim_ids=inferred,
                )
            )

    answer_claims = [by_id[claim_id] for claim_id in answer_ids if claim_id in by_id]
    if answer_claims and not _gold_matches_answer_claims(
        _proof_semantic_gold(proof), proof.get("answer_type"), answer_claims
    ):
        issues.append(
            _issue(
                "proof_gold_mismatch",
                f"{path}.gold",
                "gold cannot be independently derived from the declared answer claims.",
                answer_claim_ids=answer_ids,
            )
        )
    return issues


def _answer_values(claim: Mapping[str, Any]) -> list[Any]:
    support = _canonical_support(claim) or {}
    if "answer" in support:
        answer = support["answer"]
        return answer if isinstance(answer, list) else [answer]
    if "answer_value" in support:
        answer = support["answer_value"]
        return answer if isinstance(answer, list) else [answer]
    return list(_surface_semantics(claim)["objects"])


def _proof_semantic_gold(proof: Mapping[str, Any]) -> Any:
    """Return semantic gold, not MCQ option-label gold, when both are frozen."""

    value = proof.get("gold_answer_text")
    return value if value is not None and value != "" else proof.get("gold")


def _normalized_answer(value: Any) -> Any:
    if isinstance(value, str):
        return _normalized_text(value)
    if isinstance(value, list):
        return [_normalized_answer(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalized_answer(item) for key, item in sorted(value.items())}
    return value


def _gold_matches_answer_claims(
    gold: Any,
    answer_type: Any,
    claims: Sequence[Mapping[str, Any]],
) -> bool:
    kind = _normalized_text(answer_type)
    values = [item for claim in claims for item in _answer_values(claim)]
    if kind in {"count", "integer", "number"}:
        return _normalized_answer(gold) == len(
            {json.dumps(_normalized_answer(item), sort_keys=True, ensure_ascii=False) for item in values}
        )
    if kind in {"boolean", "bool", "yes no", "yesno"}:
        expected = all(
            _normalized_polarity(_surface_semantics(claim)["polarity"]) == "positive"
            for claim in claims
        )
        if isinstance(gold, bool):
            return gold is expected
        normalized_gold = _normalized_text(gold)
        return normalized_gold in ({"yes", "true"} if expected else {"no", "false"})
    if len(values) == 1:
        return _answer_equal(gold, values[0])
    return _json_equal(_normalized_answer(gold), _normalized_answer(values))


def _answer_equal(left: Any, right: Any) -> bool:
    """Compare semantic answers while retaining entity/string distinctions.

    Numeric claim values are serialized into ``gold_answer_text`` by the MCQ
    compiler, so ``25`` and ``"25"`` are equivalent.  Two arbitrary strings
    are never numerically coerced (for example, entity id ``"001"`` remains a
    string).
    """

    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _answer_equal(a, b) for a, b in zip(left, right)
        )
    left_numeric = isinstance(left, (int, float)) and not isinstance(left, bool)
    right_numeric = isinstance(right, (int, float)) and not isinstance(right, bool)
    if left_numeric or right_numeric:
        return _numeric_equal(left, right)
    return _json_equal(_normalized_answer(left), _normalized_answer(right))


def _infer_answer_claim_ids(
    gold: Any,
    required: Sequence[str],
    by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    matches: list[str] = []
    for claim_id in required:
        claim = by_id.get(claim_id)
        if claim is not None and _gold_matches_answer_claims(gold, "value", [claim]):
            matches.append(claim_id)
    return matches


def _flatten_surface_values(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        preferred = [
            value[key]
            for key in ("label", "text", "value", "name", "predicate", "object")
            if key in value
        ]
        if preferred:
            return [item for child in preferred for item in _flatten_surface_values(child)]
        return [item for child in value.values() for item in _flatten_surface_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_surface_values(child)]
    return [str(value)]


def _surface_present(text: str, value: Any) -> bool:
    expected_values = _flatten_surface_values(value)
    if not expected_values:
        return True
    normalized_quote = _normalized_text(text)
    quote_tokens = set(_tokens(text))
    for expected in expected_values:
        normalized_expected = _normalized_text(expected)
        if normalized_expected and normalized_expected in normalized_quote:
            continue
        expected_tokens = set(_tokens(expected))
        if not expected_tokens or not expected_tokens <= quote_tokens:
            return False
    return True


def _predicate_terms(predicate: Any) -> set[str]:
    base = set(_tokens(predicate))
    expanded = set(base)
    for family in _PREDICATE_FAMILIES:
        stemmed_family = {_stem(token) for token in family}
        if base & stemmed_family:
            expanded.update(stemmed_family)
    return expanded


def _predicate_present(text: str, predicate: Any) -> bool:
    quote_tokens = set(_tokens(text))
    predicate_tokens = set(_tokens(predicate))
    if predicate_tokens and predicate_tokens <= quote_tokens:
        return True
    terms = _predicate_terms(predicate)
    return bool(terms & quote_tokens)


def _first_token_position(text: str, value: Any, *, terms: set[str] | None = None) -> int | None:
    quote_tokens = _tokens(text)
    if terms is None:
        value_tokens = _tokens(value)
        if value_tokens:
            # Locate the complete surface before falling back to one token.
            # Otherwise an object such as "the Midwest" binds to an unrelated
            # sentence-initial "the" and creates a false direction reversal.
            width = len(value_tokens)
            for index in range(max(0, len(quote_tokens) - width + 1)):
                if quote_tokens[index : index + width] == value_tokens:
                    return index
        terms = {value_tokens[0]} if value_tokens else set()
    for index, token in enumerate(quote_tokens):
        if token in terms:
            return index
    return None


def _has_passive_by(text: str, subject: Any, object_value: Any, predicate: Any) -> bool:
    tokens = _tokens(text)
    subject_pos = _first_token_position(text, subject)
    object_pos = _first_token_position(text, object_value)
    predicate_pos = _first_token_position(text, predicate, terms=_predicate_terms(predicate))
    by_positions = [index for index, token in enumerate(tokens) if token == "by"]
    return bool(
        subject_pos is not None
        and object_pos is not None
        and predicate_pos is not None
        and object_pos < predicate_pos
        and any(predicate_pos < by_pos < subject_pos for by_pos in by_positions)
    )


def _relation_negated(text: str, subject: Any, predicate: Any, objects: Sequence[Any]) -> bool:
    if not _NEGATION_RE.search(text):
        return False
    tokens = _tokens(text)
    predicate_pos = _first_token_position(text, predicate, terms=_predicate_terms(predicate))
    subject_pos = _first_token_position(text, subject)
    object_positions = [
        position
        for value in objects
        for position in [_first_token_position(text, value)]
        if position is not None
    ]
    relation_positions = [position for position in [subject_pos, predicate_pos, *object_positions] if position is not None]
    if not relation_positions:
        return True
    lo, hi = max(0, min(relation_positions) - 3), min(len(tokens), max(relation_positions) + 4)
    relation_window = " ".join(tokens[lo:hi])
    return bool(_NEGATION_RE.search(relation_window))


def _text_without_frozen_values(text: str, values: Sequence[Any]) -> str:
    normalized = _normalized_text(text)
    for value in values:
        for surface in _flatten_surface_values(value):
            phrase = _normalized_text(surface)
            if phrase:
                normalized = normalized.replace(phrase, " ")
    return " ".join(normalized.split())


def _status_marker_present(
    text: str,
    status: Any,
    *,
    frozen_object_values: Sequence[Any] = (),
) -> bool:
    normalized_status = _normalized_status(status)
    if normalized_status in {"", "active", "current"}:
        return True
    markers = _STATUS_MARKERS.get(normalized_status)
    if markers is None:
        return _surface_present(text, status)
    normalized_quote = _text_without_frozen_values(text, frozen_object_values)
    return any(_normalized_text(marker) in normalized_quote for marker in markers)


def _conflicting_status_marker(
    text: str,
    status: Any,
    *,
    frozen_object_values: Sequence[Any] = (),
) -> str | None:
    normalized_status = _normalized_status(status)
    normalized_quote = _text_without_frozen_values(text, frozen_object_values)
    for candidate, markers in _STATUS_MARKERS.items():
        if candidate == normalized_status:
            continue
        # ``only`` and ``limited`` are often the independently frozen scope of
        # an otherwise active visibility claim, not a lifecycle conflict.
        if candidate in {"active", "current", "scope_limited"}:
            continue
        if any(_normalized_text(marker) in normalized_quote for marker in markers):
            return candidate
    return None


def _time_surface_values(value: Any) -> tuple[list[str], set[int]]:
    """Separate natural time labels from internal ordering indices.

    ``time_index`` and ``global_time_index`` are compiler coordinates.  They
    are checked against ledger/turn ordering and need not be awkwardly spoken
    by the API.  If a quote *does* say "time 3", it must agree with one of
    those coordinates.  Natural fields such as ``label`` or ``global_time``
    remain mandatory semantic text.
    """

    if not isinstance(value, Mapping):
        return _flatten_surface_values(value), set()
    internal_indexes = {
        item
        for key in ("time_index", "global_time_index")
        for item in [value.get(key)]
        if _is_int(item)
    }
    surfaced: list[str] = []
    for key in (
        "label",
        "text",
        "value",
        "local_time",
        "global_time",
        "date",
        "timestamp",
    ):
        if key in value:
            surfaced.extend(_flatten_surface_values(value[key]))
    return surfaced, internal_indexes


def _time_present(text: str, value: Any) -> bool:
    values, internal_indexes = _time_surface_values(value)
    if not values:
        if not internal_indexes:
            return True
        explicit_index = re.search(
            r"\b(?:time|step|update|session|turn)\s*(?:index\s*)?(\d+)\b",
            text,
            flags=re.IGNORECASE,
        )
        return explicit_index is None or int(explicit_index.group(1)) in internal_indexes
    normalized_quote = _normalized_text(text)
    quote_tokens = set(_tokens(text))
    for raw in values:
        normalized = _normalized_text(raw)
        if normalized and normalized in normalized_quote:
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError):
            number = -1
        variants = set(_NUMBER_WORDS.get(number, ())) | {str(number)} if number >= 0 else set()
        if not variants or not ({_stem(item) for item in variants} & quote_tokens):
            return False
    return True


def _validate_quote_semantics(
    claim: Mapping[str, Any],
    quote: str,
    *,
    path: str,
) -> list[dict[str, Any]]:
    semantics = _surface_semantics(claim)
    subject = semantics["subject"]
    predicate = semantics["predicate"]
    objects = semantics["objects"]
    issues: list[dict[str, Any]] = []

    subject_present = _surface_present(quote, subject)
    if not subject_present:
        issues.append(
            _issue(
                "support_subject_missing",
                path,
                "The exact quote does not contain the claim subject's semantic tokens.",
                claim_id=claim.get("claim_id"),
            )
        )
    predicate_present = _predicate_present(quote, predicate)
    if not predicate_present:
        issues.append(
            _issue(
                "support_predicate_missing",
                path,
                "The exact quote does not express the canonical predicate.",
                claim_id=claim.get("claim_id"),
                predicate=predicate,
            )
        )
    missing_objects = [value for value in objects if not _surface_present(quote, value)]
    if missing_objects:
        issues.append(
            _issue(
                "support_object_missing",
                path,
                "The exact quote omits semantic tokens for one or more claim objects.",
                claim_id=claim.get("claim_id"),
                missing_objects=missing_objects,
            )
        )

    same_single_role = bool(
        len(objects) == 1
        and _normalized_text(subject) == _normalized_text(objects[0])
    )
    if (
        subject_present
        and predicate_present
        and not missing_objects
        and objects
        and not same_single_role
    ):
        subject_pos = _first_token_position(quote, subject)
        predicate_pos = _first_token_position(quote, predicate, terms=_predicate_terms(predicate))
        object_positions = [_first_token_position(quote, value) for value in objects]
        active_direction = bool(
            subject_pos is not None
            and predicate_pos is not None
            and all(position is not None for position in object_positions)
            and subject_pos < predicate_pos
            and all(subject_pos < int(position) for position in object_positions)
        )
        passive_direction = len(objects) == 1 and _has_passive_by(
            quote, subject, objects[0], predicate
        )
        if not active_direction and not passive_direction:
            issues.append(
                _issue(
                    "support_direction_mismatch",
                    path,
                    "Subject, predicate, and object occur in a direction inconsistent with the claim.",
                    claim_id=claim.get("claim_id"),
                )
            )

    expected_polarity = _normalized_polarity(semantics["polarity"])
    negated = _relation_negated(quote, subject, predicate, objects)
    if expected_polarity == "negative" and not negated:
        issues.append(
            _issue(
                "support_negation_missing",
                path,
                "A negative claim must have explicit negation in its relation clause.",
                claim_id=claim.get("claim_id"),
            )
        )
    status_name = _normalized_status(semantics["status"])
    if expected_polarity == "positive" and negated and status_name in _ACTIVE_STATUSES:
        issues.append(
            _issue(
                "support_unexpected_negation",
                path,
                "The quote negates a relation whose frozen polarity is positive.",
                claim_id=claim.get("claim_id"),
            )
        )

    if not _status_marker_present(
        quote,
        semantics["status"],
        frozen_object_values=objects,
    ):
        issues.append(
            _issue(
                "support_status_missing",
                path,
                "The quote does not express the frozen non-active/status qualifier.",
                claim_id=claim.get("claim_id"),
                status=semantics["status"],
            )
        )
    conflict = _conflicting_status_marker(
        quote,
        semantics["status"],
        frozen_object_values=objects,
    )
    if conflict is not None and status_name in _ACTIVE_STATUSES:
        issues.append(
            _issue(
                "support_status_conflict",
                path,
                "The quote carries a status marker that conflicts with the frozen claim.",
                claim_id=claim.get("claim_id"),
                conflicting_status=conflict,
            )
        )

    if not _time_present(quote, semantics["time"]):
        issues.append(
            _issue(
                "support_time_missing",
                path,
                "The quote does not express the frozen time value.",
                claim_id=claim.get("claim_id"),
                time=semantics["time"],
            )
        )

    condition = semantics["condition"]
    has_condition = bool(_flatten_surface_values(condition))
    quote_has_condition = bool(_CONDITION_RE.search(quote))
    if has_condition and (not quote_has_condition or not _surface_present(quote, condition)):
        issues.append(
            _issue(
                "support_condition_missing",
                path,
                "The quote does not preserve the claim's conditional qualifier.",
                claim_id=claim.get("claim_id"),
                condition=condition,
            )
        )
    elif not has_condition and quote_has_condition:
        issues.append(
            _issue(
                "support_unexpected_condition",
                path,
                "The quote adds a condition that is absent from the frozen claim.",
                claim_id=claim.get("claim_id"),
            )
        )

    scope = semantics["scope"]
    if _flatten_surface_values(scope) and not _surface_present(quote, scope):
        issues.append(
            _issue(
                "support_scope_missing",
                path,
                "The quote omits the frozen visibility/scope qualifier.",
                claim_id=claim.get("claim_id"),
                scope=scope,
            )
        )
    return issues


def _ledger_entries(value: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    if isinstance(value, list):
        return value, []
    if isinstance(value, Mapping) and isinstance(value.get("entries"), list):
        issues: list[dict[str, Any]] = []
        by_claim_id = value.get("by_claim_id")
        if by_claim_id is not None:
            if not isinstance(by_claim_id, Mapping):
                issues.append(
                    _issue(
                        "invalid_ledger_index",
                        "$.support_ledger.by_claim_id",
                        "by_claim_id must be a JSON object when supplied.",
                    )
                )
            else:
                for index, entry in enumerate(value["entries"]):
                    if not isinstance(entry, Mapping) or not _nonempty_string(entry.get("claim_id")):
                        continue
                    indexed = by_claim_id.get(entry["claim_id"])
                    if indexed is None or not _json_equal(indexed, entry):
                        issues.append(
                            _issue(
                                "ledger_index_mismatch",
                                f"$.support_ledger.entries[{index}]",
                                "by_claim_id must exactly mirror every ledger entry.",
                                claim_id=entry["claim_id"],
                            )
                        )
        return value["entries"], issues
    return [], [
        _issue(
            "invalid_support_ledger",
            "$.support_ledger",
            "support_ledger must be an array or an object containing entries.",
        )
    ]


def _turns(value: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    if isinstance(value, list):
        return value, []
    if isinstance(value, Mapping) and isinstance(value.get("turns"), list):
        return value["turns"], []
    return [], [_issue("invalid_dialogue", "$.dialogue", "dialogue must provide a turns array.")]


def _turn_order(turns: Sequence[Any]) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
    order: dict[str, int] = {}
    texts: dict[str, str] = {}
    issues: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        path = f"$.dialogue.turns[{index}]"
        if not isinstance(turn, Mapping):
            issues.append(_issue("invalid_turn", path, "A dialogue turn must be a JSON object."))
            continue
        turn_id, text = turn.get("turn_id"), turn.get("text")
        if not _nonempty_string(turn_id):
            issues.append(_issue("invalid_turn_id", f"{path}.turn_id", "turn_id must be a non-empty string."))
            continue
        if turn_id in order:
            issues.append(_issue("duplicate_turn_id", f"{path}.turn_id", "turn_id must be unique.", turn_id=turn_id))
            continue
        if not _nonempty_string(text):
            issues.append(_issue("invalid_turn_text", f"{path}.text", "turn text must be a non-empty string."))
            continue
        order[turn_id] = index
        texts[turn_id] = text
    return order, texts, issues


def _state_key(claim: Mapping[str, Any]) -> tuple[str, str, str, str]:
    semantics = _surface_semantics(claim)
    return (
        _normalized_text(semantics["subject"]),
        _normalized_text(semantics["predicate"]),
        json.dumps(semantics["condition"], sort_keys=True, ensure_ascii=False),
        json.dumps(semantics["scope"], sort_keys=True, ensure_ascii=False),
    )


def _time_key(value: Any) -> tuple[int, Any]:
    if _is_int(value) or isinstance(value, float):
        return (0, value)
    normalized = _normalized_text(value)
    if normalized.isdigit():
        return (0, int(normalized))
    return (1, normalized)


def _is_effective(claim: Mapping[str, Any]) -> bool:
    semantics = _surface_semantics(claim)
    status = _normalized_status(semantics["status"])
    polarity = _normalized_polarity(semantics["polarity"])
    return polarity == "positive" and status not in _NON_ACTIVE_STATUS_NAMES


def _objects_key(claim: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_normalized_text(value) for value in _surface_semantics(claim)["objects"])


def _proof_answer_ids(proof: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    derivation = proof.get("derivation")
    answer_ids = _extract_answer_claim_ids(derivation)
    if answer_ids:
        return answer_ids
    if _has_explicit_answer_claim_ids(derivation):
        # Explicit [] is a derived value, not permission to reinterpret an
        # operand as the direct answer claim.
        return []
    required = proof.get("required_claim_ids")
    if not isinstance(required, list):
        return []
    inferred = _infer_answer_claim_ids(_proof_semantic_gold(proof), required, by_id)
    return inferred if len(inferred) == 1 else []


def _proof_requires_current_state_resolution(proof: Mapping[str, Any]) -> bool:
    """Return whether chronology/override is part of this proof's semantics.

    A status-conditioned question may intentionally ask for the canceled,
    belief, or historical value.  Likewise relation, count, and comparison
    proofs can mention several states without asking for the latest one.  The
    override gate therefore activates only when the frozen derivation itself
    declares latest/current-state resolution.
    """

    derivation = proof.get("derivation")
    if not isinstance(derivation, Mapping):
        return False
    rule = _normalized_text(derivation.get("rule"))
    projection = _normalized_text(
        derivation.get("answer_projection") or proof.get("answer_projection")
    )
    if rule in {
        "select unique status in entity slot",
        "select status conditioned value",
    }:
        return False
    return (
        "latest" in set(_tokens(rule))
        or "current state" in rule
        or projection in {"current value", "latest value", "current state"}
    )


def _later_override_issues(
    claims: Sequence[Mapping[str, Any]],
    proofs: Sequence[Mapping[str, Any]],
    entry_by_claim: Mapping[str, Mapping[str, Any]],
    turns: Sequence[Any],
    turn_order: Mapping[str, int],
) -> list[dict[str, Any]]:
    by_id = {
        claim.get("claim_id"): claim
        for claim in claims
        if _nonempty_string(claim.get("claim_id"))
    }
    current_state_proofs = [
        proof for proof in proofs if _proof_requires_current_state_resolution(proof)
    ]
    answer_ids = {
        claim_id
        for proof in current_state_proofs
        for claim_id in _proof_answer_ids(proof, by_id)
    }
    issues: list[dict[str, Any]] = []

    for answer_id in sorted(answer_ids):
        answer = by_id.get(answer_id)
        if answer is None:
            continue
        answer_time = _time_key(_claim_time(answer))
        answer_entry = entry_by_claim.get(answer_id)
        answer_turn_index = (
            turn_order.get(str(answer_entry.get("turn_id")))
            if isinstance(answer_entry, Mapping)
            else None
        )
        for other_id, other in by_id.items():
            if other_id == answer_id or _state_key(other) != _state_key(answer):
                continue
            other_entry = entry_by_claim.get(str(other_id))
            other_turn_index = (
                turn_order.get(str(other_entry.get("turn_id")))
                if isinstance(other_entry, Mapping)
                else None
            )
            later_by_time = _time_key(_claim_time(other)) > answer_time
            later_by_turn = (
                answer_turn_index is not None
                and other_turn_index is not None
                and other_turn_index > answer_turn_index
            )
            supersedes = (
                _is_effective(other) and _objects_key(other) != _objects_key(answer)
            ) or (
                not _is_effective(other) and _objects_key(other) == _objects_key(answer)
            )
            if (later_by_time or later_by_turn) and supersedes:
                issues.append(
                    _issue(
                        "answer_claim_later_overridden",
                        "$.proof.derivation",
                        "The declared answer claim is superseded by a later frozen claim.",
                        answer_claim_id=answer_id,
                        overriding_claim_id=other_id,
                    )
                )

        if answer_turn_index is None:
            continue
        semantics = _surface_semantics(answer)
        subject, predicate, objects = semantics["subject"], semantics["predicate"], semantics["objects"]
        modeled_turn_ids = {
            str(entry_by_claim[other_id].get("turn_id"))
            for other_id, other in by_id.items()
            if other_id != answer_id
            and _state_key(other) == _state_key(answer)
            and isinstance(entry_by_claim.get(str(other_id)), Mapping)
            and _nonempty_string(entry_by_claim[str(other_id)].get("turn_id"))
        }
        for turn_index in range(answer_turn_index + 1, len(turns)):
            turn = turns[turn_index]
            if not isinstance(turn, Mapping) or not _nonempty_string(turn.get("text")):
                continue
            if str(turn.get("turn_id") or "") in modeled_turn_ids:
                # A frozen same-relation claim already accounts for this
                # support turn.  It may still trigger the precise
                # ``answer_claim_later_overridden`` issue above, but it is not
                # an *unmodeled* override.
                continue
            text = turn["text"]
            if (
                _OVERRIDE_RE.search(text)
                and _surface_present(text, subject)
                and _predicate_present(text, predicate)
                and any(not _surface_present(text, value) for value in objects)
            ):
                issues.append(
                    _issue(
                        "answer_claim_later_unmodeled_override",
                        f"$.dialogue.turns[{turn_index}].text",
                        "A later turn appears to override the answer claim but has no frozen answer resolution.",
                        answer_claim_id=answer_id,
                        later_turn_id=turn.get("turn_id"),
                    )
                )
                break
    return issues


def validate_support_ledger(
    claims: Sequence[Mapping[str, Any]],
    proofs: Sequence[Mapping[str, Any]],
    support_ledger: Any,
    dialogue: Any,
) -> list[dict[str, Any]]:
    """Validate exact evidence binding and semantic fidelity."""

    entries, issues = _ledger_entries(support_ledger)
    turns, turn_issues = _turns(dialogue)
    issues.extend(turn_issues)
    turn_order, turn_text, turn_shape_issues = _turn_order(turns)
    issues.extend(turn_shape_issues)
    by_id = {
        claim.get("claim_id"): claim
        for claim in claims
        if isinstance(claim, Mapping) and _nonempty_string(claim.get("claim_id"))
    }
    required_ids = {
        claim_id
        for proof in proofs
        if isinstance(proof, Mapping)
        for claim_id in (proof.get("required_claim_ids") or [])
        if _nonempty_string(claim_id)
    }
    required_ids.update(
        claim.get("claim_id")
        for claim in claims
        if isinstance(claim, Mapping)
        and claim.get("answer_bearing") is True
        and claim.get("qa_eligible") is True
        and _nonempty_string(claim.get("claim_id"))
    )

    entry_by_claim: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        path = f"$.support_ledger.entries[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(_issue("invalid_ledger_entry", path, "A ledger entry must be a JSON object."))
            continue
        for field in CANONICAL_LEDGER_FIELDS:
            if field not in entry:
                issues.append(
                    _issue(
                        "missing_ledger_field",
                        f"{path}.{field}",
                        f"Required ledger field '{field}' is missing.",
                        field=field,
                    )
                )
        claim_id = entry.get("claim_id")
        if not _nonempty_string(claim_id):
            issues.append(_issue("invalid_ledger_claim_id", f"{path}.claim_id", "claim_id must be a non-empty string."))
            continue
        if claim_id in entry_by_claim:
            issues.append(
                _issue(
                    "duplicate_ledger_claim_id",
                    f"{path}.claim_id",
                    "Each claim may have exactly one support-ledger binding.",
                    claim_id=claim_id,
                )
            )
            continue
        entry_by_claim[claim_id] = entry
        claim = by_id.get(claim_id)
        if claim is None:
            issues.append(
                _issue(
                    "ledger_unknown_claim",
                    f"{path}.claim_id",
                    "The ledger references an unknown claim.",
                    claim_id=claim_id,
                )
            )
            continue

        turn_id, quote = entry.get("turn_id"), entry.get("exact_support_quote")
        is_required = claim_id in required_ids
        if not _nonempty_string(turn_id) or not _nonempty_string(quote):
            if is_required:
                issues.append(
                    _issue(
                        "required_claim_unbound",
                        path,
                        "Every required/answer-bearing claim needs claim_id, turn_id, and exact_support_quote.",
                        claim_id=claim_id,
                    )
                )
            continue
        if turn_id not in turn_text:
            issues.append(
                _issue(
                    "support_turn_not_found",
                    f"{path}.turn_id",
                    "turn_id does not exist in the dialogue.",
                    claim_id=claim_id,
                    turn_id=turn_id,
                )
            )
            continue
        if quote not in turn_text[turn_id]:
            issues.append(
                _issue(
                    "support_quote_not_exact_substring",
                    f"{path}.exact_support_quote",
                    "exact_support_quote must be a case-sensitive substring of its cited turn.",
                    claim_id=claim_id,
                    turn_id=turn_id,
                )
            )
            continue
        issues.extend(
            _validate_quote_semantics(
                claim,
                quote,
                path=f"{path}.exact_support_quote",
            )
        )

    for claim_id in sorted(required_ids):
        if claim_id not in entry_by_claim:
            issues.append(
                _issue(
                    "missing_required_claim_ledger",
                    "$.support_ledger",
                    "A required/answer-bearing claim has no support-ledger entry.",
                    claim_id=claim_id,
                )
            )

    # A later semantic state must be realised later in the dialogue.  This
    # catches a time-label mutation even when both individual quotes happen to
    # contain their (mutated) labels.
    chronology_claim_ids = {
        claim_id
        for proof in proofs
        if _proof_requires_current_state_resolution(proof)
        for claim_id in (proof.get("required_claim_ids") or [])
        if _nonempty_string(claim_id)
    }
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for claim in claims:
        if (
            isinstance(claim, Mapping)
            and claim.get("claim_id") in chronology_claim_ids
        ):
            grouped.setdefault(_state_key(claim), []).append(claim)
    for group in grouped.values():
        ordered = sorted(group, key=lambda claim: _time_key(_claim_time(claim)))
        previous_turn: int | None = None
        previous_id: str | None = None
        for claim in ordered:
            claim_id = str(claim.get("claim_id") or "")
            entry = entry_by_claim.get(claim_id)
            turn_index = turn_order.get(str(entry.get("turn_id"))) if entry else None
            if turn_index is None:
                continue
            if previous_turn is not None and turn_index <= previous_turn:
                issues.append(
                    _issue(
                        "support_time_order_mismatch",
                        "$.support_ledger",
                        "Claim time order disagrees with dialogue support-turn order.",
                        earlier_claim_id=previous_id,
                        later_claim_id=claim_id,
                    )
                )
            previous_turn, previous_id = turn_index, claim_id

    issues.extend(
        _later_override_issues(claims, proofs, entry_by_claim, turns, turn_order)
    )
    return issues


def validate_claim_artifacts(
    claims: Any,
    proof: Any,
    support_ledger: Any,
    dialogue: Any,
) -> list[dict[str, Any]]:
    """Validate four plain-JSON artifacts through independent implementations."""

    issues: list[dict[str, Any]] = []
    if not isinstance(claims, list) or not claims:
        return [_issue("invalid_claims", "$.claims", "claims must be a non-empty JSON array.")]

    valid_claims: list[Mapping[str, Any]] = []
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        path = f"$.claims[{index}]"
        issues.extend(validate_claim(claim, path=path))
        if not isinstance(claim, Mapping):
            continue
        valid_claims.append(claim)
        claim_id = claim.get("claim_id")
        if _nonempty_string(claim_id):
            if claim_id in claim_ids:
                issues.append(
                    _issue(
                        "duplicate_claim_id",
                        f"{path}.claim_id",
                        "claim_id must be unique within the frozen bundle.",
                        claim_id=claim_id,
                    )
                )
            claim_ids.add(claim_id)

    if isinstance(proof, list):
        raw_proofs = proof
        proof_base_path = "$.proofs"
    else:
        raw_proofs = [proof]
        proof_base_path = "$.proof"
    valid_proofs: list[Mapping[str, Any]] = []
    proof_ids: set[str] = set()
    for index, candidate in enumerate(raw_proofs):
        path = proof_base_path if not isinstance(proof, list) else f"{proof_base_path}[{index}]"
        issues.extend(validate_proof(candidate, valid_claims, path=path))
        if not isinstance(candidate, Mapping):
            continue
        valid_proofs.append(candidate)
        proof_id = candidate.get("proof_id")
        if _nonempty_string(proof_id):
            if proof_id in proof_ids:
                issues.append(
                    _issue(
                        "duplicate_proof_id",
                        f"{path}.proof_id",
                        "proof_id must be unique within a multi-QA bundle.",
                        proof_id=proof_id,
                    )
                )
            proof_ids.add(proof_id)

    issues.extend(
        validate_support_ledger(
            valid_claims,
            valid_proofs,
            support_ledger,
            dialogue,
        )
    )
    return issues


def validate_claim_bundle(bundle: Any) -> list[dict[str, Any]]:
    """Validate the canonical bundle shape and return structured issues."""

    if not isinstance(bundle, Mapping):
        return [_issue("invalid_bundle_type", "$", "A claim bundle must be a JSON object.")]
    try:
        json.dumps(bundle, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        return [
            _issue(
                "bundle_not_plain_json",
                "$",
                "The bundle contains a value that is not plain JSON.",
                error=str(error),
            )
        ]

    claim_container = bundle.get("claim_ir")
    claims = bundle.get("claims")
    if claims is None and isinstance(claim_container, Mapping):
        claims = claim_container.get("claims")
    support_ledger = bundle.get("support_ledger")
    if support_ledger is None and isinstance(claim_container, Mapping):
        support_ledger = claim_container.get("support_ledger")
    proof: Any
    if "proofs" in bundle:
        proof = bundle.get("proofs")
    else:
        proof = bundle.get("proof")
    dialogue = bundle.get("dialogue", bundle.get("turns"))
    return validate_claim_artifacts(claims, proof, support_ledger, dialogue)


def validation_report(bundle: Any) -> dict[str, Any]:
    """Return a JSON-ready pass/fail report for pipelines and replay scripts."""

    issues = validate_claim_bundle(bundle)
    counts: dict[str, int] = {}
    for issue in issues:
        code = str(issue.get("code") or "unknown")
        counts[code] = counts.get(code, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": not issues,
        "issue_count": len(issues),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": issues,
    }


def is_valid_claim_bundle(bundle: Any) -> bool:
    return not validate_claim_bundle(bundle)


# Short aliases make the module convenient in replay code while retaining the
# explicit names above for readers.
validate_bundle = validate_claim_bundle
is_valid_bundle = is_valid_claim_bundle


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Bundle JSON path; omit to read stdin.")
    args = parser.parse_args(argv)
    if args.path:
        with open(args.path, "r", encoding="utf-8") as handle:
            bundle = json.load(handle)
    else:
        bundle = json.load(sys.stdin)
    report = validation_report(bundle)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
