"""Lightweight independent semantic gate for Claim-first Stage3 QAs.

The proof validator checks that a QA is mechanically derived from its frozen
claims.  This module asks a different question: do the predicate, value,
semantic roles, status and question *mean* what that derivation says they
mean?  It deliberately consumes only plain JSON and does not import either
the claim compiler or the proof validator.

The gate is intentionally conservative and extensible.  Predicate/value
checks are organized as type families (location, possession, numeric, and so
on), rather than as rules for individual artifacts.  Every QA receives its
own report; the batch API never mutates or removes the dialogue or sibling
QAs.  A caller may therefore discard one bad sibling while retaining the
rest of the dialogue family.
"""

from __future__ import annotations

import collections
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage3_claim_semantic_validator_v1"

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)
_TOKEN_SEPARATOR_RE = re.compile(r"[\W_]+", flags=re.UNICODE)

# These families describe value *shapes*, not domain instances.  New slots
# should be added here rather than patched into QA-specific conditionals.
PREDICATE_TYPE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "name": "location",
        "predicates": frozenset(
            {
                "destination",
                "location",
                "venue",
                "birthplace",
                "hometown",
                "residence",
                "address",
                "origin_location",
                "travel_destination",
                "meeting_location",
                "event_location",
            }
        ),
        "suffixes": ("_destination", "_location", "_venue"),
        "max_tokens": 12,
        "value_kind": "noun_phrase",
    },
    {
        "name": "possession",
        "predicates": frozenset(
            {
                "possession",
                "possessions",
                "owned_item",
                "owned_object",
                "belonging",
                "belongings",
                "equipment",
                "vehicle",
            }
        ),
        "suffixes": ("_possession", "_owned_item", "_owned_object"),
        "max_tokens": 12,
        "value_kind": "noun_phrase",
    },
    {
        "name": "numeric",
        "predicates": frozenset(
            {
                "count",
                "quantity",
                "age",
                "score",
                "amount",
                "service_hours",
                "duration_minutes",
            }
        ),
        "suffixes": ("_count", "_quantity", "_score", "_hours", "_minutes"),
        "value_kind": "number",
    },
    {
        "name": "boolean",
        "predicates": frozenset({"enabled", "available", "completed", "approved"}),
        "suffixes": ("_enabled", "_available", "_completed", "_approved"),
        "value_kind": "boolean",
    },
)

_PROPOSITION_START_RE = re.compile(
    r"^(?:i|you|he|she|we|they|it|this|that|[A-Z][\w'-]*)\s+"
    r"(?:am|is|are|was|were|has|have|had|plans?|planned|intends?|intended|"
    r"wants?|wanted|will|would|can|could|should|must|went|goes?|traveled|"
    r"travelled|works?|worked|volunteers?|volunteered|hosts?|hosted|"
    r"owns?|owned|belongs?|became)\b",
    flags=re.IGNORECASE,
)
_BARE_PREDICATE_START_RE = re.compile(
    r"^(?:am|is|are|was|were|has|have|had|plans?|planned|intends?|intended|"
    r"wants?|wanted|will|would|can|could|should|must|went|goes?|traveled|"
    r"travelled|works?|worked|volunteers?|volunteered|hosts?|hosted|"
    r"owns?|owned|belongs?|became)\b",
    flags=re.IGNORECASE,
)
_CLAUSE_LINK_RE = re.compile(
    r"\b(?:because|although|while|whereas|and then|which|who|that)\b",
    flags=re.IGNORECASE,
)
_RELATIONAL_DIRECTION_RE = re.compile(
    r"^(?:am|is|are|was|were|became|becomes?)\b.{0,80}\b(?:of|by|to)\s+"
    r"(?:[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3})\b",
    flags=re.IGNORECASE,
)

_ACTIVE_STATUSES = frozenset({"", "active", "current", "confirmed", "restored", "true"})
_STATUS_ALIASES: dict[str, frozenset[str]] = {
    "belief": frozenset({"belief", "believed", "thinks", "thought", "may", "might", "unsure"}),
    "report": frozenset({"report", "reported", "alleged", "unconfirmed", "not confirmed"}),
    "reported": frozenset({"report", "reported", "alleged", "unconfirmed", "not confirmed"}),
    "unconfirmed": frozenset({"unconfirmed", "not confirmed", "reported", "alleged"}),
    "proposed": frozenset({"proposed", "proposal", "suggested", "tentative"}),
    "planned": frozenset({"planned", "plan", "intended", "scheduled", "upcoming"}),
    "pending": frozenset({"pending", "waiting", "not yet", "awaiting"}),
    "delayed_not_active": frozenset({"delayed", "not active", "not yet", "pending"}),
    "expired": frozenset({"expired", "lapsed", "no longer valid", "formerly"}),
    "outdated": frozenset({"outdated", "obsolete", "superseded", "no longer applies", "formerly"}),
    "canceled": frozenset({"canceled", "cancelled", "called off", "withdrawn", "did not take effect"}),
    "cancelled": frozenset({"cancelled", "canceled", "called off", "withdrawn", "did not take effect"}),
    "scope_limited": frozenset({"limited", "restricted", "authorized", "only", "private"}),
    "false": frozenset({"false", "incorrect", "denied", "not true"}),
    "inactive": frozenset({"inactive", "not active", "former", "formerly"}),
    "superseded": frozenset({"superseded", "replaced", "outdated", "former", "formerly"}),
}
_NONACTIVE_STATUSES = frozenset(_STATUS_ALIASES)
_CURRENT_QUESTION_RE = re.compile(
    r"\b(?:current|currently|now|latest|still|confirmed|actually|in effect)\b",
    flags=re.IGNORECASE,
)
_TEMPORAL_QUESTION_RE = re.compile(
    r"\b(?:as of|at time|on \w+ \d|in \d{4}|before|after|then|previously|"
    r"formerly|latest|current|currently|now|first|last|earlier|later)\b",
    flags=re.IGNORECASE,
)
_STATUS_SELECTION_RULES = frozenset(
    {"select_unique_status_in_entity_slot", "select_by_status", "status_conditioned_lookup"}
)
_QUESTION_PREDICATE_MODIFIERS = frozenset(
    {
        "active",
        "believed",
        "belief",
        "canceled",
        "cancelled",
        "confirmed",
        "current",
        "expired",
        "former",
        "latest",
        "old",
        "outdated",
        "pending",
        "previous",
        "proposed",
        "record",
        "recorded",
        "reported",
        "status",
        "unconfirmed",
        "value",
    }
)


def _norm_token(value: Any) -> str:
    return _TOKEN_SEPARATOR_RE.sub("_", str(value or "").casefold()).strip("_")


def _norm_text(value: Any) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def _tokens(value: Any) -> list[str]:
    return _WORD_RE.findall(str(value or ""))


def _issue(code: str, path: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "path": path, "message": message}
    if details:
        result["details"] = details
    return result


def _claim_list(claim_ir: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    claims = claim_ir.get("claims")
    if not isinstance(claims, list):
        nested = claim_ir.get("claim_ir")
        if isinstance(nested, Mapping):
            claims = nested.get("claims")
    return [item for item in claims or [] if isinstance(item, Mapping)]


def _claims_by_id(claim_ir: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("claim_id")): item
        for item in _claim_list(claim_ir)
        if str(item.get("claim_id") or "").strip()
    }


def _claim_entity(claim: Mapping[str, Any]) -> str:
    support = claim.get("canonical_support_claim")
    if isinstance(support, Mapping) and support.get("subject") is not None:
        return str(support.get("subject"))
    return str(claim.get("entity") or claim.get("subject") or "")


def _claim_predicate(claim: Mapping[str, Any]) -> str:
    return _norm_token(claim.get("slot") or claim.get("predicate"))


def _claim_value(claim: Mapping[str, Any]) -> Any:
    if _norm_token(claim.get("claim_type")) == "visibility" and claim.get("observer") is not None:
        return claim.get("observer")
    if "value" in claim:
        return claim.get("value")
    support = claim.get("canonical_support_claim")
    if isinstance(support, Mapping):
        objects = support.get("objects", support.get("object", support.get("value")))
        if isinstance(objects, list) and len(objects) == 1:
            return objects[0]
        if objects is not None:
            return objects
    return claim.get("object")


def _claim_status(claim: Mapping[str, Any]) -> str:
    support = claim.get("canonical_support_claim")
    value = support.get("status") if isinstance(support, Mapping) else None
    return _norm_token(value if value is not None else claim.get("status"))


def _claim_time(claim: Mapping[str, Any]) -> Any:
    for key in ("global_time_index", "time_index", "time"):
        if claim.get(key) is not None:
            return claim.get(key)
    support = claim.get("canonical_support_claim")
    return support.get("time") if isinstance(support, Mapping) else None


def predicate_type_family(predicate: Any) -> Mapping[str, Any] | None:
    """Return the configured value-shape family for ``predicate``."""

    normalized = _norm_token(predicate)
    for family in PREDICATE_TYPE_FAMILIES:
        if normalized in family["predicates"] or any(
            normalized.endswith(suffix) for suffix in family.get("suffixes", ())
        ):
            return family
    return None


def _looks_like_proposition(value: str) -> bool:
    text = value.strip()
    return bool(
        _PROPOSITION_START_RE.search(text)
        or _BARE_PREDICATE_START_RE.search(text)
        or _CLAUSE_LINK_RE.search(text)
        or len(re.findall(r"[.!?]", text)) > 1
    )


def _value_shape_issues(
    claim: Mapping[str, Any], *, path: str
) -> list[dict[str, Any]]:
    predicate = _claim_predicate(claim)
    family = predicate_type_family(predicate)
    if family is None:
        return []
    value = _claim_value(claim)
    kind = family["value_kind"]
    issues: list[dict[str, Any]] = []

    valid = True
    reason = ""
    if kind == "noun_phrase":
        valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            reason = "expected a non-empty noun phrase"
        elif _looks_like_proposition(value):
            valid = False
            reason = "a complete action/relation proposition was used where a noun phrase is required"
        elif len(_tokens(value)) > int(family.get("max_tokens", 10**9)):
            valid = False
            reason = "the value is too broad for the predicate's answer granularity"
    elif kind == "number":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ) or bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(value or "").strip()))
        reason = "expected a numeric scalar"
    elif kind == "boolean":
        valid = isinstance(value, bool) or _norm_text(value) in {
            "true", "false", "yes", "no", "enabled", "disabled"
        }
        reason = "expected a boolean value"

    if not valid:
        issues.append(
            _issue(
                "predicate_value_type_mismatch",
                path,
                f"predicate family {family['name']!r} has an incompatible value: {reason}",
                predicate=predicate,
                predicate_family=family["name"],
                value=value,
                expected_value_kind=kind,
            )
        )

    if isinstance(value, str) and kind == "noun_phrase" and _looks_like_proposition(value):
        issues.append(
            _issue(
                "answer_value_wrong_granularity",
                path,
                "the answer is a full proposition instead of the predicate's atomic value",
                predicate=predicate,
                value=value,
            )
        )

    if (
        family["name"] == "possession"
        and isinstance(value, str)
        and _RELATIONAL_DIRECTION_RE.search(value.strip())
    ):
        issues.append(
            _issue(
                "semantic_role_direction_mismatch",
                path,
                "a possession value describes the subject as somebody else's object instead of naming what the subject possesses",
                subject=_claim_entity(claim),
                predicate=predicate,
                value=value,
            )
        )
    return issues


def _proof(qa: Mapping[str, Any]) -> Mapping[str, Any]:
    value = qa.get("proof")
    return value if isinstance(value, Mapping) else {}


def _derivation(qa: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _proof(qa).get("derivation")
    return value if isinstance(value, Mapping) else {}


def _string_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _gold_option_claim_ids(qa: Mapping[str, Any]) -> list[str]:
    labels = set(_string_ids(qa.get("gold")))
    result: list[str] = []
    for option in qa.get("options") or []:
        if not isinstance(option, Mapping):
            continue
        is_gold = option.get("truth_value") is True or str(option.get("id")) in labels
        claim_id = str(option.get("source_claim_id") or "")
        if is_gold and claim_id:
            result.append(claim_id)
    return list(dict.fromkeys(result))


def _answer_claim_ids(qa: Mapping[str, Any]) -> list[str]:
    derivation = _derivation(qa)
    if _norm_token(derivation.get("rule")) in {
        "absolute_difference",
        "count_distinct",
        "compare_values",
    }:
        return []
    result = _string_ids(derivation.get("answer_claim_ids"))
    if not result:
        result = _string_ids(derivation.get("target_claim_ids"))
    if not result:
        result = _gold_option_claim_ids(qa)
    return list(dict.fromkeys(result))


def _binder_claim_ids(qa: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(_string_ids(_derivation(qa).get("binder_claim_ids"))))


def _question(qa: Mapping[str, Any]) -> str:
    return str(
        qa.get("question_text")
        or qa.get("question")
        or qa.get("question_text_draft")
        or ""
    ).strip()


def _requested_status(qa: Mapping[str, Any]) -> str:
    derivation = _derivation(qa)
    status = _norm_token(derivation.get("status"))
    if status:
        return status
    question = _norm_text(_question(qa))
    for name, aliases in _STATUS_ALIASES.items():
        if any(alias in question for alias in aliases):
            return name
    return ""


def _question_qualifies_status(question: str, status: str) -> bool:
    normalized = _norm_text(question)
    aliases = _STATUS_ALIASES.get(status, frozenset({status.replace("_", " ")}))
    return any(alias in normalized for alias in aliases)


def _status_issues(
    qa: Mapping[str, Any],
    answer_claims: Sequence[Mapping[str, Any]],
    binder_claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    question = _question(qa)
    derivation = _derivation(qa)
    requested = _requested_status(qa)
    rule = _norm_token(derivation.get("rule"))
    is_status_selection = rule in _STATUS_SELECTION_RULES
    issues: list[dict[str, Any]] = []

    for index, claim in enumerate(answer_claims):
        status = _claim_status(claim)
        if status in _ACTIVE_STATUSES:
            continue
        qualified = _question_qualifies_status(question, status)
        if is_status_selection and requested == status and qualified:
            continue
        if _CURRENT_QUESTION_RE.search(question):
            issues.append(
                _issue(
                    "nonactive_claim_as_current",
                    f"$.answer_claims[{index}].status",
                    "a non-active claim cannot answer a current/confirmed question",
                    status=status,
                    claim_id=claim.get("claim_id"),
                )
            )
        if not qualified:
            issues.append(
                _issue(
                    "answer_status_qualification_missing",
                    f"$.answer_claims[{index}].status",
                    "the question treats a qualified/non-active answer as an unqualified fact",
                    status=status,
                    claim_id=claim.get("claim_id"),
                )
            )

    for index, claim in enumerate(binder_claims):
        status = _claim_status(claim)
        if status in _ACTIVE_STATUSES:
            continue
        if not _question_qualifies_status(question, status):
            issues.append(
                _issue(
                    "binder_status_qualification_missing",
                    f"$.binder_claims[{index}].status",
                    "an uncertain, proposed, expired, or restricted binder cannot identify an entity as an unqualified fact",
                    status=status,
                    claim_id=claim.get("claim_id"),
                )
            )
    return issues


_POSSESSIVE_QUESTION_RE = re.compile(
    r"\b(?:what|which)\s+(?:is|was|are|were)\s+(.+?)(?:['’]s)\s+([\w -]+?)\s*\?*$",
    flags=re.IGNORECASE,
)


def _role_issues(
    qa: Mapping[str, Any],
    answer_claims: Sequence[Mapping[str, Any]],
    binder_claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    question = _question(qa)
    match = _POSSESSIVE_QUESTION_RE.search(question)
    if match and answer_claims:
        asked_entity = _norm_text(match.group(1))
        asked_predicate = _norm_token(match.group(2))
        for index, claim in enumerate(answer_claims):
            entity = _norm_text(_claim_entity(claim))
            predicate = _claim_predicate(claim)
            if asked_entity and entity and asked_entity != entity:
                issues.append(
                    _issue(
                        "question_subject_mismatch",
                        f"$.answer_claims[{index}].entity",
                        "the question subject does not match the answer-bearing claim subject",
                        question_entity=match.group(1),
                        claim_entity=_claim_entity(claim),
                    )
                )
            asked_parts = asked_predicate.split("_")
            while asked_parts and asked_parts[0] in _QUESTION_PREDICATE_MODIFIERS:
                asked_parts.pop(0)
            while asked_parts and asked_parts[-1] in _QUESTION_PREDICATE_MODIFIERS:
                asked_parts.pop()
            semantic_asked_predicate = "_".join(asked_parts)
            if (
                semantic_asked_predicate
                and predicate
                and semantic_asked_predicate != predicate
            ):
                issues.append(
                    _issue(
                        "question_predicate_mismatch",
                        f"$.answer_claims[{index}].predicate",
                        "the question predicate does not match the answer-bearing claim predicate",
                        question_predicate=semantic_asked_predicate,
                        claim_predicate=predicate,
                    )
                )

    if binder_claims and answer_claims:
        binder_entities = {_norm_text(_claim_entity(item)) for item in binder_claims}
        binder_values = {_norm_text(_claim_value(item)) for item in binder_claims}
        answer_entities = {_norm_text(_claim_entity(item)) for item in answer_claims}
        direct_subject_match = binder_entities == answer_entities
        value_identifies_subject = bool(
            "" not in binder_values
            and "" not in answer_entities
            and answer_entities.issubset(binder_values)
        )
        if (
            "" not in binder_entities
            and "" not in answer_entities
            and not direct_subject_match
            and not value_identifies_subject
        ):
            issues.append(
                _issue(
                    "binder_target_subject_mismatch",
                    "$.proof.derivation",
                    "the binder neither shares nor uniquely identifies the answer subject",
                    binder_entities=sorted(binder_entities),
                    binder_values=sorted(binder_values),
                    target_entities=sorted(answer_entities),
                )
            )
    return issues


def _gold_alignment_issues(
    qa: Mapping[str, Any], claims_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    answer_ids = set(_answer_claim_ids(qa))
    gold_ids = set(_gold_option_claim_ids(qa))
    proof_id = str(_proof(qa).get("proof_id") or "")
    non_derived_gold = {item for item in gold_ids if item != proof_id}
    if answer_ids and non_derived_gold and answer_ids != non_derived_gold:
        issues.append(
            _issue(
                "gold_claim_role_mismatch",
                "$.options",
                "gold options do not point to the proof's answer-bearing claims",
                answer_claim_ids=sorted(answer_ids),
                gold_option_claim_ids=sorted(non_derived_gold),
            )
        )

    gold_text = qa.get("gold_answer_text")
    if isinstance(gold_text, str) and len(answer_ids) == 1:
        claim = claims_by_id.get(next(iter(answer_ids)))
        value = _claim_value(claim) if claim else None
        if isinstance(value, (str, int, float, bool)) and _norm_text(gold_text) != _norm_text(value):
            issues.append(
                _issue(
                    "gold_value_claim_mismatch",
                    "$.gold_answer_text",
                    "gold answer text is not the atomic value of its answer-bearing claim",
                    gold_answer_text=gold_text,
                    claim_value=value,
                )
            )
    return issues


def _turns(dialogue: Any) -> list[Mapping[str, Any]]:
    if isinstance(dialogue, Mapping):
        value = dialogue.get("turns")
    else:
        value = dialogue
    return [item for item in value or [] if isinstance(item, Mapping)] if isinstance(value, list) else []


def _equivalent_value(left: Any, right: Any) -> bool:
    a, b = _norm_text(left), _norm_text(right)
    return bool(a and b and (a == b or a in b or b in a))


def _structured_dialogue_values(
    entity: str,
    predicate: str,
    family: Mapping[str, Any] | None,
    dialogue: Any,
) -> list[dict[str, str]]:
    """Extract auditable answer candidates; never infer arbitrary open text."""

    if not entity or not predicate:
        return []
    entity_re = re.escape(entity)
    display_predicate = r"[\s_-]+".join(re.escape(part) for part in predicate.split("_"))
    labelled = re.compile(
        rf"\b{entity_re}(?:['’]s)\s+{display_predicate}\s+(?:is|was|are|were|"
        rf"is listed as|was listed as)\s+(.+)$",
        flags=re.IGNORECASE,
    )
    # Location questions are especially prone to old-trip/new-plan ambiguity.
    # Restrict extraction to an utterance by the entity and an explicit motion
    # expression followed by a named location; this is a syntactic signal, not
    # a gazetteer or an artifact-specific place list.
    travel_location = re.compile(
        r"\b(?:trip|travel(?:ed|led|ing)?|went|going|visit(?:ed|ing)?|headed|"
        r"flew|flying|destination)\b.{0,100}?\b(?:to|in|at)\s+"
        r"((?:the\s+)?[A-Z][\w'-]*(?:\s+(?:[A-Z][\w'-]*|community|hall|"
        r"center|centre|park|shelter|bank|studio|office)){0,5})",
    )
    results: list[dict[str, str]] = []
    for turn in _turns(dialogue):
        text = str(turn.get("text") or "")
        turn_id = str(turn.get("turn_id") or "")
        found = labelled.search(text)
        if found:
            results.append({"value": found.group(1).strip(), "turn_id": turn_id, "source": "labelled_fact"})
        if family and family["name"] == "location" and _norm_text(turn.get("speaker")) == _norm_text(entity):
            for match in travel_location.finditer(text):
                results.append({"value": match.group(1).strip(), "turn_id": turn_id, "source": "explicit_travel_location"})
    return results


def _ambiguity_issues(
    qa: Mapping[str, Any],
    all_claims: Sequence[Mapping[str, Any]],
    answer_claims: Sequence[Mapping[str, Any]],
    dialogue: Any,
) -> list[dict[str, Any]]:
    if len(answer_claims) != 1:
        return []
    derivation = _derivation(qa)
    if _norm_token(derivation.get("rule")) in _STATUS_SELECTION_RULES:
        return []
    question = _question(qa)
    if _requested_status(qa) or _TEMPORAL_QUESTION_RE.search(question):
        return []

    target = answer_claims[0]
    binder_ids = _binder_claim_ids(qa)
    if binder_ids:
        by_id = {str(item.get("claim_id")): item for item in all_claims}
        binders = [by_id[item] for item in binder_ids if item in by_id]
        target_event_id = str(target.get("subject_event_id") or "")
        binder_event_ids = {str(item.get("subject_event_id") or "") for item in binders}
        dependency_edges = [
            item
            for item in all_claims
            if _norm_token(item.get("claim_type")) == "dependency"
            and str(item.get("subject_event_id") or "") == target_event_id
            and binder_event_ids.intersection(
                str(value) for value in (item.get("object_event_ids") or [])
            )
        ]
        normalized_question = _norm_text(question)
        binder_named = bool(binders) and all(
            _norm_text(_claim_value(item)) in normalized_question for item in binders
        )
        predicate_named = _norm_text(_claim_predicate(target)) in normalized_question
        if len(dependency_edges) == 1 and binder_named and predicate_named:
            return []
    entity = _claim_entity(target)
    predicate = _claim_predicate(target)
    gold_value = _claim_value(target)
    candidates: list[dict[str, Any]] = [
        {
            "value": _claim_value(item),
            "claim_id": item.get("claim_id"),
            "status": _claim_status(item),
            "time": _claim_time(item),
            "source": "claim_ir",
        }
        for item in all_claims
        if _norm_text(_claim_entity(item)) == _norm_text(entity)
        and _claim_predicate(item) == predicate
        and not _equivalent_value(_claim_value(item), gold_value)
    ]
    family = predicate_type_family(predicate)
    for item in _structured_dialogue_values(entity, predicate, family, dialogue):
        if not _equivalent_value(item["value"], gold_value):
            candidates.append(item)
    if not candidates:
        return []
    return [
        _issue(
            "dialogue_value_ambiguity",
            "$.question",
            "the dialogue contains another plausible value for the same subject and predicate, but the question has no status/time disambiguator",
            entity=entity,
            predicate=predicate,
            gold_value=gold_value,
            conflicting_candidates=candidates[:8],
        )
    ]


def _numeric_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _derived_truth_issues(
    qa: Mapping[str, Any],
    claims_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    derivation = _derivation(qa)
    rule = _norm_token(derivation.get("rule"))
    if rule not in {"absolute_difference", "count_distinct"}:
        return []
    operand_ids = _string_ids(derivation.get("operand_claim_ids"))
    proof_operand_ids = _string_ids(_proof(qa).get("gold_operand_claim_ids"))
    issues: list[dict[str, Any]] = []
    if not operand_ids or operand_ids != proof_operand_ids:
        return [
            _issue(
                "derived_operand_path_mismatch",
                "1proof.gold_operand_claim_ids",
                "The proof and derivation must preserve the same ordered operands.",
                derivation_operand_claim_ids=operand_ids,
                proof_operand_claim_ids=proof_operand_ids,
            )
        ]
    missing = [claim_id for claim_id in operand_ids if claim_id not in claims_by_id]
    if missing:
        return [
            _issue(
                "derived_operand_claim_missing",
                "1proof.gold_operand_claim_ids",
                "A derived operand is absent from the frozen Claim IR.",
                missing_claim_ids=missing,
            )
        ]
    claims = [claims_by_id[claim_id] for claim_id in operand_ids]
    expected: int | float | None = None
    if rule == "absolute_difference":
        if len(claims) != 2:
            issues.append(
                _issue(
                    "absolute_difference_arity",
                    "1proof.gold_operand_claim_ids",
                    "absolute_difference requires exactly two operands.",
                )
            )
        else:
            values = [_numeric_scalar(_claim_value(claim)) for claim in claims]
            if any(value is None for value in values):
                issues.append(
                    _issue(
                        "absolute_difference_non_numeric_operand",
                        "1proof.gold_operand_claim_ids",
                        "absolute_difference operands must be numeric node values.",
                    )
                )
            else:
                difference = abs(values[0] - values[1])
                expected = int(difference) if difference.is_integer() else difference
    else:
        if any(_norm_token(claim.get("claim_type")) != "membership" for claim in claims):
            issues.append(
                _issue(
                    "count_invalid_membership_operand",
                    "1proof.gold_operand_claim_ids",
                    "count_distinct operands must all be membership relations.",
                )
            )
        included_rows = [
            (claim_id, claim.get("included"))
            for claim_id, claim in zip(operand_ids, claims)
        ]
        if any(not isinstance(included, bool) for _claim_id_value, included in included_rows):
            issues.append(
                _issue(
                    "count_invalid_membership_flag",
                    "1proof.gold_operand_claim_ids",
                    "Every membership operand must carry a boolean included flag.",
                )
            )
        dimensions = {
            (
                _norm_text(claim.get("set_id")),
                _norm_text(claim.get("condition")),
            )
            for claim in claims
        }
        members = {
            _norm_text(
                (claim.get("canonical_support_claim") or {}).get("subject")
                if isinstance(claim.get("canonical_support_claim"), Mapping)
                else ""
            )
            for claim in claims
        }
        if (
            len(dimensions) != 1
            or any(not item for item in next(iter(dimensions), ()))
            or "" in members
            or len(members) != len(claims)
        ):
            issues.append(
                _issue(
                    "count_membership_domain_mismatch",
                    "1proof.gold_operand_claim_ids",
                    "Count operands must be distinct members of one set and condition.",
                )
            )
        expected_included = [
            claim_id
            for claim_id, included in included_rows
            if included is True
        ]
        declared_included = _string_ids(derivation.get("included_claim_ids"))
        if set(declared_included) != set(expected_included) or len(
            declared_included
        ) != len(expected_included):
            issues.append(
                _issue(
                    "count_included_claim_ids_mismatch",
                    "1proof.derivation.included_claim_ids",
                    "included_claim_ids disagree with typed membership truth.",
                    expected_claim_ids=expected_included,
                )
            )
        expected = len(expected_included)
    if expected is not None:
        computed = _numeric_scalar(derivation.get("computed_value"))
        gold = _numeric_scalar(qa.get("gold_answer_text"))
        if computed is None or abs(computed - float(expected)) > 1e-9:
            issues.append(
                _issue(
                    "derivation_computed_value_mismatch",
                    "1proof.derivation.computed_value",
                    "computed_value disagrees with independently recomputed truth.",
                    expected=expected,
                )
            )
        if gold is None or abs(gold - float(expected)) > 1e-9:
            issues.append(
                _issue(
                    "gold_value_claim_mismatch",
                    "1gold_answer_text",
                    "gold answer disagrees with independently recomputed truth.",
                    expected=expected,
                )
            )
    return issues


def validate_qa_semantics(
    claim_ir: Mapping[str, Any],
    qa: Mapping[str, Any],
    dialogue: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an immutable, per-QA semantic validation report."""

    claims = _claim_list(claim_ir)
    by_id = _claims_by_id(claim_ir)
    answer_ids = _answer_claim_ids(qa)
    binder_ids = _binder_claim_ids(qa)
    issues: list[dict[str, Any]] = []

    missing_answer_ids = [item for item in answer_ids if item not in by_id]
    if missing_answer_ids:
        issues.append(
            _issue(
                "answer_claim_missing",
                "$.proof.derivation.answer_claim_ids",
                "answer-bearing claim ids are absent from the supplied Claim IR",
                missing_claim_ids=missing_answer_ids,
            )
        )
    answer_claims = [by_id[item] for item in answer_ids if item in by_id]
    binder_claims = [by_id[item] for item in binder_ids if item in by_id]
    if not answer_claims and str(_derivation(qa).get("rule") or "") not in {
        "absolute_difference", "count_distinct", "compare_values"
    }:
        issues.append(
            _issue(
                "answer_claim_unresolved",
                "$.proof.derivation",
                "the QA has no resolvable answer-bearing claim",
            )
        )

    issues.extend(_derived_truth_issues(qa, by_id))
    issues.extend(_gold_alignment_issues(qa, by_id))
    for index, claim in enumerate(answer_claims):
        issues.extend(_value_shape_issues(claim, path=f"$.answer_claims[{index}].value"))
    issues.extend(_role_issues(qa, answer_claims, binder_claims))
    issues.extend(_status_issues(qa, answer_claims, binder_claims))
    issues.extend(_ambiguity_issues(qa, claims, answer_claims, dialogue))

    counts = collections.Counter(item["code"] for item in issues)
    return {
        "schema_version": SCHEMA_VERSION,
        "qa_id": str(qa.get("qa_id") or ""),
        "pass": not issues,
        "issue_count": len(issues),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": issues,
    }


def validate_multiqa_semantics(
    claim_ir: Mapping[str, Any],
    qas: Sequence[Mapping[str, Any]],
    dialogue: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate siblings independently without filtering or mutating them."""

    reports = [validate_qa_semantics(claim_ir, qa, dialogue) for qa in qas]
    accepted = [item["qa_id"] for item in reports if item["pass"]]
    rejected = [item["qa_id"] for item in reports if not item["pass"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": not rejected,
        "qa_count": len(reports),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_qa_ids": accepted,
        "rejected_qa_ids": rejected,
        "reports": reports,
    }


def validate_sample_semantics(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Convenience adapter for a stored Claim-first clean sample."""

    bundle = sample.get("claim_first_multiqa_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("sample.claim_first_multiqa_bundle must be a mapping")
    claim_ir = bundle.get("claim_ir")
    qas = bundle.get("qas")
    if not isinstance(claim_ir, Mapping) or not isinstance(qas, list):
        raise ValueError("Claim-first bundle must contain claim_ir and qas")
    return validate_multiqa_semantics(claim_ir, qas, sample.get("dialogue"))


__all__ = [
    "PREDICATE_TYPE_FAMILIES",
    "SCHEMA_VERSION",
    "predicate_type_family",
    "validate_multiqa_semantics",
    "validate_qa_semantics",
    "validate_sample_semantics",
]
