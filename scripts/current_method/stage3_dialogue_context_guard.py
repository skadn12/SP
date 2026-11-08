"""Independent guard for API-authored Stage3 dialogue context.

The Truth-first dialogue API owns only the context around locally frozen
support clauses.  This module validates that context before it is combined
with those clauses.  It intentionally does not import the claim compiler or
the claim validator: frozen claim semantics are consumed as plain JSON.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage3_dialogue_context_guard_v2"
NEAR_DUPLICATE_LOOKBACK_SENTENCES = 24

_ALLOWED_TURN_KEYS = frozenset(
    {"turn_slot_id", "lead_in_sentences", "follow_up_sentences"}
)
_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)
_INTERNAL_ID_RE = re.compile(
    r"\b(?:(?:clm|evt|tfd|rel|node)[:_-][a-z0-9_.:-]+|"
    r"(?:event|claim|turn[_ -]?slot)[_ -]?id\s*[:=]?\s*[a-z0-9_.:-]+|"
    r"TS\d{2,}|E\d{1,})\b",
    flags=re.IGNORECASE,
)
_INTERNAL_META_RE = re.compile(
    r"\b(?:frozen\s+(?:claim|graph|truth)|claim\s+id|event\s+id|turn\s+slot|"
    r"graph\s+snapshot|schema\s+version|reasoning\s+path|gold\s+answer|"
    r"validator|validation\s+gate|dataset\s+row|system\s+prompt)\b",
    flags=re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|neither|nor|without|cannot|can't|"
    r"didn't|doesn't|isn't|wasn't|won't|denied|refused)\b",
    flags=re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:whether|might|may|could|possibly|perhaps|uncertain|unsure|"
    r"wonder(?:ed|ing)?|question(?:ed|ing)?|hypothetical)\b",
    flags=re.IGNORECASE,
)
_STRONG_CONDITIONAL_RE = re.compile(
    r"\b(?:if|only\s+if|unless|provided\s+that|assuming\s+that|"
    r"as\s+long\s+as|whenever)\b",
    flags=re.IGNORECASE,
)
_NARRATIVE_WHEN_RE = re.compile(
    r"\b(?:remember|recall|recalled|remembered|discussed|mentioned|"
    r"talked\s+about|think\s+back\s+to)\b(?:\s+\w+){0,5}\s+when\b",
    flags=re.IGNORECASE,
)
_OUTER_NEGATION_RE = re.compile(
    r"\b(?:it\s+is\s+not\s+true\s+that|it\s+was\s+not\s+true\s+that|"
    r"it\s+is\s+false\s+that|(?:deny|denied|dispute|disputed)\s+that|"
    r"there\s+is\s+no\s+evidence\s+that)\b",
    flags=re.IGNORECASE,
)
_COMPATIBLE_NONACTIVE_RE = re.compile(
    r"\b(?:no\s+longer|not\s+(?:part|included)\s+(?:of|in)\s+(?:the\s+)?"
    r"current|not\s+currently|used\s+to|formerly|previously|once)\b",
    flags=re.IGNORECASE,
)
_EVALUATIVE_COMPLEMENT_RE = re.compile(
    r"\b(?:solid|good|great|best|better|logical|practical|sensible|wise|"
    r"right|wrong|crucial|key|significant|important)\s+"
    r"(?:choice|fit|decision|idea|option|detail|element|part|development)\b",
    flags=re.IGNORECASE,
)
_UNCONDITIONAL_RE = re.compile(
    r"\b(?:unconditionally|regardless\s+of|under\s+all\s+circumstances|"
    r"without\s+(?:any\s+)?conditions?|no\s+conditions?\s+(?:apply|required))\b",
    flags=re.IGNORECASE,
)
_UNRESTRICTED_RE = re.compile(
    r"\b(?:anyone|everyone|publicly|unrestricted|without\s+restrictions?|"
    r"without(?:\s+\w+){0,4}\s+access\s+restrictions?|"
    r"no\s+access\s+restrictions?|open\s+to\s+all)\b",
    flags=re.IGNORECASE,
)
_GENERIC_FILLER_RE = re.compile(
    r"^(?:okay|ok|sure|sounds\s+good|that\s+makes\s+sense|i\s+agree|"
    r"thanks\s+for\s+sharing|let(?:'s|\s+us)\s+continue)[.!?]*$",
    flags=re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

_PREDICATE_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"destination", "destined", "headed"}),
    frozenset({"schedule", "scheduled", "scheduling"}),
    frozenset({"cause", "caused", "causing", "trigger", "triggered", "lead", "led", "result"}),
    frozenset({"depend", "depends", "depended", "require", "requires", "rely", "relies"}),
    frozenset({"transition", "transitions", "transitioned", "update", "updates", "updated", "change", "changed", "become", "became"}),
    frozenset({"member", "membership", "belong", "belongs", "include", "includes"}),
    frozenset({"view", "views", "viewed", "see", "sees", "access", "visible"}),
    frozenset({"edit", "edits", "edited", "modify", "modifies", "modified"}),
    frozenset({"own", "owns", "owned", "owner", "possess", "possesses"}),
    frozenset({"compare", "compares", "difference", "differ", "higher", "lower", "more", "less"}),
)

_GENERIC_PREDICATE_ROLE_TOKENS = frozenset(
    {"event", "item", "value", "status", "type", "class", "level", "details"}
)
_TRANSITION_CUE_TERMS = frozenset(
    {
        "transition",
        "transitions",
        "transitioned",
        "update",
        "updates",
        "updated",
        "change",
        "changes",
        "changed",
        "become",
        "becomes",
        "became",
        "switch",
        "switches",
        "switched",
        "move",
        "moves",
        "moved",
    }
)
_ASSERTIVE_FRAME_TERMS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "became",
        "becomes",
        "remains",
        "selected",
        "assigned",
        "chosen",
        "named",
        "listed",
        "set",
    }
)
_NON_VALUE_COMPLEMENT_RE = re.compile(
    r"\b(?:still\s+)?(?:under\s+(?:discussion|review|consideration)|"
    r"undecided|unknown|unclear|unresolved|pending|open\s+question|"
    r"a\s+question|being\s+(?:discussed|reviewed)|"
    r"to\s+be\s+(?:decided|determined|discussed|confirmed))\b",
    flags=re.IGNORECASE,
)
_NON_VALUE_TOKENS = frozenset(
    {
        "anything",
        "consideration",
        "details",
        "discussion",
        "issue",
        "matter",
        "nothing",
        "pending",
        "planners",
        "planning",
        "question",
        "review",
        "something",
        "topic",
        "unclear",
        "undecided",
        "unknown",
        "unresolved",
    }
)
_RELATION_TAIL_TERMS = frozenset(
    {"according", "although", "because", "but", "during", "however", "while", "which"}
)
_SLOT_LINKERS = frozenset({"for", "of", "as"})
_ASYMMETRIC_PREDICATE_TERMS = frozenset(
    {
        "cause",
        "caused",
        "causing",
        "trigger",
        "triggered",
        "lead",
        "led",
        "result",
        "depend",
        "depends",
        "depended",
        "require",
        "requires",
        "rely",
        "relies",
    }
)
_VERBAL_ASSERTION_TERMS = frozenset(
    {
        "assign",
        "assigned",
        "cause",
        "caused",
        "belong",
        "belongs",
        "depend",
        "depends",
        "edit",
        "edited",
        "include",
        "included",
        "own",
        "owned",
        "schedule",
        "scheduled",
        "select",
        "selected",
        "view",
        "viewed",
        "rely",
        "relies",
        "require",
        "requires",
    }
)

_STATUS_MARKERS: dict[str, re.Pattern[str]] = {
    "active": re.compile(
        r"\b(?:active|current|currently|still\s+(?:active|valid|applies)|"
        r"in\s+effect|remains?\s+valid|confirmed|verified)\b",
        flags=re.IGNORECASE,
    ),
    "canceled": re.compile(
        r"\b(?:canceled|cancelled|called\s+off|withdrawn|did\s+not\s+take\s+effect)\b",
        flags=re.IGNORECASE,
    ),
    "expired": re.compile(
        r"\b(?:expired|lapsed)\b", flags=re.IGNORECASE
    ),
    "outdated": re.compile(
        r"\b(?:outdated|obsolete|superseded)\b",
        flags=re.IGNORECASE,
    ),
    "reported": re.compile(
        r"\b(?:reported|alleged|rumou?red|unconfirmed|hearsay)\b",
        flags=re.IGNORECASE,
    ),
    "belief": re.compile(
        r"\b(?:believes?|uncertain|unsure|might|may\s+be)\b",
        flags=re.IGNORECASE,
    ),
    "inactive": re.compile(
        r"\b(?:no\s+longer\s+(?:valid|applies|current)|not\s+currently\s+active)\b",
        flags=re.IGNORECASE,
    ),
}


def _issue(code: str, path: str, message: str, **details: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"code": code, "path": path, "message": message}
    if details:
        row["details"] = details
    return row


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _tokens(value: Any) -> list[str]:
    return [token.casefold().strip("'-") for token in _WORD_RE.findall(str(value or ""))]


def _content_tokens(value: Any) -> list[str]:
    return [token for token in _tokens(value) if token not in _STOPWORDS and len(token) > 1]


def _flatten_values(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        result: list[Any] = []
        for child in value.values():
            result.extend(_flatten_values(child))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for child in value:
            result.extend(_flatten_values(child))
        return result
    return [value]


def _surface_present(sentence_tokens: set[str], value: Any) -> bool:
    values = _flatten_values(value)
    if not values:
        return False
    for item in values:
        required = set(_content_tokens(item))
        if required and required.issubset(sentence_tokens):
            return True
        if isinstance(item, (int, float)) and str(item) in sentence_tokens:
            return True
    return False


def _surface_position(sentence: str, value: Any) -> int | None:
    sentence_tokens = _tokens(sentence)
    positions: list[int] = []
    for item in _flatten_values(value):
        required = _content_tokens(item)
        if not required:
            continue
        found = [sentence_tokens.index(token) for token in required if token in sentence_tokens]
        if len(found) == len(set(required)):
            positions.append(min(found))
    return min(positions) if positions else None


def _same_semantic_surface(left: Any, right: Any) -> bool:
    left_tokens = set(_content_tokens(left))
    right_tokens = set(_content_tokens(right))
    return bool(left_tokens and left_tokens == right_tokens)


def _predicate_terms(predicate: Any) -> set[str]:
    raw = set(_content_tokens(str(predicate or "").replace("_", " ").replace("-", " ")))
    if not raw:
        return set()
    # Compound IR slots often end in a generic carrier such as ``event`` or
    # ``item``.  Letting that carrier match by itself made an ordinary phrase
    # like "reschedule the event" look like the typed ``scheduled_event``
    # predicate.  Keep the carrier only when it is the whole predicate.
    core = raw - _GENERIC_PREDICATE_ROLE_TOKENS
    if not core:
        core = raw
    expanded = set(core)
    for family in _PREDICATE_FAMILIES:
        if core & family:
            expanded.update(family)
    return expanded


def _predicate_present(sentence_tokens: set[str], predicate: Any) -> bool:
    terms = _predicate_terms(predicate)
    return bool(terms and sentence_tokens.intersection(terms))


def _predicate_position(sentence: str, predicate: Any) -> int | None:
    terms = _predicate_terms(predicate)
    return next(
        (index for index, token in enumerate(_tokens(sentence)) if token in terms),
        None,
    )


def _claim_type(claim: Mapping[str, Any], support: Mapping[str, Any]) -> str:
    raw = re.sub(
        r"[^a-z0-9]+",
        "_",
        _text(claim.get("claim_type") or claim.get("claim_kind")).casefold(),
    ).strip("_")
    if raw and raw != "relation":
        return raw
    predicate = re.sub(
        r"[^a-z0-9]+", "_", _text(support.get("predicate")).casefold()
    ).strip("_")
    if predicate == "transitions_to" or {
        "from_status",
        "to_status",
    }.issubset(support):
        return "status_transition"
    if "cause_event" in support or "effect_event" in support:
        return "cause"
    if "subject_event" in support or "object_events" in support:
        return "dependency"
    if "included" in support:
        return "membership"
    if "owner" in support or (
        isinstance(support.get("scope"), Mapping)
        and "permission" in support.get("scope", {})
    ):
        return "visibility"
    return "node"


def _relation_positions(
    sentence: str, semantics: Mapping[str, Any]
) -> tuple[int | None, int | None, list[int | None]]:
    return (
        _surface_position(sentence, semantics["subject"]),
        _predicate_position(sentence, semantics["predicate"]),
        [_surface_position(sentence, item) for item in semantics["objects"]],
    )


def _relation_window(
    sentence: str,
    positions: tuple[int | None, int | None, list[int | None]],
    *,
    trailing_tokens: int = 8,
) -> str:
    subject_position, predicate_position, object_positions = positions
    present = [
        int(position)
        for position in (subject_position, predicate_position, *object_positions)
        if position is not None
    ]
    if not present:
        return ""
    tokens = _tokens(sentence)
    start = min(present)
    end = min(len(tokens), max(present) + 1 + trailing_tokens)
    return " ".join(tokens[start:end])


def _has_transition_cue(sentence: str) -> bool:
    return bool(set(_tokens(sentence)).intersection(_TRANSITION_CUE_TERMS))


def _alternate_object_asserted(
    sentence: str,
    semantics: Mapping[str, Any],
    subject_position: int | None,
    predicate_position: int | None,
) -> bool:
    """Conservatively recognize a typed slot assertion with another value.

    This is intentionally a frame test, not the previous subject+predicate
    shortcut.  A subject and slot may be discussed harmlessly; rejection
    requires an assertive copular/assignment frame that supplies a complement.
    """

    if subject_position is None or predicate_position is None:
        return False
    tokens = _tokens(sentence)
    if not tokens:
        return False

    def concrete_candidate(raw: Sequence[str]) -> bool:
        candidate = list(raw)
        for index, token in enumerate(candidate):
            if token in _RELATION_TAIL_TERMS:
                candidate = candidate[:index]
                break
        surface = " ".join(candidate)
        if not surface or _NON_VALUE_COMPLEMENT_RE.search(surface):
            return False
        known = set(_content_tokens(semantics["subject"])) | _predicate_terms(
            semantics["predicate"]
        )
        content = [
            token
            for token in candidate
            if token not in known
            and token not in _STOPWORDS
            and token not in _ASSERTIVE_FRAME_TERMS
            and token not in _SLOT_LINKERS
            and token not in _NON_VALUE_TOKENS
            and len(token) > 1
        ]
        return bool(content)

    # Slot-nominal forward frame: "S's destination is VALUE" or
    # "S's destination at VALUE".
    for frame_index in range(predicate_position + 1, min(len(tokens), predicate_position + 5)):
        if tokens[frame_index] in _ASSERTIVE_FRAME_TERMS | {"at", "to"}:
            if tokens[frame_index] in {"at", "to"} and frame_index != predicate_position + 1:
                continue
            if concrete_candidate(tokens[frame_index + 1 : frame_index + 8]):
                return True
            break

    # Object-first slot frame: "VALUE is the destination for S".
    for frame_index in range(max(0, predicate_position - 4), predicate_position):
        if tokens[frame_index] in _ASSERTIVE_FRAME_TERMS:
            if concrete_candidate(tokens[max(0, frame_index - 5) : frame_index]):
                return True
            break

    # Verbal action frame: "S selected/viewed/scheduled VALUE".  Require a
    # concrete complement after the typed predicate; merely discussing the
    # action or its generic carrier does not suffice.
    if (
        subject_position < predicate_position
        and tokens[predicate_position] in _VERBAL_ASSERTION_TERMS
        and concrete_candidate(tokens[predicate_position + 1 : predicate_position + 7])
    ):
        return True
    return False


def _answer_bearing_match(
    sentence: str,
    sentence_tokens: set[str],
    semantics: Mapping[str, Any],
) -> tuple[bool, str, tuple[int | None, int | None, list[int | None]]]:
    positions = _relation_positions(sentence, semantics)
    subject_position, predicate_position, object_positions = positions
    subject_present = subject_position is not None
    predicate_present = predicate_position is not None
    object_present = bool(semantics["objects"]) and all(
        position is not None for position in object_positions
    )
    if subject_present and predicate_present and object_present:
        return True, "complete_typed_roles", positions

    if semantics["claim_type"] == "status_transition":
        to_value = semantics.get("to_value")
        if (
            subject_present
            and _has_transition_cue(sentence)
            and _surface_present(sentence_tokens, to_value)
        ):
            return True, "transition_to_value", positions
        return False, "", positions

    if (
        semantics["claim_type"]
        in {"node", "visibility", "membership", "cause", "dependency"}
        and subject_present
        and predicate_present
        and _alternate_object_asserted(
            sentence, semantics, subject_position, predicate_position
        )
    ):
        return True, "alternate_typed_object", positions
    return False, "", positions


def _normalized_polarity(value: Any, predicate: Any) -> str:
    if value is False:
        return "negative"
    if value is True:
        return "positive"
    normalized = _text(value).casefold()
    if normalized in {"negative", "negated", "false", "-1"}:
        return "negative"
    if str(predicate or "").casefold().startswith("not_"):
        return "negative"
    return "positive"


def _status_bucket(value: Any) -> str:
    normalized = re.sub(r"[\W_]+", "_", _text(value).casefold()).strip("_")
    if normalized in {"canceled", "cancelled"}:
        return "canceled"
    if normalized in {"expired", "lapsed"}:
        return "expired"
    if normalized in {"outdated", "superseded", "obsolete"}:
        return "outdated"
    if normalized in {"report", "reported", "unconfirmed", "alleged"}:
        return "reported"
    if normalized in {"belief", "uncertain", "proposed"}:
        return "belief"
    if normalized in {"scope_limited", "restricted"}:
        return "scope_limited"
    return "active"


def _sentence_statuses(sentence: str) -> set[str]:
    return {name for name, pattern in _STATUS_MARKERS.items() if pattern.search(sentence)}


def _relation_negated(
    sentence: str, subject: Any, predicate: Any, objects: Sequence[Any]
) -> bool:
    """Return whether negation occurs inside the matched semantic relation."""

    subject_position = _surface_position(sentence, subject)
    predicate_position = _predicate_position(sentence, predicate)
    object_positions = [_surface_position(sentence, item) for item in objects]
    if (
        subject_position is None
        or predicate_position is None
        or not object_positions
        or any(position is None for position in object_positions)
    ):
        return False
    relation_start = min(
        subject_position, predicate_position, *(int(position) for position in object_positions)
    )
    relation_end = max(
        subject_position, predicate_position, *(int(position) for position in object_positions)
    )
    tokens = _tokens(sentence)
    relation_window = " ".join(tokens[relation_start : relation_end + 1])
    return bool(_NEGATION_RE.search(relation_window))


def _relation_statuses(
    sentence: str,
    positions: tuple[int | None, int | None, list[int | None]],
) -> set[str]:
    """Return status markers that govern the matched relation, not the turn.

    Speaker stance about whether a choice is *good* is not epistemic status of
    the embedded fact.  Conversely, a prefix such as "reportedly" or "I think
    that" does qualify the proposition and must remain visible to the gate.
    """

    window = _relation_window(sentence, positions)
    statuses = _sentence_statuses(window)
    evaluation = _EVALUATIVE_COMPLEMENT_RE.search(sentence)
    if evaluation is not None:
        evaluation_prefix = sentence[max(0, evaluation.start() - 96) : evaluation.start()]
        if re.search(
            r"\b(?:i\s+(?:think|believe)|might|may|could|seems?|appears?)\b",
            evaluation_prefix,
            re.IGNORECASE,
        ):
            statuses.discard("belief")
    if _COMPATIBLE_NONACTIVE_RE.search(window):
        statuses.add("inactive")
        # ``current`` inside "not part of the current plan" is in the scope
        # of negation and therefore cannot make an inactive fact active.
        statuses.discard("active")

    subject_position, predicate_position, object_positions = positions
    present = [
        int(position)
        for position in (subject_position, predicate_position, *object_positions)
        if position is not None
    ]
    tokens = _tokens(sentence)
    if present:
        start = min(present)
        prefix = " ".join(tokens[max(0, start - 6) : start])
        if _STATUS_MARKERS["reported"].search(prefix):
            statuses.add("reported")
        if _STATUS_MARKERS["belief"].search(prefix):
            statuses.add("belief")
        if re.search(r"\bi\s+(?:think|believe)\b", prefix, re.IGNORECASE):
            if not _EVALUATIVE_COMPLEMENT_RE.search(sentence):
                statuses.add("belief")

    # "the canceled choice was confirmed" confirms the cancellation; it does
    # not reactivate it.  Literal active/current language still wins.
    if statuses.intersection({"canceled", "expired", "outdated"}):
        strong_active = re.search(
            r"\b(?:active|current|currently|in\s+effect|remains?\s+valid)\b",
            window,
            re.IGNORECASE,
        )
        if strong_active is None:
            statuses.discard("active")
    return statuses


def _has_governing_condition(
    sentence: str,
    positions: tuple[int | None, int | None, list[int | None]],
) -> bool:
    if _STRONG_CONDITIONAL_RE.search(sentence):
        return True
    for match in re.finditer(r"\bwhen\b", sentence, re.IGNORECASE):
        prefix = sentence[: match.end()]
        if _NARRATIVE_WHEN_RE.search(prefix):
            continue
        # A non-narrative when-clause before or after the relation is a real
        # temporal condition.  The role positions need not be exposed here;
        # their presence was already required by the caller.
        if any(
            position is not None
            for position in (positions[0], positions[1], *positions[2])
        ):
            return True
    return False


def _direction_contradiction(
    sentence: str,
    semantics: Mapping[str, Any],
    positions: tuple[int | None, int | None, list[int | None]],
) -> bool:
    subject_position, predicate_position, object_positions = positions
    if (
        subject_position is None
        or predicate_position is None
        or not object_positions
        or any(position is None for position in object_positions)
    ):
        return False
    # Self-dependencies do not expose a lexical direction between two entity
    # surfaces, so token order cannot prove reversal.
    if any(int(position) == subject_position for position in object_positions):
        return False
    tokens = _tokens(sentence)
    predicate_terms = _predicate_terms(semantics["predicate"])
    asymmetric = bool(predicate_terms.intersection(_ASYMMETRIC_PREDICATE_TERMS))
    all_objects_before = all(
        int(position) < predicate_position < subject_position
        for position in object_positions
    )
    if all_objects_before:
        between = tokens[predicate_position + 1 : subject_position + 1]
        # "O was caused by S" is a passive realization of S -> O.  Slot
        # nominals use "O is the destination/partner for S" in the same way.
        if "by" in between:
            return False
        if semantics["claim_type"] == "node" and set(between).intersection(
            _SLOT_LINKERS
        ):
            return False
        return asymmetric or semantics["claim_type"] != "node"

    all_objects_after = all(
        subject_position < predicate_position < int(position)
        for position in object_positions
    )
    if all_objects_after and asymmetric:
        first_object = min(int(position) for position in object_positions)
        # "S was caused by O" reverses frozen S -> O even though the surface
        # tokens are in subject/predicate/object order.
        return "by" in tokens[predicate_position + 1 : first_object + 1]
    return False


def _claim_semantics(claim: Mapping[str, Any]) -> dict[str, Any] | None:
    support = claim.get("canonical_support_claim")
    if not isinstance(support, Mapping):
        return None
    objects = support.get("objects", support.get("object", support.get("value")))
    if not isinstance(objects, list):
        objects = [] if objects is None else [objects]
    claim_type = _claim_type(claim, support)
    filtered_objects = [item for item in objects if item is not None and item != ""]
    return {
        "claim_id": _text(claim.get("claim_id")),
        "claim_type": claim_type,
        "subject": support.get("subject", claim.get("subject")),
        "predicate": support.get("predicate", claim.get("predicate")),
        "objects": filtered_objects,
        "slot": support.get("slot", support.get("predicate")),
        "from_value": filtered_objects[0]
        if claim_type == "status_transition" and filtered_objects
        else None,
        "to_value": filtered_objects[1]
        if claim_type == "status_transition" and len(filtered_objects) > 1
        else None,
        "polarity": support.get("polarity", claim.get("polarity")),
        "status": support.get("status", claim.get("status")),
        "condition": support.get("condition", claim.get("condition")),
        "scope": support.get("scope", claim.get("scope")),
    }


def _claim_contradiction_issues(
    sentence: str,
    path: str,
    claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sentence_tokens = set(_tokens(sentence))
    issues: list[dict[str, Any]] = []
    answer_matches: list[tuple[str, str, tuple[str, str]]] = []
    hedged = bool(_HEDGE_RE.search(sentence))

    for raw_claim in claims:
        semantics = _claim_semantics(raw_claim)
        if semantics is None:
            continue
        answer_bearing, match_kind, positions = _answer_bearing_match(
            sentence, sentence_tokens, semantics
        )
        subject_position, predicate_position, object_positions = positions
        subject_present = subject_position is not None
        predicate_present = predicate_position is not None
        object_present = bool(semantics["objects"]) and all(
            position is not None for position in object_positions
        )
        if answer_bearing:
            relation_key = (
                " ".join(_content_tokens(semantics["subject"])),
                " ".join(sorted(_predicate_terms(semantics["predicate"]))),
            )
            answer_matches.append(
                (semantics["claim_id"], match_kind, relation_key)
            )

        expected_status = _status_bucket(semantics["status"])
        compatible_nonactive = (
            expected_status in {"canceled", "expired", "outdated"}
            and _COMPATIBLE_NONACTIVE_RE.search(sentence) is not None
        )
        outer_negation = bool(
            subject_present
            and object_present
            and _OUTER_NEGATION_RE.search(sentence)
        )
        paraphrased_surface_negation = bool(
            subject_present
            and object_present
            and not predicate_present
            and not any(
                _same_semantic_surface(semantics["subject"], item)
                for item in semantics["objects"]
            )
            and _NEGATION_RE.search(sentence)
            and not compatible_nonactive
        )
        if outer_negation or paraphrased_surface_negation:
            issues.append(
                _issue(
                    "frozen_claim_surface_contradiction",
                    path,
                    "API-authored context negates a frozen subject/object fact even when the predicate is paraphrased.",
                    claim_id=semantics["claim_id"] or None,
                )
            )
        if not (subject_present and predicate_present and object_present):
            continue

        claim_id = semantics["claim_id"] or None
        expected_polarity = _normalized_polarity(
            semantics["polarity"], semantics["predicate"]
        )
        negated = _relation_negated(
            sentence,
            semantics["subject"],
            semantics["predicate"],
            semantics["objects"],
        )
        if (
            expected_polarity == "positive"
            and negated
            and not compatible_nonactive
        ) or (
            expected_polarity == "negative" and not negated and not hedged
        ):
            issues.append(
                _issue(
                    "frozen_claim_polarity_contradiction",
                    path,
                    "API-authored context explicitly reverses a frozen claim's polarity.",
                    claim_id=claim_id,
                    expected_polarity=expected_polarity,
                )
            )

        relation_statuses = _relation_statuses(sentence, positions)
        conflicting_statuses: set[str] = set()
        if expected_status == "active":
            conflicting_statuses = relation_statuses.intersection(
                {
                    "canceled",
                    "expired",
                    "outdated",
                    "reported",
                    "belief",
                    "inactive",
                }
            )
        elif expected_status in {"canceled", "expired", "outdated"}:
            conflicting_statuses = relation_statuses.intersection(
                {"active", "reported", "belief"}
            )
            conflicting_statuses.update(
                relation_statuses.intersection(
                    {"canceled", "expired", "outdated"} - {expected_status}
                )
            )
        elif expected_status in {"reported", "belief"}:
            conflicting_statuses = relation_statuses.intersection(
                {"active", "canceled", "expired", "outdated"}
            )
        elif expected_status == "scope_limited" and _UNRESTRICTED_RE.search(sentence):
            conflicting_statuses = {"unrestricted"}
        if conflicting_statuses:
            issues.append(
                _issue(
                    "frozen_claim_status_contradiction",
                    path,
                    "API-authored context assigns a status incompatible with frozen truth.",
                    claim_id=claim_id,
                    expected_status=expected_status,
                    conflicting_statuses=sorted(conflicting_statuses),
                )
            )

        condition_values = _flatten_values(semantics["condition"])
        condition_present = _surface_present(sentence_tokens, semantics["condition"])
        governing_condition = _has_governing_condition(sentence, positions)
        if condition_values:
            if _UNCONDITIONAL_RE.search(sentence):
                issues.append(
                    _issue(
                        "frozen_claim_condition_contradiction",
                        path,
                        "A conditional frozen claim is described as unconditional.",
                        claim_id=claim_id,
                    )
                )
            elif governing_condition and not condition_present:
                issues.append(
                    _issue(
                        "frozen_claim_condition_mismatch",
                        path,
                        "API-authored context supplies a different explicit condition.",
                        claim_id=claim_id,
                    )
                )
        elif governing_condition:
            issues.append(
                _issue(
                    "frozen_claim_unexpected_condition",
                    path,
                    "API-authored context adds a condition absent from frozen truth.",
                    claim_id=claim_id,
                )
            )

        scope_values = _flatten_values(semantics["scope"])
        if scope_values and _UNRESTRICTED_RE.search(sentence):
            issues.append(
                _issue(
                    "frozen_claim_scope_contradiction",
                    path,
                    "Scoped frozen truth is described as unrestricted.",
                    claim_id=claim_id,
                )
            )

        if _direction_contradiction(sentence, semantics, positions):
            issues.append(
                _issue(
                    "frozen_claim_direction_contradiction",
                    path,
                    "API-authored context reverses the frozen subject/object direction.",
                    claim_id=claim_id,
                )
            )

    if answer_matches:
        exact_relation_keys = {
            relation_key
            for _claim_id, kind, relation_key in answer_matches
            if kind != "alternate_typed_object"
        }
        answer_matches = [
            row
            for row in answer_matches
            if row[1] != "alternate_typed_object"
            or row[2] not in exact_relation_keys
        ]
        matched_claim_ids = list(
            dict.fromkeys(
                claim_id
                for claim_id, _kind, _relation_key in answer_matches
                if claim_id
            )
        )
        match_kinds = sorted(
            {kind for _claim_id, kind, _relation_key in answer_matches}
        )
        issues.insert(
            0,
            _issue(
                "api_authored_answer_bearing_relation",
                path,
                "API-authored context must not restate or replace a frozen answer-bearing relation.",
                claim_id=matched_claim_ids[0] if matched_claim_ids else None,
                matched_claim_ids=matched_claim_ids,
                match_kinds=match_kinds,
            ),
        )
    return issues


def _sentence_substance_issues(sentence: Any, path: str) -> list[dict[str, Any]]:
    if not isinstance(sentence, str) or not _text(sentence):
        return [_issue("context_sentence_invalid", path, "Context sentence must be non-empty text.")]
    words = _tokens(sentence)
    content = _content_tokens(sentence)
    issues: list[dict[str, Any]] = []
    if len(words) < 6 or len(content) < 4:
        issues.append(
            _issue(
                "context_sentence_not_substantive",
                path,
                "Every API-authored sentence needs at least six words and four content words.",
                word_count=len(words),
                content_word_count=len(content),
            )
        )
    if _GENERIC_FILLER_RE.fullmatch(_text(sentence)):
        issues.append(
            _issue(
                "generic_context_filler",
                path,
                "Generic acknowledgement is not substantive dialogue context.",
            )
        )
    if words and len(set(words)) / len(words) < 0.45:
        issues.append(
            _issue(
                "low_information_context_sentence",
                path,
                "Sentence repeats too few lexical items to be substantive.",
            )
        )
    if _INTERNAL_ID_RE.search(sentence):
        issues.append(
            _issue(
                "internal_identifier_leak",
                path,
                "API-authored context exposes an internal claim, event, or turn-slot identifier.",
            )
        )
    if _INTERNAL_META_RE.search(sentence):
        issues.append(
            _issue(
                "internal_meta_leak",
                path,
                "API-authored context exposes internal pipeline terminology.",
            )
        )
    return issues


def _normalized_sentence(value: str) -> str:
    return " ".join(_tokens(value))


def _near_duplicate(left: str, right: str) -> bool:
    left_normalized = _normalized_sentence(left)
    right_normalized = _normalized_sentence(right)
    if not left_normalized or not right_normalized:
        return False
    left_tokens = left_normalized.split()
    right_tokens = right_normalized.split()
    length_ratio = min(len(left_tokens), len(right_tokens)) / max(
        len(left_tokens), len(right_tokens)
    )
    if length_ratio < 0.72:
        return False
    sequence_ratio = SequenceMatcher(
        None, left_normalized, right_normalized, autojunk=False
    ).ratio()
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union = left_set | right_set
    jaccard = len(left_set & right_set) / len(union) if union else 0.0
    return sequence_ratio >= 0.9 or (length_ratio >= 0.8 and jaccard >= 0.82)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dynamic_internal_ids(
    slots: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]]
) -> set[str]:
    values = {
        _text(slot.get("turn_slot_id"))
        for slot in slots
        if _text(slot.get("turn_slot_id"))
    }
    for claim in claims:
        for key in ("claim_id", "subject_event_id"):
            if _text(claim.get(key)):
                values.add(_text(claim.get(key)))
        for key in ("object_event_ids", "source_event_ids"):
            raw = claim.get(key)
            if isinstance(raw, list):
                values.update(_text(item) for item in raw if _text(item))
    return values


def _dynamic_id_issue(
    sentence: str, path: str, identifiers: set[str]
) -> dict[str, Any] | None:
    folded = sentence.casefold()
    for identifier in identifiers:
        candidate = identifier.casefold()
        if candidate and re.search(
            rf"(?<![\w]){re.escape(candidate)}(?![\w])", folded
        ):
            return _issue(
                "dynamic_internal_identifier_leak",
                path,
                "API-authored context exposes an identifier from the frozen plan.",
                identifier=identifier,
            )
    return None


def validate_api_authored_context(
    discourse_plan: Mapping[str, Any], raw_flat_turns: Any
) -> dict[str, Any]:
    """Validate API-owned context without compiling or changing frozen support."""

    issues: list[dict[str, Any]] = []
    if not isinstance(discourse_plan, Mapping):
        return _report(
            [_issue("invalid_discourse_plan", "$.plan", "Discourse plan must be an object.")],
            turn_count=0,
            sentence_count=0,
            input_sha256=_canonical_sha256(raw_flat_turns),
        )
    slots = discourse_plan.get("turn_slots")
    claims = discourse_plan.get("semantic_claims")
    if not isinstance(slots, list) or not all(isinstance(row, Mapping) for row in slots):
        issues.append(
            _issue("invalid_plan_turn_slots", "$.plan.turn_slots", "Plan turn_slots must be an object array.")
        )
        slots = []
    if not isinstance(claims, list) or not all(isinstance(row, Mapping) for row in claims):
        issues.append(
            _issue("invalid_plan_semantic_claims", "$.plan.semantic_claims", "Plan semantic_claims must be an object array.")
        )
        claims = []
    slot_ids = [_text(row.get("turn_slot_id")) for row in slots]
    if any(not item for item in slot_ids) or len(slot_ids) != len(set(slot_ids)):
        issues.append(
            _issue("invalid_plan_turn_slot_ids", "$.plan.turn_slots", "Plan slot IDs must be non-empty and unique.")
        )
    dynamic_ids = _dynamic_internal_ids(slots, claims)

    if not isinstance(raw_flat_turns, list):
        return _report(
            issues
            + [_issue("flat_turns_not_array", "$.turns", "Raw flat turns must be an array.")],
            turn_count=0,
            sentence_count=0,
            input_sha256=_canonical_sha256(raw_flat_turns),
        )
    if len(raw_flat_turns) != len(slots):
        issues.append(
            _issue(
                "flat_turn_count_mismatch",
                "$.turns",
                "Raw flat turn count differs from the frozen slot count.",
                expected=len(slots),
                actual=len(raw_flat_turns),
            )
        )

    sentence_rows: list[tuple[str, str]] = []
    for index, raw_turn in enumerate(raw_flat_turns):
        path = f"$.turns[{index}]"
        if not isinstance(raw_turn, Mapping):
            issues.append(_issue("flat_turn_not_object", path, "Each flat turn must be an object."))
            continue
        extra = sorted(set(raw_turn) - _ALLOWED_TURN_KEYS)
        missing = sorted(_ALLOWED_TURN_KEYS - set(raw_turn))
        if extra:
            issues.append(
                _issue(
                    "unexpected_flat_turn_keys",
                    path,
                    "API-authored context contains fields outside its three-field schema.",
                    keys=extra,
                )
            )
        if missing:
            issues.append(
                _issue(
                    "missing_flat_turn_keys",
                    path,
                    "API-authored context is missing required fields.",
                    keys=missing,
                )
            )
        if index < len(slots) and _text(raw_turn.get("turn_slot_id")) != slot_ids[index]:
            issues.append(
                _issue(
                    "turn_slot_order_mismatch",
                    f"{path}.turn_slot_id",
                    "turn_slot_id must match the frozen slot at this array position.",
                    expected=slot_ids[index],
                    actual=_text(raw_turn.get("turn_slot_id")),
                )
            )
        for field, count_field in (
            ("lead_in_sentences", "lead_sentence_count"),
            ("follow_up_sentences", "follow_sentence_count"),
        ):
            sentences = raw_turn.get(field)
            expected_count = slots[index].get(count_field) if index < len(slots) else None
            if not isinstance(sentences, list):
                issues.append(
                    _issue(
                        "context_sentences_not_array",
                        f"{path}.{field}",
                        "Sentence fields must be arrays.",
                    )
                )
                continue
            if isinstance(expected_count, int) and len(sentences) != expected_count:
                issues.append(
                    _issue(
                        "context_sentence_count_mismatch",
                        f"{path}.{field}",
                        "Sentence count differs from the frozen slot contract.",
                        expected=expected_count,
                        actual=len(sentences),
                    )
                )
            for sentence_index, sentence in enumerate(sentences):
                sentence_path = f"{path}.{field}[{sentence_index}]"
                issues.extend(_sentence_substance_issues(sentence, sentence_path))
                if isinstance(sentence, str) and _text(sentence):
                    dynamic_issue = _dynamic_id_issue(
                        sentence, sentence_path, dynamic_ids
                    )
                    if dynamic_issue is not None:
                        issues.append(dynamic_issue)
                    sentence_rows.append((sentence_path, sentence))
                    issues.extend(
                        _claim_contradiction_issues(
                            sentence, sentence_path, claims  # type: ignore[arg-type]
                        )
                    )

    normalized_groups: dict[str, list[tuple[str, str]]] = {}
    for path, sentence in sentence_rows:
        normalized_groups.setdefault(_normalized_sentence(sentence), []).append(
            (path, sentence)
        )
    for rows in normalized_groups.values():
        if len(rows) < 2:
            continue
        first_path = rows[0][0]
        for duplicate_path, _sentence in rows[1:]:
            issues.append(
                _issue(
                    "exact_context_sentence_duplicate",
                    duplicate_path,
                    "API-authored context repeats an earlier sentence.",
                    first_path=first_path,
                )
            )

    for right_index, (right_path, right_sentence) in enumerate(sentence_rows):
        lookback_start = max(
            0, right_index - NEAR_DUPLICATE_LOOKBACK_SENTENCES
        )
        for left_path, left_sentence in sentence_rows[
            lookback_start:right_index
        ]:
            if _normalized_sentence(left_sentence) == _normalized_sentence(right_sentence):
                continue
            if _near_duplicate(left_sentence, right_sentence):
                issues.append(
                    _issue(
                        "near_context_sentence_duplicate",
                        right_path,
                        "API-authored context is a near duplicate of an earlier sentence.",
                        first_path=left_path,
                    )
                )
                break

    return _report(
        issues,
        turn_count=len(raw_flat_turns),
        sentence_count=len(sentence_rows),
        input_sha256=_canonical_sha256(raw_flat_turns),
    )


def _report(
    issues: Sequence[Mapping[str, Any]],
    *,
    turn_count: int,
    sentence_count: int,
    input_sha256: str,
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(row)) for row in issues]
    counts = Counter(_text(row.get("code")) or "unknown" for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": not rows,
        "issue_count": len(rows),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "turn_count": turn_count,
        "sentence_count": sentence_count,
        "near_duplicate_lookback_sentences": (
            NEAR_DUPLICATE_LOOKBACK_SENTENCES
        ),
        "input_sha256": input_sha256,
        "issues": rows,
    }


__all__ = [
    "NEAR_DUPLICATE_LOOKBACK_SENTENCES",
    "SCHEMA_VERSION",
    "validate_api_authored_context",
]
