"""Pure-local discourse planning and validation for Truth-first Stage3.

The graph and claim bundle are immutable inputs.  This module exposes only a
read-only semantic brief to a naturalisation API and accepts only dialogue
context sentences in return.  Exact support clauses are compiled locally and
inserted between those sentences.  In particular, the API
response has no field through which it can change events, slots, relations,
proofs, gold answers, or reasoning paths.

The planner deliberately keeps the generation surface bounded: at most two
coherent acts and at most two ordered beats per act.  Each beat is generated
as one contiguous segment so long dialogues can use natural short turns
without asking one response to carry the complete token budget.  The
validator independently re-hashes every frozen input and delegates semantic
quote checks to :mod:`stage3_claim_validator`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from scripts.current_method import stage3_claim_validator
from scripts.current_method import stage3_dialogue_context_guard
from scripts.current_method import stage3_truth_support


PLAN_SCHEMA_VERSION = "stage3_truth_first_discourse_plan_v3"
API_CONTRACT_SCHEMA_VERSION = "stage3_truth_first_dialogue_api_contract_v6"
API_RESPONSE_SCHEMA_VERSION = "stage3_truth_first_dialogue_api_response_v1"
API_RESPONSE_NORMALIZER_VERSION = "stage3_truth_first_dialogue_response_normalizer_v3"
API_SEGMENT_VALIDATION_VERSION = "stage3_truth_first_dialogue_segment_validation_v2"
API_SEGMENT_OVERFLOW_NORMALIZER_VERSION = (
    "stage3_truth_first_dialogue_segment_overflow_normalizer_v1"
)
VALIDATION_SCHEMA_VERSION = "stage3_truth_first_dialogue_validation_v1"
LIVE_CONTENT_BINDING_CONTRACT_SCHEMA_VERSION = (
    "stage3_truth_first_live_content_binding_contract_v2"
)
LIVE_CONTENT_BINDING_VALIDATION_SCHEMA_VERSION = (
    "stage3_truth_first_live_content_binding_validation_v1"
)
LIVE_SINGLE_SCENE_CONTRACT_SCHEMA_VERSION = (
    "stage3_truth_first_live_single_scene_contract_v2"
)
LIVE_SINGLE_SCENE_VALIDATION_SCHEMA_VERSION = (
    "stage3_truth_first_live_single_scene_validation_v1"
)

MAX_ACTS = 2
MAX_BEATS_PER_ACT = 2
MAX_SPEAKERS = 6
MAX_TURNS_PER_BEAT = 30
TOPIC_WINDOW_TURNS = 2
CONTEXT_SENTENCE_TARGET_WORDS = 34
CONTEXT_SENTENCE_COUNT_TOLERANCE = 1
CONTEXT_WORD_BUDGET_FACTOR = 0.80
NATURAL_CONTEXT_TARGET_WORDS_PER_TURN = 28
MIN_CONTEXT_SENTENCES_PER_TURN = 2
ACCEPTED_PREFIX_TAIL_TURNS = 8

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)
_TOKEN_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?][\"'\u2019\u201d)\]]*\s+(?=[A-Z0-9\"'\u2018\u201c(])"
)
_DIALOGUE_ATTRIBUTION_RE = re.compile(
    r"^[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?\s+"
    r"(?:asked|answered|added|continued|explained|noted|replied|said|told)\b"
)
_NONTERMINAL_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "e.g.",
        "etc.",
        "i.e.",
        "jr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "st.",
        "u.k.",
        "u.s.",
        "vs.",
    }
)
_META_RE = re.compile(
    r"\b(?:api|chatgpt|claim[_\s-]?id|event[_\s-]?id|frozen\s+graph|"
    r"graph\s+schema|json|prompt|reasoning\s+path|gold\s+answer|"
    r"training\s+(?:row|sample)|dataset|language\s+model)\b",
    flags=re.IGNORECASE,
)
_GENERIC_FILLER_RE = re.compile(
    r"^(?:okay|ok|sure|right|great|thanks|thank you|i agree|that makes sense|"
    r"let'?s continue|what else|anything else|good to know|interesting)"
    r"[.!?\s]*$",
    flags=re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "because", "been",
        "being", "by", "current", "for", "from", "has", "have", "in",
        "is", "it", "its", "of", "on", "or", "record", "status", "that",
        "the", "their", "this", "time", "to", "was", "were", "when",
        "with",
    }
)

_FORBIDDEN_RESPONSE_KEYS = frozenset(
    {
        "answer",
        "answer_role",
        "current_event_plan",
        "edge",
        "edges",
        "event",
        "events",
        "gold",
        "graph",
        "graph_delta",
        "membership",
        "proof",
        "proofs",
        "reasoning_path",
        "relation",
        "relations",
        "slot",
        "slots",
        "truth_signature",
    }
)

_OPERATION_PURPOSES: dict[str, tuple[str, str]] = {
    "temporal_state_update": (
        "Establish the earlier state and why the update matters.",
        "Develop the update and its current consequence without changing it.",
    ),
    "bridge": (
        "Establish the first binding in the shared scenario.",
        "Follow that binding to the second fact and resolve the two-hop link.",
    ),
    "comparison": (
        "Introduce the two comparable subjects in one shared context.",
        "Clarify the comparable values and practical contrast.",
    ),
    "visibility": (
        "Establish the resource, participants, and ordinary setting.",
        "Clarify who may do what, under which condition and scope.",
    ),
    "count": (
        "Introduce the candidate members and the counting condition.",
        "Resolve membership cases while keeping the explicit criterion fixed.",
    ),
    "cause_consequence": (
        "Establish the cause in its natural context.",
        "Develop the directed consequence and its practical aftermath.",
    ),
    "relation_lookup": (
        "Introduce the related people or objects in one scenario.",
        "Develop the fixed relation and its practical significance.",
    ),
}

_SUPPORT_DISCOURSE_MOVES: tuple[tuple[str, str], ...] = (
    (
        "situational_lead_and_practical_reaction",
        "Set the immediate situation before the local insertion, then react to its practical significance without restating it.",
    ),
    (
        "participant_motivation_and_social_response",
        "Develop a participant's non-factual motivation, then respond socially to the locally inserted detail without paraphrasing it.",
    ),
    (
        "shared_concern_and_next_step",
        "Frame a shared concern, then discuss a conversational next step after the local insertion without adding a competing fact.",
    ),
    (
        "topic_transition_and_implication",
        "Connect the previous exchange to this topic, then discuss a general implication without repeating the inserted relation.",
    ),
)

_CONTEXT_DISCOURSE_MOVES: tuple[tuple[str, str], ...] = (
    ("scene_development", "Develop a concrete sensory or social aspect of the shared setting."),
    ("participant_reaction", "Give a substantive personal reaction to the preceding conversational move."),
    ("motivation", "Explain a participant's motivation without asserting a new answer-bearing fact."),
    ("practical_tradeoff", "Discuss a practical tradeoff while leaving all frozen facts unstated."),
    ("nonfactual_clarification", "Ask or answer a clarification about priorities, not about the hidden fact."),
    ("shared_memory", "Connect to a general shared experience without introducing a fixed relation or value."),
    ("social_tension", "Develop mild social tension or hesitation without negating or qualifying frozen truth."),
    ("reconciliation", "Reconcile two conversational priorities without resolving any hidden answer."),
    ("implication", "Discuss a general practical implication of the topic without inventing a factual premise."),
    ("transition", "Bridge naturally from the previous topic beat to the next one."),
    ("next_step", "Propose a conversational or procedural next step without asserting its outcome."),
    ("closing_reflection", "Close the local exchange with a specific reflection grounded in the topic."),
)


class DiscourseContractError(ValueError):
    """Raised when frozen inputs cannot form a safe discourse contract."""


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 for plain JSON data."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DiscourseContractError(f"value is not plain JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _token(value: Any) -> str:
    return _TOKEN_RE.sub("_", _text(value).casefold()).strip("_")


def _issue(code: str, path: str, message: str, **details: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"code": code, "path": path, "message": message}
    if details:
        row["details"] = details
    return row


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _text(value)
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            result.append(text)
    return result


def _claims(claim_bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = claim_bundle.get("claims")
    if not isinstance(raw, list):
        nested = claim_bundle.get("claim_ir")
        raw = nested.get("claims") if isinstance(nested, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise DiscourseContractError("claim bundle requires a non-empty claims array")
    claims = [item for item in raw if isinstance(item, Mapping)]
    if len(claims) != len(raw):
        raise DiscourseContractError("every claim must be a JSON object")
    claim_ids = [_text(item.get("claim_id")) for item in claims]
    if any(not item for item in claim_ids) or len(set(claim_ids)) != len(claim_ids):
        raise DiscourseContractError("claim ids must be non-empty and unique")
    return claims


def _proof_required_ids(claim_bundle: Mapping[str, Any]) -> set[str]:
    candidates: list[Any] = []
    if isinstance(claim_bundle.get("proofs"), list):
        candidates.extend(claim_bundle["proofs"])
    elif isinstance(claim_bundle.get("proof"), Mapping):
        candidates.append(claim_bundle["proof"])
    result: set[str] = set()
    for proof in candidates:
        if not isinstance(proof, Mapping):
            continue
        for key in ("required_claim_ids", "reasoning_path_claim_ids"):
            values = proof.get(key)
            if isinstance(values, list):
                result.update(_text(item) for item in values if _text(item))
    return result


def _required_claims(claim_bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    claims = _claims(claim_bundle)
    proof_ids = _proof_required_ids(claim_bundle)
    required = [
        claim
        for claim in claims
        if _text(claim.get("claim_id")) in proof_ids
        or claim.get("qa_candidate") is True
        or claim.get("required_for_dialogue") is True
        or claim.get("answer_bearing") is True
        or claim.get("qa_eligible") is True
    ]
    if not required:
        raise DiscourseContractError(
            "claim bundle has no proof-required or dialogue-required semantic claims"
        )
    for index, claim in enumerate(required):
        claim_issues = stage3_claim_validator.validate_claim(
            claim, path=f"$.claim_bundle.claims[{index}]"
        )
        if claim_issues:
            codes = sorted({str(item.get("code")) for item in claim_issues})
            raise DiscourseContractError(
                f"required claim {_text(claim.get('claim_id'))!r} is not canonical: {codes}"
            )
    return required


def _events(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = graph.get("events")
    if not isinstance(raw, list) or not raw:
        raise DiscourseContractError("frozen graph requires a non-empty events array")
    if not all(isinstance(item, Mapping) for item in raw):
        raise DiscourseContractError("every frozen graph event must be an object")
    return list(raw)


def _event_id(event: Mapping[str, Any]) -> str:
    return _text(
        event.get("event_id")
        or event.get("event_instance_id")
        or event.get("id")
        or event.get("node_id")
    )


def _extract_speaker_roster(parent_context: Mapping[str, Any]) -> list[str]:
    candidates: list[Any] = []
    for key in ("speaker_roster", "speaker_label_allowlist", "speaker_set"):
        value = parent_context.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    for nested_key in ("dialogue_structure", "speaker_structure"):
        nested = parent_context.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        for key in ("speaker_label_allowlist", "speaker_set", "recent_speaker_order"):
            value = nested.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    dialogue = parent_context.get("dialogue")
    turns = dialogue.get("turns") if isinstance(dialogue, Mapping) else None
    if isinstance(turns, list):
        candidates.extend(
            item.get("speaker") for item in turns if isinstance(item, Mapping)
        )
    roster = _ordered_unique(candidates)
    if not roster:
        raise DiscourseContractError("parent context requires an explicit speaker roster")
    if len(roster) > MAX_SPEAKERS:
        roster = roster[:MAX_SPEAKERS]
    return roster


def _topic_frame(
    graph: Mapping[str, Any], parent_context: Mapping[str, Any]
) -> tuple[str, str]:
    spine = parent_context.get("topic_spine")
    spine = spine if isinstance(spine, Mapping) else {}
    topic = _text(parent_context.get("topic") or spine.get("topic"))
    setting = _text(parent_context.get("setting") or spine.get("setting"))
    if not topic or not setting:
        metadata = graph.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        discourse = metadata.get("discourse_plan")
        discourse = discourse if isinstance(discourse, Mapping) else {}
        topic = topic or _text(discourse.get("topic"))
        setting = setting or _text(discourse.get("setting"))
    operation = _text(graph.get("operation") or graph.get("qa_operation"))
    topic = topic or operation.replace("_", " ")
    setting = setting or "an ongoing conversation"
    return topic, setting


def _flatten_semantic_values(value: Any) -> list[str]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (str, int, float)):
        return [_text(value)] if _text(value) else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value):
            if key in {"time_index", "global_time_index"}:
                continue
            result.extend(_flatten_semantic_values(value[key]))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            result.extend(_flatten_semantic_values(item))
        return result
    return []


def _topic_terms(
    topic: str, setting: str, required_claims: Sequence[Mapping[str, Any]]
) -> list[str]:
    surfaces: list[str] = [topic, setting]
    for claim in required_claims:
        canonical = claim.get("canonical_support_claim")
        if isinstance(canonical, Mapping):
            surfaces.extend(_flatten_semantic_values(canonical.get("subject")))
            surfaces.extend(_flatten_semantic_values(canonical.get("predicate")))
            surfaces.extend(_flatten_semantic_values(canonical.get("objects")))
            surfaces.extend(_flatten_semantic_values(canonical.get("condition")))
            surfaces.extend(_flatten_semantic_values(canonical.get("scope")))
    terms: list[str] = []
    seen: set[str] = set()
    for surface in surfaces:
        for word in _WORD_RE.findall(surface.casefold()):
            normalized = word.strip("'-")
            if len(normalized) < 3 or normalized in _STOPWORDS or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(normalized)
    if not terms:
        raise DiscourseContractError("topic spine has no usable lexical anchors")
    return terms[:64]


def _claim_order_key(
    claim: Mapping[str, Any], event_order: Mapping[str, int]
) -> tuple[int, int, str]:
    source_ids = [
        _text(item)
        for item in claim.get("source_event_ids") or []
        if _text(item) in event_order
    ]
    subject_id = _text(claim.get("subject_event_id"))
    if subject_id in event_order:
        source_ids.append(subject_id)
    position = max((event_order[item] for item in source_ids), default=10**9)
    kind = _token(claim.get("claim_kind") or claim.get("claim_type"))
    relation_rank = 1 if kind != "node" else 0
    return position, relation_rank, _text(claim.get("claim_id"))


def _balanced_chunks(values: Sequence[Any], count: int) -> list[list[Any]]:
    count = max(1, min(count, len(values)))
    base, remainder = divmod(len(values), count)
    result: list[list[Any]] = []
    offset = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        result.append(list(values[offset : offset + size]))
        offset += size
    return result


def _balanced_chunks_exact(values: Sequence[Any], count: int) -> list[list[Any]]:
    """Return exactly ``count`` ordered chunks, allowing empty chunks."""

    count = max(1, int(count))
    base, remainder = divmod(len(values), count)
    result: list[list[Any]] = []
    offset = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        result.append(list(values[offset : offset + size]))
        offset += size
    return result


def _planned_turn_ids(parent_context: Mapping[str, Any], count: int) -> list[str]:
    dialogue = parent_context.get("dialogue")
    turns = dialogue.get("turns") if isinstance(dialogue, Mapping) else None
    last_id = ""
    if isinstance(turns, list):
        for turn in reversed(turns):
            if isinstance(turn, Mapping) and _text(turn.get("turn_id")):
                last_id = _text(turn.get("turn_id"))
                break
    match = re.fullmatch(r"(.*?)(\d+)", last_id)
    if match:
        prefix, raw_number = match.groups()
        start = int(raw_number) + 1
        width = max(1, len(raw_number))
        return [f"{prefix}{start + index:0{width}d}" for index in range(count)]
    return [f"TFD:{index:03d}" for index in range(1, count + 1)]


def compile_frozen_support_quote(claim: Mapping[str, Any]) -> str:
    """Compatibility wrapper around the independent typed support compiler."""

    try:
        return stage3_truth_support.compile_frozen_support_quote(claim)
    except stage3_truth_support.TruthSupportError as error:
        raise DiscourseContractError(str(error)) from error


def _build_turn_slots(
    acts: Sequence[Mapping[str, Any]],
    roster: Sequence[str],
    parent_context: Mapping[str, Any],
    semantic_by_id: Mapping[str, Mapping[str, Any]],
    *,
    minimum_turns: int,
    target_dialogue_tokens: int,
) -> list[dict[str, Any]]:
    beat_rows = [
        (act, beat)
        for act in acts
        for beat in act.get("beats") or []
        if isinstance(act, Mapping) and isinstance(beat, Mapping)
    ]
    required_count = sum(
        len(beat.get("required_claim_ids") or []) for _act, beat in beat_rows
    )
    context_word_budget = int(
        math.ceil(
            max(0, int(target_dialogue_tokens or 0))
            * CONTEXT_WORD_BUDGET_FACTOR
        )
    )
    natural_turn_count = (
        int(
            math.ceil(
                context_word_budget / NATURAL_CONTEXT_TARGET_WORDS_PER_TURN
            )
        )
        if context_word_budget
        else 0
    )
    requested_count = max(
        len(beat_rows),
        required_count,
        int(minimum_turns or 0),
        natural_turn_count,
    )
    maximum_count = len(beat_rows) * MAX_TURNS_PER_BEAT
    if requested_count > maximum_count:
        raise DiscourseContractError(
            f"requested {requested_count} turns exceeds discourse cap {maximum_count}"
        )
    counts = [max(1, len(beat.get("required_claim_ids") or [])) for _act, beat in beat_rows]
    cursor = 0
    while sum(counts) < requested_count:
        if counts[cursor % len(counts)] < MAX_TURNS_PER_BEAT:
            counts[cursor % len(counts)] += 1
        cursor += 1
    planned_ids = _planned_turn_ids(parent_context, requested_count)
    target_words = max(
        24,
        int(math.ceil(context_word_budget / requested_count)),
    )
    slots: list[dict[str, Any]] = []
    global_index = 0
    for (act, beat), beat_turn_count in zip(beat_rows, counts):
        claim_ids = list(beat.get("required_claim_ids") or [])
        for local_index in range(beat_turn_count):
            required_claim_ids = claim_ids[local_index : local_index + 1]
            if required_claim_ids:
                discourse_move, discourse_move_instruction = (
                    _SUPPORT_DISCOURSE_MOVES[
                        global_index % len(_SUPPORT_DISCOURSE_MOVES)
                    ]
                )
            else:
                discourse_move, discourse_move_instruction = (
                    _CONTEXT_DISCOURSE_MOVES[
                        global_index % len(_CONTEXT_DISCOURSE_MOVES)
                    ]
                )
            typed_support_records = []
            frozen_support = []
            for claim_id in required_claim_ids:
                try:
                    typed_record = stage3_truth_support.compile_typed_frozen_support(
                        semantic_by_id[claim_id]
                    )
                except stage3_truth_support.TruthSupportError as error:
                    raise DiscourseContractError(str(error)) from error
                typed_support_records.append(typed_record)
                frozen_support.append(
                    {
                        "claim_id": claim_id,
                        "exact_support_quote": typed_record["quote"],
                    }
                )
            # Keep turns short enough for conversational prose.  The complete
            # dialogue budget is represented by more turn slots and split into
            # beat-sized API segments; it is not compressed into essay-length
            # individual turns.
            sentence_count = max(
                2,
                int(math.ceil(target_words / CONTEXT_SENTENCE_TARGET_WORDS)),
            )
            lead_sentence_count = sentence_count // 2
            if global_index % 2 and sentence_count > 2:
                lead_sentence_count += 1
            follow_sentence_count = sentence_count - lead_sentence_count
            minimum_context_sentence_count = max(
                2, sentence_count - CONTEXT_SENTENCE_COUNT_TOLERANCE
            )
            maximum_context_sentence_count = (
                sentence_count + CONTEXT_SENTENCE_COUNT_TOLERANCE
            )
            slots.append(
                {
                    "turn_slot_id": f"TS{global_index + 1:03d}",
                    "order_index": global_index,
                    "turn_id": planned_ids[global_index],
                    "speaker": roster[global_index % len(roster)],
                    "act_id": _text(act.get("act_id")),
                    "beat_id": _text(beat.get("beat_id")),
                    "purpose": (
                        "Write only a non-answer-bearing lead-in and reaction around a local insertion; do not state its subject, predicate, value, status, time, condition, or direction."
                        if required_claim_ids
                        else discourse_move_instruction
                    ),
                    "discourse_move": discourse_move,
                    "discourse_move_instruction": discourse_move_instruction,
                    "contains_local_support": bool(required_claim_ids),
                    "required_claim_ids": required_claim_ids,
                    "frozen_support": frozen_support,
                    "typed_support_records": typed_support_records,
                    "target_words": target_words,
                    # This is an aggregate generation budget, not a semantic
                    # field. Segment validation sums it across a complete beat
                    # so harmless variation between neighboring turns remains
                    # legal while a shortened long dialogue cannot pass.
                    "minimum_context_words": target_words,
                    "target_context_sentence_count": sentence_count,
                    "minimum_context_sentence_count": minimum_context_sentence_count,
                    "maximum_context_sentence_count": maximum_context_sentence_count,
                    "lead_sentence_count": lead_sentence_count,
                    "follow_sentence_count": follow_sentence_count,
                }
            )
            global_index += 1
    return slots


def _semantic_snapshot(claim: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "claim_id",
        "claim_kind",
        "claim_type",
        "operation",
        "subject_event_id",
        "predicate",
        "object_event_ids",
        "edge_role",
        "qa_eligible",
        "qa_candidate",
        "answer_bearing",
        "polarity",
        "status",
        "time_index",
        "global_time_index",
        "condition",
        "scope",
        "source_event_ids",
        "canonical_support_claim",
        "truth_signature",
    )
    return {key: copy.deepcopy(claim.get(key)) for key in fields if key in claim}


def _require_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping) or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise DiscourseContractError("invalid discourse plan schema")
    candidate = copy.deepcopy(dict(plan))
    claimed_hash = _text(candidate.pop("plan_sha256", ""))
    if not claimed_hash or canonical_sha256(candidate) != claimed_hash:
        raise DiscourseContractError("discourse plan hash mismatch")


def build_discourse_plan(
    frozen_graph: Mapping[str, Any],
    frozen_claim_bundle: Mapping[str, Any],
    parent_context: Mapping[str, Any],
    *,
    minimum_turns: int = 0,
    target_dialogue_tokens: int = 0,
) -> dict[str, Any]:
    """Build a deterministic, immutable discourse plan with at most two acts."""

    if not all(
        isinstance(item, Mapping)
        for item in (frozen_graph, frozen_claim_bundle, parent_context)
    ):
        raise DiscourseContractError("graph, claim bundle, and parent context must be objects")
    graph_copy = copy.deepcopy(dict(frozen_graph))
    bundle_copy = copy.deepcopy(dict(frozen_claim_bundle))
    parent_copy = copy.deepcopy(dict(parent_context))
    events = _events(graph_copy)
    event_ids = [_event_id(event) for event in events]
    if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        raise DiscourseContractError("graph event ids must be non-empty and unique")
    event_order = {event_id: index for index, event_id in enumerate(event_ids)}
    required = sorted(
        _required_claims(bundle_copy), key=lambda row: _claim_order_key(row, event_order)
    )
    roster = _extract_speaker_roster(parent_copy)
    topic, setting = _topic_frame(graph_copy, parent_copy)
    operation = _text(graph_copy.get("operation") or graph_copy.get("qa_operation"))
    if not operation:
        operation = _text(required[0].get("operation")) or "relation_lookup"

    context_word_budget = int(
        math.ceil(
            max(0, int(target_dialogue_tokens or 0))
            * CONTEXT_WORD_BUDGET_FACTOR
        )
    )
    natural_turn_count = (
        int(
            math.ceil(
                context_word_budget / NATURAL_CONTEXT_TARGET_WORDS_PER_TURN
            )
        )
        if context_word_budget
        else 0
    )
    requested_turn_count = max(
        1,
        len(required),
        max(0, int(minimum_turns or 0)),
        natural_turn_count,
    )
    capacity_beat_count = int(
        math.ceil(requested_turn_count / MAX_TURNS_PER_BEAT)
    )
    semantic_beat_count = (
        1
        if len(required) <= 2
        else 2
        if len(required) <= 4
        else 3
        if len(required) == 5
        else 4
    )
    beat_count = max(capacity_beat_count, semantic_beat_count)
    maximum_beat_count = MAX_ACTS * MAX_BEATS_PER_ACT
    if beat_count > maximum_beat_count:
        raise DiscourseContractError(
            f"requested {requested_turn_count} turns requires {beat_count} beats; "
            f"discourse cap is {maximum_beat_count} beats"
        )
    act_count = int(math.ceil(beat_count / MAX_BEATS_PER_ACT))
    beat_counts_by_act = [
        len(group)
        for group in _balanced_chunks(list(range(beat_count)), act_count)
    ]
    claim_groups_by_beat = _balanced_chunks_exact(required, beat_count)
    purposes = _OPERATION_PURPOSES.get(
        operation,
        (
            "Establish the shared scenario and the first relevant facts.",
            "Develop the fixed relations and practical outcome.",
        ),
    )
    acts: list[dict[str, Any]] = []
    global_beat_index = 0
    for act_index, act_beat_count in enumerate(beat_counts_by_act):
        beats: list[dict[str, Any]] = []
        for local_index in range(act_beat_count):
            beat_claims = claim_groups_by_beat[global_beat_index]
            beat_id = f"A{act_index + 1}B{local_index + 1}"
            beats.append(
                {
                    "beat_id": beat_id,
                    "order_index": global_beat_index,
                    "purpose": (
                        "Set up this part of the scenario."
                        if local_index == 0
                        else "Develop its consequence or clarification."
                    ),
                    "required_claim_ids": [
                        _text(claim.get("claim_id")) for claim in beat_claims
                    ],
                }
            )
            global_beat_index += 1
        acts.append(
            {
                "act_id": f"A{act_index + 1}",
                "order_index": act_index,
                "purpose": purposes[min(act_index, len(purposes) - 1)],
                "beats": beats,
            }
        )

    semantic_claims = [_semantic_snapshot(claim) for claim in required]
    semantic_by_id = {
        _text(claim.get("claim_id")): claim for claim in semantic_claims
    }
    turn_slots = _build_turn_slots(
        acts,
        roster,
        parent_copy,
        semantic_by_id,
        minimum_turns=max(0, int(minimum_turns or 0)),
        target_dialogue_tokens=max(0, int(target_dialogue_tokens or 0)),
    )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "operation": operation,
        "frozen_input_hashes": {
            "graph_sha256": canonical_sha256(graph_copy),
            "claim_bundle_sha256": canonical_sha256(bundle_copy),
            "parent_context_sha256": canonical_sha256(parent_copy),
        },
        "speaker_roster": roster,
        "topic_spine": {
            "topic": topic,
            "setting": setting,
            "anchor_terms": _topic_terms(topic, setting, required),
            "window_turns": TOPIC_WINDOW_TURNS,
        },
        "acts": acts,
        "turn_slots": turn_slots,
        "required_claim_ids": [
            _text(claim.get("claim_id")) for claim in required
        ],
        "semantic_claims": semantic_claims,
        "constraints": {
            "max_acts": MAX_ACTS,
            "max_beats_per_act": MAX_BEATS_PER_ACT,
            "max_turns_per_beat": MAX_TURNS_PER_BEAT,
            "claim_coverage_exactly_once": True,
            "exact_quote_must_be_visible_substring": True,
            "isolated_fact_turns_forbidden": True,
            "unrelated_filler_forbidden": True,
            "api_may_change_truth": False,
            "required_turn_count": len(turn_slots),
            "target_dialogue_tokens": max(0, int(target_dialogue_tokens or 0)),
            "accepted_prefix_tail_turns": ACCEPTED_PREFIX_TAIL_TURNS,
            "near_duplicate_lookback_sentences": (
                stage3_dialogue_context_guard.NEAR_DUPLICATE_LOOKBACK_SENTENCES
            ),
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _contract_claim_obligation(claim: Mapping[str, Any]) -> dict[str, Any]:
    canonical = claim.get("canonical_support_claim")
    if not isinstance(canonical, Mapping):
        raise DiscourseContractError(
            f"claim {_text(claim.get('claim_id'))!r} lacks canonical_support_claim"
        )
    return {
        "claim_id": _text(claim.get("claim_id")),
        "canonical_fact": copy.deepcopy(dict(canonical)),
        "instruction": (
            "Read-only semantic context. The program inserts its frozen support clause; "
            "do not return, repeat, paraphrase, negate, reverse, or qualify this fact."
        ),
    }


def _reserved_truth_tokens(claims: Sequence[Mapping[str, Any]]) -> set[str]:
    """Collect child-truth lexical surfaces that API topic hints must hide."""

    tokens: set[str] = set()
    for claim in claims:
        canonical = claim.get("canonical_support_claim")
        if not isinstance(canonical, Mapping):
            continue
        for key in (
            "subject",
            "predicate",
            "objects",
            "object",
            "value",
            "status",
            "condition",
            "scope",
        ):
            for surface in _flatten_semantic_values(canonical.get(key)):
                tokens.update(
                    word.casefold().strip("'-")
                    for word in _WORD_RE.findall(surface)
                    if word.casefold().strip("'-")
                )
    return tokens


def _redact_topic_text(value: Any, reserved_tokens: set[str]) -> str:
    """Keep parent prose tokens that cannot reconstruct a child truth field."""

    safe = [
        word
        for word in _WORD_RE.findall(_text(value))
        if word.casefold().strip("'-") not in reserved_tokens
    ]
    return " ".join(safe)


def _api_safe_topic_spine(
    topic_spine: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return parent-oriented topic hints with all child truth tokens removed."""

    reserved_tokens = _reserved_truth_tokens(claims)
    topic = _redact_topic_text(topic_spine.get("topic"), reserved_tokens)
    setting = _redact_topic_text(topic_spine.get("setting"), reserved_tokens)
    fallback_tokens = [
        token
        for token in ("ongoing", "conversation", "shared", "discussion")
        if token not in reserved_tokens
    ]
    # Redaction can leave only a stopword (run230h produced the topic
    # literally ``and``).  Such a value is non-empty but gives the API no
    # usable continuity anchor, so judge topic and setting independently by
    # the same content-token rule used by the validator.
    try:
        topic_terms = _topic_terms(topic, "", []) if topic else []
    except DiscourseContractError:
        topic_terms = []
    try:
        setting_terms = _topic_terms("", setting, []) if setting else []
    except DiscourseContractError:
        setting_terms = []
    if not topic_terms:
        topic = " ".join(fallback_tokens[:2]) or "dialogue"
    if not setting_terms:
        setting = " ".join(fallback_tokens[2:] or fallback_tokens[:2]) or "exchange"
    original_safe_anchors = [
        _text(term).casefold()
        for term in topic_spine.get("anchor_terms") or []
        if _text(term)
        and _text(term).casefold() not in reserved_tokens
        and _text(term).casefold() not in _STOPWORDS
    ]
    if not original_safe_anchors:
        raise DiscourseContractError(
            "truth-redacted topic spine has no parent-safe lexical anchor"
        )
    return {
        "topic": topic,
        "setting": setting,
        # Keep the live generation vocabulary a strict subset of the frozen
        # validator vocabulary.  Fallback prose such as "ongoing discussion"
        # may describe the frame, but cannot by itself satisfy topic continuity.
        "anchor_terms": list(dict.fromkeys(original_safe_anchors)),
        "window_turns": int(
            topic_spine.get("window_turns") or TOPIC_WINDOW_TURNS
        ),
        "redaction": "parent_topic_minus_child_truth_surfaces",
    }


def build_api_contract(
    plan: Mapping[str, Any],
    *,
    act_id: str | None = None,
    segment_id: str | None = None,
    accepted_prefix_tail: Sequence[Mapping[str, Any]] | None = None,
    retry_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose a dialogue-only creative surface around locally owned truth.

    The API does not need the frozen quote or canonical fact in order to write
    connective dialogue.  Hiding both also removes the strongest prompt-side
    invitation to paraphrase answer-bearing facts.  The immutable plan retains
    them, and the program inserts them after the API context has passed the
    independent guard.
    """

    _require_plan(plan)
    plan_copy = copy.deepcopy(dict(plan))
    claims = plan_copy.get("semantic_claims")
    if not isinstance(claims, list):
        raise DiscourseContractError("discourse plan semantic_claims must be an array")
    api_topic_spine = _api_safe_topic_spine(plan_copy["topic_spine"], claims)
    selected_act_id = _text(act_id)
    selected_segment_id = _text(segment_id)
    if selected_act_id and selected_segment_id:
        raise DiscourseContractError(
            "API contract accepts either act_id or segment_id, not both"
        )
    segment_act_id = ""
    segment_purpose = ""
    if selected_segment_id:
        for act in plan_copy["acts"]:
            for beat in act.get("beats") or []:
                if _text(beat.get("beat_id")) == selected_segment_id:
                    segment_act_id = _text(act.get("act_id"))
                    segment_purpose = _text(beat.get("purpose"))
                    break
            if segment_act_id:
                break
        if not segment_act_id:
            raise DiscourseContractError(
                f"unknown discourse segment for API contract: {selected_segment_id!r}"
            )
    selected_acts = [
        act
        for act in plan_copy["acts"]
        if (
            not selected_act_id
            and not selected_segment_id
        )
        or _text(act.get("act_id")) == (selected_act_id or segment_act_id)
    ]
    if selected_act_id and not selected_acts:
        raise DiscourseContractError(
            f"unknown discourse act for API contract: {selected_act_id!r}"
        )
    act_purposes = [
        {
            "act_id": act["act_id"],
            "purpose": act["purpose"],
        }
        for act in selected_acts
    ]
    selected_act_ids = {_text(act.get("act_id")) for act in selected_acts}
    turn_slots = [
        {
            key: copy.deepcopy(slot[key])
            for key in (
                "turn_slot_id",
                "order_index",
                "speaker",
                "act_id",
                "beat_id",
                "purpose",
                "discourse_move",
                "discourse_move_instruction",
                "contains_local_support",
                "target_words",
                "minimum_context_words",
                "target_context_sentence_count",
                "minimum_context_sentence_count",
                "maximum_context_sentence_count",
            )
        }
        for slot in plan_copy["turn_slots"]
        if _text(slot.get("act_id")) in selected_act_ids
        and (
            not selected_segment_id
            or _text(slot.get("beat_id")) == selected_segment_id
        )
    ]
    prefix_tail: list[dict[str, Any]] = []
    for raw in list(accepted_prefix_tail or [])[-ACCEPTED_PREFIX_TAIL_TURNS:]:
        if not isinstance(raw, Mapping):
            continue
        context = raw.get("context_sentences")
        if not isinstance(context, list) or not all(
            isinstance(item, str) and _text(item) for item in context
        ):
            continue
        prefix_tail.append(
            {
                "speaker": _text(raw.get("speaker")),
                "context_sentences": copy.deepcopy(context),
            }
        )
    minimum_segment_words = sum(
        int(slot.get("minimum_context_words") or slot.get("target_words") or 0)
        for slot in turn_slots
    )
    required_turn_slot_ids = [
        _text(slot.get("turn_slot_id")) for slot in turn_slots
    ]
    minimum_sentences_per_turn = min(
        int(slot.get("minimum_context_sentence_count") or 2)
        for slot in turn_slots
    )
    maximum_sentences_per_turn = max(
        int(slot.get("maximum_context_sentence_count") or 3)
        for slot in turn_slots
    )
    retry_contract = copy.deepcopy(dict(retry_feedback or {}))
    contract: dict[str, Any] = {
        "schema_version": API_CONTRACT_SCHEMA_VERSION,
        "plan_sha256": plan_copy["plan_sha256"],
        "frozen_input_hashes": copy.deepcopy(plan_copy["frozen_input_hashes"]),
        "read_only_brief": {
            "operation": plan_copy["operation"],
            "speaker_roster": copy.deepcopy(plan_copy["speaker_roster"]),
            "topic_spine": api_topic_spine,
            "act_purposes": act_purposes,
            "generation_scope": {
                "act_id": selected_act_id or segment_act_id or "all",
                "segment_id": selected_segment_id or selected_act_id or "all",
                "segment_purpose": segment_purpose or "realize the selected contiguous dialogue scope",
                "contiguous_segment": True,
                "minimum_segment_context_words": minimum_segment_words,
                "accepted_prefix_turn_count": len(prefix_tail),
            },
            "accepted_api_neutral_prefix_tail": prefix_tail,
            "retry_feedback": retry_contract,
            "required_turn_row_count": len(turn_slots),
            "turn_slots": turn_slots,
        },
        "generation_rules": [
            f"Return exactly one JSON object whose only top-level key is turns. Its turns value must be an array of exactly {len(turn_slots)} row objects; never return one row as the top-level object.",
            f"The turn_slot_id values must occur exactly once in this exact order: {required_turn_slot_ids}.",
            "Do not return speaker, turn_id, act_id, or beat_id; the program attaches those frozen control fields from turn_slot_id.",
            f"Every row must contain exactly turn_slot_id and context_sentences. context_sentences must contain {minimum_sentences_per_turn}-{maximum_sentences_per_turn} distinct substantive sentences; never return a one-sentence row.",
            "Every sentence must be distinct, substantive, responsive to neighboring turns, and on topic.",
            f"Across this contiguous segment, return at least {minimum_segment_words} API-authored context words; do not shorten later rows to save output tokens.",
            "Use each slot's target_words as its approximate share of the segment budget, while allowing natural variation between neighboring turns.",
            "Do not state, guess, question, qualify, negate, or paraphrase any concrete answer-bearing fact; the program inserts all frozen truth locally at a deterministic point in each turn.",
            "The program preserves every returned row in the exact required slot sequence. Compact rows appended after that complete sequence are recorded as decoder overflow and excluded before full validation; never rely on suffix rows.",
            f"In every consecutive window of {api_topic_spine['window_turns']} turns, at least one turn must use an exact safe topic anchor from {api_topic_spine['anchor_terms']}; use pronouns and ordinary local continuity elsewhere so the prose is not templated.",
            "When an accepted_api_neutral_prefix_tail is supplied, continue its conversational thread without copying or closely paraphrasing it.",
            "Do not mention internal IDs, graphs, schemas, prompts, proofs, gold, or datasets in visible text.",
            "Do not return or rewrite any event, slot, relation, proof, answer, or reasoning path.",
        ],
        "local_response_envelope": {
            "schema_version": API_RESPONSE_SCHEMA_VERSION,
            "plan_sha256": plan_copy["plan_sha256"],
            "frozen_input_hashes": copy.deepcopy(plan_copy["frozen_input_hashes"]),
            "ownership": "program_attaches_if_omitted_api_must_not_invent_or_change",
        },
        "response_schema": {
            "top_level_type": "object",
            "top_level_required_keys": ["turns"],
            "top_level_additional_keys_allowed": False,
            "turns": {
                "type": "array",
                "item_count_exact": len(turn_slots),
                "turn_slot_ids_exact_order": required_turn_slot_ids,
                "item_schema": {
                    "type": "object",
                    "required_keys": [
                        "turn_slot_id",
                        "context_sentences",
                    ],
                    "additional_keys_allowed": False,
                    "context_sentences": {
                        "type": "array",
                        "minimum_items": minimum_sentences_per_turn,
                        "maximum_items": maximum_sentences_per_turn,
                        "items": "nonempty substantive natural-language sentence",
                    },
                },
            },
        },
        "forbidden_response_keys": sorted(_FORBIDDEN_RESPONSE_KEYS),
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def _context_sentence_bounds(slot: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return the locally frozen target range, including v1-plan fallback."""

    legacy_target = int(slot.get("lead_sentence_count") or 0) + int(
        slot.get("follow_sentence_count") or 0
    )
    target = int(slot.get("target_context_sentence_count") or legacy_target or 2)
    minimum = int(
        slot.get("minimum_context_sentence_count")
        or max(2, target - CONTEXT_SENTENCE_COUNT_TOLERANCE)
    )
    maximum = int(
        slot.get("maximum_context_sentence_count")
        or target + CONTEXT_SENTENCE_COUNT_TOLERANCE
    )
    return minimum, target, maximum


def _split_context_fragment(value: str) -> list[str]:
    """Split only unambiguous prose boundaries, preserving visible text.

    Models commonly return several ordinary sentences inside one JSON array
    element.  Array grouping is not truth.  This conservative splitter avoids
    common abbreviations, initials, and decimal points; its normalized visible
    text is required to remain byte-for-byte equivalent after whitespace
    canonicalisation.
    """

    text = str(value)
    if not _text(text):
        return [text]
    result: list[str] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        matched = match.group(0)
        stripped = matched.rstrip()
        terminal_end = match.start() + len(stripped)
        candidate = text[cursor:terminal_end].strip()
        next_start = match.end()
        terminal_core = candidate.rstrip("\"'\u2019\u201d)]").casefold()
        last_token = terminal_core.rsplit(None, 1)[-1] if terminal_core else ""
        punctuation = terminal_core[-1:] if terminal_core else ""
        next_char = text[next_start : next_start + 1]
        previous_char = terminal_core[-2:-1]
        quoted_attribution = bool(
            candidate.endswith(("?\"", "!\"", "?\u201d", "!\u201d"))
            and _DIALOGUE_ATTRIBUTION_RE.match(text[next_start:])
        )
        if quoted_attribution:
            continue
        if punctuation == "." and (
            last_token in _NONTERMINAL_ABBREVIATIONS
            or re.fullmatch(r"[a-z]\.", last_token) is not None
            or (
                previous_char.isdigit()
                and next_char.isdigit()
            )
        ):
            continue
        if candidate:
            result.append(candidate)
        cursor = next_start
    tail = text[cursor:].strip()
    if tail:
        result.append(tail)
    if not result:
        return [text]
    if _text(" ".join(result)) != _text(text):
        return [text]
    return result


def _normalize_context_fragments(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if isinstance(value, str):
            result.extend(_split_context_fragment(value))
        else:
            result.append(copy.deepcopy(value))
    return result


def _normalize_flat_context_representation(
    slots: Sequence[Mapping[str, Any]], raw_turns: Any
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Canonicalize harmless sentence grouping without changing visible text.

    New live responses use one ordered ``context_sentences`` array.  Legacy
    responses may still use separate lead/follow arrays.  For either form, the
    program chooses the frozen support insertion boundary and emits the old
    canonical two-array view consumed by the independent context guard.

    Unambiguous boundaries inside a JSON string may be restored, but no visible
    word or punctuation is rewritten, reordered, or dropped.  Consequently an
    extra repeated or contradictory sentence becomes *more* visible to the
    semantic guard and can never be hidden by representation repair.
    """

    if not isinstance(raw_turns, list):
        return copy.deepcopy(raw_turns), [], []
    canonical_rows: list[Any] = []
    audit_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_turns):
        if not isinstance(raw_row, Mapping):
            canonical_rows.append(copy.deepcopy(raw_row))
            continue
        row = copy.deepcopy(dict(raw_row))
        compact = "context_sentences" in row
        legacy = "lead_in_sentences" in row or "follow_up_sentences" in row
        if compact and legacy:
            issues.append(
                {
                    "code": "ambiguous_flat_turn_sentence_formats",
                    "turn_index": index,
                }
            )
            canonical_rows.append(row)
            continue
        if compact:
            allowed = {"turn_slot_id", "context_sentences"}
            raw_sentences = row.get("context_sentences")
            source_format = "ordered_context_sentences"
            original_lead_count = None
            original_follow_count = None
        else:
            allowed = {
                "turn_slot_id",
                "lead_in_sentences",
                "follow_up_sentences",
            }
            raw_lead = row.get("lead_in_sentences")
            raw_follow = row.get("follow_up_sentences")
            source_format = "legacy_lead_follow_sentences"
            original_lead_count = len(raw_lead) if isinstance(raw_lead, list) else None
            original_follow_count = (
                len(raw_follow) if isinstance(raw_follow, list) else None
            )
            raw_sentences = (
                list(raw_lead) + list(raw_follow)
                if isinstance(raw_lead, list) and isinstance(raw_follow, list)
                else None
            )
        extra = sorted(set(row) - allowed)
        missing = sorted(allowed - set(row))
        if extra:
            issues.append(
                {
                    "code": "unexpected_flat_turn_keys",
                    "turn_index": index,
                    "keys": extra,
                }
            )
        if missing:
            issues.append(
                {
                    "code": "missing_flat_turn_keys",
                    "turn_index": index,
                    "keys": missing,
                }
            )
        if not isinstance(raw_sentences, list):
            issues.append(
                {
                    "code": "flat_turn_context_sentences_not_array",
                    "turn_index": index,
                }
            )
            canonical_rows.append(row)
            continue

        sentences = _normalize_context_fragments(raw_sentences)
        total = len(sentences)
        slot = slots[index] if index < len(slots) else {}
        minimum, target, maximum = _context_sentence_bounds(slot)
        if total < MIN_CONTEXT_SENTENCES_PER_TURN:
            issues.append(
                {
                    "code": "context_sentence_total_below_structural_minimum",
                    "turn_index": index,
                    "minimum": MIN_CONTEXT_SENTENCES_PER_TURN,
                    "target": target,
                    "actual": total,
                }
            )

        planned_lead = int(slot.get("lead_sentence_count") or 0)
        planned_total = planned_lead + int(slot.get("follow_sentence_count") or 0)
        ratio = planned_lead / planned_total if planned_total else 0.5
        if total >= 2:
            insertion_index = min(total - 1, max(1, int(round(total * ratio))))
        else:
            insertion_index = total
        lead_sentences = sentences[:insertion_index]
        follow_sentences = sentences[insertion_index:]
        before_hash = canonical_sha256(raw_sentences)
        after_hash = canonical_sha256(lead_sentences + follow_sentences)
        visible_before = _text(
            " ".join(item for item in raw_sentences if isinstance(item, str))
        )
        visible_after = _text(
            " ".join(item for item in sentences if isinstance(item, str))
        )
        canonical_rows.append(
            {
                "turn_slot_id": copy.deepcopy(row.get("turn_slot_id")),
                "lead_in_sentences": lead_sentences,
                "follow_up_sentences": follow_sentences,
            }
        )
        audit_rows.append(
            {
                "turn_index": index,
                "source_format": source_format,
                "original_lead_sentence_count": original_lead_count,
                "original_follow_sentence_count": original_follow_count,
                "raw_context_element_count": len(raw_sentences),
                "context_sentence_count": total,
                "target_sentence_count_range": [minimum, maximum],
                "sentence_count_within_target_range": minimum <= total <= maximum,
                "support_insertion_index": insertion_index,
                "sentence_sequence_sha256_before": before_hash,
                "sentence_sequence_sha256_after": after_hash,
                "visible_text_sha256_before": canonical_sha256(visible_before),
                "visible_text_sha256_after": canonical_sha256(visible_after),
                "sentence_grouping_normalized": len(raw_sentences) != total,
                "sentence_text_modified": visible_before != visible_after,
                "sentence_order_modified": False,
                "sentences_dropped": 0,
            }
        )
    return canonical_rows, audit_rows, issues


def discourse_act_ids(plan: Mapping[str, Any]) -> list[str]:
    """Return the frozen act order used for bounded live naturalisation."""

    _require_plan(plan)
    return [_text(act.get("act_id")) for act in plan["acts"]]


def discourse_segment_ids(plan: Mapping[str, Any]) -> list[str]:
    """Return frozen beat IDs used as bounded live generation segments."""

    _require_plan(plan)
    return [
        _text(beat.get("beat_id"))
        for act in plan["acts"]
        for beat in act.get("beats") or []
    ]


def act_turn_slots(
    plan: Mapping[str, Any], act_id: str
) -> list[dict[str, Any]]:
    """Return a deep-copied contiguous slot slice for one frozen act."""

    _require_plan(plan)
    wanted = _text(act_id)
    slots = [
        copy.deepcopy(dict(slot))
        for slot in plan["turn_slots"]
        if _text(slot.get("act_id")) == wanted
    ]
    if not slots:
        raise DiscourseContractError(f"unknown discourse act: {wanted!r}")
    indexes = [int(slot.get("order_index") or 0) for slot in slots]
    if indexes != list(range(min(indexes), min(indexes) + len(indexes))):
        raise DiscourseContractError(
            f"discourse act {wanted!r} does not own a contiguous turn-slot slice"
        )
    return slots


def segment_turn_slots(
    plan: Mapping[str, Any], segment_id: str
) -> list[dict[str, Any]]:
    """Return the contiguous frozen slot slice assigned to one beat."""

    _require_plan(plan)
    wanted = _text(segment_id)
    slots = [
        copy.deepcopy(dict(slot))
        for slot in plan["turn_slots"]
        if _text(slot.get("beat_id")) == wanted
    ]
    if not slots:
        raise DiscourseContractError(f"unknown discourse segment: {wanted!r}")
    indexes = [int(slot.get("order_index") or 0) for slot in slots]
    if indexes != list(range(min(indexes), min(indexes) + len(indexes))):
        raise DiscourseContractError(
            f"discourse segment {wanted!r} is not a contiguous turn-slot slice"
        )
    return slots


def normalize_api_response_segment_overflow(
    plan: Mapping[str, Any],
    response: Any,
    *,
    segment_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Remove only an unambiguous compact suffix after a complete slot prefix."""

    _require_plan(plan)
    selected_slots = segment_turn_slots(plan, segment_id)
    expected_ids = [_text(slot.get("turn_slot_id")) for slot in selected_slots]
    diagnostics: dict[str, Any] = {
        "schema_version": API_SEGMENT_OVERFLOW_NORMALIZER_VERSION,
        "used": False,
        "failure": False,
        "reason": None,
        "expected_turn_count": len(expected_ids),
        "original_turn_count": None,
        "normalized_turn_count": None,
        "overflow_suffix_count": 0,
        "overflow_suffix_pattern": None,
        "overflow_suffix_turn_slot_ids": [],
        "original_response_sha256": None,
        "normalized_response_sha256": None,
        "retained_prefix_sha256_before": None,
        "retained_prefix_sha256_after": None,
        "retained_prefix_content_preserved": False,
        "overflow_suffix_sha256": None,
        "overflow_suffix_rows": [],
        "truth_fields_modified": [],
    }

    def reject(reason: str) -> tuple[Any, dict[str, Any]]:
        diagnostics["failure"] = True
        diagnostics["reason"] = reason
        return copy.deepcopy(response), diagnostics

    if not isinstance(response, Mapping):
        return reject("response_not_object")
    allowed_top = {
        "schema_version",
        "plan_sha256",
        "frozen_input_hashes",
        "turns",
    }
    if not set(response).issubset(allowed_top):
        return reject("unexpected_top_level_keys")
    raw_turns = response.get("turns")
    if not isinstance(raw_turns, list):
        return reject("turns_not_array")
    diagnostics["original_turn_count"] = len(raw_turns)
    diagnostics["original_response_sha256"] = canonical_sha256(response)
    if len(raw_turns) <= len(expected_ids):
        diagnostics["reason"] = "no_overflow_suffix"
        return copy.deepcopy(response), diagnostics

    def compact_row(row: Any) -> bool:
        if not isinstance(row, Mapping) or set(row) != {
            "turn_slot_id",
            "context_sentences",
        }:
            return False
        context = row.get("context_sentences")
        return (
            bool(_text(row.get("turn_slot_id")))
            and isinstance(context, list)
            and bool(context)
            and all(isinstance(item, str) and _text(item) for item in context)
        )

    if not all(compact_row(row) for row in raw_turns):
        return reject("noncompact_turn_row")
    prefix = raw_turns[: len(expected_ids)]
    actual_prefix_ids = [_text(row.get("turn_slot_id")) for row in prefix]
    if actual_prefix_ids != expected_ids:
        return reject("required_prefix_slot_ids_mismatch")

    suffix = copy.deepcopy(raw_turns[len(expected_ids) :])
    suffix_ids = [_text(row.get("turn_slot_id")) for row in suffix]
    all_slot_ids = [_text(slot.get("turn_slot_id")) for slot in plan["turn_slots"]]
    prefix_end = all_slot_ids.index(expected_ids[-1]) + 1
    expected_future_ids = all_slot_ids[prefix_end : prefix_end + len(suffix)]
    future_slot_suffix = (
        len(expected_future_ids) == len(suffix_ids)
        and suffix_ids == expected_future_ids
    )
    prefix_by_id = {
        _text(row.get("turn_slot_id")): row for row in prefix
    }
    exact_copy_suffix = all(
        suffix_id in prefix_by_id
        and canonical_sha256(row) == canonical_sha256(prefix_by_id[suffix_id])
        for suffix_id, row in zip(suffix_ids, suffix)
    )
    if not future_slot_suffix and not exact_copy_suffix:
        return reject("overflow_suffix_not_future_slots_or_exact_copies")
    suffix_pattern = (
        "contiguous_future_slot_ids"
        if future_slot_suffix
        else "exact_retained_prefix_row_copies"
    )
    normalized = copy.deepcopy(dict(response))
    normalized["turns"] = copy.deepcopy(prefix)
    retained_before = canonical_sha256(prefix)
    retained_after = canonical_sha256(normalized["turns"])
    diagnostics.update(
        {
            "used": True,
            "reason": "complete_required_prefix_with_decoder_overflow_suffix",
            "normalized_turn_count": len(normalized["turns"]),
            "overflow_suffix_count": len(suffix),
            "overflow_suffix_pattern": suffix_pattern,
            "overflow_suffix_turn_slot_ids": suffix_ids,
            "normalized_response_sha256": canonical_sha256(normalized),
            "retained_prefix_sha256_before": retained_before,
            "retained_prefix_sha256_after": retained_after,
            "retained_prefix_content_preserved": retained_before == retained_after,
            "overflow_suffix_sha256": canonical_sha256(suffix),
            "overflow_suffix_rows": suffix,
        }
    )
    if not diagnostics["retained_prefix_content_preserved"]:
        raise RuntimeError("overflow normalization changed retained prefix content")
    return normalized, diagnostics


def validate_api_response_segment(
    plan: Mapping[str, Any],
    response: Any,
    *,
    act_id: str | None = None,
    segment_id: str | None = None,
    accepted_prefix_turns: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one API-authored beat segment without assembling frozen support.

    The accepted prefix contains only previously validated API-neutral context,
    never locally inserted support.  Validation is cumulative so repetition or
    semantic drift across a segment boundary is caught before another paid call.
    """

    _require_plan(plan)
    issues: list[dict[str, Any]] = []
    selected_segment_id = _text(segment_id)
    selected_act_id = _text(act_id)
    if selected_segment_id and selected_act_id:
        raise DiscourseContractError(
            "segment validation accepts either act_id or segment_id, not both"
        )
    selected_slots = (
        segment_turn_slots(plan, selected_segment_id)
        if selected_segment_id
        else act_turn_slots(plan, selected_act_id)
    )
    all_slots = list(plan["turn_slots"])
    first_index = int(selected_slots[0].get("order_index") or 0)
    prefix = [copy.deepcopy(dict(row)) for row in (accepted_prefix_turns or [])]
    if len(prefix) != first_index:
        issues.append(
            {
                "code": "accepted_prefix_turn_count_mismatch",
                "expected": first_index,
                "actual": len(prefix),
            }
        )
    raw_turns: Any = None
    if not isinstance(response, Mapping):
        issues.append({"code": "segment_response_not_object"})
    else:
        allowed_top = {
            "schema_version",
            "plan_sha256",
            "frozen_input_hashes",
            "turns",
        }
        extra_top = sorted(set(response) - allowed_top)
        if extra_top:
            issues.append(
                {"code": "unexpected_segment_response_keys", "keys": extra_top}
            )
        if "acts" in response:
            issues.append({"code": "segment_flat_turn_body_required"})
        for field, expected in (
            ("schema_version", API_RESPONSE_SCHEMA_VERSION),
            ("plan_sha256", plan["plan_sha256"]),
            ("frozen_input_hashes", plan["frozen_input_hashes"]),
        ):
            if field in response and response.get(field) != expected:
                issues.append(
                    {
                        "code": "segment_explicit_envelope_mismatch",
                        "field": field,
                    }
                )
        raw_turns = response.get("turns")
    if not isinstance(raw_turns, list):
        issues.append({"code": "segment_turns_not_array"})
        raw_turns = []
    if len(raw_turns) != len(selected_slots):
        issues.append(
            {
                "code": "segment_turn_count_mismatch",
                "expected": len(selected_slots),
                "actual": len(raw_turns),
            }
        )
    current = [copy.deepcopy(row) for row in raw_turns]
    for index, row in enumerate(current):
        if not isinstance(row, Mapping) or set(row) != {
            "turn_slot_id",
            "context_sentences",
        }:
            issues.append(
                {
                    "code": "segment_compact_turn_format_required",
                    "turn_index": index,
                }
            )
    cumulative_raw = prefix + current
    cumulative_slots = all_slots[: len(cumulative_raw)]
    canonical_turns, representation_rows, representation_issues = (
        _normalize_flat_context_representation(cumulative_slots, cumulative_raw)
    )
    issues.extend(representation_issues)

    current_word_count = sum(
        len(_WORD_RE.findall(item))
        for row in current
        if isinstance(row, Mapping)
        for value in (
            row.get("context_sentences")
            if isinstance(row.get("context_sentences"), list)
            else list(row.get("lead_in_sentences") or [])
            + list(row.get("follow_up_sentences") or [])
        )
        for item in [value]
        if isinstance(item, str)
    )
    minimum_word_count = sum(
        int(slot.get("minimum_context_words") or slot.get("target_words") or 0)
        for slot in selected_slots
    )
    if current_word_count < minimum_word_count:
        issues.append(
            {
                "code": "segment_context_word_shortfall",
                "minimum": minimum_word_count,
                "actual": current_word_count,
                "deficit": minimum_word_count - current_word_count,
            }
        )

    guard_plan = copy.deepcopy(dict(plan))
    guard_plan["turn_slots"] = copy.deepcopy(cumulative_slots)
    if isinstance(canonical_turns, list):
        for index, row in enumerate(canonical_turns):
            if index >= len(guard_plan["turn_slots"]) or not isinstance(row, Mapping):
                continue
            lead = row.get("lead_in_sentences")
            follow = row.get("follow_up_sentences")
            if isinstance(lead, list) and isinstance(follow, list):
                guard_plan["turn_slots"][index]["lead_sentence_count"] = len(lead)
                guard_plan["turn_slots"][index]["follow_sentence_count"] = len(
                    follow
                )
    context_guard = stage3_dialogue_context_guard.validate_api_authored_context(
        guard_plan, canonical_turns
    )
    issues.extend(copy.deepcopy(context_guard.get("issues") or []))

    if isinstance(canonical_turns, list):
        for index, (slot, row) in enumerate(zip(cumulative_slots, canonical_turns)):
            if not isinstance(row, Mapping):
                continue
            lead = row.get("lead_in_sentences")
            follow = row.get("follow_up_sentences")
            if not isinstance(lead, list) or not isinstance(follow, list):
                continue
            authored_text = " ".join(
                _text(part) for part in lead + follow if _text(part)
            ).casefold()
            for support_row in slot.get("frozen_support") or []:
                quote = _text(support_row.get("exact_support_quote"))
                if quote and quote.casefold() in authored_text:
                    issues.append(
                        {
                            "code": "api_repeated_frozen_support",
                            "turn_index": index,
                        }
                    )

    counts = Counter(_text(row.get("code")) or "unknown" for row in issues)
    return {
        "schema_version": API_SEGMENT_VALIDATION_VERSION,
        "pass": not issues,
        "act_id": _text(selected_slots[0].get("act_id")),
        "segment_id": selected_segment_id or selected_act_id,
        "expected_turn_count": len(selected_slots),
        "actual_turn_count": len(current),
        "accepted_prefix_turn_count": len(prefix),
        "minimum_context_word_count": minimum_word_count,
        "actual_context_word_count": current_word_count,
        "context_word_deficit": max(0, minimum_word_count - current_word_count),
        "issue_count": len(issues),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": issues,
        "representation_normalization": representation_rows[first_index:],
        "context_guard": context_guard,
        "cumulative_context_sha256": canonical_sha256(canonical_turns),
    }


def normalize_api_response_envelope(
    plan: Mapping[str, Any],
    response: Any,
    *,
    allow_nested_replay: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Attach local correlation metadata and compile a flat creative turn body."""

    _require_plan(plan)
    if not isinstance(response, Mapping):
        return copy.deepcopy(response), {
            "schema_version": API_RESPONSE_NORMALIZER_VERSION,
            "pass": False,
            "used": False,
            "issues": [{"code": "response_not_object"}],
        }
    raw = copy.deepcopy(dict(response))
    normalized = copy.deepcopy(raw)
    expected_envelope = {
        "schema_version": API_RESPONSE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "frozen_input_hashes": copy.deepcopy(plan["frozen_input_hashes"]),
    }
    injected_fields: list[str] = []
    for field, value in expected_envelope.items():
        if field not in normalized:
            normalized[field] = copy.deepcopy(value)
            injected_fields.append(field)

    issues: list[dict[str, Any]] = []
    context_guard: dict[str, Any] | None = None
    representation_normalization: list[dict[str, Any]] = []
    body_format = "nested_acts"
    creative_before: Any = raw.get("acts")
    creative_after: Any = normalized.get("acts")
    if "turns" in raw:
        body_format = "flat_turn_slots"
        creative_before = raw.get("turns")
        extra_top = sorted(
            set(raw)
            - {"schema_version", "plan_sha256", "frozen_input_hashes", "turns"}
        )
        if extra_top:
            issues.append({"code": "unexpected_flat_response_keys", "keys": extra_top})
        if "acts" in raw:
            issues.append({"code": "ambiguous_dialogue_body_formats"})
        raw_turns = raw.get("turns")
        slots = plan.get("turn_slots")
        if not isinstance(slots, list):
            issues.append({"code": "plan_turn_slots_missing"})
            slots = []
        canonical_turns, representation_normalization, representation_issues = (
            _normalize_flat_context_representation(slots, raw_turns)
        )
        issues.extend(representation_issues)

        # The independent guard receives a canonical view whose structural
        # counts describe the actual, preserved sequence.  The normalizer has
        # already enforced the frozen target range; this prevents the guard's
        # legacy exact-left/right check from rejecting a harmless grouping
        # difference while retaining every semantic mutation check.
        guard_plan = copy.deepcopy(dict(plan))
        guard_slots = guard_plan.get("turn_slots")
        if isinstance(guard_slots, list) and isinstance(canonical_turns, list):
            for index, canonical_row in enumerate(canonical_turns):
                if index >= len(guard_slots) or not isinstance(canonical_row, Mapping):
                    continue
                lead = canonical_row.get("lead_in_sentences")
                follow = canonical_row.get("follow_up_sentences")
                if isinstance(lead, list) and isinstance(follow, list):
                    guard_slots[index]["lead_sentence_count"] = len(lead)
                    guard_slots[index]["follow_sentence_count"] = len(follow)
        context_guard = stage3_dialogue_context_guard.validate_api_authored_context(
            guard_plan, canonical_turns
        )
        issues.extend(copy.deepcopy(context_guard["issues"]))

        # Exact frozen-text repetition is checked over the complete original
        # sentence sequence before any assembly.  It remains independently
        # visible even when another representation issue also rejects the row.
        if isinstance(canonical_turns, list):
            for index, (slot, row) in enumerate(zip(slots, canonical_turns)):
                if not isinstance(row, Mapping):
                    continue
                lead = row.get("lead_in_sentences")
                follow = row.get("follow_up_sentences")
                if not isinstance(lead, list) or not isinstance(follow, list):
                    continue
                authored_text = " ".join(
                    _text(part) for part in lead + follow if _text(part)
                ).casefold()
                for support_row in slot.get("frozen_support") or []:
                    quote = _text(support_row.get("exact_support_quote"))
                    if quote and quote.casefold() in authored_text:
                        issues.append(
                            {
                                "code": "api_repeated_frozen_support",
                                "turn_index": index,
                            }
                        )

        if issues or context_guard.get("pass") is not True:
            return normalized, {
                "schema_version": API_RESPONSE_NORMALIZER_VERSION,
                "pass": False,
                "used": True,
                "body_format": body_format,
                "injected_fields": injected_fields,
                "preserved_explicit_fields": sorted(
                    set(expected_envelope).intersection(raw)
                ),
                "api_writable_fields": ["turns"],
                "creative_body_sha256_before": canonical_sha256(raw_turns),
                "creative_body_sha256_after": canonical_sha256(raw_turns),
                "canonical_context_sha256": canonical_sha256(canonical_turns),
                "truth_fields_modified": [],
                "representation_normalization": copy.deepcopy(
                    representation_normalization
                ),
                "context_guard": copy.deepcopy(context_guard),
                "assembled_dialogue_sha256": None,
                "issues": issues,
            }
        if not isinstance(canonical_turns, list):
            issues.append({"code": "flat_turns_not_array"})
            canonical_turns = []
        if len(canonical_turns) != len(slots):
            issues.append(
                {
                    "code": "flat_turn_count_mismatch",
                    "expected": len(slots),
                    "actual": len(canonical_turns),
                }
            )
        nested_by_act: dict[str, dict[str, Any]] = {
            act["act_id"]: {
                "act_id": act["act_id"],
                "beats": [
                    {"beat_id": beat["beat_id"], "turns": []}
                    for beat in act["beats"]
                ],
            }
            for act in plan["acts"]
        }
        beat_targets = {
            (act["act_id"], beat["beat_id"]): nested_by_act[act["act_id"]]["beats"][beat_index]["turns"]
            for act in plan["acts"]
            for beat_index, beat in enumerate(act["beats"])
        }
        for index, slot in enumerate(slots):
            if index >= len(canonical_turns):
                break
            row = canonical_turns[index]
            if not isinstance(row, Mapping):
                issues.append({"code": "flat_turn_not_object", "turn_index": index})
                continue
            allowed = {
                "turn_slot_id", "lead_in_sentences", "follow_up_sentences",
            }
            extra = sorted(set(row) - allowed)
            if extra:
                issues.append(
                    {"code": "unexpected_flat_turn_keys", "turn_index": index, "keys": extra}
                )
            if _text(row.get("turn_slot_id")) != slot["turn_slot_id"]:
                issues.append({"code": "turn_slot_order_mismatch", "turn_index": index})
            lead_sentences = row.get("lead_in_sentences")
            follow_sentences = row.get("follow_up_sentences")
            for field, sentences in (
                ("lead_in_sentences", lead_sentences),
                ("follow_up_sentences", follow_sentences),
            ):
                if not isinstance(sentences, list):
                    issues.append(
                        {
                            "code": "flat_turn_sentences_not_array",
                            "turn_index": index,
                            "field": field,
                        }
                    )
                    continue
                for sentence_index, sentence in enumerate(sentences):
                    if not isinstance(sentence, str) or len(_WORD_RE.findall(sentence)) < 6:
                        issues.append(
                            {
                                "code": "flat_turn_sentence_not_substantive",
                                "turn_index": index,
                                "field": field,
                                "sentence_index": sentence_index,
                            }
                        )
            if not isinstance(lead_sentences, list):
                lead_sentences = []
            if not isinstance(follow_sentences, list):
                follow_sentences = []
            support = copy.deepcopy(slot["frozen_support"])
            quotes = [item["exact_support_quote"] for item in support]
            text_parts = list(lead_sentences) + quotes + list(follow_sentences)
            text = " ".join(_text(part) for part in text_parts if _text(part))
            target = beat_targets.get((slot["act_id"], slot["beat_id"]))
            if target is not None:
                target.append(
                    {
                        "turn_id": slot["turn_id"],
                        "speaker": slot["speaker"],
                        "text": text,
                        "support": copy.deepcopy(support),
                    }
                )
        normalized.pop("turns", None)
        normalized["acts"] = [nested_by_act[act["act_id"]] for act in plan["acts"]]
        creative_after = copy.deepcopy(raw_turns)
    elif not allow_nested_replay:
        issues.append({"code": "flat_turn_body_required_for_live"})
    return normalized, {
        "schema_version": API_RESPONSE_NORMALIZER_VERSION,
        "pass": not issues,
        "used": bool(injected_fields) or body_format == "flat_turn_slots",
        "body_format": body_format,
        "injected_fields": injected_fields,
        "preserved_explicit_fields": sorted(set(expected_envelope).intersection(raw)),
        "api_writable_fields": (
            ["turns"]
            if body_format == "flat_turn_slots"
            else ["acts"] if allow_nested_replay else []
        ),
        "creative_body_sha256_before": canonical_sha256(creative_before),
        "creative_body_sha256_after": canonical_sha256(creative_after),
        "canonical_context_sha256": (
            canonical_sha256(canonical_turns)
            if body_format == "flat_turn_slots"
            else None
        ),
        "truth_fields_modified": [],
        "representation_normalization": copy.deepcopy(
            representation_normalization
        ),
        "context_guard": copy.deepcopy(context_guard),
        "assembled_dialogue_sha256": (
            canonical_sha256(normalized.get("acts"))
            if isinstance(normalized.get("acts"), list)
            else None
        ),
        "issues": issues,
    }


def _nested_forbidden_keys(value: Any, path: str = "$") -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_token = _token(key)
            child_path = f"{path}.{key}"
            if key_token in _FORBIDDEN_RESPONSE_KEYS:
                issues.append(
                    _issue(
                        "forbidden_truth_mutation_surface",
                        child_path,
                        "API response contains a field that could mutate frozen truth.",
                        key=key,
                    )
                )
            issues.extend(_nested_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_nested_forbidden_keys(child, f"{path}[{index}]"))
    return issues


def _allowed_exact_keys(
    value: Any, allowed: set[str], path: str, issues: list[dict[str, Any]]
) -> bool:
    if not isinstance(value, Mapping):
        issues.append(_issue("invalid_object", path, "Expected a JSON object."))
        return False
    extra = sorted(set(value) - allowed)
    if extra:
        issues.append(
            _issue(
                "unexpected_response_keys",
                path,
                "API response contains fields outside the dialogue-only schema.",
                keys=extra,
            )
        )
    return True


def _topic_overlap(text: str, anchors: set[str]) -> bool:
    words = {word.casefold().strip("'-") for word in _WORD_RE.findall(text)}
    return bool(words & anchors)


def _quote_occurrences(texts: Sequence[str], quote: str) -> int:
    return sum(text.count(quote) for text in texts)


def _semantic_quote_issues(
    claim: Mapping[str, Any], quote: str, path: str
) -> list[dict[str, Any]]:
    return stage3_claim_validator._validate_quote_semantics(  # noqa: SLF001
        claim, quote, path=path
    )


def _typed_semantic_quote_issues(
    claim: Mapping[str, Any], quote: str, path: str
) -> list[dict[str, Any]]:
    try:
        frozen_record = stage3_truth_support.compile_typed_frozen_support(claim)
    except stage3_truth_support.TruthSupportError as error:
        return [
            _issue(
                "typed_support_compile_failed",
                path,
                "The locally frozen typed support could not be compiled.",
                error=_text(error),
            )
        ]
    candidate_record = copy.deepcopy(frozen_record)
    candidate_record["quote"] = quote
    # Keep the original signature: any changed quote must fail the independent
    # record-hash check as well as the semantic checks below.
    report = stage3_truth_support.validate_typed_frozen_support(
        claim, candidate_record
    )
    return [
        _issue(
            f"typed_support_{item.get('code') or 'invalid'}",
            path,
            _text(item.get("message")) or "Typed support validation failed.",
            semantic_path=item.get("path"),
            typed_details=copy.deepcopy(item.get("details")),
        )
        for item in report["issues"]
    ]


def _status_realization_instruction(status_value: Any) -> str:
    status = _token(status_value)
    return {
        "": "State the relation directly without adding uncertainty or an inactive qualifier.",
        "active": "State the relation as current and in effect, without uncertainty or an inactive qualifier.",
        "current": "State the relation as current and in effect, without uncertainty or an inactive qualifier.",
        "belief": "Explicitly say the relation is uncertain or only believed; do not state it as confirmed fact.",
        "report": "Explicitly say the relation is reported or unconfirmed; do not state it as confirmed fact.",
        "reported": "Explicitly say the relation is reported or unconfirmed; do not state it as confirmed fact.",
        "outdated": "Explicitly say the relation is outdated or no longer applies; past tense alone is insufficient.",
        "expired": "Explicitly say the relation expired or is no longer valid; past tense alone is insufficient.",
        "canceled": "Explicitly say the relation was canceled or did not take effect.",
        "cancelled": "Explicitly say the relation was canceled or did not take effect.",
        "scope_limited": "Explicitly preserve the limited or restricted scope.",
        "delayed_not_active": "Explicitly say the relation is delayed and not yet active.",
    }.get(
        status,
        f"Explicitly express the canonical status {status!r} without changing it.",
    )


def build_live_content_binding_probe_contract(
    claims: Sequence[Mapping[str, Any]],
    *,
    topic: str,
    setting: str,
) -> dict[str, Any]:
    """Build the bounded live contract used before any scene generation.

    This is deliberately a production-shaped probe rather than a synthetic
    fixture: the selected canonical claims come from an admitted graph, and
    the API owns only their natural-language surface realization.
    """

    selected = [copy.deepcopy(dict(claim)) for claim in claims]
    if not 2 <= len(selected) <= 4:
        raise DiscourseContractError(
            "a live content-binding probe requires between two and four claims"
        )
    claim_ids: list[str] = []
    obligations: list[dict[str, Any]] = []
    for index, claim in enumerate(selected):
        claim_id = _text(claim.get("claim_id"))
        if not claim_id or claim_id in claim_ids:
            raise DiscourseContractError(
                "live content-binding claim ids must be non-empty and unique"
            )
        claim_issues = stage3_claim_validator.validate_claim(
            claim, path=f"$.claims[{index}]"
        )
        if claim_issues:
            codes = sorted({_text(item.get("code")) for item in claim_issues})
            raise DiscourseContractError(
                f"claim {claim_id!r} is not canonical: {codes}"
            )
        canonical = claim.get("canonical_support_claim")
        if not isinstance(canonical, Mapping):
            raise DiscourseContractError(
                f"claim {claim_id!r} lacks canonical_support_claim"
            )
        claim_ids.append(claim_id)
        obligations.append(
            {
                "claim_ref": f"C{index + 1}",
                "immutable_canonical_fact": copy.deepcopy(dict(canonical)),
                "required_status_realization": _status_realization_instruction(
                    canonical.get("status")
                ),
            }
        )

    contract: dict[str, Any] = {
        "schema_version": LIVE_CONTENT_BINDING_CONTRACT_SCHEMA_VERSION,
        "probe_kind": "content_binding_live_canary",
        "selected_claims_sha256": canonical_sha256(selected),
        "topic": _text(topic) or "an ongoing conversation",
        "setting": _text(setting) or "a conversation between established participants",
        "claim_obligations": obligations,
        "generation_rules": [
            "Return one natural standalone support sentence for every immutable canonical fact.",
            "Preserve subject, predicate, object roles, polarity, status, direction, condition, scope, and any surfaced time exactly; obey every required_status_realization literally.",
            "Do not add a second fact, alternative value, qualification, or contradiction to a support sentence.",
            "Do not mention APIs, prompts, claims, identifiers, graphs, schemas, gold answers, validators, or datasets in visible text.",
            "Return each opaque claim_ref only as JSON sidecar metadata; never place it inside exact_support_quote.",
        ],
        "response_schema": {
            "top_level_type": "object",
            "top_level_required_keys": ["realizations"],
            "top_level_additional_keys_allowed": False,
            "realizations": {
                "type": "array",
                "item_count_exact": len(obligations),
                "claim_refs_exactly_once": [
                    obligation["claim_ref"] for obligation in obligations
                ],
                "item_schema": {
                    "required_keys": ["claim_ref", "exact_support_quote"],
                    "additional_keys_allowed": False,
                },
            },
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_live_content_binding_probe_response(
    contract: Mapping[str, Any],
    response: Any,
    *,
    claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one raw live content-binding response without repair."""

    issues: list[dict[str, Any]] = []
    contract_copy = copy.deepcopy(dict(contract)) if isinstance(contract, Mapping) else {}
    claimed_contract_hash = _text(contract_copy.pop("contract_sha256", ""))
    if (
        contract_copy.get("schema_version")
        != LIVE_CONTENT_BINDING_CONTRACT_SCHEMA_VERSION
        or not claimed_contract_hash
        or canonical_sha256(contract_copy) != claimed_contract_hash
    ):
        issues.append(
            _issue(
                "invalid_live_binding_contract",
                "$.contract",
                "The live content-binding contract is missing, changed, or corrupt.",
            )
        )

    selected = [copy.deepcopy(dict(claim)) for claim in claims]
    if canonical_sha256(selected) != contract.get("selected_claims_sha256"):
        issues.append(
            _issue(
                "live_binding_claim_hash_mismatch",
                "$.claims",
                "The selected claims changed after the live contract was built.",
            )
        )
    obligations = contract.get("claim_obligations")
    obligations = obligations if isinstance(obligations, list) else []
    expected_refs = [
        _text(row.get("claim_ref"))
        for row in obligations
        if isinstance(row, Mapping)
    ]
    claim_by_ref = {
        claim_ref: selected[index]
        for index, claim_ref in enumerate(expected_refs)
        if claim_ref and index < len(selected)
    }

    if not isinstance(response, Mapping):
        issues.append(
            _issue(
                "live_binding_response_not_object",
                "$.response",
                "The response must be a JSON object.",
            )
        )
        return _live_binding_report(issues, expected_refs, [], {})
    extra_top = sorted(set(response) - {"realizations"})
    if extra_top:
        issues.append(
            _issue(
                "live_binding_unexpected_response_keys",
                "$.response",
                "The response contains unexpected top-level keys.",
                keys=extra_top,
            )
        )
    raw_realizations = response.get("realizations")
    if not isinstance(raw_realizations, list):
        issues.append(
            _issue(
                "live_binding_realizations_not_array",
                "$.response.realizations",
                "realizations must be an array.",
            )
        )
        return _live_binding_report(issues, expected_refs, [], {})
    if len(raw_realizations) != len(expected_refs):
        issues.append(
            _issue(
                "live_binding_realization_count_mismatch",
                "$.response.realizations",
                "The response must realize every requested claim exactly once.",
                expected=len(expected_refs),
                actual=len(raw_realizations),
            )
        )

    observed_refs: list[str] = []
    quote_by_id: dict[str, str] = {}
    quote_rows: list[tuple[str, str, str]] = []
    for index, row in enumerate(raw_realizations):
        path = f"$.response.realizations[{index}]"
        if not isinstance(row, Mapping):
            issues.append(
                _issue(
                    "live_binding_realization_not_object",
                    path,
                    "Every realization must be an object.",
                )
            )
            continue
        extra = sorted(set(row) - {"claim_ref", "exact_support_quote"})
        missing = sorted({"claim_ref", "exact_support_quote"} - set(row))
        if extra:
            issues.append(
                _issue(
                    "live_binding_unexpected_realization_keys",
                    path,
                    "A realization contains unexpected keys.",
                    keys=extra,
                )
            )
        if missing:
            issues.append(
                _issue(
                    "live_binding_missing_realization_keys",
                    path,
                    "A realization is missing required keys.",
                    keys=missing,
                )
            )
        claim_ref = _text(row.get("claim_ref"))
        quote = row.get("exact_support_quote")
        observed_refs.append(claim_ref)
        if claim_ref not in claim_by_ref:
            issues.append(
                _issue(
                    "live_binding_unknown_claim_ref",
                    f"{path}.claim_ref",
                    "The response returned an unrequested opaque claim reference.",
                    claim_ref=claim_ref,
                )
            )
            continue
        claim = claim_by_ref[claim_ref]
        claim_id = _text(claim.get("claim_id"))
        if not isinstance(quote, str) or not _text(quote):
            issues.append(
                _issue(
                    "live_binding_quote_invalid",
                    f"{path}.exact_support_quote",
                    "The support quote must be non-empty text.",
                    claim_id=claim_id,
                )
            )
            continue
        quote_by_id.setdefault(claim_id, quote)
        quote_path = f"{path}.exact_support_quote"
        quote_rows.append((claim_id, quote, quote_path))
        if _META_RE.search(quote):
            issues.append(
                _issue(
                    "live_binding_internal_meta_leak",
                    quote_path,
                    "The visible support quote exposes internal pipeline terminology.",
                    claim_id=claim_id,
                )
            )
        if claim_id.casefold() in quote.casefold():
            issues.append(
                _issue(
                    "live_binding_internal_identifier_leak",
                    quote_path,
                    "The visible support quote exposes its internal claim id.",
                    claim_id=claim_id,
                )
            )
        issues.extend(
            _semantic_quote_issues(claim, quote, quote_path)
        )
        contradiction_rows = (
            stage3_dialogue_context_guard._claim_contradiction_issues(  # noqa: SLF001
                quote, quote_path, selected
            )
        )
        issues.extend(
            row
            for row in contradiction_rows
            if _text(row.get("code"))
            != "api_authored_answer_bearing_relation"
        )

        other_matches = []
        for other_ref, other_claim in claim_by_ref.items():
            if other_ref == claim_ref:
                continue
            if not _semantic_quote_issues(other_claim, quote, quote_path):
                other_matches.append(_text(other_claim.get("claim_id")))
        if other_matches:
            issues.append(
                _issue(
                    "live_binding_quote_semantically_ambiguous",
                    quote_path,
                    "One support quote fully realizes more than one selected claim.",
                    claim_id=claim_id,
                    also_matches=other_matches,
                )
            )

    counts = Counter(observed_refs)
    missing_refs = sorted(set(expected_refs) - set(observed_refs))
    duplicate_refs = sorted(
        claim_ref for claim_ref, count in counts.items() if claim_ref and count > 1
    )
    if missing_refs:
        issues.append(
            _issue(
                "live_binding_missing_claim_coverage",
                "$.response.realizations",
                "Requested claims are missing from the response.",
                claim_refs=missing_refs,
            )
        )
    if duplicate_refs:
        issues.append(
            _issue(
                "live_binding_duplicate_claim_coverage",
                "$.response.realizations",
                "A requested claim was returned more than once.",
                claim_refs=duplicate_refs,
            )
        )
    for left_index, (left_id, left_quote, left_path) in enumerate(quote_rows):
        for right_id, right_quote, _right_path in quote_rows[left_index + 1 :]:
            if _text(left_quote).casefold() == _text(right_quote).casefold():
                issues.append(
                    _issue(
                        "live_binding_exact_duplicate_quote",
                        left_path,
                        "Two claims received the same support quote.",
                        claim_ids=[left_id, right_id],
                    )
                )
            elif stage3_dialogue_context_guard._near_duplicate(  # noqa: SLF001
                left_quote, right_quote
            ):
                issues.append(
                    _issue(
                        "live_binding_near_duplicate_quote",
                        left_path,
                        "Two claims received near-duplicate support quotes.",
                        claim_ids=[left_id, right_id],
                    )
                )
    return _live_binding_report(issues, expected_refs, observed_refs, quote_by_id)


def _live_binding_report(
    issues: Sequence[Mapping[str, Any]],
    expected_refs: Sequence[str],
    observed_refs: Sequence[str],
    quote_by_id: Mapping[str, str],
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(item)) for item in issues]
    counts = Counter(_text(item.get("code")) or "unknown" for item in rows)
    observed_counts = Counter(observed_refs)
    return {
        "schema_version": LIVE_CONTENT_BINDING_VALIDATION_SCHEMA_VERSION,
        "pass": not rows,
        "issue_count": len(rows),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": rows,
        "expected_claim_refs": list(expected_refs),
        "observed_claim_refs": list(observed_refs),
        "missing_claim_refs": sorted(set(expected_refs) - set(observed_refs)),
        "duplicate_claim_refs": sorted(
            claim_ref
            for claim_ref, count in observed_counts.items()
            if claim_ref and count > 1
        ),
        "quote_by_claim_id": copy.deepcopy(dict(quote_by_id)),
    }


def build_live_single_scene_probe_contract(
    claims: Sequence[Mapping[str, Any]],
    *,
    topic: str,
    setting: str,
    speaker_roster: Sequence[str],
) -> dict[str, Any]:
    """Build the minimal variable-length scene canary contract."""

    binding_contract = build_live_content_binding_probe_contract(
        claims, topic=topic, setting=setting
    )
    speakers = _ordered_unique(_text(value) for value in speaker_roster)
    if len(speakers) < 2:
        raise DiscourseContractError(
            "a live single-scene probe requires at least two established speakers"
        )
    obligations = copy.deepcopy(binding_contract["claim_obligations"])
    for obligation in obligations:
        canonical = obligation.get("immutable_canonical_fact")
        canonical = canonical if isinstance(canonical, Mapping) else {}
        subject = _text(canonical.get("subject")).casefold()
        obligation["preferred_support_speaker"] = next(
            (speaker for speaker in speakers if speaker.casefold() != subject),
            speakers[0],
        )
    contract: dict[str, Any] = {
        "schema_version": LIVE_SINGLE_SCENE_CONTRACT_SCHEMA_VERSION,
        "probe_kind": "single_scene_live_canary",
        "selected_claims_sha256": binding_contract["selected_claims_sha256"],
        "topic": binding_contract["topic"],
        "setting": binding_contract["setting"],
        "speaker_roster": speakers,
        "claim_obligations": obligations,
        "generation_rules": [
            "Write one natural direct conversation of 4-8 turns.",
            "Use only the listed speakers.",
            "Express every immutable fact exactly once; do not move one claim's status or value onto another claim.",
            "Return one support row per claim_ref; exact_support_quote must be a byte-for-byte substring of the referenced turn.",
            "Each exact_support_quote must be self-contained and explicitly name its canonical subject, predicate, object, and required status; do not rely on pronouns or another turn.",
            "Prefer the obligation's preferred_support_speaker for its supporting turn so the named subject is natural in the quote.",
            "Preserve every canonical fact and required_status_realization; belief/report is not confirmed, active is not canceled, and outdated is not current.",
        ],
        "response_schema": {
            "turns": [{"speaker": "speaker_roster value", "text": "visible dialogue"}],
            "supports": [
                {
                    "claim_ref": "opaque C1-C4 value",
                    "turn_index": "zero-based integer",
                    "exact_support_quote": "exact substring of turns[turn_index].text",
                }
            ],
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_live_single_scene_probe_response(
    contract: Mapping[str, Any],
    response: Any,
    *,
    claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply only the minimal gates proven necessary before the first live scene."""

    issues: list[dict[str, Any]] = []
    contract_copy = copy.deepcopy(dict(contract)) if isinstance(contract, Mapping) else {}
    claimed_hash = _text(contract_copy.pop("contract_sha256", ""))
    if (
        contract_copy.get("schema_version")
        != LIVE_SINGLE_SCENE_CONTRACT_SCHEMA_VERSION
        or not claimed_hash
        or canonical_sha256(contract_copy) != claimed_hash
    ):
        issues.append(
            _issue(
                "invalid_live_scene_contract",
                "$.contract",
                "The frozen scene contract changed before validation.",
            )
        )

    selected = [copy.deepcopy(dict(claim)) for claim in claims]
    if canonical_sha256(selected) != contract.get("selected_claims_sha256"):
        issues.append(
            _issue(
                "live_scene_claim_hash_mismatch",
                "$.claims",
                "The selected frozen claims changed before validation.",
            )
        )
    obligations = contract.get("claim_obligations")
    obligations = obligations if isinstance(obligations, list) else []
    expected_refs = [
        _text(row.get("claim_ref"))
        for row in obligations
        if isinstance(row, Mapping)
    ]
    claim_by_ref = {
        claim_ref: selected[index]
        for index, claim_ref in enumerate(expected_refs)
        if claim_ref and index < len(selected)
    }
    roster = {
        _text(value)
        for value in contract.get("speaker_roster") or []
        if _text(value)
    }

    if not isinstance(response, Mapping):
        issues.append(
            _issue(
                "live_scene_response_not_object",
                "$.response",
                "The parsed response must be a JSON object.",
            )
        )
        return _minimal_live_scene_report(issues, expected_refs, [], [], {})

    raw_turns = response.get("turns")
    raw_supports = response.get("supports")
    if not isinstance(raw_turns, list):
        issues.append(
            _issue(
                "live_scene_turns_not_array",
                "$.response.turns",
                "turns must be an array.",
            )
        )
        raw_turns = []
    if not isinstance(raw_supports, list):
        issues.append(
            _issue(
                "live_scene_supports_not_array",
                "$.response.supports",
                "supports must be an array.",
            )
        )
        raw_supports = []

    turn_texts: list[str] = []
    for index, row in enumerate(raw_turns):
        path = f"$.response.turns[{index}]"
        if not isinstance(row, Mapping):
            issues.append(
                _issue("live_scene_turn_not_object", path, "A turn must be an object.")
            )
            turn_texts.append("")
            continue
        speaker = _text(row.get("speaker"))
        text = row.get("text")
        if speaker not in roster:
            issues.append(
                _issue(
                    "live_scene_speaker_outside_roster",
                    f"{path}.speaker",
                    "The turn speaker is not in the allowed parent roster.",
                    speaker=speaker,
                )
            )
        if not isinstance(text, str) or not _text(text):
            issues.append(
                _issue(
                    "live_scene_turn_text_invalid",
                    f"{path}.text",
                    "The turn text must be non-empty.",
                )
            )
            turn_texts.append("")
        else:
            turn_texts.append(text)

    observed_refs: list[str] = []
    resolved_supports: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_supports):
        path = f"$.response.supports[{index}]"
        if not isinstance(row, Mapping):
            issues.append(
                _issue(
                    "live_scene_support_not_object",
                    path,
                    "A support binding must be an object.",
                )
            )
            continue
        claim_ref = _text(row.get("claim_ref"))
        turn_index = row.get("turn_index")
        quote = row.get("exact_support_quote")
        observed_refs.append(claim_ref)
        claim = claim_by_ref.get(claim_ref)
        if claim is None:
            issues.append(
                _issue(
                    "live_scene_unknown_claim_ref",
                    f"{path}.claim_ref",
                    "The response returned an unrequested claim reference.",
                    claim_ref=claim_ref,
                )
            )
            continue
        claim_id = _text(claim.get("claim_id"))
        if (
            not isinstance(turn_index, int)
            or isinstance(turn_index, bool)
            or not 0 <= turn_index < len(turn_texts)
        ):
            issues.append(
                _issue(
                    "live_scene_support_turn_index_invalid",
                    f"{path}.turn_index",
                    "turn_index must identify one returned turn.",
                    claim_id=claim_id,
                )
            )
            continue
        if not isinstance(quote, str) or not _text(quote):
            issues.append(
                _issue(
                    "live_scene_support_quote_invalid",
                    f"{path}.exact_support_quote",
                    "The support quote must be non-empty text.",
                    claim_id=claim_id,
                )
            )
            continue
        if quote not in turn_texts[turn_index]:
            issues.append(
                _issue(
                    "live_scene_support_quote_not_in_bound_turn",
                    f"{path}.exact_support_quote",
                    "The support quote is not an exact substring of its referenced turn.",
                    claim_id=claim_id,
                )
            )
        issues.extend(
            _semantic_quote_issues(
                claim, quote, f"{path}.exact_support_quote"
            )
        )
        resolved_supports[claim_id] = {
            "claim_ref": claim_ref,
            "turn_index": turn_index,
            "exact_support_quote": quote,
        }

    observed_counts = Counter(observed_refs)
    missing_refs = sorted(set(expected_refs) - set(observed_refs))
    duplicate_refs = sorted(
        ref for ref, count in observed_counts.items() if ref and count > 1
    )
    if missing_refs:
        issues.append(
            _issue(
                "live_scene_missing_claim_coverage",
                "$.response.supports",
                "Every requested claim must have one support binding.",
                claim_refs=missing_refs,
            )
        )
    if duplicate_refs:
        issues.append(
            _issue(
                "live_scene_duplicate_claim_coverage",
                "$.response.supports",
                "Each requested claim may have only one support binding.",
                claim_refs=duplicate_refs,
            )
        )
    return _minimal_live_scene_report(
        issues, expected_refs, observed_refs, turn_texts, resolved_supports
    )


def _minimal_live_scene_report(
    issues: Sequence[Mapping[str, Any]],
    expected_refs: Sequence[str],
    observed_refs: Sequence[str],
    turn_texts: Sequence[str],
    resolved_supports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(item)) for item in issues]
    counts = Counter(_text(item.get("code")) or "unknown" for item in rows)
    observed_counts = Counter(observed_refs)
    return {
        "schema_version": LIVE_SINGLE_SCENE_VALIDATION_SCHEMA_VERSION,
        "pass": not rows,
        "issue_count": len(rows),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": rows,
        "turn_count": len(turn_texts),
        "word_count": sum(len(_WORD_RE.findall(text)) for text in turn_texts),
        "expected_claim_refs": list(expected_refs),
        "observed_claim_refs": list(observed_refs),
        "missing_claim_refs": sorted(set(expected_refs) - set(observed_refs)),
        "duplicate_claim_refs": sorted(
            ref for ref, count in observed_counts.items() if ref and count > 1
        ),
        "resolved_supports_by_claim_id": copy.deepcopy(dict(resolved_supports)),
    }


def validate_api_response(
    plan: Mapping[str, Any],
    response: Any,
    *,
    frozen_graph: Mapping[str, Any],
    frozen_claim_bundle: Mapping[str, Any],
    parent_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate dialogue generation without repairing or accepting ambiguity."""

    issues: list[dict[str, Any]] = []
    try:
        _require_plan(plan)
    except DiscourseContractError as error:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "pass": False,
            "issue_count": 1,
            "issue_counts_by_code": {"invalid_frozen_plan": 1},
            "issues": [
                _issue("invalid_frozen_plan", "$.plan", str(error))
            ],
        }
    expected_hashes = plan["frozen_input_hashes"]
    supplied_inputs = {
        "graph_sha256": frozen_graph,
        "claim_bundle_sha256": frozen_claim_bundle,
        "parent_context_sha256": parent_context,
    }
    for key, value in supplied_inputs.items():
        try:
            actual = canonical_sha256(value)
        except DiscourseContractError as error:
            issues.append(_issue("invalid_frozen_input", f"$.{key}", str(error)))
            continue
        if actual != expected_hashes.get(key):
            issues.append(
                _issue(
                    "frozen_input_hash_mismatch",
                    f"$.{key}",
                    "A frozen input changed after discourse planning.",
                    expected=expected_hashes.get(key),
                    actual=actual,
                )
            )

    if not isinstance(response, Mapping):
        issues.append(_issue("invalid_response_type", "$.response", "Response must be an object."))
        return _report(issues)
    issues.extend(_nested_forbidden_keys(response, "$.response"))
    if not _allowed_exact_keys(
        response,
        {"schema_version", "plan_sha256", "frozen_input_hashes", "acts"},
        "$.response",
        issues,
    ):
        return _report(issues)
    if response.get("schema_version") != API_RESPONSE_SCHEMA_VERSION:
        issues.append(
            _issue("response_schema_mismatch", "$.response.schema_version", "Wrong response schema.")
        )
    if _text(response.get("plan_sha256")) != plan["plan_sha256"]:
        issues.append(
            _issue("response_plan_hash_mismatch", "$.response.plan_sha256", "Response targets another plan.")
        )
    if response.get("frozen_input_hashes") != expected_hashes:
        issues.append(
            _issue(
                "response_frozen_hashes_mismatch",
                "$.response.frozen_input_hashes",
                "Response did not echo the exact frozen input hashes.",
            )
        )

    response_acts = response.get("acts")
    if not isinstance(response_acts, list):
        issues.append(_issue("invalid_acts", "$.response.acts", "acts must be an array."))
        return _report(issues)
    expected_acts = plan["acts"]
    if len(response_acts) != len(expected_acts):
        issues.append(
            _issue(
                "act_count_mismatch",
                "$.response.acts",
                "Response must return every planned act exactly once.",
                expected=len(expected_acts),
                actual=len(response_acts),
            )
        )

    roster = set(plan["speaker_roster"])
    anchors = set(plan["topic_spine"]["anchor_terms"])
    semantic_by_id = {
        _text(claim.get("claim_id")): claim for claim in plan["semantic_claims"]
    }
    expected_claim_ids = set(plan["required_claim_ids"])
    assigned_claim_ids: list[str] = []
    all_turn_ids: list[str] = []
    all_texts: list[str] = []
    all_turn_rows: list[dict[str, Any]] = []
    quote_by_claim: dict[str, str] = {}

    for act_index, expected_act in enumerate(expected_acts):
        if act_index >= len(response_acts):
            break
        raw_act = response_acts[act_index]
        act_path = f"$.response.acts[{act_index}]"
        if not _allowed_exact_keys(raw_act, {"act_id", "beats"}, act_path, issues):
            continue
        if _text(raw_act.get("act_id")) != expected_act["act_id"]:
            issues.append(
                _issue("act_order_mismatch", f"{act_path}.act_id", "Act id or order differs from the plan.")
            )
        raw_beats = raw_act.get("beats")
        if not isinstance(raw_beats, list):
            issues.append(_issue("invalid_beats", f"{act_path}.beats", "beats must be an array."))
            continue
        expected_beats = expected_act["beats"]
        if len(raw_beats) != len(expected_beats):
            issues.append(
                _issue(
                    "beat_count_mismatch",
                    f"{act_path}.beats",
                    "Response must return every planned beat exactly once.",
                    expected=len(expected_beats),
                    actual=len(raw_beats),
                )
            )
        for beat_index, expected_beat in enumerate(expected_beats):
            if beat_index >= len(raw_beats):
                break
            raw_beat = raw_beats[beat_index]
            beat_path = f"{act_path}.beats[{beat_index}]"
            if not _allowed_exact_keys(raw_beat, {"beat_id", "turns"}, beat_path, issues):
                continue
            if _text(raw_beat.get("beat_id")) != expected_beat["beat_id"]:
                issues.append(
                    _issue("beat_order_mismatch", f"{beat_path}.beat_id", "Beat id or order differs from the plan.")
                )
            raw_turns = raw_beat.get("turns")
            if not isinstance(raw_turns, list) or not raw_turns:
                issues.append(_issue("invalid_turns", f"{beat_path}.turns", "Each beat needs turns."))
                continue
            if len(raw_turns) > MAX_TURNS_PER_BEAT:
                issues.append(
                    _issue("too_many_turns", f"{beat_path}.turns", "Beat exceeds the local turn cap.")
                )
            allowed_claim_ids = set(expected_beat["required_claim_ids"])
            for turn_index, raw_turn in enumerate(raw_turns):
                turn_path = f"{beat_path}.turns[{turn_index}]"
                if not _allowed_exact_keys(
                    raw_turn,
                    {"turn_id", "speaker", "text", "support"},
                    turn_path,
                    issues,
                ):
                    continue
                turn_id = _text(raw_turn.get("turn_id"))
                speaker = _text(raw_turn.get("speaker"))
                text = raw_turn.get("text")
                support = raw_turn.get("support")
                if not turn_id:
                    issues.append(_issue("invalid_turn_id", f"{turn_path}.turn_id", "turn_id is required."))
                else:
                    all_turn_ids.append(turn_id)
                if speaker not in roster:
                    issues.append(
                        _issue(
                            "speaker_outside_roster",
                            f"{turn_path}.speaker",
                            "Speaker is not in the frozen parent roster.",
                            speaker=speaker,
                        )
                    )
                if not isinstance(text, str) or not _text(text):
                    issues.append(_issue("invalid_turn_text", f"{turn_path}.text", "Turn text is required."))
                    continue
                all_texts.append(text)
                support_rows = support if isinstance(support, list) else []
                all_turn_rows.append(
                    {
                        "path": turn_path,
                        "text": text,
                        "has_support": bool(support_rows),
                    }
                )
                if not isinstance(support, list):
                    issues.append(_issue("invalid_support", f"{turn_path}.support", "support must be an array."))
                    continue
                if _META_RE.search(text):
                    issues.append(
                        _issue("visible_internal_meta", f"{turn_path}.text", "Visible dialogue mentions internal pipeline concepts.")
                    )
                word_count = len(_WORD_RE.findall(text))
                if word_count < 6 or _GENERIC_FILLER_RE.fullmatch(_text(text)):
                    issues.append(
                        _issue("generic_filler_turn", f"{turn_path}.text", "Turn is too short or generic to be substantive.")
                    )
                if not support_rows and not _topic_overlap(text, anchors):
                    issues.append(
                        _issue("unrelated_filler_turn", f"{turn_path}.text", "Unsupported turn leaves the topic spine.")
                    )
                for support_index, raw_support in enumerate(support_rows):
                    support_path = f"{turn_path}.support[{support_index}]"
                    if not _allowed_exact_keys(
                        raw_support,
                        {"claim_id", "exact_support_quote"},
                        support_path,
                        issues,
                    ):
                        continue
                    claim_id = _text(raw_support.get("claim_id"))
                    quote = raw_support.get("exact_support_quote")
                    assigned_claim_ids.append(claim_id)
                    if claim_id not in allowed_claim_ids:
                        issues.append(
                            _issue(
                                "claim_assigned_to_wrong_beat",
                                f"{support_path}.claim_id",
                                "Claim is not assigned to this planned beat.",
                                claim_id=claim_id,
                            )
                        )
                    if claim_id not in semantic_by_id:
                        issues.append(
                            _issue("unknown_claim_id", f"{support_path}.claim_id", "Unknown support claim.")
                        )
                    if not isinstance(quote, str) or not quote:
                        issues.append(
                            _issue("invalid_exact_support_quote", f"{support_path}.exact_support_quote", "Quote is required.")
                        )
                        continue
                    quote_by_claim.setdefault(claim_id, quote)
                    if quote not in text:
                        issues.append(
                            _issue(
                                "exact_quote_not_substring",
                                f"{support_path}.exact_support_quote",
                                "Exact support quote is not a byte-for-byte substring of its turn.",
                            )
                        )
                    remainder = text.replace(quote, " ", 1)
                    if len(_WORD_RE.findall(remainder)) < 4:
                        issues.append(
                            _issue(
                                "isolated_fact_turn",
                                f"{turn_path}.text",
                                "A support clause must be embedded in substantive surrounding discourse.",
                                claim_id=claim_id,
                            )
                        )
                    claim = semantic_by_id.get(claim_id)
                    if claim is not None:
                        issues.extend(
                            _typed_semantic_quote_issues(
                                claim, quote, support_path
                            )
                        )

    if len(all_turn_ids) != len(set(all_turn_ids)):
        duplicates = sorted(
            turn_id for turn_id, count in Counter(all_turn_ids).items() if count > 1
        )
        issues.append(
            _issue("duplicate_turn_id", "$.response.acts", "turn_id values must be globally unique.", turn_ids=duplicates)
        )
    assignment_counts = Counter(assigned_claim_ids)
    missing_claims = sorted(expected_claim_ids - set(assignment_counts))
    duplicate_claims = sorted(
        claim_id for claim_id, count in assignment_counts.items() if count > 1
    )
    if missing_claims:
        issues.append(
            _issue("missing_claim_coverage", "$.response.acts", "Required claims are not covered.", claim_ids=missing_claims)
        )
    if duplicate_claims:
        issues.append(
            _issue("duplicate_claim_coverage", "$.response.acts", "Each claim may be supported exactly once.", claim_ids=duplicate_claims)
        )
    for claim_id, quote in quote_by_claim.items():
        occurrences = _quote_occurrences(all_texts, quote)
        if occurrences != 1:
            issues.append(
                _issue(
                    "support_quote_occurrence_mismatch",
                    "$.response.acts",
                    "Each exact support quote must occur once in the complete dialogue.",
                    claim_id=claim_id,
                    occurrences=occurrences,
                )
            )

    window_size = int(plan["topic_spine"].get("window_turns") or TOPIC_WINDOW_TURNS)
    if all_turn_rows:
        for start in range(max(1, len(all_turn_rows) - window_size + 1)):
            window = all_turn_rows[start : start + window_size]
            if not any(row["has_support"] or _topic_overlap(row["text"], anchors) for row in window):
                issues.append(
                    _issue(
                        "topic_window_drift",
                        window[0]["path"],
                        "A consecutive dialogue window has no claim evidence or topic-spine anchor.",
                        window_turns=len(window),
                    )
                )
    return _report(issues)


def _report(issues: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(item)) for item in issues]
    counts = Counter(_text(item.get("code")) or "unknown" for item in rows)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "pass": not rows,
        "issue_count": len(rows),
        "issue_counts_by_code": dict(sorted(counts.items())),
        "issues": rows,
    }


def is_valid_api_response(
    plan: Mapping[str, Any],
    response: Any,
    *,
    frozen_graph: Mapping[str, Any],
    frozen_claim_bundle: Mapping[str, Any],
    parent_context: Mapping[str, Any],
) -> bool:
    """Convenience wrapper for callers that need only the gate result."""

    return bool(
        validate_api_response(
            plan,
            response,
            frozen_graph=frozen_graph,
            frozen_claim_bundle=frozen_claim_bundle,
            parent_context=parent_context,
        )["pass"]
    )
