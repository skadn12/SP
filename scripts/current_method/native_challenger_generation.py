from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.current_method import api_challenger_generation as adapter
except ModuleNotFoundError:
    from current_method import api_challenger_generation as adapter


base = adapter.base
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SUPPORTED_QUESTION_TYPES = ("ordering", "multi_select", "evidence_selection", "state_reconstruction")
DISABLED_QUESTION_TYPES = {"invalidated_statement_detection"}
DEFAULT_QUESTION_TYPE_WEIGHTS = {
    "ordering": 0.36,
    "multi_select": 0.28,
    "evidence_selection": 0.20,
    "state_reconstruction": 0.16,
}
NATIVE_ATTACK_SKILL_BY_QUESTION_TYPE = {
    "ordering": (
        "timeline_fork_merge",
        "multi_object_dependency",
        "who_knows_what",
    ),
    "multi_select": (
        "belief_vs_reality",
        "multi_object_dependency",
        "cancel_restore",
        "expired_memory",
        "conflicting_reports",
    ),
    "evidence_selection": (
        "who_knows_what",
        "timeline_fork_merge",
        "multi_object_dependency",
    ),
    "state_reconstruction": (
        "belief_vs_reality",
        "cancel_restore",
        "delayed_activation",
        "conditional_plan",
    ),
}
NATIVE_ATTACK_SKILLS = tuple(
    dict.fromkeys(skill for skills in NATIVE_ATTACK_SKILL_BY_QUESTION_TYPE.values() for skill in skills)
)
FALSE_REASON_TYPES = {
    "contradicted",
    "invalidated",
    "canceled",
    "expired",
    "restored",
    "delayed_not_active",
    "scope_limited",
}
GRAPH_TURN_ID_FIELDS = (
    "turn_ids",
    "evidence_turn_ids",
    "activation_turn_ids",
    "invalidation_turn_ids",
    "cancellation_turn_ids",
    "restoration_turn_ids",
    "event_turn_ids",
    "turn_id",
)


@dataclass
class PlannedEvent:
    event_id: str
    entity: str
    slot: str
    value: str
    event_type: str
    status: str
    support_text_hint: str
    planned_turn_id: str | None = None
    visible_turn_id: str | None = None
    source_node_id: str | None = None
    source_turn_ids: list[str] = field(default_factory=list)


@dataclass
class FalseCandidate:
    candidate_id: str
    text: str
    false_reason_type: str
    evidence_event_ids: list[str] = field(default_factory=list)
    support_turn_ids: list[str] = field(default_factory=list)
    support_text_hint: str = ""
    replacement_text: str | None = None


@dataclass
class EventPlan:
    plan_id: str
    source_seed_id: str
    attack_skill: str
    question_type: str
    events: list[PlannedEvent]
    false_candidates: list[FalseCandidate] = field(default_factory=list)
    target_state: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_space(value).lower()).strip()


def token_set(value: Any) -> set[str]:
    return {token for token in normalize_text(value).split() if len(token) >= 3}


def stable_hash(value: Any, *, length: int = 10) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def native_attack_skill_for_question_type(question_type: str, rotation_index: int) -> str:
    skills = NATIVE_ATTACK_SKILL_BY_QUESTION_TYPE.get(question_type) or NATIVE_ATTACK_SKILLS
    return skills[rotation_index % len(skills)]


def normalized_attack_skill(value: Any) -> str:
    skill = normalize_space(value)
    return skill if skill in NATIVE_ATTACK_SKILLS else "unknown"


def sample_attack_skill(sample: dict[str, Any]) -> str:
    attack_plan = sample.get("attack_plan") if isinstance(sample.get("attack_plan"), dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    native_event_plan = sample.get("native_event_plan") if isinstance(sample.get("native_event_plan"), dict) else {}
    generation_metadata = sample.get("generation_metadata") if isinstance(sample.get("generation_metadata"), dict) else {}
    frontier_metadata = sample.get("frontier_metadata") if isinstance(sample.get("frontier_metadata"), dict) else {}
    for value in (
        attack_plan.get("attack_skill"),
        sample.get("attack_skill"),
        metadata.get("attack_skill"),
        native_event_plan.get("attack_skill"),
        generation_metadata.get("attack_skill"),
        frontier_metadata.get("attack_skill"),
    ):
        skill = normalized_attack_skill(value)
        if skill != "unknown":
            return skill
    return "unknown"


def sync_attack_skill_metadata(sample: dict[str, Any], attack_skill: str) -> None:
    skill = normalized_attack_skill(attack_skill)
    sample["attack_skill"] = skill
    for key in ("attack_plan", "metadata", "native_event_plan"):
        container = sample.get(key)
        if not isinstance(container, dict):
            container = {}
            sample[key] = container
        container["attack_skill"] = skill
    for key in ("generation_metadata", "frontier_metadata"):
        container = sample.get(key)
        if isinstance(container, dict):
            container["attack_skill"] = skill


def attack_skill_rotation_index(policy: dict[str, Any], question_type: str, sample_index: int) -> int:
    if policy.get("attack_skill_index") is not None:
        try:
            return int(policy.get("attack_skill_index"))
        except (TypeError, ValueError):
            pass
    raw_indices = policy.get("attack_skill_rotation_indices")
    if isinstance(raw_indices, dict) and raw_indices.get(question_type) is not None:
        try:
            return int(raw_indices.get(question_type))
        except (TypeError, ValueError):
            pass
    return sample_index


def turn_sort_key(turn_id: Any) -> tuple[int, int, str]:
    match = re.match(r"^D(\d+):(\d+)$", str(turn_id or ""))
    if match:
        return int(match.group(1)), int(match.group(2)), str(turn_id)
    return 999999, 999999, str(turn_id or "")


def normalize_turn_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def read_jsonl_if_exists(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return []
    try:
        return read_jsonl(candidate)
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def visible_turns(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    dialogue = row.get("dialogue") if isinstance(row.get("dialogue"), dict) else {}
    turns = dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else []
    normalized_turns: list[dict[str, Any]] = []
    turn_text_by_id: dict[str, str] = {}
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        turn_id = str(turn.get("turn_id") or turn.get("dia_id") or "").strip()
        if not turn_id:
            continue
        normalized = {
            "turn_id": turn_id,
            "speaker": normalize_space(turn.get("speaker") or ("Analyst" if index % 2 == 0 else "Reviewer")),
            "text": normalize_space(turn.get("text")),
        }
        normalized_turns.append(normalized)
        turn_text_by_id[turn_id] = normalized["text"]
    return normalized_turns, turn_text_by_id


def text_supports_candidate(turn_text: str, expected_text: str) -> bool:
    turn_norm = normalize_text(turn_text)
    expected_norm = normalize_text(expected_text)
    if not turn_norm or not expected_norm:
        return False
    if expected_norm in turn_norm or turn_norm in expected_norm:
        return True
    left = token_set(turn_norm)
    right = token_set(expected_norm)
    if not left or not right:
        return False
    overlap = len(left & right) / max(1, min(len(left), len(right)))
    return overlap >= 0.45


def supported_visible_turn_ids(
    turn_ids: Any,
    turn_text_by_id: dict[str, str],
    expected_text: str,
    *,
    allow_explicit_ref_fallback: bool = True,
) -> list[str]:
    visible: list[str] = []
    fallback: list[str] = []
    for turn_id in normalize_turn_ids(turn_ids):
        if turn_id not in turn_text_by_id:
            continue
        if text_supports_candidate(turn_text_by_id[turn_id], expected_text):
            visible.append(turn_id)
        elif allow_explicit_ref_fallback:
            fallback.append(turn_id)
    return sorted(visible or fallback, key=turn_sort_key)


def item_text(item: dict[str, Any]) -> str:
    for key in (
        "value",
        "to",
        "restored_state",
        "cancelled_state",
        "invalidated_statement",
        "final_true_state",
        "planned_state",
        "activation_state",
        "from",
    ):
        text = normalize_space(item.get(key))
        if text:
            return text
    return ""


def item_turn_ids(item: dict[str, Any]) -> list[str]:
    turn_ids: list[str] = []
    for key in GRAPH_TURN_ID_FIELDS:
        for turn_id in normalize_turn_ids(item.get(key)):
            if turn_id not in turn_ids:
                turn_ids.append(turn_id)
    return sorted(turn_ids, key=turn_sort_key)


def _source_entity_slot(row: dict[str, Any]) -> tuple[str, str]:
    graph_update = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    events = graph_update.get("events") if isinstance(graph_update.get("events"), list) else []
    for event in events:
        if isinstance(event, dict):
            entity = normalize_space(event.get("entity"))
            slot = normalize_space(event.get("slot"))
            if entity or slot:
                return entity or "conversation", slot or "memory_state"
    seed = row.get("seed") if isinstance(row.get("seed"), dict) else {}
    source_id = normalize_space(seed.get("locomo_conversation_id") or row.get("source_id") or row.get("sample_id"))
    return source_id or "conversation", "memory_state"


def _dedupe_planned_events(events: list[PlannedEvent]) -> list[PlannedEvent]:
    seen: set[str] = set()
    deduped: list[PlannedEvent] = []
    for event in events:
        key = normalize_text(event.value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def extract_true_event_candidates(row: dict[str, Any]) -> list[PlannedEvent]:
    _turns, turn_text_by_id = visible_turns(row)
    graph_update = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    entity, slot = _source_entity_slot(row)
    candidates: list[PlannedEvent] = []

    def add_candidate(raw: dict[str, Any], source: str, index: int, *, value_override: str | None = None) -> None:
        value = normalize_space(value_override or item_text(raw))
        if len(value) < 8:
            return
        support_turn_ids = supported_visible_turn_ids(item_turn_ids(raw), turn_text_by_id, value)
        if not support_turn_ids:
            return
        status = normalize_space(raw.get("status") or ("active" if source == "state_change" else "true")) or "true"
        event_id = f"E{len(candidates) + 1}"
        candidates.append(
            PlannedEvent(
                event_id=event_id,
                entity=normalize_space(raw.get("entity") or entity) or "conversation",
                slot=normalize_space(raw.get("slot") or slot) or "memory_state",
                value=value,
                event_type=source,
                status=status,
                support_text_hint=f"The tracked memory update is: {value}.",
                source_node_id=normalize_space(raw.get("node_id") or raw.get("event_id") or f"{source}_{index}") or None,
                source_turn_ids=support_turn_ids,
            )
        )

    for index, event in enumerate(graph_update.get("events") or [], start=1):
        if isinstance(event, dict):
            add_candidate(event, "graph_event", index)
    for index, change in enumerate(graph_update.get("state_changes") or [], start=1):
        if isinstance(change, dict):
            add_candidate(change, "state_change", index, value_override=normalize_space(change.get("to")))

    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    for index, option in enumerate(options, start=1):
        if not isinstance(option, dict):
            continue
        value = normalize_space(option.get("text"))
        support_turn_ids = supported_visible_turn_ids(option.get("turn_ids"), turn_text_by_id, value)
        if value and support_turn_ids:
            candidates.append(
                PlannedEvent(
                    event_id=f"E{len(candidates) + 1}",
                    entity=entity,
                    slot=slot,
                    value=value,
                    event_type="qa_option",
                    status="true",
                    support_text_hint=f"The tracked memory update is: {value}.",
                    source_node_id=f"qa_option_{index}",
                    source_turn_ids=support_turn_ids,
                )
            )
    candidates.sort(key=lambda event: (turn_sort_key(event.source_turn_ids[0] if event.source_turn_ids else ""), event.event_id))
    return _dedupe_planned_events(candidates)


def extract_safe_false_candidates(row: dict[str, Any], true_events: list[PlannedEvent]) -> list[FalseCandidate]:
    _turns, turn_text_by_id = visible_turns(row)
    graph_update = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    false_candidates: list[FalseCandidate] = []

    def replacement_for(text: str) -> str:
        for event in reversed(true_events):
            if normalize_text(event.value) != normalize_text(text):
                return event.value
        return true_events[-1].value if true_events else "the newer tracked state"

    def add_false(text: str, reason_type: str, turn_ids: Any, replacement_text: str | None = None) -> None:
        normalized_text = normalize_space(text)
        if len(normalized_text) < 8 or reason_type not in FALSE_REASON_TYPES:
            return
        support_turn_ids = sorted([turn_id for turn_id in normalize_turn_ids(turn_ids) if turn_id in turn_text_by_id], key=turn_sort_key)
        if not support_turn_ids and true_events:
            support_turn_ids = list(true_events[-1].source_turn_ids[:1])
        key = normalize_text(normalized_text)
        if not key or any(normalize_text(candidate.text) == key for candidate in false_candidates):
            return
        replacement = normalize_space(replacement_text or replacement_for(normalized_text))
        false_candidates.append(
            FalseCandidate(
                candidate_id=f"F{len(false_candidates) + 1}",
                text=normalized_text,
                false_reason_type=reason_type,
                evidence_event_ids=[event.event_id for event in true_events[-2:]],
                support_turn_ids=support_turn_ids,
                support_text_hint=(
                    f"Correction for the tracked memory: {normalized_text} is no longer current; "
                    f"{replacement} is the active state."
                ),
                replacement_text=replacement,
            )
        )

    for change in graph_update.get("state_changes") or []:
        if not isinstance(change, dict):
            continue
        add_false(
            normalize_space(change.get("from")),
            "contradicted",
            change.get("evidence_turn_ids"),
            replacement_text=normalize_space(change.get("to")),
        )

    for item in graph_update.get("invalidations") or []:
        if isinstance(item, dict):
            add_false(item.get("invalidated_statement"), "invalidated", item_turn_ids(item), item.get("final_true_state"))
    for item in graph_update.get("cancellations") or []:
        if isinstance(item, dict):
            add_false(item.get("cancelled_state") or item.get("planned_state") or item.get("value"), "canceled", item_turn_ids(item))
    for item in graph_update.get("restorations") or []:
        if isinstance(item, dict):
            add_false(item.get("value") or item.get("cancelled_state"), "restored", item_turn_ids(item), item.get("restored_state"))
    for item in graph_update.get("delayed_activations") or []:
        if isinstance(item, dict):
            add_false(item.get("planned_state") or item.get("value"), "delayed_not_active", item_turn_ids(item), item.get("activation_state"))

    # State-change chains are the main native-v0 source of safe negatives. If a row has
    # enough visible true events but few explicit invalidation objects, stale earlier
    # states are still graph-derived negatives because later state_changes supersede them.
    if len(false_candidates) < 2 and len(true_events) >= 3:
        for event in true_events[:-1]:
            add_false(event.value, "contradicted", event.source_turn_ids, replacement_text=true_events[-1].value)
            if len(false_candidates) >= 2:
                break
    return false_candidates


def _select_distinct_events(events: list[PlannedEvent], count: int, rng: random.Random) -> list[PlannedEvent]:
    by_turn: dict[tuple[int, int, str], PlannedEvent] = {}
    for event in events:
        first_turn = event.source_turn_ids[0] if event.source_turn_ids else ""
        by_turn.setdefault(turn_sort_key(first_turn), event)
    ordered = [by_turn[key] for key in sorted(by_turn)]
    if len(ordered) <= count:
        return ordered
    window = ordered[: max(count, min(len(ordered), count + 4))]
    selected = rng.sample(window, count)
    return sorted(selected, key=lambda event: turn_sort_key(event.source_turn_ids[0] if event.source_turn_ids else ""))


def build_event_plan_from_seed(row: dict[str, Any], policy: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    question_type = str(policy.get("question_type") or policy.get("requested_question_type") or "ordering")
    source_seed_id = str(row.get("sample_id") or row.get("source_id") or stable_hash(row))
    sample_index = int(policy.get("sample_index") or 0)
    attack_skill_index = attack_skill_rotation_index(policy, question_type, sample_index)
    attack_skill = native_attack_skill_for_question_type(question_type, attack_skill_index)
    true_candidates = extract_true_event_candidates(row)
    false_candidates = extract_safe_false_candidates(row, true_candidates)
    selected_events: list[PlannedEvent] = []
    selected_false: list[FalseCandidate] = []
    target_state: str | None = None
    not_ready_reasons: list[str] = []

    if question_type in DISABLED_QUESTION_TYPES or question_type not in SUPPORTED_QUESTION_TYPES:
        not_ready_reasons.append(f"unsupported_question_type:{question_type}")
    elif question_type == "ordering":
        selected_events = _select_distinct_events(true_candidates, 4, rng)
        if len(selected_events) < 4:
            not_ready_reasons.append("ordering_insufficient_true_events")
    elif question_type == "multi_select":
        selected_events = _select_distinct_events(true_candidates, 2, rng)
        selected_false = false_candidates[:2]
        if len(selected_events) < 2:
            not_ready_reasons.append("multi_select_insufficient_true_candidates")
        if len(selected_false) < 2:
            not_ready_reasons.append("multi_select_insufficient_safe_negative_candidates")
    elif question_type == "evidence_selection":
        selected_events = _select_distinct_events(true_candidates, 1, rng)
        if not selected_events:
            not_ready_reasons.append("evidence_selection_no_supported_event")
        else:
            target_state = selected_events[0].value
    elif question_type == "state_reconstruction":
        selected_events = _select_distinct_events(true_candidates, min(3, max(1, len(true_candidates))), rng)
        if not selected_events:
            not_ready_reasons.append("state_reconstruction_no_supported_state")
        else:
            target_state = selected_events[-1].value

    plan = EventPlan(
        plan_id=f"native_v0_{sample_index:04d}_{stable_hash([source_seed_id, question_type, sample_index])}",
        source_seed_id=source_seed_id,
        attack_skill=attack_skill,
        question_type=question_type,
        events=copy.deepcopy(selected_events),
        false_candidates=copy.deepcopy(selected_false),
        target_state=target_state,
        metadata={
            "ready": not not_ready_reasons,
            "not_ready_reasons": not_ready_reasons,
            "source_true_candidate_count": len(true_candidates),
            "source_false_candidate_count": len(false_candidates),
            "sample_index": sample_index,
            "native_generator_version": "v0",
        },
    )
    return asdict(plan)


def _dialogue_turn(turn_id: str, speaker: str, text: str) -> dict[str, Any]:
    return {"turn_id": turn_id, "speaker": speaker, "text": normalize_space(text)}


def render_dialogue_from_event_plan(event_plan: dict[str, Any], seed_row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    plan_id = str(event_plan.get("plan_id") or stable_hash(event_plan))
    session_id = 1000 + int((event_plan.get("metadata") or {}).get("sample_index") or 0)
    speakers = ["Analyst", "Reviewer"]
    source_turns, _source_text_by_id = visible_turns(seed_row)
    for turn in source_turns[:2]:
        speaker = normalize_space(turn.get("speaker"))
        if speaker and speaker not in speakers:
            speakers.append(speaker)

    turns: list[dict[str, Any]] = [
        _dialogue_turn(f"D{session_id}:1", speakers[0], "Let's review the tracked memory carefully before updating the graph."),
        _dialogue_turn(f"D{session_id}:2", speakers[1], "Agreed. I will state only the updates that are supported by the conversation."),
    ]
    turn_id_by_event_id: dict[str, str] = {}
    turn_id_by_false_candidate_id: dict[str, str] = {}
    next_pos = 3

    for index, event in enumerate(event_plan.get("events") or [], start=1):
        if not isinstance(event, dict):
            continue
        if next_pos in {11, 12}:
            next_pos += 1
        turn_id = f"D{session_id}:{next_pos}"
        event["planned_turn_id"] = turn_id
        event["visible_turn_id"] = turn_id
        event["source_turn_ids"] = [turn_id]
        speaker = speakers[index % len(speakers)]
        turns.append(_dialogue_turn(turn_id, speaker, event.get("support_text_hint") or f"The tracked memory update is: {event.get('value')}."))
        turn_id_by_event_id[str(event.get("event_id"))] = turn_id
        next_pos += 2

    replacement_text = normalize_space((event_plan.get("target_state") or "") or ((event_plan.get("events") or [{}])[-1].get("value") if event_plan.get("events") else "the active state"))
    for index, candidate in enumerate(event_plan.get("false_candidates") or [], start=1):
        if not isinstance(candidate, dict):
            continue
        turn_id = f"D{session_id}:{next_pos}"
        candidate["support_turn_ids"] = [turn_id]
        if not candidate.get("replacement_text"):
            candidate["replacement_text"] = replacement_text
        hint = candidate.get("support_text_hint") or (
            f"Correction for the tracked memory: {candidate.get('text')} is no longer current; "
            f"{candidate.get('replacement_text')} is the active state."
        )
        turns.append(_dialogue_turn(turn_id, speakers[(index + 1) % len(speakers)], hint))
        turn_id_by_false_candidate_id[str(candidate.get("candidate_id"))] = turn_id
        next_pos += 2

    while len(turns) < 12:
        turn_id = f"D{session_id}:{next_pos}"
        turns.append(
            _dialogue_turn(
                turn_id,
                speakers[len(turns) % len(speakers)],
                "I am adding no new graph fact here; this turn is only context for the review.",
            )
        )
        next_pos += 1
    turns.sort(key=lambda turn: turn_sort_key(turn.get("turn_id")))
    return {
        "turns": turns,
        "turn_id_by_event_id": turn_id_by_event_id,
        "turn_id_by_false_candidate_id": turn_id_by_false_candidate_id,
        "plan_id": plan_id,
    }


def build_graph_update_from_event_plan(event_plan: dict[str, Any], dialogue: dict[str, Any]) -> dict[str, Any]:
    visible_turn_ids = {str(turn.get("turn_id")) for turn in dialogue.get("turns", []) if isinstance(turn, dict)}
    events: list[dict[str, Any]] = []
    for index, event in enumerate(event_plan.get("events") or [], start=1):
        if not isinstance(event, dict):
            continue
        turn_id = str(event.get("visible_turn_id") or "")
        if turn_id not in visible_turn_ids:
            continue
        events.append(
            {
                "event_id": str(event.get("event_id") or f"E{index}"),
                "node_id": event.get("source_node_id") or f"{event_plan.get('plan_id')}:event_{index}",
                "role": event.get("event_type") or "planned_event",
                "entity": event.get("entity") or "conversation",
                "slot": event.get("slot") or "memory_state",
                "value": event.get("value"),
                "turn_ids": [turn_id],
                "status": event.get("status") or "active",
            }
        )

    state_changes: list[dict[str, Any]] = []
    for previous, current in zip(events[:-1], events[1:]):
        state_changes.append(
            {
                "entity": current.get("entity") or "conversation",
                "slot": current.get("slot") or "memory_state",
                "from": previous.get("value"),
                "to": current.get("value"),
                "evidence_turn_ids": list(current.get("turn_ids") or []),
            }
        )

    invalidations: list[dict[str, Any]] = []
    cancellations: list[dict[str, Any]] = []
    restorations: list[dict[str, Any]] = []
    delayed_activations: list[dict[str, Any]] = []
    final_true_state = normalize_space(event_plan.get("target_state") or (events[-1].get("value") if events else ""))
    for candidate in event_plan.get("false_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        support_turn_ids = [turn_id for turn_id in normalize_turn_ids(candidate.get("support_turn_ids")) if turn_id in visible_turn_ids]
        if not support_turn_ids:
            continue
        reason_type = str(candidate.get("false_reason_type") or "contradicted")
        invalidation = {
            "invalidated_statement": candidate.get("text"),
            "invalidating_event": candidate.get("replacement_text") or final_true_state,
            "invalidation_turn_ids": support_turn_ids,
            "final_true_state": candidate.get("replacement_text") or final_true_state,
            "false_reason_type": reason_type,
        }
        invalidations.append(invalidation)
        if reason_type == "canceled":
            cancellations.append({"cancelled_state": candidate.get("text"), "cancellation_turn_ids": support_turn_ids})
        elif reason_type == "restored":
            restorations.append({"restored_state": final_true_state, "restoration_turn_ids": support_turn_ids})
        elif reason_type == "delayed_not_active":
            delayed_activations.append(
                {
                    "planned_state": candidate.get("text"),
                    "activation_state": final_true_state,
                    "activation_turn_ids": support_turn_ids,
                }
            )

    return {
        "events": events,
        "state_changes": state_changes,
        "invalidations": invalidations,
        "cancellations": cancellations,
        "restorations": restorations,
        "delayed_activations": delayed_activations,
        "visibility_edges": [
            {"observer": "solver", "event_turn_id": turn_id, "visibility": "directly_stated_in_dialogue"}
            for event in events
            for turn_id in event.get("turn_ids", [])
        ],
        "entities": [{"entity_id": events[0].get("entity") if events else "conversation", "type": "conversation_memory"}],
    }


def _rotate_if_identity(options: list[dict[str, Any]], chronological_event_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    label_by_event_id = {str(option.get("event_id")): str(option.get("id")) for option in options}
    gold = [label_by_event_id[event_id] for event_id in chronological_event_ids]
    if gold == [option["id"] for option in options]:
        options = options[1:] + options[:1]
        for index, option in enumerate(options):
            option["id"] = LETTERS[index]
        label_by_event_id = {str(option.get("event_id")): str(option.get("id")) for option in options}
        gold = [label_by_event_id[event_id] for event_id in chronological_event_ids]
    return options, gold


def build_qa_from_event_plan(event_plan: dict[str, Any], dialogue: dict[str, Any], question_type: str, rng: random.Random) -> dict[str, Any]:
    events = [event for event in event_plan.get("events") or [] if isinstance(event, dict) and event.get("visible_turn_id")]
    false_candidates = [candidate for candidate in event_plan.get("false_candidates") or [] if isinstance(candidate, dict)]
    if question_type == "ordering":
        chronological = sorted(events[:4], key=lambda event: turn_sort_key(event.get("visible_turn_id")))
        display = copy.deepcopy(chronological)
        rng.shuffle(display)
        options = []
        for index, event in enumerate(display):
            options.append(
                {
                    "id": LETTERS[index],
                    "text": normalize_space(event.get("value")),
                    "turn_ids": [str(event.get("visible_turn_id"))],
                    "event_id": str(event.get("event_id")),
                }
            )
        options, gold = _rotate_if_identity(options, [str(event.get("event_id")) for event in chronological])
        for option in options:
            option.pop("event_id", None)
        return {
            "question_type": "ordering",
            "question": "Order the option letters by the first dialogue turn that supports each memory update, from earliest to latest. Answer as letters separated by '>'.",
            "options": options,
            "gold": gold,
            "gold_source": "programmatic_earliest_visible_turn",
            "evidence_turn_ids": sorted({turn_id for option in options for turn_id in option.get("turn_ids", [])}, key=turn_sort_key),
        }

    if question_type == "multi_select":
        selected_true = events[:2]
        selected_false = false_candidates[:2]
        raw_options: list[dict[str, Any]] = []
        for event in selected_true:
            raw_options.append(
                {
                    "kind": "true",
                    "text": normalize_space(event.get("value")),
                    "turn_ids": [str(event.get("visible_turn_id"))],
                    "native_candidate_id": str(event.get("event_id")),
                }
            )
        for candidate in selected_false:
            raw_options.append(
                {
                    "kind": "false",
                    "text": normalize_space(candidate.get("text")),
                    "turn_ids": normalize_turn_ids(candidate.get("support_turn_ids"))[:1],
                    "native_candidate_id": str(candidate.get("candidate_id")),
                }
            )
        rng.shuffle(raw_options)
        options: list[dict[str, Any]] = []
        gold: list[str] = []
        false_reasons_by_option: dict[str, dict[str, Any]] = {}
        for index, option in enumerate(raw_options):
            label = LETTERS[index]
            if option["kind"] == "true":
                gold.append(label)
            else:
                candidate = next((item for item in selected_false if str(item.get("candidate_id")) == option["native_candidate_id"]), {})
                false_reasons_by_option[label] = {
                    "candidate_id": option["native_candidate_id"],
                    "false_reason_type": candidate.get("false_reason_type"),
                    "replacement_text": candidate.get("replacement_text"),
                }
            options.append({"id": label, "text": option["text"], "turn_ids": option["turn_ids"]})
        return {
            "question_type": "multi_select",
            "question": "Select exactly the two option letters that remain active supported memory states. Answer with option letters only.",
            "options": options,
            "gold": sorted(gold),
            "gold_source": "programmatic_candidate_bank_labels",
            "evidence_turn_ids": sorted({turn_id for option in options if option["id"] in gold for turn_id in option.get("turn_ids", [])}, key=turn_sort_key),
            "native_false_reasons_by_option": false_reasons_by_option,
        }

    if question_type == "evidence_selection":
        target = events[0]
        target_turn_id = str(target.get("visible_turn_id"))
        all_turns = dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else []
        distractor_turns = [
            turn
            for turn in all_turns
            if isinstance(turn, dict)
            and str(turn.get("turn_id")) != target_turn_id
            and not text_supports_candidate(turn.get("text") or "", target.get("value") or "")
        ]
        selected_distractors = distractor_turns[:4]
        option_turns = [{"turn_id": target_turn_id, "text": next((turn.get("text") for turn in all_turns if turn.get("turn_id") == target_turn_id), "")}]
        option_turns.extend({"turn_id": str(turn.get("turn_id")), "text": str(turn.get("text") or "")} for turn in selected_distractors)
        rng.shuffle(option_turns)
        options = [
            {
                "id": item["turn_id"],
                "text": f"{item['turn_id']}: {normalize_space(item['text'])[:180]}",
            }
            for item in option_turns[:5]
        ]
        return {
            "question_type": "evidence_selection",
            "question": "Which listed turn ID directly supports the target memory update? Answer with turn IDs only.",
            "options": options,
            "gold": [target_turn_id],
            "gold_source": "programmatic_visible_turn",
            "evidence_turn_ids": [target_turn_id],
            "target_state": normalize_space(target.get("value")),
        }

    target = events[-1]
    return {
        "question_type": "state_reconstruction",
        "question": "What is the final remembered state for the tracked topic? Answer with only the exact state text.",
        "options": [],
        "gold": [normalize_space(target.get("value"))],
        "gold_source": "programmatic_final_state",
        "evidence_turn_ids": [str(target.get("visible_turn_id"))],
    }


def graph_turn_refs(graph_update: dict[str, Any]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for group in ("events", "state_changes", "invalidations", "cancellations", "restorations", "delayed_activations"):
        items = graph_update.get(group)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for field_name in GRAPH_TURN_ID_FIELDS:
                for turn_id in normalize_turn_ids(item.get(field_name)):
                    refs.append((f"graph_update.{group}[{index}].{field_name}", turn_id))
    return refs


def validate_native_sample_contract(sample: dict[str, Any]) -> dict[str, Any]:
    dialogue = sample.get("dialogue") if isinstance(sample.get("dialogue"), dict) else {}
    turns = dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else []
    visible_turn_ids = {str(turn.get("turn_id")) for turn in turns if isinstance(turn, dict)}
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    graph_update = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    qtype = str(qa.get("question_type") or "unknown")
    missing_refs: list[dict[str, str]] = []

    def check_refs(field: str, value: Any) -> None:
        for turn_id in normalize_turn_ids(value):
            if turn_id not in visible_turn_ids:
                missing_refs.append({"field": field, "turn_id": turn_id})

    check_refs("qa.evidence_turn_ids", qa.get("evidence_turn_ids"))
    for index, option in enumerate(qa.get("options") or []):
        if isinstance(option, dict):
            check_refs(f"qa.options[{index}].turn_ids", option.get("turn_ids"))
    for field, turn_id in graph_turn_refs(graph_update):
        if turn_id not in visible_turn_ids:
            missing_refs.append({"field": field, "turn_id": turn_id})

    warnings: list[str] = []
    attack_skill = sample_attack_skill(sample)
    attack_skill_valid = attack_skill != "unknown"
    if not attack_skill_valid:
        warnings.append("attack_skill_missing_or_unknown")
    ordering_gold_derivable = True
    ordering_identity_gold = False
    multi_select_gold_derivable = True
    positive_negative_mixed = True
    if qtype == "ordering":
        option_order: list[tuple[str, str]] = []
        for option in qa.get("options") or []:
            if isinstance(option, dict):
                turn_ids = sorted(normalize_turn_ids(option.get("turn_ids")), key=turn_sort_key)
                if turn_ids:
                    option_order.append((str(option.get("id")), turn_ids[0]))
        recomputed_gold = [label for label, _turn_id in sorted(option_order, key=lambda item: (turn_sort_key(item[1]), item[0]))]
        original_gold = [str(item) for item in (qa.get("gold") or [])]
        ordering_gold_derivable = recomputed_gold == original_gold and len(recomputed_gold) >= 4
        ordering_identity_gold = original_gold == ["A", "B", "C", "D"]
        if not ordering_gold_derivable:
            warnings.append("ordering_gold_not_derivable_from_visible_turns")
        if ordering_identity_gold:
            warnings.append("ordering_identity_gold_shortcut_risk")
    elif qtype == "multi_select":
        false_reasons = qa.get("native_false_reasons_by_option") if isinstance(qa.get("native_false_reasons_by_option"), dict) else {}
        option_ids = {str(option.get("id")) for option in qa.get("options") or [] if isinstance(option, dict)}
        gold_ids = {str(item) for item in (qa.get("gold") or [])}
        false_ids = set(false_reasons)
        multi_select_gold_derivable = bool(gold_ids) and gold_ids.issubset(option_ids) and not gold_ids.intersection(false_ids)
        positive_negative_mixed = bool(gold_ids) and bool(false_ids) and len(option_ids) >= 4
        if not multi_select_gold_derivable:
            warnings.append("multi_select_gold_not_derivable_from_candidate_labels")
        if not positive_negative_mixed:
            warnings.append("multi_select_positive_negative_not_mixed")

    cleaned_once, cleanup_report_once = adapter.cleanup_generated_sample_turn_refs(sample)
    cleaned_twice, cleanup_report_twice = adapter.cleanup_generated_sample_turn_refs(cleaned_once)
    cleanup_idempotent = (
        cleanup_report_once.get("valid_after_cleanup") == cleanup_report_twice.get("valid_after_cleanup")
        and cleanup_report_once.get("missing_ref_count") == cleanup_report_twice.get("missing_ref_count")
        and cleanup_report_once.get("unsupported_ref_count") == cleanup_report_twice.get("unsupported_ref_count")
    )
    if not cleanup_idempotent:
        warnings.append("cleanup_not_idempotent")
    base_validation = base.validate_sample(sample)
    base_valid = base.base_valid({**sample, "validation": base_validation})
    if not base_valid and base_validation.get("anti_shortcut_pass") is False:
        warnings.append("anti_shortcut_risk")

    valid = (
        not missing_refs
        and ordering_gold_derivable
        and multi_select_gold_derivable
        and positive_negative_mixed
        and cleanup_idempotent
        and cleanup_report_once.get("valid_after_cleanup") is True
    )
    return {
        "valid": valid,
        "question_type": qtype,
        "attack_skill": attack_skill,
        "attack_skill_valid": attack_skill_valid,
        "missing_refs": missing_refs,
        "missing_ref_count": len(missing_refs),
        "ordering_gold_derivable": ordering_gold_derivable,
        "ordering_identity_gold": ordering_identity_gold,
        "multi_select_gold_derivable": multi_select_gold_derivable,
        "positive_negative_mixed": positive_negative_mixed,
        "cleanup_idempotent": cleanup_idempotent,
        "cleanup_valid": cleanup_report_once.get("valid_after_cleanup") is True,
        "base_valid": base_valid,
        "base_validation": base_validation,
        "warnings": warnings,
    }


def assemble_native_sample(
    *,
    sample_index: int,
    seed_row: dict[str, Any],
    event_plan: dict[str, Any],
    dialogue: dict[str, Any],
    graph_update: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, Any]:
    sample_id = f"native_current_method_r4_v0_{sample_index:04d}_{stable_hash([event_plan.get('plan_id'), qa])}"
    question_type = str(event_plan.get("question_type") or qa.get("question_type") or "ordering")
    attack_skill = normalized_attack_skill(event_plan.get("attack_skill"))
    if attack_skill == "unknown":
        attack_skill = native_attack_skill_for_question_type(question_type, sample_index)
    event_plan = copy.deepcopy(event_plan)
    event_plan["attack_skill"] = attack_skill
    sample = {
        "sample_id": sample_id,
        "source_id": event_plan.get("source_seed_id"),
        "generation_source": "current_method.native_challenger_generation.v0",
        "generation_backend": "native_current_method",
        "native_generator_version": "v0",
        "challenger_api_used": False,
        "attack_skill": attack_skill,
        "frontier_from_split": seed_row.get("frontier_from_split"),
        "frontier_from_sample_name": seed_row.get("frontier_from_sample_name"),
        "seed_source": copy.deepcopy(seed_row.get("seed_source")) if isinstance(seed_row.get("seed_source"), dict) else None,
        "seed": copy.deepcopy(seed_row.get("seed")) if isinstance(seed_row.get("seed"), dict) else None,
        "attack_plan": {
            "attack_skill": attack_skill,
            "reasoning_mechanism": "native_structured_event_plan",
            "target_solver_weakness": "long_dialogue_graph_state_reasoning",
            "requires_ordering": event_plan.get("question_type") == "ordering",
            "requires_invalidated_memory": event_plan.get("question_type") == "multi_select",
            "shortcut_to_avoid": ["gold_label_leakage", "evidence_window_position_leakage"],
        },
        "native_event_plan": copy.deepcopy(event_plan),
        "dialogue": {"turns": dialogue.get("turns") or []},
        "graph_update": graph_update,
        "qa": qa,
        "metadata": {
            "method": "current_method_native_v0",
            "source_seed_id": event_plan.get("source_seed_id"),
            "plan_id": event_plan.get("plan_id"),
            "attack_skill": attack_skill,
        },
    }
    sync_attack_skill_metadata(sample, attack_skill)
    sample["validation"] = base.validate_sample(sample)
    sample["native_contract"] = validate_native_sample_contract(sample)
    cleaned_sample, cleanup_report = adapter.cleanup_generated_sample_turn_refs(sample)
    sync_attack_skill_metadata(cleaned_sample, attack_skill)
    cleaned_sample["native_contract"] = validate_native_sample_contract(cleaned_sample)
    cleaned_sample["native_cleanup_report"] = cleanup_report
    return cleaned_sample


def _weighted_question_type(rng: random.Random, weights: dict[str, float]) -> str:
    items = [(qtype, max(0.0, float(weights.get(qtype, 0.0)))) for qtype in SUPPORTED_QUESTION_TYPES]
    total = sum(weight for _qtype, weight in items)
    if total <= 0:
        items = [(qtype, DEFAULT_QUESTION_TYPE_WEIGHTS[qtype]) for qtype in SUPPORTED_QUESTION_TYPES]
        total = sum(weight for _qtype, weight in items)
    threshold = rng.random() * total
    cumulative = 0.0
    for qtype, weight in items:
        cumulative += weight
        if cumulative >= threshold:
            return qtype
    return items[-1][0]


def _question_type_weights(policy: dict[str, Any] | None) -> dict[str, float]:
    policy = policy or {}
    raw = policy.get("question_type_weights") if isinstance(policy.get("question_type_weights"), dict) else policy
    weights = dict(DEFAULT_QUESTION_TYPE_WEIGHTS)
    for qtype in SUPPORTED_QUESTION_TYPES:
        try:
            value = float(raw.get(qtype)) if isinstance(raw, dict) and raw.get(qtype) is not None else weights[qtype]
        except (TypeError, ValueError):
            value = weights[qtype]
        weights[qtype] = max(0.0, value)
    for qtype in DISABLED_QUESTION_TYPES:
        weights.pop(qtype, None)
    return weights


def _build_ready_plan_for_row(
    row: dict[str, Any],
    preferred_qtype: str,
    policy: dict[str, Any],
    rng: random.Random,
    sample_index: int,
    not_ready_counts: Counter[str],
) -> dict[str, Any] | None:
    qtypes = [preferred_qtype] + [qtype for qtype in SUPPORTED_QUESTION_TYPES if qtype != preferred_qtype]
    for qtype in qtypes:
        plan_policy = {
            **policy,
            "question_type": qtype,
            "sample_index": sample_index,
            "attack_skill_index": attack_skill_rotation_index(policy, qtype, sample_index),
        }
        plan = build_event_plan_from_seed(row, plan_policy, rng)
        metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
        if metadata.get("ready") is True:
            return plan
        for reason in metadata.get("not_ready_reasons") or ["unknown_not_ready"]:
            not_ready_counts[str(reason)] += 1
    return None


def _build_ready_plan_for_question_type(
    seed_rows: list[dict[str, Any]],
    question_type: str,
    policy: dict[str, Any],
    rng: random.Random,
    sample_index: int,
    not_ready_counts: Counter[str],
    *,
    max_row_attempts: int = 32,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not seed_rows:
        return None, None
    for _attempt in range(max_row_attempts):
        row = seed_rows[rng.randrange(len(seed_rows))]
        plan_policy = {
            **policy,
            "question_type": question_type,
            "sample_index": sample_index,
            "attack_skill_index": attack_skill_rotation_index(policy, question_type, sample_index),
        }
        plan = build_event_plan_from_seed(row, plan_policy, rng)
        metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
        if metadata.get("ready") is True:
            return plan, row
        for reason in metadata.get("not_ready_reasons") or ["unknown_not_ready"]:
            not_ready_counts[str(reason)] += 1
    return None, None


def _summarize_native_generation(samples: list[dict[str, Any]], not_ready_counts: Counter[str]) -> dict[str, Any]:
    by_question_type: dict[str, dict[str, int]] = {}
    by_attack_skill: Counter[str] = Counter()
    by_question_type_and_attack_skill: dict[str, dict[str, int]] = {}
    examples: list[dict[str, Any]] = []
    contract_valid = 0
    base_valid = 0
    turn_ref_missing_count = 0
    ordering_identity_gold_count = 0
    anti_shortcut_risk_count = 0
    unknown_attack_skill_count = 0
    multi_select_insufficient_negative_count = int(not_ready_counts.get("multi_select_insufficient_safe_negative_candidates", 0))
    for sample in samples:
        qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
        qtype = str(qa.get("question_type") or "unknown")
        skill = sample_attack_skill(sample)
        contract = sample.get("native_contract") if isinstance(sample.get("native_contract"), dict) else {}
        validation = sample.get("validation") if isinstance(sample.get("validation"), dict) else {}
        stats = by_question_type.setdefault(qtype, {"n": 0, "contract_valid": 0, "base_valid": 0, "anti_shortcut_risk": 0})
        stats["n"] += 1
        by_attack_skill[skill] += 1
        qtype_skill_stats = by_question_type_and_attack_skill.setdefault(qtype, {})
        qtype_skill_stats[skill] = qtype_skill_stats.get(skill, 0) + 1
        if skill == "unknown":
            unknown_attack_skill_count += 1
        if contract.get("valid") is True:
            contract_valid += 1
            stats["contract_valid"] += 1
        if base.base_valid(sample):
            base_valid += 1
            stats["base_valid"] += 1
        if validation.get("anti_shortcut_pass") is False or "anti_shortcut_risk" in (contract.get("warnings") or []):
            anti_shortcut_risk_count += 1
            stats["anti_shortcut_risk"] += 1
        turn_ref_missing_count += int(contract.get("missing_ref_count") or 0)
        if contract.get("ordering_identity_gold") is True:
            ordering_identity_gold_count += 1
        if (contract.get("valid") is not True or not base.base_valid(sample)) and len(examples) < 10:
            examples.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "question_type": qtype,
                    "attack_skill": skill,
                    "contract_valid": contract.get("valid"),
                    "base_valid": base.base_valid(sample),
                    "contract_warnings": contract.get("warnings"),
                    "validation": validation,
                    "qa": qa,
                }
            )
    largest_attack_skill_share = round(max(by_attack_skill.values()) / len(samples), 6) if samples else 0.0
    return {
        "generated": len(samples),
        "contract_valid": contract_valid,
        "base_valid": base_valid,
        "by_question_type": by_question_type,
        "by_attack_skill": dict(by_attack_skill),
        "by_question_type_and_attack_skill": by_question_type_and_attack_skill,
        "unknown_attack_skill_count": unknown_attack_skill_count,
        "largest_attack_skill_share": largest_attack_skill_share,
        "not_ready_reasons": dict(not_ready_counts),
        "turn_ref_missing_count": turn_ref_missing_count,
        "ordering_identity_gold_count": ordering_identity_gold_count,
        "multi_select_insufficient_negative_count": multi_select_insufficient_negative_count,
        "anti_shortcut_risk_count": anti_shortcut_risk_count,
        "examples": examples,
    }


def generate_native_samples(
    seed_rows: list[dict[str, Any]],
    policy: dict[str, Any] | None,
    n_samples: int,
    output_dir: str | Path,
    rng_seed: int = 123,
    native_challenger_mode: str = "template",
    api_args: argparse.Namespace | None = None,
    round_id: str = "current_method_r4",
    partial_samples: str | Path | None = None,
    no_resume: bool = False,
) -> list[dict[str, Any]]:
    if native_challenger_mode == "graph_conditioned_api_dialogue":
        try:
            from scripts.current_method import native_api_graph_dialogue_expansion as graph_dialogue
        except ModuleNotFoundError:
            from current_method import native_api_graph_dialogue_expansion as graph_dialogue
        if api_args is None:
            raise ValueError("api_args is required for native_challenger_mode=graph_conditioned_api_dialogue")
        return graph_dialogue.generate_graph_conditioned_api_dialogue_samples(
            seed_rows=seed_rows,
            policy=policy,
            n_samples=n_samples,
            output_dir=output_dir,
            rng_seed=rng_seed,
            api_args=api_args,
            round_id=round_id,
            partial_samples=partial_samples,
            no_resume=no_resume,
        )
    if native_challenger_mode != "template":
        raise ValueError(f"Unsupported native_challenger_mode: {native_challenger_mode}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed)
    weights = _question_type_weights(policy)
    policy_payload = dict(policy or {})
    attack_skill_rotation_indices: Counter[str] = Counter()
    policy_payload["attack_skill_rotation_indices"] = attack_skill_rotation_indices
    samples: list[dict[str, Any]] = []
    not_ready_counts: Counter[str] = Counter()
    attempts = 0
    max_attempts = max(500, n_samples * 80)

    partial_path = Path(partial_samples) if partial_samples else output / "samples.partial.jsonl"

    def finalize(final_samples: list[dict[str, Any]], finalize_attempts: int) -> list[dict[str, Any]]:
        contract_summary = _summarize_native_generation(final_samples, not_ready_counts)
        contract_summary["attempts"] = finalize_attempts
        contract_summary["requested"] = n_samples
        contract_summary["generator"] = "current_method.native_challenger_generation.v0"
        contract_summary["generation_backend"] = "native_current_method"
        contract_summary["native_generator_version"] = "v0"
        contract_summary["generated_count"] = len(final_samples)
        contract_summary["valid_count"] = sum(1 for sample in final_samples if base.base_valid(sample))
        contract_summary["clean_count"] = sum(1 for sample in final_samples if base.is_clean(sample))
        contract_summary["challenger_api_called_count"] = 0
        contract_summary["challenger_api_used"] = False
        contract_summary["old_gate_generate_samples_called"] = False
        contract_summary["api_failure_count"] = 0
        contract_summary["challenger_generation_sources"] = {
            "current_method.native_challenger_generation.v0": len(final_samples)
        }
        valid_samples = [sample for sample in final_samples if base.base_valid(sample)]
        clean_samples = [sample for sample in final_samples if base.is_clean(sample)]
        write_jsonl(output / "native_samples.jsonl", final_samples)
        write_jsonl(output / "native_valid_samples.jsonl", valid_samples)
        write_jsonl(output / "samples.jsonl", final_samples)
        write_jsonl(output / "valid_samples.jsonl", valid_samples)
        write_jsonl(output / "clean_samples.jsonl", clean_samples)
        write_json(output / "native_contract_summary.json", contract_summary)
        write_json(output / "native_generation_summary.json", contract_summary)
        write_json(output / "generation_summary.json", contract_summary)
        return final_samples

    # Resume: prefer the crash-safe partial file, then fall back to a completed samples.jsonl.
    existing_samples: list[dict[str, Any]] = []
    if not no_resume:
        existing_samples = read_jsonl_if_exists(partial_path)
        if not existing_samples:
            existing_samples = read_jsonl_if_exists(output / "samples.jsonl")
    if existing_samples and len(existing_samples) >= n_samples:
        # Already have enough; trim, rewrite the canonical outputs, and return without generating.
        trimmed = existing_samples[:n_samples]
        write_jsonl(partial_path, trimmed)
        return finalize(trimmed, attempts)

    if not seed_rows:
        raise ValueError("No seed rows supplied for native current_method generation.")

    # Seed the in-memory list and the partial file with whatever we recovered, so appends are
    # consistent regardless of whether existing samples came from the partial or the final file.
    samples = list(existing_samples)
    write_jsonl(partial_path, samples)
    partial_handle = partial_path.open("a", encoding="utf-8")

    try:
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            preferred_qtype = _weighted_question_type(rng, weights)
            sample_index = len(samples)
            plan, row = _build_ready_plan_for_question_type(
                seed_rows,
                preferred_qtype,
                policy_payload,
                rng,
                sample_index,
                not_ready_counts,
            )
            if plan is None:
                fallback_row = seed_rows[rng.randrange(len(seed_rows))]
                plan = _build_ready_plan_for_row(fallback_row, preferred_qtype, policy_payload, rng, sample_index, not_ready_counts)
                row = fallback_row
            if plan is None or row is None:
                continue
            dialogue = render_dialogue_from_event_plan(plan, row, rng)
            graph_update = build_graph_update_from_event_plan(plan, dialogue)
            qa = build_qa_from_event_plan(plan, dialogue, str(plan.get("question_type")), rng)
            sample = assemble_native_sample(
                sample_index=sample_index,
                seed_row=row,
                event_plan=plan,
                dialogue=dialogue,
                graph_update=graph_update,
                qa=qa,
            )
            samples.append(sample)
            final_qtype = str(plan.get("question_type") or qa.get("question_type") or "unknown")
            attack_skill_rotation_indices[final_qtype] += 1
            # Crash-safe append: persist each accepted sample immediately so an interrupted
            # run can resume from the partial file without losing progress.
            partial_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            partial_handle.flush()
    finally:
        partial_handle.close()

    return finalize(samples, attempts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native current_method challenger generation v0 smoke runner.")
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--native-challenger-mode", choices=("template", "graph_conditioned_api_dialogue"), default="template")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_rows = read_jsonl(args.seed_file)
    samples = generate_native_samples(
        seed_rows,
        {"question_type_weights": DEFAULT_QUESTION_TYPE_WEIGHTS},
        int(args.n_samples),
        args.output_dir,
        rng_seed=int(args.seed),
        native_challenger_mode=args.native_challenger_mode,
    )
    summary_path = Path(args.output_dir) / "native_generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(samples) < int(args.n_samples):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
