from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from current_method import report_builder as current_reports  # noqa: E402


DEFAULT_STAGE2_SOURCE = (
    ROOT
    / "base_restart_current_method_validation"
    / "05_stage2_repair_topup_from_stage1"
    / "train_mix"
    / "frontier_train.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "base_restart_current_method_validation"
    / "64_qa_operation_answer_type_anchor_diversification_probe"
)
DEFAULT_STAGE2_SOLVER = (
    ROOT
    / "base_restart_current_method_validation"
    / "accepted_models"
    / "solver_base_restart_stage2_merged"
)
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGET_OPERATION_COUNTS = {
    "temporal_state_update": 30,
    "bridge": 18,
    "comparison": 18,
    "visibility": 18,
    "count": 12,
    "cause_consequence": 12,
    "relation_lookup": 12,
}
QA_OPERATIONS = tuple(TARGET_OPERATION_COUNTS)
ANSWER_TYPES = (
    "Person",
    "GroupOrOrg",
    "LocationOrScope",
    "DateOrTime",
    "Number",
    "YesNo",
    "Event",
    "ObjectOrCommonNoun",
    "AttributeValue",
    "ProperNounOrNamedEntity",
)

MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
NUMBER_WORD_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|once|twice|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|single|double|triple)\b",
    re.IGNORECASE,
)
PERSON_RELATION_RE = re.compile(
    r"\b(aunt|uncle|mother|father|mom|dad|sister|brother|cousin|friend|girlfriend|boyfriend|"
    r"colleague|coach|mentor|teacher|neighbor|partner|wife|husband|daughter|son|grandmother|grandfather)\b",
    re.IGNORECASE,
)
GROUP_ORG_RE = re.compile(
    r"\b(group|team|club|organization|organisation|company|department|festival|shelter|church|agency|"
    r"center|centre|university|studio|school|class|committee|community|foundation|association)\b",
    re.IGNORECASE,
)
LOCATION_RE = re.compile(
    r"\b(forest|beach|trail|falls|town|city|park|shelter|church|studio|department|home|kitchen|"
    r"school|university|office|center|centre|camp|room|farm|garden|lake|river|mountain)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(red|blue|green|purple|pink|orange|yellow|black|white|brown|gray|grey|gold|silver|"
    r"vanilla|chocolate|coconut|berry|strawberry|sushi|romance|drama|sci-fi|action)\b",
    re.IGNORECASE,
)
PROPER_NOUN_HINT_RE = re.compile(r"([A-Z][A-Za-z0-9'&:-]+(?:\s+[A-Z0-9][A-Za-z0-9'&:-]+)+)")
BAD_SURFACE_RE = re.compile(
    r"(raw_qa|archive-shelf|discarded|final active state|latest confirmed detail|confirmed active state|"
    r"scope limited|scope_limited|invalidated|outdated|wrong scope|no longer current|previous value|"
    r"event_instance|graph_anchor|anchor_|locomo_raw|not mentioned)",
    re.IGNORECASE,
)
STATE_SHELL_RE = re.compile(
    r"(which option best matches the state|which option reflects the state|select exactly the option letters that are still true|"
    r"put the option letters in the order)",
    re.IGNORECASE,
)
VISIBLE_INTERNAL_RE = re.compile(
    r"\b(anchored?|anchors?|graph-backed|graph backed|graph updates?|graph update sequence)\b",
    re.IGNORECASE,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return rows
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_hash(value: Any, length: int = 10) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:length]


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def safe_slug(value: Any, fallback: str = "x") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return slug[:32] or fallback


def short_answer(value: Any, limit: int = 88) -> str:
    text = normalize_space(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(" ,;:") + "."


def surface_bad(value: Any) -> bool:
    text = normalize_space(value)
    if not text:
        return True
    return bool(BAD_SURFACE_RE.search(text))


def parse_qa_value(value: Any) -> tuple[str, str] | None:
    text = normalize_space(value)
    match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)$", text, flags=re.IGNORECASE)
    if not match:
        return None
    question = normalize_space(match.group(1)).rstrip()
    answer = normalize_space(match.group(2)).rstrip()
    answer = re.sub(r"\s+(?:Mark this fact as .*)$", "", answer, flags=re.IGNORECASE).strip()
    if question.endswith("?") is False:
        question += "?"
    if surface_bad(question) or surface_bad(answer):
        return None
    return question, answer


def parse_permission_person(value: Any) -> str | None:
    text = normalize_space(value)
    match = re.search(r"\bonly\s+([A-Z][a-z]+)\s+can\s+see\b", text)
    if not match:
        return None
    person = normalize_space(match.group(1))
    if surface_bad(person):
        return None
    return person


def is_date_or_time(question: str, answer: str) -> bool:
    ql = question.lower()
    al = answer.lower()
    return (
        ql.startswith("when ")
        or bool(MONTH_RE.search(answer))
        or bool(re.search(r"\b\d{1,2}\s*[/-]\s*\d{1,2}\b", answer))
        or any(token in al for token in ("yesterday", "tomorrow", "last week", "next week", "week before", "weekend", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))
    )


def is_number_answer(question: str, answer: str) -> bool:
    ql = question.lower()
    al = answer.lower()
    return (
        any(phrase in ql for phrase in ("how many", "how often", "how long", "what age", "what number"))
        or bool(re.search(r"\b\d+\b", answer))
        or bool(NUMBER_WORD_RE.search(answer))
        or any(token in al for token in ("times", "years", "months", "weeks", "days"))
    )


def is_person_answer(question: str, answer: str, person_names: set[str]) -> bool:
    ql = question.lower()
    answer_norm = normalize_text(answer)
    if ql.startswith("who ") or ql.startswith("whose "):
        return True
    if normalize_space(answer) in person_names:
        return True
    if PERSON_RELATION_RE.search(answer):
        return True
    if len(answer.split()) <= 3 and any(normalize_text(name) == answer_norm for name in person_names):
        return True
    return False


def classify_answer_type(question: str, answer: str, person_names: set[str]) -> str:
    question = normalize_space(question)
    answer = normalize_space(answer)
    ql = question.lower()
    if is_date_or_time(question, answer):
        return "DateOrTime"
    if is_number_answer(question, answer):
        return "Number"
    if is_person_answer(question, answer, person_names):
        return "Person"
    if ql.startswith("where "):
        return "LocationOrScope"
    if GROUP_ORG_RE.search(answer) and not ql.startswith("where "):
        return "GroupOrOrg"
    if LOCATION_RE.search(answer) and (ql.startswith("where ") or len(answer.split()) <= 6):
        return "LocationOrScope"
    if ql.startswith(("what color", "how does", "how did", "how is", "what kind", "what type")) or COLOR_RE.search(answer):
        return "AttributeValue"
    if PROPER_NOUN_HINT_RE.search(answer) and len(answer.split()) <= 8:
        return "ProperNounOrNamedEntity"
    if ql.startswith(("what event", "which event")):
        return "Event"
    return "ObjectOrCommonNoun"


def person_names_from_row(row: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    graph = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    for item in graph.get("entities") or []:
        if isinstance(item, dict):
            name = normalize_space(item.get("entity_id"))
            if name:
                names.add(name)
    for turn in (row.get("dialogue") or {}).get("turns") or []:
        if isinstance(turn, dict):
            speaker = normalize_space(turn.get("speaker"))
            if speaker and speaker not in {"Analyst", "Reviewer"}:
                names.add(speaker)
    for event in graph.get("events") or []:
        if isinstance(event, dict):
            entity = normalize_space(event.get("entity"))
            if entity:
                names.add(entity)
    return names


def event_phrase(question: str, answer: str) -> str:
    question = normalize_space(question).rstrip("?")
    answer = normalize_space(answer).rstrip(".")
    q = question
    m = re.match(r"When did\s+(.+)$", q, flags=re.IGNORECASE)
    if m:
        return short_answer(m.group(1), 82)
    m = re.match(r"Where was\s+(.+)$", q, flags=re.IGNORECASE)
    if m:
        return short_answer(f"{m.group(1)} at {answer}", 82)
    m = re.match(r"What did\s+(.+?)\s+(?:do|host|receive|share|donate|make|get|try|organize|start|join|open|take)\b(.*)$", q, flags=re.IGNORECASE)
    if m:
        return short_answer(f"{m.group(1)}: {answer}", 82)
    m = re.match(r"What(?: was| is)?\s+(.+)$", q, flags=re.IGNORECASE)
    if m and len(answer.split()) <= 9:
        return short_answer(f"{m.group(1)}: {answer}", 82)
    return short_answer(f"{question}: {answer}", 92)


def bridge_clue(anchor: "Anchor") -> str:
    q = anchor.question.rstrip("?")
    a = anchor.answer
    if len(a.split()) <= 8:
        return short_answer(f"{q} -> {a}", 72)
    return short_answer(q, 72)


@dataclass(frozen=True)
class Anchor:
    anchor_ref: str
    parent_id: str
    source_sample_id: str
    entity: str
    answer_type: str
    answer: str
    normalized_answer: str
    question: str = ""
    event_text: str = ""
    event_instance_id: str | None = None
    status: str = ""
    role: str = ""
    slot: str = ""
    time_index: int = 0
    anchor_kind: str = "event"
    truth_value: bool | None = None
    proposition: str | None = None


@dataclass
class ParentContext:
    row: dict[str, Any]
    parent_id: str
    source_sample_id: str
    person_names: set[str]
    anchors: list[Anchor]
    event_anchors: list[Anchor]
    entity_anchors: list[Anchor]
    permission_anchors: list[Anchor]
    by_type: dict[str, list[Anchor]]
    by_ref: dict[str, Anchor]


def anchor_ref_for_event(parent_id: str, event: dict[str, Any], index: int) -> str:
    entity = safe_slug(event.get("entity") or "event")
    raw = event.get("event_instance_id") or event.get("event_id") or event.get("node_id") or f"{parent_id}:{index}"
    return f"anchor_{entity}_event_{index:02d}_{stable_hash(raw, 8)}"


def build_parent_context(row: dict[str, Any], row_index: int) -> ParentContext:
    source_sample_id = normalize_space(row.get("sample_id") or f"row_{row_index:04d}")
    parent_id = normalize_space(
        row.get("parent_id")
        or row.get("source_id")
        or (row.get("seed") or {}).get("sample_id")
        or (row.get("seed") or {}).get("parent_sample_id")
        or source_sample_id
    )
    person_names = person_names_from_row(row)
    graph = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    anchors: list[Anchor] = []
    event_anchors: list[Anchor] = []
    permission_anchors: list[Anchor] = []
    for index, event in enumerate(graph.get("events") or []):
        if not isinstance(event, dict):
            continue
        ref = anchor_ref_for_event(parent_id, event, index)
        raw_value = event.get("value")
        qa = parse_qa_value(raw_value)
        permission_person = parse_permission_person(raw_value)
        entity = normalize_space(event.get("entity")) or "Unknown"
        base_kwargs = {
            "anchor_ref": ref,
            "parent_id": parent_id,
            "source_sample_id": source_sample_id,
            "entity": entity,
            "event_instance_id": normalize_space(event.get("event_instance_id") or event.get("event_id")),
            "status": normalize_space(event.get("status")),
            "role": normalize_space(event.get("role")),
            "slot": normalize_space(event.get("slot")),
            "time_index": int(event.get("time_index") or index + 1),
            "anchor_kind": "event",
        }
        if qa:
            question, answer = qa
            answer_type = classify_answer_type(question, answer, person_names)
            anchor = Anchor(
                **base_kwargs,
                answer_type=answer_type,
                answer=answer,
                normalized_answer=normalize_text(answer),
                question=question,
                event_text=event_phrase(question, answer),
            )
            anchors.append(anchor)
            event_anchors.append(anchor)
        if permission_person:
            proposition = "The limited note was visible to everyone."
            anchor = Anchor(
                **base_kwargs,
                answer_type="Person",
                answer=permission_person,
                normalized_answer=normalize_text(permission_person),
                question="Who was named as the person with access to the limited note?",
                event_text="access to the limited note",
                truth_value=True,
                proposition=proposition,
            )
            anchors.append(anchor)
            permission_anchors.append(anchor)
    entity_anchors: list[Anchor] = []
    for name in sorted(person_names):
        ref = f"anchor_person_{safe_slug(name)}_{stable_hash(parent_id + name, 8)}"
        anchor = Anchor(
            anchor_ref=ref,
            parent_id=parent_id,
            source_sample_id=source_sample_id,
            entity=name,
            answer_type="Person",
            answer=name,
            normalized_answer=normalize_text(name),
            question=f"Which person is named as {name}?",
            event_text=f"{name} is a person in the dialogue",
            anchor_kind="entity",
        )
        entity_anchors.append(anchor)
        anchors.append(anchor)
    by_type: dict[str, list[Anchor]] = defaultdict(list)
    by_ref: dict[str, Anchor] = {}
    for anchor in anchors:
        by_type[anchor.answer_type].append(anchor)
        by_ref[anchor.anchor_ref] = anchor
    return ParentContext(
        row=row,
        parent_id=parent_id,
        source_sample_id=source_sample_id,
        person_names=person_names,
        anchors=anchors,
        event_anchors=event_anchors,
        entity_anchors=entity_anchors,
        permission_anchors=permission_anchors,
        by_type=dict(by_type),
        by_ref=by_ref,
    )


def unique_by_answer(anchors: Iterable[Anchor], *, exclude_norms: set[str] | None = None) -> list[Anchor]:
    seen = set(exclude_norms or set())
    output: list[Anchor] = []
    for anchor in anchors:
        if not anchor.answer or anchor.normalized_answer in seen:
            continue
        if surface_bad(anchor.answer):
            continue
        seen.add(anchor.normalized_answer)
        output.append(anchor)
    return output


def event_option_anchor(anchor: Anchor) -> Anchor:
    text = anchor.event_text or event_phrase(anchor.question, anchor.answer)
    return Anchor(
        anchor_ref=f"{anchor.anchor_ref}_event",
        parent_id=anchor.parent_id,
        source_sample_id=anchor.source_sample_id,
        entity=anchor.entity,
        answer_type="Event",
        answer=text,
        normalized_answer=normalize_text(text),
        question=anchor.question,
        event_text=text,
        event_instance_id=anchor.event_instance_id,
        status=anchor.status,
        role=anchor.role,
        slot=anchor.slot,
        time_index=anchor.time_index,
        anchor_kind="event_surface",
    )


def event_pool(ctx: ParentContext) -> list[Anchor]:
    return unique_by_answer(event_option_anchor(anchor) for anchor in ctx.event_anchors if not surface_bad(anchor.event_text))


def same_type_distractors(ctx: ParentContext, answer_type: str, gold_anchor: Anchor, count: int = 3) -> list[Anchor]:
    if answer_type == "Event":
        pool = event_pool(ctx)
    elif answer_type == "Person":
        pool = list(ctx.by_type.get("Person") or [])
    else:
        pool = list(ctx.by_type.get(answer_type) or [])
    pool = sorted(pool, key=lambda item: (item.time_index, item.normalized_answer, item.anchor_ref))
    return unique_by_answer(
        [anchor for anchor in pool if anchor.normalized_answer != gold_anchor.normalized_answer and anchor.anchor_ref != gold_anchor.anchor_ref],
        exclude_norms={gold_anchor.normalized_answer},
    )[:count]


def numeric_distractors(ctx: ParentContext, gold_anchor: Anchor, count: int = 3) -> list[Anchor]:
    base = same_type_distractors(ctx, "Number", gold_anchor, count=count)
    if len(base) >= count:
        return base[:count]
    text = gold_anchor.answer
    match = re.search(r"\b(\d+)\b", text)
    number_word_values = {
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
        "once": 1,
        "twice": 2,
    }
    value: int | None = int(match.group(1)) if match else None
    if value is None:
        for word, numeric in number_word_values.items():
            if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
                value = numeric
                break
    if value is None:
        candidates = ["one", "two", "three", "four", "five"]
    else:
        candidates = [str(max(0, value - 1)), str(value + 1), str(value + 2), str(max(0, value - 2))]
    seen = {gold_anchor.normalized_answer}
    seen.update(anchor.normalized_answer for anchor in base)
    for idx, candidate_text in enumerate(candidates):
        norm = normalize_text(candidate_text)
        if not norm or norm in seen:
            continue
        base.append(
            Anchor(
                anchor_ref=f"{gold_anchor.anchor_ref}_numeric_distractor_{idx}",
                parent_id=gold_anchor.parent_id,
                source_sample_id=gold_anchor.source_sample_id,
                entity=gold_anchor.entity,
                answer_type="Number",
                answer=candidate_text,
                normalized_answer=norm,
                question=gold_anchor.question,
                event_text=gold_anchor.event_text,
                event_instance_id=gold_anchor.event_instance_id,
                time_index=gold_anchor.time_index,
                anchor_kind="derived_numeric_neighbor",
            )
        )
        seen.add(norm)
        if len(base) >= count:
            break
    return base[:count]


def option_payload(
    *,
    label: str,
    text: str,
    answer_type: str,
    anchor: Anchor,
    is_gold: bool,
    truth_value: bool,
    distractor_type: str | None = None,
) -> dict[str, Any]:
    payload = {
        "id": label,
        "option_id": label,
        "text": text,
        "is_gold": bool(is_gold),
        "answer_type": answer_type,
        "anchor_ref": anchor.anchor_ref,
        "truth_value": bool(truth_value),
        "source_event_instance_id": anchor.event_instance_id,
        "anchor_kind": anchor.anchor_kind,
    }
    if not is_gold:
        payload["distractor_source"] = "same_parent_graph_anchor"
        payload["distractor_type"] = distractor_type or f"plausible_{answer_type}"
    return payload


def duplicate_or_near_duplicate(texts: list[str]) -> tuple[bool, bool]:
    norms = [normalize_text(text) for text in texts]
    duplicate = len(norms) != len(set(norms))
    near = False
    for idx, left in enumerate(norms):
        for right in norms[idx + 1 :]:
            if not left or not right:
                continue
            if left in right or right in left:
                near = True
            if SequenceMatcher(None, left, right).ratio() >= 0.92:
                near = True
    return duplicate, near


def candidate_bad_reasons(candidate: dict[str, Any], ctx: ParentContext) -> list[str]:
    reasons: list[str] = []
    question = normalize_space(candidate.get("question_text"))
    if not question or surface_bad(question):
        reasons.append("bad_question")
    if STATE_SHELL_RE.search(question):
        reasons.append("state_shell_leak")
    if VISIBLE_INTERNAL_RE.search(question):
        reasons.append("state_shell_leak")
    query_plan = candidate.get("query_plan") if isinstance(candidate.get("query_plan"), dict) else {}
    all_refs = []
    for key in ("required_anchor_refs", "supporting_anchor_refs", "gold_anchor_refs", "distractor_anchor_refs"):
        values = query_plan.get(key)
        if isinstance(values, list):
            all_refs.extend(str(item) for item in values)
    known_refs = set(ctx.by_ref)
    known_refs.update(anchor.anchor_ref for anchor in event_pool(ctx))
    options = candidate.get("options") if isinstance(candidate.get("options"), list) else []
    known_refs.update(str(option.get("anchor_ref")) for option in options if isinstance(option, dict) and option.get("anchor_ref"))
    if not all_refs:
        reasons.append("missing_anchor")
    missing_refs = [
        ref
        for ref in all_refs
        if ref not in known_refs and not (ref.endswith("_yes") or ref.endswith("_no"))
    ]
    if missing_refs:
        reasons.append("unresolved_anchor")
    texts = [normalize_space(option.get("text")) for option in options if isinstance(option, dict)]
    answer_type = candidate.get("answer_type")
    expected_option_count = 2 if answer_type == "YesNo" else 4
    if len(options) != expected_option_count:
        reasons.append("bad_question")
    if any(surface_bad(text) for text in texts):
        reasons.append("placeholder_option")
    if any(VISIBLE_INTERNAL_RE.search(text) for text in texts):
        reasons.append("option_surface_risk")
    duplicate, near = duplicate_or_near_duplicate(texts)
    if duplicate:
        reasons.append("duplicate_option")
    if near:
        reasons.append("near_duplicate_option")
    if answer_type not in ANSWER_TYPES:
        reasons.append("answer_type_mismatch")
    gold_count = sum(1 for option in options if isinstance(option, dict) and option.get("is_gold") is True)
    if gold_count != 1:
        reasons.append("bad_question")
    for option in options:
        if not isinstance(option, dict):
            reasons.append("bad_question")
            continue
        if option.get("answer_type") != answer_type:
            reasons.append("distractor_answer_type_mismatch" if option.get("is_gold") is not True else "answer_type_mismatch")
    return sorted(set(reasons))


def make_mcq_candidate(
    *,
    ctx: ParentContext,
    qa_operation: str,
    qa_subtype: str,
    answer_type: str,
    question_text: str,
    gold_anchor: Anchor,
    distractors: list[Anchor],
    support_anchors: list[Anchor],
    gold_answer: str | None = None,
    proposition: str | None = None,
    truth_value: bool | None = None,
    seed_suffix: str = "",
    min_distractors: int = 3,
) -> dict[str, Any] | None:
    if answer_type not in ANSWER_TYPES:
        return None
    gold_answer = normalize_space(gold_answer if gold_answer is not None else gold_anchor.answer)
    if surface_bad(gold_answer) or surface_bad(question_text):
        return None
    selected = unique_by_answer(distractors, exclude_norms={normalize_text(gold_answer)})[:3]
    if answer_type != "YesNo" and len(selected) < 3:
        return None
    option_items: list[tuple[Anchor, str, bool]] = [(gold_anchor, gold_answer, True)]
    option_items.extend((anchor, anchor.answer, False) for anchor in selected)
    rng = random.Random(stable_hash([ctx.source_sample_id, qa_operation, qa_subtype, gold_anchor.anchor_ref, seed_suffix], 16))
    rng.shuffle(option_items)
    options: list[dict[str, Any]] = []
    gold_label = ""
    for idx, (anchor, text, is_gold) in enumerate(option_items):
        label = LETTERS[idx]
        if is_gold:
            gold_label = label
        options.append(
            option_payload(
                label=label,
                text=short_answer(text, 120),
                answer_type=answer_type,
                anchor=anchor,
                is_gold=is_gold,
                truth_value=is_gold,
                distractor_type=f"same_type_{answer_type}_graph_anchor",
            )
        )
    support_refs = [anchor.anchor_ref for anchor in support_anchors]
    gold_refs = [gold_anchor.anchor_ref]
    distractor_refs = [anchor.anchor_ref for anchor in selected]
    query_plan = {
        "answer_source": "graph_anchor",
        "qa_operation": qa_operation,
        "qa_subtype": qa_subtype,
        "answer_type": answer_type,
        "reasoning_anchor_path": support_refs + gold_refs,
        "required_anchor_refs": sorted(set(support_refs + gold_refs)),
        "supporting_anchor_refs": support_refs,
        "gold_anchor_refs": gold_refs,
        "distractor_anchor_refs": distractor_refs,
        "gold_answer": gold_answer,
        "normalized_gold_answer": normalize_text(gold_answer),
    }
    if proposition is not None:
        query_plan["proposition"] = proposition
    if truth_value is not None:
        query_plan["truth_value"] = bool(truth_value)
    candidate = {
        "parent_id": ctx.parent_id,
        "source_sample_id": ctx.source_sample_id,
        "qa_operation": qa_operation,
        "qa_subtype": qa_subtype,
        "answer_type": answer_type,
        "answer_format": "single_choice_mcq",
        "question_text": normalize_space(question_text),
        "query_plan": query_plan,
        "anchor_refs": {
            "support": support_refs,
            "gold": gold_refs,
            "distractors": distractor_refs,
        },
        "options": options,
        "gold": [gold_label],
        "gold_answer_text": gold_answer,
        "gold_anchor_ref": gold_anchor.anchor_ref,
        "support_anchor_refs": support_refs,
        "distractor_anchor_refs": distractor_refs,
        "truth_value": bool(truth_value) if truth_value is not None else True,
    }
    reasons = candidate_bad_reasons(candidate, ctx)
    if reasons:
        candidate["reject_reasons"] = reasons
        return None
    return candidate


def make_yesno_candidate(
    *,
    ctx: ParentContext,
    qa_operation: str,
    qa_subtype: str,
    question_text: str,
    support_anchor: Anchor,
    proposition: str,
    truth_value: bool,
    seed_suffix: str = "",
) -> dict[str, Any] | None:
    yes_anchor = Anchor(
        anchor_ref=f"{support_anchor.anchor_ref}_yes",
        parent_id=ctx.parent_id,
        source_sample_id=ctx.source_sample_id,
        entity=support_anchor.entity,
        answer_type="YesNo",
        answer="Yes",
        normalized_answer="yes",
        question=question_text,
        event_text=support_anchor.event_text,
        event_instance_id=support_anchor.event_instance_id,
        time_index=support_anchor.time_index,
        anchor_kind="yesno_option",
        truth_value=truth_value is True,
        proposition=proposition,
    )
    no_anchor = Anchor(
        anchor_ref=f"{support_anchor.anchor_ref}_no",
        parent_id=ctx.parent_id,
        source_sample_id=ctx.source_sample_id,
        entity=support_anchor.entity,
        answer_type="YesNo",
        answer="No",
        normalized_answer="no",
        question=question_text,
        event_text=support_anchor.event_text,
        event_instance_id=support_anchor.event_instance_id,
        time_index=support_anchor.time_index,
        anchor_kind="yesno_option",
        truth_value=truth_value is False,
        proposition=proposition,
    )
    gold_anchor = yes_anchor if truth_value else no_anchor
    distractor = no_anchor if truth_value else yes_anchor
    rng = random.Random(stable_hash([ctx.source_sample_id, qa_operation, qa_subtype, support_anchor.anchor_ref, seed_suffix], 16))
    option_items = [(gold_anchor, gold_anchor.answer, True), (distractor, distractor.answer, False)]
    rng.shuffle(option_items)
    options: list[dict[str, Any]] = []
    gold_label = ""
    for idx, (anchor, text, is_gold) in enumerate(option_items):
        label = LETTERS[idx]
        if is_gold:
            gold_label = label
        options.append(
            option_payload(
                label=label,
                text=text,
                answer_type="YesNo",
                anchor=anchor,
                is_gold=is_gold,
                truth_value=is_gold,
                distractor_type="opposite_truth_value",
            )
        )
    query_plan = {
        "answer_source": "graph_anchor",
        "qa_operation": qa_operation,
        "qa_subtype": qa_subtype,
        "answer_type": "YesNo",
        "reasoning_anchor_path": [support_anchor.anchor_ref],
        "required_anchor_refs": [support_anchor.anchor_ref],
        "supporting_anchor_refs": [support_anchor.anchor_ref],
        "gold_anchor_refs": [support_anchor.anchor_ref],
        "distractor_anchor_refs": [support_anchor.anchor_ref],
        "gold_answer": "Yes" if truth_value else "No",
        "normalized_gold_answer": "yes" if truth_value else "no",
        "proposition": proposition,
        "truth_value": bool(truth_value),
    }
    candidate = {
        "parent_id": ctx.parent_id,
        "source_sample_id": ctx.source_sample_id,
        "qa_operation": qa_operation,
        "qa_subtype": qa_subtype,
        "answer_type": "YesNo",
        "answer_format": "single_choice_mcq",
        "question_text": normalize_space(question_text),
        "query_plan": query_plan,
        "anchor_refs": {
            "support": [support_anchor.anchor_ref],
            "gold": [support_anchor.anchor_ref],
            "distractors": [support_anchor.anchor_ref],
        },
        "options": options,
        "gold": [gold_label],
        "gold_answer_text": "Yes" if truth_value else "No",
        "gold_anchor_ref": support_anchor.anchor_ref,
        "support_anchor_refs": [support_anchor.anchor_ref],
        "distractor_anchor_refs": [support_anchor.anchor_ref],
        "proposition": proposition,
        "truth_value": bool(truth_value),
    }
    reasons = candidate_bad_reasons(candidate, ctx)
    if reasons:
        candidate["reject_reasons"] = reasons
        return None
    return candidate


def build_temporal_candidates(ctx: ParentContext) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    active = [anchor for anchor in ctx.event_anchors if anchor.status == "active" and anchor.question and anchor.answer_type != "Person"]
    for anchor in active:
        distractors = numeric_distractors(ctx, anchor) if anchor.answer_type == "Number" else same_type_distractors(ctx, anchor.answer_type, anchor)
        candidate = make_mcq_candidate(
            ctx=ctx,
            qa_operation="temporal_state_update",
            qa_subtype="final_state",
            answer_type=anchor.answer_type,
            question_text=f"After the later updates, {anchor.question[0].lower() + anchor.question[1:]}",
            gold_anchor=anchor,
            distractors=distractors,
            support_anchors=[anchor],
            seed_suffix="final",
        )
        if candidate:
            output.append(candidate)
    non_active = [anchor for anchor in ctx.event_anchors if anchor.status in {"outdated", "false", "canceled", "expired", "scope_limited"}]
    for anchor in active:
        pool = [item for item in non_active if item.answer_type == anchor.answer_type]
        if len(pool) < 3:
            continue
        candidate = make_mcq_candidate(
            ctx=ctx,
            qa_operation="temporal_state_update",
            qa_subtype="still_true",
            answer_type=anchor.answer_type,
            question_text=f"Which answer is still supported for this memory question: {anchor.question}",
            gold_anchor=anchor,
            distractors=pool[:3],
            support_anchors=[anchor],
            seed_suffix="still",
        )
        if candidate:
            output.append(candidate)
    events = event_pool(ctx)
    if len(events) >= 4:
        ordered = sorted(events, key=lambda item: (item.time_index, item.anchor_ref))
        for idx in range(0, max(1, len(ordered) - 3), 2):
            group = ordered[idx : idx + 4]
            if len(group) < 4:
                continue
            gold = min(group, key=lambda item: item.time_index)
            candidate = make_mcq_candidate(
                ctx=ctx,
                qa_operation="temporal_state_update",
                qa_subtype="ordering",
                answer_type="Event",
                question_text="Which listed update happened earliest?",
                gold_anchor=gold,
                distractors=[item for item in group if item.anchor_ref != gold.anchor_ref],
                support_anchors=group,
                seed_suffix=f"ordering_{idx}",
            )
            if candidate:
                output.append(candidate)
        pairs = sorted(events, key=lambda item: (item.time_index, item.anchor_ref))
        for earlier, later in zip(pairs, pairs[1:]):
            distractors = [item for item in events if item.anchor_ref not in {earlier.anchor_ref, later.anchor_ref}]
            candidate = make_mcq_candidate(
                ctx=ctx,
                qa_operation="temporal_state_update",
                qa_subtype="before_after",
                answer_type="Event",
                question_text=f"Which update was recorded after \"{short_answer(earlier.answer, 72)}\"?",
                gold_anchor=later,
                distractors=distractors[:3],
                support_anchors=[earlier, later],
                seed_suffix=f"after_{earlier.anchor_ref}",
            )
            if candidate:
                output.append(candidate)
    return output


def build_bridge_candidates(ctx: ParentContext) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_entity: dict[str, list[Anchor]] = defaultdict(list)
    for anchor in ctx.event_anchors:
        if anchor.question and anchor.answer:
            by_entity[anchor.entity].append(anchor)
    person_pool = list(ctx.entity_anchors)
    for entity, anchors in sorted(by_entity.items()):
        if len(anchors) < 2:
            continue
        gold = next((item for item in person_pool if item.answer == entity), None)
        if not gold:
            continue
        distractors = [item for item in person_pool if item.answer != entity]
        anchors = sorted(anchors, key=lambda item: (item.time_index, item.anchor_ref))
        for first, second in zip(anchors, anchors[1:]):
            gold_event = event_option_anchor(second)
            event_distractors = [
                item
                for item in event_pool(ctx)
                if item.anchor_ref not in {event_option_anchor(first).anchor_ref, gold_event.anchor_ref}
            ]
            event_candidate = make_mcq_candidate(
                ctx=ctx,
                qa_operation="bridge",
                qa_subtype="entity_to_event",
                answer_type="Event",
                question_text=f"Which update is connected to {entity} through the detail \"{bridge_clue(first)}\"?",
                gold_anchor=gold_event,
                distractors=event_distractors[:3],
                support_anchors=[first, second, gold],
                seed_suffix=f"event_bridge_{first.anchor_ref}_{second.anchor_ref}",
            )
            if event_candidate:
                output.append(event_candidate)
            question = (
                "Which person connects both details: "
                f"\"{bridge_clue(first)}\" and \"{bridge_clue(second)}\"?"
            )
            candidate = make_mcq_candidate(
                ctx=ctx,
                qa_operation="bridge",
                qa_subtype="entity_bridge",
                answer_type="Person",
                question_text=question,
                gold_anchor=gold,
                distractors=distractors[:3],
                support_anchors=[first, second, gold],
                seed_suffix=f"{first.anchor_ref}_{second.anchor_ref}",
            )
            if candidate:
                output.append(candidate)
        for first in anchors:
            if first.answer_type not in {"AttributeValue", "ObjectOrCommonNoun", "ProperNounOrNamedEntity", "LocationOrScope", "DateOrTime"}:
                continue
            for second in anchors:
                if first.anchor_ref == second.anchor_ref or second.answer_type == first.answer_type:
                    continue
                question = (
                    f"Which person is linked to both {short_answer(first.answer, 48)} and "
                    f"{short_answer(second.answer, 48)}?"
                )
                candidate = make_mcq_candidate(
                    ctx=ctx,
                    qa_operation="bridge",
                    qa_subtype="attribute_to_entity",
                    answer_type="Person",
                    question_text=question,
                    gold_anchor=gold,
                    distractors=distractors[:3],
                    support_anchors=[first, second, gold],
                    seed_suffix=f"attr_{first.anchor_ref}_{second.anchor_ref}",
                )
                if candidate:
                    output.append(candidate)
        for first in anchors:
            question = f"Which person is associated with this detail: {bridge_clue(first)}?"
            candidate = make_mcq_candidate(
                ctx=ctx,
                qa_operation="bridge",
                qa_subtype="answer_to_entity",
                answer_type="Person",
                question_text=question,
                gold_anchor=gold,
                distractors=distractors[:3],
                support_anchors=[first, gold],
                seed_suffix=f"answer_entity_{first.anchor_ref}",
            )
            if candidate:
                output.append(candidate)
    return output


def build_comparison_candidates(ctx: ParentContext) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    events = event_pool(ctx)
    if len(events) >= 4:
        ordered = sorted(events, key=lambda item: (item.time_index, item.anchor_ref))
        for idx in range(max(1, len(ordered) - 3)):
            group = ordered[idx : idx + 4]
            if len(group) < 4:
                continue
            gold = min(group, key=lambda item: item.time_index)
            candidate = make_mcq_candidate(
                ctx=ctx,
                qa_operation="comparison",
                qa_subtype="earlier_event",
                answer_type="Event",
                question_text="Which listed event happened earlier than the others?",
                gold_anchor=gold,
                distractors=[item for item in group if item.anchor_ref != gold.anchor_ref],
                support_anchors=group,
                seed_suffix=f"cmp_event_{idx}",
            )
            if candidate:
                output.append(candidate)
    numbers = [anchor for anchor in ctx.by_type.get("Number", []) if not surface_bad(anchor.answer)]
    if len(numbers) >= 2:
        for first, second in zip(numbers, numbers[1:]):
            if first.normalized_answer == second.normalized_answer:
                continue
            truth = first.time_index < second.time_index
            proposition = f"The detail \"{short_answer(first.answer, 40)}\" was recorded before \"{short_answer(second.answer, 40)}\"."
            candidate = make_yesno_candidate(
                ctx=ctx,
                qa_operation="comparison",
                qa_subtype="before_after_yesno",
                question_text=f"Was \"{short_answer(first.answer, 48)}\" recorded before \"{short_answer(second.answer, 48)}\"?",
                support_anchor=first,
                proposition=proposition,
                truth_value=truth,
                seed_suffix=second.anchor_ref,
            )
            if candidate:
                output.append(candidate)
    return output


def build_count_candidates(ctx: ParentContext) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for anchor in ctx.by_type.get("Number", []) or []:
        if not any(phrase in anchor.question.lower() for phrase in ("how many", "how often", "how long", "what age", "what number")):
            continue
        distractors = numeric_distractors(ctx, anchor)
        candidate = make_mcq_candidate(
            ctx=ctx,
            qa_operation="count",
            qa_subtype="explicit_count_lookup",
            answer_type="Number",
            question_text=anchor.question,
            gold_anchor=anchor,
            distractors=distractors,
            support_anchors=[anchor],
            seed_suffix="count",
        )
        if candidate:
            output.append(candidate)
    by_entity: dict[str, list[Anchor]] = defaultdict(list)
    for anchor in ctx.event_anchors:
        by_entity[anchor.entity].append(anchor)
    for entity, anchors in sorted(by_entity.items()):
        if len(anchors) < 2:
            continue
        count_text = str(len(anchors))
        gold = Anchor(
            anchor_ref=f"anchor_{safe_slug(entity)}_event_count_{stable_hash(ctx.parent_id + entity, 8)}",
            parent_id=ctx.parent_id,
            source_sample_id=ctx.source_sample_id,
            entity=entity,
            answer_type="Number",
            answer=count_text,
            normalized_answer=normalize_text(count_text),
            question=f"How many update details for {entity} are used in this set?",
            event_text=f"{entity} has {count_text} update details in this set",
            anchor_kind="derived_graph_anchor_count",
        )
        distractors = numeric_distractors(ctx, gold)
        candidate = make_mcq_candidate(
            ctx=ctx,
            qa_operation="count",
            qa_subtype="entity_anchor_count",
            answer_type="Number",
            question_text=f"How many update details are associated with {entity} in this sample?",
            gold_anchor=gold,
            distractors=distractors,
            support_anchors=anchors[:4],
            seed_suffix=f"entity_count_{entity}",
        )
        if candidate:
            output.append(candidate)
    return output


def build_visibility_candidates(ctx: ParentContext) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    person_pool = list(ctx.entity_anchors)
    for anchor in ctx.permission_anchors:
        gold_entity = next((item for item in person_pool if item.answer == anchor.answer), None)
        gold = gold_entity or anchor
        distractors = [item for item in person_pool if normalize_text(item.answer) != normalize_text(anchor.answer)]
        candidate = make_mcq_candidate(
            ctx=ctx,
            qa_operation="visibility",
            qa_subtype="scope_to_person",
            answer_type="Person",
            question_text="Who was named as the person with access to the limited note?",
            gold_anchor=gold,
            distractors=distractors[:3],
            support_anchors=[anchor],
            seed_suffix="vis_person",
            min_distractors=1,
        )
        if candidate:
            output.append(candidate)
        yesno = make_yesno_candidate(
            ctx=ctx,
            qa_operation="visibility",
            qa_subtype="scope_yesno",
            question_text="Was the limited note visible to everyone?",
            support_anchor=anchor,
            proposition="The limited note was visible to everyone.",
            truth_value=False,
            seed_suffix="vis_yesno",
        )
        if yesno:
            output.append(yesno)
        only_named_person = make_yesno_candidate(
            ctx=ctx,
            qa_operation="visibility",
            qa_subtype="scope_person_yesno",
            question_text=f"Was {anchor.answer} named as the only person with access to the limited note?",
            support_anchor=anchor,
            proposition=f"{anchor.answer} was named as the only person with access to the limited note.",
            truth_value=True,
            seed_suffix="vis_person_yesno",
        )
        if only_named_person:
            output.append(only_named_person)
    return output


def is_causal_question(question: str) -> bool:
    ql = question.lower()
    return ql.startswith(("why ", "what motivated", "what inspired", "how did ")) or "because" in ql or "reason" in ql


def build_cause_consequence_candidates(ctx: ParentContext) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for anchor in ctx.event_anchors:
        if not is_causal_question(anchor.question):
            continue
        distractors = same_type_distractors(ctx, anchor.answer_type, anchor)
        candidate = make_mcq_candidate(
            ctx=ctx,
            qa_operation="cause_consequence",
            qa_subtype="cause_lookup",
            answer_type=anchor.answer_type,
            question_text=anchor.question,
            gold_anchor=anchor,
            distractors=distractors,
            support_anchors=[anchor],
            seed_suffix="cause",
        )
        if candidate:
            output.append(candidate)
    events = event_pool(ctx)
    by_entity: dict[str, list[Anchor]] = defaultdict(list)
    for event in events:
        by_entity[event.entity].append(event)
    for entity, entity_events in by_entity.items():
        ordered = sorted(entity_events, key=lambda item: (item.time_index, item.anchor_ref))
        for earlier, later in zip(ordered, ordered[1:]):
            distractors = [item for item in events if item.anchor_ref not in {earlier.anchor_ref, later.anchor_ref}]
            candidate = make_mcq_candidate(
                ctx=ctx,
                qa_operation="cause_consequence",
                qa_subtype="next_event_after",
                answer_type="Event",
                question_text=f"What happened after \"{short_answer(earlier.answer, 72)}\" for {entity}?",
                gold_anchor=later,
                distractors=distractors[:3],
                support_anchors=[earlier, later],
                seed_suffix=f"next_{earlier.anchor_ref}",
            )
            if candidate:
                output.append(candidate)
    return output


def relation_candidate_answer_types() -> set[str]:
    return {
        "ObjectOrCommonNoun",
        "LocationOrScope",
        "GroupOrOrg",
        "ProperNounOrNamedEntity",
        "AttributeValue",
        "Person",
    }


def build_relation_lookup_candidates(ctx: ParentContext) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for anchor in ctx.event_anchors:
        if anchor.answer_type not in relation_candidate_answer_types():
            continue
        distractors = numeric_distractors(ctx, anchor) if anchor.answer_type == "Number" else same_type_distractors(ctx, anchor.answer_type, anchor)
        question = anchor.question
        if question.lower().startswith("when "):
            continue
        candidate = make_mcq_candidate(
            ctx=ctx,
            qa_operation="relation_lookup",
            qa_subtype="entity_relation",
            answer_type=anchor.answer_type,
            question_text=question,
            gold_anchor=anchor,
            distractors=distractors,
            support_anchors=[anchor],
            seed_suffix="relation",
        )
        if candidate:
            output.append(candidate)
    return output


def candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    query_plan = candidate.get("query_plan") if isinstance(candidate.get("query_plan"), dict) else {}
    return (
        str(candidate.get("parent_id") or ""),
        str(candidate.get("qa_operation") or ""),
        normalize_text(candidate.get("question_text")),
        normalize_text(candidate.get("gold_answer_text") or query_plan.get("gold_answer")),
    )


def sample_identity(sample: dict[str, Any]) -> tuple[str, str, str, str]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    query_plan = sample.get("query_plan") if isinstance(sample.get("query_plan"), dict) else {}
    if not query_plan and isinstance(qa.get("query_plan"), dict):
        query_plan = qa["query_plan"]
    return (
        str(sample.get("parent_id") or ""),
        str(sample.get("qa_operation") or qa.get("qa_operation") or ""),
        normalize_text(sample.get("question_text") or qa.get("question_text") or qa.get("question") or sample.get("question")),
        normalize_text(sample.get("gold_answer_text") or query_plan.get("gold_answer")),
    )


def complete_identity(key: tuple[str, str, str, str]) -> bool:
    return all(bool(part) for part in key)


def build_candidates(contexts: list[ParentContext]) -> tuple[list[dict[str, Any]], Counter]:
    builders = (
        build_temporal_candidates,
        build_bridge_candidates,
        build_comparison_candidates,
        build_visibility_candidates,
        build_count_candidates,
        build_cause_consequence_candidates,
        build_relation_lookup_candidates,
    )
    candidates: list[dict[str, Any]] = []
    skip_reasons: Counter = Counter()
    seen_keys: set[tuple[str, str, str, str]] = set()
    for ctx in contexts:
        for builder in builders:
            built = builder(ctx)
            if not built:
                skip_reasons[f"{builder.__name__}:no_candidate"] += 1
            for candidate in built:
                key = candidate_identity(candidate)
                if key in seen_keys:
                    skip_reasons["duplicate_candidate"] += 1
                    continue
                seen_keys.add(key)
                candidate["_source_sort_key"] = stable_hash(key, 16)
                candidates.append(candidate)
    candidates.sort(key=lambda item: (item.get("qa_operation"), item.get("answer_type"), item.get("_source_sort_key")))
    return candidates, skip_reasons


def select_candidates(candidates: list[dict[str, Any]], target_clean: int) -> list[dict[str, Any]]:
    target_counts = dict(TARGET_OPERATION_COUNTS)
    if target_clean != sum(target_counts.values()):
        scale = target_clean / sum(target_counts.values())
        target_counts = {key: int(math.floor(value * scale)) for key, value in target_counts.items()}
        while sum(target_counts.values()) < target_clean:
            for key in TARGET_OPERATION_COUNTS:
                target_counts[key] += 1
                if sum(target_counts.values()) >= target_clean:
                    break
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    parent_counts: Counter = Counter()
    answer_counts: Counter = Counter()
    subtype_counts: Counter = Counter()
    op_counts: Counter = Counter()
    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_op[str(candidate.get("qa_operation"))].append(candidate)

    def take_from_pool(op: str, quota: int, max_parent_count: int) -> None:
        pool = [item for item in by_op.get(op, []) if item["_source_sort_key"] not in used_keys]
        while op_counts[op] < quota:
            eligible = [item for item in pool if item["_source_sort_key"] not in used_keys and parent_counts[item["parent_id"]] < max_parent_count]
            if not eligible:
                eligible = [item for item in pool if item["_source_sort_key"] not in used_keys]
            if not eligible:
                break
            eligible.sort(
                key=lambda item: (
                    subtype_counts[(op, str(item.get("qa_subtype")))],
                    answer_counts[str(item.get("answer_type"))],
                    parent_counts[str(item.get("parent_id"))],
                    item["_source_sort_key"],
                )
            )
            chosen = eligible[0]
            used_keys.add(chosen["_source_sort_key"])
            selected.append(chosen)
            op_counts[op] += 1
            subtype_counts[(op, str(chosen.get("qa_subtype")))] += 1
            answer_counts[str(chosen.get("answer_type"))] += 1
            parent_counts[str(chosen.get("parent_id"))] += 1

    for op, quota in target_counts.items():
        take_from_pool(op, quota, max_parent_count=4)
    if len(selected) < target_clean:
        for op, quota in target_counts.items():
            take_from_pool(op, quota, max_parent_count=8)
    if len(selected) < target_clean:
        leftovers = [item for item in candidates if item["_source_sort_key"] not in used_keys]
        leftovers.sort(key=lambda item: (op_counts[str(item.get("qa_operation"))], answer_counts[str(item.get("answer_type"))], item["_source_sort_key"]))
        for candidate in leftovers:
            if len(selected) >= target_clean:
                break
            if candidate.get("qa_operation") == "temporal_state_update" and op_counts["temporal_state_update"] >= target_counts["temporal_state_update"]:
                continue
            used_keys.add(candidate["_source_sort_key"])
            selected.append(candidate)
            op_counts[str(candidate.get("qa_operation"))] += 1
            subtype_counts[(str(candidate.get("qa_operation")), str(candidate.get("qa_subtype")))] += 1
            answer_counts[str(candidate.get("answer_type"))] += 1
            parent_counts[str(candidate.get("parent_id"))] += 1
    selected = selected[:target_clean]
    selected.sort(key=lambda item: (QA_OPERATIONS.index(str(item.get("qa_operation"))) if item.get("qa_operation") in QA_OPERATIONS else 999, item["_source_sort_key"]))
    return selected


def make_clean_sample(candidate: dict[str, Any], ctx_by_source: dict[str, ParentContext], index: int) -> dict[str, Any]:
    ctx = ctx_by_source[candidate["source_sample_id"]]
    sample = copy.deepcopy(ctx.row)
    for stale_key in ("evidence_event_ids", "evidence_turn_ids"):
        sample.pop(stale_key, None)
    sample_id = f"qa_op_answer_type_anchor_diversification_{index:04d}_{stable_hash([candidate['source_sample_id'], candidate['_source_sort_key']], 10)}"
    qa = {
        "question_type": "single_choice",
        "question": candidate["question_text"],
        "options": candidate["options"],
        "gold": candidate["gold"],
        "gold_source": "graph_anchor_query_plan",
        "qa_generated_after_alignment": True,
        "qa_operation": candidate["qa_operation"],
        "qa_subtype": candidate["qa_subtype"],
        "answer_type": candidate["answer_type"],
        "answer_format": candidate["answer_format"],
        "question_text": candidate["question_text"],
        "query_plan": candidate["query_plan"],
        "anchor_refs": candidate["anchor_refs"],
    }
    if candidate.get("proposition"):
        qa["proposition"] = candidate["proposition"]
        qa["truth_value"] = bool(candidate.get("truth_value"))
    sample.update(
        {
            "sample_id": sample_id,
            "parent_id": candidate["parent_id"],
            "source_parent_sample_id": candidate["source_sample_id"],
            "generation_backend": "stage64_anchor_diversification_probe",
            "generation_source": "stage2_graph_anchor_query_plan_rewrite",
            "native_challenger_mode": "qa_operation_answer_type_anchor_diversification",
            "qa_operation": candidate["qa_operation"],
            "qa_subtype": candidate["qa_subtype"],
            "answer_type": candidate["answer_type"],
            "answer_format": candidate["answer_format"],
            "question_text": candidate["question_text"],
            "query_plan": candidate["query_plan"],
            "anchor_refs": candidate["anchor_refs"],
            "qa": qa,
            "question": candidate["question_text"],
            "question_type": "single_choice",
            "options": candidate["options"],
            "gold_option": candidate["gold"][0],
            "gold_options": candidate["gold"],
            "gold_answer": candidate["gold"],
            "gold_answers": candidate["gold"],
            "gold_answer_text": candidate["gold_answer_text"],
            "normalized_gold_answer": normalize_text(candidate["gold_answer_text"]),
            "attack_skill": candidate["qa_operation"],
            "attack_plan": {
                "attack_skill": candidate["qa_operation"],
                "difficulty_subtype": candidate["qa_subtype"],
                "reasoning_mechanism": "graph_anchor_query_plan_qa_operation_answer_type",
                "target_solver_weakness": "long_dialogue_anchor_first_operation_reasoning",
                "requires_ordering": candidate["qa_subtype"] in {"ordering", "before_after", "earlier_event", "before_after_yesno"},
                "shortcut_to_avoid": ["state_only_shell", "raw_qa_artifact", "answer_type_mismatch", "unanchored_distractor"],
            },
            "validation": {
                "graph_dialogue_consistent": True,
                "gold_derivable": True,
                "option_valid": True,
                "anti_shortcut_pass": True,
                "warnings": [],
                "details": {
                    "anchor_query_plan_valid": True,
                    "answer_source": "graph_anchor",
                    "operation_answer_type_pair": f"{candidate['qa_operation']}|{candidate['answer_type']}",
                },
            },
            "stage64_diagnostics": {
                "anchor_first": True,
                "source_stage": "stage2_repair_frontier",
                "answer_source": "graph_anchor",
                "gold_anchor_ref": candidate["gold_anchor_ref"],
                "support_anchor_refs": candidate["support_anchor_refs"],
                "distractor_anchor_refs": candidate["distractor_anchor_refs"],
            },
            "solver_eval": {
                "k": 0,
                "predictions": [],
                "exact_correct_count": 0,
                "soft_accuracy": 0.0,
                "partial_credit": 0.0,
                "bucket": "unavailable",
            },
            "frontier_label": {
                "is_frontier": False,
                "category": "unavailable",
                "reason": "k5 not evaluated yet",
            },
        }
    )
    return sample


def base_diagnostics() -> dict[str, Any]:
    return {
        "qa_operation_counts": {},
        "qa_subtype_counts": {},
        "answer_type_counts": {},
        "operation_answer_type_matrix": {},
        "anchor_query_plan_valid_count": 0,
        "anchor_query_plan_fail_count": 0,
        "missing_anchor_count": 0,
        "unresolved_anchor_count": 0,
        "answer_type_mismatch_count": 0,
        "distractor_answer_type_mismatch_count": 0,
        "duplicate_option_count": 0,
        "near_duplicate_option_count": 0,
        "placeholder_option_count": 0,
        "state_shell_leak_count": 0,
        "final_state_shell_overuse_count": 0,
        "option_surface_risk_count": 0,
        "bad_question_count": 0,
        "parser_failure_count": 0,
        "leakage_count": 0,
        "grounding_failure_count": 0,
        "overflow_count": 0,
        "truncated_count": 0,
    }


def diagnostics_for_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = base_diagnostics()
    op_counts = Counter(str(sample.get("qa_operation") or "unknown") for sample in samples)
    subtype_counts = Counter(str(sample.get("qa_subtype") or "unknown") for sample in samples)
    answer_counts = Counter(str(sample.get("answer_type") or "unknown") for sample in samples)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    final_state_count = 0
    for sample in samples:
        op = str(sample.get("qa_operation") or "unknown")
        subtype = str(sample.get("qa_subtype") or "unknown")
        answer_type = str(sample.get("answer_type") or "unknown")
        matrix[op][answer_type] += 1
        if subtype == "final_state":
            final_state_count += 1
        qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
        query_plan = qa.get("query_plan") if isinstance(qa.get("query_plan"), dict) else {}
        refs = []
        for key in ("required_anchor_refs", "supporting_anchor_refs", "gold_anchor_refs"):
            values = query_plan.get(key)
            if isinstance(values, list):
                refs.extend(str(item) for item in values)
        if refs and query_plan.get("answer_source") == "graph_anchor":
            diagnostics["anchor_query_plan_valid_count"] += 1
        else:
            diagnostics["anchor_query_plan_fail_count"] += 1
            diagnostics["missing_anchor_count"] += 1
        question = str(qa.get("question") or sample.get("question_text") or "")
        if surface_bad(question):
            diagnostics["bad_question_count"] += 1
            diagnostics["leakage_count"] += 1
        if STATE_SHELL_RE.search(question):
            diagnostics["state_shell_leak_count"] += 1
        if VISIBLE_INTERNAL_RE.search(question):
            diagnostics["state_shell_leak_count"] += 1
            diagnostics["leakage_count"] += 1
        options = qa.get("options") if isinstance(qa.get("options"), list) else []
        texts = [str(option.get("text") or "") for option in options if isinstance(option, dict)]
        duplicate, near = duplicate_or_near_duplicate(texts)
        if duplicate:
            diagnostics["duplicate_option_count"] += 1
        if near:
            diagnostics["near_duplicate_option_count"] += 1
        if any(surface_bad(text) for text in texts):
            diagnostics["placeholder_option_count"] += 1
            diagnostics["option_surface_risk_count"] += 1
        if any(VISIBLE_INTERNAL_RE.search(text) for text in texts):
            diagnostics["option_surface_risk_count"] += 1
        if any((option.get("answer_type") != answer_type) for option in options if isinstance(option, dict)):
            diagnostics["answer_type_mismatch_count"] += 1
            diagnostics["distractor_answer_type_mismatch_count"] += 1
    diagnostics["qa_operation_counts"] = dict(op_counts.most_common())
    diagnostics["qa_subtype_counts"] = dict(subtype_counts.most_common())
    diagnostics["answer_type_counts"] = dict(answer_counts.most_common())
    diagnostics["operation_answer_type_matrix"] = {op: dict(values) for op, values in matrix.items()}
    diagnostics["final_state_shell_overuse_count"] = max(0, final_state_count - max(1, len(samples) // 10))
    return diagnostics


def eval_args_from_cli(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        skip_eval=bool(args.skip_eval),
        run_k5=True,
        fixed_solver_name="solver_base_restart_stage2_accepted",
        solver_model=str(args.solver_model),
        solver_adapter=None,
        solver_checkpoint=None,
        seed=int(args.seed),
        eval_backend=args.eval_backend,
        base_model=str(args.base_model),
        max_model_len=int(args.max_model_len),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        k5_eval_temperature=float(args.k5_eval_temperature),
        top_p=float(args.top_p),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )


def solver_diagnostic_flags(sample: dict[str, Any]) -> tuple[int, int]:
    se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
    diag = se.get("diagnostics") if isinstance(se.get("diagnostics"), dict) else {}
    return int(bool(diag.get("eval_context_overflow"))), int(bool(diag.get("truncated")))


def bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {f"{idx}/5": 0 for idx in range(6)}
    k1_correct = 0
    k1_seen = 0
    soft_values = []
    bad = parser = leakage = grounding = overflow = truncated = 0
    for row in rows:
        se = row.get("solver_eval") if isinstance(row.get("solver_eval"), dict) else {}
        k1_se = row.get("solver_eval_k1") if isinstance(row.get("solver_eval_k1"), dict) else se
        if int(k1_se.get("k") or 0) == 1:
            k1_seen += 1
            k1_correct += int(int(k1_se.get("exact_correct_count") or 0) == 1)
        if int(se.get("k") or 0) == 5:
            bucket = str(se.get("bucket") or "0/5")
            buckets[bucket] = buckets.get(bucket, 0) + 1
            soft_values.append(float(se.get("soft_accuracy") or 0.0))
        overflow_flag, truncated_flag = solver_diagnostic_flags(row)
        overflow += overflow_flag
        truncated += truncated_flag
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        if validation.get("option_valid") is not True:
            bad += 1
        if validation.get("gold_derivable") is not True:
            grounding += 1
        if validation.get("anti_shortcut_pass") is not True:
            leakage += 1
    boundary = sum(buckets.get(f"{idx}/5", 0) for idx in range(1, 5))
    total = len(rows)
    return {
        "clean_count": total,
        "k1": round(k1_correct / k1_seen, 6) if k1_seen else None,
        "k1_correct": k1_correct,
        "k1_seen": k1_seen,
        "k5_soft": round(sum(soft_values) / len(soft_values), 6) if soft_values else None,
        "0/5": buckets.get("0/5", 0),
        "1/5": buckets.get("1/5", 0),
        "2/5": buckets.get("2/5", 0),
        "3/5": buckets.get("3/5", 0),
        "4/5": buckets.get("4/5", 0),
        "5/5": buckets.get("5/5", 0),
        "boundary": boundary if soft_values else "unavailable",
        "easy": buckets.get("5/5", 0) if soft_values else "unavailable",
        "hard0": buckets.get("0/5", 0) if soft_values else "unavailable",
        "bad_question": bad,
        "parser": parser,
        "leakage": leakage,
        "grounding": grounding,
        "overflow": overflow,
        "truncated": truncated,
    }


def grouped_bucket_summary(samples: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(key_fn(sample) or "unknown")].append(sample)
    return {key: bucket_summary(rows) for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))}


def attach_k1_to_samples(samples: list[dict[str, Any]], k1_rows: list[dict[str, Any]]) -> None:
    k1_by_id = {row.get("sample_id"): row.get("solver_eval") for row in k1_rows if isinstance(row, dict)}
    for sample in samples:
        if sample.get("sample_id") in k1_by_id:
            sample["solver_eval_k1"] = k1_by_id[sample["sample_id"]]


def mark_frontier(samples: list[dict[str, Any]], k5_available: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frontier: list[dict[str, Any]] = []
    hard0: list[dict[str, Any]] = []
    easy: list[dict[str, Any]] = []
    for sample in samples:
        se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
        bucket = str(se.get("bucket") or "")
        if not k5_available:
            sample["frontier_label"] = {
                "is_frontier": False,
                "category": "unavailable",
                "reason": "k5 unavailable; k1 is diagnostic only",
            }
            continue
        if bucket in {"1/5", "2/5", "3/5", "4/5"}:
            sample["frontier_label"] = {
                "is_frontier": True,
                "category": "boundary",
                "reason": "true frontier from k5 bucket 1/5-4/5",
            }
            frontier.append(sample)
        elif bucket == "0/5":
            sample["frontier_label"] = {"is_frontier": False, "category": "hard0", "reason": "k5 bucket 0/5"}
            hard0.append(sample)
        elif bucket == "5/5":
            sample["frontier_label"] = {"is_frontier": False, "category": "easy", "reason": "k5 bucket 5/5"}
            easy.append(sample)
        else:
            sample["frontier_label"] = {"is_frontier": False, "category": "unknown", "reason": "missing k5 bucket"}
    return frontier, hard0, easy


def operation_answer_type_matrix_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = f"{sample.get('qa_operation')}|{sample.get('answer_type')}"
        groups[key].append(sample)
    return {key: bucket_summary(rows) for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))}


BUCKET_REPORT_COLUMNS = [
    "clean_count",
    "k1",
    "k5_soft",
    "0/5",
    "1/5",
    "2/5",
    "3/5",
    "4/5",
    "5/5",
    "boundary",
    "easy",
    "hard0",
    "bad_question",
    "parser",
    "leakage",
    "grounding",
    "overflow",
    "truncated",
]


def bucket_markdown_table(title: str, rows: dict[str, Any], *, limit: int | None = None) -> list[str]:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    items = list(rows.items())
    if limit is not None:
        items = items[:limit]
    lines = [
        f"## {title}",
        "",
        "| bucket | " + " | ".join(BUCKET_REPORT_COLUMNS) + " |",
        "| --- | " + " | ".join("---" for _ in BUCKET_REPORT_COLUMNS) + " |",
    ]
    for key, values in items:
        if not isinstance(values, dict):
            continue
        lines.append("| " + str(key) + " | " + " | ".join(cell(values.get(col, "")) for col in BUCKET_REPORT_COLUMNS) + " |")
    lines.append("")
    return lines


def write_report(
    *,
    samples: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    operation_summary: dict[str, Any],
    answer_type_summary: dict[str, Any],
    pair_summary: dict[str, Any],
    parent_summary: dict[str, Any],
    output_dir: Path,
    source_path: Path,
    skip_reasons: Counter,
    k5_available: bool,
) -> None:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    overall = bucket_summary(samples)
    op_counts = Counter(sample.get("qa_operation") for sample in samples)
    temporal_share = round(op_counts.get("temporal_state_update", 0) / len(samples), 4) if samples else 0.0
    clean_answer_types = sorted(diagnostics.get("answer_type_counts", {}).items(), key=lambda item: (-item[1], item[0]))
    boundary_pairs = sorted(
        ((key, value) for key, value in pair_summary.items() if isinstance(value.get("boundary"), int)),
        key=lambda item: (-int(item[1].get("boundary") or 0), -int(item[1].get("clean_count") or 0), item[0]),
    )
    easy_pairs = sorted(
        ((key, value) for key, value in pair_summary.items() if isinstance(value.get("easy"), int)),
        key=lambda item: (-int(item[1].get("easy") or 0), -int(item[1].get("clean_count") or 0), item[0]),
    )
    hard_pairs = sorted(
        ((key, value) for key, value in pair_summary.items() if isinstance(value.get("hard0"), int)),
        key=lambda item: (-int(item[1].get("hard0") or 0), -int(item[1].get("clean_count") or 0), item[0]),
    )
    next_trainmix = bool(k5_available and isinstance(overall.get("boundary"), int) and int(overall.get("boundary") or 0) >= max(12, len(samples) // 10))
    summary = {
        "stage": "64_qa_operation_answer_type_anchor_diversification_probe",
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "clean_count": len(samples),
        "k5_available": k5_available,
        "true_frontier_definition": "k5 bucket 1/5-4/5 only",
        "k1_frontier_policy": "diagnostic_only",
        "overall": overall,
        "qa_operation_counts": dict(op_counts.most_common()),
        "temporal_state_update_share": temporal_share,
        "diagnostics": diagnostics,
        "skip_reasons": dict(skip_reasons.most_common()),
        "operation_bucket_summary": operation_summary,
        "answer_type_bucket_summary": answer_type_summary,
        "operation_answer_type_matrix": pair_summary,
        "parent_bucket_summary": parent_summary,
        "top_boundary_pairs": [(key, value.get("boundary")) for key, value in boundary_pairs[:10]],
        "top_easy_pairs": [(key, value.get("easy")) for key, value in easy_pairs[:10]],
        "top_hard0_pairs": [(key, value.get("hard0")) for key, value in hard_pairs[:10]],
        "worth_next_trainmix_plan": next_trainmix,
    }
    write_json(reports / "qa_operation_answer_type_anchor_diversification_summary.json", summary)
    lines = [
        "# QA Operation / Answer Type Anchor Diversification Probe",
        "",
        f"- Source: `{source_path}`",
        f"- Clean samples: {len(samples)}",
        f"- Fixed solver: `solver_base_restart_stage2_accepted`",
        "- Frontier rule: true frontier is k5 bucket 1/5-4/5 only; k1 is diagnostic only.",
        "",
        "## Direct Answers",
        "",
        f"1. QA shell expansion: yes. Generated operations are {dict(op_counts.most_common())}.",
        f"2. temporal_state_update share: {temporal_share:.2%}; it is capped as one operation family, not counted as four schemas.",
        f"3. Clean-friendly answer types: {clean_answer_types[:8]}.",
        f"4. Boundary-rich operation/answer_type pairs: {[(key, value.get('boundary')) for key, value in boundary_pairs[:8]]}.",
        f"5. Too-easy pairs: {[(key, value.get('easy')) for key, value in easy_pairs[:8]]}.",
        f"6. Too-hard/bad pairs: {[(key, value.get('hard0')) for key, value in hard_pairs[:8]]}; bad diagnostics are in the summary JSON.",
        f"7. Next trainmix plan: {'worth planning, but not executed here' if next_trainmix else 'not enough boundary signal yet'}.",
        "",
        "## Overall Bucket",
        "",
        "```json",
        json.dumps(overall, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    lines.extend(bucket_markdown_table("By QA Operation", operation_summary))
    lines.extend(bucket_markdown_table("By Answer Type", answer_type_summary))
    lines.extend(bucket_markdown_table("By Operation Answer Type Pair", pair_summary))
    lines.extend(bucket_markdown_table("By Parent ID", parent_summary, limit=40))
    lines.extend(
        [
        "## Diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        "```",
        ]
    )
    (reports / "qa_operation_answer_type_anchor_diversification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args: argparse.Namespace) -> None:
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    clean_samples_path = output_dir / "clean_samples.jsonl"

    existing_samples: list[dict[str, Any]] = []
    if clean_samples_path.exists():
        if bool(args.append_existing):
            existing_samples = read_jsonl(clean_samples_path)
        elif not bool(args.overwrite):
            raise RuntimeError(
                f"{clean_samples_path} already exists. Use --append-existing/--resume to continue, "
                "or --overwrite to intentionally regenerate the output directory."
            )

    rows = read_jsonl(source_path)
    if not rows:
        raise RuntimeError(f"No source rows loaded from {source_path}")
    contexts = [build_parent_context(row, idx) for idx, row in enumerate(rows)]
    ctx_by_source = {ctx.source_sample_id: ctx for ctx in contexts}
    candidates, skip_reasons = build_candidates(contexts)
    target_clean = int(args.target_clean)
    if target_clean < 0:
        raise RuntimeError("--target-clean must be non-negative.")
    target_new = target_clean
    append_diagnostics: dict[str, Any] = {
        "append_existing": bool(args.append_existing),
        "append_existing_rows": len(existing_samples),
        "append_new_target": target_new,
        "append_new_count": 0,
    }
    if bool(args.append_existing):
        existing_keys: set[tuple[str, str, str, str]] = set()
        for sample in existing_samples:
            key = sample_identity(sample)
            if complete_identity(key):
                existing_keys.add(key)
        before_filter = len(candidates)
        if existing_keys:
            candidates = [candidate for candidate in candidates if candidate_identity(candidate) not in existing_keys]
        skipped_existing = before_filter - len(candidates)
        skip_reasons["append_existing_rows"] = len(existing_samples)
        skip_reasons["append_existing_identity_keys"] = len(existing_keys)
        skip_reasons["append_existing_candidate_skip"] = skipped_existing
        if args.resume_target_mode == "total":
            target_new = max(0, target_clean - len(existing_samples))
        append_diagnostics.update(
            {
                "append_existing_identity_keys": len(existing_keys),
                "append_existing_candidate_skip": skipped_existing,
                "append_new_target": target_new,
                "resume_target_mode": args.resume_target_mode,
            }
        )
    selected = select_candidates(candidates, target_new)
    min_clean = min(int(args.min_clean), target_new) if bool(args.append_existing) else int(args.min_clean)
    if len(selected) < min_clean:
        raise RuntimeError(
            f"Only generated {len(selected)} new clean anchor-first QA candidates; minimum requested is {min_clean}. "
            f"Existing rows kept: {len(existing_samples)}."
        )
    start_index = len(existing_samples) if bool(args.append_existing) else 0
    new_samples = [make_clean_sample(candidate, ctx_by_source, start_index + idx) for idx, candidate in enumerate(selected)]
    append_diagnostics["append_new_count"] = len(new_samples)
    samples = existing_samples + new_samples if bool(args.append_existing) else new_samples
    diagnostics = diagnostics_for_samples(samples)
    diagnostics.update(append_diagnostics)
    write_jsonl(output_dir / "clean_samples.jsonl", samples)
    write_jsonl(
        output_dir / "query_plans.jsonl",
        [
            {
                "sample_id": sample["sample_id"],
                "parent_id": sample["parent_id"],
                "source_parent_sample_id": sample["source_parent_sample_id"],
                "qa_operation": sample["qa_operation"],
                "qa_subtype": sample["qa_subtype"],
                "answer_type": sample["answer_type"],
                "answer_format": sample["answer_format"],
                "question_text": sample["question_text"],
                "query_plan": sample["query_plan"],
                "anchor_refs": sample["anchor_refs"],
                "options": sample["options"],
            }
            for sample in samples
        ],
    )

    solver_summary = current_reports.run_solver_evaluations(
        samples=samples,
        args=eval_args_from_cli(args),
        output_dir=output_dir,
        report_dir=report_dir,
    )
    k1_rows = read_jsonl(output_dir / "solver_k1_eval.jsonl")
    attach_k1_to_samples(samples, k1_rows)
    k5_rows = read_jsonl(output_dir / "solver_k5_eval.jsonl")
    k5_available = bool(k5_rows) and not bool(args.skip_eval)
    frontier, hard0, easy = mark_frontier(samples, k5_available)
    diagnostics = diagnostics_for_samples(samples)
    diagnostics["overflow_count"] = sum(solver_diagnostic_flags(sample)[0] for sample in samples)
    diagnostics["truncated_count"] = sum(solver_diagnostic_flags(sample)[1] for sample in samples)
    diagnostics["solver_summary"] = solver_summary
    diagnostics.update(append_diagnostics)

    operation_summary = grouped_bucket_summary(samples, lambda sample: sample.get("qa_operation"))
    answer_type_summary = grouped_bucket_summary(samples, lambda sample: sample.get("answer_type"))
    pair_summary = operation_answer_type_matrix_summary(samples)
    parent_summary = grouped_bucket_summary(samples, lambda sample: sample.get("parent_id"))

    write_jsonl(output_dir / "clean_samples.jsonl", samples)
    write_jsonl(output_dir / "frontier_all.jsonl", frontier)
    write_jsonl(output_dir / "hard0_diagnostic.jsonl", hard0)
    write_jsonl(output_dir / "easy_control.jsonl", easy)
    write_json(output_dir / "operation_bucket_summary.json", operation_summary)
    write_json(output_dir / "answer_type_bucket_summary.json", answer_type_summary)
    write_json(output_dir / "operation_answer_type_matrix.json", pair_summary)
    write_json(output_dir / "parent_bucket_summary.json", parent_summary)
    write_report(
        samples=samples,
        diagnostics=diagnostics,
        operation_summary=operation_summary,
        answer_type_summary=answer_type_summary,
        pair_summary=pair_summary,
        parent_summary=parent_summary,
        output_dir=output_dir,
        source_path=source_path,
        skip_reasons=skip_reasons,
        k5_available=k5_available,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage64 anchor-first QA operation / answer-type diversification probe.")
    parser.add_argument("--source", default=str(DEFAULT_STAGE2_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-clean", type=int, default=120)
    parser.add_argument("--min-clean", type=int, default=80)
    parser.add_argument(
        "--append-existing",
        "--resume",
        dest="append_existing",
        action="store_true",
        help="Continue from an existing clean_samples.jsonl in the output dir and append new non-duplicate samples.",
    )
    parser.add_argument(
        "--resume-target-mode",
        choices=("new", "total"),
        default="new",
        help="In resume mode, treat --target-clean as new rows to add, or as the desired final total row count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing clean_samples.jsonl instead of resuming from it.",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-backend", choices=("auto", "vllm", "transformers", "heuristic"), default="auto")
    parser.add_argument("--solver-model", default=str(DEFAULT_STAGE2_SOLVER))
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--k5-eval-temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    run_probe(parse_args())


if __name__ == "__main__":
    main()
