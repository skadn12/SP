#!/usr/bin/env python3
"""Rebuild evidence-grounded Stage2 parent graphs with a frozen local model.

This is a migration utility, not a Stage3 challenger.  It never exposes the
legacy QA/options/gold or legacy semantic labels to the model.  Each request
contains only clean dialogue turns referenced by a source event.  Model output
is accepted only when entity, value, relation, and status are independently
supported by those turns and the resulting event passes semantic_graph_v2.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import build_stage3_parent_v2  # noqa: E402
from scripts.current_method import semantic_graph_v2  # noqa: E402


SCHEMA_VERSION = "stage2_canonical_graph_rebuild_v3"
PARENT_SCHEMA_VERSION = "stage3_parent_v3"
SEMANTIC_EXCLUSION_SCHEMA_VERSION = "stage2_semantic_audit_exclusion_v1"
DEFAULT_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
LEGACY_QA_RE = re.compile(
    r"raw[\s_-]*qa|(?:^|\s)[QA]\s*:\s*|\b(?:question|answer)\s*:\s*",
    re.IGNORECASE,
)
SLOT_RE = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
GENERIC_SLOTS = {
    "activity",
    "answer",
    "attribute",
    "detail",
    "event",
    "fact",
    "generic_fact",
    "generic_state",
    "information",
    "interest",
    "condition",
    "date",
    "location",
    "method",
    "memory_state",
    "name",
    "plan",
    "place",
    "property",
    "qa",
    "relation",
    "state",
    "status",
    "thing",
    "time",
    "topic",
    "type",
    "value",
}
STATUS_EVIDENCE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "belief": (re.compile(r"\bbeliev\w*|\bthink\w*|\bassum\w*|\bunsure\b", re.I),),
    "canceled": (re.compile(r"\bcancel\w*|\bcalled\s+off\b|\bwithdraw\w*", re.I),),
    "delayed_not_active": (
        re.compile(r"\bdelay\w*|\bnot\s+(?:yet\s+)?active\b|\bwaiting\b|\bplanned\s+for\s+later\b|\bnot\s+taken\s+effect\b", re.I),
    ),
    "expired": (re.compile(r"\bexpir\w*|\blapsed\b|\bno\s+longer\s+valid\b", re.I),),
    "false": (re.compile(r"\bfalse\b|\bincorrect\b|\bmistaken\b|\bnot\s+true\b", re.I),),
    "outdated": (
        re.compile(r"\boutdated\b|\bold\s+version\b|\bno\s+longer\b|\bsupersed\w*|\bstale\b", re.I),
    ),
    "report": (re.compile(r"\breport\w*|\bclaim\w*|\bsomeone\s+said\b|\bunconfirmed\b", re.I),),
    "scope_limited": (
        re.compile(r"\blimit\w*|\brestrict\w*|\bvisible\s+(?:only\s+)?to\b|\bpermission\b", re.I),
    ),
}
NONACTIVE_STATUSES = frozenset(STATUS_EVIDENCE_PATTERNS)
ENTITY_STOPWORDS = {
    "analyst",
    "assistant",
    "coordinator",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "it",
    "reviewer",
    "she",
    "speaker",
    "someone",
    "thanks",
    "they",
    "them",
    "their",
    "we",
    "you",
}
SLOT_STOPWORDS = {
    "current",
    "detail",
    "fact",
    "favorite",
    "latest",
    "new",
    "personal",
    "primary",
    "specific",
}
SLOT_TYPE_TOKENS = {
    "age",
    "amount",
    "count",
    "date",
    "duration",
    "frequency",
    "location",
    "method",
    "number",
    "quantity",
    "source",
    "status",
    "time",
    "type",
}
SLOT_CUE_ALIASES: dict[str, set[str]] = {
    "city": {"move", "moved", "live", "lived", "visit", "visited", "travel"},
    "location": {"move", "moved", "live", "lived", "visit", "visited", "travel", "place"},
    "destination": {"go", "going", "move", "moved", "trip", "travel", "visit"},
    "reason": {"because", "due", "inspired", "why"},
    "duration": {"for", "since", "year", "years", "month", "months", "day", "days"},
    "frequency": {"daily", "weekly", "monthly", "yearly", "often", "once", "twice", "every"},
    "date": {"date", "dated", "before", "after", "since", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"},
    "time": {"time", "before", "after", "since", "morning", "afternoon", "evening"},
    "preference": {"like", "likes", "enjoy", "enjoys", "prefer", "prefers", "favorite"},
    "method": {"by", "using", "through", "with"},
    "reliance": {"rely", "relies", "relied", "reliant"},
    "visit": {"visit", "visited", "went", "tried", "ate"},
    "played": {"play", "played", "playing"},
    "joining": {"join", "joined", "joining"},
}
DATE_VALUE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|\b\d{1,2}(?:st|nd|rd|th)?\b|\b(?:recently|today|yesterday|tomorrow|last|next|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
DURATION_VALUE_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|week|month|year)s?\b|\bsince\b",
    re.I,
)
FREQUENCY_VALUE_RE = re.compile(
    r"\b(?:daily|weekly|monthly|yearly|once|twice|every|often|times?\s+(?:a|per))\b",
    re.I,
)
DATE_SPAN_PATTERNS = (
    re.compile(
        r"\b(?:the\s+)?(?:day|week|month)\s+(?:before|after)\s+\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[, ]+\s*(?:19|20)\d{2}\b",
        re.I,
    ),
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[, ]+\s*(?:19|20)\d{2}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
        r"\d{1,2}(?:st|nd|rd|th)?[,]?\s+(?:19|20)\d{2}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[, ]+\s*(?:19|20)\d{2}\b",
        re.I,
    ),
    re.compile(r"\b(?:19|20)\d{2}\b", re.I),
    re.compile(
        r"\b(?:last|next)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year)\b|\b(?:recently|today|yesterday|tomorrow)\b",
        re.I,
    ),
)
PROSPECTIVE_FACT_RE = re.compile(
    r"\b(?:plan(?:s|ned|ning)?|proposal|proposed|suggest(?:s|ed|ing)?|intend(?:s|ed|ing)?|prepar(?:e|es|ed|ing)|getting\s+ready|slated|pending|due\s+to|aim(?:s|ed|ing)?|hope(?:s|d|ing)?|expect(?:s|ed|ing)?|schedule(?:s|d)?|will|would|is\s+to|are\s+to|going\s+to|set\s+to)\b",
    re.I,
)
OUTER_MODAL_SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("considering", re.compile(r"\bconsider(?:s|ed|ing)?\b", re.I)),
    ("plans_to", re.compile(r"\bplan(?:s|ned|ning)?\s+to\b", re.I)),
    ("intends_to", re.compile(r"\bintend(?:s|ed|ing)?\s+to\b", re.I)),
    ("hopes_to", re.compile(r"\bhope(?:s|d|ing)?\s+to\b", re.I)),
    ("thinking_about", re.compile(r"\bthink(?:s|ing)?\s+about\b", re.I)),
)
FALLBACK_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|I'm|I've|I'd|my|mine|we|we've|our|ours)\b",
    re.I,
)
FALLBACK_FACT_PREDICATE_RE = re.compile(
    r"\b(?:adopt(?:ed|ing)?|bought|chose|found|got|had|have|hiked|joined|live|lived|made|moved|own|owned|set\s+up|signed\s+up|started|visited|went|work|worked)\b",
    re.I,
)
FALLBACK_UNCERTAIN_RE = re.compile(
    r"\b(?:can't\s+wait|considering|could|hope|hoping|i'm\s+sure|looking\s+for|meaning\s+to|maybe|might|think|would|will|you'll)\b",
    re.I,
)
NEGATIVE_ABSENCE_RE = re.compile(
    r"\b(?:is|are|was|were)\s+not\s+(?:mentioned|provided|recorded|stated)\b|\bno\s+(?:information|detail|record)\b",
    re.I,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_match(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_space(value).casefold()).strip()


def phrase_in_text(phrase: Any, text: Any) -> bool:
    left = normalize_match(phrase)
    right = normalize_match(text)
    return bool(left and f" {left} " in f" {right} ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            value = copy.deepcopy(value)
            value["_source_line_number"] = line_number
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_semantic_audit_exclusions(
    paths: Iterable[Path],
) -> list[dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for record in read_jsonl(path.resolve()):
            source_event_id = normalize_space(record.get("source_event_id"))
            if (
                record.get("schema_version") != SEMANTIC_EXCLUSION_SCHEMA_VERSION
                or not source_event_id
                or source_event_id in seen
                or int(record.get("source_line_number") or 0) <= 0
                or normalize_space(record.get("decision")) != "quarantine"
                or not normalize_space(record.get("reason"))
            ):
                raise RuntimeError(
                    f"Invalid or duplicate semantic exclusion: {record}"
                )
            seen.add(source_event_id)
            exclusions.append(copy.deepcopy(record))
    return exclusions


def apply_semantic_audit_exclusions(
    requests: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    event_locations = {
        normalize_space(event.get("source_event_id")): (request, event)
        for request in requests
        for event in request.get("events") or []
        if isinstance(event, dict) and normalize_space(event.get("source_event_id"))
    }
    applied: list[dict[str, Any]] = []
    touched_request_ids: set[int] = set()
    for exclusion in exclusions:
        source_event_id = normalize_space(exclusion.get("source_event_id"))
        located = event_locations.get(source_event_id)
        if not located:
            raise RuntimeError(
                f"Semantic exclusion does not match an eligible source event: {source_event_id}"
            )
        request, _event = located
        source_line_number = int(request.get("source_line_number") or 0)
        if source_line_number != int(exclusion.get("source_line_number") or 0):
            raise RuntimeError(
                f"Semantic exclusion source line mismatch for {source_event_id}"
            )
        request["events"] = [
            event
            for event in request.get("events") or []
            if normalize_space(event.get("source_event_id")) != source_event_id
        ]
        request.setdefault("skipped_events", []).append(
            {
                "source_event_id": source_event_id,
                "source_event_order": int(_event.get("source_event_order") or 0),
                "reason": "independent_semantic_audit_exclusion",
                "semantic_audit_reason": normalize_space(exclusion.get("reason")),
                "semantic_audit_detail": normalize_space(exclusion.get("detail")),
            }
        )
        touched_request_ids.add(id(request))
        applied.append(
            {
                **copy.deepcopy(exclusion),
                "request_id": request.get("request_id"),
                "source_sample_id": request.get("source_sample_id"),
                "disposition": "quarantined_before_canonical_admission",
            }
        )
    for request in requests:
        if id(request) not in touched_request_ids:
            continue
        request["request_id"] = (
            "stage2_graph_extract_"
            f"{sha256_json([request.get('source_sample_id'), request.get('events'), request.get('skipped_events')])[:20]}"
        )
    for record in applied:
        request = next(
            request
            for request in requests
            if request.get("source_sample_id") == record.get("source_sample_id")
        )
        record["request_id"] = request.get("request_id")
    return applied


def dialogue_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue = row.get("dialogue")
    if isinstance(dialogue, dict) and isinstance(dialogue.get("turns"), list):
        return [turn for turn in dialogue["turns"] if isinstance(turn, dict)]
    if isinstance(dialogue, list):
        return [turn for turn in dialogue if isinstance(turn, dict)]
    turns = row.get("turns")
    return [turn for turn in turns if isinstance(turn, dict)] if isinstance(turns, list) else []


def event_has_legacy_qa(event: dict[str, Any]) -> bool:
    return bool(
        LEGACY_QA_RE.search(normalize_space(event.get("slot")))
        or LEGACY_QA_RE.search(normalize_space(event.get("value")))
        or "?" in normalize_space(event.get("value"))
    )


def turn_has_legacy_qa(turn: dict[str, Any]) -> bool:
    return bool(LEGACY_QA_RE.search(normalize_space(turn.get("text"))))


def event_turn_ids(event: dict[str, Any]) -> list[str]:
    values = event.get("evidence_turn_ids") or event.get("turn_ids") or []
    if not isinstance(values, list):
        values = [values]
    return list(
        dict.fromkeys(normalize_space(value) for value in values if normalize_space(value))
    )


def clean_existing_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event_has_legacy_qa(event):
        return None
    candidate = {
        "event_id": normalize_space(
            event.get("event_id") or event.get("event_instance_id")
        ),
        "entity": copy.deepcopy(event.get("entity")),
        "slot": copy.deepcopy(event.get("slot")),
        "value": copy.deepcopy(event.get("value")),
        "status": copy.deepcopy(event.get("status")),
        "time_index": copy.deepcopy(event.get("time_index")),
        "evidence_turn_ids": event_turn_ids(event),
        "required_for_dialogue": True,
        "provenance": {
            "migration": SCHEMA_VERSION,
            "source": "existing_semantic_event",
            "source_event_id": normalize_space(
                event.get("event_instance_id") or event.get("event_id")
            ),
        },
    }
    if event.get("previous_value") not in (None, "", "unknown"):
        candidate["previous_value"] = copy.deepcopy(event.get("previous_value"))
    dependencies = event.get("depends_on")
    if isinstance(dependencies, list) and dependencies:
        candidate["depends_on"] = copy.deepcopy(dependencies)
    if isinstance(event.get("scope"), dict) and event["scope"]:
        candidate["scope"] = copy.deepcopy(event["scope"])
    if isinstance(event.get("set_membership"), dict):
        candidate["set_membership"] = copy.deepcopy(event["set_membership"])
    return candidate if not semantic_graph_v2.validate_event(candidate) else None


def extraction_request_for_row(
    row: dict[str, Any],
    *,
    allow_dialogue_fallback: bool = False,
) -> dict[str, Any] | None:
    turns = dialogue_turns(row)
    by_turn_id = {
        normalize_space(turn.get("turn_id") or turn.get("id")): turn
        for turn in turns
        if normalize_space(turn.get("turn_id") or turn.get("id"))
    }
    graph = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    sample_id = normalize_space(row.get("sample_id") or row.get("source_id")) or (
        f"stage2_source_line_{int(row.get('_source_line_number') or 0):04d}"
    )
    requests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    source_events = graph.get("events") or []
    for event_order, event in enumerate(source_events, start=1):
        if not isinstance(event, dict):
            continue
        source_event_id = normalize_space(
            event.get("event_instance_id") or event.get("event_id")
        ) or f"source_event_{event_order:03d}"
        ids = event_turn_ids(event)
        evidence = [by_turn_id.get(turn_id) for turn_id in ids]
        if not ids or any(not isinstance(turn, dict) for turn in evidence):
            skipped.append(
                {
                    "source_event_id": source_event_id,
                    "source_event_order": event_order,
                    "reason": "missing_evidence_turn",
                }
            )
            continue
        if any(turn_has_legacy_qa(turn) for turn in evidence if isinstance(turn, dict)):
            skipped.append(
                {
                    "source_event_id": source_event_id,
                    "source_event_order": event_order,
                    "reason": "legacy_qa_evidence_turn",
                }
            )
            continue
        if any(
            build_stage3_parent_v2.known_deterministic_padding_turn(turn)
            for turn in evidence
            if isinstance(turn, dict)
        ):
            skipped.append(
                {
                    "source_event_id": source_event_id,
                    "source_event_order": event_order,
                    "reason": "deterministic_padding_evidence_turn",
                }
            )
            continue
        clean_evidence_texts = [
            normalize_space(turn.get("text"))
            for turn in evidence
            if isinstance(turn, dict)
        ]
        if any(NEGATIVE_ABSENCE_RE.search(text) for text in clean_evidence_texts):
            skipped.append(
                {
                    "source_event_id": source_event_id,
                    "source_event_order": event_order,
                    "reason": "negative_absence_evidence_turn",
                }
            )
            continue
        derived_status, derived_status_span, detected_statuses = (
            derive_status_and_span(clean_evidence_texts)
        )
        if derived_status is None:
            skipped.append(
                {
                    "source_event_id": source_event_id,
                    "source_event_order": event_order,
                    "reason": "ambiguous_status_evidence_turn",
                    "detected_statuses": sorted(detected_statuses),
                }
            )
            continue
        nonstatus_fact_surface = " ".join(
            fact_surface_without_status_sentence(text, derived_status_span)
            for text in clean_evidence_texts
        )
        if derived_status in {"canceled", "delayed_not_active"}:
            if (
                not PROSPECTIVE_FACT_RE.search(nonstatus_fact_surface)
                and not direct_nonactive_status_binding(
                    derived_status,
                    derived_status_span,
                    clean_evidence_texts,
                )
            ):
                skipped.append(
                    {
                        "source_event_id": source_event_id,
                        "source_event_order": event_order,
                        "reason": (
                            "nonprospective_canceled_or_delayed_evidence"
                        ),
                    }
                )
                continue
        requests.append(
            {
                "source_event_id": source_event_id,
                "source_event_order": event_order,
                "reconstruction_source": "source_event_evidence_link",
                "evidence": [
                    {
                        "turn_id": turn_id,
                        "speaker": normalize_space(turn.get("speaker") or "Speaker"),
                        "text": normalize_space(turn.get("text")),
                    }
                    for turn_id, turn in zip(ids, evidence)
                    if isinstance(turn, dict)
                ],
            }
        )
    fallback_event_count = 0
    if allow_dialogue_fallback and len(requests) < 2:
        used_turn_ids = {
            normalize_space(item.get("turn_id"))
            for request in requests
            for item in request.get("evidence") or []
            if isinstance(item, dict) and normalize_space(item.get("turn_id"))
        }
        fallback_candidates = [
            turn
            for turn in turns
            if normalize_space(turn.get("turn_id") or turn.get("id"))
            not in used_turn_ids
            and not turn_has_legacy_qa(turn)
            and not build_stage3_parent_v2.known_deterministic_padding_turn(turn)
            and "?" not in normalize_space(turn.get("text"))
            and len(normalize_space(turn.get("text"))) >= 40
            and FALLBACK_FIRST_PERSON_RE.search(normalize_space(turn.get("text")))
            and FALLBACK_FACT_PREDICATE_RE.search(normalize_space(turn.get("text")))
            and not FALLBACK_UNCERTAIN_RE.search(normalize_space(turn.get("text")))
        ]
        for fallback_index, turn in enumerate(
            evenly_select(fallback_candidates, max(0, 3 - len(requests))),
            start=1,
        ):
            turn_id = normalize_space(turn.get("turn_id") or turn.get("id"))
            requests.append(
                {
                    "source_event_id": (
                        f"fallback:{sha256_json([sample_id, turn_id])[:16]}"
                    ),
                    "source_event_order": len(source_events) + fallback_index,
                    "reconstruction_source": "clean_dialogue_fallback",
                    "evidence": [
                        {
                            "turn_id": turn_id,
                            "speaker": normalize_space(
                                turn.get("speaker") or "Speaker"
                            ),
                            "text": normalize_space(turn.get("text")),
                        }
                    ],
                }
            )
            fallback_event_count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": f"stage2_graph_extract_{sha256_json([sample_id, requests])[:20]}",
        "source_sample_id": sample_id,
        "source_line_number": int(row.get("_source_line_number") or 0),
        "source_graph_snapshot_hash": normalize_space(graph.get("graph_snapshot_hash")),
        "source_event_count": sum(
            1 for event in source_events if isinstance(event, dict)
        ),
        "fallback_event_count": fallback_event_count,
        "events": requests,
        "skipped_events": skipped,
    }


def evenly_select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or len(rows) <= count:
        return rows
    if count == 1:
        return [rows[0]]
    indexes = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in dict.fromkeys(indexes)]


def build_prompt(request: dict[str, Any]) -> str:
    specification = {
        "task": "extract canonical semantic graph events from evidence turns",
        "requirements": [
            "Return exactly one JSON object with exactly one key named events.",
            "Return exactly one output event for every input source_event_id, in the same order.",
            "Infer each event independently from its evidence only. No source entity, relation, value, status, question, answer, option, or label is provided.",
            "Choose entity as an exact named subject span copied from the evidence text. For a first-person statement only, use the exact evidence speaker name.",
            "Choose value as one exact contiguous, non-question fact span copied from the evidence text. Never infer a more precise date, name, quantity, or answer than the evidence states.",
            "Choose a concrete lower_snake_case semantic slot describing the relation. The first non-type slot token must reuse an evidence predicate word or its direct inflectional lemma (for example donated_item_date for donated, adopted_pet_date for adopted, or playing_game for playing). Do not replace the evidence predicate with an unsupported synonym.",
            "Preserve the outermost modality in the slot. For example, 'is considering adopting' must use a slot beginning considering_, and 'plans to go' must use a slot beginning plans_to_; never collapse a plan into a completed action.",
            "For got/received clauses, choose the direct object as value rather than a beneficiary or purpose after 'for'. For '<subject> had <number> <items>', choose the number phrase as value and a count slot.",
            "For a claim explicitly corrected as false, retain the destination, recipient, date, or other qualifier that distinguishes the false proposition; do not reduce it to an item shared with a true claim.",
            "Copy supporting_span as one exact contiguous evidence span that directly binds entity to value; do not choose an unrelated sentence merely containing the words.",
            "Do not return status, time_index, evidence_turn_ids, dependencies, or provenance. The validator derives those fields deterministically outside the model.",
            "If the evidence cannot support all required fields, set supported=false and explain rejection_reason. Do not guess.",
            "Never emit legacy question/answer wrappers, a question, options, labels, gold, graph commentary, or facts absent from the evidence.",
        ],
        "input_events": request["events"],
        "output_schema": {
            "events": [
                {
                    "source_event_id": "exact input id",
                    "entity": "exact named evidence subject or exact first-person speaker",
                    "slot": "concrete lower_snake_case semantic relation",
                    "value": "exact contiguous evidence span",
                    "supporting_span": "exact contiguous evidence span",
                    "supported": True,
                    "rejection_reason": "empty when supported=true",
                }
            ]
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a conservative information-extraction program. "
                "Use only quoted evidence, never answer questions, and return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(specification, ensure_ascii=False)},
    ]
    return json.dumps(messages, ensure_ascii=False)


def render_chat_prompt(tokenizer: Any, request: dict[str, Any]) -> str:
    messages = json.loads(build_prompt(request))
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def parse_first_atomic_event(
    text: str,
    expected_source_event_id: str,
) -> dict[str, Any] | None:
    """Recover only the first complete event from an over-generating atomic reply."""

    cleaned = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    match = re.search(r'"events"\s*:\s*\[', cleaned)
    if not match:
        return None
    start = match.end()
    while start < len(cleaned) and cleaned[start].isspace():
        start += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned, start)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if normalize_space(value.get("source_event_id")) != normalize_space(
        expected_source_event_id
    ):
        return None
    return value


def exact_evidence_span(value: Any, evidence_texts: list[str]) -> bool:
    candidate = normalize_space(value)
    return bool(
        candidate
        and any(candidate.casefold() in text.casefold() for text in evidence_texts)
    )


def evidence_indexes_for_span(value: Any, evidence_texts: list[str]) -> set[int]:
    candidate = normalize_space(value)
    if not candidate:
        return set()
    return {
        index
        for index, text in enumerate(evidence_texts)
        if candidate.casefold() in text.casefold()
    }


def canonical_exact_evidence_span(
    value: Any,
    evidence_texts: list[str],
) -> str:
    candidate = normalize_space(value)
    if not candidate:
        return ""
    variants = [candidate]
    if candidate[-1:] in ".,;:!?":
        variants.append(candidate[:-1].rstrip())
    for variant in variants:
        for text in evidence_texts:
            match = re.search(re.escape(variant), text, re.IGNORECASE)
            if match:
                return text[match.start() : match.end()]
    return ""


def extract_exact_date_span(texts: Iterable[str]) -> str:
    for text in texts:
        for pattern in DATE_SPAN_PATTERNS:
            match = pattern.search(text)
            if match:
                return text[match.start() : match.end()]
    return ""


def evidence_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalize_space(text))
        if sentence.strip()
    ]


def derive_status_and_span(
    evidence_texts: list[str],
) -> tuple[str | None, str, set[str]]:
    detected = evidence_nonactive_statuses(evidence_texts)
    if detected == {"outdated", "expired"}:
        # "Outdated ... expired" is one compatible terminal invalidation,
        # not two competing truths. Use the stronger final state.
        for text in evidence_texts:
            matches = [
                match
                for pattern in STATUS_EVIDENCE_PATTERNS["expired"]
                for match in pattern.finditer(text)
            ]
            if matches:
                match = matches[-1]
                return "expired", text[match.start() : match.end()], detected
    if len(detected) > 1:
        return None, "", detected
    if not detected:
        return "active", "", set()
    status = next(iter(detected))
    for text in evidence_texts:
        for pattern in STATUS_EVIDENCE_PATTERNS[status]:
            match = pattern.search(text)
            if match:
                return status, text[match.start() : match.end()], detected
    return None, "", detected


def fact_surface_without_status_sentence(text: str, status_span: str) -> str:
    if not status_span:
        return normalize_space(text)
    retained = [
        sentence
        for sentence in evidence_sentences(text)
        if status_span.casefold() not in sentence.casefold()
    ]
    return normalize_space(" ".join(retained))


def direct_nonactive_status_binding(
    status: str,
    status_span: str,
    evidence_texts: list[str],
) -> bool:
    """Accept a terminal status only when it directly modifies a named fact."""

    if status not in {"canceled", "delayed_not_active"} or not status_span:
        return False
    for text in evidence_texts:
        for sentence in evidence_sentences(text):
            if status_span.casefold() not in sentence.casefold():
                continue
            if re.match(r"^(?:That|This|It)\b", sentence, re.I):
                continue
            if status == "canceled" and re.search(
                r"\b(?:was|were|is|are|has\s+been|have\s+been)\s+"
                r"(?:cancel\w*|called\s+off)\b",
                sentence,
                re.I,
            ):
                return True
            if status == "delayed_not_active" and re.search(
                r"\b(?:was|were|is|are|has\s+been|have\s+been)\s+"
                r"delay\w*\b",
                sentence,
                re.I,
            ):
                return True
    return False


def derive_fact_supporting_span(
    entity: str,
    value: str,
    evidence: list[dict[str, Any]],
) -> str:
    for item in evidence:
        text = normalize_space(item.get("text"))
        speaker = normalize_space(item.get("speaker"))
        for sentence in evidence_sentences(text):
            value_present = phrase_in_text(value, sentence)
            entity_present = phrase_in_text(entity, sentence)
            first_person = (
                speaker.casefold() == entity.casefold()
                and FALLBACK_FIRST_PERSON_RE.search(sentence) is not None
            )
            if value_present and (entity_present or first_person):
                return sentence
    return ""


def evidence_nonactive_statuses(evidence_texts: list[str]) -> set[str]:
    joined = "\n".join(evidence_texts)
    return {
        status
        for status, patterns in STATUS_EVIDENCE_PATTERNS.items()
        if any(pattern.search(joined) for pattern in patterns)
    }


def status_supported(status: str, span: str, evidence_texts: list[str]) -> bool:
    detected = evidence_nonactive_statuses(evidence_texts)
    if len(detected) > 1:
        return False
    expected = next(iter(detected), "active")
    if status != expected:
        return False
    if status == "active":
        return not normalize_space(span)
    if not exact_evidence_span(span, evidence_texts):
        return False
    return any(pattern.search(span) for pattern in STATUS_EVIDENCE_PATTERNS[status])


def value_contains_status_language(value: Any, status: str) -> bool:
    text = normalize_space(value)
    return any(pattern.search(text) for pattern in STATUS_EVIDENCE_PATTERNS.get(status, ()))


def first_person_entity_binding(
    entity: str,
    supporting_span: str,
    evidence: list[dict[str, Any]],
) -> bool:
    for item in evidence:
        if normalize_space(item.get("speaker")) != entity:
            continue
        text = normalize_space(item.get("text"))
        if supporting_span.casefold() not in text.casefold():
            continue
        if re.search(r"\b(?:I|I'm|I've|I'd|me|my|mine)\b", supporting_span, re.I):
            return True
    return False


def named_entity_starts_fact_span(entity: str, supporting_span: str) -> bool:
    if not entity or not supporting_span:
        return False
    return bool(
        re.search(
            rf"(?:^|[.;!?]\s+|\b(?:and|but|while)\s+)"
            rf"(?:(?:according\s+to|regarding)\s+)?"
            rf"{re.escape(entity)}(?:'s|’s|\b)",
            supporting_span,
            re.IGNORECASE,
        )
    )


def derive_named_subject_from_value_sentence(
    value: str,
    evidence: list[dict[str, Any]],
) -> str:
    for item in evidence:
        text = normalize_space(item.get("text"))
        speaker = normalize_space(item.get("speaker"))
        for sentence in evidence_sentences(text):
            if not phrase_in_text(value, sentence):
                continue
            if FALLBACK_FIRST_PERSON_RE.search(sentence) and speaker:
                return speaker
            match = re.match(
                r"^(?:So\s+|Well\s+|Actually\s+)?([A-Z][A-Za-z'-]{1,39})",
                sentence,
            )
            if not match:
                continue
            subject = re.sub(r"(?:'s|’s)$", "", match.group(1))
            if subject.casefold() not in {"a", "an", "that", "the", "this"}:
                return subject
    return ""


def evidence_native_slot_repair(
    slot: str,
    supporting_span: str,
    entity: str,
) -> str:
    if not supporting_span:
        return slot
    tail = supporting_span
    entity_match = re.search(re.escape(entity), tail, re.IGNORECASE) if entity else None
    if entity_match:
        tail = tail[entity_match.end() :]
    raw_tokens = re.findall(r"[A-Za-z]+(?:n't|'re|'ve|'m|'d|'s)?", tail)
    auxiliary = {
        "a", "an", "and", "are", "at", "be", "been", "being", "for",
        "from", "had", "has", "have", "her", "his", "in", "is", "its",
        "of", "on", "s", "the", "their", "to", "was", "were", "with",
    }
    predicate = ""
    for token in raw_tokens:
        lowered = token.casefold()
        if lowered in auxiliary:
            continue
        if lowered.endswith("n't"):
            lowered = f"{lowered[:-3]}_not"
        predicate = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
        if predicate:
            break
    if not predicate:
        return slot
    evidence_tokens = set(normalize_match(supporting_span).split())
    suffix_tokens = [
        token
        for token in slot.split("_")[1:]
        if token
        and (
            token in SLOT_TYPE_TOKENS
            or token not in GENERIC_SLOTS
            and light_stem(token)
            in {light_stem(item) for item in evidence_tokens}
        )
    ]
    repaired = "_".join([predicate, *suffix_tokens])
    return repaired if SLOT_RE.fullmatch(repaired) else slot


def outer_modal_slot_repair(
    slot: str,
    supporting_span: str,
    value: str,
) -> tuple[str, str]:
    """Preserve an outer plan/consideration predicate before an inner action."""

    value_match = (
        re.search(re.escape(value), supporting_span, re.IGNORECASE)
        if value
        else None
    )
    if not value_match:
        return slot, ""
    slot_tokens = set(normalize_match(slot).split())
    for canonical_cue, pattern in OUTER_MODAL_SLOT_PATTERNS:
        cue_match = pattern.search(supporting_span)
        if not cue_match or cue_match.start() >= value_match.start():
            continue
        required_tokens = set(normalize_match(canonical_cue).split())
        if required_tokens <= slot_tokens:
            return slot, ""
        return canonical_cue, "local_outer_modal_slot"
    return slot, ""


def _fact_phrase_slot(fact_phrase: str) -> str:
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", fact_phrase)
    ]
    retained: list[str] = []
    for token in tokens:
        if token in {
            "a", "an", "the", "in", "on", "to", "with", "where", "for",
            "from", "due", "but", "of", "at", "during", "is", "was",
        }:
            if retained:
                break
            continue
        retained.append(re.sub(r"[^a-z0-9]+", "_", token).strip("_"))
        if len(retained) >= 2:
            break
    slot = "_".join(token for token in retained if token)
    return slot if SLOT_RE.fullmatch(slot) and slot not in GENERIC_SLOTS else ""


def deterministic_terminal_extraction(
    source: dict[str, Any],
) -> dict[str, Any] | None:
    """Recover explicit named terminal facts without trusting model semantics."""

    evidence = [
        item
        for item in source.get("evidence") or []
        if isinstance(item, dict)
    ]
    evidence_texts = [normalize_space(item.get("text")) for item in evidence]
    status, _status_span, _detected = derive_status_and_span(evidence_texts)
    if status not in {"canceled", "delayed_not_active", "expired", "outdated"}:
        return None
    status_tail = re.compile(
        r"\s+(?:is|are|was|were|has|have|has\s+been|have\s+been)\s+(?:now\s+)?"
        r"(?:outdated(?:\s+or\s+expired)?|expired|cancel\w*|called\s+off|"
        r"delayed(?:_not_active)?)(?:\s+due\s+to\s+.+)?\s*$",
        re.I,
    )
    for text in evidence_texts:
        sentences = evidence_sentences(text)
        for sentence_index, sentence in enumerate(sentences):
            candidate = sentence.rstrip(". ")
            referenced_fact_clause = False
            if re.match(r"^(?:That|This|It)\b", candidate, re.I):
                if sentence_index == 0:
                    continue
                previous_sentence = sentences[sentence_index - 1].rstrip(". ")
                if (
                    evidence_nonactive_statuses([previous_sentence])
                    and not status_tail.search(previous_sentence)
                ):
                    continue
                candidate = previous_sentence
            if ", but " in candidate:
                left, right = candidate.rsplit(", but ", 1)
                right_is_status_reference = bool(
                    status_tail.search(right)
                    and re.match(r"^(?:That|This|It)\b", right, re.I)
                )
                candidate = (
                    left
                    if right_is_status_reference
                    else right if status_tail.search(right) else left
                )
                referenced_fact_clause = right_is_status_reference
                owner_match = re.match(r"^([A-Z][A-Za-z'-]+)'s\b", left)
                inherited_owner = owner_match.group(1) if owner_match else ""
            else:
                inherited_owner = ""

            tail_match = status_tail.search(candidate)
            if not tail_match and not referenced_fact_clause:
                continue
            fact_surface = (
                normalize_space(candidate[: tail_match.start()])
                if tail_match
                else normalize_space(candidate)
            )
            possessive = re.match(
                r"^([A-Z][A-Za-z'-]+)'s\s+(.+)$",
                fact_surface,
            )
            if possessive:
                entity = possessive.group(1)
                fact_phrase = normalize_space(possessive.group(2))
                slot = _fact_phrase_slot(fact_phrase)
                if slot and canonical_exact_evidence_span(fact_phrase, [text]):
                    return {
                        "source_event_id": normalize_space(source.get("source_event_id")),
                        "entity": entity,
                        "slot": slot,
                        "value": fact_phrase,
                        "supporting_span": sentence,
                        "supported": True,
                        "_deterministic_terminal_recovery": True,
                    }

            pronoun_fact = re.match(r"^(?:his|her|their)\s+(.+)$", fact_surface, re.I)
            if inherited_owner and pronoun_fact:
                fact_phrase = normalize_space(pronoun_fact.group(1))
                slot = _fact_phrase_slot(fact_phrase)
                if slot and canonical_exact_evidence_span(fact_phrase, [text]):
                    return {
                        "source_event_id": normalize_space(source.get("source_event_id")),
                        "entity": inherited_owner,
                        "slot": slot,
                        "value": fact_phrase,
                        "supporting_span": sentence,
                        "supported": True,
                        "_deterministic_terminal_recovery": True,
                    }

            named_clause = re.match(
                r"^([A-Z][A-Za-z'-]+)\s+([A-Za-z][A-Za-z'-]*(?:\s+(?:down|up|out|off))?)\s+(.+)$",
                fact_surface,
            )
            if named_clause:
                entity = named_clause.group(1)
                predicate = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    named_clause.group(2).casefold(),
                ).strip("_")
                value = normalize_space(named_clause.group(3))
                if (
                    SLOT_RE.fullmatch(predicate)
                    and canonical_exact_evidence_span(value, [text])
                ):
                    return {
                        "source_event_id": normalize_space(source.get("source_event_id")),
                        "entity": entity,
                        "slot": predicate,
                        "value": value,
                        "supporting_span": sentence,
                        "supported": True,
                        "_deterministic_terminal_recovery": True,
                    }
    return None


def light_stem(token: str) -> str:
    irregular = {
        "attendees": "attendee",
        "named": "name",
        "said": "say",
        "relies": "rely",
        "reliance": "rely",
        "went": "go",
        "tried": "try",
    }
    value = irregular.get(token, token)
    for suffix in ("ingly", "edly", "ing", "ied", "ed", "es", "s"):
        if len(value) >= len(suffix) + 3 and value.endswith(suffix):
            value = value[: -len(suffix)]
            if suffix == "ied":
                value += "y"
            if len(value) >= 2 and value[-1] == value[-2]:
                value = value[:-1]
            break
    return value


def slot_relation_supported(slot: str, relation_supporting_span: str, value: str) -> bool:
    all_slot_tokens = [
        token
        for token in slot.split("_")
        if token and token not in SLOT_STOPWORDS
    ]
    slot_tokens = [
        token for token in all_slot_tokens if token not in SLOT_TYPE_TOKENS
    ]
    evidence_tokens = set(normalize_match(relation_supporting_span).split())
    evidence_stems = {light_stem(token) for token in evidence_tokens}
    if not slot_tokens:
        return False

    def related(token: str) -> bool:
        if token in evidence_tokens or light_stem(token) in evidence_stems:
            return True
        aliases = SLOT_CUE_ALIASES.get(token, set())
        return bool(
            aliases & evidence_tokens
            or {light_stem(alias) for alias in aliases} & evidence_stems
        )

    if not related(slot_tokens[0]):
        return False
    if "date" in all_slot_tokens and (
        not DATE_VALUE_RE.search(value) or len(normalize_match(value).split()) > 7
    ):
        return False
    if "duration" in all_slot_tokens and not DURATION_VALUE_RE.search(value):
        return False
    if "frequency" in all_slot_tokens and not FREQUENCY_VALUE_RE.search(value):
        return False
    if {"count", "number", "age", "quantity"} & set(all_slot_tokens) and not re.search(
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        value,
        re.I,
    ):
        return False
    return True


def validate_extracted_event(
    source: dict[str, Any],
    extracted: dict[str, Any],
    *,
    request_id: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    evidence = source.get("evidence") if isinstance(source.get("evidence"), list) else []
    evidence_texts = [normalize_space(item.get("text")) for item in evidence if isinstance(item, dict)]
    expected_turn_ids = [normalize_space(item.get("turn_id")) for item in evidence if isinstance(item, dict)]
    expected_id = normalize_space(source.get("source_event_id"))
    model_slot_output = normalize_space(extracted.get("slot"))
    model_entity_output = normalize_space(extracted.get("entity"))
    model_supported_output = extracted.get("supported") is True
    terminal_recovery = deterministic_terminal_extraction(source)
    if terminal_recovery:
        extracted = {**extracted, **terminal_recovery}
    if normalize_space(extracted.get("source_event_id")) != expected_id:
        errors.append("source_event_id_mismatch")

    original_slot = normalize_space(extracted.get("slot"))
    slot = original_slot
    raw_entity = normalize_space(extracted.get("entity"))
    model_supporting_span = canonical_exact_evidence_span(
        extracted.get("supporting_span"), evidence_texts
    )
    status, status_span, detected_statuses = derive_status_and_span(evidence_texts)

    raw_value = extracted.get("value")
    relation_repair_source = ""
    if terminal_recovery:
        relation_repair_source = "local_terminal_fact_clause"
    if model_supporting_span:
        counted_clause = re.match(
            r"^(.+?)\s+(?:had|has)\s+"
            r"((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"([A-Za-z][A-Za-z'-]*))\b",
            model_supporting_span,
            re.I,
        )
        direct_object_clause = re.match(
            r"^(.+?)\s+(got|received)\s+(.+?)\s+for\s+"
            r"(?:his|her|their|the)\s+.+",
            model_supporting_span,
            re.I,
        )
        false_destination_clause = (
            re.match(
                r"^(.+?)\s+([A-Za-z][A-Za-z'-]*)\s+.+?\s+to\s+(.+?)(?:[.]|$)",
                model_supporting_span,
                re.I,
            )
            if status == "false"
            else None
        )
        if counted_clause:
            raw_entity = normalize_space(counted_clause.group(1))
            noun = re.sub(r"s$", "", counted_clause.group(3).casefold())
            slot = f"{noun}_count"
            raw_value = normalize_space(counted_clause.group(2))
            relation_repair_source = "local_exact_counted_clause"
        elif direct_object_clause:
            raw_entity = normalize_space(direct_object_clause.group(1))
            verb = direct_object_clause.group(2).casefold()
            slot = f"{verb}_item"
            raw_value = normalize_space(direct_object_clause.group(3))
            relation_repair_source = "local_exact_direct_object_clause"
        elif false_destination_clause:
            raw_entity = normalize_space(false_destination_clause.group(1))
            predicate = false_destination_clause.group(2).casefold()
            slot = f"{predicate}_to"
            raw_value = normalize_space(false_destination_clause.group(3))
            relation_repair_source = "local_false_destination_qualifier"
    value = (
        canonical_exact_evidence_span(raw_value, evidence_texts)
        if isinstance(raw_value, str)
        else ""
    )
    value_repair_source = relation_repair_source or (
        "model_exact_evidence_span" if value else ""
    )
    if "date" in original_slot.split("_") and (
        not value
        or not DATE_VALUE_RE.search(value)
        or len(normalize_match(value).split()) > 7
    ):
        repaired_date = extract_exact_date_span(
            [model_supporting_span, *evidence_texts]
        )
        if repaired_date:
            value = repaired_date
            value_repair_source = "local_exact_typed_date_span"
    if not value and isinstance(raw_value, str) and raw_entity:
        possessive = re.match(r"^(?:his|her|their)\s+(.+)$", raw_value, re.I)
        if possessive:
            candidate = f"{raw_entity}'s {possessive.group(1)}"
            repaired_possessive = canonical_exact_evidence_span(
                candidate, evidence_texts
            )
            if repaired_possessive:
                value = repaired_possessive
                value_repair_source = "local_possessive_coreference_exact_span"
    if not value and isinstance(raw_value, str) and model_supporting_span:
        raw_tokens = normalize_match(raw_value).split()
        support_matches = list(
            re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", model_supporting_span)
        )
        support_tokens = [
            normalize_match(match.group(0)) for match in support_matches
        ]
        if raw_tokens:
            for start in range(0, len(support_tokens) - len(raw_tokens) + 1):
                candidate_tokens = support_tokens[
                    start : start + len(raw_tokens)
                ]
                if (
                    light_stem(candidate_tokens[0]) == light_stem(raw_tokens[0])
                    and candidate_tokens[1:] == raw_tokens[1:]
                ):
                    value = model_supporting_span[
                        support_matches[start].start() :
                        support_matches[start + len(raw_tokens) - 1].end()
                    ]
                    value_repair_source = (
                        "local_inflectional_exact_evidence_span"
                    )
                    break
    if not value and model_supporting_span:
        counted_clause = re.match(
            r"^(.+?)\s+(had|has)\s+((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([A-Za-z][A-Za-z'-]*).*)$",
            model_supporting_span,
            re.I,
        )
        if counted_clause:
            raw_entity = normalize_space(counted_clause.group(1))
            value = normalize_space(counted_clause.group(3))
            noun = re.sub(r"s$", "", counted_clause.group(4).casefold())
            original_slot = f"{counted_clause.group(2).casefold()}_{noun}_count"
            slot = original_slot
            value_repair_source = "local_exact_counted_clause"
    relation_inversion_repair = False
    if (
        value
        and raw_entity
        and normalize_match(value) == normalize_match(raw_entity)
        and model_supporting_span
    ):
        name_clause = re.match(
            r"^(.+?)\s+is\s+the\s+name\s+of\s+(.+?)(?:,|$)",
            model_supporting_span,
            re.I,
        )
        if name_clause and normalize_match(name_clause.group(1)) == normalize_match(
            value
        ):
            value = normalize_space(name_clause.group(1))
            raw_entity = normalize_space(name_clause.group(2))
            original_slot = "named"
            slot = original_slot
            value_repair_source = "local_name_relation_inversion"
            relation_inversion_repair = True
    if not value:
        errors.append("value_not_exact_evidence_span")
    if value and (LEGACY_QA_RE.search(value) or "?" in value):
        errors.append("legacy_qa_value")

    entity = canonical_exact_evidence_span(raw_entity, evidence_texts)
    if not entity:
        for item in evidence:
            if not isinstance(item, dict):
                continue
            speaker = normalize_space(item.get("speaker"))
            if speaker and speaker.casefold() == raw_entity.casefold():
                entity = speaker
                break

    supporting_span = (
        derive_fact_supporting_span(
            entity,
            value,
            [item for item in evidence if isinstance(item, dict)],
        )
        if entity and value
        else ""
    )
    if not supporting_span:
        fallback_entity = derive_named_subject_from_value_sentence(
            value,
            [item for item in evidence if isinstance(item, dict)],
        )
        if fallback_entity:
            entity = fallback_entity
            supporting_span = derive_fact_supporting_span(
                entity,
                value,
                [item for item in evidence if isinstance(item, dict)],
            )
    if not entity or normalize_match(entity) in ENTITY_STOPWORDS:
        errors.append("invalid_entity")
    elif not canonical_exact_evidence_span(entity, evidence_texts) and not any(
        normalize_space(item.get("speaker")).casefold() == entity.casefold()
        for item in evidence
        if isinstance(item, dict)
    ):
        errors.append("entity_not_grounded_in_evidence")
    if not supporting_span:
        errors.append("entity_value_fact_sentence_not_found")
    first_person_bound = first_person_entity_binding(
        entity,
        supporting_span,
        [item for item in evidence if isinstance(item, dict)],
    )
    if entity and supporting_span and not first_person_bound and not relation_inversion_repair and not named_entity_starts_fact_span(
        entity, supporting_span
    ):
        errors.append("entity_not_subject_of_supporting_span")
    if entity and value and normalize_match(entity) == normalize_match(value):
        errors.append("entity_equals_value")
    if entity and value and phrase_in_text(value, entity):
        errors.append("value_redundant_within_entity")

    slot, modal_slot_repair_source = outer_modal_slot_repair(
        slot,
        supporting_span,
        value,
    )
    if modal_slot_repair_source:
        relation_repair_source = modal_slot_repair_source

    initial_slot_valid = bool(
        SLOT_RE.fullmatch(slot)
        and slot not in GENERIC_SLOTS
        and "raw_qa" not in slot
    )
    initial_relation_valid = bool(
        initial_slot_valid
        and supporting_span
        and value
        and slot_relation_supported(slot, supporting_span, value)
    )
    slot_repaired = False
    if not initial_relation_valid:
        repaired_slot = evidence_native_slot_repair(
            slot,
            supporting_span,
            entity,
        )
        if (
            repaired_slot != slot
            and repaired_slot not in GENERIC_SLOTS
            and slot_relation_supported(repaired_slot, supporting_span, value)
        ):
            slot = repaired_slot
            slot_repaired = True
    if not SLOT_RE.fullmatch(slot) or slot in GENERIC_SLOTS or "raw_qa" in slot:
        errors.append("invalid_semantic_slot")
    elif supporting_span and value and not slot_relation_supported(
        slot, supporting_span, value
    ):
        errors.append("slot_relation_not_supported")

    if status is None:
        errors.append("ambiguous_evidence_status")
        status = ""
    status_indexes = evidence_indexes_for_span(status_span, evidence_texts)
    if status in NONACTIVE_STATUSES and status_span:
        if value_contains_status_language(value, status):
            errors.append("status_language_used_as_value")
        supporting_indexes = evidence_indexes_for_span(
            supporting_span, evidence_texts
        )
        if supporting_indexes and status_indexes and not (
            supporting_indexes & status_indexes
        ):
            errors.append("fact_and_status_spans_from_different_evidence")
        if value and (
            phrase_in_text(value, status_span) or phrase_in_text(status_span, value)
        ):
            errors.append("value_overlaps_status_supporting_span")
        if status in {"canceled", "delayed_not_active"}:
            if (
                not PROSPECTIVE_FACT_RE.search(supporting_span)
                and not direct_nonactive_status_binding(
                    status,
                    status_span,
                    evidence_texts,
                )
            ):
                errors.append("nonactive_plan_status_without_prospective_fact")
    if errors:
        return None, errors
    event = {
        "event_id": f"stage2_rebuild:{request_id}:{sha256_json(expected_id)[:10]}",
        "entity": entity,
        "slot": slot,
        "value": value,
        "status": status,
        "time_index": int(source.get("source_event_order") or 0),
        "evidence_turn_ids": expected_turn_ids,
        "required_for_dialogue": True,
        "provenance": {
            "migration": SCHEMA_VERSION,
            "source": "local_model_evidence_extraction",
            "source_event_id": expected_id,
            "request_id": request_id,
            "supporting_span": supporting_span,
            "status_supporting_span": status_span,
            "status_source": "deterministic_evidence_cue",
            "time_index_source": "source_event_order",
            "model_supported": model_supported_output,
            "model_entity": model_entity_output,
            "model_slot": model_slot_output,
            "deterministic_terminal_recovery": bool(terminal_recovery),
            "value_repair_source": value_repair_source,
            "relation_repair_source": relation_repair_source or None,
            "slot_repaired_to_evidence_native": slot_repaired,
            "relation_inversion_repair": relation_inversion_repair,
        },
    }
    schema_errors = semantic_graph_v2.validate_event(event)
    if schema_errors:
        return None, [f"semantic_graph_v2:{item['code']}" for item in schema_errors]
    return event, []


def validate_response(
    request: dict[str, Any],
    response: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if not isinstance(response, dict) or set(response) != {"events"} or not isinstance(response.get("events"), list):
        return [], [
            {
                "request_id": request["request_id"],
                "reason": "response_schema_mismatch",
            }
        ]
    outputs = response["events"]
    if len(outputs) != len(request["events"]):
        return [], [
            {
                "request_id": request["request_id"],
                "reason": "response_event_count_mismatch",
                "expected": len(request["events"]),
                "actual": len(outputs),
            }
        ]
    for source, extracted in zip(request["events"], outputs):
        if not isinstance(extracted, dict):
            extracted = deterministic_terminal_extraction(source)
            if not isinstance(extracted, dict):
                rejected.append(
                    {
                        "request_id": request["request_id"],
                        "source_event_id": source.get("source_event_id"),
                        "reason": "extracted_event_not_object",
                    }
                )
                continue
        event, errors = validate_extracted_event(
            source,
            extracted,
            request_id=request["request_id"],
        )
        if event is None:
            rejected.append(
                {
                    "request_id": request["request_id"],
                    "source_event_id": source.get("source_event_id"),
                    "reason": "extracted_event_rejected",
                    "errors": errors,
                    "model_output": copy.deepcopy(extracted),
                }
            )
        else:
            accepted.append(event)
    return accepted, rejected


def link_terminal_dependencies(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Link an explicit terminal fact to one earlier, lexically matching active fact."""

    terminal_statuses = {"canceled", "delayed_not_active", "expired", "outdated"}
    stop_tokens = {
        "a", "an", "and", "are", "at", "for", "from", "has", "have", "her",
        "his", "in", "is", "its", "of", "on", "the", "their", "to", "was",
        "were", "with", "active", "canceled", "delayed", "expired", "outdated",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }
    generic_relation_tokens = {
        "attend", "attended", "doing", "event", "got", "hosted", "joined",
        "made", "received", "shared", "started", "went",
    }

    def tokens(*values: Any) -> set[str]:
        return {
            token
            for token in normalize_match(values).split()
            if token and token not in stop_tokens and not token.isdigit()
        }

    linked = copy.deepcopy(events)
    ordered = sorted(
        linked,
        key=lambda event: (
            int(event.get("time_index") or 0),
            normalize_space(event.get("event_id")),
        ),
    )
    for terminal in ordered:
        if terminal.get("status") not in terminal_statuses:
            continue
        terminal_time = int(terminal.get("time_index") or 0)
        terminal_entity = normalize_match(terminal.get("entity"))
        terminal_entity_tokens = tokens(terminal.get("entity"))
        terminal_fact_tokens = tokens(terminal.get("slot"), terminal.get("value"))
        candidates: list[tuple[int, int, str, dict[str, Any]]] = []
        for candidate in ordered:
            candidate_time = int(candidate.get("time_index") or 0)
            if candidate_time >= terminal_time or candidate.get("status") != "active":
                continue
            candidate_entity = normalize_match(candidate.get("entity"))
            candidate_entity_tokens = tokens(candidate.get("entity"))
            entity_related = bool(
                terminal_entity
                and candidate_entity
                and (
                    terminal_entity == candidate_entity
                    or phrase_in_text(terminal_entity, candidate_entity)
                    or phrase_in_text(candidate_entity, terminal_entity)
                )
            )
            if not entity_related:
                continue
            if (
                terminal_entity != candidate_entity
                and len(terminal_fact_tokens & candidate_entity_tokens) < 2
            ):
                continue
            candidate_fact_tokens = tokens(
                candidate.get("entity"),
                candidate.get("slot"),
                candidate.get("value"),
            )
            overlap = terminal_fact_tokens & candidate_fact_tokens
            slot_overlap = (
                tokens(terminal.get("slot"))
                & tokens(candidate.get("slot"))
            ) - generic_relation_tokens
            if len(overlap) < 2 and not (
                terminal_entity == candidate_entity and bool(slot_overlap)
            ):
                continue
            exact_entity_bonus = 4 if terminal_entity == candidate_entity else 0
            entity_overlap = len(terminal_entity_tokens & candidate_entity_tokens)
            score = len(overlap) * 4 + exact_entity_bonus + entity_overlap
            candidates.append(
                (
                    score,
                    candidate_time,
                    normalize_space(candidate.get("event_id")),
                    candidate,
                )
            )
        if not candidates:
            continue
        _score, _time, dependency_id, _candidate = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        terminal["depends_on"] = [dependency_id]
        provenance = terminal.setdefault("provenance", {})
        provenance["dependency_source"] = (
            "deterministic_prior_active_entity_fact_overlap"
        )
    return linked


def run_local_model(
    requests: list[dict[str, Any]],
    *,
    model_path: Path,
    max_model_len: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    seed: int,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    atomic_requests: list[dict[str, Any]] = []
    for request in requests:
        for event_index, event in enumerate(request.get("events") or []):
            atomic_request = copy.deepcopy(request)
            atomic_request["events"] = [copy.deepcopy(event)]
            atomic_request["parent_request_id"] = request["request_id"]
            atomic_request["atomic_event_index"] = event_index
            atomic_request["request_id"] = (
                f"{request['request_id']}:event:{event_index:03d}"
            )
            atomic_requests.append(atomic_request)
    prompts = [
        render_chat_prompt(tokenizer, request) for request in atomic_requests
    ]
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=min(int(max_new_tokens), 768),
        seed=int(seed),
    )
    outputs = llm.generate(prompts, params)
    grouped: dict[str, list[dict[str, Any]]] = {
        request["request_id"]: [] for request in requests
    }
    for request, output in zip(atomic_requests, outputs):
        text = output.outputs[0].text if output.outputs else ""
        grouped[request["parent_request_id"]].append(
            {
                "atomic_request_id": request["request_id"],
                "source_event_id": request["events"][0].get("source_event_id"),
                "atomic_event_index": request["atomic_event_index"],
                "raw_text": text,
                "parsed": parse_json_object(text),
                "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                "prompt_token_count": len(output.prompt_token_ids or []),
                "completion_token_count": len(output.outputs[0].token_ids or []) if output.outputs else 0,
            }
        )
    records: list[dict[str, Any]] = []
    for request in requests:
        atomic_outputs = sorted(
            grouped[request["request_id"]],
            key=lambda item: int(item["atomic_event_index"]),
        )
        parsed_events: list[dict[str, Any]] = []
        atomic_complete = True
        for item in atomic_outputs:
            parsed = item.get("parsed")
            events = parsed.get("events") if isinstance(parsed, dict) else None
            expected_source_event_id = normalize_space(item.get("source_event_id"))
            first_event = (
                events[0]
                if isinstance(events, list)
                and events
                and isinstance(events[0], dict)
                and normalize_space(events[0].get("source_event_id"))
                == expected_source_event_id
                else parse_first_atomic_event(
                    normalize_space(item.get("raw_text")),
                    expected_source_event_id,
                )
            )
            if not isinstance(first_event, dict):
                atomic_complete = False
                parsed_events.append(None)  # type: ignore[arg-type]
                continue
            item["first_event_recovered"] = not (
                isinstance(parsed, dict)
                and set(parsed) == {"events"}
                and isinstance(events, list)
                and len(events) == 1
            )
            parsed_events.append(first_event)
        records.append(
            {
                "request_id": request["request_id"],
                "atomic_request_count": len(atomic_outputs),
                "atomic_complete": atomic_complete,
                "parsed": {"events": parsed_events},
                "finish_reason": (
                    "stop"
                    if atomic_complete
                    and all(item.get("finish_reason") == "stop" for item in atomic_outputs)
                    else "atomic_incomplete"
                ),
                "prompt_token_count": sum(
                    int(item.get("prompt_token_count") or 0)
                    for item in atomic_outputs
                ),
                "completion_token_count": sum(
                    int(item.get("completion_token_count") or 0)
                    for item in atomic_outputs
                ),
                "atomic_outputs": atomic_outputs,
            }
        )
    return records


def reaggregate_atomic_output_record(record: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a row response from persisted one-event model outputs."""

    rebuilt = copy.deepcopy(record)
    rebuilt.pop("_source_line_number", None)
    atomic_outputs = sorted(
        [
            item
            for item in rebuilt.get("atomic_outputs") or []
            if isinstance(item, dict)
        ],
        key=lambda item: int(item.get("atomic_event_index") or 0),
    )
    if not atomic_outputs:
        return rebuilt
    parsed_events: list[Any] = []
    complete = True
    for item in atomic_outputs:
        expected_id = normalize_space(item.get("source_event_id"))
        parsed = item.get("parsed")
        events = parsed.get("events") if isinstance(parsed, dict) else None
        first_event = (
            events[0]
            if isinstance(events, list)
            and events
            and isinstance(events[0], dict)
            and normalize_space(events[0].get("source_event_id")) == expected_id
            else parse_first_atomic_event(
                normalize_space(item.get("raw_text")), expected_id
            )
        )
        if not isinstance(first_event, dict):
            complete = False
            parsed_events.append(None)
        else:
            parsed_events.append(first_event)
    rebuilt["atomic_complete"] = complete
    rebuilt["parsed"] = {"events": parsed_events}
    rebuilt["finish_reason"] = "stop" if complete else "atomic_incomplete"
    return rebuilt


def remap_reused_atomic_outputs(
    persisted_records: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source_event_id: dict[str, dict[str, Any]] = {}
    for record in persisted_records:
        for item in record.get("atomic_outputs") or []:
            if not isinstance(item, dict):
                continue
            source_event_id = normalize_space(item.get("source_event_id"))
            if source_event_id:
                by_source_event_id[source_event_id] = copy.deepcopy(item)
    remapped: list[dict[str, Any]] = []
    for request in requests:
        atomic_outputs: list[dict[str, Any]] = []
        for event_index, event in enumerate(request.get("events") or []):
            source_event_id = normalize_space(event.get("source_event_id"))
            item = copy.deepcopy(by_source_event_id.get(source_event_id) or {})
            item.update(
                {
                    "source_event_id": source_event_id,
                    "atomic_event_index": event_index,
                }
            )
            atomic_outputs.append(item)
        remapped.append(
            reaggregate_atomic_output_record(
                {
                    "request_id": request["request_id"],
                    "atomic_request_count": len(atomic_outputs),
                    "atomic_outputs": atomic_outputs,
                }
            )
        )
    return remapped


def build_stage3_parent_v3(
    source_row: dict[str, Any],
    canonical_row: dict[str, Any],
    *,
    source_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build an allowlisted parent around one complete reconstructed graph."""

    events = copy.deepcopy(
        (canonical_row.get("semantic_parent_graph") or {}).get("events") or []
    )
    dialogue, turns = build_stage3_parent_v2.dialogue_and_turns(source_row)
    if not turns or len(events) < 2:
        return None, {
            "reason": "missing_dialogue_or_too_few_reconstructed_events",
            "event_count": len(events),
        }
    source_line = int(source_row.get("_source_line_number") or 0)
    source_label = f"{source_path}:{source_line}"
    try:
        dialogue, turns, qa_cleanup = (
            build_stage3_parent_v2.strip_legacy_qa_dialogue_scaffolding(
                dialogue,
                turns,
                events,
                source_label=source_label,
            )
        )
        dialogue, turns, padding_cleanup = (
            build_stage3_parent_v2.strip_historical_deterministic_padding(
                dialogue,
                turns,
                events,
                source_label=source_label,
            )
        )
    except (ValueError, build_stage3_parent_v2.DeterministicPaddingEvidenceReferenceError) as exc:
        return None, {"reason": "dialogue_cleanup_failed", "error": str(exc)}
    retained_turn_ids = {
        normalize_space(turn.get("turn_id") or turn.get("id"))
        for turn in turns
        if normalize_space(turn.get("turn_id") or turn.get("id"))
    }
    dangling = sorted(
        {
            turn_id
            for event in events
            for turn_id in event.get("evidence_turn_ids") or []
            if normalize_space(turn_id) not in retained_turn_ids
        }
    )
    if dangling:
        return None, {
            "reason": "reconstructed_graph_has_dangling_evidence",
            "turn_ids": dangling,
        }
    clean_source_row = {
        key: copy.deepcopy(value)
        for key, value in source_row.items()
        if key != "_source_line_number"
    }
    row_hash = build_stage3_parent_v2.sha256_payload(clean_source_row)
    content_hash = build_stage3_parent_v2.dialogue_content_hash(turns)
    lineage = build_stage3_parent_v2.lineage_group(source_row)
    parent, _, _, _ = build_stage3_parent_v2.build_parent(
        source_row,
        source_path=source_path,
        line_number=source_line,
        dialogue=dialogue,
        turns=turns,
        content_hash=content_hash,
        row_hash=row_hash,
        lineage=lineage,
        forbidden_match_mode="exact",
        forbidden_lineage_overlap=[],
        require_nonempty_semantic_parent_graph=False,
        legacy_qa_dialogue_cleanup=qa_cleanup,
        deterministic_padding_cleanup=padding_cleanup,
    )
    graph_hash = sha256_json(events)
    parent_identity_hash = sha256_json(
        {
            "schema_version": PARENT_SCHEMA_VERSION,
            "source_row_sha256": row_hash,
            "dialogue_content_sha256": content_hash,
            "reconstructed_graph_sha256": graph_hash,
        }
    )
    parent_id = f"stage3_parent_v3_{parent_identity_hash[:20]}"
    source_sample_id = normalize_space(
        source_row.get("sample_id") or source_row.get("source_id")
    )
    parent.update(
        {
            "schema_version": PARENT_SCHEMA_VERSION,
            "sample_id": parent_id,
            "source_id": parent_id,
            "dialogue": dialogue,
            "turns": turns,
        }
    )
    parent_graph_id = normalize_space(
        (canonical_row.get("semantic_parent_graph") or {}).get("parent_graph_id")
    ) or f"stage2_source_snapshot:{row_hash[:20]}"
    parent["semantic_parent_graph"] = {
        "schema_version": semantic_graph_v2.SCHEMA_VERSION,
        "graph_id": f"{parent_id}:semantic_parent_graph",
        "parent_graph_id": parent_graph_id,
        "graph_role": "evidence_reconstructed_stage2_parent_snapshot",
        "events": events,
        "event_count": len(events),
        "empty_parent_snapshot_allowed": False,
        "reconstruction_schema_version": SCHEMA_VERSION,
        "reconstruction_complete_for_eligible_evidence": True,
        "graph_sha256": graph_hash,
    }
    parent["source_snapshot"].update(
        {
            "source_sample_id": source_sample_id,
            "parent_identity_sha256": parent_identity_hash,
            "reconstructed_graph_sha256": graph_hash,
        }
    )
    parent["provenance"]["migration"] = PARENT_SCHEMA_VERSION
    parent["provenance"]["canonical_graph_rebuild"] = {
        "schema_version": SCHEMA_VERSION,
        "request_id": canonical_row.get("request_id"),
        "legacy_semantic_labels_exposed_to_model": False,
        "old_graph_events_inherited": False,
    }
    parent["seed_source"].update(
        {
            "parent_source": PARENT_SCHEMA_VERSION,
            "sample_id": parent_id,
            "parent_sample_id": source_sample_id,
        }
    )
    parent["metadata"].update(
        {
            "parent_schema_version": PARENT_SCHEMA_VERSION,
            "semantic_graph_schema_version": semantic_graph_v2.SCHEMA_VERSION,
            "source_sample_id": source_sample_id,
            "retained_semantic_event_count": len(events),
            "source_semantic_event_count": canonical_row.get("source_event_count"),
            "removed_legacy_or_qa_graph_event_count": canonical_row.get(
                "excluded_source_event_count"
            ),
            "removed_noncanonical_graph_event_count": 0,
            "require_nonempty_semantic_parent_graph": True,
            "canonical_graph_rebuild": copy.deepcopy(canonical_row.get("coverage") or {}),
            "old_graph_events_inherited": False,
        }
    )
    audit = {
        "reason": "ok",
        "source_sample_id": source_sample_id,
        "parent_sample_id": parent_id,
        "event_count": len(events),
        "dialogue_turn_count": len(turns),
        "legacy_qa_turns_removed": int(qa_cleanup.get("removed_turn_count") or 0),
        "deterministic_padding_turns_removed": int(
            padding_cleanup.get("removed_turn_count") or 0
        ),
        "graph_sha256": graph_hash,
    }
    return parent, audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild evidence-grounded Stage2 semantic parent graphs with a local model."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Evenly select this many eligible rows; 0 means all rows.",
    )
    parser.add_argument("--minimum-events-per-parent", type=int, default=2)
    parser.add_argument(
        "--source-line-number",
        type=int,
        action="append",
        default=[],
        help="Restrict reconstruction to these 1-based source JSONL lines.",
    )
    parser.add_argument(
        "--allow-dialogue-fallback",
        action="store_true",
        help="Compatibility-only: synthesize requests from unlinked dialogue turns. Disabled by default.",
    )
    parser.add_argument(
        "--semantic-exclusion-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Independent reviewed semantic-event quarantine manifest; repeatable.",
    )
    parser.add_argument(
        "--forbidden-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Heldout JSONL checked by exact sample/dialogue identity before parent output.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--reuse-raw-model-outputs",
        type=Path,
        action="append",
        default=[],
        help="Revalidate persisted atomic outputs; repeat to overlay later correction runs.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write evidence-only extraction requests without loading the model.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    rows = read_jsonl(input_path)
    manifests = [
        extraction_request_for_row(
            row,
            allow_dialogue_fallback=bool(args.allow_dialogue_fallback),
        )
        for row in rows
    ]
    manifest_rows = [
        request
        for request in manifests
        if isinstance(request, dict)
    ]
    semantic_exclusions = load_semantic_audit_exclusions(
        args.semantic_exclusion_jsonl
    )
    applied_semantic_exclusions = apply_semantic_audit_exclusions(
        manifest_rows,
        semantic_exclusions,
    )
    requested_source_lines = {int(item) for item in args.source_line_number}
    if requested_source_lines:
        available_source_lines = {
            int(request.get("source_line_number") or 0)
            for request in manifest_rows
        }
        missing_source_lines = sorted(
            requested_source_lines - available_source_lines
        )
        if missing_source_lines:
            raise RuntimeError(
                f"Requested source lines are absent: {missing_source_lines}"
            )
        manifest_rows = [
            request
            for request in manifest_rows
            if int(request.get("source_line_number") or 0)
            in requested_source_lines
        ]
    request_rows = [request for request in manifest_rows if request.get("events")]
    requests = evenly_select(manifest_rows, int(args.max_rows))
    selected_line_numbers = {
        int(request.get("source_line_number") or 0) for request in requests
    }
    source_by_line = {
        int(row.get("_source_line_number") or 0): row for row in rows
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "extraction_requests.jsonl", requests)
    selected_request_ids = {
        normalize_space(request.get("request_id")) for request in requests
    }
    selected_semantic_exclusions = [
        record
        for record in applied_semantic_exclusions
        if normalize_space(record.get("request_id")) in selected_request_ids
    ]
    write_jsonl(
        output_dir / "semantic_audit_exclusions_applied.jsonl",
        selected_semantic_exclusions,
    )
    excluded_source_events = [
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.get("request_id"),
            "source_sample_id": request.get("source_sample_id"),
            "source_line_number": request.get("source_line_number"),
            **copy.deepcopy(skipped),
            "disposition": "quarantined_before_model",
        }
        for request in requests
        for skipped in request.get("skipped_events") or []
        if isinstance(skipped, dict)
    ]
    write_jsonl(output_dir / "source_event_exclusions.jsonl", excluded_source_events)
    selected_event_count = sum(len(request["events"]) for request in requests)
    fallback_event_count = sum(
        int(request.get("fallback_event_count") or 0) for request in requests
    )
    request_forbidden_keys = {
        "entity_hint",
        "status_hint",
        "time_index_hint",
        "slot_hint",
        "value_hint",
        "qa",
        "options",
        "gold",
    }
    exposed_forbidden_keys = sorted(
        {
            key
            for request in requests
            for mapping in build_stage3_parent_v2.walk_mappings(request)
            for key in mapping
            if str(key).casefold() in request_forbidden_keys
        }
    )
    if exposed_forbidden_keys:
        raise RuntimeError(
            f"Legacy semantic fields reached model requests: {exposed_forbidden_keys}"
        )
    if args.prepare_only:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "mode": "prepare_only",
            "input": str(input_path),
            "input_row_count": len(rows),
            "eligible_request_count": len(request_rows),
            "selected_request_count": len(requests),
            "selected_event_count": selected_event_count,
            "fallback_event_count": fallback_event_count,
            "semantic_audit_exclusion_count": len(
                selected_semantic_exclusions
            ),
            "semantic_audit_exclusion_schema_version": (
                SEMANTIC_EXCLUSION_SCHEMA_VERSION
            ),
            "excluded_source_event_count": len(excluded_source_events),
            "legacy_qa_in_model_prompt": False,
            "legacy_semantic_label_fields_in_model_request": False,
            "old_graph_events_inherited": False,
            "training_authorized": False,
        }
        write_json(output_dir / "rebuild_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.reuse_raw_model_outputs:
        persisted_records = [
            record
            for path in args.reuse_raw_model_outputs
            for record in read_jsonl(path.resolve())
        ]
        raw_outputs = remap_reused_atomic_outputs(
            persisted_records,
            requests,
        )
    else:
        if not args.model.exists():
            raise RuntimeError(f"Local model does not exist: {args.model}")
        raw_outputs = run_local_model(
            requests,
            model_path=args.model.resolve(),
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        )
    write_jsonl(output_dir / "raw_model_outputs.jsonl", raw_outputs)
    output_by_id = {record["request_id"]: record for record in raw_outputs}
    candidate_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    ineligible_source_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for request in requests:
        if not request.get("events"):
            ineligible_source_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request.get("request_id"),
                    "source_sample_id": request.get("source_sample_id"),
                    "source_line_number": request.get("source_line_number"),
                    "source_event_count": int(request.get("source_event_count") or 0),
                    "excluded_source_event_count": len(
                        request.get("skipped_events") or []
                    ),
                    "reason": "no_clean_linked_semantic_events",
                    "dialogue_fallback_used": False,
                    "disposition": "source_row_quarantined",
                }
            )
            continue
        record = output_by_id.get(request["request_id"])
        accepted, rejected = validate_response(
            request,
            record.get("parsed") if isinstance(record, dict) else None,
        )
        accepted = link_terminal_dependencies(accepted)
        dependency_schema_errors = [
            {
                "source_event_id": (event.get("provenance") or {}).get(
                    "source_event_id"
                ),
                "errors": [
                    f"semantic_graph_v2:{item['code']}"
                    for item in semantic_graph_v2.validate_event(event)
                ],
            }
            for event in accepted
            if semantic_graph_v2.validate_event(event)
        ]
        if dependency_schema_errors:
            rejected.extend(
                {
                    "request_id": request["request_id"],
                    "reason": "dependency_link_schema_rejected",
                    **item,
                }
                for item in dependency_schema_errors
            )
        fingerprints: Counter[str] = Counter(
            sha256_json(
                {
                    "entity": event.get("entity"),
                    "slot": event.get("slot"),
                    "value": event.get("value"),
                    "status": event.get("status"),
                    "evidence_turn_ids": event.get("evidence_turn_ids"),
                }
            )
            for event in accepted
        )
        duplicate_fingerprints = sorted(
            fingerprint for fingerprint, count in fingerprints.items() if count > 1
        )
        if duplicate_fingerprints:
            rejected.append(
                {
                    "request_id": request["request_id"],
                    "reason": "duplicate_extracted_semantic_events",
                    "duplicate_event_sha256": duplicate_fingerprints,
                }
            )
        allowed_exclusion_reasons = {
            "ambiguous_status_evidence_turn",
            "legacy_qa_evidence_turn",
            "negative_absence_evidence_turn",
            "deterministic_padding_evidence_turn",
            "nonprospective_canceled_or_delayed_evidence",
            "status_only_evidence_turn",
            "independent_semantic_audit_exclusion",
        }
        unknown_exclusions = [
            item
            for item in request.get("skipped_events") or []
            if isinstance(item, dict)
            and normalize_space(item.get("reason")) not in allowed_exclusion_reasons
        ]
        row_complete = (
            not rejected
            and not unknown_exclusions
            and len(accepted) == len(request["events"])
            and len(accepted) >= int(args.minimum_events_per_parent)
        )
        coverage = {
            "source_event_count": int(request.get("source_event_count") or 0),
            "eligible_evidence_event_count": len(request["events"]),
            "accepted_event_count": len(accepted),
            "rejected_eligible_event_count": len(rejected),
            "excluded_source_event_count": len(request.get("skipped_events") or []),
            "unknown_exclusion_count": len(unknown_exclusions),
            "allowed_exclusion_reasons": sorted(allowed_exclusion_reasons),
            "fallback_event_count": int(request.get("fallback_event_count") or 0),
            "complete_for_eligible_evidence": row_complete,
            "legacy_semantic_labels_exposed_to_model": False,
            "old_graph_events_inherited": False,
        }
        graph_row = {
            "schema_version": SCHEMA_VERSION,
            "source_sample_id": request["source_sample_id"],
            "source_line_number": request["source_line_number"],
            "source_event_count": int(request.get("source_event_count") or 0),
            "excluded_source_event_count": len(request.get("skipped_events") or []),
            "request_id": request["request_id"],
            "semantic_parent_graph": {
                "schema_version": semantic_graph_v2.SCHEMA_VERSION,
                "graph_id": f"{request['source_sample_id']}:canonical_stage2_graph_v2",
                "parent_graph_id": f"stage2_rebuild_root:{request['source_sample_id']}",
                "quarantined_source_graph_snapshot_hash": request.get(
                    "source_graph_snapshot_hash"
                )
                or None,
                "graph_role": (
                    "evidence_reconstructed_stage2_parent_snapshot"
                    if row_complete
                    else "quarantined_partial_reconstruction_candidate"
                ),
                "events": accepted,
                "event_count": len(accepted),
            },
            "coverage": coverage,
            "extraction": {
                "model": str(args.model.resolve()),
                "temperature": 0.0,
                "request_event_count": len(request["events"]),
                "accepted_event_count": len(accepted),
                "rejected_event_count": len(rejected),
                "row_atomic_admission": True,
            },
        }
        candidate_rows.append(graph_row)
        if row_complete:
            canonical_rows.append(graph_row)
            status_counts.update(event["status"] for event in accepted)
        else:
            rejected_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request["request_id"],
                    "source_sample_id": request["source_sample_id"],
                    "source_line_number": request["source_line_number"],
                    "reason": "row_atomic_reconstruction_incomplete",
                    "accepted_event_count": len(accepted),
                    "expected_event_count": len(request["events"]),
                    "minimum_events_per_parent": int(args.minimum_events_per_parent),
                    "unknown_source_event_exclusions": copy.deepcopy(
                        unknown_exclusions
                    ),
                    "event_rejections": rejected,
                }
            )
        rejected_rows.extend(
            {
                "schema_version": SCHEMA_VERSION,
                "source_sample_id": request["source_sample_id"],
                "source_line_number": request["source_line_number"],
                **copy.deepcopy(item),
            }
            for item in rejected
        )
    write_jsonl(output_dir / "stage2_semantic_graph_candidates.jsonl", candidate_rows)
    write_jsonl(output_dir / "stage2_semantic_graphs.jsonl", canonical_rows)
    write_jsonl(output_dir / "rejected_events.jsonl", rejected_rows)
    write_jsonl(output_dir / "ineligible_source_rows.jsonl", ineligible_source_rows)

    forbidden_paths = [path.resolve() for path in args.forbidden_jsonl]
    forbidden_index, forbidden_row_count = build_stage3_parent_v2.load_forbidden_identifiers(
        forbidden_paths
    )
    parent_rows: list[dict[str, Any]] = []
    parent_rejections: list[dict[str, Any]] = []
    parent_audits: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()
    for graph_row in canonical_rows:
        line_number = int(graph_row.get("source_line_number") or 0)
        source_row = source_by_line.get(line_number)
        if not isinstance(source_row, dict):
            parent_rejections.append(
                {"source_line_number": line_number, "reason": "source_row_missing"}
            )
            continue
        training_reasons = build_stage3_parent_v2.training_rejection_reasons(source_row)
        exact_matches = build_stage3_parent_v2.exact_forbidden_matches(
            build_stage3_parent_v2.exact_identifiers(source_row),
            forbidden_index["exact"],
        )
        if training_reasons or exact_matches:
            parent_rejections.append(
                {
                    "source_line_number": line_number,
                    "source_sample_id": graph_row.get("source_sample_id"),
                    "reason": "parent_source_not_train_safe",
                    "training_reasons": training_reasons,
                    "forbidden_exact_identity_matches": exact_matches,
                }
            )
            continue
        parent, audit = build_stage3_parent_v3(
            source_row,
            graph_row,
            source_path=input_path,
        )
        if parent is None:
            parent_rejections.append(
                {
                    "source_line_number": line_number,
                    "source_sample_id": graph_row.get("source_sample_id"),
                    **audit,
                }
            )
            continue
        parent_id = normalize_space(parent.get("sample_id"))
        if not parent_id or parent_id in seen_parent_ids:
            parent_rejections.append(
                {
                    "source_line_number": line_number,
                    "source_sample_id": graph_row.get("source_sample_id"),
                    "reason": "duplicate_or_empty_parent_id",
                    "parent_id": parent_id,
                }
            )
            continue
        seen_parent_ids.add(parent_id)
        parent_rows.append(parent)
        parent_audits.append(audit)
    write_jsonl(output_dir / "stage3_parent_v3.jsonl", parent_rows)
    write_jsonl(output_dir / "stage3_parent_v3_rejected.jsonl", parent_rejections)
    write_jsonl(output_dir / "stage3_parent_v3_audit.jsonl", parent_audits)

    eligible_selected_request_count = sum(
        1 for request in requests if request.get("events")
    )
    complete_selection = (
        len(canonical_rows) == eligible_selected_request_count
        and len(parent_rows) == len(canonical_rows)
        and not parent_rejections
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "local_model_evidence_extraction",
        "input": str(input_path),
        "model": str(args.model.resolve()),
        "input_row_count": len(rows),
        "eligible_request_count": len(request_rows),
        "selected_request_count": len(requests),
        "eligible_selected_request_count": eligible_selected_request_count,
        "ineligible_source_row_count": len(ineligible_source_rows),
        "selected_event_count": selected_event_count,
        "accepted_canonical_event_count": sum(
            row["semantic_parent_graph"]["event_count"] for row in canonical_rows
        ),
        "excluded_source_event_count": len(excluded_source_events),
        "fallback_event_count": fallback_event_count,
        "semantic_audit_exclusion_count": len(selected_semantic_exclusions),
        "semantic_audit_exclusion_schema_version": (
            SEMANTIC_EXCLUSION_SCHEMA_VERSION
        ),
        "candidate_graph_count": len(candidate_rows),
        "canonical_graph_count": len(canonical_rows),
        "row_atomic_rejected_graph_count": (
            eligible_selected_request_count - len(canonical_rows)
        ),
        "rejection_record_count": len(rejected_rows),
        "parent_v3_count": len(parent_rows),
        "parent_v3_rejected_count": len(parent_rejections),
        "forbidden_row_count": forbidden_row_count,
        "complete_selection": complete_selection,
        "status_counts": dict(sorted(status_counts.items())),
        "legacy_qa_in_model_prompt": False,
        "legacy_semantic_label_fields_in_model_request": False,
        "old_graph_events_inherited": False,
        "row_atomic_admission": True,
        "training_authorized": False,
    }
    write_json(output_dir / "rebuild_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if complete_selection else 2


if __name__ == "__main__":
    raise SystemExit(main())
