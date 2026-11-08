from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


HARDEN_TEST_DIR = Path("harden-test")
if HARDEN_TEST_DIR.exists():
    sys.path.insert(0, str(HARDEN_TEST_DIR.resolve()))

import answer_semantics  # noqa: E402
import dialogue_hardening  # noqa: E402
import evolved_qa_validation  # noqa: E402
import qa_evolution  # noqa: E402


EVENT_VERB_PATTERN = re.compile(
    r"\b("
    r"lost|started|launched|asked|mentioned|found|visited|planned|planning|performed|performing|"
    r"rehearsed|rehearsing|joined|created|decided|opened|opening|shared|told|went|took|"
    r"met|worked|working|built|building|learned|practiced|competed"
    r")\b",
    re.IGNORECASE,
)
EVENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "got",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "last",
    "me",
    "month",
    "months",
    "my",
    "of",
    "on",
    "or",
    "own",
    "s",
    "she",
    "so",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "week",
    "with",
    "year",
    "yesterday",
    "you",
}
TIME_CUE_PATTERN = re.compile(
    r"\b(yesterday|tomorrow|last week|next week|last month|next month|a few months back|few months back|"
    r"a few months ago|few months ago|this month|last year|next year|today|tonight|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug structured pairwise temporal QA evolution.")
    parser.add_argument("--input", default="harden-test/batch50_round1.json")
    parser.add_argument("--model-eval", default="harden-test/batch50_round1_model_eval_qwen3_4b_8b.json")
    parser.add_argument("--frontier", default="harden-test/batch50_round1_frontier_qwen3_4b_8b.json")
    parser.add_argument("--hardenability-report", default="harden-test/batch50_round1_hardenability_report.json")
    parser.add_argument("--output-prefix", default="debug_structured_temporal_evolution")
    parser.add_argument("--models", default="qwen3-4b,qwen3-8b")
    parser.add_argument("--challenger-model", default=None)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=int, default=None)
    parser.add_argument("--critic-temperature", type=float, default=0.1)
    parser.add_argument("--option-temperature", type=float, default=0.25)
    parser.add_argument("--light-temperature", type=float, default=0.3)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--light-candidates-per-model", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temporal-mode", choices=("real_world", "session_order", "state_change", "mention_order", "hybrid"), default="hybrid")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sentence_fragments(text: str) -> list[str]:
    normalized = evolved_qa_validation.normalize_space(text)
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\s+-\s+", normalized)
    return [part.strip(" .!?") for part in parts if part.strip(" .!?")]


def event_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in EVENT_STOPWORDS}


def keyword_similarity(left: str, right: str) -> float:
    left_keywords = event_keywords(left)
    right_keywords = event_keywords(right)
    if not left_keywords or not right_keywords:
        return 0.0
    return len(left_keywords & right_keywords) / len(left_keywords | right_keywords)


def looks_like_same_event_conflict(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if str(first.get("speaker") or "") != str(second.get("speaker") or ""):
        return False
    if first.get("time_expression") == second.get("time_expression"):
        return False
    similarity = keyword_similarity(str(first.get("quote") or ""), str(second.get("quote") or ""))
    if similarity >= 0.35:
        return True
    shared = event_keywords(str(first.get("quote") or "")) & event_keywords(str(second.get("quote") or ""))
    conflict_markers = {"lost", "job", "visited", "opened", "started", "launched", "planning", "perform"}
    return len(shared & conflict_markers) >= 2


def parse_session_datetime(value: Any) -> datetime | None:
    text = str(value or "")
    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),\s*(20\d{2})\b", text)
    if not match:
        return None
    day = int(match.group(1))
    month = MONTHS.get(match.group(2).lower())
    year = int(match.group(3))
    if not month:
        return None
    return datetime(year, month, day)


def month_end(year: int, month: int) -> datetime:
    if month == 12:
        return datetime(year + 1, 1, 1) - timedelta(days=1)
    return datetime(year, month + 1, 1) - timedelta(days=1)


def estimate_event_interval(text: str, session_time: datetime | None) -> tuple[datetime | None, datetime | None, str]:
    lowered = text.lower()
    cue_match = TIME_CUE_PATTERN.search(lowered)
    cue = cue_match.group(0) if cue_match else ""
    if session_time is None:
        return None, None, cue
    if cue in {"today", "tonight"}:
        return session_time, session_time, cue
    if cue == "yesterday":
        event_time = session_time - timedelta(days=1)
        return event_time, event_time, cue
    if cue == "tomorrow":
        event_time = session_time + timedelta(days=1)
        return event_time, event_time, cue
    if cue == "last week":
        return session_time - timedelta(days=13), session_time - timedelta(days=7), cue
    if cue == "next week":
        return session_time + timedelta(days=7), session_time + timedelta(days=13), cue
    if cue == "last month":
        month = session_time.month - 1
        year = session_time.year
        if month == 0:
            month = 12
            year -= 1
        return datetime(year, month, 1), month_end(year, month), cue
    if cue == "next month":
        month = session_time.month + 1
        year = session_time.year
        if month == 13:
            month = 1
            year += 1
        return datetime(year, month, 1), month_end(year, month), cue
    if cue in {"a few months back", "few months back", "a few months ago", "few months ago"}:
        return session_time - timedelta(days=120), session_time - timedelta(days=60), cue
    if cue == "this month":
        return session_time.replace(day=1), month_end(session_time.year, session_time.month), cue
    if cue == "last year":
        return datetime(session_time.year - 1, 1, 1), datetime(session_time.year - 1, 12, 31), cue
    if cue == "next year":
        return datetime(session_time.year + 1, 1, 1), datetime(session_time.year + 1, 12, 31), cue
    if cue in MONTHS:
        month = MONTHS[cue]
        year = session_time.year
        if month > session_time.month + 2:
            year -= 1
        return datetime(year, month, 1), month_end(year, month), cue
    return None, None, cue


def event_phrase(
    turn: dict[str, Any],
    inherited_session_time: datetime | None = None,
) -> tuple[str, str, datetime | None, datetime | None, str] | None:
    text = str(turn.get("text", ""))
    speaker = str(turn.get("speaker") or "Someone")
    fragments = sentence_fragments(text)
    if not fragments:
        return None
    chosen = None
    for fragment in fragments:
        if EVENT_VERB_PATTERN.search(fragment):
            chosen = fragment
            break
    if chosen is None:
        return None
    chosen = re.sub(r"^(hey|hi|hello)\s+\w+[!,]?\s*", "", chosen, flags=re.IGNORECASE).strip()
    if len(chosen.split()) < 3:
        return None
    session_time = parse_session_datetime(turn.get("session_time")) or inherited_session_time
    event_start, event_end, time_expression = estimate_event_interval(chosen, session_time)
    quote = chosen
    phrase_quote = quote
    if len(phrase_quote) > 90:
        phrase_quote = phrase_quote[:87].rstrip() + "..."
    phrase = f'{speaker} saying "{phrase_quote}"'
    return phrase, quote, event_start, event_end, time_expression


def turn_order(turn: dict[str, Any]) -> tuple[int, int]:
    return evolved_qa_validation.turn_sort_key(turn)


def extract_events(result: dict[str, Any], *, temporal_mode: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_session_time: dict[str, datetime] = {}
    for turn in result.get("final_dialogue", []):
        if not isinstance(turn, dict):
            continue
        session_key = str(turn.get("session_key") or "")
        explicit_session_time = parse_session_datetime(turn.get("session_time"))
        if explicit_session_time and session_key:
            current_session_time[session_key] = explicit_session_time
        turn_id = qa_evolution.display_turn_id(turn)
        phrase_quote = event_phrase(turn, current_session_time.get(session_key))
        if not turn_id or not phrase_quote:
            continue
        phrase, quote, event_start, event_end, time_expression = phrase_quote
        session_time = current_session_time.get(session_key)
        if temporal_mode == "real_world" and (event_start is None or event_end is None):
            continue
        if temporal_mode == "session_order" and session_time is None:
            continue
        key = f"{turn_id}:{phrase.lower()}"
        if key in seen:
            continue
        seen.add(key)
        events.append(
            {
                "turn_id": turn_id,
                "speaker": turn.get("speaker"),
                "phrase": phrase,
                "quote": quote,
                "event_time": event_start.isoformat() if event_start else None,
                "event_start": event_start.isoformat() if event_start else None,
                "event_end": event_end.isoformat() if event_end else None,
                "session_key": session_key,
                "session_time": session_time.isoformat() if session_time else None,
                "time_expression": time_expression,
                "order": turn_order(turn),
            }
        )
    return events


def make_pairwise_candidate(first: dict[str, Any], second: dict[str, Any], index: int, *, temporal_mode: str) -> dict[str, Any]:
    if temporal_mode == "real_world":
        question = f"Which event happened earlier in real-world time: {first['phrase']} or {second['phrase']}?"
        why_harder = "Requires comparing real-world event times using dialogue time expressions and session timing."
        expected = (
            f"{first['phrase']} is dated by '{first.get('time_expression')}' between {first.get('event_start')} and {first.get('event_end')}, "
            f"which is earlier than {second['phrase']} dated by '{second.get('time_expression')}' between {second.get('event_start')} and {second.get('event_end')}."
        )
    elif temporal_mode == "session_order":
        question = f"Which event happened in the earlier dialogue session: {first['phrase']} or {second['phrase']}?"
        why_harder = "Requires comparing two event-bearing turns across dialogue sessions using session chronology."
        expected = (
            f"{first['phrase']} is in session {first.get('session_key')} at {first.get('session_time')}, "
            f"which is earlier than {second['phrase']} in session {second.get('session_key')} at {second.get('session_time')}."
        )
    else:
        question = f"Which dialogue event is mentioned earlier: {first['phrase']} or {second['phrase']}?"
        why_harder = "Requires comparing the order of two dialogue events instead of retrieving a single fact."
        expected = f"{first['turn_id']} is mentioned before {second['turn_id']}, so {first['phrase']} is the earlier mentioned event."
    return {
        "candidate_index": index,
        "evolved_question": question,
        "evolved_answer": str(first["phrase"]),
        "evolved_evidence": [str(first["turn_id"]), str(second["turn_id"])],
        "evidence_quotes": [
            {"turn_id": str(first["turn_id"]), "quote": str(first["quote"])},
            {"turn_id": str(second["turn_id"]), "quote": str(second["quote"])},
        ],
        "evolution_type": "temporal_ordering",
        "why_harder": why_harder,
        "source_facts": [
            {
                "turn_id": str(first["turn_id"]),
                "fact": str(first["phrase"]),
                "time_expression": first.get("time_expression"),
                "event_start": first.get("event_start"),
                "event_end": first.get("event_end"),
                "session_key": first.get("session_key"),
                "session_time": first.get("session_time"),
            },
            {
                "turn_id": str(second["turn_id"]),
                "fact": str(second["phrase"]),
                "time_expression": second.get("time_expression"),
                "event_start": second.get("event_start"),
                "event_end": second.get("event_end"),
                "session_key": second.get("session_key"),
                "session_time": second.get("session_time"),
            },
        ],
        "expected_reasoning": expected,
        "temporal_relation": {
            "relation": "before",
            "scope": "earlier",
            "anchor_turn_id": str(second["turn_id"]),
            "answer_turn_id": str(first["turn_id"]),
            "anchor_event": str(second["phrase"]),
            "answer_event": str(first["phrase"]),
        },
        "_structured_generation": True,
    }


def short_quote(text: Any, limit: int = 80) -> str:
    quote = evolved_qa_validation.normalize_space(text)
    if len(quote) <= limit:
        return quote
    return quote[: limit - 3].rstrip() + "..."


def state_change_score(first: dict[str, Any], second: dict[str, Any]) -> float:
    score = keyword_similarity(str(first.get("quote") or ""), str(second.get("quote") or ""))
    if first.get("speaker") == second.get("speaker"):
        score += 0.2
    if first.get("session_key") != second.get("session_key"):
        score += 0.2
    later_words = event_keywords(str(second.get("quote") or ""))
    update_markers = {
        "launched",
        "started",
        "found",
        "opened",
        "planning",
        "working",
        "developed",
        "created",
        "joined",
        "rehearsing",
        "perform",
        "visited",
    }
    if later_words & update_markers:
        score += 0.2
    return score


def make_state_change_candidate(first: dict[str, Any], second: dict[str, Any], index: int) -> dict[str, Any]:
    question = (
        f"By the latest relevant dialogue session, what current update followed the earlier event "
        f"{first['phrase']}?"
    )
    answer = str(second["phrase"])
    return {
        "candidate_index": index,
        "evolved_question": question,
        "evolved_answer": answer,
        "evolved_evidence": [str(first["turn_id"]), str(second["turn_id"])],
        "evidence_quotes": [
            {"turn_id": str(first["turn_id"]), "quote": str(first["quote"])},
            {"turn_id": str(second["turn_id"]), "quote": str(second["quote"])},
        ],
        "evolution_type": "final_state",
        "why_harder": "Requires using an earlier event as context and selecting the later cross-session update as the final relevant state.",
        "source_facts": [
            {
                "turn_id": str(first["turn_id"]),
                "fact": str(first["phrase"]),
                "session_key": first.get("session_key"),
                "session_time": first.get("session_time"),
            },
            {
                "turn_id": str(second["turn_id"]),
                "fact": str(second["phrase"]),
                "session_key": second.get("session_key"),
                "session_time": second.get("session_time"),
            },
        ],
        "expected_reasoning": (
            f"The earlier context is {first['phrase']} in {first.get('session_key')}. "
            f"The latest relevant update appears later in {second.get('session_key')}: {second['phrase']}."
        ),
        "_structured_generation": True,
        "_temporal_generation_mode": "state_change_fallback",
    }


def generate_state_change_candidates(result: dict[str, Any], candidate_count: int) -> list[dict[str, Any]]:
    events = sorted(extract_events(result, temporal_mode="session_order"), key=lambda event: (event["session_time"], event["order"]))
    scored_pairs: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    for left_index, first in enumerate(events):
        for right_index, second in enumerate(events[left_index + 1 :], start=left_index + 1):
            if first.get("session_key") == second.get("session_key"):
                continue
            if first.get("session_time") == second.get("session_time"):
                continue
            if looks_like_same_event_conflict(first, second):
                continue
            score = state_change_score(first, second)
            if score < 0.35:
                continue
            scored_pairs.append((score, right_index - left_index, first, second))
    scored_pairs.sort(key=lambda item: (-item[0], -item[1], item[2]["order"], item[3]["order"]))
    return [
        make_state_change_candidate(first, second, index)
        for index, (_, _, first, second) in enumerate(scored_pairs[:candidate_count])
    ]


def generate_candidates_for_mode(result: dict[str, Any], candidate_count: int, *, temporal_mode: str) -> list[dict[str, Any]]:
    if temporal_mode == "state_change":
        return generate_state_change_candidates(result, candidate_count)
    if temporal_mode == "real_world":
        events = sorted(extract_events(result, temporal_mode=temporal_mode), key=lambda event: (event["event_time"], event["order"]))
    elif temporal_mode == "session_order":
        events = sorted(extract_events(result, temporal_mode=temporal_mode), key=lambda event: (event["session_time"], event["order"]))
    else:
        events = sorted(extract_events(result, temporal_mode=temporal_mode), key=lambda event: event["order"])
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for left_index, first in enumerate(events):
        for right_index, second in enumerate(events[left_index + 1 :], start=left_index + 1):
            if temporal_mode == "real_world":
                first_end = datetime.fromisoformat(str(first["event_end"]))
                second_start = datetime.fromisoformat(str(second["event_start"]))
                if first_end >= second_start:
                    continue
                if looks_like_same_event_conflict(first, second):
                    continue
            if temporal_mode == "session_order":
                if first.get("session_key") == second.get("session_key"):
                    continue
                if first.get("session_time") == second.get("session_time"):
                    continue
            if temporal_mode == "mention_order" and first["order"] == second["order"]:
                continue
            gap = right_index - left_index
            pairs.append((gap, first, second))
    pairs.sort(key=lambda item: (-item[0], item[1]["order"], item[2]["order"]))
    return [make_pairwise_candidate(first, second, index, temporal_mode=temporal_mode) for index, (_, first, second) in enumerate(pairs[:candidate_count])]


def generate_structured_temporal_candidates(result: dict[str, Any], candidate_count: int, *, temporal_mode: str = "hybrid") -> list[dict[str, Any]]:
    if temporal_mode != "hybrid":
        return generate_candidates_for_mode(result, candidate_count, temporal_mode=temporal_mode)
    mode_plan = [
        ("real_world", "real_world"),
        ("state_change", "state_change"),
        ("session_order", "session_order_fallback"),
        ("mention_order", "mention_order_fallback"),
    ]
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for mode, mode_label in mode_plan:
        for candidate in generate_candidates_for_mode(result, candidate_count, temporal_mode=mode):
            key = (
                str(candidate.get("evolved_question", "")).lower(),
                str(candidate.get("evolved_answer", "")).lower(),
                tuple(str(item) for item in candidate.get("evolved_evidence", [])),
            )
            if key in seen:
                continue
            seen.add(key)
            candidate["_temporal_generation_mode"] = mode_label
            candidate["candidate_index"] = len(merged)
            merged.append(candidate)
    return merged[: max(candidate_count, candidate_count * 3)]


def failed_summary(candidate: dict[str, Any], index: int, reason: str, rule: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "candidate_index": candidate.get("candidate_index", index),
        "evolution_type": candidate.get("evolution_type"),
        "rule_safe": False,
        "critic_safe": False,
        "option_safe": False,
        "light_eval_accuracy": None,
        "selected": False,
        "rejection_reason": reason,
        "why_harder": candidate.get("why_harder"),
        "rule_validation": rule or {"rule_safe": False, "rejection_reason": reason},
        "critic": None,
        "option_validation": None,
        "light_eval": None,
        "candidate": candidate,
    }


def select_structured_candidate(
    *,
    result: dict[str, Any],
    base_url: str,
    api_key: str,
    challenger_model: str,
    solver_models: list[str],
    request_timeout: int,
    critic_temperature: float,
    option_temperature: float,
    eval_temperature: float,
    max_retries: int,
    candidate_count: int,
    light_candidates_per_model: int,
    temporal_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dialogue = result.get("final_dialogue", [])
    original_qa = result.get("qa", {}) if isinstance(result.get("qa"), dict) else {}
    candidates = generate_structured_temporal_candidates(result, candidate_count, temporal_mode=temporal_mode)
    summaries: list[dict[str, Any]] = []
    safe_candidates: list[tuple[float | None, int, dict[str, Any], dict[str, Any]]] = []

    for index, candidate in enumerate(candidates):
        rule = evolved_qa_validation.validate_evolved_qa_rules(candidate, dialogue, original_qa)
        if not rule.get("rule_safe"):
            summaries.append(failed_summary(candidate, index, str(rule.get("rejection_reason", "")), rule))
            continue
        critic = evolved_qa_validation.validate_evolved_qa_with_critic(
            candidate,
            dialogue,
            original_qa=original_qa,
            base_url=base_url,
            api_key=api_key,
            model=challenger_model,
            temperature=critic_temperature,
            request_timeout=request_timeout,
        )
        if not critic.get("critic_safe"):
            summaries.append(
                {
                    **failed_summary(candidate, index, str(critic.get("reason") or "critic_rejected"), rule),
                    "rule_safe": True,
                    "critic": critic,
                    "critic_safe": False,
                }
            )
            continue
        option_validation = None
        light_eval = None
        light_accuracy = None
        option_safe = False
        rejection_reason = None
        candidate_result = None
        try:
            raw_options = qa_evolution.generate_evolved_mcq_options(
                candidate,
                dialogue,
                base_url=base_url,
                api_key=api_key,
                model=challenger_model,
                temperature=option_temperature,
                request_timeout=request_timeout,
            )
            ordered_options, correct_index = qa_evolution.reorder_options(
                candidate["evolved_question"],
                candidate["evolved_answer"],
                raw_options,
            )
            option_validation = evolved_qa_validation.validate_evolved_mcq_options(
                candidate,
                ordered_options,
                dialogue,
                semantic_equivalent=answer_semantics.is_semantically_equivalent_answer,
            )
            if not option_validation.get("option_safe"):
                rejection_reason = str(option_validation.get("rejection_reason", ""))
            else:
                option_safe = True
                candidate_result = qa_evolution.apply_evolved_candidate_to_result(
                    result,
                    candidate,
                    ordered_options,
                    correct_index,
                )
                light_eval = qa_evolution.evaluate_light_candidate(
                    candidate_result,
                    model_names=solver_models,
                    base_url=base_url,
                    api_key=api_key,
                    request_timeout=request_timeout,
                    temperature=eval_temperature,
                    max_retries=max_retries,
                    candidates_per_model=light_candidates_per_model,
                )
                light_accuracy = light_eval.get("average_accuracy")
                safe_candidates.append((light_accuracy, index, candidate, candidate_result))
        except Exception as error:  # noqa: BLE001
            rejection_reason = str(error)

        summaries.append(
            {
                "candidate_index": candidate.get("candidate_index", index),
                "evolution_type": candidate.get("evolution_type"),
                "rule_safe": True,
                "critic_safe": True,
                "option_safe": option_safe,
                "light_eval_accuracy": light_accuracy,
                "selected": False,
                "rejection_reason": rejection_reason,
                "why_harder": candidate.get("why_harder"),
                "rule_validation": rule,
                "critic": critic,
                "option_validation": option_validation,
                "light_eval": light_eval,
                "candidate": candidate,
            }
        )

    selected_index = None
    selected_candidate = None
    selected_result = copy.deepcopy(result)
    selected_light_accuracy = None
    status = "no_safe_evolved_candidate"
    if safe_candidates:
        viable_candidates = [
            item for item in safe_candidates
            if isinstance(item[0], (int, float)) and float(item[0]) < 0.9
        ]
        if not viable_candidates:
            status = "no_frontier_gain_still_easy"
        else:
            safe_candidates = viable_candidates
        in_band = [item for item in safe_candidates if isinstance(item[0], (int, float)) and 0.4 <= float(item[0]) <= 0.7]
        if in_band:
            chosen = min(in_band, key=lambda item: abs(float(item[0]) - 0.55))
            status = "selected_frontier_band"
        elif status != "no_frontier_gain_still_easy":
            chosen = min(safe_candidates, key=lambda item: float(item[0]) if isinstance(item[0], (int, float)) else 999.0)
            status = "selected_lowest_accuracy"
        if status != "no_frontier_gain_still_easy":
            selected_light_accuracy, selected_index, selected_candidate, selected_result = chosen
            selected_candidate_id = selected_candidate.get("candidate_index", selected_index)
            for summary in summaries:
                if summary["candidate_index"] == selected_candidate_id:
                    summary["selected"] = True
                    break

    evolution = {
        "enabled": True,
        "generation_mode": "structured_temporal",
        "temporal_mode": temporal_mode,
        "attempted": True,
        "selected_candidate_index": selected_index,
        "selected_evolution_type": selected_candidate.get("evolution_type") if selected_candidate else None,
        "selected_light_eval_accuracy": selected_light_accuracy,
        "candidate_count": candidate_count,
        "generated_candidate_count": len(candidates),
        "safe_candidate_count": len(safe_candidates),
        "candidate_summaries": summaries,
        "status": status,
    }
    selected_result["qa_evolution"] = evolution
    return selected_result, evolution


def mark_skipped(result: dict[str, Any], reason: str) -> None:
    result["qa_evolution"] = {
        "enabled": False,
        "generation_mode": "structured_temporal",
        "attempted": False,
        "selected_candidate_index": None,
        "selected_evolution_type": None,
        "selected_light_eval_accuracy": None,
        "candidate_count": 0,
        "generated_candidate_count": 0,
        "safe_candidate_count": 0,
        "candidate_summaries": [],
        "status": reason,
    }


def run_structured_temporal(args: argparse.Namespace) -> dict[str, Any]:
    api_config = dialogue_hardening.build_config(args)
    solver_models = dialogue_hardening.split_models(args.models)
    frontier_payload = load_json(args.frontier)
    model_eval_payload = load_json(args.model_eval)
    checkpoint_path = Path(args.output_prefix).with_suffix(".json")
    source_payload = load_json(checkpoint_path) if args.resume and checkpoint_path.exists() else frontier_payload
    output_payload = qa_evolution.merge_model_eval_into_payload(source_payload, model_eval_payload)
    results = qa_evolution.collect_results(output_payload)
    sample_id = str(output_payload.get("sample_id", results[0].get("sample_id", "")) if results else "")
    hmap = qa_evolution.hardenability_map(load_json(args.hardenability_report), sample_id)

    attempted = sum(
        1
        for result in results
        if isinstance(result.get("qa_evolution"), dict) and result["qa_evolution"].get("attempted") is True
    ) if args.resume else 0
    for index, result in enumerate(results, start=1):
        if args.resume and result.get("qa_evolution", {}).get("attempted") is True:
            continue
        hardenability = hmap.get(qa_evolution.result_key(result), {})
        eligible, reason = qa_evolution.is_eligible_for_evolution(result, hardenability)
        if not eligible:
            mark_skipped(result, reason)
            continue
        if args.max_samples is not None and attempted >= args.max_samples:
            mark_skipped(result, "max_samples_limit_skipped")
            continue
        attempted += 1
        print(f"Structured temporal {attempted} qa_index={result.get('qa_index')} ({index}/{len(results)})")
        try:
            selected_result, evolution = select_structured_candidate(
                result=result,
                base_url=api_config["base_url"],
                api_key=api_config["api_key"],
                challenger_model=api_config["challenger_model"],
                solver_models=solver_models,
                request_timeout=api_config["request_timeout"],
                critic_temperature=args.critic_temperature,
                option_temperature=args.option_temperature,
                eval_temperature=args.light_temperature,
                max_retries=args.max_retries,
                candidate_count=args.candidate_count,
                light_candidates_per_model=args.light_candidates_per_model,
                temporal_mode=args.temporal_mode,
            )
            result.clear()
            result.update(selected_result)
            result["qa_evolution"] = evolution
        except Exception as error:  # noqa: BLE001
            result["qa_evolution"] = {
                "enabled": True,
                "generation_mode": "structured_temporal",
                "temporal_mode": args.temporal_mode,
                "attempted": True,
                "selected_candidate_index": None,
                "selected_evolution_type": None,
                "selected_light_eval_accuracy": None,
                "candidate_count": args.candidate_count,
                "generated_candidate_count": 0,
                "safe_candidate_count": 0,
                "candidate_summaries": [],
                "status": "candidate_generation_failed",
                "error": str(error),
            }
        output_payload["qa_evolution_summary"] = qa_evolution.build_evolution_summary(output_payload, previous_payload=frontier_payload)
        output_payload["window_sharing_summary"] = {"enabled": False}
        output_payload["qa_clustering_summary"] = {"enabled": False}
        output_payload["solver_training_summary"] = {"enabled": False}
        save_json(checkpoint_path, output_payload)

    output_payload["qa_evolution_summary"] = qa_evolution.build_evolution_summary(output_payload, previous_payload=frontier_payload)
    output_payload["window_sharing_summary"] = {"enabled": False}
    output_payload["qa_clustering_summary"] = {"enabled": False}
    output_payload["solver_training_summary"] = {"enabled": False}
    return output_payload


def main() -> int:
    args = parse_args()
    payload = run_structured_temporal(args)
    output_prefix = Path(args.output_prefix)
    result_path = output_prefix.with_suffix(".json")
    report_path = output_prefix.with_name(output_prefix.name + "_report.json")
    save_json(result_path, payload)
    report = qa_evolution.build_report(load_json(args.frontier), payload)
    report["generation_mode"] = "structured_temporal"
    report["temporal_mode"] = args.temporal_mode
    save_json(report_path, report)
    print(f"Saved structured temporal result to: {result_path.resolve()}")
    print(f"Saved structured temporal report to: {report_path.resolve()}")
    print(json.dumps(
        {
            "attempted_evolution_count": report.get("attempted_evolution_count"),
            "successful_evolution_count": report.get("successful_evolution_count"),
            "safe_candidate_rate": report.get("safe_candidate_rate"),
            "average_light_eval_accuracy": report.get("average_light_eval_accuracy"),
            "failure_reason_counts": report.get("failure_reason_counts"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
