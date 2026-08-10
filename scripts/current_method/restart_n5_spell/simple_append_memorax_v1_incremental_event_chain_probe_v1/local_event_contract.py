"""Pure session-local short-event extraction contract for Stage A."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


LOCAL_EVENT_SCHEMA_VERSION = "simple_append_session_local_events_v1"
PER_SESSION_EVENT_MAX = 4
EVIDENCE_MAX = 6
_TOP_KEYS = {"local_event_schema_version", "events"}
_EVENT_KEYS = {
    "event_summary", "event_state", "participants", "evidence_turn_refs",
}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def compile_parent_session_windows(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    by_session: dict[str, dict[str, Any]] = {}
    for turn in parent["dialogue"]["turns"]:
        session_ref = str(turn["session_id"])
        if session_ref not in by_session:
            window = {
                "window_id": f"P{len(windows) + 1:02d}",
                "session_ref": session_ref,
                "turns": [],
            }
            by_session[session_ref] = window
            windows.append(window)
        by_session[session_ref]["turns"].append({
            "turn_ref": str(turn["turn_id"]),
            "speaker": str(turn["speaker"]),
            "text": str(turn["text"]),
        })
    for window in windows:
        window["turn_count"] = len(window["turns"])
        window["allowed_turn_refs"] = [row["turn_ref"] for row in window["turns"]]
    return windows


def validate_local_event_output(
    value: Any, *, window: Mapping[str, Any], parent_speakers: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _TOP_KEYS:
        return {"valid": False, "errors": ["local_event_top_level_schema"],
                "event_count": 0}
    if value.get("local_event_schema_version") != LOCAL_EVENT_SCHEMA_VERSION:
        errors.append("local_event_schema_version")
    events = value.get("events")
    if not isinstance(events, list) or len(events) > PER_SESSION_EVENT_MAX:
        errors.append("local_event_count_0_to_4")
        events = []
    allowed_refs = set(str(ref) for ref in window["allowed_turn_refs"])
    speakers = set(str(row) for row in parent_speakers)
    summaries: set[str] = set()
    for index, event in enumerate(events, 1):
        prefix = f"event_{index:02d}"
        if not isinstance(event, Mapping) or set(event) != _EVENT_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        for key in ("event_summary", "event_state"):
            if not _text(event.get(key)):
                errors.append(f"{prefix}:{key}")
        summary = " ".join(str(event.get("event_summary") or "").split()).casefold()
        if summary in summaries:
            errors.append(f"{prefix}:duplicate_summary")
        summaries.add(summary)
        participants = event.get("participants")
        if (not isinstance(participants, list) or
                not 1 <= len(participants) <= 5 or
                any(not _text(row) for row in participants) or
                len(participants) != len(set(participants))):
            errors.append(f"{prefix}:participants")
        elif any(str(row) not in speakers for row in participants):
            errors.append(f"{prefix}:participant_not_parent_speaker")
        refs = event.get("evidence_turn_refs")
        if (not isinstance(refs, list) or not 1 <= len(refs) <= EVIDENCE_MAX or
                any(not _text(ref) for ref in refs) or
                len(refs) != len(set(refs))):
            errors.append(f"{prefix}:evidence_turn_refs")
        elif any(str(ref) not in allowed_refs for ref in refs):
            errors.append(f"{prefix}:evidence_outside_local_window")
    return {"valid": not errors, "errors": errors, "event_count": len(events)}


def materialize_local_candidates(
    outputs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for window, output in outputs:
        for index, event in enumerate(output.get("events") or [], 1):
            candidates.append({
                "candidate_ref": f"{window['window_id']}:E{index:02d}",
                "window_id": window["window_id"],
                "session_ref": window["session_ref"],
                "event_summary": str(event["event_summary"]).strip(),
                "event_state": str(event["event_state"]).strip(),
                "participants": [str(row) for row in event["participants"]],
                "evidence_turn_refs": [str(ref) for ref in event["evidence_turn_refs"]],
            })
    return candidates


def _tokens(value: Any) -> set[str]:
    replacements = {
        "agreed": "agree", "accepts": "accept", "accepted": "accept",
        "coverage": "cover", "covered": "cover", "covering": "cover",
    }
    return {replacements.get(token, token) for token in re.findall(
        r"[a-z0-9]+", str(value or "").casefold())}


def audit_episode_146700_local_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_window: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_window.setdefault(str(row["window_id"]), []).append(row)
    expected = {
        "soccer": ("P01", {
            "cc-train-episode-146700:P01:T006",
            "cc-train-episode-146700:P01:T007",
        }),
        "new_cashier": ("P02", {
            "cc-train-episode-146700:P02:T001",
            "cc-train-episode-146700:P02:T003",
        }),
        "friend_apology": ("P03", {
            "cc-train-episode-146700:P03:T003",
            "cc-train-episode-146700:P03:T005",
        }),
        "health_coverage": ("P04", {
            "cc-train-episode-146700:P04:T005",
            "cc-train-episode-146700:P04:T006",
        }),
        "report_mistake": ("P05", {
            "cc-train-episode-146700:P05:T001",
            "cc-train-episode-146700:P05:T003",
            "cc-train-episode-146700:P05:T009",
        }),
    }
    matches: dict[str, str | None] = {}
    errors: list[str] = []
    for name, (window_id, required_refs) in expected.items():
        match = next((row for row in by_window.get(window_id, [])
                      if required_refs.issubset(set(row["evidence_turn_refs"]))), None)
        matches[name] = str(match["candidate_ref"]) if match else None
        if match is None:
            errors.append(f"required_local_event_missing:{name}")
            continue
        state_tokens = _tokens(match["event_state"])
        if name == "soccer" and not ({"agree", "accept"} & state_tokens):
            errors.append("soccer_latest_acceptance_missing")
        if name == "health_coverage" and "cover" not in state_tokens:
            errors.append("health_accepted_coverage_missing")
        if name == "report_mistake":
            state = " ".join(str(match["event_state"]).split()).casefold()
            if any(claim in state for claim in (
                "reported the mistake", "explained the mistake",
                "informed the boss", "boss was informed", "report was corrected",
            )):
                errors.append("report_execution_overstated")
    recalled_soccer_refs = {
        f"cc-train-episode-146700:P02:T{index:03d}" for index in range(7, 11)
    }
    for row in by_window.get("P02", []):
        refs = set(row["evidence_turn_refs"])
        if refs and refs.issubset(recalled_soccer_refs):
            errors.append(f"P02:recollected_soccer_emitted:{row['candidate_ref']}")
    return {
        "valid": not errors,
        "errors": errors,
        "expected_event_matches": matches,
        "candidate_count_by_window": {
            window_id: len(by_window.get(window_id, []))
            for window_id in ("P01", "P02", "P03", "P04", "P05")
        },
        "linking_performed": False,
        "scope": "episode_146700_stage_a_only",
    }


__all__ = [
    "EVIDENCE_MAX", "LOCAL_EVENT_SCHEMA_VERSION", "PER_SESSION_EVENT_MAX",
    "audit_episode_146700_local_candidates", "compile_parent_session_windows",
    "materialize_local_candidates", "validate_local_event_output",
]
