#!/usr/bin/env python3
"""Strict event classification for the resume pipeline.

The Codex CLI emits two item.completed/error diagnostics for ignored local
configuration settings.  Those exact diagnostics are nonsemantic startup
noise; every other error or tool-like item is a failure.  Classification
results retain the complete input event list so callers can persist evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "fgr20_event_policy_v1"
EXACT_STARTUP_WARNING = (
    "Codex is ignoring 3 unrecognized configuration settings. Check for typos or deprecated settings.\n"
    "  user (/root/.codex/config.toml): `disable_response_storage` is ignored.\n"
    "  user (/root/.codex/config.toml): `network_access` is ignored.\n"
    "  user (/root/.codex/config.toml): `windows_wsl_setup_acknowledged` is ignored."
)

_SAFE_TOP_LEVEL = {"thread.started", "turn.started", "turn.completed"}
_SAFE_ITEM_TYPES = {"reasoning", "agent_message"}


class EventInputError(ValueError):
    """Raised only when input cannot be represented as JSON events."""


def _parse_events(events: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize a JSONL string or a sequence of event mappings.

    Raw lines are returned as evidence even when a line is malformed.  A
    mapping sequence is copied before classification so the caller's objects
    are never mutated.
    """

    if isinstance(events, str):
        raw_lines = events.splitlines()
        if not raw_lines:
            raise EventInputError("missing JSON event stream")
        parsed: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw_lines, 1):
            if not line.strip():
                raise EventInputError(f"blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EventInputError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise EventInputError(f"event on line {line_number} is not an object")
            parsed.append(dict(value))
        return parsed, raw_lines

    if events is None:
        raise EventInputError("missing JSON event stream")
    if isinstance(events, Mapping):
        raise EventInputError("event stream must be a sequence, not one object")
    try:
        values = list(events)
    except TypeError as exc:
        raise EventInputError("event stream is not iterable") from exc
    if not values:
        raise EventInputError("missing JSON event stream")
    parsed = []
    raw_lines = []
    for index, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            raise EventInputError(f"event {index} is not an object")
        copied = dict(value)
        parsed.append(copied)
        raw_lines.append(json.dumps(copied, ensure_ascii=False, sort_keys=True))
    return parsed, raw_lines


def _result(
    *,
    accepted: bool,
    classification: str,
    events: list[dict[str, Any]],
    raw_lines: list[str],
    warnings: list[dict[str, Any]],
    agent_text: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted": accepted,
        "decision": "PASS" if accepted else "REJECT",
        "classification": classification,
        "raw_events": events,
        "raw_event_lines": raw_lines,
        "evidence": {
            "allowlisted_startup_warnings": warnings,
            "agent_text_events": agent_text,
            "rejected_events": rejected,
        },
        "failure_reasons": reasons,
    }


def classify_events(
    events: Any,
    *,
    timed_out: bool = False,
    backend_error: bool = False,
) -> dict[str, Any]:
    """Classify a CLI event stream and retain all evidence.

    ``events`` can be a JSONL string or a sequence of dictionaries.  A pass
    requires at least one agent_message and permits only known lifecycle,
    reasoning, and agent-message events plus the exact startup warning.  The
    optional flags make timeout/backend failures explicit for callers that
    learn them outside the event stream.
    """

    try:
        parsed, raw_lines = _parse_events(events)
    except EventInputError as exc:
        return _result(
            accepted=False,
            classification="invalid_or_missing_json",
            events=[],
            raw_lines=[] if events is None else ([events] if isinstance(events, str) else []),
            warnings=[],
            agent_text=[],
            rejected=[],
            reasons=[str(exc)],
        )

    warnings: list[dict[str, Any]] = []
    agent_text: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []

    if timed_out:
        reasons.append("timeout")
    if backend_error:
        reasons.append("backend_error")

    for index, event in enumerate(parsed):
        event_type = event.get("type")
        if not isinstance(event_type, str):
            rejected.append(event)
            reasons.append(f"event_{index}: missing_or_invalid_type")
            continue

        # Top-level error events are never the allowlisted startup diagnostic.
        if event_type == "error":
            rejected.append(event)
            reasons.append(f"event_{index}: top_level_stream_error")
            continue

        if event_type in _SAFE_TOP_LEVEL:
            continue

        if event_type != "item.completed":
            rejected.append(event)
            reasons.append(f"event_{index}: unknown_event_type:{event_type}")
            continue

        item = event.get("item")
        if not isinstance(item, Mapping):
            rejected.append(event)
            reasons.append(f"event_{index}: missing_item_object")
            continue
        item_type = item.get("type")
        if item_type == "error" and item.get("message") == EXACT_STARTUP_WARNING:
            warnings.append(event)
            continue
        if item_type in _SAFE_ITEM_TYPES and isinstance(item.get("text"), str):
            agent_text.append(event) if item_type == "agent_message" else None
            continue

        rejected.append(event)
        if item_type == "error":
            reasons.append(f"event_{index}: unknown_error_item")
        elif item_type in {"command_execution", "command", "tool", "function_call"}:
            reasons.append(f"event_{index}: forbidden_tool_or_command_item:{item_type}")
        else:
            reasons.append(f"event_{index}: unknown_or_malformed_item:{item_type!r}")

    if not agent_text:
        reasons.append("no_agent_message")

    accepted = not reasons
    classification = (
        "allowlisted_exact_codex_startup_diagnostics"
        if accepted and warnings
        else "clean_known_event_stream"
        if accepted
        else "rejected"
    )
    return _result(
        accepted=accepted,
        classification=classification,
        events=parsed,
        raw_lines=raw_lines,
        warnings=warnings,
        agent_text=agent_text,
        rejected=rejected,
        reasons=reasons,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="JSONL event file")
    parser.add_argument("--timeout", action="store_true")
    parser.add_argument("--backend-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = args.events.read_text(encoding="utf-8")
    except OSError as exc:
        result = _result(
            accepted=False,
            classification="invalid_or_missing_json",
            events=[],
            raw_lines=[],
            warnings=[],
            agent_text=[],
            rejected=[],
            reasons=[f"cannot_read_events:{exc}"],
        )
    else:
        result = classify_events(payload, timed_out=args.timeout, backend_error=args.backend_error)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
