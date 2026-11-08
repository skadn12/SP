#!/usr/bin/env python3
"""Self-check the resume event policy with four adversarial fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from event_policy import EXACT_STARTUP_WARNING, classify_events  # noqa: E402


def _event(item_type: str, **item: object) -> dict[str, object]:
    payload = {"type": item_type}
    payload.update(item)
    return {"type": "item.completed", "item": payload}


def _base_prefix() -> list[dict[str, object]]:
    return [
        {"type": "thread.started", "thread_id": "fixture-thread"},
        {"type": "turn.started"},
    ]


def run() -> dict[str, object]:
    fixtures = {
        "exact_warning_plus_agent_text": _base_prefix()
        + [
            _event("error", id="item_0", message=EXACT_STARTUP_WARNING),
            _event("agent_message", id="item_1", text='{"decision":"PASS","evidence":"quoted fact"}'),
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ],
        "command_tool_rejected": _base_prefix()
        + [
            _event("agent_message", id="item_0", text="working"),
            _event("command_execution", id="item_1", command="echo forbidden"),
        ],
        "unknown_error_item_rejected": _base_prefix()
        + [
            _event("agent_message", id="item_0", text="working"),
            _event("error", id="item_1", message="some other error"),
        ],
        "top_level_reconnect_rejected": _base_prefix()
        + [
            {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected before completion)"},
            _event("agent_message", id="item_0", text="late text"),
        ],
    }
    expected = {
        "exact_warning_plus_agent_text": True,
        "command_tool_rejected": False,
        "unknown_error_item_rejected": False,
        "top_level_reconnect_rejected": False,
    }
    checks: dict[str, object] = {}
    all_passed = True
    for name, events in fixtures.items():
        result = classify_events(events)
        ok = result["accepted"] is expected[name]
        all_passed = all_passed and ok
        checks[name] = {
            "expected_accepted": expected[name],
            "actual_accepted": result["accepted"],
            "check_passed": ok,
            "classification": result["classification"],
            "failure_reasons": result["failure_reasons"],
            "raw_event_count": len(result["raw_events"]),
        }
    return {
        "schema_version": "fgr20_event_policy_self_check_v1",
        "passed": all_passed,
        "fixtures": checks,
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
