"""Compact one-window prompt for pure short-event extraction."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .local_event_contract import (
    EVIDENCE_MAX, LOCAL_EVENT_SCHEMA_VERSION, PER_SESSION_EVENT_MAX,
)


LOCAL_EVENT_PROMPT_VERSION = "simple_append_session_local_event_prompt_v1"


def build_local_event_prompt(window: Mapping[str, Any]) -> dict[str, Any]:
    transcript = "\n".join(
        f"{row['turn_ref']} | {row['speaker']} | {row['text']}"
        for row in window["turns"])
    template = {
        "local_event_schema_version": LOCAL_EVENT_SCHEMA_VERSION,
        "events": [{
            "event_summary": "one concrete short event from this session",
            "event_state": "the furthest state this event reaches in this session",
            "participants": ["exact speaker display name"],
            "evidence_turn_refs": ["exact turn_ref from this session"],
        }],
    }
    prompt = "\n".join([
        "ROLE: authoritative session-local Short Event Extractor.",
        "You see exactly one Parent session. Do not reason about any other session.",
        "Extract only concrete short events with potential future relevance: an action,",
        "plan, proposal, acceptance, decision, explicit state, attempted execution, completed",
        "execution, result, or specific information state directly stated in this window.",
        "For each event, event_state must record the furthest point it actually reaches in",
        "this window. Distinguish proposal, acceptance, decision, attempt, execution, and",
        "result. Do not upgrade one stage into another.",
        "Do not output broad themes or traits such as responsibility, support, friendship,",
        "personal growth, work ethic, or maturity. Do not output a recollection of an older",
        "event unless this window itself changes that event's concrete state. Do not merge",
        "independent incidents. Do not create continuation links or compare windows.",
        f"Return 0-{PER_SESSION_EVENT_MAX} events; there is no minimum quota.",
        f"Each event uses 1-{EVIDENCE_MAX} unique exact turn refs from this window only.",
        "Use exact speaker display names. Return JSON only with no extra fields.",
        "",
        f"WINDOW {window['window_id']} | {window['session_ref']}",
        transcript,
        "",
        "OUTPUT SHAPE (remove the placeholder event when no concrete event is reliable)",
        json.dumps(template, ensure_ascii=False, separators=(",", ":")),
    ])
    return {
        "prompt_version": LOCAL_EVENT_PROMPT_VERSION,
        "window_id": window["window_id"],
        "session_ref": window["session_ref"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "allowed_turn_refs": list(window["allowed_turn_refs"]),
    }


__all__ = ["LOCAL_EVENT_PROMPT_VERSION", "build_local_event_prompt"]
