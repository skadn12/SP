"""Deterministic, secret-free Memorax request payload compilation."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes


MEMORAX_ADD_INTENT_VERSION = "simple_append_memorax_add_intent_v1"
MEMORAX_BUCKET_QUERY_VERSION = "simple_append_memorax_bucket_query_v1"
MEMORAX_APP_ID = "gxy-selfplay-fullr1-devdiv-probe-v1"
MEMORAX_AGENT_ID = "simple-append-fullr1-devdiv-probe-v1"
MEMORAX_USER_NAMESPACE = "gxy-selfplay:fullr1-devdiv-probe-v1"
MAX_MESSAGE_CHARACTERS = 12_000
MAX_WINDOW_MESSAGES = 50
MAX_WINDOW_CHARACTERS = 10_000


def memorax_user_id(root_id: str) -> str:
    """Return a stable per-root owner so unrelated stories never share Buckets."""
    value = str(root_id or "").strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError("memorax_root_id_invalid")
    return f"{MEMORAX_USER_NAMESPACE}:{value}"


def _turn_view(row: Mapping[str, Any]) -> dict[str, str]:
    turn_ref = str(row.get("turn_ref") or row.get("turn_id") or "").strip()
    session_ref = str(
        row.get("session_ref") or row.get("session_id") or "").strip()
    speaker = str(row.get("speaker") or "").strip()
    text = str(row.get("text") or "").strip()
    if not turn_ref or not speaker or not text:
        raise ValueError("memorax_turn_missing_required_field")
    content = f"[Speaker: {speaker}] {text}"
    if len(content) > MAX_MESSAGE_CHARACTERS:
        raise ValueError(f"memorax_message_too_long:{turn_ref}")
    return {
        "turn_ref": turn_ref,
        "session_ref": session_ref,
        "speaker": speaker,
        "text": text,
        "content": content,
    }


def _window_context(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    characters = 0
    for offset, row in enumerate(reversed(list(rows))):
        view = _turn_view(row)
        size = len(view["content"])
        if len(selected) >= MAX_WINDOW_MESSAGES or characters + size > \
                MAX_WINDOW_CHARACTERS:
            break
        selected.append({
            "role": "user" if offset % 2 == 0 else "assistant",
            "content": view["content"],
            "timestamp": -1,
        })
        characters += size
    selected.reverse()
    return selected


def build_add_intent(
    *, root_id: str, round_index: int, source_dialogue_sha256: str,
    turns: Sequence[Mapping[str, Any]],
    recent_context_turns: Sequence[Mapping[str, Any]] = (),
    async_mode: bool = False,
) -> dict[str, Any]:
    """Compile one accepted dialogue increment into a Memorax Add intent.

    Round zero represents the authoritative Parent initialization. Later calls
    must contain only the accepted Child turns from that round, never the full
    accumulated transcript or any QA/Solver artifact.
    """
    if int(round_index) < 0:
        raise ValueError("memorax_round_index_invalid")
    source_sha = str(source_dialogue_sha256 or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("memorax_source_dialogue_sha256_invalid")
    normalized = [_turn_view(row) for row in turns]
    if not normalized:
        raise ValueError("memorax_add_turns_required")
    refs = [row["turn_ref"] for row in normalized]
    if len(refs) != len(set(refs)):
        raise ValueError("memorax_add_turn_ref_duplicate")
    root = str(root_id).strip()
    user_id = memorax_user_id(root)
    round_ref = f"R{int(round_index):02d}"
    session_id = f"{root}:memorax:{round_ref}"
    messages = [{
        "role": "user" if index % 2 == 0 else "assistant",
        "timestamp": -1,
        "content": row["content"],
    } for index, row in enumerate(normalized)]
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "agent_id": MEMORAX_AGENT_ID,
        "app_id": MEMORAX_APP_ID,
        "mode": "default",
        "async_mode": bool(async_mode),
        "return_raw_text": False,
        "timestamp": -1,
        "expiration_date": None,
        "metadata": {
            "source": "simple-append-memorax-fullr1-devdiv-probe-v1",
            "root_id": root,
            "round_index": int(round_index),
            "source_dialogue_sha256": source_sha,
            "first_turn_ref": refs[0],
            "last_turn_ref": refs[-1],
            "turn_count": len(refs),
            "accepted_dialogue_only": True,
        },
        "window_context": _window_context(recent_context_turns),
        "messages": messages,
    }
    payload_sha = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return {
        "intent_version": MEMORAX_ADD_INTENT_VERSION,
        "root_id": root,
        "round_index": int(round_index),
        "user_id": user_id,
        "session_id": session_id,
        "source_dialogue_sha256": source_sha,
        "turn_refs": refs,
        "payload": payload,
        "payload_sha256": payload_sha,
        "request_id": f"memorax-add-{root}-{round_ref.lower()}-{payload_sha[:12]}",
        "api_key_in_artifact": False,
    }


def build_bucket_search_payload(
    *, root_id: str, limit: int = 20, max_buckets_per_type: int = 100,
) -> dict[str, Any]:
    if not 1 <= int(limit) <= 100:
        raise ValueError("memorax_bucket_limit_out_of_range")
    if not 1 <= int(max_buckets_per_type) <= 500:
        raise ValueError("memorax_bucket_count_out_of_range")
    return {
        "query_version": MEMORAX_BUCKET_QUERY_VERSION,
        "root_id": str(root_id),
        "payload": {
            "user_id": memorax_user_id(root_id),
            "return_raw_text": False,
            "bucket_filters": {
                "entity_names": "all",
                "location_names": "all",
                "event_names": "all",
                "item_names": "all",
                "dreaming_names": "all",
                "limit": int(limit),
                "max_buckets_per_type": int(max_buckets_per_type),
            },
        },
        "api_key_in_artifact": False,
    }
