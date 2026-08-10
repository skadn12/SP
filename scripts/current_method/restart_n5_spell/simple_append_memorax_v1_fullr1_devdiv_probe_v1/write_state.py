"""Deterministic state transitions for recoverable Memorax Add intents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


WRITE_STATE_VERSION = "simple_append_memorax_write_state_v1"
WRITE_STATUSES = frozenset({
    "not_submitted",
    "accepted_pending",
    "completed_visibility_pending",
    "verified",
    "failed_retryable",
    "ambiguous",
    "permanent_failed",
})


def _payload_sha(intent: Mapping[str, Any]) -> str:
    value = str(intent.get("payload_sha256") or "").strip()
    if value:
        return value
    payload = json.dumps(
        intent.get("payload") or {}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _response_data(value: Mapping[str, Any]) -> Mapping[str, Any]:
    data = value.get("data")
    return data if isinstance(data, Mapping) else {}


def _response_status(value: Mapping[str, Any]) -> str:
    return str(_response_data(value).get("status") or value.get("status") or "")


def _response_task_id(value: Mapping[str, Any]) -> str:
    return str(_response_data(value).get("task_id") or value.get("task_id") or "")


def new_write_state(
    *, intent: Mapping[str, Any], stage: str, attempt_index: int = 1,
) -> dict[str, Any]:
    request_id = str(intent.get("request_id") or "").strip()
    if not request_id:
        raise ValueError("memorax_write_state_request_id_required")
    if int(attempt_index) < 1:
        raise ValueError("memorax_write_state_attempt_index_invalid")
    return {
        "write_state_version": WRITE_STATE_VERSION,
        "write_intent_id": request_id,
        "attempt_id": f"{request_id}:attempt:{int(attempt_index):02d}",
        "attempt_index": int(attempt_index),
        "stage": str(stage),
        "request_id": request_id,
        "payload_sha256": _payload_sha(intent),
        "task_id": None,
        "status": "not_submitted",
        "task_status": None,
        "add_submitted": False,
        "memory_write_verified": False,
        "bucket_visibility_verified": False,
        "automatic_retry_allowed": False,
        "last_error": None,
    }


def apply_add_response(
    state: Mapping[str, Any], response: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(state)
    status = _response_status(response)
    task_id = _response_task_id(response) or None
    result.update({
        "add_submitted": True,
        "task_id": task_id,
        "task_status": status or None,
        "last_error": None,
    })
    if status == "accepted" and task_id:
        result["status"] = "accepted_pending"
    elif status == "completed":
        result.update({
            "status": "completed_visibility_pending",
            "memory_write_verified": True,
        })
    else:
        result.update({
            "status": "ambiguous",
            "last_error": f"add_response_unexpected:{status or 'missing'}",
        })
    return result


def apply_task_response(
    state: Mapping[str, Any], response: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(state)
    status = _response_status(response)
    result["task_status"] = status or None
    result["last_error"] = None
    if status == "processing":
        result["status"] = "accepted_pending"
    elif status in {"success", "completed"}:
        result.update({
            "status": "completed_visibility_pending",
            "memory_write_verified": True,
        })
    elif status == "failed":
        # A failed task is not proof that no partial write exists.
        result.update({
            "status": "ambiguous",
            "memory_write_verified": False,
            "last_error": "task_failed_write_presence_unknown",
        })
    else:
        result.update({
            "status": "ambiguous",
            "memory_write_verified": False,
            "last_error": f"task_status_unexpected:{status or 'missing'}",
        })
    return result


def apply_query_uncertainty(
    state: Mapping[str, Any], *, error: str,
) -> dict[str, Any]:
    result = dict(state)
    result["last_error"] = str(error)
    if result.get("task_id"):
        result["status"] = "accepted_pending"
    else:
        result["status"] = "ambiguous"
    return result


def apply_visibility_result(
    state: Mapping[str, Any], *, visible: bool,
    absence_confirmed: bool = False,
) -> dict[str, Any]:
    result = dict(state)
    if visible:
        result.update({
            "status": "verified",
            "memory_write_verified": True,
            "bucket_visibility_verified": True,
            "automatic_retry_allowed": False,
            "last_error": None,
        })
        return result
    result["bucket_visibility_verified"] = False
    if result.get("memory_write_verified"):
        # A completed task may precede indexing. Never resubmit it.
        result.update({
            "status": "completed_visibility_pending",
            "automatic_retry_allowed": False,
        })
    elif absence_confirmed and result.get("task_status") == "failed":
        result.update({
            "status": "failed_retryable",
            "automatic_retry_allowed": True,
            "last_error": "task_failed_and_write_confirmed_absent",
        })
    else:
        result.update({
            "status": "ambiguous",
            "automatic_retry_allowed": False,
        })
    return result


def retry_allowed(state: Mapping[str, Any]) -> bool:
    return (
        state.get("status") == "failed_retryable"
        and state.get("automatic_retry_allowed") is True
    )
