"""Verified asynchronous Memorax Add completion handling."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .write_state import (
    apply_add_response,
    apply_query_uncertainty,
    apply_task_response,
    new_write_state,
)


ASYNC_WRITE_VERSION = "simple_append_memorax_async_write_v2"
# About 4m50s total: fast visibility for small writes, then a bounded 20s
# cadence for slower extraction/summary/bucket jobs.
DEFAULT_POLL_DELAYS_SECONDS = (
    1.0, 2.0, 4.0, 8.0, 15.0,
    20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
    20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
)


class AsyncAddGateway(Protocol):
    def add(self, payload: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        ...

    def add_status(self, task_id: str, *, request_id: str) -> dict[str, Any]:
        ...


class MemoraxAsyncWriteError(RuntimeError):
    """An accepted write did not reach a verified successful state."""


WriteStateCallback = Callable[[Mapping[str, Any]], None]


def _notify(
    callback: WriteStateCallback | None, state: Mapping[str, Any],
) -> None:
    if callback is not None:
        callback(dict(state))


def submit_add_without_wait(
    *, gateway: AsyncAddGateway, intent: Mapping[str, Any],
    stage: str = "memory_add",
    on_state_change: WriteStateCallback | None = None,
) -> dict[str, Any]:
    """Submit exactly once and persist enough state for later reconciliation."""
    payload = intent.get("payload")
    request_id = str(intent.get("request_id") or "")
    if not isinstance(payload, Mapping) or not request_id:
        raise ValueError("memorax_async_intent_invalid")
    state = new_write_state(intent=intent, stage=stage)
    _notify(on_state_change, state)
    response = gateway.add(payload, request_id=request_id)
    state = apply_add_response(state, response)
    _notify(on_state_change, state)
    status = str(state.get("task_status") or "")
    if status not in {"accepted", "completed"}:
        raise MemoraxAsyncWriteError(
            f"memorax_async_add_not_accepted:{status or 'missing'}")
    return {
        "async_write_version": ASYNC_WRITE_VERSION,
        "request_id": request_id,
        "task_id": state.get("task_id"),
        "initial_status": status,
        "final_status": "completed" if status == "completed" else "pending",
        "poll_count": 0,
        "verified_complete": status == "completed",
        "pending": status == "accepted",
        "write_state": state,
    }


def _data(value: Mapping[str, Any]) -> Mapping[str, Any]:
    data = value.get("data")
    return data if isinstance(data, Mapping) else {}


def _status(value: Mapping[str, Any]) -> str:
    data = _data(value)
    return str(data.get("status") or value.get("status") or "")


def _task_id(value: Mapping[str, Any]) -> str:
    return str(_data(value).get("task_id") or value.get("task_id") or "")


def submit_add_and_wait(
    *, gateway: AsyncAddGateway, intent: Mapping[str, Any],
    before_status_request: Callable[[str, str], None] | None = None,
    poll_delays_seconds: Sequence[float] = DEFAULT_POLL_DELAYS_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    stage: str = "memory_add",
    on_state_change: WriteStateCallback | None = None,
) -> dict[str, Any]:
    """Submit one Add and return only after completion is verified.

    A synchronous ``completed`` response is accepted for compatibility.  An
    asynchronous ``accepted`` response must include a task ID and reach
    ``success``/``completed`` through the documented status endpoint.  Failed,
    unavailable, malformed, or exhausted status checks never fall back to a
    second Add because the first submission may already be durable.
    """
    payload = intent.get("payload")
    request_id = str(intent.get("request_id") or "")
    if not isinstance(payload, Mapping) or not request_id:
        raise ValueError("memorax_async_intent_invalid")
    state = new_write_state(intent=intent, stage=stage)
    _notify(on_state_change, state)
    response = gateway.add(payload, request_id=request_id)
    state = apply_add_response(state, response)
    _notify(on_state_change, state)
    initial_status = _status(response)
    task_id = _task_id(response)
    if initial_status == "completed":
        return {
            "async_write_version": ASYNC_WRITE_VERSION,
            "request_id": request_id,
            "task_id": task_id or None,
            "initial_status": initial_status,
            "final_status": "completed",
            "poll_count": 0,
            "verified_complete": True,
            "write_state": state,
        }
    if initial_status != "accepted" or not task_id:
        raise MemoraxAsyncWriteError(
            f"memorax_async_add_not_accepted:{initial_status or 'missing'}")
    delays = tuple(float(value) for value in poll_delays_seconds)
    if not delays or any(value < 0 for value in delays):
        raise ValueError("memorax_poll_delays_invalid")
    observed: list[str] = []
    for index, delay in enumerate(delays, 1):
        sleep_fn(delay)
        status_request_id = f"{request_id}-status-{index:02d}"
        if before_status_request is not None:
            before_status_request("add_status", status_request_id)
        try:
            status_response = gateway.add_status(
                task_id, request_id=status_request_id)
        except Exception as error:
            state = apply_query_uncertainty(
                state, error=f"{type(error).__name__}:{error}")
            _notify(on_state_change, state)
            raise
        current = _status(status_response)
        state = apply_task_response(state, status_response)
        _notify(on_state_change, state)
        observed.append(current)
        if current in {"success", "completed"}:
            return {
                "async_write_version": ASYNC_WRITE_VERSION,
                "request_id": request_id,
                "task_id": task_id,
                "initial_status": initial_status,
                "final_status": current,
                "poll_count": index,
                "observed_statuses": observed,
                "verified_complete": True,
                "write_state": state,
            }
        if current == "failed":
            error = _data(status_response).get("error") or status_response.get("error")
            raise MemoraxAsyncWriteError(
                f"memorax_async_add_failed:{type(error).__name__ if error else 'unknown'}")
        if current != "processing":
            raise MemoraxAsyncWriteError(
                f"memorax_async_status_unexpected:{current or 'missing'}")
    state = apply_query_uncertainty(
        state, error=f"memorax_async_add_poll_exhausted:{len(observed)}")
    _notify(on_state_change, state)
    raise MemoraxAsyncWriteError(
        f"memorax_async_add_poll_exhausted:{len(observed)}")
