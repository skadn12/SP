"""Candidate-collection transport with a durable cross-process throttle.

The v41 candidate runner uses one instance for a live run, but retries and
process restarts must not reset the provider spacing rule.  This subclass keeps
the existing Caller payload checks, receipts, and 503/504 identical-payload
retry policy.  The shared live lock is held by the caller of ``call``; its
completion marker is therefore read and replaced while that lock is held.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import runner as r


COMPLETION_SCHEMA = "h4.r02.candidate_transport_completion.v1"


class Caller(r.Caller):
    """v41 Caller with shared completion spacing and a logical-call budget."""

    def __init__(self, run):
        super().__init__(run)
        manifest = r.base.read(run / "MANIFEST.json")
        self.logical_limit = manifest.get("logical_max", manifest.get("logical_call_max"))
        if isinstance(self.logical_limit, bool) or not isinstance(self.logical_limit, int) or self.logical_limit < 1:
            raise ValueError("candidate_logical_budget_missing")
        self.completion_path = self._completion_path()
        self._refresh_counters()

    @staticmethod
    def _lock_path() -> Path:
        return Path(r.LOCK_PATH)

    @classmethod
    def _completion_path(cls) -> Path:
        lock = cls._lock_path()
        return lock.with_name(lock.name + ".candidate_completion.json")

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    @staticmethod
    def _boot_id() -> str:
        # Linux monotonic time shares an epoch across processes in one boot.
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)

    def _read_completion(self) -> dict[str, Any] | None:
        path = self.completion_path
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("candidate_completion_marker_invalid") from error
        if (not isinstance(value, dict)
                or value.get("schema") != COMPLETION_SCHEMA
                or not isinstance(value.get("boot_id"), str)
                or any(type(value.get(key)) not in (int, float)
                       or not math.isfinite(value[key]) or value[key] < 0
                       for key in ("completed_at_epoch", "completed_at_monotonic"))):
            raise ValueError("candidate_completion_marker_invalid")
        return value

    def _write_completion(self) -> None:
        path = self.completion_path
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"schema": COMPLETION_SCHEMA, "completed_at_epoch": self._now(),
                 "completed_at_monotonic": self._monotonic(), "boot_id": self._boot_id(),
                 "run": str(Path(self.run).resolve()), "pid": os.getpid()}
        # Atomic replace permits audit/recovery to observe either complete JSON
        # document; the surrounding shared live lock serializes writers.
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", delete=False)
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            handle.close()
            if temporary.exists():
                temporary.unlink()

    def wait_before_http_attempt(self):
        marker = self._read_completion()
        now = self._monotonic()
        # Missing marker or a different boot has no trustworthy common time
        # origin.  A fresh full gap also protects the first v41 call after an
        # older runner that did not write this marker.
        if marker is None or marker["boot_id"] != self._boot_id():
            deadline = now + r.MIN_HTTP_GAP_SECONDS
        else:
            previous = marker["completed_at_monotonic"]
            deadline = (previous + r.MIN_HTTP_GAP_SECONDS
                        if previous <= now else now + r.MIN_HTTP_GAP_SECONDS)
        started = now
        # Recheck monotonic time if sleep wakes early; no single blocking wait
        # exceeds 35 seconds under the production policy.
        while self._monotonic() < deadline:
            self._sleep(min(r.MIN_HTTP_GAP_SECONDS, deadline - self._monotonic()))
        return self._monotonic() - started

    def record_http_completion(self):
        super().record_http_completion()
        self._write_completion()

    def _refresh_counters(self):
        """Instances sharing a run must not independently reset its budgets."""
        requests = self.http_counter
        logical = self.request_counter
        for directory in (Path(self.run) / "requests").glob("*"):
            if not directory.is_dir():
                continue
            prefix = directory.name.partition("-")[0]
            if not prefix.isdecimal():
                raise ValueError("candidate_request_directory_invalid")
            if (directory / "REQUEST.json").exists():
                logical = max(logical, int(prefix))
            for path in directory.glob("HTTP_ATTEMPT*.json"):
                value = r.base.read(path).get("http_attempt_number")
                if type(value) is not int or value < 1:
                    raise ValueError("candidate_http_counter_invalid")
                requests = max(requests, value)
        self.request_counter, self.http_counter = logical, requests

    def call(self, label, payload, proof):
        self._refresh_counters()
        if self.request_counter >= self.logical_limit:
            raise r.LocalRequestError("logical_budget_exhausted")
        if self.http_counter >= self.limit:
            raise r.LocalRequestError("global_http_budget_exhausted")
        return super().call(label, payload, proof)

