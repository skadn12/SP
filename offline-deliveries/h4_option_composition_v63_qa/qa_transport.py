"""Bounded QA transport sharing v43's durable response-completion clock."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time

import qa as v15

base = v15.base
MODEL = v15.MODEL
GAP = 35.0
LOCK_PATH = Path(__file__).resolve().parent.parent / "h4_r02_source_unique_v2/runs/.unique_event_live.lock"
SCHEMA = "h4.r02.candidate_transport_completion.v1"


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def shared_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def marker_path():
    return LOCK_PATH.with_name(LOCK_PATH.name + ".candidate_completion.json")


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def read_marker():
    path = marker_path()
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    base.require(isinstance(value, dict) and value.get("schema") == SCHEMA
        and isinstance(value.get("boot_id"), str)
        and all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
                for k in ("completed_at_epoch", "completed_at_monotonic")), "shared_completion_marker_invalid")
    return value


def wait_gap():
    marker = read_marker()
    current = time.monotonic()
    previous = marker.get("completed_at_monotonic") if marker else None
    deadline = (previous + GAP if marker and marker["boot_id"] == boot_id()
                and previous <= current else current + GAP)
    started = current
    while time.monotonic() < deadline:
        time.sleep(min(GAP, deadline - time.monotonic()))
    return time.monotonic() - started, marker


def record_completion(run):
    path = marker_path()
    value = {"schema": SCHEMA, "completed_at_epoch": time.time(),
             "completed_at_monotonic": time.monotonic(), "boot_id": boot_id(),
             "run": str(Path(run).resolve()), "pid": os.getpid()}
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class Caller:
    """Called under shared_lock; no automatic transport replay for QA."""
    def __init__(self, run):
        self.run = Path(run)
        self.limit = base.read(self.run / "MANIFEST.json")["budget"]["http_attempts_max"]
        _, self.factory, self.annotate, self.capacity, dev = base.transport()
        self.tokenizer = dev.tokenizer_from(str(base.TOKENIZER))
        self.counter = 0
        self._refresh()

    def _refresh(self):
        for path in (self.run / "requests").glob("*/REQUEST.json"):
            number = path.parent.name.split("-", 1)[0]
            base.require(number.isdecimal(), "invalid_request_directory")
            self.counter = max(self.counter, int(number))
        self.http_count = len(list((self.run / "requests").glob("*/HTTP_ATTEMPT.json")))

    def call(self, label, payload, proof):
        self._refresh()
        base.require(self.counter < self.limit and self.http_count < self.limit, "http_budget_exhausted")
        base.require(payload.get("model") == MODEL, "unexpected_model")
        profile = deepcopy(base.read(self.run / "provider_profile.json"))
        profile["output_limit"] = payload["max_tokens"]
        profile["request_parameters"]["max_tokens"] = payload["max_tokens"]
        expected = {"model": profile["model"], "messages": payload["messages"],
                    **{k: v for k, v in profile["request_parameters"].items() if v is not None}}
        base.require(payload == expected, "request_profile_payload_mismatch")
        capacity = self.annotate(self.capacity(profile, self.tokenizer, payload["messages"], payload["max_tokens"]))
        self.counter += 1
        directory = self.run / "requests" / f"{self.counter:03d}-{label}"
        base.write(directory / "REQUEST.json", payload)
        base.write(directory / "PROOF.json", {**proof, "payload_sha256": base.digest(payload), "capacity": capacity})
        base.write(directory / "PROFILE.json", profile)
        if capacity["status"] != "PASS":
            base.write(directory / "ERROR.json", {"error_type": "LocalCapacityBlocked", "http_attempts": 0,
                       "no_truncation": True, "no_automatic_retry": True})
            return None, directory, {}
        base.write(directory / "RESERVATION.json", {"request_number": self.counter, "maximum": self.limit,
                   "reserved_at": now(), "automatic_retries": 0})
        waited, previous = wait_gap()
        started = False
        metadata = {}

        def on_attempt(actual):
            nonlocal started
            base.require(not started and actual == payload, "actual_payload_changed_or_replayed")
            self._refresh()
            base.require(self.http_count < self.limit, "http_budget_exhausted")
            self.http_count += 1
            base.write(directory / "HTTP_ATTEMPT.json", {"actual_payload_equals_frozen": True,
                "payload_sha256": base.digest(actual), "requested_model": MODEL,
                "http_attempt_number": self.http_count, "transport_retries": 0})
            base.write(directory / "THROTTLE.json", {"http_started_at_utc": now(),
                "minimum_gap_after_previous_response_seconds": GAP, "waited_seconds": waited,
                "previous_completion": previous})
            started = True

        try:
            response = self.factory(profile, on_http_attempt=on_attempt).complete(
                prompt=payload["messages"][0]["content"], max_tokens=payload["max_tokens"], json_object=True)
            metadata = response.metadata or {}
            base.write(directory / "RESPONSE.json", {"text": response.raw_text, "metadata": metadata})
            base.require(started, "response_without_http_receipt")
            base.require(metadata.get("returned_model") == MODEL, "returned_model_mismatch")
            base.require(metadata.get("finish_reason") == "stop", "response_incomplete")
            base.require(metadata.get("http_status", 200) == 200, "response_http_status_invalid")
            return response.raw_text, directory, metadata
        except Exception as error:
            meta = getattr(error, "metadata", None)
            if isinstance(meta, dict):
                metadata = meta
            base.write(directory / "ERROR.json", {"error_type": type(error).__name__,
                "http_attempted": started, "metadata": metadata, "no_automatic_retry": True,
                "raw_exception_text_saved": False})
            return None, directory, metadata
        finally:
            if started:
                base.write(directory / "HTTP_RESULT.json", {"http_completed_at_utc": now(),
                    "http_status": metadata.get("http_status"),
                    "outcome": "error" if (directory / "ERROR.json").exists() else "response"})
                record_completion(self.run)
