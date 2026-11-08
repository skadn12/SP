"""Single authorized 64-token synthetic probe; never a QA generation runner.

The frozen v2 transport/ledger require4096 and cannot honestly record this
probe. A separate runtime budget takeover blocks the old canonical v2 entry
point and owns the same original12-attempt budget. No retry or QA path exists.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import sqlite3

from . import qa_api_transport_v2 as old_transport

PROJECT = Path(__file__).resolve().parents[2]
VERSION = "base_qa_capability_probe_20261029_v1"
ARTIFACTS = PROJECT / "research-protocols/selfplay_week_20261019" / VERSION
V2 = ARTIFACTS.parent / "base_qa_executor_v2_20261029_v1"
CANONICAL = V2 / "live_state"
LEDGER_NAME = "GLOBAL_12_ATTEMPTS.sqlite3"
PURPOSE = "SYNTHETIC_CAPABILITY_PROBE_64"
TOTAL_LIMIT = 12
PROBE_LIMIT = 64
EXPECTED_TEXT = "OK"


class ProbeHold(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise ProbeHold(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_binding(path):
    p = Path(path).resolve()
    raw = p.read_bytes()
    return {"path": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path):
    return old_transport._strict_json(Path(path).read_bytes())


def durable_write(path, value, *, binary=False):
    path = Path(path)
    raw = value if binary else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def body():
    return {"model": old_transport.MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly OK. Do not add any other text."}],
            "temperature": 0.1, "max_tokens": PROBE_LIMIT, "n": 1, "stream": False}


def validate_body(value):
    require(value == body() and type(value.get("max_tokens")) is int and
            type(value.get("n")) is int and value.get("stream") is False,
            "Unexpected probe payload; no network request allowed")


def takeover_value(experiment_binding, canonical_dir):
    return {"version": VERSION, "original_experiment": "base_qa_executor_v2_20261029_v1",
            "purpose": "GLOBAL_BUDGET_TAKEOVER_PROBE_ONLY_QA_HOLD",
            "global_maximum_attempts": TOTAL_LIMIT, "probe_maximum_attempts": 1,
            "probe_max_tokens": PROBE_LIMIT, "experiment_binding": experiment_binding,
            "canonical_directory": str(Path(canonical_dir).resolve()),
            "authoritative_ledger": str(Path(canonical_dir).resolve() / LEDGER_NAME),
            "legacy_v2_initialization_forbidden": True, "qa_authorized": False,
            "note": "Not a v2 EXPERIMENT. Do not delete/reinitialize. Future QA must inherit this ledger; no root call is fabricated."}


class ProbeBudget:
    """Durable global12 ledger with only one callable probe reservation path."""
    def __init__(self, canonical_dir, experiment_binding, *, install=False):
        self.directory = Path(canonical_dir).resolve()
        self.expected_guard = takeover_value(experiment_binding, self.directory)
        if install:
            # An exclusive new runtime guard deliberately prevents frozen v2
            # initialize() from creating a separate fresh12-attempt budget.
            self.directory.mkdir(parents=True, exist_ok=False)
            parent = os.open(self.directory.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            durable_write(self.directory / "BUDGET_TAKEOVER.json", self.expected_guard)
        require(read(self.directory / "BUDGET_TAKEOVER.json") == self.expected_guard, "Budget takeover drift")
        self.path = self.directory / LEDGER_NAME
        require(not self.path.is_symlink(), "Ledger may not be a symlink")
        existed = self.path.exists()
        require(install or existed, "Existing global ledger missing; never recreate/reset")
        self.connection = sqlite3.connect(str(self.path), timeout=15, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA busy_timeout=15000")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            tables = {r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            policy = {"schema": VERSION, "original_global_limit": TOTAL_LIMIT,
                      "experiment_binding": experiment_binding, "canonical_directory": str(self.directory)}
            if not existed:
                require(not tables, "Unexpected new ledger schema")
                self.connection.execute("CREATE TABLE policy (singleton INTEGER PRIMARY KEY CHECK(singleton=1), value TEXT NOT NULL)")
                self.connection.execute("INSERT INTO policy VALUES (1,?)", (canonical(policy),))
                self.connection.execute("CREATE TABLE attempts (attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, purpose TEXT NOT NULL UNIQUE, status TEXT NOT NULL CHECK(status IN ('RESERVED','SUCCEEDED','REJECTED','UNCERTAIN')), reserved_utc TEXT NOT NULL, settled_utc TEXT, request_sha256 TEXT NOT NULL, max_tokens INTEGER NOT NULL, receipt TEXT)")
                self.connection.execute("CREATE TRIGGER global12 BEFORE INSERT ON attempts WHEN (SELECT COUNT(*) FROM attempts)>=12 BEGIN SELECT RAISE(ABORT,'GLOBAL_12_ATTEMPT_LIMIT'); END")
                self.connection.execute("CREATE TRIGGER no_attempt_delete BEFORE DELETE ON attempts BEGIN SELECT RAISE(ABORT,'NO_REFUND_OR_RESET'); END")
                self.connection.execute("CREATE TRIGGER immutable_attempt_identity BEFORE UPDATE OF purpose,reserved_utc,request_sha256,max_tokens ON attempts BEGIN SELECT RAISE(ABORT,'IMMUTABLE_ATTEMPT'); END")
            require({r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} ==
                    {"policy", "attempts", "sqlite_sequence"}, "Unexpected budget schema")
            require(old_transport._strict_json(self.connection.execute("SELECT value FROM policy WHERE singleton=1").fetchone()[0]) == policy,
                    "Global budget policy drift")
            require(self.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] <= TOTAL_LIMIT,
                    "Global budget invalid")
            self.connection.execute("COMMIT")
            directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            self.connection.execute("ROLLBACK")
            self.connection.close()
            raise

    def reserve_probe(self, request):
        validate_body(request)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            require(self.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] < TOTAL_LIMIT,
                    "Original global12 budget exhausted")
            require(self.connection.execute("SELECT COUNT(*) FROM attempts WHERE purpose=?", (PURPOSE,)).fetchone()[0] == 0,
                    "Probe already reserved/attempted; retry forbidden")
            cursor = self.connection.execute("INSERT INTO attempts(purpose,status,reserved_utc,request_sha256,max_tokens) VALUES (?,?,?,?,?)",
                (PURPOSE, "RESERVED", datetime.now(timezone.utc).isoformat(), digest(request), PROBE_LIMIT))
            self.connection.execute("COMMIT")
            return cursor.lastrowid
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def settle(self, attempt_id, status, receipt):
        require(status in ("SUCCEEDED", "REJECTED", "UNCERTAIN"), "Invalid settlement")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute("UPDATE attempts SET status=?,settled_utc=?,receipt=? WHERE attempt_id=? AND status='RESERVED'",
                (status, datetime.now(timezone.utc).isoformat(), canonical(receipt), attempt_id))
            require(cursor.rowcount == 1, "Attempt already settled; no replay")
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def snapshot(self):
        attempts = [dict(r) for r in self.connection.execute("SELECT * FROM attempts ORDER BY attempt_id")]
        for row in attempts:
            if row["receipt"]:
                row["receipt"] = old_transport._strict_json(row["receipt"])
        return {"original_global_maximum": TOTAL_LIMIT, "attempts_consumed": len(attempts),
                "remaining_global_attempts": TOTAL_LIMIT - len(attempts), "attempts": attempts,
                "probe_repeat_allowed": False, "qa_authorized": False}

    def close(self):
        self.connection.close()


class SingleProbeTransport:
    """Exactly one POST. No SDK, retries, preflight, redirects or fallback."""
    def __init__(self, credential_file, *, explicitly_live=False):
        self.credential_file = Path(credential_file)
        self.live = explicitly_live
        self.used = False

    def send(self, request):
        validate_body(request)
        require(self.live is True, "Probe is dry-run by default")
        require(not self.used, "Single transport cannot resend")
        self.used = True
        key = old_transport._read_credential(self.credential_file)
        connection = None
        try:
            connection = http.client.HTTPSConnection("ai.tju.edu.cn", port=443, timeout=60)
            connection.request("POST", "/api/v3/chat/completions", body=canonical(request).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            require(len(raw) <= 1024 * 1024, "Response too large; no retry")
            require(key.encode() not in raw, "Credential echo withheld; no retry")
            # Also reject escaped JSON echoes before any raw body can be saved.
            decoded = old_transport._strict_json(raw)
            require(key not in canonical(decoded), "Escaped credential echo withheld; no retry")
            headers = {}
            for name in ("x-request-id", "request-id", "x-correlation-id"):
                value = response.getheader(name)
                require(not isinstance(value, str) or key not in value,
                        "Credential header echo withheld; no retry")
                headers[name] = old_transport._safe_id(value)
            return {"http_status": response.status, "headers": headers,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw": raw}
        except ProbeHold:
            raise
        except (OSError, TimeoutError, http.client.HTTPException, UnicodeError, ValueError):
            raise ProbeHold("REQUEST_FAILED_OR_UNCERTAIN_NO_RETRY") from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass


def response_observation(envelope):
    data = old_transport._strict_json(envelope["raw"])
    require(isinstance(data, dict), "Response is not an object")
    choices = data.get("choices", [])
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
    message = choice.get("message", {})
    return {"http_status": envelope["http_status"], "response_id": old_transport._safe_id(data.get("id")),
            "returned_model": data.get("model"), "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"), "content": message.get("content") if isinstance(message, dict) else None,
            "headers": envelope["headers"], "raw_sha256": envelope["raw_sha256"], "_data": data}


def validate_observation(observation):
    require(observation["http_status"] == 200, "HTTP_NON_200_STOP")
    require(observation["returned_model"] == old_transport.EXPECTED_MODEL, "MODEL_MISMATCH_STOP")
    data = observation["_data"]
    choices = data.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict), "INVALID_CHOICES_STOP")
    choice = choices[0]
    message = choice.get("message")
    require(type(choice.get("index")) is int and choice["index"] == 0 and isinstance(message, dict) and
            message.get("role") == "assistant" and not message.get("tool_calls") and not message.get("function_call") and not message.get("refusal"),
            "INVALID_ASSISTANT_RESPONSE_STOP")
    usage = observation["usage"]
    require(isinstance(usage, dict), "USAGE_MISSING_STOP")
    require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")),
            "USAGE_INVALID_STOP")
    require(usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0 and
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "USAGE_INCONSISTENT_STOP")
    require(usage["completion_tokens"] <= PROBE_LIMIT, "REPORTED_OUTPUT_EXCEEDS_64_STOP")
    if "output_tokens" in usage:
        require(type(usage["output_tokens"]) is int and usage["output_tokens"] == usage["completion_tokens"],
                "AMBIGUOUS_OUTPUT_ACCOUNTING_STOP")
    reasoning = None
    details = usage.get("completion_tokens_details", {})
    require(isinstance(details, dict), "INVALID_REASONING_ACCOUNTING_STOP")
    for row in (details, usage, usage.get("output_tokens_details", {})):
        require(isinstance(row, dict), "INVALID_REASONING_ACCOUNTING_STOP")
        if "reasoning_tokens" in row:
            count = row["reasoning_tokens"]
            require(type(count) is int and 0 <= count <= usage["completion_tokens"] and
                    (reasoning is None or count == reasoning), "AMBIGUOUS_REASONING_ACCOUNTING_STOP")
            reasoning = count
    require(observation["finish_reason"] == "stop", "INCOMPLETE_FINISH_STOP")
    require(isinstance(observation["content"], str) and observation["content"].strip() == EXPECTED_TEXT,
            "FIXED_CONFIRMATION_MISSING_STOP")
    return {"response_valid": True, "reasoning_tokens": reasoning,
            "reasoning_breakdown_known": reasoning is not None,
            "server_64_hard_limit_proved": False, "server_4096_hard_limit_proved": False,
            "qa_live_authorized": False}


def verify_experiment(artifacts, canonical_dir):
    experiment = read(artifacts / "EXPERIMENT_PROBE.json")
    receipt = read(artifacts / "EXPERIMENT_PROBE_FREEZE.json")
    require(digest(experiment) == receipt["sha256"], "Probe experiment drift")
    require(experiment["version"] == VERSION and experiment["probe_maximum_attempts"] == 1 and
            experiment["original_global_maximum_attempts"] == 12 and experiment["output_limit"] == 64 and
            experiment["qa_authorized"] is False, "Probe scope drift")
    require(experiment["canonical_directory"] == str(Path(canonical_dir).resolve()), "Canonical budget path drift")
    for binding in experiment["bindings"]:
        require(file_binding(binding["path"]) == binding, "Frozen probe/source/config binding drift")
    require(read(artifacts / "REQUEST.json") == body(), "Probe request drift")
    return experiment


def execute_once(artifacts, canonical_dir, *, live=False, transport=None):
    artifacts, canonical_dir = Path(artifacts), Path(canonical_dir)
    experiment = verify_experiment(artifacts, canonical_dir)
    if not live:
        return {"status": "DRY_RUN_ONLY", "api_attempts_started": 0, "planned_max_tokens": 64,
                "original_global_budget": 12, "after_one_probe_remaining": 11, "qa_authorized": False}
    if transport is None:
        require(artifacts.resolve() == ARTIFACTS and canonical_dir.resolve() == CANONICAL,
                "Live probe paths are fixed")
        transport = SingleProbeTransport(experiment["credential_file"], explicitly_live=True)
    else:
        require(getattr(transport, "is_mock", False) is True, "Only a mock can replace the probe transport")
    ledger = ProbeBudget(canonical_dir, file_binding(artifacts / "EXPERIMENT_PROBE.json"))
    try:
        # This FULL-synchronous transaction commits before transport.send.
        attempt_id = ledger.reserve_probe(body())
        durable_write(artifacts / "PRE_SEND_RESERVATION.json", {"attempt_id": attempt_id,
                      "budget": ledger.snapshot(), "request_sha256": digest(body())})
        observation = None
        try:
            envelope = transport.send(copy.deepcopy(body()))
            durable_write(artifacts / "HTTP_RESPONSE_BODY.bin", envelope["raw"], binary=True)
            durable_write(artifacts / "HTTP_RECEIPT.json", {k: v for k, v in envelope.items() if k != "raw"})
            observation = response_observation(envelope)
            durable_write(artifacts / "RESPONSE_OBSERVATION.json", {k: v for k, v in observation.items() if k != "_data"})
            validated = validate_observation(observation)
            result = {"status": "PROBE_RESPONSE_VALID_QA_HOLD", **validated,
                      "api_attempts_consumed": 1, "usage": observation["usage"],
                      "returned_model": observation["returned_model"], "finish_reason": observation["finish_reason"]}
            ledger.settle(attempt_id, "SUCCEEDED", result)
        except BaseException as error:
            reason = str(error) if isinstance(error, ProbeHold) else type(error).__name__
            result = {"status": "HOLD_NO_RETRY", "reason": reason, "error_type": type(error).__name__,
                      "api_attempts_consumed": 1, "qa_authorized": False,
                      "usage": observation["usage"] if observation else None,
                      "returned_model": observation["returned_model"] if observation else None,
                      "finish_reason": observation["finish_reason"] if observation else None}
            ledger.settle(attempt_id, "REJECTED" if observation else "UNCERTAIN", result)
        result["budget"] = ledger.snapshot()
        durable_write(artifacts / "PROBE_RESULT.json", result)
        return result
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-probe-once", action="store_true")
    args = parser.parse_args()
    try:
        result = execute_once(ARTIFACTS, CANONICAL, live=args.live_probe_once)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] in ("DRY_RUN_ONLY", "PROBE_RESPONSE_VALID_QA_HOLD") else 2
    except Exception as error:
        # Do not expose networking/credential exception text in console output.
        print(json.dumps({"status": "HOLD_NO_RETRY", "error_type": type(error).__name__,
                          "message": "No additional request is allowed. Inspect the durable ledger and artifacts."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
