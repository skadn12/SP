"""Versioned adapter for the existing global12 probe ledger.

This module does not migrate or initialize the authoritative database. Live
reservations can only target the existing canonical SQLite file; tests use an
explicit non-authoritative fixture mode that cannot produce a live send token.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from . import base_qa_capability_probe_v1 as probe
from .qa_api_transport_v2 import validate_body

VERSION = "base_qa_executor_v3_integration_20261031_v1"
ROOTS = ("cc-train-episode-109164", "cc-train-episode-24719", "cc-train-episode-761")
STAGES = ("DISCOVERY", "AUTHOR", "BLIND", "COMPARE")
GLOBAL_LIMIT = 12
MAX_OUTPUT = 4096
LIVE_GLOBAL_LEDGER_WRITES_ENABLED = False
PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "research-protocols/selfplay_week_20261019" / VERSION
V2_ARTIFACTS = ARTIFACTS.parent / "base_qa_executor_v2_20261029_v1"
CANONICAL_DIR = V2_ARTIFACTS / "live_state"
CANONICAL_LEDGER = CANONICAL_DIR / "GLOBAL_12_ATTEMPTS.sqlite3"
V3_STATE = ARTIFACTS / "live_state"
_NAMESPACE = VERSION + ":"


class BudgetHold(RuntimeError):
    pass


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise BudgetHold("audit data is not strict JSON") from None


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _request_sha(body: dict) -> str:
    validate_body(body)
    return _sha(_canonical(body).encode("utf-8"))


def _durable_exclusive_json(path: Path, value: dict) -> None:
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@dataclass(frozen=True)
class Reservation:
    attempt_id: int
    purpose: str
    request_sha256: str
    root_id: str
    stage: str
    ledger_path: Path
    authoritative: bool


class GlobalBudgetAdapter:
    """Append-only view of the canonical budget, with fail-closed QA stages."""

    def __init__(self, *, mode="inspect", ledger_path=None, state_dir=None,
                 test_policy=None, test_guard=None):
        if mode not in ("inspect", "mock", "live"):
            raise BudgetHold("unknown budget adapter mode")
        self.mode = mode
        self.ledger_path = Path(ledger_path or CANONICAL_LEDGER).resolve()
        self.state_dir = Path(state_dir or V3_STATE).resolve()
        self._lock_handle = None
        self._connection = None
        self._test_policy = test_policy
        self._test_guard = test_guard
        self._authoritative = mode == "live"

        if mode == "live":
            if not LIVE_GLOBAL_LEDGER_WRITES_ENABLED:
                raise BudgetHold("v3 live reservations are disabled pending separate capability review")
            if self.ledger_path != CANONICAL_LEDGER.resolve():
                raise BudgetHold("live execution must use the original global12 ledger")
            if self.state_dir != V3_STATE.resolve():
                raise BudgetHold("live execution state directory drift")
            self._expected_policy = self._read_authoritative_policy()
            self._acquire_experiment_lock()
            try:
                self._connection = self._connect(readonly=False)
                self._validate_connection(self._connection)
            except BaseException:
                self.close()
                raise
        elif mode == "mock":
            if self.ledger_path == CANONICAL_LEDGER.resolve():
                raise BudgetHold("mock mode may not write the authoritative ledger")
            if not isinstance(test_policy, dict) or not isinstance(test_guard, dict):
                raise BudgetHold("mock mode requires an isolated fixture policy and guard")
            self._expected_policy = dict(test_policy)
            self._validate_guard_value(test_guard, self.ledger_path.parent, test_policy)
            self._acquire_experiment_lock()
            try:
                self._connection = self._connect(readonly=False)
                self._validate_connection(self._connection)
            except BaseException:
                self.close()
                raise
        else:
            if self.ledger_path != CANONICAL_LEDGER.resolve():
                raise BudgetHold("inspect mode must use the original global12 ledger")
            self._expected_policy = self._read_authoritative_policy()
            self._connection = self._connect(readonly=True)
            self._validate_connection(self._connection)

    def _read_authoritative_policy(self):
        if (self.ledger_path.is_symlink() or CANONICAL_DIR.is_symlink() or
                not self.ledger_path.is_file() or not (CANONICAL_DIR / "BUDGET_TAKEOVER.json").is_file()):
            raise BudgetHold("canonical global budget files are missing or redirected")
        experiment = probe.ARTIFACTS / "EXPERIMENT_PROBE.json"
        binding = probe.file_binding(experiment)
        expected_guard = probe.takeover_value(binding, CANONICAL_DIR)
        try:
            observed_guard = probe.read(CANONICAL_DIR / "BUDGET_TAKEOVER.json")
        except Exception:
            raise BudgetHold("canonical budget guard is unreadable") from None
        if observed_guard != expected_guard:
            raise BudgetHold("canonical global budget takeover guard drift")
        return {"schema": probe.VERSION, "original_global_limit": GLOBAL_LIMIT,
                "experiment_binding": binding, "canonical_directory": str(CANONICAL_DIR.resolve())}

    @staticmethod
    def _validate_guard_value(guard, ledger_parent, policy):
        expected = {"schema": probe.VERSION, "original_global_limit": GLOBAL_LIMIT,
                    "experiment_binding": guard.get("experiment_binding"),
                    "canonical_directory": str(Path(ledger_parent).resolve())}
        if policy != expected:
            raise BudgetHold("mock fixture policy does not match its isolated guard")
        if (guard.get("purpose") != "GLOBAL_BUDGET_TAKEOVER_PROBE_ONLY_QA_HOLD" or
                guard.get("global_maximum_attempts") != GLOBAL_LIMIT or
                guard.get("authoritative_ledger") != str(Path(ledger_parent).resolve() / "GLOBAL_12_ATTEMPTS.sqlite3") or
                guard.get("qa_authorized") is not False):
            raise BudgetHold("invalid mock budget guard")

    def _connect(self, *, readonly):
        if self.ledger_path.is_symlink():
            raise BudgetHold("ledger may not be a symlink")
        try:
            if readonly:
                connection = sqlite3.connect(self.ledger_path.resolve().as_uri() + "?mode=ro", uri=True,
                                             timeout=15, isolation_level=None)
            else:
                connection = sqlite3.connect(str(self.ledger_path), timeout=15, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=15000")
            connection.execute("PRAGMA synchronous=FULL")
            if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
                connection.close()
                raise BudgetHold("global ledger must retain rollback-journal mode")
            return connection
        except (sqlite3.Error, OSError) as exc:
            raise BudgetHold("global ledger could not be opened: " + type(exc).__name__) from None

    def _acquire_experiment_lock(self):
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.state_dir.is_symlink():
            raise BudgetHold("live state directory may not be a symlink")
        lock_path = self.state_dir / "EXECUTION.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self._lock_handle = os.fdopen(descriptor, "r+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_handle.close()
            self._lock_handle = None
            raise BudgetHold("another v3 executor owns the global budget") from None

    def _validate_connection(self, connection):
        try:
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise BudgetHold("global ledger integrity check failed")
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            triggers = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
            if tables != {"policy", "attempts", "sqlite_sequence"} or triggers != {
                    "global12", "no_attempt_delete", "immutable_attempt_identity"}:
                raise BudgetHold("global ledger schema or triggers drifted")
            trigger_sql = {r[0]: (r[1] or "").lower() for r in connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger'")}
            if ("count(*) from attempts" not in trigger_sql["global12"] or ">=12" not in trigger_sql["global12"] or
                    "no_refund_or_reset" not in trigger_sql["no_attempt_delete"] or
                    "immutable_attempt" not in trigger_sql["immutable_attempt_identity"]):
                raise BudgetHold("global budget triggers do not enforce the expected policy")
            policy_row = connection.execute("SELECT value FROM policy WHERE singleton=1").fetchone()
            if policy_row is None or probe.old_transport._strict_json(policy_row[0]) != self._expected_policy:
                raise BudgetHold("global budget policy binding drift")
            rows = list(connection.execute("SELECT * FROM attempts ORDER BY attempt_id"))
            if len(rows) > GLOBAL_LIMIT or [r["attempt_id"] for r in rows] != list(range(1, len(rows) + 1)):
                raise BudgetHold("global attempt count or identity sequence is invalid")
            sequence = connection.execute("SELECT seq FROM sqlite_sequence WHERE name='attempts'").fetchone()
            if (sequence[0] if sequence else 0) != len(rows):
                raise BudgetHold("global ledger sequence indicates altered history")
            allowed_status = {"RESERVED", "SUCCEEDED", "REJECTED", "UNCERTAIN"}
            for row in rows:
                if row["status"] not in allowed_status or type(row["max_tokens"]) is not int or row["max_tokens"] <= 0:
                    raise BudgetHold("global ledger contains an invalid attempt record")
                if (not isinstance(row["purpose"], str) or not row["purpose"] or
                        not isinstance(row["request_sha256"], str) or
                        len(row["request_sha256"]) != 64 or
                        any(char not in "0123456789abcdef" for char in row["request_sha256"])):
                    raise BudgetHold("global ledger attempt identity is invalid")
                if row["status"] == "RESERVED":
                    raise BudgetHold("global ledger contains an unsettled prior request")
                if not row["reserved_utc"] or not row["settled_utc"]:
                    raise BudgetHold("global ledger settlement timestamps are incomplete")
                if row["purpose"].startswith(_NAMESPACE) and row["max_tokens"] != MAX_OUTPUT:
                    raise BudgetHold("v3 QA ledger row does not bind max_tokens=4096")
                if row["receipt"]:
                    receipt = probe.old_transport._strict_json(row["receipt"])
                    if not isinstance(receipt, dict):
                        raise BudgetHold("global ledger receipt is invalid")
            self._validate_v3_history(rows)
            return rows
        except sqlite3.Error as exc:
            raise BudgetHold("global ledger validation failed: " + type(exc).__name__) from None

    @staticmethod
    def _validate_v3_history(rows):
        by_root = {root: [] for root in ROOTS}
        for row in rows:
            purpose = row["purpose"]
            if not purpose.startswith(_NAMESPACE):
                continue
            suffix = purpose[len(_NAMESPACE):]
            pieces = suffix.rsplit(":", 1)
            if len(pieces) != 2 or pieces[0] not in by_root or pieces[1] not in STAGES:
                raise BudgetHold("malformed v3 QA attempt purpose")
            by_root[pieces[0]].append((pieces[1], row["status"]))
        for history in by_root.values():
            stages = [row[0] for row in history]
            if stages != list(STAGES[:len(stages)]):
                raise BudgetHold("v3 root stages are duplicated or out of order")
            if any(status != "SUCCEEDED" for _, status in history[:-1]):
                raise BudgetHold("v3 stage followed a failed request")

    def snapshot(self):
        connection = self._connection
        if connection is None:
            raise BudgetHold("global ledger is closed")
        rows = self._validate_connection(connection)
        return {"limit": GLOBAL_LIMIT, "used": len(rows), "remaining": GLOBAL_LIMIT - len(rows),
                "attempts": [dict(row) for row in rows]}

    def claim_root(self, root_id):
        if self.mode not in ("mock", "live") or self._connection is None:
            raise BudgetHold("root claiming requires a writable adapter")
        if root_id not in ROOTS:
            raise BudgetHold("root is outside the frozen experiment")
        snapshot = self.snapshot()
        if snapshot["remaining"] < len(STAGES):
            raise BudgetHold("insufficient global budget reserved for a complete root")
        claims = self.state_dir / "ROOT_CLAIMS"
        terminals = self.state_dir / "ROOT_TERMINALS"
        claims.mkdir(parents=True, exist_ok=True, mode=0o700)
        terminals.mkdir(parents=True, exist_ok=True, mode=0o700)
        if claims.is_symlink() or terminals.is_symlink():
            raise BudgetHold("root claim or terminal directory may not be a symlink")
        for claim in claims.glob("*.json"):
            if not (terminals / claim.name).exists():
                raise BudgetHold("an earlier root is active or interrupted; do not resume/retry")
        _durable_exclusive_json(claims / (root_id + ".json"), {
            "version": VERSION, "root_id": root_id, "claimed_utc": datetime.now(timezone.utc).isoformat(),
            "budget_used_before_root": snapshot["used"], "candidate_opportunity": 1})

    def stop_root(self, root_id, reason):
        return self.finish_root(root_id, "HOLD", reason)

    def reserve(self, root_id, stage, body):
        if self.mode not in ("mock", "live") or self._connection is None:
            raise BudgetHold("request reservation requires a writable adapter")
        if root_id not in ROOTS or stage not in STAGES:
            raise BudgetHold("unknown root or stage")
        validate_body(body)
        if body.get("max_tokens") != MAX_OUTPUT:
            raise BudgetHold("each QA request must explicitly cap output at 4096")
        digest = _request_sha(body)
        purpose = _NAMESPACE + root_id + ":" + stage
        claim = self.state_dir / "ROOT_CLAIMS" / (root_id + ".json")
        terminal = self.state_dir / "ROOT_TERMINALS" / (root_id + ".json")
        if not claim.is_file() or terminal.exists():
            raise BudgetHold("root opportunity is not active")
        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = self._validate_connection(connection)
            if len(rows) >= GLOBAL_LIMIT:
                raise BudgetHold("global 12-attempt budget exhausted")
            history = [r for r in rows if r["purpose"].startswith(_NAMESPACE + root_id + ":")]
            expected_stage = STAGES[len(history)] if len(history) < len(STAGES) else None
            if stage != expected_stage:
                raise BudgetHold("duplicate or out-of-order root stage")
            if history and history[-1]["status"] != "SUCCEEDED":
                raise BudgetHold("previous request did not succeed; root remains stopped")
            cursor = connection.execute(
                "INSERT INTO attempts(purpose,status,reserved_utc,request_sha256,max_tokens) VALUES(?,?,?,?,?)",
                (purpose, "RESERVED", datetime.now(timezone.utc).isoformat(), digest, MAX_OUTPUT))
            connection.execute("COMMIT")
            return Reservation(cursor.lastrowid, purpose, digest, root_id, stage,
                               self.ledger_path, self._authoritative)
        except (sqlite3.Error, BudgetHold) as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(exc, BudgetHold):
                raise
            raise BudgetHold("global reservation failed: " + type(exc).__name__) from None

    def verify_reservation(self, reservation, body):
        if not isinstance(reservation, Reservation) or reservation.ledger_path != self.ledger_path:
            raise BudgetHold("reservation does not belong to this ledger")
        if reservation.request_sha256 != _request_sha(body):
            raise BudgetHold("request body differs from its durable reservation")
        row = self._connection.execute(
            "SELECT purpose,status,request_sha256,max_tokens FROM attempts WHERE attempt_id=?",
            (reservation.attempt_id,)).fetchone()
        expected = (reservation.purpose, "RESERVED", reservation.request_sha256, MAX_OUTPUT)
        if row is None or tuple(row) != expected:
            raise BudgetHold("a committed 4096-token reservation is required before send")

    def settle(self, reservation, status, receipt):
        if self.mode not in ("mock", "live") or status not in ("SUCCEEDED", "REJECTED", "UNCERTAIN"):
            raise BudgetHold("invalid attempt settlement")
        if not isinstance(reservation, Reservation) or reservation.ledger_path != self.ledger_path:
            raise BudgetHold("attempt reservation identity mismatch")
        safe = self._safe_receipt(receipt)
        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_connection_without_reserved(connection, reservation.attempt_id)
            cursor = connection.execute(
                "UPDATE attempts SET status=?,settled_utc=?,receipt=? WHERE attempt_id=? AND status='RESERVED'",
                (status, datetime.now(timezone.utc).isoformat(), _canonical(safe), reservation.attempt_id))
            if cursor.rowcount != 1:
                raise BudgetHold("attempt was already settled; replay forbidden")
            self._validate_connection(connection)
            connection.execute("COMMIT")
        except (sqlite3.Error, BudgetHold) as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(exc, BudgetHold):
                raise
            raise BudgetHold("attempt settlement failed: " + type(exc).__name__) from None

    def execute_stage(self, root_id, stage, body, transport):
        """Reserve durably, send exactly once, then settle; failures stay spent."""
        reservation = self.reserve(root_id, stage, body)
        try:
            value, receipt = transport.send(body, reservation)
        except BaseException as error:
            uncertain = bool(getattr(error, "uncertain", True))
            status = "UNCERTAIN" if uncertain else "REJECTED"
            safe_receipt = getattr(error, "receipt", None)
            if not isinstance(safe_receipt, dict):
                safe_receipt = {"error_type": type(error).__name__,
                                "reason_code": getattr(error, "reason", type(error).__name__),
                                "uncertain": uncertain, "retry_count": 0,
                                "transport_attempt_count": 1}
            self.settle(reservation, status, safe_receipt)
            raise
        self.settle(reservation, "SUCCEEDED", {
            "returned_model": receipt["returned_model"],
            "response_sha256": receipt["raw_response_sha256"],
            "request_id": receipt.get("request_id"), "http_status": receipt["http_status"],
            "finish_reason": receipt["finish_reason"],
            "usage": {key: receipt["usage"][key]
                      for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "reasoning_tokens": receipt.get("reasoning_tokens"), "retry_count": 0,
            "transport_attempt_count": 1})
        return value, receipt, reservation

    @staticmethod
    def _validate_connection_without_reserved(connection, current_attempt_id):
        rows = list(connection.execute("SELECT * FROM attempts ORDER BY attempt_id"))
        for row in rows:
            if row["status"] == "RESERVED" and row["attempt_id"] != current_attempt_id:
                raise BudgetHold("another unresolved attempt blocks execution")
        current = next((row for row in rows if row["attempt_id"] == current_attempt_id), None)
        if current is None or current["status"] != "RESERVED":
            raise BudgetHold("current attempt is missing or already settled")

    @staticmethod
    def _safe_receipt(receipt):
        if not isinstance(receipt, dict):
            raise BudgetHold("attempt receipt must be an object")
        _canonical(receipt)
        allowed = {"returned_model", "request_id", "http_status", "finish_reason", "usage",
                   "reasoning_tokens", "response_sha256", "error_type", "reason_code",
                   "uncertain", "retry_count", "transport_attempt_count"}
        if set(receipt) - allowed:
            raise BudgetHold("attempt receipt contains non-audit fields")
        usage = receipt.get("usage", {})
        if not isinstance(usage, dict) or set(usage) - {"prompt_tokens", "completion_tokens", "total_tokens"}:
            raise BudgetHold("unsupported usage fields")
        if any(type(value) is not int or value < 0 for value in usage.values()):
            raise BudgetHold("invalid token usage values")
        return dict(receipt)

    def finish_root(self, root_id, status, reason=""):
        if status not in ("PASS_STATIC_REVIEW", "NO_SAFE_QA", "HOLD"):
            raise BudgetHold("invalid root terminal status")
        if status != "PASS_STATIC_REVIEW" and (not isinstance(reason, str) or not reason.strip()):
            raise BudgetHold("HOLD and NO_SAFE_QA require an explicit reason")
        if root_id not in ROOTS:
            raise BudgetHold("unknown root")
        history = [row for row in self.snapshot()["attempts"]
                   if row["purpose"].startswith(_NAMESPACE + root_id + ":")]
        if status == "PASS_STATIC_REVIEW" and (
                [row["purpose"].rsplit(":", 1)[-1] for row in history] != list(STAGES) or
                any(row["status"] != "SUCCEEDED" for row in history)):
            raise BudgetHold("a passing root requires four successful ordered stages")
        if status == "NO_SAFE_QA" and (len(history) != 1 or
                history[0]["purpose"].rsplit(":", 1)[-1] != "DISCOVERY" or
                history[0]["status"] != "SUCCEEDED"):
            raise BudgetHold("NO_SAFE_QA requires one successful discovery stage")
        if not (self.state_dir / "ROOT_CLAIMS" / (root_id + ".json")).is_file():
            raise BudgetHold("root was never durably claimed")
        _durable_exclusive_json(self.state_dir / "ROOT_TERMINALS" / (root_id + ".json"), {
            "version": VERSION, "root_id": root_id, "status": status,
            "reason": reason, "finished_utc": datetime.now(timezone.utc).isoformat()})

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
                self._lock_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
