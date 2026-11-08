"""Single-root, four-attempt adapter over the existing append-only global12 ledger."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from . import base_qa_capability_probe_v1 as probe
from . import base_qa_global_budget_adapter_v3 as v3budget
from .qa_api_transport_v2 import validate_body

VERSION = "base_qa_executor_v4_single_root_20261031_v1"
ROOT_ID = "cc-train-episode-109164"
ROOTS = (ROOT_ID,)
STAGES = ("DISCOVERY", "AUTHOR", "BLIND", "COMPARE")
GLOBAL_LIMIT = 12
ROOT_ATTEMPT_LIMIT = 4
MAX_OUTPUT = 4096
LIVE_GLOBAL_LEDGER_WRITES_ENABLED = False
PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "research-protocols/selfplay_week_20261019" / VERSION
CANONICAL_DIR = v3budget.CANONICAL_DIR
CANONICAL_LEDGER = v3budget.CANONICAL_LEDGER
V4_STATE = ARTIFACTS / "live_state"
NAMESPACE = VERSION + ":"
Reservation = v3budget.Reservation


class BudgetHold(RuntimeError):
    pass


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise BudgetHold("audit data is not strict JSON") from None


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _durable_json(path, value):
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


def baseline_row(row):
    return {key: row[key] for key in (
        "attempt_id", "purpose", "status", "reserved_utc", "settled_utc", "request_sha256", "max_tokens")}


def _canonical_policy():
    if CANONICAL_DIR.is_symlink() or CANONICAL_LEDGER.is_symlink() or not CANONICAL_LEDGER.is_file():
        raise BudgetHold("canonical global ledger missing or redirected")
    experiment = probe.ARTIFACTS / "EXPERIMENT_PROBE.json"
    binding = probe.file_binding(experiment)
    guard_path = CANONICAL_DIR / "BUDGET_TAKEOVER.json"
    if probe.read(guard_path) != probe.takeover_value(binding, CANONICAL_DIR):
        raise BudgetHold("canonical global budget guard drift")
    return {"schema": probe.VERSION, "original_global_limit": GLOBAL_LIMIT,
            "experiment_binding": binding, "canonical_directory": str(CANONICAL_DIR.resolve())}


class GlobalBudgetAdapterV4:
    """Fail-closed V4 namespace with one root and at most four durable attempts."""

    def __init__(self, *, mode="inspect", ledger_path=None, state_dir=None,
                 expected_policy=None, baseline_attempts=None):
        if mode not in ("inspect", "mock", "live"):
            raise BudgetHold("unknown budget adapter mode")
        self.mode = mode
        self.ledger_path = Path(ledger_path or CANONICAL_LEDGER).resolve()
        self.state_dir = Path(state_dir or V4_STATE).resolve()
        self._connection = None
        self._lock_handle = None
        self._expected_policy = expected_policy
        self._baseline = list(baseline_attempts or [])
        self._authoritative = mode == "live"

        if mode == "live":
            if not LIVE_GLOBAL_LEDGER_WRITES_ENABLED:
                raise BudgetHold("V4 live ledger writes are disabled pending service-capacity evidence")
            if self.ledger_path != CANONICAL_LEDGER.resolve() or self.state_dir != V4_STATE.resolve():
                raise BudgetHold("V4 live paths must be canonical")
            self._expected_policy = _canonical_policy()
            if len(self._baseline) != 4:
                raise BudgetHold("V4 requires the frozen four-attempt starting ledger")
            self._acquire_lock()
            self._connection = self._connect(readonly=False)
            self._validate_connection(self._connection)
        elif mode == "inspect":
            if self.ledger_path != CANONICAL_LEDGER.resolve():
                raise BudgetHold("inspect mode only accepts the canonical ledger")
            self._expected_policy = _canonical_policy()
            self._connection = self._connect(readonly=True)
            self._validate_connection(self._connection)
        else:
            if self.ledger_path == CANONICAL_LEDGER.resolve():
                raise BudgetHold("mock mode may not open the authoritative ledger")
            if not isinstance(self._expected_policy, dict) or len(self._baseline) != 4:
                raise BudgetHold("mock mode requires an isolated fixture policy and four-row baseline")
            self._acquire_lock()
            self._connection = self._connect(readonly=False)
            self._validate_connection(self._connection)

    def _acquire_lock(self):
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.state_dir.is_symlink():
            raise BudgetHold("V4 state directory may not be a symlink")
        path = self.state_dir / "EXECUTION.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self._lock_handle = os.fdopen(descriptor, "r+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_handle.close()
            self._lock_handle = None
            raise BudgetHold("another V4 executor owns the run") from None

    def _connect(self, *, readonly):
        if self.ledger_path.is_symlink():
            raise BudgetHold("ledger may not be a symlink")
        try:
            if readonly:
                db = sqlite3.connect(self.ledger_path.as_uri() + "?mode=ro", uri=True,
                                     timeout=15, isolation_level=None)
            else:
                db = sqlite3.connect(str(self.ledger_path), timeout=15, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=15000")
            db.execute("PRAGMA synchronous=FULL")
            if db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
                db.close()
                raise BudgetHold("global ledger journal mode drift")
            return db
        except (sqlite3.Error, OSError) as error:
            raise BudgetHold("global ledger open failed: " + type(error).__name__) from None

    def _validate_connection(self, db, *, allow_reserved_id=None):
        try:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise BudgetHold("global ledger integrity check failed")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            triggers = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
            if tables != {"policy", "attempts", "sqlite_sequence"} or triggers != {
                    "global12", "no_attempt_delete", "immutable_attempt_identity"}:
                raise BudgetHold("global ledger schema or triggers drifted")
            sql = {row[0]: (row[1] or "").lower() for row in db.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger'")}
            if ("count(*) from attempts" not in sql["global12"] or ">=12" not in sql["global12"] or
                    "no_refund_or_reset" not in sql["no_attempt_delete"] or
                    "immutable_attempt" not in sql["immutable_attempt_identity"]):
                raise BudgetHold("global budget triggers do not enforce the expected limits")
            policy_row = db.execute("SELECT value FROM policy WHERE singleton=1").fetchone()
            if policy_row is None or json.loads(policy_row[0]) != self._expected_policy:
                raise BudgetHold("global budget policy binding drift")
            rows = list(db.execute("SELECT * FROM attempts ORDER BY attempt_id"))
            if len(rows) > GLOBAL_LIMIT or [row["attempt_id"] for row in rows] != list(range(1, len(rows) + 1)):
                raise BudgetHold("global attempt count or sequence invalid")
            sequence = db.execute("SELECT seq FROM sqlite_sequence WHERE name='attempts'").fetchone()
            if (sequence[0] if sequence else 0) != len(rows):
                raise BudgetHold("global attempt sequence altered")
            if self._baseline and len(rows) < len(self._baseline):
                raise BudgetHold("frozen global baseline has been removed")
            for index, expected in enumerate(self._baseline):
                if baseline_row(rows[index]) != expected:
                    raise BudgetHold("frozen global baseline changed")
            for row in rows:
                if row["status"] == "RESERVED" and row["attempt_id"] != allow_reserved_id:
                    raise BudgetHold("unsettled request blocks execution")
                if row["status"] not in {"RESERVED", "SUCCEEDED", "REJECTED", "UNCERTAIN"}:
                    raise BudgetHold("invalid global attempt status")
                if type(row["max_tokens"]) is not int or row["max_tokens"] <= 0:
                    raise BudgetHold("invalid attempt token cap")
            v4_rows = [row for row in rows if row["purpose"].startswith(NAMESPACE)]
            if len(v4_rows) > ROOT_ATTEMPT_LIMIT:
                raise BudgetHold("V4 single-root attempt limit exceeded")
            if self._baseline:
                for row in rows[len(self._baseline):]:
                    if not row["purpose"].startswith(NAMESPACE):
                        raise BudgetHold("unexpected attempt appended during V4 run")
            stages = []
            for row in v4_rows:
                suffix = row["purpose"][len(NAMESPACE):]
                pieces = suffix.rsplit(":", 1)
                if len(pieces) != 2 or pieces[0] != ROOT_ID or pieces[1] not in STAGES:
                    raise BudgetHold("malformed V4 attempt purpose")
                if row["max_tokens"] != MAX_OUTPUT:
                    raise BudgetHold("V4 ledger request must bind max_tokens=4096")
                stages.append((pieces[1], row["status"]))
            if [stage for stage, _ in stages] != list(STAGES[:len(stages)]):
                raise BudgetHold("V4 stages duplicated or out of order")
            if any(status != "SUCCEEDED" for _, status in stages[:-1]):
                raise BudgetHold("a failed V4 request was followed by another request")
            return rows
        except sqlite3.Error as error:
            raise BudgetHold("global ledger validation failed: " + type(error).__name__) from None

    def snapshot(self):
        if self._connection is None:
            raise BudgetHold("ledger is closed")
        rows = self._validate_connection(self._connection)
        return {"limit": GLOBAL_LIMIT, "used": len(rows), "remaining": GLOBAL_LIMIT - len(rows),
                "v4_used": sum(row["purpose"].startswith(NAMESPACE) for row in rows),
                "v4_remaining": ROOT_ATTEMPT_LIMIT - sum(row["purpose"].startswith(NAMESPACE) for row in rows),
                "attempts": [dict(row) for row in rows]}

    def claim_root(self, root_id):
        if root_id != ROOT_ID:
            raise BudgetHold("V4 permits only cc-train-episode-109164")
        if self.mode not in ("mock", "live") or self._connection is None:
            raise BudgetHold("root claim requires a writable adapter")
        snapshot = self.snapshot()
        if snapshot["v4_used"] != 0 or snapshot["remaining"] < ROOT_ATTEMPT_LIMIT:
            raise BudgetHold("root already started or four-call budget is unavailable")
        claims = self.state_dir / "ROOT_CLAIMS"
        terminals = self.state_dir / "ROOT_TERMINALS"
        claims.mkdir(parents=True, exist_ok=True, mode=0o700)
        terminals.mkdir(parents=True, exist_ok=True, mode=0o700)
        if claims.is_symlink() or terminals.is_symlink():
            raise BudgetHold("V4 claim directories may not be symlinks")
        if any(not (terminals / path.name).exists() for path in claims.glob("*.json")):
            raise BudgetHold("an earlier V4 run is active or interrupted")
        _durable_json(claims / (root_id + ".json"), {
            "version": VERSION, "root_id": root_id, "candidate_opportunities": 1,
            "attempt_limit": ROOT_ATTEMPT_LIMIT, "claimed_utc": datetime.now(timezone.utc).isoformat()})

    def reserve(self, root_id, stage, body):
        if self.mode not in ("mock", "live") or self._connection is None:
            raise BudgetHold("reservation requires a writable adapter")
        if root_id != ROOT_ID or stage not in STAGES:
            raise BudgetHold("root or stage outside the V4 experiment")
        validate_body(body)
        if body.get("max_tokens") != MAX_OUTPUT:
            raise BudgetHold("V4 requests require explicit max_tokens=4096")
        if not (self.state_dir / "ROOT_CLAIMS" / (ROOT_ID + ".json")).is_file():
            raise BudgetHold("V4 root was not durably claimed")
        if (self.state_dir / "ROOT_TERMINALS" / (ROOT_ID + ".json")).exists():
            raise BudgetHold("V4 root is terminal")
        digest = _digest(body)
        purpose = NAMESPACE + ROOT_ID + ":" + stage
        db = self._connection
        try:
            db.execute("BEGIN IMMEDIATE")
            rows = self._validate_connection(db)
            if len(rows) >= GLOBAL_LIMIT:
                raise BudgetHold("global 12-attempt budget exhausted")
            history = [row for row in rows if row["purpose"].startswith(NAMESPACE)]
            expected = STAGES[len(history)] if len(history) < ROOT_ATTEMPT_LIMIT else None
            if stage != expected:
                raise BudgetHold("duplicate stage or V4 four-attempt limit reached")
            if history and history[-1]["status"] != "SUCCEEDED":
                raise BudgetHold("previous stage failed; V4 root stopped")
            cursor = db.execute(
                "INSERT INTO attempts(purpose,status,reserved_utc,request_sha256,max_tokens) VALUES(?,?,?,?,?)",
                (purpose, "RESERVED", datetime.now(timezone.utc).isoformat(), digest, MAX_OUTPUT))
            db.execute("COMMIT")
            return Reservation(cursor.lastrowid, purpose, digest, root_id, stage, self.ledger_path,
                               self.mode == "live")
        except (sqlite3.Error, BudgetHold) as error:
            try:
                db.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(error, BudgetHold):
                raise
            raise BudgetHold("durable V4 reservation failed: " + type(error).__name__) from None

    def verify_reservation(self, reservation, body):
        if not isinstance(reservation, Reservation) or reservation.ledger_path != self.ledger_path:
            raise BudgetHold("reservation identity mismatch")
        if reservation.request_sha256 != _digest(body):
            raise BudgetHold("request body differs from durable reservation")
        row = self._connection.execute(
            "SELECT purpose,status,request_sha256,max_tokens FROM attempts WHERE attempt_id=?",
            (reservation.attempt_id,)).fetchone()
        if row is None or tuple(row) != (reservation.purpose, "RESERVED", reservation.request_sha256, MAX_OUTPUT):
            raise BudgetHold("committed V4 reservation is required before send")

    def settle(self, reservation, status, receipt):
        if status not in ("SUCCEEDED", "REJECTED", "UNCERTAIN"):
            raise BudgetHold("invalid V4 attempt status")
        db = self._connection
        try:
            db.execute("BEGIN IMMEDIATE")
            self._validate_connection(db, allow_reserved_id=reservation.attempt_id)
            cursor = db.execute(
                "UPDATE attempts SET status=?,settled_utc=?,receipt=? WHERE attempt_id=? AND status='RESERVED'",
                (status, datetime.now(timezone.utc).isoformat(), _canonical(receipt), reservation.attempt_id))
            if cursor.rowcount != 1:
                raise BudgetHold("V4 request already settled; replay forbidden")
            self._validate_connection(db)
            db.execute("COMMIT")
        except (sqlite3.Error, BudgetHold) as error:
            try:
                db.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(error, BudgetHold):
                raise
            raise BudgetHold("V4 settlement failed: " + type(error).__name__) from None

    def execute_stage(self, root_id, stage, body, transport):
        reservation = self.reserve(root_id, stage, body)
        try:
            value, receipt = transport.send(body, reservation)
        except BaseException as error:
            uncertain = bool(getattr(error, "uncertain", True))
            self.settle(reservation, "UNCERTAIN" if uncertain else "REJECTED", {
                "error_type": type(error).__name__, "reason_code": getattr(error, "reason", type(error).__name__),
                "uncertain": uncertain, "retry_count": 0, "transport_attempt_count": 1})
            raise
        self.settle(reservation, "SUCCEEDED", {
            "returned_model": receipt.get("returned_model"), "request_id": receipt.get("request_id"),
            "http_status": receipt.get("http_status"), "finish_reason": receipt.get("finish_reason"),
            "usage": receipt.get("usage", {}), "retry_count": 0, "transport_attempt_count": 1})
        return value, receipt, reservation

    def finish_root(self, root_id, status, reason=""):
        if root_id != ROOT_ID or status not in ("PASS_STATIC_REVIEW", "NO_SAFE_QA", "HOLD"):
            raise BudgetHold("invalid V4 root terminal status")
        if status != "PASS_STATIC_REVIEW" and not (isinstance(reason, str) and reason.strip()):
            raise BudgetHold("V4 HOLD requires a reason")
        history = [row for row in self.snapshot()["attempts"] if row["purpose"].startswith(NAMESPACE)]
        stages = [row["purpose"].rsplit(":", 1)[-1] for row in history]
        if status == "PASS_STATIC_REVIEW" and (stages != list(STAGES) or
                any(row["status"] != "SUCCEEDED" for row in history)):
            raise BudgetHold("PASS requires four successful ordered stages")
        if status == "NO_SAFE_QA" and (stages != ["DISCOVERY"] or history[0]["status"] != "SUCCEEDED"):
            raise BudgetHold("NO_SAFE_QA requires one successful discovery")
        claim = self.state_dir / "ROOT_CLAIMS" / (ROOT_ID + ".json")
        if not claim.is_file():
            raise BudgetHold("V4 root was not claimed")
        _durable_json(self.state_dir / "ROOT_TERMINALS" / (ROOT_ID + ".json"), {
            "version": VERSION, "root_id": ROOT_ID, "status": status,
            "reason": reason, "finished_utc": datetime.now(timezone.utc).isoformat()})

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._lock_handle is not None:
            self._lock_handle.close()
            self._lock_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
