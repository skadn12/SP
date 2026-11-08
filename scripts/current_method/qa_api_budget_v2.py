"""Fail-closed, durable request-attempt ledger for the three-root P0 pilot.

This module never opens a network connection. A RESERVED row consumes budget
before a caller may send a request. It is never deleted, refunded, or retried.
SQLite BEGIN IMMEDIATE, FULL synchronous commits and UNIQUE(root_id, stage)
make that rule apply across processes and restarts. Response content and secrets
are not stored here; only hashes and a small audit-field allowlist are persisted.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

ROOTS = ("cc-train-episode-109164", "cc-train-episode-24719", "cc-train-episode-761")
STAGES = ("DISCOVERY", "AUTHOR", "BLIND", "COMPARE")
MAX_ATTEMPTS = 12
MAX_OUTPUT_TOKENS = 4096
SCHEMA_VERSION = "qa_api_budget_v2"


class BudgetError(Exception):
    """A ledger, policy, ordering, or budget violation: do not send a request."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BudgetError("audit binding must be finite JSON") from exc


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _reason(reason):
    if not isinstance(reason, str):
        raise BudgetError("reason must be text")
    return reason


def _audit_binding(binding):
    if not isinstance(binding, dict) or not binding:
        raise BudgetError("request binding must be a non-empty object")
    _canonical(binding)
    # Keep only explicitly non-sensitive audit fields. The hash still binds the
    # complete caller-provided JSON, without copying messages or credentials.
    keys = ("model", "stage", "root_id", "request_sha256", "packet_sha256",
            "manifest_sha256", "body_sha256", "max_tokens", "max_completion_tokens", "max_output_tokens",
            "input_tokens", "input_tokens_estimate", "input_token_estimate", "attempt_label")
    safe = {k: binding[k] for k in keys if k in binding}
    if "max_tokens" not in safe:
        raise BudgetError("explicit output limit of 4096 is required before reserving")
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if key in safe and (type(safe[key]) is not int or safe[key] != MAX_OUTPUT_TOKENS):
            raise BudgetError("explicit output limit must be exactly 4096")
    for key in ("request_sha256", "packet_sha256", "manifest_sha256", "body_sha256"):
        if key in safe and not _valid_sha(safe[key]):
            raise BudgetError("invalid request binding SHA256")
    for key in ("input_tokens", "input_tokens_estimate", "input_token_estimate"):
        if key in safe and (type(safe[key]) is not int or safe[key] < 0):
            raise BudgetError("input token estimates must be non-negative integers")
    return _sha(binding), _canonical(safe)


def _audit_receipt(receipt):
    if not isinstance(receipt, dict):
        raise BudgetError("receipt must be an object")
    _canonical(receipt)
    response_hash = receipt.get("response_sha256")
    if response_hash is not None and not _valid_sha(response_hash):
        raise BudgetError("invalid response SHA256")
    usage = receipt.get("usage", {})
    if not isinstance(usage, dict):
        raise BudgetError("usage must be an object")
    allowed_usage = ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
    for key, value in usage.items():
        if key not in allowed_usage or type(value) is not int or value < 0:
            raise BudgetError("usage contains unsupported fields or non-negative-integer violation")
    keys = ("model", "response_model", "http_status", "finish_reason", "request_id",
            "response_sha256", "error_type", "uncertain", "output_limit_verified")
    safe = {k: receipt[k] for k in keys if k in receipt}
    return response_hash, _canonical(usage), _canonical(safe)


class Ledger:
    """A single experiment's non-resettable ledger; every open revalidates it."""

    def __init__(self, path, experiment_id, manifest_sha256, mode="mock"):
        if mode not in ("mock", "live") or not isinstance(experiment_id, str) or not experiment_id.strip():
            raise BudgetError("experiment_id and mock/live mode must be explicit")
        if not _valid_sha(manifest_sha256):
            raise BudgetError("manifest_sha256 must be lowercase SHA256")
        self.path = Path(path)
        if self.path.is_symlink():
            raise BudgetError("ledger may not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.binding = {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id,
                        "manifest_sha256": manifest_sha256, "mode": mode,
                        "max_attempts": str(MAX_ATTEMPTS), "max_output_tokens": str(MAX_OUTPUT_TOKENS),
                        "roots_json": _canonical(ROOTS), "stages_json": _canonical(STAGES)}
        existed = self.path.exists()
        self._conn = None
        try:
            self._conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=FULL")
            # Rollback journal is intentional: one durable DB and no dependence
            # on a WAL sidecar when copying/exporting audit evidence.
            if self._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() != "delete":
                raise BudgetError("durable rollback journal could not be enabled")
            self._conn.execute("BEGIN IMMEDIATE")
            tables = {r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                if existed:
                    raise BudgetError("existing empty/unknown ledger cannot be initialized")
                self._initialize()
            self._validate_state()
            self._conn.execute("COMMIT")
            # Ensure a newly created ledger directory entry is durable too.
            if not existed:
                fd = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except (sqlite3.Error, OSError, BudgetError) as exc:
            if self._conn is not None:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                self._conn.close()
                self._conn = None
            if isinstance(exc, BudgetError):
                raise
            raise BudgetError("unable to open/validate durable ledger: " + type(exc).__name__) from exc

    def _initialize(self):
        self._conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.execute("CREATE TABLE roots(root_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('UNCLAIMED','ACTIVE','HOLD','NO_SAFE_QA','PASS_STATIC_REVIEW')), claimed_utc TEXT, finished_utc TEXT, reason TEXT NOT NULL DEFAULT '')")
        self._conn.execute("CREATE TABLE attempts(attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, root_id TEXT NOT NULL REFERENCES roots(root_id), stage TEXT NOT NULL CHECK(stage IN ('DISCOVERY','AUTHOR','BLIND','COMPARE')), status TEXT NOT NULL CHECK(status IN ('RESERVED','SUCCESS','FAILED','UNCERTAIN')), request_sha256 TEXT NOT NULL, request_audit_json TEXT NOT NULL, reserved_utc TEXT NOT NULL, settled_utc TEXT, response_sha256 TEXT, usage_json TEXT NOT NULL DEFAULT '{}', receipt_json TEXT NOT NULL DEFAULT '{}', UNIQUE(root_id,stage))")
        self._conn.execute("CREATE TRIGGER attempt_cap BEFORE INSERT ON attempts WHEN (SELECT COUNT(*) FROM attempts)>=12 BEGIN SELECT RAISE(ABORT,'request attempt budget exhausted'); END")
        self._conn.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", self.binding.items())
        self._conn.executemany("INSERT INTO roots(root_id,status) VALUES(?,'UNCLAIMED')", [(r,) for r in ROOTS])

    def _validate_state(self):
        if self._conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise BudgetError("ledger integrity check failed")
        tables = {r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        triggers = {r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        if tables != {"metadata", "roots", "attempts", "sqlite_sequence"} or triggers != {"attempt_cap"}:
            raise BudgetError("unknown ledger schema")
        actual = dict(self._conn.execute("SELECT key,value FROM metadata"))
        if actual != self.binding:
            raise BudgetError("ledger mode, experiment, manifest or policy binding mismatch")
        roots = list(self._conn.execute("SELECT * FROM roots"))
        if len(roots) != len(ROOTS) or {r["root_id"] for r in roots} != set(ROOTS):
            raise BudgetError("unknown or missing roots in ledger")
        attempts = list(self._conn.execute("SELECT * FROM attempts ORDER BY attempt_id"))
        if len(attempts) > MAX_ATTEMPTS:
            raise BudgetError("ledger exceeds total attempt budget")
        if [a["attempt_id"] for a in attempts] != list(range(1, len(attempts) + 1)):
            raise BudgetError("attempt history is missing or corrupted")
        sequence = self._conn.execute("SELECT seq FROM sqlite_sequence WHERE name='attempts'").fetchone()
        if (sequence[0] if sequence else 0) != len(attempts):
            raise BudgetError("attempt sequence shows deleted or altered request history")
        for root in roots:
            status = root["status"]
            chain = [a for a in attempts if a["root_id"] == root["root_id"]]
            if status not in ("UNCLAIMED", "ACTIVE", "HOLD", "NO_SAFE_QA", "PASS_STATIC_REVIEW"):
                raise BudgetError("unknown root state")
            if status == "UNCLAIMED" and (chain or root["claimed_utc"] or root["finished_utc"]):
                raise BudgetError("unclaimed root has request history")
            if status != "UNCLAIMED" and not root["claimed_utc"]:
                raise BudgetError("claimed root is missing claim receipt")
            if status in ("UNCLAIMED", "ACTIVE") and root["finished_utc"]:
                raise BudgetError("nonterminal root has terminal receipt")
            if status not in ("UNCLAIMED", "ACTIVE") and not root["finished_utc"]:
                raise BudgetError("terminal root is missing receipt")
            if [a["stage"] for a in chain] != list(STAGES[:len(chain)]):
                raise BudgetError("root request stages are missing, duplicated or out of order")
            if any(a["status"] != "SUCCESS" for a in chain[:-1]):
                raise BudgetError("a request followed a failed or unsettled request")
            if status == "PASS_STATIC_REVIEW" and (len(chain) != len(STAGES) or any(a["status"] != "SUCCESS" for a in chain)):
                raise BudgetError("PASS root lacks complete successful request chain")
            if status == "NO_SAFE_QA" and (not chain or len(chain) > 2 or any(a["status"] != "SUCCESS" for a in chain)):
                raise BudgetError("NO_SAFE_QA lacks a successful discovery/author stop")
        if any(a["root_id"] not in ROOTS for a in attempts):
            raise BudgetError("attempt has unknown root")
        for a in attempts:
            if not _valid_sha(a["request_sha256"]) or not a["reserved_utc"] or a["status"] not in ("RESERVED", "SUCCESS", "FAILED", "UNCERTAIN"):
                raise BudgetError("invalid request audit receipt")
            if (a["status"] == "RESERVED") != (a["settled_utc"] is None):
                raise BudgetError("request settlement state is corrupt")
            if a["response_sha256"] is not None and not _valid_sha(a["response_sha256"]):
                raise BudgetError("invalid stored response hash")
            try:
                for k in ("request_audit_json", "usage_json", "receipt_json"):
                    if not isinstance(json.loads(a[k]), dict):
                        raise ValueError("not an object")
                request_audit = json.loads(a["request_audit_json"])
                _, request_clean = _audit_binding(request_audit)
                if request_clean != _canonical(request_audit):
                    raise ValueError("unknown request audit fields")
                if request_audit.get("root_id", a["root_id"]) != a["root_id"] or request_audit.get("stage", a["stage"]) != a["stage"]:
                    raise ValueError("request audit root/stage drift")
                receipt_audit = json.loads(a["receipt_json"])
                usage = json.loads(a["usage_json"])
                response_hash, _, receipt_clean = _audit_receipt({**receipt_audit, "usage": usage})
                if receipt_clean != _canonical(receipt_audit):
                    raise ValueError("unknown receipt audit fields")
                if response_hash != a["response_sha256"]:
                    raise ValueError("response hash columns disagree")
                if a["status"] == "RESERVED" and (receipt_audit or usage or a["response_sha256"]):
                    raise ValueError("unsettled request has a response receipt")
            except (TypeError, ValueError) as exc:
                raise BudgetError("invalid stored audit JSON") from exc
        if self._conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise BudgetError("ledger foreign key check failed")

    @contextmanager
    def _transaction(self):
        if self._conn is None:
            raise BudgetError("ledger is closed")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._validate_state()
            yield
            self._validate_state()
            self._conn.execute("COMMIT")
        except (sqlite3.Error, BudgetError) as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(exc, BudgetError):
                raise
            raise BudgetError("ledger transaction failed: " + type(exc).__name__) from exc

    def _root(self, root_id):
        if root_id not in ROOTS:
            raise BudgetError("unknown root")
        return self._conn.execute("SELECT * FROM roots WHERE root_id=?", (root_id,)).fetchone()

    def claim_root(self, root_id):
        with self._transaction():
            if self._root(root_id)["status"] != "UNCLAIMED":
                raise BudgetError("root generation opportunity was already claimed; no restart/retry")
            self._conn.execute("UPDATE roots SET status='ACTIVE',claimed_utc=? WHERE root_id=?", (_now(), root_id))

    def reserve(self, root_id, stage, request_binding):
        digest, audit = _audit_binding(request_binding)
        if stage not in STAGES:
            raise BudgetError("unknown request stage")
        if "root_id" in request_binding and request_binding["root_id"] != root_id:
            raise BudgetError("request binding root mismatch")
        if "stage" in request_binding and request_binding["stage"] != stage:
            raise BudgetError("request binding stage mismatch")
        with self._transaction():
            if self._root(root_id)["status"] != "ACTIVE":
                raise BudgetError("request requires an active claimed root")
            if self._conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] >= MAX_ATTEMPTS:
                raise BudgetError("global 12-attempt budget exhausted")
            chain = list(self._conn.execute("SELECT stage,status FROM attempts WHERE root_id=? ORDER BY attempt_id", (root_id,)))
            if len(chain) >= len(STAGES) or stage != STAGES[len(chain)]:
                raise BudgetError("stage duplicate or out of order")
            if chain and chain[-1]["status"] != "SUCCESS":
                raise BudgetError("previous request failed or is unsettled; no retry")
            row = self._conn.execute("INSERT INTO attempts(root_id,stage,status,request_sha256,request_audit_json,reserved_utc) VALUES(?,?,'RESERVED',?,?,?)", (root_id, stage, digest, audit, _now()))
            attempt_id = row.lastrowid
        return attempt_id

    def settle(self, attempt_id, status, receipt):
        if type(attempt_id) is not int or status not in ("SUCCESS", "FAILED", "UNCERTAIN"):
            raise BudgetError("invalid attempt settlement")
        response_hash, usage, audit = _audit_receipt(receipt)
        with self._transaction():
            row = self._conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None or row["status"] != "RESERVED":
                raise BudgetError("unknown or already settled attempt")
            self._conn.execute("UPDATE attempts SET status=?,settled_utc=?,response_sha256=?,usage_json=?,receipt_json=? WHERE attempt_id=?", (status, _now(), response_hash, usage, audit, attempt_id))

    def stop_root(self, root_id, reason, status="HOLD"):
        if status != "HOLD":
            raise BudgetError("stop_root only permits HOLD")
        return self.finish_root(root_id, status=status, reason=reason)

    def finish_root(self, root_id, status, reason=""):
        if status not in ("PASS_STATIC_REVIEW", "NO_SAFE_QA", "HOLD"):
            raise BudgetError("invalid terminal root status")
        reason = _reason(reason)
        if status != "PASS_STATIC_REVIEW" and not reason.strip():
            raise BudgetError("HOLD/NO_SAFE_QA requires an audit reason")
        with self._transaction():
            if self._root(root_id)["status"] != "ACTIVE":
                raise BudgetError("only an active root can be finalized once")
            self._conn.execute("UPDATE roots SET status=?,finished_utc=?,reason=? WHERE root_id=?", (status, _now(), reason, root_id))

    def snapshot(self):
        with self._transaction():
            roots = [dict(r) for r in self._conn.execute("SELECT * FROM roots ORDER BY root_id")]
            attempts = [dict(a) for a in self._conn.execute("SELECT * FROM attempts ORDER BY attempt_id")]
            for a in attempts:
                for key in ("request_audit_json", "usage_json", "receipt_json"):
                    a[key[:-5]] = json.loads(a.pop(key))
            result = {"metadata": dict(self.binding), "attempt_count": len(attempts), "roots": roots, "attempts": attempts,
                      "remaining_attempts": MAX_ATTEMPTS-len(attempts)}
        return result

    def export(self, path):
        # Exclusive creation: exports never replace frozen audit evidence.
        value = self.snapshot()
        try:
            with Path(path).open("x", encoding="utf-8") as fh:
                fh.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise BudgetError("unable to create exclusive ledger export") from exc
        return value

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
