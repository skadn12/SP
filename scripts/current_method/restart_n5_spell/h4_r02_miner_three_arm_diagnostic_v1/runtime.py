"""Arm-isolated, ordered six-call scope around the original ledger/client.

Logical requests are identified globally by arm + root + original request ID.
Original transport retries remain inside the unchanged Challenger client.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import (
    GuardrailStop, ObservedClient, append, now, exclusive_run,
)
from .constants import ROOTS, ARMS, SCHEDULE, VERSION, root_path, global_request_id

PLAN_TOTAL = LIMIT_TOTAL = 6
LIMIT_PER_PAIR = 1
LIMIT_PER_ROOT = 3
LIMIT_PER_ARM = 2
MODE = "MINER_THREE_ARM_DIAGNOSTIC"
MINER_INTENT = "h4_k3_intent_miner"
BUDGET_CLASS = "fresh_qa_generation"
MAX_OUTPUT_TOKENS = 4096
IDENTITY_KEYS = ("order", "root_id", "arm_id", "request_id", "global_request_id")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _identity(row):
    return {key: row.get(key) for key in IDENTITY_KEYS}


def _frozen_pairing(run):
    """Require the exact six frozen pair identities and fixed serial order."""
    run = Path(run)
    pairing_path, schedule_path = run / "pairing_manifest.jsonl", run / "schedule.json"
    if not pairing_path.is_file() or not schedule_path.is_file():
        raise GuardrailStop("frozen_pairing_and_schedule_required")
    try:
        pairing = _read_rows(pairing_path)
        schedule_document = read(schedule_path)
        schedule = (schedule_document if isinstance(schedule_document, list) else
                    schedule_document.get("schedule", schedule_document.get("rows")))
        if ([ _identity(row) for row in pairing ] != list(SCHEDULE) or
                not isinstance(schedule, list) or
                [_identity(row) for row in schedule] != list(SCHEDULE)):
            raise GuardrailStop("fixed_pairing_or_schedule_drift")
        for row in pairing:
            if not isinstance(row.get("prompt_sha256"), str) or not row["prompt_sha256"]:
                raise GuardrailStop("pairing_prompt_identity_required")
            for key in ("max_output_tokens", "max_tokens"):
                if key in row and (type(row[key]) is not int or row[key] != MAX_OUTPUT_TOKENS):
                    raise GuardrailStop("pairing_output_budget_drift")
            for key in ("logical_call_limit", "maximum_logical_calls"):
                if key in row and row[key] != LIMIT_PER_PAIR:
                    raise GuardrailStop("pairing_logical_budget_drift")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise GuardrailStop(f"frozen_pairing_or_schedule_invalid:{error}") from error
    return pairing, sha256_file(pairing_path), sha256_file(schedule_path)


def state_paths(run):
    # Scan the whole new run: moving a ledger to another arm/shard/backup cannot
    # make a reservation disappear from the budget. Global call events below
    # also stop recovery if a request file is deleted or moved elsewhere.
    return sorted(Path(run).glob("**/ledger/requests/*.json"))


def _scope(arm_id, root_id, request_id, *, intent, attempt, budget_class, metadata):
    if (arm_id not in ARMS or root_id not in ROOTS or
            request_id != f"{root_id}-intent-miner"):
        raise GuardrailStop(f"request_outside_six_miner_scope:{arm_id}:{request_id}")
    if intent != MINER_INTENT or type(attempt) is not int or attempt != 1:
        raise GuardrailStop(f"miner_role_or_attempt_drift:{request_id}")
    if budget_class != BUDGET_CLASS:
        raise GuardrailStop(f"miner_budget_class_drift:{request_id}")
    if (not isinstance(metadata, dict) or
            type(metadata.get("max_tokens")) is not int or
            metadata["max_tokens"] != MAX_OUTPUT_TOKENS or
            metadata.get("temperature") != 0.1 or metadata.get("request_id") != request_id):
        raise GuardrailStop(f"miner_request_config_drift:{request_id}")
    if "seed" in metadata:
        raise GuardrailStop(f"miner_seed_forbidden:{request_id}")
    return {"intent_id": None, "mode": "intent_miner", "attempt": "primary"}


def _audit_events(run, states_by_global):
    event_path = Path(run) / "audit/call_events.jsonl"
    events = _read_rows(event_path) if event_path.exists() else []
    reservations = [row for row in events if row.get("event") == "new_logical_call"]
    expected = list(SCHEDULE[:len(reservations)])
    if len(reservations) > LIMIT_TOTAL or [_identity(row) for row in reservations] != expected:
        raise GuardrailStop("logical_call_event_schedule_drift")
    event_ids = {row["global_request_id"] for row in reservations}
    if event_ids != set(states_by_global):
        raise GuardrailStop("logical_call_event_reservation_inventory_mismatch")
    return reservations


def audit_reservations(run):
    """Validate all arms' budgets and observed outcomes before any new send."""
    pairing, pairing_sha, schedule_sha = _frozen_pairing(run)
    by_global = {row["global_request_id"]: row for row in pairing}
    paths = state_paths(run)
    if len(paths) > LIMIT_TOTAL:
        raise GuardrailStop("experiment_logical_budget_exceeded")
    pair_counts, root_counts, arm_counts = Counter(), Counter(), Counter()
    states_by_global = {}
    for path in paths:
        state = read(path)
        metadata = state.get("request_metadata")
        if not isinstance(metadata, dict):
            raise GuardrailStop("reservation_metadata_missing")
        arm_id, root_id = metadata.get("arm_id"), metadata.get("root_id")
        request_id = state.get("request_id")
        _scope(arm_id, root_id, request_id, intent=state.get("intent"),
               attempt=state.get("attempt"), budget_class=state.get("budget_class"),
               metadata=metadata)
        identity = global_request_id(arm_id, root_id)
        row = by_global[identity]
        root = root_path(run, arm_id, root_id)
        expected_path = root / "ledger/requests" / f"{request_id}.json"
        if path.resolve() != expected_path.resolve():
            raise GuardrailStop("reservation_directory_scope_drift")
        pair_counts[(arm_id, root_id)] += 1
        root_counts[root_id] += 1
        arm_counts[arm_id] += 1
        if (pair_counts[(arm_id, root_id)] > LIMIT_PER_PAIR or
                root_counts[root_id] > LIMIT_PER_ROOT or arm_counts[arm_id] > LIMIT_PER_ARM):
            raise GuardrailStop("root_arm_logical_budget_exceeded")
        if (metadata.get("global_request_id") != identity or
                metadata.get("assigned_mode") != MODE or
                metadata.get("order") != row["order"] or
                metadata.get("mode") != "intent_miner" or metadata.get("attempt") != "primary" or
                metadata.get("pairing_manifest_sha256") != pairing_sha or
                metadata.get("schedule_sha256") != schedule_sha or
                metadata.get("pairing_row_sha256") != sha256_bytes(canonical_json(row).encode()) or
                state.get("prompt_sha256") != row["prompt_sha256"] or
                state.get("budget_limit") != LIMIT_PER_PAIR or
                state.get("total_budget_limit") != LIMIT_PER_PAIR):
            raise GuardrailStop("reservation_frozen_contract_drift")
        if state.get("request_metadata_sha256") != sha256_bytes(canonical_json(metadata).encode()):
            raise GuardrailStop("reservation_metadata_hash_drift")
        states_by_global[identity] = state
        if state.get("status") != "completed":
            failure_path = root / "ledger/observed_failures" / path.name
            if not failure_path.is_file():
                raise GuardrailStop(f"unknown_paid_status_no_automatic_resend:{identity}")
            failure = read(failure_path)
            if (failure.get("global_request_id") != identity or
                    failure.get("reservation_sha256") != sha256_bytes(canonical_json(state).encode()) or
                    failure.get("automatic_resend") is not False):
                raise GuardrailStop("observed_failure_reservation_drift")
    _audit_events(run, states_by_global)
    return paths


class BoundedLedger:
    """One request per frozen pair, exact cache only within that pair's ledger."""
    def __init__(self, root_dir, manifest_sha256, run, arm_id, root_id, *,
                 before_request=None, after_response=None, backend_mode="live"):
        self.root_dir, self.run = Path(root_dir), Path(run)
        self.arm_id, self.root_id = arm_id, root_id
        if (arm_id not in ARMS or root_id not in ROOTS or
                self.root_dir.resolve() != root_path(self.run, arm_id, root_id).resolve()):
            raise GuardrailStop("root_arm_outside_frozen_acceptance")
        if backend_mode not in ("live", "mock"):
            raise GuardrailStop("miner_backend_mode_invalid")
        if backend_mode == "live" and (before_request is None or after_response is None):
            raise GuardrailStop("live_miner_requires_freeze_token_and_usage_guards")
        pairing, self.pairing_sha, self.schedule_sha = _frozen_pairing(self.run)
        self.global_id = global_request_id(arm_id, root_id)
        self.pairing_row = next(row for row in pairing if row["global_request_id"] == self.global_id)
        self.delegate = RequestLedger(self.root_dir, manifest_sha256, backend_mode)
        self.before_request, self.after_response = before_request, after_response

    def _event(self, event, **values):
        append(self.run / "audit/call_events.jsonl", {
            "event": event, "at_utc": now(), **_identity(self.pairing_row), **values,
        })

    def invoke(self, **kwargs):
        request_id = kwargs["request_id"]
        scope = _scope(self.arm_id, self.root_id, request_id, intent=kwargs.get("intent"),
                       attempt=kwargs.get("attempt"), budget_class=kwargs.get("budget_class"),
                       metadata=kwargs.get("request_metadata"))
        pairing, pairing_sha, schedule_sha = _frozen_pairing(self.run)
        if pairing_sha != self.pairing_sha or schedule_sha != self.schedule_sha:
            raise GuardrailStop("pairing_or_schedule_changed_after_ledger_creation")
        if kwargs["prompt_sha256"] != self.pairing_row["prompt_sha256"]:
            raise GuardrailStop("request_prompt_outside_frozen_pairing")
        states = audit_reservations(self.run)
        existing = self.delegate.read(request_id)
        metadata = {**kwargs["request_metadata"], "assigned_mode": MODE,
                    **_identity(self.pairing_row), **scope,
                    "pairing_manifest_sha256": pairing_sha, "schedule_sha256": schedule_sha,
                    "pairing_row_sha256": sha256_bytes(canonical_json(self.pairing_row).encode()),
                    "original_budget_limit": kwargs.get("budget_limit"),
                    "original_total_budget_limit": kwargs.get("total_budget_limit")}
        kwargs.update(request_metadata=metadata, budget_limit=LIMIT_PER_PAIR,
                      total_budget_limit=LIMIT_PER_PAIR)
        expected = {
            "intent": kwargs["intent"], "attempt": kwargs["attempt"],
            "prompt_sha256": kwargs["prompt_sha256"],
            "manifest_sha256": self.delegate.manifest_sha256,
            "backend_mode": self.delegate.backend_mode, "budget_class": kwargs["budget_class"],
            "request_metadata_sha256": sha256_bytes(canonical_json(metadata).encode()),
            "backend_identity_sha256": sha256_bytes(canonical_json(kwargs.get("backend_identity", {})).encode()),
        }
        if existing:
            if any(existing.get(key) != value for key, value in expected.items()):
                raise GuardrailStop(f"cached_request_contract_drift:{self.global_id}")
            if existing["status"] != "completed":
                failure = read(self.root_dir / "ledger/observed_failures" / f"{request_id}.json")
                self._event("known_failure_replayed_no_send")
                error_type = type(failure["error_type"], (Exception,), {})
                raise error_type(failure["error_message"])
            try:
                result = self.delegate.invoke(**kwargs)
            except Exception as error:
                raise GuardrailStop(f"completed_cache_verification_failed:{self.global_id}:{error}") from error
            self._event("completed_cache_reused_no_send")
            if self.after_response is not None:
                self.after_response(request_id, result)
            return result
        if len(states) >= LIMIT_TOTAL:
            raise GuardrailStop("experiment_logical_budget_exhausted")
        if _identity(self.pairing_row) != SCHEDULE[len(states)]:
            raise GuardrailStop(f"request_out_of_frozen_order:{self.global_id}")
        if len(list(self.delegate.state_dir.glob("*.json"))) >= LIMIT_PER_PAIR:
            raise GuardrailStop("root_arm_logical_budget_exhausted")
        if self.before_request is not None:
            self.before_request(request_id, metadata)
        # Durable global identity prevents recovering budget by moving/removing
        # an arm directory. A crash here is a guarded unknown outcome on resume.
        self._event("new_logical_call", **scope, experiment_call_index=len(states) + 1)
        try:
            result = self.delegate.invoke(**kwargs)
        except Exception as error:
            reservation = self.delegate.read(request_id)
            if reservation is None:
                raise GuardrailStop(f"ledger_failure_before_reservation:{self.global_id}:{error}") from error
            if reservation.get("status") == "completed":
                raise GuardrailStop(f"ledger_failure_after_completion:{self.global_id}:{error}") from error
            failure = {**_identity(self.pairing_row), "error_type": type(error).__name__,
                       "error_message": str(error), "at_utc": now(),
                       "reservation_sha256": sha256_bytes(canonical_json(reservation).encode()),
                       "automatic_resend": False, "billing_status": "unknown_no_completion_recorded"}
            atomic_json(self.root_dir / "ledger/observed_failures" / f"{request_id}.json", failure)
            self._event("logical_call_failed", error_type=type(error).__name__,
                        billing_status="unknown_no_completion_recorded")
            raise
        self._event("logical_call_completed", usage=result.metadata.get("usage"),
                    finish_reason=result.metadata.get("finish_reason"))
        if self.after_response is not None:
            self.after_response(request_id, result)
        return result


def checkpoint_root(root):
    root = Path(root)
    path = root / "miner_completion_checkpoint.json"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p != path)
    value = {"version": VERSION,
             "files": {str(p.relative_to(root)): sha256_file(p) for p in files}}
    if path.exists():
        if read(path) != value:
            raise GuardrailStop(f"completed_artifact_drift:{root}")
    else:
        atomic_json(path, value)


def verify_checkpoint(root):
    root = Path(root)
    path = root / "miner_completion_checkpoint.json"
    if not path.is_file():
        return False
    checkpoint = read(path)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != path}
    if checkpoint.get("version") != VERSION or actual != set(checkpoint["files"]):
        raise GuardrailStop(f"completed_artifact_inventory_drift:{root}")
    for name, expected in checkpoint["files"].items():
        target = root / name
        if sha256_file(target) != expected:
            raise GuardrailStop(f"completed_artifact_drift:{target}")
    return True
