"""Arm-isolated, ordered four-call scope around the original ledger/client.

Logical requests are identified globally by arm + root + original request ID.
Original transport retries remain inside the unchanged Challenger client.
"""
from __future__ import annotations

from collections import Counter
import json
import urllib.error
from pathlib import Path

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import (
    GuardrailStop, ObservedClient as _OriginalObservedClient, append, now, exclusive_run,
)
from .constants import ROOTS, ARMS, SCHEDULE, VERSION, root_path, global_request_id

PLAN_TOTAL = LIMIT_TOTAL = 4
LIMIT_PER_PAIR = 1
LIMIT_PER_ROOT = 2
LIMIT_PER_ARM = 2
MODE = "BASIC_MINER_V3_OFF_PAIRED"
MINER_INTENT = "h4_k3_intent_miner"
BUDGET_CLASS = "fresh_qa_generation"
MAX_OUTPUT_TOKENS = 4096
IDENTITY_KEYS = ("order", "root_id", "arm_id", "request_id", "global_request_id")
# A received HTTP rejection establishes that this call did not produce a model
# completion. Timeouts, HTTP 408/5xx, malformed service responses and other
# exceptions remain unknown. This only classifies outcomes; it adds no retries.
KNOWN_REJECTION_CODES = frozenset((400, 401, 402, 403, 404, 405, 406, 413, 415, 422, 429))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _identity(row):
    return {key: row.get(key) for key in IDENTITY_KEYS}


def record_guardrail(run, stage, error, *, arm_id=None, root_id=None, request_id=None):
    """Durable stop condition even when _call_json swallows an Exception."""
    value = {"version": VERSION, "at_utc": now(), "stage": stage,
             "error_type": type(error).__name__, "reason": str(error),
             "arm_id": arm_id, "root_id": root_id, "request_id": request_id,
             "global_request_id": (global_request_id(arm_id, root_id)
                                   if arm_id in ARMS and root_id in ROOTS else None),
             "status": "BLOCKED", "sticky": True, "automatic_resend": False}
    append(Path(run) / "audit/guardrail_events.jsonl", value)
    if arm_id in ARMS and root_id in ROOTS:
        append(root_path(run, arm_id, root_id) / "ledger/guardrails.jsonl", value)
    return value


def assert_no_guardrails(run):
    path = Path(run) / "audit/guardrail_events.jsonl"
    rows = _read_rows(path) if path.exists() else []
    if rows:
        raise GuardrailStop("sticky_guardrail_blocks_paid_calls:" + rows[0]["reason"])


def _failure_status(error):
    return ("KNOWN_HTTP_REJECTION" if isinstance(error, urllib.error.HTTPError)
            and error.code in KNOWN_REJECTION_CODES else "UNKNOWN_COMPLETION")


def _transport_path(run, arm_id, root_id):
    return root_path(run, arm_id, root_id) / "ledger/transport_outcomes" / f"{root_id}-intent-miner.json"


class ObservedClient(_OriginalObservedClient):
    """Reuse original HTTP observer; add global identity and outcome certainty.

    The original client and observer still own every request and retry. This
    wrapper records their final observed status and never issues another call.
    """
    def __init__(self, delegate, event_path, arm_id, root_id):
        super().__init__(delegate, Path(event_path), arm_id, root_id)
        self.run = Path(event_path).parent.parent
        self.arm_id = arm_id

    def complete(self, *, prompt, max_tokens, json_object):
        assert_no_guardrails(self.run)
        target = _transport_path(self.run, self.arm_id, self.root_id)
        if target.exists():
            record_guardrail(self.run, "duplicate_transport_outcome", "transport called twice for one cell",
                             arm_id=self.arm_id, root_id=self.root_id, request_id=self.root_id + "-intent-miner")
            raise GuardrailStop("duplicate_transport_outcome")
        initial_count = len(_read_rows(self.event_path)) if self.event_path.exists() else 0
        response, error = None, None
        try:
            response = super().complete(prompt=prompt, max_tokens=max_tokens, json_object=json_object)
            return response
        except BaseException as caught:
            error = caught
            raise
        finally:
            events = _read_rows(self.event_path)[initial_count:] if self.event_path.exists() else []
            finishes = [row for row in events if row.get("event") == "http_attempt_finished"]
            originals = [row for row in events if row.get("event") == "logical_call_transport_outcome"]
            status = "COMPLETED" if response is not None else _failure_status(error)
            value = {"version": VERSION, "arm_id": self.arm_id, "root_id": self.root_id,
                     "request_id": self.root_id + "-intent-miner",
                     "global_request_id": global_request_id(self.arm_id, self.root_id),
                     "prompt_sha256": sha256_bytes(prompt.encode()), "max_tokens": max_tokens,
                     "service_completion_status": status, "completion_received": response is not None,
                     "http_status": finishes[-1].get("http_status") if finishes else None,
                     "transport_attempt_count": sum(row.get("event") == "http_attempt_started" for row in events),
                     "original_transport_outcomes": originals,
                     "error_type": type(error).__name__ if error else None,
                     "usage": response.metadata.get("usage") if response is not None else None,
                     "finish_reason": response.metadata.get("finish_reason") if response is not None else None,
                     "at_utc": now()}
            atomic_json(target, value)
            append(self.event_path, {**value, "event": "paired_transport_outcome"})
            if status == "UNKNOWN_COMPLETION":
                record_guardrail(self.run, "unknown_service_completion", error,
                                 arm_id=self.arm_id, root_id=self.root_id, request_id=value["request_id"])


def _check_transport(run, arm_id, root_id, state):
    """Live requests must have matching, conclusive actual transport evidence."""
    path = _transport_path(run, arm_id, root_id)
    if state.get("backend_mode") == "mock" and not path.exists():
        return None
    if not path.is_file():
        raise GuardrailStop("missing_actual_transport_outcome")
    outcome = read(path)
    if (outcome.get("global_request_id") != global_request_id(arm_id, root_id) or
            outcome.get("request_id") != state["request_id"] or
            outcome.get("prompt_sha256") != state["prompt_sha256"] or
            outcome.get("max_tokens") != MAX_OUTPUT_TOKENS or
            not outcome.get("transport_attempt_count")):
        raise GuardrailStop("actual_transport_identity_drift")
    status = outcome.get("service_completion_status")
    if state["status"] == "completed":
        if status != "COMPLETED" or not outcome.get("completion_received"):
            raise GuardrailStop("ledger_completion_disagrees_with_transport")
        if (state.get("response_metadata") or {}).get("transport_attempt_count") != outcome["transport_attempt_count"]:
            raise GuardrailStop("ledger_transport_attempt_count_drift")
    elif status != "KNOWN_HTTP_REJECTION" or outcome.get("http_status") not in KNOWN_REJECTION_CODES:
        raise GuardrailStop("unknown_service_completion_blocks_paid_calls")
    return outcome


def _frozen_pairing(run):
    """Require the exact four frozen pair identities and fixed serial order."""
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
        raise GuardrailStop(f"request_outside_four_miner_scope:{arm_id}:{request_id}")
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
    assert_no_guardrails(run)
    try:
        return _audit_reservations(run)
    except BaseException as error:
        record_guardrail(run, "reservation_audit", error)
        raise


def _audit_reservations(run):
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
            if failure.get("service_completion_status") != "KNOWN_HTTP_REJECTION":
                raise GuardrailStop("unknown_service_completion_blocks_paid_calls")
        _check_transport(run, arm_id, root_id, state)
    _audit_events(run, states_by_global)
    return paths


class BoundedLedger:
    """One request per frozen pair, exact cache only within that pair's ledger."""
    def __init__(self, root_dir, manifest_sha256, run, arm_id, root_id, *,
                 before_request=None, after_response=None, backend_mode="live"):
        self.root_dir, self.run = Path(root_dir), Path(run)
        self.arm_id, self.root_id = arm_id, root_id
        self._ordinary_failure = None
        try:
            self._initialize(manifest_sha256, before_request, after_response, backend_mode)
        except BaseException as error:
            record_guardrail(self.run, "ledger_initialization", error, arm_id=arm_id, root_id=root_id)
            raise

    def _initialize(self, manifest_sha256, before_request, after_response, backend_mode):
        arm_id, root_id = self.arm_id, self.root_id
        assert_no_guardrails(self.run)
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
        self._ordinary_failure = None
        try:
            return self._invoke(**kwargs)
        except BaseException as error:
            if error is not self._ordinary_failure:
                record_guardrail(self.run, "ledger_or_callback_guard", error, arm_id=self.arm_id,
                                 root_id=self.root_id, request_id=kwargs.get("request_id"))
            raise

    def _invoke(self, **kwargs):
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
                self._ordinary_failure = error_type(failure["error_message"])
                raise self._ordinary_failure
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
                       "automatic_resend": False, "service_completion_status": _failure_status(error),
                       "http_status": error.code if isinstance(error, urllib.error.HTTPError) else None,
                       "billing_status": ("known_http_rejection_no_completion" if _failure_status(error) == "KNOWN_HTTP_REJECTION"
                                          else "unknown_no_completion_recorded")}
            atomic_json(self.root_dir / "ledger/observed_failures" / f"{request_id}.json", failure)
            self._event("logical_call_failed", error_type=type(error).__name__,
                        billing_status=failure["billing_status"], service_completion_status=failure["service_completion_status"])
            if failure["service_completion_status"] == "UNKNOWN_COMPLETION":
                record_guardrail(self.run, "unknown_service_completion", error, arm_id=self.arm_id,
                                 root_id=self.root_id, request_id=request_id)
            self._ordinary_failure = error
            raise
        self._event("logical_call_completed", usage=result.metadata.get("usage"),
                    finish_reason=result.metadata.get("finish_reason"))
        if self.after_response is not None:
            self.after_response(request_id, result)
        return result


def post_miner_check(run, root, arm_id, root_id, ledger):
    """Mandatory after shared Miner returns, before checkpoint or next cell.

    Normal function return may contain a swallowed transport/preflight error.
    Only a completed response or observed conclusive HTTP rejection can finish
    a cell and allow the remaining fixed schedule. This function never sends.
    """
    run, root = Path(run), Path(root)
    request_id = root_id + "-intent-miner"
    result = {"version": VERSION, "arm_id": arm_id, "root_id": root_id,
              "request_id": request_id, "global_request_id": global_request_id(arm_id, root_id),
              "at_utc": now(), "calls_added": 0}
    try:
        assert_no_guardrails(run)
        if (root.resolve() != root_path(run, arm_id, root_id).resolve() or
                ledger.arm_id != arm_id or ledger.root_id != root_id or
                ledger.root_dir.resolve() != root.resolve()):
            raise GuardrailStop("post_miner_identity_mismatch")
        audit_reservations(run)
        state = ledger.delegate.read(request_id)
        if state is None:
            raise GuardrailStop("shared_miner_returned_without_request_reservation")
        if state.get("manifest_sha256") != ledger.delegate.manifest_sha256:
            raise GuardrailStop("post_miner_frozen_manifest_drift")
        transport = _check_transport(run, arm_id, root_id, state)
        if state["status"] == "completed":
            raw_path = root / "raw/backend_responses" / (request_id + ".txt")
            if (Path(state["raw_response_path"]).resolve() != raw_path.resolve() or
                    not raw_path.is_file() or sha256_file(raw_path) != state["raw_response_sha256"]):
                raise GuardrailStop("post_miner_completed_raw_drift")
            result.update(status="COMPLETED", ledger_status="completed", may_continue=True,
                          service_completion_status="COMPLETED",
                          finish_reason=(state.get("response_metadata") or {}).get("finish_reason"))
        else:
            failure = read(root / "ledger/observed_failures" / (request_id + ".json"))
            if failure.get("service_completion_status") != "KNOWN_HTTP_REJECTION":
                raise GuardrailStop("unknown_service_completion_blocks_paid_calls")
            result.update(status="KNOWN_HTTP_FAILURE", ledger_status=state["status"], may_continue=True,
                          service_completion_status="KNOWN_HTTP_REJECTION", http_status=failure["http_status"])
        result["transport"] = transport
    except BaseException as error:
        result.update(status="BLOCKED", may_continue=False, reason=str(error))
        atomic_json(root / "post_miner_check.json", result)
        record_guardrail(run, "post_miner_check", error, arm_id=arm_id, root_id=root_id, request_id=request_id)
        raise
    atomic_json(root / "post_miner_check.json", result)
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
