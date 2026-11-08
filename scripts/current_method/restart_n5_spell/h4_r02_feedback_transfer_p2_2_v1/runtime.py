"""Fixed three-root request scope around the original ledger and HTTP client.

Only logical scheduling is new. Original transport retries, completed cache,
normalization and failure fallback remain owned by the existing implementation.
"""
from __future__ import annotations

from pathlib import Path

from ..core import atomic_json, canonical_json, sha256_bytes
from ..ledger import RequestLedger
from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import (
    GuardrailStop, ObservedClient, append, now, exclusive_run, request_scope,
    checkpoint_root, verify_checkpoint,
)

FIRST_ROOTS = tuple(f"cc-train-episode-{n}" for n in (198480, 122711, 197635))
PLAN_TOTAL, LIMIT_TOTAL, LIMIT_PER_ROOT = 21, 39, 13
MODE = "P2_2_CONCRETE_TRANSFER"


def root_path(run, root_id):
    return Path(run) / "qa/shards/shard-00/roots" / root_id


def state_paths(run):
    return sorted((Path(run) / "qa/shards/shard-00/roots").glob("*/ledger/requests/*.json"))


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_reservations(run):
    """Do not send another paid request while any orphan's outcome is unknown."""
    paths = state_paths(run)
    if len(paths) > LIMIT_TOTAL:
        raise GuardrailStop("experiment_logical_budget_exceeded")
    counts = {}
    for path in paths:
        state = read(path)
        root = path.parents[2]
        root_id = root.name
        counts[root_id] = counts.get(root_id, 0) + 1
        if root_id not in FIRST_ROOTS or counts[root_id] > LIMIT_PER_ROOT:
            raise GuardrailStop("root_scope_or_budget_violation")
        if state.get("status") != "completed":
            failure = root / "ledger/observed_failures" / path.name
            if not failure.is_file():
                raise GuardrailStop(f"unknown_paid_status_no_automatic_resend:{state.get('request_id')}")
            if read(failure).get("reservation_sha256") != sha256_bytes(canonical_json(state).encode()):
                raise GuardrailStop("observed_failure_reservation_drift")
    return paths


class BoundedLedger:
    """Budget failed reservations; replay known failures; never replay an orphan.

    The before_request callback verifies frozen sources and token capacity before
    RequestLedger reserves a logical call, so a preflight block is not a paid call.
    """
    def __init__(self, root_dir, manifest_sha256, run, root_id, *,
                 before_request=None, after_response=None, backend_mode="live"):
        self.root_dir, self.run, self.root_id = Path(root_dir), Path(run), root_id
        if root_id not in FIRST_ROOTS:
            raise GuardrailStop("root_outside_first_acceptance")
        self.delegate = RequestLedger(self.root_dir, manifest_sha256, backend_mode)
        self.before_request = before_request
        self.after_response = after_response

    def invoke(self, **kwargs):
        request_id = kwargs["request_id"]
        scope = request_scope(self.root_dir, self.root_id, request_id)
        states = audit_reservations(self.run)
        existing = self.delegate.read(request_id)
        failure_path = self.root_dir / "ledger/observed_failures" / f"{request_id}.json"
        metadata = {**kwargs.get("request_metadata", {}), "assigned_mode": MODE,
                    "root_id": self.root_id, **scope,
                    "original_budget_limit": kwargs["budget_limit"],
                    "original_total_budget_limit": kwargs["total_budget_limit"]}
        kwargs.update(request_metadata=metadata, budget_limit=LIMIT_PER_ROOT,
                      total_budget_limit=LIMIT_PER_ROOT)
        expected = {
            "intent": kwargs["intent"], "attempt": kwargs["attempt"],
            "prompt_sha256": kwargs["prompt_sha256"],
            "manifest_sha256": self.delegate.manifest_sha256,
            "backend_mode": self.delegate.backend_mode,
            "budget_class": kwargs["budget_class"],
            "request_metadata_sha256": sha256_bytes(canonical_json(metadata).encode()),
            "backend_identity_sha256": sha256_bytes(canonical_json(kwargs.get("backend_identity", {})).encode()),
        }
        if existing:
            if any(existing.get(key) != value for key, value in expected.items()):
                raise GuardrailStop(f"cached_request_contract_drift:{request_id}")
            if existing["status"] != "completed":
                failure = read(failure_path)
                error_type = type(failure["error_type"], (Exception,), {})
                append(self.run / "audit/call_events.jsonl", {
                    "event": "known_failure_replayed_no_send", "at_utc": now(),
                    "root_id": self.root_id, "request_id": request_id,
                })
                raise error_type(failure["error_message"])
            try:
                result = self.delegate.invoke(**kwargs)
            except Exception as error:
                raise GuardrailStop(f"completed_cache_verification_failed:{request_id}:{error}") from error
            append(self.run / "audit/call_events.jsonl", {
                "event": "completed_cache_reused_no_send", "at_utc": now(),
                "root_id": self.root_id, "request_id": request_id,
            })
            return result
        if len(list(self.delegate.state_dir.glob("*.json"))) >= LIMIT_PER_ROOT:
            raise GuardrailStop("root_logical_budget_exhausted")
        if len(states) >= LIMIT_TOTAL:
            raise GuardrailStop("experiment_logical_budget_exhausted")
        if self.before_request is not None:
            self.before_request(request_id, kwargs["request_metadata"])
        append(self.run / "audit/call_events.jsonl", {
            "event": "new_logical_call", "at_utc": now(), "root_id": self.root_id,
            "request_id": request_id, **scope, "experiment_call_index": len(states) + 1,
        })
        try:
            result = self.delegate.invoke(**kwargs)
        except Exception as error:
            reservation = self.delegate.read(request_id)
            if reservation is None:
                raise GuardrailStop(f"ledger_failure_before_reservation:{request_id}:{error}") from error
            atomic_json(failure_path, {
                "request_id": request_id, "error_type": type(error).__name__,
                "error_message": str(error), "at_utc": now(),
                "reservation_sha256": sha256_bytes(canonical_json(reservation).encode()),
                "automatic_resend": False, "billing_status": "unknown_no_completion_recorded",
            })
            append(self.run / "audit/call_events.jsonl", {
                "event": "logical_call_failed", "at_utc": now(), "root_id": self.root_id,
                "request_id": request_id, "error_type": type(error).__name__,
                "billing_status": "unknown_no_completion_recorded",
            })
            raise
        append(self.run / "audit/call_events.jsonl", {
            "event": "logical_call_completed", "at_utc": now(), "root_id": self.root_id,
            "request_id": request_id, "usage": result.metadata.get("usage"),
            "finish_reason": result.metadata.get("finish_reason"),
        })
        if self.after_response is not None:
            self.after_response(request_id, result)
        return result
