"""Three Miner reservations only, around the original ledger and HTTP client.

Transport retries belong exclusively to the original client. An observed failed
logical request consumes its reservation and is replayed without another send.
The callbacks check frozen identities/tokens before reserving and current usage
after the original ledger has saved the completed response.
"""
from __future__ import annotations

from pathlib import Path

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import (
    GuardrailStop, ObservedClient, append, now, exclusive_run,
)

FIRST_ROOTS = tuple(f"cc-train-episode-{n}" for n in (198480, 122711, 197635))
PLAN_TOTAL = LIMIT_TOTAL = 3
LIMIT_PER_ROOT = 1
MODE = "P2_2_MINER_BINDING_V2"
VERSION = "h4_r02_miner_binding_v2"
MINER_INTENT = "h4_k3_intent_miner"
BUDGET_CLASS = "fresh_qa_generation"
MAX_OUTPUT_TOKENS = 4096


def root_path(run, root_id):
    return Path(run) / "qa/shards/shard-00/roots" / root_id


def state_paths(run):
    # Include unexpected shards so moving a reservation cannot restore budget.
    return sorted((Path(run) / "qa/shards").glob("*/roots/*/ledger/requests/*.json"))


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _scope(root_id, request_id, *, intent, attempt, budget_class, metadata):
    """Validate request identity, never infer Writer eligibility from contracts."""
    if root_id not in FIRST_ROOTS or request_id != f"{root_id}-intent-miner":
        raise GuardrailStop(f"request_outside_miner_only_scope:{request_id}")
    if intent != MINER_INTENT or type(attempt) is not int or attempt != 1:
        raise GuardrailStop(f"miner_role_or_attempt_drift:{request_id}")
    if budget_class != BUDGET_CLASS:
        raise GuardrailStop(f"miner_budget_class_drift:{request_id}")
    if not isinstance(metadata, dict):
        raise GuardrailStop(f"miner_metadata_invalid:{request_id}")
    if (type(metadata.get("max_tokens")) is not int or
            metadata["max_tokens"] != MAX_OUTPUT_TOKENS or
            metadata.get("temperature") != 0.1 or
            metadata.get("request_id") != request_id):
        raise GuardrailStop(f"miner_request_config_drift:{request_id}")
    if "seed" in metadata:
        raise GuardrailStop(f"miner_seed_forbidden:{request_id}")
    return {"intent_id": None, "mode": "intent_miner", "attempt": "primary"}


def audit_reservations(run):
    """Count successes and failures; block unknown paid outcomes and scope drift."""
    paths = state_paths(run)
    if len(paths) > LIMIT_TOTAL:
        raise GuardrailStop("experiment_logical_budget_exceeded")
    counts = {}
    for path in paths:
        state = read(path)
        root = path.parents[2]
        root_id = root.name
        counts[root_id] = counts.get(root_id, 0) + 1
        if (root_id not in FIRST_ROOTS or counts[root_id] > LIMIT_PER_ROOT or
                root.resolve() != root_path(run, root_id).resolve()):
            raise GuardrailStop("root_scope_or_budget_violation")
        request_id = state.get("request_id")
        metadata = state.get("request_metadata")
        _scope(root_id, request_id, intent=state.get("intent"),
               attempt=state.get("attempt"), budget_class=state.get("budget_class"),
               metadata=metadata)
        if (path.name != f"{request_id}.json" or
                state.get("budget_limit") != LIMIT_PER_ROOT or
                state.get("total_budget_limit") != LIMIT_PER_ROOT or
                metadata.get("assigned_mode") != MODE or
                metadata.get("root_id") != root_id or
                metadata.get("mode") != "intent_miner" or
                metadata.get("attempt") != "primary"):
            raise GuardrailStop("reservation_contract_scope_drift")
        if state.get("request_metadata_sha256") != sha256_bytes(canonical_json(metadata).encode()):
            raise GuardrailStop("reservation_metadata_hash_drift")
        if state.get("status") != "completed":
            failure = root / "ledger/observed_failures" / path.name
            if not failure.is_file():
                raise GuardrailStop(f"unknown_paid_status_no_automatic_resend:{request_id}")
            observed = read(failure)
            if (observed.get("request_id") != request_id or
                    observed.get("reservation_sha256") != sha256_bytes(canonical_json(state).encode()) or
                    observed.get("automatic_resend") is not False):
                raise GuardrailStop("observed_failure_reservation_drift")
    return paths


class BoundedLedger:
    """Expose the shared ledger API with a separate, exhausted-at-three budget."""
    def __init__(self, root_dir, manifest_sha256, run, root_id, *,
                 before_request=None, after_response=None, backend_mode="live"):
        self.root_dir, self.run, self.root_id = Path(root_dir), Path(run), root_id
        if (root_id not in FIRST_ROOTS or
                self.root_dir.resolve() != root_path(self.run, root_id).resolve()):
            raise GuardrailStop("root_outside_miner_only_acceptance")
        if backend_mode not in ("live", "mock"):
            raise GuardrailStop("miner_backend_mode_invalid")
        if backend_mode == "live" and (before_request is None or after_response is None):
            raise GuardrailStop("live_miner_requires_freeze_token_and_usage_guards")
        self.delegate = RequestLedger(self.root_dir, manifest_sha256, backend_mode)
        self.before_request = before_request
        self.after_response = after_response

    def invoke(self, **kwargs):
        request_id = kwargs["request_id"]
        scope = _scope(self.root_id, request_id, intent=kwargs.get("intent"),
                       attempt=kwargs.get("attempt"), budget_class=kwargs.get("budget_class"),
                       metadata=kwargs.get("request_metadata"))
        states = audit_reservations(self.run)
        existing = self.delegate.read(request_id)
        failure_path = self.root_dir / "ledger/observed_failures" / f"{request_id}.json"
        metadata = {**kwargs["request_metadata"], "assigned_mode": MODE,
                    "root_id": self.root_id, **scope,
                    "original_budget_limit": kwargs.get("budget_limit"),
                    "original_total_budget_limit": kwargs.get("total_budget_limit")}
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
            # A previous post-completion usage block remains a block on replay.
            if self.after_response is not None:
                self.after_response(request_id, result)
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
            if reservation.get("status") == "completed":
                raise GuardrailStop(f"ledger_failure_after_completion:{request_id}:{error}") from error
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


def checkpoint_root(root):
    """Freeze completed Miner artifacts, including recorded ordinary failures."""
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
