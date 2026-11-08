"""Bounded, alternating on/off scheduling around the unchanged R1 QA generator.

Prepare is entirely offline. Live requires the separately signed-off P0 report.
The original client owns every transport retry; this module only observes it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import urllib.error
import urllib.request
from unittest.mock import patch

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from ..h4_batch30_memory_event_k3_probe_v1 import qa_generation_only as original
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa, MINIMUM_COUNTS
from ..ledger import RequestLedger
from ..live_backends import RATE_LIMIT_RETRY_LIMIT

VERSION = "h4_r02_feedback_ablation_qa_schedule_v1"
ROOTS = ["cc-train-episode-122711", "cc-train-episode-136202", "cc-train-episode-197635",
         "cc-train-episode-198480", "cc-train-episode-254852", "cc-train-episode-94289"]
ARMS = ("feedback_off", "feedback_on")
MODES = ("contract_guided", "free_exploration")
PLAN_PER_ROOT_ARM = 7
LIMIT_PER_ROOT_ARM = 13
PLAN_TOTAL = 84
LIMIT_TOTAL = 156


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def write_frozen(path: Path, value: dict):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"frozen_input_drift:{path}")
    else:
        atomic_json(path, value)


class GuardrailStop(BaseException):
    """A scheduler violation must bypass the old generator's parsing fallback."""


@contextmanager
def exclusive_run(run: Path):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "qa_scheduler.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("qa_scheduler_already_running") from error
        yield


def request_scope(root_dir: Path, root_id: str, request_id: str):
    if request_id == f"{root_id}-intent-miner":
        return {"intent_id": None, "mode": "intent_miner", "attempt": "primary"}
    contracts = read(root_dir / "reasoning_contracts.json")
    ids = [str(row["intent_id"]) for row in contracts]
    if len(ids) > 3 or len(ids) != len(set(ids)):
        raise GuardrailStop("invalid_contract_scope")
    for intent_id in ids:
        for mode in MODES:
            for label, attempt in (("generator", "primary"), ("repair", "repair")):
                if request_id == f"{root_id}-{mode}-{label}-{intent_id}":
                    if attempt == "repair":
                        primary = root_dir / "ledger/requests" / f"{root_id}-{mode}-generator-{intent_id}.json"
                        if not primary.is_file():
                            raise GuardrailStop("repair_without_primary")
                    return {"intent_id": intent_id, "mode": mode, "attempt": attempt}
    raise GuardrailStop(f"request_outside_frozen_scope:{request_id}")


def state_paths(run: Path):
    return sorted(path for arm in ARMS for path in (run / arm / "qa/shards").glob("*/roots/*/ledger/requests/*.json"))


class BoundedLedger:
    """Delegate exact caching to RequestLedger; constrain its generic 10,000 cap.

    Explicitly observed failed calls are replayed as the same exception, never
    resent. An interrupted reservation with no recorded outcome stops live work.
    """
    def __init__(self, root_dir: Path, manifest_sha256: str, run: Path, arm: str, root_id: str,
                 backend_mode="live"):
        self.root_dir, self.run, self.arm, self.root_id = root_dir, run, arm, root_id
        self.delegate = RequestLedger(root_dir, manifest_sha256, backend_mode)

    def invoke(self, **kwargs):
        request_id = kwargs["request_id"]
        scope = request_scope(self.root_dir, self.root_id, request_id)
        existing = self.delegate.read(request_id)
        failure_path = self.root_dir / "ledger/observed_failures" / f"{request_id}.json"
        metadata = {**kwargs.get("request_metadata", {}), "arm": self.arm, "root_id": self.root_id,
                    **scope, "original_budget_limit": kwargs["budget_limit"],
                    "original_total_budget_limit": kwargs["total_budget_limit"]}
        kwargs.update(request_metadata=metadata, budget_limit=LIMIT_PER_ROOT_ARM,
                      total_budget_limit=LIMIT_PER_ROOT_ARM)
        expected_contract = {
            "intent": kwargs["intent"], "attempt": kwargs["attempt"],
            "prompt_sha256": kwargs["prompt_sha256"],
            "manifest_sha256": self.delegate.manifest_sha256,
            "backend_mode": self.delegate.backend_mode, "budget_class": kwargs["budget_class"],
            "request_metadata_sha256": sha256_bytes(canonical_json(metadata).encode()),
            "backend_identity_sha256": sha256_bytes(canonical_json(kwargs.get("backend_identity", {})).encode()),
        }
        if existing:
            if any(existing.get(key) != value for key, value in expected_contract.items()):
                raise GuardrailStop(f"cached_request_contract_drift:{request_id}")
            if existing["status"] != "completed":
                if not failure_path.is_file():
                    raise GuardrailStop(f"unknown_paid_status_no_automatic_resend:{request_id}")
                failure = read(failure_path)
                if failure["reservation_sha256"] != sha256_bytes(canonical_json(existing).encode()):
                    raise GuardrailStop(f"failure_reservation_drift:{request_id}")
                error_type = type(failure["error_type"], (Exception,), {})
                raise error_type(failure["error_message"])
            try:
                return self.delegate.invoke(**kwargs)
            except Exception as error:
                raise GuardrailStop(f"completed_cache_verification_failed:{request_id}:{error}") from error
        if len(list(self.delegate.state_dir.glob("*.json"))) >= LIMIT_PER_ROOT_ARM:
            raise GuardrailStop("root_arm_logical_budget_exhausted")
        if len(state_paths(self.run)) >= LIMIT_TOTAL:
            raise GuardrailStop("experiment_logical_budget_exhausted")
        append(self.run / "audit/qa_call_order.jsonl", {
            "event": "new_logical_call", "at_utc": now(), "arm": self.arm,
            "root_id": self.root_id, "request_id": request_id, **scope,
            "experiment_call_index": len(state_paths(self.run)) + 1,
        })
        try:
            return self.delegate.invoke(**kwargs)
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
            raise


class ObservedClient:
    """Observe the existing HTTP implementation without owning retries or payloads."""
    def __init__(self, delegate, event_path: Path, arm: str, root_id: str):
        self.delegate, self.config = delegate, delegate.config
        self.event_path, self.arm, self.root_id = event_path, arm, root_id

    def complete(self, *, prompt: str, max_tokens: int, json_object: bool):
        identity = {"arm": self.arm, "root_id": self.root_id,
                    "prompt_sha256": sha256_bytes(prompt.encode()), "max_tokens": max_tokens}
        real_urlopen = urllib.request.urlopen
        attempts = []

        def observed_urlopen(*args, **kwargs):
            index = len(attempts) + 1
            row = {**identity, "transport_attempt": index, "started_at_utc": now()}
            attempts.append(row)
            append(self.event_path, {**row, "event": "http_attempt_started"})
            try:
                response = real_urlopen(*args, **kwargs)
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__,
                           http_status=getattr(error, "code", None),
                           timeout=isinstance(error, (TimeoutError, socket.timeout)) or
                           isinstance(getattr(error, "reason", None), (TimeoutError, socket.timeout)))
                append(self.event_path, {**row, "event": "http_attempt_finished"})
                raise
            row.update(status="headers_received", http_status=getattr(response, "status", None))
            append(self.event_path, {**row, "event": "http_attempt_finished"})
            return response

        try:
            with patch("urllib.request.urlopen", observed_urlopen):
                response = self.delegate.complete(prompt=prompt, max_tokens=max_tokens, json_object=json_object)
        except Exception as error:
            append(self.event_path, {**identity, "event": "logical_call_transport_outcome",
                                    "at_utc": now(), "status": "failed", "error_type": type(error).__name__,
                                    "timeout": isinstance(error, (TimeoutError, socket.timeout)) or
                                    isinstance(getattr(error, "reason", None), (TimeoutError, socket.timeout)),
                                    "transport_attempt_count": len(attempts),
                                    "completion_received": False, "billing_status": "unknown"})
            raise
        append(self.event_path, {**identity, "event": "logical_call_transport_outcome", "at_utc": now(),
                                "status": "completed", "transport_attempt_count": len(attempts),
                                "completion_received": True, "billing_status": "completion_received",
                                "usage": response.metadata.get("usage")})
        if response.metadata.get("transport_attempt_count") != len(attempts):
            raise GuardrailStop("transport_observer_count_disagrees_with_original_client")
        return response


def schedule_rows():
    rows = []
    for index, root_id in enumerate(ROOTS):
        arms = ARMS if index % 2 == 0 else ARMS[::-1]
        offset = len(rows)
        rows.extend({"order": offset + slot + 1, "root_id": root_id, "arm": arm}
                    for slot, arm in enumerate(arms))
    return rows


def code_sources():
    # Freeze transitive generation dependencies plus scheduler/auditor themselves.
    method = Path(__file__).resolve().parent.parent
    paths = [method / p for p in ["core.py", "backends.py", "ledger.py", "live_backends.py",
             "simple_append_v1/live_runtime.py", "h4_batch30_memory_event_k3_probe_v1/qa_generation_only.py",
             "h4_batch30_memory_event_k3_probe_v1/plan.py", "h4_batch30_memory_event_k3_probe_v1/execute.py"]]
    for package in ("h4_qa_multiroot_k3_t08_probe_v1", "h4_qa_mining_probe_v1", "h4_qa_mining_hybrid_generation_v1"):
        paths.extend((method / package).glob("*.py"))
    paths += [Path(__file__).resolve(), Path(__file__).with_name("prompt_audit.py").resolve()]
    return [{"path": str(p), "sha256": sha256_file(p)} for p in sorted(set(paths))]


def prepare(args):
    run = args.run.resolve()
    manifests = {}
    for arm in ARMS:
        output = run / arm / "qa"
        context_path = run / "inputs" / arm / "feedback_context_manifest.json"
        original.inventory_phase(argparse.Namespace(source_run=args.source_run, output_run=output,
                                 shard=[], allow_partial=False, feedback_context_manifest=context_path))
        manifest = read(output / "manifest.json")
        if {row["root_id"] for row in manifest["roots"]} != set(ROOTS) or len(manifest["roots"]) != 6:
            raise RuntimeError("six_fixed_roots_required")
        contexts = original._verify_feedback_context(manifest, context_path)
        for entry in manifest["roots"]:
            context = original._load_feedback_context(contexts["roots"][entry["root_id"]], entry["root_id"])
            if arm == "feedback_off" and context != {"intent_suffix": "", "slot_suffixes": ["", "", ""]}:
                raise RuntimeError("off_context_is_not_original_prompt")
            if arm == "feedback_on" and (not context["intent_suffix"] or not all(context["slot_suffixes"])):
                raise RuntimeError("on_context_empty")
            original._prepare_root(entry, output)
        manifests[arm] = {"path": str(output / "manifest.json"), "sha256": sha256_file(output / "manifest.json")}
    payload = {"version": VERSION, "run": str(run), "source_run": str(args.source_run.resolve()),
               "arms": list(ARMS), "root_ids": ROOTS, "schedule": schedule_rows(), "qa_manifests": manifests,
               "api_config_path": str(args.api_config.resolve()), "api_config_sha256": sha256_file(args.api_config),
               "model": "Qwen/Qwen3-14B", "temperature": 0.1, "thinking": False,
               "generation_seed_policy": "not_sent_by_original_client", "intent_max_tokens": 4096,
               "writer_max_tokens": 8192, "planned_per_root_arm": PLAN_PER_ROOT_ARM,
               "logical_limit_per_root_arm": LIMIT_PER_ROOT_ARM, "planned_total": PLAN_TOTAL,
               "logical_limit_total": LIMIT_TOTAL, "maximum_miner_calls_per_root_arm": 1,
               "maximum_primary_calls_per_root_arm": 6, "maximum_repair_calls_per_root_arm": 6,
               "repair_policy": "one_original_full_batch_replaces_primary_if_below_minimum",
               "minimum_counts": MINIMUM_COUNTS, "guided_requested_counts": MINIMUM_COUNTS,
               "free_requested_counts": {k: 2 * v for k, v in MINIMUM_COUNTS.items()},
               "guided_free_requested_ratio": [1, 2], "transport_retry_owner": "original_client_unchanged",
               "original_http_429_retry_limit": RATE_LIMIT_RETRY_LIMIT, "force_supported": False,
               "code_sources": code_sources()}
    write_frozen(run / "qa_schedule.json", payload)
    return payload


def verify_schedule(run: Path):
    schedule = read(run / "qa_schedule.json")
    if schedule["schedule"] != schedule_rows() or schedule["version"] != VERSION:
        raise GuardrailStop("frozen_schedule_drift")
    if schedule["code_sources"] != code_sources():
        raise GuardrailStop("frozen_generation_code_drift")
    if sha256_file(Path(schedule["api_config_path"])) != schedule["api_config_sha256"]:
        raise GuardrailStop("api_configuration_file_drift")
    for arm in ARMS:
        entry = schedule["qa_manifests"][arm]
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise GuardrailStop(f"qa_manifest_drift:{arm}")
        manifest = read(Path(entry["path"]))
        original._verify_feedback_context(manifest, None)
        if sha256_file(Path(manifest["source_run"]) / "batch_plan.json") != manifest["source_batch_plan_sha256"]:
            raise GuardrailStop("source_batch_plan_drift")
    return schedule


def checkpoint_root(root: Path):
    path = root / "qa_completion_checkpoint.json"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p != path)
    write_frozen(path, {"version": VERSION, "files": {str(p.relative_to(root)): sha256_file(p) for p in files}})


def verify_checkpoint(root: Path):
    path = root / "qa_completion_checkpoint.json"
    if not path.is_file():
        return False
    checkpoint = read(path)
    for name, expected in checkpoint["files"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise GuardrailStop(f"completed_artifact_drift:{target}")
    return True


def summarize(run: Path):
    all_rows, arm_reports = [], {}
    for arm in ARMS:
        reports = []
        qa = run / arm / "qa"
        for root_id in ROOTS:
            root = original._root_dir(qa, "shard-00", root_id)
            report_path = root / "qa_generation_report.json"
            if report_path.is_file():
                reports.append(read(report_path))
            rows = [read(p) for p in (root / "ledger/requests").glob("*.json")]
            all_rows.extend(rows)
        arm_report = {"status": "completed" if len(reports) == 6 else "in_progress", "root_count": len(reports),
                      "expected_root_count": 6, "root_reports": reports,
                      "candidate_count": sum(r["candidate_count"] for r in reports),
                      "valid_candidate_count": sum(r["valid_candidate_count"] for r in reports),
                      "source_run": read(qa / "manifest.json")["source_run"], "output_run": str(qa),
                      "solver_calls": 0, "semantic_admission_applied": False, "difficulty_filter_applied": False,
                      "errors": [r["root_id"] for r in reports if r["status"] != "ready"]}
        atomic_json(qa / "generation_only_report.json", arm_report)
        arm_reports[arm] = arm_report
    completed = [row for row in all_rows if row["status"] == "completed"]
    transport_file = run / "audit/qa_transport_events.jsonl"
    transport = [json.loads(line) for line in transport_file.read_text().splitlines()] if transport_file.is_file() else []
    outcomes = [row for row in transport if row["event"] == "logical_call_transport_outcome"]
    attempts = [row for row in transport if row["event"] == "http_attempt_started"]
    usage = Counter()
    for row in completed:
        usage.update({k: v for k, v in (row.get("response_metadata", {}).get("usage") or {}).items()
                      if isinstance(v, (float, int))})
    result = {"version": VERSION, "status": "completed" if all(r["root_count"] == 6 for r in arm_reports.values()) else "in_progress",
              "completed_root_arms": sum(r["root_count"] for r in arm_reports.values()), "expected_root_arms": 12,
              "planned_logical_calls": PLAN_TOTAL, "maximum_logical_calls": LIMIT_TOTAL,
              "logical_calls": len(all_rows), "completed_calls": len(completed),
              "unknown_billing_status_calls": len(all_rows) - len(completed),
              "miner_calls": sum(row["request_metadata"]["mode"] == "intent_miner" for row in all_rows),
              "writer_primary_calls": sum(row["request_metadata"]["mode"] != "intent_miner" and row["request_metadata"]["attempt"] == "primary" for row in all_rows),
              "repair_calls": sum(row["request_metadata"]["attempt"] == "repair" for row in all_rows),
              "http_attempts": len(attempts), "http_retries": sum(max(0, row["transport_attempt_count"] - 1) for row in outcomes),
              "http_429_responses": sum(row.get("http_status") == 429 for row in transport if row["event"] == "http_attempt_finished"),
              "transport_failures": sum(row["status"] == "failed" for row in outcomes),
              "timeout_calls": sum(bool(row.get("timeout")) for row in outcomes),
              "completion_usage": dict(usage), "monetary_cost": None,
              "monetary_cost_note": "Provider response has no price; retain actual token usage and unknown billing separately.",
              "arms": arm_reports}
    atomic_json(run / "audit/qa_execution_summary.json", result)
    return result


def live(args):
    run = args.run.resolve()
    schedule = verify_schedule(run)
    preflight = read(run / "p0_preflight_report.json")
    if not (preflight.get("pass") is True or preflight.get("status") in {"PASS", "PASS_COMPLETE", "passed", "P0_PASS", "offline_preflight_passed"}):
        raise GuardrailStop("p0_preflight_not_passed")
    if args.api_config.resolve() != Path(schedule["api_config_path"]):
        raise GuardrailStop("live_api_config_path_drift")
    delegate_client = original._client(args)
    if delegate_client.config.model != schedule["model"]:
        raise GuardrailStop("generation_model_identity_mismatch")
    for slot in schedule["schedule"]:
        arm, root_id = slot["arm"], slot["root_id"]
        qa = run / arm / "qa"
        manifest = read(qa / "manifest.json")
        entry = next(row for row in manifest["roots"] if row["root_id"] == root_id)
        root = original._root_dir(qa, entry["shard"], root_id)
        if sha256_file(Path(entry["source_path"])) != entry["source_sha256"]:
            raise GuardrailStop(f"source_hash_drift:{root_id}")
        if verify_checkpoint(root):
            print(json.dumps({**slot, "status": "verified_completed_root_reused"}), flush=True)
            continue
        contexts = original._verify_feedback_context(manifest, None)
        context = original._load_feedback_context(contexts["roots"][root_id], root_id)
        root, expanded = original._prepare_root(entry, qa)
        ledger = BoundedLedger(root, manifest["manifest_sha256"], run, arm, root_id)
        client = ObservedClient(delegate_client, run / "audit/qa_transport_events.jsonl", arm, root_id)
        print(json.dumps({**slot, "status": "started", "at_utc": now()}), flush=True)
        report = generate_root_qa(expanded=expanded, root_dir=root, root_id=root_id, api_config=args.api_config,
                                  ledger=ledger, client=client, feedback_context=context)
        report.update(probe_version=original.PROBE_VERSION, solver_calls=0,
                      difficulty_filter_applied=False, semantic_admission_applied=False)
        atomic_json(root / "qa_generation_report.json", report)
        checkpoint_root(root)
        summary = summarize(run)
        print(json.dumps({**slot, "status": report["status"], "valid_candidate_count": report["valid_candidate_count"],
                          "completed_root_arms": summary["completed_root_arms"],
                          "logical_calls": summary["logical_calls"], "at_utc": now()}), flush=True)
    return summarize(run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "live", "status"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_r1_aligned_feedback_r02_qa_source_v1"))
    parser.add_argument("--api-config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    args = parser.parse_args(argv)
    try:
        if args.phase == "status":
            result = summarize(args.run.resolve())
        else:
            with exclusive_run(args.run.resolve()):
                result = prepare(args) if args.phase == "prepare" else live(args)
        print(json.dumps({k: v for k, v in result.items() if k not in {"code_sources", "arms"}}, ensure_ascii=False, indent=2), flush=True)
        return 0
    except GuardrailStop as error:
        atomic_json(args.run / "audit/qa_scheduler_stop.json", {"at_utc": now(), "reason": str(error), "automatic_resend": False})
        print(f"QA scheduler stopped: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
