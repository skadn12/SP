"""Serial, fail-closed R01 bootstrap for feedback namespaces.

The default mode is offline and only inspects the frozen batch plan.  The
explicit ``--live`` mode performs exactly two Memory Adds per pending root:
Parent followed by R01.  It never generates R02 and never retries an Add whose
outcome is uncertain.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.async_write import (
    DEFAULT_POLL_DELAYS_SECONDS,
    submit_add_and_wait,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.client import (
    MemoraxClient,
    load_memorax_config,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.memory_context import (
    audit_bucket_delta_visibility,
    normalize_bucket_snapshot,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.payloads import (
    build_bucket_search_payload,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.write_state import (
    apply_visibility_result,
    new_write_state,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_PLAN_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_v1_live_3"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_execution_v1_live_1"
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
FEEDBACK_PREFIX = "gxy-selfplay:round1-paired-feedback:"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing_file:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    """Write an artifact and its digest atomically enough for interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2,
                      sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)
    sidecar = path.with_name(path.name + ".sha256")
    side_tmp = sidecar.with_name(sidecar.name + f".tmp.{os.getpid()}")
    side_tmp.write_text(_sha(path) + "\n", encoding="utf-8")
    os.replace(side_tmp, sidecar)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"jsonl_row_not_object:{path}")
            rows.append(value)
    return rows


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"plan_sidecar_missing:{path}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()
    actual = _sha(path)
    if not expected or expected[0] != actual:
        raise RuntimeError(f"plan_sidecar_mismatch:{path}")
    return actual


def _search(client: MemoraxClient, *, root_id: str, namespace: str,
            request_id: str) -> dict[str, Any]:
    query = build_bucket_search_payload(
        root_id=root_id, limit=100, max_buckets_per_type=500,
        user_namespace=namespace)
    response = client.search_buckets(query["payload"], request_id=request_id)
    return normalize_bucket_snapshot(
        response, root_id=root_id, user_id=query["payload"]["user_id"])


def _validate_intent(intent: Mapping[str, Any], *, root_id: str,
                    user_id: str, round_index: int) -> None:
    if intent.get("live_submission") is not False:
        raise RuntimeError(f"intent_not_offline_frozen:{root_id}:R{round_index:02d}")
    if str(intent.get("root_id") or "") != root_id:
        raise RuntimeError(f"intent_root_mismatch:{root_id}:R{round_index:02d}")
    raw_round = intent.get("round_index")
    if raw_round is None or int(raw_round) != round_index:
        raise RuntimeError(f"intent_round_mismatch:{root_id}:R{round_index:02d}")
    if str(intent.get("user_id") or "") != user_id:
        raise RuntimeError(f"intent_user_mismatch:{root_id}:R{round_index:02d}")
    payload = intent.get("payload")
    if not isinstance(payload, Mapping) or str(payload.get("user_id") or "") != user_id:
        raise RuntimeError(f"payload_user_mismatch:{root_id}:R{round_index:02d}")
    if round_index == 0 and str(intent.get("session_id") or "") != f"{root_id}:memorax:R00":
        raise RuntimeError(f"parent_session_mismatch:{root_id}")
    if round_index == 1:
        if str(intent.get("session_id") or "") != f"{root_id}:memorax:R01":
            raise RuntimeError(f"r01_session_mismatch:{root_id}")
        if any(str(ref).upper().startswith("R02") for ref in intent.get("turn_refs") or []):
            raise RuntimeError(f"r02_ref_in_r01_intent:{root_id}")


def _root_report(root_dir: Path) -> Path:
    return root_dir / "final_r01_bootstrap_report.json"


def _quarantine(root_dir: Path, *, root_id: str, reason: str,
                memorax_calls: int) -> dict[str, Any]:
    report = {
        "status": "QUARANTINED",
        "root_id": root_id,
        "reason": reason,
        "memorax_calls": memorax_calls,
        "parent_add_calls": 0,
        "r01_child_add_calls": 0,
        "r02_generation": 0,
        "r02_add_calls": 0,
        "automatic_retry": False,
        "next_action": "manual reconciliation; do not resubmit an uncertain Add",
    }
    _write(_root_report(root_dir), report)
    return report


def bootstrap_root(client: MemoraxClient, row: Mapping[str, Any], *, root_dir: Path,
                   allow_existing_report: bool = True) -> dict[str, Any]:
    root_id = str(row.get("root_id") or "")
    namespace = str(row.get("namespace") or "")
    user_id = str(row.get("user_id") or "")
    if not root_id or not namespace.startswith(FEEDBACK_PREFIX) or not user_id:
        raise RuntimeError(f"bootstrap_row_identity_invalid:{root_id}")
    existing = _root_report(root_dir)
    if allow_existing_report and existing.is_file():
        report = _load(existing)
        if report.get("status") in {"PASS", "QUARANTINED"}:
            return report
    if root_dir.exists() and any(root_dir.iterdir()):
        raise RuntimeError(f"partial_root_output_requires_manual_reconciliation:{root_id}")
    root_dir.mkdir(parents=True, exist_ok=False)
    (root_dir / "memory").mkdir()
    calls_before = len(client.calls)

    try:
        empty = _search(client, root_id=root_id, namespace=namespace,
                        request_id=f"{root_id}-feedback-bootstrap-empty")
        _write(root_dir / "memory/namespace_empty_snapshot.json", empty)
        if sum(empty["bucket_counts"].values()) != 0:
            return _quarantine(root_dir, root_id=root_id,
                               reason="target_feedback_namespace_not_empty",
                               memorax_calls=len(client.calls) - calls_before)

        parent_intent = row.get("parent_add_intent")
        r01_intent = row.get("r01_add_intent")
        if not isinstance(parent_intent, Mapping) or not isinstance(r01_intent, Mapping):
            raise RuntimeError(f"bootstrap_intents_missing:{root_id}")
        _validate_intent(parent_intent, root_id=root_id, user_id=user_id,
                         round_index=0)
        _validate_intent(r01_intent, root_id=root_id, user_id=user_id,
                         round_index=1)

        before = empty
        add_counts = {"parent": 0, "r01": 0}
        stage_specs = (("parent", parent_intent, "parent_add"),
                       ("r01", r01_intent, "r01_child_add"))
        stage_records: dict[str, Any] = {}
        for stage, intent, label in stage_specs:
            _write(root_dir / f"memory/{stage}_intent.json", dict(intent))
            state = new_write_state(intent=intent,
                                    stage=f"feedback_{label}_memory_add")
            _write(root_dir / f"memory/{stage}_write_state.json", state)

            def on_state(value: Mapping[str, Any], *, stage_name: str = stage) -> None:
                _write(root_dir / f"memory/{stage_name}_write_state.json",
                       dict(value))

            try:
                completion = submit_add_and_wait(
                    gateway=client, intent=intent,
                    poll_delays_seconds=DEFAULT_POLL_DELAYS_SECONDS,
                    stage=f"feedback_{label}_memory_add",
                    on_state_change=on_state)
            except Exception as error:
                # The Add may have been accepted before transport/polling
                # failed. Persist the failure and quarantine; never resubmit.
                _write(root_dir / f"memory/{stage}_failure.json", {
                    "status": "UNCERTAIN_ADD",
                    "stage": stage,
                    "error": f"{type(error).__name__}:{error}",
                    "automatic_retry": False,
                })
                return _quarantine(
                    root_dir, root_id=root_id,
                    reason=f"uncertain_{stage}_add_no_retry",
                    memorax_calls=len(client.calls) - calls_before)
            _write(root_dir / f"memory/{stage}_completion.json", completion)
            add_counts[stage] = 1
            after = _search(
                client, root_id=root_id, namespace=namespace,
                request_id=f"{root_id}-feedback-bootstrap-{stage}-visible")
            visibility = audit_bucket_delta_visibility(before, after)
            visibility["stage"] = stage
            _write(root_dir / f"memory/{stage}_visibility_audit.json", visibility)
            state_after = apply_visibility_result(
                _load(root_dir / f"memory/{stage}_write_state.json"),
                visible=bool(visibility.get("bucket_visibility_verified")))
            _write(root_dir / f"memory/{stage}_write_state.json", state_after)
            stage_records[stage] = {
                "completion": completion,
                "visibility": visibility,
                "write_state": state_after,
            }
            if not visibility.get("bucket_visibility_verified"):
                return _quarantine(
                    root_dir, root_id=root_id,
                    reason=f"{stage}_add_not_visible",
                    memorax_calls=len(client.calls) - calls_before)
            before = after

        report = {
            "status": "PASS",
            "root_id": root_id,
            "namespace": namespace,
            "user_id": user_id,
            "parent_add_calls": add_counts["parent"],
            "r01_child_add_calls": add_counts["r01"],
            "r02_generation": 0,
            "r02_add_calls": 0,
            "memorax_calls": len(client.calls) - calls_before,
            "automatic_retry": False,
            "old_control_modified": False,
            "stages": stage_records,
            "next_status": "READY_FOR_R02_FEEDBACK_GENERATION",
        }
        _write(_root_report(root_dir), report)
        return report
    except Exception as error:
        # Configuration/schema failures are local blockers, not Add retries.
        return _quarantine(
            root_dir, root_id=root_id,
            reason=f"local_bootstrap_contract_error:{type(error).__name__}:{error}",
            memorax_calls=len(client.calls) - calls_before)


def _plan_rows(plan_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _verify_sidecar(plan_root / "preflight_report.json")
    _verify_sidecar(plan_root / "r01_feedback_bootstrap_plan.jsonl")
    preflight = _load(plan_root / "preflight_report.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("batch_preflight_not_pass")
    rows = _load_jsonl(plan_root / "r01_feedback_bootstrap_plan.jsonl")
    if len(rows) != int(preflight.get("formal_root_count") or 0):
        raise RuntimeError("bootstrap_plan_count_mismatch")
    preflight["_preflight_sha256"] = _sha(plan_root / "preflight_report.json")
    preflight["_bootstrap_plan_sha256"] = _sha(
        plan_root / "r01_feedback_bootstrap_plan.jsonl")
    return preflight, rows


def run(*, plan_root: Path = DEFAULT_PLAN_ROOT,
        output_root: Path = DEFAULT_RUN_ROOT,
        credentials_path: Path = DEFAULT_CREDENTIALS,
        allow_insecure_http: bool = True, live: bool = False) -> dict[str, Any]:
    preflight, rows = _plan_rows(plan_root)
    pending = [row for row in rows if row.get("status") == "REQUIRED_BEFORE_R02"]
    if not live:
        return {
            "status": "OFFLINE_PLAN_ONLY",
            "plan_root": str(plan_root.resolve()),
            "pending_root_ids": [str(row.get("root_id")) for row in pending],
            "live_started": False,
            "memorax_calls": 0,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("bootstrap_batch_already_running") from error
    client = MemoraxClient(load_memorax_config(
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http))
    _write(output_root / "run_status.json", {
        "status": "LIVE_STARTED", "started_at": _now(),
        "plan_root": str(plan_root.resolve()),
        "preflight_sha256": preflight["_preflight_sha256"],
        "bootstrap_plan_sha256": preflight["_bootstrap_plan_sha256"],
        "pending_root_count": len(pending), "serial": True,
        "r02_generation": 0, "r02_add_calls": 0,
    })
    reports: list[dict[str, Any]] = []
    try:
        for row in pending:
            root_id = str(row.get("root_id") or "")
            report = bootstrap_root(
                client, row, root_dir=output_root / "roots" / root_id)
            reports.append(report)
            _write(output_root / "progress.json", {
                "completed_roots": [str(item.get("root_id")) for item in reports],
                "remaining_roots": [str(item.get("root_id")) for item in pending[len(reports):]],
                "reports": reports,
                "memorax_calls": len(client.calls),
            })
        status = "PASS" if all(item.get("status") == "PASS" for item in reports) else "PARTIAL_QUARANTINE"
        result = {
            "status": status,
            "plan_root": str(plan_root.resolve()),
            "preflight_sha256": preflight["_preflight_sha256"],
            "bootstrap_plan_sha256": preflight["_bootstrap_plan_sha256"],
            "output_root": str(output_root.resolve()),
            "pending_root_count": len(pending),
            "passed_root_ids": [str(item.get("root_id")) for item in reports if item.get("status") == "PASS"],
            "quarantined_root_ids": [str(item.get("root_id")) for item in reports if item.get("status") != "PASS"],
            "parent_add_calls": sum(int(item.get("parent_add_calls") or 0) for item in reports),
            "r01_child_add_calls": sum(int(item.get("r01_child_add_calls") or 0) for item in reports),
            "r02_generation": 0, "r02_add_calls": 0,
            "memorax_calls": len(client.calls), "automatic_retry": False,
            "serial": True, "completed_at": _now(),
        }
        _write(output_root / "final_bootstrap_batch_report.json", result)
        return result
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--memorax-credentials", type=Path,
                        default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="perform the two Memory Adds per pending root")
    args = parser.parse_args(argv)
    result = run(plan_root=args.plan_root.resolve(),
                 output_root=args.output_root.resolve(),
                 credentials_path=args.memorax_credentials.resolve(),
                 allow_insecure_http=args.allow_insecure_memorax_http,
                 live=args.live)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"OFFLINE_PLAN_ONLY", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
