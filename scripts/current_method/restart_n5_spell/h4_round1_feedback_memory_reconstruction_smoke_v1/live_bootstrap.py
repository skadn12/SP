"""Single-root live R01-only Memory bootstrap.

The command is intentionally narrower than the eventual feedback expansion:
it performs only empty checks and exactly one Parent Add plus one R01 Child Add
in each frozen namespace. It stops before any R02 generation or write.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


DEFAULT_PREFLIGHT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_2")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_3")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8")


def _search(client: MemoraxClient, *, root_id: str, namespace: str,
            request_id: str) -> dict[str, Any]:
    query = build_bucket_search_payload(root_id=root_id, limit=100,
                                        max_buckets_per_type=500,
                                        user_namespace=namespace)
    response = client.search_buckets(query["payload"], request_id=request_id)
    return normalize_bucket_snapshot(
        response, root_id=root_id, user_id=query["payload"]["user_id"])


def _run(preflight_root: Path, run_root: Path,
         credentials: Path, allow_insecure_http: bool) -> dict[str, Any]:
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"live_bootstrap_output_not_empty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    preflight = _load(preflight_root / "preflight_report.json")
    if preflight.get("status") != "PASS" or preflight.get("decision") != "pass":
        raise RuntimeError("offline_preflight_required")
    mapping = _load(preflight_root / "namespace_mapping.json")
    root_id = str(preflight.get("root_id") or "")
    if set(mapping) != {"control", "feedback"}:
        raise RuntimeError("namespace_mapping_incomplete")
    for branch in mapping:
        if not mapping[branch].get("user_id"):
            raise RuntimeError(f"namespace_user_id_missing:{branch}")
    client = MemoraxClient(load_memorax_config(
        credentials_path=credentials,
        allow_insecure_http=allow_insecure_http))
    empty: dict[str, Any] = {}
    for branch in ("control", "feedback"):
        namespace = str(mapping[branch]["namespace"])
        snapshot = _search(
            client, root_id=root_id, namespace=namespace,
            request_id=f"{root_id}-bootstrap-{branch}-empty")
        empty[branch] = {
            "namespace": namespace,
            "user_id": mapping[branch]["user_id"],
            "snapshot": snapshot,
            "bucket_total": sum(snapshot["bucket_counts"].values()),
            "empty": sum(snapshot["bucket_counts"].values()) == 0,
        }
    _write(run_root / "namespace_empty_check.json", {
        "version": "h4_round1_namespace_empty_check_v1",
        "root_id": root_id,
        "checked_at": _now(),
        "branches": empty,
        "all_empty": all(x["empty"] for x in empty.values()),
        "memorax_call_count": len(client.calls),
    })
    if not all(x["empty"] for x in empty.values()):
        report = {
            "status": "NAMESPACE_NOT_EMPTY",
            "root_id": root_id,
            "memorax_calls": len(client.calls),
            "parent_add_calls": 0,
            "r01_child_add_calls": 0,
            "r02_add_calls": 0,
            "automatic_retry": False,
        }
        _write(run_root / "final_feedback_memory_smoke_report.json", report)
        return report

    branch_reports: dict[str, Any] = {}
    for branch in ("control", "feedback"):
        namespace = str(mapping[branch]["namespace"])
        branch_root = run_root / f"{branch}_reconstruction"
        branch_root.mkdir(parents=True, exist_ok=True)
        before_parent = empty[branch]["snapshot"]
        writes: dict[str, Any] = {}
        for stage, filename, state_filename, completion_filename in (
            ("parent", "parent_add_intent.json", "parent_add_write_state.json",
             "parent_add_completion.json"),
            ("r01", "r01_child_add_intent.json", "r01_child_add_write_state.json",
             "r01_child_add_completion.json"),
        ):
            intent = _load(preflight_root / f"{branch}_reconstruction" / filename)
            if intent.get("live_submission") is not False:
                raise RuntimeError(f"offline_plan_not_frozen:{branch}:{stage}")
            if intent.get("user_id") != mapping[branch]["user_id"]:
                raise RuntimeError(f"intent_namespace_mismatch:{branch}:{stage}")
            _write(branch_root / filename, intent)
            state = new_write_state(intent=intent,
                                    stage=f"{branch}_{stage}_memory_add")
            _write(branch_root / state_filename, state)
            updates: list[dict[str, Any]] = []

            def on_state(value: Any) -> None:
                updates.append(dict(value))
                _write(branch_root / state_filename, dict(value))

            completion = submit_add_and_wait(
                gateway=client, intent=intent,
                poll_delays_seconds=DEFAULT_POLL_DELAYS_SECONDS,
                stage=f"{branch}_{stage}_memory_add",
                on_state_change=on_state)
            _write(branch_root / completion_filename, completion)
            if completion.get("verified_complete") is not True:
                raise RuntimeError(f"add_not_verified:{branch}:{stage}")
            after = _search(
                client, root_id=root_id, namespace=namespace,
                request_id=f"{root_id}-bootstrap-{branch}-{stage}-visible")
            visibility = audit_bucket_delta_visibility(before_parent, after)
            visibility["stage"] = stage
            _write(branch_root / f"{stage}_visibility_audit.json", visibility)
            state = apply_visibility_result(
                updates[-1] if updates else state,
                visible=bool(visibility["bucket_visibility_verified"]),
            )
            _write(branch_root / state_filename, state)
            if not state.get("bucket_visibility_verified"):
                raise RuntimeError(f"add_not_visible:{branch}:{stage}")
            writes[stage] = {
                "request_id": completion.get("request_id"),
                "task_id": completion.get("task_id"),
                "completion": completion,
                "visibility": visibility,
                "state": state,
            }
            before_parent = after
        branch_reports[branch] = {
            "namespace": namespace,
            "user_id": mapping[branch]["user_id"],
            "parent_add": writes["parent"],
            "r01_child_add": writes["r01"],
            "r02_add": 0,
            "r02_generation": 0,
        }

    _write(run_root / "r01_reconstruction_search_comparison.json", {
        "status": "ANCHOR_EQUIVALENT",
        "initial_both_empty": True,
        "control_r01_visible": bool(branch_reports["control"]["r01_child_add"]["visibility"]["bucket_visibility_verified"]),
        "feedback_r01_visible": bool(branch_reports["feedback"]["r01_child_add"]["visibility"]["bucket_visibility_verified"]),
        "retrieval_queries_identical": True,
        "note": "Semantic anchor comparison is limited to normalized bucket visibility; no R02 was generated.",
    })
    report = {
        "status": "ROUND1_R01_BOOTSTRAP_SMOKE_PASS",
        "next_status": "READY_FOR_FEEDBACK_R02_SINGLE_ROOT_SMOKE",
        "root_id": root_id,
        "branches": branch_reports,
        "parent_add_calls": 2,
        "r01_child_add_calls": 2,
        "r02_add_calls": 0,
        "r02_generation": 0,
        "qa_calls": 0,
        "solver_calls": 0,
        "semantic_auditor_calls": 0,
        "memorax_calls": len(client.calls),
        "automatic_retry": False,
        "old_control_modified": False,
        "test_access": 0,
    }
    _write(run_root / "memory_write_count_audit.json", {
        "status": "PASS",
        "control_reconstruction": {"parent_add": 1, "r01_child_add": 1, "r02_add": 0},
        "feedback_reconstruction": {"parent_add": 1, "r01_child_add": 1, "r02_add": 0},
        "automatic_retry": False,
    })
    _write(run_root / "final_feedback_memory_smoke_report.json", report)
    (run_root / "final_feedback_memory_smoke_report.md").write_text(
        "# R01-only Memory Bootstrap Smoke\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Root: `{root_id}`\n"
        "- Parent/R01 Add calls: `2/2`\n"
        "- R02 generation/write: `0/0`\n"
        f"- Memorax calls: `{report['memorax_calls']}`\n"
        "- Next status: `READY_FOR_FEEDBACK_R02_SINGLE_ROOT_SMOKE`\n",
        encoding="utf-8")
    md = run_root / "final_feedback_memory_smoke_report.md"
    import hashlib
    md.with_name(md.name + ".sha256").write_text(
        hashlib.sha256(md.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", type=Path, default=DEFAULT_PREFLIGHT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--memorax-credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    args = parser.parse_args(argv)
    report = _run(args.preflight_root.resolve(), args.run_root.resolve(),
                  args.memorax_credentials.resolve(), args.allow_insecure_memorax_http)
    print(json.dumps({
        "status": report["status"],
        "root_id": report.get("root_id"),
        "memorax_calls": report.get("memorax_calls", 0),
        "r02_add_calls": report.get("r02_add_calls", 0),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ROUND1_R01_BOOTSTRAP_SMOKE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
