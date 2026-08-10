"""Read-only recovery for an accepted Add that completed after local timeout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from .client import MemoraxClient, load_memorax_config
from .memory_context import compile_memory_overview, normalize_bucket_snapshot
from .payloads import build_bucket_search_payload, memorax_user_id
from .prompts import build_memory_augmented_mcq_prompt
from .write_state import (
    apply_add_response,
    apply_query_uncertainty,
    apply_task_response,
    apply_visibility_result,
    new_write_state,
)


RECOVERY_VERSION = "simple_append_memorax_async_recovery_v2"
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_memorax_v1_live_canary_2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_memorax_v1_"
    "child_memory_recovery_audit_1"
)
DEFAULT_MEMORAX_CREDENTIALS = Path("/root/.config/memorax/credentials.env")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _legacy_task_id(report: Mapping[str, Any]) -> str:
    values = {
        str(row.get("path") or "").rsplit("/", 1)[-1]
        for row in report.get("memorax_call_records") or []
        if "/v1/memories/add/status/" in str(row.get("path") or "")
    }
    if len(values) != 1:
        raise ValueError("memorax_recovery_task_id_not_unique")
    return next(iter(values))


def _recoverable_write_state(
    source_run: Path, report: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = source_run / "memory/child_add_write_state.json"
    if state_path.is_file():
        state = _load(state_path)
        task_id = str(state.get("task_id") or "")
        if not task_id:
            raise ValueError("memorax_recovery_task_id_missing_from_write_state")
        return state
    task_id = _legacy_task_id(report)
    state = new_write_state(
        intent={
            "request_id": f"legacy-recovery-{task_id}",
            "payload": {},
        },
        stage="child_memory_add",
    )
    return apply_add_response(state, {
        "data": {"status": "accepted", "task_id": task_id},
    })


def preflight(*, source_run: Path, output_root: Path) -> dict[str, Any]:
    report_path = source_run / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.json"
    expanded_path = source_run / "expanded_dialogue.json"
    child_path = source_run / "child_dialogue.json"
    validation_path = source_run / "expanded_dialogue_validation.json"
    checks = {
        "source_run_exists": source_run.is_dir(),
        "output_root_absent": not output_root.exists(),
        "source_report_exists": report_path.is_file(),
        "expanded_dialogue_exists": expanded_path.is_file(),
        "child_dialogue_exists": child_path.is_file(),
        "expanded_validation_exists": validation_path.is_file(),
        "api_calls_zero": True,
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        try:
            report = _load(report_path)
            expanded = _load(expanded_path)
            validation = _load(validation_path)
            write_state = _recoverable_write_state(source_run, report)
            task_id = str(write_state["task_id"])
            pending_stages = {
                str(row.get("stage") or "")
                for row in report.get("pending_memory_writes") or []
            }
            checks.update({
                "source_has_recoverable_child_write": (
                    report.get("failure_stage") == "child_memory_add"
                    or "child_memory_add" in pending_stages
                    or write_state.get("stage") == "child_memory_add"),
                "generated_dialogue_valid": validation.get("valid") is True,
                "expanded_has_child": int(validation.get("child_turn_count") or 0) > 0,
                "qa_fact_source_complete_dialogue": (
                    expanded.get("qa_fact_source") == "complete_parent_plus_child"),
                "task_id_available": bool(task_id),
            })
            details = {
                "source_run_tree_sha256": _hash_tree(source_run),
                "source_report_sha256": sha256_file(report_path),
                "expanded_dialogue_sha256": sha256_file(expanded_path),
                "root_id": expanded.get("root_id"),
                "task_id": task_id,
                "write_state": write_state,
                "request_budget": {
                    "add_status": 1, "bucket_search": 1,
                    "memorax_add": 0, "planner": 0, "realizer": 0,
                    "qa_writer": 0, "solver": 0,
                },
            }
        except Exception as error:
            checks["source_contract_valid"] = False
            details["error"] = f"{type(error).__name__}:{error}"
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": RECOVERY_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "source_run": str(source_run.resolve()),
        "output_root": str(output_root.resolve()),
        "api_calls": 0,
    }


def run(
    *, source_run: Path, output_root: Path,
    memorax_credentials: Path, allow_insecure_http: bool,
) -> dict[str, Any]:
    frozen = preflight(source_run=source_run, output_root=output_root)
    if frozen["decision"] != "pass":
        raise RuntimeError("MEMORAX_RECOVERY_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))
    output_root.mkdir(parents=True, exist_ok=False)
    for relative in ("memory", "qa", "reports"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        **frozen,
        "freeze_version": RECOVERY_VERSION,
        "frozen_at": _now(),
        "fresh": True,
        "no_resume": True,
        "read_only_recovery": True,
    }
    atomic_json(output_root / "run_freeze_manifest.json", freeze)

    expanded = _load(source_run / "expanded_dialogue.json")
    child = _load(source_run / "child_dialogue.json")
    root_id = str(expanded["root_id"])
    task_id = str(frozen["details"]["task_id"])
    client = MemoraxClient(load_memorax_config(
        credentials_path=memorax_credentials,
        allow_insecure_http=allow_insecure_http,
    ))
    write_state = dict(frozen["details"]["write_state"])
    status_response: dict[str, Any] = {}
    final_status = ""
    try:
        status_response = client.add_status(
            task_id, request_id=f"memorax-recovery-status-{task_id}")
        write_state = apply_task_response(write_state, status_response)
        final_status = str(write_state.get("task_status") or "")
        atomic_json(
            output_root / "memory/task_status_response.json", status_response)
    except Exception as error:
        write_state = apply_query_uncertainty(
            write_state, error=f"{type(error).__name__}:{error}")
    atomic_json(output_root / "memory/child_add_write_state.json", write_state)

    query: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    overview: dict[str, Any] | None = None
    prompt: dict[str, Any] | None = None
    if final_status in {"success", "completed", "failed"}:
        query = build_bucket_search_payload(
            root_id=root_id, limit=100, max_buckets_per_type=500)
        try:
            response = client.search_buckets(
                query["payload"], request_id=f"memorax-recovery-search-{root_id}")
            snapshot = normalize_bucket_snapshot(
                response, root_id=root_id, user_id=memorax_user_id(root_id))
        except Exception as error:
            write_state["last_error"] = (
                f"visibility_query:{type(error).__name__}:{error}")
        if final_status == "failed":
            # A root-level Bucket query cannot prove that this exact intent is
            # absent, especially when older memories already exist. Keep the
            # write ambiguous until stronger provider evidence is available.
            write_state = apply_visibility_result(
                write_state, visible=False, absence_confirmed=False)
        elif snapshot is not None:
            # Task completion verifies the write, but the current Bucket API
            # does not echo source_dialogue_sha256/payload_sha256. Therefore a
            # non-empty root Bucket is not exact visibility proof.
            write_state = apply_visibility_result(
                write_state, visible=False, absence_confirmed=False)
        atomic_json(output_root / "memory/child_add_write_state.json", write_state)

    child_turns = [
        row for scene in child.get("scenes") or []
        for row in scene.get("dialogue_turns") or []
    ]
    if final_status in {"success", "completed"} and snapshot is not None:
        overview = compile_memory_overview(
            snapshot,
            recent_dialogue_text="\n".join(
                str(row.get("text") or "") for row in child_turns),
        )
        prompt = build_memory_augmented_mcq_prompt(
            expanded, overview,
            target_questions=8,
            focus_round_index=int(child.get("round_index") or 1),
        )
    if query is not None:
        atomic_json(output_root / "memory/bucket_query.json", query)
    if snapshot is not None:
        atomic_json(
            output_root / "memory/post_generation_bucket_snapshot.json", snapshot)
    if overview is not None and prompt is not None:
        atomic_json(output_root / "memory/post_generation_overview.json", overview)
        atomic_text(output_root / "qa/mcq_prompt.txt", str(prompt["prompt"]))
        atomic_json(output_root / "qa/mcq_prompt_manifest.json", {
            key: value for key, value in prompt.items() if key != "prompt"})

    source_unchanged = _hash_tree(source_run) == frozen["details"][
        "source_run_tree_sha256"]
    recovered = final_status in {"success", "completed"}
    report_status = (
        "completed" if recovered and prompt is not None else
        "ambiguous" if write_state.get("status") == "ambiguous" else
        "pending_recovery")
    status_data = status_response.get("data")
    report = {
        "report_version": RECOVERY_VERSION,
        "status": report_status,
        "source_run": str(source_run),
        "output_root": str(output_root),
        "task_id": task_id,
        "task_final_status": final_status,
        "task_memory_result": status_response.get("memory") or (
            status_data.get("memory") if isinstance(status_data, Mapping) else None),
        "write_state": write_state,
        "bucket_counts": snapshot["bucket_counts"] if snapshot else None,
        "memory_snapshot_sha256": (
            snapshot["snapshot_sha256"] if snapshot else None),
        "memory_overview_sha256": (
            overview["memory_overview_sha256"] if overview else None),
        "complete_dialogue_sha256": expanded["complete_dialogue_sha256"],
        "parent_turn_count": expanded["parent_turn_count"],
        "child_turn_count": expanded["child_turn_count"],
        "turn_count": expanded["turn_count"],
        "mcq_prompt_token_estimate": prompt["token_estimate"] if prompt else None,
        "memorax_calls": client.calls,
        "memorax_add_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "source_run_unchanged": source_unchanged,
        "generation_ready": True,
        "memory_write_verified": bool(
            write_state.get("memory_write_verified")),
        "memory_visibility_verified": bool(
            write_state.get("bucket_visibility_verified")),
        "memory_context_degraded": not bool(
            write_state.get("bucket_visibility_verified")),
        "automatic_retry_allowed": bool(
            write_state.get("automatic_retry_allowed")),
        "qa_prompt_ready": prompt is not None,
        "qa_pipeline_ready": False,
        "completed_at": _now(),
    }
    atomic_json(output_root / "reports/MEMORAX_ASYNC_RECOVERY_REPORT.json", report)
    atomic_text(output_root / "reports/MEMORAX_ASYNC_RECOVERY_REPORT.md", "\n".join([
        "# Memorax Async Child Recovery",
        "",
        f"- Task status: `{final_status}`",
        f"- Parent turns: `{report['parent_turn_count']}`",
        f"- Child turns: `{report['child_turn_count']}`",
        f"- Complete turns: `{report['turn_count']}`",
        f"- Memorax read calls: `{len(client.calls)}`",
        "- Memorax Add calls: `0`",
        f"- Write state: `{write_state.get('status')}`",
        f"- Automatic retry: `{str(report['automatic_retry_allowed']).lower()}`",
        f"- QA prompt ready: `{str(report['qa_prompt_ready']).lower()}`",
        "- QA Writer and Solver calls: `0`",
        "",
    ]))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--memorax-credentials", type=Path,
        default=DEFAULT_MEMORAX_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        report = preflight(
            source_run=args.source_run.resolve(),
            output_root=args.output_root.resolve())
    else:
        report = run(
            source_run=args.source_run.resolve(),
            output_root=args.output_root.resolve(),
            memorax_credentials=args.memorax_credentials.resolve(),
            allow_insecure_http=args.allow_insecure_memorax_http)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (
        report.get("decision") == "pass" if args.preflight_only
        else report.get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
