"""Shard-serial Round-2 continuation runner.

The batch layer is deliberately thin: the existing live workflow remains the
single source of truth for planning, realization, promotion, and Memorax
writes. This wrapper freezes each root command, gives every attempt a fresh
directory, and records a recoverable state without retrying an uncertain
Memory write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import canonical_json, sha256_bytes, sha256_file


LIVE_MODULE = (
    "scripts.current_method.restart_n5_spell."
    "simple_append_memorax_v1_memory_event_continuation_probe_v1.live"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_cumulative_continuation_fresh_mining_v1_live_2")
DEFAULT_SMOKE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_cumulative_continuation_fresh_mining_v1_live_1/"
    "smoke_root_cc-train-episode-122711_recovery_2")
ROUND_INDEX = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="utf-8")


def _directory_manifest(path: Path) -> dict[str, Any]:
    files = []
    for child in sorted(path.rglob("*")):
        if not child.is_file() or child.name.endswith(".sha256"):
            continue
        files.append({"path": child.relative_to(path).as_posix(),
                      "size": child.stat().st_size,
                      "sha256": sha256_file(child)})
    body = canonical_json(files).encode("utf-8")
    return {"version": "h4_round1_root_artifact_manifest_v1",
            "root": str(path.resolve()), "files": files,
            "manifest_sha256": hashlib.sha256(body).hexdigest()}


def _root_namespace(prior_user_id: str, root_id: str) -> str:
    suffix = ":" + root_id
    if not prior_user_id.endswith(suffix):
        raise RuntimeError(f"prior_user_id_root_mismatch:{root_id}")
    namespace = prior_user_id[:-len(suffix)]
    if not namespace:
        raise RuntimeError(f"prior_user_namespace_empty:{root_id}")
    return namespace


def _load_plan(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen_names = ("preflight_report.json", "round1_contract.json",
                    "round1_root_manifest.jsonl", "logical_shards.json",
                    "model_identity.json", "fresh_qa_generator_identity.json")
    for name in frozen_names:
        path = run_root / name
        sidecar = path.with_name(path.name + ".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise RuntimeError(f"frozen_manifest_missing:{name}")
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if expected != sha256_file(path):
            raise RuntimeError(f"frozen_manifest_sha_mismatch:{name}")
    report = _load(run_root / "preflight_report.json")
    if report.get("status") != "PASS":
        raise RuntimeError("round1_preflight_not_pass")
    rows = []
    for line in (run_root / "round1_root_manifest.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) != 70 or len({str(x.get("root_id")) for x in rows}) != 70:
        raise RuntimeError("round1_root_manifest_not_closed")
    logical = _load(run_root / "logical_shards.json")
    expected = {str(x) for values in logical.get("shards", {}).values()
                for x in values}
    actual = {str(x.get("root_id")) for x in rows}
    if expected != actual:
        raise RuntimeError("logical_shard_root_set_mismatch")
    for row in rows:
        source = Path(str(row.get("canonical_dialogue_path") or ""))
        if not source.is_file() or sha256_file(source) != str(
                row.get("canonical_dialogue_sha256") or ""):
            raise RuntimeError(f"r01_source_sha_drift:{row.get('root_id')}")
        for key in ("memory_intent_path", "memory_completion_path"):
            path = Path(str(row.get(key) or ""))
            sha_key = key.replace("_path", "_sha256")
            if not path.is_file() or sha256_file(path) != str(
                    row.get(sha_key) or ""):
                raise RuntimeError(f"r01_memory_sha_drift:{row.get('root_id')}:{key}")
    return logical, sorted(rows, key=lambda x: (
        str(x.get("logical_shard")), int(x.get("logical_shard_rank", 0))))


def _row_command(row: Mapping[str, Any], attempt_root: Path) -> list[str]:
    root_id = str(row["root_id"])
    prior_user_id = str(row["prior_user_id"])
    namespace = _root_namespace(prior_user_id, root_id)
    return [sys.executable, "-m", LIVE_MODULE,
            "--input", str(Path(row["canonical_dialogue_path"]).resolve()),
            "--run-root", str(attempt_root.resolve()),
            "--config", "/root/autodl-tmp/gxy-selfplay/config.json",
            "--tokenizer", "/root/autodl-tmp/Qwen3-4B",
            "--memorax-credentials", "/root/.config/memorax/credentials.env",
            "--allow-insecure-memorax-http", "--reuse-existing-namespace",
            "--source-run-root", str(Path(row["memory_physical_root"]).resolve()),
            "--memorax-user-namespace", namespace, "--memory-only-fast",
            "--event-bucket-only", "--round-index", str(ROUND_INDEX), "--live"]


def _attempt_paths(run_root: Path, row: Mapping[str, Any]) -> tuple[Path, Path]:
    shard = str(row["logical_shard"])
    root_id = str(row["root_id"])
    base = run_root / "shards" / shard / "roots" / root_id
    return base / "attempt-01", base / "root_state.json"


def _state(run_root: Path, row: Mapping[str, Any], *, status: str,
           attempt_root: Path, **extra: Any) -> dict[str, Any]:
    return {"version": "h4_round1_root_state_v1", "root_id": row["root_id"],
            "logical_shard": row["logical_shard"], "round_index": ROUND_INDEX,
            "status": status, "attempt_root": str(attempt_root.resolve()),
            "input_path": str(Path(row["canonical_dialogue_path"]).resolve()),
            "input_sha256": row.get("canonical_dialogue_sha256"),
            "prior_user_id": row.get("prior_user_id"),
            "r02_session_id": row.get("r02_session_id"),
            "updated_at": _now(), **extra}


def _verify_smoke_reference(smoke_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    report_path = smoke_root / "reports" / "MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"
    if not report_path.is_file():
        raise RuntimeError(f"smoke_report_missing:{smoke_root}")
    report = _load(report_path)
    if (report.get("status") != "completed" or
            report.get("root_id") != row.get("root_id") or
            report.get("round_2") is not True or
            report.get("memory_write_verified") is not True or
            report.get("memory_visibility_verified") is not True or
            int(report.get("qa_calls") or 0) != 0 or
            int(report.get("solver_calls") or 0) != 0):
        raise RuntimeError(f"smoke_reference_not_complete:{row.get('root_id')}")
    dialogue = smoke_root / "promotion" / "promotable_expanded_dialogue.json"
    intent = smoke_root / "memory" / "child_add_intent.json"
    completion = smoke_root / "memory" / "child_add_completion.json"
    if not dialogue.is_file() or not intent.is_file() or not completion.is_file():
        raise RuntimeError(f"smoke_reference_artifact_missing:{row.get('root_id')}")
    value = _load(intent)
    if (str(value.get("session_id") or "") != row.get("r02_session_id") or
            str(value.get("source_dialogue_sha256") or "") == ""):
        raise RuntimeError(f"smoke_reference_memory_contract:{row.get('root_id')}")
    return {"smoke_root": str(smoke_root.resolve()),
            "report_path": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "dialogue_path": str(dialogue.resolve()),
            "dialogue_sha256": sha256_file(dialogue),
            "intent_path": str(intent.resolve()),
            "intent_sha256": sha256_file(intent),
            "completion_path": str(completion.resolve()),
            "completion_sha256": sha256_file(completion)}


def _load_prior_states(prior_run_root: Path,
                       selected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Import only roots whose prior R02 write was fully verified.

    A running/submitted prior state is never resumed automatically. A
    quarantined state is retryable only when the old report proves that no
    child add was submitted.
    """
    reused: dict[str, dict[str, Any]] = {}
    for row in selected:
        state_path = (prior_run_root / "shards" / str(row["logical_shard"])
                      / "roots" / str(row["root_id"]) / "root_state.json")
        if not state_path.is_file():
            continue
        state = _load(state_path)
        status = str(state.get("status") or "")
        root_id = str(row["root_id"])
        if status in {"COMPLETED", "COMPLETED_REUSED_SMOKE",
                      "COMPLETED_REUSED_PRIOR"}:
            if (state.get("root_id") != root_id or
                    state.get("input_sha256") != row.get(
                        "canonical_dialogue_sha256") or
                    int(state.get("round_index") or 0) != ROUND_INDEX or
                    state.get("memory_write_verified") is not True or
                    state.get("memory_visibility_verified") is not True or
                    int(state.get("qa_calls") or 0) != 0 or
                    int(state.get("solver_calls") or 0) != 0):
                raise RuntimeError(f"prior_completed_state_invalid:{root_id}")
            reused[root_id] = {"state_path": str(state_path.resolve()),
                               "state_sha256": sha256_file(state_path),
                               "state": state}
        elif status in {"RUNNING", "SUBMITTED", "MEMORY_ADD_SUBMITTED"}:
            raise RuntimeError(f"prior_state_uncertain_write:{root_id}:{status}")
        elif status == "QUARANTINED":
            if (int(state.get("child_add_calls") or 0) != 0 or
                    state.get("memory_write_verified") is True):
                raise RuntimeError(f"prior_quarantine_write_uncertain:{root_id}")
    return reused


def run_batch(*, run_root: Path, shard: str | None = None,
              max_roots: int | None = None, plan_only: bool = False,
              smoke_root: Path | None = DEFAULT_SMOKE_ROOT,
              prior_run_root: Path | None = None,
              skip_root_ids: list[str] | None = None) -> dict[str, Any]:
    logical, rows = _load_plan(run_root)
    allowed_shards = sorted(str(x) for x in logical.get("shards", {}))
    if shard is not None and shard not in allowed_shards:
        raise RuntimeError(f"unknown_logical_shard:{shard}")
    requested_skips = {str(root_id) for root_id in (skip_root_ids or [])}
    known_ids = {str(row["root_id"]) for row in rows}
    unknown_skips = sorted(requested_skips - known_ids)
    if unknown_skips:
        raise RuntimeError("unknown_skip_root_id:" + ",".join(unknown_skips))
    selected = [row for row in rows if shard is None or row["logical_shard"] == shard]
    skipped = [row for row in selected if str(row["root_id"]) in requested_skips]
    selected = [row for row in selected if str(row["root_id"]) not in requested_skips]
    if max_roots is not None:
        selected = selected[:max(0, int(max_roots))]
    prior_states = (_load_prior_states(prior_run_root, selected)
                    if prior_run_root is not None else {})
    plan_rows = []
    for row in selected:
        attempt_root, state_path = _attempt_paths(run_root, row)
        plan_rows.append({"root_id": row["root_id"],
                          "logical_shard": row["logical_shard"],
                          "attempt_root": str(attempt_root),
                          "command": _row_command(row, attempt_root)})
        if state_path.exists() and _load(state_path).get("status") in {
                "COMPLETED", "COMPLETED_REUSED_SMOKE"}:
            continue
        if attempt_root.exists():
            raise RuntimeError(f"root_attempt_exists_requires_manual_review:{attempt_root}")
    smoke_by_root = {}
    if smoke_root is not None:
        smoke_row = next((x for x in selected if x.get("root_id") ==
                          "cc-train-episode-122711"), None)
        if smoke_row is not None:
            smoke_by_root[str(smoke_row["root_id"])] = _verify_smoke_reference(
                smoke_root, smoke_row)
    if plan_only:
        return {"status": "PLAN_ONLY", "run_root": str(run_root.resolve()),
                "shards": allowed_shards, "selected_roots": plan_rows,
                "skipped_root_ids": [str(row["root_id"]) for row in skipped],
                "smoke_references": smoke_by_root,
                "prior_references": prior_states,
                "api_calls": 0, "qa_calls": 0, "solver_calls": 0}

    batch_path = run_root / ("batch_manifest_" +
                             (shard or "all") + ".json")
    if batch_path.exists():
        existing = _load(batch_path)
        if existing.get("selected_roots") != plan_rows or existing.get(
                "source_preflight_sha256") != sha256_file(
                    run_root / "preflight_report.json"):
            raise RuntimeError(f"batch_manifest_conflict:{batch_path}")
    manifest_body = {"version": "h4_round1_batch_manifest_v1",
                     "run_root": str(run_root.resolve()),
                     "selected_shard": shard, "selected_roots": plan_rows,
                     "skipped_root_ids": [str(row["root_id"]) for row in skipped],
                     "skipped_root_reasons": {
                         str(row["root_id"]): "explicit_operator_quarantine_skip"
                         for row in skipped
                     },
                     "smoke_references": smoke_by_root,
                     "prior_references": prior_states,
                     "prior_run_root": (str(prior_run_root.resolve())
                                        if prior_run_root else None),
                     "source_preflight_sha256": sha256_file(
                         run_root / "preflight_report.json")}
    _write_json(batch_path, manifest_body)
    results = []
    for row in selected:
        attempt_root, state_path = _attempt_paths(run_root, row)
        if str(row["root_id"]) in smoke_by_root:
            if not state_path.exists():
                _write_json(state_path, _state(
                    run_root, row, status="COMPLETED_REUSED_SMOKE",
                    attempt_root=Path(smoke_by_root[str(row["root_id"])]
                                      ["smoke_root"]),
                    finished_at=_now(), exit_code=0,
                    report_path=smoke_by_root[str(row["root_id"])]
                    ["report_path"], report_status="completed",
                    expansion_calls=0, memorax_calls=0, child_add_calls=0,
                    qa_calls=0, solver_calls=0,
                    memory_write_verified=True,
                    memory_visibility_verified=True,
                    no_automatic_retry=True,
                    reused_successful_smoke=True,
                    smoke_reference=smoke_by_root[str(row["root_id"])]))
            results.append(_load(state_path))
            continue
        if str(row["root_id"]) in prior_states:
            prior = prior_states[str(row["root_id"])]
            if not state_path.exists():
                prior_attempt = Path(prior["state"].get("attempt_root") or
                                     (prior_run_root or run_root))
                _write_json(state_path, _state(
                    run_root, row, status="COMPLETED_REUSED_PRIOR",
                    attempt_root=prior_attempt, finished_at=_now(),
                    exit_code=0,
                    report_path=prior["state"].get("report_path"),
                    report_status="completed", expansion_calls=0,
                    memorax_calls=0, child_add_calls=0, qa_calls=0,
                    solver_calls=0, memory_write_verified=True,
                    memory_visibility_verified=True, no_automatic_retry=True,
                    reused_prior_run=True, prior_reference=prior))
            results.append(_load(state_path))
            continue
        if state_path.exists() and _load(state_path).get("status") in {
                "COMPLETED", "COMPLETED_REUSED_SMOKE",
                "COMPLETED_REUSED_PRIOR"}:
            results.append(_load(state_path))
            continue
        # The live runner owns creation of its run root and fails closed when
        # that path already exists. Create only the parent that holds the
        # state and log records.
        attempt_root.parent.mkdir(parents=True, exist_ok=True)
        log_path = attempt_root.parent / "attempt-01.log"
        _write_json(state_path, _state(run_root, row, status="RUNNING",
                                       attempt_root=attempt_root,
                                       started_at=_now(), log_path=str(log_path)))
        command = _row_command(row, attempt_root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path.cwd()) + (os.pathsep + env["PYTHONPATH"]
                                                if env.get("PYTHONPATH") else "")
        with log_path.open("w", encoding="utf-8") as log:
            log.write(json.dumps({"command": command, "started_at": _now()},
                                 ensure_ascii=False) + "\n")
            completed = subprocess.run(command, cwd=str(Path.cwd()), env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       check=False)
        report_path = attempt_root / "reports" / "MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"
        report = _load(report_path) if report_path.is_file() else {}
        status = "COMPLETED" if completed.returncode == 0 and report.get("status") == "completed" and report.get("memory_write_verified") is True and report.get("memory_visibility_verified") is True else "QUARANTINED"
        artifact_path = attempt_root / "artifact_manifest.json"
        _write_json(artifact_path, _directory_manifest(attempt_root))
        state = _state(run_root, row, status=status, attempt_root=attempt_root,
                       started_at=_load(state_path).get("started_at"),
                       finished_at=_now(), exit_code=completed.returncode,
                       report_path=str(report_path),
                       report_status=report.get("status"),
                       expansion_calls=report.get("expansion_calls", 0),
                       memorax_calls=report.get("memorax_calls", 0),
                       child_add_calls=report.get("child_add_calls", 0),
                       qa_calls=report.get("qa_calls", 0),
                       solver_calls=report.get("solver_calls", 0),
                       memory_write_verified=report.get("memory_write_verified", False),
                       memory_visibility_verified=report.get("memory_visibility_verified", False),
                       no_automatic_retry=True)
        _write_json(state_path, state)
        results.append(state)
        if status != "COMPLETED":
            # A failed root is isolated. Never continue to another root while
            # an uncertain child write could be hidden in the log.
            break
    quarantined_count = sum(x.get("status") == "QUARANTINED"
                            for x in results)
    summary = {"version": "h4_round1_batch_summary_v1", "completed_at": _now(),
               "run_root": str(run_root.resolve()), "selected_shard": shard,
               "prior_run_root": (str(prior_run_root.resolve())
                                  if prior_run_root else None),
               "selected_count": len(selected),
               "skipped_root_ids": [str(row["root_id"]) for row in skipped],
               "skipped_root_reasons": {
                   str(row["root_id"]): "explicit_operator_quarantine_skip"
                   for row in skipped
               },
               "results": results,
               "completed_count": sum(x.get("status") in
                                       {"COMPLETED", "COMPLETED_REUSED_SMOKE",
                                        "COMPLETED_REUSED_PRIOR"}
                                       for x in results),
               "quarantined_count": quarantined_count,
               "status": "QUARANTINED" if quarantined_count else "COMPLETED",
               "api_calls": sum(int(x.get("expansion_calls") or 0) for x in results),
               "qa_calls": sum(int(x.get("qa_calls") or 0) for x in results),
               "solver_calls": sum(int(x.get("solver_calls") or 0) for x in results)}
    _write_json(run_root / "batch_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round-2 shard-serial continuation runner")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--shard")
    parser.add_argument("--max-roots", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-smoke-reference", action="store_true")
    parser.add_argument("--prior-run-root", type=Path,
                        help="Import verified completed roots from an older run")
    parser.add_argument("--skip-root-id", action="append", default=[],
                        help="Explicitly skip a quarantined root; recorded in the immutable manifest")
    args = parser.parse_args(argv)
    result = run_batch(run_root=args.run_root, shard=args.shard,
                       max_roots=args.max_roots, plan_only=args.plan_only,
                       smoke_root=(None if args.no_smoke_reference
                                   else DEFAULT_SMOKE_ROOT),
                       prior_run_root=args.prior_run_root,
                       skip_root_ids=args.skip_root_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PLAN_ONLY" or result.get("quarantined_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
