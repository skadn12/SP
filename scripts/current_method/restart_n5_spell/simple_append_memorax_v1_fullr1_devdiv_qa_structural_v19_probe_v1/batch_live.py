"""Failure-isolated batch runner for Simple Append + Memorax.

The command is intentionally preflight-only unless ``--live`` is supplied.
Each root gets a fresh expansion and QA directory; no root can resume another
root's artifacts or Memory user.  A pending asynchronous Add is reported as
``pending_recovery`` and is never submitted again by this runner.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..simple_append_v1.qa_live import build_local_solver_runtime
from .batch_preflight import (
    DEFAULT_API_CONFIG,
    DEFAULT_BATCH_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_ROOT_IDS,
    DEFAULT_TOKENIZER,
    INPUT_MANIFEST_VERSION,
    load_input_manifest,
    run as run_preflight,
)
from .generation_registry import (
    DEFAULT_GENERATION_REGISTRY,
    GenerationAlreadyRegistered,
    build_generation_identity,
    files_manifest_sha,
    finalize_generation,
    memorax_namespace_sha,
    reserve_generation,
)
from .qa_live import DEFAULT_SOLVER_MODEL, run as run_qa
from .single_root_live import (
    DEFAULT_MEMORAX_CREDENTIALS,
    execute_live,
    run_preflight as run_single_preflight,
)


BATCH_RUNNER_VERSION = "simple_append_memorax_batch_live_v4_spell_frontier_k5"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint(
    *, batch_root: Path, rows: Sequence[Mapping[str, Any]],
    root_count: int, current_root_id: str | None,
) -> None:
    atomic_json(batch_root / "batch_status.json", {
        "runner_version": BATCH_RUNNER_VERSION,
        "status": "running",
        "root_count": int(root_count),
        "processed_root_count": len(rows),
        "completed_root_count": sum(
            row.get("status") == "completed" for row in rows),
        "pending_recovery_count": sum(
            row.get("status") in {"pending_recovery", "qa_pending_recovery"}
            for row in rows),
        "failed_root_count": sum(
            row.get("status") not in {
                "completed", "pending_recovery", "qa_pending_recovery"}
            for row in rows),
        "current_root_id": current_root_id,
        "updated_at": _now(),
    })


def _failure_class(summary: Mapping[str, Any]) -> str:
    stage = str(summary.get("failure_stage") or "")
    if stage == "child_memory_add":
        return "pending_recovery"
    return stage or "unknown"


def classify_batch_status(rows: Sequence[Mapping[str, Any]]) -> str:
    completed = sum(row.get("status") == "completed" for row in rows)
    pending_statuses = {"pending_recovery", "qa_pending_recovery"}
    pending = sum(row.get("status") in pending_statuses for row in rows)
    if rows and completed == len(rows):
        return "completed"
    if completed:
        return "partial"
    if pending:
        return "pending_recovery"
    return "failed"


def batch_exit_code(summary: Mapping[str, Any]) -> int:
    """Return success only when every requested root completed."""
    return 0 if summary.get("status") == "completed" else 1


def _execute_batch(
    *, batch_root: Path, root_ids: Sequence[str], api_config: Path,
    tokenizer_path: Path, solver_model: Path, memorax_credentials: Path,
    allow_insecure_http: bool, input_paths: Mapping[str, Path] | None,
    preflight: Mapping[str, Any], solver_runtime: tuple[Any, Any],
    registry_path: Path, identities: Mapping[str, Mapping[str, Any]],
    registry_reservation: Mapping[str, Any],
) -> dict[str, Any]:
    batch_root.mkdir(parents=True, exist_ok=False)
    (batch_root / "reports").mkdir(parents=True, exist_ok=True)
    freeze = {
        "runner_version": BATCH_RUNNER_VERSION,
        "frozen_at": _now(),
        "fresh": True,
        "no_resume": True,
        "root_ids": list(root_ids),
        "input_sha256": {
            row["root_id"]: row["input_sha256"] for row in preflight["roots"]},
        "api_config_sha256": sha256_file(api_config),
        "preflight": preflight,
        "allow_insecure_memorax_http": allow_insecure_http,
        "input_manifest_version": (
            INPUT_MANIFEST_VERSION if input_paths is not None else None),
        "input_paths": {
            str(root_id): str(path)
            for root_id, path in (input_paths or {}).items()
        },
        "solver_preloaded_before_run_root": True,
        "solver_model": str(solver_model.resolve()),
        "generation_registry": dict(registry_reservation),
        "generation_identities": {
            str(root_id): dict(identity)
            for root_id, identity in identities.items()},
    }
    atomic_json(batch_root / "batch_freeze_manifest.json", freeze)
    atomic_json(batch_root / "batch_status.json", {
        "status": "initialized", "api_calls": 0, "root_count": len(root_ids),
    })

    rows: list[dict[str, Any]] = []
    for root_id in root_ids:
        _checkpoint(
            batch_root=batch_root, rows=rows, root_count=len(root_ids),
            current_root_id=str(root_id))
        expansion_root = batch_root / "roots" / root_id / "expansion"
        qa_root = batch_root / "roots" / root_id / "qa"
        input_path = Path(next(row["input_path"] for row in preflight["roots"]
                               if row["root_id"] == root_id))
        row: dict[str, Any] = {"root_id": root_id,
                               "input_path": str(input_path),
                               "status": "initialized"}
        try:
            single = run_single_preflight(
                input_path=input_path, run_root=expansion_root,
                api_config_path=api_config, tokenizer_path=tokenizer_path,
                memorax_credentials_path=memorax_credentials,
                allow_insecure_memorax_http=allow_insecure_http,
                reuse_existing_parent_memory=False,
                wait_for_memory_completion=False)
            expansion = execute_live(
                preflight=single, input_path=input_path,
                run_root=expansion_root, api_config_path=api_config,
                tokenizer_path=tokenizer_path,
                memorax_credentials_path=memorax_credentials,
                allow_insecure_memorax_http=allow_insecure_http,
                reuse_existing_parent_memory=False,
                wait_for_memory_completion=False)
            row["expansion"] = expansion
            if expansion.get("status") != "completed":
                row["status"] = _failure_class(expansion)
                rows.append(row)
                atomic_json(batch_root / "reports" / f"{root_id}.json", row)
                _checkpoint(
                    batch_root=batch_root, rows=rows,
                    root_count=len(root_ids), current_root_id=None)
                continue
            qa = run_qa(
                source_run=expansion_root, memory_recovery=expansion_root,
                run_root=qa_root, api_config=api_config,
                tokenizer_path=tokenizer_path, solver_model=solver_model,
                solver_runtime=solver_runtime)
            row["qa"] = qa
            if qa.get("status") == "completed":
                row["status"] = "completed"
            elif qa.get("status") == "pending_recovery":
                row["status"] = "qa_pending_recovery"
            else:
                row["status"] = "qa_failed"
        except Exception as error:
            row["status"] = "failed"
            row["error"] = f"{type(error).__name__}:{error}"
        rows.append(row)
        atomic_json(batch_root / "reports" / f"{root_id}.json", row)
        _checkpoint(
            batch_root=batch_root, rows=rows, root_count=len(root_ids),
            current_root_id=None)

    summary = {
        "runner_version": BATCH_RUNNER_VERSION,
        "status": classify_batch_status(rows),
        "batch_root": str(batch_root),
        "roots": rows,
        "root_count": len(rows),
        "completed_root_count": sum(row["status"] == "completed" for row in rows),
        "pending_recovery_count": sum(
            row["status"] in {"pending_recovery", "qa_pending_recovery"}
            for row in rows),
        "failed_root_count": sum(
            row["status"] not in {
                "completed", "pending_recovery", "qa_pending_recovery"}
            for row in rows),
        "memorax_calls": sum(
            int((row.get("expansion") or {}).get("memorax_calls") or 0)
            for row in rows),
        "qa_writer_calls": sum(
            int((row.get("qa") or {}).get("qa_writer_calls") or 0)
            for row in rows),
        "observation_ready_count": sum(
            int((row.get("qa") or {}).get("observation_ready_count") or 0)
            for row in rows),
        "training_ready_count": sum(
            int((row.get("qa") or {}).get("training_ready_count") or 0)
            for row in rows),
        "cache_reuse": 0,
        "imported_response_reuse": 0,
        "completed_at": _now(),
    }
    atomic_json(batch_root / "reports/BATCH_SUMMARY.json", summary)
    atomic_json(batch_root / "batch_status.json", summary)
    finalize_generation(
        registry_path=registry_path, identities=identities, rows=rows)
    return summary


def run(
    *, batch_root: Path = DEFAULT_BATCH_ROOT,
    root_ids: Sequence[str] = DEFAULT_ROOT_IDS,
    api_config: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    solver_model: Path = DEFAULT_SOLVER_MODEL,
    memorax_credentials: Path = DEFAULT_MEMORAX_CREDENTIALS,
    allow_insecure_http: bool = False,
    input_paths: Mapping[str, Path] | None = None,
    generation_registry: Path = DEFAULT_GENERATION_REGISTRY,
) -> dict[str, Any]:
    if batch_root.exists():
        raise FileExistsError(f"batch_run_root_exists:{batch_root}")
    preflight = run_preflight(
        batch_root=batch_root, root_ids=root_ids,
        api_config=api_config, tokenizer_path=tokenizer_path,
        solver_model=solver_model, memorax_credentials=memorax_credentials,
        allow_insecure_http=allow_insecure_http, input_paths=input_paths)
    if preflight["status"] != "passed":
        raise RuntimeError("BATCH_PRE_API_PREFLIGHT_FAILED")

    # Loading the local Solver is a hard pre-API gate.  A GPU/model failure must
    # not create a run root or spend Planner/Realizer tokens first.
    solver_runtime = build_local_solver_runtime(
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    solver_backend = solver_runtime[1]
    try:
        module_root = Path(__file__).resolve().parent
        implementation_paths = [
            *module_root.glob("*.py"),
            *module_root.parent.joinpath("simple_append_v1").glob("*.py"),
            module_root.parent / "vllm_backends.py",
        ]
        implementation_sha = files_manifest_sha(implementation_paths)
        namespace_sha = memorax_namespace_sha(memorax_credentials)
        identities = {
            str(row["root_id"]): build_generation_identity(
                root_id=str(row["root_id"]),
                parent_sha256=str(row["input_sha256"]),
                round_index=1,
                generation_implementation_sha256=implementation_sha,
                api_config_sha256=sha256_file(api_config),
                generation_configuration={
                    "runner_version": BATCH_RUNNER_VERSION,
                    "allow_insecure_memorax_http": bool(allow_insecure_http),
                    "solver_model": str(solver_model.resolve()),
                    "root_generation_mode": "simple_append_one_round",
                },
                memorax_namespace_sha256=namespace_sha,
            )
            for row in preflight["roots"]
        }
        try:
            reservation = reserve_generation(
                registry_path=generation_registry,
                identities=identities,
                run_root=batch_root,
            )
        except GenerationAlreadyRegistered as error:
            raise RuntimeError(
                "BATCH_GENERATION_ALREADY_REGISTERED:" +
                json.dumps(error.conflicts, ensure_ascii=False,
                            sort_keys=True))
        return _execute_batch(
            batch_root=batch_root, root_ids=root_ids,
            api_config=api_config, tokenizer_path=tokenizer_path,
            solver_model=solver_model,
            memorax_credentials=memorax_credentials,
            allow_insecure_http=allow_insecure_http,
            input_paths=input_paths, preflight=preflight,
            solver_runtime=solver_runtime,
            registry_path=generation_registry,
            identities=identities,
            registry_reservation=reservation)
    finally:
        close = getattr(solver_backend, "close", None)
        if callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--root-id", action="append", dest="root_ids",
        help="Authoritative CC root ID; repeat for each batch root (1-11 roots).")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="Actually call the frozen live roles.")
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument(
        "--input-manifest", type=Path,
        help=("Frozen JSON manifest with manifest_version="
              f"{INPUT_MANIFEST_VERSION} and roots mapping."))
    parser.add_argument(
        "--generation-registry", type=Path,
        default=DEFAULT_GENERATION_REGISTRY,
        help="Project-scoped one-generation identity registry.")
    args = parser.parse_args(argv)
    root_ids = tuple(args.root_ids) if args.root_ids else DEFAULT_ROOT_IDS
    input_paths = (load_input_manifest(args.input_manifest.resolve())
                   if args.input_manifest else None)
    if input_paths is not None:
        missing = sorted(set(root_ids) - set(input_paths))
        if missing:
            parser.error("input manifest missing roots: " + ",".join(missing))
    if not args.live:
        report = run_preflight(batch_root=args.batch_root.resolve(),
                               root_ids=root_ids,
                               allow_insecure_http=args.allow_insecure_memorax_http,
                               input_paths=input_paths)
        atomic_json(args.report_root / "MEMORAX_BATCH_PREFLIGHT.json",
                    report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    if args.preflight_only:
        parser.error("--preflight-only and --live are mutually exclusive")
    summary = run(batch_root=args.batch_root.resolve(), root_ids=root_ids,
                  allow_insecure_http=args.allow_insecure_memorax_http,
                  input_paths=input_paths,
                  generation_registry=args.generation_registry.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return batch_exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
