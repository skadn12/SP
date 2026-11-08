"""Staged CLI for full-pool B0 K=3 difficulty screening.

Only ``smoke`` and ``batch`` can load the Solver, and both require ``--live``.
``preflight`` and ``finalize`` are local file/tokenizer phases.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from ..core import sha256_file
from . import (
    BASE_SOLVER_MODEL,
    BASE_TOKENIZER,
    DEFAULT_RUN_ROOT,
    DEFAULT_SOURCE_RUN,
    PROBE_VERSION,
    RUN_BASE,
)
from .difficulty import finalize
from .gate import build_solver_contract, build_solver_plan, run_program_gate
from .inventory import (
    build_exposure_inventory,
    build_source_inventory,
    discover_exposure_artifacts,
    freeze_root_split,
)
from .io_utils import (
    RunLock,
    load_json,
    prepare_run_root,
    verify_sha_sidecar,
    write_once_json,
    write_once_jsonl,
)
from .request_ledger import Round0RequestLedger
from .solver import create_backend, execute_scope, validate_solver_runtime


def _tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _assert_run_identity(run_root: Path, source_run: Path) -> None:
    identity = load_json(run_root / "run_identity.json")
    if identity.get("probe_version") != PROBE_VERSION:
        raise RuntimeError("run_identity_probe_version_mismatch")
    if identity.get("source_run") != str(source_run.resolve()):
        raise RuntimeError("run_identity_source_run_mismatch")


def preflight_phase(args: argparse.Namespace) -> dict[str, Any]:
    inventory = build_source_inventory(args.source_run)
    prepare_run_root(args.run_root, args.source_run)
    write_once_json(args.run_root / "source_inventory.json", inventory)
    if inventory["blockers"]:
        blocker = {
            "preflight_version": "h4_round0_preflight_v1",
            "status": "BLOCKED_SOURCE_INVENTORY",
            "blockers": inventory["blockers"],
            "solver_calls": 0,
        }
        write_once_json(args.run_root / "preflight_report.json", blocker)
        raise RuntimeError(f"source_inventory_blocked:{inventory['blockers']}")
    artifacts = discover_exposure_artifacts(
        RUN_BASE, [Path(value) for value in args.exposure_artifact],
    )
    exposure = build_exposure_inventory(artifacts)
    eligible_root_ids = {
        str(row["root_id"]) for row in inventory["roots"]
        if int(row["candidate_count"]) > 0
    }
    exposed_source_ids = sorted(
        eligible_root_ids & set(exposure["exposed_root_ids"])
    )
    exposure = {
        **exposure,
        "source_candidate_bearing_root_count": len(eligible_root_ids),
        "exposed_source_root_count": len(exposed_source_ids),
        "exposed_source_root_ids": exposed_source_ids,
        "unexposed_source_root_count": len(eligible_root_ids) - len(exposed_source_ids),
        "unexposed_source_root_ids": sorted(eligible_root_ids - set(exposed_source_ids)),
    }
    write_once_json(args.run_root / "analysis_exposed_roots.json", exposure)
    split_rows, split_summary = freeze_root_split(inventory, exposure)
    write_once_jsonl(args.run_root / "root_split_manifest.jsonl", split_rows)
    write_once_json(args.run_root / "root_split_summary.json", split_summary)
    tokenizer = _tokenizer(args.tokenizer)
    contract = build_solver_contract(
        tokenizer, model_path=args.solver_model, tokenizer_path=args.tokenizer,
    )
    write_once_json(args.run_root / "solver_contract.json", contract)
    gate_rows, gate_report = run_program_gate(inventory, tokenizer)
    write_once_jsonl(args.run_root / "program_gate_results.jsonl", gate_rows)
    plan = build_solver_plan(gate_rows, split_rows)
    write_once_json(args.run_root / "solver_plan.json", plan)
    report = {
        "preflight_version": "h4_round0_preflight_v1",
        "status": "PASS",
        "source_root_count": inventory["source_root_count"],
        "candidate_bearing_root_count": inventory["candidate_bearing_root_count"],
        "candidate_count": inventory["candidate_count"],
        "incomplete_root_id": inventory["incomplete_root_id"],
        "incomplete_root_status": "QA_INCOMPLETE_EXCLUDED",
        **gate_report,
        "root_split_counts": split_summary["root_counts"],
        "analysis_exposed_root_count": split_summary["analysis_exposed_root_count"],
        "analysis_exposure_union_root_count": exposure["exposed_root_count"],
        "planned_solver_request_count": plan["planned_request_count"],
        "planned_unique_prompt_count": plan["planned_unique_prompt_count"],
        "smoke_candidate_count": plan["smoke_candidate_count"],
        "smoke_request_count": plan["smoke_request_count"],
        "solver_contract_sha256": sha256_file(args.run_root / "solver_contract.json"),
        "solver_calls": 0,
    }
    write_once_json(args.run_root / "preflight_report.json", report)
    return report


def assert_gpu_idle() -> None:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"gpu_inventory_failed:{completed.stderr.strip()}")
    active = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError(f"gpu_conflict_detected:{active}")


def solver_phase(args: argparse.Namespace) -> dict[str, Any]:
    if not args.live:
        raise RuntimeError(f"--live_required_for_{args.phase}")
    if not 0 < float(args.gpu_memory_utilization) <= 1:
        raise ValueError("gpu_memory_utilization_must_be_in_0_1")
    _assert_run_identity(args.run_root, args.source_run)
    preflight_path = args.run_root / "preflight_report.json"
    verify_sha_sidecar(preflight_path)
    preflight = load_json(preflight_path)
    if preflight.get("status") != "PASS":
        raise RuntimeError("passing_preflight_required")
    tokenizer = _tokenizer(args.tokenizer)
    with RunLock(args.run_root):
        validated_runtime = validate_solver_runtime(args.run_root, tokenizer)
        assert_gpu_idle()
        ledger = Round0RequestLedger(
            args.run_root, solver_contract_sha256=validated_runtime[0],
        )
        ledger.recover_orphaned_requests()
        backend = create_backend(
            model_path=args.solver_model,
            tokenizer_path=args.tokenizer,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        try:
            return execute_scope(
                run_root=args.run_root,
                scope=args.phase,
                tokenizer=tokenizer,
                backend=backend,
                validated_runtime=validated_runtime,
            )
        finally:
            backend.close()


def finalize_phase(args: argparse.Namespace) -> dict[str, Any]:
    _assert_run_identity(args.run_root, args.source_run)
    return finalize(args.run_root)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--phase", required=True,
        choices=("preflight", "smoke", "batch", "finalize"),
    )
    value.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument("--solver-model", type=Path, default=BASE_SOLVER_MODEL)
    value.add_argument("--tokenizer", type=Path, default=BASE_TOKENIZER)
    value.add_argument(
        "--exposure-artifact", action="append", default=[],
        help="Additional analysis/manual-review artifact included in exposure union.",
    )
    value.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    value.add_argument("--live", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.phase == "preflight":
        report = preflight_phase(args)
    elif args.phase in {"smoke", "batch"}:
        report = solver_phase(args)
    else:
        report = finalize_phase(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
