"""Run the frozen Round-0 Train-only post-training Solver reevaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..h4_round0_b0_k3_difficulty_v1.io_utils import RunLock, load_json
from . import DEFAULT_RUN_ROOT
from .pipeline import finalize_reports, run_preflight, run_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "clean", "inclusive", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    phases = (
        ("preflight", "clean", "inclusive", "finalize")
        if args.phase == "all" else (args.phase,)
    )
    final: dict = {}
    if "preflight" in phases:
        final = run_preflight(run_root)
        print(json.dumps({
            "phase": "preflight",
            "status": final["status"],
            "planned_request_count_per_trained_model": final["planned_request_count_per_trained_model"],
        }, indent=2), flush=True)
    if any(phase in phases for phase in ("clean", "inclusive", "finalize")):
        preflight_path = run_root / "preflight_report.json"
        if not preflight_path.is_file() or load_json(preflight_path).get("status") != "PASS":
            raise RuntimeError("preflight_must_pass_before_solver_or_finalize")
    for phase, model_name in (
        ("clean", "M_clean_policy"),
        ("inclusive", "M_inclusive_policy"),
    ):
        if phase in phases:
            with RunLock(run_root):
                final = run_trained_model(run_root, model_name)
            print(json.dumps({
                "phase": phase,
                "status": final["status"],
                "candidate_count": final["candidate_count"],
                "request_count": final["request_count"],
                "technical_failure_count": final["technical_failure_count"],
            }, indent=2), flush=True)
    if "finalize" in phases:
        with RunLock(run_root):
            final = finalize_reports(run_root)
        print(json.dumps({
            "phase": "finalize",
            "status": final["status"],
            "next_status": final["next_status"],
            "primary_transition_counts": final["primary_transition_counts"],
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
