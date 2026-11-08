"""Safe entry point for the isolated 30-root H4 batch probe.

Only the ``preflight`` phase is implemented here.  It validates the cleaned
batch and writes a frozen execution plan; it deliberately does not import or
invoke the live expansion, Challenger, Memorax, QA, or Solver runtimes.  Live
commands are emitted in the plan and must be started explicitly after review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plan import (
    DEFAULT_CLEANED_BATCH,
    DEFAULT_CONFIG,
    DEFAULT_CREDENTIALS,
    DEFAULT_RUN_ROOT,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    DEFAULT_EXPECTED_COUNT,
    DEFAULT_SHARD_SIZE,
    build_batch_plan,
    write_batch_plan,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "plan"), default="preflight",
        help="Both names perform the same zero-API validation and plan write.",
    )
    parser.add_argument("--cleaned-batch", type=Path,
                        default=DEFAULT_CLEANED_BATCH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--expected-count", type=int,
                        default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path,
                        default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--credentials", type=Path,
                        default=DEFAULT_CREDENTIALS)
    parser.add_argument("--namespace-prefix", default=None)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate and print the plan without creating a run directory.",
    )
    args = parser.parse_args(argv)
    args._argv = list(argv or [])
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kwargs = {
        "cleaned_batch": args.cleaned_batch,
        "run_root": args.run_root,
        "expected_count": args.expected_count,
        "shard_size": args.shard_size,
        "config": args.config,
        "tokenizer": args.tokenizer,
        "solver_model": args.solver_model,
        "credentials": args.credentials,
    }
    if args.namespace_prefix is not None:
        kwargs["namespace_prefix"] = args.namespace_prefix
    plan = build_batch_plan(**kwargs)
    if args.check_only:
        result = {
            "status": "preflight_passed_check_only",
            "run_root": str(args.run_root.resolve()),
            "expected_count": plan["expected_count"],
            "shard_count": plan["shard_count"],
            "api_calls": 0,
            "solver_calls": 0,
            "mainline_untouched": plan["mainline_untouched"],
        }
    else:
        output = write_batch_plan(plan)
        result = {
            "status": "offline_preflight_passed",
            "run_root": str(output),
            "expected_count": plan["expected_count"],
            "shard_count": plan["shard_count"],
            "shard_size": plan["shard_size"],
            "api_calls": 0,
            "solver_calls": 0,
            "mainline_untouched": plan["mainline_untouched"],
            "historical_runs_untouched": plan["historical_runs_untouched"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
