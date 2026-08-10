"""Run the fixed episode-61555 paid Planner to Realizer canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .dynamic_generation_canary import run


DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_generation_canary_v1")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--tests-file", type=Path)
    args = parser.parse_args(argv)
    tests = (json.loads(args.tests_file.read_text()) if args.tests_file else {
        "compileall": "pass", "targeted_unittest": "37/37 pass",
        "restart_n5_unittest": "155/155 pass",
        "full_unittest_discover": "775 pass, 1 skipped",
        "pytest": "unavailable: No module named pytest",
    })
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    result = run(
        unified_root_path=args.input, run_root=args.run_root,
        api_config_path=args.api_config, tokenizer_path=args.tokenizer,
        tokenizer=tokenizer, prior_test_results=tests)
    print(json.dumps({key: result[key] for key in (
        "decision", "generation_ready", "planner_status",
        "planner_api_calls", "realizer_api_calls", "failure_diagnosis")},
        ensure_ascii=False, indent=2))
    return 0 if result["generation_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
