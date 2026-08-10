"""Run the frozen-Planner/Graph episode-61555 S1-only Realizer canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .realizer_s1_canary import DEFAULT_SOURCE_RUN, run


DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_realizer_s1_only_canary_v1")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--tests-file", type=Path)
    args = parser.parse_args(argv)
    tests = (json.loads(args.tests_file.read_text(encoding="utf-8"))
             if args.tests_file else {})
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    result = run(
        source_run=args.source_run, run_root=args.run_root,
        api_config_path=args.api_config, tokenizer_path=args.tokenizer,
        tokenizer=tokenizer, prior_test_results=tests)
    print(json.dumps({key: result[key] for key in (
        "decision", "realizer_s1_ready", "generation_ready",
        "planner_api_calls", "graph_api_calls", "realizer_api_calls",
        "accepted_attempt", "failure_diagnosis")},
        ensure_ascii=False, indent=2))
    return 0 if result["realizer_s1_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
