"""Run the episode-61555 frozen-Graph S2--S10 sequential canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .realizer_s1_canary import DEFAULT_SOURCE_RUN
from .realizer_s2_s10_canary import DEFAULT_S1_RUN, run


DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_realizer_s2_s10_"
    "sequential_canary_v3")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--s1-run", type=Path, default=DEFAULT_S1_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--tests-file", type=Path)
    parser.add_argument("--scene-end", type=int, default=10)
    parser.add_argument("--accepted-prefix-run", type=Path)
    args = parser.parse_args(argv)
    tests = (json.loads(args.tests_file.read_text(encoding="utf-8"))
             if args.tests_file else {})
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    result = run(
        source_run=args.source_run, s1_run=args.s1_run,
        run_root=args.run_root, api_config_path=args.api_config,
        tokenizer_path=args.tokenizer, tokenizer=tokenizer,
        prior_test_results=tests, scene_end=args.scene_end,
        accepted_prefix_run=args.accepted_prefix_run)
    print(json.dumps({key: result[key] for key in (
        "decision", "generation_ready", "planner_api_calls",
        "graph_api_calls", "realizer_api_calls", "completed_scene_count",
        "realizer_s2_ready", "realizer_scene_scope_ready",
        "failure_diagnosis")},
        ensure_ascii=False, indent=2))
    return 0 if (result["realizer_scene_scope_ready"] if args.scene_end < 10
                 else result["generation_ready"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
