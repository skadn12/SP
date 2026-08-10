"""Run the CC plus LoCoMo dynamic Planner preflight without API access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .dynamic_generation_preflight import run_generation_preflight_batch


DEFAULT_CC_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots")
DEFAULT_LOCOMO_SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "cross_dialogue_outline_graph_expansion_10x3_v4/dialogues/conv-43/"
    "frozen/parent_dialogue.json")
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_dynamic_planner_preflight_v1")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc-roots", type=Path, default=DEFAULT_CC_ROOT)
    parser.add_argument("--locomo-parent-source", type=Path,
                        default=DEFAULT_LOCOMO_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args(argv)
    roots = sorted(args.cc_roots.glob("*/unified_root.json"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    report = run_generation_preflight_batch(
        cc_root_paths=roots,
        locomo_parent_source=args.locomo_parent_source,
        output_root=args.output_root,
        token_counter=lambda value: len(tokenizer.encode(
            value, add_special_tokens=False)))
    print(json.dumps({key: report[key] for key in (
        "generation_preflight_ready", "generation_ready", "root_count",
        "api_calls")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
