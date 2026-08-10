"""Identity-frozen entry point for the fifth v11 continuous-expansion canary."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_canary_v11 as runtime
import scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 as preflight_runtime

from .provenance import hash_tree


RUN_ID = "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_5"
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_4 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_4"
)
PROBE_11 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_v11_planner_probe_11"
)
EXPECTED_CANARY_4_SHA = (
    "a07b0f9a09968141f0a72250fe2fb516227783fa495c78440ef6fbc6f9649c70"
)
EXPECTED_PROBE_11_SHA = (
    "81e34471fc5fe1dd2435cce1cd05869538d6911c0388fd230262eace5885c016"
)

_base_historical_manifests = preflight_runtime.historical_manifests_v11


def historical_manifests_canary_5() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests.update({
        "live_canary_4": hash_tree(CANARY_4),
        "planner_probe_11": hash_tree(PROBE_11),
    })
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_5
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_5


def _history_valid(preflight: dict[str, Any]) -> bool:
    manifests = preflight.get("historical_manifests") or {}
    return (
        (manifests.get("live_canary_4") or {}).get("tree_sha256")
        == EXPECTED_CANARY_4_SHA
        and (manifests.get("planner_probe_11") or {}).get("tree_sha256")
        == EXPECTED_PROBE_11_SHA
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use fifth v11 live continuous-expansion canary"
    )
    parser.add_argument("--input", type=Path, default=runtime.DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=runtime.DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=runtime.DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight = preflight_runtime.run_live_preflight_b2_v11(
        runner_path=Path(__file__).resolve(), input_path=args.input,
        run_root=args.run_root, api_config_path=args.api_config,
        tokenizer_path=args.tokenizer)
    if not _history_valid(preflight):
        preflight["valid"] = False
        preflight["errors"] = [*preflight.get("errors", []),
                               "canary_5_history_baseline_mismatch"]
        preflight["checks"]["canary_5_history_baseline"] = False
    else:
        preflight["checks"]["canary_5_history_baseline"] = True
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["valid"] else 2
    if not preflight["valid"]:
        print(json.dumps({"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    random.seed(runtime.ORCHESTRATION_SEED)
    summary = runtime.execute_live_canary(
        preflight=preflight, input_path=args.input.resolve(),
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
