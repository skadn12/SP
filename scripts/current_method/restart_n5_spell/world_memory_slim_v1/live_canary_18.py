"""Single-use continuous-expansion live canary 18."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text
from . import live_canary_17 as previous
from .provenance import hash_tree
from .round_validator_contract import (
    ROUND_VALIDATOR_PROMPT_VERSION, ROUND_VALIDATOR_RETRY_PROMPT_VERSION,
)
from .round_validator_contract_repair_v11 import (
    ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11,
)


runtime = previous.runtime
preflight_runtime = previous.preflight_runtime
RUN_ID = (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_18"
)
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_17 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_17"
)
EXPECTED_CANARY_17_SHA = (
    "a6285ea9c3244162a3a1ce8724fbe35b23e59d3c2d1b6abe6fe37179a37bc02f"
)
_base_historical_manifests = previous.historical_manifests_canary_17


def historical_manifests_canary_18() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests["live_canary_17"] = hash_tree(CANARY_17)
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_18
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_18


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    preflight = previous._augment_preflight(preflight, input_path)
    checks = {
        "canary17_tree_sha": CANARY_17.is_dir() and
            hash_tree(CANARY_17)["tree_sha256"] == EXPECTED_CANARY_17_SHA,
        "round_validator_field_placement_prompt":
            ROUND_VALIDATOR_PROMPT_VERSION.endswith("v7_field_placement"),
        "round_validator_field_placement_retry":
            ROUND_VALIDATOR_RETRY_PROMPT_VERSION.endswith("v7_field_placement"),
        "round_validator_field_rehome_repair":
            ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11.endswith(
                "v8_field_rehome"),
    }
    for name, value in checks.items():
        preflight["checks"][name] = value
    if not all(checks.values()):
        preflight["errors"] = [*preflight.get("errors", []),
                               "canary18_validator_repair_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and all(checks.values()))
    return preflight


def _markdown(summary: dict[str, Any]) -> str:
    usage = summary.get("api_usage") or {}
    roles = usage.get("roles") or {}
    return f"""# v11 Continuous Expansion Live Canary 18

- status: `{summary.get('status')}`
- conclusion: **{summary.get('conclusion')}**
- planned / realized scenes: {summary.get('planned_scene_count')} / {summary.get('realized_scene_count')}
- realized / promotable turns: {summary.get('realized_turn_count')} / {summary.get('promotable_turn_count')}
- Planner requests: {((roles.get('planner') or {}).get('physical_requests') or 0)}
- Realizer requests: {((roles.get('realizer') or {}).get('physical_requests') or 0)}
- Round Validator requests: {((roles.get('round_validator') or {}).get('physical_requests') or 0)}
- generation_ready: `{str(summary.get('generation_ready')).lower()}`
- fresh_e2e_ready: `{str(summary.get('fresh_e2e_ready')).lower()}`
- qa_pipeline_ready: `{str(summary.get('qa_pipeline_ready')).lower()}`

This is one fresh generation canary. It does not execute QA, Round 2, Evidence
Committer, Questioner, Distractor, Solver, Strategy, Training, or batch work.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 continuous-expansion live canary 18")
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
    preflight = _augment_preflight(preflight, args.input.resolve())
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
        tokenizer_path=args.tokenizer.resolve(),
        growth_profile=preflight["growth_profile"],
        dialogue_first_participants=True)
    atomic_json(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_18.json",
                summary)
    atomic_text(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_18.md",
                _markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
