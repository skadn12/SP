"""Single-use continuous-expansion live canary 34."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text
from . import live_canary_33 as previous
from .live_contracts_v11 import (
    PLANNER_PROMPT_VERSION_V11, PLANNER_RETRY_PROMPT_VERSION_V11,
)
from .planner_action_repair_v11 import (
    PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11,
    PLANNER_ACTION_REPAIR_VERSION_V11,
)
from .provenance import hash_tree


runtime = previous.runtime
preflight_runtime = previous.preflight_runtime
RUN_ID = (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_34"
)
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_33 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_33"
)
EXPECTED_CANARY_33_SHA = (
    "4140779f9b0bc66c5c9ec4cdb2762d31ecda591e14e278586f947848634132fd"
)
_base_historical_manifests = previous.historical_manifests_canary_33


def historical_manifests_canary_34() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests["live_canary_33"] = hash_tree(CANARY_33)
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_34
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_34


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    preflight = previous._augment_preflight(preflight, input_path)
    stale_key = "planner_action_repair_field_scoped_aliases_frozen"
    companions = {
        "canary32_tree_sha",
        "planner_outline_authoritative_connection_refs_frozen",
        "action_availability_authoritative_connection_refs_frozen",
    }
    if (preflight.get("checks", {}).get(stale_key) is False and
            all(preflight.get("checks", {}).get(key) is True for key in companions)):
        preflight["checks"].pop(stale_key)
        preflight["errors"] = [
            error for error in preflight.get("errors", [])
            if error != "canary33_planner_reference_preflight_failed"
        ]
        preflight["valid"] = bool(
            all(preflight.get("checks", {}).values()) and
            not preflight.get("errors"))
    checks = {
        "canary33_tree_sha": CANARY_33.is_dir() and
            hash_tree(CANARY_33)["tree_sha256"] == EXPECTED_CANARY_33_SHA,
        "planner_exact_authoritative_connection_prompt_frozen":
            PLANNER_PROMPT_VERSION_V11.endswith(
                "source_novelty_exact_authoritative_connections") and
            PLANNER_RETRY_PROMPT_VERSION_V11.endswith(
                "v21_aggregate_growth_contract"),
        "planner_action_repair_exact_connection_options_frozen":
            PLANNER_ACTION_REPAIR_VERSION_V11.endswith(
                "v8_exact_connection_options") and
            PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11.endswith(
                "v9_exact_connection_options"),
    }
    preflight["checks"].update(checks)
    if not all(checks.values()):
        preflight["errors"] = [*preflight.get("errors", []),
                               "canary34_planner_repair_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and all(checks.values()))
    return preflight


def _markdown(summary: dict[str, Any]) -> str:
    roles = ((summary.get("api_usage") or {}).get("roles") or {})
    return f"""# v11 Continuous Expansion Live Canary 34

- status: `{summary.get('status')}`
- conclusion: **{summary.get('conclusion')}**
- planned / realized scenes: {summary.get('planned_scene_count')} / {summary.get('realized_scene_count')}
- realized / promotable turns: {summary.get('realized_turn_count')} / {summary.get('promotable_turn_count')}
- Planner requests: {((roles.get('planner') or {}).get('physical_requests') or 0)}
- Realizer requests: {((roles.get('realizer') or {}).get('physical_requests') or 0)}
- Round Validator requests: {((roles.get('round_validator') or {}).get('physical_requests') or 0)}
- authoritative person/event growth: `{str(summary.get('growth_profile_ready')).lower()}`
- generation_ready: `{str(summary.get('generation_ready')).lower()}`
- fresh_e2e_ready: `{str(summary.get('fresh_e2e_ready')).lower()}`
- qa_pipeline_ready: `{str(summary.get('qa_pipeline_ready')).lower()}`

Planner coverage and schema errors are aggregated in the same attempt, and action
repair receives exact authoritative connection pairs. No QA, Round 2, Evidence
Committer, Questioner, Distractor, Solver, Strategy, Training, or batch work is run.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 continuous-expansion live canary 34")
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
    atomic_json(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_34.json",
                summary)
    atomic_text(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_34.md",
                _markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
