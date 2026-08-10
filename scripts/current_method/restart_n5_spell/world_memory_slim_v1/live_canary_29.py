"""Single-use continuous-expansion live canary 29."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text
from . import live_canary_28 as previous
from . import round_validator_contract_repair_v11 as contract_repair
from .provenance import hash_tree
from .round_expansion_validator_v1 import ROUND_VALIDATOR_INPUT_VERSION
from .round_validator_contract import (
    ROUND_VALIDATOR_MAX_ATTEMPTS,
    ROUND_VALIDATOR_PROMPT_VERSION,
)


runtime = previous.runtime
preflight_runtime = previous.preflight_runtime
RUN_ID = (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_29"
)
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_28 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_28"
)
EXPECTED_CANARY_28_SHA = (
    "8f853b89b4ea46b803a23aaf8b9e69e2a3b99bf15ce3c06d7b8a7c3fc566647f"
)
_base_historical_manifests = previous.historical_manifests_canary_28


def historical_manifests_canary_29() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests["live_canary_28"] = hash_tree(CANARY_28)
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_29
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_29


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    preflight = previous._augment_preflight(preflight, input_path)
    stale_keys = {
        "round_validator_candidate_scope_prompt_frozen",
        "round_validator_nonempty_repair_frozen",
    }
    canary27_companions = (
        "canary26_tree_sha",
        "validator_request_budget_unchanged_canary27",
    )
    if (all(preflight.get("checks", {}).get(key) is False for key in stale_keys) and
            all(preflight["checks"].get(key) is True
                for key in canary27_companions)):
        for key in stale_keys:
            preflight["checks"].pop(key)
        preflight["errors"] = [
            error for error in preflight.get("errors", [])
            if error != "canary27_validator_scope_preflight_failed"
        ]
        preflight["valid"] = bool(
            all(preflight.get("checks", {}).values()) and
            not preflight.get("errors"))
    checks = {
        "canary28_tree_sha": CANARY_28.is_dir() and
            hash_tree(CANARY_28)["tree_sha256"] == EXPECTED_CANARY_28_SHA,
        "round_validator_input_v5_frozen":
            ROUND_VALIDATOR_INPUT_VERSION.endswith("v5_validator_checks_only"),
        "round_validator_nonempty_prompt_frozen":
            ROUND_VALIDATOR_PROMPT_VERSION.endswith("v11_nonempty_evidence"),
        "round_validator_marker_repair_frozen":
            contract_repair.ROUND_VALIDATOR_CONTRACT_REPAIR_VERSION_V11 ==
            "round_validator_contract_only_repair_v11_v14_markers",
        "validator_request_budget_unchanged_canary29":
            ROUND_VALIDATOR_MAX_ATTEMPTS == 2,
    }
    for name, value in checks.items():
        preflight["checks"][name] = value
    if not all(checks.values()):
        preflight["errors"] = [*preflight.get("errors", []),
                               "canary29_validator_template_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and all(checks.values()))
    return preflight


def _markdown(summary: dict[str, Any]) -> str:
    roles = ((summary.get("api_usage") or {}).get("roles") or {})
    return f"""# v11 Continuous Expansion Live Canary 29

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

Required evidence selections are model-owned and locally whitelist-validated;
the templates no longer suggest empty required arrays. No QA, Round 2,
Evidence Committer, Questioner, Distractor, Solver, Strategy, or Training runs.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 continuous-expansion live canary 29")
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
    atomic_json(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_29.json",
                summary)
    atomic_text(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_29.md",
                _markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
