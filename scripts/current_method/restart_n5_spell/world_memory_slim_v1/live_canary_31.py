"""Single-use continuous-expansion live canary 31."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text
from . import live_canary_30 as previous
from . import round_expansion_validator_v1 as validator_runtime
from . import round_validator_contract_repair_v11 as contract_repair
from .provenance import hash_tree
from .round_validator_contract import (
    ROUND_VALIDATOR_ABSOLUTE_MAX_ATTEMPTS,
    ROUND_VALIDATOR_MAX_ATTEMPTS,
    ROUND_VALIDATOR_PROMPT_VERSION,
    ROUND_VALIDATOR_TERMINAL_CONTRACT_REPAIR_ATTEMPTS,
)


runtime = previous.runtime
preflight_runtime = previous.preflight_runtime
RUN_ID = (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_31"
)
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_30 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_30"
)
EXPECTED_CANARY_30_SHA = (
    "1d70b7bddf6dd04fe94128f5857a742ff5ca5d656225328d348cb72336e580ef"
)
_base_historical_manifests = previous.historical_manifests_canary_30


def historical_manifests_canary_31() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests["live_canary_30"] = hash_tree(CANARY_30)
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_31
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_31


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    preflight = previous._augment_preflight(preflight, input_path)
    stale_false_keys = {
        "round_validator_input_v5_frozen",
        "round_validator_nonempty_prompt_frozen",
        "round_validator_marker_repair_frozen",
        "round_validator_slot_template_prompt_frozen",
    }
    companions = {
        "canary28_tree_sha", "canary29_tree_sha",
        "round_validator_scoped_contract_repair_frozen",
        "round_validator_terminal_retry_policy_frozen",
        "round_validator_base_budget_frozen",
        "round_validator_terminal_budget_frozen",
    }
    if (all(preflight.get("checks", {}).get(key) is False
            for key in stale_false_keys) and
            all(preflight.get("checks", {}).get(key) is True for key in companions)):
        for key in stale_false_keys:
            preflight["checks"].pop(key)
        preflight["errors"] = [
            error for error in preflight.get("errors", [])
            if error not in {
                "canary29_validator_template_preflight_failed",
                "canary30_terminal_validator_preflight_failed",
            }
        ]
        preflight["valid"] = bool(
            all(preflight.get("checks", {}).values()) and
            not preflight.get("errors"))
    checks = {
        "canary30_tree_sha": CANARY_30.is_dir() and
            hash_tree(CANARY_30)["tree_sha256"] == EXPECTED_CANARY_30_SHA,
        "round_validator_input_v6_projection_frozen":
            validator_runtime.ROUND_VALIDATOR_INPUT_VERSION.endswith(
                "v6_knowledge_order_projection"),
        "round_validator_exact_slot_prompt_frozen":
            ROUND_VALIDATOR_PROMPT_VERSION.endswith(
                "v13_compact_exact_slots"),
        "round_validator_output_budget_frozen":
            validator_runtime.ROUND_VALIDATOR_MAX_OUTPUT_TOKENS == 12_288,
        "round_validator_scoped_repair_still_frozen":
            contract_repair.ROUND_VALIDATOR_CONTRACT_REPAIR_VERSION_V11 ==
            "round_validator_contract_only_repair_v11_v15_scope",
        "round_validator_retry_budget_still_frozen":
            ROUND_VALIDATOR_MAX_ATTEMPTS == 2 and
            ROUND_VALIDATOR_TERMINAL_CONTRACT_REPAIR_ATTEMPTS == 1 and
            ROUND_VALIDATOR_ABSOLUTE_MAX_ATTEMPTS == 3,
    }
    preflight["checks"].update(checks)
    if not all(checks.values()):
        preflight["errors"] = [*preflight.get("errors", []),
                               "canary31_validator_budget_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and all(checks.values()))
    return preflight


def _markdown(summary: dict[str, Any]) -> str:
    roles = ((summary.get("api_usage") or {}).get("roles") or {})
    return f"""# v11 Continuous Expansion Live Canary 31

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

The Validator request preserves all dialogue and historical turns while using
the v6 non-duplicating knowledge/order projection and a 12,288-token output
budget. No QA, Round 2, Evidence Committer, Questioner, Distractor, Solver,
Strategy, Training, or batch work is executed.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 continuous-expansion live canary 31")
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
    atomic_json(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_31.json",
                summary)
    atomic_text(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_31.md",
                _markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
