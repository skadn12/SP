"""Single-use continuous-expansion live canary 33."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text
from . import live_canary_32 as previous
from .action_availability_v11 import ACTION_AVAILABILITY_VIEW_VERSION
from .planner_action_repair_v11 import (
    PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11,
    PLANNER_ACTION_REPAIR_VERSION_V11,
)
from .planner_outline_v11 import (
    PLANNER_OUTLINE_COMPILER_VERSION_V11,
    PLANNER_OUTLINE_SCHEMA_VERSION_V11,
)
from .provenance import hash_tree


runtime = previous.runtime
preflight_runtime = previous.preflight_runtime
RUN_ID = (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_33"
)
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_32 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_32"
)
EXPECTED_CANARY_32_SHA = (
    "a92387aa0840f78137364f5110b2f329ba8814823b8e6131e5939326e2410128"
)
_base_historical_manifests = previous.historical_manifests_canary_32


def historical_manifests_canary_33() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests["live_canary_32"] = hash_tree(CANARY_32)
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_33
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_33


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    preflight = previous._augment_preflight(preflight, input_path)
    # Canary17 asserted exact prompt-version suffixes that have since been
    # replaced by stricter field-scoped action repair and link-choice repair.
    # Retire only those implementation-string assertions while retaining the
    # two historical artifact baselines that made them auditable.
    stale_repair_checks = {
        "action_repair_explicit_envelope_prompt",
        "namespace_repair_explicit_envelope_prompt",
    }
    historical_companions = {
        "canary16_tree_sha",
        "planner_namespace_repair_probe10_tree_sha",
    }
    if (all(preflight.get("checks", {}).get(key) is False
            for key in stale_repair_checks) and
            all(preflight.get("checks", {}).get(key) is True
                for key in historical_companions)):
        for key in stale_repair_checks:
            preflight["checks"].pop(key)
        preflight["errors"] = [
            error for error in preflight.get("errors", [])
            if error != "canary17_frozen_repair_preflight_failed"
        ]
        preflight["valid"] = bool(
            all(preflight.get("checks", {}).values()) and
            not preflight.get("errors"))
    checks = {
        "canary32_tree_sha": CANARY_32.is_dir() and
            hash_tree(CANARY_32)["tree_sha256"] == EXPECTED_CANARY_32_SHA,
        "planner_outline_authoritative_connection_refs_frozen":
            PLANNER_OUTLINE_SCHEMA_VERSION_V11.endswith(
                "v10_authoritative_connection_refs") and
            PLANNER_OUTLINE_COMPILER_VERSION_V11.endswith(
                "v6_authoritative_connection_refs"),
        "action_availability_authoritative_connection_refs_frozen":
            ACTION_AVAILABILITY_VIEW_VERSION.endswith(
                "v6_authoritative_connection_refs"),
        "planner_action_repair_field_scoped_aliases_frozen":
            PLANNER_ACTION_REPAIR_VERSION_V11.endswith(
                "v7_authoritative_connection_refs") and
            PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11.endswith(
                "v8_field_scoped_aliases"),
    }
    preflight["checks"].update(checks)
    if not all(checks.values()):
        preflight["errors"] = [*preflight.get("errors", []),
                               "canary33_planner_reference_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and all(checks.values()))
    return preflight


def _markdown(summary: dict[str, Any]) -> str:
    roles = ((summary.get("api_usage") or {}).get("roles") or {})
    return f"""# v11 Continuous Expansion Live Canary 33

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

Round-local people may speak and participate immediately. Explicit connected-world
edges remain bound to frozen authoritative references. No QA, Round 2, Evidence
Committer, Questioner, Distractor, Solver, Strategy, Training, or batch work is run.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 continuous-expansion live canary 33")
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
    atomic_json(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_33.json",
                summary)
    atomic_text(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_33.md",
                _markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
