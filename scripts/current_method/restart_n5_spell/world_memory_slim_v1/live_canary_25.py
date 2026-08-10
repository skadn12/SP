"""Single-use continuous-expansion live canary 25."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text
from . import durable_update_compiler
from . import live_canary_24 as previous
from . import round_validator_contract_repair_v11 as contract_repair
from .provenance import hash_tree
from .round_validator_contract import ROUND_VALIDATOR_MAX_ATTEMPTS


runtime = previous.runtime
preflight_runtime = previous.preflight_runtime
RUN_ID = (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_25"
)
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_24 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_24"
)
EXPECTED_CANARY_24_SHA = (
    "da856fba136a1dddcb6d5887ad82d5e69e35944775ab077ac3d1652288e12578"
)
_base_historical_manifests = previous.historical_manifests_canary_24


def historical_manifests_canary_25() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests["live_canary_24"] = hash_tree(CANARY_24)
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_25
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_25


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    # Canary21 owns the current common live-contract baseline. Canary22–24
    # froze superseded repair versions, so retain their integrity via the
    # historical manifest and assert only the current replacement contracts.
    preflight = previous.previous.previous._augment_preflight(
        preflight, input_path)
    runtime_source = Path(runtime.__file__).read_text(encoding="utf-8")
    checks = {
        "canary24_tree_sha": CANARY_24.is_dir() and
            hash_tree(CANARY_24)["tree_sha256"] == EXPECTED_CANARY_24_SHA,
        "identity_continuation_repair_frozen":
            contract_repair.ROUND_VALIDATOR_CONTRACT_REPAIR_VERSION_V11 ==
            "round_validator_contract_only_repair_v11_v12_identity_continuation",
        "round_participant_binding_compiler_frozen":
            durable_update_compiler.DURABLE_UPDATE_COMPILER_VERSION ==
            "durable_update_compiler_post_review_v8_round_participant_binding",
        "cross_scene_temporary_ref_binding_frozen":
            "committed_temporary_ref_allocations" in runtime_source,
        "validator_request_budget_unchanged": ROUND_VALIDATOR_MAX_ATTEMPTS == 2,
    }
    for name, value in checks.items():
        preflight["checks"][name] = value
    if not all(checks.values()):
        preflight["errors"] = [*preflight.get("errors", []),
                               "canary25_durable_binding_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and all(checks.values()))
    return preflight


def _markdown(summary: dict[str, Any]) -> str:
    roles = ((summary.get("api_usage") or {}).get("roles") or {})
    return f"""# v11 Continuous Expansion Live Canary 25

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

This is one fresh generation canary. It executes no QA, Round 2, Evidence
Committer, Questioner, Distractor, Solver, Strategy, Training, or batch work.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 continuous-expansion live canary 25")
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
    atomic_json(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_25.json",
                summary)
    atomic_text(args.run_root / "reports/V11_CONTINUOUS_EXPANSION_LIVE_CANARY_25.md",
                _markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
