"""Single-use dialogue-first person/event growth live canary 16."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text
from . import live_canary_15 as previous
from .dialogue_first_participant_preflight import run_eleven_root_preflight
from .provenance import hash_tree


runtime = previous.runtime
preflight_runtime = previous.preflight_runtime

RUN_ID = (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_16"
)
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_15 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_15"
)
PARTICIPANT_PROBE_4 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_4"
)
EXPECTED_CANARY_15_SHA = (
    "fca44d7b6e822ba9d6e8cf547d604d86b34008c47643f5f93212e536f31fc4b0"
)
EXPECTED_PARTICIPANT_PROBE_4_SHA = (
    "4ba22cffef25a940c5feb0af6acb0645db0887c642ad2a36d709ddb9805cfcb0"
)

_base_historical_manifests = previous.historical_manifests_canary_15


def historical_manifests_canary_16() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests.update({
        "live_canary_15": hash_tree(CANARY_15),
        "dialogue_first_participant_probe_4": hash_tree(PARTICIPANT_PROBE_4),
    })
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_16
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_16


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    preflight = previous._augment_preflight(preflight, input_path)
    roots = run_eleven_root_preflight()
    checks = {
        "canary15_tree_sha": CANARY_15.is_dir() and
            hash_tree(CANARY_15)["tree_sha256"] == EXPECTED_CANARY_15_SHA,
        "participant_probe4_tree_sha": PARTICIPANT_PROBE_4.is_dir() and
            hash_tree(PARTICIPANT_PROBE_4)["tree_sha256"] ==
            EXPECTED_PARTICIPANT_PROBE_4_SHA,
        "dialogue_first_eleven_root_preflight": roots.get("all_passed") is True and
            roots.get("passed_count") == 11 and roots.get("api_calls") == 0,
    }
    preflight["dialogue_first_participants"] = True
    preflight["dialogue_first_contract_version"] = (
        "round_participant_registry_v2_declare_then_reference"
    )
    preflight["dialogue_first_eleven_root_preflight"] = roots
    for name, value in checks.items():
        preflight["checks"][name] = value
    if not all(checks.values()):
        preflight["errors"] = [*preflight.get("errors", []),
                               "dialogue_first_canary_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and all(checks.values()))
    return preflight


def _markdown(summary: dict[str, Any]) -> str:
    usage = summary.get("api_usage") or {}
    roles = usage.get("roles") or {}
    return f"""# v11 Dialogue-First Growth Live Canary 16

- status: `{summary.get('status')}`
- conclusion: **{summary.get('conclusion')}**
- planned / realized scenes: {summary.get('planned_scene_count')} / {summary.get('realized_scene_count')}
- realized / promotable turns: {summary.get('realized_turn_count')} / {summary.get('promotable_turn_count')}
- Planner requests: {((roles.get('planner') or {}).get('physical_requests') or 0)}
- Realizer requests: {((roles.get('realizer') or {}).get('physical_requests') or 0)}
- Round Validator requests: {((roles.get('round_validator') or {}).get('physical_requests') or 0)}
- generation_ready: `{str(summary.get('generation_ready')).lower()}`
- fresh_e2e_ready: `{str(summary.get('fresh_e2e_ready')).lower()}`
- qa_pipeline_ready: `false`

New dialogue participants are declared once and referenced by alias thereafter. Speaking
does not itself create authoritative Memory. No Round 2, QA, Evidence Committer,
Questioner, Distractor, Solver, Strategy, Training, batch runner, or Git commit ran.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 dialogue-first growth live canary 16")
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
    atomic_json(args.run_root / "reports/V11_DIALOGUE_FIRST_GROWTH_LIVE_CANARY_REPORT.json",
                summary)
    atomic_text(args.run_root / "reports/V11_DIALOGUE_FIRST_GROWTH_LIVE_CANARY_REPORT.md",
                _markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
