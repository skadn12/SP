"""Single-use v11 person/event authoritative-growth live canary."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_canary_v11 as runtime
import scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 as preflight_runtime

from ..core import atomic_json, atomic_text, sha256_file
from ..unified_dialogue_root import read_unified_parent
from .action_availability_v11 import compile_action_availability_view
from .growth_profile_v11 import (
    frozen_person_event_growth_profile, validate_growth_profile_definition,
)
from .interaction_links import compile_source_interaction_links
from .planning_state_view import compile_planning_state_view
from .provenance import hash_tree
from .source_memory_initializer_v1 import initialize_source_world_memory


RUN_ID = "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_15"
RUN_ROOT = preflight_runtime.RUNS / RUN_ID
CANARY_14 = preflight_runtime.RUNS / (
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_14")
PHASE_C1_REPORT = preflight_runtime.PACKAGE_ROOT / (
    "reports/V11_PERSON_EVENT_DURABLE_GROWTH_REPORT.json")
EXPECTED_CANARY_14_SHA = (
    "f950428152ab5937d5ba41c0774d35e60e3be63a81be450767ebc1ae6a2596b7")
EXPECTED_PHASE_C1_REPORT_SHA = (
    "8bd6f44b4f335cfb44bd22502f82df3f9f8ca65fe004409dc51efdc3d01fd925")
EXPECTED_APPEND_ONLY_MEMORY_SHA = (
    "2911b70dc020e56ceea4dbe0aad4507cd72f87d51b20e7354e9a25ec7bf1618d")

_base_historical_manifests = preflight_runtime.historical_manifests_v11


def historical_manifests_canary_15() -> dict[str, Any]:
    manifests = _base_historical_manifests()
    manifests.update({
        "live_canary_14": hash_tree(CANARY_14),
        "phase_c1_growth_report": {
            "path": str(PHASE_C1_REPORT),
            "file_sha256": sha256_file(PHASE_C1_REPORT)
                if PHASE_C1_REPORT.is_file() else None,
        },
    })
    return manifests


preflight_runtime.DEFAULT_RUN_ROOT = RUN_ROOT
preflight_runtime.historical_manifests_v11 = historical_manifests_canary_15
runtime.DEFAULT_RUN_ROOT = RUN_ROOT
runtime.RUN_ID = RUN_ID
runtime.historical_manifests_v11 = historical_manifests_canary_15


def _growth_contract_preflight(input_path: Path) -> dict[str, Any]:
    root = json.loads(input_path.read_text(encoding="utf-8"))
    parent = read_unified_parent(input_path)
    source = initialize_source_world_memory(root, parent)
    links = compile_source_interaction_links(
        root_id=str(root["root_id"]), sessions=runtime._sessions(parent),
        speaker_to_entity=source["speaker_to_entity_ref"])
    planning = compile_planning_state_view(
        memory=source["memory"], interaction_links=links)
    availability = compile_action_availability_view(planning_state=planning)
    profile = frozen_person_event_growth_profile()
    profile_validation = validate_growth_profile_definition(profile)
    actions = availability.get("actions") or {}
    required = ["introduce_connected_person", "introduce_connected_event"]
    checks = {
        "growth_profile_definition_valid": profile_validation["valid"],
        "required_narrative_actions_available": all(
            (actions.get(value) or {}).get("available") is True for value in required),
        "connected_person_is_not_a_relation_assertion": (
            "relation_type" not in ((actions.get("introduce_connected_person") or {}).get(
                "argument_contract") or {})),
        "phase_c1_report_sha": PHASE_C1_REPORT.is_file() and
            sha256_file(PHASE_C1_REPORT) == EXPECTED_PHASE_C1_REPORT_SHA,
        "canary14_tree_sha": CANARY_14.is_dir() and
            hash_tree(CANARY_14)["tree_sha256"] == EXPECTED_CANARY_14_SHA,
        "append_only_core_unchanged": sha256_file(
            preflight_runtime.PACKAGE_ROOT / "append_only_memory.py") ==
            EXPECTED_APPEND_ONLY_MEMORY_SHA,
    }
    return {
        "valid": all(checks.values()), "checks": checks,
        "growth_profile": profile,
        "required_action_availability": {
            value: actions.get(value) for value in required},
        "api_calls": 0, "network_attempts": 0,
    }


def _augment_preflight(preflight: dict[str, Any], input_path: Path) -> dict[str, Any]:
    growth = _growth_contract_preflight(input_path)
    preflight["growth_canary_version"] = "v11_person_event_growth_live_canary_v1"
    preflight["growth_contract_preflight"] = growth
    preflight["growth_profile"] = growth["growth_profile"]
    for name, value in growth["checks"].items():
        preflight["checks"][f"growth_{name}"] = value
    if not growth["valid"]:
        preflight["errors"] = [*preflight.get("errors", []),
                               "person_event_growth_preflight_failed"]
    preflight["valid"] = bool(preflight.get("valid") and growth["valid"])
    return preflight


def _growth_markdown(summary: dict[str, Any]) -> str:
    audit = summary.get("authoritative_growth_audit") or {}
    usage = summary.get("api_usage") or {}
    roles = usage.get("roles") or {}
    return f"""# v11 Person/Event Growth Live Canary 15

## Outcome

- status: `{summary.get('status')}`
- conclusion: **{summary.get('conclusion')}**
- generation_ready: `{str(summary.get('generation_ready')).lower()}`
- fresh_e2e_ready: `{str(summary.get('fresh_e2e_ready')).lower()}`
- growth_profile_ready: `{str(summary.get('growth_profile_ready')).lower()}`
- qa_pipeline_ready: `false`

## Authoritative growth

- new entities: {audit.get('new_authoritative_entity_count', 0)}
- new connections: {audit.get('new_authoritative_connection_count', 0)}
- new events: {audit.get('new_authoritative_event_count', 0)}
- orphan entities: {len(audit.get('orphan_entity_refs') or [])}
- cross-root refs: {len(audit.get('cross_root_refs') or [])}
- partial atomic groups: {len(audit.get('partial_atomic_groups') or [])}
- Memory ID collisions: {len(audit.get('memory_id_collisions') or [])}

## Requests

- Planner: {((roles.get('planner') or {}).get('physical_requests') or 0)}
- Realizer: {((roles.get('realizer') or {}).get('physical_requests') or 0)}
- Round Validator: {((roles.get('round_validator') or {}).get('physical_requests') or 0)}
- total requests: {usage.get('total_physical_requests', 0)}
- total tokens: {usage.get('total_tokens', 0)}

No Round 2, QA, Evidence Committer, Questioner, Distractor, Solver, Strategy,
Training, batch runner, or Git commit was executed.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-use v11 person/event authoritative-growth live canary")
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
        growth_profile=preflight["growth_profile"])
    atomic_json(args.run_root / "reports/V11_PERSON_EVENT_GROWTH_LIVE_CANARY_REPORT.json",
                summary)
    atomic_text(args.run_root / "reports/V11_PERSON_EVENT_GROWTH_LIVE_CANARY_REPORT.md",
                _growth_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
