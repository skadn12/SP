"""One-shot Planner-only live transport probe for v11.

The probe regenerates the Planner prompt from the authoritative Parent and
requires byte-identical prompt SHA with the failed canary.  It cannot invoke a
Realizer, Round Validator, Binder, QA component, or the full-canary runner.
"""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..unified_dialogue_root import (
    read_unified_parent, validate_unified_dialogue_root,
)
from .complete_dialogue_view import compose_complete_dialogue_view
from .action_availability_v11 import compile_action_availability_view
from .continuous_round_plan_v1 import (
    build_scene_dependency_dag,
)
from .controlled_expansion import compile_action_menu_v3, validate_action_menu_v3
from .interaction_links import compile_source_interaction_links
from .live_contracts_v11 import (
    PLANNER_PROMPT_VERSION_V11, build_planner_prompt_v11,
    prompt_schema_versions_v11,
)
from .live_preflight_b2_v11 import (
    DEFAULT_API_CONFIG, DEFAULT_INPUT, DEFAULT_TOKENIZER, PACKAGE_ROOT,
    PROJECT_ROOT, implementation_manifest_v11,
)
from .live_role_runtime_v11 import LiveJsonRoleCallerV11, aggregate_v11_usage
from .parent_anchor_compiler import compile_anchor_registry
from .narrative_action_contracts import action_contract_registry
from .placement_choice_registry import (
    compile_placement_choice_registry, compile_source_anchor_registry,
)
from .planner_transport_v11 import (
    PLANNER_MAX_OUTPUT_TOKENS, PlannerTransportRecoveryClientV11,
    planner_transport_configuration_v11,
)
from .planner_outline_v11 import (
    compile_planner_outline_v11, validate_planner_outline_v11,
)
from .planning_state_view import compile_planning_state_view
from .provenance import git_value, hash_tree, project_diff_sha256
from .source_memory_initializer_v1 import (
    authoritative_affordance_view, initialize_source_world_memory,
)
from .structural_expansion_gate import validate_round_blueprint_structure


PROBE_VERSION = "v11_planner_only_transport_probe_v1"
ROOT_ID = "cc-train-episode-61555"
SEED = 61555
FAILED_CANARY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_1")
DEFAULT_PROBE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_planner_transport_probe_1")
FAILED_CANARY_TREE_SHA = "4e0c7a7d0e390ce3a43167dce5f7031a25166a5336feaa8fd3de9f8f58c607b3"
FAILED_PROMPT_SHA = "aad4621f44622d53efbb7f7a2fe411c4fb5180a0f9a4761241661a3f7598bce8"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def _sessions(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for turn in (parent.get("dialogue") or {}).get("turns") or []:
        ref = str(turn.get("session_id") or "")
        if ref not in grouped:
            order.append(ref)
            grouped[ref] = []
        grouped[ref].append(dict(turn))
    return [{"session_id": ref, "turns": grouped[ref]} for ref in order]


def build_planner_probe_input(input_path: Path) -> dict[str, Any]:
    root = json.loads(input_path.read_text(encoding="utf-8"))
    validation = validate_unified_dialogue_root(root)
    if not validation["valid"] or root.get("root_id") != ROOT_ID:
        raise ValueError("planner_probe_authoritative_input_invalid")
    parent = read_unified_parent(input_path)
    source = initialize_source_world_memory(root, parent)
    interactions = compile_source_interaction_links(
        root_id=ROOT_ID, sessions=_sessions(parent),
        speaker_to_entity=source["speaker_to_entity_ref"])
    complete = compose_complete_dialogue_view(
        original_parent=parent, promoted_rounds=(),
        speaker_to_entity_ref=source["speaker_to_entity_ref"])
    planning = compile_planning_state_view(
        memory=source["memory"], interaction_links=interactions,
        scene_lineage=(), recent_round_end={})
    anchors = compile_anchor_registry(
        complete_dialogue=complete, planning_state=planning, tail_turn_count=7)
    source_anchors = compile_source_anchor_registry(anchors)
    placement_choices = compile_placement_choice_registry(
        complete_dialogue=complete, anchor_registry=anchors)
    action_menu = compile_action_menu_v3(
        authoritative_affordance_view(source["memory"]), root_id=ROOT_ID,
        interaction_links=interactions, maximum_actions=16)
    action_validation = validate_action_menu_v3(action_menu)
    if not action_validation["valid"]:
        raise ValueError("planner_probe_action_menu_invalid")
    role_table = [{"entity_ref": row["entity_ref"],
                   "display_name": row["display_name"],
                   "entity_type": row.get("entity_type") or "person"}
                  for row in planning.get("entities") or []]
    shared_action_contracts = action_contract_registry()
    action_availability = compile_action_availability_view(planning_state=planning)
    prompt, frozen_planner_input_sha = build_planner_prompt_v11(
        root_id=ROOT_ID, planning_state=planning,
        source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        action_contracts=shared_action_contracts,
        action_availability_view=action_availability)
    return {
        "root": root, "parent": parent, "source": source,
        "interaction_links": interactions, "complete_dialogue": complete,
        "planning_state": planning, "anchor_registry": anchors,
        "source_anchor_registry": source_anchors,
        "placement_choice_registry": placement_choices,
        "action_menu": action_menu, "role_table": role_table,
        "action_availability_view": action_availability,
        "shared_narrative_action_contracts": shared_action_contracts, "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "frozen_planner_input_sha256": frozen_planner_input_sha,
    }


def planner_probe_preflight(
    *, runner_path: Path, input_path: Path, probe_root: Path,
    api_config_path: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    values: dict[str, Any] = {}
    try:
        values = build_planner_probe_input(input_path)
    except Exception as error:
        errors.append(f"input_compilation:{type(error).__name__}:{error}")
    freeze = json.loads((FAILED_CANARY / "run_freeze_manifest.json").read_text())
    failed_ledger = json.loads((FAILED_CANARY /
        "ledger/requests/planner_v11_attempt_01.json").read_text())
    config = load_challenger_config(api_config_path)
    schema_versions = prompt_schema_versions_v11()
    schema_sha = hashlib.sha256(json.dumps(
        schema_versions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    tokenizer_valid = False
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tokenizer_valid = True
    except Exception as error:
        errors.append(f"tokenizer:{type(error).__name__}:{error}")
    compile_ok = compileall.compile_dir(str(PACKAGE_ROOT), quiet=1)
    diff_ok = subprocess.run(
        ["git", "diff", "--check"], cwd=PROJECT_ROOT,
        capture_output=True).returncode == 0
    canary_tree = hash_tree(FAILED_CANARY)
    implementation = implementation_manifest_v11()
    transport = planner_transport_configuration_v11()
    checks = {
        "probe_root_absent": not probe_root.exists(),
        "probe_root_exact": probe_root.resolve() == DEFAULT_PROBE_ROOT.resolve(),
        "runner_path_exact": runner_path.resolve() == Path(__file__).resolve(),
        "authoritative_input_exact": input_path.resolve() == DEFAULT_INPUT.resolve(),
        "input_sha_unchanged": sha256_file(input_path) == freeze["input_sha256"],
        "failed_canary_read_only_sha": canary_tree["tree_sha256"] == FAILED_CANARY_TREE_SHA,
        "failed_request_was_read_timeout": failed_ledger.get("status") == "reserved",
        "prompt_sha_regenerated_exact": values.get("prompt_sha256") == FAILED_PROMPT_SHA,
        "prompt_sha_matches_failed_ledger": values.get("prompt_sha256") ==
            failed_ledger.get("prompt_sha256"),
        "prompt_schema_sha_unchanged": schema_sha == freeze[
            "prompt_schema_contract_sha256"],
        "seed_unchanged": freeze.get("orchestration_seed") == SEED,
        "model_unchanged": config.model == freeze["api_configuration"]["model"],
        "api_config_unchanged": sha256_file(api_config_path) == freeze["api_config_sha256"],
        "transport_connect_30": transport["connect_timeout_seconds"] == 30,
        "transport_read_600": transport["read_timeout_seconds"] == 600,
        "transport_streaming": transport["streaming"] is True,
        "transport_retry_once": transport["timeout_retry_limit"] == 1,
        "planner_max_output_10240": transport["max_output_tokens"] == 10_240,
        "implementation_manifest_valid": implementation["valid"],
        "tokenizer_valid": tokenizer_valid,
        "compileall_passed": compile_ok, "git_diff_check_passed": diff_ok,
        "allowed_roles_planner_only": True,
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        "preflight_version": "v11_planner_probe_preflight_v1",
        "valid": not errors, "errors": list(dict.fromkeys(errors)),
        "checks": checks, "api_calls": 0, "network_attempts": 0,
        "runner_path": str(runner_path.resolve()),
        "probe_root": str(probe_root.resolve()), "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "failed_canary_tree": canary_tree,
        "failed_prompt_sha256": FAILED_PROMPT_SHA,
        "regenerated_prompt_sha256": values.get("prompt_sha256"),
        "frozen_planner_input_sha256": values.get("frozen_planner_input_sha256"),
        "prompt_schema_versions": schema_versions,
        "prompt_schema_contract_sha256": schema_sha,
        "orchestration_seed": SEED,
        "model_configuration": {"model": config.model,
            "base_url": config.base_url, "temperature": config.temperature,
            "api_key_recorded": False},
        "api_config_path": str(api_config_path.resolve()),
        "api_config_sha256": sha256_file(api_config_path),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "transport_configuration": transport,
        "implementation_manifest": implementation,
        "working_tree_manifest_sha256": project_diff_sha256(PROJECT_ROOT),
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "request_budget": {"planner_logical_requests": 1,
                           "planner_transport_attempts_maximum": 2,
                           "all_other_roles": 0},
    }


def _guard(freeze: Mapping[str, Any]) -> dict[str, Any]:
    values = build_planner_probe_input(Path(freeze["input_path"]))
    checks = {
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD") == freeze["git_head"],
        "implementation_manifest": implementation_manifest_v11()[
            "implementation_manifest_sha256"] == freeze["implementation_manifest"][
                "implementation_manifest_sha256"],
        "working_tree_manifest": project_diff_sha256(PROJECT_ROOT) == freeze[
            "working_tree_manifest_sha256"],
        "input_sha": sha256_file(Path(freeze["input_path"])) == freeze["input_sha256"],
        "api_config_sha": sha256_file(Path(freeze["api_config_path"])) == freeze[
            "api_config_sha256"],
        "failed_canary_sha": hash_tree(FAILED_CANARY)["tree_sha256"] ==
            FAILED_CANARY_TREE_SHA,
        "prompt_sha": values["prompt_sha256"] == freeze["regenerated_prompt_sha256"],
        "seed": freeze["orchestration_seed"] == SEED,
    }
    return {"valid": all(checks.values()), "checks": checks}


def _report(summary: Mapping[str, Any]) -> str:
    usage = summary["api_usage"]
    return f"""# V11 Planner-only Transport Probe

- Status: `{summary['status']}`
- Failure stage: `{summary.get('failure_stage')}`
- Transport completed: `{str(summary['transport_completed']).lower()}`
- JSON parsed: `{str(summary['json_parsed']).lower()}`
- Planner schema valid: `{str(summary['planner_schema_valid']).lower()}`
- Planner structural plan valid: `{str(summary['planner_structural_valid']).lower()}`
- Logical Planner requests: {usage['roles']['planner']['logical_requests']}
- Physical transport attempts: {usage['roles']['planner']['physical_requests']}
- Realizer / Round Validator / Evidence Committer calls: 0 / 0 / 0
- Prompt/completion/total tokens: {usage['prompt_tokens']} / {usage['completion_tokens']} / {usage['total_tokens']}
- Prompt SHA matches failed canary: `{str(summary['prompt_sha_exact']).lower()}`
- Historical integrity: `{str(summary['post_run_integrity']['valid']).lower()}`

The probe stopped immediately after the Planner transport, parsing, schema, compile, and local structural checks. It did not start a complete canary or any downstream role.
"""


def execute_probe(
    *, preflight: Mapping[str, Any], input_path: Path, probe_root: Path,
    api_config_path: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    if not preflight["valid"] or probe_root.exists():
        raise RuntimeError("planner_probe_preflight_or_freshness_failure")
    probe_root.mkdir(parents=True, exist_ok=False)
    freeze = {**deepcopy(dict(preflight)), "probe_version": PROBE_VERSION,
              "fresh": True, "no_resume": True, "frozen_at": _now()}
    atomic_json(probe_root / "probe_freeze_manifest.json", freeze)
    atomic_json(probe_root / "run_status.json", {
        "status": "initialized", "api_calls": 0, "created_at": _now()})
    values = build_planner_probe_input(input_path)
    for relative, value in {
        "complete_dialogue_view.json": values["complete_dialogue"],
        "planning_state_view.json": values["planning_state"],
        "anchor_registry.json": values["anchor_registry"],
        "source_anchor_registry.json": values["source_anchor_registry"],
        "placement_choice_registry.json": values["placement_choice_registry"],
        "action_menu.json": values["action_menu"],
        "action_availability_view.json": values["action_availability_view"],
    }.items():
        atomic_json(probe_root / relative, value)
    atomic_text(probe_root / "planner_prompt.txt", values["prompt"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    client = PlannerTransportRecoveryClientV11(
        config=load_challenger_config(api_config_path),
        authoritative_input_sha256=freeze["input_sha256"],
        orchestration_seed=SEED,
        prompt_schema_contract_sha256=freeze["prompt_schema_contract_sha256"])
    ledger = RequestLedger(
        probe_root, freeze["implementation_manifest"]["implementation_manifest_sha256"],
        "live")

    def before_request(logical_role: str, role_version: str) -> None:
        if logical_role != "planner" or role_version != "anchor_first_continuous_planner_v11":
            raise RuntimeError("planner_probe_forbidden_role")
        integrity = _guard(freeze)
        if not integrity["valid"]:
            atomic_json(probe_root / "pre_request_integrity_failure.json", integrity)
            raise RuntimeError("planner_probe_freeze_drift")
        atomic_json(probe_root / "run_status.json", {
            "status": "planner_request_started", "started_at": _now(),
            "all_other_roles_enabled": False})

    caller = LiveJsonRoleCallerV11(
        run_root=probe_root, ledger=ledger, client=client,
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        root_id=ROOT_ID, role_budget_limits={"planner": 1, "realizer": 0,
                                             "round_validator": 0},
        total_request_limit=1, context_limit=40_960,
        before_request=before_request)
    parsed = None
    parse_errors: list[str] = []
    failure_stage = None
    runtime_error = None
    validation = None
    structural = None
    compiled = None
    try:
        parsed, _, parse_errors = caller.call_json(
            request_id="planner_v11_transport_probe_01", role="planner_v11",
            attempt=1, prompt=values["prompt"],
            prompt_version=PLANNER_PROMPT_VERSION_V11,
            max_tokens=PLANNER_MAX_OUTPUT_TOKENS)
        if parsed is not None:
            validation = validate_planner_outline_v11(
                parsed,
                source_anchor_registry=values["source_anchor_registry"],
                placement_choice_registry=values["placement_choice_registry"],
                action_availability_view=values["action_availability_view"],
                authoritative_entity_refs={row["entity_ref"]
                                           for row in values["role_table"]})
            if validation["valid"]:
                compiled = compile_planner_outline_v11(
                    parsed,
                    source_anchor_registry=values["source_anchor_registry"],
                    placement_choice_registry=values["placement_choice_registry"],
                    planning_state=values["planning_state"],
                    action_availability_view=values["action_availability_view"],
                    round_index=1)
                structural = validate_round_blueprint_structure(
                    compiled_plan=compiled,
                    source_anchor_registry=values["source_anchor_registry"],
                    placement_choice_registry=values["placement_choice_registry"],
                    complete_dialogue=values["complete_dialogue"],
                    planning_state=values["planning_state"])
        atomic_json(probe_root / "planner_response_parsed.json", parsed)
        atomic_json(probe_root / "planner_validation.json", {
            "parse_errors": parse_errors, "schema_validation": validation,
            "structural_validation": structural})
        if compiled is not None:
            atomic_json(probe_root / "compiled_round_plan_probe.json", compiled)
    except Exception as error:
        failure_stage = "Planner transport"
        runtime_error = f"{type(error).__name__}:{error}"
        atomic_json(probe_root / "failure.json", {
            "failure_stage": failure_stage, "runtime_error": runtime_error})
    usage = aggregate_v11_usage(caller.records)
    post = _guard(freeze)
    transport_completed = bool(caller.records and caller.records[-1].get("request_completed"))
    summary = {
        "probe_version": PROBE_VERSION,
        "status": "completed" if transport_completed else "failed",
        "failure_stage": failure_stage, "runtime_error": runtime_error,
        "transport_completed": transport_completed,
        "json_parsed": parsed is not None and not parse_errors,
        "planner_schema_valid": bool(validation and validation["valid"]),
        "planner_structural_valid": bool(structural and structural["valid"]),
        "scene_count": int((validation or {}).get("scene_count") or 0),
        "prompt_sha256": values["prompt_sha256"],
        "prompt_sha_exact": values["prompt_sha256"] == FAILED_PROMPT_SHA,
        "api_usage": usage, "post_run_integrity": post,
        "realizer_calls": 0, "round_validator_calls": 0,
        "evidence_committer_calls": 0, "initial_structure_extractor_calls": 0,
        "round_2_calls": 0, "qa_calls": 0, "solver_calls": 0,
        "training_calls": 0,
    }
    atomic_json(probe_root / "reports/V11_PLANNER_ONLY_PROBE_REPORT.json", summary)
    atomic_text(probe_root / "reports/V11_PLANNER_ONLY_PROBE_REPORT.md", _report(summary))
    atomic_json(probe_root / "reports/post_run_integrity.json", post)
    atomic_json(probe_root / "api_usage.json", usage)
    if ledger.events.is_file():
        atomic_text(probe_root / "request_ledger.jsonl",
                    ledger.events.read_text(encoding="utf-8"))
    else:
        atomic_text(probe_root / "request_ledger.jsonl", "")
    atomic_json(probe_root / "run_status.json", {
        "status": summary["status"], "failure_stage": failure_stage,
        "transport_completed": transport_completed,
        "planner_schema_valid": summary["planner_schema_valid"],
        "planner_structural_valid": summary["planner_structural_valid"],
        "completed_at": _now(), "api_usage": usage})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot v11 Planner-only transport probe")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight = planner_probe_preflight(
        runner_path=Path(__file__), input_path=args.input.resolve(),
        probe_root=args.probe_root.resolve(),
        api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["valid"] else 2
    if not preflight["valid"]:
        print(json.dumps({"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = execute_probe(
        preflight=preflight, input_path=args.input.resolve(),
        probe_root=args.probe_root.resolve(),
        api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["transport_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
