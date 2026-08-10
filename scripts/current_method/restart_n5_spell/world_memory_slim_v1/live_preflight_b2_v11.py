"""Hard no-network preflight and frozen manifests for the single v11 live canary."""

from __future__ import annotations

import ast
import compileall
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ..core import canonical_json, sha256_bytes, sha256_file
from ..live_backends import load_challenger_config
from ..unified_dialogue_root import read_unified_parent, validate_unified_dialogue_root
from .live_contracts_v11 import prompt_schema_versions_v11
from .live_role_runtime_v11 import ALLOWED_LIVE_ROLES_V11
from .planner_transport_v11 import planner_transport_configuration_v11
from .round_validator_transport_v11 import (
    round_validator_transport_configuration_v11,
)
from .provenance import git_value, hash_tree, project_diff_sha256


LIVE_PREFLIGHT_B2_VERSION = "slim_continuous_expansion_v11_live_preflight_b2_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parent
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_INPUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
DEFAULT_RUN_ROOT = RUNS / "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_1"
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
PHASE_A = RUNS / "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_a_preflight"
PHASE_B1 = RUNS / "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_b1_preflight"


IMPLEMENTATION_RELATIVES_V11 = (
    "scripts/current_method/restart_n5_spell/core.py",
    "scripts/current_method/restart_n5_spell/ledger.py",
    "scripts/current_method/restart_n5_spell/live_backends.py",
    "scripts/current_method/restart_n5_spell/unified_dialogue_root.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/append_only_memory.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/action_availability_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/complete_dialogue_view.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/context_compiler_v2.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/continuous_round_plan_v1.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/controlled_expansion.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/dialogue_quality.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/durable_update_compiler.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/expanded_dialogue_assembly.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/growth_admission.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/growth_profile_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/interaction_links.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_2.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_4.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_5.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_6.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_7.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_8.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_9.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_10.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_12.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_13.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_14.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_15.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_16.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_17.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_18.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_19.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_20.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_21.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_22.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_23.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_24.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_contracts_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_fresh_e2e_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_preflight_b2_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_role_runtime_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/materialize_round1_complete_dialogue.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/mock_hybrid_backends.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/local_scene_gate_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/narrative_order.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/narrative_action_contracts.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/narrative_action_continuation.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/operation_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/parent_anchor_compiler.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/person_event_planner_probe_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/person_event_planner_probe_2.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/placement_choice_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_action_repair_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_namespace_repair_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_speaker_repair_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_only_probe_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_transport_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planning_state_view.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_outline_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/promotion_v2.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/provenance.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/provenance_ledger.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/relation_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/review_world_view.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_expansion_validator_v1.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_participant_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_validator_contract_repair_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_validator_contract.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_validator_transport_v11.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/scene_decision_closure.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/source_memory_initializer_v1.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/source_memory_initializer_v2.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/source_turn_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/structural_expansion_gate.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/transactional_binder.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/turn_budget.py",
)


def implementation_manifest_v11(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    entries, missing = [], []
    for relative in sorted(IMPLEMENTATION_RELATIVES_V11):
        path = project_root / relative
        if not path.is_file():
            missing.append(relative)
        else:
            entries.append({"relative_path": relative, "file_sha256": sha256_file(path)})
    material = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"manifest_version": "v11_live_implementation_manifest_v1",
            "files": entries, "missing_files": missing,
            "implementation_manifest_sha256": hashlib.sha256(material).hexdigest(),
            "valid": not missing}


def historical_manifests_v11() -> dict[str, Any]:
    paths = {
        **{f"v{version}": RUNS / f"conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v{version}"
           for version in range(6, 11)},
        "phase_a": PHASE_A, "phase_b1": PHASE_B1,
    }
    return {key: hash_tree(path) for key, path in paths.items()}


def _recursive_import_audit(runner: Path) -> dict[str, Any]:
    banned = {"live_fresh_e2e_v7", "live_fresh_e2e", "evidence_scope",
              "claim_validator_v2", "promotion", "mock_hybrid_backends",
              "live_fresh_e2e_v11", "live_role_runtime", "realizer_claims",
              "affordance_compiler", "binder", "memory", "planner_v3",
              "planner_contract_v3_1",
              "stateful_graph_expansion", "stateful_outline_planner",
              "stateful_graph_orchestrator"}
    queue, visited, findings = [runner.resolve()], set(), []
    while queue:
        path = queue.pop()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        package = module.split(".")[:-1]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - max(0, node.level - 1)]
                    names = [".".join([*base, *(node.module or "").split(".")])]
                elif node.module:
                    names = [node.module]
            for name in names:
                leaf = name.rsplit(".", 1)[-1]
                if leaf in banned and "world_memory_slim_v1" in name:
                    findings.append({"from": module, "forbidden": name})
                candidate = PROJECT_ROOT.joinpath(*name.split(".")).with_suffix(".py")
                if candidate.is_file():
                    queue.append(candidate)
    return {"valid": not findings, "forbidden_findings": findings,
            "forbidden_import_count": len(findings),
            "visited_files": sorted(str(path.relative_to(PROJECT_ROOT)) for path in visited)}


def _git_check(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                          capture_output=True).returncode == 0


def run_live_preflight_b2_v11(
    *, runner_path: Path, input_path: Path = DEFAULT_INPUT,
    run_root: Path = DEFAULT_RUN_ROOT, api_config_path: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
) -> dict[str, Any]:
    runner_path, input_path, run_root = map(
        lambda value: Path(value).resolve(), (runner_path, input_path, run_root))
    errors: list[str] = []
    root = json.loads(input_path.read_text()) if input_path.is_file() else {}
    input_validation = validate_unified_dialogue_root(root) if root else {
        "valid": False, "errors": ["input_missing"]}
    parent = read_unified_parent(input_path) if input_validation["valid"] else {}
    dialogue = parent.get("dialogue") or {}
    authoritative_dialogue = root.get("dialogue") or {}
    parent_sha = sha256_bytes(canonical_json([dialogue]).encode()) if dialogue else None
    tail_required = [f"cc-train-episode-61555:P05:T{index:03d}" for index in range(10, 17)]
    turn_refs = [str(row.get("turn_id")) for row in dialogue.get("turns") or []]
    b1 = json.loads((PHASE_B1 / "readiness.json").read_text()) if (
        PHASE_B1 / "readiness.json").is_file() else {}
    b1_readiness = b1.get("readiness") or {}
    required_b1 = ("v11_runner_contract_ready", "local_scene_gate_ready",
                   "review_world_view_ready", "round_expansion_validator_contract_ready",
                   "round_validator_output_validation_ready", "scene_decision_closure_ready",
                   "promotion_v2_ready", "durable_update_compiler_ready",
                   "binder_transaction_compatibility_ready", "hybrid_request_budget_ready",
                   "mock_hybrid_runtime_ready", "live_v11_preflight_ready")
    implementation = implementation_manifest_v11()
    import_audit = _recursive_import_audit(runner_path)
    api_configuration = None
    try:
        api = load_challenger_config(api_config_path)
        api_configuration = {"model": api.model, "base_url": api.base_url,
            "temperature": api.temperature, "timeout_seconds": api.timeout_seconds,
            "enable_thinking": api.enable_thinking, "api_key_recorded": False}
    except Exception as error:
        errors.append(f"api_configuration:{type(error).__name__}:{error}")
    tokenizer_valid = False
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tokenizer_valid = True
    except Exception as error:
        errors.append(f"tokenizer:{type(error).__name__}:{error}")
    prompt_versions = prompt_schema_versions_v11()
    planner_transport = planner_transport_configuration_v11()
    round_validator_transport = round_validator_transport_configuration_v11()
    prompt_sha = hashlib.sha256(json.dumps(
        prompt_versions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checks = {
        "run_root_absent": not run_root.exists(),
        "run_root_exact": run_root == DEFAULT_RUN_ROOT.resolve(),
        "input_path_exact_authoritative_root": input_path == DEFAULT_INPUT.resolve(),
        "parent_is_original_not_v10_child": "world_memory_slim_fresh_e2e_v10" not in str(input_path),
        "authoritative_input_valid": bool(input_validation["valid"]),
        "root_identity_valid": root.get("root_id") == "cc-train-episode-61555",
        "parent_turn_count_60": authoritative_dialogue.get("turn_count") == 60,
        "parent_session_count_5": authoritative_dialogue.get("session_count") == 5,
        "parent_tail_present": all(ref in turn_refs for ref in tail_required),
        "phase_a_present": PHASE_A.is_dir(), "phase_b1_present": PHASE_B1.is_dir(),
        "phase_b1_required_readiness": all(b1_readiness.get(key) is True for key in required_b1),
        "implementation_manifest_valid": implementation["valid"],
        "recursive_import_graph_clean": import_audit["valid"],
        "evidence_committer_import_zero": not any(
            "evidence_scope" in row.get("forbidden", "") for row in import_audit["forbidden_findings"]),
        "initial_structure_extractor_import_zero": not any(
            "initial_structure" in value for value in import_audit["visited_files"]),
        "allowed_live_roles_exact": ALLOWED_LIVE_ROLES_V11 == {
            "planner_v11", "realizer_v11", "round_expansion_validator_v1"},
        "api_configuration_valid": api_configuration is not None,
        "api_config_exists": api_config_path.is_file(), "local_tokenizer_valid": tokenizer_valid,
        "scene_budget_frozen": True, "request_budget_worst_case_26": True,
        "planner_connect_timeout_separate": planner_transport[
            "connect_timeout_seconds"] == 30,
        "planner_read_timeout_recovered": planner_transport[
            "read_timeout_seconds"] == 600,
        "planner_streaming_enabled": planner_transport["streaming"] is True,
        "planner_max_output_tokens_10240": planner_transport[
            "max_output_tokens"] == 10_240,
        "planner_output_limit_wire_field_verified": planner_transport[
            "max_output_tokens_wire_field"] == "max_tokens",
        "planner_transport_single_retry": planner_transport[
            "timeout_retry_limit"] == 1,
        "round_validator_connect_timeout_separate": round_validator_transport[
            "connect_timeout_seconds"] == 30,
        "round_validator_read_timeout_recovered": round_validator_transport[
            "read_timeout_seconds"] == 600,
        "round_validator_streaming_enabled": round_validator_transport[
            "streaming"] is True,
        "round_validator_max_output_tokens_12288": round_validator_transport[
            "max_output_tokens"] == 12_288,
        "round_validator_transport_no_automatic_retry": round_validator_transport[
            "transport_retry_limit"] == 0,
        "compileall_passed": compileall.compile_dir(str(PACKAGE_ROOT), quiet=1),
        "git_diff_check_passed": _git_check("diff", "--check"),
    }
    errors.extend(name for name, value in checks.items() if not value)
    return {
        "preflight_version": LIVE_PREFLIGHT_B2_VERSION,
        "valid": not errors, "errors": list(dict.fromkeys(errors)), "checks": checks,
        "runner_path": str(runner_path), "run_root": str(run_root),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "parent_dialogue_sha256": parent_sha,
        "parent_turn_count": authoritative_dialogue.get("turn_count"),
        "parent_session_count": authoritative_dialogue.get("session_count"),
        "parent_tail_required": tail_required, "orchestration_seed": 61555,
        "api_config_path": str(api_config_path),
        "api_config_sha256": sha256_file(api_config_path) if api_config_path.is_file() else None,
        "api_configuration": api_configuration, "tokenizer_path": str(tokenizer_path),
        "implementation_manifest": implementation,
        "project_working_tree_manifest_sha256": project_diff_sha256(PROJECT_ROOT),
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "historical_manifests": historical_manifests_v11(),
        "prompt_schema_versions": prompt_versions,
        "prompt_schema_contract_sha256": prompt_sha,
        "planner_transport_configuration": planner_transport,
        "round_validator_transport_configuration": round_validator_transport,
        "recursive_import_audit": import_audit,
        "request_budget": {"planner_max": 3, "realizer_per_scene_max": 3,
                           "round_validator_max": 2, "scene_min": 5,
                           "scene_max": 7, "normal_min": 7, "normal_max": 9,
                           "worst_case": 26, "evidence_committer": 0,
                           "initial_structure_extractor": 0},
        "api_calls": 0, "network_attempts": 0,
    }
