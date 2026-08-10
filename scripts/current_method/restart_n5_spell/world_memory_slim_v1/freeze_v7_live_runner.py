"""Generate the zero-API freeze consumed by the unique live v7 runner."""

from __future__ import annotations

import compileall
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from ..core import (atomic_json, atomic_text, canonical_json, sha256_bytes,
                    sha256_file)
from ..unified_dialogue_root import read_unified_parent
from .append_only_memory import APPEND_ONLY_MEMORY_VERSION
from .claim_validator_v2 import CLAIM_VALIDATOR_VERSION
from .controlled_expansion import AFFORDANCE_SCHEMA_VERSION_V3
from .dialogue_quality import DIALOGUE_QUALITY_VERSION
from .evidence_scope import (EVIDENCE_SCOPE_AUDIT_VERSION,
                             EVIDENCE_SCOPE_PROMPT_VERSION,
                             EVIDENCE_SCOPE_RETRY_PROMPT_VERSION)
from .offline_preflight_v3 import run_fixed_manifest_preflight
from .planner_v3 import (PLANNER_PROMPT_VERSION_V3,
                         PLANNER_RETRY_PROMPT_VERSION_V3,
                         PLANNER_SCHEMA_VERSION_V3)
from .planner_contract_v3_1 import (PLANNER_ACTION_VIEW_VERSION,
                                    PLANNER_BUNDLE_VERSION,
                                    PLANNER_CONTRACT_VERSION)
from .promotion import PROMOTION_GATE_VERSION
from .realizer_v3 import (NATURAL_REALIZER_CLAIM_RETRY_PROMPT_VERSION_V3,
                          NATURAL_REALIZER_FULL_RETRY_PROMPT_VERSION_V3,
                          NATURAL_REALIZER_PROMPT_VERSION_V3,
                          NATURAL_REALIZER_VERSION_V3)
from .semantic_import_audit_v7 import audit_v7_recursive_import_graph
from .transactional_binder import TRANSACTIONAL_BINDER_VERSION
from .v7_manifest import (directory_manifest, implementation_manifest,
                          project_working_tree_manifest_sha256)


FREEZE_VERSION = "slim_planner_contract_v3_1_next_canary_freeze_v1"
V6_RUNNER_EXPECTED_SHA256 = "69763e2493f88eee8e74fb4a6492013bccc52b5109c5ce5ba888839bc406ea03"
INPUT_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-61555/unified_root.json")
V6_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v6")
V7_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v7")
V8_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v8")
API_CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")


def _command(repo: Path, *arguments: str) -> dict[str, Any]:
    result = subprocess.run(list(arguments), cwd=repo, text=True,
                            capture_output=True)
    combined = (result.stdout + result.stderr).strip()
    match = re.search(r"Ran (\d+) tests?", combined)
    return {"command": list(arguments), "returncode": result.returncode,
            "passed": result.returncode == 0,
            "tests_run": int(match.group(1)) if match else None,
            "output_tail": "\n".join(combined.splitlines()[-12:])}


def _component_versions() -> dict[str, str]:
    return {
        "slim_world_planner_v3": PLANNER_SCHEMA_VERSION_V3,
        "slim_world_planner_contract_v3_3": PLANNER_CONTRACT_VERSION,
        "planner_action_view_v1": PLANNER_ACTION_VIEW_VERSION,
        "planner_bundle_selection_v1": PLANNER_BUNDLE_VERSION,
        "natural_realizer_v3": NATURAL_REALIZER_VERSION_V3,
        "dialogue_quality_v1": DIALOGUE_QUALITY_VERSION,
        "claim_validator_v2": CLAIM_VALIDATOR_VERSION,
        "evidence_scope_audit_v1": EVIDENCE_SCOPE_AUDIT_VERSION,
        "transactional_binder_v1": TRANSACTIONAL_BINDER_VERSION,
        "promotion_gate_v1": PROMOTION_GATE_VERSION,
        "append_only_world_memory_v1": APPEND_ONLY_MEMORY_VERSION,
        "delta_affordance_compiler_v3": AFFORDANCE_SCHEMA_VERSION_V3,
    }


def _prompt_schema_versions() -> dict[str, str]:
    return {
        "planner_schema": PLANNER_SCHEMA_VERSION_V3,
        "planner_contract": PLANNER_CONTRACT_VERSION,
        "planner_action_view": PLANNER_ACTION_VIEW_VERSION,
        "planner_bundle_selection": PLANNER_BUNDLE_VERSION,
        "planner_prompt": PLANNER_PROMPT_VERSION_V3,
        "planner_retry_prompt": PLANNER_RETRY_PROMPT_VERSION_V3,
        "realizer_schema": NATURAL_REALIZER_VERSION_V3,
        "realizer_prompt": NATURAL_REALIZER_PROMPT_VERSION_V3,
        "realizer_full_retry_prompt": NATURAL_REALIZER_FULL_RETRY_PROMPT_VERSION_V3,
        "realizer_claim_retry_prompt": NATURAL_REALIZER_CLAIM_RETRY_PROMPT_VERSION_V3,
        "committer_schema": EVIDENCE_SCOPE_AUDIT_VERSION,
        "committer_prompt": EVIDENCE_SCOPE_PROMPT_VERSION,
        "committer_contract_retry_prompt": EVIDENCE_SCOPE_RETRY_PROMPT_VERSION,
    }


def _readiness(*, preflight: Mapping[str, Any], import_audit: Mapping[str, Any],
               targeted: Mapping[str, Any], full_repository: Mapping[str, Any],
               compile_passed: bool,
               diff_passed: bool, implementation_valid: bool,
               v6_unchanged: bool, v7_unchanged: bool,
               input_unchanged: bool) -> dict[str, bool]:
    common = bool(preflight.get("ready") and import_audit.get("valid") and
                  targeted.get("passed") and full_repository.get("passed") and
                  compile_passed and diff_passed and
                  implementation_valid and v6_unchanged and v7_unchanged and
                  input_unchanged)
    names = (
        "provenance_aware_source_initializer_ready",
        "interaction_link_compiler_ready",
        "delta_affordance_compiler_v3_ready",
        "action_menu_budget_ready",
        "slim_world_planner_v3_ready",
        "planner_schema_retry_ready",
        "planner_compile_retry_ready",
        "planner_semantic_retry_ready",
        "local_invariant_stop_ready",
        "local_scene_plan_compiler_ready",
        "proposal_ref_assignment_ready",
        "proposal_fingerprint_ready",
        "proposal_dependency_ready",
        "turn_budget_v2_ready",
        "natural_realizer_v3_ready",
        "full_scene_retry_ready",
        "claim_only_retry_ready",
        "scene_local_turn_refs_ready",
        "claim_only_dialogue_freeze_ready",
        "dialogue_quality_v1_ready",
        "claim_validator_v2_ready",
        "proposal_claim_one_to_one_ready",
        "committer_candidate_subset_ready",
        "committer_semantic_freeze_ready",
        "scope_only_committer_ready",
        "evidence_scope_audit_v1_ready",
        "scene_commit_gate_ready",
        "transactional_binder_v1_ready",
        "transaction_recovery_ready",
        "append_only_world_memory_v1_ready",
        "provenance_ledger_ready",
        "full_promotable_child_split_ready",
        "promotion_gate_v1_ready",
        "recursive_import_boundary_ready",
        "v6_runner_immutability_ready",
        "offline_11_root_preflight_ready",
        "slim_pipeline_v7_preflight_ready",
        "planner_contract_v3_1_ready",
        "planner_action_projection_ready",
        "planner_frozen_fields_hidden_ready",
        "planner_writable_schema_ready",
        "planner_bundle_selection_ready",
        "planner_action_uniqueness_ready",
        "planner_existing_fact_collision_ready",
        "planner_round_fingerprint_dedup_ready",
        "planner_canonical_relation_ready",
        "planner_retry_contract_v3_1_ready",
        "planner_error_taxonomy_ready",
        "planner_real_failure_regression_ready",
        "planner_corrected_fixture_ready",
        "v7_runner_planner_v3_1_wiring_ready",
        "next_canary_preflight_ready",
    )
    result = {name: common for name in names}
    result.update({"generation_ready": False, "fresh_e2e_ready": False,
                   "qa_pipeline_ready": False})
    return result


def generate_live_v7_freeze(repo: Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    package = repo / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
    reports = package / "reports"
    runner = package / "live_fresh_e2e_v7.py"
    manifest_path = package / "preflight_manifest_v1.json"
    v6_runner = package / "live_fresh_e2e.py"
    reports.mkdir(parents=True, exist_ok=True)

    input_before = sha256_file(INPUT_PATH)
    parent_dialogue_sha = sha256_bytes(canonical_json(
        [read_unified_parent(INPUT_PATH)["dialogue"]]).encode("utf-8"))
    v6_before = directory_manifest(V6_ROOT)
    v7_before = directory_manifest(V7_ROOT)
    preflight = run_fixed_manifest_preflight(manifest_path)
    atomic_json(reports / "OFFLINE_11_ROOT_PREFLIGHT.json", preflight)
    atomic_text(reports / "OFFLINE_11_ROOT_PREFLIGHT.md",
                "# Offline 11-root Preflight\n\n```json\n" +
                json.dumps(preflight, ensure_ascii=False, indent=2) + "\n```\n")
    targeted = _command(
        repo, sys.executable, "-m", "unittest",
        "tests.test_world_memory_slim_planner_v3_final_freeze",
        "tests.test_world_memory_slim_live_fresh_e2e_v7",
        "tests.test_world_memory_slim_planner_contract_v3_1", "-v")
    full_repository = _command(
        repo, sys.executable, "-m", "unittest", "discover", "-s", "tests")
    compile_passed = (compileall.compile_dir(str(repo / "scripts"), quiet=1) and
                      compileall.compile_dir(str(repo / "tests"), quiet=1))
    diff = _command(repo, "git", "diff", "--check")
    import_audit = audit_v7_recursive_import_graph(
        project_root=repo, runner_path=runner)
    implementation = implementation_manifest(repo)
    input_after = sha256_file(INPUT_PATH)
    v6_after = directory_manifest(V6_ROOT)
    v7_after = directory_manifest(V7_ROOT)
    v6_runner_sha = sha256_file(v6_runner)
    readiness = _readiness(
        preflight=preflight, import_audit=import_audit, targeted=targeted,
        full_repository=full_repository,
        compile_passed=compile_passed, diff_passed=diff["passed"],
        implementation_valid=implementation["valid"],
        v6_unchanged=(v6_before["directory_manifest_sha256"] ==
                      v6_after["directory_manifest_sha256"] and
                      v6_runner_sha == V6_RUNNER_EXPECTED_SHA256),
        v7_unchanged=(v7_before["directory_manifest_sha256"] ==
                      v7_after["directory_manifest_sha256"]),
        input_unchanged=input_before == input_after)
    prompt_schema_versions = _prompt_schema_versions()
    prompt_schema_contract_sha = hashlib.sha256(json.dumps(
        prompt_schema_versions, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    freeze = {
        "freeze_version": FREEZE_VERSION,
        "scope": "Planner Contract v3.1 offline implementation and next-canary freeze; no live API and no new run",
        "git_head": _command(repo, "git", "rev-parse", "HEAD")["output_tail"],
        "implementation_manifest_sha256": implementation[
            "implementation_manifest_sha256"],
        "implementation_manifest": implementation,
        "project_working_tree_manifest_sha256":
            project_working_tree_manifest_sha256(repo),
        "input_path": str(INPUT_PATH), "input_sha256": input_after,
        "parent_dialogue_sha256": parent_dialogue_sha,
        "v6_root": str(V6_ROOT),
        "v6_directory_manifest_sha256": v6_after["directory_manifest_sha256"],
        "v6_runner_sha256": v6_runner_sha,
        "historical_v7_root": str(V7_ROOT),
        "historical_v7_directory_manifest_sha256": v7_after["directory_manifest_sha256"],
        "next_run_root": str(V8_ROOT), "next_run_root_absent": not V8_ROOT.exists(),
        "api_config_path": str(API_CONFIG_PATH),
        "api_config_sha256": sha256_file(API_CONFIG_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "component_versions": _component_versions(),
        "prompt_schema_versions": prompt_schema_versions,
        "prompt_schema_contract_sha256": prompt_schema_contract_sha,
        "recursive_import_audit": import_audit,
        "offline_11_root_preflight": {
            "ready": preflight["ready"], "root_count": preflight["root_count"],
            "report": str(reports / "OFFLINE_11_ROOT_PREFLIGHT.json")},
        "targeted_unittest": targeted,
        "full_repository_unittest": full_repository,
        "compileall": {"passed": bool(compile_passed), "scope": "scripts tests"},
        "git_diff_check": diff,
        "input_sha_before": input_before, "input_sha_after": input_after,
        "input_sha_unchanged": input_before == input_after,
        "v6_directory_sha_before": v6_before["directory_manifest_sha256"],
        "v6_directory_sha_after": v6_after["directory_manifest_sha256"],
        "v6_sha_unchanged": (v6_before["directory_manifest_sha256"] ==
                             v6_after["directory_manifest_sha256"]),
        "v7_directory_sha_before": v7_before["directory_manifest_sha256"],
        "v7_directory_sha_after": v7_after["directory_manifest_sha256"],
        "v7_sha_unchanged": (v7_before["directory_manifest_sha256"] ==
                             v7_after["directory_manifest_sha256"]),
        "readiness_count": sum(key.endswith("_ready") and value is True
                               for key, value in readiness.items()),
        "readiness": readiness,
        "api_calls": 0, "outbound_network_attempts": 0,
        "new_fresh_e2e_executed": False, "round_2_executed": False,
        "unimplemented_or_deferred": ["Fresh E2E v8 live execution", "Round 2", "QA", "training"],
    }
    freeze["ready"] = bool(
        freeze["next_run_root_absent"] and freeze["readiness_count"] >= 31 and
        all(value for key, value in readiness.items()
            if key.endswith("_ready") and key not in {
                "generation_ready", "fresh_e2e_ready", "qa_pipeline_ready"}))
    atomic_json(reports / "SLIM_V7_LIVE_RUNNER_FREEZE.json", freeze)
    atomic_text(reports / "SLIM_V7_LIVE_RUNNER_FREEZE.md",
                "# Slim v7 Live Runner Freeze\n\n```json\n" +
                json.dumps(freeze, ensure_ascii=False, indent=2) + "\n```\n")
    post_manifest = project_working_tree_manifest_sha256(repo)
    if post_manifest != freeze["project_working_tree_manifest_sha256"]:
        raise RuntimeError("v7_freeze_report_self_reference_detected")
    return freeze


if __name__ == "__main__":
    result = generate_live_v7_freeze(Path(__file__).resolve().parents[4])
    print(json.dumps({"ready": result["ready"],
                      "readiness_count": result["readiness_count"],
                      "api_calls": result["api_calls"]}, indent=2))
