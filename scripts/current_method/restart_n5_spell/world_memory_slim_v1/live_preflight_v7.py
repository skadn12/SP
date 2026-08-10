"""Hard pre-API boundary for the unique Slim Fresh E2E v7 canary."""

from __future__ import annotations

import compileall
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..core import (atomic_json, atomic_text, canonical_json, sha256_bytes,
                    sha256_file)
from ..live_backends import load_challenger_config
from ..unified_dialogue_root import (read_unified_parent,
                                     validate_unified_dialogue_root)
from .append_only_memory import APPEND_ONLY_MEMORY_VERSION
from .claim_validator_v2 import CLAIM_VALIDATOR_VERSION
from .controlled_expansion import AFFORDANCE_SCHEMA_VERSION_V3
from .dialogue_quality import DIALOGUE_QUALITY_VERSION
from .evidence_scope import (EVIDENCE_SCOPE_AUDIT_VERSION,
                             EVIDENCE_SCOPE_PROMPT_VERSION,
                             EVIDENCE_SCOPE_RETRY_PROMPT_VERSION)
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


LIVE_PREFLIGHT_VERSION = "slim_fresh_e2e_v7_pre_api_preflight_v2"


def run_live_preflight_v7(
    *, project_root: Path, package_root: Path, runner_path: Path,
    input_path: Path, historical_v6_root: Path, run_root: Path,
    freeze_report_path: Path, api_config_path: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    project_root, package_root = Path(project_root).resolve(), Path(package_root).resolve()
    runner_path, input_path = Path(runner_path).resolve(), Path(input_path).resolve()
    historical_v6_root, run_root = Path(historical_v6_root).resolve(), Path(run_root).resolve()
    freeze_report_path, api_config_path = Path(freeze_report_path).resolve(), Path(api_config_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    errors: list[str] = []
    freeze: Mapping[str, Any] = {}
    if freeze_report_path.is_file():
        freeze = json.loads(freeze_report_path.read_text(encoding="utf-8"))
    else:
        errors.append("v7_live_freeze_report_missing")
    root = json.loads(input_path.read_text(encoding="utf-8")) if input_path.is_file() else {}
    validation = validate_unified_dialogue_root(root) if root else {"valid": False, "errors": ["input_missing"]}
    parent_dialogue_sha = None
    if validation["valid"]:
        parent_value = read_unified_parent(input_path)
        parent_dialogue_sha = sha256_bytes(canonical_json(
            [parent_value["dialogue"]]).encode("utf-8"))
    if not validation["valid"]:
        errors.extend(f"authoritative_input:{value}" for value in validation["errors"])
    current_impl = implementation_manifest(project_root)
    current_working_tree = project_working_tree_manifest_sha256(project_root)
    v6_manifest = directory_manifest(historical_v6_root) if historical_v6_root.is_dir() else {}
    current_head = _git(project_root, "rev-parse", "HEAD")
    api_configuration: dict[str, Any] | None = None
    try:
        api = load_challenger_config(api_config_path)
        api_configuration = {
            "model": api.model, "base_url": api.base_url,
            "temperature": api.temperature, "timeout_seconds": api.timeout_seconds,
            "enable_thinking": api.enable_thinking, "api_key_recorded": False,
        }
    except Exception as error:
        errors.append(f"api_configuration_invalid:{type(error).__name__}:{error}")
    tokenizer_valid = False
    if tokenizer_path.exists():
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            tokenizer_valid = True
        except Exception as error:
            errors.append(f"local_tokenizer_invalid:{type(error).__name__}:{error}")
    import_audit = audit_v7_recursive_import_graph(
        project_root=project_root, runner_path=runner_path)
    component_versions = {
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
    prompt_schema_versions = {
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
    prompt_schema_sha = hashlib.sha256(json.dumps(
        prompt_schema_versions, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    visited_names = {Path(value).name for value in import_audit.get("visited_files") or []}
    required_entry_files = {
        "planner_v3.py", "planner_contract_v3_1.py", "realizer_v3.py", "dialogue_quality.py",
        "claim_validator_v2.py", "evidence_scope.py", "transactional_binder.py",
        "promotion.py", "append_only_memory.py",
    }
    expected_versions = freeze.get("component_versions") or {}
    checks = {
        "run_root_absent": not run_root.exists(),
        "run_root_parent_valid": run_root.parent.name == "restart-n5-runs",
        "authoritative_input_valid": validation["valid"],
        "input_sha_matches_freeze": bool(freeze) and sha256_file(input_path) == freeze.get("input_sha256"),
        "parent_dialogue_sha_matches_freeze": parent_dialogue_sha ==
            freeze.get("parent_dialogue_sha256"),
        "v6_manifest_matches_freeze": bool(v6_manifest) and
            v6_manifest.get("directory_manifest_sha256") == freeze.get("v6_directory_manifest_sha256"),
        "git_head_matches_freeze": current_head == freeze.get("git_head"),
        "implementation_manifest_valid": current_impl["valid"],
        "implementation_manifest_matches_freeze": current_impl["implementation_manifest_sha256"] ==
            freeze.get("implementation_manifest_sha256"),
        "project_working_tree_matches_freeze": current_working_tree ==
            freeze.get("project_working_tree_manifest_sha256"),
        "component_versions_match_freeze": component_versions == expected_versions,
        "prompt_schema_versions_match_freeze": prompt_schema_versions ==
            freeze.get("prompt_schema_versions"),
        "prompt_schema_contract_sha_matches_freeze": prompt_schema_sha ==
            freeze.get("prompt_schema_contract_sha256"),
        "required_entry_modules_reachable": required_entry_files.issubset(visited_names),
        "recursive_import_graph_clean": import_audit["valid"],
        "legacy_validator_import_count_zero": import_audit["forbidden_import_count"] == 0,
        "compileall_passed": compileall.compile_dir(str(package_root), quiet=1) and
            compileall.compile_file(str(project_root / "tests/test_world_memory_slim_live_fresh_e2e_v7.py"), quiet=1) and
            compileall.compile_file(str(project_root / "tests/test_world_memory_slim_planner_contract_v3_1.py"), quiet=1),
        "git_diff_check_passed": _git_check(project_root, "diff", "--check"),
        "all_frozen_readiness_true": _readiness_true(freeze.get("readiness") or {}),
        "api_config_exists": api_config_path.is_file(),
        "api_config_sha_matches_freeze": api_config_path.is_file() and
            sha256_file(api_config_path) == freeze.get("api_config_sha256"),
        "api_configuration_valid": api_configuration is not None,
        "local_tokenizer_valid": tokenizer_valid,
        "tokenizer_path_matches_freeze": str(tokenizer_path) ==
            freeze.get("tokenizer_path"),
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        "preflight_version": LIVE_PREFLIGHT_VERSION,
        "valid": not errors, "errors": list(dict.fromkeys(errors)), "checks": checks,
        "project_root": str(project_root), "runner_path": str(runner_path),
        "run_root": str(run_root), "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "parent_dialogue_sha256": parent_dialogue_sha,
        "v6_directory_manifest": v6_manifest,
        "git_head": current_head, "implementation_manifest": current_impl,
        "project_working_tree_manifest_sha256": current_working_tree,
        "component_versions": component_versions, "recursive_import_audit": import_audit,
        "prompt_schema_versions": prompt_schema_versions,
        "prompt_schema_contract_sha256": prompt_schema_sha,
        "api_config_path": str(api_config_path),
        "api_config_sha256": sha256_file(api_config_path) if api_config_path.is_file() else None,
        "api_configuration": api_configuration, "tokenizer_path": str(tokenizer_path),
        "api_calls": 0,
    }


def write_preflight_failure_report(package_root: Path, result: Mapping[str, Any]) -> tuple[Path, Path]:
    reports = Path(package_root) / "reports"
    json_path = reports / "PRE_API_PREFLIGHT_FAILED_V7_CANARY.json"
    md_path = reports / "PRE_API_PREFLIGHT_FAILED_V7_CANARY.md"
    atomic_json(json_path, dict(result))
    atomic_text(md_path, "\n".join([
        "# Slim Fresh E2E v7 — PRE_API_PREFLIGHT_FAILED", "",
        "No run root was created and no API request was sent.", "",
        "## Errors", "", *(f"- `{value}`" for value in result.get("errors") or []), "",
    ]))
    return json_path, md_path


def _readiness_true(readiness: Mapping[str, Any]) -> bool:
    required = [value for key, value in readiness.items()
                if key.endswith("_ready") and key not in {
                    "generation_ready", "fresh_e2e_ready", "qa_pipeline_ready"}]
    return len(required) >= 31 and all(value is True for value in required)


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=project_root,
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _git_check(project_root: Path, *args: str) -> bool:
    return subprocess.run(["git", *args], cwd=project_root,
                          text=True, capture_output=True).returncode == 0
