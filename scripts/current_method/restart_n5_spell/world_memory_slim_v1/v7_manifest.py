"""Stable implementation and directory manifests for Slim live v7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..core import sha256_file
from .provenance import project_diff_sha256_excluding


V7_IMPLEMENTATION_MANIFEST_VERSION = "slim_v7_implementation_manifest_v1"
V7_GENERATED_FREEZE_RELATIVES = frozenset({
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/SLIM_V7_LIVE_RUNNER_FREEZE.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/SLIM_V7_LIVE_RUNNER_FREEZE.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PRE_API_PREFLIGHT_FAILED_V7_CANARY.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PRE_API_PREFLIGHT_FAILED_V7_CANARY.md",
    "WORLD_MEMORY_PLANNER_CONTRACT_V3_1_FREEZE_REPORT.json",
    "WORLD_MEMORY_PLANNER_CONTRACT_V3_1_FREEZE_REPORT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_ACTION_VIEW_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_ACTION_VIEW_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V3_1_REGRESSION_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V3_1_REGRESSION_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_PROPOSAL_NOVELTY_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_PROPOSAL_NOVELTY_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V3_1_PRE_API_PREFLIGHT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V3_1_PRE_API_PREFLIGHT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/ARTIFACT_MANIFEST.json",
    "WORLD_MEMORY_PLANNER_BUNDLE_SELECTION_V3_3_FREEZE_REPORT.json",
    "WORLD_MEMORY_PLANNER_BUNDLE_SELECTION_V3_3_FREEZE_REPORT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_BUNDLE_SELECTION_SCHEMA_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_BUNDLE_SELECTION_SCHEMA_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V9_BUNDLE_REGRESSION_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V9_BUNDLE_REGRESSION_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_BUNDLE_EXPANSION_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_BUNDLE_EXPANSION_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V3_3_PRE_API_PREFLIGHT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_V3_3_PRE_API_PREFLIGHT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/SLIM_V10_CANARY_FREEZE.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/SLIM_V10_CANARY_FREEZE.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PLANNER_BUNDLE_SELECTION_V3_3_ARTIFACT_MANIFEST.json",
})
V7_IMPLEMENTATION_RELATIVE_PATHS = (
    "scripts/current_method/restart_n5_spell/backends.py",
    "scripts/current_method/restart_n5_spell/core.py",
    "scripts/current_method/restart_n5_spell/input_adapter.py",
    "scripts/current_method/restart_n5_spell/ledger.py",
    "scripts/current_method/restart_n5_spell/live_backends.py",
    "scripts/current_method/restart_n5_spell/unified_dialogue_root.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/__init__.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/append_only_memory.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/claim_validator_v2.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/context_compiler.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/controlled_expansion.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/dialogue_quality.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/evidence_scope.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/interaction_links.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_fresh_e2e_v7.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_preflight_v7.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_role_runtime.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/operation_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/offline_preflight_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_contract_v3_1.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_legacy_bundle_audit.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_bundle_selection_v3_3_reports.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/promotion.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/provenance.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/provenance_ledger.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/realizer_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/relation_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_plan_validation.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/scene_local_refs.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/semantic_import_audit_v7.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/source_memory_initializer_v1.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/transactional_binder.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/turn_budget.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/v7_manifest.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/freeze_v7_live_runner.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_contract_v3_1_reports.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/preflight_manifest_v1.json",
    "tests/test_world_memory_slim_live_fresh_e2e_v7.py",
    "tests/test_world_memory_slim_planner_v3_final_freeze.py",
    "tests/test_world_memory_slim_planner_contract_v3_1.py",
    "tests/test_world_memory_slim_planner_bundle_selection_v3_3.py",
)


def project_working_tree_manifest_sha256(project_root: Path) -> str:
    """Hash the complete working tree while excluding self-referential reports."""
    return project_diff_sha256_excluding(
        Path(project_root).resolve(), V7_GENERATED_FREEZE_RELATIVES)


def implementation_manifest(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    missing = [relative for relative in V7_IMPLEMENTATION_RELATIVE_PATHS
               if not (project_root / relative).is_file()]
    entries = [{"relative_path": relative,
                "file_sha256": sha256_file(project_root / relative)}
               for relative in sorted(V7_IMPLEMENTATION_RELATIVE_PATHS)
               if (project_root / relative).is_file()]
    material = json.dumps(entries, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    return {"manifest_version": V7_IMPLEMENTATION_MANIFEST_VERSION,
            "files": entries, "missing_files": missing,
            "implementation_manifest_sha256": hashlib.sha256(material).hexdigest(),
            "valid": not missing}


def directory_manifest(
    directory: Path, *, excluded_relatives: Iterable[str] = (),
) -> dict[str, Any]:
    directory = Path(directory).resolve()
    excluded = {Path(value).as_posix() for value in excluded_relatives}
    entries = [{"relative_path": path.relative_to(directory).as_posix(),
                "file_sha256": sha256_file(path)}
               for path in sorted(candidate for candidate in directory.rglob("*")
                                  if candidate.is_file() and
                                  candidate.relative_to(directory).as_posix() not in excluded)]
    material = b"".join(row["relative_path"].encode("utf-8") + b"\0" +
                        row["file_sha256"].encode("ascii") + b"\0" for row in entries)
    return {"directory": str(directory), "excluded_relatives": sorted(excluded),
            "file_count": len(entries), "files": entries,
            "directory_manifest_sha256": hashlib.sha256(material).hexdigest(),
            "scope": "sorted(relative_path NUL file_sha256 NUL)"}
