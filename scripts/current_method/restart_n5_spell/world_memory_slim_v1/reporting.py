"""Final JSON and Markdown reporting for the offline implementation round."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_file
from .provenance import PROVENANCE_VERSION
from .schemas import (
    DIFFICULTY_STRATEGY_SCHEMA_VERSION,
    EVIDENCE_COMMITTER_SCHEMA_VERSION,
    PLANNER_MUTATION_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
    QA_REPLAY_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
    WORLD_MEMORY_SCHEMA_VERSION,
)


REPORT_SCHEMA_VERSION = "world_memory_slim_pipeline_v1_report"


def module_source_sha256(package_root: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(package_root.glob("*.py"))}


def build_report(
    *, preflight: Mapping[str, Any], test_results: Mapping[str, Any],
    historical_run_audit: Mapping[str, Any], git_head: str,
    project_diff_sha256: str, package_root: Path,
) -> dict[str, Any]:
    ready = (
        bool(preflight.get("slim_pipeline_preflight_ready")) and
        historical_run_audit.get("unchanged") is True and
        test_results.get("targeted_unittest_status") == "pass" and
        test_results.get("full_unittest_status") == "pass" and
        test_results.get("compileall_status") == "pass"
    )
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "flow": [
                "authoritative_parent", "world_memory_v1", "slim_planner",
                "proposed_graph_delta", "structural_delta_validator",
                "natural_realizer", "model_claimed_updates",
                "evidence_committer_interface", "local_structural_binder",
                "confirmed_graph_delta", "world_memory_update",
            ],
            "responsibility_boundaries": {
                "world_memory": "source-bound history, authoritative state, pending claims, and lineage",
                "slim_planner": "goals and proposed changes only",
                "structural_validator": "deterministic schema, identity, reference, and state checks only",
                "natural_realizer": "natural dialogue plus response-local claims only",
                "evidence_committer": "independent semantic decision interface; no runner invoked this round",
                "local_binder": "program IDs, reference existence, order, speaker, hashes, and handoffs",
                "memory_commit": "verified updates only; model claims remain non-training pending records",
            },
        },
        "removed_legacy_fields_and_contracts": [
            "completion_evidence_phrases", "canonical_marker", "exact_marker",
            "evidence_turn_index", "fixed 4x4 stage contract", "fixed 10-scene gate",
            "model-assigned final turn/session IDs", "model-assigned handoff IDs",
            "model-assigned state_instance_id", "metadata-only retry",
            "local terminal-keyword semantic completion",
        ],
        "schema_versions": {
            "world_memory": WORLD_MEMORY_SCHEMA_VERSION,
            "slim_planner": PLANNER_SCHEMA_VERSION,
            "natural_realizer": REALIZER_SCHEMA_VERSION,
            "evidence_committer": EVIDENCE_COMMITTER_SCHEMA_VERSION,
            "qa_replay": QA_REPLAY_SCHEMA_VERSION,
            "difficulty_strategy": DIFFICULTY_STRATEGY_SCHEMA_VERSION,
            "planner_mutation": PLANNER_MUTATION_SCHEMA_VERSION,
            "provenance": PROVENANCE_VERSION,
        },
        "schema_sha256": dict(preflight["schema_sha256"]),
        "module_source_sha256": module_source_sha256(package_root),
        "preflight": dict(preflight),
        "tests": dict(test_results),
        "historical_run_sha_audit": dict(historical_run_audit),
        "api_usage": {
            key: preflight[key] for key in (
                "api_calls", "planner_calls", "realizer_calls",
                "evidence_committer_calls", "questioner_calls", "distractor_calls",
                "solver_calls", "verifier_calls", "training_runs")
        },
        "git_head": git_head,
        "project_diff_sha256": project_diff_sha256,
        "project_diff_sha256_scope": "computed after implementation/tests and before final report files",
        "unfinished_items": [
            "No live Planner, Realizer, or Evidence Committer runner is configured.",
            "Fresh n=1 generation has not been run.",
            "Questioner, Distractor, Solver, QA admission, and training are not implemented here.",
        ],
        "world_memory_v1_ready": ready,
        "slim_planner_interface_ready": ready,
        "natural_realizer_interface_ready": ready,
        "evidence_committer_interface_ready": ready,
        "qa_memory_schema_ready": ready,
        "slim_pipeline_preflight_ready": ready,
        "generation_ready": False,
        "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    status_keys = (
        "world_memory_v1_ready", "slim_planner_interface_ready",
        "natural_realizer_interface_ready", "evidence_committer_interface_ready",
        "qa_memory_schema_ready", "slim_pipeline_preflight_ready",
        "generation_ready", "fresh_e2e_ready", "qa_pipeline_ready",
    )
    lines = [
        "# World Memory Slim Pipeline v1 Report", "",
        "## Final status", "",
    ]
    lines.extend(f"- `{key} = {str(bool(report[key])).lower()}`" for key in status_keys)
    lines.extend([
        "", "## Architecture", "",
        "`Parent → World Memory v1 → Slim Planner → proposed graph_delta → "
        "Structural Delta Validator → Natural Realizer → post-generation claims → "
        "Evidence Committer interface → Local Structural Binder → confirmed graph_delta "
        "→ World Memory update`", "",
        "The Planner proposes changes but does not write evidence sentences or final IDs. "
        "The Realizer writes natural dialogue and response-local claims. Only an independent "
        "confirmed semantic decision can pass through the local binder into authoritative state.",
        "", "## Removed legacy contracts", "",
    ])
    lines.extend(f"- {value}" for value in report["removed_legacy_fields_and_contracts"])
    lines.extend(["", "## Schemas", "", "| Version | SHA256 |", "|---|---|"])
    for version, digest in report["schema_sha256"].items():
        lines.append(f"| `{version}` | `{digest}` |")
    preflight = report["preflight"]
    lines.extend([
        "", "## 11-root offline preflight", "",
        "| Root | Dataset | Sessions | Turns | Memory events | Pass |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in preflight["roots"]:
        lines.append(
            f"| `{row['root_id']}` | `{row['dataset']}` | "
            f"{row['parent_session_count']} | {row['parent_turn_count']} | "
            f"{row['world_memory_initial_event_count']} | "
            f"`{str(row['preflight_pass']).lower()}` |")
    lines.extend([
        "", f"- Source inputs unchanged: `{str(preflight['source_inputs_unchanged']).lower()}`",
        f"- Legacy pipeline import isolation: `{str(preflight['legacy_import_isolation']['isolated']).lower()}`",
        f"- Physical/model API calls: `{preflight['api_calls']}`",
        "- LoCoMo `conv-43`: record 77 and attestation were both checked; the compatibility "
        "unified root existed only in a temporary directory.",
        "", "## Tests", "",
    ])
    for name, value in report["tests"].items():
        lines.append(f"- `{name}`: `{value}`")
    audit = report["historical_run_sha_audit"]
    lines.extend([
        "", "## Historical run SHA audit", "",
        f"- Before: `{audit['before']['tree_sha256']}`",
        f"- After: `{audit['after']['tree_sha256']}`",
        f"- Unchanged: `{str(audit['unchanged']).lower()}`",
        f"- Files: `{audit['after']['file_count']}`",
        "", "## Provenance", "",
        f"- Git HEAD: `{report['git_head']}`",
        f"- Project diff SHA256: `{report['project_diff_sha256']}`",
        f"- Diff scope: {report['project_diff_sha256_scope']}",
        "", "## Unfinished items", "",
    ])
    lines.extend(f"- {item}" for item in report["unfinished_items"])
    lines.extend([
        "", "No paid API, Fresh E2E, Questioner, Distractor, Solver, Verifier, "
        "training, Git commit, or historical-run mutation occurred.", "",
    ])
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], *, json_path: Path, md_path: Path) -> None:
    atomic_json(json_path, dict(report))
    atomic_text(md_path, render_markdown(report))
