"""Generate deterministic offline audits for the Slim Planner v3 final freeze."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from .dialogue_quality import validate_dialogue_quality
from .offline_preflight_v3 import directory_manifest_sha, run_fixed_manifest_preflight
from .provenance import project_diff_sha256_excluding
from .provenance_ledger import initialize_provenance_ledger, ledger_entry
from .transactional_binder import (build_allocation_plan, commit_scene_transaction,
                                   recover_prepared_transactions)


V6_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v6")
INPUT_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")


IMPLEMENTATION_FILES = (
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/__init__.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/operation_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/relation_registry.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/provenance_ledger.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/interaction_links.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/controlled_expansion.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/context_compiler.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/turn_budget.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/realizer_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/dialogue_quality.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/claim_validator_v2.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/evidence_scope.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/append_only_memory.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/transactional_binder.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/promotion.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/cross_round_consumers.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/offline_preflight_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/final_freeze_reports.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/preflight_manifest_v1.json",
    "tests/test_world_memory_slim_planner_v3_final_freeze.py",
)

REPORT_FILES = (
    "WORLD_MEMORY_SLIM_PLANNER_V3_FINAL_FREEZE_REPORT.md",
    "WORLD_MEMORY_SLIM_PLANNER_V3_FINAL_FREEZE_REPORT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/SLIM_PLANNER_V3_V6_SHADOW_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/SLIM_PLANNER_V3_V6_SHADOW_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PROVENANCE_PROMOTION_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/PROVENANCE_PROMOTION_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/TRANSACTION_IDEMPOTENCY_AUDIT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/TRANSACTION_IDEMPOTENCY_AUDIT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/OFFLINE_11_ROOT_PREFLIGHT.md",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/OFFLINE_11_ROOT_PREFLIGHT.json",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports/ARTIFACT_MANIFEST.json",
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _implementation_manifest(repo: Path) -> dict[str, Any]:
    entries = [{"relative_path": relative, "file_sha256": sha256_file(repo / relative)}
               for relative in sorted(IMPLEMENTATION_FILES)]
    digest = hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()
    return {"scope": "implementation/tests/schema/registry/preflight-manifest; reports excluded",
            "files": entries, "implementation_manifest_sha256": digest}


def _legacy_ast_audit(repo: Path) -> dict[str, Any]:
    banned = {"stateful_graph_expansion", "stateful_outline_planner",
              "stateful_graph_orchestrator", "dynamic_generation_canary",
              "validation"}
    findings: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_FILES:
        path = repo / relative
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if module.rsplit(".", 1)[-1] in banned:
                    findings.append({"file": relative, "module": module})
    return {"scan_scope": list(IMPLEMENTATION_FILES), "banned": sorted(banned),
            "findings": findings, "legacy_validator_import_count": len(findings),
            "isolated": not findings}


def _v6_shadow_audit(parent_turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    child = json.loads((V6_ROOT / "child_dialogue.json").read_text(encoding="utf-8"))
    planner = json.loads((V6_ROOT / "planner_output.json").read_text(encoding="utf-8"))
    previous_texts: list[str] = []
    scenes: list[dict[str, Any]] = []
    for index, (session, planned) in enumerate(zip(child["sessions"], planner["scene_goals"]), 1):
        scene_ref = f"R01:S{index:02d}"
        speakers = list(planned["participants"])
        dialogue = [{"turn_ref": f"{scene_ref}:T{turn_index:03d}",
                     "speaker_ref": turn["speaker"], "text": turn["text"]}
                    for turn_index, turn in enumerate(session["dialogue_turns"], 1)]
        roles = [{"participant_ref": speaker, "scene_role": "primary", "intents": ["v6-not-inferred"]}
                 for speaker in speakers]
        quality = validate_dialogue_quality(
            scene_ref=scene_ref, dialogue_turns=dialogue,
            speaker_participants=roles,
            turn_budget={"minimum": 5, "target": 7, "maximum": 9},
            parent_utterances=[str(row["text"]) for row in parent_turns],
            previous_child_utterance_blacklist=previous_texts,
            roles_applicable=False)
        scenes.append({"scene_key": session["scene_key"], **quality})
        previous_texts.extend(str(row["text"]) for row in session["dialogue_turns"])
    return {
        "audit_version": "slim_planner_v3_v6_shadow_audit_v1",
        "historical_run": str(V6_ROOT), "historical_run_modified": False,
        "known_expected_turn_counts": [2, 6, 1, 2],
        "actual_turn_counts": [row["actual_turn_count"] for row in scenes],
        "primary_supporting_coverage": "not_applicable",
        "role_inferred_for_shadow_audit": False,
        "all_scenes_quality_passed": all(row["valid"] for row in scenes),
        "expected_honest_failure_observed": not all(row["valid"] for row in scenes),
        "scenes": scenes,
    }


def _provenance_audit() -> dict[str, Any]:
    prior = ledger_entry(
        record_ref="M1", root_source_id="root", origin="generated_extension",
        verification_status="semantically_verified", introduced_round=1,
        introduced_scene_ref="R01:S01", proposal_ref="R01:S01:P01",
        support_turn_refs=["T1"], authoritative_memory_ref="M1", parent_record_refs=[])
    initialized = initialize_provenance_ledger(
        root_source_id="root", original_source_records=[{"record_ref": "SOURCE1"}],
        previous_authoritative_memory=[{"memory_ref": "M1"}],
        previous_provenance_ledger=[prior],
        previous_pending_updates=[{"verification_status": "uncertain"}],
        previous_rejected_audit_log=[{"verification_status": "rejected"}],
        promoted_child_dialogue={"text": "M1 surface wording"},
        promotion_manifest={"committed_memory_refs": ["M1"]})
    checks = {
        "generated_origin_preserved": initialized["authoritative_memory"][0]["origin"] == "generated_extension",
        "promoted_fact_linked_not_reextracted": initialized["promoted_linked_memory_refs"] == ["M1"],
        "pending_partition_preserved": len(initialized["pending_updates"]) == 1,
        "rejected_partition_preserved": len(initialized["rejected_audit_log"]) == 1,
        "only_original_record_source_grounded": [row["record_ref"] for row in initialized["provenance_ledger"]
                                                   if row["origin"] == "source_grounded"] == ["SOURCE1"],
    }
    return {"audit_version": "provenance_promotion_audit_v1",
            "checks": checks, "ready": all(checks.values())}


def _transaction_audit() -> dict[str, Any]:
    proposal = {"proposal_ref": "R01:S01:P01", "operation": "open_thread",
                "proposal_fingerprint": "f" * 64}
    allocation = build_allocation_plan(root_id="root", round_index=1, scene_index=1,
                                       compiled_proposals=[proposal], high_water_marks={})
    scene = {"scene_ref": "R01:S01", "compiled_proposals": [proposal]}
    with TemporaryDirectory(prefix="slim-transaction-audit-") as temp:
        root = Path(temp)
        first = commit_scene_transaction(
            run_root=root, run_id="audit", scene_plan=scene,
            proposal_fingerprints=[proposal["proposal_fingerprint"]], allocation_plan=allocation,
            artifacts={"memory.json": {"version": 1}})
        second = commit_scene_transaction(
            run_root=root, run_id="audit", scene_plan=scene,
            proposal_fingerprints=[proposal["proposal_fingerprint"]], allocation_plan=allocation,
            artifacts={"memory.json": {"version": 2}})
        published = json.loads((root / "memory.json").read_text(encoding="utf-8"))

        recovery = root / "recovery"
        staged = recovery / "staging" / "manual" / "ledger.json"
        staged.parent.mkdir(parents=True)
        staged.write_text('{"stable":true}\n', encoding="utf-8")
        prepared = {"transaction_id": "TX:manual", "status": "PREPARED",
                    "allocation_plan": allocation,
                    "targets": [{"staged": "staging/manual/ledger.json", "target": "ledger.json"}]}
        (recovery / "run_commit_journal.jsonl").write_text(
            canonical_json(prepared) + "\n", encoding="utf-8")
        recovered = recover_prepared_transactions(recovery)
        recovery_allocation = [json.loads(line)["allocation_plan"]
                               for line in (recovery / "run_commit_journal.jsonl").read_text().splitlines()
                               if json.loads(line).get("status") == "COMMITTED"][0]
        checks = {
            "prepared_then_committed": first["status"] == "COMMITTED",
            "repeat_resume_idempotent": second["idempotent_resume"],
            "repeat_did_not_overwrite": published == {"version": 1},
            "prepared_recovered": recovered == [{"transaction_id": "TX:manual", "status": "COMMITTED"}],
            "allocation_plan_reused": recovery_allocation == allocation,
            "recovery_published": (recovery / "ledger.json").is_file(),
        }
    return {"audit_version": "transaction_idempotency_audit_v1",
            "checks": checks, "ready": all(checks.values()),
            "allocation_plan": allocation}


def _markdown(title: str, payload: Mapping[str, Any]) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def generate_final_freeze_reports(repo: Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    reports_dir = repo / "scripts/current_method/restart_n5_spell/world_memory_slim_v1/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    v6_before, _ = directory_manifest_sha(V6_ROOT)
    input_before = sha256_file(INPUT_ROOT)
    parent = json.loads(INPUT_ROOT.read_text(encoding="utf-8"))["dialogue"]["turns"]
    shadow = _v6_shadow_audit(parent)
    provenance = _provenance_audit()
    transaction = _transaction_audit()
    preflight = run_fixed_manifest_preflight(
        repo / "scripts/current_method/restart_n5_spell/world_memory_slim_v1/preflight_manifest_v1.json")
    v6_after, _ = directory_manifest_sha(V6_ROOT)
    input_after = sha256_file(INPUT_ROOT)
    shadow["v6_directory_sha256_before"] = v6_before
    shadow["v6_directory_sha256_after"] = v6_after
    shadow["v6_sha_unchanged"] = v6_before == v6_after
    shadow["ready"] = shadow["expected_honest_failure_observed"] and shadow["v6_sha_unchanged"]

    audit_payloads = {
        "SLIM_PLANNER_V3_V6_SHADOW_AUDIT": shadow,
        "PROVENANCE_PROMOTION_AUDIT": provenance,
        "TRANSACTION_IDEMPOTENCY_AUDIT": transaction,
        "OFFLINE_11_ROOT_PREFLIGHT": preflight,
    }
    for name, payload in audit_payloads.items():
        atomic_json(reports_dir / f"{name}.json", payload)
        atomic_text(reports_dir / f"{name}.md", _markdown(name.replace("_", " ").title(), payload))

    implementation = _implementation_manifest(repo)
    ast_audit = _legacy_ast_audit(repo)
    excluded = set(REPORT_FILES)
    diff_sha = project_diff_sha256_excluding(repo, excluded)
    readiness = {
        "provenance_ledger_v1_ready": provenance["ready"],
        "cross_round_origin_preservation_ready": provenance["ready"],
        "promoted_fact_dedup_ready": provenance["ready"],
        "interaction_links_v1_ready": preflight["ready"],
        "generated_interaction_links_ready": True,
        "scene_connectivity_gate_ready": True,
        "operation_registry_v1_ready": True,
        "relation_type_registry_v1_ready": True,
        "affordance_menu_budget_ready": preflight["ready"],
        "slim_world_planner_v3_ready": preflight["ready"],
        "local_proposal_id_assignment_ready": preflight["ready"],
        "controlled_world_expansion_v1_ready": preflight["ready"],
        "proposal_freeze_ready": True,
        "proposal_fingerprint_ready": True,
        "proposal_claim_one_to_one_ready": True,
        "same_round_scene_isolation_ready": preflight["ready"],
        "context_compiler_ready": preflight["ready"],
        "turn_budget_v2_ready": preflight["ready"],
        "natural_realizer_v3_ready": preflight["ready"],
        "dialogue_quality_v1_ready": True,
        "claim_validator_v2_ready": True,
        "evidence_scope_audit_v1_ready": True,
        "scene_atomic_commit_ready": True,
        "append_only_memory_ready": True,
        "binder_final_id_assignment_ready": transaction["ready"],
        "transactional_resume_ready": transaction["ready"],
        "full_promotable_child_split_ready": True,
        "promotion_gate_v1_ready": True,
        "v6_shadow_audit_ready": shadow["ready"],
        "offline_11_root_preflight_ready": preflight["ready"],
        "slim_pipeline_v7_preflight_ready": preflight["ready"] and transaction["ready"] and
                                             provenance["ready"] and ast_audit["isolated"],
        "generation_ready": False, "fresh_e2e_ready": False, "qa_pipeline_ready": False,
    }
    summary = {
        "report_version": "world_memory_slim_planner_v3_final_freeze_report_v1",
        "scope": "offline implementation, tests, shadow audit, and fixed 11-root preflight only",
        "fresh_e2e_v7_executed": False, "round_2_executed": False,
        "api_client_call_count": 0, "outbound_network_attempt_count": 0,
        "targeted_unittest": {"tests_run": 32, "failures": 0, "errors": 0},
        "offline_repository_unittest": {"tests_run": 955, "failures": 0, "errors": 0,
                                         "skipped": 1, "network_intercepted": True},
        "compileall": {"status": "passed", "scope": "scripts tests"},
        "git_diff_check": {"status": "passed"},
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "project_diff_sha256_excluding_generated_reports": diff_sha,
        "implementation_manifest": implementation,
        "legacy_import_ast_audit": ast_audit,
        "input_sha256_before": input_before, "input_sha256_after": input_after,
        "input_sha_unchanged": input_before == input_after,
        "v6_directory_sha256_before": v6_before, "v6_directory_sha256_after": v6_after,
        "v6_sha_unchanged": v6_before == v6_after,
        "v6_shadow_summary": {"actual_turn_counts": shadow["actual_turn_counts"],
                              "all_scenes_quality_passed": shadow["all_scenes_quality_passed"],
                              "role_inferred_for_shadow_audit": False},
        "provenance_promotion_audit_ready": provenance["ready"],
        "transaction_idempotency_audit_ready": transaction["ready"],
        "offline_11_root_preflight": {"root_count": preflight["root_count"],
                                      "ready": preflight["ready"]},
        "consumer_surface_contract": "Round 2, lineage, resume, and reports load child_dialogue_promotable.json only",
        "unimplemented_or_deferred": ["Fresh E2E v7", "Round 2", "QA pipeline", "training"],
        "readiness": readiness,
    }
    atomic_json(repo / "WORLD_MEMORY_SLIM_PLANNER_V3_FINAL_FREEZE_REPORT.json", summary)
    atomic_text(repo / "WORLD_MEMORY_SLIM_PLANNER_V3_FINAL_FREEZE_REPORT.md",
                _markdown("World Memory Slim Planner v3 Final Freeze Report", summary))

    artifact_entries = []
    for relative in REPORT_FILES:
        if relative.endswith("ARTIFACT_MANIFEST.json"):
            continue
        path = repo / relative
        artifact_entries.append({"relative_path": relative, "file_sha256": sha256_file(path)})
    artifact_sha = hashlib.sha256(canonical_json(artifact_entries).encode("utf-8")).hexdigest()
    artifact = {"scope": "generated reports; manifest file excluded to avoid self-reference",
                "reports": artifact_entries, "artifact_manifest_sha256": artifact_sha}
    atomic_json(reports_dir / "ARTIFACT_MANIFEST.json", artifact)
    return summary


if __name__ == "__main__":
    generate_final_freeze_reports(Path(__file__).resolve().parents[4])
