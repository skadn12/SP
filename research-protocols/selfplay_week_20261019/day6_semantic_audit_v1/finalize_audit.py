"""Seal the completed offline audit and add a separate navigation pointer."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from prepare_audit import HERE, PRIOR, DAY5, bind, check, read, write_new


def main():
    decision_path = HERE / "DECISION.json"
    manifest_path = HERE / "FINAL_MANIFEST.json"
    pointer_path = HERE.parent / "DAY6_SEMANTIC_LATEST.json"
    if any(path.exists() for path in (decision_path, manifest_path, pointer_path)):
        raise RuntimeError("NO_OVERWRITE_SEMANTIC_CLOSEOUT")
    independent_path = HERE / "INDEPENDENT_AUDIT.json"
    independent = read(independent_path)
    if independent.get("status") != "PASS" or independent.get("scope") != "DAY6_POSTHOC_SOURCE_SEMANTIC_REVIEW":
        raise RuntimeError("INDEPENDENT_AUDIT_PASS_REQUIRED")
    check(independent["audited_files"])
    required = [HERE / name for name in (
        "PROTOCOL_ZH.md", "prepare_audit.py", "aggregate_audit.py", "AUDIT_PACKET.json",
        "INPUT_MANIFEST.json", "SEMANTIC_AUDIT.json", "SUMMARY.json",
        "B1_FEEDBACK_CANDIDATES.json", "ADJUDICATIONS.json")]
    required += sorted((HERE / "reviews").glob("roots_*.json"))
    for path in required:
        if bind(path) not in independent["audited_files"]:
            raise RuntimeError("INDEPENDENT_AUDIT_BINDING_MISSING:" + str(path))
    check(read(HERE / "INPUT_MANIFEST.json")["files"])
    prior_manifest = read(PRIOR / "FINAL_MANIFEST.json")
    check(prior_manifest["files"] + [prior_manifest["upstream_selection_status_preserved"]])
    summary = read(HERE / "SUMMARY.json")
    if summary["status"] != "SOURCE_SEMANTIC_AUDIT_COMPLETE" or summary["unresolved_items"] != 0:
        raise RuntimeError("AUDIT_NOT_COMPLETE")
    b1 = summary["models"]["B1"]
    now = datetime.now(timezone.utc).isoformat()
    decision = {
        "schema_version": "day6_posthoc_source_semantic_decision_v1",
        "completed_at_utc": now,
        "status": "SOURCE_SEMANTIC_AUDIT_COMPLETE_FORMAL_B_NO_GO",
        "day6_source_semantic_audit_complete": True,
        "day6_overall_selfplay_goal_complete": False,
        "questions": 12, "roots": 6, "responses": 24, "item_judgments": 144,
        "source_gold_matches": summary["source_gold_matches"],
        "new_solver_requests": 0, "new_training_runs": 0, "new_questions": 0,
        "new_dialogue_expansions": 0, "dev_test_requests": 0,
        "qualified_current_B1_narrow_roots": b1["qualified_fixed_citation_failure_roots"],
        "qualified_current_B1_narrow_fact_families": b1["qualified_fixed_citation_failure_fact_families"],
        "required_distinct_roots": 3, "required_distinct_fact_families": 3,
        "controlled_support": "NOT_EXECUTED",
        "second_wave_formal_B_GO": False, "B1_formal_reproduced": False,
        "feedback_causal_benefit_established": False, "challenger_feedback_released": False,
        "counterexample": "Historical 112812 qualification is preserved, but the selected new-interface 112812 ordering response is correct; no current repeated failure is inferred there.",
        "reason": "The interface passes structure checks, but source citations remain incomplete and the current fully source-qualified B1 fixed-citation failure covers only one root/family, with no controlled support.",
        "next_action": "Independently verify the existing frozen narrow ordering probe reserve (three historical Train roots, two mappings, paired control/position treatment, at most 24 A1/B1 requests) before release/execution. Do not train or start formal targeted B yet.",
        "report": bind(HERE / "REPORT_ZH.md"), "summary": bind(HERE / "SUMMARY.json"),
        "semantic_audit": bind(HERE / "SEMANTIC_AUDIT.json"),
        "independent_audit": bind(independent_path),
        "original_prepared_probe": bind(DAY5 / "NARROW_PROBE_RESERVE_V2.json"),
        "original_probe_cpu_freeze": bind(DAY5 / "NARROW_PROBE_CPU_FREEZE.json"),
        "old_status_file_preserved": prior_manifest["upstream_selection_status_preserved"],
    }
    write_new(decision_path, decision)
    files = [bind(path) for path in sorted(HERE.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path != manifest_path]
    write_new(manifest_path, {
        "schema_version": "day6_posthoc_source_semantic_final_manifest_v1",
        "sealed_at_utc": now, "status": "SEALED_OFFLINE_SOURCE_AUDIT",
        "files": files, "file_count": len(files),
        "prior_v11_manifest": bind(PRIOR / "FINAL_MANIFEST.json"),
        "old_status_file_preserved": prior_manifest["upstream_selection_status_preserved"],
        "final_manifest_excludes_itself": True,
    })
    check(files)
    write_new(pointer_path, {
        "schema_version": "day6_semantic_latest_navigation_v1", "updated_at_utc": now,
        "package": str(HERE), "status": decision["status"],
        "report": bind(HERE / "REPORT_ZH.md"), "decision": bind(decision_path),
        "manifest": bind(manifest_path), "next_action": decision["next_action"],
        "second_wave_formal_B_GO": False, "old_navigation_records_modified": False,
    })
    print(json.dumps({"status": decision["status"], "sealed_files": len(files),
                      "pointer": str(pointer_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
