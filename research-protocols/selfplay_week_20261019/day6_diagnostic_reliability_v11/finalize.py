"""Seal completed format-only results without changing execution inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import run as frozen


def write_new(path, obj):
    with path.open("x") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    here = frozen.HERE
    decision_path = here / "DECISION.json"
    manifest_path = here / "FINAL_MANIFEST.json"
    pointer_path = here.parent / "DAY6_LATEST.json"
    if any(path.exists() for path in (decision_path, manifest_path, pointer_path)):
        raise RuntimeError("CLOSEOUT_TARGET_EXISTS_NO_OVERWRITE")
    audit_path = here / "INDEPENDENT_POSTEXECUTION_AUDIT.json"
    audit = frozen.read(audit_path)
    if audit.get("status") != "PASS" or audit.get("scope") != "DAY6_DIAGNOSTIC_INTERFACE_POSTEXECUTION":
        raise RuntimeError("INDEPENDENT_POSTEXECUTION_PASS_REQUIRED")
    frozen.check(audit["audited_files"])
    score_path = here / "FORMAT_AUDIT.json"
    required_audited = [score_path, here / "score_format.py", here / "REQUESTS.json",
                        frozen.FREEZE_PATH, here / "INDEPENDENT_PREFLIGHT.json"]
    required_audited += [here / "runs" / role / "RAW_RESPONSE_FREEZE.json" for role in ("A1", "B1")]
    for path in required_audited:
        if frozen.binding(path) not in audit["audited_files"]:
            raise RuntimeError("FINAL_AUDIT_MISSING_BINDING:" + str(path))
    score = frozen.read(score_path)
    if score["status"] != "PASS_FORMAT_ONLY_GATE":
        raise RuntimeError("FORMAT_GATE_NOT_PASSED")
    freeze = frozen.read(frozen.FREEZE_PATH)
    # These include every execution input and the immutable status selection file.
    frozen.check(freeze["files"] + freeze["sources"] + freeze["prompts"] +
                 freeze["upstream"] + freeze["legacy_readout_bindings"] +
                 [freeze["selection_root_file"]])
    counts = {}
    for role in ("A1", "B1"):
        metrics = score["models"][role]
        raw = frozen.read(here / "runs" / role / "RAW_RESPONSE_FREEZE.json")
        frozen.check(raw["files"] + [raw["preparation_freeze"], raw["audit"]])
        if any(metrics[key] != 12 for key in ("planned", "submitted", "returned", "mechanically_valid")):
            raise RuntimeError("UNEXPECTED_FINAL_COUNTS")
        if raw["automatic_retry"] or raw["failure"] or not metrics["engineering_gate"]["passed"]:
            raise RuntimeError("EXECUTION_OR_GATE_INCONSISTENT")
        counts[role] = {key: metrics[key] for key in (
            "planned", "submitted", "returned", "mechanically_valid",
            "json_schema_valid", "legacy_schema_valid_same_questions", "wall_seconds", "model_load_seconds")}
    now = datetime.now(timezone.utc).isoformat()
    decision = {
        "schema_version": "day6_diagnostic_interface_decision_v1",
        "completed_at_utc": now,
        "status": "PASS_FORMAT_ONLY_GATE",
        "engineering_interface_GO": True,
        "source_semantic_audit_status": "PENDING_NOT_PERFORMED_IN_THIS_PILOT",
        "new_reproduced_weakness_admission": False,
        "second_wave_formal_B_GO": False,
        "formal_B_status": "NO_GO_UNCHANGED_FROM_DAY5",
        "causal_feedback_benefit_established": False,
        "answer_accuracy_evaluated": False,
        "new_training_runs": 0,
        "new_questions": 0,
        "new_dialogue_expansions": 0,
        "dev_test_requests": 0,
        "new_solver_requests": 24,
        "automatic_retries": 0,
        "counts": counts,
        "next_action": "Audit source support and judgment-to-answer consistency for all 24 outputs / 144 item judgments, then reassess weakness eligibility under the unchanged Day5 policy.",
        "report": frozen.binding(here / "REPORT_ZH.md"),
        "format_audit": frozen.binding(score_path),
        "independent_postexecution_audit": frozen.binding(audit_path),
        "preparation_freeze": frozen.binding(frozen.FREEZE_PATH),
        "preserved_selection_root_file": freeze["selection_root_file"],
    }
    write_new(decision_path, decision)
    files = [frozen.binding(path) for path in sorted(here.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path != manifest_path]
    write_new(manifest_path, {
        "schema_version": "day6_diagnostic_interface_final_manifest_v1",
        "sealed_at_utc": now,
        "status": "SEALED_FORMAT_PILOT_ONLY",
        "files": files,
        "file_count": len(files),
        "upstream_selection_status_preserved": freeze["selection_root_file"],
        "final_manifest_excludes_itself": True,
    })
    frozen.check(files + [freeze["selection_root_file"]])
    write_new(pointer_path, {
        "schema_version": "day6_latest_navigation_v1",
        "updated_at_utc": now,
        "package": str(here),
        "status": "PASS_FORMAT_ONLY_GATE",
        "second_wave_formal_B_GO": False,
        "report": frozen.binding(here / "REPORT_ZH.md"),
        "decision": frozen.binding(decision_path),
        "manifest": frozen.binding(manifest_path),
        "next_action": decision["next_action"],
        "CURRENT_STATUS_modified": False,
    })
    print(json.dumps({"status": decision["status"], "sealed_files": len(files),
                      "manifest": str(manifest_path), "pointer": str(pointer_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
