#!/usr/bin/env python3
"""Bind the actual Train diagnosis and independent audit into Day5 closeout."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def binding(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify_binding_rows(value):
    checked = 0
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            actual = binding(Path(value["path"]))
            if actual != {key: value[key] for key in actual}:
                raise ValueError(f"Changed evidence binding: {value['path']}")
            checked += 1
        for child in value.values():
            checked += verify_binding_rows(child)
    elif isinstance(value, list):
        for child in value:
            checked += verify_binding_rows(child)
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    summary_path = ROOT / "SUMMARY.json"
    report_path = ROOT / "REPORT_ZH.md"
    diagnosis_path = ROOT / "train_diagnosis_prepare/DIAGNOSIS_SUMMARY.json"
    source_freeze_path = ROOT / "train_diagnosis_prepare/DIAGNOSIS_RECORDS_FREEZE.json"
    summary = read(summary_path)
    diagnosis = read(diagnosis_path)
    audit = read(args.audit)
    audit_status = str(audit.get("status", ""))
    if not audit_status.startswith("PASS"):
        raise ValueError(f"Independent actual diagnosis audit is not PASS: {audit_status}")
    if audit.get("failed_checks") != 0 or audit.get("total_checks") != 2690:
        raise ValueError("Expected the actual2690/2690 independent diagnosis audit")
    verify_binding_rows(audit)
    verify_binding_rows(read(source_freeze_path))
    checked = verify_binding_rows(diagnosis)
    if not diagnosis["closure"]["train_diagnosis_records_closed"]:
        raise ValueError("Train diagnosis records are not closed")
    if diagnosis["feedback"]["second_wave_formal_B_GO"]:
        raise ValueError("This closeout is restricted to the actual NO_GO decision")

    summary.update({
        "schema_version": "day5_actual_execution_closeout_v1",
        "status": "D5_EXECUTED_CLOSED_SECOND_WAVE_B_NO_GO",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_records_closed": True,
        "day5_complete": True,
        "day5_completion_meaning": "All prescribed Day5 execution/diagnosis/review/decision deliverables closed, not all GO gates passed",
        "full_day5_GO": False,
        "second_wave_formal_B_GO": False,
        "pending": [],
    })
    for item in [
        "actual_A1_B1_Train_diagnosis_768_new_responses",
        "private_A1_and_versioned_B1_source_semantic_reviews",
        "B1_hypothesis_feedback_snapshot_and_formal_second_wave_NO_GO",
        "independent_actual_diagnosis_and_source_review_audit",
        "final_actual_resource_accounting",
    ]:
        if item not in summary["completed"]:
            summary["completed"].append(item)

    summary["train_diagnosis"] = {
        "summary": binding(diagnosis_path),
        "source_records_freeze": binding(source_freeze_path),
        "report": binding(ROOT / "train_diagnosis_prepare/REPORT_ZH.md"),
        "independent_audit": binding(args.audit),
        "independent_audit_status": audit_status,
        "independent_audit_total_checks": audit["total_checks"],
        "independent_audit_failed_checks": audit["failed_checks"],
        "independent_audit_revision": audit["audit_revision"],
        "independent_reviewer": audit["reviewer"],
        "scope": diagnosis["scope"],
        "primary_metric": diagnosis["primary_metric"],
        "S0": {
            "status": diagnosis["S0"]["status"],
            "new_calls": diagnosis["S0"]["new_calls"],
            "reused_direct_responses": diagnosis["S0"]["reused_direct_responses"],
            "direct": diagnosis["S0"]["direct_groups_relative_to_A1"],
            "exposure_note": diagnosis["exposure_note_for_S0"],
        },
        "models": {
            role: {key: model[key] for key in [
                "endpoint", "direct", "brief_evidence_secondary_not_primary",
                "source_semantic_audit", "source_semantic_dimensions", "usage",
            ]}
            for role, model in diagnosis["models"].items()
        },
        "actual_new_usage": diagnosis["actual_new_usage"],
        "controlled_probe": diagnosis["controlled_probe"],
        "feedback": diagnosis["feedback"],
        "error_interpretation": diagnosis["error_interpretation"],
        "closure": diagnosis["closure"],
        "A1_details_released_to_A_Challenger": False,
        "no_heldout_generalization_claim": True,
    }

    usage = diagnosis["actual_new_usage"]
    cost = summary["cost_completed_only"]
    cost.update({
        "train_diagnosis_new_requests": usage["new_model_requests"],
        "train_diagnosis_new_returned_requests": usage["new_returned_requests"],
        "train_diagnosis_input_tokens": usage["new_input_tokens"],
        "train_diagnosis_generated_tokens": usage["new_output_tokens"],
        "train_diagnosis_summed_branch_seconds": usage["summed_actual_branch_wall_seconds"],
        "train_diagnosis_included_model_load_seconds": usage["summed_actual_model_load_wall_seconds"],
        "train_diagnosis_reused_S0_new_cost": 0,
        "train_diagnosis_reused_historical_input_tokens_not_added": diagnosis["S0"]["historical_reused_input_tokens"],
        "train_diagnosis_reused_historical_output_tokens_not_added": diagnosis["S0"]["historical_reused_output_tokens"],
        "new_probe_requests": 0,
        "retries": 0,
        "cost_scope_excludes": ["CPU_prepare_merge_and_review", "external_orchestration_gaps_for_diagnosis"],
    })
    cost["total_new_input_tokens"] = sum(cost[key] for key in [
        "resource_smoke_input_tokens", "formal_training_input_tokens",
        "common_DEV_input_tokens", "train_diagnosis_input_tokens",
    ])
    cost["total_training_supervised_tokens"] = cost["resource_smoke_supervised_tokens"] + cost["formal_training_supervised_tokens"]
    cost["total_evaluation_diagnosis_generated_tokens"] = cost["common_DEV_generated_tokens"] + cost["train_diagnosis_generated_tokens"]
    cost["summed_observed_nonoverlapping_stage_seconds"] = sum(cost[key] for key in [
        "resource_smoke_whole_launch_seconds", "formal_training_whole_launch_seconds",
        "common_DEV_load_inference_seconds", "train_diagnosis_summed_branch_seconds",
    ])
    cost["summed_stage_wall_scope"] = "Resource smoke/formal training whole launches plus DEV and Train diagnostic executor branches; excludes CPU review and diagnostic external orchestration gaps; not end-to-end elapsed time"

    decision = summary["scientific_decision"]
    decision.update({
        "actual_train_diagnosis_completed": True,
        "independent_actual_diagnosis_audit_passed": True,
        "B1_new_formal_reproduced_cards": 0,
        "B1_current_qualified_narrow_failure_roots": diagnosis["feedback"]["qualified_narrow_failure_roots"],
        "B1_current_qualified_narrow_failure_fact_families": diagnosis["feedback"]["qualified_narrow_failure_fact_families"],
        "B1_reproduced": False,
        "next_formal_targeted_B_wave_GO": False,
        "residual_direct_errors_remain": True,
        "formal_training_failed": False,
        "feedback_stable_extra_benefit_established": False,
        "next_narrow_work": "Train-side diagnostic output, source citation, actor/status and first-report reliability; freeze any new procedure before another formal feedback qualification attempt",
    })
    decision["limits"] = [line for line in decision["limits"] if line != "training-side B1 feedback qualification is still pending"]
    decision["limits"].extend([
        "Train diagnosis panel has6 shared training roots; own/opposite arm is not independent heldout generalization",
        "Only1 source-qualified narrow B1 ordering failure root/family; minimum3 and controlled support not met",
        "Visible brief reasons are not faithful internal CoT evidence; malformed outputs are not rescued",
    ])

    status_old = "**当前状态：ACTUAL_TRAIN_DIAGNOSIS_COMPLETE_INDEPENDENT_REVIEW_PENDING。隔离资源试运行、A1/B1正式训练、共同DEV、训练侧诊断与来源审核已完成；下一波正式B反馈资格为NO_GO，正在等待独立诊断审核最终绑定。训练执行成功不等于下一波反馈准入。**"
    status_new = "**当前状态：D5_EXECUTED_CLOSED_SECOND_WAVE_B_NO_GO。第五天规定的训练、DEV、Train诊断、来源审核及资格裁决记录均已闭合，独立诊断审核通过；full_day5_GO=false、下一波正式B为NO_GO。训练执行成功不等于下一波反馈准入，也不表示残余弱点已消失。**"
    report = report_path.read_text()
    if status_old not in report:
        raise ValueError("Expected pending independent-review status missing from report")
    report = report.replace(status_old, status_new)
    pending_text = "独立诊断审核结果待最终绑定。"
    if pending_text not in report:
        raise ValueError("Expected pending audit binding missing from report")
    relative_audit = args.audit.resolve().relative_to(ROOT)
    report = report.replace(pending_text, f"独立诊断审核通过，见[实际诊断独立审核]({relative_audit})；来源资格不足的NO_GO裁决已纳入结项，没有遗留执行任务。")
    report_path.write_text(report)
    summary["report"] = binding(report_path)
    add_paths = [
        diagnosis_path,
        source_freeze_path,
        ROOT / "train_diagnosis_prepare/REPORT_ZH.md",
        Path(diagnosis["models"]["A1"]["source_semantic_audit"]["path"]),
        Path(diagnosis["models"]["B1"]["source_semantic_audit"]["path"]),
        Path(diagnosis["feedback"]["snapshot"]["path"]),
        Path(diagnosis["feedback"]["decision"]["path"]),
        args.audit,
        ROOT / "review/B1_FEEDBACK_AUDIT_ZH.md",
        Path(audit["audit_revision"]["initial_checker_failure_receipt"]["path"]),
    ]
    existing = {row["path"] for row in summary["evidence_bindings"]}
    for path in add_paths:
        row = binding(path)
        if row["path"] not in existing:
            summary["evidence_bindings"].append(row)
            existing.add(row["path"])
    final_verified = verify_binding_rows(summary)
    summary["closeout_validation"] = {
        "diagnosis_nested_bindings_verified": checked,
        "summary_nested_bindings_verified_before_this_receipt": final_verified,
        "pending_tasks": 0,
        "TEST_requests": 0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "summary": binding(summary_path), "report": summary["report"], "verified_bindings": final_verified}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
