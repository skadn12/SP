"""Publish the completed live batch without rewriting historical experiment files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = WEEK.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def check(bound):
    assert binding(bound["path"]) == bound, "BINDING_DRIFT: " + bound["path"]


def save(path, value, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    assert not (HERE / "SUMMARY.json").exists(), "NO_CLOSEOUT_OVERWRITE"
    batches = {
        "four_real_examples": WEEK / "postweek_reason_feedback_repair_v1/outputs",
        "seven_operation_diagnostic": WEEK / "postweek_diversity_reason_v1/outputs",
        "two_feedback_routed_questions": HERE / "question_execution",
    }
    expected = {"four_real_examples": 4, "seven_operation_diagnostic": 7, "two_feedback_routed_questions": 2}
    batch_stats = {}
    artifacts = []
    for name, directory in batches.items():
        frozen = read(directory / "RAW_RESPONSE_FREEZE.json")
        assert len(frozen["bindings"]) == expected[name]
        for bound in frozen["bindings"]:
            check(bound)
        raws = [read(bound["path"]) for bound in frozen["bindings"]]
        packet = read(directory / "REASON_REVIEW_PACKET.json")
        assert len(packet["requests"]) == expected[name]
        semantic = read(directory / "SEMANTIC_REVIEW_AI.json")
        reviewed = {row["request_id"]: row for row in semantic["reviews"]}
        assert len(reviewed) == expected[name]
        assert set(reviewed) == {row["request_id"] for row in packet["requests"]}
        for row in packet["requests"]:
            review = reviewed[row["request_id"]]
            assert review["review_complete"] is True
            assert review["raw_binding"] == row["raw_binding"]
            check(review["raw_binding"])
        batch_stats[name] = {
            "planned_requests": expected[name],
            "submitted_requests": len(list((directory / "submitted").glob("*.json"))),
            "returned_requests": sum(row.get("status") == "RETURNED" for row in raws),
            "structurally_valid_requests": sum(row.get("status") == "STRUCTURALLY_VALID_REASONS_SAVED" for row in packet["requests"]),
            "wall_seconds": frozen.get("wall_seconds"),
            "raw_freeze": binding(directory / "RAW_RESPONSE_FREEZE.json"),
            "semantic_review": binding(directory / "SEMANTIC_REVIEW_AI.json"),
        }
        artifacts.extend([directory / "RAW_RESPONSE_FREEZE.json", directory / "REASON_REVIEW_PACKET.json", directory / "SEMANTIC_REVIEW_AI.json"])
    sources = read(HERE / "SOURCE_ACCEPTANCE_SUMMARY.json")
    assert len(sources["roots"]) == 4
    for row in sources["roots"]:
        check(row["question_ready"])
        check(row["source_binding"])
        root_dir = HERE / "live_run_v2" / row["root_id"]
        receipt = read(root_dir / "CHALLENGER_READ_RECEIPT.json")
        assert receipt["root_id"] == row["root_id"]
        assert receipt["consumer_type"] == "codex_ai_agent"
        assert receipt["model_consumption_recorded"] is True
        for key, bound in receipt.items():
            if key.endswith("_binding") and bound is not None:
                check(bound)
        artifacts.extend(root_dir / name for name in ("QUESTION_READY_SOURCE.json", "CHALLENGER_PLAN.json", "CHALLENGER_READ_RECEIPT.json"))
    content = read(HERE / "questions/CONTENT_REVIEW_AI.json")
    assert len(content["reviews"]) == 2
    for row in content["reviews"]:
        assert row["status"] == "PASS"
        check(row["candidate_binding"])
        check(row["source_binding"])
    seven = read(batches["seven_operation_diagnostic"] / "SCORES.json")
    two = read(batches["two_feedback_routed_questions"] / "SCORES.json")
    artifacts += [
        HERE / "REPORT_ZH.md", HERE / "QUESTION_EXECUTION_FREEZE.json",
        HERE / "SOURCE_ACCEPTANCE_SUMMARY.json", HERE / "questions/CONTENT_REVIEW_AI.json",
        HERE / "questions/QUESTION_AUTHOR_READ_RECEIPT.json",
        batches["four_real_examples"] / "EXAMPLES_ZH.md",
        batches["seven_operation_diagnostic"] / "SCORES.json",
        batches["seven_operation_diagnostic"] / "REPORT_ZH.md",
        batches["two_feedback_routed_questions"] / "SCORES.json",
        batches["two_feedback_routed_questions"] / "REPORT_ZH.md",
        batches["two_feedback_routed_questions"] / "PREPARATION.json",
        batches["two_feedback_routed_questions"] / "PRIVATE_SCORING_BINDINGS.json",
        batches["two_feedback_routed_questions"] / "STARTED.json",
        REPO / "scripts/current_method/reason_material_feedback_pipeline.py",
        REPO / "scripts/current_method/test_reason_material_feedback_pipeline.py",
        REPO / "scripts/current_method/REASON_MATERIAL_FEEDBACK_PIPELINE.md",
        HERE / "question_batch.py", HERE / "test_question_batch.py",
        HERE / "render_new_examples.py",
        Path(__file__),
    ]
    now = datetime.now(timezone.utc)
    summary = {
        "schema_version": "postweek_live_reason_mainline_closeout_v1",
        "updated_at_utc": now.isoformat(),
        "scope": "Train-only exploratory engineering and task-realization validation",
        "status": "LIVE_REASON_MAINLINE_AND_DIVERSITY_CHECK_COMPLETED",
        "solver_model_role": "B1",
        "batches": batch_stats,
        "new_solver_requests_submitted": sum(v["submitted_requests"] for v in batch_stats.values()),
        "new_solver_responses_returned": sum(v["returned_requests"] for v in batch_stats.values()),
        "question_ready_roots": 4,
        "challenger_actual_read_receipts": 4,
        "new_questions_authored": 2,
        "new_questions_content_pass": 2,
        "seven_operation_diagnostic": seven["totals"],
        "two_feedback_routed_exact": {"correct": two["correct"], "denominator": two["requests"]},
        "new_dialogues_generated": 0,
        "reused_historical_extension": {"root_id": "cc-train-episode-104814", "original_turns": 178, "descendant_turns": 270, "historical_added_turns": 92, "generated_from_current_feedback": False},
        "active_handoff": str(HERE / "live_run_v2"),
        "superseded_cpu_handoff": str(HERE / "live_run"),
        "formal_weakness_admission": False,
        "causal_feedback_benefit_proven": False,
        "cot_accuracy_gain_proven": False,
        "training_admission": False,
        "new_training_runs": 0,
        "dev_test_requests": 0,
        "new_six_scene_extension_live_validation": False,
        "cpu_pipeline_regression_tests": 19,
        "cpu_question_runner_regression_tests": 5,
        "limitations": [
            "Visible brief explanations are model outputs, not verified faithful internal reasoning.",
            "Final correctness does not certify citations or explanations; failed review fields are retained.",
            "The seven diagnostic tasks were reused, not newly authored in this batch.",
            "Single display and sampling seed per task cannot establish stable weaknesses or causal benefits.",
            "The two new tasks contain explicit task cues and do not establish increased difficulty.",
            "New six-scene generation is implemented and CPU-tested; live source paths here reused original or historical sources.",
        ],
    }
    artifact_bindings = [binding(p) for p in artifacts]
    backups = HERE / "navigation_before_closeout"
    backups.mkdir()
    current_path = WEEK / "CURRENT_STATUS.json"
    interface_path = WEEK / "FEEDBACK_INTERFACE_LATEST.json"
    for path in (current_path, interface_path):
        shutil.copy2(path, backups / path.name)
        artifact_bindings.append(binding(backups / path.name))
    current = read(current_path)
    interface = read(interface_path)
    # Validate the old implementation/document pointers before preserving them.
    for key in ("implementation", "documentation"):
        check(interface[key])
    save(HERE / "SUMMARY.json", summary)
    artifact_bindings.append(binding(HERE / "SUMMARY.json"))
    save(HERE / "FINAL_MANIFEST.json", {"schema_version": "postweek_closeout_artifacts_v1", "bindings": artifact_bindings})
    stamp = now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()
    live_fields = {
        "status": "LIVE_B1_REASONS_REVIEWED_AND_MAINLINE_CONSUMED",
        "new_model_calls": summary["new_solver_requests_submitted"],
        "model_call_scope": "Local B1 Solver requests; excludes Codex author, Challenger and review agent work.",
        "gpu_inference_performed": True,
        "current_model_reasons_collected": True,
        "semantic_review_complete": True,
        "challenger_consumed_new_reasons": True,
        "formal_feedback_gain_proven": False,
        "current_summary": binding(HERE / "SUMMARY.json"),
        "current_report": binding(HERE / "REPORT_ZH.md"),
        "batches": batch_stats,
    }
    current.update(updated_at_beijing=stamp, updated_at_utc=now.isoformat(),
        active_workstream="postweek_live_reason_feedback_mainline_and_diversity", status=summary["status"],
        reason_feedback_interface={**current.get("reason_feedback_interface", {}), **live_fields},
        postweek_live_execution=summary)
    current["next_action"] = "Prospectively freeze a broader Train-only diversity and feedback-control batch using reviewed reasons and source-qualified material goals; establish matched controls before claiming feedback benefit or starting training."
    current["mainline_next_action_before_diversity_pilot"] = current["next_action"]
    interface.update(updated_at_beijing=stamp, **live_fields)
    interface.update(live_model_validation="PASS_WITH_RETAINED_SEMANTIC_FAILURES", cpu_validation_status="PASS_CPU_AND_BOUNDED_LIVE_INTEGRATION", note="The original implementation and README remain historical frozen interface records. Current operation-specific prompts, actual results and mainline execution are linked in current_summary/current_report; no causal or training-benefit claim.")
    pipeline_pointer = {
        "schema_version": "reason_material_feedback_pipeline_latest_v1",
        "updated_at_utc": now.isoformat(), "status": summary["status"],
        "implementation": binding(REPO / "scripts/current_method/reason_material_feedback_pipeline.py"),
        "documentation": binding(REPO / "scripts/current_method/REASON_MATERIAL_FEEDBACK_PIPELINE.md"),
        "operation_reason_implementation": binding(WEEK / "postweek_diversity_reason_v1/run.py"),
        "active_handoff": str(HERE / "live_run_v2"),
        "summary": binding(HERE / "SUMMARY.json"), "report": binding(HERE / "REPORT_ZH.md"),
        "manifest": binding(HERE / "FINAL_MANIFEST.json"),
        "new_six_scene_extension_live_validation": False,
        "formal_feedback_gain_proven": False,
    }
    save(current_path, current, exclusive=False)
    save(interface_path, interface, exclusive=False)
    save(WEEK / "FEEDBACK_PIPELINE_LATEST.json", pipeline_pointer)
    print(json.dumps({"status": summary["status"], "new_model_calls": summary["new_solver_requests_submitted"], "new_task_scores": summary["two_feedback_routed_exact"]}))


if __name__ == "__main__":
    main()
