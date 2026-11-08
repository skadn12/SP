"""Seal the completed pilot and add separate pointers to the mutable navigation."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binding(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    target = HERE / "FINAL_MANIFEST.json"
    assert not target.exists(), "Pilot already sealed; do not silently overwrite."
    summary = read(HERE / "SUMMARY.json")
    assert summary["admitted_questions"] == 20 and summary["diagnosis"]["calls"] == 258
    for name in ["POST_SCORE_RECOMPUTATION.json", "POST_SCORE_INDEPENDENT_REVIEW.json"]:
        assert read(HERE / "review" / name)["status"].startswith("PASS"), name
    assert (HERE / "review/SOLVER_OBSERVABLE_ERROR_ANALYSIS.json").exists()
    checked, errors = {}, []

    def verify_bindings(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                path = Path(value["path"])
                if path.is_file():
                    actual = binding(path)
                    checked[str(path)] = actual
                    if actual["sha256"] != value["sha256"]:
                        errors.append({"path": str(path), "error": "HASH_MISMATCH"})
                else:
                    errors.append({"path": str(path), "error": "BOUND_FILE_MISSING"})
            for child in value.values():
                verify_bindings(child)
        elif isinstance(value, list):
            for child in value:
                verify_bindings(child)

    for name in ["GENERATION_FREEZE.json", "CONTENT_ADMISSION.json", "EXECUTION_FREEZE.json",
                 "runtime/direct_v1/RAW_RESPONSE_FREEZE.json", "runtime/direct_v1/SCORES.json"]:
        verify_bindings(read(HERE / name))
    old_manifest = WEEK / "day3_solver187_v1/FINAL_MANIFEST.json"
    verify_bindings(read(old_manifest))
    assert not errors, errors
    save(HERE / "review/FINAL_BINDING_CHECK.json", {
        "status": "PASS", "checked_file_count": len(checked), "errors": errors,
        "old_187_manifest_unchanged": binding(old_manifest),
        "scope": "Recorded file bindings in generation, admission, execution, raw/scored records and old187 final manifest; not an independent repeat hash of all model weights."
    })
    history = HERE / "navigation_history"
    history.mkdir(exist_ok=False)
    for name in ["README_ZH.md", "CURRENT_STATUS.json", "DAY3_LATEST.json"]:
        shutil.copy2(WEEK / name, history / name)
    files = [binding(path) for path in sorted(HERE.rglob("*"))
             if path.is_file() and path != target and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    now = datetime.now(timezone.utc)
    save(target, {"status": "COMPLETE_VERIFIED_TRAIN_DIAGNOSTIC_ONLY", "created_at_utc": now.isoformat(),
        "files": files, "scope": "All pilot artifacts except this manifest and disposable Python bytecode; navigation pre-update snapshots included.",
        "planned_roots": 4, "planned_slots": 24, "content_pass": 20, "content_hold": 1, "source_shortfall": 3,
        "new_solver_calls": 258, "new_training_runs": 0, "dev_test_calls": 0,
        "old_frozen187_modified": False, "new_training_admitted": 0, "day3_overall_complete": False})
    progress = {"status": summary["status"], "report": binding(HERE / "REPORT_ZH.md"),
        "summary": binding(HERE / "SUMMARY.json"), "manifest": binding(target),
        "next_step": binding(HERE / "NEXT_STEP_ZH.md"), "planned_roots": 4, "planned_slots": 24,
        "content_pass": 20, "content_hold": 1, "source_shortfall": 3,
        "MC_questions": 9, "ordering_questions": 11, "solver_calls": 258,
        "by_type": summary["by_type"], "by_ordering_arity": {key: value for key, value in summary["by_arity"].items() if value["question_type"] == "ordering"},
        "new_training_admitted": 0, "feedback_consumed_by_new_expansion": False,
        "training_started": False, "dev_test_calls": 0, "old_frozen187_modified": False,
        "overall_day3_complete": False}
    next_action = "Run a separately frozen Train expansion/authoring loop using material-gap and shortcut feedback; preserve 3 old-event +3 dataset-new-event scenes. Formal common-evaluation admission and matched A/B training freezes remain pending."
    for name in ["CURRENT_STATUS.json", "DAY3_LATEST.json"]:
        value = read(WEEK / name)
        value["updated_at_utc"] = now.isoformat()
        if "updated_at_beijing" in value:
            value["updated_at_beijing"] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
        value["diversity_crossroot_progress"] = progress
        value["active_workstream"] = "DIVERSITY_CROSSROOT_PILOT_COMPLETE_NEXT_FEEDBACK_EXPANSION"
        value["mainline_next_action_before_diversity_pilot"] = value["next_action"]
        value["next_action"] = next_action
        value["new_solver_calls"] = 2946
        value["solver_call_scope_breakdown"] = {"day3_old187_diagnostic": 2688, "day3_diversity_crossroot": 258, "day3_train_diagnostic_total": 2946, "day2_excluded_from_this_total": True}
        save(WEEK / name, value)
    readme = WEEK / "README_ZH.md"
    content = readme.read_text()
    marker = "已完成S0＝M-clean对全部187道训练候选的统一诊断"
    index = content.index(marker)
    update = ("新增：已完成[新出题形式跨4root验证](day3_diversity_crossroot_v1/REPORT_ZH.md)：24个固定题位，20道内容通过（9多选、11排序）、1暂缓、3素材缺额，S0完成258次作答。多选完整正确率37.65%；排序按3/4/6项分别91.67%/16.67%/25.00%，均为题目等权。多选已涵盖原因、计算、条件和证据，排序已涵盖叙述、时刻和步骤；三项步骤偏易与题面捷径仍需处理。题目未入训练池，本批是Train可行性与诊断，不是反馈/训练收益证明。\n\n"
              "当前下一步见[反馈扩充与出题闭环](day3_diversity_crossroot_v1/NEXT_STEP_ZH.md)：继承旧事件延续3场＋数据集新事件3场，使用素材缺口与设计反馈生成独立新版本；如需验证反馈有效，另设同root同预算对照。共同评估准入及正式A/B训练准备仍待完成。\n\n")
    content = content[:index] + update + content[index:]
    content = content.replace("下一步完成共同评估题的事实族边界与当前准入核验", "正式训练前仍需完成共同评估题的事实族边界与当前准入核验")
    content += "\n本次跨root验证前的导航副本保存在day3_diversity_crossroot_v1/navigation_history/；旧187题及其诊断记录保持封存。\n"
    readme.write_text(content)
    for row in files:
        assert binding(Path(row["path"]))["sha256"] == row["sha256"], row["path"]
    print(json.dumps({"status": "SEALED_AND_NAVIGATION_UPDATED", "pilot_files": len(files), "verified_prior_bindings": len(checked), "manifest": binding(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
