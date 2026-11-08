"""Seal Day 4 CPU artifacts and update mutable navigation after independent checks."""
import datetime
import json
import shutil
from pathlib import Path

from build_schedule import bind, read, save, verify_ref

OUT = Path(__file__).resolve().parent
WEEK = OUT.parent


def main():
    summary = read(OUT / "SCHEDULE_SUMMARY.json")
    policy = read(OUT / "EXECUTION_POLICY.json")
    audit = read(OUT / "review/INDEPENDENT_SCHEDULE_AUDIT.json")
    review = read(OUT / "review/RUNTIME_CODE_REVIEW.json")
    runtime_binding = read(OUT / "runtime/RUNTIME_BINDING.json")
    assert audit["status"] == "PASS_INDEPENDENT_CPU_SCHEDULE_REPLAY"
    assert review["status"].startswith("PASS"), review["status"]
    assert runtime_binding["status"].startswith("PASS"), runtime_binding["status"]
    runtime_ref = bind(OUT / "runtime/schedule_runner.py")
    preflights = {a: read(OUT / f"runtime/PREFLIGHT_FINAL_{a}.json") for a in ["A", "B"]}
    for arm, preflight in preflights.items():
        assert preflight["status"] == "PASS_CPU_PREFLIGHT"
        assert preflight["runtime"] == runtime_ref
        assert preflight["policy"] == bind(OUT / "EXECUTION_POLICY.json")
        assert preflight["schedule"]["train_binding"] == summary["train_pools"][arm]
        assert preflight["schedule"]["microsteps"] == summary["arms"][arm]["microsteps"]
        assert preflight["schedule"]["optimizer_steps"] == summary["arms"][arm]["optimizer_steps"]
        assert preflight["schedule"]["full_input_tokens"] == summary["arms"][arm]["totals"]["full_input_tokens"]
        assert preflight["model_weights_loaded"] is False
    for ref in read(OUT / "SELECTION.json")["input_bindings"]:
        verify_ref(ref)
    for ref in summary["train_pools"].values():
        verify_ref(ref)
    identity = read(WEEK / "day1/MODEL_BINDINGS.json")
    identity_checks = []
    for role in ["base", "S0_M_clean_adapter"]:
        for ref in identity["models"][role]["files"]:
            verify_ref(ref)
            identity_checks.append({"role": role, **bind(ref["path"])})
    save("MODEL_IDENTITY_VERIFICATION.json", {"status": "PASS_CPU_BYTES_ONLY", "files": identity_checks,
         "manifest": bind(WEEK / "day1/MODEL_BINDINGS.json"), "weights_loaded": False})
    save("TRAINING_BUDGET_FREEZE.json", {
        "schema_version": "day4_actual_training_budget_freeze_v1",
        "status": "ACTUAL_CPU_DATA_AND_SCHEDULE_FROZEN_DAY5_SMOKE_PENDING",
        "arms": summary["arms"], "relative_full_input_token_difference": summary["relative_full_input_token_difference"],
        "training_data": summary["train_pools"], "policy": bind(OUT / "EXECUTION_POLICY.json"),
        "runtime": runtime_ref, "runtime_binding": bind(OUT / "runtime/RUNTIME_BINDING.json"),
        "actual_model_identity_verification": bind(OUT / "MODEL_IDENTITY_VERIFICATION.json"),
        "training_schedule": policy["training_schedule"], "budget": policy["budget"],
        "initialization": "Original base + actual M-clean custom LoRA loaded once per independent arm; no adapter on merged weights",
        "epochs_preexpanded": 2, "runtime_passes": 1, "shuffle": False, "runtime_permutation": False,
        "checkpoint": {"fixed_final_optimizer_step": summary["arms"]["A"]["optimizer_steps"],
                       "score_selection": False, "intermediate_test": False},
        "evaluation_lock": bind(OUT / "EVALUATION_LOCK.json"),
        "independent_audit": bind(OUT / "review/INDEPENDENT_SCHEDULE_AUDIT.json"),
        "runtime_code_review": bind(OUT / "review/RUNTIME_CODE_REVIEW.json"),
        "training_start_go": False, "required_next_gate": "DAY5_ISOLATED_LONGEST_INPUT_RESOURCE_SMOKE",
        "resource_smoke_runs": 0, "formal_training_runs": 0, "new_solver_calls": 0})
    save("DECISION.json", {
        "schema_version": "day4_schedule_decision_v1",
        "status": "D4_SCHEDULE_PASS_CPU_FROZEN_READY_FOR_DAY5_RESOURCE_SMOKE",
        "d4_cpu_schedule_complete": True, "q": 0.0, "selected_unique_questions_per_arm": 112,
        "common_foundation_selected": 24, "new_structure_selected_per_arm": 88,
        "legacy_selected": 0, "microsteps_per_arm": 224, "optimizer_steps_per_arm": 56,
        "no_score_selection": True, "common_foundation_identical": True,
        "two_distinct_epoch_displays": True, "zero_truncation": True,
        "relative_full_input_token_difference": summary["relative_full_input_token_difference"],
        "training_budget": bind(OUT / "TRAINING_BUDGET_FREEZE.json"),
        "independent_schedule_audit": bind(OUT / "review/INDEPENDENT_SCHEDULE_AUDIT.json"),
        "evaluation_lock": bind(OUT / "EVALUATION_LOCK.json"),
        "runtime_code_review": bind(OUT / "review/RUNTIME_CODE_REVIEW.json"),
        "formal_training_started": False, "training_start_go": False, "new_solver_calls": 0,
        "model_weights_loaded": False, "gpu_smoke_executed": False,
        "next_action": "DAY5: run isolated longest-input forward/backward/optimizer smoke for both arms with the frozen runtime and input hashes; only then train A1/B1 independently from S0 at the fixed final endpoint.",
        "limitations": ["Inherited bounded semantic QA and dependency labels; no exhaustive semantic independence claim.",
                       "No GPU memory/forward/backward result, Solver score, training improvement or feedback-causal benefit.",
                       "Frozen training package versions differ from historical evaluation; actual CPU token replay was independently checked.",
                       "All evaluation roots were historically exposed; cohorts remain separate and TEST final-only.",
                       "Shared filesystem and overlapping author/reviewer history are disclosed; no technical blind review claim."]})
    report = """# 第四天：实际训练池与日程冻结

**结论：D4-SCHEDULE 的 CPU 数据与日程检查通过，可以进入第五天隔离资源试运行。未运行 GPU、未加载模型、未训练，也没有新增 Solver 调用。**

| 项目 | A：多样化、无失败反馈 | B：多样化、训练侧排序反馈 |
|---|---:|---:|
| 独立题目 | 112 | 112 |
| 新结构题 | 88（35多选、53排序） | 88（35多选、53排序） |
| 相同基础题 | 24（15多选、9排序） | 24（15多选、9排序） |
| 第一轮旧题回放 | 0 | 0 |
| Train root | 35 | 35 |
| 两轮展示后的微步 / 优化器步 | 224 / 56 | 224 / 56 |
| 完整输入 token | 1,669,716 | 1,672,796 |
| 监督 token | 2,992 | 2,992 |

两臂输入 token 差 0.1841%，小于冻结的5%上限；全文输入6543—8057 tokens，最大长度12288，零截断。B只有53道新排序题具有既有合格的窄跨场景反馈链；35道多选采用共同设计，不宣称新多选弱点已正式复现。

旧题没有最终准入，按v2.1冻结q=0。新题/基础题监督token目标75%/25%，实际76.4037%/23.5963%；24道基础题是从已准入363道储备中按固定哈希前缀与完整累积步数规则选出，两臂相同。未使用Solver分数筛选。11道不成对的新候选不用于本次匹配训练，不代表内容失败。

每题每epoch恰好一次，两个展示均保留原文、题意和答案内容。数据已展平两轮，runtime只遍历一次，不再次洗牌、换位或运行两轮。新执行适配器复用原LoRA、损失和学习率实现，替换会洗牌/截断的原入口；两臂独立从原base+M-clean adapter开始，optimizer/scheduler独立，固定第56优化器步作为终点，不按DEV/TEST选checkpoint。

独立CPU审核重建了448条实际训练记录并复算363道基础候选和90种前缀，共392377项机械校验通过。最终runtime CPU预检逐条验证原生tokenization和监督mask；负向检查及代码审核已封存。该审核继承现有内容QA，不是重新逐题语义审核，也没有重新搜索全部首次提及。

本次训练冻结环境为/root/miniconda3/bin/python3（torch2.8.0、transformers4.57.3），与历史评估环境.venv-local4b（torch2.11.0、transformers5.9.0）不同，已单独披露和核验实际CPU token；尚未证明GPU执行支持。

共同评估保留115道历史回归、此前4道多选及12道多题型补充三个独立集合，来源/题面/Gold/请求字节不变。16道Gold6控制题在历史115中另列，不混入主指标。所有root历史暴露，不能据此声称未见root泛化；TEST只在最终模型及runtime身份锁定后执行，不用于反馈或挑checkpoint。

下一步是第五天：先用冻结环境、输入和runtime对两臂最长输入做隔离的前向、反向及一次梯度累积更新检查，丢弃试运行权重；检查通过后才独立训练A1/B1。当前training_start_go=false，不需要继续出题或追加同类诊断来替代训练验证。

数据：[TRAIN_POOL_A.jsonl](TRAIN_POOL_A.jsonl)、[TRAIN_POOL_B.jsonl](TRAIN_POOL_B.jsonl)。准入与预算：[DECISION.json](DECISION.json)、[TRAINING_BUDGET_FREEZE.json](TRAINING_BUDGET_FREEZE.json)。运行入口与具体命令见[runtime说明](runtime/README_ZH.md)。
"""
    (OUT / "REPORT_ZH.md").write_text(report)
    manifest_files = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                      and "navigation_history" not in p.parts and p.name != "FINAL_MANIFEST.json"
                      and p.suffix != ".log"]
    save("FINAL_MANIFEST.json", {"schema_version": "day4_schedule_manifest_v1", "files": [bind(p) for p in sorted(manifest_files)],
         "scope": "CPU-only actual schedule; training and inference not executed"})
    history = OUT / "navigation_history"
    history.mkdir()
    for name in ["CURRENT_STATUS.json", "DAY4_LATEST.json", "README_ZH.md"]:
        if (WEEK / name).exists():
            shutil.copy2(WEEK / name, history / name)
    now = datetime.datetime.now(datetime.timezone.utc)
    next_action = read(OUT / "DECISION.json")["next_action"]
    latest = {"schema_version": "selfplay_day4_latest_v2", "updated_at_utc": now.isoformat(),
              "status": read(OUT / "DECISION.json")["status"],
              "decision": bind(OUT / "DECISION.json"), "report": bind(OUT / "REPORT_ZH.md"),
              "budget": bind(OUT / "TRAINING_BUDGET_FREEZE.json"), "manifest": bind(OUT / "FINAL_MANIFEST.json"),
              "previous_common_admission": bind(WEEK / "day4_common_set_admission_v1/DECISION.json"),
              "training_start_go": False, "next_action": next_action}
    (WEEK / "DAY4_LATEST.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n")
    status = read(WEEK / "CURRENT_STATUS.json")
    status.update(status=latest["status"], updated_at_utc=now.isoformat(),
                  updated_at_beijing=now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  next_action=next_action)
    status["training"].update(new_AB_pools_frozen=True, new_AB_training_runs=0,
                              actual_unique_questions_per_arm=112, actual_foundation_selected_per_arm=24,
                              actual_new_structure_selected_per_arm=88, actual_legacy_selected_per_arm=0,
                              actual_optimizer_steps_per_arm=56, training_start_go=False)
    status["daily_planning"].update(day4="D4_SCHEDULE_PASS_CPU_FROZEN", day5="ISOLATED_RESOURCE_SMOKE_PENDING_NOT_STARTED")
    status["before_execution_freeze_remaining"] = [
        "DAY5 isolated longest-input GPU resource smoke under the frozen input/runtime/package bindings before formal training.",
        "Actual trained adapter/merged hashes and per-model public evaluation execution manifest before Solver; TEST final-only."]
    status["day4_schedule_progress"] = {**latest, "actual_arms": summary["arms"], "new_solver_calls": 0,
                                        "new_training_runs": 0, "gpu_smoke_runs": 0,
                                        "latest": bind(WEEK / "DAY4_LATEST.json")}
    (WEEK / "CURRENT_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    readme = WEEK / "README_ZH.md"
    existing = readme.read_text() if readme.exists() else ""
    readme.write_text("当前进度（第四天）：实际A/B池各112题、两轮展示、56优化器步已完成CPU冻结；"
                      "尚未训练，下一步为第五天隔离资源试运行。见[第四天报告](day4_schedule_v1/REPORT_ZH.md)"
                      "与[DAY4_LATEST.json](DAY4_LATEST.json)。\n\n" + existing)
    print(json.dumps({"status": latest["status"], "training_start_go": False, "next_action": next_action}))


if __name__ == "__main__":
    main()
