"""Summarize completed repair artifacts without editing frozen old results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def main():
    replay = HERE / "evaluation/corrected_replay"
    production = HERE / "production"
    probe_dir = production / "controlled_anchor_live_probe_24719_generic_v1"
    probe = read(probe_dir / "RESULT.json")
    pipeline = read(replay / "REPLAY_STATUS.json")
    assert pipeline["status"] == "COMPLETED"
    audits = read(HERE / "evaluation/CORRECTED_REPLAY_INDEPENDENT_CHECK.json")
    assert audits["status"] == "PASS_COMPLETED_CORRECTED_REPLAY_WITH_EXPLICIT_LIMITS"
    rows = []
    scores = {}
    for role in ("B1", "B1_GENERIC", "B1_TARGETED"):
        summary = read(replay / "final_evaluation" / role / "SUMMARY.json")
        by_task = summary["metrics"]["by_task"]
        rows.append({"model": role, "main_mc": by_task["MC2_5_PRIMARY"],
                     "ordering": by_task["ORDERING"], "mc6_control": by_task["MC6_CONTROL"],
                     "overall": summary["metrics"]["overall"],
                     "summary": binding(replay / "final_evaluation" / role / "SUMMARY.json")})
        scores[role] = {item["request_id"]: item for item in map(json.loads,
            (replay / "final_evaluation" / role / "SCORES.jsonl").read_text().splitlines())}
    transitions = {}
    for role in ("B1_GENERIC", "B1_TARGETED"):
        transitions[role] = {}
        assert set(scores[role]) == set(scores["B1"])
        for task in ("MC2_5_PRIMARY", "ORDERING", "MC6_CONTROL"):
            def belongs(row):
                if task == "ORDERING":
                    return row["question_type"] == "ordering"
                return row["question_type"] == "multi_select" and (
                    2 <= row["gold_cardinality"] <= 5 if task == "MC2_5_PRIMARY" else row["gold_cardinality"] == 6)
            keys = [key for key, row in scores["B1"].items() if belongs(row)]
            transitions[role][task] = {"requests": len(keys),
                "B1_correct_to_wrong": sum(bool(scores["B1"][key]["exact"]) and not bool(scores[role][key]["exact"]) for key in keys),
                "B1_wrong_to_correct": sum(not bool(scores["B1"][key]["exact"]) and bool(scores[role][key]["exact"]) for key in keys)}
    training = {}
    for arm in ("GENERIC", "TARGETED"):
        result = read(replay / "runs" / arm / "RESULT.json")
        assert result["completed_optimizer_steps"] == 24 and result["completed_microsteps"] == 96
        assert result["dev_steps"] == [0, 20, 24]
        training[arm] = {key: result[key] for key in ("completed_optimizer_steps", "completed_microsteps",
                    "initial_lora_sha256", "mean_train_loss", "source_truncation", "dev_steps", "wandb", "adapter")}
        training[arm]["unique_questions"] = 24
        training[arm]["display_permutations"] = 2
        training[arm]["epochs"] = 2
        training[arm]["input_pool"] = "OLD_QFP2_FALLBACK_ONLY_NOT_NEW_ROOT_PRODUCTION"
    assert training["GENERIC"]["initial_lora_sha256"] == training["TARGETED"]["initial_lora_sha256"]
    live_old = read(production / "LIVE_PROBE_REPORT.json")
    regression = read(production / "REPAIR_REPORT_V3.json")
    evidence_paths = [HERE / "evaluation/MODEL_COMPAT_CPU_REPORT.json",
        HERE / "evaluation/CONFIG_BRIDGE_AMENDMENT_V2.json",
        HERE / "evaluation/HISTORICAL_IMPACT_AUDIT.json",
        HERE / "evaluation/CORRECTED_REPLAY_INDEPENDENT_CHECK.json",
        replay / "CORRECTED_COMPARISON.json", HERE / "REVIEW_INDEPENDENT.json",
        production / "REPAIR_REPORT_V3.json", production / "LIVE_PROBE_REPORT.json",
        production / "CONTROLLED_ANCHOR_CPU.log", production / "CONTROLLED_ANCHOR_LIVE_REPORT.json",
        probe_dir / "PROTOCOL.json", probe_dir / "RESULT.json"]
    report = {"schema_version": "oet1_followup_repair_closeout_v1",
        "status": "REPAIR_AND_FIXED_BUDGET_REPLAY_COMPLETED_MAIN_PRODUCTION_OBJECTIVE_INCOMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_new_root_objective_complete": False, "bulk_production_resumed": False,
        "new_questions_this_repair": 0, "new_training_pool_admissions_this_repair": 0,
        "confirmed_runtime_cause": {"field": "rope_theta", "incorrect_effective_value": 10000,
            "correct_effective_value": 1000000, "runtime": "transformers 4.57.3",
            "trigger": "merged config stored theta under 5.9 rope_parameters schema",
            "repair": "validated in-memory config bridge; original checkpoint config unchanged",
            "old_oet1_training_and_scoring_valid_for_accuracy_claims": False,
            "B1_original_training_affected": False,
            "all_historical_experiments_invalidated": False},
        "corrected_replay": {"scope": "historically_exposed_common_DEV_not_TEST",
            "unique_dev_questions": 42, "displays_per_question": 2,
            "main_mc_unique_questions": 21, "ordering_unique_questions": 16, "mc6_control_unique_questions": 5,
            "models": rows, "training": training,
            "paired_prediction_transitions": transitions,
            "training_improves_common_dev": False,
            "feedback_causal_gain_established": False,
            "toy_short_control_exact": {"correct": 0, "total": 2},
            "short_control_failures_are_preserved": True},
        "production_repairs": {"cpu_regression_tests": regression["tests_run"],
            "claimed_historical_protection_scope": "873 specified files plus separately enumerated old repair artifacts; not all historical directories",
            "old_free_quote_probe": {key: live_old[key] for key in ("model_calls", "elapsed_seconds", "scenes_attempted",
                "scenes_accepted", "rejection_code", "mismatched_quote_anchors")},
            "controlled_anchor_cpu_tests": 6, "prospective_controlled_anchor_probe": probe,
            "controlled_anchors_scope": "A:S01 one new author and one new reviewer; no B/native/multiple scenes/roots/questions",
            "controller_selects_evidence_or_changes_verdicts": False,
            "quote_protocol_success_establishes_semantic_accuracy": False},
        "remaining_work": ["Replace long copied hash-suffixed IDs with compact ordinal anchor selections and locally validated schema enums in a new prospective protocol; preserve this failed probe",
            "Only after prospective success integrate controlled citations into unified multi-scene source runner",
            "Verify both extension routes and question generation on multiple roots without weakening source admission",
            "Investigate fixed-budget training regression before selecting a new model; retain B1 as unchanged reference",
            "Only then resume new-root balanced production and matched training/evaluation"],
        "artifacts": [binding(path) for path in evidence_paths]}
    write_path = HERE / "REPAIR_CLOSEOUT.json"
    assert not write_path.exists(), "Write a new report revision instead of overwriting closeout"
    write_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["本轮已修复运行配置错误，并完成两组等预算重训与共同 Dev 重评。新 roots 的扩充加出题主目标仍未完成，尚未恢复批量生产。", "",
        "确认的根因：B1 合并模型的配置由 Transformers 5.9 导出，RoPE theta 存在嵌套字段；过夜脚本用 4.57 加载时忽略了它，把 1,000,000 错当成 10,000。现在按检查后的配置在内存中桥接，未改 B1 原文件。原 OET1 训练出的两份 adapter 及其准确率结果不能作为有效实验使用；B1 原训练和有 5.9 运行证据的历史结果不因此作废。", "",
        "以下为完全正确率，每模型评估相同 42 道 Dev 题、每题两个选项排列，共 84 次；各模型格式有效率均为 100%。MC6 为全选控制题，单独报告。", "",
        "| 模型 | 多选 2–5 gold（42次） | 排序（32次） | MC6控制（10次） |",
        "|---|---:|---:|---:|"]
    names = {"B1": "B1 起点", "B1_GENERIC": "普通组重训", "B1_TARGETED": "定向组重训"}
    for row in rows:
        lines.append(f"| {names[row['model']]} | {100*row['main_mc']['exact']:.2f}% | {100*row['ordering']['exact']:.2f}% | {100*row['mc6_control']['exact']:.2f}% |")
    lines += ["", "两组都使用旧 QFP2 的各 24 道独立训练题，2种排列×2个epoch，完成96个microsteps、24次参数更新；在第0、20、24步评估Dev，未按Dev择优换checkpoint。定向组多选高于普通组，但两组在共同Dev均未超过B1。这是小样本、单种子、已多次使用的Dev回归结果，不能证明反馈的因果收益，也不能据此断定所有定向训练无效。W&B指标已保存在各训练目录，模式为offline，尚未云端同步。", "",
        "独立代码与产物复核163/163检查通过：两组step0与B1、step24与各自final在42条共享请求上输出完全一致，评分独立重算一致，初始LoRA和最终adapter哈希吻合。另加的两道简短指令控制仍为0/2，保留失败，不能把本任务工程检查通过解释成通用指令遵循可靠。", "",
        f"生产工程修复通过{regression['tests_run']}项CPU回归，覆盖日志、累计计数、限定native引文域、审核计数、正常完成校验和空白quote拒绝。历史保护只涵盖已列举的873个文件及另行枚举的旧修复文件，并非全目录OS只读保护。", "",
        "第一场新生产小试实际调用2次、约111秒，扩充14条turn，审核数202正确；但审核把实际T011的原句标成T013，被严格拒绝。0场准入，后5场未做，没有自动改指针放行。这只证明引用定位错误，不能说该扩充内容已被证明错误。", "",
        "为减少这类机械错配，新协议把原始turn的ref、speaker、整段原文绑定成确定的锚点。审核者仍需读全量对话、自己挑相关锚点并逐维解释；程序仅展开其所选锚点，不选择证据、不改判断。raw审核、catalog、展开结果和映射分别保存。旧失败不重分类。6项CPU测试覆盖非法锚点、换catalog、跨native域、空白、自由quote覆盖和语义失败拒绝。", "",
        f"新的单场真实小试结果：{probe['status']}，耗时255.31秒、2次正常完成的模型调用。审核返回READY，catalog绑定正确，32条维度引用均为已有锚点；但warning中的7条引用有1条给长ID多抄了一个字符，被UNKNOWN_REVIEW_ANCHOR_ID严格拒绝。没有自动纠正、没有准入，也不能据此宣称受控引用已稳定。它仅覆盖同一root的A:S01，不能外推到B路线native、多场、多root或出题质量。", "",
        "下一步生产修复应集中于短序号锚点和动态schema enum约束，同时保留catalog内容绑定和本地拒绝规则；必须另立新协议前瞻验证，不修正本次失败的ID。CLI显示支持output-schema参数，但当前后端尚未实测这条调用链。之后才应把引用机制接入完整两路线小批量。", "",
        "当前保留B1作为比较起点；训练回归需要在冻结数据与预算下进一步定位。本轮新增训练题为0，批量生产尚未恢复，新roots的扩充加出题目标仍未完成。", "",
        "完整机器记录：[REPAIR_CLOSEOUT.json](REPAIR_CLOSEOUT.json)；正确配置重放：[evaluation/corrected_replay/CORRECTED_COMPARISON.json](evaluation/corrected_replay/CORRECTED_COMPARISON.json)；新小试：[production/controlled_anchor_live_probe_24719_generic_v1/RESULT.json](production/controlled_anchor_live_probe_24719_generic_v1/RESULT.json)。"]
    (HERE / "REPAIR_CLOSEOUT_ZH.md").write_text("\n".join(lines) + "\n")
    print(write_path)


if __name__ == "__main__":
    main()
