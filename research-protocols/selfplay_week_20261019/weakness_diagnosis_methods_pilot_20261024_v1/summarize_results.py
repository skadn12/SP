"""Summarize the completed pilot without editing frozen inputs or model outputs."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def write(name, obj):
    target = P / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    scores = read(P / "DIAGNOSIS_SCORES.json")["records"]
    solver = read(P / "SOLVER_SCORES.json")
    selected = read(P / "SELECTION_PRIVATE.json")["cases"]
    mapping = {row["candidate_id"]: row for row in read(P / "BLIND_MAPPING_PRIVATE.json")["records"]}
    required = read(P / "REVIEW_CONTRACT.json")["per_card_required_fields"]
    reviews = []
    review_bindings = []
    for reviewer in (1, 2):
        path = P / f"semantic_review/reviewer_{reviewer}/REVIEWS.json"
        data = read(path)
        review_bindings.append(binding(path))
        for row in data.get("records", data.get("cases", [])):
            for card in row.get("cards", row.get("reviews", [])):
                assert set(required).issubset(card), (row["candidate_id"], card)
                assert card["anonymous_label"] in ("X", "Y")
                arm = mapping[row["candidate_id"]][card["anonymous_label"]]
                reviews.append({"candidate_id": row["candidate_id"], "arm": arm,
                                "reviewer": reviewer, **card})
    index = {(row["candidate_id"], row["arm"]): row for row in reviews}
    assert len(reviews) == len(index) == 12
    assert set(index) == {(row["candidate_id"], row["arm"]) for row in scores}
    by_arm = {}
    per_card = []
    for score in scores:
        review = index[(score["candidate_id"], score["arm"])]
        usable = bool(score["contract_valid"] and review["source_grounded"] == "PASS"
                      and review["trajectory_claims_supported"] == "PASS"
                      and review["specific_observed_error"] is True
                      and review["generalization_calibrated"] is True
                      and review["task_instruction_actionable"] is True
                      and review["actionability_score"] >= 1)
        assert usable == review["usable_candidate"], (score["request_id"], "eligibility disagrees")
        per_card.append({"request_id": score["request_id"], "candidate_id": score["candidate_id"],
                         "arm": score["arm"], "contract_valid": score["contract_valid"],
                         "semantic_review": review, "raw_binding": score["raw_binding"]})
    for arm in ("A", "B"):
        rows = [row for row in scores if row["arm"] == arm]
        checks = [row for row in reviews if row["arm"] == arm]
        by_arm[arm] = {
            "diagnosis_requests": len(rows),
            "diagnosis_contract_valid": sum(row["contract_valid"] for row in rows),
            "diagnosis_input_tokens": sum(row["input_tokens"] for row in rows),
            "diagnosis_output_tokens": sum(row["output_tokens"] for row in rows),
            "all_quotes_verbatim": sum(row["mechanical"].get("all_quotes_verbatim", False) for row in rows),
            "correct_trace_availability_claim_matches": sum(row["mechanical"].get("success_availability_claim_matches", False) for row in rows),
            "success_id_valid": sum(row["mechanical"].get("success_id_valid", False) for row in rows),
            "source_grounded_pass": sum(row["source_grounded"] == "PASS" for row in checks),
            "trajectory_claims_supported_pass": sum(row["trajectory_claims_supported"] == "PASS" for row in checks),
            "specific_observed_error": sum(row["specific_observed_error"] is True for row in checks),
            "generalization_calibrated": sum(row["generalization_calibrated"] is True for row in checks),
            "task_instruction_actionable": sum(row["task_instruction_actionable"] is True for row in checks),
            "actionability_score_total": sum(row["actionability_score"] for row in checks),
            "usable_candidates": sum(row["usable_candidate"] is True for row in checks),
            "new_solver_requests": 24 if arm == "B" else 0,
            "new_solver_input_tokens": sum(row["input_tokens"] for row in solver["records"]) if arm == "B" else 0,
            "new_solver_output_tokens": sum(row["output_tokens"] for row in solver["records"]) if arm == "B" else 0,
        }
        by_arm[arm]["total_new_calls_excluding_engineering_failure"] = len(rows) + by_arm[arm]["new_solver_requests"]
        by_arm[arm]["total_new_input_tokens"] = by_arm[arm]["diagnosis_input_tokens"] + by_arm[arm]["new_solver_input_tokens"]
        by_arm[arm]["total_new_output_tokens"] = by_arm[arm]["diagnosis_output_tokens"] + by_arm[arm]["new_solver_output_tokens"]
    freezes = {name: read(P / name / "RAW_RESPONSE_FREEZE.json")
               for name in ("solver", "diagnosis", "diagnosis_compat_v2")}
    old_manifest = read(P / "PREPARATION.json")["old_manifest_binding"]
    result = {
        "schema_version": "diagnostic_methods_pilot_comparison_v1",
        "scope": {"questions": 6, "roots": 4, "multi_select": 4, "ordering": 2,
                  "failure_conditioned_selection": True, "same_solver_and_diagnoser": "local B1 Qwen3-4B finetune",
                  "enable_thinking": False, "formal_reproduced_weaknesses": 0,
                  "challenger_generations": 0, "training_runs": 0,
                  "reviewers": "Two separate AI reviewers, disjoint question subsets; hidden X/Y labels, not strict blind review"},
        "arms": by_arm,
        "solver_new_sampling": {"requests": len(solver["records"]),
                                "contract_valid": sum(row["strict"]["valid"] for row in solver["records"]),
                                "strict_correct": sum(row["strict"]["valid"] and row["strict"]["exact"] for row in solver["records"]),
                                "questions_with_correct_wrong_contrast": sum(row["has_correct_and_wrong"] for row in solver["per_question"]),
                                "per_question": solver["per_question"],
                                "not_population_accuracy": True},
        "shared_historical_failure_cost": {
            "solver_calls": len(selected),
            "input_tokens": sum(row["old_row"]["prompt_tokens"] for row in selected),
            "output_tokens": sum(row["base_score"]["output_tokens"] for row in selected),
            "reused_not_new_calls": True,
            "production_logic_per_question": {"A": {"solver": 1, "diagnosis": 1}, "B": {"solver": 5, "diagnosis": 1}}},
        "execution_accounting": {
            "planned_terminal_records_across_stages": sum(row["requests"] for row in freezes.values()),
            "actually_submitted": sum(len(row["submitted_ids"]) for row in freezes.values()),
            "returned_outputs": sum(row["returned"] for row in freezes.values()),
            "unique_new_normal_request_slots": 36,
            "diagnosis_logical_slots": 12,
            "failed_engineering_submitted": len(freezes["diagnosis"]["submitted_ids"]),
            "failed_engineering_not_submitted": freezes["diagnosis"]["status_counts"]["NOT_SUBMITTED"],
            "stage_wall_seconds": {name: row["wall_seconds"] for name, row in freezes.items()},
            "total_stage_wall_seconds": sum(row["wall_seconds"] for row in freezes.values()),
            "successful_stage_wall_seconds": freezes["solver"]["wall_seconds"] + freezes["diagnosis_compat_v2"]["wall_seconds"],
            "wall_time_scope": "GPU stage wall clock including loading and closure; excludes development and semantic review. Mixed A/B batches do not support per-arm latency.",
            "engineering_recovery": "Generation grammar omitted uniqueItems only; original final validation preserved. Original failed outputs retained. No successful-answer selection or automatic retry."},
        "per_card": per_card,
        "bindings": {"reviews": review_bindings, "scores": [binding(P / "SOLVER_SCORES.json"), binding(P / "DIAGNOSIS_SCORES.json")],
                     "review_contract": binding(P / "REVIEW_CONTRACT.json"), "mapping": binding(P / "BLIND_MAPPING_PRIVATE.json"),
                     "old_manifest": old_manifest},
        "interpretation": {
            "implementation": "A uses fewer calls and tokens. B required 24 extra solver samples and yielded a true correct/wrong contrast for only one question.",
            "diagnostic_quality": "Use independent source/trajectory review, not JSON validity, to assess generated cards. Findings apply to this B1 diagnoser and selected small sample.",
            "not_established": ["General superiority of either diagnosis method", "Stable or transferable model weaknesses", "Challenger targeting benefit", "Feedback training causal gain"],
            "next_recommendation": "First repair diagnosis reliability using the existing Challenger or a more reliable diagnoser on the same frozen materials; prefill trace correctness mechanically. Prefer A as low-cost candidate generation, reserve B for bounded ambiguity checks. Do not promote current cards automatically."}}
    write("COMPARISON.json", result)
    write("semantic_review/SUMMARY.json", {"review_bindings": review_bindings, "arms": by_arm, "cards": reviews})

    A, B = by_arm["A"], by_arm["B"]
    table = ["| 指标 | A：单条失败诊断 | B：多次采样对照诊断 |", "|---|---:|---:|"]
    for label, key in [("新Solver调用", "new_solver_requests"), ("诊断调用", "diagnosis_requests"),
                       ("诊断合同有效（分母6）", "diagnosis_contract_valid"),
                       ("来源支持通过（分母6）", "source_grounded_pass"),
                       ("轨迹陈述支持通过（分母6）", "trajectory_claims_supported_pass"),
                       ("包含具体可观察错误（分母6）", "specific_observed_error"),
                       ("整卡可用候选（分母6）", "usable_candidates"),
                       ("新增输入tokens", "total_new_input_tokens"),
                       ("新增输出tokens", "total_new_output_tokens")]:
        table.append(f"| {label} | {A[key]:,} | {B[key]:,} |")
    report = f"""# 两种弱点诊断方法小试结果

## 结论

A在操作上更省：本次复用旧失败后只需6次诊断；B需要24次额外作答和6次诊断。B的6题里只有1题获得真正的正误轨迹对照，其余5题只能比较多条失败。独立审核确认A有{A['usable_candidates']}/6、B有{B['usable_candidates']}/6整卡符合候选准入。当前B1诊断器存在错误算术、错误来源解释和虚构成功轨迹，不能把格式完整的JSON直接用作可靠弱点反馈。

这能比较本次的实施成本，尚不能证明某一种诊断方法一般更有效。整卡不合格不等于其中每一句观察都错；具体正确观察及不能准入的原因保存在逐卡审核中，未人工修卡后重计成功。

## 固定设计

6道既有失败题、4个Train roots，4道数量多选和2道钟面排序。两组Solver及诊断器均为本地B1（Qwen3-4B微调），thinking关闭，保留逐项理由、引用及最终答案。A看一条旧失败，B看同一失败及4次新采样。两组诊断prompt模板、schema、配对seed及输出预算一致，只有轨迹数量不同。

采用失败条件选择：同题THINK_OFF/FULL按perm_00、perm_01取首个合同有效错误显示。218519-ORDER1为perm_01，其余为perm_00。选题和旧实验的原审核显示并非完全相同；没有根据新结果换题。样本不能用于估计总体模型正确率。

## 成本与质量

{chr(10).join(table)}

表中tokens只统计正常新增返回的输入/输出，历史共同输入和工程失败另列，不代表含失败的全部实际算力或API费用。

24次新采样中23次合同有效、3次严格答对；3次答对都在218519-ORDER1。该题另一次排序漏项，其余5题各4次均未严格答对。没有继续采样直到成功，也没有借用thinking开启或另一排列的成功回答。

引文全部逐字匹配：A {A['all_quotes_verbatim']}/6、B {B['all_quotes_verbatim']}/6；是否存在正确轨迹的声明匹配：A {A['correct_trace_availability_claim_matches']}/6、B {B['correct_trace_availability_claim_matches']}/6。这些机械指标分别描述局部错误，不替代整卡语义审核。A所有输入轨迹均为已知失败，因此正确做法应是明确无成功对照，不能给它虚构成功ID。

历史共同基础为6次旧失败：输入{result['shared_historical_failure_cost']['input_tokens']:,}、输出{result['shared_historical_failure_cost']['output_tokens']:,} tokens，本轮没有重新调用。生产逻辑上A每题需要1次Solver+1次诊断，B需要5次Solver+1次诊断，不能宣传A的作答成本为零。本次正常新增输入成本B/A为{B['total_new_input_tokens']/A['total_new_input_tokens']:.2f}倍，输出为{B['total_new_output_tokens']/A['total_new_output_tokens']:.2f}倍；不是货币成本或时间比。

## 独立语义审核

两名独立AI审核员按题分工各审6卡，先读完整共同来源和所有轨迹，再核对卡片。X/Y隐藏A/B名称，但轨迹数量可能泄露方法，因此不是严格盲审、人工审核或每卡双人复核。合同有效、来源支持、轨迹陈述支持、具体错误、校准外推及可执行建议全部过关才算可用候选；候选也不等于已正式复现的弱点。

逐题两组真实卡片及对应审核见EXAMPLES_ZH.md；完整评价见semantic_review/SUMMARY.json。所有失败保留，无强模型代写修卡，也没有把局部正确观察自动晋升为正式弱点。

三个具体发现：

- 218519-MC2：A把15:00到20:00说成17小时；B也未正确纠正。按题目忽略日期的规则应为5小时。两卡还把错误轨迹称为成功，说明诊断器会复述Solver的错算。
- 218519-ORDER1：B确实看到了最终正确的对照轨迹，但该轨迹的理由仍误把票价13.28当作13:28发车。最终答案正确不保证其解释或引用正确。
- 229686-MC3：B把07:55到08:15说成10分钟，实际为20分钟；多条轨迹有分歧是可观察事实，但诊断卡把正确方向反了。

因此，本轮直接观察到B1诊断器在事实核验和轨迹标签上出错。未测试延长思考文本或更换诊断器能否改善，也未测试更换诊断器后两种方法是否会分出优劣。

## 工程失败和耗时

第一轮诊断因grammar不支持uniqueItems失败，0份答案返回，8次已提交标记BACKEND_ERROR、4次未提交。恢复仅从生成grammar删除该关键字，最终合同仍验证引用不重复；prompt、seed和模型不变。独立工程检查已核验，原失败raw和日志完整保存。

原失败8次提交的计划输入共74,956 tokens，实际后台处理token及算力未观测，不能记为实际零成本；另4次未提交的计划输入为38,106 tokens。

三阶段墙钟：Solver {freezes['solver']['wall_seconds']:.2f}秒；诊断失败 {freezes['diagnosis']['wall_seconds']:.2f}秒；兼容诊断 {freezes['diagnosis_compat_v2']['wall_seconds']:.2f}秒；合计{result['execution_accounting']['total_stage_wall_seconds']:.2f}秒（约2分25秒）。成功两阶段约{result['execution_accounting']['successful_stage_wall_seconds']:.2f}秒。这包括各阶段加载和收尾，不含编码、准备和审核，不能当作端到端交付耗时或分摊成A/B独立延迟。

跨三阶段有48条计划终态、44次实际提交、36份实际返回；诊断逻辑槽位只有12，兼容恢复不增加独立样本数。失败工程成本单列，不能混入正常A/B诊断效果比较，也不能隐去。

## 对主流程的建议

1. 若继续开发，先以A作为低成本候选诊断的实验入口；B只对少数需要区分稳定错答与偶发错答的题做有上限的复核。当前A卡片也未通过，尚不具备直接进入生产的资格。
2. 当前首先要修复诊断端可靠度：让现有Challenger或更可靠的诊断器在同一冻结材料上验证，而不是继续增加Solver采样。机械预填每条轨迹的对错和合法ID，并独立审核来源、算术及任务建议，防止编造成功对照。
3. 本轮没有生成新Challenger题、正式复现新弱点或训练；不能据此声称反馈闭环或训练收益已成立。当前卡片不自动进入正式弱点库、训练池或生产任务。

## 复查入口

- PROTOCOL_ZH.md：冻结方案。
- COMPARISON.json：逐题结果、成本和审核绑定。
- EXAMPLES_ZH.md：6题A/B完整真实卡片及审核，含失败。
- solver/raw/：24次新作答完整输出。
- diagnosis/raw/：第一次工程失败记录。
- diagnosis_compat_v2/raw/：12次诊断真实原始输出。
- SEMANTIC_REVIEW_PACKET.json：共同来源、问题、完整轨迹和匿名卡片。
- semantic_review/reviewer_1/、reviewer_2/：独立审核记录。

所有本轮新增产物只在本目录；旧实验、Gold、模型权重和正式训练池未作修改。最终文件哈希及旧依赖完整性以FINAL_MANIFEST.json、FINAL_MANIFEST_CHECK.json为准。
"""
    (P / "REPORT_ZH.md").write_text(report)
    examples = ["# 全部6题、12张真实诊断卡与审核", "", "以下原样展示解析卡片，不修正其中错误。完整原始文本和被截断输出保存在raw；卡片无法解析时仅展示原始文本。"]
    for case in selected:
        cid = case["candidate_id"]
        row = case["old_row"]
        examples += ["", f"## {cid}（{row['variant_id']}）", "", row["question"], "",
                     *[f"- {label}: {value}" for label, value in row["display"].items()], "",
                     "显示Gold：" + " ".join(case["base_score"]["strict"]["display_gold"]),
                     "旧失败预测：" + " ".join(case["base_score"]["strict"]["display_prediction"])]
        for arm in ("A", "B"):
            score = next(item for item in scores if item["candidate_id"] == cid and item["arm"] == arm)
            rev = index[(cid, arm)]
            examples += ["", f"### 方法{arm}", "", f"合同有效：{score['contract_valid']}；整卡候选可用：{rev['usable_candidate']}。", "", "```json"]
            if score.get("card") is not None:
                examples += [json.dumps(score["card"], ensure_ascii=False, indent=2)]
            else:
                raw = read(score["raw_binding"]["path"])
                examples += [str(raw.get("decoded_output_skip_special_tokens_false", raw.get("raw_output", raw)))]
            examples += ["```", "", "审核原文：", "", "```json", json.dumps(rev, ensure_ascii=False, indent=2), "```", "",
                         "原始输出：" + score["raw_binding"]["path"]]
    (P / "EXAMPLES_ZH.md").write_text("\n".join(examples) + "\n")
    (P / "README_ZH.md").write_text("# 两种弱点诊断方法小试交付\n\n先读REPORT_ZH.md查看结论，EXAMPLES_ZH.md查看全部真实卡片和失败。COMPARISON.json为机器可读汇总，SEMANTIC_REVIEW_PACKET.json含完整共同来源与轨迹。\n\n本轮6题、4个Train roots，只比较本地B1下的诊断可行性，不含新训练或正式弱点复现。原始失败与兼容恢复均保留。FINAL_STATUS.json记录完成状态，FINAL_MANIFEST_CHECK.json记录完整性验证。\n")
    write("FINAL_STATUS.json", {"status": "PILOT_COMPLETED", "model_calls_complete": True,
                               "semantic_review_complete": True, "candidate_cards_promoted": 0,
                               "formal_weaknesses_added": 0, "training_runs": 0,
                               "new_challenger_questions": 0,
                               "scope_limit": "Implementation comparison completed; diagnostic effectiveness and feedback benefit not established.",
                               "summary_binding": binding(P / "COMPARISON.json"),
                               "report_binding": binding(P / "REPORT_ZH.md")})
    print(json.dumps({"arms": by_arm, "execution": result["execution_accounting"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
