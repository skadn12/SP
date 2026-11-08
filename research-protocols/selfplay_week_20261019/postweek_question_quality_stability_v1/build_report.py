"""Render the frozen cross-root validation without changing questions or scores."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stability_report_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
read, save = runner.read, runner.save


def write_new(path, text):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def fraction(correct, count):
    return f"{correct}/{count}（{correct/count:.1%}）" if count else "无样本"


def main():
    candidates = read(HERE / "CANDIDATES_PRIVATE.json")
    reviews = read(HERE / "CONTENT_REVIEW_AI.json")["reviews"]
    summary = read(HERE / "execution/ANALYSIS_SUMMARY.json")
    scores = read(HERE / "execution/SCORES.json")["records"]
    rows = read(HERE / "execution/PUBLIC_REQUESTS.json")["requests"]
    reasons = {r["request_id"]: r for r in read(HERE / "execution/REASON_REVIEW_PACKET.json")["requests"]}
    raw_freeze = read(HERE / "execution/RAW_RESPONSE_FREEZE.json")
    mechanical = read(HERE / "audit/MECHANICAL_AUDIT.json")
    execution = read(HERE / "audit/EXECUTION_AUDIT.json")
    assert mechanical["status"].startswith("PASS"), "MECHANICAL_AUDIT_NOT_PASS"
    assert execution["status"].startswith("PASS"), "EXECUTION_AUDIT_NOT_PASS"
    counts = Counter(r["decision"] for r in reviews)
    cues = []
    overlaps = []
    for c in candidates:
        flags = runner.gate.lint_candidate(c)["flags"]
        cues.extend({"candidate_id": c["candidate_id"], **f} for f in flags)
        if c["question_type"] == "ordering" and c.get("ordering_analysis", {}).get("mention_order_overlap"):
            overlaps.append(c["candidate_id"])
    invalids = Counter(s.get("invalid_error") for s in scores if not s["valid"])
    both_full = [p for p in summary["variant_pairs"] if p["condition"] == "FULL"]
    verdicts = {
        "overall": "MECHANICAL_CHECKS_PASS_BUT_ALL_FOUR_FIXES_NOT_YET_CERTIFIED_STABLE",
        "shuffle_and_answer_mapping": "PASS_THIS_BATCH",
        "gold_2_to_6_coverage": "PASS_ALL_FOUR_ROOTS",
        "final_source_supported_content": "PASS_INDEPENDENT_AI_REVIEW_AFTER_ONE_REPAIR",
        "first_draft_combined_stability": "PASS" if summary["initial_content_stability_threshold_met"] else "FROZEN_90_PERCENT_THRESHOLD_NOT_MET",
        "editorial_ordering_cues": "NO_LINT_FLAGS_FINAL_SEMANTIC_REVIEW_PASS" if not cues else "SEE_REVIEWED_FLAGS",
        "semantic_ordering_shortcuts": "NOT_ELIMINATED_OR_CERTIFIED_ABSENT",
        "solver_output_validity": f"{summary['valid']}/{summary['requests']}",
        "full_accuracy_changed_between_variants": sum(p["accuracy_changed"] for p in both_full),
        "next_scope_recommendation": "QUESTION_VALIDATION_REPAIRS_BEFORE_EXPANSION",
    }
    validation = {
        "status": "CROSSROOT_VALIDATION_COMPLETE_SEMANTIC_SHORTCUTS_REQUIRE_SEPARATE_JUDGMENT",
        "planned_candidates": 28, "content_decisions": dict(counts),
        "first_review_pass": summary["initial_content_pass"],
        "first_review_pass_rate": summary["initial_content_pass_rate"],
        "first_review_pass_definition": "Combined content and source-binding review, retaining the frozen 90% threshold and denominator 28.",
        "first_review_semantic_content_pass": summary["initial_semantic_content_pass"],
        "first_review_semantic_content_pass_rate": summary["initial_semantic_content_pass_rate"],
        "first_review_stability_threshold_met": summary["initial_content_stability_threshold_met"],
        "coverage_by_root_complete": {rid: c["coverage_complete"] for rid, c in summary["coverage_by_root"].items()},
        "gold_histogram": summary["new_mc_coverage"]["gold_count_histogram"],
        "mechanical_audit": runner.bind(HERE / "audit/MECHANICAL_AUDIT.json"),
        "execution_audit": runner.bind(HERE / "audit/EXECUTION_AUDIT.json"),
        "visible_reason_review": runner.bind(HERE / "audit/VISIBLE_REASON_REVIEW_AI.json"),
        "lint_flags": cues, "ordering_mention_order_overlap": overlaps,
        "question_only_both_variants_correct": summary["question_only_both_variants_correct"],
        "question_only_both_correct_excluding_six_gold": [cid for cid in summary["question_only_both_variants_correct"]
            if not any(c["candidate_id"] == cid and c["question_type"] == "multi_select" and len(c["gold"]) == 6 for c in candidates)],
        "invalid_response_reasons": dict(invalids), "new_dialogue_turns": 0,
        "new_training_runs": 0, "training_admission": False,
        "content_review_is_ai_not_human": True, "gold_blind_review": False,
        "source_dependence_proven": False, "population_generation_stability_proven": False,
        "verdicts": verdicts,
    }
    save(HERE / "ROOT_VALIDATION.json", validation)
    lines = ["# 四项修复跨 root 稳定性验证", "",
        "结论：排序打乱、答案映射及23456 Gold覆盖通过本批验证；修复后28题均获独立AI内容审核通过。完整初稿流程未达到冻结的90%首次通过门槛，且仍有格式无效、两种排列回答不一致和排序语义捷径风险，因此不能宣称四项已经稳定完善。", "",
        "本批在4个上批未使用的Train root上新出28题（20多选、8排序），没有新增对话、训练或使用DEV/TEST。每题两种排列，分别提供全文和不给来源，共最多112次独立作答。模型仍为B1 step56合并模型。", "",
        "## 四项验收", "",
        "| 项目 | 实测与边界 |", "| --- | --- |",
        f"| 干扰项与答案内容 | 初次联合审核PASS {summary['initial_content_pass']}/28；独立语义内容初审PASS {summary['initial_semantic_content_pass']}/28；最终联合PASS {counts['PASS']}/28、HOLD {counts['HOLD']}、FAIL {counts['FAIL']}。每项引用、真值与Gold经另一AI作者重建；不是人类或Gold盲审，内容通过不保证难度。 |",
        "| 排序打乱及映射 | 每题100固定seed及所有合法预测映射往返、改Gold不改排列均由独立脚本核验；实际两个排列不同，非作者原排列。独立机械审核结果见下方链接。 |",
        f"| 23456 Gold覆盖 | 准入后各档数量：{json.dumps(summary['new_mc_coverage']['gold_count_histogram'], ensure_ascii=False)}；逐root覆盖见ROOT_VALIDATION。6 Gold全选基线100%，不算干扰设计证据。 |",
        f"| 提示词与捷径 | 机械lint共{len(cues)}项；语义审核另检查编辑提示、clock和mention-order。{len(overlaps)}/8排序与首次相关提及顺序重合；不能把删词成功称为来源依赖已解决。 |", "",
        "初审主指标同时要求内容和来源绑定合格，保持运行前冻结的90%阈值和28题分母。B组14题初稿路径写为相对路径，因而联合初审均HOLD；其中13题语义内容通过。A组13题联合通过，1题含超出原文的概括；B组另1题题干范围与正项不一致。两处语义问题及路径在运行前一次修复并独立复审。语义26/28不能替代联合13/28宣称初稿流程稳定。", "",
        "## Solver结果", "", "| 输入条件 | 有效回答 | 多选完全正确 | 排序完全正确 | 合计完全正确 |",
        "| --- | --- | --- | --- | --- |"]
    for cell in summary["conditions"]:
        lines.append(f"| {cell['condition']} | {cell['valid_requests']}/{cell['frozen_requests']} | {fraction(cell['mc_correct'],cell['mc_requests'])} | {fraction(cell['ordering_correct'],cell['ordering_requests'])} | {fraction(cell['correct'],cell['frozen_requests'])} |")
    lines.extend(["", "本批混合Gold分布下，始终全选的答案集合结构基线：" +
        "；".join(f"{cell['condition']} {fraction(cell['mc_allselect_correct'], cell['mc_requests'])}" for cell in summary["conditions"]) + "。该基线不模拟引用或输出格式有效性；6 Gold桶的答案集合结构基线为100%。"])
    lines.extend(["", "每题两种排列为相关观测，不能当作两道独立题。无效回答保留在整题正确率分母中。无来源成绩只作捷径检查，不是真实阅读分数。", "",
        "| 条件与排列 | 多选正确 | 排序正确 | 合计正确 |", "| --- | --- | --- | --- |"])
    for cell in summary["cells"]:
        lines.append(f"| {cell['condition']} / {cell['variant_id']} | {cell['mc_correct']}/{cell['mc_requests']} | {cell['ordering_correct']}/{cell['ordering_requests']} | {cell['correct']}/{cell['frozen_requests']} |")
    lines.extend(["", "| 输入条件 | 两排列均有效 | 规范预测相同 | 两排列都正确 | 正误随排列变化 |", "| --- | --- | --- | --- | --- |"])
    for condition in runner.CONDITIONS:
        pairs = [p for p in summary["variant_pairs"] if p["condition"] == condition]
        lines.append(f"| {condition} | {sum(p['both_valid'] for p in pairs)}/{len(pairs)} | {sum(p['canonical_prediction_same'] for p in pairs)}/{len(pairs)} | {sum(p['both_exact'] for p in pairs)}/{len(pairs)} | {sum(p['accuracy_changed'] for p in pairs)}/{len(pairs)} |")
    lines.extend(["", "| 输入条件 | 排序关系对正确 | 成对正确率 |", "| --- | --- | --- |"])
    for cell in summary["conditions"]:
        lines.append(f"| {cell['condition']} | {cell['ordering_pairwise_correct']}/{cell['ordering_pairwise_total']} | {cell['ordering_pairwise_accuracy']:.1%} |")
    lines.extend(["", "无来源两种排列均答对的题中，6 Gold全选题属于结构基线，单列且不作为额外捷径证据。其余候选：" +
        (", ".join(validation["question_only_both_correct_excluding_six_gold"]) or "无") + "。这些只是待解释观察，不直接证明模型所用策略。"])
    lines.extend(["", "剔除没有干扰项的6 Gold题后，仅2—5 Gold多选：" + "；".join(
        f"{condition} {fraction(sum(s['exact'] for s in scores if s['condition'] == condition and s['question_type'] == 'multi_select' and s['gold_count'] < 6), sum(s['condition'] == condition and s['question_type'] == 'multi_select' and s['gold_count'] < 6 for s in scores))}"
        for condition in runner.CONDITIONS) + "。这支持本批题目提供来源后表现更好，尚不等于所有错误选项都具有足够区分度。"])
    lines.extend(["", "## Gold桶与root", "", "| 条件 | Gold数 | 完全正确 | 误选 | 漏选 |", "| --- | --- | --- | --- | --- |"])
    for bucket in summary["new_mc_gold_buckets"]:
        lines.append(f"| {bucket['condition']} | {bucket['gold_count']} | {bucket['mc_correct']}/{bucket['mc_requests']} | {bucket['mc_false_positive']} | {bucket['mc_false_negative']} |")
    lines.extend(["", "描述性误选/漏选统计将无效多选预测表示为空集。FULL的6 Gold桶漏选6来自重复引用使整条响应无效，并非模型实际box漏掉6项；原始box实际全选。主正确率仍按冻结的完整格式准入规则计算。"])
    lines.extend(["", "| root | 条件 | 多选正确 | 排序正确 |", "| --- | --- | --- | --- |"])
    for cell in summary["root_cells"]:
        lines.append(f"| {cell['root_id']} | {cell['condition']} | {cell['mc_correct']}/{cell['mc_requests']} | {cell['ordering_correct']}/{cell['ordering_requests']} |")
    lines.extend(["", "## 执行与限制", "",
        f"实际返回{raw_freeze['returned_requests']}/{raw_freeze['requests']}；GPU批次耗时{raw_freeze['wall_seconds']:.1f}秒（含加载与关闭）；最长prompt {max(r['prompt_tokens'] for r in rows)} tokens，输出预算3072；未截断输入。无效原因：{json.dumps(dict(invalids), ensure_ascii=False)}。", "",
        "三条无效响应均正常停止，未触及长度上限：两条排序漏项，一条多选重复引用。同一引用重复使该6 Gold响应按冻结的完整格式规则计零分，虽然其box答案为全选；不能把这一次失分解释成干扰项或答案判断失败。", "",
        "可见理由AI抽核另发现：部分FULL排序已正确引用相关轮次或列出正确钟点，最终box却违反自身列出的先后关系。这是可观察到的最终答案整合矛盾，不能把所有失分归于来源检索失败、题目有错或难度提升。唯一无来源排序整题正确的记录是拨号、未接、草稿、修改发送、查看sent的常见流程；另一排列答错，只能记录为潜在常识捷径，不能声称稳定复现。逐例证据见可见理由抽核。", "",
        "作者和审核者按相同规则完成4个root，不能代表任意root或自动化模型批量稳定率。仅整题重复才排除，难题不按成绩筛掉。本批不以答错多作为出题质量好，也不自动签发正式弱点卡。保存的是可见简短依据，不是忠实内部CoT；理由结构检查不替代语义审核。", "",
        "两排列使用相同采样seed，但prompt和随机轨迹不同，正误变化只能诊断位置敏感性，不能当作单独位置因素的确定性因果效应。上批未用的root不等于模型训练未见；本批是新题跨root验证，不是未见root泛化评估。B1训练过；无来源正确可能来自题面、常识或训练记忆。", "",
        "Gold桶对应的题目内容、任务和领域并非随机分配，桶间成绩只作描述，不能归因于正确选项数量。", "",
        "如果排序仍可按话语提及或常识排序，应继续记录为题目设计/来源依赖问题，不能默默转去扩充并宣称四项已解决。", "",
        "[全部题目与作答](NEW_QUESTIONS_AND_REASONS_ZH.md) · [逐题审核](CONTENT_REVIEW_AI.json) · [机械审核](audit/MECHANICAL_AUDIT.json) · [执行独立审计](audit/EXECUTION_AUDIT.json) · [可见理由AI抽核](audit/VISIBLE_REASON_REVIEW_ZH.md) · [结构化结果](ROOT_VALIDATION.json)", ""])
    write_new(HERE / "REPORT_ZH.md", "\n".join(lines))
    details = ["# 28个题位、来源和模型回答", "", "Gold及作者说明仅用于审核评分；这些私有内容没有进入Solver的公开请求。完整原始输出以raw文件为准。", ""]
    score_by_id = {s["request_id"]: s for s in scores}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        review = next(r for r in reviews if r["candidate_id"] == cid)
        field = "options" if candidate["question_type"] == "multi_select" else "items"
        details.extend([f"## {cid}", "", f"审核：{review['decision']}；规范Gold：{' '.join(candidate['gold'])}", "", candidate["question"], ""])
        details.extend(f"- {k}. {v}" for k, v in candidate[field].items())
        details.extend(["", f"来源：[完整对话]({candidate['source_binding']['path']})", ""])
        for row in [r for r in rows if r["candidate_id"] == cid]:
            rid = row["request_id"]; score = score_by_id[rid]; record = reasons[rid]
            details.extend([f"### {row['variant_id']} / {row['condition']}", ""])
            details.extend(f"- {k}. {v}" for k, v in row["display"].items())
            details.extend(["", f"显示Gold：{' '.join(score['display_gold'])}；预测：{' '.join(score['display_prediction']) or '无有效预测'}；完全正确：{score['exact']}。", "",
                f"响应有效：{score['valid']}；无效原因：{score.get('invalid_error') or '无'}。", "",
                f"[实际prompt]({row['prompt_path']}) · [原始响应]({record['raw_binding']['path']})", ""])
            parsed = record.get("parsed")
            if parsed:
                for label, item in parsed["item_evidence"].items():
                    details.append(f"- {label}：{item['judgment']}；{item['reason']}；引用：{', '.join(item['turn_refs']) or '无'}")
                details.extend(["", parsed["answer_reason"], "", parsed["final_answer"], ""])
            else:
                details.extend([f"格式/执行错误：{record.get('error')}", ""])
    write_new(HERE / "NEW_QUESTIONS_AND_REASONS_ZH.md", "\n".join(details))
    print(json.dumps({"status": validation["status"], "first_pass": summary["initial_content_pass"], "final_pass": counts["PASS"], "requests": len(scores)}))


if __name__ == "__main__":
    main()
