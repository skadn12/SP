"""Render the sealed direct and visible-reason results without altering inputs."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def binding(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

def pct(value):
    return f"{100*value:.1f}%"

def main():
    direct_path = HERE / "solver_v5/direct/SCORES.json"
    reason_path = HERE / "reason_diagnostic_v5/execution/ANALYSIS_SUMMARY.json"
    direct, reason = read(direct_path), read(reason_path)
    review_paths = [HERE / f"reason_diagnostic_v5/review_{root}/SEMANTIC_REVIEW_AI.json" for root in ("19708", "20432")]
    review_a, review_b = [read(p) for p in review_paths]
    assert len(review_a["reviews"]) == len(review_b["records"]) == 7
    assert review_a["summary"]["items_reviewed"] == review_b["summary"]["items"] == 38
    semantic = {"requests_reviewed":14,"items_reviewed":76,"whole_response_all_checks_pass":0,
        "citation_support_pass":review_a["summary"]["per_item_counts"]["citation_support"].get("PASS",0)+review_b["summary"]["citation_support_counts"].get("PASS",0),
        "citation_support_partial_or_hold":review_a["summary"]["per_item_counts"]["citation_support"].get("HOLD",0)+review_b["summary"]["citation_support_counts"].get("PARTIAL",0),
        "citation_support_fail":review_a["summary"]["per_item_counts"]["citation_support"].get("FAIL",0)+review_b["summary"]["citation_support_counts"].get("FAIL",0),
        "reviewer_type":"independent_AI_agents_with_disclosed_prior_gold_exposure",
        "root_review_summaries":[review_a["summary"],review_b["summary"]],
        "precision_rubrics_differ_do_not_pool_strict_fail_hold_counts":True,
        "bindings":[binding(p) for p in review_paths]}
    assert sum(semantic[k] for k in ("citation_support_pass","citation_support_partial_or_hold","citation_support_fail")) == 76
    scores_reason = {s["request_id"]:s for s in read(HERE / "reason_diagnostic_v5/execution/SCORES.json")["records"]}
    for r in review_a["reviews"]:
        score=scores_reason[r["request_id"]]
        key=r["reconstructed_display_answer"]
        assert (set(key)==set(score["display_gold"]) if score["question_type"]=="multi_select" else key==score["display_gold"])
        assert r["final_answer_correct_against_source_key"]==score["exact"]
    for r in review_b["records"]:
        score=scores_reason[r["request_id"]]
        key=r["independent_answer"].removeprefix("<box>").removesuffix("</box>").split()
        assert (set(key)==set(score["display_gold"]) if score["question_type"]=="multi_select" else key==score["display_gold"])
        assert r["final_answer_correct"]==score["exact"]
    manifest = read(HERE / "solver_v5/direct/manifest.json")
    assert manifest["solver_model"] == "B1"
    assert "day5_B1_schedule_v1/merged_model" in manifest["model"]
    records = direct["requests"]
    stats = {}
    for typ in ("multi_select", "ordering"):
        selected = [r for r in records if r["question_type"] == typ]
        stats[typ] = {"requests": len(selected), "valid": sum(r["valid"] for r in selected),
            "exact": sum(r["exact"] for r in selected), "exact_accuracy": sum(r["exact"] for r in selected)/len(selected)}
    mc = [r for r in records if r["question_type"] == "multi_select"]
    order = [r for r in records if r["question_type"] == "ordering"]
    invalid = sum(not r["valid"] for r in records)
    mean_jaccard = sum(r["metrics"]["jaccard"] for r in mc)/len(mc)
    pairs = sum(r["metrics"]["total_pairs"] for r in order)
    correct_pairs = sum(r["metrics"]["correct_pairs"] for r in order)
    fp = sum(r["metrics"]["fp"] or 0 for r in mc)
    fn = sum(r["metrics"]["fn"] or 0 for r in mc)
    strata = dict(collections.Counter(q["stratum"] for q in direct["questions"]))
    all_select = sum(r["metrics"]["gold_count"] == 6 for r in mc)
    summary = {"independent_questions":14,"roots":2,"direct_requests":len(records),"model":"B1",
        "direct_by_type":stats,"invalid_responses":invalid,"mean_mc_jaccard":mean_jaccard,
        "mc_fp_valid_responses":fp,"mc_fn_valid_responses":fn,
        "ordering_pairwise_correct":correct_pairs,"ordering_pairwise_total":pairs,
        "all_select_baseline_correct":all_select,"all_select_baseline_total":len(mc),
        "question_strata":strata,"reason_summary":reason,"independent_semantic_review":semantic,"training_performed":False,
        "solver_used_for_admission":False,"formal_weakness_claim":False,"causal_feedback_benefit_claim":False,
        "bindings":[binding(direct_path),binding(reason_path),binding(HERE / "QUESTION_FREEZE.json")]+semantic["bindings"]}
    lines=["# v5 扩充后题目评估", "", "模型为 B1（day5_B1_schedule_v1/merged_model），两种输出模式分开冻结、分开统计。来源和题目未按solver分数筛选。", "",
        "## 已完成", "", "- 两个Train root，两条路线各三场，共12场真实扩充，新增160 turns。", "- 独立内容审核通过14题：多选10题，排序4题；gold数量2/3/4/5/6各两题。", "- 直接答案评估：多选每题6个排列×3次采样，排序每题12个排列×1次采样，共228次。", "- 可见理由诊断：全部14题各一次，保存逐项判断、引用、简短理由和最终框；与直接答案评估分开。", "",
        "## 直接答案结果", "", "| 题型 | 独立题 | 请求 | 有效回答 | 完全正确 |", "| --- | ---: | ---: | ---: | ---: |"]
    for typ,label,n in (("multi_select","多选",10),("ordering","排序",4)):
        s=stats[typ]
        lines.append(f"| {label} | {n} | {s['requests']} | {s['valid']}/{s['requests']} | {s['exact']}/{s['requests']}（{pct(s['exact_accuracy'])}） |")
    lines += ["", f"合计按请求计完全正确{direct['exact_calls']}/228（{pct(direct['exact_calls']/228)}）；14题等权准确率{pct(direct['question_equal_exact'])}。多选和排序每题测量次数不同，两个总体口径不可混用。", "",
        f"多选部分得分使用集合Jaccard（交集/并集），平均{pct(mean_jaccard)}；有效回答的误选{fp}项、漏选{fn}项。排序成对顺序正确{correct_pairs}/{pairs}（{pct(correct_pairs/pairs)}）。无效回答{invalid}，仍留在冻结分母；无效多选FP/FN为未知，不伪造为0。", "",
        f"固定全选基线{all_select}/{len(mc)}（{pct(all_select/len(mc))}）。6-gold题答对不能证明干扰项质量。", "", "## 每个root", "", "| root | 完全正确请求 | 题目等权准确率 |", "| --- | ---: | ---: |"]
    for root,s in direct["by_root"].items():
        lines.append(f"| {root} | {s['exact_calls']}/{s['calls']} | {pct(s['question_equal_exact'])} |")
    lines += ["", "## 多选Gold覆盖", "", "| Gold数量 | 独立题 | 完全正确 | Jaccard |", "| --- | ---: | ---: | ---: |"]
    for count,s in direct["multi_select_by_gold_count"].items():
        lines.append(f"| {count} | {s['questions']} | {s['exact_calls']}/{s['calls']}（{pct(s['exact_calls']/s['calls'])}） | {pct(s['mc_mean_jaccard'])} |")
    lines += ["", "## 逐题结果", "", "| 题目 | 完全正确 | 排列/采样状态 |", "| --- | ---: | --- |"]
    for q in direct["questions"]:
        lines.append(f"| {q['candidate_id']} | {q['exact_calls']}/{q['denominator_calls']} | {q['stratum']} |")
    lines += ["", "状态只是本批重复测量的描述，不等于正式弱点准入；排列与采样不是独立新题，只有2个root，不能按228个独立样本外推。", "",
        "## 可见理由", "", f"结构有效{reason['valid_requests']}/14，最终答案完全正确{reason['correct']}/14；多选{reason['mc_correct']}/10、排序{reason['ordering_correct']}/4。逐项supported集合与最终选择不一致{reason['mc_judgment_answer_inconsistent']}条。原生thinking关闭，保存的是可见简短说明。", "",
        f"两名独立AI已完成14条、76项语义审核；均披露之前可能/实际接触Gold，不是人工盲审。14条模型最终答案与独立来源重建的对错一致；但整条理由、引用和说明全部通过为0/14。提交引用覆盖关键主张的项为{semantic['citation_support_pass']}/76，部分覆盖/待定{semantic['citation_support_partial_or_hold']}项，无支持或错场景{semantic['citation_support_fail']}项。两位审核者对额外无关引用精度的严格程度不同，故不把各自严格FAIL/HOLD数合并成统一误差率。", "",
        "ANALYSIS_SUMMARY和REASON_REVIEW_PACKET保留生成时的PENDING快照；最新审核以两个review目录为准。没有把模型错理由替换成审核者的正确理由。输出预算、格式及显示顺序不同，两种输出模式的准确率差异不能证明CoT因果作用。", "",
        "## 排序操作标签限制", "", "复核发现3题操作标签未准确对应题意：19708 ORDER1与20432 ORDER1标成ORDER_EVENT_TIME，实际要求问询记录的披露/进展顺序；20432 ORDER2标成ORDER_CORRECTION_CHAIN，实际是gallery问询进展，没有修正链。19708 ORDER2才是记录修正/核对顺序。本批4题中，没有真正按钟点值排序的题。题干当前含义下的Gold经独立重建相符，但不能据这些操作名宣称完成时间排序多样性，也不能把ORDER1错误直接称为时序推理弱点。", "",
        "本批冻结题、输入、原始回答及分数均保留；操作标签问题另行登记，后续更改操作提示需要新版本与新运行，旧分数不可转挂。", "",
        "## 耗时与文件", "", f"直接评估GPU调用流程耗时{direct['cost']['live_wall_seconds']:.2f}秒；理由调用流程耗时{read(HERE / 'reason_diagnostic_v5/execution/RAW_RESPONSE_FREEZE.json')['wall_seconds']:.2f}秒，均含加载/关闭；无自动重试、无输入截断。", "",
        "- 题目与内容审核：[questions](questions/README_ZH.md)", "- 直接答案逐请求评分：[SCORES.json](solver_v5/direct/SCORES.json)", "- 直接答案原始记录：[responses](solver_v5/direct/responses/)", "- 可见理由完整样例：[EXAMPLES_ZH.md](reason_diagnostic_v5/execution/EXAMPLES_ZH.md)", "- 可见理由原始记录：[raw](reason_diagnostic_v5/execution/raw/)", "",
        "本轮未训练、未直接写入最终训练混合；没有正式弱点、反馈因果收益或新训练效果结论。"]
    for name,value in (("EVALUATION_SUMMARY.json",summary),):
        with (HERE / name).open("x",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2); stream.write("\n")
    with (HERE / "EVALUATION_REPORT_ZH.md").open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    print(json.dumps({"direct_by_type":stats,"invalid":invalid,"reason_correct":reason['correct']}))

if __name__ == "__main__":
    main()
