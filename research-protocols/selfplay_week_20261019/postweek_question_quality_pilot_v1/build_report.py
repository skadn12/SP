"""Render the frozen results; never choose questions or resample answers."""
import collections
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_report_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def read(path):
    return json.loads(path.read_text())


def fraction(correct, total):
    return f"{correct}/{total}（{correct / total:.1%}）" if total else "—"


def write_once(path, text):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def main():
    prep, requests, tokens, freeze, private = runner.sealed_inputs()
    scores = read(HERE / "execution/SCORES.json")["records"]
    summary = read(HERE / "execution/ANALYSIS_SUMMARY.json")
    reasons = read(HERE / "execution/REASON_REVIEW_PACKET.json")["requests"]
    by_request = {r["request_id"]: r for r in requests}
    by_score = {r["request_id"]: r for r in scores}
    by_reason = {r["request_id"]: r for r in reasons}
    candidates = read(HERE / "CANDIDATES_PRIVATE.json")
    review = read(HERE / "CONTENT_REVIEW_AI.json")
    reason_review = read(HERE / "SOLVER_REASON_REVIEW_AI.json")
    assert reason_review["summary"]["response_count"] == 7
    assert reason_review["summary"]["item_count"] == 42
    assert summary == runner.summarize(scores, private)
    assert set(by_request) == set(by_score) == set(by_reason)
    raw = [read(HERE / "execution/raw" / (r["request_id"] + ".json")) for r in requests]
    for b in read(HERE / "HISTORICAL_FREEZE_BEFORE.json")["bindings"]:
        runner.check(b)
        if Path(b["path"]).name == "FINAL_MANIFEST.json":
            for child in read(Path(b["path"]))["bindings"]:
                runner.check(child)
    new = [s for s in scores if s["cohort"] == "new"]
    matches = summary["paired_changes"]
    report = ["# 题面与选项最小修复：实测报告", "",
        f"本批完成{len(candidates)}道新题的内容审核与Solver评估；多选7道、排序4道，审核结果{review['summary']}。覆盖5个train root、6个来源版本；没有新增对话或训练。历史9题在同一模型和协议下重新评估，没有继承旧成绩。", "",
        "## 实际结果", "",
        "| 题目版本 | 输入条件 | 整题正确 | 多选正确 | 排序正确 | 排序成对正确 | 有效响应 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in summary["cells"]:
        report.append(f"| {c['cohort']} | {c['condition']} | {fraction(c['correct'], c['frozen_requests'])} | {fraction(c['mc_correct'], c['mc_requests'])} | {fraction(c['ordering_correct'], c['ordering_requests'])} | {fraction(c['ordering_pairwise_correct'], c['ordering_pairwise_total'])} | {c['valid_requests']}/{c['frozen_requests']} |")
    report += ["", "本次最直接的发现：5道配对多选在FULL下由3/5变为1/5，无来源下由2/5变为0/5，且这些多选回答全部格式有效；它们在这次B1单次采样中更难答。排序FULL仍为3/4，新排序无来源也为3/4，因此不能宣称删提示词已解决来源依赖或排序难度问题。新增6 Gold两条件都全选答对，与全选基线相同。", ""]
    invalid = [s for s in scores if not s["valid"]]
    report += [f"{len(invalid)}次无效响应全部是QUESTION_ONLY排序漏排选项（ORDERING_INCOMPLETE）；JSON均完整返回，未达到长度上限，不是后端失败。旧题2次、新题1次。旧/新无来源排序差异因此混有指令遵循变化，不能据此判断题目变难或变简单。", ""]
    report += ["", "FULL提供来源全文；QUESTION_ONLY不提供对话、不得引用隐藏来源，用于检查无来源可解风险。两条件都要求给出最佳预测，分数不能混为真实阅读成绩。B1已经训练过，无来源答对还可能来自常识或既有训练记忆，不能仅凭本试验归因于题面提示。整题分母包含所有冻结请求，格式或后端失败不删除。", "",
        "## 新多选Gold覆盖", "", "| Gold数量 | 题数 | FULL整题正确 | 无来源整题正确 | FULL误选/漏选 |",
        "| --- | --- | --- | --- | --- |"]
    for count in (2, 3, 4, 5, 6):
        full = next(b for b in summary["new_mc_gold_buckets"] if b["gold_count"] == count and b["condition"] == "FULL")
        only = next(b for b in summary["new_mc_gold_buckets"] if b["gold_count"] == count and b["condition"] == "QUESTION_ONLY")
        report.append(f"| {count} | {full['frozen_requests']} | {fraction(full['correct'],full['frozen_requests'])} | {fraction(only['correct'],only['frozen_requests'])} | {full['mc_false_positive']}/{full['mc_false_negative']} |")
    report += ["", "7道新多选的始终全选基线为1/7（14.3%）；6 Gold子集该基线为100%，没有干扰项，不能据此证明逐项验证或干扰项质量。5 Gold也只有一个负项。", "",
        "## 控制配对结果", "", "以下仅比较9对父题与改写题，排除新增的4/6 Gold题；保持来源、显示排列、采样种子相同。", "",
        "| 输入条件 | 题型 | 配对数 | 旧题正确 | 新题正确 | 对→错 | 错→对 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for condition in runner.CONDITIONS:
        for kind in ("multi_select", "ordering"):
            pairs = [p for p in matches if p["strict_paired_comparison"] and p["condition"] == condition and next(c for c in candidates if c["candidate_id"] == p["new_candidate_id"])["question_type"] == kind]
            report.append(f"| {condition} | {kind} | {len(pairs)} | {sum(p['old_exact'] for p in pairs)} | {sum(p['new_exact'] for p in pairs)} | {sum(p['old_exact'] and not p['new_exact'] for p in pairs)} | {sum(not p['old_exact'] and p['new_exact'] for p in pairs)} |")
    report += ["", "配对是题目包层面的比较：部分干扰项命题及引用也被改写，不能将差异归因于单个词的删除。每题每条件仅一个随机采样，不作显著性、总体难度或反馈因果收益结论。", "",
        "## 可见理由独立审核", "",
        "7道新多选FULL回答、42项已完成独立理由审核，复用已全文读过的绑定来源。最终答案与逐项supported集合一致6/7；数量题存在不一致。主要可观察错误如下。", "",
        "- 数量题把07:17至09:30、08:08至09:30均算成73分钟，正确值分别是133、82；另用51分钟理由支持82分钟在途选项。",
        "- 5 Gold题选项写白色BMW，理由及引用却是蓝色Audi，模型仍判支持并全选。",
        "- 供应商证据题将agenda未完成作为供应商询问未发的充分证据，跨流程误选。",
        "- 引用题混淆Emily的calendar和James的note，并把时间提议升级为已经记录。",
        "- 4 Gold题只核对地址而遗漏错误的north地区条件；还有排除结论正确、理由却错误声称全文不存在记录的情况。", "",
        "两道最终答对题也有理由过度断言或引用未覆盖完整合取的情况。细节见SOLVER_REASON_REVIEW_AI.json及SOLVER_REASON_REVIEW_ZH.md；这些是单次可观察错误，可形成后续复现假设，尚不构成正式复现弱点卡，也不证明理由忠实反映内部计算。", "",
        "## 已修复与剩余限制", "",
        "已实施来源内元素与错误关系的干扰项设计、2/3/4/5/6 Gold覆盖、排序程序打乱与双向映射、删除显式话语阶段词与选项钟点，并分别做作者与独立内容审核。没有按Solver成绩重试、修改或筛选题目。", "",
        "话语题与烹饪流程题仍可按首次相关语义提及顺序解答；两道时间题的时间答案不同于原文提及顺序，但仍有部分常识线索。因此本批完成的是最小修题和诊断，尚未完成排序关系多样性的全面修复。", "",
        "固定随机排列使日程父题及新题的显示Gold碰巧为A B C。它相对作者输入顺序确实发生变化，但恰好等于答案顺序；按预先约定不依Gold重抽排列。该题的无来源答对尤其不能单独证明理解了日程。", "",
        "素材缺口已记录于MATERIAL_GAP_FEEDBACK.json，供后续扩充消费；该文件不等于已复现Solver弱点，本批也未新增扩充。可见理由已保存，execution原始理由包保留当时PENDING状态；后续审核以独立SOLVER_REASON_REVIEW_AI.json为准，不改写冻结包，也不直接签发正式弱点卡。", "",
        "## 执行与复现", "",
        f"B1是M-clean继续训练后的step56合并模型。temperature=0.8、top_p=0.9、输出预算3072、上下文16384；关闭原生thinking，记录可见简短理由的单次rollout，不能称为忠实内部CoT。实际冻结{len(requests)}请求，提交{freeze['submitted_requests']}，返回{freeze['returned_requests']}；有效{summary['valid']}，失败{len(requests)-summary['valid']}。耗时{freeze['wall_seconds']:.1f}秒（含模型初始化和关闭）。", "",
        f"最长实际prompt为{max(r['prompt_tokens'] for r in requests)} token，加3072输出预算未超过16384，无截断输入。结束原因：{dict(collections.Counter(r.get('finish_reason',r['status']) for r in raw))}。模型清单旧32-token答案设置由本批显式3072-token理由协议覆盖。", "",
        "4项CPU协议检查通过。独立执行审计完成1558项检查，重构所有prompt/schema、重新编码token、独立重算40项评分及全部汇总，结果一致；所有旧冻结manifest及其61项绑定未变。审计见EXECUTION_REVIEW_AI.json。完整命令、逐题raw、prompt/token与私有评分映射均在本目录。", "",
        "新题、显示答案与理由见NEW_QUESTIONS_AND_REASONS_ZH.md；逐题分数见execution/SCORES.json；汇总见execution/ANALYSIS_SUMMARY.json；文件导航见ARTIFACT_GUIDE_ZH.md。", ""]
    write_once(HERE / "REPORT_ZH.md", "\n".join(report))
    examples = ["# 本批新题、实际答案与可见理由", "",
        "每题列出实际显示顺序，Gold只在全部raw封存后用于此处评分。理由是模型的可见简短解释，尚未经语义审核，不等于忠实内部CoT或正式弱点证据。完整对话见各题FULL提示词链接。", ""]
    for candidate in candidates:
        cid = candidate["candidate_id"]
        full = next(r for r in requests if r["candidate_id"] == cid and r["condition"] == "FULL")
        examples += [f"## {cid}", "", full["question"], ""]
        examples += [f"- {label}. {text}" for label, text in full["display"].items()]
        examples += ["", f"[完整对话与实际提示词]({full['prompt_path']})", ""]
        for condition in runner.CONDITIONS:
            req = next(r for r in requests if r["candidate_id"] == cid and r["condition"] == condition)
            score, reason = by_score[req["request_id"]], by_reason[req["request_id"]]
            examples += [f"### {condition}", "", f"Gold：{' '.join(score['display_gold'])}；模型：{' '.join(score['display_prediction']) or '无有效答案'}；整题正确：{score['exact']}。", ""]
            examples += [f"[原始模型响应]({HERE / 'execution/raw' / (req['request_id'] + '.json')})", ""]
            if reason.get("parsed"):
                examples += ["```json", json.dumps(reason["parsed"], ensure_ascii=False, indent=2), "```", ""]
            else:
                examples += [f"无有效结构化理由：{reason.get('error')}", ""]
    write_once(HERE / "NEW_QUESTIONS_AND_REASONS_ZH.md", "\n".join(examples))
    with (HERE / "ROOT_VALIDATION.json").open("x") as f:
        json.dump({"raw_chain_verified": True, "summary_recomputed": True,
            "all_request_denominators_preserved": True, "old_manifests_and_61_bindings_verified": True,
            "requests": len(requests), "valid": summary["valid"],
            "failures": len(requests) - summary["valid"],
            "finish_reasons": dict(collections.Counter(r.get("finish_reason", r["status"]) for r in raw)),
            "semantic_reason_review": {"new_mc_full_responses": 7, "items": 42,
                "other_responses_reviewed": False,
                "binding": runner.bind(HERE / "SOLVER_REASON_REVIEW_AI.json")}}, f, indent=2)
    print(json.dumps({"report_written": True, "examples": len(candidates), "requests": len(requests)}))


if __name__ == "__main__":
    main()
