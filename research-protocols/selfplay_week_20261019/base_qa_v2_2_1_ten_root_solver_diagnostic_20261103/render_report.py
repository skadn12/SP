#!/usr/bin/env python3
"""Render the completed frozen diagnostic; never modify inputs, answers or Gold."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "base_qa_v2_2_1_ten_root_20261103"


def read(path):
    return json.loads(path.read_text())


def save(path, text):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def semantic_to_display(option_ids, item):
    reverse = {v: k for k, v in item["label_to_option_id"].items()}
    return ",".join(sorted(reverse[oid] for oid in option_ids)) or "（空）"


def main():
    scores = read(HERE / "SCORES.json")
    items = read(HERE / "ITEMS_PRIVATE.json")
    assert read(HERE / "COMPLETED.json")["returned_calls"] == 360
    bindings = read(HERE / "INPUT_BINDINGS.json")
    checks = [{**b, "unchanged": hashlib.sha256(Path(b["path"]).read_bytes()).hexdigest() == b["sha256_bytes"]} for b in bindings["files"]]
    assert all(b["unchanged"] for b in checks)
    assert hashlib.sha256((HERE / "run_diagnostic.py").read_bytes()).hexdigest() == bindings["adapter_sha256"]
    assert hashlib.sha256((HERE.parent / "base_qa_v2_2_1_format_restore_solver_eval_20261102/multiselect_permutation_repeat_20261102/run_multiselect_permutation_eval.py").read_bytes()).hexdigest() == bindings["legacy_runner_sha256"]
    save(HERE / "POSTRUN_INTEGRITY.json", json.dumps({"status": "PASS", "inputs_checked": len(checks), "adapter_and_legacy_runner_unchanged": True, "checks": checks}, ensure_ascii=False, indent=2) + "\n")

    table = ["| root | 展示副本 Gold | Full 整题正确/18 | Full 平均 J | No-context 正确/18 | No-context 平均 J |",
             "|---|---|---:|---:|---:|---:|"]
    detail = ["# 10-root 题目、Gold 与实际 Solver 输出", "", "本文件采用既有打乱展示副本的 A–F 标签呈现题目。实际评估另有 6 个独立排列；下表中的每次回答均先还原稳定 option_id，再转换回本页标签，不能直接跨排列比较原始字母。", "", "本轮为 diagnostic_only；沿用既有 Gold，没有执行新的独立盲审，也没有修改题目资格。", ""]
    csv_rows = []
    for item in items:
        qid = item["candidate_id"]
        root = item["root_id"].removeprefix("cc-train-episode-")
        full = scores["by_question"][qid]["full_context"]
        nc = scores["by_question"][qid]["no_context"]
        table.append(f"| {root} | {','.join(item['gold_labels'])} | {full['exact_calls']} | {full['mean_J']:.4f} | {nc['exact_calls']} | {nc['mean_J']:.4f} |")
        csv_rows.append({"root_id": item["root_id"], "candidate_id": qid,
                         "base_display_gold": ",".join(item["gold_labels"]),
                         "full_exact_of_18": full["exact_calls"], "full_mean_J": full["mean_J"],
                         "full_FP": full["fp_total"], "full_FN": full["fn_total"],
                         "full_invalid": full["invalid_calls"], "full_semantic_modes": full["semantic_prediction_modes"],
                         "no_context_exact_of_18": nc["exact_calls"], "no_context_mean_J": nc["mean_J"]})
        detail.extend([f"## {root} · {qid}", "", f"**题型：多选；Gold：{','.join(item['gold_labels'])}。**", "", item["question"], ""])
        for option in item["options"]:
            detail.extend([f"{option['id']}. {option['text']}", ""])
        detail.extend([f"来源：[FULL_SOURCE.json]({INPUT / 'roots' / item['root_id'] / 'FULL_SOURCE.json'})；作者原始证书：[HISTORICAL_CANDIDATE.json]({INPUT / 'roots' / item['root_id'] / 'HISTORICAL_CANDIDATE.json'})。", ""])
        for mode, summary in (("full_context", full), ("no_context", nc)):
            detail.extend([f"### {mode}", "", f"整题 {summary['exact_calls']}/18；平均 J={summary['mean_J']:.6f}；有效 {summary['valid_calls']}/18；已知 FP={summary['fp_total']}、FN={summary['fn_total']}。语义答案共 {summary['semantic_prediction_modes']} 种，众数一致率 {summary['modal_semantic_agreement'] if summary['modal_semantic_agreement'] is not None else 'NA'}。", "", "| 排列 | 三次原始输出（当前排列字母） | 还原为本页展示标签 | 整题正确/3 | J 均分 |", "|---|---|---|---:|---:|"])
            for i in range(6):
                variant = f"perm_{i:02d}"
                rows = [r for r in scores["requests"] if r["candidate_id"] == qid and r["context_mode"] == mode and r["variant_id"] == variant]
                raw = " / ".join(json.dumps(r["raw_prediction"], ensure_ascii=False) for r in rows).replace("|", "\\|")
                restored = " / ".join(semantic_to_display(r["predicted_option_ids"], item) if r["valid"] else "INVALID" for r in rows)
                vs = summary["by_permutation"][variant]
                detail.append(f"| {variant} | {raw} | {restored} | {vs['exact_calls']} | {vs['mean_J']:.4f} |")
            detail.extend(["", "有效回答的语义分布：", ""])
            for row in summary["prediction_distribution"]:
                detail.append(f"- {semantic_to_display(row['option_ids'], item)}：{row['count']}/18")
            detail.extend(["", "各选项被选次数（本页展示标签）：", "", "| 标签 | Gold | 选中次数/18 | 已知错误 |", "|---|---|---:|---|"])
            for label, oid in sorted(item["label_to_option_id"].items()):
                count = summary["option_counts"][oid]
                errors = f"漏选 {summary['valid_calls'] - count['selected']}" if count["gold"] else f"错选 {count['selected']}"
                detail.append(f"| {label} | {'是' if count['gold'] else '否'} | {count['selected']} | {errors} |")
            detail.append("")

    full, nc = scores["summaries"]["full_context"], scores["summaries"]["no_context"]
    stable = sum(scores["by_question"][item["candidate_id"]]["full_context"]["exact_calls"] == 18 for item in items)
    per_variant = [scores["by_question"][i["candidate_id"]]["full_context"]["by_permutation"] for i in items]
    varying = sum(len({v["exact_calls"] for v in values.values()}) > 1 for values in per_variant)
    report = ["# Ten-root 本地 Qwen3-4B Solver 诊断", "", "本轮只评估已有 10 道历史 MC3 多选候选；没有生成新题、修改 Gold、运行 Reviewer 或升级候选状态。所有结果基于既有 Gold，标记 diagnostic_only。", "", f"**Full-context：{full['exact_calls']}/180（{full['exact_accuracy']:.2%}），平均 J={full['mean_J']:.6f}；No-context：{nc['exact_calls']}/180（{nc['exact_accuracy']:.2%}），平均 J={nc['mean_J']:.6f}。**", "", f"Full 有效 {full['valid_calls']}/180，No-context 有效 {nc['valid_calls']}/180；Full 已知错选 FP={full['fp_total']}、漏选 FN={full['fn_total']}。{stable}/10 题在 Full 的 18 次回答中始终整题正确。", "", "每题 6 个互异选项排列 × 每排列 3 次；Full 和 No-context 各 18 次，共 360 次本地回答。表中分母是回答次数，不是 180 道独立题目；独立题目只有 10 道。", "", *table, "", "## 配置与评分", "", "本地基础模型 `/root/autodl-tmp/Qwen3-4B`；temperature=0.8、top_p=0.9、max_new_tokens=32、thinking=False；上下文上限 16384。沿用历史多选 Prompt、随机排列算法及解析/评分函数，只在独立实验适配器中把选项标签泛化为 A–F、转接 turn_ref。Full 保留全部 218–236 轮，输入 8488–9780 tokens，不截断；没有向模型传入候选 ID（其中 MC3 会泄漏数量）、Gold、证书或来源风险标记。", "", "整题分要求预测语义集合与 Gold 完全一致；部分分 J=TP/g−FP/(n−g)。本批 n=6、g=3，所以 J=(TP−FP)/3，允许负分。格式无效或长度终止按既有协议计整题错误、部分分0，保留固定分母；FP/FN仅统计有效回答的已知错误。本轮未自动纠正或重试任何模型答案。", "", "全部排列由稳定 option_id 重绑，Gold 不依赖展示字母。多次作答的一致性按排序后的语义 ID 集合统计；不把标签位置变化当作模型答案变化。", "", "## 观察范围与题目问题", "", f"{varying}/10 题在不同排列的整题正确次数存在差异。该值同时受到随机采样影响，不能单独证明位置偏差。每题的排列明细、语义答案分布、漏选/错选选项见 [ITEMS_AND_RESULTS.md](ITEMS_AND_RESULTS.md)。", "", "已有题目质量复核（能看见 Gold，并非盲审）发现：6/10 题的 Gold 恰好是最长三个选项；这是已知 g=3 的事后启发式控制，不能当成未泄漏基线。24719 证书把 notes 子句绑定到错误轮次，完整原文另有正确支持；109164 的未来截止表述存在措辞风险。详见 [ITEM_QUALITY_NOTES.md](ITEM_QUALITY_NOTES.md)。这些题目缺陷不能归为 Solver 弱点。", "", "本轮可确认的模型现象是已知 Gold 下的具体漏选、错选和输出稳定性；没有模型推理过程，不能仅凭最终标签断言某个内部推理步骤失败。Full 与 No-context 的差异只能作为上下文利用的诊断线索。", "", "当前 10 题尚未完成本轮正式准入和隔离盲审；本轮没有补造审核记录。高分不证明困难 QA 生成能力，错误也不能直接进入正式弱点反馈池。现有选项长度捷径、同构摘要形式、全部 MC3 及证书问题均保留。", "", "## 记录与完整性", "", "- [完整题目与逐次答案](ITEMS_AND_RESULTS.md)", "- [机器评分与 TP/FP/FN](SCORES.json)", "- [逐题汇总 CSV](PER_QUESTION.csv)", "- [原始输出](RAW_ANSWERS.json) / raw/", "- [冻结请求与 Prompt 路径](REQUESTS.json)", "- [配置](MODEL_CONFIG.json)、[排列和私有 Gold 映射](PERMUTATIONS_PRIVATE.json)", "- [预检](PREFLIGHT.json)、[输入绑定](INPUT_BINDINGS.json)、[推理后哈希核验](POSTRUN_INTEGRITY.json)", "", "首次离线预检发现适配器混用了候选文件字节哈希与规范 JSON 哈希；推理开始前已修正，未改变候选内容或 Gold。所有冻结请求一次性执行，没有失败答案重采样。原始生成流程及历史实验文件未修改。", ""]
    save(HERE / "REPORT.md", "\n".join(report))
    save(HERE / "ITEMS_AND_RESULTS.md", "\n".join(detail))
    with (HERE / "PER_QUESTION.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"Full": {k: full[k] for k in ("exact_calls", "calls", "mean_J", "valid_calls")}, "No_context": {k: nc[k] for k in ("exact_calls", "calls", "mean_J", "valid_calls")}, "always_correct_questions": stable}, ensure_ascii=False))


if __name__ == "__main__":
    main()
