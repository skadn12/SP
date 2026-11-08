"""Summarize sealed paired results, separating new questions from controls.

Reads existing artifacts only; performs no model calls, training or promotion.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "h4_codex_controlled_batch_v105_solver_eval"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return {"fraction": str(value), "value": float(value)}


def main():
    outputs = [HERE / name for name in ("SUMMARY.json", "REPORT.md", "COMPLETE_MANIFEST.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Summary exists; do not overwrite the completed analysis.")
    seal_path = EVAL / "EVALUATION_COMPLETE_MANIFEST.json"
    seal = read(seal_path)
    assert seal["status"] == "PAIRED_SOLVER_EVALUATION_AUDITED_COMPLETE"
    for row in seal["files"]:
        path = (EVAL / row["path"]).resolve()
        assert path.is_relative_to(EVAL) and path.stat().st_size == row["bytes"]
        assert sha(path) == row["sha256"], str(path)
    audit = read(EVAL / "AUDIT_RESULTS.json")
    assert audit["status"] == "VALID" and not audit["data_errors"]
    pool = {q["candidate_id"]: q for q in read(EVAL / "EVALUATION_POOL.json")["questions"]}
    calls = read(EVAL / "SCORED_CALLS.json")["calls"]
    assert len(calls) == 912 and all(c["evaluation_valid"] for c in calls)
    per_question = {side: {q["candidate_id"]: q for q in audit["models"][side]["questions"]}
                    for side in ("base", "trained")}

    def stats(side, ids):
        qs = [per_question[side][cid] for cid in ids]
        cs = [c for c in calls if c["model"] == side and c["candidate_id"] in ids]
        count = len(cs)
        exact = sum(c["independent_score"]["exact_match"] for c in cs)
        result = {"questions": len(qs), "valid_calls": count, "exact_correct": exact,
                  "exact_rate": number(Fraction(exact, count)),
                  "call_weighted_partial_mean": number(sum((Fraction(c["independent_score"]["partial_score"]) for c in cs), Fraction()) / count),
                  "question_equal_partial_mean": number(sum((Fraction(q["partial_mean"]["fraction"]) for q in qs), Fraction()) / len(qs)),
                  "native_selected": sum(q["selected_by_policy"] is True for q in qs)}
        if all(q["question_type"] == "multi_select" for q in qs):
            result.update(false_positive_options=sum(c["independent_score"]["fp"] for c in cs),
                          false_negative_options=sum(c["independent_score"]["fn"] for c in cs),
                          calls_with_false_positive=sum(c["independent_score"]["fp"] > 0 for c in cs),
                          prediction_size_counts=dict(sorted(Counter(len(c["independent_score"]["content_prediction"]) for c in cs).items())))
            result["mean_prediction_size"] = number(Fraction(sum(len(c["independent_score"]["content_prediction"]) for c in cs), count))
        return result

    def compare(ids):
        ids = set(ids)
        result = {side: stats(side, ids) for side in ("base", "trained")}
        result["exact_change_percentage_points"] = 100 * (result["trained"]["exact_rate"]["value"] - result["base"]["exact_rate"]["value"])
        return result

    def ids_where(predicate):
        return {cid for cid, q in pool.items() if predicate(q)}

    new_ms = ids_where(lambda q: q["question_type"] == "multi_select" and q["role"] == "new_child")
    controls = ids_where(lambda q: q["question_type"] == "multi_select" and len(q["gold"]) == 6)
    ordering = ids_where(lambda q: q["question_type"] == "ordering")
    assert len(new_ms) == 16 and len(controls) == 4 and len(ordering) == 8
    assert new_ms.isdisjoint(controls) and new_ms | controls | ordering == set(pool)
    groups = {"new_multi_gold2_to5": compare(new_ms), "new_ordering": compare(ordering),
              "gold6_controls": compare(controls), "all_multi_including_controls": compare(new_ms | controls)}
    by_gold = {str(g): compare(ids_where(lambda q: q["question_type"] == "multi_select" and len(q["gold"]) == g)) for g in range(2, 7)}
    roots = sorted({q["root_id"] for q in pool.values()})
    by_root = {root: {name: compare(ids & {cid for cid, q in pool.items() if q["root_id"] == root})
                     for name, ids in (("new_multi_gold2_to5", new_ms), ("new_ordering", ordering), ("gold6_controls", controls))}
               for root in roots}
    selected_ids = [cid for cid in pool if per_question["trained"][cid]["selected_by_policy"] is True]
    selection = {"status": "PROVISIONAL_NATIVE_SELECTION_ONLY", "total": len(selected_ids),
                 "new_multi": len(set(selected_ids) & new_ms), "new_ordering": len(set(selected_ids) & ordering),
                 "gold6_controls": len(set(selected_ids) & controls), "candidate_ids": selected_ids}
    assert selection["total"] == selection["new_multi"] + selection["new_ordering"] + selection["gold6_controls"]
    timing = {s: read(EVAL / (s.upper() + "_EXECUTION.COMPLETE.json"))["wall_seconds"] for s in ("base", "trained")}
    timing["combined_model_execution_seconds"] = timing["base"] + timing["trained"]
    summary = {"status": "PAIRED_SOLVER_ANALYSIS_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "bindings": {name: {"path": str(EVAL / name), "sha256": sha(EVAL / name)} for name in
                            ("EVALUATION_COMPLETE_MANIFEST.json", "AUDIT_RESULTS.json", "SCORED_CALLS.json", "EVALUATION_POOL.json")},
               "groups": groups, "by_gold": by_gold, "by_root": by_root, "native_selection": selection,
               "execution_seconds": timing, "new_model_calls": 0, "training_started": False, "training_promotion": False,
               "scope": "Existing M0 and M_clean_policy checkpoints on the same purposively selected v105 batch; not training on v105 or held-out evaluation."}

    def table_row(label, result):
        a, b = result["base"], result["trained"]
        return (f"| {label} | {a['questions']} | {a['exact_correct']}/{a['valid_calls']}（{a['exact_rate']['value']:.2%}）"
                f" | {b['exact_correct']}/{b['valid_calls']}（{b['exact_rate']['value']:.2%}） | {result['exact_change_percentage_points']:+.2f}"
                f" | {a['question_equal_partial_mean']['value']:.4f} → {b['question_equal_partial_mean']['value']:.4f} |")

    text = ["# v105 配对 Solver 结果解读", "",
            "评估与独立机械审核已完成：两个模型各 456 次，共 912 次响应全部有效，无解析失败或截断。执行合计 %.2f 秒。" % timing["combined_model_execution_seconds"], "",
            "比较的是已有 M0 与 M_clean_policy 检查点；本轮未训练。主结果应把新增 2–5gold 多选与 6gold 对照分开。", "",
            "| 组别 | 题数 | M0 exact | M_clean_policy exact | 变化（百分点） | Partial（每题等权） |",
            "|---|---:|---:|---:|---:|---:|"]
    for label, key in (("新增多选 2–5gold", "new_multi_gold2_to5"), ("新增排序", "new_ordering"),
                       ("6gold 对照", "gold6_controls"), ("全部多选（含对照）", "all_multi_including_controls")):
        text.append(table_row(label, groups[key]))
    text.extend(["", "| Gold | 题数 | M0 exact | M_clean_policy exact | 变化（百分点） | Partial（每题等权） |",
                 "|---|---:|---:|---:|---:|---:|"])
    text.extend(table_row(g + ("（对照）" if g == "6" else ""), result) for g, result in by_gold.items())
    text.extend(["", "低 Gold 的退步伴随误选增加，提示选项判别与选择数量校准问题。Gold2 的 false-positive 选项累计从 %d 增至 %d，含误选调用从 %d/72 增至 %d/72；平均选择 %.3f → %.3f 个选项。Gold3 的 false-positive 从 %d 增至 %d。尚不能由此归因于特定训练数据或 Gold 错误。" % (
        by_gold["2"]["base"]["false_positive_options"], by_gold["2"]["trained"]["false_positive_options"],
        by_gold["2"]["base"]["calls_with_false_positive"], by_gold["2"]["trained"]["calls_with_false_positive"],
        by_gold["2"]["base"]["mean_prediction_size"]["value"], by_gold["2"]["trained"]["mean_prediction_size"]["value"],
        by_gold["3"]["base"]["false_positive_options"], by_gold["3"]["trained"]["false_positive_options"]), "",
        "全部多选的提升受到 6gold 对照影响；仅新增多选下降 2.78 个百分点，不能表述为本轮多选能力整体提升。始终全选在 2–5gold 上 exact=0、partial=0，在 6gold 对照上 exact=1、partial=1。本轮两个模型在 2–5gold 上均未输出全选。", "",
        "| Root / 组别 | 题数 | M0 exact | M_clean_policy exact | 变化（百分点） | Partial（每题等权） |",
        "|---|---:|---:|---:|---:|---:|"])
    for root, rows in by_root.items():
        for key, label in (("new_multi_gold2_to5", "新增多选"), ("new_ordering", "排序"), ("gold6_controls", "对照")):
            text.append(table_row(root + " / " + label, rows[key]))
    text.extend(["", "按既有 0 < full partial mean < 1 策略，训练后模型暂入选 %d 题：%d 道新增多选、%d 道排序、%d 道 6gold 对照。合计新增题 %d/24；另 3 道新增多选及 1 道排序均全对，按原规则作为过易题排除。暂入选不等于 Gold 正确率，也未导出训练集。" % (
        selection["total"], selection["new_multi"], selection["new_ordering"], selection["gold6_controls"], selection["new_multi"] + selection["new_ordering"]), "",
        "排序难度差异较大：root19708-ordering02 为 0/12，root9557-ordering01 为 1/12；root25652 两题为 12/12 和 11/12。下一步适合逐项复核低 Gold 的误选及两道低准确率排序题的锚点，保留当前题面和 Gold；之后再决定训练池配比。", "",
        "只有 4 个历史上已暴露、按兼容性选取的 root；多次排列与采样不是独立的新题样本。本结果描述该批次，不作为泛化提升证据。结构化 choice 解码约束了输出格式；零解析失败不证明自由输出格式能力或题意无歧义。执行审核校验封存、来源、配对配置与评分，不新增独立语义 Gold 判断。", "",
        f"原始结果与逐题报告：[Solver 报告]({EVAL / 'REPORT.md'})。逐次独立评分：[SCORED_CALLS.json]({EVAL / 'SCORED_CALLS.json'})。", ""])
    outputs[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs[1].write_text("\n".join(text), encoding="utf-8")
    manifest = {"status": summary["status"], "source_evaluation_seal": {"path": str(seal_path), "sha256": sha(seal_path)},
                "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in [Path(__file__), *outputs[:2]]],
                "new_model_calls": 0, "training_started": False, "training_promotion": False}
    outputs[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "calls": len(calls), "selection": selection, "execution_seconds": timing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
