"""Summarize the authorized frozen-question M0/M_clean comparison."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "h4_groundless_random_feedback_v1/evaluation/runs/full_only_pass4"
PAIRED = HERE / "runs/old7_m_clean_01"
R02 = HERE.parent / "h4_r02_qa_batch_v1/solver/pilot_01_m_clean"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows):
    means = [Fraction(row["full_partial_mean"]) for row in rows]
    exact_count = sum(Fraction(row["full_exact_rate"]) * row["actual_full_calls"] for row in rows)
    count = sum(row["actual_full_calls"] for row in rows)
    assert exact_count.denominator == 1
    return {
        "question_count": len(rows),
        "selected": sum(row["selected_by_policy"] is True for row in rows),
        "perfect_questions": sum(value == 1 for value in means),
        "nonpositive_questions": sum(value <= 0 for value in means),
        "mean_partial": str(sum(means) / len(means)),
        "mean_partial_decimal": float(sum(means) / len(means)),
        "exact_correct_calls": int(exact_count),
        "valid_calls": count,
        "exact_rate": str(exact_count / count),
        "exact_rate_decimal": float(exact_count / count),
    }


def main():
    proof = read(HERE / "COMPARISON_PREFLIGHT.json")
    assert digest(OLD / "manifest.json") == proof["old_manifest_sha256"]
    assert digest(PAIRED / "manifest.json") == proof["new_manifest_sha256"]
    assert digest(R02 / "manifest.json") == proof["current_R02_manifest_sha256"]
    complete = read(PAIRED / "COMPLETE.json")
    assert complete["status"] == "COMPLETE" and complete["actual_call_count"] == 126
    results = read(PAIRED / "RESULTS.json")
    assert not results["call_set_errors"] and results["solver_model"] == "M_clean_policy"
    ids = [q["candidate_id"] for q in results["questions"]]
    assert len(ids) == len(set(ids)) == 7
    roots = {q["root_id"] for q in results["questions"]}
    old_rows = [q for q in read(OLD / "RESULTS.json")["questions"] if q["candidate_id"] in ids]
    r02_rows = [q for q in read(R02 / "RESULTS.json")["questions"]
                if q["root_id"] in roots and q["question_type"] == "multi_select"]
    assert len(old_rows) == len(r02_rows) == 7
    records = [read(path) for path in (PAIRED / "responses").glob("*.json")]
    assert len(records) == 126
    for q in results["questions"]:
        assert q["actual_full_calls"] == q["expected_full_calls"] == 18 and not q["data_errors"]
        responses = [row for row in records if row["candidate_id"] == q["candidate_id"]]
        assert len(responses) == 18 and all(row["evaluation_valid"] for row in responses)
        assert len({(row["variant_id"], row["slot"]) for row in responses}) == 18
        mean = sum(Fraction(row["score"]["partial_score_fraction"]) for row in responses) / 18
        exact = sum(row["score"]["exact_match"] is True for row in responses)
        assert mean == Fraction(q["full_partial_mean"])
        assert Fraction(exact, 18) == Fraction(q["full_exact_rate"])
        assert q["selected_by_policy"] == (0 < mean < 1)
    old_by_id = {q["candidate_id"]: q for q in old_rows}
    pairs = []
    for q in results["questions"]:
        previous = old_by_id[q["candidate_id"]]
        pairs.append({
            "candidate_id": q["candidate_id"], "root_id": q["root_id"],
            "M0_mean": previous["full_partial_mean"], "Mclean_mean": q["full_partial_mean"],
            "M0_exact_correct": int(Fraction(previous["full_exact_rate"]) * 18),
            "Mclean_exact_correct": int(Fraction(q["full_exact_rate"]) * 18),
            "M0_selected": previous["selected_by_policy"], "Mclean_selected": q["selected_by_policy"],
            "partial_delta": str(Fraction(q["full_partial_mean"]) - Fraction(previous["full_partial_mean"])),
        })
    summary = {"old_questions_M0": aggregate(old_rows),
               "old_questions_Mclean": aggregate(results["questions"]),
               "R02_questions_Mclean": aggregate(r02_rows)}
    output = {"status": "COMPLETE", "paired_question_count": 7, "new_solver_calls": 126,
              "new_generation_calls": 0, "ground_calls": 0, "training_started": False,
              "historical_results_overwritten": False, "comparison_preflight": proof,
              "summary": summary, "paired_questions": pairs,
              "limitations": ["Seven questions across two roots; no estimate of broad-scale training yield.",
                              "Old/new questions differ in content, Gold cardinality and dialogue; this is not an isolated expansion-rule ablation.",
                              "No new semantic certification of Qwen questions or Gold was performed."]}
    (HERE / "COMPARISON.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 旧七题 M_clean 补评", "",
             "补评完成：126/126 次有效回答。题干、选项、Gold、旧 P+R01 对话、42 个排列、采样种子、126 份提示词及解码配置与旧 M0 评估一致。仅替换为本次 R02 使用的同一 M_clean 权重。", "",
             "未重新生成题目、未执行 ground、未训练，未覆盖历史结果。", "",
             "| 条件 | 题数 | 入选 | 18/18 全对题数 | 平均部分分 | 整题正确 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, label in [("old_questions_M0", "旧题 / M0"), ("old_questions_Mclean", "旧题 / M_clean"),
                       ("R02_questions_Mclean", "R02 新题 / M_clean（复用已有结果）")]:
        s = summary[key]
        lines.append(f"| {label} | {s['question_count']} | {s['selected']} | {s['perfect_questions']} | "
                     f"{s['mean_partial_decimal']:.4f} | {s['exact_correct_calls']}/{s['valid_calls']} ({100*s['exact_rate_decimal']:.2f}%) |")
    lines += ["", "## 同题模型对照", "",
              "| 题号 | M0 部分分 | M_clean 部分分 | M0 全对次数 | M_clean 全对次数 | M_clean 入选 |",
              "|---|---:|---:|---:|---:|---|"]
    for p in pairs:
        lines.append(f"| {p['candidate_id']} | {p['M0_mean']} | {p['Mclean_mean']} | "
                     f"{p['M0_exact_correct']}/18 | {p['Mclean_exact_correct']}/18 | {'是' if p['Mclean_selected'] else '否'} |")
    lines += ["", "本次实测中，同一七道旧题的部分分均随 M_clean 提高，整题正确率由 35.71% 升至 66.67%，符合训练后 Solver 在这些题上表现更好的判断。统一到 M_clean 后，旧题仍有 5/7 入选，R02 新题只有 2/7；新题整题正确率为 98.41%。因此模型版本变化不能解释全部差距，本批新题在当前评估下仍更容易。",
              "", "比较只覆盖两个 root 的七道旧题、七道新题。新旧题的题干、考查事实、Gold 数量及上下文均不同，即使统一 Solver，也不能把差异全部归因于扩充规则。",
              "", "全部源文件、模型绑定及提示词一致性记录见 COMPARISON_PREFLIGHT.json；逐回答结果见 runs/old7_m_clean_01/responses。"]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": summary, "paired_questions": pairs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
