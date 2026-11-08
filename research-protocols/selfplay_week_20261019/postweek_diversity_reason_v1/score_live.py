"""Score only after raw responses are sealed; never import this in generation."""
from __future__ import annotations

import json
from pathlib import Path

import run as diagnostic


def main():
    out = diagnostic.HERE / "outputs"
    frozen = diagnostic.read(out / "RAW_RESPONSE_FREEZE.json")
    for bound in frozen["bindings"]:
        diagnostic.check(bound)
    packet = diagnostic.read(out / "REASON_REVIEW_PACKET.json")
    private_path = diagnostic.ORIGINAL / "frozen_inputs/private/PRIVATE_QUESTIONS.json"
    golds = {q["candidate_id"]: q for q in diagnostic.read(private_path)["questions"]}
    records = []
    for record in packet["requests"]:
        diagnostic.check(record["raw_binding"])
        gold = golds[record["candidate_id"]]["gold"]
        prediction = record.get("canonical_prediction", [])
        valid = record["status"] == "STRUCTURALLY_VALID_REASONS_SAVED"
        exact = valid and (set(prediction) == set(gold) if record["question_type"] == "multi_select" else prediction == gold)
        records.append({"candidate_id": record["candidate_id"], "root_id": record["root_id"],
            "question_type": record["question_type"], "task_operation": record["task_operation"],
            "valid": valid, "exact": exact, "prediction_canonical": prediction, "gold_canonical": gold,
            "display_answer": record.get("answer_labels"), "raw_binding": record["raw_binding"]})
    totals = {}
    for task in ("all", "multi_select", "ordering"):
        selected = [r for r in records if task == "all" or r["question_type"] == task]
        correct = sum(r["exact"] for r in selected)
        totals[task] = {"correct": correct, "denominator": len(selected), "exact_rate": correct / len(selected)}
    diagnostic.save(out / "SCORES.json", {
        "status": "SCORED_AFTER_RAW_FREEZE", "model_role": "B1", "roots": 4,
        "one_display_one_seed_per_question": True, "formal_weakness_admission": False,
        "cot_causal_effect_comparison": False, "historical_model_and_budget_differ": True,
        "totals": totals, "records": records, "gold_binding": diagnostic.bind(private_path),
        "raw_freeze_binding": diagnostic.bind(out / "RAW_RESPONSE_FREEZE.json")})
    lines = ["# 七种出题操作的真实理由诊断", "",
        "固定7题、4个Train roots，每题一个显示顺序和一个种子，全部完整来源输入；模型为B1。原始返回先封存，随后才由本评分程序读取Gold。",
        "历史直接作答使用M-clean且输出预算不同，因此本批不能估计CoT收益、难度变化或反馈的训练收益。", "",
        f"多选 {totals['multi_select']['correct']}/4，排序 {totals['ordering']['correct']}/3，合计 {totals['all']['correct']}/7。这是小样本诊断记录。", "",
        "|题目|操作|完全正确|", "|---|---|---|"]
    for row in records:
        lines.append(f"|{row['candidate_id']}|{row['task_operation']}|{'是' if row['exact'] else '否'}|")
    for record in packet["requests"]:
        raw = diagnostic.read(record["raw_binding"]["path"])
        row = record["request"]
        lines.extend(["", f"## {record['candidate_id']}", "", row["question"], ""])
        lines.extend(f"- {label}. {text}" for label, text in row["display"].items())
        lines.extend(["", "```json", raw["raw_output"], "```", "",
            f"原始输出：`{record['raw_binding']['path']}`", "",
            "格式有效不代表引用或理由可靠；逐项语义审核单独保存在SEMANTIC_REVIEW_AI文件中。"])
    with (out / "REPORT_ZH.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
