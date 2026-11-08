#!/usr/bin/env python3
"""Summarize audited v137 fit results beside sealed v127/v134 results.

No inference, model loading, training, question selection, or score parsing occurs
here. Run once after the official report and independent arithmetic audit finish.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


HISTORICAL = {
    "v127": (
        "snapshots/v132/evaluations/dev_v127/PAIRED_SUMMARY.json",
        "50ff538ba97df6da79b3772df3b4aa72015aaffbc239f0ea7148aa7829336d85",
    ),
    "v134": (
        "snapshots/v134/evaluation/reports/STRATIFIED.json",
        "26ab9e3466b6f95f7f6ecd6eaf253fde270ddc42fcf69d2f89ed150f94774700",
    ),
    "v134_summary": (
        "snapshots/v134/evaluation/reports/SUMMARY.json",
        "6d3c82339b4e20336ea1adab2512f8ebf9e34a8059b04cf8f6d72efdee9060c4",
    ),
    "v128": (
        "snapshots/v132/evaluations/ordering_ceiling_v128/REPORT_ZH.md",
        "5c413a5fb982a2baa283259e6248bc482df708e549cc4a2176363a8a2fc75789",
    ),
}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sources:
    def __init__(self):
        self.bindings = {}

    def bind(self, path, expected=None):
        path = path.resolve()
        require(path.is_file(), "NOT_READY_MISSING_FILE: " + str(path))
        actual = sha(path)
        require(expected is None or actual == expected,
                "SOURCE_HASH_MISMATCH: " + str(path))
        self.bindings[str(path)] = {
            "path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())


def metric(value, fraction=None):
    if value is not None:
        require(isinstance(value, (int, float)) and math.isfinite(value),
                "NONFINITE_METRIC")
        if fraction is not None:
            require(abs(value - float(Fraction(fraction))) < 1e-12,
                    "FRACTION_MISMATCH")
    return {"value": value, "fraction": fraction}


def group_row(source_path, source_key, group, row_id, title, question_type,
              fixed_roots, expected_n, expected_roots, *, old=False):
    require(group["candidate_count"] == expected_n,
            "UNEXPECTED_GROUP_COUNT: " + row_id)
    covered = expected_roots if old else group["covered_root_count"]
    require(covered == expected_roots, "UNEXPECTED_ROOT_COUNT: " + row_id)
    result = {
        "row_id": row_id, "title": title, "question_type": question_type,
        "question_count": expected_n, "covered_root_count": covered,
        "fixed_root_population": fixed_roots,
        "source_path": str(source_path.resolve()), "source_json_pointer": source_key,
        "weighting": "question_equal", "models": {},
    }
    for model in ("clean", "new"):
        source_model = "baseline" if old and model == "clean" else model
        base = source_key + "/models/" + source_model
        qkey = "equal_candidate" if old else "question_equal"
        q = group["models"][source_model][qkey]
        require(q["candidate_count"] == expected_n, "MODEL_GROUP_COUNT_MISMATCH")
        fields = {
            "exact": base + "/" + qkey + "/exact_rate_equal_candidate",
            "signed_partial": base + "/" + qkey + "/signed_partial_equal_candidate",
        }
        entry = {
            "source_model_key": source_model,
            "exact": metric(q["exact_rate_equal_candidate"],
                            q.get("exact_rate_equal_candidate_fraction")),
            "signed_partial": metric(q["signed_partial_equal_candidate"],
                                     q.get("signed_partial_equal_candidate_fraction")),
            "ordering_pair_accuracy": metric(None), "source_fields": fields,
        }
        if question_type == "ordering":
            # Every ordering question has 12 calls; this pooled old pair metric
            # therefore equals the mean over questions, unlike a mixed-type pool.
            if old:
                p = group["models"][source_model]["pooled_calls"]["ordering"]["pair_accuracy"]
                fields["ordering_pair_accuracy"] = base + "/pooled_calls/ordering/pair_accuracy/mean"
                entry["ordering_pair_accuracy"] = metric(p["mean"], p.get("mean_fraction"))
            else:
                p = q["ordering_pair_accuracy_equal_candidate"]
                fields["ordering_pair_accuracy"] = base + "/question_equal/ordering_pair_accuracy_equal_candidate/mean"
                entry["ordering_pair_accuracy"] = metric(p["mean"])
            v = entry["ordering_pair_accuracy"]["value"]
            signed = entry["signed_partial"]["value"]
            require(v is None or signed is None or abs(v - (signed + 1) / 2) < 1e-12,
                    "ORDERING_PAIR_SIGNED_IDENTITY_FAILED")
        result["models"][model] = entry
    return result


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def markdown(rows, supplementary, gaps, overlap, sources):
    lines = [
        "# 训练池拟合与既有 DEV/TEST 的描述性对照", "",
        "本报告在 v137 正式结果 COMPLETE、独立原始回答复算 PASS 后生成。只读取已保存成绩，不新增训练或推理，不修改题目、Gold 或历史结果。",
        "", "主表按题目等权：先平均每题的展示与采样，再平均题目。所有数字为百分比。M-clean 是第二轮训练起点，Round2 是冻结的训练后模型。Gold6 全对控制不进入主表。", "",
        "| 范围 | 题数 / 有题 root | M-clean 整题正确率 | Round2 整题正确率 | M-clean 部分得分 | Round2 部分得分 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        models = row["models"]
        vals = [models[m][k]["value"] for k in ("exact", "signed_partial") for m in ("clean", "new")]
        lines.append("| " + row["title"] + " | " + str(row["question_count"]) + " / " +
                     str(row["covered_root_count"]) + " | " + " | ".join(map(percent, vals)) + " |")
    lines += ["", "排序的成对准确率单列：", "",
              "| 范围 | M-clean 成对准确率 | Round2 成对准确率 |",
              "|---|---:|---:|"]
    for row in rows:
        if row["question_type"] == "ordering":
            lines.append("| " + row["title"] + " | " + " | ".join(
                percent(row["models"][m]["ordering_pair_accuracy"]["value"])
                for m in ("clean", "new")) + " |")
    lines += ["", "多选部分得分为 TP/g−FP/(6−g)；排序部分得分为（正确对数−错误对数）/15。部分得分允许负值，不是部分答对题的占比。排序成对准确率为正确对数/15，与部分得分的关系是 (部分得分+1)/2。", "",
              "全部 363 道训练题覆盖 68 个 root；多选与排序各覆盖其中 67 个 root。按调用平均会让每题18次的多选比每题12次的排序获得更高权重，因此本表不采用调用等权总分。训练全集和 Gold 数量分层如下：", "",
              "| 训练范围 | 题数 / 有题 root | M-clean 整题正确率 | Round2 整题正确率 | M-clean 部分得分 | Round2 部分得分 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in supplementary:
        vals = [row["models"][m][k]["value"] for k in ("exact", "signed_partial") for m in ("clean", "new")]
        lines.append("| " + row["title"] + " | " + str(row["question_count"]) + " / " +
                     str(row["covered_root_count"]) + " | " + " | ".join(map(percent, vals)) + " |")
    lines += ["", "训练减去既有评估的整题正确率差值如下，单位为百分点；它们只是不同题集间的描述性差值，不是严格同分布的泛化差距估计。", "",
              "| 训练题型 − 评估范围 | M-clean 差值 | Round2 差值 |",
              "|---|---:|---:|"]
    for gap in gaps:
        vals = [gap["models"][m]["exact_difference_percentage_points"] for m in ("clean", "new")]
        lines.append("| " + gap["train_title"] + " − " + gap["evaluation_title"] + " | " +
                     " | ".join("—" if x is None else f"{x:+.2f}" for x in vals) + " |")
    lines += ["", "## 如何解释这些结果", "",
              "v137 重新评估的是实际训练过的原题和完整对话。它采用与 v134 相同的 v15 公开提示，原训练使用 v122 用户格式，因此这不是训练 prompt 的逐字重放。随机选项展示用于检查位置与标签适应，不能把训练原题变成未见题。",
              f"本次每模型有 {overlap['evaluation_distinct_displays_per_model']} 种不同展示、{overlap['evaluation_calls_per_model']} 次调用，其中 {overlap['overlapping_display_count']} 种展示（{overlap['overlapping_calls_per_model']} 次调用）与原训练展示相同；完整训练消息 prompt 重合调用为 {overlap['full_training_message_prompt_overlap_calls_per_model']}。这些是源文件核验结果，不是新的模型实验。",
              "训练集高分表明模型在已见题上拟合良好；训练题与当前评估题同时高分仍不能排除固定模板过拟合。训练与 DEV/TEST 的 root、题数、Gold 数量和来源难度分布不同；新增评估来自历史失败 root 的补救流程与隔离安全场景，不能将表中的差距作为受控实验。",
              "v128 已证实 125 道训练排序题与 14 道原 DEV 排序题使用完全相同的题干，六个固定场景各取一项；其原 DEV 规则检索基线也可全部排对。这项模板天花板证据只对该已审核范围成立；此报告不把它未经复核扩展为新增 DEV/TEST 全部题目的结论。它也不直接证明多选存在相同天花板。",
              "旧 M-clean 的排序监督还存在历史噪声、四项到六项的格式变化。这些因素与模板适应、新监督可能共同影响提升；本轮不能分离它们，也不能证明扩充反馈或出题反馈的独立因果收益。",
              "同 root 的题、Gold 家族、随机排列与重复采样相关，11568 次计划调用不等于11568道独立题。原 DEV 仅7个有题 root；新增 DEV 为5个、新增 TEST 为2个，最新 TEST 排序只有4题。没有进行把调用当独立样本的显著性推断。",
              "本报告只引用原 DEV 48题中的主任务41题，以及新增49题中的主任务42题。扩展 DEV/TEST 共184题没有在本轮全部重评，原 TEST 87题的成绩没有在这里补造。Gold6 控制仍保留于历史原报告，但训练池没有 Gold6。",
              "TEST 曾在生成和审核流程中有历史曝光，不能声称是从未接触的盲测；这本身也不是 Solver 把 TEST 用于训练的证据。已核验本轮实际入训只有363道 train 题，最新49道 DEV/TEST 未入训。",
              "v137 使用 batch=8 的 vLLM 调度，历史 v127/v134 使用串行提交。每请求展示、种子、采样与公开提示合同沿用既有协议；不声称批调度与历史串行运行逐 token 数值等价，不能把跨报告的小差异解释成新的模型变化。",
              "因此本结果可用于交接训练拟合情况与当前题型表现，不能单独回答跨模板泛化是否可靠。要排除模板过拟合，需要冻结模型后用新 root 与不同结构的题目作独立评估；本轮没有开展该新实验。", "",
              "## 复算来源", "",
              "FIT_COMPARISON.json 保存每行原始小数、精确分数字段、来源 JSON 指针、SHA256，以及整题/部分/成对分数的描述性差值。历史快照保持原样。"]
    for binding in sources.bindings.values():
        lines.append("- `" + binding["path"] + "`，SHA256 `" + binding["sha256"] + "`。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history-bundle", type=Path,
                        default=Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_round2_lora_bundle_v136"))
    args = parser.parse_args()
    batch, history = args.batch.resolve(), args.history_bundle.resolve()
    out = batch / "diagnostics"
    json_out, md_out = out / "FIT_COMPARISON.json", out / "FIT_DIAGNOSTIC_ZH.md"
    require(not json_out.exists() and not md_out.exists(), "OUTPUT_EXISTS_REFUSE_OVERWRITE")
    sources = Sources()
    summary_path = batch / "evaluation/reports/SUMMARY.json"
    stratified_path = batch / "evaluation/reports/STRATIFIED.json"
    summary = sources.read(summary_path)
    require(summary.get("status") == "COMPLETE" and summary.get("errors") == [],
            "NOT_READY_REPORT_NOT_COMPLETE")
    require(summary.get("question_count") == 363 and summary.get("calls_per_model") == 5784,
            "UNEXPECTED_V137_POPULATION")
    completion = sources.read(batch / "evaluation/RUN_COMPLETE.json")
    require(completion.get("status") == "COMPLETE", "NOT_READY_INFERENCE_NOT_COMPLETE")
    audit = sources.read(batch / "evidence/INDEPENDENT_RESULTS_AUDIT.json")
    require(audit.get("status") == "PASS" and audit.get("failure_count") == 0 and audit.get("errors") == [],
            "NOT_READY_INDEPENDENT_AUDIT_NOT_PASS")
    require(audit.get("fixed_questions") == 363 and audit.get("calls_per_model") == 5784,
            "AUDIT_POPULATION_MISMATCH")
    audited = {str(Path(b["path"]).resolve()): b["sha256"] for b in audit["audited_files"]}
    for path in (summary_path, stratified_path):
        require(audited.get(str(path.resolve())) == sha(path),
                "CURRENT_REPORT_NOT_BOUND_BY_PASS_AUDIT: " + str(path))
    stratified = sources.read(stratified_path)
    overlap = sources.read(batch / "evidence/DISPLAY_OVERLAP_TRAINING.json")
    require(overlap.get("status") == "PASS" and overlap.get("training_questions") == 363,
            "DISPLAY_OVERLAP_NOT_PASS")
    sources.bind(batch / "PROTOCOL_ZH.md")
    sources.bind(Path(__file__))
    historical = {}
    paths = {}
    for name, (relative, expected_sha) in HISTORICAL.items():
        paths[name] = history / relative
        historical[name] = (sources.bind(paths[name], expected_sha).read_text() if name == "v128"
                            else sources.read(paths[name], expected_sha))
    for key in ("v127", "v134_summary"):
        require(historical[key].get("status") == "COMPLETE" and historical[key].get("errors") == [],
                "HISTORICAL_REPORT_INCOMPLETE: " + key)
    v127, v134 = historical["v127"], historical["v134"]
    rows = []
    for name, title, qtype, n, roots in [
        ("multi_select_gold2345", "训练多选 Gold2–5", "multi_select", 238, 67),
        ("ordering", "训练排序", "ordering", 125, 67),
    ]:
        rows.append(group_row(stratified_path, "/train/" + name, stratified["train"][name],
                              "train_" + qtype, title, qtype, 68, n, roots))
    for name, title, qtype, n in [
        ("multi_select_gold2345_27", "原 DEV 多选 Gold2–5", "multi_select", 27),
        ("ordering14", "原 DEV 排序", "ordering", 14),
    ]:
        require(v127["root_macro"][name]["covered_root_denominator"] == 7,
                "HISTORICAL_DEV_ROOT_COUNT_MISMATCH")
        rows.append(group_row(paths["v127"], "/strata/" + name, v127["strata"][name],
                              "old_dev_" + qtype, title, qtype, 15, n, 7, old=True))
    for split, split_label, fixed, roots, multi_n, order_n in [
        ("dev", "新增 DEV", 15, 5, 20, 10), ("test", "新增 TEST", 16, 2, 8, 4),
    ]:
        for name, title, qtype, n in [
            ("multi_select_gold2345", "多选 Gold2–5", "multi_select", multi_n),
            ("ordering", "排序", "ordering", order_n),
        ]:
            rows.append(group_row(paths["v134"], "/" + split + "/" + name, v134[split][name],
                                  "new_" + split + "_" + qtype, split_label + " " + title,
                                  qtype, fixed, n, roots))
    supplementary = []
    for name, title, n in [("all", "全部训练题", 363), ("gold2", "Gold2", 65),
                           ("gold3", "Gold3", 59), ("gold4", "Gold4", 57), ("gold5", "Gold5", 57)]:
        g = stratified["train"][name]
        supplementary.append(group_row(stratified_path, "/train/" + name, g,
                                        "train_" + name, title, "mixed" if name == "all" else "multi_select",
                                        68, n, 68 if name == "all" else g["covered_root_count"]))
    gaps = []
    for row in rows[2:]:
        train = next(r for r in rows[:2] if r["question_type"] == row["question_type"])
        gap = {"train_row_id": train["row_id"], "train_title": train["title"],
               "evaluation_row_id": row["row_id"], "evaluation_title": row["title"],
               "interpretation": "descriptive_train_minus_different_question_set_not_iid_generalization_gap",
               "models": {}}
        for model in ("clean", "new"):
            values = {}
            for name in ("exact", "signed_partial", "ordering_pair_accuracy"):
                a, b = train["models"][model][name]["value"], row["models"][model][name]["value"]
                values[name + "_difference_percentage_points"] = None if a is None or b is None else 100 * (a - b)
            gap["models"][model] = values
        gaps.append(gap)
    text = markdown(rows, supplementary, gaps, overlap, sources)
    result = {
        "schema_version": "v137_train_fit_historical_comparison_v1", "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "postprocess_only": True, "new_inference_calls": 0, "new_training_runs": 0,
        "official_v137_report_status": summary["status"], "independent_results_audit_status": audit["status"],
        "model_aliases": {"clean": "M_clean_policy", "new": "M_v125_newonly"},
        "main_rows": rows, "additional_training_strata": supplementary,
        "descriptive_train_minus_heldout": gaps, "display_overlap_training": overlap,
        "scheduler_difference": {"v137": "vLLM batch 8", "v127_v134": "serial submission",
                                 "claims_tokenwise_equivalence": False},
        "question_weighting": "mean_over_displays_and_samples_within_question_then_equal_questions",
        "source_bindings": list(sources.bindings.values()),
        "limitations": [
            "Training questions and sources were seen during SFT; new displays do not create unseen questions.",
            "v137 uses v15 public evaluation prompting, not verbatim v122 training-prompt replay.",
            "Train and prior DEV/TEST differ in roots, question and Gold distributions and source difficulty.",
            "184 extended DEV/TEST questions were not all re-evaluated; no historical TEST87 scores are invented.",
            "Questions, Gold families, permutations and repeated samples within roots are correlated.",
            "v128 template-ceiling audit covers 125 train ordering and 14 old DEV ordering; no unverified extension to new sets.",
            "High training and same-style heldout scores do not exclude template overfitting.",
            "Historical TEST exposure during generation/review is not evidence of Solver training contamination.",
            "Changes confound format adaptation, old supervision noise, new supervision and template learning.",
            "No isolated causal effect of expansion feedback or QA feedback is established.",
        ],
        "markdown_output": {"path": str(md_out), "sha256": hashlib.sha256(text.encode()).hexdigest()},
    }
    # Build both documents completely before any mutation. Exclusive creation
    # protects sealed or previously generated results from accidental overwrite.
    out.mkdir(parents=True, exist_ok=True)
    with json_out.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with md_out.open("x") as handle:
        handle.write(text)
    print(json.dumps({"status": result["status"], "output_json": str(json_out),
                      "output_markdown": str(md_out), "main_rows": len(rows),
                      "source_files": len(sources.bindings)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
