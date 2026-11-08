"""Recompute exact/pair scores from frozen raw boxes and publish pilot artifacts."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
from statistics import mean

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binding(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def main():
    run = HERE / "runtime/direct_v1"
    scores = read(run / "SCORES.json")
    freeze = read(run / "RAW_RESPONSE_FREEZE.json")
    assert freeze["status"] == "COMMITTED"
    for row in freeze["files"]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
    ledger = read(HERE / "exports/CANDIDATE_LEDGER.json")
    metadata = {row["candidate_id"]: row for row in ledger["records"]}
    manifest = read(run / "manifest.json")
    public = {row["candidate_id"]: row for row in read(run / "PUBLIC_QUESTIONS.json")["questions"]}
    private = {row["candidate_id"]: row for row in read(HERE / "exports/PRIVATE_QUESTIONS.json")["questions"]}
    mappings = read(run / "DISPLAY_MAPPINGS.json")
    native = {row["request_id"]: row for row in scores["requests"]}
    audit_errors, calls = [], []
    for request in manifest["requests"]:
        rid, qid = request["request_id"], request["candidate_id"]
        raw = read(run / "responses" / (rid + ".json"))
        text = raw.get("raw_output")
        match = re.fullmatch(r"<box>([A-FI0-9,]+)</box>", text or "")
        labels = match.group(1).split(",") if match else None
        mapping = mappings["requests"][rid]["display_to_content"]
        valid = bool(raw["status"] == "RETURNED" and match and labels
            and len(labels) == len(set(labels)) and set(labels) <= set(mapping)
            and not raw.get("output_validation_errors")
            and raw.get("metadata", {}).get("finish_reason") in ("stop", "eos"))
        typ = request["question_type"]
        if typ == "ordering":
            valid = valid and set(labels or []) == set(mapping)
        prediction = [mapping[label] for label in labels] if valid else []
        canonical = mappings["public_to_canonical"][qid]
        gold = [canonical[label] for label in private[qid]["gold"]]
        exact = bool(valid and (set(prediction) == set(gold) if typ == "multi_select" else prediction == gold))
        if valid != native[rid]["valid"] or exact != native[rid]["exact"]:
            audit_errors.append({"request_id": rid, "kind": "EXACT_OR_VALID_MISMATCH"})
        n_pairs, correct_pairs = 0, 0
        if typ == "ordering":
            n_pairs = len(gold) * (len(gold) - 1) // 2
            rank = {value: i for i, value in enumerate(prediction)}
            correct_pairs = sum(valid and rank[left] < rank[right] for left, right in itertools.combinations(gold, 2))
            if n_pairs != native[rid]["metrics"]["total_pairs"] or correct_pairs != native[rid]["metrics"]["correct_pairs"]:
                audit_errors.append({"request_id": rid, "kind": "PAIR_MISMATCH"})
        reverse = {value: label for label, value in canonical.items()}
        fp = len(set(prediction) - set(gold)) if valid and typ == "multi_select" else None
        fn = len(set(gold) - set(prediction)) if valid and typ == "multi_select" else None
        calls.append({"request_id": rid, "candidate_id": qid, "root_id": request["root_id"],
            "question_type": typ, "template_id": metadata[qid]["template_id"],
            "display_prediction": labels, "canonical_prediction": [reverse[value] for value in prediction] if valid else None,
            "canonical_gold": private[qid]["gold"], "raw_output": text, "valid": valid, "exact": exact,
            "item_count": len(mapping), "pair_correct": correct_pairs if typ == "ordering" else None,
            "pair_total": n_pairs if typ == "ordering" else None,
            "false_positive_options": fp, "false_negative_options": fn,
            "display_to_canonical": {display: reverse[value] for display, value in mapping.items()},
            "raw_response_path": str((run / "responses" / (rid + ".json")).resolve())})
    save(HERE / "review/POST_SCORE_RECOMPUTATION.json", {"status": "PASS" if not audit_errors else "FAIL",
        "reviewer": "/root independent literal-box/content-map recomputation, without native scoring helpers",
        "requests": len(calls), "exact_recomputed": True, "all_ordering_pairs_recomputed": True,
        "errors": audit_errors, "raw_freeze": binding(run / "RAW_RESPONSE_FREEZE.json")})
    assert not audit_errors, audit_errors
    questions = []
    for row in scores["questions"]:
        qid = row["candidate_id"]
        meta = metadata[qid]
        questions.append({key: value for key, value in row.items() if key not in ("items", "pairs", "design")} | {
            "template_id": meta["template_id"], "primary_relation": meta["primary_relation"],
            "timeline_kind": meta["timeline_kind"], "fact_family_id": meta["fact_family_id"],
            "gold_count": meta["actual_gold_count"], "canonical_gold": private[qid]["gold"],
            "self_contained_operands_or_constraints": meta["self_contained_operands_or_constraints"],
            "strong_intrinsic_procedure_cues": meta["strong_intrinsic_procedure_cues"],
            "question": public[qid]["question"]})
    with (HERE / "exports/SOLVER_RESPONSES.jsonl").open("w") as stream:
        for row in calls:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(HERE / "exports/PER_QUESTION.json", {"questions": questions})
    fields = ["candidate_id", "root_id", "question_type", "template_id", "item_count", "gold_count",
        "denominator_calls", "valid_calls", "invalid_calls", "exact_calls", "exact_rate", "pair_accuracy",
        "fp_calls", "fn_calls", "stratum", "fact_family_id", "self_contained_operands_or_constraints"]
    with (HERE / "per_question.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(questions)
    groups = {}
    for template in dict.fromkeys(row["template_id"] for row in ledger["records"]):
        planned = [row for row in ledger["records"] if row["template_id"] == template]
        subset = [row for row in questions if row["template_id"] == template]
        result = {"planned_slots": len(planned), "admitted": len(subset),
            "holds": sum(row["status"] == "CONTENT_HOLD" for row in planned),
            "shortfalls": sum(row["status"] == "SOURCE_SHORTFALL" for row in planned),
            "calls": sum(row["denominator_calls"] for row in subset),
            "question_equal_exact": mean(row["exact_rate"] for row in subset) if subset else None,
            "arity_counts": dict(Counter(row["item_count"] for row in subset))}
        if subset and subset[0]["question_type"] == "ordering":
            result["question_equal_pair_accuracy"] = mean(row["pair_accuracy"] for row in subset)
        groups[template] = result
    roots = {}
    for source in read(HERE / "GENERATION_FREEZE.json")["sources"]:
        root = source["root_id"]
        subset = [row for row in questions if row["root_id"] == root]
        planned = [row for row in ledger["records"] if row["root_id"] == root]
        roots[root] = {"planned_slots": len(planned), "admitted": len(subset),
            "holds": sum(row["status"] == "CONTENT_HOLD" for row in planned),
            "shortfalls": sum(row["status"] == "SOURCE_SHORTFALL" for row in planned),
            "MC": sum(row["question_type"] == "multi_select" for row in subset),
            "ordering": sum(row["question_type"] == "ordering" for row in subset)}
    summary = {"schema_version": "diversity_crossroot_pilot_summary_v1", "status": "COMPLETE_TRAIN_FEASIBILITY_DIAGNOSTIC",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "roots": roots,
        "production_planned_slots": 24, "production_status_counts": ledger["counts"],
        "admitted_questions": len(questions), "by_type": scores["by_type"], "by_arity": scores["by_arity"],
        "by_template": groups, "gold_count_distribution_MC": dict(Counter(row["gold_count"] for row in questions if row["question_type"] == "multi_select")),
        "fact_families": len({row["fact_family_id"] for row in questions}),
        "source_independent_inputs_admitted": [row["candidate_id"] for row in questions if row["self_contained_operands_or_constraints"]],
        "strata": {typ: dict(Counter(row["stratum"] for row in questions if row["question_type"] == typ)) for typ in ("multi_select", "ordering")},
        "diagnosis": {"calls": len(calls), "valid_calls": sum(row["valid"] for row in calls),
            "invalid_calls": sum(not row["valid"] for row in calls), "raw_responses_frozen_before_scoring": True,
            "output_mode": "Finite-choice constrained decoding, direct answer only; validity does not measure unconstrained format compliance.",
            "max_prompt_tokens": manifest["max_prompt_tokens"], "source_truncation": False, "cost": scores["cost"]},
        "questions": questions, "solver_model": read(HERE / "EXECUTION_FREEZE.json")["solver"],
        "feedback": binding(HERE / "exports/MATERIAL_FEEDBACK.json"),
        "limits": ["Four historically exposed Train roots, not unseen evaluation; two appeared in preceding pilot.",
            "Content PASS is not training admission or a formal measured reliability rate.",
            "Time ordering includes proposed schedule/cutoff markers; procedure includes completed local operations and planned agenda.",
            "All four procedure questions have3items, strong intrinsic cues, and report-order overlap.",
            "Four admitted questions restate all relevant operands/conditions; evidence-selection also supplies candidate excerpts by design.",
            "Question/fact-family reuse, different arity and template mix prevent controlled difficulty or training-benefit claims.",
            "No source-masked control, feedback A/B, DEV/TEST calls, or training in this pilot."],
        "training_started": False, "new_training_admitted": 0, "dev_test_calls": 0, "feedback_effect_proved": False,
        "old_frozen187_modified": False, "next_action": "Use material gaps and shortcut observations in a new frozen Train expansion/authoring loop; preserve matched controls for any feedback-benefit claim."}
    mc_calls = [row for row in calls if row["question_type"] == "multi_select"]
    summary["MC_observable_errors"] = {
        "exact_calls": sum(row["exact"] for row in mc_calls), "denominator_calls": len(mc_calls),
        "false_positive_option_decisions": sum(row["false_positive_options"] or 0 for row in mc_calls),
        "false_negative_option_decisions": sum(row["false_negative_options"] or 0 for row in mc_calls),
        "responses_with_FP": sum(bool(row["false_positive_options"]) for row in mc_calls),
        "responses_with_FN": sum(bool(row["false_negative_options"]) for row in mc_calls),
        "note": "FP/FN option decisions can exceed response counts. Both counts describe outputs, not an internal causal diagnosis."
    }
    error_review = HERE / "review/SOLVER_OBSERVABLE_ERROR_ANALYSIS.json"
    if error_review.exists():
        summary["observable_error_analysis"] = {
            "path": str(error_review.resolve()),
            "hash_bound_by": "FINAL_MANIFEST.json; avoid cyclic hash binding because the analysis audits this summary."
        }
    save(HERE / "SUMMARY.json", summary)
    pct = lambda value: f"{100 * value:.2f}%" if value is not None else "—"
    labels = {"MC_CAUSAL_SUPPORT": "多选：明确原因", "MC_QUANTITATIVE_RELATION": "多选：数量/时间计算",
        "MC_CONSTRAINT_CONJUNCTION": "多选：条件联合", "MC_EVIDENCE_SELECTION": "多选：最小证据集",
        "ORDER_DISCOURSE_RECONSTRUCTION": "排序：叙述顺序", "ORDER_EXPLICIT_TIME": "排序：明确计划时刻", "ORDER_EXPLICIT_PROCEDURE": "排序：明确步骤"}
    text = ["# 新形式跨 root 小批验证", "", f"固定4个Train root、24个题位，内容通过{len(questions)}道、暂缓{ledger['counts'].get('CONTENT_HOLD',0)}道、素材缺额{ledger['counts'].get('SOURCE_SHORTFALL',0)}道。S0=M-clean完成{len(calls)}次直接作答，有效{sum(row['valid'] for row in calls)}次；无自动重试、无对话截断。通过题全部评估，未按分数修改准入。", "",
        "## 生成与诊断", "", "|考察关系|计划|通过|暂缓|缺额|项数|完整正确率（题目等权）|", "|---|---:|---:|---:|---:|---|---:|"]
    for template, result in groups.items():
        text.append(f"|{labels[template]}|{result['planned_slots']}|{result['admitted']}|{result['holds']}|{result['shortfalls']}|{','.join(str(n) for n in result['arity_counts'])}|{pct(result['question_equal_exact'])}|")
    text.extend(["", "不同项数排序必须分层；三项全序只有6种合法答案，六项有720种，完整正确率不可直接视作同等难度。", "",
        "|排序项数|题数|调用数|完整正确率（题目等权）|成对正确率（题目等权）|", "|---|---:|---:|---:|---:|"])
    for key, result in scores["by_arity"].items():
        if result["question_type"] == "ordering":
            text.append(f"|{result['item_count']}|{result['questions']}|{result['calls']}|{pct(result['question_equal_exact'])}|{pct(result['question_equal_pair_accuracy'])}|")
    text.extend(["", "|root|通过/计划|多选|排序|暂缓|缺额|", "|---|---:|---:|---:|---:|---:|"])
    for root, result in roots.items():
        text.append(f"|{root}|{result['admitted']}/6|{result['MC']}|{result['ordering']}|{result['holds']}|{result['shortfalls']}|")
    text.extend(["", f"多选Gold分布：{summary['gold_count_distribution_MC']}；{len(questions)}道题只对应{summary['fact_families']}个作者标注事实族，不能按题数当作独立素材数。", "",
        "## 本轮实际观察", "",
        f"多选题目等权完整正确率{pct(scores['by_type']['multi_select']['question_equal_exact'])}（61/162次全对）。2道明确原因题36/36全对；3道数量/时间计算题0/54全对；2道条件联合题为25.00%，2道最小证据题为44.44%。每种关系仅2—3题，这些是本批表现，尚不能升级为跨题族稳定弱点或宣称比旧题更难。", "",
        f"多选共有{summary['MC_observable_errors']['false_positive_option_decisions']}次错误选项入选、{summary['MC_observable_errors']['false_negative_option_decisions']}次正确选项漏选；它们是逐选项判定次数，不能当成作答次数，也不能把本轮概括为单纯过选。R3-MC2的冗余证据B在18/18次入选；R3-MC1的合法无网方案A、E均0/18次入选，分别体现冗余保留和正确方案遗漏。", "",
        "R4-MC3的错误F（超过中午截止8分钟）18/18次入选，其中14次同时选择正确D（提前1小时52分钟），答案集合内部直接矛盾。可先针对这种可核查矛盾设计一致性反馈；仅有直接答案记录，不能据此判定模型具体如何算错。", "",
        "排序总体题目等权60.61%，调用等权47/96＝48.96%，两种权重不同。3项题91.67%（33/36），包括四道步骤题87.50%（21/24）和两道直接时间排序100%（12/12）；4项题16.67%（2/12）；6项叙述题25.00%（12/48），成对正确率90.28%。这表示长全序仍有局部错位，三项步骤/时刻题偏易的观察更明确，不能用总体排序分数掩盖项数差异。", "",
        "本次使用有限合法答案集合约束解码，因此258/258格式有效不能解释为模型在自由输出中也100%遵守格式。推理执行约89.72秒（含约19.50秒模型加载），这不包括作者生成、内容审核和运行准备时间。", "",
        "## 内容与形式的结论", "", "这批实际落地了原因、计算、条件联合和证据选择四种多选关系，以及叙述、计划时刻和步骤三种排序依据。四道步骤题都只有3项，并与报告顺序重合；其正确率不足以证明已获得困难的过程推理题。", "",
        "4道准入题把全部必要运算数或约束条件写在题面/条目中，可以绕过长对话检索；步骤题也有first/again、初稿/修订/发送等语义线索。这些是内容审核发现的潜在捷径，本轮没有做移除原对话的模型对照，不能声称已量化模型对对话或memory的依赖。证据题提供候选引文本身符合该任务定义，也不应当作全历史记忆能力。", "",
        "DIVX-R2-MC2的Gold A/B/C/E/F正确，但只有D因质量下降被排除，仅看quality就能得到整套答案。它未满足候选集上至少两个非冗余条件的目标，暂缓。6选5的联合条件筛选只有一个负例，无法让两条排除条件分别成为候选判别必需项；后续可把这种门槛放在Gold2–4，Gold5保留给适合的计算等题型。本轮冻结题位没有改写替补。", "",
        "另外3个缺额分别是：R2缺少3个同轴可比时间点；R3没有找到同一决定的4个不同明确理由；R4没有找到自然的5项不可缺少证据集。这表示本轮未找到合格题，不证明这些root永远不可出题。", "",
        "## 下一步", "", "将素材缺口和题面捷径反馈到下一轮Train对话扩充与出题：优先补充自然的多阶段依赖、明确时间锚点和决定理由；为记忆检索题减少题面对必要事实的完整复述。先冻结下一批题型/Gold配额并继续保留缺额：条件联合优先Gold2—4，最小证据不强配Gold5。继承旧事件延续3场＋数据集新事件3场，两条扩充路线与出题A/B实验组分别记账。素材/覆盖反馈和Solver可观察错误分别追踪，不把本批新发现直接升为已复现弱点。若要证明反馈有效，需要同root、同预算的材料扩充对照，不能用本轮新旧题准确率之差代替。", "",
        "本轮未训练、未入训练池、未调用项目DEV/TEST，旧187道冻结题与结果保持封存。4root全部为历史已暴露Train，其中2个参加过上一批试写。本轮是可行性和Solver诊断，不是未见泛化、正式难度对照或反馈因果实验。", "",
        "## 文件", "", "- `exports/ALL_AUTHOR_CANDIDATES.jsonl`：24题位全部作者稿，含失败与答案证据。", "- `exports/PUBLIC_QUESTIONS.json`、`exports/PRIVATE_QUESTIONS.json`：准入题与私有答案。", "- `exports/SOLVER_RESPONSES.jsonl`：每次展示答案、还原答案、Gold与原始输出路径。", "- `per_question.csv`、`exports/PER_QUESTION.json`：逐题正确率和错误统计。", "- `review/`：主审、不同作者代理复核、引文核查及原始输出独立重算。", "- `exports/MATERIAL_FEEDBACK.json`：素材缺口和后续生成反馈，尚未消费到新扩充。", "- `runtime/direct_v1/`：完整公共提示、随机映射、原始作答与封存记录。", ""])
    (HERE / "REPORT_ZH.md").write_text("\n".join(text))
    print(json.dumps({key: summary[key] for key in ("status", "admitted_questions", "production_status_counts", "by_type", "by_arity", "strata")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
