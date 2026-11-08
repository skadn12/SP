"""Read-only input aggregation and additive exports after paired diagnostics.

Run only after finalize_diagnostic_v1.py and both native Solver sides complete:
  python tools/report_diagnostic_v1.py --check-inputs
  python tools/report_diagnostic_v1.py

Default output: delivery/diagnostic_v2/RETRY_DIAGNOSTIC_REPORT.{json,md},
plus roots/root*/FULL_DIALOGUE.md and QUESTIONS_AND_ANSWERS.md.
Existing output directories/files are never replaced. This helper cannot
authorize training, repair the original procedure, generate QA or run a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parents[1]
NATIVE = BATCH.parent / "h4_feedback_training_pool_v15_qa/evaluation"
MODEL_IDS = {"base": "M0", "trained": "M_clean_policy"}
LIMITATIONS = {
    "evaluation_scope": "CONTENT_ONLY_DIAGNOSTIC",
    "batch_status": "CONTENT_AUDITED_PROCEDURE_HOLD",
    "batch_procedural_success": False,
    "training_pool_eligible": False,
    "training_started": False,
    "training_pool_merged": False,
    "native_selection_confers_training_eligibility": False,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Inputs:
    """Track every consumed input and detect any later byte change."""

    def __init__(self):
        self.files = {}

    def bind(self, path):
        path = Path(path).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(str(path) not in self.files or self.files[str(path)] == digest,
                "INPUT_CHANGED_WHILE_READING:" + str(path))
        self.files[str(path)] = digest
        return {"path": str(path), "sha256": digest}

    def read(self, path):
        self.bind(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def verify_binding(self, row):
        require(self.bind(row["path"]) == {k: row[k] for k in ("path", "sha256")},
                "BOUND_INPUT_CHANGED:" + row["path"])

    def verify(self):
        for path, digest in list(self.files.items()):
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,
                    "INPUT_CHANGED_BEFORE_REPORT_SAVE:" + path)

    def bindings(self):
        return [{"path": p, "sha256": s} for p, s in sorted(self.files.items())]


def load_pure_module(inputs, name, path):
    inputs.bind(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(value):
    value = Fraction(value)
    return {"fraction": str(value), "value": float(value)}


def mean(values):
    return number(sum(values, Fraction()) / len(values)) if values else None


def metric(records):
    """Call-micro metrics; option errors and ordering pair errors stay separate."""
    scores = [r["score"] for r in records]
    total = len(scores)
    ms = [r for r in records if r["question_type"] == "multi_select"]
    ordering = [r for r in records if r["question_type"] == "ordering"]
    ms_scores = [r["score"] for r in ms]
    order_scores = [r["score"] for r in ordering]
    all_select = sum(len(s["content_prediction"]) == 6 for s in ms_scores)
    fp_calls = sum(s["fp"] > 0 for s in ms_scores)
    cardinality_over = sum(len(s["content_prediction"]) > len(s["content_gold"]) for s in ms_scores)
    cardinality_under = sum(len(s["content_prediction"]) < len(s["content_gold"]) for s in ms_scores)
    out = {
        "calls": total,
        "candidates": len({r["candidate_id"] for r in records}),
        "exact_calls": sum(s["exact_match"] is True for s in scores),
        "exact_rate": number(Fraction(sum(s["exact_match"] is True for s in scores), total)) if total else None,
        "partial_mean": mean([Fraction(s["partial_score_fraction"]) for s in scores]),
        "option_confusion_counts": {
            "calls": len(ms),
            "tp": sum(s["tp"] for s in ms_scores),
            "fp": sum(s["fp"] for s in ms_scores),
            "fn": sum(s["fn"] for s in ms_scores),
            "all_select_calls": all_select,
            "all_select_rate": number(Fraction(all_select, len(ms))) if ms else None,
            "false_positive_calls": fp_calls,
            "false_positive_call_rate": number(Fraction(fp_calls, len(ms))) if ms else None,
            "cardinality_overselect_calls": cardinality_over,
            "cardinality_overselect_rate": number(Fraction(cardinality_over, len(ms))) if ms else None,
            "cardinality_underselect_calls": cardinality_under,
            "predicted_cardinality_counts": dict(sorted(Counter(str(len(s["content_prediction"])) for s in ms_scores).items())),
        },
        "ordering_pair_confusion_counts": {
            "calls": len(ordering),
            "correct_pairs": sum(s["tp"] for s in order_scores),
            "inverted_pairs": sum(s["fp"] for s in order_scores),
        },
    }
    per_candidate = defaultdict(list)
    for row in records:
        per_candidate[row["candidate_id"]].append(row["score"])
    out["candidate_macro_exact_rate"] = mean([
        Fraction(sum(s["exact_match"] is True for s in rows), len(rows))
        for rows in per_candidate.values()
    ])
    out["candidate_macro_partial_mean"] = mean([
        sum((Fraction(s["partial_score_fraction"]) for s in rows), Fraction()) / len(rows)
        for rows in per_candidate.values()
    ])
    return out


def grouped_metrics(records, metadata):
    groups = {"all": records}
    groups["multi_select"] = [r for r in records if r["question_type"] == "multi_select"]
    groups["multi_select_gold2_to_5"] = [r for r in records if r["question_type"] == "multi_select" and metadata[r["candidate_id"]]["gold_cardinality"] < 6]
    groups["multi_select_control6"] = [r for r in records if metadata[r["candidate_id"]]["role"] == "control6"]
    groups["ordering"] = [r for r in records if r["question_type"] == "ordering"]
    for cardinality in range(2, 7):
        groups[f"multi_select_gold{cardinality}"] = [r for r in records if r["question_type"] == "multi_select" and metadata[r["candidate_id"]]["gold_cardinality"] == cardinality]
    return {name: metric(rows) for name, rows in groups.items()}


def percent(value):
    return "—" if value is None else f"{100 * value['value']:.2f}%"


def decimal(value):
    return "—" if value is None else f"{value['value']:.4f}"


def inline(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def link(path, label=None):
    return f"[{label or Path(path).name}]({path})"


def inspect_root(inputs, row, previous_by_id):
    folder = Path(row["root_path"]).resolve(strict=True)
    root = row["root_id"]
    require(folder.is_relative_to(BATCH / "units"), "ROOT_OUTSIDE_BATCH")
    names = ("CUMULATIVE_SOURCE.json", "PUBLIC_QUESTIONS.json", "PUBLIC_ASSETS.json",
             "PRIVATE_GOLD.json", "FINAL_CROSSCHECK.json", "COMPOSITION_RECEIPT.json",
             "PUBLIC_SOURCE_JUDGMENTS.json", "FINAL_BLIND_SOURCE_REVIEW.json",
             "DIAGNOSTIC_CONTENT_DISPOSITION.json", "FINAL_PUBLIC_CONSISTENCY_REVIEW.json")
    files = {name: inputs.read(folder / name) for name in names}
    source, public, gold, cross, receipt = (files[n] for n in (
        "CUMULATIVE_SOURCE.json", "PUBLIC_QUESTIONS.json", "PRIVATE_GOLD.json",
        "FINAL_CROSSCHECK.json", "COMPOSITION_RECEIPT.json"))
    require(source["root_id"] == root, "ROOT_SOURCE_MISMATCH")
    require(cross["private_sha256"] == inputs.bind(folder / "PRIVATE_GOLD.json")["sha256"], "CROSSCHECK_GOLD_HASH")
    require(cross["public_sha256"] == inputs.bind(folder / "PUBLIC_QUESTIONS.json")["sha256"], "CROSSCHECK_PUBLIC_HASH")
    require(gold["source_sha256"] == inputs.bind(folder / "CUMULATIVE_SOURCE.json")["sha256"], "GOLD_SOURCE_HASH")
    require(receipt["public_count"] == len(public["questions"]), "COMPOSE_COUNT")
    require(receipt["public_sha256"] == cross["public_sha256"], "COMPOSE_PUBLIC_HASH")
    disposition = files["DIAGNOSTIC_CONTENT_DISPOSITION.json"]
    require(disposition["training_pool_eligible"] is False and disposition["batch_procedural_success"] is False, "ROOT_MUST_REMAIN_PROCEDURE_HOLD")
    questions = {q["candidate_id"]: q for q in public["questions"]}
    gold_by_id = {q["candidate_id"]: q for q in gold["questions"]}
    outcomes = {q["candidate_id"]: q for q in cross["results"]}
    require(len(questions) == len(public["questions"]) and set(questions) == set(gold_by_id) == set(outcomes), "FINAL_QUESTION_COVERAGE")
    require(all(q["status"] in {"CONTENT_PASS", "EXCLUDED"} for q in outcomes.values()), "FINAL_STATUS_INVALID")
    previous = previous_by_id[root]
    old = {key: previous.get(key) for key in ("root_id", "status", "materialized_scenes", "valid_ready_scenes", "historical_ready_reviews_retained", "six_scene_reviews_ready", "qa_procedure_hold", "source_only_complete_qa_hold", "failure_details")}
    if root.endswith("-103866"):
        require(old["status"] == "EXPANSION_STOPPED" and old["qa_procedure_hold"] is True, "DO_NOT_REWRITE_103866_ORIGINAL_STATUS")
    new_turns = [t for t in source["turns"] if (t.get("scene_ref") or "").startswith("R02:")]
    meta = {}
    for cid, question in questions.items():
        meta[cid] = {
            "candidate_id": cid, "root_id": root, "question_type": question["question_type"],
            "role": gold_by_id[cid]["role"], "gold": gold_by_id[cid]["gold"],
            "gold_cardinality": len(gold_by_id[cid]["gold"]) if question["question_type"] == "multi_select" else None,
            "status": outcomes[cid]["status"], "issues": outcomes[cid]["issues"],
            "training_pool_eligible": False,
        }
    accepted = [m for m in meta.values() if m["status"] == "CONTENT_PASS"]
    counts = {
        "planned_question_slots": len(questions) + len(receipt["unfilled_slots"]),
        "public_composed": len(questions), "composition_unfilled": len(receipt["unfilled_slots"]),
        "final_content_excluded": len(questions) - len(accepted), "content_pass": len(accepted),
        "pass_multi_select": sum(m["question_type"] == "multi_select" for m in accepted),
        "pass_ordering": sum(m["question_type"] == "ordering" for m in accepted),
        "pass_multiselect_gold_counts": {str(n): sum(m["question_type"] == "multi_select" and m["gold_cardinality"] == n for m in accepted) for n in range(2, 7)},
    }
    require(counts["planned_question_slots"] == counts["composition_unfilled"] + counts["final_content_excluded"] + counts["content_pass"], "SLOT_ACCOUNTING")
    return {
        "root_id": root, "folder": str(folder), "v118_original_delivery_state": old,
        "procedural_success": False, "training_pool_eligible": False,
        "source_reused_byte_identically_from_v118": inputs.bind(folder / "CUMULATIVE_SOURCE.json")["sha256"] == row["old_root_binding"]["sha256"],
        "new_dialogue_expansion_performed_in_v119": False,
        "dialogue": {"turns": len(source["turns"]), "r02_turns": len(new_turns),
                     "r02_scene_turn_counts": dict(Counter(t["scene_ref"] for t in new_turns)),
                     "characters_with_speaker": sum(len(t["speaker"] + ": " + t["text"]) for t in source["turns"]),
                     "words_whitespace_split": sum(len(t["text"].split()) for t in source["turns"]),
                     "dialogue_token_count": None, "token_count_note": "No tokenizer was called; prompt token counts are not dialogue token counts."},
        "counts": counts, "composition_unfilled_slots": receipt["unfilled_slots"],
        "final_excluded_candidates": [m for m in meta.values() if m["status"] == "EXCLUDED"],
        "content_pass_candidate_ids": [m["candidate_id"] for m in accepted],
        "metadata": meta, "files": files,
    }


def read_solver(inputs, side, pool_ids, metadata, scoring, selection):
    scope = BATCH / "evaluations/diagnostic"
    complete = inputs.read(scope / (side.upper() + "_COMPLETE.json"))
    require(complete["training_pool_eligible"] is False, "SOLVER_COMPLETE_MUST_REMAIN_DIAGNOSTIC")
    run = scope / "run" / side
    manifest = inputs.read(run / "manifest.json")
    results = inputs.read(run / "RESULTS.json")
    finished = inputs.read(run / "COMPLETE.json")
    questions = inputs.read(run / "questions.json")["candidates"]
    gold = inputs.read(run / "gold.json")
    inputs.read(run / "canonical_bindings.json")
    require(manifest["solver_model"] == results["solver_model"] == MODEL_IDS[side], "SOLVER_SIDE_IDENTITY")
    require(not results["call_set_errors"], "NATIVE_CALL_SET_ERROR")
    require({q["candidate_id"] for q in questions} == pool_ids == set(gold), "SOLVER_POOL_COVERAGE")
    requests = {r["request_id"]: r for r in manifest["requests"]}
    records = [inputs.read(path) for path in sorted((run / "responses").glob("*.json"))]
    require(len(records) == len(requests) == manifest["request_count"] == finished["actual_call_count"] == complete["actual_calls"], "SOLVER_CALL_COUNT")
    require(not selection.audit_complete_call_set(manifest, records), "RESPONSE_MANIFEST_MISMATCH")
    qmap = {q["candidate_id"]: q for q in questions}
    native = {q["candidate_id"]: q for q in results["questions"]}
    require(set(native) == pool_ids, "NATIVE_RESULTS_COVERAGE")
    for record in records:
        cid = record["candidate_id"]
        question = dict(qmap[cid], gold_content_ids=gold[cid])
        recomputed = scoring.score_response(question, requests[record["request_id"]], record["raw_output"])
        require(recomputed == record["score"], "NATIVE_RESPONSE_SCORE_MISMATCH:" + record["request_id"])
        require(record["evaluation_valid"] is True and record["score"]["scored"] is True, "INVALID_SOLVER_RESPONSE")
    by_candidate = {}
    for cid in sorted(pool_ids):
        rows = [r for r in records if r["candidate_id"] == cid]
        qmetric = metric(rows)
        require(Fraction(native[cid]["full_partial_mean"]) == Fraction(qmetric["partial_mean"]["fraction"]), "NATIVE_PARTIAL_MISMATCH:" + cid)
        require(Fraction(native[cid]["full_exact_rate"]) == Fraction(qmetric["exact_rate"]["fraction"]), "NATIVE_EXACT_MISMATCH:" + cid)
        require(native[cid]["selected_by_policy"] == (0 < Fraction(native[cid]["full_partial_mean"]) < 1), "NATIVE_SELECTION_MISMATCH:" + cid)
        require(len(rows) == (18 if metadata[cid]["question_type"] == "multi_select" else 12), "CANDIDATE_CALL_COUNT")
        by_candidate[cid] = {"metrics": qmetric, "native_policy": native[cid]}
    return {"side": side, "solver_model": manifest["solver_model"], "model_path": manifest["model"],
            "policy_version": manifest["policy_version"], "full_slots": manifest["full_slots"],
            "random_display": manifest["random_display"], "decoder_contract": manifest["decoder_contract"],
            "calls": len(records), "grouped_metrics": grouped_metrics(records, metadata),
            "by_candidate": by_candidate, "by_root": {root: grouped_metrics([r for r in records if r["root_id"] == root], metadata) for root in sorted({m["root_id"] for m in metadata.values()})},
            "_manifest": manifest, "_records": records}


def render_dialogue(root):
    source = root["files"]["CUMULATIVE_SOURCE.json"]
    lines = [f"# {root['root_id']} 完整累计对话", "",
             "本文件是 v119 内容诊断用的只读可读导出，不表示流程合规或可进入训练池。",
             "本轮复用 v118 六场对话，只重试 QA；没有新增扩充对话。", "",
             f"原始来源：{link(Path(root['folder']) / 'CUMULATIVE_SOURCE.json')}", "",
             f"总 {len(source['turns'])} turns；R02 六场 {root['dialogue']['r02_turns']} turns。", ""]
    group = None
    for turn in source["turns"]:
        current = turn.get("scene_ref") or turn.get("session_ref") or "既有对话"
        if current != group:
            lines.extend([f"## {current}", ""])
            group = current
        lines.extend([f"**{turn['turn_ref']} · {turn['speaker']}**", "", turn["text"], ""])
    return "\n".join(lines)


def render_questions(root, solvers):
    files = root["files"]
    public = files["PUBLIC_QUESTIONS.json"]["questions"]
    gold = {r["candidate_id"]: r for r in files["PRIVATE_GOLD.json"]["questions"]}
    review = files["PUBLIC_SOURCE_JUDGMENTS.json"]["questions"]
    lines = [f"# {root['root_id']} 题目、答案与失败记录", "",
             "内容诊断，不可进入训练池。control6 是全选对照，单独展示，不作训练推荐。", "",
             "EXCLUDED 的 Gold 仅为原始设计答案，未获内容审核认可；不参与 Solver 评估。", ""]
    for index, question in enumerate(public, 1):
        cid = question["candidate_id"]
        meta = root["metadata"][cid]
        lines.extend([f"## {index}. {cid}", "", f"类型：{question['question_type']}；角色：{meta['role']}；状态：{meta['status']}", "", question["question"], ""])
        for label, text in question.get("options", question.get("items", {})).items():
            lines.append(f"- **{label}**：{text}")
        lines.extend(["", ("审核通过答案：" if meta["status"] == "CONTENT_PASS" else "原始设计 Gold（排除，不是已认可答案）：") + ", ".join(gold[cid]["gold"]), ""])
        if meta["issues"]:
            lines.extend(["排除原因：" + "；".join(meta["issues"]), ""])
        judgment = review[cid]
        if judgment.get("reason"):
            lines.extend(["独立复读说明：" + judgment["reason"], ""])
        for label, judgment_item in judgment.get("options", judgment.get("items", {})).items():
            relation = judgment_item.get("relation", "supports" if judgment_item.get("supported") else "not_supported")
            lines.extend([f"### {label} · {relation}", "", judgment_item.get("reason", ""), ""])
            citations = judgment_item.get("citations", [])
            if judgment_item.get("order_anchor"):
                citations = [judgment_item["order_anchor"], *judgment_item.get("alternative_matching_anchors", [])]
            for cite in citations:
                lines.extend([f"- `{cite.get('turn_ref', '')}` · {cite.get('speaker', '')}：{cite.get('quote', '')}"])
            lines.append("")
        if meta["status"] == "CONTENT_PASS":
            lines.extend(["| Solver | Exact | Partial | native policy 入选 |", "|---|---:|---:|---|"])
            for side, solver in solvers.items():
                value = solver["by_candidate"][cid]
                lines.append(f"| {solver['solver_model']} | {percent(value['metrics']['exact_rate'])} | {decimal(value['metrics']['partial_mean'])} | {value['native_policy']['selected_by_policy']} |")
            lines.append("")
    if root["composition_unfilled_slots"]:
        lines.extend(["## compose 阶段未填槽位", "", "这些槽位没有进入最终公开题列表，没有最终 Gold，也未运行 Solver。", ""])
        for failure in root["composition_unfilled_slots"]:
            cid = failure.get("candidate_id")
            lines.extend([f"### {cid or failure.get('slot_id', 'unknown')}", "", "```json", json.dumps(failure, ensure_ascii=False, indent=2), "```", ""])
            candidate = next((q for q in files["PUBLIC_ASSETS.json"].get("ordering_questions", []) if q["candidate_id"] == cid), None)
            if candidate:
                lines.extend([candidate["question"], ""])
                lines.extend(f"- **{label}**：{text}" for label, text in candidate["items"].items())
                lines.append("")
    return "\n".join(lines)


def render_report(report):
    totals = report["totals"]
    lines = ["# v119 QA retry 配对内容诊断", "",
             f"**流程成功 {report['procedural_success_roots']}/{report['attempted_roots']}；training_pool_eligible=false。**", "",
             "本轮是三条已有六场 source 的 QA retry；内容通过、native policy 入选均不修复原审核身份偏差，也不构成训练入池。v118 原始状态和失败记录保持原样。", "",
             f"计划 {totals['planned_question_slots']} 槽；compose 未填 {totals['composition_unfilled']}；final 排除 {totals['final_content_excluded']}；最终内容通过 {totals['content_pass']}（多选 {totals['pass_multi_select']}，排序 {totals['pass_ordering']}）。", "",
             "| Root | v118 原始状态 | compose 未填 | final 排除 | 内容通过 | 多选 | 排序 |", "|---|---|---:|---:|---:|---:|---:|"]
    for root in report["roots"]:
        c = root["counts"]
        lines.append(f"| {root['root_id']} | {root['v118_original_delivery_state']['status']} | {c['composition_unfilled']} | {c['final_content_excluded']} | {c['content_pass']} | {c['pass_multi_select']} | {c['pass_ordering']} |")
    lines.extend(["", "103866 的原始状态是 EXPANSION_STOPPED，原因是 QA 程序 HOLD；六场 source 实际已 materialized 且历史 source 审核保留。不能把它改写为普通扩充失败或统一 QA_FAILED。", "",
                  "## 原始未填槽位与 final 排除", ""])
    for root in report["roots"]:
        for failure in root["composition_unfilled_slots"]:
            lines.append(f"- {root['root_id']} / compose / {failure.get('candidate_id', failure.get('slot_id'))}：{inline(failure.get('reason'))}；{inline(failure.get('issues', failure.get('failure', [])))}")
        for failure in root["final_excluded_candidates"]:
            lines.append(f"- {root['root_id']} / final / {failure['candidate_id']}：{inline(failure['issues'])}")
    lines.extend(["", "## 配对 Solver 指标", "",
                  "每题使用同一 native v15 调用集合：多选 6 个置换 × 3 rollouts，排序 12 个置换 × 1 rollout。下表按调用数加权；JSON 同时提供按题平均、每 root、每题指标。", "",
                  "多选 Partial=TP/Gold−FP/非Gold（全选对照为 TP/Gold）；排序 Partial=(正确对数−逆序对数)/总对数。排序逆序对不能解释成多选过选。", "",
                  "| 分组 | 题数 | Base Exact | Trained Exact | Base Partial | Trained Partial |", "|---|---:|---:|---:|---:|---:|"])
    for group in ("multi_select_gold2_to_5", "multi_select_gold2", "multi_select_gold3", "multi_select_gold4", "multi_select_gold5", "multi_select_control6", "ordering"):
        base = report["solvers"]["base"]["grouped_metrics"][group]
        trained = report["solvers"]["trained"]["grouped_metrics"][group]
        lines.append(f"| {group} | {base['candidates']} | {percent(base['exact_rate'])} | {percent(trained['exact_rate'])} | {decimal(base['partial_mean'])} | {decimal(trained['partial_mean'])} |")
    lines.extend(["", "## 多选误选、漏选与全选", "",
                  "FP/FN 为选项计数；含误选调用指 FP>0，数量过选指所选数量大于 Gold 数，两者含义不同。", "",
                  "| Solver | 分组 | calls | FP | FN | 全选调用 | 含误选调用 | 数量过选调用 |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for side, solver in report["solvers"].items():
        for group in ("multi_select_gold2_to_5", "multi_select_control6"):
            c = solver["grouped_metrics"][group]["option_confusion_counts"]
            lines.append(f"| {solver['solver_model']} | {group} | {c['calls']} | {c['fp']} | {c['fn']} | {c['all_select_calls']} | {c['false_positive_calls']} | {c['cardinality_overselect_calls']} |")
    lines.extend(["", "## Native trained policy 入选诊断", "",
                  f"按固定 policy 满足 0<Partial<1 的非 control6 题：{report['trained_policy_diagnostic_selection']['non_control_count']}；control6 入选：{report['trained_policy_diagnostic_selection']['control6_count']}。以上只是难度诊断；可入训练池数量仍为 0。", ""])
    for category in ("non_control_candidate_ids", "control6_candidate_ids"):
        lines.extend([f"{category}：", ""])
        lines.extend(f"- `{cid}`" for cid in report["trained_policy_diagnostic_selection"][category])
        if not report["trained_policy_diagnostic_selection"][category]:
            lines.append("- 无")
        lines.append("")
    lines.extend(["## 解释限制", "",
                  "- 原始公开排序标签已经按 I1… I6 的答案顺序显示，直接阅读版本存在展示捷径；本次 native Solver 独立置换显示标签，不能据此认定实际配对结果靠恒输出通过。",
                  "- 原始 asset reviewer 未遵守冻结角色分配，新增独立内容复读不追认原流程成功。",
                  "- 三个 root 是失败样本的定向 retry，题目变体共享六场 source，不是独立随机样本；配对结果不代表一般训练收益。",
                  "- control6 是全选对照，单列诊断，不作训练推荐；没有新训练、没有合并训练池。", "",
                  "## 对话与题目位置", ""])
    for root in report["roots"]:
        short = root["root_id"].rsplit("-", 1)[-1]
        lines.append(f"- {root['root_id']}：{link(f'roots/root{short}/FULL_DIALOGUE.md', '完整对话')}；{link(f'roots/root{short}/QUESTIONS_AND_ANSWERS.md', '题目、答案和失败记录')}")
    lines.extend(["", f"v118 原始 delivery 索引：{link(report['v118_delivery_index_binding']['path'])}", ""])
    return "\n".join(lines)


def assemble(executor):
    inputs = Inputs()
    inputs.bind(__file__)
    policy = inputs.read(BATCH / "DIAGNOSTIC_CLOSEOUT_POLICY.json")
    require(policy["training_pool_eligible"] is False and policy["batch_procedural_success"] is False, "POLICY_MUST_REMAIN_HOLD")
    manifest = inputs.read(BATCH / "MANIFEST.json")
    require(manifest["retry_limit_per_root"] == 1, "RETRY_LIMIT_CHANGED")
    inputs.verify_binding(manifest["source_batch"])
    previous_path = Path(manifest["source_batch"]["path"]).parent / "delivery/all/ALL_ATTEMPTS_INDEX.json"
    previous_by_id = {r["root_id"]: r for r in inputs.read(previous_path)["roots"]}
    for filename in ("MASTER_FREEZE.json", "PREFINAL_EXISTING_ARTIFACT_SNAPSHOT.json"):
        frozen = inputs.read(BATCH / filename)
        rows = frozen.get("files", frozen.get("artifacts", []))
        require(isinstance(rows, list) and rows, "FREEZE_FILE_LIST_REQUIRED:" + filename)
        for row in rows:
            inputs.verify_binding(row)
    roots = [inspect_root(inputs, r, previous_by_id) for r in manifest["roots"]]
    metadata = {cid: m for root in roots for cid, m in root["metadata"].items()}
    pool_ids = {cid for cid, m in metadata.items() if m["status"] == "CONTENT_PASS"}
    scope = BATCH / "evaluations/diagnostic"
    pool = inputs.read(scope / "EVALUATION_POOL.json")
    require({q["candidate_id"] for q in pool["questions"]} == pool_ids, "DIAGNOSTIC_POOL_COVERAGE")
    preflight = inputs.read(scope / "PAIRED_PREFLIGHT.json")
    require(preflight["training_pool_eligible"] is False and preflight["paired_requests_seeds_prompts_and_gold_identical"] is True, "PAIRED_PREFLIGHT_REQUIRED")
    adapter = inputs.read(scope / "ADAPTER_BINDINGS.json")
    for artifact in [*preflight["files"], *adapter["files"]]:
        inputs.verify_binding(artifact)
    scoring = load_pure_module(inputs, "v119_report_native_scoring", NATIVE / "scoring.py")
    selection = load_pure_module(inputs, "v119_report_native_selection", NATIVE / "selection.py")
    solvers = {side: read_solver(inputs, side, pool_ids, metadata, scoring, selection) for side in MODEL_IDS}
    require(solvers["base"]["_manifest"]["requests"] == solvers["trained"]["_manifest"]["requests"], "PAIRED_REQUESTS_DIFFER")
    require(solvers["base"]["decoder_contract"] == solvers["trained"]["decoder_contract"], "PAIRED_DECODERS_DIFFER")
    for filename in ("questions.json", "gold.json", "canonical_bindings.json"):
        require(inputs.read(scope / "run/base" / filename) == inputs.read(scope / "run/trained" / filename),
                "PAIRED_INPUTS_DIFFER:" + filename)
    totals = {key: sum(root["counts"][key] for root in roots) for key in roots[0]["counts"] if key != "pass_multiselect_gold_counts"}
    totals["pass_multiselect_gold_counts"] = {str(n): sum(root["counts"]["pass_multiselect_gold_counts"][str(n)] for root in roots) for n in range(2, 7)}
    selected = [cid for cid, r in solvers["trained"]["by_candidate"].items() if r["native_policy"]["selected_by_policy"] is True]
    non_control = [cid for cid in selected if metadata[cid]["role"] != "control6"]
    control = [cid for cid in selected if metadata[cid]["role"] == "control6"]
    paired_deltas = {}
    for name, bmetric in solvers["base"]["grouped_metrics"].items():
        tmetric = solvers["trained"]["grouped_metrics"][name]
        paired_deltas[name] = {"exact_trained_minus_base": number(Fraction(tmetric["exact_rate"]["fraction"]) - Fraction(bmetric["exact_rate"]["fraction"])) if bmetric["calls"] else None,
                               "partial_trained_minus_base": number(Fraction(tmetric["partial_mean"]["fraction"]) - Fraction(bmetric["partial_mean"]["fraction"])) if bmetric["calls"] else None}
    for root in roots:
        root["paired_solver_by_root"] = {side: solver["by_root"][root["root_id"]] for side, solver in solvers.items()}
        root["trained_policy_non_control_selected_ids"] = [cid for cid in non_control if metadata[cid]["root_id"] == root["root_id"]]
        root["trained_policy_control6_selected_ids"] = [cid for cid in control if metadata[cid]["root_id"] == root["root_id"]]
    ms_metadata = [m for m in metadata.values() if m["status"] == "CONTENT_PASS" and m["question_type"] == "multi_select"]
    all_select_baseline = mean([Fraction(1 if m["gold_cardinality"] == 6 else 0) for m in ms_metadata])
    report = {"schema_version": "retry_paired_content_diagnostic_report_v1", "batch_id": BATCH.name,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "report_executor": executor, "report_tool_author": "/root/final_public_review",
              **LIMITATIONS, "attempted_roots": len(roots), "procedural_success_roots": 0,
              "training_eligible_roots": 0, "training_eligible_questions": 0,
              "content_nonempty_roots": sum(root["counts"]["content_pass"] > 0 for root in roots),
              "totals": totals, "v118_delivery_index_binding": inputs.bind(previous_path),
              "roots": roots, "solvers": solvers, "paired_trained_minus_base": paired_deltas,
              "always_all_select_multiselect_baseline": {"exact_rate": all_select_baseline, "partial_mean": all_select_baseline,
                  "gold2_to_5_exact_and_partial": number(0), "control6_exact_and_partial": number(1)},
              "trained_policy_diagnostic_selection": {"policy": "Native unchanged full-only selection, 0 < mean Partial < 1",
                  "non_control_count": len(non_control), "non_control_candidate_ids": non_control,
                  "non_control_type_counts": dict(Counter(metadata[cid]["question_type"] for cid in non_control)),
                  "non_control_ms_gold_counts": dict(Counter(str(metadata[cid]["gold_cardinality"]) for cid in non_control if metadata[cid]["question_type"] == "multi_select")),
                  "control6_count": len(control), "control6_candidate_ids": control,
                  "training_recommendation": False, "training_pool_eligible": False},
              "definitions": {"exact_rate": "Correct calls / complete valid calls. Main metrics are call-weighted; candidate-macro is also reported.",
                  "false_positive_calls": "Multi-select calls with FP > 0, even if predicted and Gold cardinalities match.",
                  "cardinality_overselect_calls": "Multi-select calls where prediction cardinality exceeds Gold cardinality.",
                  "ordering_fp": "Inverted item pairs, not false-selected options.",
                  "control6": "All-six-true diagnostic control, never a training recommendation."},
              "limitations": ["Original frozen reviewer assignment deviation remains; content review is not retroactive procedural compliance.",
                  "Raw ordering presentation is sorted; native evaluation uses independent display permutations.",
                  "Directed retry of three failed roots and shared-source variants does not estimate general training benefit.",
                  "No new dialogue expansion, no new QA retry, no training and no pool merge were performed by this reporter."],
              "read_only_inputs": True, "original_files_overwritten": False}
    exports = {}
    for root in roots:
        short = root["root_id"].rsplit("-", 1)[-1]
        exports[f"roots/root{short}/FULL_DIALOGUE.md"] = render_dialogue(root)
        exports[f"roots/root{short}/QUESTIONS_AND_ANSWERS.md"] = render_questions(root, solvers)
    for root in roots:
        del root["files"]
    for solver in solvers.values():
        del solver["_manifest"]
        del solver["_records"]
    inputs.verify()
    report["input_bindings"] = inputs.bindings()
    exports["RETRY_DIAGNOSTIC_REPORT.md"] = render_report(report)
    exports["RETRY_DIAGNOSTIC_REPORT.json"] = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    return inputs, report, exports


def main():
    import re
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BATCH / "delivery/diagnostic_v2")
    parser.add_argument("--executor", required=True, help="Actual canonical executing agent, declared explicitly.")
    parser.add_argument("--check-inputs", action="store_true", help="Validate and aggregate in memory; write no output.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output.is_relative_to(BATCH), "OUTPUT_MUST_BE_INSIDE_V119")
    if not args.check_inputs:
        require(not output.exists(), "OUTPUT_EXISTS_NO_OVERWRITE:" + str(output))
    require(re.fullmatch(r"/root(?:/[a-z0-9_]+)*", args.executor) is not None, "CANONICAL_EXECUTOR_REQUIRED")
    inputs, report, exports = assemble(args.executor)
    if not args.check_inputs:
        inputs.verify()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
        for relative, content in exports.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
        inputs.verify()
    print(json.dumps({"status": "INPUT_CHECK_PASS" if args.check_inputs else "DIAGNOSTIC_REPORT_WRITTEN",
                      **LIMITATIONS, "procedural_success_roots": 0,
                      "attempted_roots": report["attempted_roots"], "totals": report["totals"],
                      "calls_per_model": report["solvers"]["base"]["calls"],
                      "output": None if args.check_inputs else str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError) as error:
        print(json.dumps({"status": "NOT_READY_OR_INVALID_INPUTS", "error": str(error), "outputs_overwritten": False}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
