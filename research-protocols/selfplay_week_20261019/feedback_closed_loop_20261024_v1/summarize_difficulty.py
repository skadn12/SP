#!/usr/bin/env python3
"""Descriptive F/G difficulty tables, after the official frozen-raw analysis.

This standard-library-only script never imports the model runner, calls a model,
reads individual raw responses, changes questions, or writes official scores.
Run only once SCORES.json and ANALYSIS_SUMMARY.json have been produced by the
official analyze step. Outputs are exclusive-create and cannot be overwritten.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path


P = Path(__file__).resolve().parent
OUT = P / "execution"
ARMS = ("F", "G")
ROOTS = tuple("cc-train-episode-" + x for x in ("218519", "222937", "225575", "229686"))
SLOTS = ("MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER2", "ORDER3")
GROUPS = ("NON_MC6_PRIMARY", "MC2_5_PRIMARY", "ORDERING", "MC6_CONTROL", "ALL")
EXPECTED_SEMANTIC_PER_ARM = {"ALL": 32, "NON_MC6_PRIMARY": 28,
    "MC2_5_PRIMARY": 16, "ORDERING": 12, "MC6_CONTROL": 4}
GROUP_ZH = {"ALL": "全部（含全选控制）", "NON_MC6_PRIMARY": "主比较：非MC6",
    "MC2_5_PRIMARY": "多选Gold2–5", "ORDERING": "排序", "MC6_CONTROL": "MC6全选控制"}


def need(value, message):
    if not value:
        raise ValueError(message)


def read_bound(path):
    data = path.read_bytes()
    return json.loads(data), {"path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def check_binding(binding, current):
    need(Path(binding["path"]).resolve() == Path(current["path"]).resolve(), "BOUND_PATH_DRIFT")
    need(binding["sha256"] == current["sha256"] and binding["bytes"] == current["bytes"], "BOUND_INPUT_DRIFT")


def in_group(row, group):
    s = row["strict"]
    return (group == "ALL" or group == "NON_MC6_PRIMARY" and s["gold_count"] != 6
        or group == "MC2_5_PRIMARY" and s["gold_count"] in (2, 3, 4, 5)
        or group == "MC6_CONTROL" and s["gold_count"] == 6
        or group == "ORDERING" and s["question_type"] == "ordering")


def ratio(a, b):
    return a / b if b else None


def mean(values):
    return sum(values) / len(values) if values else None


def close(actual, expected, code):
    need((actual is None and expected is None) or
         actual is not None and expected is not None and math.isclose(actual, expected, abs_tol=1e-12), code)


def normalized_prediction(s):
    # Multi-select answers are sets; ordering answers preserve order.
    values = s["prediction"]
    return tuple(sorted(values)) if s["question_type"] == "multi_select" else tuple(values)


def semantic_rows(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["strict"]["candidate_id"]].append(row)
    result = []
    for cid, rs in sorted(grouped.items()):
        rs.sort(key=lambda r: r["strict"]["variant_id"])
        need(len(rs) == 2, "TWO_DISPLAYS_REQUIRED:" + cid)
        first = rs[0]
        exact_count = sum(r["strict"]["exact"] for r in rs)
        valid_count = sum(r["strict"]["valid"] for r in rs)
        result.append({"candidate_id": cid, "arm": first["arm"],
            "root_id": first["strict"]["root_id"], "slot_id": first["slot_id"],
            "question_type": first["strict"]["question_type"],
            "task_operation": first["strict"]["task_operation"],
            "gold_count": first["strict"]["gold_count"],
            "variant_ids": [r["strict"]["variant_id"] for r in rs],
            "exact_count": exact_count, "exact_mean": exact_count / 2,
            "valid_count": valid_count, "both_correct": exact_count == 2,
            "at_least_one_correct": exact_count >= 1, "both_valid": valid_count == 2,
            "same_canonical_prediction": valid_count == 2 and len({normalized_prediction(r["strict"]) for r in rs}) == 1})
    return result


def metrics(rows):
    ss = [r["strict"] for r in rows]
    valid = [r for r in rows if r["strict"]["valid"]]
    mc = [r for r in rows if r["strict"]["question_type"] == "multi_select"]
    vmc = [r for r in mc if r["strict"]["valid"]]
    order = [r for r in rows if r["strict"]["question_type"] == "ordering"]
    vorder = [r for r in order if r["strict"]["valid"]]
    qs = semantic_rows(rows)
    n = len(rows)
    both_valid = [q for q in qs if q["both_valid"]]
    pairs = sum(r["strict"]["pairwise_total"] for r in order)
    v_pairs = sum(r["strict"]["pairwise_total"] for r in vorder)
    return {"requests": n, "semantic_questions": len(qs),
        "correct": sum(s["exact"] for s in ss), "exact_accuracy": ratio(sum(s["exact"] for s in ss), n),
        "valid_requests": len(valid), "invalid_requests": n - len(valid),
        "valid_but_wrong_requests": sum(s["valid"] and not s["exact"] for s in ss),
        "valid_conditioned_exact_accuracy": ratio(sum(r["strict"]["exact"] for r in valid), len(valid)),
        "mc_requests": len(mc), "mc_correct": sum(r["strict"]["exact"] for r in mc),
        "mc_macro": {k: mean([r["partial"][k] for r in mc]) for k in ("precision", "recall", "f1")},
        "mc_valid_conditioned_macro": {k: mean([r["partial"][k] for r in vmc]) for k in ("precision", "recall", "f1")},
        "mc_false_positive": sum(r["strict"]["false_positive"] for r in mc),
        "mc_false_negative": sum(r["strict"]["false_negative"] for r in mc),
        "mc_valid_conditioned_false_positive": sum(r["strict"]["false_positive"] for r in vmc),
        "mc_valid_conditioned_false_negative": sum(r["strict"]["false_negative"] for r in vmc),
        "ordering_requests": len(order), "ordering_correct": sum(r["strict"]["exact"] for r in order),
        "ordering_pairwise_correct": sum(r["strict"]["pairwise_correct"] for r in order),
        "ordering_pairwise_total": pairs,
        "ordering_pairwise_accuracy": ratio(sum(r["strict"]["pairwise_correct"] for r in order), pairs),
        "ordering_valid_conditioned_pairwise_accuracy": ratio(sum(r["strict"]["pairwise_correct"] for r in vorder), v_pairs),
        "semantic_both_correct": sum(q["both_correct"] for q in qs),
        "semantic_at_least_one_correct": sum(q["at_least_one_correct"] for q in qs),
        "semantic_both_valid": len(both_valid),
        "semantic_same_canonical_prediction": sum(q["same_canonical_prediction"] for q in qs),
        "semantic_both_correct_rate": ratio(sum(q["both_correct"] for q in qs), len(qs)),
        "semantic_at_least_one_correct_rate": ratio(sum(q["at_least_one_correct"] for q in qs), len(qs)),
        "semantic_prediction_consistency_rate": ratio(sum(q["same_canonical_prediction"] for q in qs), len(qs)),
        "semantic_valid_conditioned_consistency_rate": ratio(sum(q["same_canonical_prediction"] for q in both_valid), len(both_valid))}


def difference(f, g):
    rate_names = ("exact_accuracy", "valid_conditioned_exact_accuracy", "ordering_pairwise_accuracy",
        "ordering_valid_conditioned_pairwise_accuracy", "semantic_both_correct_rate",
        "semantic_at_least_one_correct_rate", "semantic_prediction_consistency_rate",
        "semantic_valid_conditioned_consistency_rate")
    result = {k + "_pp": None if f[k] is None or g[k] is None else 100 * (f[k] - g[k]) for k in rate_names}
    for family in ("mc_macro", "mc_valid_conditioned_macro"):
        result[family + "_pp"] = {k: None if f[family][k] is None or g[family][k] is None
            else 100 * (f[family][k] - g[family][k]) for k in ("precision", "recall", "f1")}
    for k in ("correct", "invalid_requests", "mc_false_positive", "mc_false_negative",
              "mc_valid_conditioned_false_positive", "mc_valid_conditioned_false_negative"):
        result[k + "_count"] = f[k] - g[k]
    return result


def comparison(rows):
    arms = {a: metrics([r for r in rows if r["arm"] == a]) for a in ARMS}
    return {**arms, "F_minus_G": difference(arms["F"], arms["G"])}


def paired_slots(rows):
    qs = semantic_rows(rows)
    slots = collections.defaultdict(dict)
    for q in qs:
        key = (q["root_id"], q["slot_id"])
        need(q["arm"] not in slots[key], "DUPLICATE_PAIRED_ARM")
        slots[key][q["arm"]] = q
    result = []
    for (root, slot), arms in sorted(slots.items()):
        need(set(arms) == set(ARMS), "MISSING_MATCHED_ARM")
        f, g = arms["F"], arms["G"]
        delta = f["exact_mean"] - g["exact_mean"]
        result.append({"root_id": root, "slot_id": slot, "F_candidate_id": f["candidate_id"],
            "G_candidate_id": g["candidate_id"], "F_exact_count_of_2": f["exact_count"],
            "G_exact_count_of_2": g["exact_count"], "F_exact_mean": f["exact_mean"],
            "G_exact_mean": g["exact_mean"], "F_minus_G_exact_pp": 100 * delta,
            "direction": "F_MORE_DIFFICULT" if delta < 0 else "F_LESS_DIFFICULT" if delta > 0 else "TIE"})
    return result


def pair_direction_counts(rows):
    count = collections.Counter(r["direction"] for r in rows)
    return {"matched_semantic_slots": len(rows), "F_more_difficult": count["F_MORE_DIFFICULT"],
        "tie": count["TIE"], "F_less_difficult": count["F_LESS_DIFFICULT"]}


def validate_records(records, summary):
    need(len(records) == 128 and len({r["request_id"] for r in records}) == 128, "128_UNIQUE_REQUESTS_REQUIRED")
    need({r["arm"] for r in records} == set(ARMS), "ARM_SET_DRIFT")
    need({r["strict"]["root_id"] for r in records} == set(ROOTS), "ROOT_SET_DRIFT")
    by_candidate = collections.defaultdict(list)
    for r in records:
        s = r["strict"]
        need(r["request_id"] == s["request_id"] and r["slot_id"] in SLOTS, "REQUEST_OR_SLOT_DRIFT")
        need(type(s["valid"]) is bool and type(s["exact"]) is bool and (not s["exact"] or s["valid"]), "INVALID_EXACT_FLAGS")
        need(s["variant_id"] in ("perm_00", "perm_01"), "VARIANT_DRIFT")
        if s["question_type"] == "multi_select":
            need(r["slot_id"] == "MC" + str(s["gold_count"]) and s["gold_count"] in range(2, 7), "MC_QUOTA_DRIFT")
            gold, pred = set(s["gold"]), set(s["prediction"])
            need(len(gold) == s["gold_count"] and len(s["prediction"]) == len(pred), "INVALID_MC_LABELS")
            fp, fn, tp = len(pred - gold), len(gold - pred), len(gold & pred)
            need(fp == s["false_positive"] and fn == s["false_negative"], "FP_FN_DRIFT")
            close(r["partial"]["precision"], tp / (tp + fp) if tp + fp else 0, "PRECISION_DRIFT")
            close(r["partial"]["recall"], tp / (tp + fn), "RECALL_DRIFT")
            close(r["partial"]["f1"], 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, "F1_DRIFT")
            need(s["exact"] == bool(s["valid"] and pred == gold), "MC_EXACT_DRIFT")
        else:
            need(s["question_type"] == "ordering" and r["slot_id"].startswith("ORDER") and s["gold_count"] is None, "ORDER_TYPE_DRIFT")
            need(len(s["gold"]) == len(set(s["gold"])) == 6 and s["pairwise_total"] == 15, "ORDER_DENOMINATOR_DRIFT")
            if s["valid"]:
                need(len(s["prediction"]) == 6 and set(s["prediction"]) == set(s["gold"]), "ORDER_PREDICTION_DRIFT")
                positions = {x: i for i, x in enumerate(s["prediction"])}
                correct = sum(positions[left] < positions[right] for i, left in enumerate(s["gold"]) for right in s["gold"][i+1:])
            else:
                need(s["prediction"] == [], "INVALID_PREDICTION_NOT_EMPTY")
                correct = 0
            need(s["pairwise_correct"] == correct and s["exact"] == bool(s["valid"] and s["prediction"] == s["gold"]), "ORDER_SCORE_DRIFT")
            close(r["partial"]["pairwise_accuracy"], correct / 15, "PAIRWISE_DRIFT")
        by_candidate[s["candidate_id"]].append(r)
    need(len(by_candidate) == 64, "64_SEMANTIC_QUESTIONS_REQUIRED")
    for cid, rs in by_candidate.items():
        need(len(rs) == 2 and {r["strict"]["variant_id"] for r in rs} == {"perm_00", "perm_01"}, "MISSING_DISPLAY:" + cid)
        for field in ("root_id", "question_type", "task_operation", "gold_count", "gold"):
            need(rs[0]["strict"][field] == rs[1]["strict"][field], "SEMANTIC_METADATA_DRIFT:" + cid)
        need(rs[0]["arm"] == rs[1]["arm"] and rs[0]["slot_id"] == rs[1]["slot_id"], "SEMANTIC_ARM_SLOT_DRIFT")
    summaries = {(r["arm"], r["group"]): r for r in summary["summaries"]}
    for group in GROUPS:
        for arm in ARMS:
            chosen = [r for r in records if r["arm"] == arm and in_group(r, group)]
            own = metrics(chosen)
            need(own["semantic_questions"] == EXPECTED_SEMANTIC_PER_ARM[group], "GROUP_SEMANTIC_DENOMINATOR_DRIFT")
            official = summaries[(arm, group)]
            for own_key, official_key in (("requests", "frozen_requests"), ("correct", "correct"),
                ("valid_requests", "valid_requests"), ("invalid_requests", "invalid_requests"),
                ("mc_false_positive", "mc_false_positive"), ("mc_false_negative", "mc_false_negative"),
                ("ordering_pairwise_correct", "ordering_pairwise_correct"), ("ordering_pairwise_total", "ordering_pairwise_total")):
                need(own[own_key] == official[official_key], "OFFICIAL_AGGREGATE_DRIFT:" + own_key)
            close(own["exact_accuracy"], official["exact_accuracy"], "OFFICIAL_EXACT_DRIFT")
            close(own["ordering_pairwise_accuracy"], official["ordering_pairwise_accuracy"], "OFFICIAL_PAIRWISE_DRIFT")
            for metric in ("precision", "recall", "f1"):
                close(own["mc_macro"][metric], official["all_admitted_requests_mc_macro"][metric], "OFFICIAL_MC_MACRO_DRIFT")
    need(summary["denominators"]["planned_semantic_questions"] == 64 and summary["denominators"]["frozen_requests"] == 128, "OFFICIAL_PLAN_DENOMINATOR_DRIFT")
    return {"status": "PASS", "unique_requests": 128, "semantic_questions": 64,
        "all_expected_root_slot_arm_cells": True, "official_group_aggregates_reproduced": True}


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def pp(value):
    return "—" if value is None else f"{value:+.2f}"


def fraction(correct, total, value):
    return f"{correct}/{total}（{pct(value)}）"


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |" for row in rows], ""]


def render(doc):
    lines = ["# 同源 F/G 题目难度描述性统计", "",
        "F 接收候选弱点卡与出题反馈，G 接收相同生产质量、操作与答案配额。两臂共用 R03 扩充来源；每道语义题有两次固定随机展示。", "",
        "所有差值均为 F − G；exact 或 pairwise 差值为负表示本次固定 Solver 在 F 上得分更低。这里只描述这批题，不能据此直接判定原弱点命中。", "",
        "## 完整性与分母", "",
        "64 道内容有效语义题、128 次展示作答；F/G 各 32 道题、64 次作答。主比较排除 MC6：每臂 28 道语义题、56 次作答。", "",
        "MC2–5 每臂 16 道题 × 2 展示；排序每臂 12 道题 × 2 展示；MC6 每臂 4 道控制题 × 2 展示。内容未通过不能当作 Solver 失败，本次统计要求完整的 64 道已准入题。", "",
        "## Strict exact", ""]
    rows = []
    for entry in doc["group_comparisons"]:
        f, g = entry["F"], entry["G"]
        rows.append((GROUP_ZH[entry["group"]], fraction(f["correct"], f["requests"], f["exact_accuracy"]),
            fraction(g["correct"], g["requests"], g["exact_accuracy"]), pp(entry["F_minus_G"]["exact_accuracy_pp"]),
            f'{f["invalid_requests"]} / {g["invalid_requests"]}',
            f'{pct(f["valid_conditioned_exact_accuracy"])} / {pct(g["valid_conditioned_exact_accuracy"])}'))
    lines += markdown_table(("分组", "F 完全正确", "G 完全正确", "差值 pp", "无效 F/G", "有效回答条件 exact F/G"), rows)
    lines += ["## 多选部分正确与错误计数", ""]
    rows = []
    for entry in doc["group_comparisons"]:
        if entry["group"] not in ("MC2_5_PRIMARY", "MC6_CONTROL"):
            continue
        for arm in ARMS:
            m = entry[arm]
            rows.append((GROUP_ZH[entry["group"]], arm, m["mc_requests"],
                pct(m["mc_macro"]["precision"]), pct(m["mc_macro"]["recall"]), pct(m["mc_macro"]["f1"]),
                m["mc_false_positive"], m["mc_false_negative"],
                f'{pct(m["mc_valid_conditioned_macro"]["precision"])} / {pct(m["mc_valid_conditioned_macro"]["recall"])} / {pct(m["mc_valid_conditioned_macro"]["f1"])}'))
    lines += markdown_table(("分组", "臂", "作答数", "宏P", "宏R", "宏F1", "过选FP", "漏选FN", "有效回答条件 P/R/F1"), rows)
    for entry in doc["group_comparisons"]:
        if entry["group"] in ("MC2_5_PRIMARY", "MC6_CONTROL"):
            delta = entry["F_minus_G"]
            lines.append(f'{GROUP_ZH[entry["group"]]} F − G：宏P {pp(delta["mc_macro_pp"]["precision"])} pp，宏R {pp(delta["mc_macro_pp"]["recall"])} pp，宏F1 {pp(delta["mc_macro_pp"]["f1"])} pp；FP {delta["mc_false_positive_count"]:+d}，FN {delta["mc_false_negative_count"]:+d}。')
    lines += ["", "无效多选回答按空预测集合计描述性 FP/FN，严格 exact 记错；因此漏选计数可能含格式/执行失败，须与有效回答条件统计一起看。MC6 全真题没有错误选项，不能单独证明干扰项难度。", "", "## 排序 pairwise", ""]
    order = next(x for x in doc["group_comparisons"] if x["group"] == "ORDERING")
    lines += markdown_table(("臂", "完全正确", "正确有序对", "有效回答条件 pairwise"), [
        (arm, fraction(order[arm]["correct"], order[arm]["requests"], order[arm]["exact_accuracy"]),
            fraction(order[arm]["ordering_pairwise_correct"], order[arm]["ordering_pairwise_total"], order[arm]["ordering_pairwise_accuracy"]),
            pct(order[arm]["ordering_valid_conditioned_pairwise_accuracy"])) for arm in ARMS])
    lines += [f'排序 pairwise F − G：{pp(order["F_minus_G"]["ordering_pairwise_accuracy_pp"])} pp。无效排序回答的 15 个有序对均记错。', "", "## 两展示语义题稳定性", ""]
    rows = []
    for entry in doc["group_comparisons"]:
        for arm in ARMS:
            m = entry[arm]
            rows.append((GROUP_ZH[entry["group"]], arm, m["semantic_questions"],
                m["semantic_both_correct"], m["semantic_at_least_one_correct"], m["semantic_both_valid"],
                fraction(m["semantic_same_canonical_prediction"], m["semantic_questions"], m["semantic_prediction_consistency_rate"]),
                pct(m["semantic_valid_conditioned_consistency_rate"])))
    lines += markdown_table(("分组", "臂", "语义题", "两展示均对", "至少一展示对", "两展示均有效", "同canonical预测", "均有效条件一致率"), rows)
    lines += ["多选的一致性按 canonical 标签集合，排序按 canonical 标签序列；存在无效展示时，该题不算一致。两次展示是同题重复测量，不能视为两道独立语义题。", "", "## 各 root 主比较", ""]
    rows = []
    for entry in doc["root_primary_comparisons"]:
        f, g = entry["F"], entry["G"]
        rows.append((entry["root_id"], fraction(f["correct"], f["requests"], f["exact_accuracy"]),
            fraction(g["correct"], g["requests"], g["exact_accuracy"]), pp(entry["F_minus_G"]["exact_accuracy_pp"]),
            entry["direction"], f'{f["invalid_requests"]} / {g["invalid_requests"]}'))
    lines += markdown_table(("root", "F exact", "G exact", "差值 pp", "得分方向", "无效 F/G"), rows)
    root_counts = doc["root_direction_counts"]
    lines += [f'Root 方向：F 得分更低 {root_counts["F_more_difficult"]}，相同 {root_counts["tie"]}，F 得分更高 {root_counts["F_less_difficult"]}。仅 4 个发现集 Train roots，不作符号显著性或总体泛化推断。', "", "## 匹配语义槽方向", ""]
    rows = []
    for entry in doc["matched_slot_direction_summaries"]:
        rows.append((GROUP_ZH[entry["group"]], entry["matched_semantic_slots"], entry["F_more_difficult"], entry["tie"], entry["F_less_difficult"]))
    lines += markdown_table(("分组", "匹配槽", "F得分更低", "得分相同", "F得分更高"), rows)
    lines += ["匹配键是同 root/slot，方向依据每臂该语义题两展示 exact 的平均值。它们共享来源和操作配额，但 F/G 不是同一道题，也不保证实际展示排列相同。", "", "## 各 slot", ""]
    rows = []
    for entry in doc["slot_comparisons"]:
        f, g = entry["F"], entry["G"]
        rows.append((entry["slot_id"], fraction(f["correct"], f["requests"], f["exact_accuracy"]),
            fraction(g["correct"], g["requests"], g["exact_accuracy"]), pp(entry["F_minus_G"]["exact_accuracy_pp"]),
            f'{pct(f["ordering_pairwise_accuracy"])} / {pct(g["ordering_pairwise_accuracy"])}'))
    lines += markdown_table(("slot", "F exact", "G exact", "差值 pp", "排序 pairwise F/G"), rows)
    lines += ["## 多选 Gold 配额", ""]
    rows = []
    for entry in doc["gold_quota_comparisons"]:
        f, g = entry["F"], entry["G"]
        rows.append((entry["gold_count"], fraction(f["correct"], f["requests"], f["exact_accuracy"]),
            fraction(g["correct"], g["requests"], g["exact_accuracy"]), pp(entry["F_minus_G"]["exact_accuracy_pp"]),
            f'{pct(f["mc_macro"]["f1"])} / {pct(g["mc_macro"]["f1"])}',
            f'{f["mc_false_positive"]}/{f["mc_false_negative"]}', f'{g["mc_false_positive"]}/{g["mc_false_negative"]}'))
    lines += markdown_table(("Gold/6", "F exact", "G exact", "差值 pp", "宏F1 F/G", "F FP/FN", "G FP/FN"), rows)
    lines += ["## 解释边界", "", *["- " + x for x in doc["interpretation_limits"]], "",
        "## 输入与执行", "", f'原始生成时长：{doc["execution"]["raw_wall_seconds"]:.2f} 秒；输入 tokens：{doc["execution"]["total_prompt_tokens"]}；输出 tokens：{doc["execution"]["total_output_tokens"]}。', "",
        "本脚本只读取官方评分与汇总、raw 封存清单及本轮配置；不读取单条原始输出、不调用模型、不执行训练、不修改题目或官方结果。", "",
        "机器可读结果与完整输入哈希：`DIFFICULTY_COMPARISON.json`。", ""]
    return "\n".join(lines)


def main():
    json_path, md_path = P / "DIFFICULTY_COMPARISON.json", P / "DIFFICULTY_TABLES_ZH.md"
    need(not json_path.exists() and not md_path.exists(), "NO_OUTPUT_OVERWRITE")
    for name in ("SCORES.json", "ANALYSIS_SUMMARY.json", "RAW_RESPONSE_FREEZE.json"):
        need((OUT / name).is_file(), "OFFICIAL_ANALYSIS_NOT_AVAILABLE:" + name)
    scores, scores_binding = read_bound(OUT / "SCORES.json")
    summary, summary_binding = read_bound(OUT / "ANALYSIS_SUMMARY.json")
    freeze, freeze_binding = read_bound(OUT / "RAW_RESPONSE_FREEZE.json")
    config, config_binding = read_bound(P / "CONFIG.json")
    check_binding(scores["raw_freeze_binding"], freeze_binding)
    need(config["training"] is False and config["planned_solver_requests"] == 128
        and config["planned_semantic_questions"] == 64 and config["display_variants"] == 2, "CONFIG_DRIFT")
    need(set(config["roots"]) == {r.rsplit("-", 1)[-1] for r in ROOTS}, "CONFIG_ROOT_DRIFT")
    need(freeze["requests"] == 128 and freeze["returned"] == summary["denominators"]["returned_requests"], "RAW_FREEZE_COUNT_DRIFT")
    records = scores["records"]
    validation = validate_records(records, summary)
    # Ensure every planned matched root/slot/arm exists, not only total counts.
    cells = {(r["strict"]["root_id"], r["slot_id"], r["arm"]) for r in records}
    need(cells == {(root, slot, arm) for root in ROOTS for slot in SLOTS for arm in ARMS}, "MATCHED_CELL_SET_DRIFT")
    groups = [{"group": group, **comparison([r for r in records if in_group(r, group)])} for group in GROUPS]
    paired = paired_slots(records)
    primary = [r for r in records if in_group(r, "NON_MC6_PRIMARY")]
    roots = []
    for root in ROOTS:
        entry = {"root_id": root, **comparison([r for r in primary if r["strict"]["root_id"] == root])}
        delta = entry["F_minus_G"]["exact_accuracy_pp"]
        entry["direction"] = "F_MORE_DIFFICULT" if delta < 0 else "F_LESS_DIFFICULT" if delta > 0 else "TIE"
        roots.append(entry)
    slot_dirs = []
    for group in GROUPS:
        allowed = {r["slot_id"] for r in records if in_group(r, group)}
        slot_dirs.append({"group": group, **pair_direction_counts([x for x in paired if x["slot_id"] in allowed])})
    doc = {"schema_version": "closed_loop_descriptive_difficulty_v1", "experiment_id": P.name,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "training": False,
        "analysis_type": "DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_TESTS",
        "difference_convention": "F minus G; lower exact/pairwise is lower score on F for this fixed Solver and sample",
        "input_bindings": [scores_binding, summary_binding, freeze_binding, config_binding],
        "script_binding": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "bytes": Path(__file__).stat().st_size},
        "validation": validation, "official_denominators": summary["denominators"],
        "group_comparisons": groups, "root_primary_comparisons": roots,
        "root_direction_counts": pair_direction_counts(roots),
        "slot_comparisons": [{"slot_id": slot, **comparison([r for r in records if r["slot_id"] == slot])} for slot in SLOTS],
        "gold_quota_comparisons": [{"gold_count": count, **comparison([r for r in records if r["strict"]["gold_count"] == count])} for count in range(2, 7)],
        "matched_semantic_slot_comparisons": paired,
        "matched_slot_direction_summaries": slot_dirs,
        "semantic_question_results": semantic_rows(records),
        "execution": {key: summary[key] for key in ("raw_wall_seconds", "total_prompt_tokens", "total_output_tokens")},
        "metric_policies": {"exact_denominator": "all frozen content-admitted requests; invalid responses count as incorrect",
            "partial_score_invalid": "MC empty prediction (FP=0, FN=Gold count); ordering pairwise=0/15; valid-conditioned metrics also reported",
            "semantic_consistency": "both displays valid and same canonical label set for MC or label sequence for ordering",
            "paired_semantic_slot_direction": "same root/slot F vs G mean exact over two dependent display repeats",
            "MC6": "separate all-true control; excluded from primary comparison"},
        "interpretation_limits": [
            "仅 4 个已用于发现弱点的 Train roots；不能声称未见 root 泛化或总体题目难度。",
            "每题两展示为重复测量，不按 128 个独立样本推断，不做显著性检验。",
            "F/G 共用含反馈扩充的来源；本对照不能单独归因扩充反馈收益，也不能声称完成训练或修复弱点。",
            "F/G 是同源、同 root/slot 的不同题，且实际展示排列未保证相同。",
            "作者收到通用质量提醒的时点不完全相同；不得将初稿有效率差解释为严格反馈因果收益。",
            "内容审核由同 Codex 家族不同上下文完成，属于 AI 内容审核，非外部人工审核。",
            "难度得分下降不自动等于命中原弱点；原卡目标实现及具体错误机制需按预冻结标准另行审核。",
            "旧 30 题 16/30 仅是历史背景，不与本轮结果直接相减推断反馈因果收益。"]}
    text = render(doc)
    # Recheck the two score documents immediately before exclusive creation.
    for path, binding in ((OUT / "SCORES.json", scores_binding), (OUT / "ANALYSIS_SUMMARY.json", summary_binding)):
        _, current = read_bound(path)
        check_binding(binding, current)
    with json_path.open("x", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with md_path.open("x", encoding="utf-8") as f:
        f.write(text)
    print(json.dumps({"status": "COMPLETE", "json": str(json_path), "tables": str(md_path),
        "primary": groups[0], "root_direction_counts": doc["root_direction_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
