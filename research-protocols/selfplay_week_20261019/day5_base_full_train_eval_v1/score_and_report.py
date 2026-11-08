"""Post-freeze original-base scoring of the full, fixed A/B training pools.

This module has no inference or retry entry point. All outputs are exclusive.
Gold is parsed only after every one of the 2,910 base raw records is frozen
and hash verified. Existing experiment artifacts are never modified.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
MODEL = "/root/autodl-tmp/Qwen3-4B"
OLD_BASE = WEEK / "day5_base_panel_eval_v1/direct/responses"
OLD_PANEL = WEEK / "day5_training_v1/train_diagnosis_prepare/public/DIRECT_REQUESTS.json"
OLD_S0 = WEEK / "day3_solver187_v1/runtime/direct_v2"
IDENTITY = ("request_id", "candidate_id", "root_id", "question_type", "variant_id", "slot", "seed", "prompt_sha256")
OUTPUTS = ("SCORING_STARTED.json", "BASE_SCORES.json", "S0_SCORES.json", "COMPARISON.json", "REPORT_ZH.md", "per_question.csv")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify(bindings):
    for item in bindings:
        path = Path(item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "FROZEN_FILE_CHANGED:" + str(path))


def verify_nested(payload):
    if isinstance(payload, dict):
        if {"path", "sha256", "bytes"} <= payload.keys():
            verify([payload])
        else:
            for value in payload.values():
                verify_nested(value)
    elif isinstance(payload, list):
        for value in payload:
            verify_nested(value)


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def mean(values):
    values = list(values)
    require(bool(values), "EMPTY_METRIC_DENOMINATOR")
    return sum(values, Fraction()) / len(values)


def rational(value):
    value = Fraction(value)
    return {"fraction": str(value), "value": float(value)}


def aggregate(questions, records):
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["candidate_id"]].append(record)
    roots = collections.defaultdict(list)
    for question in questions:
        selected = grouped[question["candidate_id"]]
        require(len(selected) == question["denominator_calls"], "QUESTION_REQUEST_DENOMINATOR_DRIFT")
        roots[question["root_id"]].append(Fraction(sum(r["exact"] for r in selected), len(selected)))
    result = {"questions": len(questions), "roots": len(roots), "planned_calls": len(records),
              "valid_calls": sum(r["valid"] for r in records), "invalid_calls": sum(not r["valid"] for r in records),
              "exact_calls": sum(r["exact"] for r in records),
              "question_equal_exact": rational(mean(x for values in roots.values() for x in values)),
              "root_equal_exact": rational(mean(mean(values) for values in roots.values())),
              "call_pooled_exact_secondary": rational(Fraction(sum(r["exact"] for r in records), len(records)))}
    types = {q["question_type"] for q in questions}
    if types == {"multi_select"}:
        scores = []
        for question in questions:
            per_call = []
            for record in grouped[question["candidate_id"]]:
                pred, gold = set(record["prediction_content"] or []), set(record["gold_content"])
                per_call.append(Fraction(len(pred & gold), len(pred | gold)) if record["valid"] else Fraction())
            scores.append(mean(per_call))
        result["question_equal_mc_jaccard_secondary"] = rational(mean(scores))
    elif types == {"ordering"}:
        result["question_equal_order_pair_accuracy_secondary"] = rational(mean(
            mean(Fraction(r["metrics"]["correct_pairs"], r["metrics"]["total_pairs"])
                 for r in grouped[q["candidate_id"]]) for q in questions))
    return result


def build_groups(questions, records, allowed_categories=None):
    groups = {}
    for cohort in ("UNION", "A", "B"):
        for category in ("ALL", "new_structure", "foundation"):
            if allowed_categories and category not in ("ALL", *allowed_categories):
                continue
            for typ in ("ALL", "multi_select", "ordering"):
                subset = [q for q in questions if (cohort == "UNION" or cohort in q["arms"])
                          and (category == "ALL" or q["category"] == category)
                          and (typ == "ALL" or q["question_type"] == typ)]
                if not subset:
                    continue
                ids = {q["candidate_id"] for q in subset}
                groups[cohort + "/" + category + "/" + typ] = aggregate(subset, [r for r in records if r["candidate_id"] in ids])
    return groups


def load_scorer():
    directory = WEEK / "day3_solver187_v1/runtime"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("day5_full_base_native_score", directory / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(public, records, memberships, scorer):
    return [scorer.summarize_question(q, [r for r in records if r["candidate_id"] == q["candidate_id"]])
            | memberships[q["candidate_id"]] for q in public]


def first_value(payload, names):
    for name in names:
        if name in payload:
            return payload[name]
    return None


def run():
    require(all(not (HERE / name).exists() for name in OUTPUTS), "NO_OUTPUT_OVERWRITE")
    prep = read(HERE / "PREPARATION_FREEZE.json")
    manifest = read(HERE / "EXECUTION_MANIFEST.json")
    require(manifest["model_role"] == "BASE" and manifest["model_directory"] == MODEL, "ORIGINAL_BASE_IDENTITY_REQUIRED")
    verify_nested(prep)
    verify_nested(manifest)
    if (HERE / "CPU_VALIDATION.json").exists():
        cpu = read(HERE / "CPU_VALIDATION.json")
        require(str(cpu.get("status", "")).startswith("PASS"), "CPU_VALIDATION_NOT_PASS")
        verify_nested(cpu)
    rows = read(HERE / "public/DIRECT_REQUESTS.json")["requests"]
    public = read(HERE / "public/PUBLIC_QUESTIONS.json")["questions"]
    maps = read(HERE / "public/DISPLAY_MAPPINGS.json")
    membership = read(HERE / "POOL_MEMBERSHIP.json")
    members = {q["candidate_id"]: q for q in membership["questions"]}
    qids, rids = {q["candidate_id"] for q in public}, {r["request_id"] for r in rows}
    require(len(rows) == len(rids) == 2910 and len(public) == len(qids) == len(members) == 200,
            "FIXED_FULL_POOL_DENOMINATOR_DRIFT")
    require(qids == set(members) and set(maps["requests"]) == rids and set(maps["public_to_canonical"]) == qids,
            "PUBLIC_MAP_MEMBERSHIP_IDENTITY_DRIFT")
    require(collections.Counter(q["question_type"] for q in public) == {"multi_select": 85, "ordering": 115}, "UNION_TYPE_COUNTS_DRIFT")
    require(collections.Counter(q["category"] for q in members.values()) == {"new_structure": 176, "foundation": 24}, "CATEGORY_COUNTS_DRIFT")
    for arm in ("A", "B"):
        ids = membership["arms"][arm]
        require(len(ids) == len(set(ids)) == 112 and set(ids) <= qids, "ARM_DENOMINATOR_DRIFT")
        subset = [members[qid] for qid in ids]
        require(set(ids) == {q["candidate_id"] for q in members.values() if arm in q["arms"]}, "ARM_MEMBERSHIP_DRIFT")
        require(collections.Counter(q["question_type"] for q in subset) == {"multi_select": 50, "ordering": 62}, "ARM_TYPE_COUNTS_DRIFT")
        require(collections.Counter(q["category"] for q in subset) == {"new_structure": 88, "foundation": 24}, "ARM_CATEGORY_COUNTS_DRIFT")
    common = set(membership["arms"]["A"]) & set(membership["arms"]["B"])
    require(common == {q["candidate_id"] for q in members.values() if q["category"] == "foundation"} and len(common) == 24,
            "COMMON_FOUNDATION_IDENTITY_DRIFT")
    for question in public:
        qid = question["candidate_id"]
        require(all(members[qid][k] == question[k] for k in ("root_id", "question_type")), "MEMBERSHIP_QUESTION_DRIFT")
        require(sha(question["source_path"]) == question["source_sha256"], "SOURCE_CHANGED")
        subset = [row for row in rows if row["candidate_id"] == qid]
        require(len(subset) == (18 if question["question_type"] == "multi_select" else 12), "PER_QUESTION_DENOMINATOR_DRIFT")
        for row in subset:
            require(all(row[k] == question[k] for k in ("root_id", "question_type")) and sha(row["prompt_path"]) == row["prompt_sha256"],
                    "PROMPT_OR_QUESTION_IDENTITY_DRIFT")
    direct = HERE / "direct"
    freeze = read(direct / "RAW_RESPONSE_FREEZE.json")
    complete = read(direct / "COMPLETE.json")
    require(freeze["status"] == "COMMITTED" and freeze["planned_request_count"] == 2910, "FULL_RAW_FREEZE_REQUIRED")
    require(complete["status"] == "COMPLETE", "COMPLETE_FULL_RUN_REQUIRED")
    verify(freeze["files"] + [freeze["request_freeze"], freeze["execution_manifest"]])
    require(Path(freeze["request_freeze"]["path"]).resolve() == HERE / "PREPARATION_FREEZE.json"
            and Path(freeze["execution_manifest"]["path"]).resolve() == HERE / "EXECUTION_MANIFEST.json", "RAW_FREEZE_WRONG_RUN")
    expected = {direct / "responses" / (r["request_id"] + ".json") for r in rows}
    require(len(freeze["files"]) == 2910 and {Path(b["path"]).resolve() for b in freeze["files"]} == expected
            and set((direct / "responses").glob("*.json")) == expected, "RAW_FREEZE_SET_OR_DENOMINATOR_DRIFT")
    old_ids = {r["request_id"] for r in read(OLD_PANEL)["requests"]}
    reused_ids = rids & old_ids
    require(len(reused_ids) == 360, "FIXED_REUSE360_DRIFT")
    old_base_freeze = read(OLD_BASE.parent / "RAW_RESPONSE_FREEZE.json")
    old_base_bindings = {Path(b["path"]).stem: b for b in old_base_freeze["files"]}
    verify([old_base_bindings[rid] for rid in reused_ids])
    raws = {}
    for row in rows:
        raw = read(direct / "responses" / (row["request_id"] + ".json"))
        require(all(raw[k] == row[k] for k in IDENTITY), "RAW_REQUEST_IDENTITY_DRIFT")
        require(raw["status"] in ("RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"), "NONTERMINAL_RAW")
        require(raw["model_role"] == "BASE" and raw["model_directory"] == MODEL and raw["branch"] == "direct", "RAW_MODEL_IDENTITY_DRIFT")
        require(raw.get("gold_opened_by_solver") is False, "SOLVER_GOLD_ACCESS_FLAG_DRIFT")
        require(datetime.fromisoformat(raw["response_committed_at_utc"]) <= datetime.fromisoformat(freeze["created_at_utc"]), "RAW_COMMIT_AFTER_FREEZE")
        if row["request_id"] in reused_ids:
            old = read(OLD_BASE / (row["request_id"] + ".json"))
            require(all(old[k] == raw[k] for k in (*IDENTITY, "raw_output", "metadata", "prompt_token_ids", "status")), "REUSED_BASE_RESPONSE_CHANGED")
        raws[row["request_id"]] = raw
    # Nothing above parses gold or any previous score file.
    gold_start = now()
    save(HERE / "SCORING_STARTED.json", {"created_at_utc": gold_start, "gold_first_read_at_utc": gold_start,
         "all2910_raws_frozen_and_hash_verified_before_gold": True, "raw_freeze": binding(direct / "RAW_RESPONSE_FREEZE.json"),
         "scoring_code": binding(__file__), "prior_author_or_reviewer_gold_exposure_not_denied": True})
    private = {q["candidate_id"]: q for q in read(HERE / "private/GOLD.json")["questions"]}
    require(set(private) == qids, "PRIVATE_PUBLIC_CANDIDATE_SET_MISMATCH")
    for question in public:
        require(all(private[question["candidate_id"]][key] == value for key, value in question.items()), "PRIVATE_PUBLIC_CONTENT_MISMATCH")
    scorer = load_scorer()
    def score_rows(selected_rows, response_map):
        return [scorer.score_one(row, response_map[row["request_id"]], maps["requests"][row["request_id"]],
                maps["public_to_canonical"][row["candidate_id"]], private[row["candidate_id"]]["gold"]) for row in selected_rows]
    base_records = score_rows(rows, raws)
    base_questions = summarize(public, base_records, members, scorer)
    base_groups = build_groups(base_questions, base_records)
    # Legacy foundation payloads are normalized only after the full base freeze.
    reference = read(HERE / "S0_REFERENCE_BINDINGS.json")
    s0_bindings = reference["requests"]
    require(set(s0_bindings) == rids, "S0_MATCHED_REQUEST_SET_DRIFT")
    verify(list(s0_bindings.values()))
    s0_rows = rows
    s0_raws = {}
    for row in s0_rows:
        ref = s0_bindings[row["request_id"]]
        raw = read(ref["path"])
        cohort = ref["cohort"]
        require(cohort in ("day3", "foundation_v137"), "UNKNOWN_S0_COHORT")
        if cohort == "day3":
            require(all(raw[k] == row[k] for k in IDENTITY), "S0_DAY3_REQUEST_IDENTITY_DRIFT")
            require(raw.get("prompt_token_ids") == raws[row["request_id"]].get("prompt_token_ids"), "S0_BASE_PROMPT_TOKENS_DIFFER")
        else:
            original = ref.get("original_identity", {})
            require(all(raw[k] == original.get(k, row[k]) for k in IDENTITY if k in raw), "S0_LEGACY_REQUEST_IDENTITY_DRIFT")
            require(raw["metadata"]["prompt_token_count"] == row["prompt_tokens"], "S0_LEGACY_PROMPT_LENGTH_DRIFT")
            raw = {**{k: row[k] for k in IDENTITY}, "status": "RETURNED", "raw_output": raw["raw_output"],
                   "metadata": raw["metadata"], "output_validation_errors": raw.get("data_errors") or [],
                   "response_committed_at_utc": reference["legacy_terminal_at_utc"],
                   "legacy_conversion": "Missing terminal fields normalized after base freeze; no token IDs fabricated."}
        s0_raws[row["request_id"]] = raw
    s0_records = score_rows(s0_rows, s0_raws)
    s0_questions = summarize(public, s0_records, members, scorer)
    s0_groups = build_groups(s0_questions, s0_records)
    def usage(selected_ids):
        selected = [raws[rid] for rid in selected_ids]
        returned = [r for r in selected if r["status"] == "RETURNED"]
        return {"planned_requests": len(selected), "returned_requests": len(returned),
                "input_tokens": sum(r["metadata"]["prompt_token_count"] for r in returned),
                "output_tokens": sum(r["metadata"]["completion_token_count"] for r in returned)}
    new_usage, reused_usage = usage(rids - reused_ids), usage(reused_ids)
    require(new_usage["planned_requests"] == 2550 and reused_usage["planned_requests"] == 360, "USAGE_DENOMINATOR_DRIFT")
    for names, expected_count in ((("new_submitted_requests", "actual_new_submitted_requests", "actual_submitted_requests", "new_requests"), 2550),
                                   (("reused_requests", "reused_response_count", "reused_responses"), 360),
                                   (("total_requests", "all_requests", "planned_request_count", "total_returned_requests"), 2910)):
        count = first_value(complete, names)
        if count is not None:
            require(count == expected_count, "COMPLETE_REUSE_COUNT_DRIFT:" + names[0])
    wall = first_value(complete, ("wall_seconds", "new_run_wall_seconds", "new_inference_wall_seconds"))
    load = first_value(complete, ("model_load_wall_seconds", "load_wall_seconds"))
    metadata = {"schema_version": "day5_original_base_full_AB_train_pools_v1", "created_at_utc": now(), "status": "COMPLETE_FULL_FIXED_AB_BASE_EVALUATION",
        "primary_metric": "Question-equal whole-question exact; invalids remain zero in every fixed planned denominator.",
        "secondary_metrics": "Root-equal exact and call-pooled exact; MC Jaccard and ordering pair accuracy stay separate.",
        "scope": {"A_questions": 112, "B_questions": 112, "shared_foundation_questions": 24, "union_unique_questions": 200,
                  "new_structure_questions_per_arm": 88, "new_structure_union_questions": 176, "union_base_requests": 2910,
                  "new_base_requests": 2550, "reused_base_requests": 360, "A_B_each_MC": 50, "A_B_each_ordering": 62,
                  "S0_comparable_questions": 200, "S0_comparable_calls": len(s0_rows), "full_S0_comparison_available": True,
                  "full_R2_comparison_available": False},
        "usage": {"new_base_only": new_usage, "reused_old_base_not_new_cost": reused_usage,
                  "new_base_wall_seconds_including_model_load_close": wall, "model_load_wall_seconds": load,
                  "complete_record_verbatim": complete, "new_S0_calls": 0, "new_R2_calls": 0, "new_DEV_TEST_calls": 0},
        "interpretation": {"selection_from_new_base_scores": False, "all_data_are_training_pool_questions": True,
            "not_independent_heldout_generalization": True, "cannot_establish_feedback_causality": True,
            "common24_base_scored_once_counted_in_each_arm": True,
            "S0_comparison_scope": "Full200 union, request-matched inputs within cohort; Day3 new_structure176 and legacy foundation24 caches.",
            "display_seed_inheritance": {"new_structure": 2026101903, "foundation": 2026100800},
            "finite_choice_order_provenance": {"new_structure": "EXTRACTED_FROM_ACTUAL_DAY3_BATCH_ARRAYS",
                "foundation": "CODE_RECONSTRUCTED_FROM_HASH_MATCHED_V137_EXECUTION_CODE"},
            "legacy_S0_normalization": "Missing status/commit/promptsha normalized after base freeze using bound request and historical COMPLETE; no token IDs fabricated.",
            "original_base_saw_no_gold_in_this_run": True},
        "bindings": {"preparation": binding(HERE / "PREPARATION_FREEZE.json"), "manifest": binding(HERE / "EXECUTION_MANIFEST.json"),
            "raw_freeze": binding(direct / "RAW_RESPONSE_FREEZE.json"), "complete": binding(direct / "COMPLETE.json"),
            "pool_membership": binding(HERE / "POOL_MEMBERSHIP.json"), "private_gold": binding(HERE / "private/GOLD.json"),
            "S0_reference_bindings": binding(HERE / "S0_REFERENCE_BINDINGS.json"), "scoring_code": binding(__file__),
            "native_score_code": binding(WEEK / "day3_solver187_v1/runtime/score.py")}}
    save(HERE / "BASE_SCORES.json", {"model_role": "BASE", "gold_first_read_at_utc": gold_start,
        "records": base_records, "questions": base_questions, "groups": base_groups})
    save(HERE / "S0_SCORES.json", {"model_role": "S0", "scope": "FULL200_FROM_TWO_HISTORICAL_COHORTS",
        "records": s0_records, "questions": s0_questions, "groups": s0_groups, "new_model_calls": 0})
    save(HERE / "COMPARISON.json", {**metadata, "models": {"BASE": {"label": "原始 Qwen3-4B base", "groups": base_groups, "questions": base_questions},
        "S0": {"label": "R1 clean / S0；完整200题历史缓存", "groups": s0_groups, "questions": s0_questions},
        "R2_A1": {"status": "NOT_EVALUATED_ON_FULL_POOLS"}, "R2_B1": {"status": "NOT_EVALUATED_ON_FULL_POOLS"}}})
    stream = io.StringIO(newline="")
    fields = ["candidate_id", "root_id", "question_type", "category", "arms"]
    for label in ("BASE", "S0"):
        fields.extend(label + "_" + name for name in ("exact_calls", "valid_calls", "invalid_calls", "denominator_calls", "exact_fraction", "exact_rate"))
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    s0_by_id = {q["candidate_id"]: q for q in s0_questions}
    for question in base_questions:
        row = {key: question[key] for key in fields[:4]}
        row["arms"] = ",".join(question["arms"])
        for label, value in (("BASE", question), ("S0", s0_by_id.get(question["candidate_id"]))):
            if value is None:
                continue
            for name in ("exact_calls", "valid_calls", "invalid_calls", "denominator_calls"):
                row[label + "_" + name] = value[name]
            fraction = Fraction(value["exact_calls"], value["denominator_calls"])
            row[label + "_exact_fraction"], row[label + "_exact_rate"] = str(fraction), float(fraction)
        writer.writerow(row)
    save(HERE / "per_question.csv", stream.getvalue())
    def pct(groups, key, metric="question_equal_exact"):
        return f"{100 * groups[key][metric]['value']:.2f}%"
    def table_row(label, key):
        value = base_groups[key]
        return f"| {label} | {value['questions']} | {pct(base_groups,key)} | {pct(s0_groups,key)} | {pct(base_groups,key,'root_equal_exact')} | {value['valid_calls']}/{value['planned_calls']} |"
    lines = ["# 原始 base：完整 A/B 训练池评估", "",
        "本次覆盖两条路线的全部冻结训练题：A、B 各 112 道，其中各 88 道新增结构题和共同的 24 道基础题；去重后共 200 道。原始 base 完成固定 2,910 次直接作答：复用旧面板的 360 条回答，新增 2,550 次。", "",
        "主指标按题等权：先计算每道题在固定请求下整题完全正确的比例，再对题目取平均。多选每题 18 次、排序每题 12 次。按 root 等权和按调用汇总另行保留，不能混用。", "",
        "| 范围 | 题数 | base 按题等权 | R1 clean 按题等权 | base 按 root 等权 | base 有效/固定请求 |", "|---|---:|---:|---:|---:|---:|"]
    for cohort, category, label in (("A", "ALL", "A 全部训练池"), ("B", "ALL", "B 全部训练池"),
            ("A", "new_structure", "A 新增结构题"), ("B", "new_structure", "B 新增结构题"),
            ("UNION", "foundation", "共同基础题（去重）"), ("UNION", "ALL", "两路去重合计")):
        lines.append(table_row(label, cohort + "/" + category + "/ALL"))
    lines += ["", "按题型查看原始 base（整题完全正确率）：", "", "| 范围 | 多选题数 | 多选正确率 | 排序题数 | 排序正确率 |", "|---|---:|---:|---:|---:|"]
    for cohort, category, label in (("A", "ALL", "A 全部"), ("B", "ALL", "B 全部"),
            ("A", "new_structure", "A 新增结构"), ("B", "new_structure", "B 新增结构"),
            ("UNION", "foundation", "共同基础"), ("UNION", "ALL", "去重合计")):
        mc, order = cohort + "/" + category + "/multi_select", cohort + "/" + category + "/ordering"
        lines.append(f"| {label} | {base_groups[mc]['questions']} | {pct(base_groups,mc)} | {base_groups[order]['questions']} | {pct(base_groups,order)} |")
    lines += ["", "全部 200 题均绑定训练前 R1 clean（S0）的同请求缓存：176 道新增结构题来自 Day3，24 道基础题来自历史 v137。两模型在每题上使用相同来源、题干、显示、提示词和随机种子；新增结构与基础两个批次分别继承显示 seed 2026101903、2026100800，不宣称两批之间采用同一显示 seed。新增结构题的有限答案数组提取自实际 Day3 批次；基础题数组按 hash 匹配的 v137 执行代码重建，来源标记为 CODE_RECONSTRUCTED_FROM_HASH_MATCHED_V137_EXECUTION_CODE。历史基础 raw 缺失的终止字段仅在评分阶段按已绑定请求和历史 COMPLETE 时间转换，不补造原始 token IDs，也不复用旧 score 字段。本次未给 R2 模型补测全池，不能将先前 24 题的 R2 成绩填入本表。", "",
        "| 新增结构子集 | 题数 | 原始 base | R1 clean / S0 |", "|---|---:|---:|---:|"]
    for cohort, label in (("A", "A 新增结构"), ("B", "B 新增结构"), ("UNION", "去重新增结构")):
        for typ, text in (("ALL", "全部"), ("multi_select", "多选"), ("ordering", "排序")):
            key = cohort + "/new_structure/" + typ
            lines.append(f"| {label}：{text} | {base_groups[key]['questions']} | {pct(base_groups,key)} | {pct(s0_groups,key)} |")
    lines += ["", "部分正确分别计分，不合并题型：", "", "| 去重 200 题中的题型 | 原始 base 的题等权部分分 |", "|---|---:|",
        "| 85 道多选：Jaccard | " + pct(base_groups, "UNION/ALL/multi_select", "question_equal_mc_jaccard_secondary") + " |",
        "| 115 道排序：两两顺序准确率 | " + pct(base_groups, "UNION/ALL/ordering", "question_equal_order_pair_accuracy_secondary") + " |", "",
        "共同基础 24 题只推理一次，在 A/B 各自 112 题的训练池口径中分别纳入；去重 200 题表仅计一次。A/B 彼此不是完全相同的题，分数差只能描述各自完整池对该模型的难度，不能直接归因为反馈。比较反馈策略应重点看新增结构排序题；多选采用两路共同流程。", "",
        "这是训练池难度与拟合基线，不是独立 DEV/TEST 泛化评估。重复作答不是独立新题，统计解释应保留 root 与来源相关性。base 与 S0 的差异反映两个训练阶段模型在固定题池上的表现，不能单凭该差异证明自主发现弱点或反馈的因果收益。排序整题低分也需结合两两顺序分数，避免误判为完全不会排序。", "",
        f"本次新增成本只计 2,550 次原始 base 调用：输入 {new_usage['input_tokens']:,} tokens，输出 {new_usage['output_tokens']:,} tokens；复用 360 条旧 base 及 {len(s0_rows):,} 条 S0 回答的历史消耗未记入新增成本。新增执行耗时（包括加载/关闭）为 {wall} 秒，模型加载为 {load} 秒，原始 COMPLETE 全记录保留于 COMPARISON.json。没有新增 S0、R2、DEV/TEST 或简短证据调用。", "",
        "在本评分进程中，全部 2,910 条原始 base 回答提交、冻结并核验 hash 后才首次解析 Gold JSON；作者与准备阶段可能已接触答案，不作相反主张。没有根据成绩筛题、变更池子或自动重试。逐请求评分见 BASE_SCORES.json；S0 同口径评分见 S0_SCORES.json；每题计数与精确分数见 per_question.csv；完整分组和来源绑定见 COMPARISON.json。旧实验文件均未修改。", ""]
    save(HERE / "REPORT_ZH.md", "\n".join(lines))
    print(json.dumps({"status": metadata["status"], "base_groups": base_groups, "new_usage": new_usage,
        "reused_base_requests": 360, "S0_reused_requests": len(s0_rows)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
