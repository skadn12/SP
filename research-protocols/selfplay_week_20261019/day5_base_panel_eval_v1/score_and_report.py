"""Score a terminal-frozen original-base run and compare the same 24-question panel.

No inference, selection, retry, or training entry point is provided. Existing
frozen Day 5 records are read-only. Every new output is created exclusively.
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
UPSTREAM = WEEK / "day5_training_v1/train_diagnosis_prepare"
ROLES = ("BASE", "S0", "A1", "B1")
LABELS = {"BASE": "原始 base（Qwen3-4B）", "S0": "R1 clean（S0 / M-clean）",
          "A1": "R2-A1（无失败反馈）", "B1": "R2-B1（含排序反馈）"}
IDENTITY = ("request_id", "candidate_id", "root_id", "question_type", "variant_id", "slot", "seed")
OUTPUTS = ("SCORING_STARTED.json", "BASE_SCORES.json", "COMPARISON.json", "REPORT_ZH.md", "per_question.csv")


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


def verify_nested_bindings(payload):
    """Hash verification does not parse any bound private gold file."""
    if isinstance(payload, dict):
        if {"path", "sha256", "bytes"} <= payload.keys():
            verify([payload])
        else:
            for value in payload.values():
                verify_nested_bindings(value)
    elif isinstance(payload, list):
        for value in payload:
            verify_nested_bindings(value)


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def rational(value):
    value = Fraction(value)
    return {"fraction": str(value), "value": float(value)}


def mean(values):
    values = list(values)
    require(bool(values), "EMPTY_METRIC_DENOMINATOR")
    return sum(values, Fraction()) / len(values)


def aggregate(questions, records):
    by_id = collections.defaultdict(list)
    for record in records:
        by_id[record["candidate_id"]].append(record)
    roots = collections.defaultdict(list)
    rates = []
    for question in questions:
        qrows = by_id[question["candidate_id"]]
        require(len(qrows) == question["denominator_calls"], "QUESTION_DENOMINATOR_DRIFT")
        rate = Fraction(sum(record["exact"] for record in qrows), len(qrows))
        rates.append(rate)
        roots[question["root_id"]].append(rate)
    result = {"questions": len(questions), "roots": len(roots), "planned_calls": len(records),
              "valid_calls": sum(record["valid"] for record in records),
              "invalid_calls": sum(not record["valid"] for record in records),
              "exact_calls": sum(record["exact"] for record in records),
              "question_equal_exact": rational(mean(rates)),
              "root_equal_exact": rational(mean(mean(values) for values in roots.values())),
              "call_pooled_exact_secondary": rational(Fraction(sum(record["exact"] for record in records), len(records)))}
    types = {question["question_type"] for question in questions}
    # These task-specific secondary measures are never combined across types.
    if types == {"multi_select"}:
        values = []
        for question in questions:
            per_call = []
            for record in by_id[question["candidate_id"]]:
                predicted = set(record["prediction_content"] or [])
                gold = set(record["gold_content"])
                per_call.append(Fraction(len(predicted & gold), len(predicted | gold)) if record["valid"] else Fraction())
            values.append(mean(per_call))
        result["question_equal_mc_jaccard_secondary"] = rational(mean(values))
    if types == {"ordering"}:
        result["question_equal_order_pair_accuracy_secondary"] = rational(mean(
            mean(Fraction(record["metrics"]["correct_pairs"], record["metrics"]["total_pairs"])
                 for record in by_id[question["candidate_id"]]) for question in questions))
    return result


def group_all(questions, records):
    groups = {}
    for arm in ("ALL", "A", "B"):
        for typ in ("ALL", "multi_select", "ordering"):
            subset = [q for q in questions if (arm == "ALL" or q["source_arm"] == arm)
                      and (typ == "ALL" or q["question_type"] == typ)]
            ids = {q["candidate_id"] for q in subset}
            groups[arm + "/" + typ] = aggregate(subset, [r for r in records if r["candidate_id"] in ids])
    return groups


def assert_same_group(actual, expected, label):
    for key in ("questions", "roots", "planned_calls", "valid_calls", "invalid_calls", "exact_calls"):
        require(actual[key] == expected[key], "OLD_SUMMARY_COUNT_MISMATCH:" + label + ":" + key)
    for key in ("question_equal_exact", "root_equal_exact", "call_pooled_exact_secondary"):
        require(Fraction(actual[key]["fraction"]) == Fraction(expected[key]["fraction"]),
                "OLD_SUMMARY_FRACTION_MISMATCH:" + label + ":" + key)
        require(abs(actual[key]["value"] - expected[key]["value"]) < 1e-12,
                "OLD_SUMMARY_FLOAT_MISMATCH:" + label + ":" + key)


def validate_summary(models, summary):
    checked = []
    own = "OWN_ARM_TRAINED_QUESTION"
    opposite = "OPPOSITE_ARM_QUESTION_SHARED_ROOT_MAY_SHARE_FACTS"
    for role in ("A1", "B1"):
        for label in ("S0", role):
            expected = summary["S0"]["direct_groups_relative_to_" + role] if label == "S0" else summary["models"][role]["direct"]
            for typ in ("ALL", "multi_select", "ordering"):
                for exposure, arm in (("ALL", "ALL"), (own, role[0]), (opposite, "B" if role == "A1" else "A")):
                    key = typ + "/" + exposure
                    assert_same_group(models[label]["groups"][arm + "/" + typ], expected[key], label + ":" + role + ":" + key)
                    checked.append(label + ":" + role + ":" + key)
    return {"status": "PASS", "groups_checked": len(checked), "checked": checked}


def load_scorer():
    directory = WEEK / "day3_solver187_v1/runtime"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("day5_base_panel_native_score", directory / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    for output in OUTPUTS:
        require(not (HERE / output).exists(), "NO_OVERWRITE:" + output)
    preparation = read(HERE / "PREPARATION_FREEZE.json")
    validation = read(HERE / "CPU_VALIDATION.json")
    manifest = read(HERE / "EXECUTION_MANIFEST.json")
    require(validation.get("status", "").startswith("PASS"), "CPU_VALIDATION_NOT_PASS")
    require(manifest["model_role"] == "BASE", "ORIGINAL_BASE_ROLE_REQUIRED")
    require(Path(manifest["model_directory"]).resolve() == Path("/root/autodl-tmp/Qwen3-4B").resolve(),
            "ORIGINAL_BASE_DIRECTORY_REQUIRED")
    verify_nested_bindings(preparation)
    verify_nested_bindings(validation)
    verify_nested_bindings(manifest)
    rows = read(UPSTREAM / "public/DIRECT_REQUESTS.json")["requests"]
    public = read(UPSTREAM / "public/PUBLIC_QUESTIONS.json")["questions"]
    maps = read(UPSTREAM / "public/DIRECT_DISPLAY_MAPS.json")
    selection = read(UPSTREAM / "SELECTION.json")
    annotations = {q["candidate_id"]: q for q in selection["question_annotations"]}
    row_by_id = {r["request_id"]: r for r in rows}
    question_ids = {q["candidate_id"] for q in public}
    require(len(rows) == len(row_by_id) == 360 and len(question_ids) == len(public) == 24, "FIXED_PANEL_SIZE_DRIFT")
    require(len({q["root_id"] for q in public}) == 6 and set(annotations) == question_ids, "FIXED_ROOT_PANEL_DRIFT")
    require(set(maps["requests"]) == set(row_by_id) and set(maps["public_to_canonical"]) == question_ids,
            "DISPLAY_MAPPING_IDENTITY_DRIFT")
    for row in rows:
        require(row["candidate_id"] in question_ids and sha(row["prompt_path"]) == row["prompt_sha256"], "PROMPT_OR_CANDIDATE_DRIFT")
    for question in public:
        require(sha(question["source_path"]) == question["source_sha256"], "SOURCE_DRIFT")
        planned = [r for r in rows if r["candidate_id"] == question["candidate_id"]]
        require(len(planned) == (18 if question["question_type"] == "multi_select" else 12), "FIXED_PER_QUESTION_CALLS_DRIFT")
        require(all(r["root_id"] == question["root_id"] and r["question_type"] == question["question_type"] for r in planned),
                "QUESTION_REQUEST_IDENTITY_DRIFT")
    direct = HERE / "direct"
    freeze = read(direct / "RAW_RESPONSE_FREEZE.json")
    complete = read(direct / "COMPLETE.json")
    started = read(direct / "STARTED.json")
    require(freeze["status"] == "COMMITTED" and freeze["planned_request_count"] == 360, "TERMINAL_RAW_FREEZE_REQUIRED")
    require(complete["status"] == "COMPLETE" and complete["actual_submitted_requests"] == complete["returned_requests"] == 360,
            "COMPLETE_FIXED360_REQUIRED")
    verify(freeze["files"] + [freeze["request_freeze"], freeze["execution_manifest"]])
    require(Path(freeze["execution_manifest"]["path"]).resolve() == HERE / "EXECUTION_MANIFEST.json", "WRONG_EXECUTION_MANIFEST")
    require(Path(freeze["request_freeze"]["path"]).resolve() == HERE / "PREPARATION_FREEZE.json", "WRONG_REQUEST_FREEZE")
    expected_paths = {direct / "responses" / (row["request_id"] + ".json") for row in rows}
    require(len(freeze["files"]) == 360 and {Path(b["path"]).resolve() for b in freeze["files"]} == expected_paths,
            "RAW_FREEZE_DENOMINATOR_MISMATCH")
    require(set((direct / "responses").glob("*.json")) == expected_paths, "EXTRA_OR_MISSING_RAW_RESPONSE")
    raws = {}
    for row in rows:
        raw = read(direct / "responses" / (row["request_id"] + ".json"))
        require(all(raw[k] == row[k] for k in IDENTITY + ("prompt_sha256",)), "BASE_RAW_IDENTITY_DRIFT")
        require(raw["model_role"] == "BASE" and raw["model_directory"] == manifest["model_directory"] and raw["branch"] == "direct", "BASE_RAW_MODEL_DRIFT")
        require(raw["status"] in ("RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"), "NONTERMINAL_RAW_RESPONSE")
        require(datetime.fromisoformat(raw["response_committed_at_utc"]) <= datetime.fromisoformat(freeze["created_at_utc"]), "RAW_AFTER_FREEZE")
        require(raw.get("gold_opened_by_solver") is False, "SOLVER_GOLD_ACCESS_FLAG_DRIFT")
        raws[row["request_id"]] = raw
    # First parse of any gold file or old score follows verification of all 360 new raws.
    gold_start = now()
    save(HERE / "SCORING_STARTED.json", {"created_at_utc": gold_start, "gold_first_read_at_utc": gold_start,
         "all_360_terminal_raws_hash_verified_before_gold": True, "raw_freeze": binding(direct / "RAW_RESPONSE_FREEZE.json"),
         "code": binding(__file__), "prior_author_or_reviewer_gold_exposure_not_denied": True})
    baseline = read(UPSTREAM / "private/S0_BASELINE_BINDINGS.json")
    verify([baseline["gold_binding"]] + baseline["selected_raw_responses"])
    private = {q["candidate_id"]: q for q in read(baseline["gold_binding"]["path"])["questions"]}
    require(question_ids <= set(private), "GOLD_PANEL_MISSING")
    for question in public:
        saved = private[question["candidate_id"]]
        require(all(saved[k] == v for k, v in question.items()), "PRIVATE_PUBLIC_CONTENT_DRIFT")
    scorer = load_scorer()
    base_records = [scorer.score_one(row, raws[row["request_id"]], maps["requests"][row["request_id"]],
                    maps["public_to_canonical"][row["candidate_id"]], private[row["candidate_id"]]["gold"]) for row in rows]
    base_questions = [scorer.summarize_question(q, [r for r in base_records if r["candidate_id"] == q["candidate_id"]])
                      | annotations[q["candidate_id"]] for q in public]
    models = {"BASE": {"records": base_records, "questions": base_questions, "groups": group_all(base_questions, base_records)}}
    summary = read(UPSTREAM / "DIAGNOSIS_SUMMARY.json")
    source_scores = {}
    for role in ("A1", "B1"):
        score_binding = summary["models"][role]["direct_scores"]
        verify([score_binding])
        payload = read(score_binding["path"])
        require(payload["role"] == role and payload["scope"] == "TRAIN_DIAGNOSIS_ONLY", "OLD_SCORE_ROLE_DRIFT")
        source_scores[role] = score_binding
        for label in ("S0", role):
            saved = payload["models"][label]
            records = saved["requests"]
            require(len(records) == 360 and {r["request_id"] for r in records} == set(row_by_id), "OLD_RECORD_DENOMINATOR_DRIFT")
            for record in records:
                require(all(record[k] == row_by_id[record["request_id"]][k] for k in IDENTITY), "OLD_RECORD_IDENTITY_DRIFT")
                require(record["gold_canonical"] == private[record["candidate_id"]]["gold"], "OLD_RECORD_GOLD_DRIFT")
            questions = [scorer.summarize_question(q, [r for r in records if r["candidate_id"] == q["candidate_id"]])
                         | annotations[q["candidate_id"]] for q in public]
            groups = group_all(questions, records)
            if label in models:
                require(models[label]["records"] == records and models[label]["groups"] == groups, "DUPLICATED_S0_RECORDS_DIFFER")
            else:
                models[label] = {"records": records, "questions": questions, "groups": groups}
    old_check = validate_summary(models, summary)
    input_tokens = sum(raw["metadata"]["prompt_token_count"] for raw in raws.values() if raw["status"] == "RETURNED")
    output_tokens = sum(raw["metadata"]["completion_token_count"] for raw in raws.values() if raw["status"] == "RETURNED")
    usage = {"scope": "NEW_ORIGINAL_BASE_DIRECT360_ONLY", "new_model_calls": 360, "reused_old_response_records": 1080,
             "input_tokens": input_tokens, "output_tokens": output_tokens,
             "wall_seconds_including_model_load_and_close": complete["wall_seconds"],
             "model_load_wall_seconds": complete["model_load_wall_seconds"],
             "started_at_utc": started["started_at_utc"], "completed_at_utc": complete["completed_at_utc"],
             "old_S0_A1_B1_new_calls": 0, "new_brief_evidence_calls": 0, "new_dev_test_calls": 0}
    metadata = {"schema_version": "day5_original_base_fixed24_comparison_v1", "created_at_utc": now(),
                "status": "COMPLETE_SAME_FIXED24_SAME_DIRECT360_FOUR_MODEL_COMPARISON", "model_labels": LABELS,
                "scope": {"questions": 24, "roots": 6, "source_arms": {"A": 12, "B": 12},
                          "question_types": {"multi_select": 12, "ordering": 12}, "direct_calls_per_model": 360,
                          "MC_calls_per_question": 18, "ordering_calls_per_question": 12},
                "primary_metric": "Question-equal whole-question exact; invalids remain zero in fixed denominators.",
                "secondary_metrics": "Root-equal exact and call-pooled exact; MC Jaccard and ordering pair accuracy are separate task-specific measures.",
                "selection": "Existing Day5 frozen panel, fixed before A1/B1 inference; no selection from original-base scores.",
                "interpretation": {"panel_is_training_diagnosis": True, "independent_heldout_generalization": False,
                     "A1_trained_on_A_questions": True, "B1_trained_on_B_questions": True,
                     "opposite_arm_shares_roots_sources_and_may_share_facts": True,
                     "base_vs_S0_can_compare_pre_R2_relative_difficulty_on_this_panel_only": True,
                     "formal_feedback_causal_benefit_established_by_this_comparison": False},
                "actual_new_usage": usage, "old_summary_crosscheck": old_check,
                "bindings": {"preparation": binding(HERE / "PREPARATION_FREEZE.json"), "CPU_validation": binding(HERE / "CPU_VALIDATION.json"),
                     "execution_manifest": binding(HERE / "EXECUTION_MANIFEST.json"), "raw_freeze": binding(direct / "RAW_RESPONSE_FREEZE.json"),
                     "base_complete": binding(direct / "COMPLETE.json"), "selection": binding(UPSTREAM / "SELECTION.json"),
                     "old_diagnosis_summary": binding(UPSTREAM / "DIAGNOSIS_SUMMARY.json"), "old_direct_scores": source_scores,
                     "private_gold": baseline["gold_binding"], "scoring_code": binding(__file__),
                     "native_score_code": binding(WEEK / "day3_solver187_v1/runtime/score.py")}}
    save(HERE / "BASE_SCORES.json", {"role": "BASE", "scope": "TRAIN_DIAGNOSIS_FIXED24_ONLY", "gold_first_read_at_utc": gold_start,
         "raw_freeze": binding(direct / "RAW_RESPONSE_FREEZE.json"), **models["BASE"]})
    save(HERE / "COMPARISON.json", {**metadata, "models": {role: {"label": LABELS[role],
         "groups": models[role]["groups"], "questions": models[role]["questions"]} for role in ROLES}})
    stream = io.StringIO(newline="")
    fields = ["candidate_id", "root_id", "source_arm", "question_type", "design_id"]
    for role in ROLES:
        fields.extend(role + "_" + name for name in ("exact_calls", "valid_calls", "invalid_calls", "denominator_calls", "exact_fraction", "exact_rate"))
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for question in public:
        qid = question["candidate_id"]
        row = {key: annotations[qid][key] for key in fields[:5]}
        for role in ROLES:
            item = next(q for q in models[role]["questions"] if q["candidate_id"] == qid)
            for name in ("exact_calls", "valid_calls", "invalid_calls", "denominator_calls"):
                row[role + "_" + name] = item[name]
            fraction = Fraction(item["exact_calls"], item["denominator_calls"])
            row[role + "_exact_fraction"], row[role + "_exact_rate"] = str(fraction), float(fraction)
        writer.writerow(row)
    save(HERE / "per_question.csv", stream.getvalue())
    def percent(role, key, metric="question_equal_exact"):
        return f"{100 * models[role]['groups'][key][metric]['value']:.2f}%"
    lines = ["# 原始 base 补测与 R1 / R2 同题比较", "",
             "已补齐原始 Qwen3-4B base 在既定 24 道 Train 诊断题上的 360 次直接作答。R1 clean 就是 S0 / M-clean；本次 R2 分别为 A1、B1 两个训练终点。", "",
             "主表为按题等权的整题完全正确率：先计算每题固定请求的正确比例，再对题目等权平均。多选每题 18 次、排序每题 12 次，因此不把 360 次调用直接混合平均当主指标。", "",
             "| 模型 | A 组 12 题 | B 组 12 题 | 合计 24 题 |", "|---|---:|---:|---:|"]
    for role in ROLES:
        lines.append(f"| {LABELS[role]} | {percent(role, 'A/ALL')} | {percent(role, 'B/ALL')} | {percent(role, 'ALL/ALL')} |")
    lines += ["", "按题型比较（均为整题完全正确率）：", "", "| 范围 | 题数 | 原始 base | R1 clean | R2-A1 | R2-B1 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, arm_label in (("ALL", "全部"), ("A", "A 组"), ("B", "B 组")):
        for typ, typ_label in (("multi_select", "多选"), ("ordering", "排序")):
            key = arm + "/" + typ
            lines.append(f"| {arm_label}{typ_label} | {models['BASE']['groups'][key]['questions']} | " + " | ".join(percent(role, key) for role in ROLES) + " |")
    lines += ["", "整体的 root 等权正确率与按题等权正确率在本面板上相同，因为每个 root 的题数和题型配额相同；完整精确分数、分组计数及按调用汇总保存在 COMPARISON.json。", "",
              "部分正确仅分题型报告：多选用 Jaccard，排序用两两顺序准确率；两者不合并为一个部分正确率。无效输出始终计入既定分母且得零分。", "",
              "| 范围与指标 | 原始 base | R1 clean | R2-A1 | R2-B1 |", "|---|---:|---:|---:|---:|"]
    for key, metric, label in (("ALL/multi_select", "question_equal_mc_jaccard_secondary", "12 道多选：Jaccard"),
                                ("ALL/ordering", "question_equal_order_pair_accuracy_secondary", "12 道排序：两两顺序准确率")):
        lines.append("| " + label + " | " + " | ".join(percent(role, key, metric) for role in ROLES) + " |")
    lines += ["", "这 24 题来自 6 个训练 root：A1 训练过 A 组 12 题，B1 训练过 B 组 12 题；对侧组也共享 root、来源及部分事实。因此，R2 结果主要反映训练拟合和同源迁移，不能视为独立泛化，也不能单独证明反馈的因果收益。训练前的 base、R1 表现可用于比较该固定面板内 A/B 题的相对难度，样本仅每组 12 题。", "",
              f"本次只新增原始 base 的 360 次直接作答：输入 {input_tokens:,} tokens，输出 {output_tokens:,} tokens，执行耗时 {complete['wall_seconds']:.2f} 秒（包括模型加载及关闭；模型加载 {complete['model_load_wall_seconds']:.2f} 秒）。S0、A1、B1 共 1,080 条既有作答记录复用，没有新增这些模型的调用，没有调用 DEV / TEST，也没有补测简短证据分支。", "",
              "固定 360 条原始输出全部提交、冻结、校验 hash 后才读取 gold 并评分。旧 S0/A1/B1 分组按整数计数与 Fraction 重算，与原 DIAGNOSIS_SUMMARY.json 的 36 组统计一致。未根据本次分数选题、调整模型或重试。", "",
              "每题四个模型的 exact / valid / invalid / 固定分母见 per_question.csv；原始 base 的逐请求评分见 BASE_SCORES.json；原始输出在 direct/responses。既有冻结记录未作修改。", ""]
    save(HERE / "REPORT_ZH.md", "\n".join(lines))
    print(json.dumps({"status": metadata["status"], "base_groups": models["BASE"]["groups"], "actual_new_usage": usage}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
