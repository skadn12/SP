"""Verify the complete v123 run and report descriptive paired Solver results.

This tool makes no network calls and never changes questions or training pools.
Run only after evaluate_tju.py has written run/COMPLETE.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parents[1]
LOCAL = BATCH.parent / "h4_codex_remaining_six_repair_v122"
SCORING = BATCH.parent / "h4_feedback_training_pool_v15_qa/evaluation/scoring.py"
GROUPS = ("all", "ms_gold2_5", "ms_gold6_control", "ordering", "gold2", "gold3", "gold4", "gold5")
LABELS = {"all": "全部", "ms_gold2_5": "多选 Gold2–5", "ms_gold6_control": "多选 Gold6 对照",
          "ordering": "排序", "gold2": "Gold2", "gold3": "Gold3", "gold4": "Gold4", "gold5": "Gold5"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def verify_binding(row, allowed, expected=None):
    need(isinstance(row, dict) and "path" in row and "sha256" in row, "MALFORMED_FILE_BINDING")
    path = Path(row["path"]).resolve()
    need(any(path == root or root in path.parents for root in allowed), "BINDING_OUTSIDE_PUBLIC_EVIDENCE:" + str(path))
    if expected is not None:
        need(path == Path(expected).resolve(), "BINDING_PATH_MISMATCH:" + str(path))
    need(path.is_file() and sha(path) == row["sha256"], "BOUND_FILE_CHANGED:" + str(path))


def fraction_value(value):
    return {"fraction": str(value), "value": float(value)} if value is not None else None


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def mean_partial(rows):
    return fraction_value(sum((Fraction(r["score"]["partial_score"]) for r in rows), Fraction()) / len(rows)) if rows else None


def group_for(question, gold):
    if question["question_type"] == "ordering":
        return {"all", "ordering"}
    size = len(gold[question["candidate_id"]])
    return {"all", "ms_gold6_control"} if size == 6 else {"all", "ms_gold2_5", "gold" + str(size)}


def count_labels(values):
    return dict(sorted(Counter("<missing>" if value is None else str(value) for value in values).items()))


def usage_summary(rows):
    fields = {"prompt_tokens": ("prompt_tokens",), "completion_tokens": ("completion_tokens",),
              "total_tokens": ("total_tokens",), "reasoning_tokens": ("completion_tokens_details", "reasoning_tokens")}
    result = {"record_count": len(rows), "records_with_usage_object": sum(isinstance(r.get("usage"), dict) for r in rows),
              "missing_usage_object_count": sum(not isinstance(r.get("usage"), dict) for r in rows),
              "scope": "All completed evaluation request records, including evaluation-invalid responses; excludes probes.",
              "missing_is_zero": False, "fields": {}}
    for label, path in fields.items():
        observed, missing, invalid = [], 0, 0
        for row in rows:
            value = row.get("usage")
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if value is None:
                missing += 1
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                observed.append(value)
            else:
                invalid += 1
        result["fields"][label] = {"reported_sum": sum(observed) if observed else None,
            "reported_record_count": len(observed), "missing_record_count": missing,
            "invalid_value_record_count": invalid,
            "mean_among_reported": sum(observed) / len(observed) if observed else None}
    return result


def aggregate(ids, requests, questions, gold, records):
    planned = [requests[rid] for rid in ids]
    completed = [records[rid] for rid in ids]
    valid = [r for r in completed if r["_valid"]]
    exact = sum(r["score"]["exact_match"] is True for r in valid)
    invalid = [r for r in completed if not r["_valid"]]
    ms = [r for r in valid if r["question_type"] == "multi_select"]
    ms_ids = [r["request_id"] for r in ms]
    nongold_opportunities = sum(len(requests[rid]["variant"]["options"]) - len(gold[requests[rid]["candidate_id"]]) for rid in ms_ids)
    gold_opportunities = sum(len(gold[requests[rid]["candidate_id"]]) for rid in ms_ids)
    fp = sum(r["score"]["fp"] for r in ms)
    fn = sum(r["score"]["fn"] for r in ms)
    pred_counts = [len(r["score"]["content_prediction"]) for r in ms]
    all_selected = sum(len(r["score"]["content_prediction"]) == len(requests[r["request_id"]]["variant"]["options"]) for r in ms)
    return {"question_count": len({r["candidate_id"] for r in planned}),
        "planned_call_count": len(ids), "complete_call_count": len(completed),
        "evaluation_valid_call_count": len(valid), "evaluation_invalid_call_count": len(invalid),
        "valid_coverage": rate(len(valid), len(ids)),
        "exact_match_on_valid": rate(exact, len(valid)),
        "exact_success_on_planned": rate(exact, len(ids)),
        "partial_score_on_valid_only": {"denominator": len(valid), "mean": mean_partial(valid)},
        "invalid_reasons": dict(sorted(Counter(reason for r in invalid for reason in r["_invalid_reasons"]).items())),
        "native_scoring_status": count_labels(r["score"].get("scoring_status") for r in completed),
        "multi_select_valid_only": {"valid_call_count": len(ms), "false_positive_count": fp,
            "omitted_gold_count": fn, "any_false_selection": rate(sum(r["score"]["fp"] > 0 for r in ms), len(ms)),
            "any_omission": rate(sum(r["score"]["fn"] > 0 for r in ms), len(ms)),
            "false_selection_per_nongold_opportunity": rate(fp, nongold_opportunities),
            "omission_per_gold_opportunity": rate(fn, gold_opportunities),
            "all_options_selected": rate(all_selected, len(ms)),
            "all_selected_exact_match_count": sum(r["score"]["exact_match"] is True and len(r["score"]["content_prediction"]) == len(requests[r["request_id"]]["variant"]["options"]) for r in ms),
            "mean_selected_count": sum(pred_counts) / len(ms) if ms else None,
            "predicted_count_above_gold": rate(sum(len(r["score"]["content_prediction"]) > len(gold[r["candidate_id"]]) for r in ms), len(ms)),
            "predicted_count_below_gold": rate(sum(len(r["score"]["content_prediction"]) < len(gold[r["candidate_id"]]) for r in ms), len(ms))}}


def paired(ids, school, local):
    shared = [rid for rid in ids if school[rid]["_valid"] and local[rid]["_valid"]]
    srows, lrows = [school[rid] for rid in shared], [local[rid] for rid in shared]
    school_right = sum(r["score"]["exact_match"] is True for r in srows)
    local_right = sum(r["score"]["exact_match"] is True for r in lrows)
    partial_delta = sum((Fraction(s["score"]["partial_score"]) - Fraction(l["score"]["partial_score"]) for s, l in zip(srows, lrows)), Fraction()) / len(shared) if shared else None
    table = Counter((s["score"]["exact_match"] is True, l["score"]["exact_match"] is True) for s, l in zip(srows, lrows))
    return {"planned_call_count": len(ids), "paired_valid_call_count": len(shared),
            "paired_valid_coverage": rate(len(shared), len(ids)),
            "school_exact_match": rate(school_right, len(shared)), "local_exact_match": rate(local_right, len(shared)),
            "school_minus_local_exact_percentage_points": 100 * (school_right - local_right) / len(shared) if shared else None,
            "school_mean_partial": mean_partial(srows), "local_mean_partial": mean_partial(lrows),
            "school_minus_local_mean_partial": fraction_value(partial_delta),
            "school_only_correct": table[(True, False)], "local_only_correct": table[(False, True)],
            "both_correct": table[(True, True)], "both_incorrect": table[(False, False)],
            "excluded_school_invalid_count": sum(not school[rid]["_valid"] for rid in ids),
            "excluded_local_invalid_count": sum(not local[rid]["_valid"] for rid in ids),
            "paired_request_ids": shared}


def verify_record(record, request, question, gold, scorer, school):
    rid = request["request_id"]
    identity = {key: request[key] for key in ("request_id", "candidate_id", "root_id", "question_type", "mode", "slot", "seed")}
    identity["variant_id"] = request["variant"]["variant_id"]
    for key, value in identity.items():
        need(record.get(key) == value, "RESPONSE_IDENTITY_MISMATCH:" + rid + ":" + key)
    need(type(record.get("evaluation_valid")) is bool and isinstance(record.get("data_errors"), list), "RESPONSE_VALIDITY_SCHEMA:" + rid)
    score = scorer.score_response(dict(question, gold_content_ids=gold[request["candidate_id"]]), request, record.get("raw_output"))
    recorded_score = record.get("score")
    need(isinstance(recorded_score, dict), "SCORE_MISSING:" + rid)
    for key, value in score.items():
        need(key in recorded_score and recorded_score[key] == value, "NATIVE_SCORE_MISMATCH:" + rid + ":" + key)
    reasons = []
    if record["evaluation_valid"] is not True:
        reasons.append("RECORDED_EVALUATION_INVALID")
    reasons.extend("DATA_ERROR:" + str(error) for error in record["data_errors"])
    if score["scored"] is not True:
        reasons.append("NATIVE_SCORE:" + str(score["scoring_status"]))
    if school:
        if record.get("transport_status") != "HTTP_OK":
            reasons.append("TRANSPORT:" + str(record.get("transport_status")))
        if record.get("finish_reason") not in {"stop", "eos"}:
            reasons.append("INCOMPLETE_FINISH_REASON:" + str(record.get("finish_reason")))
        need(record.get("http_attempt_count") == 1, "HTTP_ATTEMPT_COUNT_NOT_ONE:" + rid)
        expected_errors = []
        if record.get("transport_status") != "HTTP_OK":
            expected_errors.append(record.get("transport_status"))
        if not isinstance(record.get("raw_output"), str):
            expected_errors.append("CONTENT_MISSING_OR_NOT_TEXT")
        if record.get("finish_reason") not in {"stop", "eos"}:
            expected_errors.append("INCOMPLETE_FINISH_REASON:" + str(record.get("finish_reason")))
        if not score.get("scored"):
            expected_errors.append("NATIVE_SCORE_FAILURE:" + str(score.get("scoring_status")))
        need(record["data_errors"] == expected_errors and record["evaluation_valid"] == (not expected_errors), "SCHOOL_VALIDITY_INCONSISTENT:" + rid)
    record["score"] = score
    record["_invalid_reasons"] = reasons
    record["_valid"] = not reasons
    return record


def load_records(folder, requests, questions, gold, scorer, school, allowed):
    complete = read(folder / "COMPLETE.json")
    need(complete.get("status") == "COMPLETE", "RUN_NOT_COMPLETE:" + str(folder))
    need(complete.get("actual_call_count") == len(requests), "COMPLETE_CALL_COUNT_MISMATCH:" + str(folder))
    paths = {p.stem: p for p in (folder / "responses").glob("*.json")}
    need(set(paths) == set(requests), "MISSING_OR_UNPLANNED_RESPONSE_RECORDS:" + str(folder))
    if school:
        verify_binding(complete.get("tooling_freeze_binding"), allowed, BATCH / "TOOLING_FREEZE.json")
        bindings = complete.get("response_bindings")
        need(isinstance(bindings, list) and len(bindings) == len(requests), "COMPLETE_RESPONSE_BINDINGS_COUNT")
        need({str(Path(row["path"]).resolve()) for row in bindings} == {str(p.resolve()) for p in paths.values()}, "COMPLETE_RESPONSE_BINDING_SET")
        for row in bindings:
            verify_binding(row, allowed)
        for subfolder in ("attempts", "raw"):
            need({p.stem for p in (folder / subfolder).glob("*.json")} == set(requests), "INCOMPLETE_" + subfolder.upper() + "_SET")
    records = {}
    for rid in requests:
        record = read(paths[rid])
        if school:
            raw_path = folder / "raw" / (rid + ".json")
            verify_binding(record.get("raw_response_binding"), allowed, raw_path)
            raw = read(raw_path)
            response = raw.get("response_json")
            choices = response.get("choices") if isinstance(response, dict) else None
            choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
            message = choice.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            expected = {"raw_output": content, "finish_reason": choice.get("finish_reason"),
                        "returned_model": response.get("model") if isinstance(response, dict) else None,
                        "usage": response.get("usage") if isinstance(response, dict) else None,
                        "transport_status": raw.get("transport_status"), "http_status": raw.get("http_status")}
            for key, value in expected.items():
                need(record.get(key) == value, "RAW_RESPONSE_RECORD_MISMATCH:" + rid + ":" + key)
            attempt = read(folder / "attempts" / (rid + ".json"))
            need(attempt.get("request_id") == rid and attempt.get("http_attempt_number") == 1 and attempt.get("automatic_retry") is False, "ATTEMPT_CONTRACT_MISMATCH:" + rid)
            verify_binding(attempt.get("payload_binding"), allowed, BATCH / "prepared/payloads" / (rid + ".json"))
        records[rid] = verify_record(record, requests[rid], questions[requests[rid]["candidate_id"]], gold, scorer, school)
    if school:
        valid_count = sum(r["_valid"] for r in records.values())
        need(complete.get("valid_call_count") == valid_count and complete.get("invalid_call_count") == len(records) - valid_count, "COMPLETE_VALIDITY_COUNTS_MISMATCH")
        need(complete.get("training_or_pool_merge") is False, "COMPLETE_TRAINING_OR_POOL_MERGE_FLAG")
    digest = hashlib.sha256("".join(rid + "\t" + sha(paths[rid]) + "\n" for rid in sorted(paths)).encode()).hexdigest()
    return records, {"complete_binding": binding(folder / "COMPLETE.json"), "response_count": len(paths), "response_set_sha256": digest}


def pct(result):
    return "—" if result["rate"] is None else f'{100 * result["rate"]:.2f}% ({result["numerator"]}/{result["denominator"]})'


def num(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(summary):
    s = summary["models"]["school_api"]
    lines = ["# 学校 API 同题 Solver 评估", "", "本报告仅在 684 个冻结请求全部留有终态记录、COMPLETE 及证据绑定通过核验后生成。完整调用数与有效作答数分别报告；无效调用不生成部分分数。", "",
        "学校 API 与本地 M0、M_clean_policy 使用相同题目、公开消息内容、选项/条目排列、slot、Gold 和原生严格评分。学校 API 为 max_tokens=8192、temperature=0.8、top_p=0.9，无有限选择解码器，也未显式关闭思考；本地为 max_tokens=32、有限选择解码、enable_thinking=false。服务端模板、分词器与实际思考行为可能不同。这是外部 Solver 的描述性比较，不是严格控制解码条件的因果比较。原 seed 已发送，服务端实际采用与否未验证。", "",
        "同一题的排列、rollout 以及同一 root 的共享事实组不是独立样本；调用数不能直接视为独立样本量，也不据此推断统计显著性。", "",
        "未修复或抽取返回答案；含框外说明等不满足原生严格解析的响应按无效计。valid 分母表示有效作答中的准确率；planned 分母表示计划请求中的正确作答成功率，无效调用未正确完成。部分分数只对有效作答取均值，且可能为负。未修改题目、未据分数筛选训练池，也未执行训练或合池。", "",
        f'计划请求 {summary["counts"]["planned_call_count"]}；完整终态记录 {summary["counts"]["complete_call_count"]}；HTTP_OK {summary["counts"]["transport_success_count"]}；有效 {summary["counts"]["evaluation_valid_call_count"]}；无效 {summary["counts"]["evaluation_invalid_call_count"]}。', "",
        "| 分组 | 题数 | 完整/计划调用 | 有效调用 | 学校 exact/valid | 学校 exact/planned | 学校部分分数（仅有效） | M0 exact/valid | 训练后 exact/valid |",
        "|---|---:|---:|---:|---|---|---:|---|---|"]
    for group in GROUPS:
        r = s["groups"][group]
        p = r["partial_score_on_valid_only"]["mean"]
        lines.append(f'| {LABELS[group]} | {r["question_count"]} | {r["complete_call_count"]}/{r["planned_call_count"]} | {r["evaluation_valid_call_count"]} | {pct(r["exact_match_on_valid"])} | {pct(r["exact_success_on_planned"])} | {num(p["value"] if p else None)} | {pct(summary["models"]["M0"]["groups"][group]["exact_match_on_valid"])} | {pct(summary["models"]["M_clean_policy"]["groups"][group]["exact_match_on_valid"])} |')
    lines += ["", "逐 root 的学校结果：", "", "| root | 题数 | 完整/计划 | 有效 | exact/valid | exact/planned |", "|---|---:|---:|---:|---|---|"]
    for root, groups in s["per_root"].items():
        r = groups["all"]
        lines.append(f'| {root} | {r["question_count"]} | {r["complete_call_count"]}/{r["planned_call_count"]} | {r["evaluation_valid_call_count"]} | {pct(r["exact_match_on_valid"])} | {pct(r["exact_success_on_planned"])} |')
    lines += ["", "以下比较对每个本地模型分别取 paired-valid 请求交集，双方使用完全相同的交集分母。", "", "| 本地模型 | 分组 | paired-valid/计划 | 学校 exact | 本地 exact | 学校−本地（百分点） | 学校−本地部分均分 | 学校独对/本地独对 |", "|---|---|---:|---|---|---:|---:|---:|"]
    for model, comp in summary["paired_comparisons"].items():
        for group in GROUPS:
            r = comp["groups"][group]
            delta = r["school_minus_local_mean_partial"]
            lines.append(f'| {model} | {LABELS[group]} | {r["paired_valid_call_count"]}/{r["planned_call_count"]} | {pct(r["school_exact_match"])} | {pct(r["local_exact_match"])} | {num(r["school_minus_local_exact_percentage_points"], 2)} | {num(delta["value"] if delta else None)} | {r["school_only_correct"]}/{r["local_only_correct"]} |')
    lines += ["", "学校多选行为（仅有效响应；排序题的逆序对不计为过选）：", "", "| 分组 | 有效多选调用 | 有误选 | 有漏选 | 全选 | 平均选择数 | FP/非Gold机会 | FN/Gold机会 |", "|---|---:|---|---|---|---:|---|---|"]
    for group in ("ms_gold2_5", "ms_gold6_control", "gold2", "gold3", "gold4", "gold5"):
        r = s["groups"][group]["multi_select_valid_only"]
        lines.append(f'| {LABELS[group]} | {r["valid_call_count"]} | {pct(r["any_false_selection"])} | {pct(r["any_omission"])} | {pct(r["all_options_selected"])} | {num(r["mean_selected_count"], 2)} | {pct(r["false_selection_per_nongold_opportunity"])} | {pct(r["omission_per_gold_opportunity"])} |')
    lines += ["", "Usage 仅汇总接口实际返回的数值；缺失值不补零，以下累计排除独立连通性 probe。", "", "| 字段 | 已报告累计 | 有数值记录 | 缺失记录 | 非法值记录 |", "|---|---:|---:|---:|---:|"]
    for field, value in summary["school_api_accounting"]["usage"]["fields"].items():
        lines.append(f'| {field} | {value["reported_sum"] if value["reported_sum"] is not None else "—"} | {value["reported_record_count"]} | {value["missing_record_count"]} | {value["invalid_value_record_count"]} |')
    accounting = summary["school_api_accounting"]
    lines += ["", "返回模型计数：`" + json.dumps(accounting["returned_models"], ensure_ascii=False) + "`。",
        "", "finish_reason 计数：`" + json.dumps(accounting["finish_reasons"], ensure_ascii=False) + "`。",
        "", "无效原因计数（同一调用可有多项）：`" + json.dumps(s["groups"]["all"]["invalid_reasons"], ensure_ascii=False) + "`。",
        "", "JSON 同时提供各模型在每个 root 内的全部分组、逐 root 的 paired-valid 比较、HTTP 状态/尝试数、原生评分状态、完整证据文件绑定以及请求交集列表。有效样本覆盖不足时，应结合 invalid 原因理解准确率；不能将有效子集表现直接解释为全部计划请求的表现。", ""]
    return "\n".join(lines)


def build_summary(local_batch):
    local_batch = local_batch.resolve()
    need(local_batch != BATCH and local_batch.name == LOCAL.name, "LOCAL_REFERENCE_BATCH_MUST_BE_V122")
    complete_path = BATCH / "run/COMPLETE.json"
    need(complete_path.is_file(), "INCOMPLETE_RUN_NO_FORMAL_REPORT:run/COMPLETE.json missing")
    allowed = (BATCH, local_batch, SCORING.parent.resolve())
    frozen = read(BATCH / "TOOLING_FREEZE.json")
    need(frozen.get("status") == "FROZEN_BEFORE_ANY_API_CALL", "TOOLING_NOT_FROZEN")
    frozen_rows = frozen.get("files", [])
    frozen_paths = {Path(row["path"]).resolve() for row in frozen_rows}
    required = {Path(__file__).resolve(), BATCH / "tools/evaluate_tju.py", BATCH / "PROTOCOL.json", SCORING,
                *(p.resolve() for p in (BATCH / "prepared").rglob("*") if p.is_file())}
    need(required <= frozen_paths, "TOOLING_FREEZE_MISSING_REQUIRED_BINDINGS")
    need(len(frozen_paths) == len(frozen_rows), "DUPLICATE_TOOLING_BINDINGS")
    for row in frozen_rows:
        verify_binding(row, allowed)
    verify_binding(frozen.get("reference_manifest_binding"), allowed, local_batch / "COMPLETE_MANIFEST.json")
    sealed = read(local_batch / "COMPLETE_MANIFEST.json")
    sealed_rows = sealed.get("files", [])
    need(bool(sealed_rows), "SEALED_V122_FILE_BINDINGS_MISSING")
    for row in sealed_rows:
        verify_binding(row, allowed)
    sealed_paths = {Path(row["path"]).resolve() for row in sealed_rows} | {local_batch / "COMPLETE_MANIFEST.json"}
    need({p.resolve() for p in local_batch.rglob("*") if p.is_file()} == sealed_paths, "SEALED_V122_FILE_SET_CHANGED")
    manifest = read(BATCH / "prepared/MANIFEST.json")
    for row in manifest.get("input_bindings", []):
        verify_binding(row, allowed)
    protocol = read(BATCH / "PROTOCOL.json")
    sampling = protocol.get("sampling", {})
    need(sampling.get("max_tokens") == 8192 and sampling.get("temperature") == 0.8 and sampling.get("top_p") == 0.9, "SCHOOL_SAMPLING_PROTOCOL_CHANGED")
    need(protocol.get("training_or_pool_merge") is False, "PROTOCOL_TRAINING_OR_POOL_MERGE")
    questions_document = read(BATCH / "prepared/QUESTIONS.json")
    questions_list = questions_document["candidates"]
    questions = {q["candidate_id"]: q for q in questions_list}
    gold = read(BATCH / "prepared/GOLD.json")
    requests_list = manifest["requests"]
    requests = {r["request_id"]: r for r in requests_list}
    need(len(questions) == len(questions_list) == 42 and len(requests) == len(requests_list) == 684, "FROZEN_SCOPE_MUST_BE_42_QUESTIONS_684_CALLS")
    need(manifest.get("request_count") == 684 and frozen.get("planned_evaluation_calls") == 684, "PLANNED_CALL_COUNT_MISMATCH")
    need(set(questions) == set(gold) == {r["candidate_id"] for r in requests_list}, "QUESTION_GOLD_REQUEST_SET_MISMATCH")
    for request in requests_list:
        question = questions[request["candidate_id"]]
        need(request["root_id"] == question["root_id"] and request["question_type"] == question["question_type"], "REQUEST_QUESTION_IDENTITY_MISMATCH")
        need(request["mode"] == "full_context", "UNEXPECTED_EVALUATION_MODE")
    spec = importlib.util.spec_from_file_location("v123_summary_native_scoring", SCORING)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    records, integrity, local_manifests = {}, {}, {}
    records["school_api"], integrity["school_api"] = load_records(BATCH / "run", requests, questions, gold, scorer, True, allowed)
    for name, folder in (("M0", "base"), ("M_clean_policy", "trained")):
        run = local_batch / "evaluations/paired_v1/run" / folder
        lm = read(run / "manifest.json")
        need(lm["requests"] == requests_list, "LOCAL_REQUESTS_DIFFER:" + name)
        need(read(run / "questions.json") == questions_document and read(run / "gold.json") == gold, "LOCAL_QUESTIONS_OR_GOLD_DIFFER:" + name)
        decoder = lm.get("decoder_contract", {})
        need(lm.get("solver_model") == name and decoder.get("max_tokens") == 32 and decoder.get("enable_thinking") is False
             and decoder.get("structured_output_version") == "finite_choice_set_v1"
             and decoder.get("temperature") == 0.8 and decoder.get("top_p") == 0.9, "LOCAL_DECODER_OR_MODEL_CONTRACT_CHANGED:" + name)
        local_manifests[name] = lm
        records[name], integrity[name] = load_records(run, requests, questions, gold, scorer, False, allowed)
    group_ids = {g: [rid for rid, row in requests.items() if g in group_for(questions[row["candidate_id"]], gold)] for g in GROUPS}
    roots = sorted({q["root_id"] for q in questions_list})
    need(len(roots) == 6, "FROZEN_SCOPE_MUST_BE_SIX_ROOTS")
    need({g: len(ids) for g, ids in group_ids.items()} == {"all": 684, "ms_gold2_5": 432, "ms_gold6_control": 108,
         "ordering": 144, "gold2": 108, "gold3": 108, "gold4": 108, "gold5": 108}, "FROZEN_GROUP_CALL_COUNTS_CHANGED")
    root_ids = {root: {g: [rid for rid in group_ids[g] if requests[rid]["root_id"] == root] for g in GROUPS} for root in roots}
    models = {name: {"groups": {g: aggregate(ids, requests, questions, gold, rec) for g, ids in group_ids.items()},
                     "per_root": {root: {g: aggregate(ids, requests, questions, gold, rec) for g, ids in groups.items()} for root, groups in root_ids.items()}}
              for name, rec in records.items()}
    comparisons = {name: {"groups": {g: paired(ids, records["school_api"], records[name]) for g, ids in group_ids.items()},
                         "per_root": {root: {g: paired(ids, records["school_api"], records[name]) for g, ids in groups.items()} for root, groups in root_ids.items()}}
                   for name in ("M0", "M_clean_policy")}
    school_rows = list(records["school_api"].values())
    counts = dict(models["school_api"]["groups"]["all"])
    counts = {k: counts[k] for k in ("question_count", "planned_call_count", "complete_call_count", "evaluation_valid_call_count", "evaluation_invalid_call_count")}
    counts["transport_success_count"] = sum(r.get("transport_status") == "HTTP_OK" for r in school_rows)
    return {"schema_version": "school_solver_summary_v1", "status": "COMPLETE_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "batch_dir": str(BATCH),
        "counts": counts, "roots": roots, "group_order": list(GROUPS), "models": models, "paired_comparisons": comparisons,
        "school_api_accounting": {"transport_statuses": count_labels(r.get("transport_status") for r in school_rows),
            "finish_reasons": count_labels(r.get("finish_reason") for r in school_rows),
            "returned_models": count_labels(r.get("returned_model") for r in school_rows),
            "http_attempt_count": sum(r["http_attempt_count"] for r in school_rows), "probe_calls_excluded": True,
            "usage": usage_summary(school_rows)},
        "comparison_contract": {"same_questions_gold_permutations_slots_seeds_and_native_scoring": True,
            "same_public_system_and_user_message_contents": True, "strict_same_decoding_causal_comparison": False,
            "seed_sent_but_provider_adoption_unverified": True, "school_protocol": protocol,
            "permutations_rollouts_and_within_root_shared_fact_groups_are_independent_samples": False,
            "local_decoder_contracts": {name: m.get("decoder_contract") for name, m in local_manifests.items()},
            "partial_scores_only_on_evaluation_valid_responses": True,
            "planned_exact_denominator": "All planned requests; invalid calls supply no exact success, never an invented partial score.",
            "paired_comparison_denominator": "Separate school-valid intersection with each local-model-valid request set, within each group/root.",
            "answer_repair": False, "question_changes": False, "formal_training_pool_filtering": False, "training_or_pool_merge": False},
        "integrity": {"all_native_recorded_scores_recomputed_and_equal": True, "tooling_freeze_bindings_verified": True,
            "all_sealed_v122_files_and_file_set_verified": True,
            "reference_input_bindings_verified": True, "complete_response_and_raw_bindings_verified": True,
            "same_question_gold_and_request_objects_verified_against_both_locals": True,
            "evidence": [binding(p) for p in (BATCH / "TOOLING_FREEZE.json", BATCH / "PROTOCOL.json", BATCH / "prepared/MANIFEST.json", BATCH / "prepared/QUESTIONS.json", BATCH / "prepared/GOLD.json", SCORING)],
            "runs": integrity}}


def main(argv=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-batch-dir", type=Path, default=LOCAL, help="Read-only sealed v122 reference batch.")
    parser.add_argument("--validate-only", action="store_true", help="Validate complete inputs without writing a report.")
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.local_batch_dir)
        if not args.validate_only:
            delivery = BATCH / "delivery"
            delivery.mkdir(exist_ok=True)
            outputs = {delivery / "SCHOOL_SOLVER_SUMMARY_V1.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                       delivery / "SCHOOL_SOLVER_REPORT_ZH_V1.md": render_report(summary)}
            need(not any(target.exists() for target in outputs), "FORMAL_REPORT_ALREADY_EXISTS_NO_OVERWRITE")
            for target, content in outputs.items():
                with target.open("x", encoding="utf-8") as stream:
                    stream.write(content)
        print(json.dumps({"status": "VALIDATED" if args.validate_only else "REPORT_COMPLETE", **summary["counts"]}, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError) as error:
        print(json.dumps({"status": "REFUSED_NO_FORMAL_REPORT", "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
