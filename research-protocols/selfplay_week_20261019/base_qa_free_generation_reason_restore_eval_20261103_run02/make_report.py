#!/usr/bin/env python3
"""Report a paired visible-reason rerun without changing questions or old files.

Run only after SCORES.json and RAW_ANSWERS.json have been finalized.  Outputs are
created exclusively: an existing report or CSV is never silently overwritten.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_qa_free_generation_solver_eval_20261103_run02"
GENERATION = HERE.parent / "base_qa_free_generation_20261103_run02"
MODES = ("full_context", "no_context")
MODE_NAMES = {"full_context": "Full", "no_context": "No-context"}
ZERO = {"valid": False, "exact_correct": False, "J_fraction": "0", "J": 0.0,
        "parsed": None, "predicted_option_ids": None, "tp": None, "fp": None,
        "fn": None, "recall": 0.0, "parse_reason": "NO_VALID_FINAL_ANSWER"}
OUTPUTS = ("REPORT.md", "PER_ITEM_COMPARISON.csv", "COMPARISON.json", "ANSWERS_ALL.csv")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(name, rows):
    assert rows, name
    fields = list(rows[0])
    assert all(set(row) == set(fields) for row in rows), name
    with (HERE / name).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def final_score(row):
    """Use the already frozen designated-field parser, never repair its output."""
    answer = row.get("answer_only_score")
    return answer if isinstance(answer, dict) else ZERO


def summarize(rows):
    assert rows
    count = len(rows)
    correct = sum(bool(row["exact_correct"]) for row in rows)
    valid = sum(bool(row["valid"]) for row in rows)
    mean_j = sum((Fraction(row["J_fraction"]) for row in rows), Fraction()) / count
    assert all(not row["exact_correct"] or row["valid"] for row in rows)
    return {"calls": count, "valid_calls": valid, "invalid_calls": count - valid,
            "valid_rate": valid / count, "exact_correct_calls": correct,
            "exact_accuracy": correct / count,
            "mean_J_fraction": str(mean_j), "mean_J": float(mean_j)}


def compare(old_rows, new_rows):
    assert len(old_rows) == len(new_rows)
    old, new = summarize(old_rows), summarize(new_rows)
    flips = Counter()
    validity_flips = Counter()
    for left, right in zip(old_rows, new_rows):
        flips[(bool(left["exact_correct"]), bool(right["exact_correct"]))] += 1
        validity_flips[(bool(left["valid"]), bool(right["valid"]))] += 1
    exact_delta = new["exact_correct_calls"] - old["exact_correct_calls"]
    assert flips[(False, True)] - flips[(True, False)] == exact_delta
    return {"baseline": old, "rerun": new,
            "accuracy_delta_percentage_points": 100 * (new["exact_accuracy"] - old["exact_accuracy"]),
            "exact_correct_delta": exact_delta,
            "mean_J_delta": new["mean_J"] - old["mean_J"],
            "mean_J_delta_fraction": str(Fraction(new["mean_J_fraction"]) - Fraction(old["mean_J_fraction"])),
            "valid_rate_delta_percentage_points": 100 * (new["valid_rate"] - old["valid_rate"]),
            "paired": {"wrong_to_correct": flips[(False, True)],
                       "correct_to_wrong": flips[(True, False)],
                       "both_correct": flips[(True, True)], "both_wrong": flips[(False, False)],
                       "invalid_to_valid": validity_flips[(False, True)],
                       "valid_to_invalid": validity_flips[(True, False)]}}


def percent(value):
    return f"{100 * value:.2f}%"


def signed(value, places=2):
    return f"{value:+.{places}f}"


def result_cell(value):
    return f"{value['exact_correct_calls']}/{value['calls']}（{percent(value['exact_accuracy'])}）"


def table(header, rows):
    return ["| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |"] + [
                "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |"
                for row in rows]


def main():
    for name in OUTPUTS:
        assert not (HERE / name).exists(), f"NO_OVERWRITE: {name}"
    assert (HERE / "COMPLETED.json").is_file(), "INFERENCE_NOT_COMPLETE"
    items = read(HERE / "ITEMS_PRIVATE.json")
    old_items = read(BASE / "ITEMS_PRIVATE.json")
    assert items == old_items, "QUESTION_OR_GOLD_CHANGED"
    assert read(HERE / "PERMUTATIONS_PRIVATE.json") == read(BASE / "PERMUTATIONS_PRIVATE.json")
    old_requests, new_requests = read(BASE / "REQUESTS.json"), read(HERE / "REQUESTS.json")
    old_scores, new_scores = read(BASE / "SCORES.json"), read(HERE / "SCORES.json")
    old_rows, new_rows = old_scores["requests"], new_scores["requests"]
    old_raw, new_raw = read(BASE / "RAW_ANSWERS.json"), read(HERE / "RAW_ANSWERS.json")
    assert len(old_requests) == len(new_requests) == len(old_rows) == len(new_rows) == len(old_raw) == len(new_raw) == 396
    assert len(items) == 12
    identity_fields = ("request_id", "candidate_id", "root_id", "question_type", "context_mode",
                       "variant_id", "repeat_index", "sampling_seed", "display_to_option_id", "options",
                       "gold_labels", "gold_option_ids", "n", "g", "all_true_control", "turn_count",
                       "source_sha256_bytes")
    for old_req, new_req, old, new, old_output, new_output in zip(
            old_requests, new_requests, old_rows, new_rows, old_raw, new_raw):
        assert all(old_req[key] == new_req[key] for key in identity_fields), old_req["request_id"]
        assert len({row["request_id"] for row in (old_req, new_req, old, new, old_output, new_output)}) == 1
        for key in ("candidate_id", "question_type", "context_mode", "variant_id", "repeat_index", "sampling_seed", "n", "g"):
            assert old_req[key] == old[key] == new[key], (old_req["request_id"], key)
        assert old["raw_prediction"] == old_output["raw_prediction"]
        assert new["raw_prediction"] == new_output["raw_prediction"]
        assert new["protocol_valid"] == new["valid"]
        assert new["final_answer_valid"] == final_score(new)["valid"]
        if new["protocol_valid"]:
            for key in ("valid", "exact_correct", "J_fraction", "predicted_option_ids"):
                assert new[key] == final_score(new)[key], (new["request_id"], key)
    old_config, new_config = read(BASE / "MODEL_CONFIG.json"), read(HERE / "MODEL_CONFIG.json")
    for key in ("model", "temperature", "top_p", "max_model_len", "dtype", "gpu_memory_utilization"):
        assert old_config[key] == new_config[key], key
    assert old_config["thinking"] is False and new_config["native_thinking"] is False
    old_start, new_start = read(BASE / "STARTED.json"), read(HERE / "STARTED.json")

    def group(predicate):
        indices = [index for index, request in enumerate(old_requests) if predicate(request)]
        baseline_rows = [old_rows[index] for index in indices]
        strict_rows = [new_rows[index] for index in indices]
        final_rows = [final_score(new_rows[index]) for index in indices]
        return {"strict_protocol": compare(baseline_rows, strict_rows),
                "final_answer_only": compare(baseline_rows, final_rows)}

    all_comparisons = {mode: group(lambda request, m=mode: request["context_mode"] == m) for mode in MODES}
    by_type = {kind: {mode: group(lambda request, m=mode, k=kind:
                                request["context_mode"] == m and request["question_type"] == k)
                     for mode in MODES} for kind in sorted({item["question_type"] for item in items})}
    by_item = {item["candidate_id"]: {mode: group(lambda request, m=mode, cid=item["candidate_id"]:
                                                   request["context_mode"] == m and request["candidate_id"] == cid)
                                      for mode in MODES} for item in items}
    errors = Counter(error for row in new_rows for error in row["protocol_errors"])
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": "same frozen 12 questions, semantic Gold, source versions, permutations, repeat counts, sampling seeds and base checkpoint",
        "baseline_directory": str(BASE), "rerun_directory": str(HERE),
        "candidate_count": len(items), "paired_request_count": len(old_rows),
        "binding_verification": {"status": "PASS", "request_fields_checked": list(identity_fields),
                                 "questions_and_gold_equal": True, "permutations_equal": True,
                                 "base_model_equal": True, "native_thinking_both_disabled": True},
        "changed_protocol": {"baseline": {key: old_config.get(key) for key in (
                "max_new_tokens", "stop_sequences", "thinking", "prompt_protocol_note")},
            "rerun": {key: new_config.get(key) for key in (
                "max_new_tokens", "stop_sequences", "native_thinking", "visible_reasons", "structured_outputs", "comparison")}},
        "runtime": {"baseline_vllm": old_start.get("vllm_version"), "rerun_vllm": new_start.get("vllm_version"),
                    "same_vllm_version": old_start.get("vllm_version") == new_start.get("vllm_version")},
        "scoring": {"multi_select": "J=TP/g-FP/(n-g)",
                    "ordering": "J=(correct_pairs-wrong_pairs)/C(k,2)",
                    "invalid_policy": "zero in unchanged fixed denominator; no repair or retry",
                    "strict_protocol": "old bare-label validity vs new complete JSON/evidence/reason/final-answer protocol validity",
                    "final_answer_only": "new designated final_answer parsed from complete valid JSON; reasons/citations may fail; no fallback extraction"},
        "overall": all_comparisons, "by_question_type": by_type, "items": by_item,
        "protocol_error_counts": dict(errors),
        "interpretation_limit": "Observed change compares visible-reason prompt, JSON grammar and output-budget protocols together; not an isolated native-thinking causal test and not evidence of improved question quality.",
        "inputs": [{"path": str(directory / name), "sha256": sha(directory / name)}
                   for directory in (BASE, HERE) for name in (
                       "ITEMS_PRIVATE.json", "REQUESTS.json", "SCORES.json", "RAW_ANSWERS.json", "MODEL_CONFIG.json", "STARTED.json")],
    }

    per_item = []
    for item in items:
        for mode in MODES:
            for policy, values in by_item[item["candidate_id"]][mode].items():
                row = {"number": item["number"], "candidate_id": item["candidate_id"],
                       "root_id": item["root_id"], "question_type": item["question_type"],
                       "question": item["question"], "original_display_gold": item["gold_labels"],
                       "context_mode": mode, "scoring_policy": policy}
                for prefix, value in (("baseline", values["baseline"]), ("rerun", values["rerun"])):
                    row.update({prefix + "_" + key: field for key, field in value.items()})
                row.update({key: values[key] for key in ("accuracy_delta_percentage_points", "exact_correct_delta",
                                                       "mean_J_delta", "mean_J_delta_fraction", "valid_rate_delta_percentage_points")})
                row.update(values["paired"])
                per_item.append(row)

    answer_rows = []
    item_numbers = {item["candidate_id"]: item["number"] for item in items}
    for request, old, new in zip(new_requests, old_rows, new_rows):
        final = final_score(new)
        answer_rows.append({
            "number": item_numbers[request["candidate_id"]], "request_id": request["request_id"],
            "candidate_id": request["candidate_id"], "question_type": request["question_type"],
            "context_mode": request["context_mode"], "variant_id": request["variant_id"],
            "repeat_index": request["repeat_index"], "sampling_seed": request["sampling_seed"],
            "display_to_option_id": request["display_to_option_id"], "gold_labels": request["gold_labels"],
            "gold_option_ids": request["gold_option_ids"], "baseline_raw": old["raw_prediction"],
            "rerun_raw": new["raw_prediction"], "rerun_final_answer": new["final_answer"],
            "rerun_protocol_errors": new["protocol_errors"],
            "baseline_valid": old["valid"], "strict_protocol_valid": new["valid"],
            "final_answer_valid": final["valid"], "baseline_exact_correct": old["exact_correct"],
            "strict_exact_correct": new["exact_correct"], "final_only_exact_correct": final["exact_correct"],
            "baseline_J_fraction": old["J_fraction"], "strict_J_fraction": new["J_fraction"],
            "final_only_J_fraction": final["J_fraction"], "baseline_J": old["J"],
            "strict_J": new["J"], "final_only_J": final["J"],
            "baseline_semantic_prediction": old["predicted_option_ids"],
            "rerun_semantic_prediction": final["predicted_option_ids"],
            "baseline_parse_error": old.get("parse_reason"), "final_answer_parse_error": final.get("parse_reason"),
            "strict_wrong_to_correct": not old["exact_correct"] and new["exact_correct"],
            "strict_correct_to_wrong": old["exact_correct"] and not new["exact_correct"],
            "final_wrong_to_correct": not old["exact_correct"] and final["exact_correct"],
            "final_correct_to_wrong": old["exact_correct"] and not final["exact_correct"],
            "baseline_finish_reason": old["finish_reason"], "rerun_finish_reason": new["finish_reason"],
        })

    full = all_comparisons["full_context"]
    direction = lambda v: "上升" if v > 0 else "下降" if v < 0 else "不变"
    lines = ["# 同题配对复测：恢复可见理由后，正确率是否变化", "",
             f"Full 严格协议整题正确率从 {percent(full['strict_protocol']['baseline']['exact_accuracy'])} "
             f"变为 {percent(full['strict_protocol']['rerun']['exact_accuracy'])}，"
             f"{direction(full['strict_protocol']['accuracy_delta_percentage_points'])} "
             f"{abs(full['strict_protocol']['accuracy_delta_percentage_points']):.2f} 个百分点。"
             f"只看合法 final_answer 的口径为 {percent(full['final_answer_only']['rerun']['exact_accuracy'])}，"
             f"相对原来变化 {signed(full['final_answer_only']['accuracy_delta_percentage_points'])} 个百分点。", "",
             "本次对冻结的12题重新作答396次：Full与No-context各198次。10道多选每题6种选项排列、每排列3次；"
             "5项完整排序12种排列各1次；3项完整排序6种排列各1次。所有原始回答均保留，无重试、无答案修复。", "",
             "## 总体结果", ""]
    summary_table = []
    for mode in MODES:
        for policy, label in (("strict_protocol", "严格完整协议"), ("final_answer_only", "仅final_answer")):
            value = all_comparisons[mode][policy]
            old, new = value["baseline"], value["rerun"]
            summary_table.append([MODE_NAMES[mode], label, result_cell(old), result_cell(new),
                                  signed(value["accuracy_delta_percentage_points"]), f"{old['mean_J']:.4f} → {new['mean_J']:.4f}",
                                  f"{old['valid_calls']}/{old['calls']} → {new['valid_calls']}/{new['calls']}",
                                  value["paired"]["wrong_to_correct"], value["paired"]["correct_to_wrong"]])
    lines += table(["条件", "复测评分口径", "原直接答案正确率", "恢复理由后正确率", "变化/百分点", "平均J", "合法回答", "错→对", "对→错"], summary_table)
    lines += ["", "原直接答案的合法性指裸标签格式合法；复测严格协议还要求JSON结构、逐项判断、引用及简短理由满足冻结协议。"
              "仅final_answer口径仍须输出完整、可解析JSON，并从指定字段得到合法答案；不从其他文字补提答案。"
              "两口径均使用全部计划调用作分母，非法作答按0分计算。", "",
              "多选部分分继续使用 J=TP/g−FP/(n−g)；排序使用 J=(正确对数−反转对数)/C(k,2)。"
              "下面按调用汇总；各题调用数不同，整体值不是12题简单平均。", "",
              "## 分题型结果", ""]
    type_rows = []
    for kind, modes in by_type.items():
        for mode in MODES:
            strict, final = modes[mode]["strict_protocol"], modes[mode]["final_answer_only"]
            type_rows.append(["多选" if kind == "multi_select" else "完整排序", MODE_NAMES[mode],
                              result_cell(strict["baseline"]), result_cell(strict["rerun"]), result_cell(final["rerun"]),
                              f"{strict['baseline']['mean_J']:.4f} → {strict['rerun']['mean_J']:.4f} / {final['rerun']['mean_J']:.4f}"])
    lines += table(["题型", "条件", "原正确率", "复测严格正确率", "复测仅final正确率", "平均J：原→严格/仅final"], type_rows)
    for mode in MODES:
        lines += ["", f"## 逐题 {MODE_NAMES[mode]} 对比", ""]
        rows = []
        for item in items:
            strict, final = (by_item[item["candidate_id"]][mode][policy]
                             for policy in ("strict_protocol", "final_answer_only"))
            rows.append([item["number"], "多选" if item["question_type"] == "multi_select" else "完整排序",
                         f"{strict['baseline']['exact_correct_calls']}/{strict['baseline']['calls']}",
                         f"{strict['rerun']['exact_correct_calls']}/{strict['rerun']['calls']}",
                         f"{final['rerun']['exact_correct_calls']}/{final['rerun']['calls']}",
                         signed(strict["accuracy_delta_percentage_points"]), signed(final["accuracy_delta_percentage_points"]),
                         f"{strict['baseline']['mean_J']:.4f}", f"{strict['rerun']['mean_J']:.4f}",
                         f"{final['rerun']['mean_J']:.4f}",
                         f"{strict['paired']['wrong_to_correct']}/{strict['paired']['correct_to_wrong']}",
                         f"{final['paired']['wrong_to_correct']}/{final['paired']['correct_to_wrong']}"])
        lines += table(["题号", "题型", "原正确次数", "严格正确次数", "仅final正确次数", "严格Δ/百分点", "仅finalΔ/百分点",
                        "原J", "严格J", "仅final J", "严格错→对/对→错", "仅final错→对/对→错"], rows)
    lines += ["", "## 原题与Gold", "", "以下Gold对应生成时的原始展示标签，各次打乱后的Gold已按稳定option_id同步映射。"
              f"完整选项见[原题及审核记录]({GENERATION / 'QUESTIONS_WITH_REVIEW.md'})。", ""]
    for item in items:
        kind = "多选" if item["question_type"] == "multi_select" else "完整排序"
        separator = "、" if kind == "多选" else " → "
        lines += [f"{item['number']}. **{kind}｜{item['root_id']}**  ",
                  item["question"] + "  ", f"Gold：{separator.join(item['gold_labels'])}  ",
                  f"candidate_id：`{item['candidate_id']}`", ""]
    lines += ["## 对照条件与解释范围", "",
              f"基础模型均为 `{new_config['model']}`，题干、选项内容、语义Gold、完整来源版本、选项排列、采样seed、温度0.8、top_p=0.9、"
              f"bf16及16384上下文窗口保持一致。新旧实际vLLM版本分别为 `{old_start.get('vllm_version')}` / `{new_start.get('vllm_version')}`，"
              "以两份STARTED.json为准。", "",
              f"改变的是输出协议：原来只输出标签、最多{old_config['max_new_tokens']} token，遇换行等停止；"
              f"现在恢复逐项证据/判断/简短理由及最终boxed答案的JSON格式，使用结构化输出，预算{new_config['max_new_tokens']} token，"
              "并去除原来的短停止序列。两次均关闭native thinking。", "",
              "因此这里测到的是恢复可见理由、JSON约束和输出预算这一组协议变化后的表现；"
              "不能单独归因为native thinking或某种内部推理机制，也不能由正确率变化推出题目质量提升。"
              "相同seed配对用于对应同一道题的同一排列/重复次序，不保证不同Prompt下得到相同随机轨迹。", "",
              "本报告只比较本批12道既有题；没有生成新题、修改Gold、运行训练或扩充对话。", "",
              "## 协议错误及产物", ""]
    if errors:
        lines += table(["冻结校验捕获的错误", "调用次数"], sorted(errors.items(), key=lambda pair: (-pair[1], pair[0])))
    else:
        lines += ["本轮未出现严格协议解析错误。"]
    lines += ["", "- `COMPARISON.json`：整体、题型、逐题及配对变化，含输入哈希。",
              "- `PER_ITEM_COMPARISON.csv`：两种评分口径、两种上下文条件的逐题数值。",
              "- `ANSWERS_ALL.csv`：396个配对调用的原始回答、指定final答案、Gold映射及两种评分。",
              "- `RAW_ANSWERS.json` / `raw/`：复测原始输出；`SCORES.json`：冻结解析与评分结果。", ""]

    write_json("COMPARISON.json", comparison)
    write_csv("PER_ITEM_COMPARISON.csv", per_item)
    write_csv("ANSWERS_ALL.csv", answer_rows)
    with (HERE / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"status": "REPORT_COMPLETE", "outputs": list(OUTPUTS),
                      "full_context": full, "no_context": all_comparisons["no_context"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
