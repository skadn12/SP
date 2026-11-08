"""Analyze this completed run without changing frozen questions or scores."""
import csv
import hashlib
import itertools
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "base_qa_free_generation_20261103_run01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    document = read(HERE / "SCORES.json")
    requests = {r["request_id"]: r for r in read(HERE / "REQUESTS.json")}
    items = read(HERE / "ITEMS_PRIVATE.json")
    candidates = {q["candidate_id"]: q for q in read(SOURCE / "CANDIDATES_FIRST_DRAFT.json")["candidates"]}
    results, csv_rows = {}, []
    questions = ["# 自由出题：完整题目与实际评估", "", "题干、选项和 Gold 沿用冻结首稿；以下答案均映射回首稿 A–F／I1–I6，未改写模型输出。", ""]
    for item in items:
        qid = item["candidate_id"]
        candidate = candidates[qid]
        original = {oid: label for label, oid in item["label_to_option_id"].items()}
        label_rank = {label: i for i, label in enumerate(item["label_to_option_id"])}
        qtype = item["question_type"]
        results[qid] = {}
        field = "options" if qtype == "multi_select" else "items"
        questions.extend([f"**{qid}**", "", candidate["question"], ""])
        for label, option in candidate[field].items():
            questions.append(f"- **{label}**　{option['text']}")
        joiner = "、" if qtype == "multi_select" else " → "
        questions.extend(["", "**Gold：** " + joiner.join(candidate["gold"]), "",
                          f"来源：`{item['root_id']}`，完整 {item['turn_count']} 轮；首稿辅助难度标签：`{candidate['difficulty_auxiliary']}`，不作为实测难度。", "",
                          "| 条件 | 整题正确 | 平均 J | 格式有效 |", "|---|---:|---:|---:|"])
        for mode in ("full_context", "no_context"):
            rows = [r for r in document["requests"] if r["candidate_id"] == qid and r["context_mode"] == mode]
            valid = [r for r in rows if r["valid"]]
            summary = document["items"][qid][mode]
            distribution = Counter()
            by_id = {}
            for row in rows:
                if row["valid"]:
                    labels = [original[oid] for oid in row["predicted_option_ids"]]
                    if qtype == "multi_select":
                        labels.sort(key=label_rank.get)
                    label_answer = ("," if qtype == "multi_select" else ">").join(labels)
                    distribution[label_answer] += 1
                else:
                    label_answer = "INVALID"
                by_id[row["request_id"]] = label_answer
                request = requests[row["request_id"]]
                csv_rows.append({"candidate_id": qid, "context_mode": mode, "variant_id": row["variant_id"],
                                 "repeat_index": row["repeat_index"], "request_id": row["request_id"],
                                 "gold_displayed": ("," if qtype == "multi_select" else ">").join(request["gold_labels"]),
                                 "raw_prediction": row["raw_prediction"], "prediction_original_labels": label_answer,
                                 "exact_correct": row["exact_correct"], "J_fraction": row["J_fraction"],
                                 "J": row["J"], "valid": row["valid"], "tp": row["tp"], "fp": row["fp"], "fn": row["fn"],
                                 "raw_path": "raw/" + row["request_id"] + ".json", "prompt_path": request["prompt_path"]})
            extra = {"summary": summary, "original_label_prediction_counts": dict(distribution.most_common()),
                     "original_label_prediction_by_request": by_id}
            if qtype == "multi_select":
                extra["option_counts"] = {
                    original[oid]: {"gold": oid in item["gold_option_ids"],
                                    "selected": sum(oid in r["predicted_option_ids"] for r in valid),
                                    "missed_if_gold": sum(oid not in r["predicted_option_ids"] for r in valid) if oid in item["gold_option_ids"] else None,
                                    "false_selected_if_distractor": sum(oid in r["predicted_option_ids"] for r in valid) if oid not in item["gold_option_ids"] else None}
                    for oid in item["label_to_option_id"].values()}
                pairs = []
                for variant in {r["variant_id"] for r in rows}:
                    group = [r for r in rows if r["variant_id"] == variant]
                    pairs.extend(set(a["predicted_option_ids"]) == set(b["predicted_option_ids"])
                                 for a, b in itertools.combinations(group, 2) if a["valid"] and b["valid"])
                extra["within_mapping_repeat_pair_agreement"] = {"same": sum(pairs), "pairs": len(pairs), "rate": sum(pairs) / len(pairs)}
            else:
                extra["pair_errors"] = []
                for a, b in itertools.combinations(item["gold_option_ids"], 2):
                    wrong = sum(r["predicted_option_ids"].index(a) > r["predicted_option_ids"].index(b) for r in valid)
                    extra["pair_errors"].append({"gold_before": original[a], "gold_after": original[b], "wrong": wrong, "valid_calls": len(valid)})
                extra["pair_errors"].sort(key=lambda x: -x["wrong"])
                extra["correct_pairs_total"] = sum(r["correct_pairs"] for r in valid)
                extra["wrong_pairs_total"] = sum(r["wrong_pairs"] for r in valid)
                extra["within_mapping_repeat_pair_agreement"] = None
                extra["repeat_note"] = "Historical order sampling has one answer per mapping; no within-mapping repeat consistency estimate."
            results[qid][mode] = extra
            questions.append(f"| {mode} | {summary['exact_correct_calls']}/{len(rows)} ({summary['exact_accuracy']:.1%}) | {summary['mean_J']:.4f} | {len(valid)}/{len(rows)} |")
        questions.extend(["", "原标签答案分布（每行中的次数合计为该条件全部调用）：", "",
                          "| 条件 | 原题标签答案 | 次数 |", "|---|---|---:|"])
        for mode, extra in results[qid].items():
            for prediction, count in extra["original_label_prediction_counts"].items():
                questions.append(f"| {mode} | `{prediction}` | {count} |")
        questions.append("")
    write("DIAGNOSTICS.json", results)
    write("QUESTIONS_WITH_RESULTS.md", "\n".join(questions) + "\n")
    with (HERE / "ANSWERS_ALL.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    checks = []
    bindings = read(HERE / "INPUT_BINDINGS.json")
    for row in bindings["files"]:
        checks.append({"path": row["path"], "expected": row["sha256_bytes"], "actual": digest(Path(row["path"])),
                       "match": digest(Path(row["path"])) == row["sha256_bytes"]})
    for name, expected in read(SOURCE / "MANIFEST_FINAL.json").items():
        path = SOURCE / name
        checks.append({"path": str(path), "expected": expected, "actual": digest(path), "match": digest(path) == expected})
    assert all(row["match"] for row in checks)
    assert read(HERE / "INDEPENDENT_SCORE_AUDIT.json")["status"] == "PASS"
    write("INPUT_INTEGRITY_FINAL.json", {"status": "PASS", "checks": checks,
                                         "frozen_questions_and_gold_unchanged": True,
                                         "prior_version_and_generation_scripts_edited": False,
                                         "candidate_generation_or_rewrite_in_this_run": False})
    print(json.dumps({qid: {mode: {"option_counts": extra.get("option_counts"),
                                             "repeat_consistency": extra["within_mapping_repeat_pair_agreement"],
                                             "pair_errors": extra.get("pair_errors")}
                                for mode, extra in modes.items()}
                      for qid, modes in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
