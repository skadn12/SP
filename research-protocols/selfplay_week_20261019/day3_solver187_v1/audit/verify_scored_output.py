"""Independently recompute committed scoring mathematics from raw output.

Gold is taken from the completed SCORES artifact, after terminal RAW freeze;
this script never opens the original private answer sidecar.
"""
from __future__ import annotations
import argparse
import collections
import itertools
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "runtime"))
import common as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=HERE.parent / "SUMMARY.json")
    args = parser.parse_args()
    run = args.run.resolve()
    c.environment()
    if args.output.exists():
        raise RuntimeError("NO_OVERWRITE_AUDIT")
    checks = collections.Counter()
    def check(condition, name):
        if not condition:
            raise RuntimeError(name)
        checks[name] += 1
    check((run / "RAW_RESPONSE_FREEZE.json").is_file() and
          (run / "SCORES.json").is_file(), "REQUIRE_TERMINAL_FROZEN_SCORES")
    freeze = c.read(run / "RAW_RESPONSE_FREEZE.json")
    check(freeze["status"] == "COMMITTED", "RAW_COMMITTED")
    c.verify_bindings(freeze["files"] + [freeze["manifest_binding"]])
    manifest = c.read(run / "manifest.json")
    scores = c.read(run / "SCORES.json")
    maps = c.read(run / "DISPLAY_MAPPINGS.json")
    check(scores["status"] == "COMPLETE", "COMPLETE_EVALUATION")
    check(c.timestamp(freeze["created_at_utc"]) <= c.timestamp(scores["gold_first_read_at_utc"]),
          "GOLD_SCORING_AFTER_RAW_FREEZE")
    scored = {record["request_id"]: record for record in scores["requests"]}
    check(len(scored) == len(manifest["requests"]) == 2688, "EXACT_2688_SCORED_REQUESTS")
    per_question = collections.defaultdict(list)
    exact_total = valid_total = fp_total = fn_total = correct_pairs_total = total_pairs = 0
    raw_prompt_tokens = raw_completion_tokens = 0
    for row in manifest["requests"]:
        rid, qid, typ = row["request_id"], row["candidate_id"], row["question_type"]
        record = scored[rid]
        raw_path = run / "responses" / (rid + ".json")
        raw = c.read(raw_path)
        check(c.timestamp(raw["response_committed_at_utc"]) <= c.timestamp(freeze["created_at_utc"]),
              "RAW_COMMITTED_BEFORE_FREEZE")
        variant = maps["requests"][rid]
        canonical = maps["public_to_canonical"][qid]
        reverse = {value: label for label, value in canonical.items()}
        gold = record["gold_canonical"]
        gold_content = [canonical[label] for label in gold]
        check(record["gold_content"] == gold_content, "CANONICAL_GOLD_BINDING")
        field = "options" if typ == "multi_select" else "items"
        labels = list(variant[field])
        sequences = (itertools.chain.from_iterable(itertools.combinations(labels, count)
                     for count in range(1, 7)) if typ == "multi_select" else itertools.permutations(labels))
        legal = {"<box>" + ",".join(sequence) + "</box>" for sequence in sequences}
        raw_text = raw.get("raw_output")
        valid = (raw["status"] == "RETURNED" and not raw.get("output_validation_errors") and
                 raw.get("metadata", {}).get("finish_reason") in ("stop", "eos") and raw_text in legal)
        if valid:
            display_labels = raw_text[5:-6].split(",")
            prediction_content = [variant["display_to_content"][label] for label in display_labels]
            if typ == "multi_select":
                prediction_content.sort()
            prediction = [reverse[value] for value in prediction_content]
            exact = (set(prediction) == set(gold) if typ == "multi_select" else prediction == gold)
        else:
            prediction_content, prediction, exact = None, None, False
        check(record["valid"] == valid and record["exact"] == exact,
              "INDEPENDENT_VALIDITY_AND_EXACT")
        check(record["canonical_prediction"] == prediction and
              record["prediction_content"] == prediction_content, "INDEPENDENT_CANONICAL_PREDICTION")
        if typ == "multi_select":
            outcomes = []
            for item in record["items"]:
                label = item["canonical_label"]
                truth = label in gold
                selected = label in prediction if valid else None
                outcome = ("INVALID" if not valid else "TP" if truth and selected else
                           "FN" if truth else "FP" if selected else "TN")
                check(item["gold_selected"] == truth and item["predicted_selected"] == selected and
                      item["outcome"] == outcome and
                      item["fp"] == ((outcome == "FP") if valid else None) and
                      item["fn"] == ((outcome == "FN") if valid else None), "MC_ITEM_CONFUSION_RECOMPUTATION")
                outcomes.append(outcome)
            fp, fn = outcomes.count("FP"), outcomes.count("FN")
            tp = outcomes.count("TP")
            check(record["metrics"]["fp"] == (fp if valid else None) and
                  record["metrics"]["fn"] == (fn if valid else None) and
                  record["metrics"]["tp"] == (tp if valid else None), "MC_AGGREGATE_CONFUSION")
            expected_jaccard = len(set(prediction) & set(gold)) / len(set(prediction) | set(gold)) if valid else 0
            check(record["metrics"]["jaccard"] == expected_jaccard, "MC_JACCARD")
            fp_total += fp
            fn_total += fn
        else:
            rank = {label: index for index, label in enumerate(prediction or [])}
            pairs = list(itertools.combinations(gold, 2))
            pair_correct = sum(bool(valid and rank[left] < rank[right]) for left, right in pairs)
            check(record["metrics"]["total_pairs"] == 15 and
                  record["metrics"]["correct_pairs"] == pair_correct and
                  record["metrics"]["pair_accuracy"] == pair_correct / 15, "ORDERING_ALL15_PAIR_MATH")
            check(len(record["pairs"]) == 15, "ORDERING_ALL15_PAIRS_PRESENT")
            for pair, (left, right) in zip(record["pairs"], pairs):
                check(pair["before_canonical_label"] == left and pair["after_canonical_label"] == right and
                      pair["correct"] == bool(valid and rank[left] < rank[right]), "ORDERING_EACH_PAIR")
            correct_pairs_total += pair_correct
            total_pairs += 15
        exact_total += exact
        valid_total += valid
        per_question[qid].append(record)
        raw_prompt_tokens += raw.get("metadata", {}).get("prompt_token_count", 0)
        raw_completion_tokens += raw.get("metadata", {}).get("completion_token_count", 0)
    check(scores["exact_calls"] == exact_total and scores["valid_calls"] == valid_total,
          "OVERALL_EXACT_AND_VALID_TOTALS")
    check(len(scores["questions"]) == len(per_question) == 187 and
          scores["denominator_roots"] == 20, "QUESTION_AND_ROOT_DENOMINATORS")
    root_rates = collections.defaultdict(list)
    for question in scores["questions"]:
        qrecords = per_question[question["candidate_id"]]
        correct, n = sum(record["exact"] for record in qrecords), len(qrecords)
        check(question["exact_calls"] == correct and question["denominator_calls"] == n and
              question["exact_rate"] == correct / n, "PER_QUESTION_EXACT_RATE")
        stratum = "incorrect" if correct == 0 else "stably_correct" if correct == n else "occasionally_correct"
        check(question["stratum"] == stratum, "PER_QUESTION_STRATUM")
        preds = [json.dumps(record["prediction_content"]) for record in qrecords if record["valid"]]
        consistency = max(collections.Counter(preds).values()) / n if preds else 0
        check(question["display_permutation_consistency"] == consistency,
              "PER_QUESTION_PREDICTION_CONSISTENCY")
        root_rates[question["root_id"]].append(correct / n)
    check(scores["question_equal_exact"] == statistics.mean(q["exact_rate"] for q in scores["questions"]),
          "QUESTION_EQUAL_MACRO")
    check(scores["root_equal_exact"] == statistics.mean(statistics.mean(values) for values in root_rates.values()),
          "ROOT_EQUAL_MACRO")
    for typ in ("multi_select", "ordering"):
        sub = [q for q in scores["questions"] if q["question_type"] == typ]
        groups = collections.defaultdict(list)
        for question in sub:
            groups[question["root_id"]].append(question["exact_rate"])
        expected = {"questions": len(sub), "calls": sum(q["denominator_calls"] for q in sub),
                    "question_equal_exact": statistics.mean(q["exact_rate"] for q in sub),
                    "root_equal_exact": statistics.mean(statistics.mean(values) for values in groups.values())}
        check(scores["by_type"][typ] == expected, "BY_TYPE_MACRO_AND_COUNTS")
    check(scores["cost"]["known_prompt_tokens"] == raw_prompt_tokens and
          scores["cost"]["known_completion_tokens"] == raw_completion_tokens,
          "RAW_TOKEN_USAGE_TOTALS")
    check(scores["production_denominator"] == 240 and scores["production_unadmitted_slots"] == 53 and
          scores["production_admitted_numerator"] == 187, "PRODUCTION_DENOMINATOR_RETAINED")
    summary = c.read(args.summary.resolve())
    protocol = c.read(run / "PROTOCOL.json")
    slot_index = {slot["candidate_id"]: slot for slot in protocol["slots"]}
    metadata = {question["candidate_id"]: question for question in
                c.read(HERE.parent / "CANDIDATE_METADATA.json")["questions"]}
    summary_q = {question["candidate_id"]: question for question in summary["questions"]}
    check(set(summary_q) == set(per_question) and len(summary["questions"]) == 187,
          "SUMMARY_EXACT187_QUESTION_SET")
    check(any(Path(row["path"]).resolve() == run / "SCORES.json" and
              row["sha256"] == c.sha(run / "SCORES.json") for row in summary["input_bindings"]),
          "SUMMARY_BOUND_TO_ACTUAL_DIRECT_V2_SCORES")
    for qid, question in summary_q.items():
        records = per_question[qid]
        check(question["arm"] == metadata[qid]["arm"] == qid.split("-", 1)[0] and
              slot_index[qid]["candidate_id"] == qid and
              question["question_type"] == records[0]["question_type"] and
              question["actual_gold_count"] == len(records[0]["gold_canonical"]),
              "SUMMARY_ARM_TYPE_GOLD_JOIN")

    def check_group(group, selected_qids):
        qrecords = [record for qid in selected_qids for record in per_question[qid]]
        q_rates = {qid: sum(record["exact"] for record in per_question[qid]) / len(per_question[qid])
                   for qid in selected_qids}
        root_groups = collections.defaultdict(list)
        for qid in selected_qids:
            root_groups[per_question[qid][0]["root_id"]].append(q_rates[qid])
        check(group["questions"] == len(selected_qids) and group["roots"] == len(root_groups) and
              group["calls"] == len(qrecords) and group["exact_calls"] == sum(record["exact"] for record in qrecords) and
              group["valid_calls"] == sum(record["valid"] for record in qrecords) and
              group["invalid_calls"] == sum(not record["valid"] for record in qrecords),
              "SUMMARY_GROUP_COUNTS_RECOMPUTED")
        check(group["question_equal_exact"] == statistics.mean(q_rates.values()) and
              group["root_equal_exact"] == statistics.mean(statistics.mean(values) for values in root_groups.values()) and
              group["call_weighted_exact"] == sum(record["exact"] for record in qrecords) / len(qrecords),
              "SUMMARY_GROUP_EXACT_WEIGHTS_RECOMPUTED")
        strata = collections.Counter("incorrect" if rate == 0 else "stably_correct" if rate == 1 else
                                     "occasionally_correct" for rate in q_rates.values())
        check(group["question_strata"] == dict(strata), "SUMMARY_GROUP_STRATA_RECOMPUTED")
        mc = [record for record in qrecords if record["question_type"] == "multi_select"]
        if mc:
            valid_mc = [record for record in mc if record["valid"]]
            ms = group["multi_select"]
            check(ms["known_fp_selections"] == sum(record["metrics"]["fp"] for record in valid_mc) and
                  ms["known_fn_omissions"] == sum(record["metrics"]["fn"] for record in valid_mc) and
                  ms["overselection_calls"] == sum(record["metrics"]["selected_count"] > record["metrics"]["gold_count"]
                                                      for record in valid_mc), "SUMMARY_GROUP_MC_FP_FN_OVERSELECT")
            check(ms["jaccard_mean_fixed_denominator"] == statistics.mean(record["metrics"]["jaccard"] for record in mc),
                  "SUMMARY_GROUP_MC_JACCARD")
        ordering = [record for record in qrecords if record["question_type"] == "ordering"]
        if ordering:
            correct_pairs = sum(record["metrics"]["correct_pairs"] for record in ordering)
            os = group["ordering"]
            check(os["correct_pairs"] == correct_pairs and os["total_pairs_fixed_denominator"] == len(ordering) * 15 and
                  os["pair_accuracy_fixed_denominator"] == correct_pairs / (len(ordering) * 15),
                  "SUMMARY_GROUP_ORDERING_PAIRS")

    check_group(summary["overall"], list(summary_q))
    for key, key_func in {
        "by_arm": lambda q: q["arm"],
        "by_type": lambda q: q["question_type"],
        "by_arm_type": lambda q: q["arm"] + "|" + q["question_type"],
        "by_type_gold_count": lambda q: q["question_type"] + "|gold" + str(q["actual_gold_count"]),
        "by_arm_type_gold_count": lambda q: q["arm"] + "|" + q["question_type"] + "|gold" + str(q["actual_gold_count"]),
    }.items():
        groups = collections.defaultdict(list)
        for qid, question in summary_q.items():
            groups[key_func(question)].append(qid)
        check(set(summary[key]) == set(groups), "SUMMARY_EXACT_GROUP_SET")
        for group_key, selected_qids in groups.items():
            check_group(summary[key][group_key], selected_qids)
    report = {"status": "PASS", "created_at_utc": c.now(),
              "role": "separate Codex scoring-math check; shared filesystem/model",
              "original_private_answer_sidecar_opened": False,
              "gold_source": "completed SCORES.json after committed terminal raw freeze",
              "new_solver_calls": 0, "requests": 2688, "questions": 187, "roots": 20,
              "valid_calls": valid_total, "exact_calls": exact_total,
              "MC_fp_total": fp_total, "MC_fn_total": fn_total,
              "ordering_correct_pairs": correct_pairs_total, "ordering_total_pairs": total_pairs,
              "known_prompt_tokens": raw_prompt_tokens, "known_completion_tokens": raw_completion_tokens,
              "check_counts": dict(checks), "script_binding": c.bind(Path(__file__).resolve()),
              "summary_group_checks": "overall, arm, type, arm-type, type-Gold and arm-type-Gold",
              "summary_binding": c.bind(args.summary.resolve()),
              "audited_files": [c.bind(run / name) for name in ("SCORES.json", "RAW_RESPONSE_FREEZE.json",
                   "manifest.json", "DISPLAY_MAPPINGS.json")]}
    c.save(args.output.resolve(), report)
    print(json.dumps({key: report[key] for key in ("status", "questions", "requests", "valid_calls",
          "exact_calls", "MC_fp_total", "MC_fn_total", "ordering_correct_pairs", "ordering_total_pairs")}), flush=True)


if __name__ == "__main__":
    main()
