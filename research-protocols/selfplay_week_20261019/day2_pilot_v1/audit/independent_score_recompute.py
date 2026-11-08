"""Post-score stdlib recomputation. Never imports the runtime or native scorer."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics

PILOT = Path(__file__).resolve().parents[1]


def require(condition, name):
    if not condition:
        raise RuntimeError(name)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify(row):
    path = Path(row["path"])
    require(path.is_file() and sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], "BINDING_MISMATCH:" + str(path))


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def same(actual, expected, name):
    require(actual == expected, name + ":" + repr(actual) + "!=" + repr(expected))


def independently_decode(raw, qtype, display_to_canonical):
    """Strict finite-box vocabulary recreated without importing scorer code."""
    text = raw.get("raw_output")
    valid = raw["status"] == "RETURNED" and not raw.get("output_validation_errors")
    valid = valid and raw.get("metadata", {}).get("finish_reason") in {"stop", "eos"}
    if not isinstance(text, str):
        return False, None
    pattern = r"<box>([A-F](?:,[A-F])*)</box>" if qtype == "multi_select" else r"<box>(I[1-6](?:,I[1-6])*)</box>"
    match = re.fullmatch(pattern, text)
    if not match:
        return False, None
    labels = match.group(1).split(",")
    if qtype == "multi_select":
        valid = valid and labels == sorted(set(labels)) and bool(labels)
    else:
        valid = valid and len(labels) == 6 and set(labels) == {"I1", "I2", "I3", "I4", "I5", "I6"}
    valid = valid and set(labels) <= set(display_to_canonical)
    return (True, [display_to_canonical[label] for label in labels]) if valid else (False, None)


def mean(values):
    return statistics.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PILOT / "runtime/direct")
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PILOT / "audit/INDEPENDENT_SCORE_RECOMPUTATION.json")
    args = parser.parse_args()
    run, private = args.run.resolve(), args.private.resolve()
    require(run == (PILOT / "runtime/direct").resolve(), "WRONG_PUBLIC_RUN")
    require(not private.is_relative_to(run) and not run.is_relative_to(private), "PRIVATE_SIDECAR_NOT_DISJOINT")
    require((run / "SCORES.json").is_file(), "WAIT_FOR_COMPLETED_SCORING_BEFORE_PRIVATE_ACCESS")
    require(not args.output.exists(), "NO_RECOMPUTATION_OVERWRITE")
    scores = read(run / "SCORES.json")
    manifest = read(run / "manifest.json")
    preflight = read(run / "PREFLIGHT_FREEZE.json")
    freeze = read(run / "RAW_RESPONSE_FREEZE.json")
    scoring_started = read(run / "SCORING_STARTED.json")
    for row in preflight["files"]:
        verify(row)
    require(freeze["status"] == "COMMITTED", "RAW_FREEZE_NOT_COMMITTED")
    verify(freeze["manifest_binding"])
    require(Path(freeze["manifest_binding"]["path"]).resolve() == run / "manifest.json", "FREEZE_MANIFEST_PATH")
    rows = manifest["requests"]
    same(len(rows), 360, "EXPECTED_360_REQUESTS")
    expected = {run / "responses" / (row["request_id"] + ".json") for row in rows}
    same({Path(row["path"]).resolve() for row in freeze["files"]}, expected, "RAW_FREEZE_COVERAGE")
    same(len(freeze["files"]), 360, "RAW_FREEZE_COUNT")
    same(set((run / "responses").glob("*.json")), expected, "EXACT_TERMINAL_RAW_SET")
    for row in freeze["files"]:
        verify(row)
    require(stamp(freeze["created_at_utc"]) <= stamp(scoring_started["gold_first_read_at_utc"]), "GOLD_PARSED_BEFORE_RAW_FREEZE")
    same(scores["gold_first_read_at_utc"], scoring_started["gold_first_read_at_utc"], "SCORE_GOLD_START_TIMESTAMP")
    require(stamp(scores["gold_read_completed_at_utc"]) >= stamp(scores["gold_first_read_at_utc"]), "GOLD_COMPLETION_TIMESTAMP")
    require(stamp(scores["scored_at_utc"]) >= stamp(scores["gold_read_completed_at_utc"]), "SCORED_TIMESTAMP")
    verify(scoring_started["raw_response_freeze_binding"])
    same(Path(scoring_started["raw_response_freeze_binding"]["path"]).resolve(), run / "RAW_RESPONSE_FREEZE.json", "SCORING_FREEZE_PATH")
    require(scoring_started["all_terminal_raws_committed_before_gold"] is True, "SCORING_TERMINAL_DECLARATION")
    raw_rows = {}
    for row in rows:
        raw = read(run / "responses" / (row["request_id"] + ".json"))
        for key in ["request_id", "candidate_id", "root_id", "question_type", "branch", "variant_id", "slot", "seed", "prompt_sha256"]:
            same(raw[key], row[key], "RAW_IDENTITY:" + key)
        require(raw["status"] in {"RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"}, "NONTERMINAL_RAW_STATUS")
        require(stamp(raw["response_committed_at_utc"]) <= stamp(freeze["created_at_utc"]), "RAW_AFTER_FREEZE")
        require(raw["automatic_retry"] is False and raw["source_truncation"] is False and raw["gold_read_by_solver"] is False, "RAW_ISOLATION_OR_RETRY")
        same(raw["model_path"], manifest["model"], "RAW_MODEL_PATH")
        if raw["status"] == "RETURNED":
            metadata = raw["metadata"]
            same(metadata["temperature"], .8, "RAW_TEMPERATURE")
            same(metadata["top_p"], .9, "RAW_TOP_P")
            same(metadata["max_tokens"], 32, "RAW_MAX_TOKENS")
            same(metadata["enable_thinking"], False, "RAW_THINKING")
            same(metadata["structured_output_api"], manifest["decoder_contract"]["structured_output_api"], "RAW_STRUCTURED_API")
            same(metadata["prompt_token_count"], len(raw["prompt_token_ids"]), "RAW_PROMPT_TOKEN_COUNT")
            same(metadata["prompt_token_count"], row["prompt_tokens"], "RAW_PLANNED_PROMPT_TOKEN_COUNT")
            same(metadata["completion_token_count"], len(raw["completion_token_ids"]), "RAW_COMPLETION_TOKEN_COUNT")
            require(metadata["completion_token_count"] <= 32, "RAW_OUTPUT_CAP")
            require(stamp(raw["request_submitted_at_utc"]) <= stamp(raw["response_committed_at_utc"]), "RAW_SUBMISSION_TIME")
        raw_rows[row["request_id"]] = raw
    # No sidecar is touched above this line; all raw bindings and timestamps are verified first.
    independent_gold_first_possible_read = datetime.now(timezone.utc).isoformat()
    require(stamp(independent_gold_first_possible_read) >= stamp(scores["scored_at_utc"]), "INDEPENDENT_GOLD_READ_BEFORE_SCORE")
    bindings = read(private / "PRIVATE_BINDINGS.json")
    for key in ["public_manifest_binding", "public_preflight_binding", "private_answer_binding"]:
        verify(bindings[key])
    same(Path(bindings["public_manifest_binding"]["path"]).resolve(), run / "manifest.json", "SIDECAR_MANIFEST_PATH")
    same(Path(bindings["public_preflight_binding"]["path"]).resolve(), run / "PREFLIGHT_FREEZE.json", "SIDECAR_PREFLIGHT_PATH")
    same(Path(bindings["private_answer_binding"]["path"]).resolve(), private / "PRIVATE_QUESTIONS.json", "SIDECAR_GOLD_PATH")
    same(bindings["private_answer_binding"], scoring_started["private_answer_binding"], "GOLD_BINDING_IDENTITY")
    gold_payload = read(private / "PRIVATE_QUESTIONS.json")
    independent_gold_parse_done = datetime.now(timezone.utc).isoformat()
    private_qs = {q["candidate_id"]: q for q in gold_payload["questions"]}
    public = read(run / "PUBLIC_QUESTIONS.json")["questions"]
    questions = {q["candidate_id"]: q for q in public}
    same(set(private_qs), set(questions), "PUBLIC_PRIVATE_CANDIDATE_COVERAGE")
    same(len(questions), 24, "EXPECTED_24_QUESTIONS")
    public_keys = {"candidate_id", "root_id", "question_type", "question", "options", "items", "source_path", "source_sha256"}
    mappings = read(run / "DISPLAY_MAPPINGS.json")
    policy = read(run / "PROTOCOL.json")
    slots = {s["candidate_id"]: s for s in policy["slots"]}
    same(set(slots), set(questions), "ALL_24_SLOTS_ADMITTED")
    for qid, q in questions.items():
        same({key: value for key, value in private_qs[qid].items() if key in public_keys}, q, "PUBLIC_PRIVATE_CONTENT")
        field = "options" if q["question_type"] == "multi_select" else "items"
        canonical = {label: "content_" + hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label, text in q[field].items()}
        same(canonical, mappings["public_to_canonical"][qid], "INDEPENDENT_CANONICAL_CONTENT_HASH")
        gold = private_qs[qid]["gold"]
        require(len(gold) == len(set(gold)) and set(gold) <= set(q[field]), "INVALID_GOLD_LABELS")
        if field == "items":
            same(set(gold), set(q[field]), "INCOMPLETE_ORDERING_GOLD")
    reported_requests = {r["request_id"]: r for r in scores["requests"]}
    same(set(reported_requests), set(raw_rows), "SCORE_REQUEST_COVERAGE")
    reported_questions = {q["candidate_id"]: q for q in scores["questions"]}
    same(set(reported_questions), set(questions), "SCORE_QUESTION_COVERAGE")
    independent = collections.defaultdict(list)
    for row in rows:
        rid, qid = row["request_id"], row["candidate_id"]
        q, raw = questions[qid], raw_rows[rid]
        field = "options" if q["question_type"] == "multi_select" else "items"
        canonical = mappings["public_to_canonical"][qid]
        reverse = {content: label for label, content in canonical.items()}
        variant = mappings["requests"][rid]
        same(variant[field], row["display"][field], "DISPLAY_CONTENT_IDENTITY")
        same(set(variant["display_to_content"]), set(q[field]), "DISPLAY_LABEL_SET")
        same(set(variant["display_to_content"].values()), set(canonical.values()), "DISPLAY_BIJECTION")
        require(len(set(variant["display_to_content"].values())) == 6, "DISPLAY_DUPLICATE_CONTENT")
        display_to_canonical = {label: reverse[content] for label, content in variant["display_to_content"].items()}
        valid, prediction = independently_decode(raw, q["question_type"], display_to_canonical)
        gold = private_qs[qid]["gold"]
        exact = valid and (set(prediction) == set(gold) if field == "options" else prediction == gold)
        record = {"request_id": rid, "valid": valid, "exact": bool(exact), "prediction": prediction}
        reported = reported_requests[rid]
        same(reported["valid"], valid, "REQUEST_VALID")
        same(reported["exact"], bool(exact), "REQUEST_EXACT")
        same(reported["status"], raw["status"], "REQUEST_STATUS")
        same(reported["gold_canonical"], gold, "REQUEST_GOLD_CANONICAL")
        same(reported["gold_first_read_at_utc"], scoring_started["gold_first_read_at_utc"], "REQUEST_GOLD_TIME")
        verify(reported["response_binding"])
        if valid:
            same(set(reported["canonical_prediction"]), set(prediction), "PREDICTION_CONTENT_SET")
            if field == "items":
                same(reported["canonical_prediction"], prediction, "ORDER_PREDICTION_SEQUENCE")
        else:
            same(reported["canonical_prediction"], None, "INVALID_PREDICTION_NULL")
        if field == "options":
            predset = set(prediction or [])
            goldset = set(gold)
            items = []
            for label in q[field]:
                truth = label in goldset
                selected = label in predset if valid else None
                outcome = "INVALID" if not valid else "TP" if truth and selected else "FN" if truth else "FP" if selected else "TN"
                items.append({"canonical_label": label, "gold_selected": truth, "predicted_selected": selected, "outcome": outcome,
                    "fp": outcome == "FP" if valid else None, "fn": outcome == "FN" if valid else None})
            for actual, expected_item in zip(reported["items"], items):
                for key, value in expected_item.items():
                    same(actual[key], value, "MC_REQUEST_ITEM:" + key)
            fp = len(predset-goldset) if valid else None
            fn = len(goldset-predset) if valid else None
            tp = len(predset & goldset) if valid else None
            for key, value in {"fp": fp, "fn": fn, "tp": tp, "gold_count": len(gold), "selected_count": len(prediction) if valid else None,
                    "jaccard": len(predset & goldset)/len(predset | goldset) if valid else 0}.items():
                same(reported["metrics"][key], value, "MC_REQUEST_METRIC:" + key)
            record.update(items=items, fp=fp, fn=fn)
        else:
            rank = {label: index for index, label in enumerate(prediction or [])}
            pairs = [(left, right) for index, left in enumerate(gold) for right in gold[index+1:]]
            same(len(pairs), 15, "ORDER_15_PAIRS")
            expected_pairs = [{"before_canonical_label": left, "after_canonical_label": right, "valid": valid,
                "correct": bool(valid and rank[left] < rank[right]), "predicted_before_rank": rank.get(left), "predicted_after_rank": rank.get(right)} for left, right in pairs]
            same(len(reported["pairs"]), 15, "REPORTED_ORDER_15_PAIRS")
            for actual, expected_pair in zip(reported["pairs"], expected_pairs):
                for key, value in expected_pair.items():
                    same(actual[key], value, "ORDER_REQUEST_PAIR:" + key)
            correct_pairs = sum(pair["correct"] for pair in expected_pairs)
            same(reported["metrics"]["correct_pairs"], correct_pairs, "ORDER_CORRECT_PAIRS")
            same(reported["metrics"]["total_pairs"], 15, "ORDER_TOTAL_PAIRS")
            same(reported["metrics"]["pair_accuracy"], correct_pairs/15, "ORDER_PAIR_ACCURACY")
            record.update(pairs=expected_pairs, correct_pairs=correct_pairs)
        independent[qid].append(record)
    summaries = []
    for qid, q in questions.items():
        records = independent[qid]
        n = len(records)
        same(n, 18 if q["question_type"] == "multi_select" else 12, "QUESTION_DENOMINATOR")
        valid = sum(r["valid"] for r in records)
        exact = sum(r["exact"] for r in records)
        summary = {"candidate_id": qid, "root_id": q["root_id"], "arm": slots[qid]["arm"], "question_type": q["question_type"],
            "denominator_calls": n, "valid_calls": valid, "invalid_calls": n-valid, "exact_calls": exact, "exact_rate": exact/n,
            "stratum": "incorrect" if exact == 0 else "stably_correct" if exact == n else "occasionally_correct"}
        reported = reported_questions[qid]
        for key in ["candidate_id", "root_id", "question_type", "denominator_calls", "valid_calls", "invalid_calls", "exact_calls", "exact_rate", "stratum"]:
            same(reported[key], summary[key], "QUESTION_SUMMARY:" + key)
        canonical = mappings["public_to_canonical"][qid]
        canonical_predictions = [json.dumps(sorted(canonical[label] for label in r["prediction"]) if q["question_type"] == "multi_select" else [canonical[label] for label in r["prediction"]]) for r in records if r["valid"]]
        consistency = max(collections.Counter(canonical_predictions).values()) / n if canonical_predictions else 0
        same(reported["display_permutation_consistency"], consistency, "QUESTION_PERMUTATION_CONSISTENCY")
        if q["question_type"] == "multi_select":
            item_summaries = []
            for index, label in enumerate(q["options"]):
                subset = [record["items"][index] for record in records]
                outcomes = collections.Counter(item["outcome"] for item in subset)
                item_summary = {"canonical_label": label, "content_id": canonical[label], "gold_selected": label in private_qs[qid]["gold"],
                    "denominator_calls": n, "valid_calls": n-outcomes["INVALID"], "invalid_calls": outcomes["INVALID"],
                    **{outcome.lower()+"_calls": outcomes[outcome] for outcome in ["TP", "FP", "FN", "TN"]}}
                same(reported["items"][index], item_summary, "MC_QUESTION_ITEM_SUMMARY")
                item_summaries.append(item_summary)
            summary["items"] = item_summaries
            summary["fp_calls"] = sum(r["fp"] or 0 for r in records)
            summary["fn_calls"] = sum(r["fn"] or 0 for r in records)
            same(reported["fp_calls"], summary["fp_calls"], "MC_QUESTION_FP")
            same(reported["fn_calls"], summary["fn_calls"], "MC_QUESTION_FN")
        else:
            summary["correct_pairs"] = sum(r["correct_pairs"] for r in records)
            summary["total_pairs"] = n*15
            summary["pair_accuracy"] = summary["correct_pairs"] / summary["total_pairs"]
            for key in ["correct_pairs", "total_pairs", "pair_accuracy"]:
                same(reported[key], summary[key], "ORDER_QUESTION_SUMMARY:" + key)
            for index, actual in enumerate(reported["pairs"]):
                correct = sum(record["pairs"][index]["correct"] for record in records)
                same(actual["denominator_calls"], n, "ORDER_PAIR_DENOMINATOR")
                same(actual["valid_calls"], valid, "ORDER_PAIR_VALID_CALLS")
                same(actual["correct_calls"], correct, "ORDER_PAIR_CORRECT_CALLS")
                same(actual["accuracy"], correct/n, "ORDER_PAIR_SUMMARY_ACCURACY")
        summaries.append(summary)
    same(scores["denominator_calls"], 360, "TOP_CALL_DENOMINATOR")
    same(scores["denominator_questions"], 24, "TOP_QUESTION_DENOMINATOR")
    same(scores["denominator_roots"], 6, "TOP_ROOT_DENOMINATOR")
    same(scores["valid_calls"], sum(q["valid_calls"] for q in summaries), "TOP_VALID_CALLS")
    same(scores["exact_calls"], sum(q["exact_calls"] for q in summaries), "TOP_EXACT_CALLS")
    def aggregate(subset):
        roots = collections.defaultdict(list)
        for q in subset:
            roots[q["root_id"]].append(q["exact_rate"])
        result = {"questions": len(subset), "calls": sum(q["denominator_calls"] for q in subset),
            "question_equal_exact": mean([q["exact_rate"] for q in subset]), "root_equal_exact": mean([mean(values) for values in roots.values()])}
        return result
    by_type = {typ: aggregate([q for q in summaries if q["question_type"] == typ]) for typ in ["multi_select", "ordering"]}
    same(scores["by_type"], by_type, "BY_TYPE_AGGREGATION")
    overall = aggregate(summaries)
    same(scores["question_equal_exact"], overall["question_equal_exact"], "TOP_QUESTION_EQUAL")
    same(scores["root_equal_exact"], overall["root_equal_exact"], "TOP_ROOT_EQUAL")
    by_arm_type = {}
    for arm, typ in itertools.product(["A", "B"], ["multi_select", "ordering"]):
        subset = [q for q in summaries if q["arm"] == arm and q["question_type"] == typ]
        result = aggregate(subset)
        result.update(valid_calls=sum(q["valid_calls"] for q in subset), exact_calls=sum(q["exact_calls"] for q in subset))
        if typ == "multi_select":
            result.update(fp_calls=sum(q["fp_calls"] for q in subset), fn_calls=sum(q["fn_calls"] for q in subset))
        else:
            result.update(correct_pairs=sum(q["correct_pairs"] for q in subset), total_pairs=sum(q["total_pairs"] for q in subset))
            result["pair_accuracy"] = result["correct_pairs"] / result["total_pairs"]
        by_arm_type[arm+"|"+typ] = result
    status_counts = dict(collections.Counter(raw["status"] for raw in raw_rows.values()))
    returned = [raw for raw in raw_rows.values() if raw["status"] == "RETURNED"]
    known_prompt = sum(raw["metadata"]["prompt_token_count"] for raw in returned)
    known_completion = sum(raw["metadata"]["completion_token_count"] for raw in returned)
    cost_expected = {"planned_calls": 360, "returned_calls": len(returned),
        "submitted_attempts": sum(raw["status"] not in {"NOT_SUBMITTED", "MISSING"} for raw in raw_rows.values()),
        "not_submitted_calls": status_counts.get("NOT_SUBMITTED", 0), "requests_with_unknown_usage": status_counts.get("BACKEND_ERROR", 0)+status_counts.get("MISSING", 0),
        "known_prompt_tokens": known_prompt, "known_completion_tokens": known_completion, "known_total_tokens": known_prompt+known_completion}
    for key, value in cost_expected.items():
        same(scores["cost"][key], value, "COST:" + key)
    require(scores["cost"]["currency_cost"] is None, "CURRENCY_UNKNOWN_NOT_NULL")
    terminal_path = run / ("COMPLETE.json" if (run / "COMPLETE.json").exists() else "FAILED.json")
    execution = read(terminal_path)
    same(scores["status"], "COMPLETE" if terminal_path.name == "COMPLETE.json" else "FAILED_WITH_FIXED_DENOMINATOR", "EXECUTION_STATUS")
    verify(scores["cost"]["execution_binding"])
    require(scores["automatic_retry"] is False and scores["source_truncation"] is False and scores["training_started"] is False, "SCORE_RETRY_OR_TRAINING")
    report = {"schema_version": "day2_independent_score_recomputation_v1", "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Standalone Python standard library only; does not import or call existing runtime/native scoring functions. Independently parses exact finite-box vocabulary and maps displayed labels through content hashes to canonical labels.",
        "reviewer_scope": "A author, independent of runtime author; A authorship/Gold exposure disclosed. B Gold first accessed by this audit only after actual SCORES existed and every terminal raw binding/timestamp had been verified.",
        "checker_binding": bind(__file__), "scores_binding": bind(run / "SCORES.json"), "manifest_binding": bind(run / "manifest.json"),
        "preflight_binding": bind(run / "PREFLIGHT_FREEZE.json"), "raw_freeze_binding": bind(run / "RAW_RESPONSE_FREEZE.json"),
        "scoring_started_binding": bind(run / "SCORING_STARTED.json"), "private_answer_binding": bind(private / "PRIVATE_QUESTIONS.json"),
        "gold_timing": {"raw_freeze_created_at_utc": freeze["created_at_utc"], "score_gold_first_parse_at_utc": scoring_started["gold_first_read_at_utc"],
            "score_gold_parse_completed_at_utc": scores["gold_read_completed_at_utc"], "independent_gold_first_possible_read_at_utc": independent_gold_first_possible_read,
            "independent_gold_parse_completed_at_utc": independent_gold_parse_done, "raw_freeze_precedes_both_gold_parse_processes": True},
        "denominator_calls": 360, "denominator_questions": 24, "denominator_roots": 6, "terminal_status_counts": status_counts,
        "valid_calls": scores["valid_calls"], "exact_calls": scores["exact_calls"], "question_equal_exact": overall["question_equal_exact"], "root_equal_exact": overall["root_equal_exact"],
        "by_type": by_type, "by_arm_type": by_arm_type, "questions": summaries, "cost_recomputation": cost_expected,
        "all_request_results_match_existing_scores": True, "all_question_and_type_aggregates_match_existing_scores": True,
        "all_ordering_requests_recomputed_with_15_pairs": True, "invalid_policy": "Fixed denominators; invalid exact/pair credit zero and MC unknown item selection/FP/FN null.",
        "GPU_or_model_backend_created": False, "new_solver_calls": 0, "total_api_token_cost": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({"status": "PASS", "report": bind(args.output), "valid_calls": scores["valid_calls"], "exact_calls": scores["exact_calls"], "by_arm_type": by_arm_type}))


if __name__ == "__main__":
    main()
