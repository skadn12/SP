"""Post-terminal-freeze canonical scoring; this is the only Gold JSON reader."""
from __future__ import annotations
import argparse
import collections
import itertools
import json
from pathlib import Path
import statistics
import common as c


def score_one(row, raw, variant, canonical, gold_labels):
    field = "options" if row["question_type"] == "multi_select" else "items"
    mapping = variant["display_to_content"]
    c.require(set(mapping) == set(row["display"][field]) and len(set(mapping.values())) == len(mapping), "NONBIJECTIVE_MAPPING")
    c.require(set(mapping.values()) == set(canonical.values()), "CANONICAL_MAPPING_MISMATCH")
    gold = [canonical[label] for label in gold_labels]
    c.require(len(gold) == len(set(gold)) and len(gold) >= 2, "INVALID_GOLD")
    if field == "items":
        c.require(set(gold) == set(mapping.values()), "ORDERING_GOLD_INCOMPLETE")
    scorer = c.a.native_scoring()
    result = scorer.score_response({"question_type": row["question_type"], "gold_content_ids": gold},
                                   {"variant": variant}, raw.get("raw_output"))
    errors = list(raw.get("output_validation_errors") or [])
    if raw["status"] != "RETURNED":
        errors.append("NOT_RETURNED:" + raw["status"])
    if raw.get("raw_output") not in c.a.choices(row["question_type"], row["display"][field]):
        errors.append("OUTSIDE_NATIVE_FINITE_CHOICE_SET")
    if raw.get("metadata", {}).get("finish_reason") not in ("stop", "eos"):
        errors.append("INVALID_FINISH_REASON:" + str(raw.get("metadata", {}).get("finish_reason")))
    if result.get("scored") is not True:
        errors.append(result.get("scoring_status", "NOT_SCORED"))
    valid = not errors
    prediction = result["content_prediction"] if valid else None
    reverse = {content: label for label, content in canonical.items()}
    base = {key: row[key] for key in ("request_id", "candidate_id", "root_id", "question_type", "variant_id", "slot", "seed")}
    base.update(status=raw["status"], valid=valid, exact=bool(valid and result["exact_match"]),
                parse_errors=errors, native_score=result, canonical_prediction=[reverse[item] for item in prediction] if prediction is not None else None,
                prediction_content=prediction, gold_canonical=gold_labels, gold_content=gold,
                finish_reason=raw.get("metadata", {}).get("finish_reason"),
                response_committed_at_utc=raw["response_committed_at_utc"])
    if field == "options":
        pred_set = set(prediction) if valid else set()
        gold_set = set(gold)
        items = []
        for label, content in canonical.items():
            truth, selected = content in gold_set, content in pred_set if valid else None
            outcome = "INVALID" if not valid else "TP" if truth and selected else "FN" if truth else "FP" if selected else "TN"
            items.append({"canonical_label": label, "content_id": content, "gold_selected": truth,
                          "predicted_selected": selected, "outcome": outcome,
                          "fp": bool(outcome == "FP") if valid else None, "fn": bool(outcome == "FN") if valid else None})
        base["items"] = items
        base["metrics"] = {"tp": sum(item["outcome"] == "TP" for item in items) if valid else None,
            "fp": sum(item["outcome"] == "FP" for item in items) if valid else None,
            "fn": sum(item["outcome"] == "FN" for item in items) if valid else None,
            "gold_count": len(gold), "selected_count": len(prediction) if valid else None,
            "jaccard": len(pred_set & gold_set) / len(pred_set | gold_set) if valid else 0,
            "invalid_item_policy": "Unknown selection and FP/FN are null; invalid remains in every planned-call denominator."}
    else:
        rank = {content: index for index, content in enumerate(prediction or [])}
        pairs = [{"before_content_id": left, "after_content_id": right,
                  "before_canonical_label": reverse[left], "after_canonical_label": reverse[right],
                  "predicted_before_rank": rank.get(left), "predicted_after_rank": rank.get(right),
                  "valid": valid, "correct": bool(valid and rank[left] < rank[right])}
                 for index, left in enumerate(gold) for right in gold[index + 1:]]
        base["pairs"] = pairs
        base["metrics"] = {"total_pairs": len(pairs), "correct_pairs": sum(pair["correct"] for pair in pairs),
                           "pair_accuracy": sum(pair["correct"] for pair in pairs) / len(pairs)}
    return base


def summarize_question(question, records):
    n = len(records)
    predictions = [json.dumps(record["prediction_content"]) for record in records if record["valid"]]
    exact = sum(record["exact"] for record in records)
    result = {key: question[key] for key in ("candidate_id", "root_id", "question_type")}
    result.update(family_id=question["root_id"] + "|" + question["question_type"], denominator_calls=n,
        valid_calls=sum(record["valid"] for record in records), invalid_calls=sum(not record["valid"] for record in records),
        exact_calls=exact, exact_rate=exact / n,
        stratum="incorrect" if exact == 0 else "stably_correct" if exact == n else "occasionally_correct",
        display_permutation_consistency=max(collections.Counter(predictions).values()) / n if predictions else 0)
    if question["question_type"] == "multi_select":
        items = []
        for label in question["options"]:
            subset = [next(item for item in record["items"] if item["canonical_label"] == label) for record in records]
            outcomes = collections.Counter(item["outcome"] for item in subset)
            items.append({"canonical_label": label, "content_id": subset[0]["content_id"], "gold_selected": subset[0]["gold_selected"],
                "denominator_calls": n, "valid_calls": n - outcomes["INVALID"], "invalid_calls": outcomes["INVALID"],
                **{key.lower() + "_calls": outcomes[key] for key in ("TP", "FP", "FN", "TN")}})
        result["items"] = items
        result["fp_calls"] = sum(item["fp_calls"] for item in items)
        result["fn_calls"] = sum(item["fn_calls"] for item in items)
    else:
        pairs = []
        for first in records[0]["pairs"]:
            key = (first["before_content_id"], first["after_content_id"])
            subset = [next(pair for pair in record["pairs"] if (pair["before_content_id"], pair["after_content_id"]) == key) for record in records]
            pairs.append({key: first[key] for key in ("before_content_id", "after_content_id", "before_canonical_label", "after_canonical_label")} |
                {"denominator_calls": n, "valid_calls": sum(pair["valid"] for pair in subset),
                 "correct_calls": sum(pair["correct"] for pair in subset), "accuracy": sum(pair["correct"] for pair in subset) / n})
        result["pairs"] = pairs
        result["total_pairs"] = n * len(pairs)
        result["correct_pairs"] = sum(pair["correct_calls"] for pair in pairs)
        result["pair_accuracy"] = result["correct_pairs"] / result["total_pairs"]
    return result


def cost(run, rows, raws):
    returned = [raw for raw in raws.values() if raw["status"] == "RETURNED"]
    keys = ("prompt_token_count", "completion_token_count")
    known = {key: sum(raw.get("metadata", {}).get(key) for raw in returned if type(raw.get("metadata", {}).get(key)) is int) for key in keys}
    execution_path = run / ("COMPLETE.json" if (run / "COMPLETE.json").exists() else "FAILED.json")
    execution = c.read(execution_path)
    return {"planned_calls": len(rows), "returned_calls": len(returned),
        "submitted_attempts": sum(raw["status"] not in ("NOT_SUBMITTED", "MISSING") for raw in raws.values()),
        "not_submitted_calls": sum(raw["status"] == "NOT_SUBMITTED" for raw in raws.values()),
        "requests_with_unknown_usage": sum(raw["status"] in ("BACKEND_ERROR", "MISSING") or
            (raw["status"] == "RETURNED" and any(type(raw.get("metadata", {}).get(key)) is not int for key in keys)) for raw in raws.values()),
        "known_prompt_tokens": known[keys[0]], "known_completion_tokens": known[keys[1]], "known_total_tokens": sum(known.values()),
        "live_wall_seconds": execution.get("wall_seconds"), "model_load_wall_seconds": execution.get("model_load_wall_seconds"),
        "execution_binding": c.bind(execution_path), "currency_cost": None,
        "currency_cost_reason": "Local GPU inference; monetary billing unavailable; unknown usage is not zero."}


def score(run, private):
    c.environment()
    run, private = Path(run).resolve(), Path(private).resolve()
    c.require(not (run / "SCORING_STARTED.json").exists() and not (run / "SCORES.json").exists(), "SCORING_ALREADY_STARTED_NO_OVERWRITE")
    manifest = c.verify(run, check_models=False)
    freeze = c.read(run / "RAW_RESPONSE_FREEZE.json")
    c.require(freeze["status"] == "COMMITTED", "COMMITTED_RAW_FREEZE_REQUIRED")
    c.verify_bindings(freeze["files"] + [freeze["manifest_binding"]])
    c.require(Path(freeze["manifest_binding"]["path"]).resolve() == run / "manifest.json", "WRONG_FROZEN_MANIFEST")
    rows = manifest["requests"]
    expected = {run / "responses" / (row["request_id"] + ".json") for row in rows}
    c.require(len(freeze["files"]) == len(rows) and {Path(item["path"]).resolve() for item in freeze["files"]} == expected, "RAW_DENOMINATOR_MISMATCH")
    c.require(set((run / "responses").glob("*.json")) == expected, "UNEXPECTED_RAW_FILES")
    raws = {}
    for row in rows:
        raw = c.read(run / "responses" / (row["request_id"] + ".json"))
        c.require(all(raw[key] == row[key] for key in ("request_id", "candidate_id", "root_id", "variant_id", "slot", "seed", "prompt_sha256")), "RAW_IDENTITY_MISMATCH")
        c.require(raw["status"] in ("RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"), "NONTERMINAL_RAW")
        c.require(c.timestamp(raw["response_committed_at_utc"]) <= c.timestamp(freeze["created_at_utc"]), "RAW_AFTER_FREEZE")
        raws[row["request_id"]] = raw
    # Private sidecar is first touched only after all terminal traces are verified.
    bindings = c.read(private / "PRIVATE_BINDINGS.json")
    c.verify_bindings([bindings["public_manifest_binding"], bindings["public_preflight_binding"], bindings["private_answer_binding"]])
    c.require(Path(bindings["public_manifest_binding"]["path"]).resolve() == run / "manifest.json", "PRIVATE_PUBLIC_MANIFEST_MISMATCH")
    gold_start = c.now()
    c.require(c.timestamp(gold_start) >= c.timestamp(freeze["created_at_utc"]), "CLOCK_BEFORE_FREEZE")
    c.save(run / "SCORING_STARTED.json", {"gold_first_read_at_utc": gold_start,
        "raw_response_freeze_binding": c.bind(run / "RAW_RESPONSE_FREEZE.json"),
        "private_answer_binding": bindings["private_answer_binding"], "all_terminal_raws_committed_before_gold": True,
        "gold_read_scope": "First Gold JSON parse in this scoring process; prior author/reviewer/parent exposure is not denied."})
    gold_payload = c.read(bindings["private_answer_binding"]["path"])
    gold_done = c.now()
    private_questions = {q["candidate_id"]: q for q in gold_payload["questions"]}
    public = c.read(run / "PUBLIC_QUESTIONS.json")["questions"]
    maps = c.read(run / "DISPLAY_MAPPINGS.json")
    c.require(set(private_questions) == {q["candidate_id"] for q in public}, "PRIVATE_PUBLIC_CANDIDATE_SET_MISMATCH")
    for question in public:
        saved = private_questions[question["candidate_id"]]
        field = "options" if saved["question_type"] == "multi_select" else "items"
        normalized = {key: saved[key] for key in ("candidate_id", "root_id", "question_type", "question", field)}
        normalized.update(source_path=saved["source_binding"]["path"], source_sha256=saved["source_binding"]["sha256"])
        c.require(normalized == question, "PRIVATE_PUBLIC_CONTENT_MISMATCH")
    records = []
    for row in rows:
        qid = row["candidate_id"]
        item = score_one(row, raws[row["request_id"]], maps["requests"][row["request_id"]],
                         maps["public_to_canonical"][qid], private_questions[qid]["gold"])
        item.update(response_binding=c.bind(run / "responses" / (row["request_id"] + ".json")),
                    gold_first_read_at_utc=gold_start)
        records.append(item)
    questions = [summarize_question(question, [record for record in records if record["candidate_id"] == question["candidate_id"]]) for question in public]
    for question in questions:
        subset = [record for record in records if record["candidate_id"] == question["candidate_id"]]
        question["gold_count"] = len(private_questions[question["candidate_id"]]["gold"])
        if question["question_type"] == "multi_select":
            question["mean_jaccard"] = statistics.mean(record["metrics"]["jaccard"] for record in subset)
    protocol = c.read(run / "PROTOCOL.json")
    design = {slot["candidate_id"]: slot for slot in protocol.get("slots", [])}
    for question in questions:
        question["design"] = design.get(question["candidate_id"])
    by_type = {}
    for typ in ("multi_select", "ordering"):
        subset = [question for question in questions if question["question_type"] == typ]
        roots = collections.defaultdict(list)
        for question in subset:
            roots[question["root_id"]].append(question["exact_rate"])
        by_type[typ] = {"questions": len(subset), "calls": sum(q["denominator_calls"] for q in subset),
            "question_equal_exact": statistics.mean(q["exact_rate"] for q in subset) if subset else None,
            "root_equal_exact": statistics.mean(statistics.mean(values) for values in roots.values()) if roots else None}
    roots = collections.defaultdict(list)
    for question in questions:
        roots[question["root_id"]].append(question["exact_rate"])
    def grouped_summary(subset):
        calls = [record for record in records if record["candidate_id"] in {q["candidate_id"] for q in subset}]
        return {"questions": len(subset), "calls": len(calls), "exact_calls": sum(r["exact"] for r in calls),
            "question_equal_exact": statistics.mean(q["exact_rate"] for q in subset) if subset else None,
            "valid_calls": sum(r["valid"] for r in calls),
            "mc_mean_jaccard": statistics.mean(r["metrics"]["jaccard"] for r in calls if r["question_type"] == "multi_select") if any(r["question_type"] == "multi_select" for r in calls) else None,
            "ordering_pair_accuracy": statistics.mean(r["metrics"]["pair_accuracy"] for r in calls if r["question_type"] == "ordering") if any(r["question_type"] == "ordering" for r in calls) else None}
    by_root = {root: grouped_summary([q for q in questions if q["root_id"] == root]) for root in sorted(roots)}
    by_gold_count = {str(count): grouped_summary([q for q in questions if q["question_type"] == "multi_select" and q["gold_count"] == count]) for count in range(2, 7)}
    result = {"status": "COMPLETE" if (run / "COMPLETE.json").exists() else "FAILED_WITH_FIXED_DENOMINATOR",
        "scored_at_utc": c.now(), "gold_first_read_at_utc": gold_start, "gold_read_completed_at_utc": gold_done,
        "manifest_binding": c.bind(run / "manifest.json"), "preflight_freeze_binding": c.bind(run / "PREFLIGHT_FREEZE.json"),
        "raw_response_freeze_binding": c.bind(run / "RAW_RESPONSE_FREEZE.json"),
        "private_answer_binding": bindings["private_answer_binding"], "scoring_code_binding": c.bind(__file__),
        "denominator_calls": len(rows), "denominator_questions": len(questions), "denominator_roots": len(roots),
        "planned_question_slots": manifest["planned_question_slots"], "production_denominator": manifest["production_denominator"],
        "production_unadmitted_slots": manifest["production_unadmitted_slots"],
        "denominator_scope": manifest["denominator_scope"],
        "production_admitted_numerator": len(questions), "missing_candidate_ids": manifest["missing_candidate_ids"],
        "planned_slot_outcomes": [{"candidate_id": candidate_id,
            "status": "ADMITTED_EVALUATED" if candidate_id not in manifest["missing_candidate_ids"] else "MISSING_NOT_EVALUATED",
            "planned_slot": slot} for candidate_id, slot in design.items()],
        "valid_calls": sum(record["valid"] for record in records), "exact_calls": sum(record["exact"] for record in records),
        "question_equal_exact": statistics.mean(question["exact_rate"] for question in questions) if questions else None,
        "root_equal_exact": statistics.mean(statistics.mean(values) for values in roots.values()) if roots else None,
        "by_type": by_type, "by_root": by_root, "multi_select_by_gold_count": by_gold_count,
        "cost": cost(run, rows, raws), "questions": questions, "requests": records,
        "use_as_final_eval": False, "training_started": False, "automatic_retry": False, "source_truncation": False,
        "invalid_policy": "All invalid and terminal failures remain in fixed denominators; exact and ordering pair credit zero; MC item selection/FP/FN unknown is null, never fabricated."}
    c.save(run / "SCORES.json", result)
    print(json.dumps({key: result[key] for key in ("status", "denominator_calls", "denominator_questions", "valid_calls", "exact_calls", "by_type")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    score(args.run, args.private)


if __name__ == "__main__":
    main()
