"""Read-only independent replay of the sealed v96 paired Solver experiment.

No engine scoring/selection code is imported. The only write is an exclusive
audit JSON in evidence. Run only after both models and the mixed report finish.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

MODELS = {"trained": "M_clean_policy", "base": "M0"}
CALLS = {"multi_select": 18, "ordering": 12}
ALLOWED_MODEL_DIFFERENCES = {
    "prepared_at_utc", "model", "solver_model", "solver_identity"
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.utcoffset() is not None, "timezone_missing")
    return result


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify_files(rows):
    require(isinstance(rows, list), "frozen_file_list_missing")
    seen = set()
    for row in rows:
        path = Path(row["path"]).resolve()
        require(path not in seen, "duplicate_frozen_binding:" + str(path))
        seen.add(path)
        require(sha(path) == row["sha256"], "frozen_bytes_changed:" + str(path))
    return len(seen)


def score_raw(raw, request, gold):
    """Derive content predictions from raw boxed labels and frozen permutation."""
    require(isinstance(raw, str), "raw_answer_not_string")
    match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", raw, re.S)
    require(match is not None, "raw_answer_box_invalid")
    displayed = [value.strip().upper() for value in match.group(1).split(",")]
    kind = request["question_type"]
    labels = set("ABCDEF") if kind == "multi_select" else {f"I{i}" for i in range(1, 7)}
    mapping = request["variant"]["display_to_content"]
    require(set(mapping) == labels and len(set(mapping.values())) == 6, "bad_display_mapping")
    require(displayed and len(displayed) == len(set(displayed))
            and set(displayed) <= labels, "raw_answer_labels_invalid")
    prediction = [mapping[value] for value in displayed]
    require(len(set(gold)) == len(gold) and set(gold) <= set(mapping.values()), "bad_gold")
    if kind == "multi_select":
        require(2 <= len(gold) <= 6, "multi_select_gold_count_invalid")
        selected, supported = set(prediction), set(gold)
        tp = len(selected & supported)
        fp = len(selected - supported)
        fn = len(supported - selected)
        partial = Fraction(tp, len(gold))
        if len(gold) < 6:
            partial -= Fraction(fp, 6 - len(gold))
        exact = selected == supported
    else:
        require(kind == "ordering" and len(gold) == len(prediction) == 6
                and set(prediction) == set(gold), "ordering_not_complete_permutation")
        rank = {value: index for index, value in enumerate(prediction)}
        fp = sum(rank[left] > rank[right]
                 for index, left in enumerate(gold) for right in gold[index + 1:])
        tp, fn = 15 - fp, 0
        partial, exact = Fraction(tp - fp, 15), prediction == gold
    return {"exact": exact, "partial": str(partial), "tp": tp, "fp": fp,
            "fn": fn, "selected_count": len(prediction), "gold_count": len(gold)}


def statistics(rows):
    count = len(rows)
    candidates = sorted({row["candidate_id"] for row in rows})
    if not count:
        return {"unique_questions": 0, "calls": 0, "exact_correct": 0,
                "exact_accuracy": None, "partial_mean": None,
                "question_macro_exact_accuracy": None, "question_macro_partial_mean": None}
    exact = Fraction(sum(row["exact"] for row in rows), count)
    partial = sum((Fraction(row["partial"]) for row in rows), Fraction()) / count
    macro_exact, macro_partial = [], []
    for qid in candidates:
        values = [row for row in rows if row["candidate_id"] == qid]
        macro_exact.append(Fraction(sum(row["exact"] for row in values), len(values)))
        macro_partial.append(sum((Fraction(row["partial"]) for row in values), Fraction()) / len(values))
    return {"unique_questions": len(candidates), "calls": count,
            "exact_correct": sum(row["exact"] for row in rows),
            "exact_accuracy": str(exact), "exact_accuracy_float": float(exact),
            "partial_mean": str(partial), "partial_mean_float": float(partial),
            "question_macro_exact_accuracy": str(sum(macro_exact, Fraction()) / len(candidates)),
            "question_macro_partial_mean": str(sum(macro_partial, Fraction()) / len(candidates))}


def audit_side(home, side, pool, frozen, tracked):
    run = home / "solver" / side
    manifest = read(run / "manifest.json")
    start = read(run / "STARTED.json")
    complete = read(run / "COMPLETE.json")
    stage_complete = read(home / (side.upper() + "_EVALUATION.COMPLETE.json"))
    require(not (run / "FAILED.json").exists()
            and not (home / (side.upper() + "_EVALUATION.ERROR.json")).exists(), side + ":failed_marker")
    require(complete["status"] == stage_complete["status"] == "COMPLETE", side + ":not_complete")
    require(manifest["solver_model"] == MODELS[side], side + ":wrong_model")
    require(sha(run / "manifest.json") == start["manifest_sha256"], side + ":manifest_changed")
    require(timestamp(frozen["completed_at_utc"]) < timestamp(start["started_at_utc"]),
            side + ":content_not_frozen_before_solver")
    require(manifest["candidate_count"] == len(pool) == 11
            and manifest["request_count"] == complete["actual_call_count"]
            == stage_complete["calls"] == 192, side + ":wrong_counts")
    require(manifest["training_started"] is False and manifest["training_promotion"] is False
            and complete["training_started"] is False and complete["training_promotion"] is False,
            side + ":training_not_outside_scope")
    require(manifest["pool_sha256"] == manifest["input_sha256"]["pool"]
            == sha(home / "EVALUATION_POOL.json"), side + ":pool_changed")
    for name, expected in manifest["input_sha256"].items():
        path = home / "EVALUATION_POOL.json" if name == "pool" else run / name
        require(sha(path) == expected, side + ":input_changed:" + name)
        tracked.add(path)
    for path, expected in manifest["code_sha256"].items():
        require(sha(path) == expected, side + ":code_changed:" + path)
    require(read(run / "solver_identity.json") == manifest["solver_identity"], side + ":identity_changed")
    requests = manifest["requests"]
    by_request = {row["request_id"]: row for row in requests}
    require(len(by_request) == len(requests) == 192, side + ":duplicate_requests")
    require(Counter(row["question_type"] for row in requests)
            == Counter({"multi_select": 180, "ordering": 12}), side + ":typed_call_count")
    expected_positions, golden = set(), read(run / "gold.json")
    require(set(golden) == set(pool), side + ":gold_coverage")
    seed0 = requests[0]["seed"]
    for index, request in enumerate(requests):
        qid, variant = request["candidate_id"], request["variant"]
        question = pool[qid]
        kind, vid = question["question_type"], variant["variant_id"]
        require(request["root_id"] == question["root_id"] and request["question_type"] == kind
                and request["mode"] == "full_context" and request["seed"] == seed0 + index,
                side + ":request_identity_or_seed")
        require(request["request_id"] == qid + "__" + vid + "__full_context__s" + f"{request['slot']:02d}",
                side + ":request_id_changed")
        position = (qid, vid, request["slot"])
        require(position not in expected_positions, side + ":duplicate_variant_slot")
        expected_positions.add(position)
        field = "options" if kind == "multi_select" else "items"
        texts = variant["content_to_text"]
        require(len(texts) == len(set(texts.values())) == 6
                and set(texts.values()) == set(question[field].values()), side + ":content_text_changed")
        reverse = {value: content for content, value in texts.items()}
        derived_gold = [reverse[question[field][label]] for label in question["gold"]]
        require(golden[qid] == derived_gold, side + ":gold_content_mapping_changed")
        require(set(variant[field]) == set(variant["display_to_content"])
                and all(variant[field][label] == texts[content]
                        for label, content in variant["display_to_content"].items()),
                side + ":display_text_changed")
        prompt = run / "prompts" / (request["request_id"] + ".txt")
        require(sha(prompt) == request["prompt_sha256"], side + ":prompt_changed")
    for qid, question in pool.items():
        kind = question["question_type"]
        expected = {(qid, f"perm_{i:02d}", slot)
                    for i in range(6 if kind == "multi_select" else 12)
                    for slot in range(3 if kind == "multi_select" else 1)}
        require({p for p in expected_positions if p[0] == qid} == expected, side + ":variant_slots_incomplete")
    files = sorted((run / "responses").glob("*.json"))
    require(len(files) == 192, side + ":raw_response_count")
    records = [read(path) for path in files]
    require(Counter(row["request_id"] for row in records) == Counter(by_request.keys()), side + ":raw_response_set")
    replayed = []
    for path, record in zip(files, records):
        request = by_request[record["request_id"]]
        for key in ("candidate_id", "root_id", "question_type", "slot", "seed", "mode"):
            require(record[key] == request[key], side + ":response_identity:" + key)
        require(record["variant_id"] == request["variant"]["variant_id"]
                and record["solver_model"] == manifest["solver_model"]
                and record["model_path"] == manifest["model"], side + ":response_model_variant")
        metadata = record["metadata"]
        require(record["evaluation_valid"] is True and record["data_errors"] == []
                and metadata["finish_reason"] in ("stop", "eos"), side + ":invalid_response")
        require(metadata["sampling_seed"] == request["seed"]
                and metadata["prompt_token_count"] == request["prompt_tokens"], side + ":metadata_seed_prompt")
        for key in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
            require(metadata[key] == manifest["decoder_contract"][key], side + ":decoder_changed:" + key)
        scored = score_raw(record["raw_output"], request, golden[record["candidate_id"]])
        saved = record["score"]
        require(saved["parse_status"] == "success" and saved["scored"] is True
                and saved["exact_match"] == scored["exact"]
                and Fraction(saved["partial_score_fraction"]) == Fraction(scored["partial"])
                and all(saved[key] == scored[key] for key in ("tp", "fp", "fn")), side + ":saved_score_disagrees")
        replayed.append({"request_id": record["request_id"], "candidate_id": record["candidate_id"],
                         "root_id": record["root_id"], "question_type": record["question_type"],
                         "variant_id": record["variant_id"], "slot": record["slot"], "seed": record["seed"],
                         **scored, "raw_response": binding(path)})
        tracked.add(path)
    results = read(run / "RESULTS.json")
    require(results["call_set_errors"] == [] and results["solver_model"] == MODELS[side], side + ":results_invalid")
    summaries = []
    for qid, question in pool.items():
        rows = [row for row in replayed if row["candidate_id"] == qid]
        means = {vid: sum((Fraction(row["partial"]) for row in rows if row["variant_id"] == vid), Fraction())
                 / sum(row["variant_id"] == vid for row in rows)
                 for vid in sorted({row["variant_id"] for row in rows})}
        mean = sum(means.values(), Fraction()) / len(means)
        selected = 0 < mean < 1
        expected_reason = ("FULL_PARTIAL_IN_OPEN_INTERVAL" if selected else
                           "FULL_PARTIAL_NONPOSITIVE" if mean <= 0 else
                           "FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM")
        saved_rows = [row for row in results["questions"] if row["candidate_id"] == qid]
        require(len(saved_rows) == 1, side + ":selection_coverage")
        saved = saved_rows[0]
        require(saved["selected_by_policy"] is selected and saved["selection_reason"] == expected_reason
                and saved["data_errors"] == [] and Fraction(saved["full_partial_mean"]) == mean
                and saved["permutation_means"] == {key: str(value) for key, value in means.items()}
                and Fraction(saved["full_exact_rate"]) == Fraction(sum(row["exact"] for row in rows), len(rows))
                and saved["expected_full_calls"] == saved["actual_full_calls"] == CALLS[question["question_type"]],
                side + ":selection_disagrees:" + qid)
        role = "fresh_parent_gold6_control" if question["question_type"] == "multi_select" and len(question["gold"]) == 6 else "new_question"
        summaries.append({"candidate_id": qid, "root_id": question["root_id"],
                          "question_type": question["question_type"], "gold_count": len(question["gold"]),
                          "role": role, "selected_by_policy": selected, "selection_reason": expected_reason,
                          "full_partial_mean": str(mean), **statistics(rows)})
    expected_selected = [row["candidate_id"] for row in summaries if row["selected_by_policy"]]
    require(results["selected_questions"] == expected_selected and len(results["questions"]) == 11,
            side + ":selected_ids_changed")
    tracked.update(run / name for name in ("manifest.json", "STARTED.json", "COMPLETE.json", "RESULTS.json", "solver_identity.json"))
    return {"model": MODELS[side], "status": "PASS", "calls": 192, "valid_calls": 192,
            "manifest": binding(run / "manifest.json"), "solver_started_at_utc": start["started_at_utc"],
            "questions": summaries, "responses": replayed}


def run_audit(home):
    home = Path(home).resolve()
    for name in ("TRAINED_EVALUATION.COMPLETE.json", "BASE_EVALUATION.COMPLETE.json", "MIXED_EVALUATION_REPORT.json"):
        require((home / name).is_file(), "not_ready:" + name)
    frozen = read(home / "BATCH_REVIEW.COMPLETE.json")
    prepared = read(home / "EVALUATION_PREPARATION.COMPLETE.json")
    report = read(home / "MIXED_EVALUATION_REPORT.json")
    pool_list = read(home / "EVALUATION_POOL.json")["questions"]
    pool = {question["candidate_id"]: question for question in pool_list}
    require(len(pool) == len(pool_list) == 11 and Counter(q["question_type"] for q in pool_list)
            == Counter({"multi_select": 10, "ordering": 1}), "unexpected_content_denominator")
    require(frozen["status"] == "FROZEN" and frozen["before_any_solver"] is True
            and frozen["solver_scores_used"] is False, "content_not_frozen")
    bindings_verified = {"content_freeze": verify_files(frozen["files"]),
                         "evaluation_preparation": verify_files(prepared["files"]),
                         "mixed_evaluation_report": verify_files(report["files"])}
    require(prepared["status"] == "PASS" and prepared["calls_per_model"] == 192
            and prepared["review_frozen_at_utc"] == frozen["completed_at_utc"], "preparation_invalid")
    tracked = {home / name for name in ("BATCH_REVIEW.COMPLETE.json", "EVALUATION_POOL.json",
               "EVALUATION_PREPARATION.COMPLETE.json", "TRAINED_EVALUATION.COMPLETE.json",
               "BASE_EVALUATION.COMPLETE.json", "MIXED_EVALUATION_REPORT.json", "SELECTED_CANDIDATES.json", "PAIRED_CALLS.json")}
    models = {side: audit_side(home, side, pool, frozen, tracked) for side in MODELS}
    manifests = [read(home / "solver" / side / "manifest.json") for side in MODELS]
    kept = [{key: value for key, value in m.items() if key not in ALLOWED_MODEL_DIFFERENCES} for m in manifests]
    require(kept[0] == kept[1], "paired_manifests_not_equal")
    left = {row["request_id"]: row for row in models["trained"]["responses"]}
    right = {row["request_id"]: row for row in models["base"]["responses"]}
    require(set(left) == set(right) and len(left) == 192, "pair_count_not_192")
    for rid, row in left.items():
        require(all(row[key] == right[rid][key] for key in ("candidate_id", "root_id", "question_type", "variant_id", "slot", "seed")),
                "paired_response_identity_changed")
    new_ids = {q["candidate_id"] for q in pool_list if q["question_type"] == "ordering" or len(q["gold"]) < 6}
    control_ids = set(pool) - new_ids
    selected = [row["candidate_id"] for row in models["trained"]["questions"] if row["selected_by_policy"]]
    require(report["status"] == "PASS" and report["evaluation_status"] == "COMPLETED"
            and report["calls_per_model"] == 192 and report["selected_candidate_ids"] == selected
            and report["selected_new_candidate_count"] == len(set(selected) & new_ids)
            and report["content_frozen_before_any_solver"] is True
            and report["training_started"] is False and report["formal_training_admission"] is False,
            "mixed_report_or_selection_disagrees")
    require(read(home / "SELECTED_CANDIDATES.json")["questions"] == [q for q in pool_list if q["candidate_id"] in selected],
            "selected_pool_disagrees")
    summaries = {}
    for side, model in models.items():
        rows = model["responses"]
        summary = {"all": statistics(rows),
                   "new_only": statistics([row for row in rows if row["candidate_id"] in new_ids]),
                   "fresh_parent_gold6_controls": statistics([row for row in rows if row["candidate_id"] in control_ids]),
                   "by_type": {}, "by_multiselect_gold": {}, "by_root_type_gold": []}
        for kind in CALLS:
            typed = [row for row in rows if row["question_type"] == kind]
            summary["by_type"][kind] = {
                "all": statistics(typed),
                "new_only": statistics([row for row in typed if row["candidate_id"] in new_ids]),
                "fresh_parent_gold6_controls": statistics([row for row in typed if row["candidate_id"] in control_ids])}
            observed = statistics(typed)
            stated = report["by_type"][kind]["models"][side]
            require(all(stated[key] == observed[key] for key in ("calls", "exact_correct", "exact_accuracy", "partial_mean")),
                    "mixed_typed_stats_disagree:" + side + ":" + kind)
        for g in range(2, 7):
            values = [row for row in rows if row["question_type"] == "multi_select" and row["gold_count"] == g]
            observed = statistics(values)
            summary["by_multiselect_gold"][str(g)] = {
                "role": "fresh_parent_gold6_control" if g == 6 else "new_question", **observed}
            stated = report["by_type"]["multi_select"]["by_gold_count"][str(g)]
            require(stated["content_usable"] == observed["unique_questions"]
                    and stated["solver_selected"] == sum(pool[qid]["question_type"] == "multi_select"
                        and len(pool[qid]["gold"]) == g for qid in selected)
                    and all(stated["models"][side][key] == observed[key]
                            for key in ("calls", "exact_correct", "exact_accuracy", "partial_mean")),
                    "mixed_gold_stats_disagree:" + side + ":" + str(g))
        for root in sorted({q["root_id"] for q in pool_list}):
            for kind in CALLS:
                for g in sorted({len(q["gold"]) for q in pool_list if q["root_id"] == root and q["question_type"] == kind}):
                    values = [row for row in rows if row["root_id"] == root and row["question_type"] == kind and row["gold_count"] == g]
                    summary["by_root_type_gold"].append({"root_id": root, "question_type": kind, "gold_count": g,
                        "role": "fresh_parent_gold6_control" if kind == "multi_select" and g == 6 else "new_question",
                        **statistics(values)})
        summaries[side] = summary
    selection_rows = models["trained"]["questions"]
    return {"version": "v96_independent_raw_paired_result_audit_v1", "status": "PASS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "auditor": "/root/v92_plan_semantic_gate", "audit_script": binding(Path(__file__)),
            "exposure_note": "Earlier source-first judgments were saved before current private child Gold or Solver access. This later mechanical result audit reads frozen private Gold, evaluation pool and raw Solver responses; it does not relabel the earlier semantic judgments as blind or change them.",
            "method": "Independent standard-library raw-box parsing, displayed-label-to-content mapping, exact set/order comparison and Fraction arithmetic; no production scoring/selection function imported.",
            "score_definitions": {"multi_select": "TP/G - FP/(6-G); for G=6 use TP/6 without FP term",
                                  "ordering": "(concordant - discordant)/15 over all 15 ordered pairs"},
            "paired_calls_per_model": 192, "total_paired_calls": 192, "total_model_calls": 384,
            "unique_question_count": 11, "unique_roots": 2, "new_unique_questions": len(new_ids),
            "fresh_parent_gold6_controls": len(control_ids),
            "content_freeze_at_utc": frozen["completed_at_utc"], "content_frozen_before_both_solvers": True,
            "frozen_binding_counts_verified": bindings_verified,
            "paired_prompts_permutations_seeds_and_decoder_equal": True,
            "every_raw_score_and_saved_selection_recomputed": True,
            "selection_rule": "Valid full-context evaluation and 0 < trained mean partial score across permutation means < 1",
            "selected_candidate_ids": selected, "selected_total_questions": len(selected),
            "selected_new_questions": len(set(selected) & new_ids),
            "selected_fresh_parent_controls": len(set(selected) & control_ids),
            "selected_new_by_type": dict(Counter(pool[qid]["question_type"] for qid in selected if qid in new_ids)),
            "selected_new_multiselect_by_gold": dict(sorted(Counter(str(len(pool[qid]["gold"])) for qid in selected if qid in new_ids and pool[qid]["question_type"] == "multi_select").items())),
            "selected_by_root": dict(Counter(pool[qid]["root_id"] for qid in selected)),
            "selected_new_by_root": dict(Counter(pool[qid]["root_id"] for qid in selected if qid in new_ids)),
            "question_selection": selection_rows, "statistics": summaries, "models": models,
            "control_label_correction": "The frozen engine's historical_six_supported_control role string is legacy wording. These two Gold-6 questions are fresh parents authored from this round's new expansion sources, not retests of historical questions. They are excluded from all new-question yield counts.",
            "sampling_unit_note": "192 calls/model are repeated permutations and rollouts of 11 fixed questions, not 192 independent question samples. Results describe two purposively chosen roots and their current families; no population yield, independent-sample confidence interval, or stable production claim is made.",
            "training_started": False, "formal_training_admission": False,
            "artifacts": [binding(path) for path in sorted(tracked)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=Path(__file__).resolve().parents[1] / "shards/shard_01")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "INDEPENDENT_RESULT_AUDIT.json")
    args = parser.parse_args()
    # Missing completion is an explicit prerequisite error, not a partial audit.
    for name in ("TRAINED_EVALUATION.COMPLETE.json", "BASE_EVALUATION.COMPLETE.json", "MIXED_EVALUATION_REPORT.json"):
        require((args.home / name).is_file(), "not_ready:" + name)
    result = run_audit(args.home)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "sha256": sha(args.output), "calls_per_model": result["paired_calls_per_model"],
                      "selected_new_questions": result["selected_new_questions"]}))


if __name__ == "__main__":
    main()
