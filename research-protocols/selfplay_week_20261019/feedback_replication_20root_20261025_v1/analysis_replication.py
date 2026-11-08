#!/usr/bin/env python3
"""CPU-only, outcome-neutral F/G replication analysis; never run a model.

Default: read frozen real artifacts and exclusively create three reports in this
experiment. --self-check uses only in-memory synthetic fixtures and writes no
files. --output-dir permits a separate snapshot directory; --overwrite replaces
only this script's three derived reports, never inputs, raw responses or scores.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

P = Path(__file__).resolve().parent
ARMS = ("F", "G")
VARIANTS = ("perm_00", "perm_01")
SLOTS = ("MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER2", "ORDER3")
GROUPS = ("NON_MC6_PRIMARY", "MC2_5_PRIMARY", "ORDERING", "MC6_CONTROL", "ALL")
GROUP_SLOTS = {
    "NON_MC6_PRIMARY": tuple(s for s in SLOTS if s != "MC6"),
    "MC2_5_PRIMARY": SLOTS[:4], "ORDERING": SLOTS[5:],
    "MC6_CONTROL": ("MC6",), "ALL": SLOTS,
}
GROUP_ZH = {"NON_MC6_PRIMARY": "主比较（不含 MC6）", "MC2_5_PRIMARY": "多选 Gold2–5",
            "ORDERING": "排序", "MC6_CONTROL": "MC6 全选控制", "ALL": "全部（描述性）"}
BOOTSTRAPS, SEED = 10000, 20261025
OUTPUT_NAMES = ("REPLICATION_COMPARISON.json", "REPLICATION_TABLES_ZH.md", "REPLICATION_ROOT_DIFFERENCES.csv")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def root_id(value):
    value = str(value)
    return value if value.startswith("cc-train-episode-") else "cc-train-episode-" + value


def mean(values):
    return statistics.mean(values) if values else None


def ratio(a, b):
    return a / b if b else None


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


class Inputs:
    def __init__(self):
        self.bound = {}
        self.absent = set()

    def read(self, path, optional=False):
        path = Path(path).resolve()
        if optional and not path.exists():
            self.absent.add(str(path))
            return None
        data = path.read_bytes()
        b = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        need(str(path) not in self.bound or self.bound[str(path)] == b, "INPUT_CHANGED_DURING_READ:" + str(path))
        self.bound[str(path)] = b
        return json.loads(data)

    def check(self, expected):
        actual = binding(expected["path"])
        need(all(actual[k] == expected[k] for k in ("sha256", "bytes")), "BOUND_INPUT_DRIFT:" + expected["path"])
        self.bound[actual["path"]] = actual

    def recheck(self):
        for b in self.bound.values():
            need(binding(b["path"]) == b, "INPUT_CHANGED_BEFORE_REPORT:" + b["path"])
        for path in self.absent:
            need(not Path(path).exists(), "INPUT_APPEARED_DURING_ANALYSIS:" + path)


def same_binding(expected, path):
    actual = binding(path)
    need(Path(expected["path"]).resolve() == Path(actual["path"]), "BOUND_PATH_DRIFT:" + str(path))
    need(all(expected[k] == actual[k] for k in ("sha256", "bytes")), "BOUND_CONTENT_DRIFT:" + str(path))


def split_final(text):
    """Same boundaries as frozen structured-v2 split_output; no JSON repair."""
    need(isinstance(text, str), "NO_OUTPUT_TEXT")
    text = text.strip()
    if text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")].rstrip()
    need("<tool_call>" not in text and "<|im_start|>" not in text, "UNEXPECTED_TOOL_OR_CHAT_WRAPPER")
    if "<think>" not in text and "</think>" not in text:
        return text
    need(text.startswith("<think>") and text.count("<think>") == text.count("</think>") == 1,
         "UNBALANCED_OR_MULTIPLE_THINK")
    _, final = text[len("<think>"):].split("</think>", 1)
    need(final.strip(), "MISSING_FINAL_AFTER_THINK")
    return final.strip()


def answer_only(raw, public, presentation):
    """Original legal_answer syntax, independent of evidence/finish penalties.

    A final JSON object and the exact original box grammar are still required.
    This is an answer diagnostic, not a relaxation of the official strict score.
    """
    result = {"usable": False, "exact": False, "prediction": [], "display_prediction": [], "error": None}
    if raw["status"] != "RETURNED":
        return {**result, "exact": None, "error": "NO_REAL_RETURN"}
    try:
        obj = json.loads(split_final(raw.get("raw_output")))
        need(isinstance(obj, dict) and isinstance(obj.get("final_answer"), str), "NO_FINAL_ANSWER_FIELD")
        match = re.fullmatch(r"<box>([A-F](?: [A-F]){0,5})</box>", obj["final_answer"])
        need(match is not None, "INVALID_ANSWER_BOX")
        answer = match.group(1).split()
        need(len(answer) == len(set(answer)) and set(answer) <= set(public["labels"]), "INVALID_ANSWER_LABELS")
        if public["question_type"] == "ordering":
            need(set(answer) == set(public["labels"]), "INCOMPLETE_ORDERING")
        prediction = [presentation["display_to_canonical"][label] for label in answer]
        if public["question_type"] == "multi_select":
            prediction.sort()
        gold = presentation["canonical_gold"]
        exact = set(prediction) == set(gold) if public["question_type"] == "multi_select" else prediction == gold
        result.update(usable=True, exact=exact, prediction=prediction, display_prediction=answer)
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


def validate_strict(record, public, presentation):
    """Check existing official scores, without changing the original contract."""
    s = record["strict"]
    for key in ("request_id", "candidate_id", "root_id", "question_type", "task_operation"):
        need(s[key] == public[key], "SCORE_PUBLIC_DRIFT:" + key)
    need(record["arm"] == public["arm"] and record["slot_id"] == public["slot_id"], "SCORE_ARM_SLOT_DRIFT")
    need(s["variant_id"] == public["variant_id"] in VARIANTS, "SCORE_VARIANT_DRIFT")
    need(type(s["valid"]) is bool and type(s["exact"]) is bool, "INVALID_SCORE_FLAGS")
    gold, pred = s["gold"], s["prediction"]
    need(gold == presentation["canonical_gold"] and s["display_gold"] == presentation["display_gold"], "SCORE_GOLD_DRIFT")
    need(len(pred) == len(set(pred)) and set(pred) <= set("ABCDEF"), "INVALID_CANONICAL_PREDICTION")
    if s["valid"]:
        restored = [presentation["display_to_canonical"][x] for x in s["display_prediction"]]
        need(set(pred) == set(restored) if s["question_type"] == "multi_select" else pred == restored,
             "CANONICAL_RESTORE_DRIFT")
    else:
        need(pred == [] and s["display_prediction"] == [], "INVALID_RESPONSE_HAS_STRICT_PREDICTION")
    if s["question_type"] == "multi_select":
        need(s["gold_count"] == len(gold) == int(record["slot_id"][2:]), "SCORE_GOLD_QUOTA_DRIFT")
        need(s["false_positive"] == len(set(pred) - set(gold)) and
             s["false_negative"] == len(set(gold) - set(pred)), "SCORE_FP_FN_DRIFT")
        expected_exact = bool(s["valid"] and set(pred) == set(gold))
    else:
        need(s["gold_count"] is None and len(gold) == 6 and s["pairwise_total"] == 15, "PAIRWISE_DENOMINATOR_DRIFT")
        if s["valid"]:
            need(len(pred) == 6 and set(pred) == set(gold), "INVALID_ORDERING_PREDICTION")
        positions = {label: i for i, label in enumerate(pred)}
        correct = sum(bool(s["valid"] and positions[a] < positions[b])
                      for i, a in enumerate(gold) for b in gold[i+1:])
        need(s["pairwise_correct"] == correct, "PAIRWISE_SCORE_DRIFT")
        expected_exact = bool(s["valid"] and pred == gold)
    need(s["exact"] == expected_exact, "STRICT_EXACT_DRIFT")


def normalize_prediction(prediction, question_type):
    return tuple(sorted(prediction)) if question_type == "multi_select" else tuple(prediction)


def mc_metrics(rows, answer=False):
    data = []
    for r in rows:
        s = r["strict"]
        pred = r["answer_only"]["prediction"] if answer else s["prediction"]
        gold = s["gold"]
        tp, fp, fn = len(set(pred) & set(gold)), len(set(pred) - set(gold)), len(set(gold) - set(pred))
        data.append({"precision": tp / (tp+fp) if tp+fp else 0., "recall": tp / (tp+fn),
                     "f1": 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0., "fp": fp, "fn": fn,
                     "overselection": len(pred) > len(gold), "underselection": len(pred) < len(gold)})
    return {"requests": len(rows), "macro": {key: mean([d[key] for d in data]) for key in ("precision", "recall", "f1")},
            "false_positive_labels": sum(d["fp"] for d in data), "false_negative_labels": sum(d["fn"] for d in data),
            "overselection_response_rate": mean([int(d["overselection"]) for d in data]),
            "underselection_response_rate": mean([int(d["underselection"]) for d in data])}


def consistency(rows):
    by_question = collections.defaultdict(list)
    for r in rows:
        by_question[(r["strict"]["root_id"], r["slot_id"], r["arm"])].append(r)
    results = []
    for (root, slot, arm), rs in sorted(by_question.items()):
        rs.sort(key=lambda r: r["strict"]["variant_id"])
        complete = len(rs) == 2 and {r["strict"]["variant_id"] for r in rs} == set(VARIANTS)
        both_returned = complete and all(r["raw_status"] == "RETURNED" for r in rs)
        both_valid = both_returned and all(r["strict"]["valid"] for r in rs)
        usable = both_returned and all(r["answer_only"]["usable"] for r in rs)
        same = usable and len({normalize_prediction(r["answer_only"]["prediction"], r["strict"]["question_type"]) for r in rs}) == 1
        results.append({"root_id": root, "slot_id": slot, "arm": arm, "candidate_id": rs[0]["strict"]["candidate_id"],
                        "both_returned": both_returned, "both_contract_valid": both_valid, "both_final_usable": usable,
                        "both_exact": both_returned and all(r["strict"]["exact"] for r in rs),
                        "at_least_one_exact": both_returned and any(r["strict"]["exact"] for r in rs),
                        "same_canonical_prediction": same, "same_canonical_contract_valid_prediction": both_valid and same})
    pairs = [q for q in results if q["both_returned"]]
    valid = [q for q in pairs if q["both_contract_valid"]]
    usable_pairs = [q for q in pairs if q["both_final_usable"]]
    return {"complete_returned_semantic_questions": len(pairs),
            **{key: sum(q[key] for q in pairs) for key in ("both_exact", "at_least_one_exact", "both_contract_valid", "both_final_usable", "same_canonical_prediction")},
            "both_exact_rate": ratio(sum(q["both_exact"] for q in pairs), len(pairs)),
            "at_least_one_exact_rate": ratio(sum(q["at_least_one_exact"] for q in pairs), len(pairs)),
            "same_canonical_prediction_rate": ratio(sum(q["same_canonical_prediction"] for q in pairs), len(pairs)),
            "usable_final_conditioned_consistency_rate": ratio(sum(q["same_canonical_prediction"] for q in usable_pairs), len(usable_pairs)),
            "contract_valid_conditioned_consistency_rate": ratio(sum(q["same_canonical_contract_valid_prediction"] for q in valid), len(valid)),
            "questions": results}


def target_status(value):
    if isinstance(value, dict):
        value = value.get("decision", value.get("target_realization"))
    return str(value) if value is not None else "NOT_REVIEWED"


def target_summary(ledger, arm):
    rows = [r for r in ledger if r["arm"] == arm]
    by_slot = {}
    for slot in ("ALL", *SLOTS):
        selected = rows if slot == "ALL" else [r for r in rows if r["slot_id"] == slot]
        valid = [r for r in selected if r["final_content_valid"]]
        states = collections.Counter(target_status(r.get("target_realization")) for r in valid)
        reviewed = states["PASS"] + states["HOLD"]
        plan_states = collections.Counter(r.get("target_plan_status") or "NOT_RECORDED" for r in selected)
        by_slot[slot] = {"planned_slots": len(selected), "final_content_valid_slots": len(valid),
                         "target_realization_reviewed_slots": reviewed,
                         "target_realization_pass": states["PASS"], "target_realization_hold": states["HOLD"],
                         "target_realization_other": sum(n for status, n in states.items() if status not in {"PASS", "HOLD", "NOT_REVIEWED"}),
                         "target_realization_not_reviewed": states["NOT_REVIEWED"],
                         "target_realization_pass_rate_among_reviewed": ratio(states["PASS"], reviewed),
                         "target_realization_pass_per_content_valid_slot": ratio(states["PASS"], len(valid)),
                         "author_plan_target_status_counts": dict(plan_states)}
    return by_slot


def reason_counts(rows, field):
    counts = collections.Counter()
    locations = collections.defaultdict(set)
    for row in rows:
        reasons = ([{"stage": row.get("stage", "unknown"), "reason": row.get("reason", "unspecified")}]
                   if field == "_failure_event_reasons" else row.get(field, []))
        for reason in reasons:
            key = (reason.get("stage", "review"), reason.get("reason", "unspecified")) if isinstance(reason, dict) else ("review", str(reason))
            counts[key] += 1
            locations[key].add((row.get("root_id"), row.get("slot_id")))
    return [{"stage": stage, "reason": reason, "count": count,
             "root_slot_count": len(locations[(stage, reason)]),
             "root_slot_ids": [f"{root}/{slot}" for root, slot in sorted(locations[(stage, reason)], key=lambda x: (str(x[0]), str(x[1])))]}
            for (stage, reason), count in sorted(counts.items())]


def metrics(rows):
    valid = [r for r in rows if r["strict"]["valid"]]
    returned = [r for r in rows if r["raw_status"] == "RETURNED"]
    mc = [r for r in rows if r["strict"]["question_type"] == "multi_select"]
    ordering = [r for r in rows if r["strict"]["question_type"] == "ordering"]
    vordering = [r for r in ordering if r["strict"]["valid"]]
    pairs = sum(r["strict"]["pairwise_total"] for r in ordering)
    vpairs = sum(r["strict"]["pairwise_total"] for r in vordering)
    return {"requests": len(rows), "returned_requests": len(returned), "nonresponses": len(rows)-len(returned),
            "semantic_questions": len({(r["strict"]["root_id"], r["slot_id"], r["arm"]) for r in rows}),
            "strict_correct": sum(r["strict"]["exact"] for r in rows),
            "strict_exact": ratio(sum(r["strict"]["exact"] for r in rows), len(rows)),
            "contract_valid_requests": len(valid), "returned_contract_invalid": sum(not r["strict"]["valid"] for r in returned),
            "returned_contract_valid_requests": sum(r["strict"]["valid"] for r in returned),
            "returned_contract_validity_rate": ratio(sum(r["strict"]["valid"] for r in returned), len(returned)),
            "valid_conditioned_strict_exact": ratio(sum(r["strict"]["exact"] for r in valid), len(valid)),
            "answer_only_correct": sum(bool(r["answer_only"]["exact"]) for r in returned),
            "answer_only_exact": ratio(sum(bool(r["answer_only"]["exact"]) for r in returned), len(returned)),
            "usable_final_answers": sum(r["answer_only"]["usable"] for r in returned),
            "mc_strict_original_rule": mc_metrics(mc),
            "mc_contract_valid_only": mc_metrics([r for r in mc if r["strict"]["valid"]]),
            "mc_answer_only_returned": mc_metrics([r for r in mc if r["raw_status"] == "RETURNED"], answer=True),
            "ordering_requests": len(ordering), "ordering_strict_correct": sum(r["strict"]["exact"] for r in ordering),
            "ordering_strict_exact": ratio(sum(r["strict"]["exact"] for r in ordering), len(ordering)),
            "ordering_answer_only_exact": ratio(sum(bool(r["answer_only"]["exact"]) for r in ordering if r["raw_status"] == "RETURNED"),
                                                 sum(r["raw_status"] == "RETURNED" for r in ordering)),
            "ordering_pairwise_correct": sum(r["strict"]["pairwise_correct"] for r in ordering),
            "ordering_pairwise_total": pairs,
            "ordering_pairwise_strict_original_rule": ratio(sum(r["strict"]["pairwise_correct"] for r in ordering), pairs),
            "ordering_pairwise_contract_valid_correct": sum(r["strict"]["pairwise_correct"] for r in vordering),
            "ordering_pairwise_contract_valid_total": vpairs,
            "ordering_pairwise_contract_valid_only": ratio(sum(r["strict"]["pairwise_correct"] for r in vordering), vpairs),
            "presentation_consistency": consistency(rows)}


def percentile(sorted_values, probability):
    index = (len(sorted_values)-1)*probability
    lo, hi = math.floor(index), math.ceil(index)
    return sorted_values[lo]*(hi-index) + sorted_values[hi]*(index-lo) if lo != hi else sorted_values[lo]


def bootstrap(values, enabled=True):
    result = {"eligible_roots": len(values), "point_estimate_F_minus_G": mean(values), "resamples": BOOTSTRAPS,
              "seed": SEED, "unit": "root", "interval_method": "two-sided percentile 95%, linear quantile",
              "ci95": None, "status": "NOT_ESTIMABLE_NO_ELIGIBLE_ROOTS"}
    if not enabled:
        return {**result, "status": "DESCRIPTIVE_ORIGINAL4_NO_NEW_CONFIRMATION"}
    if not values:
        return result
    rng = random.Random(SEED)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(BOOTSTRAPS))
    return {**result, "ci95": [percentile(draws, .025), percentile(draws, .975)],
            "status": "ESTIMATED" if n > 1 else "ONE_ROOT_DEGENERATE_INTERVAL_NO_REPLICATION_CLAIM"}


def cells(records):
    grouped = collections.defaultdict(list)
    for r in records:
        grouped[(r["strict"]["root_id"], r["slot_id"], r["arm"])].append(r)
    return grouped


def common_slots(records):
    grouped = cells(records)
    possible = {(root, slot) for root, slot, arm in grouped}
    return {(root, slot) for root, slot in possible if all(
        len(grouped[(root, slot, arm)]) == 2 and
        {r["strict"]["variant_id"] for r in grouped[(root, slot, arm)]} == set(VARIANTS) and
        all(r["content_valid_frozen"] and r["raw_status"] == "RETURNED" for r in grouped[(root, slot, arm)])
        for arm in ARMS)}


def group_analysis(records, roots, group, bootstrap_enabled):
    allowed = set(GROUP_SLOTS[group])
    matched = {key for key in common_slots(records) if key[1] in allowed}
    chosen = [r for r in records if (r["strict"]["root_id"], r["slot_id"]) in matched]
    root_rows = []
    for root in roots:
        used_slots = sorted(slot for rid, slot in matched if rid == root)
        row = {"root_id": root, "group": group, "planned_common_slots": len(allowed), "eligible_common_slots": len(used_slots),
               "common_slot_ids": used_slots, "complete_group_root": len(used_slots) == len(allowed)}
        for arm in ARMS:
            rs = [r for r in chosen if r["strict"]["root_id"] == root and r["arm"] == arm]
            row[arm + "_requests"] = len(rs)
            row[arm + "_strict_correct"] = sum(r["strict"]["exact"] for r in rs)
            row[arm + "_strict_exact"] = mean([int(r["strict"]["exact"]) for r in rs])
            row[arm + "_answer_only_exact"] = mean([int(r["answer_only"]["exact"]) for r in rs])
        for key in ("strict_exact", "answer_only_exact"):
            row["F_minus_G_" + key] = row["F_" + key] - row["G_" + key] if used_slots else None
        root_rows.append(row)
    eligible = [row for row in root_rows if row["eligible_common_slots"]]
    complete = [row for row in eligible if row["complete_group_root"]]
    arm_metrics = {arm: metrics([r for r in chosen if r["arm"] == arm]) for arm in ARMS}
    def endpoint(rs, key):
        result = bootstrap([row["F_minus_G_" + key] for row in rs], bootstrap_enabled)
        result["F_root_equal_accuracy"] = mean([row["F_" + key] for row in rs])
        result["G_root_equal_accuracy"] = mean([row["G_" + key] for row in rs])
        return result
    return {"group": group, "planned_roots": len(roots), "eligible_roots": len(eligible),
            "missing_root_ids": [r["root_id"] for r in root_rows if not r["eligible_common_slots"]],
            "planned_common_slots": len(roots)*len(allowed), "eligible_common_slots": len(matched),
            "paired_slot_coverage": ratio(len(matched), len(roots)*len(allowed)),
            "strict_root_equal_endpoint": endpoint(eligible, "strict_exact"),
            "answer_only_same_cohort_endpoint": endpoint(eligible, "answer_only_exact"),
            "complete_group_roots_sensitivity": {"eligible_roots": len(complete), "root_ids": [r["root_id"] for r in complete],
                                                 "strict": endpoint(complete, "strict_exact"),
                                                 "answer_only": endpoint(complete, "answer_only_exact")},
            "all_available_common_slot_pooled_descriptive": arm_metrics,
            "pooled_F_minus_G_strict": arm_metrics["F"]["strict_exact"] - arm_metrics["G"]["strict_exact"] if chosen else None,
            "per_root": root_rows}


def cohort_analysis(name, roots, records):
    records = [r for r in records if r["strict"]["root_id"] in set(roots)]
    returned = [r for r in records if r["raw_status"] == "RETURNED"]
    groups = {group: group_analysis(records, roots, group, name != "original4") for group in GROUPS}
    strata = []
    matched = common_slots(records)
    for scope, rs in (("paired_complete_returned", [r for r in records if (r["strict"]["root_id"], r["slot_id"]) in matched]),
                      ("per_arm_returned_descriptive", returned)):
        for dimension in ("root_id", "task_operation", "gold_count", "variant_id", "slot_id"):
            def value(r):
                return r["slot_id"] if dimension == "slot_id" else r["strict"][dimension]
            for v in sorted({value(r) for r in rs}, key=str):
                strata.append({"scope": scope, "dimension": dimension, "value": v,
                               **{arm: metrics([r for r in rs if r["arm"] == arm and value(r) == v]) for arm in ARMS}})
    return {"role": {"new16": "PRIMARY_REPLICATION", "original4": "READ_ONLY_EXPLORATORY_BACKGROUND", "combined20": "SECONDARY_POOLED_EXTENSION"}[name],
            "planned_roots": roots, "planned_semantic_questions": len(roots)*16, "planned_solver_presentations": len(roots)*32,
            "scored_frozen_requests": len(records), "returned_scored_requests": len(returned),
            "groups": groups, "strata": strata,
            "per_arm_returned_descriptive": {group: {arm: metrics([r for r in returned if r["arm"] == arm and r["slot_id"] in GROUP_SLOTS[group]])
                                                      for arm in ARMS} for group in GROUPS},
            "legacy_all_frozen_diagnostic": {"policy": "Reproduces existing SCORES strict fields, including legacy zero for backend nonresponse; never primary.",
                                             "groups": {group: {arm: metrics([r for r in records if r["arm"] == arm and r["slot_id"] in GROUP_SLOTS[group]])
                                                                for arm in ARMS} for group in GROUPS}}}


def keyed_candidates(doc):
    rows = doc if isinstance(doc, list) else (doc or {}).get("candidates", [])
    return {c["candidate_id"]: c for c in rows}


def read_experiment(directory, roots, inputs, required=False):
    """Read only sealed raw outputs; pending stages remain completion deficits."""
    out = directory / "execution"
    scores = inputs.read(out / "SCORES.json", optional=not required)
    freeze = inputs.read(out / "RAW_RESPONSE_FREEZE.json", optional=not required)
    public_doc = inputs.read(out / "PUBLIC_REQUESTS.json", optional=not required)
    private_doc = inputs.read(out / "PRIVATE_SCORING.json", optional=not required)
    content_freeze = inputs.read(directory / "CONTENT_FREEZE.json", optional=not required)
    candidates_doc = inputs.read(directory / "CANDIDATES_PRIVATE.json", optional=not required)
    final_review = inputs.read(directory / "CONTENT_REVIEW_AI.json", optional=not required)
    initial_candidates_doc = inputs.read(directory / "CANDIDATES_INITIAL_PRIVATE.json", optional=True)
    initial_review = inputs.read(directory / "CONTENT_REVIEW_INITIAL.json", optional=True)
    initial_gate = inputs.read(directory / "CONTENT_GATE_INITIAL.json", optional=True)
    if initial_gate is None:
        initial_gate = inputs.read(directory / "CONTENT_INITIAL_GATE.json", optional=True)
    final_gate = inputs.read(directory / "CONTENT_GATE_FINAL.json", optional=True)
    target_doc = inputs.read(directory / "TARGET_PRE_SOLVER_FREEZE.json", optional=True)
    failure_doc = inputs.read(directory / "SLOT_FAILURES.json", optional=True)
    feedback_ledger_doc = inputs.read(directory / "question_authoring" / "F" / "FEEDBACK_CONSUMPTION_LEDGER.json", optional=True)
    source_doc = inputs.read(directory / "SOURCE_SELECTION_FINAL.json", optional=True)
    status_doc = inputs.read(directory / "TASK_STATUS.json", optional=True)
    inputs.read(directory / "CONFIG.json")
    if content_freeze:
        for b in content_freeze["bindings"]:
            inputs.check(b)
    public_rows = (public_doc or {}).get("requests", [])
    public = {r["request_id"]: r for r in public_rows}
    need(len(public) == len(public_rows), "DUPLICATE_PUBLIC_REQUESTS")
    private_rows = (private_doc or {}).get("requests", [])
    private = {r["request_id"]: r for r in private_rows}
    need(len(private) == len(private_rows), "DUPLICATE_PRIVATE_REQUESTS")
    candidates = keyed_candidates(candidates_doc)
    initial_candidates = keyed_candidates(initial_candidates_doc)
    reviews = {r["candidate_id"]: r for r in (final_review or {}).get("reviews", [])}
    initial_reviews = {r["candidate_id"]: r for r in (initial_review or {}).get("reviews", [])}
    gate_rows = (final_gate or {}).get("records", (final_gate or {}).get("validations", []))
    admitted = {r["candidate_id"]: bool(r.get("eligible_for_solver", r.get("admitted", False))) for r in gate_rows}
    gate_by_id = {r["candidate_id"]: r for r in gate_rows}
    initial_admitted = {r["candidate_id"]: bool(r["admitted"]) for r in (initial_gate or {}).get("validations", [])}
    targets = {r["candidate_id"]: target_status(r.get("target_realization")) for r in (target_doc or {}).get("records", [])}
    feedback_targets = {r["candidate_id"]: r.get("target_status") for r in (feedback_ledger_doc or {}).get("entries", [])}
    failure_rows = (failure_doc or {}).get("failures", []) if isinstance(failure_doc, dict) else failure_doc or []
    failures_by_cell = collections.defaultdict(list)
    for failure in failure_rows:
        if not isinstance(failure, dict):
            continue
        failure_key = (failure.get("root_id"), failure.get("arm"), failure.get("slot_id"))
        failures_by_cell[failure_key].append({"stage": str(failure.get("stage", "unknown")),
                                              "reason": str(failure.get("reason", "unspecified"))})
    raw_by_id = {}
    if freeze:
        need(public_doc is not None, "RAW_FREEZE_WITHOUT_PUBLIC_REQUESTS")
        need(freeze["request_ids"] == list(public) and freeze["requests"] == len(public), "RAW_FREEZE_REQUEST_SET_DRIFT")
        raw_paths = {str((out / "raw" / (rid + ".json")).resolve()) for rid in public}
        bindings = freeze["bindings"]
        need(len(bindings) == len(raw_paths) and {str(Path(b["path"]).resolve()) for b in bindings} == raw_paths, "RAW_FREEZE_BINDING_SET_DRIFT")
        for b in bindings:
            inputs.check(b)
        for rid, row in public.items():
            raw = inputs.read(out / "raw" / (rid + ".json"))
            need(raw["request_id"] == rid and raw["status"] in ("RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED"), "RAW_ID_OR_STATUS_DRIFT")
            need((raw["status"] != "NOT_SUBMITTED") == (rid in set(freeze["submitted_ids"])), "SUBMISSION_STATUS_DRIFT")
            raw_by_id[rid] = raw
        need(sum(raw["status"] == "RETURNED" for raw in raw_by_id.values()) == freeze["returned"], "RAW_RETURNED_COUNT_DRIFT")
    if scores:
        need(freeze is not None and private_doc is not None and content_freeze is not None, "SCORES_WITHOUT_REQUIRED_FREEZES")
        same_binding(scores["raw_freeze_binding"], out / "RAW_RESPONSE_FREEZE.json")
        same_binding(scores["private_scoring_binding"], out / "PRIVATE_SCORING.json")
        frozen_content_paths = {str(Path(b["path"]).resolve()) for b in content_freeze["bindings"]}
        need(all(str((directory / name).resolve()) in frozen_content_paths for name in
                 ("CANDIDATES_PRIVATE.json", "CONTENT_REVIEW_AI.json", "SOURCE_SELECTION_FINAL.json")),
             "PRIMARY_CONTENT_INPUT_NOT_FROZEN")
        for key in ("candidate_binding", "review_binding", "selection_binding"):
            if key in private_doc:
                inputs.check(private_doc[key])
        for key, name in (("candidate_binding", "CANDIDATES_PRIVATE.json"), ("review_binding", "CONTENT_REVIEW_AI.json"),
                          ("selection_binding", "SOURCE_SELECTION_FINAL.json")):
            need(key in private_doc, "MISSING_PRIVATE_CONTENT_BINDING:" + key)
            same_binding(private_doc[key], directory / name)
        records = scores["records"]
        need(len(records) == len(public) == len(private) and {r["request_id"] for r in records} == set(public) == set(private), "SCORED_REQUEST_SET_DRIFT")
    else:
        records = []
    normalized = []
    seen_cells = set()
    for record in records:
        rid = record["request_id"]
        row, raw, pres = public[rid], raw_by_id[rid], private[rid]["presentation"]
        need(row["root_id"] in roots and row["slot_id"] in SLOTS and row["arm"] in ARMS, "UNPLANNED_SCORED_CELL")
        key = (row["root_id"], row["slot_id"], row["arm"], row["variant_id"])
        need(key not in seen_cells, "DUPLICATE_SCORED_CELL")
        seen_cells.add(key)
        same_binding(record["raw_binding"], out / "raw" / (rid + ".json"))
        cid = row["candidate_id"]
        need(cid in candidates and cid in reviews and reviews[cid]["decision"] == "PASS", "FROZEN_REQUEST_NOT_CONTENT_PASS:" + cid)
        need(admitted.get(cid, False), "FROZEN_REQUEST_FAILED_CONTENT_GATE:" + cid)
        for k in ("root_id", "arm", "slot_id"):
            need(candidates[cid][k] == row[k], "CONTENT_PUBLIC_CELL_DRIFT:" + k)
        validate_strict(record, row, pres)
        if raw["status"] != "RETURNED":
            need(not record["strict"]["valid"] and not record["strict"]["exact"], "NONRESPONSE_SCORED_AS_VALID")
        diagnostic = answer_only(raw, row, pres)
        if record["strict"]["valid"]:
            need(diagnostic["usable"] and diagnostic["exact"] == record["strict"]["exact"], "STRICT_ANSWER_DIAGNOSTIC_DRIFT")
        normalized.append({"request_id": rid, "arm": row["arm"], "slot_id": row["slot_id"], "strict": record["strict"],
                           "raw_status": raw["status"], "finish_reason": raw.get("finish_reason"),
                           "content_valid_frozen": True, "target_realization": targets.get(cid, "NOT_REVIEWED"),
                           "target_plan_status": feedback_targets.get(cid), "answer_only": diagnostic})
    public_cells = collections.defaultdict(list)
    candidate_cells = {}
    for row in public_rows:
        key = (row["root_id"], row["slot_id"], row["arm"])
        need(key[0] in roots and key[1] in SLOTS and key[2] in ARMS, "UNPLANNED_PUBLIC_CELL")
        public_cells[key].append(row)
    for candidate in candidates.values():
        key = (candidate["root_id"], candidate["slot_id"], candidate["arm"])
        need(key not in candidate_cells, "DUPLICATE_FINAL_CONTENT_CELL")
        candidate_cells[key] = candidate
    initial_cells = {(c["root_id"], c["slot_id"], c["arm"]): c for c in initial_candidates.values()}
    score_ids = {r["request_id"] for r in normalized}
    scored_by_id = {r["request_id"]: r for r in normalized}
    source_roots = {r["root_id"] for r in (source_doc or {}).get("roots", [])}
    ledger = []
    for root in roots:
        for slot in SLOTS:
            for arm in ARMS:
                key = (root, slot, arm)
                c, initial = candidate_cells.get(key), initial_cells.get(key)
                cid = c["candidate_id"] if c else None
                initial_cid = initial["candidate_id"] if initial else cid
                final_decision = reviews.get(cid, {}).get("decision")
                initial_decision = initial_reviews.get(initial_cid, {}).get("decision")
                content_valid = final_decision == "PASS" and admitted.get(cid, False)
                review = reviews.get(cid, {})
                gate_row = gate_by_id.get(cid, {})
                rejection_reasons = list(failures_by_cell.get(key, []))
                if not content_valid:
                    if cid is None:
                        if not rejection_reasons:
                            rejection_reasons.append({"stage": "finalize", "reason": "no_final_candidate"})
                    else:
                        if final_decision != "PASS":
                            rejection_reasons.append({"stage": "content_review", "reason": f"review_decision:{final_decision or 'missing'}"})
                        origin = review.get("effective_decision_origin")
                        if origin:
                            rejection_reasons.append({"stage": "content_review", "reason": f"effective_decision_origin:{origin}"})
                        for name, check in review.get("checks", {}).items():
                            if isinstance(check, dict) and check.get("decision") != "PASS":
                                rejection_reasons.append({"stage": "content_review", "reason": f"check:{name}:{check.get('decision')}:{check.get('rationale', '')}"})
                        rejection_reasons.extend({"stage": "content_review", "reason": f"issue:{issue}"}
                                                 for issue in review.get("issues", []))
                        rejection_reasons.extend({"stage": "content_gate", "reason": f"review_error:{error}"}
                                                 for error in gate_row.get("review_validation", {}).get("review_errors", []))
                        duplicate = gate_row.get("duplicate_disposition", {})
                        if duplicate.get("status") not in (None, "UNIQUE"):
                            rejection_reasons.append({"stage": "content_gate", "reason": f"duplicate:{duplicate.get('status')}:{duplicate.get('kept_candidate_id')}"})
                        if not rejection_reasons:
                            rejection_reasons.append({"stage": "content_gate", "reason": "candidate_not_admitted"})
                rejection_reasons = list({(r["stage"], r["reason"]): r for r in rejection_reasons}.values())
                rs = public_cells.get(key, [])
                statuses = collections.Counter(raw_by_id[r["request_id"]]["status"] if r["request_id"] in raw_by_id else "UNFROZEN_PENDING" for r in rs)
                state = "not_started" if not c else "content_pending"
                if initial_decision in ("FAIL", "HOLD") and not final_decision:
                    state = "draft_failed"
                if final_decision in ("FAIL", "HOLD") or (cid in admitted and not admitted[cid]):
                    state = "content_failed"
                elif content_valid:
                    state = "target_hold" if targets.get(cid) == "HOLD" else "content_valid"
                if rs:
                    state = "frozen"
                    if statuses["RETURNED"]:
                        state = "returned_scored" if all(r["request_id"] in score_ids for r in rs) else "returned_unscored"
                    elif statuses["BACKEND_ERROR"]:
                        state = "backend_failed_no_response"
                    elif statuses["NOT_SUBMITTED"]:
                        state = "request_not_submitted"
                ledger.append({"root_id": root, "slot_id": slot, "arm": arm, "planned_presentations": 2,
                               "source_available": root in source_roots, "candidate_id": cid, "initial_draft_present": initial is not None,
                               "initial_content_decision": initial_decision,
                               "initial_content_valid": initial_admitted.get(initial_cid, initial_decision == "PASS" if initial_decision else None),
                               "final_content_decision": final_decision, "final_content_valid": content_valid,
                               "target_realization": targets.get(cid, "NOT_REVIEWED"),
                               "target_plan_status": feedback_targets.get(cid),
                               "rejection_reasons": rejection_reasons,
                               "frozen_requests": len(rs),
                               "raw_status_counts": dict(statuses), "scored_requests": sum(r["request_id"] in score_ids for r in rs),
                               "returned_contract_invalid": sum(r["request_id"] in scored_by_id and
                                   scored_by_id[r["request_id"]]["raw_status"] == "RETURNED" and
                                   not scored_by_id[r["request_id"]]["strict"]["valid"] for r in rs),
                               "state": state})
    completion = {"experiment_directory": str(directory.resolve()), "planned_roots": len(roots),
                  "planned_semantic_questions": len(roots)*16, "planned_solver_presentations": len(roots)*32,
                  "scores_available": scores is not None, "raw_freeze_available": freeze is not None,
                  "public_requests": len(public_rows), "scored_requests": len(records),
                  "raw_status_counts": dict(collections.Counter(r["status"] for r in raw_by_id.values())),
                  "unavailable_source_root_ids": sorted(set(roots)-source_roots),
                  "raw_wall_seconds": (freeze or {}).get("wall_seconds"), "runner_failure": (freeze or {}).get("failure"),
                  "task_status_snapshot": status_doc, "by_arm": {}, "planned_slot_ledger": ledger}
    for arm in ARMS:
        ls = [r for r in ledger if r["arm"] == arm]
        completion["by_arm"][arm] = {"planned_semantic_slots": len(ls), "initial_drafts_present": sum(r["initial_draft_present"] for r in ls),
                                      "initial_valid_slots": sum(r["initial_content_valid"] is True for r in ls),
                                      "initial_review_unknown_slots": sum(r["initial_content_valid"] is None for r in ls),
                                      "final_content_valid_slots": sum(r["final_content_valid"] for r in ls),
                                      "frozen_semantic_slots": sum(r["frozen_requests"] > 0 for r in ls),
                                      "frozen_presentations": sum(r["frozen_requests"] for r in ls),
                                      "returned_presentations": sum(r["raw_status_counts"].get("RETURNED", 0) for r in ls),
                                      "planned_presentation_completion_deficit": 2*len(ls) - sum(r["raw_status_counts"].get("RETURNED", 0) for r in ls),
                                      "content_or_request_freeze_deficit_presentations": 2*len(ls) - sum(r["frozen_requests"] for r in ls),
                                      "backend_failed_no_response": sum(r["raw_status_counts"].get("BACKEND_ERROR", 0) for r in ls),
                                      "request_not_submitted": sum(r["raw_status_counts"].get("NOT_SUBMITTED", 0) for r in ls),
                                      "returned_contract_invalid": sum(r["returned_contract_invalid"] for r in ls),
                                      "state_counts": dict(collections.Counter(r["state"] for r in ls)),
                                      "target_realization_by_slot": target_summary(ledger, arm),
                                      "rejected_slot_reason_counts": reason_counts([r for r in ls if not r["final_content_valid"]], "rejection_reasons"),
                                      "complete_returned_root_ids": [root for root in roots if all(
                                          r["raw_status_counts"].get("RETURNED", 0) == 2 for r in ls if r["root_id"] == root)]}
        completion["by_arm"][arm]["slot_failure_event_reason_counts"] = reason_counts(
            [r for r in failure_rows if r.get("arm") == arm], "_failure_event_reasons")
    if required:
        need(len(normalized) == len(roots)*32 and all(r["raw_status"] == "RETURNED" for r in normalized), "ORIGINAL4_FROZEN_COMPLETION_DRIFT")
    return normalized, completion


def pct(value):
    return "—" if value is None else f"{100*value:.2f}%"


def pp(value):
    return "—" if value is None else f"{100*value:+.2f}"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows], ""]


def render(doc):
    lines = ["# F/G 扩大复现：新16 / 原4 / 合并20", "", f"快照状态：{doc['status']}。分析时间：{doc['created_at_utc']}。", "",
             "主结果只用新16；原4只读复用，合并20为次要描述。F−G 为负表示 F 更难。MC6 全选题不进入主比较。", "",
             "同 root/槽两臂各两次真实 RETURNED 才进入配对；后台无响应、未提交、生成失败是缺额。真实返回但原合同不合格仍严格记错。每 root 先对共同槽求均值，再等权平均 root。", "",
             "## 配对结果与 root bootstrap", ""]
    rows = []
    for name, cohort in doc["cohorts"].items():
        for group in GROUPS:
            result = cohort["groups"][group]
            end = result["strict_root_equal_endpoint"]
            ci = end["ci95"]
            rows.append((name, GROUP_ZH[group], f"{result['eligible_roots']}/{len(cohort['planned_roots'])}",
                         f"{result['eligible_common_slots']}/{result['planned_common_slots']}", pct(end["F_root_equal_accuracy"]),
                         pct(end["G_root_equal_accuracy"]), pp(end["point_estimate_F_minus_G"]),
                         "—" if ci is None else f"[{pp(ci[0])}, {pp(ci[1])}]",
                         pp(result["answer_only_same_cohort_endpoint"]["point_estimate_F_minus_G"])))
    lines += table(("批次", "分组", "root", "共同槽", "F exact", "G exact", "F−G pp", "95%CI pp", "答案-only差 pp"), rows)
    lines += ["Bootstrap 按完整 root 配对重采样10000次，seed=20261025，两侧百分位95%区间；不把两次展示当独立样本。原4不作为新的确认检验。", "",
              "## 生产和运行缺额", ""]
    rows = []
    for name, complete in doc["completion"].items():
        for arm, x in complete["by_arm"].items():
            rows.append((name, arm, x["planned_semantic_slots"], x["initial_valid_slots"], x["initial_review_unknown_slots"],
                         x["final_content_valid_slots"], x["frozen_presentations"], x["returned_presentations"],
                         json.dumps(x["state_counts"], ensure_ascii=False)))
    lines += table(("批次", "臂", "计划语义槽", "初稿有效", "初审未知", "最终有效", "冻结展示", "真实返回", "槽状态"), rows)
    lines += ["未纳入配对的全部计划槽及原因保存在 JSON 的 completion.planned_slot_ledger；不能把缺额补记为 Solver 答错。初稿有效率的分母仍为计划槽，未知审核单列。", "",
              "## 目标实现与内容拒绝原因", "",
              "目标实现 PASS/HOLD 仅汇总冻结的目标审核；作者计划 TARGET_HOLD 单独列出，不作为未实现审核。目标实现率分母为已审核的有效内容槽；未审核项不当作未命中。", ""]
    target_rows = []
    cohort_ledgers = {
        "new16": doc["completion"]["new16"]["planned_slot_ledger"],
        "original4": doc["completion"]["original4"]["planned_slot_ledger"],
    }
    cohort_ledgers["combined20"] = cohort_ledgers["original4"] + cohort_ledgers["new16"]
    for cohort_name, ledger in cohort_ledgers.items():
        for arm in ARMS:
            summary = doc["target_realization_summary"][cohort_name][arm]
            for slot in ("ALL", *SLOTS):
                item = summary[slot]
                plans = item["author_plan_target_status_counts"]
                target_rows.append((cohort_name, arm, slot, item["planned_slots"], item["final_content_valid_slots"],
                                    item["target_realization_pass"], item["target_realization_hold"],
                                    item["target_realization_other"], item["target_realization_not_reviewed"],
                                    pct(item["target_realization_pass_rate_among_reviewed"]),
                                    json.dumps(plans, ensure_ascii=False, sort_keys=True)))
    lines += table(("批次", "臂", "槽", "计划", "内容有效", "目标 PASS", "目标 HOLD", "其他目标状态", "未审核", "PASS/已审核", "作者计划状态"), target_rows)
    lines += ["未通过或未产出的计划槽及阶段/拒绝理由保存在 JSON 的 rejected_slot_reason_counts 和 slot_failure_event_reason_counts。下表汇总冻结 failure 记录中的阶段与原始原因。", ""]
    failure_rows = []
    for cohort_name in ("new16", "original4"):
        for arm in ARMS:
            arm_doc = doc["completion"][cohort_name]["by_arm"][arm]
            failure_rows.append((cohort_name, arm, "rejected planned slots", sum(
                1 for row in doc["completion"][cohort_name]["planned_slot_ledger"]
                if row["arm"] == arm and not row["final_content_valid"]), ""))
            failure_rows.extend((cohort_name, arm, item["stage"], item["count"], item["reason"])
                                for item in arm_doc["slot_failure_event_reason_counts"])
            failure_rows.extend((cohort_name, arm, "rejection:" + item["stage"], item["count"], item["reason"])
                                for item in arm_doc["rejected_slot_reason_counts"])
    lines += table(("批次", "臂", "阶段", "次数/槽数", "原始拒绝原因"), failure_rows)
    lines += ["", "## 双排列一致性", ""]
    consistency_rows = []
    for name, cohort in doc["cohorts"].items():
        for arm, arm_metrics in cohort["groups"]["ALL"]["all_available_common_slot_pooled_descriptive"].items():
            item = arm_metrics["presentation_consistency"]
            denominator = item["complete_returned_semantic_questions"]
            consistency_rows.append((name, arm, denominator, item["both_exact"], item["at_least_one_exact"],
                                     item["both_contract_valid"], item["same_canonical_prediction"],
                                     pct(item["same_canonical_prediction_rate"])))
    lines += table(("批次", "臂", "两次均真实返回的语义题", "两次均答对", "至少一次答对", "两次均合同有效", "规范预测相同", "相同/均返回"), consistency_rows)
    lines += ["两次展示是同一语义题的相关重复测量；规范预测一致率以两次都返回为分母。", "",
              "## 每臂真实返回作答", "",
              "此处按各臂自身真实返回数作分母，臂间可能不等；格式有效率只在真实返回中计算。", ""]
    response_rows = []
    for name, cohort in doc["cohorts"].items():
        for group in ("NON_MC6_PRIMARY", "MC6_CONTROL", "ALL"):
            for arm, m in cohort["per_arm_returned_descriptive"][group].items():
                response_rows.append((name, GROUP_ZH[group], arm, m["returned_requests"],
                                      f"{m['returned_contract_valid_requests']}/{m['returned_requests']}",
                                      pct(m["returned_contract_validity_rate"]), pct(m["strict_exact"]),
                                      pct(m["answer_only_exact"])))
    lines += table(("批次", "分组", "臂", "真实返回", "合同有效/返回", "格式有效率", "strict exact", "答案-only"), response_rows)
    lines += ["",
              "## 共同完整返回队列的多选与排序", ""]
    rows = []
    for name, cohort in doc["cohorts"].items():
        for group in ("MC2_5_PRIMARY", "ORDERING", "MC6_CONTROL"):
            for arm, m in cohort["groups"][group]["all_available_common_slot_pooled_descriptive"].items():
                mc = m["mc_strict_original_rule"]
                rows.append((name, GROUP_ZH[group], arm, m["requests"], pct(m["strict_exact"]),
                             f"{m['returned_contract_valid_requests']}/{m['returned_requests']}", pct(m["returned_contract_validity_rate"]),
                             pct(m["answer_only_exact"]),
                             pct(mc["macro"]["precision"]), pct(mc["macro"]["recall"]), pct(mc["macro"]["f1"]),
                             f"{mc['false_positive_labels']}/{mc['false_negative_labels']}",
                             f"{m['ordering_pairwise_correct']}/{m['ordering_pairwise_total']}",
                             pct(m["ordering_pairwise_strict_original_rule"]), pct(m["ordering_pairwise_contract_valid_only"])))
    lines += table(("批次", "分组", "臂", "作答", "strict exact", "合同有效/返回", "格式有效率", "答案-only", "P", "R", "F1", "FP/FN", "排序对", "pairwise", "有效合同pairwise"), rows)
    lines += ["无效返回的多选 strict 预测按原规则为空集合（可能增加 FN）；无效排序的15对均记错。有效合同条件指标及答案-only指标另列。", "",
              "## 新16个 root 的主比较", ""]
    lines += table(("root", "共同主槽/7", "F exact", "G exact", "F−G pp", "答案-only差 pp"), [
        (r["root_id"], f"{r['eligible_common_slots']}/7", pct(r["F_strict_exact"]), pct(r["G_strict_exact"]),
         pp(r["F_minus_G_strict_exact"]), pp(r["F_minus_G_answer_only_exact"]))
        for r in doc["cohorts"]["new16"]["groups"]["NON_MC6_PRIMARY"]["per_root"]])
    lines += ["## 分析边界", "", *["- " + x for x in doc["interpretation_limits"]], "",
              "完整的分层、每臂不等分母描述、完整七槽 root 敏感性、原 strict legacy 统计和输入哈希见 REPLICATION_COMPARISON.json。每 root 差值见 REPLICATION_ROOT_DIFFERENCES.csv。", ""]
    return "\n".join(lines)


def csv_report(doc):
    rows = [{"cohort": name, **row} for name, cohort in doc["cohorts"].items() for group in GROUPS
            for row in cohort["groups"][group]["per_root"]]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "common_slot_ids": ";".join(row["common_slot_ids"])})
    return stream.getvalue()


def fixture(root, slot, arm, variant, exact, status="RETURNED", valid=True):
    gold = list("ABCDEF") if slot.startswith("ORDER") else list("ABCDEF"[:int(slot[2:])])
    pred = gold[:] if exact else (list(reversed(gold)) if slot.startswith("ORDER") else gold[:-1])
    if not valid or status != "RETURNED":
        pred, valid, exact = [], False, False
    positions = {x: i for i, x in enumerate(pred)}
    order = slot.startswith("ORDER")
    s = {"request_id": f"{root}-{slot}-{arm}-{variant}", "candidate_id": f"{root}-{slot}-{arm}", "root_id": root,
         "question_type": "ordering" if order else "multi_select", "task_operation": slot, "variant_id": variant,
         "valid": valid, "exact": exact, "prediction": pred, "gold": gold, "gold_count": None if order else len(gold),
         "false_positive": None if order else len(set(pred)-set(gold)),
         "false_negative": None if order else len(set(gold)-set(pred)), "pairwise_total": 15 if order else None,
         "pairwise_correct": sum(bool(valid and positions[a] < positions[b]) for i, a in enumerate(gold) for b in gold[i+1:]) if order else None}
    return {"request_id": s["request_id"], "arm": arm, "slot_id": slot, "strict": s, "raw_status": status,
            "content_valid_frozen": True, "answer_only": {"usable": valid, "exact": exact if status == "RETURNED" else None, "prediction": pred}}


def self_check():
    """In-memory fixtures cover missingness, equal-root weighting and mapping."""
    roots = ["synthetic-1", "synthetic-2", "synthetic-missing"]
    rs = []
    for root in roots[:2]:
        for slot in (("MC2",) if root == roots[0] else ("MC2", "MC3")):
            for arm in ARMS:
                for variant in VARIANTS:
                    rs.append(fixture(root, slot, arm, variant, root == roots[0] and arm == "F"))
    # An apparently favorable slot is excluded because one G presentation did not return.
    for arm in ARMS:
        for variant in VARIANTS:
            rs.append(fixture(roots[0], "MC4", arm, variant, arm == "F",
                              "BACKEND_ERROR" if arm == "G" and variant == VARIANTS[1] else "RETURNED"))
    # Returned malformed responses remain eligible and score zero.
    for arm in ARMS:
        for variant in VARIANTS:
            rs.append(fixture(roots[0], "MC6", arm, variant, False, valid=False))
    result = group_analysis(rs, roots, "NON_MC6_PRIMARY", True)
    need(result["eligible_common_slots"] == 3 and result["eligible_roots"] == 2, "FIXTURE_MISSINGNESS")
    end = result["strict_root_equal_endpoint"]
    need(end["point_estimate_F_minus_G"] == .5, "FIXTURE_ROOT_EQUAL_WEIGHT")
    need(result["pooled_F_minus_G_strict"] == 1/3, "FIXTURE_POOLED_DIFFERENCE")
    need(end["ci95"] == [0., 1.] and end["resamples"] == 10000, "FIXTURE_BOOTSTRAP")
    need(end == group_analysis(rs, roots, "NON_MC6_PRIMARY", True)["strict_root_equal_endpoint"], "FIXTURE_DETERMINISM")
    need(group_analysis(rs, roots, "MC6_CONTROL", False)["eligible_common_slots"] == 1, "FIXTURE_INVALID_RETURN_STAYS")
    need(result["missing_root_ids"] == [roots[2]], "FIXTURE_MISSING_ROOT_NO_ZERO")
    pub = {"labels": list("ABCDEF"), "question_type": "multi_select"}
    pres = {"display_to_canonical": dict(zip("ABCDEF", "FABCDE")), "canonical_gold": ["F", "A"]}
    raw = {"status": "RETURNED", "raw_output": '{"final_answer":"<box>A B</box>"}', "finish_reason": "length"}
    need(answer_only(raw, pub, pres)["exact"], "FIXTURE_FINAL_ONLY_AND_MAPPING")
    for bad in ('{"final_answer":"<box>A A</box>"}', '{"final_answer":"<box>A,B</box>"}',
                '{"final_answer":"<box>a b</box>"}', '{"final_answer":"<box>A B</box>"'):
        need(not answer_only({"status": "RETURNED", "raw_output": bad}, pub, pres)["usable"], "FIXTURE_NO_LEGALITY_RELAXATION")
    need(not answer_only({**raw, "raw_output": '{"final_answer":"<box>A B</box>"}'},
                         {**pub, "question_type": "ordering"}, pres)["usable"], "FIXTURE_INCOMPLETE_ORDERING")
    pair = [fixture("order", "ORDER1", "F", v, False, valid=False) for v in VARIANTS]
    m = metrics(pair)
    need(m["ordering_pairwise_total"] == 30 and m["ordering_pairwise_strict_original_rule"] == 0 and
         m["ordering_pairwise_contract_valid_total"] == 0 and m["ordering_pairwise_contract_valid_only"] is None,
         "FIXTURE_PAIRWISE_DENOMINATORS")
    need(m["returned_contract_valid_requests"] == 0 and m["returned_contract_validity_rate"] == 0.,
         "FIXTURE_RETURNED_FORMAT_VALIDITY")
    need(m["presentation_consistency"]["complete_returned_semantic_questions"] == 1, "FIXTURE_CONSISTENCY")
    targets = target_summary([
        {"arm": "F", "slot_id": "MC2", "final_content_valid": True, "target_realization": "PASS", "target_plan_status": "PROSPECTIVE_PROBE"},
        {"arm": "F", "slot_id": "MC2", "final_content_valid": True, "target_realization": "HOLD", "target_plan_status": "TARGET_HOLD"},
        {"arm": "F", "slot_id": "MC2", "final_content_valid": True, "target_realization": "NOT_REVIEWED", "target_plan_status": "PROSPECTIVE_PROBE"},
        {"arm": "F", "slot_id": "MC2", "final_content_valid": False, "target_realization": "PASS", "target_plan_status": "PROSPECTIVE_PROBE"}], "F")["ALL"]
    need(targets["final_content_valid_slots"] == 3 and targets["target_realization_pass"] == 1 and
         targets["target_realization_hold"] == 1 and targets["target_realization_not_reviewed"] == 1 and
         targets["target_realization_pass_rate_among_reviewed"] == .5, "FIXTURE_TARGET_DENOMINATOR")
    reasons = reason_counts([{"root_id": "r1", "slot_id": "MC2", "stage": "draft", "reason": "no candidate"}],
                            "_failure_event_reasons")
    need(len(reasons) == 1 and reasons[0]["count"] == 1 and reasons[0]["stage"] == "draft",
         "FIXTURE_REJECTION_REASON_AGGREGATION")
    print(json.dumps({"status": "PASS", "fixtures": ["missingness_not_zero", "root_equal_weight", "bootstrap_10000_seed20261025",
                                                      "invalid_return_included", "answer_box_legality", "canonical_mapping", "pairwise_denominators",
                                                      "returned_format_validity", "target_denominator", "rejection_reasons", "consistency"],
                      "models_called": False, "real_inputs_read": False, "files_written": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=P)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this script's three derived reports.")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    inputs = Inputs()
    plan = inputs.read(P / "ANALYSIS_PLAN.json")
    selection = inputs.read(P / "SELECTION_FREEZE.json")
    config = inputs.read(P / "CONFIG.json")
    need(plan["bootstrap"]["resamples"] == BOOTSTRAPS and plan["bootstrap"]["seed"] == SEED, "FROZEN_BOOTSTRAP_DRIFT")
    new_roots = [root_id(r) for r in selection["new_roots"]]
    old_roots = [root_id(r) for r in selection["original_roots"]]
    need(len(new_roots) == 16 and len(old_roots) == 4 and len(set(new_roots+old_roots)) == 20, "FROZEN_COHORT_DRIFT")
    need(new_roots == [root_id(r) for r in config["roots"]] and old_roots == [root_id(r) for r in plan["original_roots_read_only"]], "CONFIG_SELECTION_DRIFT")
    need([slot["slot"] for slot in plan["slots_per_root_per_arm"]] == list(SLOTS), "FROZEN_SLOTS_DRIFT")
    old_directory = (P / plan["original_experiment"]).resolve()
    need(old_directory == Path(config["original_experiment"]).resolve(), "ORIGINAL_EXPERIMENT_DRIFT")
    old_records, old_complete = read_experiment(old_directory, old_roots, inputs, required=True)
    new_records, new_complete = read_experiment(P, new_roots, inputs)
    records = old_records+new_records
    cohorts = {name: cohort_analysis(name, roots, records) for name, roots in
               (("new16", new_roots), ("original4", old_roots), ("combined20", old_roots+new_roots))}
    target_ledgers = {"new16": new_complete["planned_slot_ledger"],
                      "original4": old_complete["planned_slot_ledger"]}
    target_ledgers["combined20"] = target_ledgers["original4"] + target_ledgers["new16"]
    target_realization_summary = {name: {arm: target_summary(ledger, arm) for arm in ARMS}
                                  for name, ledger in target_ledgers.items()}
    old_primary = cohorts["original4"]["groups"]["NON_MC6_PRIMARY"]["all_available_common_slot_pooled_descriptive"]
    need(old_primary["F"]["strict_correct"] == 12 and old_primary["G"]["strict_correct"] == 8 and
         old_primary["F"]["requests"] == old_primary["G"]["requests"] == 56, "ORIGINAL4_STRICT_REPRODUCTION_DRIFT")
    complete = len(new_records) == 512 and all(r["raw_status"] == "RETURNED" for r in new_records)
    doc = {"schema_version": "fg_replication_root_paired_analysis_v1", "experiment_id": P.name,
           "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "COMPLETE_FULL_PLANNED_SAMPLE" if complete else "PARTIAL_PLANNED_SAMPLE_NO_IMPUTATION",
           "training": False, "models_called_by_analysis": False,
           "difference_direction": "F minus G; positive means F easier and negative means F harder for this frozen Solver",
           "primary_cohort": "new16", "primary_group": "NON_MC6_PRIMARY", "cohorts": cohorts,
           "completion": {"new16": new_complete, "original4": old_complete},
           "target_realization_summary": target_realization_summary,
           "response_diagnostics": [{"request_id": r["request_id"], "root_id": r["strict"]["root_id"], "arm": r["arm"],
                                     "slot_id": r["slot_id"], "variant_id": r["strict"]["variant_id"], "raw_status": r["raw_status"],
                                     "finish_reason": r["finish_reason"], "target_realization": r["target_realization"],
                                     "target_plan_status": r["target_plan_status"],
                                     "strict_valid": r["strict"]["valid"], "strict_exact": r["strict"]["exact"],
                                     "answer_only": r["answer_only"]} for r in records],
           "metric_policies": {"primary_eligibility": plan["missingness_and_denominators"]["primary_common_slot_eligibility"],
                               "root_equal_weight": "Average two returned displays within each slot, common primary slots within root, then eligible roots equally.",
                               "target_hold": "Content-valid frozen TARGET_HOLD remains included.",
                               "answer_only": "Diagnostic only: exact original JSON final_answer/box/unique A-F label grammar and full six-item ordering; use private display-to-canonical mapping. Ignores evidence schema, finish_reason and capture contract penalties but does not repair JSON or parse free text. No final usable answer is zero only for real returned output.",
                               "legacy": "Original strict fields reproduced separately, including legacy nonresponse zeros. Excluded from primary.",
                               "pairwise": "Original strict: 15 gold pairs/returned six-item ordering, invalid returned answers 0/15. Contract-valid denominator separately shown.",
                               "bootstrap": plan["bootstrap"]},
           "interpretation_limits": [
               "新16是预冻结的训练来源 roots；父对话曾用于 M-clean，不是未见训练 root 泛化。",
               "原4曾用于发现卡片；不重跑、不删负结果，合并20不能作为独立确认。",
               "两臂共用反馈引导的扩充来源，无法独立归因扩充反馈收益；没有训练收益或弱点修复结论。",
               "F/G同 root/槽是不同题；两次排列是相关重复测量，不是独立样本。",
               "内容审核是不同上下文的 AI 审核，不是外部人工证明；TARGET_HOLD不按得分筛除。",
               "答案-only仍要求可解析的最终JSON和原box语法，无法从截断JSON或自由文本可靠恢复答案；它不验证解释或引用的语义正确。",
               "可见理由未必忠实反映隐藏思考；难度降低或升高单独不能证明命中指定弱点。",
               "置信区间只描述这组来源、卡库、固定模型及单次生成的 root 变异；跨零不能宣称等效。",
               "初稿/最终内容失败和后台非响应作为缺额，主结果按可用共同槽估计；缺失非随机仍可能影响解释。",
               "尚未完成的机制语义审核不在本自动分析中伪造为已完成。"],
           "input_bindings": sorted(inputs.bound.values(), key=lambda b: b["path"]),
           "absent_optional_inputs_at_snapshot": sorted(inputs.absent), "script_binding": binding(__file__)}
    inputs.recheck()
    output_dir = args.output_dir.resolve()
    need(output_dir == P or P in output_dir.parents, "OUTPUTS_MUST_STAY_INSIDE_NEW_EXPERIMENT")
    output_paths = [output_dir / name for name in OUTPUT_NAMES]
    if not args.overwrite:
        need(not any(path.exists() for path in output_paths), "NO_OUTPUT_OVERWRITE_USE_SEPARATE_OUTPUT_DIR_OR_EXPLICIT_FLAG")
    contents = (json.dumps(doc, ensure_ascii=False, indent=2)+"\n", render(doc), csv_report(doc))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in zip(output_paths, contents):
        if args.overwrite:
            temporary = path.with_name(path.name+".tmp")
            with temporary.open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
            temporary.replace(path)
        else:
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
    primary = cohorts["new16"]["groups"]["NON_MC6_PRIMARY"]
    print(json.dumps({"status": doc["status"], "outputs": [str(path) for path in output_paths],
                      "new16_primary": primary["strict_root_equal_endpoint"], "eligible_common_slots": primary["eligible_common_slots"],
                      "original4_reproduced": {"F_correct": 12, "G_correct": 8, "requests_per_arm": 56}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
