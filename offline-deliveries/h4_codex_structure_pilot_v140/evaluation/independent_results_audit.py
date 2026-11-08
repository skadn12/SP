"""CPU-only independent arithmetic audit. Imports no production scorer/report.

The official audit refuses to write until both frozen pilot native
COMPLETE/RESULTS files and the final report exist. Inputs are
read-only; the audit JSON is exclusively created, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parents[1]
SIDES = ("clean", "new")
SPLITS = ("all", "dev")
STRATA = ("all", "main_excluding_gold6", "multi_select", "multi_select_gold2345",
          "gold6_control", "gold2", "gold3", "gold4", "gold5", "gold6", "ordering",
          "new_structure", "old_structure_control", "new_multi_select", "old_multi_select",
          "new_ordering", "old_ordering", "new_gold3", "old_gold3",
          "ordering_scenes_3-2-1", "ordering_scenes_2-2-2", "ordering_scenes_4-2", "ordering_scenes_1-1-1-1-1-1")
PAIRS = (("clean", "new"),)
MODEL_IDENTITIES = {
    "clean": ("M_clean_policy", "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model"),
    "new": ("M_v125_newonly", "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_codex_combined_sft_v125_merged"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_id(text):
    return "content_" + hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def average(values):
    return sum(values, Fraction()) / len(values) if values else None


def score(qtype, gold, mapping, raw):
    """Derive displayed/canonical predictions and counts from raw text only."""
    values, reason = None, None
    if not isinstance(raw, str):
        reason = "raw_output_missing_or_not_text"
    else:
        match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", raw, re.S)
        if match is None:
            reason = "exactly_one_box_required"
        else:
            text = match.group(1).strip()
            values = [x.strip().upper() for x in text.split(",")] if text else []
            if any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", x) is None for x in values):
                values, reason = None, "malformed_id_list"
            elif len(set(values)) != len(values):
                values, reason = None, "duplicate_id"
    out = {"parse_status": "failure" if reason else "success",
           "parse_reason": reason, "scored": False,
           "display_prediction": values, "content_prediction": None,
           "content_gold": gold, "tp": None, "fp": None, "fn": None,
           "exact_match": None, "partial_score": None}
    if values is None:
        out["scoring_status"] = "PARSE_FAILURE"
        return out
    bad = [x for x in values if x not in mapping]
    if bad:
        out.update(scoring_status="INVALID_ID", invalid_ids=bad)
        return out
    pred = [mapping[x] for x in values]
    if len(set(pred)) != len(pred):
        out.update(scoring_status="DUPLICATE_CONTENT_ID", invalid_ids=[])
        return out
    if qtype == "ordering":
        if len(pred) != len(gold) or set(pred) != set(gold):
            out.update(scoring_status="INCOMPLETE_ORDER", content_prediction=pred)
            return out
        # Enumerate gold-ordered pairs, independently of native rank counting.
        ordered_pairs = {(a, b) for i, a in enumerate(pred) for b in pred[i + 1:]}
        tp = sum((a, b) in ordered_pairs for i, a in enumerate(gold) for b in gold[i + 1:])
        fp = len(gold) * (len(gold) - 1) // 2 - tp
        fn, exact, final = 0, pred == gold, pred
        partial = Fraction(tp - fp, tp + fp)
    else:
        p, g = set(pred), set(gold)
        tp, fp, fn = len(p & g), len(p - g), len(g - p)
        partial = Fraction(tp, len(g))
        if len(g) != len(mapping):
            partial -= Fraction(fp, len(mapping) - len(g))
        exact, final = p == g, sorted(p)
    out.update(scored=True, scoring_status="SCORED", content_prediction=final,
               tp=tp, fp=fp, fn=fn, exact_match=exact,
               partial_score=str(partial), partial_score_fraction=str(partial))
    return out


def tags(q):
    extra = {q["pilot_arm"], ("new" if q["pilot_arm"] == "new_structure" else "old") + "_" + q["question_type"]}
    if q["question_type"] == "ordering":
        counts = Counter(v["turn_ref"].rsplit(":T",1)[0] for v in q["order_anchor_spans"].values())
        sig = "-".join(map(str, sorted(counts.values(), reverse=True)))
        return {"all", "main_excluding_gold6", "ordering", "ordering_scenes_" + sig} | extra
    g = len(q["gold"])
    if g == 3:
        extra.add(("new" if q["pilot_arm"] == "new_structure" else "old") + "_gold3")
    return {"all", "multi_select", "gold" + str(g)} | extra | (
        {"gold6_control"} if g == 6 else {"main_excluding_gold6", "multi_select_gold2345"})


def bounded(rows, key, limits):
    known, lows, highs = [], [], []
    for row in rows:
        v = row.get(key)
        if v is None:
            a, b = limits(row)
            lows.append(Fraction(a))
            highs.append(Fraction(b))
        else:
            known.append(Fraction(v))
            lows.append(Fraction(v))
            highs.append(Fraction(v))
    point = average(known) if len(known) == len(rows) else None
    low, high = average(lows), average(highs)
    return {"mean": None if point is None else float(point),
            "mean_fraction": None if point is None else str(point),
            "lower": float(low), "upper": float(high),
            "lower_fraction": str(low), "upper_fraction": str(high),
            "planned_denominator": len(rows), "defined_count": len(known),
            "undefined_count": len(rows) - len(known)}


def aggregate(rows):
    if not rows:
        return {"planned_calls": 0, "valid_calls": 0,
                "invalid_or_missing_calls": 0, "exact_rate_fixed": None}
    correct, valid = sum(r["exact_match"] for r in rows), sum(r["valid"] for r in rows)
    parsed = sum(r.get("parse_status") == "success" for r in rows)
    out = {"planned_calls": len(rows), "valid_calls": valid,
           "invalid_or_missing_calls": len(rows) - valid, "valid_rate": valid / len(rows),
           "exact_correct": correct, "exact_rate_fixed": correct / len(rows),
           "exact_rate_fraction": str(Fraction(correct, len(rows))),
           "exact_rate_upper_if_failures_resolved": (correct + len(rows) - valid) / len(rows),
           "parse_success_calls": parsed, "parse_success_rate_fixed": parsed / len(rows),
           "signed_partial": bounded(rows, "signed_partial", lambda r: (-1, 1))}
    ms = [r for r in rows if r["question_type"] == "multi_select"]
    if ms:
        bounds = {"tp": lambda r: (0, r["gold_count"]),
                  "fp": lambda r: (0, 6 - r["gold_count"]),
                  "fn": lambda r: (0, r["gold_count"]),
                  "predicted_count": lambda r: (0, 6),
                  "predicted_minus_gold_count": lambda r: (-r["gold_count"], 6 - r["gold_count"]),
                  "all6": lambda r: (0, 1), "overselection": lambda r: (0, 1),
                  "underselection": lambda r: (0, 1)}
        rename = {"all6": "all6_rate", "overselection": "overselection_rate",
                  "underselection": "underselection_rate"}
        out["multi_select"] = {"planned_calls": len(ms)}
        out["multi_select"].update({rename.get(k, k): bounded(ms, k, lim) for k, lim in bounds.items()})
        out["multi_select"].update({"known_" + k + "_total": sum(r[k] or 0 for r in ms)
                                    for k in ("tp", "fp", "fn")})
    order = [r for r in rows if r["question_type"] == "ordering"]
    if order:
        out["ordering"] = {"planned_calls": len(order),
                          "pair_accuracy": bounded(order, "pair_accuracy", lambda r: (0, 1)),
                          "correct_pairs": bounded(order, "correct_pairs", lambda r: (0, 15)),
                          "wrong_pairs": bounded(order, "wrong_pairs", lambda r: (0, 15))}
    return out


class Audit:
    def __init__(self):
        self.checks, self.errors, self.bindings = 0, [], {}

    def check(self, condition, code, **detail):
        self.checks += 1
        if not condition:
            self.errors.append({"code": code, **detail})

    def read(self, path):
        path = Path(path).resolve()
        self.bindings[str(path)] = digest(path)
        return json.loads(path.read_text())

    def equal(self, actual, expected, where):
        """Compare independently derived fields; ignore explanatory text."""
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), "REPORT_OBJECT", where=where)
            if isinstance(actual, dict):
                for k, v in expected.items():
                    self.check(k in actual, "REPORT_KEY_MISSING", where=where + "." + k)
                    if k in actual:
                        self.equal(actual[k], v, where + "." + k)
        elif isinstance(expected, float):
            self.check(isinstance(actual, (int, float)) and
                       math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                       "NUMERIC_MISMATCH", where=where, actual=actual, expected=expected)
        else:
            self.check(actual == expected, "VALUE_MISMATCH", where=where,
                       actual=actual, expected=expected)


def derive_call(audit, side, q, r, manifest, gold, found):
    row = {"question_type": q["question_type"], "gold_count": len(gold),
           "candidate_id": q["candidate_id"], "valid": False, "exact_match": 0,
           "parse_status": None, "scoring_status": None, "display_prediction": None,
           "content_prediction": None, "signed_partial": None, "pair_accuracy": None,
           "tp": None, "fp": None, "fn": None, "predicted_count": None,
           "predicted_minus_gold_count": None, "all6": None, "overselection": None,
           "underselection": None, "correct_pairs": None, "wrong_pairs": None}
    where = side + ":" + r["request_id"]
    audit.check(len(found) == 1, "RESPONSE_MULTIPLICITY", where=where, count=len(found))
    if len(found) != 1:
        return row
    response = found[0]
    expected = {"request_id": r["request_id"], "candidate_id": q["candidate_id"],
                "root_id": q["root_id"], "question_type": q["question_type"],
                "variant_id": r["variant"]["variant_id"], "slot": r["slot"],
                "seed": r["seed"], "mode": "full_context",
                "solver_model": manifest["solver_model"], "model_path": manifest["model"]}
    valid = all(response.get(k) == v for k, v in expected.items())
    valid &= type(response.get("slot")) is int and type(response.get("seed")) is int
    metadata = response.get("metadata") or {}
    valid &= isinstance(metadata, dict)
    if isinstance(metadata, dict):
        valid &= all(metadata.get(k) == manifest["decoder_contract"].get(k) for k in (
            "temperature", "top_p", "max_tokens", "structured_output_api",
            "structured_output_version"))
        valid &= metadata.get("finish_reason") in ("stop", "eos")
    valid &= response.get("evaluation_valid") is True and not response.get("data_errors")
    native_score = score(q["question_type"], gold, r["variant"]["display_to_content"],
                         response.get("raw_output"))
    audit.equal(response.get("score"), native_score, where + ".saved_score")
    valid &= response.get("score") == native_score and native_score["scored"]
    audit.check(valid, "INVALID_CALL", where=where)
    for k in ("parse_status", "scoring_status", "display_prediction", "content_prediction"):
        row[k] = native_score[k]
    if valid:
        row.update(valid=True, exact_match=int(native_score["exact_match"]),
                   signed_partial=native_score["partial_score_fraction"])
        if q["question_type"] == "multi_select":
            count = len(native_score["content_prediction"])
            row.update(tp=native_score["tp"], fp=native_score["fp"], fn=native_score["fn"],
                       predicted_count=count, predicted_minus_gold_count=count - len(gold),
                       all6=int(count == 6), overselection=int(count > len(gold)),
                       underselection=int(count < len(gold)))
        else:
            row.update(correct_pairs=native_score["tp"], wrong_pairs=native_score["fp"],
                       pair_accuracy=str(Fraction(native_score["tp"], 15)))
    return row


def self_test():
    m = dict(zip("ABCDEF", "abcdef"))
    exact = score("multi_select", ["a", "c"], m, "<box>C,A</box>")
    assert exact["exact_match"] and (exact["tp"], exact["fp"], exact["fn"]) == (2, 0, 0)
    assert score("multi_select", ["a", "b"], m, "<box>A,C</box>")["partial_score"] == "1/4"
    assert score("multi_select", ["a", "b"], m, "<box>A,B,C,D,E,F</box>")["partial_score"] == "0"
    assert score("multi_select", list("abcdef"), m, "<box>A,B,C,D,E,F</box>")["exact_match"]
    assert score("multi_select", ["a", "b"], m, "<box></box>")["partial_score"] == "0"
    for raw in ("A,B", "<box>A,A</box>", "<box>A-B</box>", None):
        assert not score("multi_select", ["a", "b"], m, raw)["scored"]
    assert score("multi_select", ["a", "b"], m, "<box>Z</box>")["scoring_status"] == "INVALID_ID"
    o = score("ordering", list("abcdef"), m, "<box>A,B,C,D,F,E</box>")
    assert (o["tp"], o["fp"], o["fn"], o["partial_score"]) == (14, 1, 0, "13/15")
    rev = score("ordering", list("abcdef"), m, "<box>F,E,D,C,B,A</box>")
    assert rev["tp"] == 0 and rev["fp"] == 15 and rev["partial_score"] == "-1"
    assert score("ordering", list("abcdef"), m, "<box>A,B</box>")["scoring_status"] == "INCOMPLETE_ORDER"
    rows = [{"question_type": "ordering", "valid": True, "exact_match": 1,
             "parse_status": "success", "signed_partial": "1", "pair_accuracy": "1",
             "correct_pairs": 15, "wrong_pairs": 0},
            {"question_type": "ordering", "valid": False, "exact_match": 0}]
    agg = aggregate(rows)
    assert agg["planned_calls"] == 2 and agg["exact_rate_fraction"] == "1/2"
    assert agg["signed_partial"]["mean"] is None
    assert agg["signed_partial"]["lower_fraction"] == "0"
    assert agg["ordering"]["pair_accuracy"]["lower_fraction"] == "1/2"
    assert agg["ordering"]["pair_accuracy"]["upper_fraction"] == "1"
    print("PASS: independent parser, exact, TP/FP/FN, pair inversion, all-select and fixed-failure-denominator fixtures")


def main(batch, report_dir, output):
    ev = batch / "evaluation"
    # No official conclusion for a live or unfinished run.
    required = [ev / "SHARED_REQUESTS.json", ev / "SCOPE.json",
                ev / "EVALUATION_POOL.json", report_dir / "SUMMARY.json"]
    required += [ev / "run" / s / name for s in SIDES
                 for name in ("COMPLETE.json", "RESULTS.json")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("NOT_READY_NO_AUDIT_WRITTEN: " + ", ".join(missing))
    if output.exists():
        raise SystemExit("AUDIT_EXISTS_NO_OVERWRITE: " + str(output))
    a = Audit()
    a.bindings[str(Path(__file__).resolve())] = digest(__file__)
    plan, scope = a.read(ev / "SHARED_REQUESTS.json"), a.read(ev / "SCOPE.json")
    pool = a.read(ev / "EVALUATION_POOL.json")["questions"]
    qs = {q["candidate_id"]: q for q in pool}
    requests = plan["requests"]
    rids = {r["request_id"] for r in requests}
    design = a.read(batch / "PLAN.json")
    admission = a.read(batch / "private/ADMITTED_QUESTIONS.json")["questions"]
    validation = a.read(batch / "evidence/INPUT_VALIDATION.json")
    n = len(admission)
    expected_calls = sum(18 if q["question_type"] == "multi_select" else 12 for q in admission)
    a.check(len(pool) == len(qs) == scope["question_count"] == n, "FROZEN_ADMITTED_COUNT")
    a.check(len(requests) == len(rids) == scope["calls_per_model"] == expected_calls, "FROZEN_REQUEST_COUNT")
    a.check(dict(Counter(q["split"] for q in pool)) == {"dev": n}, "SPLIT_QUESTION_COUNTS")
    a.check(plan["seed"] == 2026100800 and scope["seed"] == 2026100800, "FIXED_SEED")
    a.check(validation["planned_slots"] == design["planned_slots"] and
            validation["admitted_count"] == n and validation["rejected_count"] + n == design["planned_slots"], "ADMISSION_ACCOUNTING")
    original_qs = {q["candidate_id"]: q for q in admission}
    a.check(set(original_qs) == set(qs), "ADMITTED_ID_SET")
    original_root_ids = {r["root_id"] for r in design["roots"]}
    a.check(scope["planned_roots"] == {"dev": [r["root_id"] for r in design["roots"]]} and
            {q["root_id"] for q in pool} <= original_root_ids, "PLANNED_ROOT_POPULATION")
    for cid, q in qs.items():
        original = original_qs.get(cid, {})
        field = "options" if q["question_type"] == "multi_select" else "items"
        for key in ("candidate_id", "root_id", "question_type", "question", field,
                    "gold", "source_sha256", "pilot_arm", "slot_id"):
            a.check(q.get(key) == original.get(key), "ADMITTED_CONTENT", candidate=cid, field=key)
    rby = defaultdict(list)
    for r in requests:
        rby[r["candidate_id"]].append(r)
    for cid, q in qs.items():
        variants, slots = (6, 3) if q["question_type"] == "multi_select" else (12, 1)
        a.check(len(rby[cid]) == variants * slots, "CANDIDATE_FIXED_DENOMINATOR", candidate=cid)
        a.check({(r["variant"]["variant_id"], r["slot"]) for r in rby[cid]} ==
                {(f"perm_{v:02d}", s) for v in range(variants) for s in range(slots)},
                "CANDIDATE_VARIANT_SLOT_SET", candidate=cid)
        field = "options" if q["question_type"] == "multi_select" else "items"
        expected_content = {content_id(x) for x in q[field].values()}
        for r in rby[cid]:
            mapping = r["variant"]["display_to_content"]
            shown = r["variant"][field]
            a.check(len(mapping) == 6 and set(mapping) == set(shown) and
                    set(mapping.values()) == expected_content, "VARIANT_CONTENT_BIJECTION",
                    request=r["request_id"])
            a.check(all(mapping[k] == content_id(v) for k, v in shown.items()),
                    "DISPLAY_TEXT_CONTENT_BINDING", request=r["request_id"])
    per_call = a.read(report_dir / "PER_CALL.json")
    report_calls = {x["request_id"]: x for x in per_call}
    a.check(len(per_call) == len(report_calls) == len(requests) and
            set(report_calls) == rids, "REPORT_CALL_PLAN_SET")
    per_question = a.read(report_dir / "PER_QUESTION.json")
    report_qs = {x["candidate_id"]: x for x in per_question}
    a.check(len(per_question) == len(report_qs) == len(qs) and
            set(report_qs) == set(qs), "REPORT_QUESTION_SET")
    calls, qmetrics = {}, {}
    for side in SIDES:
        run = ev / "run" / side
        manifest, complete = a.read(run / "manifest.json"), a.read(run / "COMPLETE.json")
        a.check((manifest["solver_model"], manifest["model"]) == MODEL_IDENTITIES[side],
                "FROZEN_MODEL_IDENTITY", side=side)
        gold = a.read(run / "gold.json")
        a.check(manifest["requests"] == requests, "SHARED_PLAN_EQUAL", side=side)
        a.check(manifest["decoder_contract"] == plan["decoder_contract"],
                "DECODER_EQUAL", side=side)
        a.check(complete.get("actual_call_count") == len(requests), "NATIVE_COMPLETE_COUNT", side=side)
        observed = defaultdict(list)
        paths = sorted((run / "responses").glob("*.json"))
        for path in paths:
            response = a.read(path)
            observed[response.get("request_id")].append(response)
        a.check(len(paths) == len(requests) and set(observed) == rids,
                "RAW_RESPONSE_PLAN_SET", side=side, files=len(paths))
        for cid, q in qs.items():
            field = "options" if q["question_type"] == "multi_select" else "items"
            a.check(gold[cid] == [content_id(q[field][label]) for label in q["gold"]],
                    "CANONICAL_GOLD_BINDING", side=side, candidate=cid)
        calls[side] = {}
        for r in requests:
            cid, rid = r["candidate_id"], r["request_id"]
            prompt = run / "prompts" / (rid + ".txt")
            a.check(digest(prompt) == r["prompt_sha256"], "PROMPT_HASH", side=side, request=rid)
            row = derive_call(a, side, qs[cid], r, manifest, gold[cid], observed.get(rid, []))
            calls[side][rid] = row
            a.equal(report_calls[rid]["models"][side], row, "PER_CALL." + side + "." + rid)
        qmetrics[side] = {}
        for cid in qs:
            rows = [calls[side][r["request_id"]] for r in rby[cid]]
            qmetrics[side][cid] = aggregate(rows)
            a.equal(report_qs[cid]["models"][side], qmetrics[side][cid],
                    "PER_QUESTION." + side + "." + cid)
    stratified = a.read(report_dir / "STRATIFIED.json")
    summary = a.read(report_dir / "SUMMARY.json")

    def group(actual, selected, where):
        a.equal(actual, {"candidate_count": len(selected),
                         "covered_root_count": len({q["root_id"] for q in selected})}, where)
        roots = {q["root_id"] for q in selected}
        for side in SIDES:
            sub = actual["models"][side]
            rows = [calls[side][r["request_id"]] for q in selected for r in rby[q["candidate_id"]]]
            a.equal(sub["call_weighted"], aggregate(rows), where + "." + side + ".calls")
            if not selected:
                a.equal(sub["question_equal"], {"candidate_count": 0,
                        "exact_rate_equal_candidate": None, "signed_partial_equal_candidate": None},
                        where + "." + side + ".questions")
                a.equal(sub["root_equal_among_covered"], {"root_count": 0, "exact_rate": None,
                        "exact_rate_fraction": None, "signed_partial": None,
                        "signed_partial_lower": None, "signed_partial_upper": None},
                        where + "." + side + ".roots")
                continue
            qm = [qmetrics[side][q["candidate_id"]] for q in selected]
            exact = average([Fraction(x["exact_rate_fraction"]) for x in qm])
            part = [x["signed_partial"] for x in qm]
            point = average([Fraction(x["mean_fraction"]) for x in part]) if all(
                x["mean_fraction"] is not None for x in part) else None
            a.equal(sub["question_equal"], {
                "candidate_count": len(qm), "planned_calls": len(rows),
                "valid_calls": sum(x["valid_calls"] for x in qm),
                "exact_rate_equal_candidate": float(exact),
                "exact_rate_equal_candidate_fraction": str(exact),
                "signed_partial_equal_candidate": None if point is None else float(point),
                "signed_partial_equal_candidate_fraction": None if point is None else str(point),
                "signed_partial_equal_candidate_lower": float(average([Fraction(x["lower_fraction"]) for x in part])),
                "signed_partial_equal_candidate_upper": float(average([Fraction(x["upper_fraction"]) for x in part]))},
                where + "." + side + ".questions")
            root_exact, root_partial, root_low, root_high = [], [], [], []
            for root in roots:
                members = [qmetrics[side][q["candidate_id"]] for q in selected if q["root_id"] == root]
                root_exact.append(average([Fraction(x["exact_rate_fraction"]) for x in members]))
                root_low.append(average([Fraction(x["signed_partial"]["lower_fraction"]) for x in members]))
                root_high.append(average([Fraction(x["signed_partial"]["upper_fraction"]) for x in members]))
                root_partial.append(average([Fraction(x["signed_partial"]["mean_fraction"]) for x in members])
                                    if all(x["signed_partial"]["mean_fraction"] is not None for x in members) else None)
            rex = average(root_exact)
            rp = average(root_partial) if all(x is not None for x in root_partial) else None
            a.equal(sub["root_equal_among_covered"], {
                "root_count": len(roots), "exact_rate": float(rex), "exact_rate_fraction": str(rex),
                "signed_partial": None if rp is None else float(rp),
                "signed_partial_lower": float(average(root_low)),
                "signed_partial_upper": float(average(root_high))}, where + "." + side + ".roots")
            for qtype, subkey, metric in (
                ("ordering", "ordering", "pair_accuracy"),
                ("multi_select", "multi_select", "predicted_count"),
                ("multi_select", "multi_select", "all6_rate"),
                ("multi_select", "multi_select", "overselection_rate")):
                members = [qmetrics[side][q["candidate_id"]][subkey][metric]
                           for q in selected if q["question_type"] == qtype]
                if members:
                    point = average([Fraction(x["mean_fraction"]) for x in members]) if all(
                        x["mean_fraction"] is not None for x in members) else None
                    a.equal(sub["question_equal"][qtype + "_" + metric + "_equal_candidate"], {
                        "candidate_count": len(members),
                        "mean": None if point is None else float(point),
                        "lower": float(average([Fraction(x["lower_fraction"]) for x in members])),
                        "upper": float(average([Fraction(x["upper_fraction"]) for x in members]))},
                        where + "." + side + ".question_equal." + metric)
        for left, right in PAIRS:
            key = right + "_minus_" + left
            ex, partial, lows, highs = [], [], [], []
            root_deltas = []
            for q in selected:
                x, y = qmetrics[left][q["candidate_id"]], qmetrics[right][q["candidate_id"]]
                ex.append(Fraction(y["exact_rate_fraction"]) - Fraction(x["exact_rate_fraction"]))
                xp, yp = x["signed_partial"], y["signed_partial"]
                partial.append(Fraction(yp["mean_fraction"]) - Fraction(xp["mean_fraction"])
                               if xp["mean_fraction"] is not None and yp["mean_fraction"] is not None else None)
                lows.append(Fraction(yp["lower_fraction"]) - Fraction(xp["upper_fraction"]))
                highs.append(Fraction(yp["upper_fraction"]) - Fraction(xp["lower_fraction"]))
            for root in roots:
                members = [q for q in selected if q["root_id"] == root]
                root_deltas.append(average([
                    Fraction(qmetrics[right][q["candidate_id"]]["exact_rate_fraction"]) -
                    Fraction(qmetrics[left][q["candidate_id"]]["exact_rate_fraction"])
                    for q in members]))
            point = average(partial) if partial and all(x is not None for x in partial) else None
            a.equal(actual["paired_deltas"][key], {
                "exact_rate_question_equal": None if not ex else float(average(ex)),
                "signed_partial_question_equal": None if point is None else float(point),
                "signed_partial_question_equal_lower": None if not lows else float(average(lows)),
                "signed_partial_question_equal_upper": None if not highs else float(average(highs)),
                "exact_rate_root_equal": None if not root_deltas else float(average(root_deltas))},
                where + ".paired_deltas." + key)

    for split in SPLITS:
        selected = [q for q in pool if split == "all" or q["split"] == split]
        for name in STRATA:
            group(stratified[split][name], [q for q in selected if name in tags(q)],
                  "STRATIFIED." + split + "." + name)
        a.equal(summary["all_question_strata"][split], stratified[split]["all"],
                "SUMMARY.all_question_strata." + split)
        if split != "all":
            a.equal(summary["primary_strata"][split], stratified[split]["all"],
                    "SUMMARY.primary_strata." + split)
    roots = a.read(report_dir / "PER_ROOT.json")
    expected_roots = {(split, root) for split, ids in scope["planned_roots"].items() for root in ids}
    a.check(len(roots) == len(expected_roots) == len(design["roots"]) and
            {(x["split"], x["root_id"]) for x in roots} == expected_roots, "REPORT_PLANNED_DEV_ROOT_ROWS")
    for row in roots:
        selected = [q for q in pool if q["root_id"] == row["root_id"]]
        a.equal(row, {"question_count": len(selected), "has_questions": bool(selected)},
                "PER_ROOT." + row["root_id"])
        for name in STRATA:
            group(row["strata"][name], [q for q in selected if name in tags(q)],
                  "PER_ROOT." + row["root_id"] + "." + name)
    for left, right in PAIRS:
        key = right + "_minus_" + left
        for rid in rids:
            x, y = calls[left][rid], calls[right][rid]
            point = str(Fraction(y["signed_partial"]) - Fraction(x["signed_partial"])) if (
                x["signed_partial"] is not None and y["signed_partial"] is not None) else None
            a.equal(report_calls[rid]["paired_deltas"][key],
                    {"exact_match": y["exact_match"] - x["exact_match"], "signed_partial": point},
                    "CALL_DELTA." + key + "." + rid)
        for cid in qs:
            x, y = qmetrics[left][cid], qmetrics[right][cid]
            xp, yp = x["signed_partial"], y["signed_partial"]
            partial = Fraction(yp["mean_fraction"]) - Fraction(xp["mean_fraction"]) if (
                xp["mean_fraction"] is not None and yp["mean_fraction"] is not None) else None
            low = Fraction(yp["lower_fraction"]) - Fraction(xp["upper_fraction"])
            high = Fraction(yp["upper_fraction"]) - Fraction(xp["lower_fraction"])
            exact = Fraction(y["exact_rate_fraction"]) - Fraction(x["exact_rate_fraction"])
            a.equal(report_qs[cid]["paired_deltas"][key], {
                "exact_rate_delta_new_minus_baseline": float(exact),
                "exact_rate_delta_fraction": str(exact),
                "signed_partial_delta": None if partial is None else float(partial),
                "signed_partial_delta_fraction": None if partial is None else str(partial),
                "signed_partial_delta_lower": float(low),
                "signed_partial_delta_lower_fraction": str(low),
                "signed_partial_delta_upper": float(high),
                "signed_partial_delta_upper_fraction": str(high)},
                "QUESTION_DELTA." + key + "." + cid)
    baseline = a.read(report_dir / "ALL_SELECT_BASELINE.json")
    baseline_calls = []
    for q in pool:
        if q["question_type"] != "multi_select":
            continue
        g = len(q["gold"])
        row = {"question_type": "multi_select", "gold_count": g, "valid": True,
               "parse_status": "success", "exact_match": int(g == 6), "signed_partial": str(int(g == 6)),
               "tp": g, "fp": 6 - g, "fn": 0, "predicted_count": 6,
               "predicted_minus_gold_count": 6 - g, "all6": 1,
               "overselection": int(g < 6), "underselection": 0}
        baseline_calls.extend([(q, row)] * len(rby[q["candidate_id"]]))
    for split in SPLITS:
        for name in STRATA:
            if name != "ordering":
                a.equal(baseline["strata"][split][name],
                        aggregate([row for q, row in baseline_calls if
                                   (split == "all" or q["split"] == split) and name in tags(q)]),
                        "ALL_SELECT." + split + "." + name)
    arms = a.read(report_dir / "ARMS.json")
    for arm in ["new_structure", "old_structure_control"]:
        selected = [q for q in pool if q["pilot_arm"] == arm]
        obj = arms["arms"][arm]
        group(obj["all"], selected, "ARMS." + arm + ".all")
        for kind in ["multi_select", "ordering"]:
            group(obj["by_type"][kind], [q for q in selected if q["question_type"] == kind], "ARMS." + arm + "." + kind)
        for g in [2,3,4,5]:
            group(obj["by_gold"][str(g)], [q for q in selected if q["question_type"] == "multi_select" and len(q["gold"]) == g], "ARMS." + arm + ".gold" + str(g))
        for root in original_root_ids:
            for layer in ["all", "multi_select", "gold3", "ordering"]:
                group(obj["by_root"][root][layer], [q for q in selected if q["root_id"] == root and (layer == "all" or layer in tags(q))], "ARMS." + arm + "." + root + "." + layer)
    for root in original_root_ids:
        for signature in ["3-2-1","2-2-2","4-2","1-1-1-1-1-1"]:
            selected = [q for q in pool if q["root_id"] == root and "ordering_scenes_" + signature in tags(q)]
            group(arms["ordering_by_root_and_scene_signature"][root][signature], selected, "SCENES." + root + "." + signature)
    a.equal(summary, {"status": "COMPLETE", "errors": [], "question_count": n, "calls_per_model": expected_calls,
                      "scope": "KNOWN_DEV_ROOTS_NEW_STRUCTURE_INTERNAL_PILOT"}, "SUMMARY")
    result = {"status": "PASS" if not a.errors else "FAIL", "actual_auditor": args.auditor,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "independent_arithmetic": True, "imports_production_scoring_or_reporting": False,
              "inference_executed_by_audit": False, "checks": a.checks, "failure_count": len(a.errors),
              "errors": a.errors, "fixed_questions": n, "calls_per_model": expected_calls,
              "planned_calls_all_models": expected_calls * len(SIDES),
              "models": {side: aggregate(list(calls[side].values())) for side in SIDES},
              "verification_scope": ["Raw box parsing and content mapping; exact/partial/TP/FP/FN/pairs.",
                   "Every planned denominator, per-call and per-question metrics and paired deltas.",
                   "DEV/type/Gold/arm/root/ordering-scene aggregates and analytic all-select.",
                   "Original admitted content, model identity, prompts, admission accounting."],
              "limitations": ["Arithmetic/integrity only; does not certify semantics or causal effects.",
                   "Separate internal actor on same inherited model and shared filesystem; no external independence claim.",
                   "Known DEV sources and four roots; descriptive structural pilot."],
              "audited_files": [{"path": p, "sha256": h} for p, h in sorted(a.bindings.items())]}
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "checks": a.checks,
                      "failures": len(a.errors), "output": str(output)}))
    return bool(a.errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--auditor", help="Actual executing actor identity, required for official audit")
    parser.add_argument("--batch", type=Path, default=BATCH)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.auditor:
            parser.error("--auditor is required; do not misattribute an independent actor")
        raise SystemExit(main(args.batch.resolve(),
            (args.report_dir or args.batch / "evaluation/reports").resolve(),
            (args.output or args.batch / "evidence/INDEPENDENT_RESULTS_AUDIT.json").resolve()))
