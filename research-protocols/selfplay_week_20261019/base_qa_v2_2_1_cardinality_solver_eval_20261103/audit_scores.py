#!/usr/bin/env python3
"""Independent deterministic re-score of frozen local Solver outputs."""
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))

requests = read("REQUESTS.json")
scores = read("SCORES.json")
raws = read("RAW_ANSWERS.json")
assert len(requests) == len(raws) == 396
assert len(scores["requests"]) == 396
assert len({r["request_id"] for r in requests}) == 396
assert Counter(r["context_mode"] for r in requests) == {"full_context": 198, "no_context": 198}
assert len({r["sampling_seed"] for r in requests}) == 198

bad = []
invalid_by_mode = Counter()
exact_by_mode = Counter()
planned_by_mode = Counter()
sum_j = Counter()
sum_recall = Counter()
known_fp = Counter()
known_fn = Counter()
control = {"full_context": Counter(), "no_context": Counter()}
for request, raw, scored in zip(requests, raws, scores["requests"]):
    assert request["request_id"] == raw["request_id"] == scored["request_id"]
    prompt = (HERE / request["prompt_path"]).read_bytes()
    if hashlib.sha256(prompt).hexdigest() != request["prompt_sha256"]:
        bad.append(request["request_id"] + ":prompt_hash")
    labels = sorted(request["display_to_option_id"])
    pattern = re.compile(r"\s*([" + "".join(labels) + r"](?:\s*,\s*[" + "".join(labels) + r"])*)\s*\Z")
    raw_text = raw.get("raw_prediction") if raw.get("status") == "RETURNED" else None
    parsed = pattern.fullmatch(raw_text.strip()) if isinstance(raw_text, str) else None
    values = parsed.group(1).split(",") if parsed else []
    values = [v.strip() for v in values]
    valid = bool(values) and len(values) == len(set(values)) and raw.get("finish_reason") in ("stop", "eos")
    if valid:
        pred = {request["display_to_option_id"][label] for label in values}
        gold = set(request["gold_option_ids"])
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        exact = pred == gold
        recall = tp / len(gold)
        j = None if request["n"] == request["g"] else str(Fraction(tp, request["g"]) - Fraction(fp, request["n"] - request["g"]))
        if request["n"] > request["g"] and float(Fraction(tp, request["g"]) - Fraction(fp, request["n"] - request["g"])) < -1:
            bad.append(request["request_id"] + ":J_range")
        known_fp[request["context_mode"]] += fp
        known_fn[request["context_mode"]] += fn
    else:
        pred, tp, fp, fn, exact, recall = set(), None, None, None, False, 0.0
        j = None if request["n"] == request["g"] else "0"
        invalid_by_mode[request["context_mode"]] += 1
    if request["n"] == request["g"]:
        mode_control = control[request["context_mode"]]
        mode_control["exact"] += int(exact)
        mode_control["valid"] += int(valid)
        mode_control["recall_milli"] += round(recall * 1000)
        assert j is None
    else:
        sum_j[request["context_mode"]] += Fraction(j)
        sum_recall[request["context_mode"]] += Fraction(str(recall))
    exact_by_mode[request["context_mode"]] += int(exact)
    planned_by_mode[request["context_mode"]] += 1
    if scored["valid"] != bool(valid):
        bad.append(request["request_id"] + ":validity_mismatch")
    if scored["exact_correct"] != bool(exact):
        bad.append(request["request_id"] + ":exact_mismatch")
    if scored.get("predicted_option_ids") != (sorted(pred) if valid else None):
        bad.append(request["request_id"] + ":mapping_mismatch")
    if scored.get("tp") != tp or scored.get("fp") != fp or scored.get("fn") != fn:
        bad.append(request["request_id"] + ":TP_FP_FN_mismatch")
    # Invalid outputs have fixed-denominator J=0 in aggregates, while the
    # frozen per-call artifact leaves the undefined prediction score blank.
    expected_recorded_j = j if valid else None
    if scored.get("J_fraction") != expected_recorded_j:
        bad.append(request["request_id"] + ":J_mismatch")

assert not bad, bad[:20]
summary = {}
for mode in ("full_context", "no_context"):
    ordinary_n = 180
    ordinary = [r for r in scores["requests"] if r["context_mode"] == mode and not r["all_true_control"]]
    ctrl = [r for r in scores["requests"] if r["context_mode"] == mode and r["all_true_control"]]
    assert len(ordinary) == ordinary_n and len(ctrl) == 18
    summary[mode] = {
        "ordinary_calls": len(ordinary),
        "ordinary_valid": sum(r["valid"] for r in ordinary),
        "ordinary_exact": sum(r["exact_correct"] for r in ordinary),
        "ordinary_accuracy": sum(r["exact_correct"] for r in ordinary) / len(ordinary),
        "ordinary_mean_J_exact_fraction": str(sum((Fraction(r["J_fraction"]) if r["valid"] else Fraction(0)) for r in ordinary) / len(ordinary)),
        "ordinary_mean_J": float(sum((Fraction(r["J_fraction"]) if r["valid"] else Fraction(0)) for r in ordinary) / len(ordinary)),
        "ordinary_known_FP_FN": {"FP": sum(r["fp"] or 0 for r in ordinary if r["valid"]), "FN": sum(r["fn"] or 0 for r in ordinary if r["valid"])},
        "ordinary_invalid": sum(not r["valid"] for r in ordinary),
        "all_true_control": {"calls": len(ctrl), "exact": sum(r["exact_correct"] for r in ctrl), "accuracy": sum(r["exact_correct"] for r in ctrl) / len(ctrl), "J": None, "J_reason": "n==g so n-g is zero"},
    }
doc = {
    "status": "PASS" if not bad else "FAIL", "recomputed_calls": 396,
    "unique_sampling_seeds": 198, "paired_full_no_context_seeds": True,
    "mapping_errors": 0, "scoring_errors": 0, "prompt_hash_errors": 0,
    "invalid_policy": "invalid is fixed-denominator zero for exact accuracy, recall and J aggregates; per-call FP/FN/J remain unknown or blank; no answer repair or retry",
    "J_formula": "J=TP/g-FP/(n-g), only when n>g",
    "summaries": summary, "errors": bad,
}
with (HERE / "SCORING_AUDIT.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": doc["status"], "recomputed_calls": 396, "summary": summary}, ensure_ascii=False, indent=2))
