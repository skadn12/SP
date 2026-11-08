#!/usr/bin/env python3
"""Build per-item diagnostic summaries from frozen Solver outputs."""
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


requests = read("REQUESTS.json")
raws = {row["request_id"]: row for row in read("RAW_ANSWERS.json")}
items = {row["candidate_id"]: row for row in read("ITEMS_PRIVATE.json")}
gold_rows = {row["request_id"]: row for row in read("SCORES.json")["requests"]}

prepared = {}
option_texts = defaultdict(dict)
for request in requests:
    labels = sorted(request["display_to_option_id"])
    pattern = re.compile(r"\s*(?:" + "|".join(re.escape(label) for label in labels) + r")(?:\s*,\s*(?:" + "|".join(re.escape(label) for label in labels) + r"))*\s*\Z")
    raw = raws[request["request_id"]]
    text = raw.get("raw_prediction") if raw.get("status") == "RETURNED" else None
    valid = bool(isinstance(text, str) and pattern.fullmatch(text.strip()) and raw.get("finish_reason") in ("stop", "eos"))
    labels_out = [part.strip() for part in text.strip().split(",")] if valid else []
    valid = valid and len(labels_out) == len(set(labels_out))
    prediction = {request["display_to_option_id"][label] for label in labels_out} if valid else set()
    gold = set(request["gold_option_ids"])
    tp, fp, fn = (len(prediction & gold), len(prediction - gold), len(gold - prediction)) if valid else (None, None, None)
    exact = valid and prediction == gold
    if request["n"] == request["g"]:
        j = None
        j_aggregate = None
    elif valid:
        j = Fraction(tp, request["g"]) - Fraction(fp, request["n"] - request["g"])
        j_aggregate = j
    else:
        j = None
        j_aggregate = Fraction(0)
    prepared[request["request_id"]] = {
        "request": request, "raw": raw, "valid": valid,
        "prediction": prediction, "gold": gold, "tp": tp, "fp": fp,
        "fn": fn, "exact": bool(exact), "j": j, "j_aggregate": j_aggregate,
    }
    for option in request["options"]:
        prior = option_texts[request["candidate_id"]].setdefault(option["option_id"], option["text"])
        assert prior == option["text"], (request["candidate_id"], option["option_id"])

candidate_ids = list(items)
candidate_summaries = {}
paired = Counter()
for candidate_id in candidate_ids:
    item_rows = [row for row in prepared.values() if row["request"]["candidate_id"] == candidate_id]
    candidate = items[candidate_id]
    n = len(item_rows[0]["request"]["display_to_option_id"])
    g = len(candidate["gold_option_ids"])
    by_mode = {}
    mode_semantics = {}
    for mode in ("full_context", "no_context"):
        rows = [row for row in item_rows if row["request"]["context_mode"] == mode]
        valid_rows = [row for row in rows if row["valid"]]
        predictions = Counter(tuple(sorted(row["prediction"])) for row in valid_rows)
        modal_count = max(predictions.values(), default=0)
        by_perm = {}
        pair_matches = pair_total = 0
        unanimous_permutations = 0
        for perm in sorted({row["request"]["variant_id"] for row in rows}):
            subset = [row for row in rows if row["request"]["variant_id"] == perm]
            valid_subset = [row for row in subset if row["valid"]]
            normalized = [tuple(sorted(row["prediction"])) for row in valid_subset]
            local_counts = Counter(normalized)
            for i in range(len(valid_subset)):
                for jdx in range(i + 1, len(valid_subset)):
                    pair_total += 1
                    pair_matches += int(normalized[i] == normalized[jdx])
            unanimous = len(valid_subset) == len(subset) and len(local_counts) == 1
            unanimous_permutations += int(unanimous)
            by_perm[perm] = {
                "calls": len(subset), "valid": len(valid_subset),
                "exact": sum(row["exact"] for row in subset),
                "mean_J": float(sum(row["j_aggregate"] for row in subset if row["j_aggregate"] is not None) / len(subset)) if g < n else None,
                "distinct_valid_answer_sets": len(local_counts),
                "all_three_valid_and_identical": unanimous,
            }
        denominator = len(rows)
        entry = {
            "calls": denominator,
            "valid": len(valid_rows),
            "invalid": denominator - len(valid_rows),
            "exact": sum(row["exact"] for row in rows),
            "exact_accuracy_fixed_denominator": sum(row["exact"] for row in rows) / denominator,
            "mean_J_fixed_denominator": float(sum(row["j_aggregate"] for row in rows if row["j_aggregate"] is not None) / denominator) if g < n else None,
            "mean_recall_fixed_denominator": sum((row["tp"] / g if row["valid"] else 0) for row in rows) / denominator,
            "valid_only_FP_total": sum(row["fp"] or 0 for row in rows),
            "valid_only_FN_total": sum(row["fn"] or 0 for row in rows),
            "distinct_valid_semantic_answer_sets": len(predictions),
            "modal_valid_answer_set_count": modal_count,
            "modal_valid_answer_set_fraction": modal_count / len(valid_rows) if valid_rows else None,
            "same_permutation_repeat_pair_agreement": pair_matches / pair_total if pair_total else None,
            "permutations_with_three_identical_valid_answers": unanimous_permutations,
            "by_permutation": by_perm,
        }
        by_mode[mode] = entry
        mode_semantics[mode] = rows

    stable_counts = Counter()
    for row in mode_semantics["full_context"]:
        if row["valid"]:
            stable_counts.update(row["prediction"])
    option_rows = []
    for option_id, text in sorted(option_texts[candidate_id].items()):
        is_gold = option_id in set(candidate["gold_option_ids"])
        selected = stable_counts[option_id]
        option_rows.append({
            "option_id": option_id, "text": text, "gold": is_gold,
            "full_context_selected": selected,
            "full_context_missed": 18 - selected if is_gold else 0,
            "full_context_false_selected": selected if not is_gold else 0,
        })
    candidate_summaries[candidate_id] = {
        "root_id": candidate["root_id"],
        "question": candidate["question"],
        "gold_count": g, "option_count": n,
        "gold_option_ids": candidate["gold_option_ids"],
        "gold_answers": [
            {"option_id": option_id, "text": option_texts[candidate_id][option_id]}
            for option_id in sorted(candidate["gold_option_ids"])
        ],
        "content_certificate_status": candidate["content_certificate_status"],
        "formal_qa_eligible": candidate["formal_qa_eligible"],
        "full_context": by_mode["full_context"],
        "no_context": by_mode["no_context"],
        "full_context_option_diagnostics": option_rows,
    }

by_seed = {}
for row in prepared.values():
    req = row["request"]
    by_seed[(req["candidate_id"], req["sampling_seed"])] = by_seed.get((req["candidate_id"], req["sampling_seed"]), {})
    by_seed[(req["candidate_id"], req["sampling_seed"])][req["context_mode"]] = row
for pair in by_seed.values():
    if len(pair) != 2:
        continue
    full, noctx = pair["full_context"], pair["no_context"]
    key = ("full_correct" if full["exact"] else "full_wrong", "no_context_correct" if noctx["exact"] else "no_context_wrong")
    paired[key] += 1

doc = {
    "status": "DIAGNOSTIC_ONLY",
    "formal_qa_eligible": False,
    "model": read("MODEL_CONFIG.json")["model"],
    "requests": len(requests),
    "scoring": "Exact set accuracy plus J=TP/g-FP/(n-g); invalid outputs score zero in fixed-denominator aggregates. FP/FN totals include valid outputs only.",
    "repeat_consistency_note": "Agreement is computed on stable option_id sets within each option permutation; repeated sampling is stochastic.",
    "paired_full_vs_no_context_exact_outcomes": {"|".join(key): count for key, count in paired.items()},
    "candidates": candidate_summaries,
}
(HERE / "ITEM_DIAGNOSTICS.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

ordinary = [row for row in candidate_summaries.values() if row["gold_count"] < row["option_count"]]
ordinary_calls = sum(row["full_context"]["calls"] for row in ordinary)
ordinary_exact = sum(row["full_context"]["exact"] for row in ordinary)
ordinary_j = sum(row["full_context"]["mean_J_fixed_denominator"] * row["full_context"]["calls"] for row in ordinary) / ordinary_calls
ordinary_no_exact = sum(row["no_context"]["exact"] for row in ordinary)
ordinary_no_j = sum(row["no_context"]["mean_J_fixed_denominator"] * row["no_context"]["calls"] for row in ordinary) / ordinary_calls
control = next(row for row in candidate_summaries.values() if row["gold_count"] == row["option_count"])

lines = [
    "# Local Qwen3-4B Solver Evaluation",
    "",
    "This is a diagnostic evaluation of frozen candidate drafts, not a formal QA pass or approved weakness signal. The source admission gate is still unresolved. No school API or training was used.",
    "",
    f"Model: `{doc['model']}`. The run used temperature 0.8, top_p 0.9, max 32 new tokens, offline inference, and no retries. It contains 6 unique answer-option permutations × 3 samples per permutation × Full/No-context for each of 11 candidates (396 requests).",
    "",
    "## Overall",
    "",
    "| Condition | New targets | Exact | Accuracy | Mean J | Recall | Valid outputs | Invalid |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    f"| Full source context | {ordinary_calls} | {ordinary_exact} | {ordinary_exact / ordinary_calls:.2%} | {ordinary_j:.4f} | {sum(row['full_context']['mean_recall_fixed_denominator'] * row['full_context']['calls'] for row in ordinary) / ordinary_calls:.4f} | {sum(row['full_context']['valid'] for row in ordinary)} | {sum(row['full_context']['invalid'] for row in ordinary)} |",
    f"| No context | {ordinary_calls} | {ordinary_no_exact} | {ordinary_no_exact / ordinary_calls:.2%} | {ordinary_no_j:.4f} | {sum(row['no_context']['mean_recall_fixed_denominator'] * row['no_context']['calls'] for row in ordinary) / ordinary_calls:.4f} | {sum(row['no_context']['valid'] for row in ordinary)} | {sum(row['no_context']['invalid'] for row in ordinary)} |",
    "",
    "For non-control items, `J=TP/g-FP/(n-g)`. Invalid answers count as zero in fixed-denominator exact accuracy, J, and recall; FP/FN are unknown for invalid outputs and are reported only over valid outputs. Full and No-context use paired sampling seeds.",
    "",
    "The 10 non-control targets cover all requested Gold cardinalities: 2 correct options (2 items), 3 (1), 4 (3), 5 (3), and 6 (1). `c_meadow` is the additional 6/6 all-true control.",
    "",
    f"The all-true control `{next(key for key, row in candidate_summaries.items() if row['gold_count'] == row['option_count'])}` has `n=g=6`, so J is undefined: Full {control['full_context']['exact']}/{control['full_context']['calls']} exact ({control['full_context']['exact_accuracy_fixed_denominator']:.2%}); No-context {control['no_context']['exact']}/{control['no_context']['calls']} ({control['no_context']['exact_accuracy_fixed_denominator']:.2%}).",
    "",
    "## Per-Item Full-Context Results",
    "",
    "| Candidate | Gold | Exact / 18 | Accuracy | Mean J | Recall | Invalid | No-context exact / 18 | Repeat agreement* |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for candidate_id, row in candidate_summaries.items():
    full, noctx = row["full_context"], row["no_context"]
    j = "n/a" if full["mean_J_fixed_denominator"] is None else f"{full['mean_J_fixed_denominator']:.3f}"
    repeat = full["same_permutation_repeat_pair_agreement"]
    repeat_text = "n/a" if repeat is None else f"{repeat:.0%}"
    lines.append(f"| `{candidate_id}` | {row['gold_count']}/{row['option_count']} | {full['exact']}/18 | {full['exact_accuracy_fixed_denominator']:.1%} | {j} | {full['mean_recall_fixed_denominator']:.3f} | {full['invalid']} | {noctx['exact']}/18 | {repeat_text} |")
lines += [
    "",
    "*Repeat agreement is pairwise agreement among the three valid semantic answer sets for each fixed option permutation, pooled across permutations. The separate `ITEM_DIAGNOSTICS.json` records option-level counts and per-permutation outcomes.",
    "",
    "## Option-Level Errors",
    "",
    "Counts below use stable option IDs across all 18 Full-context calls. A missed count is for Gold options; false-selected count is for non-Gold options.",
    "",
]
for candidate_id, row in candidate_summaries.items():
    errors = [opt for opt in row["full_context_option_diagnostics"] if opt["full_context_missed"] or opt["full_context_false_selected"]]
    lines.append(f"### `{candidate_id}`")
    lines.append("")
    if not errors:
        lines.append("No option-level errors in valid Full-context outputs.")
    else:
        for opt in errors:
            if opt["gold"]:
                lines.append(f"- Missed `{opt['option_id']}` in {opt['full_context_missed']}/18 calls: {opt['text']}")
            else:
                lines.append(f"- Incorrectly selected `{opt['option_id']}` in {opt['full_context_false_selected']}/18 calls: {opt['text']}")
    lines.append("")
lines += [
    "",
    "## Questions And Gold",
    "",
    "Stable option IDs are shown; display letters were shuffled for the Solver runs.",
    "",
]
for candidate_id, row in candidate_summaries.items():
    lines.append(f"### `{candidate_id}` ({row['gold_count']}/{row['option_count']})")
    lines.append("")
    lines.append(row["question"])
    lines.append("")
    lines.append("Gold:")
    for answer in row["gold_answers"]:
        lines.append(f"- `{answer['option_id']}`: {answer['text']}")
    lines.append("")
lines += [
    "## Interpretation",
    "",
    "Full dialogue substantially improved performance over the no-context baseline (84/180 vs 18/180 exact on the 10 non-control targets), showing that answers depend on source information. Performance still varies sharply by item: two targets were answered exactly in all 18 Full-context runs, while three received 0/18 exact answers. Exact-set failure can reflect one or more omissions or false selections; the option-level counts above show which propositions drove each result.",
    "",
    "All 14 invalid No-context outputs came from `c_cedar` and used an `Answer:` prefix, violating the required bare comma-separated label format; only 4/18 No-context `c_cedar` responses were valid. One Full-context output for `fresh_e_six_selfcare_01` repeats a label. Invalid outputs were scored as zero without repair or retry. `c_river` was excluded before inference because independent blind review held it for caller-ownership ambiguity. The all-true item is only a control and has no J score.",
    "",
    "All candidates in this experiment are marked `formal_qa_eligible=false`. In particular, the `fresh_progress_25652` evidence-deletion certificate has a recorded contradiction; interpret its Solver score as a diagnostic of that draft only, not a validated Solver weakness. No output from this run is promoted to formal training or feedback data.",
    "",
]
(HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"status": "PASS", "items": len(candidate_summaries), "report": str(HERE / "REPORT.md"), "diagnostics": str(HERE / "ITEM_DIAGNOSTICS.json")}, ensure_ascii=False))
