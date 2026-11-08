#!/usr/bin/env python3
"""Create a descriptive post-score report for the frozen four-root pilot."""
import json
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUN = BASE / "solver_runs/s0_direct_v1"
SCORES = RUN / "SCORES.json"
LEDGER = BASE / "frozen_inputs/ADMISSION_LEDGER_FROZEN.json"
SLOT_PLAN = BASE / "SLOT_PLAN.json"

def main():
    score = json.loads(SCORES.read_text())
    ledger = json.loads(LEDGER.read_text())
    plan = json.loads(SLOT_PLAN.read_text())
    slots = {x["slot_id"]: x for x in ledger["slots"]}
    planned = {x["slot_id"]: x for x in plan["slots"]}
    qs = []
    by_relation = defaultdict(list)
    by_root = defaultdict(list)
    by_gold = defaultdict(list)
    by_type = defaultdict(list)
    for q in score["questions"]:
        slot = slots[q["candidate_id"]]
        row = {"candidate_id": q["candidate_id"], "root_id": q["root_id"],
               "question_type": q["question_type"], "relation_target": slot["relation_target"],
               "gold_count": slot.get("target_gold_count"), "item_count": q["item_count"],
               "exact_calls": q["exact_calls"], "denominator_calls": q["denominator_calls"],
               "call_exact_rate": q["exact_calls"] / q["denominator_calls"],
               "stratum": q["stratum"], "valid_calls": q["valid_calls"],
               "fp_calls": q.get("fp_calls"), "fn_calls": q.get("fn_calls"),
               "pair_accuracy": q.get("pair_accuracy"), "total_pairs": q.get("total_pairs"),
               "correct_pairs": q.get("correct_pairs")}
        qs.append(row)
        by_relation[row["relation_target"]].append(row)
        by_root[row["root_id"]].append(row)
        by_type[row["question_type"]].append(row)
        if row["question_type"] == "multi_select":
            by_gold[str(row["gold_count"])].append(row)
    def aggregate(rows):
        calls = sum(x["denominator_calls"] for x in rows)
        exact = sum(x["exact_calls"] for x in rows)
        result = {"questions": len(rows), "calls": calls, "exact_calls": exact,
                  "call_weighted_exact_rate": exact / calls if calls else None,
                  "question_macro_exact_rate": sum(x["call_exact_rate"] for x in rows) / len(rows) if rows else None}
        if rows and all(x["question_type"] == "ordering" for x in rows):
            pairs = sum(x["total_pairs"] for x in rows)
            correct = sum(x["correct_pairs"] for x in rows)
            result.update(total_pairs=pairs, correct_pairs=correct,
                          pair_weighted_accuracy=correct / pairs if pairs else None,
                          question_macro_pair_accuracy=sum(x["pair_accuracy"] for x in rows) / len(rows))
        if rows and all(x["question_type"] == "multi_select" for x in rows):
            result.update(fp_decisions=sum(x["fp_calls"] for x in rows),
                          fn_decisions=sum(x["fn_calls"] for x in rows))
        return result
    analysis = {
        "status": "COMPLETE_DESCRIPTIVE_ANALYSIS",
        "run_binding": score["manifest_binding"],
        "raw_freeze_binding": score["raw_response_freeze_binding"],
        "preflight_binding": score["preflight_freeze_binding"],
        "denominators": {"planned_slots": score["production_denominator"],
            "admitted_questions": score["denominator_questions"], "planned_roots": score["planned_root_count"],
            "requests": score["denominator_calls"], "valid_requests": score["valid_calls"],
            "shortfall_slots": score["production_unadmitted_slots"]},
        "overall": {"exact_calls": score["exact_calls"], "request_weighted_exact_rate": score["exact_calls"] / score["denominator_calls"],
            "question_macro_exact_rate": score["question_equal_exact"], "root_macro_exact_rate": score["root_equal_exact"],
            "prompt_tokens": score["cost"]["known_prompt_tokens"], "completion_tokens": score["cost"]["known_completion_tokens"],
            "wall_seconds": score["cost"]["live_wall_seconds"], "model_load_seconds": score["cost"]["model_load_wall_seconds"]},
        "by_type": {k: aggregate(v) for k, v in sorted(by_type.items())},
        "by_relation": {k: aggregate(v) for k, v in sorted(by_relation.items())},
        "by_root": {k: aggregate(v) for k, v in sorted(by_root.items())},
        "multi_select_by_gold_count": {k: aggregate(v) for k, v in sorted(by_gold.items(), key=lambda x: int(x[0]))},
        "questions": qs,
        "interpretation_limits": [
            "Four historically exposed Train roots provide exploratory feasibility evidence, not unseen-root generalization.",
            "Relation and gold-count groups contain at most three questions and are confounded with source/root and item content.",
            "The direct-answer run does not test CoT, memory, training gain, or causal usefulness of feedback.",
            "R3-ORD2 and R4-ORD2 order clock values; their deadline/threshold semantics are not event chronology.",
            "The one R2-ORD2 shortfall remains in the 24-slot production denominator and has no Solver score."
        ]}
    out_json = RUN / "ANALYSIS.json"
    out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    def pct(x): return "—" if x is None else f"{100*x:.1f}%"
    md = ["# Four-root diversity pilot: S0 direct-answer report", "",
          "The frozen pilot completed on 2026-10-22. This is an exploratory Solver diagnosis on four historically exposed Train roots.", "",
          "## Frozen denominator and execution", "",
          f"- Slots: **23/24 admitted**, with one source-supported shortfall (`DIVV3-R2-ORD2`) retained in the planned denominator.",
          f"- Questions and calls: **23 questions, 336/336 returned and valid calls**; no retry or source truncation.",
          f"- Exact answer calls: **{score['exact_calls']}/336 ({pct(analysis['overall']['request_weighted_exact_rate'])})**.",
          f"- Question-macro exact rate: **{pct(score['question_equal_exact'])}**; root-macro exact rate: **{pct(score['root_equal_exact'])}**.",
          f"- Runtime: {score['cost']['live_wall_seconds']:.1f}s including {score['cost']['model_load_wall_seconds']:.1f}s model load; prompt tokens {score['cost']['known_prompt_tokens']:,}, output tokens {score['cost']['known_completion_tokens']:,}.",
          "", "## By relation", "", "| Relation | Questions | Exact calls / calls | Call exact | Question macro | Pairwise (ordering) |", "|---|---:|---:|---:|---:|---:|"]
    for rel, row in analysis["by_relation"].items():
        pair = f"{row['correct_pairs']}/{row['total_pairs']} ({pct(row['pair_weighted_accuracy'])})" if "pair_weighted_accuracy" in row else "—"
        md.append(f"| {rel} | {row['questions']} | {row['exact_calls']}/{row['calls']} | {pct(row['call_weighted_exact_rate'])} | {pct(row['question_macro_exact_rate'])} | {pair} |")
    md += ["", "## By root", "", "| Root | Questions | Exact calls / calls | Call exact | Question macro |", "|---|---:|---:|---:|---:|"]
    for root, row in analysis["by_root"].items():
        md.append(f"| {root} | {row['questions']} | {row['exact_calls']}/{row['calls']} | {pct(row['call_weighted_exact_rate'])} | {pct(row['question_macro_exact_rate'])} |")
    md += ["", "## Multi-select gold cardinality", "", "| Gold count | Questions | Exact calls / calls | Exact rate |", "|---:|---:|---:|---:|"]
    for count, row in analysis["multi_select_by_gold_count"].items():
        md.append(f"| {count} | {row['questions']} | {row['exact_calls']}/{row['calls']} | {pct(row['call_weighted_exact_rate'])} |")
    md += ["", "## Per-question results", "", "| Candidate | Type / relation | Gold count | Exact calls / calls | Exact | FP / FN decisions or pairwise |", "|---|---|---:|---:|---:|---:|"]
    for q in qs:
        aux = f"FP {q['fp_calls']} / FN {q['fn_calls']}" if q['question_type'] == 'multi_select' else f"{q['correct_pairs']}/{q['total_pairs']} pairs ({pct(q['pair_accuracy'])})"
        md.append(f"| {q['candidate_id']} | {q['question_type']} / {q['relation_target']} | {q['gold_count'] if q['gold_count'] is not None else '—'} | {q['exact_calls']}/{q['denominator_calls']} | {pct(q['call_exact_rate'])} | {aux} |")
    md += ["", "## Reading the result", "",
           "Performance varied sharply by relation and root. Quantitative-relation MC was the weakest group in this small sample (8/54 exact calls); causal and constraint-conjunction MC were higher (45/54 each). Multi-select exactness by Gold count was 35/54 for 2, 26/54 for 3, 48/54 for 4, and 24/54 for 5. These are descriptive only: each stratum has three questions and root/content differ, so they do not establish that a relation or Gold count caused the change.",
           "", "Ordering questions had 71/120 exact calls (59.2%); pairwise ordering accuracy was 813/888 (91.6%). The gap means the model often got many relative pairs right while missing the complete order. Ordering arity was also uneven: two three-item questions, six four-item, two five-item, and one six-item question.",
           "", "This run measures the M-clean Solver's direct-answer behavior only. It does not show that training helps, that the new formats generalize, or that feedback caused learning. No CoT or memory condition was included. R3-ORD2 and R4-ORD2 are explicit clock-value sorting tasks, not event timelines.", ""]
    out_md = RUN / "REPORT.md"
    out_md.write_text("\n".join(md))
    print(json.dumps({"analysis": str(out_json), "report": str(out_md),
        "overall": analysis["overall"], "by_relation": analysis["by_relation"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
