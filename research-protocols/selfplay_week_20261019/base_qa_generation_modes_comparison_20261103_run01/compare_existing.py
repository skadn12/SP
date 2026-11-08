"""Descriptive comparison of already completed question-generation runs."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
OLD = WEEK / "base_qa_v2_2_1_cardinality_solver_eval_20261103"
NEW = WEEK / "base_qa_free_generation_solver_eval_20261103_run01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary(doc, qid, mode):
    rows = [r for r in doc["requests"] if r["candidate_id"] == qid and r["context_mode"] == mode]
    assert len(rows) == 18 and all(r["valid"] for r in rows)
    js = [Fraction(r["J_fraction"]) for r in rows]
    mean = sum(js, Fraction()) / len(rows)
    return {"calls": len(rows), "exact": sum(r["exact_correct"] for r in rows),
            "exact_accuracy": sum(r["exact_correct"] for r in rows) / len(rows),
            "mean_J_fraction": str(mean), "mean_J": float(mean), "valid": len(rows)}


def main():
    a, b = read(OLD / "SCORES.json"), read(NEW / "SCORES.json")
    old_items = {i["candidate_id"]: i for i in read(OLD / "ITEMS_PRIVATE.json")}
    new_items = {i["candidate_id"]: i for i in read(NEW / "ITEMS_PRIVATE.json")}
    comparisons = []
    for old_id, new_id in [("fresh_b_179186_01", "free_reading_plans"), ("c_harbor", "free_travel_notes")]:
        x, y = old_items[old_id], new_items[new_id]
        assert x["root_id"] == y["root_id"]
        assert x["source_sha256_bytes"] == y["source_sha256"]
        row = {"root_id": x["root_id"], "source_sha256": y["source_sha256"], "same_full_source": True,
               "old_gold_count": len(x["gold_option_ids"]), "new_gold_count": len(y["gold_option_ids"]),
               "old_question": old_id, "new_question": new_id}
        for name, doc, qid in [("cardinality_exploration", a, old_id), ("free_authoring", b, new_id)]:
            data = {mode: summary(doc, qid, mode) for mode in ("full_context", "no_context")}
            gain = Fraction(data["full_context"]["mean_J_fraction"]) - Fraction(data["no_context"]["mean_J_fraction"])
            data.update(full_minus_no_context_J=float(gain), full_minus_no_context_J_fraction=str(gain))
            row[name] = data
        comparisons.append(row)
    paths = [OLD / name for name in ("SCORES.json", "MODEL_CONFIG.json", "ITEMS_PRIVATE.json")]
    paths += [NEW / name for name in ("SCORES.json", "MODEL_CONFIG.json", "ITEMS_PRIVATE.json", "INDEPENDENT_SCORE_AUDIT.json")]
    paths += [WEEK / "base_qa_v2_2_1_fresh_cardinality_20261103" / name for name in ("CANDIDATES_COMPLETE.json", "REPORT.md")]
    paths += [WEEK / "base_qa_free_generation_20261103_run01" / name for name in ("CANDIDATES_FIRST_DRAFT.json", "REPORT.md", "AUTHOR_QUALITY_OBSERVATIONS.json")]
    output = {"type": "descriptive_existing_results_not_randomized_mode_experiment", "pairs": comparisons,
              "new_model_calls": 0, "new_questions": 0, "sorting_excluded_from_multi_select_comparison": True,
              "confounds": ["English vs Chinese questions", "different semantic targets within each root",
                            "Gold cardinality differs in root179186", "different seeds and display assignments",
                            "few independently authored question targets; repeated solver samples are not new questions",
                            "different author and blind-review context setup"],
              "input_bindings": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    with (HERE / "COMPARISON.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
