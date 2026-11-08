"""Verify source, display and execution integrity; do not certify semantics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("production_batch_integrity_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
read, bind, check, save = run.read, run.bind, run.check, run.save

HISTORICAL_QUESTION_FILES = (
    "day3_solver187_v1/inputs_private_v2/PRIVATE_QUESTIONS.json",
    "day3_diversity_crossroot_v1/exports/PRIVATE_QUESTIONS.json",
    "day3_targeted_4root_v1/exports/PRIVATE_QUESTIONS.json",
    "day3_feedback_realization_v1/exports/PRIVATE_QUESTIONS.json",
    "day3_minimal_repair_control_v1/exports/PRIVATE_QUESTIONS.json",
    "postweek_question_quality_patch_v1/audit/HISTORICAL_CANDIDATES_PRIVATE.json",
    "postweek_question_quality_pilot_v1/CANDIDATES_PRIVATE.json",
    "postweek_question_quality_stability_v1/CANDIDATES_PRIVATE.json",
)


def question_signature(c):
    choices = c["options" if c["question_type"] == "multi_select" else "items"]
    return run.base.fingerprint({"type": c["question_type"], "question": c["question"],
                                 "choices": sorted(choices.values())})


def main(completed=False):
    prep, rows, tokens = run.public_preparation()
    private = read(HERE / "execution/PRIVATE_SCORING.json")
    check(prep["private_scoring_binding"])
    for b in private["bindings"]:
        check(b)
    candidates = {c["candidate_id"]: c for c in read(HERE / "CANDIDATES_PRIVATE.json")}
    reviews = {r["candidate_id"]: r for r in read(HERE / "CONTENT_REVIEW_AI.json")["reviews"]}
    admitted = {cid: c for cid, c in candidates.items() if reviews[cid]["decision"] == "PASS"}
    scoring = {r["request_id"]: r for r in private["requests"]}
    assert len(scoring) == len(rows) == 4 * len(admitted)
    assert {r["candidate_id"] for r in rows} == set(admitted)
    assert all(run.production_gate.validate_review(c, reviews[cid], HERE / "SELECTION_FREEZE.json")["admitted"]
               for cid, c in admitted.items())
    replay = {cid: dict((variant, (public, mapping)) for variant, public, mapping in run.displays(c))
              for cid, c in admitted.items()}
    gold_independent, distinct = [], []
    for cid, c in admitted.items():
        field = "options" if c["question_type"] == "multi_select" else "items"
        other = dict(c)
        # A valid alternative Gold may change scoring but cannot change display.
        other["gold"] = list(c[field]) if c["question_type"] == "multi_select" else c["gold"][1:] + c["gold"][:1]
        if other["gold"] == c["gold"]:
            other["gold"] = list(c[field])[:2]
        changed = list(run.displays(other))
        assert all(p == replay[cid][v][0] and m["display_to_canonical"] == replay[cid][v][1]["display_to_canonical"]
                   for v, p, m in changed), "GOLD_DEPENDENT_DISPLAY"
        permutations = [tuple(m["display_to_canonical"].values()) for _, m in replay[cid].values()]
        assert len(set(permutations)) == 2 and all(p != tuple(c[field]) for p in permutations)
        gold_independent.append(cid)
        distinct.append(cid)
    for row in rows:
        expected_public, expected_private = replay[row["candidate_id"]][row["variant_id"]]
        field = "options" if row["question_type"] == "multi_select" else "items"
        public = {k: row[k] for k in expected_public if k != field}
        public[field] = row["display"]
        assert public == expected_public and scoring[row["request_id"]]["presentation"] == expected_private
        prompt = Path(row["prompt_path"]).read_text()
        assert "<box>A B C</box>" in prompt and "item_evidence" in prompt
        assert not any(k in row for k in ("gold", "item_analysis", "ordering_analysis", "author_id", "checks"))
        assert len(tokens[row["request_id"]]) + run.MAX_TOKENS <= run.MODEL_LEN
        if row["condition"] == "QUESTION_ONLY":
            assert "Dialogue:\n" not in prompt
        else:
            source = read(row["source_binding"]["path"])
            assert all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source["turns"])
    historical_bindings, historical, n = [], {}, 0
    for rel in HISTORICAL_QUESTION_FILES:
        p = HERE.parent / rel
        doc = read(p)
        old = doc if isinstance(doc, list) else next(doc[k] for k in ("questions", "candidates") if k in doc)
        for c in old:
            # All files enumerated here are historical Train inventories.
            assert c["root_id"].startswith("cc-train-")
            historical.setdefault(question_signature(c), []).append(c["candidate_id"])
            n += 1
        historical_bindings.append(bind(p))
    duplicates = [{"candidate_id": cid, "historical_ids": historical[question_signature(c)]}
                  for cid, c in candidates.items() if question_signature(c) in historical]
    assert not duplicates, duplicates
    baseline = read(HERE / "audit/HISTORICAL_BASELINE.json")
    for b in baseline["bindings"]:
        check(b)
    result = {
        "schema_version": "production_batch_integrity_v1",
        "status": "PASS_EXECUTION_INTEGRITY" if completed else "PASS_PREPARED_INTEGRITY",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_id": "/root", "semantic_certification": False,
        "admitted_questions": len(admitted), "requests": len(rows),
        "gold_independent_counterfactual_checks": len(gold_independent),
        "two_distinct_nonidentity_displays": len(distinct),
        "every_full_prompt_contains_all_source_turns": True,
        "public_request_field_allowlist_pass": True,
        "sources_reviews_and_display_mappings_replayed": True,
        "historical_baseline_bindings_unchanged": len(baseline["bindings"]),
        "whole_question_duplicate_audit": {"historical_rows": n,
            "historical_unique_signatures": len(historical), "duplicates": duplicates,
            "policy": "Exact question text and complete unordered item texts plus type; labels ignored. Partial overlap retained.",
            "scope": "Eight enumerated historical Train question files, not every artifact or semantic paraphrase.",
            "bindings": historical_bindings},
        "bindings": [bind(HERE / "run.py"), bind(Path(__file__)), bind(HERE / "execution/PREPARATION.json"),
                     bind(HERE / "CANDIDATES_PRIVATE.json"), bind(HERE / "CONTENT_REVIEW_AI.json")],
    }
    if completed:
        _, _, _, raw_freeze, _ = run.base.sealed_inputs()
        scores_doc = read(HERE / "execution/SCORES.json")
        scores = scores_doc["records"]
        assert scores_doc["gold_read_after_all_raw_sealed"]
        assert scores_doc["raw_freeze_binding"] == bind(HERE / "execution/RAW_RESPONSE_FREEZE.json")
        assert [s["request_id"] for s in scores] == [r["request_id"] for r in rows]
        recomputed = []
        for row, score in zip(rows, scores):
            expected = run.score_record(row, scoring[row["request_id"]]["presentation"], score["display_prediction"], score["valid"])
            assert all(score[k] == value for k, value in expected.items())
            recomputed.append(score)
        summary = read(HERE / "execution/ANALYSIS_SUMMARY.json")
        assert summary == run.summarize(recomputed, private)
        assert summary["planned_questions"] == 70 and summary["requests"] == len(rows)
        assert summary["initial_content_pass_rate"] == summary["initial_content_pass"] / 70
        result.update(raw_requests=raw_freeze["requests"], returned_requests=raw_freeze["returned_requests"],
                      score_records_recomputed=len(recomputed), all_raw_bindings_verified=True,
                      raw_freeze_binding=bind(HERE / "execution/RAW_RESPONSE_FREEZE.json"),
                      invalid_response_counts=dict(Counter(s["invalid_error"] for s in scores if not s["valid"])),
                      all_frozen_requests_kept_in_accuracy_denominators=True)
    output = HERE / "audit" / ("COMPLETED_INTEGRITY.json" if completed else "PREPARED_INTEGRITY.json")
    save(output, result)
    print(json.dumps({"status": result["status"], "questions": len(admitted), "requests": len(rows),
                      "historical_rows_checked": n, "historical_bindings_unchanged": len(baseline["bindings"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepared", "completed"))
    main(parser.parse_args().stage == "completed")
