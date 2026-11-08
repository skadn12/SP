"""Prepare the bounded follow-up request list without loading or calling a model."""

from __future__ import annotations

import json
from pathlib import Path

from reanalyze import HERE, RUN, SOURCE, VERSION, load, question_sha256, sha256


def main() -> None:
    manifest = load(RUN / "manifest.json")
    questions = {q["candidate_id"]: q for q in load(SOURCE / "candidates.json")["candidates"]
                 if q["question_type"] == "multi_select"}
    baseline = [row for row in manifest["requests"] if row["candidate_id"] in questions
                and row["variant"]["variant_id"] in {f"perm_{i:02d}" for i in range(6)}]
    if len(baseline) != 24 or any(row["slot"] != 0 for row in baseline):
        raise ValueError("expected_two_questions_six_displays_two_modes_at_slot_zero")
    requests = []
    for index, row in enumerate(baseline):
        for slot in (1, 2):
            variant = row["variant"]["variant_id"]
            requests.append({**row,
                             "request_id": f"{row['candidate_id']}__{variant}__{row['mode']}__s{slot:02d}",
                             "slot": slot, "seed": 2026100500 + index * 2 + slot,
                             "baseline_request_id": row["request_id"],
                             "question_sha256": question_sha256(questions[row["candidate_id"]]),
                             "legacy_diagnostic_variant_id": f"shift_{int(variant[-2:])}"})
    path = HERE / "followup_48_requests_revision_2.json"
    output = {
        "policy_version": VERSION, "status": "PREPARED_NOT_EXECUTED", "request_count": len(requests),
        "model": manifest["model"], "decoding": {key: manifest[key] for key in
                                                   ("temperature", "top_p", "max_tokens", "enable_thinking")},
        "baseline_manifest": {"path": str(RUN / "manifest.json"), "sha256": sha256(RUN / "manifest.json")},
        "source_input_sha256": manifest["input_sha256"],
        "selection_rule": "All six forward cyclic displays chosen before any new results; no score-based selection.",
        "unselected_reverse_displays": "Retained as supplemental evidence, not counted as same-display repeats.",
        "output_directory": str(HERE / "followup_k3_revision_2"),
        "preconditions": [
            "Explicit authorization to run Solver inference; prior offline-only instruction remains active.",
            "Verify exact model/checkpoint, prompt construction and frozen question/source versions.",
            "No immutable model hash exists in baseline: establish identity before merging old and new rows; otherwise stop and propose a fresh complete evaluation.",
            "Rebuilt prompt hashes must equal the frozen baseline prompt hashes.",
            "Use existing solver contract without adding Gold-count hints or exposing review findings.",
        ],
        "decision_scope": "Complete existing k=3 mechanism diagnostics; no guaranteed PASS and no automatic training promotion.",
        "requests": requests,
    }
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(path), len(requests), "prepared; no inference executed")


if __name__ == "__main__":
    main()
