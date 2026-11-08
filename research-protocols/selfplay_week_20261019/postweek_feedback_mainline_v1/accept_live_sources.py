"""Accept real reviewed sources, retaining the original extension audit chain."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
sys.path.insert(0, str(HERE.parents[2]))
from scripts.current_method import reason_material_feedback_pipeline as pipeline


def main():
    index = pipeline.read(HERE / "source_reviews/INDEX.json")
    reviews = {row["root_id"]: row["review_binding"] for row in index["source_reviews"]}
    run = HERE / "live_run_v2"
    accepted = []
    for root in pipeline.read(run / "RUN_CONFIG.json")["roots"]:
        root_id = root["root_id"]
        folder = run / root_id
        author_input = folder / "SOURCE_AUTHOR_INPUT.json"
        common = {"root_id": root_id, "author_input_binding": pipeline.binding(author_input),
            "author_identity_note": "A provenance-backed upstream author role is recorded; an exact historical service request ID is unavailable."}
        if root_id.endswith("104814"):
            submission = {**common, "mode": "EXTEND", "existing_extension_import": {
                "namespace": "T4R",
                "source_freeze_binding": pipeline.binding(WEEK / "day3_targeted_4root_v1/SOURCE_FREEZE.json"),
                "source_author_id": "historical_day3_targeted_4root_R2_source_author",
                "current_source_review_binding": reviews[root_id],
                "material_realization_review_binding": index["material_review_binding"]}}
        else:
            submission = {**common, "mode": "REUSE", "source_binding": root["source_binding"],
                "source_author_id": "historical_day3_production_source_authors",
                "review_binding": reviews[root_id]}
        submission_path = folder / "SOURCE_SUBMISSION.json"
        pipeline.write(submission_path, submission)
        result = pipeline.accept_source(folder, submission_path)
        accepted.append({"root_id": root_id, "mode": result["mode"],
            "question_ready": pipeline.binding(folder / "QUESTION_READY_SOURCE.json"),
            "source_binding": result["source_binding"], "new_source_generated": False})
    pipeline.write(HERE / "SOURCE_ACCEPTANCE_SUMMARY.json", {
        "status": "FOUR_SOURCES_QUESTION_READY", "active_run": str(run), "roots": accepted,
        "new_dialogue_generation_calls": 0, "historical_extension_reused": True,
        "historical_extension_generated_from_current_feedback": False,
        "source_reviews_are_independent_ai_attestations": True,
        "formal_weakness_admission": False, "training_admission": False})
    print({"accepted": len(accepted), "modes": [row["mode"] for row in accepted]})


if __name__ == "__main__":
    main()
