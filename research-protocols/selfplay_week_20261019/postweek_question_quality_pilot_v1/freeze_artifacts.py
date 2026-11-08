"""Freeze the completed diagnostic and add a new navigation entry."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def bind(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    required = ["REPORT_ZH.md", "NEW_QUESTIONS_AND_REASONS_ZH.md", "ROOT_VALIDATION.json",
                "EXECUTION_REVIEW_AI.json", "SOLVER_REASON_REVIEW_AI.json",
                "MATERIAL_GAP_FEEDBACK.json", "execution/ANALYSIS_SUMMARY.json"]
    assert all((HERE / name).is_file() for name in required), "REQUIRED_COMPLETION_ARTIFACT_MISSING"
    before = read(HERE / "HISTORICAL_FREEZE_BEFORE.json")
    for previous in before["bindings"]:
        assert bind(Path(previous["path"])) == previous, "HISTORICAL_ARTIFACT_CHANGED"
        if Path(previous["path"]).name == "FINAL_MANIFEST.json":
            for child in read(Path(previous["path"]))["bindings"]:
                assert bind(Path(child["path"])) == child
    paths = {p.resolve() for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.name != "FINAL_MANIFEST.json" and p.suffix != ".pyc"}
    prep = read(HERE / "execution/PREPARATION.json")
    paths.update(Path(b["path"]).resolve() for b in prep["bindings"])
    paths.update(Path(b["path"]).resolve() for b in before["bindings"])
    summary = read(HERE / "execution/ANALYSIS_SUMMARY.json")
    freeze = read(HERE / "execution/RAW_RESPONSE_FREEZE.json")
    manifest = {
        "schema_version": "question_quality_pilot_final_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PILOT_COMPLETE_ORDERING_SOURCE_DEPENDENCE_UNRESOLVED",
        "scope": "train_only_exploratory_minimum_question_repair",
        "new_questions": 11, "historical_questions_reassessed": 9,
        "roots": 5, "source_versions": 6,
        "new_multiselect_gold_histogram": summary["new_mc_coverage"]["gold_count_histogram"],
        "coverage_complete": summary["new_mc_coverage"]["coverage_complete"],
        "solver_role": "B1_step56_merged",
        "solver_requests": summary["requests"], "returned_requests": freeze["returned_requests"],
        "valid_requests": summary["valid"], "wall_seconds": freeze["wall_seconds"],
        "new_dialogue_turns": 0, "new_training_runs": 0,
        "formal_reproduced_solver_weakness_claim": False,
        "causal_feedback_benefit_established": False,
        "training_admission": False,
        "original_pending_reason_packets_preserved": True,
        "independent_reason_review": bind(HERE / "SOLVER_REASON_REVIEW_AI.json"),
        "independent_execution_review": bind(HERE / "EXECUTION_REVIEW_AI.json"),
        "historical_manifests_and_61_bindings_unchanged": True,
        "bindings": [bind(path) for path in sorted(paths)],
    }
    save(HERE / "FINAL_MANIFEST.json", manifest)
    navigation = {
        "schema_version": "question_quality_pilot_latest_v1",
        "created_at_utc": manifest["created_at_utc"], "status": manifest["status"],
        "manifest": bind(HERE / "FINAL_MANIFEST.json"),
        "report": bind(HERE / "REPORT_ZH.md"),
        "examples": bind(HERE / "NEW_QUESTIONS_AND_REASONS_ZH.md"),
        "new_multiselect_gold_histogram": manifest["new_multiselect_gold_histogram"],
        "solver_requests": summary["requests"],
    }
    save(HERE.parent / "QUESTION_QUALITY_PILOT_LATEST.json", navigation)
    for item in manifest["bindings"]:
        assert bind(Path(item["path"])) == item
    for key in ("manifest", "report", "examples"):
        assert bind(Path(navigation[key]["path"])) == navigation[key]
    print(json.dumps({"status": manifest["status"], "verified_bindings": len(manifest["bindings"]),
                      "manifest": str(HERE / "FINAL_MANIFEST.json")}))


if __name__ == "__main__":
    main()
