"""Freeze the finished replication and retain all earlier experiment records."""
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


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def binding_nodes(value):
    if isinstance(value, dict):
        if all(k in value for k in ("path", "sha256", "bytes")):
            yield value
        for child in value.values():
            yield from binding_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from binding_nodes(child)


def main():
    for name in ("REPORT_ZH.md", "NEW_QUESTIONS_AND_REASONS_ZH.md", "ROOT_VALIDATION.json",
                 "CONTENT_REVIEW_AI.json", "audit/MECHANICAL_AUDIT.json", "audit/EXECUTION_AUDIT.json",
                 "execution/ANALYSIS_SUMMARY.json", "execution/RAW_RESPONSE_FREEZE.json"):
        assert (HERE / name).is_file(), "MISSING_FINAL_ARTIFACT:" + name
    for name in ("audit/MECHANICAL_AUDIT.json", "audit/EXECUTION_AUDIT.json"):
        assert read(HERE / name)["status"].startswith("PASS"), "INDEPENDENT_AUDIT_NOT_PASS"
    history = read(HERE / "audit/HISTORICAL_BASELINE.json")
    historical = list(binding_nodes(history))
    for old in historical:
        assert all(bind(Path(old["path"]))[k] == old[k] for k in ("path", "sha256", "bytes")), "HISTORICAL_DRIFT:" + old["path"]
    prep = read(HERE / "execution/PREPARATION.json")
    summary = read(HERE / "execution/ANALYSIS_SUMMARY.json")
    raw = read(HERE / "execution/RAW_RESPONSE_FREEZE.json")
    validation = read(HERE / "ROOT_VALIDATION.json")
    paths = {p.resolve() for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "FINAL_MANIFEST.json"}
    paths.update(Path(b["path"]).resolve() for b in prep["bindings"])
    paths.update(Path(b["path"]).resolve() for b in historical)
    manifest = {"schema_version": "crossroot_question_quality_stability_final_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STABILITY_VALIDATION_COMPLETE_SEPARATE_CONTENT_AND_SHORTCUT_JUDGMENTS",
        "planned_questions": 28, "admitted_questions": summary["admitted_questions"], "roots": 4,
        "initial_content_pass": summary["initial_content_pass"], "gold_histogram": summary["new_mc_coverage"]["gold_count_histogram"],
        "initial_semantic_content_pass": summary["initial_semantic_content_pass"],
        "initial_combined_review_threshold_met": summary["initial_content_stability_threshold_met"],
        "validation_verdicts": validation["verdicts"],
        "solver_requests": summary["requests"], "returned_requests": raw["returned_requests"],
        "valid_requests": summary["valid"], "wall_seconds": raw["wall_seconds"],
        "new_dialogue_turns": 0, "new_training_runs": 0, "training_admission": False,
        "formal_reproduced_weakness_claim": False, "causal_feedback_benefit_established": False,
        "historical_bound_artifacts_verified": len(historical),
        "bindings": [bind(p) for p in sorted(paths)]}
    save_new(HERE / "FINAL_MANIFEST.json", manifest)
    navigation = {"schema_version": "question_quality_stability_latest_v1", "status": manifest["status"],
        "created_at_utc": manifest["created_at_utc"], "manifest": bind(HERE / "FINAL_MANIFEST.json"),
        "report": bind(HERE / "REPORT_ZH.md"), "examples": bind(HERE / "NEW_QUESTIONS_AND_REASONS_ZH.md"),
        "solver_requests": summary["requests"], "new_dialogue_turns": 0}
    save_new(HERE.parent / "QUESTION_QUALITY_STABILITY_LATEST.json", navigation)
    for bound in manifest["bindings"]:
        assert bind(Path(bound["path"])) == bound
    print(json.dumps({"status": manifest["status"], "verified_bindings": len(manifest["bindings"]),
        "historical_bindings_verified": len(historical), "manifest": str(HERE / "FINAL_MANIFEST.json")}))


if __name__ == "__main__":
    main()
