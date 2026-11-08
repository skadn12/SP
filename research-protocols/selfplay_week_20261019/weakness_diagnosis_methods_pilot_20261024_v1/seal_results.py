"""Seal new pilot artifacts and recheck all old experiment bindings read-only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

P = Path(__file__).resolve().parent
EXCLUDED = {"FINAL_MANIFEST.json", "FINAL_MANIFEST_CHECK.json"}


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def check_binding(expected):
    actual = binding(expected["path"])
    assert actual["sha256"] == expected["sha256"], (expected["path"], "hash mismatch")
    if "bytes" in expected:
        assert actual["bytes"] == expected["bytes"], (expected["path"], "size mismatch")
    return actual


def write(name, data):
    (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    checks = []
    report_review = read(P / "FINAL_REPORT_REVIEW.json")
    assert report_review.get("status") == "PASS", "Final independent report review must pass"
    checks.append({"check": "final_report_review", "passed": True})
    report_binding = read(P / "FINAL_STATUS.json")["report_binding"]
    check_binding(report_binding)
    check_binding(read(P / "FINAL_STATUS.json")["summary_binding"])
    comparison = read(P / "COMPARISON.json")
    assert comparison["scope"]["questions"] == 6
    assert len(comparison["per_card"]) == 12
    assert all(comparison["arms"][arm]["usable_candidates"] == 0 for arm in ("A", "B"))
    checks.append({"check": "six_questions_twelve_reviews_zero_promoted", "passed": True})
    stages = {}
    for name in ("solver", "diagnosis", "diagnosis_compat_v2"):
        freeze = read(P / name / "RAW_RESPONSE_FREEZE.json")
        checked = 0
        for key, value in freeze.items():
            if isinstance(value, dict) and {"path", "sha256"}.issubset(value):
                check_binding(value)
                checked += 1
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and {"path", "sha256"}.issubset(entry):
                        check_binding(entry)
                        checked += 1
        assert set(freeze["status_counts"]).issubset({"RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED"})
        assert sum(freeze["status_counts"].values()) == freeze["requests"]
        assert len(freeze["bindings"]) == freeze["requests"]
        stages[name] = {"bindings_checked": checked, "requests": freeze["requests"],
                        "status_counts": freeze["status_counts"]}
    checks.append({"check": "raw_runs_and_terminal_statuses_preserved", "passed": True})
    preparation = read(P / "PREPARATION.json")
    old_manifest_binding = preparation["old_manifest_binding"]
    check_binding(old_manifest_binding)
    old_manifest = read(old_manifest_binding["path"])
    old_counts = {}
    dependencies = {old_manifest_binding["path"]: old_manifest_binding}
    for group in ("artifacts", "dependencies"):
        for entry in old_manifest[group]:
            check_binding(entry)
            dependencies[entry["path"]] = entry
        old_counts[group] = len(old_manifest[group])
    checks.append({"check": "old_manifest_and_all_old_artifacts_dependencies_unchanged", "passed": True,
                   "counts": old_counts})
    for key in ("protocol_binding", "selection_binding", "solver_jobs_binding", "code_binding", "old_selection_binding"):
        check_binding(preparation[key])
        if not Path(preparation[key]["path"]).is_relative_to(P):
            dependencies[preparation[key]["path"]] = preparation[key]
    checks.append({"check": "frozen_preparation_bindings_unchanged", "passed": True})
    artifacts = [binding(path) for path in sorted(P.rglob("*"))
                 if path.is_file() and path.relative_to(P).as_posix() not in EXCLUDED]
    manifest = {"schema_version": "diagnostic_methods_pilot_final_manifest_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": str(P), "artifacts": artifacts,
                "dependencies": sorted(dependencies.values(), key=lambda row: row["path"]),
                "excluded_relative_paths": sorted(EXCLUDED)}
    write("FINAL_MANIFEST.json", manifest)
    reread = read(P / "FINAL_MANIFEST.json")
    for entry in reread["artifacts"] + reread["dependencies"]:
        check_binding(entry)
    actual_paths = {str(path) for path in P.rglob("*")
                    if path.is_file() and path.relative_to(P).as_posix() not in EXCLUDED}
    assert actual_paths == {row["path"] for row in reread["artifacts"]}
    checks.append({"check": "all_new_artifacts_and_dependencies_rehashed_complete", "passed": True})
    write("FINAL_MANIFEST_CHECK.json", {"status": "PASS", "checks": checks,
                                       "artifacts": len(artifacts), "dependencies": len(dependencies),
                                       "old_manifest_checks": old_counts, "stages": stages,
                                       "manifest_binding": binding(P / "FINAL_MANIFEST.json"),
                                       "notes": ["DIAGNOSIS_SCORES mechanical.semantic_review=PENDING is a preserved pre-review snapshot; completed reviews are in COMPARISON and semantic_review/SUMMARY.",
                                                 "No model inference, training, candidate promotion or changes to old experiment performed by this seal script."]})
    print(json.dumps({"status": "PASS", "artifacts": len(artifacts), "dependencies": len(dependencies),
                      "old_counts": old_counts, "stages": stages}, ensure_ascii=False))


if __name__ == "__main__":
    main()
