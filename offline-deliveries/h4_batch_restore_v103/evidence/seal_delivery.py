"""Seal completed v103 artifacts after independent report and integrity checks."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

DELIVERY = Path(__file__).resolve().parents[1]
SHARD = DELIVERY / "shards/shard_01"
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    outputs = [DELIVERY / n for n in ("PRESERVATION_AFTER.json", "FINAL_AUDIT.json", "COMPLETE_MANIFEST.json")]
    assert not any(p.exists() for p in outputs)
    sys.path.insert(0, str(SHARD))
    import batch
    frozen = batch.verify_frozen(SHARD)
    assert frozen["by_type"] == {"multi_select": 10, "ordering": 1}
    before = read(DELIVERY / "PRESERVATION_BEFORE.json")
    assert len(before["files"]) == 4127
    mismatches = [r for r in before["files"] if not Path(r["path"]).is_file() or bind(r["path"])["sha256"] != r["sha256"]]
    assert not mismatches, mismatches
    reader_path = DELIVERY / "evidence/FINAL_REPORT_SECOND_READER_RECHECK.json"
    reader = read(reader_path)
    assert reader["status"] == "PASS"
    accounting = read(DELIVERY / "evidence/FINAL_BATCH_ACCOUNTING.json")
    assert accounting["status"] == "PASS"
    assert read(DELIVERY / "evidence/INDEPENDENT_LIVE_AUDIT.json")["status"] == "PASS"
    qa_calls = read(DELIVERY / "evidence/QA_CALL_ACCOUNTING.json")
    assert qa_calls["status"] == "PASS"
    assert qa_calls["all_v103_model_work"]["http_attempts"] == 29
    assert qa_calls["all_v103_model_work"]["usage"]["total_tokens"] == 488249
    pool = read(SHARD / "EVALUATION_POOL.json")["questions"]
    config = read(SHARD / "CONFIG.json")
    controls = {s["parent_candidate_id"] for s in config["multiselect_config"]["slots"]}
    new = read(DELIVERY / "NEW_CONTENT_CANDIDATES.json")
    control = read(DELIVERY / "CONTROL_CANDIDATES.json")
    assert new["questions"] == [q for q in pool if q["candidate_id"] not in controls]
    assert control["questions"] == [q for q in pool if q["candidate_id"] in controls]
    assert len(new["questions"]) == new["question_count"] == 9
    assert len(control["questions"]) == control["question_count"] == 2
    assert Counter(len(q["gold"]) for q in new["questions"] if q["question_type"] == "multi_select") == Counter({2: 2, 3: 2, 4: 2, 5: 2})
    assert sum(q["question_type"] == "ordering" for q in new["questions"]) == 1
    assert all(v["training_allowed"] is False and v["formal_training_admission"] is False and v["solver_evaluated"] is False for v in (new, control))
    outcome = read(DELIVERY / "ALL_OUTCOMES_INDEX.json")
    assert len(outcome["ordering_failures"]) == 3
    assert Counter(r["stage"] for r in outcome["ordering_failures"]) == Counter(mechanical=1, source_content_review=2)
    save(outputs[0], {"status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                      "protected_count": 4127, "unchanged_count": 4127, "mismatches": [],
                      "before": bind(DELIVERY / "PRESERVATION_BEFORE.json")})
    evidence_paths = [DELIVERY / n for n in ("COMPLETE_RESULTS_ZH.md", "NEW_CONTENT_CANDIDATES.json", "CONTROL_CANDIDATES.json",
                     "ALL_OUTCOMES_INDEX.json", "PRESERVATION_AFTER.json", "PRECALL_PLAN_ZH.md")]
    evidence_paths += [DELIVERY / "evidence" / n for n in ("FINAL_BATCH_ACCOUNTING.json", "INDEPENDENT_STATIC_AUDIT.json",
                       "INDEPENDENT_LIVE_AUDIT.json", "QA_CALL_ACCOUNTING.json", "FINAL_REPORT_SECOND_READER.json",
                       "FINAL_REPORT_SECOND_READER_RECHECK.json")]
    evidence_paths += [SHARD / "BATCH_REVIEW.COMPLETE.json", SHARD / "EVALUATION_POOL.json"]
    save(outputs[1], {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                      "scope": "Artifact integrity, source/QA reviews, immutable source-first judgments, export equality, report consistency and historical preservation. Does not certify automatic generation reliability or difficulty.",
                      "preselected_roots": 4, "model_accepted_new_turns": 45, "independently_ready_roots": 2,
                      "qa_admitted_new_turns": 30, "content_usable_including_controls": 11,
                      "new_content_candidates": 9, "multiselect_new": 8, "ordering_new": 1, "gold6_controls": 2,
                      "new_multiselect_by_gold": {"2": 2, "3": 2, "4": 2, "5": 2},
                      "http_attempts_including_side_probe": 29, "model_reported_total_tokens": 488249,
                      "solver_calls": 0, "training_started": False, "formal_training_admission": False,
                      "preserved_files": 4127,
                      "known_limits": ["Two of four exposed roots READY; not random/fresh holdout success-rate evidence.",
                                       "Eight multiselect children share two parent families.",
                                       "One of four ordering slots content-usable.",
                                       "Source and final question reviews not fully blind; exposures disclosed.",
                                       "One side-probe correct control failed; remaining four controls untested; checker not integrated.",
                                       "No Solver evaluation or training admission.",
                                       "Two local asset-finish commands initially failed on missing family_id; original judgments and failures retained before metadata-only correction.",
                                       "Export index initially omitted source-content ordering failures due adapter status key; initial index retained and corrected; frozen artifacts unchanged.",
                                       "Initial second-reader text comparison expected two final newlines; corrected EOF comparison passed with unchanged dialogue and Markdown, initial failure retained."],
                      "files": [bind(p) for p in evidence_paths]})
    paths = sorted(p for p in DELIVERY.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p != outputs[2])
    manifest = {"status": "SEALED", "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
                "listed_file_count_excluding_this_manifest": len(paths), "protected_file_count_including_this_manifest": len(paths) + 1,
                "excluded": ["__pycache__/**", "COMPLETE_MANIFEST.json self-hash"], "files": [bind(p) for p in paths]}
    save(outputs[2], manifest)
    assert all(bind(r["path"]) == r for r in manifest["files"])
    print(json.dumps({"status": "SEALED", "listed_files": len(paths), "protected_total": len(paths) + 1,
                      "manifest_sha256": bind(outputs[2])["sha256"], "historical_files_unchanged": 4127}))


if __name__ == "__main__":
    main()
