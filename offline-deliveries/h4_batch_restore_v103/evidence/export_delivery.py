"""Export exact frozen content and an index of retained failures; no model calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DELIVERY = Path(__file__).resolve().parents[1]
SHARD = DELIVERY / "shards/shard_01"


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    frozen = read(SHARD / "BATCH_REVIEW.COMPLETE.json")
    assert frozen["status"] == "FROZEN" and frozen["before_any_solver"] is True
    accounting = read(DELIVERY / "evidence/FINAL_BATCH_ACCOUNTING.json")
    assert accounting["status"] == "PASS"
    pool_path = SHARD / "EVALUATION_POOL.json"
    questions = read(pool_path)["questions"]
    config = read(SHARD / "CONFIG.json")
    controls = {slot["parent_candidate_id"] for slot in config["multiselect_config"]["slots"]}
    new = [q for q in questions if q["candidate_id"] not in controls]
    control_rows = [q for q in questions if q["candidate_id"] in controls]
    assert len(questions) == 11 and len(new) == 9 and len(control_rows) == 2
    assert Counter(q["question_type"] for q in new) == Counter(multi_select=8, ordering=1)
    assert Counter(len(q["gold"]) for q in new if q["question_type"] == "multi_select") == Counter({2: 2, 3: 2, 4: 2, 5: 2})
    assert all(q["question_type"] == "multi_select" and len(q["gold"]) == 6 for q in control_rows)
    common = {"version": "v103_exact_content_export_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "origin_pool": bind(pool_path), "review_freeze": bind(SHARD / "BATCH_REVIEW.COMPLETE.json"),
              "content_reviewed": True, "solver_evaluated": False, "difficulty_selected": False,
              "training_allowed": False, "formal_training_admission": False,
              "limitation": "Two exposed source roots; multiselect children share two parent families. Content approval is not difficulty validation or a large-scale reliability estimate."}
    save(DELIVERY / "NEW_CONTENT_CANDIDATES.json", {**common, "question_count": len(new), "questions": new})
    save(DELIVERY / "CONTROL_CANDIDATES.json", {**common, "question_count": len(control_rows), "questions": control_rows})
    # Copy only exact frozen source text into a readable view, including rejected
    # source 14304, which must remain visibly distinct from READY sources.
    root_rows = []
    run = DELIVERY / "expansion_engine/runs/four_roots_B_scene01"
    for root in read(run / "MANIFEST.json")["roots"]:
        number = root.rsplit("-", 1)[-1]
        source_path = DELIVERY / f"sources/root{number}_SOURCE.json"
        review_path = DELIVERY / f"inputs/root{number}_SOURCE_REVIEW.json"
        output_path = DELIVERY / f"sources/root{number}_READABLE.md"
        if source_path.exists():
            source, review = read(source_path), read(review_path)
            text = [f"# {root} — {review['decision']}", "", "此文件逐字展示冻结累计对话；来源判断见配套 SOURCE_REVIEW.json。", ""]
            for turn in source["turns"]:
                text.extend([f"**{turn['turn_ref']} · {turn['speaker']}**", "", turn["text"], ""])
            with output_path.open("x") as stream:
                stream.write("\n".join(text))
            root_rows.append({"root_id": root, "decision": review["decision"], "source": bind(source_path),
                              "source_review": bind(review_path), "readable": bind(output_path)})
        else:
            root_rows.append({"root_id": root, "decision": "NO_NEW_SOURCE", "failure": bind(DELIVERY / f"evidence/expansion_failures/root{number}_FAILURE.json")})
    ordering_failures = []
    for path in sorted((SHARD / "ordering").glob("*/REJECTIONS.json")):
        for row in read(path):
            ordering_failures.append({"stage": "mechanical", **row, "record": bind(path)})
    ordering_review = read(SHARD / "ordering/CONTENT_REVIEW.json")
    for row in ordering_review["questions"]:
        if row.get("status") == "EXCLUDED":
            ordering_failures.append({"stage": "source_content_review", **row,
                                      "record": bind(SHARD / "ordering/CONTENT_REVIEW.json")})
    # Source judgments and the full review retain counterversions that failed
    # before composition, even when their parent family still yields all slots.
    asset_reviews = []
    for path in sorted((SHARD / "multiselect/assets").glob("*/rounds/initial/ASSET_REVIEW.json")):
        data = read(path)
        asset_reviews.append({"root_id": data["root_id"], "review": bind(path),
                              "source_judgments": bind(path.parent / "SOURCE_JUDGMENTS.json"),
                              "excluded_versions": [r for r in data["versions"] if not r["usable"]]})
    assert len(ordering_failures) == 3
    save(DELIVERY / "ALL_OUTCOMES_INDEX.json", {
        "version": "v103_all_outcomes_index_v1", "roots": root_rows,
        "new_content_candidates": bind(DELIVERY / "NEW_CONTENT_CANDIDATES.json"),
        "control_candidates": bind(DELIVERY / "CONTROL_CANDIDATES.json"),
        "all_assembled_multiselect_public": bind(SHARD / "multiselect/PUBLIC_QUESTIONS.json"),
        "all_mechanically_valid_ordering_public": bind(SHARD / "ordering/PUBLIC_QUESTIONS.json"),
        "ordering_failures": ordering_failures, "multiselect_asset_reviews": asset_reviews,
        "all_raw_expansion_requests_directory": str(run / "requests"),
        "all_raw_qa_requests_directory": str(SHARD),
        "all_stage_failures_and_attempts": bind(DELIVERY / "evidence/EXPANSION_STAGE_AND_PROBE_ACCOUNTING.json"),
        "accounting": bind(DELIVERY / "evidence/FINAL_BATCH_ACCOUNTING.json"),
        "raw_failures_preserved": True, "question_or_gold_edits": [], "training_started": False})
    print(json.dumps({"new_content": len(new), "controls": len(control_rows),
                      "ordering_failure_records": len(ordering_failures), "exported_exactly": True}))


if __name__ == "__main__":
    main()
