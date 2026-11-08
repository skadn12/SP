"""CPU-only check for the v5.1 operation metadata overlay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
QUESTIONS = BASE / "questions"

def sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main() -> None:
    revision = json.loads((HERE / "OPERATION_METADATA_REVISION.json").read_text())
    changes = {item["candidate_id"]: item for item in revision["changes"]}
    assert len(changes) == 3
    for candidate_id, item in changes.items():
        question = json.loads((QUESTIONS / (candidate_id + ".json")).read_text())
        assert question["candidate_id"] == candidate_id
        assert question["question_type"] == "ordering"
        assert question["task_operation"] == item["old_task_operation"]
        assert item["new_task_operation"] == "ORDER_ENQUIRY_PROGRESSION"
        assert len(question["gold"]) == len(question["items"]) == 4
        assert set(question["display_order"]) == set(question["items"])
        assert question["source_read_scope"]["all_turns_read"] is True
    unchanged = json.loads((QUESTIONS / "cc-train-episode-19708__ORDER2.json").read_text())
    assert unchanged["task_operation"] == revision["unchanged_ordering_operation"]["task_operation"]
    assert revision["policy"]["base_question_files_unchanged"] is True
    assert revision["policy"]["base_solver_scores_reusable"] is False
    print(json.dumps({"status": "PASS_CPU_METADATA_OVERLAY", "changed": len(changes), "unchanged": 1}))

if __name__ == "__main__":
    main()
