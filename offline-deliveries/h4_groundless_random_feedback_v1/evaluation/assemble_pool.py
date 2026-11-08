"""Assemble the immutable multi-select and Qwen-reviewed ordering pools."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


MODEL = "Qwen/Qwen3.6-35B-A3B"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(multi_path: Path, ordering_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("OUTPUT_EXISTS_NO_OVERWRITE")
    multi = read(multi_path)
    ordering = read(ordering_path)
    multi_questions = copy.deepcopy(multi.get("questions", []))
    ordering_questions = copy.deepcopy(ordering.get("questions", []))
    if not multi_questions or not ordering_questions:
        raise ValueError("BOTH_POOLS_MUST_BE_NONEMPTY")
    for row in ordering_questions:
        root = row.get("root_id")
        source = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_crossroot_ordering_batch_v5_fresh_roots") / root / "source.json"
        if not source.is_file():
            raise ValueError("ORDERING_SOURCE_MISSING:" + str(root))
        row["source_path"] = str(source)
        row["source_sha256"] = sha(source)
    questions = multi_questions + ordering_questions
    ids = [row.get("candidate_id") for row in questions]
    if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("CANDIDATE_IDS_NOT_UNIQUE")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "h4_groundless_random_feedback_solver_pool_v1",
        "status": "ASSEMBLED_NO_SOLVER_CALLS",
        "provenance": {
            "requested_model": MODEL,
            "returned_model": MODEL,
            "semantic_review_performed": False,
            "codex_question_edits": False,
            "ordering_qwen_review": "only freshly parsed Qwen review artifacts; rejected roots omitted",
            "multi_select_pool": str(multi_path.resolve()),
            "multi_select_pool_sha256": sha(multi_path),
            "ordering_reviewed_pool": str(ordering_path.resolve()),
            "ordering_reviewed_pool_sha256": sha(ordering_path),
            "ground_calls": 0,
        },
        "candidate_count": len(questions),
        "questions": questions,
        "solver_calls": 0,
        "training_started": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": result["status"], "candidate_count": len(questions),
            "multi_select_count": len(multi_questions), "ordering_count": len(ordering_questions),
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", type=Path, required=True)
    parser.add_argument("--ordering", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.multi.resolve(), args.ordering.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
