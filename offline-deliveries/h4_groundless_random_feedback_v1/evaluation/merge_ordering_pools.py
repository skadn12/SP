"""Merge independently reviewed ordering pool fragments without overwriting them."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge(base_path: Path, fragments: list[Path], output: Path) -> dict:
    if output.exists():
        raise RuntimeError("OUTPUT_EXISTS_NO_OVERWRITE")
    base = read(base_path)
    questions = copy.deepcopy(base.get("questions", []))
    existing = {row["candidate_id"] for row in questions}
    fragment_meta = []
    for path in fragments:
        fragment = read(path)
        for row in fragment.get("questions", []):
            root = row.get("root_id")
            source = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_crossroot_ordering_batch_v5_fresh_roots") / root / "source.json"
            if not source.is_file():
                raise ValueError("ORDERING_SOURCE_MISSING:" + str(root))
            row["source_path"] = str(source)
            row["source_sha256"] = sha(source)
            if row["candidate_id"] in existing:
                raise ValueError("DUPLICATE_ORDERING_CANDIDATE:" + row["candidate_id"])
            questions.append(copy.deepcopy(row))
            existing.add(row["candidate_id"])
        fragment_meta.append({"path": str(path.resolve()), "sha256": sha(path),
                              "selected_count": fragment.get("selected_count", len(fragment.get("questions", []))),
                              "rejected_count": fragment.get("rejected_count")})
    result = copy.deepcopy(base)
    result["version"] = "h4_groundless_random_feedback_solver_pool_v2"
    result["status"] = "ASSEMBLED_NO_SOLVER_CALLS"
    result["candidate_count"] = len(questions)
    result["questions"] = questions
    result["provenance"]["ordering_reviewed_fragments"] = fragment_meta
    result["provenance"]["ground_calls"] = 0
    result["solver_calls"] = 0
    result["training_started"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": result["status"], "candidate_count": len(questions),
            "added_ordering": len(questions) - len(base.get("questions", [])), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.base.resolve(), [p.resolve() for p in args.fragment], args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
