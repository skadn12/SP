"""Assemble the audited two-root pool without modifying source artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
sys.path.insert(0, str(DELIVERIES / "h4_crossroot_multiselect_solver_selection_v2"))
from protocol import LABELS, read, sha, validate_pool


V11 = DELIVERIES / "h4_crossroot_multiselect_targeted_revision_v11/cc-train-episode-136202"
READY_254852 = DELIVERIES / "h4_crossroot_multiselect_content_audit_254852_v1/READY_FOR_EVALUATION.json"
AUDIT_254852 = DELIVERIES / "h4_crossroot_multiselect_content_audit_254852_v1/CONTENT_AUDIT.json"


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    audit = read(HERE / "CONTENT_AUDIT.json")
    v11_validation = read(V11 / "VALIDATION.json")
    v11_candidate = read(V11 / "CANDIDATE_PENDING_AUDIT.json")
    prior_pool = read(READY_254852)
    prior_audit = read(AUDIT_254852)
    if (v11_validation["status"] != "MECHANICALLY_VALID_CONTENT_AUDIT_PENDING"
            or v11_candidate["status"] != v11_validation["status"]
            or v11_validation["returned_model"] != "Qwen/Qwen3.6-35B-A3B"
            or prior_pool["status"] != "CONTENT_AUDITED_READY_FOR_SOLVER"
            or prior_audit["decision"] != "CONTENT_AUDITED_READY_FOR_SOLVER"):
        raise ValueError("SOURCE_AUDIT_OR_MODEL_STATUS_INVALID")
    if (sha(V11 / "source.json") != v11_validation["source_sha256"]
            or sha(V11 / "RESPONSE.json") != v11_validation["response_sha256"]):
        raise ValueError("V11_SOURCE_OR_RESPONSE_CHANGED")

    first = dict(v11_candidate["question"])
    first["source_path"] = str(V11 / "source.json")
    first["source_sha256"] = v11_validation["source_sha256"]
    second = prior_pool["questions"][0]
    questions = [first, second]
    if [question["candidate_id"] for question in questions] != list(audit["candidate_reviews"]):
        raise ValueError("AUDIT_CANDIDATES_MISMATCH")
    patterns = []
    for question in questions:
        gold = question["gold"]
        pattern = "".join(label for label in LABELS if label in gold)
        review = audit["candidate_reviews"][question["candidate_id"]]
        if (not 2 <= len(gold) <= 6 or len(gold) != len(set(gold))
                or set(gold) != set(pattern) or gold != review["gold"]
                or (len(gold) < 6 and pattern == "".join(LABELS[:len(gold)]))):
            raise ValueError("GOLD_COUNT_OR_POSITION_INVALID:" + question["candidate_id"])
        patterns.append(pattern)
    if len(set(patterns)) != len(patterns):
        raise ValueError("GOLD_PATTERNS_NOT_DISTINCT")

    pool = {
        "version": "h4_crossroot_multiselect_content_audit_two_roots_v1",
        "status": "CONTENT_AUDITED_READY_FOR_SOLVER",
        "candidate_count": len(questions),
        "provenance": {
            "provider": "siliconflow",
            "requested_model": "Qwen/Qwen3.6-35B-A3B",
            "returned_model": "Qwen/Qwen3.6-35B-A3B",
            "136202_original_generation_version": "h4_crossroot_multiselect_batch_v8",
            "136202_final_revision_version": "h4_crossroot_multiselect_targeted_revision_v11",
            "136202_revision_response_id": v11_validation["response_id"],
            "136202_revision_response_sha256": v11_validation["response_sha256"],
            "254852_prior_ready_pool": str(READY_254852),
            "254852_prior_ready_pool_sha256": sha(READY_254852),
            "254852_prior_content_audit_sha256": sha(AUDIT_254852),
            "codex_role": "CONTENT_AUDIT_AND_POOL_ASSEMBLY",
            "solver_evaluated": False,
            "training_started": False,
        },
        "gold_position_summary": {
            "patterns": {question["root_id"]: pattern for question, pattern in zip(questions, patterns)},
            "patterns_distinct": True,
            "two_to_five_non_prefix": True,
            "six_correct_position_exception": False,
        },
        "content_audit_path": str(HERE / "CONTENT_AUDIT.json"),
        "content_audit_sha256": sha(HERE / "CONTENT_AUDIT.json"),
        "questions": questions,
    }
    write_once(HERE / "READY_FOR_EVALUATION.json", pool)
    validate_pool(HERE / "READY_FOR_EVALUATION.json")
    print(json.dumps({"status": pool["status"], "candidate_count": len(questions),
                      "gold_patterns": patterns, "solver_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
