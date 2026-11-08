"""Freeze the two-pass automatically approved ordering questions for Solver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
PILOT = DELIVERIES / "h4_crossroot_ordering_no_codex_pilot_v1"
BATCH = DELIVERIES / "h4_crossroot_ordering_batch_v5_fresh_roots"
FIRST = PILOT / "semantic_review_deepseek_v2" / "SEMANTIC_REVIEW.json"
CHALLENGE = PILOT / "semantic_challenge_deepseek_v3" / "CHALLENGE_REVIEW.json"
INDEPENDENT = PILOT / "semantic_review_glm_v4" / "SEMANTIC_REVIEW.json"
REBOUND = PILOT / "LABEL_REBOUND_UNREVIEWED.json"
MODEL = "deepseek-ai/DeepSeek-V3.2"
INDEPENDENT_MODEL = "zai-org/GLM-5.2"
LABELS = tuple(f"I{number}" for number in range(1, 7))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def write_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def make_pool() -> tuple[dict, dict]:
    first, challenge, independent, rebound = (read(FIRST), read(CHALLENGE),
                                             read(INDEPENDENT), read(REBOUND))
    require(first.get("status") == "AUTOMATIC_SEMANTIC_REVIEW_COMPLETED"
            and challenge.get("status") == "ADVERSARIAL_SEMANTIC_CHALLENGE_COMPLETED"
            and first.get("no_codex_content_review") is True
            and challenge.get("no_codex_content_review") is True
            and first.get("reviewer_model_requested") == challenge.get("reviewer_model_requested") == MODEL
            and challenge.get("first_pass_review_sha256") == sha(FIRST),
            "automatic_review_provenance_invalid")
    require(independent.get("status") == "DIFFERENT_CHECKPOINT_SEMANTIC_REVIEW_COMPLETED"
            and independent.get("no_codex_content_review") is True
            and independent.get("reviewer_model_requested") == INDEPENDENT_MODEL,
            "independent_review_provenance_invalid")
    by_id = {question["candidate_id"]: question for question in rebound}
    require(len(by_id) == len(rebound), "rebound_duplicate_id")
    first_roots = {group["root_id"]: group for group in first["roots"]}
    challenge_roots = {group["root_id"]: group for group in challenge["roots"]}
    independent_roots = {group["root_id"]: group for group in independent["roots"]}
    require(set(first_roots) == set(independent_roots)
            and set(challenge_roots) == {root for root, group in first_roots.items()
                                         if any(row["semantic_pass"] is True
                                                for row in group["checks"])}
            and len(first_roots) == 4,
            "review_root_set_invalid")
    questions, audits = [], []
    for root, first_group in first_roots.items():
        challenge_group = challenge_roots.get(root)
        independent_group = independent_roots[root]
        source_path = BATCH / root / "source.json"
        source = read(source_path)
        source_hash = sha(source_path)
        root_ids = {qid for qid, row in by_id.items() if row["root_id"] == root}
        first_checks = {row["candidate_id"]: row for row in first_group["checks"]}
        challenge_checks = ({row["candidate_id"]: row for row in challenge_group["checks"]}
                            if challenge_group else {})
        independent_checks = {row["candidate_id"]: row for row in independent_group["checks"]}
        require(len(first_checks) == len(first_group["checks"])
                and len(challenge_checks) == len(challenge_group["checks"] if challenge_group else [])
                and set(first_checks) == root_ids
                and len(independent_checks) == len(independent_group["checks"])
                and set(independent_checks) == root_ids
                and set(challenge_checks) == {qid for qid, row in first_checks.items()
                                              if row["semantic_pass"] is True},
                "review_candidate_set_invalid:" + root)
        require(source["root_id"] == root
                and first_group["source_sha256"] == independent_group["source_sha256"] == source_hash
                and (challenge_group is None or challenge_group["source_sha256"] == source_hash)
                and first_group["candidate_file_sha256"] == sha(REBOUND)
                and first_group["reviewer_model_returned"] == MODEL
                and (challenge_group is None or challenge_group["reviewer_model_returned"] == MODEL)
                and independent_group["reviewer_model_returned"] == INDEPENDENT_MODEL
                and not first_group["schema_or_binding_errors"]
                and (challenge_group is None or not challenge_group["schema_or_binding_errors"])
                and not independent_group["schema_or_binding_errors"],
                "review_source_or_model_invalid:" + root)
        first_response = PILOT / "semantic_review_deepseek_v2" / root / "RESPONSE.json"
        challenge_response = PILOT / "semantic_challenge_deepseek_v3" / root / "RESPONSE.json"
        independent_response = PILOT / "semantic_review_glm_v4" / root / "RESPONSE.json"
        require(first_group["response_file_sha256"] == sha(first_response)
                and (challenge_group is None or challenge_group["response_file_sha256"] == sha(challenge_response))
                and independent_group["response_file_sha256"] == sha(independent_response),
                "review_response_changed:" + root)
        turns = {row["turn_ref"]: (index, row) for index, row in enumerate(source["turns"])}
        require(len(turns) == len(source["turns"]), "source_duplicate_turn_ref:" + root)
        for qid, row in challenge_checks.items():
            if (row.get("challenge_pass") is not True
                    or independent_checks[qid].get("semantic_pass") is not True):
                continue
            question = by_id[qid]
            require(first_checks[qid]["semantic_pass"] is True
                    and question["question_type"] == "ordering"
                    and set(question["items"]) == set(question["item_evidence"]) == set(LABELS)
                    and set(question["gold"]) == set(LABELS)
                    and len(question["gold"]) == 6,
                    "approved_question_contract_invalid:" + qid)
            positions = []
            for label in question["gold"]:
                evidence = question["item_evidence"][label]
                require(evidence["turn_ref"] in turns
                        and evidence["quote"] in turns[evidence["turn_ref"]][1]["text"],
                        "approved_evidence_invalid:" + qid + ":" + label)
                positions.append(turns[evidence["turn_ref"]][0])
            require(positions == sorted(set(positions)), "approved_gold_order_invalid:" + qid)
            questions.append({**question, "source_path": str(source_path),
                              "source_sha256": source_hash})
            paths = (FIRST, CHALLENGE, INDEPENDENT, REBOUND, source_path, first_response,
                     challenge_response, independent_response, HERE / "build.py", HERE / "run_solver.py",
                     HERE / "report_results.py")
            audits.append({"derived_candidate_id": qid, "root_id": root,
                           "source_sha256": source_hash,
                           "source_turn_count": len(source["turns"]),
                           "original_model_provenance": {"returned_model": "Qwen/Qwen3.6-35B-A3B"},
                           "original_files_sha256": {str(path): sha(path) for path in paths}})
    require(questions, "no_two_pass_approved_questions")
    pool = {"status": "AUTOMATIC_REVIEWED_LABEL_REBOUND_READY_FOR_SOLVER",
            "candidate_count": len(questions),
            "provenance": {"requested_model": "Qwen/Qwen3.6-35B-A3B",
                           "returned_model": "Qwen/Qwen3.6-35B-A3B",
                           "content_review": "independent_model_two_pass",
                           "reviewer_model": MODEL,
                           "normalization": "deterministic_label_rebind",
                           "first_review_sha256": sha(FIRST),
                           "challenge_review_sha256": sha(CHALLENGE),
                           "independent_review_sha256": sha(INDEPENDENT)},
            "questions": questions, "training_started": False}
    audit = {"status": "AUTOMATIC_REVIEW_INTERSECTION_BOUND",
             "candidate_count": len(audits), "candidates": audits,
             "first_review_sha256": sha(FIRST), "challenge_review_sha256": sha(CHALLENGE),
             "independent_review_sha256": sha(INDEPENDENT)}
    return pool, audit


if __name__ == "__main__":
    pool, audit = make_pool()
    write_once(HERE / "READY_FOR_EVALUATION.json", pool)
    write_once(HERE / "AUTOMATIC_REVIEW_AUDIT.json", audit)
    print(json.dumps({"status": "READY_FOR_EVALUATION", "candidate_count": len(pool["questions"]),
                      "candidate_ids": [row["candidate_id"] for row in pool["questions"]]}))
