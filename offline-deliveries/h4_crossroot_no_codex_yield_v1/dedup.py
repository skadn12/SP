"""Bind provisional model-audited dedup to the frozen no-Codex Solver result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
ORDER = DELIVERIES / "h4_crossroot_ordering_no_codex_pilot_v1"
SOLVER = DELIVERIES / "h4_crossroot_ordering_no_codex_solver_v1"
FIRST = ORDER / "semantic_review_deepseek_v2" / "SEMANTIC_REVIEW.json"
GLM = ORDER / "semantic_review_glm_v4" / "SEMANTIC_REVIEW.json"
SOLVER_RESULT = SOLVER / "RESULTS.json"
POOL = SOLVER / "READY_FOR_EVALUATION.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def review_check(report: dict, qid: str, root: str, model: str) -> dict:
    require(report["reviewer_model_requested"] == model, "dedup_review_model_invalid")
    groups = [group for group in report["roots"] if group["root_id"] == root]
    require(len(groups) == 1, "dedup_review_root_invalid")
    group = groups[0]
    checks = [check for check in group["checks"] if check["candidate_id"] == qid]
    require(len(checks) == 1 and checks[0]["semantic_pass"] is True
            and group["reviewer_model_returned"] == model
            and not group["schema_or_binding_errors"],
            "dedup_review_candidate_invalid")
    verdict = checks[0]["review"]
    require(verdict["novel_vs_history"] is True
            and verdict["distinct_within_batch"] is True,
            "dedup_review_novelty_invalid")
    return verdict


def decide() -> dict:
    result, pool = read(SOLVER_RESULT), read(POOL)
    require(result["evaluation_data_status"] == "VALID"
            and result["counts"]["unresolved_questions"] == 0
            and result["generator_provenance"] == pool["provenance"]
            and set(result["questions"]) == {row["candidate_id"] for row in pool["questions"]},
            "dedup_solver_or_pool_invalid")
    selected = {qid: row for qid, row in result["questions"].items()
                if row["selected_by_policy"] is True}
    require(len(selected) == 1, "one_selected_candidate_required_for_this_pilot")
    qid, selected_row = next(iter(selected.items()))
    question = next(row for row in pool["questions"] if row["candidate_id"] == qid)
    root = question["root_id"]
    require(selected_row["canonical_question"] == question
            and selected_row["data_errors"] == []
            and selected_row["selection_reason"] == "GROUND_PASS_AND_FULL_PARTIAL_IN_RANGE",
            "dedup_selected_content_changed")
    first, independent = read(FIRST), read(GLM)
    deepseek_verdict = review_check(first, qid, root, "deepseek-ai/DeepSeek-V3.2")
    glm_verdict = review_check(independent, qid, root, "zai-org/GLM-5.2")
    prior_path = (DELIVERIES / "h4_crossroot_ordering_batch_v5_fresh_roots"
                  / root / "PRIOR_CANDIDATE_SUMMARY.json")
    prior = read(prior_path)
    original_path = Path(prior["original_pool_path"])
    require(prior["root_id"] == root and sha(original_path) == prior["original_pool_sha256"],
            "dedup_old_pool_context_changed")
    for name in ("semantic_review_deepseek_v2", "semantic_review_glm_v4"):
        index = read(ORDER / name / root / "INDEX.json")
        require(index["prior_summary_sha256"] == sha(prior_path)
                and index["label_candidate_file_sha256"]
                == sha(ORDER / "LABEL_REBOUND_UNREVIEWED.json")
                and qid in index["candidate_ids"],
                "dedup_review_context_changed:" + name)
    return {"version": "no_codex_auto_dedup_v1",
            "status": "PROVISIONAL_MODEL_AUDITED_SEMANTIC_DEDUP",
            "solver_result_sha256": {"ordering": sha(SOLVER_RESULT)},
            "candidate_id": qid, "root_id": root,
            "retained_ids": [qid], "held_ids": [],
            "basis": "Two independent reviewer models judged novelty against the frozen old pool; one selected candidate has no same-batch selected peer.",
            "dedup_certification": "MODEL_AUDITED_PROVISIONAL_NOT_HUMAN_CERTIFIED",
            "source_question_sha256": hashlib.sha256(json.dumps(question, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "prior_summary_sha256": sha(prior_path),
            "prior_pool_sha256": sha(original_path),
            "deepseek_review_sha256": sha(FIRST), "glm_review_sha256": sha(GLM),
            "deepseek_novel_vs_history": deepseek_verdict["novel_vs_history"],
            "glm_novel_vs_history": glm_verdict["novel_vs_history"],
            "same_batch_selected_peer_count": 0,
            "training_started": False}


if __name__ == "__main__":
    decision = decide()
    path = HERE / "DEDUP_DECISION.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": decision["status"], "retained_ids": decision["retained_ids"]}))
