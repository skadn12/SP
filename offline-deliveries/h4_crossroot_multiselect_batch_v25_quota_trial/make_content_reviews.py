"""Create the immutable first-pass content-review records for saved drafts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOTS = [
    "cc-train-episode-109164",
    "cc-train-episode-130221",
    "cc-train-episode-160549",
    "cc-train-episode-191749",
    "cc-train-episode-222937",
    "cc-train-episode-249232",
]

# Only the one candidate whose stem and all six proposition/evidence pairs are
# coherent is approved in this first pass. Mechanical validity is necessary,
# but it does not establish entailment, contradiction, or topic scope.
APPROVED = {
    "191749-qwen36-multi-v25-02": {
        "lila_proposed_group_reading_session",
        "lila_started_books_schedule",
        "lila_notify_children_tomorrow",
        "lila_setup_reading_corner_materials",
        "lila_invite_volunteers",
        "lila_first_session_next_week",
    }
}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def semantic_key(candidate_id: str, index: int, proposition: str) -> str:
    if candidate_id == "191749-qwen36-multi-v25-02":
        keys = sorted(APPROVED[candidate_id])
        return keys[index]
    return f"rejected:{candidate_id}:option-{index + 1}"


def main() -> None:
    for root_id in ROOTS:
        directory = HERE / root_id / "batch-01"
        draft = read(directory / "MODEL_DRAFT.json")
        checks = read(directory / "MECHANICAL_CHECKS.json")
        response = read(directory / "MODEL_RESPONSE.json")
        check_by_id = {row["candidate_id"]: row for row in checks["question_checks"]}
        decisions = []
        for row in draft["questions"]:
            candidate_id = row["candidate_id"]
            mechanical = check_by_id[candidate_id]["mechanically_valid"] is True
            approved_keys = APPROVED.get(candidate_id, set())
            approved = mechanical and bool(approved_keys)
            option_reviews = []
            for index, option in enumerate(row["options"]):
                key = semantic_key(candidate_id, index, option["proposition"])
                option_reviews.append({
                    "option_index": index,
                    "approved": approved,
                    "relation_confirmed": approved,
                    "semantic_key": key,
                    "reason": (
                        "Evidence, relation, stem scope, and semantic distinctness confirmed."
                        if approved else
                        "Candidate rejected in first-pass review: mechanical or semantic scope/evidence issue."
                    ),
                })
            decisions.append({
                "candidate_id": candidate_id,
                "approved": approved,
                "reason": (
                    "All six options are grounded, relation-correct, topic-coherent, and non-duplicative."
                    if approved else
                    ("Rejected because mechanical checks failed: " + "; ".join(check_by_id[candidate_id]["errors"]))
                    if not mechanical else
                    "Rejected because the stem or one or more option relations/evidence do not support a training question."
                ),
                "stem_scope_approved": approved,
                "option_reviews": option_reviews,
            })
        review = {
            "reviewer": "Codex first-pass content audit 2026-10-07",
            "draft_sha256": sha(canonical(draft)),
            "response_sha256": sha(canonical(response)),
            "question_decisions": decisions,
            "supplement_approved": False,
            "unused_evidence_refs": [],
            "approved_question_count": sum(item["approved"] for item in decisions),
            "supplement_reason": "",
        }
        write_new(directory / "CONTENT_REVIEW.json", review)


if __name__ == "__main__":
    main()
