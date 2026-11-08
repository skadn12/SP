"""Combine independently authored/reviewed records without modifying drafts."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bind(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def records(doc):
    return doc if isinstance(doc, list) else doc["reviews"]


def main():
    candidates, reviews, bindings = [], [], []
    for arm in ("a", "b"):
        final_path = HERE / f"author_{arm}/CANDIDATES_PRIVATE.json"
        draft_path = HERE / f"author_{arm}/CANDIDATES_PRIVATE_DRAFT.json"
        review_path = HERE / f"review_{arm}/CONTENT_REVIEW_AI.json"
        initial_path = HERE / f"review_{arm}/INITIAL_CONTENT_REVIEW_AI.json"
        final, drafts = read(final_path), read(draft_path)
        initial = {r["candidate_id"]: r for r in records(read(initial_path))}
        final_reviews = records(read(review_path))
        draft_ids = {c["candidate_id"]: c for c in drafts}
        assert len(final) == len(drafts) == len(initial) == len(final_reviews) == 14
        assert set(draft_ids) == set(initial) == {c["candidate_id"] for c in final}
        by_id = {r["candidate_id"]: r for r in final_reviews}
        for c in final:
            cid = c["candidate_id"]; review = by_id[cid]
            assert review["candidate_sha256"] == fingerprint(c), "FINAL_REVIEW_DRIFT:" + cid
            assert review["author_id"] == c["author_id"] != review["reviewer_id"], "AUTHOR_REVIEWER_NOT_SEPARATE"
            initial_decision = initial[cid]["decision"]
            initial_content_decision = initial[cid]["initial_content_decision"]
            assert initial_decision in ("PASS", "HOLD", "FAIL")
            assert initial_content_decision in ("PASS", "HOLD", "FAIL")
            assert review.get("initial_decision", initial_decision) == initial_decision
            review = dict(review, initial_decision=initial_decision,
                initial_content_decision=initial_content_decision,
                initial_review_binding=bind(initial_path), initial_candidate_sha256=fingerprint(draft_ids[cid]),
                repaired_after_initial_review=fingerprint(c) != fingerprint(draft_ids[cid]))
            candidates.append(c); reviews.append(review)
        bindings.extend(bind(p) for p in (final_path, draft_path, review_path, initial_path))
    candidates.sort(key=lambda c:c["candidate_id"])
    reviews.sort(key=lambda r:r["candidate_id"])
    assert len({c["candidate_id"] for c in candidates}) == 28
    signatures = []
    for c in candidates:
        field = "options" if c["question_type"] == "multi_select" else "items"
        signatures.append(fingerprint({"question_type": c["question_type"], "question": c["question"], "choices": sorted(c[field].values())}))
    assert len(set(signatures)) == len(signatures), "WHOLE_QUESTION_DUPLICATE_PRESENT"
    save(HERE / "CANDIDATES_PRIVATE.json", candidates)
    save(HERE / "CONTENT_REVIEW_AI.json", {"schema_version": "crossroot_independent_content_review_v1",
        "gold_blind": False, "reviewer_is_human": False, "solver_results_read": False,
        "author_reviewer_separation": True, "reviews": reviews, "bindings": bindings})
    save(HERE / "COLLECTION_AUDIT.json", {"planned_candidates": 28, "submitted_candidates": len(candidates),
        "whole_question_exact_duplicates": 0, "new_dialogue_turns": 0,
        "initial_pass": sum(r["initial_decision"] == "PASS" for r in reviews),
        "initial_semantic_content_pass": sum(r["initial_content_decision"] == "PASS" for r in reviews),
        "initial_pass_definition": "Combined content and source-binding review; semantic-only content is separately counted.",
        "final_pass": sum(r["decision"] == "PASS" for r in reviews),
        "repaired_candidates": [r["candidate_id"] for r in reviews if r["repaired_after_initial_review"]],
        "bindings": bindings + [bind(HERE / "CANDIDATES_PRIVATE.json"), bind(HERE / "CONTENT_REVIEW_AI.json")]})
    print(json.dumps(read(HERE / "COLLECTION_AUDIT.json"), ensure_ascii=False))


if __name__ == "__main__":
    main()
