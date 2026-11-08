"""Bind independent, non-Gold-blind reviews without changing authored questions."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def bind(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def main():
    candidates = read(HERE / "CANDIDATES_PRIVATE.json")
    by_id = {c["candidate_id"]: c for c in candidates}
    paths = [HERE / part / "CONTENT_REVIEW_AI.json" for part in ("review_mc", "review_order")]
    docs = [read(p) for p in paths]
    reviews = [r for doc in docs for r in doc["reviews"]]
    assert len(reviews) == len(by_id) == len(candidates)
    assert {r["candidate_id"] for r in reviews} == set(by_id)
    for review in reviews:
        candidate = by_id[review["candidate_id"]]
        assert review["candidate_sha256"] == fingerprint(candidate)
        assert review["author_id"] == candidate["author_id"]
        assert review["reviewer_id"] != review["author_id"]
        assert review["source_binding"] == candidate["source_binding"]
        assert review["decision"] in ("PASS", "HOLD", "FAIL")
        reconstructed = review.get("reconstructed_gold", review.get("independent_reconstructed_gold"))
        assert reconstructed is not None, "Independent Gold reconstruction missing"
        if review["decision"] == "PASS":
            if candidate["question_type"] == "multi_select":
                assert set(reconstructed) == set(candidate["gold"])
            else:
                assert reconstructed == candidate["gold"]
        if candidate.get("parent_candidate_id"):
            assert isinstance(review["parent_baseline_gold_valid"], bool)
    document = {
        "schema_version": "quality_pilot_independent_content_review_v1",
        "reviewer_type": "AI_separate_author_and_reviewer_agents",
        "gold_blind": False,
        "disclosure_zh": "审核者与相应作者不同，完整读取绑定来源；接触作者私有Gold，不是Gold盲审；未依据本批Solver成绩审核。",
        "bindings": [bind(p) for p in paths] + [bind(HERE / "CANDIDATES_PRIVATE.json")],
        "reviews": reviews,
        "summary": {decision: sum(r["decision"] == decision for r in reviews)
                    for decision in ("PASS", "HOLD", "FAIL")},
        "content_pass_proves_difficulty_gain": False,
        "training_admission": False,
    }
    with (HERE / "CONTENT_REVIEW_AI.json").open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(document["summary"]))


if __name__ == "__main__":
    main()
