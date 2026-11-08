"""Deterministic assembly after claim-level content review; no model calls."""

from __future__ import annotations

import importlib.util
from itertools import combinations
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline.py"
spec = importlib.util.spec_from_file_location("pool_pipeline", PIPELINE)
assert spec and spec.loader
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)

GOLD_PATTERN = {
    2: "BE", 3: "ACE", 4: "BCEF", 5: "ABCEF", 6: "ABCDEF",
}
QUESTIONS = {
    "reunion": ("197635-qwen36-multi-25", "Reunion planning",
                "Which of the following statements did Taylor or Casey explicitly make in the R02 reunion-planning conversations? Select all that apply."),
    "workshop": ("197635-qwen36-multi-26", "Workshop participation and reflection",
                 "Which of the following statements did Taylor or Casey explicitly make in the R02 workshop conversations? Select all that apply."),
}


def reviewed_claims() -> tuple[dict, list[dict]]:
    _, _, prompt_input, _, _, _ = pool.preflight()
    response = pool.read(pool.RUN / "POOL_RESPONSE.json")
    draft, checks, review = [pool.read(pool.RUN / name) for name in
                             ("POOL_DRAFT.json", "POOL_CHECKS.json", "CONTENT_REVIEW.json")]
    if checks["fatal_errors"] or checks["draft_sha256"] != pool.sha(pool.encoded(draft)):
        raise ValueError("pool_draft_or_checks_invalid")
    if review.get("draft_sha256") != checks["draft_sha256"] or review.get("response_sha256") != pool.sha(pool.encoded(response)):
        raise ValueError("content_review_not_bound_to_response")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("reviewer_missing")
    all_claims = prompt_input["reviewed_reusable_claims_candidates_only"] + draft["claims"]
    by_id = {item["claim_id"]: item for item in all_claims}
    if len(by_id) != len(all_claims):
        raise ValueError("claim_ids_not_unique")
    decisions = review.get("claim_decisions")
    if not isinstance(decisions, list):
        raise ValueError("claim_decisions_missing")
    keyed = {}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {
                "claim_id", "approved", "reason", "semantic_key", "selected_fact_duplicate"}:
            raise ValueError("review_decision_schema_invalid")
        cid = decision["claim_id"]
        if cid in keyed or not isinstance(decision["approved"], bool) or not isinstance(decision["reason"], str):
            raise ValueError("review_decision_invalid_or_duplicate")
        if not isinstance(decision["selected_fact_duplicate"], bool):
            raise ValueError("selected_duplicate_decision_invalid")
        keyed[cid] = decision
    if set(keyed) != set(by_id):
        raise ValueError("content_review_must_cover_every_claim")
    mechanically_valid = {item["claim_id"] for item in checks["new_claim_checks"] if not item["errors"]}
    mechanically_valid.update(item["claim_id"] for item in prompt_input["reviewed_reusable_claims_candidates_only"])
    approved, seen_semantics = [], set()
    for cid, claim in by_id.items():
        decision = keyed[cid]
        if not decision["approved"]:
            continue
        semantic_key = decision["semantic_key"]
        if (cid not in mechanically_valid or decision["selected_fact_duplicate"] or
                not isinstance(semantic_key, str) or not semantic_key.strip()):
            raise ValueError(f"approved_claim_invalid:{cid}")
        semantic_key = pool.normalized(semantic_key)
        if semantic_key in seen_semantics:
            raise ValueError(f"approved_semantic_duplicate:{cid}")
        seen_semantics.add(semantic_key)
        approved.append(claim)
    return review, approved


def six_for_scene(claims: list[dict], scene: str) -> list[dict] | None:
    scene_claims = [x for x in claims if x["scene"] == scene]
    positive = sorted((x for x in scene_claims if x["relation"] == "supports"),
                      key=lambda x: x["claim_id"])
    negative = sorted((x for x in scene_claims if x["relation"] == "contradicts"),
                      key=lambda x: x["claim_id"])
    for false_count in (2, 1, 3, 4, 0):
        true_count = 6 - false_count
        if len(positive) < true_count or len(negative) < false_count:
            continue
        for true_items in combinations(positive, true_count):
            true_refs = {item["turn_ref"] for item in true_items}
            if len(true_refs) != true_count:
                continue
            for false_items in combinations(negative, false_count):
                chosen = list(true_items + false_items)
                if len({item["turn_ref"] for item in chosen}) == 6:
                    return chosen
    return None


def assemble() -> dict:
    review, claims = reviewed_claims()
    _, _, prompt_input, manifest, _, _ = pool.preflight()
    source = pool.read(pool.RUN / "source.json")
    v16, _, _ = pool.dependency()
    _, v8, _ = v16.base()
    rows, used = [], {}
    for scene, (candidate_id, topic, question) in QUESTIONS.items():
        chosen = six_for_scene(claims, scene)
        if chosen is None:
            continue
        positive = [x for x in chosen if x["relation"] == "supports"]
        negative = [x for x in chosen if x["relation"] == "contradicts"]
        gold = GOLD_PATTERN[len(positive)]
        ordered = {}
        for label, claim in zip((x for x in pool.LABELS if x in gold), positive):
            ordered[label] = claim
        for label, claim in zip((x for x in pool.LABELS if x not in gold), negative):
            ordered[label] = claim
        row = {"candidate_id": candidate_id, "root_id": pool.ROOT, "question_type": "multi_select",
               "topic": topic, "question": question,
               "options": {label: ordered[label]["proposition"] for label in pool.LABELS},
               "gold": [label for label in pool.LABELS if label in gold],
               "option_evidence": {label: {"turn_ref": ordered[label]["turn_ref"],
                                           "quote": ordered[label]["quote"],
                                           "relation": ordered[label]["relation"],
                                           "reason": ordered[label]["rationale"]}
                                   for label in pool.LABELS},
               "answer_reason": "Each option is labeled according to its reviewed direct support or same-speaker contradiction in the cited R02 turn."}
        shadow = dict(row, candidate_id=f"{pool.ROOT[-6:]}-qwen36-multi-01")
        errors = v8.check_candidate(shadow, pool.ROOT, source["turns"])
        if len({item["turn_ref"] for item in row["option_evidence"].values()}) != 6:
            errors.append("repeated_turn_within_question")
        if errors:
            raise ValueError(f"assembled_candidate_mechanical_failure:{candidate_id}:{errors}")
        rows.append(row)
        used[candidate_id] = {label: ordered[label]["claim_id"] for label in pool.LABELS}
    result = {"status": "ASSEMBLED_DRAFT_FULL_CONTENT_REVIEW_REQUIRED",
              "root_id": pool.ROOT, "source_sha256": manifest["source_sha256"],
              "pool_draft_sha256": pool.sha(pool.encoded(pool.read(pool.RUN / "POOL_DRAFT.json"))),
              "claim_content_review_sha256": pool.sha(pool.encoded(review)),
              "approved_claim_count": len(claims), "question_count": len(rows),
              "shortfall_from_two": 2 - len(rows), "used_claim_ids": used,
              "same_batch_semantic_dedup_required": True,
              "full_question_review_required": True, "solver_calls": 0, "training_started": False}
    pool.write_new(pool.RUN / "ASSEMBLED_DRAFT.json", {"questions": rows})
    pool.write_new(pool.RUN / "ASSEMBLY_CHECKS.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(assemble(), ensure_ascii=False, indent=2))
