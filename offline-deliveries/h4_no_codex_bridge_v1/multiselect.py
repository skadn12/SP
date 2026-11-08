"""Bind fail-closed model review of a multi-select batch to Solver input."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path


LABELS = tuple("ABCDEF")
POLICY = "multiselect_quote_cues_v3"
CONTRADICTION_CUE = re.compile(
    r"\b(?:not|no|never|none|without|only|instead|rather|less|cannot|can't|"
    r"didn't|doesn't|won't|wouldn't|isn't|wasn't|other than)\b", re.I)
REFERENT_CUE = re.compile(r"\b(?:it|this|that|her|him|them|they|these|those)\b", re.I)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def model_reviews(response: dict, model: str, expected_ids: set[str]) -> dict:
    require(response.get("http_status") == 200, "review_http_status_invalid")
    envelope = json.loads(response["raw_body"])
    require(envelope.get("model") == model, "review_returned_model_invalid")
    choices = envelope.get("choices")
    require(isinstance(choices, list) and len(choices) == 1
            and choices[0].get("finish_reason") == "stop", "review_finish_invalid")
    content = choices[0]["message"]["content"].strip()
    if content.startswith("```json\n") and content.endswith("\n```"):
        content = content[len("```json\n"):-len("\n```")]
    parsed = json.loads(content)
    require(isinstance(parsed, dict) and set(parsed) == {"reviews"}
            and isinstance(parsed["reviews"], list), "review_schema_invalid")
    reviews = parsed["reviews"]
    require(len(reviews) == len(expected_ids), "review_count_invalid")
    keyed = {row["candidate_id"]: row for row in reviews}
    require(len(keyed) == len(reviews) and set(keyed) == expected_ids,
            "review_candidate_set_invalid")
    return keyed


def verify_judgment(judge: dict, original: dict) -> tuple[bool, dict]:
    fields = {"candidate_id", "stem_scope_ok", "novelty_ok", "batch_distinct_ok",
              "gold_consistent", "overlap_candidate_ids", "reason", "options"}
    require(isinstance(judge, dict) and set(judge) == fields,
            "judge_fields_invalid")
    flags = ("stem_scope_ok", "novelty_ok", "batch_distinct_ok", "gold_consistent")
    require(all(type(judge[k]) is bool for k in flags)
            and isinstance(judge["reason"], str) and bool(judge["reason"].strip())
            and isinstance(judge["overlap_candidate_ids"], list)
            and all(isinstance(x, str) for x in judge["overlap_candidate_ids"]),
            "judge_metadata_invalid")
    options = judge["options"]
    require(isinstance(options, list) and len(options) == 6,
            "judge_option_count_invalid")
    for index, option in enumerate(options, 1):
        require(isinstance(option, dict)
                and set(option) == {"index", "verdict", "quote_sufficient", "reason"}
                and type(option["index"]) is int and option["index"] == index
                and option["verdict"] in ("supports", "contradicts", "unsupported")
                and type(option["quote_sufficient"]) is bool
                and isinstance(option["reason"], str) and bool(option["reason"].strip()),
                "judge_option_invalid")
    source_options = original["options"]
    require(len(source_options) == 6, "source_option_count_invalid")
    relation_ok = all(j["quote_sufficient"] and j["verdict"] == o["relation"]
                      for j, o in zip(options, source_options))
    contradiction_cue_ok = all(o["relation"] != "contradicts"
                               or CONTRADICTION_CUE.search(o["quote"]) is not None
                               for o in source_options)
    self_contained_quote_ok = all(REFERENT_CUE.search(o["quote"]) is None
                                  for o in source_options)
    flags_result = {"relation_ok": relation_ok,
                    "contradiction_cue_ok": contradiction_cue_ok,
                    "self_contained_quote_ok": self_contained_quote_ok}
    return all(judge[k] for k in flags) and all(flags_result.values()), flags_result


def normalize(question: dict, root: str, index: int) -> dict:
    options = question["options"]
    positive = [o for o in options if o["relation"] == "supports"]
    negative = [o for o in options if o["relation"] == "contradicts"]
    require(len(positive) + len(negative) == 6 and 2 <= len(positive) <= 6,
            "approved_gold_cardinality_invalid")
    seed = int(sha_bytes(question["candidate_id"].encode())[:8], 16)
    patterns = list(combinations(range(6), len(positive)))
    positions = set(patterns[(seed + index) % len(patterns)])
    positives, negatives = iter(positive), iter(negative)
    ordered = [next(positives) if i in positions else next(negatives) for i in range(6)]
    return {"candidate_id": question["candidate_id"], "root_id": root,
            "question_type": "multi_select", "topic": question["topic"],
            "question": question["question"],
            "options": {label: option["proposition"] for label, option in zip(LABELS, ordered)},
            "gold": [LABELS[i] for i in range(6) if i in positions],
            "option_evidence": {label: {"turn_ref": option["turn_ref"],
                                        "quote": option["quote"],
                                        "relation": option["relation"],
                                        "reason": option["rationale"]}
                                for label, option in zip(LABELS, ordered)},
            "answer_reason": "Gold labels correspond to model-reviewed supported propositions; other options have explicit same-speaker contradictions."}


def build(batch_dir: Path, review_dir: Path, *, policy: str = POLICY) -> dict:
    require(policy == POLICY, "unsupported_review_policy")
    source_files = ("MODEL_DRAFT.json", "MECHANICAL_CHECKS.json", "source.json",
                    "DEDUP_SUMMARY.json", "MODEL_RESPONSE.json")
    prepared, payload, request, result = (read(review_dir / name) for name in
                                          ("PREPARED.json", "INPUT.json", "REQUEST.json",
                                           "REVIEW_RESULT_POLICY_V3.json"))
    draft, checks, source, response = (read(batch_dir / name) for name in
                                       ("MODEL_DRAFT.json", "MECHANICAL_CHECKS.json",
                                        "source.json", "MODEL_RESPONSE.json"))
    root = draft["root_id"]
    require(root == source["root_id"] == checks["root_id"]
            == prepared["root_id"] == payload["root_id"] == result["root_id"],
            "root_binding_invalid")
    require(prepared["source_hashes"] == {name: sha(batch_dir / name)
                                            for name in source_files},
            "review_source_hashes_invalid")
    require(checks["fatal_errors"] == []
            and checks["status"] == "MECHANICAL_CHECKS_ONLY_CONTENT_REVIEW_REQUIRED"
            and checks["draft_sha256"] == sha_bytes(canonical(draft))
            and checks["response_sha256"] == sha_bytes(canonical(response))
            and checks["returned_model"] == checks["requested_model"],
            "mechanical_or_generator_binding_invalid")
    require(prepared["input_sha256"] == sha_bytes(canonical(payload))
            and prepared["request_sha256"] == sha_bytes(canonical(request))
            and prepared["prompt_sha256"] == sha_bytes(
                request["messages"][0]["content"].encode())
            and request["messages"][0]["content"]
            == (review_dir / "PROMPT.txt").read_text(encoding="utf-8")
            and request["model"] == prepared["review_model"]
            and "codex" not in prepared["review_model"].casefold(),
            "review_request_binding_invalid")
    check_rows = checks["question_checks"]
    draft_rows = draft["questions"]
    require(len(check_rows) == len(draft_rows)
            and [r["candidate_id"] for r in check_rows]
            == [r["candidate_id"] for r in draft_rows]
            and len({r["candidate_id"] for r in draft_rows}) == len(draft_rows),
            "mechanical_candidate_set_invalid")
    originals = {row["candidate_id"]: row for row, check in zip(draft_rows, check_rows)
                 if check["mechanically_valid"] is True and not check["errors"]}
    require(len(originals) == checks["mechanically_valid_count"]
            == prepared["mechanically_valid_count"]
            == len(payload["candidate_questions"]), "mechanical_count_invalid")
    for compact in payload["candidate_questions"]:
        qid = compact["candidate_id"]
        require(qid in originals
                and compact == {"candidate_id": qid, "topic": originals[qid]["topic"],
                                "question": originals[qid]["question"],
                                "options": [{k: option[k] for k in
                                             ("proposition", "relation", "turn_ref", "speaker", "quote")}
                                            for option in originals[qid]["options"]]},
                "review_input_differs_from_draft:" + qid)
    require(len({q["candidate_id"] for q in payload["candidate_questions"]}) == len(originals),
            "review_input_candidate_set_invalid")
    if not originals:
        require(result["status"] == "NO_RESPONSE" and not result["approved_ids"]
                and not result["rejected_ids"] and not result["unresolved_ids"],
                "empty_review_result_invalid")
        approved_ids: set[str] = set()
        response_hash = None
    else:
        response_path = review_dir / "RESPONSE.json"
        require(result["status"] == "REVIEWED_FAIL_CLOSED"
                and result["response_sha256"] == sha(response_path)
                and result["returned_model"] == prepared["review_model"]
                and result["unresolved_ids"] == [],
                "review_result_binding_invalid")
        response_hash = sha(response_path)
        judgments = model_reviews(read(response_path), prepared["review_model"], set(originals))
        decisions = result["candidate_decisions"]
        require(isinstance(decisions, dict) and set(decisions) == set(originals),
                "review_decision_set_invalid")
        approved_ids = set()
        for qid, original in originals.items():
            decision = decisions[qid]
            expected, flags = verify_judgment(judgments[qid], original)
            require(decision == {"approved": expected, **flags, "judge": judgments[qid]},
                    "review_decision_differs_from_model_or_policy:" + qid)
            if expected:
                approved_ids.add(qid)
        require(set(result["approved_ids"]) == approved_ids
                and set(result["rejected_ids"]) == set(originals) - approved_ids
                and len(result["approved_ids"]) + len(result["rejected_ids"]) == len(originals),
                "review_outcome_set_invalid")
    turns = {turn["turn_ref"]: turn for turn in source["turns"]}
    require(len(turns) == len(source["turns"]), "source_turn_refs_duplicate")
    approved_rows = [row for row in draft_rows if row["candidate_id"] in approved_ids]
    for row in approved_rows:
        for option in row["options"]:
            turn = turns.get(option["turn_ref"])
            require(turn is not None and turn["speaker"] == option["speaker"]
                    and option["quote"] in turn["text"],
                    "approved_evidence_not_in_source:" + row["candidate_id"])
    source_path = (batch_dir / "source.json").resolve(strict=True)
    source_hash = sha(source_path)
    questions = [{**normalize(row, root, index), "source_path": str(source_path),
                  "source_sha256": source_hash}
                 for index, row in enumerate(approved_rows)]
    require(len({" ".join(o["proposition"].casefold().split())
                 for row in approved_rows for o in row["options"]})
            == len(approved_rows) * 6, "approved_exact_fact_duplicate")
    return {"version": "h4_no_codex_multiselect_bridge_v1",
            "status": "CONTENT_AUDITED_READY_FOR_SOLVER" if questions else "NO_APPROVED_CANDIDATES",
            "candidate_count": len(questions), "root_id": root,
            "provenance": {"requested_model": checks["requested_model"],
                           "returned_model": checks["returned_model"],
                           "reviewer_model": prepared["review_model"],
                           "no_codex_content_review": True,
                           "review_policy": POLICY,
                           "normalization": "deterministic_label_rebind",
                           "source_hashes": prepared["source_hashes"],
                           "review_prepared_sha256": sha(review_dir / "PREPARED.json"),
                           "review_result_sha256": sha(review_dir / "REVIEW_RESULT_POLICY_V3.json"),
                           "review_response_sha256": response_hash},
            "questions": questions, "solver_calls": 0, "training_started": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--review-policy", choices=(POLICY,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.batch_dir, args.review_dir, policy=args.review_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "candidate_count": result["candidate_count"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
