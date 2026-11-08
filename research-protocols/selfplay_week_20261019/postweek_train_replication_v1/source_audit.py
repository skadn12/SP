"""Source-first admission audit for the fixed six-root replication batch.

This script is deliberately an offline data audit. It does not import/run a
solver, inspect model predictions, or touch DEV/TEST outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
D3 = WEEK / "day3_solver187_v1/runtime/direct_v2"
PRIVATE = WEEK / "day3_solver187_v1/runtime/direct_v2_private/PRIVATE_QUESTIONS.json"


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def anchor_state_clear(value) -> bool:
    if value is False:
        return True
    if isinstance(value, dict):
        return value.get("status") in ("NONE_FOUND", "CLEAR", "NO_AMBIGUITY")
    return False


def main() -> None:
    requests_doc = load(HERE / "REQUESTS.json")
    packet = load(HERE / "SOURCE_REVIEW_PACKET.json")
    public_rows = {q["candidate_id"]: q for q in packet["questions"]}
    req_by_candidate = {}
    for request in requests_doc["requests"]:
        req_by_candidate.setdefault(request["candidate_id"], []).append(request)
    gold_rows = {q["candidate_id"]: q for q in load(PRIVATE)["questions"]}
    review_files = sorted((HERE / "reviews").glob("source_*.json"))
    reviews = {}
    review_bindings = []
    for path in review_files:
        review_bindings.append(digest(path))
        doc = load(path)
        for row in doc["questions"]:
            candidate = row["candidate_id"]
            assert candidate not in reviews, f"DUPLICATE_REVIEW:{candidate}"
            reviews[candidate] = row

    expected_candidates = set(req_by_candidate)
    assert len(expected_candidates) == 6
    assert set(reviews) == expected_candidates, "REVIEW_CANDIDATE_SET_MISMATCH"
    mappings = load(D3 / "DISPLAY_MAPPINGS.json")["requests"]
    cases = []
    private_anchor_rows = []

    for candidate in sorted(expected_candidates):
        public = public_rows[candidate]
        review = reviews[candidate]
        source = Path(public["source_path"]).resolve()
        source_hash = sha(source)
        assert source_hash == public["source_sha256"] == review["source_sha256"], f"SOURCE_HASH:{candidate}"
        gold = gold_rows[candidate]
        assert gold["root_id"] == public["root_id"]
        assert gold["question"] == public["question"]
        assert gold["items"] == public["items"]
        assert anchor_state_clear(review["ambiguity"]), f"AMBIGUOUS_SOURCE:{candidate}"

        dialogue = load(source)
        turn_order = {turn["turn_ref"]: i for i, turn in enumerate(dialogue["turns"])}
        assert len(turn_order) == len(dialogue["turns"]), f"DUPLICATE_TURN_REF:{candidate}"
        assert set(review["items"]) == set(public["items"]) == set(gold["gold"])
        anchors = {}
        for item_id, detail in review["items"].items():
            first = detail["first_turn_ref"]
            assert first in turn_order, f"UNKNOWN_FIRST_ANCHOR:{candidate}:{item_id}"
            anchors[item_id] = turn_order[first]
            for ref in detail.get("context_turn_refs", []):
                # Context is reviewer aid and may explain a later reply to an
                # earlier request; only the cited first-report anchor defines
                # the ordering position.
                assert ref in turn_order, f"BAD_CONTEXT:{candidate}:{item_id}:{ref}"
            for near in detail.get("earlier_near_matches", []):
                ref = near["turn_ref"]
                assert ref in turn_order and turn_order[ref] < turn_order[first], f"BAD_NEAR_MATCH:{candidate}:{item_id}:{ref}"
        source_order = sorted(public["items"], key=lambda item: anchors[item])
        assert len(set(anchors.values())) == 6, f"SAME_TURN_ORDER_TIE:{candidate}"
        assert source_order == review["source_order_canonical"], f"REVIEW_ORDER_NOT_FROM_ANCHORS:{candidate}"
        assert source_order == gold["gold"], f"SOURCE_GOLD_MISMATCH:{candidate}"

        request_rows = req_by_candidate[candidate]
        assert len(request_rows) == 2 and {r["variant_id"] for r in request_rows} == {"perm_00", "perm_01"}
        mapped_orders = {}
        for request in request_rows:
            old_id = candidate + "__" + request["variant_id"] + "__direct__s00"
            original_mapping = mappings[old_id]
            assert request["display"]["items"] == original_mapping["items"]
            text_to_canonical = {text: item_id for item_id, text in public["items"].items()}
            expected_map = {display_label: text_to_canonical[text]
                            for display_label, text in request["display"]["items"].items()}
            assert request["display_to_canonical"] == expected_map, f"DISPLAY_MAP:{candidate}:{request['variant_id']}"
            displayed_gold = [label for label in request["labels"]
                              if expected_map[label] in gold["gold"]]
            displayed_gold.sort(key=lambda label: gold["gold"].index(expected_map[label]))
            mapped_orders[request["variant_id"]] = [expected_map[label] for label in displayed_gold]
            assert mapped_orders[request["variant_id"]] == gold["gold"]

        cases.append({
            "candidate_id": candidate,
            "root_id": public["root_id"],
            "source_sha256": source_hash,
            "status": "PASS",
            "ambiguity": "NONE_FOUND",
            "source_order_canonical": source_order,
            "canonical_gold": gold["gold"],
            "request_ids": sorted(r["request_id"] for r in request_rows),
            "display_orders_canonical": mapped_orders,
            "review_files": [str(p.resolve()) for p in review_files
                             if candidate in {r["candidate_id"] for r in load(p)["questions"]}],
        })
        private_anchor_rows.append({
            "candidate_id": candidate,
            "root_id": public["root_id"],
            "source_sha256": source_hash,
            "canonical_gold": gold["gold"],
            "first_report_turn_refs": {item: review["items"][item]["first_turn_ref"]
                                        for item in sorted(review["items"])},
            "reviewer_rationale": {item: review["items"][item]["reason"]
                                   for item in sorted(review["items"])},
        })

    private_packet = {
        "status": "PRIVATE_DO_NOT_SUPPLY_TO_SOLVER",
        "scope": "source_first_admission_and_later_outcome_scoring_support",
        "cases": private_anchor_rows,
        "source_review_bindings": review_bindings,
        "canonical_gold_binding": digest(PRIVATE),
    }
    out_private = HERE / "private_audit" / "SOURCE_GOLD_ADJUDICATION.json"
    out_private.parent.mkdir(parents=True, exist_ok=True)
    with out_private.open("x") as stream:
        json.dump(private_packet, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    admission = {
        "schema_version": "postweek_train_replication_source_admission_v1",
        "status": "PASS",
        "created_by": "root source-first gate plus independent blind review and independent Gold/mapping cross-check",
        "review_bindings": review_bindings,
        "source_packet_binding": digest(HERE / "SOURCE_REVIEW_PACKET.json"),
        "public_question_binding": digest(HERE / "PUBLIC_QUESTIONS.json"),
        "request_binding": digest(HERE / "REQUESTS.json"),
        "private_gold_dataset_binding": digest(PRIVATE),
        "private_adjudication_binding": digest(out_private),
        "cases": cases,
        "gate_result": "ALL_SIX_CLEAR_SOURCE_ORDER_MATCHES_GOLD_AND_BOTH_FIXED_DISPLAY_MAPPINGS",
        "solver_calls_authorized_by_this_gate": False,
    }
    out_admission = HERE / "SOURCE_ADMISSION.json"
    with out_admission.open("x") as stream:
        json.dump(admission, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "SOURCE_ADMISSION_PASS", "cases": len(cases),
                      "roots": [c["root_id"] for c in cases]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
