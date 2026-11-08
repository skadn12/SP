"""Offline mechanical scoring of the committed twelve-response population.

Only run after RAW_RESPONSE_FREEZE.json exists. Semantic evidence judgments
remain separate human/Codex source audits under the prospectively frozen rule.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/B1"


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def pairs_agree(order, gold):
    positions = {x: i for i, x in enumerate(order)}
    return sum(positions[a] < positions[b]
               for i, a in enumerate(gold) for b in gold[i + 1:])


def main():
    raw_freeze = read(RUN / "RAW_RESPONSE_FREEZE.json")
    assert raw_freeze["status"] == "COMMITTED"
    for row in raw_freeze["files"]:
        assert binding(row["path"]) == row, "RAW_FILE_DRIFT"
    requests = read(HERE / "REQUESTS.json")["requests"]
    assert len(requests) == len(raw_freeze["files"]) == 12
    private = {r["candidate_id"]: r for r in read(HERE / "private_audit/SOURCE_GOLD_ADJUDICATION.json")["cases"]}
    public = {r["candidate_id"]: r for r in read(HERE / "PUBLIC_QUESTIONS.json")["questions"]}
    rows = []
    evidence_rows = []
    for request in requests:
        path = RUN / "responses" / (request["request_id"] + ".json")
        record = read(path)
        assert record["request_id"] == request["request_id"]
        canonical_gold = private[request["candidate_id"]]["canonical_gold"]
        anchors = private[request["candidate_id"]]["first_report_turn_refs"]
        errors = []
        answer = None
        valid = False
        try:
            assert record["status"] == "RETURNED", "NO_RETURNED_RESPONSE"
            assert record["metadata"]["finish_reason"] == "stop", "NON_STOP_FINISH_REASON"
            assert record["prompt_sha256"] == request["prompt_sha256"]
            assert record["schema_sha256"] == request["schema_sha256"]
            assert record["source_sha256"] == request["source_sha256"]
            ids = record["prompt_token_ids"]
            assert len(ids) == request["prompt_tokens"]
            assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == request["prompt_token_ids_sha256"], "TOKEN_IDS_DRIFT"
            answer = json.loads(record["raw_output"])
            jsonschema.validate(answer, read(request["schema_path"]))
            assert len(set(answer["final_answer"])) == len(request["labels"]), "DUPLICATE_FINAL_LABEL"
            assert set(answer["final_answer"]) == set(request["labels"]), "INCOMPLETE_FINAL_LABELS"
            for item in answer["item_evidence"].values():
                assert len(item["turn_refs"]) == len(set(item["turn_refs"])), "DUPLICATE_EVIDENCE_REF"
            valid = True
        except Exception as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
        canonical_prediction = ([request["display_to_canonical"][x] for x in answer["final_answer"]]
                                if valid else None)
        concordant = pairs_agree(canonical_prediction, canonical_gold) if valid else 0
        source = read(request["source_path"])
        by_ref = {t["turn_ref"]: t for t in source["turns"]}
        gold_position = {x: i for i, x in enumerate(canonical_gold)}
        evidence = []
        if valid:
            for display_label in request["labels"]:
                canonical = request["display_to_canonical"][display_label]
                item = answer["item_evidence"][display_label]
                refs = item["turn_refs"]
                evidence.append({
                    "display_label": display_label,
                    "canonical_label": canonical,
                    "item_text": public[request["candidate_id"]]["items"][canonical],
                    "judgment": item["judgment"],
                    "cited_turn_refs": refs,
                    "cited_turns": [by_ref[ref] for ref in refs],
                    "reviewer_first_report_turn_ref": anchors[canonical],
                    "reviewer_first_report_turn": by_ref[anchors[canonical]],
                    "exact_first_anchor_in_citations": anchors[canonical] in refs,
                    "expected_judgment": "supported",
                })
        # Anchor matching is an observable structural statistic. It never
        # replaces semantic review of the actually selected citation sets.
        anchor_hit_count = sum(e["exact_first_anchor_in_citations"] for e in evidence)
        supported_count = sum(e["judgment"] == "supported" for e in evidence)
        row = {
            "request_id": request["request_id"], "candidate_id": request["candidate_id"],
            "root_id": request["root_id"], "variant_id": request["variant_id"],
            "fact_family_id": request["fact_family_id"],
            "response_binding": binding(path), "status": record["status"],
            "mechanically_valid": valid, "validation_errors": errors,
            "display_prediction": answer["final_answer"] if valid else None,
            "canonical_prediction": canonical_prediction, "canonical_gold": canonical_gold,
            "exact_correct": bool(valid and canonical_prediction == canonical_gold),
            "pairwise_correct_pairs": concordant, "pairwise_total_pairs": 15,
            "pairwise_accuracy": concordant / 15,
            "kendall_tau": (2 * concordant / 15 - 1) if valid else None,
            "position_correct_items": sum(p == g for p, g in zip(canonical_prediction, canonical_gold)) if valid else 0,
            "all_judgments_supported": bool(valid and supported_count == 6),
            "supported_judgment_count": supported_count,
            "exact_first_anchor_hit_count": anchor_hit_count,
            "all_first_anchors_in_cited_sets": bool(valid and anchor_hit_count == 6),
            "semantic_citation_gate": "PENDING_SOURCE_AUDIT",
            "prompt_tokens": record.get("metadata", {}).get("prompt_tokens"),
            "completion_tokens": record.get("metadata", {}).get("completion_tokens"),
            "finish_reason": record.get("metadata", {}).get("finish_reason"),
        }
        rows.append(row)
        evidence_rows.append({
            "request_id": request["request_id"], "candidate_id": request["candidate_id"],
            "root_id": request["root_id"], "variant_id": request["variant_id"],
            "mechanically_valid": valid, "source_path": request["source_path"],
            "source_sha256": request["source_sha256"], "question": request["question"],
            "display_to_canonical": request["display_to_canonical"],
            "canonical_prediction": canonical_prediction, "canonical_gold": canonical_gold,
            "exact_correct": row["exact_correct"], "evidence_items": evidence,
        })
    by_root = {}
    for row in rows:
        by_root.setdefault(row["root_id"], []).append(row)
    root_rows = [{"root_id": root,
                  "two_valid": all(r["mechanically_valid"] for r in rr),
                  "same_canonical_prediction_both_mappings": rr[0]["canonical_prediction"] == rr[1]["canonical_prediction"],
                  "correct_mappings": sum(r["exact_correct"] for r in rr),
                  "both_have_all_six_first_anchor_hits": all(r["all_first_anchors_in_cited_sets"] for r in rr)}
                 for root, rr in sorted(by_root.items())]
    summary = {
        "planned_requests": 12, "roots": 6,
        "returned_requests": sum(r["status"] == "RETURNED" for r in rows),
        "valid_requests": sum(r["mechanically_valid"] for r in rows),
        "exact_correct": sum(r["exact_correct"] for r in rows),
        "exact_accuracy": sum(r["exact_correct"] for r in rows) / 12,
        "pairwise_accuracy_all_planned": sum(r["pairwise_correct_pairs"] for r in rows) / 180,
        "position_accuracy_all_planned": sum(r["position_correct_items"] for r in rows) / 72,
        "supported_judgments": sum(r["supported_judgment_count"] for r in rows),
        "first_anchor_hits": sum(r["exact_first_anchor_hit_count"] for r in rows),
        "citation_sets_planned": 72,
        "stable_canonical_order_roots": sum(r["same_canonical_prediction_both_mappings"] for r in root_rows),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        "model_wall_seconds": raw_freeze["wall_seconds"],
        "model_load_seconds": raw_freeze["model_load_seconds"],
        "automatic_retries": 0,
        "semantic_evidence_gate": "PENDING_SOURCE_AUDIT",
        "accuracy_scope": "B1 own-arm trained ordering questions, new evidence interface, two fixed mappings; descriptive only",
    }
    save(HERE / "MECHANICAL_SCORES.json", {
        "status": "SCORED_COMMITTED_RAW_POPULATION_SEMANTIC_AUDIT_PENDING",
        "raw_freeze_binding": binding(RUN / "RAW_RESPONSE_FREEZE.json"),
        "requests_binding": binding(HERE / "REQUESTS.json"),
        "score_code_binding": binding(Path(__file__)),
        "summary": summary, "requests": rows, "roots": root_rows,
    })
    save(HERE / "MODEL_CITATION_REVIEW_PACKET.json", {
        "status": "POST_RAW_FREEZE_SEMANTIC_CITATION_REVIEW",
        "rubric": read(HERE / "PROTOCOL.json")["citation_rubric"],
        "important": "Judge the model-selected citation sets in full source context. Do not replace them with reviewer anchors. All6 judgments and selected citation sets must pass; correct final order never qualifies as a wrong-order diagnosis.",
        "requests": evidence_rows,
    })
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
