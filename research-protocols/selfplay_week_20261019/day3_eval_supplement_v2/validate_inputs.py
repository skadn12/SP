"""Mechanically verify frozen draft evidence anchors and scope bindings."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_new(path, obj):
    path = Path(path)
    if path.exists():
        raise RuntimeError("Refusing to overwrite: " + str(path))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    freeze_path = OUT / "EXECUTION_FREEZE.json"
    draft_path = OUT / "author/RAW_DRAFTS.jsonl"
    qa_path = OUT / "review/CONTENT_QA.json"
    freeze = json.loads(freeze_path.read_text())
    drafts = [json.loads(line) for line in draft_path.read_text().splitlines() if line.strip()]
    qa_doc = json.loads(qa_path.read_text())
    qa = {r["candidate_id"]: r for r in qa_doc["records"]}
    slot_map = {s["candidate_id"]: s for s in freeze["slots"]}
    root_map = {r["root_id"]: r for r in freeze["roots"]}
    source_map = {root: json.loads(Path(r["source"]["path"]).read_text()) for root, r in root_map.items()}
    evidence_checks = []
    errors = []
    for draft in drafts:
        cid = draft["candidate_id"]
        slot = slot_map[cid]
        root = draft["root_id"]
        if draft["question_type"] != slot["question_type"] or root != slot["root_id"]:
            errors.append({"candidate_id": cid, "error": "SLOT_BINDING_MISMATCH"})
        allowed = set(root_map[root]["allowed_target_scenes"])
        if not set(draft.get("scene_refs", [])) <= allowed:
            errors.append({"candidate_id": cid, "error": "TARGET_SCENE_OUT_OF_SCOPE"})
        source = source_map[root]
        content = qa[cid]
        field = "options" if draft["question_type"] == "multi_select" else "items"
        draft_hash = hashlib.sha256(draft_path.read_bytes()).hexdigest()
        if content.get("initial_draft_sha256") != draft_hash:
            errors.append({"candidate_id": cid, "error": "QA_DRAFT_HASH_MISMATCH"})
        for item in draft["evidence"]:
            anchors = item.get("anchors") or [item.get("first_semantic_report_anchor")]
            for anchor in anchors:
                idx = anchor["turn_index_1_based"] - 1
                if idx < 0 or idx >= len(source["turns"]):
                    errors.append({"candidate_id": cid, "label": item["label"], "error": "TURN_INDEX_OUT_OF_RANGE"})
                    continue
                turn = source["turns"][idx]
                ok = (turn["turn_ref"] == anchor["turn_ref"] and turn["speaker"] == anchor["speaker"] and
                      turn["scene_ref"] == anchor["scene_ref"] and turn["text"] == anchor["exact_quote"])
                evidence_checks.append({"candidate_id": cid, "label": item["label"], "turn_index_1_based": idx + 1,
                                        "turn_ref": anchor["turn_ref"], "match": ok})
                if not ok:
                    errors.append({"candidate_id": cid, "label": item["label"], "error": "SOURCE_ANCHOR_MISMATCH"})
        qa_rows = {e.get("label"): e for e in content.get("label_evidence", [])}
        if set(qa_rows) != {e["label"] for e in draft[field]}:
            errors.append({"candidate_id": cid, "error": "QA_LABEL_SET_MISMATCH"})
        reconstructed = content.get("reconstructed_gold", [])
        if draft.get("gold") != reconstructed:
            errors.append({"candidate_id": cid, "error": "QA_RECONSTRUCTED_GOLD_MISMATCH"})
        # Multiple-choice records reconstruct truth polarity. Ordering records
        # reconstruct the first-semantic-report position of every event; they do
        # not assign a supported/contradicted verdict to event labels.
        if draft["question_type"] == "multi_select":
            for label, qae in qa_rows.items():
                verdict = "SUPPORTED" if label in reconstructed else "CONTRADICTED"
                if qae.get("verdict") != verdict:
                    errors.append({"candidate_id": cid, "label": label, "error": "QA_POLARITY_MISMATCH"})
        else:
            ir = content.get("independent_reconstruction", {})
            if ir.get("earlier_turns_searched_for_each_ordering_item") is not True or ir.get("complete_source_used_for_first_mention") is not True:
                errors.append({"candidate_id": cid, "error": "ORDERING_FULL_HISTORY_AUDIT_MISSING"})
            if set(reconstructed) != set(qa_rows) or len(reconstructed) != len(qa_rows):
                errors.append({"candidate_id": cid, "error": "QA_ORDERING_LABEL_SET_MISMATCH"})
            first_indices = {}
            for label, qae in qa_rows.items():
                idx1 = qae.get("first_semantic_report_index_1_based")
                anchors = qae.get("source_evidence", [])
                if not isinstance(idx1, int) or not anchors:
                    errors.append({"candidate_id": cid, "label": label, "error": "QA_ORDERING_ANCHOR_MISSING"})
                    continue
                first_indices[label] = idx1
                if anchors[0].get("turn_index_1_based") != idx1:
                    errors.append({"candidate_id": cid, "label": label, "error": "QA_ORDERING_FIRST_INDEX_MISMATCH"})
            if len(first_indices) == len(qa_rows):
                expected_order = sorted(first_indices, key=first_indices.get)
                if reconstructed != expected_order:
                    errors.append({"candidate_id": cid, "error": "QA_ORDERING_SEQUENCE_MISMATCH"})
        # Bind every independent-review citation to the exact source turn too.
        for label, qae in qa_rows.items():
            for anchor in qae.get("source_evidence", []):
                idx = anchor.get("turn_index_1_based", 0) - 1
                if idx < 0 or idx >= len(source["turns"]):
                    errors.append({"candidate_id": cid, "label": label, "error": "QA_TURN_INDEX_OUT_OF_RANGE"})
                    continue
                turn = source["turns"][idx]
                ok = (turn["turn_ref"] == anchor.get("turn_ref") and turn["speaker"] == anchor.get("speaker") and
                      turn["scene_ref"] == anchor.get("scene_ref") and turn["text"] == anchor.get("exact_quote"))
                if not ok:
                    errors.append({"candidate_id": cid, "label": label, "error": "QA_SOURCE_ANCHOR_MISMATCH"})
    result = {"schema_version": "day3_source_scope_mechanical_validation_v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if not errors else "FAIL", "candidate_count": len(drafts),
              "source_anchor_count": len(evidence_checks), "verified_anchor_count": sum(c["match"] for c in evidence_checks),
              "errors": errors, "evidence_checks": evidence_checks,
              "input_bindings": [bind(p) for p in [freeze_path, draft_path, qa_path] + [Path(r["source"]["path"]) for r in freeze["roots"]]],
              "solver_calls": 0, "model_api_calls": 0, "training_runs": 0}
    write_new(OUT / "review/MECHANICAL_SOURCE_VALIDATION_V2.json", result)
    print(json.dumps({"status": result["status"], "candidate_count": len(drafts), "verified_anchor_count": result["verified_anchor_count"], "errors": len(errors)}))


if __name__ == "__main__":
    main()
