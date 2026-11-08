"""Join independently reviewed drafts and export immutable public/private content."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def planned_bind(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_new(path, obj):
    path = Path(path)
    if path.exists():
        raise RuntimeError("Refusing to overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    if (OUT / "CONTENT_ADMISSION.json").exists():
        raise RuntimeError("Admission already frozen")
    freeze = read(OUT / "EXECUTION_FREEZE.json")
    draft_path = OUT / "author/RAW_DRAFTS.jsonl"
    drafts = [json.loads(line) for line in draft_path.read_text().splitlines() if line.strip()]
    qa_path = OUT / "review/CONTENT_QA.json"
    mechanical_path = OUT / "review/MECHANICAL_SOURCE_VALIDATION_V2.json"
    family_path = OUT / "review/FAMILY_BOUNDARY_AUDIT.json"
    author_log_path = OUT / "author/AUTHOR_LOG.json"
    qa_doc = read(qa_path)
    mechanical = read(mechanical_path)
    qa_rows = qa_doc["records"]
    qa = {r["candidate_id"]: r for r in qa_rows}
    family_doc = read(family_path)
    author_log = read(author_log_path)
    family_rows = family_doc["records"]
    families = {r["candidate_id"]: r for r in family_rows}
    slots = {s["candidate_id"]: s for s in freeze["slots"]}
    roots = {r["root_id"]: r for r in freeze["roots"]}
    require(len(drafts) == len(slots) == len({d["candidate_id"] for d in drafts}), "DRAFT_SLOT_SET_MISMATCH")
    require(len(qa_rows) == len(qa) == len(slots), "DUPLICATE_OR_EXTRA_CONTENT_REVIEW_ROWS")
    require(len(family_rows) == len(families) == len(slots), "DUPLICATE_OR_EXTRA_FAMILY_REVIEW_ROWS")
    require(set(qa) == set(families) == set(slots), "REVIEW_SLOT_SET_MISMATCH")
    require({d["candidate_id"] for d in drafts} == set(slots), "DRAFT_SLOT_IDS_MISMATCH")
    # Bind the exact bytes reviewed. A row-level Gold match is not sufficient if a
    # draft changed after review.
    draft_binding = bind(draft_path)
    require(author_log.get("raw_drafts_sha256") == draft_binding["sha256"] and
            author_log.get("raw_drafts_bytes") == draft_binding["bytes"], "AUTHOR_LOG_DRAFT_BINDING_MISMATCH")
    require(Path(author_log.get("freeze_path", "")).resolve() == (OUT / "EXECUTION_FREEZE.json").resolve(), "AUTHOR_LOG_FREEZE_PATH_MISMATCH")
    require(author_log.get("solver_calls") == 0 and author_log.get("training_runs") == 0 and
            author_log.get("external_model_api_calls") == 0, "AUTHORING_SCOPE_VIOLATION")
    qa_bindings = {r["path"]: r for r in qa_doc.get("input_bindings", [])}
    required_qa_paths = [OUT / "EXECUTION_FREEZE.json", draft_path] + [Path(r["source"]["path"]) for r in freeze["roots"]]
    for p in required_qa_paths:
        require(str(p) in qa_bindings and qa_bindings[str(p)] == bind(p), "QA_INPUT_BINDING_MISMATCH: " + str(p))
    require(qa_doc.get("status") == "CONTENT_SOURCE_REVIEW_COMPLETE", "CONTENT_QA_NOT_COMPLETE")
    require(mechanical.get("status") == "PASS" and mechanical.get("candidate_count") == len(slots), "MECHANICAL_SOURCE_VALIDATION_FAILED")
    mechanical_bindings = {r["path"]: r for r in mechanical.get("input_bindings", [])}
    for p in required_qa_paths:
        require(mechanical_bindings.get(str(p)) == bind(p), "MECHANICAL_INPUT_BINDING_MISMATCH: " + str(p))
    family_bindings = {r["path"]: r for r in family_doc.get("input_bindings", [])}
    for p in [OUT / "EXECUTION_FREEZE.json", draft_path, qa_path, mechanical_path]:
        require(str(p) in family_bindings and family_bindings[str(p)] == bind(p), "FAMILY_INPUT_BINDING_MISMATCH: " + str(p))
    family_bindings_valid = True
    draft_index = {d["candidate_id"]: d for d in drafts}
    clarifications_path = OUT / "author/CLARIFICATIONS.jsonl"
    clarification_ids = set()
    if clarifications_path.exists():
        clarifications = [json.loads(line) for line in clarifications_path.read_text().splitlines() if line.strip()]
        require(len({d["candidate_id"] for d in clarifications}) == len(clarifications), "DUPLICATE_CLARIFICATION")
        require(len(clarifications) <= len(slots), "CLARIFICATION_LIMIT_EXCEEDED")
        clarification_ids = {d["candidate_id"] for d in clarifications}
        clarification_qa_path = OUT / "review/CLARIFICATION_QA.json"
        require(clarification_qa_path.exists(), "CLARIFICATION_REVIEW_REQUIRED")
        clarification_qa = read(clarification_qa_path)
        clarification_bindings = {r["path"]: r for r in clarification_qa.get("input_bindings", [])}
        require(clarification_bindings.get(str(OUT / "EXECUTION_FREEZE.json")) == bind(OUT / "EXECUTION_FREEZE.json"), "CLARIFICATION_QA_FREEZE_BINDING_MISMATCH")
        require(clarification_bindings.get(str(draft_path)) == bind(draft_path), "CLARIFICATION_QA_DRAFT_BINDING_MISMATCH")
        require(clarification_bindings.get(str(clarifications_path)) == bind(clarifications_path), "CLARIFICATION_QA_TEXT_BINDING_MISMATCH")
        clarif_qa_records = {r["candidate_id"]: r for r in clarification_qa.get("records", [])}
        require(set(clarif_qa_records) == {r["candidate_id"] for r in clarifications}, "CLARIFICATION_QA_SLOT_SET_MISMATCH")
        for row in clarifications:
            cid = row["candidate_id"]
            old = draft_index[cid]
            require(row.get("candidate_id") == cid and set(row) == set(old), "CLARIFICATION_SCHEMA_CHANGED: " + cid)
            require(all(row.get(key) == value for key, value in old.items() if key != "question"),
                    "NON_WORDING_CLARIFICATION: " + cid)
            require(row.get("question") != old.get("question"), "EMPTY_CLARIFICATION: " + cid)
            require(clarif_qa_records[cid].get("status") == "PASS" and
                    clarif_qa_records[cid].get("final_draft_sha256") == hashlib.sha256(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "CLARIFICATION_REVIEW_NOT_BOUND: " + cid)
            draft_index[cid] = row
    rows = []
    pending_files = []
    freeze_root_map = {r["root_id"]: r for r in freeze["roots"]}
    for cid, slot in slots.items():
        draft, content, family = draft_index[cid], qa[cid], families[cid]
        require(draft["root_id"] == slot["root_id"], "DRAFT_ROOT_MISMATCH: " + cid)
        require(draft["question_type"] == slot["question_type"], "DRAFT_TYPE_MISMATCH: " + cid)
        require(set(draft.get("scene_refs", [])) <= set(freeze_root_map[slot["root_id"]]["allowed_target_scenes"]),
                "OFF_SCOPE_TARGET_SCENE: " + cid)
        require(content.get("initial_draft_sha256") == draft_binding["sha256"], "QA_DRAFT_HASH_MISMATCH: " + cid)
        require(content.get("status") in ("PASS", "HOLD", "EXCLUDE"), "INVALID_CONTENT_STATUS: " + cid)
        if content.get("status") == "PASS" and cid not in clarification_ids:
            require(not content.get("issues"), "CONTENT_PASS_WITH_ISSUES: " + cid)
            field = "options" if slot["question_type"] == "multi_select" else "items"
            evidence_by_label = {e.get("label"): e for e in content.get("label_evidence", [])}
            labels = [e["label"] for e in draft.get(field, [])]
            require(set(evidence_by_label) == set(labels), "QA_EVIDENCE_LABEL_SET_MISMATCH: " + cid)
            for label in labels:
                evidence = evidence_by_label[label]
                anchors = evidence.get("source_evidence", [])
                require(anchors and all(a.get("exact_quote") and a.get("turn_index_1_based") and a.get("turn_ref") for a in anchors),
                        "QA_SOURCE_ANCHOR_MISSING: " + cid + ":" + label)
                if slot["question_type"] == "multi_select":
                    expected_verdict = "SUPPORTED" if label in content["reconstructed_gold"] else "CONTRADICTED"
                    require(evidence.get("verdict") == expected_verdict, "QA_EVIDENCE_GOLD_POLARITY_MISMATCH: " + cid + ":" + label)
                else:
                    require(isinstance(evidence.get("first_semantic_report_index_1_based"), int) and
                            evidence["first_semantic_report_index_1_based"] == anchors[0]["turn_index_1_based"],
                            "QA_ORDERING_FIRST_SEMANTIC_ANCHOR_MISMATCH: " + cid + ":" + label)
            ir = content.get("independent_reconstruction", {})
            expected_turns = author_log["source_read"]["root" + slot["root_id"].rsplit("-", 1)[1]]["complete_turns"]
            require(ir.get("complete_bound_source_turns_read") == expected_turns,
                    "QA_SOURCE_TURN_COVERAGE_MISMATCH: " + cid)
            if slot["question_type"] == "ordering":
                require(ir.get("earlier_turns_searched_for_each_ordering_item") is True and
                        ir.get("complete_source_used_for_first_mention") is True,
                        "ORDERING_FULL_HISTORY_AUDIT_MISSING: " + cid)
                ranks = {label: evidence_by_label[label].get("first_semantic_report_order_rank") for label in labels}
                require(all(isinstance(rank, int) for rank in ranks.values()), "QA_ORDERING_RANK_MISSING: " + cid)
                reconstructed_order = sorted(labels, key=lambda label: ranks[label])
                require(content.get("reconstructed_gold") == reconstructed_order,
                        "QA_ORDERING_RECONSTRUCTION_MISMATCH: " + cid)
        require(family.get("status") in ("PASS", "HOLD", "EXCLUDE"), "INVALID_FAMILY_STATUS: " + cid)
        if family.get("status") == "EXCLUDE" and not family.get("whole_task_equivalent_ids"):
            family = dict(family)
            family["status"] = "HOLD"
            family["review_integrity_issue"] = "EXCLUDE_WITHOUT_WHOLE_TASK_EQUIVALENCE"
        if family.get("whole_task_equivalent_ids") and family.get("status") == "PASS":
            family = dict(family)
            family["status"] = "EXCLUDE"
            family["review_integrity_issue"] = "WHOLE_TASK_EQUIVALENT_CANNOT_PASS_AS_NEW_FAMILY"
        families[cid] = family
        source = roots[draft["root_id"]]["source"]
        require(bind(source["path"]) == source, "SOURCE_BINDING_MISMATCH: " + cid)
        status = "CONTENT_ACCEPTED" if content["status"] == family["status"] == "PASS" else "HOLD"
        if draft.get("status") == "SHORTFALL":
            status = "SHORTFALL"
        if "EXCLUDE" in [content["status"], family["status"]]:
            status = "EXCLUDE" if family.get("whole_task_equivalent_ids") else "HOLD"
        row = {"candidate_id": cid, "root_id": slot["root_id"], "original_split": "test",
               "question_type": slot["question_type"], "global_slot_index": slot["global_slot_index"],
               "planned_design": slot["design"], "status": status,
               "content_status": content["status"], "family_status": family["status"],
               "new_family_claim_allowed": family["new_family_claim_allowed"],
               "historical_task_exposure": family.get("historical_task_exposure"),
               "whole_task_equivalent_ids": family.get("whole_task_equivalent_ids", []),
               "partial_overlap_links": family.get("partial_overlap_links", []),
               "conservative_dependency_group": slot["root_id"],
               "content_issues": content.get("issues", [])}
        if status == "CONTENT_ACCEPTED":
            # These dispositions should normally arrive as explicit HOLD/EXCLUDE
            # rows from the reviewer. Fail closed rather than aborting the entire
            # ledger if a PASS record carries contradictory evidence.
            if family.get("whole_task_equivalent_ids"):
                status = row["status"] = "EXCLUDE"
                row["status_reason"] = "WHOLE_TASK_EQUIVALENT"
            elif family.get("new_family_claim_allowed") is not True:
                status = row["status"] = "HOLD"
                row["status_reason"] = "NEW_FAMILY_NOT_ALLOWED"
            leakage = family.get("current_training_required_fact_leakage")
            if leakage not in (False, "NONE_FOUND", "NONE_FOUND_IN_BOUNDED_REVIEW"):
                status = row["status"] = "HOLD"
                row["status_reason"] = "TRAINING_LEAKAGE_NOT_CLEARED"
        if status == "CONTENT_ACCEPTED":
            field = "options" if slot["question_type"] == "multi_select" else "items"
            author_labels = [entry["label"] for entry in draft[field]]
            expected_author_labels = list("ABCDEF") if field == "options" else list("ABCDEF")[:slot["item_count"]]
            require(author_labels == expected_author_labels, "DRAFT_LABELS_MISMATCH: " + cid)
            author_gold = draft["gold"]
            require(len(author_gold) == len(set(author_gold)) and set(author_gold) <= set(author_labels), "INVALID_GOLD: " + cid)
            require(author_gold == content["reconstructed_gold"], "INDEPENDENT_GOLD_MISMATCH: " + cid)
            require(len(author_gold) == (slot["gold_count"] if field == "options" else slot["item_count"]), "GOLD_ARITY_MISMATCH: " + cid)
            protocol_labels = author_labels if field == "options" else ["I" + str(i + 1) for i in range(slot["item_count"])]
            author_to_protocol = dict(zip(author_labels, protocol_labels))
            protocol_gold = [author_to_protocol[label] for label in author_gold]
            public_items = {author_to_protocol[entry["label"]]: entry["text"] for entry in draft[field]}
            public = {"candidate_id": cid, "root_id": draft["root_id"], "question_type": draft["question_type"],
                      "question": draft["question"], field: public_items,
                      "source_path": source["path"], "source_sha256": source["sha256"]}
            public_path = OUT / "content/public" / (cid + ".json")
            private_path = OUT / "content/private" / (cid + ".json")
            pending_files.extend([(public_path, public), (private_path, {"candidate_id": cid, "gold": protocol_gold,
                                     "author_gold_labels": author_gold, "author_label_to_protocol_label": author_to_protocol,
                                     "author_record": draft, "independent_content_review": content,
                                     "independent_family_review": family})])
            private_obj = pending_files[-1][1]
            row.update(public_question=planned_bind(public_path, public), private_record=planned_bind(private_path, private_obj),
                       relation_class=draft.get("relation_class", draft.get("design", slot["design"])),
                       author_label_to_protocol_label=author_to_protocol,
                       arity=len(protocol_labels), gold_count=len(protocol_gold))
        rows.append(row)
    count = Counter(r["status"] for r in rows)
    accepted = [r for r in rows if r["status"] == "CONTENT_ACCEPTED"]
    require(all(not path.exists() for path, _ in pending_files), "ADMISSION_OUTPUT_ALREADY_EXISTS")
    for path, obj in pending_files:
        write_new(path, obj)
    result = {"schema_version": "day3_supplement_content_admission_v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "CONTENT_ADMISSION_FINAL", "records": rows, "counts": dict(count),
              "input_bindings": [bind(p) for p in [OUT / "EXECUTION_FREEZE.json", draft_path, author_log_path, qa_path, mechanical_path, family_path]],
              "accepted_count": len(accepted), "planned_count": len(slots),
              "new_solver_calls": 0, "new_training_runs": 0, "score_independent": True,
              "technical_blinding": False, "human_review": False,
              "scope": "Historically exposed TEST roots, prospectively frozen tasks. Historical fact exposure and whole-task novelty are separately adjudicated; no unseen-root claim.",
              "family_review_bindings_valid": family_bindings_valid,
              "minimum_coverage_status": "PENDING_REQUEST_LENGTH_AND_SCOPED_COVERAGE_DECISION"}
    if clarifications_path.exists():
        result["input_bindings"].append(bind(clarifications_path))
    write_new(OUT / "CONTENT_ADMISSION.json", result)
    print(json.dumps({"accepted": len(accepted), "counts": dict(count)}))


if __name__ == "__main__":
    main()
