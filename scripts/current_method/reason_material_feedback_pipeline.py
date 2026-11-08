"""Bind reviewed Train feedback to source reuse/extension handoffs.

This artifact controller does not call authors or models. A prepared prompt is
not a model-consumption receipt. Semantic source reviews are supplied by a
declared independent reader and checked against the exact reviewed bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

VERSION = "reason_material_feedback_pipeline_v1"
CHECKS = {
    "time", "location", "actor_identity", "plan_vs_completion",
    "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity",
}
PASS = {"PASS", "WARNING", "PASS_WITH_WARNING"}
FORBIDDEN_AUTHOR_KEYS = {
    "gold", "gold_labels", "target_gold_count", "actual_gold_count", "answer",
    "answers", "final_answer", "answer_reason", "expected_error",
    "model_reported_explanations", "mechanical_consistency", "semantic_review",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def checked(bound):
    require(isinstance(bound, dict), "BINDING_REQUIRED")
    require(Path(bound["path"]).is_absolute(), "ABSOLUTE_BOUND_PATH_REQUIRED")
    actual = binding(bound["path"])
    require(actual["sha256"] == bound["sha256"], "BOUND_FILE_CHANGED:" + bound["path"])
    if "bytes" in bound:
        require(actual["bytes"] == bound["bytes"], "BOUND_FILE_SIZE_CHANGED")
    return actual


def bound_json(bound):
    checked(bound)
    return read(bound["path"])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_text(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(value + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def no_answer_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require("gold" not in key.lower() and key.lower() not in FORBIDDEN_AUTHOR_KEYS, "ANSWER_DATA_IN_AUTHOR_INPUT:" + key)
            no_answer_keys(child)
    elif isinstance(value, list):
        for child in value:
            no_answer_keys(child)


def source_record(bound, root_id):
    source = bound_json(bound)
    require(source["root_id"] == root_id, "SOURCE_ROOT_MISMATCH")
    turns = source["turns"]
    require(isinstance(turns, list) and turns, "EMPTY_SOURCE")
    refs = [turn["turn_ref"] for turn in turns]
    require(len(refs) == len(set(refs)), "DUPLICATE_SOURCE_TURN_REF")
    require(all(isinstance(t.get("text"), str) and t["text"].strip() for t in turns), "EMPTY_SOURCE_TEXT")
    return source


def reason_entries(packet_bound):
    if packet_bound is None:
        return []
    packet = bound_json(packet_bound)
    require(packet.get("scope") == "train_only_exploratory", "TRAIN_REASON_PACKET_REQUIRED")
    require(packet.get("model_reasons_included") is True, "REASONS_REQUIRED")
    require(packet.get("formal_B_admission") is False, "EXPLORATORY_REASON_SCOPE_REQUIRED")
    require(packet.get("entries"), "EMPTY_REASON_PACKET")
    for upstream in packet.get("bindings", []):
        checked(upstream)
    seen = set()
    for entry in packet["entries"]:
        rid = entry["request_id"]
        require(rid not in seen, "DUPLICATE_REASON_REQUEST")
        seen.add(rid)
        source_record(entry["source_binding"], entry["root_id"])
        raw = bound_json(entry["raw_binding"])
        parsed = json.loads(raw["raw_output"])
        require(parsed == entry["model_reported_explanations"], "REASONS_MODIFIED_AFTER_MODEL_OUTPUT")
        require(isinstance(parsed.get("answer_reason"), str) and parsed["answer_reason"].strip(), "ANSWER_REASON_REQUIRED")
        items = parsed.get("item_evidence")
        require(isinstance(items, dict) and items, "ITEM_REASONS_REQUIRED")
        review = entry["semantic_review"]
        require(review.get("raw_binding") == entry["raw_binding"], "REASON_REVIEW_RAW_MISMATCH")
        require(review.get("review_complete") is True, "REASON_REVIEW_INCOMPLETE")
        require(review.get("reviewer_type") in {"human", "ai_agent"}, "REASON_REVIEWER_REQUIRED")
        require(set(review.get("item_reviews", {})) == set(items), "REASON_REVIEW_ITEM_COVERAGE")
        require(review.get("answer_reason_consistency") in {"pass", "fail"}, "UNRESOLVED_REASON_REVIEW")
        for label, item in items.items():
            require(isinstance(item.get("reason"), str) and item["reason"].strip(), "ITEM_REASON_REQUIRED")
            verdict = review["item_reviews"][label]
            require(all(verdict.get(key) in {"pass", "fail"} for key in ("source_support", "reason_support")), "UNRESOLVED_REASON_REVIEW")
            require(isinstance(verdict.get("notes"), str) and verdict["notes"].strip(), "REASON_REVIEW_NOTES_REQUIRED")
    return packet["entries"]


def shortfalls(ledger_bound):
    if ledger_bound is None:
        return {}
    ledger = bound_json(ledger_bound)
    rows = ledger.get("slots", ledger.get("records", ledger.get("rows", [])))
    require(isinstance(rows, list), "LEDGER_ROWS_REQUIRED")
    result = {}
    for row in rows:
        status = row.get("status", row.get("state"))
        if status not in {"SHORTFALL", "SOURCE_SHORTFALL"}:
            continue
        sid = row["slot_id"]
        require(sid not in result, "DUPLICATE_SHORTFALL_SLOT")
        review_bound = row.get("review_binding")
        require(review_bound is not None, "SHORTFALL_INDEPENDENT_REVIEW_BINDING_REQUIRED")
        review = bound_json(review_bound)
        review_rows = review.get("slot_reviews", review.get("reviews", []))
        matches = [r for r in review_rows if r.get("slot_id", r.get("candidate_id")) == sid]
        require(len(matches) == 1, "SHORTFALL_REVIEW_SLOT_MISMATCH")
        decision = matches[0].get("recommendation", matches[0].get("status"))
        require(decision in {"PASS_SHORTFALL", "SOURCE_SHORTFALL"}, "SHORTFALL_NOT_REVIEWED_AS_MATERIAL_GAP")
        require(isinstance(row.get("review_finding", row.get("reason")), str), "SHORTFALL_FINDING_REQUIRED")
        result[sid] = row
    return result


def prepare(config_path, output_dir):
    config_path, output_dir = Path(config_path), Path(output_dir)
    require(not output_dir.exists(), "OUTPUT_EXISTS_NO_OVERWRITE")
    config = read(config_path)
    require(config.get("schema_version") == VERSION + "_config", "CONFIG_VERSION")
    namespace = config["namespace"]
    require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,60}", namespace), "INVALID_NEW_NAMESPACE")
    reasons = reason_entries(config.get("reason_packet"))
    gaps = shortfalls(config.get("shortfall_ledger"))
    roots = config["roots"]
    ids = [row["root_id"] for row in roots]
    require(ids and len(ids) == len(set(ids)), "EMPTY_OR_DUPLICATE_ROOTS")
    prepared = []
    # Validate the entire batch before writing any output.
    for row in roots:
        root = row["root_id"]
        require(re.fullmatch(r"cc-train-episode-\d+", root), "TRAIN_ROOT_REQUIRED")
        source = source_record(row["source_binding"], root)
        no_answer_keys({"root_id": root, "turns": source["turns"]})
        require(not any(t["turn_ref"].startswith(namespace + ":") for t in source["turns"]), "NAMESPACE_ALREADY_IN_SOURCE")
        operations = row.get("task_operations")
        require(isinstance(operations, list) and operations, "TASK_OPERATIONS_REQUIRED")
        no_answer_keys(operations)
        local_reasons = [r for r in reasons if r["root_id"] == root]
        require(all(r["source_binding"]["sha256"] == row["source_binding"]["sha256"] for r in local_reasons), "REASON_SOURCE_VERSION_MISMATCH")
        targets = []
        selected = row.get("material_targets", [])
        require(len({t["slot_id"] for t in selected}) == len(selected), "DUPLICATE_MATERIAL_TARGET")
        for target in selected:
            gap = gaps[target["slot_id"]]
            require(gap["root_id"] == root, "MATERIAL_TARGET_ROOT_MISMATCH")
            require(gap["source_sha256"] == row["source_binding"]["sha256"], "MATERIAL_SOURCE_VERSION_MISMATCH")
            goal = target.get("material_goal")
            require(isinstance(goal, str) and goal.strip(), "EXPLICIT_MATERIAL_GOAL_REQUIRED")
            no_answer_keys(target)
            targets.append({"slot_id": target["slot_id"], "relation_target": gap.get("relation_target", gap.get("primary_relation")), "material_goal": goal})
        mode = "EXTEND" if targets else "REUSE"
        require(local_reasons or targets, "ROOT_HAS_NO_SELECTED_FEEDBACK")
        scene_refs = [f"{namespace}:{route}:S{index:02d}" for route in "AB" for index in range(1, 4)]
        events = row.get("native_events", [])
        expected_events = set(scene_refs[3:])
        require(mode == "EXTEND" or not events, "REUSE_MUST_NOT_RESERVE_EXTENSION_EVENTS")
        require(not events or {e["scene_ref"] for e in events} == expected_events, "THREE_B_SCENE_EVENT_RESERVATIONS_REQUIRED")
        require(len(events) in {0, 3}, "DUPLICATE_OR_INCOMPLETE_EVENT_RESERVATIONS")
        require(len({e["source_id"] for e in events}) == len(events), "DUPLICATE_NATIVE_EVENT")
        for event in events:
            require(":train:" in event["source_id"] and event.get("split") == "train", "NATIVE_EVENT_MUST_BE_TRAIN")
            checked(event["binding"])
            require(event.get("reserved_before_authoring") is True, "EVENT_RESERVATION_REQUIRED")
        prepared.append((row, source, local_reasons, targets, mode, scene_refs, events))
    used_event_ids = [e["source_id"] for item in prepared for e in item[-1]]
    require(len(used_event_ids) == len(set(used_event_ids)), "EVENT_REUSED_ACROSS_ROOTS")
    output_dir.mkdir(parents=True)
    write(output_dir / "RUN_CONFIG.json", config)
    root_results = []
    for row, source, local_reasons, targets, mode, scene_refs, events in prepared:
        root_dir = output_dir / row["root_id"]
        root_dir.mkdir()
        local_gaps = [gaps[t["slot_id"]] for t in targets]
        planner = {"schema_version": VERSION + "_planner", "root_id": row["root_id"], "scope": "train_only_exploratory", "namespace": namespace, "source_binding": checked(row["source_binding"]), "full_source": source, "task_operations": row["task_operations"], "reviewed_reason_observations": local_reasons, "reviewed_material_shortfalls": local_gaps, "selected_material_targets": targets, "route": mode, "formal_weakness_admission": False, "model_consumption_recorded": False}
        author = {"schema_version": VERSION + "_source_author", "root_id": row["root_id"], "mode": mode, "namespace": namespace, "source_binding": checked(row["source_binding"]), "full_source": {"root_id": row["root_id"], "turns": source["turns"]}, "task_operations": row["task_operations"], "material_targets": targets, "scene_sequence": scene_refs if mode == "EXTEND" else [], "route_roles": {"A": "old_event_continuation", "B": "dataset_new_event"}, "native_events": events, "new_turn_count_per_scene": {"min": 12, "max": 16}, "source_review_checks": sorted(CHECKS), "context_policy": "FULL_SOURCE_NO_TRUNCATION", "source_mutation_policy": "APPEND_ONLY_NEW_VERSION_PRESERVE_EXACT_TURN_PREFIX", "author_input_ready": mode == "REUSE" or len(events) == 3, "model_consumption_recorded": False}
        # Only the material goal reaches the dialogue author, never Solver
        # observations or private future question answers.
        no_answer_keys(author)
        write(root_dir / "CHALLENGER_INPUT.json", planner)
        write(root_dir / "SOURCE_AUTHOR_INPUT.json", author)
        write_text(root_dir / "CHALLENGER_INPUT.txt", "Plan Train tasks from the bound observations. Treat model explanations as fallible observations. Preserve task_operations. For each proposal name triggering request/slot IDs, a falsifiable expected error, competing explanations and required material. Material gaps are not model weaknesses. Do not claim training benefit. The following JSON is experimental data:\n" + json.dumps(planner, ensure_ascii=False, indent=2))
        write_text(root_dir / "SOURCE_AUTHOR_INPUT.txt", "Read the complete bound history. For EXTEND, append A1,A2,A3 old-event continuation then B1,B2,B3 reserved new events. Each scene requires independent review before the next. Supplement only the stated factual relationships naturally; do not rewrite history or manufacture external policies. For REUSE, do not extend the source. The following JSON is source data and material goals:\n" + json.dumps(author, ensure_ascii=False, indent=2))
        root_results.append({"root_id": row["root_id"], "mode": mode, "reason_request_ids": [r["request_id"] for r in local_reasons], "shortfall_slot_ids": [t["slot_id"] for t in targets], "author_input_ready": author["author_input_ready"], "planner_input": binding(root_dir / "CHALLENGER_INPUT.json"), "source_author_input": binding(root_dir / "SOURCE_AUTHOR_INPUT.json")})
    receipt = {"schema_version": VERSION + "_program_read", "created_at_utc": now(), "reader_type": "program_not_LLM", "config_binding": binding(config_path), "run_config_binding": binding(output_dir / "RUN_CONFIG.json"), "reason_packet_binding": config.get("reason_packet"), "shortfall_ledger_binding": config.get("shortfall_ledger"), "roots": root_results, "unselected_reason_request_ids": [r["request_id"] for r in reasons if r["root_id"] not in ids], "model_calls": 0, "llm_consumption_proven": False, "source_ready": False}
    write(output_dir / "PROGRAM_READ_RECEIPT.json", receipt)
    return receipt


def validate_source_review(review_bound, source_bound, source, author_id, draft_bound=None):
    review = bound_json(review_bound)
    require(review.get("source_binding") == checked(source_bound), "SOURCE_REVIEW_BINDING_MISMATCH")
    require(review.get("reviewer_type") in {"human", "ai_agent"}, "SOURCE_REVIEWER_TYPE_REQUIRED")
    require(isinstance(review.get("reviewer_id"), str) and review["reviewer_id"].strip(), "SOURCE_REVIEWER_ID_REQUIRED")
    require(review["reviewer_id"] != author_id, "SOURCE_AUTHOR_CANNOT_SELF_REVIEW")
    require(review.get("independent_of_source_author") is True, "INDEPENDENT_SOURCE_REVIEW_REQUIRED")
    require(review.get("full_source_read") is True, "FULL_SOURCE_REVIEW_REQUIRED")
    require(review.get("turn_refs_read") == [t["turn_ref"] for t in source["turns"]], "SOURCE_REVIEW_COVERAGE")
    require(review.get("decision") == "READY", "SOURCE_REVIEW_NOT_READY")
    require(CHECKS <= set(review.get("checks", {})), "SOURCE_REVIEW_CHECKS_INCOMPLETE")
    for name in CHECKS:
        item = review["checks"][name]
        require(item.get("status") in PASS and isinstance(item.get("reason"), str) and item["reason"].strip(), "SOURCE_REVIEW_CHECK_FAILED:" + name)
    require(not any(f.get("severity") == "BLOCKING" for f in review.get("findings", [])), "SOURCE_BLOCKING_FINDING")
    stamp = datetime.fromisoformat(review["created_at_utc"].replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "SOURCE_REVIEW_TIMESTAMP_NEEDS_TIMEZONE")
    if draft_bound:
        require(review.get("draft_binding") == checked(draft_bound), "SOURCE_REVIEW_DRAFT_BINDING_MISMATCH")
    return review, stamp


def validate_material_realization(review_bound, source_bound, source, targets):
    review = bound_json(review_bound)
    require(review.get("source_binding") == checked(source_bound), "MATERIAL_REVIEW_SOURCE_MISMATCH")
    require(review.get("reviewer_type") in {"human", "ai_agent"} and review.get("reviewer_id"), "MATERIAL_REVIEWER_REQUIRED")
    require(review.get("full_source_read") is True, "MATERIAL_REVIEW_FULL_SOURCE_REQUIRED")
    rows = review.get("targets", [])
    require({r.get("slot_id") for r in rows} == {t["slot_id"] for t in targets} and len(rows) == len(targets), "MATERIAL_REVIEW_TARGET_COVERAGE")
    turns = {t["turn_ref"]: t["text"] for t in source["turns"]}
    for row in rows:
        require(row.get("status") == "READY", "MATERIAL_GAP_STILL_UNRESOLVED")
        require(isinstance(row.get("notes"), str) and row["notes"].strip(), "MATERIAL_REVIEW_NOTES_REQUIRED")
        evidence = row.get("evidence", [])
        require(evidence, "MATERIAL_REALIZATION_EVIDENCE_REQUIRED")
        for anchor in evidence:
            require(anchor.get("turn_ref") in turns, "MATERIAL_EVIDENCE_REF_NOT_IN_SOURCE")
            require(isinstance(anchor.get("quote"), str) and anchor["quote"] and anchor["quote"] in turns[anchor["turn_ref"]], "MATERIAL_EVIDENCE_QUOTE_MISMATCH")
    return review


def import_existing_extension(spec, author, baseline_bound, baseline):
    """Import the original T4R audit chain without rewriting its old schema."""
    require(spec.get("namespace") == "T4R", "ONLY_BOUND_T4R_LEGACY_IMPORT_SUPPORTED")
    root = author["root_id"]
    freeze = bound_json(spec["source_freeze_binding"])
    require(freeze.get("schema_version") == "targeted_four_root_source_freeze_v1" and freeze.get("status") == "CONTENT_ACCEPTED_BEFORE_QUESTIONS", "LEGACY_SOURCE_FREEZE_REQUIRED")
    frozen_root = freeze["sources"][root]
    require(frozen_root.get("mode") == "targeted_extension" and frozen_root.get("accepted_scenes") == 6, "LEGACY_ROOT_NOT_SIX_SCENE_EXTENSION")
    final_bound = checked(frozen_root["source"])
    final_source = source_record(final_bound, root)
    audit_bound = checked(freeze["independent_source_review"])
    audit = bound_json(audit_bound)
    require(audit.get("source_content_gate") == "PASS" and audit.get("reviewer") and audit.get("independence"), "LEGACY_INDEPENDENT_SOURCE_QA_REQUIRED")
    root_audits = [r for r in audit["roots"] if r.get("root_id") == root]
    require(len(root_audits) == 1, "LEGACY_ROOT_AUDIT_MISSING")
    root_audit = root_audits[0]
    require(root_audit.get("status") in {"PASS", "PASS_WITH_DOCUMENTED_LIMITATIONS"}, "LEGACY_ROOT_SOURCE_NOT_ACCEPTED")
    require(checked(root_audit["prefix"])["sha256"] == baseline_bound["sha256"], "LEGACY_PREFIX_VERSION_MISMATCH")
    require(checked(root_audit["final_source"])["sha256"] == final_bound["sha256"], "LEGACY_FINAL_SOURCE_AUDIT_MISMATCH")
    reservations_bound = checked(freeze["source_reservations"])
    require(checked(audit["reservation_binding"]) == reservations_bound, "LEGACY_RESERVATION_AUDIT_MISMATCH")
    reservation_rows = bound_json(reservations_bound)["reservations"]
    events = {(r["root_id"], r["scene_ref"]): r for r in reservation_rows}
    sequence = [f"T4R:{route}:S{index:02d}" for route in "AB" for index in range(1, 4)]
    records = root_audit["scene_records"]
    require([r.get("scene_ref") for r in records] == sequence, "LEGACY_SIX_SCENE_AUDIT_SEQUENCE_REQUIRED")
    prior_bound, prior, prior_ready = baseline_bound, baseline, None
    accepted = []
    for record in records:
        scene = record["scene_ref"]
        require(record.get("independent_content_status") == "PASS", "LEGACY_SCENE_INDEPENDENT_QA_NOT_PASS")
        acceptance_bound = checked(record["acceptance"])
        acceptance = bound_json(acceptance_bound)
        require(acceptance.get("status") == "ACCEPTED" and acceptance.get("root_id") == root and acceptance.get("scene_ref") == scene, "LEGACY_SCENE_ACCEPTANCE_MISMATCH")
        source_bound, draft_bound, review_bound = checked(record["source"]), checked(record["draft"]), checked(record["parent_review"])
        require(acceptance["source"] == source_bound and acceptance["draft"] == draft_bound and acceptance["review"] == review_bound, "LEGACY_ACCEPTANCE_BINDING_MISMATCH")
        draft, review = bound_json(draft_bound), bound_json(review_bound)
        require(draft.get("root_id") == root and draft.get("scene_ref") == scene, "LEGACY_DRAFT_ROOT_SCENE_MISMATCH")
        require(checked(draft["input_source"])["sha256"] == prior_bound["sha256"], "LEGACY_PREDECESSOR_SOURCE_MISMATCH")
        require(review.get("status") == "PASS" and review.get("draft") == draft_bound and review.get("full_prefix_read") is True and review.get("reviewer"), "LEGACY_PARENT_REVIEW_REQUIRED")
        started = datetime.fromisoformat(draft["generation_metadata"]["started_at_utc"].replace("Z", "+00:00"))
        ended = datetime.fromisoformat(draft["generation_metadata"]["ended_at_utc"].replace("Z", "+00:00"))
        reviewed = datetime.fromisoformat(review["created_at_utc"].replace("Z", "+00:00"))
        ready = datetime.fromisoformat(acceptance["created_at_utc"].replace("Z", "+00:00"))
        require(all(t.tzinfo is not None for t in (started, ended, reviewed, ready)), "LEGACY_TIMESTAMPS_NEED_TIMEZONE")
        require(started <= ended <= reviewed <= ready and (prior_ready is None or started >= prior_ready), "LEGACY_PREDECESSOR_NOT_READY_AT_AUTHORING")
        turns = draft["turns"]
        require(12 <= len(turns) <= 16, "LEGACY_SCENE_TURN_COUNT")
        speakers = {t["speaker"] for t in baseline["turns"]}
        for index, turn in enumerate(turns, 1):
            require(turn.get("turn_ref") == f"{scene}:T{index:03d}" and turn.get("speaker") in speakers, "LEGACY_SCENE_REF_OR_SPEAKER_MISMATCH")
        if ":B:" in scene:
            event = events[(root, scene)]
            require(event.get("split") == "train" and event.get("status") == "PERMANENTLY_RESERVED_BEFORE_GENERATION", "LEGACY_NATIVE_EVENT_NOT_RESERVED_TRAIN")
            require(draft.get("native_event") == checked(event["native_record"]), "LEGACY_NATIVE_EVENT_SUBSTITUTION")
        else:
            require(draft.get("native_event") is None, "LEGACY_A_ROUTE_NATIVE_EVENT")
        cumulative = source_record(source_bound, root)
        require(cumulative["turns"] == prior["turns"] + turns, "LEGACY_PREFIX_OR_DRAFT_CHANGED")
        accepted.append({"scene_ref": scene, "draft_binding": draft_bound, "source_binding": source_bound, "review_binding": review_bound, "acceptance_binding": acceptance_bound, "independent_audit_binding": audit_bound})
        prior_bound, prior, prior_ready = source_bound, cumulative, ready
    require(prior_bound["sha256"] == final_bound["sha256"], "LEGACY_FINAL_NOT_ACCEPTED_B3_BYTES")
    require(final_source["turns"][:len(baseline["turns"])] == baseline["turns"], "LEGACY_ORIGINAL_PREFIX_CHANGED")
    source_author = spec.get("source_author_id")
    require(isinstance(source_author, str) and source_author.strip(), "LEGACY_SOURCE_AUTHOR_ID_REQUIRED")
    validate_source_review(spec["current_source_review_binding"], final_bound, final_source, source_author)
    validate_material_realization(spec["material_realization_review_binding"], final_bound, final_source, author["material_targets"])
    return final_bound, accepted


def accept_source(handoff_dir, submission_path):
    handoff_dir, submission_path = Path(handoff_dir), Path(submission_path)
    output = handoff_dir / "QUESTION_READY_SOURCE.json"
    require(not output.exists(), "SOURCE_ACCEPTANCE_EXISTS_NO_OVERWRITE")
    author_input_path = handoff_dir / "SOURCE_AUTHOR_INPUT.json"
    preparation = read(handoff_dir.parent / "PROGRAM_READ_RECEIPT.json")
    checked(preparation["config_binding"])
    checked(preparation["run_config_binding"])
    handoff_rows = [r for r in preparation["roots"] if r.get("root_id") == handoff_dir.name]
    require(len(handoff_rows) == 1, "ROOT_NOT_IN_PREPARATION_RECEIPT")
    require(checked(handoff_rows[0]["source_author_input"]) == binding(author_input_path), "PREPARED_AUTHOR_INPUT_CHANGED")
    author = read(author_input_path)
    submission = read(submission_path)
    root = author["root_id"]
    require(submission.get("root_id") == root and submission.get("mode") == author["mode"], "SUBMISSION_ROOT_OR_MODE_MISMATCH")
    require(submission.get("author_input_binding") == binding(author_input_path), "SUBMISSION_AUTHOR_INPUT_CHANGED")
    baseline_bound = checked(author["source_binding"])
    baseline = source_record(baseline_bound, root)
    accepted = []
    existing_import = submission.get("existing_extension_import")
    if existing_import is not None:
        require(author["mode"] == "EXTEND", "LEGACY_IMPORT_REQUIRES_MATERIAL_EXTENSION_ROUTE")
        final, accepted = import_existing_extension(existing_import, author, baseline_bound, baseline)
    elif author["mode"] == "REUSE":
        require(submission.get("source_binding") == baseline_bound, "REUSE_SOURCE_MUST_BE_UNCHANGED")
        source_author = submission.get("source_author_id")
        require(isinstance(source_author, str) and source_author.strip(), "SOURCE_AUTHOR_ID_REQUIRED")
        validate_source_review(submission["review_binding"], baseline_bound, baseline, source_author)
        final = baseline_bound
        accepted.append({"source_binding": final, "review_binding": checked(submission["review_binding"])})
    else:
        require(author.get("author_input_ready") is True, "EXTENSION_EVENTS_NOT_RESERVED")
        rows = submission.get("scenes", [])
        require([s.get("scene_ref") for s in rows] == author["scene_sequence"], "ALL_SIX_SCENES_IN_ORDER_REQUIRED")
        events = {e["scene_ref"]: e for e in author["native_events"]}
        prior_bound, prior, prior_stamp = baseline_bound, baseline, None
        for row in rows:
            scene = row["scene_ref"]
            require(scene.startswith(author["namespace"] + ":"), "SCENE_NAMESPACE_MISMATCH")
            draft_bound = checked(row["draft_binding"])
            draft = bound_json(draft_bound)
            require(draft.get("root_id") == root and draft.get("scene_ref") == scene, "DRAFT_ROOT_OR_SCENE_MISMATCH")
            require(draft.get("input_source_binding") == prior_bound, "PREDECESSOR_SOURCE_BINDING_MISMATCH")
            require(draft.get("author_input_binding") == binding(author_input_path), "DRAFT_AUTHOR_INPUT_BINDING_MISMATCH")
            writer = draft.get("author_id")
            require(isinstance(writer, str) and writer.strip(), "DRAFT_AUTHOR_ID_REQUIRED")
            created = datetime.fromisoformat(draft["created_at_utc"].replace("Z", "+00:00"))
            require(created.tzinfo is not None, "DRAFT_TIMESTAMP_NEEDS_TIMEZONE")
            require(prior_stamp is None or created >= prior_stamp, "DRAFT_CREATED_BEFORE_PREDECESSOR_READY")
            if scene in events:
                require(draft.get("native_event_binding") == checked(events[scene]["binding"]), "DRAFT_NATIVE_EVENT_MISMATCH")
                require(draft.get("native_event_source_id") == events[scene]["source_id"], "DRAFT_NATIVE_EVENT_ID_MISMATCH")
            else:
                require(draft.get("native_event_binding") is None and draft.get("native_event_source_id") is None, "OLD_EVENT_ROUTE_CANNOT_TRANSPLANT_DATASET_EVENT")
            turns = draft["turns"]
            require(12 <= len(turns) <= 16, "SCENE_TURN_COUNT_MUST_BE_12_TO_16")
            speakers = {turn["speaker"] for turn in baseline["turns"]}
            for index, turn in enumerate(turns, 1):
                require(turn.get("turn_ref") == f"{scene}:T{index:03d}", "NEW_TURN_REF_MISMATCH")
                require(turn.get("scene_ref") == scene and turn.get("session_ref") == scene, "NEW_TURN_SCENE_MISMATCH")
                require(turn.get("speaker") in speakers, "NEW_SPEAKER_NOT_ALLOWED")
                require(isinstance(turn.get("text"), str) and turn["text"].strip(), "EMPTY_NEW_TURN")
            source_bound = checked(row["source_binding"])
            cumulative = source_record(source_bound, root)
            require(cumulative["turns"] == prior["turns"] + turns, "SOURCE_MUST_BE_EXACT_PREFIX_PLUS_NEW_TURNS")
            require(cumulative["turns"][:len(baseline["turns"])] == baseline["turns"], "ORIGINAL_PREFIX_CHANGED")
            _, reviewed_at = validate_source_review(row["review_binding"], source_bound, cumulative, writer, draft_bound)
            require(reviewed_at >= created, "SOURCE_REVIEW_PRECEDES_DRAFT")
            accepted.append({"scene_ref": scene, "draft_binding": draft_bound, "source_binding": source_bound, "review_binding": checked(row["review_binding"])})
            prior_bound, prior, prior_stamp = source_bound, cumulative, reviewed_at
        final = prior_bound
    if author["mode"] == "EXTEND" and existing_import is None:
        validate_material_realization(submission["material_realization_review_binding"], final, source_record(final, root), author["material_targets"])
    result = {"schema_version": VERSION + "_question_ready", "created_at_utc": now(), "status": "SOURCE_READY_FOR_QUESTION_AUTHORING", "root_id": root, "mode": "REUSE_EXISTING_EXTENSION" if existing_import is not None else author["mode"], "namespace": existing_import["namespace"] if existing_import is not None else author["namespace"], "source_binding": final, "baseline_binding": baseline_bound, "task_operations": author["task_operations"], "selected_material_targets": author["material_targets"], "accepted_source_reviews": accepted, "existing_extension_import": existing_import, "existing_source_generated_for_current_feedback": False if existing_import is not None else None, "submission_binding": binding(submission_path), "source_author_input_binding": binding(author_input_path), "source_review_is_external_semantic_attestation": True, "reader_type": "program_not_LLM", "new_model_calls_by_controller": 0, "question_generation_started": False, "training_admitted": False}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    accept_parser = commands.add_parser("accept-source")
    accept_parser.add_argument("--handoff-dir", required=True, type=Path)
    accept_parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.config, args.output_dir) if args.command == "prepare" else accept_source(args.handoff_dir, args.submission)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"REJECTED: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
