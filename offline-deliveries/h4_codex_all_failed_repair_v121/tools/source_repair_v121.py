"""One immutable, explicitly synthetic source-repair attempt per frozen root.

This tool renders and binds inputs, applies an author's exact text-only patch,
and records two assigned readers' judgments. It does not decide whether a
contradiction is resolved, an edit is minimal, or a source is semantically sound.
Shared-filesystem executor and reading declarations are attestations, not
cryptographic authentication or proof of cognition.

SOURCE_INPUTS.json has role_assignments_binding, optional protocol_inputs, and
roots containing root_id, original_baseline_binding, allow_source_edit, inputs.
Every root's inputs must contain the exact original binding with label baseline.
SOURCE_PROTOCOL_FREEZE.json has status FROZEN_BEFORE_V121_SOURCE_AUTHORING,
created_at_utc, and bindings covering this tool, the roles, SOURCE_INPUTS and all
declared protocol/root inputs. Freeze this helper before any formal author read.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

SOURCE_CHECKS = {
    "time", "location", "actor_identity", "plan_vs_completion",
    "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity",
}
BASELINE_CHECKS = (
    "identity_coreference", "gender_relationship_titles", "fact_state_time",
    "unsupported_reconciliation",
)
BASELINE_RANK = {"PASS": 0, "WARNING": 1, "HOLD": 2}
BASELINE_CLASSES = {
    "HARD_CONTRADICTION": "HOLD", "UNRESOLVED_IDENTITY_OR_TITLE": "HOLD",
    "UNRESOLVED_CORE_EVENT": "HOLD", "UNRESOLVED_STATE_TIME": "WARNING",
    "ATTRIBUTED_UNVERIFIED_REPORT": "WARNING", "UNDERSPECIFIED_ATTRIBUTE": "WARNING",
    "COMPATIBLE_TIME_CHANGE": "PASS", "COMPATIBLE_NONEXCLUSIVE": "PASS",
}
BASELINE_JUDGMENT_FIELDS = {
    "full_source_read", "covered_turn_refs", "checks", "findings",
    "continuation_constraints", "decision",
}
AUTHOR_FIELDS = {
    "root_id", "executor", "status", "author_decision", "patches",
    "continuation_constraints", "unresolved_issues", "failure_explanation",
    "read_attestation",
}
PATCH_FIELDS = {"turn_ref", "old_text", "new_text", "reason", "change_kind"}
REVIEW_FIELDS = {
    "root_id", "executor", "decision", "checks", "baseline_judgment",
    "source_repair_assertions", "read_attestation", "failure_explanation",
}
READY_ASSERTIONS = (
    "original_contradictions_resolved", "text_only_minimal_scope_ok",
    "no_new_person_gender_or_relationship", "continuation_constraints_adequate",
    "no_original_fact_recovery_claim",
)
AUTHOR_ATTESTATIONS = (
    "full_original_source_read", "all_declared_inputs_read",
    "no_private_qa_or_solver_read", "minimal_text_only_changes",
    "no_new_person_gender_or_relationship_added",
)
REVIEW_ATTESTATIONS = (
    "full_original_source_read", "full_candidate_source_read", "all_patches_read",
    "all_author_constraints_read", "all_declared_inputs_read",
    "independent_of_author", "no_other_current_reviewer_judgment_read",
    "no_private_qa_or_solver_read",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "DUPLICATE_JSON_KEY:" + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
         "UTC_TIMESTAMP_REQUIRED")
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(document):
    body = {key: value for key, value in document.items() if key != "complete_dialogue_sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def verify(item):
    need(isinstance(item, dict) and {"path", "sha256"} <= set(item), "INVALID_BINDING")
    path = Path(item["path"])
    need(path.is_absolute() and path.is_file(), "BOUND_FILE_MISSING:" + str(path))
    need(digest(path) == item["sha256"], "BOUND_FILE_CHANGED:" + str(path))
    return path.resolve()


def write_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def save(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def constraints(value):
    need(isinstance(value, list) and value and all(nonempty_text(x) for x in value),
         "NONEMPTY_CONTINUATION_CONSTRAINTS_REQUIRED")
    need(len(set(value)) == len(value), "DUPLICATE_CONTINUATION_CONSTRAINT")
    return value


def executor_valid(value):
    return isinstance(value, str) and re.fullmatch(r"/root(?:/[A-Za-z0-9_][A-Za-z0-9_/-]*)?", value) is not None


def source_document(path, root):
    document = read(path)
    need(isinstance(document, dict) and document.get("root_id") == root, "SOURCE_ROOT_MISMATCH")
    turns = document.get("turns")
    need(isinstance(turns, list) and turns, "FULL_SOURCE_TURNS_REQUIRED")
    refs = []
    for turn in turns:
        need(isinstance(turn, dict) and all(nonempty_text(turn.get(k)) for k in ("turn_ref", "speaker", "text")),
             "INVALID_SOURCE_TURN")
        refs.append(turn["turn_ref"])
    need(len(set(refs)) == len(refs), "DUPLICATE_SOURCE_TURN_REF")
    return document


def source_only(path):
    name = Path(path).name.upper()
    need(not any(token in name for token in ("PRIVATE", "QA_DRAFT", "QA_AUTHOR", "GOLD", "SOLVER", "RESULTS")),
         "PRIVATE_OR_SOLVER_INPUT_FORBIDDEN:" + str(path))


def load(batch, root):
    batch = Path(batch).resolve()
    need(re.fullmatch(r"cc-train-episode-\d+", root) is not None, "INVALID_ROOT")
    freeze_path = batch / "SOURCE_PROTOCOL_FREEZE.json"
    need(freeze_path.is_file(), "SOURCE_PROTOCOL_NOT_FROZEN")
    frozen = read(freeze_path)
    need(frozen.get("status") == "FROZEN_BEFORE_V121_SOURCE_AUTHORING", "SOURCE_PROTOCOL_NOT_FROZEN")
    need(timestamp(frozen["created_at_utc"]) <= datetime.now(timezone.utc), "FUTURE_PROTOCOL_FREEZE")
    rows = frozen.get("bindings")
    need(isinstance(rows, list) and rows, "FROZEN_BINDINGS_REQUIRED")
    frozen_paths = [str(verify(row)) for row in rows]
    need(len(frozen_paths) == len(set(frozen_paths)), "DUPLICATE_FROZEN_BINDING")
    required = [batch / "ROLE_ASSIGNMENTS.json", batch / "SOURCE_INPUTS.json", Path(__file__).resolve()]
    need(all(str(p.resolve()) in frozen_paths for p in required), "SOURCE_PROTOCOL_RUNTIME_NOT_FROZEN")
    need(Path(__file__).resolve() == batch / "tools/source_repair_v121.py", "FOREIGN_SOURCE_REPAIR_RUNTIME")
    roles = read(required[0]).get("roots")
    need(isinstance(roles, list) and len({r.get("root_id") for r in roles}) == len(roles), "UNIQUE_ROLE_ROOTS_REQUIRED")
    matches = [r for r in roles if r.get("root_id") == root]
    need(len(matches) == 1, "ROOT_NOT_ASSIGNED")
    role = matches[0]
    author = role.get("author")
    reviewers = role.get("baseline_reviewers")
    need(executor_valid(author) and isinstance(reviewers, list) and len(reviewers) == 2
         and len(set(reviewers)) == 2 and all(executor_valid(x) and x != author for x in reviewers),
         "DISTINCT_AUTHOR_AND_TWO_BASELINE_REVIEWERS_REQUIRED")
    manifest = read(required[1])
    need(manifest.get("role_assignments_binding") == binding(required[0]), "SOURCE_ROLES_CHANGED")
    records = manifest.get("roots")
    need(isinstance(records, list) and len({r.get("root_id") for r in records}) == len(records),
         "UNIQUE_SOURCE_INPUT_ROOTS_REQUIRED")
    matches = [r for r in records if r.get("root_id") == root]
    need(len(matches) == 1, "ROOT_OUTSIDE_SOURCE_INPUTS")
    row = matches[0]
    need(type(row.get("allow_source_edit")) is bool, "EXPLICIT_SOURCE_EDIT_POLICY_REQUIRED")
    original_path = verify(row["original_baseline_binding"])
    source_only(original_path)
    original = source_document(original_path, root)
    declared = [*manifest.get("protocol_inputs", []), *row.get("inputs", [])]
    need(declared and all(isinstance(r, dict) and nonempty_text(r.get("label")) for r in declared),
         "LABELED_FULL_INPUTS_REQUIRED")
    baseline_rows = [r for r in row["inputs"] if r["label"] == "baseline"]
    need(len(baseline_rows) == 1 and {k: baseline_rows[0][k] for k in ("path", "sha256")} == row["original_baseline_binding"],
         "ORIGINAL_BASELINE_MUST_BE_RENDERED")
    paths, content = [], []
    for item in declared:
        path = verify(item)
        source_only(path)
        need(str(path) in frozen_paths, "DECLARED_INPUT_NOT_FROZEN:" + str(path))
        paths.append(path)
        raw = path.read_text(encoding="utf-8")
        try:
            value = json.loads(raw, object_pairs_hook=pairs)
        except json.JSONDecodeError:
            value = raw
        content.append({"label": item["label"], "binding": binding(path), "content": value})
    need(len(paths) == len(set(paths)), "DUPLICATE_DECLARED_SOURCE_INPUT")
    unit = batch / "units" / ("root" + root.rsplit("-", 1)[1])
    inputs = [binding(freeze_path), *[binding(p) for p in required], *[binding(p) for p in paths]]
    rendered = json.dumps({
        "root_id": root, "source_repair_policy": row,
        "protocol_freeze": frozen, "complete_source_inputs": content,
        "limitation": "Text edits are synthetic repair, never evidence that original facts were recovered. Semantic decisions and full reading remain executor attestations.",
    }, ensure_ascii=False, indent=2) + "\n"
    return {"batch": batch, "root": root, "row": row, "role": role, "unit": unit,
            "original": original, "original_path": original_path,
            "input_bindings": inputs, "render": rendered, "frozen_at": frozen["created_at_utc"]}


def source_folder(data):
    return data["unit"] / "source_repair"


def original_refs(data):
    return [turn["turn_ref"] for turn in data["original"]["turns"]]


def author_paths(data):
    folder = source_folder(data)
    return folder / "AUTHOR_READ_CONTEXT.json", folder / "AUTHOR_FULL_INPUT.txt"


def author_read(data, executor):
    need(executor == data["role"]["author"], "ACTUAL_ASSIGNED_AUTHOR_REQUIRED")
    context_path, render_path = author_paths(data)
    folder = source_folder(data)
    need(not any((folder / n).exists() for n in ("AUTHOR_PAYLOAD.json", "AUTHOR_FAILURE.json", "AUTHOR_RECEIPT.json")),
         "SOURCE_AUTHOR_ALREADY_ATTEMPTED")
    need(not context_path.exists() and not render_path.exists(), "SOURCE_AUTHOR_READ_ALREADY_STARTED")
    started = now()
    write_bytes(render_path, data["render"].encode("utf-8"))
    save(context_path, {"schema_version": "v121_source_author_read_v1", "root_id": data["root"],
         "executor": executor, "started_at_utc": started, "input_bindings": data["input_bindings"],
         "render_binding": binding(render_path), "turn_refs_read": original_refs(data),
         "allow_source_edit": data["row"]["allow_source_edit"], "source_semantics_certified_by_program": False})
    print(data["render"], end="")


def checked_author_context(data, executor):
    need(executor == data["role"]["author"], "ACTUAL_ASSIGNED_AUTHOR_REQUIRED")
    context_path, render_path = author_paths(data)
    context = read(context_path)
    need(context.get("root_id") == data["root"] and context.get("executor") == executor,
         "SOURCE_AUTHOR_READ_IDENTITY_MISMATCH")
    need(context.get("input_bindings") == data["input_bindings"], "SOURCE_AUTHOR_INPUT_SET_CHANGED")
    need(context.get("render_binding") == binding(render_path) and render_path.read_text() == data["render"],
         "SOURCE_AUTHOR_RENDER_CHANGED_OR_INCOMPLETE")
    need(context.get("turn_refs_read") == original_refs(data)
         and context.get("allow_source_edit") is data["row"]["allow_source_edit"], "SOURCE_AUTHOR_READ_COVERAGE_CHANGED")
    need(timestamp(data["frozen_at"]) <= timestamp(context["started_at_utc"]) <= datetime.now(timezone.utc),
         "SOURCE_AUTHOR_READ_PRECEDES_FREEZE")
    return context, context_path


def validate_author(raw, data, executor):
    need(isinstance(raw, dict) and set(raw) == AUTHOR_FIELDS, "SOURCE_AUTHOR_PAYLOAD_FIELDS")
    need(raw["root_id"] == data["root"] and raw["executor"] == executor, "SOURCE_AUTHOR_PAYLOAD_IDENTITY")
    need(raw["status"] in {"UNCHANGED", "EDITED_SYNTHETIC"}, "EXPLICIT_SYNTHETIC_STATUS_REQUIRED")
    need(raw["author_decision"] in {"READY", "HOLD"}, "AUTHOR_DECISION_REQUIRED")
    att = raw["read_attestation"]
    need(isinstance(att, dict) and all(att.get(k) is True for k in AUTHOR_ATTESTATIONS), "SOURCE_AUTHOR_READ_ATTESTATION")
    need(att.get("turn_refs_read") == original_refs(data), "SOURCE_AUTHOR_FULL_TURN_COVERAGE")
    constraints(raw["continuation_constraints"])
    need(isinstance(raw["unresolved_issues"], list) and all(nonempty_text(x) for x in raw["unresolved_issues"]),
         "UNRESOLVED_ISSUES_MUST_BE_LIST")
    need(isinstance(raw["failure_explanation"], str), "AUTHOR_FAILURE_EXPLANATION_REQUIRED")
    if raw["author_decision"] == "HOLD":
        need(nonempty_text(raw["failure_explanation"]), "HOLD_NEEDS_INDEPENDENT_FAILURE_EXPLANATION")
    else:
        need(not raw["unresolved_issues"] and not raw["failure_explanation"].strip(), "READY_WITH_UNRESOLVED_SOURCE_ISSUES")
    patches = raw["patches"]
    need(isinstance(patches, list), "TEXT_PATCH_LIST_REQUIRED")
    need(data["row"]["allow_source_edit"] or not patches, "SOURCE_EDIT_FORBIDDEN_FOR_ROOT")
    need((raw["status"] == "EDITED_SYNTHETIC") == bool(patches), "SOURCE_STATUS_PATCH_MISMATCH")
    candidate = copy.deepcopy(data["original"])
    turns = {turn["turn_ref"]: turn for turn in candidate["turns"]}
    refs = []
    for patch in patches:
        need(isinstance(patch, dict) and set(patch) == PATCH_FIELDS, "ONLY_EXACT_TURN_TEXT_PATCH_ALLOWED")
        need(patch["turn_ref"] in turns and patch["turn_ref"] not in refs, "PATCH_TURN_UNKNOWN_OR_DUPLICATE")
        need(all(nonempty_text(patch[k]) for k in ("old_text", "new_text", "reason", "change_kind")), "PATCH_TEXT_AND_REASON_REQUIRED")
        turn = turns[patch["turn_ref"]]
        need(patch["old_text"] == turn["text"], "PATCH_OLD_TEXT_NOT_EXACT")
        need(patch["new_text"] != patch["old_text"], "NOOP_PATCH_FORBIDDEN")
        turn["text"] = patch["new_text"]
        refs.append(patch["turn_ref"])
    if "max_patched_turns" in data["row"]:
        need(len(patches) <= data["row"]["max_patched_turns"], "FROZEN_PATCH_TURN_BUDGET_EXCEEDED")
    need([t["turn_ref"] for t in candidate["turns"]] == original_refs(data), "SOURCE_TURN_ORDER_CHANGED")
    for old, new in zip(data["original"]["turns"], candidate["turns"]):
        need({k: v for k, v in old.items() if k != "text"} == {k: v for k, v in new.items() if k != "text"},
             "SOURCE_NON_TEXT_FIELDS_CHANGED")
    if patches:
        candidate = {
            "schema_version": "v121_synthetic_repaired_source_v1", "root_id": data["root"],
            "source_status": "EDITED_SYNTHETIC", "original_facts_recovered": False,
            "original_baseline_binding": data["row"]["original_baseline_binding"],
            "original_source_metadata": {k: v for k, v in data["original"].items() if k != "turns"},
            "turn_count": len(candidate["turns"]), "turns": candidate["turns"],
        }
        candidate["complete_dialogue_sha256"] = canonical_hash(candidate)
    return candidate


def author_save(data, executor, payload_path):
    context, context_path = checked_author_context(data, executor)
    folder = source_folder(data)
    targets = [folder / n for n in ("AUTHOR_PAYLOAD.json", "AUTHOR_FAILURE.json", "AUTHOR_RECEIPT.json",
                                   "ORIGINAL_BASELINE.json", "CANDIDATE_BASELINE.json", "PATCH_MANIFEST.json")]
    need(not any(p.exists() for p in targets), "SOURCE_FIRST_AUTHOR_ATTEMPT_ALREADY_SAVED")
    raw_path = folder / "AUTHOR_PAYLOAD.json"
    write_bytes(raw_path, Path(payload_path).read_bytes())
    try:
        raw = read(raw_path)
        candidate = validate_author(raw, data, executor)
        need(load(data["batch"], data["root"])["input_bindings"] == data["input_bindings"], "SOURCE_INPUTS_CHANGED_DURING_SAVE")
        write_bytes(folder / "ORIGINAL_BASELINE.json", data["original_path"].read_bytes())
        if raw["status"] == "UNCHANGED":
            write_bytes(folder / "CANDIDATE_BASELINE.json", data["original_path"].read_bytes())
        else:
            save(folder / "CANDIDATE_BASELINE.json", candidate)
        save(folder / "PATCH_MANIFEST.json", {"schema_version": "v121_exact_text_patch_manifest_v1",
             "root_id": data["root"], "author": executor, "status": raw["status"],
             "allow_source_edit": data["row"]["allow_source_edit"], "patches": raw["patches"],
             "original_binding": binding(folder / "ORIGINAL_BASELINE.json"),
             "candidate_binding": binding(folder / "CANDIDATE_BASELINE.json"),
             "upstream_original_binding": data["row"]["original_baseline_binding"],
             "source_is_edited_synthetic": raw["status"] == "EDITED_SYNTHETIC",
             "original_facts_recovered": False})
        finished = now()
        save(folder / "AUTHOR_RECEIPT.json", {"schema_version": "v121_source_author_receipt_v1",
             "root_id": data["root"], "executor": executor, "author": executor,
             "started_at_utc": context["started_at_utc"], "finished_at_utc": finished,
             "full_source_read": True, "turn_refs_read": original_refs(data),
             "input_bindings": data["input_bindings"], "read_context_binding": binding(context_path),
             "raw_payload_binding": binding(raw_path), "patch_manifest_binding": binding(folder / "PATCH_MANIFEST.json"),
             "candidate_binding": binding(folder / "CANDIDATE_BASELINE.json"),
             "status": raw["status"], "author_decision": raw["author_decision"],
             "attempt": 1, "semantic_repairs_after_save": 0, "source_semantics_certified_by_program": False})
    except Exception as exc:
        save(folder / "AUTHOR_FAILURE.json", {"root_id": data["root"], "executor": executor,
             "failed_at_utc": now(), "status": "FIRST_SOURCE_AUTHOR_ATTEMPT_FAILED",
             "raw_payload_binding": binding(raw_path), "error": str(exc)})
        raise
    print(json.dumps({"root_id": data["root"], "status": raw["status"], "author_decision": raw["author_decision"],
                      "patch_count": len(raw["patches"]), "candidate_binding": binding(folder / "CANDIDATE_BASELINE.json")}))


def checked_candidate(data):
    folder = source_folder(data)
    receipt = read(folder / "AUTHOR_RECEIPT.json")
    need(not (folder / "AUTHOR_FAILURE.json").exists(), "SOURCE_AUTHOR_FAILED")
    need(receipt.get("schema_version") == "v121_source_author_receipt_v1"
         and receipt.get("full_source_read") is True
         and receipt.get("root_id") == data["root"]
         and receipt.get("executor") == receipt.get("author") == data["role"]["author"],
         "SOURCE_AUTHOR_RECEIPT_IDENTITY")
    context, context_path = checked_author_context(data, data["role"]["author"])
    need(receipt.get("read_context_binding") == binding(context_path)
         and receipt.get("input_bindings") == data["input_bindings"]
         and receipt.get("turn_refs_read") == original_refs(data)
         and receipt.get("started_at_utc") == context["started_at_utc"]
         and receipt.get("attempt") == 1 and receipt.get("semantic_repairs_after_save") == 0,
         "SOURCE_AUTHOR_RECEIPT_INPUTS_CHANGED")
    for key in ("raw_payload_binding", "patch_manifest_binding", "candidate_binding"):
        verify(receipt[key])
    need(receipt["raw_payload_binding"] == binding(folder / "AUTHOR_PAYLOAD.json")
         and receipt["patch_manifest_binding"] == binding(folder / "PATCH_MANIFEST.json")
         and receipt["candidate_binding"] == binding(folder / "CANDIDATE_BASELINE.json"), "SOURCE_AUTHOR_RECEIPT_PATHS")
    raw = read(folder / "AUTHOR_PAYLOAD.json")
    expected = validate_author(raw, data, data["role"]["author"])
    need(receipt.get("status") == raw["status"] and receipt.get("author_decision") == raw["author_decision"],
         "SOURCE_AUTHOR_RECEIPT_DECISION_CHANGED")
    candidate_path = folder / "CANDIDATE_BASELINE.json"
    candidate = source_document(candidate_path, data["root"])
    need(candidate == expected, "CANDIDATE_NOT_EXACT_TEXT_PATCH_RESULT")
    need((folder / "ORIGINAL_BASELINE.json").read_bytes() == data["original_path"].read_bytes(), "ARCHIVED_ORIGINAL_CHANGED")
    if raw["status"] == "UNCHANGED":
        need(candidate_path.read_bytes() == data["original_path"].read_bytes(), "UNCHANGED_SOURCE_NOT_VERBATIM")
    else:
        need(candidate.get("complete_dialogue_sha256") == canonical_hash(candidate), "SYNTHETIC_SOURCE_CANONICAL_HASH_CHANGED")
    manifest = read(folder / "PATCH_MANIFEST.json")
    expected_manifest = {"schema_version": "v121_exact_text_patch_manifest_v1",
        "root_id": data["root"], "author": data["role"]["author"], "status": raw["status"],
        "allow_source_edit": data["row"]["allow_source_edit"], "patches": raw["patches"],
        "original_binding": binding(folder / "ORIGINAL_BASELINE.json"),
        "candidate_binding": binding(candidate_path), "upstream_original_binding": data["row"]["original_baseline_binding"],
        "source_is_edited_synthetic": raw["status"] == "EDITED_SYNTHETIC", "original_facts_recovered": False}
    need(manifest == expected_manifest, "PATCH_MANIFEST_CHANGED")
    need(timestamp(context["started_at_utc"]) <= timestamp(receipt["finished_at_utc"]) <= datetime.now(timezone.utc),
         "SOURCE_AUTHOR_CHRONOLOGY_INVALID")
    return candidate, raw, receipt


def reviewer_folder(data, executor):
    readers = data["role"]["baseline_reviewers"]
    need(executor in readers, "ACTUAL_ASSIGNED_BASELINE_REVIEWER_REQUIRED")
    return data["unit"] / "baseline_reviews" / ("reader" + str(readers.index(executor) + 1))


def review_inputs(data):
    candidate, author, receipt = checked_candidate(data)
    folder = source_folder(data)
    paths = [folder / n for n in ("AUTHOR_READ_CONTEXT.json", "AUTHOR_FULL_INPUT.txt", "AUTHOR_PAYLOAD.json",
                                  "ORIGINAL_BASELINE.json", "CANDIDATE_BASELINE.json", "PATCH_MANIFEST.json", "AUTHOR_RECEIPT.json")]
    inputs = [*data["input_bindings"], *[binding(p) for p in paths]]
    rendered = data["render"] + json.dumps({
        "complete_original_source": read(folder / "ORIGINAL_BASELINE.json"),
        "complete_candidate_source": candidate, "complete_exact_patch_manifest": read(folder / "PATCH_MANIFEST.json"),
        "complete_author_payload_and_constraints": author, "author_receipt": receipt,
        "review_requirement": "Independently judge both source checks and native baseline checks; constraints cannot excuse unresolved contradictions in the candidate. Edited sources are synthetic, not recovered original facts.",
    }, ensure_ascii=False, indent=2) + "\n"
    return candidate, author, inputs, rendered


def review_read(data, executor):
    folder = reviewer_folder(data, executor)
    candidate, author, inputs, rendered = review_inputs(data)
    need(not any((folder / n).exists() for n in ("READ_CONTEXT.json", "FULL_INPUT.txt", "RAW_JUDGMENT.json", "REVIEW.json", "REVIEW_FAILURE.json")),
         "SOURCE_REVIEW_ALREADY_STARTED_OR_ATTEMPTED")
    started = now()
    write_bytes(folder / "FULL_INPUT.txt", rendered.encode("utf-8"))
    save(folder / "READ_CONTEXT.json", {"schema_version": "v121_source_reviewer_read_v1",
         "root_id": data["root"], "reviewer": executor, "executor": executor,
         "started_at_utc": started, "input_bindings": inputs, "render_binding": binding(folder / "FULL_INPUT.txt"),
         "original_turn_refs_read": original_refs(data), "candidate_turn_refs_read": [t["turn_ref"] for t in candidate["turns"]],
         "author_constraints_read": author["continuation_constraints"], "reading_proven_by_program": False})
    print(rendered, end="")


def checked_review_context(data, executor):
    folder = reviewer_folder(data, executor)
    candidate, author, inputs, rendered = review_inputs(data)
    context = read(folder / "READ_CONTEXT.json")
    need(context.get("root_id") == data["root"] and context.get("executor") == context.get("reviewer") == executor,
         "SOURCE_REVIEW_READ_IDENTITY_MISMATCH")
    need(context.get("input_bindings") == inputs, "SOURCE_REVIEW_EXACT_INPUT_SET_CHANGED")
    need(context.get("render_binding") == binding(folder / "FULL_INPUT.txt")
         and (folder / "FULL_INPUT.txt").read_text() == rendered, "SOURCE_REVIEW_RENDER_CHANGED_OR_INCOMPLETE")
    need(context.get("original_turn_refs_read") == original_refs(data)
         and context.get("candidate_turn_refs_read") == [t["turn_ref"] for t in candidate["turns"]]
         and context.get("author_constraints_read") == author["continuation_constraints"], "SOURCE_REVIEW_READ_COVERAGE_CHANGED")
    need(timestamp(data["frozen_at"]) <= timestamp(context["started_at_utc"]) <= datetime.now(timezone.utc),
         "SOURCE_REVIEW_CHRONOLOGY_INVALID")
    author_receipt = read(source_folder(data) / "AUTHOR_RECEIPT.json")
    need(timestamp(author_receipt["finished_at_utc"]) <= timestamp(context["started_at_utc"]),
         "SOURCE_REVIEW_PREDATES_AUTHOR_COMPLETION")
    return folder, candidate, author, inputs, context


def validate_baseline(judgment, candidate):
    """Validate the original native four-check shape without inventing judgments."""
    need(isinstance(judgment, dict) and set(judgment) == BASELINE_JUDGMENT_FIELDS, "NATIVE_BASELINE_JUDGMENT_FIELDS")
    turns = {t["turn_ref"]: t for t in candidate["turns"]}
    need(judgment["full_source_read"] is True and judgment["covered_turn_refs"] == list(turns), "NATIVE_BASELINE_FULL_COVERAGE")
    need(isinstance(judgment["checks"], dict) and set(judgment["checks"]) == set(BASELINE_CHECKS), "NATIVE_FOUR_BASELINE_CHECKS_REQUIRED")
    need(isinstance(judgment["findings"], list), "NATIVE_BASELINE_FINDINGS_REQUIRED")
    statuses = {key: [] for key in BASELINE_CHECKS}
    ids = set()

    def cite(value):
        need(isinstance(value, dict) and set(value) == {"turn_ref", "speaker", "quote"}
             and value["turn_ref"] in turns, "NATIVE_BASELINE_CITATION_FIELDS")
        turn = turns[value["turn_ref"]]
        need(value["speaker"] == turn["speaker"] and nonempty_text(value["quote"])
             and value["quote"] in turn["text"], "NATIVE_BASELINE_CITATION_NOT_EXACT")

    for finding in judgment["findings"]:
        need(isinstance(finding, dict) and set(finding) == {"id", "category", "classification", "disposition", "reason", "citations", "temporal_evidence"},
             "NATIVE_BASELINE_FINDING_FIELDS")
        need(nonempty_text(finding["id"]) and finding["id"] not in ids, "NATIVE_BASELINE_FINDING_ID")
        ids.add(finding["id"])
        category, classification = finding["category"], finding["classification"]
        need(category in BASELINE_CHECKS and classification in BASELINE_CLASSES
             and finding["disposition"] == BASELINE_CLASSES[classification], "NATIVE_BASELINE_FINDING_CLASSIFICATION")
        need(nonempty_text(finding["reason"]) and isinstance(finding["citations"], list)
             and finding["citations"], "NATIVE_BASELINE_FINDING_EVIDENCE_REQUIRED")
        for item in finding["citations"]:
            cite(item)
        need(finding["disposition"] != "HOLD" or len({c["turn_ref"] for c in finding["citations"]}) >= 2,
             "NATIVE_HOLD_NEEDS_TWO_DISTINCT_TURNS")
        need(isinstance(finding["temporal_evidence"], list), "NATIVE_TEMPORAL_EVIDENCE_LIST_REQUIRED")
        for item in finding["temporal_evidence"]:
            cite(item)
        need(classification != "COMPATIBLE_TIME_CHANGE" or finding["temporal_evidence"], "NATIVE_TIME_CHANGE_NEEDS_EVIDENCE")
        statuses[category].append(finding["disposition"])
    for category, check in judgment["checks"].items():
        need(isinstance(check, dict) and set(check) == {"complete", "status", "reason"}
             and check["complete"] is True and nonempty_text(check["reason"]), "NATIVE_BASELINE_CHECK_INCOMPLETE")
        expected = max(statuses[category], key=BASELINE_RANK.get, default="PASS")
        need(check["status"] == expected, "NATIVE_BASELINE_CHECK_FINDINGS_DISAGREE:" + category)
    need(judgment["decision"] == max((c["status"] for c in judgment["checks"].values()), key=BASELINE_RANK.get),
         "NATIVE_BASELINE_DECISION_DISAGREES")
    constraints(judgment["continuation_constraints"])


def validate_review(raw, data, executor, candidate, author):
    need(isinstance(raw, dict) and set(raw) == REVIEW_FIELDS, "SOURCE_REVIEW_JUDGMENT_FIELDS")
    need(raw["root_id"] == data["root"] and raw["executor"] == executor, "SOURCE_REVIEW_JUDGMENT_IDENTITY")
    need(raw["decision"] in {"READY", "HOLD"}, "SOURCE_REVIEW_DECISION_REQUIRED")
    need(isinstance(raw["checks"], dict) and set(raw["checks"]) == SOURCE_CHECKS, "EIGHT_SOURCE_CHECKS_REQUIRED")
    for key, check in raw["checks"].items():
        need(isinstance(check, dict) and set(check) == {"status", "reason"}
             and check["status"] in {"PASS", "WARNING", "FAIL"} and nonempty_text(check["reason"]),
             "SOURCE_CHECK_INCOMPLETE:" + key)
    att = raw["read_attestation"]
    need(isinstance(att, dict) and all(att.get(k) is True for k in REVIEW_ATTESTATIONS), "SOURCE_REVIEW_READ_ATTESTATION")
    refs = [t["turn_ref"] for t in candidate["turns"]]
    need(att.get("original_turn_refs_read") == original_refs(data) and att.get("candidate_turn_refs_read") == refs
         and att.get("author_constraints_read") == author["continuation_constraints"], "SOURCE_REVIEW_FULL_INPUT_COVERAGE")
    assertions = raw["source_repair_assertions"]
    need(isinstance(assertions, dict) and set(assertions) == set(READY_ASSERTIONS)
         and all(type(assertions[k]) is bool for k in READY_ASSERTIONS), "EXPLICIT_SOURCE_REPAIR_ASSERTIONS_REQUIRED")
    need(isinstance(raw["failure_explanation"], str), "SOURCE_REVIEW_FAILURE_EXPLANATION_REQUIRED")
    validate_baseline(raw["baseline_judgment"], candidate)
    if raw["decision"] == "READY":
        need(all(check["status"] != "FAIL" for check in raw["checks"].values()), "SOURCE_READY_WITH_FAILED_CHECK")
        need(all(assertions.values()), "SOURCE_READY_WITH_FAILED_REPAIR_ASSERTION")
        need(raw["baseline_judgment"]["decision"] != "HOLD" and author["author_decision"] == "READY",
             "SOURCE_READY_WITH_BASELINE_OR_AUTHOR_HOLD")
        need(not raw["failure_explanation"].strip(), "SOURCE_READY_WITH_FAILURE_EXPLANATION")
    else:
        need(nonempty_text(raw["failure_explanation"]), "SOURCE_HOLD_NEEDS_INDEPENDENT_FAILURE_EXPLANATION")


def review_save(data, executor, judgment_path):
    folder, candidate, author, inputs, context = checked_review_context(data, executor)
    need(not any((folder / n).exists() for n in ("RAW_JUDGMENT.json", "REVIEW.json", "REVIEW_FAILURE.json", "SAVE_RECEIPT.json")),
         "SOURCE_FIRST_REVIEW_ALREADY_SAVED")
    raw_path = folder / "RAW_JUDGMENT.json"
    write_bytes(raw_path, Path(judgment_path).read_bytes())
    try:
        raw = read(raw_path)
        validate_review(raw, data, executor, candidate, author)
        need(review_inputs(load(data["batch"], data["root"]))[2] == inputs, "SOURCE_REVIEW_INPUTS_CHANGED_DURING_SAVE")
        finished = now()
        review = {**raw["baseline_judgment"], "schema_version": "codex_baseline_review_v1",
            "policy_version": "v121_source_repair", "root_id": data["root"], "reviewer": executor,
            "baseline_path": str(source_folder(data) / "CANDIDATE_BASELINE.json"),
            "baseline_sha256": digest(source_folder(data) / "CANDIDATE_BASELINE.json"),
            "started_at_utc": context["started_at_utc"], "finished_at_utc": finished,
            "input_bindings": inputs, "read_context_binding": binding(folder / "READ_CONTEXT.json"),
            "raw_judgment_binding": binding(raw_path), "source_repair_decision": raw["decision"],
            "source_repair_checks": raw["checks"], "source_repair_assertions": raw["source_repair_assertions"],
            "source_status": author["status"], "source_semantics_certified_by_program": False}
        save(folder / "REVIEW.json", review)
        save(folder / "SAVE_RECEIPT.json", {"root_id": data["root"], "actual_executor": executor,
             "saved_at_utc": finished, "read_context_binding": binding(folder / "READ_CONTEXT.json"),
             "raw_judgment_binding": binding(raw_path), "review_binding": binding(folder / "REVIEW.json"),
             "decision": raw["decision"]})
    except Exception as exc:
        save(folder / "REVIEW_FAILURE.json", {"root_id": data["root"], "actual_executor": executor,
             "failed_at_utc": now(), "raw_judgment_binding": binding(raw_path), "error": str(exc)})
        raise
    print(json.dumps({"root_id": data["root"], "reviewer": executor, "decision": raw["decision"],
                      "baseline_decision": raw["baseline_judgment"]["decision"], "review_binding": binding(folder / "REVIEW.json")}))


def checked_review(data, executor):
    folder, candidate, author, inputs, context = checked_review_context(data, executor)
    need(not (folder / "REVIEW_FAILURE.json").exists(), "SOURCE_REVIEW_FAILED")
    review, raw, receipt = read(folder / "REVIEW.json"), read(folder / "RAW_JUDGMENT.json"), read(folder / "SAVE_RECEIPT.json")
    validate_review(raw, data, executor, candidate, author)
    need(review.get("reviewer") == executor and review.get("root_id") == data["root"]
         and review.get("schema_version") == "codex_baseline_review_v1" and review.get("policy_version") == "v121_source_repair",
         "SOURCE_REVIEW_SAVED_IDENTITY")
    need({k: review[k] for k in BASELINE_JUDGMENT_FIELDS} == raw["baseline_judgment"], "SOURCE_REVIEW_DIFFERS_FROM_RAW_NATIVE_JUDGMENT")
    for key, value in (("source_repair_decision", raw["decision"]), ("source_repair_checks", raw["checks"]),
                       ("source_repair_assertions", raw["source_repair_assertions"]), ("input_bindings", inputs),
                       ("read_context_binding", binding(folder / "READ_CONTEXT.json")),
                       ("raw_judgment_binding", binding(folder / "RAW_JUDGMENT.json")),
                       ("baseline_sha256", digest(source_folder(data) / "CANDIDATE_BASELINE.json")),
                       ("baseline_path", str(source_folder(data) / "CANDIDATE_BASELINE.json")),
                       ("started_at_utc", context["started_at_utc"]), ("source_status", author["status"])):
        need(review.get(key) == value, "SOURCE_REVIEW_SAVED_FIELD_CHANGED:" + key)
    need(receipt.get("actual_executor") == executor and receipt.get("root_id") == data["root"]
         and receipt.get("review_binding") == binding(folder / "REVIEW.json")
         and receipt.get("read_context_binding") == binding(folder / "READ_CONTEXT.json")
         and receipt.get("raw_judgment_binding") == binding(folder / "RAW_JUDGMENT.json")
         and receipt.get("decision") == raw["decision"]
         and receipt.get("saved_at_utc") == review.get("finished_at_utc"), "SOURCE_REVIEW_SAVE_RECEIPT_CHANGED")
    need(timestamp(context["started_at_utc"]) <= timestamp(review["finished_at_utc"]) <= datetime.now(timezone.utc),
         "SOURCE_REVIEW_FINISH_CHRONOLOGY_INVALID")
    return review


def finalize_source(data, executor):
    need(executor == "/root", "SOURCE_FINALIZER_MUST_BE_PARENT_CONTROLLER")
    folder = source_folder(data)
    target = data["unit"] / "inputs"
    need(not any(p.exists() for p in (folder / "FINALIZE_RECEIPT.json", target / "BASELINE.json", target / "CONTINUATION_CONSTRAINTS.json")),
         "SOURCE_FINALIZATION_ALREADY_ATTEMPTED_OR_OUTPUT_EXISTS")
    candidate, author, _ = checked_candidate(data)
    reviews = [checked_review(data, reviewer) for reviewer in data["role"]["baseline_reviewers"]]
    ready = author["author_decision"] == "READY" and all(r["source_repair_decision"] == "READY" and r["decision"] != "HOLD" for r in reviews)
    review_bindings = [binding(reviewer_folder(data, r) / "REVIEW.json") for r in data["role"]["baseline_reviewers"]]
    decision = max((r["decision"] for r in reviews), key=BASELINE_RANK.get) if ready else "HOLD"
    output = {"root_id": data["root"], "decision": decision,
         "continuation_constraints": list(dict.fromkeys([*author["continuation_constraints"],
             *(c for r in reviews for c in r["continuation_constraints"])])),
         "baseline_binding": binding(folder / "CANDIDATE_BASELINE.json"), "review_bindings": review_bindings,
         "input_bindings": data["input_bindings"], "executor": executor, "created_at_utc": now(),
         "semantic_correctness_certified_by_program": False, "source_status": author["status"],
         "source_is_edited_synthetic": author["status"] == "EDITED_SYNTHETIC", "original_facts_recovered": False,
         "original_baseline_binding": data["row"]["original_baseline_binding"],
         "patch_manifest_binding": binding(folder / "PATCH_MANIFEST.json")}
    if ready:
        write_bytes(target / "BASELINE.json", (folder / "CANDIDATE_BASELINE.json").read_bytes())
        output["baseline_binding"] = binding(target / "BASELINE.json")
        save(target / "CONTINUATION_CONSTRAINTS.json", output)
    else:
        save(folder / "SOURCE_HOLD.json", output)
    save(folder / "FINALIZE_RECEIPT.json", {"root_id": data["root"], "actual_executor": executor,
         "finished_at_utc": now(), "status": "SOURCE_READY" if ready else "SOURCE_HOLD",
         "decision": decision, "source_status": author["status"], "review_bindings": review_bindings,
         "source_output_binding": binding(target / "CONTINUATION_CONSTRAINTS.json" if ready else folder / "SOURCE_HOLD.json"),
         "training_or_pool_merge": False, "source_semantics_certified_by_program": False})
    print(json.dumps({"root_id": data["root"], "status": "SOURCE_READY" if ready else "SOURCE_HOLD",
                      "decision": decision, "source_status": author["status"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["author-read", "author-save", "review-read", "review-save", "finalize-source"])
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--judgment", type=Path)
    args = parser.parse_args()
    data = load(args.batch, args.root)
    if args.action == "author-read":
        author_read(data, args.executor)
    elif args.action == "author-save":
        need(args.payload, "AUTHOR_PAYLOAD_REQUIRED")
        author_save(data, args.executor, args.payload)
    elif args.action == "review-read":
        review_read(data, args.executor)
    elif args.action == "review-save":
        need(args.judgment, "REVIEW_JUDGMENT_REQUIRED")
        review_save(data, args.executor, args.judgment)
    else:
        finalize_source(data, args.executor)


if __name__ == "__main__":
    main()
