"""Fail-closed metadata gate for assigned source-first QA reviews.

Future controllers must invoke this gate before consuming an asset/public
review. It checks the review_v4 v3 record chain; it does not perform semantic
review, authenticate a filesystem caller, or prove reading/private isolation.
Existing incomplete reviews cannot be retroactively certified by this tool.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from public_anchor_v1 import validate_plan, validate_public, anchor_rows, candidates, validate_review_anchor, prefix_input_paths, validate_runtime

VERSION = "assigned_review_provenance_gate_v2"
OUTPUTS = {"asset": "ASSET_SOURCE_JUDGMENTS.json", "public": "PUBLIC_SOURCE_JUDGMENTS.json"}
ROLES = {"asset": "non_author_source_first_public_asset_reviewer",
         "public": "non_author_source_first_final_public_question_reviewer"}
SCENES = tuple(f"{route}_S{i:02d}" for route in "AB" for i in (1, 2, 3))


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps({"schema_version": VERSION, "status": "REVIEW_PROVENANCE_BLOCKED", "pass": False,
                          "errors": [{"code": "CLI_ARGUMENT_ERROR", "detail": message}]}, ensure_ascii=False))
        raise SystemExit(2)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("DUPLICATE_JSON_KEY:" + key)
        obj[key] = value
    return obj


class Audit:
    def __init__(self):
        self.errors = []
        self.verified = {}

    def need(self, condition: bool, code: str, detail: str = "") -> bool:
        if not condition:
            row = {"code": code}
            if detail:
                row["detail"] = detail
            if row not in self.errors:
                self.errors.append(row)
        return bool(condition)

    def read(self, path: Path, label: str) -> dict:
        if not self.need(path.is_file(), label + ":FILE_MISSING", str(path)):
            return {}
        try:
            obj = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
            if not self.need(isinstance(obj, dict), label + ":JSON_OBJECT_REQUIRED"):
                return {}
            self.verified[str(path.resolve())] = binding(path)
            return obj
        except (OSError, UnicodeError, ValueError) as exc:
            self.need(False, label + ":INVALID_JSON", str(exc))
            return {}

    def bound(self, value: Any, expected: Path, label: str) -> bool:
        if not self.need(isinstance(value, dict) and isinstance(value.get("path"), str),
                         label + ":BINDING_MISSING"):
            return False
        path = Path(value["path"])
        if not self.need(path.is_absolute() and path.resolve() == expected.resolve(),
                         label + ":PATH_MISMATCH", str(path)):
            return False
        if not self.need(expected.is_file(), label + ":FILE_MISSING", str(expected)):
            return False
        valid = self.need(value.get("sha256") == sha(expected), label + ":HASH_MISMATCH", str(expected))
        if valid:
            self.verified[str(expected.resolve())] = binding(expected)
        return valid


def canonical_identity(value):
    return isinstance(value, str) and re.fullmatch(r"/root(?:/[a-z0-9_]+)*", value) is not None


def timestamp(audit, value, label):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed
    except (TypeError, ValueError) as exc:
        audit.need(False, label + ":INVALID_TIMESTAMP", str(exc))
        return None


def helper_literals(audit, helper):
    """Read frozen rendering constants as literals; never execute the helper."""
    values = {}
    if not helper.is_file():
        audit.need(False, "HELPER:FILE_MISSING", str(helper))
        return values
    try:
        for node in ast.parse(helper.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"IDENTITY_LIMITATION", "READ_LIMITATION"}:
                        values[target.id] = ast.literal_eval(node.value)
    except (SyntaxError, ValueError, OSError) as exc:
        audit.need(False, "HELPER:UNSUPPORTED_RENDER_CONSTANTS", str(exc))
    audit.need(set(values) == {"IDENTITY_LIMITATION", "READ_LIMITATION"} and
               all(isinstance(x, str) for x in values.values()), "HELPER:UNSUPPORTED_RENDER_CONSTANTS")
    return values


def check_scene(batch, root, scene):
    try:
        from scene_review_provenance_v1 import check_scene as implementation
        validate_runtime(batch, __file__, {"scene_review_provenance_v1.py": implementation,
                                          "public_anchor_v1.py": validate_public})
        return implementation(batch, root, scene)
    except Exception as exc:
        audit = Audit()
        audit.need(False, "SCENE_GATE_INTERNAL_OR_INVALID_METADATA", type(exc).__name__ + ": " + str(exc))
        outcome = result(audit, Path(batch).resolve(), root, "scene")
        outcome["scene_ref"] = scene
        return outcome


def check(batch: Path | str, root: str, phase: str, scene: str | None = None) -> dict:
    if phase == "scene":
        return check_scene(batch, root, scene)
    audit = Audit()
    batch = Path(batch).resolve()
    audit.need(phase in OUTPUTS, "INVALID_PHASE")
    audit.need(isinstance(root, str) and re.fullmatch(r"cc-train-episode-\d+", root) is not None, "INVALID_ROOT")
    if audit.errors:
        return result(audit, batch, root, phase)
    try:
        validate_runtime(batch, __file__, {"public_anchor_v1.py": validate_public})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        audit.need(False, "GATE_RUNTIME:NOT_FROZEN_OR_LOADED_CODE_CHANGED", str(exc))
        return result(audit, batch, root, phase)
    short = root.rsplit("-", 1)[-1]
    unit = batch / "units" / ("root" + short)
    folder = unit / "roots" / ("root" + short)
    # Consuming either QA judgment is a pre-Gold stage. Recheck at dispatch,
    # including Gold created after a valid source-first save receipt.
    audit.need(not (folder / "PRIVATE_GOLD.json").exists(), "QA_STAGE:PRIVATE_GOLD_EXISTS_BEFORE_CONSUMPTION")
    role_path, helper = batch / "ROLE_ASSIGNMENTS.json", batch / "tools/review_v4.py"
    review_path = folder / OUTPUTS[phase]
    source_path = folder / "CUMULATIVE_SOURCE.json"
    public_path = folder / ("PUBLIC_ASSETS.json" if phase == "asset" else "PUBLIC_QUESTIONS.json")
    constraints_path = unit / "inputs/CONTINUATION_CONSTRAINTS.json"
    roles = audit.read(role_path, "ROLE_ASSIGNMENTS")
    freeze = audit.read(batch / "MASTER_FREEZE.json", "MASTER_FREEZE")
    frozen = freeze.get("files")
    audit.need(isinstance(frozen, list), "MASTER_FREEZE:FILES_REQUIRED")
    frozen = frozen if isinstance(frozen, list) else []
    anchor_helper = batch / "tools/public_anchor_v1.py"
    scene_adapter = batch / "tools/scene_controller_v120.py"
    for path, label in ((role_path, "FROZEN_ROLES"), (helper, "FROZEN_HELPER"),
                        (anchor_helper, "FROZEN_ANCHOR_HELPER"), (scene_adapter, "FROZEN_SCENE_ADAPTER")):
        matches = [x for x in frozen if isinstance(x, dict) and x.get("path") == str(path)]
        if audit.need(len(matches) == 1, label + ":UNIQUE_FROZEN_BINDING_REQUIRED"):
            audit.bound(matches[0], path, label)
    assigned = roles.get("roots")
    audit.need(isinstance(assigned, list), "ROLE_ASSIGNMENTS:ROOTS_REQUIRED")
    assigned = assigned if isinstance(assigned, list) else []
    ids = [x.get("root_id") for x in assigned if isinstance(x, dict)]
    audit.need(len(ids) == len(assigned) and len(ids) == len(set(ids)), "ROLE_ASSIGNMENTS:DUPLICATE_OR_INVALID_ROOT")
    matches = [x for x in assigned if isinstance(x, dict) and x.get("root_id") == root]
    audit.need(len(matches) == 1, "ROLE_ASSIGNMENTS:ROOT_NOT_UNIQUELY_ASSIGNED")
    role = matches[0] if len(matches) == 1 else {}
    author, reviewer = role.get("author"), role.get("reviewer")
    audit.need(canonical_identity(author) and canonical_identity(reviewer), "ROLE_ASSIGNMENTS:CANONICAL_IDENTITIES_REQUIRED")
    audit.need(author is not None and reviewer != author, "ROLE_ASSIGNMENTS:SELF_REVIEW_FORBIDDEN")
    review = audit.read(review_path, "REVIEW")
    audit.need(review.get("reviewer") == reviewer and reviewer is not None, "REVIEW:REVIEWER_NOT_FROZEN_ASSIGNEE")
    audit.need(review.get("reviewer") != author and author is not None, "REVIEW:AUTHOR_CANNOT_REVIEW")
    for key, expected in (("schema_version", "codex_assigned_source_review_v3"), ("root_id", root),
                          ("review_role", ROLES[phase]), ("backend", "CURRENT_CODEX_SESSION")):
        audit.need(review.get(key) == expected, "REVIEW:FIELD_MISMATCH:" + key)
    # Derive canonical paths from the frozen assignment, never from an
    # untrusted review's arbitrary context/receipt location.
    executor_hash = hashlib.sha256(str(reviewer).encode()).hexdigest()[:16]
    directory = unit / "evidence/reads" / f"{phase}_root_{executor_hash}"
    context_path, render_path = directory / "READ_CONTEXT.json", directory / "INPUT_RENDER.txt"
    raw_path, receipt_path = directory / "RAW_JUDGMENT.json", directory / "SAVE_RECEIPT.json"
    for field, expected in (("helper_binding", helper), ("read_context_binding", context_path),
                            ("input_render_binding", render_path), ("raw_judgment_binding", raw_path),
                            ("role_assignments_binding", role_path), ("constraints_binding", constraints_path)):
        audit.bound(review.get(field), expected, "REVIEW:" + field)
    context = audit.read(context_path, "READ_CONTEXT")
    raw = audit.read(raw_path, "RAW_JUDGMENT")
    receipt = audit.read(receipt_path, "SAVE_RECEIPT")
    source = audit.read(source_path, "SOURCE")
    public = audit.read(public_path, "PUBLIC")
    anchor_path = folder / "ORDERING_ANCHOR_PLAN.json"
    anchor_plan = audit.read(anchor_path, "ORDERING_ANCHOR_PLAN")
    constraints = audit.read(constraints_path, "CONSTRAINTS")
    rows = source.get("turns")
    audit.need(source.get("root_id") == root and isinstance(rows, list) and bool(rows), "SOURCE:ROOT_OR_TURNS_INVALID")
    rows = rows if isinstance(rows, list) else []
    refs = [x.get("turn_ref") for x in rows if isinstance(x, dict)]
    audit.need(len(refs) == len(rows) and bool(refs) and len(refs) == len(set(refs)) and
               all(isinstance(r, str) and r for r in refs), "SOURCE:TURN_REFS_INVALID")
    source_digest = sha(source_path) if source_path.is_file() else None
    public_digest = sha(public_path) if public_path.is_file() else None
    planned = {}
    try:
        validate_plan(anchor_plan, root, source, source_digest)
        planned = validate_public(anchor_plan, public, phase)
    except (ValueError, KeyError, TypeError) as exc:
        audit.need(False, "ORDERING_ANCHOR_PLAN:INVALID_PUBLIC_ANCHORS", str(exc))
    for obj, label in ((review, "REVIEW"), (context, "READ_CONTEXT"), (receipt, "SAVE_RECEIPT")):
        audit.bound(obj.get("ordering_anchor_plan_binding"), anchor_path, label + ":ordering_anchor_plan_binding")
    rendered_anchors = anchor_rows(anchor_plan) if planned else []
    audit.need(bool(rendered_anchors) and context.get("ordering_anchors_rendered") == rendered_anchors,
               "READ_CONTEXT:ANCHOR_COVERAGE_MISMATCH")
    audit.need(bool(rendered_anchors) and review.get("ordering_anchors_read") == rendered_anchors,
               "REVIEW:ANCHOR_COVERAGE_MISMATCH")
    audit.need(review.get("source_sha256") == source_digest and source_digest is not None, "REVIEW:SOURCE_HASH_MISMATCH")
    audit.need(review.get("public_sha256") == public_digest and public_digest is not None, "REVIEW:PUBLIC_HASH_MISMATCH")
    audit.need(review.get("turn_refs_read") == refs and bool(refs), "REVIEW:SOURCE_COVERAGE_MISMATCH")
    texts = constraints.get("continuation_constraints")
    audit.need(constraints.get("root_id") == root and constraints.get("decision") in {"PASS", "WARNING"}, "CONSTRAINTS:NOT_ELIGIBLE")
    audit.need(isinstance(texts, list) and bool(texts) and all(isinstance(x, str) and x.strip() for x in texts), "CONSTRAINTS:TEXTS_REQUIRED")
    for field in ("full_source_read", "all_continuation_constraints_read", "independent_of_question_author",
                  "public_anchor_plan_full_read", "source_read_before_anchor_plan_and_public_content"):
        audit.need(review.get(field) is True, "REVIEW:EXPLICIT_ATTESTATION_REQUIRED:" + field)
    audit.need(review.get("private_answers_read_before_save") is False, "REVIEW:SOURCE_FIRST_ATTESTATION_REQUIRED")
    if phase == "public":
        audit.need(review.get("final_public_questions_actually_reread") is True, "REVIEW:FINAL_PUBLIC_REREAD_REQUIRED")
    for field, expected in (("schema_version", "codex_assigned_read_context_v3"), ("root_id", root),
                            ("phase", phase), ("executor_id", reviewer), ("scene_ref", None)):
        audit.need(field in context and context[field] == expected, "READ_CONTEXT:FIELD_MISMATCH:" + field)
    for field, path in (("source_binding", source_path), ("public_binding", public_path),
                        ("role_assignments_binding", role_path), ("constraints_binding", constraints_path),
                        ("input_render_binding", render_path)):
        audit.bound(context.get(field), path, "READ_CONTEXT:" + field)
    audit.need(context.get("turn_refs_rendered") == refs and bool(refs), "READ_CONTEXT:SOURCE_COVERAGE_MISMATCH")
    audit.need(isinstance(texts, list) and context.get("constraint_count_rendered") == len(texts), "READ_CONTEXT:CONSTRAINT_COVERAGE_MISMATCH")
    audit.need(context.get("native_event_binding") is None and "native_event_binding" in context, "READ_CONTEXT:UNEXPECTED_NATIVE_INPUTS")
    selection_path = unit / "selection.json"
    selection = audit.read(selection_path, "SELECTION")
    audit.bound(context.get("selection_binding"), selection_path, "READ_CONTEXT:selection_binding")
    continuations = selection.get("a_continuations")
    audit.need(selection.get("root_id") == root and isinstance(continuations, list) and len(continuations) == 3 and
               [row.get("scene_ref") for row in continuations] == [f"R02:A:S{i:02d}" for i in (1, 2, 3)] and
               all(isinstance(row, dict) and {"scene_ref", "baseline_scene_ref", "thread", "evidence"} <= set(row)
                   for row in continuations), "SELECTION:A_PLAN_REQUIRED")
    expected_inputs = [role_path, unit / "RUN_CONFIG.json", constraints_path, source_path, helper,
                       unit / "inputs/BASELINE.json", folder / "QA_AUTHOR_RECEIPT.json", public_path,
                       anchor_path, anchor_helper, scene_adapter, batch / "MASTER_FREEZE.json", selection_path]
    try:
        expected_inputs.extend(prefix_input_paths(unit))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        audit.need(False, "PREFIX_INPUTS:INVALID_OR_PRIVATE_LINEAGE", str(exc))
    audit.bound(constraints.get("baseline_binding"), unit / "inputs/BASELINE.json", "CONSTRAINTS:BASELINE")
    baseline_refs = constraints.get("review_bindings")
    audit.need(isinstance(baseline_refs, list) and len(baseline_refs) == 2, "CONSTRAINTS:TWO_BASELINE_REVIEWS_REQUIRED")
    baseline_reviewers = []
    for index, value in enumerate(baseline_refs if isinstance(baseline_refs, list) else []):
        if not audit.need(isinstance(value, dict) and isinstance(value.get("path"), str), "BASELINE_REVIEW:BINDING_REQUIRED"):
            continue
        path = Path(value["path"])
        path = path if path.is_absolute() else unit / path
        audit.bound({**value, "path": str(path.resolve())}, path, f"BASELINE_REVIEW:{index}")
        baseline = audit.read(path, f"BASELINE_REVIEW:{index}")
        actual = value.get("reviewer", baseline.get("reviewer"))
        audit.need(actual == baseline.get("reviewer"), "BASELINE_REVIEW:IDENTITY_MISMATCH")
        baseline_reviewers.append(actual)
        expected_inputs.append(path)
    assigned_baseline = role.get("baseline_reviewers")
    audit.need(isinstance(assigned_baseline, list) and len(assigned_baseline) == 2 and
               len(set(assigned_baseline)) == 2 and set(baseline_reviewers) == set(assigned_baseline), "BASELINE_REVIEW:NOT_FROZEN_ASSIGNEES")
    author_record = audit.read(folder / "QA_AUTHOR_RECEIPT.json", "QA_AUTHOR_RECEIPT")
    audit.need(author_record.get("author") == author and author_record.get("source_sha256") == source_digest,
               "QA_AUTHOR_RECEIPT:AUTHOR_OR_SOURCE_MISMATCH")
    audit.bound(author_record.get("ordering_anchor_plan_binding"), anchor_path, "QA_AUTHOR_RECEIPT:ordering_anchor_plan_binding")
    receipt_file = folder / "QA_AUTHOR_RECEIPT.json"
    audit.need(anchor_path.is_file() and receipt_file.is_file() and
               anchor_path.stat().st_mtime_ns <= receipt_file.stat().st_mtime_ns,
               "QA_AUTHOR_RECEIPT:ANCHOR_PLAN_NOT_FROZEN_FIRST")
    expected_inputs += [folder / "scenes" / scene / "SOURCE_REVIEW.json" for scene in SCENES]
    if phase == "asset":
        audit.need(public.get("root_id") == root and public.get("source_sha256") == source_digest, "PUBLIC_ASSETS:ROOT_OR_SOURCE_MISMATCH")
        audit.need("asset_judgments_binding" in context and context["asset_judgments_binding"] is None,
                   "READ_CONTEXT:UNEXPECTED_ASSET_JUDGMENTS_BINDING")
    else:
        asset_path = folder / OUTPUTS["asset"]
        expected_inputs.append(asset_path)
        audit.bound(context.get("asset_judgments_binding"), asset_path, "READ_CONTEXT:asset_judgments_binding")
        audit.need(asset_path.is_file() and review.get("asset_judgments_sha256") == sha(asset_path), "REVIEW:ASSET_JUDGMENTS_HASH_MISMATCH")
        asset = audit.read(asset_path, "PRIOR_ASSET_REVIEW")
        audit.need(asset.get("reviewer") == reviewer and asset.get("source_sha256") == source_digest, "PRIOR_ASSET_REVIEW:REVIEWER_OR_SOURCE_MISMATCH")
        questions = public.get("questions")
        audit.need(isinstance(questions, list) and bool(questions), "PUBLIC_QUESTIONS:NONEMPTY_REQUIRED")
        for question in questions if isinstance(questions, list) else []:
            audit.need(isinstance(question, dict) and question.get("root_id") == root and
                       question.get("source_sha256") == source_digest, "PUBLIC_QUESTIONS:ROOT_OR_SOURCE_MISMATCH")
            if isinstance(question, dict) and isinstance(question.get("source_path"), str):
                path = Path(question["source_path"])
                choices = {path.resolve()} if path.is_absolute() else {
                    (base / path).resolve() for base in (unit, batch, batch.parent.parent) if (base / path).is_file()}
                audit.need(choices == {source_path.resolve()}, "PUBLIC_QUESTIONS:SOURCE_PATH_MISMATCH")
            else:
                audit.need(False, "PUBLIC_QUESTIONS:SOURCE_PATH_REQUIRED")
    actual_bindings = context.get("input_bindings")
    audit.need(isinstance(actual_bindings, list), "READ_CONTEXT:INPUT_BINDINGS_REQUIRED")
    expected_by_path = {str(p.resolve()): p for p in expected_inputs}
    actual_paths = [x.get("path") for x in actual_bindings if isinstance(x, dict)] if isinstance(actual_bindings, list) else []
    audit.need(isinstance(actual_bindings, list) and len(actual_paths) == len(actual_bindings) and
               len(actual_paths) == len(set(actual_paths)) and set(actual_paths) == set(expected_by_path),
               "READ_CONTEXT:INPUT_SET_MISMATCH")
    for value in actual_bindings if isinstance(actual_bindings, list) else []:
        if isinstance(value, dict) and value.get("path") in expected_by_path:
            audit.bound(value, expected_by_path[value["path"]], "READ_CONTEXT:INPUT")
    constants = helper_literals(audit, helper)
    if len(constants) == 2:
        for obj, label in ((review, "REVIEW"), (context, "READ_CONTEXT")):
            for field, constant in (("identity_verification_scope", "IDENTITY_LIMITATION"),
                                    ("read_attestation_scope", "READ_LIMITATION")):
                audit.need(obj.get(field) == constants[constant], label + ":LIMITATION_SCOPE_MISMATCH:" + field)
    audit.need(review.get("external_semantic_certification") is False and
               review.get("semantic_review_is_programmatic") is False, "REVIEW:SEMANTIC_CERTIFICATION_SCOPE_REQUIRED")
    if len(constants) == 2 and rows and constraints and public:
        rendered = ["REVIEW_ASSIGNMENT", json.dumps({"root_id": root, "phase": phase, "scene_ref": None,
                    "executor_id": reviewer, "assigned_roles": role}, ensure_ascii=False),
                    "ALL_CONTINUATION_CONSTRAINTS", json.dumps(constraints, ensure_ascii=False, indent=2), "FULL_CUMULATIVE_SOURCE"]
        try:
            rendered += [t["turn_ref"] + " | " + t["speaker"] + ": " + t["text"] for t in rows]
            rendered += ["PUBLIC_ORDERING_ANCHOR_PLAN_FULL", json.dumps(anchor_plan, ensure_ascii=False, indent=2)]
            rendered += ["SELECTION_FULL", json.dumps(selection, ensure_ascii=False, indent=2)]
            rendered += ["PUBLIC_CONTENT_FULL", json.dumps(public, ensure_ascii=False, indent=2),
                         "EVIDENCE_LIMITATIONS", constants["IDENTITY_LIMITATION"], constants["READ_LIMITATION"]]
            audit.need(render_path.is_file() and render_path.read_text(encoding="utf-8") == "\n".join(rendered) + "\n", "INPUT_RENDER:NOT_EXACT_CURRENT_INPUT_RENDER")
        except (KeyError, TypeError, UnicodeError):
            audit.need(False, "INPUT_RENDER:SOURCE_SCHEMA_INVALID")
    attest = raw.get("read_attestation")
    audit.need(isinstance(attest, dict), "RAW_JUDGMENT:READ_ATTESTATION_REQUIRED")
    attest = attest if isinstance(attest, dict) else {}
    for field in ("full_source_read", "all_continuation_constraints_read", "public_anchor_plan_full_read",
                  "source_read_before_anchor_plan_and_public_content"):
        audit.need(attest.get(field) is True, "RAW_JUDGMENT:EXPLICIT_ATTESTATION_REQUIRED:" + field)
    audit.need(attest.get("private_answers_read_before_save") is False, "RAW_JUDGMENT:SOURCE_FIRST_ATTESTATION_REQUIRED")
    if phase == "public":
        audit.need(attest.get("final_public_questions_actually_reread") is True, "RAW_JUDGMENT:FINAL_PUBLIC_REREAD_REQUIRED")
    audit.need(review.get("executor_read_attestation") == attest and bool(attest), "REVIEW:RAW_ATTESTATION_MISMATCH")
    fields = ("multiselect_stem_valid", "statements", "ordering_questions", "findings") if phase == "asset" else ("questions", "findings")
    for field in fields:
        audit.need(field in raw and field in review and raw[field] == review[field], "REVIEW:RAW_JUDGMENT_FIELD_MISMATCH:" + field)
    ordering_reviews = review.get("ordering_questions") if phase == "asset" else review.get("questions")
    if isinstance(ordering_reviews, dict):
        ordering_reviews = [v for cid, v in ordering_reviews.items() if cid in planned]
    expected_ordering_ids = {q.get("candidate_id") for q in public.get("ordering_questions" if phase == "asset" else "questions", [])
                             if phase == "asset" or q.get("question_type") == "ordering"}
    actual_ordering_ids = [q.get("candidate_id") for q in ordering_reviews] if isinstance(ordering_reviews, list) else []
    audit.need(len(actual_ordering_ids) == len(set(actual_ordering_ids)) and set(actual_ordering_ids) == expected_ordering_ids,
               "REVIEW:ORDERING_ANCHOR_CANDIDATE_COVERAGE")
    for question in ordering_reviews if isinstance(ordering_reviews, list) else []:
        cid = question.get("candidate_id")
        if cid not in planned:
            audit.need(False, "REVIEW:ORDERING_CANDIDATE_NOT_IN_PLAN")
            continue
        items = question.get("items")
        audit.need(isinstance(items, dict) and set(items) == set(planned[cid]), "REVIEW:ORDERING_ANCHOR_ITEM_COVERAGE")
        for label, item in items.items() if isinstance(items, dict) else []:
            try:
                validate_review_anchor(item, planned[cid][label])
            except (ValueError, KeyError, TypeError) as exc:
                audit.need(False, "REVIEW:INVALID_FROZEN_ANCHOR:" + str(cid) + ":" + str(label), str(exc))
    for field, expected in (("status", "RECORDED_EXECUTOR_JUDGMENT_NOT_EXTERNAL_CERTIFICATION"),
                            ("executor_id", reviewer), ("root_id", root), ("phase", phase),
                            ("semantic_judgment_supplied_by_executor", True),
                            ("private_file_access_proven_by_program", False), ("internal_reading_proven_by_program", False)):
        audit.need(field in receipt and type(receipt[field]) is type(expected) and receipt[field] == expected,
                   "SAVE_RECEIPT:FIELD_MISMATCH:" + field)
    for field, path in (("read_context_binding", context_path), ("raw_judgment_binding", raw_path), ("output_binding", review_path)):
        audit.bound(receipt.get(field), path, "SAVE_RECEIPT:" + field)
    audit.need(not (directory / "SAVE_FAILURE.json").exists(), "SAVE_RECEIPT:FAILURE_RECORD_PRESENT")
    started = timestamp(audit, context.get("started_at_utc"), "READ_CONTEXT:STARTED")
    finished = timestamp(audit, review.get("finished_at_utc"), "REVIEW:FINISHED")
    saved = timestamp(audit, receipt.get("created_at_utc"), "SAVE_RECEIPT:CREATED")
    audit.need(review.get("started_at_utc") == context.get("started_at_utc") and started is not None, "REVIEW:READ_INTERVAL_START_MISMATCH")
    audit.need(review.get("created_at_utc") == review.get("finished_at_utc") and finished is not None, "REVIEW:CREATED_TIME_MISMATCH")
    if started is not None and finished is not None and saved is not None:
        audit.need(started <= finished <= saved, "REVIEW:TIME_ORDER_INVALID")
    if phase == "public":
        predecessor = check(batch, root, "asset")
        for error in predecessor["errors"]:
            audit.need(False, "PRIOR_ASSET_PROVENANCE:" + error["code"], error.get("detail", ""))
        for item in predecessor["verified_file_bindings"]:
            audit.verified[item["path"]] = item
    return result(audit, batch, root, phase)


def result(audit, batch, root, phase):
    return {"schema_version": VERSION, "batch": str(batch), "root_id": root, "phase": phase,
            "status": "REVIEW_PROVENANCE_PASS" if not audit.errors else "REVIEW_PROVENANCE_BLOCKED",
            "pass": not audit.errors, "errors": audit.errors, "verified_file_bindings": list(audit.verified.values()),
            "metadata_checks_only": True, "semantic_quality_certified": False,
            "original_procedural_success_retroactively_certified": False,
            "caller_identity_cryptographically_authenticated": False, "reading_or_private_isolation_proven": False,
            "controller_integration_required": True}


def main(argv=None):
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check",))
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=(*OUTPUTS, "scene"), required=True)
    parser.add_argument("--scene")
    args = parser.parse_args(argv)
    try:
        outcome = check(args.batch, args.root, args.phase, args.scene)
    except Exception as exc:
        # Unexpected malformed metadata still fails closed and remains machine
        # readable; no caller may interpret a traceback as a passing review.
        outcome = {"schema_version": VERSION, "status": "REVIEW_PROVENANCE_BLOCKED", "pass": False,
                   "errors": [{"code": "GATE_INTERNAL_OR_INVALID_METADATA", "detail": type(exc).__name__ + ": " + str(exc)}],
                   "metadata_checks_only": True, "semantic_quality_certified": False}
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0 if outcome["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
