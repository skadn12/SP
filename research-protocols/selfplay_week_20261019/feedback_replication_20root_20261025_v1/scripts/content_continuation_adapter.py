#!/usr/bin/env python3
"""Read append-only question continuations through the frozen finalizer.

Artifact-directory and actual-plan provenance lookups are adapted in memory.
The input reader also reuses the approved exact false non-access-attestation
validation copy. Actual prompts and candidate/duplicate/review/gate logic stay
unchanged. This module neither authors content nor calls models.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import importlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q  # noqa: E402

PLAN_TREES = ("question_plan_continuation", "question_plan_continuation_v3")
SHAPE_TREES = ("question_candidate_shape_continuation",)
SAVED_REVIEW_TREES = ("question_saved_revision_verification",)
TREES = (*PLAN_TREES, *SHAPE_TREES, *SAVED_REVIEW_TREES,
         "question_unfilled_continuation", "question_unfilled_slot_continuation")
READY_NAME = "CONTINUATION_FINALIZER_READY.json"
KINDS = {"mechanical_saved_plan_schema_recovery", "mechanical_saved_candidate_shape_recovery",
         "saved_revision_gold_first_verifier", "single_unused_content_revision"}
RECEIPT = BASE / "CONTENT_FINALIZER_ADAPTATION.json"


def local_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    path.relative_to(BASE.resolve())
    return path


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify(row: dict[str, Any]) -> None:
    if bind(local_path(row["path"])) != row:
        raise RuntimeError(f"CONTINUATION_BINDING_DRIFT:{row.get('path')}")


def publish(path: Path, value: Any) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"IMMUTABLE_CONTINUATION_ADAPTATION_DRIFT:{path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return bind(path)


def by_id(rows: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"CONTINUATION_NOT_LIST:{label}")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str) or row["candidate_id"] in result:
            raise RuntimeError(f"CONTINUATION_DUPLICATE_OR_INVALID_CANDIDATE:{label}")
        result[row["candidate_id"]] = row
    return result


def saved_verifier_variant(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve only independently recorded, exact engineering wrapper variants.

    These variants replay no author or Gold call.  A READY receipt binds the
    concrete wrapper while the public finalizer READY continues to identify the
    unchanged RAM schema adapter separately.  The distinction is necessary
    because the normalization log truthfully names the concrete wrapper.
    """
    variant = row.get("serialization_variant")
    # The frozen G136 wrapper cloned the generic publisher before that publisher
    # exposed serialization_variant.  Resolve that one existing omission from
    # its exact root/arm/concrete wrapper binding; the input freeze still carries
    # the explicit variant and is checked below.  No artifact is rewritten.
    if (variant is None and (row.get("root_id"), row.get("arm")) == ("cc-train-episode-136202", "G")
            and row.get("driver_binding", {}).get("path") == str(BASE / "scripts" / "g136202_saved_global_gold_verification.py")):
        variant = "g136202_global_attestation_one_r03_alias"
    variants = {
        "g136202_global_attestation_one_r03_alias": (
            "cc-train-episode-136202", "G", "G136202_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
            "g136202_saved_global_gold_verification.py"),
        "g109164_global_attestation_seven_saved_revision_Gold_ORDER2_failure_retained": (
            "cc-train-episode-109164", "G", "G109164_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
            "g109164_saved_global7_gold_verification.py"),
        "f179186_shape_final_revision_one_r03_alias_no_global_copy": (
            "cc-train-episode-179186", "F", "F179186_SAVED_SHAPE_GOLD_VERIFICATION_READY.json",
            "f179186_saved_shape_gold_verification.py"),
    }
    generic = BASE / "scripts" / "saved_revision_gold_verification.py"
    if variant is None:
        return {"driver_path": generic, "normalization_adapter_path": BASE / "scripts" / "review_schema_adapter.py",
                "approval_bindings": [], "omit_original_order2_failure": False, "serialization_variant": None}
    if variant not in variants:
        raise RuntimeError(f"CONTINUATION_SAVED_REVISION_UNAPPROVED_VARIANT:{variant}")
    root, arm, receipt_name, script_name = variants[variant]
    if (row.get("root_id"), row.get("arm")) != (root, arm):
        raise RuntimeError("CONTINUATION_SAVED_REVISION_VARIANT_IDENTITY_DRIFT")
    receipt_path = BASE / receipt_name
    approved = read(receipt_path)
    driver = BASE / "scripts" / script_name
    ram = BASE / "scripts" / "review_schema_adapter.py"
    if (not isinstance(approved, dict) or approved.get("status") != "READY_ENGINEERING_ONLY"
            or (approved.get("root_id"), approved.get("arm"), approved.get("serialization_variant")) != (root, arm, variant)
            or approved.get("wrapper_binding") != bind(driver)
            or approved.get("frozen_saved_verifier_helper_binding") != bind(generic)
            or approved.get("frozen_RAM_adapter_binding") != bind(ram)
            or approved.get("semantic_calls_started_by_engineering") != 0
            or approved.get("semantic_admission_asserted") is not False):
        raise RuntimeError(f"CONTINUATION_SAVED_REVISION_VARIANT_APPROVAL_DRIFT:{receipt_path}")
    checks = approved.get("engineering_checks_binding")
    if not isinstance(checks, dict):
        raise RuntimeError(f"CONTINUATION_SAVED_REVISION_VARIANT_CHECKS_MISSING:{receipt_path}")
    for binding in (approved["wrapper_binding"], approved["frozen_saved_verifier_helper_binding"],
                    approved["frozen_RAM_adapter_binding"], checks):
        verify(binding)
    return {"driver_path": driver, "normalization_adapter_path": driver,
            "approval_bindings": [bind(receipt_path), checks, bind(driver)],
            "omit_original_order2_failure": root == "cc-train-episode-109164", "serialization_variant": variant}


def verify_saved_final_candidates(row: dict[str, Any], path: Path,
                                  original_candidates: dict[str, dict[str, Any]],
                                  original_reviews: dict[str, dict[str, Any]],
                                  final_candidates: dict[str, dict[str, Any]],
                                  final_reviews: dict[str, dict[str, Any]],
                                  variant: dict[str, Any]) -> list[dict[str, Any]]:
    """Allow a seven-question output only for the exact original ORDER2 failure.

    The omitted object has never been a question, has no Gold, and remains in
    the immutable original archive and slot failure ledger.  No semantic
    reviewer decision is invented for it or changed by this partition.
    """
    if not variant["omit_original_order2_failure"]:
        if final_candidates != original_candidates:
            raise RuntimeError(f"CONTINUATION_SAVED_REVISION_CHANGED_FINAL_CANDIDATES:{path}")
        return []
    original = local_path(row["original_work_dir"])
    final = local_path(row["final_work_dir"])
    covered = ["MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER3"]
    failures = [candidate for candidate in original_candidates.values() if candidate.get("slot_id") == "ORDER2"]
    if len(failures) != 1:
        raise RuntimeError(f"CONTINUATION_SAVED_REVISION_ORDER2_FAILURE_MISSING:{path}")
    failure = failures[0]
    failure_id = failure["candidate_id"]
    archive = row.get("original_final_candidates_archive_binding")
    original_review_binding = row.get("uncovered_order2_original_review_binding")
    expected_candidates = {cid: candidate for cid, candidate in original_candidates.items() if cid != failure_id}
    if (row.get("covered_slot_ids") != covered or row.get("uncovered_slot_ids") != ["ORDER2"]
            or row.get("uncovered_ORDER2_original_failure_notice_and_review_preserved") is not True
            or len(original_candidates) != 8 or len(final_candidates) != 7
            or {candidate.get("slot_id") for candidate in final_candidates.values()} != set(covered)
            or final_candidates != expected_candidates or set(final_reviews) != set(final_candidates)
            or failure.get("candidate_status") != "SLOT_FAILURE"
            or any(field in failure for field in ("question", "items", "options", "gold"))
            or original_reviews.get(failure_id, {}).get("decision") != "HOLD"
            or not isinstance(archive, dict)
            or local_path(archive.get("path", "")) != final / "ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json"
            or original_review_binding != bind(original / "CONTENT_REVIEW_FINAL.json")):
        raise RuntimeError(f"CONTINUATION_SAVED_REVISION_ORDER2_PARTITION_DRIFT:{path}")
    verify(archive)
    if local_path(archive["path"]).read_bytes() != (original / "CANDIDATES_FINAL.json").read_bytes():
        raise RuntimeError(f"CONTINUATION_SAVED_REVISION_ORDER2_ARCHIVE_BYTES_DRIFT:{path}")
    failures_path = final / "SLOT_FAILURES.json"
    old_failures = read(original / "SLOT_FAILURES.json")
    new_failures = read(failures_path)
    original_order2_failures = [failure for failure in old_failures if failure.get("slot_id") == "ORDER2"]
    if not original_order2_failures or any(failure not in new_failures for failure in original_order2_failures):
        raise RuntimeError(f"CONTINUATION_SAVED_REVISION_ORDER2_FAILURE_LEDGER_DRIFT:{path}")
    return [archive, original_review_binding, bind(original / "SLOT_FAILURES.json")]


class Continuations:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.selected: dict[tuple[str, str], dict[str, Any]] = {}
        self.inputs: list[dict[str, Any]] = []
        self.tree_files: list[Path] = []
        for tree in TREES:
            base = BASE / tree
            if not base.is_dir():
                continue
            self.tree_files.extend(path for path in sorted(base.rglob("*"))
                                   if path.is_file() and "__pycache__" not in path.parts
                                   and not path.name.startswith(".") and path.suffix not in {".lock", ".pyc"})
            for path in sorted(base.rglob(READY_NAME)):
                record = self.validate_ready(path)
                self.records.append(record)
        self.select_heads()

    def validate_ready(self, path: Path) -> dict[str, Any]:
        row = read(path)
        if not isinstance(row, dict) or row.get("schema_version") != "fgr20_question_continuation_finalizer_ready_v1":
            raise RuntimeError(f"CONTINUATION_READY_SCHEMA:{path}")
        root, arm = row.get("root_id"), row.get("arm")
        if not isinstance(root, str) or root != q.root_id(root) or arm not in q.ARMS or row.get("continuation_kind") not in KINDS:
            raise RuntimeError(f"CONTINUATION_READY_IDENTITY:{path}")
        main_original = BASE / "question_authoring" / arm / "roots" / q.root_number(root)
        original = local_path(row["original_work_dir"])
        final = local_path(row["final_work_dir"])
        if row["continuation_kind"] == "mechanical_saved_plan_schema_recovery" and original != main_original.resolve():
            raise RuntimeError(f"CONTINUATION_PLAN_ORIGINAL_DIRECTORY_DRIFT:{path}")
        if original != main_original.resolve():
            if local_path(row.get("main_original_work_dir", "")) != main_original.resolve():
                raise RuntimeError(f"CONTINUATION_MAIN_ORIGINAL_DIRECTORY_DRIFT:{path}")
            parent_ready = row.get("supersedes_ready_binding")
            if not isinstance(parent_ready, dict) or local_path(parent_ready["path"]).parent != original:
                raise RuntimeError(f"CONTINUATION_INPUT_READY_LINEAGE_MISSING:{path}")
            verify(parent_ready)
        if final == original.resolve():
            raise RuntimeError(f"CONTINUATION_ORIGINAL_DIRECTORY_DRIFT:{path}")
        if final != path.parent.resolve():
            raise RuntimeError(f"CONTINUATION_READY_NOT_IN_FINAL_DIRECTORY:{path}")
        saved_variant = None
        if row["continuation_kind"] == "saved_revision_gold_first_verifier":
            saved_variant = saved_verifier_variant(row)
            self.inputs.extend(saved_variant["approval_bindings"])
            if (row.get("review_stage") != "revision_01"
                    or local_path(row.get("initial_work_dir", "")) != original):
                raise RuntimeError(f"CONTINUATION_SAVED_REVISION_REVIEW_STAGE_DRIFT:{path}")
            for flag in ("no_new_gold_call", "no_new_plan_or_author_call", "no_second_revision",
                         "initial_candidate_and_reviews_not_overwritten", "final_revision_retained_without_initial_fallback"):
                if row.get(flag) is not True:
                    raise RuntimeError(f"CONTINUATION_SAVED_REVISION_GUARD_MISSING:{flag}:{path}")
            saved_paths = {
                "input_freeze_binding": final / "RECOVERY_INPUT_FREEZE.json",
                "saved_gold_receipt_reference_binding": final / "SAVED_REVISION_GOLD_REFERENCE.json",
                "normalization_binding": final / "MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json",
                "gate_validation_binding": final / "FINAL_REVIEW_GATE.json",
                "driver_binding": saved_variant["driver_path"],
                "adapter_binding": BASE / "scripts" / "review_schema_adapter.py",
            }
            for key, expected_path in saved_paths.items():
                value = row.get(key)
                if not isinstance(value, dict) or local_path(value["path"]) != expected_path:
                    raise RuntimeError(f"CONTINUATION_SAVED_REVISION_BINDING_MISSING:{key}:{path}")
                verify(value)
                self.inputs.append(value)
            saved_reference = read(saved_paths["saved_gold_receipt_reference_binding"])
            expected_raw = bind(original / "BLIND_GOLD_revision_01.json")
            expected_call = bind(original / "BLIND_GOLD_CALL_revision_01.json")
            if (saved_reference.get("raw_binding") != expected_raw
                    or saved_reference.get("call_receipt_binding") != expected_call
                    or saved_reference.get("copied_as_new_call") is not False):
                raise RuntimeError(f"CONTINUATION_SAVED_REVISION_GOLD_REFERENCE_DRIFT:{path}")
            self.inputs.extend((expected_raw, expected_call))
            normalization = read(saved_paths["normalization_binding"])
            normalized_binding = normalization.get("normalized_input_binding")
            if (normalization.get("saved_raw_binding") != expected_raw
                    or normalization.get("saved_call_receipt_binding") != expected_call
                    or normalization.get("adapter_binding") != bind(saved_variant["normalization_adapter_path"])
                    or normalization.get("exact_round_trip_verified") is not True
                    or normalization.get("gold_decisions_evidence_and_all_values_changed") is not False
                    or normalization.get("no_new_gold_call") is not True
                    or normalization.get("semantic_admission_asserted") is not False
                    or not isinstance(normalized_binding, dict)
                    or local_path(normalized_binding["path"]) != final / "MECHANICAL_SAVED_REVISION_GOLD_INPUT.json"):
                raise RuntimeError(f"CONTINUATION_SAVED_REVISION_GOLD_NORMALIZATION_DRIFT:{path}")
            verify(normalized_binding)
            self.inputs.append(normalized_binding)
            frozen = read(saved_paths["input_freeze_binding"])
            if ((frozen.get("root_id"), frozen.get("arm"), frozen.get("input_work_dir")) != (root, arm, str(original))
                    or frozen.get("serialization_variant") != saved_variant["serialization_variant"]
                    or frozen.get("saved_gold_receipt_binding") != expected_call
                    or frozen.get("no_new_gold_call") is not True
                    or frozen.get("no_new_plan_or_author_call") is not True
                    or frozen.get("no_second_revision") is not True):
                raise RuntimeError(f"CONTINUATION_SAVED_REVISION_INPUT_FREEZE_DRIFT:{path}")
            for value in frozen.get("input_bindings", []):
                verify(value)
                self.inputs.append(value)
            if any(file.name != "CONTENT_VERIFIER_CALL_revision_01.json" for file in final.glob("*_CALL*.json")):
                raise RuntimeError(f"CONTINUATION_SAVED_REVISION_UNAUTHORIZED_NEW_CALL_REFERENCE:{path}")
        if row["continuation_kind"] == "mechanical_saved_candidate_shape_recovery":
            expected_shape = {
                "original_draft_receipt_reference_binding": final / "ORIGINAL_DRAFT_RECEIPT_REFERENCE.json",
                "normalization_binding": final / "MECHANICAL_CANDIDATE_NORMALIZATION_INITIAL.json",
            }
            for key, target in expected_shape.items():
                value = row.get(key)
                if not isinstance(value, dict) or local_path(value["path"]) != target:
                    raise RuntimeError(f"CONTINUATION_CANDIDATE_SHAPE_REFERENCE_MISSING:{key}:{path}")
                verify(value)
                self.inputs.append(value)
            reference = read(expected_shape["original_draft_receipt_reference_binding"])
            for key, name in (("original_candidates_draft_response", "CANDIDATES_DRAFT_RESPONSE.json"),
                              ("original_candidates_draft_call_receipt", "CANDIDATES_DRAFT_CALL.json"),
                              ("original_candidates_draft", "CANDIDATES_DRAFT.json")):
                if reference.get(key) != bind(original / name):
                    raise RuntimeError(f"CONTINUATION_CANDIDATE_SHAPE_ORIGINAL_DRIFT:{key}:{path}")
                self.inputs.append(reference[key])
            if (reference.get("initial_author_call_repeated") is not False
                    or reference.get("content_revision_consumed") is not False
                    or row.get("initial_author_call_repeated") is not False
                    or row.get("mechanical_normalization_consumed_content_revision") is not False
                    or (final / "CANDIDATES_DRAFT_CALL.json").exists()
                    or read(final / "CANDIDATES_DRAFT_RESPONSE.json")
                        != read(original / "CANDIDATES_DRAFT_RESPONSE.json")):
                raise RuntimeError(f"CONTINUATION_CANDIDATE_SHAPE_SAME_INITIAL_DRAFT_LINEAGE:{path}")
            normalization = read(expected_shape["normalization_binding"])
            if (normalization.get("exact_round_trip_verified") is not True
                    or normalization.get("question_text_gold_evidence_and_semantic_fields_changed") is not False
                    or normalization.get("no_extra_semantic_budget_consumed") is not True
                    or normalization.get("authored_response_sha256")
                        != q.digest(read(original / "CANDIDATES_DRAFT_RESPONSE.json"))):
                raise RuntimeError(f"CONTINUATION_CANDIDATE_SHAPE_REVERSIBLE_LOG_MISSING:{path}")
        final_outcome_name = Path(row.get("final_outcome_binding", {}).get("path", "")).name
        if final_outcome_name not in {"OUTCOME.json", "CONTINUATION_INTEGRATION_OUTCOME.json"}:
            raise RuntimeError(f"CONTINUATION_UNRECOGNIZED_FINAL_OUTCOME:{path}")
        expected = {"original_outcome_binding": original / "OUTCOME.json",
                    "final_outcome_binding": final / final_outcome_name,
                    "final_candidates_binding": final / "CANDIDATES_FINAL.json",
                    "final_review_binding": final / "CONTENT_REVIEW_FINAL.json",
                    "slot_failures_binding": final / "SLOT_FAILURES.json"}
        for key, target in expected.items():
            value = row.get(key)
            if key == "original_outcome_binding" and value is None and not target.exists():
                continue
            if not isinstance(value, dict) or local_path(value["path"]) != target.resolve():
                raise RuntimeError(f"CONTINUATION_EXPECTED_ARTIFACT:{key}:{path}")
            verify(value)
            self.inputs.append(value)
        if final_outcome_name == "CONTINUATION_INTEGRATION_OUTCOME.json":
            driver_outcome = row.get("driver_outcome_binding")
            if not isinstance(driver_outcome, dict) or local_path(driver_outcome["path"]) != final / "OUTCOME.json":
                raise RuntimeError(f"CONTINUATION_DRIVER_OUTCOME_BINDING_MISSING:{path}")
            verify(driver_outcome)
            self.inputs.append(driver_outcome)
        plan_path = local_path(row["actual_plan_path"])
        plan_freeze_path = local_path(row["actual_plan_freeze_path"])
        plan, plan_freeze = read(plan_path), read(plan_freeze_path)
        if plan.get("root_id") != root or plan.get("arm") != arm or plan_freeze.get("plans_binding") != bind(plan_path):
            raise RuntimeError(f"CONTINUATION_ACTUAL_FROZEN_PLAN_DRIFT:{path}")
        verify(plan_freeze["plans_binding"])
        self.inputs.extend((bind(plan_path), bind(plan_freeze_path), bind(path)))
        outcome = read(local_path(row["final_outcome_binding"]["path"]))
        if outcome.get("root_id") != root or outcome.get("arm") != arm:
            raise RuntimeError(f"CONTINUATION_OUTCOME_IDENTITY_DRIFT:{path}")
        if outcome.get("review_binding") != row["final_review_binding"] or outcome.get("slot_failures_binding") != row["slot_failures_binding"]:
            raise RuntimeError(f"CONTINUATION_OUTCOME_BINDING_DRIFT:{path}")
        final_candidates = by_id(read(final / "CANDIDATES_FINAL.json"), str(final))
        review_doc = read(final / "CONTENT_REVIEW_FINAL.json")
        final_reviews = by_id(review_doc.get("reviews", []), str(final))
        if set(final_candidates) != set(final_reviews):
            raise RuntimeError(f"CONTINUATION_FINAL_REVIEW_SET_DRIFT:{path}")
        final_slots = set()
        for cid, candidate in final_candidates.items():
            slot = candidate.get("slot_id")
            if (candidate.get("root_id"), candidate.get("arm")) != (root, arm) or slot not in q.SLOTS or slot in final_slots:
                raise RuntimeError(f"CONTINUATION_FINAL_SLOT_DRIFT:{cid}")
            final_slots.add(slot)
            review = final_reviews[cid]
            if review.get("candidate_sha256") != q.gate.fingerprint(candidate) or review.get("author_id") != candidate.get("author_id") or review.get("source_binding") != candidate.get("source_binding"):
                raise RuntimeError(f"CONTINUATION_FINAL_REVIEW_BINDING_DRIFT:{cid}")
        original_candidates = by_id(read(original / "CANDIDATES_FINAL.json"), str(original)) if (original / "CANDIDATES_FINAL.json").is_file() else {}
        original_reviews = by_id(read(original / "CONTENT_REVIEW_FINAL.json").get("reviews", []), str(original)) if (original / "CONTENT_REVIEW_FINAL.json").is_file() else {}
        if row["continuation_kind"] == "saved_revision_gold_first_verifier":
            assert saved_variant is not None
            self.inputs.extend(verify_saved_final_candidates(row, path, original_candidates, original_reviews,
                                                            final_candidates, final_reviews, saved_variant))
        for cid, candidate in original_candidates.items():
            if original_reviews.get(cid, {}).get("decision") == "PASS":
                if final_candidates.get(cid) != candidate or final_reviews.get(cid) != original_reviews[cid]:
                    raise RuntimeError(f"CONTINUATION_CHANGED_OR_DROPPED_PRIOR_PASS:{cid}")
        if row["continuation_kind"] == "single_unused_content_revision":
            slots = row.get("eligible_slot_ids")
            if not isinstance(slots, list) or len(set(slots)) != len(slots) or any(slot not in q.SLOTS for slot in slots):
                raise RuntimeError(f"CONTINUATION_ELIGIBLE_SLOT_LIST_MISSING:{path}")
            for cid, candidate in final_candidates.items():
                if candidate != original_candidates.get(cid) and candidate["slot_id"] not in slots:
                    raise RuntimeError(f"CONTINUATION_CHANGED_UNAUTHORIZED_SLOT:{cid}")
        return {**row, "ready_binding": bind(path), "ready_path": str(path.resolve())}

    def select_heads(self) -> None:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.records:
            grouped.setdefault((row["root_id"], row["arm"]), []).append(row)
        for key, rows in grouped.items():
            bindings = {row["ready_binding"]["sha256"]: row for row in rows}
            superseded = set()
            for row in rows:
                prior = row.get("supersedes_ready_binding")
                if prior is not None:
                    verify(prior)
                    parent = bindings.get(prior["sha256"])
                    if parent is None or parent["ready_binding"] != prior or parent is row:
                        raise RuntimeError(f"CONTINUATION_INVALID_SUPERSESSION:{key}")
                    superseded.add(prior["sha256"])
            heads = [row for row in rows if row["ready_binding"]["sha256"] not in superseded]
            if len(heads) != 1:
                raise RuntimeError(f"CONTINUATION_AMBIGUOUS_HEAD:{key}")
            self.selected[key] = heads[0]

    def final_work(self, stage_roots: Path, arm: str, root: str) -> Path:
        row = self.selected.get((root, arm))
        return local_path(row["final_work_dir"]) if row else stage_roots / arm / "roots" / q.root_number(root)

    def final_outcome(self, work: Path, root: str, arm: str) -> Path:
        row = self.selected.get((root, arm))
        return local_path(row["final_outcome_binding"]["path"]) if row else work / "OUTCOME.json"

    def available_work(self, root: str, arm: str) -> list[Path]:
        paths = [BASE / tree / q.root_number(root) / arm for tree in TREES]
        paths.extend(local_path(row["final_work_dir"]) for row in self.records
                     if (row["root_id"], row["arm"]) == (root, arm))
        return sorted({path.resolve() for path in paths if path.is_dir()}, key=str)

    def call_work(self, stage_roots: Path, arm: str, root: str) -> Path:
        original = stage_roots / arm / "roots" / q.root_number(root)
        if original.exists():
            return original
        available = self.available_work(root, arm)
        return available[0] if available else self.final_work(stage_roots, arm, root)

    def actual_plan(self, call_path: Path | None, root: str, arm: str) -> Path:
        if call_path is not None:
            path = call_path.resolve()
            for row in self.records:
                if (root, arm) == (row["root_id"], row["arm"]) and path.parent == local_path(row["final_work_dir"]):
                    return local_path(row["actual_plan_path"])
            for work in self.available_work(root, arm):
                if path.parent == work and (work / "TASK_PLANS.json").is_file():
                    return self.checked_available_plan(work, root, arm)
        row = self.selected.get((root, arm)) if call_path is None else None
        if row:
            return local_path(row["actual_plan_path"])
        if call_path is None:
            for tree in reversed(PLAN_TREES):
                work = BASE / tree / q.root_number(root) / arm
                if (work / "TASK_PLANS.json").is_file():
                    return self.checked_available_plan(work, root, arm)
        return BASE / "question_authoring" / arm / "roots" / q.root_number(root) / "TASK_PLANS.json"

    def checked_available_plan(self, work: Path, root: str, arm: str) -> Path:
        path, freeze_path = work / "TASK_PLANS.json", work / "TASK_PLANS_FREEZE.json"
        plan, freeze = read(path), read(freeze_path)
        if plan.get("root_id") != root or plan.get("arm") != arm or freeze.get("plans_binding") != bind(path):
            raise RuntimeError(f"PARTIAL_CONTINUATION_PLAN_PROVENANCE_DRIFT:{work}")
        verify(freeze["plans_binding"])
        return path

    def initial_work(self, root: str, arm: str, original: Path) -> Path:
        shape_stages = [row for row in self.records if (row["root_id"], row["arm"]) == (root, arm)
                        and row["continuation_kind"] == "mechanical_saved_candidate_shape_recovery"]
        if len(shape_stages) > 1:
            raise RuntimeError(f"CONTINUATION_MULTIPLE_INITIAL_SERIALIZATION_RECOVERIES:{root}:{arm}")
        if shape_stages:
            # This is the same consumed author's draft in a mechanically
            # normalized serialization, followed by its first fresh reviews.
            # The original raw/schema failure stays bound separately; no extra
            # semantic draft or revision enters the initial yield denominator.
            work = local_path(shape_stages[0]["final_work_dir"])
            if not (work / "CANDIDATES_DRAFT.json").is_file():
                raise RuntimeError(f"CONTINUATION_INITIAL_NORMALIZED_DRAFT_MISSING:{root}:{arm}")
            return work
        stages = [row for row in self.records if (row["root_id"], row["arm"]) == (root, arm)
                  and row["continuation_kind"] == "mechanical_saved_plan_schema_recovery"]
        paths = {local_path(row["final_work_dir"]) for row in stages}
        paths.update(BASE / tree / q.root_number(root) / arm for tree in PLAN_TREES)
        actual = [path for path in paths if (path / "CANDIDATES_DRAFT.json").is_file()]
        if len(actual) > 1:
            raise RuntimeError(f"CONTINUATION_MULTIPLE_ACTUAL_FIRST_DRAFT_STAGES:{root}:{arm}")
        if actual:
            self.checked_available_plan(actual[0], root, arm)
            return actual[0]
        head = self.selected.get((root, arm))
        if head and head["continuation_kind"] == "mechanical_saved_plan_schema_recovery":
            return local_path(head["final_work_dir"])
        if len(stages) == 1:
            return local_path(stages[0]["final_work_dir"])
        return original

    def call_paths(self, work: Path, root: str, arm: str) -> list[Path]:
        original = BASE / "question_authoring" / arm / "roots" / q.root_number(root)
        paths = list(original.glob("*_CALL*.json"))
        for directory in self.available_work(root, arm):
            paths.extend(directory.glob("*_CALL*.json"))
        # One actual call may be referenced from another stage; identical saved
        # receipt bytes represent that same call and are included only once.
        unique: dict[tuple[str, int], Path] = {}
        for path in sorted(paths, key=lambda item: (item.parent != work, str(item))):
            value = bind(path)
            unique.setdefault((value["sha256"], value["bytes"]), path)
        return sorted(unique.values(), key=str)


class PathLookupTransformer(ast.NodeTransformer):
    """Replace exactly the reviewed artifact path lookup expressions."""
    def __init__(self, role: str) -> None:
        self.role = role
        self.counts = {"final_stage_directory": 0, "actual_call_paths": 0,
                       "final_outcome_path": 0, "provenance_stage_directory": 0,
                       "feedback_plan_path": 0, "call_plan_path": 0,
                       "g_nested_scope_validation_copy": 0, "g_payload_scope_validation_copy": 0}
        self.stage_work_seen = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        target = ast.unparse(node.targets[0]) if len(node.targets) == 1 else ""
        value = ast.unparse(node.value)
        replacement = None
        if self.role == "finalize" and target == "work" and value == "stage_roots / arm / 'roots' / root_number(root)":
            self.stage_work_seen += 1
            if self.stage_work_seen == 1:
                replacement = "_continuations.final_work(stage_roots, arm, root)"
                self.counts["final_stage_directory"] += 1
            elif self.stage_work_seen == 2:
                replacement = "_continuations.call_work(stage_roots, arm, root)"
                self.counts["provenance_stage_directory"] += 1
        elif self.role == "finalize" and target == "plan_path" and value == "stage_roots / 'F' / 'roots' / root_number(root) / 'TASK_PLANS.json'":
            replacement = "_continuations.actual_plan(None, root, 'F')"
            self.counts["feedback_plan_path"] += 1
        elif self.role == "finalize" and target == "outcome_path" and value == "work / 'OUTCOME.json'":
            replacement = "_continuations.final_outcome(work, root, arm)"
            self.counts["final_outcome_path"] += 1
        elif self.role == "author_call_input_record" and target == "plan_path" and value == "BASE / 'question_authoring' / arm / 'roots' / root_number(root) / 'TASK_PLANS.json'":
            replacement = "_continuations.actual_plan(call_path, root, arm)"
            self.counts["call_plan_path"] += 1
        if replacement:
            node.value = ast.parse(replacement, mode="eval").body
            return node
        return self.generic_visit(node)

    def visit_For(self, node: ast.For) -> ast.AST:
        if self.role == "finalize" and ast.unparse(node.target) == "call_path" and ast.unparse(node.iter) == "sorted(work.glob('*_CALL*.json'))":
            node.iter = ast.parse("_continuations.call_paths(work, root, arm)", mode="eval").body
            self.counts["actual_call_paths"] += 1
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self.role == "author_call_input_record":
            source = ast.unparse(node)
            if source == "_nested_material_keys(payload)":
                self.counts["g_nested_scope_validation_copy"] += 1
                return ast.parse("_input_scope.nested_keys(payload, call_path)", mode="eval").body
            if source == "validate_g_payload_fields(payload, role, root)":
                self.counts["g_payload_scope_validation_copy"] += 1
                return ast.parse("_input_scope.validate_payload(payload, role, root, call_path)", mode="eval").body
        return self.generic_visit(node)


class InputScopeValidation:
    """Reuse the approved non-access-attestation check on a local copy only."""

    def __init__(self) -> None:
        self._module = None
        self._approval: dict[str, Any] | None = None

    def require_approved_module(self) -> dict[str, Any]:
        if self._module is None:
            ready_path = BASE / "REVIEW_SCHEMA_ADAPTER_READY.json"
            ready = read(ready_path)
            adapter_path = BASE / "scripts" / "review_schema_adapter.py"
            if (not isinstance(ready, dict)
                    or ready.get("schema_version") != "fgr20_review_schema_adapter_ready_v1"
                    or ready.get("status") != "READY_ENGINEERING_ONLY"
                    or ready.get("adapter_binding") != bind(adapter_path)):
                raise RuntimeError("INPUT_READER_REVIEW_SCHEMA_ADAPTER_APPROVAL_DRIFT")
            verify(ready["adapter_binding"])
            module = importlib.import_module("review_schema_adapter")
            if Path(module.__file__).resolve() != adapter_path.resolve():
                raise RuntimeError("INPUT_READER_REVIEW_SCHEMA_ADAPTER_IMPORT_PATH_DRIFT")
            self._module = module
            self._approval = {"ready_binding": bind(ready_path),
                              "adapter_binding": ready["adapter_binding"]}
        assert self._approval is not None
        return copy.deepcopy(self._approval)

    def validation_copy(self, payload: dict[str, Any], role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if role != "content_verifier":
            return payload, []
        self.require_approved_module()
        return self._module.normalize_g_verifier_validation_payload(payload, role)

    def nested_keys(self, payload: dict[str, Any], call_path: Path) -> list[str]:
        role = "content_verifier" if call_path.name.startswith("CONTENT_VERIFIER_CALL") else ""
        copied, _ = self.validation_copy(payload, role)
        return q._nested_material_keys(copied)

    def validate_payload(self, payload: dict[str, Any], role: str, root: str, call_path: Path) -> None:
        if role == "content_verifier" and not call_path.name.startswith("CONTENT_VERIFIER_CALL"):
            raise RuntimeError(f"INPUT_READER_CONTENT_VERIFIER_CALL_ROLE_DRIFT:{call_path}")
        copied, _ = self.validation_copy(payload, role)
        q.validate_g_payload_fields(copied, role, root)


def adapted_function(function: Any, role: str, namespace: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    source = inspect.getsource(function)
    tree = ast.parse(source)
    transformer = PathLookupTransformer(role)
    result = ast.fix_missing_locations(transformer.visit(copy.deepcopy(tree)))
    expected = ({"final_stage_directory": 1, "provenance_stage_directory": 1, "final_outcome_path": 1,
                 "actual_call_paths": 1, "feedback_plan_path": 1, "call_plan_path": 0,
                 "g_nested_scope_validation_copy": 0, "g_payload_scope_validation_copy": 0}
                if role == "finalize" else
                {"final_stage_directory": 0, "provenance_stage_directory": 0, "final_outcome_path": 0,
                 "actual_call_paths": 0, "feedback_plan_path": 0, "call_plan_path": 1,
                 "g_nested_scope_validation_copy": 1, "g_payload_scope_validation_copy": 1})
    if transformer.counts != expected or role == "finalize" and transformer.stage_work_seen != 2:
        raise RuntimeError(f"FROZEN_FINALIZER_LOOKUP_STRUCTURE_DRIFT:{role}:{transformer.counts}")
    exec(compile(result, str(Path(__file__).resolve()) + ":" + role, "exec"), namespace)
    return namespace[function.__name__], {"function": role, "lookup_replacements": transformer.counts,
        "original_function_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapted_function_ast_sha256": hashlib.sha256(ast.dump(result, include_attributes=False).encode()).hexdigest()}


def finalize(continuations: Continuations | None = None) -> dict[str, Any]:
    audit = read(BASE / "QUESTION_DRIVER_AUDIT_READY.json")
    q.verify_binding(audit["driver_binding"])
    actual = continuations or Continuations()
    namespace = dict(q.__dict__)
    namespace["_continuations"] = actual
    input_scope = InputScopeValidation()
    namespace["_input_scope"] = input_scope
    input_scope_approval = input_scope.require_approved_module()
    input_reader, reader_receipt = adapted_function(q.author_call_input_record, "author_call_input_record", namespace)
    namespace["author_call_input_record"] = input_reader
    function, finalizer_receipt = adapted_function(q.finalize, "finalize", namespace)
    receipt = {"schema_version": "fgr20_content_finalizer_artifact_path_adaptation_v1",
        "status": "ENGINEERING_ONLY_ARTIFACT_AND_SCOPE_VALIDATION_ADAPTATION_BEFORE_SOLVER",
        "frozen_question_driver_binding": bind(Path(q.__file__).resolve()),
        "adapter_binding": bind(Path(__file__).resolve()),
        "function_adaptations": [reader_receipt, finalizer_receipt],
        "ready_records": actual.records,
        "selected_final_work_directories": [{"root_id": root, "arm": arm, "final_work_dir": row["final_work_dir"],
                                              "ready_binding": row["ready_binding"]}
                                             for (root, arm), row in sorted(actual.selected.items())],
        "input_bindings": sorted({row["path"]: row for row in actual.inputs}.values(), key=lambda row: row["path"]),
        "candidate_review_duplicate_and_gate_logic_changed": False,
        "input_reader_scope_validation": {
            "policy": "Only a validation-only copy of content_verifier review_cases[*].saved_independent_gold.solver_results_read=false may omit that exact non-access attestation. Actual saved prompts and their source/plan checks stay unchanged; other keys or nonfalse values remain banned.",
            "pure_validation_copy_function": "review_schema_adapter.normalize_g_verifier_validation_payload",
            "approved_module": input_scope_approval,
            "actual_prompt_modified": False, "semantic_admission_asserted": False,
        },
        "model_calls": 0, "semantic_judgments_authored": 0}
    publish(RECEIPT, receipt)
    extra = [Path(__file__).resolve(), RECEIPT, *actual.tree_files]
    for name in ("question_plan_schema_continuation.py", "question_plan_schema_continuation_v2.py",
                 "question_plan_schema_continuation_v3.py",
                 "question_candidate_shape_continuation.py",
                 "review_schema_adapter.py",
                 "unused_failed_lineage_adapter.py",
                 "saved_revision_gold_verification.py",
                 "g136202_saved_global_gold_verification.py",
                 "g109164_saved_global7_gold_verification.py",
                 "f179186_saved_shape_gold_verification.py",
                 "question_unfilled_slot_continuation.py"):
        path = BASE / "scripts" / name
        if path.is_file():
            extra.append(path)
    for name in ("QUESTION_PLAN_SCHEMA_CONTINUATION_READY.json",
                 "QUESTION_PLAN_SCHEMA_CONTINUATION_V2_READY.json",
                 "QUESTION_PLAN_SCHEMA_CONTINUATION_V3_READY.json",
                 "QUESTION_CANDIDATE_SHAPE_CONTINUATION_READY.json",
                 "REVIEW_SCHEMA_ADAPTER_READY.json",
                 "UNUSED_FAILED_LINEAGE_ADAPTER_READY.json",
                 "G136202_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
                 "G109164_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
                 "F179186_SAVED_SHAPE_GOLD_VERIFICATION_READY.json",
                 "PRODUCTION_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json",
                 "PRODUCTION_G109_QUESTION_PROCESS_GUARD_READY.json",
                 "PRODUCTION_G109_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json",
                 "SAVED_REVISION_GOLD_VERIFICATION_READY.json",
                 "QUESTION_UNFILLED_SLOT_CONTINUATION_READY.json"):
        path = BASE / name
        if path.is_file():
            extra.append(path)
    namespace["EXTRA_DRIVER_BINDINGS"] = list(q.EXTRA_DRIVER_BINDINGS) + extra
    return function()


def self_check() -> dict[str, Any]:
    namespace = dict(q.__dict__)
    namespace["_continuations"] = None
    namespace["_input_scope"] = InputScopeValidation()
    _, reader = adapted_function(q.author_call_input_record, "author_call_input_record", namespace)
    _, finalizer = adapted_function(q.finalize, "finalize", namespace)
    scope = namespace["_input_scope"]
    fixture = {"review_cases": [{"saved_independent_gold": {"solver_results_read": False}}]}
    verifier_path = Path("CONTENT_VERIFIER_CALL_revision_01.json")
    assert scope.nested_keys(fixture, verifier_path) == []
    assert fixture["review_cases"][0]["saved_independent_gold"]["solver_results_read"] is False
    for invalid in (True, "false", None, 0):
        malformed = {"review_cases": [{"saved_independent_gold": {"solver_results_read": invalid}}]}
        assert scope.nested_keys(malformed, verifier_path)
    assert scope.nested_keys(fixture, Path("BLIND_GOLD_CALL.json"))
    # Exercise the one permitted archive-only failure partition with synthetic
    # fixture records.  No real experiment candidate, review, or READY is read.
    with tempfile.TemporaryDirectory(prefix="fgr20_content_partition_", dir=BASE / "runtime_cache") as directory:
        fixture_base = Path(directory)
        original = fixture_base / "original"
        final = fixture_base / "final"
        original.mkdir()
        final.mkdir()
        covered = ["MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER3"]
        true_candidates = [{"candidate_id": "FIXTURE-" + slot, "slot_id": slot} for slot in covered]
        failure = {"candidate_id": "FIXTURE-ORDER2", "slot_id": "ORDER2", "candidate_status": "SLOT_FAILURE"}
        original_rows = [*true_candidates, failure]
        original_review_doc = {"reviews": [{"candidate_id": item["candidate_id"], "decision": "HOLD"} for item in original_rows]}
        failure_rows = [{"slot_id": "ORDER2", "reason": "FIXTURE_ORIGINAL_AUTHOR_FAILURE"}]
        publish(original / "CANDIDATES_FINAL.json", original_rows)
        publish(original / "CONTENT_REVIEW_FINAL.json", original_review_doc)
        publish(original / "SLOT_FAILURES.json", failure_rows)
        (final / "ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json").write_bytes((original / "CANDIDATES_FINAL.json").read_bytes())
        publish(final / "SLOT_FAILURES.json", failure_rows)
        partition_row = {"original_work_dir": str(original), "final_work_dir": str(final),
                         "covered_slot_ids": covered, "uncovered_slot_ids": ["ORDER2"],
                         "uncovered_ORDER2_original_failure_notice_and_review_preserved": True,
                         "original_final_candidates_archive_binding": bind(final / "ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json"),
                         "uncovered_order2_original_review_binding": bind(original / "CONTENT_REVIEW_FINAL.json")}
        original_map = by_id(original_rows, "fixture")
        original_review_map = by_id(original_review_doc["reviews"], "fixture")
        final_map = by_id(true_candidates, "fixture")
        final_review_map = {key: value for key, value in original_review_map.items() if key != failure["candidate_id"]}
        approved = {"omit_original_order2_failure": True}
        assert len(verify_saved_final_candidates(partition_row, final / READY_NAME, original_map,
                   original_review_map, final_map, final_review_map, approved)) == 3
        # The generic saved-verifier family cannot silently omit any candidate.
        for defect in ("generic_omission", "authored_question", "gold", "prior_pass", "missing_real_candidate", "changed_real_candidate", "review_shell"):
            args = [copy.deepcopy(partition_row), final / READY_NAME, copy.deepcopy(original_map),
                    copy.deepcopy(original_review_map), copy.deepcopy(final_map), copy.deepcopy(final_review_map), dict(approved)]
            if defect == "generic_omission": args[6]["omit_original_order2_failure"] = False
            elif defect == "authored_question": args[2][failure["candidate_id"]]["question"] = "FIXTURE_NOT_A_FAILURE"
            elif defect == "gold": args[2][failure["candidate_id"]]["gold"] = ["A"]
            elif defect == "prior_pass": args[3][failure["candidate_id"]]["decision"] = "PASS"
            elif defect == "missing_real_candidate": args[4].pop("FIXTURE-MC2")
            elif defect == "changed_real_candidate": args[4]["FIXTURE-MC2"]["question"] = "FIXTURE_CHANGED"
            else: args[5][failure["candidate_id"]] = args[3][failure["candidate_id"]]
            try:
                verify_saved_final_candidates(*args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Unsafe saved-candidate partition accepted: " + defect)
    return {"status": "PASS", "function_adaptations": [reader, finalizer],
            "scope_validation_approval": scope.require_approved_module(),
            "scope_validation_checks": ["false_nonaccess_only_for_content_verifier",
                "actual_payload_unchanged", "true_string_null_zero_remain_banned", "blind_role_not_allowlisted"],
            "frozen_question_driver_binding": bind(Path(q.__file__).resolve()),
            "saved_candidate_partition_checks": ["only_exact_original_ORDER2_failure_omitted",
                "seven_real_candidates_unchanged", "byte_exact_original_archive_required",
                "original_HOLD_and_failure_ledger_retained", "generic_omission_rejected",
                "authored_question_gold_prior_PASS_missing_or_changed_candidate_and_review_shell_rejected"],
            "temporary_partition_fixture_files_removed": True,
            "models_called": False, "real_continuations_read": False,
            "files_written": False, "semantic_judgments_authored": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_check(), ensure_ascii=False, indent=2))
