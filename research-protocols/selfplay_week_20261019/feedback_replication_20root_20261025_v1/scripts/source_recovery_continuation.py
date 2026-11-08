"""Continue source review for saved drafts that failed only mechanical validation.

This file does not alter any source_driver or source_call_recovery artifact. It
uses the already-saved draft, obtains a fresh independent review, and allows at
most one fresh revision plus recheck. Calls run only after the recovery batch is
idle and use distinct call IDs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import codex_calls as cc
import source_call_recovery as recovery
import source_driver as source

RECOVERY = BASE / "source_recovery"
CONTINUATIONS = BASE / "source_recovery_continuations"
MAX_PARALLEL_CALLS = 3


class ContinuationError(RuntimeError):
    pass


def _write(root_dir: Path, name: str, value: Any, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    binding = recovery.write_immutable(root_dir / name, value)
    if binding not in artifacts:
        artifacts.append(binding)
    return binding


def _call_id(root_id: str, role: str) -> str:
    if not root_id.startswith("cc-train-episode-") or not role:
        raise ContinuationError("Malformed continuation root_id or call role")
    return f"{root_id.replace('-', '_')}_source_recovery_continuation_{role}"


def _call(root_id: str, role: str, prompt: str, call_ids_attempted: list[str]):
    call_id = _call_id(root_id, role)
    cdir = BASE / "cli_calls" / call_id
    if cdir.exists():
        raise ContinuationError(f"Refusing to reuse an existing CLI call_id: {call_id}")
    call_ids_attempted.append(call_id)
    return call_id, cc.call(call_id, prompt)


def _review(root_id: str, suffix: str, embedded: dict[str, Any], draft: dict[str, Any],
            mechanical: dict[str, Any], root_dir: Path, artifacts: list[dict[str, Any]],
            call_ids_attempted: list[str]):
    review_input = {"complete_inputs": embedded, "complete_normalized_draft": draft,
                    "mechanical_validation": mechanical}
    prompt = (cc.prefix("independent shared-source reviewer") + source.REVIEW_RULES
              + "\nCOMPLETE_REVIEW_INPUTS:\n"
              + json.dumps(review_input, ensure_ascii=False, separators=(",", ":")))
    call_id, (review, call_result) = _call(root_id, suffix, prompt, call_ids_attempted)
    review_binding = _write(root_dir, f"{suffix.upper()}_REVIEW.json", review, artifacts)
    call_binding = _write(root_dir, f"{suffix.upper()}_REVIEW_CALL.json", call_result, artifacts)
    audit = source.validate_review(review, embedded["neutral_packet"],
                                   embedded["complete_parent_source"], draft,
                                   [{"reservation": row["reservation"], "complete_record": row["complete_record"]}
                                    for row in embedded["complete_native_records"]])
    audit_binding = _write(root_dir, f"{suffix.upper()}_REVIEW_VALIDATION.json", audit, artifacts)
    return review, audit, {"call_id": call_id, "review_binding": review_binding,
                           "call_binding": call_binding, "audit_binding": audit_binding}


def _revision(root_id: str, embedded: dict[str, Any], draft: dict[str, Any], mechanical: dict[str, Any],
              review: dict[str, Any], audit: dict[str, Any], packet: dict[str, Any], parent: dict[str, Any],
              natives: list[dict[str, Any]], root_dir: Path, artifacts: list[dict[str, Any]],
              call_ids_attempted: list[str]):
    correction = {"previous_normalized_draft": draft, "mechanical_validation": mechanical,
                  "independent_review": review, "review_contract_validation": audit}
    prompt = (cc.prefix("neutral shared-source author performing the one allowed revision")
              + source.AUTHOR_RULES
              + "\nThis is the only revision. Resolve all blocking findings; preserve all valid material.\nCOMPLETE_INPUTS:\n"
              + json.dumps(embedded, ensure_ascii=False, separators=(",", ":"))
              + "\nPREVIOUS_DRAFT_AND_FEEDBACK:\n"
              + json.dumps(correction, ensure_ascii=False, separators=(",", ":")))
    call_id, (raw, call_result) = _call(root_id, "author_revision", prompt, call_ids_attempted)
    raw_binding = _write(root_dir, "AUTHOR_REVISION_RESPONSE.json", raw, artifacts)
    call_binding = _write(root_dir, "AUTHOR_REVISION_CALL.json", call_result, artifacts)
    normalized, warnings = source.normalize(raw, packet)
    draft_binding = _write(root_dir, "DRAFT_REVISION.json", normalized, artifacts)
    warning_binding = _write(root_dir, "METADATA_NORMALIZATION_REVISION.json", warnings, artifacts)
    mechanical_revision = source.validate_draft(normalized, packet, parent, natives)
    mechanical_binding = _write(root_dir, "MECHANICAL_REVISION.json", mechanical_revision, artifacts)
    lineage = {"call_id": call_id, "author_response_binding": raw_binding, "call_binding": call_binding,
               "draft_binding": draft_binding, "normalization_warnings_binding": warning_binding,
               "mechanical_binding": mechanical_binding, "fresh_thread": True,
               "revision_count": 1, "original_author_lineage_only": True}
    return normalized, warnings, mechanical_revision, draft_binding, lineage


def _active_recovery_batch() -> list[str]:
    result = subprocess.run(["ps", "-eo", "args="], check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines()
            if "scripts/source_call_recovery.py --recover" in line]


def continue_root(root_id: str) -> dict[str, Any]:
    outcome_path = RECOVERY / root_id / "ROOT_RECOVERY.json"
    if not outcome_path.exists():
        raise ContinuationError(f"Missing completed recovery outcome for {root_id}")
    original_outcome = recovery.read(outcome_path)
    if original_outcome.get("root_id") != root_id or original_outcome.get("status") != "FAILED_MECHANICAL":
        raise ContinuationError(f"Only a terminal FAILED_MECHANICAL root may continue: {root_id}")
    for binding in original_outcome.get("artifacts", []):
        if recovery.bind(binding["path"]) != binding:
            raise ContinuationError(f"Original recovery artifact drift: {binding['path']}")

    root_dir = CONTINUATIONS / root_id
    root_dir.mkdir(parents=True, exist_ok=True)
    lock_path = RECOVERY / ".locks" / f"{root_id}.lock"
    if not lock_path.exists():
        raise ContinuationError(f"Missing recovery lock for {root_id}")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContinuationError(f"Recovery batch still holds root lock: {root_id}") from exc

        result_path = root_dir / "CONTINUATION_OUTCOME.json"
        if result_path.exists():
            result = recovery.read(result_path)
            for binding in result.get("artifacts", []):
                if recovery.bind(binding["path"]) != binding:
                    raise ContinuationError(f"Continuation artifact drift: {binding['path']}")
            return result

        artifacts: list[dict[str, Any]] = []
        result: dict[str, Any] = {
            "schema_version": "fgr20_source_recovery_mechanical_continuation_v1",
            "root_id": root_id, "status": "FAILED_CONTINUATION",
            "original_recovery_outcome_binding": recovery.bind(outcome_path),
            "original_recovery_outcome_untouched": True,
            "initial_draft_reused_without_model_call": True,
            "initial_mechanical_failure_requires_review_and_revision": True,
            "max_additional_revisions": 1, "substitution_allowed": False,
            "call_ids_attempted": [], "artifacts": artifacts,
        }
        try:
            if _active_recovery_batch():
                raise ContinuationError("Recovery batch is still active; continuation calls are refused")
            if cc.stop_generation():
                raise ContinuationError("Generation cutoff reached before continuation review")

            rows = {row["root_id"]: row for row in source.validate_preparation()}
            if root_id not in rows:
                raise ContinuationError(f"Root is not in frozen Train selection: {root_id}")
            packet, parent, natives, embedded = recovery._source_prompt_inputs(source, rows[root_id])
            saved_dir = RECOVERY / root_id
            draft_path, mechanical_path = saved_dir / "DRAFT_0.json", saved_dir / "MECHANICAL_0.json"
            for expected in original_outcome.get("artifacts", []):
                if Path(expected["path"]).resolve() in (draft_path.resolve(), mechanical_path.resolve()):
                    if recovery.bind(expected["path"]) != expected:
                        raise ContinuationError("Saved initial draft/mechanical binding drift")
            if not draft_path.exists() or not mechanical_path.exists():
                raise ContinuationError("Saved DRAFT_0 or MECHANICAL_0 is missing")
            draft, mechanical = recovery.read(draft_path), recovery.read(mechanical_path)
            if mechanical.get("status") == "PASS":
                raise ContinuationError("Saved mechanical result no longer has a failure")
            _write(root_dir, "INITIAL_DRAFT_REFERENCE.json", {
                "draft_binding": recovery.bind(draft_path),
                "mechanical_binding": recovery.bind(mechanical_path),
                "original_outcome_binding": recovery.bind(outcome_path),
                "reused_without_semantic_call": True,
            }, artifacts)

            review, audit, review_lineage = _review(root_id, "initial_review", embedded, draft,
                                                     mechanical, root_dir, artifacts,
                                                     result["call_ids_attempted"])
            result["initial_review_lineage"] = review_lineage
            result["initial_review_validation"] = audit
            result["mechanical_initial"] = mechanical

            if cc.stop_generation():
                raise ContinuationError("Generation cutoff reached after initial review")
            revised, warnings, mechanical_revised, revised_binding, revision_lineage = _revision(
                root_id, embedded, draft, mechanical, review, audit, packet, parent, natives,
                root_dir, artifacts, result["call_ids_attempted"])
            result["revision_lineage"] = revision_lineage
            result["mechanical_revision"] = mechanical_revised
            if cc.stop_generation():
                raise ContinuationError("Generation cutoff reached after revision; no recheck started")
            recheck, recheck_audit, recheck_lineage = _review(root_id, "revision_recheck", embedded,
                                                               revised, mechanical_revised,
                                                               root_dir, artifacts, result["call_ids_attempted"])
            result["revision_recheck_lineage"] = recheck_lineage
            result["revision_recheck_validation"] = recheck_audit

            if mechanical_revised.get("status") == "PASS" and recheck_audit.get("status") == "PASS":
                source_value = recovery._assemble_source(
                    root_id, packet, parent, revised, recheck, recheck_lineage["review_binding"],
                    revised_binding, warnings, 1, source)
                source_binding = _write(root_dir, "SOURCE_FINAL.json", source_value, artifacts)
                result.update(status="PASS", source_binding=source_binding,
                              parent_prefix_preserved=True, revision_count=1,
                              warnings=recheck.get("warnings", []))
            else:
                result.update(status="FAILED_REVIEW_OR_MECHANICAL", revision_count=1,
                              failure_reason="The single revision did not pass mechanical and independent review checks")
        except Exception as exc:
            result.update(status="FAILED_CONTINUATION", failure_reason=f"{type(exc).__name__}: {exc}",
                          error_type=type(exc).__name__)
        result["artifacts"] = artifacts
        recovery.write_immutable(result_path, result)
        return result


def publish_continued_freeze() -> dict[str, Any]:
    if _active_recovery_batch():
        raise ContinuationError("Recovery batch is still active; publication is refused")
    sd_roots = source.validate_preparation()
    selection_roots = []
    root_outcomes, continuation_outcomes = [], []
    failures = []
    for row in sd_roots:
        root_id = row["root_id"]
        base_path = RECOVERY / root_id / "ROOT_RECOVERY.json"
        if not base_path.exists():
            raise ContinuationError(f"Missing terminal recovery outcome for {root_id}")
        base_outcome = recovery.read(base_path)
        root_outcomes.append(recovery.bind(base_path))
        source_binding = base_outcome.get("source_binding") if base_outcome.get("status") in (
            "PASS", "PASS_ORIGINAL_SOURCE_REUSED") else None
        continuation_path = CONTINUATIONS / root_id / "CONTINUATION_OUTCOME.json"
        if base_outcome.get("status") == "FAILED_MECHANICAL":
            if not continuation_path.exists():
                raise ContinuationError(f"Missing mechanical continuation outcome for {root_id}")
            continuation = recovery.read(continuation_path)
            for binding in continuation.get("artifacts", []):
                if recovery.bind(binding["path"]) != binding:
                    raise ContinuationError(f"Continuation artifact drift: {binding['path']}")
            continuation_outcomes.append(recovery.bind(continuation_path))
            if continuation.get("status") == "PASS":
                source_binding = continuation.get("source_binding")
        if source_binding:
            if not recovery._binding_matches(source_binding):
                raise ContinuationError(f"Passing source binding drift for {root_id}")
            selection_roots.append({"root_id": root_id, **source_binding})
        else:
            failures.append({"root_id": root_id, "base_outcome_binding": recovery.bind(base_path),
                             "base_status": base_outcome.get("status"),
                             "continuation_outcome_binding": (recovery.bind(continuation_path)
                                 if continuation_path.exists() else None)})

    selection = {"schema_version": "closed_loop_source_selection_v1", "split": "train",
                 "roots": selection_roots}
    selection_path = BASE / "SOURCE_SELECTION_RECOVERY_CONTINUED.json"
    selection_binding = recovery.write_immutable(selection_path, selection)
    failure_value = {"schema_version": "fgr20_source_failures_continued_v1", "planned_roots": 16,
                     "passed_roots": len(selection_roots), "failed_roots": failures,
                     "substitutions": [], "original_recovery_outcomes_preserved": True}
    failure_path = BASE / "SOURCE_FAILURES_RECOVERY_CONTINUED.json"
    failure_binding = recovery.write_immutable(failure_path, failure_value)
    freeze = {"schema_version": "fgr20_source_freeze_v1",
              "status": "PASS" if len(selection_roots) == 16 else "PARTIAL",
              "planned_root_count": 16, "passed_root_count": len(selection_roots),
              "failed_root_count": 16 - len(selection_roots),
              "source_selection_binding": selection_binding, "failures_binding": failure_binding,
              "parent_selection_binding": recovery.bind(BASE / "PARENT_SOURCE_SELECTION.json"),
              "root_outcomes": root_outcomes, "continuation_outcomes": continuation_outcomes,
              "training": False, "old_sources_modified": False, "original_four_rerun": False,
              "same_shared_source_for_paired_F_G": True,
              "mechanical_failure_continuations_are_separately_bound": True}
    freeze_path = BASE / "SOURCE_FREEZE_RECOVERY_CONTINUED.json"
    freeze_binding = recovery.write_immutable(freeze_path, freeze)
    return {"status": freeze["status"], "passed": len(selection_roots),
            "failed": 16 - len(selection_roots), "selection_binding": selection_binding,
            "failures_binding": failure_binding, "freeze_binding": freeze_binding}


def self_check() -> dict[str, Any]:
    sample_root = "cc-train-episode-75241"
    call_ids = [_call_id(sample_root, role) for role in
                ("initial_review", "author_revision", "revision_recheck")]
    if len(set(call_ids)) != 3 or any("_source_recovery_continuation_" not in call_id for call_id in call_ids):
        raise ContinuationError("Continuation CLI call_id isolation check failed")
    if MAX_PARALLEL_CALLS != 3 or not source.AUTHOR_RULES or not source.REVIEW_RULES:
        raise ContinuationError("Frozen source author/reviewer contracts are unavailable")
    if not hasattr(source, "validate_draft") or not hasattr(source, "validate_review"):
        raise ContinuationError("Frozen source mechanical gates are unavailable")
    return {"status": "PASS", "model_calls": 0, "task_artifacts_written": False,
            "unique_call_id_roles": ["initial_review", "author_revision", "revision_recheck"],
            "max_parallel_calls": MAX_PARALLEL_CALLS,
            "uses_frozen_source_contract_and_gates": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--continue-roots", nargs="+", metavar="NUMERIC_ROOT_ID")
    group.add_argument("--publish-continued-freeze", action="store_true")
    group.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, indent=2))
        return 0
    if args.publish_continued_freeze:
        print(json.dumps(publish_continued_freeze(), ensure_ascii=False, indent=2))
        return 0
    numeric = [int(value) for value in args.continue_roots]
    if len(set(numeric)) != len(numeric):
        raise ContinuationError("Duplicate root IDs are not allowed")
    roots = [f"cc-train-episode-{value}" for value in numeric]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        outcomes = list(pool.map(continue_root, roots))
    print(json.dumps({"schema_version": "fgr20_source_recovery_continuation_batch_v1",
                      "outcomes": outcomes,
                      "counts": {status: sum(row.get("status") == status for row in outcomes)
                                 for status in sorted({row.get("status") for row in outcomes})},
                      "original_recovery_outcomes_modified": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
