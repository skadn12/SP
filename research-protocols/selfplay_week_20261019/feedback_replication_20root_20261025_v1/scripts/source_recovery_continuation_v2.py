"""Evidence-preserving continuation for four source reviews misclassified by the helper.

The initial review responses already exist in bound Codex CLI artifacts. This
script classifies and reuses those responses without a model call, then uses
fresh call IDs for the single allowed author revision and independent recheck.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import json
from pathlib import Path
import sys
from typing import Any

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import codex_calls as cc  # noqa: E402
import question_driver  # noqa: E402,F401  # installs the exact event adapter
import source_call_recovery as recovery  # noqa: E402
import source_driver  # noqa: E402

OLD_CONT = BASE / "source_recovery_continuations"
CONT = BASE / "source_recovery_continuations_v2"
MAX_PARALLEL = 3


def write(root_dir: Path, name: str, value: Any, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    binding = recovery.write_immutable(root_dir / name, value)
    if binding not in artifacts:
        artifacts.append(binding)
    return binding


def call_id(root_id: str, role: str) -> str:
    return f"{root_id.replace('-', '_')}_source_recovery_continuation_v2_{role}"


def fresh_call(root_id: str, role: str, prompt: str, attempted: list[str]):
    cid = call_id(root_id, role)
    if (BASE / "cli_calls" / cid).exists():
        raise RuntimeError(f"Refusing to reuse existing v2 call id: {cid}")
    attempted.append(cid)
    return cid, cc.call(cid, prompt)


def recovered_initial_review(root_id: str, embedded: dict[str, Any], draft: dict[str, Any],
                            mechanical: dict[str, Any], root_dir: Path,
                            artifacts: list[dict[str, Any]]):
    cid = f"{root_id.replace('-', '_')}_source_recovery_continuation_initial_review"
    cdir = BASE / "cli_calls" / cid
    attempt = cdir / "attempt_01"
    terminal = recovery.read(attempt / "TERMINAL.json")
    classification = recovery.classify_saved_attempt(cid, cdir, attempt,
                                                     terminal,
                                                     expected_root_id=root_id)
    sidecar = write(root_dir, "INITIAL_REVIEW_EVENT_CLASSIFICATION.json", classification, artifacts)
    raw = recovery._strict_json_text((attempt / "LAST_MESSAGE.txt").read_text())
    response_binding = write(root_dir, "INITIAL_REVIEW_RECOVERED.json", raw, artifacts)
    call_record = {
        "schema_version": "fgr20_recovered_saved_source_review_v2",
        "call_id": cid,
        "classification_binding": sidecar,
        "response_binding": response_binding,
        "original_cli_failure_preserved": recovery.bind(cdir / "SEMANTIC_FAILURE.json"),
        "no_model_call_for_reconstruction": True,
    }
    call_binding = write(root_dir, "INITIAL_REVIEW_CALL_RECOVERED.json", call_record, artifacts)
    audit = source_driver.validate_review(
        raw, embedded["neutral_packet"], embedded["complete_parent_source"], draft,
        [{"reservation": row["reservation"], "complete_record": row["complete_record"]}
         for row in embedded["complete_native_records"]],
    )
    audit_binding = write(root_dir, "INITIAL_REVIEW_VALIDATION_RECOVERED.json", audit, artifacts)
    return raw, audit, {"call_id": cid, "classification_binding": sidecar,
                        "response_binding": response_binding, "call_binding": call_binding,
                        "audit_binding": audit_binding}


def review_prompt(root_id: str, embedded: dict[str, Any], draft: dict[str, Any],
                  mechanical: dict[str, Any]) -> str:
    payload = {"complete_inputs": embedded, "complete_normalized_draft": draft,
               "mechanical_validation": mechanical}
    return (cc.prefix("independent shared-source reviewer") + source_driver.REVIEW_RULES
            + "\nCOMPLETE_REVIEW_INPUTS:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def continue_one(root_id: str) -> dict[str, Any]:
    old_path = OLD_CONT / root_id / "CONTINUATION_OUTCOME.json"
    if not old_path.exists():
        raise RuntimeError(f"Missing prior continuation outcome: {root_id}")
    old = recovery.read(old_path)
    if old.get("status") != "FAILED_CONTINUATION":
        raise RuntimeError(f"Unexpected prior continuation status for {root_id}: {old.get('status')}")
    root_dir = CONT / root_id
    root_dir.mkdir(parents=True, exist_ok=True)
    result_path = root_dir / "CONTINUATION_OUTCOME.json"
    if result_path.exists():
        return recovery.read(result_path)
    artifacts: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": "fgr20_source_recovery_mechanical_continuation_v2",
        "root_id": root_id,
        "status": "FAILED_CONTINUATION",
        "prior_continuation_binding": recovery.bind(old_path),
        "prior_continuation_untouched": True,
        "saved_initial_review_reused_without_model_call": True,
        "max_additional_revisions": 1,
        "call_ids_attempted": [],
        "artifacts": artifacts,
    }
    try:
        rows = {row["root_id"]: row for row in source_driver.validate_preparation()}
        if root_id not in rows:
            raise RuntimeError(f"Root not in frozen preparation: {root_id}")
        packet, parent, natives, embedded = recovery._source_prompt_inputs(source_driver, rows[root_id])
        saved = BASE / "source_recovery" / root_id
        draft = recovery.read(saved / "DRAFT_0.json")
        mechanical = recovery.read(saved / "MECHANICAL_0.json")
        write(root_dir, "INITIAL_DRAFT_REFERENCE.json", {
            "draft_binding": recovery.bind(saved / "DRAFT_0.json"),
            "mechanical_binding": recovery.bind(saved / "MECHANICAL_0.json"),
            "prior_continuation_binding": recovery.bind(old_path),
            "reused_without_semantic_call": True,
        }, artifacts)
        review, audit, initial_lineage = recovered_initial_review(
            root_id, embedded, draft, mechanical, root_dir, artifacts)
        result["initial_review_lineage"] = initial_lineage
        result["initial_review_validation"] = audit
        result["mechanical_initial"] = mechanical
        correction = {"previous_normalized_draft": draft, "mechanical_validation": mechanical,
                      "independent_review": review, "review_contract_validation": audit}
        revision_prompt = (
            cc.prefix("neutral shared-source author performing the one allowed revision")
            + source_driver.AUTHOR_RULES
            + "\nThis is the only revision. Resolve all blocking findings; preserve all valid material.\nCOMPLETE_INPUTS:\n"
            + json.dumps(embedded, ensure_ascii=False, separators=(",", ":"))
            + "\nPREVIOUS_DRAFT_AND_FEEDBACK:\n"
            + json.dumps(correction, ensure_ascii=False, separators=(",", ":"))
        )
        cid, (raw_revision, call_result) = fresh_call(
            root_id, "author_revision", revision_prompt, result["call_ids_attempted"])
        raw_binding = write(root_dir, "AUTHOR_REVISION_RESPONSE.json", raw_revision, artifacts)
        call_binding = write(root_dir, "AUTHOR_REVISION_CALL.json", call_result, artifacts)
        revised, warnings = source_driver.normalize(raw_revision, packet)
        revised_binding = write(root_dir, "DRAFT_REVISION.json", revised, artifacts)
        warning_binding = write(root_dir, "METADATA_NORMALIZATION_REVISION.json", warnings, artifacts)
        mechanical_revision = source_driver.validate_draft(revised, packet, parent, natives)
        mechanical_binding = write(root_dir, "MECHANICAL_REVISION.json", mechanical_revision, artifacts)
        result["revision_lineage"] = {"call_id": cid, "author_response_binding": raw_binding,
                                       "call_binding": call_binding, "draft_binding": revised_binding,
                                       "normalization_warnings_binding": warning_binding,
                                       "mechanical_binding": mechanical_binding,
                                       "fresh_thread": True, "revision_count": 1,
                                       "original_author_lineage_only": True}
        result["mechanical_revision"] = mechanical_revision
        recheck_prompt = review_prompt(root_id, embedded, revised, mechanical_revision)
        rcid, (recheck, recheck_call) = fresh_call(
            root_id, "revision_recheck", recheck_prompt, result["call_ids_attempted"])
        recheck_binding = write(root_dir, "REVISION_RECHECK.json", recheck, artifacts)
        recheck_call_binding = write(root_dir, "REVISION_RECHECK_CALL.json", recheck_call, artifacts)
        recheck_audit = source_driver.validate_review(
            recheck, embedded["neutral_packet"], embedded["complete_parent_source"], revised,
            [{"reservation": row["reservation"], "complete_record": row["complete_record"]}
             for row in embedded["complete_native_records"]],
        )
        recheck_audit_binding = write(root_dir, "REVISION_RECHECK_VALIDATION.json", recheck_audit, artifacts)
        result["revision_recheck_lineage"] = {"call_id": rcid, "review_binding": recheck_binding,
                                              "call_binding": recheck_call_binding,
                                              "audit_binding": recheck_audit_binding}
        result["revision_recheck_validation"] = recheck_audit
        if mechanical_revision.get("status") == "PASS" and recheck_audit.get("status") == "PASS":
            final = recovery._assemble_source(
                root_id, packet, parent, revised, recheck, recheck_binding, revised_binding,
                warnings, 1, source_driver)
            source_binding = write(root_dir, "SOURCE_FINAL.json", final, artifacts)
            result.update(status="PASS", source_binding=source_binding, revision_count=1,
                          parent_prefix_preserved=True, warnings=recheck.get("warnings", []))
        else:
            result.update(status="FAILED_REVIEW_OR_MECHANICAL", revision_count=1,
                          failure_reason="v2 single revision did not pass mechanical and review checks")
    except Exception as exc:
        result.update(status="FAILED_CONTINUATION", failure_reason=f"{type(exc).__name__}: {exc}",
                      error_type=type(exc).__name__)
    result["artifacts"] = artifacts
    recovery.write_immutable(result_path, result)
    return result


def publish() -> dict[str, Any]:
    roots = source_driver.validate_preparation()
    selected = []
    failures = []
    outcomes = []
    final_outcomes = []
    for row in roots:
        rid = row["root_id"]
        original_authoring = BASE / "source_authoring" / rid / "OUTCOME.json"
        original = BASE / "source_recovery" / rid / "ROOT_RECOVERY.json"
        prior = OLD_CONT / rid / "CONTINUATION_OUTCOME.json"
        v2 = CONT / rid / "CONTINUATION_OUTCOME.json"
        if not original_authoring.exists():
            raise RuntimeError(f"Missing original authoring outcome: {rid}")
        authoring_outcome = recovery.read(original_authoring)
        recovery_outcome = recovery.read(original) if original.exists() else None
        prior_outcome = recovery.read(prior) if prior.exists() else None
        outcome = recovery.read(v2) if v2.exists() else None
        if outcome is not None:
            for binding in outcome.get("artifacts", []):
                if recovery.bind(binding["path"]) != binding:
                    raise RuntimeError(f"v2 artifact binding drift for {rid}: {binding['path']}")
            outcomes.append(recovery.bind(v2))
        resolved_binding = None
        resolved_status = None
        if recovery_outcome and recovery_outcome.get("status") in ("PASS", "PASS_ORIGINAL_SOURCE_REUSED"):
            resolved_binding = recovery_outcome.get("source_binding")
            resolved_status = recovery_outcome.get("status")
        elif outcome and outcome.get("status") == "PASS":
            resolved_binding = outcome.get("source_binding")
            resolved_status = "PASS_CONTINUATION_V2"
        if resolved_binding:
            if not recovery._binding_matches(resolved_binding):
                raise RuntimeError(f"Passing source binding drift for {rid}")
            selected.append({"root_id": rid, **resolved_binding})
        else:
            failures.append({"root_id": rid,
                "original_authoring_outcome_binding": recovery.bind(original_authoring),
                "original_recovery_outcome_binding": recovery.bind(original) if original.exists() else None,
                "prior_continuation_binding": recovery.bind(prior) if prior.exists() else None,
                "v2_continuation_binding": recovery.bind(v2) if v2.exists() else None,
                "original_authoring_status": authoring_outcome.get("status"),
                "original_recovery_status": recovery_outcome.get("status") if recovery_outcome else None,
                "prior_continuation_status": prior_outcome.get("status") if prior_outcome else None,
                "v2_continuation_status": outcome.get("status") if outcome else None,
                "failure_reason": ((outcome or {}).get("failure_reason")
                    or (recovery_outcome or {}).get("failure_reason")
                    or authoring_outcome.get("failure_reason"))})
        final_record = {"schema_version": "fgr20_source_root_resolution_v2", "root_id": rid,
                        "status": "PASS" if resolved_binding else "FAILED",
                        "resolved_source_binding": resolved_binding,
                        "resolved_from": resolved_status,
                        "original_authoring_outcome_binding": recovery.bind(original_authoring),
                        "original_recovery_outcome_binding": recovery.bind(original) if original.exists() else None,
                        "prior_continuation_binding": recovery.bind(prior) if prior.exists() else None,
                        "v2_continuation_binding": recovery.bind(v2) if v2.exists() else None}
        final_path = BASE / "source_recovery_continuations_v2" / "ROOT_RESOLUTIONS" / (rid + ".json")
        final_outcomes.append(recovery.write_immutable(final_path, final_record))
    selection = {"schema_version": "closed_loop_source_selection_v1", "split": "train", "roots": selected}
    sel_path = BASE / "SOURCE_SELECTION_RECOVERY_V2.json"
    sel_binding = recovery.write_immutable(sel_path, selection)
    fail = {"schema_version": "fgr20_source_failures_recovery_v2", "planned_roots": 16,
            "passed_roots": len(selected), "failed_roots": failures, "substitutions": [],
            "prior_recovery_preserved": True, "v2_outcomes": outcomes}
    fail_path = BASE / "SOURCE_FAILURES_RECOVERY_V2.json"
    fail_binding = recovery.write_immutable(fail_path, fail)
    freeze = {"schema_version": "fgr20_source_freeze_v1", "status": "PASS" if len(selected) == 16 else "PARTIAL",
              "planned_root_count": 16, "passed_root_count": len(selected),
              "failed_root_count": 16 - len(selected), "source_selection_binding": sel_binding,
              "failures_binding": fail_binding, "parent_selection_binding": recovery.bind(BASE / "PARENT_SOURCE_SELECTION.json"),
              "root_outcomes": final_outcomes,
              "continuation_v2_outcomes": outcomes, "training": False, "old_sources_modified": False,
              "original_four_rerun": False, "same_shared_source_for_paired_F_G": True,
              "source_recovery_v2": True, "prior_artifacts_preserved": True}
    freeze_path = BASE / "SOURCE_FREEZE_RECOVERY_V2.json"
    freeze_binding = recovery.write_immutable(freeze_path, freeze)
    return {"status": freeze["status"], "passed": len(selected), "failed": 16-len(selected),
            "selection_binding": sel_binding, "failures_binding": fail_binding,
            "freeze_binding": freeze_binding}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--continue-roots", nargs="+")
    group.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        print(json.dumps(publish(), ensure_ascii=False, indent=2))
        return 0
    roots = [f"cc-train-episode-{int(x)}" for x in args.continue_roots]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        outcomes = list(pool.map(continue_one, roots))
    print(json.dumps({"schema_version": "fgr20_source_recovery_continuation_v2_batch_v1",
                      "outcomes": outcomes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
