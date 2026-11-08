"""Consume only the unused revision/recheck budget after three invalid reviews.

This engineering continuation never repairs review JSON or authors source facts.
The saved initial draft and the full invalid independent review are preserved;
one fresh author revision and one independent recheck use codex_calls.call.
Publication is separate and waits for all v2 and nonJSON outcomes to terminate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import codex_calls as cc  # noqa: E402
import question_driver  # noqa: E402,F401; installs exact startup-event adapter
import source_call_recovery as recovery  # noqa: E402
import source_driver as source  # noqa: E402

ROOT_NUMBERS = ("156136", "238359", "246750")
ROOT_IDS = tuple("cc-train-episode-" + number for number in ROOT_NUMBERS)
V2_ROOT_IDS = tuple("cc-train-episode-" + number
                    for number in ("24719", "25652", "45849", "75241"))
WORK = BASE / "source_recovery_nonjson_continuations"
OLD_RECOVERY = BASE / "source_recovery"
V2 = BASE / "source_recovery_continuations_v2"
PRIOR_CONT = BASE / "source_recovery_continuations"


def verify(binding: dict[str, Any]) -> None:
    if not recovery._binding_matches(binding):
        raise RuntimeError(f"Artifact binding drift: {binding}")


def write(directory: Path, name: str, value: Any,
          artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    result = recovery.write_immutable(directory / name, value)
    if result not in artifacts:
        artifacts.append(result)
    return result


def check_outcome(path: Path) -> dict[str, Any]:
    row = recovery.read(path)
    for item in row.get("artifacts", []):
        verify(item)
    if row.get("source_binding"):
        verify(row["source_binding"])
    return row


def active_cli_calls() -> list[dict[str, Any]]:
    """Count only registry entries whose actual PID/start token is still live."""
    active = []
    for path in sorted((BASE / "process_registry").glob("*.json")):
        row = recovery.read(path)
        pid = row.get("pid")
        if (row.get("active") is True and isinstance(pid, int)
                and cc.process_start(pid) == row.get("start_token")):
            active.append({"registry_binding": recovery.bind(path),
                           "pid": pid, "call_id": row.get("call_id")})
    return active


def other_source_workers() -> list[dict[str, Any]]:
    names = {"source_driver.py", "source_driver_recovery.py", "source_call_recovery.py",
             "source_recovery_continuation.py", "source_recovery_continuation_v2.py",
             Path(__file__).name}
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = [part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        except (OSError, UnicodeDecodeError):
            continue
        scripts = [arg for arg in args if Path(arg).name in names]
        if scripts and any(str(BASE) in arg for arg in scripts):
            found.append({"pid": int(proc.name), "arguments": args})
    return found


def invalid_review_evidence(root_id: str) -> dict[str, Any]:
    call_id = root_id.replace("-", "_") + "_source_recovery_independent_review_0"
    call_dir = BASE / "cli_calls" / call_id
    attempt = call_dir / "attempt_01"
    failure_path = call_dir / "SEMANTIC_FAILURE.json"
    failure = recovery.read(failure_path)
    terminal_path = attempt / "TERMINAL.json"
    terminal = recovery.read(terminal_path)
    start = recovery.read(attempt / "START.json")
    if (failure.get("semantic_attempt_consumed") is not True
            or not str(failure.get("error", "")).startswith("JSONDecodeError:")
            or failure.get("terminal") != recovery.bind(terminal_path)):
        raise RuntimeError("Prior review is not the bound consumed nonJSON case")
    if ((call_dir / "RESULT.json").exists() or terminal.get("returncode") != 0
            or terminal.get("stopped") is not False or terminal.get("timed_out") is not False):
        raise RuntimeError("Prior nonJSON review has an unexpected completion state")
    if start.get("fresh_session") is not True or start.get("tool_calls_permitted") is not False:
        raise RuntimeError("Prior review is not documented as a fresh no-tools session")
    for key, filename in (("request", "REQUEST.txt"), ("events", "EVENTS.jsonl"),
                          ("stderr", "STDERR.log"), ("raw_response", "LAST_MESSAGE.txt")):
        if terminal.get(key) != recovery.bind(attempt / filename):
            raise RuntimeError(f"Prior review {key} binding drift")
    if start.get("request") != terminal.get("request"):
        raise RuntimeError("Prior review START/request binding drift")
    classification = question_driver.classify_cli_events(attempt / "EVENTS.jsonl", terminal)
    raw = (attempt / "LAST_MESSAGE.txt").read_text()
    parsed_text = raw.strip()
    if parsed_text.startswith("```") and parsed_text.endswith("```"):
        parsed_text = "\n".join(parsed_text.splitlines()[1:-1])
    try:
        json.loads(parsed_text)
    except json.JSONDecodeError as exc:
        parse_failure = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise RuntimeError("Prior review is JSON; this limited continuation does not apply")
    return {"call_id": call_id, "raw_review_text_unmodified": raw,
            "semantic_failure_receipt": failure,
            "raw_review_binding": recovery.bind(attempt / "LAST_MESSAGE.txt"),
            "failure_binding": recovery.bind(failure_path),
            "terminal_binding": recovery.bind(terminal_path),
            "start_binding": recovery.bind(attempt / "START.json"),
            "event_classification": classification, "json_parse_failure": parse_failure,
            "initial_review_semantic_attempt_consumed": True,
            "valid_structured_review_available": False,
            "initial_review_will_not_be_retried_or_repaired": True}


def inputs_for(root_id: str, rows: dict[str, Any]) -> dict[str, Any]:
    if root_id not in ROOT_IDS or root_id not in rows:
        raise RuntimeError("Continuation is restricted to the three frozen nonJSON roots")
    saved = OLD_RECOVERY / root_id
    prior_path = saved / "ROOT_RECOVERY.json"
    prior = check_outcome(prior_path)
    expected_id = root_id.replace("-", "_") + "_source_recovery_independent_review_0"
    if (prior.get("root_id") != root_id or prior.get("status") != "FAILED_CALL_OR_ENGINEERING"
            or expected_id not in str(prior.get("failure_reason", ""))):
        raise RuntimeError("Prior source recovery is not the expected invalid-initial-review case")
    for key in ("original_outcome_binding", "material_packet_binding"):
        verify(prior[key])
    packet, parent, natives, embedded = recovery._source_prompt_inputs(source, rows[root_id])
    draft = recovery.read(saved / "DRAFT_0.json")
    mechanical = recovery.read(saved / "MECHANICAL_0.json")
    if mechanical.get("status") != "PASS" or source.validate_draft(draft, packet, parent, natives) != mechanical:
        raise RuntimeError("Saved initial draft no longer matches its mechanical PASS")
    # No prior author revision or recheck may be renamed into a new budget.
    prefix = root_id.replace("-", "_")
    allowed = {prefix + "_source_draft", expected_id,
               prefix + "_source_nonjson_continuation_author_revision",
               prefix + "_source_nonjson_continuation_revision_recheck"}
    unexpected = [path.name for path in (BASE / "cli_calls").glob(prefix + "_source*")
                  if path.name not in allowed]
    if unexpected:
        raise RuntimeError(f"Unexpected prior source calls consume/obscure revision budget: {unexpected}")
    for parent_dir in (BASE / "source_authoring" / root_id, saved,
                       PRIOR_CONT / root_id, V2 / root_id):
        if parent_dir.exists() and any("REVISION" in path.name.upper() or path.name in
                                      ("AUTHOR_1.json", "DRAFT_1.json", "REVIEW_1.json")
                                      for path in parent_dir.iterdir()):
            raise RuntimeError(f"Prior source revision artifacts already exist: {parent_dir}")
    evidence = invalid_review_evidence(root_id)
    return {"packet": packet, "parent": parent, "natives": natives, "embedded": embedded,
            "draft": draft, "mechanical": mechanical, "prior": prior,
            "prior_binding": recovery.bind(prior_path), "invalid_review": evidence,
            "draft_binding": recovery.bind(saved / "DRAFT_0.json"),
            "mechanical_binding": recovery.bind(saved / "MECHANICAL_0.json")}


def verify_receipt(receipt: dict[str, Any], forbidden_threads: set[str]) -> str:
    backend = receipt.get("backend", {})
    required = ("thread_id", "model", "model_provider", "effort", "cli_version")
    if (receipt.get("response_received") is not True or backend.get("model_override") is not None
            or not all(backend.get(key) for key in required)):
        raise RuntimeError("Actual fresh default-backend receipt is incomplete or overridden")
    thread = backend["thread_id"]
    if thread in forbidden_threads:
        raise RuntimeError("Author/reviewer thread independence failed")
    for key in ("request", "raw_response", "parsed_response", "terminal_binding",
                "event_classification_binding"):
        verify(receipt[key])
    return thread


def call_once(root_id: str, role: str, prompt: str, directory: Path,
              artifacts: list[dict[str, Any]], attempted: list[str]):
    call_id = root_id.replace("-", "_") + "_source_nonjson_continuation_" + role
    call_dir = BASE / "cli_calls" / call_id
    intent = {"call_id": call_id, "role": role, "root_id": root_id,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "semantic_attempt_budget": 1, "only_helper_transport_budget_permitted": True}
    write(directory, role.upper() + "_CALL_INTENT.json", intent, artifacts)
    if call_id not in attempted:
        attempted.append(call_id)
    if call_dir.exists() and not (directory / (role.upper() + "_EXISTING_CALL_CLASSIFICATION.json")).exists():
        write(directory, role.upper() + "_EXISTING_CALL_CLASSIFICATION.json", {
            "call_id": call_id, "reuse_only_existing_receipt_or_terminal": True,
            "has_result": (call_dir / "RESULT.json").exists(),
            "has_semantic_failure": (call_dir / "SEMANTIC_FAILURE.json").exists(),
            "start_bindings": [recovery.bind(path) for path in sorted(call_dir.glob("attempt_*/START.json"))],
            "interrupted_start_policy": "helper closes unresolved START without regeneration",
        }, artifacts)
    # cc.call reuses a completed receipt, closes interrupted START without replay,
    # and fails a nonJSON semantic response rather than regenerating it.
    while len(active_cli_calls()) >= 3:
        if cc.stop_generation():
            raise RuntimeError("Generation cutoff while waiting for a CLI concurrency slot")
        time.sleep(1)
    return call_id, cc.call(call_id, prompt)


def continue_one(root_id: str) -> dict[str, Any]:
    directory = WORK / root_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "CONTINUATION_OUTCOME.json"
    if path.exists():
        return check_outcome(path)
    artifacts: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": "fgr20_source_nonjson_continuation_v1", "root_id": root_id,
        "status": "FAILED_CONTINUATION", "initial_draft_reused_without_model_call": True,
        "initial_review_semantic_attempt_consumed": True,
        "initial_review_not_retried_or_repaired": True,
        "max_additional_revisions": 1, "max_additional_rechecks": 1,
        "call_ids_attempted": [], "artifacts": artifacts,
    }
    lock_path = OLD_RECOVERY / ".locks" / (root_id + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if cc.stop_generation():
                raise RuntimeError("Generation cutoff before unused revision budget could start")
            rows = {row["root_id"]: row for row in source.validate_preparation()}
            data = inputs_for(root_id, rows)
            result["prior_recovery_outcome_binding"] = data["prior_binding"]
            write(directory, "INITIAL_DRAFT_REFERENCE.json", {
                "draft_binding": data["draft_binding"], "mechanical_binding": data["mechanical_binding"],
                "prior_recovery_outcome_binding": data["prior_binding"],
                "initial_author_calls_consumed": 1, "initial_review_calls_consumed": 1,
                "prior_author_revision_calls_consumed": 0, "prior_revision_recheck_calls_consumed": 0,
            }, artifacts)
            write(directory, "INVALID_INITIAL_REVIEW_EVIDENCE.json", data["invalid_review"], artifacts)
            correction = {"previous_normalized_draft": data["draft"],
                          "mechanical_validation": data["mechanical"],
                          "invalid_independent_review": data["invalid_review"],
                          "review_contract_validation": {
                              "status": "FAILED_SERIALIZATION", "semantic_verdict_available": False,
                              "initial_review_attempt_consumed": True}}
            prompt = (cc.prefix("neutral shared-source author performing the one allowed revision")
                      + source.AUTHOR_RULES
                      + "\nThis is the only author revision. The prior independent review returned invalid JSON; "
                        "its full raw text and failure receipt are preserved below without managed semantic interpretation. "
                        "Apply the same neutral source contract to this single revision; no additional author attempt is available.\nCOMPLETE_INPUTS:\n"
                      + json.dumps(data["embedded"], ensure_ascii=False, separators=(",", ":"))
                      + "\nPREVIOUS_DRAFT_AND_FAILED_REVIEW_EVIDENCE:\n"
                      + json.dumps(correction, ensure_ascii=False, separators=(",", ":")))
            author_id, (raw, author_call) = call_once(root_id, "author_revision", prompt, directory,
                                                    artifacts, result["call_ids_attempted"])
            prior_author_call = recovery.read(OLD_RECOVERY / root_id / "AUTHOR_CALL_0_RECOVERY.json")
            initial_thread = recovery.read(prior_author_call["call_evidence"]["terminal"]["path"])["backend"]["thread_id"]
            initial_review_thread = recovery.read(data["invalid_review"]["terminal_binding"]["path"])["backend"]["thread_id"]
            author_thread = verify_receipt(author_call, {initial_thread, initial_review_thread})
            raw_binding = write(directory, "AUTHOR_REVISION_RESPONSE.json", raw, artifacts)
            call_binding = write(directory, "AUTHOR_REVISION_CALL.json", author_call, artifacts)
            revised, warnings = source.normalize(raw, data["packet"])
            draft_binding = write(directory, "DRAFT_REVISION.json", revised, artifacts)
            warning_binding = write(directory, "METADATA_NORMALIZATION_REVISION.json", warnings, artifacts)
            mechanical = source.validate_draft(revised, data["packet"], data["parent"], data["natives"])
            mechanical_binding = write(directory, "MECHANICAL_REVISION.json", mechanical, artifacts)
            result["revision_lineage"] = {"call_id": author_id, "fresh_thread_id": author_thread,
                "author_response_binding": raw_binding, "call_binding": call_binding,
                "draft_binding": draft_binding, "normalization_warnings_binding": warning_binding,
                "mechanical_binding": mechanical_binding, "revision_count": 1,
                "actual_backend": author_call["backend"]}
            result["revision_count"] = 1
            if cc.stop_generation():
                raise RuntimeError("Generation cutoff after revision; no unreviewed source admitted")
            review_input = {"complete_inputs": data["embedded"],
                            "complete_normalized_draft": revised, "mechanical_validation": mechanical}
            review_prompt = (cc.prefix("independent shared-source reviewer") + source.REVIEW_RULES
                             + "\nCOMPLETE_REVIEW_INPUTS:\n"
                             + json.dumps(review_input, ensure_ascii=False, separators=(",", ":")))
            review_id, (review, review_call) = call_once(root_id, "revision_recheck", review_prompt,
                directory, artifacts, result["call_ids_attempted"])
            reviewer_thread = verify_receipt(review_call, {initial_thread, initial_review_thread, author_thread})
            review_binding = write(directory, "REVISION_RECHECK.json", review, artifacts)
            review_call_binding = write(directory, "REVISION_RECHECK_CALL.json", review_call, artifacts)
            audit = source.validate_review(review, data["packet"], data["parent"], revised, data["natives"])
            audit_binding = write(directory, "REVISION_RECHECK_VALIDATION.json", audit, artifacts)
            result["revision_recheck_lineage"] = {"call_id": review_id, "fresh_thread_id": reviewer_thread,
                "review_binding": review_binding, "call_binding": review_call_binding,
                "audit_binding": audit_binding, "actual_backend": review_call["backend"]}
            result["mechanical_revision"] = mechanical
            result["revision_recheck_validation"] = audit
            if mechanical.get("status") == "PASS" and audit.get("status") == "PASS":
                final = recovery._assemble_source(root_id, data["packet"], data["parent"], revised,
                    review, review_binding, draft_binding, warnings, 1, source)
                result.update(status="PASS", source_binding=write(directory, "SOURCE_FINAL.json", final, artifacts),
                              final_draft_binding=draft_binding, final_review_binding=review_binding,
                              parent_prefix_preserved=True, warnings=review.get("warnings", []))
            else:
                result.update(status="FAILED_REVIEW_OR_MECHANICAL",
                              failure_reason="The sole revision/recheck did not pass both mechanical and independent review checks")
        except Exception as exc:
            result.update(status="FAILED_CONTINUATION", failure_reason=f"{type(exc).__name__}: {exc}",
                          error_type=type(exc).__name__)
        result["artifacts"] = artifacts
        recovery.write_immutable(path, result)
        cc.progress("SOURCE_NONJSON_CONTINUATION_FINISHED", root_id=root_id, status=result["status"],
                    call_ids_attempted=result["call_ids_attempted"])
        return result


def publish() -> dict[str, Any]:
    if active_cli_calls() or other_source_workers():
        raise RuntimeError("Source/CLI processes are still live; V3 publication waits for terminal outcomes")
    for root_id in V2_ROOT_IDS:
        if not (V2 / root_id / "CONTINUATION_OUTCOME.json").is_file():
            raise RuntimeError(f"Missing terminal v2 outcome: {root_id}")
    for root_id in ROOT_IDS:
        if not (WORK / root_id / "CONTINUATION_OUTCOME.json").is_file():
            raise RuntimeError(f"Missing terminal nonJSON continuation outcome: {root_id}")
    rows = source.validate_preparation()
    selected, failures, resolutions = [], [], []
    original_bindings, v2_bindings, continuation_bindings = [], [], []
    for row in rows:
        rid = row["root_id"]
        paths = {"original_authoring": BASE / "source_authoring" / rid / "OUTCOME.json",
                 "original_recovery": OLD_RECOVERY / rid / "ROOT_RECOVERY.json",
                 "prior_continuation": PRIOR_CONT / rid / "CONTINUATION_OUTCOME.json",
                 "v2_continuation": V2 / rid / "CONTINUATION_OUTCOME.json",
                 "nonjson_continuation": WORK / rid / "CONTINUATION_OUTCOME.json"}
        if not paths["original_authoring"].is_file():
            raise RuntimeError(f"Missing original source outcome: {rid}")
        available = {key: check_outcome(path) for key, path in paths.items() if path.is_file()}
        histories = {key + "_binding": recovery.bind(paths[key]) for key in available}
        original_bindings.extend(histories[key + "_binding"] for key in
                                 ("original_authoring", "original_recovery") if key in available)
        if "v2_continuation" in available:
            v2_bindings.append(histories["v2_continuation_binding"])
        continuation_bindings.extend(histories[key + "_binding"] for key in
                                     ("prior_continuation", "nonjson_continuation") if key in available)
        passes = [(key, value) for key, value in available.items()
                  if value.get("status") in ("PASS", "PASS_ORIGINAL_SOURCE_REUSED")]
        if len(passes) > 1:
            unique = {json.dumps(value.get("source_binding"), sort_keys=True) for _, value in passes}
            if len(unique) != 1:
                raise RuntimeError(f"Multiple distinct passing source versions: {rid}")
        resolution = {"schema_version": "fgr20_source_root_resolution_v3", "root_id": rid,
                      "status": "FAILED", **histories,
                      "observed_statuses": {key: value.get("status") for key, value in available.items()}}
        if passes:
            key, value = passes[-1]
            source_binding = value.get("source_binding")
            verify(source_binding)
            final = recovery.read(source_binding["path"])
            selected_row = {"root_id": rid, **source_binding}
            # This checks existing independent PASS lineage; it creates no verdict.
            records = [(rid, str(paths[name]), outcome) for name, outcome in available.items()]
            question_driver._validate_extended_source(rid, selected_row, final, records)
            selected.append(selected_row)
            resolution.update(status="PASS", source_binding=source_binding,
                              resolved_source_binding=source_binding, resolved_from=key,
                              resolved_outcome_binding=histories[key + "_binding"],
                              final_review_binding=final["closed_loop_provenance"]["review_binding"])
        else:
            latest = next((available[key] for key in ("nonjson_continuation", "v2_continuation",
                          "prior_continuation", "original_recovery", "original_authoring") if key in available))
            failures.append({"root_id": rid, **histories,
                "observed_statuses": resolution["observed_statuses"],
                "failure_reason": latest.get("failure_reason", "No independently admitted R03 source")})
        resolutions.append(recovery.write_immutable(WORK / "ROOT_RESOLUTIONS" / (rid + ".json"), resolution))
    selection = {"schema_version": "closed_loop_source_selection_v1", "split": "train", "roots": selected}
    selection_binding = recovery.write_immutable(BASE / "SOURCE_SELECTION_RECOVERY_V3.json", selection)
    failures_binding = recovery.write_immutable(BASE / "SOURCE_FAILURES_RECOVERY_V3.json", {
        "schema_version": "fgr20_source_failures_recovery_v3", "planned_roots": 16,
        "passed_roots": len(selected), "failed_roots": failures, "substitutions": [],
        "prior_outcomes_and_raw_failures_preserved": True,
        "native_reservations_remain_consumed": True})
    freeze = {"schema_version": "fgr20_source_freeze_v1", "status": "PASS" if len(selected) == 16 else "PARTIAL",
        "planned_root_count": 16, "passed_root_count": len(selected), "failed_root_count": 16 - len(selected),
        "source_selection_binding": selection_binding, "failures_binding": failures_binding,
        "parent_selection_binding": recovery.bind(BASE / "PARENT_SOURCE_SELECTION.json"),
        "root_outcomes": resolutions, "original_root_outcomes": original_bindings,
        "continuation_v2_outcomes": v2_bindings, "continuation_outcomes": continuation_bindings,
        "publisher_binding": recovery.bind(__file__), "source_recovery_version": 3,
        "training": False, "old_sources_modified": False, "original_four_rerun": False,
        "same_shared_source_for_paired_F_G": True, "prior_artifacts_preserved": True,
        "initial_nonjson_reviews_consumed_not_retried": True,
        "nonjson_root_max_additional_revisions": 1, "nonjson_root_max_additional_rechecks": 1}
    freeze_binding = recovery.write_immutable(BASE / "SOURCE_FREEZE_RECOVERY_V3.json", freeze)
    return {"status": freeze["status"], "passed": len(selected), "failed": 16 - len(selected),
            "selection_binding": selection_binding, "freeze_binding": freeze_binding,
            "failures_binding": failures_binding}


def self_check() -> dict[str, Any]:
    rows = {row["root_id"]: row for row in source.validate_preparation()}
    counts = []
    for rid in ROOT_IDS:
        data = inputs_for(rid, rows)
        counts.append({"root_id": rid, "draft_binding": data["draft_binding"],
                       "mechanical_status": data["mechanical"]["status"],
                       "invalid_review_binding": data["invalid_review"]["raw_review_binding"],
                       "initial_author_calls_consumed": 1, "initial_review_calls_consumed": 1,
                       "prior_author_revision_calls_consumed": 0, "prior_revision_recheck_calls_consumed": 0})
    return {"status": "PASS", "model_calls": 0, "files_written": 0,
            "checks": "Frozen inputs, existing artifact bindings, true nonJSON initial review, unused single revision budget",
            "roots": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--continue-roots", nargs="+", choices=ROOT_NUMBERS)
    group.add_argument("--publish", action="store_true")
    group.add_argument("--self-check", action="store_true")
    parser.add_argument("--max-workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if args.self_check:
        result = self_check()
    elif args.publish:
        result = publish()
    else:
        if len(set(args.continue_roots)) != len(args.continue_roots):
            raise RuntimeError("Duplicate continuation root arguments")
        active = active_cli_calls()
        if len(active) + min(args.max_workers, len(args.continue_roots)) > 3:
            raise RuntimeError("Requested workers plus actual live CLI calls would exceed three")
        ids = ["cc-train-episode-" + number for number in args.continue_roots]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            outcomes = list(pool.map(continue_one, ids))
        result = {"schema_version": "fgr20_source_nonjson_continuation_batch_v1",
                  "outcomes": outcomes, "max_workers": args.max_workers}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
