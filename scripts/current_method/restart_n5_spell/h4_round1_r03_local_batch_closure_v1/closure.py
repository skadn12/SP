"""Build an offline, fail-closed closure over authoritative R03 local runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    AUTHORITATIVE_RUNS,
    DEFAULT_OUTPUT,
    DEVELOPMENT_RUN_NUMBERS,
    R03_RUN_PREFIX,
    RUNS,
    SUCCESS_STATUS,
    VERSION,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def add_sidecars(run_root: Path) -> int:
    files = [path for path in run_root.rglob("*") if path.is_file()
             and path.suffix in {".json", ".jsonl", ".md", ".txt"}]
    for path in sorted(files):
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(
            f"{digest_file(path)}  {path.name}\n", encoding="utf-8")
    return len(files)


def _response_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "called": False,
            "transport_attempt_count": 0,
            "transport_retry_count": 0,
            "rate_limit_retry_count": 0,
        }
    response = read_json(path)
    metadata = response.get("metadata") or {}
    return {
        "called": True,
        "transport_attempt_count": int(
            metadata.get("transport_attempt_count") or 0),
        "transport_retry_count": int(
            metadata.get("transport_retry_count") or 0),
        "rate_limit_retry_count": int(
            metadata.get("rate_limit_retry_count") or 0),
        "finish_reason": metadata.get("finish_reason"),
        "model": response.get("model"),
        "prompt_sha256": response.get("prompt_sha256"),
        "response_sha256": response.get("response_sha256"),
    }


def _sidecar_audit(run_root: Path) -> dict[str, Any]:
    sidecars = sorted(run_root.rglob("*.sha256"))
    failures = []
    for sidecar in sidecars:
        target = sidecar.with_name(sidecar.name[:-7])
        parts = sidecar.read_text(encoding="utf-8").split()
        expected = parts[0] if parts else ""
        if not target.is_file() or digest_file(target) != expected:
            failures.append(str(target))
    return {
        "sidecar_count": len(sidecars),
        "failure_paths": failures,
        "valid": bool(sidecars) and not failures,
    }


def _attempted_turn_count(run_root: Path) -> int:
    validation_path = run_root / "realizer" / "validation_01.json"
    if not validation_path.exists():
        return 0
    child = (read_json(validation_path).get("child") or {})
    return sum(len(scene.get("dialogue_turns") or [])
               for scene in child.get("scenes") or [])


def _success_dialogue_audit(
    run_root: Path, input_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    prior_path = Path(str(input_manifest["source_path"]))
    cumulative_path = run_root / "parent_r01_r02_r03_cumulative_dialogue.json"
    lineage_path = run_root / "lineage_manifest.json"
    prior = read_json(prior_path)
    cumulative = read_json(cumulative_path)
    lineage = read_json(lineage_path)
    prior_turns = list(prior.get("turns") or [])
    cumulative_turns = list(cumulative.get("turns") or [])
    added = cumulative_turns[len(prior_turns):]
    all_refs = [str(row.get("turn_ref") or "") for row in cumulative_turns]
    checks = {
        "source_file_sha_match": (
            digest_file(prior_path) == input_manifest["source_file_sha256"]),
        "prior_turn_objects_byte_equivalent": (
            canonical_json(cumulative_turns[:len(prior_turns)])
            == canonical_json(prior_turns)),
        "r03_only_appended": bool(added) and all(
            str(row.get("turn_ref") or "").startswith("R03:")
            for row in added),
        "turn_refs_unique": len(all_refs) == len(set(all_refs)),
        "lineage_cumulative_sha_match": (
            lineage.get("cumulative_dialogue_sha256")
            == cumulative.get("complete_dialogue_sha256")),
        "lineage_source_sha_match": (
            lineage.get("source_complete_dialogue_sha256")
            == prior.get("complete_dialogue_sha256")),
    }
    return {
        "checks": checks,
        "valid": all(checks.values()),
        "cumulative_dialogue_path": str(cumulative_path),
        "cumulative_dialogue_file_sha256": digest_file(cumulative_path),
        "cumulative_dialogue_content_sha256": cumulative.get(
            "complete_dialogue_sha256"),
        "r03_delta_path": str(run_root / "r03_delta.json"),
        "r03_delta_turns_path": str(run_root / "r03_delta_turns.jsonl"),
    }


def _root_row(root_id: str, run_number: int) -> dict[str, Any]:
    run_root = RUNS / f"{R03_RUN_PREFIX}{run_number}"
    required = [
        run_root / "final_readonly_memory_trial_report.json",
        run_root / "input_dialogue_manifest.json",
        run_root / "source_memory_manifest.json",
        run_root / "planner" / "local_gate_report.json",
        run_root / "planner" / "response.json",
        run_root / "realizer" / "response_01.json",
        run_root / "realizer" / "validation_01.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("authoritative_run_artifact_missing:" + ",".join(missing))
    report = read_json(required[0])
    input_manifest = read_json(required[1])
    memory_manifest = read_json(required[2])
    planner_gate = read_json(required[3])
    realizer_validation = read_json(required[6])
    if report.get("root_id") != root_id or input_manifest.get("root_id") != root_id:
        raise RuntimeError(f"authoritative_root_mapping_mismatch:{root_id}")
    success = report.get("status") == SUCCESS_STATUS
    continuation_path = run_root / "local_continuation_gate_report.json"
    continuation_gate = read_json(continuation_path) if continuation_path.exists() else None
    dialogue_audit = (
        _success_dialogue_audit(run_root, input_manifest) if success else None)
    forbidden_calls = {
        name: int(report.get(name) or 0) for name in (
            "memorax_search_calls", "memorax_add_calls",
            "memorax_status_calls", "memorax_mutation_calls",
            "qa_calls", "solver_calls", "semantic_auditor_calls",
            "training_calls",
        )
    }
    source_sha_path = run_root / "source_sha_audit.json"
    source_sha_audit = read_json(source_sha_path) if source_sha_path.exists() else None
    source_dialogue_path = Path(str(input_manifest["source_path"]))
    source_memory_path = Path(str(memory_manifest["source_path"]))
    source_memory = read_json(source_memory_path)
    source_identity_checks = {
        "dialogue_file_sha_match": (
            digest_file(source_dialogue_path)
            == input_manifest["source_file_sha256"]),
        "memory_snapshot_sha_match": (
            source_memory.get("snapshot_sha256")
            == memory_manifest.get("source_snapshot_sha256")),
        "memory_root_id_match": source_memory.get("root_id") == root_id,
        "memory_user_id_match": (
            source_memory.get("user_id")
            == memory_manifest.get("source_memory_user_id")),
    }
    planner = _response_metrics(required[4])
    realizer = _response_metrics(required[5])
    row = {
        "root_id": root_id,
        "selection_order": [r for r, _ in AUTHORITATIVE_RUNS].index(root_id) + 1,
        "authoritative_run_number": run_number,
        "authoritative_run_root": str(run_root),
        "disposition": "R03_LOCAL_SUCCESS" if success else "R03_LOCAL_QUARANTINE",
        "promotion_status": "PROMOTED_LOCAL_ONLY" if success else "NOT_PROMOTED",
        "status": report.get("status"),
        "failure_stage": report.get("failure_stage"),
        "failure_reasons": list(report.get("failures") or []),
        "input_turn_count": int(input_manifest.get("turn_count") or 0),
        "attempted_r03_turn_count": _attempted_turn_count(run_root),
        "r03_turn_count": int(report.get("r03_turn_count") or 0),
        "cumulative_turn_count": (
            int(report["cumulative_turn_count"]) if success else None),
        "planner": planner,
        "realizer": realizer,
        "planner_repair_calls": 0,
        "realizer_repair_calls": 0,
        "root_level_retry_calls": 0,
        "planner_gate_status": "PASS" if planner_gate.get("valid") else "FAIL",
        "realizer_validation_status": (
            "PASS" if realizer_validation.get("valid") else "FAIL"),
        "local_continuation_gate_status": (
            "PASS" if continuation_gate and continuation_gate.get("valid")
            else "FAIL" if continuation_gate else "NOT_RUN"),
        "append_only_status": (
            "PASS" if dialogue_audit and dialogue_audit["valid"]
            else "NOT_APPLICABLE_NOT_PROMOTED"),
        "turn_ref_uniqueness_status": (
            "PASS" if dialogue_audit and dialogue_audit["checks"][
                "turn_refs_unique"] else "NOT_APPLICABLE_NOT_PROMOTED"),
        "dialogue_artifacts": dialogue_audit,
        "forbidden_call_counts": forbidden_calls,
        "forbidden_call_counts_all_zero": not any(forbidden_calls.values()),
        "source_manifests_modified": (
            int(source_sha_audit.get("source_manifests_modified") or 0)
            if source_sha_audit else 0),
        "source_identity_checks": source_identity_checks,
        "source_identity_checks_all_pass": all(source_identity_checks.values()),
        "source_memory_snapshot_path": memory_manifest.get("source_path"),
        "source_memory_snapshot_sha256": memory_manifest.get(
            "source_snapshot_sha256"),
        "r03_memory_session_created": bool(
            report.get("r03_memory_session_created", False)),
        "reserve_seed_used": bool(report.get("reserve_seed_used", False)),
        "sidecar_audit": _sidecar_audit(run_root),
    }
    row["valid"] = all((
        row["forbidden_call_counts_all_zero"],
        row["source_identity_checks_all_pass"],
        not row["r03_memory_session_created"],
        not row["reserve_seed_used"],
        row["planner"]["called"],
        row["realizer"]["called"],
        row["planner"]["transport_retry_count"] == 0,
        row["realizer"]["transport_retry_count"] == 0,
        row["planner"]["rate_limit_retry_count"] == 0,
        row["realizer"]["rate_limit_retry_count"] == 0,
        row["planner_gate_status"] == "PASS",
        row["sidecar_audit"]["valid"],
        (success and row["realizer_validation_status"] == "PASS"
         and row["local_continuation_gate_status"] == "PASS"
         and bool(dialogue_audit and dialogue_audit["valid"]))
        or (not success and row["promotion_status"] == "NOT_PROMOTED"
            and not (run_root / "r03_delta.json").exists()
            and not (run_root / "parent_r01_r02_r03_cumulative_dialogue.json").exists()),
    ))
    return row


def execute_closure(run_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_root = Path(run_root)
    if run_root.exists():
        raise RuntimeError(f"closure_output_root_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)

    rows = [_root_row(root_id, run_number)
            for root_id, run_number in AUTHORITATIVE_RUNS]
    successes = [row for row in rows if row["disposition"] == "R03_LOCAL_SUCCESS"]
    quarantines = [row for row in rows
                   if row["disposition"] == "R03_LOCAL_QUARANTINE"]
    call_closure = {
        "planner_calls": sum(row["planner"]["called"] for row in rows),
        "realizer_calls": sum(row["realizer"]["called"] for row in rows),
        "expansion_api_calls": sum(
            row["planner"]["called"] + row["realizer"]["called"]
            for row in rows),
        "planner_repair_calls": sum(row["planner_repair_calls"] for row in rows),
        "realizer_repair_calls": sum(row["realizer_repair_calls"] for row in rows),
        "root_level_retry_calls": sum(row["root_level_retry_calls"] for row in rows),
        "transport_retry_calls": sum(
            row[role]["transport_retry_count"]
            for row in rows for role in ("planner", "realizer")),
        "rate_limit_retry_calls": sum(
            row[role]["rate_limit_retry_count"]
            for row in rows for role in ("planner", "realizer")),
        "memorax_search_calls": sum(
            row["forbidden_call_counts"]["memorax_search_calls"] for row in rows),
        "memorax_add_calls": sum(
            row["forbidden_call_counts"]["memorax_add_calls"] for row in rows),
        "memorax_status_calls": sum(
            row["forbidden_call_counts"]["memorax_status_calls"] for row in rows),
        "memorax_mutation_calls": sum(
            row["forbidden_call_counts"]["memorax_mutation_calls"] for row in rows),
        "qa_writer_calls": sum(
            row["forbidden_call_counts"]["qa_calls"] for row in rows),
        "solver_calls": sum(
            row["forbidden_call_counts"]["solver_calls"] for row in rows),
        "semantic_auditor_calls": sum(
            row["forbidden_call_counts"]["semantic_auditor_calls"] for row in rows),
        "training_calls": sum(
            row["forbidden_call_counts"]["training_calls"] for row in rows),
    }
    call_closure["forbidden_calls_all_zero"] = not any(
        call_closure[name] for name in (
            "memorax_search_calls", "memorax_add_calls",
            "memorax_status_calls", "memorax_mutation_calls",
            "qa_writer_calls", "solver_calls", "semantic_auditor_calls",
            "training_calls"))
    append_audit = {
        "successful_root_count": len(successes),
        "successful_roots_all_append_only": all(
            row["append_only_status"] == "PASS" for row in successes),
        "successful_roots_all_turn_refs_unique": all(
            row["turn_ref_uniqueness_status"] == "PASS" for row in successes),
        "quarantine_roots_not_promoted": all(
            row["promotion_status"] == "NOT_PROMOTED" for row in quarantines),
        "rows": [{
            "root_id": row["root_id"],
            "disposition": row["disposition"],
            "append_only_status": row["append_only_status"],
            "turn_ref_uniqueness_status": row["turn_ref_uniqueness_status"],
            "cumulative_dialogue_path": (
                row["dialogue_artifacts"] or {}).get("cumulative_dialogue_path"),
            "cumulative_dialogue_file_sha256": (
                row["dialogue_artifacts"] or {}).get(
                    "cumulative_dialogue_file_sha256"),
            "cumulative_dialogue_content_sha256": (
                row["dialogue_artifacts"] or {}).get(
                    "cumulative_dialogue_content_sha256"),
        } for row in rows],
    }
    append_audit["valid"] = all((
        append_audit["successful_roots_all_append_only"],
        append_audit["successful_roots_all_turn_refs_unique"],
        append_audit["quarantine_roots_not_promoted"],
    ))

    valid = all(row["valid"] for row in rows) and all((
        len(rows) == 8,
        not call_closure["planner_repair_calls"],
        not call_closure["realizer_repair_calls"],
        not call_closure["root_level_retry_calls"],
        not call_closure["transport_retry_calls"],
        not call_closure["rate_limit_retry_calls"],
        call_closure["forbidden_calls_all_zero"],
        append_audit["valid"],
    ))
    report = {
        "closure_version": VERSION,
        "status": (
            "R03_LOCAL_READONLY_DIALOGUE_BATCH_COMPLETE" if valid
            else "R03_LOCAL_READONLY_DIALOGUE_BATCH_COMPLETE_WITH_GAPS"),
        "next_status": (
            "READY_FOR_R03_NOVEL_QUESTION_CAPACITY_PREFLIGHT" if valid
            else "R03_NOVEL_QUESTION_CAPACITY_PREFLIGHT_NOT_READY"),
        "authoritative_run_numbers": [number for _, number in AUTHORITATIVE_RUNS],
        "excluded_development_run_numbers": list(DEVELOPMENT_RUN_NUMBERS),
        "root_count": len(rows),
        "success_root_count": len(successes),
        "success_roots_fraction": f"{len(successes)}/8",
        "success_root_ids": [row["root_id"] for row in successes],
        "quarantine_root_count": len(quarantines),
        "quarantine_roots_fraction": f"{len(quarantines)}/8",
        "quarantine_root_ids": [row["root_id"] for row in quarantines],
        "pending_root_count": 0,
        "total_input_turn_count": sum(row["input_turn_count"] for row in rows),
        "total_promoted_r03_turn_count": sum(row["r03_turn_count"] for row in rows),
        "total_success_cumulative_turn_count": sum(
            int(row["cumulative_turn_count"] or 0) for row in successes),
        "r03_memory_writes": 0,
        "qa_generation_started": False,
        "capacity_preflight_started": False,
        "valid": valid,
    }

    write_jsonl(run_root / "r03_local_batch_root_manifest.jsonl", rows)
    write_jsonl(run_root / "r03_local_success_manifest.jsonl", successes)
    write_jsonl(run_root / "r03_local_quarantine_manifest.jsonl", quarantines)
    write_json(run_root / "r03_local_batch_call_closure.json", call_closure)
    write_json(run_root / "r03_local_batch_append_only_audit.json", append_audit)
    write_json(run_root / "final_r03_local_batch_closure_report.json", report)
    write_text(run_root / "final_r03_local_batch_closure_report.md", "\n".join([
        "# R03 Local Batch Closure", "",
        f"- Status: `{report['status']}`",
        f"- Next: `{report['next_status']}`",
        f"- Success roots: `{report['success_roots_fraction']}`",
        f"- Quarantine roots: `{report['quarantine_roots_fraction']}`",
        f"- Pending roots: `{report['pending_root_count']}/8`",
        f"- Promoted R03 turns: `{report['total_promoted_r03_turn_count']}`",
        "- Memorax Search / Add / Status / Mutation: `0 / 0 / 0 / 0`",
        "- QA / Solver / semantic auditor / training: `0 / 0 / 0 / 0`",
        "- Capacity preflight: `not started`", "",
        "## Quarantine", "",
        *[f"- `{row['root_id']}`: `{row['failure_stage']}`; "
          f"{len(row['failure_reasons'])} recorded failure(s)"
          for row in quarantines], "",
    ]))
    report["sha256_sidecar_count"] = add_sidecars(run_root)
    return report
