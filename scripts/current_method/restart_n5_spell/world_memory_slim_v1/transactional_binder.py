"""Recoverable two-phase logical transaction Binder v1."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


TRANSACTIONAL_BINDER_VERSION = "transactional_binder_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def transaction_id(run_id: str, scene_plan_sha: str, proposal_fingerprints: Sequence[str]) -> str:
    material = _canonical({"run_id": run_id, "scene_plan_sha": scene_plan_sha,
                           "proposal_fingerprints": list(proposal_fingerprints)})
    return "TX:" + hashlib.sha256(material).hexdigest()[:24]


def build_allocation_plan(
    *, root_id: str, round_index: int, scene_index: int,
    compiled_proposals: Sequence[Mapping[str, Any]], high_water_marks: Mapping[str, int],
) -> dict[str, Any]:
    marks = {key: int(value) for key, value in high_water_marks.items()}
    for key in ("memory", "entity", "event", "relation", "thread", "handoff", "global_time"):
        marks.setdefault(key, 0)
    allocations: dict[str, dict[str, Any]] = {}
    temporary_ref_allocations: dict[str, str] = {}
    event_sequence = 0
    for proposal in sorted(compiled_proposals, key=lambda row: str(row["proposal_ref"])):
        ref = str(proposal["proposal_ref"])
        operation = str(proposal["operation"])
        marks["memory"] += 1
        marks["global_time"] += 1
        event_sequence += 1
        allocation: dict[str, Any] = {
            "memory_ref": f"{root_id}:M{marks['memory']:06d}",
            "event_sequence_index": event_sequence,
            "global_time_index": marks["global_time"],
            "entity_ref": None, "event_ref": None, "relation_ref": None,
            "thread_ref": None, "handoff_ref": None,
        }
        kind = {"add_person": "entity", "add_event": "event", "add_relation": "relation"}.get(operation)
        if kind:
            marks[kind] += 1
            prefix = {"entity": "E", "event": "EV", "relation": "REL"}[kind]
            allocation[f"{kind}_ref"] = f"{root_id}:{prefix}{marks[kind]:06d}"
            if kind == "entity":
                for temporary_ref in proposal.get("temporary_refs") or []:
                    temporary_ref_allocations[str(temporary_ref)] = allocation["entity_ref"]
        if operation == "open_thread":
            marks["thread"] += 1
            allocation["thread_ref"] = f"{root_id}:thread_{marks['thread']:04d}"
        if operation == "add_role_handoff":
            marks["handoff"] += 1
            allocation["handoff_ref"] = f"{root_id}:H{marks['handoff']:04d}"
        allocations[ref] = allocation
    return {
        "binder_version": TRANSACTIONAL_BINDER_VERSION,
        "round_index": round_index, "scene_index": scene_index,
        "proposal_allocations": allocations,
        "temporary_ref_allocations": temporary_ref_allocations,
        "resulting_high_water_marks": marks,
    }


def validate_scene_local_evidence_refs(
    *, scene_ref: str, dialogue_turns: Sequence[Mapping[str, Any]],
    accepted_evidence_by_proposal: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Validate frozen dialogue refs; Binder never renumbers dialogue turns."""
    turn_by_ref = {str(row["turn_ref"]): row for row in dialogue_turns}
    evidence: dict[str, list[str]] = {}
    evidence_sha: dict[str, list[str]] = {}
    for proposal_ref, refs in accepted_evidence_by_proposal.items():
        if any(ref not in turn_by_ref for ref in refs):
            raise ValueError("binder_evidence_ref_missing")
        if any(not str(ref).startswith(scene_ref + ":T") for ref in refs):
            raise ValueError("binder_evidence_ref_not_current_scene")
        evidence[proposal_ref] = list(refs)
        evidence_sha[proposal_ref] = [hashlib.sha256(str(turn_by_ref[ref]["text"]).encode("utf-8")).hexdigest()
                                      for ref in refs]
    return {"scene_ref": scene_ref, "dialogue_turn_refs_frozen": True,
            "evidence_final_turn_refs": evidence, "evidence_text_sha256": evidence_sha}


def commit_scene_transaction(
    *, run_root: Path, run_id: str, scene_plan: Mapping[str, Any],
    proposal_fingerprints: Sequence[str], allocation_plan: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish staged files idempotently using PREPARED/COMMITTED journal records."""
    run_root = Path(run_root)
    scene_sha = hashlib.sha256(_canonical(scene_plan)).hexdigest()
    tx_id = transaction_id(run_id, scene_sha, proposal_fingerprints)
    journal = run_root / "run_commit_journal.jsonl"
    manifest_path = run_root / "scene_commit_manifest.json"
    staging = run_root / "staging" / tx_id.replace(":", "_")
    existing = _read_journal(journal)
    if any(row.get("transaction_id") == tx_id and row.get("status") == "COMMITTED" for row in existing):
        return {"transaction_id": tx_id, "status": "COMMITTED", "idempotent_resume": True,
                "allocation_plan": deepcopy(dict(allocation_plan))}
    staging.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, str]] = []
    for relative, value in sorted(artifacts.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("binder_artifact_path_invalid")
        staged = staging / relative_path
        staged.parent.mkdir(parents=True, exist_ok=True)
        _write_fsynced(staged, _canonical(value) + b"\n")
        targets.append({"staged": str(staged.relative_to(run_root)), "target": str(relative_path)})
    prepared = {"transaction_id": tx_id, "status": "PREPARED",
                "scene_plan_sha": scene_sha, "proposal_fingerprints": list(proposal_fingerprints),
                "allocation_plan": deepcopy(dict(allocation_plan)), "targets": targets}
    _append_journal(journal, prepared)
    for row in targets:
        staged = run_root / row["staged"]
        target = run_root / row["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)
        _fsync_directory(target.parent)
    committed = {"transaction_id": tx_id, "status": "COMMITTED",
                 "allocation_plan": deepcopy(dict(allocation_plan))}
    _append_journal(journal, committed)
    manifest = _load_json(manifest_path, default={"binder_version": TRANSACTIONAL_BINDER_VERSION,
                                                  "committed_transactions": []})
    if tx_id not in manifest["committed_transactions"]:
        manifest["committed_transactions"].append(tx_id)
    _write_fsynced(manifest_path, _canonical(manifest) + b"\n")
    shutil.rmtree(staging, ignore_errors=True)
    return {"transaction_id": tx_id, "status": "COMMITTED", "idempotent_resume": False,
            "allocation_plan": deepcopy(dict(allocation_plan))}


def recover_prepared_transactions(run_root: Path) -> list[dict[str, Any]]:
    run_root = Path(run_root)
    journal = run_root / "run_commit_journal.jsonl"
    rows = _read_journal(journal)
    committed = {row["transaction_id"] for row in rows if row.get("status") == "COMMITTED"}
    recovered: list[dict[str, Any]] = []
    for prepared in [row for row in rows if row.get("status") == "PREPARED"]:
        tx_id = prepared["transaction_id"]
        if tx_id in committed:
            continue
        targets = prepared.get("targets") or []
        publishable = all((run_root / row["staged"]).exists() or (run_root / row["target"]).exists()
                          for row in targets)
        if not publishable:
            recovered.append({"transaction_id": tx_id, "status": "ROLLBACK_REQUIRED"})
            continue
        for row in targets:
            staged, target = run_root / row["staged"], run_root / row["target"]
            if staged.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
        _append_journal(journal, {"transaction_id": tx_id, "status": "COMMITTED",
                                  "allocation_plan": prepared["allocation_plan"],
                                  "recovered": True})
        recovered.append({"transaction_id": tx_id, "status": "COMMITTED"})
    return recovered


def _append_journal(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_fsynced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json(path: Path, *, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else deepcopy(default)
