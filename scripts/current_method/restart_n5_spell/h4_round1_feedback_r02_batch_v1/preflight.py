"""Offline-only preflight for the remaining nine feedback R02 roots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import VERSION
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.payloads import (
    memorax_user_id,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
COMPILER_ROOT = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
ROUND1_ROOT_MANIFEST = RUN_BASE / (
    "h4_round1_cumulative_continuation_fresh_mining_v1_live_47/"
    "round1_root_manifest.jsonl")
RECONSTRUCTION_ROOT = RUN_BASE / (
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_3")
COMPLETED_SMOKE_ROOT = RUN_BASE / (
    "h4_round1_feedback_r02_single_root_v1_live_6")
DEFAULT_OUTPUT = RUN_BASE / "h4_round1_feedback_r02_batch_v1_live_1"
FEEDBACK_NAMESPACE_PREFIX = "gxy-selfplay:round1-paired-feedback"
R02_SESSION_SUFFIX = ":memorax:R02-feedback"
EXPECTED_ROOTS = 10
COMPLETED_SMOKE_ROOT_ID = "cc-train-episode-254852"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing_file:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing_file:{path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"jsonl_row_not_object:{path}")
            rows.append(value)
    return rows


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        _sha(path) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False,
                                        sort_keys=True) + "\n"
                          for row in rows), encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        _sha(path) + "\n", encoding="utf-8")


def _verify_source(path: Path, expected_sha: str | None = None) -> str:
    if not path.is_file():
        raise RuntimeError(f"source_missing:{path}")
    actual = _sha(path)
    if expected_sha and actual != str(expected_sha):
        raise RuntimeError(f"source_sha_mismatch:{path}")
    return actual


def _sidecar_ok(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    return (path.is_file() and sidecar.is_file() and
            sidecar.read_text(encoding="utf-8").strip().split()[0] == _sha(path))


def _scrub_transport(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _scrub_transport(v)
            for k, v in value.items()
            if str(k) not in {"user_id", "request_id", "task_id",
                              "timestamp", "payload_sha256", "api_key",
                              "authorization"}
        }
    if isinstance(value, list):
        return [_scrub_transport(v) for v in value]
    return value


def _remap_intent(intent: Mapping[str, Any], *, root_id: str,
                  user_id: str, branch: str) -> dict[str, Any]:
    """Create a deterministic local bootstrap plan for a new namespace."""
    result = copy.deepcopy(dict(intent))
    payload = result.setdefault("payload", {})
    payload["user_id"] = user_id
    metadata = payload.setdefault("metadata", {})
    metadata["root_id"] = root_id
    result["root_id"] = root_id
    result["user_id"] = user_id
    result["task_id"] = None
    result["request_id"] = (
        f"{VERSION}-{branch}-{root_id}-r{int(result.get('round_index') or 0):02d}-"
        f"{_canonical_sha(_scrub_transport(intent))[:12]}"
    )
    result["payload_sha256"] = _canonical_sha(payload)
    result["reconstruction_branch"] = branch
    result["live_submission"] = False
    result["api_key_in_artifact"] = False
    return result


def _dialogue_facts(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    value = _load(path)
    root_id = str(row.get("root_id") or "")
    if str(value.get("root_id") or "") != root_id:
        raise RuntimeError(f"dialogue_root_mismatch:{root_id}")
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RuntimeError(f"dialogue_turns_missing:{root_id}")
    refs = [str(turn.get("turn_ref") or "") for turn in turns]
    if any(not ref for ref in refs) or len(refs) != len(set(refs)):
        raise RuntimeError(f"dialogue_refs_invalid:{root_id}")
    if any(re.search(r"R02", ref, re.I) for ref in refs):
        raise RuntimeError(f"r02_ref_in_r01_input:{root_id}")
    sessions = {str(turn.get("session_ref") or "") for turn in turns}
    if any("R02" in session.upper() for session in sessions):
        raise RuntimeError(f"r02_session_in_r01_input:{root_id}")
    expected_complete = str(row.get("complete_dialogue_sha256") or "")
    if str(value.get("complete_dialogue_sha256") or "") != expected_complete:
        raise RuntimeError(f"dialogue_complete_sha_mismatch:{root_id}")
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha(path),
        "parent_dialogue_sha256": str(value.get("parent_dialogue_sha256") or ""),
        "complete_dialogue_sha256": expected_complete,
        "root_id": root_id,
        "turn_count": len(turns),
        "r02_content_present": False,
        "round_refs": sorted({
            "parent" if ref.startswith(root_id + ":P") else ref.split(":", 1)[0]
            for ref in refs
        }),
    }


def _card_for_root(cards: Mapping[str, Mapping[str, Any]], root_id: str,
                   dialogue: Mapping[str, Any]) -> dict[str, Any]:
    card = cards.get(root_id)
    if card is None:
        raise RuntimeError(f"feedback_card_missing:{root_id}")
    target = card.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeError(f"feedback_card_target_missing:{root_id}")
    refs = set()
    value = _load(Path(str(dialogue["path"])))
    refs.update(str(row.get("turn_ref") or "") for row in value.get("turns") or [])
    anchors = [str(ref) for ref in target.get("anchor_refs") or []]
    if not anchors or any(anchor not in refs for anchor in anchors):
        raise RuntimeError(f"feedback_anchor_missing:{root_id}")
    forbidden = {"question", "options", "items", "gold", "answer",
                 "solver_output", "raw_rollouts", "auditor_explanation"}
    if forbidden & set(card) or forbidden & set(target):
        raise RuntimeError(f"feedback_card_forbidden_fields:{root_id}")
    text = json.dumps(card, ensure_ascii=False, sort_keys=True).casefold()
    if any(token in text for token in ("<box>", "raw k3", "solver output",
                                       "a/b/c/d", "i1/i2", "i3/i4")):
        raise RuntimeError(f"feedback_card_forbidden_text:{root_id}")
    return {
        "path": str((COMPILER_ROOT / "root_feedback_cards.jsonl").resolve()),
        "file_sha256": _sha(COMPILER_ROOT / "root_feedback_cards.jsonl"),
        "row_sha256": _canonical_sha(card),
        "root_id": root_id,
        "feedback_version": card.get("feedback_version"),
        "anchor_refs": anchors,
        "target": dict(target),
    }


def _bootstrap_evidence(root_id: str) -> dict[str, Any]:
    if root_id != COMPLETED_SMOKE_ROOT_ID:
        return {
            "status": "MISSING_LOCAL_EVIDENCE",
            "reason": "R01-only feedback namespace bootstrap has not been run for this root",
            "memorax_calls": 0,
            "parent_add_calls": 0,
            "r01_child_add_calls": 0,
        }
    report_path = COMPLETED_SMOKE_ROOT / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"
    intent_path = COMPLETED_SMOKE_ROOT / "memory/child_add_intent.json"
    completion_path = COMPLETED_SMOKE_ROOT / "memory/child_add_completion.json"
    visibility_path = COMPLETED_SMOKE_ROOT / "memory/child_visibility_audit.json"
    for path in (report_path, intent_path, completion_path, visibility_path):
        if not path.is_file():
            raise RuntimeError(f"completed_smoke_artifact_missing:{path}")
    report = _load(report_path)
    intent = _load(intent_path)
    completion = _load(completion_path)
    visibility = _load(visibility_path)
    if (report.get("status") != "completed" or report.get("root_id") != root_id
            or report.get("parent_add_calls") != 0
            or report.get("child_add_calls") != 1
            or report.get("parent_add_skipped") is not True
            or completion.get("final_status") != "success"
            or completion.get("verified_complete") is not True
            or visibility.get("valid") is not True):
        raise RuntimeError(f"completed_smoke_not_verified:{root_id}")
    return {
        "status": "COMPLETED_R02_FEEDBACK_SMOKE",
        "reason": "existing live R02-feedback smoke is the immutable completion authority",
        "run_root": str(COMPLETED_SMOKE_ROOT.resolve()),
        "report_sha256": _sha(report_path),
        "intent_sha256": _sha(intent_path),
        "completion_sha256": _sha(completion_path),
        "visibility_sha256": _sha(visibility_path),
        "memorax_calls": int(report.get("memorax_calls") or 0),
        "parent_add_calls": int(report.get("parent_add_calls") or 0),
        "r01_child_add_calls": 0,
        "r02_add_calls": int(report.get("child_add_calls") or 0),
    }


def run(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output_not_empty:{output_root}")
    selected = _load_jsonl(COMPILER_ROOT / "selected_10_roots.jsonl")
    cards_rows = _load_jsonl(COMPILER_ROOT / "root_feedback_cards.jsonl")
    fork = _load(COMPILER_ROOT / "r01_only_memory_fork_plan.json")
    compiler_report = _load(COMPILER_ROOT / "final_feedback_compiler_report.json")
    manifest_rows = _load_jsonl(ROUND1_ROOT_MANIFEST)
    manifest = {str(row.get("root_id")): row for row in manifest_rows}
    cards: dict[str, Mapping[str, Any]] = {}
    for card in cards_rows:
        root_id = str(card.get("root_id") or "")
        if not root_id or root_id in cards:
            raise RuntimeError(f"feedback_card_duplicate:{root_id}")
        cards[root_id] = card
    if (len(selected) != EXPECTED_ROOTS or len(fork.get("roots") or []) != EXPECTED_ROOTS
            or len(manifest) != 70 or compiler_report.get("candidate_count") != 2294):
        raise RuntimeError("frozen_root_or_compiler_count_invalid")
    selected_ids = [str(row.get("root_id") or "") for row in selected]
    if len(set(selected_ids)) != EXPECTED_ROOTS or set(selected_ids) != {
            str(row.get("root_id") or "") for row in fork["roots"]}:
        raise RuntimeError("selected_fork_root_set_mismatch")
    fork_by_root = {str(row["root_id"]): row for row in fork["roots"]}
    source_hashes = {
        str(path): _sha(path)
        for path in (COMPILER_ROOT / "selected_10_roots.jsonl",
                     COMPILER_ROOT / "root_feedback_cards.jsonl",
                     COMPILER_ROOT / "r01_only_memory_fork_plan.json",
                     COMPILER_ROOT / "feedback_prompt_token_budget.json",
                     COMPILER_ROOT / "final_feedback_compiler_report.json",
                     ROUND1_ROOT_MANIFEST)
    }
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    r02_rows: list[dict[str, Any]] = []
    for selection_index, selection in enumerate(selected):
        root_id = str(selection.get("root_id") or "")
        if root_id not in manifest:
            raise RuntimeError(f"root_manifest_missing:{root_id}")
        lineage = manifest[root_id]
        dialogue_path = Path(str(lineage.get("canonical_dialogue_path") or ""))
        dialogue = _dialogue_facts(dialogue_path, lineage)
        card = _card_for_root(cards, root_id, dialogue)
        fork_row = fork_by_root[root_id]
        branches = fork_row.get("branches") or {}
        feedback_branch = branches.get("feedback") or {}
        namespace = str(feedback_branch.get("namespace") or "")
        expected_user = memorax_user_id(root_id, namespace=namespace)
        if not namespace.startswith(FEEDBACK_NAMESPACE_PREFIX + ":"):
            raise RuntimeError(f"feedback_namespace_invalid:{root_id}")
        if expected_user != f"{namespace}:{root_id}":
            raise RuntimeError(f"feedback_user_id_invalid:{root_id}")
        expected_session = f"{root_id}{R02_SESSION_SUFFIX}"
        if feedback_branch.get("r02_session_id") != expected_session:
            raise RuntimeError(f"feedback_session_invalid:{root_id}")
        source = fork_row.get("r01_only_reconstruction") or {}
        parent_path = Path(str(source.get("parent_add_intent_path") or ""))
        r01_path = Path(str(source.get("r01_add_intent_path") or ""))
        r01_completion_path = Path(str(source.get("r01_add_completion_path") or ""))
        parent_completion_path = parent_path.with_name("parent_add_completion.json")
        parent_sha = _verify_source(parent_path, source.get("parent_add_intent_sha256"))
        r01_sha = _verify_source(r01_path, source.get("r01_add_intent_sha256"))
        parent_completion_sha = _verify_source(parent_completion_path)
        r01_completion_sha = _verify_source(
            r01_completion_path, source.get("r01_add_completion_sha256"))
        parent_intent = _load(parent_path)
        r01_intent = _load(r01_path)
        parent_completion = _load(parent_completion_path)
        r01_completion = _load(r01_completion_path)
        parent_round = parent_intent.get("round_index")
        r01_round = r01_intent.get("round_index")
        if parent_round is None or int(parent_round) != 0:
            raise RuntimeError(f"parent_round_invalid:{root_id}")
        if r01_round is None or int(r01_round) != 1:
            raise RuntimeError(f"r01_round_invalid:{root_id}")
        if parent_intent.get("source_dialogue_sha256") != dialogue["parent_dialogue_sha256"]:
            raise RuntimeError(f"parent_source_dialogue_mismatch:{root_id}")
        if r01_intent.get("source_dialogue_sha256") != dialogue["complete_dialogue_sha256"]:
            raise RuntimeError(f"r01_source_dialogue_mismatch:{root_id}")
        if r01_intent.get("session_id") != f"{root_id}:memorax:R01":
            raise RuntimeError(f"r01_session_invalid:{root_id}")
        if any(str(ref).startswith("R02") for ref in r01_intent.get("turn_refs") or []):
            raise RuntimeError(f"r02_ref_in_r01_plan:{root_id}")
        for label, completion in (("parent", parent_completion), ("r01", r01_completion)):
            if (completion.get("final_status") != "success"
                    or completion.get("verified_complete") is not True):
                raise RuntimeError(f"{label}_completion_not_verified:{root_id}")
        remapped_parent = _remap_intent(parent_intent, root_id=root_id,
                                        user_id=expected_user, branch="feedback")
        remapped_r01 = _remap_intent(r01_intent, root_id=root_id,
                                     user_id=expected_user, branch="feedback")
        bootstrap_status = _bootstrap_evidence(root_id)
        pending = root_id != COMPLETED_SMOKE_ROOT_ID
        bootstrap_row = {
            "root_id": root_id,
            "selection_index": selection_index,
            "branch": "feedback",
            "namespace": namespace,
            "user_id": expected_user,
            "parent_add_intent": remapped_parent,
            "r01_add_intent": remapped_r01,
            "source_parent_intent_sha256": parent_sha,
            "source_r01_intent_sha256": r01_sha,
            "source_parent_completion_sha256": parent_completion_sha,
            "source_r01_completion_sha256": r01_completion_sha,
            "source_dialogue_binding_verified": True,
            "source_parent_completion_verified": True,
            "source_r01_completion_verified": True,
            "status": "REQUIRED_BEFORE_R02" if pending else "ALREADY_SATISFIED_BY_SMOKE",
            "live_submission": False,
        }
        bootstrap_rows.append(bootstrap_row)
        r02_rows.append({
            "root_id": root_id,
            "selection_index": selection_index,
            "canonical_dialogue_path": str(dialogue_path.resolve()),
            "canonical_dialogue_sha256": dialogue["file_sha256"],
            "complete_dialogue_sha256": dialogue["complete_dialogue_sha256"],
            "feedback_card_path": card["path"],
            "feedback_card_file_sha256": card["file_sha256"],
            "feedback_card_row_sha256": card["row_sha256"],
            "feedback_card": card["target"],
            "namespace": namespace,
            "user_id": expected_user,
            "r01_session_id": f"{root_id}:memorax:R01",
            "r02_feedback_session_id": expected_session,
            "r01_bootstrap_required": pending,
            "execution_allowed": False if pending else True,
            "memorax_payload_contract": {
                "messages": "R02-feedback dialogue only",
                "window_context": "R02-feedback dialogue only",
                "feedback_card_in_payload": False,
                "feedback_card_in_metadata": False,
                "provenance_storage": "local reports and request ledger only",
            },
            "execution_block_reason": (
                "R01-only feedback namespace bootstrap evidence missing"
                if pending else None),
            "prohibited": ["old control R02", "QA", "Solver", "Test",
                           "semantic auditor"],
        })
        rows.append({
            "root_id": root_id,
            "selection_index": selection_index,
            "status": bootstrap_status["status"],
            "r01_bootstrap_required": pending,
            "r02_feedback_execution_allowed": not pending,
            "canonical_dialogue": dialogue,
            "feedback_card": card,
            "feedback_namespace": namespace,
            "feedback_user_id": expected_user,
            "r02_feedback_session_id": expected_session,
            "bootstrap_evidence": bootstrap_status,
        })
    pending_ids = [row["root_id"] for row in rows if row["r01_bootstrap_required"]]
    completed_ids = [row["root_id"] for row in rows if not row["r01_bootstrap_required"]]
    checks = {
        "selected_root_count_10": len(selected_ids) == 10,
        "selected_root_ids_unique": len(set(selected_ids)) == 10,
        "fork_plan_root_set_matches": set(selected_ids) == set(fork_by_root),
        "canonical_dialogue_sha_closed": all(
            row["canonical_dialogue"]["file_sha256"] == manifest[row["root_id"]]["canonical_dialogue_sha256"]
            for row in rows),
        "canonical_inputs_parent_r01_only": all(
            row["canonical_dialogue"]["r02_content_present"] is False
            and set(row["canonical_dialogue"]["round_refs"]) <= {"parent", "R01"}
            for row in rows),
        "one_feedback_card_per_root": len(cards) == 10 and all(
            row["feedback_card"]["root_id"] in cards for row in rows),
        "feedback_namespaces_unique": len({row["feedback_namespace"] for row in rows}) == 10,
        "feedback_user_ids_unique": len({row["feedback_user_id"] for row in rows}) == 10,
        "feedback_sessions_unique": len({row["r02_feedback_session_id"] for row in rows}) == 10,
        "r01_source_intents_closed": len(bootstrap_rows) == 10,
        "r01_source_dialogue_bindings_verified": all(
            row["source_dialogue_binding_verified"] is True for row in bootstrap_rows),
        "r01_source_completions_verified": all(
            row["source_parent_completion_verified"] is True
            and row["source_r01_completion_verified"] is True
            for row in bootstrap_rows),
        "completed_smoke_is_exactly_one": completed_ids == [COMPLETED_SMOKE_ROOT_ID],
        "pending_root_count_9": len(pending_ids) == 9,
        "api_calls_zero": True,
        "memorax_calls_zero": True,
        "r02_generation_zero": True,
        "r02_add_zero": True,
        "qa_calls_zero": True,
        "solver_calls_zero": True,
        "test_access_zero": True,
        "old_r02_content_read_zero": True,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    _write(output_root / "source_manifest.json", {
        "manifest_version": VERSION + "_source_manifest_v1",
        "created_at": _now(),
        "compiler_root": str(COMPILER_ROOT.resolve()),
        "round1_root_manifest": str(ROUND1_ROOT_MANIFEST.resolve()),
        "source_hashes": source_hashes,
        "selected_root_ids": selected_ids,
        "completed_feedback_smoke_root": COMPLETED_SMOKE_ROOT_ID,
        "old_r02_content_read": False,
    })
    _write_jsonl(output_root / "feedback_r02_batch_manifest.jsonl", rows)
    _write_jsonl(output_root / "r01_feedback_bootstrap_plan.jsonl", bootstrap_rows)
    _write_jsonl(output_root / "r02_feedback_execution_plan.jsonl", r02_rows)
    _write(output_root / "batch_plan.json", {
        "plan_version": VERSION + "_plan_v1",
        "status": "BLOCKED_UNTIL_R01_BOOTSTRAP" if pending_ids else "READY_FOR_R02_FEEDBACK_BATCH",
        "scope": "feedback_branch_only",
        "total_formal_roots": 10,
        "completed_feedback_smoke_roots": completed_ids,
        "pending_r01_bootstrap_roots": pending_ids,
        "pending_r02_feedback_roots": pending_ids,
        "r01_bootstrap_adds_per_pending_root": {"parent": 1, "r01_child": 1},
        "r02_feedback_adds_per_pending_root": 1,
        "control_branch_touched": False,
        "r02_input": "canonical Parent + R01 only",
        "r02_write_scope": "R02-feedback messages/window_context only",
        "feedback_persistence": {
            "card_in_memorax_messages": False,
            "card_in_memorax_window_context": False,
            "card_in_memorax_payload_metadata": False,
            "provenance_storage": "local reports and request ledger only",
        },
        "execution_order": ["R01 Parent Add", "R01 Child Add", "visibility audit", "R02-feedback generation", "R02-feedback Child Add", "visibility audit"],
        "recovery": "serial; never retry an uncertain Child Add; quarantine before next root",
    })
    _write(output_root / "preflight_report.json", {
        "preflight_version": VERSION + "_preflight_v1",
        "created_at": _now(),
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "decision": "BLOCK_R02_BATCH_UNTIL_R01_BOOTSTRAP" if pending_ids else "READY_FOR_R02_FEEDBACK_BATCH",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "formal_root_count": 10,
        "completed_feedback_smoke_root_ids": completed_ids,
        "pending_r01_bootstrap_root_ids": pending_ids,
        "pending_r02_feedback_root_ids": pending_ids,
        "api_calls": 0,
        "memorax_calls": 0,
        "r02_generation": 0,
        "r02_add_calls": 0,
        "qa_calls": 0,
        "solver_calls": 0,
        "test_access": 0,
        "old_r02_content_access": 0,
        "source_manifest_sha256": _sha(output_root / "source_manifest.json"),
        "batch_plan_sha256": _sha(output_root / "batch_plan.json"),
    })
    report = _load(output_root / "preflight_report.json")
    md = "\n".join([
        "# Round-1 Feedback R02 Batch Preflight",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        "- Scope: `feedback branch only`",
        f"- Formal roots: `{len(rows)}`",
        f"- Existing feedback R02 smoke: `{','.join(completed_ids)}`",
        f"- Pending R01-only bootstrap: `{len(pending_ids)}` roots",
        "- R02 generation/Add performed: `0/0`",
        "- API/Memorax calls performed: `0/0`",
        "- Old R02 content read: `0`",
        "",
        "The pending roots require a live Parent Add and R01 Child Add in the",
        "new feedback namespace before R02-feedback can be generated.",
        "",
    ])
    (output_root / "final_batch_preflight_report.md").write_text(md,
                                                                  encoding="utf-8")
    (output_root / "final_batch_preflight_report.md.sha256").write_text(
        _sha(output_root / "final_batch_preflight_report.md") + "\n",
        encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(args.output_root.resolve())
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "completed_feedback_smoke_root_ids": report["completed_feedback_smoke_root_ids"],
        "pending_r01_bootstrap_root_ids": report["pending_r01_bootstrap_root_ids"],
        "failed_checks": report["failed_checks"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
