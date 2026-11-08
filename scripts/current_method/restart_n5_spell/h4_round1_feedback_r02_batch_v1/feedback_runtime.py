"""Parameterized live R02-feedback runner for one bootstrapped pilot root.

Feedback cards are injected into the Planner prompt only.  The Add payload is
checked by the adapter and contains only the generated R02 dialogue delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..h4_round1_feedback_r02_single_root_v1.adapter import (
    assert_dialogue_only_memorax_payload,
    patch_existing_runtime,
    write_artifact_sidecars,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1 import live


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_PLAN_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_v1_live_4"
DEFAULT_BOOTSTRAP_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_execution_v1_live_2"
DEFAULT_OUTPUT_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_single_root_v1_live_1"
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [_load_line(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _load_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("plan_row_not_object")
    return value


def _stage_r01_source(*, bootstrap_root: Path, root_id: str,
                      output_root: Path) -> Path:
    source_root = bootstrap_root / "roots" / root_id
    source_memory = source_root / "memory"
    for name in ("r01_intent.json", "r01_completion.json"):
        if not (source_memory / name).is_file():
            raise RuntimeError(f"bootstrap_artifact_missing:{source_memory / name}")
    staged = output_root.parent / (output_root.name + ".r01_source")
    if staged.exists():
        raise RuntimeError(f"staged_source_exists:{staged}")
    memory = staged / "memory"
    memory.mkdir(parents=True)
    shutil.copyfile(source_memory / "r01_intent.json", memory / "child_add_intent.json")
    shutil.copyfile(source_memory / "r01_completion.json", memory / "child_add_completion.json")
    return staged


def _find_row(plan_root: Path, root_id: str) -> dict[str, Any]:
    rows = _rows(plan_root / "r02_feedback_execution_plan.jsonl")
    selected = [row for row in rows if row.get("root_id") == root_id]
    if len(selected) != 1:
        raise RuntimeError(f"root_plan_not_unique:{root_id}:{len(selected)}")
    return selected[0]


def _card(row: Mapping[str, Any]) -> dict[str, Any]:
    target = row.get("feedback_card")
    if not isinstance(target, Mapping):
        raise RuntimeError("feedback_card_target_missing")
    return {
        "feedback_version": "round1_feedback_compiler_v1",
        "root_id": row.get("root_id"),
        "target": dict(target),
    }


def _payload_audit(output_root: Path, card: Mapping[str, Any]) -> dict[str, Any]:
    intent_path = output_root / "memory/child_add_intent.json"
    if not intent_path.is_file():
        return {"status": "BLOCKED", "reason": "child_add_intent_missing"}
    intent = _load(intent_path)
    payload = intent.get("payload")
    if not isinstance(payload, Mapping):
        return {"status": "BLOCKED", "reason": "payload_missing"}
    try:
        assert_dialogue_only_memorax_payload(payload)
        guard_status = "PASS"
        guard_error = None
    except ValueError as error:
        guard_status = "BLOCKED"
        guard_error = str(error)
    forbidden_text = {"feedback card", "feedback_card", "gold", "solver output"}
    serialized_dialogue = json.dumps(
        {"messages": payload.get("messages"),
         "window_context": payload.get("window_context")},
        ensure_ascii=False).casefold()
    dialogue_hits = sorted(token for token in forbidden_text
                           if token in serialized_dialogue)
    turn_refs = [str(ref) for ref in intent.get("turn_refs") or []]
    result = {
        "audit_version": "h4_round1_feedback_dialogue_only_payload_v1",
        "status": "PASS" if guard_status == "PASS" and not dialogue_hits else "BLOCKED",
        "payload_path": str(intent_path.resolve()),
        "payload_sha256": _sha(intent_path),
        "session_id": intent.get("session_id"),
        "round2_only_turn_refs": bool(turn_refs) and all(
            ref.startswith("R02:") for ref in turn_refs),
        "turn_count": len(turn_refs),
        "feedback_control_keys_in_payload": [],
        "feedback_control_keys_in_metadata": [],
        "feedback_control_text_in_dialogue": dialogue_hits,
        "guard_status": guard_status,
        "guard_error": guard_error,
        "feedback_card_sha256_local_only": hashlib.sha256(
            json.dumps(card, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
    }
    return result


def _no_write_audit(output_root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    report = _load(
        output_root / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json")
    add_records = [
        row for row in report.get("memorax_call_records") or []
        if isinstance(row, Mapping)
        and str(row.get("path") or "") == "/v1/memories/add"
    ]
    write_state = _load(output_root / "memory/child_add_write_state.json")
    audit = {
        "audit_version": "h4_round1_feedback_r02_no_memorax_writes_v1",
        "status": "PASS" if (
            not add_records
            and int(report.get("child_add_calls") or 0) == 0
            and write_state.get("add_submitted") is False
            and result.get("no_memorax_writes") is True
        ) else "BLOCKED",
        "no_memorax_writes_requested": result.get("no_memorax_writes") is True,
        "memorax_add_call_count": len(add_records),
        "child_add_calls": int(report.get("child_add_calls") or 0),
        "child_add_submitted": write_state.get("add_submitted"),
        "memorax_read_call_count": len([
            row for row in report.get("memorax_call_records") or []
            if isinstance(row, Mapping)
            and str(row.get("path") or "") != "/v1/memories/add"
        ]),
    }
    return audit


def run(*, plan_root: Path, bootstrap_root: Path, root_id: str,
        output_root: Path, config_path: Path, tokenizer_path: Path,
        credentials_path: Path, allow_insecure_http: bool,
        live_mode: bool, no_memorax_writes: bool = False) -> dict[str, Any]:
    plan_root = plan_root.resolve()
    bootstrap_root = bootstrap_root.resolve()
    output_root = output_root.resolve()
    row = _find_row(plan_root, root_id)
    bootstrap_report = _load(
        bootstrap_root / "roots" / root_id / "final_r01_bootstrap_report.json")
    if bootstrap_report.get("status") != "PASS":
        raise RuntimeError(f"r01_bootstrap_not_pass:{root_id}")
    if output_root.exists():
        raise RuntimeError(f"output_exists:{output_root}")
    if not live_mode:
        return {
            "status": "OFFLINE_PLAN_ONLY",
            "root_id": root_id,
            "canonical_dialogue_path": row.get("canonical_dialogue_path"),
            "namespace": row.get("namespace"),
            "feedback_card_file_sha256": row.get("feedback_card_file_sha256"),
            "memorax_calls": 0,
            "memorax_add_calls": 0,
            "no_memorax_writes": bool(no_memorax_writes),
        }
    input_path = Path(str(row.get("canonical_dialogue_path") or ""))
    if not input_path.is_file():
        raise RuntimeError(f"canonical_dialogue_missing:{input_path}")
    card = _card(row)
    patch_record = patch_existing_runtime(card)
    staged_source = _stage_r01_source(
        bootstrap_root=bootstrap_root, root_id=root_id, output_root=output_root)
    runtime_report = live.preflight(
        input_path=input_path,
        run_root=output_root,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http,
        reuse_existing_namespace=True,
        source_run_root=staged_source,
        memorax_user_namespace=str(row["namespace"]),
        memory_only_fast=True,
        event_bucket_only=True,
        round_index=2,
        no_memorax_writes=no_memorax_writes,
    )
    if runtime_report.get("decision") != "pass":
        raise RuntimeError("runtime_preflight_failed:" + ",".join(
            runtime_report.get("failed_checks") or []))
    runtime_report["feedback_adapter"] = patch_record
    runtime_report["feedback_card_sha256"] = row.get("feedback_card_row_sha256")
    result = live.execute_live(
        report=runtime_report,
        input_path=input_path,
        run_root=output_root,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http,
        reuse_existing_namespace=True,
        memorax_user_namespace=str(row["namespace"]),
        memory_only_fast=True,
        event_bucket_only=True,
        round_index=2,
        no_memorax_writes=no_memorax_writes,
    )
    payload_audit = _payload_audit(output_root, card)
    (output_root / "feedback_payload_audit.json").write_text(
        json.dumps(payload_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output_root / "feedback_payload_audit.json.sha256").write_text(
        _sha(output_root / "feedback_payload_audit.json") + "\n", encoding="utf-8")
    result["feedback_payload_audit"] = payload_audit
    result["no_memorax_writes"] = bool(no_memorax_writes)
    result["memorax_add_calls"] = 0 if no_memorax_writes else int(
        result.get("child_add_calls") or 0)
    if no_memorax_writes:
        no_write_audit = _no_write_audit(output_root, result)
        (output_root / "no_memorax_write_audit.json").write_text(
            json.dumps(no_write_audit, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        (output_root / "no_memorax_write_audit.json.sha256").write_text(
            _sha(output_root / "no_memorax_write_audit.json") + "\n",
            encoding="utf-8")
        result["no_memorax_write_audit"] = no_write_audit
    result["artifact_sha256_manifest"] = write_artifact_sidecars(output_root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--bootstrap-root", type=Path, default=DEFAULT_BOOTSTRAP_ROOT)
    parser.add_argument("--root-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--memorax-credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-memorax-writes", action="store_true",
                        help="Generate R02 locally without submitting the child Add.")
    args = parser.parse_args(argv)
    result = run(
        plan_root=args.plan_root,
        bootstrap_root=args.bootstrap_root,
        root_id=args.root_id,
        output_root=args.output_root,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http,
        live_mode=args.live,
        no_memorax_writes=args.no_memorax_writes,
    )
    print(json.dumps({
        "status": result.get("status"),
        "root_id": args.root_id,
        "run_root": str(args.output_root.resolve()),
        "live_started": args.live,
        "feedback_payload_audit": result.get("feedback_payload_audit"),
    }, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"OFFLINE_PLAN_ONLY", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
