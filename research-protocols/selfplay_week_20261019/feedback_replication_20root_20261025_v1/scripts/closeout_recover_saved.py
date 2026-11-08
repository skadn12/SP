#!/usr/bin/env python3
"""Recover one already-returned closeout JSON after a narrow event-parser bug.

This script never starts Codex and never changes a semantic value.  It is
restricted to the one consumed ``q20_closeout_frozen_report_01`` attempt.  The
saved attempt contains a completed turn and a valid JSON response, but the
locked helper classified the transient top-level reconnect event as a semantic
failure.  Recovery preserves that failure receipt and the event verbatim while
publishing the exact parsed response for final-artifact validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
CALL_ID = "q20_closeout_frozen_report_01"
CALL_DIR = BASE / "cli_calls" / CALL_ID
ATTEMPT = CALL_DIR / "attempt_01"
RAW = ATTEMPT / "LAST_MESSAGE.txt"
EVENTS = ATTEMPT / "EVENTS.jsonl"
TERMINAL = ATTEMPT / "TERMINAL.json"
START = ATTEMPT / "START.json"
REQUEST = ATTEMPT / "REQUEST.txt"
AUDIT = BASE / "INDEPENDENT_CLOSEOUT_AUDIT.json"
RESPONSE = BASE / "INDEPENDENT_CLOSEOUT_AUDIT_RESPONSE.json"
RECEIPT = BASE / "INDEPENDENT_CLOSEOUT_AUDIT_CALL.json"
METADATA = BASE / "INDEPENDENT_CLOSEOUT_AUDIT_METADATA.json"
PACKET = BASE / "CLOSEOUT_AUDIT_INPUT.json"

CONFIG_WARNING = (
    "Codex is ignoring 3 unrecognized configuration settings. Check for typos or deprecated settings.\n"
    "  user (/root/.codex/config.toml): `disable_response_storage` is ignored.\n"
    "  user (/root/.codex/config.toml): `network_access` is ignored.\n"
    "  user (/root/.codex/config.toml): `windows_wsl_setup_acknowledged` is ignored."
)
RECONNECT_ERROR = "Reconnecting... 1/5 (stream disconnected before completion: Transport error: network error: error decoding response body)"


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def publish(path: Path, value: Any) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"IMMUTABLE_OUTPUT_DRIFT:{path}")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    return bind(path)


def event_rows() -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(EVENTS.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"INVALID_EVENT_JSON:{line_number}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"EVENT_NOT_OBJECT:{line_number}")
        rows.append(value)
    return rows


def verify_saved_attempt() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    required = (RAW, EVENTS, TERMINAL, START, REQUEST, CALL_DIR / "SEMANTIC_FAILURE.json", PACKET, BASE / "FINAL_REPORT.json")
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"MISSING_SAVED_CLOSEOUT_EVIDENCE:{path}")
    terminal = read(TERMINAL)
    start = read(START)
    failure = read(CALL_DIR / "SEMANTIC_FAILURE.json")
    if terminal.get("returncode") != 0 or terminal.get("timed_out") is not False or terminal.get("stopped") is not False:
        raise RuntimeError("SAVED_CLOSEOUT_DID_NOT_EXIT_CLEANLY")
    backend = terminal.get("backend", {})
    for key in ("thread_id", "model", "model_provider", "effort", "cli_version"):
        if not backend.get(key):
            raise RuntimeError(f"MISSING_BACKEND_FIELD:{key}")
    if start.get("fresh_session") is not True or start.get("tool_calls_permitted") is not False:
        raise RuntimeError("SAVED_CLOSEOUT_START_CONTRACT_DRIFT")
    for key, path in (("request", REQUEST), ("events", EVENTS), ("raw_response", RAW)):
        if terminal.get(key) != bind(path):
            raise RuntimeError(f"TERMINAL_BINDING_DRIFT:{key}")
    if start.get("request") != terminal.get("request"):
        raise RuntimeError("START_TERMINAL_REQUEST_DRIFT")
    if failure.get("semantic_attempt_consumed") is not True:
        raise RuntimeError("CONSUMED_ATTEMPT_FLAG_MISSING")
    rows = event_rows()
    threads = [row for row in rows if row.get("type") == "thread.started"]
    turns_started = [row for row in rows if row.get("type") == "turn.started"]
    turns_completed = [row for row in rows if row.get("type") == "turn.completed"]
    if len(threads) != 1 or threads[0].get("thread_id") != backend.get("thread_id"):
        raise RuntimeError("FRESH_THREAD_EVENT_DRIFT")
    if len(turns_started) != 1 or len(turns_completed) != 1 or turns_completed[0].get("usage") != terminal.get("usage"):
        raise RuntimeError("TURN_COMPLETION_USAGE_DRIFT")
    top_errors = [row for row in rows if row.get("type") == "error"]
    if top_errors != [{"type": "error", "message": RECONNECT_ERROR}]:
        raise RuntimeError("UNEXPECTED_TOP_LEVEL_ERROR_EVENT")
    forbidden = []
    diagnostics = []
    for row in rows:
        if row.get("type") not in {"item.started", "item.updated", "item.completed"}:
            continue
        item = row.get("item") or {}
        if item.get("type") in {"agent_message", "reasoning"}:
            continue
        summary = {"event_type": row.get("type"), "item_type": item.get("type"), "id": item.get("id")}
        forbidden.append(summary)
        if row.get("type") != "item.completed" or item.get("type") != "error" or item.get("message") != CONFIG_WARNING or item.get("id") not in {"item_0", "item_1"}:
            raise RuntimeError("UNEXPECTED_FORBIDDEN_ITEM_EVENT")
        diagnostics.append(summary)
    if forbidden != terminal.get("forbidden_tool_items") or [row["id"] for row in diagnostics] != ["item_0", "item_1"]:
        raise RuntimeError("FORBIDDEN_ITEM_TERMINAL_DRIFT")
    messages = [row["item"].get("text", "") for row in rows if row.get("type") == "item.completed" and (row.get("item") or {}).get("type") == "agent_message"]
    if not messages:
        raise RuntimeError("NO_COMPLETED_AGENT_MESSAGE")
    raw_text = RAW.read_text(encoding="utf-8").strip()
    if raw_text != str(messages[-1]).strip():
        raise RuntimeError("RAW_AND_AGENT_MESSAGE_DRIFT")
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise RuntimeError("CLOSEOUT_RESPONSE_NOT_OBJECT")
    report_hash = hashlib.sha256((BASE / "FINAL_REPORT.json").read_bytes()).hexdigest()
    if parsed.get("verdict") not in {"PASS", "FAIL"} or parsed.get("report_sha256") != report_hash or parsed.get("independent_of_report_author") is not True:
        raise RuntimeError("CLOSEOUT_RESPONSE_SCHEMA_OR_REPORT_HASH_INVALID")
    packet = read(PACKET)
    if packet.get("report_binding", {}).get("sha256") != report_hash:
        raise RuntimeError("CLOSEOUT_PACKET_REPORT_BINDING_DRIFT")
    return parsed, rows, {"terminal": terminal, "failure": failure, "diagnostics": diagnostics, "top_errors": top_errors, "report_sha256": report_hash}


def recover() -> dict[str, Any]:
    parsed, rows, evidence = verify_saved_attempt()
    parsed_binding = publish(RESPONSE, parsed)
    audit_binding = publish(AUDIT, parsed)
    terminal = evidence["terminal"]
    classification = {
        "schema_version": "fgr20_saved_closeout_response_recovery_v1",
        "classification": "SAVED_VALID_JSON_WITH_PRESERVED_TOP_LEVEL_RECONNECT_EVENT",
        "call_id": CALL_ID,
        "fresh_thread_id": terminal["backend"]["thread_id"],
        "semantic_attempt_consumed": True,
        "new_model_call_started": False,
        "locked_helper_semantic_failure_preserved": True,
        "top_level_error_event_preserved": evidence["top_errors"],
        "startup_diagnostics_preserved": evidence["diagnostics"],
        "raw_response_binding": bind(RAW),
        "terminal_binding": bind(TERMINAL),
        "events_binding": bind(EVENTS),
        "request_binding": bind(REQUEST),
        "helper_failure_binding": bind(CALL_DIR / "SEMANTIC_FAILURE.json"),
        "response_binding": parsed_binding,
        "audit_binding": audit_binding,
        "semantic_value_source": "exact parsed JSON from saved completed agent_message/LAST_MESSAGE; no manager-authored fields",
    }
    receipt = {**terminal, "parsed_response": parsed_binding, "terminal_binding": bind(TERMINAL),
               "response_received": True, "event_classification": classification,
               "recovery_from_helper_failure": True}
    receipt_binding = publish(RECEIPT, receipt)
    metadata = {"schema_version": "fgr20_closeout_audit_call_metadata_saved_recovery_v1",
        "call_id": CALL_ID, "audit_binding": audit_binding,
        "report_binding": {"path": str(BASE / "FINAL_REPORT.json"), "sha256": evidence["report_sha256"], "bytes": (BASE / "FINAL_REPORT.json").stat().st_size},
        "audit_input_binding": bind(PACKET), "call_receipt_binding": receipt_binding,
        "backend": terminal.get("backend"), "usage": terminal.get("usage"),
        "actual_event_adapter": "locked helper rejected top-level reconnect event; saved-response recovery adapter",
        "locked_codex_calls_file_modified": False, "closeout_existing_artifacts_only": True,
        "semantic_verdict_authored_by_manager": False,
        "reviewer_response_preserved_without_semantic_edits": True,
        "recovery_classification": classification,
        "original_helper_semantic_failure": evidence["failure"],
    }
    metadata_binding = publish(METADATA, metadata)
    return {"status": "PASS_SAVED_RESPONSE_RECOVERED", "model_calls": 0,
            "audit_binding": audit_binding, "metadata_binding": metadata_binding,
            "helper_failure_preserved": True, "top_level_reconnect_preserved": True,
            "reviewer_verdict": parsed.get("verdict")}


def self_check() -> dict[str, Any]:
    parsed, rows, evidence = verify_saved_attempt()
    return {"status": "PASS_ENGINEERING_ONLY", "model_calls": 0,
            "semantic_attempt_consumed": True, "event_count": len(rows),
            "top_level_error_count": len(evidence["top_errors"]),
            "parsed_verdict": parsed.get("verdict"), "report_hash_matches": True,
            "files_written": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "recover"))
    args = parser.parse_args()
    result = self_check() if args.command == "self-check" else recover()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
