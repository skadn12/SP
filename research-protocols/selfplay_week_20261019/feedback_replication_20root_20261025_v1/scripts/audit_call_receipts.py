#!/usr/bin/env python3
"""Append-only mechanical call-receipt audit; never calls a model or judges content."""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import types
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
EXPECTED_BACKEND = ("gpt-6.1-sol", "hua", "ultra")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def active(attempt: Path) -> dict[str, Any] | None:
    candidates = [attempt / "PID.json", BASE / "process_registry" /
                  f"{attempt.parent.name}__{attempt.name}.json"]
    for path in candidates:
        if not path.is_file():
            continue
        row = read(path)
        pid, token = row.get("pid"), row.get("start_token")
        if (isinstance(pid, int) and token is not None and process_start(pid) == str(token)):
            return {"pid": pid, "start_token": str(token), "registry_path": str(path)}
    return None


def event_classifier():
    """Reuse only the frozen pure event-contract function, not a semantic driver."""
    path = BASE / "scripts/question_driver.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                 target.id == "_CONFIG_WARNING" for target in node.targets) or
             isinstance(node, ast.FunctionDef) and node.name == "classify_cli_events"]
    if len(nodes) != 2:
        raise RuntimeError("Frozen event classifier AST does not match its explicit contract")
    namespace = {"Path": Path, "Any": Any, "json": json, "cc": types.SimpleNamespace(bind=bind)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + "::<event-audit>", "exec"), namespace)
    return namespace["classify_cli_events"], bind(path)


def audit() -> dict[str, Any]:
    started = utc()
    directories = sorted(path for path in (BASE / "cli_calls").iterdir() if path.is_dir())
    classifier, classifier_binding = event_classifier()
    bindings: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    threads: dict[str, list[str]] = collections.defaultdict(list)
    backend_counts: collections.Counter = collections.Counter()
    usages: collections.Counter = collections.Counter()
    checked_bindings = 0

    def add(path: Path | str) -> dict[str, Any]:
        value = bind(path)
        bindings[value["path"]] = value
        return value

    def verify(value: Any) -> None:
        nonlocal checked_bindings
        if isinstance(value, dict):
            if all(key in value for key in ("path", "sha256", "bytes")):
                expected = {key: value[key] for key in ("path", "sha256", "bytes")}
                if add(expected["path"]) != expected:
                    raise ValueError("Binding drift: " + expected["path"])
                checked_bindings += 1
            for child in value.values():
                verify(child)
        elif isinstance(value, list):
            for child in value:
                verify(child)

    def issue(row: dict[str, Any], message: str) -> None:
        row.setdefault("errors", []).append(message)
        errors.append({"call_id": row.get("call_id"), "attempt": row.get("attempt"), "error": message})

    add(Path(__file__))
    add(classifier_binding["path"])
    prior = BASE / "CALL_RECEIPT_AUDIT_RESUME.json"
    if prior.is_file():
        add(prior)
    for call_dir in directories:
        attempts = sorted(path for path in call_dir.glob("attempt_*") if path.is_dir())
        result_path, failure_path = call_dir / "RESULT.json", call_dir / "SEMANTIC_FAILURE.json"
        result = read(result_path) if result_path.is_file() else None
        failure = read(failure_path) if failure_path.is_file() else None
        expected_request = None
        semantic_attempts = 0
        for index, attempt in enumerate(attempts, 1):
            row: dict[str, Any] = {"call_id": call_dir.name, "attempt": attempt.name,
                                   "errors": [], "semantic_content_reviewed": False}
            rows.append(row)
            start_path, terminal_path = attempt / "START.json", attempt / "TERMINAL.json"
            try:
                if not start_path.is_file():
                    recent = dt.datetime.now().timestamp() - attempt.stat().st_mtime < 15
                    row["state"] = "PENDING_START_PUBLICATION" if recent else "UNCLASSIFIED_ATTEMPT_NO_START"
                    if not recent:
                        issue(row, "Attempt directory has no immutable START receipt")
                    continue
                start = read(start_path)
                row["start_binding"] = add(start_path)
                verify(start)
                if expected_request is None:
                    expected_request = start.get("request")
                if start.get("request") != expected_request:
                    issue(row, "One call_id has different request hashes across attempts")
                command = start.get("command", [])
                if (len(command) < 2 or command[1] != "exec" or "resume" in command or
                        "--model" in command or "-m" in command or any(
                        re.search(r"(?:^|[.\s])(?:model|model_provider|model_reasoning_effort|reasoning_effort)=", str(arg))
                        for arg in command)):
                    issue(row, "Call command is not fresh default-model exec")
                if (start.get("fresh_session") is not True or start.get("tool_calls_permitted") is not False or
                        start.get("semantic_attempt") != 1 or start.get("transport_attempt") != index or index > 3):
                    issue(row, "Fresh-session/semantic/transport-budget START metadata drift")
                if start.get("request") != bind(attempt / "REQUEST.txt"):
                    issue(row, "START request does not bind its own REQUEST.txt")
                if not terminal_path.is_file():
                    live = active(attempt)
                    if live:
                        row.update(state="ACTIVE", active_process=live,
                                   note="Mutable events/stderr/raw and pending RESULT deliberately excluded")
                        event_path = attempt / "EVENTS.jsonl"
                        if event_path.is_file():
                            observed = []
                            for line in event_path.read_text(encoding="utf-8").splitlines():
                                try:
                                    event = json.loads(line)
                                except ValueError:
                                    continue
                                if event.get("type") == "thread.started" and event.get("thread_id"):
                                    observed.append(event["thread_id"])
                            row["observed_active_thread_ids"] = observed
                            for thread in set(observed):
                                threads[thread].append(call_dir.name + "/" + attempt.name)
                        continue
                    if failure and failure.get("semantic_attempt_consumed") is True:
                        verify(failure)
                        add(failure_path)
                        row.update(state="INTERRUPTED_EXPLICITLY_CONSUMED",
                                   recovery_classification=failure.get("recovery_classification"),
                                   semantic_failure_binding=bind(failure_path))
                        semantic_attempts += 1
                    else:
                        row["state"] = "UNCLASSIFIED_INTERRUPTED_START"
                        issue(row, "START without TERMINAL has no live PID or explicit consumed classification")
                    continue
                terminal = read(terminal_path)
                row["terminal_binding"] = add(terminal_path)
                verify(terminal)
                for key, filename in (("events", "EVENTS.jsonl"), ("stderr", "STDERR.log"),
                                      ("request", "REQUEST.txt")):
                    if terminal.get(key) != bind(attempt / filename):
                        issue(row, "TERMINAL does not bind its own " + filename)
                if terminal.get("raw_response") is not None and terminal["raw_response"] != bind(attempt / "LAST_MESSAGE.txt"):
                    issue(row, "TERMINAL does not bind its own LAST_MESSAGE.txt")
                if terminal.get("request") != start.get("request"):
                    issue(row, "START and TERMINAL request bindings differ")
                backend = terminal.get("backend", {})
                row["backend"] = backend
                if any(not backend.get(key) for key in ("thread_id", "model", "model_provider", "effort", "cli_version")):
                    issue(row, "Terminal backend metadata is incomplete")
                if backend.get("model_override") is not None:
                    issue(row, "Terminal reports a model override")
                if backend.get("executable") != command[0]:
                    issue(row, "Terminal executable differs from START command")
                actual_backend = tuple(backend.get(key) for key in ("model", "model_provider", "effort"))
                if actual_backend != EXPECTED_BACKEND:
                    issue(row, "Actual terminal model/provider/effort differs from frozen observed default")
                backend_counts[actual_backend] += 1
                thread = backend.get("thread_id")
                if thread:
                    threads[thread].append(call_dir.name + "/" + attempt.name)
                events = [json.loads(line) for line in (attempt / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
                thread_events = [event.get("thread_id") for event in events if event.get("type") == "thread.started"]
                if thread_events != [thread]:
                    issue(row, "Raw EVENTS do not identify exactly the terminal's fresh thread")
                messages = [event.get("item", {}).get("text", "") for event in events if
                            event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message"]
                row["events_summary"] = dict(collections.Counter(event.get("type", "MISSING_TYPE") for event in events))
                row["completed_agent_messages"] = len(messages)
                try:
                    classification = classifier(attempt / "EVENTS.jsonl", terminal)
                    classification["terminal_binding"] = bind(terminal_path)
                    row["event_contract"] = {"mechanical_status": "PASS", "classification": classification["classification"]}
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    classification = None
                    row["event_contract"] = {"mechanical_status": "REJECTED", "reason": f"{type(exc).__name__}: {exc}"}
                transport_path = attempt / "TRANSPORT_FAILURE.json"
                if transport_path.is_file():
                    transport = read(transport_path)
                    verify(transport)
                    add(transport_path)
                    raw = (attempt / "LAST_MESSAGE.txt").read_text() if (attempt / "LAST_MESSAGE.txt").exists() else ""
                    activity = any(event.get("type") in {"item.started", "item.updated", "item.completed"} for event in events)
                    if (raw.strip() or messages or activity or terminal.get("returncode") == 0 or
                            terminal.get("stopped") or terminal.get("timed_out")):
                        issue(row, "Transport retry classification contains semantic activity/response or a nontransport terminal")
                    if index < len(attempts) and transport.get("retry_authorized") is not True:
                        issue(row, "A later attempt followed a transport failure without retry authorization")
                    row.update(state="NO_SEMANTIC_TRANSPORT_FAILURE", transport_binding=bind(transport_path))
                    continue
                semantic_attempts += 1
                if index < len(attempts):
                    issue(row, "A subsequent attempt followed an already consumed semantic attempt")
                if result and result.get("terminal_binding", {}).get("path") == str(terminal_path.resolve()):
                    verify(result)
                    add(result_path)
                    row.update(state="RESULT", result_binding=bind(result_path))
                    if failure:
                        issue(row, "Call has both RESULT and SEMANTIC_FAILURE")
                    if (result.get("terminal_binding") != bind(terminal_path) or
                            any(result.get(key) != value for key, value in terminal.items()) or
                            result.get("response_received") is not True):
                        issue(row, "RESULT does not preserve its exact terminal payload")
                    if result.get("parsed_response") != bind(call_dir / "PARSED_RESPONSE.json"):
                        issue(row, "RESULT does not bind its own PARSED_RESPONSE.json")
                    raw_path = attempt / "LAST_MESSAGE.txt"
                    raw_text = raw_path.read_text(encoding="utf-8").strip()
                    if not messages or raw_text != messages[-1].strip():
                        issue(row, "Returned raw response differs from the final completed agent_message")
                    if raw_text.startswith("```") and raw_text.endswith("```"):
                        raw_text = "\n".join(raw_text.splitlines()[1:-1])
                    parsed = json.loads(raw_text)
                    if not isinstance(parsed, dict) or parsed != read(call_dir / "PARSED_RESPONSE.json"):
                        issue(row, "PARSED_RESPONSE is not an identical JSON-object parse of the raw response")
                    if classification is None or result.get("event_classification") != classification:
                        issue(row, "Successful RESULT lacks the exact passing independent event-format classification")
                    if terminal.get("stopped") or terminal.get("timed_out") or terminal.get("returncode") != 0:
                        issue(row, "Successful RESULT follows an incomplete terminal")
                elif failure:
                    verify(failure)
                    add(failure_path)
                    row.update(state="TERMINAL_CONSUMED_SEMANTIC_FAILURE", semantic_failure_binding=bind(failure_path),
                               failure_error=failure.get("error"), recovery_classification=failure.get("recovery_classification"))
                    if failure.get("semantic_attempt_consumed") is not True:
                        issue(row, "Semantic failure does not explicitly consume its attempt")
                    if failure.get("terminal") != bind(terminal_path):
                        issue(row, "Semantic failure does not bind its own terminal")
                else:
                    age = dt.datetime.now().timestamp() - terminal_path.stat().st_mtime
                    row["state"] = "FINALIZING_RECENT_TERMINAL" if age < 15 else "UNCLASSIFIED_TERMINAL"
                    if age >= 15:
                        issue(row, "Terminal lacks RESULT, SEMANTIC_FAILURE, or transport classification")
                for path in sorted(attempt.glob("*CLASSIFICATION*.json")):
                    verify(read(path))
                    add(path)
                for key, value in (terminal.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usages[key] += value
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                issue(row, f"{type(exc).__name__}: {exc}")
                row.setdefault("state", "AUDIT_READ_ERROR")
        if semantic_attempts > 1:
            errors.append({"call_id": call_dir.name, "error": "More than one consumed semantic attempt under one call_id"})
        if len(attempts) > 3:
            errors.append({"call_id": call_dir.name, "error": "More than three transport attempts under one call_id"})
    duplicates = {thread: locations for thread, locations in threads.items() if len(locations) > 1}
    if duplicates:
        errors.append({"error": "Fresh thread ids reused across attempts", "duplicates": duplicates})
    for value in list(bindings.values()):
        if bind(value["path"]) != value:
            errors.append({"error": "Immutable receipt changed during audit", "path": value["path"]})
    states = dict(collections.Counter(row.get("state", "UNKNOWN") for row in rows))
    return {"schema_version": "fgr20_call_receipt_mechanical_audit_v2", "at_utc": utc(),
            "started_at_utc": started, "status": "PASS" if not errors else "FAIL",
            "scope": "Mechanical receipt/hash/fresh-thread/default-backend/event/attempt classification only; no semantic acceptance",
            "expected_observed_default_backend": dict(zip(("model", "provider", "effort"), EXPECTED_BACKEND)),
            "calls_snapshotted": len(directories), "attempts_checked": len(rows), "attempt_state_counts": states,
            "active_calls_excluded_from_terminal_verdict": [row["call_id"] for row in rows if row.get("state") == "ACTIVE"],
            "new_call_directories_after_snapshot": sorted(path.name for path in (BASE / "cli_calls").iterdir()
                if path.is_dir() and path not in directories),
            "model_backend_effort_counts": [dict(zip(("model", "provider", "effort", "attempts"), (*key, count)))
                for key, count in sorted(backend_counts.items(), key=lambda item: str(item[0]))],
            "duplicate_threads": duplicates, "verified_binding_references": checked_bindings,
            "terminal_usage_totals": dict(usages), "usage_is_terminal_actual_not_cost_estimate": True,
            "errors": errors, "attempts": rows, "bindings": sorted(bindings.values(), key=lambda value: value["path"]),
            "model_calls": 0, "old_receipts_modified": False, "semantic_review_performed": False,
            "event_classifier_binding": classifier_binding,
            "limitations": ["Active mutable output streams are observed but neither frozen nor failed.",
                "Recorded backend metadata and the raw CLI thread event are cross-checked; this does not establish model-family independence.",
                "Consumed semantic failures remain consumed even if an event-format classifier accepts their preserved startup diagnostics.",
                "Token usage sums all terminal attempts, including failed attempts; it is not monetary cost."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        classifier, binding = event_classifier()
        assert callable(classifier) and binding["path"].endswith("scripts/question_driver.py")
        assert process_start(-1) is None
        print(json.dumps({"status": "PASS", "model_calls": 0, "files_written": 0}))
        return 0
    report = audit()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = BASE / ("CALL_RECEIPT_AUDIT_" + stamp + ".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "calls_snapshotted", "attempts_checked",
        "attempt_state_counts", "active_calls_excluded_from_terminal_verdict", "model_backend_effort_counts",
        "duplicate_threads", "verified_binding_references", "errors")}, ensure_ascii=False, indent=2))
    print(json.dumps({"audit_binding": bind(output), "model_calls": 0, "old_receipts_modified": False}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
