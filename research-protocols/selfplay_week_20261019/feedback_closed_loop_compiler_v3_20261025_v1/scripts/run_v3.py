#!/usr/bin/env python3
"""Single locked controller for feedback closed-loop compiler v3."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_core import (ARMS, OPERATIONS, binding, build_ledger, compile_root,
                     cross_check_catalog, digest_bytes, stable_json)

BASE = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((BASE / "PROTOCOL_LOCK.json").read_text())
INPUTS = json.loads((BASE / "INPUT_BINDINGS.json").read_text())
PRED = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/feedback_replication_20root_resume_20261025_v1")
CODEX = Path("/root/.vscode-server/extensions/openai.chatgpt-26.51002.51308-linux-x64/bin/linux-x86_64/codex")
PILOT = tuple(PROTOCOL["stability_gates"]["pilot_roots"])
ALL_ROOTS = tuple(PROTOCOL["fixed_cohorts"]["new16"])
SEM_LIMIT = 2
SEM_LOCK = threading.RLock()
SEM_ACTIVE: dict[int, dict[str, Any]] = {}
CONFIG_WARNING = (
    "Codex is ignoring 3 unrecognized configuration settings. Check for typos or deprecated settings.\n"
    "  user (/root/.codex/config.toml): `disable_response_storage` is ignored.\n"
    "  user (/root/.codex/config.toml): `network_access` is ignored.\n"
    "  user (/root/.codex/config.toml): `windows_wsl_setup_acknowledged` is ignored."
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def start_token(pid: int) -> str | None:
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return parts[19]
    except (OSError, IndexError):
        return None


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic(path: Path, value: Any) -> dict[str, Any]:
    path = path.resolve()
    if BASE not in path.parents and path != BASE:
        raise ValueError(f"write outside v3 directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)
    return binding(path)


def status(phase: str, **changes: Any) -> None:
    lock_path = BASE / ".run_status.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load(BASE / "RUN_STATUS.json") if (BASE / "RUN_STATUS.json").exists() else {}
        data.update(changes)
        data.update({"phase": phase, "updated_at_utc": now(), "training": False,
                     "controller_pid": os.getpid(), "controller_start_token": start_token(os.getpid()),
                     "semantic_parallel_limit": SEM_LIMIT,
                     "live_semantic_children": list(SEM_ACTIVE.values())})
        atomic(BASE / "RUN_STATUS.json", data)
        fcntl.flock(lock, fcntl.LOCK_UN)


def parse_tag(raw: str, tag: str, root_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    text = raw.strip()
    errors: list[str] = []
    if "\n" in text or not text.startswith(tag + "|"):
        return None, ["malformed_tagged_response"]
    fields: dict[str, str] = {}
    for piece in text.split("|")[1:]:
        if "=" not in piece:
            errors.append("field_without_equals")
            continue
        key, value = piece.split("=", 1)
        if key in fields:
            errors.append("duplicate_field:" + key)
        fields[key] = value
    if fields.get("root") != root_id:
        errors.append("root_mismatch")
    if tag == "CARD":
        if fields.get("arm") not in ("BOTH", "F", "G"):
            errors.append("invalid_arm")
        for key in ("f_targets", "g_targets"):
            if key not in fields:
                errors.append("missing_" + key)
        targets = {}
        for key in ("f_targets", "g_targets"):
            vals = [x for x in fields.get(key, "").split(",") if x]
            if len(vals) > 8:
                errors.append("too_many_targets")
            targets[key] = vals
        if errors:
            return None, sorted(set(errors))
        return {"root_id": root_id, "arm": fields["arm"], "f_targets": targets["f_targets"],
                "g_targets": targets["g_targets"], "rationale": fields.get("rationale", "")}, []
    if tag == "AUDIT":
        if fields.get("decision") not in ("PASS", "FAIL"):
            errors.append("invalid_decision")
        if "bad" not in fields:
            errors.append("missing_bad")
        if errors:
            return None, sorted(set(errors))
        return {"root_id": root_id, "decision": fields["decision"], "bad": fields.get("bad", "")}, []
    return None, ["unknown_tag"]


def validate_semantic_events(events_path: Path, raw: str) -> list[str]:
    """Accept only a complete text-only Codex turn plus known local warnings."""
    errors: list[str] = []
    try:
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        return ["invalid_event_stream:" + repr(exc)]
    if sum(x.get("type") == "thread.started" for x in events) != 1:
        errors.append("require_one_thread")
    completed = [x for x in events if x.get("type") == "turn.completed"]
    if len(completed) != 1:
        errors.append("require_one_turn_completion")
    messages: list[str] = []
    for event in events:
        typ = event.get("type")
        if typ == "error":
            errors.append("top_level_error")
            continue
        if not str(typ).startswith("item."):
            continue
        item = event.get("item") or {}
        ityp = item.get("type")
        if ityp == "agent_message" and typ == "item.completed":
            messages.append(str(item.get("text", "")))
        elif ityp in ("reasoning", None):
            continue
        elif ityp == "error" and item.get("message") == CONFIG_WARNING:
            continue
        else:
            errors.append("forbidden_or_unknown_item:" + str(ityp))
    if not messages or messages[-1].strip() != raw.strip():
        errors.append("final_agent_message_mismatch")
    return sorted(set(errors))


def invoke_semantic(name: str, prompt: str, tag: str, root_id: str) -> dict[str, Any]:
    """The only Codex invocation path. Existing directories are never retried."""
    directory = BASE / "semantic_calls" / name
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "REQUEST.txt"
    request.write_text(prompt)
    args = [str(CODEX), "exec", "--skip-git-repo-check", "--sandbox", "danger-full-access",
            "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="medium"',
            "--model", "gpt-6.1-sol", "--color", "never", "--json", "-C", str(BASE),
            "-o", str(directory / "LAST_MESSAGE.txt"), "-"]
    started = {"at": now(), "pid": None, "controller_pid": os.getpid(), "args": args,
               "request": binding(request), "kind": "semantic", "root_id": root_id}
    atomic(directory / "START.json", started)
    with SEM_LOCK:
        if len(SEM_ACTIVE) >= SEM_LIMIT:
            raise RuntimeError("semantic child limit exceeded")
    with (directory / "EVENTS.jsonl").open("wb") as events, (directory / "STDERR.log").open("wb") as stderr:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=events, stderr=stderr,
                                cwd=BASE, start_new_session=True)
        child = {"pid": proc.pid, "start_token": start_token(proc.pid), "call_id": name,
                 "root_id": root_id, "kind": "semantic", "started_at_utc": now()}
        started.update(child)
        atomic(directory / "START.json", started)
        with SEM_LOCK:
            SEM_ACTIVE[proc.pid] = child
            status("semantic_call", live_semantic_children=list(SEM_ACTIVE.values()))
        try:
            proc.communicate(prompt.encode(), timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        finally:
            with SEM_LOCK:
                SEM_ACTIVE.pop(proc.pid, None)
                status("semantic_call_finished", live_semantic_children=list(SEM_ACTIVE.values()))
    raw = (directory / "LAST_MESSAGE.txt").read_text() if (directory / "LAST_MESSAGE.txt").exists() else ""
    event_errors = validate_semantic_events(directory / "EVENTS.jsonl", raw)
    response, errors = parse_tag(raw, tag, root_id) if proc.returncode == 0 and not event_errors else (None, [f"exit_code:{proc.returncode}"] if proc.returncode else event_errors)
    if not raw:
        errors.append("missing_semantic_response")
    receipt = {"schema_version": "feedback_closed_loop_v3_semantic_receipt", "name": name,
               "root_id": root_id, "tag": tag, "accepted": not errors, "response": response,
               "failure_reasons": sorted(set(errors)), "event_validation_errors": event_errors, "returncode": proc.returncode,
               "finished_at_utc": now(), "request": binding(request)}
    if (directory / "LAST_MESSAGE.txt").exists():
        receipt["response_binding"] = binding(directory / "LAST_MESSAGE.txt")
    atomic(directory / "RECEIPT.json", receipt)
    return receipt


def draft_path_for(root_id: str) -> Path:
    sf = load(PRED / "SOURCE_FREEZE.json")
    row = next((x for x in sf["outcomes"] if x.get("root_id") == root_id), None)
    if not row or "fixed_initial_draft" not in row:
        raise ValueError("missing saved initial draft: " + root_id)
    p = Path(row["fixed_initial_draft"]["path"])
    raw = p.read_bytes()
    if digest_bytes(raw) != row["fixed_initial_draft"]["sha256"] or len(raw) != row["fixed_initial_draft"]["bytes"]:
        raise ValueError("initial draft binding mismatch: " + root_id)
    return p


def verify_input_bindings() -> list[str]:
    errors = []
    for row in INPUTS.get("bindings", []):
        p = Path(row["path"])
        try:
            raw = p.read_bytes()
        except OSError as exc:
            errors.append(f"missing_binding:{p}:{exc}")
            continue
        if len(raw) != row["bytes"] or digest_bytes(raw) != row["sha256"]:
            errors.append(f"binding_mismatch:{p}")
    return errors


def compact_ledger(ledger: dict[str, Any]) -> str:
    rows = [f"{f['fact_id']}\t{f['turn_id']}\t{f['predicate']}\t{int(f['truth'])}\t{f['source_substring']}" for f in ledger["facts"]]
    return "\n".join(rows)


def compact_catalog(catalog: list[dict[str, Any]]) -> str:
    rows = []
    for slot in catalog:
        for label, option in slot["options"].items():
            rows.append(f"{slot['arm']}:{slot['slot_id']}:{label}\t{option['transform']}\t{option['fact_id']}\t{int(option['truth'])}")
    return "\n".join(rows)


def challenger_prompt(root_id: str, ledger: dict[str, Any]) -> str:
    return ("You are a challenger targeting one source ledger. Return exactly one compact tagged line, no JSON, markdown, questions, options, Gold, evidence, citations, or rewritten facts. "
            "Select only existing fact IDs, up to eight per arm.\n"
            "Format: CARD|root=<root>|arm=BOTH|f_targets=id,id|g_targets=id,id|rationale=short\n"
            f"CARD DATA root={root_id}\n{compact_ledger(ledger)}")


def audit_prompt(root_id: str, ledger: dict[str, Any], catalog: list[dict[str, Any]]) -> str:
    return ("You are an independent audit of a compact fact ledger and compiled catalog. Return exactly one tagged line, no JSON, markdown, question text, options, Gold, or new evidence. PASS only if all source spans and operation rows are internally consistent. "
            "Format: AUDIT|root=<root>|decision=PASS|bad=none (or decision=FAIL with one concise reason).\n"
            f"AUDIT DATA root={root_id}\nLEDGER\n{compact_ledger(ledger)}\nCATALOG\n{compact_catalog(catalog)}")


def process_root(root_id: str) -> dict[str, Any]:
    out = BASE / "roots" / root_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        ledger = build_ledger(root_id, draft_path_for(root_id))
        atomic(out / "LEDGER.json", ledger)
    except Exception as exc:
        result = {"root_id": root_id, "status": "FAIL", "failure_reasons": ["ledger:" + repr(exc)], "semantic_calls": 0}
        atomic(out / "OUTCOME.json", result)
        return result
    card_receipt = invoke_semantic(root_id.replace("cc-train-episode-", "") + "_challenger", challenger_prompt(root_id, ledger), "CARD", root_id)
    card = card_receipt.get("response") or {"root_id": root_id, "arm": "BOTH", "f_targets": [], "g_targets": []}
    challengers = {"F": {"target_fact_ids": card.get("f_targets", [])}, "G": {"target_fact_ids": card.get("g_targets", [])}}
    atomic(out / "CHALLENGER_CARD.json", {"receipt": card_receipt, "card": card,
                                            "constraints": ["no_questions", "no_options", "no_gold", "no_evidence"]})
    try:
        catalog = compile_root(ledger, challengers)
        atomic(out / "CATALOG.json", {"slots": catalog})
    except Exception as exc:
        catalog = []
        compile_error = repr(exc)
    else:
        compile_error = None
    audit_receipt = invoke_semantic(root_id.replace("cc-train-episode-", "") + "_audit", audit_prompt(root_id, ledger, catalog), "AUDIT", root_id)
    mechanical = cross_check_catalog(catalog, ledger) if catalog else [compile_error or "empty_catalog"]
    audit_ok = bool(audit_receipt.get("accepted")) and audit_receipt.get("response", {}).get("decision") == "PASS"
    repair = {"used": False, "removed_fact_id": None}
    if mechanical or not audit_ok:
        # One deterministic fact-removal repair is allowed. Recompile and re-audit locally;
        # no third semantic call is permitted by the two-call contract.
        repair["used"] = True
        refs = {o.get("fact_id") for s in catalog for o in s.get("options", {}).values()}
        removable = [f["fact_id"] for f in ledger["facts"] if f["fact_id"] not in refs]
        if removable:
            repair["removed_fact_id"] = sorted(removable)[-1]
            ledger["facts"] = [f for f in ledger["facts"] if f["fact_id"] != repair["removed_fact_id"]]
        try:
            catalog = compile_root(ledger, challengers)
            mechanical = cross_check_catalog(catalog, ledger)
        except Exception as exc:
            mechanical = ["repair_compile:" + repr(exc)]
    local_reaudit = {"decision": "PASS" if not mechanical else "FAIL", "mechanical_errors": mechanical}
    atomic(out / "AUDIT.json", {"receipt": audit_receipt, "local_reaudit": local_reaudit, "repair": repair})
    if catalog:
        atomic(out / "CATALOG_FINAL.json", {"slots": catalog})
    passed = bool(card_receipt.get("accepted")) and audit_ok and not mechanical and len(catalog) == 16
    result = {"root_id": root_id, "status": "PASS" if passed else "FAIL",
              "failure_reasons": ([] if passed else (mechanical + audit_receipt.get("failure_reasons", []) + (["audit_not_pass"] if not audit_ok else []))),
              "semantic_calls": 2, "ledger": binding(out / "LEDGER.json"),
              "challenger": binding(out / "CHALLENGER_CARD.json"), "audit": binding(out / "AUDIT.json"),
              "catalog": binding(out / "CATALOG_FINAL.json") if (out / "CATALOG_FINAL.json").exists() else None,
              "repair": repair}
    atomic(out / "OUTCOME.json", result)
    return result


def solver_requests(slots: list[dict[str, Any]], roots: list[str]) -> list[dict[str, Any]]:
    rows = []
    for slot in slots:
        if slot["root_id"] not in roots:
            continue
        field = "options"
        display_key = "display" if slot["question_type"] == "multi_select" else "display"
        source = json.loads(Path(slot["source_binding"]["path"]).read_text())
        source_context = "\n".join(
            f"{turn['turn_ref']} {turn['speaker']}: {turn['text']}"
            for scene in source["scenes"] for turn in scene["turns"]
        )
        for presentation in slot["presentations"]:
            rows.append({"request_id": f"{slot['candidate_id']}__{presentation['variant_id']}__FULL__THINK_OFF__V3",
                         "root_id": slot["root_id"], "arm": slot["arm"], "slot_id": slot["slot_id"],
                         "question_type": slot["question_type"], "question": slot["question"],
                         "sampling_seed": int.from_bytes(hashlib.sha256((slot["candidate_id"] + presentation["variant_id"]).encode()).digest()[:4], "big") % 2147483647,
                         "display": {k: v["text"] for k, v in presentation[display_key].items()},
                         "prompt": "Frozen dialogue source:\n" + source_context + "\n\nTask: use only this source. Follow the requested operation. Return concise evidence and a final boxed label answer.\n" + slot["question"] + "\n" + "\n".join(f"{k}. {v['text']}" for k, v in presentation[display_key].items()) +
                                  "\nAnswer with the requested labels and concise evidence.", "gold": presentation["gold"]})
    return rows


def run_solver(slots: list[dict[str, Any]], roots: list[str]) -> dict[str, Any]:
    requests = solver_requests(slots, roots)
    ipc = BASE / "ipc"
    ipc.mkdir(exist_ok=True)
    request_path = ipc / "b1_requests.jsonl"
    request_path.write_text("\n".join(stable_json(r) for r in requests) + "\n")
    out_path = BASE / "RAW_SCORES.json"
    env = dict(os.environ)
    runtime = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b/lib/python3.10/site-packages")
    env["PYTHONPATH"] = str(runtime) + os.pathsep + env.get("PYTHONPATH", "")
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    proc = subprocess.run([sys.executable, str(BASE / "scripts" / "b1_worker.py"), "--requests", "ipc/b1_requests.jsonl", "--out", "RAW_SCORES.json"], cwd=BASE, env=env, timeout=7200)
    if not out_path.exists():
        raw = {"prepared_count": len(requests), "returned_count": 0, "records": [], "backend": {"exit_code": proc.returncode}, "training": False}
        atomic(out_path, raw)
    else:
        raw = load(out_path)
    raw["prepared_count"] = len(requests)
    raw["returned_count"] = len(raw.get("records", []))
    raw["missing_count"] = len(requests) - raw["returned_count"]
    atomic(out_path, raw)
    status("solver_batch", solver_prepared=len(requests), solver_returned=raw["returned_count"], solver_missing=raw["missing_count"])
    return raw


def write_analysis(raw: dict[str, Any], slots: list[dict[str, Any]], roots: list[str]) -> None:
    returned = raw.get("returned_count", 0)
    analysis = {"schema_version": "feedback_closed_loop_v3_analysis", "roots": roots,
                "planned_presentations": len(roots) * 16 * 2, "returned_presentations": returned,
                "missing_presentations": len(roots) * 16 * 2 - returned,
                "paired_root_unit": True, "missing_not_scored_incorrect": True,
                "feedback_benefit_claim": False, "training": False}
    atomic(BASE / "ANALYSIS.json", analysis)
    (BASE / "RESULTS_ZH.md").write_text("# v3 结果\n\n状态：" + ("通过" if analysis["missing_presentations"] == 0 else "阻断") +
                                     f"\n计划 B1 展示：{analysis['planned_presentations']}\n返回：{returned}\n缺失：{analysis['missing_presentations']}\n")


def blocker(reason: list[str], root_results: list[dict[str, Any]], slots: list[dict[str, Any]], raw: dict[str, Any] | None = None) -> None:
    text = ["# STABILITY BLOCKER", "", "The locked v3 stability gate did not pass.", "", "Reasons:"]
    text += [f"- {x}" for x in reason]
    (BASE / "STABILITY_BLOCKER.md").write_text("\n".join(text) + "\n")
    if raw is None:
        raw = {"schema_version": "feedback_closed_loop_v3_raw_scores", "prepared_count": 0, "returned_count": 0,
               "records": [], "training": False}
        atomic(BASE / "RAW_SCORES.json", raw)
    write_analysis(raw, slots, [r["root_id"] for r in root_results if r.get("status") == "PASS"])
    report = {"schema_version": "feedback_closed_loop_v3_final_report", "status": "BLOCKED",
              "pilot_roots": list(PILOT), "root_results": root_results,
              "compiled_slots": len(slots), "raw_scores": binding(BASE / "RAW_SCORES.json"),
              "reasons": reason, "training": False, "sealed_old_reruns": 0}
    atomic(BASE / "FINAL_REPORT.json", report)
    status("blocked", blocker_reasons=reason, pilot_passed=False)


def main() -> int:
    lock_path = BASE / ".controller.lock"
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("controller lock already held", file=sys.stderr)
        return 2
    controller = {"pid": os.getpid(), "start_token": start_token(os.getpid()), "started_at_utc": now(),
                  "cwd": str(BASE), "kind": "v3_controller", "semantic_parallel_limit": SEM_LIMIT}
    atomic(BASE / "CONTROLLER.json", controller)
    status("preflight", controller=controller, pilot_roots=list(PILOT), new16_roots=len(ALL_ROOTS), training=False)
    binding_errors = verify_input_bindings()
    atomic(BASE / "INPUT_BINDING_CHECK.json", {"at": now(), "errors": binding_errors, "bindings": INPUTS.get("bindings", [])})
    if binding_errors:
        blocker(["input_binding_check_failed"] + binding_errors, [], [])
        return 1
    # The test gate is executed by the controller before any semantic child starts.
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], cwd=BASE)
    atomic(BASE / "TEST_RECEIPT.json", {"at": now(), "returncode": test.returncode, "command": "python -m unittest discover -s tests -p test_*.py"})
    if test.returncode:
        blocker(["unit_tests_failed"], [], [])
        return 1
    status("source_ledgers", tests_passed=True)
    all_results: list[dict[str, Any]] = []
    # Two roots per wave means at most two live semantic Codex children.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(process_root, root) for root in PILOT]):
            result = future.result()
            all_results.append(result)
            atomic(BASE / "ROOT_PROGRESS.json", {"at": now(), "results": sorted(all_results, key=lambda x: x["root_id"])})
            status("roots", roots_finished=len(all_results), roots_passed=sum(x["status"] == "PASS" for x in all_results), roots_failed=sum(x["status"] != "PASS" for x in all_results))
    all_results.sort(key=lambda x: x["root_id"])
    ledgers = []
    slots: list[dict[str, Any]] = []
    for result in all_results:
        p = BASE / "roots" / result["root_id"] / "LEDGER.json"
        if p.exists():
            ledgers.append(load(p))
        p = BASE / "roots" / result["root_id"] / "CATALOG_FINAL.json"
        if p.exists():
            slots.extend(load(p)["slots"])
    atomic(BASE / "LEDGER_FREEZE.json", {"schema_version": "feedback_closed_loop_v3_ledger_freeze", "at": now(), "ledgers": ledgers, "root_results": all_results})
    atomic(BASE / "CHALLENGER_CARDS.json", {"roots": [{"root_id": r["root_id"], "path": str(BASE / "roots" / r["root_id"] / "CHALLENGER_CARD.json")} for r in all_results]})
    pilot_reasons = []
    if len(all_results) != 4 or any(r["status"] != "PASS" for r in all_results):
        pilot_reasons.append("not_all_4_source_and_audit_pass")
    if len(slots) != 64:
        pilot_reasons.append(f"pilot_compiled_slots:{len(slots)}/64")
    atomic(BASE / "QUESTION_FREEZE.json", {"schema_version": "feedback_closed_loop_v3_question_freeze", "at": now(), "planned_slots": 64, "valid_slots": len(slots), "slots": slots, "pilot": True})
    status("pilot_gate", pilot_root_results=all_results, pilot_valid_slots=len(slots), pilot_gate_passed=not pilot_reasons)
    if pilot_reasons:
        blocker(pilot_reasons, all_results, slots)
        return 1
    raw = run_solver(slots, list(PILOT))
    if raw.get("returned_count") != 128:
        blocker([f"pilot_b1_returned:{raw.get('returned_count', 0)}/128"], all_results, slots, raw)
        return 1
    # Full production is unlocked only after the pilot solver gate.
    remaining = [r for r in ALL_ROOTS if r not in PILOT]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(process_root, root) for root in remaining]):
            all_results.append(future.result())
    all_results.sort(key=lambda x: x["root_id"])
    slots = []
    for result in all_results:
        p = BASE / "roots" / result["root_id"] / "CATALOG_FINAL.json"
        if p.exists():
            slots.extend(load(p)["slots"])
    atomic(BASE / "QUESTION_FREEZE.json", {"schema_version": "feedback_closed_loop_v3_question_freeze", "at": now(), "planned_slots": 256, "valid_slots": len(slots), "slots": slots, "pilot": False})
    if len(slots) != 256 or any(r["status"] != "PASS" for r in all_results):
        blocker(["full_root_or_slot_gate_failed"], all_results, slots)
        return 1
    raw = run_solver(slots, list(ALL_ROOTS))
    if raw.get("returned_count") != 512:
        blocker([f"full_b1_returned:{raw.get('returned_count', 0)}/512"], all_results, slots, raw)
        return 1
    write_analysis(raw, slots, list(ALL_ROOTS))
    report = {"schema_version": "feedback_closed_loop_v3_final_report", "status": "COMPLETED",
              "root_count": 16, "compiled_slots": 256, "b1_presentations": 512,
              "raw_scores": binding(BASE / "RAW_SCORES.json"), "training": False, "sealed_old_reruns": 0}
    atomic(BASE / "FINAL_REPORT.json", report)
    status("complete", full_gate_passed=True, solver_returned=512)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
