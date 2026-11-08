"""Evidence-preserving recovery for completed source calls misclassified as tools.

This module never creates semantic content itself. Saved drafts are reconstructed
from their bound LAST_MESSAGE.txt; review/revision model calls, when explicitly
run, go through the task's locked codex_calls.call implementation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

BASE = Path(__file__).resolve().parents[1]
CALLS = BASE / "cli_calls"
RECOVERY = BASE / "source_recovery"
RULE_VERSION = "fgr20_exact_ignored_local_config_diagnostic_v1"
EXPECTED_CONFIG = Path("/root/.codex/config.toml")
EXPECTED_KEYS = {
    "disable_response_storage",
    "network_access",
    "windows_wsl_setup_acknowledged",
}
HEADER = "Codex is ignoring 3 unrecognized configuration settings. Check for typos or deprecated settings."
KEY_LINE = re.compile(r"^  user \(/root/\.codex/config\.toml\): `([^`]+)` is ignored\.$")
USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                "output_tokens", "reasoning_output_tokens")
BACKEND_FIELDS = ("thread_id", "model", "model_provider", "effort", "cli_version")
ALLOWED_ITEM_TYPES = {"agent_message", "reasoning", "error"}
ALLOWED_EVENT_TYPES = {"thread.started", "turn.started", "turn.completed",
                       "item.started", "item.updated", "item.completed"}


class RecoveryError(RuntimeError):
    pass


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bind(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": digest(raw), "bytes": len(raw)}


def write_immutable(path: Path | str, value: Any) -> dict[str, Any]:
    """Create once; repeated publication must be semantically identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise RecoveryError(f"Immutable recovery artifact drift: {path}")
    else:
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return bind(path)


def _strict_json_text(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip() not in ("```", "```json") or lines[-1].strip() != "```":
            raise RecoveryError("Raw response has an unsupported code-fence wrapper")
        text = "\n".join(lines[1:-1])
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"Saved raw response is not JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise RecoveryError("Saved raw response must be a JSON object")
    return result


def _warning_keys(message: Any) -> list[str] | None:
    """Return exact ignored keys only for the single known local-config warning."""
    if not isinstance(message, str):
        return None
    lines = message.splitlines()
    if len(lines) != 4 or lines[0] != HEADER:
        return None
    keys = []
    for line in lines[1:]:
        match = KEY_LINE.fullmatch(line)
        if not match:
            return None
        keys.append(match.group(1))
    if len(set(keys)) != 3 or set(keys) != EXPECTED_KEYS:
        return None
    return keys


def _binding_matches(binding: Any) -> bool:
    if not isinstance(binding, dict) or set(("path", "sha256", "bytes")) - set(binding):
        return False
    try:
        return bind(binding["path"]) == binding
    except OSError:
        return False


def _event_rows(events_path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(events_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecoveryError(f"Invalid event JSON on line {number}") from exc
        if not isinstance(event, dict):
            raise RecoveryError(f"Event line {number} is not an object")
        rows.append(event)
    return rows


def classify_saved_attempt(call_id: str, cdir: Path | str, attempt: Path | str,
                           terminal: dict[str, Any] | None = None,
                           *, expected_root_id: str | None = None) -> dict[str, Any]:
    """Validate complete evidence for a no-tool config-warning-only response.

    No file is written. All original evidence remains byte-for-byte untouched.
    Any event outside the narrow startup-warning plus ordinary text-only turn fails.
    """
    cdir, attempt = Path(cdir), Path(attempt)
    if terminal is None:
        terminal = read(attempt / "TERMINAL.json")
    if not isinstance(terminal, dict):
        raise RecoveryError("Terminal record must be an object")
    if terminal.get("returncode") != 0 or terminal.get("timed_out") is not False or terminal.get("stopped") is not False:
        raise RecoveryError("Call did not exit cleanly")
    if not isinstance(terminal.get("forbidden_tool_items"), list):
        raise RecoveryError("Terminal forbidden_tool_items is missing")
    if not all(terminal.get("backend", {}).get(k) for k in BACKEND_FIELDS):
        raise RecoveryError("Actual backend metadata is incomplete")
    if terminal["backend"].get("model_override") is not None:
        raise RecoveryError("Call used an unexpected model override")
    if not isinstance(terminal["usage"], dict) or any(
            not isinstance(terminal["usage"].get(k), int) or isinstance(terminal["usage"].get(k), bool)
            or terminal["usage"][k] < 0 for k in USAGE_FIELDS):
        raise RecoveryError("Actual token usage is incomplete")
    if terminal["usage"]["input_tokens"] <= 0 or terminal["usage"]["output_tokens"] <= 0:
        raise RecoveryError("Completed response must report nonzero input and output tokens")

    start_path = attempt / "START.json"
    request_path = attempt / "REQUEST.txt"
    events_path = attempt / "EVENTS.jsonl"
    raw_path = attempt / "LAST_MESSAGE.txt"
    for key, path in (("request", request_path), ("events", events_path), ("raw_response", raw_path)):
        if not _binding_matches(terminal.get(key)) or Path(terminal[key]["path"]).resolve() != path.resolve():
            raise RecoveryError(f"Terminal {key} evidence binding is missing or drifted")
    if not start_path.exists():
        raise RecoveryError("START record is missing")
    start = read(start_path)
    if start.get("fresh_session") is not True or start.get("tool_calls_permitted") is not False:
        raise RecoveryError("START does not document a fresh no-tools session")
    if not _binding_matches(start.get("request")) or start["request"] != terminal["request"]:
        raise RecoveryError("START/terminal request bindings differ")
    request_text = request_path.read_text()
    if not request_text.startswith("You are the ") or "Do not use ANY tools, commands, files, network" not in request_text:
        raise RecoveryError("Request does not carry the locked no-tools author/reviewer contract")
    if expected_root_id and expected_root_id not in request_text:
        raise RecoveryError("Request does not contain the expected root context")

    raw_text = raw_path.read_text()
    parsed = _strict_json_text(raw_text)
    if expected_root_id is not None and parsed.get("root_id") != expected_root_id:
        raise RecoveryError("Returned JSON root_id does not match the frozen root")
    events = _event_rows(events_path)
    # Reuse the exact allowlist used by the question driver for every recovered
    # call, including its refusal of top-level reconnect/error events.
    import sys
    scripts = str(BASE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import question_driver
    try:
        event_classification = question_driver.classify_cli_events(events_path, terminal)
    except Exception as exc:
        raise RecoveryError(f"question_driver event classifier refused saved attempt: {exc}") from exc
    if event_classification.get("status") != "PASS":
        raise RecoveryError("question_driver event classifier did not pass")
    event_types = [event.get("type") for event in events]
    thread_rows = [e for e in events if e.get("type") == "thread.started"]
    turn_started = [e for e in events if e.get("type") == "turn.started"]
    turn_completed = [e for e in events if e.get("type") == "turn.completed"]
    if len(thread_rows) != 1 or len(turn_started) != 1 or len(turn_completed) != 1:
        raise RecoveryError("Expected one complete fresh thread and one completed turn")
    if thread_rows[0].get("thread_id") != terminal["backend"].get("thread_id"):
        raise RecoveryError("Event and terminal thread IDs differ")
    if turn_completed[0].get("usage") != terminal["usage"]:
        raise RecoveryError("Event and terminal usage differ")

    item_events = [e for e in events if str(e.get("type", "")).startswith("item.")]
    if any((e.get("item") or {}).get("type") not in ALLOWED_ITEM_TYPES for e in item_events):
        raise RecoveryError("Tool or unknown item event present")
    errors = [e for e in item_events if (e.get("item") or {}).get("type") == "error"]
    if not errors:
        raise RecoveryError("No ignored-config diagnostic to classify")
    warning_messages = []
    warning_keys: set[str] = set()
    for event in errors:
        item = event.get("item", {})
        if event.get("type") != "item.completed":
            raise RecoveryError("Error diagnostic was not a completed startup item")
        keys = _warning_keys(item.get("message"))
        if keys is None:
            raise RecoveryError("An error item is not exclusively the exact ignored-local-config diagnostic")
        warning_keys.update(keys)
        warning_messages.append(item["message"])
    if warning_keys != EXPECTED_KEYS:
        raise RecoveryError("Config-warning set is incomplete or unexpected")

    agent_messages = [e.get("item", {}).get("text") for e in item_events
                      if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"]
    if not agent_messages or not isinstance(agent_messages[-1], str):
        raise RecoveryError("At least one completed assistant JSON response is required")
    if agent_messages[-1].strip() != raw_text.strip():
        raise RecoveryError("Saved raw response differs from the completed agent_message event")

    computed_forbidden = [
        {"event_type": event.get("type"), "item_type": (event.get("item") or {}).get("type"),
         "id": (event.get("item") or {}).get("id")}
        for event in item_events if (event.get("item") or {}).get("type") not in ("agent_message", "reasoning")
    ]
    if terminal["forbidden_tool_items"] != computed_forbidden:
        raise RecoveryError("Terminal forbidden-item summary does not match raw events")
    if any(row.get("item_type") != "error" for row in computed_forbidden):
        raise RecoveryError("A forbidden tool item was recorded")

    terminal_path = attempt / "TERMINAL.json"
    return {
        "schema_version": "fgr20_call_startup_diagnostic_classification_v1",
        "rule_version": RULE_VERSION,
        "call_id": call_id,
        "classification": "STARTUP_IGNORED_LOCAL_CONFIG_DIAGNOSTICS_ONLY",
        "semantic_response_present": True,
        "tool_calls_observed": False,
        "config_path": str(EXPECTED_CONFIG),
        "ignored_config_keys": sorted(warning_keys),
        "diagnostic_event_count": len(errors),
        "diagnostic_messages": warning_messages,
        "thread_id": terminal["backend"]["thread_id"],
        "backend": copy.deepcopy(terminal["backend"]),
        "usage": copy.deepcopy(terminal["usage"]),
        "evidence": {
            "start": bind(start_path), "request": bind(request_path), "events": bind(events_path),
            "terminal": bind(terminal_path), "raw_response": bind(raw_path),
        },
        "question_driver_event_classification": event_classification,
        "returned_json": {"is_object": True, "root_id": parsed.get("root_id"),
                           "raw_response_sha256": bind(raw_path)["sha256"]},
    }


def _sidecar_path(attempt: Path) -> Path:
    return attempt / "STARTUP_DIAGNOSTIC_CLASSIFICATION.json"


def install_codex_call_adapter(codex_calls_module: Any) -> None:
    """Require question_driver's exact event adapter without changing call()."""
    if getattr(codex_calls_module, "_fgr20_startup_diagnostic_adapter", False):
        return
    import sys
    scripts = str(BASE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import question_driver  # noqa: F401
    if not getattr(codex_calls_module, "_fgr20_event_adapter_installed", False):
        raise RecoveryError("question_driver exact CLI event adapter was not installed")
    original_call = codex_calls_module.call
    codex_calls_module._fgr20_startup_diagnostic_adapter = True
    if codex_calls_module.call is not original_call:
        raise AssertionError("Adapter must not replace the locked call API")


def source_driver_module():
    import sys
    scripts = str(BASE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import source_driver
    return source_driver


def _source_prompt_inputs(sd: Any, row: dict[str, Any]):
    packet, parent, natives, safe_packet = sd.load_inputs(row)
    embedded = {"neutral_packet": safe_packet, "complete_parent_source": parent,
                "complete_native_records": natives}
    return packet, parent, natives, embedded


def _source_call_id(root_id: str, purpose: str) -> str:
    return f"{root_id.replace('-', '_')}_source_recovery_{purpose}"


def _write_run_artifact(root_dir: Path, name: str, value: Any, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    binding = write_immutable(root_dir / name, value)
    if binding not in artifacts:
        artifacts.append(binding)
    return binding


def _saved_author_response(root_id: str, row: dict[str, Any], root_dir: Path,
                           sd: Any, artifacts: list[dict[str, Any]]):
    call_id = f"{root_id.replace('-', '_')}_source_draft"
    cdir = CALLS / call_id
    attempt = cdir / "attempt_01"
    outcome_path = BASE / "source_authoring" / root_id / "OUTCOME.json"
    if not outcome_path.exists():
        raise RecoveryError("Original source OUTCOME.json is missing")
    original_outcome = read(outcome_path)
    if original_outcome.get("root_id") != root_id or original_outcome.get("status") != "FAILED":
        raise RecoveryError("Recovery is limited to an original failed source outcome")
    if original_outcome.get("artifacts") != [] or "Author/reviewer violated no-tools contract" not in original_outcome.get("failure_reason", ""):
        raise RecoveryError("Original failure is not the documented helper misclassification")
    packet, parent, natives, embedded = _source_prompt_inputs(sd, row)
    classification = classify_saved_attempt(call_id, cdir, attempt, expected_root_id=root_id)
    failure_path = cdir / "SEMANTIC_FAILURE.json"
    if not failure_path.exists():
        raise RecoveryError("Original semantic-failure receipt is missing")
    failure = read(failure_path)
    if failure.get("error") != "Author/reviewer violated no-tools contract" or failure.get("semantic_attempt_consumed") is not True:
        raise RecoveryError("Original call failure receipt does not match the narrow recovery case")
    if failure.get("terminal") != classification["evidence"]["terminal"]:
        raise RecoveryError("Original failure receipt does not bind this terminal record")
    call_sidecar = write_immutable(_sidecar_path(attempt), classification)
    response = _strict_json_text((attempt / "LAST_MESSAGE.txt").read_text())
    response_binding = _write_run_artifact(root_dir, "AUTHOR_RESPONSE_0.json", response, artifacts)
    call_record = {
        "schema_version": "fgr20_recovered_saved_author_call_v1",
        "call_id": call_id,
        "classification_binding": call_sidecar,
        "codex_call_failure_binding": bind(failure_path),
        "call_evidence": classification["evidence"],
        "response_binding": response_binding,
        "no_model_call_made_for_reconstruction": True,
        "original_outcome_binding": bind(outcome_path),
    }
    call_record_binding = _write_run_artifact(root_dir, "AUTHOR_CALL_0_RECOVERY.json", call_record, artifacts)
    draft, normalization_warnings = sd.normalize(response, packet)
    draft_binding = _write_run_artifact(root_dir, "DRAFT_0.json", draft, artifacts)
    _write_run_artifact(root_dir, "METADATA_NORMALIZATION_0.json", normalization_warnings, artifacts)
    mechanical = sd.validate_draft(draft, packet, parent, natives)
    mechanical_binding = _write_run_artifact(root_dir, "MECHANICAL_0.json", mechanical, artifacts)
    return {
        "packet": packet, "parent": parent, "natives": natives, "embedded": embedded,
        "response": response, "draft": draft, "mechanical": mechanical,
        "normalization_warnings": normalization_warnings, "call_id": call_id,
        "call_record_binding": call_record_binding, "response_binding": response_binding,
        "draft_binding": draft_binding, "mechanical_binding": mechanical_binding,
        "classification_binding": call_sidecar,
    }


def _review_one(root_id: str, round_number: int, embedded: dict[str, Any], draft: dict[str, Any],
                mechanical: dict[str, Any], sd: Any, root_dir: Path, artifacts: list[dict[str, Any]],
                cc: Any):
    label = _source_call_id(root_id, f"independent_review_{round_number}")
    review_input = {"complete_inputs": embedded, "complete_normalized_draft": draft,
                    "mechanical_validation": mechanical}
    prompt = (cc.prefix("independent shared-source reviewer") + sd.REVIEW_RULES
              + "\nCOMPLETE_REVIEW_INPUTS:\n"
              + json.dumps(review_input, ensure_ascii=False, separators=(",", ":")))
    review, call_result = cc.call(label, prompt)
    review_binding = _write_run_artifact(root_dir, f"REVIEW_{round_number}.json", review, artifacts)
    call_binding = _write_run_artifact(root_dir, f"REVIEW_CALL_{round_number}.json", call_result, artifacts)
    audit = sd.validate_review(review, embedded["neutral_packet"],
                               embedded["complete_parent_source"], draft,
                               [{"reservation": n["reservation"], "complete_record": n["complete_record"]}
                                for n in embedded["complete_native_records"]])
    audit_binding = _write_run_artifact(root_dir, f"REVIEW_VALIDATION_{round_number}.json", audit, artifacts)
    return review, audit, {"call_id": label, "review_binding": review_binding,
                           "call_binding": call_binding, "audit_binding": audit_binding}


def _assemble_source(root_id: str, packet: dict[str, Any], parent: dict[str, Any], draft: dict[str, Any],
                     review: dict[str, Any], review_binding: dict[str, Any], draft_binding: dict[str, Any],
                     normalization_warnings: list[Any], revision_count: int, sd: Any) -> dict[str, Any]:
    final = copy.deepcopy(parent)
    new_turns = sd.turns_for(draft)
    final.update(schema_version="feedback_closed_loop_r03_source_v1", source_status=packet["source_status"],
                 scene_ref=sd.SCENES[-1], input_source_sha256=packet["parent_source_binding"]["sha256"],
                 draft_sha256=draft_binding["sha256"], previous_turn_count=len(parent["turns"]),
                 new_turn_count=48, cumulative_new_turn_count=parent["cumulative_new_turn_count"] + 48,
                 turns=copy.deepcopy(parent["turns"]) + new_turns)
    final["closed_loop_provenance"] = {
        "round": "R03", "mode": "A_then_B_sequential_cumulative",
        "parent_source_binding": packet["parent_source_binding"], "draft_binding": draft_binding,
        "review_binding": review_binding, "native_events": packet["native_events"],
        "warnings": review["warnings"], "metadata_normalization_warnings": normalization_warnings,
        "status": "SOURCE_REVIEWED_BEFORE_QUESTION_AUTHORING",
        "extension_purpose": "EXPLORATORY_NEUTRAL_TRANSFER_NOT_ROOT_SPECIFIC_DIAGNOSIS",
        "historical_parent_seen_by_M_clean": True, "unseen_root_claim_permitted": False,
        "same_source_for_F_and_G": True, "revision_count": revision_count,
        "recovered_from_saved_codex_response": True,
    }
    if final["turns"][:len(parent["turns"])] != parent["turns"]:
        raise RecoveryError("Parent source prefix changed during assembly")
    if len(final["turns"]) != final["baseline_turn_count"] + final["cumulative_new_turn_count"]:
        raise RecoveryError("Cumulative source turn counts do not reconcile")
    return final


def recover_root(root_id: str) -> dict[str, Any]:
    """Recover a saved initial draft, run independent review, and at most one revision."""
    sd = source_driver_module()
    import sys
    scripts = str(BASE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import codex_calls as cc
    install_codex_call_adapter(cc)

    roots = sd.validate_preparation()
    rows = {row["root_id"]: row for row in roots}
    if root_id not in rows:
        raise RecoveryError(f"Root is not in the frozen Train selection: {root_id}")
    root_dir = RECOVERY / root_id
    root_dir.mkdir(parents=True, exist_ok=True)
    lock_path = RECOVERY / ".locks" / f"{root_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        outcome_path = root_dir / "ROOT_RECOVERY.json"
        if outcome_path.exists():
            saved = read(outcome_path)
            for binding in saved.get("artifacts", []):
                if bind(binding["path"]) != binding:
                    raise RecoveryError(f"Saved recovery artifact drift: {binding['path']}")
            if saved.get("source_binding") and bind(saved["source_binding"]["path"]) != saved["source_binding"]:
                raise RecoveryError("Saved recovered source drift")
            return saved

        artifacts: list[dict[str, Any]] = []
        outcome: dict[str, Any] = {
            "schema_version": "fgr20_source_root_recovery_v1", "root_id": root_id,
            "original_outcome_binding": bind(BASE / "source_authoring" / root_id / "OUTCOME.json"),
            "material_packet_binding": rows[root_id]["material_packet_binding"],
            "max_revisions": 1, "substitution_allowed": False,
            "original_outcome_untouched": True, "model_calls_for_saved_draft_reconstruction": 0,
            "recovery_calls_through_locked_codex_calls": True, "status": "FAILED",
            "artifacts": artifacts,
        }
        try:
            original = read(BASE / "source_authoring" / root_id / "OUTCOME.json")
            if original.get("status") == "PASS":
                source_binding = original.get("source_binding")
                if not isinstance(source_binding, dict) or not _binding_matches(source_binding):
                    raise RecoveryError("Original PASS source binding is missing or drifted")
                source = read(source_binding["path"])
                _, parent, _, _ = _source_prompt_inputs(sd, rows[root_id])
                if source.get("root_id") != root_id:
                    raise RecoveryError("Original PASS source root id mismatch")
                if source.get("turns", [])[:len(parent["turns"])] != parent["turns"]:
                    raise RecoveryError("Original PASS source changed the frozen parent prefix")
                if len(source.get("turns", [])) != source.get("baseline_turn_count", -1) + source.get("cumulative_new_turn_count", -2):
                    raise RecoveryError("Original PASS source cumulative turn counts do not reconcile")
                outcome.update(status="PASS_ORIGINAL_SOURCE_REUSED", source_binding=source_binding,
                               original_review_binding=original.get("final_review_binding"),
                               original_outcome_untouched=True, no_recovery_model_calls=True,
                               parent_prefix_preserved=True)
                outcome["artifacts"] = artifacts
                write_immutable(outcome_path, outcome)
                return outcome
            saved = _saved_author_response(root_id, rows[root_id], root_dir, sd, artifacts)
            packet, parent, natives, embedded = (saved["packet"], saved["parent"], saved["natives"], saved["embedded"])
            draft, mechanical = saved["draft"], saved["mechanical"]
            normalization_warnings = saved["normalization_warnings"]
            call_lineage = [{"role": "saved_initial_author", "call_id": saved["call_id"],
                             "classification_binding": saved["classification_binding"],
                             "call_record_binding": saved["call_record_binding"],
                             "response_binding": saved["response_binding"], "draft_binding": saved["draft_binding"]}]
            if mechanical["status"] != "PASS":
                outcome.update(status="FAILED_MECHANICAL", failure_reason="Saved initial draft failed source_driver.validate_draft")
            else:
                review, audit, review_lineage = _review_one(root_id, 0, embedded, draft, mechanical,
                                                            sd, root_dir, artifacts, cc)
                call_lineage.append({"role": "independent_source_review_0", **review_lineage})
                revision_count = 0
                review_binding = review_lineage["review_binding"]
                if mechanical["status"] == audit["status"] == "PASS":
                    pass
                else:
                    correction = {"previous_normalized_draft": draft, "mechanical_validation": mechanical,
                                  "independent_review": review, "review_contract_validation": audit}
                    revision_prompt = (cc.prefix("neutral shared-source author performing the one allowed revision")
                                       + sd.AUTHOR_RULES
                                       + "\nThis is the only revision. Resolve all blocking findings; preserve all valid material.\nCOMPLETE_INPUTS:\n"
                                       + json.dumps(embedded, ensure_ascii=False, separators=(",", ":"))
                                       + "\nPREVIOUS_DRAFT_AND_FEEDBACK:\n"
                                       + json.dumps(correction, ensure_ascii=False, separators=(",", ":")))
                    revision_id = _source_call_id(root_id, "author_revision")
                    revision, revision_call = cc.call(revision_id, revision_prompt)
                    author_binding = _write_run_artifact(root_dir, "AUTHOR_1.json", revision, artifacts)
                    revision_call_binding = _write_run_artifact(root_dir, "AUTHOR_CALL_1.json", revision_call, artifacts)
                    revision_draft, revision_warnings = sd.normalize(revision, packet)
                    revision_draft_binding = _write_run_artifact(root_dir, "DRAFT_1.json", revision_draft, artifacts)
                    revision_meta_binding = _write_run_artifact(root_dir, "METADATA_NORMALIZATION_1.json", revision_warnings, artifacts)
                    revision_mechanical = sd.validate_draft(revision_draft, packet, parent, natives)
                    revision_mechanical_binding = _write_run_artifact(root_dir, "MECHANICAL_1.json", revision_mechanical, artifacts)
                    call_lineage.append({"role": "one_allowed_source_revision", "call_id": revision_id,
                                         "author_binding": author_binding, "call_binding": revision_call_binding,
                                         "draft_binding": revision_draft_binding,
                                         "metadata_binding": revision_meta_binding,
                                         "mechanical_binding": revision_mechanical_binding,
                                         "author_id_is_original_author_lineage_only": True,
                                         "revision_is_fresh_thread": True})
                    draft, mechanical = revision_draft, revision_mechanical
                    normalization_warnings, revision_count = revision_warnings, 1
                    if cc.stop_generation():
                        raise RecoveryError("Generation cutoff reached after revision; no further call started")
                    review, audit, review_lineage = _review_one(root_id, 1, embedded, draft, mechanical,
                                                                sd, root_dir, artifacts, cc)
                    call_lineage.append({"role": "independent_source_recheck_1", **review_lineage})
                    review_binding = review_lineage["review_binding"]

                if mechanical["status"] == audit["status"] == "PASS":
                    final = _assemble_source(root_id, packet, parent, draft, review, review_binding,
                                             revision_draft_binding if revision_count else saved["draft_binding"],
                                             normalization_warnings, revision_count, sd)
                    source_binding = _write_run_artifact(root_dir, "SOURCE_FINAL.json", final, artifacts)
                    outcome.update(status="PASS", revision_count=revision_count, source_binding=source_binding,
                                   final_draft_binding=(revision_draft_binding if revision_count else saved["draft_binding"]),
                                   final_review_binding=review_binding, parent_prefix_preserved=True,
                                   warnings=review["warnings"])
                else:
                    outcome.update(status="FAILED_REVIEW", revision_count=revision_count,
                                   mechanical_status=mechanical["status"], review_validation=audit,
                                   failure_reason="Source did not pass independent review after the single allowed revision")
            outcome["call_lineage"] = call_lineage
            outcome["recovery_constraints"] = {
                "max_source_revisions": 1, "max_reviewer_rechecks_after_revision": 1,
                "source_author_calls_for_initial_saved_response": 0,
                "source_review_and_revision_calls_use_codex_calls_call": True,
                "original_cli_events_terminal_raw_response_preserved": True,
            }
        except Exception as exc:
            outcome.update(status="FAILED_CALL_OR_ENGINEERING", failure_reason=f"{type(exc).__name__}: {exc}")
            outcome["error_type"] = type(exc).__name__
        outcome["artifacts"] = artifacts
        write_immutable(outcome_path, outcome)
        return outcome


def _atomic_replace_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return bind(path)


def publish_canonical_freeze() -> dict[str, Any]:
    """Publish recovery selection/freeze under separate immutable filenames."""
    lock_path = BASE / ".source_driver.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecoveryError("source_driver is still active; canonical publication refused") from exc
        sd = source_driver_module()
        roots = sd.validate_preparation()
        original_outcomes = []
        recovered = []
        for row in roots:
            original_path = BASE / "source_authoring" / row["root_id"] / "OUTCOME.json"
            recovery_path = RECOVERY / row["root_id"] / "ROOT_RECOVERY.json"
            if not original_path.exists() or not recovery_path.exists():
                raise RecoveryError("All 16 original and recovery outcomes are required before publication")
            original_outcomes.append(bind(original_path))
            recovered.append(read(recovery_path))
        if any(x.get("status") not in ("PASS", "PASS_ORIGINAL_SOURCE_REUSED", "FAILED_MECHANICAL", "FAILED_REVIEW", "FAILED_CALL_OR_ENGINEERING")
               for x in recovered):
            raise RecoveryError("Every recovery outcome must be final before publication")

        passing = [x for x in recovered if x["status"] in ("PASS", "PASS_ORIGINAL_SOURCE_REUSED")]
        by_id = {x["root_id"]: x for x in recovered}
        selection = {"schema_version": "closed_loop_source_selection_v1", "split": "train",
                     "roots": [{"root_id": x["root_id"], **x["source_binding"]} for x in passing]}
        selection_path = BASE / "SOURCE_SELECTION_RECOVERY.json"
        selection_binding = write_immutable(selection_path, selection)
        failures = {"schema_version": "fgr20_source_failures_v1", "planned_roots": 16,
                    "failed_roots": [{"root_id": row["root_id"], "recovery_outcome_binding": bind(RECOVERY / row["root_id"] / "ROOT_RECOVERY.json"),
                                      "original_outcome_binding": bind(BASE / "source_authoring" / row["root_id"] / "OUTCOME.json"),
                                      "status": by_id[row["root_id"]]["status"],
                                      "failure_reason": by_id[row["root_id"]].get("failure_reason")}
                                     for row in roots if by_id[row["root_id"]]["status"] not in ("PASS", "PASS_ORIGINAL_SOURCE_REUSED")],
                    "substitutions": [], "retries_after_terminal_outcome_permitted": False,
                    "native_reservations_remain_consumed": True,
                    "original_failed_outcomes_preserved": True,
                    "recovery_outcomes_are_separately_bound": True}
        failure_path = BASE / "SOURCE_FAILURES_RECOVERY.json"
        failure_binding = write_immutable(failure_path, failures)
        freeze = {"schema_version": "fgr20_source_freeze_v1",
                  "status": "PASS" if len(passing) == 16 else "PARTIAL",
                  "planned_root_count": 16, "passed_root_count": len(passing),
                  "failed_root_count": 16 - len(passing), "source_selection_binding": selection_binding,
                  "failures_binding": failure_binding, "parent_selection_binding": bind(BASE / "PARENT_SOURCE_SELECTION.json"),
                  "root_outcomes": [bind(RECOVERY / row["root_id"] / "ROOT_RECOVERY.json") for row in roots],
                  "original_root_outcomes": original_outcomes,
                  "training": False, "old_sources_modified": False, "original_four_rerun": False,
                  "same_shared_source_for_paired_F_G": True, "source_prefix_preservation_checked": True,
                  "source_recovery_stage": True,
                  "canonical_source_driver_artifacts_preserved": True}
        freeze_binding = write_immutable(BASE / "SOURCE_FREEZE_RECOVERY.json", freeze)
        return {"status": freeze["status"], "passed": len(passing), "failed": 16 - len(passing),
                "selection_binding": selection_binding, "failures_binding": failure_binding,
                "freeze_binding": freeze_binding}


def promote_recovery_to_canonical() -> dict[str, Any]:
    """Promote a complete recovered source freeze for the frozen solver.

    Existing source-driver artifacts are snapshotted before replacement. This
    command refuses partial recovery, active source generation, and drift.
    """
    recovery_selection = BASE / "SOURCE_SELECTION_RECOVERY.json"
    recovery_freeze = BASE / "SOURCE_FREEZE_RECOVERY.json"
    if not recovery_selection.exists() or not recovery_freeze.exists():
        raise RecoveryError("Recovery selection/freeze must be published first")
    selection = read(recovery_selection)
    freeze = read(recovery_freeze)
    if freeze.get("status") != "PASS" or len(selection.get("roots", [])) != 16:
        raise RecoveryError("Canonical promotion requires all 16 recovered roots")
    lock_path = BASE / ".source_driver.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecoveryError("source_driver is still active; canonical promotion refused") from exc
        snapshots = []
        for name in ("SOURCE_SELECTION_FINAL.json", "SOURCE_FREEZE.json", "SOURCE_FAILURES.json"):
            canonical = BASE / name
            if not canonical.exists():
                continue
            snapshot = BASE / f"SOURCE_DRIVER_INITIAL_{name}"
            if snapshot.exists() and snapshot.read_bytes() != canonical.read_bytes():
                raise RecoveryError(f"Initial canonical snapshot drift: {snapshot}")
            if not snapshot.exists():
                snapshot.write_bytes(canonical.read_bytes())
            snapshots.append({"canonical": bind(canonical), "snapshot": bind(snapshot)})
        for source_name, target_name in (("SOURCE_SELECTION_RECOVERY.json", "SOURCE_SELECTION_FINAL.json"),
                                         ("SOURCE_FREEZE_RECOVERY.json", "SOURCE_FREEZE.json"),
                                         ("SOURCE_FAILURES_RECOVERY.json", "SOURCE_FAILURES.json")):
            source = BASE / source_name
            target = BASE / target_name
            raw = source.read_bytes()
            temporary = target.with_suffix(target.suffix + ".recovery.tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, target)
        receipt = {"schema_version": "fgr20_recovery_canonical_promotion_v1",
                   "status": "PROMOTED_COMPLETE_RECOVERY", "recovery_selection": bind(recovery_selection),
                   "recovery_freeze": bind(recovery_freeze), "snapshots": snapshots,
                   "canonical_selection": bind(BASE / "SOURCE_SELECTION_FINAL.json"),
                   "canonical_freeze": bind(BASE / "SOURCE_FREEZE.json"),
                   "canonical_failures": bind(BASE / "SOURCE_FAILURES.json"),
                   "original_driver_artifacts_preserved": True}
        return write_immutable(BASE / "SOURCE_RECOVERY_CANONICAL_PROMOTION.json", receipt)


def _mock_event_rows(message: str | None, *, include_tool: bool = False) -> list[dict[str, Any]]:
    events = [{"type": "thread.started", "thread_id": "fixture-thread"}]
    if message is not None:
        events.extend({"type": "item.completed", "item": {"id": item_id, "type": "error", "message": message}}
                      for item_id in ("item_0", "item_1"))
    events.append({"type": "turn.started"})
    if include_tool:
        events.append({"type": "item.completed", "item": {"id": "t1", "type": "shell_command", "command": "forbidden"}})
    events.extend([
        {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "{}"}},
        {"type": "turn.completed", "usage": {k: 1 for k in USAGE_FIELDS}},
    ])
    return events


def _write_fixture_attempt(parent: Path, events: list[dict[str, Any]], *, forbidden: list[dict[str, Any]]):
    attempt = parent / "attempt_01"
    attempt.mkdir(parents=True)
    request = attempt / "REQUEST.txt"
    request.write_text("You are the source reviewer. Do not use ANY tools, commands, files, network.\n")
    raw = attempt / "LAST_MESSAGE.txt"
    raw.write_text("{}")
    event_path = attempt / "EVENTS.jsonl"
    event_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    (attempt / "STDERR.log").write_text("")
    start = {"fresh_session": True, "tool_calls_permitted": False, "request": bind(request)}
    (attempt / "START.json").write_text(json.dumps(start))
    usage = next(e["usage"] for e in events if e.get("type") == "turn.completed")
    thread = next(e["thread_id"] for e in events if e.get("type") == "thread.started")
    terminal = {"returncode": 0, "timed_out": False, "stopped": False,
                "request": bind(request), "events": bind(event_path), "raw_response": bind(raw),
                "stderr": bind(attempt / "STDERR.log"),
                "backend": {"thread_id": thread, "model": "fixture-model", "model_provider": "fixture",
                            "effort": "fixture", "cli_version": "fixture", "model_override": None},
                "usage": usage, "forbidden_tool_items": forbidden}
    (attempt / "TERMINAL.json").write_text(json.dumps(terminal))
    return attempt, terminal


def smoke_test() -> dict[str, Any]:
    """Read-only real fixture plus temp-only mocked clean/diagnostic/unknown cases."""
    fixture_root = "cc-train-episode-55788"
    call_id = "cc_train_episode_55788_source_draft"
    real = classify_saved_attempt(call_id, CALLS / call_id, CALLS / call_id / "attempt_01",
                                  expected_root_id=fixture_root)
    if real["classification"] != "STARTUP_IGNORED_LOCAL_CONFIG_DIAGNOSTICS_ONLY":
        raise AssertionError("Existing 55788 fixture not accepted")
    message = (HEADER + "\n  user (/root/.codex/config.toml): `disable_response_storage` is ignored."
               "\n  user (/root/.codex/config.toml): `network_access` is ignored."
               "\n  user (/root/.codex/config.toml): `windows_wsl_setup_acknowledged` is ignored.")
    # Exercise the same production event classifier used by recovery.  A fake
    # codex_calls module cannot receive question_driver's module-level adapter,
    # so keep this smoke test focused on the classifier's fail-closed contract.
    import sys
    scripts = str(BASE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import question_driver

    with tempfile.TemporaryDirectory(prefix="fgr20_source_recovery_smoke_") as tmp:
        tmp = Path(tmp)
        clean_events = _mock_event_rows(None)
        clean_attempt, clean_terminal = _write_fixture_attempt(tmp / "clean", clean_events, forbidden=[])
        clean_result = question_driver.classify_cli_events(clean_attempt / "EVENTS.jsonl", clean_terminal)
        if clean_result["status"] != "PASS" or clean_result["classification"] != "no_forbidden_items":
            raise AssertionError("Clean response was not accepted unchanged")

        recognized_events = _mock_event_rows(message)
        recognized_forbidden = [{"event_type": "item.completed", "item_type": "error", "id": item_id}
                                for item_id in ("item_0", "item_1")]
        recognized_attempt, recognized_terminal = _write_fixture_attempt(
            tmp / "recognized", recognized_events, forbidden=recognized_forbidden)
        recognized_result = question_driver.classify_cli_events(
            recognized_attempt / "EVENTS.jsonl", recognized_terminal)
        if recognized_result["status"] != "PASS" or recognized_result["classification"] != "allowlisted_exact_codex_startup_diagnostics":
            raise AssertionError("Recognized startup warning was not allowlisted")
        if recognized_terminal["forbidden_tool_items"] != recognized_forbidden:
            raise AssertionError("Classifier modified original terminal evidence")

        unknown_events = _mock_event_rows("Different failure: unknown key or service error")
        unknown_forbidden = [{"event_type": "item.completed", "item_type": "error", "id": "e1"}]
        unknown_attempt, unknown_terminal = _write_fixture_attempt(
            tmp / "unknown", unknown_events, forbidden=unknown_forbidden)
        try:
            question_driver.classify_cli_events(unknown_attempt / "EVENTS.jsonl", unknown_terminal)
        except (OSError, KeyError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("Unknown error item was incorrectly accepted")

        mixed_events = _mock_event_rows(message, include_tool=True)
        mixed_forbidden = [*recognized_forbidden,
                           {"event_type": "item.completed", "item_type": "shell_command", "id": "t1"}]
        mixed_attempt, mixed_terminal = _write_fixture_attempt(tmp / "mixed", mixed_events, forbidden=mixed_forbidden)
        try:
            question_driver.classify_cli_events(mixed_attempt / "EVENTS.jsonl", mixed_terminal)
        except (OSError, KeyError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("Mixed diagnostic/tool stream was incorrectly accepted")
    return {"status": "PASS", "existing_saved_fixture": fixture_root,
            "existing_fixture_classification": real["classification"],
            "mocked_clean": "ACCEPTED_BY_CLASSIFIER", "mocked_exact_warning": "ALLOWLISTED",
            "mocked_unknown_error": "REJECTED_BY_CLASSIFIER",
            "mocked_mixed_tool_event": "REJECTED_BY_CLASSIFIER",
            "model_calls": 0, "task_artifacts_written": False}


def audit_roots(numeric_ids: list[str]) -> dict[str, Any]:
    rows = []
    for numeric_id in numeric_ids:
        root_id = f"cc-train-episode-{int(numeric_id)}"
        call_id = f"cc_train_episode_{int(numeric_id)}_source_draft"
        try:
            result = classify_saved_attempt(call_id, CALLS / call_id, CALLS / call_id / "attempt_01",
                                            expected_root_id=root_id)
            rows.append({"root_id": root_id, "status": "ELIGIBLE_FOR_SAVED_DRAFT_RECOVERY",
                         "classification": result["classification"],
                         "evidence": result["evidence"], "usage": result["usage"],
                         "thread_id": result["thread_id"], "ignored_config_keys": result["ignored_config_keys"]})
        except Exception as exc:
            rows.append({"root_id": root_id, "status": "REFUSED", "reason": f"{type(exc).__name__}: {exc}"})
    return {"schema_version": "fgr20_source_recovery_read_only_audit_v1", "rows": rows,
            "files_written": False, "model_calls": 0, "root_outcomes_modified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit-roots", nargs="+", metavar="NUMERIC_ROOT_ID")
    group.add_argument("--recover-roots", nargs="+", metavar="NUMERIC_ROOT_ID")
    group.add_argument("--recover-all", action="store_true")
    group.add_argument("--publish-canonical-freeze", action="store_true")
    group.add_argument("--promote-recovery-to-canonical", action="store_true")
    group.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        result = smoke_test()
    elif args.audit_roots:
        result = audit_roots(args.audit_roots)
    elif args.recover_roots or args.recover_all:
        sd = source_driver_module()
        roots = sd.validate_preparation()
        ids = ([row["root_id"].rsplit("-", 1)[1] for row in roots]
               if args.recover_all else args.recover_roots)
        numeric = [int(x) for x in ids]
        if len(set(numeric)) != len(numeric):
            raise RecoveryError("Duplicate root IDs are not allowed")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outcomes = list(pool.map(recover_root, [f"cc-train-episode-{n}" for n in numeric]))
        result = {"schema_version": "fgr20_source_recovery_batch_v1", "outcomes": outcomes,
                  "counts": {status: sum(x.get("status") == status for x in outcomes)
                             for status in sorted({x.get("status") for x in outcomes})},
                  "canonical_freeze_published": False}
    elif args.publish_canonical_freeze:
        result = publish_canonical_freeze()
    elif args.promote_recovery_to_canonical:
        result = promote_recovery_to_canonical()
    else:
        result = smoke_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
