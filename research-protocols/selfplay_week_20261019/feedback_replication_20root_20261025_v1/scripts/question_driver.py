#!/usr/bin/env python3
"""Fresh-session F/G question authoring and review for the frozen 20-root study.

This driver only coordinates codex_calls sessions and mechanical validation. It
does not author questions, Gold keys, or semantic review decisions itself.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import codex_calls as cc

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
SOURCE_SELECTION_PATH = BASE / "SOURCE_SELECTION_FINAL.json"
SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
EXTRA_DRIVER_BINDINGS: list[Path] = []
GATE_DIR = REPO / "scripts" / "current_method"
sys.path.insert(0, str(GATE_DIR))
import production_question_gate as gate  # noqa: E402

SLOTS = {
    "MC2": ("multi_select", "MC_QUANTITATIVE_RELATION", 2),
    "MC3": ("multi_select", "MC_CONSTRAINT_CONJUNCTION", 3),
    "MC4": ("multi_select", "MC_EVIDENCE_SELECTION", 4),
    "MC5": ("multi_select", "MC_CAUSAL_SUPPORT", 5),
    "MC6": ("multi_select", "MC_CONSTRAINT_CONJUNCTION", 6),
    "ORDER1": ("ordering", "ORDER_EXPLICIT_TIME", None),
    "ORDER2": ("ordering", "ORDER_EXPLICIT_PROCEDURE", None),
    "ORDER3": ("ordering", "ORDER_DISCOURSE_RECONSTRUCTION", None),
}
ARMS = ("F", "G")
LABELS = list("ABCDEF")

_CONFIG_WARNING = (
    "Codex is ignoring 3 unrecognized configuration settings. Check for typos or deprecated settings.\n"
    "  user (/root/.codex/config.toml): `disable_response_storage` is ignored.\n"
    "  user (/root/.codex/config.toml): `network_access` is ignored.\n"
    "  user (/root/.codex/config.toml): `windows_wsl_setup_acknowledged` is ignored."
)


def classify_cli_events(events_path: Path, terminal: dict[str, Any]) -> dict[str, Any]:
    """Allow only the exact duplicated Codex config diagnostics seen at startup."""
    if cc.bind(events_path) != terminal.get("events"):
        raise ValueError("Codex EVENTS binding mismatch")
    events = []
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"Malformed Codex event line {line_number}") from exc
    if sum(event.get("type") == "thread.started" for event in events) != 1:
        raise ValueError("Expected one fresh Codex thread.started event")
    started = next(event for event in events if event.get("type") == "thread.started")
    if started.get("thread_id") != terminal.get("backend", {}).get("thread_id"):
        raise ValueError("EVENTS thread id does not match terminal backend")
    if any(event.get("type") == "error" for event in events):
        raise ValueError("Unexpected top-level Codex error event")

    forbidden = []
    allowed_diagnostics = []
    for event in events:
        if event.get("type") not in {"item.started", "item.updated", "item.completed"}:
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type in {"agent_message", "reasoning"}:
            continue
        row = {"event_type": event.get("type"), "item_type": item_type, "id": item.get("id")}
        forbidden.append(row)
        if (event.get("type") != "item.completed" or item_type != "error"
                or item.get("id") not in {"item_0", "item_1"}
                or item.get("message") != _CONFIG_WARNING):
            continue
        allowed_diagnostics.append({**row, "message": item["message"]})

    if forbidden != terminal.get("forbidden_tool_items", []):
        raise ValueError("Terminal forbidden-item list differs from raw EVENTS")
    diagnostic_ids = [row["id"] for row in allowed_diagnostics]
    if diagnostic_ids not in ([], ["item_0", "item_1"]):
        raise ValueError("Only the exact item_0/item_1 startup diagnostic pair is allowlisted")
    if len(allowed_diagnostics) != len(forbidden):
        raise ValueError("Unexpected forbidden Codex item; no-tools contract remains fail-closed")

    completed = [event for event in events if event.get("type") == "turn.completed"]
    if len(completed) != 1 or completed[0].get("usage") != terminal.get("usage"):
        raise ValueError("Expected one completed turn with usage matching TERMINAL")
    messages = [event["item"].get("text", "") for event in events
                if event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "agent_message"]
    if not messages:
        raise ValueError("No completed agent_message event")
    return {"schema_version": "fgr20_cli_event_classification_v1",
            "status": "PASS", "classification": ("allowlisted_exact_codex_startup_diagnostics"
                if allowed_diagnostics else "no_forbidden_items"),
            "allowed_startup_diagnostics": allowed_diagnostics,
            "raw_helper_forbidden_tool_items_preserved": forbidden,
            "events_binding": terminal["events"], "stderr_binding": terminal["stderr"],
            "request_binding": terminal["request"], "raw_response_binding": terminal.get("raw_response"),
            "thread_id": terminal["backend"]["thread_id"], "usage": terminal.get("usage")}


def _finish_saved_with_event_adapter(call_id: str, cdir: Path, attempt: Path,
                                    terminal: dict[str, Any]):
    """Finish a helper call while preserving strict validation and startup evidence."""
    raw = attempt / "LAST_MESSAGE.txt"
    classification_path = attempt / "EVENT_CLASSIFICATION.json"
    try:
        start = cc.read(attempt / "START.json")
        for name in ("events", "stderr", "request"):
            path = attempt / {"events": "EVENTS.jsonl", "stderr": "STDERR.log", "request": "REQUEST.txt"}[name]
            if cc.bind(path) != terminal.get(name):
                raise ValueError(f"Codex {name.upper()} binding mismatch")
        if start.get("request") != terminal.get("request"):
            raise ValueError("START and TERMINAL request bindings differ")
        if terminal.get("stopped") or terminal.get("timed_out") or terminal.get("returncode") != 0:
            raise ValueError("Incomplete returned semantic response / interrupted process")
        backend = terminal.get("backend", {})
        if not all(backend.get(key) for key in ("thread_id", "model", "model_provider", "effort", "cli_version")):
            raise ValueError("Missing actual Codex backend metadata")
        classification = classify_cli_events(attempt / "EVENTS.jsonl", terminal)
        if not raw.exists() or cc.bind(raw) != terminal.get("raw_response"):
            raise ValueError("Raw response missing or binding drift")
        raw_text = raw.read_text(encoding="utf-8").strip()
        message_rows = [json.loads(line) for line in (attempt / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
        final_messages = [event["item"].get("text", "") for event in message_rows
                          if event.get("type") == "item.completed"
                          and event.get("item", {}).get("type") == "agent_message"]
        if (not raw_text or not isinstance(final_messages[-1], str)
                or raw_text != final_messages[-1].strip()):
            raise ValueError("LAST_MESSAGE differs from final completed agent_message")
        if raw_text.startswith("```") and raw_text.endswith("```"):
            parsed_text = "\n".join(raw_text.splitlines()[1:-1])
        else:
            parsed_text = raw_text
        parsed = json.loads(parsed_text)
        if not isinstance(parsed, dict):
            raise ValueError("JSON object required")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        failure_path = cdir / "SEMANTIC_FAILURE.json"
        failure = {"error": f"{type(exc).__name__}: {exc}",
                   "terminal": cc.bind(attempt / "TERMINAL.json"),
                   "at_utc": cc.utc(), "semantic_attempt_consumed": True,
                   "recovery_classification": "CALL_EVENT_ADAPTER_FAIL_CLOSED"}
        if not failure_path.exists():
            cc.write(failure_path, failure)
        raise RuntimeError(f"Failed semantic attempt, no hidden retry: {call_id}: {exc}") from exc

    classification["terminal_binding"] = cc.bind(attempt / "TERMINAL.json")
    if classification_path.exists():
        if cc.read(classification_path) != classification:
            raise RuntimeError("Saved event classification drift")
    else:
        cc.write(classification_path, classification)
    parsed_path = cdir / "PARSED_RESPONSE.json"
    if parsed_path.exists():
        if cc.read(parsed_path) != parsed:
            raise RuntimeError("Saved parsed response drift")
    else:
        cc.write(parsed_path, parsed)
    result = {**terminal, "parsed_response": cc.bind(parsed_path),
              "terminal_binding": cc.bind(attempt / "TERMINAL.json"),
              "response_received": True, "event_classification": classification,
              "event_classification_binding": cc.bind(classification_path)}
    result_path = cdir / "RESULT.json"
    if result_path.exists():
        if cc.read(result_path) != result:
            raise RuntimeError("Saved CLI result drift")
    else:
        cc.write(result_path, result)
    cc.progress("CLI_COMPLETE", call_id=call_id, thread_id=backend["thread_id"],
                event_classification=classification["classification"])
    return parsed, result


if not getattr(cc, "_fgr20_event_adapter_installed", False):
    cc.finish_saved = _finish_saved_with_event_adapter
    cc._fgr20_event_adapter_installed = True


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(stable(value).encode("utf-8")).hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    return cc.bind(path)


def verify_binding(value: dict[str, Any]) -> None:
    actual = binding(value["path"])
    expected = {key: value.get(key) for key in ("path", "sha256", "bytes")}
    if actual != expected:
        raise RuntimeError(f"Immutable binding drift: {value.get('path')}")


def publish(path: str | Path, value: Any) -> dict[str, Any]:
    """Write once; an equal existing artifact is a resumable result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Immutable artifact drift: {path}")
    else:
        cc.write(path, value)
    return binding(path)


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def root_id(value: str) -> str:
    return value if value.startswith("cc-train-episode-") else f"cc-train-episode-{value}"


def root_number(value: str) -> str:
    return root_id(value).removeprefix("cc-train-episode-")


def call_id(stage: str, arm: str, root: str, suffix: str = "") -> str:
    stem = f"q20_{stage}_{arm.lower()}_{root_number(root)}"
    return stem + (f"_{suffix}" if suffix else "")


def extract_list(value: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        for key in keys:
            rows = value.get(key)
            if isinstance(rows, list):
                return [v for v in rows if isinstance(v, dict)]
    return []


def source_warnings(source: dict[str, Any]) -> list[Any]:
    provenance = source.get("closed_loop_provenance", {})
    rows = provenance.get("warnings", []) if isinstance(provenance, dict) else []
    if not rows:
        rows = source.get("warnings", [])
    return rows if isinstance(rows, list) else []


def _nested_material_keys(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Return paths whose field names can carry arm-specific material."""
    forbidden = ("feedback", "adapter", "public_card", "f_card", "prior_question",
                 "solver_result", "solver_output", "f_material", "error_analysis")
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = path + (str(key),)
            if any(token in key_text for token in forbidden):
                found.append(".".join(child_path))
            found.extend(_nested_material_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_nested_material_keys(child, path + (str(index),)))
    return found


def _source_outcome_records(freeze: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Read every bound source outcome, including recovery continuation outcomes."""
    records: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for field in ("root_outcomes", "original_root_outcomes", "continuation_outcomes",
                  "continuation_v2_outcomes"):
        rows = freeze.get(field, [])
        if not isinstance(rows, list):
            continue
        for outcome_binding in rows:
            if not isinstance(outcome_binding, dict):
                raise ValueError(f"Source freeze has malformed {field} binding")
            verify_binding(outcome_binding)
            path = str(outcome_binding["path"])
            if path in seen:
                continue
            seen.add(path)
            outcome = read(path)
            outcome_id = root_id(str(outcome.get("root_id", "")))
            records.append((outcome_id, path, outcome))
    return records


def _validate_extended_source(rid: str, entry: dict[str, Any], source: dict[str, Any],
                              outcome_records: list[tuple[str, str, dict[str, Any]]]) -> None:
    """Reject parent-source substitutions and sources detached from a PASS outcome."""
    source_path = Path(entry["path"]).resolve()
    try:
        source_path.relative_to(BASE.resolve())
    except ValueError as exc:
        raise ValueError(f"Selected source is outside task directory: {rid}") from exc
    if "initial_sources" in source_path.parts:
        raise ValueError(f"Selected source is a frozen parent, not an R03 extension: {rid}")
    if source.get("schema_version") != "feedback_closed_loop_r03_source_v1":
        raise ValueError(f"Selected source is not an R03 extended source: {rid}")
    if source.get("root_id") != rid or source.get("scene_ref") != "R03:B:S03":
        raise ValueError(f"Selected source identity/scene mismatch: {rid}")
    if source.get("source_status") != "GENERATED_EXTENSION_ON_SOURCE_GROUNDED_PARENT":
        raise ValueError(f"Selected source status is not an extension: {rid}")
    provenance = source.get("closed_loop_provenance")
    if not isinstance(provenance, dict) or provenance.get("same_source_for_F_and_G") is not True:
        raise ValueError(f"Selected source lacks paired F/G provenance: {rid}")
    if not isinstance(provenance.get("parent_source_binding"), dict):
        raise ValueError(f"Selected source lacks parent binding: {rid}")
    if not isinstance(provenance.get("review_binding"), dict):
        raise ValueError(f"Selected source lacks independent review binding: {rid}")
    if not isinstance(provenance.get("native_events"), list) or len(provenance["native_events"]) != 3:
        raise ValueError(f"Selected source lacks the three reserved native events: {rid}")
    baseline = source.get("baseline_turn_count")
    cumulative = source.get("cumulative_new_turn_count")
    new_turns = source.get("new_turn_count")
    turns = source.get("turns")
    if (not isinstance(baseline, int) or not isinstance(cumulative, int) or new_turns != 48
            or not isinstance(turns, list) or len(turns) != baseline + cumulative):
        raise ValueError(f"Selected source turn-count contract failed: {rid}")
    previous = source.get("previous_turn_count")
    if not isinstance(previous, int) or previous + 48 != len(turns):
        raise ValueError(f"Selected source appended-turn count is inconsistent: {rid}")
    new_rows = turns[previous:]
    refs = [row.get("turn_ref") for row in new_rows if isinstance(row, dict)]
    if (len(new_rows) != 48 or len(refs) != 48 or len(set(refs)) != 48
            or any(not isinstance(ref, str) or not ref.startswith("R03:") for ref in refs)):
        raise ValueError(f"Selected source does not contain 48 unique R03 turns: {rid}")

    parent_selection = read(BASE / "PARENT_SOURCE_SELECTION.json")
    parent_rows = [row for row in parent_selection.get("roots", [])
                   if isinstance(row, dict) and root_id(str(row.get("root_id", ""))) == rid]
    if len(parent_rows) != 1:
        raise ValueError(f"Frozen parent selection does not uniquely include {rid}")
    parent_binding = parent_rows[0].get("frozen_initial_source_binding")
    if not isinstance(parent_binding, dict):
        raise ValueError(f"Frozen parent selection lacks its task-local input binding: {rid}")
    if provenance["parent_source_binding"] != parent_binding:
        raise ValueError(f"Selected source parent binding differs from frozen parent: {rid}")
    verify_binding(parent_binding)
    parent_source = read(parent_binding["path"])
    if parent_source.get("root_id") != rid or turns[:len(parent_source.get("turns", []))] != parent_source.get("turns"):
        raise ValueError(f"Selected source does not preserve its exact frozen parent prefix: {rid}")
    if previous != len(parent_source.get("turns", [])):
        raise ValueError(f"Selected source previous_turn_count differs from parent length: {rid}")

    review_binding = provenance["review_binding"]
    verify_binding(review_binding)
    review = read(review_binding["path"])
    if (review.get("root_id") != rid or review.get("decision") != "PASS"
            or review.get("complete_inputs_read") is not True):
        raise ValueError(f"Selected source review is not a complete independent PASS: {rid}")

    selected_binding = binding(entry["path"])
    matches = []
    for outcome_id, _, outcome in outcome_records:
        if outcome_id != rid:
            continue
        if outcome.get("status") != "PASS":
            continue
        bound = outcome.get("source_binding")
        if isinstance(bound, dict) and bound == selected_binding:
            matches.append(outcome)
    if not matches:
        raise ValueError(f"Selected source is not tied to a PASS source outcome: {rid}")
    for outcome in matches:
        final_review = outcome.get("final_review_binding")
        if isinstance(final_review, dict) and final_review != review_binding:
            raise ValueError(f"Selected source review binding differs from PASS outcome: {rid}")


def validate_source_freeze(selection_path: Path, freeze_path: Path,
                           selection: dict[str, Any]) -> dict[str, Any]:
    """Require the selected roots to be exactly those named by the source freeze."""
    if not freeze_path.is_file():
        raise ValueError(f"Missing source freeze: {freeze_path}")
    freeze = read(freeze_path)
    freeze_selection = freeze.get("source_selection_binding")
    if not isinstance(freeze_selection, dict):
        raise ValueError("Source freeze has no selection binding")
    expected = binding(selection_path)
    if freeze_selection != expected:
        raise ValueError("Source freeze selection binding does not match selected source file")
    planned = freeze.get("planned_root_count")
    passed = freeze.get("passed_root_count")
    failed = freeze.get("failed_root_count")
    if (not isinstance(planned, int) or planned != 16 or not isinstance(passed, int)
            or not isinstance(failed, int)):
        raise ValueError("Source freeze root counts are missing or inconsistent")
    roots = selection.get("roots")
    if not isinstance(roots, list):
        raise ValueError("Source selection roots must be a list")
    ids = [root_id(row.get("root_id", "")) for row in roots if isinstance(row, dict)]
    if len(ids) != len(roots) or len(set(ids)) != len(ids):
        raise ValueError("Source selection contains duplicate or malformed root ids")
    if passed != len(ids) or passed > planned or failed != planned - passed:
        raise ValueError("Source freeze passed count does not equal selected unique roots")
    config = read(BASE / "CONFIG.json")
    configured_order = [root_id(value) for value in config.get("roots", [])]
    configured = set(configured_order)
    if len(configured_order) != planned or not set(ids).issubset(configured):
        raise ValueError("Source selection contains a root outside frozen CONFIG")
    if ids != [rid for rid in configured_order if rid in set(ids)]:
        raise ValueError("Source selection does not preserve frozen root order")
    expected_status = "PASS" if failed == 0 else "PARTIAL"
    if freeze.get("status") != expected_status:
        raise ValueError("Source freeze status disagrees with its selected/failed root counts")
    failure_binding = freeze.get("failures_binding")
    if isinstance(failure_binding, dict):
        verify_binding(failure_binding)
    outcomes = freeze.get("root_outcomes")
    if outcomes is not None:
        outcome_ids = []
        for outcome_binding in outcomes:
            verify_binding(outcome_binding)
            outcome = read(outcome_binding["path"])
            outcome_id = root_id(str(outcome.get("root_id", "")))
            if outcome_id not in configured:
                raise ValueError("Source freeze contains an outcome for an unplanned root")
            outcome_ids.append(outcome_id)
        if len(outcome_ids) != planned or len(set(outcome_ids)) != planned or set(outcome_ids) != configured:
            raise ValueError("Source freeze does not bind exactly one outcome per planned root")
    outcome_records = _source_outcome_records(freeze)
    if not outcome_records:
        raise ValueError("Source freeze has no bound source outcomes")
    return freeze


def load_selection(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = read(path)
    if document.get("split") != "train" or not isinstance(document.get("roots"), list):
        raise ValueError("SOURCE_SELECTION_FINAL must be a Train selection")
    freeze = validate_source_freeze(path, SOURCE_FREEZE_PATH, document)
    outcome_records = _source_outcome_records(freeze)
    sources = {}
    for entry in document["roots"]:
        rid = root_id(entry["root_id"])
        verify_binding(entry)
        source = read(entry["path"])
        if source.get("root_id") != rid or not isinstance(source.get("turns"), list):
            raise ValueError(f"Selected source root/turn mismatch: {rid}")
        _validate_extended_source(rid, entry, source, outcome_records)
        sources[rid] = {"entry": entry, "source": source, "binding": binding(entry["path"])}
    return document, sources


def source_view(row: dict[str, Any]) -> dict[str, Any]:
    return {"root_id": row["source"]["root_id"], "source": row["source"],
            "source_status": row["source"].get("source_status"),
            "warnings": source_warnings(row["source"]),
            "source_binding": row["binding"]}


def feedback_material() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    feedback_path = BASE / "question_authoring" / "F" / "PUBLIC_FEEDBACK_INPUT.json"
    adapter_path = BASE / "FEEDBACK_TRANSFER_ADAPTER_ZH.md"
    feedback = read(feedback_path)
    adapter = adapter_path.read_text(encoding="utf-8")
    cards = feedback.get("cards") if isinstance(feedback, dict) else None
    if not isinstance(cards, list) or len(cards) != 19:
        raise ValueError("Frozen F input must contain all 19 public cards")
    return feedback, adapter, binding(feedback_path), binding(adapter_path)


def prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def identity_prompt(role: str, allowed: str) -> str:
    return cc.prefix(role) + (
        "All semantic work must use only the embedded inputs. Return one JSON object and no prose. "
        "Do not use tools or infer missing source facts. " + allowed + "\n"
    )


def task_plan_prompt(arm: str, root: str, view: dict[str, Any], contract: str,
                     feedback: dict[str, Any] | None, adapter: str | None) -> str:
    rules = (
        "Create a frozen TASK_PLANS response with a `plans` array containing one plan for each assigned slot. "
        "The exact slots are MC2, MC3, MC4, MC5, MC6, ORDER1, ORDER2, ORDER3. For every plan give slot_id, "
        "task_operation exactly as specified in the contract, source_relations as exact turn_ref values, "
        "necessary_r03_premises with turn_refs and a concrete removal_consequence, r03_necessity, and "
        "quality_acceptance. Do not write any question, option, Gold, or answer key. Preserve source_status "
        "and warnings; do not infer an event from an unresolved warning.\n"
    )
    payload: dict[str, Any] = {"assignment": {"root_id": root, "arm": arm, "slots": list(SLOTS)},
                               "common_contract": contract, "source_and_warnings": view}
    if arm == "F":
        if feedback is None or adapter is None:
            raise AssertionError("F plan requires the complete frozen public feedback and adapter")
        rules += (
            "For each F slot, add feedback_target containing feedback_id, trigger_realization, "
            "predicted_observable_error, counterexample, target_acceptance_test, and target_status. "
            "Use only an original feedback_id in the supplied 19-card library. If this source/slot cannot "
            "faithfully instantiate a narrow trigger, set target_status=TARGET_HOLD and explain in "
            "trigger_realization; do not broaden the card or claim a reproduced weakness. These are "
            "prospective hypotheses.\n"
        )
        payload["public_feedback_input_19_cards"] = feedback
        payload["feedback_transfer_adapter"] = adapter
    else:
        rules += "This is G. No feedback, prior questions, outcomes, or F material is authorized.\n"
    prompt = identity_prompt(f"isolated {arm} question task planner", rules) + "FROZEN_INPUTS:\n" + prompt_json(payload)
    if arm == "G":
        validate_g_prompt(prompt, "task_planner", root)
    return prompt


def validate_plans(response: Any, arm: str, root: str, source: dict[str, Any],
                   feedback_ids: set[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rows = extract_list(response, "plans", "task_plans")
    by_slot: dict[str, dict[str, Any]] = {}
    available_refs = {t.get("turn_ref") for t in source.get("turns", []) if isinstance(t, dict)}
    for plan in rows:
        slot = plan.get("slot_id")
        if slot not in SLOTS or slot in by_slot:
            errors.append(f"invalid_or_duplicate_slot:{slot}")
            continue
        if arm == "G":
            leaked_paths = _nested_material_keys(plan)
            if leaked_paths:
                errors.append(f"unauthorized_nested_feedback_material_G:{','.join(leaked_paths)}:{slot}")
            allowed_fields = {"slot_id", "task_operation", "source_relations",
                              "necessary_r03_premises", "r03_necessity", "quality_acceptance"}
            extra_fields = sorted(set(plan) - allowed_fields)
            if extra_fields:
                errors.append(f"unauthorized_g_plan_fields:{','.join(extra_fields)}:{slot}")
        by_slot[slot] = copy.deepcopy(plan)
    for slot, (_, operation, _) in SLOTS.items():
        plan = by_slot.get(slot)
        if not plan:
            errors.append(f"missing_plan:{slot}")
            continue
        if plan.get("task_operation") != operation:
            errors.append(f"task_operation_drift:{slot}")
        refs = plan.get("source_relations")
        if not isinstance(refs, list) or not refs or any(ref not in available_refs for ref in refs):
            errors.append(f"invalid_source_relations:{slot}")
        necessary = plan.get("necessary_r03_premises")
        if not isinstance(necessary, dict) or not isinstance(necessary.get("turn_refs"), list) or not necessary["turn_refs"] or not isinstance(necessary.get("removal_consequence"), str) or not necessary["removal_consequence"].strip():
            errors.append(f"incomplete_r03_plan:{slot}")
        elif any(ref not in available_refs or not str(ref).startswith("R03:") for ref in necessary["turn_refs"]):
            errors.append(f"invalid_r03_refs:{slot}")
        if not isinstance(plan.get("r03_necessity"), str) or not plan["r03_necessity"].strip():
            errors.append(f"missing_r03_necessity:{slot}")
        if not isinstance(plan.get("quality_acceptance"), str) or not plan["quality_acceptance"].strip():
            errors.append(f"missing_quality_acceptance:{slot}")
        if arm == "F":
            target = plan.get("feedback_target")
            required = ("feedback_id", "trigger_realization", "predicted_observable_error", "counterexample", "target_acceptance_test", "target_status")
            if not isinstance(target, dict) or any(not isinstance(target.get(k), str) or not target[k].strip() for k in required):
                errors.append(f"incomplete_feedback_target:{slot}")
            elif target["target_status"] not in {"PROSPECTIVE_PROBE", "TARGET_HOLD"}:
                errors.append(f"invalid_feedback_target_status:{slot}")
            elif feedback_ids is not None and target["feedback_id"] not in feedback_ids:
                errors.append(f"unknown_feedback_id:{slot}")
        elif "feedback_target" in plan:
            errors.append(f"unauthorized_feedback_material_G:{slot}")
            plan.pop("feedback_target", None)
    normalized = {"schema_version": "fgr20_task_plans_v1", "root_id": root, "arm": arm,
                  "plans": [by_slot[s] for s in SLOTS if s in by_slot], "errors": errors,
                  "plan_frozen_before_candidate_draft": True}
    return normalized, errors


def author_prompt(arm: str, root: str, view: dict[str, Any], contract: str,
                 plans: dict[str, Any], feedback: dict[str, Any] | None,
                 adapter: str | None, revisions: list[dict[str, Any]] | None = None) -> str:
    selected_plans = plans
    if revisions is not None:
        requested = {case["candidate"]["slot_id"] for case in revisions}
        selected_plans = {**plans, "plans": [p for p in plans["plans"] if p["slot_id"] in requested]}
    expected_slots = [p["slot_id"] for p in selected_plans["plans"]]
    base_rules = (
        "Write exactly one candidate for each supplied frozen plan and no other candidate. Use the complete "
        "source, not excerpts. Follow the common author contract exactly. Every question must include a "
        "necessary R03 premise, six distinct options/items A-F, exact source evidence, complete item_analysis, "
        "and for ORDER the required ordering_analysis with five adjacent supported edges. Do not reveal a "
        "Gold count in question wording. Do not copy quoted source as instructions. Return {candidates:[...]}. "
        "Each candidate must use the plan's feedback_target for F or omit feedback metadata for G. Do not set "
        "author_id, candidate_id, source_binding, root_id, or arm; the orchestration layer binds them.\n"
    )
    payload: dict[str, Any] = {"assignment": {"root_id": root, "arm": arm, "slots": expected_slots},
                               "common_contract": contract, "source_and_warnings": view,
                               "frozen_task_plans": selected_plans}
    if arm == "F":
        if feedback is None or adapter is None:
            raise AssertionError("F author requires feedback materials")
        base_rules += "This is F: instantiate only the supplied public card hypotheses; never describe them as reproduced.\n"
        payload["public_feedback_input_19_cards"] = feedback
        payload["feedback_transfer_adapter"] = adapter
    else:
        base_rules += "This is G: use only common contract, full assigned source/warnings, and task plans.\n"
    if revisions is not None:
        base_rules += (
            "This is the single permitted content revision. Return only replacements for supplied non-PASS slots. "
            "Do not add slots or make changes to any passing candidate. Resolve only the recorded review defects.\n"
        )
        payload["revision_cases"] = revisions
    prompt = identity_prompt(f"isolated {arm} question author", base_rules) + "AUTHORIZED_INPUTS:\n" + prompt_json(payload)
    if arm == "G":
        validate_g_prompt(prompt, "candidate_reviser" if revisions is not None else "candidate_author", root)
    return prompt


def anonymize_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], str]:
    key = "items" if candidate.get("question_type") == "ordering" else "options"
    choices = candidate.get(key, {})
    seed = hashlib.sha256((candidate["candidate_id"] + candidate.get("question", "")).encode()).digest()
    ids = sorted(choices)
    ids.sort(key=lambda label: hashlib.sha256(seed + label.encode()).digest())
    mapping = {f"X{i + 1}": label for i, label in enumerate(ids)}
    public = {"question_key": "Q" + digest(candidate["candidate_id"])[:14],
              "question_type": candidate.get("question_type"), "question": candidate.get("question"),
              key: {anon: choices[label] for anon, label in mapping.items()}}
    return public, mapping, key


def blind_prompt(root: str, view: dict[str, Any], cases: list[tuple[dict[str, Any], dict[str, str], str]]) -> str:
    public_cases = [x[0] for x in cases]
    rules = (
        "Independently reconstruct the answer for every anonymous question from the COMPLETE source and warnings. "
        "You are not told whether any question is F or G, its slot, author, feedback, or author analysis. Do not "
        "guess a target. For each MC question return the exact independent_gold set and item_reasons for every "
        "anonymous option, with truth, specific rationale, and exact source evidence {turn_ref,speaker,quote}. "
        "For ordering return the full six-item independent_gold sequence and exactly five adjacent_edges matching "
        "that sequence, each with before, after, rationale, and exact source evidence. "
        "State r03_necessity with decision PASS/HOLD/FAIL, concrete removal "
        "reason and R03 turn_refs. Explicitly confirm complete_source_read=true and solver_results_read=false. "
        "Return {reviews:[{question_key,...},...]}; do not make content-admission decisions.\n"
    )
    payload = {"root_id": root, "complete_source_and_warnings": view, "anonymous_questions": public_cases}
    return identity_prompt("blind independent question-answer reconstruction reviewer", rules) + "REVIEW_INPUT:\n" + prompt_json(payload)


def verify_prompt(root: str, view: dict[str, Any],
                  cases: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> str:
    rules = (
        "Review each candidate independently for admission. Read its full source and warnings, the saved blind "
        "independent Gold with reasons, the author's Gold and item_analysis, and all mechanical audit checks. "
        "The blind Gold was produced in a separate fresh session before you saw the author's key. Check source "
        "support, exact gold_truth, option distinctness and distractor_relation (or six_gold_control), and every "
        "ordering required check, including operation alignment, complete unique total order, first relevant "
        "mentions, common-sense shortcuts, editorial cues, source dependence, and R03 necessity. Do not infer "
        "truth from the authored key or agreement alone. Explain every PASS/HOLD/FAIL with concrete source refs. "
        "If the saved blind Gold, authored key, and your reconstruction disagree, provide a substantive "
        "gold_discrepancy_resolution citing exact turn_refs and the source-supported answer; do not use a bare assertion. "
        "Return one record per question with question_key, decision, checks mapping check names to {decision,rationale}, "
        "reconstructed_gold, gold_discrepancy_resolution (string or null), r03_necessity {decision,rationale,turn_refs}, "
        "source_read_complete=true, solver_results_read=false. No F/G comparison or feedback is provided. Return {reviews:[...]}.\n"
    )
    payload = {"root_id": root, "complete_source_and_warnings": view, "review_cases": cases}
    return identity_prompt("fresh independent content and Gold verifier", rules) + "REVIEW_INPUT:\n" + prompt_json(payload)


def anonymize_mechanical_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Keep useful gate findings without exposing the candidate's arm-bearing ID."""
    hidden = {"candidate_id", "candidate_sha256", "source_binding", "selection_binding",
              "author_id", "reviewer_id", "arm", "feedback_target"}

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in hidden}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(packet)


def source_evidence_valid(value: Any, turn_index: dict[str, dict[str, Any]]) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for ev in value:
        if not isinstance(ev, dict):
            return False
        turn = turn_index.get(ev.get("turn_ref"))
        if not turn or ev.get("speaker") != turn.get("speaker"):
            return False
        quote = ev.get("quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in turn.get("text", ""):
            return False
    return True


def normalize_blind(response: Any, candidates: list[dict[str, Any]],
                    source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    rows = extract_list(response, "reviews", "answers")
    by_key = {r.get("question_key"): r for r in rows}
    normalized: dict[str, Any] = {}
    errors: list[str] = []
    turns = {t.get("turn_ref"): t for t in source.get("turns", []) if isinstance(t, dict)}
    for candidate in candidates:
        public, mapping, field = anonymize_candidate(candidate)
        key = public["question_key"]
        row = by_key.get(key)
        cid = candidate["candidate_id"]
        if not isinstance(row, dict):
            errors.append(f"blind_review_missing:{cid}")
            continue
        raw_gold = row.get("independent_gold")
        expected_count = SLOTS[candidate["slot_id"]][2] if candidate["question_type"] == "multi_select" else 6
        if (not isinstance(raw_gold, list) or len(raw_gold) != expected_count
                or not all(isinstance(x, str) for x in raw_gold)
                or len(set(raw_gold)) != len(raw_gold) or any(x not in mapping for x in raw_gold)):
            errors.append(f"blind_gold_invalid:{cid}")
            continue
        gold = [mapping[x] for x in raw_gold]
        if candidate["question_type"] == "multi_select":
            gold = sorted(gold)
        key_name = "items" if field == "items" else "options"
        reason_rows = row.get("item_reasons")
        mapped_reasons = {}
        if candidate["question_type"] == "multi_select" and (not isinstance(reason_rows, dict) or set(reason_rows) != set(mapping)):
            errors.append(f"blind_item_reasons_incomplete:{cid}")
        elif isinstance(reason_rows, dict) and set(reason_rows) == set(mapping):
            for anon, label in mapping.items():
                reason = reason_rows[anon]
                if not isinstance(reason, dict) or reason.get("truth") not in (True, False) or not isinstance(reason.get("rationale"), str) or not reason["rationale"].strip() or not source_evidence_valid(reason.get("evidence"), turns):
                    errors.append(f"blind_item_reason_invalid:{cid}:{label}")
                    continue
                if reason["truth"] is not (label in gold):
                    errors.append(f"blind_item_reason_gold_disagreement:{cid}:{label}")
                mapped_reasons[label] = {**reason, "evidence": reason["evidence"]}
        if candidate["question_type"] == "ordering":
            edges = row.get("adjacent_edges")
            if not isinstance(edges, list) or len(edges) != 5:
                errors.append(f"blind_order_edges_incomplete:{cid}")
            else:
                mapped_edges = []
                for edge in edges:
                    if (not isinstance(edge, dict) or edge.get("before") not in mapping
                            or edge.get("after") not in mapping
                            or not isinstance(edge.get("rationale"), str)
                            or not edge["rationale"].strip()
                            or not source_evidence_valid(edge.get("evidence"), turns)):
                        errors.append(f"blind_order_edge_invalid:{cid}")
                        break
                    mapped_edges.append((mapping[edge["before"]], mapping[edge["after"]]))
                expected_edges = list(zip(gold, gold[1:]))
                if mapped_edges and mapped_edges != expected_edges:
                    errors.append(f"blind_order_edges_disagree_with_gold:{cid}")
                for edge in edges:
                    if isinstance(edge, dict):
                        edge["before"] = mapping.get(edge.get("before"), edge.get("before"))
                        edge["after"] = mapping.get(edge.get("after"), edge.get("after"))
            row["adjacent_edges"] = edges if isinstance(edges, list) else []
        r03 = row.get("r03_necessity")
        if (not isinstance(r03, dict) or r03.get("decision") not in {"PASS", "HOLD", "FAIL"}
                or not isinstance(r03.get("rationale"), str) or not r03["rationale"].strip()
                or not isinstance(r03.get("turn_refs"), list) or not r03["turn_refs"]
                or any(r not in turns or not str(r).startswith("R03:") for r in r03["turn_refs"])):
            errors.append(f"blind_r03_invalid:{cid}")
        normalized[cid] = {"independent_gold": gold, "item_reasons": mapped_reasons,
                           "adjacent_edges": row.get("adjacent_edges", []), "r03_necessity": r03,
                           "complete_source_read": row.get("complete_source_read") is True,
                           "solver_results_read": row.get("solver_results_read"),
                           "question_key": key, "anonymous_label_mapping": mapping}
        if row.get("complete_source_read") is not True or row.get("solver_results_read") is not False:
            errors.append(f"blind_read_attestation_invalid:{cid}")
    return normalized, errors


def candidate_id(root: str, arm: str, slot: str) -> str:
    return f"FGR20-{arm}-{root_number(root)}-{slot}"


def normalize_candidates(response: Any, arm: str, root: str, author_thread: str,
                         source_row: dict[str, Any], plans: dict[str, Any],
                         selection_binding: dict[str, Any], revision_thread: str | None = None
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows = extract_list(response, "candidates", "questions")
    by_slot: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    logs: list[str] = []
    for index, raw in enumerate(rows):
        slot = raw.get("slot_id")
        if slot not in SLOTS:
            failures.append({"slot_id": None, "reason": "candidate_has_no_valid_slot_id", "raw_index": index})
            continue
        if slot in by_slot:
            failures.append({"slot_id": slot, "reason": "duplicate_candidate_for_slot", "raw_index": index})
            continue
        by_slot[slot] = raw
    plans_by_slot = {p["slot_id"]: p for p in plans.get("plans", [])}
    result = []
    for slot in SLOTS:
        if slot not in plans_by_slot:
            failures.append({"slot_id": slot, "reason": "frozen_plan_missing; no draft authorized"})
            continue
        if slot not in by_slot:
            failures.append({"slot_id": slot, "reason": "candidate_not_returned_by_author"})
            continue
        raw = copy.deepcopy(by_slot[slot])
        normalized = copy.deepcopy(raw)
        generated_id = candidate_id(root, arm, slot)
        for key, value in (("candidate_id", generated_id), ("root_id", root), ("arm", arm),
                           ("author_id", author_thread)):
            if normalized.get(key) != value:
                logs.append(f"{slot}:{key}:{normalized.get(key)!r}->{value!r}")
            normalized[key] = value
        normalized["slot_id"] = slot
        expected_binding = source_row["binding"]
        supplied = normalized.get("source_binding")
        if supplied is None:
            normalized["source_binding"] = copy.deepcopy(expected_binding)
            logs.append(f"{slot}:source_binding:derived_from_frozen_selection")
        elif isinstance(supplied, dict) and supplied.get("sha256") == expected_binding["sha256"]:
            normalized["source_binding"] = copy.deepcopy(expected_binding)
            if supplied != expected_binding:
                logs.append(f"{slot}:source_binding:absolute_path_normalized")
        normalized["source_status"] = source_row["source"].get("source_status")
        normalized["source_warnings"] = source_warnings(source_row["source"])
        if arm == "F" and plans_by_slot[slot].get("feedback_target"):
            if normalized.get("feedback_target") != plans_by_slot[slot]["feedback_target"]:
                logs.append(f"{slot}:feedback_target:bound_to_frozen_task_plan")
            normalized["feedback_target"] = copy.deepcopy(plans_by_slot[slot]["feedback_target"])
        elif arm == "G" and "feedback_target" in normalized:
            normalized.pop("feedback_target")
            logs.append(f"{slot}:feedback_target:removed_unauthorized_G_metadata")
        if revision_thread:
            normalized["revision_author_thread_id"] = revision_thread
            normalized["content_revision_count"] = 1
        result.append(normalized)
    return result, failures, logs


def basic_solver_shape(candidate: dict[str, Any], source_row: dict[str, Any]) -> bool:
    slot = candidate.get("slot_id")
    if slot not in SLOTS:
        return False
    kind, operation, gold_count = SLOTS[slot]
    field = "options" if kind == "multi_select" else "items"
    return (candidate.get("question_type") == kind and candidate.get("task_operation") == operation
            and isinstance(candidate.get(field), dict) and set(candidate[field]) == set(LABELS)
            and isinstance(candidate.get("gold"), list)
            and (gold_count is None or len(candidate["gold"]) == gold_count)
            and candidate.get("source_binding") == source_row["binding"])


def audit_candidate(candidate: dict[str, Any], selection_path: Path) -> dict[str, Any]:
    try:
        result = gate.audit_candidate(candidate, selection_path)
        return result
    except Exception as exc:  # malformed but parseable candidate remains visible and held
        return {"candidate_id": candidate.get("candidate_id"),
                "candidate_sha256": gate.fingerprint(candidate),
                "source_binding": candidate.get("source_binding"), "mechanical_pass": False,
                "mechanical_errors": [{"location": "candidate", "error": f"{type(exc).__name__}: {exc}"}],
                "semantically_approved": False}


def review_checks_valid(candidate: dict[str, Any], response: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    required = gate.required_checks(candidate)
    raw_checks = response.get("checks")
    checks = raw_checks if isinstance(raw_checks, dict) else {}
    errors = []
    for name in required:
        check = checks.get(name)
        if (not isinstance(check, dict) or check.get("decision") not in {"PASS", "HOLD", "FAIL"}
                or not isinstance(check.get("rationale"), str) or not check["rationale"].strip()):
            errors.append(f"required_check_incomplete:{name}")
    return checks, errors


def normalized_review(candidate: dict[str, Any], blind: dict[str, Any] | None,
                      response: dict[str, Any] | None, reviewer_thread: str | None,
                      source: dict[str, Any]) -> dict[str, Any]:
    actual_author = candidate.get("author_id")
    if response is None or blind is None:
        try:
            required = gate.required_checks(candidate)
        except Exception:
            required = {"source_support": "Check source support.", "gold_truth": "Reconstruct the answer from source."}
        return {"candidate_id": candidate["candidate_id"],
                "candidate_sha256": gate.fingerprint(candidate),
                "source_binding": candidate.get("source_binding"), "author_id": actual_author,
                "reviewer_id": reviewer_thread or "NO_VALID_REVIEW_RESPONSE",
                "decision": "HOLD", "checks": {name: {"decision": "HOLD", "rationale": "No valid independent verification response was received; no semantic admission is claimed."}
                                                     for name in required},
                "independent_gold": blind.get("independent_gold") if blind else [],
                "reconstructed_gold": [], "r03_necessity": {"decision": "HOLD", "rationale": "No valid independent R03 review was received.", "turn_refs": []},
                "issues": ["independent_review_missing_or_invalid"], "source_read_complete": False,
                "solver_results_read": False, "gold_blind_initial_stage": bool(blind),
                "arm_and_feedback_hidden": True, "ai_family_review": True,
                "effective_decision_origin": "engineering_fail_closed"}
    checks, check_errors = review_checks_valid(candidate, response)
    independent = blind.get("independent_gold", [])
    reconstructed = response.get("reconstructed_gold")
    if not isinstance(reconstructed, list) or any(x not in LABELS for x in reconstructed):
        reconstructed = []
        check_errors.append("reconstructed_gold_invalid")
    if candidate["question_type"] == "multi_select":
        reconstructed = sorted(reconstructed)
        expected_count = SLOTS[candidate["slot_id"]][2]
    else:
        expected_count = 6
    if (len(reconstructed) != expected_count
            or len(set(reconstructed)) != len(reconstructed)):
        check_errors.append("reconstructed_gold_cardinality_or_uniqueness_invalid")
    turns = {t.get("turn_ref") for t in source.get("turns", []) if isinstance(t, dict)}
    r03 = response.get("r03_necessity")
    if (not isinstance(r03, dict) or r03.get("decision") not in {"PASS", "HOLD", "FAIL"}
            or not isinstance(r03.get("rationale"), str) or not r03["rationale"].strip()
            or not isinstance(r03.get("turn_refs"), list) or not r03["turn_refs"]
            or any(ref not in turns or not str(ref).startswith("R03:") for ref in r03["turn_refs"])):
        check_errors.append("r03_necessity_incomplete")
        r03 = {"decision": "HOLD", "rationale": "Reviewer did not provide a valid R03 necessity review.", "turn_refs": []}
    reviewer_decision = response.get("decision")
    if reviewer_decision not in {"PASS", "HOLD", "FAIL"}:
        reviewer_decision = "HOLD"
        check_errors.append("review_decision_invalid")
    if reviewer_decision == "PASS" and any(
            not isinstance(checks.get(name), dict) or checks[name].get("decision") != "PASS"
            for name in gate.required_checks(candidate)):
        check_errors.append("pass_decision_contains_nonpass_required_check")
    if response.get("source_read_complete") is not True:
        check_errors.append("source_read_attestation_missing")
    if response.get("solver_results_read") is not False:
        check_errors.append("solver_results_isolation_attestation_invalid")
    if reviewer_thread is not None and reviewer_thread == actual_author:
        check_errors.append("reviewer_thread_matches_author_thread")
    if not blind.get("complete_source_read") or blind.get("solver_results_read") is not False:
        check_errors.append("blind_stage_attestation_invalid")
    author_gold = candidate.get("gold")
    if candidate["question_type"] == "multi_select":
        author_gold_compare = sorted(author_gold) if isinstance(author_gold, list) else author_gold
        independent_compare = sorted(independent)
        reconstructed_compare = sorted(reconstructed)
    else:
        author_gold_compare = author_gold
        independent_compare = independent
        reconstructed_compare = reconstructed
    gold_conflict = (independent_compare != author_gold_compare or reconstructed_compare != author_gold_compare)
    if reconstructed_compare != author_gold_compare:
        check_errors.append("verifier_reconstructed_gold_differs_from_authored_key")
    discrepancy_resolution = response.get("gold_discrepancy_resolution")
    discrepancy_resolution_valid = (
        isinstance(discrepancy_resolution, str)
        and len(discrepancy_resolution.strip()) >= 24
        and any(isinstance(ref, str) and ref in discrepancy_resolution for ref in turns)
    )
    if gold_conflict and not discrepancy_resolution_valid:
        check_errors.append("gold_discrepancy_resolution_missing_or_not_source_cited")
    reviewer_id = reviewer_thread or "NO_VALID_REVIEW_RESPONSE"
    record = {"candidate_id": candidate["candidate_id"],
              "candidate_sha256": gate.fingerprint(candidate),
              "source_binding": candidate.get("source_binding"), "author_id": actual_author,
              "reviewer_id": reviewer_id, "decision": reviewer_decision, "checks": checks,
              "independent_gold": independent, "reconstructed_gold": reconstructed,
              "gold_discrepancy": {"present": gold_conflict,
                  "author_gold": author_gold,
                  "blind_independent_gold": independent,
                  "verifier_reconstructed_gold": reconstructed},
              "gold_discrepancy_resolution_valid": discrepancy_resolution_valid,
              "r03_necessity": r03,
              "issues": check_errors + (response.get("issues", []) if isinstance(response.get("issues"), list) else []),
              "gold_discrepancy_resolution": discrepancy_resolution,
              "source_read_complete": response.get("source_read_complete") is True,
              "solver_results_read": response.get("solver_results_read"),
              "gold_blind_initial_stage": True, "arm_and_feedback_hidden": True,
              "ai_family_review": True,
              "review_id": "QREV-" + digest([candidate["candidate_id"], reviewer_id])[:10],
              "effective_decision_origin": "independent_ai_reviewer"}
    if check_errors and record["decision"] == "PASS":
        record["decision"] = "HOLD"
        record["semantic_reviewer_decision"] = reviewer_decision
        record["effective_decision_origin"] = "engineering_hold_incomplete_review_record"
    if gold_conflict and not discrepancy_resolution_valid and record["decision"] == "PASS":
        record["decision"] = "HOLD"
        record["semantic_reviewer_decision"] = reviewer_decision
        record["effective_decision_origin"] = "engineering_hold_gold_discrepancy"
    if r03.get("decision") != "PASS" and record["decision"] == "PASS":
        record["decision"] = "HOLD"
        record["semantic_reviewer_decision"] = reviewer_decision
        record["effective_decision_origin"] = "engineering_hold_r03_necessity"
    return record


def check_for_candidate(response: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    public, _, _ = anonymize_candidate(candidate)
    row = next((r for r in extract_list(response, "reviews", "answers")
                if r.get("question_key") == public["question_key"]), None)
    return row


def review_batch(root: str, candidates: list[dict[str, Any]], source_row: dict[str, Any],
                 selection_path: Path, work: Path, suffix: str = "") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run blind reconstruction, then a separate candidate verifier."""
    blind_map: dict[str, Any] = {}
    audit_map = {c["candidate_id"]: audit_candidate(c, selection_path) for c in candidates}
    blind_receipt = verify_receipt = None
    blind_response = verify_response = None
    blind_errors: list[str] = []
    if candidates:
        blind_cases = [anonymize_candidate(c) for c in candidates]
        blind_key = call_id("gold", candidates[0]["arm"], root, suffix or "initial")
        try:
            prompt = blind_prompt(root, source_view(source_row), blind_cases)
            if candidates[0]["arm"] == "G":
                validate_g_prompt(prompt, "blind_gold_reviewer", root)
            blind_response, blind_receipt = cc.call(blind_key, prompt)
            publish(work / f"BLIND_GOLD{('_' + suffix) if suffix else ''}.json", blind_response)
            publish(work / f"BLIND_GOLD_CALL{('_' + suffix) if suffix else ''}.json", blind_receipt)
            blind_map, blind_errors = normalize_blind(blind_response, candidates, source_row["source"])
            publish(work / f"BLIND_GOLD_VALIDATION{('_' + suffix) if suffix else ''}.json",
                    {"errors": blind_errors, "question_count": len(blind_map), "blind_before_author_key": True})
        except Exception as exc:
            blind_errors.append(f"blind_call_failed:{type(exc).__name__}:{exc}")
            publish(work / f"BLIND_GOLD_FAILURE{('_' + suffix) if suffix else ''}.json", {"error": blind_errors[-1]})
    verify_candidates = [c for c in candidates if c["candidate_id"] in blind_map
                         and not any(c["candidate_id"] in e for e in blind_errors)]
    verify_rows = []
    for candidate in verify_candidates:
        packet = gate.build_review_packet(candidate, selection_path)
        verify_rows.append({"question_key": blind_map[candidate["candidate_id"]]["question_key"],
                            "anonymous_candidate": {"question": candidate.get("question"),
                                "question_type": candidate.get("question_type"), "task_operation": candidate.get("task_operation"),
                                "options": candidate.get("options"), "items": candidate.get("items"),
                                "gold": candidate.get("gold"), "item_analysis": candidate.get("item_analysis"),
                                "ordering_analysis": candidate.get("ordering_analysis"),
                                "r03_necessity": candidate.get("r03_necessity")},
                            "saved_independent_gold": blind_map[candidate["candidate_id"]],
                            "mechanical_review_packet": anonymize_mechanical_packet(packet)})
    if verify_rows:
        verify_key = call_id("verify", candidates[0]["arm"], root, suffix or "initial")
        try:
            prompt = verify_prompt(root, source_view(source_row), verify_rows)
            if candidates[0]["arm"] == "G":
                validate_g_prompt(prompt, "content_verifier", root)
            verify_response, verify_receipt = cc.call(verify_key, prompt)
            publish(work / f"CONTENT_VERIFIER{('_' + suffix) if suffix else ''}.json", verify_response)
            publish(work / f"CONTENT_VERIFIER_CALL{('_' + suffix) if suffix else ''}.json", verify_receipt)
        except Exception as exc:
            publish(work / f"CONTENT_VERIFIER_FAILURE{('_' + suffix) if suffix else ''}.json",
                    {"error": f"{type(exc).__name__}: {exc}"})
    review_map = {}
    for candidate in candidates:
        blind = blind_map.get(candidate["candidate_id"])
        row = check_for_candidate(verify_response, candidate)
        reviewer_thread = (verify_receipt or {}).get("backend", {}).get("thread_id") if verify_receipt else None
        review_map[candidate["candidate_id"]] = normalized_review(candidate, blind, row, reviewer_thread, source_row["source"])
        review_map[candidate["candidate_id"]]["blind_stage_errors"] = [e for e in blind_errors if candidate["candidate_id"] in e]
    publish(work / f"CONTENT_REVIEW_NORMALIZED{('_' + suffix) if suffix else ''}.json",
            {"reviews": list(review_map.values()), "mechanical_audits": list(audit_map.values()),
             "blind_call": blind_receipt, "verification_call": verify_receipt})
    return blind_map, review_map, {"blind": blind_receipt, "verification": verify_receipt}, audit_map


def slot_failure(slot: str, reason: str, root: str, arm: str, stage: str) -> dict[str, Any]:
    return {"root_id": root, "arm": arm, "slot_id": slot, "stage": stage, "reason": reason}


def author_call_input_record(call_path: Path, root: str, arm: str,
                             source_row: dict[str, Any], contract_binding: dict[str, Any],
                             feedback_binding: dict[str, Any] | None,
                             adapter_binding: dict[str, Any] | None) -> dict[str, Any]:
    """Describe only the material present in this call's immutable prompt."""
    receipt = read(call_path)
    request = receipt.get("request")
    if not isinstance(request, dict) or not request.get("path"):
        raise RuntimeError(f"Call receipt has no saved prompt request: {call_path}")
    verify_binding(request)
    prompt = Path(request["path"]).read_text(encoding="utf-8")
    marker = next((value for value in ("FROZEN_INPUTS:\n", "AUTHORIZED_INPUTS:\n", "REVIEW_INPUT:\n")
                   if value in prompt), None)
    if marker is None:
        raise RuntimeError(f"No recognized embedded-input marker in {request['path']}")
    payload_start = prompt.index(marker) + len(marker)
    payload, payload_end = json.JSONDecoder().raw_decode(prompt[payload_start:])
    if prompt[payload_start + payload_end:].strip():
        raise RuntimeError(f"Unexpected text after embedded prompt JSON: {request['path']}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Prompt payload must be an object: {request['path']}")

    assignment = payload.get("assignment")
    embedded_root = assignment.get("root_id") if isinstance(assignment, dict) else payload.get("root_id")
    embedded_arm = assignment.get("arm") if isinstance(assignment, dict) else arm
    if embedded_root != root or embedded_arm != arm:
        raise RuntimeError(f"Prompt assignment does not match call path: {call_path}")
    embedded_source = payload.get("source_and_warnings")
    if embedded_source is None:
        embedded_source = payload.get("complete_source_and_warnings")
    embedded_binding = (embedded_source.get("source_binding")
                        if isinstance(embedded_source, dict) else None)
    if (not isinstance(embedded_source, dict) or embedded_binding != source_row["binding"]
            or embedded_source.get("source") != source_row["source"]):
        raise RuntimeError(f"Prompt does not embed the assigned complete source: {call_path}")

    embedded_fields = sorted(payload)
    materials: dict[str, Any] = {
        "complete_assigned_source": embedded_binding,
        "source_status_and_warnings": {
            "source_status": embedded_source.get("source_status"),
            "warnings_embedded": isinstance(embedded_source.get("warnings"), list),
        },
    }
    if "common_contract" in payload:
        expected_contract = Path(contract_binding["path"]).read_text(encoding="utf-8")
        if payload["common_contract"] != expected_contract:
            raise RuntimeError(f"Embedded common contract differs from its binding: {call_path}")
        materials["common_contract"] = contract_binding
    if "public_feedback_input_19_cards" in payload:
        cards = payload["public_feedback_input_19_cards"].get("cards")
        if arm != "F" or not isinstance(cards, list) or len(cards) != 19 or feedback_binding is None:
            raise RuntimeError(f"Unexpected or incomplete public feedback in prompt: {call_path}")
        if payload["public_feedback_input_19_cards"] != read(feedback_binding["path"]):
            raise RuntimeError(f"Embedded public feedback differs from its binding: {call_path}")
        materials["public_feedback_input"] = {**feedback_binding, "embedded_card_count": len(cards)}
    elif arm == "G":
        leaked_paths = _nested_material_keys(payload)
        if leaked_paths:
            raise RuntimeError(f"G prompt contains unauthorized feedback material at: {call_path}: {leaked_paths}")
    if "feedback_transfer_adapter" in payload:
        if arm != "F" or adapter_binding is None:
            raise RuntimeError(f"Unexpected feedback adapter in prompt: {call_path}")
        if payload["feedback_transfer_adapter"] != Path(adapter_binding["path"]).read_text(encoding="utf-8"):
            raise RuntimeError(f"Embedded feedback adapter differs from its binding: {call_path}")
        materials["feedback_transfer_adapter"] = adapter_binding
    if arm == "G" and "feedback_transfer_adapter" in payload:
        raise RuntimeError(f"G prompt contains an unauthorized adapter field: {call_path}")
    if "frozen_task_plans" in payload:
        plan_path = BASE / "question_authoring" / arm / "roots" / root_number(root) / "TASK_PLANS.json"
        supplied_plan = payload["frozen_task_plans"]
        frozen_plan = read(plan_path) if plan_path.exists() else None
        if (not isinstance(supplied_plan, dict) or not isinstance(frozen_plan, dict)
                or {key: value for key, value in supplied_plan.items() if key != "plans"}
                != {key: value for key, value in frozen_plan.items() if key != "plans"}
                or not isinstance(supplied_plan.get("plans"), list)
                or not isinstance(frozen_plan.get("plans"), list)):
            raise RuntimeError(f"Embedded task plan differs from the frozen root plan: {call_path}")
        frozen_slots = {row.get("slot_id"): row for row in frozen_plan["plans"] if isinstance(row, dict)}
        supplied_slots = [row.get("slot_id") for row in supplied_plan["plans"] if isinstance(row, dict)]
        if (len(supplied_slots) != len(supplied_plan["plans"])
                or len(set(supplied_slots)) != len(supplied_slots)
                or any(row != frozen_slots.get(row.get("slot_id")) for row in supplied_plan["plans"])
                or not isinstance(assignment, dict) or supplied_slots != assignment.get("slots")):
            raise RuntimeError(f"Embedded task plan is not an exact frozen slot subset: {call_path}")
        materials["frozen_task_plans"] = {**binding(plan_path), "embedded_slot_ids": supplied_slots,
                                             "exact_frozen_slot_subset": True}
    if "revision_cases" in payload:
        materials["revision_cases"] = {"count": len(payload["revision_cases"])
                                       if isinstance(payload["revision_cases"], list) else None,
                                       "included_in_saved_prompt": True}
    if "anonymous_questions" in payload:
        materials["anonymous_questions"] = {"count": len(payload["anonymous_questions"])
                                             if isinstance(payload["anonymous_questions"], list) else None,
                                             "author_gold_and_arm_hidden": True}
    if "review_cases" in payload:
        materials["review_cases"] = {"count": len(payload["review_cases"])
                                      if isinstance(payload["review_cases"], list) else None,
                                      "included_in_saved_prompt": True}

    call_name = call_path.name.removesuffix(".json")
    role = ("task_planner" if call_name.startswith("TASK_PLAN_CALL") else
            "candidate_author" if call_name.startswith("CANDIDATES_DRAFT_CALL") else
            "candidate_reviser" if call_name.startswith("CANDIDATES_REVISION_01_CALL") else
            "blind_gold_reviewer" if call_name.startswith("BLIND_GOLD_CALL") else
            "content_verifier" if call_name.startswith("CONTENT_VERIFIER_CALL") else "unclassified")
    if role == "unclassified":
        raise RuntimeError(f"Unclassified semantic call receipt: {call_path}")
    if arm == "G":
        try:
            validate_g_payload_fields(payload, role, root)
        except ValueError as exc:
            raise RuntimeError(f"{exc}: {call_path}") from exc
    thread_id = receipt.get("backend", {}).get("thread_id")
    if not thread_id:
        raise RuntimeError(f"Call receipt has no fresh thread id: {call_path}")
    parsed = {}
    if receipt.get("parsed_response"):
        verify_binding(receipt["parsed_response"])
        parsed = read(receipt["parsed_response"]["path"])
    attestations = []
    for row in extract_list(parsed, "candidates", "reviews", "answers"):
        fields = {key: row[key] for key in ("complete_source_read", "source_read_complete", "solver_results_read")
                  if key in row}
        if fields:
            attestations.append(fields)
    return {
        "call_id": call_name.removesuffix("_CALL"),
        "role": role,
        "root_id": root,
        "arm": arm,
        "thread_id": thread_id,
        "fresh_thread": True,
        "saved_prompt_request_binding": request,
        "embedded_payload_keys": embedded_fields,
        "embedded_materials": materials,
        "response_read_attestations": attestations,
        "call_receipt_binding": binding(call_path),
    }


def validate_g_prompt(prompt: str, role: str, root: str) -> None:
    """Check the actual serialized G inputs before any semantic call starts."""
    marker = next((value for value in ("FROZEN_INPUTS:\n", "AUTHORIZED_INPUTS:\n", "REVIEW_INPUT:\n")
                   if value in prompt), None)
    if marker is None:
        raise ValueError("G prompt has no recognized embedded-input marker")
    payload_start = prompt.index(marker) + len(marker)
    payload, payload_end = json.JSONDecoder().raw_decode(prompt[payload_start:])
    if not isinstance(payload, dict) or prompt[payload_start + payload_end:].strip():
        raise ValueError("G prompt does not contain one complete authorized JSON payload")
    validate_g_payload_fields(payload, role, root)
    forbidden = _nested_material_keys(payload)
    if forbidden:
        raise ValueError(f"G {role} input contains unauthorized nested material: {forbidden}")


def validate_g_payload_fields(payload: dict[str, Any], role: str, root: str) -> None:
    """Fail closed on G top-level inputs outside each role's explicit allowlist."""
    role_payload_fields = {
        "task_planner": {"assignment", "common_contract", "source_and_warnings"},
        "candidate_author": {"assignment", "common_contract", "source_and_warnings", "frozen_task_plans"},
        "candidate_reviser": {"assignment", "common_contract", "source_and_warnings",
                              "frozen_task_plans", "revision_cases"},
        "blind_gold_reviewer": {"root_id", "complete_source_and_warnings", "anonymous_questions"},
        "content_verifier": {"root_id", "complete_source_and_warnings", "review_cases"},
    }
    allowed = role_payload_fields[role]
    if set(payload) != allowed:
        extra = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        raise ValueError(f"G {role} payload fields violate allowlist; extra={extra}, missing={missing}")
    if role in {"task_planner", "candidate_author", "candidate_reviser"}:
        assignment = payload.get("assignment")
        if (not isinstance(assignment, dict)
                or set(assignment) != {"root_id", "arm", "slots"}
                or assignment.get("arm") != "G"
                or assignment.get("root_id") != root
                or not isinstance(assignment.get("slots"), list)
                or any(slot not in SLOTS for slot in assignment["slots"])):
            raise ValueError("G assignment is outside the frozen root/slot allowlist")


def _arm_call_artifacts(arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind all question-call files, including failed/interrupted calls without RESULT."""
    call_records = []
    file_bindings = []
    arm_token = f"_{arm.lower()}_"
    for call_dir in sorted((BASE / "cli_calls").glob("q20_*")):
        if not call_dir.is_dir() or arm_token not in call_dir.name:
            continue
        files = [binding(path) for path in sorted(call_dir.rglob("*")) if path.is_file()]
        file_bindings.extend(files)
        result_path = call_dir / "RESULT.json"
        received = False
        if result_path.exists():
            result = read(result_path)
            received = result.get("response_received") is True
        if not received:
            call_records.append({"call_id": call_dir.name,
                                 "status": "NO_ACCEPTED_RESPONSE_OR_INTERRUPTED",
                                 "artifacts": files})
    registry_files = sorted((BASE / "process_registry").glob("q20_*.json"))
    registry_bindings = [binding(path) for path in registry_files if arm_token in path.name]
    file_bindings.extend(registry_bindings)
    return call_records, file_bindings


def process_root_arm(root: str, arm: str, source_row: dict[str, Any], selection_path: Path,
                     contract: str, contract_binding: dict[str, Any],
                     feedback: dict[str, Any] | None, adapter: str | None,
                     feedback_binding: dict[str, Any] | None, adapter_binding: dict[str, Any] | None) -> dict[str, Any]:
    work = BASE / "question_authoring" / arm / "roots" / root_number(root)
    work.mkdir(parents=True, exist_ok=True)
    lock_path = work / ".driver.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = work / "OUTCOME.json"
        if existing.exists():
            outcome = read(existing)
            if outcome.get("source_binding") != source_row["binding"] or outcome.get("arm") != arm:
                raise RuntimeError(f"Root/arm cached outcome input drift: {root}/{arm}")
            return outcome
        outcome: dict[str, Any] = {"schema_version": "fgr20_question_root_arm_v1", "root_id": root,
            "arm": arm, "source_binding": source_row["binding"], "status": "PARTIAL",
            "candidate_ids": [], "slot_failures": [], "calls": [], "max_content_revisions": 1}
        try:
            if cc.stop_generation():
                raise RuntimeError("Generation cutoff/request reached before this root/arm")
            plan_key = call_id("plan", arm, root)
            plan_path = work / "TASK_PLANS.json"
            plan_freeze = work / "TASK_PLANS_FREEZE.json"
            if plan_path.exists() and plan_freeze.exists():
                frozen = read(plan_freeze)
                verify_binding(frozen["plans_binding"])
                task_plans = read(plan_path)
                if task_plans.get("root_id") != root or task_plans.get("arm") != arm:
                    raise RuntimeError("Frozen task plan identity drift")
                plan_receipt = read(work / "TASK_PLAN_CALL.json")
            else:
                prompt = task_plan_prompt(arm, root, source_view(source_row), contract, feedback, adapter)
                raw, plan_receipt = cc.call(plan_key, prompt)
                publish(work / "TASK_PLAN_RESPONSE.json", raw)
                publish(work / "TASK_PLAN_CALL.json", plan_receipt)
                feedback_ids = {str(card.get("feedback_id")) for card in (feedback or {}).get("cards", []) if isinstance(card, dict)}
                task_plans, plan_errors = validate_plans(raw, arm, root, source_row["source"], feedback_ids)
                publish(plan_path, task_plans)
                frozen_value = {"schema_version": "fgr20_task_plan_freeze_v1", "root_id": root, "arm": arm,
                                "plans_binding": binding(plan_path), "call_receipt": binding(work / "TASK_PLAN_CALL.json"),
                                "errors": plan_errors, "frozen_before_candidate_draft": True}
                publish(plan_freeze, frozen_value)
            outcome["calls"].append(plan_receipt)
            frozen_plans = task_plans
            plan_errors = frozen_plans.get("errors", [])
            invalid_plan_slots = {error.rsplit(":", 1)[-1] for error in plan_errors
                                  if error.rsplit(":", 1)[-1] in SLOTS}
            usable_plans = [p for p in frozen_plans.get("plans", []) if p.get("slot_id") not in invalid_plan_slots]
            for bad_slot in invalid_plan_slots:
                outcome["slot_failures"].append(slot_failure(bad_slot,
                    ";".join(error for error in plan_errors if error.endswith(":" + bad_slot)), root, arm, "plan"))
            if not usable_plans:
                for slot in SLOTS:
                    if any(item["slot_id"] == slot for item in outcome["slot_failures"]):
                        continue
                    outcome["slot_failures"].append(slot_failure(slot, "no valid frozen task plan", root, arm, "plan"))
                publish(work / "SLOT_FAILURES.json", outcome["slot_failures"])
                outcome["status"] = "FAILED_NO_PLANS"
                publish(existing, outcome)
                return outcome

            author_key = call_id("author", arm, root)
            draft_plans = {**frozen_plans, "plans": usable_plans}
            prompt = author_prompt(arm, root, source_view(source_row), contract, draft_plans, feedback, adapter)
            raw_candidates, author_receipt = cc.call(author_key, prompt)
            publish(work / "CANDIDATES_DRAFT_RESPONSE.json", raw_candidates)
            publish(work / "CANDIDATES_DRAFT_CALL.json", author_receipt)
            candidates, failures, normalization = normalize_candidates(raw_candidates, arm, root,
                author_receipt["backend"]["thread_id"], source_row, draft_plans, binding(selection_path))
            publish(work / "CANDIDATES_DRAFT.json", candidates)
            publish(work / "DRAFT_NORMALIZATION.json", {"changes": normalization, "input_binding": binding(work / "CANDIDATES_DRAFT_RESPONSE.json")})
            outcome["calls"].append(author_receipt)
            outcome["slot_failures"].extend(slot_failure(f["slot_id"] or "UNASSIGNED", f["reason"], root, arm, "draft") for f in failures)
            usable = [c for c in candidates if basic_solver_shape(c, source_row)]
            for c in candidates:
                if c not in usable:
                    outcome["slot_failures"].append(slot_failure(c.get("slot_id", "UNKNOWN"),
                        "parseable candidate does not satisfy fixed slot/source runtime shape; retained in candidate archive", root, arm, "draft"))
            blind0, reviews0, review_calls0, audits0 = review_batch(root, usable, source_row, selection_path, work)
            outcome["calls"].extend(x for x in review_calls0.values() if x)
            current = {c["candidate_id"]: c for c in usable}
            final_reviews = dict(reviews0)
            final_audits = dict(audits0)
            needs_revision = [c for c in usable if reviews0.get(c["candidate_id"], {}).get("decision") != "PASS"
                              or not audits0.get(c["candidate_id"], {}).get("mechanical_pass")]
            if needs_revision and not cc.stop_generation():
                revision_cases = []
                for candidate in needs_revision:
                    cid = candidate["candidate_id"]
                    review = reviews0.get(cid, {})
                    revision_cases.append({"candidate": candidate,
                        "frozen_slot_plan": next((p for p in frozen_plans["plans"] if p["slot_id"] == candidate["slot_id"]), None),
                        "review_defects": {"decision": review.get("decision"), "checks": review.get("checks"),
                                           "issues": review.get("issues"), "mechanical_audit": audits0.get(cid)}})
                revision_key = call_id("revision", arm, root, "01")
                revision_prompt = author_prompt(arm, root, source_view(source_row), contract, frozen_plans,
                                                feedback, adapter, revision_cases)
                try:
                    revised_raw, revision_receipt = cc.call(revision_key, revision_prompt)
                    publish(work / "CANDIDATES_REVISION_01_RESPONSE.json", revised_raw)
                    publish(work / "CANDIDATES_REVISION_01_CALL.json", revision_receipt)
                    revisions, revision_failures, revision_normalization = normalize_candidates(
                        revised_raw, arm, root, author_receipt["backend"]["thread_id"], source_row,
                        frozen_plans, binding(selection_path), revision_receipt["backend"]["thread_id"])
                    publish(work / "CANDIDATES_REVISION_01.json", revisions)
                    publish(work / "REVISION_NORMALIZATION.json", {"changes": revision_normalization,
                        "revision_author_thread_id": revision_receipt["backend"]["thread_id"],
                        "candidate_lineage_author_thread_id": author_receipt["backend"]["thread_id"]})
                    outcome["calls"].append(revision_receipt)
                    revised_by_slot = {c["slot_id"]: c for c in revisions}
                    authorized_slots = {c["slot_id"] for c in needs_revision}
                    changed = []
                    for revised in revisions:
                        if revised["slot_id"] not in authorized_slots:
                            outcome["slot_failures"].append(slot_failure(revised["slot_id"],
                                "revision returned an unrequested slot; replacement retained only in revision archive", root, arm, "revision"))
                    for old in needs_revision:
                        replacement = revised_by_slot.get(old["slot_id"])
                        if replacement and basic_solver_shape(replacement, source_row):
                            current[old["candidate_id"]] = replacement
                            changed.append(replacement)
                        else:
                            outcome["slot_failures"].append(slot_failure(old["slot_id"],
                                "single revision missing or invalid; original parseable candidate retained", root, arm, "revision"))
                    if changed:
                        blind1, reviews1, review_calls1, audits1 = review_batch(root, changed, source_row,
                            selection_path, work, "revision_01")
                        outcome["calls"].extend(x for x in review_calls1.values() if x)
                        final_reviews.update(reviews1)
                        final_audits.update(audits1)
                except Exception as exc:
                    publish(work / "REVISION_FAILURE.json", {"error": f"{type(exc).__name__}: {exc}",
                        "revision_semantic_attempt_consumed_or_unavailable": True})
                    for old in needs_revision:
                        final_reviews[old["candidate_id"]]["decision"] = "HOLD"
                        final_reviews[old["candidate_id"]]["effective_decision_origin"] = "revision_not_completed"
            elif needs_revision:
                for old in needs_revision:
                    final_reviews[old["candidate_id"]]["decision"] = "HOLD"
                    final_reviews[old["candidate_id"]]["effective_decision_origin"] = "generation_cutoff_before_revision"

            # Keep one final candidate per slot; drafts and all revision responses
            # remain archived separately, including malformed or unrequested rows.
            all_by_slot = {c["slot_id"]: c for c in candidates}
            all_by_slot.update({current[cid]["slot_id"]: current[cid] for cid in current})
            final_candidates = [all_by_slot[s] for s in SLOTS if s in all_by_slot]
            for c in final_candidates:
                cid = c["candidate_id"]
                if cid not in final_reviews:
                    final_reviews[cid] = normalized_review(c, None, None, None, source_row["source"])
                    final_audits[cid] = audit_candidate(c, selection_path)
                if not basic_solver_shape(c, source_row):
                    final_reviews[cid]["decision"] = "HOLD"
                    final_reviews[cid]["effective_decision_origin"] = "runtime_shape_failure"
                review = final_reviews[cid]
                audit = final_audits.get(cid, {})
                if review.get("decision") != "PASS" or not audit.get("mechanical_pass"):
                    details = []
                    for name, check in review.get("checks", {}).items():
                        if isinstance(check, dict) and check.get("decision") != "PASS":
                            details.append(f"{name}:{check.get('rationale', '')}")
                    details.extend(str(issue) for issue in review.get("issues", []))
                    if review.get("decision") != "PASS" and not details:
                        details.append(f"review_decision:{review.get('decision')}")
                    if not audit.get("mechanical_pass"):
                        details.extend(str(error) for error in audit.get("mechanical_errors", []))
                    outcome["slot_failures"].append(slot_failure(c["slot_id"],
                        "content not admitted: " + "; ".join(details), root, arm, "content_review"))
            publish(work / "CANDIDATES_FINAL.json", final_candidates)
            publish(work / "CONTENT_REVIEW_FINAL.json", {"reviews": [final_reviews[c["candidate_id"]] for c in final_candidates],
                "mechanical_audits": [final_audits.get(c["candidate_id"], audit_candidate(c, selection_path)) for c in final_candidates],
                "all_reviews_ai_family_not_human": True})
            publish(work / "SLOT_FAILURES.json", outcome["slot_failures"])
            publish(work / "TASK_PLAN_BINDING.json", {"plans": binding(work / "TASK_PLANS.json"),
                "freeze": binding(work / "TASK_PLANS_FREEZE.json"), "candidate_draft": binding(work / "CANDIDATES_DRAFT.json")})
            outcome.update(status="COMPLETED_WITH_HOLES" if outcome["slot_failures"] or any(r["decision"] != "PASS" for r in final_reviews.values()) else "COMPLETED",
                           candidate_ids=[c["candidate_id"] for c in final_candidates],
                           candidate_bindings=[{"candidate_id": c["candidate_id"], "candidate_sha256": gate.fingerprint(c)} for c in final_candidates],
                           final_candidates_binding=binding(work / "CANDIDATES_FINAL.json"),
                           review_binding=binding(work / "CONTENT_REVIEW_FINAL.json"),
                           slot_failures_binding=binding(work / "SLOT_FAILURES.json"),
                           plan_freeze_binding=binding(work / "TASK_PLANS_FREEZE.json"))
        except Exception as exc:
            outcome["status"] = "FAILED_STAGE"
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            for slot in SLOTS:
                if not any(r["slot_id"] == slot for r in outcome["slot_failures"]):
                    outcome["slot_failures"].append(slot_failure(slot, outcome["error"], root, arm, "stage"))
            publish(work / "SLOT_FAILURES.json", outcome["slot_failures"])
        publish(existing, outcome)
        cc.progress("QUESTION_ROOT_ARM_COMPLETE", root_id=root, arm=arm, status=outcome["status"],
                    candidate_count=len(outcome.get("candidate_ids", [])))
        return outcome


def normalized_question_fingerprint(candidate: dict[str, Any]) -> str:
    key = "items" if candidate.get("question_type") == "ordering" else "options"
    texts = candidate.get(key, {})
    normalized = sorted(re.sub(r"\s+", " ", value).strip() for value in texts.values() if isinstance(value, str))
    return digest({"question_type": candidate.get("question_type"),
                   "question": re.sub(r"\s+", " ", str(candidate.get("question", ""))).strip(),
                   "items": normalized})


def finalize() -> dict[str, Any]:
    config = read(BASE / "CONFIG.json")
    selection_path = SOURCE_SELECTION_PATH
    selection, source_rows = load_selection(selection_path)
    contract_path = BASE / "question_authoring" / "COMMON_AUTHOR_CONTRACT_ZH.md"
    contract = contract_path.read_text(encoding="utf-8")
    contract_binding = binding(contract_path)
    feedback, adapter, feedback_binding, adapter_binding = feedback_material()
    root_order = [root_id(x) for x in config["roots"]]
    slot_order = list(SLOTS)
    stage_roots = BASE / "question_authoring"
    outcomes = {}
    candidates = []
    review_by_id = {}
    candidate_stage_bindings = []
    failures = []
    for root in root_order:
        for arm in ARMS:
            work = stage_roots / arm / "roots" / root_number(root)
            outcome_path = work / "OUTCOME.json"
            if root not in source_rows:
                for slot in SLOTS:
                    failures.append(slot_failure(slot, "root has no passing final source selection", root, arm, "source"))
                continue
            if not outcome_path.exists():
                for slot in SLOTS:
                    failures.append(slot_failure(slot, "question authoring stage not run", root, arm, "not_run"))
                continue
            outcome = read(outcome_path)
            outcomes[(root, arm)] = outcome
            verify_binding(outcome["review_binding"]) if outcome.get("review_binding") else None
            if outcome.get("slot_failures_binding"):
                verify_binding(outcome["slot_failures_binding"])
                failures.extend(read(outcome["slot_failures_binding"]["path"]))
            for artifact in ("CANDIDATES_DRAFT_RESPONSE.json", "CANDIDATES_DRAFT.json", "CONTENT_REVIEW_FINAL.json",
                             "TASK_PLANS.json", "TASK_PLANS_FREEZE.json", "TASK_PLAN_CALL.json",
                             "CANDIDATES_DRAFT_CALL.json", "DRAFT_NORMALIZATION.json"):
                path = work / artifact
                if path.exists():
                    candidate_stage_bindings.append(binding(path))
            rows = read(work / "CANDIDATES_FINAL.json") if (work / "CANDIDATES_FINAL.json").exists() else []
            final_reviews = read(work / "CONTENT_REVIEW_FINAL.json")["reviews"] if (work / "CONTENT_REVIEW_FINAL.json").exists() else []
            review_by_id.update({r["candidate_id"]: r for r in final_reviews})
            final_by_slot = {c.get("slot_id"): c for c in rows}
            for slot in slot_order:
                candidate = final_by_slot.get(slot)
                if candidate:
                    candidates.append(candidate)
                elif not any(f.get("root_id") == root and f.get("arm") == arm and f.get("slot_id") == slot for f in failures):
                    failures.append(slot_failure(slot, "candidate missing", root, arm, "finalize"))
    # Fixed root / slot / F-before-G order controls exact duplicate disposition.
    candidates.sort(key=lambda c: (root_order.index(c["root_id"]), slot_order.index(c["slot_id"]), ARMS.index(c["arm"])))
    duplicate_owner: dict[str, str] = {}
    duplicate_rows = []
    gate_rows = []
    for c in candidates:
        cid = c["candidate_id"]
        review = review_by_id.setdefault(cid, normalized_review(c, None, None, None, source_rows[c["root_id"]]["source"]))
        audit = audit_candidate(c, selection_path)
        fingerprint = normalized_question_fingerprint(c)
        prior = duplicate_owner.get(fingerprint)
        duplicate = prior is not None
        if not duplicate:
            duplicate_owner[fingerprint] = cid
        else:
            review["semantic_reviewer_decision"] = review.get("semantic_reviewer_decision", review.get("decision"))
            review["decision"] = "HOLD"
            review["duplicate_disposition"] = {"status": "HOLD_EXACT_DUPLICATE", "kept_candidate_id": prior,
                                                "fingerprint": fingerprint}
            review["effective_decision_origin"] = "engineering_exact_duplicate_exclusion"
        review["candidate_sha256"] = gate.fingerprint(c)
        review["source_binding"] = c.get("source_binding")
        review["author_id"] = c.get("author_id")
        review["duplicate_fingerprint"] = fingerprint
        try:
            validation = gate.validate_review(c, review, selection_path)
        except Exception as exc:
            validation = {"candidate_id": cid, "admitted": False, "mechanical_pass": False,
                          "review_errors": [f"gate_validation_error:{type(exc).__name__}:{exc}"]}
        if not audit.get("mechanical_pass") and review.get("decision") == "PASS":
            review["semantic_reviewer_decision"] = review.get("decision")
            review["decision"] = "HOLD"
            review["effective_decision_origin"] = "engineering_mechanical_gate_hold"
            validation["admitted"] = False
        if duplicate:
            validation["admitted"] = False
        gate_rows.append({"candidate_id": cid, "candidate_sha256": gate.fingerprint(c),
            "mechanical_audit": audit, "review_validation": validation,
            "eligible_for_solver": bool(validation.get("admitted")) and not duplicate,
            "duplicate_fingerprint": fingerprint,
            "duplicate_disposition": review.get("duplicate_disposition", {"status": "UNIQUE"})})
        if duplicate:
            duplicate_rows.append({"candidate_id": cid, "kept_candidate_id": prior,
                                   "fingerprint": fingerprint, "disposition": "HOLD_EXACT_DUPLICATE"})
    review_list = [review_by_id[c["candidate_id"]] for c in candidates]
    for c, gate_row in zip(candidates, gate_rows):
        if not gate_row["eligible_for_solver"]:
            review = review_by_id[c["candidate_id"]]
            details = [f"review_decision:{review.get('decision')}"]
            details.extend(str(issue) for issue in review.get("issues", []))
            if gate_row["duplicate_disposition"].get("status") != "UNIQUE":
                details.append("exact_duplicate:" + str(gate_row["duplicate_disposition"].get("kept_candidate_id")))
            details.extend(str(error) for error in gate_row["review_validation"].get("review_errors", []))
            failures.append(slot_failure(c["slot_id"], "final content gate not admitted: " + "; ".join(details),
                                         c["root_id"], c["arm"], "content_gate"))
    failure_doc = {"schema_version": "fgr20_slot_failures_v1", "planned_slots": len(root_order) * len(ARMS) * len(SLOTS),
                   "failures": failures, "failure_count": len(failures),
                   "candidate_count": len(candidates), "parseable_candidates_retained": True}
    author_inputs = {}
    all_thread_ids: dict[str, str] = {}
    cli_artifact_bindings: list[dict[str, Any]] = []
    for arm in ARMS:
        call_records = []
        thread_ids = set()
        for root in root_order:
            work = stage_roots / arm / "roots" / root_number(root)
            if not work.exists():
                continue
            if root not in source_rows:
                if list(work.glob("*_CALL*.json")):
                    raise RuntimeError(f"Question call exists without a frozen assigned source: {root}/{arm}")
                continue
            for call_path in sorted(work.glob("*_CALL*.json")):
                record = author_call_input_record(call_path, root, arm, source_rows[root],
                    contract_binding, feedback_binding if arm == "F" else None,
                    adapter_binding if arm == "F" else None)
                if record["thread_id"] in thread_ids:
                    raise RuntimeError(f"Fresh question/review calls reused a thread in arm {arm}")
                prior_arm = all_thread_ids.get(record["thread_id"])
                if prior_arm is not None and prior_arm != arm:
                    raise RuntimeError(f"Fresh question/review call thread reused across arms: {record['thread_id']}")
                thread_ids.add(record["thread_id"])
                all_thread_ids[record["thread_id"]] = arm
                call_records.append(record)
        incomplete_calls, arm_bindings = _arm_call_artifacts(arm)
        cli_artifact_bindings.extend(arm_bindings)
        author_inputs[arm] = {
            "schema_version": "fgr20_question_author_input_v2_call_specific",
            "arm": arm,
            "calls": call_records,
            "failed_or_interrupted_calls": incomplete_calls,
            "all_cli_call_and_process_registry_bindings": arm_bindings,
            "fresh_session_isolation": True,
            "input_scope": "Each row records one actual saved prompt, its one assigned root/source, and only materials embedded in that prompt.",
            "g_author_input_contains_no_f_material": arm == "G",
        }
    feedback_ledger = []
    candidate_by_slot = {(c["root_id"], c["slot_id"]): c for c in candidates if c.get("arm") == "F"}
    admitted_ids = {row["candidate_id"] for row in gate_rows if row.get("eligible_for_solver")}
    for root in root_order:
        plan_path = stage_roots / "F" / "roots" / root_number(root) / "TASK_PLANS.json"
        plans = read(plan_path).get("plans", []) if plan_path.exists() else []
        for slot in slot_order:
            plan = next((p for p in plans if p.get("slot_id") == slot), {})
            target = plan.get("feedback_target", {}) if isinstance(plan, dict) else {}
            if not isinstance(target, dict):
                target = {}
            candidate = candidate_by_slot.get((root, slot))
            if root not in source_rows:
                assessment = "UNIMPLEMENTED_NO_PASSING_SOURCE"
            elif not plan_path.exists() or not plan:
                assessment = "UNIMPLEMENTED_NO_PLAN"
            elif target.get("target_status") == "TARGET_HOLD":
                assessment = "UNIMPLEMENTED_TARGET_HOLD"
            elif candidate is None:
                assessment = "UNIMPLEMENTED_NO_CANDIDATE"
            elif candidate["candidate_id"] in admitted_ids:
                assessment = "CANDIDATE_ADMITTED; TARGET REALIZATION REMAINS PROSPECTIVE"
            else:
                assessment = "CANDIDATE_NOT_ADMITTED; TARGET REALIZATION UNIMPLEMENTED"
            feedback_ledger.append({
                "candidate_id": candidate.get("candidate_id") if candidate else None,
                "root_id": root, "slot_id": slot,
                "feedback_id": target.get("feedback_id"),
                "trigger_realization": target.get("trigger_realization"),
                "predicted_observable_error": target.get("predicted_observable_error"),
                "counterexample": target.get("counterexample"),
                "target_acceptance_test": target.get("target_acceptance_test"),
                "target_status": target.get("target_status", "TARGET_HOLD"),
                "plan_present": bool(plan), "candidate_present": candidate is not None,
                "candidate_admitted": bool(candidate and candidate["candidate_id"] in admitted_ids),
                "status_basis": "frozen prospective plan and pre-Solver content gate only; no Solver output or post-hoc score used",
                "implementation_assessment": assessment,
            })

    candidate_path = BASE / "CANDIDATES_PRIVATE.json"
    review_path = BASE / "CONTENT_REVIEW_AI.json"
    failure_path = BASE / "SLOT_FAILURES.json"
    duplicate_path = BASE / "DUPLICATE_DISPOSITIONS.json"
    gate_path = BASE / "CONTENT_GATE_FINAL.json"
    author_f = BASE / "question_authoring" / "F" / "AUTHOR_INPUT.json"
    author_g = BASE / "question_authoring" / "G" / "AUTHOR_INPUT.json"
    ledger_path = BASE / "question_authoring" / "F" / "FEEDBACK_CONSUMPTION_LEDGER.json"
    publish(author_f, author_inputs["F"])
    publish(author_g, author_inputs["G"])
    publish(ledger_path, {"schema_version": "fgr20_feedback_consumption_ledger_v1",
                          "status": "PROSPECTIVE_ONLY_NO_SOLVER_RESULTS_CONSUMED", "entries": feedback_ledger})
    publish(candidate_path, candidates)
    publish(review_path, {"schema_version": "fgr20_content_review_ai_v1", "reviews": review_list,
                          "review_process": "blind independent Gold followed by separate content verification",
                          "reviewers_are_same_codex_family_ai_not_human": True})
    publish(failure_path, failure_doc)
    publish(duplicate_path, {"schema_version": "fgr20_duplicate_dispositions_v1", "policy": "type + whitespace-normalized question + sorted six item texts; labels/order ignored",
                             "fixed_retention_order": "CONFIG root order, slot order, F before G", "duplicates": duplicate_rows})
    publish(gate_path, {"schema_version": "fgr20_content_gate_final_v1", "candidate_count": len(candidates),
                        "mechanical_pass_count": sum(g["mechanical_audit"].get("mechanical_pass", False) for g in gate_rows),
                        "admitted_count": sum(g["eligible_for_solver"] for g in gate_rows),
                        "semantic_review_required": True, "training_launched": False,
                        "records": gate_rows})
    freeze_bindings = [binding(candidate_path), binding(review_path), binding(selection_path),
        binding(BASE / "PROTOCOL_ZH.md"), binding(BASE / "CONFIG.json"), contract_binding,
        binding(Path(__file__).resolve()), binding(Path(cc.__file__).resolve()),
        binding(REPO / "scripts" / "current_method" / "production_question_gate.py"),
        binding(author_f), binding(author_g), binding(ledger_path), binding(failure_path),
        binding(duplicate_path), binding(gate_path)]
    freeze_bindings.extend(binding(path) for path in EXTRA_DRIVER_BINDINGS)
    freeze_bindings.extend(b for b in (feedback_binding, adapter_binding) if b is not None)
    for path in (SOURCE_FREEZE_PATH, SOURCE_SELECTION_PATH):
        if path.exists():
            freeze_bindings.append(binding(path))
    for p in sorted(stage_roots.glob("*/roots/*/*")):
        if p.is_file() and p.name != ".driver.lock":
            freeze_bindings.append(binding(p))
    freeze_bindings.extend(cli_artifact_bindings)
    freeze_bindings = sorted({b["path"]: b for b in freeze_bindings}.values(), key=lambda b: b["path"])
    freeze = {"schema_version": "fgr20_content_freeze_v1", "status": "FROZEN_BEFORE_SOLVER",
              "candidate_count": len(candidates), "admitted_candidate_count": sum(g["eligible_for_solver"] for g in gate_rows),
              "planned_slot_count": len(root_order) * len(ARMS) * len(SLOTS), "slot_failure_count": len(failures),
              "bindings": freeze_bindings, "duplicate_binding": binding(duplicate_path),
              "content_gate_binding": binding(gate_path),
              "freeze_note": "No Solver response or score was read during authoring, review, or duplicate handling."}
    freeze_path = BASE / "CONTENT_FREEZE.json"
    publish(freeze_path, freeze)
    cc.progress("QUESTION_CONTENT_FROZEN", candidates=len(candidates), admitted=freeze["admitted_candidate_count"], failures=len(failures))
    return freeze


def run(args: argparse.Namespace) -> int:
    config = read(BASE / "CONFIG.json")
    selection_path = SOURCE_SELECTION_PATH
    _, source_rows = load_selection(selection_path)
    contract_path = BASE / "question_authoring" / "COMMON_AUTHOR_CONTRACT_ZH.md"
    contract = contract_path.read_text(encoding="utf-8")
    contract_binding = binding(contract_path)
    if args.arm:
        arms = args.arm
    else:
        arms = list(ARMS)
    feedback = adapter = feedback_binding = adapter_binding = None
    if "F" in arms:
        feedback, adapter, feedback_binding, adapter_binding = feedback_material()
    roots = [root_id(r) for r in (args.root or config["roots"])]
    roots = [r for r in roots if r in source_rows]
    if not roots:
        raise RuntimeError("No selected passing source roots to author")
    max_calls = min(3, int(config.get("max_parallel_cli_calls", 3)))
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_calls) as pool:
        for root in roots:
            for arm in arms:
                futures.append(pool.submit(process_root_arm, root, arm, source_rows[root], selection_path,
                    contract, contract_binding, feedback if arm == "F" else None,
                    adapter if arm == "F" else None, feedback_binding if arm == "F" else None,
                    adapter_binding if arm == "F" else None))
        failed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                failed += result["status"].startswith("FAILED")
            except Exception as exc:
                failed += 1
                cc.progress("QUESTION_ROOT_ARM_EXCEPTION", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps({"roots": len(roots), "arms": arms, "root_arm_failures": failed}, ensure_ascii=False))
    return 2 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="author/review selected complete sources")
    run_parser.add_argument("--root", action="append", help="root number or full root id; repeatable")
    run_parser.add_argument("--arm", action="append", choices=ARMS, help="arm; repeatable")
    sub.add_parser("finalize", help="assemble immutable content, gate, ledgers, and freeze")
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    finalize()
    print(json.dumps({"status": "FROZEN_BEFORE_SOLVER", "path": str(BASE / "CONTENT_FREEZE.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
