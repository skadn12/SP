"""One-root paid Dynamic Planner to Realizer generation canary."""

from __future__ import annotations

import collections
import difflib
import json
import re
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .ledger import RequestLedger
from .lineage_three_stage_canary import ApiBudget, run_single_object_task
from .live_backends import (
    LiveRealizerBackend, LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient, load_challenger_config,
)
from .questioner_contract import build_span_catalog
from .run_canary import create_source_snapshot, git_value, project_diff_sha256
from .stateful_graph_expansion import (
    STATEFUL_COMPLETION_CONTRACT_VERSION,
    STATEFUL_GRAPH_EXPANSION_VERSION, STATEFUL_GRAPH_LEDGER_VERSION,
    STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
    STATEFUL_GRAPH_SCENE_CONTRACT_VERSION, STANDARD_STAGE_SCHEDULE,
    STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
    STATEFUL_HANDOFF_WHITELIST_VERSION,
    STATEFUL_REALIZER_JSON_SCHEMA,
    STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
    build_stateful_graph_realizer_prompt, canonical_sha256,
    compile_stateful_scene_contracts, initialize_stateful_ledger,
    normalize_terminal_marker_text, validate_stateful_outline,
    cross_arc_stage_conflict_gate,
)
from .stateful_graph_orchestrator import (
    STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
    StatefulGraphRoundOrchestrator,
)
from .stateful_outline_planner import (
    RAW_STATEFUL_OUTLINE_SCHEMA, STATEFUL_OUTLINE_CHUNKED_VERSION,
    STATEFUL_OUTLINE_E1_TERMINAL_INTERFACE_VERSION,
    STATEFUL_OUTLINE_E1_RETRY_FREEZE_VERSION,
    STATEFUL_OUTLINE_EVENT_IDENTITY_INTERFACE_VERSION,
    STATEFUL_OUTLINE_CROSS_ARC_REPAIR_VERSION,
    STATEFUL_OUTLINE_OVERVIEW_ANCHOR_SELECTION_VERSION,
    STATEFUL_OUTLINE_OVERVIEW_PARTICIPANT_INTERFACE_VERSION,
    STATEFUL_OUTLINE_SCHEMA_VERSION, build_dynamic_stateful_planner_input,
    StatefulOutlineArcRetryValidator, CrossArcStageConflictRepairValidator,
    build_cross_arc_stage_conflict_repair_suffix,
    build_stateful_outline_e1_repair_suffix,
    build_stateful_outline_overview_interface_repair_suffix,
    build_stateful_outline_overview_anchor_repair_suffix,
    build_stateful_outline_arc_prompt, build_stateful_outline_overview_prompt,
    normalize_stateful_outline, validate_raw_stateful_outline,
    validate_stateful_outline_arc, validate_stateful_outline_overview,
)
from .unified_dialogue_root import (
    read_unified_parent, validate_unified_dialogue_root,
)


DYNAMIC_GENERATION_CANARY_VERSION = "cc_dynamic_generation_canary_v1"
FRESH_E2E_AUDIT_VERSION = "cc_fresh_e2e_final_audit_v1"
CHALLENGER_MODEL = "Qwen/Qwen3-32B"
CHALLENGER_CONTEXT_LIMIT = 40_960
OUTLINE_OVERVIEW_MAX_TOKENS = 2_500
OUTLINE_ARC_MAX_TOKENS = 3_200
REALIZER_MAX_TOKENS = 4_096
PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent


def _tokens(tokenizer: Any, turns: Sequence[Mapping[str, Any]]) -> int:
    text = "\n".join(
        f"[{turn.get('turn_id', '')}] {turn['speaker']}: {turn['text']}"
        for turn in turns)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _arc_scene_indices(arc_index: int) -> tuple[int, int, int, int]:
    assigned: list[int | None] = [None, None, None, None]
    for scene_index, schedule in enumerate(STANDARD_STAGE_SCHEDULE, 1):
        for scheduled_arc, stage_index in schedule:
            if scheduled_arc == arc_index - 1:
                assigned[stage_index] = scene_index
    if any(value is None for value in assigned):
        raise RuntimeError("stateful_arc_schedule_incomplete")
    return tuple(int(value) for value in assigned)


def _planner_view(parent: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dialogue = [dict(turn) for turn in parent]
    catalog = build_span_catalog(dialogue)
    return {
        "combined_dialogue": dialogue, "span_catalog": catalog,
        "view_sha256": canonical_sha256(
            {"dialogue": dialogue, "span_catalog": catalog}),
    }


def api_usage(run_root: Path) -> dict[str, Any]:
    by_intent: collections.Counter[str] = collections.Counter()
    tokens: collections.Counter[str] = collections.Counter()
    retries = 0
    states = []
    for path in sorted(run_root.glob("**/ledger/requests/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        states.append(row)
        if row.get("status") != "completed":
            continue
        by_intent[str(row.get("intent") or "unknown")] += 1
        usage = (row.get("response_metadata") or {}).get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            tokens[key] += int(usage.get(key) or 0)
        retries += int(int(row.get("attempt") or 1) > 1)
    return {
        "physical_api_requests": sum(by_intent.values()),
        "completed_requests_by_intent": dict(sorted(by_intent.items())),
        "ambiguous_request_count": sum(row.get("status") != "completed"
                                       for row in states),
        "retry_request_count": retries,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "reused_response_count": sum(bool(
            (row.get("response_metadata") or {}).get("reused"))
            for row in states),
        "usage_by_intent": {
            intent: {
                key: sum(
                    int(((row.get("response_metadata") or {}).get("usage") or
                         {}).get(key) or 0)
                    for row in states
                    if row.get("status") == "completed" and
                    str(row.get("intent") or "unknown") == intent)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            for intent in sorted(by_intent)
        },
    }


def _normalized_utterance(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", str(text).casefold()))


def audit_child_naturalness(
    *, sessions: Sequence[Mapping[str, Any]],
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deterministic soft-quality metrics; never changes acceptance."""
    contract_by_scene = {str(row["scene_id"]): row for row in contracts}
    utterances: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    marker_extra_occurrences = 0
    per_scene: list[dict[str, Any]] = []
    for session in sessions:
        scene_id = str(session.get("graph_scene_id") or "")
        turns = list(session.get("dialogue_turns") or [])
        contract = contract_by_scene.get(scene_id) or {}
        for index, turn in enumerate(turns):
            utterances.append({
                "scene_id": scene_id, "turn_index": index,
                "turn_id": turn.get("turn_id"),
                "text": str(turn.get("text") or ""),
                "normalized": _normalized_utterance(turn.get("text") or ""),
            })
        updates = {
            str(row.get("stage_id")): row for row in
            (session.get("authoritative_stage_updates") or
             session.get("candidate_stage_updates") or [])
        }
        scene_evidence_count = 0
        for stage in contract.get("assigned_event_stages") or []:
            stage_id = str(stage["stage_id"])
            update = updates.get(stage_id) or {}
            index = update.get("evidence_turn_index")
            if isinstance(index, int) and 0 <= index < len(turns):
                scene_evidence_count += 1
                evidence_rows.append({
                    "scene_id": scene_id, "stage_id": stage_id,
                    "turn_index": index,
                    "turn_id": turns[index].get("turn_id"),
                    "text": str(turns[index].get("text") or ""),
                    "normalized": _normalized_utterance(
                        turns[index].get("text") or ""),
                })
            for phrase in stage.get("completion_evidence_phrases") or []:
                marker = normalize_terminal_marker_text(phrase)
                occurrences = sum(
                    marker in normalize_terminal_marker_text(
                        turn.get("text") or "") for turn in turns)
                marker_extra_occurrences += max(0, occurrences - 1)
        per_scene.append({
            "scene_id": scene_id,
            "turn_count": len(turns),
            "evidence_turn_count": scene_evidence_count,
            "manual_review_status": "pending",
            "manual_naturalness_note": "",
        })
    utterance_counts = collections.Counter(
        row["normalized"] for row in utterances if row["normalized"])
    exact_repeated = sum(count - 1 for count in utterance_counts.values()
                         if count > 1)
    near_duplicate_pairs = []
    for left_index, left in enumerate(utterances):
        if not left["normalized"]:
            continue
        for right in utterances[left_index + 1:]:
            if (not right["normalized"] or
                    left["normalized"] == right["normalized"]):
                continue
            ratio = difflib.SequenceMatcher(
                None, left["normalized"], right["normalized"]).ratio()
            if ratio >= 0.92:
                near_duplicate_pairs.append({
                    "left_turn_id": left["turn_id"],
                    "right_turn_id": right["turn_id"],
                    "similarity": round(ratio, 6),
                })
    evidence_counts = collections.Counter(
        row["normalized"] for row in evidence_rows if row["normalized"])
    repeated_evidence_texts = {
        text for text, count in evidence_counts.items() if count > 1}
    repeated_evidence_scenes = {
        row["scene_id"] for row in evidence_rows
        if row["normalized"] in repeated_evidence_texts}
    scene_texts = [{
        "scene_id": str(session.get("graph_scene_id") or ""),
        "text": " ".join(_normalized_utterance(turn.get("text") or "")
                           for turn in session.get("dialogue_turns") or []),
    } for session in sessions]
    template_like_scenes: set[str] = set()
    for left_index, left in enumerate(scene_texts):
        for right in scene_texts[left_index + 1:]:
            if difflib.SequenceMatcher(
                    None, left["text"], right["text"]).ratio() >= 0.90:
                template_like_scenes.update(
                    (left["scene_id"], right["scene_id"]))
    evidence_refs = {(row["scene_id"], row["turn_index"])
                     for row in evidence_rows}
    return {
        "audit_version": FRESH_E2E_AUDIT_VERSION,
        "duplicate_terminal_evidence_count": sum(
            count - 1 for count in evidence_counts.values() if count > 1),
        "duplicate_terminal_evidence_scene_count": len(
            repeated_evidence_scenes),
        "duplicate_marker_turn_count": marker_extra_occurrences,
        "exact_repeated_utterance_count": exact_repeated,
        "near_duplicate_utterance_count": len(near_duplicate_pairs),
        "near_duplicate_utterance_pairs": near_duplicate_pairs,
        "template_like_scene_count": len(template_like_scenes),
        "template_like_scene_ids": sorted(template_like_scenes),
        "unique_evidence_turn_rate": (
            len(evidence_refs) / len(evidence_rows) if evidence_rows else 0.0),
        "evidence_turn_count": len(evidence_rows),
        "scene_manual_review": per_scene,
        "soft_quality_only": True,
    }


def _read_sessions(realizer_root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in
            sorted((realizer_root / "accepted").glob("session_*.json"))]


def _request_audit(run_root: Path) -> None:
    raw_api = run_root / "raw_api"
    raw_api.mkdir(parents=True, exist_ok=True)
    event_lines = []
    for stage in ("planner", "realizer"):
        stage_root = run_root / stage
        events = stage_root / "ledger/request_events.jsonl"
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    row["canary_stage"] = stage
                    event_lines.append(json.dumps(
                        row, ensure_ascii=False, sort_keys=True))
        for state_path in sorted((stage_root / "ledger/requests").glob("*.json")):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            raw_path = state.get("raw_response_path")
            if raw_path and Path(raw_path).is_file():
                target = raw_api / stage / f"{state_path.stem}.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw_path, target)
                atomic_json(raw_api / stage / f"{state_path.stem}.metadata.json", {
                    "request_id": state_path.stem,
                    "raw_response_sha256": sha256_file(target),
                    "response_metadata": state.get("response_metadata"),
                    "final_decision": state.get("final_decision"),
                })
    atomic_text(run_root / "request_ledger.jsonl",
                "\n".join(event_lines) + ("\n" if event_lines else ""))


def audit_child_generation(
    *, parent: Sequence[Mapping[str, Any]], outline: Mapping[str, Any] | None,
    contracts: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]],
    final_ledger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    child = [dict(turn) for session in sessions
             for turn in session.get("dialogue_turns") or []]
    parent_turn_ids = [str(turn["turn_id"]) for turn in parent]
    child_turn_ids = [str(turn["turn_id"]) for turn in child]
    parent_session_ids = list(dict.fromkeys(
        str(turn["session_id"]) for turn in parent))
    child_session_ids = list(dict.fromkeys(
        str(turn["session_id"]) for turn in child))
    accepted_session_ids = [str(session.get("session_id") or "")
                            for session in sessions]
    parent_people = {str(turn["speaker"]) for turn in parent}
    new_people = ({str(turn["speaker"]) for turn in child} - parent_people)
    scene_by_id = {str(contract["scene_id"]): contract for contract in contracts}
    isolated = []
    for person in sorted(new_people):
        related = False
        for session in sessions:
            if person not in {str(turn["speaker"]) for turn in
                              session.get("dialogue_turns") or []}:
                continue
            contract = scene_by_id.get(str(session.get("graph_scene_id"))) or {}
            participants = set(contract.get("required_participants") or [])
            if (participants & parent_people or
                    contract.get("parent_anchor_evidence")):
                related = True
                break
        if not related:
            isolated.append(person)
    normalized_scenes = [" ".join(
        " ".join(str(turn.get("text") or "").casefold().split())
        for turn in session.get("dialogue_turns") or [])
        for session in sessions]
    repeated_scene_count = len(normalized_scenes) - len(set(normalized_scenes))
    reports = []
    realizer_root = Path(str(sessions[0].get("_realizer_root"))) \
        if sessions and sessions[0].get("_realizer_root") else None
    if realizer_root:
        reports = [json.loads(path.read_text()) for path in
                   realizer_root.glob("reports/stateful-realizer-*.json")]
    contradiction_reasons = {
        "pre_state_ledger_mismatch", "state_instance_identity_drift",
        "completed_state_instance_reopened", "result_state_not_allowed",
    }
    state_contradictions = sum(
        any(reason in contradiction_reasons for reason in row.get("reasons") or [])
        for row in reports)
    stage_states = (final_ledger or {}).get("stage_states") or {}
    evidence_stage_count = sum(
        row.get("status") in {"completed", "failed"} and
        row.get("evidence_turn_index") is not None
        for row in stage_states.values())
    expected_stages = len(stage_states)
    stage_coverage = (evidence_stage_count / expected_stages
                      if expected_stages else 0.0)
    continuity = bool(outline) and all(
        contract.get("parent_anchor_evidence") for contract in contracts)
    expected_child_session_ids = [
        f"session_{index:02d}" for index in range(1, len(sessions) + 1)]
    unauthorized_speaker_rows = []
    for session in sessions:
        scene_id = str(session.get("graph_scene_id") or "")
        contract = scene_by_id.get(scene_id) or {}
        allowed = set(contract.get("allowed_speakers") or
                      contract.get("required_participants") or [])
        for turn in session.get("dialogue_turns") or []:
            if str(turn.get("speaker") or "") not in allowed:
                unauthorized_speaker_rows.append({
                    "scene_id": scene_id,
                    "turn_id": turn.get("turn_id"),
                    "speaker": turn.get("speaker"),
                })
    durable_rows = [{
        "scene_id": str(session.get("graph_scene_id") or ""),
        "candidate": candidate,
    } for session in sessions
        for candidate in session.get("durable_assertion_candidates") or []]
    residual_statuses = {
        stage_id: row.get("status") for stage_id, row in stage_states.items()
        if row.get("status") not in {"completed", "failed"}}
    return {
        "parent_session_count": len(parent_session_ids),
        "child_session_count": len(child_session_ids),
        "parent_turn_count": len(parent), "child_turn_count": len(child),
        "parent_turn_order_preserved": parent_turn_ids == [
            str(turn["turn_id"]) for turn in parent],
        "parent_child_turn_ids_disjoint": not (
            set(parent_turn_ids) & set(child_turn_ids)),
        "parent_child_session_ids_disjoint": not (
            set(parent_session_ids) & set(child_session_ids)),
        "child_turn_ids_unique": (
            len(child_turn_ids) == len(set(child_turn_ids))),
        "child_session_ids_unique": (
            len(accepted_session_ids) == len(set(accepted_session_ids)) and
            child_session_ids == accepted_session_ids),
        "child_session_ids_in_expected_order": (
            child_session_ids == expected_child_session_ids),
        "new_characters": sorted(new_people),
        "isolated_new_characters": isolated,
        "isolated_new_character_count": len(isolated),
        "scene_parent_continuity": continuity,
        "planned_stage_count": expected_stages,
        "evidence_bearing_stage_count": evidence_stage_count,
        "planner_event_evidence_coverage": stage_coverage,
        "state_contradiction_count": state_contradictions,
        "exact_repeated_or_template_scene_count": repeated_scene_count,
        "unauthorized_speaker_count": len(unauthorized_speaker_rows),
        "unauthorized_speaker_rows": unauthorized_speaker_rows,
        "unauthorized_durable_fact_count": len(durable_rows),
        "unauthorized_durable_fact_rows": durable_rows,
        "residual_nonterminal_stage_count": len(residual_statuses),
        "residual_nonterminal_stage_statuses": residual_statuses,
        "child_turns": child,
    }


def _scene_results(
    realizer_root: Path, sessions: Sequence[Mapping[str, Any]], *,
    not_started_reason: str | None = None,
) -> list[dict[str, Any]]:
    accepted = {str(row["graph_scene_id"]): row for row in sessions}
    result = []
    for index in range(1, 11):
        scene_id = f"S{index}"
        row = accepted.get(scene_id)
        rejected = []
        for path in sorted(realizer_root.glob(
                f"reports/stateful-realizer-s{index:02d}-a*.json")):
            record = json.loads(path.read_text())
            if record.get("decision") == "reject":
                rejected.append({"request_id": record["request_id"],
                                 "reasons": record.get("reasons") or []})
        result.append({
            "scene_id": scene_id,
            "success": row is not None,
            "accepted_attempt": row.get("accepted_attempt") if row else None,
            "accepted_turn_count": len(row.get("dialogue_turns") or [])
            if row else 0,
            "accepted_tokens": row.get("dialogue_token_count") if row else 0,
            "failed_attempts": rejected,
            "not_started_reason": not_started_reason if row is None else None,
        })
    return result


def _realizer_attempt_accounting(realizer_root: Path) -> dict[str, Any]:
    attempts = []
    for path in sorted((realizer_root / "ledger/requests").glob(
            "stateful-realizer-s*-a*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        report_path = realizer_root / f"reports/{path.name}"
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {})
        attempts.append({
            "request_id": state.get("request_id"),
            "scene_id": (state.get("request_metadata") or {}).get("scene_id"),
            "attempt": state.get("attempt"),
            "decision": report.get("decision") or state.get("final_decision"),
            "errors": list(report.get("reasons") or []),
            "actual_dialogue_tokens": report.get("actual_dialogue_tokens"),
            "actual_turn_count": report.get("actual_turn_count"),
            "metadata_only_retry": bool(report.get(
                "metadata_only_dialogue_preservation_required")),
            "metadata_only_dialogue_preserved": report.get(
                "metadata_only_dialogue_preserved"),
            "dialogue_turns_sha_before": report.get(
                "dialogue_turns_sha_before"),
            "dialogue_turns_sha_after": report.get(
                "dialogue_turns_sha_after"),
            "usage": (state.get("response_metadata") or {}).get("usage") or {},
        })
    return {
        "attempts": attempts,
        "normal_retry_count": sum(
            int(row.get("attempt") or 1) > 1 and
            not row["metadata_only_retry"] for row in attempts),
        "metadata_only_retry_count": sum(
            row["metadata_only_retry"] for row in attempts),
    }


def _render_summary(summary: Mapping[str, Any]) -> str:
    usage = summary["api_usage"]
    audit = summary["generation_audit"]
    lines = [
        "# Conversation Chronicles Generation Canary", "",
        f"- decision: `{summary['decision']}`",
        f"- generation_ready: `{str(summary['generation_ready']).lower()}`",
        f"- root: `{summary['root_id']}`",
        f"- API requests: `{usage['physical_api_requests']}`",
        f"- total API tokens: `{usage['total_tokens']}`",
        f"- parent: `{audit['parent_session_count']}` sessions / "
        f"`{audit['parent_turn_count']}` turns",
        f"- child: `{audit['child_session_count']}` sessions / "
        f"`{audit['child_turn_count']}` turns",
        f"- event evidence coverage: "
        f"`{audit['planner_event_evidence_coverage']:.3f}`", "",
        "| Scene | Success | Accepted attempt | Turns | Tokens | Failed attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["realizer_scenes"]:
        lines.append(
            f"| `{row['scene_id']}` | `{str(row['success']).lower()}` | "
            f"{row['accepted_attempt'] or '-'} | {row['accepted_turn_count']} | "
            f"{row['accepted_tokens']} | {len(row['failed_attempts'])} |")
    lines.extend(["", "No Questioner, Solver, Verifier, or Training call was made.", ""])
    return "\n".join(lines)


def _finalize(
    *, run_root: Path, input_path: Path, input_sha: str,
    parent: Sequence[Mapping[str, Any]], parent_hash: str, tokenizer: Any,
    manifest: Mapping[str, Any], planner_status: str,
    graph_validation: Mapping[str, Any], failure: str | None,
) -> dict[str, Any]:
    realizer_root = run_root / "realizer"
    realizer_root.mkdir(parents=True, exist_ok=True)
    sessions = _read_sessions(realizer_root)
    for session in sessions:
        session["_realizer_root"] = str(realizer_root)
    outline_path = run_root / "planner/stateful_outline.json"
    outline = json.loads(outline_path.read_text()) if outline_path.exists() else None
    contracts_path = run_root / "planner/stateful_scene_contracts.json"
    contracts = ((json.loads(contracts_path.read_text()).get("contracts") or [])
                 if contracts_path.exists() else [])
    ledger_path = realizer_root / "frozen/final_state_ledger.json"
    if not ledger_path.exists():
        ledgers = sorted(realizer_root.glob("frozen/state_ledger_after_*.json"))
        ledger_path = ledgers[-1] if ledgers else Path("")
    final_ledger = (json.loads(ledger_path.read_text())
                    if str(ledger_path) and ledger_path.is_file() else None)
    completion_path = realizer_root / "reports/stateful_completion_report.json"
    completion = (json.loads(completion_path.read_text())
                  if completion_path.exists() else {"complete": False})
    audit = audit_child_generation(
        parent=parent, outline=outline, contracts=contracts,
        sessions=sessions, final_ledger=final_ledger)
    child = audit.pop("child_turns")
    audit["parent_tokens"] = _tokens(tokenizer, parent)
    audit["child_tokens"] = _tokens(tokenizer, child) if child else 0
    naturalness = audit_child_naturalness(
        sessions=sessions, contracts=contracts)
    assembly_preconditions = (
        len(sessions) == 10 and completion.get("complete") is True)
    child_dialogue_assembled = False
    if assembly_preconditions:
        atomic_json(run_root / "child_dialogue.json", {
            "schema_version": "cc_dynamic_child_dialogue_v1",
            "root_id": manifest["root_id"],
            "parent_dialogue_hash": parent_hash,
            "session_count": audit["child_session_count"],
            "turn_count": len(child), "dialogue_tokens": audit["child_tokens"],
            "turns": child,
        })
        child_dialogue_assembled = True
    atomic_json(run_root / "reports/naturalness_observation.json",
                naturalness)
    naturalness_lines = [
        "# Naturalness observation (soft quality only)", "",
        "These fields do not change the formal generation decision.", "",
        "| Scene | Turns | Evidence turns | Manual status | Notes |",
        "|---|---:|---:|---|---|",
    ]
    for row in naturalness["scene_manual_review"]:
        naturalness_lines.append(
            f"| `{row['scene_id']}` | {row['turn_count']} | "
            f"{row['evidence_turn_count']} | pending |  |")
    atomic_text(run_root / "reports/naturalness_observation.md",
                "\n".join(naturalness_lines) + "\n")
    _request_audit(run_root)
    usage = api_usage(run_root)
    realizer_started = (realizer_root / "config/run_manifest.json").is_file()
    if not realizer_started:
        atomic_json(realizer_root / "reports/not_started.json", {
            "decision": "not_started", "reason": failure or planner_status,
            "api_calls": 0})
    scene_results = _scene_results(
        realizer_root, sessions,
        not_started_reason=None if realizer_started else
        "not_called_due_to_planner_failure")
    realizer_accounting = _realizer_attempt_accounting(realizer_root)
    attempts_by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(
        list)
    for attempt in realizer_accounting["attempts"]:
        attempts_by_scene[str(attempt.get("scene_id") or "")].append(attempt)
    for row in scene_results:
        attempts = attempts_by_scene.get(row["scene_id"], [])
        row["request_count"] = len(attempts)
        row["first_failure_reasons"] = (
            attempts[0]["errors"] if attempts and attempts[0]["errors"] else [])
        row["final_result"] = (
            "accept" if row["success"] else
            "not_called" if not attempts else "retry_exhausted")
    overview_task_path = run_root / "planner/overview_task.json"
    overview_task = (json.loads(overview_task_path.read_text())
                     if overview_task_path.exists() else {})
    planner_attempts = [{
        "request_id": row.get("request_id"),
        "decision": row.get("decision"),
        "validation_decision": (row.get("validation") or {}).get("decision"),
        "validation_reasons": (row.get("validation") or {}).get("reasons") or [],
        "response_metadata": row.get("response_metadata") or {},
    } for row in overview_task.get("attempts") or []]
    proposed_scene_count = 0
    raw_overview_paths = sorted((run_root /
        "planner/raw/stateful_outline_overview").glob("*.json"))
    if raw_overview_paths:
        raw_wrapper = json.loads(raw_overview_paths[-1].read_text())
        try:
            proposed = json.loads(raw_wrapper.get("raw_response") or "")
        except json.JSONDecodeError:
            proposed = {}
        proposed_scene_count = len(proposed.get("scenes") or []) \
            if isinstance(proposed, dict) else 0
    input_unchanged = sha256_file(input_path) == input_sha
    no_forbidden_intents = all(
        "questioner" not in name and "solver" not in name and
        "verifier" not in name and "training" not in name
        for name in usage["completed_requests_by_intent"])
    provenance_checks = {
        "git_head_match": git_value(PROJECT, "rev-parse", "HEAD") ==
        manifest["project_git_head"],
        "project_diff_sha256_match": project_diff_sha256(PROJECT) ==
        manifest["project_diff_sha256"],
        "source_snapshot_match": sha256_file(Path(
            manifest["source_snapshot_manifest_path"])) ==
        manifest["source_snapshot_manifest_sha256"],
        "input_unified_root_unchanged": input_unchanged,
        "fresh_no_resume": manifest["resume"] is False,
        "no_reused_api_response": usage["reused_response_count"] == 0,
        "no_imported_response": manifest.get("imported_response_count") == 0,
        "no_forbidden_generation_stage": no_forbidden_intents,
    }
    provenance_pass = all(provenance_checks.values())
    core_pass = all((
        planner_status == "complete",
        graph_validation.get("valid") is True,
        completion.get("complete") is True,
        len(sessions) == 10,
        audit["parent_session_count"] == 5,
        audit["parent_turn_count"] == 60,
        audit["parent_child_turn_ids_disjoint"],
        audit["parent_child_session_ids_disjoint"],
        audit["child_turn_ids_unique"],
        audit["child_session_ids_unique"],
        audit["child_session_ids_in_expected_order"],
        audit["scene_parent_continuity"],
        audit["isolated_new_character_count"] == 0,
        audit["unauthorized_speaker_count"] == 0,
        audit["unauthorized_durable_fact_count"] == 0,
        audit["state_contradiction_count"] == 0,
        audit["residual_nonterminal_stage_count"] == 0,
        audit["exact_repeated_or_template_scene_count"] == 0,
        audit["planner_event_evidence_coverage"] == 1.0,
        len((final_ledger or {}).get("stage_states") or {}) == 16,
        all(row.get("status") == "completed" for row in
            ((final_ledger or {}).get("stage_states") or {}).values()),
        audit["child_tokens"] >= 3000,
        child_dialogue_assembled,
        input_unchanged, provenance_pass,
    ))
    summary = {
        "schema_version": "cc_dynamic_generation_canary_summary_v1",
        "decision": "generation_canary_complete" if core_pass else
        "generation_canary_failed",
        "generation_ready": core_pass,
        "fresh_e2e_ready": core_pass,
        "fresh_e2e_audit_version": FRESH_E2E_AUDIT_VERSION,
        "scope_claim": "planner_to_realizer_generation_chain_only",
        "root_id": manifest["root_id"], "sample_id": manifest["sample_id"],
        "lineage_id": manifest["lineage_id"], "run_id": manifest["run_id"],
        "outline_id": outline.get("outline_id") if outline else None,
        "planner_status": planner_status,
        "planner_api_calls": sum(count for intent, count in
            usage["completed_requests_by_intent"].items()
            if "stateful_outline" in intent),
        "planner_api_usage": {
            intent: row for intent, row in usage["usage_by_intent"].items()
            if "stateful_outline" in intent},
        "graph_api_calls": 0,
        "graph_api_usage": {"prompt_tokens": 0,
                            "completion_tokens": 0, "total_tokens": 0},
        "planner_outline_scene_count": len(
            (outline or {}).get("scene_plan") or []),
        "planner_proposed_scene_count": proposed_scene_count,
        "planner_attempts": planner_attempts,
        "graph_validation": dict(graph_validation),
        "realizer_api_calls": sum(count for intent, count in
            usage["completed_requests_by_intent"].items()
            if intent == "stateful_realize_scene"),
        "realizer_attempt_accounting": realizer_accounting,
        "normal_retry_count": realizer_accounting["normal_retry_count"],
        "metadata_only_retry_count": realizer_accounting[
            "metadata_only_retry_count"],
        "realizer_status": "complete" if len(sessions) == 10 else
        "not_started" if not realizer_started else "incomplete",
        "realizer_scenes": scene_results,
        "stateful_completion": completion,
        "generation_audit": audit,
        "naturalness_observation": naturalness,
        "child_dialogue_assembled": child_dialogue_assembled,
        "api_usage": usage,
        "input_unified_root_path": str(input_path),
        "input_unified_root_sha256_before": input_sha,
        "input_unified_root_sha256_after": sha256_file(input_path),
        "input_sha_unchanged": input_unchanged,
        "cache_resume_pollution": not provenance_checks[
            "no_reused_api_response"],
        "imported_response_count": manifest.get("imported_response_count", 0),
        "failure_diagnosis": failure,
        "questioner_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
    }
    atomic_json(run_root / "reports/generation_canary_summary.json", summary)
    atomic_text(run_root / "reports/generation_canary_summary.md",
                _render_summary(summary))
    atomic_json(run_root / "reports/post_run_provenance.json", {
        "schema_version": "cc_dynamic_generation_canary_provenance_v1",
        "checks": provenance_checks,
        "authoritative_release_provenance": provenance_pass,
        "provenance_mixed": False,
        "generation_ready": core_pass,
        "fresh_e2e_ready": core_pass,
        "run_manifest_sha256": sha256_file(
            run_root / "config/run_manifest.json"),
        "summary_sha256": sha256_file(
            run_root / "reports/generation_canary_summary.json"),
    })
    return summary


def run(
    *, unified_root_path: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path, tokenizer: Any,
    prior_test_results: Mapping[str, Any],
) -> dict[str, Any]:
    unified_root_path = unified_root_path.resolve()
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"generation_canary_run_root_exists:{run_root}")
    root = json.loads(unified_root_path.read_text(encoding="utf-8"))
    root_validation = validate_unified_dialogue_root(
        root, expected_session_count=5)
    if not root_validation["valid"]:
        raise ValueError("generation_canary_parent_contract_invalid:" +
                         ",".join(root_validation["errors"]))
    if (root["source"]["source_id"] != "episode-61555" or
            root["dialogue"]["turn_count"] != 60):
        raise ValueError("generation_canary_fixed_root_mismatch")
    parent_object = read_unified_parent(unified_root_path)
    parent = parent_object["dialogue"]["turns"]
    parent_hash = parent_object["dialogue"]["dialogue_hash"]
    input_sha = sha256_file(unified_root_path)
    root_id = str(root["root_id"])
    run_instance = run_root.name
    lineage_id = f"{root_id}:lineage:{run_instance}"
    run_id = f"{root_id}:dynamic-generation:{run_instance}"
    overview_prompt = build_stateful_outline_overview_prompt(
        parent_turns=parent)
    planner_input = build_dynamic_stateful_planner_input(
        parent_turns=parent, root_id=root_id,
        sample_id=parent_object["sample_id"], lineage_id=lineage_id,
        run_id=run_id, dialogue_hash=parent_hash,
        output_directory=str(run_root),
        outline_id_prefix=f"{root_id}:outline", round_index=1)
    overview_prompt_tokens = len(tokenizer.encode(
        overview_prompt, add_special_tokens=False))
    if (overview_prompt_tokens + OUTLINE_OVERVIEW_MAX_TOKENS >
            CHALLENGER_CONTEXT_LIMIT):
        raise RuntimeError("generation_canary_planner_context_overflow")
    api = replace(
        load_challenger_config(api_config_path), model=CHALLENGER_MODEL,
        temperature=0.25, enable_thinking=False, timeout_seconds=600,
        realizer_max_tokens=REALIZER_MAX_TOKENS,
        questioner_max_tokens=REALIZER_MAX_TOKENS)
    run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, run_root, False)
    diff_sha = project_diff_sha256(PROJECT)
    manifest = {
        "schema_version": "cc_dynamic_generation_canary_manifest_v1",
        "run_id": run_id, "mode": DYNAMIC_GENERATION_CANARY_VERSION,
        "root_id": root_id, "sample_id": parent_object["sample_id"],
        "lineage_id": lineage_id,
        "outline_id_prefix": planner_input["outline_id_prefix"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_head": git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256": diff_sha, **snapshot,
        "input_unified_root_path": str(unified_root_path),
        "input_unified_root_sha256": input_sha,
        "parent_dialogue_hash": parent_hash,
        "parent_session_count": 5, "parent_turn_count": 60,
        "api_config_path": str(api_config_path.resolve()),
        "api_config_sha256": sha256_file(api_config_path),
        "challenger_model": api.model,
        "sampling": {"temperature": api.temperature,
                     "enable_thinking": api.enable_thinking,
                     "overview_max_tokens": OUTLINE_OVERVIEW_MAX_TOKENS,
                     "arc_max_tokens": OUTLINE_ARC_MAX_TOKENS,
                     "realizer_max_tokens": REALIZER_MAX_TOKENS},
        "planner_prompt_version": STATEFUL_OUTLINE_CHUNKED_VERSION,
        "planner_anchor_selection_version":
        STATEFUL_OUTLINE_OVERVIEW_ANCHOR_SELECTION_VERSION,
        "planner_schema_version": STATEFUL_OUTLINE_SCHEMA_VERSION,
        "completion_contract_version":
            STATEFUL_COMPLETION_CONTRACT_VERSION,
        "planner_schema_sha256": canonical_sha256(
            RAW_STATEFUL_OUTLINE_SCHEMA),
        "graph_expansion_version": STATEFUL_GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version":
            STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
        "graph_ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
        "graph_validator_source_sha256": sha256_file(
            MODULE_ROOT / "stateful_graph_expansion.py"),
        "realizer_prompt_version": STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
        "realizer_schema_sha256": canonical_sha256(
            STATEFUL_REALIZER_JSON_SCHEMA),
        "terminal_marker_normalization_version":
            STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
        "multi_stage_retry_version":
            STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
        "planner_e1_retry_freeze_version":
            STATEFUL_OUTLINE_E1_RETRY_FREEZE_VERSION,
        "planner_event_identity_interface_version":
            STATEFUL_OUTLINE_EVENT_IDENTITY_INTERFACE_VERSION,
        "cross_arc_conflict_gate_version":
            STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
        "cross_arc_conflict_repair_version":
            STATEFUL_OUTLINE_CROSS_ARC_REPAIR_VERSION,
        "handoff_whitelist_version": STATEFUL_HANDOFF_WHITELIST_VERSION,
        "fresh_e2e_audit_version": FRESH_E2E_AUDIT_VERSION,
        "imported_response_count": 0,
        "resume": False, "n": 1,
        "budgets": {"planner_normal": 9, "planner_format_retry": 9,
                    "planner_physical_max": 18,
                    "realizer_physical_max": 30,
                    "realizer_attempts_per_scene": 3},
        "prior_test_results": dict(prior_test_results),
        "questioner_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
    }
    atomic_json(run_root / "config/run_manifest.json", manifest)
    atomic_json(run_root / "frozen/unified_root_attestation.json", {
        "path": str(unified_root_path), "sha256": input_sha,
        "validation": root_validation})
    atomic_json(run_root / "frozen/parent_contract.json", parent_object)
    atomic_json(run_root / "frozen/dynamic_planner_input.json", {
        key: value for key, value in planner_input.items()
        if key != "overview_prompt"})
    atomic_text(run_root / "planner/overview_prompt.txt", overview_prompt)
    atomic_json(run_root / "reports/preflight.json", {
        "decision": "pass", "fresh_run": True, "resume": False,
        "root_validation": root_validation,
        "parent_session_count": 5, "parent_turn_count": 60,
        "overview_prompt_tokens": overview_prompt_tokens,
        "reserved_completion_tokens": OUTLINE_OVERVIEW_MAX_TOKENS,
        "context_limit": CHALLENGER_CONTEXT_LIMIT,
        "api_calls_before_execution": 0,
        "cache_reuse_count_before_execution": 0,
        "imported_response_count_before_execution": 0,
        "summary_and_time_interval_forwarded": False,
    })
    planner_status = "not_started"
    graph_validation: dict[str, Any] = {"valid": False, "errors": []}
    client = OpenAICompatibleChallengerClient(api)
    try:
        planner_root = run_root / "planner"
        planner_manifest = {
            "run_id": f"{run_id}:planner", "root_id": root_id,
            "lineage_id": lineage_id, "mode": "stateful_outline_planner_chunked",
            "project_git_head": manifest["project_git_head"],
            "project_diff_sha256": diff_sha,
            "source_parent_sha256": canonical_sha256(parent),
            "source_parent_turn_count": len(parent),
            "source_parent_session_count": 5,
            "prompt_version": STATEFUL_OUTLINE_CHUNKED_VERSION,
            "overview_participant_interface_version":
                STATEFUL_OUTLINE_OVERVIEW_PARTICIPANT_INTERFACE_VERSION,
            "e1_terminal_interface_version":
                STATEFUL_OUTLINE_E1_TERMINAL_INTERFACE_VERSION,
            "event_identity_interface_version":
                STATEFUL_OUTLINE_EVENT_IDENTITY_INTERFACE_VERSION,
            "cross_arc_conflict_gate_version":
                STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
            "cross_arc_conflict_repair_version":
                STATEFUL_OUTLINE_CROSS_ARC_REPAIR_VERSION,
            "schema_version": STATEFUL_OUTLINE_SCHEMA_VERSION,
            "completion_contract_version":
                STATEFUL_COMPLETION_CONTRACT_VERSION,
            "schema_sha256": canonical_sha256(RAW_STATEFUL_OUTLINE_SCHEMA),
            "challenger_model": api.model,
        }
        atomic_json(planner_root / "config/run_manifest.json", planner_manifest)
        ledger = RequestLedger(
            planner_root, sha256_file(planner_root / "config/run_manifest.json"),
            "live")
        budget = ApiBudget(
            normal_caps={"stateful_outline_overview": 1,
                         "stateful_outline_arc": 4,
                         "stateful_outline_cross_arc_repair": 4},
            format_retry_cap=9, physical_cap=18)
        view = _planner_view(parent)
        overview_task = run_single_object_task(
            run_root=planner_root, ledger=ledger,
            backend=LiveSemanticRepairBackend(
                client, OUTLINE_OVERVIEW_MAX_TOKENS), budget=budget,
            stage="stateful_outline_overview",
            logical_request_id="stateful-outline-overview",
            semantic_prompt=overview_prompt, lineage_view=view,
            validator=lambda raw: validate_stateful_outline_overview(
                raw, parent_turns=parent),
            schema_retry_decisions=(
                "stateful_outline_overview_schema_invalid",),
            retry_prompt_suffix_factory=lambda raw, validation:
                build_stateful_outline_overview_interface_repair_suffix(
                    previous_overview=raw, validation=validation,
                    parent_turns=parent)
                if (validation.get("decision") ==
                    "stateful_outline_overview_schema_invalid" and
                    isinstance(raw, Mapping)) else None)
        atomic_json(planner_root / "overview_task.json", {
            key: value for key, value in overview_task.items()
            if key not in {"accepted_raw", "accepted_validation"}})
        if not overview_task["accepted"]:
            planner_status = "overview_failed"
            return _finalize(
                run_root=run_root, input_path=unified_root_path,
                input_sha=input_sha, parent=parent, parent_hash=parent_hash,
                tokenizer=tokenizer, manifest=manifest,
                planner_status=planner_status, graph_validation=graph_validation,
                failure="planner_overview_invalid_after_retry")
        overview = overview_task["accepted_validation"]["overview"]
        atomic_json(planner_root / "stateful_outline_overview.json", overview)
        arcs = []
        tasks = [overview_task]
        for arc_index in range(1, 5):
            assigned = _arc_scene_indices(arc_index)
            prompt = build_stateful_outline_arc_prompt(
                parent_turns=parent, overview=overview, arc_index=arc_index,
                assigned_scene_indices=assigned, prior_arcs=arcs)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            atomic_json(planner_root / f"arc_E{arc_index}_preflight.json", {
                "prompt_tokens": prompt_tokens,
                "reserved_completion_tokens": OUTLINE_ARC_MAX_TOKENS,
                "context_limit": CHALLENGER_CONTEXT_LIMIT,
                "decision": "pass" if prompt_tokens + OUTLINE_ARC_MAX_TOKENS <=
                CHALLENGER_CONTEXT_LIMIT else "reject_context_overflow"})
            if prompt_tokens + OUTLINE_ARC_MAX_TOKENS > CHALLENGER_CONTEXT_LIMIT:
                planner_status = f"arc_E{arc_index}_context_overflow"
                return _finalize(
                    run_root=run_root, input_path=unified_root_path,
                    input_sha=input_sha, parent=parent, parent_hash=parent_hash,
                    tokenizer=tokenizer, manifest=manifest,
                    planner_status=planner_status,
                    graph_validation=graph_validation,
                    failure=planner_status)
            arc_validator = StatefulOutlineArcRetryValidator(
                parent_turns=parent, overview=overview,
                assigned_scene_indices=assigned, arc_index=arc_index)
            task = run_single_object_task(
                run_root=planner_root, ledger=ledger,
                backend=LiveSemanticRepairBackend(client, OUTLINE_ARC_MAX_TOKENS),
                budget=budget, stage="stateful_outline_arc",
                logical_request_id=f"stateful-outline-E{arc_index}",
                semantic_prompt=prompt, lineage_view=view,
                validator=arc_validator,
                schema_retry_decisions=("stateful_outline_arc_schema_invalid",),
                retry_prompt_suffix_factory=(
                    lambda raw, validation, arc_index=arc_index:
                    build_stateful_outline_e1_repair_suffix(
                        previous_arc=raw, validation=validation,
                        arc_id=f"E{arc_index}")
                    if isinstance(raw, Mapping) else None))
            tasks.append(task)
            atomic_json(planner_root / f"arc_E{arc_index}_task.json", {
                key: value for key, value in task.items()
                if key not in {"accepted_raw", "accepted_validation"}})
            if not task["accepted"]:
                planner_status = f"arc_E{arc_index}_failed"
                return _finalize(
                    run_root=run_root, input_path=unified_root_path,
                    input_sha=input_sha, parent=parent, parent_hash=parent_hash,
                    tokenizer=tokenizer, manifest=manifest,
                    planner_status=planner_status,
                    graph_validation=graph_validation,
                    failure=f"planner_arc_E{arc_index}_invalid_after_retry")
            arcs.append(dict(task["accepted_validation"]["arc"]))
        initial_conflict_gate = cross_arc_stage_conflict_gate(arcs)
        atomic_json(planner_root / "cross_arc_stage_conflict_gate_initial.json",
                    initial_conflict_gate)
        repaired_arc_ids: list[str] = []
        repair_tasks: list[dict[str, Any]] = []
        conflict_gate = initial_conflict_gate
        while not conflict_gate["valid"]:
            remaining = [arc_id for arc_id in
                         conflict_gate.get("conflicting_arc_ids") or []
                         if arc_id not in repaired_arc_ids]
            if not remaining:
                planner_status = "cross_arc_conflict_unresolved"
                atomic_json(
                    planner_root / "cross_arc_stage_conflict_gate_final.json",
                    conflict_gate)
                return _finalize(
                    run_root=run_root, input_path=unified_root_path,
                    input_sha=input_sha, parent=parent,
                    parent_hash=parent_hash, tokenizer=tokenizer,
                    manifest=manifest, planner_status=planner_status,
                    graph_validation={"valid": False,
                                      "cross_arc_conflict_gate": conflict_gate},
                    failure="planner_cross_arc_conflict_unresolved")
            arc_id = remaining[0]
            arc_index = int(arc_id[1:])
            repaired_arc_ids.append(arc_id)
            assigned = _arc_scene_indices(arc_index)
            base_prompt = build_stateful_outline_arc_prompt(
                parent_turns=parent, overview=overview,
                arc_index=arc_index, assigned_scene_indices=assigned,
                prior_arcs=arcs[:arc_index - 1])
            repair_prompt = base_prompt + "\n\n" + (
                build_cross_arc_stage_conflict_repair_suffix(
                    arcs=arcs, conflict_gate=conflict_gate, arc_id=arc_id))
            repair_prompt_tokens = len(tokenizer.encode(
                repair_prompt, add_special_tokens=False))
            atomic_json(
                planner_root / f"cross_arc_repair_{arc_id}_preflight.json", {
                    "prompt_tokens": repair_prompt_tokens,
                    "reserved_completion_tokens": OUTLINE_ARC_MAX_TOKENS,
                    "context_limit": CHALLENGER_CONTEXT_LIMIT,
                    "decision": "pass" if repair_prompt_tokens +
                    OUTLINE_ARC_MAX_TOKENS <= CHALLENGER_CONTEXT_LIMIT else
                    "reject_context_overflow",
                })
            if (repair_prompt_tokens + OUTLINE_ARC_MAX_TOKENS >
                    CHALLENGER_CONTEXT_LIMIT):
                planner_status = f"cross_arc_repair_{arc_id}_context_overflow"
                return _finalize(
                    run_root=run_root, input_path=unified_root_path,
                    input_sha=input_sha, parent=parent,
                    parent_hash=parent_hash, tokenizer=tokenizer,
                    manifest=manifest, planner_status=planner_status,
                    graph_validation={"valid": False,
                                      "cross_arc_conflict_gate": conflict_gate},
                    failure=planner_status)
            repair_validator = CrossArcStageConflictRepairValidator(
                parent_turns=parent, overview=overview,
                assigned_scene_indices=assigned, arcs=arcs,
                conflict_gate=conflict_gate, arc_id=arc_id)
            repair_task = run_single_object_task(
                run_root=planner_root, ledger=ledger,
                backend=LiveSemanticRepairBackend(
                    client, OUTLINE_ARC_MAX_TOKENS), budget=budget,
                stage="stateful_outline_cross_arc_repair",
                logical_request_id=f"stateful-outline-{arc_id}-cross-arc-repair",
                semantic_prompt=repair_prompt, lineage_view=view,
                validator=repair_validator,
                schema_retry_decisions=("stateful_outline_arc_schema_invalid",))
            repair_tasks.append(repair_task)
            tasks.append(repair_task)
            atomic_json(planner_root / f"cross_arc_repair_{arc_id}_task.json", {
                key: value for key, value in repair_task.items()
                if key not in {"accepted_raw", "accepted_validation"}})
            if not repair_task["accepted"]:
                planner_status = f"cross_arc_repair_{arc_id}_failed"
                return _finalize(
                    run_root=run_root, input_path=unified_root_path,
                    input_sha=input_sha, parent=parent,
                    parent_hash=parent_hash, tokenizer=tokenizer,
                    manifest=manifest, planner_status=planner_status,
                    graph_validation={"valid": False,
                                      "cross_arc_conflict_gate": conflict_gate},
                    failure=f"planner_cross_arc_repair_{arc_id}_failed")
            arcs[arc_index - 1] = dict(
                repair_task["accepted_validation"]["arc"])
            conflict_gate = cross_arc_stage_conflict_gate(arcs)
        atomic_json(planner_root / "cross_arc_stage_conflict_gate_final.json",
                    conflict_gate)
        atomic_json(planner_root / "cross_arc_stage_conflict_repairs.json", {
            "version": STATEFUL_OUTLINE_CROSS_ARC_REPAIR_VERSION,
            "initial_gate": initial_conflict_gate,
            "repaired_arc_ids": repaired_arc_ids,
            "repair_task_count": len(repair_tasks),
            "final_gate": conflict_gate,
        })
        raw_outline = {**dict(overview), "event_arcs": arcs}
        assembled = validate_raw_stateful_outline(
            raw_outline, parent_turns=parent)
        if assembled["decision"] != "stateful_outline_accept":
            planner_status = "assembly_failed"
            return _finalize(
                run_root=run_root, input_path=unified_root_path,
                input_sha=input_sha, parent=parent, parent_hash=parent_hash,
                tokenizer=tokenizer, manifest=manifest,
                planner_status=planner_status,
                graph_validation={"valid": False,
                                  "errors": assembled["reasons"]},
                failure="planner_outline_assembly_invalid")
        outline = normalize_stateful_outline(
            raw_outline, parent_turns=parent, root_id=root_id,
            sample_id=parent_object["sample_id"], dialogue_hash=parent_hash,
            parent_session_count=5,
            outline_id_prefix=f"{root_id}:outline", round_index=1)
        outline_errors = validate_stateful_outline(outline)
        contracts = compile_stateful_scene_contracts(outline)
        initial_ledger = initialize_stateful_ledger(contracts)
        continuity_errors = []
        if not all(contract.get("parent_anchor_evidence")
                   for contract in contracts):
            continuity_errors.append("scene_missing_parent_continuity")
        graph_validation = {
            "valid": not outline_errors and not continuity_errors and
            len(contracts) == 10 and
            len(initial_ledger["stage_states"]) == 16,
            "outline_errors": outline_errors,
            "continuity_errors": continuity_errors,
            "cross_arc_conflict_gate_initial": initial_conflict_gate,
            "cross_arc_conflict_gate_final": conflict_gate,
            "cross_arc_repaired_arc_ids": repaired_arc_ids,
            "scene_contract_count": len(contracts),
            "initial_stage_count": len(initial_ledger["stage_states"]),
            "event_arc_count": len(outline.get("planned_event_arcs") or []),
            "scene_stage_assignment": [{
                "scene_id": contract["scene_id"],
                "stage_ids": [stage["stage_id"] for stage in
                              contract["assigned_event_stages"]],
            } for contract in contracts],
            "evidence_requirement_count": sum(
                len(stage.get("completion_evidence_phrases") or [])
                for contract in contracts
                for stage in contract["assigned_event_stages"]),
            "planned_handoff_ids": sorted({
                str(handoff["handoff_id"])
                for contract in contracts
                for stage in contract["assigned_event_stages"]
                for handoff in stage.get("outcome_handoffs") or []
            }),
            "parent_anchor_scene_coverage": sum(bool(
                contract.get("parent_anchor_evidence"))
                for contract in contracts) / len(contracts),
        }
        atomic_json(planner_root / "raw_stateful_outline.json", raw_outline)
        atomic_json(planner_root / "stateful_outline.json", outline)
        atomic_json(planner_root / "stateful_scene_contracts.json", {
            "contract_version": STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
            "contracts": contracts,
            "contracts_sha256": canonical_sha256(contracts)})
        atomic_json(planner_root / "graph_validation.json", graph_validation)
        if not graph_validation["valid"]:
            planner_status = "graph_validation_failed"
            return _finalize(
                run_root=run_root, input_path=unified_root_path,
                input_sha=input_sha, parent=parent, parent_hash=parent_hash,
                tokenizer=tokenizer, manifest=manifest,
                planner_status=planner_status,
                graph_validation=graph_validation,
                failure="initial_or_proposed_graph_invalid")
        planner_status = "complete"
        realizer_root = run_root / "realizer"
        realizer_manifest = {
            "run_id": f"{run_id}:realizer", "root_id": root_id,
            "lineage_id": lineage_id, "mode": "live_stateful_graph_expansion",
            "project_git_head": manifest["project_git_head"],
            "project_diff_sha256": diff_sha,
            "source_parent_sha256": canonical_sha256(parent),
            "outline_sha256": canonical_sha256(outline),
            "scene_contracts_sha256": canonical_sha256(contracts),
            "graph_expansion_version": STATEFUL_GRAPH_EXPANSION_VERSION,
            "graph_scene_contract_version": STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
            "state_ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
            "realizer_prompt_version": STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
            "realizer_schema_sha256": canonical_sha256(
                STATEFUL_REALIZER_JSON_SCHEMA),
            "terminal_marker_normalization_version":
                STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
            "multi_stage_retry_version":
                STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
            "handoff_whitelist_version": STATEFUL_HANDOFF_WHITELIST_VERSION,
            "realizer_prompt_template_sha256": sha256_bytes(
                build_stateful_graph_realizer_prompt(
                    contract=contracts[0], recent_sessions=[],
                    active_handoffs=[]).encode()),
            "challenger_model": api.model,
            "questioner_calls": 0, "solver_calls": 0,
            "verifier_calls": 0, "training_runs": 0,
        }
        atomic_json(realizer_root / "config/run_manifest.json", realizer_manifest)
        expansion = StatefulGraphRoundOrchestrator(
            run_root=realizer_root, manifest=realizer_manifest,
            contracts=contracts, tokenizer=tokenizer,
            realizer=LiveRealizerBackend(client), backend_mode="live",
            max_attempts_per_scene=3, max_realizer_generations=30).run()
        if len(expansion["sessions"]) != 10 or not expansion[
                "completion"]["complete"]:
            return _finalize(
                run_root=run_root, input_path=unified_root_path,
                input_sha=input_sha, parent=parent, parent_hash=parent_hash,
                tokenizer=tokenizer, manifest=manifest,
                planner_status=planner_status,
                graph_validation=graph_validation,
                failure="realizer_or_stateful_completion_incomplete")
        return _finalize(
            run_root=run_root, input_path=unified_root_path,
            input_sha=input_sha, parent=parent, parent_hash=parent_hash,
            tokenizer=tokenizer, manifest=manifest,
            planner_status=planner_status, graph_validation=graph_validation,
            failure=None)
    except Exception as error:
        atomic_json(run_root / "reports/fatal_error.json", {
            "error_type": type(error).__name__, "message": str(error)})
        return _finalize(
            run_root=run_root, input_path=unified_root_path,
            input_sha=input_sha, parent=parent, parent_hash=parent_hash,
            tokenizer=tokenizer, manifest=manifest,
            planner_status=planner_status, graph_validation=graph_validation,
            failure=f"{type(error).__name__}:{error}")
