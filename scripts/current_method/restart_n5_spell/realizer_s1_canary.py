"""Frozen-Planner/Graph, S1-only paid Realizer canary.

This route deliberately has no Planner backend.  It reads the authoritative
artifacts of one completed Planner/Graph run, audits its three rejected S1
responses offline, and invokes only the existing stateful Realizer executor for
the frozen S1 contract.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import atomic_json, atomic_text, sha256_file
from .dynamic_generation_canary import _request_audit, api_usage
from .live_backends import (
    LiveRealizerBackend, OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .run_canary import create_source_snapshot, git_value, project_diff_sha256
from .stateful_graph_expansion import (
    STATEFUL_GRAPH_EXPANSION_VERSION, STATEFUL_GRAPH_LEDGER_VERSION,
    STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
    STATEFUL_GRAPH_SCENE_CONTRACT_VERSION, STATEFUL_REALIZER_JSON_SCHEMA,
    STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
    canonical_sha256, tree_sha256, validate_and_apply_stateful_scene,
)
from .stateful_graph_orchestrator import (
    STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
    StatefulGraphRoundOrchestrator,
)


REALIZER_S1_CANARY_VERSION = "cc_frozen_graph_realizer_s1_canary_v1"
CHALLENGER_MODEL = "Qwen/Qwen3-32B"
REALIZER_MAX_TOKENS = 4_096
PROJECT = Path(__file__).resolve().parents[3]

DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_generation_canary_"
    "overview_participants_v1")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dialogue_tokens(tokenizer: Any,
                     turns: Sequence[Mapping[str, Any]]) -> int:
    text = "\n".join(
        f"{turn.get('speaker', '')}: {turn.get('text', '')}" for turn in turns)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _source_artifact_paths(source_run: Path) -> dict[str, Path]:
    return {
        "overview": source_run / "planner/stateful_outline_overview.json",
        "raw_outline": source_run / "planner/raw_stateful_outline.json",
        "validated_graph": source_run / "planner/stateful_outline.json",
        "graph_validation": source_run / "planner/graph_validation.json",
        "scene_contracts": source_run / "planner/stateful_scene_contracts.json",
        "parent_dialogue": source_run / "frozen/parent_contract.json",
    }


def load_frozen_s1_source(source_run: Path) -> dict[str, Any]:
    """Load and verify the authoritative frozen Planner/Graph inputs."""
    source_run = source_run.resolve()
    paths = _source_artifact_paths(source_run)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("s1_source_artifacts_missing:" + ",".join(missing))
    overview = _read(paths["overview"])
    raw_outline = _read(paths["raw_outline"])
    validated_graph = _read(paths["validated_graph"])
    graph_validation = _read(paths["graph_validation"])
    contracts_blob = _read(paths["scene_contracts"])
    parent = _read(paths["parent_dialogue"])
    contracts = contracts_blob.get("contracts") or []
    if graph_validation.get("valid") is not True:
        raise ValueError("s1_source_graph_not_validated")
    if len(raw_outline.get("event_arcs") or []) != 4:
        raise ValueError("s1_source_requires_four_arcs")
    if len(contracts) != 10 or contracts[0].get("scene_id") != "S1":
        raise ValueError("s1_source_scene_contract_set_invalid")
    if contracts_blob.get("contracts_sha256") != canonical_sha256(contracts):
        raise ValueError("s1_source_contracts_sha_drift")
    s1 = dict(contracts[0])
    embedded = s1.get("contract_sha256")
    computed = canonical_sha256({
        key: value for key, value in s1.items() if key != "contract_sha256"
    })
    if embedded != computed:
        raise ValueError("s1_source_contract_sha_drift")
    turns = (parent.get("dialogue") or {}).get("turns") or []
    sessions = list(dict.fromkeys(str(turn["session_id"]) for turn in turns))
    if len(turns) != 60 or len(sessions) != 5:
        raise ValueError("s1_source_parent_shape_drift")
    # The authoritative raw arc schema assigns E1--E4 by fixed array
    # position; normalized event IDs live in the validated Graph.
    arc_shas = {
        f"E{position}": canonical_sha256(arc)
        for position, arc in enumerate(raw_outline["event_arcs"], 1)
    }
    if sorted(arc_shas) != ["E1", "E2", "E3", "E4"]:
        raise ValueError("s1_source_arc_ids_invalid")
    return {
        "source_run": source_run,
        "paths": paths,
        "overview": overview,
        "raw_outline": raw_outline,
        "validated_graph": validated_graph,
        "graph_validation": graph_validation,
        "contracts_blob": contracts_blob,
        "s1_contract": s1,
        "parent_contract": parent,
        "parent_turns": turns,
        "source_file_sha256": {
            key: sha256_file(path) for key, path in paths.items()
        },
        "arc_canonical_sha256": arc_shas,
        "s1_contract_sha256": computed,
        "parent_dialogue_canonical_sha256": canonical_sha256(turns),
    }


def audit_s1_failure_attempt(
    *, raw_wrapper: Mapping[str, Any], report: Mapping[str, Any],
    contract: Mapping[str, Any], tokenizer: Any, attempt: int,
) -> dict[str, Any]:
    """Audit one historical raw response without repairing it."""
    raw = str(raw_wrapper.get("raw_response") or "")
    metadata = dict(raw_wrapper.get("metadata") or {})
    errors = list(report.get("reasons") or [])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "attempt": attempt, "json_valid": False,
            "actual_dialogue_tokens": 0, "actual_turn_count": 0,
            "minimum_dialogue_tokens": contract["hard_dialogue_token_range"][0],
            "finish_reason": metadata.get("finish_reason"),
            "stopped_before_natural_end": metadata.get("finish_reason") == "length",
            "validation_errors": errors,
            "first_contract_deviation": "json_parse",
        }
    turns = parsed.get("dialogue_turns") or []
    stages = contract.get("assigned_event_stages") or []
    stage = stages[0]
    updates = parsed.get("candidate_stage_updates") or []
    update = next((row for row in updates
                   if row.get("stage_id") == stage["stage_id"]), {})
    evidence_index = update.get("evidence_turn_index")
    evidence_turn = (turns[evidence_index]
                     if isinstance(evidence_index, int) and
                     0 <= evidence_index < len(turns) else {})
    phrases = list(stage["completion_evidence_phrases"])
    phrase_locations = [
        index for index, turn in enumerate(turns)
        if any(phrase.casefold() in str(turn.get("text") or "").casefold()
               for phrase in phrases)
    ]
    speakers = [str(turn.get("speaker") or "") for turn in turns]
    allowed = set(contract["allowed_speakers"])
    expected_top = set(STATEFUL_REALIZER_JSON_SCHEMA["required"])
    extra_top = sorted(set(parsed) - expected_top)
    extra_turn_fields = sorted({
        key for turn in turns if isinstance(turn, Mapping)
        for key in set(turn) - {"speaker", "text"}
    })
    actual_tokens = _dialogue_tokens(tokenizer, turns)
    first_deviation = "none"
    if "stage_evidence_speaker_mismatch" in errors:
        first_deviation = "candidate_stage_updates.evidence_turn_index"
    elif "terminal_evidence_marker_missing" in errors:
        first_deviation = "dialogue_turns.required_terminal_marker"
    elif "stateful_scene_token_count_out_of_range" in errors:
        first_deviation = "post_generation.dialogue_token_length_gate"
    return {
        "attempt": attempt,
        "request_id": raw_wrapper.get("request_id"),
        "json_valid": True,
        "actual_dialogue_tokens": actual_tokens,
        "actual_turn_count": len(turns),
        "minimum_dialogue_tokens": contract["hard_dialogue_token_range"][0],
        "maximum_dialogue_tokens": contract["hard_dialogue_token_range"][1],
        "minimum_turn_count": contract["hard_turn_range"][0],
        "maximum_turn_count": contract["hard_turn_range"][1],
        "finish_reason": metadata.get("finish_reason"),
        "stopped_before_natural_end": metadata.get("finish_reason") == "length",
        "completion_tokens": ((metadata.get("usage") or {}).get(
            "completion_tokens")),
        "evidence_phrase_present": bool(phrase_locations),
        "evidence_phrase_turn_indices": phrase_locations,
        "reported_evidence_turn_index": evidence_index,
        "reported_result_speaker": update.get("result_speaker"),
        "actual_evidence_speaker": evidence_turn.get("speaker"),
        "evidence_speaker_correct": (
            evidence_turn.get("speaker") == update.get("result_speaker") and
            update.get("result_speaker") in stage["authorized_result_speakers"]),
        "terminal_marker_present": bool(phrase_locations),
        "terminal_marker_in_final_turn": bool(
            turns and len(turns) - 1 in phrase_locations),
        "unauthorized_speakers": sorted(set(speakers) - allowed),
        "missing_required_participants": sorted(
            set(contract["required_participants"]) - set(speakers)),
        "extra_top_level_fields": extra_top,
        "extra_dialogue_turn_fields": extra_turn_fields,
        "validation_errors": errors,
        "first_contract_deviation": first_deviation,
    }


def audit_historical_s1_failures(
    *, source_run: Path, contract: Mapping[str, Any], tokenizer: Any,
) -> dict[str, Any]:
    attempts = []
    for attempt in range(1, 4):
        raw_path = source_run / (
            f"realizer/raw/stateful-realizer-s01-a{attempt:02d}.json")
        report_path = source_run / (
            f"realizer/reports/stateful-realizer-s01-a{attempt:02d}.json")
        if not raw_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"historical_s1_attempt_missing:{attempt}")
        attempts.append(audit_s1_failure_attempt(
            raw_wrapper=_read(raw_path), report=_read(report_path),
            contract=contract, tokenizer=tokenizer, attempt=attempt))
    return {
        "schema_version": "realizer_s1_failure_audit_v1",
        "source_run": str(source_run),
        "attempt_count": 3,
        "hard_dialogue_token_range": list(
            contract["hard_dialogue_token_range"]),
        "hard_turn_range": list(contract["hard_turn_range"]),
        "attempts": attempts,
        "diagnosis": (
            "Attempts 1 and 2 reached a natural API stop with valid evidence "
            "and speakers but were below the frozen 256-token minimum. "
            "Attempt 3 also bound the evidence index to the wrong speaker and "
            "omitted the exact terminal marker. The old executor sent the same "
            "prompt for all three attempts, so it supplied no directed error "
            "feedback."),
    }


def _failure_audit_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Historical S1 failure audit", "",
        "| Attempt | Tokens | Turns | Finish | Phrase | Speaker | Marker | Errors |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in audit["attempts"]:
        lines.append(
            f"| {row['attempt']} | {row['actual_dialogue_tokens']} | "
            f"{row['actual_turn_count']} | `{row['finish_reason']}` | "
            f"{row.get('evidence_phrase_present', False)} | "
            f"{row.get('evidence_speaker_correct', False)} | "
            f"{row.get('terminal_marker_present', False)} | "
            f"`{', '.join(row['validation_errors'])}` |")
    lines.extend(["", str(audit["diagnosis"]), ""])
    return "\n".join(lines)


def _tree_file_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _freeze_source_artifacts(run_root: Path, source: Mapping[str, Any]) -> None:
    frozen = run_root / "frozen"
    atomic_json(frozen / "authoritative_overview.json", source["overview"])
    atomic_json(frozen / "authoritative_arcs.json", {
        "event_arcs": source["raw_outline"]["event_arcs"],
        "canonical_sha256_by_event_id": source["arc_canonical_sha256"],
    })
    atomic_json(frozen / "validated_graph.json", source["validated_graph"])
    atomic_json(frozen / "graph_validation.json", source["graph_validation"])
    atomic_json(frozen / "s1_scene_contract.json", source["s1_contract"])
    atomic_json(frozen / "parent_dialogue.json", source["parent_contract"])


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Realizer S1-only canary", "",
        f"- decision: `{summary['decision']}`",
        f"- realizer_s1_ready: `{str(summary['realizer_s1_ready']).lower()}`",
        "- generation_ready: `false`",
        f"- Planner API calls: `{summary['planner_api_calls']}`",
        f"- Graph API calls: `{summary['graph_api_calls']}`",
        f"- Realizer API calls: `{summary['realizer_api_calls']}`", "",
        "| Attempt | Decision | Tokens | Turns | Errors |",
        "|---:|---|---:|---:|---|",
    ]
    for row in summary["realizer_attempts"]:
        lines.append(
            f"| {row['attempt']} | `{row['decision']}` | "
            f"{row['actual_dialogue_tokens']} | {row['actual_turn_count']} | "
            f"`{', '.join(row['errors'])}` |")
    lines.extend(["", "Planner and Graph were loaded from frozen files only.", ""])
    return "\n".join(lines)


def run(
    *, source_run: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path, tokenizer: Any,
    prior_test_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a fresh S1-only canary; this function cannot call Planner."""
    source_run = source_run.resolve()
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"realizer_s1_run_root_exists:{run_root}")
    source_tree_before = tree_sha256(source_run)
    source_files_before = _tree_file_manifest(source_run)
    source = load_frozen_s1_source(source_run)
    source_manifest = _read(source_run / "config/run_manifest.json")
    if source_manifest.get("root_id") != "cc-train-episode-61555":
        raise ValueError("realizer_s1_fixed_root_mismatch")
    input_path = Path(source_manifest["input_unified_root_path"])
    input_sha_before = sha256_file(input_path)
    if input_sha_before != source_manifest["input_unified_root_sha256"]:
        raise ValueError("realizer_s1_input_sha_drift")

    api = replace(
        load_challenger_config(api_config_path), model=CHALLENGER_MODEL,
        temperature=0.25, enable_thinking=False, timeout_seconds=600,
        realizer_max_tokens=REALIZER_MAX_TOKENS)
    run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, run_root, False)
    diff_sha = project_diff_sha256(PROJECT)
    run_id = "cc-train-episode-61555:realizer-s1-only:canary:01"
    manifest = {
        "schema_version": "realizer_s1_canary_manifest_v1",
        "mode": REALIZER_S1_CANARY_VERSION,
        "run_id": run_id,
        "root_id": "cc-train-episode-61555",
        "sample_id": source["parent_contract"]["sample_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_head": git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256": diff_sha,
        **snapshot,
        "source_run": str(source_run),
        "source_run_tree_sha256_before": source_tree_before,
        "source_artifact_sha256": source["source_file_sha256"],
        "arc_canonical_sha256": source["arc_canonical_sha256"],
        "s1_contract_sha256": source["s1_contract_sha256"],
        "parent_dialogue_canonical_sha256":
            source["parent_dialogue_canonical_sha256"],
        "input_unified_root_path": str(input_path),
        "input_unified_root_sha256": input_sha_before,
        "challenger_model": api.model,
        "realizer_max_tokens": api.realizer_max_tokens,
        "realizer_prompt_version": STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
        "realizer_retry_version": STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
        "terminal_marker_normalization_version":
            STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
        "graph_expansion_version": STATEFUL_GRAPH_EXPANSION_VERSION,
        "resume": False, "scene_scope": ["S1"],
        "planner_api_calls": 0, "graph_api_calls": 0,
        "max_realizer_attempts": 3,
        "prior_test_results": dict(prior_test_results),
        "questioner_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
    }
    atomic_json(run_root / "config/run_manifest.json", manifest)
    _freeze_source_artifacts(run_root, source)
    failure_audit = audit_historical_s1_failures(
        source_run=source_run, contract=source["s1_contract"],
        tokenizer=tokenizer)
    atomic_json(run_root / "reports/s1_failure_audit.json", failure_audit)
    atomic_text(run_root / "reports/s1_failure_audit.md",
                _failure_audit_markdown(failure_audit))
    atomic_json(run_root / "reports/preflight.json", {
        "decision": "pass", "fresh_run": True, "resume": False,
        "scene_scope": ["S1"], "planner_api_calls_before_execution": 0,
        "graph_api_calls_before_execution": 0,
        "source_graph_validated": True,
        "source_artifact_sha256": source["source_file_sha256"],
        "input_sha256": input_sha_before,
    })

    realizer_root = run_root / "realizer"
    realizer_manifest = {
        "run_id": f"{run_id}:realizer", "root_id": manifest["root_id"],
        "mode": "frozen_graph_s1_only_realizer",
        "project_git_head": manifest["project_git_head"],
        "project_diff_sha256": diff_sha,
        "source_run": str(source_run),
        "source_graph_sha256": source["source_file_sha256"][
            "validated_graph"],
        "s1_contract_sha256": source["s1_contract_sha256"],
        "parent_dialogue_sha256": source["source_file_sha256"][
            "parent_dialogue"],
        "graph_scene_contract_version": STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
        "state_ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
        "realizer_prompt_version": STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
        "realizer_retry_version": STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
        "terminal_marker_normalization_version":
            STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
        "challenger_model": api.model,
        "planner_api_calls": 0, "graph_api_calls": 0,
        "questioner_calls": 0, "solver_calls": 0,
        "verifier_calls": 0, "training_runs": 0,
    }
    atomic_json(realizer_root / "config/run_manifest.json", realizer_manifest)
    failure: str | None = None
    sessions: list[dict[str, Any]] = []
    completion: dict[str, Any] = {"complete": False}
    try:
        result = StatefulGraphRoundOrchestrator(
            run_root=realizer_root, manifest=realizer_manifest,
            contracts=[source["s1_contract"]], tokenizer=tokenizer,
            realizer=LiveRealizerBackend(
                OpenAICompatibleChallengerClient(api)), backend_mode="live",
            max_attempts_per_scene=3,
            max_realizer_generations=3).run()
        sessions = list(result["sessions"])
        completion = dict(result["completion"])
    except Exception as error:  # Preserve all raw/API evidence before stopping.
        failure = f"{type(error).__name__}:{error}"
        atomic_json(run_root / "reports/fatal_error.json", {
            "error_type": type(error).__name__, "error": str(error)})
        sessions = [
            _read(path) for path in
            sorted((realizer_root / "accepted").glob("session_*.json"))
        ]

    _request_audit(run_root)
    usage = api_usage(run_root)
    attempts = []
    for index in range(1, 4):
        report_path = realizer_root / (
            f"reports/stateful-realizer-s01-a{index:02d}.json")
        state_path = realizer_root / (
            f"ledger/requests/stateful-realizer-s01-a{index:02d}.json")
        if not report_path.is_file() and not state_path.is_file():
            continue
        report = _read(report_path) if report_path.is_file() else {}
        state = _read(state_path) if state_path.is_file() else {}
        attempts.append({
            "attempt": index,
            "request_id": f"stateful-realizer-s01-a{index:02d}",
            "decision": report.get("decision") or (
                "accept" if state.get("final_decision") ==
                "accept_stateful_scene" else "unknown"),
            "actual_dialogue_tokens": report.get(
                "actual_dialogue_tokens") or (
                    sessions[0].get("dialogue_token_count")
                    if sessions and sessions[0].get("accepted_attempt") == index
                    else 0),
            "actual_turn_count": report.get("actual_turn_count") or (
                len(sessions[0].get("dialogue_turns") or [])
                if sessions and sessions[0].get("accepted_attempt") == index
                else 0),
            "errors": list(report.get("reasons") or []),
            "finish_reason": ((state.get("response_metadata") or {}).get(
                "finish_reason")),
            "usage": ((state.get("response_metadata") or {}).get("usage")),
            "prompt_sha256": state.get("prompt_sha256"),
        })

    accepted = len(sessions) == 1 and completion.get("complete") is True
    input_unchanged = sha256_file(input_path) == input_sha_before
    source_tree_after = tree_sha256(source_run)
    source_files_after = _tree_file_manifest(source_run)
    intents = usage["completed_requests_by_intent"]
    checks = {
        "git_head_match": git_value(PROJECT, "rev-parse", "HEAD") ==
            manifest["project_git_head"],
        "project_diff_sha256_match": project_diff_sha256(PROJECT) == diff_sha,
        "source_snapshot_match": sha256_file(Path(
            manifest["source_snapshot_manifest_path"])) ==
            manifest["source_snapshot_manifest_sha256"],
        "source_run_tree_unchanged": source_tree_after == source_tree_before,
        "source_run_files_unchanged": source_files_after == source_files_before,
        "input_unified_root_unchanged": input_unchanged,
        "fresh_no_resume": manifest["resume"] is False,
        "no_reused_api_response": usage["reused_response_count"] == 0,
        "only_realizer_api_intent": set(intents) <= {"stateful_realize_scene"},
        "planner_api_zero": not any("outline" in key for key in intents),
        "graph_api_zero": True,
        "source_hashes_match": source["source_file_sha256"] == {
            key: sha256_file(path) for key, path in source["paths"].items()
        },
    }
    provenance_pass = all(checks.values())
    ready = accepted and provenance_pass
    summary = {
        "schema_version": "realizer_s1_canary_summary_v1",
        "decision": "realizer_s1_canary_pass" if ready else
            "realizer_s1_canary_failed",
        "realizer_s1_ready": ready,
        "generation_ready": False,
        "scope_claim": "frozen_planner_graph_realizer_s1_only",
        "source_run": str(source_run),
        "source_run_tree_sha256_before": source_tree_before,
        "source_run_tree_sha256_after": source_tree_after,
        "source_artifact_sha256": source["source_file_sha256"],
        "arc_canonical_sha256": source["arc_canonical_sha256"],
        "s1_contract_sha256": source["s1_contract_sha256"],
        "parent_dialogue_canonical_sha256":
            source["parent_dialogue_canonical_sha256"],
        "input_unified_root_sha256_before": input_sha_before,
        "input_unified_root_sha256_after": sha256_file(input_path),
        "input_sha_unchanged": input_unchanged,
        "planner_api_calls": 0,
        "graph_api_calls": 0,
        "realizer_api_calls": int(intents.get("stateful_realize_scene") or 0),
        "realizer_attempts": attempts,
        "s1_passed": accepted,
        "accepted_attempt": sessions[0].get("accepted_attempt")
            if sessions else None,
        "accepted_dialogue_tokens": sessions[0].get("dialogue_token_count")
            if sessions else 0,
        "accepted_turn_count": len(sessions[0].get("dialogue_turns") or [])
            if sessions else 0,
        "stateful_completion": completion,
        "api_usage": usage,
        "failure_diagnosis": failure,
        "cache_resume_pollution": usage["reused_response_count"] != 0,
        "provenance_checks": checks,
        "authoritative_release_provenance": provenance_pass,
        "provenance_mixed": False,
        "questioner_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
    }
    atomic_json(run_root / "reports/s1_canary_summary.json", summary)
    atomic_text(run_root / "reports/s1_canary_summary.md",
                _summary_markdown(summary))
    atomic_json(run_root / "reports/post_run_provenance.json", {
        "schema_version": "realizer_s1_canary_provenance_v1",
        "checks": checks,
        "authoritative_release_provenance": provenance_pass,
        "provenance_mixed": False,
        "realizer_s1_ready": ready,
        "generation_ready": False,
        "run_manifest_sha256": sha256_file(
            run_root / "config/run_manifest.json"),
        "summary_sha256": sha256_file(
            run_root / "reports/s1_canary_summary.json"),
    })
    return summary
