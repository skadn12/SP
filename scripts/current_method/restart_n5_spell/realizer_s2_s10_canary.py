"""Sequential S2--S10 Realizer canary seeded by the accepted frozen S1."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import atomic_json, atomic_text, sha256_file
from .dynamic_generation_canary import (
    _request_audit, _scene_results, api_usage, audit_child_generation,
)
from .live_backends import (
    LiveRealizerBackend, OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .realizer_s1_canary import (
    CHALLENGER_MODEL, DEFAULT_SOURCE_RUN, REALIZER_MAX_TOKENS,
    load_frozen_s1_source,
)
from .run_canary import create_source_snapshot, git_value, project_diff_sha256
from .stateful_graph_expansion import (
    STATEFUL_GRAPH_EXPANSION_VERSION, STATEFUL_GRAPH_LEDGER_VERSION,
    STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
    STATEFUL_GRAPH_SCENE_CONTRACT_VERSION, canonical_sha256,
    STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
    initialize_stateful_ledger, tree_sha256, validate_and_apply_stateful_scene,
    normalize_terminal_marker_text,
)
from .stateful_graph_orchestrator import (
    STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
    StatefulGraphRoundOrchestrator,
)


REALIZER_S2_S10_CANARY_VERSION = "cc_frozen_graph_realizer_s2_s10_canary_v3"
DEFAULT_S1_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_realizer_s1_only_canary_v1")
PROJECT = Path(__file__).resolve().parents[3]


def _dialogue_tokens(tokenizer: Any,
                     turns: Sequence[Mapping[str, Any]]) -> int:
    text = "\n".join(
        f"{turn.get('speaker', '')}: {turn.get('text', '')}" for turn in turns)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_accepted_s1(s1_run: Path) -> dict[str, Any]:
    """Verify that the source S1 was authoritative and accepted."""
    s1_run = s1_run.resolve()
    summary_path = s1_run / "reports/s1_canary_summary.json"
    session_path = s1_run / "realizer/accepted/session_01.json"
    audit_path = s1_run / "reports/s1_accepted_output_audit.json"
    for path in (summary_path, session_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"accepted_s1_source_missing:{path}")
    summary = _read(summary_path)
    audit = _read(audit_path)
    session = _read(session_path)
    if (summary.get("realizer_s1_ready") is not True or
            summary.get("generation_ready") is not False or
            summary.get("planner_api_calls") != 0 or
            summary.get("graph_api_calls") != 0 or
            audit.get("decision") != "pass"):
        raise ValueError("accepted_s1_source_not_authoritative")
    if (session.get("accepted_attempt") != 3 or
            session.get("graph_scene_id") != "S1"):
        raise ValueError("accepted_s1_session_identity_invalid")
    accepted_turns = [
        {"speaker": turn["speaker"], "text": turn["text"]}
        for turn in session.get("dialogue_turns") or []
    ]
    # Reconstruct only the original schema fields from the authoritative
    # accepted session.  This route intentionally does not read the source
    # Realizer raw response or its request ledger.
    model_payload = {
        "dialogue_turns": accepted_turns,
        "used_parent_anchor_ids": list(session["used_parent_anchor_ids"]),
        "advanced_event_arc_ids": list(session["advanced_event_arc_ids"]),
        "candidate_stage_updates": list(session["candidate_stage_updates"]),
        "candidate_handoff_updates": list(
            session["candidate_handoff_updates"]),
        "durable_assertion_candidates": list(
            session["durable_assertion_candidates"]),
        "local_scene_facts": list(session["local_scene_facts"]),
    }
    return {
        "run": s1_run,
        "summary": summary,
        "audit": audit,
        "session": session,
        "model_payload": model_payload,
        "sha256": {
            "summary": sha256_file(summary_path),
            "session": sha256_file(session_path),
            "accepted_output_audit": sha256_file(audit_path),
        },
    }


def _session_model_payload(session: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dialogue_turns": [
            {"speaker": turn["speaker"], "text": turn["text"]}
            for turn in session.get("dialogue_turns") or []],
        "used_parent_anchor_ids": list(session["used_parent_anchor_ids"]),
        "advanced_event_arc_ids": list(session["advanced_event_arc_ids"]),
        "candidate_stage_updates": list(session["candidate_stage_updates"]),
        "candidate_handoff_updates": list(
            session["candidate_handoff_updates"]),
        "durable_assertion_candidates": list(
            session["durable_assertion_candidates"]),
        "local_scene_facts": list(session["local_scene_facts"]),
    }


def load_accepted_scene_prefix(prefix_run: Path) -> dict[str, Any]:
    """Load accepted scene sessions and final ledger; never read API raws."""
    prefix_run = prefix_run.resolve()
    realizer_root = prefix_run / "realizer"
    session_paths = sorted((realizer_root / "accepted").glob(
        "session_*.json"))
    if not session_paths:
        raise FileNotFoundError("accepted_scene_prefix_missing")
    sessions = [_read(path) for path in session_paths]
    expected_scene_ids = [f"S{index}" for index in range(1, len(sessions) + 1)]
    if [str(row.get("graph_scene_id")) for row in sessions] != expected_scene_ids:
        raise ValueError("accepted_scene_prefix_not_contiguous")
    last_scene = expected_scene_ids[-1]
    ledger_path = realizer_root / f"frozen/state_ledger_after_{last_scene}.json"
    index_path = realizer_root / "accepted/index.json"
    manifest_path = realizer_root / "config/run_manifest.json"
    for path in (ledger_path, index_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"accepted_scene_prefix_artifact_missing:{path}")
    ledger = _read(ledger_path)
    index = _read(index_path)
    actual_session_sha = {
        path.name: sha256_file(path) for path in session_paths}
    if (index.get("run_manifest_sha256") != sha256_file(manifest_path) or
            index.get("session_sha256") != actual_session_sha):
        raise ValueError("accepted_scene_prefix_index_drift")
    if ledger.get("ledger_sha256") != sessions[-1].get(
            "state_ledger_sha256_after"):
        raise ValueError("accepted_scene_prefix_ledger_drift")
    return {
        "run": prefix_run,
        "sessions": sessions,
        "payloads": [_session_model_payload(row) for row in sessions],
        "ledger": ledger,
        "scene_count": len(sessions),
        "last_scene_id": last_scene,
        "sha256": {
            "sessions": actual_session_sha,
            "ledger": sha256_file(ledger_path),
            "accepted_index": sha256_file(index_path),
            "realizer_manifest": sha256_file(manifest_path),
        },
        "source_realizer_raw_read": False,
        "source_request_ledger_read": False,
        "source_state_ledger_read": True,
    }


def seed_frozen_scene_prefix(
    *, realizer_root: Path, realizer_manifest: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]], prefix: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate and freeze a contiguous accepted prefix in a fresh run."""
    count = int(prefix["scene_count"])
    if not 1 <= count < len(contracts):
        raise ValueError("accepted_scene_prefix_count_invalid")
    ledger = initialize_stateful_ledger(contracts)
    computed_ledgers = []
    for position, payload in enumerate(prefix["payloads"]):
        result = validate_and_apply_stateful_scene(
            payload, contracts[position], ledger)
        if result["decision"] != "accept_stateful_scene":
            raise ValueError(
                f"accepted_scene_prefix_revalidation_failed:S{position + 1}:" +
                ",".join(result["errors"]))
        ledger = result["ledger"]
        computed_ledgers.append(ledger)
    if ledger["ledger_sha256"] != prefix["ledger"]["ledger_sha256"]:
        raise ValueError("accepted_scene_prefix_recomputed_ledger_drift")
    for position, (session, scene_ledger) in enumerate(zip(
            prefix["sessions"], computed_ledgers), 1):
        if session.get("state_ledger_sha256_after") != scene_ledger[
                "ledger_sha256"]:
            raise ValueError(f"accepted_scene_prefix_session_ledger_drift:S{position}")
        atomic_json(realizer_root / f"accepted/session_{position:02d}.json",
                    session)
        atomic_json(realizer_root / f"frozen/state_ledger_after_S{position}.json",
                    scene_ledger)
    manifest_sha = sha256_file(realizer_root / "config/run_manifest.json")
    session_sha = {
        f"session_{position:02d}.json": sha256_file(
            realizer_root / f"accepted/session_{position:02d}.json")
        for position in range(1, count + 1)}
    atomic_json(realizer_root / "accepted/index.json", {
        "schema_version": "stateful_accepted_session_index_v1",
        "run_id": realizer_manifest["run_id"],
        "run_manifest_sha256": manifest_sha,
        "session_sha256": session_sha,
        "seeded_scene_ids": [f"S{index}" for index in range(1, count + 1)],
        "source_prefix_sha256": prefix["sha256"],
    })
    atomic_json(realizer_root / "frozen/accepted_prefix_attestation.json", {
        "source_run": str(prefix["run"]),
        "source_sha256": prefix["sha256"],
        "source_realizer_raw_read": False,
        "source_request_ledger_read": False,
        "source_state_ledger_read": True,
        "scene_count": count,
        "last_scene_id": prefix["last_scene_id"],
        "run_local_ledger_sha256": ledger["ledger_sha256"],
        "dialogue_preserved": all(
            _session_model_payload(source) == payload
            for source, payload in zip(prefix["sessions"], prefix["payloads"])),
    })
    return {"sessions": list(prefix["sessions"]), "ledger": ledger}


def seed_frozen_s1(
    *, realizer_root: Path, realizer_manifest: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]], accepted_s1: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehydrate S1 into the full ten-scene ledger without model generation."""
    if len(contracts) != 10 or contracts[0].get("scene_id") != "S1":
        raise ValueError("s1_seed_requires_full_scene_contracts")
    initial = initialize_stateful_ledger(contracts)
    result = validate_and_apply_stateful_scene(
        accepted_s1["model_payload"], contracts[0], initial)
    if result["decision"] != "accept_stateful_scene":
        raise ValueError("frozen_s1_revalidation_failed:" + ",".join(
            result["errors"]))
    next_ledger = result["ledger"]
    source_session = dict(accepted_s1["session"])
    seeded_session = {
        **source_session,
        "state_ledger_sha256_after": next_ledger["ledger_sha256"],
        "seeded_from_frozen_s1": True,
        "source_session_sha256": accepted_s1["sha256"]["session"],
        "source_payload_canonical_sha256": canonical_sha256(
            accepted_s1["model_payload"]),
    }
    atomic_json(realizer_root / "accepted/session_01.json", seeded_session)
    manifest_sha = sha256_file(realizer_root / "config/run_manifest.json")
    session_sha = sha256_file(realizer_root / "accepted/session_01.json")
    atomic_json(realizer_root / "accepted/index.json", {
        "schema_version": "stateful_accepted_session_index_v1",
        "run_id": realizer_manifest["run_id"],
        "run_manifest_sha256": manifest_sha,
        "session_sha256": {"session_01.json": session_sha},
        "seeded_scene_ids": ["S1"],
        "source_session_sha256": accepted_s1["sha256"]["session"],
    })
    atomic_json(realizer_root / "frozen/state_ledger_after_S1.json", next_ledger)
    atomic_json(realizer_root / "frozen/s1_seed_attestation.json", {
        "source_run": str(accepted_s1["run"]),
        "source_sha256": accepted_s1["sha256"],
        "source_dialogue_preserved": [
            {"speaker": turn["speaker"], "text": turn["text"]}
            for turn in seeded_session["dialogue_turns"]
        ] == accepted_s1["model_payload"]["dialogue_turns"],
        "source_realizer_raw_read": False,
        "source_request_ledger_read": False,
        "full_ledger_stage_count": len(next_ledger["stage_states"]),
        "accepted_s1_stage_status": next_ledger["stage_states"][
            "E1:stage_01"]["status"],
        "run_local_ledger_sha256": next_ledger["ledger_sha256"],
    })
    return {"session": seeded_session, "ledger": next_ledger}


def _all_attempts(realizer_root: Path) -> list[dict[str, Any]]:
    result = []
    for state_path in sorted((realizer_root / "ledger/requests").glob(
            "stateful-realizer-s*-a*.json")):
        state = _read(state_path)
        report_path = realizer_root / f"reports/{state_path.name}"
        report = _read(report_path) if report_path.is_file() else {}
        result.append({
            "request_id": state["request_id"],
            "scene_id": (state.get("request_metadata") or {}).get("scene_id"),
            "attempt": state.get("attempt"),
            "decision": report.get("decision") or state.get("final_decision"),
            "actual_dialogue_tokens": report.get("actual_dialogue_tokens"),
            "actual_turn_count": report.get("actual_turn_count"),
            "errors": list(report.get("reasons") or []),
            "finish_reason": ((state.get("response_metadata") or {}).get(
                "finish_reason")),
            "usage": ((state.get("response_metadata") or {}).get("usage")),
            "prompt_sha256": state.get("prompt_sha256"),
            "stage_diagnostics": list(report.get("stage_diagnostics") or []),
            "handoff_diagnostics": list(
                report.get("handoff_diagnostics") or []),
            "allowed_handoff_ids": list(
                report.get("allowed_handoff_ids") or []),
            "metadata_only_dialogue_preservation_required": report.get(
                "metadata_only_dialogue_preservation_required"),
            "metadata_only_dialogue_preserved": report.get(
                "metadata_only_dialogue_preserved"),
            "dialogue_turns_sha_before": report.get(
                "dialogue_turns_sha_before"),
            "dialogue_turns_sha_after": report.get(
                "dialogue_turns_sha_after"),
        })
    return result


def _marker_normalization_audit(
    *, sessions: Sequence[Mapping[str, Any]],
    contracts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report exact-vs-apostrophe-normalized matching without rewriting text."""
    contract_by_scene = {str(row["scene_id"]): row for row in contracts}
    rows = []
    for session in sessions:
        scene_id = str(session["graph_scene_id"])
        contract = contract_by_scene[scene_id]
        updates = {str(row["stage_id"]): row for row in
                   session.get("candidate_stage_updates") or []}
        turns = session.get("dialogue_turns") or []
        stages = []
        for stage in contract["assigned_event_stages"]:
            update = updates[stage["stage_id"]]
            evidence_index = update.get("evidence_turn_index")
            evidence = (turns[evidence_index]
                        if isinstance(evidence_index, int) and
                        0 <= evidence_index < len(turns) else {})
            text = str(evidence.get("text") or "")
            phrases = list(stage["completion_evidence_phrases"])
            exact = any(str(phrase).casefold() in text.casefold()
                        for phrase in phrases)
            normalized = any(
                normalize_terminal_marker_text(phrase) in
                normalize_terminal_marker_text(text)
                for phrase in phrases)
            stages.append({
                "stage_id": stage["stage_id"],
                "evidence_turn_index": evidence_index,
                "evidence_speaker": evidence.get("speaker"),
                "required_speakers": list(stage[
                    "authorized_result_speakers"]),
                "exact_marker_match": exact,
                "normalized_marker_match": normalized,
                "apostrophe_normalization_used": normalized and not exact,
                "original_evidence_text": text,
                "required_markers": phrases,
            })
        rows.append({
            "scene_id": scene_id,
            "stages": stages,
            "apostrophe_normalization_used": any(
                row["apostrophe_normalization_used"] for row in stages),
        })
    return rows


def _render_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Realizer S2–S10 sequential canary", "",
        f"- decision: `{summary['decision']}`",
        f"- generation_ready: `{str(summary['generation_ready']).lower()}`",
        f"- realizer_s2_ready: `{str(summary['realizer_s2_ready']).lower()}`",
        f"- scene scope end: `S{summary['scene_scope_end']}`",
        f"- Planner API: `{summary['planner_api_calls']}`",
        f"- Graph API: `{summary['graph_api_calls']}`",
        f"- Realizer API: `{summary['realizer_api_calls']}`",
        "- terminal marker placement: `per_stage_evidence_turn`",
        "", "| Scene | Success | Attempt | Turns | Tokens | Failed attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["realizer_scenes"]:
        lines.append(
            f"| `{row['scene_id']}` | {row['success']} | "
            f"{row['accepted_attempt'] or '-'} | "
            f"{row['accepted_turn_count']} | {row['accepted_tokens']} | "
            f"{len(row['failed_attempts'])} |")
    lines.append("")
    return "\n".join(lines)


def run(
    *, source_run: Path, s1_run: Path, run_root: Path,
    api_config_path: Path, tokenizer_path: Path, tokenizer: Any,
    prior_test_results: Mapping[str, Any],
    scene_end: int = 10,
    accepted_prefix_run: Path | None = None,
) -> dict[str, Any]:
    if not 2 <= scene_end <= 10:
        raise ValueError("realizer_scene_end_out_of_range")
    partial_scope = scene_end < 10
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"realizer_s2_s10_run_root_exists:{run_root}")
    source_run = source_run.resolve()
    s1_run = s1_run.resolve()
    source_tree_before = tree_sha256(source_run)
    s1_tree_before = tree_sha256(s1_run)
    source = load_frozen_s1_source(source_run)
    accepted_s1 = load_accepted_s1(s1_run)
    prefix = (load_accepted_scene_prefix(accepted_prefix_run)
              if accepted_prefix_run is not None else None)
    prefix_count = int(prefix["scene_count"]) if prefix else 1
    if prefix_count >= scene_end:
        raise ValueError("accepted_scene_prefix_already_reaches_scope_end")
    prefix_tree_before = tree_sha256(prefix["run"]) if prefix else None
    contracts = list(source["contracts_blob"]["contracts"])
    source_manifest = _read(source_run / "config/run_manifest.json")
    input_path = Path(source_manifest["input_unified_root_path"])
    input_sha_before = sha256_file(input_path)
    if input_sha_before != source_manifest["input_unified_root_sha256"]:
        raise ValueError("s2_s10_input_sha_drift")

    api = replace(
        load_challenger_config(api_config_path), model=CHALLENGER_MODEL,
        temperature=0.25, enable_thinking=False, timeout_seconds=600,
        realizer_max_tokens=REALIZER_MAX_TOKENS)
    run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, run_root, False)
    diff_sha = project_diff_sha256(PROJECT)
    run_id = (f"cc-train-episode-61555:realizer-s{scene_end}-only:canary:01"
              if partial_scope else
              "cc-train-episode-61555:realizer-s2-s10:canary:03")
    manifest = {
        "schema_version": "realizer_s2_s10_canary_manifest_v1",
        "mode": REALIZER_S2_S10_CANARY_VERSION,
        "run_id": run_id, "root_id": "cc-train-episode-61555",
        "sample_id": source["parent_contract"]["sample_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_head": git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256": diff_sha, **snapshot,
        "source_planner_graph_run": str(source_run),
        "source_planner_graph_tree_sha256_before": source_tree_before,
        "source_s1_run": str(s1_run),
        "source_s1_tree_sha256_before": s1_tree_before,
        "source_artifact_sha256": source["source_file_sha256"],
        "arc_canonical_sha256": source["arc_canonical_sha256"],
        "source_s1_sha256": accepted_s1["sha256"],
        "source_accepted_prefix_run": (
            str(prefix["run"]) if prefix else None),
        "source_accepted_prefix_tree_sha256_before": prefix_tree_before,
        "source_accepted_prefix_sha256": (
            prefix["sha256"] if prefix else None),
        "accepted_prefix_scene_count": prefix_count,
        "input_unified_root_path": str(input_path),
        "input_unified_root_sha256": input_sha_before,
        "scene_scope": [f"S{index}" for index in range(
            prefix_count + 1, scene_end + 1)],
        "scene_scope_start": prefix_count + 1,
        "scene_scope_end": scene_end,
        "partial_scope": partial_scope,
        "s1_reused_without_api": True,
        "accepted_prefix_reused_without_api": prefix is not None,
        "resume": False, "max_realizer_attempts_per_scene": 3,
        "max_realizer_physical_requests": 3 * (scene_end - prefix_count),
        "planner_api_calls": 0, "graph_api_calls": 0,
        "terminal_marker_placement_contract": {
            "status": "resolved",
            "prompt_requirement": "each marker in its stage evidence turn",
            "validator_requirement": "each marker in its stage evidence turn",
            "final_turn_required": False,
        },
        "metadata_only_dialogue_preservation_enforced": True,
        "realizer_prompt_version": STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
        "realizer_retry_version": STATEFUL_REALIZER_DIRECTED_RETRY_VERSION,
        "terminal_marker_normalization_version":
            STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
        "prior_test_results": dict(prior_test_results),
        "questioner_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
    }
    atomic_json(run_root / "config/run_manifest.json", manifest)
    for name, value in (
        ("authoritative_overview", source["overview"]),
        ("authoritative_arcs", source["raw_outline"]),
        ("validated_graph", source["validated_graph"]),
        ("scene_contracts", source["contracts_blob"]),
        ("parent_dialogue", source["parent_contract"]),
        ("accepted_s1_source_summary", accepted_s1["summary"]),
    ):
        atomic_json(run_root / f"frozen/{name}.json", value)
    if prefix is not None:
        atomic_json(run_root / "frozen/accepted_prefix_source.json", {
            "source_run": str(prefix["run"]),
            "scene_count": prefix_count,
            "last_scene_id": prefix["last_scene_id"],
            "sha256": prefix["sha256"],
        })
    atomic_json(run_root / "reports/preflight.json", {
        "decision": "pass", "fresh_run": True, "resume": False,
        "planner_api_calls": 0, "graph_api_calls": 0,
        "source_graph_validated": source["graph_validation"]["valid"],
        "authoritative_arc_count": len(
            source["raw_outline"].get("event_arcs") or []),
        "arc_canonical_sha256": source["arc_canonical_sha256"],
        "scene_contract_count": len(contracts),
        "s1_source_authoritative": True,
        "s1_source_dialogue_sha256": accepted_s1["sha256"]["session"],
        "accepted_prefix_scene_count": prefix_count,
        "accepted_prefix_state_ledger_revalidated": prefix is not None,
        "terminal_marker_placement_contract": manifest[
            "terminal_marker_placement_contract"],
    })

    realizer_root = run_root / "realizer"
    realizer_manifest = {
        "run_id": f"{run_id}:realizer", "root_id": manifest["root_id"],
        "mode": "frozen_graph_s2_s10_sequential_realizer",
        "scene_scope_end": scene_end,
        "project_git_head": manifest["project_git_head"],
        "project_diff_sha256": diff_sha,
        "source_graph_sha256": source["source_file_sha256"][
            "validated_graph"],
        "scene_contracts_sha256": canonical_sha256(contracts),
        "source_s1_session_sha256": accepted_s1["sha256"]["session"],
        "source_accepted_prefix_sha256": (
            prefix["sha256"] if prefix else None),
        "graph_scene_contract_version": STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
        "state_ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
        "graph_expansion_version": STATEFUL_GRAPH_EXPANSION_VERSION,
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
    if prefix is None:
        seed_frozen_s1(
            realizer_root=realizer_root, realizer_manifest=realizer_manifest,
            contracts=contracts, accepted_s1=accepted_s1)
    else:
        seed_frozen_scene_prefix(
            realizer_root=realizer_root, realizer_manifest=realizer_manifest,
            contracts=contracts, prefix=prefix)

    failure: str | None = None
    sessions: list[dict[str, Any]] = []
    completion: dict[str, Any] = {"complete": False}
    try:
        result = StatefulGraphRoundOrchestrator(
            run_root=realizer_root, manifest=realizer_manifest,
            contracts=contracts, tokenizer=tokenizer,
            realizer=LiveRealizerBackend(
                OpenAICompatibleChallengerClient(api)), backend_mode="live",
            max_attempts_per_scene=3,
            max_realizer_generations=3 * (scene_end - prefix_count)).run(
                scene_limit=scene_end)
        sessions = list(result["sessions"])
        completion = dict(result["completion"])
    except Exception as error:
        failure = f"{type(error).__name__}:{error}"
        atomic_json(run_root / "reports/fatal_error.json", {
            "error_type": type(error).__name__, "error": str(error)})
        sessions = [_read(path) for path in sorted(
            (realizer_root / "accepted").glob("session_*.json"))]

    _request_audit(run_root)
    usage = api_usage(run_root)
    parent = source["parent_turns"]
    final_ledger_path = realizer_root / "frozen/final_state_ledger.json"
    if final_ledger_path.is_file():
        final_ledger = _read(final_ledger_path)
    else:
        ledgers = sorted(realizer_root.glob("frozen/state_ledger_after_S*.json"))
        final_ledger = _read(ledgers[-1]) if ledgers else None
    audit = audit_child_generation(
        parent=parent, outline=source["validated_graph"], contracts=contracts,
        sessions=sessions, final_ledger=final_ledger)
    child = audit.pop("child_turns")
    child_turn_ids = [str(turn.get("turn_id")) for turn in child]
    child_session_ids = list(dict.fromkeys(
        str(turn.get("session_id")) for turn in child))
    normalized_turn_texts = [
        " ".join(str(turn.get("text") or "").casefold().split())
        for turn in child
    ]
    audit.update({
        "child_turn_ids_unique":
            len(child_turn_ids) == len(set(child_turn_ids)),
        "child_session_ids_in_expected_order":
            child_session_ids == [f"session_{index:02d}"
                                  for index in range(1, len(sessions) + 1)],
        "exact_duplicate_child_turn_text_count":
            len(normalized_turn_texts) - len(set(normalized_turn_texts)),
    })
    marker_audit = _marker_normalization_audit(
        sessions=sessions, contracts=contracts)
    marker_by_scene = {row["scene_id"]: row for row in marker_audit}
    marker_alignment_pass = all(
        stage["normalized_marker_match"] and
        stage["evidence_speaker"] in stage["required_speakers"]
        for scene in marker_audit for stage in scene["stages"])
    source_tree_after = tree_sha256(source_run)
    s1_tree_after = tree_sha256(s1_run)
    prefix_tree_after = tree_sha256(prefix["run"]) if prefix else None
    input_sha_after = sha256_file(input_path)
    intents = usage["completed_requests_by_intent"]
    source_session_dialogue = [
        {"speaker": turn["speaker"], "text": turn["text"]}
        for turn in accepted_s1["session"]["dialogue_turns"]
    ]
    run_s1_dialogue = [
        {"speaker": turn["speaker"], "text": turn["text"]}
        for turn in (sessions[0].get("dialogue_turns") or [])
    ] if sessions else []
    checks = {
        "git_head_match": git_value(PROJECT, "rev-parse", "HEAD") ==
            manifest["project_git_head"],
        "project_diff_sha256_match": project_diff_sha256(PROJECT) == diff_sha,
        "source_snapshot_match": sha256_file(Path(
            manifest["source_snapshot_manifest_path"])) ==
            manifest["source_snapshot_manifest_sha256"],
        "planner_graph_source_tree_unchanged":
            source_tree_before == source_tree_after,
        "s1_source_tree_unchanged": s1_tree_before == s1_tree_after,
        "input_unified_root_unchanged": input_sha_before == input_sha_after,
        "s1_dialogue_unchanged": source_session_dialogue == run_s1_dialogue,
        "accepted_prefix_source_tree_unchanged": (
            prefix_tree_before == prefix_tree_after if prefix else True),
        "accepted_prefix_dialogues_unchanged": (
            all(_session_model_payload(source_session) ==
                _session_model_payload(run_session)
                for source_session, run_session in zip(
                    prefix["sessions"], sessions[:prefix_count]))
            if prefix else True),
        "fresh_no_resume": manifest["resume"] is False,
        "no_reused_api_response": usage["reused_response_count"] == 0,
        "only_realizer_api_intent": set(intents) <= {"stateful_realize_scene"},
        "planner_api_zero": not any("outline" in key for key in intents),
        "graph_api_zero": True,
    }
    scope_pass = all((
        len(sessions) == scene_end,
        audit["parent_child_turn_ids_disjoint"],
        audit["parent_child_session_ids_disjoint"],
        audit["child_turn_ids_unique"],
        audit["child_session_ids_in_expected_order"],
        audit["scene_parent_continuity"],
        audit["isolated_new_character_count"] == 0,
        audit["state_contradiction_count"] == 0,
        audit["exact_repeated_or_template_scene_count"] == 0,
        marker_alignment_pass,
        all(checks.values()),
    ))
    if partial_scope:
        expected_evidence_stages = sum(
            len(contract["assigned_event_stages"])
            for contract in contracts[:scene_end])
        scope_pass = scope_pass and all((
            [session.get("graph_scene_id") for session in sessions] ==
                [f"S{index}" for index in range(1, scene_end + 1)],
            audit["evidence_bearing_stage_count"] == expected_evidence_stages,
            completion.get("complete") is False,
        ))
    else:
        scope_pass = scope_pass and all((
            completion.get("complete") is True,
            audit["planner_event_evidence_coverage"] == 1.0,
        ))
    full_pass = scope_pass and not partial_scope
    if full_pass:
        atomic_json(run_root / "child_dialogue.json", {
            "schema_version": "cc_frozen_graph_child_dialogue_v1",
            "root_id": manifest["root_id"],
            "session_count": len(sessions), "turn_count": len(child),
            "turns": child,
        })
    elif partial_scope and scope_pass:
        scope_session = sessions[scene_end - 1]
        atomic_json(run_root / f"s{scene_end}_dialogue.json", {
            "schema_version": f"cc_frozen_graph_s{scene_end}_dialogue_v1",
            "root_id": manifest["root_id"],
            "session_count": 1,
            "turn_count": len(scope_session.get("dialogue_turns") or []),
            "turns": scope_session.get("dialogue_turns") or [],
            "source_s1_session_sha256": accepted_s1["sha256"]["session"],
            "source_accepted_prefix_sha256": (
                prefix["sha256"] if prefix else None),
        })
    attempts = _all_attempts(realizer_root)
    scene_results = _scene_results(
        realizer_root, sessions,
        not_started_reason=(failure if failure else None))[:scene_end]
    for row in scene_results:
        scene_attempts = [attempt for attempt in attempts
                          if attempt["scene_id"] == row["scene_id"]]
        row["request_count"] = len(scene_attempts)
        row["first_failure_reasons"] = (
            scene_attempts[0]["errors"]
            if scene_attempts and scene_attempts[0]["errors"] else [])
        row["final_result"] = "accept" if row["success"] else (
            "not_called" if not scene_attempts else "retry_exhausted")
        row["marker_normalization"] = marker_by_scene.get(row["scene_id"])
        row["stage_diagnostics_by_attempt"] = [{
            "attempt": attempt["attempt"],
            "decision": attempt["decision"],
            "stage_diagnostics": attempt["stage_diagnostics"],
            "handoff_diagnostics": attempt["handoff_diagnostics"],
            "allowed_handoff_ids": attempt["allowed_handoff_ids"],
            "metadata_only_dialogue_preservation_required": attempt[
                "metadata_only_dialogue_preservation_required"],
            "metadata_only_dialogue_preserved": attempt[
                "metadata_only_dialogue_preserved"],
            "dialogue_turns_sha_before": attempt[
                "dialogue_turns_sha_before"],
            "dialogue_turns_sha_after": attempt[
                "dialogue_turns_sha_after"],
        } for attempt in scene_attempts]
    parent_tokens = _dialogue_tokens(tokenizer, parent)
    child_tokens = _dialogue_tokens(tokenizer, child) if child else 0
    summary = {
        "schema_version": (f"realizer_s{scene_end}_only_canary_summary_v1"
                           if partial_scope else
                           "realizer_s2_s10_canary_summary_v3"),
        "decision": ((f"realizer_s{scene_end}_only_canary_pass"
                      if scope_pass else
                      f"realizer_s{scene_end}_only_canary_failed")
                     if partial_scope else
                     ("realizer_s2_s10_canary_pass" if full_pass else
                      "realizer_s2_s10_canary_failed")),
        "realizer_s1_ready": True,
        "realizer_s2_ready": len(sessions) >= 2,
        "realizer_scene_scope_ready": scope_pass,
        f"realizer_s{scene_end}_ready": scope_pass,
        "generation_ready": full_pass,
        "scene_scope_end": scene_end,
        "scope_claim": "frozen_planner_graph_s2_s10_sequential_realizer",
        "planner_api_calls": 0, "graph_api_calls": 0,
        "realizer_api_calls": int(intents.get("stateful_realize_scene") or 0),
        "source_s1_reused_without_api": True,
        "old_realizer_raw_response_read": False,
        "old_realizer_request_ledger_read": False,
        "terminal_marker_placement_contract": manifest[
            "terminal_marker_placement_contract"],
        "metadata_only_dialogue_preservation_enforced": manifest[
            "metadata_only_dialogue_preservation_enforced"],
        "terminal_marker_normalization_version":
            STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION,
        "marker_normalization_audit": marker_audit,
        "marker_normalization_used_scene_count": sum(
            row["apostrophe_normalization_used"] for row in marker_audit),
        "realizer_attempts": attempts,
        "realizer_scenes": scene_results,
        "completed_scene_count": len(sessions),
        "stateful_completion": completion,
        "generation_audit": audit,
        "parent_dialogue_tokens": parent_tokens,
        "child_dialogue_tokens": child_tokens,
        "new_child_tokens": child_tokens,
        "final_child_session_count": len(sessions),
        "final_child_turn_count": len(child),
        "api_usage": usage,
        "failure_diagnosis": failure,
        "input_unified_root_sha256_before": input_sha_before,
        "input_unified_root_sha256_after": input_sha_after,
        "source_planner_graph_tree_sha256_before": source_tree_before,
        "source_planner_graph_tree_sha256_after": source_tree_after,
        "source_s1_tree_sha256_before": s1_tree_before,
        "source_s1_tree_sha256_after": s1_tree_after,
        "source_accepted_prefix_tree_sha256_before": prefix_tree_before,
        "source_accepted_prefix_tree_sha256_after": prefix_tree_after,
        "provenance_checks": checks,
        "authoritative_release_provenance": all(checks.values()),
        "provenance_mixed": False,
        "questioner_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
    }
    summary_stem = (f"s{scene_end}_canary_summary" if partial_scope
                    else "s2_s10_canary_summary")
    atomic_json(run_root / f"reports/{summary_stem}.json", summary)
    atomic_text(run_root / f"reports/{summary_stem}.md",
                _render_summary(summary))
    atomic_json(run_root / "reports/post_run_provenance.json", {
        "schema_version": "realizer_s2_s10_canary_provenance_v1",
        "checks": checks,
        "authoritative_release_provenance": all(checks.values()),
        "provenance_mixed": False,
        "realizer_s2_ready": summary["realizer_s2_ready"],
        "realizer_scene_scope_ready": scope_pass,
        f"realizer_s{scene_end}_ready": scope_pass,
        "generation_ready": full_pass,
        "run_manifest_sha256": sha256_file(
            run_root / "config/run_manifest.json"),
        "summary_sha256": sha256_file(
            run_root / f"reports/{summary_stem}.json"),
    })
    return summary
