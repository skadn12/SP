from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .graph_expansion import (GRAPH_EXPANSION_VERSION, GRAPH_REALIZER_PROMPT_VERSION,
                              GRAPH_REALIZER_RETRY_FEEDBACK_VERSION,
                              GRAPH_REALIZER_TURN_NORMALIZATION_VERSION,
                              GRAPH_SCENE_HARD_TOKEN_RANGE,
                              GRAPH_SCENE_CONTRACT_VERSION,
                              GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
                              GRAPH_TOTAL_HARD_MINIMUM_TOKENS,
                              GRAPH_TOTAL_TARGET_TOKEN_RANGE,
                              compile_graph_scene_contracts,
                              graph_contract_sha256,
                              graph_scene_coverage_report)
from .input_adapter import load_lineage, read_json
from .live_backends import (LiveRealizerBackend, OpenAICompatibleChallengerClient,
                            load_challenger_config)
from .orchestrator import CanaryOrchestrator
from .prompts import graph_realizer_template_hash
from .run_canary import (DEFAULTS, create_source_snapshot, freeze_json, git_value,
                         initialize_run, project_diff_sha256, write_post_run_provenance)
from .validation import validate_config


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
SPELL_COMMIT = "4e5aee32280febb7b0844237ba087f837e3c8905"
CHALLENGER_MODEL = "Qwen/Qwen3-32B"

def _tree_sha256(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(f"{path.relative_to(root)}\0{sha256_file(path)}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _attest_accepted_prefix(
        source_root: Path, *, lineage: str, contracts: list[dict[str, Any]],
        outline_sha256: str, parent_anchor_sha256: str,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    manifest_path = source_root / "config/run_manifest.json"
    index_path = source_root / "accepted/index.json"
    if not manifest_path.is_file() or not index_path.is_file():
        raise RuntimeError("accepted_prefix_source_files_missing")
    manifest = read_json(manifest_path)
    index = read_json(index_path)
    manifest_sha = sha256_file(manifest_path)
    if (manifest.get("lineage") != lineage or
            manifest.get("outline_sha256") != outline_sha256 or
            manifest.get("parent_anchor_snapshot_sha256") != parent_anchor_sha256 or
            manifest.get("graph_scene_contracts_sha256") !=
            graph_contract_sha256(contracts) or
            manifest.get("graph_stage_accounting_mode") !=
            GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY):
        raise RuntimeError("accepted_prefix_source_contract_mismatch")
    if (index.get("run_id") != manifest.get("run_id") or
            index.get("run_manifest_sha256") != manifest_sha):
        raise RuntimeError("accepted_prefix_index_identity_mismatch")
    declared = dict(index.get("session_sha256") or {})
    session_paths = sorted((source_root / "accepted").glob("session_*.json"))
    actual = {path.name: sha256_file(path) for path in session_paths}
    if declared != actual or not session_paths or len(session_paths) >= len(contracts):
        raise RuntimeError("accepted_prefix_session_index_invalid")
    expected_names = [f"session_{position:02d}.json"
                      for position in range(1, len(session_paths) + 1)]
    if [path.name for path in session_paths] != expected_names:
        raise RuntimeError("accepted_prefix_not_contiguous")
    request_attestations = []
    for position, path in enumerate(session_paths, 1):
        session = read_json(path)
        request_id = str(session.get("request_id") or "")
        state_path = source_root / f"ledger/requests/{request_id}.json"
        if (session.get("session_index") != position or
                session.get("graph_scene_id") != contracts[position - 1]["scene_id"] or
                session.get("graph_scene_contract_sha256") !=
                contracts[position - 1]["contract_sha256"] or
                session.get("final_decision") != "accept" or
                not state_path.is_file()):
            raise RuntimeError("accepted_prefix_session_contract_invalid")
        state = read_json(state_path)
        raw_path = Path(str(state.get("raw_response_path") or ""))
        if (state.get("status") != "completed" or
                state.get("final_decision") != "accept_session" or
                not raw_path.is_file() or
                sha256_file(raw_path) != state.get("raw_response_sha256")):
            raise RuntimeError("accepted_prefix_request_attestation_invalid")
        request_attestations.append({
            "session_index": position, "request_id": request_id,
            "request_state_sha256": sha256_file(state_path),
            "raw_response_sha256": state["raw_response_sha256"],
        })
    return {
        "source_root": str(source_root),
        "source_tree_sha256_before": _tree_sha256(source_root),
        "source_manifest_sha256": manifest_sha,
        "source_index_sha256": sha256_file(index_path),
        "accepted_prefix_count": len(session_paths),
        "session_sha256": actual,
        "request_attestations": request_attestations,
        "source_lineage": lineage,
        "source_challenger_model": manifest.get("challenger_model_identifier"),
        "source_historical_run_modified": False,
    }


def _materialize_accepted_prefix(
        run_root: Path, *, attestation: dict[str, Any],
        run_id: str, run_manifest_sha256: str,
) -> None:
    source_root = Path(attestation["source_root"])
    atomic_json(run_root / "frozen/accepted_prefix_attestation.json", attestation)
    copied = {}
    for name, expected_sha in attestation["session_sha256"].items():
        source = source_root / "accepted" / name
        destination = run_root / "accepted" / name
        atomic_text(destination, source.read_text(encoding="utf-8"))
        if sha256_file(destination) != expected_sha:
            raise RuntimeError("accepted_prefix_copy_hash_mismatch")
        copied[name] = expected_sha
    atomic_json(run_root / "accepted/index.json", {
        "schema_version": "accepted_session_integrity_v2",
        "run_id": run_id,
        "run_manifest_sha256": run_manifest_sha256,
        "session_sha256": copied,
        "source_prefix_attestation_sha256": sha256_bytes(
            canonical_json(attestation).encode()),
    })


def _usage_summary(run_root: Path) -> dict[str, int]:
    totals = {"physical_requests":0, "prompt_tokens":0,
              "completion_tokens":0, "total_tokens":0}
    for path in sorted((run_root / "ledger/requests").glob("realizer-*.json")):
        state = read_json(path)
        if state.get("status") != "completed":
            continue
        totals["physical_requests"] += 1
        usage = (state.get("response_metadata") or {}).get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[key] += int(usage.get(key) or 0)
    return totals


def _complexity_summary(outline: dict[str, Any], sessions: list[dict[str, Any]],
                        child_tokens: int) -> dict[str, Any]:
    speakers = sorted({turn.get("speaker") for session in sessions
                       for turn in session.get("dialogue_turns") or []})
    introduced = sorted({name for session in sessions
                         for name in session.get("newly_introduced_character_names") or []})
    locations = [session.get("session_location") for session in sessions]
    return {"accepted_session_count":len(sessions),
        "dialogue_turn_count":sum(len(session.get("dialogue_turns") or []) for session in sessions),
        "dialogue_token_count":child_tokens,
        "speaker_names":speakers,
        "newly_introduced_character_names":introduced,
        "planned_new_character_names":[item.get("name") for item in
                                         outline.get("allowed_new_characters") or []],
        "session_locations":locations, "distinct_session_locations":sorted(set(locations)),
        "relative_times":[session.get("graph_relative_time") for session in sessions],
        "covered_event_stage_count":len({stage_id for session in sessions
                                          for stage_id in session.get(
                                              "covered_event_stage_ids") or []}),
        "planned_event_arc_count":len(outline.get("planned_event_arcs") or []),
        "planned_scene_count":len(outline.get("scene_plan") or [])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one complete restart-n5 Graph expansion")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lineage", default="conv-41")
    parser.add_argument("--api-config", type=Path,
        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--accepted-prefix-root", type=Path)
    args = parser.parse_args(argv)
    if args.resume and args.accepted_prefix_root is not None:
        raise ValueError("accepted_prefix_cannot_be_combined_with_resume")
    initialize_run(args.run_root, args.resume)
    legacy = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
    data = load_lineage(args.input_root, args.lineage, legacy)
    contracts = compile_graph_scene_contracts(data["outline"], coverage_only=True)
    outline_sha = sha256_file(args.input_root / args.lineage /
                              "expansion_outline.json")
    parent_anchor_sha = sha256_file(args.input_root / args.lineage /
                                    "parent_anchor_snapshot.json")
    prefix_attestation = (_attest_accepted_prefix(
        args.accepted_prefix_root, lineage=args.lineage, contracts=contracts,
        outline_sha256=outline_sha, parent_anchor_sha256=parent_anchor_sha)
        if args.accepted_prefix_root is not None else None)
    tokenizer_path = Path("/root/autodl-tmp/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    api = replace(load_challenger_config(args.api_config),
                  model=CHALLENGER_MODEL, temperature=0.25,
                  enable_thinking=False, timeout_seconds=600)
    spell_root = Path("/root/autodl-tmp/third_party_readonly/Qwen-Doc")
    if (git_value(spell_root, "rev-parse", "HEAD") != SPELL_COMMIT or
            git_value(spell_root, "status", "--porcelain")):
        raise RuntimeError("SPELL source provenance drift")
    config = {**DEFAULTS, "mode":"live_graph_expansion", "backend_mode":"live",
        "lineage":args.lineage, "input_root":str(args.input_root.resolve()),
        "run_root":str(args.run_root.resolve()), "max_total_paid_api_generations":30,
        "max_realizer_generations":30, "max_session_attempts_per_session":3,
        "max_questioner_generations":0, "max_post_freeze_memory_generations":0,
        "max_verifier_generations":0, "max_sessions":len(contracts),
        "graph_expansion_version":GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version":GRAPH_SCENE_CONTRACT_VERSION,
        "graph_stage_accounting_mode":GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "minimum_accepted_child_tokens":GRAPH_TOTAL_HARD_MINIMUM_TOKENS,
        "hard_minimum_child_tokens":GRAPH_TOTAL_HARD_MINIMUM_TOKENS,
        "target_total_child_token_range":list(GRAPH_TOTAL_TARGET_TOKEN_RANGE),
        "scene_hard_token_range":list(GRAPH_SCENE_HARD_TOKEN_RANGE),
        "target_child_tokens":int(data["outline"]["target"]["child_token_growth"]),
        "challenger_model_identifier":api.model,
        "challenger_temperature":api.temperature,
        "challenger_realizer_max_tokens":api.realizer_max_tokens}
    validate_config(config)
    config_path = args.run_root / "config/resolved_canary_config.json"
    contracts_path = args.run_root / "frozen/graph_scene_contracts.json"
    freeze_json(config_path, config, args.resume)
    freeze_json(contracts_path, {"version":GRAPH_SCENE_CONTRACT_VERSION,
        "contracts":contracts, "contracts_sha256":graph_contract_sha256(contracts)}, args.resume)
    snapshot = create_source_snapshot(PROJECT, args.run_root, args.resume)
    manifest_path = args.run_root / "config/run_manifest.json"
    manifest = {"run_id":args.run_root.name, "mode":"live_graph_expansion",
        "lineage":args.lineage, "created_at":datetime.now(timezone.utc).isoformat(),
        "project_git_commit":git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256":project_diff_sha256(PROJECT),
        "resolved_config_sha256":sha256_file(config_path),
        "source_snapshot_manifest_path":snapshot["source_snapshot_manifest_path"],
        "source_snapshot_manifest_sha256":snapshot["source_snapshot_manifest_sha256"],
        "orchestrator_source_sha256":sha256_file(MODULE_ROOT / "orchestrator.py"),
        "validation_source_sha256":sha256_file(MODULE_ROOT / "validation.py"),
        "ledger_source_sha256":sha256_file(MODULE_ROOT / "ledger.py"),
        "run_canary_source_sha256":sha256_file(MODULE_ROOT / "run_canary.py"),
        "mcq_parser_source_sha256":sha256_file(MODULE_ROOT / "core.py"),
        "questioner_prompt_builder_source_sha256":sha256_file(MODULE_ROOT / "prompts.py"),
        "questioner_contract_source_sha256":sha256_file(MODULE_ROOT / "questioner_contract.py"),
        "live_backend_source_sha256":sha256_file(MODULE_ROOT / "live_backends.py"),
        "graph_expansion_source_sha256":sha256_file(MODULE_ROOT / "graph_expansion.py"),
        "graph_expansion_runner_source_sha256":sha256_file(Path(__file__)),
        "graph_expansion_version":GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version":GRAPH_SCENE_CONTRACT_VERSION,
        "graph_stage_accounting_mode":GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "graph_realizer_turn_normalization_version":
            GRAPH_REALIZER_TURN_NORMALIZATION_VERSION,
        "graph_realizer_retry_feedback_version":
            GRAPH_REALIZER_RETRY_FEEDBACK_VERSION,
        "scene_hard_token_range":list(GRAPH_SCENE_HARD_TOKEN_RANGE),
        "target_total_child_token_range":list(GRAPH_TOTAL_TARGET_TOKEN_RANGE),
        "hard_minimum_child_tokens":GRAPH_TOTAL_HARD_MINIMUM_TOKENS,
        "source_declared_minimum_child_tokens":int(data["outline"]["target"][
            "minimum_accepted_child_tokens"]),
        "graph_scene_contracts_sha256":graph_contract_sha256(contracts),
        "graph_scene_contracts_file_sha256":sha256_file(contracts_path),
        "graph_realizer_prompt_version":GRAPH_REALIZER_PROMPT_VERSION,
        "graph_realizer_prompt_sha256":graph_realizer_template_hash(),
        "outline_sha256":outline_sha,
        "parent_anchor_snapshot_sha256":parent_anchor_sha,
        "parent_dialogue_sha256":data["anchors"]["dialogue_hash"],
        "tokenizer_path":str(tokenizer_path),
        "tokenizer_config_sha256":sha256_file(tokenizer_path / "tokenizer_config.json"),
        "spell_repo_path":str(spell_root), "spell_commit":SPELL_COMMIT, "spell_dirty":False,
        "challenger_model_identifier":api.model,
        "sampling_parameters":{"temperature":api.temperature,
                               "max_tokens":api.realizer_max_tokens},
        "accepted_prefix_attestation":prefix_attestation,
        "accepted_prefix_attestation_sha256":(sha256_bytes(
            canonical_json(prefix_attestation).encode())
            if prefix_attestation is not None else None),
        "budget":{"maximum_realizer_requests":30, "maximum_attempts_per_scene":3,
                  "questioner":0, "solver":0, "verifier":0, "training":0}}
    if args.resume and manifest_path.exists():
        frozen = read_json(manifest_path)
        old = {key:value for key,value in frozen.items() if key != "created_at"}
        new = {key:value for key,value in manifest.items() if key != "created_at"}
        if canonical_json(old) != canonical_json(new):
            raise RuntimeError("graph expansion resume manifest drift")
        manifest = frozen
    else:
        atomic_json(manifest_path, manifest)
    if prefix_attestation is not None:
        _materialize_accepted_prefix(
            args.run_root, attestation=prefix_attestation,
            run_id=manifest["run_id"],
            run_manifest_sha256=sha256_file(manifest_path))
    client = OpenAICompatibleChallengerClient(api)
    orchestrator = CanaryOrchestrator(run_root=args.run_root, manifest=manifest,
        config=config, data=data, tokenizer=tokenizer,
        realizer=LiveRealizerBackend(client), questioner=None, solver=None,
        backend_mode="live")
    sessions = orchestrator.realize()
    child, memory = orchestrator.freeze(sessions)
    child_tokens = orchestrator._tokens(child)
    completion = graph_scene_coverage_report(
        data["outline"], contracts, sessions, child_tokens,
        hard_minimum_child_tokens=GRAPH_TOTAL_HARD_MINIMUM_TOKENS,
        target_child_token_range=GRAPH_TOTAL_TARGET_TOKEN_RANGE)
    if not completion["complete"]:
        raise RuntimeError(f"graph expansion incomplete: {completion}")
    report = {"mode":"live_graph_expansion", "lineage":args.lineage,
        "accepted_prefix":prefix_attestation,
        "completion":completion,
        "complexity":_complexity_summary(data["outline"], sessions, child_tokens),
        "usage":_usage_summary(args.run_root), "durable_memory":{
            "accepted_count":memory.get("accepted_count"),
            "assertion_count":len(memory.get("assertions") or [])},
        "questioner_generations":0, "solver_generations":0,
        "verifier_generations":0, "training_runs":0}
    atomic_json(args.run_root / "reports/graph_expansion_report.json", report)
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    if prefix_attestation is not None:
        source_tree_after = _tree_sha256(Path(prefix_attestation["source_root"]))
        provenance.update({
            "accepted_prefix_source_tree_sha256_before":
                prefix_attestation["source_tree_sha256_before"],
            "accepted_prefix_source_tree_sha256_after":source_tree_after,
            "accepted_prefix_source_tree_unchanged":source_tree_after ==
                prefix_attestation["source_tree_sha256_before"],
        })
        provenance["authoritative_release_provenance"] = bool(
            provenance.get("authoritative_release_provenance") and
            provenance["accepted_prefix_source_tree_unchanged"])
        provenance["provenance_mixed"] = not provenance[
            "authoritative_release_provenance"]
        atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("graph expansion provenance mixed")
    print(json.dumps({"run_root":str(args.run_root), "report":report,
                      "provenance":provenance}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
