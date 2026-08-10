"""Run the original outline-Graph mainline repeatedly and probe QA after each round."""

from __future__ import annotations

import argparse
import collections
import http.client
import json
import ssl
import urllib.error
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from .core import atomic_json, sha256_bytes, sha256_file
from .graph_expansion import (
    GRAPH_EXPANSION_VERSION, GRAPH_REALIZER_PROMPT_VERSION,
    GRAPH_SCENE_CONTRACT_VERSION, GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
    compile_graph_scene_contracts, graph_contract_sha256,
    graph_scene_coverage_report,
)
from .input_adapter import read_json
from .ledger import RequestLedger
from .lineage_three_stage_canary import ApiBudget, run_single_object_task
from .live_backends import (LiveQuestionerBackend, LiveRealizerBackend,
                            LiveSemanticRepairBackend,
                            OpenAICompatibleChallengerClient,
                            load_challenger_config)
from .minimal_admission import SPAN_IDS_ONLY_SCHEMA_PROFILE, SPELL_STYLE_MINIMAL_MODE
from .orchestrator import CanaryOrchestrator
from .prompts import (QUESTIONER_FORMAT_RETRY_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
                      build_questioner_span_ids_only_feedback_prompt,
                      graph_realizer_template_hash,
                      questioner_format_retry_suffix_hash,
                      span_ids_only_questioner_template_hash)
from .questioner_contract import (build_span_catalog, default_reasoning_contracts,
                                  reasoning_contract_sha256, span_catalog_sha256)
from .rolling_outline_30k import (
    CANARY_VERSION, MAX_EXPANSION_ROUNDS, OUTLINE_PLANNER_PROMPT_VERSION,
    OUTLINE_PLANNER_SCHEMA_VERSION, RAW_OUTLINE_SCHEMA, TARGET_DIALOGUE_TOKENS,
    build_outline_planner_prompt, canonical_sha256, normalize_outline_plan,
    remap_baseline_turns, remap_round_child_turns, validate_raw_outline_plan,
)
from .run_canary import (DEFAULTS, create_source_snapshot, freeze_json, git_value,
                         project_diff_sha256, write_post_run_provenance)
from .run_cross_dialogue_calibration import (_base_manifest, _questioner_config,
                                              _run_questioner, _run_solver,
                                              _solver_config)
from .validation import validate_config
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_outline_graph_30k_round_canary_v1")
DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_spell_full_graph_expansion_v7")
DEFAULT_FEEDBACK = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_cross_dialogue_calibration_5x3_v1/frozen/"
    "calibration_feedback_memory.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
SOLVER = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "accepted_models/solver_base_restart_stage2_merged")
CHALLENGER_MODEL = "Qwen/Qwen3-32B"
MODEL_CONTEXT_LIMIT = 40_960


def _tokens(tokenizer: Any, turns: Sequence[Mapping[str, Any]]) -> int:
    text = "\n".join(f"[{turn['turn_id']}] {turn['speaker']}: {turn['text']}"
                     for turn in turns)
    return len(tokenizer.encode(text, add_special_tokens=False))


def select_baseline_turns(source: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    dialogue = source["dialogue"]
    parent = [dict(turn) for turn in dialogue["parent_turns"]]
    if scope == "parent_only":
        return parent
    if scope == "full":
        return parent + [dict(turn) for turn in dialogue["child_turns"]]
    raise ValueError(f"unsupported_baseline_scope:{scope}")


def load_imported_raw_outline(
        path: Path, cumulative: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    raw = read_json(resolved)
    validation = validate_raw_outline_plan(
        raw, cumulative_turns=list(cumulative))
    if validation.get("decision") != "rolling_outline_accept":
        raise RuntimeError(f"imported_raw_outline_invalid:{validation}")
    return raw, {
        "source_path": str(resolved),
        "source_sha256": sha256_file(resolved),
        "validation": validation,
        "semantic_regeneration_performed": False,
    }


def _source_attestation(source: Path) -> dict[str, Any]:
    manifest_path = source / "config/run_manifest.json"
    provenance_path = source / "reports/post_run_provenance.json"
    index_path = source / "accepted/index.json"
    dialogue_path = source / "frozen_parent_plus_child.json"
    for path in (manifest_path, provenance_path, index_path, dialogue_path):
        if not path.exists():
            raise RuntimeError(f"baseline_source_missing:{path.name}")
    manifest, provenance, index = (read_json(path) for path in
                                   (manifest_path, provenance_path, index_path))
    manifest_sha = sha256_file(manifest_path)
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False or
            provenance.get("run_manifest_sha256") != manifest_sha or
            index.get("run_manifest_sha256") != manifest_sha):
        raise RuntimeError("baseline_source_not_authoritative")
    session_hashes = index.get("session_sha256") or {}
    for name, expected in session_hashes.items():
        if sha256_file(source / "accepted" / name) != expected:
            raise RuntimeError(f"baseline_session_sha_mismatch:{name}")
    frozen = read_json(dialogue_path)
    if len(frozen.get("parent_turns") or []) != 237 or len(
            frozen.get("child_turns") or []) != 134:
        raise RuntimeError("baseline_dialogue_shape_mismatch")
    return {"source_root": str(source.resolve()),
        "source_manifest_sha256": manifest_sha,
        "source_provenance_sha256": sha256_file(provenance_path),
        "source_index_sha256": sha256_file(index_path),
        "source_dialogue_sha256": sha256_file(dialogue_path),
        "source_session_sha256": session_hashes,
        "parent_turn_count": 237, "child_turn_count": 134,
        "dialogue": frozen}


def _view(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dialogue = [dict(turn) for turn in turns]
    catalog = build_span_catalog(dialogue)
    return {"combined_dialogue": dialogue, "span_catalog": catalog,
        "view_sha256": canonical_sha256({"dialogue": dialogue,
                                           "span_catalog": catalog})}


def _usage(root: Path) -> dict[str, Any]:
    paid = collections.Counter()
    local = collections.Counter()
    usage = collections.Counter()
    for path in root.glob("**/ledger/requests/*.json"):
        row = read_json(path)
        if row.get("status") != "completed":
            continue
        if row.get("backend_mode") == "live":
            paid[str(row.get("intent"))] += 1
            api_usage = (row.get("response_metadata") or {}).get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += int(api_usage.get(key) or 0)
        elif row.get("backend_mode") == "local":
            local[str(row.get("intent"))] += 1
    return {"paid_api_by_intent": dict(sorted(paid.items())),
        "total_paid_api_requests": sum(paid.values()),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "local_vllm_by_intent": dict(sorted(local.items())),
        "local_vllm_generations": sum(local.values())}


def _attest_imported_round(root: Path, baseline: Sequence[Mapping[str, Any]],
                           tokenizer: Any) -> dict[str, Any]:
    report_path = root / "reports/round_expansion.json"
    cumulative_path = root / "frozen/cumulative_dialogue_after_round.json"
    index_path = root / "accepted/index.json"
    manifest_path = root / "config/run_manifest.json"
    for path in (report_path, cumulative_path, index_path, manifest_path):
        if not path.exists():
            raise RuntimeError(f"imported_round_missing:{path.name}")
    report, frozen, index = (read_json(path) for path in
                             (report_path, cumulative_path, index_path))
    if (not isinstance(report.get("round_index"), int) or
            report["round_index"] < 1 or
            report.get("decision") != "round_complete"):
        raise RuntimeError("imported_round_not_complete")
    turns = frozen.get("turns") or []
    if turns[:len(baseline)] != list(baseline):
        raise RuntimeError("imported_round_baseline_mismatch")
    if (_tokens(tokenizer, turns) != report.get("output_dialogue_tokens") or
            len(turns) != report.get("output_turn_count")):
        raise RuntimeError("imported_round_length_mismatch")
    manifest_sha = sha256_file(manifest_path)
    if index.get("run_manifest_sha256") != manifest_sha:
        raise RuntimeError("imported_round_manifest_mismatch")
    for name, expected in (index.get("session_sha256") or {}).items():
        if sha256_file(root / "accepted" / name) != expected:
            raise RuntimeError(f"imported_round_session_sha_mismatch:{name}")
    return {"source_round_root": str(root.resolve()),
        "source_round_manifest_sha256": manifest_sha,
        "source_round_index_sha256": sha256_file(index_path),
        "source_round_report_sha256": sha256_file(report_path),
        "source_round_cumulative_sha256": sha256_file(cumulative_path),
        "source_session_sha256": index.get("session_sha256") or {},
        "report": report, "turns": turns}


def _attest_imported_questioner_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError("imported_questioner_report_missing")
    report = read_json(path)
    if (not isinstance(report.get("round_index"), int) or
            report["round_index"] < 1 or not str(
                report.get("questioner_decision") or "")):
        raise RuntimeError("imported_questioner_report_invalid")
    qa_root = path.parent.parent
    manifest_path = qa_root / "config/run_manifest.json"
    provenance_path = qa_root / "reports/post_run_provenance.json"
    result = {"source_report_path": str(path.resolve()),
        "source_report_sha256": sha256_file(path),
        "round_index": report["round_index"], "report": report}
    if manifest_path.exists() and provenance_path.exists():
        provenance = read_json(provenance_path)
        if (provenance.get("authoritative_release_provenance") is not True or
                provenance.get("provenance_mixed") is not False or
                provenance.get("run_manifest_sha256") !=
                sha256_file(manifest_path)):
            raise RuntimeError("imported_questioner_not_authoritative")
        result.update({"source_manifest_sha256": sha256_file(manifest_path),
            "source_provenance_sha256": sha256_file(provenance_path),
            "attestation_mode": "completed_questioner_provenance"})
    else:
        preflight_path = qa_root / "frozen/questioner_prompt_preflight.json"
        if (report.get("questioner_decision") != "questioner_context_overflow" or
                not preflight_path.exists() or
                (report.get("questioner_prompt_preflight") or {}).get(
                    "decision") != "reject_context_overflow"):
            raise RuntimeError("imported_questioner_attestation_missing")
        result.update({"source_preflight_sha256": sha256_file(preflight_path),
            "attestation_mode": "deterministic_context_overflow"})
    return result


def _round_manifest(*, root: Path, round_index: int, cumulative: list[dict[str, Any]],
                    outline_prompt: str, top_manifest: Mapping[str, Any],
                    tokenizer: Any, api: Any) -> dict[str, Any]:
    return {"run_id": root.name, "mode": "live_graph_expansion",
        "canary_version": CANARY_VERSION, "round_index": round_index,
        "project_git_commit": top_manifest["project_git_commit"],
        "project_diff_sha256": top_manifest["project_diff_sha256"],
        "source_snapshot_manifest_path": top_manifest[
            "source_snapshot_manifest_path"],
        "source_snapshot_manifest_sha256": top_manifest[
            "source_snapshot_manifest_sha256"],
        "cumulative_input_sha256": canonical_sha256(cumulative),
        "cumulative_input_turn_count": len(cumulative),
        "cumulative_input_tokens": _tokens(tokenizer, cumulative),
        "outline_planner_prompt_version": OUTLINE_PLANNER_PROMPT_VERSION,
        "outline_planner_schema_version": OUTLINE_PLANNER_SCHEMA_VERSION,
        "outline_planner_schema_sha256": canonical_sha256(RAW_OUTLINE_SCHEMA),
        "outline_planner_semantic_prompt_sha256": sha256_bytes(
            outline_prompt.encode("utf-8")),
        "graph_expansion_version": GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version": GRAPH_SCENE_CONTRACT_VERSION,
        "graph_realizer_prompt_version": GRAPH_REALIZER_PROMPT_VERSION,
        "graph_realizer_prompt_sha256": graph_realizer_template_hash(),
        "tokenizer_path": str(TOKENIZER),
        "tokenizer_config_sha256": sha256_file(TOKENIZER / "tokenizer_config.json"),
        "challenger_model_identifier": api.model,
        "sampling_parameters": {"temperature": api.temperature,
            "outline_max_tokens": 5000, "realizer_max_tokens": api.realizer_max_tokens},
        "budget": {"outline_normal": 1, "outline_format_retry": 1,
            "realizer_physical_max": 30, "questioner": 0,
            "verifier": 0, "training": 0}}


def _expansion_config(root: Path, outline: Mapping[str, Any], api: Any) -> dict[str, Any]:
    config = {**DEFAULTS, "mode": "live_graph_expansion", "backend_mode": "live",
        "lineage": "conv-41", "run_root": str(root),
        "max_total_paid_api_generations": 30, "max_realizer_generations": 30,
        "max_session_attempts_per_session": 3, "max_questioner_generations": 0,
        "max_post_freeze_memory_generations": 0, "max_verifier_generations": 0,
        "max_sessions": 10, "graph_expansion_version": GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version": GRAPH_SCENE_CONTRACT_VERSION,
        "graph_stage_accounting_mode": GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "minimum_accepted_child_tokens": int(outline["target"][
            "minimum_accepted_child_tokens"]),
        "target_child_tokens": int(outline["target"]["child_token_growth"]),
        "challenger_model_identifier": api.model,
        "challenger_temperature": api.temperature,
        "challenger_realizer_max_tokens": api.realizer_max_tokens}
    validate_config(config)
    return config


def _run_expansion_round(*, root: Path, round_index: int,
                         cumulative: list[dict[str, Any]], tokenizer: Any,
                         api: Any, top_manifest: Mapping[str, Any],
                         imported_raw_outline: dict[str, Any] | None = None,
                         imported_outline_attestation: Mapping[str, Any] | None = None,
                         ) -> dict[str, Any]:
    root.mkdir(parents=True)
    prompt = build_outline_planner_prompt(cumulative_turns=cumulative,
                                          round_index=round_index)
    manifest = _round_manifest(root=root, round_index=round_index,
        cumulative=cumulative, outline_prompt=prompt, top_manifest=top_manifest,
        tokenizer=tokenizer, api=api)
    atomic_json(root / "config/run_manifest.json", manifest)
    client = OpenAICompatibleChallengerClient(api)
    if imported_raw_outline is None:
        ledger = RequestLedger(
            root, sha256_file(root / "config/run_manifest.json"), "live")
        budget = ApiBudget(normal_caps={"outline_planner": 1},
                           format_retry_cap=1, physical_cap=2)
        task = run_single_object_task(run_root=root, ledger=ledger,
            backend=LiveSemanticRepairBackend(client, 5000), budget=budget,
            stage="outline_planner",
            logical_request_id=f"outline-round-{round_index:02d}",
            semantic_prompt=prompt, lineage_view=_view(cumulative),
            validator=lambda raw: validate_raw_outline_plan(
                raw, cumulative_turns=cumulative),
            schema_retry_decisions=("rolling_outline_schema_invalid",))
        atomic_json(root / "reports/outline_planner_request.json", {
            key: value for key, value in task.items()
            if key not in {"accepted_raw", "accepted_validation"}})
        if not task["accepted"]:
            return {"round_index": round_index,
                    "decision": task["final_decision"],
                    "outline_task": task, "cumulative_turns": cumulative}
        raw = task["accepted_raw"]
    else:
        raw = dict(imported_raw_outline)
        atomic_json(root / "reports/outline_planner_request.json", {
            "stage": "outline_planner",
            "decision": "imported_raw_outline_accept",
            "physical_request_count": 0,
            "format_retry_used": False,
            "source_attestation": dict(imported_outline_attestation or {}),
        })
    outline = normalize_outline_plan(raw, cumulative_turns=cumulative,
                                     round_index=round_index)
    contracts = compile_graph_scene_contracts(outline, coverage_only=True)
    atomic_json(root / "frozen/raw_outline_plan.json", raw)
    atomic_json(root / "frozen/expansion_outline.json", outline)
    atomic_json(root / "frozen/graph_scene_contracts.json", {
        "version": GRAPH_SCENE_CONTRACT_VERSION, "contracts": contracts,
        "contracts_sha256": graph_contract_sha256(contracts)})
    config = _expansion_config(root, outline, api)
    atomic_json(root / "config/resolved_canary_config.json", config)
    data = {"outline": outline, "parent": {"dialogue": {"turns": cumulative}}}
    orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest,
        config=config, data=data, tokenizer=tokenizer,
        realizer=LiveRealizerBackend(client), questioner=None, solver=None,
        backend_mode="live")
    sessions = orchestrator.realize()
    child, _ = orchestrator.freeze(sessions)
    child_tokens = orchestrator._tokens(child)
    completion = graph_scene_coverage_report(
        outline, contracts, sessions, child_tokens)
    atomic_json(root / "reports/stage_state_accounting.json", completion)
    if not completion["complete"]:
        raise RuntimeError(f"rolling_graph_round_incomplete:{round_index}")
    remapped = remap_round_child_turns(child, round_index=round_index)
    combined = cumulative + remapped
    introduced = [row["name"] for row in outline["allowed_new_characters"]]
    result = {"round_index": round_index, "decision": "round_complete",
        "input_turn_count": len(cumulative), "input_dialogue_tokens": _tokens(
            tokenizer, cumulative), "new_turn_count": len(remapped),
        "new_dialogue_tokens": _tokens(tokenizer, remapped),
        "output_turn_count": len(combined),
        "output_dialogue_tokens": _tokens(tokenizer, combined),
        "new_named_characters": introduced,
        "new_named_character_count": len(introduced),
        "outline_sha256": sha256_file(root / "frozen/expansion_outline.json"),
        "raw_outline_sha256": sha256_file(root / "frozen/raw_outline_plan.json"),
        "completion": completion,
        "stage_accounting_mode": GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "semantic_transition_status": completion[
            "semantic_transition_status"],
        "cumulative_turns": combined}
    atomic_json(root / "frozen/remapped_round_child.json", {
        "round_index": round_index, "turns": remapped})
    atomic_json(root / "frozen/cumulative_dialogue_after_round.json", {
        "round_index": round_index, "turns": combined})
    atomic_json(root / "reports/round_expansion.json", {
        key: value for key, value in result.items() if key != "cumulative_turns"})
    return result


def _questioner_round(*, root: Path, round_index: int,
                      cumulative: list[dict[str, Any]], tokenizer: Any, api: Any,
                      feedback: Mapping[str, Any], feedback_sha: str,
                      snapshot: Mapping[str, Any], project_diff: str,
                      solver_backend: Any | None) -> tuple[dict[str, Any], Any | None]:
    qa_root = root / "qa"
    qa_root.mkdir(parents=True)
    catalog = build_span_catalog(cumulative)
    letters_by_round = (("A", "B", "C"), ("D", "A", "B"),
                        ("C", "D", "A"), ("B", "C", "D"),
                        ("A", "C", "D"))
    letters = letters_by_round[(round_index - 1) % len(letters_by_round)]
    contracts = default_reasoning_contracts(letters)
    prompt = build_questioner_span_ids_only_feedback_prompt(
        frozen_dialogue=cumulative,
        valid_turn_ids=[turn["turn_id"] for turn in cumulative],
        valid_session_ids=sorted({turn["session_id"] for turn in cumulative}),
        assigned_gold_letters=letters, solver_feedback=feedback["feedback"],
        batch_index=round_index - 1, span_catalog=catalog)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    preflight = {"rendered_prompt_tokens": prompt_tokens,
        "max_generation_tokens": api.questioner_max_tokens,
        "total_tokens": prompt_tokens + api.questioner_max_tokens,
        "backend_context_limit": MODEL_CONTEXT_LIMIT,
        "decision": "pass" if prompt_tokens + api.questioner_max_tokens <=
            MODEL_CONTEXT_LIMIT else "reject_context_overflow", "truncated": False}
    atomic_json(qa_root / "frozen/questioner_prompt_preflight.json", preflight)
    if preflight["decision"] != "pass":
        report = {"round_index": round_index,
            "questioner_decision": "questioner_context_overflow",
            "questioner_prompt_preflight": preflight,
            "raw_candidate_count": 0, "minimal_accepted_count": 0,
            "solver_evaluated_count": 0, "final_decisions": {},
            "retain_frontier_count": 0, "items": []}
        atomic_json(qa_root / "reports/round_qa_filter.json", report)
        return report, solver_backend
    context = {"schema_version": "rolling_30k_questioner_context_v1",
        "lineage": "conv-41", "round_index": round_index,
        "combined_dialogue": cumulative}
    artifacts = {"context": context, "catalog": catalog, "contracts": contracts,
        "letters": letters, "prompt": prompt, "preflight": preflight}
    for name, value in (("questioner_smoke_context.json", context),
                        ("questioner_span_catalog.json", catalog),
                        ("questioner_reasoning_contracts.json", contracts)):
        atomic_json(qa_root / "frozen" / name, value)
    atomic_json(qa_root / "raw/prompts/questioner_semantic_prompt.json", {
        "prompt_version": QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        "rendered_prompt_sha256": sha256_bytes(prompt.encode()), "prompt": prompt})
    qconfig = _questioner_config(run_root=qa_root, api=api)
    qmanifest = {**_base_manifest(run_id=f"{root.name}-qa",
        mode="rolling_30k_questioner_round", snapshot=dict(snapshot),
        project_diff=project_diff), "canary_version": CANARY_VERSION,
        "round_index": round_index, "frozen_dialogue_sha256": canonical_sha256(
            cumulative), "span_catalog_sha256": span_catalog_sha256(catalog),
        "reasoning_contract_sha256": reasoning_contract_sha256(contracts),
        "target_gold_letters": list(letters),
        "questioner_prompt_version": QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        "questioner_prompt_sha256": span_ids_only_questioner_template_hash(),
        "rendered_prompt_sha256": sha256_bytes(prompt.encode()),
        "questioner_schema_version": QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
        "questioner_output_profile": SPAN_IDS_ONLY_SCHEMA_PROFILE,
        "format_retry_version": QUESTIONER_FORMAT_RETRY_VERSION,
        "format_retry_suffix_sha256": questioner_format_retry_suffix_hash(),
        "feedback_memory_sha256": feedback_sha,
        "questioner_prompt_preflight": preflight,
        "challenger_model_identifier": api.model,
        "sampling_parameters": {"temperature": api.temperature,
            "max_tokens": api.questioner_max_tokens}}
    atomic_json(qa_root / "config/resolved_canary_config.json", qconfig)
    atomic_json(qa_root / "config/run_manifest.json", qmanifest)
    qresult = _run_questioner(root=qa_root, manifest=qmanifest, config=qconfig,
        artifacts=artifacts, backend=LiveQuestionerBackend(
            OpenAICompatibleChallengerClient(api)), tokenizer=tokenizer,
        feedback_sha=feedback_sha)
    write_post_run_provenance(qa_root, PROJECT, qmanifest)
    candidates = {row["item_index"]: row for row in qresult["dual"].get("items", [])
                  if row.get("solver_eligible") is True and row["item_index"] in
                  qresult["materialized"]}
    if candidates and solver_backend is None:
        solver_backend = VLLMSolverBackend(model_path=SOLVER,
            tokenizer_path=TOKENIZER, max_model_len=MODEL_CONTEXT_LIMIT,
            temperature=0.7, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=0.90)
    items = []
    for admission in qresult["dual"].get("items", []):
        record = {"qa_id": admission.get("qa_id"),
            "item_index": admission.get("item_index"),
            "minimal_decision": (admission.get(SPELL_STYLE_MINIMAL_MODE) or {}).get(
                "decision"), "solver_evaluated": False,
            "final_decision": "minimal_admission_reject"}
        index = admission.get("item_index")
        if index in candidates:
            materialized = qresult["materialized"][index]
            solver_root = qa_root / f"solvers/item_{index:02d}"
            solver_root.mkdir(parents=True)
            sconfig = _solver_config(run_root=solver_root)
            sconfig.update({"mode": "local_solver_rolling_30k",
                            "solver_effective_max_model_len": MODEL_CONTEXT_LIMIT,
                            "max_local_solver_generations": 80})
            smanifest = {**_base_manifest(run_id=f"{root.name}-{materialized['qa_id']}",
                mode="local_solver_rolling_30k", snapshot=dict(snapshot),
                project_diff=project_diff), "canary_version": CANARY_VERSION,
                "round_index": round_index, "qa_id": materialized["qa_id"],
                "solver_model": str(SOLVER),
                "solver_max_model_len": MODEL_CONTEXT_LIMIT,
                "source_questioner_manifest_sha256": sha256_file(
                    qa_root / "config/run_manifest.json")}
            atomic_json(solver_root / "config/resolved_canary_config.json", sconfig)
            atomic_json(solver_root / "config/run_manifest.json", smanifest)
            solver_report = _run_solver(root=solver_root, manifest=smanifest,
                config=sconfig, raw_batch=qresult["raw_batch"], admission=admission,
                materialized=materialized, artifacts=artifacts,
                tokenizer=tokenizer, backend=solver_backend)
            write_post_run_provenance(solver_root, PROJECT, smanifest)
            no_context = solver_report.get("initial_no_context") or {}
            full_context = solver_report.get("full_context") or {}
            record.update({"solver_evaluated": True,
                "solver_root": str(solver_root),
                "no_context_raw_answers": [row.get("raw_response") for row in
                    no_context.get("rollouts") or []],
                "no_context_rewards": no_context.get("verified_rewards"),
                "full_context_raw_answers": [row.get("raw_response") for row in
                    full_context.get("rollouts") or []],
                "full_context_rewards": full_context.get("verified_rewards"),
                "final_decision": solver_report["final_decision"],
                "no_context_permutation_diagnostic": solver_report.get(
                    "no_context_permutation_diagnostic")})
        items.append(record)
    decisions = collections.Counter(row["final_decision"] for row in items)
    report = {"round_index": round_index, "questioner_decision": "completed",
        "questioner_prompt_preflight": preflight,
        "raw_candidate_count": qresult["report"]["raw_candidate_count"],
        "minimal_accepted_count": qresult["report"]["minimal_accepted_count"],
        "solver_evaluated_count": sum(row["solver_evaluated"] for row in items),
        "final_decisions": dict(sorted(decisions.items())),
        "grounding_pass_count": sum(row["solver_evaluated"] and
            row["final_decision"] != "bad_ground" for row in items),
        "retain_frontier_count": decisions["retain_frontier"], "items": items}
    atomic_json(qa_root / "reports/round_qa_filter.json", report)
    return report, solver_backend


def _questioner_round_without_stopping_expansion(
        *, root: Path, round_index: int, cumulative: list[dict[str, Any]],
        tokenizer: Any, api: Any, feedback: Mapping[str, Any],
        feedback_sha: str, snapshot: Mapping[str, Any], project_diff: str,
        solver_backend: Any | None) -> tuple[dict[str, Any], Any | None]:
    """Record an ambiguous transport failure without retrying the QA request.

    A Questioner read timeout must not erase a completed expansion round or stop
    later outline/Realizer rounds. The request ledger remains authoritative about
    whether the physical request was ambiguous; this wrapper does not retry it.
    """
    try:
        return _questioner_round(root=root, round_index=round_index,
            cumulative=cumulative, tokenizer=tokenizer, api=api,
            feedback=feedback, feedback_sha=feedback_sha, snapshot=snapshot,
            project_diff=project_diff, solver_backend=solver_backend)
    except (urllib.error.URLError, TimeoutError, ssl.SSLError) as error:
        report = {"round_index": round_index,
            "questioner_decision": "questioner_transport_error_ambiguous",
            "transport_error_type": type(error).__name__,
            "transport_error": str(error),
            "transport_retry_performed": False,
            "raw_candidate_count": 0, "minimal_accepted_count": 0,
            "solver_evaluated_count": 0, "grounding_pass_count": 0,
            "final_decisions": {}, "retain_frontier_count": 0, "items": []}
        atomic_json(root / "qa/reports/round_qa_filter.json", report)
        return report, solver_backend


def _character_summary(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    speakers = sorted({str(turn["speaker"]) for turn in turns})
    narrative = [name for name in speakers if name not in {"Analyst", "Reviewer"}]
    return {"unique_speaker_count": len(speakers), "unique_speakers": speakers,
        "named_story_character_count": len(narrative),
        "named_story_characters": narrative,
        "non_story_role_speakers": [name for name in speakers
                                    if name in {"Analyst", "Reviewer"}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--feedback-source", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--api-config", type=Path,
        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--max-rounds", type=int, default=MAX_EXPANSION_ROUNDS)
    parser.add_argument("--baseline-scope", choices=("full", "parent_only"),
                        default="full")
    parser.add_argument("--import-raw-outline", type=Path)
    parser.add_argument("--import-completed-round", type=Path)
    parser.add_argument("--import-questioner-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")
    if not 1 <= args.max_rounds <= MAX_EXPANSION_ROUNDS:
        raise ValueError("max_rounds_out_of_range")
    source = _source_attestation(args.source_root.resolve())
    feedback = read_json(args.feedback_source)
    feedback_sha = sha256_file(args.feedback_source)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    api = replace(load_challenger_config(args.api_config), model=CHALLENGER_MODEL,
                  temperature=0.25, enable_thinking=False, timeout_seconds=600)
    solver_config = read_json(SOLVER / "config.json")
    if int(solver_config.get("max_position_embeddings") or 0) < MODEL_CONTEXT_LIMIT:
        raise RuntimeError("solver_native_context_too_short")
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    baseline = remap_baseline_turns(select_baseline_turns(
        source, args.baseline_scope))
    baseline_tokens = _tokens(tokenizer, baseline)
    imported_raw_outline = None
    imported_outline_attestation = None
    if args.import_raw_outline:
        imported_raw_outline, imported_outline_attestation = (
            load_imported_raw_outline(args.import_raw_outline, baseline))
    top_manifest = {"run_id": args.run_root.name, "mode": CANARY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff,
        "source_snapshot_manifest_path": snapshot["source_snapshot_manifest_path"],
        "source_snapshot_manifest_sha256": snapshot["source_snapshot_manifest_sha256"],
        "orchestrator_source_sha256": sha256_file(MODULE_ROOT / "orchestrator.py"),
        "validation_source_sha256": sha256_file(MODULE_ROOT / "validation.py"),
        "ledger_source_sha256": sha256_file(MODULE_ROOT / "ledger.py"),
        "run_canary_source_sha256": sha256_file(MODULE_ROOT / "run_canary.py"),
        "mcq_parser_source_sha256": sha256_file(MODULE_ROOT / "core.py"),
        "questioner_prompt_builder_source_sha256": sha256_file(
            MODULE_ROOT / "prompts.py"),
        "questioner_contract_source_sha256": sha256_file(
            MODULE_ROOT / "questioner_contract.py"),
        "live_backend_source_sha256": sha256_file(MODULE_ROOT / "live_backends.py"),
        "vllm_backend_source_sha256": sha256_file(MODULE_ROOT / "vllm_backends.py"),
        "source_attestation": {key: value for key, value in source.items()
                               if key != "dialogue"},
        "feedback_source": str(args.feedback_source.resolve()),
        "feedback_source_sha256": feedback_sha,
        "challenger_model_identifier": api.model,
        "baseline_scope": args.baseline_scope,
        "baseline_parent_turn_count": len(source["dialogue"]["parent_turns"]),
        "baseline_child_turn_count": (len(source["dialogue"]["child_turns"])
                                      if args.baseline_scope == "full" else 0),
        "imported_raw_outline_attestation": imported_outline_attestation,
        "target_dialogue_tokens": TARGET_DIALOGUE_TOKENS,
        "maximum_expansion_rounds": args.max_rounds,
        "model_context_limit": MODEL_CONTEXT_LIMIT,
        "solver_model": str(SOLVER),
        "solver_native_max_position_embeddings": solver_config[
            "max_position_embeddings"],
        "ordinary_questioner_prompt_version":
            QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        "ordinary_questioner_prompt_sha256":
            span_ids_only_questioner_template_hash(),
        "rolling_outline_source_sha256": sha256_file(MODULE_ROOT /
            "rolling_outline_30k.py"),
        "rolling_runner_source_sha256": sha256_file(Path(__file__))}
    imported = None
    imported_qa = None
    if args.import_completed_round:
        imported = _attest_imported_round(args.import_completed_round.resolve(),
                                          baseline, tokenizer)
        top_manifest["imported_completed_round_attestation"] = {
            key: value for key, value in imported.items()
            if key not in {"report", "turns"}}
    if args.import_questioner_report:
        if imported is None:
            raise ValueError("questioner_report_requires_imported_round")
        imported_qa = _attest_imported_questioner_report(
            args.import_questioner_report.resolve())
        top_manifest["imported_questioner_report_attestation"] = {
            key: value for key, value in imported_qa.items() if key != "report"}
    atomic_json(args.run_root / "config/run_manifest.json", top_manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "canary_version": CANARY_VERSION, "target_dialogue_tokens":
        TARGET_DIALOGUE_TOKENS, "max_rounds": args.max_rounds,
        "baseline_scope": args.baseline_scope,
        "graph_stage_accounting_mode": GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "imported_raw_outline": imported_outline_attestation is not None,
        "challenger_model": api.model, "questioner_context_limit":
        MODEL_CONTEXT_LIMIT, "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "questioner_semantics_changed": False, "historical_runs_modified": False})
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        key: value for key, value in source.items() if key != "dialogue"})
    atomic_json(args.run_root / "frozen/baseline_dialogue.json", {
        "turns": baseline, "dialogue_tokens": baseline_tokens,
        "character_summary": _character_summary(baseline)})
    if args.dry_run:
        prompt = build_outline_planner_prompt(cumulative_turns=baseline,
                                              round_index=1)
        preflight = {"decision": "pass" if len(tokenizer.encode(
            prompt, add_special_tokens=False)) + 5000 <= MODEL_CONTEXT_LIMIT else
            "reject_context_overflow", "prompt_tokens": len(tokenizer.encode(
                prompt, add_special_tokens=False)), "max_tokens": 5000,
            "context_limit": MODEL_CONTEXT_LIMIT}
        atomic_json(args.run_root / "reports/dry_run_preflight.json", preflight)
        print(json.dumps(preflight, indent=2))
        return 0
    cumulative = baseline
    rounds = []
    solver_backend = None
    first_round = 1
    if imported is not None:
        imported_round_index = int(imported["report"]["round_index"])
        cumulative = imported["turns"]
        imported_root = args.run_root / f"rounds/round_{imported_round_index:02d}"
        imported_root.mkdir(parents=True)
        atomic_json(imported_root / "frozen/imported_source_attestation.json", {
            key: value for key, value in imported.items()
            if key not in {"report", "turns"}})
        if imported_qa is not None:
            if imported_qa["round_index"] != imported_round_index:
                raise RuntimeError("imported_questioner_round_mismatch")
            qa = imported_qa["report"]
            atomic_json(imported_root / "qa/frozen/imported_report_attestation.json",
                {key: value for key, value in imported_qa.items()
                 if key != "report"})
            atomic_json(imported_root / "qa/reports/round_qa_filter.json", qa)
        else:
            qa, solver_backend = _questioner_round_without_stopping_expansion(
                root=imported_root, round_index=imported_round_index,
                cumulative=cumulative,
                tokenizer=tokenizer, api=api, feedback=feedback,
                feedback_sha=feedback_sha, snapshot=snapshot,
                project_diff=project_diff, solver_backend=solver_backend)
        rounds.append({"expansion": {**imported["report"],
            "imported_from": imported["source_round_root"]}, "qa": qa,
            "character_summary": _character_summary(cumulative)})
        atomic_json(args.run_root /
                    f"frozen/cumulative_round_{imported_round_index:02d}.json",
                    {"turns": cumulative,
                     "dialogue_tokens": _tokens(tokenizer, cumulative)})
        first_round = imported_round_index + 1
    for round_index in range(first_round, args.max_rounds + 1):
        round_root = args.run_root / f"rounds/round_{round_index:02d}"
        try:
            expansion = _run_expansion_round(root=round_root,
                round_index=round_index, cumulative=cumulative,
                tokenizer=tokenizer, api=api, top_manifest=top_manifest,
                imported_raw_outline=(imported_raw_outline if round_index == 1
                                      else None),
                imported_outline_attestation=(
                    imported_outline_attestation if round_index == 1 else None))
        except (urllib.error.URLError, TimeoutError, ssl.SSLError,
                http.client.RemoteDisconnected) as error:
            expansion = {"round_index": round_index,
                "decision": "expansion_transport_error_ambiguous",
                "transport_error_type": type(error).__name__,
                "transport_error": str(error),
                "transport_retry_performed": False,
                "input_turn_count": len(cumulative),
                "input_dialogue_tokens": _tokens(tokenizer, cumulative),
                "cumulative_turns": cumulative}
            atomic_json(round_root / "reports/round_expansion.json", {
                key: value for key, value in expansion.items()
                if key != "cumulative_turns"})
        public_expansion = {key: value for key, value in expansion.items()
                            if key != "cumulative_turns"}
        if expansion["decision"] != "round_complete":
            rounds.append({"expansion": public_expansion, "qa": None})
            break
        cumulative = expansion["cumulative_turns"]
        qa, solver_backend = _questioner_round_without_stopping_expansion(
            root=round_root,
            round_index=round_index, cumulative=cumulative, tokenizer=tokenizer,
            api=api, feedback=feedback, feedback_sha=feedback_sha,
            snapshot=snapshot, project_diff=project_diff,
            solver_backend=solver_backend)
        rounds.append({"expansion": public_expansion, "qa": qa,
                       "character_summary": _character_summary(cumulative)})
        atomic_json(args.run_root / f"frozen/cumulative_round_{round_index:02d}.json",
                    {"turns": cumulative,
                     "dialogue_tokens": _tokens(tokenizer, cumulative)})
        if _tokens(tokenizer, cumulative) >= TARGET_DIALOGUE_TOKENS:
            break
    final_tokens = _tokens(tokenizer, cumulative)
    decisions = collections.Counter()
    totals = collections.Counter()
    for row in rounds:
        qa = row.get("qa") or {}
        totals["raw"] += int(qa.get("raw_candidate_count") or 0)
        totals["minimal"] += int(qa.get("minimal_accepted_count") or 0)
        totals["solver"] += int(qa.get("solver_evaluated_count") or 0)
        totals["grounding_pass"] += int(qa.get("grounding_pass_count") or 0)
        totals["frontier"] += int(qa.get("retain_frontier_count") or 0)
        decisions.update(qa.get("final_decisions") or {})
    result = {"canary_version": CANARY_VERSION,
        "route": "outline_graph_api_expansion_then_unchanged_questioner_solver",
        "challenger_model": api.model,
        "baseline_turn_count": len(baseline), "baseline_dialogue_tokens": baseline_tokens,
        "baseline_scope": args.baseline_scope,
        "target_dialogue_tokens": TARGET_DIALOGUE_TOKENS,
        "target_reached": final_tokens >= TARGET_DIALOGUE_TOKENS,
        "final_turn_count": len(cumulative), "final_dialogue_tokens": final_tokens,
        "completed_round_count": sum(row["expansion"]["decision"] ==
                                     "round_complete" for row in rounds),
        "rounds": rounds, "final_character_summary": _character_summary(cumulative),
        "qa_totals": dict(totals), "final_decisions": dict(sorted(decisions.items())),
        "minimal_accept_rate": totals["minimal"] / totals["raw"] if totals["raw"] else 0,
        "grounding_pass_rate": totals["grounding_pass"] / totals["solver"]
            if totals["solver"] else 0,
        "retain_frontier_rate": totals["frontier"] / totals["solver"]
            if totals["solver"] else 0,
        "questioner_semantics_changed": False,
        "reference_material_used": False,
        "dependency_planner_used": False, "semantic_repair_used": False,
        "graph_stage_accounting_mode": GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "semantic_transition_completion_claimed": False,
        "distractor_stage_used": False, "verifier_generations": 0,
        "training_runs": 0}
    atomic_json(args.run_root / "frozen/final_dialogue.json", {"turns": cumulative})
    atomic_json(args.run_root / "reports/rolling_30k_canary.json", result)
    api_usage = _usage(args.run_root)
    atomic_json(args.run_root / "reports/api_usage.json", api_usage)
    provenance = write_post_run_provenance(args.run_root, PROJECT, top_manifest)
    print(json.dumps({"run_root": str(args.run_root), "summary": result,
                      "api_usage": api_usage, "provenance": provenance},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
