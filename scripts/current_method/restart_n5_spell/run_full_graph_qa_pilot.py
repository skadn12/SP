from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .cross_dialogue_calibration import freeze_parent_dialogue
from .graph_expansion import (compile_graph_scene_contracts,
                              graph_scene_coverage_report)
from .input_adapter import load_lineage, read_json
from .live_backends import (LiveQuestionerBackend, OpenAICompatibleChallengerClient,
                            load_challenger_config)
from .minimal_admission import SPAN_IDS_ONLY_SCHEMA_PROFILE, SPELL_STYLE_MINIMAL_MODE
from .prompts import (QUESTIONER_FORMAT_RETRY_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
                      build_questioner_span_ids_only_feedback_prompt,
                      questioner_format_retry_suffix_hash,
                      span_ids_only_questioner_template_hash)
from .questioner_contract import (build_span_catalog, default_reasoning_contracts,
                                  reasoning_contract_sha256, span_catalog_sha256)
from .run_canary import (create_source_snapshot, git_value, project_diff_sha256,
                         write_post_run_provenance)
from .run_cross_dialogue_calibration import (_base_manifest, _questioner_config,
                                              _run_questioner, _run_solver,
                                              _solver_config)
from .validation import validate_config
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
PILOT_VERSION = "full_graph_qa_complexity_pilot_v1"
CHALLENGER_MODEL = "Qwen/Qwen3-32B"


def attest_graph_source(source: Path, data: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    manifest_path = source / "config/run_manifest.json"
    provenance_path = source / "reports/post_run_provenance.json"
    index_path = source / "accepted/index.json"
    contracts_path = source / "frozen/graph_scene_contracts.json"
    for path in (manifest_path, provenance_path, index_path, contracts_path):
        if not path.exists():
            raise RuntimeError(f"graph_source_missing:{path.name}")
    manifest, provenance, index = map(read_json,
        (manifest_path, provenance_path, index_path))
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("graph_source_not_authoritative")
    manifest_sha = sha256_file(manifest_path)
    if (provenance.get("run_manifest_sha256") != manifest_sha or
            index.get("run_manifest_sha256") != manifest_sha or
            index.get("run_id") != manifest.get("run_id")):
        raise RuntimeError("graph_source_identity_mismatch")
    sessions = []
    expected_session_hashes = index.get("session_sha256") or {}
    for position in range(1, 11):
        path = source / f"accepted/session_{position:02d}.json"
        if expected_session_hashes.get(path.name) != sha256_file(path):
            raise RuntimeError("graph_source_session_sha_mismatch")
        sessions.append(read_json(path))
    parent_turns = freeze_parent_dialogue(
        str(data["outline"]["source_parent"]["lineage_group"]), data["parent"]
    )["context"]["combined_dialogue"]
    child_turns = [turn for session in sessions for turn in session["dialogue_turns"]]
    child_text = "\n".join(f"[{turn['turn_id']}] {turn['speaker']}: {turn['text']}"
                           for turn in child_turns)
    child_tokens = len(tokenizer.encode(child_text, add_special_tokens=False))
    contracts = compile_graph_scene_contracts(data["outline"], coverage_only=True)
    completion = graph_scene_coverage_report(
        data["outline"], contracts, sessions, child_tokens)
    if not completion["complete"]:
        raise RuntimeError("graph_source_completion_failed")
    if manifest.get("parent_dialogue_sha256") != data["anchors"]["dialogue_hash"]:
        raise RuntimeError("graph_source_parent_mismatch")
    return {"source_root":str(source), "source_run_id":manifest["run_id"],
        "source_manifest_sha256":manifest_sha,
        "source_provenance_sha256":sha256_file(provenance_path),
        "source_accepted_index_sha256":sha256_file(index_path),
        "source_graph_contracts_sha256":sha256_file(contracts_path),
        "source_session_sha256":expected_session_hashes,
        "parent_dialogue_sha256":data["anchors"]["dialogue_hash"],
        "child_dialogue_tokens":child_tokens, "completion":completion,
        "sessions":sessions, "parent_turns":parent_turns, "child_turns":child_turns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Questioner/Solver pilot on a full Graph child")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-graph-root", type=Path, required=True)
    parser.add_argument("--lineage", default="conv-41")
    parser.add_argument("--api-config", type=Path,
        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--feedback-source", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "spell_cross_dialogue_calibration_5x3_v1/frozen/calibration_feedback_memory.json"))
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"pilot root already exists: {args.run_root}")
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    legacy = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
    tokenizer_path = Path("/root/autodl-tmp/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    data = load_lineage(args.input_root, args.lineage, legacy)
    source = attest_graph_source(args.source_graph_root.resolve(), data, tokenizer)
    source_public = {key:value for key,value in source.items()
                     if key not in {"sessions", "parent_turns", "child_turns"}}
    atomic_json(args.run_root / "frozen/source_graph_attestation.json", source_public)
    feedback = read_json(args.feedback_source)
    feedback_sha = sha256_file(args.feedback_source)
    atomic_json(args.run_root / "frozen/calibration_feedback_memory.json", feedback)
    context = {"schema_version":"full_graph_questioner_context_v1",
        "questioner_smoke_only":True, "not_full_canary_context":False,
        "lineage":args.lineage, "parent_turns":source["parent_turns"],
        "accepted_child_sessions":source["sessions"],
        "combined_dialogue":source["parent_turns"] + source["child_turns"],
        "source_graph_root":source["source_root"]}
    catalog = build_span_catalog(context["combined_dialogue"])
    letters = ("A", "B", "C")
    contracts = default_reasoning_contracts(letters)
    prompt = build_questioner_span_ids_only_feedback_prompt(
        frozen_dialogue=context["combined_dialogue"],
        valid_turn_ids=[turn["turn_id"] for turn in context["combined_dialogue"]],
        valid_session_ids=sorted({turn["session_id"] for turn in context["combined_dialogue"]}),
        assigned_gold_letters=letters, solver_feedback=feedback["feedback"],
        batch_index=0, span_catalog=catalog)
    api = replace(load_challenger_config(args.api_config),
                  model=CHALLENGER_MODEL, temperature=0.25,
                  enable_thinking=False, timeout_seconds=600)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    preflight = {"rendered_prompt_tokens":prompt_tokens,
        "max_generation_tokens":api.questioner_max_tokens,
        "total_tokens":prompt_tokens + api.questioner_max_tokens,
        "backend_context_limit":40960,
        "decision":"pass" if prompt_tokens + api.questioner_max_tokens <= 40960
                   else "reject_context_overflow", "truncated":False}
    if preflight["decision"] != "pass":
        raise RuntimeError(f"full_graph_questioner_context_overflow:{preflight}")
    questioner_root = args.run_root / "questioner"
    questioner_root.mkdir()
    artifacts = {"context":context, "catalog":catalog, "contracts":contracts,
        "letters":letters, "prompt":prompt, "preflight":preflight,
        "source_attestation":source_public}
    for name, value in (("questioner_smoke_context.json", context),
                        ("questioner_span_catalog.json", catalog),
                        ("questioner_reasoning_contracts.json", contracts),
                        ("questioner_prompt_preflight.json", preflight),
                        ("source_graph_attestation.json", source_public)):
        atomic_json(questioner_root / "frozen" / name, value)
    atomic_json(questioner_root / "raw/prompts/questioner_semantic_prompt.json",
        {"prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
         "rendered_prompt_sha256":sha256_bytes(prompt.encode()), "prompt":prompt})
    questioner_config = _questioner_config(run_root=questioner_root, api=api)
    validate_config(questioner_config)
    base = _base_manifest(run_id=f"{args.run_root.name}-questioner",
        mode="full_graph_questioner_pilot", snapshot=snapshot, project_diff=project_diff)
    questioner_manifest = {**base, "pilot_version":PILOT_VERSION,
        "lineage":args.lineage, "source_graph_attestation":source_public,
        "source_graph_attestation_sha256":sha256_file(
            args.run_root / "frozen/source_graph_attestation.json"),
        "frozen_dialogue_sha256":sha256_file(
            questioner_root / "frozen/questioner_smoke_context.json"),
        "span_catalog_sha256":span_catalog_sha256(catalog),
        "reasoning_contract_sha256":reasoning_contract_sha256(contracts),
        "target_gold_letters":list(letters),
        "questioner_prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        "questioner_prompt_sha256":span_ids_only_questioner_template_hash(),
        "rendered_prompt_sha256":sha256_bytes(prompt.encode()),
        "questioner_schema_version":QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
        "questioner_output_profile":SPAN_IDS_ONLY_SCHEMA_PROFILE,
        "format_retry_version":QUESTIONER_FORMAT_RETRY_VERSION,
        "format_retry_suffix_sha256":questioner_format_retry_suffix_hash(),
        "feedback_memory_sha256":feedback_sha,
        "questioner_prompt_preflight":preflight,
        "challenger_model_identifier":api.model,
        "sampling_parameters":{"temperature":api.temperature,
                               "max_tokens":api.questioner_max_tokens},
        "budget":{"normal_batches":1, "maximum_format_retries":1,
                  "maximum_paid_requests":2, "realizer":0, "verifier":0,
                  "training":0},
        "full_graph_qa_pilot_source_sha256":sha256_file(Path(__file__))}
    atomic_json(questioner_root / "config/resolved_canary_config.json", questioner_config)
    atomic_json(questioner_root / "config/run_manifest.json", questioner_manifest)
    questioner_result = _run_questioner(root=questioner_root,
        manifest=questioner_manifest, config=questioner_config, artifacts=artifacts,
        backend=LiveQuestionerBackend(OpenAICompatibleChallengerClient(api)),
        tokenizer=tokenizer, feedback_sha=feedback_sha)
    write_post_run_provenance(questioner_root, PROJECT, questioner_manifest)
    candidates = [admission for admission in questioner_result["dual"].get("items", [])
                  if admission.get("solver_eligible") is True and admission["item_index"]
                  in questioner_result["materialized"]]
    backend = None
    if candidates:
        backend = VLLMSolverBackend(
            model_path=legacy / "accepted_models/solver_base_restart_stage2_merged",
            tokenizer_path=tokenizer_path, max_model_len=16384,
            temperature=0.7, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=0.80)
    items = []
    for admission in questioner_result["dual"].get("items", []):
        record = {"qa_id":admission.get("qa_id"),
            "item_index":admission.get("item_index"),
            "strict_v7":admission.get("strict_v7"),
            SPELL_STYLE_MINIMAL_MODE:{key:value for key,value in
                (admission.get(SPELL_STYLE_MINIMAL_MODE) or {}).items()
                if key != "materialized_qa"}, "solver_evaluated":False,
            "final_decision":"minimal_admission_reject"}
        if admission in candidates:
            qa = questioner_result["materialized"][admission["item_index"]]
            solver_root = args.run_root / f"solvers/item_{admission['item_index']:02d}"
            solver_root.mkdir(parents=True)
            solver_config = _solver_config(run_root=solver_root)
            solver_manifest = {**_base_manifest(run_id=f"{args.run_root.name}-{qa['qa_id']}",
                    mode="local_solver_full_graph_pilot", snapshot=snapshot,
                    project_diff=project_diff), "pilot_version":PILOT_VERSION,
                "lineage":args.lineage, "qa_id":qa["qa_id"],
                "source_questioner_root":str(questioner_root),
                "source_questioner_manifest_sha256":sha256_file(
                    questioner_root / "config/run_manifest.json"),
                "source_graph_manifest_sha256":source["source_manifest_sha256"],
                "materialized_qa_sha256":sha256_bytes(canonical_json(qa).encode()),
                "frozen_dialogue_sha256":sha256_file(
                    questioner_root / "frozen/questioner_smoke_context.json"),
                "span_catalog_sha256":span_catalog_sha256(catalog),
                "solver_max_model_len":16384,
                "vllm_backend_source_sha256":sha256_file(MODULE_ROOT / "vllm_backends.py"),
                "solver_protocol":{"enable_thinking":False, "temperature":0.7,
                    "top_p":0.9, "max_tokens":32, "parser":"boxed_letter_v1",
                    "max_parse_retries_per_target":1}}
            atomic_json(solver_root / "config/resolved_canary_config.json", solver_config)
            atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
            solver_report = _run_solver(root=solver_root, manifest=solver_manifest,
                config=solver_config, raw_batch=questioner_result["raw_batch"],
                admission=admission, materialized=qa, artifacts=artifacts,
                tokenizer=tokenizer, backend=backend)
            write_post_run_provenance(solver_root, PROJECT, solver_manifest)
            record.update({"solver_evaluated":True, "solver_root":str(solver_root),
                "final_decision":solver_report["final_decision"],
                "no_context_permutation_diagnostic":solver_report[
                    "no_context_permutation_diagnostic"],
                "answer_span_count":len(qa.get("evidence_span_ids") or []),
                "answer_turn_count":len(qa.get("evidence_turn_ids") or []),
                "answer_session_count":len(qa.get("evidence_session_ids") or [])})
        items.append(record)
    summary = {"raw_candidate_count":questioner_result["report"]["raw_candidate_count"],
        "minimal_accepted_count":questioner_result["report"]["minimal_accepted_count"],
        "solver_evaluated_count":sum(item["solver_evaluated"] for item in items),
        "final_decisions":{}, "retain_frontier_count":0}
    for item in items:
        decision = item["final_decision"]
        summary["final_decisions"][decision] = summary["final_decisions"].get(decision, 0) + 1
    summary["retain_frontier_count"] = summary["final_decisions"].get("retain_frontier", 0)
    report = {"pilot_version":PILOT_VERSION, "source_graph_attestation":source_public,
        "questioner_prompt_preflight":preflight, "summary":summary, "items":items,
        "paid_questioner_generations":questioner_result["report"]["paid_api_generations"],
        "realizer_generations":0, "verifier_generations":0, "training_runs":0}
    atomic_json(args.run_root / "reports/full_graph_qa_pilot.json", report)
    top_manifest = {**_base_manifest(run_id=args.run_root.name,
        mode="full_graph_qa_complexity_pilot", snapshot=snapshot,
        project_diff=project_diff), "pilot_version":PILOT_VERSION,
        "source_graph_attestation":source_public,
        "source_graph_attestation_sha256":sha256_file(
            args.run_root / "frozen/source_graph_attestation.json"),
        "questioner_manifest_sha256":sha256_file(
            questioner_root / "config/run_manifest.json"),
        "report_sha256":sha256_file(args.run_root / "reports/full_graph_qa_pilot.json"),
        "full_graph_qa_pilot_source_sha256":sha256_file(Path(__file__)),
        "vllm_backend_source_sha256":sha256_file(MODULE_ROOT / "vllm_backends.py")}
    atomic_json(args.run_root / "config/run_manifest.json", top_manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode":"full_graph_qa_complexity_pilot", "source_dialogues":1,
        "questioner_batches":1, "mcqs_per_batch":3,
        "maximum_paid_questioner_requests":2, "realizer_generations":0,
        "verifier_generations":0, "training_runs":0,
        "solver_effective_max_model_len":16384})
    provenance = write_post_run_provenance(args.run_root, PROJECT, top_manifest)
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("full graph QA pilot provenance mixed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
