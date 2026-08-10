from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .evidence_view_ablation import (
    BALANCED_VIEW, CHILD_ONLY_VIEW, EVIDENCE_VIEW_ABLATION_VERSION,
    EVIDENCE_VIEW_MODES, build_questioner_evidence_view,
    selected_evidence_diagnostics,
)
from .graph_expansion import compile_graph_scene_contracts
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
from .run_canary import (create_source_snapshot, project_diff_sha256,
                         write_post_run_provenance)
from .run_cross_dialogue_calibration import (_base_manifest, _questioner_config,
                                              _run_questioner, _run_solver,
                                              _solver_config)
from .run_full_graph_qa_pilot import attest_graph_source
from .validation import validate_config
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_FEEDBACK = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_cross_dialogue_calibration_5x3_v1/frozen/calibration_feedback_memory.json")


def _freeze_inputs(*, root: Path, view: dict[str, Any], lineage: str,
                   source_attestation: dict[str, Any], scene_contracts: list[dict[str, Any]],
                   feedback: dict[str, Any], tokenizer: Any, api: Any) -> dict[str, Any]:
    turns = view["combined_dialogue"]
    context = {"schema_version":"questioner_evidence_view_context_v1",
        "questioner_smoke_only":True, "lineage":lineage,
        "view_mode":view["view_mode"], "view_sha256":view["view_sha256"],
        "combined_dialogue":turns}
    letters = ("A", "B", "C")
    catalog = build_span_catalog(turns)
    contracts = default_reasoning_contracts(letters)
    prompt = build_questioner_span_ids_only_feedback_prompt(
        frozen_dialogue=turns, valid_turn_ids=[turn["turn_id"] for turn in turns],
        valid_session_ids=sorted({turn["session_id"] for turn in turns}),
        assigned_gold_letters=letters, solver_feedback=feedback["feedback"],
        batch_index=0, span_catalog=catalog)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    total = prompt_tokens + api.questioner_max_tokens
    preflight = {"rendered_prompt_tokens":prompt_tokens,
        "max_generation_tokens":api.questioner_max_tokens, "total_tokens":total,
        "backend_context_limit":32768,
        "decision":"pass" if total <= 32768 else "reject_context_overflow",
        "truncated":False}
    if preflight["decision"] != "pass":
        raise RuntimeError(f"evidence_view_questioner_context_overflow:{preflight}")
    for name, value in (("questioner_evidence_view.json", view),
                        ("questioner_smoke_context.json", context),
                        ("questioner_span_catalog.json", catalog),
                        ("questioner_reasoning_contracts.json", contracts),
                        ("questioner_prompt_preflight.json", preflight),
                        ("source_graph_attestation.json", source_attestation),
                        ("graph_scene_contracts.json", {"contracts":scene_contracts})):
        atomic_json(root / "frozen" / name, value)
    atomic_json(root / "raw/prompts/questioner_semantic_prompt.json",
        {"prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
         "rendered_prompt_sha256":sha256_bytes(prompt.encode()), "prompt":prompt})
    return {"context":context, "catalog":catalog, "contracts":contracts,
        "letters":letters, "prompt":prompt, "preflight":preflight,
        "source_attestation":source_attestation, "scene_contracts":scene_contracts,
        "view":view}


def _manifest(*, run_id: str, mode: str, snapshot: dict[str, Any], project_diff: str,
              source: dict[str, Any], artifacts: dict[str, Any], feedback_sha: str,
              api: Any) -> dict[str, Any]:
    base = _base_manifest(run_id=run_id, mode=mode, snapshot=snapshot,
                          project_diff=project_diff)
    return {**base, "evidence_view_ablation_version":EVIDENCE_VIEW_ABLATION_VERSION,
        "evidence_view_mode":artifacts["view"]["view_mode"],
        "evidence_view_sha256":artifacts["view"]["view_sha256"],
        "source_graph_manifest_sha256":source["source_manifest_sha256"],
        "source_graph_provenance_sha256":source["source_provenance_sha256"],
        "source_graph_accepted_index_sha256":source["source_accepted_index_sha256"],
        "source_graph_contracts_sha256":source["source_graph_contracts_sha256"],
        "frozen_dialogue_sha256":sha256_bytes(canonical_json(
            artifacts["context"]).encode()),
        "span_catalog_sha256":span_catalog_sha256(artifacts["catalog"]),
        "reasoning_contract_sha256":reasoning_contract_sha256(artifacts["contracts"]),
        "target_gold_letters":list(artifacts["letters"]),
        "questioner_prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        "questioner_prompt_sha256":span_ids_only_questioner_template_hash(),
        "rendered_prompt_sha256":sha256_bytes(artifacts["prompt"].encode()),
        "questioner_schema_version":QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
        "questioner_output_profile":SPAN_IDS_ONLY_SCHEMA_PROFILE,
        "format_retry_version":QUESTIONER_FORMAT_RETRY_VERSION,
        "format_retry_suffix_sha256":questioner_format_retry_suffix_hash(),
        "feedback_memory_sha256":feedback_sha,
        "questioner_prompt_preflight":artifacts["preflight"],
        "challenger_model_identifier":api.model,
        "sampling_parameters":{"temperature":api.temperature,
                               "max_tokens":api.questioner_max_tokens},
        "evidence_view_ablation_source_sha256":sha256_file(
            MODULE_ROOT / "evidence_view_ablation.py"),
        "evidence_view_ablation_runner_source_sha256":sha256_file(Path(__file__)),
        "vllm_backend_source_sha256":sha256_file(MODULE_ROOT / "vllm_backends.py"),
        "budget":{"normal_batches":1, "maximum_format_retries":1,
                  "maximum_paid_requests":2, "realizer":0, "verifier":0,
                  "training":0}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Child-only/balanced Questioner view ablation")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-graph-root", type=Path, required=True)
    parser.add_argument("--view-mode", choices=EVIDENCE_VIEW_MODES, required=True)
    parser.add_argument("--lineage", default="conv-41")
    parser.add_argument("--api-config", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--feedback-source", type=Path, default=DEFAULT_FEEDBACK)
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"evidence view root already exists: {args.run_root}")
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    tokenizer_path = Path("/root/autodl-tmp/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    legacy = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
    data = load_lineage(args.input_root, args.lineage, legacy)
    source = attest_graph_source(args.source_graph_root.resolve(), data, tokenizer)
    source_public = {key:value for key,value in source.items()
                     if key not in {"sessions", "parent_turns", "child_turns"}}
    scene_contracts = compile_graph_scene_contracts(data["outline"])
    view = build_questioner_evidence_view(mode=args.view_mode,
        parent_turns=source["parent_turns"], child_turns=source["child_turns"])
    atomic_json(args.run_root / "frozen/source_graph_attestation.json", source_public)
    atomic_json(args.run_root / "frozen/questioner_evidence_view.json", view)
    feedback = read_json(args.feedback_source)
    feedback_sha = sha256_file(args.feedback_source)
    atomic_json(args.run_root / "frozen/calibration_feedback_memory.json", feedback)
    api = load_challenger_config(args.api_config)
    questioner_root = args.run_root / "questioner"
    questioner_root.mkdir()
    artifacts = _freeze_inputs(root=questioner_root, view=view, lineage=args.lineage,
        source_attestation=source_public, scene_contracts=scene_contracts,
        feedback=feedback, tokenizer=tokenizer, api=api)
    questioner_config = _questioner_config(run_root=questioner_root, api=api)
    validate_config(questioner_config)
    questioner_manifest = _manifest(run_id=f"{args.run_root.name}-questioner",
        mode="questioner_evidence_view_ablation", snapshot=snapshot,
        project_diff=project_diff, source=source, artifacts=artifacts,
        feedback_sha=feedback_sha, api=api)
    atomic_json(questioner_root / "config/resolved_canary_config.json", questioner_config)
    atomic_json(questioner_root / "config/run_manifest.json", questioner_manifest)
    questioner_result = _run_questioner(root=questioner_root,
        manifest=questioner_manifest, config=questioner_config, artifacts=artifacts,
        backend=LiveQuestionerBackend(OpenAICompatibleChallengerClient(api)),
        tokenizer=tokenizer, feedback_sha=feedback_sha)
    questioner_provenance = write_post_run_provenance(
        questioner_root, PROJECT, questioner_manifest)
    if not questioner_provenance["authoritative_release_provenance"]:
        raise RuntimeError("evidence view questioner provenance mixed")
    eligible_indexes = {item["item_index"] for item in questioner_result["dual"].get(
        "items", []) if item.get("solver_eligible") is True and
        item["item_index"] in questioner_result["materialized"]}
    backend = None
    if eligible_indexes:
        backend = VLLMSolverBackend(
            model_path=legacy / "accepted_models/solver_base_restart_stage2_merged",
            tokenizer_path=tokenizer_path, max_model_len=16384,
            temperature=0.7, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=0.80)
    items = []
    for admission in questioner_result["dual"].get("items", []):
        item_index = admission["item_index"]
        record = {"qa_id":admission.get("qa_id"), "item_index":item_index,
            "strict_v7":admission.get("strict_v7"),
            SPELL_STYLE_MINIMAL_MODE:{key:value for key,value in
                (admission.get(SPELL_STYLE_MINIMAL_MODE) or {}).items()
                if key != "materialized_qa"}, "solver_evaluated":False,
            "final_decision":"minimal_admission_reject"}
        raw = questioner_result["raw_batch"][item_index] if item_index < len(
            questioner_result["raw_batch"]) else {}
        span_ids = raw.get("answer_support_span_ids") if isinstance(raw, dict) else []
        if isinstance(span_ids, list) and all(isinstance(value, str) for value in span_ids):
            try:
                record["selection_diagnostics"] = selected_evidence_diagnostics(
                    span_ids=span_ids, catalog=artifacts["catalog"],
                    scene_contracts=scene_contracts)
            except ValueError as error:
                record["selection_diagnostics"] = {"diagnostic_error":str(error)}
        if item_index in eligible_indexes:
            qa = questioner_result["materialized"][item_index]
            solver_root = args.run_root / f"solvers/item_{item_index:02d}"
            solver_root.mkdir(parents=True)
            solver_config = _solver_config(run_root=solver_root)
            solver_manifest = {**_manifest(
                    run_id=f"{args.run_root.name}-{qa['qa_id']}",
                    mode="local_solver_evidence_view_ablation", snapshot=snapshot,
                    project_diff=project_diff, source=source, artifacts=artifacts,
                    feedback_sha=feedback_sha, api=api),
                "qa_id":qa["qa_id"], "source_questioner_root":str(questioner_root),
                "source_questioner_manifest_sha256":sha256_file(
                    questioner_root / "config/run_manifest.json"),
                "materialized_qa_sha256":sha256_bytes(canonical_json(qa).encode()),
                "solver_max_model_len":16384,
                "solver_protocol":{"enable_thinking":False, "temperature":0.7,
                    "top_p":0.9, "max_tokens":32, "parser":"boxed_letter_v1",
                    "max_parse_retries_per_target":1}}
            atomic_json(solver_root / "config/resolved_canary_config.json", solver_config)
            atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
            solver_report = _run_solver(root=solver_root, manifest=solver_manifest,
                config=solver_config, raw_batch=questioner_result["raw_batch"],
                admission=admission, materialized=qa, artifacts=artifacts,
                tokenizer=tokenizer, backend=backend)
            solver_provenance = write_post_run_provenance(
                solver_root, PROJECT, solver_manifest)
            if not solver_provenance["authoritative_release_provenance"]:
                raise RuntimeError("evidence view solver provenance mixed")
            result = solver_report["result"]
            record.update({"solver_evaluated":True, "solver_root":str(solver_root),
                "materialized_qa_sha256":sha256_bytes(canonical_json(qa).encode()),
                "final_decision":solver_report["final_decision"],
                "no_context_permutation_diagnostic":solver_report[
                    "no_context_permutation_diagnostic"],
                "no_context_rewards":(result.get("no_context") or {}).get(
                    "verified_rewards"),
                "full_context_rewards":(result.get("full_context") or {}).get(
                    "verified_rewards"),
                "full_context_decision":(result.get("full_context_decision") or
                    (result.get("full_context") or {}).get("decision")),
                "evidence_audit_decision":(result.get("evidence_audit") or {}).get(
                    "decision"),
                "option_audit_decision":(result.get("option_audit") or {}).get(
                    "decision")})
        items.append(record)
    decisions: dict[str, int] = {}
    for item in items:
        decision = item["final_decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
    summary = {"view_mode":args.view_mode,
        "source_parent_turn_count":view["source_parent_turn_count"],
        "source_child_turn_count":view["source_child_turn_count"],
        "visible_parent_turn_count":view["selected_parent_turn_count"],
        "visible_child_turn_count":view["selected_child_turn_count"],
        "raw_candidate_count":questioner_result["report"]["raw_candidate_count"],
        "minimal_accepted_count":questioner_result["report"]["minimal_accepted_count"],
        "solver_evaluated_count":sum(item["solver_evaluated"] for item in items),
        "final_decisions":decisions,
        "retain_frontier_count":decisions.get("retain_frontier", 0)}
    report = {"evidence_view_ablation_version":EVIDENCE_VIEW_ABLATION_VERSION,
        "view_mode":args.view_mode, "view_sha256":view["view_sha256"],
        "source_graph_attestation":source_public,
        "questioner_prompt_preflight":artifacts["preflight"],
        "summary":summary, "items":items,
        "paid_questioner_generations":questioner_result["report"][
            "paid_api_generations"], "realizer_generations":0,
        "verifier_generations":0, "training_runs":0}
    report_path = args.run_root / "reports/evidence_view_ablation.json"
    atomic_json(report_path, report)
    top_manifest = _manifest(run_id=args.run_root.name,
        mode="questioner_evidence_view_ablation_series", snapshot=snapshot,
        project_diff=project_diff, source=source, artifacts=artifacts,
        feedback_sha=feedback_sha, api=api)
    top_manifest.update({"questioner_manifest_sha256":sha256_file(
        questioner_root / "config/run_manifest.json"),
        "report_sha256":sha256_file(report_path)})
    atomic_json(args.run_root / "config/run_manifest.json", top_manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode":"questioner_evidence_view_ablation_series", "view_mode":args.view_mode,
        "questioner_batches":1, "mcqs_per_batch":3,
        "maximum_paid_questioner_requests":2, "realizer_generations":0,
        "verifier_generations":0, "training_runs":0,
        "solver_effective_max_model_len":16384})
    provenance = write_post_run_provenance(args.run_root, PROJECT, top_manifest)
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("evidence view ablation provenance mixed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
