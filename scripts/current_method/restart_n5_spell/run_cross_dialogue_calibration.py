from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .cross_dialogue_calibration import (
    CROSS_DIALOGUE_CALIBRATION_VERSION, calibration_plan, freeze_parent_dialogue,
    summarize_calibration, verify_semantic_repair_feedback_source,
)
from .input_adapter import load_lineage, read_json
from .live_backends import (LiveQuestionerBackend, OpenAICompatibleChallengerClient,
                            load_challenger_config)
from .minimal_admission import SPAN_IDS_ONLY_SCHEMA_PROFILE, SPELL_STYLE_MINIMAL_MODE
from .orchestrator import CanaryOrchestrator
from .prompts import (QUESTIONER_FORMAT_RETRY_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
                      build_questioner_span_ids_only_feedback_prompt,
                      questioner_format_retry_suffix_hash,
                      span_ids_only_questioner_template_hash)
from .questioner_contract import (build_span_catalog, default_reasoning_contracts,
                                  reasoning_contract_sha256, span_catalog_sha256)
from .run_canary import (DEFAULTS, create_source_snapshot, git_value, project_diff_sha256,
                         select_minimal_solver_qa, write_post_run_provenance)
from .validation import validate_config, validate_dual_admission_batch
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
SPELL_COMMIT = "4e5aee32280febb7b0844237ba087f837e3c8905"


def _base_manifest(*, run_id: str, mode: str, snapshot: dict[str, Any],
                   project_diff: str) -> dict[str, Any]:
    return {"run_id":run_id, "mode":mode,
        "project_git_commit":git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256":project_diff,
        "source_snapshot_manifest_path":snapshot["source_snapshot_manifest_path"],
        "source_snapshot_manifest_sha256":snapshot["source_snapshot_manifest_sha256"],
        "orchestrator_source_sha256":sha256_file(MODULE_ROOT / "orchestrator.py"),
        "validation_source_sha256":sha256_file(MODULE_ROOT / "validation.py"),
        "ledger_source_sha256":sha256_file(MODULE_ROOT / "ledger.py"),
        "run_canary_source_sha256":sha256_file(MODULE_ROOT / "run_canary.py"),
        "mcq_parser_source_sha256":sha256_file(MODULE_ROOT / "core.py"),
        "questioner_prompt_builder_source_sha256":sha256_file(MODULE_ROOT / "prompts.py"),
        "questioner_contract_source_sha256":sha256_file(
            MODULE_ROOT / "questioner_contract.py"),
        "minimal_validator_source_sha256":sha256_file(MODULE_ROOT / "minimal_admission.py"),
        "normalization_function_source_sha256":sha256_file(
            MODULE_ROOT / "minimal_admission.py"),
        "minimal_replay_source_sha256":sha256_file(MODULE_ROOT / "minimal_replay.py"),
        "cross_dialogue_calibration_source_sha256":sha256_file(
            MODULE_ROOT / "cross_dialogue_calibration.py"),
        "cross_dialogue_runner_source_sha256":sha256_file(Path(__file__)),
        "admission_mode":SPELL_STYLE_MINIMAL_MODE,
        "calibration_version":CROSS_DIALOGUE_CALIBRATION_VERSION,
        "spell_repo_path":"/root/autodl-tmp/third_party_readonly/Qwen-Doc",
        "spell_commit":SPELL_COMMIT, "spell_dirty":False}


def _freeze_questioner_inputs(*, root: Path, lineage_data: dict[str, Any],
                              plan: dict[str, Any], feedback: dict[str, Any],
                              tokenizer: Any, max_generation_tokens: int,
                              backend_context_limit: int) -> dict[str, Any]:
    frozen = freeze_parent_dialogue(plan["lineage"], lineage_data["parent"])
    context = frozen["context"]
    turns = context["combined_dialogue"]
    letters = tuple(plan["assigned_target_letters"])
    catalog = build_span_catalog(turns)
    contracts = default_reasoning_contracts(letters)
    prompt = build_questioner_span_ids_only_feedback_prompt(
        frozen_dialogue=turns, valid_turn_ids=[turn["turn_id"] for turn in turns],
        valid_session_ids=sorted({turn["session_id"] for turn in turns}),
        assigned_gold_letters=letters, solver_feedback=feedback["feedback"],
        batch_index=plan["dialogue_index"], span_catalog=catalog)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    preflight = {"rendered_prompt_tokens":prompt_tokens,
        "max_generation_tokens":max_generation_tokens,
        "total_tokens":prompt_tokens + max_generation_tokens,
        "backend_context_limit":backend_context_limit,
        "decision":"pass" if prompt_tokens + max_generation_tokens <=
            backend_context_limit else "reject_context_overflow",
        "truncated":False}
    if preflight["decision"] != "pass":
        raise RuntimeError(f"calibration_questioner_context_overflow:{plan['lineage']}")
    atomic_json(root / "frozen/questioner_smoke_context.json", context)
    atomic_json(root / "frozen/questioner_span_catalog.json", catalog)
    atomic_json(root / "frozen/questioner_reasoning_contracts.json", contracts)
    atomic_json(root / "frozen/parent_source_attestation.json",
                frozen["source_attestation"])
    atomic_json(root / "frozen/challenger_solver_feedback.json", feedback)
    atomic_json(root / "frozen/questioner_prompt_preflight.json", preflight)
    atomic_json(root / "raw/prompts/questioner_semantic_prompt.json",
        {"prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
         "rendered_prompt_sha256":sha256_bytes(prompt.encode()), "prompt":prompt})
    return {"context":context, "catalog":catalog, "contracts":contracts,
        "letters":letters, "prompt":prompt, "preflight":preflight,
        "source_attestation":frozen["source_attestation"]}


def _questioner_config(*, run_root: Path, api: Any) -> dict[str, Any]:
    config = {**DEFAULTS, "mode":"cross_dialogue_questioner_batch",
        "backend_mode":"live", "run_root":str(run_root),
        "max_total_paid_api_generations":2, "max_realizer_generations":0,
        "max_questioner_generations":2, "max_post_freeze_memory_generations":0,
        "max_verifier_generations":0, "questioner_generations":1,
        "questioner_mcqs_per_generation":3, "questioner_total_candidates":3,
        "max_questioner_format_retries":1, "questioner_physical_request_limit":2,
        "challenger_model_identifier":api.model, "challenger_temperature":api.temperature,
        "challenger_questioner_max_tokens":api.questioner_max_tokens,
        "semantic_repair_generations":0, "training_runs":0}
    validate_config(config)
    return config


def _solver_config(*, run_root: Path) -> dict[str, Any]:
    config = {**DEFAULTS, "mode":"local_solver_cross_dialogue_calibration",
        "backend_mode":"local", "run_root":str(run_root),
        "max_total_paid_api_generations":0, "max_realizer_generations":0,
        "max_questioner_generations":0, "max_post_freeze_memory_generations":0,
        "max_verifier_generations":0, "max_local_solver_generations":64,
        "solver_temperature":0.7, "solver_top_p":0.9,
        "solver_max_new_tokens":32, "solver_enable_thinking":False,
        "solver_add_generation_prompt":True, "solver_effective_max_model_len":16384,
        "solver_tokenizer_path":"/root/autodl-tmp/Qwen3-4B",
        "solver_gpu_memory_utilization":0.80, "semantic_repair_generations":0,
        "training_runs":0}
    validate_config(config)
    return config


def _run_questioner(*, root: Path, manifest: dict[str, Any], config: dict[str, Any],
                    artifacts: dict[str, Any], backend: LiveQuestionerBackend,
                    tokenizer: Any, feedback_sha: str) -> dict[str, Any]:
    orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest, config=config,
        data={}, tokenizer=tokenizer, realizer=None, questioner=backend, solver=None,
        backend_mode="live")
    context = artifacts["context"]["combined_dialogue"]
    smoke = orchestrator.question_smoke(context,
        tuple(turn["turn_id"] for turn in context),
        tuple(sorted({turn["session_id"] for turn in context})), artifacts["letters"],
        span_catalog=artifacts["catalog"], reasoning_contracts=artifacts["contracts"],
        rendered_prompt=artifacts["prompt"],
        prompt_protocol=QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        max_format_retries=1, solver_feedback_sha256=feedback_sha,
        raw_schema_mode=SPAN_IDS_ONLY_SCHEMA_PROFILE,
        questioner_schema_version=QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION)
    raw_batch: list[Any] = []
    if smoke.get("parse_status") == "json_parsed" and smoke.get("raw_object_count") == 3:
        state = orchestrator.ledger.read(smoke["final_request_id"]) or {}
        raw_batch = json.loads(Path(state["raw_response_path"]).read_text(encoding="utf-8"))
    turn_text = {turn["turn_id"]:turn["text"] for turn in context}
    turn_sessions = {turn["turn_id"]:turn["session_id"] for turn in context}
    dual = validate_dual_admission_batch(raw_batch, list(artifacts["letters"]),
        span_catalog=artifacts["catalog"], reasoning_contracts=artifacts["contracts"],
        frozen_turns=turn_text, turn_session_ids=turn_sessions,
        minimal_schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)
    materialized: dict[int, dict[str, Any]] = {}
    report_items = []
    for item in dual.get("items", []):
        record = json.loads(canonical_json(item))
        minimal = record[SPELL_STYLE_MINIMAL_MODE]
        qa = minimal.pop("materialized_qa", None)
        if minimal["decision"] == "accept" and isinstance(qa, dict):
            item_index = item["item_index"]
            path = root / f"materialized/item_{item_index:02d}.json"
            atomic_json(path, qa)
            record["materialized_qa_path"] = str(path.relative_to(root))
            record["materialized_qa_sha256"] = sha256_file(path)
            materialized[item_index] = qa
        else:
            record["materialized_qa_path"] = None
            record["materialized_qa_sha256"] = None
        report_items.append(record)
    report = {"lineage":artifacts["context"]["lineage"],
        "questioner_prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        "raw_candidate_count":dual.get("raw_count", 0),
        "strict_v7_accepted_count":dual.get("strict_v7_accepted_count", 0),
        "minimal_accepted_count":dual.get("minimal_accepted_count", 0),
        "items":report_items, "questioner_smoke":smoke,
        "paid_api_generations":orchestrator.ledger.summary()["paid_api_generation_count"],
        "realizer_generations":0, "semantic_repair_generations":0,
        "verifier_generations":0, "training_runs":0}
    atomic_json(root / "reports/cross_dialogue_questioner_batch.json", report)
    return {"raw_batch":raw_batch, "dual":dual, "materialized":materialized,
        "report":report, "ledger_summary":orchestrator.ledger.summary()}


def _run_solver(*, root: Path, manifest: dict[str, Any], config: dict[str, Any],
                raw_batch: list[dict[str, Any]], admission: dict[str, Any],
                materialized: dict[str, Any], artifacts: dict[str, Any],
                tokenizer: Any, backend: VLLMSolverBackend) -> dict[str, Any]:
    atomic_json(root / "frozen/questioner_smoke_context.json", artifacts["context"])
    atomic_json(root / "frozen/questioner_span_catalog.json", artifacts["catalog"])
    atomic_json(root / "frozen/questioner_reasoning_contracts.json", artifacts["contracts"])
    atomic_json(root / "frozen/solver_qa.json", materialized)
    _, qa, _ = select_minimal_solver_qa(raw_batch, admission, admission["qa_id"],
        materialized_qa=materialized, span_catalog=artifacts["catalog"],
        reasoning_contracts=artifacts["contracts"])
    orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest, config=config,
        data={}, tokenizer=tokenizer, realizer=None, questioner=None, solver=backend,
        backend_mode="local")
    result = orchestrator.audit_minimal_candidate(qa)
    diagnostic = None
    if result.get("final_decision") == "bad_ground":
        initial = result.get("initial_no_context") or {}
        if (initial.get("verified_rewards") == [1, 1, 1] and
                result.get("adaptive_extension") is None):
            diagnostic = orchestrator.diagnose_no_context_permutations(qa, initial)
        else:
            diagnostic = {"qa_id":qa["qa_id"],
                "authoritative_grounding_decision":"bad_ground",
                "diagnostic_decision":"bad_ground_permutation_inconclusive",
                "diagnostic_not_run_reason":"adaptive_grounding_boundary_result",
                "new_vllm_generations":0}
    report = {"lineage":artifacts["context"]["lineage"], "qa_id":qa["qa_id"],
        "result":result, "no_context_permutation":diagnostic,
        "final_decision":result.get("final_decision"),
        "no_context_permutation_diagnostic":(diagnostic or {}).get(
            "diagnostic_decision"), **orchestrator.ledger.summary(),
        "paid_api_generations":0, "realizer_generations":0,
        "questioner_generations":0, "semantic_repair_generations":0,
        "verifier_generations":0, "training_runs":0}
    atomic_json(root / "reports/cross_dialogue_solver_audit.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Five-dialogue SPELL rejection-sampling calibration")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--api-config", type=Path,
        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--feedback-source-root", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conv-41_spell_v8_semantic_repair_series_v1"))
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"calibration root already exists: {args.run_root}")
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    spell_root = Path("/root/autodl-tmp/third_party_readonly/Qwen-Doc")
    if git_value(spell_root, "rev-parse", "HEAD") != SPELL_COMMIT or git_value(
            spell_root, "status", "--porcelain"):
        raise RuntimeError("SPELL source provenance drift")
    feedback = verify_semantic_repair_feedback_source(args.feedback_source_root)
    atomic_json(args.run_root / "frozen/calibration_feedback_memory.json", feedback)
    api_config = load_challenger_config(args.api_config)
    tokenizer_path = Path("/root/autodl-tmp/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    legacy = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
    plan = calibration_plan()
    atomic_json(args.run_root / "frozen/calibration_plan.json", plan)
    prepared = []
    for entry in plan:
        lineage = entry["lineage"]
        root = args.run_root / f"dialogues/{lineage}/questioner"
        root.mkdir(parents=True)
        data = load_lineage(args.input_root, lineage, legacy)
        artifacts = _freeze_questioner_inputs(root=root, lineage_data=data, plan=entry,
            feedback=feedback, tokenizer=tokenizer,
            max_generation_tokens=api_config.questioner_max_tokens,
            backend_context_limit=DEFAULTS["questioner_backend_context_limit"])
        config = _questioner_config(run_root=root, api=api_config)
        manifest = {**_base_manifest(run_id=f"{args.run_root.name}-{lineage}-questioner",
                mode="cross_dialogue_questioner_batch", snapshot=snapshot,
                project_diff=project_diff),
            "lineage":lineage, "parent_only":True,
            "parent_source_attestation":artifacts["source_attestation"],
            "frozen_dialogue_sha256":sha256_file(
                root / "frozen/questioner_smoke_context.json"),
            "span_catalog_sha256":span_catalog_sha256(artifacts["catalog"]),
            "span_catalog_file_sha256":sha256_file(
                root / "frozen/questioner_span_catalog.json"),
            "reasoning_contract_sha256":reasoning_contract_sha256(artifacts["contracts"]),
            "target_gold_letters":list(artifacts["letters"]),
            "questioner_prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
            "questioner_prompt_sha256":span_ids_only_questioner_template_hash(),
            "rendered_prompt_sha256":sha256_bytes(artifacts["prompt"].encode()),
            "questioner_schema_version":QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
            "questioner_output_profile":SPAN_IDS_ONLY_SCHEMA_PROFILE,
            "format_retry_version":QUESTIONER_FORMAT_RETRY_VERSION,
            "format_retry_suffix_sha256":questioner_format_retry_suffix_hash(),
            "feedback_memory_sha256":sha256_file(
                args.run_root / "frozen/calibration_feedback_memory.json"),
            "questioner_prompt_preflight":artifacts["preflight"],
            "challenger_model_identifier":api_config.model,
            "sampling_parameters":{"temperature":api_config.temperature,
                "max_tokens":api_config.questioner_max_tokens},
            "budget":{"normal_batches":1, "maximum_format_retries":1,
                "maximum_paid_requests":2, "realizer":0, "semantic_repair":0,
                "verifier":0, "training":0},
            "live_backend_source_sha256":sha256_file(MODULE_ROOT / "live_backends.py")}
        atomic_json(root / "config/resolved_canary_config.json", config)
        atomic_json(root / "config/run_manifest.json", manifest)
        prepared.append({"plan":entry, "root":root, "data":data,
                         "artifacts":artifacts, "config":config, "manifest":manifest})
    top_manifest = {**_base_manifest(run_id=args.run_root.name,
            mode="cross_dialogue_rejection_sampling_calibration", snapshot=snapshot,
            project_diff=project_diff), "plan":plan,
        "feedback_memory_sha256":sha256_file(
            args.run_root / "frozen/calibration_feedback_memory.json"),
        "maximum_source_dialogues":5, "maximum_raw_candidates":15,
        "maximum_normal_questioner_requests":5, "maximum_format_retry_requests":5,
        "semantic_repair_enabled":False,
        "v8_generation_prompt_sha256":span_ids_only_questioner_template_hash(),
        "live_backend_source_sha256":sha256_file(MODULE_ROOT / "live_backends.py"),
        "vllm_backend_source_sha256":sha256_file(MODULE_ROOT / "vllm_backends.py")}
    atomic_json(args.run_root / "config/run_manifest.json", top_manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode":"cross_dialogue_rejection_sampling_calibration",
        "dialogues":5, "batches_per_dialogue":1, "mcqs_per_batch":3,
        "maximum_paid_questioner_requests":10, "realizer_generations":0,
        "semantic_repair_generations":0, "verifier_generations":0,
        "training_runs":0, "solver_effective_max_model_len":16384})

    client = OpenAICompatibleChallengerClient(api_config)
    questioner_backend = LiveQuestionerBackend(client)
    questioner_results = []
    for item in prepared:
        result = _run_questioner(root=item["root"], manifest=item["manifest"],
            config=item["config"], artifacts=item["artifacts"],
            backend=questioner_backend, tokenizer=tokenizer,
            feedback_sha=top_manifest["feedback_memory_sha256"])
        item["questioner_result"] = result
        output = {"questioner_report_sha256":sha256_file(
                item["root"] / "reports/cross_dialogue_questioner_batch.json"),
            "raw_candidate_count":result["report"]["raw_candidate_count"],
            "minimal_accepted_count":result["report"]["minimal_accepted_count"],
            "paid_api_generations":result["report"]["paid_api_generations"]}
        atomic_json(item["root"] / "reports/questioner_output_provenance.json", output)
        write_post_run_provenance(item["root"], PROJECT, item["manifest"])
        questioner_results.append(output)

    candidates = [(item, admission, item["questioner_result"]["materialized"][
                    admission["item_index"]])
        for item in prepared for admission in item["questioner_result"]["dual"].get("items", [])
        if admission.get("solver_eligible") is True and
        admission["item_index"] in item["questioner_result"]["materialized"]]
    solver_backend = None
    if candidates:
        solver_backend = VLLMSolverBackend(
            model_path=legacy / "accepted_models/solver_base_restart_stage2_merged",
            tokenizer_path=tokenizer_path, max_model_len=16384,
            temperature=0.7, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=0.80)
    dialogue_reports = []
    for item in prepared:
        dialogue_record = {"lineage":item["plan"]["lineage"],
            "questioner_root":str(item["root"]),
            "raw_candidate_count":item["questioner_result"]["report"][
                "raw_candidate_count"],
            "minimal_accepted_count":item["questioner_result"]["report"][
                "minimal_accepted_count"], "items":[]}
        for admission in item["questioner_result"]["dual"].get("items", []):
            item_record = {"qa_id":admission.get("qa_id"),
                "item_index":admission.get("item_index"),
                "strict_v7":admission.get("strict_v7"),
                SPELL_STYLE_MINIMAL_MODE:{key:value for key,value in
                    (admission.get(SPELL_STYLE_MINIMAL_MODE) or {}).items()
                    if key != "materialized_qa"},
                "solver_evaluated":False, "final_decision":"minimal_admission_reject"}
            if admission.get("solver_eligible") is True:
                qa = item["questioner_result"]["materialized"][admission["item_index"]]
                qa_path_key = sha256_bytes(str(qa["qa_id"]).encode())[:12]
                solver_root = args.run_root / (
                    f"dialogues/{item['plan']['lineage']}/solvers/item_"
                    f"{admission['item_index']:02d}_{qa_path_key}")
                solver_root.mkdir(parents=True)
                solver_config = _solver_config(run_root=solver_root)
                solver_manifest = {**_base_manifest(
                        run_id=f"{args.run_root.name}-{item['plan']['lineage']}-{qa['qa_id']}",
                        mode="local_solver_cross_dialogue_calibration", snapshot=snapshot,
                        project_diff=project_diff),
                    "lineage":item["plan"]["lineage"], "qa_id":qa["qa_id"],
                    "source_questioner_root":str(item["root"]),
                    "source_questioner_manifest_sha256":sha256_file(
                        item["root"] / "config/run_manifest.json"),
                    "source_questioner_report_sha256":sha256_file(
                        item["root"] / "reports/cross_dialogue_questioner_batch.json"),
                    "materialized_qa_sha256":sha256_bytes(canonical_json(qa).encode()),
                    "frozen_dialogue_sha256":sha256_file(
                        item["root"] / "frozen/questioner_smoke_context.json"),
                    "span_catalog_sha256":span_catalog_sha256(item["artifacts"]["catalog"]),
                    "solver_max_model_len":16384,
                    "vllm_backend_source_sha256":sha256_file(MODULE_ROOT / "vllm_backends.py"),
                    "solver_protocol":{"enable_thinking":False,
                        "temperature":0.7, "top_p":0.9, "max_tokens":32,
                        "parser":"boxed_letter_v1", "max_parse_retries_per_target":1},
                    "semantic_repair_enabled":False}
                atomic_json(solver_root / "config/resolved_canary_config.json", solver_config)
                atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
                solver_report = _run_solver(root=solver_root, manifest=solver_manifest,
                    config=solver_config, raw_batch=item["questioner_result"]["raw_batch"],
                    admission=admission, materialized=qa, artifacts=item["artifacts"],
                    tokenizer=tokenizer, backend=solver_backend)
                write_post_run_provenance(solver_root, PROJECT, solver_manifest)
                item_record.update({"solver_evaluated":True,
                    "solver_root":str(solver_root),
                    "solver_report_sha256":sha256_file(
                        solver_root / "reports/cross_dialogue_solver_audit.json"),
                    "final_decision":solver_report["final_decision"],
                    "no_context_permutation_diagnostic":solver_report[
                        "no_context_permutation_diagnostic"]})
            dialogue_record["items"].append(item_record)
        dialogue_reports.append(dialogue_record)
    summary = summarize_calibration(dialogue_reports)
    paid = sum(result["paid_api_generations"] for result in questioner_results)
    report = {"calibration_version":CROSS_DIALOGUE_CALIBRATION_VERSION,
        "summary":summary, "dialogues":dialogue_reports,
        "normal_questioner_physical_requests":paid,
        "format_retry_requests":sum(max(0, result["paid_api_generations"] - 1)
                                    for result in questioner_results),
        "realizer_generations":0, "semantic_repair_generations":0,
        "verifier_generations":0, "training_runs":0}
    atomic_json(args.run_root / "reports/cross_dialogue_calibration.json", report)
    atomic_json(args.run_root / "reports/calibration_output_provenance.json", {
        "report_sha256":sha256_file(args.run_root / "reports/cross_dialogue_calibration.json"),
        "questioner_runs":questioner_results,
        "source_snapshot_manifest_sha256":snapshot["source_snapshot_manifest_sha256"],
        "project_diff_sha256":project_diff_sha256(PROJECT)})
    write_post_run_provenance(args.run_root, PROJECT, top_manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
