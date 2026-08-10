from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .lineage_distractor_writer import (
    DISTRACTOR_WRITER_PROMPT_VERSION, DISTRACTOR_WRITER_SCHEMA_VERSION,
    build_distractor_writer_prompt, distractor_prompt_template_sha256,
    distractor_schema_sha256, validate_distractor_object,
)
from .lineage_full_dialogue_prompts import (
    DISTRACTOR_WRITER_FULL_DIALOGUE_PROMPT_VERSION,
    FULL_DIALOGUE_INPUT_VERSION, QA_WRITER_FULL_DIALOGUE_PROMPT_VERSION,
    RELATION_MINER_FULL_DIALOGUE_PROMPT_VERSION,
    build_full_dialogue_distractor_writer_prompt,
    build_full_dialogue_qa_writer_prompt,
    build_full_dialogue_relation_miner_prompt,
    full_dialogue_distractor_prompt_sha256,
    full_dialogue_qa_prompt_sha256, full_dialogue_relation_prompt_sha256,
)
from .lineage_qa_writer import (
    QA_WRITER_PROMPT_VERSION, QA_WRITER_SCHEMA_VERSION,
    build_qa_writer_prompt, qa_writer_prompt_template_sha256,
    qa_writer_schema_sha256, validate_question_answer_object,
)
from .lineage_relation_miner import (
    RELATION_MINER_PROMPT_VERSION, RELATION_MINER_SCHEMA_VERSION,
    build_relation_miner_prompt, relation_prompt_template_sha256,
    relation_schema_sha256, validate_relation_candidate,
)
from .lineage_three_stage_canary import (
    ANSWER_PLACEMENT_VERSION, FORMAT_ONLY_RETRY_VERSION,
    LINEAGE_THREE_STAGE_CANARY_VERSION, ApiBudget, artifact_sha256_manifest,
    build_relation_diagnostic_contexts, classify_joint_necessity_v1,
    deterministic_answer_placement, format_only_retry_suffix_sha256,
    materialize_three_stage_mcq, run_single_object_task,
    should_stop_after_first_retain_frontier,
)
from .ledger import RequestLedger
from .live_backends import (LiveSemanticRepairBackend,
                            OpenAICompatibleChallengerClient,
                            load_challenger_config)
from .orchestrator import CanaryOrchestrator
from .questioner_contract import build_span_catalog, span_catalog_sha256
from .run_canary import (create_source_snapshot, project_diff_sha256,
                         render_solver_prompt, write_post_run_provenance)
from .run_cross_dialogue_calibration import _base_manifest, _solver_config
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_spell_three_scope_questioner_canary_v1")
DEFAULT_SOLVER = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "accepted_models/solver_base_restart_stage2_merged")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
ARC_ORDER = ("E1", "E4")
LINEAGE_SPAN_INPUT = "lineage_span_catalog_v1"
FULL_DIALOGUE_INPUT = FULL_DIALOGUE_INPUT_VERSION
SOURCE_GRAPH_CONTRACTS = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_spell_full_graph_expansion_v7/frozen/graph_scene_contracts.json")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_code_fields(input_mode: str = LINEAGE_SPAN_INPUT) -> dict[str, Any]:
    if input_mode == FULL_DIALOGUE_INPUT:
        prompt_fields = {
            "relation_miner_prompt_version":
                RELATION_MINER_FULL_DIALOGUE_PROMPT_VERSION,
            "relation_miner_prompt_sha256":
                full_dialogue_relation_prompt_sha256(),
            "qa_writer_prompt_version":QA_WRITER_FULL_DIALOGUE_PROMPT_VERSION,
            "qa_writer_prompt_sha256":full_dialogue_qa_prompt_sha256(),
            "distractor_writer_prompt_version":
                DISTRACTOR_WRITER_FULL_DIALOGUE_PROMPT_VERSION,
            "distractor_writer_prompt_sha256":
                full_dialogue_distractor_prompt_sha256(),
            "lineage_full_dialogue_prompt_source_sha256":sha256_file(
                MODULE_ROOT / "lineage_full_dialogue_prompts.py")}
    elif input_mode == LINEAGE_SPAN_INPUT:
        prompt_fields = {
            "relation_miner_prompt_version":RELATION_MINER_PROMPT_VERSION,
            "relation_miner_prompt_sha256":relation_prompt_template_sha256(),
            "qa_writer_prompt_version":QA_WRITER_PROMPT_VERSION,
            "qa_writer_prompt_sha256":qa_writer_prompt_template_sha256(),
            "distractor_writer_prompt_version":DISTRACTOR_WRITER_PROMPT_VERSION,
            "distractor_writer_prompt_sha256":
                distractor_prompt_template_sha256()}
    else:
        raise ValueError(f"unknown_challenger_input_mode:{input_mode}")
    return {
        "challenger_input_mode":input_mode,
        "lineage_relation_miner_source_sha256":sha256_file(
            MODULE_ROOT / "lineage_relation_miner.py"),
        "lineage_qa_writer_source_sha256":sha256_file(
            MODULE_ROOT / "lineage_qa_writer.py"),
        "lineage_distractor_writer_source_sha256":sha256_file(
            MODULE_ROOT / "lineage_distractor_writer.py"),
        "lineage_three_stage_canary_source_sha256":sha256_file(
            MODULE_ROOT / "lineage_three_stage_canary.py"),
        "lineage_three_stage_runner_source_sha256":sha256_file(Path(__file__)),
        "vllm_backend_source_sha256":sha256_file(MODULE_ROOT / "vllm_backends.py"),
        **prompt_fields,
        "relation_miner_schema_version":RELATION_MINER_SCHEMA_VERSION,
        "relation_miner_schema_sha256":relation_schema_sha256(),
        "qa_writer_schema_version":QA_WRITER_SCHEMA_VERSION,
        "qa_writer_schema_sha256":qa_writer_schema_sha256(),
        "distractor_writer_schema_version":DISTRACTOR_WRITER_SCHEMA_VERSION,
        "distractor_writer_schema_sha256":distractor_schema_sha256(),
        "format_only_retry_version":FORMAT_ONLY_RETRY_VERSION,
        "format_only_retry_suffix_sha256":format_only_retry_suffix_sha256(),
        "answer_placement_version":ANSWER_PLACEMENT_VERSION,
    }


def _request_view(view: Mapping[str, Any], full_context: Mapping[str, Any],
                  input_mode: str) -> dict[str, Any]:
    result = dict(view)
    if input_mode == FULL_DIALOGUE_INPUT:
        result["combined_dialogue"] = [dict(turn) for turn in
                                       full_context["combined_dialogue"]]
    return result


def _relation_prompt(*, input_mode: str, arc_id: str, candidate_index: int,
        view: Mapping[str, Any], full_context: Mapping[str, Any],
        membership: Mapping[str, list[str]], prior: Mapping[str, Any] | None
        ) -> str:
    if input_mode == FULL_DIALOGUE_INPUT:
        return build_full_dialogue_relation_miner_prompt(arc_id=arc_id,
            candidate_index=candidate_index, lineage_view=view,
            full_dialogue=full_context["combined_dialogue"],
            span_stage_membership=membership, prior_relation=prior)
    return build_relation_miner_prompt(arc_id=arc_id,
        candidate_index=candidate_index, lineage_view=view,
        span_stage_membership=membership, prior_relation=prior)


def _qa_prompt(*, input_mode: str, view: Mapping[str, Any],
        full_context: Mapping[str, Any], relation: Mapping[str, Any],
        relation_sha: str) -> str:
    if input_mode == FULL_DIALOGUE_INPUT:
        return build_full_dialogue_qa_writer_prompt(lineage_view=view,
            full_dialogue=full_context["combined_dialogue"],
            relation_bundle=relation, relation_bundle_sha256=relation_sha)
    return build_qa_writer_prompt(lineage_view=view, relation_bundle=relation,
                                  relation_bundle_sha256=relation_sha)


def _distractor_prompt(*, input_mode: str, view: Mapping[str, Any],
        full_context: Mapping[str, Any], relation: Mapping[str, Any],
        relation_sha: str, qa: Mapping[str, Any], qa_sha: str,
        attempt_index: int, feedback: Mapping[str, Any] | None) -> str:
    if input_mode == FULL_DIALOGUE_INPUT:
        return build_full_dialogue_distractor_writer_prompt(lineage_view=view,
            full_dialogue=full_context["combined_dialogue"],
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            question_answer_object=qa, question_answer_sha256=qa_sha,
            attempt_index=attempt_index, structured_feedback=feedback)
    return build_distractor_writer_prompt(lineage_view=view,
        relation_bundle=relation, relation_bundle_sha256=relation_sha,
        question_answer_object=qa, question_answer_sha256=qa_sha,
        attempt_index=attempt_index, structured_feedback=feedback)


def _attest_source(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    main_manifest_path = source_root / "config/run_manifest.json"
    main_report_path = source_root / "reports/three_scope_canary.json"
    main_provenance_path = source_root / "reports/post_run_provenance.json"
    for path in (main_manifest_path, main_report_path, main_provenance_path,
                 SOURCE_GRAPH_CONTRACTS):
        if not path.is_file():
            raise RuntimeError(f"source_attestation_file_missing:{path}")
    provenance = _read(main_provenance_path)
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("source_three_scope_provenance_not_authoritative")
    views: dict[str, dict[str, Any]] = {}
    view_sources: dict[str, dict[str, Any]] = {}
    for arc_id in ARC_ORDER:
        qroot = source_root / f"arcs/{arc_id}/questioner"
        qprov_path = qroot / "reports/post_run_provenance.json"
        view_path = qroot / "frozen/lineage_view.json"
        catalog_path = qroot / "frozen/questioner_span_catalog.json"
        qmanifest_path = qroot / "config/run_manifest.json"
        qprov = _read(qprov_path)
        if (qprov.get("authoritative_release_provenance") is not True or
                qprov.get("provenance_mixed") is not False):
            raise RuntimeError(f"source_arc_provenance_not_authoritative:{arc_id}")
        view, catalog = _read(view_path), _read(catalog_path)
        if canonical_json(view.get("span_catalog")) != canonical_json(catalog):
            raise RuntimeError(f"source_lineage_catalog_drift:{arc_id}")
        body = {key:value for key,value in view.items() if key != "view_sha256"}
        if sha256_bytes(canonical_json(body).encode()) != view.get("view_sha256"):
            raise RuntimeError(f"source_lineage_view_hash_invalid:{arc_id}")
        if view.get("event_arc_id") != arc_id:
            raise RuntimeError(f"source_lineage_arc_mismatch:{arc_id}")
        views[arc_id] = view
        view_sources[arc_id] = {"source_lineage_view_path":str(view_path),
            "source_lineage_view_file_sha256":sha256_file(view_path),
            "source_lineage_view_sha256":view["view_sha256"],
            "source_span_catalog_path":str(catalog_path),
            "source_span_catalog_file_sha256":sha256_file(catalog_path),
            "source_span_catalog_sha256":span_catalog_sha256(catalog),
            "source_questioner_manifest_sha256":sha256_file(qmanifest_path),
            "source_questioner_provenance_sha256":sha256_file(qprov_path)}
    full_context_paths = [source_root /
        f"arcs/{arc}/solvers/item_00/frozen/questioner_smoke_context.json"
        for arc in ARC_ORDER]
    full_catalog_paths = [source_root /
        f"arcs/{arc}/solvers/item_00/frozen/questioner_span_catalog.json"
        for arc in ARC_ORDER]
    contexts = [_read(path) for path in full_context_paths]
    catalogs = [_read(path) for path in full_catalog_paths]
    if canonical_json(contexts[0]) != canonical_json(contexts[1]):
        raise RuntimeError("source_full_dialogue_drift_between_arcs")
    if canonical_json(catalogs[0]) != canonical_json(catalogs[1]):
        raise RuntimeError("source_full_catalog_drift_between_arcs")
    full_dialogue = contexts[0]["combined_dialogue"]
    if len(full_dialogue) != 371:
        raise RuntimeError("source_full_dialogue_turn_count_invalid")
    parent_count = sum(not str(turn["turn_id"]).startswith("C")
                       for turn in full_dialogue)
    child_count = len(full_dialogue) - parent_count
    if (parent_count, child_count) != (237, 134):
        raise RuntimeError("source_parent_child_turn_count_invalid")
    contracts_wrapper = _read(SOURCE_GRAPH_CONTRACTS)
    contracts = contracts_wrapper["contracts"]
    return {"source_root":str(source_root), "views":views,
        "view_sources":view_sources, "full_context":contexts[0],
        "full_catalog":catalogs[0], "graph_scene_contracts":contracts,
        "public":{"source_three_scope_run":str(source_root),
            "source_three_scope_manifest_sha256":sha256_file(main_manifest_path),
            "source_three_scope_report_sha256":sha256_file(main_report_path),
            "source_three_scope_provenance_sha256":sha256_file(main_provenance_path),
            "source_frozen_dialogue_file_sha256":sha256_file(full_context_paths[0]),
            "source_frozen_dialogue_sha256":sha256_bytes(
                canonical_json(full_dialogue).encode()),
            "source_full_span_catalog_file_sha256":sha256_file(full_catalog_paths[0]),
            "source_full_span_catalog_sha256":span_catalog_sha256(catalogs[0]),
            "source_graph_scene_contracts_sha256":sha256_file(SOURCE_GRAPH_CONTRACTS),
            "parent_turn_count":parent_count, "child_turn_count":child_count,
            "full_turn_count":len(full_dialogue),
            "lineage_views":view_sources}}


def _span_stage_membership(view: Mapping[str, Any],
                           contracts: list[dict[str, Any]]) -> dict[str, list[str]]:
    sessions = {f"session_{int(contract['scene_index']):02d}":contract
                for contract in contracts}
    result = {}
    for span in view["span_catalog"]["spans"]:
        contract = sessions.get(str(span["session_id"]))
        result[span["span_id"]] = (list(contract["assigned_event_stage_ids"])
            if contract is not None else [])
    return result


def _placeholder_relation(arc_id: str, view: Mapping[str, Any],
                          candidate_index: int) -> dict[str, Any]:
    parent = [view["selected_parent_span_ids"][0]]
    child = [view["selected_child_span_ids"][0]]
    return {"relation_bundle_version":"preflight_placeholder",
        "arc_id":arc_id, "candidate_index":candidate_index,
        "relation_id":f"{arc_id}_R{candidate_index + 1}",
        "parent_condition_span_ids":parent,
        "child_consequence_span_ids":child,
        "canonical_support_span_ids":parent + child,
        "relation_type":"preflight", "relation_description":"preflight",
        "joint_dependency_explanation":"preflight",
        "source_lineage_view_sha256":view["view_sha256"]}


def _preflight(*, source: dict[str, Any], tokenizer: Any,
               solver_path: Path, tokenizer_path: Path,
               api_max_tokens: int,
               input_mode: str = LINEAGE_SPAN_INPUT) -> dict[str, Any]:
    if not solver_path.is_dir():
        raise RuntimeError("accepted_stage2_solver_missing")
    if not tokenizer_path.is_dir():
        raise RuntimeError("solver_tokenizer_missing")
    prompt_rows = []
    for arc_id in ARC_ORDER:
        view = source["views"][arc_id]
        membership = _span_stage_membership(view, source["graph_scene_contracts"])
        prior = None
        for index in (0, 1):
            relation = _placeholder_relation(arc_id, view, index)
            relation_sha = sha256_bytes(canonical_json(relation).encode())
            prompts = [("relation_miner", _relation_prompt(
                input_mode=input_mode, arc_id=arc_id, candidate_index=index,
                view=view, full_context=source["full_context"],
                membership=membership, prior=prior))]
            qa = {"question_answer_object_version":"preflight_placeholder",
                "qa_id":f"{arc_id}_R{index + 1}_QA",
                "question":"How did the frozen parent condition affect the child outcome?",
                "correct_answer_text":"The child outcome revised the parent condition.",
                "answer_support_span_ids":relation["canonical_support_span_ids"],
                "reasoning_steps":[], "relation_id":relation["relation_id"],
                "relation_bundle_sha256":relation_sha,
                "source_lineage_view_sha256":view["view_sha256"]}
            qa_sha = sha256_bytes(canonical_json(qa).encode())
            prompts.append(("qa_writer", _qa_prompt(input_mode=input_mode,
                view=view, full_context=source["full_context"],
                relation=relation, relation_sha=relation_sha)))
            prompts.append(("distractor_writer_attempt0", _distractor_prompt(
                input_mode=input_mode, view=view,
                full_context=source["full_context"], relation=relation,
                relation_sha=relation_sha, qa=qa, qa_sha=qa_sha,
                attempt_index=0, feedback=None)))
            prompts.append(("distractor_writer_attempt1", _distractor_prompt(
                input_mode=input_mode, view=view,
                full_context=source["full_context"], relation=relation,
                relation_sha=relation_sha, qa=qa, qa_sha=qa_sha,
                attempt_index=1, feedback={"previous_distractors":["one", "two", "three"],
                    "previous_final_decision":"bad_ground",
                    "previous_grounding_diagnosis":"bad_ground_semantic_leakage"})))
            for stage, prompt in prompts:
                tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                prompt_rows.append({"arc_id":arc_id, "candidate_index":index,
                    "stage":stage, "rendered_prompt_tokens":tokens,
                    "max_generation_tokens":api_max_tokens,
                    "total_tokens":tokens + api_max_tokens,
                    "backend_context_limit":32768,
                    "decision":"pass" if tokens + api_max_tokens <= 32768
                        else "reject_context_overflow", "truncated":False})
            prior = relation
    if any(row["decision"] != "pass" for row in prompt_rows):
        raise RuntimeError("three_stage_api_prompt_preflight_failed")
    example_qa = materialize_three_stage_mcq(question_answer=qa,
        distractors=["Alternative consequence one", "Alternative consequence two",
                     "Alternative consequence three"], target_letter="A",
        span_catalog=source["full_catalog"])["materialized_qa"]
    if example_qa is None:
        raise RuntimeError("three_stage_solver_preflight_materialization_failed")
    solver_prompt = render_solver_prompt(tokenizer, example_qa,
        source["full_context"]["combined_dialogue"])
    solver_tokens = len(tokenizer.encode(solver_prompt, add_special_tokens=False))
    solver_preflight = {"rendered_prompt_tokens":solver_tokens,
        "reserved_generation_tokens":256, "total_tokens":solver_tokens + 256,
        "solver_max_model_len":16384,
        "decision":"pass" if solver_tokens + 256 <= 16384 else
            "reject_context_overflow", "truncated":False}
    if solver_preflight["decision"] != "pass":
        raise RuntimeError("three_stage_solver_context_preflight_failed")
    return {"manifest_preflight":"pass", "api_budget_preflight":"pass",
        "challenger_input_mode":input_mode,
        "source_attestation_preflight":"pass", "solver_path_preflight":"pass",
        "tokenizer_path_preflight":"pass", "api_prompt_preflights":prompt_rows,
        "solver_context_preflight":solver_preflight,
        "budget":{"normal_relation_miner":4, "normal_qa_writer":4,
            "normal_distractor_writer":8, "format_only_retries":8,
            "physical_paid_api_requests":24}}


def _manifest(*, run_id: str, snapshot: dict[str, Any], project_diff: str,
              source: dict[str, Any], preimplementation_diff: str,
              mode: str = "lineage_three_stage_canary",
              input_mode: str = LINEAGE_SPAN_INPUT) -> dict[str, Any]:
    manifest = _base_manifest(run_id=run_id, mode=mode, snapshot=snapshot,
                              project_diff=project_diff)
    manifest.update({"lineage_three_stage_canary_version":
                         LINEAGE_THREE_STAGE_CANARY_VERSION,
        "pre_implementation_project_diff_sha256":preimplementation_diff,
        **_source_code_fields(input_mode), **source["public"]})
    return manifest


def _solver_root_manifest(*, root: Path, main_manifest: dict[str, Any],
                          relation: Mapping[str, Any], relation_sha: str,
                          qa_obj: Mapping[str, Any], qa_sha: str,
                          materialized_sha: str, mode: str,
                          context: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = dict(main_manifest)
    manifest.update({"run_id":f"{main_manifest['run_id']}-{relation['relation_id']}-{root.name}",
        "mode":mode, "relation_id":relation["relation_id"],
        "relation_bundle_sha256":relation_sha,
        "qa_id":qa_obj["qa_id"], "question_answer_sha256":qa_sha,
        "materialized_qa_sha256":materialized_sha,
        "solver_max_model_len":16384,
        "solver_context_sha256":sha256_bytes(canonical_json(context).encode()),
        "solver_context_turn_count":len(context),
        "accepted_stage2_solver_path":str(DEFAULT_SOLVER),
        "solver_tokenizer_path":str(DEFAULT_TOKENIZER),
        "solver_protocol":{"enable_thinking":False, "temperature":0.7,
            "top_p":0.9, "max_tokens":32, "parser":"boxed_letter_v1",
            "max_parse_retries_per_target":1}})
    return manifest


def _run_diagnostic(*, root: Path, qa: dict[str, Any], context: list[dict[str, Any]],
                    context_mode: str, backend: Any, tokenizer: Any,
                    main_manifest: dict[str, Any], relation: Mapping[str, Any],
                    relation_sha: str, qa_obj: Mapping[str, Any], qa_sha: str,
                    materialized_sha: str) -> dict[str, Any]:
    root.mkdir(parents=True)
    catalog = build_span_catalog(context)
    catalog_ids = {span["span_id"] for span in catalog["spans"]}
    diagnostic_qa = dict(qa)
    diagnostic_qa["evidence_span_ids"] = [span_id for span_id in
        qa["evidence_span_ids"] if span_id in catalog_ids]
    atomic_json(root / "frozen/questioner_smoke_context.json", {
        "schema_version":"lineage_partition_diagnostic_context_v1",
        "context_mode":context_mode, "combined_dialogue":context})
    atomic_json(root / "frozen/questioner_span_catalog.json", catalog)
    atomic_json(root / "frozen/diagnostic_qa.json", diagnostic_qa)
    config = _solver_config(run_root=root)
    manifest = _solver_root_manifest(root=root, main_manifest=main_manifest,
        relation=relation, relation_sha=relation_sha, qa_obj=qa_obj, qa_sha=qa_sha,
        materialized_sha=materialized_sha, mode="lineage_partition_diagnostic",
        context=context)
    manifest["context_mode"] = context_mode
    atomic_json(root / "config/resolved_canary_config.json", config)
    atomic_json(root / "config/run_manifest.json", manifest)
    orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest,
        config=config, data={}, tokenizer=tokenizer, realizer=None,
        questioner=None, solver=backend, backend_mode="local")
    rollout = orchestrator._rollouts(diagnostic_qa, context_mode, "original", 3,
                                     diagnostic_qa["options"])
    report = {"context_mode":context_mode, "qa_id":qa["qa_id"],
        "requested_valid_k":3, "obtained_valid_k":rollout["obtained_valid_rollouts"],
        "verified_rewards":rollout["verified_rewards"], "rollout":rollout,
        **orchestrator.ledger.summary()}
    atomic_json(root / "reports/context_diagnostic.json", report)
    provenance = write_post_run_provenance(root, PROJECT, manifest)
    if provenance["authoritative_release_provenance"] is not True:
        raise RuntimeError("diagnostic_provenance_mixed")
    return report


def _run_solver_attempt(*, root: Path, qa: dict[str, Any],
                        relation: Mapping[str, Any], relation_sha: str,
                        qa_obj: Mapping[str, Any], qa_sha: str,
                        materialized_sha: str, lineage_view: Mapping[str, Any],
                        full_context: dict[str, Any], full_catalog: dict[str, Any],
                        backend: Any, tokenizer: Any,
                        main_manifest: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True)
    full_dialogue = full_context["combined_dialogue"]
    atomic_json(root / "frozen/questioner_smoke_context.json", full_context)
    atomic_json(root / "frozen/questioner_span_catalog.json", full_catalog)
    atomic_json(root / "frozen/solver_qa.json", qa)
    config = _solver_config(run_root=root)
    manifest = _solver_root_manifest(root=root, main_manifest=main_manifest,
        relation=relation, relation_sha=relation_sha, qa_obj=qa_obj, qa_sha=qa_sha,
        materialized_sha=materialized_sha, mode="lineage_three_stage_full_solver_audit",
        context=full_dialogue)
    manifest["formal_full_context"] = "complete_frozen_parent_plus_child_371_turns"
    atomic_json(root / "config/resolved_canary_config.json", config)
    atomic_json(root / "config/run_manifest.json", manifest)
    orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest,
        config=config, data={}, tokenizer=tokenizer, realizer=None,
        questioner=None, solver=backend, backend_mode="local")
    formal = orchestrator.audit_minimal_candidate(qa)
    contexts = build_relation_diagnostic_contexts(lineage_view=lineage_view)
    diagnostics = {}
    for name in ("parent_only", "child_only", "full_lineage"):
        diagnostics[name] = _run_diagnostic(root=root / f"diagnostics/{name}",
            qa=qa, context=contexts[name], context_mode=f"{name}_diagnostic",
            backend=backend, tokenizer=tokenizer, main_manifest=main_manifest,
            relation=relation, relation_sha=relation_sha, qa_obj=qa_obj,
            qa_sha=qa_sha, materialized_sha=materialized_sha)
    necessity = classify_joint_necessity_v1(
        parent_rewards=diagnostics["parent_only"]["verified_rewards"],
        child_rewards=diagnostics["child_only"]["verified_rewards"],
        lineage_rewards=diagnostics["full_lineage"]["verified_rewards"])
    permutation = None
    if formal.get("final_decision") == "bad_ground":
        if formal["initial_no_context"].get("verified_rewards") == [1, 1, 1]:
            permutation = orchestrator.diagnose_no_context_permutations(
                qa, formal["initial_no_context"])
        else:
            permutation = orchestrator.diagnose_adaptive_bad_ground_permutations(
                qa, formal["initial_no_context"])
    report = {"qa_id":qa["qa_id"], "relation_id":relation["relation_id"],
        "formal_solver":formal, "context_diagnostics":diagnostics,
        "joint_necessity":necessity, "no_context_permutation":permutation,
        "final_decision":formal["final_decision"],
        **orchestrator.ledger.summary()}
    atomic_json(root / "reports/solver_full_audit.json", report)
    provenance = write_post_run_provenance(root, PROJECT, manifest)
    if provenance["authoritative_release_provenance"] is not True:
        raise RuntimeError("solver_provenance_mixed")
    local_calls = report["request_count"] + sum(
        diagnostic["request_count"] for diagnostic in diagnostics.values())
    report["local_vllm_generations_all_contexts"] = local_calls
    return report


def _usage(ledger: RequestLedger) -> dict[str, int]:
    prompt = completion = total = 0
    for state in ledger.completed():
        usage = (state.get("response_metadata") or {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    return {"prompt_tokens":prompt, "completion_tokens":completion,
            "total_tokens":total}


def _grounding_feedback(solver_report: Mapping[str, Any] | None,
                        minimal_decision: str) -> tuple[str, str]:
    if minimal_decision != "accept" or solver_report is None:
        return "minimal_admission_reject", "minimal_admission_reject"
    final = str(solver_report["final_decision"])
    if final == "bad_ground":
        diagnosis = str((solver_report.get("no_context_permutation") or {}).get(
            "diagnostic_decision") or "bad_ground_permutation_inconclusive")
        return final, diagnosis
    return final, final


def _route_conclusion(summary: Mapping[str, int],
                      candidate_records: list[dict[str, Any]]) -> str:
    if summary.get("retain_frontier", 0) >= 1:
        return "three_stage_fixed_api_challenger_supported"
    if summary.get("materialized_mcqs", 0) < 4:
        return "three_stage_canary_inconclusive"
    joint = [record for record in candidate_records
             if record.get("joint_necessity_label") == "lineage_jointly_required"]
    if joint:
        finals = collections.Counter(record.get("final_decision") for record in joint)
        if finals["bad_ground"] >= finals["reject_easy"]:
            return "joint_relation_generation_supported_distractor_bottleneck"
        return "joint_relation_generation_supported_but_solver_easy"
    return "fixed_api_joint_question_generation_not_supported"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixed three-stage parent-child lineage canary")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--api-config", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--solver-path", type=Path, default=DEFAULT_SOLVER)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--pre-implementation-project-diff-sha256", default=
        "7002a38f8e114b1cb860c35e7ff80169f4d1ef79d4ad88caef486b201138d8f3")
    parser.add_argument("--challenger-input-mode", choices=(
        LINEAGE_SPAN_INPUT, FULL_DIALOGUE_INPUT), default=LINEAGE_SPAN_INPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                               local_files_only=True)
    source = _attest_source(args.source_root)
    api = load_challenger_config(args.api_config)
    api_max_tokens = min(api.questioner_max_tokens, 2048)
    preflight = _preflight(source=source, tokenizer=tokenizer,
        solver_path=args.solver_path, tokenizer_path=args.tokenizer_path,
        api_max_tokens=api_max_tokens, input_mode=args.challenger_input_mode)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if args.run_root is None:
        raise ValueError("--run-root is required unless --preflight-only")
    if args.run_root.exists():
        raise FileExistsError(f"lineage three-stage root exists:{args.run_root}")
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    for arc_id in ARC_ORDER:
        atomic_json(args.run_root / f"frozen/{arc_id.lower()}_lineage_view.json",
                    source["views"][arc_id])
    if args.challenger_input_mode == FULL_DIALOGUE_INPUT:
        atomic_json(args.run_root / "frozen/full_dialogue_challenger_input.json",
                    source["full_context"])
    atomic_json(args.run_root / "frozen/source_attestation.json", source["public"])
    atomic_json(args.run_root / "frozen/api_solver_preflight.json", preflight)
    manifest = _manifest(run_id=args.run_root.name, snapshot=snapshot,
        project_diff=project_diff, source=source,
        preimplementation_diff=args.pre_implementation_project_diff_sha256,
        mode=("lineage_three_stage_full_dialogue_canary" if
              args.challenger_input_mode == FULL_DIALOGUE_INPUT else
              "lineage_three_stage_canary"),
        input_mode=args.challenger_input_mode)
    manifest.update({"accepted_stage2_solver_path":str(args.solver_path),
        "solver_tokenizer_path":str(args.tokenizer_path),
        "solver_max_model_len":16384,
        "challenger_model_identifier":api.model,
        "challenger_temperature":api.temperature,
        "challenger_max_tokens":api_max_tokens,
        "budget":{"relation_miner_normal_requests":4,
            "qa_writer_normal_requests":4, "distractor_writer_normal_requests":8,
            "normal_requests_total":16, "format_only_retries":8,
            "physical_paid_api_requests":24, "realizer":0, "verifier":0,
            "training":0, "semantic_repair":0}})
    config = {"mode":manifest["mode"],
        "challenger_input_mode":args.challenger_input_mode,
        "challenger_full_dialogue_turns":371 if
            args.challenger_input_mode == FULL_DIALOGUE_INPUT else None,
        "event_arcs":["E1", "E4"],
        "relation_candidates_per_arc":2, "distractor_attempts_per_qa":2,
        "solver_effective_max_model_len":16384,
        "solver_enable_thinking":False, "solver_temperature":0.7,
        "solver_top_p":0.9, "solver_max_new_tokens":32,
        "full_context_turns":371, "realizer_generations":0,
        "verifier_generations":0, "training_runs":0,
        "semantic_repair_generations":0}
    atomic_json(args.run_root / "config/resolved_canary_config.json", config)
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    budget = ApiBudget()
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),
                                        api_max_tokens)
    solver_backend = None
    relation_reports: list[dict[str, Any]] = []
    qa_reports: list[dict[str, Any]] = []
    distractor_reports: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    accepted_relations: dict[str, list[dict[str, Any]]] = {arc:[] for arc in ARC_ORDER}
    stopped = False
    for arc_id in ARC_ORDER:
        view = source["views"][arc_id]
        request_view = _request_view(view, source["full_context"],
                                     args.challenger_input_mode)
        membership = _span_stage_membership(view, source["graph_scene_contracts"])
        for candidate_index in (0, 1):
            if stopped:
                break
            relation_id = f"{arc_id}_R{candidate_index + 1}"
            prior = accepted_relations[arc_id][0] if candidate_index == 1 and \
                accepted_relations[arc_id] else None
            relation_prompt = _relation_prompt(input_mode=args.challenger_input_mode,
                arc_id=arc_id, candidate_index=candidate_index, view=view,
                full_context=source["full_context"], membership=membership,
                prior=prior)
            relation_task = run_single_object_task(run_root=args.run_root,
                ledger=ledger, backend=backend, budget=budget,
                stage="relation_miner", logical_request_id=f"relation-{relation_id}",
                semantic_prompt=relation_prompt, lineage_view=request_view,
                validator=lambda raw, arc_id=arc_id, candidate_index=candidate_index,
                    view=view:validate_relation_candidate(raw, arc_id=arc_id,
                        candidate_index=candidate_index, lineage_view=view,
                        prior_support_sets=[item["canonical_support_span_ids"]
                            for item in accepted_relations[arc_id]]),
                schema_retry_decisions=("relation_schema_invalid",))
            relation_reports.append({key:value for key,value in relation_task.items()
                if key not in {"accepted_raw", "accepted_validation"}})
            if not relation_task["accepted"]:
                candidate_records.append({"arc_id":arc_id,
                    "relation_id":relation_id,
                    "relation_candidate_index":candidate_index,
                    "stage_exit":relation_task["final_decision"],
                    "materialized":False, "solver_evaluated":False,
                    "final_decision":relation_task["final_decision"]})
                continue
            relation_result = relation_task["accepted_validation"]
            relation = relation_result["relation_bundle"]
            relation_sha = relation_result["relation_bundle_sha256"]
            accepted_relations[arc_id].append(relation)
            relation_path = args.run_root / f"frozen/relation_bundles/{relation_id}.json"
            atomic_json(relation_path, relation)
            relation_reports[-1].update({"relation_bundle_path":str(
                relation_path.relative_to(args.run_root)),
                "relation_bundle_file_sha256":sha256_file(relation_path),
                "relation_bundle_sha256":relation_sha})

            qa_prompt = _qa_prompt(input_mode=args.challenger_input_mode,
                view=view, full_context=source["full_context"],
                relation=relation, relation_sha=relation_sha)
            qa_task = run_single_object_task(run_root=args.run_root, ledger=ledger,
                backend=backend, budget=budget, stage="qa_writer",
                logical_request_id=f"qa-writer-{relation_id}",
                semantic_prompt=qa_prompt, lineage_view=request_view,
                validator=lambda raw, relation=relation, relation_sha=relation_sha,
                    view=view:validate_question_answer_object(raw,
                        relation_bundle=relation,
                        relation_bundle_sha256=relation_sha, lineage_view=view),
                schema_retry_decisions=("qa_schema_invalid",))
            qa_reports.append({key:value for key,value in qa_task.items()
                if key not in {"accepted_raw", "accepted_validation"}})
            if not qa_task["accepted"]:
                candidate_records.append({"arc_id":arc_id,
                    "relation_id":relation_id,
                    "relation_candidate_index":candidate_index,
                    "relation_support_spans":relation["canonical_support_span_ids"],
                    "stage_exit":qa_task["final_decision"], "materialized":False,
                    "solver_evaluated":False,
                    "final_decision":qa_task["final_decision"]})
                continue
            qa_result = qa_task["accepted_validation"]
            qa_obj = qa_result["question_answer_object"]
            qa_sha = qa_result["question_answer_object_sha256"]
            qa_path = args.run_root / f"frozen/question_answer_objects/{relation_id}.json"
            atomic_json(qa_path, qa_obj)
            qa_reports[-1].update({"question_answer_path":str(
                qa_path.relative_to(args.run_root)),
                "question_answer_file_sha256":sha256_file(qa_path),
                "question_answer_sha256":qa_sha})
            previous_feedback = None
            for distractor_attempt in (0, 1):
                if stopped:
                    break
                distractor_prompt = _distractor_prompt(
                    input_mode=args.challenger_input_mode, view=view,
                    full_context=source["full_context"], relation=relation,
                    relation_sha=relation_sha, qa=qa_obj, qa_sha=qa_sha,
                    attempt_index=distractor_attempt,
                    feedback=previous_feedback)
                logical = f"distractor-{relation_id}-d{distractor_attempt}"
                distractor_task = run_single_object_task(run_root=args.run_root,
                    ledger=ledger, backend=backend, budget=budget,
                    stage="distractor_writer", logical_request_id=logical,
                    semantic_prompt=distractor_prompt, lineage_view=request_view,
                    validator=lambda raw, qa_obj=qa_obj:validate_distractor_object(
                        raw, correct_answer_text=qa_obj["correct_answer_text"]),
                    schema_retry_decisions=("distractor_schema_invalid",))
                d_report = {key:value for key,value in distractor_task.items()
                    if key not in {"accepted_raw", "accepted_validation"}}
                d_report.update({"arc_id":arc_id, "relation_id":relation_id,
                                 "distractor_attempt":distractor_attempt})
                distractor_reports.append(d_report)
                if not distractor_task["accepted"]:
                    candidate_records.append({"arc_id":arc_id,
                        "relation_id":relation_id,
                        "relation_candidate_index":candidate_index,
                        "relation_support_spans":relation["canonical_support_span_ids"],
                        "qa_object":qa_obj, "distractor_attempt":distractor_attempt,
                        "stage_exit":distractor_task["final_decision"],
                        "materialized":False, "solver_evaluated":False,
                        "final_decision":distractor_task["final_decision"]})
                    break
                distractors = distractor_task["accepted_validation"]["distractors"]
                d_path = args.run_root / (
                    f"frozen/distractor_objects/{relation_id}_d{distractor_attempt}.json")
                atomic_json(d_path, {"distractors":distractors})
                d_report.update({"distractor_object_path":str(
                    d_path.relative_to(args.run_root)),
                    "distractor_object_sha256":sha256_file(d_path)})
                placement = deterministic_answer_placement(
                    arc_id, candidate_index, distractor_attempt)
                materialization = materialize_three_stage_mcq(
                    question_answer=qa_obj, distractors=distractors,
                    target_letter=placement["target_letter"],
                    span_catalog=view["span_catalog"])
                materialized = materialization["materialized_qa"]
                materialized_path = args.run_root / (
                    f"frozen/materialized_mcqs/{relation_id}_d{distractor_attempt}.json")
                if isinstance(materialized, dict):
                    atomic_json(materialized_path, materialized)
                    materialized_sha = sha256_file(materialized_path)
                else:
                    materialized_sha = ""
                record: dict[str, Any] = {"arc_id":arc_id,
                    "relation_id":relation_id,
                    "relation_candidate_index":candidate_index,
                    "relation_bundle_sha256":relation_sha,
                    "relation_support_spans":relation["canonical_support_span_ids"],
                    "qa_object":qa_obj, "question_answer_sha256":qa_sha,
                    "distractor_attempt":distractor_attempt,
                    "distractors":distractors, **placement,
                    "minimal_result":{"decision":materialization["minimal_decision"],
                        "reasons":materialization["minimal_reasons"],
                        "diagnostics":materialization["minimal_diagnostics"]},
                    "materialized":isinstance(materialized, dict),
                    "materialized_mcq_path":str(materialized_path.relative_to(
                        args.run_root)) if isinstance(materialized, dict) else None,
                    "materialized_mcq_sha256":materialized_sha or None,
                    "solver_evaluated":False}
                if not isinstance(materialized, dict):
                    record["final_decision"] = "minimal_admission_reject"
                    candidate_records.append(record)
                    previous_feedback = {"previous_distractors":distractors,
                        "previous_final_decision":"minimal_admission_reject",
                        "previous_grounding_diagnosis":"minimal_admission_reject"}
                    continue
                if solver_backend is None:
                    solver_backend = VLLMSolverBackend(model_path=args.solver_path,
                        tokenizer_path=args.tokenizer_path, max_model_len=16384,
                        temperature=0.7, top_p=0.9, max_tokens=32,
                        gpu_memory_utilization=0.80)
                solver_root = args.run_root / (
                    f"solver_runs/{relation_id}/d{distractor_attempt}")
                solver_report = _run_solver_attempt(root=solver_root, qa=materialized,
                    relation=relation, relation_sha=relation_sha, qa_obj=qa_obj,
                    qa_sha=qa_sha, materialized_sha=materialized_sha,
                    lineage_view=view, full_context=source["full_context"],
                    full_catalog=source["full_catalog"], backend=solver_backend,
                    tokenizer=tokenizer, main_manifest=manifest)
                formal = solver_report["formal_solver"]
                permutation = solver_report.get("no_context_permutation") or {}
                record.update({"solver_evaluated":True,
                    "solver_root":str(solver_root),
                    "no_context_rewards":formal["no_context"]["verified_rewards"],
                    "full_context_rewards":(formal.get("full_context") or {}).get(
                        "verified_rewards"),
                    "parent_only_rewards":solver_report["context_diagnostics"][
                        "parent_only"]["verified_rewards"],
                    "child_only_rewards":solver_report["context_diagnostics"][
                        "child_only"]["verified_rewards"],
                    "full_lineage_rewards":solver_report["context_diagnostics"][
                        "full_lineage"]["verified_rewards"],
                    "joint_necessity_label":solver_report["joint_necessity"]["decision"],
                    "joint_necessity":solver_report["joint_necessity"],
                    "permutation_diagnosis":permutation.get("diagnostic_decision"),
                    "final_decision":solver_report["final_decision"],
                    "local_vllm_generations":solver_report[
                        "local_vllm_generations_all_contexts"]})
                candidate_records.append(record)
                if should_stop_after_first_retain_frontier(
                        solver_report["final_decision"]):
                    stopped = True
                    break
                final_label, diagnosis = _grounding_feedback(
                    solver_report, materialization["minimal_decision"])
                previous_feedback = {"previous_distractors":distractors,
                    "previous_final_decision":final_label,
                    "previous_grounding_diagnosis":diagnosis}
        if stopped:
            break

    atomic_json(args.run_root / "reports/relation_miner_summary.json",
                {"items":relation_reports})
    atomic_json(args.run_root / "reports/question_answer_summary.json",
                {"items":qa_reports})
    atomic_json(args.run_root / "reports/distractor_summary.json",
                {"items":distractor_reports})
    atomic_json(args.run_root / "reports/solver_summary.json",
                {"items":candidate_records})
    joint_counter = collections.Counter(record.get("joint_necessity_label")
        for record in candidate_records if record.get("joint_necessity_label"))
    atomic_json(args.run_root / "reports/joint_necessity_summary.json",
                {"counts":dict(joint_counter), "items":[record for record in
                    candidate_records if record.get("joint_necessity_label")]})
    final_counter = collections.Counter(record.get("final_decision")
                                        for record in candidate_records)
    permutation_counter = collections.Counter(record.get("permutation_diagnosis")
        for record in candidate_records if record.get("permutation_diagnosis"))
    summary = {"event_arcs_attempted":len({item["arc_id"] for item in
        candidate_records}) if candidate_records else 0,
        "relation_requests":sum(item["physical_request_count"] for item in relation_reports),
        "relation_valid":len(accepted_relations["E1"]) + len(accepted_relations["E4"]),
        "relation_scope_valid":sum(item["final_decision"] == "relation_accept"
                                    for item in relation_reports),
        "duplicate_relations":sum(item["final_decision"] ==
            "duplicate_relation_candidate" for item in relation_reports),
        "qa_writer_requests":sum(item["physical_request_count"] for item in qa_reports),
        "qa_valid":sum(item["final_decision"] == "qa_accept" for item in qa_reports),
        "qa_relation_drift":sum(item["final_decision"] == "qa_relation_drift"
                                 for item in qa_reports),
        "distractor_requests":sum(item["physical_request_count"]
                                   for item in distractor_reports),
        "distractor_valid":sum(item["final_decision"] == "distractor_accept"
                                for item in distractor_reports),
        "materialized_mcqs":sum(item.get("materialized") is True
                                 for item in candidate_records),
        "minimal_accepted":sum((item.get("minimal_result") or {}).get("decision") ==
                                "accept" for item in candidate_records),
        "solver_evaluated":sum(item.get("solver_evaluated") is True
                                for item in candidate_records),
        "bad_ground":final_counter["bad_ground"],
        "reject_easy":final_counter["reject_easy"],
        "reject_all_wrong":final_counter["reject_all_wrong"],
        "retain_frontier":final_counter["retain_frontier"],
        "bad_ground_semantic_leakage":permutation_counter[
            "bad_ground_semantic_leakage"],
        "bad_ground_position_bias":permutation_counter["bad_ground_position_bias"],
        "bad_ground_mixed_bias":permutation_counter["bad_ground_mixed_bias"],
        "bad_ground_inconclusive":permutation_counter[
            "bad_ground_permutation_inconclusive"],
        "parent_only_sufficient":joint_counter["parent_only_sufficient"],
        "child_only_sufficient":joint_counter["child_only_sufficient"],
        "lineage_jointly_required":joint_counter["lineage_jointly_required"],
        "lineage_not_answerable":joint_counter["lineage_not_answerable"],
        "joint_necessity_inconclusive":joint_counter[
            "joint_necessity_inconclusive"]}
    route = _route_conclusion(summary, candidate_records)
    usage = {**budget.snapshot(), **_usage(ledger),
        "local_vllm_generations":sum(item.get("local_vllm_generations", 0)
                                      for item in candidate_records),
        "realizer_generations":0, "verifier_generations":0,
        "training_runs":0, "semantic_repair_generations":0,
        "ordinary_questioner_generations":0}
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"lineage_three_stage_canary_version":LINEAGE_THREE_STAGE_CANARY_VERSION,
        "challenger_input_mode":args.challenger_input_mode,
        "source_attestation":source["public"], "summary":summary,
        "route_conclusion":route,
        "stopped_after_first_retain_frontier":stopped,
        "candidates":candidate_records, "api_usage":usage,
        "realizer_generations":0, "verifier_generations":0,
        "training_runs":0, "semantic_repair_generations":0,
        "ordinary_questioner_generations":0}
    report_path = args.run_root / "reports/lineage_three_stage_canary.json"
    atomic_json(report_path, report)
    artifacts = artifact_sha256_manifest(args.run_root)
    artifact_path = args.run_root / "reports/artifact_sha256.json"
    atomic_json(artifact_path, artifacts)
    manifest.update({"final_report_sha256":sha256_file(report_path),
        "artifact_manifest_sha256":sha256_file(artifact_path),
        "post_implementation_project_diff_sha256":project_diff,
        "source_snapshot_manifest_sha256":snapshot[
            "source_snapshot_manifest_sha256"],
        "route_conclusion":route, "stopped_after_first_retain_frontier":stopped,
        "usage":usage})
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    # The API ledger is bound to the initial immutable manifest; freeze its SHA explicitly.
    manifest["api_ledger_bound_manifest_sha256"] = ledger.manifest_sha256
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    if provenance["authoritative_release_provenance"] is not True:
        raise RuntimeError("lineage_three_stage_provenance_mixed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
