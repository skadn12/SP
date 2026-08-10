from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .evidence_view_ablation import selected_evidence_diagnostics
from .graph_expansion import compile_graph_scene_contracts
from .input_adapter import load_lineage, read_json
from .ledger import RequestLedger
from .live_backends import (LiveQuestionerBackend, OpenAICompatibleChallengerClient,
                            load_challenger_config)
from .minimal_admission import SPAN_IDS_ONLY_SCHEMA_PROFILE, SPELL_STYLE_MINIMAL_MODE
from .questioner_contract import (build_span_catalog, default_reasoning_contracts,
                                  reasoning_contract_sha256, span_catalog_sha256)
from .run_canary import (create_source_snapshot, project_diff_sha256,
                         select_minimal_solver_qa, write_post_run_provenance)
from .run_cross_dialogue_calibration import (_base_manifest, _questioner_config,
                                              _solver_config)
from .run_full_graph_qa_pilot import attest_graph_source
from .three_scope_canary import (
    CHILD_LOCAL, PARENT_CHILD_JOINT, PARENT_LOCAL, SCHEMA_ONLY_RETRY_VERSION,
    SCOPED_BATCH_JSON_SCHEMA, SCOPE_ASSIGNMENTS, THREE_SCOPE_CANARY_VERSION,
    THREE_SCOPE_CONTRACT_VERSION, THREE_SCOPE_PROMPT_VERSION,
    THREE_SCOPE_SCHEMA_VERSION, build_three_scope_prompt,
    classify_joint_necessity, compile_arc_lineage_views,
    format_only_retry_suffix_hash, run_scoped_questioner_batch,
    schema_only_retry_suffix_hash,
    strip_scope_field, three_scope_template_hash, validate_scope_item,
)
from .validation import validate_config, validate_dual_admission_batch
from .vllm_backends import VLLMSolverBackend
from .orchestrator import CanaryOrchestrator


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
TARGET_LETTERS = {"E1":("A", "B", "C"), "E2":("B", "C", "D"),
                  "E3":("C", "D", "A"), "E4":("D", "A", "B")}
DEFAULT_FEEDBACK = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_cross_dialogue_calibration_5x3_v1/frozen/calibration_feedback_memory.json")


def _source_fields(*, snapshot: dict[str, Any], project_diff: str) -> dict[str, Any]:
    return {**_base_manifest(run_id="placeholder", mode="placeholder",
            snapshot=snapshot, project_diff=project_diff),
        "three_scope_canary_version":THREE_SCOPE_CANARY_VERSION,
        "three_scope_contract_version":THREE_SCOPE_CONTRACT_VERSION,
        "three_scope_prompt_version":THREE_SCOPE_PROMPT_VERSION,
        "three_scope_schema_version":THREE_SCOPE_SCHEMA_VERSION,
        "three_scope_prompt_sha256":three_scope_template_hash(),
        "schema_only_retry_version":SCHEMA_ONLY_RETRY_VERSION,
        "schema_only_retry_suffix_sha256":schema_only_retry_suffix_hash(),
        "format_only_retry_suffix_sha256":format_only_retry_suffix_hash(),
        "native_structured_output_supported":False,
        "native_structured_output_reason":"current backend exposes text or json_object, not top-level array JSON schema",
        "three_scope_canary_source_sha256":sha256_file(
            MODULE_ROOT / "three_scope_canary.py"),
        "three_scope_canary_runner_source_sha256":sha256_file(Path(__file__)),
        "vllm_backend_source_sha256":sha256_file(MODULE_ROOT / "vllm_backends.py")}


def _manifest(*, run_id: str, mode: str, snapshot: dict[str, Any], project_diff: str,
              source: dict[str, Any], feedback_sha: str,
              lineage_view: dict[str, Any] | None = None) -> dict[str, Any]:
    base = _source_fields(snapshot=snapshot, project_diff=project_diff)
    base.update({"run_id":run_id, "mode":mode,
        "source_graph_manifest_sha256":source["source_manifest_sha256"],
        "source_graph_provenance_sha256":source["source_provenance_sha256"],
        "source_graph_accepted_index_sha256":source["source_accepted_index_sha256"],
        "source_graph_contracts_sha256":source["source_graph_contracts_sha256"],
        "feedback_memory_sha256":feedback_sha})
    if lineage_view is not None:
        base.update({"event_arc_id":lineage_view["event_arc_id"],
            "lineage_view_sha256":lineage_view["view_sha256"],
            "lineage_span_catalog_sha256":span_catalog_sha256(
                lineage_view["span_catalog"]),
            "covered_event_stage_ids":lineage_view["covered_event_stage_ids"],
            "assigned_qa_scopes":list(SCOPE_ASSIGNMENTS)})
    return base


def _backend_identity(api: Any) -> dict[str, Any]:
    return {"backend":"openai_compatible_questioner", "model_identifier":api.model,
        "base_url":api.base_url, "temperature":api.temperature,
        "max_tokens":api.questioner_max_tokens,
        "response_format":"text_with_prompt_json_schema"}


def _questioner_arc(*, root: Path, lineage_view: dict[str, Any], source: dict[str, Any],
                    feedback: dict[str, Any], feedback_sha: str, api: Any,
                    backend: LiveQuestionerBackend, tokenizer: Any,
                    snapshot: dict[str, Any], project_diff: str,
                    scene_contracts: list[dict[str, Any]]) -> dict[str, Any]:
    event_id = lineage_view["event_arc_id"]
    letters = TARGET_LETTERS[event_id]
    prompt = build_three_scope_prompt(lineage_view=lineage_view,
        assigned_gold_letters=letters, solver_feedback=feedback["feedback"])
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    total_tokens = prompt_tokens + api.questioner_max_tokens
    preflight = {"rendered_prompt_tokens":prompt_tokens,
        "max_generation_tokens":api.questioner_max_tokens,
        "total_tokens":total_tokens, "backend_context_limit":32768,
        "decision":"pass" if total_tokens <= 32768 else "reject_context_overflow",
        "truncated":False}
    if preflight["decision"] != "pass":
        raise RuntimeError(f"three_scope_prompt_overflow:{event_id}:{preflight}")
    atomic_json(root / "frozen/lineage_view.json", lineage_view)
    atomic_json(root / "frozen/questioner_span_catalog.json", lineage_view["span_catalog"])
    atomic_json(root / "frozen/questioner_prompt_preflight.json", preflight)
    atomic_json(root / "frozen/scope_assignments.json", [
        {"item_index":index, "qa_scope":scope}
        for index, scope in enumerate(SCOPE_ASSIGNMENTS)])
    atomic_json(root / "raw/prompts/semantic_prompt.json", {
        "prompt_version":THREE_SCOPE_PROMPT_VERSION,
        "prompt_sha256":sha256_bytes(prompt.encode()), "prompt":prompt})
    config = _questioner_config(run_root=root, api=api)
    validate_config(config)
    manifest = _manifest(run_id=f"{root.parents[3].name}-{event_id}-questioner",
        mode="three_scope_questioner_arc", snapshot=snapshot,
        project_diff=project_diff, source=source, feedback_sha=feedback_sha,
        lineage_view=lineage_view)
    manifest.update({"target_gold_letters":list(letters),
        "rendered_prompt_sha256":sha256_bytes(prompt.encode()),
        "questioner_prompt_preflight":preflight,
        "challenger_model_identifier":api.model,
        "sampling_parameters":{"temperature":api.temperature,
                               "max_tokens":api.questioner_max_tokens},
        "budget":{"normal_batches":1, "maximum_single_retry":1,
                  "maximum_paid_requests":2, "realizer":0, "verifier":0,
                  "training":0}})
    atomic_json(root / "config/resolved_canary_config.json", config)
    atomic_json(root / "config/run_manifest.json", manifest)
    ledger = RequestLedger(root, sha256_file(root / "config/run_manifest.json"), "live")
    smoke = run_scoped_questioner_batch(ledger=ledger, backend=backend,
        semantic_prompt=prompt, event_arc_id=event_id,
        assigned_gold_letters=letters, lineage_view=lineage_view,
        prompt_protocol=THREE_SCOPE_PROMPT_VERSION,
        backend_identity=_backend_identity(api), budget_limit=2)
    atomic_json(root / "reports/questioner_scope_smoke.json", {
        **{key:value for key,value in smoke.items() if key != "raw_batch"},
        **ledger.summary()})
    if smoke["raw_batch"] is None:
        report = {"event_arc_id":event_id, "batch_decision":smoke["final_decision"],
            "content_admission_evaluated":False, "raw_candidate_count":0,
            "minimal_accepted_count":0, "scope_followed_count":0,
            "solver_eligible_count":0, "items":[], **ledger.summary()}
        atomic_json(root / "reports/three_scope_questioner_batch.json", report)
        provenance = write_post_run_provenance(root, PROJECT, manifest)
        if not provenance["authoritative_release_provenance"]:
            raise RuntimeError("three scope questioner provenance mixed")
        return {"report":report, "raw_batch":[], "sanitized_batch":[],
                "dual":{"items":[]}, "materialized":{}, "manifest":manifest,
                "scope_records":[]}
    raw_batch = smoke["raw_batch"]
    atomic_json(root / "raw/questioner_batch_final.json", raw_batch)
    scope_records = [validate_scope_item(item, item_index=index,
        lineage_view=lineage_view) for index, item in enumerate(raw_batch)]
    sanitized = [strip_scope_field(item) for item in raw_batch]
    contracts = default_reasoning_contracts(letters)
    turn_text = {turn["turn_id"]:turn["text"] for turn in
                 lineage_view["combined_dialogue"]}
    turn_sessions = {turn["turn_id"]:turn["session_id"] for turn in
                     lineage_view["combined_dialogue"]}
    dual = validate_dual_admission_batch(sanitized, list(letters),
        span_catalog=lineage_view["span_catalog"], reasoning_contracts=contracts,
        frozen_turns=turn_text, turn_session_ids=turn_sessions,
        minimal_schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)
    materialized: dict[int, dict[str, Any]] = {}
    items = []
    for admission, scope in zip(dual.get("items", []), scope_records):
        index = admission["item_index"]
        minimal = admission.get(SPELL_STYLE_MINIMAL_MODE) or {}
        qa = minimal.get("materialized_qa")
        scope_followed = scope["decision"] == "qa_scope_followed"
        solver_eligible = (admission.get("solver_eligible") is True and
                           isinstance(qa, dict) and scope_followed)
        record = {"item_index":index, "qa_id":admission.get("qa_id"),
            "qa_scope":SCOPE_ASSIGNMENTS[index], "scope_validation":scope,
            "strict_v7":admission.get("strict_v7"),
            SPELL_STYLE_MINIMAL_MODE:{key:value for key,value in minimal.items()
                if key != "materialized_qa"}, "solver_eligible":solver_eligible,
            "pre_solver_decision":("solver_eligible" if solver_eligible else
                "qa_scope_not_followed" if not scope_followed else
                "minimal_admission_reject")}
        diagnostics = selected_evidence_diagnostics(
            span_ids=sanitized[index].get("answer_support_span_ids") or [],
            catalog=lineage_view["span_catalog"], scene_contracts=scene_contracts)
        record["selection_diagnostics"] = diagnostics
        if isinstance(qa, dict):
            path = root / f"materialized/item_{index:02d}.json"
            atomic_json(path, qa)
            record.update({"materialized_qa_path":str(path.relative_to(root)),
                           "materialized_qa_sha256":sha256_file(path)})
            materialized[index] = qa
        else:
            record.update({"materialized_qa_path":None,
                           "materialized_qa_sha256":None})
        items.append(record)
    report = {"event_arc_id":event_id,
        "batch_decision":"continue_to_solver" if any(
            item["solver_eligible"] for item in items) else "no_solver_eligible_items",
        "content_admission_evaluated":True, "raw_candidate_count":len(raw_batch),
        "minimal_accepted_count":dual.get("minimal_accepted_count", 0),
        "scope_followed_count":sum(item["scope_validation"]["decision"] ==
                                   "qa_scope_followed" for item in items),
        "solver_eligible_count":sum(item["solver_eligible"] for item in items),
        "items":items, "questioner_smoke":{
            key:value for key,value in smoke.items() if key != "raw_batch"},
        **ledger.summary()}
    atomic_json(root / "reports/three_scope_questioner_batch.json", report)
    provenance = write_post_run_provenance(root, PROJECT, manifest)
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("three scope questioner provenance mixed")
    return {"report":report, "raw_batch":raw_batch,
        "sanitized_batch":sanitized, "dual":dual, "materialized":materialized,
        "manifest":manifest, "scope_records":scope_records, "contracts":contracts}


def _run_context_diagnostic(*, root: Path, qa: dict[str, Any], context: list[dict[str, Any]],
                            catalog: dict[str, Any], mode: str, backend: Any,
                            tokenizer: Any, snapshot: dict[str, Any], project_diff: str,
                            source: dict[str, Any], feedback_sha: str,
                            lineage_view: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True)
    context_span_ids = {span["span_id"] for span in catalog["spans"]}
    diagnostic_qa = dict(qa)
    diagnostic_qa["evidence_span_ids"] = [span_id for span_id in
        qa["evidence_span_ids"] if span_id in context_span_ids]
    atomic_json(root / "frozen/questioner_smoke_context.json", {
        "schema_version":"joint_single_partition_diagnostic_v1",
        "lineage":"conv-41", "context_mode":mode, "combined_dialogue":context})
    atomic_json(root / "frozen/questioner_span_catalog.json", catalog)
    atomic_json(root / "frozen/diagnostic_qa.json", diagnostic_qa)
    config = _solver_config(run_root=root)
    manifest = _manifest(run_id=f"{root.parents[5].name}-{lineage_view['event_arc_id']}-{qa['qa_id']}-{mode}",
        mode="joint_partition_solver_diagnostic", snapshot=snapshot,
        project_diff=project_diff, source=source, feedback_sha=feedback_sha,
        lineage_view=lineage_view)
    manifest.update({"qa_id":qa["qa_id"], "qa_scope":PARENT_CHILD_JOINT,
        "context_mode":mode, "context_sha256":sha256_bytes(canonical_json(context).encode()),
        "solver_max_model_len":16384,
        "solver_protocol":{"valid_k":3, "enable_thinking":False,
                           "temperature":0.7, "top_p":0.9,
                           "max_tokens":32, "parser":"boxed_letter_v1"}})
    atomic_json(root / "config/resolved_canary_config.json", config)
    atomic_json(root / "config/run_manifest.json", manifest)
    orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest, config=config,
        data={}, tokenizer=tokenizer, realizer=None, questioner=None, solver=backend,
        backend_mode="local")
    rollout = orchestrator._rollouts(diagnostic_qa, mode, "original", 3,
                                     diagnostic_qa["options"])
    report = {"qa_id":qa["qa_id"], "qa_scope":PARENT_CHILD_JOINT,
        "context_mode":mode, "requested_valid_k":3,
        "obtained_valid_k":rollout["obtained_valid_rollouts"],
        "verified_rewards":rollout["verified_rewards"],
        "stable_correct":rollout["verified_rewards"] == [1, 1, 1],
        "rollout":rollout, **orchestrator.ledger.summary()}
    atomic_json(root / "reports/joint_partition_diagnostic.json", report)
    provenance = write_post_run_provenance(root, PROJECT, manifest)
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("joint partition provenance mixed")
    return report


def _run_solver_item(*, root: Path, event_id: str, item: dict[str, Any],
                     questioner: dict[str, Any], lineage_view: dict[str, Any],
                     full_dialogue: list[dict[str, Any]], full_catalog: dict[str, Any],
                     scene_contracts: list[dict[str, Any]], backend: Any, tokenizer: Any,
                     snapshot: dict[str, Any], project_diff: str, source: dict[str, Any],
                     feedback_sha: str) -> dict[str, Any]:
    index = item["item_index"]
    admission = questioner["dual"]["items"][index]
    base_qa = questioner["materialized"][index]
    _, qa, _ = select_minimal_solver_qa(questioner["sanitized_batch"], admission,
        admission["qa_id"], materialized_qa=base_qa,
        span_catalog=lineage_view["span_catalog"],
        reasoning_contracts=questioner["contracts"])
    root.mkdir(parents=True)
    atomic_json(root / "frozen/questioner_smoke_context.json", {
        "schema_version":"three_scope_full_solver_context_v1", "lineage":"conv-41",
        "context_mode":"complete_frozen_parent_plus_child",
        "combined_dialogue":full_dialogue})
    atomic_json(root / "frozen/questioner_span_catalog.json", full_catalog)
    atomic_json(root / "frozen/solver_qa.json", qa)
    atomic_json(root / "frozen/lineage_view.json", lineage_view)
    config = _solver_config(run_root=root)
    validate_config(config)
    manifest = _manifest(run_id=f"{root.parents[3].name}-{event_id}-{qa['qa_id']}",
        mode="three_scope_full_solver_audit", snapshot=snapshot,
        project_diff=project_diff, source=source, feedback_sha=feedback_sha,
        lineage_view=lineage_view)
    manifest.update({"qa_id":qa["qa_id"], "qa_scope":item["qa_scope"],
        "source_questioner_manifest_sha256":sha256_file(
            root.parents[1] / "questioner/config/run_manifest.json"),
        "materialized_qa_sha256":sha256_bytes(canonical_json(qa).encode()),
        "formal_full_context_sha256":sha256_bytes(canonical_json(full_dialogue).encode()),
        "formal_full_context_turn_count":len(full_dialogue),
        "solver_max_model_len":16384,
        "solver_protocol":{"enable_thinking":False, "temperature":0.7,
            "top_p":0.9, "max_tokens":32, "parser":"boxed_letter_v1",
            "max_parse_retries_per_target":1,
            "full_context":"complete_frozen_parent_plus_child"}})
    atomic_json(root / "config/resolved_canary_config.json", config)
    atomic_json(root / "config/run_manifest.json", manifest)
    orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest, config=config,
        data={}, tokenizer=tokenizer, realizer=None, questioner=None, solver=backend,
        backend_mode="local")
    result = orchestrator.audit_minimal_candidate(qa)
    joint = None
    if item["qa_scope"] == PARENT_CHILD_JOINT:
        parent_ids = set(lineage_view["selected_parent_span_ids"])
        child_ids = set(lineage_view["selected_child_span_ids"])
        parent_turns = [turn for turn in lineage_view["combined_dialogue"]
                        if f"{turn['turn_id']}#S0001" in parent_ids]
        child_turns = [turn for turn in lineage_view["combined_dialogue"]
                       if f"{turn['turn_id']}#S0001" in child_ids]
        parent_catalog = build_span_catalog(parent_turns)
        child_catalog = build_span_catalog(child_turns)
        parent_report = _run_context_diagnostic(root=root / "joint_diagnostics/parent_only",
            qa=qa, context=parent_turns, catalog=parent_catalog,
            mode="parent_only_diagnostic", backend=backend, tokenizer=tokenizer,
            snapshot=snapshot, project_diff=project_diff, source=source,
            feedback_sha=feedback_sha, lineage_view=lineage_view)
        child_report = _run_context_diagnostic(root=root / "joint_diagnostics/child_only",
            qa=qa, context=child_turns, catalog=child_catalog,
            mode="child_only_diagnostic", backend=backend, tokenizer=tokenizer,
            snapshot=snapshot, project_diff=project_diff, source=source,
            feedback_sha=feedback_sha, lineage_view=lineage_view)
        joint = classify_joint_necessity(
            no_context_rewards=(result.get("no_context") or {}).get("verified_rewards"),
            parent_only_rewards=parent_report["verified_rewards"],
            child_only_rewards=child_report["verified_rewards"],
            full_context_rewards=(result.get("full_context") or {}).get("verified_rewards"))
        joint.update({"parent_only":parent_report, "child_only":child_report})
    report = {"event_arc_id":event_id, "qa_id":qa["qa_id"],
        "qa_scope":item["qa_scope"], "scope_validation":item["scope_validation"],
        "selection_diagnostics":selected_evidence_diagnostics(
            span_ids=qa["evidence_span_ids"], catalog=full_catalog,
            scene_contracts=scene_contracts),
        "formal_full_context_mode":"complete_frozen_parent_plus_child",
        "result":result, "joint_necessity":joint,
        "final_decision":result["final_decision"], **orchestrator.ledger.summary(),
        "paid_api_generations":0, "realizer_generations":0,
        "verifier_generations":0, "training_runs":0}
    atomic_json(root / "reports/three_scope_solver_audit.json", report)
    provenance = write_post_run_provenance(root, PROJECT, manifest)
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("three scope solver provenance mixed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Four-arc parent/child/joint Questioner canary")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-graph-root", type=Path, required=True)
    parser.add_argument("--lineage", default="conv-41")
    parser.add_argument("--api-config", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--feedback-source", type=Path, default=DEFAULT_FEEDBACK)
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"three scope root already exists: {args.run_root}")
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
    views = compile_arc_lineage_views(outline=data["outline"],
        parent_turns=source["parent_turns"], child_turns=source["child_turns"],
        scene_contracts=scene_contracts)
    feedback = read_json(args.feedback_source)
    feedback_sha = sha256_file(args.feedback_source)
    atomic_json(args.run_root / "frozen/source_graph_attestation.json", source_public)
    atomic_json(args.run_root / "frozen/lineage_views.json", [
        {key:value for key,value in view.items() if key not in
            {"combined_dialogue", "span_catalog"}} for view in views])
    atomic_json(args.run_root / "frozen/three_scope_batch_schema.json",
                SCOPED_BATCH_JSON_SCHEMA)
    atomic_json(args.run_root / "frozen/calibration_feedback_memory.json", feedback)
    api = load_challenger_config(args.api_config)
    backend = LiveQuestionerBackend(OpenAICompatibleChallengerClient(api))
    questioner_results: dict[str, dict[str, Any]] = {}
    for view in views:
        qroot = args.run_root / f"arcs/{view['event_arc_id']}/questioner"
        qroot.mkdir(parents=True)
        questioner_results[view["event_arc_id"]] = _questioner_arc(
            root=qroot, lineage_view=view, source=source, feedback=feedback,
            feedback_sha=feedback_sha, api=api, backend=backend,
            tokenizer=tokenizer, snapshot=snapshot, project_diff=project_diff,
            scene_contracts=scene_contracts)
    candidates = [(view, item) for view in views
        for item in questioner_results[view["event_arc_id"]]["report"]["items"]
        if item.get("solver_eligible") is True]
    solver_backend = None
    if candidates:
        solver_backend = VLLMSolverBackend(
            model_path=legacy / "accepted_models/solver_base_restart_stage2_merged",
            tokenizer_path=tokenizer_path, max_model_len=16384,
            temperature=0.7, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=0.80)
    full_dialogue = source["parent_turns"] + source["child_turns"]
    full_catalog = build_span_catalog(full_dialogue)
    item_reports = []
    for view in views:
        event_id = view["event_arc_id"]
        questioner = questioner_results[event_id]
        for item in questioner["report"]["items"]:
            if not item.get("solver_eligible"):
                item_reports.append({"event_arc_id":event_id,
                    "qa_id":item.get("qa_id"), "qa_scope":item["qa_scope"],
                    "scope_validation":item["scope_validation"],
                    "selection_diagnostics":item["selection_diagnostics"],
                    "solver_evaluated":False,
                    "final_decision":item["pre_solver_decision"]})
                continue
            sroot = args.run_root / f"arcs/{event_id}/solvers/item_{item['item_index']:02d}"
            solver_report = _run_solver_item(root=sroot, event_id=event_id,
                item=item, questioner=questioner, lineage_view=view,
                full_dialogue=full_dialogue, full_catalog=full_catalog,
                scene_contracts=scene_contracts, backend=solver_backend,
                tokenizer=tokenizer, snapshot=snapshot, project_diff=project_diff,
                source=source, feedback_sha=feedback_sha)
            item_reports.append({"event_arc_id":event_id,
                "qa_id":item["qa_id"], "qa_scope":item["qa_scope"],
                "scope_validation":item["scope_validation"],
                "selection_diagnostics":solver_report["selection_diagnostics"],
                "solver_evaluated":True, "solver_root":str(sroot),
                "no_context_rewards":(solver_report["result"].get("no_context") or {}).get(
                    "verified_rewards"),
                "full_context_rewards":(solver_report["result"].get("full_context") or {}).get(
                    "verified_rewards"),
                "joint_necessity":(solver_report.get("joint_necessity") or {}).get("decision"),
                "final_decision":solver_report["final_decision"]})
    scope_counts = {scope:sum(item["scope_validation"]["decision"] ==
        "qa_scope_followed" for item in item_reports if item["qa_scope"] == scope)
        for scope in SCOPE_ASSIGNMENTS}
    decisions: dict[str, int] = {}
    joint_decisions: dict[str, int] = {}
    for item in item_reports:
        decisions[item["final_decision"]] = decisions.get(item["final_decision"], 0) + 1
        if item.get("joint_necessity"):
            value = item["joint_necessity"]
            joint_decisions[value] = joint_decisions.get(value, 0) + 1
    paid = sum(result["report"].get("paid_api_generation_count", 0)
               for result in questioner_results.values())
    report = {"three_scope_canary_version":THREE_SCOPE_CANARY_VERSION,
        "source_graph_attestation":source_public,
        "summary":{"event_arc_count":4, "raw_candidate_target":12,
            "raw_candidate_count":sum(result["report"]["raw_candidate_count"]
                for result in questioner_results.values()),
            "scope_followed_parent_local":scope_counts[PARENT_LOCAL],
            "scope_followed_child_local":scope_counts[CHILD_LOCAL],
            "scope_followed_joint":scope_counts[PARENT_CHILD_JOINT],
            "solver_evaluated_count":sum(item["solver_evaluated"] for item in item_reports),
            "final_decisions":decisions, "joint_necessity_decisions":joint_decisions,
            "retain_frontier_count":decisions.get("retain_frontier", 0)},
        "questioner_batches":[result["report"] for result in questioner_results.values()],
        "items":item_reports, "paid_questioner_generations":paid,
        "realizer_generations":0, "verifier_generations":0, "training_runs":0}
    report_path = args.run_root / "reports/three_scope_canary.json"
    atomic_json(report_path, report)
    manifest = _manifest(run_id=args.run_root.name, mode="three_scope_questioner_canary",
        snapshot=snapshot, project_diff=project_diff, source=source,
        feedback_sha=feedback_sha)
    manifest.update({"lineage_view_sha256":sha256_bytes(canonical_json([
        view["view_sha256"] for view in views]).encode()),
        "questioner_manifest_sha256":{
            event_id:sha256_file(args.run_root /
                f"arcs/{event_id}/questioner/config/run_manifest.json")
            for event_id in TARGET_LETTERS},
        "report_sha256":sha256_file(report_path),
        "budget":{"normal_questioner_requests":4,
                  "maximum_conditional_retries":4,
                  "maximum_paid_questioner_requests":8,
                  "maximum_solver_qa_items":12,
                  "realizer":0, "verifier":0, "training":0}})
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode":"three_scope_questioner_canary", "event_arc_count":4,
        "questioner_batches":4, "mcqs_per_batch":3,
        "maximum_paid_questioner_requests":8,
        "solver_effective_max_model_len":16384,
        "formal_full_context":"complete_frozen_parent_plus_child",
        "realizer_generations":0, "verifier_generations":0,
        "training_runs":0})
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("three scope canary provenance mixed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
