#!/usr/bin/env python3
"""Run the positional-ID revision of the dependency-conditioned canary.

The semantic contract, parent input, model, sampling settings, and downstream
gates are inherited unchanged.  Stage 0 alone changes its output ownership:
the model emits four array entries without IDs and the local program assigns
D1-D4 by array position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .dependency_conditioned_child_canary import (
    POSITIONAL_DEPENDENCY_BATCH_SCHEMA,
    POSITIONAL_DEPENDENCY_PROMPT_VERSION,
    POSITIONAL_DEPENDENCY_SCHEMA_VERSION,
    assign_dependency_ids_by_position,
    build_positional_dependency_planner_prompt,
    normalize_dependency_batch,
    validate_positional_dependency_batch_schema,
)
from .ledger import RequestLedger
from .lineage_three_stage_canary import (
    artifact_sha256_manifest,
    run_single_object_task,
)
from .live_backends import (
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .run_canary import create_source_snapshot, project_diff_sha256
from . import run_dependency_conditioned_child_canary as base


POSITIONAL_CANARY_VERSION = "dependency_conditioned_child_expansion_positional_id_canary_v1"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_dependency_conditioned_child_expansion_positional_canary_v1"
)
DEFAULT_SHADOW_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_dependency_conditioned_child_expansion_canary_v1_shadow_positional_audit_v1"
)
SOURCE_RUN_ROOT = base.DEFAULT_RUN_ROOT


def positional_preflight(source: Mapping[str, Any], tokenizer_path: Path,
                         api_config: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    api = load_challenger_config(api_config)
    contract = base.PROMPT_FILES["dependency"].read_text(encoding="utf-8")
    prompt = build_positional_dependency_planner_prompt(source, contract)
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if tokens + base.STAGE0_MAX_TOKENS > base.CONTEXT_LIMIT:
        raise RuntimeError("dependency_planner_context_overflow")
    if source["parent_count"] != 237 or source["child_count_in_generation_input"] != 0:
        raise RuntimeError("parent_only_source_preflight_failed")
    if api.model != "Qwen/Qwen3-14B":
        raise RuntimeError(f"unexpected_challenger_model:{api.model}")
    return {"decision": "pass", "model": api.model,
        "temperature": api.temperature, "parent_turn_count": 237,
        "child_turn_count_in_generation_input": 0,
        "parent_graph_event_count": len(source["parent_graph"]["events"]),
        "stage0_prompt_tokens": tokens,
        "stage0_max_tokens": base.STAGE0_MAX_TOKENS,
        "stage0_total_tokens": tokens + base.STAGE0_MAX_TOKENS,
        "context_limit": base.CONTEXT_LIMIT, "max_child_expansions": 2,
        "normal_request_cap": sum(base.NORMAL_CAPS.values()),
        "format_retry_cap": base.FORMAT_RETRY_CAP,
        "physical_request_cap": base.PHYSICAL_CAP,
        "dependency_id_owner": "local_program",
        "dependency_id_assignment_rule": "array_position_1_to_4_maps_to_D1_to_D4"}


def _positional_manifest(run_root: Path, source: Mapping[str, Any],
                         preflight: Mapping[str, Any], snapshot: Mapping[str, Any],
                         historical: Mapping[str, str]) -> dict[str, Any]:
    manifest = base._manifest(run_root, source, preflight, snapshot, historical)
    manifest.update({
        "mode": POSITIONAL_CANARY_VERSION,
        "dependency_prompt_version": POSITIONAL_DEPENDENCY_PROMPT_VERSION,
        "dependency_schema_version": POSITIONAL_DEPENDENCY_SCHEMA_VERSION,
        "dependency_schema_sha256": sha256_bytes(canonical_json(
            POSITIONAL_DEPENDENCY_BATCH_SCHEMA).encode()),
        "dependency_id_owner": "local_program",
        "dependency_id_assignment_rule": "array_position_1_to_4_maps_to_D1_to_D4",
        "positional_runner_source_sha256": sha256_file(Path(__file__)),
        "project_diff_sha256": project_diff_sha256(base.PROJECT),
    })
    return manifest


def run_stage0(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")
    source = base._source()
    preflight = positional_preflight(source, args.tokenizer, args.api_config)
    historical = {str(root): base.tree_manifest_sha256(root) for root in (
        base.DEFAULT_GRAPH_SOURCE_RUN, base.DEFAULT_DIALOGUE_SOURCE_RUN)}
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(base.PROJECT, args.run_root, False)
    manifest = _positional_manifest(args.run_root, source, preflight, snapshot, historical)
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", preflight)
    atomic_json(args.run_root / "frozen/authoritative_parent_graph.json",
                source["parent_graph"])
    atomic_json(args.run_root / "frozen/parent_dialogue_with_aliases.json",
                {"turns": source["parent_dialogue"], "parent_count": 237,
                 "child_count_in_generation_input": 0})
    prompt = build_positional_dependency_planner_prompt(source,
        base.PROMPT_FILES["dependency"].read_text(encoding="utf-8"))
    atomic_text(args.run_root / "prompts/stage0_dependency_planner.txt", prompt)
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    api = load_challenger_config(args.api_config)
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),
                                        base.STAGE0_MAX_TOKENS)
    task = run_single_object_task(run_root=args.run_root, ledger=ledger,
        backend=backend, budget=base._budget_from_ledger(ledger),
        stage="dependency_planner",
        logical_request_id="dependency-planner-positional-batch-4x",
        semantic_prompt=prompt,
        lineage_view=base._view(source["parent_dialogue"], {"parent_only": True,
            "parent_graph_sha": source["parent_graph_projection_sha256"]}),
        validator=validate_positional_dependency_batch_schema,
        schema_retry_decisions=("dependency_schema_invalid",))
    assigned = (assign_dependency_ids_by_position(task["accepted_raw"])
                if task["accepted"] else None)
    normalized = (normalize_dependency_batch(assigned, source) if assigned else
        {"schema_version": "normalized_dependency_contracts_v1",
         "dependencies": [], "local_valid_count": 0, "empty_count": 0})
    if task["accepted"]:
        atomic_json(args.run_root /
            "frozen/stage0_model_dependencies_without_ids.json", task["accepted_raw"])
        atomic_json(args.run_root /
            "frozen/stage0_dependencies_with_local_ids.json", assigned)
    request_report = {key: value for key, value in task.items()
                      if key not in {"accepted_raw", "accepted_validation"}}
    atomic_json(args.run_root / "reports/stage0_dependency_request.json", request_report)
    contracts_path = args.run_root / "frozen/stage0_dependency_contracts.json"
    atomic_json(contracts_path, normalized)
    contracts_sha = sha256_file(contracts_path)
    atomic_json(args.run_root / "review/stage0_dependency_audit_template.json",
                base._dependency_audit_template(normalized, contracts_sha))
    atomic_text(args.run_root / "review/stage0_dependency_review_packet.md",
                base._render_dependency_review(normalized))
    usage = base._usage(ledger)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"phase": "stage0_complete_waiting_for_offline_audit",
        "schema_accepted": task["accepted"],
        "dependency_id_owner": "local_program",
        "dependency_id_assignment_rule": "array_position_1_to_4_maps_to_D1_to_D4",
        "local_valid_dependencies": normalized["local_valid_count"],
        "empty_dependencies": normalized["empty_count"],
        "dependency_contracts_sha256": contracts_sha,
        "request": request_report, "api_usage": usage}
    atomic_json(args.run_root / "reports/stage0_summary.json", report)
    return report


def finalize_stage0_stop(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze a formal zero-pass Stage-0 result without constructing later stages."""
    normalized, audit = base._load_dependency_audit(args.run_root)
    if audit["dependency_contract_pass"] != 0:
        raise ValueError("stage0_stop_requires_zero_dependency_pass")
    atomic_json(args.run_root / "reports/stage0_dependency_audit.json", audit)
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    usage = base._usage(ledger)
    manifest = base._read(args.run_root / "config/run_manifest.json")
    historical_after = {str(root): base.tree_manifest_sha256(root) for root in (
        base.DEFAULT_GRAPH_SOURCE_RUN, base.DEFAULT_DIALOGUE_SOURCE_RUN)}
    unchanged = historical_after == manifest["historical_run_tree_sha256_before"]
    stage0 = base._read(args.run_root / "reports/stage0_summary.json")
    report = {"canary_version": POSITIONAL_CANARY_VERSION,
        "parent_turn_count": 237, "child_turn_count_in_generation_input": 0,
        "stage0_dependency_slots": 4,
        "stage0_local_valid": normalized["local_valid_count"],
        "stage0_dependency_contract_pass": 0,
        "stage0_semantic_quality_identifiable": bool(stage0.get("schema_accepted")),
        "stage0_success_threshold_met": False,
        "attempted_child_expansions": 0, "stage1_realization_pass": 0,
        "qa_writer_outputs": 0, "lineage_jointly_required_qa": 0,
        "route_conclusion": base.derive_route_conclusion(
            stage0_schema_accepted=bool(stage0.get("schema_accepted")),
            dependency_pass=0, realization_pass=0, joint_qa=0),
        "dependency_audit": audit, "realization_audit": None,
        "final_qa_audit": None, "api_usage": usage,
        "stopped_after_stage0_zero_pass": True,
        "historical_run_tree_sha256_after": historical_after,
        "historical_run_tree_sha_unchanged": unchanged,
        "ordinary_questioner_generation": 0, "distractor_generations": 0,
        "solver_generations": 0, "verifier_generations": 0,
        "training_runs": 0, "semantic_repair": 0}
    atomic_json(args.run_root / "reports/dependency_conditioned_child_canary.json", report)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    provenance = base.write_post_run_provenance(args.run_root, base.PROJECT, manifest)
    checks = {
        "contract_source_sha_match": sha256_file(base.MODULE_ROOT /
            "dependency_conditioned_child_canary.py") == manifest["contract_source_sha256"],
        "base_runner_source_sha_match": sha256_file(Path(base.__file__)) ==
            manifest["runner_source_sha256"],
        "positional_runner_source_sha_match": sha256_file(Path(__file__)) ==
            manifest["positional_runner_source_sha256"],
        "historical_run_tree_sha_unchanged": unchanged,
        "parent_count_exact": manifest["parent_turn_count"] == 237,
        "child_generation_input_zero":
            manifest["child_turn_count_in_generation_input"] == 0,
    }
    provenance.update(checks)
    provenance["authoritative_release_provenance"] = all(checks.values()) and bool(
        provenance.get("authoritative_release_provenance"))
    provenance["provenance_mixed"] = not provenance["authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    if provenance["provenance_mixed"]:
        raise RuntimeError("provenance_mixed")
    return report


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    report = base.finalize(args)
    manifest = base._read(args.run_root / "config/run_manifest.json")
    provenance_path = args.run_root / "reports/post_run_provenance.json"
    provenance = base._read(provenance_path)
    positional_match = sha256_file(Path(__file__)) == manifest[
        "positional_runner_source_sha256"]
    provenance["positional_runner_source_sha_match"] = positional_match
    provenance["authoritative_release_provenance"] = bool(
        provenance.get("authoritative_release_provenance")) and positional_match
    provenance["provenance_mixed"] = not provenance["authoritative_release_provenance"]
    atomic_json(provenance_path, provenance)
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    if provenance["provenance_mixed"]:
        raise RuntimeError("provenance_mixed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "stage0", "stage1",
        "stage2", "finalize", "finalize-stage0-stop"), required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=base.DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=base.DEFAULT_TOKENIZER)
    args = parser.parse_args(argv)
    args.run_root = args.run_root.resolve()
    if args.phase == "preflight":
        if args.run_root.exists():
            raise FileExistsError(f"run_root_exists:{args.run_root}")
        result = positional_preflight(base._source(), args.tokenizer, args.api_config)
    elif args.phase == "stage0":
        result = run_stage0(args)
    elif args.phase == "stage1":
        result = base.run_stage1(args)
    elif args.phase == "stage2":
        result = base.run_stage2(args)
    elif args.phase == "finalize-stage0-stop":
        result = finalize_stage0_stop(args)
    else:
        result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
