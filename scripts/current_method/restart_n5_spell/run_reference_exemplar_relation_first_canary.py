#!/usr/bin/env python3
"""Run the four-slot alias-only exemplar-guided relation-first canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .ledger import RequestLedger
from .lineage_three_stage_canary import (
    ApiBudget, FORMAT_ONLY_RETRY_VERSION, artifact_sha256_manifest,
    format_only_retry_suffix_sha256, run_single_object_task,
)
from .live_backends import (
    LiveSemanticRepairBackend, OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .reference_exemplar_relation_first import (
    CANARY_VERSION, RELATION_PROMPT_VERSION, RELATION_SCHEMA_VERSION,
    build_relation_miner_prompt, normalize_relation_batch,
    relation_prompt_sha256, relation_schema_sha256,
    validate_relation_batch_schema,
)
from .reference_guided_relation_first import (
    QA_AUDIT_VERSION, QA_PROMPT_VERSION, QA_SCHEMA_VERSION,
    RELATION_AUDIT_VERSION, qa_prompt_sha256, qa_schema_sha256,
    route_conclusion, validate_stage2_audit,
)
from .run_canary import create_source_snapshot, project_diff_sha256, write_post_run_provenance
from . import run_reference_guided_relation_first_canary as base


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_EXEMPLARS = MODULE_ROOT / "prompts/relation_miner_joint_exemplars_v1.md"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_full_dialogue_unrestricted_exemplar_guided_relation_first_canary_v1"
)
STAGE1_MAX_TOKENS = 4096


def preflight(source: Mapping[str, Any], exemplars: str,
              tokenizer_path: Path, api_config: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    api = load_challenger_config(api_config)
    if api.model != "Qwen/Qwen3-14B":
        raise RuntimeError(f"unexpected model:{api.model}")
    prompt = build_relation_miner_prompt(source["turns"], exemplars)
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if tokens + STAGE1_MAX_TOKENS > base.CONTEXT_LIMIT:
        raise RuntimeError("stage1 prompt context overflow")
    return {"decision": "pass", "model": api.model, "temperature": api.temperature,
        "stage1_prompt_tokens": tokens, "stage1_max_tokens": STAGE1_MAX_TOKENS,
        "stage1_total_tokens": tokens + STAGE1_MAX_TOKENS,
        "context_limit": base.CONTEXT_LIMIT, "full_dialogue_turns": 371,
        "parent_turns": 237, "child_turns": 134,
        "full_dialogue_unrestricted": True,
        "qa_specific_evidence_restriction": False,
        "model_returns_quotes": False, "local_program_retrieves_turn_text": True,
        "relation_slots": 4, "normal_request_cap": 5,
        "format_retry_cap": base.FORMAT_RETRY_CAP,
        "physical_request_cap": base.PHYSICAL_CAP,
        "not_single_variable_experiment": True}


def _manifest(run_root: Path, source: Mapping[str, Any], exemplars_path: Path,
              pre: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {"run_id": run_root.name, "mode": CANARY_VERSION,
        "project_git_commit": base._git("rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT), **dict(snapshot),
        "source_run": str(base.DEFAULT_SOURCE_RUN),
        "source_run_manifest_sha256": sha256_file(
            base.DEFAULT_SOURCE_RUN / "config/run_manifest.json"),
        "source_full_dialogue_path": str(source["dialogue_path"]),
        "source_full_dialogue_sha256": sha256_file(source["dialogue_path"]),
        "source_alias_map_sha256": sha256_file(source["alias_map_path"]),
        "source_reference_path": str(source["reference_path"]),
        "source_reference_sha256": sha256_file(source["reference_path"]),
        "relation_exemplars_path": str(exemplars_path),
        "relation_exemplars_sha256": sha256_file(exemplars_path),
        "relation_prompt_version": RELATION_PROMPT_VERSION,
        "relation_prompt_sha256": relation_prompt_sha256(),
        "relation_schema_version": RELATION_SCHEMA_VERSION,
        "relation_schema_sha256": relation_schema_sha256(),
        "relation_audit_version": RELATION_AUDIT_VERSION,
        "qa_prompt_version": QA_PROMPT_VERSION,
        "qa_prompt_sha256": qa_prompt_sha256(),
        "qa_schema_version": QA_SCHEMA_VERSION, "qa_schema_sha256": qa_schema_sha256(),
        "qa_audit_version": QA_AUDIT_VERSION,
        "format_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "relation_contract_source_sha256": sha256_file(
            MODULE_ROOT / "reference_exemplar_relation_first.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "qa_contract_source_sha256": sha256_file(
            MODULE_ROOT / "reference_guided_relation_first.py"),
        "full_dialogue_unrestricted": True, "model_returns_quotes": False,
        "local_program_retrieves_turn_text": True,
        "vague_conclusion_fixed_blacklist": True,
        "duplicate_support_bundle_rejected": True,
        "relation_slots": 4, "max_qa_writer_inputs": 4,
        "strict_reference_only_causal_attribution_allowed": False,
        "challenger": {"model": pre["model"], "temperature": pre["temperature"],
            "stage1_max_tokens": STAGE1_MAX_TOKENS,
            "stage2_max_tokens": base.STAGE2_MAX_TOKENS,
            "response_format": "json_object"},
        "budget": {"relation_miner_normal_requests": 1,
            "qa_writer_normal_requests": 4, "normal_requests_total": 5,
            "format_only_retries": base.FORMAT_RETRY_CAP,
            "physical_paid_api_requests": base.PHYSICAL_CAP,
            "distractor_writer": 0, "solver": 0, "realizer": 0,
            "verifier": 0, "training": 0, "semantic_repair": 0}}


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise FileExistsError(f"run root exists:{args.run_root}")
    source = base._source_inputs(args.dialogue, args.alias_map, args.reference)
    exemplars = args.exemplars.read_text(encoding="utf-8")
    pre = preflight(source, exemplars, args.tokenizer, args.api_config)
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    manifest = _manifest(args.run_root, source, args.exemplars, pre, snapshot)
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", pre)
    atomic_json(args.run_root / "frozen/full_dialogue_with_aliases.json", source["dialogue"])
    atomic_json(args.run_root / "frozen/turn_alias_map.json", source["alias_map"])
    atomic_text(args.run_root / "frozen/relation_miner_joint_exemplars_v1.md", exemplars)
    prompt = build_relation_miner_prompt(source["turns"], exemplars)
    atomic_text(args.run_root / "prompts/stage1_relation_miner.txt", prompt)
    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    budget = ApiBudget(normal_caps=base.NORMAL_CAPS,
        format_retry_cap=base.FORMAT_RETRY_CAP, physical_cap=base.PHYSICAL_CAP)
    api = load_challenger_config(args.api_config)
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),
                                        STAGE1_MAX_TOKENS)
    task = run_single_object_task(run_root=args.run_root, ledger=ledger,
        backend=backend, budget=budget, stage="relation_miner",
        logical_request_id="unrestricted-exemplar-relation-batch-4x",
        semantic_prompt=prompt, lineage_view=base._request_view(source),
        validator=validate_relation_batch_schema,
        schema_retry_decisions=("relation_batch_schema_invalid",))
    public = {key: value for key, value in task.items()
              if key not in {"accepted_raw", "accepted_validation"}}
    normalized = (normalize_relation_batch(task["accepted_raw"], source["aliases"])
                  if task["accepted"] else {"schema_version":
                    "normalized_exemplar_alias_only_relations_v1", "relations": [],
                    "valid_relation_count": 0, "empty_relation_count": 0})
    atomic_json(args.run_root / "reports/stage1_relation_request.json", public)
    normalized_path = args.run_root / "reports/stage1_normalized_relations.json"
    atomic_json(normalized_path, normalized)
    normalized_sha = sha256_file(normalized_path)
    atomic_json(args.run_root / "review/stage1_relation_audit_template.json",
                base._stage1_template(normalized, normalized_sha))
    atomic_text(args.run_root / "review/stage1_relation_review_packet.md",
                base._render_relation_review(normalized))
    usage = base._usage(ledger)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"phase": "stage1_complete_waiting_for_offline_audit",
        "stage1_schema_accepted": task["accepted"],
        "stage1_valid_relations": normalized["valid_relation_count"],
        "stage1_empty_relations": normalized["empty_relation_count"],
        "normalized_relations_sha256": normalized_sha,
        "request": public, "api_usage": usage}
    atomic_json(args.run_root / "reports/stage1_summary.json", report)
    return report


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    if (args.run_root / "reports/reference_exemplar_relation_first_canary.json").exists():
        raise FileExistsError("already finalized")
    stage1 = base._read(args.run_root / "reports/stage1_relation_audit.json")
    qa_summary_path = args.run_root / "reports/stage2_qa_summary.json"
    qa_summary = base._read(qa_summary_path)
    qa_summary_sha = sha256_file(qa_summary_path)
    qa_audit = validate_stage2_audit(base._read(args.stage2_audit),
        qa_summary["qa_objects"], qa_summary_sha)
    atomic_json(args.run_root / "reports/stage2_qa_audit.json", qa_audit)
    valid, joint = stage1["stage1_valid_relations"], stage1["stage1_joint_relations"]
    inputs, qa_joint = qa_summary["stage2_input_count"], qa_audit["qa_joint_preserved"]
    preservation = qa_joint / inputs if inputs else 0.0
    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    usage = base._usage(ledger)
    report = {"canary_version": CANARY_VERSION,
        "configuration_label":
            "full-dialogue unrestricted exemplar-guided alias-only relation-first canary",
        "stage1_relation_slots": 4, "stage1_valid_relations": valid,
        "stage1_joint_relations": joint,
        "stage1_joint_rate": joint / valid if valid else 0.0,
        "stage1_audit_items": stage1["items"],
        "stage2_input_count": inputs, "qa_valid_count": qa_summary["qa_valid_count"],
        "qa_joint_preserved": qa_joint,
        "relation_preservation_rate": preservation,
        "stage2_audit_items": qa_audit["items"],
        "stage1_success_threshold_met": joint >= 2,
        "stage2_success_threshold_met": qa_joint >= 2 and preservation >= .75,
        "route_conclusion": route_conclusion(joint, inputs, qa_joint),
        "api_usage": usage, "ordinary_questioner_generation": 0,
        "distractor_generations": 0, "solver_generations": 0,
        "realizer_generations": 0, "verifier_generations": 0,
        "training_runs": 0, "semantic_repair": 0}
    atomic_json(args.run_root / "reports/reference_exemplar_relation_first_canary.json",
                report)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    manifest = base._read(args.run_root / "config/run_manifest.json")
    prov = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {"source_dialogue_sha_match": sha256_file(base.DEFAULT_DIALOGUE)
                == manifest["source_full_dialogue_sha256"],
        "exemplars_sha_match": sha256_file(DEFAULT_EXEMPLARS)
                == manifest["relation_exemplars_sha256"],
        "relation_contract_sha_match": sha256_file(
            MODULE_ROOT / "reference_exemplar_relation_first.py")
                == manifest["relation_contract_source_sha256"],
        "runner_sha_match": sha256_file(Path(__file__))
                == manifest["runner_source_sha256"]}
    prov.update(checks)
    prov["authoritative_release_provenance"] = all(checks.values())
    prov["provenance_mixed"] = not prov["authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", prov)
    if prov["provenance_mixed"]:
        raise RuntimeError("provenance mixed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "stage1", "stage2", "finalize"),
                        required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dialogue", type=Path, default=base.DEFAULT_DIALOGUE)
    parser.add_argument("--alias-map", type=Path, default=base.DEFAULT_ALIAS_MAP)
    parser.add_argument("--reference", type=Path, default=base.DEFAULT_REFERENCE)
    parser.add_argument("--exemplars", type=Path, default=DEFAULT_EXEMPLARS)
    parser.add_argument("--api-config", type=Path, default=base.DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=base.DEFAULT_TOKENIZER)
    parser.add_argument("--stage1-audit", type=Path)
    parser.add_argument("--stage2-audit", type=Path)
    args = parser.parse_args(argv)
    args.run_root = args.run_root.resolve()
    if args.phase == "preflight":
        if args.run_root.exists():
            raise FileExistsError(f"run root exists:{args.run_root}")
        result = preflight(base._source_inputs(args.dialogue, args.alias_map,
            args.reference), args.exemplars.read_text(encoding="utf-8"),
            args.tokenizer, args.api_config)
    elif args.phase == "stage1":
        result = run_stage1(args)
    elif args.phase == "stage2":
        if args.stage1_audit is None:
            args.stage1_audit = args.run_root / "review/stage1_relation_audit.json"
        result = base.run_stage2(args)
    else:
        if args.stage2_audit is None:
            args.stage2_audit = args.run_root / "review/stage2_qa_audit.json"
        result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
