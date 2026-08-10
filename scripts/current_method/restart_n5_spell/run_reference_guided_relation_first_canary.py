#!/usr/bin/env python3
"""Run the two paid stages of the unrestricted reference-guided canary."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .questioner_joint_reference_ab import alias_lookup
from .reference_guided_relation_first import (
    CANARY_VERSION, MAX_QA_RELATIONS, QA_AUDIT_BOOLEAN_FIELDS,
    QA_AUDIT_VERSION, QA_PROMPT_VERSION, QA_SCHEMA_VERSION,
    STAGE1_AUDIT_BOOLEAN_FIELDS, RELATION_AUDIT_VERSION,
    RELATION_PROMPT_VERSION, RELATION_SCHEMA_VERSION,
    build_qa_writer_prompt, build_relation_miner_prompt,
    extract_relation_reference, freeze_relation, normalize_relation_batch,
    qa_prompt_sha256, qa_schema_sha256, relation_prompt_sha256,
    relation_schema_sha256, route_conclusion, select_stage2_relations,
    validate_qa_writer_object, validate_relation_batch_schema,
    validate_stage1_audit, validate_stage2_audit,
)
from .run_canary import (
    create_source_snapshot, project_diff_sha256, write_post_run_provenance,
)


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_questioner_joint_reference_ab_canary_v1"
)
DEFAULT_DIALOGUE = DEFAULT_SOURCE_RUN / "frozen/full_dialogue_with_aliases.json"
DEFAULT_ALIAS_MAP = DEFAULT_SOURCE_RUN / "frozen/turn_alias_map.json"
DEFAULT_REFERENCE = MODULE_ROOT / "prompts/questioner_joint_reference_v1.md"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_full_dialogue_unrestricted_reference_guided_relation_first_canary_v1"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
STAGE1_MAX_TOKENS = 6144
STAGE2_MAX_TOKENS = 1536
CONTEXT_LIMIT = 32768
NORMAL_CAPS = {"relation_miner": 1, "qa_writer": 4, "distractor_writer": 0}
FORMAT_RETRY_CAP = 5
PHYSICAL_CAP = 10


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_inputs(
    dialogue_path: Path = DEFAULT_DIALOGUE,
    alias_map_path: Path = DEFAULT_ALIAS_MAP,
    reference_path: Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    for path in (dialogue_path, alias_map_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"fixed input missing:{path}")
    dialogue, alias_map = _read(dialogue_path), _read(alias_map_path)
    aliases = alias_lookup(alias_map, dialogue)
    turns = dialogue.get("combined_dialogue")
    if (
        not isinstance(turns, list) or len(turns) != 371
        or dialogue.get("parent_turn_count") != 237
        or dialogue.get("child_turn_count") != 134
    ):
        raise RuntimeError("full dialogue 237/134/371 attestation failed")
    reference_text = reference_path.read_text(encoding="utf-8")
    relation_reference = extract_relation_reference(reference_text)
    return {
        "dialogue": dialogue, "alias_map": alias_map, "aliases": aliases,
        "turns": turns, "reference_text": reference_text,
        "relation_reference": relation_reference,
        "dialogue_path": dialogue_path, "alias_map_path": alias_map_path,
        "reference_path": reference_path,
    }


def _request_view(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "combined_dialogue": [dict(turn) for turn in source["turns"]],
        "span_catalog": {"schema_version": "unused_alias_only_catalog_v1", "spans": []},
        "view_sha256": sha256_file(source["dialogue_path"]),
    }


def _token_count(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def preflight(
    source: Mapping[str, Any], tokenizer_path: Path, api_config_path: Path,
) -> dict[str, Any]:
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"tokenizer missing:{tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    api = load_challenger_config(api_config_path)
    if api.model != "Qwen/Qwen3-14B":
        raise RuntimeError(f"unexpected Challenger model:{api.model}")
    stage1_prompt = build_relation_miner_prompt(
        source["turns"], source["relation_reference"]
    )
    placeholder = {
        "relation_id": "R1", "reasoning_type": "background_requirement_matching",
        "parent_turn_aliases": ["P001"],
        "parent_quotes": [source["aliases"]["P001"]["text"]],
        "child_turn_aliases": ["C001"],
        "child_quotes": [source["aliases"]["C001"]["text"]],
        "parent_unique_fact": "parent fact", "child_unique_fact": "child fact",
        "combined_conclusion": "combined conclusion",
        "missing_from_parent_only": "child fact",
        "missing_from_child_only": "parent fact", "stage1_decision":
            "lineage_jointly_required", "stage1_audit_sha256": "0" * 64,
    }
    stage2_prompt = build_qa_writer_prompt(placeholder)
    rows = []
    for stage, prompt, maximum in (
        ("relation_miner", stage1_prompt, STAGE1_MAX_TOKENS),
        ("qa_writer", stage2_prompt, STAGE2_MAX_TOKENS),
    ):
        tokens = _token_count(tokenizer, prompt)
        row = {
            "stage": stage, "rendered_prompt_tokens": tokens,
            "max_completion_tokens": maximum, "total_tokens": tokens + maximum,
            "context_limit": CONTEXT_LIMIT,
            "decision": "pass" if tokens + maximum <= CONTEXT_LIMIT else
                "reject_context_overflow",
        }
        rows.append(row)
    if any(row["decision"] != "pass" for row in rows):
        raise RuntimeError(f"API prompt context preflight failed:{rows}")
    return {
        "decision": "pass", "model": api.model, "temperature": api.temperature,
        "prompt_preflights": rows, "full_dialogue_turns": 371,
        "parent_turns": 237, "child_turns": 134,
        "full_dialogue_unrestricted": True,
        "qa_specific_evidence_restriction": False,
        "not_single_variable_experiment": True,
        "normal_request_cap": 5, "format_retry_cap": FORMAT_RETRY_CAP,
        "physical_request_cap": PHYSICAL_CAP,
    }


def _manifest(
    run_root: Path, source: Mapping[str, Any], pre: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_root.name, "mode": CANARY_VERSION,
        "project_git_commit": _git("rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT),
        **dict(snapshot),
        "source_run": str(DEFAULT_SOURCE_RUN),
        "source_run_manifest_sha256": sha256_file(
            DEFAULT_SOURCE_RUN / "config/run_manifest.json"
        ),
        "source_full_dialogue_path": str(source["dialogue_path"]),
        "source_full_dialogue_sha256": sha256_file(source["dialogue_path"]),
        "source_alias_map_sha256": sha256_file(source["alias_map_path"]),
        "source_reference_path": str(source["reference_path"]),
        "source_reference_sha256": sha256_file(source["reference_path"]),
        "relation_reference_subset_sha256": sha256_bytes(
            source["relation_reference"].encode("utf-8")
        ),
        "relation_prompt_version": RELATION_PROMPT_VERSION,
        "relation_prompt_sha256": relation_prompt_sha256(),
        "relation_schema_version": RELATION_SCHEMA_VERSION,
        "relation_schema_sha256": relation_schema_sha256(),
        "relation_audit_version": RELATION_AUDIT_VERSION,
        "qa_prompt_version": QA_PROMPT_VERSION,
        "qa_prompt_sha256": qa_prompt_sha256(),
        "qa_schema_version": QA_SCHEMA_VERSION,
        "qa_schema_sha256": qa_schema_sha256(),
        "qa_audit_version": QA_AUDIT_VERSION,
        "format_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "contract_source_sha256": sha256_file(
            MODULE_ROOT / "reference_guided_relation_first.py"
        ),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "ledger_source_sha256": sha256_file(MODULE_ROOT / "ledger.py"),
        "backend_source_sha256": sha256_file(MODULE_ROOT / "live_backends.py"),
        "full_dialogue_unrestricted": True,
        "qa_specific_evidence_restriction": False,
        "reference_and_evidence_pool_changed_together": True,
        "strict_reference_only_causal_attribution_allowed": False,
        "challenger": {
            "model": pre["model"], "temperature": pre["temperature"],
            "stage1_max_tokens": STAGE1_MAX_TOKENS,
            "stage2_max_tokens": STAGE2_MAX_TOKENS,
            "response_format": "json_object",
        },
        "budget": {
            "relation_miner_normal_requests": 1,
            "qa_writer_normal_requests": 4,
            "normal_requests_total": 5,
            "format_only_retries": FORMAT_RETRY_CAP,
            "physical_paid_api_requests": PHYSICAL_CAP,
            "distractor_writer": 0, "solver": 0, "realizer": 0,
            "verifier": 0, "training": 0, "semantic_repair": 0,
        },
    }


def _budget_from_ledger(ledger: RequestLedger) -> ApiBudget:
    budget = ApiBudget(
        normal_caps=NORMAL_CAPS, format_retry_cap=FORMAT_RETRY_CAP,
        physical_cap=PHYSICAL_CAP,
    )
    for state in ledger.completed():
        metadata = state.get("request_metadata") or {}
        stage = str(metadata.get("stage") or "")
        if stage not in NORMAL_CAPS:
            continue
        is_retry = int(metadata.get("attempt_index") or 0) == 1
        if is_retry:
            budget.retry_counts[stage] += 1
        else:
            budget.normal_counts[stage] += 1
        budget.physical_count += 1
    return budget


def _usage(ledger: RequestLedger) -> dict[str, Any]:
    prompt = completion = total = 0
    rows = []
    for state in sorted(ledger.completed(), key=lambda row: row["request_id"]):
        usage = (state.get("response_metadata") or {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
        rows.append({
            "request_id": state["request_id"],
            "stage": (state.get("request_metadata") or {}).get("stage"),
            "attempt_index": (state.get("request_metadata") or {}).get("attempt_index"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": (state.get("response_metadata") or {}).get("finish_reason"),
        })
    return {
        "total_paid_api_requests": len(rows), "prompt_tokens": prompt,
        "completion_tokens": completion, "total_tokens": total,
        "requests": rows, "ordinary_questioner_generation": 0,
        "distractor_generations": 0, "solver_generations": 0,
        "realizer_generations": 0, "verifier_generations": 0,
        "training_runs": 0, "semantic_repair": 0,
    }


def _stage1_template(normalized: Mapping[str, Any], normalized_sha: str) -> dict[str, Any]:
    items = []
    for row in normalized["relations"]:
        if row.get("local_validation") != "accept":
            continue
        items.append({
            "relation_id": row["relation_id"],
            **{field: None for field in STAGE1_AUDIT_BOOLEAN_FIELDS},
            "notes": "",
        })
    return {
        "audit_version": RELATION_AUDIT_VERSION,
        "source_normalized_relations_sha256": normalized_sha,
        "items": items,
    }


def _render_relation_review(normalized: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 1 relation review packet", "",
        "该文件只展示本地结构、alias 和逐字 quote 校验通过的 relation。语义字段尚未审计。", "",
    ]
    for row in normalized["relations"]:
        lines.extend([f"## {row['relation_id']}", "", f"Status: `{row['status']}`", ""])
        if row.get("local_validation") != "accept":
            lines.extend([f"Local validation: `{row.get('local_validation')}`", ""])
            continue
        lines.extend([
            f"Reasoning type: `{row['reasoning_type']}`", "",
            f"Parent unique fact: {row['parent_unique_fact']}", "",
            f"Child unique fact: {row['child_unique_fact']}", "",
            f"Combined conclusion: {row['combined_conclusion']}", "",
            f"Missing from parent-only: {row['missing_from_parent_only']}", "",
            f"Missing from child-only: {row['missing_from_child_only']}", "",
            "### Parent evidence", "",
        ])
        for turn in row["resolved_parent_turns"]:
            lines.append(f"- `{turn['alias']}` {turn['speaker']}: {turn['text']}")
        lines.extend(["", "### Child evidence", ""])
        for turn in row["resolved_child_turns"]:
            lines.append(f"- `{turn['alias']}` {turn['speaker']}: {turn['text']}")
        lines.extend(["", "### Audit", ""])
        for field in STAGE1_AUDIT_BOOLEAN_FIELDS:
            lines.append(f"- {field}: ")
        lines.extend(["- notes: ", ""])
    return "\n".join(lines)


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise FileExistsError(f"run root exists:{args.run_root}")
    source = _source_inputs(args.dialogue, args.alias_map, args.reference)
    pre = preflight(source, args.tokenizer, args.api_config)
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    manifest = _manifest(args.run_root, source, pre, snapshot)
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", pre)
    atomic_json(args.run_root / "frozen/full_dialogue_with_aliases.json", source["dialogue"])
    atomic_json(args.run_root / "frozen/turn_alias_map.json", source["alias_map"])
    atomic_text(args.run_root / "frozen/questioner_joint_reference_v1.md",
                source["reference_text"])
    atomic_text(args.run_root / "frozen/relation_reference_subset.md",
                source["relation_reference"])
    prompt = build_relation_miner_prompt(source["turns"], source["relation_reference"])
    atomic_text(args.run_root / "prompts/stage1_relation_miner.txt", prompt)
    ledger = RequestLedger(
        args.run_root, sha256_file(args.run_root / "config/run_manifest.json"), "live"
    )
    budget = ApiBudget(normal_caps=NORMAL_CAPS, format_retry_cap=FORMAT_RETRY_CAP,
                       physical_cap=PHYSICAL_CAP)
    api = load_challenger_config(args.api_config)
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),
                                        STAGE1_MAX_TOKENS)
    task = run_single_object_task(
        run_root=args.run_root, ledger=ledger, backend=backend, budget=budget,
        stage="relation_miner", logical_request_id="unrestricted-relation-batch-8x",
        semantic_prompt=prompt, lineage_view=_request_view(source),
        validator=validate_relation_batch_schema,
        schema_retry_decisions=("relation_batch_schema_invalid",),
    )
    task_public = {key: value for key, value in task.items()
                   if key not in {"accepted_raw", "accepted_validation"}}
    if task["accepted"]:
        normalized = normalize_relation_batch(task["accepted_raw"], source["aliases"])
    else:
        normalized = {
            "schema_version": "normalized_unrestricted_relation_batch_v1",
            "relations": [], "valid_relation_count": 0, "empty_relation_count": 0,
        }
    atomic_json(args.run_root / "reports/stage1_relation_request.json", task_public)
    normalized_path = args.run_root / "reports/stage1_normalized_relations.json"
    atomic_json(normalized_path, normalized)
    normalized_sha = sha256_file(normalized_path)
    atomic_json(args.run_root / "review/stage1_relation_audit_template.json",
                _stage1_template(normalized, normalized_sha))
    atomic_text(args.run_root / "review/stage1_relation_review_packet.md",
                _render_relation_review(normalized))
    usage = _usage(ledger)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {
        "phase": "stage1_complete_waiting_for_offline_audit",
        "request": task_public,
        "stage1_schema_accepted": task["accepted"],
        "stage1_valid_relations": normalized["valid_relation_count"],
        "stage1_empty_relations": normalized["empty_relation_count"],
        "normalized_relations_sha256": normalized_sha,
        "audit_template": "review/stage1_relation_audit_template.json",
        "review_packet": "review/stage1_relation_review_packet.md",
        "api_usage": usage,
    }
    atomic_json(args.run_root / "reports/stage1_summary.json", report)
    return report


def _stage2_template(qa_objects: Sequence[Mapping[str, Any]], source_sha: str) -> dict[str, Any]:
    return {
        "audit_version": QA_AUDIT_VERSION,
        "source_qa_summary_sha256": source_sha,
        "items": [{
            "relation_id": row["relation_id"],
            **{field: None for field in QA_AUDIT_BOOLEAN_FIELDS},
            "notes": "",
        } for row in qa_objects],
    }


def _render_qa_review(
    frozen_by_id: Mapping[str, Mapping[str, Any]], qa_objects: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# Stage 2 QA preservation review packet", ""]
    for qa in qa_objects:
        relation = frozen_by_id[qa["relation_id"]]
        lines.extend([
            f"## {qa['relation_id']}", "", f"Question: {qa['question']}", "",
            f"Answer: {qa['answer']}", "", "### Parent evidence", "",
        ])
        for alias, quote in zip(relation["parent_turn_aliases"], relation["parent_quotes"]):
            lines.append(f"- `{alias}`: {quote}")
        lines.extend(["", "### Child evidence", ""])
        for alias, quote in zip(relation["child_turn_aliases"], relation["child_quotes"]):
            lines.append(f"- `{alias}`: {quote}")
        lines.extend(["", f"Combined conclusion: {relation['combined_conclusion']}", ""])
        for field in QA_AUDIT_BOOLEAN_FIELDS:
            lines.append(f"- {field}: ")
        lines.extend(["- notes: ", ""])
    return "\n".join(lines)


def run_stage2(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_root.is_dir():
        raise FileNotFoundError(f"stage1 run root missing:{args.run_root}")
    if (args.run_root / "reports/stage2_qa_summary.json").exists():
        raise FileExistsError("stage2 has already been run")
    source = _source_inputs(args.dialogue, args.alias_map, args.reference)
    manifest = _read(args.run_root / "config/run_manifest.json")
    normalized_path = args.run_root / "reports/stage1_normalized_relations.json"
    normalized = _read(normalized_path)
    normalized_sha = sha256_file(normalized_path)
    audit_raw = _read(args.stage1_audit)
    audited = validate_stage1_audit(audit_raw, normalized, normalized_sha)
    stage1_audit_sha = sha256_bytes(canonical_json(audited).encode("utf-8"))
    atomic_json(args.run_root / "reports/stage1_relation_audit.json", audited)
    selected = select_stage2_relations(normalized, audited, MAX_QA_RELATIONS)
    frozen_by_id: dict[str, dict[str, Any]] = {}
    for row in selected:
        frozen = freeze_relation(row, stage1_audit_sha)
        frozen_by_id[row["relation_id"]] = frozen
        atomic_json(args.run_root / f"frozen/passed_relations/{row['relation_id']}.json",
                    frozen)
    ledger = RequestLedger(
        args.run_root, sha256_file(args.run_root / "config/run_manifest.json"), "live"
    )
    budget = _budget_from_ledger(ledger)
    api = load_challenger_config(args.api_config)
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),
                                        STAGE2_MAX_TOKENS)
    request_view = _request_view(source)
    qa_objects: list[dict[str, Any]] = []
    request_reports: list[dict[str, Any]] = []
    for relation_id, frozen in frozen_by_id.items():
        prompt = build_qa_writer_prompt(frozen)
        atomic_text(args.run_root / f"prompts/stage2_{relation_id}.txt", prompt)
        task = run_single_object_task(
            run_root=args.run_root, ledger=ledger, backend=backend, budget=budget,
            stage="qa_writer", logical_request_id=f"frozen-relation-qa-{relation_id}",
            semantic_prompt=prompt, lineage_view=request_view,
            validator=lambda raw, frozen=frozen: validate_qa_writer_object(raw, frozen),
            schema_retry_decisions=("qa_schema_invalid",),
        )
        public = {key: value for key, value in task.items()
                  if key not in {"accepted_raw", "accepted_validation"}}
        public["relation_id"] = relation_id
        request_reports.append(public)
        if task["accepted"]:
            qa = task["accepted_validation"]["qa_object"]
            qa_objects.append(qa)
            atomic_json(args.run_root / f"frozen/question_answer_objects/{relation_id}.json",
                        qa)
    qa_summary = {
        "phase": "stage2_complete_waiting_for_offline_audit",
        "stage1_audit_sha256": stage1_audit_sha,
        "stage1_valid_relations": audited["stage1_valid_relations"],
        "stage1_joint_relations": audited["stage1_joint_relations"],
        "stage2_selected_relation_ids": list(frozen_by_id),
        "stage2_input_count": len(frozen_by_id),
        "qa_valid_count": len(qa_objects),
        "qa_objects": qa_objects,
        "requests": request_reports,
    }
    qa_summary_path = args.run_root / "reports/stage2_qa_summary.json"
    atomic_json(qa_summary_path, qa_summary)
    qa_summary_sha = sha256_file(qa_summary_path)
    atomic_json(args.run_root / "review/stage2_qa_audit_template.json",
                _stage2_template(qa_objects, qa_summary_sha))
    atomic_text(args.run_root / "review/stage2_qa_review_packet.md",
                _render_qa_review(frozen_by_id, qa_objects))
    atomic_json(args.run_root / "reports/api_usage.json", _usage(ledger))
    return {**qa_summary, "qa_summary_sha256": qa_summary_sha,
            "audit_template": "review/stage2_qa_audit_template.json"}


def _report_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Full-dialogue unrestricted reference-guided relation-first canary", "",
        "本实验同时加入 reference 并取消 E1/E4 QA-specific evidence 限制，不能做 reference 单变量归因。", "",
        f"- Stage 1 valid relations: `{report['stage1_valid_relations']}`",
        f"- Stage 1 jointly required: `{report['stage1_joint_relations']}`",
        f"- Stage 1 joint rate: `{report['stage1_joint_rate']}`", "",
        f"- Stage 2 Writer inputs: `{report['stage2_input_count']}`",
        f"- Stage 2 valid outputs: `{report['qa_valid_count']}`",
        f"- Final joint QA: `{report['qa_joint_preserved']}`",
        f"- Relation preservation rate: `{report['relation_preservation_rate']}`", "",
        f"Route conclusion: `{report['route_conclusion']}`", "",
        "未生成 distractors，未运行 Solver。", "",
    ])


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    if (args.run_root / "reports/reference_guided_relation_first_canary.json").exists():
        raise FileExistsError("canary already finalized")
    stage1 = _read(args.run_root / "reports/stage1_relation_audit.json")
    qa_summary_path = args.run_root / "reports/stage2_qa_summary.json"
    qa_summary = _read(qa_summary_path)
    qa_summary_sha = sha256_file(qa_summary_path)
    qa_audit_raw = _read(args.stage2_audit)
    qa_audit = validate_stage2_audit(
        qa_audit_raw, qa_summary["qa_objects"], qa_summary_sha
    )
    atomic_json(args.run_root / "reports/stage2_qa_audit.json", qa_audit)
    stage1_valid = int(stage1["stage1_valid_relations"])
    stage1_joint = int(stage1["stage1_joint_relations"])
    stage2_inputs = int(qa_summary["stage2_input_count"])
    qa_joint = int(qa_audit["qa_joint_preserved"])
    preservation = qa_joint / stage2_inputs if stage2_inputs else 0.0
    ledger = RequestLedger(
        args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live"
    )
    usage = _usage(ledger)
    report = {
        "canary_version": CANARY_VERSION,
        "configuration_label":
            "full-dialogue unrestricted reference-guided relation-first canary",
        "strict_reference_only_causal_attribution_allowed": False,
        "changed_factors": [
            "joint-reference principles added to Relation Miner",
            "E1/E4 QA-specific selectable-evidence restriction removed",
        ],
        "stage1_relation_slots": 8,
        "stage1_valid_relations": stage1_valid,
        "stage1_joint_relations": stage1_joint,
        "stage1_joint_rate": stage1_joint / stage1_valid if stage1_valid else 0.0,
        "stage1_audit_items": stage1["items"],
        "stage2_selection_rule": "first four Stage-1-passed relations in R1-R8 order",
        "stage2_input_count": stage2_inputs,
        "qa_valid_count": qa_summary["qa_valid_count"],
        "qa_joint_preserved": qa_joint,
        "relation_preservation_rate": preservation,
        "stage2_audit_items": qa_audit["items"],
        "stage1_success_threshold_met": stage1_joint >= 2,
        "stage2_success_threshold_met": qa_joint >= 2 and preservation >= 0.75,
        "route_conclusion": route_conclusion(stage1_joint, stage2_inputs, qa_joint),
        "api_usage": usage,
        "distractor_generations": 0, "solver_generations": 0,
        "ordinary_questioner_generation": 0, "realizer_generations": 0,
        "verifier_generations": 0, "training_runs": 0, "semantic_repair": 0,
    }
    atomic_json(args.run_root / "reports/reference_guided_relation_first_canary.json",
                report)
    atomic_text(args.run_root / "reports/reference_guided_relation_first_canary.md",
                _report_markdown(report))
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    manifest = _read(args.run_root / "config/run_manifest.json")
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    custom_checks = {
        "source_dialogue_sha_match": sha256_file(DEFAULT_DIALOGUE)
            == manifest["source_full_dialogue_sha256"],
        "source_reference_sha_match": sha256_file(DEFAULT_REFERENCE)
            == manifest["source_reference_sha256"],
        "contract_source_sha_match": sha256_file(
            MODULE_ROOT / "reference_guided_relation_first.py"
        ) == manifest["contract_source_sha256"],
        "runner_source_sha_match": sha256_file(Path(__file__))
            == manifest["runner_source_sha256"],
    }
    provenance.update(custom_checks)
    provenance["authoritative_release_provenance"] = all(custom_checks.values())
    provenance["provenance_mixed"] = not provenance["authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    if provenance["provenance_mixed"]:
        raise RuntimeError("final provenance mixed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "stage1", "stage2", "finalize"),
                        required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dialogue", type=Path, default=DEFAULT_DIALOGUE)
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--stage1-audit", type=Path)
    parser.add_argument("--stage2-audit", type=Path)
    args = parser.parse_args(argv)
    args.run_root = args.run_root.resolve()
    if args.phase == "preflight":
        if args.run_root.exists():
            raise FileExistsError(f"run root exists:{args.run_root}")
        result = preflight(
            _source_inputs(args.dialogue, args.alias_map, args.reference),
            args.tokenizer, args.api_config,
        )
    elif args.phase == "stage1":
        result = run_stage1(args)
    elif args.phase == "stage2":
        if args.stage1_audit is None:
            args.stage1_audit = args.run_root / "review/stage1_relation_audit.json"
        result = run_stage2(args)
    else:
        if args.stage2_audit is None:
            args.stage2_audit = args.run_root / "review/stage2_qa_audit.json"
        result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
