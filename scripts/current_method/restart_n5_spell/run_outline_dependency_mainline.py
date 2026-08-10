#!/usr/bin/env python3
"""Run the restored outline → dependency → child → constrained-QA mainline."""

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
    FORMAT_ONLY_RETRY_VERSION, artifact_sha256_manifest,
    format_only_retry_suffix_sha256, run_single_object_task,
)
from .live_backends import (
    LiveSemanticRepairBackend, OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .outline_dependency_mainline import (
    CHILD_PROMPT_VERSION, CHILD_SCHEMA_VERSION, MAINLINE_VERSION,
    PLANNER_PROMPT_VERSION, PLANNER_SCHEMA_VERSION, QA_PROMPT_VERSION,
    QA_SCHEMA_VERSION, REALIZER_PROMPT_VERSION, REALIZER_SCHEMA_VERSION,
    apply_anti_leak_gate, assign_outline_dependency_ids, build_child_prompt,
    build_planner_prompt, build_qa_prompt, build_realizer_prompt, contract_shas,
    normalize_outline_dependencies, validate_child_graph, validate_planner_schema,
    validate_qa, validate_realizer,
)
from .run_canary import (
    create_source_snapshot, project_diff_sha256, write_post_run_provenance,
)
from . import run_dependency_conditioned_child_canary as base


PROJECT = base.PROJECT
MODULE_ROOT = base.MODULE_ROOT
DEFAULT_INPUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5/conv-41")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_outline_dependency_constrained_questioner_canary_v1")
PROMPTS = {
    "planner": MODULE_ROOT / "prompts/outline_dependency_planner_v1.md",
    "child": MODULE_ROOT / "prompts/outline_dependency_child_expander_v1.md",
    "realizer": MODULE_ROOT / "prompts/outline_dependency_realizer_v1.md",
    "qa": MODULE_ROOT / "prompts/frozen_outline_dependency_qa_writer_v1.md",
}
PLANNER_MAX_TOKENS = 4096
CHILD_MAX_TOKENS = 2048
QA_MAX_TOKENS = 1024
CONTEXT_LIMIT = 32768
HISTORICAL_ROOTS = (base.DEFAULT_GRAPH_SOURCE_RUN, base.DEFAULT_DIALOGUE_SOURCE_RUN)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(input_root: Path = DEFAULT_INPUT) -> dict[str, Any]:
    outline = _read(input_root / "expansion_outline.json")
    anchors = _read(input_root / "parent_anchor_snapshot.json")
    verifier = _read(input_root / "outline_verifier_report.json")
    parent = base._source()
    if outline.get("outline_role") != "dialogue_expansion_plan_only":
        raise ValueError("outline_role_invalid")
    if verifier.get("decision") != "ACCEPT_FOR_SCENE_REALIZATION_CANARY":
        raise ValueError("outline_verifier_not_accepted")
    if outline["source_parent"]["dialogue_hash"] != parent[
            "source_dialogue_content_sha256"]:
        raise ValueError("outline_parent_dialogue_hash_mismatch")
    if anchors["dialogue_hash"] != parent["source_dialogue_content_sha256"]:
        raise ValueError("anchor_parent_dialogue_hash_mismatch")
    if len(outline.get("planned_event_arcs") or []) != 4 or len(
            outline.get("parent_anchors") or []) != 8:
        raise ValueError("outline_shape_invalid")
    return {"outline": outline, "anchors": anchors, "verifier": verifier,
        "parent_dialogue": parent["parent_dialogue"], "parent": parent,
        "input_root": input_root,
        "outline_sha256": sha256_file(input_root / "expansion_outline.json"),
        "anchor_sha256": sha256_file(input_root / "parent_anchor_snapshot.json"),
        "verifier_sha256": sha256_file(input_root / "outline_verifier_report.json")}


def _view(turns: Sequence[Mapping[str, Any]], material: Mapping[str, Any]) -> dict[str, Any]:
    return base._view(turns, material)


def preflight(source: Mapping[str, Any], tokenizer_path: Path,
              api_config: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    api = load_challenger_config(api_config)
    prompt = build_planner_prompt(source["outline"],
        PROMPTS["planner"].read_text(encoding="utf-8"))
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if tokens + PLANNER_MAX_TOKENS > CONTEXT_LIMIT:
        raise RuntimeError("outline_dependency_context_overflow")
    if api.model != "Qwen/Qwen3-14B":
        raise RuntimeError(f"unexpected_challenger_model:{api.model}")
    return {"decision": "pass", "model": api.model, "temperature": api.temperature,
        "outline_id": source["outline"]["outline_id"], "event_arc_count": 4,
        "parent_anchor_count": 8, "parent_turn_count": 237,
        "child_turn_count_in_generation_input": 0,
        "planner_prompt_tokens": tokens, "planner_max_tokens": PLANNER_MAX_TOKENS,
        "planner_total_tokens": tokens + PLANNER_MAX_TOKENS,
        "context_limit": CONTEXT_LIMIT, "max_child_expansions": 2,
        "questioner_evidence_selection": "forbidden",
        "questioner_distractor_generation": "forbidden"}


def _manifest(run_root: Path, source: Mapping[str, Any], pre: Mapping[str, Any],
              snapshot: Mapping[str, Any], historical: Mapping[str, str]) -> dict[str, Any]:
    return {"run_id": run_root.name, "mode": MAINLINE_VERSION,
        "project_git_commit": _git("rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT), **dict(snapshot),
        "orchestrator_source_sha256": sha256_file(MODULE_ROOT / "orchestrator.py"),
        "validation_source_sha256": sha256_file(MODULE_ROOT / "validation.py"),
        "ledger_source_sha256": sha256_file(MODULE_ROOT / "ledger.py"),
        "run_canary_source_sha256": sha256_file(MODULE_ROOT / "run_canary.py"),
        "mcq_parser_source_sha256": sha256_file(MODULE_ROOT / "core.py"),
        "questioner_prompt_builder_source_sha256": sha256_file(MODULE_ROOT / "prompts.py"),
        "questioner_contract_source_sha256": sha256_file(
            MODULE_ROOT / "questioner_contract.py"),
        "live_backend_source_sha256": sha256_file(MODULE_ROOT / "live_backends.py"),
        "outline_dependency_source_sha256": sha256_file(
            MODULE_ROOT / "outline_dependency_mainline.py"),
        "outline_dependency_runner_source_sha256": sha256_file(Path(__file__)),
        "source_outline_path": str(source["input_root"] / "expansion_outline.json"),
        "source_outline_sha256": source["outline_sha256"],
        "source_parent_anchor_path": str(source["input_root"] /
                                          "parent_anchor_snapshot.json"),
        "source_parent_anchor_sha256": source["anchor_sha256"],
        "source_outline_verifier_sha256": source["verifier_sha256"],
        "source_parent_dialogue_sha256": source["parent"][
            "source_dialogue_content_sha256"],
        "source_parent_row_path": str(source["parent"]["source_parent_row_path"]),
        "source_parent_row_sha256": source["parent"]["source_parent_row_sha256"],
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
        "child_prompt_version": CHILD_PROMPT_VERSION,
        "realizer_prompt_version": REALIZER_PROMPT_VERSION,
        "qa_prompt_version": QA_PROMPT_VERSION,
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "child_schema_version": CHILD_SCHEMA_VERSION,
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "qa_schema_version": QA_SCHEMA_VERSION,
        "prompt_file_sha256": {key: sha256_file(path) for key, path in PROMPTS.items()},
        **contract_shas(), "format_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "historical_tree_sha256_before": dict(historical),
        "challenger": {"model": pre["model"], "temperature": pre["temperature"],
                       "response_format": "json_object"},
        "budget": {"dependency_planner": 1, "child_expander": 2,
            "realizer": 2, "dependency_qa_writer": 2, "format_retries": 7,
            "physical_requests": 14, "distractor": 0, "solver": 0,
            "verifier": 0, "training": 0, "semantic_repair": 0}}


def _review_packet(normalized: Mapping[str, Any]) -> str:
    lines = ["# Outline dependency review", "",
        "Only local-valid dependencies may receive the semantic joint audit.", ""]
    for row in normalized["dependencies"]:
        lines += [f"## {row['dependency_id']} / {row['arc_id']}", "",
                  f"Local validation: `{row.get('local_validation')}`", ""]
        if row.get("local_validation") != "accept":
            lines += [f"Reasons: `{row.get('local_validation_errors', [])}`", ""]
            continue
        lines += [f"Operator: `{row['operator']}`", "",
            f"Parent condition: `{canonical_json(row['parent_condition'])}`", "",
            f"Planned stage/event: `{row['planned_child_stage']}` / "
            f"`{canonical_json(row['planned_child_event'])}`", "",
            f"Joint result: `{canonical_json(row['joint_result'])}`", "",
            "Parent evidence:", ""]
        for turn in row["resolved_parent_turns"]:
            lines.append(f"- `{turn['outline_anchor_id']}` → `{turn['alias']}`: {turn['text']}")
        lines += [""] + [f"- {field}: " for field in base.DEPENDENCY_AUDIT_FIELDS] + ["- notes: ", ""]
    return "\n".join(lines)


def run_stage0(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")
    source = _source(args.input_root); pre = preflight(source, args.tokenizer, args.api_config)
    historical = {str(root): base.tree_manifest_sha256(root) for root in HISTORICAL_ROOTS}
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    manifest = _manifest(args.run_root, source, pre, snapshot, historical)
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", pre)
    atomic_json(args.run_root / "frozen/expansion_outline.json", source["outline"])
    atomic_json(args.run_root / "frozen/parent_anchor_snapshot.json", source["anchors"])
    atomic_json(args.run_root / "frozen/parent_dialogue_with_aliases.json",
                {"turns": source["parent_dialogue"], "parent_count": 237,
                 "child_count_in_generation_input": 0})
    prompt = build_planner_prompt(source["outline"],
        PROMPTS["planner"].read_text(encoding="utf-8"))
    atomic_text(args.run_root / "prompts/stage0_outline_dependency_planner.txt", prompt)
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    api = load_challenger_config(args.api_config)
    task = run_single_object_task(run_root=args.run_root, ledger=ledger,
        backend=LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),
                                          PLANNER_MAX_TOKENS),
        budget=base._budget_from_ledger(ledger), stage="dependency_planner",
        logical_request_id="outline-dependency-planner-4x", semantic_prompt=prompt,
        lineage_view=_view([], {"outline_sha256": source["outline_sha256"]}),
        validator=validate_planner_schema,
        schema_retry_decisions=("outline_dependency_schema_invalid",))
    normalized = (normalize_outline_dependencies(task["accepted_raw"], source["outline"],
        source["parent_dialogue"]) if task["accepted"] else {
        "schema_version": "normalized_outline_dependencies_v1", "dependencies": [],
        "local_valid_count": 0, "empty_count": 0})
    if task["accepted"]:
        atomic_json(args.run_root / "frozen/stage0_model_dependencies.json",
                    task["accepted_raw"])
        atomic_json(args.run_root / "frozen/stage0_assigned_dependencies.json",
                    assign_outline_dependency_ids(task["accepted_raw"]))
    path = args.run_root / "frozen/stage0_dependency_contracts.json"
    atomic_json(path, normalized); source_sha = sha256_file(path)
    atomic_json(args.run_root / "review/stage0_dependency_audit_template.json",
                base._dependency_audit_template(normalized, source_sha))
    atomic_text(args.run_root / "review/stage0_dependency_review_packet.md",
                _review_packet(normalized))
    request = {key: value for key, value in task.items()
               if key not in {"accepted_raw", "accepted_validation"}}
    usage = base._usage(ledger)
    report = {"phase": "stage0_complete_waiting_for_offline_audit",
        "schema_accepted": task["accepted"],
        "local_valid_dependencies": normalized["local_valid_count"],
        "empty_dependencies": normalized["empty_count"],
        "dependency_contracts_sha256": source_sha, "request": request,
        "api_usage": usage}
    atomic_json(args.run_root / "reports/stage0_summary.json", report)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    return report


def _load_dep_audit(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "frozen/stage0_dependency_contracts.json"
    normalized = _read(path)
    audit = base.validate_dependency_audit(_read(
        root / "reports/stage0_dependency_audit.json"), normalized, sha256_file(path))
    return normalized, audit


REALIZATION_FIELDS = base.REALIZATION_AUDIT_FIELDS
FINAL_FIELDS = base.FINAL_AUDIT_FIELDS


def _validate_realization_audit(audit: Any, items: Sequence[Mapping[str, Any]],
                                source_sha: str) -> dict[str, Any]:
    expected = [row["dependency_id"] for row in items if row.get("local_gate", {}).get(
        "decision") == "outline_realization_local_gate_pass"]
    if (not isinstance(audit, dict) or audit.get("audit_version") !=
            base.REALIZATION_AUDIT_VERSION or audit.get(
            "source_realized_child_dialogues_sha256") != source_sha or
            [row.get("dependency_id") for row in audit.get("items", [])] != expected):
        raise ValueError("realization_audit_binding_invalid")
    output = []
    for row in audit["items"]:
        if any(type(row.get(field)) is not bool for field in REALIZATION_FIELDS):
            raise ValueError("realization_audit_incomplete")
        output.append({**row, "final_realization_judgment":
                       base._realization_judgment(row)})
    return {"audit_version": base.REALIZATION_AUDIT_VERSION,
        "source_realized_child_dialogues_sha256": source_sha, "items": output,
        "realization_pass": sum(row["final_realization_judgment"] ==
                                "realization_pass" for row in output)}


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    source = _source(args.input_root); normalized, audit = _load_dep_audit(args.run_root)
    atomic_json(args.run_root / "reports/stage0_dependency_audit.json", audit)
    selected = base.select_dependencies_for_stage1(normalized, audit)
    atomic_json(args.run_root / "frozen/accepted_dependency_contracts.json", selected)
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    client = OpenAICompatibleChallengerClient(load_challenger_config(args.api_config))
    items = []
    for dep in selected:
        did = dep["dependency_id"]
        prompt = build_child_prompt(source["outline"], dep,
            PROMPTS["child"].read_text(encoding="utf-8"))
        atomic_text(args.run_root / f"prompts/stage1_child_expander_{did}.txt", prompt)
        child = run_single_object_task(run_root=args.run_root, ledger=ledger,
            backend=LiveSemanticRepairBackend(client, CHILD_MAX_TOKENS),
            budget=base._budget_from_ledger(ledger), stage="child_expander",
            logical_request_id=f"outline-child-expander-{did}", semantic_prompt=prompt,
            lineage_view=_view([], {"dependency_id": did}),
            validator=lambda raw, dependency=dep: validate_child_graph(raw, dependency),
            schema_retry_decisions=("outline_child_graph_schema_invalid",))
        item = {"dependency_id": did, "child_expander_request": {
            key: value for key, value in child.items()
            if key not in {"accepted_raw", "accepted_validation"}}}
        if not child["accepted"]:
            item["stage1_decision"] = "child_graph_reject"; items.append(item); continue
        child_graph = child["accepted_raw"]
        atomic_json(args.run_root / f"frozen/child_graphs/{did}.json", child_graph)
        real_prompt = build_realizer_prompt(dep, child_graph,
            PROMPTS["realizer"].read_text(encoding="utf-8"))
        atomic_text(args.run_root / f"prompts/stage1_realizer_{did}.txt", real_prompt)
        realizer = run_single_object_task(run_root=args.run_root, ledger=ledger,
            backend=base._FormalRealizerBatchAdapter(client),
            budget=base._budget_from_ledger(ledger), stage="realizer",
            logical_request_id=f"outline-realizer-{did}", semantic_prompt=real_prompt,
            lineage_view=_view([], {"dependency_id": did, "child_graph": child_graph}),
            validator=lambda raw, dependency=dep, graph=child_graph:
                validate_realizer(raw, dependency, graph),
            schema_retry_decisions=("outline_realizer_schema_invalid",))
        item["child_graph"] = child_graph
        item["realizer_request"] = {key: value for key, value in realizer.items()
            if key not in {"accepted_raw", "accepted_validation"}}
        if not realizer["accepted"]:
            item["stage1_decision"] = "realizer_schema_reject"; items.append(item); continue
        gate = apply_anti_leak_gate(realizer["accepted_raw"], dep, child_graph)
        item.update({"realizer_output": realizer["accepted_raw"], "local_gate": gate,
                     "stage1_decision": gate["decision"]})
        items.append(item)
    path = args.run_root / "frozen/realized_child_dialogues.json"
    atomic_json(path, items); source_sha = sha256_file(path)
    template = {"audit_version": base.REALIZATION_AUDIT_VERSION,
        "source_realized_child_dialogues_sha256": source_sha,
        "items": [{"dependency_id": row["dependency_id"],
            **{field: None for field in REALIZATION_FIELDS}, "notes": ""}
            for row in items if row.get("local_gate", {}).get("decision") ==
            "outline_realization_local_gate_pass"]}
    atomic_json(args.run_root / "review/stage1_realization_audit_template.json", template)
    lines = ["# Outline realization review", ""]
    for row in items:
        lines += [f"## {row['dependency_id']}", "",
                  f"Decision: `{row['stage1_decision']}`", ""]
        for turn in row.get("local_gate", {}).get("child_dialogue_with_aliases", []):
            lines.append(f"- `{turn['alias']}` {turn['speaker']}: {turn['text']}")
        lines += [""] + [f"- {field}: " for field in REALIZATION_FIELDS] + ["- notes: ", ""]
    atomic_text(args.run_root / "review/stage1_realization_review_packet.md", "\n".join(lines))
    usage = base._usage(ledger); atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"phase": "stage1_complete_waiting_for_offline_audit",
        "dependency_contract_pass": audit["dependency_contract_pass"],
        "attempted_child_expansions": len(selected), "items": items,
        "realized_child_dialogues_sha256": source_sha, "api_usage": usage}
    atomic_json(args.run_root / "reports/stage1_summary.json", report)
    return report


def _validate_final_audit(audit: Any, qas: Sequence[Mapping[str, Any]],
                          source_sha: str) -> dict[str, Any]:
    expected = [row["dependency_id"] for row in qas if row.get("decision") ==
                "outline_qa_accept"]
    if (not isinstance(audit, dict) or audit.get("audit_version") !=
            base.FINAL_QA_AUDIT_VERSION or audit.get("source_qa_outputs_sha256") !=
            source_sha or [row.get("dependency_id") for row in audit.get(
                "items", [])] != expected):
        raise ValueError("final_audit_binding_invalid")
    output = []
    for row in audit["items"]:
        if any(type(row.get(field)) is not bool for field in FINAL_FIELDS):
            raise ValueError("final_audit_incomplete")
        output.append({**row, "final_joint_judgment": base._final_judgment(row)})
    return {"audit_version": base.FINAL_QA_AUDIT_VERSION,
        "source_qa_outputs_sha256": source_sha, "items": output,
        "lineage_jointly_required_qa": sum(row["final_joint_judgment"] ==
                                           "lineage_jointly_required" for row in output)}


def run_stage2(args: argparse.Namespace) -> dict[str, Any]:
    normalized, _ = _load_dep_audit(args.run_root)
    dep_map = {row["dependency_id"]: row for row in normalized["dependencies"]}
    realized_path = args.run_root / "frozen/realized_child_dialogues.json"
    realized = _read(realized_path)
    audit = _validate_realization_audit(_read(
        args.run_root / "reports/stage1_realization_audit.json"), realized,
        sha256_file(realized_path))
    atomic_json(args.run_root / "reports/stage1_realization_audit.json", audit)
    by_id = {row["dependency_id"]: row for row in realized}
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(
        load_challenger_config(args.api_config)), QA_MAX_TOKENS)
    qas, frozen_inputs = [], []
    for item in audit["items"]:
        if item["final_realization_judgment"] != "realization_pass":
            continue
        did = item["dependency_id"]; dep = dep_map[did]
        parent_turns = dep["resolved_parent_turns"]
        child_turns = by_id[did]["local_gate"]["planned_event_evidence_turns"]
        frozen = {"dependency_id": did, "parent_turns": parent_turns,
                  "child_turns": child_turns, "joint_result": dep["joint_result"]}
        frozen_inputs.append(frozen)
        prompt = build_qa_prompt(dep, parent_turns, child_turns,
            PROMPTS["qa"].read_text(encoding="utf-8"))
        atomic_text(args.run_root / f"prompts/stage2_qa_writer_{did}.txt", prompt)
        task = run_single_object_task(run_root=args.run_root, ledger=ledger,
            backend=backend, budget=base._budget_from_ledger(ledger),
            stage="dependency_qa_writer", logical_request_id=f"outline-qa-writer-{did}",
            semantic_prompt=prompt, lineage_view=_view(
                [*parent_turns, *child_turns], frozen),
            validator=lambda raw, dependency=dep: validate_qa(raw, dependency),
            schema_retry_decisions=("outline_qa_schema_invalid",))
        qas.append({"dependency_id": did,
            "decision": "outline_qa_accept" if task["accepted"] else task["final_decision"],
            "qa": task["accepted_raw"],
            "fixed_parent_evidence_aliases": [turn["alias"] for turn in parent_turns],
            "fixed_child_evidence_aliases": [turn["alias"] for turn in child_turns],
            "request": {key: value for key, value in task.items()
                if key not in {"accepted_raw", "accepted_validation"}}})
    atomic_json(args.run_root / "frozen/frozen_qa_inputs.json", frozen_inputs)
    path = args.run_root / "frozen/qa_writer_outputs.json"
    atomic_json(path, qas); source_sha = sha256_file(path)
    template = {"audit_version": base.FINAL_QA_AUDIT_VERSION,
        "source_qa_outputs_sha256": source_sha,
        "items": [{"dependency_id": row["dependency_id"],
            **{field: None for field in FINAL_FIELDS}, "notes": ""}
            for row in qas if row["decision"] == "outline_qa_accept"]}
    atomic_json(args.run_root / "review/final_joint_necessity_audit_template.json", template)
    lines = ["# Constrained QA joint-necessity review", ""]
    for row in qas:
        lines += [f"## {row['dependency_id']}", "", "```json",
                  json.dumps(row["qa"], ensure_ascii=False, indent=2), "```", ""]
        lines += [f"- {field}: " for field in FINAL_FIELDS] + ["- notes: ", ""]
    atomic_text(args.run_root / "review/final_qa_review_packet.md", "\n".join(lines))
    usage = base._usage(ledger); atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"phase": "stage2_complete_waiting_for_offline_audit",
        "realization_pass": audit["realization_pass"], "qa_writer_outputs": len(qas),
        "qa_valid": sum(row["decision"] == "outline_qa_accept" for row in qas),
        "qa_outputs_sha256": source_sha, "items": qas, "api_usage": usage}
    atomic_json(args.run_root / "reports/stage2_summary.json", report)
    return report


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    normalized, dep_audit = _load_dep_audit(args.run_root)
    realized_path = args.run_root / "frozen/realized_child_dialogues.json"
    realized = _read(realized_path)
    real_audit = _validate_realization_audit(_read(
        args.run_root / "reports/stage1_realization_audit.json"), realized,
        sha256_file(realized_path))
    qa_path = args.run_root / "frozen/qa_writer_outputs.json"; qas = _read(qa_path)
    final_audit = _validate_final_audit(_read(
        args.run_root / "reports/final_joint_necessity_audit.json"), qas,
        sha256_file(qa_path))
    atomic_json(args.run_root / "reports/final_joint_necessity_audit.json", final_audit)
    manifest = _read(args.run_root / "config/run_manifest.json")
    historical_after = {str(root): base.tree_manifest_sha256(root) for root in HISTORICAL_ROOTS}
    history_ok = historical_after == manifest["historical_tree_sha256_before"]
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    usage = base._usage(ledger)
    result = {"canary_version": MAINLINE_VERSION,
        "outline_input_restored": True, "flat_semantic_graph_used_for_planning": False,
        "parent_turn_count": 237, "child_turn_count_in_generation_input": 0,
        "dependency_contract_pass": dep_audit["dependency_contract_pass"],
        "attempted_child_expansions": len(_read(
            args.run_root / "frozen/accepted_dependency_contracts.json")),
        "realization_pass": real_audit["realization_pass"],
        "qa_writer_outputs": len(qas),
        "lineage_jointly_required_qa": final_audit["lineage_jointly_required_qa"],
        "questioner_selected_evidence": False,
        "questioner_generated_distractors": False,
        "route_conclusion": ("outline_dependency_questioner_supported" if
            final_audit["lineage_jointly_required_qa"] >= 2 else
            "outline_dependency_questioner_not_yet_supported"),
        "dependency_audit": dep_audit, "realization_audit": real_audit,
        "final_qa_audit": final_audit, "api_usage": usage,
        "historical_tree_sha256_after": historical_after,
        "historical_run_tree_sha_unchanged": history_ok,
        "distractor_generations": 0, "solver_generations": 0,
        "verifier_generations": 0, "training_runs": 0, "semantic_repair": 0}
    atomic_json(args.run_root / "reports/outline_dependency_mainline.json", result)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {"outline_dependency_source_sha_match": sha256_file(MODULE_ROOT /
        "outline_dependency_mainline.py") == manifest["outline_dependency_source_sha256"],
        "outline_dependency_runner_sha_match": sha256_file(Path(__file__)) ==
            manifest["outline_dependency_runner_source_sha256"],
        "source_outline_sha_match": sha256_file(Path(manifest["source_outline_path"])) ==
            manifest["source_outline_sha256"],
        "source_parent_anchor_sha_match": sha256_file(Path(
            manifest["source_parent_anchor_path"])) == manifest[
                "source_parent_anchor_sha256"],
        "historical_run_tree_sha_unchanged": history_ok}
    provenance.update(checks)
    provenance["authoritative_release_provenance"] = bool(provenance.get(
        "authoritative_release_provenance")) and all(checks.values())
    provenance["provenance_mixed"] = not provenance["authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    if provenance["provenance_mixed"]:
        raise RuntimeError("provenance_mixed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
        choices=("preflight", "stage0", "stage1", "stage2", "finalize"))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=base.DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=base.DEFAULT_TOKENIZER)
    args = parser.parse_args(argv); args.input_root = args.input_root.resolve()
    args.run_root = args.run_root.resolve()
    if args.phase == "preflight":
        if args.run_root.exists():
            raise FileExistsError(f"run_root_exists:{args.run_root}")
        result = preflight(_source(args.input_root), args.tokenizer, args.api_config)
    elif args.phase == "stage0": result = run_stage0(args)
    elif args.phase == "stage1": result = run_stage1(args)
    elif args.phase == "stage2": result = run_stage2(args)
    else: result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

