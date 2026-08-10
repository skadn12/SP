#!/usr/bin/env python3
"""Run the Dependency-Conditioned Child Expansion Canary in audited phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from transformers import AutoTokenizer

from .backends import RealizerRequest
from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .dependency_conditioned_child_canary import (
    CANARY_VERSION, CHILD_EXPANDER_PROMPT_VERSION, CHILD_GRAPH_SCHEMA_VERSION,
    DEPENDENCY_AUDIT_VERSION, DEPENDENCY_PROMPT_VERSION, DEPENDENCY_SCHEMA_VERSION,
    FINAL_QA_AUDIT_VERSION, QA_PROMPT_VERSION, QA_SCHEMA_VERSION,
    REALIZATION_AUDIT_VERSION, REALIZER_PROMPT_VERSION, REALIZER_SCHEMA_VERSION,
    apply_anti_leak_gate, build_child_expander_prompt, build_dependency_planner_prompt,
    build_qa_writer_prompt, build_realizer_prompt, contract_shas,
    normalize_dependency_batch, project_authoritative_parent, validate_child_graph,
    validate_dependency_batch_schema, validate_qa_object, validate_realizer_output,
)
from .input_adapter import load_lineage
from .ledger import RequestLedger
from .lineage_three_stage_canary import (
    ApiBudget, FORMAT_ONLY_RETRY_VERSION, artifact_sha256_manifest,
    format_only_retry_suffix_sha256, run_single_object_task,
)
from .live_backends import (
    LiveRealizerBackend, LiveSemanticRepairBackend, OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .run_canary import create_source_snapshot, project_diff_sha256, write_post_run_provenance


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_dependency_conditioned_child_expansion_canary_v1")
DEFAULT_INPUT_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5")
DEFAULT_LEGACY_ROOT = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_GRAPH_SOURCE_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_spell_full_graph_expansion_v7")
DEFAULT_DIALOGUE_SOURCE_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_questioner_joint_reference_ab_canary_v1")
PROMPT_FILES = {
    "dependency": MODULE_ROOT / "prompts/dependency_planner_v1.md",
    "expander": MODULE_ROOT / "prompts/dependency_conditioned_child_expander_v1.md",
    "qa": MODULE_ROOT / "prompts/frozen_dependency_qa_writer_v1.md",
}
NORMAL_CAPS = {"dependency_planner": 1, "child_expander": 2, "realizer": 2,
               "dependency_qa_writer": 2}
FORMAT_RETRY_CAP = 7
PHYSICAL_CAP = 14
CONTEXT_LIMIT = 32768
STAGE0_MAX_TOKENS = 4096
EXPANDER_MAX_TOKENS = 3072
QA_MAX_TOKENS = 1536

DEPENDENCY_AUDIT_FIELDS = (
    "parent_only_sufficient", "planned_child_only_sufficient", "combined_sufficient",
    "natural_relation", "minimal_parent_evidence",
)
REALIZATION_AUDIT_FIELDS = (
    "parent_fact_restated", "joint_answer_leaked", "single_turn_shortcut",
    "planned_child_event_verbalized", "dialogue_matches_graph",
    "unsupported_realizer_fact", "natural_dialogue",
)
FINAL_AUDIT_FIELDS = (
    "parent_only_can_answer_uniquely", "child_only_can_answer_uniquely",
    "combined_can_answer_uniquely", "single_turn_contains_full_answer",
    "evidence_is_repetitive", "question_leaks_reasoning_step",
    "question_answer_supported", "minimal_evidence", "natural_wording",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def tree_manifest_sha256(root: Path) -> str:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            records.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json(records).encode())


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source() -> dict[str, Any]:
    loaded = load_lineage(DEFAULT_INPUT_ROOT, "conv-41", DEFAULT_LEGACY_ROOT)
    projected = project_authoritative_parent(loaded["parent"])
    projected.update({
        "source_parent_row_path": Path(loaded["provenance"]["source_path"]),
        "source_parent_row_line_number": loaded["provenance"]["source_line_number"],
        "source_parent_row_sha256": sha256_bytes(canonical_json(loaded["parent"]).encode()),
        "source_parent_sample_id": loaded["parent"]["sample_id"],
        "declared_parent_graph_hash": loaded["parent"]["semantic_parent_graph"].get(
            "parent_graph_id"),
        "parent_anchor_path": DEFAULT_INPUT_ROOT / "conv-41/parent_anchor_snapshot.json",
    })
    return projected


def _view(turns: Sequence[Mapping[str, Any]], material: Mapping[str, Any]) -> dict[str, Any]:
    dialogue = [{"turn_id": turn["alias"], "session_id": turn["session_id"],
                 "speaker": turn["speaker"], "text": turn["text"]} for turn in turns]
    return {"combined_dialogue": dialogue, "span_catalog": {},
            "view_sha256": sha256_bytes(canonical_json(material).encode())}


def _budget_from_ledger(ledger: RequestLedger) -> ApiBudget:
    budget = ApiBudget(normal_caps=NORMAL_CAPS, format_retry_cap=FORMAT_RETRY_CAP,
                       physical_cap=PHYSICAL_CAP)
    for state in ledger.completed():
        metadata = state.get("request_metadata") or {}
        stage = str(metadata.get("stage") or "")
        if stage not in NORMAL_CAPS:
            continue
        if int(metadata.get("attempt_index") or 0) == 1:
            budget.retry_counts[stage] += 1
        else:
            budget.normal_counts[stage] += 1
        budget.physical_count += 1
    return budget


def _usage(ledger: RequestLedger) -> dict[str, Any]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    requests = []
    stage_counts = {stage: {"logical_requests": 0, "physical_requests": 0,
                            "format_retries": 0} for stage in NORMAL_CAPS}
    logical = {stage: set() for stage in NORMAL_CAPS}
    for state in sorted(ledger.completed(), key=lambda row: row["request_id"]):
        metadata = state.get("request_metadata") or {}
        stage = str(metadata.get("stage") or "")
        usage = (state.get("response_metadata") or {}).get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
        if stage in stage_counts:
            stage_counts[stage]["physical_requests"] += 1
            logical[stage].add(str(metadata.get("logical_request_id") or state["request_id"]))
            if int(metadata.get("attempt_index") or 0) == 1:
                stage_counts[stage]["format_retries"] += 1
        requests.append({"request_id": state["request_id"], "stage": stage,
            "attempt_index": metadata.get("attempt_index"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": (state.get("response_metadata") or {}).get("finish_reason")})
    for stage in stage_counts:
        stage_counts[stage]["logical_requests"] = len(logical[stage])
    return {"total_paid_api_requests": len(requests), **totals,
        "by_stage": stage_counts, "requests": requests,
        "ordinary_questioner_generation": 0, "distractor_generations": 0,
        "solver_generations": 0, "verifier_generations": 0, "training_runs": 0,
        "semantic_repair": 0,
        "realizer_generations": stage_counts["realizer"]["physical_requests"]}


class _FormalRealizerBatchAdapter:
    """Expose the formal LiveRealizerBackend through the shared single-object runner."""
    def __init__(self, client: OpenAICompatibleChallengerClient):
        self.client = client
        self.max_tokens = client.config.realizer_max_tokens
        self.realizer = LiveRealizerBackend(client)

    def generate_batch(self, request: Any) -> Any:
        realizer_request = RealizerRequest(request.request_id, 1,
            int(request.request_id.rsplit("a", 1)[-1]) + 1,
            request.prompt, request.prompt_sha256)
        return self.realizer.generate_session(realizer_request)


def preflight(source: Mapping[str, Any], tokenizer_path: Path,
              api_config: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    api = load_challenger_config(api_config)
    contract = PROMPT_FILES["dependency"].read_text(encoding="utf-8")
    prompt = build_dependency_planner_prompt(source, contract)
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if tokens + STAGE0_MAX_TOKENS > CONTEXT_LIMIT:
        raise RuntimeError("dependency_planner_context_overflow")
    if source["parent_count"] != 237 or source["child_count_in_generation_input"] != 0:
        raise RuntimeError("parent_only_source_preflight_failed")
    if api.model != "Qwen/Qwen3-14B":
        raise RuntimeError(f"unexpected_challenger_model:{api.model}")
    return {"decision": "pass", "model": api.model, "temperature": api.temperature,
        "parent_turn_count": 237, "child_turn_count_in_generation_input": 0,
        "parent_graph_event_count": len(source["parent_graph"]["events"]),
        "stage0_prompt_tokens": tokens, "stage0_max_tokens": STAGE0_MAX_TOKENS,
        "stage0_total_tokens": tokens + STAGE0_MAX_TOKENS,
        "context_limit": CONTEXT_LIMIT, "max_child_expansions": 2,
        "normal_request_cap": sum(NORMAL_CAPS.values()),
        "format_retry_cap": FORMAT_RETRY_CAP, "physical_request_cap": PHYSICAL_CAP}


def _manifest(run_root: Path, source: Mapping[str, Any], pre: Mapping[str, Any],
              snapshot: Mapping[str, Any], historical: Mapping[str, str]) -> dict[str, Any]:
    prompt_shas = {name: sha256_file(path) for name, path in PROMPT_FILES.items()}
    return {"run_id": run_root.name, "mode": CANARY_VERSION,
        "git_head": _git("rev-parse", "HEAD"),
        "project_git_commit": _git("rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT), **dict(snapshot),
        "orchestrator_source_sha256": sha256_file(MODULE_ROOT / "orchestrator.py"),
        "validation_source_sha256": sha256_file(MODULE_ROOT / "validation.py"),
        "ledger_source_sha256": sha256_file(MODULE_ROOT / "ledger.py"),
        "run_canary_source_sha256": sha256_file(MODULE_ROOT / "run_canary.py"),
        "mcq_parser_source_sha256": sha256_file(MODULE_ROOT / "core.py"),
        "questioner_prompt_builder_source_sha256": sha256_file(MODULE_ROOT / "prompts.py"),
        "questioner_contract_source_sha256": sha256_file(MODULE_ROOT / "questioner_contract.py"),
        "live_backend_source_sha256": sha256_file(MODULE_ROOT / "live_backends.py"),
        "source_parent_row_path": str(source["source_parent_row_path"]),
        "source_parent_row_line_number": source["source_parent_row_line_number"],
        "source_parent_row_sha256": source["source_parent_row_sha256"],
        "source_parent_sample_id": source["source_parent_sample_id"],
        "source_parent_dialogue_content_sha256": source["source_dialogue_content_sha256"],
        "source_parent_graph_sha256": source["parent_graph_source_sha256"],
        "parent_graph_projection_sha256": source["parent_graph_projection_sha256"],
        "parent_dialogue_with_aliases_sha256": source["parent_dialogue_sha256"],
        "parent_anchor_snapshot_path": str(source["parent_anchor_path"]),
        "parent_anchor_snapshot_sha256": sha256_file(source["parent_anchor_path"]),
        "parent_turn_count": 237, "child_turn_count_in_generation_input": 0,
        "dependency_prompt_version": DEPENDENCY_PROMPT_VERSION,
        "child_expander_prompt_version": CHILD_EXPANDER_PROMPT_VERSION,
        "realizer_prompt_version": REALIZER_PROMPT_VERSION,
        "qa_prompt_version": QA_PROMPT_VERSION,
        "dependency_schema_version": DEPENDENCY_SCHEMA_VERSION,
        "child_graph_schema_version": CHILD_GRAPH_SCHEMA_VERSION,
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "qa_schema_version": QA_SCHEMA_VERSION,
        "prompt_file_sha256": prompt_shas, **contract_shas(),
        "format_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "contract_source_sha256": sha256_file(MODULE_ROOT / "dependency_conditioned_child_canary.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "historical_run_tree_sha256_before": dict(historical),
        "challenger": {"model": pre["model"], "temperature": pre["temperature"],
            "response_format": "json_object"},
        "budget": {"normal_caps": NORMAL_CAPS, "format_retry_cap": FORMAT_RETRY_CAP,
            "physical_paid_api_requests": PHYSICAL_CAP, "ordinary_questioner": 0,
            "distractor": 0, "solver": 0, "verifier": 0, "training": 0,
            "semantic_repair": 0}}


def _dependency_audit_template(normalized: Mapping[str, Any], source_sha: str) -> dict[str, Any]:
    return {"audit_version": DEPENDENCY_AUDIT_VERSION,
        "source_dependency_contracts_sha256": source_sha,
        "items": [{"dependency_id": row["dependency_id"],
                   **{field: None for field in DEPENDENCY_AUDIT_FIELDS}, "notes": ""}
                  for row in normalized["dependencies"]
                  if row.get("local_validation") == "accept"]}


def _dependency_judgment(item: Mapping[str, Any]) -> str:
    if item["parent_only_sufficient"]: return "parent_alone_sufficient"
    if item["planned_child_only_sufficient"]: return "planned_child_alone_sufficient"
    if not item["combined_sufficient"]: return "combined_not_supported"
    if not item["natural_relation"]: return "unnatural_relation"
    if not item["minimal_parent_evidence"]: return "nonminimal_parent_evidence"
    return "dependency_contract_pass"


def validate_dependency_audit(audit: Any, normalized: Mapping[str, Any],
                              source_sha: str) -> dict[str, Any]:
    expected = [row["dependency_id"] for row in normalized["dependencies"]
                if row.get("local_validation") == "accept"]
    if (not isinstance(audit, dict) or audit.get("audit_version") != DEPENDENCY_AUDIT_VERSION
            or audit.get("source_dependency_contracts_sha256") != source_sha
            or not isinstance(audit.get("items"), list)
            or [row.get("dependency_id") for row in audit["items"]] != expected):
        raise ValueError("dependency_audit_binding_invalid")
    items = []
    for row in audit["items"]:
        if any(type(row.get(field)) is not bool for field in DEPENDENCY_AUDIT_FIELDS):
            raise ValueError("dependency_audit_incomplete")
        items.append({**row, "final_dependency_judgment": _dependency_judgment(row)})
    return {"audit_version": DEPENDENCY_AUDIT_VERSION,
        "source_dependency_contracts_sha256": source_sha, "items": items,
        "dependency_contract_pass": sum(row["final_dependency_judgment"] ==
                                        "dependency_contract_pass" for row in items)}


def select_dependencies_for_stage1(normalized: Mapping[str, Any],
                                   audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    passed = {row["dependency_id"] for row in audit["items"]
              if row["final_dependency_judgment"] == "dependency_contract_pass"}
    return [dict(row) for row in normalized["dependencies"]
            if row["dependency_id"] in passed][:2]


def select_realization_ids_for_qa(audit: Mapping[str, Any]) -> list[str]:
    return [str(row["dependency_id"]) for row in audit["items"]
            if row["final_realization_judgment"] == "realization_pass"]


def _render_dependency_review(normalized: Mapping[str, Any]) -> str:
    lines = ["# Stage 0 dependency review packet", "",
        "Only alias/event-aligned local-valid dependencies receive semantic audit.", ""]
    for row in normalized["dependencies"]:
        lines += [f"## {row['dependency_id']}", "", f"Status: `{row['status']}`", ""]
        if row.get("local_validation") != "accept":
            lines += [f"Local validation: `{row.get('local_validation')}`",
                      f"Reasons: `{row.get('local_validation_errors', [])}`", ""]
            continue
        lines += [f"Operator: `{row['operator']}`", "",
            f"Parent operand: `{canonical_json(row['parent_operand'])}`", "",
            f"Planned child event: `{canonical_json(row['planned_child_event'])}`", "",
            f"Joint result: `{canonical_json(row['joint_result'])}`", "",
            "### Authoritative parent evidence", ""]
        for turn in row["resolved_parent_turns"]:
            lines.append(f"- `{turn['alias']}` / `{turn['turn_id']}` {turn['speaker']}: {turn['text']}")
        lines += ["", "### Audit", ""]
        lines += [f"- {field}: " for field in DEPENDENCY_AUDIT_FIELDS]
        lines += ["- notes: ", ""]
    return "\n".join(lines)


def run_stage0(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")
    source = _source()
    pre = preflight(source, args.tokenizer, args.api_config)
    historical = {str(root): tree_manifest_sha256(root)
                  for root in (DEFAULT_GRAPH_SOURCE_RUN, DEFAULT_DIALOGUE_SOURCE_RUN)}
    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    manifest = _manifest(args.run_root, source, pre, snapshot, historical)
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", pre)
    atomic_json(args.run_root / "frozen/authoritative_parent_graph.json", source["parent_graph"])
    atomic_json(args.run_root / "frozen/parent_dialogue_with_aliases.json",
                {"turns": source["parent_dialogue"], "parent_count": 237,
                 "child_count_in_generation_input": 0})
    prompt = build_dependency_planner_prompt(source,
        PROMPT_FILES["dependency"].read_text(encoding="utf-8"))
    atomic_text(args.run_root / "prompts/stage0_dependency_planner.txt", prompt)
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    api = load_challenger_config(args.api_config)
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),
                                        STAGE0_MAX_TOKENS)
    task = run_single_object_task(run_root=args.run_root, ledger=ledger,
        backend=backend, budget=_budget_from_ledger(ledger), stage="dependency_planner",
        logical_request_id="dependency-planner-batch-4x", semantic_prompt=prompt,
        lineage_view=_view(source["parent_dialogue"], {"parent_only": True,
            "parent_graph_sha": source["parent_graph_projection_sha256"]}),
        validator=validate_dependency_batch_schema,
        schema_retry_decisions=("dependency_schema_invalid",))
    normalized = (normalize_dependency_batch(task["accepted_raw"], source)
                  if task["accepted"] else {"schema_version":
                      "normalized_dependency_contracts_v1", "dependencies": [],
                      "local_valid_count": 0, "empty_count": 0})
    atomic_json(args.run_root / "reports/stage0_dependency_request.json",
        {key: value for key, value in task.items()
         if key not in {"accepted_raw", "accepted_validation"}})
    contracts_path = args.run_root / "frozen/stage0_dependency_contracts.json"
    atomic_json(contracts_path, normalized)
    source_sha = sha256_file(contracts_path)
    atomic_json(args.run_root / "review/stage0_dependency_audit_template.json",
                _dependency_audit_template(normalized, source_sha))
    atomic_text(args.run_root / "review/stage0_dependency_review_packet.md",
                _render_dependency_review(normalized))
    usage = _usage(ledger); atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"phase": "stage0_complete_waiting_for_offline_audit",
        "schema_accepted": task["accepted"], "local_valid_dependencies":
        normalized["local_valid_count"], "empty_dependencies": normalized["empty_count"],
        "dependency_contracts_sha256": source_sha, "request": {
            key: value for key, value in task.items()
            if key not in {"accepted_raw", "accepted_validation"}}, "api_usage": usage}
    atomic_json(args.run_root / "reports/stage0_summary.json", report)
    return report


def _load_dependency_audit(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_root / "frozen/stage0_dependency_contracts.json"
    normalized = _read(path)
    audit = validate_dependency_audit(_read(run_root / "reports/stage0_dependency_audit.json"),
                                      normalized, sha256_file(path))
    return normalized, audit


def _realization_template(items: Sequence[Mapping[str, Any]], source_sha: str) -> dict[str, Any]:
    return {"audit_version": REALIZATION_AUDIT_VERSION,
        "source_realized_child_dialogues_sha256": source_sha,
        "items": [{"dependency_id": row["dependency_id"],
                   **{field: None for field in REALIZATION_AUDIT_FIELDS}, "notes": ""}
                  for row in items if row.get("local_gate", {}).get("decision") ==
                  "realization_local_gate_pass"]}


def _realization_judgment(row: Mapping[str, Any]) -> str:
    if row["parent_fact_restated"]: return "parent_fact_restatement"
    if row["joint_answer_leaked"]: return "joint_answer_leakage"
    if row["single_turn_shortcut"]: return "single_turn_shortcut"
    if not row["planned_child_event_verbalized"]: return "unverbalized_child_event"
    if not row["dialogue_matches_graph"]: return "dialogue_graph_mismatch"
    if row["unsupported_realizer_fact"]: return "unsupported_realizer_fact"
    if not row["natural_dialogue"]: return "unnatural_dialogue"
    return "realization_pass"


def validate_realization_audit(audit: Any, items: Sequence[Mapping[str, Any]],
                               source_sha: str) -> dict[str, Any]:
    expected = [row["dependency_id"] for row in items if row.get("local_gate", {}).get(
        "decision") == "realization_local_gate_pass"]
    if (not isinstance(audit, dict) or audit.get("audit_version") != REALIZATION_AUDIT_VERSION
            or audit.get("source_realized_child_dialogues_sha256") != source_sha
            or [row.get("dependency_id") for row in audit.get("items", [])] != expected):
        raise ValueError("realization_audit_binding_invalid")
    output = []
    for row in audit["items"]:
        if any(type(row.get(field)) is not bool for field in REALIZATION_AUDIT_FIELDS):
            raise ValueError("realization_audit_incomplete")
        output.append({**row, "final_realization_judgment": _realization_judgment(row)})
    return {"audit_version": REALIZATION_AUDIT_VERSION,
        "source_realized_child_dialogues_sha256": source_sha, "items": output,
        "realization_pass": sum(row["final_realization_judgment"] == "realization_pass"
                                for row in output)}


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    source = _source(); normalized, audit = _load_dependency_audit(args.run_root)
    atomic_json(args.run_root / "reports/stage0_dependency_audit.json", audit)
    selected = select_dependencies_for_stage1(normalized, audit)
    atomic_json(args.run_root / "frozen/accepted_dependency_contracts.json", selected)
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    api = load_challenger_config(args.api_config)
    client = OpenAICompatibleChallengerClient(api)
    parent_ids = {event["event_id"] for event in source["parent_graph"]["events"]}
    items = []
    for dependency in selected:
        did = dependency["dependency_id"]
        expander_prompt = build_child_expander_prompt(source["parent_graph"], dependency,
            PROMPT_FILES["expander"].read_text(encoding="utf-8"))
        atomic_text(args.run_root / f"prompts/stage1_child_expander_{did}.txt", expander_prompt)
        expander = run_single_object_task(run_root=args.run_root, ledger=ledger,
            backend=LiveSemanticRepairBackend(client, EXPANDER_MAX_TOKENS),
            budget=_budget_from_ledger(ledger), stage="child_expander",
            logical_request_id=f"child-expander-{did}", semantic_prompt=expander_prompt,
            lineage_view=_view(source["parent_dialogue"], {"dependency_id": did,
                "parent_graph_sha": source["parent_graph_projection_sha256"]}),
            validator=lambda raw, dep=dependency: validate_child_graph(raw, dep, parent_ids),
            schema_retry_decisions=("child_graph_schema_invalid",))
        item: dict[str, Any] = {"dependency_id": did,
            "child_expander_request": {key: value for key, value in expander.items()
                if key not in {"accepted_raw", "accepted_validation"}}}
        if not expander["accepted"]:
            item["stage1_decision"] = "child_graph_reject"; items.append(item); continue
        child_graph = expander["accepted_raw"]
        atomic_json(args.run_root / f"frozen/child_graphs/{did}.json", child_graph)
        realizer_prompt = build_realizer_prompt(dependency, child_graph)
        atomic_text(args.run_root / f"prompts/stage1_realizer_{did}.txt", realizer_prompt)
        realizer = run_single_object_task(run_root=args.run_root, ledger=ledger,
            backend=_FormalRealizerBatchAdapter(client), budget=_budget_from_ledger(ledger),
            stage="realizer", logical_request_id=f"dependency-realizer-{did}",
            semantic_prompt=realizer_prompt,
            lineage_view=_view([], {"dependency_id": did, "child_graph": child_graph}),
            validator=lambda raw, dep=dependency, graph=child_graph:
                validate_realizer_output(raw, dep, graph),
            schema_retry_decisions=("realizer_schema_invalid",))
        item["realizer_request"] = {key: value for key, value in realizer.items()
            if key not in {"accepted_raw", "accepted_validation"}}
        item["child_graph"] = child_graph
        if not realizer["accepted"]:
            item["stage1_decision"] = "realizer_schema_reject"; items.append(item); continue
        local_gate = apply_anti_leak_gate(realizer["accepted_raw"], dependency, child_graph)
        item.update({"realizer_output": realizer["accepted_raw"], "local_gate": local_gate,
                     "stage1_decision": local_gate["decision"]})
        items.append(item)
    realized_path = args.run_root / "frozen/realized_child_dialogues.json"
    atomic_json(realized_path, items); source_sha = sha256_file(realized_path)
    atomic_json(args.run_root / "review/stage1_realization_audit_template.json",
                _realization_template(items, source_sha))
    lines = ["# Stage 1 realization review packet", ""]
    for item in items:
        lines += [f"## {item['dependency_id']}", "",
                  f"Local decision: `{item['stage1_decision']}`", ""]
        for turn in item.get("local_gate", {}).get("child_dialogue_with_aliases", []):
            lines.append(f"- `{turn['alias']}` {turn['speaker']}: {turn['text']}")
        lines += [""] + [f"- {field}: " for field in REALIZATION_AUDIT_FIELDS] + ["- notes: ", ""]
    atomic_text(args.run_root / "review/stage1_realization_review_packet.md", "\n".join(lines))
    usage = _usage(ledger); atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"phase": "stage1_complete_waiting_for_offline_audit",
        "dependency_contract_pass": audit["dependency_contract_pass"],
        "attempted_child_expansions": len(selected), "items": items,
        "realized_child_dialogues_sha256": source_sha, "api_usage": usage}
    atomic_json(args.run_root / "reports/stage1_generation_summary.json", report)
    return report


def _final_template(qas: Sequence[Mapping[str, Any]], source_sha: str) -> dict[str, Any]:
    return {"audit_version": FINAL_QA_AUDIT_VERSION,
        "source_qa_outputs_sha256": source_sha,
        "items": [{"dependency_id": row["dependency_id"],
                   **{field: None for field in FINAL_AUDIT_FIELDS}, "notes": ""}
                  for row in qas if row.get("decision") == "qa_accept"]}


def _final_judgment(row: Mapping[str, Any]) -> str:
    if row["parent_only_can_answer_uniquely"]: return "parent_only_sufficient"
    if row["child_only_can_answer_uniquely"]: return "child_only_sufficient"
    if not row["combined_can_answer_uniquely"]: return "combined_not_supported"
    if row["single_turn_contains_full_answer"]: return "single_turn_shortcut"
    if row["evidence_is_repetitive"]: return "repetitive_evidence"
    if row["question_leaks_reasoning_step"]: return "question_leakage"
    if not row["question_answer_supported"]: return "unsupported_inference"
    if not row["minimal_evidence"]: return "nonminimal_evidence"
    if not row["natural_wording"]: return "unnatural_question"
    return "lineage_jointly_required"


def validate_final_audit(audit: Any, qas: Sequence[Mapping[str, Any]], source_sha: str) -> dict[str, Any]:
    expected = [row["dependency_id"] for row in qas if row.get("decision") == "qa_accept"]
    if (not isinstance(audit, dict) or audit.get("audit_version") != FINAL_QA_AUDIT_VERSION
            or audit.get("source_qa_outputs_sha256") != source_sha
            or [row.get("dependency_id") for row in audit.get("items", [])] != expected):
        raise ValueError("final_audit_binding_invalid")
    output = []
    for row in audit["items"]:
        if any(type(row.get(field)) is not bool for field in FINAL_AUDIT_FIELDS):
            raise ValueError("final_audit_incomplete")
        output.append({**row, "final_joint_judgment": _final_judgment(row)})
    return {"audit_version": FINAL_QA_AUDIT_VERSION,
        "source_qa_outputs_sha256": source_sha, "items": output,
        "lineage_jointly_required_qa": sum(row["final_joint_judgment"] ==
                                           "lineage_jointly_required" for row in output)}


def derive_route_conclusion(*, stage0_schema_accepted: bool, dependency_pass: int,
                            realization_pass: int, joint_qa: int) -> str:
    if not stage0_schema_accepted:
        return "dependency_planner_schema_invalid_exhausted"
    if dependency_pass < 2:
        return "dependency_planning_bottleneck"
    if realization_pass < 2:
        return "child_expansion_or_realizer_bottleneck"
    if joint_qa < 2:
        return "qa_writer_bottleneck"
    return "dependency_conditioned_generation_supported"


def run_stage2(args: argparse.Namespace) -> dict[str, Any]:
    normalized, dep_audit = _load_dependency_audit(args.run_root)
    stage1_path = args.run_root / "frozen/realized_child_dialogues.json"
    stage1_items = _read(stage1_path)
    real_audit = validate_realization_audit(
        _read(args.run_root / "reports/stage1_realization_audit.json"), stage1_items,
        sha256_file(stage1_path))
    atomic_json(args.run_root / "reports/stage1_realization_audit.json", real_audit)
    dependencies = {row["dependency_id"]: row for row in normalized["dependencies"]}
    realized = {row["dependency_id"]: row for row in stage1_items}
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    api = load_challenger_config(args.api_config)
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api), QA_MAX_TOKENS)
    qas, frozen_inputs = [], []
    for did in select_realization_ids_for_qa(real_audit):
        dependency = dependencies[did]
        parent_turns = dependency["resolved_parent_turns"]
        child_turns = realized[did]["local_gate"]["child_dialogue_with_aliases"]
        frozen_input = {"dependency_id": did, "operator": dependency["operator"],
            "joint_result": dependency["joint_result"], "parent_turns": parent_turns,
            "child_turns": child_turns}
        frozen_inputs.append(frozen_input)
        prompt = build_qa_writer_prompt(dependency, parent_turns, child_turns,
            PROMPT_FILES["qa"].read_text(encoding="utf-8"))
        atomic_text(args.run_root / f"prompts/stage2_qa_writer_{did}.txt", prompt)
        task = run_single_object_task(run_root=args.run_root, ledger=ledger,
            backend=backend, budget=_budget_from_ledger(ledger),
            stage="dependency_qa_writer", logical_request_id=f"dependency-qa-writer-{did}",
            semantic_prompt=prompt,
            lineage_view=_view([*parent_turns, *child_turns], frozen_input),
            validator=lambda raw, dep=dependency, p=parent_turns, c=child_turns:
                validate_qa_object(raw, dep, [t["alias"] for t in p],
                                   [t["alias"] for t in c]),
            schema_retry_decisions=("qa_schema_invalid",))
        qas.append({"dependency_id": did,
            "decision": "qa_accept" if task["accepted"] else task["final_decision"],
            "qa": task["accepted_raw"], "request": {key: value for key, value in task.items()
                if key not in {"accepted_raw", "accepted_validation"}}})
    atomic_json(args.run_root / "frozen/frozen_qa_inputs.json", frozen_inputs)
    qa_path = args.run_root / "frozen/qa_writer_outputs.json"; atomic_json(qa_path, qas)
    qa_sha = sha256_file(qa_path)
    atomic_json(args.run_root / "review/final_joint_necessity_audit_template.json",
                _final_template(qas, qa_sha))
    lines = ["# Final QA review packet", ""]
    for row in qas:
        lines += [f"## {row['dependency_id']}", "", f"Decision: `{row['decision']}`", "",
                  "```json", json.dumps(row.get("qa"), ensure_ascii=False, indent=2), "```", ""]
        lines += [f"- {field}: " for field in FINAL_AUDIT_FIELDS] + ["- notes: ", ""]
    atomic_text(args.run_root / "review/final_qa_review_packet.md", "\n".join(lines))
    usage = _usage(ledger); atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {"phase": "stage2_complete_waiting_for_offline_audit",
        "realization_pass": real_audit["realization_pass"], "qa_writer_outputs": len(qas),
        "qa_valid": sum(row["decision"] == "qa_accept" for row in qas),
        "qa_outputs_sha256": qa_sha, "items": qas, "api_usage": usage}
    atomic_json(args.run_root / "reports/stage2_qa_summary.json", report)
    return report


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    if (args.run_root / "reports/dependency_conditioned_child_canary.json").exists():
        raise FileExistsError("already_finalized")
    normalized, dep_audit = _load_dependency_audit(args.run_root)
    stage1_path = args.run_root / "frozen/realized_child_dialogues.json"
    stage1_items = _read(stage1_path)
    real_audit = validate_realization_audit(_read(
        args.run_root / "reports/stage1_realization_audit.json"), stage1_items,
        sha256_file(stage1_path))
    qa_path = args.run_root / "frozen/qa_writer_outputs.json"
    qas = _read(qa_path)
    final_audit = validate_final_audit(_read(
        args.run_root / "reports/final_joint_necessity_audit.json"), qas,
        sha256_file(qa_path))
    atomic_json(args.run_root / "reports/final_joint_necessity_audit.json", final_audit)
    ledger = RequestLedger(args.run_root, sha256_file(
        args.run_root / "config/run_manifest.json"), "live")
    usage = _usage(ledger)
    stage0_summary = _read(args.run_root / "reports/stage0_summary.json")
    conclusion = derive_route_conclusion(
        stage0_schema_accepted=bool(stage0_summary.get("schema_accepted")),
        dependency_pass=dep_audit["dependency_contract_pass"],
        realization_pass=real_audit["realization_pass"],
        joint_qa=final_audit["lineage_jointly_required_qa"])
    manifest = _read(args.run_root / "config/run_manifest.json")
    historical_after = {str(root): tree_manifest_sha256(root)
                        for root in (DEFAULT_GRAPH_SOURCE_RUN, DEFAULT_DIALOGUE_SOURCE_RUN)}
    history_unchanged = historical_after == manifest["historical_run_tree_sha256_before"]
    report = {"canary_version": CANARY_VERSION,
        "parent_turn_count": 237, "child_turn_count_in_generation_input": 0,
        "stage0_dependency_slots": 4,
        "stage0_local_valid": normalized["local_valid_count"],
        "stage0_dependency_contract_pass": dep_audit["dependency_contract_pass"],
        "stage0_semantic_quality_identifiable": bool(stage0_summary.get("schema_accepted")),
        "stage0_success_threshold_met": dep_audit["dependency_contract_pass"] >= 2,
        "attempted_child_expansions": len(_read(
            args.run_root / "frozen/accepted_dependency_contracts.json")),
        "stage1_realization_pass": real_audit["realization_pass"],
        "stage1_success_threshold_met": real_audit["realization_pass"] >= 2,
        "qa_writer_outputs": len(qas),
        "lineage_jointly_required_qa": final_audit["lineage_jointly_required_qa"],
        "stage2_success_threshold_met": final_audit["lineage_jointly_required_qa"] >= 2,
        "route_conclusion": conclusion, "dependency_audit": dep_audit,
        "realization_audit": real_audit, "final_qa_audit": final_audit,
        "api_usage": usage, "historical_run_tree_sha256_after": historical_after,
        "historical_run_tree_sha_unchanged": history_unchanged,
        "ordinary_questioner_generation": 0, "distractor_generations": 0,
        "solver_generations": 0, "verifier_generations": 0, "training_runs": 0,
        "semantic_repair": 0}
    atomic_json(args.run_root / "reports/dependency_conditioned_child_canary.json", report)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {"contract_source_sha_match": sha256_file(
        MODULE_ROOT / "dependency_conditioned_child_canary.py") == manifest[
            "contract_source_sha256"],
        "runner_source_sha_match": sha256_file(Path(__file__)) == manifest[
            "runner_source_sha256"], "historical_run_tree_sha_unchanged": history_unchanged,
        "parent_count_exact": manifest["parent_turn_count"] == 237,
        "child_generation_input_zero": manifest["child_turn_count_in_generation_input"] == 0}
    provenance.update(checks)
    provenance["authoritative_release_provenance"] = all(checks.values()) and bool(
        provenance.get("authoritative_release_provenance"))
    provenance["provenance_mixed"] = not provenance["authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    if provenance["provenance_mixed"]:
        raise RuntimeError("provenance_mixed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "stage0", "stage1", "stage2",
                                            "finalize"), required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args(argv); args.run_root = args.run_root.resolve()
    if args.phase == "preflight":
        if args.run_root.exists(): raise FileExistsError(f"run_root_exists:{args.run_root}")
        result = preflight(_source(), args.tokenizer, args.api_config)
    elif args.phase == "stage0": result = run_stage0(args)
    elif args.phase == "stage1": result = run_stage1(args)
    elif args.phase == "stage2": result = run_stage2(args)
    else: result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
