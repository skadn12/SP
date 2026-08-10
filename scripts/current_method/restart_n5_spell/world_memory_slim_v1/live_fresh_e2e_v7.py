"""Unique live Slim Fresh E2E v7 runner for the frozen v3/v7 architecture."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..unified_dialogue_root import read_unified_parent
from .append_only_memory import apply_scene_commit, evaluate_scene_commit
from .claim_validator_v2 import realizer_retry_route, validate_proposal_claims
from .context_compiler import compile_round_planner_context, compile_scene_context
from .controlled_expansion import compile_action_menu_v3, validate_action_menu_v3
from .dialogue_quality import validate_dialogue_quality
from .evidence_scope import (
    EVIDENCE_SCOPE_PROMPT_VERSION, EVIDENCE_SCOPE_RETRY_PROMPT_VERSION,
    build_committer_scope_prompt, build_committer_scope_retry_prompt,
    committer_retry_disposition, validate_committer_scope_output,
)
from .interaction_links import compile_generated_interaction_links, compile_source_interaction_links
from .live_preflight_v7 import run_live_preflight_v7, write_preflight_failure_report
from .live_role_runtime import LiveJsonRoleCaller, RoleCaller, aggregate_role_usage
from .planner_v3 import (
    PLANNER_PROMPT_VERSION_V3, PLANNER_RETRY_PROMPT_VERSION_V3,
    build_planner_v3_prompt, build_planner_v3_retry_prompt,
)
from .planner_contract_v3_1 import (
    PLANNER_ACTION_VIEW_VERSION, PLANNER_CONTRACT_VERSION,
    compile_allowed_proposal_bundles, compile_planner_action_view,
)
from .promotion import assemble_child_surfaces
from .realizer_v3 import (
    NATURAL_REALIZER_CLAIM_RETRY_PROMPT_VERSION_V3,
    NATURAL_REALIZER_FULL_RETRY_PROMPT_VERSION_V3,
    NATURAL_REALIZER_PROMPT_VERSION_V3,
    build_claim_only_v3_prompt, build_realizer_v3_full_retry_prompt,
    build_realizer_v3_prompt, validate_claim_only_realizer_shape,
)
from .round_plan_validation import compile_validate_round_plan
from .scene_local_refs import (
    dialogue_sha256, normalize_full_realizer_output,
    reconstruct_claim_only_output,
)
from .semantic_import_audit_v7 import audit_v7_recursive_import_graph
from .source_memory_initializer_v1 import (
    authoritative_affordance_view, initialize_source_world_memory,
)
from .transactional_binder import (
    build_allocation_plan, commit_scene_transaction,
    recover_prepared_transactions,
    validate_scene_local_evidence_refs,
)
from .turn_budget import compile_turn_budget
from .v7_manifest import (directory_manifest, implementation_manifest,
                          project_working_tree_manifest_sha256)


LIVE_FRESH_E2E_V7_RUNNER_VERSION = "world_memory_slim_live_fresh_e2e_v7"
SUMMARY_VERSION = "slim_fresh_e2e_v7_summary_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parent
HISTORY_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-61555/unified_root.json")
DEFAULT_V6_ROOT = HISTORY_ROOT / "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v6"
DEFAULT_RUN_ROOT = HISTORY_ROOT / "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v7"
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_FREEZE_REPORT = PACKAGE_ROOT / "reports/SLIM_V7_LIVE_RUNNER_FREEZE.json"


@dataclass(frozen=True)
class V7RunLimits:
    planner_attempts: int = 3
    realizer_attempts_per_scene: int = 3
    committer_attempts_per_scene: int = 3
    planner_max_tokens: int = 4096
    realizer_max_tokens: int = 4096
    committer_max_tokens: int = 3072
    context_limit: int = 40960
    total_physical_request_limit: int = 32


class PreflightFailure(RuntimeError):
    pass


class LocalInvariantFailure(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for turn in turns:
        grouped[str(turn.get("session_id") or "parent")].append(turn)
    return [{"session_id": key, "turns": value} for key, value in sorted(grouped.items())]


def _authoritative_prompt_view(memory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "append_only_memory_version": memory["append_only_memory_version"],
        "root_id": memory["root_id"], "entities": memory["entities"],
        "authoritative_records": memory["authoritative_records"],
        "active_head_index": memory["active_head_index"],
        "memory_transitions": memory["memory_transitions"],
    }


def _semantic_sha(*values: Any) -> str:
    return sha256_bytes(canonical_json(list(values)).encode("utf-8"))


class FreezeGuard:
    def __init__(self, *, project_root: Path, input_path: Path, v6_root: Path,
                 api_config_path: Path, frozen: Mapping[str, Any], run_root: Path):
        self.project_root = project_root
        self.input_path = input_path
        self.v6_root = v6_root
        self.api_config_path = api_config_path
        self.frozen = dict(frozen)
        self.run_root = run_root
        self.request_count = 0

    def before_request(self, logical_role: str, role_version: str) -> None:
        checks = {
            "implementation": implementation_manifest(self.project_root)[
                "implementation_manifest_sha256"] == self.frozen["implementation_manifest_sha256"],
            "working_tree": project_working_tree_manifest_sha256(self.project_root) == self.frozen[
                "project_working_tree_manifest_sha256"],
            "input": sha256_file(self.input_path) == self.frozen["input_sha256"],
            "v6": directory_manifest(self.v6_root)["directory_manifest_sha256"] ==
                self.frozen["v6_directory_manifest_sha256"],
            "api_config": sha256_file(self.api_config_path) == self.frozen["api_config_sha256"],
            "git_head": _git("rev-parse", "HEAD") == self.frozen["git_head"],
        }
        if not all(checks.values()):
            raise LocalInvariantFailure("frozen_runtime_drift:" + ",".join(
                key for key, value in checks.items() if not value))
        self.request_count += 1
        atomic_json(self.run_root / "run_status.json", {
            "status": "api_started", "first_api_started": self.request_count == 1,
            "physical_request_ordinal": self.request_count,
            "logical_role": logical_role, "role_version": role_version,
            "updated_at_utc": _utc_now(),
        })


def run(
    *, input_path: Path = DEFAULT_INPUT, run_root: Path = DEFAULT_RUN_ROOT,
    historical_v6_root: Path = DEFAULT_V6_ROOT,
    api_config_path: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    freeze_report_path: Path = DEFAULT_FREEZE_REPORT,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    preflight = run_live_preflight_v7(
        project_root=PROJECT_ROOT, package_root=PACKAGE_ROOT,
        runner_path=runner_path, input_path=input_path,
        historical_v6_root=historical_v6_root, run_root=run_root,
        freeze_report_path=freeze_report_path, api_config_path=api_config_path,
        tokenizer_path=tokenizer_path)
    if not preflight["valid"]:
        write_preflight_failure_report(PACKAGE_ROOT, preflight)
        raise PreflightFailure("PRE_API_PREFLIGHT_FAILED:" + ",".join(preflight["errors"]))
    # Load all local runtime configuration before the only permitted run root is
    # created.  Both operations are read-only and cannot issue a network request.
    api = load_challenger_config(Path(api_config_path))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    root = json.loads(Path(input_path).read_text(encoding="utf-8"))
    parent = read_unified_parent(Path(input_path))
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=False)
    frozen = {
        "runner_version": LIVE_FRESH_E2E_V7_RUNNER_VERSION,
        "created_at_utc": _utc_now(), "fresh": True, "no_resume": True,
        "git_head": preflight["git_head"],
        "implementation_manifest_sha256": preflight["implementation_manifest"][
            "implementation_manifest_sha256"],
        "project_working_tree_manifest_sha256": project_working_tree_manifest_sha256(PROJECT_ROOT),
        "input_sha256": preflight["input_sha256"],
        "parent_dialogue_sha256": preflight["parent_dialogue_sha256"],
        "v6_directory_manifest_sha256": preflight["v6_directory_manifest"][
            "directory_manifest_sha256"],
        "api_config_sha256": preflight["api_config_sha256"],
        "api_configuration": preflight["api_configuration"],
        "tokenizer_path": preflight["tokenizer_path"],
        "runner_path": str(runner_path),
        "component_versions": preflight["component_versions"],
        "prompt_schema_versions": preflight["prompt_schema_versions"],
        "prompt_schema_contract_sha256": preflight["prompt_schema_contract_sha256"],
        "run_limits": asdict(V7RunLimits()),
        "cache_reuse_allowed": False, "imported_response_allowed": False,
        "legacy_response_reuse_allowed": False,
    }
    atomic_json(run_root / "run_freeze_manifest.json", frozen)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "api_calls": 0, "updated_at_utc": _utc_now()})
    atomic_json(run_root / "frozen" / "authoritative_parent.json", parent)
    atomic_json(run_root / "reports" / "pre_api_preflight.json", preflight)
    ledger = RequestLedger(
        run_root, sha256_file(run_root / "run_freeze_manifest.json"), "live")
    guard = FreezeGuard(
        project_root=PROJECT_ROOT, input_path=Path(input_path),
        v6_root=Path(historical_v6_root), api_config_path=Path(api_config_path),
        frozen=frozen, run_root=run_root)
    caller = LiveJsonRoleCaller(
        run_root=run_root, ledger=ledger,
        client=OpenAICompatibleChallengerClient(api),
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        root_id=str(root["root_id"]),
        role_budget_limits={"planner": 3, "realizer": 12, "evidence_committer": 12},
        total_request_limit=V7RunLimits().total_physical_request_limit,
        context_limit=V7RunLimits().context_limit,
        before_request=guard.before_request)
    model_config = {"model": api.model, "base_url": api.base_url,
                    "temperature": api.temperature, "timeout_seconds": api.timeout_seconds,
                    "enable_thinking": api.enable_thinking, "api_key_recorded": False}
    try:
        return _execute_v7(
            root=root, parent=parent, run_root=run_root, caller=caller,
            limits=V7RunLimits(), freeze_manifest=frozen,
            input_path=Path(input_path), historical_v6_root=Path(historical_v6_root),
            model_config=model_config)
    except Exception as error:
        return _write_bootstrap_failure(
            run_root=run_root, caller=caller, freeze_manifest=frozen,
            input_path=Path(input_path), historical_v6_root=Path(historical_v6_root),
            model_config=model_config, error=error)


def _write_bootstrap_failure(
    *, run_root: Path, caller: RoleCaller, freeze_manifest: Mapping[str, Any],
    input_path: Path, historical_v6_root: Path,
    model_config: Mapping[str, Any], error: Exception,
) -> dict[str, Any]:
    """Persist an honest terminal report for failures before normal finalization."""
    failure_stage = ("local_invariant_failure" if isinstance(error, LocalInvariantFailure)
                     else "runtime_system_failure")
    usage = aggregate_role_usage(caller.records)
    ledger_events = run_root / "ledger" / "request_events.jsonl"
    atomic_text(run_root / "request_ledger.jsonl",
                ledger_events.read_text(encoding="utf-8") if ledger_events.is_file() else "")
    summary = {
        "schema_version": SUMMARY_VERSION,
        "runner_version": LIVE_FRESH_E2E_V7_RUNNER_VERSION,
        "run_root": str(run_root), "failure_stage": failure_stage,
        "failure_message": f"{type(error).__name__}:{error}",
        "freeze_manifest": dict(freeze_manifest), "model_config": dict(model_config),
        "api_usage": usage,
        "forbidden_role_calls": {"questioner": 0, "distractor": 0, "solver": 0,
                                 "qa_replay": 0, "strategy": 0, "training": 0,
                                 "legacy_api_roles": 0},
        "cache_imported_legacy_reuse": {"cache": 0, "imported": 0, "legacy": 0},
        "generation_ready": False, "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }
    atomic_json(run_root / "reports" / "slim_fresh_e2e_v7_summary.json", summary)
    atomic_text(run_root / "reports" / "slim_fresh_e2e_v7_summary.md",
                "# Slim Fresh E2E v7 Summary\n\n```json\n" +
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    post_checks = {
        "input_sha_unchanged": sha256_file(input_path) == freeze_manifest["input_sha256"],
        "v6_sha_unchanged": directory_manifest(historical_v6_root)[
            "directory_manifest_sha256"] == freeze_manifest["v6_directory_manifest_sha256"],
        "git_head_unchanged": _git("rev-parse", "HEAD") == freeze_manifest["git_head"],
        "implementation_manifest_unchanged": implementation_manifest(PROJECT_ROOT)[
            "implementation_manifest_sha256"] == freeze_manifest["implementation_manifest_sha256"],
        "working_tree_manifest_unchanged": project_working_tree_manifest_sha256(PROJECT_ROOT) ==
            freeze_manifest["project_working_tree_manifest_sha256"],
    }
    atomic_json(run_root / "run_status.json", {
        "status": "failed", "failure_stage": failure_stage,
        "generation_ready": False, "fresh_e2e_ready": False,
        "qa_pipeline_ready": False, "updated_at_utc": _utc_now()})
    directory = directory_manifest(
        run_root, excluded_relatives={"reports/post_run_provenance.json"})
    atomic_json(run_root / "reports" / "post_run_provenance.json", {
        "post_run_provenance_version": "slim_v7_post_run_provenance_v1",
        "checks": post_checks, "all_checks_pass": all(post_checks.values()),
        "v7_directory_manifest": directory, "api_usage": usage,
        "terminal_failure": summary["failure_message"],
    })
    return summary


def _execute_v7(
    *, root: Mapping[str, Any], parent: Mapping[str, Any], run_root: Path,
    caller: RoleCaller, limits: V7RunLimits, freeze_manifest: Mapping[str, Any],
    input_path: Path, historical_v6_root: Path,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    root_id = str(root["root_id"])
    child_id = f"{root_id}:child:fresh_v7"
    source = initialize_source_world_memory(root, parent)
    memory = source["memory"]
    initial_memory = deepcopy(memory)
    initial_record_shas = {str(row["memory_ref"]): _semantic_sha(row)
                           for row in memory["authoritative_records"]}
    atomic_json(run_root / "world_memory_initial.json", initial_memory)
    speaker_map = source["speaker_to_entity_ref"]
    interaction_links = compile_source_interaction_links(
        root_id=root_id, sessions=_sessions(parent["dialogue"]["turns"]),
        speaker_to_entity=speaker_map)
    atomic_json(run_root / "interaction_links_initial.json", interaction_links)
    affordance_view = authoritative_affordance_view(memory)
    action_menu = compile_action_menu_v3(
        affordance_view, root_id=root_id, interaction_links=interaction_links)
    action_validation = validate_action_menu_v3(action_menu)
    if not action_validation["valid"] or not action_menu["actions"]:
        raise LocalInvariantFailure("affordance_compiler:" +
                                    ",".join(action_validation["errors"] or ["empty_menu"]))
    atomic_json(run_root / "affordance_menu.json", action_menu)
    round_context = compile_round_planner_context(
        action_menu=action_menu, authoritative_memory=affordance_view,
        parent_turns=parent["dialogue"]["turns"], interaction_links=interaction_links)
    atomic_json(run_root / "planner" / "round_context_manifest.json", round_context)
    role_table = [{"entity_ref": row["entity_ref"],
                   "display_name": row["display_name"],
                   "entity_type": row.get("entity_type") or "person"}
                  for row in memory["entities"]]
    planner_action_view = compile_planner_action_view(
        action_menu, role_table=role_table)
    allowed_bundles = compile_allowed_proposal_bundles(action_menu)
    atomic_json(run_root / "planner_action_view.json", planner_action_view)
    atomic_json(run_root / "planner_allowed_bundles.json", allowed_bundles)
    scene_budget = {"minimum": 2, "target": 3, "maximum": 4}
    planner_prompt = build_planner_v3_prompt(
        root_id=root_id, role_table=role_table,
        action_menu=action_menu, context_summary=round_context,
        planner_action_view=planner_action_view,
        allowed_bundles=allowed_bundles)
    known_action_ids = {str(row["action_id"]) for row in action_menu["actions"]}
    planner_result: dict[str, Any] | None = None
    planner_output: dict[str, Any] | None = None
    planner_attempts: list[dict[str, Any]] = []
    prompt = planner_prompt
    for attempt in range(1, limits.planner_attempts + 1):
        parsed, raw, parse_errors = caller.call_json(
            request_id=f"planner-v3-a{attempt:02d}", role="planner_v3",
            attempt=attempt, prompt=prompt,
            prompt_version=(PLANNER_PROMPT_VERSION_V3 if attempt == 1 else
                            PLANNER_RETRY_PROMPT_VERSION_V3),
            max_tokens=limits.planner_max_tokens)
        result = (compile_validate_round_plan(
            parsed, action_menu=action_menu, authoritative_memory=memory,
            interaction_links=interaction_links, round_index=1,
            planner_action_view=planner_action_view,
            allowed_bundles=allowed_bundles)
                  if parsed is not None else {
                      "valid": False, "classification": "planner_repairable",
                      "errors": parse_errors})
        planner_attempts.append({"attempt": attempt, "validation": result})
        atomic_json(run_root / "planner" / f"validation_attempt_{attempt:02d}.json", result)
        if result["valid"]:
            planner_result, planner_output = result, dict(parsed)
            break
        if result.get("classification") == "local_invariant_failure":
            raise LocalInvariantFailure("planner_local_invariant:" + ",".join(result["errors"]))
        relevant = sorted({token for error in result["errors"]
                           for token in str(error).replace(":", " ").split()
                           if token.startswith("A") and token[1:].isdigit()})
        if isinstance(parsed, Mapping):
            selected_ids: set[str] = set()
            for scene_value in parsed.get("scene_goals") or []:
                if not isinstance(scene_value, Mapping):
                    continue
                selection = scene_value.get("proposal_selection")
                if not isinstance(selection, Mapping):
                    continue
                if selection.get("selection_type") == "single_action":
                    selected_ids.add(str(selection.get("action_id") or ""))
                elif selection.get("selection_type") == "bundle":
                    selected_ids.update(str(value) for value in
                                        (selection.get("proposal_contents") or {}))
            relevant = sorted(set(relevant) |
                              (selected_ids & known_action_ids))
        if "planner_scene_count" in result["errors"]:
            # A complete-round cardinality repair must be able to choose an
            # unused frozen action.  An empty filter means the retry builder
            # exposes the full Planner-facing action view.
            relevant = []
        prompt = build_planner_v3_retry_prompt(
            root_id=root_id, frozen_action_menu=action_menu,
            frozen_role_table=role_table, frozen_context=round_context,
            frozen_scene_budget=scene_budget, errors=result["errors"],
            relevant_action_ids=relevant,
            frozen_authoritative_memory_sha256=_semantic_sha(_authoritative_prompt_view(memory)),
            frozen_interaction_links_sha256=_semantic_sha(interaction_links),
            frozen_round_context_sha256=_semantic_sha(round_context),
            frozen_planner_action_view=planner_action_view,
            frozen_allowed_bundles=allowed_bundles)
    if planner_result is None or planner_output is None:
        return _finalize_run(
            root=root, parent=parent, run_root=run_root, caller=caller,
            freeze_manifest=freeze_manifest, input_path=input_path,
            historical_v6_root=historical_v6_root, model_config=model_config,
            memory=memory, initial_memory=initial_memory, action_menu=action_menu,
            interaction_links=interaction_links, planner_output=None,
            compiled_plan=None, planner_attempts=planner_attempts,
            scene_results=[], scene_reports=[], initial_record_shas=initial_record_shas,
            failure_stage="Planner v3 schema/compile/selection",
            failure_message="planner_retry_limit_exhausted")
    compiled_plan = planner_result["compiled_round_plan"]
    atomic_json(run_root / "planner_output.json", planner_output)
    atomic_json(run_root / "compiled_scene_plan.json", compiled_plan)

    scene_results: list[dict[str, Any]] = []
    scene_reports: list[dict[str, Any]] = []
    previous_child_blacklist: list[str] = []
    fatal_stage: str | None = None
    fatal_message: str | None = None
    action_by_id = {str(row["action_id"]): row for row in action_menu["actions"]}
    for scene_index, scene in enumerate(compiled_plan["scenes"], 1):
        try:
            report, result, memory = _execute_scene(
                root_id=root_id, child_id=child_id, scene_index=scene_index,
                scene=scene, memory=memory, parent=parent,
                interaction_links=interaction_links, action_by_id=action_by_id,
                previous_child_blacklist=previous_child_blacklist,
                run_root=run_root, caller=caller, limits=limits)
            scene_reports.append(report)
            if result is not None:
                scene_results.append(result)
                if result.get("full_eligible"):
                    previous_child_blacklist.extend(
                        str(row["text"]) for row in result["bound_scene"]["dialogue_turns"])
        except LocalInvariantFailure as error:
            fatal_stage = "local_invariant_failure"
            fatal_message = f"{scene['scene_ref']}:{error}"
            break
        except Exception as error:
            fatal_stage = "local_invariant_failure"
            fatal_message = f"{scene['scene_ref']}:{type(error).__name__}:{error}"
            break
    return _finalize_run(
        root=root, parent=parent, run_root=run_root, caller=caller,
        freeze_manifest=freeze_manifest, input_path=input_path,
        historical_v6_root=historical_v6_root, model_config=model_config,
        memory=memory, initial_memory=initial_memory, action_menu=action_menu,
        interaction_links=interaction_links, planner_output=planner_output,
        compiled_plan=compiled_plan, planner_attempts=planner_attempts,
        scene_results=scene_results, scene_reports=scene_reports,
        initial_record_shas=initial_record_shas,
        failure_stage=fatal_stage, failure_message=fatal_message)


def _execute_scene(
    *, root_id: str, child_id: str, scene_index: int,
    scene: Mapping[str, Any], memory: Mapping[str, Any],
    parent: Mapping[str, Any], interaction_links: Sequence[Mapping[str, Any]],
    action_by_id: Mapping[str, Mapping[str, Any]],
    previous_child_blacklist: Sequence[str], run_root: Path,
    caller: RoleCaller, limits: V7RunLimits,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    scene_ref = str(scene["scene_ref"])
    actions = [action_by_id[str(row["action_id"])] for row in scene["compiled_proposals"]]
    context = compile_scene_context(
        scene=scene, actions=actions,
        authoritative_memory=authoritative_affordance_view(memory),
        parent_turns=parent["dialogue"]["turns"], interaction_links=interaction_links)
    budget = compile_turn_budget(scene)
    atomic_json(run_root / "scenes" / scene_ref / "context_manifest.json", context)
    atomic_json(run_root / "scenes" / scene_ref / "turn_budget.json", budget)
    base_prompt = build_realizer_v3_prompt(scene=scene, context=context, turn_budget=budget)
    prompt = base_prompt
    retry_mode = "full_scene_retry"
    frozen_dialogue: list[Mapping[str, Any]] | None = None
    frozen_dialogue_sha: str | None = None
    realizer_output: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    claims_validation: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    full_retries = claim_retries = 0
    for attempt in range(1, limits.realizer_attempts_per_scene + 1):
        is_claim_only = retry_mode == "claim_only_retry" and frozen_dialogue is not None
        parsed, raw, parse_errors = caller.call_json(
            request_id=f"realizer-v3-{scene_ref.replace(':', '_')}-a{attempt:02d}",
            role="realizer_v3", attempt=attempt, prompt=prompt,
            prompt_version=(NATURAL_REALIZER_CLAIM_RETRY_PROMPT_VERSION_V3
                            if is_claim_only else NATURAL_REALIZER_PROMPT_VERSION_V3
                            if attempt == 1 else NATURAL_REALIZER_FULL_RETRY_PROMPT_VERSION_V3),
            max_tokens=limits.realizer_max_tokens)
        normalization: dict[str, Any]
        freeze_check: dict[str, Any] | None = None
        if is_claim_only:
            shape = validate_claim_only_realizer_shape(parsed)
            if parsed is None or not shape["valid"]:
                normalization = {"valid": False, "errors": parse_errors + shape["errors"]}
            else:
                freeze_check = reconstruct_claim_only_output(
                    frozen_dialogue_turns=list(frozen_dialogue), claim_only_output=parsed)
                normalization = {"valid": True, "errors": [],
                                 "normalized_output": freeze_check["normalized_output"],
                                 "dialogue_sha256": freeze_check["dialogue_sha256_after"]}
        else:
            normalization = (normalize_full_realizer_output(parsed, scene_ref=scene_ref)
                             if parsed is not None else {"valid": False, "errors": parse_errors,
                                                        "normalized_output": None})
        normalized = normalization.get("normalized_output")
        dialogue = list((normalized or {}).get("dialogue_turns") or [])
        normalization_errors = list(normalization.get("errors") or [])
        normalization_claim_errors = [
            error for error in normalization_errors
            if error.startswith("realizer_raw_claim") or
            error == "realizer_proposal_claims_not_list"]
        normalization_dialogue_errors = [
            error for error in normalization_errors
            if error not in normalization_claim_errors]
        quality = (validate_dialogue_quality(
            scene_ref=scene_ref, dialogue_turns=dialogue,
            speaker_participants=scene["speaker_participants"], turn_budget=budget,
            parent_utterances=[str(row["text"]) for row in parent["dialogue"]["turns"]],
            previous_child_utterance_blacklist=previous_child_blacklist)
                   if dialogue else {"valid": False, "errors": normalization.get("errors") or
                                     ["dialogue_normalization_failed"]})
        if normalization_dialogue_errors:
            quality = {**quality, "valid": False,
                       "errors": list(dict.fromkeys([
                           *quality.get("errors", []), *normalization_dialogue_errors]))}
        claims_validation = (validate_proposal_claims(
            scene_ref=scene_ref, proposal_claims=(normalized or {}).get("proposal_claims"),
            compiled_proposals=scene["compiled_proposals"], dialogue_turns=dialogue)
                             if normalized else {"valid": False,
                                                 "errors": normalization.get("errors") or parse_errors})
        if normalization_claim_errors:
            claims_validation = {**claims_validation, "valid": False,
                                 "errors": list(dict.fromkeys([
                                     *claims_validation.get("errors", []),
                                     *normalization_claim_errors]))}
        route = realizer_retry_route(
            json_parsed=parsed is not None, dialogue_quality=quality,
            claim_validation=claims_validation,
            claims=list((normalized or {}).get("proposal_claims") or []),
            requests_used=attempt, maximum_requests=limits.realizer_attempts_per_scene)
        attempt_record = {
            "attempt": attempt, "input_mode": "claim_only" if is_claim_only else "full_scene",
            "normalization": normalization, "dialogue_quality": quality,
            "claim_validation": claims_validation, "retry_route": route,
            "dialogue_sha256": dialogue_sha256(dialogue) if dialogue else None,
            "frozen_dialogue_check": freeze_check,
        }
        attempts.append(attempt_record)
        atomic_json(run_root / "scenes" / scene_ref /
                    f"realizer_validation_attempt_{attempt:02d}.json", attempt_record)
        if route == "submit_to_committer" and normalized is not None:
            realizer_output = dict(normalized)
            break
        if route == "scene_failed":
            break
        if route == "claim_only_retry" and dialogue:
            frozen_dialogue = deepcopy(dialogue)
            frozen_dialogue_sha = dialogue_sha256(frozen_dialogue)
            prompt = build_claim_only_v3_prompt(
                scene_ref=scene_ref, frozen_dialogue_turns=frozen_dialogue,
                compiled_proposals=scene["compiled_proposals"],
                errors=claims_validation["errors"])
            retry_mode = "claim_only_retry"
            claim_retries += 1
        else:
            forbidden = list(quality.get("parent_exact_copies") or []) + list(
                quality.get("previous_child_exact_copies") or []) + list(
                quality.get("internal_exact_duplicates") or [])
            prompt = build_realizer_v3_full_retry_prompt(
                scene=scene, frozen_context_manifest=context,
                frozen_turn_budget=budget,
                quality_errors=list(quality.get("errors") or normalization.get("errors") or []),
                forbidden_exact_utterances=forbidden)
            retry_mode = "full_scene_retry"
            frozen_dialogue = None
            frozen_dialogue_sha = None
            full_retries += 1
    if realizer_output is None:
        report = {"scene_ref": scene_ref, "status": "scene_failed",
                  "failure_stage": "Natural Realizer structure/quality/claim",
                  "action_ids": [str(row["action_id"]) for row in scene["compiled_proposals"]],
                  "turn_budget": budget, "realizer_attempts": attempts,
                  "full_scene_retry_count": full_retries,
                  "claim_only_retry_count": claim_retries}
        atomic_json(run_root / "scenes" / scene_ref / "scene_review_report.json", report)
        return report, None, dict(memory)

    atomic_json(run_root / "scenes" / scene_ref / "realizer_output_normalized.json", realizer_output)
    committer_base = build_committer_scope_prompt(
        compiled_proposals=scene["compiled_proposals"],
        dialogue_turns=realizer_output["dialogue_turns"],
        proposal_claims=realizer_output["proposal_claims"],
        authoritative_world=_authoritative_prompt_view(memory))
    frozen_committer_sha = _semantic_sha(
        scene["compiled_proposals"], realizer_output["dialogue_turns"],
        realizer_output["proposal_claims"], _authoritative_prompt_view(memory))
    committer_output: dict[str, Any] | None = None
    committer_attempts: list[dict[str, Any]] = []
    committer_prompt = committer_base
    for attempt in range(1, limits.committer_attempts_per_scene + 1):
        semantic_sha = _semantic_sha(
            scene["compiled_proposals"], realizer_output["dialogue_turns"],
            realizer_output["proposal_claims"], _authoritative_prompt_view(memory))
        if semantic_sha != frozen_committer_sha:
            raise LocalInvariantFailure("committer_retry_semantic_input_drift")
        parsed, raw, parse_errors = caller.call_json(
            request_id=f"committer-scope-{scene_ref.replace(':', '_')}-a{attempt:02d}",
            role="evidence_committer_scope_v1", attempt=attempt,
            prompt=committer_prompt,
            prompt_version=(EVIDENCE_SCOPE_PROMPT_VERSION if attempt == 1 else
                            EVIDENCE_SCOPE_RETRY_PROMPT_VERSION),
            max_tokens=limits.committer_max_tokens)
        validation = (validate_committer_scope_output(
            parsed, compiled_proposals=scene["compiled_proposals"],
            proposal_claims=realizer_output["proposal_claims"])
                      if parsed is not None else {"valid": False, "errors": parse_errors})
        disposition = committer_retry_disposition(validation, parsed)
        attempt_row = {"attempt": attempt, "semantic_input_sha256": semantic_sha,
                       "validation": validation, "disposition": disposition}
        committer_attempts.append(attempt_row)
        atomic_json(run_root / "scenes" / scene_ref / "committer" /
                    f"validation_attempt_{attempt:02d}.json", attempt_row)
        if validation["valid"]:
            committer_output = dict(parsed)
            break
        committer_prompt = build_committer_scope_retry_prompt(
            frozen_base_prompt=committer_base,
            contract_errors=validation["errors"])
    if committer_output is None:
        report = {"scene_ref": scene_ref, "status": "scene_failed",
                  "failure_stage": "Evidence Committer contract",
                  "action_ids": [str(row["action_id"]) for row in scene["compiled_proposals"]],
                  "turn_budget": budget, "realizer_attempts": attempts,
                  "committer_attempts": committer_attempts,
                  "full_scene_retry_count": full_retries,
                  "claim_only_retry_count": claim_retries}
        atomic_json(run_root / "scenes" / scene_ref / "scene_review_report.json", report)
        return report, None, dict(memory)

    atomic_json(run_root / "scenes" / scene_ref / "committer_output.json", committer_output)
    gate = evaluate_scene_commit(
        compiled_proposals=scene["compiled_proposals"],
        proposal_claims=realizer_output["proposal_claims"],
        committer_output=committer_output)
    transaction: dict[str, Any] | None = None
    committed_refs: list[str] = []
    candidate_memory = dict(memory)
    high_before = deepcopy(memory["high_water_marks"])
    if gate["commit_ready"] and scene["compiled_proposals"]:
        accepted = {str(row["proposal_ref"]): list(row["accepted_evidence_turn_refs"])
                    for row in committer_output["proposal_reviews"]}
        evidence = validate_scene_local_evidence_refs(
            scene_ref=scene_ref, dialogue_turns=realizer_output["dialogue_turns"],
            accepted_evidence_by_proposal=accepted)
        allocation = build_allocation_plan(
            root_id=root_id, round_index=1, scene_index=scene_index,
            compiled_proposals=scene["compiled_proposals"],
            high_water_marks=memory["high_water_marks"])
        candidate_memory = apply_scene_commit(
            memory, scene_ref=scene_ref, round_index=1, scene_index=scene_index,
            compiled_proposals=scene["compiled_proposals"],
            proposal_claims=realizer_output["proposal_claims"],
            committer_output=committer_output, allocation_plan=allocation,
            evidence_final_turn_refs=evidence["evidence_final_turn_refs"],
            evidence_text_sha256=evidence["evidence_text_sha256"])
        committed_refs = [str(row["memory_ref"]) for row in
                          candidate_memory["authoritative_records"][
                              len(memory["authoritative_records"]):]]
        try:
            transaction = commit_scene_transaction(
                run_root=run_root, run_id=run_root.name, scene_plan=scene,
                proposal_fingerprints=[str(row["proposal_fingerprint"])
                                       for row in scene["compiled_proposals"]],
                allocation_plan=allocation,
                artifacts={
                    "authoritative_memory.json": {
                        "root_id": root_id,
                        "entities": candidate_memory["entities"],
                        "authoritative_records": candidate_memory["authoritative_records"],
                        "active_head_index": candidate_memory["active_head_index"],
                        "high_water_marks": candidate_memory["high_water_marks"]},
                    "provenance_ledger.json": candidate_memory["provenance_ledger"],
                    "memory_transitions.json": candidate_memory["memory_transitions"],
                    f"scenes/{scene_ref}/committed_memory_snapshot.json": candidate_memory,
                })
        except Exception as error:
            recovery = recover_prepared_transactions(run_root)
            atomic_json(run_root / "scenes" / scene_ref /
                        "transaction_recovery.json", {
                            "trigger": f"{type(error).__name__}:{error}",
                            "recovery_results": recovery})
            raise LocalInvariantFailure(
                f"transaction_failure:{type(error).__name__}:{error}:recovery={recovery}") from error
        if transaction["status"] != "COMMITTED":
            recovery = recover_prepared_transactions(run_root)
            raise LocalInvariantFailure(
                f"transaction_not_committed:recovery={recovery}")
    elif scene["compiled_proposals"]:
        candidate_memory = apply_scene_commit(
            memory, scene_ref=scene_ref, round_index=1, scene_index=scene_index,
            compiled_proposals=scene["compiled_proposals"],
            proposal_claims=realizer_output["proposal_claims"],
            committer_output=committer_output, allocation_plan={},
            evidence_final_turn_refs={}, evidence_text_sha256={})
    bound_scene = {"scene_ref": scene_ref, "session_ref": f"{scene_ref}:SESSION",
                   "dialogue_turns": deepcopy(realizer_output["dialogue_turns"])}
    full_eligible = bool(quality and quality["valid"] and claims_validation and
                         claims_validation["valid"] and committer_output)
    result = {
        "scene_ref": scene_ref, "quality_valid": bool(quality and quality["valid"]),
        "claims_valid": bool(claims_validation and claims_validation["valid"]),
        "committer_contract_valid": True,
        "compiled_proposals": scene["compiled_proposals"],
        "committer_output": committer_output, "scene_commit_result": gate,
        "bound_scene": bound_scene, "committed_memory_refs": committed_refs,
        "full_eligible": full_eligible,
    }
    speaker_counts = collections.Counter(
        str(row["speaker_ref"]) for row in realizer_output["dialogue_turns"])
    report = {
        "scene_ref": scene_ref, "status": "completed",
        "action_ids": [str(row["action_id"]) for row in scene["compiled_proposals"]],
        "proposal_refs": [str(row["proposal_ref"]) for row in scene["compiled_proposals"]],
        "turn_budget": budget, "actual_turns": len(realizer_output["dialogue_turns"]),
        "speaker_turn_counts": dict(speaker_counts),
        "dialogue_quality": quality, "claim_validation": claims_validation,
        "realizer_attempts": attempts, "full_scene_retry_count": full_retries,
        "claim_only_retry_count": claim_retries,
        "claim_only_frozen_dialogue_sha256": frozen_dialogue_sha,
        "committer_attempts": committer_attempts,
        "committer_output": committer_output, "scene_commit_gate": gate,
        "review_decisions": [str(row["review_decision"])
                             for row in committer_output["proposal_reviews"]],
        "accepted_evidence_turn_refs": {
            str(row["proposal_ref"]): list(row["accepted_evidence_turn_refs"])
            for row in committer_output["proposal_reviews"]},
        "scope_status": committer_output["scope_status"],
        "world_consistency_status": committer_output["world_consistency_status"],
        "transaction": transaction, "high_water_mark_before": high_before,
        "high_water_mark_after": candidate_memory["high_water_marks"],
        "committed_memory_refs": committed_refs,
    }
    atomic_json(run_root / "scenes" / scene_ref / "scene_review_report.json", report)
    return report, result, candidate_memory


def _finalize_run(
    *, root: Mapping[str, Any], parent: Mapping[str, Any], run_root: Path,
    caller: RoleCaller, freeze_manifest: Mapping[str, Any], input_path: Path,
    historical_v6_root: Path, model_config: Mapping[str, Any],
    memory: Mapping[str, Any], initial_memory: Mapping[str, Any],
    action_menu: Mapping[str, Any], interaction_links: Sequence[Mapping[str, Any]],
    planner_output: Mapping[str, Any] | None,
    compiled_plan: Mapping[str, Any] | None,
    planner_attempts: Sequence[Mapping[str, Any]],
    scene_results: Sequence[Mapping[str, Any]],
    scene_reports: Sequence[Mapping[str, Any]],
    initial_record_shas: Mapping[str, str],
    failure_stage: str | None, failure_message: str | None,
) -> dict[str, Any]:
    root_id = str(root["root_id"])
    child_id = f"{root_id}:child:fresh_v7"
    surfaces = assemble_child_surfaces(
        child_id=child_id, root_id=root_id, scene_results=scene_results,
        source_full_child_path=str(run_root / "child_dialogue_full.json"))
    for filename, payload in surfaces.items():
        atomic_json(run_root / filename, payload)
    generated_links: list[dict[str, Any]] = []
    authoritative_entities = {str(row.get("entity_ref") or row.get("entity_id"))
                              for row in memory.get("entities") or []}
    promoted_refs = set(surfaces["promotion_manifest.json"]["promoted_scene_refs"])
    for result in scene_results:
        if (result["scene_ref"] in promoted_refs and
                result["scene_commit_result"].get("commit_ready")):
            speakers = [str(row["speaker_ref"]) for row in
                        result["bound_scene"]["dialogue_turns"]]
            generated_links.extend(compile_generated_interaction_links(
                root_id=root_id, scene_ref=result["scene_ref"], speaker_refs=speakers,
                committed=True, promoted=True,
                authoritative_entity_refs=authoritative_entities))
    atomic_json(run_root / "interaction_links_final.json", {
        "source_grounded": list(interaction_links), "generated_extension": generated_links})
    atomic_json(run_root / "world_memory_final.json", memory)
    atomic_json(run_root / "pending_updates.json", memory.get("pending_updates") or [])
    atomic_json(run_root / "rejected_audit_log.json", memory.get("rejected_audit_log") or [])
    atomic_json(run_root / "reports" / "dialogue_quality_audit.json", list(scene_reports))
    journal_rows = []
    journal_path = run_root / "run_commit_journal.jsonl"
    if journal_path.is_file():
        journal_rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
    transaction_audit = {
        "prepared_count": sum(row.get("status") == "PREPARED" for row in journal_rows),
        "committed_count": sum(row.get("status") == "COMMITTED" for row in journal_rows),
        "staging_clean": not (run_root / "staging").exists() or
                         not any((run_root / "staging").iterdir()),
        "journal_rows": journal_rows,
    }
    atomic_json(run_root / "reports" / "transaction_audit.json", transaction_audit)
    promotion_audit = {
        "child_equals_promotable": surfaces["child_dialogue.json"] ==
                                   surfaces["child_dialogue_promotable.json"],
        "full_scene_count": len(surfaces["child_dialogue_full.json"]["scenes"]),
        "promotable_scene_count": len(surfaces["child_dialogue_promotable.json"]["scenes"]),
        "manifest": surfaces["promotion_manifest.json"],
    }
    atomic_json(run_root / "reports" / "promotion_audit.json", promotion_audit)
    initial_unchanged = all(
        _semantic_sha(next(row for row in memory["authoritative_records"]
                           if row["memory_ref"] == ref)) == digest
        for ref, digest in initial_record_shas.items())
    generated_count = sum(row.get("origin") == "generated_extension"
                          for row in memory["authoritative_records"])
    scene_blocked = sum(row.get("commit_status") == "scene_blocked"
                        for row in memory.get("pending_updates") or [])
    provenance_audit = {
        "source_grounded_record_count": sum(
            row.get("origin") == "source_grounded" for row in memory["authoritative_records"]),
        "generated_extension_committed_count": generated_count,
        "pending_count": len(memory.get("pending_updates") or []),
        "rejected_count": len(memory.get("rejected_audit_log") or []),
        "scene_blocked_count": scene_blocked,
        "old_authoritative_record_sha_unchanged": initial_unchanged,
        "generated_records_verified": all(
            row.get("verification_status") == "semantically_verified"
            for row in memory["authoritative_records"] if row.get("origin") == "generated_extension"),
        "ledger_linkage_complete": all(
            any(entry.get("authoritative_memory_ref") == row.get("memory_ref")
                for entry in memory["provenance_ledger"])
            for row in memory["authoritative_records"]),
    }
    usage = aggregate_role_usage(caller.records)
    ledger_events = run_root / "ledger" / "request_events.jsonl"
    atomic_text(run_root / "request_ledger.jsonl",
                ledger_events.read_text(encoding="utf-8") if ledger_events.is_file() else "")
    minimum = int(((planner_output or {}).get("scene_budget") or {}).get("minimum") or 2)
    promotable_count = promotion_audit["promotable_scene_count"]
    committer_calls = usage["roles"]["evidence_committer"]["physical_requests"]
    generation_ready = bool(
        failure_stage is None and promotable_count >= minimum and committer_calls >= 1 and
        promotion_audit["child_equals_promotable"])
    fresh_ready = bool(
        generation_ready and generated_count >= 1 and
        transaction_audit["committed_count"] >= 1 and
        surfaces["promotion_manifest.json"]["committed_memory_refs"] and
        provenance_audit["ledger_linkage_complete"])
    runtime_import_audit = audit_v7_recursive_import_graph(
        project_root=PROJECT_ROOT, runner_path=Path(__file__))
    current_input_sha = sha256_file(input_path)
    current_parent_dialogue_sha = _semantic_sha(parent["dialogue"])
    current_v6_sha = directory_manifest(historical_v6_root)["directory_manifest_sha256"]
    current_implementation_sha = implementation_manifest(PROJECT_ROOT)[
        "implementation_manifest_sha256"]
    current_working_tree_sha = project_working_tree_manifest_sha256(PROJECT_ROOT)
    current_git_head = _git("rev-parse", "HEAD")
    integrity = {
        "input_sha_before": freeze_manifest["input_sha256"],
        "input_sha_after": current_input_sha,
        "input_sha_unchanged": current_input_sha == freeze_manifest["input_sha256"],
        "parent_dialogue_sha_before": freeze_manifest["parent_dialogue_sha256"],
        "parent_dialogue_sha_after": current_parent_dialogue_sha,
        "parent_dialogue_sha_unchanged": current_parent_dialogue_sha ==
            freeze_manifest["parent_dialogue_sha256"],
        "v6_directory_sha_before": freeze_manifest["v6_directory_manifest_sha256"],
        "v6_directory_sha_after": current_v6_sha,
        "v6_directory_sha_unchanged": current_v6_sha ==
            freeze_manifest["v6_directory_manifest_sha256"],
        "git_head_before": freeze_manifest["git_head"], "git_head_after": current_git_head,
        "git_head_unchanged": current_git_head == freeze_manifest["git_head"],
        "implementation_manifest_sha_before": freeze_manifest["implementation_manifest_sha256"],
        "implementation_manifest_sha_after": current_implementation_sha,
        "implementation_manifest_unchanged": current_implementation_sha ==
            freeze_manifest["implementation_manifest_sha256"],
        "project_working_tree_manifest_sha_before": freeze_manifest[
            "project_working_tree_manifest_sha256"],
        "project_working_tree_manifest_sha_after": current_working_tree_sha,
        "project_working_tree_manifest_unchanged": current_working_tree_sha ==
            freeze_manifest["project_working_tree_manifest_sha256"],
    }
    summary = {
        "schema_version": SUMMARY_VERSION,
        "runner_version": LIVE_FRESH_E2E_V7_RUNNER_VERSION,
        "run_root": str(run_root), "root_id": root_id,
        "failure_stage": failure_stage, "failure_message": failure_message,
        "freeze_manifest": dict(freeze_manifest), "model_config": dict(model_config),
        "action_menu": {"total": len(action_menu["actions"]),
                        "class_distribution": dict(collections.Counter(
                            row["operation_class"] for row in action_menu["actions"])),
                        "operation_distribution": dict(collections.Counter(
                            row["operation"] for row in action_menu["actions"])),
                        "create_action_deficit": action_menu["create_action_deficit"],
                        "connected_memory_refs_by_action": {
                            str(row["action_id"]): list(row["connected_memory_refs"])
                            for row in action_menu["actions"]}},
        "planner": {"attempts": list(planner_attempts),
                    "planner_contract_version": PLANNER_CONTRACT_VERSION,
                    "planner_action_view_version": PLANNER_ACTION_VIEW_VERSION,
                    "scene_count": len((planner_output or {}).get("scene_goals") or []),
                    "output": planner_output},
        "compiled_plan": compiled_plan,
        "scenes": list(scene_reports),
        "memory": {"initial_authoritative_records": len(initial_memory["authoritative_records"]),
                   "final_authoritative_records": len(memory["authoritative_records"]),
                   "generated_extension_count": generated_count,
                   "transition_count": len(memory["memory_transitions"])},
        "transaction_audit": transaction_audit,
        "promotion_audit": promotion_audit,
        "provenance_audit": provenance_audit,
        "integrity": integrity,
        "module_boundary": {
            "runner": str(Path(__file__).resolve()),
            "allowed_live_roles": ["planner_v3", "realizer_v3",
                                   "evidence_committer_scope_v1"],
            "recursive_import_graph_valid": runtime_import_audit["valid"],
            "forbidden_semantic_import_count": runtime_import_audit[
                "forbidden_import_count"],
            "visited_files": runtime_import_audit["visited_files"],
        },
        "api_usage": usage,
        "forbidden_role_calls": {"questioner": 0, "distractor": 0, "solver": 0,
                                 "qa_replay": 0, "strategy": 0, "training": 0,
                                 "legacy_api_roles": 0},
        "cache_imported_legacy_reuse": {"cache": 0, "imported": 0, "legacy": 0},
        "generation_ready": generation_ready,
        "fresh_e2e_ready": fresh_ready, "qa_pipeline_ready": False,
    }
    atomic_json(run_root / "reports" / "slim_fresh_e2e_v7_summary.json", summary)
    atomic_text(run_root / "reports" / "slim_fresh_e2e_v7_summary.md",
                "# Slim Fresh E2E v7 Summary\n\n```json\n" +
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    post_checks = {
        "input_sha_unchanged": integrity["input_sha_unchanged"],
        "parent_dialogue_sha_unchanged": integrity["parent_dialogue_sha_unchanged"],
        "v6_sha_unchanged": integrity["v6_directory_sha_unchanged"],
        "git_head_unchanged": integrity["git_head_unchanged"],
        "implementation_manifest_unchanged": integrity[
            "implementation_manifest_unchanged"],
        "working_tree_manifest_unchanged": integrity[
            "project_working_tree_manifest_unchanged"],
        "legacy_recursive_import_clean": runtime_import_audit["valid"],
    }
    atomic_json(run_root / "run_status.json", {
        "status": "completed" if failure_stage is None else "failed",
        "failure_stage": failure_stage, "generation_ready": generation_ready,
        "fresh_e2e_ready": fresh_ready, "qa_pipeline_ready": False,
        "updated_at_utc": _utc_now()})
    directory = directory_manifest(
        run_root, excluded_relatives={"reports/post_run_provenance.json"})
    post_run = {"post_run_provenance_version": "slim_v7_post_run_provenance_v1",
                "checks": post_checks, "all_checks_pass": all(post_checks.values()),
                "v7_directory_manifest": directory,
                "api_usage": usage, "forbidden_role_calls": summary["forbidden_role_calls"]}
    atomic_json(run_root / "reports" / "post_run_provenance.json", post_run)
    return summary


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--historical-v6-root", type=Path, default=DEFAULT_V6_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--freeze-report", type=Path, default=DEFAULT_FREEZE_REPORT)
    args = parser.parse_args(argv)
    try:
        summary = run(
            input_path=args.input, run_root=args.run_root,
            historical_v6_root=args.historical_v6_root,
            api_config_path=args.api_config, tokenizer_path=args.tokenizer,
            freeze_report_path=args.freeze_report)
    except PreflightFailure as error:
        print(str(error))
        return 2
    print(json.dumps({key: summary[key] for key in (
        "run_root", "failure_stage", "generation_ready",
        "fresh_e2e_ready", "qa_pipeline_ready")}, indent=2))
    return 1 if summary["failure_stage"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
