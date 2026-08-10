"""Single physical-request Planner-only probe for the repaired v11 contract."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends import BackendResponse
from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from .continuous_round_plan_v1 import (
    audit_planner_output_size_v11, build_scene_dependency_dag,
)
from .live_contracts_v11 import PLANNER_PROMPT_VERSION_V11, prompt_schema_versions_v11
from .live_preflight_b2_v11 import (
    DEFAULT_API_CONFIG, DEFAULT_INPUT, DEFAULT_TOKENIZER, PACKAGE_ROOT,
    PROJECT_ROOT, implementation_manifest_v11,
)
from .live_role_runtime_v11 import LiveJsonRoleCallerV11, aggregate_v11_usage
from .narrative_action_contracts import (
    NARRATIVE_ACTION_CONTRACTS, action_instance_fingerprint,
)
from .narrative_action_continuation import action_slot_ref
from .placement_choice_registry import (
    validate_placement_choice_registry, validate_source_anchor_registry,
)
from .planner_only_probe_v11 import ROOT_ID, SEED, build_planner_probe_input
from .planner_outline_v11 import (
    compile_planner_outline_v11, validate_planner_outline_v11,
)
from .planner_transport_v11 import (
    PLANNER_MAX_OUTPUT_TOKENS, PLANNER_TRANSPORT_RECOVERY_VERSION,
    PlannerTransportRecoveryClientV11, PlannerTransportSettingsV11,
    PlannerTransportTimeoutV11, build_frozen_planner_transport_request_v11,
    validate_frozen_planner_request_v11,
)
from .provenance import git_value, hash_tree, project_diff_sha256
from .structural_expansion_gate import validate_round_blueprint_structure


PROBE_VERSION = "v11_planner_only_probe_5_v1"
PROBE_DISPLAY_NAME = "V11 Planner-only Probe 5"
PROBE_RUN_SUFFIX = "v11_planner_probe_5"
PREFLIGHT_VERSION = "v11_planner_probe_5_preflight_v1"
REQUEST_ID = "planner_v11_probe_5_request_01"
REPORT_BASENAME = "V11_PLANNER_PROBE_5_REPORT"
REQUEST_PROMPT_VERSION = PLANNER_PROMPT_VERSION_V11
LAUNCHER_PATH = Path(__file__).resolve()
RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_5")
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 900
STREAMING = True
TIMEOUT_RETRY_LIMIT = 0
CONTEXT_LIMIT = 40_960
EXPECTED_INPUT_SHA = "c8976ae6bbb2c91403466a3987d4054bad099475c8eac16c9ab86a7be7244332"

HISTORY_PATHS = {
    "v10": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v10"),
    "live_canary_1": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_1"),
    "phase_a": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_a_preflight"),
    "phase_b1": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_b1_preflight"),
    "probe_1": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_planner_transport_probe_1"),
    "probe_2": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_2"),
    "probe_3": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_3"),
    "probe_4": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_4"),
}
EXPECTED_HISTORY = {
    "v10": "6c35efbaaa4b21cc989eee8cf721e8146b016360683cd6c2c43fc8a922e90b54",
    "live_canary_1": "4e0c7a7d0e390ce3a43167dce5f7031a25166a5336feaa8fd3de9f8f58c607b3",
    "phase_a": "de39fabcae653d49e408911bedd4486e1426e072277fca075d2b8322738c6eeb",
    "phase_b1": "1c03226386bd06a29e9782ae2fd928910ff535606a54fa633faa29ddc8af6d54",
    "probe_1": "653514bfbd3b7325bab1315816207b66b01db6022cb9d1f515663df3c437a0aa",
    "probe_2": "43eef09b5ff1b5dde62e0946bed98d754f30bc2b8a27e65c3d55abdd9d974256",
    "probe_3": "77a2d96cd65cd845c2d3fe52e49937e7df24a96652e74965cbbf32887be25fb7",
    "probe_4": "5ddb7d818ff2736803a7079cbf0a640e7b0a5113b71dd21ad66e0dbddb649f5b",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema_sha() -> str:
    return hashlib.sha256(json.dumps(
        prompt_schema_versions_v11(), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def _history_hashes() -> dict[str, str]:
    return {name: hash_tree(path)["tree_sha256"]
            for name, path in HISTORY_PATHS.items()}


def prepare_candidate_response(
    parsed: Mapping[str, Any], values: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Identity hook overridden only by frozen patch-response probes."""
    del values
    return parsed, None, []


def _planner_contract_audit(
    parsed: Any, *, values: Mapping[str, Any], validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scenes = list(parsed.get("scenes") or []) if isinstance(parsed, Mapping) else []
    source_ids = {str(row.get("source_anchor_id"))
                  for row in values["source_anchor_registry"].get("source_anchors") or []}
    placement_ids = {str(row.get("placement_choice_id"))
                     for row in values["placement_choice_registry"].get("choices") or []}
    legacy_fields = {"candidate_update_hints", "hint_mode", "action_id", "bundle_id"}
    legacy_paths: list[str] = []

    def scan(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in legacy_fields:
                    legacy_paths.append(f"{path}.{key}")
                scan(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, f"{path}[{index}]")

    scan(parsed, "$planner")
    instances_by_type: dict[str, list[str]] = {}
    primary_complete = True
    secondary_shape = True
    action_types_registered = True
    action_modes_complete = bool(scenes)
    action_branch_shapes_valid = bool(scenes)
    continuation_refs_resolved = True
    continuation_ancestry_valid = True
    continuation_progressions_valid = True
    continuation_count = 0
    resolution = ((validation or {}).get("action_continuation_validation") or {}).get(
        "resolved_action_slots") or {}
    continuation_errors = ((validation or {}).get(
        "action_continuation_validation") or {}).get("errors") or []
    for scene_index, scene in enumerate(scenes, 1):
        if not isinstance(scene, Mapping):
            primary_complete = secondary_shape = action_types_registered = False
            continue
        primary = scene.get("primary_action")
        secondary = scene.get("secondary_action")
        if not isinstance(primary, Mapping):
            primary_complete = False
        if secondary is not None and not isinstance(secondary, Mapping):
            secondary_shape = False
        for role, action in (("primary_action", primary),
                             ("secondary_action", secondary)):
            if not isinstance(action, Mapping):
                continue
            slot_ref = action_slot_ref(scene_index, role)
            resolved = resolution.get(slot_ref) or {}
            mode = action.get("action_mode")
            if mode not in {"new", "continuation"}:
                action_modes_complete = False
                action_branch_shapes_valid = False
            elif mode == "new":
                action_branch_shapes_valid = (
                    action_branch_shapes_valid and
                    set(action) == {"action_mode", "action_type", "arguments"})
            else:
                continuation_count += 1
                action_branch_shapes_valid = (
                    action_branch_shapes_valid and
                    set(action) == {"action_mode", "continuation_of", "progression"})
                continuation_refs_resolved = (
                    continuation_refs_resolved and bool(resolved) and
                    resolved.get("continuation_of") == action.get("continuation_of"))
                continuation_ancestry_valid = (
                    continuation_ancestry_valid and bool(resolved.get("dependency_path")))
                progression = action.get("progression")
                continuation_progressions_valid = (
                    continuation_progressions_valid and
                    isinstance(progression, Mapping) and
                    progression.get("stage") in {
                        "development", "escalation", "consolidation", "resolution"} and
                    isinstance(progression.get("new_step"), str) and
                    bool(progression.get("new_step", "").strip()))
            action_type = str(action.get("action_type") or
                              resolved.get("resolved_action_type") or "")
            action_types_registered = (
                action_types_registered and action_type in NARRATIVE_ACTION_CONTRACTS)
            if action.get("action_mode") == "new":
                instances_by_type.setdefault(action_type, []).append(
                    action_instance_fingerprint(action))
    repeated = {key: values for key, values in instances_by_type.items()
                if len(values) > 1}
    repeated_distinct = all(len(set(values)) == len(values)
                            for values in repeated.values())
    checks = {
        "complete_json_object": isinstance(parsed, Mapping),
        "scene_count_5_to_7": 5 <= len(scenes) <= 7,
        "every_scene_has_primary_action": primary_complete and bool(scenes),
        "secondary_action_object_or_null": secondary_shape and bool(scenes),
        "every_action_has_valid_action_mode": action_modes_complete,
        "new_continuation_branches_mutually_exclusive": action_branch_shapes_valid,
        "continuation_action_slots_resolve": continuation_refs_resolved,
        "continuation_sources_in_dependency_ancestry": continuation_ancestry_valid,
        "continuation_progressions_valid": continuation_progressions_valid,
        "no_redundant_independent_new_action": not any(
            "redundant_independent_action" in str(error)
            for error in continuation_errors),
        "all_action_types_registered": action_types_registered and bool(scenes),
        "action_argument_schemas_valid": bool(validation and validation.get("valid")),
        "repeated_action_instances_have_distinct_arguments": repeated_distinct,
        "source_anchor_ids_valid": bool(scenes) and all(
            str(scene.get("source_anchor_id")) in source_ids for scene in scenes
            if isinstance(scene, Mapping)),
        "placement_choice_ids_valid": bool(scenes) and all(
            str(scene.get("placement_choice_id")) in placement_ids for scene in scenes
            if isinstance(scene, Mapping)),
        "desired_outcomes_complete": bool(scenes) and all(
            isinstance(scene.get("desired_outcome"), str) and
            bool(scene.get("desired_outcome", "").strip()) for scene in scenes
            if isinstance(scene, Mapping)),
        "dependency_and_merge_contract_valid": bool(validation and validation.get("valid")),
        "legacy_action_allocation_fields_absent": not legacy_paths,
    }
    return {
        "valid": all(checks.values()), "checks": checks,
        "scene_count": len(scenes),
        "continuation_count": continuation_count,
        "action_type_counts": {key: len(values)
                               for key, values in sorted(instances_by_type.items())},
        "repeated_action_types": sorted(repeated),
        "legacy_field_paths": legacy_paths,
    }


class SingleAttemptPlannerClientV11(PlannerTransportRecoveryClientV11):
    """Use the recovered streaming path without transport retry."""

    def complete_planner(self, *, prompt: str, max_tokens: int,
                         json_object: bool) -> BackendResponse:
        if not json_object:
            raise ValueError("planner_probe_json_object_required")
        if int(max_tokens) != self.settings.max_output_tokens:
            raise ValueError("planner_probe_output_limit_drift")
        if self.settings.timeout_retry_limit != 0:
            raise ValueError("planner_probe_retry_must_be_zero")
        request = build_frozen_planner_transport_request_v11(
            config=self.config, prompt=prompt,
            authoritative_input_sha256=self.authoritative_input_sha256,
            orchestration_seed=self.orchestration_seed,
            prompt_schema_contract_sha256=self.prompt_schema_contract_sha256,
            settings=self.settings)
        validation = validate_frozen_planner_request_v11(request)
        if not validation["valid"]:
            raise RuntimeError("planner_probe_transport_envelope_invalid")
        response = self._send_once(request, 1)
        metadata = dict(response.metadata or {})
        metadata.update({
            "planner_transport_recovery_version": PLANNER_TRANSPORT_RECOVERY_VERSION,
            "probe_transport_policy": "single_physical_attempt_no_retry",
            "transport_attempt_count": 1,
            "transport_retry_count": 0,
            "request_fingerprint_sha256": request.request_fingerprint_sha256,
            "connect_timeout_seconds": self.settings.connect_timeout_seconds,
            "read_timeout_seconds": self.settings.read_timeout_seconds,
            "streaming": self.settings.streaming,
            "planner_max_output_tokens": self.settings.max_output_tokens,
            "planner_output_limit_wire_field": "max_tokens",
        })
        return BackendResponse(response.raw_text, response.model_identifier, metadata)


def _no_retry_self_test(config, schema_sha: str) -> dict[str, Any]:
    settings = PlannerTransportSettingsV11(
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds=READ_TIMEOUT_SECONDS, streaming=STREAMING,
        max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS,
        timeout_retry_limit=TIMEOUT_RETRY_LIMIT)

    class TimeoutClient(SingleAttemptPlannerClientV11):
        calls = 0

        def _send_once(self, request, transport_attempt):
            del request, transport_attempt
            self.calls += 1
            raise PlannerTransportTimeoutV11("read_timeout_before_first_event")

    client = TimeoutClient(
        config=config, authoritative_input_sha256=EXPECTED_INPUT_SHA,
        orchestration_seed=SEED, prompt_schema_contract_sha256=schema_sha,
        settings=settings)
    classification = None
    try:
        client.complete_planner(
            prompt="offline-no-retry-self-test",
            max_tokens=PLANNER_MAX_OUTPUT_TOKENS, json_object=True)
    except PlannerTransportTimeoutV11 as error:
        classification = error.classification
    return {"valid": client.calls == 1 and classification ==
            "read_timeout_before_first_event", "simulated_calls": client.calls,
            "classification": classification, "network_attempts": 0}


def compile_preflight(
    *, expected_git_head: str, expected_implementation_sha: str,
    expected_working_tree_sha: str, expected_schema_sha: str,
    expected_prompt_sha: str, expected_planner_input_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = build_planner_probe_input(DEFAULT_INPUT)
    config = load_challenger_config(DEFAULT_API_CONFIG)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, local_files_only=True)
    prompt_tokens = len(tokenizer.encode(values["prompt"], add_special_tokens=False))
    schema_sha = _schema_sha()
    implementation = implementation_manifest_v11()
    history = _history_hashes()
    runner_path = LAUNCHER_PATH.resolve()
    retry_test = _no_retry_self_test(config, schema_sha)
    checks = {
        "run_root_absent": not RUN_ROOT.exists(),
        "run_root_exact": RUN_ROOT.name.endswith(PROBE_RUN_SUFFIX),
        "runner_exists": runner_path.is_file(),
        "git_head_frozen": git_value(PROJECT_ROOT, "rev-parse", "HEAD") == expected_git_head,
        "implementation_frozen": implementation[
            "implementation_manifest_sha256"] == expected_implementation_sha,
        "working_tree_frozen": project_diff_sha256(
            PROJECT_ROOT) == expected_working_tree_sha,
        "schema_frozen": schema_sha == expected_schema_sha,
        "prompt_frozen": values["prompt_sha256"] == expected_prompt_sha,
        "planner_input_frozen": values[
            "frozen_planner_input_sha256"] == expected_planner_input_sha,
        "input_frozen": sha256_file(DEFAULT_INPUT) == EXPECTED_INPUT_SHA,
        "history_read_only_baseline": history == EXPECTED_HISTORY,
        "original_parent_only": values["complete_dialogue"]["turn_count"] == 60 and
            values["complete_dialogue"]["rounds"] == [],
        "source_anchor_registry_valid": validate_source_anchor_registry(
            values["source_anchor_registry"])["valid"],
        "placement_choice_registry_valid": validate_placement_choice_registry(
            values["placement_choice_registry"],
            complete_dialogue=values["complete_dialogue"])["valid"],
        "prompt_and_completion_fit_context": (
            prompt_tokens + PLANNER_MAX_OUTPUT_TOKENS <= CONTEXT_LIMIT),
        "max_output_tokens_10240": PLANNER_MAX_OUTPUT_TOKENS == 10_240,
        "connect_timeout_30": CONNECT_TIMEOUT_SECONDS == 30,
        "read_timeout_900": READ_TIMEOUT_SECONDS == 900,
        "streaming_enabled": STREAMING is True,
        "automatic_retry_disabled": TIMEOUT_RETRY_LIMIT == 0,
        "no_retry_self_test": retry_test["valid"],
        "compileall": compileall.compile_dir(str(PACKAGE_ROOT), quiet=1),
        "git_diff_check": subprocess.run(
            ["git", "diff", "--check"], cwd=PROJECT_ROOT,
            capture_output=True).returncode == 0,
    }
    errors = [name for name, passed in checks.items() if not passed]
    preflight = {
        "preflight_version": PREFLIGHT_VERSION,
        "valid": not errors, "errors": errors, "checks": checks,
        "api_calls": 0, "network_attempts": 0,
        "run_root": str(RUN_ROOT), "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "input_path": str(DEFAULT_INPUT), "input_sha256": EXPECTED_INPUT_SHA,
        "parent_dialogue_sha256": sha256_bytes(canonical_json(
            [values["parent"]["dialogue"]]).encode()),
        "parent_turn_count": values["complete_dialogue"]["turn_count"],
        "parent_session_count": len({row["session_ref"]
            for row in values["complete_dialogue"]["turns"]}),
        "prompt_sha256": values["prompt_sha256"],
        "prompt_tokens_local": prompt_tokens,
        "frozen_planner_input_sha256": values["frozen_planner_input_sha256"],
        "prompt_schema_versions": prompt_schema_versions_v11(),
        "prompt_schema_contract_sha256": schema_sha,
        "git_head": expected_git_head,
        "implementation_manifest": implementation,
        "working_tree_manifest_sha256": expected_working_tree_sha,
        "history_tree_sha256": history,
        "api_config_path": str(DEFAULT_API_CONFIG),
        "api_config_sha256": sha256_file(DEFAULT_API_CONFIG),
        "model": config.model, "backend_base_url": config.base_url,
        "temperature": config.temperature, "seed": SEED,
        "tokenizer_path": str(DEFAULT_TOKENIZER),
        "transport": {"connect_timeout_seconds": CONNECT_TIMEOUT_SECONDS,
            "read_timeout_seconds": READ_TIMEOUT_SECONDS,
            "streaming": STREAMING,
            "max_output_tokens": PLANNER_MAX_OUTPUT_TOKENS,
            "wire_limit_field": "max_tokens", "timeout_retry_limit": 0},
        "context_limit": CONTEXT_LIMIT,
        "request_budget": {"planner_logical": 1, "planner_physical": 1,
            "realizer": 0, "round_validator": 0, "evidence_committer": 0,
            "qa_solver_training": 0},
        "no_retry_self_test": retry_test,
    }
    return preflight, values


def _freeze_guard(freeze: Mapping[str, Any]) -> dict[str, Any]:
    values = build_planner_probe_input(DEFAULT_INPUT)
    checks = {
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD") == freeze["git_head"],
        "implementation": implementation_manifest_v11()[
            "implementation_manifest_sha256"] == freeze[
                "implementation_manifest"]["implementation_manifest_sha256"],
        "working_tree": project_diff_sha256(PROJECT_ROOT) == freeze[
            "working_tree_manifest_sha256"],
        "input": sha256_file(DEFAULT_INPUT) == freeze["input_sha256"],
        "prompt_schema": _schema_sha() == freeze["prompt_schema_contract_sha256"],
        "prompt": values["prompt_sha256"] == freeze["prompt_sha256"],
        "planner_input": values["frozen_planner_input_sha256"] == freeze[
            "frozen_planner_input_sha256"],
        "history": _history_hashes() == freeze["history_tree_sha256"],
        "api_config": sha256_file(DEFAULT_API_CONFIG) == freeze["api_config_sha256"],
        "runner": sha256_file(Path(freeze["runner_path"])) == freeze["runner_sha256"],
    }
    return {"valid": all(checks.values()), "checks": checks}


def _report(summary: Mapping[str, Any]) -> str:
    usage = summary["api_usage"]
    return f"""# {PROBE_DISPLAY_NAME}

- Final choice: `{summary['final_choice']}`
- Transport completed: `{str(summary['transport_completed']).lower()}`
- JSON parsed: `{str(summary['json_parsed']).lower()}`
- Planner schema valid: `{str(summary['planner_schema_valid']).lower()}`
- Compile / DAG / structural gate: `{str(summary['compile_valid']).lower()} / {str(summary['dag_valid']).lower()} / {str(summary['structural_gate_valid']).lower()}`
- Scene count: `{summary['scene_count']}`
- Planner logical / physical requests: `{usage['roles']['planner']['logical_requests']} / {usage['roles']['planner']['physical_requests']}`
- Planner retries: `0`
- Realizer / Round Validator / Evidence Committer: `0 / 0 / 0`
- Prompt / completion / total tokens: `{usage['prompt_tokens']} / {usage['completion_tokens']} / {usage['total_tokens']}`
- Latency seconds: `{summary['latency_seconds']}`
- Historical integrity: `{str(summary['post_run_integrity']['valid']).lower()}`

This probe made exactly one Planner request and stopped after local validation.
No output was repaired and no downstream role or complete canary was started.
"""


def execute(preflight: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    if not preflight["valid"] or RUN_ROOT.exists():
        raise RuntimeError("planner_probe_preflight_or_freshness_failure")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    freeze = {**deepcopy(dict(preflight)), "probe_version": PROBE_VERSION,
              "fresh": True, "no_resume": True, "frozen_at": _now()}
    atomic_json(RUN_ROOT / "freeze_manifest.json", freeze)
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "initialized", "api_calls": 0, "created_at": _now()})
    for relative, value in {
        "complete_dialogue_view.json": values["complete_dialogue"],
        **({"source_turn_registry.json": values["source_turn_registry"]}
           if "source_turn_registry" in values else {}),
        "planning_state_view.json": values["planning_state"],
        "anchor_registry.json": values["anchor_registry"],
        "source_anchor_registry.json": values["source_anchor_registry"],
        "placement_choice_registry.json": values["placement_choice_registry"],
        "internal_durable_affordance_menu.json": values["action_menu"],
        "action_availability_view.json": values["action_availability_view"],
        "shared_narrative_action_contracts.json": values[
            "shared_narrative_action_contracts"],
        **({"growth_profile.json": values["growth_profile"]}
           if "growth_profile" in values else {}),
    }.items():
        atomic_json(RUN_ROOT / relative, value)
    for relative, value in (values.get("extra_probe_artifacts") or {}).items():
        atomic_json(RUN_ROOT / str(relative), value)
    atomic_text(RUN_ROOT / "planner_prompt.txt", values["prompt"])
    atomic_text(RUN_ROOT / "probe_launcher_source.py",
                Path(preflight["runner_path"]).read_text())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, local_files_only=True)
    settings = PlannerTransportSettingsV11(
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds=READ_TIMEOUT_SECONDS, streaming=STREAMING,
        max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS, timeout_retry_limit=0)
    client = SingleAttemptPlannerClientV11(
        config=load_challenger_config(DEFAULT_API_CONFIG),
        authoritative_input_sha256=freeze["input_sha256"],
        orchestration_seed=SEED,
        prompt_schema_contract_sha256=freeze["prompt_schema_contract_sha256"],
        settings=settings)
    ledger = RequestLedger(
        RUN_ROOT, freeze["implementation_manifest"][
            "implementation_manifest_sha256"], "live")

    def before_request(logical_role: str, role_version: str) -> None:
        if logical_role != "planner" or role_version != "anchor_first_continuous_planner_v11":
            raise RuntimeError("planner_probe_forbidden_role")
        guard = _freeze_guard(freeze)
        if not guard["valid"]:
            atomic_json(RUN_ROOT / "pre_request_integrity_failure.json", guard)
            raise RuntimeError("planner_probe_freeze_drift")
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": "planner_request_started", "started_at": _now(),
            "all_other_roles_enabled": False, "automatic_retry_enabled": False})

    caller = LiveJsonRoleCallerV11(
        run_root=RUN_ROOT, ledger=ledger, client=client,
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        root_id=ROOT_ID,
        role_budget_limits={"planner": 1, "realizer": 0, "round_validator": 0},
        total_request_limit=1, context_limit=CONTEXT_LIMIT,
        before_request=before_request)
    parsed = None
    parse_errors: list[str] = []
    runtime_error = None
    timeout_classification = None
    validation = None
    compiled = None
    dag = None
    structural = None
    size_audit = None
    contract_audit = None
    response_transform_audit = None
    transform_errors: list[str] = []
    candidate_parsed = None
    started = time.monotonic()
    try:
        parsed, _, parse_errors = caller.call_json(
            request_id=REQUEST_ID, role="planner_v11",
            attempt=1, prompt=values["prompt"],
            prompt_version=REQUEST_PROMPT_VERSION,
            max_tokens=PLANNER_MAX_OUTPUT_TOKENS)
        if parsed is not None:
            candidate_parsed, response_transform_audit, transform_errors = (
                prepare_candidate_response(parsed, values))
            size_audit = audit_planner_output_size_v11(
                candidate_parsed if candidate_parsed is not None else parsed,
                token_counter=lambda text: len(tokenizer.encode(
                    text, add_special_tokens=False)))
        if candidate_parsed is not None:
            validation = validate_planner_outline_v11(
                candidate_parsed, source_anchor_registry=values["source_anchor_registry"],
                placement_choice_registry=values["placement_choice_registry"],
                action_availability_view=values["action_availability_view"],
                authoritative_entity_refs={row["entity_ref"]
                                           for row in values["role_table"]})
            contract_audit = _planner_contract_audit(
                candidate_parsed, values=values, validation=validation)
            if validation["valid"]:
                compiled = compile_planner_outline_v11(
                    candidate_parsed,
                    source_anchor_registry=values["source_anchor_registry"],
                    placement_choice_registry=values["placement_choice_registry"],
                    planning_state=values["planning_state"],
                    action_availability_view=values["action_availability_view"],
                    round_index=1)
                dag = build_scene_dependency_dag(compiled)
                structural = validate_round_blueprint_structure(
                    compiled_plan=compiled,
                    source_anchor_registry=values["source_anchor_registry"],
                    placement_choice_registry=values["placement_choice_registry"],
                    complete_dialogue=values["complete_dialogue"],
                    planning_state=values["planning_state"])
    except PlannerTransportTimeoutV11 as error:
        runtime_error = f"{type(error).__name__}:{error}"
        timeout_classification = error.classification
    except Exception as error:
        runtime_error = f"{type(error).__name__}:{error}"
    latency = round(time.monotonic() - started, 6)

    atomic_json(RUN_ROOT / "planner_response_parsed.json", parsed)
    atomic_json(RUN_ROOT / "planner_response_transform_audit.json",
                response_transform_audit)
    if candidate_parsed is not None and candidate_parsed is not parsed:
        atomic_json(RUN_ROOT / "planner_reconstructed_outline.json", candidate_parsed)
    atomic_json(RUN_ROOT / "planner_validation.json", {
        "parse": {"valid": parsed is not None and not parse_errors,
                  "errors": parse_errors},
        "response_transform": {
            "valid": candidate_parsed is not None and not transform_errors,
            "errors": transform_errors, "audit": response_transform_audit},
        "schema": validation, "compile_valid": compiled is not None,
        "dependency_dag": dag, "structural_gate": structural,
        "planner_contract_audit": contract_audit,
        "response_transform_audit": response_transform_audit,
        "output_size_audit": size_audit, "runtime_error": runtime_error,
        "timeout_classification": timeout_classification})
    if compiled is not None:
        atomic_json(RUN_ROOT / "compiled_round_plan.json", compiled)
    if dag is not None:
        atomic_json(RUN_ROOT / "scene_dependency_dag.json", dag)

    usage = aggregate_v11_usage(caller.records)
    transport_completed = bool(caller.records and caller.records[-1].get(
        "request_completed"))
    finish_reason = ((caller.records[-1].get("response_metadata") or {}).get(
        "finish_reason") if caller.records else None)
    output_truncated = finish_reason == "length"
    all_contracts = bool(
        transport_completed and finish_reason == "stop" and
        parsed is not None and not parse_errors and
        not transform_errors and candidate_parsed is not None and
        validation and validation["valid"] and compiled is not None and dag is not None and
        structural and structural["valid"] and size_audit and size_audit["valid"] and
        contract_audit and contract_audit["valid"])
    if (timeout_classification or runtime_error or output_truncated or
            not transport_completed or finish_reason != "stop"):
        final_choice = "C"
    elif all_contracts:
        final_choice = "A"
    else:
        final_choice = "B"
    post = _freeze_guard(freeze)
    summary = {
        "probe_version": PROBE_VERSION,
        "status": "completed" if transport_completed else "failed",
        "final_choice": final_choice, "runtime_error": runtime_error,
        "timeout_classification": timeout_classification,
        "finish_reason": finish_reason, "output_truncated": output_truncated,
        "finish_reason_stop": finish_reason == "stop",
        "transport_completed": transport_completed,
        "json_parsed": parsed is not None and not parse_errors,
        "response_transform_valid": (
            candidate_parsed is not None and not transform_errors),
        "planner_schema_valid": bool(validation and validation["valid"]),
        "dialogue_first_participant_coverage": (
            deepcopy(validation.get("dialogue_first_participant_coverage"))
            if validation and "dialogue_first_participant_coverage" in validation
            else None),
        "compile_valid": compiled is not None, "dag_valid": dag is not None,
        "structural_gate_valid": bool(structural and structural["valid"]),
        "output_size_valid": bool(size_audit and size_audit["valid"]),
        "planner_contract_audit": contract_audit,
        "scene_count": int((validation or {}).get("scene_count") or 0),
        "latency_seconds": latency, "api_usage": usage,
        "automatic_retry_count": 0, "post_run_integrity": post,
        "realizer_calls": 0, "round_validator_calls": 0,
        "evidence_committer_calls": 0, "qa_calls": 0, "solver_calls": 0,
        "training_calls": 0, "round_2_calls": 0,
    }
    atomic_json(RUN_ROOT / "api_usage.json", usage)
    atomic_json(RUN_ROOT / f"reports/{REPORT_BASENAME}.json", summary)
    atomic_text(RUN_ROOT / f"reports/{REPORT_BASENAME}.md", _report(summary))
    atomic_json(RUN_ROOT / "reports/post_run_integrity.json", post)
    if ledger.events.is_file():
        atomic_text(RUN_ROOT / "request_ledger.jsonl",
                    ledger.events.read_text(encoding="utf-8"))
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": summary["status"], "final_choice": final_choice,
        "runtime_error": runtime_error, "completed_at": _now(),
        "api_usage": usage})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single Planner-only v11 probe 5")
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--expected-implementation-sha", required=True)
    parser.add_argument("--expected-working-tree-sha", required=True)
    parser.add_argument("--expected-schema-sha", required=True)
    parser.add_argument("--expected-prompt-sha", required=True)
    parser.add_argument("--expected-planner-input-sha", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight, values = compile_preflight(
        expected_git_head=args.expected_git_head,
        expected_implementation_sha=args.expected_implementation_sha,
        expected_working_tree_sha=args.expected_working_tree_sha,
        expected_schema_sha=args.expected_schema_sha,
        expected_prompt_sha=args.expected_prompt_sha,
        expected_planner_input_sha=args.expected_planner_input_sha)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["valid"] else 2
    if not preflight["valid"]:
        print(json.dumps({"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = execute(preflight, values)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["transport_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
