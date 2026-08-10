"""Generate Planner Contract v3.1 zero-API audits and the next-canary freeze."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import urllib.request
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from ..live_backends import OpenAICompatibleChallengerClient
from ..unified_dialogue_root import read_unified_parent
from .controlled_expansion import compile_action_menu_v3
from .freeze_v7_live_runner import (
    API_CONFIG_PATH, INPUT_PATH, TOKENIZER_PATH, V6_ROOT, V7_ROOT, V8_ROOT,
    generate_live_v7_freeze,
)
from .interaction_links import compile_source_interaction_links
from .live_fresh_e2e_v7 import PROJECT_ROOT
from .live_preflight_v7 import run_live_preflight_v7
from .offline_preflight_v3 import run_fixed_manifest_preflight
from .planner_contract_v3_1 import (
    FROZEN_PLANNER_FIELDS, PLANNER_ACTION_VIEW_VERSION,
    PLANNER_CONTRACT_VERSION, compile_allowed_proposal_bundles,
    compile_planner_action_view, proposal_novelty_key,
    validate_planner_action_view,
)
from .planner_v3 import (
    PLANNER_PROMPT_VERSION_V3, PLANNER_RETRY_PROMPT_VERSION_V3,
    PLANNER_SCHEMA_VERSION_V3, build_planner_v3_prompt,
    build_planner_v3_retry_prompt,
)
from .provenance import hash_tree
from .round_plan_validation import compile_validate_round_plan
from .semantic_import_audit_v7 import audit_v7_recursive_import_graph
from .source_memory_initializer_v1 import (
    authoritative_affordance_view, initialize_source_world_memory,
)
from .v7_manifest import (directory_manifest, implementation_manifest,
                          project_working_tree_manifest_sha256)


REPORT_VERSION = "world_memory_planner_contract_v3_1_freeze_report_v1"
OLD_FREEZE = {
    "implementation_manifest_sha256":
        "8417d5497ebe0fd1263ccc8b3607b5d883480cb57abcd24bb168c873d9c9693e",
    "project_working_tree_manifest_sha256":
        "2c81707ae403798419686030eb9083e3c2b9dec8e1b6ce8ed0cba7e3c9932539",
    "prompt_schema_contract_sha256":
        "4025e75dbe66ff81eca2f5b689700bdd7a71c001698c3165d2cfe84cd6982328",
}
INPUT_EXPECTED = "c8976ae6bbb2c91403466a3987d4054bad099475c8eac16c9ab86a7be7244332"
V6_TREE_EXPECTED = "7121f9756bda86bc972463ce748e344f276d145c420cb9b656dd3332b5375b4d"
V7_TREE_EXPECTED = "4d0fea20040bf849678f56bd6e4f9c88c5cfaa94f16e7381e16ccd0eeba2e4aa"
V6_RUNNER_EXPECTED = "69763e2493f88eee8e74fb4a6492013bccc52b5109c5ce5ba888839bc406ea03"
NON_PLANNER_EXPECTED = {
    "realizer_v3.py": "b6ae9dab749cfcb7eb39d7e0f833b9e2d87e2fcab8fd5e7cd4ecb8b16be8a463",
    "dialogue_quality.py": "6cc41bbf5392e6e6a4da7002aba185e60f932f95defd5f493bcd134d4ced9380",
    "claim_validator_v2.py": "78666101a85593368fd723999c97f60b5e5d11caca12a03e3e89f142d685d090",
    "evidence_scope.py": "e45d8e77ae1e725f4f1650b68b7ac86e2de6248dd2ee348a1b68f53da374862a",
    "transactional_binder.py": "b0fbeb0ba4a0d2a8a53c7b8c8045df6b2d7aa19e3b32a061dd4b993801c11f2d",
    "append_only_memory.py": "03dbb8e83f94f100d4df112e69642d6bea3af874c613b48d104a9e409ae1d1df",
    "promotion.py": "63759ee59ac18d6b7f65c82b0e26d83cd2921f309bb525a81818d87bb72ab9a0",
    "provenance_ledger.py": "cacaa44ef195288d0c6683715e7f23cc8739c36a9635cff2e9d6485d0c226a69",
}


def _sessions(turns):
    grouped = defaultdict(list)
    for turn in turns:
        grouped[str(turn["session_id"])].append(turn)
    return [{"session_id": key, "turns": value}
            for key, value in sorted(grouped.items())]


def _speaker_rows(refs):
    return [
        {"participant_ref": refs[0], "scene_role": "primary",
         "intents": ["raise a concrete issue", "seek a usable next step"]},
        {"participant_ref": refs[1], "scene_role": "supporting",
         "intents": ["test the proposal against a practical constraint"]},
    ]


def _scene(refs, proposals, *, bundle=None, suffix="one"):
    return {"goal": f"Develop a concrete extension {suffix}.",
            "speaker_participants": _speaker_rows(refs),
            "interaction_beats": ["The issue is raised.", "An assumption is tested.",
                                  "A concrete direction emerges."],
            "continuity_requirements": ["Preserve authoritative facts."],
            "extension_proposals": proposals, "proposal_bundle_id": bundle,
            "desired_outcome": "A natural independently reviewable exchange."}


def _corrected_plan(refs, action_by_operation):
    event = action_by_operation["add_event"]
    person = action_by_operation["add_person"]
    relation = action_by_operation["add_relation"]
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION_V3,
        "scene_budget": {"minimum": 2, "target": 3, "maximum": 4},
        "round_arc": {"starting_situation": "A practical issue remains open.",
                      "central_tension": "The participants prefer different approaches.",
                      "ending_direction": "They leave with a concrete direction."},
        "scene_goals": [
            _scene(refs, [{"action_id": event["action_id"], "proposed_content": {
                "description": "They schedule a short planning meeting.",
                "narrative_time_expression": "the following afternoon"}}]),
            _scene(refs, [
                {"action_id": person["action_id"], "proposed_content": {
                    "display_role": "a neutral scheduling coordinator"}},
                {"action_id": relation["action_id"], "proposed_content": {
                    "relation_type": "collaborates_with"}}], bundle="B0001", suffix="two"),
            _scene(refs, [], suffix="three"),
        ],
    }


def _markdown(title: str, value: Mapping[str, Any]) -> str:
    return f"# {title}\n\n```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```\n"


def generate_planner_contract_v3_1_reports(repo: Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    package = repo / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
    reports = package / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    input_before = sha256_file(INPUT_PATH)
    v6_before, v7_before = hash_tree(V6_ROOT), hash_tree(V7_ROOT)
    non_planner_before = {name: sha256_file(package / name)
                          for name in NON_PLANNER_EXPECTED}

    root = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    parent = read_unified_parent(INPUT_PATH)
    source = initialize_source_world_memory(root, parent)
    memory = source["memory"]
    links = compile_source_interaction_links(
        root_id=root["root_id"], sessions=_sessions(parent["dialogue"]["turns"]),
        speaker_to_entity=source["speaker_to_entity_ref"])
    menu = compile_action_menu_v3(
        authoritative_affordance_view(memory), root_id=root["root_id"],
        interaction_links=links)
    roles = [{"entity_ref": row["entity_ref"], "display_name": row["display_name"]}
             for row in memory["entities"]]
    refs = [row["entity_ref"] for row in memory["entities"]]
    view = compile_planner_action_view(menu, role_table=roles)
    bundles = compile_allowed_proposal_bundles(menu)
    view_validation = validate_planner_action_view(menu, view)
    prompt = build_planner_v3_prompt(
        root_id=root["root_id"], role_table=roles, action_menu=menu,
        context_summary={"audit_context": "no parent text needed"},
        planner_action_view=view, allowed_bundles=bundles)
    retry_prompt = build_planner_v3_retry_prompt(
        root_id=root["root_id"], frozen_action_menu=menu,
        frozen_role_table=roles, frozen_context={"old_response": "must_not_appear"},
        frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
        errors=["frozen_field_present:scene=1:action=A0001:field=participant_refs"],
        relevant_action_ids=["A0001"],
        frozen_authoritative_memory_sha256="m" * 64,
        frozen_interaction_links_sha256="i" * 64,
        frozen_round_context_sha256="c" * 64,
        frozen_planner_action_view=view, frozen_allowed_bundles=bundles)
    writable_fields = {row["action_id"]: sorted(
        row["writable_payload_schema"]["properties"])
        for row in view["actions"]}
    frozen_fields = {row["action_id"]: sorted(
        (next(value for value in menu["actions"]
              if value["action_id"] == row["action_id"]).get("frozen_payload") or {}).keys())
        for row in view["actions"]}
    action_view_checks = {
        "projection_valid": view_validation["valid"],
        "additional_properties_false": all(
            row["writable_payload_schema"]["additionalProperties"] is False
            for row in view["actions"]),
        "frozen_fields_hidden": all(
            not (set(row["writable_payload_schema"]["properties"]) & FROZEN_PLANNER_FIELDS)
            for row in view["actions"]),
        "initial_prompt_internal_schema_absent": (
            '"frozen_payload"' not in prompt and '"payload_schema":' not in prompt),
        "retry_prompt_internal_schema_absent": (
            '"frozen_payload"' not in retry_prompt and '"payload_schema":' not in retry_prompt),
        "retry_omits_old_response": "must_not_appear" not in retry_prompt,
    }
    action_view_audit = {
        "audit_version": "planner_action_view_audit_v1",
        "planner_contract_version": PLANNER_CONTRACT_VERSION,
        "planner_action_view_version": PLANNER_ACTION_VIEW_VERSION,
        "checks": action_view_checks, "ready": all(action_view_checks.values()),
        "action_view": view, "allowed_bundles": bundles,
        "writable_fields_by_action": writable_fields,
        "frozen_payload_fields_by_action": frozen_fields,
    }

    history_rows = []
    for attempt in (1, 2, 3):
        path = V7_ROOT / f"attempts/planner-v3-a{attempt:02d}.json"
        validation_path = V7_ROOT / f"planner/validation_attempt_{attempt:02d}.json"
        parsed = json.loads(path.read_text(encoding="utf-8"))["parsed_response"]
        validation = compile_validate_round_plan(
            parsed, action_menu=menu, authoritative_memory=memory,
            interaction_links=links, round_index=1,
            planner_action_view=view, allowed_bundles=bundles)
        history_rows.append({
            "attempt": attempt, "source_path": str(path),
            "source_sha256": sha256_file(path),
            "historical_validation_path": str(validation_path),
            "historical_validation_sha256": sha256_file(validation_path),
            "classification_v3_1": validation["classification"],
            "errors_v3_1": validation["errors"],
        })
    action_by_operation = {row["operation"]: row for row in menu["actions"]}
    corrected = _corrected_plan(refs, action_by_operation)
    corrected_result = compile_validate_round_plan(
        corrected, action_menu=menu, authoritative_memory=memory,
        interaction_links=links, round_index=1,
        planner_action_view=view, allowed_bundles=bundles)
    first_errors = history_rows[0]["errors_v3_1"]
    regression_checks = {
        "attempt_1_too_many": any(x.startswith("too_many_proposals") for x in first_errors),
        "attempt_1_duplicate_action": any(x.startswith("duplicate_action_id") for x in first_errors),
        "attempt_1_duplicate_round": any(x.startswith("duplicate_round_proposal") for x in first_errors),
        "attempt_1_frozen_field": any(x.startswith("frozen_field_present") for x in first_errors),
        "attempt_1_invalid_relation": any(x.startswith("invalid_canonical_enum") for x in first_errors),
        "attempt_2_participant_frozen": any("field=participant_refs" in x
                                             for x in history_rows[1]["errors_v3_1"]),
        "attempt_3_participant_frozen": any("field=participant_refs" in x
                                             for x in history_rows[2]["errors_v3_1"]),
        "corrected_fixture_valid": corrected_result["valid"],
        "corrected_bundle_compiled": bool(corrected_result.get("compiled_round_plan") and
                                           len(corrected_result["compiled_round_plan"]["scenes"][1][
                                               "compiled_proposals"]) == 2),
    }
    regression_audit = {
        "audit_version": "planner_v3_1_regression_audit_v1",
        "historical_v7": str(V7_ROOT), "historical_attempts": history_rows,
        "corrected_fixture": corrected,
        "corrected_validation": corrected_result,
        "checks": regression_checks, "ready": all(regression_checks.values()),
        "historical_v7_modified": False,
    }

    event_action = action_by_operation["add_event"]
    first_payload = deepcopy(event_action["frozen_payload"])
    first_payload.update(corrected["scene_goals"][0]["extension_proposals"][0][
        "proposed_content"])
    distinct_payload = deepcopy(first_payload)
    distinct_payload["description"] = "They schedule a distinct review meeting."
    collision_memory = deepcopy(memory)
    collision_record = {
        "memory_ref": "planner-v3-1-audit:M999999",
        "logical_key": "planner-v3-1-audit:event", "record_type": "event",
        "operation": "add_event", "payload": first_payload,
        "verification_status": "source_grounded", "authoritative_version": 1,
    }
    collision_memory["authoritative_records"].append(collision_record)
    collision_memory["active_head_index"][collision_record["logical_key"]] = \
        collision_record["memory_ref"]
    collision_result = compile_validate_round_plan(
        corrected, action_menu=menu, authoritative_memory=collision_memory,
        interaction_links=links, round_index=1,
        planner_action_view=view, allowed_bundles=bundles)
    distinct_plan = deepcopy(corrected)
    distinct_plan["scene_goals"][0]["extension_proposals"][0]["proposed_content"][
        "description"] = distinct_payload["description"]
    distinct_result = compile_validate_round_plan(
        distinct_plan, action_menu=menu, authoritative_memory=collision_memory,
        interaction_links=links, round_index=1,
        planner_action_view=view, allowed_bundles=bundles)
    novelty_checks = {
        "same_event_exact_key_equal": proposal_novelty_key(
            "add_event", first_payload) == proposal_novelty_key("add_event", deepcopy(first_payload)),
        "different_event_exact_key_differs": proposal_novelty_key(
            "add_event", first_payload) != proposal_novelty_key("add_event", distinct_payload),
        "authoritative_exact_collision_rejected": any(
            value.startswith("existing_fact_collision")
            for value in collision_result["errors"]),
        "different_event_not_fuzzy_rejected": distinct_result["valid"],
        "corrected_novelty_valid": corrected_result.get("novelty_validation", {}).get("valid") is True,
        "canonical_relation_registry_enforced": all(
            schema.get("enum") for row in view["actions"]
            for key, schema in row["writable_payload_schema"]["properties"].items()
            if key == "relation_type"),
        "no_embedding_or_fuzzy_gate": True,
    }
    novelty_audit = {
        "audit_version": "planner_proposal_novelty_audit_v1",
        "checks": novelty_checks, "ready": all(novelty_checks.values()),
        "event_key": proposal_novelty_key("add_event", first_payload),
        "distinct_event_key": proposal_novelty_key("add_event", distinct_payload),
        "authoritative_collision_validation": collision_result,
        "distinct_event_validation": distinct_result,
        "collision_rule": "deterministic exact normalized structured key only",
    }

    freeze = generate_live_v7_freeze(repo)
    api_calls = 0
    network_attempts = 0
    original_connect = socket.socket.connect
    original_urlopen = urllib.request.urlopen

    def blocked_connect(*args, **kwargs):
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("outbound_network_forbidden_in_planner_v3_1_preflight")

    def blocked_urlopen(*args, **kwargs):
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("urlopen_forbidden_in_planner_v3_1_preflight")

    def blocked_complete(*args, **kwargs):
        nonlocal api_calls
        api_calls += 1
        raise AssertionError("api_call_forbidden_in_planner_v3_1_preflight")

    manifest_path = package / "preflight_manifest_v1.json"
    with patch.object(socket.socket, "connect", blocked_connect), \
            patch.object(urllib.request, "urlopen", blocked_urlopen), \
            patch.object(OpenAICompatibleChallengerClient, "complete", blocked_complete):
        eleven = run_fixed_manifest_preflight(manifest_path)
        live_preflight = run_live_preflight_v7(
            project_root=repo, package_root=package,
            runner_path=package / "live_fresh_e2e_v7.py", input_path=INPUT_PATH,
            historical_v6_root=V6_ROOT, run_root=V8_ROOT,
            freeze_report_path=reports / "SLIM_V7_LIVE_RUNNER_FREEZE.json",
            api_config_path=API_CONFIG_PATH, tokenizer_path=TOKENIZER_PATH)
    del original_connect, original_urlopen

    import_audit = audit_v7_recursive_import_graph(
        project_root=repo, runner_path=package / "live_fresh_e2e_v7.py")
    input_after = sha256_file(INPUT_PATH)
    v6_after, v7_after = hash_tree(V6_ROOT), hash_tree(V7_ROOT)
    non_planner_after = {name: sha256_file(package / name)
                         for name in NON_PLANNER_EXPECTED}
    preflight_checks = {
        "eleven_root_ready": eleven["ready"] and eleven["root_count"] == 11,
        "live_preflight_valid": live_preflight["valid"],
        "next_run_root_absent": not V8_ROOT.exists(),
        "recursive_import_graph_clean": import_audit["valid"],
        "legacy_semantic_import_count_zero": import_audit["forbidden_import_count"] == 0,
        "api_client_call_count_zero": api_calls == 0,
        "outbound_network_attempt_count_zero": network_attempts == 0,
        "input_unchanged": input_before == input_after == INPUT_EXPECTED,
        "v6_unchanged": v6_before["tree_sha256"] == v6_after["tree_sha256"] == V6_TREE_EXPECTED,
        "v7_unchanged": v7_before["tree_sha256"] == v7_after["tree_sha256"] == V7_TREE_EXPECTED,
        "v6_runner_unchanged": sha256_file(package / "live_fresh_e2e.py") == V6_RUNNER_EXPECTED,
        "non_planner_contracts_unchanged": non_planner_before == non_planner_after == NON_PLANNER_EXPECTED,
        "implementation_manifest_valid": implementation_manifest(repo)["valid"],
        "freeze_ready": freeze["ready"],
    }
    preflight_audit = {
        "audit_version": "planner_v3_1_pre_api_preflight_v1",
        "checks": preflight_checks, "ready": all(preflight_checks.values()),
        "api_client_call_count": api_calls,
        "outbound_network_attempt_count": network_attempts,
        "eleven_root_preflight": eleven,
        "next_canary_live_preflight": live_preflight,
        "recursive_import_audit": import_audit,
        "input_sha_before": input_before, "input_sha_after": input_after,
        "v6_tree_before": v6_before, "v6_tree_after": v6_after,
        "v7_tree_before": v7_before, "v7_tree_after": v7_after,
        "non_planner_module_sha_before": non_planner_before,
        "non_planner_module_sha_after": non_planner_after,
    }

    audit_payloads = {
        "PLANNER_ACTION_VIEW_AUDIT": action_view_audit,
        "PLANNER_V3_1_REGRESSION_AUDIT": regression_audit,
        "PLANNER_PROPOSAL_NOVELTY_AUDIT": novelty_audit,
        "PLANNER_V3_1_PRE_API_PREFLIGHT": preflight_audit,
    }
    for name, payload in audit_payloads.items():
        atomic_json(reports / f"{name}.json", payload)
        atomic_text(reports / f"{name}.md", _markdown(name.replace("_", " ").title(), payload))

    readiness = {
        "planner_contract_v3_1_ready": action_view_audit["ready"] and regression_audit["ready"],
        "planner_action_projection_ready": action_view_checks["projection_valid"],
        "planner_frozen_fields_hidden_ready": action_view_checks["frozen_fields_hidden"],
        "planner_writable_schema_ready": action_view_checks["additional_properties_false"],
        "planner_bundle_selection_ready": regression_checks["corrected_bundle_compiled"],
        "planner_action_uniqueness_ready": regression_checks["attempt_1_duplicate_action"],
        "planner_existing_fact_collision_ready": novelty_checks[
            "authoritative_exact_collision_rejected"],
        "planner_round_fingerprint_dedup_ready": regression_checks["attempt_1_duplicate_round"],
        "planner_canonical_relation_ready": novelty_checks["canonical_relation_registry_enforced"],
        "planner_retry_contract_v3_1_ready": action_view_checks["retry_prompt_internal_schema_absent"],
        "planner_error_taxonomy_ready": regression_audit["ready"],
        "planner_real_failure_regression_ready": regression_audit["ready"],
        "planner_corrected_fixture_ready": regression_checks["corrected_fixture_valid"],
        "v7_runner_planner_v3_1_wiring_ready": live_preflight["valid"],
        "next_canary_preflight_ready": preflight_audit["ready"],
        "generation_ready": False, "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }
    changed_files = [
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/__init__.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/controlled_expansion.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_contract_v3_1.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_v3.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_plan_validation.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_fresh_e2e_v7.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_preflight_v7.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/offline_preflight_v3.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/v7_manifest.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/freeze_v7_live_runner.py",
        "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_contract_v3_1_reports.py",
        "tests/test_world_memory_slim_planner_v3_final_freeze.py",
        "tests/test_world_memory_slim_live_fresh_e2e_v7.py",
        "tests/test_world_memory_slim_planner_contract_v3_1.py",
    ]
    summary = {
        "report_version": REPORT_VERSION,
        "change_reason": "Planner Contract v3.1",
        "scope": "Planner-facing projection, proposal selection/novelty, retry, tests, and freeze only",
        "historical_v7_failure": {
            "failure_stage": "Planner v3 schema/compile/selection",
            "failure_message": "planner_retry_limit_exhausted",
            "root": str(V7_ROOT)},
        "planner_contract_version": PLANNER_CONTRACT_VERSION,
        "planner_action_view_version": PLANNER_ACTION_VIEW_VERSION,
        "planner_prompt_version": PLANNER_PROMPT_VERSION_V3,
        "planner_retry_prompt_version": PLANNER_RETRY_PROMPT_VERSION_V3,
        "action_view_audit_ready": action_view_audit["ready"],
        "regression_audit_ready": regression_audit["ready"],
        "novelty_audit_ready": novelty_audit["ready"],
        "pre_api_preflight_ready": preflight_audit["ready"],
        "proposal_rules": {"ordinary_scene": "0 or 1 action",
                           "bundle_scene": "exactly 2 actions from a compiled allowed bundle",
                           "action_id_per_round": 1,
                           "collision": "deterministic exact structured key"},
        "old_freeze": OLD_FREEZE,
        "new_freeze": {key: freeze[key] for key in (
            "implementation_manifest_sha256", "project_working_tree_manifest_sha256",
            "prompt_schema_contract_sha256")},
        "changed_files": changed_files,
        "targeted_unittest": freeze["targeted_unittest"],
        "full_repository_unittest": freeze["full_repository_unittest"],
        "compileall": freeze["compileall"], "git_diff_check": freeze["git_diff_check"],
        "api_client_call_count": api_calls,
        "outbound_network_attempt_count": network_attempts,
        "input_sha_before": input_before, "input_sha_after": input_after,
        "v6_tree_sha_before": v6_before["tree_sha256"],
        "v6_tree_sha_after": v6_after["tree_sha256"],
        "v7_tree_sha_before": v7_before["tree_sha256"],
        "v7_tree_sha_after": v7_after["tree_sha256"],
        "v6_runner_sha": sha256_file(package / "live_fresh_e2e.py"),
        "non_planner_module_shas_unchanged": non_planner_before == non_planner_after,
        "next_run_root": str(V8_ROOT), "next_run_root_absent": not V8_ROOT.exists(),
        "new_fresh_e2e_executed": False, "round_2_executed": False,
        "readiness": readiness,
        "unimplemented_or_deferred": ["Fresh E2E v8", "Round 2", "QA", "training"],
    }
    atomic_json(repo / "WORLD_MEMORY_PLANNER_CONTRACT_V3_1_FREEZE_REPORT.json", summary)
    atomic_text(repo / "WORLD_MEMORY_PLANNER_CONTRACT_V3_1_FREEZE_REPORT.md",
                _markdown("World Memory Planner Contract v3.1 Freeze Report", summary))

    artifact_paths = [
        repo / "WORLD_MEMORY_PLANNER_CONTRACT_V3_1_FREEZE_REPORT.json",
        repo / "WORLD_MEMORY_PLANNER_CONTRACT_V3_1_FREEZE_REPORT.md",
        reports / "SLIM_V7_LIVE_RUNNER_FREEZE.json",
        reports / "SLIM_V7_LIVE_RUNNER_FREEZE.md",
        *(reports / f"{name}.{suffix}" for name in audit_payloads for suffix in ("json", "md")),
    ]
    entries = [{"relative_path": str(path.relative_to(repo)),
                "file_sha256": sha256_file(path)} for path in sorted(artifact_paths)]
    atomic_json(reports / "ARTIFACT_MANIFEST.json", {
        "scope": "Planner Contract v3.1 generated reports; this manifest excluded",
        "reports": entries,
        "artifact_manifest_sha256": hashlib.sha256(
            canonical_json(entries).encode("utf-8")).hexdigest(),
    })
    if project_working_tree_manifest_sha256(repo) != freeze[
            "project_working_tree_manifest_sha256"]:
        raise RuntimeError("planner_v3_1_report_self_reference_or_post_freeze_drift")
    return summary


if __name__ == "__main__":
    value = generate_planner_contract_v3_1_reports(PROJECT_ROOT)
    print(json.dumps({"readiness": value["readiness"],
                      "new_freeze": value["new_freeze"],
                      "api_calls": value["api_client_call_count"],
                      "network_attempts": value["outbound_network_attempt_count"]}, indent=2))
