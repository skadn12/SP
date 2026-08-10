"""Generate Planner Bundle Selection v3.3 zero-API audits and v10 freeze."""

from __future__ import annotations

import compileall
import hashlib
import json
import re
import socket
import subprocess
import sys
import urllib.request
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from ..core import (atomic_json, atomic_text, canonical_json, sha256_bytes,
                    sha256_file)
from ..live_backends import OpenAICompatibleChallengerClient
from ..unified_dialogue_root import read_unified_parent
from .controlled_expansion import compile_action_menu_v3
from .freeze_v7_live_runner import (
    API_CONFIG_PATH, INPUT_PATH, TOKENIZER_PATH, V6_ROOT,
    _component_versions, _prompt_schema_versions,
)
from .interaction_links import compile_source_interaction_links
from .live_preflight_v7 import run_live_preflight_v7
from .live_role_runtime import load_json_object_strict
from .offline_preflight_v3 import run_fixed_manifest_preflight
from .planner_contract_v3_1 import (
    PLANNER_ACTION_VIEW_VERSION, PLANNER_BUNDLE_VERSION,
    PLANNER_CONTRACT_VERSION, compile_allowed_proposal_bundles,
    compile_planner_action_view, planner_bundle_view,
    validate_bundle_selection_registry, validate_planner_action_view,
)
from .planner_legacy_bundle_audit import classify_legacy_bundle_plan
from .planner_v3 import (
    PLANNER_PROMPT_VERSION_V3, PLANNER_RETRY_PROMPT_VERSION_V3,
    PLANNER_SCHEMA_VERSION_V3, build_planner_v3_prompt,
    build_planner_v3_retry_prompt, planner_proposal_selection_schema,
)
from .provenance import hash_tree
from .round_plan_validation import compile_validate_round_plan
from .semantic_import_audit_v7 import audit_v7_recursive_import_graph
from .source_memory_initializer_v1 import (
    authoritative_affordance_view, initialize_source_world_memory,
)
from .v7_manifest import (directory_manifest, implementation_manifest,
                          project_working_tree_manifest_sha256)


REPORT_VERSION = "planner_bundle_selection_contract_v3_3_freeze_v1"
V10_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v10")
HISTORY_ROOTS = {
    version: Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        f"conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v{version}")
    for version in (6, 7, 8, 9)
}
HISTORY_DIRECTORY_EXPECTED = {
    6: "c2dfd78b2f712d88ab1c7d5d083691ff8f86ac87e41c9c77d77eb99177a0ef41",
    7: "7c2e070ac998bb18da2de365eeca5e6d24ec99c47c3eb54a516f11e1be8be05f",
    8: "160aed338080b202f616be91d284b743b9a6ea3df94e9061da2310b32d088978",
    9: "9e9992f720ec22bfa102186d42cb519d6590c6a8d41a9aa682eb68997217abd0",
}
OLD_FREEZE = {
    "implementation_manifest_sha256":
        "fb268abcfa2e39c8ec6e55df0c3ec206681391c61ae69c2046a478f429801771",
    "project_working_tree_manifest_sha256":
        "c8f5088efe6afcd79880a1ebe24933e6f12c9714776cc29fa6b1a57cc96d9ff1",
    "prompt_schema_contract_sha256":
        "2f6e53a39162842bba65be2da7529bcbff1f41c9594060702ee59eb097b8c7d7",
}
INPUT_EXPECTED = "c8976ae6bbb2c91403466a3987d4054bad099475c8eac16c9ab86a7be7244332"
V6_RUNNER_EXPECTED = "69763e2493f88eee8e74fb4a6492013bccc52b5109c5ce5ba888839bc406ea03"
PROTECTED_EXPECTED = {
    "realizer_v3.py": "b6ae9dab749cfcb7eb39d7e0f833b9e2d87e2fcab8fd5e7cd4ecb8b16be8a463",
    "dialogue_quality.py": "6cc41bbf5392e6e6a4da7002aba185e60f932f95defd5f493bcd134d4ced9380",
    "claim_validator_v2.py": "78666101a85593368fd723999c97f60b5e5d11caca12a03e3e89f142d685d090",
    "evidence_scope.py": "e45d8e77ae1e725f4f1650b68b7ac86e2de6248dd2ee348a1b68f53da374862a",
    "transactional_binder.py": "b0fbeb0ba4a0d2a8a53c7b8c8045df6b2d7aa19e3b32a061dd4b993801c11f2d",
    "append_only_memory.py": "03dbb8e83f94f100d4df112e69642d6bea3af874c613b48d104a9e409ae1d1df",
    "promotion.py": "63759ee59ac18d6b7f65c82b0e26d83cd2921f309bb525a81818d87bb72ab9a0",
    "provenance_ledger.py": "cacaa44ef195288d0c6683715e7f23cc8739c36a9635cff2e9d6485d0c226a69",
    "interaction_links.py": "e282cc9bfa3d73c046b2439d305d83aaa94cf41eb471fe297f98ad846aa80124",
    "operation_registry.py": "dc03c765c14f691c944cd03ab842ed8082405ba32ff585b3e5190dceca7ecb6d",
    "relation_registry.py": "e0674e34188644a0bac42e77890193d08705cbfd2b0bd7aad35ea4f2b4a0cabf",
}
CHANGED_FILES = [
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_contract_v3_1.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_legacy_bundle_audit.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_role_runtime.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_plan_validation.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_fresh_e2e_v7.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/offline_preflight_v3.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/freeze_v7_live_runner.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_preflight_v7.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/v7_manifest.py",
    "scripts/current_method/restart_n5_spell/world_memory_slim_v1/planner_bundle_selection_v3_3_reports.py",
    "tests/test_world_memory_slim_planner_contract_v3_1.py",
    "tests/test_world_memory_slim_planner_v3_final_freeze.py",
    "tests/test_world_memory_slim_live_fresh_e2e_v7.py",
    "tests/test_world_memory_slim_planner_bundle_selection_v3_3.py",
]


def _sessions(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for turn in turns:
        grouped[str(turn["session_id"])].append(turn)
    return [{"session_id": key, "turns": value}
            for key, value in sorted(grouped.items())]


def _speakers(refs: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"participant_ref": refs[0], "scene_role": "primary",
         "intents": ["raise one concrete issue", "seek a usable next step"]},
        {"participant_ref": refs[1], "scene_role": "supporting",
         "intents": ["test one practical constraint"]},
    ]


def _scene(refs: Sequence[str], selection: Mapping[str, Any], suffix: str) -> dict[str, Any]:
    return {
        "goal": f"Develop extension {suffix}.",
        "speaker_participants": _speakers(refs),
        "interaction_beats": ["Raise the issue.", "Test one assumption.",
                              "Reach a concrete outcome."],
        "continuity_requirements": ["Preserve authoritative source facts."],
        "proposal_selection": deepcopy(dict(selection)),
        "desired_outcome": "A natural independently reviewable exchange.",
    }


def _corrected_plan(refs: Sequence[str], actions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    person, relation, event = (
        actions["add_person"], actions["add_relation"], actions["add_event"])
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION_V3,
        "scene_budget": {"minimum": 2, "target": 3, "maximum": 4},
        "round_arc": {
            "starting_situation": "A practical issue remains open.",
            "central_tension": "The participants prefer different approaches.",
            "ending_direction": "They leave with a concrete direction.",
        },
        "scene_goals": [
            _scene(refs, {
                "selection_type": "bundle", "bundle_id": "B0001",
                "proposal_contents": {
                    person["action_id"]: {
                        "display_role": "a neutral scheduling coordinator"},
                    relation["action_id"]: {"relation_type": "collaborates_with"},
                },
            }, "one"),
            _scene(refs, {
                "selection_type": "single_action", "action_id": event["action_id"],
                "proposed_content": {
                    "description": "They schedule a short planning meeting.",
                    "narrative_time_expression": "the following afternoon",
                },
            }, "two"),
            _scene(refs, {"selection_type": "none"}, "three"),
        ],
    }


def _command(repo: Path, *arguments: str) -> dict[str, Any]:
    result = subprocess.run(list(arguments), cwd=repo, text=True,
                            capture_output=True)
    combined = (result.stdout + result.stderr).strip()
    match = re.search(r"Ran (\d+) tests?", combined)
    return {
        "command": list(arguments), "returncode": result.returncode,
        "passed": result.returncode == 0,
        "tests_run": int(match.group(1)) if match else None,
        "output_tail": "\n".join(combined.splitlines()[-16:]),
    }


def _raises_duplicate(raw: str, taxonomy: str) -> bool:
    try:
        load_json_object_strict(raw)
    except ValueError as error:
        return taxonomy in str(error)
    return False


def _write_pair(path: Path, title: str, value: Mapping[str, Any]) -> None:
    atomic_json(path.with_suffix(".json"), dict(value))
    atomic_text(path.with_suffix(".md"),
                f"# {title}\n\n```json\n" +
                json.dumps(value, ensure_ascii=False, indent=2) + "\n```\n")


def generate_planner_bundle_selection_v3_3_reports(repo: Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    package = repo / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
    reports = package / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    input_before = sha256_file(INPUT_PATH)
    history_before = {version: directory_manifest(path)
                      for version, path in HISTORY_ROOTS.items()}
    protected_before = {name: sha256_file(package / name)
                        for name in PROTECTED_EXPECTED}

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
    roles = [{"entity_ref": row["entity_ref"],
              "display_name": row["display_name"]}
             for row in memory["entities"]]
    refs = [row["entity_ref"] for row in memory["entities"]]
    action_view = compile_planner_action_view(menu, role_table=roles)
    bundles = compile_allowed_proposal_bundles(menu)
    actions = {row["operation"]: row for row in menu["actions"]}
    corrected = _corrected_plan(refs, actions)
    corrected_result = compile_validate_round_plan(
        corrected, action_menu=menu, authoritative_memory=memory,
        interaction_links=links, round_index=1,
        planner_action_view=action_view, allowed_bundles=bundles)

    none_extra = deepcopy(corrected)
    none_extra["scene_goals"][2]["proposal_selection"]["action_id"] = "A0001"
    single_extra = deepcopy(corrected)
    single_extra["scene_goals"][1]["proposal_selection"]["bundle_id"] = "B0001"
    bundle_missing = deepcopy(corrected)
    required = list(bundles["bundles"][0]["required_action_ids"])
    del bundle_missing["scene_goals"][0]["proposal_selection"]["proposal_contents"][required[0]]
    bundle_extra = deepcopy(corrected)
    bundle_extra["scene_goals"][0]["proposal_selection"]["proposal_contents"]["A9999"] = {}
    split = deepcopy(corrected)
    contents = split["scene_goals"][0]["proposal_selection"]["proposal_contents"]
    split["scene_goals"][0]["proposal_selection"] = {
        "selection_type": "single_action", "action_id": required[0],
        "proposed_content": contents[required[0]],
    }
    split["scene_goals"][1]["proposal_selection"] = {
        "selection_type": "single_action", "action_id": required[1],
        "proposed_content": contents[required[1]],
    }
    none_result = compile_validate_round_plan(
        none_extra, action_menu=menu, authoritative_memory=memory,
        interaction_links=links, round_index=1,
        planner_action_view=action_view, allowed_bundles=bundles)
    single_result = compile_validate_round_plan(
        single_extra, action_menu=menu, authoritative_memory=memory,
        interaction_links=links, round_index=1,
        planner_action_view=action_view, allowed_bundles=bundles)
    missing_result = compile_validate_round_plan(
        bundle_missing, action_menu=menu, authoritative_memory=memory,
        interaction_links=links, round_index=1,
        planner_action_view=action_view, allowed_bundles=bundles)
    extra_result = compile_validate_round_plan(
        bundle_extra, action_menu=menu, authoritative_memory=memory,
        interaction_links=links, round_index=1,
        planner_action_view=action_view, allowed_bundles=bundles)
    split_result = compile_validate_round_plan(
        split, action_menu=menu, authoritative_memory=memory,
        interaction_links=links, round_index=1,
        planner_action_view=action_view, allowed_bundles=bundles)

    prompt = build_planner_v3_prompt(
        root_id=root["root_id"], role_table=roles, action_menu=menu,
        context_summary={"audit": "compact no-parent-text context"},
        planner_action_view=action_view, allowed_bundles=bundles)
    retry = build_planner_v3_retry_prompt(
        root_id=root["root_id"], frozen_action_menu=menu,
        frozen_role_table=roles, frozen_context={"old_response": "omitted"},
        frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
        errors=["split_bundle_across_scenes:bundle=B0001"],
        relevant_action_ids=[required[0]],
        frozen_authoritative_memory_sha256="m" * 64,
        frozen_interaction_links_sha256="i" * 64,
        frozen_round_context_sha256="c" * 64,
        frozen_planner_action_view=action_view,
        frozen_allowed_bundles=bundles)

    oneof = planner_proposal_selection_schema()
    projection_validation = validate_planner_action_view(menu, action_view)
    bundle_registry_validation = validate_bundle_selection_registry(
        menu, bundles, action_view)
    schema_checks = {
        "contract_version_v3_3": PLANNER_CONTRACT_VERSION ==
            "slim_world_planner_contract_v3_3",
        "bundle_selection_version_v1": PLANNER_BUNDLE_VERSION ==
            "planner_bundle_selection_v1",
        "prompt_version_v4": PLANNER_PROMPT_VERSION_V3 ==
            "slim_world_planner_v3_prompt_v4",
        "retry_prompt_version_v4": PLANNER_RETRY_PROMPT_VERSION_V3 ==
            "slim_world_planner_v3_retry_prompt_v4",
        "oneof_has_three_branches": len(oneof.get("oneOf") or []) == 3,
        "oneof_additional_properties_false": all(
            row.get("additionalProperties") is False for row in oneof.get("oneOf") or []),
        "action_projection_valid": projection_validation["valid"],
        "bundle_registry_valid": bundle_registry_validation["valid"],
        "none_valid": corrected_result["valid"] and
            corrected["scene_goals"][2]["proposal_selection"] == {
                "selection_type": "none"},
        "none_extra_rejected": any(
            value.startswith("invalid_proposal_selection_shape")
            for value in none_result["errors"]),
        "single_valid": corrected_result["valid"],
        "single_bundle_field_rejected": any(
            value.startswith("single_action_has_bundle_fields")
            for value in single_result["errors"]),
        "bundle_valid": corrected_result["valid"],
        "bundle_missing_rejected": any(
            value.startswith("incomplete_bundle_actions")
            for value in missing_result["errors"]),
        "bundle_extra_rejected": any(
            value.startswith("extra_bundle_actions")
            for value in extra_result["errors"]),
        "bundle_only_split_rejected": any(
            value.startswith("split_bundle_across_scenes")
            for value in split_result["errors"]),
        "duplicate_top_key_detected": _raises_duplicate(
            '{"scene_goals":[],"scene_goals":[]}', "duplicate_json_object_key"),
        "duplicate_bundle_action_detected": _raises_duplicate(
            '{"proposal_contents":{"A0007":{},"A0007":{},"A0003":{}}}',
            "duplicate_bundle_action_key"),
        "initial_prompt_old_fields_absent": (
            "extension_proposals" not in prompt and "proposal_bundle_id" not in prompt),
        "retry_prompt_old_fields_absent": (
            "extension_proposals" not in retry and "proposal_bundle_id" not in retry),
        "retry_keeps_complete_action_view": all(
            row["action_id"] in retry for row in action_view["actions"]),
        "retry_keeps_complete_bundle_view": all(
            row["bundle_id"] in retry for row in bundles["bundles"]),
        "prompt_bundle_indivisible": "indivisible" in prompt and "bundle_only" in prompt,
    }
    schema_audit = {
        "audit_version": "planner_bundle_selection_schema_audit_v1",
        "planner_contract_version": PLANNER_CONTRACT_VERSION,
        "planner_bundle_selection_version": PLANNER_BUNDLE_VERSION,
        "planner_prompt_version": PLANNER_PROMPT_VERSION_V3,
        "planner_retry_prompt_version": PLANNER_RETRY_PROMPT_VERSION_V3,
        "proposal_selection_schema": oneof,
        "planner_action_view": action_view,
        "planner_bundle_view": planner_bundle_view(bundles, action_view),
        "checks": schema_checks, "ready": all(schema_checks.values()),
        "error_fixtures": {
            "none_extra": none_result["errors"],
            "single_extra": single_result["errors"],
            "bundle_missing": missing_result["errors"],
            "bundle_extra": extra_result["errors"],
            "bundle_split": split_result["errors"],
        },
    }

    v9_rows = []
    for attempt in (1, 2, 3):
        path = HISTORY_ROOTS[9] / f"attempts/planner-v3-a{attempt:02d}.json"
        validation_path = HISTORY_ROOTS[9] / f"planner/validation_attempt_{attempt:02d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        audit = classify_legacy_bundle_plan(
            record["parsed_response"], allowed_bundles=bundles,
            planner_action_view=action_view)
        v9_rows.append({
            "attempt": attempt, "source_path": str(path),
            "source_sha256": sha256_file(path),
            "raw_response_sha256": record.get("raw_response_sha256"),
            "historical_validation_path": str(validation_path),
            "historical_validation_sha256": sha256_file(validation_path),
            "historical_validation_result": json.loads(
                validation_path.read_text(encoding="utf-8")),
            "v3_3_legacy_audit": audit,
        })
    v9_checks = {
        "attempt_1_unregistered_pair": any(
            value.startswith("invalid_unregistered_action_pair")
            for value in v9_rows[0]["v3_3_legacy_audit"]["errors"]),
        "attempt_2_split_bundle": all(any(
            value.startswith(prefix)
            for value in v9_rows[1]["v3_3_legacy_audit"]["errors"])
            for prefix in ("single_action_illegally_carries_bundle",
                           "incomplete_bundle_actions", "split_bundle_across_scenes",
                           "bundle_only_action_selected_singly")),
        "attempt_3_split_bundle": all(any(
            value.startswith(prefix)
            for value in v9_rows[2]["v3_3_legacy_audit"]["errors"])
            for prefix in ("single_action_illegally_carries_bundle",
                           "incomplete_bundle_actions", "split_bundle_across_scenes",
                           "bundle_only_action_selected_singly")),
        "no_legacy_migration": all(
            not row["v3_3_legacy_audit"]["migration_performed"] for row in v9_rows),
        "historical_v9_still_failed": all(
            not row["historical_validation_result"].get("valid") for row in v9_rows),
    }
    v9_audit = {
        "audit_version": "planner_v9_bundle_regression_audit_v1",
        "historical_v9_root": str(HISTORY_ROOTS[9]),
        "diagnosis": (
            "v9 allowed model-controlled action arrays and bundle labels. Attempt 1 "
            "formed an unregistered pair; attempts 2 and 3 split atomic B0001."),
        "attempts": v9_rows, "checks": v9_checks,
        "ready": all(v9_checks.values()), "historical_write_performed": False,
    }

    compiled = corrected_result.get("compiled_round_plan") or {}
    first_scene = (compiled.get("scenes") or [{}])[0]
    proposals = first_scene.get("compiled_proposals") or []
    internal_keys = {
        "proposal_ref", "action_id", "operation", "operation_class",
        "compiled_payload", "connected_memory_refs", "target_memory_ref",
        "pre_state", "authoritative_version", "proposal_fingerprint",
        "temporary_refs", "proposal_dependencies", "atomic_group_ref",
        "proposal_bundle_id", "lifecycle", "provenance",
    }
    expansion_checks = {
        "corrected_fixture_valid": corrected_result["valid"],
        "bundle_expands_two": len(proposals) == 2,
        "required_order_preserved": [row.get("action_id") for row in proposals] == required,
        "stable_proposal_refs": [row.get("proposal_ref") for row in proposals] ==
            ["R01:S01:P01", "R01:S01:P02"],
        "same_scene": all(str(row.get("proposal_ref")).startswith("R01:S01:")
                          for row in proposals),
        "dependency_compiled": len(proposals) == 2 and
            proposals[1].get("proposal_dependencies") == ["R01:S01:P01"],
        "atomic_group_compiled": len(proposals) == 2 and
            bool(proposals[0].get("atomic_group_ref")) and
            proposals[0].get("atomic_group_ref") == proposals[1].get("atomic_group_ref"),
        "temporary_binding_stable": len(proposals) == 2 and
            proposals[0].get("temporary_refs", [None])[0] ==
            proposals[1].get("compiled_payload", {}).get("target_ref"),
        "internal_contract_keys_unchanged": bool(proposals) and
            all(set(row) == internal_keys for row in proposals),
        "planner_fields_not_in_compiled_proposal": all(
            not ({"proposal_selection", "selection_type", "bundle_id",
                  "proposal_contents"} & set(row)) for row in proposals),
    }
    expansion_audit = {
        "audit_version": "planner_bundle_expansion_audit_v1",
        "corrected_fixture": corrected,
        "validation": corrected_result,
        "checks": expansion_checks, "ready": all(expansion_checks.values()),
        "compiled_bundle_proposals": proposals,
        "internal_contract_scope": "Realizer-facing compiled proposal list unchanged",
    }

    targeted = _command(
        repo, sys.executable, "-m", "unittest",
        "tests.test_world_memory_slim_planner_bundle_selection_v3_3",
        "tests.test_world_memory_slim_planner_contract_v3_1",
        "tests.test_world_memory_slim_planner_v3_final_freeze",
        "tests.test_world_memory_slim_live_fresh_e2e_v7")
    full_repository = _command(
        repo, sys.executable, "-m", "unittest", "discover", "-s", "tests")
    compile_passed = (compileall.compile_dir(str(repo / "scripts"), quiet=1) and
                      compileall.compile_dir(str(repo / "tests"), quiet=1))
    diff_check = _command(repo, "git", "diff", "--check")
    import_audit = audit_v7_recursive_import_graph(
        project_root=repo, runner_path=package / "live_fresh_e2e_v7.py")
    protected_after_tests = {name: sha256_file(package / name)
                             for name in PROTECTED_EXPECTED}

    api_calls = 0
    network_attempts = 0

    def blocked_connect(*args, **kwargs):
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("outbound_network_forbidden_v3_3")

    def blocked_urlopen(*args, **kwargs):
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("urlopen_forbidden_v3_3")

    def blocked_complete(*args, **kwargs):
        nonlocal api_calls
        api_calls += 1
        raise AssertionError("api_call_forbidden_v3_3")

    manifest_path = package / "preflight_manifest_v1.json"
    with patch.object(socket.socket, "connect", blocked_connect), \
            patch.object(urllib.request, "urlopen", blocked_urlopen), \
            patch.object(OpenAICompatibleChallengerClient, "complete", blocked_complete):
        eleven = run_fixed_manifest_preflight(manifest_path)

    implementation = implementation_manifest(repo)
    working_manifest = project_working_tree_manifest_sha256(repo)
    component_versions = _component_versions()
    prompt_schema_versions = _prompt_schema_versions()
    prompt_schema_sha = hashlib.sha256(json.dumps(
        prompt_schema_versions, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    git_head = _command(repo, "git", "rev-parse", "HEAD")["output_tail"]
    parent_dialogue_sha = sha256_bytes(canonical_json(
        [parent["dialogue"]]).encode("utf-8"))

    common = bool(
        schema_audit["ready"] and v9_audit["ready"] and expansion_audit["ready"] and
        targeted["passed"] and full_repository["passed"] and compile_passed and
        diff_check["passed"] and eleven.get("ready") and eleven.get("root_count") == 11 and
        import_audit.get("valid") and import_audit.get("forbidden_import_count") == 0 and
        protected_before == protected_after_tests == PROTECTED_EXPECTED and
        implementation["valid"] and api_calls == 0 and network_attempts == 0 and
        not V10_ROOT.exists())
    readiness = {
        "planner_bundle_selection_v3_3_ready": common,
        "planner_proposal_selection_oneof_ready": schema_checks[
            "oneof_has_three_branches"] and schema_checks[
                "oneof_additional_properties_false"],
        "planner_duplicate_json_key_detection_ready": schema_checks[
            "duplicate_top_key_detected"] and schema_checks[
                "duplicate_bundle_action_detected"],
        "planner_none_selection_ready": schema_checks["none_valid"] and
            schema_checks["none_extra_rejected"],
        "planner_single_action_selection_ready": schema_checks["single_valid"] and
            schema_checks["single_bundle_field_rejected"],
        "planner_atomic_bundle_selection_ready": schema_checks["bundle_valid"] and
            expansion_checks["atomic_group_compiled"],
        "planner_bundle_complete_membership_ready": schema_checks[
            "bundle_missing_rejected"] and schema_checks["bundle_extra_rejected"],
        "planner_bundle_only_action_policy_ready": schema_checks[
            "bundle_only_split_rejected"],
        "planner_bundle_cross_scene_isolation_ready": schema_checks[
            "bundle_only_split_rejected"],
        "planner_bundle_action_uniqueness_ready": schema_audit["ready"],
        "planner_unregistered_pair_rejection_ready": v9_checks[
            "attempt_1_unregistered_pair"],
        "planner_bundle_local_expansion_ready": expansion_audit["ready"],
        "planner_internal_compiled_contract_preserved_ready": expansion_checks[
            "internal_contract_keys_unchanged"] and protected_before == PROTECTED_EXPECTED,
        "planner_bundle_retry_contract_ready": schema_checks[
            "retry_prompt_old_fields_absent"] and schema_checks[
                "retry_keeps_complete_action_view"] and schema_checks[
                    "retry_keeps_complete_bundle_view"],
        "planner_v9_bundle_regression_ready": v9_audit["ready"],
        "planner_bundle_corrected_fixture_ready": corrected_result["valid"],
        "v7_runner_bundle_v3_3_wiring_ready": import_audit["valid"] and
            expansion_checks["planner_fields_not_in_compiled_proposal"],
        "provenance_aware_source_initializer_ready": common,
        "interaction_link_compiler_ready": common,
        "delta_affordance_compiler_v3_ready": common,
        "round_planner_context_compiler_ready": common,
        "slim_world_planner_v3_ready": common,
        "local_scene_plan_compiler_ready": common,
        "proposal_fingerprint_dependency_ready": common,
        "natural_dialogue_turn_budget_v2_ready": common,
        "natural_realizer_v3_ready": common,
        "dialogue_quality_v1_ready": common,
        "claim_validator_v2_ready": common,
        "evidence_scope_audit_v1_ready": common,
        "transactional_binder_v1_ready": common,
        "append_only_world_memory_v1_ready": common,
        "promotion_gate_v1_ready": common,
        "provenance_ledger_ready": common,
        "offline_11_root_preflight_ready": eleven.get("ready") is True and
            eleven.get("root_count") == 11,
        "recursive_import_graph_ready": import_audit["valid"],
        "legacy_semantic_import_zero_ready": import_audit[
            "forbidden_import_count"] == 0,
        "offline_tests_ready": targeted["passed"] and full_repository["passed"],
        "compileall_ready": bool(compile_passed),
        "git_diff_check_ready": diff_check["passed"],
        "api_network_zero_ready": api_calls == 0 and network_attempts == 0,
        "historical_runs_read_only_ready": all(
            history_before[version]["directory_manifest_sha256"] ==
            HISTORY_DIRECTORY_EXPECTED[version] for version in HISTORY_ROOTS),
        "authoritative_input_unchanged_ready": input_before == INPUT_EXPECTED,
        "next_run_root_absent_ready": not V10_ROOT.exists(),
        "next_canary_preflight_ready": common,
        "generation_ready": False,
        "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }
    freeze = {
        "freeze_version": "slim_v10_planner_bundle_selection_v3_3_freeze_v1",
        "scope": "Planner Bundle Selection Contract v3.3; zero API; v10 not executed",
        "git_head": git_head,
        "implementation_manifest_sha256": implementation[
            "implementation_manifest_sha256"],
        "implementation_manifest": implementation,
        "project_working_tree_manifest_sha256": working_manifest,
        "prompt_schema_contract_sha256": prompt_schema_sha,
        "prompt_schema_versions": prompt_schema_versions,
        "component_versions": component_versions,
        "input_path": str(INPUT_PATH), "input_sha256": input_before,
        "parent_dialogue_sha256": parent_dialogue_sha,
        "v6_root": str(V6_ROOT),
        "v6_directory_manifest_sha256": history_before[6][
            "directory_manifest_sha256"],
        "v6_runner_sha256": sha256_file(package / "live_fresh_e2e.py"),
        "historical_roots": {
            str(version): {
                "path": str(HISTORY_ROOTS[version]),
                "directory_manifest_sha256": history_before[version][
                    "directory_manifest_sha256"],
            } for version in sorted(HISTORY_ROOTS)
        },
        "next_run_root": str(V10_ROOT),
        "next_run_root_absent": not V10_ROOT.exists(),
        "api_config_path": str(API_CONFIG_PATH),
        "api_config_sha256": sha256_file(API_CONFIG_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "offline_11_root_preflight": {
            "ready": eleven.get("ready"), "root_count": eleven.get("root_count")},
        "recursive_import_audit": import_audit,
        "targeted_unittest": targeted,
        "full_repository_unittest": full_repository,
        "compileall": {"passed": bool(compile_passed), "scope": "scripts tests"},
        "git_diff_check": diff_check,
        "readiness": readiness,
        "readiness_count": sum(key.endswith("_ready") and value is True
                               for key, value in readiness.items()),
        "api_calls": api_calls, "outbound_network_attempts": network_attempts,
        "new_fresh_e2e_executed": False, "round_2_executed": False,
        "generation_ready": False, "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }
    freeze["ready"] = bool(
        freeze["next_run_root_absent"] and
        all(value is True for key, value in readiness.items()
            if key.endswith("_ready") and key not in {
                "generation_ready", "fresh_e2e_ready", "qa_pipeline_ready"}))
    freeze_path = reports / "SLIM_V10_CANARY_FREEZE"
    _write_pair(freeze_path, "Slim v10 Canary Freeze", freeze)

    with patch.object(socket.socket, "connect", blocked_connect), \
            patch.object(urllib.request, "urlopen", blocked_urlopen), \
            patch.object(OpenAICompatibleChallengerClient, "complete", blocked_complete):
        hard_preflight = run_live_preflight_v7(
            project_root=repo, package_root=package,
            runner_path=package / "live_fresh_e2e_v7.py", input_path=INPUT_PATH,
            historical_v6_root=V6_ROOT, run_root=V10_ROOT,
            freeze_report_path=freeze_path.with_suffix(".json"),
            api_config_path=API_CONFIG_PATH, tokenizer_path=TOKENIZER_PATH)
    readiness["next_canary_preflight_ready"] = bool(
        hard_preflight["valid"] and api_calls == 0 and network_attempts == 0 and
        not V10_ROOT.exists())
    freeze["readiness"] = readiness
    freeze["readiness_count"] = sum(key.endswith("_ready") and value is True
                                    for key, value in readiness.items())
    freeze["ready"] = bool(
        readiness["next_canary_preflight_ready"] and
        all(value is True for key, value in readiness.items()
            if key.endswith("_ready") and key not in {
                "generation_ready", "fresh_e2e_ready", "qa_pipeline_ready"}))
    freeze["hard_preflight"] = {
        "valid": hard_preflight["valid"], "checks": hard_preflight["checks"],
        "errors": hard_preflight["errors"],
    }
    freeze["api_calls"] = api_calls
    freeze["outbound_network_attempts"] = network_attempts
    _write_pair(freeze_path, "Slim v10 Canary Freeze", freeze)

    input_after = sha256_file(INPUT_PATH)
    history_after = {version: directory_manifest(path)
                     for version, path in HISTORY_ROOTS.items()}
    protected_after = {name: sha256_file(package / name)
                       for name in PROTECTED_EXPECTED}
    integrity_checks = {
        "input_unchanged": input_before == input_after == INPUT_EXPECTED,
        "history_v6_unchanged": history_before[6]["directory_manifest_sha256"] ==
            history_after[6]["directory_manifest_sha256"] == HISTORY_DIRECTORY_EXPECTED[6],
        "history_v7_unchanged": history_before[7]["directory_manifest_sha256"] ==
            history_after[7]["directory_manifest_sha256"] == HISTORY_DIRECTORY_EXPECTED[7],
        "history_v8_unchanged": history_before[8]["directory_manifest_sha256"] ==
            history_after[8]["directory_manifest_sha256"] == HISTORY_DIRECTORY_EXPECTED[8],
        "history_v9_unchanged": history_before[9]["directory_manifest_sha256"] ==
            history_after[9]["directory_manifest_sha256"] == HISTORY_DIRECTORY_EXPECTED[9],
        "v6_runner_unchanged": sha256_file(package / "live_fresh_e2e.py") ==
            V6_RUNNER_EXPECTED,
        "protected_contracts_unchanged": protected_before == protected_after ==
            PROTECTED_EXPECTED,
        "v10_run_root_absent": not V10_ROOT.exists(),
        "api_calls_zero": api_calls == 0,
        "network_attempts_zero": network_attempts == 0,
        "implementation_manifest_stable": implementation_manifest(repo)[
            "implementation_manifest_sha256"] == freeze[
                "implementation_manifest_sha256"],
        "working_tree_manifest_stable": project_working_tree_manifest_sha256(repo) ==
            freeze["project_working_tree_manifest_sha256"],
    }
    preflight_audit = {
        "audit_version": "planner_v3_3_pre_api_preflight_v1",
        "checks": {
            **integrity_checks,
            "targeted_unittest_passed": targeted["passed"],
            "full_repository_unittest_passed": full_repository["passed"],
            "compileall_passed": bool(compile_passed),
            "git_diff_check_passed": diff_check["passed"],
            "offline_11_root_preflight_11_of_11": eleven.get("ready") is True and
                eleven.get("root_count") == 11,
            "hard_preflight_valid": hard_preflight["valid"],
            "recursive_import_graph_clean": import_audit["valid"],
            "legacy_semantic_import_count_zero": import_audit[
                "forbidden_import_count"] == 0,
        },
        "hard_preflight": hard_preflight,
        "offline_11_root_preflight": eleven,
        "targeted_unittest": targeted,
        "full_repository_unittest": full_repository,
        "compileall": {"passed": bool(compile_passed)},
        "git_diff_check": diff_check,
        "recursive_import_audit": import_audit,
        "api_client_call_count": api_calls,
        "outbound_network_attempt_count": network_attempts,
        "next_run_root": str(V10_ROOT),
        "next_run_root_absent": not V10_ROOT.exists(),
    }
    preflight_audit["ready"] = all(preflight_audit["checks"].values())

    _write_pair(reports / "PLANNER_BUNDLE_SELECTION_SCHEMA_AUDIT",
                "Planner Bundle Selection Schema Audit", schema_audit)
    _write_pair(reports / "PLANNER_V9_BUNDLE_REGRESSION_AUDIT",
                "Planner v9 Bundle Regression Audit", v9_audit)
    _write_pair(reports / "PLANNER_BUNDLE_EXPANSION_AUDIT",
                "Planner Bundle Expansion Audit", expansion_audit)
    _write_pair(reports / "PLANNER_V3_3_PRE_API_PREFLIGHT",
                "Planner v3.3 Pre-API Preflight", preflight_audit)

    summary = {
        "report_version": REPORT_VERSION,
        "scope": "Planner Bundle Selection Contract v3.3 zero-API repair and refreeze",
        "historical_failure_diagnosis": v9_audit["diagnosis"],
        "old_representation_problem": (
            "action_id arrays and proposal_bundle_id were independently model-controlled, "
            "so an unregistered pair or a split atomic bundle remained expressible."),
        "new_selection_forms": ["none", "single_action", "bundle"],
        "json_schema_oneof": oneof,
        "duplicate_json_key_detection": {
            "duplicate_json_object_key": schema_checks["duplicate_top_key_detected"],
            "duplicate_bundle_action_key": schema_checks[
                "duplicate_bundle_action_detected"],
        },
        "bundle_only_policy": {
            row["action_id"]: {
                "selection_policy": row["selection_policy"],
                "required_bundle_id": row["required_bundle_id"],
            } for row in action_view["actions"]
            if row["selection_policy"] == "bundle_only"
        },
        "bundle_registry": bundles,
        "local_bundle_expansion": expansion_audit,
        "internal_compiled_contract_preserved": expansion_audit["ready"],
        "error_classification": {
            "planner_repairable": [
                "invalid_proposal_selection_shape", "invalid_selection_type",
                "single_action_has_bundle_fields", "bundle_has_single_action_fields",
                "unknown_bundle_id", "incomplete_bundle_actions",
                "extra_bundle_actions", "duplicate_bundle_action_key",
                "duplicate_json_object_key", "invalid_unregistered_action_pair",
                "split_bundle_across_scenes", "duplicate_bundle_id",
                "bundle_action_reused_elsewhere",
                "single_action_illegally_carries_bundle",
                "bundle_only_action_selected_singly", "required_bundle_not_selected",
                "bundle_payload_schema_error",
            ],
            "local_invariant_failure": [
                "bundle_definition_required_actions_missing",
                "bundle_definition_unknown_action",
                "bundle_action_writable_schema_missing",
                "bundle_frozen_writable_conflict",
                "bundle_dependency_metadata_invalid",
                "bundle_only_required_bundle_missing",
                "bundle_definition_action_in_multiple_bundles",
            ],
        },
        "historical_v9_regression": v9_audit,
        "corrected_fixture_validation": corrected_result,
        "runner_wiring": {
            "duplicate_key_safe_parser": "live_role_runtime.load_json_object_strict",
            "schema_compile_semantic_validator": "compile_validate_round_plan",
            "compiled_proposal_contract_preserved": expansion_audit["ready"],
            "recursive_import_graph": import_audit,
        },
        "tests": {"targeted": targeted, "full_repository": full_repository,
                  "compileall": bool(compile_passed), "git_diff_check": diff_check},
        "offline_11_root_preflight": {
            "ready": eleven.get("ready"), "root_count": eleven.get("root_count")},
        "api_client_call_count": api_calls,
        "outbound_network_attempt_count": network_attempts,
        "old_freeze": OLD_FREEZE,
        "new_freeze": {
            "implementation_manifest_sha256": freeze[
                "implementation_manifest_sha256"],
            "project_working_tree_manifest_sha256": freeze[
                "project_working_tree_manifest_sha256"],
            "prompt_schema_contract_sha256": freeze[
                "prompt_schema_contract_sha256"],
        },
        "changed_files": CHANGED_FILES,
        "change_reason": "Planner Bundle Selection Contract v3.3",
        "git_head": git_head,
        "input_sha_before": input_before, "input_sha_after": input_after,
        "historical_directory_sha_before": {
            str(version): history_before[version]["directory_manifest_sha256"]
            for version in sorted(history_before)},
        "historical_directory_sha_after": {
            str(version): history_after[version]["directory_manifest_sha256"]
            for version in sorted(history_after)},
        "integrity_checks": integrity_checks,
        "readiness": readiness,
        "ready": bool(schema_audit["ready"] and v9_audit["ready"] and
                      expansion_audit["ready"] and preflight_audit["ready"] and
                      freeze["ready"]),
        "generation_ready": False,
        "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
        "unimplemented_or_deferred": [
            "Fresh E2E v10 live execution", "Round 2", "QA", "Training"],
    }
    root_report = repo / "WORLD_MEMORY_PLANNER_BUNDLE_SELECTION_V3_3_FREEZE_REPORT"
    _write_pair(root_report, "World Memory Planner Bundle Selection v3.3 Freeze Report",
                summary)

    output_paths = [
        root_report.with_suffix(".json"), root_report.with_suffix(".md"),
        reports / "PLANNER_BUNDLE_SELECTION_SCHEMA_AUDIT.json",
        reports / "PLANNER_BUNDLE_SELECTION_SCHEMA_AUDIT.md",
        reports / "PLANNER_V9_BUNDLE_REGRESSION_AUDIT.json",
        reports / "PLANNER_V9_BUNDLE_REGRESSION_AUDIT.md",
        reports / "PLANNER_BUNDLE_EXPANSION_AUDIT.json",
        reports / "PLANNER_BUNDLE_EXPANSION_AUDIT.md",
        reports / "PLANNER_V3_3_PRE_API_PREFLIGHT.json",
        reports / "PLANNER_V3_3_PRE_API_PREFLIGHT.md",
        freeze_path.with_suffix(".json"), freeze_path.with_suffix(".md"),
    ]
    artifact_manifest = {
        "artifact_manifest_version": "planner_bundle_selection_v3_3_artifacts_v1",
        "artifacts": [{"path": str(path), "sha256": sha256_file(path)}
                      for path in output_paths],
        "formal_v10_run_root_created": V10_ROOT.exists(),
        "api_calls": api_calls, "outbound_network_attempts": network_attempts,
    }
    atomic_json(reports / "PLANNER_BUNDLE_SELECTION_V3_3_ARTIFACT_MANIFEST.json",
                artifact_manifest)
    return summary


if __name__ == "__main__":
    result = generate_planner_bundle_selection_v3_3_reports(
        Path(__file__).resolve().parents[4])
    print(json.dumps({
        "ready": result["ready"],
        "api_calls": result["api_client_call_count"],
        "network_attempts": result["outbound_network_attempt_count"],
        "new_freeze": result["new_freeze"],
    }, ensure_ascii=False, indent=2))
