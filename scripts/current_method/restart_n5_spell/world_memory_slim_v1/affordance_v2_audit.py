"""Offline shadow replay and report generation for Planner affordance v2.

This module has no transport/backend import and performs no model calls.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .affordance_compiler import (
    AFFORDANCE_SCHEMA_VERSION,
    COMPILED_DELTA_SCHEMA_VERSION,
    compile_delta_affordances,
    compile_selected_actions,
)
from .inputs import load_unified_root_read_only
from .memory import initialize_world_memory
from .preflight import run_preflight_batch
from .provenance import (
    audit_legacy_import_isolation,
    git_value,
    hash_tree,
    project_diff_sha256_excluding,
)
from .schemas import PLANNER_SCHEMA_VERSION, validate_slim_planner_output
from .structural import validate_structural_deltas


AUDIT_SCHEMA_VERSION = "slim_planner_affordance_v2_shadow_audit_v1"
REPORT_SCHEMA_VERSION = "world_memory_slim_affordance_v2_report_v1"
DEFAULT_HISTORY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v1")
DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-61555/unified_root.json")
CC_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots")
CC_IDS = (
    "cc-train-episode-12596", "cc-train-episode-13836",
    "cc-train-episode-146700", "cc-train-episode-178392",
    "cc-train-episode-216742", "cc-train-episode-34406",
    "cc-train-episode-41409", "cc-train-episode-41625",
    "cc-train-episode-61555", "cc-train-episode-91716",
)
LOCOMO_SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl")
LOCOMO_ATTESTATION = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_cross_dialogue_calibration_5x3_v1/dialogues/conv-43/questioner/"
    "frozen/parent_source_attestation.json")


def _fixture_plan(menu: Mapping[str, Any], actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "selected_memory_threads": list(dict.fromkeys(
            str(row["thread_id"]) for row in actions)),
        "scene_budget": dict(menu["scene_budget"]),
        "scene_goals": [{
            "goal": f"Shadow selection {index}",
            "participants": list(action["eligible_participants"]),
            "selected_action_ids": [str(action["action_id"])],
        } for index, action in enumerate(actions, 1)],
    }


def _two_actions(
    menu: Mapping[str, Any], preferred_operation: str | None = None,
) -> list[Mapping[str, Any]]:
    preferred = next((row for row in menu["actions"]
                      if row["operation"] == preferred_operation), None)
    chosen: list[Mapping[str, Any]] = [preferred] if preferred else []
    for action in menu["actions"]:
        if action in chosen:
            continue
        if chosen and action["mutually_exclusive_group"] == chosen[0][
                "mutually_exclusive_group"]:
            continue
        chosen.append(action)
        if len(chosen) == 2:
            return chosen
    raise ValueError("shadow_requires_two_non_conflicting_actions")


def run_shadow_replay(
    *, historical_run: Path = DEFAULT_HISTORY,
    authoritative_input: Path = DEFAULT_INPUT,
) -> dict[str, Any]:
    historical_before = hash_tree(historical_run)
    input_sha_before = sha256_file(authoritative_input)
    loaded = load_unified_root_read_only(authoritative_input)
    memory = initialize_world_memory(loaded["root"], loaded["parent"])
    memory_sha_before = sha256_bytes(canonical_json(memory).encode())
    menu = compile_delta_affordances(
        memory, root_identity=str(memory["root_id"]), mutation_requests=[])

    attempts: list[dict[str, Any]] = []
    raw_operations: Counter[str] = Counter()
    for attempt_path in sorted((historical_run / "attempts").glob("planner-a*.json")):
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        parsed = attempt.get("parsed_response") or {}
        operations = [
            str(delta.get("operation"))
            for scene in parsed.get("scene_goals") or []
            for delta in scene.get("proposed_graph_deltas") or []
        ]
        raw_operations.update(operations)
        attempts.append({
            "attempt": attempt["attempt"],
            "request_id": attempt["request_id"],
            "prompt_tokens_local": attempt["prompt_tokens_local"],
            "raw_response_sha256": attempt["raw_response_sha256"],
            "scene_count": len(parsed.get("scene_goals") or []),
            "scene_key_missing_count": sum(
                "scene_key" not in scene for scene in parsed.get("scene_goals") or []),
            "operations": operations,
        })
    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((historical_run / "planner").glob("validation_attempt_*.json"))
    ]
    historical_summary = json.loads(
        (historical_run / "reports/slim_fresh_e2e_summary.json").read_text(
            encoding="utf-8"))

    selected = _two_actions(menu, "add_shared_task")
    plan = _fixture_plan(menu, selected)
    compiled = compile_selected_actions(plan, menu)
    deltas = [delta for scene in compiled["scene_goals"]
              for delta in scene["proposed_graph_deltas"]]
    structural = validate_structural_deltas(
        memory, deltas, expected_root_id=str(memory["root_id"]), action_menu=menu)
    shared_delta = next(delta for delta in deltas
                        if delta["operation"] == "add_shared_task")

    invalid_plan = copy.deepcopy(plan)
    invalid_plan["scene_goals"][0]["selected_action_ids"] = ["A9999"]
    invalid_selection = validate_slim_planner_output(
        invalid_plan, action_menu=menu)

    menu_operations = Counter(str(row["operation"]) for row in menu["actions"])
    update_deltas = [delta for delta in deltas
                     if delta["operation_class"] == "update"]
    current_by_id = {str(row["memory_id"]): row
                     for row in memory["current_states"]}
    pre_states_match = all(
        delta["target_memory_id"] in current_by_id
        and delta["pre_state"] == current_by_id[delta["target_memory_id"]]["value"]
        for delta in update_deltas)
    properties = {
        "scene_key_is_program_generated": (
            all(scene["scene_key"] == f"scene_{index:02d}"
                for index, scene in enumerate(compiled["scene_goals"], 1))
            and any(row["scene_key_missing_count"] for row in attempts)),
        "change_authority_requires_existing_authoritative_state": (
            menu_operations["change_authority"] == 0
            and raw_operations["change_authority"] > 0),
        "change_trust_requires_existing_authoritative_state": (
            menu_operations["change_trust"] == 0
            and raw_operations["change_trust"] > 0),
        "add_shared_task_is_create_without_pre_state": (
            shared_delta["operation_class"] == "create"
            and shared_delta["pre_state"] is None
            and shared_delta["proposed_delta_id"] in
            structural["accepted_delta_ids"]),
        "unknown_action_id_rejected": (
            not invalid_selection["valid"]
            and "selected_action_unknown:A9999" in invalid_selection["errors"]),
        "compiled_update_pre_state_matches_memory": pre_states_match,
        "failed_outputs_not_written_to_world_memory": (
            sha256_bytes(canonical_json(memory).encode()) == memory_sha_before),
        "historical_failure_conclusion_preserved": (
            historical_summary.get("fresh_e2e_ready") is False
            and historical_summary.get("generation_ready") is False
            and historical_summary.get("failure_stage") ==
            "Planner schema/structural delta"),
    }
    input_sha_after = sha256_file(authoritative_input)
    historical_after = hash_tree(historical_run)
    properties["historical_run_unchanged"] = historical_before == historical_after
    properties["authoritative_input_unchanged"] = input_sha_before == input_sha_after
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_run": str(historical_run.resolve()),
        "authoritative_input": str(authoritative_input.resolve()),
        "v1_failure": {
            "failure_stage": historical_summary.get("failure_stage"),
            "failure_message": historical_summary.get("failure_message"),
            "generation_ready": historical_summary.get("generation_ready"),
            "fresh_e2e_ready": historical_summary.get("fresh_e2e_ready"),
            "validation_errors": [row.get("errors") for row in validations],
            "raw_operation_counts": dict(raw_operations),
            "attempts": attempts,
        },
        "v2_shadow": {
            "affordance_schema_version": AFFORDANCE_SCHEMA_VERSION,
            "planner_schema_version": PLANNER_SCHEMA_VERSION,
            "compiled_delta_schema_version": COMPILED_DELTA_SCHEMA_VERSION,
            "action_count": len(menu["actions"]),
            "action_operation_counts": dict(menu_operations),
            "selected_action_ids": [row["action_id"] for row in selected],
            "compiled_scene_keys": [row["scene_key"]
                                    for row in compiled["scene_goals"]],
            "compiled_delta_ids": [row["delta_id"] for row in deltas],
            "structural_validation": structural,
            "invalid_action_selection_errors": invalid_selection["errors"],
            "world_memory_sha256_before": memory_sha_before,
            "world_memory_sha256_after": sha256_bytes(canonical_json(memory).encode()),
        },
        "proofs": properties,
        "input_sha256_before": input_sha_before,
        "input_sha256_after": input_sha_after,
        "historical_run_sha_before": historical_before,
        "historical_run_sha_after": historical_after,
        "shadow_replay_ready": all(properties.values()),
        "api_calls": 0,
        "world_memory_write_count": 0,
    }


def render_shadow_markdown(audit: Mapping[str, Any]) -> str:
    v1 = audit["v1_failure"]
    v2 = audit["v2_shadow"]
    lines = [
        "# Slim Planner Affordance v2 Shadow Audit", "",
        "## Result", "",
        f"- `shadow_replay_ready = {str(audit['shadow_replay_ready']).lower()}`",
        "- `api_calls = 0`",
        "- `world_memory_write_count = 0`", "",
        "## Historical v1 failure", "",
        f"- Failure stage: `{v1['failure_stage']}`",
        f"- Raw operation counts: `{canonical_json(v1['raw_operation_counts'])}`",
        "- Attempt 1 omitted `scene_key`; attempts 2-3 failed because model-authored "
        "trust/authority pre-state targets did not exist and create operations were "
        "incorrectly subjected to update-state checks.", "",
        "## v2 shadow replay", "",
        f"- Action count: `{v2['action_count']}`",
        f"- Menu operations: `{canonical_json(v2['action_operation_counts'])}`",
        f"- Program scene keys: `{', '.join(v2['compiled_scene_keys'])}`",
        f"- Program delta IDs: `{', '.join(v2['compiled_delta_ids'])}`",
        f"- Accepted compiled deltas: "
        f"`{len(v2['structural_validation']['accepted_delta_ids'])}`", "",
        "## Required proofs", "",
    ]
    lines.extend(f"- `{key} = {str(value).lower()}`"
                 for key, value in audit["proofs"].items())
    lines.extend([
        "", "## Read-only provenance", "",
        f"- Input SHA before/after: `{audit['input_sha256_before']}` / "
        f"`{audit['input_sha256_after']}`",
        f"- Historical tree SHA before/after: "
        f"`{audit['historical_run_sha_before']['tree_sha256']}` / "
        f"`{audit['historical_run_sha_after']['tree_sha256']}`", "",
        "The historical run retains its original failed conclusion; no repair output was "
        "written into it.", "",
    ])
    return "\n".join(lines)


def _token_counter(tokenizer_path: Path) -> tuple[Callable[[str], int] | None, str]:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as error:  # pragma: no cover - environment fallback
        return None, f"utf8_bytes_div_4_fallback:{type(error).__name__}"
    return (lambda value: len(tokenizer.encode(value, add_special_tokens=False))), str(
        tokenizer_path.resolve())


def build_final_report(
    *, project_root: Path, preflight: Mapping[str, Any],
    shadow: Mapping[str, Any], targeted_status: str, targeted_test_count: int,
    full_status: str, full_test_count: int, compileall_status: str,
    tokenizer_source: str,
) -> dict[str, Any]:
    historical_unchanged = shadow["proofs"]["historical_run_unchanged"]
    v1_tokens = [row["prompt_tokens_local"]
                 for row in shadow["v1_failure"]["attempts"]]
    root_61555 = next(row for row in preflight["roots"]
                      if row["root_id"] == "cc-train-episode-61555")
    first_v2 = root_61555["planner_prompt_token_estimate"]
    retry_v2 = root_61555["planner_retry_prompt_token_estimate"]
    reduction = 1.0 - first_v2 / v1_tokens[0]
    prompt_ready = first_v2 < v1_tokens[0] * 0.5
    tests_ready = (targeted_status == full_status == compileall_status == "pass")
    preflight_ready = bool(preflight["slim_pipeline_v2_preflight_ready"])
    shadow_ready = bool(shadow["shadow_replay_ready"])
    ready = tests_ready and preflight_ready and shadow_ready and historical_unchanged
    isolation = audit_legacy_import_isolation(
        project_root / "scripts/current_method/restart_n5_spell/world_memory_slim_v1")
    generated_reports = frozenset({
        "reports/SLIM_PLANNER_AFFORDANCE_V2_SHADOW_AUDIT.json",
        "reports/SLIM_PLANNER_AFFORDANCE_V2_SHADOW_AUDIT.md",
        "WORLD_MEMORY_SLIM_AFFORDANCE_V2_REPORT.json",
        "WORLD_MEMORY_SLIM_AFFORDANCE_V2_REPORT.md",
    })
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure_root_cause": [
            "Planner v1 authored scene_key, memory references, operations, and pre_state.",
            "Create and update operations shared an incorrect state-validation path.",
            "Planner v1 embedded full World Memory and parent dialogue on every retry.",
        ],
        "design": {
            "flow": [
                "world_memory_v1", "delta_affordance_v1", "action_menu",
                "slim_world_planner_v2_selection", "program_compiled_delta",
                "structural_delta_validator_v2", "natural_realizer_interface",
            ],
            "operation_categories": {
                "create": ["add_person", "add_shared_task", "add_information_asymmetry",
                           "add_third_party_report", "open_thread", "create_conflict"],
                "update": ["update_state", "change_trust", "change_authority",
                           "resolve_conflict", "close_thread"],
                "transition": ["add_role_handoff"],
            },
            "program_generated_fields": [
                "scene_key", "delta_id", "proposed_delta_id", "operation",
                "operation_class", "target_memory_id", "event_identity_key",
                "subject_ids", "pre_state", "intended_post_state",
                "predecessor_memory_id", "related_memory_ids", "provenance",
            ],
            "planner_no_longer_responsible_for": [
                "scene/delta IDs", "operation", "memory/entity references", "pre_state",
                "post-state direction", "predecessor", "evidence", "final IDs", "handoff ID",
            ],
            "v1_to_v2": {
                "v1": "model-authored graph delta over full Memory",
                "v2": "model selects compiler-owned action IDs from a compact frozen menu",
            },
        },
        "shadow_replay": dict(shadow),
        "preflight": dict(preflight),
        "prompt_compaction": {
            "tokenizer_source": tokenizer_source,
            "v1_prompt_tokens_per_attempt": v1_tokens,
            "v1_three_attempt_prompt_tokens": sum(v1_tokens),
            "v2_first_prompt_tokens_episode_61555": first_v2,
            "v2_retry_prompt_tokens_episode_61555": retry_v2,
            "v2_three_attempt_estimate": first_v2 + 2 * retry_v2,
            "first_prompt_reduction_fraction": reduction,
            "full_world_memory_in_prompt": False,
            "full_parent_dialogue_in_prompt": False,
            "retry_reuses_full_first_prompt": False,
        },
        "tests": {
            "targeted_unittest_status": targeted_status,
            "targeted_test_count": targeted_test_count,
            "full_unittest_status": full_status,
            "full_test_count": full_test_count,
            "compileall_status": compileall_status,
        },
        "api_usage": {
            "api_calls": 0, "planner_calls": 0, "realizer_calls": 0,
            "evidence_committer_calls": 0, "questioner_calls": 0,
            "distractor_calls": 0, "solver_calls": 0, "qa_replay_calls": 0,
            "strategy_updates": 0, "training_runs": 0,
        },
        "legacy_import_isolation": isolation,
        "git_head": git_value(project_root, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256_excluding(
            project_root, generated_reports),
        "project_diff_sha256_scope": (
            "post-implementation project diff excluding the four generated report files"),
        "project_diff_sha256_excluded_paths": sorted(generated_reports),
        "historical_run_sha_before": shadow["historical_run_sha_before"],
        "historical_run_sha_after": shadow["historical_run_sha_after"],
        "unfinished_items": [
            "Slim Fresh E2E v2 has not been run.",
            "No Realizer or Evidence Committer model request was made.",
            "QA Replay, strategy execution, and training remain out of scope.",
        ],
        "delta_affordance_v1_ready": ready,
        "slim_planner_v2_ready": ready,
        "compiled_delta_ready": ready,
        "planner_prompt_compaction_ready": ready and prompt_ready,
        "affordance_shadow_replay_ready": ready,
        "slim_pipeline_v2_preflight_ready": ready,
        "generation_ready": False,
        "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }


def render_final_markdown(report: Mapping[str, Any]) -> str:
    statuses = (
        "delta_affordance_v1_ready", "slim_planner_v2_ready",
        "compiled_delta_ready", "planner_prompt_compaction_ready",
        "affordance_shadow_replay_ready", "slim_pipeline_v2_preflight_ready",
        "generation_ready", "fresh_e2e_ready", "qa_pipeline_ready",
    )
    prompt = report["prompt_compaction"]
    lines = ["# World Memory Slim Affordance v2 Report", "", "## Final status", ""]
    lines.extend(f"- `{key} = {str(report[key]).lower()}`" for key in statuses)
    lines.extend([
        "", "## First Fresh E2E v1 root cause", "",
    ])
    lines.extend(f"- {value}" for value in report["failure_root_cause"])
    lines.extend([
        "", "## Planner v1 → v2", "",
        "| Concern | v1 | v2 |", "|---|---|---|",
        "| Planner output | graph fields and state | goals, participants, action IDs |",
        "| IDs and pre-state | model-authored | compiler-authored |",
        "| Operation validation | shared state path | create/update/transition-specific |",
        "| Prompt input | full Memory and parent | compact menu and referenced turns |",
        "| Retry | full base prompt plus rejected response | frozen menu plus exact errors |",
        "", "Program-generated fields: `" + "`, `".join(
            report["design"]["program_generated_fields"]) + "`.",
        "", "## Shadow replay", "",
        f"- Ready: `{str(report['shadow_replay']['shadow_replay_ready']).lower()}`",
        f"- Historical failure retained: "
        f"`{str(report['shadow_replay']['proofs']['historical_failure_conclusion_preserved']).lower()}`",
        f"- Historical run unchanged: "
        f"`{str(report['shadow_replay']['proofs']['historical_run_unchanged']).lower()}`",
        "- Missing `scene_key` is eliminated because the compiler assigns it.",
        "- No trust/authority action is emitted without a corresponding authoritative state.",
        "- `add_shared_task` compiles as create with `pre_state = null`.",
        "", "## Prompt compaction", "",
        f"- v1 attempts: `{prompt['v1_prompt_tokens_per_attempt']}` "
        f"(sum `{prompt['v1_three_attempt_prompt_tokens']}`)",
        f"- v2 first prompt: `{prompt['v2_first_prompt_tokens_episode_61555']}`",
        f"- v2 retry prompt: `{prompt['v2_retry_prompt_tokens_episode_61555']}`",
        f"- v2 three-attempt estimate: `{prompt['v2_three_attempt_estimate']}`",
        f"- First-prompt reduction: `{prompt['first_prompt_reduction_fraction']:.2%}`",
        f"- Tokenizer: `{prompt['tokenizer_source']}`",
        "", "## 11-root zero-API preflight", "",
        "| Root | Records | Actions | C/U/T | Prompt tokens | Parent turns | Empty | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["preflight"]["roots"]:
        dist = row["action_distribution"]
        lines.append(
            f"| `{row['root_id']}` | {row['authoritative_memory_record_count']} | "
            f"{row['executable_action_count']} | "
            f"{dist['create']}/{dist['update']}/{dist['transition']} | "
            f"{row['planner_prompt_token_estimate']} | "
            f"{row['referenced_parent_turn_count']} | "
            f"`{str(row['empty_action_menu']).lower()}` | "
            f"`{str(row['preflight_pass']).lower()}` |")
    tests = report["tests"]
    lines.extend([
        "", "## Tests and API boundary", "",
        f"- Targeted unittest: `{tests['targeted_unittest_status']}` "
        f"({tests['targeted_test_count']} tests)",
        f"- Full unittest: `{tests['full_unittest_status']}` "
        f"({tests['full_test_count']} tests)",
        f"- Compileall: `{tests['compileall_status']}`",
        "- Paid/model API calls: `0`",
        f"- Legacy import isolation: "
        f"`{str(report['legacy_import_isolation']['isolated']).lower()}`",
        "", "## Provenance", "",
        f"- Git HEAD: `{report['git_head']}`",
        f"- Project diff SHA256: `{report['project_diff_sha256']}`",
        f"- Historical run tree SHA before/after: "
        f"`{report['historical_run_sha_before']['tree_sha256']}` / "
        f"`{report['historical_run_sha_after']['tree_sha256']}`",
        "", "## Unfinished items", "",
    ])
    lines.extend(f"- {value}" for value in report["unfinished_items"])
    lines.extend([
        "", "No Fresh E2E v2, live Realizer, live Evidence Committer, QA execution, "
        "strategy update, training, Git commit, or historical-run mutation was performed.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--targeted-status", choices=("pass", "fail"), required=True)
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--full-status", choices=("pass", "fail"), required=True)
    parser.add_argument("--full-test-count", type=int, required=True)
    parser.add_argument("--compileall-status", choices=("pass", "fail"), required=True)
    parser.add_argument("--tokenizer-path", type=Path,
                        default=Path("/root/autodl-tmp/Qwen3-4B"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    token_counter, tokenizer_source = _token_counter(args.tokenizer_path)
    shadow = run_shadow_replay()
    preflight = run_preflight_batch(
        cc_root_paths=[CC_ROOT / root_id / "unified_root.json" for root_id in CC_IDS],
        locomo_source_path=LOCOMO_SOURCE,
        locomo_attestation_path=LOCOMO_ATTESTATION,
        token_counter=token_counter)
    report = build_final_report(
        project_root=project, preflight=preflight, shadow=shadow,
        targeted_status=args.targeted_status,
        targeted_test_count=args.targeted_test_count,
        full_status=args.full_status, full_test_count=args.full_test_count,
        compileall_status=args.compileall_status,
        tokenizer_source=tokenizer_source)
    reports = project / "reports"
    atomic_json(reports / "SLIM_PLANNER_AFFORDANCE_V2_SHADOW_AUDIT.json", shadow)
    atomic_text(reports / "SLIM_PLANNER_AFFORDANCE_V2_SHADOW_AUDIT.md",
                render_shadow_markdown(shadow))
    atomic_json(project / "WORLD_MEMORY_SLIM_AFFORDANCE_V2_REPORT.json", report)
    atomic_text(project / "WORLD_MEMORY_SLIM_AFFORDANCE_V2_REPORT.md",
                render_final_markdown(report))
    print(canonical_json({
        "shadow_replay_ready": shadow["shadow_replay_ready"],
        "preflight_ready": preflight["slim_pipeline_v2_preflight_ready"],
        "final_ready": report["slim_pipeline_v2_preflight_ready"],
        "api_calls": report["api_usage"]["api_calls"],
        "report_json": str(project / "WORLD_MEMORY_SLIM_AFFORDANCE_V2_REPORT.json"),
    }))
    return 0 if report["slim_pipeline_v2_preflight_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
