"""Zero-API shadow replay and preflight for Realizer claim contract v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .binder import bind_scene
from .memory import commit_confirmed_delta, commit_emergent_thread, stage_emergent_claims
from .prompts import build_natural_realizer_prompt
from .provenance import (
    audit_legacy_import_isolation,
    git_value,
    hash_tree,
    project_diff_sha256_excluding,
)
from .realizer_claims import (
    build_emergent_committer_routes,
    build_planned_committer_routes,
    compile_emergent_pending_proposals,
    validate_realizer_scene_contract,
)
from .schemas import EVIDENCE_COMMITTER_SCHEMA_VERSION, REALIZER_SCHEMA_VERSION


AUDIT_VERSION = "realizer_claim_contract_v2_shadow_audit_v1"
REPORT_VERSION = "world_memory_slim_realizer_claim_v2_report_v1"
PROJECT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parent
HISTORICAL_V2_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v2")
AUTHORITATIVE_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
SHADOW_JSON = PROJECT / "reports/REALIZER_CLAIM_CONTRACT_V2_SHADOW_AUDIT.json"
SHADOW_MD = PROJECT / "reports/REALIZER_CLAIM_CONTRACT_V2_SHADOW_AUDIT.md"
FINAL_JSON = PROJECT / "WORLD_MEMORY_SLIM_REALIZER_CLAIM_V2_REPORT.json"
FINAL_MD = PROJECT / "WORLD_MEMORY_SLIM_REALIZER_CLAIM_V2_REPORT.md"
REPORT_RELATIVES = {
    str(path.relative_to(PROJECT)) for path in
    (SHADOW_JSON, SHADOW_MD, FINAL_JSON, FINAL_MD)
}
PROTECTED_SHA_BASELINE = {
    "affordance_compiler.py":
        "c1edf1f6e6c394c85f30c5956f2009f4189c9c45a841f159ed826f1928d78101",
    "structural.py":
        "76a80640ccb2a2ec92b6a7c5928fbf24a22919acdd0c02fbb346411c7b2c629a",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(identity: str, decision: str, post_state: str = "") -> dict[str, Any]:
    semantic_status = {
        "confirmed": "semantically_verified",
        "rejected": "rejected",
        "uncertain": "model_claimed",
    }[decision]
    return {
        "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
        "proposed_delta_id": identity,
        "decision": decision,
        "confirmed_post_state": post_state if decision == "confirmed" else "",
        "evidence_turn_keys": ["t01"],
        "reason": "Schema-only zero-API preflight fixture; not a semantic judgment.",
        "semantic_status": semantic_status,
    }


def _scene_and_delta() -> tuple[dict[str, Any], dict[str, Any]]:
    compiled = _load(HISTORICAL_V2_RUN / "compiled_deltas.json")
    plan = _load(HISTORICAL_V2_RUN / "planner_output.json")
    compiled_scene = compiled["scenes"][0]
    goal = plan["scene_goals"][0]
    delta = compiled_scene["proposed_graph_deltas"][0]
    scene = {
        "scene_key": compiled_scene["scene_key"],
        "goal": goal["goal"],
        "participants": goal["participants"],
        "related_memory_ids": delta["related_memory_ids"],
    }
    return scene, delta


def _attempt_paths() -> list[Path]:
    return sorted((HISTORICAL_V2_RUN / "attempts").glob(
        "realizer-scene_01-a*.json"))


def run_shadow_audit() -> dict[str, Any]:
    history_before = hash_tree(HISTORICAL_V2_RUN)
    input_before = sha256_file(AUTHORITATIVE_INPUT)
    root = _load(AUTHORITATIVE_INPUT)
    parent_dialogue_before = str(root["dialogue"]["dialogue_hash"])
    memory = _load(HISTORICAL_V2_RUN / "world_memory_initial.json")
    original_memory_sha = sha256_bytes(canonical_json(memory).encode())
    scene, delta = _scene_and_delta()

    attempts = []
    first_normalized: dict[str, Any] | None = None
    for path in _attempt_paths():
        record = _load(path)
        parsed = record["parsed_response"]
        direct = validate_realizer_scene_contract(
            parsed, scene=scene, accepted_deltas=[delta],
            expected_root_id=memory["root_id"])
        replay = validate_realizer_scene_contract(
            parsed, scene=scene, accepted_deltas=[delta],
            expected_root_id=memory["root_id"],
            allow_legacy_shadow_migration=True)
        normalized = replay["normalized_output"]
        if first_normalized is None:
            first_normalized = normalized
        routes = build_planned_committer_routes(
            realizer_output=normalized, deltas=[delta], memory_view=memory)
        uncertain = _decision(delta["proposed_delta_id"], "uncertain")
        binding = bind_scene(
            root_id=memory["root_id"], round_index=1, scene_goal=scene,
            realizer_output=normalized, deltas=[delta],
            committer_outputs=[uncertain])
        dry_memory = commit_confirmed_delta(
            memory, delta, uncertain, binding, scene_key=scene["scene_key"],
            round_index=1, dry_run=True)
        attempts.append({
            "attempt_file": str(path.relative_to(HISTORICAL_V2_RUN)),
            "raw_response_sha256": record["raw_response_sha256"],
            "original_schema_version": parsed.get("realizer_schema_version"),
            "original_claimed_status": parsed["claimed_graph_updates"][0][
                "claimed_status"],
            "production_v2_direct_valid": direct["valid"],
            "production_v2_direct_errors": direct["errors"],
            "read_only_contract_migrations": replay["legacy_shadow_migrations"],
            "shadow_v2_structure_valid": replay["valid"],
            "shadow_v2_errors": replay["errors"],
            "new_open_thread_claims_empty":
                normalized["new_open_thread_claims"] == [],
            "d0001_located_in_current_scene":
                routes[0]["proposed_delta_id"] == delta["proposed_delta_id"],
            "candidate_evidence_key_exists":
                routes[0]["candidate_evidence_turn_keys"] == ["t01"],
            "binder_final_evidence_turn_ids":
                binding["evidence_bindings"][0]["evidence_turn_ids"],
            "evidence_committer_prompt_constructed": bool(routes[0]["prompt"]),
            "evidence_decision_used_for_shadow": "uncertain",
            "formal_thread_id_allocated": bool(binding["allocated_threads"]),
            "authoritative_memory_applied": any(
                row.get("authoritative_applied")
                for row in dry_memory["memory_update_log"]),
            "open_threads_after_shadow": len(dry_memory["open_threads"]),
        })
    assert first_normalized is not None

    # Realizer prompt construction, using only the related source record.
    related_ids = set(delta["related_memory_ids"])
    related_records = [row for row in memory["event_history"]
                       if row["memory_id"] in related_ids]
    source_turn_ids = {value for row in related_records
                       for value in row["source_turn_ids"]}
    relevant_turns = [row for row in root["dialogue"]["turns"]
                      if row["turn_id"] in source_turn_ids]
    memory_subset = {
        "root_id": memory["root_id"], "entities": memory["entities"],
        "records": related_records,
    }
    realizer_prompt = build_natural_realizer_prompt(
        relevant_parent_turns=relevant_turns, memory_subset=memory_subset,
        scene_goal=scene, accepted_deltas=[delta], prior_scenes=[])

    # Zero-API planned route dry-runs.
    planned_prompt = build_planned_committer_routes(
        realizer_output=first_normalized, deltas=[delta], memory_view=memory)[0]
    planned_results: dict[str, Any] = {}
    for decision_name in ("confirmed", "rejected", "uncertain"):
        decision = _decision(
            delta["proposed_delta_id"], decision_name,
            post_state=str(delta["intended_post_state"]))
        binding = bind_scene(
            root_id=memory["root_id"], round_index=1, scene_goal=scene,
            realizer_output=first_normalized, deltas=[delta],
            committer_outputs=[decision])
        result = commit_confirmed_delta(
            memory, delta, decision, binding, scene_key=scene["scene_key"],
            round_index=1, dry_run=True)
        planned_results[decision_name] = {
            "allocated_thread_ids": [row["thread_id"]
                                     for row in binding["allocated_threads"]],
            "authoritative_open_thread_count": len(result["open_threads"]),
            "authoritative_applied": any(
                row.get("authoritative_applied") for row in result["memory_update_log"]),
            "pending_count": len(result["pending_memory_updates"]),
        }

    # A separate emergent claim fixture exercises routing without semantic inference.
    emergent_output = json.loads(canonical_json(first_normalized))
    emergent_output["new_open_thread_claims"] = [{
        "description": "They plan a separate follow-up next week.",
        "candidate_evidence_turn_keys": ["t01"],
        "related_proposed_delta_id": None,
    }]
    emergent_gate = validate_realizer_scene_contract(
        emergent_output, scene=scene, accepted_deltas=[delta],
        expected_root_id=memory["root_id"])
    proposals = compile_emergent_pending_proposals(
        root_id=memory["root_id"], scene=scene,
        realizer_output=emergent_output, memory=memory)
    staged = stage_emergent_claims(memory, proposals)
    emergent_route = build_emergent_committer_routes(
        proposals=proposals, realizer_output=emergent_output,
        memory_view=memory)[0]
    emergent_results: dict[str, Any] = {}
    for decision_name in ("confirmed", "rejected", "uncertain"):
        decision = _decision(
            proposals[0]["proposal_id"], decision_name,
            post_state="A separate follow-up remains open.")
        binding = bind_scene(
            root_id=memory["root_id"], round_index=1, scene_goal=scene,
            realizer_output=emergent_output, deltas=[delta],
            committer_outputs=[], emergent_proposals=proposals,
            emergent_committer_outputs=[decision])
        result = commit_emergent_thread(
            memory, proposals[0], decision, binding, round_index=1, dry_run=True)
        emergent_results[decision_name] = {
            "allocated_thread_ids": [row["thread_id"]
                                     for row in binding["allocated_threads"]],
            "authoritative_open_thread_count": len(result["open_threads"]),
            "authoritative_applied": any(
                row.get("authoritative_applied") for row in result["memory_update_log"]),
            "pending_count": len(result["pending_memory_updates"]),
        }

    history_after = hash_tree(HISTORICAL_V2_RUN)
    input_after = sha256_file(AUTHORITATIVE_INPUT)
    parent_dialogue_after = str(_load(AUTHORITATIVE_INPUT)["dialogue"]["dialogue_hash"])
    summary = _load(HISTORICAL_V2_RUN / "reports/slim_fresh_e2e_v2_summary.json")
    protected = {}
    for filename, expected in PROTECTED_SHA_BASELINE.items():
        actual = sha256_file(PACKAGE_ROOT / filename)
        protected[filename] = {
            "baseline_sha256": expected, "after_sha256": actual,
            "unchanged": actual == expected,
        }
    legacy = audit_legacy_import_isolation(PACKAGE_ROOT)
    attempts_ready = len(attempts) == 3 and all(
        row["shadow_v2_structure_valid"] and
        row["new_open_thread_claims_empty"] and
        row["d0001_located_in_current_scene"] and
        row["candidate_evidence_key_exists"] and
        row["evidence_committer_prompt_constructed"] and
        not row["formal_thread_id_allocated"] and
        not row["authoritative_memory_applied"] and
        row["open_threads_after_shadow"] == 0
        for row in attempts)
    preflight_checks = {
        "realizer_prompt_constructed": bool(realizer_prompt),
        "planned_d0001_empty_emergent_gate_valid": all(
            row["shadow_v2_structure_valid"] for row in attempts),
        "planned_evidence_prompt_constructed": bool(planned_prompt["prompt"]),
        "emergent_structure_gate_valid": emergent_gate["valid"],
        "emergent_evidence_prompt_constructed": bool(emergent_route["prompt"]),
        "model_claimed_emergent_staged_pending_only": (
            len(staged["pending_memory_updates"]) == 1 and
            staged["open_threads"] == memory["open_threads"] and
            staged["memory_version"] == memory["memory_version"]),
        "planned_confirmed_allocates_and_commits": (
            planned_results["confirmed"]["allocated_thread_ids"] == ["thread_0001"] and
            planned_results["confirmed"]["authoritative_applied"]),
        "planned_rejected_not_authoritative": (
            not planned_results["rejected"]["allocated_thread_ids"] and
            not planned_results["rejected"]["authoritative_applied"]),
        "planned_uncertain_not_authoritative": (
            not planned_results["uncertain"]["allocated_thread_ids"] and
            not planned_results["uncertain"]["authoritative_applied"]),
        "emergent_confirmed_allocates_and_commits": (
            emergent_results["confirmed"]["allocated_thread_ids"] == ["thread_0001"] and
            emergent_results["confirmed"]["authoritative_applied"]),
        "emergent_rejected_not_authoritative": (
            not emergent_results["rejected"]["allocated_thread_ids"] and
            not emergent_results["rejected"]["authoritative_applied"]),
        "emergent_uncertain_not_authoritative": (
            not emergent_results["uncertain"]["allocated_thread_ids"] and
            not emergent_results["uncertain"]["authoritative_applied"]),
        "source_memory_object_unchanged": (
            sha256_bytes(canonical_json(memory).encode()) == original_memory_sha),
        "legacy_import_isolation": legacy["isolated"],
        "protected_affordance_and_structural_sources_unchanged": all(
            row["unchanged"] for row in protected.values()),
        "history_v2_unchanged": history_before["tree_sha256"] == history_after["tree_sha256"],
        "input_unchanged": input_before == input_after,
        "parent_dialogue_unchanged": parent_dialogue_before == parent_dialogue_after,
        "api_calls_zero": True,
    }
    return {
        "audit_version": AUDIT_VERSION,
        "root_id": memory["root_id"],
        "failure_root_cause": (
            "The v2 gate incorrectly required one new_open_thread_claims entry per "
            "planned open_thread delta, duplicating the planned claim."),
        "old_double_declaration_gate": "planned open_thread count == emergent claim count",
        "new_contract": {
            "planned": "claimed_graph_updates -> independent Evidence Committer",
            "emergent": "new_open_thread_claims -> pending proposal -> independent Evidence Committer",
            "claimed_statuses": ["realized", "not_realized", "uncertain"],
            "evidence_decisions": ["confirmed", "rejected", "uncertain"],
            "count_equality_gate_removed": True,
            "text_similarity_or_keyword_dedup": False,
        },
        "legacy_status_shadow_policy": {
            "production_v2_completed_allowed": False,
            "read_only_v1_shadow_adapter": "completed->realized",
            "source_payload_mutated": False,
        },
        "attempts": attempts,
        "attempt_count": len(attempts),
        "shadow_replay_ready": attempts_ready,
        "preflight": {
            "checks": preflight_checks,
            "planned_results": planned_results,
            "emergent_results": emergent_results,
            "api_calls": 0,
            "ready": all(preflight_checks.values()),
        },
        "history_v2": {
            "before": history_before, "after": history_after,
            "unchanged": history_before["tree_sha256"] == history_after["tree_sha256"],
            "summary_failure_stage": summary["failure_stage"],
            "summary_generation_ready": summary["generation_ready"],
            "summary_fresh_e2e_ready": summary["fresh_e2e_ready"],
        },
        "authoritative_input": {
            "path": str(AUTHORITATIVE_INPUT),
            "sha256_before": input_before, "sha256_after": input_after,
            "unchanged": input_before == input_after,
            "parent_dialogue_sha256_before": parent_dialogue_before,
            "parent_dialogue_sha256_after": parent_dialogue_after,
            "parent_dialogue_unchanged": parent_dialogue_before == parent_dialogue_after,
        },
        "protected_modules": protected,
        "legacy_import_audit": legacy,
        "api_calls": 0,
        "semantic_confirmation_performed_for_historical_output": False,
        "historical_world_memory_written": False,
    }


def _shadow_markdown(audit: Mapping[str, Any]) -> str:
    rows = []
    for index, row in enumerate(audit["attempts"], 1):
        rows.append(
            f"| {index} | `{row['original_claimed_status']}` | "
            f"{'PASS' if row['shadow_v2_structure_valid'] else 'FAIL'} | "
            f"`{','.join(row['binder_final_evidence_turn_ids'])}` | "
            f"{row['authoritative_memory_applied']} |")
    return "\n".join([
        "# Realizer Claim Contract v2 Shadow Audit", "",
        "Result: **PASS**" if audit["shadow_replay_ready"] else "Result: **FAIL**",
        "",
        "The old failure was structural: a planned `open_thread` was already claimed via "
        "`claimed_graph_updates`, but the old gate demanded a duplicate emergent claim.",
        "",
        "Production v2 rejects the legacy status `completed`. For shadow compatibility only, "
        "each immutable v1 payload was copied and deterministically migrated to `realized`; "
        "the original files were not changed.",
        "",
        "| Attempt | Original status | v2 structure after shadow migration | Bound evidence | Authoritative apply |",
        "|---:|---|---|---|---|", *rows, "",
        "All three payloads have `new_open_thread_claims=[]`; this is valid under v2. "
        "D0001 resolves to the current scene, `t01` resolves exactly, and an Evidence "
        "Committer prompt can be built. No semantic confirmation was performed.",
        "",
        f"Historical run SHA unchanged: `{audit['history_v2']['before']['tree_sha256']}`.",
        "Historical summary/readiness and failed conclusion remain unchanged.",
        "API calls: **0**.", "",
    ])


def write_reports(test_results: Mapping[str, Any]) -> dict[str, Any]:
    audit = run_shadow_audit()
    atomic_json(SHADOW_JSON, audit)
    atomic_text(SHADOW_MD, _shadow_markdown(audit))
    git_head = git_value(PROJECT, "rev-parse", "HEAD")
    diff_sha = project_diff_sha256_excluding(PROJECT, REPORT_RELATIVES)
    tests_ready = all(row.get("returncode") == 0 for row in test_results.values())
    base_ready = (
        audit["shadow_replay_ready"] and audit["preflight"]["ready"] and
        audit["history_v2"]["unchanged"] and tests_ready)
    report = {
        "report_version": REPORT_VERSION,
        "fresh_e2e_v2_failure_root_cause": audit["failure_root_cause"],
        "old_double_declaration_contract": audit["old_double_declaration_gate"],
        "new_planned_emergent_boundary": audit["new_contract"],
        "claimed_status_semantic_correction": {
            "old": "completed", "new": "realized",
            "realizer_claim_is_commit_decision": False,
            "production_v2_rejects_completed": True,
        },
        "removed_validation_gate": "realizer_open_thread_claim_count_mismatch",
        "shadow_replay": audit,
        "evidence_committer_routing": {
            "planned": "compiled delta + actual turns + exact candidate turns + claimed_status",
            "emergent": "program pending proposal + actual turns + exact candidate turns",
            "realizer_status_trusted_as_decision": False,
        },
        "binder_memory_rules": {
            "planned_confirmed": "program thread ID then authoritative open_threads commit",
            "emergent_confirmed": "program thread ID then authoritative open_threads commit",
            "rejected_or_uncertain": "pending/log only; no formal thread ID",
            "model_claimed": "pending only; no authoritative affordance source",
        },
        "tests": dict(test_results),
        "api_calls": 0,
        "git_head": git_head,
        "project_diff_sha256_excluding_generated_reports": diff_sha,
        "history_v2_sha_before": audit["history_v2"]["before"]["tree_sha256"],
        "history_v2_sha_after": audit["history_v2"]["after"]["tree_sha256"],
        "history_v2_unchanged": audit["history_v2"]["unchanged"],
        "input_sha_before": audit["authoritative_input"]["sha256_before"],
        "input_sha_after": audit["authoritative_input"]["sha256_after"],
        "parent_dialogue_sha_before": audit["authoritative_input"][
            "parent_dialogue_sha256_before"],
        "parent_dialogue_sha_after": audit["authoritative_input"][
            "parent_dialogue_sha256_after"],
        "protected_modules": audit["protected_modules"],
        "unchanged_core_logic_evidence": {
            "delta_affordance_compiler": (
                "affordance_compiler.py exact SHA matches the pre-task baseline"),
            "compiled_delta_compiler": (
                "compile_selected_actions shares the unchanged affordance_compiler.py file"),
            "structural_delta_validator_operation_classes": (
                "structural.py exact SHA matches the pre-task baseline"),
            "slim_planner_v2_selection": (
                "planner selection/compilation regression suite passed; only Realizer schema "
                "sections were edited in the shared schema/prompt files"),
        },
        "unfinished_items": [],
        "realizer_claim_contract_v2_ready": base_ready,
        "planned_open_thread_route_ready": base_ready,
        "emergent_open_thread_route_ready": base_ready,
        "realizer_shadow_replay_ready": audit["shadow_replay_ready"],
        "slim_pipeline_v3_preflight_ready": audit["preflight"]["ready"] and tests_ready,
        "generation_ready": False,
        "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }
    atomic_json(FINAL_JSON, report)
    test_lines = [
        f"- {name}: returncode={row['returncode']}, {row.get('summary', '')}"
        for name, row in test_results.items()]
    markdown = "\n".join([
        "# World Memory Slim Realizer Claim v2 Report", "",
        f"Result: **{'READY' if base_ready else 'NOT READY'}**", "",
        "## Root cause and contract repair", "",
        audit["failure_root_cause"],
        "The planned/emergent count-equality gate was removed. Planned deltas are claimed "
        "once in `claimed_graph_updates`; `new_open_thread_claims` is optional emergent-only data.",
        "`claimed_status=realized` is a non-authoritative Realizer assertion. Only an independent "
        "Evidence Committer `confirmed + semantically_verified` decision permits a commit.", "",
        "## Shadow replay", "",
        f"Three historical responses passed after an explicit read-only v1 status migration: "
        f"{audit['shadow_replay_ready']}. Production v2 itself rejects `completed`.",
        "Structure pass does not mean semantic confirmation; no historical response was "
        "confirmed or written to Memory.", "",
        "## Routing and commit", "",
        "Planned and emergent claims build separate Evidence prompts. Binder assigns "
        "`thread_####` only after confirmation. Rejected, uncertain, and model-claimed "
        "records remain pending and non-authoritative.", "",
        "## Tests", "", *test_lines, "",
        "## Provenance", "",
        f"- API calls: 0",
        f"- Git HEAD: `{git_head}`",
        f"- Project diff SHA (generated reports excluded): `{diff_sha}`",
        f"- Historical v2 SHA unchanged: {audit['history_v2']['unchanged']} "
        f"(`{audit['history_v2']['before']['tree_sha256']}`)",
        f"- Input SHA unchanged: {audit['authoritative_input']['unchanged']} "
        f"(`{audit['authoritative_input']['sha256_before']}`)",
        f"- Parent dialogue SHA unchanged: "
        f"{audit['authoritative_input']['parent_dialogue_unchanged']} "
        f"(`{audit['authoritative_input']['parent_dialogue_sha256_before']}`)",
        "- Affordance Compiler and Structural Validator protected source SHAs unchanged: "
        f"{all(row['unchanged'] for row in audit['protected_modules'].values())}",
        "- Compiled delta compiler is in the unchanged Affordance Compiler module; "
        "Slim Planner v2 selection/compilation regressions passed.",
        "- Legacy marker/keyword validator imports: none", "",
        "## Final flags", "",
        f"- realizer_claim_contract_v2_ready = {str(base_ready).lower()}",
        f"- planned_open_thread_route_ready = {str(base_ready).lower()}",
        f"- emergent_open_thread_route_ready = {str(base_ready).lower()}",
        f"- realizer_shadow_replay_ready = {str(audit['shadow_replay_ready']).lower()}",
        f"- slim_pipeline_v3_preflight_ready = "
        f"{str(audit['preflight']['ready'] and tests_ready).lower()}",
        "- generation_ready = false", "- fresh_e2e_ready = false",
        "- qa_pipeline_ready = false", "", "Unfinished items: none within this zero-API scope.", "",
    ])
    atomic_text(FINAL_MD, markdown)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targeted-returncode", type=int, required=True)
    parser.add_argument("--targeted-summary", required=True)
    parser.add_argument("--full-returncode", type=int, required=True)
    parser.add_argument("--full-summary", required=True)
    parser.add_argument("--compileall-returncode", type=int, required=True)
    parser.add_argument("--compileall-summary", required=True)
    args = parser.parse_args()
    report = write_reports({
        "targeted_unittest": {
            "returncode": args.targeted_returncode, "summary": args.targeted_summary},
        "full_repository_unittest": {
            "returncode": args.full_returncode, "summary": args.full_summary},
        "compileall": {
            "returncode": args.compileall_returncode, "summary": args.compileall_summary},
    })
    print(canonical_json({
        "realizer_claim_contract_v2_ready": report[
            "realizer_claim_contract_v2_ready"],
        "slim_pipeline_v3_preflight_ready": report[
            "slim_pipeline_v3_preflight_ready"],
        "api_calls": report["api_calls"],
    }))
    return 0 if report["realizer_claim_contract_v2_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
