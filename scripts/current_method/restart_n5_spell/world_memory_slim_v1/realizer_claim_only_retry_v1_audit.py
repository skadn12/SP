"""Zero-API v5 shadow audit for Realizer claim-only retry v1."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_file
from .binder import bind_scene
from .memory import commit_confirmed_delta
from .prompts import (
    build_evidence_committer_prompt,
    build_realizer_claim_only_retry_prompt,
)
from .provenance import (
    audit_legacy_import_isolation,
    git_value,
    hash_tree,
    project_diff_sha256_excluding,
)
from .realizer_claims import (
    build_planned_committer_routes,
    reconstruct_scene_from_claim_only,
    validate_claim_only_retry_contract,
    validate_committer_candidate_subset,
    validate_realizer_scene_contract,
)
from .schemas import EVIDENCE_COMMITTER_SCHEMA_VERSION


AUDIT_VERSION = "realizer_claim_only_retry_v1_shadow_audit_v1"
PROJECT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
V5_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v5")
INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
SHADOW_JSON = PROJECT / "reports/REALIZER_CLAIM_ONLY_RETRY_V1_SHADOW_AUDIT.json"
SHADOW_MD = PROJECT / "reports/REALIZER_CLAIM_ONLY_RETRY_V1_SHADOW_AUDIT.md"
REPORT_JSON = PROJECT / "WORLD_MEMORY_SLIM_REALIZER_CLAIM_ONLY_RETRY_V1_REPORT.json"
REPORT_MD = PROJECT / "WORLD_MEMORY_SLIM_REALIZER_CLAIM_ONLY_RETRY_V1_REPORT.md"
GENERATED_RELATIVES = frozenset({
    SHADOW_JSON.relative_to(PROJECT).as_posix(),
    SHADOW_MD.relative_to(PROJECT).as_posix(),
    REPORT_JSON.relative_to(PROJECT).as_posix(),
    REPORT_MD.relative_to(PROJECT).as_posix(),
})


def _decision(delta_id: str, decision: str, evidence: list[str]) -> dict[str, Any]:
    status = {
        "confirmed": "semantically_verified",
        "rejected": "rejected",
        "uncertain": "model_claimed",
    }[decision]
    return {
        "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
        "proposed_delta_id": delta_id,
        "decision": decision,
        "confirmed_post_state": "reflection_open" if decision == "confirmed" else "",
        "evidence_turn_keys": evidence,
        "reason": "Offline preflight fixture; no semantic decision is inferred.",
        "semantic_status": status,
    }


def _claim(delta_id: str, status: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "claimed_graph_updates": [{
            "proposed_delta_id": delta_id,
            "claimed_status": status,
            "candidate_evidence_turn_keys": evidence,
        }],
        "new_open_thread_claims": [],
    }


def build_shadow_audit() -> dict[str, Any]:
    before = hash_tree(V5_RUN)
    input_before = sha256_file(INPUT)
    plan = json.loads((V5_RUN / "planner_output.json").read_text(encoding="utf-8"))
    compiled = json.loads((V5_RUN / "compiled_deltas.json").read_text(encoding="utf-8"))
    memory = json.loads((V5_RUN / "world_memory_initial.json").read_text(encoding="utf-8"))
    delta = compiled["scenes"][2]["proposed_graph_deltas"][0]
    plan_scene = plan["scene_goals"][2]
    scene = {
        "scene_key": "scene_03", "goal": plan_scene["goal"],
        "participants": plan_scene["participants"],
        "related_memory_ids": delta["related_memory_ids"],
    }
    historical_outputs = [json.loads(
        (V5_RUN / f"attempts/realizer-scene_03-a{index:02d}.json").read_text(
            encoding="utf-8"))["parsed_response"] for index in (1, 2, 3)]
    historical_results = [validate_realizer_scene_contract(
        output, scene=scene, accepted_deltas=[delta],
        expected_root_id=memory["root_id"])
        for output in historical_outputs]
    frozen_turns = copy.deepcopy(historical_outputs[0]["dialogue_turns"])
    allowed_keys = [str(turn["turn_key"]) for turn in frozen_turns]
    invalid_key = str(historical_outputs[0]["claimed_graph_updates"][0][
        "candidate_evidence_turn_keys"][0])
    retry_prompt = build_realizer_claim_only_retry_prompt(
        scene_key=scene["scene_key"], frozen_dialogue_turns=frozen_turns,
        proposed_delta_ids=[str(delta["proposed_delta_id"])],
        errors=historical_results[0]["errors"])

    realized_claim = _claim(str(delta["proposed_delta_id"]), "realized", allowed_keys)
    realized_validation = validate_claim_only_retry_contract(
        realized_claim, frozen_scene_key=scene["scene_key"],
        frozen_dialogue_turns=frozen_turns, accepted_deltas=[delta])
    reconstructed = reconstruct_scene_from_claim_only(
        frozen_scene_key=scene["scene_key"], frozen_dialogue_turns=frozen_turns,
        claim_only_output=realized_claim)
    full_validation = validate_realizer_scene_contract(
        reconstructed, scene=scene, accepted_deltas=[delta],
        expected_root_id=memory["root_id"])
    committer_prompt = build_evidence_committer_prompt(
        proposed_delta=delta, dialogue_turns=frozen_turns,
        realizer_claims=realized_claim["claimed_graph_updates"], memory_view=memory)
    planned_route = build_planned_committer_routes(
        realizer_output=reconstructed, deltas=[delta], memory_view=memory)[0]

    decision_results: dict[str, Any] = {}
    for name in ("confirmed", "rejected", "uncertain"):
        evidence = allowed_keys[:1] if name != "rejected" else []
        decision = _decision(str(delta["proposed_delta_id"]), name, evidence)
        subset = validate_committer_candidate_subset(
            decision, candidate_evidence_turn_keys=allowed_keys,
            realizer_claimed_status="realized")
        binding = bind_scene(
            root_id=memory["root_id"], round_index=1, scene_goal=scene,
            realizer_output=reconstructed, deltas=[delta],
            committer_outputs=[decision])
        updated = commit_confirmed_delta(
            memory, delta, decision, binding, scene_key=scene["scene_key"],
            round_index=1, dry_run=True)
        decision_results[name] = {
            "candidate_subset_valid": subset["valid"],
            "allocated_thread_count": len(binding["allocated_threads"]),
            "authoritative_open_thread_count": len(updated["open_threads"]),
            "pending_count": len(updated["pending_memory_updates"]),
            "authoritative_applied": updated["memory_update_log"][-1][
                "authoritative_applied"],
        }

    outsider = _decision(
        str(delta["proposed_delta_id"]), "confirmed", ["not-a-candidate"])
    outsider_validation = validate_committer_candidate_subset(
        outsider, candidate_evidence_turn_keys=allowed_keys,
        realizer_claimed_status="realized")
    status_fixtures = {}
    for status, evidence in (("not_realized", []), ("uncertain", [])):
        fixture = _claim(str(delta["proposed_delta_id"]), status, evidence)
        result = validate_claim_only_retry_contract(
            fixture, frozen_scene_key=scene["scene_key"],
            frozen_dialogue_turns=frozen_turns, accepted_deltas=[delta])
        status_fixtures[status] = {
            "valid": result["valid"],
            "planned_route_constructible": bool(build_planned_committer_routes(
                realizer_output=reconstruct_scene_from_claim_only(
                    frozen_scene_key=scene["scene_key"],
                    frozen_dialogue_turns=frozen_turns,
                    claim_only_output=fixture),
                deltas=[delta], memory_view=memory)),
        }

    after = hash_tree(V5_RUN)
    checks = {
        "three_old_claims_strictly_rejected": all(
            not result["valid"] for result in historical_results),
        "three_old_dialogues_valid": all(
            result["dialogue_valid"] for result in historical_results),
        "three_classified_claim_only_retry": all(
            result["retry_mode"] == "claim_only_retry"
            for result in historical_results),
        "allowed_keys_are_current_scene_only": set(allowed_keys) == {
            str(turn["turn_key"]) for turn in frozen_turns},
        "invalid_key_not_in_prompt": invalid_key not in retry_prompt,
        "old_full_response_not_in_prompt": json.dumps(
            historical_outputs[0], sort_keys=True) not in retry_prompt,
        "prompt_template_is_last": set(json.loads(retry_prompt.splitlines()[-1])) == {
            "claimed_graph_updates", "new_open_thread_claims"},
        "realized_fixture_valid": realized_validation["valid"],
        "dialogue_frozen_field_exact": reconstructed["dialogue_turns"] == frozen_turns,
        "reconstructed_full_scene_valid": full_validation["valid"],
        "evidence_prompt_constructed": bool(committer_prompt),
        "planned_route_candidate_exact": set(
            planned_route["candidate_evidence_turn_keys"]) == set(allowed_keys),
        "not_realized_fixture_valid": status_fixtures["not_realized"]["valid"],
        "uncertain_fixture_valid": status_fixtures["uncertain"]["valid"],
        "committer_outside_candidate_rejected": not outsider_validation["valid"],
        "confirmed_only_authoritative": (
            decision_results["confirmed"]["authoritative_applied"] and
            not decision_results["rejected"]["authoritative_applied"] and
            not decision_results["uncertain"]["authoritative_applied"]),
        "shadow_does_not_auto_confirm_historical_claim": all(
            result["valid"] is False for result in historical_results),
        "three_attempt_exhaustion_stage_is_claim_reference": (
            historical_results[-1]["dialogue_valid"] and
            not historical_results[-1]["claims_valid"]),
        "historical_v5_unchanged": before["tree_sha256"] == after["tree_sha256"],
        "authoritative_input_unchanged": input_before == sha256_file(INPUT),
        "legacy_import_isolation": audit_legacy_import_isolation(PACKAGE)["isolated"],
        "api_calls_zero": True,
    }
    return {
        "audit_version": AUDIT_VERSION,
        "historical_run": str(V5_RUN),
        "historical_v5_before": before,
        "historical_v5_after": after,
        "input_sha256_before": input_before,
        "input_sha256_after": sha256_file(INPUT),
        "scene_key": scene["scene_key"],
        "proposed_delta_id": delta["proposed_delta_id"],
        "historical_attempts": [{
            "attempt": index,
            "dialogue_valid": result["dialogue_valid"],
            "claims_valid": result["claims_valid"],
            "retry_mode": result["retry_mode"],
            "errors": result["errors"],
        } for index, result in enumerate(historical_results, 1)],
        "frozen_dialogue_turns": frozen_turns,
        "allowed_evidence_turn_keys": allowed_keys,
        "claim_only_prompt_character_count": len(retry_prompt),
        "invalid_historical_key_exposed_to_retry": invalid_key in retry_prompt,
        "realized_fixture": {
            "claim_validation": realized_validation,
            "full_scene_validation": full_validation,
        },
        "status_fixtures": status_fixtures,
        "committer_candidate_subset": {
            "outside_candidate_validation": outsider_validation,
            "decision_dry_runs": decision_results,
        },
        "checks": checks,
        "ready": all(checks.values()),
        "api_calls": 0,
    }


def _shadow_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Realizer Claim-only Retry V1 Shadow Audit", "",
        f"- Historical run: `{audit['historical_run']}`",
        f"- Scene / delta: `{audit['scene_key']}` / `{audit['proposed_delta_id']}`",
        f"- Allowed evidence keys: `{', '.join(audit['allowed_evidence_turn_keys'])}`",
        f"- Ready: `{str(audit['ready']).lower()}`",
        f"- API calls: `{audit['api_calls']}`", "",
        "## Historical attempts", "",
        "| Attempt | Dialogue | Claims | Retry mode |",
        "|---:|---|---|---|",
    ]
    for row in audit["historical_attempts"]:
        lines.append(
            f"| {row['attempt']} | {row['dialogue_valid']} | "
            f"{row['claims_valid']} | `{row['retry_mode']}` |")
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {name}: `{str(value).lower()}`"
                 for name, value in audit["checks"].items())
    lines.extend([
        "", "Passing the structural shadow gate is not a semantic confirmation. "
        "No historical response was upgraded, bound, committed, or rewritten.", "",
    ])
    return "\n".join(lines)


def build_report(audit: Mapping[str, Any], tests: Mapping[str, str]) -> dict[str, Any]:
    source_hashes = {
        name: sha256_file(PACKAGE / name) for name in (
            "prompts.py", "schemas.py", "realizer_claims.py", "live_fresh_e2e.py",
            "binder.py", "memory.py", "affordance_compiler.py", "structural.py")
    }
    all_tests_passed = all(value == "passed" for value in tests.values())
    return {
        "report_version": "world_memory_slim_realizer_claim_only_retry_v1_report_v1",
        "failure_root_cause": (
            "v5 scene_03 dialogue was structurally valid, but every Realizer claim "
            "cited a parent turn ID absent from the generated scene; generic full-scene "
            "retry echoed the rejected response three times."),
        "retry_boundary": {
            "full_scene_retry": [
                "unparseable JSON", "scene identity", "dialogue structure",
                "speaker/text/turn-key", "invalid delta identity", "overall claim schema"],
            "claim_only_retry": [
                "candidate evidence reference", "status/evidence cardinality",
                "duplicate planned claim"],
            "frozen_fields": ["scene_key", "dialogue_turns"],
            "total_physical_request_budget_unchanged": True,
        },
        "claim_evidence_rules": {
            "realized": "nonempty exact current-scene keys",
            "not_realized": "empty evidence list",
            "uncertain": "empty or exact current-scene keys",
            "automatic_key_repair": False,
        },
        "committer_boundary": {
            "may_confirm": "all or a subset of Realizer candidate evidence",
            "may_discover_other_scene_turns": False,
            "not_realized_may_confirm": False,
            "authority_gate": "confirmed + semantically_verified only",
        },
        "shadow_audit": audit,
        "tests": dict(tests),
        "api_calls": 0,
        "git_head": git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256_excluding_generated_reports":
            project_diff_sha256_excluding(PROJECT, GENERATED_RELATIVES),
        "source_sha256": source_hashes,
        "historical_v5_sha_before": audit["historical_v5_before"]["tree_sha256"],
        "historical_v5_sha_after": audit["historical_v5_after"]["tree_sha256"],
        "uncompleted_items": [
            "Fresh E2E v6 was not run.",
            "No live semantic confirmation was attempted.",
            "Round 2 and QA remain out of scope.",
        ],
        "realizer_claim_only_retry_ready": audit["ready"] and all_tests_passed,
        "realizer_claim_evidence_contract_ready": audit["ready"] and all_tests_passed,
        "committer_candidate_subset_rule_ready": audit["ready"] and all_tests_passed,
        "v5_shadow_replay_ready": audit["ready"] and all_tests_passed,
        "slim_pipeline_v6_preflight_ready": audit["ready"] and all_tests_passed,
        "generation_ready": False,
        "fresh_e2e_ready": False,
        "qa_pipeline_ready": False,
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    audit = report["shadow_audit"]
    return "\n".join([
        "# World Memory Slim Realizer Claim-only Retry V1", "",
        "## Fresh E2E v5 failure", "",
        str(report["failure_root_cause"]), "",
        "The dialogue gate passed; the claim/reference gate failed. Claim-only retry now "
        "freezes the scene key and dialogue turns, asks the model to resubmit only claims, "
        "and performs full validation again after deterministic reconstruction.", "",
        "## Contract", "",
        "- `realized`: nonempty exact current-scene evidence.",
        "- `not_realized`: empty evidence.",
        "- `uncertain`: empty or exact current-scene evidence.",
        "- No automatic replacement, deletion, nearest-key lookup, or fuzzy matching.",
        "- Evidence Committer may return only all/a subset of Realizer candidates.",
        "- Only `confirmed + semantically_verified` can update authoritative Memory.", "",
        "## Shadow replay and preflight", "",
        f"- Three v5 dialogues valid: `{audit['checks']['three_old_dialogues_valid']}`",
        f"- Three claims rejected and claim-only classified: "
        f"`{audit['checks']['three_classified_claim_only_retry']}`",
        f"- Frozen allowed keys: `{', '.join(audit['allowed_evidence_turn_keys'])}`",
        f"- Invalid historical key omitted from retry: "
        f"`{audit['checks']['invalid_key_not_in_prompt']}`",
        f"- Reconstructed full validation: "
        f"`{audit['checks']['reconstructed_full_scene_valid']}`",
        f"- Committer outside-candidate rejection: "
        f"`{audit['checks']['committer_outside_candidate_rejected']}`",
        f"- Confirmed-only authority dry-run: "
        f"`{audit['checks']['confirmed_only_authoritative']}`", "",
        "## Verification", "",
        *[f"- {name}: `{value}`" for name, value in report["tests"].items()],
        f"- API calls: `{report['api_calls']}`",
        f"- Git HEAD: `{report['git_head']}`",
        f"- Project diff SHA256 (reports excluded): "
        f"`{report['project_diff_sha256_excluding_generated_reports']}`",
        f"- Historical v5 SHA before/after: `{report['historical_v5_sha_before']}` / "
        f"`{report['historical_v5_sha_after']}`", "",
        "## Status", "",
        *[f"- {key}: `{str(report[key]).lower()}`" for key in (
            "realizer_claim_only_retry_ready",
            "realizer_claim_evidence_contract_ready",
            "committer_candidate_subset_rule_ready",
            "v5_shadow_replay_ready",
            "slim_pipeline_v6_preflight_ready",
            "generation_ready", "fresh_e2e_ready", "qa_pipeline_ready")], "",
        "Fresh E2E v6, live semantic confirmation, Round 2, and QA were not run.", "",
    ])


def write_reports(*, tests: Mapping[str, str]) -> dict[str, Any]:
    audit = build_shadow_audit()
    atomic_json(SHADOW_JSON, audit)
    atomic_text(SHADOW_MD, _shadow_markdown(audit))
    report = build_report(audit, tests)
    atomic_json(REPORT_JSON, report)
    atomic_text(REPORT_MD, _report_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted", default="not_run")
    parser.add_argument("--full", default="not_run")
    parser.add_argument("--compileall", default="not_run")
    parser.add_argument("--diff-check", default="not_run")
    args = parser.parse_args(argv)
    report = write_reports(tests={
        "targeted_unittest": args.targeted,
        "full_unittest": args.full,
        "compileall": args.compileall,
        "git_diff_check": args.diff_check,
    })
    print(json.dumps({key: report[key] for key in (
        "realizer_claim_only_retry_ready",
        "committer_candidate_subset_rule_ready",
        "v5_shadow_replay_ready",
        "slim_pipeline_v6_preflight_ready",
        "api_calls")}, indent=2))
    return 0 if report["v5_shadow_replay_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
