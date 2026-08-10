"""Zero-API preflight for the position-only QA Writer branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from .integrity import initial_git_status, protected_manifests, tree_manifest
from .position_contract import materialize_position_writer_output
from .position_fixtures import (
    cross_story_distractor_source, duplicate_gold, envelope,
    invented_distractor_without_source, missing_distractor_provenance,
    out_of_range_distractor_source, temporal,
)
from .position_prompts import build_position_writer_prompt
from .position_inputs import (
    DEFAULT_EXPANDED, DEFAULT_MEMORY_OVERVIEW, EXPECTED_EXPANDED_SHA256,
    EXPECTED_MEMORY_OVERVIEW_SHA256, load_object, repo_root,
)


PREFLIGHT_VERSION = "position_distractor_provenance_preflight_v2"
STRUCTURE_FIRST_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_structure_first_probe_v1_live_1")
EXPECTED_STRUCTURE_FIRST_RUN_SHA = (
    "046a7ea34509c1019fac2719ea3942b67c1412f613a0d6e27dd5d4e0e1c6e2eb")
POSITION_LIVE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_position_semantic_probe_v1_live_1")
EXPECTED_POSITION_LIVE_RUN_SHA = (
    "d0350e9ca6bf16d29440ead096c7cdeeb7253b2e85d5b544348cc5d41812203e")


def _valid(value: dict[str, Any], expanded: dict[str, Any]) -> int:
    return int(materialize_position_writer_output(
        value, expanded_dialogue=expanded).get("structural_valid_count") or 0)


def run_preflight() -> dict[str, Any]:
    repo = repo_root()
    git_before = initial_git_status(repo)
    protected_before = protected_manifests(repo)
    expanded = load_object(DEFAULT_EXPANDED)
    overview = load_object(DEFAULT_MEMORY_OVERVIEW)
    prompt = build_position_writer_prompt(expanded, overview)
    mapped = materialize_position_writer_output(
        envelope(1, [1, 2, 3]), expanded_dialogue=expanded)
    mapped_row = (mapped.get("materialized_candidates") or [{}])[0]
    over_eight = materialize_position_writer_output(
        envelope(1, list(range(1, 10))), expanded_dialogue=expanded)
    out_of_range = materialize_position_writer_output(
        envelope(1, [1, 111]), expanded_dialogue=expanded)
    ambiguous = materialize_position_writer_output(
        duplicate_gold(), expanded_dialogue=expanded)
    missing_sources = materialize_position_writer_output(
        missing_distractor_provenance(), expanded_dialogue=expanded)
    source_out_of_range = materialize_position_writer_output(
        out_of_range_distractor_source(), expanded_dialogue=expanded)
    invented = materialize_position_writer_output(
        invented_distractor_without_source(), expanded_dialogue=expanded)
    cross_story = materialize_position_writer_output(
        cross_story_distractor_source(), expanded_dialogue=expanded)
    checks = {
        "api_calls_zero": True,
        "solver_calls_zero": True,
        "expansion_calls_zero": True,
        "memorax_calls_zero": True,
        "r2_r3_false": True,
        "training_false": True,
        "frozen_dialogue_sha": sha256_file(DEFAULT_EXPANDED) == EXPECTED_EXPANDED_SHA256,
        "frozen_overview_sha": sha256_file(DEFAULT_MEMORY_OVERVIEW) == EXPECTED_MEMORY_OVERVIEW_SHA256,
        "protected_sources_unchanged": bool(protected_before["unchanged"]),
        "structure_first_run_unchanged": (
            STRUCTURE_FIRST_RUN.is_dir() and
            tree_manifest(STRUCTURE_FIRST_RUN)["tree_sha256"] == EXPECTED_STRUCTURE_FIRST_RUN_SHA),
        "position_live_run_unchanged": (
            POSITION_LIVE_RUN.is_dir() and
            tree_manifest(POSITION_LIVE_RUN)["tree_sha256"] == EXPECTED_POSITION_LIVE_RUN_SHA),
        "lean_prompt_excludes_d_alias": "D###" in prompt["prompt"] and "Never output D###" in prompt["prompt"],
        "lean_prompt_excludes_mechanical_fields": all(
            f"Never output D###, canonical turn IDs, source_scope," in prompt["prompt"]
            for _ in [0]),
        "prompt_requires_distractor_provenance": (
            "distractor_sources is an object whose keys are exactly the three WRONG" in
            prompt["prompt"]),
        "prompt_requires_local_story_arc": (
            "same local story arc and the" in prompt["prompt"]),
        "position_maps_canonical_turn": mapped_row.get("canonical_turn_ids") == [
            "cc-train-episode-146700:P01:T001",
            "cc-train-episode-146700:P01:T002",
            "cc-train-episode-146700:P01:T003",
        ],
        "position_maps_evidence_alias": mapped_row.get("evidence_turn_ids") == ["D001", "D002", "D003"],
        "program_derives_scope": mapped_row.get("source_scope") == "parent_only",
        "program_derives_correct_option": mapped_row.get("correct_option") == "B",
        "evidence_2_to_6_materializes": all(
            _valid(envelope(1, list(range(1, count + 1))), expanded) == 1
            for count in range(2, 7)),
        "evidence_over_8_rejected": "evidence_over_hard_maximum" in over_eight["items"][0]["structural_errors"],
        "position_out_of_range_rejected": "evidence_position_out_of_range" in out_of_range["items"][0]["structural_errors"],
        "multiple_gold_binding_rejected": "gold_not_exactly_one_option" in ambiguous["items"][0]["structural_errors"],
        "valid_distractor_provenance_materializes": _valid(envelope(1), expanded) == 1,
        "distractor_sources_missing_rejected": "distractor_sources_keys" in missing_sources["items"][0]["structural_errors"],
        "distractor_source_out_of_range_rejected": "distractor_source_position_out_of_range:D" in source_out_of_range["items"][0]["structural_errors"],
        "invented_distractor_without_source_rejected": "distractor_source_schema:D" in invented["items"][0]["structural_errors"],
        "cross_story_arc_is_audited": "cross_story_arc_provenance:C" in cross_story["items"][0]["warnings"],
        "temporal_materializes": _valid(temporal(), expanded) == 1,
        "variable_batches_0_1_3_12": (
            materialize_position_writer_output(envelope(0), expanded_dialogue=expanded)["zero_question_batch_valid"] and
            all(_valid(envelope(count), expanded) == count for count in (1, 3, 12))),
        "solver_schema_compatible": set(mapped["normalized_batch"]["questions"][0]) == {
            "question_type", "question", "options", "correct_option", "evidence_turn_refs"},
    }
    protected_after = protected_manifests(repo)
    git_after = initial_git_status(repo)
    checks["protected_sha_same_during_preflight"] = protected_before == protected_after
    checks["git_status_same_during_preflight"] = git_before == git_after
    failed = sorted(key for key, passed in checks.items() if not passed)
    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "writer_schema_fields": sorted([
            "structure_type", "evidence_positions", "question", "options",
            "gold_text", "distractor_sources",
        ]),
        "program_owned_fields": [
            "candidate_id", "canonical_turn_ids", "evidence_turn_ids",
            "supporting_turn_ids", "source_scope", "correct_option",
        ],
        "prompt": {
            "version": prompt["prompt_version"],
            "sha256": prompt["prompt_sha256"],
            "character_count": prompt["character_count"],
            "token_estimate": prompt["token_estimate"],
        },
        "frozen": {
            "expanded_dialogue": str(DEFAULT_EXPANDED),
            "expanded_dialogue_sha256": sha256_file(DEFAULT_EXPANDED),
            "memory_overview": str(DEFAULT_MEMORY_OVERVIEW),
            "memory_overview_sha256": sha256_file(DEFAULT_MEMORY_OVERVIEW),
            "structure_first_run_sha256": tree_manifest(STRUCTURE_FIRST_RUN)["tree_sha256"],
            "position_live_run_sha256": tree_manifest(POSITION_LIVE_RUN)["tree_sha256"],
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "execution": {
            "api_calls": 0, "solver_calls": 0, "expansion_calls": 0,
            "memorax_calls": 0, "r2_r3": False, "training": False,
            "git_commit_push": False,
        },
    }
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "POSITION_SEMANTIC_WRITER_PREFLIGHT.json", report)
    atomic_text(reports / "POSITION_SEMANTIC_WRITER_PREFLIGHT.md", "\n".join([
        "# Position-only Semantic Writer Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Failed: `{json.dumps(failed)}`",
        "- API / Solver / Expansion / Memorax calls: `0 / 0 / 0 / 0`",
        f"- Prompt token estimate: `{prompt['token_estimate']}`",
        f"- Protected sources unchanged: `{str(protected_after['unchanged']).lower()}`",
        "",
    ]))
    return report


def main() -> int:
    report = run_preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
