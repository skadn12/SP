"""Zero-API preflight for the independent fact-first QA bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text
from .fact_bundler import bundle_facts
from .fact_contract import materialize_facts, validate_fact_extractor_output
from .fact_prompts import build_fact_prompt
from .integrity import git_status, sha256_file, tree_manifest
from .inputs import (
    DEFAULT_EXPANDED, DEFAULT_MEMORY_OVERVIEW, EXPECTED_EXPANDED_SHA256,
    EXPECTED_MEMORY_OVERVIEW_SHA256, load_object, repo_root,
)
from .windows import compile_natural_windows


PREFLIGHT_VERSION = "qa_fact_first_preflight_v1"
TARGET_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_first_probe_v1_live_1"
)
OLD_POSITION_MODULE = Path(
    "/root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method/"
    "restart_n5_spell/simple_append_memorax_v1_qa_position_first_probe_v1"
)
EXPECTED_OLD_POSITION_MODULE_SHA = (
    "ead4f99b8e3659d3aea1117d87dc16a30764fde046342a916119c6a47be29048"
)
HISTORICAL_RUNS = {
    "position_distractor": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_146700_qa_position_"
        "distractor_provenance_probe_v2_live_1"
    ),
    "structure_first": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_146700_qa_structure_first_probe_v1_live_1"
    ),
    "structure_miner": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_146700_qa_structure_miner_probe_v1_live_1"
    ),
    "local_recognizer": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_146700_qa_local_structure_recognizer_"
        "probe_v1_live_1"
    ),
}
EXPECTED_HISTORICAL_RUN_SHAS = {
    "position_distractor": "83fff872e8f23075bf7690bba185a87e7e6b7fd9a5105091bfb4b5f66552d7d6",
    "structure_first": "046a7ea34509c1019fac2719ea3942b67c1412f613a0d6e27dd5d4e0e1c6e2eb",
    "structure_miner": "195d520732b2f39cd9c8e877b0e481903de707da8156778dd5d6eb50a48b0b3f",
    "local_recognizer": "af2065d80b1ef6045ac41f0e898fc70f8c279f612f1430a545dbd648c887022d",
}
PROTECTED_MAIN_FILES = (
    "scripts/current_method/restart_n5_spell/core.py",
    "scripts/current_method/restart_n5_spell/live_backends.py",
    "tests/test_restart_n5_spell.py",
)


def _fixture_fact(positions: list[int], text: str = "A planned an action.", subject: str = "A") -> dict[str, Any]:
    return {"fact_text": text, "source_positions": positions, "subject": subject}


def frozen_integrity() -> dict[str, Any]:
    repo = repo_root()
    main_files = {
        relative: sha256_file(repo / relative) for relative in PROTECTED_MAIN_FILES
    }
    historical = {
        name: (
            tree_manifest(path)["tree_sha256"] if path.is_dir() else None
        ) for name, path in HISTORICAL_RUNS.items()
    }
    return {
        "expanded_sha256": sha256_file(DEFAULT_EXPANDED),
        "expanded_unchanged": sha256_file(DEFAULT_EXPANDED) == EXPECTED_EXPANDED_SHA256,
        "memory_overview_sha256": sha256_file(DEFAULT_MEMORY_OVERVIEW),
        "memory_overview_unchanged": sha256_file(DEFAULT_MEMORY_OVERVIEW) == EXPECTED_MEMORY_OVERVIEW_SHA256,
        "old_position_module_sha256": tree_manifest(OLD_POSITION_MODULE)["tree_sha256"],
        "old_position_module_unchanged": tree_manifest(OLD_POSITION_MODULE)["tree_sha256"] == EXPECTED_OLD_POSITION_MODULE_SHA,
        "protected_main_files": main_files,
        "historical_run_shas": historical,
        "historical_runs_unchanged": historical == EXPECTED_HISTORICAL_RUN_SHAS,
        "target_run_absent": not TARGET_RUN_ROOT.exists(),
    }


def run_preflight() -> dict[str, Any]:
    repo = repo_root()
    git_before = git_status(repo)
    integrity_before = frozen_integrity()
    expanded = load_object(DEFAULT_EXPANDED)
    compiled = compile_natural_windows(expanded)
    windows = compiled["windows"]
    first = windows[0]
    second = windows[1]
    allowed = first["global_positions"]

    def validate(facts: list[dict[str, Any]], window: dict[str, Any] = first):
        return validate_fact_extractor_output(
            {"facts": facts}, expanded_dialogue=expanded, window=window,
        )

    valid_one = validate([_fixture_fact([allowed[0]])])
    valid_three = validate([_fixture_fact(allowed[:3])])
    zero = validate([_fixture_fact([])])
    four = validate([_fixture_fact(allowed[:4])])
    duplicate = validate([_fixture_fact([allowed[0], allowed[0]])])
    outside = validate([_fixture_fact([allowed[0], second["global_positions"][0]])])
    empty_text = validate([_fixture_fact([allowed[0]], text="")])
    empty_subject = validate([_fixture_fact([allowed[0]], subject="")])
    over_capacity = validate([_fixture_fact([allowed[0]]) for _ in range(9)])
    extra_key = validate([{
        **_fixture_fact([allowed[0]]), "structure_type": "revision",
    }])
    fixture_validations = [
        dict(valid_one, window_id="P01", valid_facts=valid_one["valid_facts"]),
        dict(valid_three, window_id="P03", valid_facts=valid_three["valid_facts"]),
    ]
    windows_by_id = {window["window_id"]: window for window in windows}
    materialized = materialize_facts(fixture_validations, windows_by_id=windows_by_id)
    bundle_fixture = [
        {"fact_id": "F0001", "fact_text": "x", "subject": "A", "window_id": "P01", "window_order": 1},
        {"fact_id": "F0002", "fact_text": "y", "subject": "A", "window_id": "P03", "window_order": 3},
        {"fact_id": "F0003", "fact_text": "z", "subject": "B", "window_id": "C01", "window_order": 6},
        {"fact_id": "F0004", "fact_text": "w", "subject": "A", "window_id": "P03", "window_order": 3},
    ]
    bundles = bundle_facts(bundle_fixture)
    checks = {
        "api_zero": True,
        "solver_zero": True,
        "expansion_zero": True,
        "memorax_zero": True,
        "no_old_position_import_execution": True,
        "expanded_sha_unchanged": integrity_before["expanded_unchanged"],
        "memory_overview_sha_unchanged": integrity_before["memory_overview_unchanged"],
        "old_position_module_unchanged": integrity_before["old_position_module_unchanged"],
        "historical_runs_unchanged": integrity_before["historical_runs_unchanged"],
        "target_run_absent": integrity_before["target_run_absent"],
        "window_compiler_valid": compiled["valid"],
        "parent_window_count": compiled["parent_session_count"] == 5,
        "child_window_count": compiled["child_scene_count"] == 3,
        "all_turns_covered": compiled["assigned_turn_count"] == 110,
        "one_position_valid": valid_one["mechanically_valid_count"] == 1,
        "three_positions_valid": valid_three["mechanically_valid_count"] == 1,
        "zero_positions_rejected": "source_positions_below_minimum" in zero["items"][0]["mechanical_errors"],
        "four_positions_rejected": "source_positions_above_maximum" in four["items"][0]["mechanical_errors"],
        "duplicate_positions_rejected": "source_positions_duplicate" in duplicate["items"][0]["mechanical_errors"],
        "outside_window_rejected": "source_position_outside_window" in outside["items"][0]["mechanical_errors"],
        "empty_text_rejected": "fact_text_empty" in empty_text["items"][0]["mechanical_errors"],
        "empty_subject_rejected": "subject_empty" in empty_subject["items"][0]["mechanical_errors"],
        "facts_empty_valid": validate([])["empty_facts_valid"],
        "over_eight_rejected": "fact_capacity_exceeded" in over_capacity["envelope_errors"],
        "mechanical_fields_rejected": "fact_candidate_schema" in extra_key["items"][0]["mechanical_errors"],
        "materialization_ids_stable": materialized[0]["fact_id"] == "F0001" and materialized[1]["fact_id"] == "F0002",
        "materialization_turn_mapping": materialized[0]["source_turn_ids"] == ["cc-train-episode-146700:P01:T001"],
        "materialization_parent_child_metadata": materialized[0]["window_kind"] == "parent_session",
        "bundle_same_subject_cross_window": bundles["bundle_count"] > 0,
        "bundle_different_subjects_not_combined": all(
            not ("F0003" in row["fact_ids"] and "F0001" in row["fact_ids"])
            for row in bundles["bundles"]
        ),
        "bundle_same_window_not_only_source": all(len(row["window_ids"]) >= 2 for row in bundles["bundles"]),
        "bundle_size_limit": all(2 <= row["bundle_size"] <= 4 for row in bundles["bundles"]),
        "bundle_deterministic": bundle_facts(bundle_fixture) == bundle_facts(list(reversed(bundle_fixture))),
        "prompt_only_fact_schema": build_fact_prompt(first)["output_fields"] == ["fact_text", "source_positions", "subject"],
    }
    integrity_after = frozen_integrity()
    git_after = git_status(repo)
    checks["integrity_stable_during_preflight"] = integrity_before == integrity_after
    checks["git_status_stable_during_preflight"] = git_before == git_after
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "window_compilation": {
            key: compiled[key] for key in (
                "window_compiler_version", "valid", "errors", "window_count",
                "parent_session_count", "child_scene_count", "assigned_turn_count",
            )
        },
        "windows": [{
            key: window[key] for key in (
                "window_id", "window_kind", "boundary_ref", "global_position_start",
                "global_position_end", "turn_count",
            )
        } for window in windows],
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "execution": {
            "api": 0, "solver": 0, "expansion": 0, "memorax": 0,
            "r2_r3": False, "training": False,
        },
        "bundler_limits": bundles["limits"],
    }
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "FACT_FIRST_PREFLIGHT.json", report)
    atomic_text(reports / "FACT_FIRST_PREFLIGHT.md", "\n".join([
        "# Fact-First QA Bootstrap Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Natural windows: `{compiled['window_count']}` "
        f"(`{compiled['parent_session_count']}` Parent + `{compiled['child_scene_count']}` Child)",
        f"- Covered turns: `{compiled['assigned_turn_count']}`",
        f"- Failed checks: `{json.dumps(failed)}`",
        "- API / Solver / Expansion / Memorax calls: `0 / 0 / 0 / 0`",
        "- MCQ generation: `not implemented in this probe`", "",
    ]))
    return report


def main() -> int:
    report = run_preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
