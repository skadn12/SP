"""Zero-API preflight for per-natural-window structure recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from .integrity import initial_git_status, protected_manifests, tree_manifest
from .local_structure_recognizer_contract import validate_local_recognizer_output
from .local_structure_recognizer_prompts import build_local_recognizer_prompt
from .local_structure_windows import compile_natural_windows
from .position_inputs import (
    DEFAULT_EXPANDED, EXPECTED_EXPANDED_SHA256, load_object, repo_root,
)


PREFLIGHT_VERSION = "local_structure_recognizer_preflight_v1"
TARGET_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_local_structure_"
    "recognizer_probe_v1_live_1"
)
HISTORICAL_FULL_MINER_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_structure_miner_probe_v1_live_1"
)
EXPECTED_HISTORICAL_FULL_MINER_RUN_SHA = (
    "195d520732b2f39cd9c8e877b0e481903de707da8156778dd5d6eb50a48b0b3f"
)
EXPECTED_FROZEN_FILE_SHA256 = {
    "position_contract.py": "59eeec4ea22b20eba1b3c0ce6879ee0ffe452bb505f2e0fc975a919c1010aeae",
    "position_prompts.py": "6cb3764f0fb5ad88571ff9768489fa86dc22cedb9f2b07df46fe7fcd9457aa19",
    "position_writer_live.py": "460fa9d9777f7a11b58886a1f9c523189dbddf3259238247f303a98b56fd1af8",
    "structure_miner_contract.py": "1d8b8e2d591219ccb38d6abb9100cc3ba01c67119ab5ec2304c2188c377e064f",
    "structure_miner_prompts.py": "48869fc65652dd488f4d30cf90fa1a4a1a7a884ca65fce399d1c273014eeca21",
    "structure_miner_live.py": "9c4cdcfafca82e2fae8e41bd9a4b9c3fe118dcb77ba9ed52db1cf2c735da742c",
    "structure_miner_preflight.py": "98e1db75b1ab594706102ef4f9760327e6bdee9a397a933b071652c96268ea42",
    "test_structure_miner.py": "fc022fbad65081025acd34c0e176aafca7f1769b1be52ddb97688be04562acd2",
}


def _candidate(
    positions: list[int], structure_type: str = "revision",
    variable: str = "old state versus revised state",
) -> dict[str, Any]:
    return {
        "structure_type": structure_type,
        "evidence_positions": list(positions),
        "discriminative_variable": variable,
    }


def frozen_integrity() -> dict[str, Any]:
    module = Path(__file__).resolve().parent
    frozen_files = {
        name: sha256_file(module / name) for name in EXPECTED_FROZEN_FILE_SHA256
    }
    historical = (
        tree_manifest(HISTORICAL_FULL_MINER_RUN)["tree_sha256"]
        if HISTORICAL_FULL_MINER_RUN.is_dir() else None
    )
    return {
        "expanded_dialogue_sha256": sha256_file(DEFAULT_EXPANDED),
        "expanded_dialogue_unchanged": (
            sha256_file(DEFAULT_EXPANDED) == EXPECTED_EXPANDED_SHA256
        ),
        "protected_sources": protected_manifests(repo_root()),
        "frozen_file_sha256": frozen_files,
        "frozen_files_unchanged": frozen_files == EXPECTED_FROZEN_FILE_SHA256,
        "historical_full_miner_run_sha256": historical,
        "historical_full_miner_run_unchanged": (
            historical == EXPECTED_HISTORICAL_FULL_MINER_RUN_SHA
        ),
        "target_run_absent": not TARGET_RUN_ROOT.exists(),
    }


def run_preflight() -> dict[str, Any]:
    repo = repo_root()
    git_before = initial_git_status(repo)
    integrity_before = frozen_integrity()
    expanded = load_object(DEFAULT_EXPANDED)
    compiled = compile_natural_windows(expanded)
    windows = compiled["windows"]
    first = windows[0]
    second = windows[1]
    allowed = first["global_positions"]

    def validate(structures: list[dict[str, Any]]) -> dict[str, Any]:
        return validate_local_recognizer_output(
            {"structures": structures}, expanded_dialogue=expanded, window=first,
        )

    two = validate([_candidate(allowed[:2])])
    six = validate([_candidate(allowed[:6])])
    one = validate([_candidate(allowed[:1])])
    seven = validate([_candidate(allowed[:7])])
    duplicate = validate([_candidate([allowed[0], allowed[0]])])
    outside = validate([_candidate([allowed[0], second["global_positions"][0]])])
    out_of_range = validate([_candidate([allowed[0], 111])])
    unknown = validate([_candidate(allowed[:2], "ordinary_recall")])
    empty_variable = validate([_candidate(allowed[:2], variable="")])
    over_capacity = validate([_candidate(allowed[:2]) for _ in range(6)])
    prompt = build_local_recognizer_prompt(first)
    all_positions = [
        position for window in windows for position in window["global_positions"]
    ]
    checks = {
        "api_calls_zero": True,
        "solver_calls_zero": True,
        "writer_calls_zero": True,
        "full_miner_calls_zero": True,
        "expansion_calls_zero": True,
        "memorax_calls_zero": True,
        "frozen_dialogue_unchanged": integrity_before["expanded_dialogue_unchanged"],
        "protected_sources_unchanged": integrity_before["protected_sources"]["unchanged"],
        "position_and_full_miner_files_unchanged": integrity_before["frozen_files_unchanged"],
        "historical_full_miner_run_unchanged": integrity_before["historical_full_miner_run_unchanged"],
        "target_run_absent": integrity_before["target_run_absent"],
        "window_compiler_valid": compiled["valid"],
        "five_parent_windows": compiled["parent_session_count"] == 5,
        "three_child_windows": compiled["child_scene_count"] == 3,
        "eight_natural_windows": compiled["window_count"] == 8,
        "all_110_turns_covered_once": (
            sorted(all_positions) == list(range(1, 111)) and
            len(all_positions) == len(set(all_positions))
        ),
        "two_positions_valid": two["mechanically_valid_count"] == 1,
        "six_positions_valid": six["mechanically_valid_count"] == 1,
        "one_position_rejected": "evidence_below_minimum" in one["items"][0]["mechanical_errors"],
        "seven_positions_rejected": "evidence_above_maximum" in seven["items"][0]["mechanical_errors"],
        "duplicate_position_rejected": "evidence_duplicate_positions" in duplicate["items"][0]["mechanical_errors"],
        "outside_window_rejected": "evidence_position_outside_window" in outside["items"][0]["mechanical_errors"],
        "out_of_range_rejected": "evidence_position_out_of_range" in out_of_range["items"][0]["mechanical_errors"],
        "unknown_type_rejected": "structure_type_not_allowed" in unknown["items"][0]["mechanical_errors"],
        "empty_variable_rejected": "discriminative_variable_empty" in empty_variable["items"][0]["mechanical_errors"],
        "empty_structures_valid": validate([])["empty_structures_valid"],
        "zero_one_many_valid": (
            validate([])["mechanically_valid_count"] == 0 and
            validate([_candidate(allowed[:2])])["mechanically_valid_count"] == 1 and
            validate([_candidate(allowed[:2]) for _ in range(3)])["mechanically_valid_count"] == 3
        ),
        "per_window_hard_max_rejected": (
            "per_window_capacity_exceeded" in over_capacity["envelope_errors"] and
            over_capacity["mechanically_valid_count"] == 0
        ),
        "global_position_maps_canonical_id": two["valid_structures"][0]["canonical_turn_ids"] == [
            "cc-train-episode-146700:P01:T001",
            "cc-train-episode-146700:P01:T002",
        ],
        "prompt_contains_only_current_window": (
            prompt["allowed_global_positions"] == allowed and
            f"{allowed[0]} |" in prompt["prompt"] and
            f"{second['global_positions'][0]} |" not in prompt["prompt"]
        ),
        "prompt_does_not_leak_human_audit": all(
            phrase not in prompt["prompt"]
            for phrase in (
                "temporal competition        4", "revision                    3",
                "8 independent QA seeds", "SOURCE_STRUCTURE_AUDIT",
            )
        ),
    }
    integrity_after = frozen_integrity()
    git_after = initial_git_status(repo)
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
        "prompt_fixture": {
            key: prompt[key] for key in (
                "prompt_version", "prompt_sha256", "character_count",
                "token_estimate", "window_turn_count", "allowed_global_positions",
            )
        },
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "execution": {
            "api_calls": 0, "writer_calls": 0, "solver_calls": 0,
            "full_miner_calls": 0, "expansion_calls": 0, "memorax_calls": 0,
            "r2_r3": False, "training": False,
        },
    }
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "LOCAL_STRUCTURE_RECOGNIZER_PREFLIGHT.json", report)
    atomic_text(reports / "LOCAL_STRUCTURE_RECOGNIZER_PREFLIGHT.md", "\n".join([
        "# Local Structure Recognizer Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Natural windows: `{compiled['window_count']}` "
        f"(`{compiled['parent_session_count']}` Parent + "
        f"`{compiled['child_scene_count']}` Child)",
        f"- Covered turns: `{compiled['assigned_turn_count']}`",
        f"- Failed checks: `{json.dumps(failed)}`",
        "- API / Writer / Solver / full Miner / Expansion / Memorax calls: `0`",
        "",
    ]))
    return report


def main() -> int:
    report = run_preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
