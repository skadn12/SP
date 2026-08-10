"""Zero-API preflight for the minimal Structure Miner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from .integrity import initial_git_status, protected_manifests, tree_manifest
from .position_inputs import (
    DEFAULT_EXPANDED, EXPECTED_EXPANDED_SHA256, load_object, repo_root,
)
from .structure_miner_contract import validate_structure_miner_output
from .structure_miner_prompts import build_structure_miner_prompt


PREFLIGHT_VERSION = "structure_miner_minimal_preflight_v1"
TARGET_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_structure_miner_probe_v1_live_1"
)
HISTORICAL_POSITION_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_position_"
    "distractor_provenance_probe_v2_live_1"
)
EXPECTED_HISTORICAL_POSITION_RUN_SHA = (
    "83fff872e8f23075bf7690bba185a87e7e6b7fd9a5105091bfb4b5f66552d7d6"
)
EXPECTED_POSITION_FILE_SHA256 = {
    "position_contract.py": "59eeec4ea22b20eba1b3c0ce6879ee0ffe452bb505f2e0fc975a919c1010aeae",
    "position_prompts.py": "6cb3764f0fb5ad88571ff9768489fa86dc22cedb9f2b07df46fe7fcd9457aa19",
    "position_writer_live.py": "460fa9d9777f7a11b58886a1f9c523189dbddf3259238247f303a98b56fd1af8",
    "position_preflight.py": "97ffad8a06d581bf85b43e31cf7474ee52ed83a6cf29918cfc739caf05083a80",
    "test_position_contract.py": "63f102bcec298072aa4b89929379fc34c65db6f09456d6a1b5310883895a7f37",
}


def _candidate(positions=None, structure_type="revision", variable="old vs revised state"):
    return {
        "structure_type": structure_type,
        "evidence_positions": list(positions or [1, 2]),
        "discriminative_variable": variable,
    }


def frozen_integrity() -> dict[str, Any]:
    module = Path(__file__).resolve().parent
    position_files = {
        name: sha256_file(module / name) for name in EXPECTED_POSITION_FILE_SHA256
    }
    return {
        "expanded_dialogue_sha256": sha256_file(DEFAULT_EXPANDED),
        "expanded_dialogue_unchanged": (
            sha256_file(DEFAULT_EXPANDED) == EXPECTED_EXPANDED_SHA256
        ),
        "protected_sources": protected_manifests(repo_root()),
        "position_file_sha256": position_files,
        "position_files_unchanged": position_files == EXPECTED_POSITION_FILE_SHA256,
        "historical_position_run_sha256": tree_manifest(
            HISTORICAL_POSITION_RUN
        )["tree_sha256"],
        "historical_position_run_unchanged": (
            HISTORICAL_POSITION_RUN.is_dir() and
            tree_manifest(HISTORICAL_POSITION_RUN)["tree_sha256"] ==
            EXPECTED_HISTORICAL_POSITION_RUN_SHA
        ),
        "target_run_absent": not TARGET_RUN_ROOT.exists(),
    }


def run_preflight() -> dict[str, Any]:
    repo = repo_root()
    git_before = initial_git_status(repo)
    integrity_before = frozen_integrity()
    expanded = load_object(DEFAULT_EXPANDED)
    prompt = build_structure_miner_prompt(expanded)

    def validate(structures):
        return validate_structure_miner_output(
            {"structures": structures}, expanded_dialogue=expanded
        )

    one = validate([_candidate([1, 2])])
    six = validate([_candidate([1, 2, 3, 4, 5, 6])])
    one_position = validate([_candidate([1])])
    seven_positions = validate([_candidate([1, 2, 3, 4, 5, 6, 7])])
    duplicate = validate([_candidate([1, 1])])
    out_of_range = validate([_candidate([1, 111])])
    unknown = validate([_candidate(structure_type="ordinary_recall")])
    empty_variable = validate([_candidate(variable="")])
    over_capacity = validate([_candidate() for _ in range(11)])
    checks = {
        "api_calls_zero": True,
        "solver_calls_zero": True,
        "expansion_calls_zero": True,
        "memorax_calls_zero": True,
        "frozen_dialogue_unchanged": integrity_before["expanded_dialogue_unchanged"],
        "protected_sources_unchanged": integrity_before["protected_sources"]["unchanged"],
        "position_first_writer_unchanged": integrity_before["position_files_unchanged"],
        "historical_position_run_unchanged": integrity_before["historical_position_run_unchanged"],
        "target_run_absent": integrity_before["target_run_absent"],
        "prompt_uses_full_dialogue": prompt["dialogue_turn_count"] == 110,
        "prompt_has_only_miner_output_fields": prompt["output_fields"] == [
            "structure_type", "evidence_positions", "discriminative_variable",
        ],
        "prompt_does_not_leak_manual_counts": all(
            phrase not in prompt["prompt"]
            for phrase in (
                "temporal competition        4",
                "revision                    3",
                "about 8 independent QA seeds",
            )
        ),
        "two_positions_valid": one["mechanically_valid_count"] == 1,
        "six_positions_valid": six["mechanically_valid_count"] == 1,
        "one_position_rejected": "evidence_below_minimum" in one_position["items"][0]["mechanical_errors"],
        "seven_positions_rejected": "evidence_above_maximum" in seven_positions["items"][0]["mechanical_errors"],
        "duplicate_position_rejected": "evidence_duplicate_positions" in duplicate["items"][0]["mechanical_errors"],
        "out_of_range_rejected": "evidence_position_out_of_range" in out_of_range["items"][0]["mechanical_errors"],
        "unknown_type_rejected": "structure_type_not_allowed" in unknown["items"][0]["mechanical_errors"],
        "empty_variable_rejected": "discriminative_variable_empty" in empty_variable["items"][0]["mechanical_errors"],
        "empty_structures_valid": validate([])["empty_structures_valid"],
        "zero_one_many_valid": (
            validate([])["mechanically_valid_count"] == 0 and
            validate([_candidate()])["mechanically_valid_count"] == 1 and
            validate([_candidate() for _ in range(4)])["mechanically_valid_count"] == 4
        ),
        "over_capacity_rejected": (
            "structure_capacity_exceeded" in over_capacity["envelope_errors"] and
            over_capacity["mechanically_valid_count"] == 0
        ),
        "position_maps_canonical_id": one["valid_structures"][0]["canonical_turn_ids"] == [
            "cc-train-episode-146700:P01:T001",
            "cc-train-episode-146700:P01:T002",
        ],
    }
    integrity_after = frozen_integrity()
    git_after = initial_git_status(repo)
    checks["integrity_stable_during_preflight"] = integrity_before == integrity_after
    checks["git_status_stable_during_preflight"] = git_before == git_after
    failed = sorted(key for key, passed in checks.items() if not passed)
    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "prompt": {
            "version": prompt["prompt_version"],
            "sha256": prompt["prompt_sha256"],
            "character_count": prompt["character_count"],
            "token_estimate": prompt["token_estimate"],
            "dialogue_turn_count": prompt["dialogue_turn_count"],
        },
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "execution": {
            "api_calls": 0,
            "solver_calls": 0,
            "expansion_calls": 0,
            "memorax_calls": 0,
            "r2_r3": False,
            "training": False,
        },
    }
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "STRUCTURE_MINER_PREFLIGHT.json", report)
    atomic_text(reports / "STRUCTURE_MINER_PREFLIGHT.md", "\n".join([
        "# Minimal Structure Miner Preflight",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Failed: `{json.dumps(failed)}`",
        f"- Prompt tokens (estimate): `{prompt['token_estimate']}`",
        "- API / Solver / Expansion / Memorax calls: `0 / 0 / 0 / 0`",
        "",
    ]))
    return report


def main() -> int:
    report = run_preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
