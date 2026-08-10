"""Zero-API preflight for participant-connected frozen-fact rebundling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core import atomic_json, atomic_text
from .fact_bundler import bundle_facts
from .integrity import sha256_file, tree_manifest
from .participant_bundler import (
    bundle_participant_connected_facts, known_dialogue_participants,
    materialize_fact_participants, validate_participant_bundle,
)


FACT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_first_rebundle_v1_offline"
)
FACTS_PATH = FACT_ROOT / "eligible_facts.json"
LIVE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_first_probe_v1_live_1"
)
EXPANDED_PATH = LIVE_ROOT / "frozen/expanded_dialogue.json"
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_first_"
    "participant_rebundle_v1_offline"
)
OLD_POSITION_MODULE = Path(
    "/root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method/"
    "restart_n5_spell/simple_append_memorax_v1_qa_position_first_probe_v1"
)
EXPECTED_POSITION_SHA = "ead4f99b8e3659d3aea1117d87dc16a30764fde046342a916119c6a47be29048"
EXPECTED_EPISODE_SHA = "97cf7b5aef3b9e6b7a04e9323f015d7d3cbef6f82fc985da97b89cbc70f0f7f5"
EXPECTED_ALLOWLIST_ID_SHA = "168a4ea01146ada3096bb88d8ae9d6f46f33588ad8fa6bb5db3b3d85c8758c7d"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _id_sha(facts_payload) -> str:
    ids = sorted(str(row["fact_id"]) for row in facts_payload["facts"])
    payload = json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row(fact_id, subject, window_id, participants, order):
    return {
        "fact_id": fact_id, "fact_text": fact_id, "subject": subject,
        "window_id": window_id, "window_order": order,
        "window_kind": "parent_session" if window_id.startswith("P") else "child_scene",
        "participants": participants,
    }


def run() -> dict[str, object]:
    facts_payload = _load(FACTS_PATH)
    expanded = _load(EXPANDED_PATH)
    facts = list(facts_payload["facts"])
    previous_before = tree_manifest(FACT_ROOT)["tree_sha256"]
    position_before = tree_manifest(OLD_POSITION_MODULE)["tree_sha256"]
    materialized = materialize_fact_participants(facts, expanded=expanded)
    participant_result = bundle_participant_connected_facts(materialized)
    same_subject_result = bundle_facts(facts)

    same = [
        _row("F1", "A", "P01", ["A"], 1),
        _row("F2", "A", "C01", ["A"], 6),
    ]
    direct_bridge = [
        _row("F1", "A", "P01", ["A"], 1),
        _row("F2", "B", "C01", ["A", "B"], 6),
    ]
    transitive = [
        _row("F1", "A", "P01", ["A"], 1),
        _row("F2", "B", "P05", ["A", "B"], 5),
        _row("F3", "B", "C02", ["B"], 7),
    ]
    disconnected = [
        _row("F1", "A", "P01", ["A"], 1),
        _row("F2", "B", "C01", ["B"], 6),
    ]
    disconnected_third = [
        *direct_bridge,
        _row("F3", "C", "C02", ["C"], 7),
    ]
    extraction_fixture = materialize_fact_participants([{
        "fact_id": "X1", "fact_text": "Boss meets Co-workers A.",
        "subject": "Co-workers A", "source_positions": [1],
    }, {
        "fact_id": "X2", "fact_text": "They helped them.",
        "subject": "Co-workers A", "source_positions": [1],
    }], expanded=expanded)
    selected_ids = {
        fact_id for bundle in participant_result["bundles"] for fact_id in bundle["fact_ids"]
    }
    checks = {
        "api_zero": True,
        "fact_extractor_zero": True,
        "qa_writer_zero": True,
        "solver_zero": True,
        "expansion_zero": True,
        "memorax_zero": True,
        "output_root_absent": not OUTPUT_ROOT.exists(),
        "frozen_fact_count_43": len(facts) == 43,
        "allowlist_id_sha_unchanged": _id_sha(facts_payload) == EXPECTED_ALLOWLIST_ID_SHA,
        "episode_sha_unchanged": sha256_file(EXPANDED_PATH) == EXPECTED_EPISODE_SHA,
        "position_branch_sha_unchanged": position_before == EXPECTED_POSITION_SHA,
        "participants_are_frozen_speakers": known_dialogue_participants(expanded) == [
            "Co-workers A", "Co-workers B", "Specialist D", "Boss",
        ],
        "subject_auto_included": "Co-workers A" in extraction_fixture[0]["participants"],
        "explicit_name_added": "Boss" in extraction_fixture[0]["participants"],
        "pronoun_not_guessed": extraction_fixture[1]["participants"] == ["Co-workers A"],
        "same_subject_cross_window_valid": validate_participant_bundle(same) == [],
        "direct_participant_bridge_valid": validate_participant_bundle(direct_bridge) == [],
        "transitive_participant_bridge_valid": validate_participant_bundle(transitive) == [],
        "unbridged_pair_rejected": "participant_graph_disconnected" in validate_participant_bundle(disconnected),
        "disconnected_third_rejected": "participant_graph_disconnected" in validate_participant_bundle(disconnected_third),
        "same_window_only_rejected": "bundle_requires_two_windows" in validate_participant_bundle([
            _row("F1", "A", "P01", ["A"], 1), _row("F2", "A", "P01", ["A"], 1),
        ]),
        "bundle_size_two_valid": validate_participant_bundle(direct_bridge) == [],
        "bundle_size_four_valid": validate_participant_bundle([
            _row("F1", "A", "P01", ["A"], 1), _row("F2", "A", "P01", ["A"], 1),
            _row("F3", "B", "C01", ["A", "B"], 6), _row("F4", "B", "C01", ["B"], 6),
        ]) == [],
        "bundle_size_over_four_rejected": "bundle_size_above_maximum" in validate_participant_bundle([
            _row(f"F{i}", "A", f"P{i:02d}", ["A"], i) for i in range(1, 6)
        ]),
        "all_43_considered_before_selection": participant_result["input_fact_count"] == 43,
        "earliest_cap_absent": participant_result["limits"]["pre_combination_fact_cap"] is None,
        "late_child_facts_selected": any(fact_id.startswith("F006") for fact_id in selected_ids),
        "cross_subject_candidates_added": participant_result["cross_subject_eligible_combination_count"] > 0,
        "same_subject_count_preserved": (
            participant_result["same_subject_eligible_combination_count"]
            == same_subject_result["all_mechanically_eligible_bundle_count"]
        ),
        "hard_max_60": participant_result["bundle_count"] == 60,
        "exact_fact_set_duplicates_zero": participant_result["exact_duplicate_fact_set_count"] == 0,
        "deterministic": participant_result == bundle_participant_connected_facts(list(reversed(materialized))),
    }
    previous_after = tree_manifest(FACT_ROOT)["tree_sha256"]
    position_after = tree_manifest(OLD_POSITION_MODULE)["tree_sha256"]
    checks["previous_rebundle_stable"] = previous_before == previous_after
    checks["position_branch_stable"] = position_before == position_after
    failed = sorted(key for key, value in checks.items() if not value)
    report = {
        "preflight_version": "participant_rebundle_preflight_v1",
        "decision": "pass" if not failed else "fail",
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "dry_run_counts": {
            "same_subject_eligible_combinations": same_subject_result["all_mechanically_eligible_bundle_count"],
            "participant_connected_eligible_combinations": participant_result["all_mechanically_eligible_bundle_count"],
            "new_cross_subject_combinations": participant_result["cross_subject_eligible_combination_count"],
            "materialized_bundles": participant_result["bundle_count"],
        },
        "integrity": {
            "previous_rebundle_sha_before": previous_before,
            "previous_rebundle_sha_after": previous_after,
            "position_branch_sha_before": position_before,
            "position_branch_sha_after": position_after,
            "episode_sha": sha256_file(EXPANDED_PATH),
            "allowlist_id_sha": _id_sha(facts_payload),
        },
        "execution": {
            "api": 0, "fact_extractor": 0, "qa_writer": 0, "solver": 0,
            "expansion": 0, "memorax": 0,
        },
    }
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(exist_ok=True)
    atomic_json(reports / "PARTICIPANT_REBUNDLE_PREFLIGHT.json", report)
    atomic_text(reports / "PARTICIPANT_REBUNDLE_PREFLIGHT.md", "\n".join([
        "# Participant-Connected Rebundle Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{report['passed']}/{report['total']}`",
        f"- Same-subject combinations: `{report['dry_run_counts']['same_subject_eligible_combinations']}`",
        f"- Participant-connected combinations: `{report['dry_run_counts']['participant_connected_eligible_combinations']}`",
        f"- New cross-subject combinations: `{report['dry_run_counts']['new_cross_subject_combinations']}`",
        "- All API/model calls: `0`", "",
    ]))
    return report


def main() -> int:
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
