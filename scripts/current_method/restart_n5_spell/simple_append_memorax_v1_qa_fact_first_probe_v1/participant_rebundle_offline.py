"""Run the frozen participant-connected rebundle without API calls."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text
from .fact_bundler import bundle_facts
from .integrity import sha256_file, tree_manifest
from .participant_bundler import (
    PARTICIPANT_BUNDLER_VERSION, bundle_participant_connected_facts,
    known_dialogue_participants, materialize_fact_participants,
)
from .participant_rebundle_preflight import (
    EXPECTED_ALLOWLIST_ID_SHA, EXPECTED_EPISODE_SHA, EXPECTED_POSITION_SHA,
    EXPANDED_PATH, FACT_ROOT, FACTS_PATH, OLD_POSITION_MODULE, OUTPUT_ROOT,
    _id_sha, _load,
)


RUNNER_VERSION = "participant_connected_frozen_rebundle_offline_v1"
MODULE_ROOT = Path(__file__).resolve().parent


def _coverage(
    result: Mapping[str, Any], facts_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_window = Counter()
    same_cross = Counter()
    classes = Counter()
    participants = Counter()
    readable: list[dict[str, Any]] = []
    for bundle in result["bundles"]:
        for window_id in bundle["window_ids"]:
            by_window[str(window_id)] += 1
        same_cross[str(bundle["connection_pattern"])] += 1
        classes[str(bundle["bundle_class"])] += 1
        participants[" + ".join(bundle["participants"])] += 1
        readable.append({
            **bundle,
            "facts": [{
                "fact_id": fact_id,
                "fact_text": facts_by_id[fact_id]["fact_text"],
                "subject": facts_by_id[fact_id]["subject"],
                "participants": facts_by_id[fact_id]["participants"],
                "window_id": facts_by_id[fact_id]["window_id"],
                "source_positions": facts_by_id[fact_id]["source_positions"],
            } for fact_id in bundle["fact_ids"]],
        })
    child_reason: dict[str, str] = {}
    for window_id in ("C01", "C02", "C03"):
        selected = by_window.get(window_id, 0)
        all_candidate = int(result["all_candidate_bundle_count_by_window"].get(window_id, 0))
        if selected:
            child_reason[window_id] = "selected"
        elif all_candidate:
            child_reason[window_id] = "mechanically_eligible_but_not_selected_before_hard_max"
        else:
            child_reason[window_id] = "no_explicit_participant_bridge_to_another_window"
    return {
        "materialized_bundle_count": len(result["bundles"]),
        "same_subject_bundle_count": same_cross.get("same_subject", 0),
        "cross_subject_participant_connected_bundle_count": same_cross.get("cross_subject", 0),
        "bundle_class_counts": dict(sorted(classes.items())),
        "bundle_count_by_window": {
            window_id: by_window.get(window_id, 0)
            for window_id in ("P01", "P02", "P03", "P04", "P05", "C01", "C02", "C03")
        },
        "bundle_count_by_participant_set": dict(sorted(participants.items())),
        "child_window_coverage_reason": child_reason,
        "readable_bundles": readable,
    }


def run() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"participant_rebundle_output_exists:{OUTPUT_ROOT}")
    facts_payload = _load(FACTS_PATH)
    expanded = _load(EXPANDED_PATH)
    facts = list(facts_payload["facts"])
    if len(facts) != 43 or _id_sha(facts_payload) != EXPECTED_ALLOWLIST_ID_SHA:
        raise RuntimeError("frozen_eligible_fact_pool_mismatch")
    integrity_before = {
        "fact_first_branch_manifest": tree_manifest(MODULE_ROOT),
        "position_first_branch_manifest": tree_manifest(OLD_POSITION_MODULE),
        "episode_sha256": sha256_file(EXPANDED_PATH),
        "eligible_facts_sha256": sha256_file(FACTS_PATH),
        "previous_rebundle_manifest": tree_manifest(FACT_ROOT),
    }
    if integrity_before["position_first_branch_manifest"]["tree_sha256"] != EXPECTED_POSITION_SHA:
        raise RuntimeError("position_first_branch_sha_mismatch")
    if integrity_before["episode_sha256"] != EXPECTED_EPISODE_SHA:
        raise RuntimeError("episode_sha_mismatch")

    facts_with_participants = materialize_fact_participants(facts, expanded=expanded)
    facts_by_id = {str(row["fact_id"]): row for row in facts_with_participants}
    same_result = bundle_facts(facts)
    participant_result = bundle_participant_connected_facts(facts_with_participants)
    coverage = _coverage(participant_result, facts_by_id)
    participant_count_distribution = Counter(len(row["participants"]) for row in facts_with_participants)
    participant_stats = {
        "known_dialogue_participants": known_dialogue_participants(expanded),
        "eligible_fact_count": len(facts_with_participants),
        "facts_with_1_participant": participant_count_distribution.get(1, 0),
        "facts_with_2_or_more_explicit_participants": sum(
            count for number, count in participant_count_distribution.items() if number >= 2
        ),
        "fact_count_by_participant_count": {
            str(number): count for number, count in sorted(participant_count_distribution.items())
        },
        "same_subject_eligible_combinations": same_result["all_mechanically_eligible_bundle_count"],
        "participant_connected_eligible_combinations": participant_result["all_mechanically_eligible_bundle_count"],
        "new_cross_subject_eligible_combinations": participant_result["cross_subject_eligible_combination_count"],
        "participant_same_subject_eligible_combinations": participant_result["same_subject_eligible_combination_count"],
        "exact_duplicate_fact_set_count": participant_result["exact_duplicate_fact_set_count"],
    }

    OUTPUT_ROOT.mkdir(parents=True)
    atomic_json(OUTPUT_ROOT / "eligible_facts_with_participants.json", {
        "participant_extraction_version": "exact_known_name_match_fact_and_source_turn_v1",
        "source_eligible_facts_path": str(FACTS_PATH),
        "source_allowlist_id_sha256": EXPECTED_ALLOWLIST_ID_SHA,
        "fact_count": len(facts_with_participants),
        "facts": facts_with_participants,
    })
    atomic_json(OUTPUT_ROOT / "participant_connectivity_stats.json", participant_stats)
    atomic_json(OUTPUT_ROOT / "all_mechanically_eligible_bundle_stats.json", {
        "runner_version": RUNNER_VERSION,
        "bundler_version": PARTICIPANT_BUNDLER_VERSION,
        **participant_stats,
        "all_selection_class_counts": participant_result["all_selection_class_counts"],
        "all_bundle_class_counts": participant_result["all_bundle_class_counts"],
        "all_candidate_bundle_count_by_window": participant_result["all_candidate_bundle_count_by_window"],
        "limits": participant_result["limits"],
    })
    atomic_json(OUTPUT_ROOT / "candidate_bundles.json", participant_result)
    atomic_json(OUTPUT_ROOT / "bundle_coverage_report.json", coverage)

    integrity_after = {
        "fact_first_branch_manifest": tree_manifest(MODULE_ROOT),
        "position_first_branch_manifest": tree_manifest(OLD_POSITION_MODULE),
        "episode_sha256": sha256_file(EXPANDED_PATH),
        "eligible_facts_sha256": sha256_file(FACTS_PATH),
        "previous_rebundle_manifest": tree_manifest(FACT_ROOT),
    }
    integrity = {
        "before": integrity_before,
        "after": integrity_after,
        "fact_first_branch_stable_during_run": (
            integrity_before["fact_first_branch_manifest"]["tree_sha256"]
            == integrity_after["fact_first_branch_manifest"]["tree_sha256"]
        ),
        "position_first_branch_unchanged": (
            integrity_before["position_first_branch_manifest"]["tree_sha256"]
            == integrity_after["position_first_branch_manifest"]["tree_sha256"]
        ),
        "episode_unchanged": integrity_before["episode_sha256"] == integrity_after["episode_sha256"],
        "eligible_facts_unchanged": (
            integrity_before["eligible_facts_sha256"] == integrity_after["eligible_facts_sha256"]
        ),
        "previous_rebundle_unchanged": (
            integrity_before["previous_rebundle_manifest"]["tree_sha256"]
            == integrity_after["previous_rebundle_manifest"]["tree_sha256"]
        ),
    }
    atomic_json(OUTPUT_ROOT / "run_integrity_before_after.json", integrity)
    summary = {
        "runner_version": RUNNER_VERSION,
        "execution": {
            "api": 0, "fact_extractor": 0, "qa_writer": 0, "solver": 0,
            "expansion": 0, "memorax": 0, "training": False,
        },
        **participant_stats,
        "materialized_bundles": participant_result["bundle_count"],
        "materialized_selection_class_counts": participant_result["materialized_selection_class_counts"],
        "materialized_bundle_class_counts": participant_result["materialized_bundle_class_counts"],
        "bundle_count_by_window": coverage["bundle_count_by_window"],
        "semantic_audit_pending": True,
    }
    atomic_json(OUTPUT_ROOT / "participant_rebundle_summary.json", summary)
    atomic_text(OUTPUT_ROOT / "PARTICIPANT_REBUNDLE_MECHANICAL_REPORT.md", "\n".join([
        "# Participant-Connected Frozen Rebundle Mechanical Report", "",
        f"- Eligible facts: `{len(facts)}`",
        f"- Facts with one participant: `{participant_stats['facts_with_1_participant']}`",
        f"- Facts with two or more explicit participants: `{participant_stats['facts_with_2_or_more_explicit_participants']}`",
        f"- Same-subject combinations: `{participant_stats['same_subject_eligible_combinations']}`",
        f"- Participant-connected combinations: `{participant_stats['participant_connected_eligible_combinations']}`",
        f"- New cross-subject combinations: `{participant_stats['new_cross_subject_eligible_combinations']}`",
        f"- Materialized bundles: `{participant_result['bundle_count']}`",
        f"- Same-subject selected: `{coverage['same_subject_bundle_count']}`",
        f"- Cross-subject selected: `{coverage['cross_subject_participant_connected_bundle_count']}`",
        f"- Bundle classes: `{json.dumps(coverage['bundle_class_counts'], sort_keys=True)}`",
        f"- Window coverage: `{json.dumps(coverage['bundle_count_by_window'], sort_keys=True)}`",
        f"- C02 status: `{coverage['child_window_coverage_reason']['C02']}`",
        "- API / Fact Extractor / Writer / Solver / Expansion / Memorax: `0`", "",
    ]))
    return summary


def main() -> int:
    print(json.dumps(run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
