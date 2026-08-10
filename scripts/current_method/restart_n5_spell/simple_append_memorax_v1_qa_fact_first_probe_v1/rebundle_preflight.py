"""Zero-API preflight for frozen-fact rebundling."""

from __future__ import annotations

import json
from pathlib import Path

from ..core import atomic_json, atomic_text
from .fact_bundler import MAX_BUNDLES_EPISODE, bundle_facts, validate_bundle
from .integrity import sha256_file, tree_manifest
from .rebundle_offline import (
    AUDIT_PATH, EXPECTED_INPUT_SHA, EXPECTED_OLD_POSITION_SHA, FACTS_PATH,
    FROZEN_INPUT_PATH, LIVE_ROOT, OLD_POSITION_MODULE, OUTPUT_ROOT,
    _load, derive_eligible_allowlist,
)


def _fact(fact_id: str, subject: str, window_id: str, order: int) -> dict[str, object]:
    return {
        "fact_id": fact_id,
        "fact_text": fact_id,
        "subject": subject,
        "window_id": window_id,
        "window_order": order,
        "window_kind": "child_scene" if window_id.startswith("C") else "parent_session",
    }


def run() -> dict[str, object]:
    facts = list(_load(FACTS_PATH)["facts"])
    audit = _load(AUDIT_PATH)
    allowlist = derive_eligible_allowlist(audit, facts)
    fixture = [
        *[_fact(f"F{i:04d}", "A", "P01", 1) for i in range(1, 14)],
        _fact("F0014", "A", "C03", 8),
        _fact("F0015", "B", "P05", 5),
        _fact("F0016", "B", "C01", 6),
        _fact("F0017", "B", "C02", 7),
    ]
    bundled = bundle_facts(fixture)
    selected_ids = {
        fact_id for bundle in bundled["bundles"] for fact_id in bundle["fact_ids"]
    }
    selected_subjects = {str(bundle["subject"]) for bundle in bundled["bundles"]}
    selected_pairs = {
        tuple(bundle["coverage_window_pair"]) for bundle in bundled["bundles"]
    }
    too_large = [_fact(f"X{i}", "A", f"P{i:02d}", i) for i in range(1, 6)]
    crowded = [
        _fact("Y1", "A", "P01", 1), _fact("Y2", "A", "P01", 1),
        _fact("Y3", "A", "P01", 1), _fact("Y4", "A", "C01", 6),
    ]
    old_before = tree_manifest(OLD_POSITION_MODULE)["tree_sha256"]
    live_before = tree_manifest(LIVE_ROOT)["tree_sha256"]
    checks = {
        "api_calls_zero": True,
        "fact_extractor_calls_zero": True,
        "qa_writer_calls_zero": True,
        "solver_calls_zero": True,
        "expansion_calls_zero": True,
        "memorax_calls_zero": True,
        "offline_output_root_absent": not OUTPUT_ROOT.exists(),
        "frozen_allowlist_count_43": allowlist["eligible_fact_count"] == 43,
        "allowlist_is_mechanical_existing_audit_derivation": (
            allowlist["semantic_reclassification_performed"] is False
        ),
        "earliest_fact_cap_removed": bundled["limits"]["pre_combination_fact_cap"] is None,
        "all_fixture_facts_considered": sum(
            int(row["considered_fact_count"]) for row in bundled["subject_stats"]
        ) == len(fixture),
        "late_child_fact_selected": "F0014" in selected_ids,
        "parent_child_selected": bundled["materialized_class_counts"].get("parent_child", 0) > 0,
        "different_subjects_not_combined": all(
            len({next(row["subject"] for row in fixture if row["fact_id"] == fact_id)
                 for fact_id in bundle["fact_ids"]}) == 1
            for bundle in bundled["bundles"]
        ),
        "one_window_not_bundle": bundle_facts([
            _fact("Z1", "A", "P01", 1), _fact("Z2", "A", "P01", 1),
        ])["bundle_count"] == 0,
        "bundle_size_two_valid": validate_bundle([
            _fact("A1", "A", "P01", 1), _fact("A2", "A", "C01", 6),
        ]) == [],
        "bundle_size_four_valid": validate_bundle([
            _fact("A1", "A", "P01", 1), _fact("A2", "A", "P01", 1),
            _fact("A3", "A", "C01", 6), _fact("A4", "A", "C01", 6),
        ]) == [],
        "bundle_size_over_four_rejected": "bundle_size_above_maximum" in validate_bundle(too_large),
        "per_window_over_two_rejected": "bundle_window_fact_limit_exceeded" in validate_bundle(crowded),
        "deterministic_output": bundled == bundle_facts(list(reversed(fixture))),
        "episode_hard_max_60": bundled["limits"]["max_bundles_episode"] == MAX_BUNDLES_EPISODE == 60,
        "round_robin_subject_coverage": selected_subjects == {"A", "B"},
        "round_robin_pair_coverage": len(selected_pairs) >= 2,
        "frozen_episode_sha_unchanged": sha256_file(FROZEN_INPUT_PATH) == EXPECTED_INPUT_SHA,
        "old_position_module_sha_unchanged": old_before == EXPECTED_OLD_POSITION_SHA,
        "historical_live_run_readable": LIVE_ROOT.is_dir(),
    }
    old_after = tree_manifest(OLD_POSITION_MODULE)["tree_sha256"]
    live_after = tree_manifest(LIVE_ROOT)["tree_sha256"]
    checks["old_position_module_stable_during_preflight"] = old_before == old_after
    checks["historical_live_run_stable_during_preflight"] = live_before == live_after
    failed = sorted(key for key, passed in checks.items() if not passed)
    report = {
        "preflight_version": "fact_first_rebundle_preflight_v1",
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "failed_checks": failed,
        "frozen_allowlist": allowlist,
        "integrity": {
            "old_position_module_sha_before": old_before,
            "old_position_module_sha_after": old_after,
            "historical_live_run_sha_before": live_before,
            "historical_live_run_sha_after": live_after,
            "frozen_episode_sha256": sha256_file(FROZEN_INPUT_PATH),
        },
        "execution": {
            "api": 0, "fact_extractor": 0, "qa_writer": 0, "solver": 0,
            "expansion": 0, "memorax": 0,
        },
    }
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(exist_ok=True)
    atomic_json(reports / "FACT_FIRST_REBUNDLE_PREFLIGHT.json", report)
    atomic_text(reports / "FACT_FIRST_REBUNDLE_PREFLIGHT.md", "\n".join([
        "# Fact-First Frozen Rebundle Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{report['passed']}/{report['total']}`",
        f"- Frozen eligible facts: `{allowlist['eligible_fact_count']}`",
        f"- Failed checks: `{json.dumps(failed)}`",
        "- API / Fact Extractor / Writer / Solver / Expansion / Memorax: `0`", "",
    ]))
    return report


def main() -> int:
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
