"""Rebundle the frozen, human-audited fact pool without API calls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text
from .fact_bundler import BUNDLER_VERSION, bundle_facts
from .integrity import sha256_file, tree_manifest


REPLAY_VERSION = "fact_first_frozen_rebundle_v1"
LIVE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_first_probe_v1_live_1"
)
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_first_rebundle_v1_offline"
)
FACTS_PATH = LIVE_ROOT / "qa/materialized_facts.json"
AUDIT_PATH = LIVE_ROOT / "reports/FACT_FIRST_SEMANTIC_AUDIT.json"
FROZEN_INPUT_PATH = LIVE_ROOT / "frozen/expanded_dialogue.json"
OLD_POSITION_MODULE = Path(
    "/root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method/"
    "restart_n5_spell/simple_append_memorax_v1_qa_position_first_probe_v1"
)
FACT_FIRST_MODULE = Path(__file__).resolve().parent
EXPECTED_OLD_POSITION_SHA = (
    "ead4f99b8e3659d3aea1117d87dc16a30764fde046342a916119c6a47be29048"
)
EXPECTED_INPUT_SHA = (
    "97cf7b5aef3b9e6b7a04e9323f015d7d3cbef6f82fc985da97b89cbc70f0f7f5"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_eligible_allowlist(audit: Mapping[str, Any], facts: list[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = audit["fact_decisions"]
    exclusion_fields = (
        "provenance_reference_failure",
        "wrong_attribution",
        "grounded_subjective_or_low_value",
        "grounded_cross_window_duplicate",
    )
    excluded_by_reason = {
        key: sorted(str(fact_id) for fact_id in decisions[key])
        for key in exclusion_fields
    }
    excluded = {
        fact_id for rows in excluded_by_reason.values() for fact_id in rows
    }
    all_ids = {str(row["fact_id"]) for row in facts}
    eligible = sorted(all_ids - excluded)
    if len(all_ids) != 64 or len(eligible) != 43:
        raise RuntimeError(
            f"frozen_audit_count_mismatch:all={len(all_ids)}:eligible={len(eligible)}"
        )
    if int(audit["counts"]["deduplicated_grounded_atomic_facts"]) != len(eligible):
        raise RuntimeError("frozen_audit_declared_count_mismatch")
    return {
        "allowlist_version": "fact_first_human_audit_allowlist_v1",
        "source_audit_path": str(AUDIT_PATH),
        "derivation": "all_fact_ids_minus_existing_audit_exclusion_lists",
        "eligible_fact_count": len(eligible),
        "eligible_fact_ids": eligible,
        "excluded_fact_count": len(excluded),
        "excluded_fact_ids_by_existing_audit_reason": excluded_by_reason,
        "semantic_reclassification_performed": False,
    }


def _fact_coverage(facts: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_window = Counter(str(row["window_id"]) for row in facts)
    by_subject_window: dict[str, Counter[str]] = defaultdict(Counter)
    for row in facts:
        by_subject_window[str(row["subject"])][str(row["window_id"])] += 1
    parent = sum(count for window, count in by_window.items() if window.startswith("P"))
    child = sum(count for window, count in by_window.items() if window.startswith("C"))
    return {
        "eligible_fact_count": len(facts),
        "parent_eligible_fact_count": parent,
        "child_eligible_fact_count": child,
        "eligible_fact_count_by_window": {
            window: by_window.get(window, 0)
            for window in ("P01", "P02", "P03", "P04", "P05", "C01", "C02", "C03")
        },
        "eligible_fact_count_by_subject_and_window": {
            subject: dict(sorted(counts.items()))
            for subject, counts in sorted(by_subject_window.items())
        },
    }


def _bundle_coverage(bundle_result: Mapping[str, Any], facts_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    bundles = list(bundle_result["bundles"])
    by_window = Counter()
    by_subject = Counter()
    by_pair = Counter()
    child_windows_with_eligible = {
        str(row["window_id"]) for row in facts_by_id.values()
        if str(row["window_id"]).startswith("C")
    }
    readable: list[dict[str, Any]] = []
    for bundle in bundles:
        for window in bundle["window_ids"]:
            by_window[str(window)] += 1
        by_subject[str(bundle["subject"])] += 1
        by_pair["_".join(bundle["coverage_window_pair"])] += 1
        readable.append({
            **bundle,
            "facts": [{
                "fact_id": fact_id,
                "fact_text": facts_by_id[fact_id]["fact_text"],
                "window_id": facts_by_id[fact_id]["window_id"],
                "source_positions": facts_by_id[fact_id]["source_positions"],
            } for fact_id in bundle["fact_ids"]],
        })
    uncovered_child = sorted(
        window for window in child_windows_with_eligible if by_window[window] == 0
    )
    return {
        "materialized_bundle_count": len(bundles),
        "bundle_class_counts": bundle_result["materialized_class_counts"],
        "bundle_count_by_window": {
            window: by_window.get(window, 0)
            for window in ("P01", "P02", "P03", "P04", "P05", "C01", "C02", "C03")
        },
        "bundle_count_by_subject": dict(sorted(by_subject.items())),
        "bundle_count_by_coverage_window_pair": dict(sorted(by_pair.items())),
        "child_window_coverage_positive": any(
            by_window[window] > 0 for window in child_windows_with_eligible
        ),
        "any_child_bundle_coverage_positive": any(
            by_window[window] > 0 for window in child_windows_with_eligible
        ),
        "all_eligible_child_windows_covered": bool(child_windows_with_eligible) and not uncovered_child,
        "child_windows_with_eligible_facts": sorted(child_windows_with_eligible),
        "child_windows_with_zero_bundle_coverage": uncovered_child,
        "readable_bundles": readable,
    }


def run() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"offline_output_root_exists:{OUTPUT_ROOT}")
    facts_payload = _load(FACTS_PATH)
    facts = list(facts_payload["facts"])
    audit = _load(AUDIT_PATH)

    integrity_before = {
        "fact_first_branch_manifest": tree_manifest(FACT_FIRST_MODULE),
        "old_position_first_branch_manifest": tree_manifest(OLD_POSITION_MODULE),
        "frozen_episode_sha256": sha256_file(FROZEN_INPUT_PATH),
        "fact_first_live_1_manifest": tree_manifest(LIVE_ROOT),
    }
    if integrity_before["old_position_first_branch_manifest"]["tree_sha256"] != EXPECTED_OLD_POSITION_SHA:
        raise RuntimeError("old_position_module_sha_mismatch")
    if integrity_before["frozen_episode_sha256"] != EXPECTED_INPUT_SHA:
        raise RuntimeError("frozen_episode_sha_mismatch")
    # The live manifest is frozen dynamically below; no historical file is changed.

    allowlist = derive_eligible_allowlist(audit, facts)
    eligible_ids = set(allowlist["eligible_fact_ids"])
    eligible_facts = [row for row in facts if str(row["fact_id"]) in eligible_ids]
    allowlist["allowlist_sha256"] = _canonical_sha(allowlist["eligible_fact_ids"])
    fact_coverage = _fact_coverage(eligible_facts)
    bundle_result = bundle_facts(eligible_facts)
    facts_by_id = {str(row["fact_id"]): row for row in eligible_facts}
    coverage = _bundle_coverage(bundle_result, facts_by_id)

    OUTPUT_ROOT.mkdir(parents=True)
    atomic_json(OUTPUT_ROOT / "eligible_fact_allowlist.json", allowlist)
    atomic_json(OUTPUT_ROOT / "eligible_facts.json", {
        "source_fact_schema_version": facts_payload.get("fact_schema_version"),
        "allowlist_sha256": allowlist["allowlist_sha256"],
        "eligible_fact_count": len(eligible_facts),
        "facts": eligible_facts,
    })
    atomic_json(OUTPUT_ROOT / "all_mechanically_eligible_bundle_stats.json", {
        "rebundle_version": REPLAY_VERSION,
        "bundler_version": BUNDLER_VERSION,
        "eligible_fact_coverage": fact_coverage,
        "all_mechanically_eligible_bundle_count": bundle_result["all_mechanically_eligible_bundle_count"],
        "all_candidate_class_counts": bundle_result["all_candidate_class_counts"],
        "subject_stats": bundle_result["subject_stats"],
        "limits": bundle_result["limits"],
    })
    atomic_json(OUTPUT_ROOT / "candidate_bundles.json", bundle_result)
    atomic_json(OUTPUT_ROOT / "bundle_coverage_report.json", coverage)

    integrity_after = {
        "fact_first_branch_manifest": tree_manifest(FACT_FIRST_MODULE),
        "old_position_first_branch_manifest": tree_manifest(OLD_POSITION_MODULE),
        "frozen_episode_sha256": sha256_file(FROZEN_INPUT_PATH),
        "fact_first_live_1_manifest": tree_manifest(LIVE_ROOT),
    }
    integrity = {
        "before": integrity_before,
        "after": integrity_after,
        "fact_first_branch_unchanged_during_rebundle": (
            integrity_before["fact_first_branch_manifest"]["tree_sha256"]
            == integrity_after["fact_first_branch_manifest"]["tree_sha256"]
        ),
        "old_position_first_branch_unchanged": (
            integrity_before["old_position_first_branch_manifest"]["tree_sha256"]
            == integrity_after["old_position_first_branch_manifest"]["tree_sha256"]
        ),
        "frozen_episode_unchanged": (
            integrity_before["frozen_episode_sha256"] == integrity_after["frozen_episode_sha256"]
        ),
        "fact_first_live_1_unchanged": (
            integrity_before["fact_first_live_1_manifest"]["tree_sha256"]
            == integrity_after["fact_first_live_1_manifest"]["tree_sha256"]
        ),
        "allowlist_sha256": allowlist["allowlist_sha256"],
    }
    atomic_json(OUTPUT_ROOT / "integrity_report.json", integrity)
    summary = {
        "rebundle_version": REPLAY_VERSION,
        "execution": {
            "api_calls": 0, "fact_extractor_calls": 0, "qa_writer_calls": 0,
            "solver_calls": 0, "expansion_calls": 0, "memorax_calls": 0,
        },
        "allowlist_sha256": allowlist["allowlist_sha256"],
        "eligible_grounded_facts": len(eligible_facts),
        "mechanically_eligible_combinations": bundle_result["all_mechanically_eligible_bundle_count"],
        "materialized_bundles": bundle_result["bundle_count"],
        "materialized_class_counts": bundle_result["materialized_class_counts"],
        "bundle_count_by_window": coverage["bundle_count_by_window"],
        "child_window_coverage_positive": coverage["child_window_coverage_positive"],
        "all_eligible_child_windows_covered": coverage["all_eligible_child_windows_covered"],
        "integrity": {
            key: value for key, value in integrity.items() if isinstance(value, bool)
        },
        "semantic_audit_pending": True,
    }
    atomic_json(OUTPUT_ROOT / "rebundle_summary.json", summary)
    atomic_text(OUTPUT_ROOT / "REBUNDLE_MECHANICAL_REPORT.md", "\n".join([
        "# Fact-First Frozen Rebundle Mechanical Report", "",
        f"- Eligible grounded facts: `{len(eligible_facts)}`",
        f"- All mechanically eligible combinations: `{bundle_result['all_mechanically_eligible_bundle_count']}`",
        f"- Materialized bundles: `{bundle_result['bundle_count']}`",
        f"- Parent-Child: `{bundle_result['materialized_class_counts'].get('parent_child', 0)}`",
        f"- Parent-Parent: `{bundle_result['materialized_class_counts'].get('parent_parent', 0)}`",
        f"- Child-Child: `{bundle_result['materialized_class_counts'].get('child_child', 0)}`",
        f"- Window coverage: `{json.dumps(coverage['bundle_count_by_window'], sort_keys=True)}`",
        f"- Child coverage positive: `{coverage['child_window_coverage_positive']}`",
        f"- All eligible Child windows covered: `{coverage['all_eligible_child_windows_covered']}`",
        "- API / Fact Extractor / Writer / Solver / Expansion / Memorax calls: `0`", "",
    ]))
    return summary


def main() -> int:
    print(json.dumps(run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
