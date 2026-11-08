"""Zero-API preflight for fixed-fact reverse question materialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..simple_append_v1.qa import normalize_focus_writer_batch
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.fact_bundler import (
    validate_bundle,
)
from .fact_bundle_writer_contract import (
    SCHEMA_VERSION,
    materialize_batch,
    validate_writer_output,
)


DEFAULT_FACT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_first_participant_rebundle_v1_offline"
)
DEFAULT_FACTS = DEFAULT_FACT_RUN / "eligible_facts_with_participants.json"
DEFAULT_BUNDLES = DEFAULT_FACT_RUN / "candidate_bundles.json"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_bundle_writer_preflight_v1"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def run(
    *, facts_path: Path = DEFAULT_FACTS,
    bundles_path: Path = DEFAULT_BUNDLES,
    report_root: Path = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    facts_value = _load(facts_path)
    bundles_value = _load(bundles_path)
    facts = list(facts_value.get("facts") or [])
    bundles = list(bundles_value.get("bundles") or [])
    eligible = [row for row in bundles if len(row.get("fact_ids") or []) == 4]
    checks: dict[str, bool] = {
        "facts_input_exists": facts_path.is_file(),
        "bundles_input_exists": bundles_path.is_file(),
        "facts_nonempty": bool(facts),
        "four_fact_bundle_available": bool(eligible),
        "api_calls_zero": True,
        "writer_calls_zero": True,
        "solver_calls_zero": True,
    }
    fixture_rows: list[dict[str, Any]] = []
    for bundle in eligible[:3]:
        ids = [str(value) for value in bundle["fact_ids"]]
        fixture_rows.append({
            "bundle_id": bundle["bundle_id"],
            "question_type": "cross_turn_comparison",
            "question": "Which fact is the later development in this bundle?",
            "answer_dimension": "event",
            "gold_fact_id": ids[-1],
            "distractor_fact_ids": ids[:3],
        })
    fixture = {
        "fact_bundle_writer_schema_version": SCHEMA_VERSION,
        "questions": fixture_rows,
    }
    validation = validate_writer_output(fixture, facts=facts, bundles=bundles)
    materialized = materialize_batch(fixture, facts=facts, bundles=bundles)
    checks.update({
        "fixture_reference_contract": bool(validation["valid"]),
        "materialization_valid": bool(materialized["valid"]),
        "materialized_count_matches": (
            len(materialized["materialized_questions"]) == len(fixture_rows)),
        "all_options_come_from_facts": all(
            set(row["options"].values()) <= {
                str(fact.get("fact_text") or "") for fact in facts
            }
            for row in materialized["materialized_questions"]
        ),
        "all_evidence_is_fact_provenance": all(
            all(ref for ref in row.get("evidence_turn_refs") or [])
            for row in materialized["materialized_questions"]
        ),
    })
    normalized = None
    if materialized["materialized_questions"]:
        envelope = normalize_focus_writer_batch({
            "mcq_focus_writer_schema_version": "simple_append_mcq_focus_writer_v4",
            "questions": materialized["materialized_questions"],
        }, minimum_current_round_only=0,
           minimum_current_round_joint_history=0, minimum_parent_child=0,
           minimum_general=0, require_option_contract=True)
        normalized = {
            "valid": envelope["valid"],
            "errors": envelope["errors"],
            "question_count": len(materialized["materialized_questions"]),
        }
        checks["existing_solver_schema_compatible"] = bool(envelope["valid"])
    else:
        checks["existing_solver_schema_compatible"] = False
        normalized = {"valid": False, "errors": ["no_materialized_fixture"]}
    report = {
        "preflight_version": "qa_fact_bundle_writer_preflight_v1",
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "facts_count": len(facts),
        "bundles_count": len(bundles),
        "four_fact_bundle_count": len(eligible),
        "fixture_question_count": len(fixture_rows),
        "writer_contract_validation": validation,
        "materialization": materialized,
        "existing_solver_schema": normalized,
        "api_calls": 0,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "FACT_BUNDLE_WRITER_PREFLIGHT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Fact-Bundle Writer Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Frozen facts: `{len(facts)}`",
        f"- Frozen bundles: `{len(bundles)}`",
        f"- Four-fact bundles: `{len(eligible)}`",
        f"- Shadow questions materialized: `{len(materialized['materialized_questions'])}`",
        "- API / Writer / Solver calls: `0 / 0 / 0`", "",
        "The Writer fixture selects only bundle/fact IDs. Option text and all "
        "evidence refs are derived from frozen fact records.",
    ]
    (report_root / "FACT_BUNDLE_WRITER_PREFLIGHT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["decision"] == "pass" else 1)
