"""Zero-API preflight for subject-binding v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.inputs import load_object
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.integrity import git_status
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.windows import compile_natural_windows
from .inputs import DEFAULT_EXPANDED, SUBJECT_BINDING_TARGET_RUN_ROOT, frozen_input_manifest
from .subject_binding_contract import (
    materialize_subject_bound_facts,
    validate_subject_binding_output,
)
from .subject_binding_prompts import build_subject_binding_prompt


MODULE_ROOT = Path(__file__).resolve().parent
PREFLIGHT_VERSION = "qa_fact_first_episode91716_subject_binding_preflight_v1"


def _validate(expanded: dict[str, Any], window: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return validate_subject_binding_output(
        {"facts": rows}, expanded_dialogue=expanded, window=window,
    )


def run_preflight() -> dict[str, Any]:
    repo = MODULE_ROOT.parents[3]
    before_status = git_status(repo)
    inputs = frozen_input_manifest()
    expanded = load_object(DEFAULT_EXPANDED)
    compilation = compile_natural_windows(expanded)
    windows = compilation["windows"]
    c01 = next(window for window in windows if window["window_id"] == "C01")
    entities = sorted({turn["speaker"] for turn in expanded["turns"]})
    fixtures: dict[str, dict[str, Any]] = {
        "speaker_at_position": _validate(expanded, c01, [{
            "fact_text": "The speaker recalled the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "speaker_at_position", "position": 59},
        }]),
        "known_entity_other_than_speaker": _validate(expanded, c01, [{
            "fact_text": "Neighbors A remembered the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "known_entity", "entity_ref": "Neighbors A"},
        }]),
        "named_entity_conflict": _validate(expanded, c01, [{
            "fact_text": "Neighbors A remembered the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "speaker_at_position", "position": 59},
        }]),
        "legacy_free_subject": _validate(expanded, c01, [{
            "fact_text": "Neighbors A remembered the squirrel incident.",
            "source_positions": [59],
            "subject": "Neighbors A",
        }]),
        "binding_position_not_in_evidence": _validate(expanded, c01, [{
            "fact_text": "The speaker recalled the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "speaker_at_position", "position": 60},
        }]),
        "unknown_entity": _validate(expanded, c01, [{
            "fact_text": "An unknown person recalled the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "known_entity", "entity_ref": "Unknown"},
        }]),
        "empty_facts": _validate(expanded, c01, []),
    }
    valid = {name: result.get("mechanically_valid_count") == 1 for name, result in fixtures.items()}
    checks: dict[str, bool] = {
        "api_zero": True,
        "solver_zero": True,
        "expansion_zero": True,
        "memorax_zero": True,
        "input_sha_frozen": bool(inputs["expanded_unchanged"]),
        "windows_valid": bool(compilation["valid"]),
        "eight_windows": compilation.get("window_count") == 8,
        "speaker_binding_valid": valid["speaker_at_position"],
        "known_entity_binding_valid": valid["known_entity_other_than_speaker"],
        "named_entity_conflict_rejected": not valid["named_entity_conflict"] and "named_entity_requires_known_entity_binding" in (
            fixtures["named_entity_conflict"]["items"][0]["mechanical_errors"]
        ),
        "legacy_subject_rejected": not valid["legacy_free_subject"] and "free_subject_forbidden" in (
            fixtures["legacy_free_subject"]["items"][0]["mechanical_errors"]
        ),
        "binding_position_rejected": not valid["binding_position_not_in_evidence"],
        "unknown_entity_rejected": not valid["unknown_entity"],
        "empty_facts_valid": fixtures["empty_facts"].get("empty_facts_valid") is True,
        "prompt_uses_binding_contract": build_subject_binding_prompt(
            c01, entity_refs=entities,
        )["output_fields"] == ["fact_text", "source_positions", "subject_binding"],
        "new_live_root_absent": not SUBJECT_BINDING_TARGET_RUN_ROOT.exists(),
    }
    materialized = materialize_subject_bound_facts(
        [fixtures["known_entity_other_than_speaker"]],
        windows_by_id={window["window_id"]: window for window in windows},
    )
    checks["materialization_resolves_subject"] = bool(
        materialized
        and materialized[0]["subject"] == "Neighbors A"
        and materialized[0]["asserting_speaker"] == "Neighbors B"
        and materialized[0]["source_turn_ids"] == ["R01:S01:T004"]
    )
    after_status = git_status(repo)
    checks["git_status_unchanged"] = before_status == after_status
    failed = sorted(name for name, value in checks.items() if not value)
    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "input_manifest": inputs,
        "entity_refs": entities,
        "window": {"window_id": c01["window_id"], "positions": c01["global_positions"]},
        "fixture_results": fixtures,
        "materialized_fixture": materialized,
        "api_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "live_executed": False,
    }
    reports = MODULE_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "SUBJECT_BINDING_PREFLIGHT.json", report)
    atomic_text(reports / "SUBJECT_BINDING_PREFLIGHT.md", "\n".join([
        "# Subject Binding v1 Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Failed: `{json.dumps(failed)}`",
        "- API / Solver / Expansion / Memorax: `0 / 0 / 0 / 0`",
        "",
        "The legacy free-form `subject` field is rejected. Subjects are resolved from",
        "`speaker_at_position` or an exact `known_entity` reference.",
        "",
    ]))
    return report


if __name__ == "__main__":
    print(json.dumps(run_preflight(), ensure_ascii=False, indent=2))
