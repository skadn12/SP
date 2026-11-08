"""Zero-API preflight for the episode-91716 fact-first QA probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.fact_bundler import bundle_facts
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.fact_contract import (
    materialize_facts,
    validate_fact_extractor_output,
)
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.fact_prompts import build_fact_prompt
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.integrity import git_status
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.windows import compile_natural_windows
from .inputs import DEFAULT_EXPANDED, DEFAULT_MEMORY_OVERVIEW, TARGET_RUN_ROOT, frozen_input_manifest


MODULE_ROOT = Path(__file__).resolve().parent
PREFLIGHT_VERSION = "qa_fact_first_episode91716_preflight_v1"


def _fact(positions: list[int], text: str = "A completed a concrete action.") -> dict[str, Any]:
    return {"fact_text": text, "source_positions": positions, "subject": "A"}


def run_preflight() -> dict[str, Any]:
    before_status = git_status(MODULE_ROOT.parents[3])
    inputs = frozen_input_manifest()
    checks: dict[str, bool] = {
        "api_zero": True,
        "solver_zero": True,
        "expansion_zero": True,
        "memorax_zero": True,
        "target_run_absent": bool(inputs["target_run_absent"]),
        "expanded_sha_frozen": bool(inputs["expanded_unchanged"]),
        "overview_exists": DEFAULT_MEMORY_OVERVIEW.is_file(),
    }
    errors: list[str] = []
    expanded: dict[str, Any] = {}
    windows: list[dict[str, Any]] = []
    compilation: dict[str, Any] = {}
    try:
        expanded = json.loads(DEFAULT_EXPANDED.read_text(encoding="utf-8"))
        compilation = compile_natural_windows(expanded)
        windows = compilation["windows"]
    except Exception as exc:
        errors.append(f"input_or_window_error:{type(exc).__name__}:{exc}")
    checks.update({
        "windows_valid": bool(compilation.get("valid")),
        "five_parent_windows": compilation.get("parent_session_count") == 5,
        "three_child_windows": compilation.get("child_scene_count") == 3,
        "97_turns_covered": compilation.get("assigned_turn_count") == 97,
        "eight_windows": compilation.get("window_count") == 8,
    })
    first = windows[0] if windows else {"global_positions": []}
    allowed = list(first.get("global_positions") or [])
    def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return validate_fact_extractor_output(
            {"facts": rows}, expanded_dialogue=expanded, window=first,
        )
    fixtures = {
        "one_position": validate([_fact([allowed[0]])]) if allowed else {},
        "three_positions": validate([_fact(allowed[:3])]) if len(allowed) >= 3 else {},
        "zero_position": validate([_fact([])]) if allowed else {},
        "four_positions": validate([_fact(allowed[:4])]) if len(allowed) >= 4 else {},
        "duplicate_position": validate([_fact([allowed[0], allowed[0]])]) if allowed else {},
        "empty_facts": validate([]) if allowed else {},
        "over_capacity": validate([_fact([allowed[0]]) for _ in range(9)]) if allowed else {},
    }
    checks.update({
        "one_position_valid": fixtures.get("one_position", {}).get("mechanically_valid_count") == 1,
        "three_positions_valid": fixtures.get("three_positions", {}).get("mechanically_valid_count") == 1,
        "zero_position_rejected": "source_positions_below_minimum" in ((fixtures.get("zero_position", {}).get("items") or [{}])[0].get("mechanical_errors") or []),
        "four_positions_rejected": "source_positions_above_maximum" in ((fixtures.get("four_positions", {}).get("items") or [{}])[0].get("mechanical_errors") or []),
        "duplicate_position_rejected": "source_positions_duplicate" in ((fixtures.get("duplicate_position", {}).get("items") or [{}])[0].get("mechanical_errors") or []),
        "empty_facts_valid": bool(fixtures.get("empty_facts", {}).get("empty_facts_valid")),
        "over_capacity_rejected": "fact_capacity_exceeded" in (fixtures.get("over_capacity", {}).get("envelope_errors") or []),
        "fact_prompt_is_slim": bool(windows and build_fact_prompt(windows[0])["output_fields"] == ["fact_text", "source_positions", "subject"]),
    })
    if windows:
        fixture_validations = []
        fixture_rows = [
            [_fact([windows[0]["global_positions"][0]])],
            [_fact([windows[1]["global_positions"][0]])],
        ]
        for window, row in zip(windows[:2], fixture_rows):
            validation = validate_fact_extractor_output(
                {"facts": row}, expanded_dialogue=expanded, window=window,
            )
            fixture_validations.append({**validation, "window_id": window["window_id"]})
        materialized = materialize_facts(
            fixture_validations,
            windows_by_id={window["window_id"]: window for window in windows},
        )
        bundles = bundle_facts(materialized)
        checks["materialization_maps_positions"] = bool(materialized and materialized[0]["source_turn_ids"])
        checks["bundler_is_deterministic"] = bundles == bundle_facts(list(reversed(materialized)))
        checks["bundles_cross_windows"] = all(len(row["window_ids"]) >= 2 for row in bundles["bundles"])
    else:
        materialized, bundles = [], {"bundle_count": 0, "bundles": []}
        checks.update({"materialization_maps_positions": False, "bundler_is_deterministic": False, "bundles_cross_windows": False})
    after_status = git_status(MODULE_ROOT.parents[3])
    checks["git_status_unchanged"] = before_status == after_status
    failed = sorted(name for name, value in checks.items() if not value)
    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed and not errors else "fail",
        "checks": checks,
        "failed_checks": failed,
        "errors": errors,
        "inputs": inputs,
        "window_compilation": {key: compilation.get(key) for key in (
            "window_count", "parent_session_count", "child_scene_count", "assigned_turn_count", "valid", "errors")},
        "fixture_results": fixtures,
        "materialized_fixture_count": len(materialized),
        "bundle_fixture_count": bundles.get("bundle_count", 0),
        "api_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "run_root": str(TARGET_RUN_ROOT),
    }
    reports = MODULE_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "FACT_FIRST_EPISODE91716_PREFLIGHT.json", report)
    atomic_text(reports / "FACT_FIRST_EPISODE91716_PREFLIGHT.md", "\n".join([
        "# Episode 91716 Fact-First Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Windows: `{compilation.get('window_count')}` (`{compilation.get('parent_session_count')}` Parent + `{compilation.get('child_scene_count')}` Child)",
        f"- Turns covered: `{compilation.get('assigned_turn_count')}`",
        f"- Checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Failed: `{json.dumps(failed)}`",
        "- API / Solver / Expansion / Memorax: `0 / 0 / 0 / 0`",
        "",
    ]))
    return report


if __name__ == "__main__":
    print(json.dumps(run_preflight(), ensure_ascii=False, indent=2))
