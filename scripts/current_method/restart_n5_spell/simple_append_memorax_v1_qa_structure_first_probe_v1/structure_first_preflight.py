"""Zero-API preflight for the isolated structure/evidence-first QA probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from .integrity import initial_git_status, protected_manifests
from .structure_first_contract import validate_structure_first_writer_output
from .structure_first_fixtures import (
    ambiguous_gold_envelope,
    make_envelope,
    weak_distractor_envelope,
)
from .structure_first_live import (
    DEFAULT_EXPANDED,
    DEFAULT_MEMORY_OVERVIEW,
    DEFAULT_RUN_ROOT,
    EXPECTED_EXPANDED_SHA256,
    EXPECTED_MEMORY_OVERVIEW_SHA256,
    WRITER_COMPLETION_TOKENS,
    _load,
    _repo_root,
    preflight as live_preflight,
)
from .structure_first_prompts import build_structure_first_prompt


PREFLIGHT_VERSION = "structure_evidence_first_qa_preflight_v1"


def _valid_count(value: dict[str, Any], expanded: dict[str, Any]) -> int:
    return int(validate_structure_first_writer_output(
        value, expanded_dialogue=expanded).get("structural_valid_count") or 0)


def run_preflight() -> dict[str, Any]:
    repo = _repo_root()
    git_before = initial_git_status(repo)
    protected_before = protected_manifests(repo)
    expanded = _load(DEFAULT_EXPANDED)
    overview = _load(DEFAULT_MEMORY_OVERVIEW)
    prompt = build_structure_first_prompt(expanded, overview)
    checks: dict[str, bool] = {
        "api_calls_zero": True,
        "writer_calls_zero": True,
        "solver_calls_zero": True,
        "expansion_calls_zero": True,
        "memorax_calls_zero": True,
        "frozen_dialogue_sha_unchanged": (
            sha256_file(DEFAULT_EXPANDED) == EXPECTED_EXPANDED_SHA256),
        "frozen_overview_sha_unchanged": (
            sha256_file(DEFAULT_MEMORY_OVERVIEW) ==
            EXPECTED_MEMORY_OVERVIEW_SHA256),
        "protected_modules_unchanged": bool(protected_before["unchanged"]),
        "new_live_run_root_absent": not DEFAULT_RUN_ROOT.exists(),
        "writer_completion_budget_16384": WRITER_COMPLETION_TOKENS == 16_384,
        "structure_first_prompt_active": (
            "MANDATORY GENERATION ORDER" in prompt["prompt"]),
        "evidence_selected_before_materialization": (
            prompt["prompt"].find("Lock 2-6 minimal sufficient") <
            prompt["prompt"].find("Only then materialize")),
        "no_minimum_or_coverage_quota": (
            "There is no minimum question count and no coverage quota."
            in prompt["prompt"]),
        "zero_question_fixture_valid": _valid_count(
            make_envelope(0), expanded) == 0 and
            validate_structure_first_writer_output(
                make_envelope(0), expanded_dialogue=expanded
            ).get("zero_question_batch_valid") is True,
        "one_question_fixture_valid": _valid_count(make_envelope(1), expanded) == 1,
        "three_question_fixture_valid": _valid_count(make_envelope(3), expanded) == 3,
        "twelve_question_fixture_valid": _valid_count(make_envelope(12), expanded) == 12,
        "evidence_2_fixture_valid": _valid_count(
            make_envelope(1, evidence_count=2), expanded) == 1,
        "evidence_6_fixture_valid": _valid_count(
            make_envelope(1, evidence_count=6), expanded) == 1,
        "ambiguous_gold_rejected": _valid_count(
            ambiguous_gold_envelope(), expanded) == 0,
        "weak_distractor_rejected": _valid_count(
            weak_distractor_envelope(), expanded) == 0,
    }
    live = live_preflight()
    checks["live_preflight_passed_without_api"] = live["decision"] == "pass"
    protected_after = protected_manifests(repo)
    git_after = initial_git_status(repo)
    checks["protected_sha_same_before_after"] = (
        protected_before == protected_after)
    checks["git_status_same_during_preflight"] = git_before == git_after
    failed = sorted(key for key, passed in checks.items() if not passed)
    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "writer_prompt": {
            "sha256": prompt["prompt_sha256"],
            "character_count": prompt["character_count"],
            "token_estimate": prompt["token_estimate"],
            "completion_budget": WRITER_COMPLETION_TOKENS,
        },
        "frozen_inputs": {
            "expanded_dialogue": str(DEFAULT_EXPANDED),
            "expanded_dialogue_sha256": sha256_file(DEFAULT_EXPANDED),
            "memory_overview": str(DEFAULT_MEMORY_OVERVIEW),
            "memory_overview_sha256": sha256_file(DEFAULT_MEMORY_OVERVIEW),
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "api_calls": 0,
    }
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "STRUCTURE_FIRST_QA_PREFLIGHT.json", report)
    atomic_text(reports / "STRUCTURE_FIRST_QA_PREFLIGHT.md", "\n".join([
        "# Structure/Evidence-First QA Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- Checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Failed: `{json.dumps(failed)}`",
        f"- Writer calls: `0`",
        f"- Expanded dialogue SHA: `{sha256_file(DEFAULT_EXPANDED)}`",
        f"- Protected directories unchanged: `{str(protected_after['unchanged']).lower()}`",
        "",
    ]))
    return report


def main() -> int:
    report = run_preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
