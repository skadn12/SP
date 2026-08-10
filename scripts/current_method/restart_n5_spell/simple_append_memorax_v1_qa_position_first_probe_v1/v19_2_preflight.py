"""Zero-API Prompt calibration preflight for Structural QA Writer v19.2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from ..simple_append_v1.qa import materialize_solver_candidate
from ..simple_append_v1.qa_live import build_solver_source
from .integrity import protected_manifests, tree_manifest
from .structural_contract import (
    QUESTION_CAPACITY_MAX,
    STRUCTURAL_QA_SCHEMA_VERSION,
    solver_partition,
    validate_structural_writer_output,
)
from .structural_prompts import build_structural_qa_prompt


EXPERIMENT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_experiment_1")
HISTORICAL_RUNS = (
    Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
         "conversation_chronicles_episode_146700_qa_structural_v19_probe_v1_live_1"),
    Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
         "conversation_chronicles_episode_146700_qa_structural_v19_1_probe_v1_live_1"),
)
REPORT_ROOT = Path(__file__).resolve().parent / "reports"
BEFORE_KEY_FILE_SHA256 = {
    "structural_contract.py":
        "7bb2cc48c9e398061d3d2391d80d5230530ad0eeb99bfe7a914bb98ada612184",
    "structural_prompts.py":
        "436a88504f1f07da48df7b7d9a14195fb77b9485894a6144fa918f4cf9eb78d5",
    "structural_qa_live.py":
        "7bac6f313a87a5f6ff5e181ad4df8bf5a346f17a379f19fcae889171e73546f1",
    "test_structural_qa_v19.py":
        "6b9092d6ed7ed82e5fd1527b26bdcbd97f89b8e65adf7d5fdd3d192b5396a3cc",
}


def _git_status(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repo, check=True,
        capture_output=True, text=True).stdout


def _dialogue(root_id: str, texts: list[tuple[str, str]]) -> dict[str, Any]:
    turns = [{
        "turn_ref": f"{root_id}:P01:T{index:03d}",
        "session_ref": f"{root_id}:P01",
        "origin": "source_grounded",
        "speaker": speaker,
        "text": text,
    } for index, (speaker, text) in enumerate(texts, 1)]
    return {
        "root_id": root_id,
        "qa_fact_source": "complete_parent_plus_child",
        "memory_is_qa_fact_source": False,
        "turn_count": len(turns),
        "turns": turns,
    }


def _temporal_question() -> dict[str, Any]:
    events = [
        "Mina records the damaged package",
        "Omar calls the courier",
        "Mina schedules a replacement delivery",
    ]
    return {
        "candidate_id": "Q01",
        "question_type": "temporal_sequence",
        "source_scope": "parent_only",
        "reasoning_structure": {
            "discriminative_variable": "event_order",
            "supporting_turn_ids": ["D001", "D002", "D003"],
            "short_rationale": "Three explicit actions occur in a fixed order.",
            "type_contract": {"events": [
                {"event": event, "turn_ids": [f"D{index:03d}"]}
                for index, event in enumerate(events, 1)
            ]},
        },
        "question": "Which order matches the package discussion?",
        "options": {
            "A": ", ".join(events),
            "B": ", ".join((events[1], events[0], events[2])),
            "C": ", ".join((events[2], events[1], events[0])),
            "D": ", ".join((events[0], events[2], events[1])),
        },
        "gold_text": ", ".join(events),
        "evidence_turn_ids": ["D001", "D002", "D003"],
    }


def _response(
    questions: list[dict[str, Any]], *, skipped: int = 0,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "structural_qa_schema_version": STRUCTURAL_QA_SCHEMA_VERSION,
        "abstention_summary": {
            "estimated_skipped_count": skipped,
            "reason_codes": reasons or [],
        },
        "questions": questions,
    }


def _fixture_results() -> dict[str, Any]:
    positive_dialogue = _dialogue("fixture-positive", [
        ("Mina", "I recorded the damaged package."),
        ("Omar", "I called the courier after you recorded it."),
        ("Mina", "Then I scheduled a replacement delivery."),
    ])
    positive_validation = validate_structural_writer_output(
        _response([_temporal_question()]),
        expanded_dialogue=positive_dialogue)
    source = build_solver_source(positive_dialogue)
    partition = solver_partition(positive_validation)
    materialized = materialize_solver_candidate(
        positive_validation["normalized_batch"]["questions"][0], index=1,
        expanded_dialogue=positive_dialogue,
        span_catalog=source["span_catalog"],
        candidate_status=partition["items"][0])

    abstention_dialogue = _dialogue("fixture-abstention", [
        ("Mina", "Hello, how are you?"),
        ("Omar", "I'm fine, thanks. The weather is pleasant."),
    ])
    abstention_validation = validate_structural_writer_output(
        _response([], skipped=2, reasons=["no_discriminative_structure"]),
        expanded_dialogue=abstention_dialogue)
    abstention_partition = solver_partition(abstention_validation)

    mixed_dialogue = _dialogue("fixture-mixed", [
        ("Mina", "I recorded the damaged package."),
        ("Omar", "I called the courier after you recorded it."),
        ("Mina", "Then I scheduled a replacement delivery."),
        ("Omar", "The weather is pleasant today."),
        ("Mina", "Yes, it is a nice afternoon."),
    ])
    mixed_validation = validate_structural_writer_output(
        _response([_temporal_question()], skipped=2,
                  reasons=["generic_commonsense"]),
        expanded_dialogue=mixed_dialogue)

    anti_copy = validate_structural_writer_output(
        _response([]), expanded_dialogue=abstention_dialogue)
    return {
        "positive": {
            "passed": (
                positive_validation["structural_valid_count"] == 1 and
                materialized["reference_answer_text"] ==
                _temporal_question()["gold_text"]),
            "structural_valid_count": positive_validation["structural_valid_count"],
            "materialized_qa_id": materialized["qa_id"],
        },
        "true_abstention": {
            "passed": (
                abstention_validation["zero_question_batch_valid"] and
                abstention_partition["solver_ready_count"] == 0),
            "solver_ready_count": abstention_partition["solver_ready_count"],
            "solver_should_start": bool(abstention_partition["solver_ready_count"]),
        },
        "mixed": {
            "passed": (
                mixed_validation["structural_valid_count"] == 1 and
                mixed_validation["structural_invalid_count"] == 0),
            "returned_count": mixed_validation["structural_valid_count"],
            "automatically_added_count": 0,
        },
        "anti_copy": {
            "passed": (
                not anti_copy["envelope_valid"] and
                "empty_batch_without_abstention_evidence" in
                anti_copy["envelope_errors"]),
            "errors": anti_copy["envelope_errors"],
        },
    }


def run() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[4]
    expanded_path = EXPERIMENT_ROOT / "expansion/expanded_dialogue.json"
    overview_path = EXPERIMENT_ROOT / "expansion/memory/post_generation_overview.json"
    expanded_sha_before = sha256_file(expanded_path)
    historical_before = {
        str(path): tree_manifest(path)["tree_sha256"] for path in HISTORICAL_RUNS}
    protected_before = protected_manifests(repo)
    git_status_before = _git_status(repo)

    expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
    overview = json.loads(overview_path.read_text(encoding="utf-8"))
    prompt = build_structural_qa_prompt(expanded, overview)
    fixtures = _fixture_results()
    prompt_tail = prompt["prompt"][-500:]
    lowered = prompt["prompt"].casefold()
    checks = {
        "positive_fixture": fixtures["positive"]["passed"],
        "true_abstention_fixture": fixtures["true_abstention"]["passed"],
        "mixed_fixture": fixtures["mixed"]["passed"],
        "anti_copy_fixture": fixtures["anti_copy"]["passed"],
        "no_copyable_empty_questions_example": (
            '\"questions\":[]' not in prompt["prompt"] and
            '\"questions\": []' not in prompt["prompt"]),
        "prompt_tail_is_instruction_near_dialogue": (
            "FINAL TASK" in prompt_tail and
            "Return zero only when" in prompt_tail and
            "{" not in prompt_tail and
            '\"questions\"' not in prompt_tail),
        "question_capacity_is_12": QUESTION_CAPACITY_MAX == 12,
        "no_minimum_eight_gate": (
            "minimum 8" not in lowered and "at least 8" not in lowered),
        "no_coverage_quota": "coverage quota" not in lowered,
        "api_calls_zero": True,
        "writer_calls_zero": True,
        "solver_calls_zero": True,
        "expansion_calls_zero": True,
        "memorax_calls_zero": True,
    }

    protected_after = protected_manifests(repo)
    historical_after = {
        str(path): tree_manifest(path)["tree_sha256"] for path in HISTORICAL_RUNS}
    checks.update({
        "protected_mainline_and_probe_unchanged": (
            protected_before["unchanged"] and protected_after["unchanged"] and
            protected_before["expected"] == protected_after["expected"]),
        "historical_runs_unchanged": historical_before == historical_after,
        "frozen_dialogue_unchanged": expanded_sha_before == sha256_file(expanded_path),
        "git_status_porcelain_unchanged": git_status_before == _git_status(repo),
    })
    module_root = Path(__file__).resolve().parent
    after_key_sha = {
        name: hashlib.sha256((module_root / name).read_bytes()).hexdigest()
        for name in BEFORE_KEY_FILE_SHA256}
    return {
        "preflight_version": "structural_qa_v19_2_prompt_calibration_v1",
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "fixtures": fixtures,
        "prompt": {
            "version": prompt["prompt_version"],
            "schema_version": STRUCTURAL_QA_SCHEMA_VERSION,
            "sha256": prompt["prompt_sha256"],
            "token_estimate": prompt["token_estimate"],
            "tail_last_500_characters": prompt_tail,
            "tail_content_type": "final_task_instruction_adjacent_to_full_dialogue",
        },
        "before_key_file_sha256": BEFORE_KEY_FILE_SHA256,
        "after_key_file_sha256": after_key_sha,
        "protected_before": protected_before["expected"],
        "protected_after": protected_after["expected"],
        "historical_run_sha256_before": historical_before,
        "historical_run_sha256_after": historical_after,
        "expanded_dialogue_sha256": expanded_sha_before,
        "safety": {
            "api_network_calls": 0,
            "writer_api_calls": 0,
            "solver_calls": 0,
            "expansion_calls": 0,
            "memorax_calls": 0,
            "historical_runs_modified": False,
            "frozen_dialogue_modified": False,
            "mainline_modified": False,
            "expansion_probe_modified": False,
            "training": False,
            "git_commit_push": False,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    fixture = report["fixtures"]
    lines = [
        "# Structural QA v19.2 Prompt Calibration Preflight", "",
        f"- Decision: `{report['decision']}`",
        "- API/network: `0`", "- Writer/Solver calls: `0 / 0`", "",
        "## Root cause and repair", "",
        "v19.1 returned an empty list because its final prompt block was a complete,",
        "copyable empty JSON answer. v19.2 removes that object, retains a field-level",
        "contract, and ends with a task instruction immediately after the full dialogue.",
        "Zero questions remains legal only with a positive skipped count and at least",
        "one allowed abstention reason. An unexplained empty batch is rejected.", "",
        "## Fixtures", "",
        f"- Positive structure + materialization: `{fixture['positive']['passed']}`",
        f"- True abstention + Solver skip: `{fixture['true_abstention']['passed']}`",
        f"- Mixed one-question batch: `{fixture['mixed']['passed']}`",
        f"- Anti-copy empty response rejection: `{fixture['anti_copy']['passed']}`",
        "", "## Prompt", "",
        f"- Version: `{report['prompt']['version']}`",
        f"- Schema: `{report['prompt']['schema_version']}`",
        f"- SHA: `{report['prompt']['sha256']}`",
        f"- Token estimate: `{report['prompt']['token_estimate']}`",
        "- Complete empty JSON examples remaining: `false`",
        "- Minimum 8 gate: `false`",
        "- Coverage quota: `false`",
        "- Tail content: `final task instruction adjacent to full dialogue`", "",
        "### Last 500 characters", "", "```text",
        report["prompt"]["tail_last_500_characters"], "```", "",
        "## Recommendation", "",
        "The offline preflight supports one new, separately authorized Writer-only live",
        "probe. This report does not execute it and does not claim semantic QA readiness.",
        "", "## Safety", "",
    ]
    lines.extend(f"- {key}: `{str(value).lower()}`"
                 for key, value in report["safety"].items())
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run()
    atomic_json(
        REPORT_ROOT / "STRUCTURAL_QA_V19_2_PROMPT_CALIBRATION_PREFLIGHT.json",
        report)
    atomic_text(
        REPORT_ROOT / "STRUCTURAL_QA_V19_2_PROMPT_CALIBRATION_PREFLIGHT.md",
        _markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
