"""Zero-API preflight for the Structural QA Writer v19.1 contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .integrity import protected_manifests, tree_manifest
from .structural_contract import (
    QUESTION_CAPACITY_MAX,
    STRUCTURAL_QA_SCHEMA_VERSION,
    validate_structural_writer_output,
)
from .structural_prompts import build_structural_qa_prompt


EXPERIMENT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_experiment_1")
HISTORICAL_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "qa_structural_v19_probe_v1_live_1")
REPORT_ROOT = Path(__file__).resolve().parent / "reports"


def _envelope(questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "structural_qa_schema_version": STRUCTURAL_QA_SCHEMA_VERSION,
        "abstention_summary": {
            "estimated_skipped_count": 0,
            "reason_codes": [],
        },
        "questions": questions,
    }


def _temporal(candidate_id: str, index: int) -> dict[str, Any]:
    refs = ["D063", "D064", "D065"]
    events = [
        ("A raises the problem", refs[0]),
        ("B proposes a step", refs[1]),
        ("A accepts it", refs[2]),
    ]
    labels = [row[0] for row in events]
    return {
        "candidate_id": candidate_id,
        "question_type": "temporal_sequence",
        "source_scope": "child_only",
        "reasoning_structure": {
            "discriminative_variable": "event_order",
            "supporting_turn_ids": refs,
            "short_rationale": "The order is stated by three distinct turns.",
            "type_contract": {
                "events": [
                    {"event": label, "turn_ids": [ref]}
                    for label, ref in events
                ]
            },
        },
        "question": f"Which order matches development {index}?",
        "options": {
            "A": ", ".join(labels),
            "B": ", ".join((labels[1], labels[0], labels[2])),
            "C": ", ".join((labels[2], labels[1], labels[0])),
            "D": ", ".join((labels[0], labels[2], labels[1])),
        },
        "gold_text": ", ".join(labels),
        "evidence_turn_ids": refs,
    }


def run() -> dict[str, Any]:
    expanded_path = EXPERIMENT_ROOT / "expansion/expanded_dialogue.json"
    overview_path = EXPERIMENT_ROOT / "expansion/memory/post_generation_overview.json"
    raw_path = HISTORICAL_RUN / "qa/writer_output_raw.json"
    expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
    overview = json.loads(overview_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_sha_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    historical_tree_before = tree_manifest(HISTORICAL_RUN)["tree_sha256"]
    prompt = build_structural_qa_prompt(expanded, overview)

    capacity_results: dict[str, bool] = {}
    for count in (0, 1, 3, QUESTION_CAPACITY_MAX):
        rows = [_temporal(f"Q{index:02d}", index) for index in range(1, count + 1)]
        result = validate_structural_writer_output(
            _envelope(rows), expanded_dialogue=expanded)
        capacity_results[str(count)] = (
            result["envelope_valid"] and
            result["structural_valid_count"] == count and
            result["structural_invalid_count"] == 0)

    shadow = copy.deepcopy(raw)
    shadow["structural_qa_schema_version"] = STRUCTURAL_QA_SCHEMA_VERSION
    shadow_result = validate_structural_writer_output(
        shadow, expanded_dialogue=expanded)
    frozen = json.loads((HISTORICAL_RUN / "qa/writer_structural_validation.json").read_text())
    invalid = [row for row in frozen["items"] if not row["valid"]]
    error_counts: dict[str, int] = {}
    for row in invalid:
        for error in row["structural_errors"]:
            error_counts[error] = error_counts.get(error, 0) + 1

    repo = Path(__file__).resolve().parents[4]
    historical_sha_after = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    historical_tree_after = tree_manifest(HISTORICAL_RUN)["tree_sha256"]
    checks = {
        "api_calls_zero": True,
        "historical_raw_sha_unchanged": raw_sha_before == historical_sha_after,
        "historical_quarantine_count_is_9": len(invalid) == 9,
        "capacity_0_1_3_12_all_valid": all(capacity_results.values()),
        "prompt_has_no_quota_language": (
            "as many" not in prompt["prompt"].casefold() and
            "fill the quota" not in prompt["prompt"].casefold() and
            "capacity ceiling, never a target" in prompt["prompt"]),
        "shadow_q07_visibility_gate": any(
            row["candidate_id"] == "Q07" and
            "visibility_question_not_knowledge_only" in row["structural_errors"]
            for row in shadow_result["items"]),
        "protected_modules_unchanged": protected_manifests(repo)["unchanged"],
        "historical_run_tree_unchanged": historical_tree_before == historical_tree_after,
    }
    return {
        "preflight_version": "structural_qa_writer_v19_1_preflight_v1",
        "decision": "pass" if all(checks.values()) else "fail",
        "api_calls": 0,
        "checks": checks,
        "capacity_results": capacity_results,
        "historical_error_counts": error_counts,
        "historical_v19_raw_sha256": raw_sha_before,
        "historical_v19_tree_sha256": historical_tree_before,
        "shadow_v19_1_structural_valid_count": shadow_result["structural_valid_count"],
        "shadow_v19_1_structural_invalid_count": shadow_result["structural_invalid_count"],
        "prompt_sha256": prompt["prompt_sha256"],
        "prompt_token_estimate": prompt["token_estimate"],
        "question_capacity_max": QUESTION_CAPACITY_MAX,
        "historical_run": str(HISTORICAL_RUN),
    }


def main() -> int:
    report = run()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "STRUCTURAL_QA_V19_1_PREFLIGHT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Structural QA v19.1 Preflight", "",
        f"- decision: `{report['decision']}`",
        f"- API calls: `{report['api_calls']}`",
        f"- historical v19 raw SHA: `{report['historical_v19_raw_sha256']}`",
        f"- prompt SHA: `{report['prompt_sha256']}`",
        f"- prompt token estimate: `{report['prompt_token_estimate']}`",
        "", "## Checks", "",
    ]
    lines.extend(f"- {key}: `{str(value).lower()}`"
                 for key, value in report["checks"].items())
    lines.extend(["", "## Capacity Fixtures", ""])
    lines.extend(f"- {key} questions: `{str(value).lower()}`"
                 for key, value in report["capacity_results"].items())
    lines.extend(["", "No API, run creation, or historical write occurred.", ""])
    (REPORT_ROOT / "STRUCTURAL_QA_V19_1_PREFLIGHT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
