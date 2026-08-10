"""Zero-API structural preflight for the isolated simple append branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .assembly import (
    assemble_expanded_dialogue, expanded_dialogue_markdown,
    validate_expanded_dialogue,
)
from .context import build_tail_context
from .contracts import (
    PLANNER_SCHEMA_VERSION, REALIZER_SCHEMA_VERSION, compile_round_plan,
    normalize_round_realizer_output, validate_planner_output,
)
from .prompts import (
    build_mcq_batch_prompt, build_planner_prompt, build_round_realizer_prompt,
)


PREFLIGHT_VERSION = "simple_append_v1_offline_preflight_v1"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mock_plan(speakers: list[str]) -> dict[str, Any]:
    first, second = speakers[:2]
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "Continue the latest shared concern into a concrete multi-step outcome.",
        "scenes": [{
            "goal": "Turn the parent tail's concern into a concrete next step.",
            "participants": [first, second],
            "new_development": "A practical obstacle changes what the two speakers can do next.",
            "interaction_beats": [
                "One speaker proposes a next step.",
                "The other identifies a practical obstacle.",
                "They agree on an immediate response.",
            ],
            "transition_from_previous": "Continue directly from the final parent turn.",
            "desired_outcome": "The speakers leave with a specific response to the obstacle.",
        }, {
            "goal": "Test the response under a new piece of information.",
            "participants": [first, second, "Connected Person C"],
            "new_development": "A connected person provides information that revises the response.",
            "interaction_beats": [
                "The speakers explain the current response.",
                "The connected person adds concrete information.",
                "The group revises responsibilities.",
            ],
            "transition_from_previous": "Begin with the response agreed in scene one.",
            "desired_outcome": "The revised response includes the connected person's information.",
        }, {
            "goal": "Carry out the revised response and expose a lasting consequence.",
            "participants": [first, second, "Connected Person C"],
            "new_development": "The attempted response produces a result and a new future condition.",
            "interaction_beats": [
                "The group reports what happened.",
                "They compare the result with their expectation.",
                "They decide how the consequence changes the next step.",
            ],
            "transition_from_previous": "Continue after the revised responsibilities are assigned.",
            "desired_outcome": "A concrete result and future condition are established.",
        }],
    }


def _mock_realizer(compiled: dict[str, Any]) -> dict[str, Any]:
    dialogues = {}
    for scene in compiled["scenes"]:
        participants = list(scene["participants"])
        turns = []
        for index in range(16):
            speaker = participants[index % len(participants)]
            turns.append({
                "speaker": speaker,
                "text": (f"Offline fixture {scene['scene_ref']} turn {index + 1}: "
                         f"the speakers advance {scene['new_development'].lower()}"),
            })
        dialogues[scene["scene_ref"]] = turns
    return {"realizer_schema_version": REALIZER_SCHEMA_VERSION,
            "scene_dialogues": dialogues}


def run(*, parent_path: Path) -> dict[str, Any]:
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)
    tail = build_tail_context(parent, session_count=2)
    planner = _mock_plan(list(tail["parent_speakers"]))
    planner_validation = validate_planner_output(
        planner, parent_speakers=tail["parent_speakers"])
    compiled = compile_round_plan(planner, tail_context=tail)
    realizer = _mock_realizer(compiled)
    normalized = normalize_round_realizer_output(
        realizer, compiled_plan=compiled,
        parent_utterances=[row["text"] for row in parent["dialogue"]["turns"]],
        parent_turn_refs=[row["turn_id"] for row in parent["dialogue"]["turns"]])
    if not normalized["valid"]:
        raise RuntimeError("simple_append_mock_realizer_invalid:" +
                           ",".join(normalized["errors"]))
    expanded = assemble_expanded_dialogue(parent=parent, child=normalized["child"])
    expanded_validation = validate_expanded_dialogue(expanded, parent=parent)
    planner_prompt = build_planner_prompt(tail)
    realizer_prompt = build_round_realizer_prompt(
        compiled_plan=compiled, tail_context=tail)
    mcq_prompt = build_mcq_batch_prompt(expanded)
    checks = {
        "parent_sha_preserved": _sha_bytes(parent_path.read_bytes()) == _sha_bytes(parent_bytes),
        "last_two_sessions_selected": tail["selected_session_count"] == 2,
        "parent_true_tail_selected": (
            tail["parent_tail_turn_id"] == parent["dialogue"]["turns"][-1]["turn_id"]),
        "planner_contract": planner_validation["valid"],
        "three_scene_plan": len(compiled["scenes"]) == 3,
        "tail_only_append": compiled["append_mode"] == "parent_tail_only",
        "one_round_realizer_contract": normalized["valid"],
        "forty_eight_mock_child_turns": normalized["turn_count"] == 48,
        "long_scene_target": normalized["long_scene_count"] == 3,
        "complete_dialogue_valid": expanded_validation["valid"],
        "parent_prefix_preserved": expanded["parent_turn_count"] == parent["dialogue"]["turn_count"],
        "qa_uses_complete_dialogue": expanded["qa_fact_source"] == "complete_parent_plus_child",
        "memory_not_required": not expanded["memory_is_qa_fact_source"],
        "api_calls_zero": True,
    }
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "parent_path": str(parent_path),
        "parent_file_sha256": _sha_bytes(parent_bytes),
        "root_id": parent["root_id"],
        "checks": checks,
        "tail_context": tail,
        "compiled_plan": compiled,
        "mock_child": normalized["child"],
        "expanded_dialogue": expanded,
        "expanded_dialogue_markdown": expanded_dialogue_markdown(expanded),
        "prompt_metrics": {
            "planner": {key: planner_prompt[key] for key in (
                "prompt_version", "prompt_sha256", "character_count", "token_estimate")},
            "round_realizer": {key: realizer_prompt[key] for key in (
                "prompt_version", "prompt_sha256", "character_count", "token_estimate")},
            "mcq_batch": {key: mcq_prompt[key] for key in (
                "prompt_version", "prompt_sha256", "character_count", "token_estimate")},
        },
        "physical_api_requests": 0,
        "excluded_semantics": [
            "historical_insertion", "parallel_placement", "scene_dag",
            "world_memory", "binder", "transaction", "qa_execution",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- {name}: `{str(value).lower()}`"
        for name, value in report["checks"].items())
    metrics = "\n".join(
        f"- {name}: {row['token_estimate']} estimated prompt tokens"
        for name, row in report["prompt_metrics"].items())
    return f"""# Simple Append v1 Offline Preflight

Status: `{report['status']}`

Root: `{report['root_id']}`

This report exercises only the isolated tail-only branch with deterministic
fixtures. It sends no API request and does not modify the existing v11 path.

## Checks

{checks}

## Prompt estimates

{metrics}

## Scope

- Last two complete Parent sessions are the generation context.
- Planner produces three sequential scenes.
- One Round Realizer response contains all scenes.
- Forty-eight fixture Child turns are appended after the complete Parent.
- The complete Parent+Child object is the QA source.
- Memory, Binder, transaction, and historical placement are excluded.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(parent_path=args.parent)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "SIMPLE_APPEND_V1_OFFLINE_PREFLIGHT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.report_dir / "SIMPLE_APPEND_V1_OFFLINE_PREFLIGHT.md").write_text(
        _markdown(report), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
