"""Evaluate frozen episode-178392 state-transition questions with local Solver.

This probe does not call a QA Writer and does not modify the production QA
pipeline. It sends exactly five requested Solver rollouts per question in each
of two conditions: an empty context and the complete frozen Parent+Child
dialogue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    write_json,
)
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    LocalSolverRunner,
    build_local_solver_runtime,
    build_solver_source,
)


PROBE_VERSION = "episode_178392_state_transition_solver_probe_v1"
ROLLOUTS = 5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def validate_questions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) != 8:
        raise ValueError("expected_exactly_8_questions")
    required = {"qa_id", "question", "options", "gold_letter", "gold_text",
                "required_evidence", "reasoning", "expected_profile"}
    out = []
    seen: set[str] = set()
    for row in questions:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("question_schema_invalid")
        qa_id = str(row["qa_id"])
        if qa_id in seen or set(row["options"]) != set("ABCD"):
            raise ValueError(f"question_identity_or_options_invalid:{qa_id}")
        if row["gold_letter"] not in "ABCD":
            raise ValueError(f"gold_letter_invalid:{qa_id}")
        if row["options"][row["gold_letter"]] != row["gold_text"]:
            raise ValueError(f"gold_text_mismatch:{qa_id}")
        if not row["required_evidence"]:
            raise ValueError(f"evidence_missing:{qa_id}")
        seen.add(qa_id)
        out.append(row)
    return out


def solver_qa(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": row["qa_id"],
        "question": row["question"],
        "options": {key: str(row["options"][key]) for key in "ABCD"},
        "canonical_answer_text": row["gold_text"],
        "gold_text": row["gold_text"],
    }


def run(*, expanded_path: Path, questions_path: Path, run_root: Path,
        tokenizer_path: Path, solver_model: Path) -> dict[str, Any]:
    expanded = load_json(expanded_path)
    questions_payload = load_json(questions_path)
    questions = validate_questions(questions_payload)
    source = build_solver_source(expanded)
    prepare_empty(run_root)
    manifest = {
        "probe_version": PROBE_VERSION,
        "expanded_dialogue": {"path": str(expanded_path), "sha256": sha256_file(expanded_path)},
        "questions": {"path": str(questions_path), "sha256": sha256_file(questions_path)},
        "source_root_id": expanded.get("root_id"),
        "parent_turn_count": expanded.get("parent_turn_count"),
        "child_turn_count": expanded.get("child_turn_count"),
        "full_context_turn_count": len(source["combined_dialogue"]),
        "conditions": ["no_context", "full_context"],
        "rollouts_per_question_per_condition": ROLLOUTS,
        "qa_writer_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
    }
    write_json(run_root / "freeze_manifest.json", manifest)
    write_json(run_root / "solver_input_questions.json", {
        "probe_version": PROBE_VERSION,
        "conditions": ["no_context", "full_context"],
        "rollouts_per_condition": ROLLOUTS,
        "questions": [solver_qa(row) | {
            "gold_letter": row["gold_letter"],
            "required_evidence": row["required_evidence"],
        } for row in questions],
    })
    ledger = RequestLedger(run_root, sha256_file(run_root / "freeze_manifest.json"), "local")
    tokenizer, backend = build_local_solver_runtime(
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    runner = LocalSolverRunner(
        run_root=run_root, ledger=ledger, tokenizer=tokenizer, backend=backend,
        dialogue=[], physical_request_budget=128)
    results: list[dict[str, Any]] = []
    try:
        for row in questions:
            qa = solver_qa(row)
            condition_results: dict[str, Any] = {}
            for mode in ("no_context", "full_context"):
                runner.dialogue = (source["combined_dialogue"]
                                   if mode == "full_context" else [])
                rollouts = [runner.one(
                    qa=qa, options=qa["options"], mode=mode,
                    order_id="original", slot=slot)
                    for slot in range(ROLLOUTS)]
                rewards = [item.get("local_reward") for item in rollouts]
                valid = [value for value in rewards if value in (0, 1)]
                condition_results[mode] = {
                    "requested_rollouts": ROLLOUTS,
                    "rollouts": rollouts,
                    "correct": sum(value == 1 for value in valid),
                    "valid": len(valid),
                    "accuracy": (sum(valid) / len(valid)) if valid else None,
                    "k5_stable": len(valid) == ROLLOUTS and all(value == 1 for value in valid),
                }
            results.append({
                "qa_id": row["qa_id"],
                "question_type": row.get("question_type"),
                "batch": row.get("batch"),
                "expected_profile": row.get("expected_profile"),
                "gold_letter": row["gold_letter"],
                "required_evidence": row["required_evidence"],
                "no_context": condition_results["no_context"],
                "full_context": condition_results["full_context"],
            })
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    summary = {
        "probe_version": PROBE_VERSION,
        "status": "completed",
        "question_count": len(results),
        "conditions": ["no_context", "full_context"],
        "rollouts_per_question_per_condition": ROLLOUTS,
        "solver_calls_requested": len(results) * 2 * ROLLOUTS,
        "solver_calls_completed": ledger.summary()["completed_request_count"],
        "full_context_k5_stable": sum(bool(row["full_context"]["k5_stable"]) for row in results),
        "no_context_k5_stable": sum(bool(row["no_context"]["k5_stable"]) for row in results),
        "results": results,
    }
    write_json(run_root / "solver_results.json", {"questions": results})
    write_json(run_root / "summary.json", summary)
    lines = [
        "# Episode 178392 State-Transition Solver Probe v1",
        "",
        f"Questions: {len(results)}; requested Solver rollouts: {len(results) * 2 * ROLLOUTS}.",
        f"No-context K=5 stable: {summary['no_context_k5_stable']}/{len(results)}.",
        f"Full-context K=5 stable: {summary['full_context_k5_stable']}/{len(results)}.",
        "",
        "| QA | no-context | full-context |",
        "|---|---:|---:|",
    ]
    for row in results:
        lines.append(f"| {row['qa_id']} | {row['no_context']['correct']}/{row['no_context']['valid']} | {row['full_context']['correct']}/{row['full_context']['valid']} |")
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("/root/autodl-tmp/Qwen3-4B"))
    parser.add_argument("--solver-model", type=Path, required=True)
    args = parser.parse_args()
    summary = run(expanded_path=args.expanded, questions_path=args.questions,
                  run_root=args.run_root, tokenizer_path=args.tokenizer,
                  solver_model=args.solver_model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
