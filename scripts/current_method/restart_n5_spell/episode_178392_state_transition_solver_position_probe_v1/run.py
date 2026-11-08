"""Test frozen QA under balanced option-position permutations.

The original Q1-Q8 results are preserved in their own run. This probe cycles
the four option positions for every question, recomputes the gold letter from
the gold text, and runs five local Solver rollouts in each context condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    write_json,
)
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import (
    gold_letter_for_options,
    permuted_options,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    LocalSolverRunner,
    build_local_solver_runtime,
    build_solver_source,
)


PROBE_VERSION = "episode_178392_state_transition_solver_position_probe_v1"
SHIFTS = (0, 1, 2, 3)
ROLLOUTS = 5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("/root/autodl-tmp/Qwen3-4B"))
    parser.add_argument("--solver-model", type=Path, required=True)
    parser.add_argument("--expected-question-count", type=int, default=8)
    args = parser.parse_args()

    expanded = load_json(args.expanded)
    question_payload = load_json(args.questions)
    questions = question_payload.get("questions")
    if (not isinstance(questions, list) or
            len(questions) != args.expected_question_count):
        raise ValueError(f"expected_{args.expected_question_count}_frozen_questions")
    source = build_solver_source(expanded)
    prepare_empty(args.run_root)
    manifest = {
        "probe_version": PROBE_VERSION,
        "expanded_dialogue": {"path": str(args.expanded), "sha256": sha256_file(args.expanded)},
        "questions": {"path": str(args.questions), "sha256": sha256_file(args.questions)},
        "position_shifts": list(SHIFTS),
        "rollouts_per_position_per_condition": ROLLOUTS,
        "conditions": ["no_context", "full_context"],
        "original_probe_preserved": True,
        "qa_writer_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
    }
    write_json(args.run_root / "freeze_manifest.json", manifest)
    ledger = RequestLedger(args.run_root, sha256_file(args.run_root / "freeze_manifest.json"), "local")
    tokenizer, backend = build_local_solver_runtime(
        tokenizer_path=args.tokenizer, solver_model=args.solver_model)
    runner = LocalSolverRunner(
        run_root=args.run_root, ledger=ledger, tokenizer=tokenizer,
        backend=backend, dialogue=[], physical_request_budget=512)
    records: list[dict[str, Any]] = []
    try:
        for row in questions:
            base_options = {key: str(row["options"][key]) for key in "ABCD"}
            for shift in SHIFTS:
                options = permuted_options(base_options, shift)
                gold_letter = gold_letter_for_options(options, str(row["gold_text"]))
                qa = {
                    "qa_id": f"{row['qa_id']}-P{shift}",
                    "question": str(row["question"]),
                    "options": options,
                    "canonical_answer_text": str(row["gold_text"]),
                }
                condition: dict[str, Any] = {}
                for mode in ("no_context", "full_context"):
                    runner.dialogue = source["combined_dialogue"] if mode == "full_context" else []
                    rollouts = [runner.one(
                        qa=qa, options=options, mode=mode,
                        order_id=f"perm{shift}", slot=slot)
                        for slot in range(ROLLOUTS)]
                    rewards = [item.get("local_reward") for item in rollouts]
                    valid = [value for value in rewards if value in (0, 1)]
                    condition[mode] = {
                        "correct": sum(value == 1 for value in valid),
                        "valid": len(valid),
                        "accuracy": (sum(valid) / len(valid)) if valid else None,
                        "k5_stable": len(valid) == ROLLOUTS and all(value == 1 for value in valid),
                        "rollouts": rollouts,
                    }
                records.append({
                    "qa_id": row["qa_id"],
                    "position_shift": shift,
                    "gold_letter_after_shuffle": gold_letter,
                    "options": options,
                    "no_context": condition["no_context"],
                    "full_context": condition["full_context"],
                })
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    by_mode_shift: dict[str, dict[str, Any]] = {}
    by_mode_position: dict[str, dict[str, Any]] = {}
    for mode in ("no_context", "full_context"):
        by_mode_shift[mode] = {}
        position_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for shift in SHIFTS:
            cells = [r for r in records if r["position_shift"] == shift]
            correct = sum(r[mode]["correct"] for r in cells)
            total = sum(r[mode]["valid"] for r in cells)
            by_mode_shift[mode][str(shift)] = {
                "correct": correct, "total": total,
                "accuracy": correct / total if total else None,
            }
        for record in records:
            position = record["gold_letter_after_shuffle"]
            position_totals[position][0] += record[mode]["correct"]
            position_totals[position][1] += record[mode]["valid"]
        by_mode_position[mode] = {
            position: {
                "correct": values[0], "total": values[1],
                "accuracy": values[0] / values[1] if values[1] else None,
            }
            for position, values in sorted(position_totals.items())
        }
    summary = {
        "probe_version": PROBE_VERSION,
        "status": "completed",
        "question_count": len(questions),
        "position_shifts": list(SHIFTS),
        "conditions": ["no_context", "full_context"],
        "rollouts_per_position_per_condition": ROLLOUTS,
        "solver_calls_requested": len(questions) * len(SHIFTS) * 2 * ROLLOUTS,
        "solver_calls_completed": ledger.summary()["completed_request_count"],
        "no_context": {
            "aggregate_correct": sum(r["no_context"]["correct"] for r in records),
            "aggregate_total": sum(r["no_context"]["valid"] for r in records),
            "k5_stable_cells": sum(r["no_context"]["k5_stable"] for r in records),
            "by_shift": by_mode_shift["no_context"],
            "by_gold_position": by_mode_position["no_context"],
        },
        "full_context": {
            "aggregate_correct": sum(r["full_context"]["correct"] for r in records),
            "aggregate_total": sum(r["full_context"]["valid"] for r in records),
            "k5_stable_cells": sum(r["full_context"]["k5_stable"] for r in records),
            "by_shift": by_mode_shift["full_context"],
            "by_gold_position": by_mode_position["full_context"],
        },
    }
    write_json(args.run_root / "solver_position_results.json", {"records": records})
    write_json(args.run_root / "summary.json", summary)
    lines = [
        "# Episode 178392 Option-Position Solver Probe v1", "",
        f"Requests: {summary['solver_calls_completed']}; each question × shift × condition has {ROLLOUTS} rollouts.",
        "",
        "## By position shift",
        "",
        "| condition | shift 0 | shift 1 | shift 2 | shift 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("no_context", "full_context"):
        vals = [summary[mode]["by_shift"][str(shift)] for shift in SHIFTS]
        lines.append("| " + mode + " | " + " | ".join(
            f"{v['correct']}/{v['total']}" for v in vals) + " |")
    lines += ["", "## By gold answer position", "", "| condition | A | B | C | D |", "|---|---:|---:|---:|---:|"]
    for mode in ("no_context", "full_context"):
        vals = [summary[mode]["by_gold_position"].get(pos, {"correct": 0, "total": 0}) for pos in "ABCD"]
        lines.append("| " + mode + " | " + " | ".join(
            f"{v['correct']}/{v['total']}" for v in vals) + " |")
    (args.run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
