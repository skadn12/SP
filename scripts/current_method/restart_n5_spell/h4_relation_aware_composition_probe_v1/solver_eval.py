"""Local Solver evaluation with contribution-level evidence ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.core import sha256_file
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import (
    classify_balanced_no_context, classify_full_context_frontier,
    gold_letter_for_options, permuted_options,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    LocalSolverRunner, build_local_solver_runtime, build_solver_source,
)


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    rewards = [row.get("local_reward") for row in rows if row.get("local_reward") in (0, 1)]
    return sum(rewards) / len(rewards) if rewards else None


def _view(*, runner: LocalSolverRunner, qa: Mapping[str, Any], options: Mapping[str, str],
          dialogue: list[dict[str, Any]], label: str, slots: int = 5) -> dict[str, Any]:
    runner.dialogue = [dict(row) for row in dialogue]
    rows = [runner.one(qa=qa, options=dict(options), mode="full_context",
                       order_id=label, slot=slot) for slot in range(slots)]
    rewards = [row.get("local_reward") for row in rows]
    return {
        "label": label, "accuracy": _accuracy(rows),
        "correct": sum(value == 1 for value in rewards), "total": len(rows),
        "k5": classify_full_context_frontier(rewards), "rollouts": rows,
    }


def evaluate_candidate(*, composite: Mapping[str, Any], candidate: Mapping[str, Any],
                       expanded: Mapping[str, Any], run_root: Path,
                       tokenizer_path: Path, solver_model: Path,
                       freeze_manifest: Path | None = None) -> dict[str, Any]:
    source = build_solver_source(expanded)
    manifest = freeze_manifest or (run_root / "run_freeze_manifest.json")
    if not manifest.exists():
        raise FileNotFoundError(f"freeze manifest not found: {manifest}")
    ledger = RequestLedger(run_root, sha256_file(manifest), "local")
    tokenizer, solver = build_local_solver_runtime(
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    runner = LocalSolverRunner(
        run_root=run_root, ledger=ledger, tokenizer=tokenizer, backend=solver,
        dialogue=source["combined_dialogue"], physical_request_budget=128)
    qa = {
        "qa_id": f"composite-{candidate['candidate_id']}",
        "question": composite["question"],
        "canonical_answer_text": composite["gold_text"],
        "options": {key: str(composite["options"][key]) for key in "ABCD"},
    }
    options = qa["options"]
    no_context_records: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    runner.dialogue = []
    for shift, order_id in enumerate(("balanced_original", "balanced_perm1", "balanced_perm2", "balanced_perm3")):
        ordered = options if shift == 0 else permuted_options(options, shift)
        gold = gold_letter_for_options(ordered, qa["canonical_answer_text"])
        row = runner.one(qa=qa, options=ordered, mode="no_context", order_id=order_id, slot=0)
        no_context_records.append({"order_id": order_id, "options": ordered,
                                   "gold_letter": gold, "rollouts": [row]})
        orders.append({"order_id": order_id, "gold_letter": gold,
                       "valid_answer_letters": [row.get("parsed_letter")],
                       "valid_rewards": [row.get("local_reward")]})
    grounding = classify_balanced_no_context(orders, samples_per_position=1)
    full = _view(runner=runner, qa=qa, options=options,
                 dialogue=source["combined_dialogue"], label="full_context")
    ablations: dict[str, dict[str, Any]] = {}
    closure = candidate.get("unique_contribution_closure") or {}
    for qid in candidate["source_qids"]:
        removed = {str(ref) for ref in (closure.get(qid) or [])}
        dialogue = [row for row in source["combined_dialogue"]
                    if str(row.get("turn_id")) not in removed]
        ablations[f"remove_{qid}"] = _view(
            runner=runner, qa=qa, options=options, dialogue=dialogue,
            label=f"remove_{qid}")
    no_context_rate = grounding.get("semantic_correct_rate")
    full_pass = full["correct"] == 5 and full["total"] == 5
    role_pass = all(
        (view.get("accuracy") is None or view.get("accuracy") < 1.0)
        for view in ablations.values())
    necessity_pass = bool(full_pass and (no_context_rate is None or no_context_rate < 1.0) and role_pass)
    result = {
        "candidate_id": candidate["candidate_id"],
        "source_qids": candidate["source_qids"],
        "closure_refs": closure,
        "grounding": grounding,
        "no_context_records": no_context_records,
        "full_context": full,
        "ablations": ablations,
        "full_context_accuracy": full["accuracy"],
        "no_context_accuracy": no_context_rate,
        "necessity_pass": necessity_pass,
        "admission_rule": "full K5 pass AND no-context not all-correct AND every source contribution ablation not all-correct",
    }
    (run_root / "solver_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
