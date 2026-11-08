#!/usr/bin/env python3
"""Run the admitted first-root QA through the existing local Solver/score path."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/current_method"))
import diagnostics_context_ablation as evaluator  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_turns(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"turn_id": t["turn_ref"], "speaker": t["speaker"], "text": t["text"]}
        for t in source["turns"]
    ]


def build_sample(
    *, sample_id: str, candidate: dict[str, Any], turns: list[dict[str, Any]],
    option_order: list[str], mode: str,
) -> dict[str, Any]:
    key_to_label = {candidate_id: chr(ord("A") + i) for i, candidate_id in enumerate(option_order)}
    options = [{"id": key_to_label[key], "text": candidate["options"][key]} for key in option_order]
    gold = [key_to_label[candidate["gold"][0]]]
    return {
        "sample_id": sample_id,
        "attack_plan": {"attack_skill": "first_root_real_qa"},
        "experiment": {
            "context_mode": mode,
            "option_order_candidate_ids": option_order,
            "option_label_to_candidate_id": {label: key for key, label in key_to_label.items()},
            "gold_candidate_id": candidate["gold"][0],
        },
        "dialogue": {"turns": turns},
        "qa": {
            "question_type": candidate["question_type"],
            "question": candidate["question"],
            "options": options,
            "gold": gold,
            "evidence_turn_ids": [],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    state = args.run_dir / "state"
    decision = read_json(state / "DECISION.json")
    frozen = read_json(state / "FROZEN_QA.json")
    candidate = read_json(args.candidate)
    relation = frozen["relation"]
    source = read_json(args.source)
    if decision.get("decision") != "PASS_STATIC_REVIEW" or not decision.get("admitted"):
        raise SystemExit("Refusing Solver evaluation: V1 static review did not admit the candidate")
    if candidate.get("candidate_id") != frozen["candidate"]["candidate_id"]:
        raise SystemExit("Candidate differs from the statically admitted candidate")
    if source.get("root_id") != candidate.get("root_id"):
        raise SystemExit("Source root does not match the candidate")

    turns = source_turns(source)
    option_keys = list(candidate["options"])
    blind_packet = read_json(state / "BLIND_PACKET.json")
    public_order = list(blind_packet["options"])
    if set(public_order) != set(option_keys):
        raise SystemExit("Blind display option set differs from the candidate")

    cited_evidence_refs = {
        evidence["turn_ref"]
        for item in candidate["item_analysis"].values()
        for evidence in item.get("evidence", [])
    }
    necessary_refs = {
        turn_ref
        for group in relation["necessity_groups"]
        for turn_ref in group["turn_refs"]
    }
    evidence_refs = cited_evidence_refs | necessary_refs
    evidence_turns = [t for t in turns if t["turn_id"] in evidence_refs]
    if {t["turn_id"] for t in evidence_turns} != evidence_refs:
        raise SystemExit("Candidate evidence refs are not all present in the bound source")

    samples = [
        build_sample(sample_id="full_public_order", candidate=candidate, turns=turns,
                     option_order=public_order, mode="full"),
        build_sample(sample_id="no_context", candidate=candidate, turns=[],
                     option_order=public_order, mode="no_context"),
        build_sample(sample_id="evidence_only", candidate=candidate, turns=evidence_turns,
                     option_order=public_order, mode="evidence_only_cited_plus_necessary_evidence"),
    ]
    for group in relation["necessity_groups"]:
        removed = set(group["turn_refs"])
        kept = [turn for turn in turns if turn["turn_id"] not in removed]
        samples.append(build_sample(
            sample_id=f"ablate_{group['group_id']}", candidate=candidate, turns=kept,
            option_order=public_order, mode=f"full_minus_{group['group_id']}",
        ))

    # All 24 mappings are evaluated on full context; labels are reassigned by position.
    for index, order in enumerate(itertools.permutations(option_keys)):
        samples.append(build_sample(
            sample_id=f"permutation_{index:02d}", candidate=candidate, turns=turns,
            option_order=list(order), mode="full_option_permutation",
        ))

    output_dir = args.run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.write_jsonl(output_dir / "solver_samples.jsonl", samples)
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "solver_model": "/root/autodl-tmp/Qwen3-4B",
        "base_model": "/root/autodl-tmp/Qwen3-4B",
        "model_max_position_embeddings": 40960,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "k_eval": 1,
    }
    solver = evaluator.SharedSolver(
        solver_model=env["solver_model"], base_model=env["base_model"],
        max_model_len=env["max_model_len"],
        gpu_memory_utilization=env["gpu_memory_utilization"], seed=env["seed"],
    )
    rows = solver.evaluate(samples, k_eval=1, max_new_tokens=8, temperature=0.0, top_p=1.0)
    health = evaluator.solver_eval_health(samples, k_eval=1)

    output_rows = []
    for sample in samples:
        se = sample["solver_eval"]
        prediction = se["predictions"][0]
        label = str(prediction.get("parsed") or "")
        semantic_id = sample["experiment"]["option_label_to_candidate_id"].get(label)
        if not prediction["prediction"].strip() or not label:
            error_class = "invalid_output"
        elif sample["experiment"]["context_mode"] == "no_context":
            error_class = (
                "no_context_control_hit_possible_option_only_guess"
                if prediction["exact_correct"] else "no_context_control_guess_incorrect"
            )
        elif prediction["exact_correct"]:
            error_class = "correct"
        elif sample["experiment"]["context_mode"] == "full_option_permutation":
            error_class = "answer_selection_error_under_permutation"
        elif sample["experiment"]["context_mode"].startswith("full_minus_"):
            error_class = "ablated_context_answer_incorrect"
        else:
            error_class = "full_context_answer_selection_error"
        output_rows.append({
            "sample_id": sample["sample_id"],
            "context_mode": sample["experiment"]["context_mode"],
            "turn_count": len(sample["dialogue"]["turns"]),
            "option_order_candidate_ids": sample["experiment"]["option_order_candidate_ids"],
            "option_label_to_candidate_id": sample["experiment"]["option_label_to_candidate_id"],
            "raw_prediction": prediction["prediction"],
            "parsed_label": label,
            "predicted_candidate_id": semantic_id,
            "gold_label": sample["qa"]["gold"][0],
            "gold_candidate_id": sample["experiment"]["gold_candidate_id"],
            "exact_correct": prediction["exact_correct"],
            "partial_credit": prediction["partial_credit"],
            "error_class": error_class,
            "prompt_token_estimate": se["diagnostics"]["prompt_token_estimate"],
            "eval_context_overflow": se["diagnostics"]["eval_context_overflow"],
        })

    perms = [row for row in output_rows if row["context_mode"] == "full_option_permutation"]
    valid_semantics = [r["predicted_candidate_id"] for r in perms if r["predicted_candidate_id"]]
    semantic_counts = {key: valid_semantics.count(key) for key in sorted(set(valid_semantics))}
    summary = {
        "candidate_id": candidate["candidate_id"],
        "source_sha256": sha256(args.source),
        "candidate_sha256": json_hash(candidate),
        "review_status": decision["decision"],
        "reviewer_mode": "Codex same-model independent-context blind review",
        "model": env,
        "sample_count": len(samples),
        "partial_score_note": "For single-choice, partial credit equals exact whole-item accuracy (0 or 1).",
        "health": health,
        "permutation_summary": {
            "count": len(perms),
            "exact_correct": sum(bool(r["exact_correct"]) for r in perms),
            "exact_accuracy": sum(bool(r["exact_correct"]) for r in perms) / max(1, len(perms)),
            "valid_prediction_count": len(valid_semantics),
            "predicted_semantic_option_counts": semantic_counts,
            "position_sensitivity_detected": len(semantic_counts) > 1,
        },
        "evidence_only_turn_ids": [t["turn_id"] for t in evidence_turns],
        "evidence_only_turn_set_basis": "union of candidate per-option cited evidence and every relation necessity-group turn",
        "results": output_rows,
        "interpretation_policy": {
            "invalid_output_is_not_a_reasoning_error": True,
            "no_context_is_a_control_not_a_solver_weakness": True,
            "ablation_failure_is_not_a_full_context_solver_weakness": True,
            "only_a_valid_wrong_full_context_answer_on_this_admitted_item_can_be_called_a_solver_answer_error": True,
        },
    }
    write_json(output_dir / "solver_results.json", summary)
    evaluator.write_jsonl(output_dir / "solver_eval_rows.jsonl", rows)
    print(json.dumps({
        "model_loaded": True,
        "sample_count": len(samples),
        "health": health,
        "full": next(r for r in output_rows if r["sample_id"] == "full_public_order"),
        "no_context": next(r for r in output_rows if r["sample_id"] == "no_context"),
        "evidence_only": next(r for r in output_rows if r["sample_id"] == "evidence_only"),
        "ablations": [r for r in output_rows if r["sample_id"].startswith("ablate_")],
        "permutation_summary": summary["permutation_summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
