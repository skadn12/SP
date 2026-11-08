"""Isolated structured QA probe for multi-select and ordering questions.

This adapter is an isolated structured Solver probe.  It keeps the production
single-choice QA path untouched while using one strict ``<box>`` protocol for
multi-select and ordering replays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from scripts.current_method.restart_n5_spell.backends import BackendResponse, SolverRequest
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_bytes, sha256_file
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    SOLVER_GPU_MEMORY_UTILIZATION,
    SOLVER_MAX_MODEL_LEN,
)
from scripts.current_method.restart_n5_spell.structured_qa_box_protocol import (
    STRUCTURED_SOLVER_MAX_TOKENS,
    STRUCTURED_SOLVER_TEMPERATURE,
    STRUCTURED_GUIDED_DECODING_VERSION,
    classify_structured_result,
    parse_structured_box,
    structured_box_choices,
)
from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend


MSQ_SHIFTS = (0, 1, 2, 3, 4)
ORDER_SHIFTS = (0, 1, 2, 3)
ROLLOUTS = 5

STRUCTURED_SYSTEM_PROMPT = (
    "You are a strict structured dialogue QA solver. Use only the supplied "
    "dialogue when it is present. Return the final answer in exactly one "
    "<box>...</box> pair. Do not provide explanations, reasoning, or any "
    "text outside the box."
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def rotate_mapping(mapping: Mapping[str, str], shift: int) -> dict[str, str]:
    keys = list(mapping.keys())
    values = [str(mapping[key]) for key in keys]
    return {key: values[(index - shift) % len(keys)] for index, key in enumerate(keys)}


def rebound_gold(original: Mapping[str, str], shuffled: Mapping[str, str], gold: list[str]) -> list[str]:
    gold_texts = {str(original[key]) for key in gold}
    rebound = [key for key, value in shuffled.items() if str(value) in gold_texts]
    if len(rebound) != len(gold):
        raise ValueError("multi_select_gold_rebinding_failed")
    return rebound


def shuffled_question(row: Mapping[str, Any], shift: int) -> tuple[dict[str, Any], list[str]]:
    q = dict(row)
    if row["question_type"] == "multi_select":
        original = {str(k): str(v) for k, v in row["options"].items()}
        shuffled = rotate_mapping(original, shift)
        gold = rebound_gold(original, shuffled, [str(x) for x in row["gold"]])
        q["options"] = shuffled
        q["gold"] = gold
        return q, list(shuffled.keys())
    original = {str(k): str(v) for k, v in row["items"].items()}
    keys = list(original.keys())
    display_order = keys[shift:] + keys[:shift]
    q["items"] = {key: original[key] for key in display_order}
    q["gold"] = [str(x) for x in row["gold"]]
    return q, display_order


def _build_structured_user_content(
    q: Mapping[str, Any], dialogue: list[dict[str, Any]] | None,
) -> str:
    chunks: list[str] = []
    if dialogue is not None:
        chunks.append("Dialogue:\n" + "\n".join(
            f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in dialogue
        ))
    chunks.append(f"Question ({q['question_type']}):\n{q['question']}")
    if q["question_type"] == "multi_select":
        chunks.append("Options:\n" + "\n".join(
            f"{key}. {value}" for key, value in q["options"].items()
        ))
        chunks.append(
            "For this multi-select question, output only the selected option "
            "letters separated by commas. Example: <box>A,B,C</box>."
        )
    else:
        chunks.append("Items:\n" + "\n".join(
            f"{key}. {value}" for key, value in q["items"].items()
        ))
        chunks.append(
            "For this ordering question, output only the ordered item IDs "
            "separated by commas. Example: <box>I2,I4,I1,I3</box>."
        )
    return "\n\n".join(chunks)


def build_prompt(q: Mapping[str, Any], dialogue: list[dict[str, Any]] | None) -> str:
    """Return the readable prompt form used for audit/debug records."""
    return STRUCTURED_SYSTEM_PROMPT + "\n\n" + _build_structured_user_content(q, dialogue)


def build_chat_prompt(
    tokenizer: Any, q: Mapping[str, Any], dialogue: list[dict[str, Any]] | None,
) -> str:
    """Render the structured prompt with Qwen's actual chat template.

    ``enable_thinking=False`` must be passed to the tokenizer; recording it in
    metadata after raw generation does not disable Qwen's reasoning format.
    """
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": _build_structured_user_content(q, dialogue)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_structured(
    backend: VLLMSolverBackend, request: SolverRequest, *, question_type: str,
    sampling_temperature: float,
) -> BackendResponse:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    allowed_item_ids = request.options.keys() if question_type in {"multi_select", "ordering"} else None
    choices = structured_box_choices(
        question_type, allowed_item_ids=allowed_item_ids)
    if not choices:
        raise ValueError(f"structured_guided_choices_empty:{question_type}")

    params = SamplingParams(
        temperature=sampling_temperature,
        top_p=backend.top_p,
        max_tokens=STRUCTURED_SOLVER_MAX_TOKENS,
        seed=request.sampling_seed,
        stop=["</box>"],
        include_stop_str_in_output=True,
        guided_decoding=GuidedDecodingParams(choice=choices),
    )
    started = time.perf_counter()
    outputs = backend.engine.generate([request.prompt], params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    metadata = {
        "sampling_seed": request.sampling_seed,
        "latency_seconds": time.perf_counter() - started,
        "model_load_latency_seconds": backend.load_latency_seconds,
        "prompt_token_count": len(outputs[0].prompt_token_ids),
        "completion_token_count": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "max_model_len": backend.max_model_len,
        "temperature": backend.temperature,
        "top_p": backend.top_p,
        "max_tokens": STRUCTURED_SOLVER_MAX_TOKENS,
        "enable_thinking": False,
        "stop_sequences": ["</box>"],
    }
    return BackendResponse(completion.text, backend.model_identifier, metadata)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument(
        "--temperature", type=float, default=STRUCTURED_SOLVER_TEMPERATURE,
        help="structured Solver sampling temperature; default keeps deterministic replay",
    )
    parser.add_argument(
        "--permutation-count", type=int, default=0,
        help="limit each question to the first N deterministic permutations; 0 uses all",
    )
    args = parser.parse_args()
    if args.temperature < 0:
        raise ValueError("temperature_must_be_nonnegative")
    if args.permutation_count < 0:
        raise ValueError("permutation_count_must_be_nonnegative")
    if args.run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{args.run_root}")

    expanded = load_json(args.expanded)
    question_payload = load_json(args.questions)
    questions = question_payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("expected_nonempty_structured_questions")
    shifts_by_type = {
        "multi_select": MSQ_SHIFTS[:args.permutation_count] if args.permutation_count else MSQ_SHIFTS,
        "ordering": ORDER_SHIFTS[:args.permutation_count] if args.permutation_count else ORDER_SHIFTS,
    }
    required_solver_calls = sum(
        len(shifts_by_type[q["question_type"]]) * 2 * ROLLOUTS for q in questions
    )
    turns = [
        {"turn_id": row["turn_ref"], "speaker": row["speaker"], "text": row["text"]}
        for row in expanded["turns"]
    ]
    args.run_root.mkdir(parents=True, exist_ok=False)
    for rel in ("raw/prompts", "raw/backend_responses", "reports"):
        (args.run_root / rel).mkdir(parents=True, exist_ok=True)
    manifest = {
        "probe_version": "episode_178392_structured_qa_position_probe_v1",
        "source_dialogue": {"path": str(args.expanded), "sha256": sha256_file(args.expanded), "turn_count": len(turns)},
        "questions": {"path": str(args.questions), "sha256": sha256_file(args.questions), "count": len(questions)},
        "multi_select_shifts": list(MSQ_SHIFTS),
        "ordering_display_shifts": list(ORDER_SHIFTS),
        "rollouts_per_shift_per_context": ROLLOUTS,
        "structured_solver_max_tokens": STRUCTURED_SOLVER_MAX_TOKENS,
        "structured_solver_temperature": args.temperature,
        "permutation_count": args.permutation_count or None,
        "required_solver_calls": required_solver_calls,
        "boxed_protocol_version": "structured_qa_box_v1",
        "structured_guided_decoding": STRUCTURED_GUIDED_DECODING_VERSION,
        "conditions": ["no_context", "full_context"],
        "writer_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
    }
    atomic_json(args.run_root / "run_freeze_manifest.json", manifest)
    ledger = RequestLedger(args.run_root, sha256_file(args.run_root / "run_freeze_manifest.json"), "local")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    backend = VLLMSolverBackend(
        model_path=args.solver_model, tokenizer_path=args.tokenizer,
        max_model_len=16384, temperature=args.temperature, top_p=0.9,
        max_tokens=STRUCTURED_SOLVER_MAX_TOKENS,
        gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION,
    )
    records: list[dict[str, Any]] = []
    try:
        for row in questions:
            shifts = shifts_by_type[row["question_type"]]
            for shift in shifts:
                q, display_order = shuffled_question(row, shift)
                for mode, dialogue in (("no_context", None), ("full_context", turns)):
                    rollouts: list[dict[str, Any]] = []
                    prompt = build_chat_prompt(tokenizer, q, dialogue)
                    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    if prompt_tokens + STRUCTURED_SOLVER_MAX_TOKENS > 16384:
                        raise RuntimeError(f"prompt_overflow:{row['qa_id']}:{mode}:{prompt_tokens}")
                    for slot in range(ROLLOUTS):
                        request_id = f"structured-{row['qa_id']}-shift{shift}-{mode}-s{slot:02d}"
                        seed = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16)
                        prompt_sha = sha256_bytes(prompt.encode())
                        atomic_json(args.run_root / f"raw/prompts/{request_id}.json", {
                            "request_id": request_id, "prompt": prompt,
                            "qa_id": row["qa_id"], "shift": shift, "mode": mode,
                            "display_order": display_order, "gold_after_rebinding": q["gold"],
                        })
                        request = SolverRequest(
                            request_id=request_id, qa_id=row["qa_id"],
                            evaluation_mode=mode, order_id=f"shift{shift}", target_slot=slot,
                            retry_index=0, sampling_seed=seed,
                            options=q.get("options") or q.get("items") or {},
                            canonical_answer_text=",".join(q["gold"]),
                            prompt_sha256=prompt_sha, prompt=prompt,
                        )
                        response = ledger.invoke(
                            request_id=request_id, intent=f"structured_solver_{mode}", attempt=1,
                            prompt_sha256=prompt_sha,
                            call=lambda request=request: generate_structured(
                                backend, request, question_type=row["question_type"],
                                sampling_temperature=args.temperature),
                            budget_class="structured_solver", budget_limit=required_solver_calls,
                            total_budget_limit=required_solver_calls,
                            request_metadata={"qa_id": row["qa_id"], "mode": mode, "shift": shift, "slot": slot},
                            backend_identity={"backend": "VLLMSolverBackend", "model": backend.model_identifier},
                        )
                        raw = response.raw_text
                        parsed = parse_structured_box(
                            raw,
                            question_type=row["question_type"],
                            allowed_item_ids=(q.get("options") or q.get("items") or {}).keys(),
                        )
                        result = classify_structured_result(
                            parsed,
                            gold=[str(x) for x in q["gold"]],
                            question_type=row["question_type"],
                        )
                        metrics = result.get("metrics") or {}
                        exact = bool(metrics.get(
                            "exact_match" if row["question_type"] == "multi_select" else "exact_order_match",
                            False,
                        ))
                        partial = float(metrics.get(
                            "jaccard" if row["question_type"] == "multi_select" else "pairwise_accuracy",
                            0.0,
                        ))
                        ledger.record_decision(request_id, result["failure_type"] or "exact_match")
                        rollouts.append({
                            "request_id": request_id,
                            "raw_output": raw,
                            "raw_response": raw,
                            "parsed_answer": result["parsed_answer"],
                            "parse_status": result["parse_status"],
                            "failure_type": result["failure_type"],
                            "parse_reason": result.get("parse_reason"),
                            "metrics": metrics,
                            "gold_after_rebinding": q["gold"],
                            "exact_correct": exact, "partial_score": partial,
                            "backend_metadata": dict(response.metadata),
                        })
                    records.append({
                        "qa_id": row["qa_id"], "question_type": row["question_type"],
                        "shift": shift, "display_order": display_order,
                        "gold_after_rebinding": q["gold"], "mode": mode,
                        "exact_correct": sum(int(x["exact_correct"]) for x in rollouts),
                        "total": ROLLOUTS,
                        "mean_partial_score": sum(x["partial_score"] for x in rollouts) / ROLLOUTS,
                        "k5_stable": all(x["exact_correct"] for x in rollouts),
                        "rollouts": rollouts,
                    })
    finally:
        backend.close()

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "exact_correct": sum(x["exact_correct"] for x in rows),
            "exact_total": sum(x["total"] for x in rows),
            "exact_accuracy": sum(x["exact_correct"] for x in rows) / sum(x["total"] for x in rows),
            "mean_partial_score": sum(x["mean_partial_score"] for x in rows) / len(rows),
            "k5_stable_cells": sum(int(x["k5_stable"]) for x in rows),
            "parse_failure_count": sum(
                sum(int(r["failure_type"] == "parse_failure") for r in x["rollouts"])
                for x in rows
            ),
            "reasoning_error_count": sum(
                sum(int(r["failure_type"] == "reasoning_error") for r in x["rollouts"])
                for x in rows
            ),
        }
        rollouts = [rollout for row in rows for rollout in row["rollouts"]]
        multi = [rollout for rollout in rollouts if "precision" in rollout["metrics"]]
        ordering = [rollout for rollout in rollouts if "pairwise_accuracy" in rollout["metrics"]]
        if multi:
            result.update({
                "mean_precision": sum(x["metrics"]["precision"] for x in multi) / len(multi),
                "mean_recall": sum(x["metrics"]["recall"] for x in multi) / len(multi),
                "mean_jaccard": sum(x["metrics"]["jaccard"] for x in multi) / len(multi),
            })
        if ordering:
            result["mean_pairwise_accuracy"] = sum(
                x["metrics"]["pairwise_accuracy"] for x in ordering
            ) / len(ordering)
        return result

    by_q: dict[str, Any] = {}
    for qa_id in sorted({x["qa_id"] for x in records}):
        by_q[qa_id] = {
            mode: stats([x for x in records if x["qa_id"] == qa_id and x["mode"] == mode])
            for mode in ("no_context", "full_context")
        }
    report = {
        "status": "completed", "question_count": len(questions),
        "solver_calls_requested": required_solver_calls,
        "solver_calls_completed": ledger.summary()["completed_request_count"],
        "structured_solver_temperature": args.temperature,
        "structured_solver_max_tokens": STRUCTURED_SOLVER_MAX_TOKENS,
        "no_context": stats([x for x in records if x["mode"] == "no_context"]),
        "full_context": stats([x for x in records if x["mode"] == "full_context"]),
        "by_question": by_q,
        "records_path": str(args.run_root / "structured_position_results.json"),
    }
    atomic_json(args.run_root / "structured_position_results.json", {"records": records})
    atomic_json(args.run_root / "reports/STRUCTURED_POSITION_REPORT.json", report)
    lines = [
        "# Structured QA Position Probe v1", "",
        f"- Solver calls: `{report['solver_calls_completed']}/{report['solver_calls_requested']}`",
        f"- No-context exact: `{report['no_context']['exact_correct']}/{report['no_context']['exact_total']}` ({report['no_context']['exact_accuracy']:.1%})",
        f"- Full-context exact: `{report['full_context']['exact_correct']}/{report['full_context']['exact_total']}` ({report['full_context']['exact_accuracy']:.1%})",
        "", "| QA | Type | No-context | Full-context |", "|---|---|---:|---:|",
    ]
    for qa_id in sorted(by_q):
        qtype = next(x["question_type"] for x in records if x["qa_id"] == qa_id)
        n, f = by_q[qa_id]["no_context"], by_q[qa_id]["full_context"]
        lines.append(f"| {qa_id} | {qtype} | {n['exact_correct']}/{n['exact_total']} | {f['exact_correct']}/{f['exact_total']} |")
    (args.run_root / "reports/STRUCTURED_POSITION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
