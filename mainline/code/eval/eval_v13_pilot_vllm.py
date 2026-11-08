#!/usr/bin/env python3
"""vLLM evaluator for r2_auto_full/v13-style graph-expansion SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Blackwell SM 12.x + vLLM 0.21: the default FlashInfer sampler JIT-fails and degrades decoding
# to gibberish. Force it off BEFORE importing vllm so this evaluator is always clean.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_v13_pilot import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    DEFAULT_MODEL,
    discover_inputs,
    evaluate_prediction,
    load_jsonl,
    render_prompt,
    result_from_prediction,
    summarize_results,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v13/r2_auto_full records with vLLM k-sampling.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--input", nargs="*", default=None, help="Optional explicit chat JSONL files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Merged HF model path or base model path.")
    parser.add_argument("--adapter-dir", default=None, help="Recorded for provenance only when using a merged model.")
    parser.add_argument("--max-length", type=int, default=8192, help="Prompt token budget before generation.")
    parser.add_argument("--max-model-len", type=int, default=9216, help="vLLM max_model_len.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sampling-k", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_prompt_ids(tokenizer: Any, record: dict[str, Any], max_length: int) -> list[int]:
    prompt = render_prompt(tokenizer, record["messages"])
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(ids) > max_length:
        ids = ids[-max_length:]
    return ids


def sampling_params(args: argparse.Namespace) -> SamplingParams:
    do_sample = int(args.sampling_k) > 1
    return SamplingParams(
        n=max(1, int(args.sampling_k)),
        temperature=float(args.sample_temperature) if do_sample else 0.0,
        top_p=float(args.top_p) if do_sample else 1.0,
        max_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
    )


def eval_file(eval_set: str, path: Path, llm: LLM, args: argparse.Namespace) -> dict[str, Any]:
    records = load_jsonl(path)
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = llm.get_tokenizer()
    params = sampling_params(args)
    results: list[dict[str, Any]] = []
    truncated_count = 0
    token_lengths: list[int] = []

    for start in range(0, len(records), max(1, int(args.chunk_size))):
        batch = records[start : start + max(1, int(args.chunk_size))]
        prompts: list[TokensPrompt] = []
        for record in batch:
            prompt = render_prompt(tokenizer, record["messages"])
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            token_lengths.append(len(ids))
            if len(ids) > args.max_length:
                truncated_count += 1
                ids = ids[-args.max_length :]
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for batch size {len(batch)}")
        for offset, (record, output) in enumerate(zip(batch, outputs)):
            solver_samples: list[dict[str, Any]] = []
            for sample_index, completion in enumerate(output.outputs):
                prediction = str(completion.text or "").strip()
                judged = evaluate_prediction(record, prediction)
                solver_samples.append(
                    {
                        "sample_index": sample_index,
                        "prediction": prediction,
                        "correct": judged.get("correct") is True,
                        "judged": judged,
                    }
                )
            if not solver_samples:
                judged = evaluate_prediction(record, "")
                solver_samples.append({"sample_index": 0, "prediction": "", "correct": False, "judged": judged})
            first = solver_samples[0]
            results.append(
                result_from_prediction(
                    eval_set=eval_set,
                    index=start + offset,
                    record=record,
                    prediction=first["prediction"],
                    judged=first["judged"],
                    solver_samples=solver_samples,
                    args=args,
                )
            )
        logging.info("%s: evaluated %d/%d", eval_set, len(results), len(records))

    report = {
        "eval_set": eval_set,
        "input": str(path),
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "summary": summarize_results(results),
        "prompt_length_stats": {
            "max_length": args.max_length,
            "truncated_count": truncated_count,
            "min": min(token_lengths) if token_lengths else None,
            "max": max(token_lengths) if token_lengths else None,
            "mean": round(sum(token_lengths) / len(token_lengths), 3) if token_lengths else None,
        },
        "results": results,
    }
    return report


def format_status(summary: dict[str, Any]) -> str:
    lines = [
        "# solver_r1 vLLM Eval Status",
        "",
        f"- n: {summary.get('n')}",
        f"- first-sample correct: {summary.get('correct')}",
        f"- first-sample wrong: {summary.get('wrong')}",
        f"- first-sample accuracy: {summary.get('accuracy')}",
        "- k=5 soft_accuracy is stored per record for reward/frontier selection and is not used as final reporting accuracy.",
        "",
        "## Question Types",
    ]
    for key, item in summary.get("by_question_type", {}).items():
        lines.append(f"- {key}: n={item['n']} first_sample_accuracy={item['accuracy']} wrong={item['wrong']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.input:
        inputs = [(Path(path).stem, Path(path)) for path in args.input]
    else:
        inputs = discover_inputs(Path(args.input_root))
    if not inputs:
        raise RuntimeError(f"No eval inputs found under {args.input_root}")

    logging.info("Loading vLLM model from %s", args.model)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )

    per_set_reports: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for eval_set, path in inputs:
        logging.info("Evaluating %s from %s", eval_set, path)
        report = eval_file(eval_set, path, llm, args)
        write_json(output_dir / eval_set / "eval.json", report)
        per_set_reports.append({key: value for key, value in report.items() if key != "results"})
        all_results.extend(report["results"])

    overall = {
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "input_root": args.input_root,
        "backend": "vllm",
        "max_length": args.max_length,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "sampling_k": args.sampling_k,
        "sample_temperature": args.sample_temperature,
        "summary": summarize_results(all_results),
        "eval_sets": per_set_reports,
        "results": all_results,
    }
    write_json(output_dir / "overall_summary.json", overall)
    write_json(output_dir / "solver_r1_eval.json", overall)
    write_json(output_dir / "trap_type_breakdown.json", overall["summary"]["by_trap_type"])
    write_json(output_dir / "question_type_breakdown.json", overall["summary"]["by_question_type"])
    write_json(output_dir / "state_chain_depth_breakdown.json", overall["summary"]["by_state_chain_depth"])
    write_json(output_dir / "added_turns_breakdown.json", overall["summary"]["by_added_turns"])
    (output_dir / "status.md").write_text(format_status(overall["summary"]), encoding="utf-8")
    logging.info("Wrote vLLM eval reports to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
