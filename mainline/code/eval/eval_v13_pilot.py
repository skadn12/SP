#!/usr/bin/env python3
"""Evaluate solver on graph-expansion multi-format pilot candidates.

This evaluator supports the question types emitted by generate_v13_pilot.py and
generate_r2_llm_graph_expansion_challenger.py:
single_choice, multi_select, ordering, and state_reconstruction. It writes the
standard aggregate and breakdown reports used by graph_expansion_challenger_v1.

Usage:
  python3 scripts/eval_v13_pilot.py
  python3 scripts/eval_v13_pilot.py \
    --adapter-dir checkpoints/solver_r2_mix_lora \
    --input-root eval_sets/graph_expansion_challenger_v1 \
    --output-dir eval_reports/graph_expansion_challenger_v1/v13_pilot_eval

Outputs:
  - overall_summary.json
  - trap_type_breakdown.json
  - question_type_breakdown.json
  - state_chain_depth_breakdown.json
  - added_turns_breakdown.json
  - status.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT_ROOT = ROOT / "eval_sets/graph_expansion_challenger_v1"
DEFAULT_OUTPUT_DIR = ROOT / "eval_reports/graph_expansion_challenger_v1/v13_pilot_eval"
DEFAULT_MODEL = "/root/autodl-tmp/Qwen3-4B"
DEFAULT_ADAPTER = ROOT / "checkpoints/solver_r2_mix_lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate solver_r2 on v1.3 pilot candidates.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--input", nargs="*", default=None, help="Optional explicit chat JSONL files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path; useful when evaluating a merged adapter model.")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sampling-k", type=int, default=1, help="Number of solver samples per record for soft accuracy.")
    parser.add_argument("--sample-temperature", type=float, default=0.7, help="Temperature used when sampling-k > 1.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def discover_inputs(input_root: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    auto_full = input_root / "r2_auto_full"
    if (auto_full / "sft_train.jsonl").exists():
        paths.append(("sft_train", auto_full / "sft_train.jsonl"))
    if (auto_full / "sft_eval_dev.jsonl").exists():
        paths.append(("sft_eval_dev", auto_full / "sft_eval_dev.jsonl"))
    if (auto_full / "sft_eval_heldout.jsonl").exists():
        paths.append(("sft_eval_heldout", auto_full / "sft_eval_heldout.jsonl"))
    if paths:
        return paths
    for name in ("train_v13_pilot", "train_r2_llm_clean_v1_parent_pilot", "train_r2_llm_graph_expansion_pilot"):
        train = input_root / name / "sft_train.jsonl"
        if train.exists():
            paths.append((name, train))
    for name in ("heldout_v13_pilot", "heldout_r2_llm_clean_v1_parent_pilot", "heldout_r2_llm_graph_expansion_pilot"):
        heldout_root = input_root / name
        split_paths = sorted(path for path in heldout_root.glob("sample*/sft_eval.jsonl") if path.exists())
        if split_paths:
            paths.extend((f"{path.parent.name}_{name}", path) for path in split_paths)
        else:
            aggregate = heldout_root / "sft_eval.jsonl"
            if aggregate.exists():
                paths.append((name, aggregate))
    return paths


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    prompt_messages = [message for message in messages if message.get("role") != "assistant"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    system = prompt_messages[0]["content"] if prompt_messages and prompt_messages[0].get("role") == "system" else ""
    user = next((message["content"] for message in prompt_messages if message.get("role") == "user"), "")
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def parse_labels(text: str) -> list[str]:
    labels = re.findall(r"\b([A-F])\b", text.upper())
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen


def parse_single_label(text: str) -> str | None:
    labels = parse_labels(text)
    return labels[0] if labels else None


def gold_from_record(record: dict[str, Any]) -> str:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    assistant = next((item.get("content", "") for item in messages if isinstance(item, dict) and item.get("role") == "assistant"), "")
    return str(assistant).strip()


def evaluate_prediction(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    question_type = str(meta.get("question_type") or "single_choice")
    gold_text = gold_from_record(record)
    if question_type == "single_choice":
        pred_label = parse_single_label(prediction)
        gold_label = parse_single_label(gold_text) or meta.get("gold_label")
        correct = pred_label == gold_label and gold_label is not None
        return {"pred_label": pred_label, "gold_label": gold_label, "correct": correct, "gold_answer": gold_text}
    if question_type == "multi_select":
        pred_labels = sorted(parse_labels(prediction))
        gold_labels = sorted(str(item) for item in (meta.get("gold_options") or parse_labels(gold_text)))
        return {
            "pred_labels": pred_labels,
            "gold_labels": gold_labels,
            "correct": pred_labels == gold_labels,
            "gold_answer": ",".join(gold_labels),
        }
    if question_type == "ordering":
        pred_order = parse_labels(prediction)
        gold_order = [str(item) for item in (meta.get("gold_order") or parse_labels(gold_text))]
        return {
            "pred_order": pred_order,
            "gold_order": gold_order,
            "correct": pred_order == gold_order,
            "gold_answer": " > ".join(gold_order),
        }
    expected = normalize_text(meta.get("gold_answer") or gold_text)
    observed = normalize_text(prediction)
    return {
        "pred_text": prediction.strip(),
        "gold_text": meta.get("gold_answer") or gold_text,
        "correct": bool(expected) and (observed == expected or expected in observed),
        "gold_answer": meta.get("gold_answer") or gold_text,
    }


def added_turn_bucket(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    if number < 180:
        return "<180"
    if number < 200:
        return "180-199"
    if number < 220:
        return "200-219"
    if number < 240:
        return "220-239"
    if number <= 260:
        return "240-260"
    return "260+"


def grouped_accuracy(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[str(item.get(key) or "unknown")].append(item)
    output: dict[str, dict[str, Any]] = {}
    for group, items in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        correct = sum(1 for item in items if item.get("correct"))
        total = len(items)
        output[group] = {
            "n": total,
            "correct": correct,
            "wrong": total - correct,
            "accuracy": round(correct / total, 4) if total else None,
            "error_rate": round((total - correct) / total, 4) if total else None,
        }
    return output


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for item in results if item.get("correct"))
    total = len(results)
    return {
        "n": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": round(correct / total, 4) if total else None,
        "missing_prediction_label_count": sum(1 for item in results if item.get("question_type") == "single_choice" and not item.get("pred_label")),
        "by_eval_set": grouped_accuracy(results, "eval_set"),
        "by_trap_type": grouped_accuracy(results, "trap_type"),
        "by_question_type": grouped_accuracy(results, "question_type"),
        "by_state_chain_depth": grouped_accuracy(results, "state_chain_depth"),
        "by_added_turns": grouped_accuracy(results, "added_turn_bucket"),
    }


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    adapter_info = None
    if args.adapter_dir:
        import evaluate_lora_mcq

        adapter_info = evaluate_lora_mcq.add_lora_adapter(model, args.adapter_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device, adapter_info


def generate_one(model: Any, tokenizer: Any, record: dict[str, Any], device: Any, args: argparse.Namespace, *, do_sample: bool = False) -> str:
    import torch

    prompt = render_prompt(tokenizer, record["messages"])
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.sample_temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_record_k(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    device: Any,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    k = max(1, int(args.sampling_k))
    attempts: list[dict[str, Any]] = []
    first_prediction = ""
    first_judged: dict[str, Any] | None = None
    for sample_index in range(k):
        prediction = generate_one(model, tokenizer, record, device, args, do_sample=k > 1)
        judged = evaluate_prediction(record, prediction)
        if sample_index == 0:
            first_prediction = prediction
            first_judged = judged
        attempts.append(
            {
                "sample_index": sample_index,
                "prediction": prediction,
                "correct": judged.get("correct") is True,
                "judged": judged,
            }
        )
    return first_prediction, (first_judged or {}), attempts


def generate_batch(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: Any,
    args: argparse.Namespace,
    *,
    do_sample: bool,
) -> list[str]:
    import torch

    prompts = [render_prompt(tokenizer, record["messages"]) for record in records]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=False,
        padding=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.sample_temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    predictions: list[str] = []
    for row_idx in range(len(records)):
        new_tokens = output_ids[row_idx, prompt_width:]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return predictions


def result_from_prediction(
    *,
    eval_set: str,
    index: int,
    record: dict[str, Any],
    prediction: str,
    judged: dict[str, Any],
    solver_samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    correct_count = sum(1 for item in solver_samples if item.get("correct") is True)
    successful_count = len(solver_samples)
    result = {
        "index": index,
        "eval_set": eval_set,
        "record_id": record.get("record_id") or meta.get("record_id"),
        "sample_name": record.get("sample_name") or meta.get("sample_name"),
        "prediction": prediction,
        "question_type": meta.get("question_type") or "single_choice",
        "answer_format": meta.get("answer_format"),
        "trap_type": meta.get("trap_type"),
        "attack_type": meta.get("attack_type"),
        "gold_source": meta.get("gold_source"),
        "state_chain_depth": meta.get("state_chain_depth"),
        "added_turn_count": meta.get("added_turn_count"),
        "added_turn_bucket": added_turn_bucket(meta.get("added_turn_count")),
        "training_use": meta.get("training_use"),
        "difficulty_for_solver": meta.get("difficulty_for_solver"),
        "solver_sampling_k": int(args.sampling_k),
        "correct_count": correct_count,
        "successful_count": successful_count,
        "soft_accuracy": round(correct_count / successful_count, 6) if successful_count else None,
        "solver_samples": solver_samples,
    }
    result.update(judged)
    return result


def eval_file(eval_set: str, path: Path, model: Any, tokenizer: Any, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    records = load_jsonl(path)
    if args.limit is not None:
        records = records[: args.limit]
    results: list[dict[str, Any]] = []
    if args.batch_size > 1:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            solver_samples_by_record: list[list[dict[str, Any]]] = [[] for _ in batch]
            for sample_index in range(max(1, int(args.sampling_k))):
                predictions = generate_batch(model, tokenizer, batch, device, args, do_sample=args.sampling_k > 1)
                for offset, (record, prediction) in enumerate(zip(batch, predictions)):
                    judged = evaluate_prediction(record, prediction)
                    solver_samples_by_record[offset].append(
                        {
                            "sample_index": sample_index,
                            "prediction": prediction,
                            "correct": judged.get("correct") is True,
                            "judged": judged,
                        }
                    )
            for offset, record in enumerate(batch):
                first = solver_samples_by_record[offset][0]
                result = result_from_prediction(
                    eval_set=eval_set,
                    index=start + offset,
                    record=record,
                    prediction=first["prediction"],
                    judged=first["judged"],
                    solver_samples=solver_samples_by_record[offset],
                    args=args,
                )
                results.append(result)
            if len(results) % 50 == 0 or len(results) == len(records):
                logging.info("%s: evaluated %d/%d", eval_set, len(results), len(records))
        return {
            "eval_set": eval_set,
            "input": str(path),
            "model": args.model,
            "adapter_dir": args.adapter_dir,
            "summary": summarize_results(results),
            "results": results,
        }
    for index, record in enumerate(records):
        meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        prediction, judged, solver_samples = evaluate_record_k(model, tokenizer, record, device, args)
        correct_count = sum(1 for item in solver_samples if item.get("correct") is True)
        successful_count = len(solver_samples)
        result = {
            "index": index,
            "eval_set": eval_set,
            "record_id": record.get("record_id") or meta.get("record_id"),
            "sample_name": record.get("sample_name") or meta.get("sample_name"),
            "prediction": prediction,
            "question_type": meta.get("question_type") or "single_choice",
            "answer_format": meta.get("answer_format"),
            "trap_type": meta.get("trap_type"),
            "attack_type": meta.get("attack_type"),
            "gold_source": meta.get("gold_source"),
            "state_chain_depth": meta.get("state_chain_depth"),
            "added_turn_count": meta.get("added_turn_count"),
            "added_turn_bucket": added_turn_bucket(meta.get("added_turn_count")),
            "training_use": meta.get("training_use"),
            "difficulty_for_solver": meta.get("difficulty_for_solver"),
            "solver_sampling_k": int(args.sampling_k),
            "correct_count": correct_count,
            "successful_count": successful_count,
            "soft_accuracy": round(correct_count / successful_count, 6) if successful_count else None,
            "solver_samples": solver_samples,
        }
        result.update(judged)
        results.append(result)
        if (index + 1) % 50 == 0:
            logging.info("%s: evaluated %d/%d", eval_set, index + 1, len(records))
    return {
        "eval_set": eval_set,
        "input": str(path),
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "summary": summarize_results(results),
        "results": results,
    }


def format_status(summary: dict[str, Any]) -> str:
    lines = [
        "# v1.3 Pilot Eval Status",
        "",
        f"- n: {summary.get('n')}",
        f"- correct: {summary.get('correct')}",
        f"- wrong: {summary.get('wrong')}",
        f"- accuracy: {summary.get('accuracy')}",
        "",
        "## Question Types",
    ]
    for key, item in summary.get("by_question_type", {}).items():
        lines.append(f"- {key}: n={item['n']} accuracy={item['accuracy']} wrong={item['wrong']}")
    lines.extend(["", "## Trap Types"])
    for key, item in summary.get("by_trap_type", {}).items():
        lines.append(f"- {key}: n={item['n']} accuracy={item['accuracy']} wrong={item['wrong']}")
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
        raise RuntimeError(f"No v1.3 pilot eval inputs found under {args.input_root}")
    logging.info("Evaluating %d input files", len(inputs))
    model, tokenizer, device, adapter_info = load_model(args)
    per_set_reports: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for eval_set, path in inputs:
        logging.info("Evaluating %s from %s", eval_set, path)
        report = eval_file(eval_set, path, model, tokenizer, device, args)
        report["adapter_info"] = adapter_info
        write_json(output_dir / eval_set / "eval.json", report)
        per_set_reports.append({key: value for key, value in report.items() if key != "results"})
        all_results.extend(report["results"])
    overall = {
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "input_root": args.input_root,
        "summary": summarize_results(all_results),
        "eval_sets": per_set_reports,
        "results": all_results,
    }
    write_json(output_dir / "overall_summary.json", overall)
    write_json(output_dir / "trap_type_breakdown.json", overall["summary"]["by_trap_type"])
    write_json(output_dir / "question_type_breakdown.json", overall["summary"]["by_question_type"])
    write_json(output_dir / "state_chain_depth_breakdown.json", overall["summary"]["by_state_chain_depth"])
    write_json(output_dir / "added_turns_breakdown.json", overall["summary"]["by_added_turns"])
    (output_dir / "status.md").write_text(format_status(overall["summary"]), encoding="utf-8")
    logging.info("Wrote eval reports to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_record_id(sample: dict[str, Any]) -> str:
    return str(sample.get("expanded_graph_id") or sample.get("record_id") or sample.get("sample_id") or "")


