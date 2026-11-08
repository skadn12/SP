from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_qwen3_lora_sft import freeze_base_model, inject_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structural-variation chat JSONL with relaxed final-answer scoring.")
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--sampling-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_jsonl(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        eval_set = Path(path).stem
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_eval_set"] = eval_set
                rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"最终结果\s*[:：]", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s/\\-_.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def final_answer(text: str) -> str:
    matches = list(re.finditer(r"最终结果\s*[:：]", text))
    if matches:
        return text[matches[-1].end() :].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def gold_answer(row: dict[str, Any]) -> str:
    meta = metadata(row)
    if meta.get("gold_answer"):
        return str(meta["gold_answer"]).strip()
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return final_answer(str(message.get("content") or ""))
    return ""


def correct_prediction(row: dict[str, Any], prediction: str) -> bool:
    meta = metadata(row)
    answer_type = str(meta.get("answer_format") or "")
    gold = gold_answer(row)
    pred = final_answer(prediction)
    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(pred)
    if not gold_norm or not pred_norm:
        return False
    if answer_type in {"single_choice", "multi_select", "ordering"}:
        return pred_norm == gold_norm
    if len(gold_norm) < 8 or len(pred_norm) < 8:
        return pred_norm == gold_norm
    return gold_norm in pred_norm or pred_norm in gold_norm


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [m for m in row.get("messages") or [] if isinstance(m, dict) and m.get("role") != "assistant"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def add_lora_adapter(model: torch.nn.Module, adapter_dir: str | Path) -> dict[str, Any]:
    adapter_path = Path(adapter_dir)
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    freeze_base_model(model)
    replaced = inject_lora(
        model,
        set(config["target_modules"]),
        rank=int(config["lora_rank"]),
        alpha=int(config["lora_alpha"]),
        dropout=0.0,
    )
    state = torch.load(adapter_path / "adapter_model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_lora = [item for item in missing if ".lora_" in item]
    if missing_lora or unexpected:
        raise RuntimeError(f"Adapter load mismatch. missing_lora={missing_lora[:5]} unexpected={unexpected[:5]}")
    return {"adapter_config": config, "target_modules_replaced": replaced}


@torch.no_grad()
def generate_samples(model: torch.nn.Module, tokenizer: Any, row: dict[str, Any], device: torch.device, args: argparse.Namespace) -> tuple[list[str], int]:
    prompt = render_prompt(tokenizer, row)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=False)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs: list[str] = []
    for sample_idx in range(args.sampling_k):
        do_sample = args.sampling_k > 1
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs, prompt_tokens


def bucket_for_soft_accuracy(value: float) -> str:
    if value == 0:
        return "possibly_unsolvable_or_too_hard"
    if value <= 0.2:
        return "weak"
    if value <= 0.6:
        return "frontier"
    return "easy"


def grouped(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(row.get(key) or "unknown")].append(row)
    output: dict[str, dict[str, Any]] = {}
    for group, items in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        n = len(items)
        correct = sum(1 for item in items if item.get("first_correct"))
        output[group] = {
            "n": n,
            "correct": correct,
            "wrong": n - correct,
            "accuracy": round(correct / n, 4) if n else None,
            "mean_soft_accuracy": round(sum(float(item.get("soft_accuracy") or 0) for item in items) / n, 4) if n else None,
        }
    return output


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    correct = sum(1 for row in results if row.get("first_correct"))
    return {
        "n": n,
        "correct": correct,
        "wrong": n - correct,
        "accuracy": round(correct / n, 4) if n else None,
        "solver_accuracy_buckets": dict(Counter(row.get("bucket") for row in results)),
        "frontier_count": sum(1 for row in results if row.get("bucket") in {"frontier", "weak"}),
        "wrong_frontier_count": sum(1 for row in results if row.get("bucket") in {"frontier", "weak"} and not row.get("first_correct")),
        "by_eval_set": grouped(results, "eval_set"),
        "by_structure_mode": grouped(results, "structure_mode"),
        "by_prompt_bucket": grouped(results, "prompt_bucket"),
        "by_actual_prompt_bucket": grouped(results, "actual_prompt_bucket"),
    }


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    adapter_info = add_lora_adapter(model, args.adapter_dir) if args.adapter_dir else None
    model.to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    wrong_frontier_rows: list[dict[str, Any]] = []
    hard_correct_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        outputs, prompt_tokens = generate_samples(model, tokenizer, row, device, args)
        checks = [correct_prediction(row, output) for output in outputs]
        correct_count = sum(1 for item in checks if item)
        soft_accuracy = correct_count / len(checks) if checks else 0.0
        bucket = bucket_for_soft_accuracy(soft_accuracy)
        meta = metadata(row)
        result = {
            "index": idx,
            "record_id": row.get("record_id") or meta.get("record_id"),
            "sample_name": row.get("sample_name") or meta.get("sample_name"),
            "eval_set": row.get("_eval_set"),
            "structure_mode": meta.get("structure_mode"),
            "question_type": meta.get("question_type"),
            "answer_format": meta.get("answer_format"),
            "prompt_bucket": meta.get("prompt_bucket"),
            "actual_prompt_bucket": meta.get("actual_prompt_bucket"),
            "prompt_tokens": prompt_tokens,
            "gold_answer": gold_answer(row),
            "prediction": outputs[0] if outputs else "",
            "pred_final_answer": final_answer(outputs[0]) if outputs else "",
            "first_correct": checks[0] if checks else False,
            "correct_count": correct_count,
            "successful_count": len(checks),
            "soft_accuracy": round(soft_accuracy, 4),
            "bucket": bucket,
            "samples": [
                {"sample_index": sample_idx, "prediction": output, "final_answer": final_answer(output), "correct": checks[sample_idx]}
                for sample_idx, output in enumerate(outputs)
            ],
            "metadata": meta,
        }
        results.append(result)
        if bucket in {"frontier", "weak"}:
            payload = {"record_id": result["record_id"], "selection_reason": f"train_hard_gate_valid_{bucket}", "reward": result, "sample": row}
            frontier_rows.append(payload)
            if result["first_correct"]:
                hard_correct_rows.append(payload)
            else:
                wrong_frontier_rows.append(payload)
        if not args.quiet and (idx + 1) % 10 == 0:
            print(f"evaluated {idx + 1}/{len(rows)}")

    output_dir = Path(args.output_dir)
    summary = summarize(results)
    report = {
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "input": args.input,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "sampling_k": args.sampling_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "adapter_info": adapter_info,
        "summary": summary,
        "results": results,
    }
    write_json(output_dir / "overall_summary.json", report)
    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "solver_r4_frontier_candidates.jsonl", frontier_rows)
    write_jsonl(output_dir / "solver_r4_wrong_frontier.jsonl", wrong_frontier_rows)
    write_jsonl(output_dir / "solver_r4_hard_correct.jsonl", hard_correct_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved eval report to: {output_dir / 'overall_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
