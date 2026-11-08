from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_qwen3_lora_sft import freeze_base_model, inject_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base or custom-LoRA Qwen model on chat MCQ JSONL.")
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
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
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    system = prompt_messages[0]["content"] if prompt_messages and prompt_messages[0].get("role") == "system" else ""
    user = next((m["content"] for m in prompt_messages if m.get("role") == "user"), "")
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def gold_answer(record: dict[str, Any]) -> tuple[str | None, str]:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    assistant = next((m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"), "")
    label = parse_label(str(assistant))
    return label, str(assistant).strip()


def parse_label(text: str) -> str | None:
    text = text.strip()
    match = re.match(r"^\s*([A-Z])(?:[\.\):]|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:answer|option|选项|答案)\s*[:：]?\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


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
    train_missing = [item for item in missing if ".lora_" in item]
    if train_missing or unexpected:
        raise RuntimeError(f"Adapter load mismatch. missing_lora={train_missing[:5]} unexpected={unexpected[:5]}")
    return {"adapter_config": config, "target_modules_replaced": replaced}


@torch.no_grad()
def generate_one(model: torch.nn.Module, tokenizer: Any, record: dict[str, Any], device: torch.device, args: argparse.Namespace) -> str:
    prompt = render_prompt(tokenizer, record["messages"])
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output_ids = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def count_map(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def grouped_accuracy(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result.get(key) or "unknown"), []).append(result)
    return {
        group: {
            "count": len(items),
            "correct": sum(1 for item in items if item["correct"]),
            "accuracy": round(sum(1 for item in items if item["correct"]) / len(items), 4) if items else None,
        }
        for group, items in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def prompt_bucket(prompt_tokens: Any) -> str:
    try:
        tokens = int(prompt_tokens)
    except Exception:
        return "unknown"
    if tokens <= 4096:
        return "<=4k"
    if tokens <= 8192:
        return "4k-8k"
    if tokens <= 16384:
        return "8k-16k"
    if tokens <= 24576:
        return "16k-24k"
    if tokens <= 30000:
        return "24k-30k"
    return ">30k"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for item in results if item["correct"])
    return {
        "count": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results), 4) if results else None,
        "missing_prediction_label_count": sum(1 for item in results if not item.get("pred_label")),
        "gold_label_counts": count_map([item.get("gold_label") for item in results]),
        "pred_label_counts": count_map([item.get("pred_label") for item in results]),
        "by_training_use": grouped_accuracy(results, "training_use"),
        "by_difficulty": grouped_accuracy(results, "difficulty_for_solver"),
        "by_evolution_type": grouped_accuracy(results, "evolution_type"),
        "by_candidate_type": grouped_accuracy(results, "candidate_type"),
        "by_dataset": grouped_accuracy(results, "dataset"),
        "by_task_type": grouped_accuracy(results, "task_type"),
        "by_prompt_bucket": grouped_accuracy(results, "prompt_bucket"),
    }


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    adapter_info: dict[str, Any] | None = None
    if args.adapter_dir:
        adapter_info = add_lora_adapter(model, args.adapter_dir)
    model.to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        generated = generate_one(model, tokenizer, record, device, args)
        pred_label = parse_label(generated)
        gold_label, gold_text = gold_answer(record)
        meta = metadata(record)
        record_prompt_tokens = meta.get("prompt_tokens")
        result = {
            "index": index,
            "record_id": record.get("record_id") or meta.get("record_id"),
            "source_record_id": meta.get("source_record_id") or meta.get("record_id") or record.get("record_id"),
            "sample_name": record.get("sample_name") or meta.get("sample_name"),
            "option_eval_view": meta.get("option_eval_view"),
            "option_permutation_old_indices": meta.get("option_permutation_old_indices"),
            "gold_label": gold_label,
            "gold_answer": gold_text,
            "pred_label": pred_label,
            "prediction": generated,
            "correct": pred_label == gold_label and gold_label is not None,
            "training_use": meta.get("training_use"),
            "difficulty_for_solver": meta.get("difficulty_for_solver"),
            "quality_tier": meta.get("quality_tier"),
            "evolution_type": meta.get("evolution_type"),
            "candidate_type": meta.get("candidate_type"),
            "dataset": meta.get("dataset"),
            "task_type": meta.get("task_type"),
            "prompt_tokens": record_prompt_tokens,
            "prompt_bucket": meta.get("prompt_bucket") or prompt_bucket(record_prompt_tokens),
        }
        results.append(result)
        if not args.quiet:
            print(json.dumps({
                "index": index,
                "record_id": result["record_id"],
                "gold": gold_label,
                "pred": pred_label,
                "correct": result["correct"],
                "prediction": generated[:120],
            }, ensure_ascii=False), flush=True)

    summary = summarize(results)
    payload = {
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "input": args.input,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "adapter_info": adapter_info,
        "summary": summary,
        "results": results,
    }
    write_json(args.output, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved eval report to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
