from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = self.dropout(x).to(torch.float32)
        update = self.lora_b(self.lora_a(lora_in)) * self.scaling
        return base_out + update.to(base_out.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal LoRA SFT trainer for Qwen-style causal LMs.")
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path. Defaults to --model for backward compatibility.",
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", default=None)
    parser.add_argument("--output-dir", default="outputs/qwen3_4b_lora_sft")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument(
        "--init-adapter-dir",
        default=None,
        help="Optional existing adapter directory to initialize LoRA weights before continuing SFT.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=64)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def render_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
    return prompt if add_generation_prompt else f"{prompt} {assistant}"


def tokenize_record(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("record must contain chat messages")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("record messages must be objects")
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError("record messages must contain a user message")
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("record messages must contain an assistant message")
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids.append(tokenizer.eos_token_id)
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[min(len(prompt_ids), len(full_ids)) :]
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        labels = labels[overflow:]
    if not any(label != -100 for label in labels):
        raise ValueError("all labels are masked after truncation")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def collate(batch: list[dict[str, list[int]]], pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def freeze_base_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def inject_lora(model: nn.Module, target_modules: set[str], rank: int, alpha: int, dropout: float) -> int:
    replaced = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_modules and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


def load_initial_adapter(model: nn.Module, adapter_dir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    state_path = adapter_path / "adapter_model.pt"
    if not config_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Adapter directory must contain adapter_config.json and adapter_model.pt: {adapter_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_targets = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    mismatches: dict[str, Any] = {}
    for key, expected in (("lora_rank", args.lora_rank), ("lora_alpha", args.lora_alpha)):
        if int(config.get(key)) != int(expected):
            mismatches[key] = {"adapter": config.get(key), "current": expected}
    if sorted(config.get("target_modules") or []) != sorted(expected_targets):
        mismatches["target_modules"] = {"adapter": config.get("target_modules"), "current": expected_targets}
    if mismatches:
        raise ValueError(f"Initial adapter config does not match current LoRA config: {mismatches}")

    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_lora = [item for item in missing if ".lora_" in item]
    if missing_lora or unexpected:
        raise RuntimeError(f"Initial adapter load mismatch. missing_lora={missing_lora[:8]} unexpected={unexpected[:8]}")
    return {"init_adapter_dir": str(adapter_path), "init_adapter_config": config}


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def lr_at_step(step: int, total_steps: int, base_lr: float, min_lr_ratio: float, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


@torch.no_grad()
def evaluate(model: nn.Module, features: list[dict[str, list[int]]], pad_token_id: int, device: torch.device, batch_size: int) -> float | None:
    if not features:
        return None
    model.eval()
    losses: list[float] = []
    for start in range(0, len(features), batch_size):
        batch = collate(features[start : start + batch_size], pad_token_id, device)
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
    model.train()
    return round(sum(losses) / len(losses), 4)


def save_adapter(model: nn.Module, output_dir: Path, args: argparse.Namespace, step: int, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ".lora_" in name and param.requires_grad
    }
    torch.save(adapter_state, output_dir / "adapter_model.pt")
    config = {
        "step": step,
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
        "max_length": args.max_length,
        "init_adapter_dir": args.init_adapter_dir,
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_records = load_jsonl(args.train_file)
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    valid_records = load_jsonl(args.valid_file) if args.valid_file else []
    if args.max_valid_samples is not None:
        valid_records = valid_records[: args.max_valid_samples]

    train_features = [tokenize_record(tokenizer, record, args.max_length) for record in train_records]
    valid_features = [tokenize_record(tokenizer, record, args.max_length) for record in valid_records]

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    freeze_base_model(model)
    target_modules = {item.strip() for item in args.target_modules.split(",") if item.strip()}
    replaced = inject_lora(model, target_modules, args.lora_rank, args.lora_alpha, args.lora_dropout)
    if replaced == 0:
        raise RuntimeError(f"No target modules matched: {sorted(target_modules)}")
    init_adapter_info = load_initial_adapter(model, args.init_adapter_dir, args) if args.init_adapter_dir else None
    if not args.no_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)
    model.train()

    trainable, total = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(train_features) / max(1, args.batch_size * args.grad_accum))
    total_update_steps = max(1, int(math.ceil(update_steps_per_epoch * args.epochs)))

    print(json.dumps({
        "device": str(device),
        "dtype": str(dtype),
        "train_records": len(train_features),
        "valid_records": len(valid_features),
        "target_modules_replaced": replaced,
        "init_adapter": init_adapter_info,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": round(trainable / total, 6),
        "total_update_steps": total_update_steps,
    }, ensure_ascii=False, indent=2))

    output_dir = Path(args.output_dir)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    max_micro_steps = int(math.ceil(len(train_features) * args.epochs / max(1, args.batch_size)))

    while micro_step < max_micro_steps:
        indices = list(range(len(train_features)))
        random.shuffle(indices)
        for start in range(0, len(indices), args.batch_size):
            if micro_step >= max_micro_steps:
                break
            batch_indices = indices[start : start + args.batch_size]
            batch = collate([train_features[index] for index in batch_indices], tokenizer.pad_token_id, device)
            output = model(**batch)
            loss = output.loss / args.grad_accum
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum
            micro_step += 1

            if micro_step % args.grad_accum == 0 or micro_step >= max_micro_steps:
                lr = lr_at_step(global_step, total_update_steps, args.lr, args.min_lr_ratio, args.warmup_ratio)
                set_lr(optimizer, lr)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % max(1, args.eval_steps) == 0 or global_step == 1:
                    train_loss = running_loss / max(1, args.eval_steps * args.grad_accum)
                    valid_loss = evaluate(model, valid_features, tokenizer.pad_token_id, device, args.batch_size)
                    print(json.dumps({
                        "step": global_step,
                        "micro_step": micro_step,
                        "lr": lr,
                        "train_loss_recent": round(train_loss, 4),
                        "valid_loss": valid_loss,
                    }, ensure_ascii=False))
                    running_loss = 0.0
                if global_step % max(1, args.save_steps) == 0:
                    save_adapter(model, output_dir / f"checkpoint-{global_step}", args, global_step, {"valid_loss": None})

    final_valid_loss = evaluate(model, valid_features, tokenizer.pad_token_id, device, args.batch_size)
    save_adapter(model, output_dir, args, global_step, {"final_valid_loss": final_valid_loss})
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({"finished": True, "global_step": global_step, "final_valid_loss": final_valid_loss}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
