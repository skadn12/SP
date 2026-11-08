#!/usr/bin/env python3
"""Merge this repo's custom LoRA adapter into a Hugging Face causal LM.

The solver_r1 adapter in this workspace is saved by train_qwen3_lora_sft.py as
plain torch tensors named ``*.lora_a.weight`` and ``*.lora_b.weight``. vLLM does
not load that custom wrapper directly, so this utility applies the LoRA deltas
to the base model weights and writes a normal HF model directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge custom LoRA adapter weights into a base CausalLM.")
    parser.add_argument("--model", required=True, help="Base HF model path.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path to save with the merged model. Defaults to --model.",
    )
    parser.add_argument("--adapter-dir", required=True, help="Directory containing adapter_config.json and adapter_model.pt.")
    parser.add_argument("--output-dir", required=True, help="Destination merged HF model directory.")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def get_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    module: torch.nn.Module = root
    for part in dotted.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def load_adapter(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    config_path = adapter_dir / "adapter_config.json"
    state_path = adapter_dir / "adapter_model.pt"
    if not config_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Adapter directory must contain adapter_config.json and adapter_model.pt: {adapter_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected adapter state dict in {state_path}")
    return config, state


@torch.no_grad()
def merge_lora(model: torch.nn.Module, state: dict[str, torch.Tensor], rank: int, alpha: int) -> dict[str, Any]:
    scaling = float(alpha) / float(rank)
    merged = 0
    skipped: list[str] = []
    for key in sorted(state):
        if not key.endswith(".lora_a.weight"):
            continue
        module_name = key[: -len(".lora_a.weight")]
        a_key = key
        b_key = f"{module_name}.lora_b.weight"
        if b_key not in state:
            skipped.append(module_name)
            continue
        module = get_module(model, module_name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"LoRA target is not nn.Linear: {module_name} ({type(module)!r})")
        a = state[a_key].to(dtype=torch.float32)
        b = state[b_key].to(dtype=torch.float32)
        delta = torch.matmul(b, a) * scaling
        if tuple(delta.shape) != tuple(module.weight.shape):
            raise ValueError(f"Delta shape mismatch for {module_name}: {tuple(delta.shape)} vs {tuple(module.weight.shape)}")
        module.weight.data.add_(delta.to(dtype=module.weight.dtype))
        merged += 1
    return {"merged_linear_modules": merged, "skipped_missing_pairs": skipped}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists; pass --overwrite to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = Path(args.adapter_dir)
    config, state = load_adapter(adapter_dir)
    rank = int(config.get("lora_rank"))
    alpha = int(config.get("lora_alpha"))

    print(json.dumps({"event": "loading_base_model", "model": args.model, "dtype": args.dtype}, ensure_ascii=False), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype(args.dtype), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    merge_info = merge_lora(model, state, rank=rank, alpha=alpha)

    print(json.dumps({"event": "saving_merged_model", "output_dir": str(output_dir), **merge_info}, ensure_ascii=False), flush=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "base_model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "adapter_dir": str(adapter_dir),
        "adapter_config": config,
        "merge_info": merge_info,
        "note": "Merged from the custom LoRALinear format used by train_qwen3_lora_sft.py.",
    }
    (output_dir / "merged_adapter_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "output_dir": str(output_dir), **merge_info}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
