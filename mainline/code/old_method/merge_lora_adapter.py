from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_qwen3_lora_sft import LoRALinear, freeze_base_model, inject_lora, load_initial_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge this repo's minimal LoRA adapter into a base model directory.")
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def merge_lora_modules(module: nn.Module) -> int:
    merged = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            base = child.base
            update = torch.matmul(child.lora_b.weight.detach().float(), child.lora_a.weight.detach().float()) * child.scaling
            base.weight.data = (base.weight.detach().float() + update).to(base.weight.dtype)
            setattr(module, child_name, base)
            merged += 1
        else:
            merged += merge_lora_modules(child)
    return merged


def main() -> int:
    args = parse_args()
    adapter_config = json.loads((Path(args.adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    freeze_base_model(model)
    targets = set(adapter_config["target_modules"])
    replaced = inject_lora(
        model,
        targets,
        rank=int(adapter_config["lora_rank"]),
        alpha=int(adapter_config["lora_alpha"]),
        dropout=0.0,
    )

    # Reuse the existing loader by passing a small args shim with matching config.
    class LoaderArgs:
        target_modules = ",".join(adapter_config["target_modules"])
        lora_rank = int(adapter_config["lora_rank"])
        lora_alpha = int(adapter_config["lora_alpha"])

    load_info = load_initial_adapter(model, args.adapter_dir, LoaderArgs())
    merged = merge_lora_modules(model)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(output)
    info: dict[str, Any] = {
        "base_model": args.model,
        "adapter_dir": args.adapter_dir,
        "output_dir": str(output),
        "target_modules_replaced": replaced,
        "lora_modules_merged": merged,
        "adapter_load_info": load_info,
    }
    (output / "merged_adapter_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
