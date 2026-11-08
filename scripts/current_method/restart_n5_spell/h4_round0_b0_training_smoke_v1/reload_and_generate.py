"""Reload a custom adapter and run fixed-format smoke generations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_qwen3_lora_sft as trainer


BOX_RE = re.compile(r"^<box>([^<>]+)</box>$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    return parser.parse_args()


def valid_box(text: str, qtype: str, display_keys: set[str]) -> tuple[bool, str | None]:
    value = text.strip()
    match = BOX_RE.fullmatch(value)
    if not match:
        return False, None
    labels = match.group(1).split(",")
    if qtype == "single_choice":
        ok = len(labels) == 1 and labels[0] in display_keys
    elif qtype == "multi_select":
        ok = bool(labels) and len(labels) == len(set(labels)) and all(label in display_keys for label in labels)
    else:
        ok = len(labels) == len(display_keys) and set(labels) == display_keys and len(labels) == len(set(labels))
    return ok, match.group(1) if ok else None


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = []
    for qtype in ("single_choice", "multi_select", "ordering"):
        row = next((item for item in rows if item["question_type"] == qtype), None)
        if row is None:
            raise RuntimeError(f"reload_smoke_missing_qtype:{qtype}")
        chosen.append(row)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    model.eval()
    trainer.freeze_base_model(model)
    config = json.loads((Path(args.adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    targets = {str(x) for x in config["target_modules"]}
    replaced = trainer.inject_lora(model, targets, int(config["lora_rank"]), int(config["lora_alpha"]), float(config["lora_dropout"]))
    load_args = argparse.Namespace(
        target_modules=",".join(sorted(targets)), lora_rank=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
    )
    trainer.load_initial_adapter(model, args.adapter_dir, load_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    outputs = []
    for row in chosen:
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if prompt_ids.shape[-1] >= args.max_length:
            raise RuntimeError(f"reload_prompt_at_max_length:{row['candidate_id']}:{prompt_ids.shape[-1]}")
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, prompt_ids.shape[-1]:]
        decoded = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        keys = set(row["display_to_canonical"])
        legal, answer = valid_box(decoded, row["question_type"], keys)
        outputs.append({
            "candidate_id": row["candidate_id"], "root_id": row["root_id"],
            "question_type": row["question_type"], "smoke_role": row["smoke_role"],
            "generated_text": decoded, "legal_b0_box": legal, "generated_answer": answer,
            "prompt_token_count": int(prompt_ids.shape[-1]), "truncated": False,
        })
    report = {
        "adapter_reload_test_version": "h4_round0_b0_adapter_reload_test_v1",
        "status": "PASS" if replaced > 0 and all(row["legal_b0_box"] for row in outputs) else "FAIL",
        "model": args.model, "tokenizer": args.tokenizer, "adapter_dir": args.adapter_dir,
        "max_length": args.max_length, "lora_modules_replaced": replaced,
        "adapter_config_sha256": __import__("hashlib").sha256((Path(args.adapter_dir) / "adapter_config.json").read_bytes()).hexdigest(),
        "samples": outputs,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
