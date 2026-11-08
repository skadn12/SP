#!/usr/bin/env python3
"""Read-only final-training tokenization gate.

This wrapper calls the frozen trainer's own ``tokenize_record`` implementation;
it does not modify the trainer or input data.  A non-zero exit status means the
merged export is not eligible for launch.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

from transformers import AutoTokenizer

TRAINER_ROOT = pathlib.Path("/root/autodl-tmp/gxy-selfplay/current_method_github")
sys.path.insert(0, str(TRAINER_ROOT))
import train_qwen3_lora_sft as trainer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="/root/autodl-tmp/Qwen3-4B")
    p.add_argument("--max-length", type=int, default=12288)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data_path = pathlib.Path(args.data)
    out_path = pathlib.Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    records = trainer.load_jsonl(data_path)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    type_counts: collections.Counter[str] = collections.Counter()
    for index, record in enumerate(records):
        messages = record.get("messages")
        row: dict[str, object] = {
            "index": index,
            "candidate_id": record.get("candidate_id"),
            "root_id": record.get("root_id"),
            "question_type": record.get("question_type"),
        }
        type_counts[str(record.get("question_type"))] += 1
        try:
            if not isinstance(messages, list):
                raise ValueError("messages is not a list")
            role_counts = collections.Counter(str(m.get("role")) for m in messages if isinstance(m, dict))
            row["role_counts"] = dict(role_counts)
            # The frozen trainer is designed for one assistant Gold suffix.  A
            # multi-assistant record would make the masking boundary ambiguous.
            if role_counts.get("assistant", 0) != 1:
                raise ValueError(f"expected exactly one assistant message, got {role_counts.get('assistant', 0)}")
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError("assistant message must be final")
            # Reproduce the frozen trainer's pre-truncation calculation so a
            # record cannot silently pass after left truncation.
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            full_text = trainer.render_chat(tok, messages, add_generation_prompt=False)
            prompt_text = trainer.render_chat(tok, prompt_messages, add_generation_prompt=True)
            raw_full_ids = tok(full_text, add_special_tokens=False).input_ids
            prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
            if tok.eos_token_id is not None and (not raw_full_ids or raw_full_ids[-1] != tok.eos_token_id):
                raw_full_ids.append(tok.eos_token_id)
            overlap = min(len(raw_full_ids), len(prompt_ids))
            prefix_match = raw_full_ids[:overlap] == prompt_ids[:overlap]
            overflow = max(0, len(raw_full_ids) - args.max_length)
            feature = trainer.tokenize_record(tok, record, args.max_length)
            full_ids = feature["input_ids"]
            labels = feature["labels"]
            gold = sum(x != -100 for x in labels)
            row.update({
                "raw_full_tokens": len(raw_full_ids),
                "prompt_tokens": len(prompt_ids),
                "full_tokens": len(full_ids),
                "gold_tokens": gold,
                "nonpad_labels": gold,
                "overflow": overflow,
                "prefix_match": prefix_match,
            })
            if overflow:
                raise ValueError(f"full sequence exceeds max_length by {overflow}")
            if not prefix_match:
                raise ValueError("prompt/full token prefix mismatch")
            if gold == 0:
                raise ValueError("all labels masked")
        except Exception as exc:  # preserve row context in the report
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(row)
        rows.append(row)
    summary = {
        "schema": "final_training_tokenization_preflight_v1",
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "tokenizer_path": args.model,
        "max_length": args.max_length,
        "record_count": len(records),
        "question_type_counts": dict(type_counts),
        "failure_count": len(failures),
        "failures": failures,
        "max_full_tokens": max((int(r["full_tokens"]) for r in rows if "full_tokens" in r), default=None),
        "min_gold_tokens": min((int(r["gold_tokens"]) for r in rows if "gold_tokens" in r), default=None),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("record_count", "data_sha256", "question_type_counts", "failure_count", "max_full_tokens", "min_gold_tokens")}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
