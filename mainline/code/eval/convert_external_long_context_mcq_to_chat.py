#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a careful multiple-choice assistant. Read the context and question, "
    "compare the options, and answer with only the option letter."
)
LETTERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert unified external long-context MCQ JSONL to chat MCQ eval JSONL.")
    parser.add_argument("--input", default="eval_sets/external_long_context_mcq/all_mcq_30k_eligible.jsonl")
    parser.add_argument("--output", default="eval_sets/external_long_context_mcq/all_mcq_30k_eligible_chat.jsonl")
    parser.add_argument("--summary", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def render_user(row: dict[str, Any]) -> str:
    options = row.get("options") if isinstance(row.get("options"), dict) else {}
    rendered_options = "\n".join(f"{label}. {str(options.get(label) or '').strip()}" for label in LETTERS)
    return (
        f"Context:\n{str(row.get('context') or '').strip()}\n\n"
        f"Question:\n{str(row.get('question') or '').strip()}\n\n"
        f"Options:\n{rendered_options}\n\n"
        "Answer with only the option letter."
    )


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or "").strip().upper()
    options = row.get("options") if isinstance(row.get("options"), dict) else {}
    option_text = str(options.get(answer) or "").strip()
    assistant = answer if not option_text else f"{answer}. {option_text}"
    rid = str(row.get("id") or "")
    return {
        "record_id": rid,
        "sample_name": row.get("dataset"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user(row)},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "record_id": rid,
            "source_record_id": rid,
            "sample_name": row.get("dataset"),
            "dataset": row.get("dataset"),
            "task_type": row.get("task_type"),
            "prompt_tokens": row.get("prompt_tokens"),
            "prompt_bucket": prompt_bucket(row.get("prompt_tokens")),
            "gold_label": answer,
            "gold_answer_text": option_text,
            "source_format": "external_long_context_mcq",
        },
    }


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input)
    converted: list[dict[str, Any]] = []
    rejected = Counter()
    for row in rows:
        answer = str(row.get("answer") or "").strip().upper()
        options = row.get("options") if isinstance(row.get("options"), dict) else {}
        if answer not in LETTERS:
            rejected["missing_or_invalid_answer"] += 1
            continue
        if any(label not in options or not str(options.get(label) or "").strip() for label in LETTERS):
            rejected["missing_options"] += 1
            continue
        converted.append(convert_row(row))

    write_jsonl(args.output, converted)
    summary = {
        "input": args.input,
        "output": args.output,
        "input_count": len(rows),
        "converted_count": len(converted),
        "rejected": dict(rejected),
        "by_dataset": dict(Counter(str(row.get("dataset") or "unknown") for row in rows)),
        "by_prompt_bucket": dict(Counter(prompt_bucket(row.get("prompt_tokens")) for row in rows)),
    }
    summary_path = args.summary or str(Path(args.output).with_suffix(".summary.json"))
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
