from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one raw top-level sample from a LoCoMo JSON list.")
    parser.add_argument("--input", default="locomo10.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--as-list", action="store_true", help="Write [sample] instead of sample.")
    parser.add_argument("--drop-empty-answer", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = load_json(args.input)
    if not isinstance(payload, list):
        raise ValueError("Input must be a JSON list.")
    sample = payload[args.sample_index]
    dropped_count = 0
    if args.drop_empty_answer and isinstance(sample, dict) and isinstance(sample.get("qa"), list):
        original_qa = sample["qa"]
        filtered_qa = [
            item for item in original_qa
            if isinstance(item, dict)
            and str(item.get("answer", item.get("correct_answer_text", ""))).strip()
        ]
        dropped_count = len(original_qa) - len(filtered_qa)
        sample = dict(sample)
        sample["qa"] = filtered_qa
    output = args.output or f"locomo_sample{args.sample_index}_raw.json"
    save_json(output, [sample] if args.as_list else sample)
    print(json.dumps(
        {
            "output": str(Path(output).resolve()),
            "sample_index": args.sample_index,
            "sample_id": sample.get("sample_id") if isinstance(sample, dict) else None,
            "qa_count": len(sample.get("qa", [])) if isinstance(sample, dict) and isinstance(sample.get("qa"), list) else None,
            "dropped_empty_answer_count": dropped_count,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
