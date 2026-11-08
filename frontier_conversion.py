from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from frontier_filter import assign_frontier_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how many previously too_easy samples converted to frontier after another round."
    )
    parser.add_argument("--before", required=True, help="Earlier evaluated/frontier-labeled JSON.")
    parser.add_argument("--after", required=True, help="Later evaluated/frontier-labeled JSON.")
    parser.add_argument("--output", default=None, help="Optional path to save the conversion report JSON.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_result_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Input root must be a JSON object.")


def result_key(result: dict[str, Any]) -> str:
    return f"{result.get('sample_id', result.get('sample_index', ''))}::{result.get('qa_index', '')}"


def get_frontier_label(result: dict[str, Any]) -> str:
    label = result.get("frontier_label")
    if isinstance(label, str) and label:
        return label
    model_eval = result.get("model_eval")
    if isinstance(model_eval, dict):
        label = model_eval.get("frontier_label")
        if isinstance(label, str) and label:
            return label
        return assign_frontier_label(model_eval)
    return "invalid"


def build_conversion_report(before_payload: Any, after_payload: Any) -> dict[str, Any]:
    before_items = collect_result_items(before_payload)
    after_by_key = {result_key(item): item for item in collect_result_items(after_payload)}
    conversions: list[dict[str, Any]] = []
    missing_after: list[dict[str, Any]] = []

    for before in before_items:
        before_label = get_frontier_label(before)
        if before_label != "too_easy":
            continue
        key = result_key(before)
        after = after_by_key.get(key)
        if after is None:
            missing_after.append(
                {
                    "key": key,
                    "qa_index": before.get("qa_index"),
                    "question": before.get("qa", {}).get("question"),
                }
            )
            continue
        after_label = get_frontier_label(after)
        before_eval = before.get("model_eval") if isinstance(before.get("model_eval"), dict) else {}
        after_eval = after.get("model_eval") if isinstance(after.get("model_eval"), dict) else {}
        conversions.append(
            {
                "key": key,
                "sample_id": before.get("sample_id"),
                "qa_index": before.get("qa_index"),
                "question": before.get("qa", {}).get("question"),
                "before_label": before_label,
                "after_label": after_label,
                "converted_to_frontier": after_label == "frontier",
                "before_accuracy": before_eval.get("average_accuracy"),
                "after_accuracy": after_eval.get("average_accuracy"),
                "accuracy_delta": (
                    round(float(after_eval["average_accuracy"]) - float(before_eval["average_accuracy"]), 4)
                    if isinstance(before_eval.get("average_accuracy"), (int, float))
                    and isinstance(after_eval.get("average_accuracy"), (int, float))
                    else None
                ),
            }
        )

    too_easy_count = len(conversions) + len(missing_after)
    converted_count = sum(1 for item in conversions if item["converted_to_frontier"])
    return {
        "before_too_easy_count": too_easy_count,
        "matched_after_count": len(conversions),
        "missing_after_count": len(missing_after),
        "converted_to_frontier_count": converted_count,
        "frontier_conversion_rate": round(converted_count / too_easy_count, 4) if too_easy_count else None,
        "conversions": conversions,
        "missing_after": missing_after,
    }


def main() -> int:
    args = parse_args()
    report = build_conversion_report(load_json(Path(args.before)), load_json(Path(args.after)))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved frontier conversion report to: {Path(args.output).resolve()}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
