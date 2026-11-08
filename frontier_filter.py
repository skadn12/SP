from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


FRONTIER_LABELS = ("too_easy", "medium", "frontier", "hard", "risky", "invalid")


def build_frontier_output_path(input_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return input_path.with_name(f"{input_path.stem}_frontier_{timestamp}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label evaluated samples by solver-panel average accuracy so the next "
            "challenger round can decide what to make harder, keep, or inspect."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON produced by evaluate_model_difficulty.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the labeled JSON. If omitted, a timestamped sidecar file is generated.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a new sidecar JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assign_frontier_label(model_eval: dict[str, Any]) -> str:
    acc = model_eval.get("average_accuracy")

    if acc is None:
        return "invalid"

    if acc >= 0.8:
        return "too_easy"
    if acc >= 0.6:
        return "medium"
    if acc >= 0.4:
        return "frontier"
    if acc >= 0.2:
        return "hard"
    return "risky"


def collect_result_items(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)], True
    if isinstance(payload, dict) and "model_eval" in payload:
        return [payload], False
    raise ValueError(
        "Input JSON must be either a single evaluated result object or a batch object with a 'results' list."
    )


def label_result(result: dict[str, Any]) -> str:
    model_eval = result.get("model_eval")
    if not isinstance(model_eval, dict):
        label = "invalid"
    else:
        label = assign_frontier_label(model_eval)
        model_eval["frontier_label"] = label
    result["frontier_label"] = label
    return label


def build_frontier_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = {label: 0 for label in FRONTIER_LABELS}
    valid_accuracy_count = 0
    accuracies: list[float] = []

    for result in results:
        label = str(result.get("frontier_label", "invalid"))
        if label not in label_counts:
            label = "invalid"
        label_counts[label] += 1

        model_eval = result.get("model_eval")
        if not isinstance(model_eval, dict):
            continue
        acc = model_eval.get("average_accuracy")
        if isinstance(acc, (int, float)) and not isinstance(acc, bool):
            valid_accuracy_count += 1
            accuracies.append(float(acc))

    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    return {
        "labeled_case_count": len(results),
        "valid_accuracy_count": valid_accuracy_count,
        "label_counts": label_counts,
        "average_accuracy": round(average_accuracy, 4) if average_accuracy is not None else None,
        "labeled_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def main() -> int:
    args = parse_args()
    if args.output and args.in_place:
        raise ValueError("Use either --output or --in-place, not both.")

    input_path = Path(args.input)
    payload = load_json(input_path)
    result_items, is_batch = collect_result_items(payload)

    for result in result_items:
        label_result(result)

    summary = build_frontier_summary(result_items)
    if is_batch and isinstance(payload, dict):
        payload["frontier_summary"] = summary
    elif isinstance(payload, dict):
        payload["frontier_summary"] = summary

    if args.in_place:
        output_path = input_path
    else:
        output_path = Path(args.output) if args.output else build_frontier_output_path(input_path)

    save_json(output_path, payload)
    print(f"Saved frontier-labeled result to: {output_path.resolve()}")
    print(f"Label counts: {summary['label_counts']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
        raise
