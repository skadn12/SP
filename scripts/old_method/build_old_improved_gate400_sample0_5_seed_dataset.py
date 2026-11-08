from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_SAMPLE_NAMES = {f"sample{i}" for i in range(6)}
R2_S0_S2_PATH = Path("eval_sets/graph_expansion_challenger_v1/r2_auto_s0_s2_incremental_full/samples.jsonl")
R3_AUTO_FULL_PATH = Path("eval_sets/graph_expansion_challenger_v1/r3_auto_full/samples.jsonl")
OUTPUT_DIR = Path("eval_sets/graph_expansion_challenger_v1/old_improved_gate400_sample0_5_seed_aligned")
OUTPUT_PATH = OUTPUT_DIR / "samples.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
README_PATH = OUTPUT_DIR / "README.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_dialogue_path(row: dict[str, Any]) -> str:
    state_chain = row.get("state_chain") if isinstance(row.get("state_chain"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for value in (
        row.get("source_dialogue_path"),
        metadata.get("source_dialogue_path"),
        state_chain.get("source_dialogue_path"),
    ):
        if value:
            return str(value)
    return ""


def infer_source_split_and_sample(row: dict[str, Any]) -> tuple[str, str]:
    source_path = source_dialogue_path(row)
    match = re.search(r"/(train|dev|test)/(sample\d+)/", source_path)
    if match:
        return match.group(1), match.group(2)
    sample_match = re.search(r"/(sample\d+)/", source_path)
    if sample_match:
        return "unknown", sample_match.group(1)
    return "unknown", "unknown"


def normalize_row(row: dict[str, Any], source_dataset: str) -> dict[str, Any]:
    normalized = copy.deepcopy(row)
    source_split, sample_name = infer_source_split_and_sample(normalized)
    metadata = normalized.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        normalized["metadata"] = metadata
    normalized["split"] = "train"
    normalized["heldout_split"] = None
    normalized["seed_alignment"] = {
        "seed_scope": "sample0_5_train_only",
        "source_dataset": source_dataset,
        "source_split": source_split,
        "sample_name": sample_name,
        "source_dialogue_path": source_dialogue_path(normalized),
    }
    metadata["seed_alignment_source_dataset"] = source_dataset
    metadata["seed_alignment_sample_name"] = sample_name
    metadata["seed_alignment_source_split"] = source_split
    return normalized


def select_rows(rows: list[dict[str, Any]], source_dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        source_split, sample_name = infer_source_split_and_sample(row)
        row_split = row.get("split")
        if source_split != "train" or sample_name not in ALLOWED_SAMPLE_NAMES:
            rejected.append(
                {
                    "sample_id": row.get("sample_id"),
                    "source_dataset": source_dataset,
                    "row_split": row_split,
                    "source_split": source_split,
                    "sample_name": sample_name,
                    "source_dialogue_path": source_dialogue_path(row),
                }
            )
            continue
        if source_dataset == "r3_auto_full_train" and row_split != "train":
            rejected.append(
                {
                    "sample_id": row.get("sample_id"),
                    "source_dataset": source_dataset,
                    "row_split": row_split,
                    "source_split": source_split,
                    "sample_name": sample_name,
                    "source_dialogue_path": source_dialogue_path(row),
                }
            )
            continue
        selected.append(normalize_row(row, source_dataset))
    return selected, rejected


def main() -> int:
    r2_rows = load_jsonl(R2_S0_S2_PATH)
    r3_rows = load_jsonl(R3_AUTO_FULL_PATH)

    r2_selected, r2_rejected = select_rows(r2_rows, "r2_auto_s0_s2_incremental_full")
    r3_selected, r3_rejected = select_rows(r3_rows, "r3_auto_full_train")
    selected = r2_selected + r3_selected

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for row in selected:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)
            continue
        seen_ids.add(sample_id)
        deduped.append(row)

    deduped.sort(
        key=lambda row: (
            row.get("seed_alignment", {}).get("sample_name") or "",
            row.get("seed_alignment", {}).get("source_dataset") or "",
            row.get("sample_id") or "",
        )
    )

    write_jsonl(OUTPUT_PATH, deduped)
    summary = {
        "output_path": str(OUTPUT_PATH),
        "sources": {
            "r2_auto_s0_s2_incremental_full": str(R2_S0_S2_PATH),
            "r3_auto_full_train": str(R3_AUTO_FULL_PATH),
        },
        "allowed_sample_names": sorted(ALLOWED_SAMPLE_NAMES),
        "selected_count": len(deduped),
        "source_dataset_counts": dict(
            sorted(Counter(row.get("seed_alignment", {}).get("source_dataset") or "unknown" for row in deduped).items())
        ),
        "sample_name_counts": dict(
            sorted(Counter(row.get("seed_alignment", {}).get("sample_name") or "unknown" for row in deduped).items())
        ),
        "rejected_count": len(r2_rejected) + len(r3_rejected),
        "rejected_by_source_dataset": {
            "r2_auto_s0_s2_incremental_full": len(r2_rejected),
            "r3_auto_full_train": len(r3_rejected),
        },
        "duplicate_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:20],
        "pass": len(deduped) > 0,
    }
    write_json(SUMMARY_PATH, summary)
    README_PATH.write_text(
        "\n".join(
            [
                "# old_improved_gate400_sample0_5 Seed Dataset",
                "",
                "This aligned seed dataset combines:",
                "- `r2_auto_s0_s2_incremental_full` for train `sample0-2`",
                "- `r3_auto_full` train rows for `sample3-5`",
                "",
                "All exported rows are normalized to:",
                "- `split = train`",
                "- `heldout_split = null`",
                "- `seed_alignment.sample_name in sample0..sample5`",
                "",
                f"Selected rows: {len(deduped)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved aligned seed dataset to: {OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
