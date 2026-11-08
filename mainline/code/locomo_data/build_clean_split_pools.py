from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SAMPLE_RE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean train/dev/test curriculum pools from an environment split manifest.")
    parser.add_argument("--manifest", default="curriculum/clean_v1_split_manifest.json")
    parser.add_argument("--output-prefix", default="curriculum/clean_v1")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_index(record: dict[str, Any]) -> int | None:
    value = record.get("sample_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    sample_name = str(record.get("sample_name") or "")
    match = SAMPLE_RE.search(sample_name)
    if match:
        return int(match.group(1))
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    sample_name = str(meta.get("sample_name") or "")
    match = SAMPLE_RE.search(sample_name)
    if match:
        return int(match.group(1))
    return None


def split_name(index: int | None, manifest: dict[str, Any]) -> str | None:
    if index in set(manifest.get("train_samples", [])):
        return "train"
    if index in set(manifest.get("dev_samples", [])):
        return "dev"
    if index in set(manifest.get("test_samples", [])):
        return "test"
    return None


def count_map(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value is None and isinstance(record.get("metadata"), dict):
            value = record["metadata"].get(key)
        value = str(value or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def average(records: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None and isinstance(record.get("metadata"), dict):
            value = record["metadata"].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, int] = {}
    for record in records:
        idx = sample_index(record)
        key = f"sample{idx}" if idx is not None else "unknown"
        by_sample[key] = by_sample.get(key, 0) + 1
    return {
        "count": len(records),
        "by_sample": dict(sorted(by_sample.items())),
        "training_use_counts": count_map(records, "training_use"),
        "difficulty_counts": count_map(records, "difficulty_for_solver"),
        "quality_tier_counts": count_map(records, "quality_tier"),
        "evolution_type_counts": count_map(records, "evolution_type"),
        "candidate_type_counts": count_map(records, "candidate_type"),
        "average_full_context_accuracy": average(records, "full_context_accuracy"),
        "average_evidence_window_accuracy": average(records, "evidence_window_accuracy"),
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_prefix = Path(args.output_prefix)

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": [], "unknown": []}
    source_status: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    for source in manifest.get("known_sources", []):
        path_value = source.get("path")
        expected_samples = set(source.get("samples", []))
        if not path_value:
            source_status.append({**source, "loaded_count": 0, "included_count": 0})
            continue
        path = Path(path_value)
        if not path.exists():
            source_status.append({**source, "loaded_count": 0, "included_count": 0, "missing_file": True})
            continue
        # Load a shared file once; filter per source sample list below.
        all_records = load_jsonl(path)
        included = 0
        for record in all_records:
            idx = sample_index(record)
            if expected_samples and idx not in expected_samples:
                continue
            split = split_name(idx, manifest) or "unknown"
            enriched = dict(record)
            enriched.setdefault("metadata", record.get("metadata") if isinstance(record.get("metadata"), dict) else {})
            enriched["metadata"] = dict(enriched["metadata"])
            enriched["metadata"]["clean_split_id"] = manifest.get("split_id")
            enriched["metadata"]["clean_split"] = split
            enriched["metadata"]["source_manifest_entry"] = source.get("name")
            records_by_split[split].append(enriched)
            included += 1
        source_status.append({**source, "loaded_count": len(all_records), "included_count": included})
        seen_paths.add(path)

    for split, records in records_by_split.items():
        records.sort(key=lambda r: (sample_index(r) if sample_index(r) is not None else 999, str(r.get("record_id"))))
        write_jsonl(output_prefix.with_name(f"{output_prefix.name}_{split}.jsonl"), records)
        write_json(output_prefix.with_name(f"{output_prefix.name}_{split}.json"), {"records": records})

    payload = {
        "manifest": str(manifest_path),
        "output_prefix": str(output_prefix),
        "source_status": source_status,
        "splits": {split: summarize(records) for split, records in records_by_split.items()},
        "missing_train_samples": [
            item for item in manifest.get("known_sources", [])
            if item.get("split") == "train" and str(item.get("status", "")).startswith("missing")
        ],
        "notes": [
            "sample4 full train pool is currently missing, so clean_v1_train is partial until sample4 is generated.",
            "sample6 is written only to dev, even though it shares a source file with sample5.",
        ],
    }
    write_json(output_prefix.with_name(f"{output_prefix.name}_summary.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
