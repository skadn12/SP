from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a long-dialogue memory QA solver. Read the full dialogue carefully, "
    "track time, speakers, relations, and state updates, then choose the single correct option. "
    "Answer with only the option letter and the option text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert curriculum QA JSONL into SFT chat/alpaca files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="sft_data")
    parser.add_argument("--name", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-valid", type=int, default=80)
    parser.add_argument(
        "--shuffle-options",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically shuffle MCQ options per record and update the gold label.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Keep metadata fields in output JSONL for analysis. Training frameworks usually ignore extra keys.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            records.append(item)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def option_label(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return str(index)


def normalize_options(record: dict[str, Any]) -> list[str]:
    options = record.get("options")
    if not isinstance(options, list):
        return []
    return [str(option).strip() for option in options]


def correct_index(record: dict[str, Any], options: list[str]) -> int | None:
    value = record.get("correct_option_index")
    if isinstance(value, int) and 0 <= value < len(options):
        return value
    label = str(record.get("correct_option_label") or "").strip().upper()
    if len(label) == 1 and "A" <= label <= "Z":
        index = ord(label) - ord("A")
        if 0 <= index < len(options):
            return index
    answer = str(record.get("answer") or "").strip()
    for index, option in enumerate(options):
        if option == answer:
            return index
    return None




def shuffle_options(record: dict[str, Any], seed: int) -> dict[str, Any]:
    options = normalize_options(record)
    index = correct_index(record, options)
    if index is None or len(options) < 2:
        return record

    key = f"{seed}:{record.get('record_id') or record.get('sample_name') or ''}:{record.get('question') or ''}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    permutation = list(range(len(options)))
    rng.shuffle(permutation)
    if permutation == list(range(len(options))) and len(permutation) > 1:
        permutation = permutation[1:] + permutation[:1]

    shuffled = copy.deepcopy(record)
    shuffled_options = [options[old_index] for old_index in permutation]
    new_index = permutation.index(index)
    shuffled["options"] = shuffled_options
    shuffled["correct_option_index"] = new_index
    shuffled["correct_option_label"] = option_label(new_index)
    shuffled["answer"] = shuffled_options[new_index]

    metadata = shuffled.get("metadata") if isinstance(shuffled.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata["option_shuffle_enabled"] = True
    metadata["original_correct_option_label"] = option_label(index)
    metadata["shuffled_correct_option_label"] = option_label(new_index)
    metadata["option_permutation_old_indices"] = permutation
    shuffled["metadata"] = metadata
    return shuffled


def build_user_prompt(record: dict[str, Any]) -> str:
    dialogue = str(record.get("dialogue_text") or "").strip()
    question = str(record.get("question") or "").strip()
    options = normalize_options(record)
    option_lines = [f"{option_label(index)}. {option}" for index, option in enumerate(options)]
    return "\n\n".join(
        [
            "Dialogue:",
            dialogue,
            "Question:",
            question,
            "Options:",
            "\n".join(option_lines),
        ]
    )


def build_answer(record: dict[str, Any]) -> str:
    options = normalize_options(record)
    index = correct_index(record, options)
    if index is None:
        raise ValueError(f"Cannot determine correct option for record {record.get('record_id')}")
    return f"{option_label(index)}. {options[index]}"


def compact_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return {
        "record_id": record.get("record_id"),
        "sample_name": record.get("sample_name"),
        "evolution_type": metadata.get("evolution_type"),
        "candidate_type": metadata.get("candidate_type"),
        "difficulty_for_solver": metadata.get("difficulty_for_solver"),
        "quality_tier": metadata.get("quality_tier"),
        "training_use": metadata.get("training_use"),
        "full_context_accuracy": metadata.get("full_context_accuracy"),
        "evidence_window_accuracy": metadata.get("evidence_window_accuracy"),
        "retrieval_gap": metadata.get("retrieval_gap"),
        "option_shuffle_enabled": metadata.get("option_shuffle_enabled"),
        "original_correct_option_label": metadata.get("original_correct_option_label"),
        "shuffled_correct_option_label": metadata.get("shuffled_correct_option_label"),
    }


def to_chat(record: dict[str, Any], include_metadata: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
            {"role": "assistant", "content": build_answer(record)},
        ]
    }
    if include_metadata:
        payload["metadata"] = compact_metadata(record)
    return payload


def to_alpaca(record: dict[str, Any], include_metadata: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "system": SYSTEM_PROMPT,
        "instruction": "Read the dialogue and answer the multiple-choice question.",
        "input": build_user_prompt(record),
        "output": build_answer(record),
    }
    if include_metadata:
        payload["metadata"] = compact_metadata(record)
    return payload


def group_key(record: dict[str, Any]) -> tuple[str, str, str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return (
        str(record.get("sample_name") or "unknown"),
        str(metadata.get("training_use") or "unknown"),
        str(metadata.get("evolution_type") or "unknown"),
    )


def split_records(records: list[dict[str, Any]], valid_ratio: float, max_valid: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        by_group.setdefault(group_key(record), []).append(record)

    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for group_records in by_group.values():
        items = list(group_records)
        rng.shuffle(items)
        group_valid_count = int(round(len(items) * valid_ratio))
        if len(items) >= 10:
            group_valid_count = max(1, group_valid_count)
        valid.extend(items[:group_valid_count])
        train.extend(items[group_valid_count:])

    if len(valid) > max_valid:
        rng.shuffle(valid)
        train.extend(valid[max_valid:])
        valid = valid[:max_valid]

    train.sort(key=lambda item: str(item.get("record_id")))
    valid.sort(key=lambda item: str(item.get("record_id")))
    return train, valid


def count_map(records: list[dict[str, Any]], metadata_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        value = str(metadata.get(metadata_key) or record.get(metadata_key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "sample_counts": count_map(records, "sample_name"),
        "training_use_counts": count_map(records, "training_use"),
        "difficulty_counts": count_map(records, "difficulty_for_solver"),
        "quality_tier_counts": count_map(records, "quality_tier"),
        "evolution_type_counts": count_map(records, "evolution_type"),
        "candidate_type_counts": count_map(records, "candidate_type"),
        "correct_label_counts": count_map(records, "correct_option_label"),
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    name = args.name or input_path.stem
    output_dir = Path(args.output_dir)
    raw_records = read_jsonl(input_path)

    valid_source: list[dict[str, Any]] = []
    usable_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in raw_records:
        try:
            _ = build_answer(record)
            usable_records.append(record)
        except Exception as exc:
            skipped.append({"record_id": record.get("record_id"), "reason": str(exc)})

    train_source, valid_source = split_records(usable_records, args.valid_ratio, args.max_valid, args.seed)
    if args.shuffle_options:
        train_source = [shuffle_options(record, args.seed) for record in train_source]
        valid_source = [shuffle_options(record, args.seed) for record in valid_source]
    train_chat = [to_chat(record, args.include_metadata) for record in train_source]
    valid_chat = [to_chat(record, args.include_metadata) for record in valid_source]
    train_alpaca = [to_alpaca(record, args.include_metadata) for record in train_source]
    valid_alpaca = [to_alpaca(record, args.include_metadata) for record in valid_source]

    write_jsonl(output_dir / f"{name}_train_chat.jsonl", train_chat)
    write_jsonl(output_dir / f"{name}_valid_chat.jsonl", valid_chat)
    write_jsonl(output_dir / f"{name}_train_alpaca.jsonl", train_alpaca)
    write_jsonl(output_dir / f"{name}_valid_alpaca.jsonl", valid_alpaca)

    summary = {
        "input": str(input_path),
        "name": name,
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "max_valid": args.max_valid,
        "shuffle_options": args.shuffle_options,
        "raw_count": len(raw_records),
        "usable_count": len(usable_records),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "train": summarize(train_source),
        "valid": summarize(valid_source),
        "output_files": {
            "train_chat": str((output_dir / f"{name}_train_chat.jsonl").resolve()),
            "valid_chat": str((output_dir / f"{name}_valid_chat.jsonl").resolve()),
            "train_alpaca": str((output_dir / f"{name}_train_alpaca.jsonl").resolve()),
            "valid_alpaca": str((output_dir / f"{name}_valid_alpaca.jsonl").resolve()),
        },
    }
    summary_path = output_dir / f"{name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved SFT summary to: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
