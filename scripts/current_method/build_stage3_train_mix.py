#!/usr/bin/env python3
"""Build a leakage-audited Stage3 train mix from a ready accumulation manifest.

The official Stage3 heldout partition is never copied into training or model
selection data.  It is loaded only to build a strict exclusion index.  Any
optional validation data is reserved from the balanced training partition by
lineage group, so training wrappers do not need to reuse the official heldout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import api_challenger_generation as generation  # noqa: E402


MANIFEST_SCHEMA = "stage3_training_ready_manifest_v1"
OUTPUT_SCHEMA = "stage3_train_mix_v1"
PROVENANCE_SCHEMA = "stage3_train_mix_provenance_v1"
IDENTITY_KEYS = (
    "sample_id",
    "question_id",
    "dialogue_hash",
    "lineage_group",
    "content_fingerprint",
)
DEDUPE_KEYS = ("sample_id", "question_id", "content_fingerprint")
TRAIN_CONTENT_KEYS = ("sample_id", "question_id", "dialogue_hash", "content_fingerprint")
SUPPORTED_GROUP_LEVELS = {"lineage_group", "parent_sample_id"}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in iter_jsonl(path)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def artifact_record(path: Path, *, role: str, row_count: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        record["row_count"] = int(row_count)
    return record


def resolve_artifact_path(manifest_path: Path, raw_path: Any) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError(f"Manifest {manifest_path} contains an empty artifact path")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    candidates = (
        (manifest_path.parent / candidate).resolve(),
        (ROOT / candidate).resolve(),
        (Path.cwd() / candidate).resolve(),
    )
    for resolved in candidates:
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Could not resolve manifest artifact {text!r}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def qa(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("qa") if isinstance(row.get("qa"), dict) else {}


def dialogue_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue = row.get("dialogue")
    if isinstance(dialogue, dict) and isinstance(dialogue.get("turns"), list):
        return [turn for turn in dialogue["turns"] if isinstance(turn, dict)]
    if isinstance(dialogue, list):
        return [turn for turn in dialogue if isinstance(turn, dict)]
    final_dialogue = row.get("final_dialogue")
    if isinstance(final_dialogue, list):
        return [turn for turn in final_dialogue if isinstance(turn, dict)]
    return []


def normalized_dialogue_payload(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "speaker": normalize_text(turn.get("speaker") or turn.get("role")),
            "session": normalize_text(
                turn.get("session_id") or turn.get("session") or turn.get("date")
            ),
            "text": normalize_text(
                turn.get("text") or turn.get("content") or turn.get("utterance")
            ),
        }
        for turn in dialogue_turns(row)
    ]


def sample_identity(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("expanded_graph_id") or "").strip()


def normalized_options(row: dict[str, Any]) -> list[tuple[str, str]]:
    value = qa(row).get("options")
    if value is None:
        value = row.get("options")
    options: list[tuple[str, str]] = []
    if isinstance(value, dict):
        options = [(normalize_text(key), normalize_text(text)) for key, text in value.items()]
    elif isinstance(value, list):
        for index, option in enumerate(value):
            if isinstance(option, dict):
                option_id = option.get("id") or option.get("label") or index
                option_text = option.get("text") or option.get("content")
            else:
                option_id = index
                option_text = option
            options.append((normalize_text(option_id), normalize_text(option_text)))
    return options


def normalized_gold_texts(row: dict[str, Any], options: Sequence[tuple[str, str]]) -> list[str]:
    value = qa(row).get("gold")
    if value is None:
        value = row.get("gold_options") or row.get("gold_answer") or row.get("gold_answers")
    gold = value if isinstance(value, list) else [value]
    gold_ids = {normalize_text(item) for item in gold if normalize_text(item)}
    option_map = dict(options)
    texts = [option_map.get(item, item) for item in gold_ids]
    return sorted(text for text in texts if text)


def normalized_question_payload(row: dict[str, Any]) -> dict[str, Any]:
    question = qa(row).get("question") or row.get("question")
    question_type = qa(row).get("question_type") or row.get("question_type")
    options = normalized_options(row)
    return {
        "question_type": normalize_text(question_type),
        "question": normalize_text(question),
        # Sorting text makes an option-label permutation the same semantic item.
        "option_texts": sorted(text for _, text in options),
        "gold_texts": normalized_gold_texts(row, options),
    }


def question_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = row.get("question_id") or qa(row).get("question_id") or qa(row).get("qa_id")
    value = value or metadata.get("source_question_id")
    text = str(value or "").strip()
    if text:
        return text
    return "derived:" + payload_sha256(normalized_question_payload(row))


def dialogue_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    dialogue = row.get("dialogue") if isinstance(row.get("dialogue"), dict) else {}
    value = row.get("dialogue_hash") or dialogue.get("dialogue_hash")
    value = value or metadata.get("source_dialogue_hash")
    text = str(value or "").strip()
    if text:
        return text
    return "derived:" + payload_sha256(normalized_dialogue_payload(row))


def primary_lineage_group(
    row: dict[str, Any], *, group_level: str = "lineage_group"
) -> str:
    if group_level not in SUPPORTED_GROUP_LEVELS:
        raise ValueError(f"Unsupported manifest group_level: {group_level!r}")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    seed = row.get("seed") if isinstance(row.get("seed"), dict) else {}
    seed_origin = seed.get("source") if isinstance(seed.get("source"), dict) else {}
    if group_level == "lineage_group":
        candidates = (
            row.get("lineage_group"),
            metadata.get("lineage_group"),
            seed_source.get("lineage_group"),
            seed_origin.get("lineage_group"),
            seed.get("lineage_group"),
            # Compatibility fallback for old replay rows that predate the
            # explicit lineage_group propagation.
            seed_source.get("parent_sample_id"),
            seed_origin.get("parent_sample_id"),
            seed.get("parent_sample_id"),
            row.get("parent_sample_id"),
            metadata.get("root_seed_parent_id"),
            seed_source.get("root_seed_parent_id"),
        )
    else:
        candidates = (
            seed_source.get("parent_sample_id"),
            seed_origin.get("parent_sample_id"),
            seed.get("parent_sample_id"),
            row.get("parent_sample_id"),
            row.get("lineage_group"),
            metadata.get("lineage_group"),
        )
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return dialogue_identity(row) or sample_identity(row)


def content_fingerprint(row: dict[str, Any]) -> str:
    return payload_sha256(
        {
            "dialogue": normalized_dialogue_payload(row),
            "qa": normalized_question_payload(row),
        }
    )


def identity_values(
    row: dict[str, Any], *, group_level: str = "lineage_group"
) -> dict[str, set[str]]:
    sample = sample_identity(row)
    group = primary_lineage_group(row, group_level=group_level)
    return {
        "sample_id": {sample} if sample else set(),
        "question_id": {question_identity(row)},
        "dialogue_hash": {dialogue_identity(row)},
        # Exactly one canonical group is used.  Reusable raw source/chunk IDs
        # are deliberately not unioned into this hard leakage key.
        "lineage_group": {group} if group else set(),
        "content_fingerprint": {content_fingerprint(row)},
    }


def empty_identity_index() -> dict[str, set[str]]:
    return {key: set() for key in IDENTITY_KEYS}


def add_to_identity_index(
    index: dict[str, set[str]],
    row: dict[str, Any],
    *,
    group_level: str = "lineage_group",
) -> None:
    identities = identity_values(row, group_level=group_level)
    for key in IDENTITY_KEYS:
        index[key].update(identities[key])


def build_identity_index(
    rows: Iterable[dict[str, Any]], *, group_level: str = "lineage_group"
) -> dict[str, set[str]]:
    index = empty_identity_index()
    for row in rows:
        add_to_identity_index(index, row, group_level=group_level)
    return index


def overlap_details(
    row: dict[str, Any],
    index: dict[str, set[str]],
    *,
    keys: Sequence[str] = IDENTITY_KEYS,
    group_level: str = "lineage_group",
) -> dict[str, list[str]]:
    identities = identity_values(row, group_level=group_level)
    return {
        key: sorted(identities[key] & index[key])
        for key in keys
        if identities[key] & index[key]
    }


def overlap_summary(
    left: Iterable[dict[str, Any]],
    right: Iterable[dict[str, Any]],
    *,
    group_level: str = "lineage_group",
) -> dict[str, Any]:
    right_index = build_identity_index(right, group_level=group_level)
    counts = Counter()
    rows_with_overlap = 0
    for row in left:
        overlap = overlap_details(row, right_index, group_level=group_level)
        if overlap:
            rows_with_overlap += 1
            for key, matches in overlap.items():
                counts[key] += len(matches)
    by_identity = {key: int(counts.get(key, 0)) for key in IDENTITY_KEYS}
    return {
        "rows_with_overlap": rows_with_overlap,
        "by_identity": by_identity,
        "total_identity_collisions": sum(by_identity.values()),
        "zero": rows_with_overlap == 0,
    }


def validate_sft_source_rows(rows: Iterable[dict[str, Any]], *, role: str) -> None:
    for index, row in enumerate(rows):
        checks = {
            "sample_id": bool(sample_identity(row)),
            "dialogue_turns": bool(dialogue_turns(row)),
            "qa": bool(qa(row)),
            "question": bool(str(qa(row).get("question") or row.get("question") or "").strip()),
            "options": bool(normalized_options(row)),
        }
        if not all(checks.values()):
            raise ValueError(f"Invalid {role} row at index {index}: {checks}")


def dedupe_rows(
    rows: Sequence[dict[str, Any]],
    *,
    source_label: str,
    group_level: str = "lineage_group",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = empty_identity_index()
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        duplicate = overlap_details(
            row, seen, keys=DEDUPE_KEYS, group_level=group_level
        )
        if duplicate:
            excluded.append(
                {
                    "source": source_label,
                    "source_position": position,
                    "sample_id": sample_identity(row),
                    "reason": "duplicate",
                    "identity_matches": duplicate,
                }
            )
            continue
        kept.append(row)
        add_to_identity_index(seen, row, group_level=group_level)
    return kept, excluded


def deterministic_rank(seed: int, *parts: Any) -> str:
    return payload_sha256({"seed": int(seed), "parts": parts})


def split_train_derived_validation(
    rows: Sequence[dict[str, Any]],
    *,
    ratio: float,
    seed: int,
    group_level: str = "lineage_group",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[primary_lineage_group(row, group_level=group_level)].append(row)
    group_ids = sorted(grouped, key=lambda group: (deterministic_rank(seed, group), group))
    if ratio <= 0.0:
        return list(rows), [], {
            "status": "no_valid",
            "reason": "validation_ratio_zero",
            "requested_ratio": ratio,
            "source_group_count": len(group_ids),
        }
    if len(group_ids) < 2:
        return list(rows), [], {
            "status": "no_valid",
            "reason": "fewer_than_two_train_lineage_groups",
            "requested_ratio": ratio,
            "source_group_count": len(group_ids),
        }

    target_rows = max(1, int(math.floor(len(rows) * ratio + 0.5)))
    selected_groups: list[str] = []
    selected_count = 0
    for group in group_ids[:-1]:
        selected_groups.append(group)
        selected_count += len(grouped[group])
        if selected_count >= target_rows:
            break
    selected_set = set(selected_groups)
    train = [
        row
        for row in rows
        if primary_lineage_group(row, group_level=group_level) not in selected_set
    ]
    validation = [
        row
        for row in rows
        if primary_lineage_group(row, group_level=group_level) in selected_set
    ]
    if not train or not validation:
        return list(rows), [], {
            "status": "no_valid",
            "reason": "whole_group_split_could_not_leave_both_partitions_nonempty",
            "requested_ratio": ratio,
            "source_group_count": len(group_ids),
        }
    leakage = overlap_summary(train, validation, group_level=group_level)
    if not leakage["zero"]:
        raise RuntimeError(f"Train-derived validation leakage detected: {leakage}")
    return train, validation, {
        "status": "ready",
        "reason": None,
        "requested_ratio": ratio,
        "target_rows": target_rows,
        "selected_rows": len(validation),
        "actual_ratio": round(len(validation) / len(rows), 8),
        "source_group_count": len(group_ids),
        "selected_group_count": len(selected_groups),
        "selected_groups": selected_groups,
        "group_level": group_level,
        "train_group_count": len(group_ids) - len(selected_groups),
        "train_validation_overlap": leakage,
    }


def replay_target(frontier_count: int, replay_ratio: float) -> int:
    if replay_ratio <= 0.0 or frontier_count <= 0:
        return 0
    raw = frontier_count * replay_ratio / (1.0 - replay_ratio)
    return max(1, int(math.floor(raw + 0.5)))


def stamp_row(
    row: dict[str, Any],
    *,
    role: str,
    source_path: Path,
    source_sha256: str,
    group_level: str = "lineage_group",
) -> dict[str, Any]:
    output = copy.deepcopy(row)
    metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
    metadata = copy.deepcopy(metadata)
    metadata.update(
        {
            "stage3_train_mix_schema": OUTPUT_SCHEMA,
            "stage3_train_mix_role": role,
            "stage3_train_mix_source_path": str(source_path.resolve()),
            "stage3_train_mix_source_sha256": source_sha256,
            "prompt_mode": "full_dialogue",
            "evidence_window_used_as_prompt": False,
        }
    )
    output["metadata"] = metadata
    output["train_mix_role"] = role
    output["question_id"] = question_identity(output)
    output["dialogue_hash"] = dialogue_identity(output)
    output["stage3_content_fingerprint"] = content_fingerprint(output)
    output["stage3_lineage_group"] = primary_lineage_group(
        output, group_level=group_level
    )
    return output


def build_sft(
    rows: list[dict[str, Any]],
    *,
    role_by_sample: dict[str, str],
    group_level: str = "lineage_group",
) -> list[dict[str, Any]]:
    records = generation.build_sft_records(rows)
    if len(records) != len(rows):
        raise RuntimeError("build_sft_records returned a mismatched record count")
    for row, record in zip(rows, records):
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        metadata = copy.deepcopy(metadata)
        metadata.update(
            {
                "stage3_train_mix_schema": OUTPUT_SCHEMA,
                "stage3_train_mix_role": role_by_sample.get(sample_identity(row), row.get("train_mix_role")),
                "source_sample_id": sample_identity(row),
                "source_question_id": question_identity(row),
                "source_dialogue_hash": dialogue_identity(row),
                "source_lineage_group": primary_lineage_group(
                    row, group_level=group_level
                ),
                "source_content_fingerprint": content_fingerprint(row),
                "official_stage3_heldout_used": False,
            }
        )
        record["metadata"] = metadata
    return records


def validate_ready_manifest(manifest: dict[str, Any], *, path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(
            f"Expected {MANIFEST_SCHEMA} at {path}, got {manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "training_ready":
        raise RuntimeError(f"Accumulation manifest is not training_ready: {path}")
    if "training_ready" in manifest and manifest.get("training_ready") is not True:
        raise RuntimeError(f"Manifest training_ready gate is not true: {path}")
    gates = manifest.get("readiness_gates")
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        raise RuntimeError(f"Not all accumulation readiness gates are true: {gates!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Manifest artifacts missing: {path}")
    for key in ("balanced_frontier_train", "balanced_frontier_heldout"):
        if not artifacts.get(key):
            raise ValueError(f"Manifest artifact {key!r} missing: {path}")
    group_level = str(manifest.get("group_level") or "").strip()
    if group_level not in SUPPORTED_GROUP_LEVELS:
        raise ValueError(
            f"Manifest group_level must be one of {sorted(SUPPORTED_GROUP_LEVELS)}, "
            f"got {group_level!r}: {path}"
        )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the audited Stage3 train mix.")
    parser.add_argument("--training-ready-manifest", type=Path, required=True)
    parser.add_argument("--replay-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.training_ready_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    replay_ratio = float(args.replay_ratio)
    validation_ratio = float(args.validation_ratio)
    seed = int(args.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("--replay-ratio must be in [0, 1)")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be in [0, 1)")
    replay_paths = [Path(path).expanduser().resolve() for path in args.replay_jsonl]
    missing_replay = [str(path) for path in replay_paths if not path.is_file()]
    if missing_replay:
        raise FileNotFoundError(f"Replay JSONL files not found: {missing_replay}")
    if replay_ratio > 0.0 and not replay_paths:
        raise ValueError("A positive --replay-ratio requires at least one --replay-jsonl")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest = read_json(manifest_path)
    artifacts = validate_ready_manifest(manifest, path=manifest_path)
    group_level = str(manifest["group_level"])
    frontier_source = resolve_artifact_path(manifest_path, artifacts["balanced_frontier_train"])
    heldout_source = resolve_artifact_path(manifest_path, artifacts["balanced_frontier_heldout"])
    frontier_source_sha = file_sha256(frontier_source)
    heldout_source_sha = file_sha256(heldout_source)
    frontier_raw = read_jsonl(frontier_source)
    official_heldout = read_jsonl(heldout_source)
    if not frontier_raw or not official_heldout:
        raise RuntimeError("Ready manifest must point to nonempty balanced train and heldout files")
    validate_sft_source_rows(frontier_raw, role="balanced_frontier_train")

    frontier_unique, frontier_duplicate_exclusions = dedupe_rows(
        frontier_raw,
        source_label="balanced_frontier_train",
        group_level=group_level,
    )
    official_heldout_index = build_identity_index(
        official_heldout, group_level=group_level
    )
    frontier_heldout_collisions = [
        {
            "sample_id": sample_identity(row),
            "identity_matches": overlap_details(
                row, official_heldout_index, group_level=group_level
            ),
        }
        for row in frontier_unique
        if overlap_details(row, official_heldout_index, group_level=group_level)
    ]
    if frontier_heldout_collisions:
        raise RuntimeError(
            "Manifest balanced frontier train leaks into official heldout: "
            f"{frontier_heldout_collisions[:5]}"
        )

    frontier_train_raw, validation_raw, validation_split = split_train_derived_validation(
        frontier_unique,
        ratio=validation_ratio,
        seed=seed,
        group_level=group_level,
    )
    heldout_validation_index = build_identity_index(
        list(official_heldout) + list(validation_raw), group_level=group_level
    )
    frontier_train_index = build_identity_index(
        frontier_train_raw, group_level=group_level
    )

    replay_candidates: list[tuple[dict[str, Any], Path, int, str]] = []
    replay_input_records: list[dict[str, Any]] = []
    for replay_path in replay_paths:
        replay_sha = file_sha256(replay_path)
        source_rows = list(iter_jsonl(replay_path))
        replay_input_records.append(
            artifact_record(replay_path, role="replay_candidate_source", row_count=len(source_rows))
        )
        for line_number, row in source_rows:
            replay_candidates.append((row, replay_path, line_number, replay_sha))
    replay_candidates.sort(
        key=lambda item: (
            content_fingerprint(item[0]),
            sample_identity(item[0]),
            str(item[1]),
            item[2],
        )
    )

    replay_seen = empty_identity_index()
    eligible_replay: list[tuple[dict[str, Any], Path, int, str]] = []
    replay_exclusions: list[dict[str, Any]] = []
    for row, source_path, line_number, source_sha in replay_candidates:
        invalid_checks = {
            "sample_id": bool(sample_identity(row)),
            "dialogue_turns": bool(dialogue_turns(row)),
            "qa": bool(qa(row)),
            "options": bool(normalized_options(row)),
        }
        if not all(invalid_checks.values()):
            replay_exclusions.append(
                {
                    "source_path": str(source_path),
                    "source_line": line_number,
                    "sample_id": sample_identity(row),
                    "reason": "invalid_sft_source_row",
                    "checks": invalid_checks,
                }
            )
            continue
        protected_overlap = overlap_details(
            row, heldout_validation_index, group_level=group_level
        )
        if protected_overlap:
            replay_exclusions.append(
                {
                    "source_path": str(source_path),
                    "source_line": line_number,
                    "sample_id": sample_identity(row),
                    "reason": "protected_partition_overlap",
                    "identity_matches": protected_overlap,
                }
            )
            continue
        frontier_duplicate = overlap_details(
            row,
            frontier_train_index,
            keys=TRAIN_CONTENT_KEYS,
            group_level=group_level,
        )
        if frontier_duplicate:
            replay_exclusions.append(
                {
                    "source_path": str(source_path),
                    "source_line": line_number,
                    "sample_id": sample_identity(row),
                    "reason": "frontier_train_duplicate",
                    "identity_matches": frontier_duplicate,
                }
            )
            continue
        duplicate = overlap_details(
            row,
            replay_seen,
            keys=DEDUPE_KEYS,
            group_level=group_level,
        )
        if duplicate:
            replay_exclusions.append(
                {
                    "source_path": str(source_path),
                    "source_line": line_number,
                    "sample_id": sample_identity(row),
                    "reason": "duplicate_replay",
                    "identity_matches": duplicate,
                }
            )
            continue
        eligible_replay.append((row, source_path, line_number, source_sha))
        add_to_identity_index(replay_seen, row, group_level=group_level)

    requested_replay_count = replay_target(len(frontier_train_raw), replay_ratio)
    eligible_replay.sort(
        key=lambda item: (
            deterministic_rank(
                seed,
                sample_identity(item[0]),
                question_identity(item[0]),
                content_fingerprint(item[0]),
                str(item[1]),
                item[2],
            ),
            sample_identity(item[0]),
        )
    )
    selected_replay_raw = eligible_replay[:requested_replay_count]

    frontier_train = [
        stamp_row(
            row,
            role="frontier_train",
            source_path=frontier_source,
            source_sha256=frontier_source_sha,
            group_level=group_level,
        )
        for row in frontier_train_raw
    ]
    validation = [
        stamp_row(
            row,
            role="train_derived_validation",
            source_path=frontier_source,
            source_sha256=frontier_source_sha,
            group_level=group_level,
        )
        for row in validation_raw
    ]
    replay_selected = [
        stamp_row(
            row,
            role="replay",
            source_path=source_path,
            source_sha256=source_sha,
            group_level=group_level,
        )
        for row, source_path, _, source_sha in selected_replay_raw
    ]
    validate_sft_source_rows(frontier_train + validation + replay_selected, role="selected")

    train_mix = list(frontier_train) + list(replay_selected)
    train_mix.sort(
        key=lambda row: (
            deterministic_rank(seed + 1, sample_identity(row), content_fingerprint(row)),
            sample_identity(row),
        )
    )
    train_role_by_sample = {sample_identity(row): str(row.get("train_mix_role")) for row in train_mix}
    valid_role_by_sample = {sample_identity(row): "train_derived_validation" for row in validation}
    sft_train = build_sft(
        train_mix, role_by_sample=train_role_by_sample, group_level=group_level
    )
    sft_validation = build_sft(
        validation, role_by_sample=valid_role_by_sample, group_level=group_level
    )

    train_vs_heldout = overlap_summary(
        train_mix, official_heldout, group_level=group_level
    )
    validation_vs_heldout = overlap_summary(
        validation, official_heldout, group_level=group_level
    )
    train_vs_validation = overlap_summary(
        train_mix, validation, group_level=group_level
    )
    if not train_vs_heldout["zero"] or not validation_vs_heldout["zero"] or not train_vs_validation["zero"]:
        raise RuntimeError(
            "Post-materialization leakage detected: "
            f"train-heldout={train_vs_heldout}, valid-heldout={validation_vs_heldout}, "
            f"train-valid={train_vs_validation}"
        )

    paths = {
        "frontier_train": output_dir / "frontier_train.jsonl",
        "replay_selected": output_dir / "replay_selected.jsonl",
        "train_mix": output_dir / "train_mix.jsonl",
        "sft_train": output_dir / "sft_train.jsonl",
        "validation_train_derived": output_dir / "validation_train_derived.jsonl",
        "validation_train_derived_sft": output_dir / "validation_train_derived_sft.jsonl",
        "frontier_dedup_excluded": output_dir / "frontier_dedup_excluded.jsonl",
        "replay_excluded": output_dir / "replay_excluded.jsonl",
        "validation_manifest": output_dir / "validation_manifest.json",
        "summary": output_dir / "train_mix_summary.json",
        "provenance": output_dir / "provenance.json",
        "provenance_sha256": output_dir / "provenance.sha256",
    }
    jsonl_outputs: list[tuple[str, list[dict[str, Any]]]] = [
        ("frontier_train", frontier_train),
        ("replay_selected", replay_selected),
        ("train_mix", train_mix),
        ("sft_train", sft_train),
        ("validation_train_derived", validation),
        ("validation_train_derived_sft", sft_validation),
        ("frontier_dedup_excluded", frontier_duplicate_exclusions),
        ("replay_excluded", replay_exclusions),
    ]
    for key, rows in jsonl_outputs:
        write_jsonl(paths[key], rows)

    data_artifacts = {
        key: artifact_record(paths[key], role=key, row_count=len(rows))
        for key, rows in jsonl_outputs
    }
    validation_manifest = {
        "schema_version": "stage3_train_derived_validation_v1",
        **validation_split,
        "group_level": group_level,
        "official_stage3_heldout_used": False,
        "source": data_artifacts["frontier_train"],
        "validation_artifact": data_artifacts["validation_train_derived"],
        "validation_sft_artifact": data_artifacts["validation_train_derived_sft"],
        "train_validation_overlap": train_vs_validation,
        "validation_official_heldout_overlap": validation_vs_heldout,
    }
    write_json(paths["validation_manifest"], validation_manifest)
    validation_manifest_record = artifact_record(
        paths["validation_manifest"], role="validation_manifest"
    )

    actual_replay_ratio = len(replay_selected) / max(1, len(train_mix))
    pass_criteria = {
        "manifest_training_ready": True,
        "frontier_train_nonempty": bool(frontier_train),
        "replay_target_met": len(replay_selected) == requested_replay_count,
        "official_heldout_not_used_as_training_or_validation": True,
        "train_official_heldout_overlap_zero": train_vs_heldout["zero"],
        "validation_official_heldout_overlap_zero": validation_vs_heldout["zero"],
        "train_validation_overlap_zero": train_vs_validation["zero"],
        "sft_train_count_matches": len(sft_train) == len(train_mix),
        "sft_validation_count_matches": len(sft_validation) == len(validation),
    }
    status = "audit_passed" if all(pass_criteria.values()) else "audit_failed"
    input_artifacts = {
        "training_ready_manifest": artifact_record(
            manifest_path, role="training_ready_manifest"
        ),
        "balanced_frontier_train": artifact_record(
            frontier_source,
            role="balanced_frontier_train",
            row_count=len(frontier_raw),
        ),
        "balanced_frontier_heldout_exclusion_only": artifact_record(
            heldout_source,
            role="balanced_frontier_heldout_exclusion_only",
            row_count=len(official_heldout),
        ),
        "replay_sources": replay_input_records,
    }
    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": status,
        "seed": seed,
        "group_level": group_level,
        "input_artifacts": input_artifacts,
        "official_stage3_heldout_policy": "exclusion_only_never_training_or_validation",
        "frontier_input_count": len(frontier_raw),
        "frontier_duplicate_excluded_count": len(frontier_duplicate_exclusions),
        "frontier_post_dedupe_count": len(frontier_unique),
        "frontier_train_count": len(frontier_train),
        "train_derived_validation_count": len(validation),
        "validation": validation_split,
        "replay_candidate_count": len(replay_candidates),
        "replay_eligible_count": len(eligible_replay),
        "replay_excluded_count": len(replay_exclusions),
        "replay_requested_ratio": replay_ratio,
        "replay_requested_count": requested_replay_count,
        "replay_selected_count": len(replay_selected),
        "replay_actual_ratio": round(actual_replay_ratio, 8),
        "train_mix_count": len(train_mix),
        "sft_train_count": len(sft_train),
        "leakage_audit": {
            "train_vs_official_heldout": train_vs_heldout,
            "validation_vs_official_heldout": validation_vs_heldout,
            "train_vs_train_derived_validation": train_vs_validation,
            "identity_keys": list(IDENTITY_KEYS),
        },
        "pass_criteria": pass_criteria,
        "artifact_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "output_artifacts": {
            **data_artifacts,
            "validation_manifest": validation_manifest_record,
        },
    }
    write_json(paths["summary"], summary)
    summary_record = artifact_record(paths["summary"], role="train_mix_summary")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "status": status,
        "inputs": input_artifacts,
        "outputs": {
            **data_artifacts,
            "validation_manifest": validation_manifest_record,
            "train_mix_summary": summary_record,
        },
        "self": {
            "path": str(paths["provenance"].resolve()),
            "sha256_recorded_in": str(paths["provenance_sha256"].resolve()),
        },
    }
    write_json(paths["provenance"], provenance)
    provenance_sha = file_sha256(paths["provenance"])
    paths["provenance_sha256"].write_text(
        f"{provenance_sha}  {paths['provenance'].name}\n", encoding="ascii"
    )
    summary["provenance_artifact"] = {
        "path": str(paths["provenance"].resolve()),
        "sha256": provenance_sha,
        "sha256_sidecar": str(paths["provenance_sha256"].resolve()),
    }
    # Return the fully useful in-memory summary.  The on-disk summary is already
    # covered by provenance and is not rewritten, avoiding a self-hash cycle.
    return summary


def main() -> int:
    args = parse_args()
    summary = build(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "frontier_train_count": summary["frontier_train_count"],
                "train_derived_validation_count": summary["train_derived_validation_count"],
                "replay_selected_count": summary["replay_selected_count"],
                "train_mix_count": summary["train_mix_count"],
                "replay_actual_ratio": summary["replay_actual_ratio"],
                "leakage_audit": summary["leakage_audit"],
                "artifact_paths": summary["artifact_paths"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "audit_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
