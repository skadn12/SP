#!/usr/bin/env python3
"""Prepare a blind, deterministic QuALITY exemplar curation packet.

This module is intentionally offline.  It reads an existing local QuALITY
validation export and, optionally, historical per-item solver reports.  It
does not call a model, run a solver, extract evidence, or make semantic
judgements about the selected questions.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/eval_sets/external_long_context_mcq/"
    "all_mcq_32k_eligible.jsonl"
)
DEFAULT_SUMMARY = Path(
    "/root/autodl-tmp/gxy-selfplay/eval_sets/external_long_context_mcq/"
    "summary_32k.json"
)
DEFAULT_HISTORY_DIR = Path(
    "/root/autodl-tmp/gxy-selfplay/eval_reports/external_long_context_mcq"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/eval_sets/external_long_context_mcq/"
    "quality_exemplar_curation_v1"
)
DEFAULT_SEED = 20260718
EXPECTED_QUALITY_COUNT = 2086
EXPECTED_CONTEXT_COUNT = 115
LETTERS = ("A", "B", "C", "D")
BUCKET_ORDER = ("short", "medium", "long")
BUCKET_RULES = {
    "short": "prompt_tokens < 5000",
    "medium": "5000 <= prompt_tokens < 7500",
    "long": "prompt_tokens >= 7500",
}


class CurationPreparationError(RuntimeError):
    """Raised when an authoritative precondition or selection rule fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_sha256(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def prompt_length_bucket(prompt_tokens: int) -> str:
    if prompt_tokens < 5000:
        return "short"
    if prompt_tokens < 7500:
        return "medium"
    return "long"


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_options(options: Any) -> list[str] | None:
    if not isinstance(options, dict) or set(options) != set(LETTERS):
        return None
    values = [options.get(letter) for letter in LETTERS]
    if not all(_nonempty_string(value) for value in values):
        return None
    return [str(value) for value in values]


def valid_candidate_row(row: dict[str, Any]) -> bool:
    if not _nonempty_string(row.get("id")):
        return False
    if not _nonempty_string(row.get("context")):
        return False
    if not _nonempty_string(row.get("question")):
        return False
    if row.get("answer") not in LETTERS:
        return False
    if _canonical_options(row.get("options")) is None:
        return False
    tokens = row.get("prompt_tokens")
    return isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0


def load_quality_rows(
    input_path: Path,
    *,
    expected_quality_count: int = EXPECTED_QUALITY_COUNT,
    expected_context_count: int = EXPECTED_CONTEXT_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality_rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CurationPreparationError(
                    f"invalid JSON at {input_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise CurationPreparationError(
                    f"non-object JSON row at {input_path}:{line_number}"
                )
            if row.get("dataset") == "quality":
                quality_rows.append(row)

    context_hashes = {
        context_sha256(str(row.get("context") or "")) for row in quality_rows
    }
    ids = [str(row.get("id") or "") for row in quality_rows]
    stats = {
        "split": "validation",
        "quality_qa_count": len(quality_rows),
        "unique_context_count": len(context_hashes),
        "all_ids_validation": all(
            record_id.startswith("quality:validation:") for record_id in ids
        ),
        "duplicate_id_count": len(ids) - len(set(ids)),
    }
    if len(quality_rows) != expected_quality_count:
        raise CurationPreparationError(
            f"QuALITY QA count mismatch: expected {expected_quality_count}, "
            f"found {len(quality_rows)}"
        )
    if len(context_hashes) != expected_context_count:
        raise CurationPreparationError(
            f"QuALITY context count mismatch: expected {expected_context_count}, "
            f"found {len(context_hashes)}"
        )
    if not stats["all_ids_validation"]:
        raise CurationPreparationError(
            "QuALITY input contains an id outside the validation split"
        )
    if stats["duplicate_id_count"]:
        raise CurationPreparationError(
            f"QuALITY input contains {stats['duplicate_id_count']} duplicate ids"
        )
    return quality_rows, stats


def validate_summary(summary_path: Path, expected_quality_count: int) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        summary_count = int(
            payload["scored_has_gold_answer"]["under_32k_by_dataset"]["quality"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CurationPreparationError(
            f"summary lacks the QuALITY under-32k count: {summary_path}"
        ) from exc
    if summary_count != expected_quality_count:
        raise CurationPreparationError(
            f"summary QuALITY count mismatch: expected {expected_quality_count}, "
            f"found {summary_count}"
        )
    return payload


def select_candidates(
    quality_rows: Iterable[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    per_bucket: int = 8,
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        if not valid_candidate_row(row):
            continue
        bucket = prompt_length_bucket(int(row["prompt_tokens"]))
        copied = dict(row)
        copied["_bucket"] = bucket
        copied["_context_sha256"] = context_sha256(row["context"])
        pools[bucket].append(copied)

    rng = random.Random(seed)
    for bucket in BUCKET_ORDER:
        # Canonical sorting makes the seeded shuffle independent of input order.
        pools[bucket].sort(key=lambda row: str(row["id"]))
        rng.shuffle(pools[bucket])

    selected: list[dict[str, Any]] = []
    used_contexts: set[str] = set()
    for bucket in BUCKET_ORDER:
        bucket_selected: list[dict[str, Any]] = []
        for row in pools[bucket]:
            article_hash = row["_context_sha256"]
            if article_hash in used_contexts:
                continue
            bucket_selected.append(row)
            used_contexts.add(article_hash)
            if len(bucket_selected) == per_bucket:
                break
        if len(bucket_selected) != per_bucket:
            raise CurationPreparationError(
                f"bucket {bucket!r} has only {len(bucket_selected)} selectable "
                f"distinct articles; required {per_bucket}"
            )
        selected.extend(bucket_selected)

    if len(used_contexts) != per_bucket * len(BUCKET_ORDER):
        raise CurationPreparationError("selected candidates do not have unique contexts")
    return selected


def load_historical_solver_results(
    history_dir: Path,
    quality_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read exact-id historical results while omitting gold and prediction labels.

    Including both a prediction label and its correctness would reveal the gold
    answer whenever the prediction was correct.  The blind packet therefore
    retains the per-report correctness result but excludes prediction/gold text.
    """

    aligned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report_audit: list[dict[str, Any]] = []
    if not history_dir.is_dir():
        return aligned, {
            "history_directory": str(history_dir.resolve()),
            "per_item_results_found": False,
            "reports": [],
        }

    for report_path in sorted(history_dir.glob("*.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            continue

        found_ids: set[str] = set()
        matched_rows = 0
        for result in results:
            if not isinstance(result, dict):
                continue
            record_id = result.get("source_record_id") or result.get("record_id")
            if record_id not in quality_ids:
                continue
            found_ids.add(str(record_id))
            matched_rows += 1
            correct = result.get("correct")
            aligned[str(record_id)].append(
                {
                    "report_file": str(report_path.resolve()),
                    "report_sha256": sha256_file(report_path),
                    "model": payload.get("model"),
                    "correct": correct if isinstance(correct, bool) else None,
                }
            )
        if matched_rows:
            report_audit.append(
                {
                    "report_file": str(report_path.resolve()),
                    "report_sha256": sha256_file(report_path),
                    "matched_quality_rows": matched_rows,
                    "unique_matched_quality_ids": len(found_ids),
                    "exact_full_quality_id_alignment": found_ids == quality_ids,
                }
            )

    return aligned, {
        "history_directory": str(history_dir.resolve()),
        "per_item_results_found": bool(aligned),
        "reports": report_audit,
        "blind_metadata_policy": (
            "Per-item correctness is retained; gold labels, predicted labels, "
            "gold text, and prediction text are omitted from candidate_items."
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def _annotation_record(candidate_index: int, record_id: str) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "id": record_id,
        "keep_as_exemplar": None,
        "question_is_natural": None,
        "single_local_passage_sufficient": None,
        "multiple_nonredundant_evidence_required": None,
        "evidence_a_quote": "",
        "evidence_b_quote": "",
        "evidence_a_role": "",
        "evidence_b_role": "",
        "combined_evidence_explanation": "",
        "reasoning_type": "",
        "distractors_are_plausible": None,
        "compression_notes": "",
        "reviewer_notes": "",
    }


def render_candidate_markdown(candidate_items: list[dict[str, Any]]) -> str:
    parts = [
        "# QuALITY exemplar 人工审计候选",
        "",
        "本文件故意不显示正确答案。所有语义审计字段均为空白，等待人工填写。",
        "",
    ]
    for item in candidate_items:
        parts.extend(
            [
                f"## 候选 {item['candidate_index']}",
                "",
                f"- 原始 id：`{item['id']}`",
                f"- 长度分组：`{item['prompt_length_bucket']}`",
                f"- prompt_tokens：`{item['prompt_tokens']}`",
                "",
                "### 完整 context",
                "",
                f"<pre>{html.escape(item['context'])}</pre>",
                "",
                "### Question",
                "",
                item["question"],
                "",
                "### Options",
                "",
            ]
        )
        for letter, option in zip(LETTERS, item["options"]):
            parts.append(f"- {letter}. {option}")
        parts.extend(
            [
                "",
                "### 空白人工审计表",
                "",
                "| 字段 | 人工填写 |",
                "|---|---|",
                "| keep_as_exemplar | |",
                "| question_is_natural | |",
                "| single_local_passage_sufficient | |",
                "| multiple_nonredundant_evidence_required | |",
                "| evidence_a_quote / role | |",
                "| evidence_b_quote / role | |",
                "| combined_evidence_explanation | |",
                "| reasoning_type | |",
                "| distractors_are_plausible | |",
                "| compression_notes | |",
                "| reviewer_notes | |",
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def render_readme() -> str:
    return """# QuALITY exemplar curation v1

这些候选来自本地 QuALITY validation 数据。当前阶段只生成可复现的人工审计包，尚未确认任何候选满足 joint necessity，也尚未将任何候选用于 Questioner Prompt。

## 文件

- `candidate_items.jsonl`：盲审候选，包含完整原文和选项，不含正确答案。
- `candidate_items.md`：便于人工阅读的完整候选与空白审计表。
- `answer_key.json`：与盲审材料分离的答案键。
- `annotation_template.jsonl`：所有语义字段均为空白的人工标注模板。
- `selection_manifest.json`：输入哈希、固定 seed、筛选规则和零 API/零 Solver 声明。

`historical_solver_results` 仅来自已有报告。本次没有运行 Solver。为避免盲审文件泄漏答案，历史元数据只保留逐轮 `correct` 状态，不复制 gold、预测字母或预测文本。历史正确/错误也不代表题目是复杂题。

筛选过程不抽取证据、不判断 multi-hop、不判断 joint necessity，也不自动选择最终教学示例。
"""


def prepare_curation(
    *,
    input_path: Path = DEFAULT_INPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    history_dir: Path = DEFAULT_HISTORY_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
    expected_quality_count: int = EXPECTED_QUALITY_COUNT,
    expected_context_count: int = EXPECTED_CONTEXT_COUNT,
    per_bucket: int = 8,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    summary_path = summary_path.resolve()
    history_dir = history_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    source_sha_before = sha256_file(input_path)
    summary_sha = sha256_file(summary_path)
    validate_summary(summary_path, expected_quality_count)
    quality_rows, source_stats = load_quality_rows(
        input_path,
        expected_quality_count=expected_quality_count,
        expected_context_count=expected_context_count,
    )
    selected = select_candidates(quality_rows, seed=seed, per_bucket=per_bucket)
    quality_ids = {str(row["id"]) for row in quality_rows}
    historical, history_audit = load_historical_solver_results(history_dir, quality_ids)

    candidate_items: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    selected_manifest_rows: list[dict[str, Any]] = []
    for candidate_index, row in enumerate(selected, 1):
        options = _canonical_options(row["options"])
        assert options is not None
        answer = str(row["answer"])
        history = historical.get(str(row["id"])) or None
        candidate_items.append(
            {
                "candidate_index": candidate_index,
                "id": row["id"],
                "dataset": "quality",
                "prompt_length_bucket": row["_bucket"],
                "prompt_tokens": row["prompt_tokens"],
                "context": row["context"],
                "question": row["question"],
                "options": options,
                "task_type": row.get("task_type"),
                "historical_solver_results": history,
            }
        )
        answer_key.append(
            {
                "candidate_index": candidate_index,
                "id": row["id"],
                "correct_answer": answer,
                "correct_option_text": options[LETTERS.index(answer)],
            }
        )
        annotations.append(_annotation_record(candidate_index, str(row["id"])))
        selected_manifest_rows.append(
            {
                "candidate_index": candidate_index,
                "id": row["id"],
                "prompt_length_bucket": row["_bucket"],
                "context_sha256": row["_context_sha256"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_jsonl = output_dir / "candidate_items.jsonl"
    candidate_md = output_dir / "candidate_items.md"
    answer_key_path = output_dir / "answer_key.json"
    annotation_path = output_dir / "annotation_template.jsonl"
    readme_path = output_dir / "README.md"
    manifest_path = output_dir / "selection_manifest.json"

    _write_jsonl(candidate_jsonl, candidate_items)
    with candidate_md.open("x", encoding="utf-8") as handle:
        handle.write(render_candidate_markdown(candidate_items))
    _write_json(answer_key_path, answer_key)
    _write_jsonl(annotation_path, annotations)
    with readme_path.open("x", encoding="utf-8") as handle:
        handle.write(render_readme())

    source_sha_after = sha256_file(input_path)
    if source_sha_after != source_sha_before:
        raise CurationPreparationError("source JSONL changed during curation preparation")

    bucket_counts = Counter(item["prompt_length_bucket"] for item in candidate_items)
    artifact_paths = (
        candidate_jsonl,
        candidate_md,
        answer_key_path,
        annotation_path,
        readme_path,
    )
    manifest = {
        "source_file": str(input_path),
        "source_file_sha256": source_sha_before,
        "source_summary_file": str(summary_path),
        "source_summary_sha256": summary_sha,
        "selection_time_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_seed": seed,
        "selection_algorithm": (
            "sort each valid bucket by id, seeded Python Random shuffle, then "
            "greedily select previously unused context SHA256 values"
        ),
        "length_bucket_rules": BUCKET_RULES,
        "source_statistics": source_stats,
        "candidate_count_by_bucket": {
            bucket: bucket_counts.get(bucket, 0) for bucket in BUCKET_ORDER
        },
        "total_candidate_count": len(candidate_items),
        "distinct_article_count": len(
            {row["context_sha256"] for row in selected_manifest_rows}
        ),
        "selected_candidates": selected_manifest_rows,
        "historical_solver_alignment": history_audit,
        "external_api_used": False,
        "solver_run": False,
        "original_data_modified": False,
        "automatic_multihop_judgement": False,
        "automatic_joint_necessity_judgement": False,
        "artifact_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a blind, deterministic QuALITY exemplar curation packet."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-quality-count", type=int, default=EXPECTED_QUALITY_COUNT
    )
    parser.add_argument(
        "--expected-context-count", type=int, default=EXPECTED_CONTEXT_COUNT
    )
    parser.add_argument("--per-bucket", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_curation(
        input_path=args.input,
        summary_path=args.summary,
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        expected_quality_count=args.expected_quality_count,
        expected_context_count=args.expected_context_count,
        per_bucket=args.per_bucket,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
