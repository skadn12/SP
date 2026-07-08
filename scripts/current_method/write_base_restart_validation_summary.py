#!/usr/bin/env python3
"""Write reports for base_restart_current_method_validation.

This script is report-only. It summarizes artifacts already written under the
validation root and does not start mining or training.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_ROOT = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
RAW_PATH = Path("/root/autodl-tmp/gxy-selfplay/locomo10.json")
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
MAX_LEN = 16384


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def qtype(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    return str(qa.get("question_type") or row.get("question_type") or "unknown")


def difficulty_subtype(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    return str(row.get("difficulty_subtype") or qa.get("difficulty_subtype") or "ordinary")


def bucket(row: dict[str, Any]) -> str:
    se = row.get("solver_eval") if isinstance(row.get("solver_eval"), dict) else {}
    return str(se.get("bucket") or row.get("bucket") or "unknown")


def qtype_bucket_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[qtype(row)][bucket(row)] += 1
    return {key: dict(counter) for key, counter in sorted(grouped.items())}


def zero_share(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if bucket(row) == "0/5") / len(rows), 6)


def rate(n: int, d: int) -> float:
    return round(n / d, 6) if d else 0.0


def smoke_summary() -> dict[str, Any]:
    smoke = VALID_ROOT / "00_smoke50_check"
    samples = read_jsonl(smoke / "samples.jsonl")
    gen = read_json(smoke / "generation_summary.json", {}) or {}
    report = read_json(smoke / "reports" / "report.json", {}) or {}
    solver = read_json(smoke / "reports" / "solver_eval_summary.json", {}) or {}
    frontier = read_json(smoke / "reports" / "frontier_summary.json", {}) or {}

    generated = int(report.get("generated_count") or gen.get("generated_count") or len(samples))
    valid = int(report.get("valid_count") or gen.get("valid_count") or 0)
    clean = int(report.get("clean_count") or gen.get("clean_count") or 0)
    hard_rows = [row for row in samples if difficulty_subtype(row) == "hard_single_choice"]
    multi_rows = [row for row in samples if qtype(row) == "multi_select"]
    ordering_rows = [row for row in samples if qtype(row) == "ordering"]
    k1_diag = ((solver.get("k1") or {}).get("diagnostics") or {}) if isinstance(solver, dict) else {}
    k5_diag = ((solver.get("k5") or {}).get("diagnostics") or {}) if isinstance(solver, dict) else {}
    overflow_count = int(solver.get("eval_context_overflow_count") or 0)
    truncated_count = int(k1_diag.get("truncated_count") or 0) + int(k5_diag.get("truncated_count") or 0)

    summary = {
        "stage": "00_smoke50_check",
        "status": "failed",
        "passed": False,
        "failure_categories": [
            "difficulty too hard",
            "parser/eval problem",
            "length/truncation problem",
        ],
        "blockers": [
            "overall_zero_share=1.0 > 0.65",
            "frontier_plus_near_miss_count=0 < 8",
            "eval_context_overflow_count=100 and truncated_count=100 with max_model_len=16384",
            "full-dialogue prompts exceeded 16384 tokens after raw-dialogue + 4000-token expansion",
        ],
        "generated_count": generated,
        "valid_count": valid,
        "clean_count": clean,
        "valid_rate": rate(valid, generated),
        "clean_rate": rate(clean, generated),
        "rows_below_4000_delta": int(report.get("rows_below_4000_delta") or gen.get("rows_below_4000_delta") or 0),
        "leakage_counts": {
            "anchor_leakage_count": int(report.get("anchor_leakage_count") or gen.get("anchor_leakage_count") or 0),
            "solver_visible_meta_leak_count": int(report.get("solver_visible_meta_leak_count") or gen.get("solver_visible_meta_leak_count") or 0),
            "qa_meta_leak_count": int(report.get("qa_meta_leak_count") or gen.get("qa_meta_leak_count") or 0),
            "dialogue_meta_leak_count": int(report.get("dialogue_meta_leak_count") or gen.get("dialogue_meta_leak_count") or 0),
        },
        "anchor_leakage_count": int(report.get("anchor_leakage_count") or gen.get("anchor_leakage_count") or 0),
        "solver_visible_meta_leak_count": int(report.get("solver_visible_meta_leak_count") or gen.get("solver_visible_meta_leak_count") or 0),
        "frontier_count": int(report.get("frontier_count") or frontier.get("frontier_count") or 0),
        "near_miss_count": int(report.get("near_miss_count") or frontier.get("near_miss_count") or 0),
        "frontier_plus_near_miss_count": int(report.get("frontier_count") or 0) + int(report.get("near_miss_count") or 0),
        "qtype_distribution": dict(Counter(qtype(row) for row in samples).most_common()),
        "qtype_bucket_distribution": qtype_bucket_distribution(samples),
        "overall_zero_share": zero_share(samples),
        "hard_single_choice_zero_share": zero_share(hard_rows),
        "multi_select_zero_share": zero_share(multi_rows),
        "ordering_zero_share": zero_share(ordering_rows),
        "bad_question_count": int(report.get("bad_question_count") or frontier.get("bad_question_count") or 0),
        "overflow_count": overflow_count,
        "truncated_count": truncated_count,
        "unique_overflow_sample_count": int(k1_diag.get("eval_context_overflow_count") or 0),
        "unique_truncated_sample_count": int(k1_diag.get("truncated_count") or 0),
        "solver_exact_accuracy_k1": float(report.get("solver_exact_accuracy_k1") or 0.0),
        "solver_soft_accuracy_k5": float(report.get("solver_soft_accuracy_k5") or 0.0),
        "soft_buckets": report.get("soft_buckets") or {},
        "max_model_len": MAX_LEN,
        "tokenizer_path": str(TOKENIZER),
        "base_solver": str(BASE_MODEL),
        "prompt_token_estimate_mean": k1_diag.get("prompt_token_estimate_mean"),
        "prompt_token_estimate_max": k1_diag.get("prompt_token_estimate_max"),
        "parent_dialogue_tokens_mean": report.get("parent_dialogue_tokens_mean"),
        "final_dialogue_tokens_mean": report.get("final_dialogue_tokens_mean"),
        "length_delta_tokens_mean": report.get("length_delta_tokens_mean"),
        "runner_recommendation": report.get("recommendation"),
    }
    return summary


def smoke_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 00_smoke50_check Report",
        "",
        "- status: failed",
        "- generated_count: {generated_count}".format(**summary),
        "- valid_count: {valid_count}".format(**summary),
        "- clean_count: {clean_count}".format(**summary),
        "- valid_rate: {valid_rate}".format(**summary),
        "- clean_rate: {clean_rate}".format(**summary),
        "- rows_below_4000_delta: {rows_below_4000_delta}".format(**summary),
        "- leakage_counts: {leakage_counts}".format(**summary),
        "- anchor_leakage_count: {anchor_leakage_count}".format(**summary),
        "- solver_visible_meta_leak_count: {solver_visible_meta_leak_count}".format(**summary),
        "- frontier_count: {frontier_count}".format(**summary),
        "- near_miss_count: {near_miss_count}".format(**summary),
        "- frontier_plus_near_miss_count: {frontier_plus_near_miss_count}".format(**summary),
        "- qtype_distribution: {qtype_distribution}".format(**summary),
        "- qtype_bucket_distribution: {qtype_bucket_distribution}".format(**summary),
        "- overall_zero_share: {overall_zero_share}".format(**summary),
        "- hard_single_choice_zero_share: {hard_single_choice_zero_share}".format(**summary),
        "- multi_select_zero_share: {multi_select_zero_share}".format(**summary),
        "- ordering_zero_share: {ordering_zero_share}".format(**summary),
        "- bad_question_count: {bad_question_count}".format(**summary),
        "- overflow_count: {overflow_count}".format(**summary),
        "- truncated_count: {truncated_count}".format(**summary),
        "- max_model_len: {max_model_len}".format(**summary),
        "- tokenizer_path: {tokenizer_path}".format(**summary),
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["blockers"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "00_smoke50_check did not pass. Per experiment rules, 01_mining500_train was not started.",
        ]
    )
    return "\n".join(lines)


def total_markdown(summary: dict[str, Any]) -> str:
    preprocess = read_json(VALID_ROOT / "preprocess_locomo_current_method" / "preprocess_summary.json", {}) or {}
    raw_schema = read_json(VALID_ROOT / "raw_audit" / "raw_schema_summary.json", {}) or {}
    seed_path = VALID_ROOT / "seed_parents" / "seed_parents.jsonl"
    lines = [
        "# base_restart_current_method_validation Summary",
        "",
        "## 1. Experiment Goal",
        "",
        "Validate whether the existing current_method can restart from base Qwen3-4B, rebuild seed parents from raw LoCoMo, run generation/mining/train/eval, and decide whether a first from-base solver round is viable.",
        "",
        "## 2. Scope",
        "",
        "- This is from-base current_method validation.",
        "- This is not r5.",
        "- This is not r1 legacy.",
        "- This is not r4 continuation.",
        "- No r4 parent, r4 mining500 data, r4 train_mix, probe197 solver, r1/r2 solver, or r1 clean_v1 processed graph was used as experiment input.",
        "",
        "## 3. LoCoMo Raw Path",
        "",
        f"- raw_path: {RAW_PATH}",
        f"- sample_count: {raw_schema.get('sample_count')}",
        f"- sample0_5_usable: {raw_schema.get('sample0_5_usable')}",
        "",
        "## 4. Seed Adapter Result",
        "",
        f"- adapter_status: {preprocess.get('adapter_status')}",
        f"- seed_parent_path: {seed_path}",
        f"- parent_count: {preprocess.get('parent_count')}",
        f"- missing_required_fields: {preprocess.get('missing_required_fields')}",
        f"- state_path: {VALID_ROOT / 'states' / 'bootstrap_controller_state.json'}",
        "",
        "## 5. 00_smoke50_check Result",
        "",
        f"- status: {summary['status']}",
        f"- generated/valid/clean: {summary['generated_count']}/{summary['valid_count']}/{summary['clean_count']}",
        f"- valid_rate: {summary['valid_rate']}",
        f"- clean_rate: {summary['clean_rate']}",
        f"- qtype_distribution: {summary['qtype_distribution']}",
        f"- rows_below_4000_delta: {summary['rows_below_4000_delta']}",
        f"- anchor_leakage_count: {summary['anchor_leakage_count']}",
        f"- solver_visible_meta_leak_count: {summary['solver_visible_meta_leak_count']}",
        f"- overall_zero_share: {summary['overall_zero_share']}",
        f"- frontier_plus_near_miss_count: {summary['frontier_plus_near_miss_count']}",
        f"- overflow_count: {summary['overflow_count']}",
        f"- truncated_count: {summary['truncated_count']}",
        "",
        "## 6. 01_mining500_train Mining Result",
        "",
        "- not_run: 00_smoke50_check failed.",
        "",
        "## 7. Train Mix Result",
        "",
        "- not_run: mining was not started.",
        "",
        "## 8. Training Result",
        "",
        "- not_run: train mix was not built.",
        "",
        "## 9. Post-Train Eval Result",
        "",
        "- not_run: no model was trained.",
        "",
        "## 10. First-Round From-Base Validation",
        "",
        "- passed: false",
        "- reason: 00_smoke50_check failed because base eval overflowed/truncated all 50 prompts and all k5 buckets were 0/5.",
        "",
        "## 11. Next Round Recommendation",
        "",
        "- recommend_next_round: false",
        "- recommendation: fix smoke-stage length/eval feasibility first; do not start mining/training from these outputs.",
        "",
        "## 12. Blockers / Risks",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["blockers"])
    lines.extend(
        [
            "- The raw full dialogues plus required 4000-token delta produced prompts with mean token estimate above 26k, incompatible with max_model_len=16384.",
            "- Base solver zero-share cannot be interpreted as true difficulty while all eval prompts are overflow/truncated.",
            "",
            "## 13. Key Paths",
            "",
            f"- raw audit paths: {VALID_ROOT / 'raw_audit' / 'locomo_raw_paths.txt'}",
            f"- raw schema summary: {VALID_ROOT / 'raw_audit' / 'raw_schema_summary.json'}",
            f"- preprocess summary: {VALID_ROOT / 'preprocess_locomo_current_method' / 'preprocess_summary.json'}",
            f"- seed parents: {seed_path}",
            f"- controller state: {VALID_ROOT / 'states' / 'bootstrap_controller_state.json'}",
            f"- smoke samples: {VALID_ROOT / '00_smoke50_check' / 'samples.jsonl'}",
            f"- smoke valid samples: {VALID_ROOT / '00_smoke50_check' / 'valid_samples.jsonl'}",
            f"- smoke clean samples: {VALID_ROOT / '00_smoke50_check' / 'clean_samples.jsonl'}",
            f"- smoke summary: {VALID_ROOT / '00_smoke50_check' / 'adaptive_smoke_summary.json'}",
            f"- smoke report: {VALID_ROOT / '00_smoke50_check' / 'reports' / 'report.md'}",
            f"- total report: {VALID_ROOT / 'reports' / 'base_restart_current_method_validation_summary.md'}",
            f"- rate-limited archived attempt: {VALID_ROOT / '00_smoke50_check_attempts'}",
            "",
            "Stopped after 00_smoke50_check, as required.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    summary = smoke_summary()
    write_json(VALID_ROOT / "00_smoke50_check" / "adaptive_smoke_summary.json", summary)
    write_text(VALID_ROOT / "00_smoke50_check" / "reports" / "report.md", smoke_markdown(summary))
    write_text(VALID_ROOT / "reports" / "base_restart_current_method_validation_summary.md", total_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
