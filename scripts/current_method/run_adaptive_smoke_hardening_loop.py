#!/usr/bin/env python3
"""Adaptive smoke hardening loop for current_method r4 probe197.

This is a thin orchestration layer around `run_auto_graph_expansion_round.py`.
It keeps the graph/state curriculum as source of truth, uses smoke to calibrate
the challenger difficulty plan, and only launches solver training after the
requested gate conditions are met.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from math import floor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def sanitize_thread_env(env: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    for key in THREAD_ENV_VARS:
        raw = str(target.get(key, "")).strip()
        try:
            valid = int(raw) > 0
        except ValueError:
            valid = False
        if not valid:
            target[key] = "1"
    return target


sanitize_thread_env()

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_v13_pilot as v13

RUNNER = SCRIPTS / "run_auto_graph_expansion_round.py"
TRAIN_PIPELINE = SCRIPTS / "run_round_solver_training_pipeline.py"
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_TOK = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/outputs/"
    "qwen3_4b_lora_current_method_r3_restart_phase19_probe197_from_r2_ctx9216_merged"
)
DEFAULT_DIFFICULTY_STATE = ROOT / "current_method_runs/difficulty_controller_state_r4_probe197.json"
DEFAULT_PARENT_DIR = ROOT / "current_method_runs/current_method_r3_phase19_cell_reward_n500_frontier_mining/generation"
DEFAULT_RUN_ROOT_BASE = ROOT / "current_method_runs/current_method_r4_test_probe197_adaptive_smoke_hardening"
DEFAULT_BOOTSTRAP_RUN = ROOT / "current_method_runs/current_method_r4_test_probe197_hardening_smoke20_baseline7753"

QTYPE_ORDER = ("single_choice", "multi_select", "ordering", "state_reconstruction")
SKILL_ORDER = ("conditional_plan", "permission_visibility_scope", "delayed_activation", "timeline_fork_merge")
BALANCED_MINING_STAGE = "frontier_mining_balanced_1"
MINING_BONUS_STAGE = "frontier_mining_bonus_1"
MANUAL_POLICY_CORRECTION = "manual_policy_correction_to_balanced_mining"
BALANCED_QTYPE_SHARES = {"single_choice": 0.40, "multi_select": 0.35, "ordering": 0.25}
BALANCED_SUBTYPE_SHARES = {"ordinary_single_choice": 0.125, "hard_single_choice": 0.875}
BALANCED_FRONTIER_SHARES = {
    "ordinary_single_choice": 0.05,
    "hard_single_choice": 0.35,
    "multi_select": 0.35,
    "ordering": 0.25,
}
MINING_BONUS_CAPS = {
    "ordinary_single_choice": 0.05,
    "hard_single_choice": 0.50,
    "multi_select": 0.45,
    "ordering": 0.35,
}
ORDERING_BONUS_FREEZE_ZERO_THRESHOLD = 0.60
RECENT_ZERO_PENALTY_THRESHOLD = 0.60
HARD_SINGLE_CHOICE_TEMPLATES = (
    "minimal_difference",
    "latest_valid_state",
    "guaranteed_vs_possible",
    "override_sensitive",
    "sequence_as_single_choice",
)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def shell(cmd: list[str], *, cwd: Path | None = None) -> None:
    env = os.environ.copy()
    sanitize_thread_env(env)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True, env=env)


def path_cli_arg(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def maybe_token(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def nested_solver_eval(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("solver_eval"), dict):
        return row["solver_eval"]
    return {}


def flat_solver_eval_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    se = nested_solver_eval(row)
    if key in se:
        return se.get(key, default)
    return row.get(key, default)


def parse_bucket(bucket: Any) -> int | None:
    text = str(bucket or "")
    if "/" not in text:
        return None
    left = text.split("/", 1)[0]
    try:
        return int(left)
    except ValueError:
        return None


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        se = nested_solver_eval(row)
        bucket = str(se.get("bucket") or row.get("bucket") or "unknown")
        partial = se.get("partial_credit", row.get("partial_credit", 0.0))
        exact_count = se.get("exact_correct_count", row.get("exact_correct_count", 0))
        if isinstance(partial, str):
            try:
                partial = float(partial)
            except ValueError:
                partial = 0.0
        if isinstance(exact_count, str):
            try:
                exact_count = int(exact_count)
            except ValueError:
                exact_count = 0
        row = copy.deepcopy(row)
        row["solver_eval"] = {
            **se,
            "bucket": bucket,
            "partial_credit": float(partial or 0.0),
            "exact_correct_count": int(exact_count or 0),
        }
        normalized.append(row)
    return normalized


def solver_eval_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if sid:
            out[sid] = nested_solver_eval(row)
    return out


def qtype_of(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    return str(qa.get("question_type") or row.get("question_type") or "unknown")


def difficulty_subtype_of(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    if qtype_of(row) == "single_choice":
        return str(row.get("difficulty_subtype") or qa.get("difficulty_subtype") or "ordinary_single_choice")
    return str(row.get("difficulty_subtype") or qa.get("difficulty_subtype") or "ordinary")


def hard_single_choice_template_of(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    return str(row.get("hard_single_choice_template") or qa.get("hard_single_choice_template") or "unknown")


def skill_of(row: dict[str, Any]) -> str:
    return str(row.get("attack_skill") or (row.get("attack_plan") or {}).get("attack_skill") or "unknown")


def bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        bucket = str(flat_solver_eval_value(row, "bucket", "unknown"))
        counts[bucket] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def qtype_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(qtype_of(row) for row in rows).most_common())


def skill_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(skill_of(row) for row in rows).most_common())


def qtype_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[qtype_of(row)].append(row)
    for qtype in sorted(grouped):
        out[qtype] = bucket_counts(grouped[qtype])
    return out


def qtype_skill_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[qtype_of(row)].append(row)
    for qtype in sorted(grouped):
        skill_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in grouped[qtype]:
            skill_grouped[skill_of(row)].append(row)
        out[qtype] = {skill: bucket_counts(skill_rows) for skill, skill_rows in sorted(skill_grouped.items())}
    return out


def prediction_count_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        preds = nested_solver_eval(row).get("predictions")
        counts[str(len(preds) if isinstance(preds, list) else 0)] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def prediction_count_distribution_by_qtype(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[qtype_of(row)].append(row)
    return {qtype: prediction_count_distribution(qrows) for qtype, qrows in sorted(grouped.items())}


def partial_credit_mean(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    vals = [float(flat_solver_eval_value(row, "partial_credit", 0.0) or 0.0) for row in rows]
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def partial_credit_mean_by_qtype(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[qtype_of(row)].append(row)
    return {qtype: partial_credit_mean(qrows) for qtype, qrows in sorted(grouped.items())}


def solver_first_exact(eval_payload: dict[str, Any]) -> bool:
    preds = eval_payload.get("predictions")
    if isinstance(preds, list) and preds and isinstance(preds[0], dict) and isinstance(preds[0].get("exact_correct"), bool):
        return bool(preds[0].get("exact_correct"))
    exact = eval_payload.get("exact_correct")
    if isinstance(exact, bool):
        return exact
    k = maybe_token(eval_payload.get("k"), 0)
    if k == 1:
        return maybe_token(eval_payload.get("exact_correct_count"), 0) > 0
    return maybe_token(eval_payload.get("exact_correct_count"), 0) > 0 and not bool(preds)


def is_boundary_frontier_row(row: dict[str, Any]) -> bool:
    return str(flat_solver_eval_value(row, "bucket", "")) in {"1/5", "2/5", "3/5", "4/5"}


def is_strict_frontier_row(row: dict[str, Any]) -> bool:
    label = row.get("frontier_label") if isinstance(row.get("frontier_label"), dict) else {}
    return bool(label.get("is_frontier"))


def is_exact_frontier_row(row: dict[str, Any]) -> bool:
    # Deprecated name retained for compatibility; this is a boundary bucket test.
    return is_boundary_frontier_row(row)


def is_near_miss_frontier_row(row: dict[str, Any]) -> bool:
    se = nested_solver_eval(row)
    exact = bool(se.get("exact_correct")) or maybe_token(se.get("exact_correct_count"), 0) > 0
    partial = float(se.get("partial_credit") or 0.0)
    return not exact and partial >= 0.70


def boundary_frontier_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if is_boundary_frontier_row(row))


def strict_frontier_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if is_strict_frontier_row(row))


def exact_frontier_count(rows: list[dict[str, Any]]) -> int:
    return boundary_frontier_count(rows)


def boundary_frontier_count_by_qtype(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if is_boundary_frontier_row(row):
            counts[qtype_of(row)] += 1
    return dict(sorted(counts.items()))


def strict_frontier_count_by_qtype(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if is_strict_frontier_row(row):
            counts[qtype_of(row)] += 1
    return dict(sorted(counts.items()))


def exact_frontier_count_by_qtype(rows: list[dict[str, Any]]) -> dict[str, int]:
    return boundary_frontier_count_by_qtype(rows)


def near_miss_frontier_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if is_near_miss_frontier_row(row))


def near_miss_frontier_count_by_qtype(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if is_near_miss_frontier_row(row):
            counts[qtype_of(row)] += 1
    return dict(sorted(counts.items()))


def leakage_count(summary: dict[str, Any] | None, report: dict[str, Any] | None = None) -> int:
    summary = summary or {}
    report = report or {}
    return sum(
        max(maybe_token(summary.get(key), 0), maybe_token(report.get(key), 0))
        for key in ("solver_visible_meta_leak_count", "anchor_leakage_count", "qa_meta_leak_count", "dialogue_meta_leak_count")
    )


def rows_below_4000_delta(summary: dict[str, Any] | None, report: dict[str, Any] | None = None) -> int:
    summary = summary or {}
    report = report or {}
    return max(maybe_token(summary.get("rows_below_4000_delta"), 0), maybe_token(report.get("rows_below_4000_delta"), 0))


def valid_rate(summary: dict[str, Any]) -> float:
    total = maybe_token(summary.get("generated_count"), 0)
    valid = maybe_token(summary.get("valid_count"), maybe_token(summary.get("valid"), 0))
    return round(valid / total, 6) if total else 0.0


def clean_rate(summary: dict[str, Any]) -> float:
    total = maybe_token(summary.get("generated_count"), 0)
    clean = maybe_token(summary.get("clean_count"), maybe_token(summary.get("clean"), 0))
    return round(clean / total, 6) if total else 0.0


def frontier_yield(summary: dict[str, Any]) -> float:
    total = maybe_token(summary.get("generated_count"), 0)
    frontier = maybe_token(summary.get("frontier_count"), 0)
    return round(frontier / total, 6) if total else 0.0


def soft_bucket_shares(rows: list[dict[str, Any]]) -> tuple[float, float]:
    counts = Counter(str(flat_solver_eval_value(row, "bucket", "unknown")) for row in rows)
    total = sum(counts.values()) or 1
    five = counts.get("5/5", 0) / total
    zero = counts.get("0/5", 0) / total
    return round(five, 6), round(zero, 6)


def soft_bucket_shares_by_qtype(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[qtype_of(row)].append(row)
    out: dict[str, dict[str, float]] = {}
    for qtype, qrows in sorted(grouped.items()):
        five, zero = soft_bucket_shares(qrows)
        out[qtype] = {"5/5": five, "0/5": zero}
    return out


def normalize_share_dict(shares: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in shares.values())
    if total <= 0:
        return {}
    return {str(k): round(max(0.0, float(v)) / total, 6) for k, v in shares.items() if max(0.0, float(v)) > 0}


def qtype_share_to_frontier_shares(qtype_shares: dict[str, float], subtype_shares: dict[str, float] | None = None) -> dict[str, float]:
    subtype_shares = subtype_shares or BALANCED_SUBTYPE_SHARES
    single = float(qtype_shares.get("single_choice", 0.0) or 0.0)
    return {
        "ordinary_single_choice": round(single * float(subtype_shares.get("ordinary_single_choice", 0.0) or 0.0), 6),
        "hard_single_choice": round(single * float(subtype_shares.get("hard_single_choice", 0.0) or 0.0), 6),
        "multi_select": round(float(qtype_shares.get("multi_select", 0.0) or 0.0), 6),
        "ordering": round(float(qtype_shares.get("ordering", 0.0) or 0.0), 6),
    }


def frontier_shares_to_qtype_plan(frontier_shares: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    ordinary = float(frontier_shares.get("ordinary_single_choice", 0.0) or 0.0)
    hard = float(frontier_shares.get("hard_single_choice", 0.0) or 0.0)
    single = ordinary + hard
    qtype = normalize_share_dict(
        {
            "single_choice": single,
            "multi_select": float(frontier_shares.get("multi_select", 0.0) or 0.0),
            "ordering": float(frontier_shares.get("ordering", 0.0) or 0.0),
        }
    )
    subtype = {"ordinary_single_choice": 0.0, "hard_single_choice": 1.0}
    if single > 0:
        subtype = normalize_share_dict({"ordinary_single_choice": ordinary, "hard_single_choice": hard})
    return qtype, subtype


def mining_success_status(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    generated = maybe_token(summary.get("generated_count"), 0)
    valid_rate_value = float(summary.get("valid_rate") or 0.0)
    clean_rate_value = float(summary.get("clean_rate") or 0.0)
    rows_below_4000 = maybe_token(summary.get("rows_below_4000_delta"), 0)
    leakage_value = maybe_token(summary.get("leakage"), 0)
    frontier_count_value = maybe_token(summary.get("frontier_count"), 0)
    near_miss_count_value = maybe_token(summary.get("near_miss_frontier_count"), maybe_token(summary.get("near_miss_count"), 0))
    if generated < 50:
        blockers.append("smoke_lt_50")
    if valid_rate_value < 0.95:
        blockers.append("valid_rate_lt_0.95")
    if clean_rate_value < 0.95:
        blockers.append("clean_rate_lt_0.95")
    if rows_below_4000 != 0:
        blockers.append("rows_below_4000_delta")
    if leakage_value != 0:
        blockers.append("leakage")
    if generated >= 50 and not (
        frontier_count_value >= 6
        or near_miss_count_value >= 10
        or (frontier_count_value + near_miss_count_value) >= 15
    ):
        blockers.append("frontier_or_near_miss_count_low")
    return not blockers, blockers


def gate_ready_status(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    generated = maybe_token(summary.get("generated_count"), 0)
    valid_rate_value = float(summary.get("valid_rate") or 0.0)
    clean_rate_value = float(summary.get("clean_rate") or 0.0)
    rows_below_4000 = maybe_token(summary.get("rows_below_4000_delta"), 0)
    leakage_value = maybe_token(summary.get("leakage"), 0)
    frontier_yield_value = float(summary.get("frontier_yield") or 0.0)
    near_miss_yield_value = float(summary.get("near_miss_frontier_yield") or 0.0)
    overall_zero_share = float(summary.get("zero_of_five_share") or 0.0)
    multi_select_zero_share = float(summary.get("multi_select_zero_share") or 0.0)
    five_of_five_share = float(summary.get("five_of_five_share") or 0.0)
    if generated < 50:
        blockers.append("smoke_lt_50")
    if valid_rate_value < 0.95:
        blockers.append("valid_rate_lt_0.95")
    if clean_rate_value < 0.95:
        blockers.append("clean_rate_lt_0.95")
    if rows_below_4000 != 0:
        blockers.append("rows_below_4000_delta")
    if leakage_value != 0:
        blockers.append("leakage")
    if frontier_yield_value < 0.10 and near_miss_yield_value < 0.20:
        blockers.append("frontier_or_near_miss_yield_low")
    if overall_zero_share > 0.40:
        blockers.append("overall_zero_share_gt_0.40")
    if multi_select_zero_share > 0.45:
        blockers.append("multi_select_zero_share_gt_0.45")
    if five_of_five_share > 0.65:
        blockers.append("five_of_five_share_gt_0.65")
    return not blockers, blockers


def hard_single_choice_signal_ok(summary: dict[str, Any]) -> bool:
    hard_count = maybe_token(summary.get("hard_single_choice_count"), 0)
    hard_frontier = maybe_token(summary.get("hard_single_choice_boundary_count"), 0)
    hard_near_miss = maybe_token(summary.get("hard_single_choice_near_miss_count"), 0)
    if hard_count <= 0:
        return False
    if hard_frontier <= 0 and hard_near_miss <= 0:
        return False
    hard_five = float(summary.get("hard_single_choice_five_of_five_share") or 0.0)
    hard_zero = float(summary.get("hard_single_choice_zero_of_five_share") or 0.0)
    return hard_five < 1.0 and hard_zero < 1.0


def read_generation_artifacts(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    generation_dir = run_root / "generation"
    samples = normalize_rows(read_jsonl(generation_dir / "samples.jsonl"))
    k1_rows = normalize_rows(read_jsonl(generation_dir / "solver_k1_eval.jsonl"))
    k5_rows = normalize_rows(read_jsonl(generation_dir / "solver_k5_eval.jsonl"))
    k1_by_id = solver_eval_map(k1_rows)
    k5_by_id = solver_eval_map(k5_rows)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = copy.deepcopy(sample)
        sid = str(row.get("sample_id") or "")
        if sid and sid in k1_by_id:
            row["solver_eval_k1"] = k1_by_id[sid]
        if sid and sid in k5_by_id:
            row["solver_eval_k5"] = k5_by_id[sid]
        if isinstance(row.get("solver_eval_k5"), dict):
            row["solver_eval"] = row["solver_eval_k5"]
        elif isinstance(row.get("solver_eval_k1"), dict):
            row["solver_eval"] = row["solver_eval_k1"]
        rows.append(row)
    summary = read_json(generation_dir / "generation_summary.json")
    report = read_json(run_root / "reports" / "report.json")
    report_md = run_root / "reports" / "report.md"
    if report_md.exists():
        report["_report_md_path"] = str(report_md.resolve())
        report["_report_md_chars"] = len(report_md.read_text(encoding="utf-8"))
    summary["_k1_eval_count"] = len(k1_rows)
    summary["_k5_eval_count"] = len(k5_rows)
    return rows, summary, report


def run_round(
    *,
    run_root: Path,
    run_name: str,
    n_train: int,
    difficulty_state_path: Path,
    parent_dir: Path,
    solver_model: Path,
    solver_tokenizer: Path,
    length_guard_parent_baseline_tokens: int,
    min_length_delta_tokens: int,
    generation_concurrency: int,
    max_attempts_per_sample: int,
    max_api_failures: int,
    checkpoint_every: int,
    skip_closed_loop_controller_update: bool,
    train_only: bool = True,
    seed: int | None = None,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--run-name",
        run_name,
        "--run-root",
        path_cli_arg(run_root),
        "--round-id",
        "current_method_r4_test_probe197",
        "--round-number",
        "4",
        "--parent-round-id",
        "current_method_r3_phase19_cell_reward_n500_frontier_mining",
        "--parent-round-number",
        "3",
        "--parent-mode",
        "round_dataset",
        "--parent-dir",
        path_cli_arg(parent_dir),
        "--fixed-solver-name",
        "solver_current_method_r3_restart_phase19_probe197",
        "--generation-backend",
        "native_current_method",
        "--native-challenger-mode",
        "graph_conditioned_api_dialogue",
        "--native-curriculum-mode",
        "r3_long_easy_v2",
        "--closed-loop-difficulty",
        "--disable-smoke-controller-local-state",
        "--difficulty-state-path",
        path_cli_arg(difficulty_state_path),
        "--min-length-delta-tokens",
        str(min_length_delta_tokens),
        "--length-guard-parent-baseline-tokens",
        str(length_guard_parent_baseline_tokens),
        "--solver-model",
        path_cli_arg(solver_model),
        "--solver-tokenizer",
        str(solver_tokenizer),
        "--n-train",
        str(n_train),
        "--generation-concurrency",
        str(generation_concurrency),
        "--run-k5",
        "--k5-eval-temperature",
        "1.0",
        "--frontier-splits",
        "train",
        "--forbid-frontier-from-dev-heldout",
        "--max-attempts-per-sample",
        str(max_attempts_per_sample),
        "--max-api-failures",
        str(max_api_failures),
        "--checkpoint-every",
        str(checkpoint_every),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if skip_closed_loop_controller_update:
        cmd.append("--skip-closed-loop-controller-update")
    if train_only:
        cmd.append("--train-only")
    shell(cmd, cwd=ROOT)


def current_difficulty_plan(state_path: Path) -> dict[str, Any]:
    state = read_json(state_path)
    plan = state.get("difficulty_plan") if isinstance(state.get("difficulty_plan"), dict) else {}
    return copy.deepcopy(plan)


def apply_balanced_plan(plan: dict[str, Any], stage: str = BALANCED_MINING_STAGE) -> dict[str, Any]:
    plan = copy.deepcopy(plan)
    plan["question_type_shares"] = dict(BALANCED_QTYPE_SHARES)
    plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
    plan["qtype_curriculum"] = {
        **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
        "stage": stage,
        "mode": "balanced",
        "manual_policy_correction": MANUAL_POLICY_CORRECTION,
        "share_policy": {
            **dict(BALANCED_FRONTIER_SHARES),
            "qtype_delta_cap": 0.05,
            "single_choice_floor": 0.05,
            "hard_single_choice_min": 0.25,
            "hard_single_choice_default": 0.35,
            "hard_single_choice_max": 0.40,
            "multi_select_min": 0.25,
            "multi_select_default": 0.35,
            "multi_select_max": 0.40,
            "ordering_min": 0.15,
            "ordering_default": 0.25,
            "ordering_max": 0.30,
        },
    }
    plan["single_choice_subtype_shares"] = dict(BALANCED_SUBTYPE_SHARES)
    plan["single_choice_subtype_hardness_mix"] = {
        "hard_single_choice": {"h0": 0.50, "h1": 0.40, "h2": 0.10},
        "ordinary_single_choice": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
    }
    cell_weights = dict(plan.get("cell_weights") or {})
    cell_weights.update(
        {
            "single_choice|conditional_plan": 0.20,
            "single_choice|delayed_activation": 0.80,
            "multi_select|conditional_plan": 0.92,
            "multi_select|permission_visibility_scope": 0.08,
            "ordering|conditional_plan": 0.10,
            "ordering|timeline_fork_merge": 0.90,
        }
    )
    plan["cell_weights"] = cell_weights
    cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
    cell_hardness_mix.update(
        {
            "multi_select|conditional_plan": {"h0": 0.60, "h1": 0.35, "h2": 0.05},
            "multi_select|permission_visibility_scope": {"h0": 1.00, "h1": 0.00, "h2": 0.00},
            "single_choice|conditional_plan": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
            "single_choice|delayed_activation": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
            "ordering|conditional_plan": {"h0": 0.85, "h1": 0.15, "h2": 0.00},
            "ordering|timeline_fork_merge": {"h0": 0.85, "h1": 0.15, "h2": 0.00},
        }
    )
    plan["cell_hardness_mix"] = cell_hardness_mix
    plan["skill_shares"] = {
        "conditional_plan": 0.45,
        "permission_visibility_scope": 0.08,
        "delayed_activation": 0.22,
        "timeline_fork_merge": 0.25,
    }
    return plan


def apply_plan_overrides(plan: dict[str, Any], stage: str) -> dict[str, Any]:
    plan = copy.deepcopy(plan)
    if stage == "mastered_advance_initial":
        plan["question_type_shares"] = {"single_choice": 0.65, "multi_select": 0.25, "ordering": 0.10}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
    elif stage in {"frontier_mining_2", BALANCED_MINING_STAGE}:
        plan = apply_balanced_plan(plan, BALANCED_MINING_STAGE)
    elif stage == "frontier_focus_1":
        plan["question_type_shares"] = {"single_choice": 0.35, "multi_select": 0.55, "ordering": 0.10}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.18,
                "single_choice|delayed_activation": 0.17,
                "multi_select|conditional_plan": 0.34,
                "multi_select|permission_visibility_scope": 0.31,
                "ordering|conditional_plan": 0.05,
                "ordering|timeline_fork_merge": 0.05,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.30, "h1": 0.40, "h2": 0.30},
                "multi_select|permission_visibility_scope": {"h0": 0.30, "h1": 0.40, "h2": 0.30},
                "single_choice|conditional_plan": {"h0": 0.60, "h1": 0.25, "h2": 0.15},
                "single_choice|delayed_activation": {"h0": 0.60, "h1": 0.25, "h2": 0.15},
                "ordering|conditional_plan": {"h0": 0.75, "h1": 0.15, "h2": 0.10},
                "ordering|timeline_fork_merge": {"h0": 0.75, "h1": 0.15, "h2": 0.10},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.45, "permission_visibility_scope": 0.35, "delayed_activation": 0.20}
    elif stage == "frontier_focus_2":
        plan["question_type_shares"] = {"single_choice": 0.20, "multi_select": 0.65, "ordering": 0.15}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.08,
                "single_choice|delayed_activation": 0.08,
                "multi_select|conditional_plan": 0.45,
                "multi_select|permission_visibility_scope": 0.45,
                "ordering|conditional_plan": 0.03,
                "ordering|timeline_fork_merge": 0.03,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.10, "h1": 0.35, "h2": 0.55},
                "multi_select|permission_visibility_scope": {"h0": 0.10, "h1": 0.35, "h2": 0.55},
                "single_choice|conditional_plan": {"h0": 0.70, "h1": 0.20, "h2": 0.10},
                "single_choice|delayed_activation": {"h0": 0.70, "h1": 0.20, "h2": 0.10},
                "ordering|conditional_plan": {"h0": 0.80, "h1": 0.15, "h2": 0.05},
                "ordering|timeline_fork_merge": {"h0": 0.80, "h1": 0.15, "h2": 0.05},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.40, "permission_visibility_scope": 0.40, "delayed_activation": 0.20}
    elif stage == "frontier_rebalance_1":
        plan["question_type_shares"] = {"single_choice": 0.30, "multi_select": 0.60, "ordering": 0.10}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.15,
                "single_choice|delayed_activation": 0.15,
                "multi_select|conditional_plan": 0.32,
                "multi_select|permission_visibility_scope": 0.28,
                "ordering|conditional_plan": 0.05,
                "ordering|timeline_fork_merge": 0.05,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.40, "h1": 0.45, "h2": 0.15},
                "multi_select|permission_visibility_scope": {"h0": 0.40, "h1": 0.45, "h2": 0.15},
                "single_choice|conditional_plan": {"h0": 0.75, "h1": 0.20, "h2": 0.05},
                "single_choice|delayed_activation": {"h0": 0.75, "h1": 0.20, "h2": 0.05},
                "ordering|conditional_plan": {"h0": 0.85, "h1": 0.10, "h2": 0.05},
                "ordering|timeline_fork_merge": {"h0": 0.85, "h1": 0.10, "h2": 0.05},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.42, "permission_visibility_scope": 0.38, "delayed_activation": 0.15, "timeline_fork_merge": 0.05}
    elif stage == "frontier_recovery_1":
        plan["question_type_shares"] = {"single_choice": 0.35, "multi_select": 0.55, "ordering": 0.10}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.18,
                "single_choice|delayed_activation": 0.17,
                "multi_select|conditional_plan": 0.30,
                "multi_select|permission_visibility_scope": 0.25,
                "ordering|conditional_plan": 0.05,
                "ordering|timeline_fork_merge": 0.05,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.65, "h1": 0.35, "h2": 0.00},
                "multi_select|permission_visibility_scope": {"h0": 0.65, "h1": 0.35, "h2": 0.00},
                "single_choice|conditional_plan": {"h0": 0.85, "h1": 0.15, "h2": 0.00},
                "single_choice|delayed_activation": {"h0": 0.85, "h1": 0.15, "h2": 0.00},
                "ordering|conditional_plan": {"h0": 0.95, "h1": 0.05, "h2": 0.00},
                "ordering|timeline_fork_merge": {"h0": 0.95, "h1": 0.05, "h2": 0.00},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.43, "permission_visibility_scope": 0.32, "delayed_activation": 0.20, "timeline_fork_merge": 0.05}
    elif stage == "frontier_recovery_2":
        plan["question_type_shares"] = {"single_choice": 0.40, "multi_select": 0.55, "ordering": 0.05}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.20,
                "single_choice|delayed_activation": 0.20,
                "multi_select|conditional_plan": 0.22,
                "multi_select|permission_visibility_scope": 0.33,
                "ordering|conditional_plan": 0.02,
                "ordering|timeline_fork_merge": 0.03,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.80, "h1": 0.20, "h2": 0.00},
                "multi_select|permission_visibility_scope": {"h0": 0.75, "h1": 0.25, "h2": 0.00},
                "single_choice|conditional_plan": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "single_choice|delayed_activation": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "ordering|conditional_plan": {"h0": 1.00, "h1": 0.00, "h2": 0.00},
                "ordering|timeline_fork_merge": {"h0": 1.00, "h1": 0.00, "h2": 0.00},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.35, "permission_visibility_scope": 0.35, "delayed_activation": 0.25, "timeline_fork_merge": 0.05}
    elif stage == "frontier_bridge_1":
        plan["question_type_shares"] = {"single_choice": 0.35, "multi_select": 0.60, "ordering": 0.05}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.18,
                "single_choice|delayed_activation": 0.17,
                "multi_select|conditional_plan": 0.48,
                "multi_select|permission_visibility_scope": 0.12,
                "ordering|conditional_plan": 0.02,
                "ordering|timeline_fork_merge": 0.03,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.75, "h1": 0.25, "h2": 0.00},
                "multi_select|permission_visibility_scope": {"h0": 1.00, "h1": 0.00, "h2": 0.00},
                "single_choice|conditional_plan": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "single_choice|delayed_activation": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "ordering|conditional_plan": {"h0": 1.00, "h1": 0.00, "h2": 0.00},
                "ordering|timeline_fork_merge": {"h0": 1.00, "h1": 0.00, "h2": 0.00},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.45, "permission_visibility_scope": 0.20, "delayed_activation": 0.30, "timeline_fork_merge": 0.05}
    elif stage == "frontier_mining_1":
        plan["question_type_shares"] = {"single_choice": 0.15, "multi_select": 0.70, "ordering": 0.15}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.08,
                "single_choice|delayed_activation": 0.07,
                "multi_select|conditional_plan": 0.55,
                "multi_select|permission_visibility_scope": 0.15,
                "ordering|conditional_plan": 0.02,
                "ordering|timeline_fork_merge": 0.13,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.55, "h1": 0.35, "h2": 0.10},
                "multi_select|permission_visibility_scope": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "single_choice|conditional_plan": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "single_choice|delayed_activation": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "ordering|conditional_plan": {"h0": 1.00, "h1": 0.00, "h2": 0.00},
                "ordering|timeline_fork_merge": {"h0": 0.85, "h1": 0.15, "h2": 0.00},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.50, "permission_visibility_scope": 0.15, "delayed_activation": 0.15, "timeline_fork_merge": 0.20}
    elif stage == "quality_failed":
        plan["question_type_shares"] = {"single_choice": 0.45, "multi_select": 0.50, "ordering": 0.05}
        plan["allowed_question_types"] = ["single_choice", "multi_select", "ordering"]
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
        cell_weights = dict(plan.get("cell_weights") or {})
        cell_weights.update(
            {
                "single_choice|conditional_plan": 0.24,
                "single_choice|delayed_activation": 0.21,
                "multi_select|conditional_plan": 0.28,
                "multi_select|permission_visibility_scope": 0.22,
                "ordering|conditional_plan": 0.025,
                "ordering|timeline_fork_merge": 0.025,
            }
        )
        plan["cell_weights"] = cell_weights
        cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
        cell_hardness_mix.update(
            {
                "multi_select|conditional_plan": {"h0": 0.35, "h1": 0.40, "h2": 0.25},
                "multi_select|permission_visibility_scope": {"h0": 0.35, "h1": 0.40, "h2": 0.25},
                "single_choice|conditional_plan": {"h0": 0.70, "h1": 0.20, "h2": 0.10},
                "single_choice|delayed_activation": {"h0": 0.70, "h1": 0.20, "h2": 0.10},
                "ordering|conditional_plan": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
                "ordering|timeline_fork_merge": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
            }
        )
        plan["cell_hardness_mix"] = cell_hardness_mix
        plan["skill_shares"] = {"conditional_plan": 0.45, "permission_visibility_scope": 0.35, "delayed_activation": 0.20}
    else:
        plan["qtype_curriculum"] = {
            **(plan.get("qtype_curriculum") if isinstance(plan.get("qtype_curriculum"), dict) else {}),
            "stage": stage,
        }
    return plan


def bonus_group_for_row(row: dict[str, Any]) -> str | None:
    qtype = qtype_of(row)
    if qtype == "single_choice" and difficulty_subtype_of(row) == "hard_single_choice":
        template = hard_single_choice_template_of(row)
        if template in HARD_SINGLE_CHOICE_TEMPLATES:
            return f"hard_single_choice_template|{template}"
        return "hard_single_choice"
    if qtype == "multi_select" and skill_of(row) == "conditional_plan":
        return "multi_select|conditional_plan"
    if qtype == "ordering" and skill_of(row) == "timeline_fork_merge":
        return "ordering|timeline_fork_merge"
    return None


def bonus_group_qtype(group: str) -> str:
    if group.startswith("hard_single_choice"):
        return "hard_single_choice"
    if group.startswith("multi_select|"):
        return "multi_select"
    if group.startswith("ordering|"):
        return "ordering"
    return group


def utility_stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = bonus_group_for_row(row)
        if group:
            grouped[group].append(row)
    stats: dict[str, dict[str, Any]] = {}
    for group, grows in sorted(grouped.items()):
        n = len(grows)
        if n <= 0:
            continue
        exact = boundary_frontier_count(grows)
        strict = strict_frontier_count(grows)
        near = near_miss_frontier_count(grows)
        five, zero = soft_bucket_shares(grows)
        exact_yield = exact / n
        near_yield = near / n
        utility = (
            1.0 * exact_yield
            + 0.5 * near_yield
            - 0.3 * max(0.0, zero - 0.50)
            - 0.2 * max(0.0, five - 0.70)
        )
        stats[group] = {
            "n": n,
            "boundary_frontier_count": exact,
            "strict_frontier_count": strict,
            "near_miss_count": near,
            "boundary_frontier_yield": round(exact_yield, 6),
            "strict_frontier_yield": round(strict / n, 6),
            "near_miss_yield": round(near_yield, 6),
            "zero_share": zero,
            "five_share": five,
            "utility": round(utility, 6),
            "qtype_bonus_bucket": bonus_group_qtype(group),
        }
    return stats


def largest_remainder_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    if total <= 0:
        return {str(k): 0 for k in weights}
    positive = {str(k): max(0.0, float(v)) for k, v in weights.items() if max(0.0, float(v)) > 0}
    if not positive:
        return {}
    raw = {k: v / sum(positive.values()) * total for k, v in positive.items()}
    counts = {k: floor(v) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(positive, key=lambda k: (-(raw[k] - counts[k]), k))
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def capped_allocate_counts(weights: dict[str, float], capacities: dict[str, int], total: int) -> dict[str, int]:
    allocations = {str(k): 0 for k in capacities}
    remaining = max(0, int(total))
    remaining = min(remaining, sum(max(0, int(cap)) for cap in capacities.values()))
    while remaining > 0:
        available = {
            str(k): max(0.0, float(weights.get(k, 0.0) or 0.0))
            for k, cap in capacities.items()
            if allocations.get(str(k), 0) < int(cap) and max(0.0, float(weights.get(k, 0.0) or 0.0)) > 0
        }
        if not available:
            break
        proposal = largest_remainder_counts(available, remaining)
        progressed = False
        for key, count in sorted(proposal.items()):
            room = int(capacities.get(key, 0)) - allocations.get(key, 0)
            take = min(max(0, int(count)), max(0, room), remaining)
            if take <= 0:
                continue
            allocations[key] += take
            remaining -= take
            progressed = True
        if not progressed:
            break
    return {k: v for k, v in allocations.items() if v > 0}


def build_mining_bonus_plan(
    base_plan: dict[str, Any],
    base_rows: list[dict[str, Any]],
    *,
    base_count: int,
    bonus_count: int,
    recent_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = apply_balanced_plan(base_plan, MINING_BONUS_STAGE)
    total = max(1, base_count + bonus_count)
    utility_stats = utility_stats_for_rows(base_rows)
    base_frontier_counts = largest_remainder_counts(BALANCED_FRONTIER_SHARES, base_count)
    capacity_counts = {
        key: max(0, int(round(MINING_BONUS_CAPS[key] * total)) - base_frontier_counts.get(key, 0))
        for key in ("hard_single_choice", "multi_select", "ordering")
    }
    ordering_base_n = sum(1 for row in base_rows if qtype_of(row) == "ordering")
    if ordering_base_n < 8:
        capacity_counts["ordering"] = min(capacity_counts.get("ordering", 0), max(1, int(round(0.05 * total))))
    utility_by_bucket: dict[str, float] = {}
    for group, stats in utility_stats.items():
        bucket = bonus_group_qtype(group)
        if bucket not in capacity_counts:
            continue
        utility_by_bucket[bucket] = utility_by_bucket.get(bucket, 0.0) + max(0.0, float(stats.get("utility") or 0.0))
    recent_signal_rows = recent_rows if recent_rows is not None else base_rows
    recent_zero_by_bucket = {
        "hard_single_choice": soft_bucket_shares([row for row in recent_signal_rows if qtype_of(row) == "single_choice" and difficulty_subtype_of(row) == "hard_single_choice"])[1],
        "multi_select": soft_bucket_shares([row for row in recent_signal_rows if qtype_of(row) == "multi_select"])[1],
        "ordering": soft_bucket_shares([row for row in recent_signal_rows if qtype_of(row) == "ordering"])[1],
    }
    recent_n_by_bucket = {
        "hard_single_choice": sum(1 for row in recent_signal_rows if qtype_of(row) == "single_choice" and difficulty_subtype_of(row) == "hard_single_choice"),
        "multi_select": sum(1 for row in recent_signal_rows if qtype_of(row) == "multi_select"),
        "ordering": sum(1 for row in recent_signal_rows if qtype_of(row) == "ordering"),
    }
    zero_freeze: dict[str, str] = {}
    if recent_n_by_bucket.get("ordering", 0) > 0 and recent_zero_by_bucket.get("ordering", 0.0) > ORDERING_BONUS_FREEZE_ZERO_THRESHOLD:
        capacity_counts["ordering"] = 0
        utility_by_bucket.pop("ordering", None)
        zero_freeze["ordering"] = f"recent_zero_share_gt_{ORDERING_BONUS_FREEZE_ZERO_THRESHOLD}"
    for bucket, zero_share in recent_zero_by_bucket.items():
        if bucket == "hard_single_choice":
            continue
        if zero_share > RECENT_ZERO_PENALTY_THRESHOLD and bucket in utility_by_bucket:
            utility_by_bucket[bucket] *= max(0.0, 1.0 - (zero_share - RECENT_ZERO_PENALTY_THRESHOLD))
    qtype_bonus_counts = Counter(capped_allocate_counts(utility_by_bucket, capacity_counts, bonus_count))
    remaining_bonus = bonus_count - sum(qtype_bonus_counts.values())
    if remaining_bonus > 0:
        remaining_caps = {
            key: max(0, capacity_counts.get(key, 0) - qtype_bonus_counts.get(key, 0))
            for key in capacity_counts
        }
        fallback_weights = {
            "hard_single_choice": BALANCED_FRONTIER_SHARES["hard_single_choice"],
        }
        if capacity_counts.get("ordering", 0) > qtype_bonus_counts.get("ordering", 0):
            fallback_weights["ordering"] = BALANCED_FRONTIER_SHARES["ordering"]
        if utility_by_bucket.get("multi_select", 0.0) > 0:
            fallback_weights["multi_select"] = BALANCED_FRONTIER_SHARES["multi_select"]
        fallback_counts = capped_allocate_counts(fallback_weights, remaining_caps, remaining_bonus)
        qtype_bonus_counts.update(fallback_counts)
    remaining_bonus = bonus_count - sum(qtype_bonus_counts.values())
    if remaining_bonus > 0:
        remaining_caps = {
            key: max(0, capacity_counts.get(key, 0) - qtype_bonus_counts.get(key, 0))
            for key in capacity_counts
        }
        # Last resort keeps the requested candidate budget inside caps; it is recorded in meta.
        fallback_weights = dict(BALANCED_FRONTIER_SHARES)
        if zero_freeze.get("ordering"):
            fallback_weights.pop("ordering", None)
        fallback_counts = capped_allocate_counts(fallback_weights, remaining_caps, remaining_bonus)
        qtype_bonus_counts.update(fallback_counts)
    unallocated_bonus_count = max(0, bonus_count - sum(qtype_bonus_counts.values()))

    hard_template_weights = {
        group.split("|", 1)[1]: max(0.0, float(stats.get("utility") or 0.0))
        for group, stats in utility_stats.items()
        if group.startswith("hard_single_choice_template|") and max(0.0, float(stats.get("utility") or 0.0)) > 0
    }
    hard_template_bonus_counts = Counter(
        largest_remainder_counts(hard_template_weights, qtype_bonus_counts.get("hard_single_choice", 0))
    )

    remaining_counts = {
        "ordinary_single_choice": 0,
        "hard_single_choice": qtype_bonus_counts.get("hard_single_choice", 0),
        "multi_select": qtype_bonus_counts.get("multi_select", 0),
        "ordering": qtype_bonus_counts.get("ordering", 0),
    }
    bonus_frontier_shares = normalize_share_dict({k: float(v) for k, v in remaining_counts.items() if v > 0})
    if not bonus_frontier_shares:
        bonus_frontier_shares = {"hard_single_choice": 1.0}
    qtype_shares, subtype_shares = frontier_shares_to_qtype_plan(bonus_frontier_shares)
    plan["question_type_shares"] = qtype_shares
    plan["single_choice_subtype_shares"] = subtype_shares
    curriculum = dict(plan.get("qtype_curriculum") or {})
    curriculum.update(
        {
            "stage": MINING_BONUS_STAGE,
            "mode": "mining_bonus",
            "base_fraction": 0.70,
            "bonus_fraction": 0.30,
            "bonus_caps": dict(MINING_BONUS_CAPS),
            "bonus_objects": [
                "hard_single_choice_templates",
                "multi_select|conditional_plan",
                "ordering|timeline_fork_merge",
            ],
        }
    )
    plan["qtype_curriculum"] = curriculum

    cell_weights = dict(plan.get("cell_weights") or {})
    if qtype_bonus_counts.get("multi_select", 0) > 0:
        cell_weights["multi_select|conditional_plan"] = max(float(cell_weights.get("multi_select|conditional_plan", 0.0) or 0.0), 1.0)
        cell_weights["multi_select|permission_visibility_scope"] = min(float(cell_weights.get("multi_select|permission_visibility_scope", 0.08) or 0.08), 0.05)
    if qtype_bonus_counts.get("ordering", 0) > 0:
        cell_weights["ordering|timeline_fork_merge"] = max(float(cell_weights.get("ordering|timeline_fork_merge", 0.0) or 0.0), 1.0)
        cell_weights["ordering|conditional_plan"] = min(float(cell_weights.get("ordering|conditional_plan", 0.10) or 0.10), 0.05)
    plan["cell_weights"] = cell_weights
    cell_hardness_mix = dict(plan.get("cell_hardness_mix") or {})
    cell_hardness_mix["multi_select|conditional_plan"] = {"h0": 0.65, "h1": 0.35, "h2": 0.00}
    cell_hardness_mix["multi_select|permission_visibility_scope"] = {"h0": 1.00, "h1": 0.00, "h2": 0.00}
    plan["cell_hardness_mix"] = cell_hardness_mix
    if hard_template_bonus_counts:
        plan["hard_single_choice_template_shares"] = normalize_share_dict({k: float(v) for k, v in hard_template_bonus_counts.items()})

    meta = {
        "mode": "mining_bonus",
        "base_count": base_count,
        "bonus_count": bonus_count,
        "utility_formula": "1.0*boundary_frontier_yield + 0.5*near_miss_yield - 0.3*max(0, zero_share-0.50) - 0.2*max(0, five_share-0.70), then recent zero-share penalty/freeze",
        "utility_stats": utility_stats,
        "recent_zero_share_by_bonus_bucket": recent_zero_by_bucket,
        "recent_n_by_bonus_bucket": recent_n_by_bucket,
        "recent_zero_source": "prior_recent_rows" if recent_rows is not None else "base_rows",
        "zero_share_freeze": zero_freeze,
        "zero_penalty_threshold": RECENT_ZERO_PENALTY_THRESHOLD,
        "ordering_bonus_freeze_zero_threshold": ORDERING_BONUS_FREEZE_ZERO_THRESHOLD,
        "eligible_bonus_weights": utility_by_bucket,
        "bonus_counts": dict(qtype_bonus_counts),
        "unallocated_bonus_count_due_to_caps": unallocated_bonus_count,
        "base_frontier_counts": base_frontier_counts,
        "bonus_capacity_counts": capacity_counts,
        "ordering_base_n": ordering_base_n,
        "ordering_small_n_bonus_limited": ordering_base_n < 8,
        "desired_total_counts_with_caps": {
            "ordinary_single_choice": base_frontier_counts.get("ordinary_single_choice", 0),
            "hard_single_choice": base_frontier_counts.get("hard_single_choice", 0) + qtype_bonus_counts.get("hard_single_choice", 0),
            "multi_select": base_frontier_counts.get("multi_select", 0) + qtype_bonus_counts.get("multi_select", 0),
            "ordering": base_frontier_counts.get("ordering", 0) + qtype_bonus_counts.get("ordering", 0),
        },
        "bonus_frontier_shares": bonus_frontier_shares,
        "bonus_qtype_shares": qtype_shares,
        "bonus_single_choice_subtype_shares": subtype_shares,
        "hard_single_choice_template_shares": plan.get("hard_single_choice_template_shares", {}),
    }
    return plan, meta


def frontier_pool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in normalize_rows(rows):
        exact = is_boundary_frontier_row(row)
        near = is_near_miss_frontier_row(row)
        if not exact and not near:
            continue
        sid = str(row.get("sample_id") or row.get("expanded_graph_id") or id(row))
        if sid in seen:
            continue
        seen.add(sid)
        candidate = copy.deepcopy(row)
        candidate["frontier_pool_label"] = "boundary_frontier" if exact else "near_miss"
        candidate["frontier_pool_priority"] = 0 if exact else 1
        pool.append(candidate)
    pool.sort(
        key=lambda row: (
            int(row.get("frontier_pool_priority", 9)),
            -float(flat_solver_eval_value(row, "partial_credit", 0.0) or 0.0),
            str(row.get("sample_id") or ""),
        )
    )
    return pool


def update_state_and_plan(state_path: Path, plan: dict[str, Any]) -> None:
    state = read_json(state_path)
    state["difficulty_plan"] = plan
    write_json(state_path, state)


def write_stage_artifacts(run_root: Path, summary: dict[str, Any], history: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    write_json(run_root / "adaptive_smoke_summary.json", summary)
    write_jsonl(run_root / "adaptive_smoke_history.jsonl", history)
    write_jsonl(run_root / "adaptive_smoke_rows.jsonl", rows)


def build_summary(
    *,
    stage: str,
    phase: str,
    run_name: str,
    run_root: Path,
    n_train: int,
    rows: list[dict[str, Any]],
    generation_summary: dict[str, Any],
    report: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    rows = normalize_rows(rows)
    gen = generation_summary or {}
    rep = report or {}
    generated_count = maybe_token(gen.get("generated_count"), len(rows))
    strict_count = maybe_token(rep.get("frontier_count"), maybe_token(gen.get("frontier_count"), strict_frontier_count(rows)))
    strict_yield_value = round(strict_count / generated_count, 6) if generated_count else 0.0
    boundary_count = boundary_frontier_count(rows)
    boundary_yield_value = round(boundary_count / generated_count, 6) if generated_count else 0.0
    overall_five_share, overall_zero_share = soft_bucket_shares(rows)
    soft_share_by_qtype = soft_bucket_shares_by_qtype(rows)
    multi_select_zero_share = float(soft_share_by_qtype.get("multi_select", {}).get("0/5", 0.0) or 0.0)
    ordering_zero_share = float(soft_share_by_qtype.get("ordering", {}).get("0/5", 0.0) or 0.0)
    near_miss_count = near_miss_frontier_count(rows)
    hard_rows = [row for row in rows if difficulty_subtype_of(row) == "hard_single_choice"]
    ordinary_rows = [row for row in rows if difficulty_subtype_of(row) == "ordinary_single_choice"]
    hard_templates = [hard_single_choice_template_of(row) for row in hard_rows]
    hard_template_counts = dict(Counter(hard_templates).most_common())
    mining_success, mining_blockers = mining_success_status(
        {
            "generated_count": generated_count,
            "valid_rate": valid_rate(gen if gen else rep),
            "clean_rate": clean_rate(gen if gen else rep),
            "rows_below_4000_delta": rows_below_4000_delta(gen, rep),
            "leakage": leakage_count(gen, rep),
            "frontier_count": strict_count,
            "strict_frontier_count": strict_count,
            "boundary_frontier_count": boundary_count,
            "near_miss_frontier_count": near_miss_count,
        }
    )
    gate_ready, gate_blockers = gate_ready_status(
        {
            "generated_count": generated_count,
            "valid_rate": valid_rate(gen if gen else rep),
            "clean_rate": clean_rate(gen if gen else rep),
            "rows_below_4000_delta": rows_below_4000_delta(gen, rep),
            "leakage": leakage_count(gen, rep),
            "frontier_yield": strict_yield_value,
            "strict_frontier_yield": strict_yield_value,
            "boundary_frontier_yield": boundary_yield_value,
            "near_miss_frontier_yield": round(near_miss_frontier_count(rows) / len(rows), 6) if rows else 0.0,
            "zero_of_five_share": overall_zero_share,
            "multi_select_zero_share": multi_select_zero_share,
            "five_of_five_share": overall_five_share,
        }
    )
    summary = {
        "run_name": run_name,
        "run_root": str(run_root.resolve()),
        "state_path": str(state_path.resolve()),
        "stage": stage,
        "phase": phase,
        "generated_count": generated_count,
        "valid_count": maybe_token(gen.get("valid_count"), maybe_token(rep.get("valid_count"), 0)),
        "clean_count": maybe_token(gen.get("clean_count"), maybe_token(rep.get("clean_count"), 0)),
        "valid_rate": valid_rate(gen if gen else rep),
        "clean_rate": clean_rate(gen if gen else rep),
        "frontier_count": strict_count,
        "frontier_yield": strict_yield_value,
        "strict_frontier_count": strict_count,
        "strict_frontier_yield": strict_yield_value,
        "strict_frontier_count_by_qtype": strict_frontier_count_by_qtype(rows),
        "boundary_frontier_count": boundary_count,
        "boundary_frontier_yield": boundary_yield_value,
        "boundary_frontier_count_by_qtype": boundary_frontier_count_by_qtype(rows),
        "frontier_pool_count": len({str(row.get("sample_id") or row.get("expanded_graph_id") or idx) for idx, row in enumerate(rows) if is_boundary_frontier_row(row) or is_near_miss_frontier_row(row)}),
        # Deprecated aliases: exact here historically meant 1/5-4/5 boundary bucket.
        "exact_frontier_count": boundary_count,
        "exact_frontier_count_by_qtype": boundary_frontier_count_by_qtype(rows),
        "near_miss_frontier_count": near_miss_count,
        "near_miss_count": near_miss_count,
        "near_miss_frontier_count_by_qtype": near_miss_frontier_count_by_qtype(rows),
        "near_miss_frontier_yield": round(near_miss_count / len(rows), 6) if rows else 0.0,
        "frontier_plus_near_miss_count": strict_count + near_miss_count,
        "boundary_plus_near_miss_count": len({str(row.get("sample_id") or row.get("expanded_graph_id") or idx) for idx, row in enumerate(rows) if is_boundary_frontier_row(row) or is_near_miss_frontier_row(row)}),
        "rows_below_4000_delta": rows_below_4000_delta(gen, rep),
        "leakage": leakage_count(gen, rep),
        "qtype_distribution": qtype_counts(rows),
        "qtype_bucket_distribution": qtype_bucket_counts(rows),
        "qtype_skill_bucket_distribution": qtype_skill_bucket_counts(rows),
        "bucket_distribution": bucket_counts(rows),
        "overall_five_of_five_share": overall_five_share,
        "overall_zero_of_five_share": overall_zero_share,
        "multi_select_zero_share": multi_select_zero_share,
        "ordering_zero_share": ordering_zero_share,
        "hard_single_choice_count": len(hard_rows),
        "ordinary_single_choice_count": len(ordinary_rows),
        "hard_single_choice_bucket_distribution": bucket_counts(hard_rows),
        "ordinary_single_choice_bucket_distribution": bucket_counts(ordinary_rows),
        "hard_single_choice_boundary_count": boundary_frontier_count(hard_rows),
        "hard_single_choice_strict_frontier_count": strict_frontier_count(hard_rows),
        "hard_single_choice_near_miss_count": near_miss_frontier_count(hard_rows),
        "hard_single_choice_partial_credit_mean": partial_credit_mean(hard_rows),
        "hard_single_choice_five_of_five_share": soft_bucket_shares(hard_rows)[0],
        "hard_single_choice_zero_of_five_share": soft_bucket_shares(hard_rows)[1],
        "ordinary_single_choice_five_of_five_share": soft_bucket_shares(ordinary_rows)[0],
        "ordinary_single_choice_zero_of_five_share": soft_bucket_shares(ordinary_rows)[1],
        "mining_success": mining_success,
        "mining_blockers": mining_blockers,
        "hard_single_choice_by_template": hard_template_counts,
        "hard_single_choice_by_template_bucket_distribution": {
            template: bucket_counts([row for row in hard_rows if hard_single_choice_template_of(row) == template])
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_frontier_count": {
            template: boundary_frontier_count([row for row in hard_rows if hard_single_choice_template_of(row) == template])
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_boundary_count": {
            template: boundary_frontier_count([row for row in hard_rows if hard_single_choice_template_of(row) == template])
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_strict_frontier_count": {
            template: strict_frontier_count([row for row in hard_rows if hard_single_choice_template_of(row) == template])
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_near_miss_count": {
            template: near_miss_frontier_count([row for row in hard_rows if hard_single_choice_template_of(row) == template])
            for template in sorted(hard_template_counts)
        },
        "prediction_count_distribution": prediction_count_distribution(rows),
        "prediction_count_distribution_by_qtype": prediction_count_distribution_by_qtype(rows),
        "soft_bucket_share_by_qtype": soft_bucket_shares_by_qtype(rows),
        "k1_eval_count": maybe_token(gen.get("_k1_eval_count"), 0),
        "k5_eval_count": maybe_token(gen.get("_k5_eval_count"), 0),
        "k1_exact_accuracy": round(
            sum(1 for row in rows if solver_first_exact(row.get("solver_eval_k1") or {})) / len(rows), 6
        ) if rows and any(isinstance(row.get("solver_eval_k1"), dict) for row in rows) else None,
        "k5_exact_accuracy": round(
            sum(1 for row in rows if solver_first_exact(row.get("solver_eval_k5") or {})) / len(rows), 6
        ) if rows and any(isinstance(row.get("solver_eval_k5"), dict) for row in rows) else None,
        "partial_credit_mean": partial_credit_mean(rows),
        "partial_credit_mean_by_qtype": partial_credit_mean_by_qtype(rows),
        "five_of_five_share": overall_five_share,
        "zero_of_five_share": overall_zero_share,
        "total_rows": len(rows),
        "n_train": n_train,
        "frontier_yield_gt_0.10": strict_yield_value >= 0.10,
        "frontier_or_near_miss": strict_count > 0 or near_miss_frontier_count(rows) > 0,
        "gate_ready": gate_ready,
        "gate_blockers": gate_blockers,
        "report_md_path": rep.get("_report_md_path"),
        "report_md_chars": rep.get("_report_md_chars"),
    }
    return summary


def gate_passed(summary: dict[str, Any]) -> bool:
    return bool(summary.get("gate_ready"))


def mining_smoke_passed(summary: dict[str, Any]) -> bool:
    return bool(summary.get("mining_success"))


def choose_next_stage(summary: dict[str, Any], current_stage: str) -> str:
    if summary["rows_below_4000_delta"] > 0:
        return "length_failed"
    if summary["valid_rate"] < 0.90 or summary["clean_rate"] < 0.90 or summary["leakage"] > 0:
        return "quality_failed"
    soft_by_qtype = summary.get("soft_bucket_share_by_qtype", {})
    multi_select_soft = soft_by_qtype.get("multi_select", {})
    ordering_soft = soft_by_qtype.get("ordering", {})
    multi_select_zero = float(multi_select_soft.get("0/5", 0.0) or 0.0)
    ordering_zero = float(ordering_soft.get("0/5", 0.0) or 0.0)
    single_choice_five = float(soft_by_qtype.get("single_choice", {}).get("5/5", 0.0) or 0.0)
    signal_yield = float(summary.get("frontier_yield", 0.0) or 0.0) + 0.5 * float(summary.get("near_miss_frontier_yield", 0.0) or 0.0)
    if current_stage in {"frontier_bridge_1", "frontier_recovery_2"} and signal_yield >= 0.10 and summary["zero_of_five_share"] <= 0.55:
        return BALANCED_MINING_STAGE
    if current_stage == "frontier_mining_1":
        if summary["zero_of_five_share"] > 0.60 or multi_select_zero > 0.80:
            return "frontier_bridge_1"
        return BALANCED_MINING_STAGE
    if current_stage == "frontier_mining_2":
        hard_zero = float(summary.get("hard_single_choice_zero_of_five_share", 0.0) or 0.0)
        if summary["zero_of_five_share"] > 0.55 or hard_zero > 0.70:
            return "frontier_recovery_1"
        return BALANCED_MINING_STAGE
    if current_stage == BALANCED_MINING_STAGE:
        if summary["zero_of_five_share"] > 0.55 or multi_select_zero > 0.75:
            return "frontier_recovery_1"
        return BALANCED_MINING_STAGE
    if current_stage == "frontier_recovery_2" and summary["zero_of_five_share"] <= 0.40:
        if summary["frontier_yield"] < 0.10 and summary["near_miss_frontier_yield"] < 0.20:
            return "frontier_bridge_1"
        if multi_select_zero > 0.45:
            return "frontier_bridge_1"
    if current_stage == "frontier_bridge_1" and summary["zero_of_five_share"] <= 0.40:
        if summary["frontier_yield"] < 0.10 and summary["near_miss_frontier_yield"] < 0.20:
            return "frontier_bridge_1"
        if multi_select_zero > 0.45:
            return "frontier_bridge_1"
    if current_stage == "frontier_recovery_1" and (
        multi_select_zero > 0.45 or ordering_zero > 0.40 or summary["zero_of_five_share"] > 0.40
    ):
        return "frontier_recovery_2"
    if current_stage == "frontier_recovery_2" and (
        multi_select_zero > 0.45 or ordering_zero > 0.40 or summary["zero_of_five_share"] > 0.40
    ):
        return "frontier_recovery_2"
    if multi_select_zero > 0.75 or summary["zero_of_five_share"] > 0.55:
        return "frontier_recovery_1"
    if multi_select_zero > 0.55 or summary["zero_of_five_share"] > 0.40:
        return "frontier_rebalance_1"
    if ordering_zero > 0.40:
        return "frontier_rebalance_1"
    if single_choice_five > 0.80:
        return "frontier_focus_1"
    if summary["qtype_distribution"].get("multi_select", 0) and (summary["boundary_frontier_count"] > 0 or summary["near_miss_frontier_count"] > 0):
        return "frontier_focus_1" if current_stage != "frontier_focus_1" else "frontier_focus_2"
    if summary["five_of_five_share"] > 0.65 and (summary["frontier_yield"] < 0.10 and summary["near_miss_frontier_yield"] < 0.20):
        return "frontier_focus_2"
    if summary["zero_of_five_share"] > 0.40 and summary["partial_credit_mean"] < 0.40:
        return "frontier_focus_1"
    return current_stage


def latest_completed_adaptive_eval_run(run_root_base: Path) -> Path | None:
    if not run_root_base.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for child in run_root_base.iterdir():
        if not child.is_dir():
            continue
        generation_summary = child / "generation" / "generation_summary.json"
        samples = child / "generation" / "samples.jsonl"
        report = child / "reports" / "report.json"
        k5_eval = child / "generation" / "solver_k5_eval.jsonl"
        if not generation_summary.exists() or not samples.exists() or not report.exists():
            continue
        if not k5_eval.exists() and not (child / "adaptive_smoke_summary.json").exists():
            continue
        summary_path = child / "adaptive_smoke_summary.json"
        if summary_path.exists():
            try:
                summary_payload = read_json(summary_path)
                if summary_payload.get("action") == "api_failed" or maybe_token(summary_payload.get("generated_count"), 0) <= 0:
                    continue
            except (json.JSONDecodeError, ValueError):
                continue
        marker = child / "adaptive_smoke_summary.json"
        mtime = marker.stat().st_mtime if marker.exists() else generation_summary.stat().st_mtime
        candidates.append((mtime, child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def latest_mining_bonus_recent_rows(run_root_base: Path, exclude: Path | None = None) -> list[dict[str, Any]]:
    if not run_root_base.exists():
        return []
    candidates: list[tuple[float, Path]] = []
    exclude_resolved = exclude.resolve() if exclude else None
    for child in run_root_base.iterdir():
        if not child.is_dir():
            continue
        if exclude_resolved is not None:
            try:
                if child.resolve() == exclude_resolved:
                    continue
            except OSError:
                pass
        summary_path = child / "mining_bonus_summary.json"
        bonus_samples = child / "bonus_mining" / "generation" / "samples.jsonl"
        if summary_path.exists() and bonus_samples.exists():
            candidates.append((summary_path.stat().st_mtime, child))
    if not candidates:
        return []
    latest = max(candidates, key=lambda item: item[0])[1]
    try:
        rows, _gen, _rep = read_generation_artifacts(latest / "bonus_mining")
        return rows
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []


def stage_hint_from_run(run_root: Path, default: str = "mastered_advance_initial") -> str:
    summary_path = run_root / "adaptive_smoke_summary.json"
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
            stage = summary.get("stage")
            if isinstance(stage, str) and stage:
                return stage
        except (json.JSONDecodeError, ValueError):
            pass
    name = run_root.name
    for stage in (BALANCED_MINING_STAGE, "frontier_mining_2", "frontier_mining_1", "frontier_bridge_1", "frontier_recovery_2", "frontier_recovery_1", "frontier_rebalance_1", "frontier_focus_2", "frontier_focus_1", "mastered_advance_initial"):
        if stage in name:
            return stage
    return default


def bootstrap_stage_from_prior_eval(prior_run_root: Path, state_path: Path, current_stage: str) -> tuple[str, dict[str, Any] | None]:
    if not prior_run_root.exists():
        return "mastered_advance_initial", None
    rows, generation_summary, report = read_generation_artifacts(prior_run_root)
    summary = build_summary(
        stage=current_stage,
        phase="bootstrap_prior_eval",
        run_name=prior_run_root.name,
        run_root=prior_run_root,
        n_train=len(rows),
        rows=rows,
        generation_summary=generation_summary,
        report=report,
        state_path=state_path,
    )
    return choose_next_stage(summary, current_stage), summary


def build_history_entry(summary: dict[str, Any], action: str, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "phase": summary.get("phase"),
        "action": action,
        "generated_count": summary.get("generated_count"),
        "valid_rate": summary.get("valid_rate"),
        "clean_rate": summary.get("clean_rate"),
        "frontier_yield": summary.get("frontier_yield"),
        "near_miss_frontier_yield": summary.get("near_miss_frontier_yield"),
        "strict_frontier_count": summary.get("strict_frontier_count"),
        "boundary_frontier_count": summary.get("boundary_frontier_count"),
        "frontier_pool_count": summary.get("frontier_pool_count"),
        "rows_below_4000_delta": summary.get("rows_below_4000_delta"),
        "leakage": summary.get("leakage"),
        "qtype_distribution": summary.get("qtype_distribution"),
        "bucket_distribution": summary.get("bucket_distribution"),
        "exact_frontier_count": summary.get("exact_frontier_count"),
        "boundary_frontier_count": summary.get("boundary_frontier_count"),
        "near_miss_frontier_count": summary.get("near_miss_frontier_count"),
        "near_miss_count": summary.get("near_miss_count"),
        "partial_credit_mean_by_qtype": summary.get("partial_credit_mean_by_qtype"),
        "five_of_five_share": summary.get("five_of_five_share"),
        "zero_of_five_share": summary.get("zero_of_five_share"),
        "hard_single_choice_count": summary.get("hard_single_choice_count"),
        "ordinary_single_choice_count": summary.get("ordinary_single_choice_count"),
        "hard_single_choice_boundary_count": summary.get("hard_single_choice_boundary_count"),
        "hard_single_choice_strict_frontier_count": summary.get("hard_single_choice_strict_frontier_count"),
        "hard_single_choice_near_miss_count": summary.get("hard_single_choice_near_miss_count"),
        "hard_single_choice_partial_credit_mean": summary.get("hard_single_choice_partial_credit_mean"),
        "hard_single_choice_five_of_five_share": summary.get("hard_single_choice_five_of_five_share"),
        "hard_single_choice_zero_of_five_share": summary.get("hard_single_choice_zero_of_five_share"),
        "mining_success": summary.get("mining_success"),
        "gate_ready": summary.get("gate_ready"),
        "gate_blockers": summary.get("gate_blockers"),
        "frontier_plus_near_miss_count": summary.get("frontier_plus_near_miss_count"),
    }


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(k): str(v) for k, v in options.items()}
    if isinstance(options, list):
        out: dict[str, str] = {}
        for item in options:
            if not isinstance(item, dict):
                continue
            option_id = item.get("id") or item.get("label") or item.get("letter")
            option_text = item.get("text") or item.get("value") or item.get("content")
            if option_id is not None and option_text is not None:
                out[str(option_id)] = str(option_text)
        return out
    return {}


def normalize_gold_letters(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        if ">" in value:
            return [item.strip() for item in value.split(">") if item.strip()]
        return [value.strip()]
    return []


def ensure_run_root(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for sub in ("generation", "reports", "policy", "difficulty_controller", "train_mix", "models", "logs"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)


def allocate_run_root(run_root_base: Path, run_name: str) -> tuple[str, Path]:
    candidate = run_root_base / run_name
    if not candidate.exists():
        return run_name, candidate
    summary_path = candidate / "adaptive_smoke_summary.json"
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
            if summary.get("action") == "api_failed" or maybe_token(summary.get("generated_count"), 0) <= 0:
                return run_name, candidate
        except (json.JSONDecodeError, ValueError):
            return run_name, candidate
    if not summary_path.exists():
        return run_name, candidate
    suffix = 2
    while True:
        next_name = f"{run_name}_retry{suffix}"
        candidate = run_root_base / next_name
        if not candidate.exists():
            return next_name, candidate
        summary_path = candidate / "adaptive_smoke_summary.json"
        if summary_path.exists():
            try:
                summary = read_json(summary_path)
                if summary.get("action") == "api_failed" or maybe_token(summary.get("generated_count"), 0) <= 0:
                    return next_name, candidate
            except (json.JSONDecodeError, ValueError):
                return next_name, candidate
        if not summary_path.exists():
            return next_name, candidate
        suffix += 1


def sft_compatible_sample(row: dict[str, Any]) -> dict[str, Any]:
    sample = copy.deepcopy(row)
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    graph = sample.get("graph") if isinstance(sample.get("graph"), dict) else {}
    question_type = str(sample.get("question_type") or qa.get("question_type") or "single_choice")
    options = normalize_options(sample.get("options") or qa.get("options") or {})
    qa_gold = qa.get("gold")
    gold_letters = (
        normalize_gold_letters(sample.get("gold_options"))
        or normalize_gold_letters(sample.get("gold_order"))
        or normalize_gold_letters(sample.get("gold_answer"))
        or normalize_gold_letters(qa_gold)
    )
    sample["question_type"] = question_type
    sample["question"] = sample.get("question") or qa.get("question") or qa.get("prompt") or ""
    sample["options"] = options
    sample["gold_option"] = sample.get("gold_option") or qa.get("gold_option") or qa.get("answer_option")
    if question_type == "single_choice":
        if not sample.get("gold_option") and gold_letters:
            sample["gold_option"] = gold_letters[0]
        sample["gold_answer"] = options.get(str(sample.get("gold_option")), str(sample.get("gold_answer") or (gold_letters[0] if gold_letters else "")))
        sample["gold_options"] = [str(sample["gold_option"])] if sample.get("gold_option") else []
    elif question_type == "ordering":
        order = normalize_gold_letters(sample.get("gold_order")) or gold_letters
        sample["gold_order"] = order
        sample["gold_options"] = order
        sample["gold_answer"] = order
    else:
        multi = normalize_gold_letters(sample.get("gold_options")) or gold_letters
        sample["gold_options"] = multi
        sample["gold_answer"] = multi if multi else sample.get("gold_answer") or qa.get("gold_answer") or qa.get("answer") or ""
    sample["gold_source"] = sample.get("gold_source") or qa.get("gold_source") or "graph_derived"
    sample["answer_format"] = sample.get("answer_format") or qa.get("answer_format")
    sample["expanded_dialogue_text"] = (
        sample.get("expanded_dialogue_text")
        or sample.get("final_dialogue")
        or sample.get("dialogue_text")
        or sample.get("conversation")
        or ""
    )
    sample["expanded_graph_id"] = (
        sample.get("expanded_graph_id")
        or graph.get("graph_id")
        or sample.get("graph_id")
        or sample.get("sample_id")
        or sample.get("id")
        or "unknown_graph"
    )
    sample["expanded_dialogue_id"] = (
        sample.get("expanded_dialogue_id")
        or sample.get("dialogue_id")
        or sample.get("sample_id")
        or sample["expanded_graph_id"]
    )
    return sample


def ensure_sft_train(run_root: Path, rows: list[dict[str, Any]]) -> Path:
    sft_path = run_root / "generation" / "sft_train.jsonl"
    if sft_path.exists():
        return sft_path
    records = [v13.build_chat_record(sft_compatible_sample(row)) for row in rows]
    write_jsonl(sft_path, records)
    return sft_path


def build_training_command(
    *,
    run_root: Path,
    report_dir: Path,
    solver_model: Path,
    init_adapter_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_PIPELINE),
        "--round-id",
        "r4",
        "--round-number",
        "4",
        "--fixed-solver-name",
        "solver_current_method_r3_restart_phase19_probe197",
        "--solver-name",
        "solver_current_method_r4_probe197_adaptive_hardening",
        "--frontier",
        str(run_root / "generation" / "frontier.jsonl"),
        "--round-sft-train",
        str(run_root / "generation" / "sft_train.jsonl"),
        "--train-output-dir",
        str(run_root / "train_mix"),
        "--report-dir",
        str(report_dir / "training"),
        "--base-model",
        str(DEFAULT_BASE_MODEL),
        "--init-adapter-dir",
        str(init_adapter_dir),
        "--adapter-output-dir",
        str(run_root / "models" / "qwen3_4b_lora_current_method_r4_probe197_adaptive_hardening"),
        "--merged-output-dir",
        str(run_root / "models" / "qwen3_4b_lora_current_method_r4_probe197_adaptive_hardening_merged"),
        "--max-length",
        "9216",
        "--no-valid",
        "--smoke-eval-limit",
        "16",
        "--overwrite",
    ]


def infer_init_adapter_dir(solver_model: Path) -> Path:
    name = solver_model.name
    if name.endswith("_merged"):
        return solver_model.with_name(name[:-7])
    return solver_model


def run_training_if_ready(run_root: Path, summary: dict[str, Any], rows: list[dict[str, Any]], solver_model: Path, solver_tokenizer: Path) -> str:
    if not gate_passed(summary):
        return "not_ready"
    ensure_sft_train(run_root, rows)
    init_adapter_dir = infer_init_adapter_dir(solver_model)
    cmd = build_training_command(
        run_root=run_root,
        report_dir=run_root / "reports",
        solver_model=solver_model,
        init_adapter_dir=init_adapter_dir,
    )
    shell(cmd, cwd=ROOT)
    return "trained"


def run_mining_bonus_pool(
    *,
    run_root: Path,
    run_name: str,
    candidate_budget: int,
    difficulty_state_path: Path,
    parent_dir: Path,
    solver_model: Path,
    solver_tokenizer: Path,
    length_guard_parent_baseline_tokens: int,
    min_length_delta_tokens: int,
    generation_concurrency: int,
    max_attempts_per_sample: int,
    max_api_failures: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    if candidate_budget < 2:
        raise ValueError("mining bonus candidate_budget must be at least 2")
    run_root.mkdir(parents=True, exist_ok=True)
    base_count = max(1, int(round(candidate_budget * 0.70)))
    bonus_count = max(0, candidate_budget - base_count)
    base_run_root = run_root / "base_balanced"
    bonus_run_root = run_root / "bonus_mining"

    original_state = read_json(difficulty_state_path) if difficulty_state_path.exists() else {}
    original_plan = copy.deepcopy(original_state.get("difficulty_plan") if isinstance(original_state.get("difficulty_plan"), dict) else {})
    try:
        base_plan = apply_balanced_plan(original_plan, BALANCED_MINING_STAGE)
        update_state_and_plan(difficulty_state_path, base_plan)
        ensure_run_root(base_run_root)
        run_round(
            run_root=base_run_root,
            run_name=f"{run_name}_base70_balanced",
            n_train=base_count,
            difficulty_state_path=difficulty_state_path,
            parent_dir=parent_dir,
            solver_model=solver_model,
            solver_tokenizer=solver_tokenizer,
            length_guard_parent_baseline_tokens=length_guard_parent_baseline_tokens,
            min_length_delta_tokens=min_length_delta_tokens,
            generation_concurrency=generation_concurrency,
            max_attempts_per_sample=max_attempts_per_sample,
            max_api_failures=max_api_failures,
            checkpoint_every=checkpoint_every,
            skip_closed_loop_controller_update=True,
            train_only=True,
            seed=20260610,
        )
        base_rows, base_generation_summary, base_report = read_generation_artifacts(base_run_root)
        base_summary = build_summary(
            stage=BALANCED_MINING_STAGE,
            phase=f"mining_base{base_count}",
            run_name=f"{run_name}_base70_balanced",
            run_root=base_run_root,
            n_train=base_count,
            rows=base_rows,
            generation_summary=base_generation_summary,
            report=base_report,
            state_path=difficulty_state_path,
        )
        write_stage_artifacts(base_run_root, base_summary, [build_history_entry(base_summary, "mining_bonus_base_complete", BALANCED_MINING_STAGE)], base_rows)

        bonus_rows: list[dict[str, Any]] = []
        bonus_summary: dict[str, Any] = {}
        bonus_meta: dict[str, Any] = {"mode": "mining_bonus", "base_count": base_count, "bonus_count": 0}
        if bonus_count > 0:
            recent_rows = latest_mining_bonus_recent_rows(run_root.parent, exclude=run_root)
            bonus_plan, bonus_meta = build_mining_bonus_plan(
                base_plan,
                base_rows,
                base_count=base_count,
                bonus_count=bonus_count,
                recent_rows=recent_rows or None,
            )
            update_state_and_plan(difficulty_state_path, bonus_plan)
            ensure_run_root(bonus_run_root)
            run_round(
                run_root=bonus_run_root,
                run_name=f"{run_name}_bonus30_mining",
                n_train=bonus_count,
                difficulty_state_path=difficulty_state_path,
                parent_dir=parent_dir,
                solver_model=solver_model,
                solver_tokenizer=solver_tokenizer,
                length_guard_parent_baseline_tokens=length_guard_parent_baseline_tokens,
                min_length_delta_tokens=min_length_delta_tokens,
                generation_concurrency=generation_concurrency,
                max_attempts_per_sample=max_attempts_per_sample,
                max_api_failures=max_api_failures,
                checkpoint_every=checkpoint_every,
                skip_closed_loop_controller_update=True,
                train_only=True,
                seed=20260610 + base_count + 97,
            )
            bonus_rows, bonus_generation_summary, bonus_report = read_generation_artifacts(bonus_run_root)
            bonus_summary = build_summary(
                stage=MINING_BONUS_STAGE,
                phase=f"mining_bonus{bonus_count}",
                run_name=f"{run_name}_bonus30_mining",
                run_root=bonus_run_root,
                n_train=bonus_count,
                rows=bonus_rows,
                generation_summary=bonus_generation_summary,
                report=bonus_report,
                state_path=difficulty_state_path,
            )
            write_json(bonus_run_root / "mining_bonus_policy.json", bonus_meta)
            write_stage_artifacts(bonus_run_root, bonus_summary, [build_history_entry(bonus_summary, "mining_bonus_complete", MINING_BONUS_STAGE)], bonus_rows)

        candidate_rows = normalize_rows(base_rows + bonus_rows)
        pool_rows = frontier_pool_rows(candidate_rows)
        write_jsonl(run_root / "candidate_pool.jsonl", candidate_rows)
        write_jsonl(run_root / "frontier_pool.jsonl", pool_rows)
        write_json(run_root / "mining_bonus_policy.json", bonus_meta)
        summary = {
            "run_name": run_name,
            "run_root": str(run_root.resolve()),
            "stage": MINING_BONUS_STAGE,
            "mode": "mining_bonus",
            "candidate_budget": candidate_budget,
            "base_count": base_count,
            "bonus_count": bonus_count,
            "generated_count": len(candidate_rows),
            "valid_count": maybe_token(base_summary.get("valid_count"), 0) + maybe_token(bonus_summary.get("valid_count"), 0),
            "clean_count": maybe_token(base_summary.get("clean_count"), 0) + maybe_token(bonus_summary.get("clean_count"), 0),
            "valid_rate": round(
                (maybe_token(base_summary.get("valid_count"), 0) + maybe_token(bonus_summary.get("valid_count"), 0)) / len(candidate_rows),
                6,
            ) if candidate_rows else 0.0,
            "clean_rate": round(
                (maybe_token(base_summary.get("clean_count"), 0) + maybe_token(bonus_summary.get("clean_count"), 0)) / len(candidate_rows),
                6,
            ) if candidate_rows else 0.0,
            "frontier_pool_count": len(pool_rows),
            "boundary_frontier_pool_count": sum(1 for row in pool_rows if row.get("frontier_pool_label") == "boundary_frontier"),
            # Deprecated alias; use boundary_frontier_pool_count.
            "exact_frontier_pool_count": sum(1 for row in pool_rows if row.get("frontier_pool_label") == "boundary_frontier"),
            "near_miss_pool_count": sum(1 for row in pool_rows if row.get("frontier_pool_label") == "near_miss"),
            "qtype_distribution": qtype_counts(candidate_rows),
            "frontier_pool_qtype_distribution": qtype_counts(pool_rows),
            "base_summary": base_summary,
            "bonus_summary": bonus_summary,
            "bonus_policy": bonus_meta,
            "action": "frontier_pool_built",
            "training_status": "not_started",
            "next_stage": MINING_BONUS_STAGE,
            "frontier_pool_logic": {
                "boundary_frontier": "preferred",
                "near_miss": "secondary",
                "five_of_five": "excluded except future small anchor mix",
                "train_mix_equals_all_generated_samples": False,
            },
        }
        write_json(run_root / "adaptive_smoke_summary.json", summary)
        write_json(run_root / "mining_bonus_summary.json", summary)
        write_jsonl(run_root / "adaptive_smoke_history.jsonl", [build_history_entry(base_summary, "mining_bonus_base_complete", BALANCED_MINING_STAGE)])
        return summary
    finally:
        if original_state:
            restored = copy.deepcopy(original_state)
            restored["difficulty_plan"] = apply_balanced_plan(original_plan, BALANCED_MINING_STAGE)
            write_json(difficulty_state_path, restored)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive smoke hardening loop for current_method r4 probe197.")
    parser.add_argument("--run-root-base", default=str(DEFAULT_RUN_ROOT_BASE))
    parser.add_argument("--difficulty-state-path", default=str(DEFAULT_DIFFICULTY_STATE))
    parser.add_argument("--difficulty-state-backup-path", default=str(DEFAULT_DIFFICULTY_STATE.with_name("difficulty_controller_state_r4_probe197.before_adaptive_smoke.json")))
    parser.add_argument("--bootstrap-run-root", default=str(DEFAULT_BOOTSTRAP_RUN))
    parser.add_argument("--no-bootstrap-prior-eval", action="store_true")
    parser.add_argument("--parent-dir", default=str(DEFAULT_PARENT_DIR))
    parser.add_argument("--solver-model", default=str(DEFAULT_SOLVER_MODEL))
    parser.add_argument("--solver-tokenizer", default=str(DEFAULT_TOK))
    parser.add_argument("--length-guard-parent-baseline-tokens", type=int, default=7753)
    parser.add_argument("--min-length-delta-tokens", type=int, default=4000)
    parser.add_argument("--smoke-sizes", default="20,30,40,50")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--generation-concurrency", type=int, default=4)
    parser.add_argument("--max-attempts-per-sample", type=int, default=4)
    parser.add_argument("--max-api-failures", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--run-inner-controller-update", action="store_true")
    parser.add_argument("--mode", choices=("adaptive", "mining_bonus"), default="adaptive")
    parser.add_argument("--candidate-budget", type=int, default=200)
    parser.add_argument("--auto-mining-bonus-after-smoke-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root_base = Path(args.run_root_base)
    difficulty_state_path = Path(args.difficulty_state_path)
    backup_path = Path(args.difficulty_state_backup_path)
    bootstrap_run_root = Path(args.bootstrap_run_root)
    parent_dir = Path(args.parent_dir)
    solver_model = Path(args.solver_model)
    solver_tokenizer = Path(args.solver_tokenizer)
    ensure_parent_dir(backup_path)
    if difficulty_state_path.exists() and not backup_path.exists():
        shutil.copy2(difficulty_state_path, backup_path)

    smoke_sizes = [int(item) for item in str(args.smoke_sizes).split(",") if item.strip()]
    if not smoke_sizes:
        raise ValueError("--smoke-sizes must include at least one positive integer")
    phase_plan = [(f"smoke{size}", size) for size in smoke_sizes] + [("gate100", 100), ("gate200", 200)]
    first_gate_index = len(smoke_sizes)
    gate200_index = len(smoke_sizes) + 1
    phase_index = 0
    difficulty_stage = "mastered_advance_initial"
    history: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    final_summary: dict[str, Any] = {}
    bootstrap_summary: dict[str, Any] | None = None

    if not args.no_bootstrap_prior_eval:
        prior_run_root = latest_completed_adaptive_eval_run(run_root_base) or bootstrap_run_root
        prior_stage = stage_hint_from_run(prior_run_root)
        difficulty_stage, bootstrap_summary = bootstrap_stage_from_prior_eval(prior_run_root, difficulty_state_path, prior_stage)
        if bootstrap_summary is not None:
            bootstrap_summary["action"] = f"bootstrap_to_{difficulty_stage}"
            bootstrap_summary["next_stage"] = difficulty_stage
            bootstrap_summary["prior_eval_run_root"] = str(prior_run_root.resolve())
            history.append(build_history_entry(bootstrap_summary, bootstrap_summary["action"], "mastered_advance_initial"))
    if difficulty_stage == "frontier_mining_2":
        difficulty_stage = BALANCED_MINING_STAGE
    if args.dry_run:
        if args.mode == "mining_bonus":
            base_run_name = f"current_method_r4_test_probe197_adaptive_mining{args.candidate_budget}_{MINING_BONUS_STAGE}"
            run_name, _run_root = allocate_run_root(run_root_base, base_run_name)
            base_count = max(1, int(round(int(args.candidate_budget) * 0.70)))
            bonus_count = max(0, int(args.candidate_budget) - base_count)
            print(run_name)
            print(f"{run_name}_base70_balanced n={base_count}")
            print(f"{run_name}_bonus30_mining n={bonus_count}")
            return 0
        for round_index in range(1, max(1, args.max_rounds) + 1):
            phase_name, _n_train = phase_plan[min(phase_index, len(phase_plan) - 1)]
            if difficulty_stage in {"frontier_mining_1", "frontier_mining_2"}:
                difficulty_stage = BALANCED_MINING_STAGE
            base_run_name = f"current_method_r4_test_probe197_adaptive_{phase_name}_{difficulty_stage}"
            run_name, _run_root = allocate_run_root(run_root_base, base_run_name)
            print(run_name)
            phase_index = (phase_index + 1) % max(1, first_gate_index)
        return 0

    run_root_base.mkdir(parents=True, exist_ok=True)
    if bootstrap_summary is not None:
        write_json(run_root_base / "adaptive_smoke_bootstrap_summary.json", bootstrap_summary)
    if difficulty_state_path.exists():
        plan = current_difficulty_plan(difficulty_state_path)
        plan = apply_plan_overrides(plan, difficulty_stage)
        update_state_and_plan(difficulty_state_path, plan)

    if args.mode == "mining_bonus":
        base_run_name = f"current_method_r4_test_probe197_adaptive_mining{args.candidate_budget}_{MINING_BONUS_STAGE}"
        run_name, run_root = allocate_run_root(run_root_base, base_run_name)
        final_summary = run_mining_bonus_pool(
            run_root=run_root,
            run_name=run_name,
            candidate_budget=int(args.candidate_budget),
            difficulty_state_path=difficulty_state_path,
            parent_dir=parent_dir,
            solver_model=solver_model,
            solver_tokenizer=solver_tokenizer,
            length_guard_parent_baseline_tokens=args.length_guard_parent_baseline_tokens,
            min_length_delta_tokens=args.min_length_delta_tokens,
            generation_concurrency=args.generation_concurrency,
            max_attempts_per_sample=args.max_attempts_per_sample,
            max_api_failures=args.max_api_failures,
            checkpoint_every=args.checkpoint_every,
        )
        write_json(run_root_base / "adaptive_smoke_summary.json", final_summary)
        print(json.dumps(final_summary, ensure_ascii=False, indent=2))
        return 0

    for round_index in range(1, max(1, args.max_rounds) + 1):
        phase_name, n_train = phase_plan[min(phase_index, len(phase_plan) - 1)]
        base_run_name = f"current_method_r4_test_probe197_adaptive_{phase_name}_{difficulty_stage}"
        run_name, run_root = allocate_run_root(run_root_base, base_run_name)
        ensure_run_root(run_root)
        if difficulty_state_path.exists():
            plan = current_difficulty_plan(difficulty_state_path)
            plan = apply_plan_overrides(plan, difficulty_stage)
            update_state_and_plan(difficulty_state_path, plan)

        run_round(
            run_root=run_root,
            run_name=run_name,
            n_train=n_train,
            difficulty_state_path=difficulty_state_path,
            parent_dir=parent_dir,
            solver_model=solver_model,
            solver_tokenizer=solver_tokenizer,
            length_guard_parent_baseline_tokens=args.length_guard_parent_baseline_tokens,
            min_length_delta_tokens=args.min_length_delta_tokens,
            generation_concurrency=args.generation_concurrency,
            max_attempts_per_sample=args.max_attempts_per_sample,
            max_api_failures=args.max_api_failures,
            checkpoint_every=args.checkpoint_every,
            skip_closed_loop_controller_update=not args.run_inner_controller_update,
            train_only=True,
        )
        rows, generation_summary, report = read_generation_artifacts(run_root)
        summary = build_summary(
            stage=difficulty_stage,
            phase=phase_name,
            run_name=run_name,
            run_root=run_root,
            n_train=n_train,
            rows=rows,
            generation_summary=generation_summary,
            report=report,
            state_path=difficulty_state_path,
        )
        action = "hold"
        next_stage = difficulty_stage
        if summary["rows_below_4000_delta"] > 0:
            action = "length_failed"
            next_stage = "length_failed"
        elif summary["valid_rate"] < 0.90 or summary["clean_rate"] < 0.90 or summary["leakage"] > 0:
            action = "quality_failed"
            next_stage = "quality_failed"
        elif gate_passed(summary):
            action = "gate_passed"
        elif difficulty_stage == "frontier_mining_1" and mining_smoke_passed(summary):
            action = "mining_smoke_passed"
        else:
            next_stage = choose_next_stage(summary, difficulty_stage)
            if next_stage != difficulty_stage:
                action = f"advance_to_{next_stage}"

        history.append(build_history_entry(summary, action, difficulty_stage))
        summary["action"] = action
        summary["next_stage"] = next_stage
        summary["phase"] = phase_name
        summary["round_index"] = round_index
        final_summary = summary
        last_rows = rows
        write_stage_artifacts(run_root, summary, history, rows)

        plan = current_difficulty_plan(difficulty_state_path)
        if next_stage == "quality_failed":
            plan = apply_plan_overrides(plan, "quality_failed")
            update_state_and_plan(difficulty_state_path, plan)
            break
        if next_stage == "length_failed":
            break
        if next_stage != difficulty_stage:
            difficulty_stage = next_stage
            plan = apply_plan_overrides(plan, difficulty_stage)
            update_state_and_plan(difficulty_state_path, plan)
            summary["stage_transition_applied"] = True
            final_summary = summary
            write_stage_artifacts(run_root, summary, history, rows)
            break
        if phase_name.startswith("smoke"):
            if (
                args.auto_mining_bonus_after_smoke_success
                and difficulty_stage == BALANCED_MINING_STAGE
                and phase_name == "smoke50"
                and mining_smoke_passed(summary)
            ):
                mining_run_name = f"current_method_r4_test_probe197_adaptive_mining{args.candidate_budget}_{MINING_BONUS_STAGE}"
                mining_run_name, mining_run_root = allocate_run_root(run_root_base, mining_run_name)
                mining_summary = run_mining_bonus_pool(
                    run_root=mining_run_root,
                    run_name=mining_run_name,
                    candidate_budget=int(args.candidate_budget),
                    difficulty_state_path=difficulty_state_path,
                    parent_dir=parent_dir,
                    solver_model=solver_model,
                    solver_tokenizer=solver_tokenizer,
                    length_guard_parent_baseline_tokens=args.length_guard_parent_baseline_tokens,
                    min_length_delta_tokens=args.min_length_delta_tokens,
                    generation_concurrency=args.generation_concurrency,
                    max_attempts_per_sample=args.max_attempts_per_sample,
                    max_api_failures=args.max_api_failures,
                    checkpoint_every=args.checkpoint_every,
                )
                summary["action"] = "mining_success_entered_mining_bonus"
                summary["mining_bonus_run_root"] = str(mining_run_root.resolve())
                summary["mining_bonus_summary"] = mining_summary
                final_summary = summary
                write_stage_artifacts(run_root, summary, history, rows)
                break
            if action == "mining_smoke_passed" or (
                summary["valid_rate"] >= 0.95
                and summary["clean_rate"] >= 0.95
                and summary["rows_below_4000_delta"] == 0
                and summary["leakage"] == 0
                and gate_passed(summary)
            ):
                phase_index = min(first_gate_index, len(phase_plan) - 1)
            else:
                phase_index = (phase_index + 1) % max(1, first_gate_index)
            continue
        if phase_name == "gate100":
            if gate_passed(summary):
                phase_index = min(gate200_index, len(phase_plan) - 1)
            else:
                phase_index = 0
            continue
        if phase_name == "gate200":
            if gate_passed(summary):
                train_status = run_training_if_ready(run_root, summary, rows, solver_model, solver_tokenizer)
                summary["training_status"] = train_status
                final_summary = summary
                write_stage_artifacts(run_root, summary, history, rows)
                break
            phase_index = 0
            continue
        phase_index = min(phase_index + 1, len(phase_plan) - 1)

    write_json(run_root_base / "adaptive_smoke_summary.json", final_summary)
    write_jsonl(run_root_base / "adaptive_smoke_history.jsonl", history)
    if last_rows:
        write_jsonl(run_root_base / "adaptive_smoke_last_rows.jsonl", last_rows)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
