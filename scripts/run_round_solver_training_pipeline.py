#!/usr/bin/env python3
"""Round solver training pipeline wrapper.

This is an orchestration layer for the second half of self-play:

  frontier + clean replay -> SFT train file -> LoRA train -> merge -> eval

It intentionally calls the existing data builder, trainer, merger, and evaluator
instead of reimplementing their inner logic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_R2_ADAPTER = ROOT / "outputs/qwen3_4b_lora_r2_frontier_plus_clean_replay_s0_s2_from_r1_e1_ctx9216_full"
DEFAULT_CLEAN_RETENTION = ROOT / "sft_data/clean_v1_solver_r1_valid_chat.jsonl"
LEGACY_TRAIN_FILE_NAME = "r2_frontier_plus_clean_replay_sft_train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train data, train, merge, and evaluate a round solver.")
    parser.add_argument("--round-id", required=True, help="Solver round being trained, e.g. r3.")
    parser.add_argument("--round-number", type=int, default=None)
    parser.add_argument("--fixed-solver-name", required=True, help="Solver used to select frontier, e.g. solver_r2_ctx9216_full.")
    parser.add_argument("--solver-name", default=None, help="Name for the newly trained solver in reports.")
    parser.add_argument("--round-data-dir", default=None, help="Generated round dataset dir containing sft_train/dev/heldout JSONL.")
    parser.add_argument("--frontier-report-dir", default=None, help="Report dir containing <solver-prefix>_frontier_candidates.jsonl.")
    parser.add_argument("--frontier", action="append", default=None, help="Frontier JSONL. May be passed multiple times.")
    parser.add_argument("--round-sft-train", action="append", default=None, help="Matching round sft_train JSONL. May be passed multiple times.")
    parser.add_argument("--train-output-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--init-adapter-dir", default=str(DEFAULT_R2_ADAPTER))
    parser.add_argument("--adapter-output-dir", default=None)
    parser.add_argument("--merged-output-dir", default=None)
    parser.add_argument("--target-replay", type=int, default=80)
    parser.add_argument("--min-replay", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--max-length", type=int, default=9216)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=9999)
    parser.add_argument("--max-valid-samples", type=int, default=64)
    parser.add_argument("--valid-file", default=None, help="Defaults to the built train file, matching the r2 retrain pattern.")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--smoke-eval-limit", type=int, default=16)
    parser.add_argument("--eval-input", action="append", default=None, help="Extra full-eval input JSONL. May be repeated.")
    parser.add_argument("--include-clean-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-full-eval", action="store_true")
    parser.add_argument("--skip-build-train", action="store_true")
    parser.add_argument("--skip-token-check", action="store_true")
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-smoke-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output dirs owned by this pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print and record commands without running them.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "unknown"


def solver_prefix(fixed_solver_name: str) -> str:
    if fixed_solver_name.startswith("solver_"):
        parts = fixed_solver_name.split("_")
        if len(parts) >= 2:
            return "solver_" + safe_name(parts[1])
    return safe_name(fixed_solver_name)


def short_solver_id(name: str) -> str:
    match = re.search(r"\b(r\d+)\b", name)
    return match.group(1) if match else safe_name(name)


def default_paths(args: argparse.Namespace) -> argparse.Namespace:
    ctx = f"ctx{args.max_length}"
    fixed_short = short_solver_id(args.fixed_solver_name)
    if args.solver_name is None:
        args.solver_name = f"solver_{args.round_id}_{ctx}_full"
    if args.train_output_dir is None:
        args.train_output_dir = str(ROOT / f"eval_sets/graph_expansion_challenger_v1/{args.round_id}_frontier_plus_clean_replay_train")
    if args.report_dir is None:
        args.report_dir = str(ROOT / f"eval_reports/graph_expansion_challenger_v1/{args.solver_name}_train")
    if args.adapter_output_dir is None:
        args.adapter_output_dir = str(ROOT / f"outputs/qwen3_4b_lora_{args.round_id}_frontier_plus_clean_replay_from_{fixed_short}_{ctx}_full")
    if args.merged_output_dir is None:
        args.merged_output_dir = f"{args.adapter_output_dir}_merged"
    return args


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def ensure_exists(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Missing {label}: {target}")
    return target


def ensure_output_available(path: str | Path, *, overwrite: bool, dry_run: bool, label: str) -> None:
    target = Path(path)
    if not target.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{label} already exists; pass --overwrite to replace: {target}")
    if dry_run:
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def run_command(cmd: list[str], *, log_path: Path, dry_run: bool) -> None:
    printable = shlex.join(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        if dry_run:
            return
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def infer_frontiers(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    frontier_values = list(args.frontier or [])
    sft_values = list(args.round_sft_train or [])
    if not frontier_values:
        if not args.frontier_report_dir:
            raise ValueError("Pass --frontier or --frontier-report-dir.")
        frontier_values.append(str(Path(args.frontier_report_dir) / f"{solver_prefix(args.fixed_solver_name)}_frontier_candidates.jsonl"))
    if not sft_values:
        if not args.round_data_dir:
            raise ValueError("Pass --round-sft-train or --round-data-dir.")
        sft_values.append(str(Path(args.round_data_dir) / "sft_train.jsonl"))
    if len(frontier_values) != len(sft_values):
        raise ValueError("--frontier and --round-sft-train must have the same count")
    frontiers = [ensure_exists(path, "frontier JSONL") for path in frontier_values]
    sft_files = [ensure_exists(path, "round sft_train JSONL") for path in sft_values]
    return frontiers, sft_files


def built_train_file(train_output_dir: Path) -> Path:
    return train_output_dir / LEGACY_TRAIN_FILE_NAME


def build_train_data(args: argparse.Namespace, frontiers: list[Path], sft_files: list[Path], report_dir: Path) -> Path:
    output_dir = Path(args.train_output_dir)
    if args.skip_build_train:
        train_file = built_train_file(output_dir)
        ensure_exists(train_file, "built train file")
        return train_file
    ensure_output_available(output_dir, overwrite=args.overwrite, dry_run=args.dry_run, label="train output dir")
    cmd = [
        sys.executable,
        str(SCRIPTS / "build_r2_frontier_plus_clean_replay_train.py"),
        "--output-dir",
        str(output_dir),
        "--target-replay",
        str(args.target_replay),
        "--min-replay",
        str(args.min_replay),
        "--seed",
        str(args.seed),
    ]
    for frontier in frontiers:
        cmd.extend(["--frontier", str(frontier)])
    for sft_file in sft_files:
        cmd.extend(["--r2-sft-train", str(sft_file)])
    run_command(cmd, log_path=report_dir / "build_train.log", dry_run=args.dry_run)
    return built_train_file(output_dir)


def token_length_check(args: argparse.Namespace, train_file: Path, valid_file: Path | None, report_dir: Path) -> dict[str, Any]:
    if args.skip_token_check:
        summary = {"skipped": True}
        write_json(report_dir / "token_length_summary.json", summary)
        return summary
    if args.dry_run:
        summary = {"skipped": True, "reason": "dry_run"}
        write_json(report_dir / "token_length_summary.json", summary)
        return summary

    from transformers import AutoTokenizer  # Imported lazily to keep dry-runs light.
    from train_qwen3_lora_sft import render_chat

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def lengths(path: Path) -> list[int]:
        values: list[int] = []
        for record in read_jsonl(path):
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"Record without messages in {path}")
            text = render_chat(tokenizer, messages, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
                ids.append(tokenizer.eos_token_id)
            values.append(len(ids))
        return values

    files = {"train": train_file}
    if valid_file is not None and valid_file != train_file:
        files["valid"] = valid_file
    by_file = {}
    all_lengths: list[int] = []
    for label, path in files.items():
        values = lengths(path)
        all_lengths.extend(values)
        by_file[label] = {
            "path": str(path),
            "n": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": round(sum(values) / len(values), 3) if values else None,
            "over_max_length": sum(1 for value in values if value > args.max_length),
        }
    summary = {
        "max_length": args.max_length,
        "n": len(all_lengths),
        "max_tokens": max(all_lengths) if all_lengths else None,
        "over_max_length": sum(1 for value in all_lengths if value > args.max_length),
        "by_file": by_file,
    }
    write_json(report_dir / "token_length_summary.json", summary)
    if summary["over_max_length"] and not args.allow_truncation:
        raise RuntimeError(f"Token length check failed: {summary}")
    return summary


def train_solver(args: argparse.Namespace, train_file: Path, valid_file: Path | None, report_dir: Path) -> None:
    if args.skip_train:
        return
    ensure_output_available(args.adapter_output_dir, overwrite=args.overwrite, dry_run=args.dry_run, label="adapter output dir")
    cmd = [
        sys.executable,
        str(ROOT / "train_qwen3_lora_sft.py"),
        "--model",
        str(args.base_model),
        "--train-file",
        str(train_file),
        "--output-dir",
        str(args.adapter_output_dir),
        "--init-adapter-dir",
        str(args.init_adapter_dir),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--min-lr-ratio",
        str(args.min_lr_ratio),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--max-valid-samples",
        str(args.max_valid_samples),
        "--seed",
        str(args.seed),
    ]
    if valid_file is not None:
        cmd.extend(["--valid-file", str(valid_file)])
    run_command(cmd, log_path=report_dir / "train.log", dry_run=args.dry_run)


def merge_solver(args: argparse.Namespace, report_dir: Path) -> None:
    if args.skip_merge:
        return
    if not args.dry_run:
        ensure_exists(Path(args.adapter_output_dir) / "adapter_model.pt", "adapter_model.pt")
        ensure_exists(Path(args.adapter_output_dir) / "adapter_config.json", "adapter_config.json")
    ensure_output_available(args.merged_output_dir, overwrite=args.overwrite, dry_run=args.dry_run, label="merged output dir")
    cmd = [
        sys.executable,
        str(SCRIPTS / "merge_custom_lora_adapter.py"),
        "--model",
        str(args.base_model),
        "--adapter-dir",
        str(args.adapter_output_dir),
        "--output-dir",
        str(args.merged_output_dir),
        "--overwrite",
    ]
    run_command(cmd, log_path=report_dir / "merge.log", dry_run=args.dry_run)


def eval_inputs(args: argparse.Namespace, train_file: Path) -> list[Path]:
    values = [train_file]
    if args.round_data_dir:
        for name in ("sft_eval_dev.jsonl", "sft_eval_heldout.jsonl"):
            path = Path(args.round_data_dir) / name
            if path.exists():
                values.append(path)
    for item in args.eval_input or []:
        values.append(Path(item))
    if args.include_clean_retention and DEFAULT_CLEAN_RETENTION.exists():
        values.append(DEFAULT_CLEAN_RETENTION)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in values:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def run_eval(args: argparse.Namespace, inputs: list[Path], output_dir: Path, *, limit: int | None, sampling_k: int, log_name: str) -> None:
    for path in inputs:
        ensure_exists(path, "eval input")
    cmd = [
        sys.executable,
        str(SCRIPTS / "eval_v13_pilot_vllm.py"),
        "--input",
        *[str(path) for path in inputs],
        "--output-dir",
        str(output_dir),
        "--model",
        str(args.merged_output_dir),
        "--adapter-dir",
        str(args.adapter_output_dir),
        "--max-length",
        str(args.max_length),
        "--max-model-len",
        str(args.max_model_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--sampling-k",
        str(sampling_k),
        "--chunk-size",
        str(args.chunk_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    run_command(cmd, log_path=output_dir.parent / log_name, dry_run=args.dry_run)


def read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def write_status(
    args: argparse.Namespace,
    *,
    report_dir: Path,
    frontiers: list[Path],
    sft_files: list[Path],
    train_file: Path,
    valid_file: Path | None,
    token_summary: dict[str, Any],
) -> None:
    train_summary = read_optional_json(Path(args.train_output_dir) / "train_summary.json")
    adapter_config = read_optional_json(Path(args.adapter_output_dir) / "adapter_config.json")
    merge_info = read_optional_json(Path(args.merged_output_dir) / "merged_adapter_info.json")
    smoke = read_optional_json(report_dir / "smoke_eval" / "overall_summary.json")
    full = read_optional_json(report_dir / "full_eval" / "overall_summary.json")
    commands_path = report_dir / "pipeline_commands.json"
    lines = [
        f"# {args.solver_name} Training Pipeline Status",
        "",
        f"- round_id: `{args.round_id}`",
        f"- round_number: `{args.round_number}`",
        f"- fixed solver used for frontier: `{args.fixed_solver_name}`",
        f"- new solver name: `{args.solver_name}`",
        f"- dry_run: {args.dry_run}",
        f"- base model: `{args.base_model}`",
        f"- init adapter: `{args.init_adapter_dir}`",
        f"- train builder: `scripts/build_r2_frontier_plus_clean_replay_train.py`",
        f"- train file: `{train_file}`",
        f"- valid file: `{valid_file}`",
        f"- output adapter: `{args.adapter_output_dir}`",
        f"- merged model: `{args.merged_output_dir}`",
        f"- frontier files: `{json.dumps([str(path) for path in frontiers], ensure_ascii=False)}`",
        f"- round sft files: `{json.dumps([str(path) for path in sft_files], ensure_ascii=False)}`",
        f"- total train samples: {train_summary.get('total_sample_count', 'not built')}",
        f"- frontier samples: {train_summary.get('r2_frontier_count', 'not built')}",
        f"- clean replay samples: {train_summary.get('clean_v1_replay_count', 'not built')}",
        f"- contamination heldout bad: {(train_summary.get('contamination_checks') or {}).get('heldout_contamination_bad', 'not built')}",
        f"- contamination forbidden bad: {(train_summary.get('contamination_checks') or {}).get('forbidden_source_contamination_bad', 'not built')}",
        f"- max_length: {args.max_length}",
        f"- token length max: {token_summary.get('max_tokens', 'not checked')}",
        f"- token length over max: {token_summary.get('over_max_length', 'not checked')}",
        f"- train steps: {adapter_config.get('step', 'not trained')}",
        f"- final valid loss: {(adapter_config.get('metrics') or {}).get('final_valid_loss', 'not trained')}",
        f"- merged linear modules: {((merge_info.get('merge_info') or {}).get('merged_linear_modules', 'not merged'))}",
        f"- merge skipped modules: {len((merge_info.get('merge_info') or {}).get('skipped_missing_pairs', [])) if merge_info else 'not merged'}",
        f"- smoke eval accuracy: {(smoke.get('summary') or {}).get('accuracy', 'not run')}",
        f"- full eval accuracy: {(full.get('summary') or {}).get('accuracy', 'not run')}",
        f"- commands manifest: `{commands_path}`",
        "",
        "## Notes",
        "",
        "- This wrapper does not alter the existing data builder, trainer, merger, or evaluator internals.",
        "- The replay builder is the legacy r2-named script; this status file records the actual round provenance.",
    ]
    (report_dir / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_commands(report_dir: Path) -> None:
    commands = {}
    for name in ("build_train.log", "train.log", "merge.log", "smoke_eval.log", "full_eval.log"):
        path = report_dir / name
        if not path.exists():
            continue
        commands[name] = [line[2:].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("$ ")]
    write_json(report_dir / "pipeline_commands.json", commands)


def main() -> int:
    args = default_paths(parse_args())
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frontiers, sft_files = infer_frontiers(args)

    ensure_exists(args.base_model, "base model")
    ensure_exists(args.init_adapter_dir, "init adapter")
    train_file = build_train_data(args, frontiers, sft_files, report_dir)
    if not args.dry_run:
        ensure_exists(train_file, "train file")
    valid_file = Path(args.valid_file) if args.valid_file else train_file
    if valid_file and not args.dry_run:
        ensure_exists(valid_file, "valid file")

    token_summary = token_length_check(args, train_file, valid_file, report_dir)
    train_solver(args, train_file, valid_file, report_dir)
    merge_solver(args, report_dir)

    if not args.skip_smoke_eval:
        run_eval(
            args,
            [train_file],
            report_dir / "smoke_eval",
            limit=args.smoke_eval_limit,
            sampling_k=1,
            log_name="smoke_eval.log",
        )
    if args.run_full_eval:
        run_eval(args, eval_inputs(args, train_file), report_dir / "full_eval", limit=None, sampling_k=1, log_name="full_eval.log")

    collect_commands(report_dir)
    write_status(
        args,
        report_dir=report_dir,
        frontiers=frontiers,
        sft_files=sft_files,
        train_file=train_file,
        valid_file=valid_file,
        token_summary=token_summary,
    )
    print((report_dir / "status.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
