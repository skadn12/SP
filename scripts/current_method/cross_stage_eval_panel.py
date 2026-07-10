#!/usr/bin/env python3
"""Cross-stage evaluation panel for current-method self-play.

The panel is meant to answer one question consistently:

    Does solver stage N improve on the heldout frontier built for stage N?

By default this script only collects existing eval JSON files and writes a
Markdown report. Pass --run-original and/or --run-chat when you really want to
rerun vLLM evaluations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PANEL_DIR = ROOT / "base_restart_current_method_validation/70_cross_stage_eval_panel"
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")

MODELS: dict[str, Path] = {
    "base": DEFAULT_BASE_MODEL,
    "stage1": ROOT / "base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged",
    "stage2": ROOT / "base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged",
    "stage64": ROOT / "outputs/qwen3_4b_lora_stage64_r3_325f_shuffled_from_r2_ctx32000_merged",
}

ORIGINAL_DATASETS: dict[str, Path] = {
    "stage1_frontier_heldout": ROOT / "base_restart_current_method_validation/01_mining500_train/train_mix/frontier_heldout.jsonl",
    "stage2_frontier_heldout": ROOT / "base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_heldout.jsonl",
    "stage64_frontier_heldout": ROOT
    / "base_restart_current_method_validation/69_stage64_api_selfplay500_ctx32768/"
    / "option_shuffled_seed20260710/splits/frontier_group_holdout_seed20260710/frontier_heldout.jsonl",
}

CHAT_INPUTS: tuple[Path, ...] = (
    ROOT
    / "base_restart_current_method_validation/69_stage64_api_selfplay500_ctx32768/"
    / "option_shuffled_seed20260710/splits/frontier_group_holdout_seed20260710/frontier_heldout_sft.jsonl",
    ROOT / "data/locomo_processed_datasets/_global_clean_v1_solver_r1_training_dataset/actual_sft/clean_v1_solver_r1_valid_chat.jsonl",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "-"


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_original(args: argparse.Namespace) -> None:
    script = ROOT / "scripts/current_method/diagnostics_context_ablation.py"
    for model_name, model_path in MODELS.items():
        for dataset_name, samples_path in ORIGINAL_DATASETS.items():
            out_dir = args.panel_dir / "original_format" / model_name / dataset_name
            run_cmd(
                [
                    sys.executable,
                    str(script),
                    "--task",
                    "verify",
                    "--samples",
                    str(samples_path),
                    "--out-dir",
                    str(out_dir),
                    "--base-model",
                    str(DEFAULT_BASE_MODEL),
                    "--solver-model",
                    str(model_path),
                    "--solver-tokenizer",
                    str(DEFAULT_BASE_MODEL),
                    "--k-eval",
                    str(args.k_eval),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                ],
                cwd=ROOT,
            )


def run_chat(args: argparse.Namespace) -> None:
    script = ROOT / "scripts/eval_v13_pilot_vllm.py"
    for model_name, model_path in MODELS.items():
        out_dir = args.panel_dir / "chat_format" / model_name
        cmd = [
            sys.executable,
            str(script),
            "--output-dir",
            str(out_dir),
            "--model",
            str(model_path),
            "--adapter-dir",
            str(model_path),
            "--max-length",
            str(args.max_length),
            "--max-model-len",
            str(args.max_model_len),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--sampling-k",
            "1",
            "--chunk-size",
            str(args.chunk_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--input",
        ]
        cmd.extend(str(path) for path in CHAT_INPUTS)
        run_cmd(cmd, cwd=ROOT)


def collect_original(panel_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        out[model_name] = {}
        for dataset_name in ORIGINAL_DATASETS:
            p = panel_dir / "original_format" / model_name / dataset_name / "reports/verify_summary.json"
            if p.exists():
                out[model_name][dataset_name] = read_json(p)
    write_json(panel_dir / "original_format/original_panel_summary.json", out)
    return out


def collect_chat(panel_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_name in MODELS:
        p = panel_dir / "chat_format" / model_name / "overall_summary.json"
        if p.exists():
            out[model_name] = read_json(p)
    write_json(panel_dir / "chat_format/chat_panel_summary.json", out)
    return out


def _metric(original: dict[str, dict[str, Any]], model: str, dataset: str, key: str) -> Any:
    return original.get(model, {}).get(dataset, {}).get(key)


def _chat_metric(chat: dict[str, Any], model: str, eval_set: str) -> Any:
    by_set = chat.get(model, {}).get("summary", {}).get("by_eval_set", {})
    return by_set.get(eval_set, {}).get("accuracy")


def markdown_report(original: dict[str, dict[str, Any]], chat: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# 跨阶段能力增长评估面板",
            "",
            "这个面板用于验证 self-play 的阶段性收益：",
            "",
            "- Stage1 solver 是否在 Stage1 heldout frontier 上超过 base；",
            "- Stage2 solver 是否在 Stage2 heldout frontier 上超过 Stage1；",
            "- Stage64 solver 是否在 Stage64 heldout frontier 上超过 Stage2。",
            "",
            "## 结论",
            "",
            "1. Stage1 的阶段收益成立：Stage1 在 Stage1 heldout 上是 77.27%，base 是 45.45%。",
            "2. Stage2 的阶段收益成立，但幅度较小：Stage2 在 Stage2 heldout 上是 56.67%，Stage1 是 53.33%。",
            "3. Stage64 当前 heldout 不能作为能力增长证据：shuffle 后没有重新按 solver 评估重选 frontier，base/Stage1/Stage2 在该 heldout 上都达到 97.59%，说明它已经不是 Stage2-calibrated frontier。",
            "4. Stage64 当前模型对旧阶段 heldout 有明显遗忘：在 Stage1 heldout 上降到 36.36%，在 Stage2 heldout 上降到 16.67%。这和外部合并评估下降是一致的。",
            "",
            "## Original-format Verifier: K1 Accuracy",
            "",
            "| solver | Stage1 heldout | Stage2 heldout | Stage64 heldout |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in MODELS:
        lines.append(
            "| {m} | {s1} | {s2} | {s64} |".format(
                m=model,
                s1=pct(_metric(original, model, "stage1_frontier_heldout", "solver_exact_accuracy_k1")),
                s2=pct(_metric(original, model, "stage2_frontier_heldout", "solver_exact_accuracy_k1")),
                s64=pct(_metric(original, model, "stage64_frontier_heldout", "solver_exact_accuracy_k1")),
            )
        )
    lines.extend(
        [
            "",
            "## Original-format Verifier: K5 Soft Accuracy",
            "",
            "| solver | Stage1 heldout | Stage2 heldout | Stage64 heldout |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in MODELS:
        lines.append(
            "| {m} | {s1} | {s2} | {s64} |".format(
                m=model,
                s1=pct(_metric(original, model, "stage1_frontier_heldout", "solver_soft_accuracy_k5")),
                s2=pct(_metric(original, model, "stage2_frontier_heldout", "solver_soft_accuracy_k5")),
                s64=pct(_metric(original, model, "stage64_frontier_heldout", "solver_soft_accuracy_k5")),
            )
        )
    if chat:
        lines.extend(
            [
                "",
                "## Chat/SFT-format Eval",
                "",
                "这个评估使用 SFT chat 文件。它能检查训练格式和 clean retention，但不能替代 original-format frontier 校准。",
                "",
                "| solver | Stage64 SFT heldout | clean retention |",
                "| --- | ---: | ---: |",
            ]
        )
        for model in MODELS:
            lines.append(
                "| {m} | {held} | {clean} |".format(
                    m=model,
                    held=pct(_chat_metric(chat, model, "frontier_heldout_sft")),
                    clean=pct(_chat_metric(chat, model, "clean_v1_solver_r1_valid_chat")),
                )
            )
    lines.extend(
        [
            "",
            "## 下一步修复",
            "",
            "Stage64 的正确闭环应该是：",
            "",
            "1. 对 Stage64 样本做 option shuffle；",
            "2. 清空/忽略旧的 `solver_eval`；",
            "3. 用固定 Stage2 solver 重新评估 shuffle 后样本；",
            "4. 按重新评估结果重新筛 frontier / hard0 / easy5；",
            "5. 再从重新筛出的 frontier 切 train/heldout；",
            "6. 训练 Stage64 solver；",
            "7. 用这个新 heldout 证明 Stage64 solver > Stage2 solver，同时用 Stage1/Stage2 heldout 和外部集做遗忘检查。",
            "",
            "因此，目前可以在论文里使用 Stage1 和 Stage2 的递进结果；Stage64 需要先完成 shuffle 后的 frontier 重新校准，才能作为严格的第三阶段能力增长证据。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or summarize a cross-stage self-play eval panel.")
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--run-original", action="store_true", help="Rerun original-format verifier evals.")
    parser.add_argument("--run-chat", action="store_true", help="Rerun chat/SFT vLLM evals.")
    parser.add_argument("--k-eval", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=32000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.panel_dir.mkdir(parents=True, exist_ok=True)
    if args.run_original:
        run_original(args)
    if args.run_chat:
        run_chat(args)
    original = collect_original(args.panel_dir)
    chat = collect_chat(args.panel_dir)
    report = markdown_report(original, chat)
    report_path = args.panel_dir / "cross_stage_eval_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
